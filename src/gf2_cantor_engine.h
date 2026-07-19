// gf2_cantor_engine.h
// Host-side engine for GF(2)[x] multiplication by additive FFT over
// GF(2^64) on a Cantor basis (Cantor 1989; Gao-Mateer 2010 recursion,
// specialized to a Cantor basis so no scaling pass is needed).
//
// Method summary
// --------------
// The inputs are split into 32-bit chunks, each chunk lifted to an
// element of GF(2^64).  A chunk-product has degree <= 62 < 64, so field
// multiplication never wraps: the field product of two chunks IS their
// carry-less product, and sums of chunk products are XORs.  Hence the
// GF(2)[x] product equals the chunk-polynomial product over GF(2^64),
// recovered by 32-bit-stride overlap-add.
//
// The chunk polynomials are multiplied by evaluation/interpolation on
// the 2^m points  V_m = span{b1..bm}  of a Cantor basis:
//     b1 = 1,   b_{i+1}^2 + b_{i+1} = b_i
// (such a chain of length 64 exists because GF(2^64) = GF(2^(2^6)) is a
// Cantor field).  With T(y) = y^2 + y we have T(b_{i+1}) = b_i and
// T(p+1) = T(p), so writing  f(x) = g(T(x)) + x h(T(x))  (Taylor
// expansion at y^2+y) reduces evaluation on V_m to evaluating g, h on
// V_{m-1} -- with NO twiddle scaling, the hallmark of the Cantor basis:
//     f(p)   = g(T(p)) + p * h(T(p))
//     f(p+1) = f(p) + h(T(p))
// The even point of pair t at any depth is  W[t] = XOR_{bits b of t}
// b_{b+2}, independent of depth and subproblem, so one table of n/2
// twiddles serves the whole transform.  Outputs land in natural order;
// there is no bit-reversal anywhere.
//
// In-place layout: at depth L the 2^L subproblems are the residue
// classes (index mod 2^L); logical coefficient index = index >> L.  The
// in-place Taylor step for a block of logical size B (halves A|B, with
// B split C|D) is two range-XORs:  C ^= D  then  A_hi ^= C;  fused over
// all subproblems of a depth these become CONTIGUOUS range-XORs, which
// vectorize on CPU and coalesce on GPU.
//
// Everything here is plain C++ (single-threaded) and is exercised by
// cantor_ref.cpp against independent oracles.  cantor_cuda.cu includes
// this same header and checks the GPU kernels against this engine.

#ifndef GF2_CANTOR_ENGINE_H
#define GF2_CANTOR_ENGINE_H

#include "gf2_cantor_core.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// bit-serial reference field multiply (oracle for gf64_mul)
static inline u64 gf64_mul_bitref(u64 a, u64 b) {
    u64 r = 0;
    while (b) {
        if (b & 1) r ^= a;
        u64 hi = a >> 63;
        a = (a << 1) ^ (hi ? 0x1BULL : 0);   // x^64 ≡ x^4+x^3+x+1
        b >>= 1;
    }
    return r;
}

static inline u64 gf64_pow_u(u64 a, u64 e) {
    u64 r = 1;
    while (e) { if (e & 1) r = gf64_mul(r, a); a = gf64_sqr(a); e >>= 1; }
    return r;
}

// ---------------------------------------------------------------------------
// Cantor basis construction.
// Solves T(y) = y^2 + y = c by Gaussian elimination over GF(2) on the
// 64x64 bit-matrix of T (half-trace formulas do not apply in fields of
// even extension degree).  Returns the chain b[0..len).
struct CantorBasis {
    u64 b[64];
    int len = 0;

    void build() {
        // pivots for the linear map T
        u64 pv[64] = {0}, pt[64] = {0};
        for (int i = 0; i < 64; i++) {
            u64 v = gf64_sqr((u64)1 << i) ^ ((u64)1 << i);
            u64 t = (u64)1 << i;
            for (int bit = 63; bit >= 0; bit--) {
                if (!((v >> bit) & 1)) continue;
                if (pv[bit]) { v ^= pv[bit]; t ^= pt[bit]; }
                else { pv[bit] = v; pt[bit] = t; break; }
            }
        }
        b[0] = 1; len = 1;
        while (len < 64) {
            u64 c = b[len - 1], v = c, y = 0;
            bool ok = true;
            for (int bit = 63; bit >= 0; bit--) {
                if (!((v >> bit) & 1)) continue;
                if (!pv[bit]) { ok = false; break; }
                v ^= pv[bit]; y ^= pt[bit];
            }
            if (!ok || v != 0) break;            // Tr(c) = 1: chain ends
            if ((gf64_sqr(y) ^ y) != c) { fprintf(stderr, "AS solve bug\n"); exit(1); }
            b[len++] = y;
        }
    }

    bool verify() const {
        if (b[0] != 1) return false;
        for (int i = 1; i < len; i++)
            if ((gf64_sqr(b[i]) ^ b[i]) != b[i - 1]) return false;
        // linear independence
        u64 pv[64] = {0}; int rank = 0;
        for (int i = 0; i < len; i++) {
            u64 v = b[i];
            for (int bit = 63; bit >= 0 && v; bit--) {
                if (!((v >> bit) & 1)) continue;
                if (pv[bit]) v ^= pv[bit];
                else { pv[bit] = v; v = 0; rank++; }
            }
        }
        return rank == len;
    }
};

// ---------------------------------------------------------------------------
// The transform.  Buffers are arrays of n = 2^m field elements.
struct CantorFFT {
    int m = 0;
    size_t n = 0;
    std::vector<u64> W;          // W[t], t < n/2: even point of pair t

    void init(int m_, const CantorBasis &B) {
        m = m_;
        n = (size_t)1 << m;
        if (m > B.len) { fprintf(stderr, "FFT size 2^%d exceeds basis length %d\n", m, B.len); exit(1); }
        W.assign(n >> 1, 0);
        for (size_t t = 1; t < (n >> 1); t++) {
            int hb = 63 - __builtin_clzll((u64)t);
            W[t] = W[t ^ ((size_t)1 << hb)] ^ B.b[hb + 1];
        }
    }

    static void xr(u64 *f, size_t dst, size_t src, size_t len) {
        for (size_t k = 0; k < len; k++) f[dst + k] ^= f[src + k];
    }

    // forward: coefficients (natural order) -> values f(pt(i))
    void fwd(u64 *f) const {
        for (int L = 0; L <= m - 2; L++) {               // Taylor cascade
            size_t S = n >> L;
            for (size_t B = S; B >= 4; B >>= 1) {
                size_t q = B >> 2, h = B >> 1;
                for (size_t o = 0; o < S; o += B) {
                    xr(f, (o + h) << L, (o + h + q) << L, q << L);  // C ^= D
                    xr(f, (o + q) << L, (o + h) << L, q << L);      // A_hi ^= C
                }
            }
        }
        for (int L = m - 1; L >= 0; L--) {               // butterflies
            size_t half = n >> 1, mask = ((size_t)1 << L) - 1;
            for (size_t p = 0; p < half; p++) {
                size_t t = p >> L, id = p & mask;
                size_t i1 = id + (t << (L + 1)), i2 = i1 + ((size_t)1 << L);
                bfly_fwd(f[i1], f[i2], W[t]);
            }
        }
    }

    // inverse: values -> coefficients
    void inv(u64 *f) const {
        for (int L = 0; L <= m - 1; L++) {
            size_t half = n >> 1, mask = ((size_t)1 << L) - 1;
            for (size_t p = 0; p < half; p++) {
                size_t t = p >> L, id = p & mask;
                size_t i1 = id + (t << (L + 1)), i2 = i1 + ((size_t)1 << L);
                bfly_inv(f[i1], f[i2], W[t]);
            }
        }
        for (int L = m - 2; L >= 0; L--) {
            size_t S = n >> L;
            for (size_t B = 4; B <= S; B <<= 1) {
                size_t q = B >> 2, h = B >> 1;
                for (size_t o = 0; o < S; o += B) {
                    xr(f, (o + q) << L, (o + h) << L, q << L);      // undo op2
                    xr(f, (o + h) << L, (o + h + q) << L, q << L);  // undo op1
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// chunk packing: 32-bit chunks of a bit array, lifted to u64 elements
static inline u32 get_chunk(const u64 *a, size_t nwords, size_t k) {
    size_t pos = 32 * k, w = pos >> 6;
    if (w >= nwords) return 0;
    return (u32)(a[w] >> (pos & 63));    // (pos&63) is 0 or 32
}

static inline size_t bits_to_words(u64 bits) { return (size_t)((bits + 63) >> 6); }

// out must have mul_out_words(abits,bbits) words; it is fully overwritten
static inline size_t mul_out_words(u64 abits, u64 bbits) {
    return bits_to_words(abits + bbits) + 1;
}

// Full multiply c = a*b over GF(2)[x].  Scratch FFT buffers are supplied
// by the caller (each n elements) so big runs can reuse allocations.
static inline void gf2x_fft_mul_buf(const u64 *a, u64 abits,
                                    const u64 *b, u64 bbits,
                                    u64 *out, const CantorFFT &F,
                                    u64 *FA, u64 *FB) {
    size_t ca = (size_t)((abits + 31) >> 5), cb = (size_t)((bbits + 31) >> 5);
    size_t n = F.n;
    if (ca + cb - 1 > n) { fprintf(stderr, "FFT too small\n"); exit(1); }
    size_t wa = bits_to_words(abits), wb = bits_to_words(bbits);
    for (size_t k = 0; k < n; k++) FA[k] = (k < ca) ? get_chunk(a, wa, k) : 0;
    for (size_t k = 0; k < n; k++) FB[k] = (k < cb) ? get_chunk(b, wb, k) : 0;
    F.fwd(FA);
    F.fwd(FB);
    for (size_t k = 0; k < n; k++) FA[k] = gf64_mul(FA[k], FB[k]);
    F.inv(FA);
    size_t ow = mul_out_words(abits, bbits);
    memset(out, 0, ow * sizeof(u64));
    for (size_t k = 0; k < ca + cb - 1; k++) {
        u64 e = FA[k];
        size_t pos = 32 * k, w = pos >> 6, sh = pos & 63;
        out[w] ^= e << sh;
        if (sh) out[w + 1] ^= e >> 32;
    }
}

// convenience wrapper (allocates everything)
static inline void gf2x_fft_mul(const u64 *a, u64 abits,
                                const u64 *b, u64 bbits, u64 *out,
                                const CantorBasis &B) {
    size_t ca = (size_t)((abits + 31) >> 5), cb = (size_t)((bbits + 31) >> 5);
    int m = 1;
    while (((size_t)1 << m) < ca + cb - 1) m++;
    CantorFFT F;
    F.init(m, B);
    std::vector<u64> FA(F.n), FB(F.n);
    gf2x_fft_mul_buf(a, abits, b, bbits, out, F, FA.data(), FB.data());
}

// ---------------------------------------------------------------------------
// Independent oracles for testing
// naive bit-serial schoolbook (small sizes)
static inline void gf2x_naive_mul(const u64 *a, u64 abits,
                                  const u64 *b, u64 bbits, u64 *out) {
    size_t ow = mul_out_words(abits, bbits), wb = bits_to_words(bbits);
    memset(out, 0, ow * sizeof(u64));
    for (u64 i = 0; i < abits; i++) {
        if (!((a[i >> 6] >> (i & 63)) & 1)) continue;
        size_t w = (size_t)(i >> 6), sh = (size_t)(i & 63);
        for (size_t j = 0; j < wb; j++) {
            u64 v = b[j];
            out[w + j] ^= v << sh;
            if (sh) out[w + j + 1] ^= v >> (64 - sh);
        }
    }
}

// word-comb schoolbook over 64-bit words using cl64 (medium sizes)
static inline void gf2x_comb_mul(const u64 *a, u64 abits,
                                 const u64 *b, u64 bbits, u64 *out) {
    size_t wa = bits_to_words(abits), wb = bits_to_words(bbits);
    size_t ow = mul_out_words(abits, bbits);
    memset(out, 0, ow * sizeof(u64));
    for (size_t i = 0; i < wa; i++)
        for (size_t j = 0; j < wb; j++) {
            cl128 p = cl64(a[i], b[j]);
            out[i + j] ^= p.lo;
            out[i + j + 1] ^= p.hi;
        }
}

// ---------------------------------------------------------------------------
// residue checks modulo an arbitrary irreducible q of degree d <= 62
struct ModP {
    u64 q; int d;
    u64 mul(u64 a, u64 b) const {
        cl128 t = cl64(a, b);
        for (int bit = 2 * d - 2; bit >= d; bit--) {
            int w = bit >> 6, s = bit & 63;
            u64 have = ((w ? t.hi : t.lo) >> s) & 1;
            if (!have) continue;
            int sh = bit - d;                       // xor q << sh into (hi,lo)
            t.lo ^= q << sh;
            if (sh) t.hi ^= q >> (64 - sh);
        }
        return t.lo;
    }
    u64 red64(u64 v) const {                        // reduce a 64-bit value
        for (int bit = 63; bit >= d; bit--)
            if ((v >> bit) & 1) v ^= q << (bit - d);
        return v;
    }
    u64 pw(u64 a, u64 e) const {
        u64 r = 1;
        while (e) { if (e & 1) r = mul(r, a); a = mul(a, a); e >>= 1; }
        return r;
    }
    bool irreducible() const {
        // x^(2^d) == x, and x^(2^(d/p)) != x for prime p | d
        u64 t = 2;
        std::vector<u64> chain(d + 1);
        chain[0] = t;
        for (int i = 1; i <= d; i++) { t = mul(t, t); chain[i] = t; }
        if (chain[d] != 2) return false;
        int dd = d;
        for (int p = 2; p <= dd; p++) {
            if (dd % p) continue;
            if (chain[d / p] == 2) return false;   // x fixed by too-small power
            while (dd % p == 0) dd /= p;
        }
        return true;
    }
    // value of bit-array a (abits bits) mod q, Horner over 64-bit words
    u64 residue(const u64 *a, u64 abits) const {
        size_t wa = bits_to_words(abits);
        u64 x64 = pw(2, 64), r = 0;
        for (size_t w = wa; w-- > 0;)
            r = mul(r, x64) ^ red64(a[w]);
        return r;
    }
};

static inline ModP find_irreducible_modp(int d, u64 seed) {
    ModP p;
    p.d = d;
    for (u64 low = seed | 1;; low += 2) {
        p.q = ((u64)1 << d) | (low & (((u64)1 << d) - 1));
        if (p.irreducible()) return p;
    }
}

#endif // GF2_CANTOR_ENGINE_H

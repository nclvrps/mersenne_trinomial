// gf2_dlog.h
// Table-light discrete logarithms in GF(2^d), d <= 47, for coarse_sieve's
// "logs" path (-DWIDE_FIELD build only).  The sieve needs, per cyclotomic
// class, one Zech logarithm log(g^u + 1); instead of a 2^d-entry log table
// (or its bucketed slices) it computes that one log directly:
//
//   tower  (even d = 2m): GF(2^d) = K(w), K = GF(2^m), w = x.  Every element
//          is a + b*w with a, b in K; for b != 0 it equals b*(c + w),
//          c = a/b, so  log e = (2^m+1)*logK(b) + T[c].  The split e -> (a,b)
//          is GF(2)-linear (byte tables); tables have 2^m entries.
//   ph     (any d): Pohlig-Hellman over a coprime split M = P_1*...*P_n of
//          M = 2^d - 1 (n <= 4, typically 2 balanced parts of ~2^(d/2)):
//          e^(E_i) lies in the order-P_i subgroup and a hash table (with
//          baby-step/giant-step when P_i exceeds the table budget) gives
//          v mod P_i; CRT gives v.  The exponents E_i = c_i * M/P_i are
//          chosen (over c_i coprime to P_i, and rotations) to have few set
//          bits in a short window: few general multiplies, short squaring
//          chain.
//
// Also here: carry-less arithmetic that is ~8x cheaper than gf_mul's
// bit-serial loops (integer-multiply clmul with 4-bit holes, BearSSL-style;
// two-fold reduction by the sparse low part of the modulus), and the u64
// helpers the sweep needs (Mersenne mulmod/powmod, exact-division tests).
// Everything is plain host+device code: no tables in shared memory, no
// warp intrinsics, so cuda_emul.h validates it exactly.

#ifndef GF2_DLOG_H
#define GF2_DLOG_H
#include "gf2_wide_field.h"

// ---------------------------------------------------------------------------
// fast field arithmetic (d <= 47, modulus q = x^d + qlow, deg(qlow) <= 16)

// 32x32 -> 64 carry-less product via integer multiplies.  Each operand is
// split into 4 bit-classes (bits = k mod 4); an integer product of two
// classes has at most 8 terms per output position, so no carry ever
// crosses into the next position of the same class, and bit p of the
// product is the parity of its count.  XOR the four products landing in
// class k, then mask class k.
GF2_HD u64 gf2f_clmul32(u32 x, u32 y) {
    const u32 m0 = 0x11111111u, m1 = 0x22222222u, m2 = 0x44444444u,
              m3 = 0x88888888u;
    u32 x0 = x & m0, x1 = x & m1, x2 = x & m2, x3 = x & m3;
    u32 y0 = y & m0, y1 = y & m1, y2 = y & m2, y3 = y & m3;
    u64 z0 = ((u64)x0 * y0) ^ ((u64)x1 * y3) ^ ((u64)x2 * y2) ^ ((u64)x3 * y1);
    u64 z1 = ((u64)x0 * y1) ^ ((u64)x1 * y0) ^ ((u64)x2 * y3) ^ ((u64)x3 * y2);
    u64 z2 = ((u64)x0 * y2) ^ ((u64)x1 * y1) ^ ((u64)x2 * y0) ^ ((u64)x3 * y3);
    u64 z3 = ((u64)x0 * y3) ^ ((u64)x1 * y2) ^ ((u64)x2 * y1) ^ ((u64)x3 * y0);
    return (z0 & 0x1111111111111111ULL) | (z1 & 0x2222222222222222ULL) |
           (z2 & 0x4444444444444444ULL) | (z3 & 0x8888888888888888ULL);
}

// 64x64 -> 128 carry-less product, one Karatsuba level over 32-bit halves.
GF2_HD void gf2f_clmul64(u64 a, u64 b, u64 *hi, u64 *lo) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
    u64 l = gf2f_clmul32(a0, b0), h = gf2f_clmul32(a1, b1);
    u64 m = gf2f_clmul32(a0 ^ a1, b0 ^ b1) ^ l ^ h;
    *lo = l ^ (m << 32);
    *hi = h ^ (m >> 32);
}

GF2_HD int gf2f_ctz64(u64 x) {              // x != 0
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}

GF2_HD int gf2f_ctz32(u32 x) {              // x != 0
#if defined(__CUDA_ARCH__)
    return __ffs((int)x) - 1;
#else
    return __builtin_ctz(x);
#endif
}

GF2_HD int gf2f_clz64(u64 x) {              // x != 0
#if defined(__CUDA_ARCH__)
    return __clzll((long long)x);
#else
    return __builtin_clzll(x);
#endif
}

// p * qlow for p < 2^(64 - deg qlow): XOR of shifted copies, one per set
// bit of qlow (2..6 for every d <= 47; uniform across threads).  qlow has
// degree <= 8 here (< 2^25 whenever gf2f_usable holds), so the loop runs on
// 32-bit control: ~10 instead of ~18 instructions per term on the GPU.
GF2_HD u64 gf2f_mul_low(u64 p, u64 qlow) {
    u64 t = 0;
    for (u32 ql = (u32)qlow; ql; ql &= ql - 1) t ^= p << gf2f_ctz32(ql);
    return t;
}

// Reduce hi:lo (degree <= 2d-2) mod x^d + qlow.  Two folds suffice when
// 2*deg(qlow) - 2 < d and d + deg(qlow) <= 64 (checked by gf2f_usable).
GF2_HD u64 gf2f_reduce(u64 hi, u64 lo, u64 qlow, int d) {
    const u64 mask = ((u64)1 << d) - 1;
    u64 ph = (lo >> d) | (hi << (64 - d));      // P >> d, degree <= d-2
    u64 t = gf2f_mul_low(ph, qlow);             // degree <= d-2+deg(qlow)
    u64 th = t >> d;                            // degree <= deg(qlow)-2
    return (lo & mask) ^ (t & mask) ^ gf2f_mul_low(th, qlow);
}

GF2_HD u64 gf2f_mul(u64 a, u64 b, u64 qlow, int d) {
    u64 hi, lo;
    gf2f_clmul64(a, b, &hi, &lo);
    return gf2f_reduce(hi, lo, qlow, d);
}

// bit-spread of a 32-bit value: bit i -> bit 2i
GF2_HD u64 gf2f_spread32(u32 v) {
    u64 x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

GF2_HD u64 gf2f_sqr(u64 a, u64 qlow, int d) {
    return gf2f_reduce(gf2f_spread32((u32)(a >> 32)),
                       gf2f_spread32((u32)a), qlow, d);
}

// g^idx for g = x, from gpow2[t] = g^(2^t) (fast multiply)
GF2_HD u64 gf2f_antilog(u64 idx, u64 qlow, int d, const u64 *gpow2) {
    u64 v = 1;
    for (int t = 0; idx; idx >>= 1, t++)
        if (idx & 1) v = gf2f_mul(v, gpow2[t], qlow, d);
    return v;
}

// base^e (fast multiply/square)
GF2_HD u64 gf2f_pow(u64 base, u64 e, u64 qlow, int d) {
    u64 r = 1;
    for (; e; e >>= 1) {
        if (e & 1) r = gf2f_mul(r, base, qlow, d);
        if (e > 1) base = gf2f_sqr(base, qlow, d);
    }
    return r;
}

// host check: can the fast path handle this modulus?
static inline bool gf2f_usable(u64 q, int d) {
    if (d < 2 || d > 47) return false;
    u64 ql = q ^ ((u64)1 << d);
    if (!ql) return false;
    int dl = 63 - __builtin_clzll(ql);
    return 2 * dl - 2 < d && d + dl <= 64 && dl < 32;   // dl < 32: mul_low
}

// ---------------------------------------------------------------------------
// u64 helpers for exponent arithmetic modulo M = 2^d - 1

// (a*b) mod M, a,b < M, M = 2^d - 1 (d <= 63): fold the product at bit d.
GF2_HD u64 gf2f_mulmod_m(u64 a, u64 b, int d, u64 M) {
#if defined(__CUDA_ARCH__)
    u64 lo = a * b, hi = __umul64hi(a, b);
#else
    u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
    u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
    u64 lo = (mid << 32) | (u32)p00;
    u64 hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
    u64 t = (lo & M) + ((lo >> d) | (hi << (64 - d)));
    t = (t & M) + (t >> d);
    if (t >= M) t -= M;
    return t;
}

// a^e mod M (left-to-right; the exponent is uniform across threads)
GF2_HD u64 gf2f_powmod_m(u64 a, u64 e, int d, u64 M) {
    if (e == 0) return 1 % M;
    u64 r = a;
    for (int b = 62 - gf2f_clz64(e); b >= 0; b--) {   // below the top bit
        r = gf2f_mulmod_m(r, r, d, M);
        if ((e >> b) & 1) r = gf2f_mulmod_m(r, a, d, M);
    }
    return r;
}

// Exact-division test for an odd divisor p: with pinv = p^-1 mod 2^64 and
// plim = floor((2^64-1)/p),  p | x  <=>  x*pinv <= plim,  and then
// x/p == x*pinv.  No 64-bit division (software-emulated on GPUs).
struct OddDiv { u64 p, pinv, plim; };

static inline OddDiv odd_div_make(u64 p) {
    OddDiv o;
    o.p = p;
    u64 inv = p;                                // Newton: 5 steps -> 64 bits
    for (int i = 0; i < 5; i++) inv *= 2 - p * inv;
    o.pinv = inv;
    o.plim = ~0ULL / p;
    return o;
}

GF2_HD bool odd_div_test(u64 x, const OddDiv &o, u64 *quot) {
    u64 qq = x * o.pinv;
    if (qq > o.plim) return false;
    *quot = qq;
    return true;
}

// ---------------------------------------------------------------------------
// Is u the least of its d-bit rotations?  Requires u odd and u < 2^(d-1)
// (the only candidates the sweep offers).
//
// Let z >= 1 be the number of leading zeros of u as a d-bit string.  As u is
// odd, no zero run of the cyclic string passes through bit 0, so the runs
// are ordinary runs of ones in w = ~u & M.  Any rotation smaller than u
// starts with >= z zeros: it starts either in a run of >= z+1 zeros (then it
// is below 2^(d-1-z) <= u outright) or at another run of exactly z zeros
// (then it has to be compared).  zero_runs(w, k) has bit b set iff bits
// b..b+k-1 of u are all zero; it takes log2(k) shift-ANDs.
//
// This replaces a loop over all d-1 rotations whose data-dependent trip
// count kept warps busy for ~30 iterations per u, most of the sweep's time
// at large d; the result is the same for every such u (checked against
// gf2f_min_rotation_ref exhaustively for d <= 24 and on random u for every
// d <= 47 by t_rot.cpp, and on the device by --selftest-logs).
GF2_HD u64 gf2f_zero_runs(u64 w, int k) {    // k >= 1
    int m = 1;
    while (2 * m <= k) { w &= w >> m; m *= 2; }
    if (k > m) w &= w >> (k - m);
    return w;
}

GF2_HD bool gf2f_is_min_rotation(u64 u, int d, u64 M) {
    const int z = gf2f_clz64(u) - (64 - d);    // leading zeros, 1..d-1
    u64 y = gf2f_zero_runs(~u & M, z);         // starts of runs of >= z zeros
    if (y & (y >> 1)) return false;            // some run has >= z+1 zeros
    y &= ~((u64)1 << (d - z));                 // u's own leading run
    for (; y; y &= y - 1) {                    // the other runs of exactly z
        const int i = d - z - gf2f_ctz64(y);   // rotation putting it on top
        if ((((u << i) | (u >> (d - i))) & M) < u) return false;
    }
    return true;
}

// reference: compare with all d-1 rotations
GF2_HD bool gf2f_min_rotation_ref(u64 u, int d, u64 M) {
    u64 x = u;
    for (int i = 1; i < d; i++) {
        x = ((x << 1) | (x >> (d - 1))) & M;
        if (x < u) return false;
    }
    return true;
}

#if defined(__CUDACC__)
#define GF2D_UNROLL _Pragma("unroll")
#else
#define GF2D_UNROLL
#endif

// ---------------------------------------------------------------------------
// tower logs (d = 2m).  All pointers are device pointers; structs are
// passed to kernels by value.
struct TowerDev {
    int m, nb;            // nb = ceil(d/8) byte tables
    u64 MK;               // 2^m - 1 (also the coordinate mask)
    const u64 *btab;      // [nb][256]: packed coords coordK(a) | coordK(b)<<m
    const u32 *LK;        // [2^m]   coordinate -> log base h (h = g^(2^m+1))
    const u32 *AK;        // [2^m-1] log base h -> coordinate
    const u64 *T;         // [2^m]   c -> log_g(c + w)
};

GF2_HD u64 tower_coords(u64 x, const TowerDev &t) {
    u64 pk = 0;
    for (int k = 0; k < t.nb; k++) pk ^= t.btab[k * 256 + ((x >> (8 * k)) & 255)];
    return pk;
}

// log_g(x) for x != 0
GF2_HD u64 tower_log(u64 x, u64 M, const TowerDev &t) {
    u64 pk = tower_coords(x, t);
    u64 ac = pk & t.MK, bc = pk >> t.m;
    u64 h1 = t.MK + 2;                          // 2^m + 1 = log_g(h)
    if (bc == 0) return h1 * t.LK[ac];          // x in K*;  < M
    u64 lb = t.LK[bc], cc = 0;
    if (ac) {
        u64 la = t.LK[ac];
        cc = t.AK[la >= lb ? la - lb : la + t.MK - lb];   // c = a / b
    }
    u64 v = h1 * lb + t.T[cc];                  // h1*lb < M, T < M
    return v >= M ? v - M : v;
}

// ---------------------------------------------------------------------------
// Pohlig-Hellman logs
#define PH_MAXPARTS 4

struct PHPart {
    u64 P;                // order of the subgroup this part resolves
    u64 E;                // exponent: x^E = gamma^(log x mod P)
    u64 K;                // CRT constant: log x = sum (log x mod P_i)*K_i mod M
    u64 N, G;             // baby table holds gamma^0..gamma^(N-1); G = ceil(P/N)
    u64 delta;            // gamma^(-N): one giant step
    const u64 *keys;      // 2^sbits slots, 0 = empty (subgroup elements != 0)
    const u32 *vals;
    int sbits;
};

struct PHDev {
    int nparts, chain;    // chain = highest set bit over all E_i
    PHPart part[PH_MAXPARTS];
};

GF2_HD u64 ph_slot(u64 key, int sbits) {
    return (key * 0x9E3779B97F4A7C15ULL) >> (64 - sbits);
}

// index i < N with gamma^i == y, or ~0
GF2_HD u64 ph_lookup(const PHPart &p, u64 y) {
    const u64 mask = ((u64)1 << p.sbits) - 1;
    for (u64 h = ph_slot(y, p.sbits);; h = (h + 1) & mask) {
        u64 k = p.keys[h];
        if (k == y) return p.vals[h];
        if (k == 0) return ~0ULL;
    }
}

// log_g(x) for x != 0, or ~0 on an internal inconsistency (never expected;
// the caller counts it as an error)
GF2_HD u64 ph_log(u64 x, u64 M, int d, u64 qlow, const PHDev &ph) {
    u64 acc[PH_MAXPARTS];
    GF2D_UNROLL
    for (int i = 0; i < PH_MAXPARTS; i++) acc[i] = 0;    // 0 = no factor yet
    u64 cur = x;                                // x^(2^t)
    for (int t = 0; t <= ph.chain; t++) {
        GF2D_UNROLL
        for (int i = 0; i < PH_MAXPARTS; i++)
            if (i < ph.nparts && ((ph.part[i].E >> t) & 1))
                acc[i] = acc[i] ? gf2f_mul(acc[i], cur, qlow, d) : cur;
        if (t < ph.chain) cur = gf2f_sqr(cur, qlow, d);
    }
    u64 v = 0;
    GF2D_UNROLL
    for (int i = 0; i < PH_MAXPARTS; i++) {
        if (i >= ph.nparts) continue;
        const PHPart &p = ph.part[i];
        u64 y = acc[i], xi = ~0ULL;
        for (u64 s = 0; s < p.G; s++) {
            u64 idx = ph_lookup(p, y);
            if (idx != ~0ULL) { xi = idx + s * p.N; break; }
            y = gf2f_mul(y, p.delta, qlow, d);
        }
        if (xi >= p.P) return ~0ULL;
        v += gf2f_mulmod_m(xi, p.K, d, M);
        if (v >= M) v -= M;
    }
    return v;
}

// ---------------------------------------------------------------------------
// host-side builders (one-off per degree; bit-serial gf_mul is fine here)
#include "gf2_wide_setup.h"
#include <vector>
#include <algorithm>
#include <functional>

struct TowerHost {
    int m = 0, nb = 0;
    u64 MK = 0;
    std::vector<u64> btab;               // nb*256
    std::vector<u32> LK, AK;             // 2^m, 2^m - 1
};

// Byte tables for e -> (coordK(a) | coordK(b) << m), and LK/AK for K*.
// T (the 2^m logs log(c + w)) is filled on the device (k_tower_T).
// Returns false if any internal consistency check fails.
static bool tower_build_host(int d, u64 q, TowerHost &W) {
    if (d < 4 || (d & 1) || d > 46) return false;
    const int m = d / 2;
    const u64 M = ((u64)1 << d) - 1, MK = ((u64)1 << m) - 1;
    std::vector<u64> sig(d);                    // sigma(x^i) = x^(i*2^m)
    for (int i = 0; i < d; i++) {
        u64 y = (u64)1 << i;
        for (int k = 0; k < m; k++) y = gf_sqr(y, q, d);
        sig[i] = y;
    }
    auto lin_sigma = [&](u64 e) {
        u64 r = 0;
        for (int i = 0; i < d; i++) if (e >> i & 1) r ^= sig[i];
        return r;
    };
    u64 den = 2 ^ lin_sigma(2);                 // w + sigma(w), w = x
    if (!den) return false;
    u64 inv_den = gf_pow(den, M - 1, q, d);
    // basis of K: 1, h, ..., h^(m-1), h = g^(2^m+1); full RREF
    u64 h = gf_pow(2, MK + 2, q, d);
    std::vector<u64> rows;
    u64 y = 1;
    for (int k = 0; k < m; k++, y = gf_mul(y, h, q, d)) {
        u64 v = y;
        for (u64 r : rows) { int p = 63 - __builtin_clzll(r); if (v >> p & 1) v ^= r; }
        if (!v) return false;
        rows.push_back(v);
        std::sort(rows.begin(), rows.end(), std::greater<u64>());
    }
    for (size_t i = 0; i < rows.size(); i++) {
        int p = 63 - __builtin_clzll(rows[i]);
        for (size_t j = 0; j < rows.size(); j++)
            if (j != i && (rows[j] >> p & 1)) rows[j] ^= rows[i];
    }
    std::vector<int> piv(m);
    for (int k = 0; k < m; k++) piv[k] = 63 - __builtin_clzll(rows[k]);
    auto coordK = [&](u64 e) {
        u64 c = 0;
        for (int k = 0; k < m; k++) c |= (e >> piv[k] & 1) << k;
        return c;
    };
    auto embedK = [&](u64 c) {
        u64 e = 0;
        for (int k = 0; k < m; k++) if (c >> k & 1) e ^= rows[k];
        return e;
    };
    W.m = m; W.MK = MK; W.nb = (d + 7) / 8;
    W.btab.assign((size_t)W.nb * 256, 0);
    for (int k = 0; k < W.nb; k++)
        for (int byte = 0; byte < 256; byte++) {
            u64 e = ((u64)byte << (8 * k)) & M;
            u64 b = gf_mul(e ^ lin_sigma(e), inv_den, q, d);
            u64 a = e ^ gf_mul(b, 2, q, d);
            u64 ca = coordK(a), cb = coordK(b);
            if (embedK(ca) != a || embedK(cb) != b) return false;   // a,b in K
            W.btab[(size_t)k * 256 + byte] = ca | (cb << m);
        }
    // K* = <h>: step by h in coordinates (a linear map) -> AK, LK
    std::vector<u64> hcol(m);
    for (int k = 0; k < m; k++) {
        u64 t = gf_mul(h, rows[k], q, d);
        hcol[k] = coordK(t);
        if (embedK(hcol[k]) != t) return false;
    }
    const int mb = (m + 7) / 8;
    std::vector<u64> hb((size_t)mb * 256, 0);
    for (int kb = 0; kb < mb; kb++)
        for (int byte = 0; byte < 256; byte++) {
            u64 r = 0;
            for (int bit = 0; bit < 8; bit++)
                if ((byte >> bit & 1) && 8 * kb + bit < m) r ^= hcol[8 * kb + bit];
            hb[(size_t)kb * 256 + byte] = r;
        }
    W.LK.assign((size_t)MK + 1, 0xFFFFFFFFu);
    W.AK.assign((size_t)MK, 0);
    u64 c = coordK(1);
    for (u64 k = 0; k < MK; k++) {
        if (!c || c > MK || W.LK[c] != 0xFFFFFFFFu) return false;   // h not of order MK?!
        W.AK[k] = (u32)c;
        W.LK[c] = (u32)k;
        u64 nc = 0;
        for (int kb = 0; kb < mb; kb++) nc ^= hb[(size_t)kb * 256 + ((c >> (8 * kb)) & 255)];
        c = nc;
    }
    return c == coordK(1);
}

struct PHPartHost { u64 P = 0, E = 0, K = 0, N = 0, G = 1, delta = 1, gamma = 1; int sbits = 1; };
struct PHHost { int nparts = 0, chain = 0; PHPartHost part[PH_MAXPARTS]; size_t table_bytes = 0; };

// slot count 2^s >= 2N (load <= 1/2): 8-byte key + 4-byte value per slot
static inline size_t ph_table_bytes(u64 N, int *sbits_out) {
    int s = 1;
    while (((u64)1 << s) < 2 * N) s++;
    if (sbits_out) *sbits_out = s;
    return ((size_t)1 << s) * 12;
}

// Plan Pohlig-Hellman: coprime split of M into <= 2 parts (as balanced as
// possible), cheapest exponents, CRT constants, and table sizes within
// budget_bytes (shrinking the largest table into baby-step/giant-step if
// needed).  Returns false if even small tables do not fit.
static bool ph_plan_host(int d, u64 q, size_t budget_bytes, PHHost &H) {
    const u64 M = ((u64)1 << d) - 1;
    std::vector<u64> primes = factor_u64(M), pp;   // prime powers p^k || M
    u64 rest = M;
    for (u64 p : primes) {
        u64 pk = 1;
        while (rest % p == 0) { rest /= p; pk *= p; }
        pp.push_back(pk);
    }
    std::vector<u64> parts;
    if (pp.size() == 1) parts.push_back(M);
    else {
        const int n = (int)pp.size();
        u64 bestA = 1, bestmax = ~0ULL;
        for (u64 mask = 1; mask + 1 < ((u64)1 << n); mask++) {
            u64 A = 1;
            for (int i = 0; i < n; i++) if (mask >> i & 1) A *= pp[i];
            u64 mx = std::max(A, M / A);
            if (mx < bestmax) { bestmax = mx; bestA = A; }
        }
        parts.push_back(std::max(bestA, M / bestA));
        parts.push_back(std::min(bestA, M / bestA));
    }
    H = PHHost();
    H.nparts = (int)parts.size();
    for (int i = 0; i < H.nparts; i++) {
        PHPartHost &pt = H.part[i];
        const u64 P = parts[i], C = M / P;
        std::vector<u64> pf;
        for (u64 p : primes) if (P % p == 0) pf.push_back(p);
        // E = c*C mod M over c coprime to P (and its rotations, i.e.
        // c*2^-r): few set bits (general multiplies) in a low window
        // (squarings); cost ~ 2*(popcount-1) + highest bit
        const u64 lim = std::min<u64>(P, (u64)1 << 22);
        u64 E = 0, bestE = C;
        int bestcost = 1 << 30;
        for (u64 c = 1; c < lim; c++) {
            E += C; if (E >= M) E -= M;
            int pc = __builtin_popcountll(E);
            if (3 * (pc - 1) >= bestcost) continue;     // span >= pc-1
            bool cop = true;
            for (u64 p : pf) if (c % p == 0) { cop = false; break; }
            if (!cop) continue;
            u64 Er = E;
            for (int r = 0; r < d; r++) {
                int cost = 2 * (pc - 1) + (63 - __builtin_clzll(Er));
                if (cost < bestcost) { bestcost = cost; bestE = Er; }
                Er = (Er >> 1) | ((Er & 1) << (d - 1));  // * 2^-1 mod M
            }
        }
        pt.P = P;
        pt.E = bestE;
        pt.K = mulmod_u64(C, P == 1 ? 0 : inv_mod(C % P, P), M);
        pt.gamma = gf_pow(2, bestE, q, d);
        pt.N = P;
        H.chain = std::max(H.chain, 63 - __builtin_clzll(bestE));
    }
    auto total = [&]() {
        size_t t = 0;
        for (int i = 0; i < H.nparts; i++) t += ph_table_bytes(H.part[i].N, nullptr);
        return t;
    };
    while (total() > budget_bytes) {
        int big = 0;
        for (int i = 1; i < H.nparts; i++) if (H.part[i].N > H.part[big].N) big = i;
        int sb;
        ph_table_bytes(H.part[big].N, &sb);
        if (sb <= 12) return false;
        H.part[big].N = (u64)1 << (sb - 2);           // half the slots, load 1/2
    }
    for (int i = 0; i < H.nparts; i++) {
        PHPartHost &pt = H.part[i];
        ph_table_bytes(pt.N, &pt.sbits);
        pt.G = (pt.P + pt.N - 1) / pt.N;
        pt.delta = (pt.N >= pt.P) ? 1 : gf_pow(pt.gamma, pt.P - pt.N, q, d);
    }
    H.table_bytes = total();
    return true;
}

#endif // GF2_DLOG_H

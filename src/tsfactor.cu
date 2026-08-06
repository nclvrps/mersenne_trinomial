// tsfactor.cu
// Synthesis of gf2x/apps/factor.cpp (Brent/Gaudry/Thome/Zimmermann,
// GPL) and trinomial_scan.cu: finds the smallest-degree irreducible
// factor of x^r + x^s + 1 over GF(2) for each survivor s read from a
// file, using factor.cpp's multi-level blocking DDF with the
// squarings and the block modmuls on the GPU.
//
// The algorithm (decoded from factor.cpp; see tsfactor_handoff.md):
// maintain h[j] = e_{j+1}(y_0..y_{m-1}), the elementary symmetric
// functions of the Frobenius iterates y_t = x^(2^(i+t)) mod T covering
// the current block of m degrees [i, i+m-1].  Then
//     C = x^m + h[0] x^(m-1) + ... + h[m-1]
//       = prod_t (x + x^(2^(i+t)))            (mod T)
// captures exactly the degrees of the block (an irreducible of degree
// d divides x + x^(2^c) iff d | c); advancing the block by m degrees
// is h[j] <- h[j]^(2^m), i.e. m squarings per h (Frobenius commutes
// with the e_k).  Per degree: m linear-time squarings + 1/m FFT-size
// modmuls -- the same arithmetic balance as factor.cpp, with both
// legs on the GPU.  One gcd(A, T) per interval of q blocks (q grows
// linearly as in factor -f 1) runs on a CPU thread pool, fully
// overlapped with speculative scanning; verdicts are consumed
// strictly in interval order.  A hit is resolved by factoring the
// small interval gcd (CanZass); oversized gcds trigger factor.cpp's
// fineDDF-style interval bisection first.
//
// Output lines (results file and stdout), matching factor.cpp:
//     s d p<hex>     least-degree factor, numerically least mask
//     s u            gave up (-z tripped, or --maxd exceeded)
//     s primitive    no factor to the Swan bound => irreducible =>
//                    primitive for Mersenne-exponent r  (loudly)
//     s rA-B         pathological: factor with degree in [A,B],
//                    interval gcd not capturable (hand to factor.cpp)
//
// Build:   make tsfactor          (CUDA; NTL auto-detected, required
//                                  for production r)
//          make emul-tsfactor     (CPU emulation of the kernels)
// Test:    ./tsfactor --selftest
// Bench:   ./tsfactor --bench <r> [s]
// Run:     ./tsfactor <r> <survivors.txt> --skip 36 [options]
//
// This program is free software, GPL v3 or later, and incorporates
// algorithmic structure from gf2x/apps/factor.cpp (GPL).

#include "gf2_trinomial_cuda.h"
#include "gf2_gcd.h"
#include <cinttypes>
#include <algorithm>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <signal.h>
#include <cstdlib>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

// ---------------------------------------------------------------------------
// new kernels (all gather-form: every output word is produced by one
// thread from reads only -> race-free and exact under the sequential
// CPU emulation harness)

// C[w] = XOR_j (h_j << (m-1-j))[w], h_j = hbase + j*nw, for w < nwc.
// Word w of (v << sh): (v[w] << sh) | (v[w-1] >> (64-sh)); v[i>=nw]=0.
__global__ void k_horner_fused(const u64 *hbase, size_t nw, int m,
                               u64 *C, size_t nwc) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nwc; w += stride) {
        u64 acc = 0;
        for (int j = 0; j < m; j++) {
            const u64 *h = hbase + (size_t)j * nw;
            unsigned sh = (unsigned)(m - 1 - j);
            u64 a = (w < nw) ? h[w] : 0;
            u64 b = (w >= 1 && w - 1 < nw) ? h[w - 1] : 0;
            acc ^= sh ? ((a << sh) | (b >> (64 - sh))) : a;
        }
        C[w] = acc;
    }
}

// tail of the fused Horner: fold bits [r, r+m-2] of C down
// (x^(r+e) = x^(s+e) + x^e), add the leading x^m term, and leave all
// bits >= r zero.  Single thread; <= 63 sequential iterations.
__global__ void k_horner_tail(u64 *C, u64 r, u64 s, int m) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int e = 0; e <= m - 2; e++) {
        u64 p = r + (u64)e;
        u64 bit = (C[p >> 6] >> (p & 63)) & 1;
        if (bit) {
            C[p >> 6] ^= 1ULL << (p & 63);
            u64 t1 = s + (u64)e;
            C[t1 >> 6] ^= 1ULL << (t1 & 63);
            C[(u64)e >> 6] ^= 1ULL << (e & 63);
        }
    }
    C[(u64)m >> 6] ^= 1ULL << (m & 63);         // + x^m  (m < r)
}

// dst ^= (src * x) mod (x^r + x^s + 1); dst != src; both nw words with
// bits >= r zero.  Wrap: if src bit r-1 is set, the shifted x^r term
// becomes x^s + 1.
__global__ void k_mulx_xor(u64 *dst, const u64 *src, size_t nw,
                           u64 r, u64 s) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    u64 topw = (r - 1) >> 6;
    unsigned topb = (unsigned)((r - 1) & 63);
    for (; w < nw; w += stride) {
        u64 a = src[w];
        u64 b = w ? src[w - 1] : 0;
        u64 v = (a << 1) | (b >> 63);
        // clear the shifted-out x^r bit if present in this word
        if (w == (r >> 6)) v &= ~(1ULL << (r & 63));
        u64 wrap = (src[topw] >> topb) & 1;      // coefficient of x^(r-1)
        if (w == 0) v ^= wrap;                   // + 1
        if (w == (s >> 6)) v ^= wrap << (s & 63);// + x^s
        dst[w] ^= v;
    }
}

// out = in^2 mod (x^r + x^s + 1), fused spread + two fold passes.
// Derivation (tsfactor_handoff.md / verified vs the 3-kernel path):
// with SP the virtual bit-spread of `in` (SP[2i] = in[i]) of length
// 2r-1, and V(pos,lo,hi) the 64-bit window of SP at [pos,pos+64)
// clipped to SP-indices [lo,hi):
//   out_word(w) = V(b,      0,    r   )
//               ^ V(b+r,    r,    2r-1)
//               ^ V(b+r-s,  r,    2r-s)
//               ^ V(b+2r-s, 2r-s, 2r-1)
//               ^ V(b+2r-2s,2r-s, 2r-1),   b = 64w.
// Requires only 1 <= s <= (r-1)/2 (no parity constraint).
__device__ __forceinline__ u64 ts_spread32(u32 v) {
    u64 x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}
__device__ __forceinline__ u64 ts_in32(const u64 *in, size_t nw, u64 q) {
    // 32-bit window of `in` at bit q (bits beyond the array are 0)
    size_t w0 = (size_t)(q >> 6);
    unsigned sh = (unsigned)(q & 63);
    u64 a = (w0 < nw) ? in[w0] : 0;
    u64 b = (w0 + 1 < nw) ? in[w0 + 1] : 0;
    u64 v = sh ? ((a >> sh) | (b << (64 - sh))) : a;
    return v & 0xFFFFFFFFULL;
}
__device__ __forceinline__ u64 ts_vspread(const u64 *in, size_t nw,
                                          u64 pos, u64 lo, u64 hi) {
    // window of the virtual spread at [pos, pos+64), clipped to [lo,hi)
    u64 v;
    if ((pos & 1) == 0)
        v = ts_spread32(( u32)ts_in32(in, nw, pos >> 1));
    else
        v = ts_spread32((u32)ts_in32(in, nw, (pos + 1) >> 1)) << 1;
    u64 lc = (lo > pos) ? lo - pos : 0;
    u64 hc = (hi > pos) ? hi - pos : 0;
    if (lc >= 64 || hc == 0) return 0;
    if (hc > 64) hc = 64;
    u64 msk = (hc >= 64 ? ~0ULL : ((1ULL << hc) - 1));
    if (lc) msk &= ~((1ULL << lc) - 1);
    return v & msk;
}
__global__ void k_square_fused(const u64 *in, u64 *out, size_t nw,
                               u64 r, u64 s) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) {
        u64 b = (u64)w << 6;
        u64 v = ts_vspread(in, nw, b,               0,         r);
        v ^= ts_vspread(in, nw, b + r,              r,         2*r - 1);
        v ^= ts_vspread(in, nw, b + r - s,          r,         2*r - s);
        v ^= ts_vspread(in, nw, b + 2*r - s,        2*r - s,   2*r - 1);
        v ^= ts_vspread(in, nw, b + 2*(r - s),      2*r - s,   2*r - 1);
        out[w] = v;
    }
}

__global__ void k_xor_buf(u64 *dst, const u64 *src, size_t n) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < n; w += stride) dst[w] ^= src[w];
}
__global__ void k_set_zero(u64 *p, size_t nw) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) p[w] = 0;
}
__global__ void k_set_one(u64 *p, size_t nw) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) p[w] = (w == 0) ? 1ULL : 0;
}

// ---------------------------------------------------------------------------
// device context for the blocked scan

struct TsGPU {
    u64 r = 0, s = 0;
    size_t nw = 0;             // words for r bits
    size_t nw2 = 0;            // scratch words for 2r bits
    size_t nwc = 0;            // words for the unreduced Horner (r+m-1 bits)
    int m = 0;                 // block size (h slots); m <= 64
    int fftm = 0;
    CantorBasis Bb;
    CudaFFT F;
    u64 *dH[2] = {nullptr, nullptr};   // ping-pong, m*nw words each
    int hcur = 0;
    u64 *dHold = nullptr;      // fineDDF snapshot (allocated on demand)
    u64 *dA = nullptr;         // interval accumulator, nw
    u64 *dC = nullptr;         // block polynomial, nw+1 (Horner headroom)
    u64 *dTmp = nullptr;       // nw scratch (Pascal init)
    u64 *dS0 = nullptr;        // nw2 scratch
    u64 *dS1 = nullptr;        // nw2 scratch
    u64 *dFA = nullptr, *dFB = nullptr;
    bool legacy_sq = false;    // force 3-kernel squaring path
    double t_sq = 0, t_horner = 0, t_mul = 0;   // accumulated seconds

    void init(u64 r_) {
        r = r_;
        nw = bits_to_words(r);
        nw2 = 2 * nw;
        Bb.build();
        if (!Bb.verify()) { fprintf(stderr, "Cantor basis failure\n"); exit(1); }
        size_t ca = (size_t)((r + 31) >> 5);
        fftm = 1;
        while (((size_t)1 << fftm) < 2 * ca - 1) fftm++;
        F.init(fftm, Bb);
        CUCHK(cudaMalloc(&dA, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dC, (nw + 1) * sizeof(u64)));
        CUCHK(cudaMalloc(&dTmp, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dS0, nw2 * sizeof(u64)));
        CUCHK(cudaMalloc(&dS1, nw2 * sizeof(u64)));
        CUCHK(cudaMalloc(&dFA, F.n * sizeof(u64)));
        CUCHK(cudaMalloc(&dFB, F.n * sizeof(u64)));
    }
    void set_m(int m_) {
        if (m_ < 2 || m_ > 64) { fprintf(stderr, "need 2 <= m <= 64\n"); exit(1); }
        if (m == m_) return;
        for (int i = 0; i < 2; i++)
            if (dH[i]) { cudaFree(dH[i]); dH[i] = nullptr; }
        m = m_;
        nwc = bits_to_words(r + (u64)m - 1);
        CUCHK(cudaMalloc(&dH[0], (size_t)m * nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dH[1], (size_t)m * nw * sizeof(u64)));
    }
    void need_hold() {
        if (!dHold) CUCHK(cudaMalloc(&dHold, (size_t)m * nw * sizeof(u64)));
    }
    void set_s(u64 s_) {
        if (!(s_ >= 1 && 2 * s_ < r)) {
            fprintf(stderr, "need 1 <= s <= (r-1)/2 (got s=%" PRIu64 ")\n", s_);
            exit(1);
        }
        s = s_;
    }
    void fini() {
        for (int i = 0; i < 2; i++) if (dH[i]) cudaFree(dH[i]);
        if (dHold) cudaFree(dHold);
        cudaFree(dA); cudaFree(dC); cudaFree(dTmp);
        cudaFree(dS0); cudaFree(dS1);
        cudaFree(dFA); cudaFree(dFB);
        F.fini();
    }
    u64 *hslot(int j) { return dH[hcur] + (size_t)j * nw; }
    u64 *hslot_o(int j) { return dH[1 - hcur] + (size_t)j * nw; }

    // dst <- src^2 mod T (src != dst, both nw words, bits >= r zero)
    void square_slot(const u64 *src, u64 *dst) {
        int blocks, threads;
        if (!legacy_sq) {
            CudaFFT::launch_dims(nw, blocks, threads);
            k_square_fused<<<blocks, threads>>>(src, dst, nw, r, s);
        } else {
            CudaFFT::launch_dims(nw2, blocks, threads);
            k_spread_square<<<blocks, threads>>>(src, dS0, nw2);
            size_t w1 = bits_to_words(r + s - 1);
            CudaFFT::launch_dims(w1, blocks, threads);
            k_fold_pass<<<blocks, threads>>>(dS0, nw2, dS1, w1, r, s,
                                             2 * r - 1);
            CudaFFT::launch_dims(nw, blocks, threads);
            k_fold_pass<<<blocks, threads>>>(dS1, w1, dst, nw, r, s,
                                             r + s - 1);
        }
    }
    // one Frobenius step of the whole h-table: all m slots squared once
    void square_all_step() {
        for (int j = 0; j < m; j++)
            square_slot(hslot(j), hslot_o(j));
        hcur = 1 - hcur;
    }
    // C <- x^m + sum_j h[j] x^(m-1-j)  (reduced mod T)
    void horner_C() {
        int blocks, threads;
        CudaFFT::launch_dims(nwc, blocks, threads);
        k_horner_fused<<<blocks, threads>>>(dH[hcur], nw, m, dC, nwc);
        k_horner_tail<<<1, 1>>>(dC, r, s, m);
    }
    // dS0 <- x * y (full 2r-1-bit product; x,y device residues < r bits)
    void mul_full(const u64 *dx, const u64 *dy) {
        size_t ca = (size_t)((r + 31) >> 5);
        int blocks, threads;
        CudaFFT::launch_dims(F.n, blocks, threads);
        k_pack<<<blocks, threads>>>(dx, nw, dFA, ca, F.n);
        k_pack<<<blocks, threads>>>(dy, nw, dFB, ca, F.n);
        F.fwd(dFA);
        F.fwd(dFB);
        k_pointwise<<<blocks, threads>>>(dFA, dFB, F.n);
        F.inv(dFA);
        CudaFFT::launch_dims(nw2, blocks, threads);
        k_overlap_add<<<blocks, threads>>>(dFA, 2 * ca - 1, dS0, nw2);
    }
    // reduce dS0 (bit length 2r-1) mod T into dst (nw words)
    void reduce_into(u64 *dst) {
        int blocks, threads;
        size_t w1 = bits_to_words(r + s - 1);
        CudaFFT::launch_dims(w1, blocks, threads);
        k_fold_pass<<<blocks, threads>>>(dS0, nw2, dS1, w1, r, s, 2 * r - 1);
        CudaFFT::launch_dims(nw, blocks, threads);
        k_fold_pass<<<blocks, threads>>>(dS1, w1, dst, nw, r, s, r + s - 1);
    }
    // A <- A * C mod T
    void mul_A_by_C() {
        mul_full(dA, dC);
        reduce_into(dA);
    }
    void reset_A() {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_set_one<<<blocks, threads>>>(dA, nw);
    }

    // h-table init at block start kstart: Pascal-triangle build of the
    // elementary symmetric functions of (x^(2^0) .. x^(2^(m-1))), then
    // kstart Frobenius steps.  Mirrors factor.cpp's init_h exactly.
    void init_h_at(u64 kstart) {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        for (int j = 0; j < m; j++)
            k_set_zero<<<blocks, threads>>>(hslot(j), nw);
        for (int j = 1; j <= m; j++) {
            for (int i = j - 1; i >= 0; i--) {
                if (i > 0) {
                    square_slot(hslot(i - 1), dTmp);
                    CUCHK(cudaMemcpy(hslot(i - 1), dTmp, nw * sizeof(u64),
                                     cudaMemcpyDeviceToDevice));
                    k_mulx_xor<<<blocks, threads>>>(hslot(i), hslot(i - 1),
                                                    nw, r, s);
                } else {
                    k_xor_x<<<1, 1>>>(hslot(0));       // h[0] ^= x
                }
            }
        }
        for (u64 t = 0; t < kstart; t++)
            square_all_step();
    }

    void copy_h(u64 *dst_base, const u64 *src_base) {
        CUCHK(cudaMemcpy(dst_base, src_base, (size_t)m * nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));
    }
    void h_to_host(std::vector<u64> &out) {
        out.resize((size_t)m * nw);
        CUCHK(cudaMemcpy(out.data(), dH[hcur],
                         (size_t)m * nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void h_from_host(const std::vector<u64> &in) {
        CUCHK(cudaMemcpy(dH[hcur], in.data(),
                         (size_t)m * nw * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }
    void A_to_host(std::vector<u64> &out) {
        out.resize(nw);
        CUCHK(cudaMemcpy(out.data(), dA, nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void A_from_host(const std::vector<u64> &in) {
        CUCHK(cudaMemcpy(dA, in.data(), nw * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }
};

// ---------------------------------------------------------------------------
// CPU GCD thread pool (carried over from trinomial_scan.cu): jobs are
// (id, A words) against a shared_ptr modulus; verdicts retrieved by id
// and consumed by the caller strictly in interval order; stale jobs
// are dropped without blocking after a hit.

struct GcdRes {
    u64 gdeg = 0;
    std::vector<u64> g;      // always captured (up to r bits; a few MB)
    double secs = 0;
};

// gcd with full mask capture; an interval gcd is at most r bits and we
// always want it (the 4-Mbit default in gf2_gcd.h would drop deep-tail
// single-factor gcds that the shortcut in resolve_gcd needs verbatim)
static u64 g_keep_bits = ((u64)1 << 22);
static GpuMulHook *g_gm = nullptr;       // set when --gpu-gcd is active
static bool g_use_hybrid = false;
static bool g_hyb_xcheck = false;        // selftest: verify hybrid vs NTL
static u64 ts_gcd(const u64 *a, size_t aw, const u64 *b, size_t bw,
                  std::vector<u64> *gout) {
#ifdef HAVE_NTL
    if (g_use_hybrid) {
        u64 d = poly_gcd_hybrid(a, aw, b, bw, gout, g_keep_bits, g_gm);
        if (g_hyb_xcheck) {
            std::vector<u64> gref;
            u64 dr = poly_gcd_ntl(a, aw, b, bw, &gref, g_keep_bits);
            if (dr != d || (dr && gout && !gout->empty() && *gout != gref)) {
                fprintf(stderr, "HYBRID GCD MISMATCH: hyb %" PRIu64
                        " ntl %" PRIu64 " (aw %zu bw %zu) -- dumping\n",
                        d, dr, aw, bw);
                FILE *f = fopen("/tmp/hyb_mismatch.bin", "wb");
                if (f) {
                    u64 h[2] = {(u64)aw, (u64)bw};
                    fwrite(h, 8, 2, f);
                    fwrite(a, 8, aw, f);
                    fwrite(b, 8, bw, f);
                    fclose(f);
                }
                if (gout) *gout = gref;   // heal so the run continues
                d = dr;
            }
        }
        return d;
    }
    return poly_gcd_ntl(a, aw, b, bw, gout, g_keep_bits);
#else
    return poly_gcd_naive(a, aw, b, bw, gout);
#endif
}

static const char *gcd_backend() {
#ifdef HAVE_NTL
    if (!ntl_gf2x_backed())
        return "NTL (HalfGCD + CanZass) -- WARNING: NTL built WITHOUT "
               "gf2x; large-degree GCDs will be many times slower "
               "(rebuild NTL with NTL_GF2X_LIB=on)";
    if (g_use_hybrid)
        return g_gm ? "hybrid HGCD (GPU-offloaded mults) + NTL finish"
                    : "hybrid HGCD (CPU mults) + NTL finish";
    return "NTL gf2x-backed (HalfGCD + CanZass)";
#else
    return "naive fallback -- NO NTL: factor resolution will fail for "
           "multi-factor interval gcds; NOT fit for production";
#endif
}

#ifdef HAVE_NTL
// GPU multiplication service for the hybrid HGCD: packs both operands
// into 32-bit chunks, runs the Cantor FFT convolution, and returns the
// full product.  Owns its device buffers (no sharing with the scan
// state) and lazily builds one CudaFFT plan per size class; a mutex
// serializes hook users, and the legacy default CUDA stream serializes
// device work against the concurrently running scan.
struct TsGpuMul : GpuMulHook {
    CantorBasis *Bb = nullptr;
    std::map<int, CudaFFT *> plans;
    u64 *dA = nullptr, *dB = nullptr, *dP = nullptr;
    u64 *dM = nullptr, *dR = nullptr;    // batched-op scratch/accumulator
    size_t cap_n = 0;
    std::mutex mu;
    void init(CantorBasis *bb, int max_fm) {
        Bb = bb;
        cap_n = (size_t)1 << max_fm;
        CUCHK(cudaMalloc(&dA, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dB, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dM, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dR, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dP, (cap_n / 2 + 2) * sizeof(u64)));
    }
    void fini() {
        for (auto &kv : plans) { kv.second->fini(); delete kv.second; }
        plans.clear();
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dM) cudaFree(dM);
        if (dR) cudaFree(dR);
        if (dP) cudaFree(dP);
        dA = dB = dM = dR = dP = nullptr;
    }
    CudaFFT *plan_for(int fm) {
        auto it = plans.find(fm);
        if (it != plans.end()) return it->second;
        CudaFFT *F = new CudaFFT();
        F->init(fm, *Bb);
        plans[fm] = F;
        return F;
    }
    // pack a host operand and forward-transform it into dst
    void pack_fwd(CudaFFT *F, const u64 *w, size_t nw, u64 bits, u64 *dst) {
        size_t c = (size_t)((bits + 31) >> 5);
        int blocks, threads;
        CUCHK(cudaMemcpy(dP, w, nw * sizeof(u64), cudaMemcpyHostToDevice));
        CudaFFT::launch_dims(F->n, blocks, threads);
        k_pack<<<blocks, threads>>>(dP, nw, dst, c, F->n);
        F->fwd(dst);
    }
    bool mat2_apply(const Op m[4], const Op &a, const Op &b,
                    std::vector<u64> &oa, std::vector<u64> &ob) override {
        size_t ca = (size_t)((a.bits + 31) >> 5);
        size_t cb = (size_t)((b.bits + 31) >> 5);
        size_t mxc = 0;
        u64 rb[2] = {0, 0};                    // result bits per row
        for (int i = 0; i < 4; i++) {
            if (!m[i].bits) continue;
            size_t cx = (i & 1) ? cb : ca;
            if (!cx) continue;
            size_t nch = (size_t)((m[i].bits + 31) >> 5) + cx - 1;
            if (nch > mxc) mxc = nch;
            u64 pb = m[i].bits + ((i & 1) ? b.bits : a.bits) - 1;
            if (pb > rb[i >> 1]) rb[i >> 1] = pb;
        }
        if (!mxc) { oa.clear(); ob.clear(); return true; }
        int fm = 5;
        while (((size_t)1 << fm) < mxc + 1) fm++;
        if (((size_t)1 << fm) > cap_n) return false;
        std::lock_guard<std::mutex> lk(mu);
        CudaFFT *F = plan_for(fm);
        int blocks, threads;
        CudaFFT::launch_dims(F->n, blocks, threads);
        bool ua = (m[0].bits && a.bits) || (m[2].bits && a.bits);
        bool ub = (m[1].bits && b.bits) || (m[3].bits && b.bits);
        if (ua) pack_fwd(F, a.w, a.nw, a.bits, dA);
        if (ub) pack_fwd(F, b.w, b.nw, b.bits, dB);
        for (int row = 0; row < 2; row++) {
            std::vector<u64> &out = row ? ob : oa;
            const Op &e0 = m[2 * row], &e1 = m[2 * row + 1];
            bool h0 = e0.bits && a.bits, h1 = e1.bits && b.bits;
            if (!h0 && !h1) { out.clear(); continue; }
            if (h0) {
                pack_fwd(F, e0.w, e0.nw, e0.bits, dM);
                k_pointwise<<<blocks, threads>>>(dM, dA, F->n);
            }
            if (h1) {
                u64 *t = h0 ? dR : dM;
                pack_fwd(F, e1.w, e1.nw, e1.bits, t);
                k_pointwise<<<blocks, threads>>>(t, dB, F->n);
                if (h0) k_xor_buf<<<blocks, threads>>>(dM, dR, F->n);
            }
            F->inv(dM);
            size_t ow = bits_to_words(rb[row]);
            int b2, t2;
            CudaFFT::launch_dims(ow, b2, t2);
            k_overlap_add<<<b2, t2>>>(dM, mxc, dP, ow);
            out.resize(ow);
            CUCHK(cudaMemcpy(out.data(), dP, ow * sizeof(u64),
                             cudaMemcpyDeviceToHost));
        }
        return true;
    }
    bool mul(const u64 *a, size_t aw, u64 abits,
             const u64 *b, size_t bw, u64 bbits,
             std::vector<u64> &out) override {
        size_t ca = (size_t)((abits + 31) >> 5);
        size_t cb = (size_t)((bbits + 31) >> 5);
        size_t nch = ca + cb - 1;
        int fm = 5;
        while (((size_t)1 << fm) < nch + 1) fm++;
        if (((size_t)1 << fm) > cap_n) return false;   // too big: CPU
        size_t ow = bits_to_words(abits + bbits - 1);
        std::lock_guard<std::mutex> lk(mu);
        CudaFFT *F = plan_for(fm);
        int blocks, threads;
        CUCHK(cudaMemcpy(dP, a, aw * sizeof(u64), cudaMemcpyHostToDevice));
        CudaFFT::launch_dims(F->n, blocks, threads);
        k_pack<<<blocks, threads>>>(dP, aw, dA, ca, F->n);
        CUCHK(cudaMemcpy(dP, b, bw * sizeof(u64), cudaMemcpyHostToDevice));
        k_pack<<<blocks, threads>>>(dP, bw, dB, cb, F->n);
        F->fwd(dA);
        F->fwd(dB);
        k_pointwise<<<blocks, threads>>>(dA, dB, F->n);
        F->inv(dA);
        CudaFFT::launch_dims(ow, blocks, threads);
        k_overlap_add<<<blocks, threads>>>(dA, nch, dP, ow);
        out.resize(ow);
        CUCHK(cudaMemcpy(out.data(), dP, ow * sizeof(u64),
                         cudaMemcpyDeviceToHost));
        return true;
    }
};
static TsGpuMul g_ts_gm;
#endif // HAVE_NTL

class GcdPool {
    std::mutex mu;
    std::condition_variable cv_job, cv_res;
    struct Job {
        u64 id;
        std::vector<u64> acc;
        std::shared_ptr<const std::vector<u64>> mod;
    };
    std::deque<Job> jobs;
    std::map<u64, GcdRes> results;
    std::set<u64> outstanding, discard;
    bool stopf = false;
    std::vector<std::thread> ths;

    void worker() {
        for (;;) {
            std::unique_lock<std::mutex> lk(mu);
            cv_job.wait(lk, [&] { return stopf || !jobs.empty(); });
            if (jobs.empty()) { if (stopf) return; continue; }
            Job j = std::move(jobs.front());
            jobs.pop_front();
            lk.unlock();
            GcdRes rres;
            double t0 = trin_now_s();
            rres.gdeg = ts_gcd(j.acc.data(), j.acc.size(), j.mod->data(),
                               j.mod->size(), &rres.g);
            rres.secs = trin_now_s() - t0;
            lk.lock();
            outstanding.erase(j.id);
            if (discard.erase(j.id) == 0) results[j.id] = std::move(rres);
            cv_res.notify_all();
        }
    }

public:
    void start(int n) {
        for (int i = 0; i < n; i++) ths.emplace_back([this] { worker(); });
    }
    void submit(u64 id, std::vector<u64> acc,
                std::shared_ptr<const std::vector<u64>> mod) {
        std::lock_guard<std::mutex> lk(mu);
        outstanding.insert(id);
        jobs.push_back(Job{id, std::move(acc), std::move(mod)});
        cv_job.notify_one();
    }
    bool try_get(u64 id, GcdRes &out) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = results.find(id);
        if (it == results.end()) return false;
        out = std::move(it->second);
        results.erase(it);
        return true;
    }
    GcdRes wait_get(u64 id) {
        std::unique_lock<std::mutex> lk(mu);
        cv_res.wait(lk, [&] { return results.count(id) > 0; });
        GcdRes rres = std::move(results[id]);
        results.erase(id);
        return rres;
    }
    void forget_all_pending() {
        std::lock_guard<std::mutex> lk(mu);
        for (u64 id : outstanding) discard.insert(id);
        results.clear();
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stopf = true;
            jobs.clear();          // prompt exit: drop queued (not running)
        }
        cv_job.notify_all();
        for (auto &t : ths) t.join();
        ths.clear();
    }
    // signal-exit path: a worker may be inside a multi-minute GCD; any
    // checkpoint is already on disk, so don't wait -- detach and let
    // the caller _Exit before static teardown can race the workers.
    void stop_nowait() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stopf = true;
            jobs.clear();
        }
        cv_job.notify_all();
        for (auto &t : ths) t.detach();
        ths.clear();
    }
};

// ---------------------------------------------------------------------------
// checkpointing: header + CRC-protected sections, 3-generation rotation

static u32 crc32_tab[256];
static void crc32_init() {
    for (u32 i = 0; i < 256; i++) {
        u32 c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc32_tab[i] = c;
    }
}
static u32 crc32_of(const void *buf, size_t len, u32 seed = 0) {
    const unsigned char *p = (const unsigned char *)buf;
    u32 c = seed ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
        c = crc32_tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

#define TS_CKPT_MAGIC 0x31434653544ULL   // "TSFC1"-ish
struct CkptHdr {
    u64 magic, version;
    u64 r, s, skip;
    u64 m;
    u64 k_iv;          // start degree of the interval being processed
    u64 blk;           // blocks of that interval already folded into A
    u64 q;             // block count of that interval
    double f0;         // growth state (factor's f0)
    u64 has_A;         // 1 iff A section present (blk > 0)
    u64 nw;
    u32 crc_h, crc_a;  // section CRCs
    u32 crc_hdr;       // of all fields above
};

static bool write_file_atomic(const std::string &path, const void *hdr,
                              size_t hdrsz, const std::vector<u64> &h,
                              const std::vector<u64> &a) {
    std::string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(hdr, 1, hdrsz, f) == hdrsz;
    if (ok && !h.empty()) ok = fwrite(h.data(), 8, h.size(), f) == h.size();
    if (ok && !a.empty()) ok = fwrite(a.data(), 8, a.size(), f) == a.size();
    if (ok) ok = fflush(f) == 0 && fsync(fileno(f)) == 0;
    fclose(f);
    if (!ok) { remove(tmp.c_str()); return false; }
    return rename(tmp.c_str(), path.c_str()) == 0;
}

static void rotate_ckpts(const std::string &base) {
    std::string c0 = base, c1 = base + ".1", c2 = base + ".2";
    remove(c2.c_str());
    rename(c1.c_str(), c2.c_str());
    rename(c0.c_str(), c1.c_str());
}

static bool save_ckpt(const std::string &base, const CkptHdr &H,
                      const std::vector<u64> &h, const std::vector<u64> &a,
                      int verbose) {
    CkptHdr W = H;
    W.crc_h = h.empty() ? 0 : crc32_of(h.data(), h.size() * 8);
    W.crc_a = a.empty() ? 0 : crc32_of(a.data(), a.size() * 8);
    W.crc_hdr = 0;
    W.crc_hdr = crc32_of(&W, sizeof(W) - sizeof(u32));
    rotate_ckpts(base);
    bool ok = write_file_atomic(base, &W, sizeof(W), h, a);
    if (verbose)
        fprintf(stderr, "  [checkpoint %s: s=%" PRIu64 " k_iv=%" PRIu64
                " blk=%" PRIu64 "/%" PRIu64 " %s]\n",
                ok ? "written" : "FAILED", H.s, H.k_iv, H.blk, H.q,
                base.c_str());
    return ok;
}

static bool load_ckpt_one(const std::string &path, CkptHdr &H,
                          std::vector<u64> &h, std::vector<u64> &a) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&H, 1, sizeof(H), f) == sizeof(H);
    if (ok) {
        CkptHdr T = H;
        T.crc_hdr = 0;
        ok = H.magic == TS_CKPT_MAGIC &&
             crc32_of(&T, sizeof(T) - sizeof(u32)) == H.crc_hdr;
    }
    if (ok) {
        h.resize((size_t)H.m * H.nw);
        ok = fread(h.data(), 8, h.size(), f) == h.size() &&
             crc32_of(h.data(), h.size() * 8) == H.crc_h;
    }
    if (ok && H.has_A) {
        a.resize((size_t)H.nw);
        ok = fread(a.data(), 8, a.size(), f) == a.size() &&
             crc32_of(a.data(), a.size() * 8) == H.crc_a;
    } else if (ok) {
        a.clear();
    }
    fclose(f);
    return ok;
}

static bool load_ckpt(const std::string &base, CkptHdr &H,
                      std::vector<u64> &h, std::vector<u64> &a) {
    const char *sfx[3] = {"", ".1", ".2"};
    for (int i = 0; i < 3; i++) {
        std::string p = base + sfx[i];
        if (load_ckpt_one(p, H, h, a)) {
            if (i) fprintf(stderr, "note: fell back to checkpoint %s\n",
                           p.c_str());
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// interval schedule, replicating factor.cpp -q/-f semantics exactly:
//   f == 0: q = q0 always;  f == 1: q += q0;  f > 1: q = round(f0*q0),
//   f0 *= f.  Then clamp so the interval does not overshoot rhigh.
struct QSched {
    long q0 = 15;
    double f = 1.0, f0 = 1.0;
    long q = 0;
    void grow() {
        if (f == 0.0) q = q0;
        else if (f > 1.0) { q = (long)(f0 * (double)q0 + 0.5); f0 = f * f0; }
        else q = q + q0;
    }
};

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

// ---------------------------------------------------------------------------
// scan parameters and outcome

struct SParams {
    u64 skip = 0;              // no factors of degree <= skip (REQUIRED)
    u64 maxd = 0;              // 0 = no cap
    long q0 = 0;               // 0 = auto (ceil(600/m))
    double f = 1.0;
    long zq = 0;               // -z: emit "u" when q would exceed
    long Zq = 0;               // -Z: cap q, keep scanning
    u64 canzass_max = 0;       // 0 = auto (2*r^(2/3))
    int verbose = 0;
    int pend_max = 4;
    std::string ckpt_base;     // empty = no checkpointing
    double ckpt_secs = 15 * 60.0;
    long die_after_blocks = -1;   // test hook: simulate SIGTERM
};

struct ScanOut {
    enum Kind { FOUND, GAVE_UP, PRIMITIVE, RANGE, INTERRUPTED } kind;
    u64 d = 0;
    std::string hex;           // for FOUND
    u64 rlo = 0, rhi = 0;      // for RANGE
    u64 degrees_scanned = 0;
    double wall = 0;
};

// resolve a captured interval gcd g: least factor degree + numerically
// least mask at that degree.  If deg(g) < 2*ka every factor exceeds
// half of deg(g), so g is a single irreducible (trinomials are
// squarefree): skip CanZass entirely.
static bool resolve_gcd(const std::vector<u64> &g, u64 gdeg, u64 ka,
                        u64 &d, std::vector<u64> &mask) {
    if (gdeg > 0 && gdeg < 2 * ka) {
        d = gdeg;
        mask = g;
        size_t need = bits_to_words(gdeg + 1);
        if (mask.size() > need) mask.resize(need);
        return true;
    }
#ifdef HAVE_NTL
    return factor_min_degree_ntl(g.data(), g.size(), d, mask);
#else
    // no-NTL fallback: least degree by squaring mod g; mask by
    // enumeration (tiny degrees only -- selftest territory)
    int64_t dg = poly_deg_v(g);
    if (dg <= 0) return false;
    size_t gw = (size_t)(dg >> 6) + 1;
    std::vector<u64> h(gw, 0);
    h[0] = 2;                                    // x mod g
    for (u64 k = 1; k <= (u64)dg; k++) {
        std::vector<u64> sq(mul_out_words((u64)dg, (u64)dg));
        gf2x_naive_mul(h.data(), (u64)dg, h.data(), (u64)dg, sq.data());
        // reduce mod g
        int64_t da = poly_deg_v(sq);
        while (da >= dg) {
            poly_xor_shift(sq, g, dg, (u64)(da - dg));
            da = poly_deg_v(sq);
        }
        sq.resize(gw);
        h = sq;
        std::vector<u64> hx = h;
        hx[0] ^= 2ULL;                           // h + x
        std::vector<u64> gg;
        u64 gd = poly_gcd_naive(hx.data(), hx.size(), g.data(), g.size(),
                                &gg);
        if (gd > 0) {
            d = k;
            return edf_least_enum(gg, k, mask);
        }
    }
    return false;
#endif
}

static std::string mask_hex(const std::vector<u64> &mask) {
    int64_t d = poly_deg_v(mask);
    std::string out;
    if (d < 0) return out;
    int c = 0;
    static const char *t = "0123456789abcdef";
    for (int64_t i = d; i >= 0; i--) {
        c = 2 * c + (int)((mask[i >> 6] >> (i & 63)) & 1);
        if ((i & 3) == 0) { out.push_back(t[c]); c = 0; }
    }
    return out;
}

// ---------------------------------------------------------------------------
// the per-s scan

struct ResumeState {
    bool valid = false;
    CkptHdr H;
    std::vector<u64> h, a;
};

// one block: C from h, A *= C, advance h by m degrees
static void do_block(TsGPU &G, int verbose) {
    if (verbose >= 2) {
        double t0 = trin_now_s();
        G.horner_C();
        CUCHK(cudaDeviceSynchronize());
        double t1 = trin_now_s();
        G.mul_A_by_C();
        CUCHK(cudaDeviceSynchronize());
        double t2 = trin_now_s();
        for (int t = 0; t < G.m; t++) G.square_all_step();
        CUCHK(cudaDeviceSynchronize());
        double t3 = trin_now_s();
        G.t_horner += t1 - t0;
        G.t_mul += t2 - t1;
        G.t_sq += t3 - t2;
    } else {
        G.horner_C();
        G.mul_A_by_C();
        for (int t = 0; t < G.m; t++) G.square_all_step();
    }
}

// fineDDF-style localization: a factor is known to lie in
// [ka, ka + q*m - 1] but the interval gcd was too large to factor
// comfortably (deg >= canzass_max).  Rebuild h at ka and bisect,
// exactly like factor.cpp's fineDDF mode.  Returns the final captured
// gcd (all of whose factors lie in the final sub-interval), or sets
// range_only.
static bool localize(TsGPU &G, GcdPool &pool, const SParams &P, u64 /*r*/,
                     std::shared_ptr<const std::vector<u64>> T,
                     u64 ka, long q, u64 &gseq,
                     std::vector<u64> &g_out, u64 &gdeg_out,
                     u64 &ka_out, u64 &kb_out) {
    if (P.verbose)
        printf("Entering fine DDF mode: localize in %" PRIu64 "..%" PRIu64
               "\n", ka, ka + (u64)q * G.m - 1);
    double t0 = trin_now_s();
    G.init_h_at(ka);
    CUCHK(cudaDeviceSynchronize());
    if (P.verbose)
        printf("Re-exponentiation took %f\n", trin_now_s() - t0);
    G.need_hold();
    u64 kcur = ka;
    long qcur = q;
    std::vector<u64> hostA;
    for (;;) {
        long qfirst = (qcur == 1) ? 1 : qcur - qcur / 2;
        G.copy_h(G.dHold, G.dH[G.hcur]);
        G.reset_A();
        for (long b = 0; b < qfirst; b++) do_block(G, P.verbose);
        CUCHK(cudaDeviceSynchronize());
        G.A_to_host(hostA);
        u64 id = gseq++;
        pool.submit(id, hostA, T);
        GcdRes R = pool.wait_get(id);
        u64 kb = kcur + (u64)qfirst * G.m - 1;
        if (P.verbose)
            printf("   fineDDF %" PRIu64 "..%" PRIu64 ": gcd took %f, "
                   "deg %" PRIu64 "\n", kcur, kb, R.secs, R.gdeg);
        if (R.gdeg > 0) {
            bool captured = !R.g.empty();
            if (captured &&
                (R.gdeg < P.canzass_max || R.gdeg < 2 * kcur || qfirst == 1)) {
                g_out = std::move(R.g);
                gdeg_out = R.gdeg;
                ka_out = kcur;
                kb_out = kb;
                return true;
            }
            if (qfirst == 1) {          // cannot refine further, uncaptured
                ka_out = kcur;
                kb_out = kb;
                return false;
            }
            G.copy_h(G.dH[G.hcur], G.dHold);   // back to state at kcur
            qcur = qfirst;
        } else {
            if (qcur == 1) {
                fprintf(stderr, "ERROR: fineDDF lost the factor at %" PRIu64
                        "\n", kcur);
                ka_out = kcur;
                kb_out = kb;
                return false;
            }
            kcur += (u64)qfirst * G.m;         // h is already there
            qcur = qcur / 2;
        }
    }
}

static ScanOut scan_one_s(TsGPU &G, GcdPool &pool, const SParams &P,
                          u64 r, u64 s, u64 &gseq, ResumeState *resume) {
    ScanOut out;
    double s_t0 = trin_now_s();
    G.set_s(s);
    G.t_sq = G.t_horner = G.t_mul = 0;

    auto T = std::make_shared<std::vector<u64>>(bits_to_words(r + 1), 0);
    (*T)[0] |= 1;
    (*T)[s >> 6] |= 1ULL << (s & 63);
    (*T)[r >> 6] |= 1ULL << (r & 63);

    // Swan bound (flip-invariant; computed on the original s)
    bool swan = (((r & 7) == 1) || ((r & 7) == 7)) &&
                ((((s & 1) == 0) && (((2 * r) % s) != 0)) ||
                 (((s & 1) == 1) && (((2 * r) % (r - s)) != 0)));
    u64 rhigh = swan ? (r / 3) : (r / 2);
    bool maxd_limited = false;
    if (P.maxd && P.maxd < rhigh) { rhigh = P.maxd; maxd_limited = true; }
    if (P.verbose && swan && !maxd_limited)
        printf("By Swan's theorem search at most to degree %" PRIu64 "\n",
               rhigh);
    if (P.verbose && maxd_limited)
        printf("Maximal degree checked reduced to %" PRIu64 "\n", rhigh);

    QSched Q;
    Q.q0 = P.q0;
    Q.f = P.f;
    u64 k;                     // current interval start
    long blk0 = 0;             // blocks of it already folded into A

    bool resumed = false;
    if (resume && resume->valid && resume->H.s == s) {
        const CkptHdr &H = resume->H;
        k = H.k_iv;
        Q.q = (long)H.q;
        Q.f0 = H.f0;
        blk0 = (long)H.blk;
        G.h_from_host(resume->h);
        if (H.has_A) G.A_from_host(resume->a);
        else G.reset_A();
        resumed = true;
        resume->valid = false;
        if (P.verbose)
            printf("Resumed s=%" PRIu64 " at interval %" PRIu64
                   " block %ld/%ld\n", s, k, blk0, Q.q);
    } else {
        k = P.skip + 1;
        double t0 = trin_now_s();
        G.init_h_at(k);
        CUCHK(cudaDeviceSynchronize());
        if (P.verbose)
            printf("Exponentiation took %f\n", trin_now_s() - t0);
        G.reset_A();
        Q.grow();
        if (P.Zq && Q.q > P.Zq) Q.q = P.Zq;
    }

    struct Pend { u64 seq, ka, kb; long q; };
    std::deque<Pend> pending;
    u64 vf = k;                // verified frontier: all degrees < vf clear

    // checkpoint candidate: h at the start of an interval whose
    // predecessors were all verified at snapshot time
    struct Cand { bool valid = false; u64 k_iv = 0; long q = 0; double f0 = 1; 
                  std::vector<u64> h; } cand;
    double last_ckpt = trin_now_s();
    long blocks_done_hook = 0;

    std::vector<u64> hostA;

    auto write_ckpt = [&](bool exiting) -> void {
        if (P.ckpt_base.empty()) return;
        CkptHdr H;
        memset(&H, 0, sizeof(H));
        H.magic = TS_CKPT_MAGIC; H.version = 1;
        H.r = r; H.s = s; H.skip = P.skip; H.m = (u64)G.m; H.nw = (u64)G.nw;
        std::vector<u64> hh, aa;
        if (vf == k) {           // current interval fully verified behind us
            H.k_iv = k; H.blk = 0; H.q = (u64)Q.q; H.f0 = Q.f0;
            // mid-interval flavor if we are inside the interval
            // (caller ensures we are between blocks)
            G.h_to_host(hh);
            long b = blocks_done_hook;
            if (b > 0) { H.blk = (u64)b; H.has_A = 1; G.A_to_host(aa); }
            save_ckpt(P.ckpt_base, H, hh, aa, P.verbose);
        } else if (cand.valid) {
            H.k_iv = cand.k_iv; H.blk = 0; H.q = (u64)cand.q; H.f0 = cand.f0;
            save_ckpt(P.ckpt_base, H, cand.h, aa, P.verbose);
        } else if (exiting && P.verbose) {
            fprintf(stderr, "  [no verified state to checkpoint; s=%" PRIu64
                    " will restart from scratch on resume]\n", s);
        }
        last_ckpt = trin_now_s();
    };

    // returns 1 if factor resolved (out filled), 0 for clear verdicts
    auto consume_verdict = [&](const Pend &pv, GcdRes &R) -> int {
        if (P.verbose)
            printf("   gcd %" PRIu64 "..%" PRIu64 " took %f%s\n",
                   pv.ka, pv.kb, R.secs, R.gdeg ? "  ** HIT **" : "");
        if (R.gdeg == 0) {
            vf = pv.kb + 1;
            return 0;
        }
        // hit: stop speculation, resolve
        pool.forget_all_pending();
        pending.clear();
        std::vector<u64> g;
        u64 gdeg = R.gdeg, rka = pv.ka, rkb = pv.kb;
        bool have_g = false;
        if (!R.g.empty() && (R.gdeg < P.canzass_max || R.gdeg < 2 * pv.ka)) {
            g = std::move(R.g);
            have_g = true;
        } else {
            have_g = localize(G, pool, P, r, T, pv.ka, pv.q, gseq,
                              g, gdeg, rka, rkb);
        }
        if (!have_g) {
            out.kind = ScanOut::RANGE;
            out.rlo = rka;
            out.rhi = rkb;
            return 1;
        }
        u64 d = 0;
        std::vector<u64> mask;
        double t0 = trin_now_s();
        bool ok = resolve_gcd(g, gdeg, rka, d, mask);
        if (!ok) {
            fprintf(stderr, "ERROR: could not factor interval gcd (deg %"
                    PRIu64 ") for s=%" PRIu64 "\n", gdeg, s);
            out.kind = ScanOut::RANGE;
            out.rlo = rka;
            out.rhi = rkb;
            return 1;
        }
        if (P.verbose)
            printf("CanZass took %f, total degree %" PRIu64
                   ", small degree %" PRIu64 "\n",
                   trin_now_s() - t0, gdeg, d);
        if (!(d >= rka && d <= rkb))
            fprintf(stderr, "WARNING: s=%" PRIu64 " least degree %" PRIu64
                    " outside hit interval [%" PRIu64 ",%" PRIu64
                    "] -- input not a genuine survivor?\n", s, d, rka, rkb);
        out.kind = ScanOut::FOUND;
        out.d = d;
        out.hex = mask_hex(mask);
        return 1;
    };

    // main interval loop
    for (;;) {
        if (k > rhigh) break;
        long q = Q.q;
        if (P.verbose) {
            printf("Current q %ld\n", q);
            fflush(stdout);
        }
        u64 k2 = k + (u64)q * G.m - 1;
        if (k2 > rhigh) {                       // clamp to cover rhigh
            q = (long)((rhigh - k + G.m) / G.m);
            k2 = k + (u64)q * G.m - 1;
            if (P.verbose)
                printf("Reducing q to %ld\n", q);
            Q.q = q;
        }
        if (P.zq && q > P.zq) {                 // -z: give up
            while (!pending.empty()) {          // a pending gcd may hit
                Pend pv = pending.front();
                pending.pop_front();
                GcdRes R = pool.wait_get(pv.seq);
                if (consume_verdict(pv, R)) goto done;
            }
            out.kind = ScanOut::GAVE_UP;
            if (P.verbose)
                printf("q > %ld so skipping further test\n", P.zq);
            goto done;
        }
        if (P.verbose) {
            printf("Interval %" PRIu64 "..%" PRIu64 ":\n", k, k2);
            fflush(stdout);
        }
        if (!resumed && vf == k) {              // snapshot for checkpointing
            cand.valid = true;
            cand.k_iv = k; cand.q = q; cand.f0 = Q.f0;
            G.h_to_host(cand.h);
        }
        double iv_t0 = trin_now_s(), iv_poll = 0;
        for (long b = blk0; b < q; b++) {
            blocks_done_hook = b;
            // poll verdicts between blocks
            double p0 = trin_now_s();
            while (!pending.empty()) {
                GcdRes R;
                if (!pool.try_get(pending.front().seq, R)) break;
                Pend pv = pending.front();
                pending.pop_front();
                if (consume_verdict(pv, R)) goto done;
            }
            iv_poll += trin_now_s() - p0;
            if (g_stop || (P.die_after_blocks >= 0 &&
                           --((SParams &)P).die_after_blocks < 0)) {
                CUCHK(cudaDeviceSynchronize());
                write_ckpt(true);
                out.kind = ScanOut::INTERRUPTED;
                goto done;
            }
            if (!P.ckpt_base.empty() &&
                trin_now_s() - last_ckpt > P.ckpt_secs) {
                CUCHK(cudaDeviceSynchronize());
                write_ckpt(false);
            }
            do_block(G, P.verbose);
        }
        blocks_done_hook = 0;
        CUCHK(cudaDeviceSynchronize());
        double iv_wall = trin_now_s() - iv_t0 - iv_poll;
        out.degrees_scanned += (u64)(q - blk0) * G.m;
        if (P.verbose) {
            double per = iv_wall / (double)((q - blk0) * G.m);
            printf("   squares/products took %.1f (wct %.1f), per term "
                   "%.3f (wct %.3f)\n", iv_wall, iv_wall, per, per);
            if (P.verbose >= 2)
                printf("   [components cumulative: sq %.1f  horner %.1f  mul %.1f]\n",
                       G.t_sq, G.t_horner, G.t_mul);
            fflush(stdout);
        }
        G.A_to_host(hostA);
        u64 id = gseq++;
        pool.submit(id, hostA, T);
        pending.push_back(Pend{id, k, k2, q});
        G.reset_A();
        resumed = false;
        blk0 = 0;
        k = k2 + 1;
        if ((int)pending.size() >= P.pend_max) {   // bound speculation
            Pend pv = pending.front();
            pending.pop_front();
            GcdRes R = pool.wait_get(pv.seq);
            if (consume_verdict(pv, R)) goto done;
        }
        Q.grow();
        if (P.Zq && Q.q > P.Zq) Q.q = P.Zq;
    }
    // exhausted the degree range: drain all pending verdicts
    while (!pending.empty()) {
        Pend pv = pending.front();
        pending.pop_front();
        GcdRes R = pool.wait_get(pv.seq);
        if (consume_verdict(pv, R)) goto done;
    }
    out.kind = maxd_limited ? ScanOut::GAVE_UP : ScanOut::PRIMITIVE;

done:
    if (out.kind != ScanOut::INTERRUPTED) pool.forget_all_pending();
    out.wall = trin_now_s() - s_t0;
    return out;
}

// ---------------------------------------------------------------------------
// CPU oracle (per-degree naive scan) for the selftest

struct OracleOut {
    bool survivor = false, found = false;
    u64 d = 0, gdeg = 0;
    std::vector<u64> g, mask;
    bool mask_ok = false;
};

static OracleOut oracle_scan(u64 r, u64 s, u64 skip, u64 maxd) {
    OracleOut o;
    std::vector<u64> T(bits_to_words(r + 1), 0);
    T[0] |= 1;
    T[s >> 6] |= 1ULL << (s & 63);
    T[r >> 6] |= 1ULL << (r & 63);
    std::vector<u64> Fh(bits_to_words(r), 0);
    Fh[0] = 2;
    for (u64 kk = 1; kk <= maxd; kk++) {
        cpu_square_mod(Fh, r, s);
        std::vector<u64> Fx = Fh;
        Fx[0] ^= 2ULL;
        std::vector<u64> g;
        u64 gd = poly_gcd_naive(Fx.data(), Fx.size(), T.data(), T.size(), &g);
        if (gd > 0) {
            if (kk <= skip) return o;                  // not a survivor
            o.survivor = o.found = true;
            o.d = kk;
            o.gdeg = gd;
            o.g = g;
            if (gd == kk) { o.mask = g; o.mask_ok = true; }
#ifdef HAVE_NTL
            else o.mask_ok = edf_least_ntl(g.data(), g.size(), kk, o.mask);
#else
            else if (kk <= 22) o.mask_ok = edf_least_enum(g, kk, o.mask);
#endif
            return o;
        }
    }
    o.survivor = true;                                 // "u"
    return o;
}

static u64 st_rng = 0x243F6A8885A308D3ULL;
static u64 st_rnd() {
    u64 x = st_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return st_rng = x;
}
static std::vector<u64> st_res(u64 r) {                // random residue < r
    std::vector<u64> v(bits_to_words(r));
    for (auto &w : v) w = st_rnd();
    unsigned tb = (unsigned)(r & 63);
    if (tb) v.back() &= (1ULL << tb) - 1;
    return v;
}

// host reference for k_mulx_xor
static void host_mulx_xor(std::vector<u64> &dst, const std::vector<u64> &src,
                          u64 r, u64 s) {
    std::vector<u64> t(src.size() + 1, 0);
    for (size_t i = 0; i < src.size(); i++) {
        t[i] ^= src[i] << 1;
        t[i + 1] ^= src[i] >> 63;
    }
    if ((t[r >> 6] >> (r & 63)) & 1) {
        t[r >> 6] ^= 1ULL << (r & 63);
        t[s >> 6] ^= 1ULL << (s & 63);
        t[0] ^= 1ULL;
    }
    for (size_t i = 0; i < dst.size(); i++) dst[i] ^= t[i];
}

static int selftest(int gcd_threads) {
    int bad = 0;
    printf("tsfactor selftest\n");
    printf("GCD backend: %s\n", gcd_backend());

    // (1) squaring: fused vs legacy vs host, many (r, s) incl. edges
    {
        bool ok = true;
        struct RS { u64 r, s; };
        std::vector<RS> cases;
        u64 rs[] = {4423, 4159, 4097, 4229, 9689};
        for (u64 rr : rs) {
            u64 svals[] = {1, 2, 3, 63, 64, 65, 2 * 64 + 1, rr / 3,
                           (rr - 1) / 2 - 1, (rr - 1) / 2};
            for (u64 sv : svals)
                if (sv >= 1 && 2 * sv < rr) cases.push_back({rr, sv});
        }
        for (auto &cs : cases) {
            TsGPU G;
            G.init(cs.r);
            G.set_m(2);
            G.set_s(cs.s);
            for (int t = 0; t < 3 && ok; t++) {
                std::vector<u64> a = st_res(cs.r), gl(G.nw), gf(G.nw), hr = a;
                CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8,
                                 cudaMemcpyHostToDevice));
                G.legacy_sq = true;
                G.square_slot(G.dA, G.dTmp);
                CUCHK(cudaMemcpy(gl.data(), G.dTmp, G.nw * 8,
                                 cudaMemcpyDeviceToHost));
                G.legacy_sq = false;
                G.square_slot(G.dA, G.dTmp);
                CUCHK(cudaMemcpy(gf.data(), G.dTmp, G.nw * 8,
                                 cudaMemcpyDeviceToHost));
                cpu_square_mod(hr, cs.r, cs.s);
                if (gl != hr || gf != hr) {
                    printf("  squaring mismatch r=%" PRIu64 " s=%" PRIu64
                           " (legacy %s fused %s)\n", cs.r, cs.s,
                           gl == hr ? "ok" : "BAD", gf == hr ? "ok" : "BAD");
                    ok = false;
                }
            }
            G.fini();
        }
        printf("squaring: fused vs legacy vs host (%zu cases)      %s\n",
               cases.size(), ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }

    // (2) mulx kernel vs host
    {
        bool ok = true;
        u64 rr = 4423;
        u64 svals[] = {1, 5, 64, 2211};
        for (u64 sv : svals) {
            TsGPU G;
            G.init(rr);
            G.set_m(2);
            G.set_s(sv);
            std::vector<u64> a = st_res(rr), d0 = st_res(rr), dg(G.nw),
                             dh = d0;
            CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8,
                             cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.dTmp, d0.data(), G.nw * 8,
                             cudaMemcpyHostToDevice));
            int blocks, threads;
            CudaFFT::launch_dims(G.nw, blocks, threads);
            k_mulx_xor<<<blocks, threads>>>(G.dTmp, G.dA, G.nw, rr, sv);
            CUCHK(cudaMemcpy(dg.data(), G.dTmp, G.nw * 8,
                             cudaMemcpyDeviceToHost));
            host_mulx_xor(dh, a, rr, sv);
            if (dg != dh) { ok = false; printf("  mulx mismatch s=%" PRIu64
                                              "\n", sv); }
            G.fini();
        }
        printf("mul-by-x kernel vs host                            %s\n",
               ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }

    // (3) init_h + Horner vs direct product of (x + x^(2^(k+i))) mod T
    {
        bool ok = true;
        u64 rr = 4423, sv = 130;
        int ms[] = {2, 3, 5, 8};
        u64 ks[] = {1, 5, 17};
        for (int mi = 0; mi < 4 && ok; mi++)
            for (int ki = 0; ki < 3 && ok; ki++) {
                int m = ms[mi];
                u64 kst = ks[ki];
                TsGPU G;
                G.init(rr);
                G.set_m(m);
                G.set_s(sv);
                G.init_h_at(kst);
                G.horner_C();
                CUCHK(cudaDeviceSynchronize());
                std::vector<u64> Cg(G.nw);
                CUCHK(cudaMemcpy(Cg.data(), G.dC, G.nw * 8,
                                 cudaMemcpyDeviceToHost));
                // host: y_i = x^(2^(kst+i)) mod T; C = prod (x + y_i) mod T
                std::vector<u64> y(bits_to_words(rr), 0);
                y[0] = 2;
                for (u64 t = 0; t < kst; t++) cpu_square_mod(y, rr, sv);
                std::vector<u64> Ch(bits_to_words(rr), 0);
                Ch[0] = 1;
                for (int i = 0; i < m; i++) {
                    std::vector<u64> yx = y;
                    yx[0] ^= 2ULL;                      // x + y_i
                    std::vector<u64> pr(mul_out_words(rr, rr));
                    gf2x_naive_mul(Ch.data(), rr, yx.data(), rr, pr.data());
                    cpu_mod_trinomial(pr, 2 * rr - 1, rr, sv);
                    for (size_t w = 0; w < Ch.size(); w++) Ch[w] = pr[w];
                    cpu_square_mod(y, rr, sv);
                }
                if (Cg != Ch) {
                    printf("  init_h/horner mismatch m=%d kst=%" PRIu64 "\n",
                           m, kst);
                    ok = false;
                }
                G.fini();
            }
        printf("init_h + fused Horner vs host block product        %s\n",
               ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }

    // (3b) FFT variant equivalence: legacy vs fused configurations,
    //      forward byte-equality and inverse roundtrip, at m = 12
    {
        bool okk = true;
        CantorBasis bbf;
        bbf.build();
        CudaFFT Ff;
        Ff.init(12, bbf);
        size_t fn = Ff.n;
        u64 *dbuf;
        CUCHK(cudaMalloc(&dbuf, fn * 8));
        std::vector<u64> hin(fn), href(fn), hout(fn);
        u64 st = 0xabcdef12ULL;
        for (auto &w : hin) {
            st += 0x9E3779B97F4A7C15ULL;
            u64 z = st;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            w = z ^ (z >> 31);
        }
        bool sv_f = gf2c_use_fused;
        int sv_t = gf2c_taylor_lv, sv_b = gf2c_bfly_lv;
        u64 sv_c = gf2c_fuse_max_ch;
        gf2c_use_fused = false;
        CUCHK(cudaMemcpy(dbuf, hin.data(), fn * 8, cudaMemcpyHostToDevice));
        Ff.fwd(dbuf);
        CUCHK(cudaMemcpy(href.data(), dbuf, fn * 8, cudaMemcpyDeviceToHost));
        gf2c_use_fused = true;
        for (int tl : {3, 4, 5})
            for (int bl : {3, 5})
                for (u64 ch : std::vector<u64>{0, 64}) {
                    gf2c_taylor_lv = tl;
                    gf2c_bfly_lv = bl;
                    gf2c_fuse_max_ch = ch;
                    CUCHK(cudaMemcpy(dbuf, hin.data(), fn * 8,
                                     cudaMemcpyHostToDevice));
                    Ff.fwd(dbuf);
                    CUCHK(cudaMemcpy(hout.data(), dbuf, fn * 8,
                                     cudaMemcpyDeviceToHost));
                    if (hout != href) okk = false;
                    Ff.inv(dbuf);
                    CUCHK(cudaMemcpy(hout.data(), dbuf, fn * 8,
                                     cudaMemcpyDeviceToHost));
                    if (hout != hin) okk = false;
                }
        gf2c_use_fused = sv_f;
        gf2c_taylor_lv = sv_t;
        gf2c_bfly_lv = sv_b;
        gf2c_fuse_max_ch = sv_c;
        cudaFree(dbuf);
        Ff.fini();
        printf("FFT variants: legacy vs fused configs (m=12)   %s\n",
               okk ? "PASS" : "FAIL");
        if (!okk) bad++;
    }

    // (4) end-to-end vs oracle at r = 4423
#ifndef HAVE_NTL
    printf("end-to-end vs oracle                               SKIPPED (NTL required)\n");
    printf("fineDDF bisection path (forced)                    SKIPPED (NTL required)\n");
    printf("-z gives up / -Z caps and continues                SKIPPED (NTL required)\n");
    printf("checkpoint kill/resume equivalence                 SKIPPED (NTL required)\n");
    printf("NOTE: this binary cannot resolve factors from multi-factor\n"
           "      interval gcds.  Ensure ntl_check.cpp is present next to\n"
           "      the Makefile and libntl-dev is installed, then rebuild;\n"
           "      `make tsfactor` prints the NTL detection result.\n");
    (void)gcd_threads;
#else
    {
        const u64 rr = 4423, skip = 5, maxd = 200;
        std::vector<u64> surv;
        std::map<u64, OracleOut> want;
        for (u64 sv = 2; sv <= 350; sv++) {
            OracleOut o = oracle_scan(rr, sv, skip, maxd);
            if (o.survivor) { surv.push_back(sv); want[sv] = std::move(o); }
        }
        u64 nf = 0, nu = 0, nt = 0;
        for (auto &kv : want)
            if (kv.second.found) { nf++; if (kv.second.gdeg != kv.second.d) nt++; }
            else nu++;
        printf("oracle r=%" PRIu64 ": %zu survivors (%" PRIu64 " found, %"
               PRIu64 " u, %" PRIu64 " ties)\n", rr, surv.size(), nf, nu, nt);

        GcdPool pool;
        pool.start(gcd_threads);
        u64 gseq = 0;
        struct Cfg { int m; long q0; double f; bool legacy; const char *nm; };
        Cfg cfgs[] = {{4, 3, 1.0, false, "m=4 q0=3 f=1"},
                      {8, 2, 1.0, false, "m=8 q0=2 f=1"},
                      {6, 1, 0.0, true,  "m=6 q0=1 f=0 legacy-sq"},
                      {14, 15, 1.0, false, "m=14 q0=15 f=1"},
                      {3, 2, 2.0, false, "m=3 q0=2 f=2"}};
        for (auto &cf : cfgs) {
            TsGPU G;
            G.init(rr);
            G.set_m(cf.m);
            G.legacy_sq = cf.legacy;
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = cf.q0; P.f = cf.f;
            P.canzass_max = 2 * 270;   // 2*r^(2/3) ~ 540
            P.verbose = 0;
            u64 mism = 0;
            for (u64 sv : surv) {
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                const OracleOut &o = want[sv];
                bool okk;
                if (o.found) {
                    okk = res.kind == ScanOut::FOUND && res.d == o.d;
                    if (okk && o.mask_ok)
                        okk = res.hex == mask_hex(o.mask);
                } else {
                    okk = res.kind == ScanOut::GAVE_UP;
                }
                if (!okk) {
                    mism++;
                    printf("  MISMATCH cfg[%s] s=%" PRIu64 " kind=%d d=%"
                           PRIu64 " (want found=%d d=%" PRIu64 ")\n",
                           cf.nm, sv, (int)res.kind, res.d, (int)o.found,
                           o.d);
                }
            }
            printf("end-to-end vs oracle [%s]                %s\n", cf.nm,
                   mism ? "FAIL" : "PASS");
            if (mism) bad++;
            G.fini();
        }

        // (5) fineDDF forced on every hit (canzass_max = 1)
        {
            TsGPU G;
            G.init(rr);
            G.set_m(5);
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 4; P.f = 1.0;
            P.canzass_max = 1;
            P.verbose = 0;
            u64 mism = 0;
            for (size_t i = 0; i < surv.size(); i += 3) {
                u64 sv = surv[i];
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                const OracleOut &o = want[sv];
                bool okk = o.found ? (res.kind == ScanOut::FOUND &&
                                      res.d == o.d)
                                   : (res.kind == ScanOut::GAVE_UP);
                if (okk && o.found && o.mask_ok)
                    okk = res.hex == mask_hex(o.mask);
                if (!okk) mism++;
            }
            printf("fineDDF bisection path (forced)                    %s\n",
                   mism ? "FAIL" : "PASS");
            if (mism) bad++;
            G.fini();
        }

        // (6) -z / -Z semantics on a known-deep s
        {
            u64 deep_s = 0, deep_d = 0;
            for (auto &kv : want)
                if (kv.second.found && kv.second.d > 60 &&
                    kv.second.d > deep_d) { deep_s = kv.first;
                                            deep_d = kv.second.d; }
            bool okk = true;
            if (deep_s) {
                TsGPU G;
                G.init(rr);
                G.set_m(4);
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 2; P.f = 1.0;
                P.canzass_max = 540;
                P.verbose = 0;
                P.zq = 3;              // trips before reaching deep_d
                ScanOut r1 = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                        nullptr);
                okk = okk && r1.kind == ScanOut::GAVE_UP;
                P.zq = 0;
                P.Zq = 3;              // capped q but keeps going
                ScanOut r2 = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                        nullptr);
                okk = okk && r2.kind == ScanOut::FOUND && r2.d == deep_d;
                G.fini();
            }
            printf("-z gives up / -Z caps and continues                %s\n",
                   okk ? "PASS" : "FAIL");
            if (!okk) bad++;
        }

        // (7) checkpoint / resume equivalence (kill at various points)
        {
            u64 deep_s = 0, deep_d = 0;
            for (auto &kv : want)
                if (kv.second.found && kv.second.d > 100) {
                    deep_s = kv.first; deep_d = kv.second.d; break;
                }
            bool okk = true;
            if (deep_s) {
                TsGPU G;
                G.init(rr);
                G.set_m(5);
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 2; P.f = 1.0;
                P.canzass_max = 540;
                P.verbose = 0;
                P.ckpt_base = "/tmp/tsfactor_test.ckpt";
                P.ckpt_secs = 1e9;
                long kills[] = {0, 1, 3, 6, 11};
                for (long ka : kills) {
                    remove("/tmp/tsfactor_test.ckpt");
                    remove("/tmp/tsfactor_test.ckpt.1");
                    remove("/tmp/tsfactor_test.ckpt.2");
                    SParams P2 = P;
                    P2.die_after_blocks = ka;
                    ScanOut r1 = scan_one_s(G, pool, P2, rr, deep_s, gseq,
                                            nullptr);
                    if (r1.kind != ScanOut::INTERRUPTED) {
                        // died late enough to finish: fine, just check
                        okk = okk && r1.kind == ScanOut::FOUND &&
                              r1.d == deep_d;
                        continue;
                    }
                    ResumeState RS;
                    RS.valid = load_ckpt(P.ckpt_base, RS.H, RS.h, RS.a);
                    ScanOut r2;
                    if (RS.valid && RS.H.s == deep_s)
                        r2 = scan_one_s(G, pool, P, rr, deep_s, gseq, &RS);
                    else            // no verified state: restart is correct
                        r2 = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                        nullptr);
                    okk = okk && r2.kind == ScanOut::FOUND && r2.d == deep_d;
                    if (!okk) printf("  ckpt kill@%ld: d=%" PRIu64
                                     " want %" PRIu64 "\n", ka, r2.d, deep_d);
                }
                G.fini();
            }
            printf("checkpoint kill/resume equivalence                 %s\n",
                   okk ? "PASS" : "FAIL");
            if (!okk) bad++;
        }

        // (8) hybrid HGCD gcd vs NTL gcd: CPU-mult and GPU-hook paths,
        //     nonlinear and adversarial F2-linear (LFSR) data, planted
        //     common factors, degenerate shapes
        {
            bool okk = true;
            u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
            hyb_ntl_finish_bits = 64;    // force deep HGCD recursion
            hyb_gpu_min_bits = 2048;     // exercise the hook heavily
            hyb_reset_stats();
            CantorBasis bb8;
            bb8.build();
            TsGpuMul gm8;
            gm8.init(&bb8, 15);
            u64 rs8 = 0xC0FFEEULL;
            auto nl = [&]() { rs8 += 0x9E3779B97F4A7C15ULL; u64 z = rs8;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                return z ^ (z >> 31); };
            auto lf = [&]() { rs8 ^= rs8 << 13; rs8 ^= rs8 >> 7;
                rs8 ^= rs8 << 17; return rs8; };
            auto rvb = [&](u64 bits, bool lin) {
                std::vector<u64> v((size_t)((bits + 63) / 64));
                for (auto &w : v) w = lin ? lf() : nl();
                unsigned tb = (unsigned)((bits - 1) & 63);
                v.back() &= (tb == 63) ? ~0ULL : ((1ULL << (tb + 1)) - 1);
                v.back() |= 1ULL << tb;
                return v;
            };
            auto chk = [&](std::vector<u64> a, std::vector<u64> b) {
                std::vector<u64> g1, g2, g3;
                u64 d1 = poly_gcd_ntl(a.data(), a.size(), b.data(),
                                      b.size(), &g1, ~0ULL);
                u64 d2 = poly_gcd_hybrid(a.data(), a.size(), b.data(),
                                         b.size(), &g2, ~0ULL, &gm8);
                u64 d3 = poly_gcd_hybrid(a.data(), a.size(), b.data(),
                                         b.size(), &g3, ~0ULL, nullptr);
                bool o = d1 == d2 && d1 == d3 &&
                         (d1 == 0 || (g1 == g2 && g1 == g3));
                if (!o) okk = false;
            };
            for (u64 bits : std::vector<u64>{600, 5000, 60000}) {
                for (int lin = 0; lin < 2; lin++) {
                    chk(rvb(bits, lin), rvb(bits - 1 - nl() % 30, lin));
                    // planted common factor via the hook itself
                    u64 gd = 20 + nl() % (bits / 3);
                    auto g = rvb(gd + 1, false);
                    auto u = rvb(bits - gd, lin), w = rvb(bits - gd, lin);
                    std::vector<u64> pa, pb;
                    gm8.mul(g.data(), g.size(), gd + 1, u.data(), u.size(),
                            bits - gd, pa);
                    gm8.mul(g.data(), g.size(), gd + 1, w.data(), w.size(),
                            bits - gd, pb);
                    chk(pa, pb);
                }
                chk(rvb(bits, false), std::vector<u64>());
                { auto sa = rvb(bits, false); chk(sa, sa); }
                chk(rvb(bits, false), std::vector<u64>{1});
                chk(rvb(bits, false), rvb(9, false));
            }
            u64 g8, c8, f8;
            hyb_get_stats(g8, c8, f8);
            hyb_gpu_min_bits = sv_min;
            hyb_ntl_finish_bits = sv_fin;
            gm8.fini();
            printf("hybrid HGCD vs NTL (gpu-mults %" PRIu64 ", fb %" PRIu64
                   ")             %s\n", g8, f8, okk ? "PASS" : "FAIL");
            if (!okk) bad++;
        }

        // (9) end-to-end scan with the hybrid GPU gcd forced through the
        //     pool (thresholds shrunk so 4423-bit gcds take the path)
        {
            u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
            hyb_gpu_min_bits = 512;
            hyb_ntl_finish_bits = 256;
            CantorBasis bb9;
            bb9.build();
            TsGpuMul gm9;
            gm9.init(&bb9, 12);
            g_gm = &gm9;
            g_use_hybrid = true;
            g_hyb_xcheck = true;
            hyb_reset_stats();
            TsGPU G;
            G.init(rr);
            G.set_m(6);
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 3; P.f = 1.0;
            P.canzass_max = 540;
            P.verbose = 0;
            u64 mism = 0;
            for (size_t i = 0; i < surv.size(); i += 4) {
                u64 sv = surv[i];
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                const OracleOut &o = want[sv];
                bool ok9 = o.found ? (res.kind == ScanOut::FOUND &&
                                      res.d == o.d)
                                   : (res.kind == ScanOut::GAVE_UP);
                if (ok9 && o.found && o.mask_ok)
                    ok9 = res.hex == mask_hex(o.mask);
                if (!ok9) mism++;
            }
            u64 g9, c9, f9;
            hyb_get_stats(g9, c9, f9);
            g_gm = nullptr;
            g_use_hybrid = false;
            g_hyb_xcheck = false;
            hyb_gpu_min_bits = sv_min;
            hyb_ntl_finish_bits = sv_fin;
            G.fini();
            gm9.fini();
            printf("end-to-end, hybrid gcd forced (gpu-mults %" PRIu64
                   ", fb %" PRIu64 ")   %s\n", g9, f9,
                   mism ? "FAIL" : "PASS");
            if (mism) bad++;
        }
        pool.stop();
    }
#endif // HAVE_NTL

#ifdef HAVE_NTL
    printf(bad ? "SELFTEST: %d FAILURE(S)\n" : "SELFTEST: all PASS\n", bad);
#else
    printf(bad ? "SELFTEST: %d FAILURE(S)\n"
               : "SELFTEST: kernel suites PASS; resolution suites SKIPPED "
                 "(built without NTL)\n", bad);
#endif
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// bench: component times at full scale + m recommendation

static int bench(u64 r, u64 s) {
    if (!s) s = (r >= 5) ? (r - 1) / 2 : 1;
    printf("bench r=%" PRIu64 " s=%" PRIu64 "\n", r, s);
    TsGPU G;
    G.init(r);
    G.set_m(24);
    G.set_s(s);
    std::vector<u64> a = st_res(r);
    CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.dH[0], a.data(), G.nw * 8, cudaMemcpyHostToDevice));
    auto timeit = [&](const char *nm, int reps, auto fn) -> double {
        fn();                                   // warm-up
        CUCHK(cudaDeviceSynchronize());
        double t0 = trin_now_s();
        for (int i = 0; i < reps; i++) fn();
        CUCHK(cudaDeviceSynchronize());
        double dt = (trin_now_s() - t0) / reps;
        printf("  %-28s %10.3f ms\n", nm, dt * 1e3);
        return dt;
    };
    G.legacy_sq = true;
    double t_leg = timeit("squaring (legacy 3-kernel)", 10,
                          [&] { G.square_slot(G.dA, G.dTmp); });
    G.legacy_sq = false;
    double t_fus = timeit("squaring (fused)", 10,
                          [&] { G.square_slot(G.dA, G.dTmp); });
    double t_hor = timeit("Horner (m=24)", 10, [&] { G.horner_C(); });
    double t_mul = timeit("modmul (FFT + reduce)", 4,
                          [&] { G.mul_A_by_C(); });
    {   // component breakdown of the modmul (drives the C2 FFT work)
        size_t ca = (size_t)((r + 31) >> 5);
        int blocks, threads;
        CudaFFT::launch_dims(G.F.n, blocks, threads);
        timeit("  pack (one operand)", 4, [&] {
            k_pack<<<blocks, threads>>>(G.dA, G.nw, G.dFA, ca, G.F.n); });
        timeit("  forward FFT (one of 2)", 4, [&] { G.F.fwd(G.dFA); });
        timeit("  pointwise GF(2^64)", 4, [&] {
            k_pointwise<<<blocks, threads>>>(G.dFA, G.dFB, G.F.n); });
        timeit("  inverse FFT", 4, [&] { G.F.inv(G.dFA); });
        int b2, t2;
        CudaFFT::launch_dims(G.nw2, b2, t2);
        timeit("  overlap-add", 4, [&] {
            k_overlap_add<<<b2, t2>>>(G.dFA, 2 * ca - 1, G.dS0, G.nw2); });
        timeit("  fold reduce (2 passes)", 4, [&] {
            G.reduce_into(G.dTmp); });
    }
    double t_sq = t_fus < t_leg ? t_fus : t_leg;
    printf("per-degree projection (m*t_sq + (t_mul + t_horner(m))/m):\n");
    int best_m = 8;
    double best_c = 1e100;
    for (int m : {8, 12, 16, 20, 24, 32, 40, 48, 56, 64}) {
        double c = m * t_sq + (t_mul + t_hor * m / 24.0) / m;
        printf("  m=%-3d  %8.2f ms/degree\n", m, c * 1e3);
        if (c < best_c) { best_c = c; best_m = m; }
    }
    int mo = (int)(sqrt(t_mul / t_sq) + 0.5) & ~1;
    if (mo < 8) mo = 8;
    if (mo > 64) mo = 64;
    {   // FFT autotune: verify + time transform variants; the winner is
        // left active (so a following --bench-gcd uses it) and printed
        // as environment settings that production runs pick up.
        printf("FFT autotune (fwd+inv per config; ~1 min on GPU):\n");
        size_t fn = G.F.n;
        std::vector<u64> hin(fn), href(fn), hout(fn);
        u64 st = 0xfeedfaceULL;
        for (auto &w : hin) {
            st += 0x9E3779B97F4A7C15ULL;
            u64 z = st;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            w = z ^ (z >> 31);
        }
        auto push = [&] {
            CUCHK(cudaMemcpy(G.dFA, hin.data(), fn * 8,
                             cudaMemcpyHostToDevice));
        };
        auto pull = [&](std::vector<u64> &h) {
            CUCHK(cudaDeviceSynchronize());
            CUCHK(cudaMemcpy(h.data(), G.dFA, fn * 8,
                             cudaMemcpyDeviceToHost));
        };
        auto timecfg = [&]() -> double {
            G.F.fwd(G.dFA); G.F.inv(G.dFA);        // warm-up
            CUCHK(cudaDeviceSynchronize());
            double t0 = trin_now_s();
            for (int i2 = 0; i2 < 3; i2++) { G.F.fwd(G.dFA); G.F.inv(G.dFA); }
            CUCHK(cudaDeviceSynchronize());
            return (trin_now_s() - t0) / 3;
        };
        bool sv_f = gf2c_use_fused;
        int sv_t = gf2c_taylor_lv, sv_b = gf2c_bfly_lv;
        u64 sv_c = gf2c_fuse_max_ch;
        gf2c_use_fused = false;
        push();
        G.F.fwd(G.dFA);
        pull(href);
        double t_legacy = timecfg();
        printf("  %-34s %8.1f ms\n", "legacy per-level", t_legacy * 1e3);
        double best_t = t_legacy;
        int best_tl = 0, best_bl = 0;
        u64 best_ch = 0;                            // 0/0/0 = legacy
        gf2c_use_fused = true;
        for (int tl : {3, 4, 5}) {
            for (int bl : {3, 4, 5}) {
                for (u64 ch : std::vector<u64>{0, (u64)1 << 16, (u64)1 << 12}) {
                    gf2c_taylor_lv = tl;
                    gf2c_bfly_lv = bl;
                    gf2c_fuse_max_ch = ch;
                    push();
                    G.F.fwd(G.dFA);
                    pull(hout);
                    bool okf = hout == href;
                    G.F.inv(G.dFA);
                    pull(hout);
                    bool okr = hout == hin;
                    double tc = timecfg();
                    char nm[64];
                    snprintf(nm, sizeof nm, "fused tlv=%d blv=%d maxch=%s",
                             tl, bl,
                             ch == 0 ? "off" : (ch == ((u64)1 << 16) ? "2^16"
                                                                     : "2^12"));
                    printf("  %-34s %8.1f ms%s\n", nm, tc * 1e3,
                           (okf && okr) ? "" : "  VERIFY-FAIL");
                    if (okf && okr && tc < best_t) {
                        best_t = tc;
                        best_tl = tl; best_bl = bl; best_ch = ch;
                    }
                }
            }
        }
        if (best_tl) {
            gf2c_use_fused = true;
            gf2c_taylor_lv = best_tl;
            gf2c_bfly_lv = best_bl;
            gf2c_fuse_max_ch = best_ch;
            printf("best: fused tlv=%d blv=%d maxch=%" PRIu64
                   " -- fwd+inv %.1f ms (%.2fx vs legacy)\n",
                   best_tl, best_bl, best_ch, best_t * 1e3,
                   t_legacy / best_t);
            printf("  production: export GF2C_FFT=fused GF2C_TAYLOR_LV=%d "
                   "GF2C_BFLY_LV=%d GF2C_FUSE_MAX_CH=%" PRIu64 "\n",
                   best_tl, best_bl, best_ch);
        } else {
            gf2c_use_fused = false;
            printf("best: legacy per-level -- fwd+inv %.1f ms\n",
                   t_legacy * 1e3);
            printf("  production: export GF2C_FFT=legacy\n");
        }
        double t_mul2 = 1.5 * best_t + 0.003;       // 3 transforms + misc
        double bm = 8, bc = 1e100;
        for (int mm2 = 8; mm2 <= 96; mm2 += 2) {
            double c = mm2 * t_sq + (t_mul2 + t_hor * mm2 / 24.0) / mm2;
            if (c < bc) { bc = c; bm = mm2; }
        }
        printf("projected with tuned FFT: modmul ~%.0f ms, m=%d at "
               "~%.1f ms/degree\n", t_mul2 * 1e3, (int)bm, bc * 1e3);
        (void)sv_f; (void)sv_t; (void)sv_b; (void)sv_c;
    }
        printf("recommended m: %d (analytic %d); expect ~%.1f ms/degree\n",
           best_m, mo, best_c * 1e3);
    G.fini();
    
return 0;
}

static int auto_m(TsGPU &G) {
    std::vector<u64> a = st_res(G.r);
    CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8, cudaMemcpyHostToDevice));
    G.square_slot(G.dA, G.dTmp);
    CUCHK(cudaDeviceSynchronize());
    double t0 = trin_now_s();
    for (int i = 0; i < 6; i++) G.square_slot(G.dA, G.dTmp);
    CUCHK(cudaDeviceSynchronize());
    double t_sq = (trin_now_s() - t0) / 6;
    G.mul_A_by_C();
    CUCHK(cudaDeviceSynchronize());
    t0 = trin_now_s();
    for (int i = 0; i < 2; i++) G.mul_A_by_C();
    CUCHK(cudaDeviceSynchronize());
    double t_mul = (trin_now_s() - t0) / 2;
    int mo = (int)(sqrt(t_mul / t_sq) + 0.5) & ~1;
    if (mo < 8) mo = 8;
    if (mo > 64) mo = 64;
    fprintf(stderr, "auto-m: t_sq %.3f ms, t_mul %.1f ms -> m = %d\n",
            t_sq * 1e3, t_mul * 1e3, mo);
    return mo;
}

// ---------------------------------------------------------------------------
// main

static void usage() {
    fprintf(stderr,
        "Usage: tsfactor <r> <survivors.txt> --skip D [options]\n"
        "       tsfactor --selftest [--gcd-threads N]\n"
        "       tsfactor --bench <r> [s]\n"
        "Options:\n"
        "  --skip D        REQUIRED: inputs have no factor of degree <= D\n"
        "  --maxd D        do not scan degrees > D (emit \"s u\")\n"
        "  --m M           block size 2..64 (default: auto-benchmark)\n"
        "  --q0 N          blocks in the first interval (default 600/m)\n"
        "  --f F           interval growth (0 const, 1 linear, >1 geometric)\n"
        "  -z Q            emit \"s u\" when q would exceed Q (factor -z)\n"
        "  -Z Q            cap q at Q but keep scanning\n"
        "  --out PREFIX    results -> PREFIX.results.txt (default ts)\n"
        "  --no-ckpt       disable checkpointing\n"
        "  --ckpt-mins N   periodic checkpoint cadence (default 15)\n"
        "  --gcd-threads N CPU GCD pool size\n"
        "  --device N      CUDA device index\n"
        "  --canzass-max D fineDDF threshold (default 2*r^(2/3))\n"
        "  --legacy-sq     use the 3-kernel squaring path\n"
        "  --pend-max N    max speculative intervals in flight (default 4)\n"
        "  --gpu-gcd       hybrid HGCD gcd: big multiplications on the\n"
        "                  GPU, CPU load per gcd drops sharply (opt-in)\n"
        "  --gpu-gcd-min-bits N   offload mults with an operand >= N bits\n"
        "  --bench-gcd     with --bench: also time one full-size gcd\n"
        "  -v              verbose (repeat for more)\n");
    exit(1);
}

int main(int argc, char **argv) {
    crc32_init();
    u64 r = 0, s_bench = 0;
    std::string surv_path, out_prefix = "ts";
    SParams P;
    P.skip = (u64)-1;
    int gcd_threads = 0, device = -1, want_m = 0;
    bool do_selftest = false, do_bench = false, legacy = false,
         no_ckpt = false, allow_no_ntl = false, gpu_gcd = false,
         bench_gcd = false;
    u64 gcd_min_bits_opt = 0;
    double f_opt = 1.0;

    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *o) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", o);
                                 exit(1); }
            return argv[++i];
        };
        if (a == "--selftest") do_selftest = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--skip") P.skip = strtoull(need("--skip"), 0, 10);
        else if (a == "--maxd") P.maxd = strtoull(need("--maxd"), 0, 10);
        else if (a == "--m") want_m = atoi(need("--m"));
        else if (a == "--q0") P.q0 = atol(need("--q0"));
        else if (a == "--f") f_opt = atof(need("--f"));
        else if (a == "-z") P.zq = atol(need("-z"));
        else if (a == "-Z") P.Zq = atol(need("-Z"));
        else if (a == "--out") out_prefix = need("--out");
        else if (a == "--no-ckpt") no_ckpt = true;
        else if (a == "--ckpt-mins")
            P.ckpt_secs = atof(need("--ckpt-mins")) * 60.0;
        else if (a == "--gcd-threads") gcd_threads = atoi(need("--gcd-threads"));
        else if (a == "--device") device = atoi(need("--device"));
        else if (a == "--canzass-max")
            P.canzass_max = strtoull(need("--canzass-max"), 0, 10);
        else if (a == "--legacy-sq") legacy = true;
        else if (a == "--allow-no-ntl") allow_no_ntl = true;
        else if (a == "--gpu-gcd") gpu_gcd = true;
        else if (a == "--gpu-gcd-min-bits")
            gcd_min_bits_opt = strtoull(need("--gpu-gcd-min-bits"), 0, 10);
        else if (a == "--bench-gcd") bench_gcd = true;
        else if (a == "--pend-max") P.pend_max = atoi(need("--pend-max"));
        else if (a == "--die-after-blocks")
            P.die_after_blocks = atol(need("--die-after-blocks"));
        else if (a == "-v") P.verbose++;
        else if (a == "-vv") P.verbose += 2;
        else if (a.size() && a[0] == '-' && !isdigit(a[1])) {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
        }
        else pos.push_back(a);
    }
    P.f = f_opt;
    if (gcd_threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        gcd_threads = hc > 2 ? (int)std::min(3u, hc - 1) : 1;
    }

    if (do_selftest) return selftest(gcd_threads);
    if (do_bench) {
        if (pos.empty()) usage();
        r = strtoull(pos[0].c_str(), 0, 10);
        if (pos.size() > 1) s_bench = strtoull(pos[1].c_str(), 0, 10);
        if (device >= 0) cudaSetDevice(device);
        int brc = bench(r, s_bench);
#ifdef HAVE_NTL
        if (!brc && bench_gcd) {
            u64 sb = s_bench ? s_bench : (r >= 5 ? (r - 1) / 2 : 1);
            fprintf(stderr, "GCD backend: %s\n", gcd_backend());
            size_t nwt = bits_to_words(r + 1);
            std::vector<u64> T(nwt, 0), A(bits_to_words(r), 0);
            T[0] |= 1;
            T[sb >> 6] |= 1ULL << (sb & 63);
            T[r >> 6] |= 1ULL << (r & 63);
            u64 st = 0x1234abcd;
            for (auto &w : A) {
                st += 0x9E3779B97F4A7C15ULL;
                u64 z = st;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                w = z ^ (z >> 31);
            }
            A[(r - 1) >> 6] &= (((r - 1) & 63) == 63)
                ? ~0ULL : ((1ULL << (((r - 1) & 63) + 1)) - 1);
            CantorBasis bb;
            if (gpu_gcd) {
                bb.build();
                size_t ca = (size_t)((r + 31) >> 5);
                int fm = 1;
                while (((size_t)1 << fm) < 2 * ca - 1) fm++;
                g_ts_gm.init(&bb, fm);
                g_gm = &g_ts_gm;
                g_use_hybrid = true;
                if (gcd_min_bits_opt) hyb_gpu_min_bits = gcd_min_bits_opt;
                hyb_reset_stats();
            }
            g_keep_bits = r + 1;
            double t0 = trin_now_s();
            std::vector<u64> gg;
            u64 gd = ts_gcd(A.data(), A.size(), T.data(), T.size(), &gg);
            double dt = trin_now_s() - t0;
            printf("gcd bench (random A vs T, %s): %.2f s (gdeg %" PRIu64
                   ")\n", g_use_hybrid ? "hybrid" : "NTL", dt, gd);
            if (g_use_hybrid) {
                u64 ng, nc, nf;
                hyb_get_stats(ng, nc, nf);
                printf("  hybrid mults: gpu %" PRIu64 " cpu %" PRIu64
                       " fallbacks %" PRIu64 "\n", ng, nc, nf);
                g_ts_gm.fini();
            }
        }
#endif
        return brc;
    }
    if (pos.size() != 2) usage();
    r = strtoull(pos[0].c_str(), 0, 10);
    surv_path = pos[1];
    if (!(r & 1) || r < 5) { fprintf(stderr, "r must be odd, >= 5\n");
                             exit(1); }
    if (P.skip == (u64)-1) {
        fprintf(stderr, "--skip is REQUIRED (36 or 37 for the r=136279841 "
                        "pipeline);\nscanning starts at skip+1 and factors "
                        "of degree <= skip would be MISSED.\n");
        exit(1);
    }
    if (P.zq && P.Zq) { fprintf(stderr, "-z and -Z are mutually "
                                        "exclusive\n"); exit(1); }
    if (!P.canzass_max) {
        u64 r3 = 0;
        while (r3 * r3 * r3 < r) r3++;
        P.canzass_max = 2 * r3 * r3;
    }
#ifdef HAVE_NTL
    if (gpu_gcd) g_use_hybrid = true;    // reflected in the banner
#else
    if (gpu_gcd) {
        fprintf(stderr, "ERROR: --gpu-gcd needs an NTL build (the hybrid "
                        "engine uses NTL for CPU-side mults and the "
                        "finishing gcd)\n");
        exit(1);
    }
#endif
    fprintf(stderr, "GCD backend: %s\n", gcd_backend());
    if (gcd_threads == 1 && P.pend_max == 1)
        fprintf(stderr, "hint: with --gcd-threads 1, --pend-max 2 lets the "
                "scan run one interval\nahead of the single GCD worker at "
                "no extra CPU cost\n");
#ifndef HAVE_NTL
    if (!allow_no_ntl) {
        fprintf(stderr,
            "ERROR: refusing to scan survivors without NTL.  Every hit\n"
            "whose interval gcd holds more than one factor would come out\n"
            "as an unresolved range.  Ensure ntl_check.cpp sits next to\n"
            "the Makefile and libntl-dev (+libgmp-dev) is installed, then\n"
            "rebuild -- `make tsfactor` prints the NTL detection result\n"
            "and refuses silent no-NTL builds.  Override (degrees-only\n"
            "experiments): --allow-no-ntl.\n");
        exit(1);
    }
#else
    (void)allow_no_ntl;
#endif
    if (device >= 0) cudaSetDevice(device);

    // survivors
    std::vector<u64> svs;
    {
        FILE *f = fopen(surv_path.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", surv_path.c_str());
                  exit(1); }
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            char *p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '#' || *p == '\n' || !*p) continue;
            u64 sv = strtoull(p, 0, 10);
            if (!(sv >= 1 && 2 * sv < r)) {
                fprintf(stderr, "bad survivor s=%" PRIu64
                        " (need 1 <= s <= (r-1)/2)\n", sv);
                exit(1);
            }
            svs.push_back(sv);
        }
        fclose(f);
    }
    fprintf(stderr, "%zu survivors from %s\n", svs.size(), surv_path.c_str());

    // resume: completed s from the results file
    std::string res_path = out_prefix + ".results.txt";
    std::set<u64> done_s;
    {
        FILE *f = fopen(res_path.c_str(), "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                u64 sv = strtoull(line, 0, 10);
                if (sv) done_s.insert(sv);
            }
            fclose(f);
            fprintf(stderr, "%zu already completed in %s\n", done_s.size(),
                    res_path.c_str());
        }
    }
    if (!no_ckpt) P.ckpt_base = out_prefix + ".ckpt";

    // checkpoint (for the in-progress s, if any)
    ResumeState RS;
    if (!no_ckpt) {
        RS.valid = load_ckpt(P.ckpt_base, RS.H, RS.h, RS.a);
        if (RS.valid && (RS.H.r != r || RS.H.skip != P.skip ||
                         done_s.count(RS.H.s))) {
            fprintf(stderr, "checkpoint ignored (r/skip mismatch or s "
                            "already done)\n");
            RS.valid = false;
        }
    }

    TsGPU G;
    G.init(r);
    G.legacy_sq = legacy;
#ifdef HAVE_NTL
    if (gpu_gcd) {
        g_ts_gm.init(&G.Bb, G.fftm);
        g_gm = &g_ts_gm;
        if (gcd_min_bits_opt) hyb_gpu_min_bits = gcd_min_bits_opt;
        fprintf(stderr, "GPU-assisted GCD: on (GPU mults >= %" PRIu64
                " bits; NTL finish <= %" PRIu64 " bits)\n",
                hyb_gpu_min_bits, hyb_ntl_finish_bits);
        hyb_reset_stats();
    }
#endif
    int m = want_m;
    if (RS.valid) {
        if (m && m != (int)RS.H.m)
            fprintf(stderr, "note: --m %d overridden by checkpoint m=%"
                    PRIu64 "\n", m, RS.H.m);
        m = (int)RS.H.m;
    }
    if (!m) {
        G.set_m(24);            // provisional (auto_m uses dC via horner)
        G.set_s((r - 1) / 2);
        m = auto_m(G);
    }
    G.set_m(m);
    if (P.q0 <= 0) P.q0 = (long)((600 + m - 1) / m);
    fprintf(stderr, "r=%" PRIu64 " skip=%" PRIu64 " m=%d q0=%ld f=%g "
            "gcd-threads=%d pend-max=%d%s%s\n",
            r, P.skip, m, P.q0, P.f, gcd_threads, P.pend_max,
            P.zq ? " -z" : "", P.Zq ? " -Z" : "");

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    FILE *rf = fopen(res_path.c_str(), "a");
    if (!rf) { fprintf(stderr, "cannot append %s\n", res_path.c_str());
               exit(1); }

    GcdPool pool;
    pool.start(gcd_threads);
    g_keep_bits = r + 1;
    u64 gseq = 0, n_found = 0, n_u = 0;

    // process the checkpointed s first (if it is still pending)
    if (RS.valid) {
        auto it = std::find(svs.begin(), svs.end(), RS.H.s);
        if (it == svs.end() || done_s.count(RS.H.s)) RS.valid = false;
        else std::rotate(svs.begin(), it, it + 1);
    }

    bool interrupted = false;
    for (u64 sv : svs) {
        if (done_s.count(sv)) continue;
        if (g_stop) { interrupted = true; break; }
        if (P.verbose) printf("== s=%" PRIu64 " ==\n", sv);
        ScanOut o = scan_one_s(G, pool, P, r, sv, gseq,
                               RS.valid && RS.H.s == sv ? &RS : nullptr);
        if (o.kind == ScanOut::INTERRUPTED) { interrupted = true; break; }
        char buf[256];
        switch (o.kind) {
        case ScanOut::FOUND:
            snprintf(buf, sizeof(buf), "%" PRIu64 " %" PRIu64 " p", sv, o.d);
            fprintf(rf, "%s%s\n", buf, o.hex.c_str());
            printf("%s%s\n", buf, o.hex.c_str());
            n_found++;
            break;
        case ScanOut::GAVE_UP:
            fprintf(rf, "%" PRIu64 " u\n", sv);
            printf("%" PRIu64 " u\n", sv);
            n_u++;
            break;
        case ScanOut::PRIMITIVE:
            fprintf(rf, "%" PRIu64 " primitive\n", sv);
            printf("%" PRIu64 " primitive\n", sv);
            fprintf(stderr, "\n*** PRIMITIVE TRINOMIAL CANDIDATE: s=%"
                    PRIu64 " -- verify independently! ***\n\n", sv);
            break;
        case ScanOut::RANGE:
            fprintf(rf, "%" PRIu64 " r%" PRIu64 "-%" PRIu64 "\n", sv,
                    o.rlo, o.rhi);
            printf("%" PRIu64 " r%" PRIu64 "-%" PRIu64 "\n", sv, o.rlo,
                   o.rhi);
            break;
        default: break;
        }
        fflush(rf);
        fsync(fileno(rf));
        fflush(stdout);
        if (P.verbose)
            printf("== s=%" PRIu64 " done in %.1f s (%" PRIu64
                   " degrees) ==\n", sv, o.wall, o.degrees_scanned);
    }

    fclose(rf);
    if (interrupted) {
        fprintf(stderr, "interrupted: %" PRIu64 " found, %" PRIu64
                " u; exiting (in-flight GCDs abandoned)\n", n_found, n_u);
        fflush(nullptr);
        pool.stop_nowait();
        std::_Exit(0);             // workers may hold NTL state; skip
    }                              // static teardown entirely
    pool.stop();
    G.fini();
    fprintf(stderr, "done: %" PRIu64 " found, %" PRIu64 " u\n", n_found,
            n_u);
    return 0;
}

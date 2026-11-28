/*
Copyright (C) 2026  Gord Palameta
(actual author: Claude AI)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Some code may have been inspired by:
  GF2X library
  factor.cpp and other files in GF2X library apps subdirectory
  NTL library
*/

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
// legs on the GPU.  One gcd(A, T) per interval of q blocks runs on a
// CPU thread pool, overlapped with speculative scanning; verdicts are
// consumed strictly in interval order.  A hit is resolved by factoring
// the small interval gcd (CanZass); oversized gcds trigger factor.cpp's
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
// Round 10: this file holds only what is GPU-specific -- the kernels, the
// TsGPU backend, the GPU multiplier for the gcd engine, the FFT tuning,
// the GPU selftest suites and --bench.  Everything else (scan driver,
// checkpoints, schedule, gcd pool, error-checking recovery, command line)
// is in ts_core.h, shared with the CPU build tsfactor_cpu (ts_cpu.h).
//
// Build:   make tsfactor          (CUDA; NTL auto-detected, required
//                                  for production r)
//          make emul-tsfactor     (CPU emulation of the kernels)
//          make tsfactor_cpu      (no GPU at all: the CPU backend)
// Test:    ./tsfactor --selftest
// Bench:   ./tsfactor --bench <r> [s]
// Run:     ./tsfactor <r> <survivors.txt> --skip 55 [options]
//
// Error checking (ON by default; --no-ec disables): every block is
// verified on the fly -- the h-table by a Frobenius-linear Gerbicz
// chain, the Horner build / FFT product / reduction by cyclic folds,
// the h-table initialisation from a closed form, and every transfer
// and RAM snapshot by a fold on both sides.  A failed check restarts
// the scan from the newest verified state; see gf2_ec.h for the
// mechanisms and tsfactor_handoff.md for the integration.
//
// This program is free software, GPL v3 or later, and incorporates
// algorithmic structure from gf2x/apps/factor.cpp (GPL).

#include "gf2_trinomial_cuda.h"
#include "gf2_gcd.h"
#include "gf2_ec.h"
#include "ts_core.h"
#include "ts_cpu.h"     // the CPU backend: checked against TsGPU by --selftest

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
// `fired_out` (may be null) receives the mask of the e that were folded:
// it is the quotient of the unreduced Horner sum by T, which the error
// checker needs to predict phi(C) (gf2_ec.h, ecf_horner_predict).
__global__ void k_horner_tail(u64 *C, u64 r, u64 s, int m, u64 *fired_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    u64 fired = 0;
    for (int e = 0; e <= m - 2; e++) {
        u64 p = r + (u64)e;
        u64 bit = (C[p >> 6] >> (p & 63)) & 1;
        if (bit) {
            fired |= 1ULL << e;
            C[p >> 6] ^= 1ULL << (p & 63);
            u64 t1 = s + (u64)e;
            C[t1 >> 6] ^= 1ULL << (t1 & 63);
            C[(u64)e >> 6] ^= 1ULL << (e & 63);
        }
    }
    C[(u64)m >> 6] ^= 1ULL << (m & 63);         // + x^m  (m < r)
    if (fired_out) *fired_out = fired;
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
// TsGPU: the device context for the blocked scan (TsDev on CUDA)

struct TsGPU : TsDev {
    size_t nw2 = 0;            // scratch words for 2r bits
    size_t nwc = 0;            // words for the unreduced Horner (r+m-1 bits)
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

    // ---- error checking (gf2_ec.h) ----
    u64 *ecD = nullptr;        // window accumulator: XOR of the digests S_z
    u64 *ecS0 = nullptr;       // digest at the start of the window
    u64 *ecX = nullptr, *ecY = nullptr, *ecZ = nullptr;   // nw scratch each
    u64 *ecPart = nullptr;     // partial folds (3 * EC_NBLK_MAX * EC_NF)
    u64 *ecOut = nullptr;      // small result arena the host reads back
    enum { ECO_CU = 0,                  // fold of the unreduced Horner sum
           ECO_T = EC_NF,               // 3 segment folds of the product
           ECO_A = 4 * EC_NF,           // fold of the reduced product
           ECO_FIRED = 5 * EC_NF,       // Horner tail quotient mask
           ECO_FLAG = 5 * EC_NF + 1,    // full-width compare result
           ECO_TAB = 5 * EC_NF + 2,     // fold of a whole h-table
           ECO_WORDS = 6 * EC_NF + 2 };

    const char *kind() const override { return "GPU"; }
    void sync() override { CUCHK(cudaDeviceSynchronize()); }

    void init(u64 r_) override {
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
    void set_m(int m_) override {
        if (m_ < 2 || m_ > 64) { fprintf(stderr, "need 2 <= m <= 64\n"); exit(1); }
        if (m == m_) return;
        for (int i = 0; i < 2; i++)
            if (dH[i]) { cudaFree(dH[i]); dH[i] = nullptr; }
        if (dHold) { cudaFree(dHold); dHold = nullptr; }   // sized for old m
        m = m_;
        hcur = 0;
        nwc = bits_to_words(r + (u64)m - 1);
        CUCHK(cudaMalloc(&dH[0], (size_t)m * nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dH[1], (size_t)m * nw * sizeof(u64)));
    }
    void need_hold() override {
        if (!dHold) CUCHK(cudaMalloc(&dHold, (size_t)m * nw * sizeof(u64)));
    }
    void set_s(u64 s_) override {
        if (!(s_ >= 1 && 2 * s_ < r)) {
            fprintf(stderr, "need 1 <= s <= (r-1)/2 (got s=%" PRIu64 ")\n", s_);
            exit(1);
        }
        s = s_;
    }
    void fini() override {
        for (int i = 0; i < 2; i++) if (dH[i]) cudaFree(dH[i]);
        dH[0] = dH[1] = nullptr;
        if (dHold) cudaFree(dHold);
        dHold = nullptr;
        cudaFree(dA); cudaFree(dC); cudaFree(dTmp);
        cudaFree(dS0); cudaFree(dS1);
        cudaFree(dFA); cudaFree(dFB);
        dA = dC = dTmp = dS0 = dS1 = dFA = dFB = nullptr;
        if (ecD) {
            cudaFree(ecD); cudaFree(ecS0); cudaFree(ecX); cudaFree(ecY);
            cudaFree(ecZ); cudaFree(ecPart); cudaFree(ecOut);
            ecD = nullptr;
        }
        F.fini();
        m = 0;
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
    void square_all_step() override {
        for (int j = 0; j < m; j++)
            square_slot(hslot(j), hslot_o(j));
        hcur = 1 - hcur;
    }
    // C <- x^m + sum_j h[j] x^(m-1-j)  (reduced mod T)
    void horner_C() override {
        int blocks, threads;
        CudaFFT::launch_dims(nwc, blocks, threads);
        k_horner_fused<<<blocks, threads>>>(dH[hcur], nw, m, dC, nwc);
        k_horner_tail<<<1, 1>>>(dC, r, s, m,
                                ec_on ? ecOut + ECO_FIRED : nullptr);
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
    // A <- A * C mod T.  With error checking the unreduced product and the
    // reduced result are folded on the device (gf2_ec.h, mechanism 3); the
    // host compares them in ec_block_verify.
    void mul_A_by_C() override {
        mul_full(dA, dC);
        if (ec_inject_now(ECI_T)) inject(dS0, 2 * r - 1, ECI_T);
        if (ec_on) ec_after_mul();
        reduce_into(dA);
        if (ec_inject_now(ECI_RED)) inject(dA, r, ECI_RED);
        if (ec_on) ec_after_reduce();
        if (ec_inject_now(ECI_A)) inject(dA, r, ECI_A);
    }
    void reset_A() override {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_set_one<<<blocks, threads>>>(dA, nw);
        ecf_one(ecA);
    }

    // ---- error checking ------------------------------------------------
    void ec_setup(bool on) override {
        ec_on = on;
        if (!on || ecD) return;
        CUCHK(cudaMalloc(&ecD, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecS0, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecX, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecY, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecZ, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecPart,
                         (size_t)3 * EC_NBLK_MAX * EC_NF * sizeof(u64)));
        CUCHK(cudaMalloc(&ecOut, ECO_WORDS * sizeof(u64)));
        CUCHK(cudaMemset(ecOut, 0, ECO_WORDS * sizeof(u64)));
        CUCHK(cudaMemset(ecD, 0, nw * sizeof(u64)));
        ecf_one(ecA);
    }
    // test hook: flip one bit of a device buffer holding `nbits` bits
    void inject(u64 *buf, u64 nbits, int site) {
        u64 bit = ec_inject_rnd(nbits);
        k_ec_flip_bit<<<1, 1>>>(buf, bit);
        ec_inject_note(site, bit);
    }
    void inject_tab(u64 *base, int site) {
        int j = (int)ec_inject_rnd((u64)m);
        inject(base + (size_t)j * nw, r, site);
    }
    void inject_C(int site) override { inject(dC, r, site); }
    void inject_h(int site) override { inject_tab(dH[hcur], site); }
    void inject_hold(int site) override { inject_tab(dHold, site); }
    // fold of src[0..n) -> ecOut[off .. off+EC_NF)
    void ec_fold_dev(const u64 *src, size_t n, size_t off) {
        unsigned nb = ec_nblk_for(n);
        k_ec_fold_part<<<nb, EC_NF>>>(src, n, ecPart);
        k_ec_combine<<<1, EC_NF>>>(ecPart, nb, 1, ecOut + off);
    }
    void ec_read(size_t off, size_t n, u64 *dst) {
        CUCHK(cudaMemcpy(dst, ecOut + off, n * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    // fold of a whole h-table (m*nw words taken as one polynomial)
    void ec_table_fold(const u64 *base, EcFold &f) {
        ec_fold_dev(base, (size_t)m * nw, ECO_TAB);
        ec_read(ECO_TAB, EC_NF, f.w);
    }
    void table_fold(EcFold &f) override { ec_table_fold(dH[hcur], f); }
    void hold_fold(EcFold &f) override { ec_table_fold(dHold, f); }
    // per block, before the Horner build: D ^= S_z and the fold of the
    // unreduced Horner sum, in one pass over the table
    void ec_block_pre() override {
        double t0 = ec_tic();
        unsigned nb = ec_nblk_for(nwc);
        k_ec_block_pass<<<nb, EC_NF>>>(dH[hcur], nw, m, ecD, nwc, ecPart);
        k_ec_combine<<<1, EC_NF>>>(ecPart, nb, 1, ecOut + ECO_CU);
        ec_toc(t0, g_ecs.t_block);
    }
    void ec_after_mul() {                  // dS0 = unreduced product
        double t0 = ec_tic();
        unsigned nb = ec_nblk_for(nw2);
        k_ec_fold3_part<<<nb, EC_NF>>>(dS0, nw2, r, 2 * r - s, ecPart);
        k_ec_combine<<<1, EC_NF>>>(ecPart, nb, 3, ecOut + ECO_T);
        ec_toc(t0, g_ecs.t_block);
    }
    void ec_after_reduce() {
        double t0 = ec_tic();
        ec_fold_dev(dA, nw, ECO_A);
        ec_toc(t0, g_ecs.t_block);
    }
    void ec_read_block(EcFold &fCu, EcFold seg[3], EcFold &fAn,
                       u64 &fired) override {
        u64 buf[ECO_FLAG];
        ec_read(0, ECO_FLAG, buf);
        memcpy(fCu.w, buf + ECO_CU, sizeof fCu.w);
        for (int g = 0; g < 3; g++)
            memcpy(seg[g].w, buf + ECO_T + g * EC_NF, sizeof seg[g].w);
        memcpy(fAn.w, buf + ECO_A, sizeof fAn.w);
        fired = buf[ECO_FIRED];
    }
    // h-table chain: restart it from the table as it stands
    void ec_chain_reset() override {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_ec_digest<<<blocks, threads>>>(dH[hcur], nw, m, ecS0);
        k_set_zero<<<blocks, threads>>>(ecD, nw);
        ec_w = 0;
    }
    // h-table chain check (gf2_ec.h, mechanism 1), compared at full width:
    //     sigma(D) + D == S_start + S_now,   sigma = Frobenius^m.
    // With no block since the last check (ec_w == 0, D == 0) it reduces to
    // S_now == S_start: "the idle table has not changed".  Either way the
    // next window starts from the digest taken here.  True if clean.
    bool ec_window_check() override {
        double t0 = ec_tic();
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_ec_digest<<<blocks, threads>>>(dH[hcur], nw, m, ecX);
        const u64 *sig = ecD;
        if (ec_w > 0) {
            u64 *dst = ecY;
            for (int t = 0; t < m; t++) {
                square_slot(sig, dst);
                sig = dst;
                dst = (dst == ecY) ? ecZ : ecY;
            }
        }
        k_ec_set_word<<<1, 1>>>(ecOut + ECO_FLAG, 0);
        k_ec_or4<<<blocks, threads>>>(sig, ecD, ecS0, ecX, nw,
                                      ecOut + ECO_FLAG);
        u64 flag = 1;
        ec_read(ECO_FLAG, 1, &flag);
        std::swap(ecS0, ecX);
        k_set_zero<<<blocks, threads>>>(ecD, nw);
        ec_w = 0;
        g_ecs.windows++;
        ec_toc(t0, g_ecs.t_window);
        return flag == 0;
    }
    // Verify the table at block start k from first principles (gf2_ec.h,
    // mechanism 2):  (1 + y_k)(1 + S_k) == 1 + y_{k+m},  y_t = x^(2^t),
    // using 1 + y_t = (1 + x)^(2^t).  Costs k + m squarings of one residue
    // and one modmul -- about 1/m of building the table.  Leaves the chain
    // reset at this table.  True if the table is right.
    bool ec_verify_table_at(u64 k) override {
        double t0 = ec_tic();
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_set_one<<<blocks, threads>>>(ecY, nw);
        k_xor_x<<<1, 1>>>(ecY);                        // 1 + x
        u64 *cur = ecY, *oth = ecZ;
        for (u64 t = 0; t < k; t++) {
            square_slot(cur, oth);
            std::swap(cur, oth);
        }
        CUCHK(cudaMemcpy(ecX, cur, nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));   // 1 + y_k
        for (int t = 0; t < m; t++) {
            square_slot(cur, oth);
            std::swap(cur, oth);
        }                                              // cur = 1 + y_{k+m}
        ec_chain_reset();                              // ecS0 = S_k
        CUCHK(cudaMemcpy(ecD, ecS0, nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));
        k_ec_flip_bit<<<1, 1>>>(ecD, 0);               // 1 + S_k
        mul_full(ecX, ecD);
        reduce_into(oth);
        k_ec_set_word<<<1, 1>>>(ecOut + ECO_FLAG, 0);
        k_ec_neq<<<blocks, threads>>>(oth, cur, nw, ecOut + ECO_FLAG);
        u64 flag = 1;
        ec_read(ECO_FLAG, 1, &flag);
        k_set_zero<<<blocks, threads>>>(ecD, nw);
        g_ecs.n_init++;
        ec_toc(t0, g_ecs.t_init);
        return flag == 0;
    }

    // h-table init at block start kstart: Pascal-triangle build of the
    // elementary symmetric functions of (x^(2^0) .. x^(2^(m-1))), then
    // kstart Frobenius steps.  Mirrors factor.cpp's init_h exactly.
    void init_h_at(u64 kstart) override {
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
        if (ec_inject_now(ECI_INIT)) inject_h(ECI_INIT);
        for (u64 t = 0; t < kstart; t++) {
            square_all_step();
            if (ec_inject_now(ECI_INIT)) inject_h(ECI_INIT);
        }
    }

    void copy_h(u64 *dst_base, const u64 *src_base) {
        CUCHK(cudaMemcpy(dst_base, src_base, (size_t)m * nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));
    }
    void hold_save() override { copy_h(dHold, dH[hcur]); }
    void hold_restore() override { copy_h(dH[hcur], dHold); }
    void h_to_host(std::vector<u64> &out) override {
        out.resize((size_t)m * nw);
        CUCHK(cudaMemcpy(out.data(), dH[hcur],
                         (size_t)m * nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void h_from_host(const std::vector<u64> &in) override {
        CUCHK(cudaMemcpy(dH[hcur], in.data(),
                         (size_t)m * nw * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }
    void A_to_host(std::vector<u64> &out) override {
        out.resize(nw);
        CUCHK(cudaMemcpy(out.data(), dA, nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void A_from_host_raw(const std::vector<u64> &in) override {
        CUCHK(cudaMemcpy(dA, in.data(), nw * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }

    // ---- measurement ----
    double time_square(int reps) override {
        std::vector<u64> a = st_res(r);
        CUCHK(cudaMemcpy(dA, a.data(), nw * 8, cudaMemcpyHostToDevice));
        square_slot(dA, dTmp);
        sync();
        double t0 = trin_now_s();
        for (int i = 0; i < reps; i++) square_slot(dA, dTmp);
        sync();
        return (trin_now_s() - t0) / reps;
    }
    double time_mul(int reps) override {
        std::vector<u64> a = st_res(r), c = st_res(r);
        CUCHK(cudaMemcpy(dA, a.data(), nw * 8, cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(dC, c.data(), nw * 8, cudaMemcpyHostToDevice));
        const bool sv = ec_on;
        ec_on = false;
        mul_A_by_C();
        sync();
        double t0 = trin_now_s();
        for (int i = 0; i < reps; i++) mul_A_by_C();
        sync();
        ec_on = sv;
        return (trin_now_s() - t0) / reps;
    }

    // ---- selftest access ----
    void C_to_host(std::vector<u64> &out) override {
        out.resize(nw);
        CUCHK(cudaMemcpy(out.data(), dC, nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void ecD_to_host(std::vector<u64> &out) override {
        out.resize(nw);
        CUCHK(cudaMemcpy(out.data(), ecD, nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void ecD_zero() override {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_set_zero<<<blocks, threads>>>(ecD, nw);
    }
};

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
    bool alive = false;
    std::mutex mu;
    void init(CantorBasis *bb, int max_fm) {
        std::lock_guard<std::mutex> lk(mu);
        Bb = bb;
        cap_n = (size_t)1 << max_fm;
        CUCHK(cudaMalloc(&dA, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dB, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dM, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dR, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dP, (cap_n / 2 + 2) * sizeof(u64)));
        alive = true;
    }
    // Takes the hook mutex, so a call already in flight completes
    // first, and clears `alive`, so later calls decline (the engine
    // then multiplies on the CPU) instead of touching freed buffers.
    void fini() {
        std::lock_guard<std::mutex> lk(mu);
        alive = false;
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
        size_t nu = bits_to_words(bits);      // ignore caller padding: a
        if (nu > nw) nu = nw;                 // trailing zero word must
        int blocks, threads;                  // not overrun the staging
        CUCHK(cudaMemcpy(dP, w, nu * sizeof(u64), cudaMemcpyHostToDevice));
        CudaFFT::launch_dims(F->n, blocks, threads);
        k_pack<<<blocks, threads>>>(dP, nu, dst, c, F->n);
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
        if (!alive) return false;
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
        if (!alive) return false;
        CudaFFT *F = plan_for(fm);
        int blocks, threads;
        size_t au = bits_to_words(abits), bu = bits_to_words(bbits);
        if (au > aw) au = aw;
        if (bu > bw) bu = bw;
        CUCHK(cudaMemcpy(dP, a, au * sizeof(u64), cudaMemcpyHostToDevice));
        CudaFFT::launch_dims(F->n, blocks, threads);
        k_pack<<<blocks, threads>>>(dP, au, dA, ca, F->n);
        CUCHK(cudaMemcpy(dP, b, bu * sizeof(u64), cudaMemcpyHostToDevice));
        k_pack<<<blocks, threads>>>(dP, bu, dB, cb, F->n);
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

// ---------------------------------------------------------------------------
// GPU-only selftest suites

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

static std::unique_ptr<TsDev> make_gpu_dev(u64 r, int m, int variant) {
    std::unique_ptr<TsGPU> G(new TsGPU());
    G->init(r);
    G->set_m(m);
    G->legacy_sq = (variant == 1);
    return std::unique_ptr<TsDev>(G.release());
}
static std::unique_ptr<TsDev> make_cpu_dev(u64 r, int m, int variant) {
    std::unique_ptr<TsCPU> G(new TsCPU());
    G->init(r);
    G->set_m(m);
    G->force_generic = (variant == 1);
    return std::unique_ptr<TsDev>(G.release());
}
#ifdef HAVE_NTL
static GpuMulHook *gpu_hook_new(int max_fm) {
    struct Hook : TsGpuMul { CantorBasis bb; };
    Hook *h = new Hook();
    h->bb.build();
    h->init(&h->bb, max_fm);
    return h;
}
static void gpu_hook_free(GpuMulHook *h) {
    TsGpuMul *g = static_cast<TsGpuMul *>(h);
    g->fini();
    delete g;
}
#endif

static int gpu_kernel_suites() {
    int bad = 0;
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
        for (int tl : {2, 3, 4, 5})
            for (int bl : {2, 3, 5})
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
    return bad;
}

// TsCPU must hold exactly what TsGPU holds, step for step: the h-table
// after init and after each block, C, A, the chain accumulator and every
// fold the per-block check reads.  (Its own kernel suite checks it against
// the bit-level oracles; this checks it against the hardware-validated
// GPU path, on whatever machine runs the selftest.)
static int cross_backend_suite() {
    bool ok = true;
    int ncase = 0;
    struct C3 { u64 r, s; int m; u64 k; };
    const C3 cases[] = {{4423, 130, 5, 1}, {4423, 131, 4, 7},
                        {4423, 2211, 6, 3}, {4423, 2, 3, 2},
                        {9689, 129, 8, 5}, {9689, 4844, 5, 9},
                        {19937, 1, 4, 3}, {19937, 9967, 7, 11},
                        {3217, 67, 6, 4}, {3217, 1608, 9, 2}};
    for (const C3 &c : cases) {
        auto Gg = make_gpu_dev(c.r, c.m, 0);
        auto Gc = make_cpu_dev(c.r, c.m, 0);
        TsDev *D[2] = {Gg.get(), Gc.get()};
        for (TsDev *d : D) {
            d->set_s(c.s);
            d->ec_setup(true);
            d->init_h_at(c.k);
            d->reset_A();
            d->ec_chain_reset();
        }
        auto same = [&](const char *what, bool with_c) {
            std::vector<u64> a, b;
            const char *bad = nullptr;
            D[0]->h_to_host(a); D[1]->h_to_host(b);
            if (a != b) bad = "h-table";
            D[0]->A_to_host(a); D[1]->A_to_host(b);
            if (!bad && a != b) bad = "A";
            if (with_c) {
                D[0]->C_to_host(a); D[1]->C_to_host(b);
                a.resize(D[0]->nw); b.resize(D[0]->nw);
                if (!bad && a != b) bad = "C";
            }
            D[0]->ecD_to_host(a); D[1]->ecD_to_host(b);
            if (!bad && a != b) bad = "D";
            EcFold f0, f1;
            D[0]->table_fold(f0); D[1]->table_fold(f1);
            if (!bad && !ecf_eq(f0, f1)) bad = "table fold";
            if (bad) {
                printf("  CPU != GPU (%s) after %s (r=%" PRIu64 " s=%" PRIu64
                       " m=%d)\n", bad, what, c.r, c.s, c.m);
                ok = false;
            }
        };
        same("init", false);
        for (int b = 0; b < 3; b++) {
            EcFold cu[2], sg[2][3], an[2];
            u64 fi[2];
            for (int i = 0; i < 2; i++) {
                D[i]->ec_block_pre();
                D[i]->horner_C();
                D[i]->mul_A_by_C();
                for (int t = 0; t < c.m; t++) D[i]->square_all_step();
                D[i]->ec_read_block(cu[i], sg[i], an[i], fi[i]);
            }
            bool o = ecf_eq(cu[0], cu[1]) && ecf_eq(an[0], an[1]) &&
                     fi[0] == fi[1];
            for (int g = 0; g < 3; g++) o = o && ecf_eq(sg[0][g], sg[1][g]);
            if (!o) {
                printf("  CPU != GPU block folds (r=%" PRIu64 " s=%" PRIu64
                       ")\n", c.r, c.s);
                ok = false;
            }
            for (int i = 0; i < 2; i++) {
                if (D[i]->ec_block_verify()) {
                    printf("  %s block check failed on clean data\n",
                           D[i]->kind());
                    ok = false;
                }
                D[i]->ec_w++;
            }
            same("a block", true);
        }
        for (int i = 0; i < 2; i++)
            if (!D[i]->ec_window_check() ||
                !D[i]->ec_verify_table_at(c.k + 3 * (u64)c.m)) {
                printf("  %s chain/table check failed on clean data\n",
                       D[i]->kind());
                ok = false;
            }
        ncase++;
        for (TsDev *d : D) d->fini();
    }
    printf("CPU backend == GPU backend, step by step (%d cases) %s\n", ncase,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int gpu_selftest(int gcd_threads) {
    int bad = 0;
    printf("tsfactor selftest (%s, GPU build)\n", TS_ROUND_STR);
    printf("GCD backend: %s\n", gcd_backend());
    bad += gpu_kernel_suites();
    bad += cpu_kernel_suites();
    bad += cross_backend_suite();
    bad += selftest_sched();
    TsTestEnv E;
    E.name = "GPU";
    E.make = make_gpu_dev;
#ifdef HAVE_NTL
    E.hook_new = gpu_hook_new;
    E.hook_free = gpu_hook_free;
#endif
    bad += selftest_table_suites(E);
    bad += selftest_scan_suites(E, gcd_threads);
    selftest_footer(bad);
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// FFT tuning: time transform variants (each verified on the device against
// the legacy transform first) and pick the fastest

struct FftCfg { bool fused; int tlv, blv; u64 maxch; };

static void fft_apply(const FftCfg &c) {
    gf2c_use_fused = c.fused;
    if (c.fused) {
        gf2c_taylor_lv = c.tlv;
        gf2c_bfly_lv = c.blv;
        gf2c_fuse_max_ch = c.maxch;
    }
    gf2c_clamp_config();
}

// returns the best verified configuration; *t_best and *t_legacy receive
// fwd+inv seconds.  `print` lists every candidate.
static FftCfg fft_sweep(TsGPU &G, const std::vector<FftCfg> &cands, int reps,
                        bool print, double *t_best, double *t_legacy) {
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
        CUCHK(cudaMemcpy(G.dFA, hin.data(), fn * 8, cudaMemcpyHostToDevice));
    };
    auto pull = [&](std::vector<u64> &h) {
        CUCHK(cudaDeviceSynchronize());
        CUCHK(cudaMemcpy(h.data(), G.dFA, fn * 8, cudaMemcpyDeviceToHost));
    };
    auto timecfg = [&]() -> double {
        G.F.fwd(G.dFA); G.F.inv(G.dFA);        // warm-up
        CUCHK(cudaDeviceSynchronize());
        double t0 = trin_now_s();
        for (int i2 = 0; i2 < reps; i2++) { G.F.fwd(G.dFA); G.F.inv(G.dFA); }
        CUCHK(cudaDeviceSynchronize());
        return (trin_now_s() - t0) / reps;
    };
    FftCfg leg = {false, 0, 0, 0};
    fft_apply(leg);
    push();
    G.F.fwd(G.dFA);
    pull(href);
    double tl = timecfg();
    if (print) printf("  %-34s %8.1f ms\n", "legacy per-level", tl * 1e3);
    FftCfg best = leg;
    double bt = tl;
    for (const FftCfg &c : cands) {
        fft_apply(c);
        push();
        G.F.fwd(G.dFA);
        pull(hout);
        bool okf = hout == href;
        G.F.inv(G.dFA);
        pull(hout);
        bool okr = hout == hin;
        double tc = timecfg();
        if (print) {
            char nm[64];
            snprintf(nm, sizeof nm, "fused tlv=%d blv=%d maxch=%s", c.tlv,
                     c.blv, c.maxch == 0 ? "off" :
                     (c.maxch == ((u64)1 << 16) ? "2^16" : "2^12"));
            printf("  %-34s %8.1f ms%s\n", nm, tc * 1e3,
                   (okf && okr) ? "" : "  VERIFY-FAIL");
        }
        if (okf && okr && tc < bt) { bt = tc; best = c; }
    }
    fft_apply(best);
    if (t_best) *t_best = bt;
    if (t_legacy) *t_legacy = tl;
    return best;
}

static std::vector<FftCfg> fft_candidates(bool full) {
    std::vector<FftCfg> v;
    const int lo = 2, hi = full ? 5 : 4;
    for (int tl = lo; tl <= hi; tl++)
        for (int bl = lo; bl <= hi; bl++) {
            v.push_back({true, tl, bl, 0});
            if (full) {
                v.push_back({true, tl, bl, (u64)1 << 16});
                v.push_back({true, tl, bl, (u64)1 << 12});
            }
        }
    return v;
}

// ---------------------------------------------------------------------------
// bench: component times at full scale + m recommendation

static int gpu_bench(u64 r, u64 s) {
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
    // error-checking costs at full scale (m = 24 here; the h-table pass
    // and the chain check scale with m, the product folds do not)
    G.ec_setup(true);
    G.ec_chain_reset();
    printf("error checking (per block unless noted):\n");
    double te_pre = timeit("EC h-table pass (m=24)", 10,
                           [&] { G.ec_block_pre(); });
    double te_fold = timeit("EC product + result folds", 10,
                            [&] { G.ec_after_mul(); G.ec_after_reduce(); });
    double te_ver = timeit("EC readback + host algebra", 10,
                           [&] { G.ec_block_verify(); });
    double te_win = timeit("EC chain check (per window)", 4,
                           [&] { G.ec_w = 1; G.ec_window_check(); });
    G.ec_on = false;
    printf("per-degree projection (m*t_sq + (t_mul + t_horner(m))/m), and "
           "the error-checking overhead at --ecw %ld:\n", g_ec.window);
    int best_m = 8;
    double best_c = 1e100;
    for (int m : {8, 12, 16, 20, 24, 32, 40, 48, 56, 64}) {
        double c = m * t_sq + (t_mul + t_hor * m / 24.0) / m;
        double ce = (te_pre * m / 24.0 + te_fold + te_ver +
                     te_win * m / 24.0 / (double)g_ec.window) / m;
        printf("  m=%-3d  %8.2f ms/degree   + EC %6.3f ms (%.2f%%)\n", m,
               c * 1e3, ce * 1e3, 100.0 * ce / c);
        if (c < best_c) { best_c = c; best_m = m; }
    }
    int mo = (int)(sqrt(t_mul / t_sq) + 0.5) & ~1;
    if (mo < 8) mo = 8;
    if (mo > 64) mo = 64;
    {   // the full FFT sweep (a run's start-up tuning times a subset)
        printf("FFT autotune (fwd+inv per config; ~1 min on GPU):\n");
        const bool sv_f = gf2c_use_fused;
        const int sv_t = gf2c_taylor_lv, sv_b = gf2c_bfly_lv;
        const u64 sv_c = gf2c_fuse_max_ch;
        double best_t = 0, t_legacy = 0;
        FftCfg b = fft_sweep(G, fft_candidates(true), 3, true, &best_t,
                             &t_legacy);
        if (b.fused)
            printf("best: fused tlv=%d blv=%d maxch=%" PRIu64
                   " -- fwd+inv %.1f ms (%.2fx vs legacy)\n", b.tlv, b.blv,
                   b.maxch, best_t * 1e3, t_legacy / best_t);
        else
            printf("best: legacy per-level -- fwd+inv %.1f ms\n",
                   t_legacy * 1e3);
        printf("  (a run tunes itself at start-up; to pin this choice: "
               "--fft %s)\n", b.fused ? "T,B[,C] as above" : "legacy");
        double t_mul2 = 1.5 * best_t + 0.003;       // 3 transforms + misc
        double bm = 8, bc = 1e100;
        for (int mm2 = 8; mm2 <= 96; mm2 += 2) {
            double c = mm2 * t_sq + (t_mul2 + t_hor * mm2 / 24.0) / mm2;
            if (c < bc) { bc = c; bm = mm2; }
        }
        printf("projected with tuned FFT: modmul ~%.0f ms, m=%d at "
               "~%.1f ms/degree\n", t_mul2 * 1e3, (int)bm, bc * 1e3);
        gf2c_use_fused = sv_f;       // the sweep must not leave its last
        gf2c_taylor_lv = sv_t;       // configuration behind for a
        gf2c_bfly_lv = sv_b;         // following --bench-gcd
        gf2c_fuse_max_ch = sv_c;
    }
    printf("recommended m: %d (analytic %d); expect ~%.1f ms/degree\n",
           best_m, mo, best_c * 1e3);
    G.fini();
    return 0;
}

// ---------------------------------------------------------------------------
// the GPU build's platform

struct GpuPlatform : TsPlatform {
    int device = -1;
    bool legacy = false;
    bool fft_pinned = false;
    FftCfg fft_pin = {true, GF2C_TAYLOR_LV_DEFAULT, GF2C_BFLY_LV_DEFAULT, 0};

    const char *prog() const override { return "tsfactor"; }
    const char *build() const override { return "GPU build"; }
    std::string fingerprint() override {
        cudaDeviceProp p;
        int dev = device >= 0 ? device : 0;
        char b[176];
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
            p.name[sizeof p.name - 1] = '\0';
            snprintf(b, sizeof b, "tsfactor %s on %.120s (sm_%d%d, %d SMs)",
                     TS_ROUND_STR, p.name, p.major, p.minor,
                     p.multiProcessorCount);
        } else {
            snprintf(b, sizeof b, "tsfactor %s on an unidentified GPU",
                     TS_ROUND_STR);
        }
        return b;
    }
    bool has_gpu_gcd() const override { return true; }
    bool parse_opt(const std::string &a, int &i, int argc,
                   char **argv) override {
        auto need = [&](const char *o) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", o);
                                 exit(1); }
            return argv[++i];
        };
        if (a == "--device") { device = atoi(need("--device")); return true; }
        if (a == "--legacy-sq") { legacy = true; return true; }
        if (a == "--fft") {
            std::string v = need("--fft");
            fft_pinned = true;
            if (v == "legacy") { fft_pin = {false, 0, 0, 0}; return true; }
            int t = 0, b = 0;
            unsigned long long c = 0;
            int n = sscanf(v.c_str(), "%d,%d,%llu", &t, &b, &c);
            if (n < 2 || t < 1 || t > 5 || b < 1 || b > 5) {
                fprintf(stderr, "bad --fft %s (legacy, or TLV,BLV[,MAXCH] "
                        "with levels 1..5)\n", v.c_str());
                exit(1);
            }
            fft_pin = {true, t, b, (u64)c};
            return true;
        }
        return false;
    }
    const char *usage_extra() const override {
        return "GPU build:\n"
               "  --device N      CUDA device index (multi-GPU: one process\n"
               "                  per GPU, each with its own --out)\n"
               "  --gpu-gcd       gcd engine with its big multiplications on\n"
               "                  the GPU (certified as usual)\n"
               "  --gpu-gcd-min-bits N   offload mults with an operand >= N\n"
               "  --fft legacy|TLV,BLV[,MAXCH]   pin the FFT configuration\n"
               "                  instead of timing it at start-up\n"
               "  --legacy-sq     use the 3-kernel squaring path\n";
    }
    void start() override { if (device >= 0) cudaSetDevice(device); }
    TsDev *make_dev() override {
        TsGPU *g = new TsGPU();
        g->legacy_sq = legacy;
        return g;
    }
    void defaults(TsTune &T) override {
        T.fft_fused = 1;
        T.fft_tlv = GF2C_TAYLOR_LV_DEFAULT;
        T.fft_blv = GF2C_BFLY_LV_DEFAULT;
        T.fft_maxch = 0;
        T.measured = 0;
    }
    bool pinned(TsTune &T) override {
        if (!fft_pinned) return false;
        T.fft_fused = fft_pin.fused ? 1 : 0;
        T.fft_tlv = (u32)fft_pin.tlv;
        T.fft_blv = (u32)fft_pin.blv;
        T.fft_maxch = fft_pin.maxch;
        T.measured = 0;
        return true;
    }
    void apply(const TsTune &T) override {
        fft_apply({T.fft_fused != 0, (int)T.fft_tlv, (int)T.fft_blv,
                   T.fft_maxch});
    }
    void tune(TsDev &Gd, TsTune &T, int verbose) override {
        TsGPU &G = static_cast<TsGPU &>(Gd);
        if (verbose) printf("FFT tuning (fwd+inv per configuration):\n");
        double bt = 0, tl = 0;
        FftCfg b = fft_sweep(G, fft_candidates(false), 2, verbose > 0, &bt,
                             &tl);
        T.fft_fused = b.fused ? 1 : 0;
        T.fft_tlv = (u32)b.tlv;
        T.fft_blv = (u32)b.blv;
        T.fft_maxch = b.maxch;
        T.measured = 1;
        ts_measure_m(*this, G, T);
    }
    std::string describe(const TsTune &T) override {
        char b[200];
        if (T.fft_fused)
            snprintf(b, sizeof b, "FFT fused tlv=%u blv=%u maxch=%" PRIu64
                     ", m=%" PRIu64 " (t_sq %.3f ms, t_mul %.1f ms)",
                     T.fft_tlv, T.fft_blv, T.fft_maxch, T.m, T.t_sq * 1e3,
                     T.t_mul * 1e3);
        else
            snprintf(b, sizeof b, "FFT legacy, m=%" PRIu64 " (t_sq %.3f ms, "
                     "t_mul %.1f ms)", T.m, T.t_sq * 1e3, T.t_mul * 1e3);
        return b;
    }
    int choose_m(double t_sq, double t_mul) override {
        int mo = (int)(sqrt(t_mul / t_sq) + 0.5) & ~1;
        return mo < 8 ? 8 : mo > 64 ? 64 : mo;
    }
    int selftest(int gcd_threads) override { return gpu_selftest(gcd_threads); }
    int bench(u64 r, u64 s, bool bench_gcd, bool gpu_gcd, u64 min_bits,
              bool ecg_xgcd, bool gcd_ntl) override {
        int rc = gpu_bench(r, s);
#ifdef HAVE_NTL
        if (!rc && bench_gcd) {
            CantorBasis bb;
            TsGpuMul gm;
            if (gpu_gcd) {
                bb.build();
                size_t ca = (size_t)((r + 31) >> 5);
                int fm = 1;
                while (((size_t)1 << fm) < 2 * ca - 1) fm++;
                gm.init(&bb, fm);
                g_gm = &gm;
                g_use_hybrid = true;
                if (min_bits) hyb_gpu_min_bits = min_bits;
                hyb_reset_stats();
            }
            ts_bench_gcd(r, s, ecg_xgcd, gcd_ntl);
            if (gpu_gcd) { g_gm = nullptr; gm.fini(); }
        }
#else
        (void)bench_gcd; (void)gpu_gcd; (void)min_bits; (void)ecg_xgcd;
        (void)gcd_ntl;
#endif
        return rc;
    }
    void gcd_hook_init(TsDev &Gd, u64 min_bits) override {
#ifdef HAVE_NTL
        TsGPU &G = static_cast<TsGPU &>(Gd);
        g_ts_gm.init(&G.Bb, G.fftm);
        g_gm = &g_ts_gm;
        if (min_bits) hyb_gpu_min_bits = min_bits;
        fprintf(stderr, "GPU-assisted GCD: on (GPU mults >= %" PRIu64
                " bits; NTL finish <= %" PRIu64 " bits)\n",
                hyb_gpu_min_bits, hyb_ntl_finish_bits);
        hyb_reset_stats();
#else
        (void)Gd; (void)min_bits;
#endif
    }
    void gcd_hook_fini() override {
#ifdef HAVE_NTL
        g_ts_gm.fini();
#endif
    }
    u64 default_G0() const override { return 1500; }
    void mem_report(TsDev &Gd, const SParams &P, bool gpu_gcd) override {
        TsGPU &G = static_cast<TsGPU &>(Gd);
        double mb = 1.0 / (1024 * 1024);
        double htab = 2.0 * G.m * G.nw * 8 * mb;
        double fft = (2.0 * G.F.n + G.F.n) * 8 * mb;   // dFA+dFB+twiddles
        double scr = (4.0 * G.nw + 2.0 * G.nw2 + G.nw) * 8 * mb;
        double ecm = g_ec.on ? 5.0 * G.nw * 8 * mb : 0;
        fprintf(stderr, "GPU memory (fixed at startup, independent of "
                "q/depth): h-tables %.0f MB, FFT %.0f MB, scratch %.0f MB, "
                "error checking %.0f MB%s\n", htab, fft, scr, ecm,
                gpu_gcd ? ", +gcd hook ~610 MB" : "");
        double snap = (double)G.m * G.nw * 8 * mb;
        long nsn = (P.hsnap_mb <= 0) ? 0 : (long)(P.hsnap_mb / snap);
        if (P.hsnap_mb > 0 && nsn < 1) nsn = 1;
        fprintf(stderr, "host memory: h-table snapshot %.0f MB each, up to "
                "%ld kept (--hsnap-mb %ld)%s\n", snap, nsn, P.hsnap_mb,
                nsn >= P.pend_max + 1 ? "" :
                "  [< pend-max+1: a deep hit may need the slow rebuild]");
    }
};

int main(int argc, char **argv) {
    GpuPlatform PL;
    return ts_main(argc, argv, PL);
}

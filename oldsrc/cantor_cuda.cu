// cantor_cuda.cu
// CUDA implementation of GF(2)[x] multiplication via the additive FFT
// over GF(2^64) on a Cantor basis.  The device arithmetic (cl32 via
// masked integer multiplies, Karatsuba cl64, reduction) comes from
// gf2_cantor_core.h -- the SAME code validated on CPU by cantor_ref
// against bit-serial, schoolbook, word-comb and residue oracles.  The
// host reference engine (gf2_cantor_engine.h) is compiled into this
// binary too, so `selftest` checks every GPU stage against the
// validated CPU engine on this machine, end to end.
//
// Build:   nvcc -O3 -arch=native -I../common -o cantor_cuda cantor_cuda.cu
// Run:     ./cantor_cuda selftest              (GPU vs CPU engine, many sizes)
//          ./cantor_cuda bench [bits]          (timed big multiply + residues,
//                                               default 136279841)
//
// Kernel structure mirrors the two phases of the host engine:
//   Phase 1  Taylor cascade: contiguous range-XORs (perfectly coalesced);
//            two dependent range ops per (depth, blocksize) pass, hence
//            two kernel launches per pass (~m^2/2 tiny launches total).
//   Phase 2  butterflies: one launch per depth, thread per pair; for
//            depth L >= 5 consecutive threads touch consecutive words.
// The twiddle table W (n/2 elements, shared by all depths) is built on
// the host from the validated basis and copied once.

#include "gf2_cantor_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <vector>
#include <chrono>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// full multiply on GPU; a,b,out are HOST arrays
static void gpu_mul(const u64 *a, u64 abits, const u64 *b, u64 bbits,
                    u64 *out, const CantorBasis &Bb, bool verbose) {
    size_t ca = (size_t)((abits + 31) >> 5), cb = (size_t)((bbits + 31) >> 5);
    int m = 1;
    while (((size_t)1 << m) < ca + cb - 1) m++;
    size_t n = (size_t)1 << m;
    size_t wa = bits_to_words(abits), wb = bits_to_words(bbits);
    size_t ow = mul_out_words(abits, bbits);

    CudaFFT F;
    F.init(m, Bb);
    u64 *dA, *dB, *dFA, *dFB, *dOut;
    CUCHK(cudaMalloc(&dA, wa * sizeof(u64)));
    CUCHK(cudaMalloc(&dB, wb * sizeof(u64)));
    CUCHK(cudaMalloc(&dFA, n * sizeof(u64)));
    CUCHK(cudaMalloc(&dFB, n * sizeof(u64)));
    CUCHK(cudaMalloc(&dOut, ow * sizeof(u64)));
    CUCHK(cudaMemcpy(dA, a, wa * sizeof(u64), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, b, wb * sizeof(u64), cudaMemcpyHostToDevice));

    int blocks, threads;
    CudaFFT::launch_dims(n, blocks, threads);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    k_pack<<<blocks, threads>>>(dA, wa, dFA, ca, n);
    k_pack<<<blocks, threads>>>(dB, wb, dFB, cb, n);
    F.fwd(dFA);
    F.fwd(dFB);
    k_pointwise<<<blocks, threads>>>(dFA, dFB, n);
    F.inv(dFA);
    CudaFFT::launch_dims(ow, blocks, threads);
    k_overlap_add<<<blocks, threads>>>(dFA, ca + cb - 1, dOut, ow);

    cudaEventRecord(ev1);
    CUCHK(cudaEventSynchronize(ev1));
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (verbose)
        printf("GPU multiply (%" PRIu64 " x %" PRIu64 " bits, FFT 2^%d): %.1f ms\n",
               abits, bbits, m, ms);

    CUCHK(cudaMemcpy(out, dOut, ow * sizeof(u64), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dFA); cudaFree(dFB); cudaFree(dOut);
    F.fini();
}

// ---------------------------------------------------------------------------
static u64 rng_state = 0x9E3779B97F4A7C15ULL;
static u64 rnd() {
    u64 x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}
static void rand_bits(std::vector<u64> &v, u64 bits) {
    v.assign(bits_to_words(bits), 0);
    for (auto &w : v) w = rnd();
    if (bits & 63) v.back() &= (~0ULL) >> (64 - (bits & 63));
}
static bool eq_words(const u64 *x, const u64 *y, size_t w) {
    return memcmp(x, y, w * sizeof(u64)) == 0;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "selftest";
    CantorBasis B;
    B.build();
    if (!B.verify() || B.len < 28) { fprintf(stderr, "basis failure\n"); return 1; }

    if (!strcmp(mode, "selftest")) {
        int bad = 0;
        for (int fp = 0; fp <= 1; fp++) {
        gf2c_use_fused = (fp == 1);
        printf("--- %s transform path ---\n", fp ? "FUSED" : "LEGACY");
        // stage 1: raw transform vs host engine at many sizes
        for (int m = 1; m <= 20; m += (m < 10 ? 1 : 5)) {
            size_t n = (size_t)1 << m;
            std::vector<u64> h(n);
            for (auto &x : h) x = rnd();
            std::vector<u64> g = h, gpu(n);

            CantorFFT ref; ref.init(m, B);
            ref.fwd(g.data());

            CudaFFT F; F.init(m, B);
            u64 *df; CUCHK(cudaMalloc(&df, n * sizeof(u64)));
            CUCHK(cudaMemcpy(df, h.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
            F.fwd(df);
            CUCHK(cudaMemcpy(gpu.data(), df, n * sizeof(u64), cudaMemcpyDeviceToHost));
            bool okf = (gpu == g);
            F.inv(df);
            CUCHK(cudaMemcpy(gpu.data(), df, n * sizeof(u64), cudaMemcpyDeviceToHost));
            bool okr = (gpu == h);
            cudaFree(df); F.fini();
            printf("m=%2d  fwd %s  roundtrip %s\n", m, okf ? "PASS" : "FAIL",
                   okr ? "PASS" : "FAIL");
            if (!okf || !okr) bad++;
        }
        // stage 2: end-to-end multiplies vs host engine
        u64 sizes[][2] = {{1, 1}, {100, 77}, {4096, 4096}, {100000, 100000},
                          {1048576, 524288}, {8000000, 8000000}};
        for (auto &sz : sizes) {
            u64 ab = sz[0], bb = sz[1];
            std::vector<u64> a, b;
            rand_bits(a, ab); rand_bits(b, bb);
            size_t ow = mul_out_words(ab, bb);
            std::vector<u64> c1(ow), c2(ow);
            gpu_mul(a.data(), ab, b.data(), bb, c1.data(), B, false);
            gf2x_fft_mul(a.data(), ab, b.data(), bb, c2.data(), B);
            bool ok = eq_words(c1.data(), c2.data(), ow);
            printf("mul %8" PRIu64 " x %8" PRIu64 " bits: %s\n", ab, bb,
                   ok ? "PASS" : "FAIL");
            if (!ok) bad++;
        }
        }
        gf2c_use_fused = true;
        printf(bad ? "\nGPU SELFTEST: %d FAILURES\n" : "\nGPU SELFTEST: ALL PASS\n", bad);
        return bad ? 1 : 0;
    }

    if (!strcmp(mode, "bench")) {
        u64 nb = 136279841ULL;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "legacy") || !strcmp(argv[i], "--legacy"))
                gf2c_use_fused = false;     // A/B against per-level kernels
            else
                nb = strtoull(argv[i], 0, 10);
        }
        printf("transform path: %s\n", gf2c_use_fused ? "FUSED" : "LEGACY");
        std::vector<u64> a, b;
        rand_bits(a, nb); rand_bits(b, nb);
        size_t ow = mul_out_words(nb, nb);
        std::vector<u64> out(ow);
        double t0 = now_s();
        gpu_mul(a.data(), nb, b.data(), nb, out.data(), B, true);
        printf("wall including transfers/alloc: %.2fs\n", now_s() - t0);
        bool ok = true;
        u64 seeds[3] = {0x1234567ULL, 0xDEADBEEFULL, 0xC0FFEEULL};
        for (int i = 0; i < 3; i++) {
            ModP p = find_irreducible_modp(61, seeds[i]);
            bool good = p.mul(p.residue(a.data(), nb), p.residue(b.data(), nb))
                        == p.residue(out.data(), 2 * nb - 1);
            printf("residue check mod 0x%" PRIx64 ": %s\n", p.q, good ? "PASS" : "FAIL");
            ok &= good;
        }
        return ok ? 0 : 1;
    }

    fprintf(stderr, "usage: %s selftest | bench [bits]\n", argv[0]);
    return 1;
}

// trinomial_stage2.cu
// GPU accelerator for the deep (stage-2) factor search on coarse-sieve
// survivors: Brent-style squares-and-products in
//     R = GF(2)[x] / (x^r + x^s + 1),   s <= (r-1)/2.
//
// Computes the Frobenius iterates F_k = x^(2^k) mod T and the interval
// product  ACC = prod_{k=k0..k1} (F_k + x) mod T.   Because
// gcd(x^(2^k) + x, T) collects every irreducible factor of T whose
// degree divides k, one CPU GCD of ACC against T per interval (gf2x /
// NTL FastGCD, exactly as in factor.cpp) reveals whether any factor
// with degree in the interval divides T; binary-searching back inside
// the interval isolates the k, after which the factor is
// gcd(F_k + x, T).  This program supplies the GPU-heavy parts
// (squarings and products) and writes ACC and F_k1 for the CPU GCD.
//
//   Squaring is linear over GF(2): a bit-spread (interleave zeros)
//   kernel, O(n) -- not an FFT multiply.
//   Reduction mod the trinomial is exactly two fold passes
//   (x^e -> x^(e-r) + x^(e-r+s)), race-free in gather form.
//   Products use the Cantor additive-FFT multiplier
//   (gf2_cantor_cuda.h), validated CPU-side at full 136279841-bit
//   scale by cantor_ref (residue checks mod independent irreducibles).
//
// Build:  nvcc -O3 -arch=native -I../common -o trinomial_stage2 trinomial_stage2.cu
// Modes:
//   ./trinomial_stage2 selftest
//        r=31,s=3    vs an independent u64 oracle, k = 1..40, and the
//                    known-answer Frobenius identity F_31 == x;
//        r=4423      vs a bit-serial naive square/mod oracle, and an
//                    interval product vs naive multiply+mod.
//   ./trinomial_stage2 run <r> <s> <k0> <k1> <out_prefix> [state_file]
//        writes <out_prefix>.acc  raw LE u64 words of ACC (r bits)
//               <out_prefix>.fk   raw LE u64 words of F_k1 (r bits)
//               <out_prefix>.meta text: r s k0 k1
//        checkpoints to state_file every 1024 squarings if given.
//   ./trinomial_stage2 bench <r> <s>
//        times one squaring+reduction and one product+reduction at
//        full scale with internal cross-check (spread-square vs
//        FFT-square).
//
// CPU follow-up (user's existing gf2x/NTL flow):
//   read <out>.acc (ceil(r/64) LE words = ACC), gcd(ACC, T) with
//   FastGCD; if nontrivial, re-run with narrower [k0,k1] to isolate k,
//   then gcd(F_k + x, T) gives the factor product for that k.

#include "gf2_cantor_cuda.h"
#include "gf2_small_field.h"
#include <cinttypes>
#include <chrono>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// spread 32 bits to 64 (interleave zeros): squaring of a GF(2)[x] poly
GF2C_HD u64 spread32(u32 v) {
    u64 x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

// out has 2*nw words; out[w] = spread of half (w&1) of in[w>>1]
__global__ void k_spread_square(const u64 *in, u64 *out, size_t nw2) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw2; w += stride) {
        u64 v = in[w >> 1];
        out[w] = spread32((u32)((w & 1) ? (v >> 32) : v));
    }
}

// 64-bit window of src (nw words) starting at bit pos, restricted to
// source bit indices in [lo, hi)
__device__ __forceinline__ u64 window(const u64 *src, size_t nw, u64 pos,
                                      u64 lo, u64 hi) {
    size_t w0 = (size_t)(pos >> 6);
    unsigned sh = (unsigned)(pos & 63);
    u64 a = (w0 < nw) ? src[w0] : 0;
    u64 b = (w0 + 1 < nw) ? src[w0 + 1] : 0;
    u64 v = sh ? ((a >> sh) | (b << (64 - sh))) : a;
    // clip to [lo, hi): window bit i is source bit pos+i
    u64 lc = (lo > pos) ? lo - pos : 0;
    u64 hc = (hi > pos) ? hi - pos : 0;
    if (lc >= 64 || hc == 0) return 0;
    if (hc > 64) hc = 64;
    u64 m = (hc >= 64 ? ~0ULL : ((1ULL << hc) - 1));
    if (lc) m &= ~((1ULL << lc) - 1);
    return v & m;
}

// one trinomial fold pass: dst = low(src) ^ (src[r..L) >> r) ^ (src[r..L) >> (r-s))
// dst is fully overwritten (gather form; dst != src)
__global__ void k_fold_pass(const u64 *src, size_t src_nw, u64 *dst,
                            size_t dst_nw, u64 r, u64 s, u64 L) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < dst_nw; w += stride) {
        u64 pos = (u64)w << 6;
        u64 low = window(src, src_nw, pos, 0, r);
        u64 t1 = window(src, src_nw, pos + r, r, L);          // -> e-r
        u64 t2 = window(src, src_nw, pos + (r - s), r, L);    // -> e-r+s
        dst[w] = low ^ t1 ^ t2;
    }
}

__global__ void k_xor_x(u64 *p) {          // p ^= x  (flip bit 1)
    if (blockIdx.x == 0 && threadIdx.x == 0) p[0] ^= 2ULL;
}

__global__ void k_set_x(u64 *p, size_t nw) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) p[w] = (w == 0) ? 2ULL : 0;
}

// ---------------------------------------------------------------------------
struct Stage2 {
    u64 r, s;
    size_t nw;                 // words for r bits
    size_t nw2;                // scratch words = 2*nw (>= words(2r))
    int m = 0;                 // FFT size for products
    CantorBasis Bb;
    CudaFFT F;
    u64 *dA = nullptr;         // current F_k (r bits)
    u64 *dACC = nullptr;       // accumulator (r bits)
    u64 *dT0 = nullptr;        // scratch 2r bits
    u64 *dT1 = nullptr;        // scratch 2r bits
    u64 *dFA = nullptr, *dFB = nullptr;   // FFT buffers

    void init(u64 r_, u64 s_) {
        r = r_; s = s_;
        if (!(s >= 1 && 2 * s <= r - 1 + 1)) {  // s <= (r-1)/2 required by 2-fold reduction
            if (2 * s > r) { fprintf(stderr, "need s <= (r-1)/2\n"); exit(1); }
        }
        nw = bits_to_words(r);
        nw2 = 2 * nw;              // spread output occupies exactly 2*nw words
        Bb.build();
        if (!Bb.verify()) { fprintf(stderr, "basis failure\n"); exit(1); }
        size_t ca = (size_t)((r + 31) >> 5);
        m = 1;
        while (((size_t)1 << m) < 2 * ca - 1) m++;
        F.init(m, Bb);
        CUCHK(cudaMalloc(&dA, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dACC, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dT0, nw2 * sizeof(u64)));
        CUCHK(cudaMalloc(&dT1, nw2 * sizeof(u64)));
        CUCHK(cudaMalloc(&dFA, F.n * sizeof(u64)));
        CUCHK(cudaMalloc(&dFB, F.n * sizeof(u64)));
    }
    void fini() {
        cudaFree(dA); cudaFree(dACC); cudaFree(dT0); cudaFree(dT1);
        cudaFree(dFA); cudaFree(dFB);
        F.fini();
    }

    // reduce dT0 (bit length L) mod trinomial -> dst (r bits used)
    void reduce(u64 *dst_rbits, u64 L) {
        int blocks, threads;
        CudaFFT::launch_dims(nw2, blocks, threads);
        // pass 1: dT0 -> dT1
        k_fold_pass<<<blocks, threads>>>(dT0, nw2, dT1, nw2, r, s, L);
        // pass 2: dT1 -> dT0 ; new length r + s - 1
        u64 L2 = (L > r) ? (r + s - 1) : L;
        k_fold_pass<<<blocks, threads>>>(dT1, nw2, dT0, nw2, r, s, L2);
        CUCHK(cudaMemcpy(dst_rbits, dT0, nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));
    }

    // dA <- dA^2 mod T (spread + two folds)
    void square_A() {
        int blocks, threads;
        CudaFFT::launch_dims(2 * nw, blocks, threads);
        k_spread_square<<<blocks, threads>>>(dA, dT0, nw2);
        reduce(dA, 2 * r - 1);
    }

    // dOut2r <- x * y (both r bits, device);  result left in dT0 (2r bits)
    void mul_dev(const u64 *dx, const u64 *dy) {
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
        k_overlap_add<<<blocks, threads>>>(dFA, 2 * ca - 1, dT0, nw2);
    }

    // dACC <- dACC * (dA + x) mod T
    void accumulate() {
        k_xor_x<<<1, 1>>>(dA);                    // dA + x
        mul_dev(dACC, dA);
        k_xor_x<<<1, 1>>>(dA);                    // restore dA
        reduce(dACC, 2 * r - 1);
    }
};

// ---------------------------------------------------------------------------
// CPU oracles for selftest
static void cpu_mod_trinomial(std::vector<u64> &v, u64 bits, u64 r, u64 s) {
    for (u64 e = bits; e-- > r;) {
        if (!((v[e >> 6] >> (e & 63)) & 1)) continue;
        v[e >> 6] ^= 1ULL << (e & 63);
        u64 e1 = e - r, e2 = e - r + s;
        v[e1 >> 6] ^= 1ULL << (e1 & 63);
        v[e2 >> 6] ^= 1ULL << (e2 & 63);
    }
}
static void cpu_square_mod(std::vector<u64> &a, u64 r, u64 s) {
    size_t ow = mul_out_words(r, r);
    std::vector<u64> sq(ow);
    gf2x_naive_mul(a.data(), r, a.data(), r, sq.data());
    cpu_mod_trinomial(sq, 2 * r - 1, r, s);
    a.assign(bits_to_words(r), 0);
    for (size_t i = 0; i < a.size(); i++) a[i] = sq[i];
}

// u64 oracle for r = 31, s = 3
static u64 sq31(u64 v) {
    u64 x = spread32((u32)v);                     // v < 2^31 -> deg <= 60
    while (x >> 31) {
        u64 hi = x >> 31;
        x = (x & 0x7FFFFFFFULL) ^ hi ^ (hi << 3);
    }
    return x;
}

static bool eqw(const u64 *a, const u64 *b, size_t w) {
    return memcmp(a, b, w * sizeof(u64)) == 0;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "selftest";

    if (!strcmp(mode, "selftest")) {
        int bad = 0;
        // ---- r = 31, s = 3 vs u64 oracle -------------------------------
        {
            Stage2 S;
            S.init(31, 3);
            int blocks, threads;
            CudaFFT::launch_dims(S.nw, blocks, threads);
            k_set_x<<<blocks, threads>>>(S.dA, S.nw);
            u64 oracle = 2;                        // x
            bool ok = true;
            u64 f31 = 0;
            for (int k = 1; k <= 40; k++) {
                S.square_A();
                oracle = sq31(oracle);
                u64 got = 0;
                CUCHK(cudaMemcpy(&got, S.dA, 8, cudaMemcpyDeviceToHost));
                if (got != oracle) ok = false;
                if (k == 31) f31 = got;
            }
            printf("r=31 s=3: squarings vs u64 oracle k=1..40          %s\n",
                   ok ? "PASS" : "FAIL");
            printf("r=31 s=3: Frobenius order (F_31 == x)              %s\n",
                   f31 == 2 ? "PASS" : "FAIL");
            if (!ok || f31 != 2) bad++;
            S.fini();
        }
        // ---- r = 4423, random s vs naive oracle ------------------------
        {
            u64 r = 4423, s = 1200;
            Stage2 S;
            S.init(r, s);
            int blocks, threads;
            CudaFFT::launch_dims(S.nw, blocks, threads);
            k_set_x<<<blocks, threads>>>(S.dA, S.nw);
            std::vector<u64> oracle(bits_to_words(r), 0);
            oracle[0] = 2;
            bool ok = true;
            std::vector<u64> got(S.nw);
            for (int k = 1; k <= 24; k++) {
                S.square_A();
                cpu_square_mod(oracle, r, s);
                CUCHK(cudaMemcpy(got.data(), S.dA, S.nw * 8,
                                 cudaMemcpyDeviceToHost));
                if (!eqw(got.data(), oracle.data(), S.nw)) ok = false;
            }
            printf("r=4423 s=1200: squarings vs naive oracle k=1..24   %s\n",
                   ok ? "PASS" : "FAIL");
            if (!ok) bad++;

            // interval product ACC = prod_{k=5..12}(F_k + x) vs naive
            k_set_x<<<blocks, threads>>>(S.dA, S.nw);
            std::vector<u64> oA(bits_to_words(r), 0), oACC(bits_to_words(r), 0);
            oA[0] = 2; oACC[0] = 1;
            // device ACC = 1
            CUCHK(cudaMemset(S.dACC, 0, S.nw * 8));
            u64 one = 1;
            CUCHK(cudaMemcpy(S.dACC, &one, 8, cudaMemcpyHostToDevice));
            for (int k = 1; k <= 12; k++) {
                S.square_A();
                cpu_square_mod(oA, r, s);
                if (k >= 5) {
                    S.accumulate();
                    // oracle: oACC *= (oA + x) mod T
                    std::vector<u64> term = oA;
                    term[0] ^= 2ULL;
                    size_t ow = mul_out_words(r, r);
                    std::vector<u64> prod(ow);
                    gf2x_naive_mul(oACC.data(), r, term.data(), r, prod.data());
                    cpu_mod_trinomial(prod, 2 * r - 1, r, s);
                    for (size_t i = 0; i < oACC.size(); i++) oACC[i] = prod[i];
                }
            }
            CUCHK(cudaMemcpy(got.data(), S.dACC, S.nw * 8,
                             cudaMemcpyDeviceToHost));
            bool okp = eqw(got.data(), oACC.data(), S.nw);
            printf("r=4423 s=1200: interval product k=5..12 vs naive   %s\n",
                   okp ? "PASS" : "FAIL");
            if (!okp) bad++;
            S.fini();
        }
        printf(bad ? "\nSTAGE2 SELFTEST: %d FAILURES\n"
                   : "\nSTAGE2 SELFTEST: ALL PASS\n", bad);
        return bad ? 1 : 0;
    }

    if (!strcmp(mode, "bench") && argc >= 4) {
        u64 r = strtoull(argv[2], 0, 10), s = strtoull(argv[3], 0, 10);
        Stage2 S;
        S.init(r, s);
        int blocks, threads;
        CudaFFT::launch_dims(S.nw, blocks, threads);
        k_set_x<<<blocks, threads>>>(S.dA, S.nw);
        // a few squarings to reach a generic element
        for (int k = 0; k < 8; k++) S.square_A();
        // cross-check: spread-square vs FFT-square of the same element
        std::vector<u64> a(S.nw);
        CUCHK(cudaMemcpy(a.data(), S.dA, S.nw * 8, cudaMemcpyDeviceToHost));
        S.mul_dev(S.dA, S.dA);
        S.reduce(S.dACC, 2 * r - 1);              // FFT-squared into dACC
        S.square_A();                             // spread-squared into dA
        std::vector<u64> v1(S.nw), v2(S.nw);
        CUCHK(cudaMemcpy(v1.data(), S.dA, S.nw * 8, cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(v2.data(), S.dACC, S.nw * 8, cudaMemcpyDeviceToHost));
        printf("cross-check spread-square == FFT-square: %s\n",
               eqw(v1.data(), v2.data(), S.nw) ? "PASS" : "FAIL");
        // timing
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int k = 0; k < 10; k++) S.square_A();
        cudaEventRecord(e1);
        CUCHK(cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        printf("squaring+reduction: %.2f ms each\n", ms / 10);
        cudaEventRecord(e0);
        for (int k = 0; k < 3; k++) S.accumulate();
        cudaEventRecord(e1);
        CUCHK(cudaEventSynchronize(e1));
        cudaEventElapsedTime(&ms, e0, e1);
        printf("product+reduction:  %.2f ms each\n", ms / 3);
        S.fini();
        return 0;
    }

    if (!strcmp(mode, "run") && argc >= 7) {
        u64 r = strtoull(argv[2], 0, 10), s = strtoull(argv[3], 0, 10);
        u64 k0 = strtoull(argv[4], 0, 10), k1 = strtoull(argv[5], 0, 10);
        const char *prefix = argv[6];
        const char *state_path = argc > 7 ? argv[7] : nullptr;

        Stage2 S;
        S.init(r, s);
        int blocks, threads;
        CudaFFT::launch_dims(S.nw, blocks, threads);
        u64 k_start = 0;
        bool resumed = false;
        if (state_path) {
            FILE *sf = fopen(state_path, "rb");
            if (sf) {
                u64 hdr[4];
                if (fread(hdr, 8, 4, sf) == 4 && hdr[0] == 0x53544732ULL
                    && hdr[1] == r && hdr[2] == s) {
                    k_start = hdr[3];
                    std::vector<u64> buf(S.nw);
                    if (fread(buf.data(), 8, S.nw, sf) == S.nw)
                        CUCHK(cudaMemcpy(S.dA, buf.data(), S.nw * 8,
                                         cudaMemcpyHostToDevice));
                    if (fread(buf.data(), 8, S.nw, sf) == S.nw)
                        CUCHK(cudaMemcpy(S.dACC, buf.data(), S.nw * 8,
                                         cudaMemcpyHostToDevice));
                    resumed = true;
                    fprintf(stderr, "resumed at k=%" PRIu64 "\n", k_start);
                }
                fclose(sf);
            }
        }
        if (!resumed) {
            k_set_x<<<blocks, threads>>>(S.dA, S.nw);
            CUCHK(cudaMemset(S.dACC, 0, S.nw * 8));
            u64 one = 1;
            CUCHK(cudaMemcpy(S.dACC, &one, 8, cudaMemcpyHostToDevice));
        }

        double t0 = now_s();
        for (u64 k = k_start + 1; k <= k1; k++) {
            S.square_A();
            if (k >= k0) S.accumulate();
            if (state_path && (k % 1024 == 0 || k == k1)) {
                CUCHK(cudaDeviceSynchronize());
                std::vector<u64> bufA(S.nw), bufC(S.nw);
                CUCHK(cudaMemcpy(bufA.data(), S.dA, S.nw * 8,
                                 cudaMemcpyDeviceToHost));
                CUCHK(cudaMemcpy(bufC.data(), S.dACC, S.nw * 8,
                                 cudaMemcpyDeviceToHost));
                FILE *sf = fopen(state_path, "wb");
                u64 hdr[4] = {0x53544732ULL, r, s, k};
                fwrite(hdr, 8, 4, sf);
                fwrite(bufA.data(), 8, S.nw, sf);
                fwrite(bufC.data(), 8, S.nw, sf);
                fclose(sf);
            }
            if (k % 4096 == 0)
                fprintf(stderr, "k=%" PRIu64 " (%.1f sq/s)\n", k,
                        (k - k_start) / (now_s() - t0));
        }
        CUCHK(cudaDeviceSynchronize());

        char path[1024];
        std::vector<u64> buf(S.nw);
        snprintf(path, sizeof path, "%s.acc", prefix);
        CUCHK(cudaMemcpy(buf.data(), S.dACC, S.nw * 8, cudaMemcpyDeviceToHost));
        FILE *f = fopen(path, "wb");
        fwrite(buf.data(), 8, S.nw, f);
        fclose(f);
        snprintf(path, sizeof path, "%s.fk", prefix);
        CUCHK(cudaMemcpy(buf.data(), S.dA, S.nw * 8, cudaMemcpyDeviceToHost));
        f = fopen(path, "wb");
        fwrite(buf.data(), 8, S.nw, f);
        fclose(f);
        snprintf(path, sizeof path, "%s.meta", prefix);
        f = fopen(path, "w");
        fprintf(f, "%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                r, s, k0, k1);
        fclose(f);
        fprintf(stderr, "done: %" PRIu64 " squarings, %.1fs; ACC and F_k1 "
                "written for CPU GCD\n", k1 - k_start, now_s() - t0);
        S.fini();
        return 0;
    }

    fprintf(stderr,
        "usage: %s selftest\n"
        "       %s run <r> <s> <k0> <k1> <out_prefix> [state_file]\n"
        "       %s bench <r> <s>\n", argv[0], argv[0], argv[0]);
    return 1;
}

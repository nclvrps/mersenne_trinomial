// coarse_sieve.cu
// GPU coarse sieve for smallest-degree irreducible factors of the
// trinomials  x^r + x^s + 1  over GF(2), all s in [1, smax], factor
// degrees d <= depth.
//
// This is a port of sieve_ref.cpp, which was validated two ways:
//   (1) against an independent brute-force oracle (direct irreducible
//       enumeration + per-polynomial evaluation) for three r values;
//   (2) against verified `factor` output for r = 136279841: all 999,999
//       lines of the s < 10^6 log reproduced exactly at depth 27
//       (935,227 factors matched in degree AND mask including every
//       tie-break; 64,772 deeper lines confirmed as sieve survivors).
// The device field arithmetic and the per-class number theory come from
// the same gf2_small_field.h header the reference uses.
//
// Method (see sieve_ref.cpp for the full derivation): for each degree d,
// work in GF(2^d) with a primitive polynomial q_d so g = x generates the
// multiplicative group of order M = 2^d - 1.  A degree-d irreducible p
// with root beta = g^j divides x^r + x^s + 1  iff  j*s == Z(j*r mod M)
// (mod M) where Z is the Zech logarithm -- an arithmetic progression in
// s of modulus n = M / gcd(j, M).  Only the minimal j of each cyclotomic
// coset is processed (conjugates give the same factor and progression);
// proper-subfield elements are skipped.  Each progression is marked into
// best[s] with atomicMin on the packed key (d << 48) | mask, which
// implements exactly "smallest degree, then lexicographically least
// mask" (equal-degree masks have equal hex length, so lexicographic
// order on the printed string is numeric order on the mask).
//
// Output: <prefix>.survivors.txt  s values with no factor of degree <=
//                                 depth (the deep-search worklist)
//         <prefix>.found.bin      (optional, --found) little-endian u64
//                                 packed keys for s = 1..smax, one per
//                                 s, UINT64_MAX where no factor found;
//                                 mask = key & 0xFFFFFFFFFFFF, d = key>>48
//
// Build:  nvcc -O3 -arch=native -I../common -o coarse_sieve coarse_sieve.cu
// Run:    ./coarse_sieve <r> <depth> <smax> <out_prefix> [--found]
//                        [--selftest]
// Sizing: device tables need 8*2^d bytes at degree d (8.6 GB at d=30,
//         17.2 GB at d=31) plus 8*(smax+1) bytes for best[] (545 MB at
//         smax = 68,139,920).  The program checks and reports before
//         each degree.
// Selftest: --selftest runs r=136279841 depth=20 smax=999999 entirely
//         on the GPU and compares every best[] entry against the CPU
//         reference computed in-process.

#include "gf2_small_field.h"
#include "gf2_field_setup.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <vector>
#include <chrono>

#define CUCHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
            __FILE__, __LINE__); exit(1); } } while (0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

typedef unsigned long long ull;

// heavy-progression record for the second-pass marking kernel
struct Heavy { u64 s_first, n, packed; };

// classes whose progressions have more than this many marks are deferred
// to k_mark_heavy (one thread block per record) for load balance
#define INLINE_MARKS 1024
#define HEAVY_CAP    (1u << 22)

// ---------------------------------------------------------------------------
// kernels
// antilog table: A[i] = g^i computed from the precomputed squares
// gpow2[t] = g^(2^t); i's set bits select which squares to multiply.
__global__ void k_antilog(u32 *A, u64 M64, u32 q, int d, const u32 *gpow2) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < M64; i += stride) {
        u32 v = 1;
        u64 bits = i;
        int t = 0;
        while (bits) {
            if (bits & 1) v = gf_mul(v, gpow2[t], q, d);
            bits >>= 1;
            t++;
        }
        A[i] = v;
    }
}

__global__ void k_logscatter(const u32 *A, u32 *L, u64 M64) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < M64; i += stride) L[A[i]] = (u32)i;
}

// one thread per candidate j; does the full per-class computation and
// either marks inline or defers to the heavy queue
__global__ void k_classes(int d, u64 M64, u32 q, u64 rmod, u64 smax,
                          const u32 *A, const u32 *L,
                          const u64 *subq, int nsub,
                          ull *best, Heavy *heavy, unsigned *nheavy,
                          unsigned *overflow) {
    const u32 M = (u32)M64;
    u64 j64 = (u64)blockIdx.x * blockDim.x + threadIdx.x + 1;
    u64 stride = (u64)gridDim.x * blockDim.x;
    u32 P[40];                                    // minpoly scratch

    for (; j64 < M64; j64 += stride) {
        u32 j = (u32)j64;
        // skip proper-subfield elements
        bool skip = false;
        for (int k = 0; k < nsub; k++)
            if (j64 % subq[k] == 0) { skip = true; break; }
        if (skip) continue;
        // only the minimal element of each cyclotomic coset
        u32 jj = j;
        bool minimal = true;
        for (int i = 1; i < d; i++) {
            jj = rotl_d(jj, d, M);
            if (jj < j) { minimal = false; break; }
        }
        if (!minimal) continue;

        u64 u = (j64 * rmod) % M64;
        if (u == 0) continue;                     // beta^r = 1: no solutions
        u32 v = L[A[u] ^ 1u];                     // Zech logarithm

        u64 gg = gcd_u64(j64, M64);
        if ((u64)v % gg != 0) continue;
        u64 n = M64 / gg;
        u64 s0 = (inv_mod(j64 / gg, n) * ((u64)v / gg)) % n;

        u64 mask = minpoly_mask(j, d, M, q, A, P);
        // mask == 0 would mean non-GF(2) coefficients: impossible if the
        // tables are right; validated exhaustively on CPU.
        u64 packed = ((u64)d << 48) | mask;

        u64 s = (s0 == 0 ? n : s0);
        if (s > smax) continue;
        u64 count = (smax - s) / n + 1;
        if (count <= INLINE_MARKS) {
            for (; s <= smax; s += n) atomicMin(&best[s], (ull)packed);
        } else {
            unsigned idx = atomicAdd(nheavy, 1u);
            if (idx < HEAVY_CAP) heavy[idx] = {s, n, packed};
            else atomicAdd(overflow, 1u);
        }
    }
}

// one block per heavy record; threads stride the progression
__global__ void k_mark_heavy(const Heavy *heavy, unsigned nheavy, u64 smax,
                             ull *best) {
    unsigned rec = blockIdx.x;
    if (rec >= nheavy) return;
    u64 s0 = heavy[rec].s_first, n = heavy[rec].n;
    ull packed = (ull)heavy[rec].packed;
    for (u64 s = s0 + (u64)threadIdx.x * n; s <= smax; s += (u64)blockDim.x * n)
        atomicMin(&best[s], packed);
}

__global__ void k_fill(ull *p, u64 n, ull v) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = v;
}

// ---------------------------------------------------------------------------
static void launch_dims(u64 work, int &blocks, int &threads) {
    threads = 256;
    u64 b = (work + threads - 1) / threads;
    blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
}

// run the sieve for one degree; best[] stays resident on device
static void sieve_degree_gpu(int d, u64 r, u64 smax, ull *d_best,
                             Heavy *d_heavy, unsigned *d_nheavy,
                             unsigned *d_overflow) {
    const u64 M64 = ((u64)1 << d) - 1;
    const u32 q = find_primitive_poly(d);
    const u64 rmod = r % M64;
    if (gcd_u64(rmod ? rmod : M64, M64) != 1) {
        fprintf(stderr, "d=%d: gcd(r, 2^d-1) != 1; unsupported r\n", d);
        exit(1);
    }

    // subfield moduli
    std::vector<u64> subq;
    for (u64 p : factor_u64((u64)d)) {
        int e = d / (int)p;
        subq.push_back(M64 / (((u64)1 << e) - 1));
    }

    // g^(2^t) chain on host
    u32 gpow2[32];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);

    double t0 = now_s();
    u32 *dA, *dL, *dgp;
    u64 *dsub = nullptr;
    CUCHK(cudaMalloc(&dA, M64 * sizeof(u32)));
    CUCHK(cudaMalloc(&dL, (M64 + 1) * sizeof(u32)));
    CUCHK(cudaMalloc(&dgp, d * sizeof(u32)));
    CUCHK(cudaMemcpy(dgp, gpow2, d * sizeof(u32), cudaMemcpyHostToDevice));
    if (!subq.empty()) {
        CUCHK(cudaMalloc(&dsub, subq.size() * sizeof(u64)));
        CUCHK(cudaMemcpy(dsub, subq.data(), subq.size() * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }

    int blocks, threads;
    launch_dims(M64, blocks, threads);
    k_antilog<<<blocks, threads>>>(dA, M64, q, d, dgp);
    k_logscatter<<<blocks, threads>>>(dA, dL, M64);

    unsigned zero = 0;
    CUCHK(cudaMemcpy(d_nheavy, &zero, 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_overflow, &zero, 4, cudaMemcpyHostToDevice));
    k_classes<<<blocks, threads>>>(d, M64, q, rmod, smax, dA, dL, dsub,
                                   (int)subq.size(), d_best, d_heavy,
                                   d_nheavy, d_overflow);
    unsigned nheavy = 0, overflow = 0;
    CUCHK(cudaMemcpy(&nheavy, d_nheavy, 4, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(&overflow, d_overflow, 4, cudaMemcpyDeviceToHost));
    if (overflow) {
        fprintf(stderr, "d=%d: heavy queue overflow (%u); raise HEAVY_CAP\n",
                d, overflow);
        exit(1);
    }
    if (nheavy) {
        unsigned use = nheavy > HEAVY_CAP ? HEAVY_CAP : nheavy;
        k_mark_heavy<<<use, 256>>>(d_heavy, use, smax, d_best);
    }
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaGetLastError());

    cudaFree(dA); cudaFree(dL); cudaFree(dgp);
    if (dsub) cudaFree(dsub);
    fprintf(stderr, "d=%2d done: %u heavy progressions, %.2fs\n", d, nheavy,
            now_s() - t0);
}

static void run_gpu(u64 r, int depth, u64 smax, std::vector<ull> &best) {
    ull *d_best;
    Heavy *d_heavy;
    unsigned *d_nheavy, *d_overflow;
    CUCHK(cudaMalloc(&d_best, (smax + 1) * sizeof(ull)));
    CUCHK(cudaMalloc(&d_heavy, HEAVY_CAP * sizeof(Heavy)));
    CUCHK(cudaMalloc(&d_nheavy, 4));
    CUCHK(cudaMalloc(&d_overflow, 4));
    int blocks, threads;
    launch_dims(smax + 1, blocks, threads);
    k_fill<<<blocks, threads>>>(d_best, smax + 1, ~0ULL);

    for (int d = 2; d <= depth; d++) {
        size_t need = ((size_t)8 << d) + 16;
        size_t freeb = 0, totb = 0;
        cudaMemGetInfo(&freeb, &totb);
        if (need > freeb) {
            fprintf(stderr, "d=%d needs %.2f GB tables but only %.2f GB free;"
                    " stopping at depth %d\n", d, need / 1073741824.0,
                    freeb / 1073741824.0, d - 1);
            break;
        }
        sieve_degree_gpu(d, r, smax, d_best, d_heavy, d_nheavy, d_overflow);
    }

    best.resize(smax + 1);
    CUCHK(cudaMemcpy(best.data(), d_best, (smax + 1) * sizeof(ull),
                     cudaMemcpyDeviceToHost));
    cudaFree(d_best); cudaFree(d_heavy); cudaFree(d_nheavy);
    cudaFree(d_overflow);
}

// ---------------------------------------------------------------------------
// CPU reference (identical logic to sieve_ref.cpp) for --selftest
static void sieve_cpu(u64 r, int depth, u64 smax, std::vector<ull> &best) {
    best.assign(smax + 1, ~0ULL);
    for (int d = 2; d <= depth; d++) {
        const u64 M64 = ((u64)1 << d) - 1;
        const u32 M = (u32)M64;
        const u32 q = find_primitive_poly(d);
        std::vector<u64> subq;
        for (u64 p : factor_u64((u64)d)) {
            int e = d / (int)p;
            subq.push_back(M64 / (((u64)1 << e) - 1));
        }
        std::vector<u32> A((size_t)M64), L((size_t)M64 + 1);
        u32 t = 1;
        for (u64 i = 0; i < M64; i++) { A[i] = t; t = gf_mulx(t, q, d); }
        for (u64 i = 0; i < M64; i++) L[A[i]] = (u32)i;
        const u64 rmod = r % M64;
        u32 P[40];
        for (u32 j = 1; j < M; j++) {
            bool skip = false;
            for (u64 sq_ : subq) if (j % sq_ == 0) { skip = true; break; }
            if (skip) continue;
            u32 jj = j; bool minimal = true;
            for (int i = 1; i < d; i++) {
                jj = rotl_d(jj, d, M);
                if (jj < j) { minimal = false; break; }
            }
            if (!minimal) continue;
            u64 u = ((u64)j * rmod) % M64;
            if (u == 0) continue;
            u32 v = L[A[u] ^ 1u];
            u64 gg = gcd_u64(j, M64);
            if ((u64)v % gg != 0) continue;
            u64 n = M64 / gg;
            u64 s0 = (inv_mod((u64)j / gg, n) * ((u64)v / gg)) % n;
            u64 mask = minpoly_mask(j, d, M, q, A.data(), P);
            u64 packed = ((u64)d << 48) | mask;
            for (u64 s = (s0 == 0 ? n : s0); s <= smax; s += n)
                if (packed < best[s]) best[s] = packed;
        }
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) {
        u64 r = 136279841, smax = 999999;
        int depth = 20;
        printf("selftest: r=%" PRIu64 " depth=%d smax=%" PRIu64
               " (GPU vs in-process CPU reference)\n", r, depth, smax);
        std::vector<ull> g, c;
        run_gpu(r, depth, smax, g);
        double t0 = now_s();
        sieve_cpu(r, depth, smax, c);
        fprintf(stderr, "cpu reference: %.2fs\n", now_s() - t0);
        u64 bad = 0;
        for (u64 s = 1; s <= smax; s++)
            if (g[s] != c[s] && ++bad <= 10)
                fprintf(stderr, "MISMATCH s=%" PRIu64 " gpu=%" PRIx64
                        " cpu=%" PRIx64 "\n", s, (u64)g[s], (u64)c[s]);
        printf("selftest: %s (%" PRIu64 " mismatches)\n",
               bad ? "FAIL" : "PASS", bad);
        return bad ? 1 : 0;
    }

    if (argc < 5) {
        fprintf(stderr, "usage: %s <r> <depth> <smax> <out_prefix> [--found]\n"
                        "       %s --selftest\n", argv[0], argv[0]);
        return 1;
    }
    u64 r = strtoull(argv[1], 0, 10);
    int depth = atoi(argv[2]);
    u64 smax = strtoull(argv[3], 0, 10);
    const char *prefix = argv[4];
    bool write_found = argc > 5 && !strcmp(argv[5], "--found");

    double t0 = now_s();
    std::vector<ull> best;
    run_gpu(r, depth, smax, best);
    fprintf(stderr, "total sieve time: %.2fs\n", now_s() - t0);

    char path[1024];
    snprintf(path, sizeof path, "%s.survivors.txt", prefix);
    FILE *fs = fopen(path, "w");
    if (!fs) { perror(path); return 1; }
    u64 nsurv = 0;
    for (u64 s = 1; s <= smax; s++)
        if (best[s] == ~0ULL) { fprintf(fs, "%" PRIu64 "\n", s); nsurv++; }
    fclose(fs);
    fprintf(stderr, "%" PRIu64 " survivors (no factor of degree <= %d) -> %s\n",
            nsurv, depth, path);

    if (write_found) {
        snprintf(path, sizeof path, "%s.found.bin", prefix);
        FILE *ff = fopen(path, "wb");
        if (!ff) { perror(path); return 1; }
        fwrite(best.data() + 1, sizeof(ull), smax, ff);
        fclose(ff);
        fprintf(stderr, "packed keys for s=1..%" PRIu64 " -> %s\n", smax, path);
    }
    return 0;
}

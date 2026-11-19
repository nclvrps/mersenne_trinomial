// randread.cu (round 9): how many random 8-byte reads per second can this
// GPU's DRAM serve?  coarse_sieve's RR engine makes two such reads per Zech
// log (the quadratic tower one), at random places in a table of 64 MB - 2 GB;
// this measures the ceiling those reads face, alone and beside streaming
// traffic shaped like the batch arrays' local memory (~70-80% of the DRAM
// bytes in the round-7 profiles).
//
//   nvcc -O3 -arch=native -o randread randread.cu
//   ./randread [max_table_MB]          (default 2048; well under a minute)
//
// Each thread of the random part runs a private xorshift and keeps R
// independent reads in flight (R = 1, 2, 4, 8 per loop iteration), XOR-ing
// them into a register stored at the end (so nothing is optimized away).
// Reads go past L1 (__ldcg), as coarse_sieve's do; a table far larger than
// L2 makes each one a DRAM access of a 32-byte sector.  In the mixed test,
// half the blocks instead stream through their own slices of a separate
// 512 MB buffer, 256 B per warp access (8 B per lane, read then written
// back: local memory's layout), until the random blocks are done.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

typedef unsigned long long u64;

template <int R>
__global__ void k_rand(const u64 *__restrict__ tab, u64 mask, int iters,
                       u64 *out, int sblocks, u64 *sbuf, u64 slice,
                       unsigned *done, unsigned *passes) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 acc = 0;
    if ((int)blockIdx.x < sblocks) {            // streaming, until done
        const u64 warp = tid / 32, lane = tid & 31;
        u64 *p = sbuf + warp * slice + lane;
        const unsigned nrand = gridDim.x - sblocks;
        unsigned np = 0;
        for (;;) {
            for (u64 i = 0; i + 32 <= slice; i += 32) {
                const u64 v = __ldcg(p + i);
                acc ^= v;
                p[i] = v + 1;
            }
            np++;
            if (*(volatile unsigned *)done >= nrand || np >= 100000) break;
        }
        if (threadIdx.x == 0) passes[blockIdx.x] = np;
    } else {
        u64 x = 0x9E3779B97F4A7C15ULL * (tid + 1);
        for (int it = 0; it < iters; it++) {
            u64 idx[R], v[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                idx[r] = x & mask;
            }
#pragma unroll
            for (int r = 0; r < R; r++) v[r] = __ldcg(tab + idx[r]);
#pragma unroll
            for (int r = 0; r < R; r++) acc ^= v[r];
        }
        __syncthreads();
        if (threadIdx.x == 0) { __threadfence(); atomicAdd(done, 1u); }
    }
    out[tid] = acc;
}

struct Ctx { int blocks, threads; u64 *out, *sbuf; u64 swords; unsigned *done, *passes; };

// G random reads/s; with sblocks > 0 also the streaming GB/s beside them
template <int R>
static double run(const Ctx &c, const u64 *tab, u64 mask, int iters, int sblocks,
                  double *sgbs) {
    const u64 nwarps = (u64)(sblocks ? sblocks : 1) * c.threads / 32;
    const u64 slice = c.swords / nwarps / 32 * 32;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int rep = 0; rep < 2; rep++) {         // warm-up, then timed
        CK(cudaMemset(c.done, 0, 4));
        CK(cudaEventRecord(a));
        k_rand<R><<<c.blocks, c.threads>>>(tab, mask, rep ? iters : 1, c.out, sblocks,
                                           c.sbuf, slice, c.done, c.passes);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaGetLastError());
    }
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    const double s = ms / 1e3;
    if (sgbs) {
        std::vector<unsigned> np(c.blocks);
        CK(cudaMemcpy(np.data(), c.passes, c.blocks * 4, cudaMemcpyDeviceToHost));
        double bytes = 0;
        for (int i = 0; i < sblocks; i++)
            bytes += (double)np[i] * (c.threads / 32) * slice * 8.0 * 2;   // read + write
        *sgbs = bytes / s / 1e9;
    }
    return (double)(c.blocks - sblocks) * c.threads * (double)iters * R / s / 1e9;
}

int main(int argc, char **argv) {
    const size_t max_mb = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2048;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0, mclk = 0, bus = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("%s: %d SMs, %.0f MHz, L2 %d MB, DRAM %d-bit, peak %.0f GB/s\n", p.name,
           p.multiProcessorCount, clk / 1e3, p.l2CacheSize >> 20, bus,
           2.0 * mclk * 1e3 * (bus / 8) / 1e9);
    Ctx c;
    c.threads = 256;
    int per = 1;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_rand<8>, c.threads, 0));
    c.blocks = p.multiProcessorCount * per;     // all resident at once
    CK(cudaMalloc(&c.out, (size_t)c.blocks * c.threads * sizeof(u64)));
    CK(cudaMalloc(&c.done, 4));
    CK(cudaMalloc(&c.passes, c.blocks * 4));
    c.swords = (u64)64 << 20;                   // 512 MB stream buffer
    CK(cudaMalloc(&c.sbuf, c.swords * 8));
    CK(cudaMemset(c.sbuf, 0, c.swords * 8));
    printf("%d blocks of %d threads (%d per SM)\n", c.blocks, c.threads, per * c.threads);
    double sga = 0;                             // every block streams, 8 passes
    {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        const u64 slice = c.swords / ((u64)c.blocks * c.threads / 32) / 32 * 32;
        CK(cudaMemset(c.done, 0xFF, 4));        // "done" at once: one pass each
        CK(cudaEventRecord(a));
        for (int i = 0; i < 8; i++)
            k_rand<1><<<c.blocks, c.threads>>>(c.sbuf, 0, 1, c.out, c.blocks, c.sbuf,
                                               slice, c.done, c.passes);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        sga = 8.0 * c.blocks * (c.threads / 32) * slice * 8.0 * 2 / (ms / 1e3) / 1e9;
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    printf("streaming alone (read + write, 256 B per warp access): %.0f GB/s\n", sga);
    for (size_t mb = 64; mb <= max_mb; mb *= 4) {
        size_t freeb = 0, totb = 0;
        CK(cudaMemGetInfo(&freeb, &totb));
        if (mb * 1048576 + ((size_t)256 << 20) > freeb) { printf("table %zu MB: no room\n", mb); break; }
        u64 *tab;
        const u64 n = (u64)mb * 1048576 / 8;
        CK(cudaMalloc(&tab, n * 8));
        CK(cudaMemset(tab, 0x5A, n * 8));
        const u64 mask = n - 1;
        const double r1 = run<1>(c, tab, mask, 256, 0, nullptr);
        const double r2 = run<2>(c, tab, mask, 128, 0, nullptr);
        const double r4 = run<4>(c, tab, mask, 64, 0, nullptr);
        const double r8 = run<8>(c, tab, mask, 32, 0, nullptr);
        printf("table %4zu MB: random 8-byte reads, G/s with 1/2/4/8 in flight per thread: "
               "%.2f %.2f %.2f %.2f  (= %.0f GB/s of 32-byte sectors at best)\n", mb,
               r1, r2, r4, r8, 32 * std::max(std::max(r1, r2), std::max(r4, r8)));
        double sgb = 0;
        const double rm = run<4>(c, tab, mask, 64, c.blocks / 2, &sgb);
        printf("               half the blocks streaming instead: random %.2f G/s beside %.0f GB/s "
               "of streaming\n", rm, sgb);
        CK(cudaFree(tab));
    }
    printf("(coarse_sieve, round 8, 2070 SUPER: RR at d = 45-53 makes ~1.7-1.9 G table reads/s, "
           "the classic tower at d=46 ~1.8 G/s, the subfield tower at d=44 ~1.3 G/s, beside "
           "~140-200 GB/s of local-memory traffic)\n");
    return 0;
}

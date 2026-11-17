// gf2_trinomial_cuda.h
// Shared CUDA machinery for arithmetic in GF(2)[x]/(x^r + x^s + 1):
// bit-spread squaring, two-pass trinomial fold reduction (gather form,
// race-free, valid for all 1 <= s <= (r-1)/2), FFT modular products via
// the Cantor multiplier, plus host-side naive oracles.  Extracted
// verbatim from the hardware-validated trinomial_stage2.cu; included by
// trinomial_stage2.cu and trinomial_scan.cu (one translation unit per
// binary -- __global__ definitions live here).
#ifndef GF2_TRINOMIAL_CUDA_H
#define GF2_TRINOMIAL_CUDA_H
#include "gf2_cantor_cuda.h"
#include <vector>
#include <chrono>

static inline double trin_now_s() {
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
struct TrinGPU {
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

    void init(u64 r_) {           // buffers and FFT depend only on r
        r = r_; s = 0;
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
    // s may change between uses of the same context (fold shift only)
    void set_s(u64 s_) {
        if (!(s_ >= 1 && 2 * s_ < r)) {       // s <= (r-1)/2 for odd r
            fprintf(stderr, "need 1 <= s <= (r-1)/2\n"); exit(1);
        }
        s = s_;
    }
    void reset_acc() {                        // ACC <- 1
        CUCHK(cudaMemset(dACC, 0, nw * sizeof(u64)));
        u64 one = 1;
        CUCHK(cudaMemcpy(dACC, &one, 8, cudaMemcpyHostToDevice));
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

#endif // GF2_TRINOMIAL_CUDA_H

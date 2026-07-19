// gf2_cantor_cuda.h
// CUDA kernels and orchestration for the Cantor additive FFT.  Included
// by cantor_cuda.cu (test driver) and trinomial_stage2.cu.  Single
// translation unit per binary: __global__ definitions live here.
#ifndef GF2_CANTOR_CUDA_H
#define GF2_CANTOR_CUDA_H
#include "gf2_cantor_core.h"
#include "gf2_cantor_engine.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef CUCHK
#define CUCHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
            __FILE__, __LINE__); exit(1); } } while (0)
#endif

// ---------------------------------------------------------------------------
// kernels
__global__ void k_xor_range(u64 *f, size_t dst, size_t src, size_t len) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < len; i += stride) f[dst + i] ^= f[src + i];
}

// Taylor pass, op1 for every block of one (depth, blocksize) layer:
// C ^= D for all blocks at once.  gidx enumerates all (block, k) pairs.
__global__ void k_taylor(u64 *f, int L, size_t S, size_t Bsz, int op,
                         size_t total) {
    // logical: quarter q = Bsz/4, half h = Bsz/2
    size_t q = Bsz >> 2, h = Bsz >> 1;
    size_t per_block = q << L;                    // physical elems per block op
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        size_t blk = i / per_block, k = i % per_block;
        size_t o = blk * Bsz;                     // logical block offset
        size_t dst, src;
        if (op == 0) { dst = (o + h) << L; src = (o + h + q) << L; }
        else         { dst = (o + q) << L; src = (o + h) << L; }
        f[dst + k] ^= f[src + k];
    }
}

__global__ void k_bfly(u64 *f, const u64 *W, int L, size_t half, int forward) {
    size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t mask = (((size_t)1) << L) - 1;
    for (; p < half; p += stride) {
        size_t t = p >> L, id = p & mask;
        size_t i1 = id + (t << (L + 1)), i2 = i1 + (((size_t)1) << L);
        u64 tw = W[t];
        u64 e1 = f[i1], e2 = f[i2];
        if (forward) bfly_fwd(e1, e2, tw);
        else         bfly_inv(e1, e2, tw);
        f[i1] = e1; f[i2] = e2;
    }
}

__global__ void k_pointwise(u64 *fa, const u64 *fb, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) fa[i] = gf64_mul_portable(fa[i], fb[i]);
}

// pack 32-bit chunks of the bit array into field elements
__global__ void k_pack(const u64 *a, size_t nwords, u64 *F, size_t nchunks,
                       size_t n) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; k < n; k += stride) {
        u64 v = 0;
        if (k < nchunks) {
            size_t pos = 32 * k, w = pos >> 6;
            if (w < nwords) v = (u32)(a[w] >> (pos & 63));
        }
        F[k] = v;
    }
}

// Overlap-add in GATHER form: output word w overlaps exactly chunks
// 2w-1, 2w, 2w+1 (64-bit elements at 32-bit stride), so each output
// word is computed and written by exactly one thread.  No prior zeroing
// needed: out[0..owords) is fully overwritten.
//   (History: an earlier scatter formulation XORed chunk k into words w
//   and w+1 with an even/odd phase split; but odd chunks k and k+2
//   share a word, so adjacent threads in the same launch raced and lost
//   updates on real GPUs -- caught by hardware selftest, invisible to
//   sequential emulation.)
__global__ void k_overlap_add(const u64 *F, size_t nchunks, u64 *out,
                              size_t owords) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < owords; w += stride) {
        u64 v = 0;
        size_t k = 2 * w;
        if (k < nchunks) v ^= F[k];                     // aligned chunk
        if (k + 1 < nchunks) v ^= F[k + 1] << 32;       // low half of odd chunk
        if (w && k - 1 < nchunks) v ^= F[k - 1] >> 32;  // high half of prior odd
        out[w] = v;
    }
}

__global__ void k_zero(u64 *p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = 0;
}

// ---------------------------------------------------------------------------
// host-side orchestration
struct CudaFFT {
    int m = 0;
    size_t n = 0;
    u64 *dW = nullptr;

    void init(int m_, const CantorBasis &B) {
        m = m_;
        n = (size_t)1 << m;
        CantorFFT ref;                 // reuse validated host twiddle build
        ref.init(m, B);
        CUCHK(cudaMalloc(&dW, (n >> 1) * sizeof(u64)));
        CUCHK(cudaMemcpy(dW, ref.W.data(), (n >> 1) * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }
    void fini() { if (dW) cudaFree(dW); dW = nullptr; }

    static void launch_dims(size_t work, int &blocks, int &threads) {
        threads = 256;
        size_t b = (work + threads - 1) / threads;
        blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
    }

    void taylor_dir(u64 *df, bool forward) const {
        int blocks, threads;
        if (forward) {
            for (int L = 0; L <= m - 2; L++) {
                size_t S = n >> L;
                for (size_t Bsz = S; Bsz >= 4; Bsz >>= 1) {
                    size_t nblk = S / Bsz, per = (Bsz >> 2) << L;
                    size_t total = nblk * per;
                    launch_dims(total, blocks, threads);
                    k_taylor<<<blocks, threads>>>(df, L, S, Bsz, 0, total);
                    k_taylor<<<blocks, threads>>>(df, L, S, Bsz, 1, total);
                }
            }
        } else {
            for (int L = m - 2; L >= 0; L--) {
                size_t S = n >> L;
                for (size_t Bsz = 4; Bsz <= S; Bsz <<= 1) {
                    size_t nblk = S / Bsz, per = (Bsz >> 2) << L;
                    size_t total = nblk * per;
                    launch_dims(total, blocks, threads);
                    k_taylor<<<blocks, threads>>>(df, L, S, Bsz, 1, total);
                    k_taylor<<<blocks, threads>>>(df, L, S, Bsz, 0, total);
                }
            }
        }
    }

    void fwd(u64 *df) const {
        taylor_dir(df, true);
        int blocks, threads;
        launch_dims(n >> 1, blocks, threads);
        for (int L = m - 1; L >= 0; L--)
            k_bfly<<<blocks, threads>>>(df, dW, L, n >> 1, 1);
    }
    void inv(u64 *df) const {
        int blocks, threads;
        launch_dims(n >> 1, blocks, threads);
        for (int L = 0; L <= m - 1; L++)
            k_bfly<<<blocks, threads>>>(df, dW, L, n >> 1, 0);
        taylor_dir(df, false);
    }
};


#endif // GF2_CANTOR_CUDA_H

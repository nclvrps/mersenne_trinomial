// gf2_cantor_cuda.h
// CUDA kernels and orchestration for the Cantor additive FFT.  Included
// by cantor_cuda.cu (test driver) and trinomial_stage2.cu.  Single
// translation unit per binary: __global__ definitions live here.
#ifndef GF2_CANTOR_CUDA_H
#define GF2_CANTOR_CUDA_H
#include "gf2_cantor_core.h"
#include "gf2_cantor_engine.h"
#include <cuda_runtime.h>
#include <vector>

// GF2C_LAUNCH lets the CPU-emulation shim intercept templated kernel
// launches (cuda_emul.h pre-defines it); real CUDA builds get <<<>>>.
#ifndef GF2C_LAUNCH
#define GF2C_LAUNCH(kern, blocks, threads, ...) \
    kern<<<blocks, threads>>>(__VA_ARGS__)
#endif
#if defined(__CUDACC__)
#define GF2C_UNROLL _Pragma("unroll")
#else
#define GF2C_UNROLL
#endif

// Fused (register-tile) transform path: default ON.  Toggled by the
// test drivers to A/B against the legacy per-level kernels.
static bool gf2c_use_fused = true;
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
    (void)S;
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
// FUSED register-tile kernels.
//
// At any depth L every Taylor-cascade op range is physically
// contiguous, so a group of g consecutive levels (blocksizes B_hi ..
// B_lo) acts inside contiguous windows of NS*CH elements, where
// CH = (B_lo/4)<<L is the op granularity and NS = 4*B_hi/B_lo = 2^(g+1)
// is the window size in CH-chunks ("slots").  The g levels form an
// NS-slot cascade applied elementwise over the CH columns of a window,
// so one THREAD per (window, column) holds all NS slot values in
// REGISTERS, applies every level locally, and touches global memory
// exactly once each way -- one pass instead of g, with perfectly
// coalesced accesses (warp lanes take consecutive columns) and NO
// cross-thread dependencies (hence no __syncthreads, and the CPU
// emulation harness validates these kernels exactly).  Butterfly
// depths fuse the same way (NS = 2^g slots, CH = 2^L_lo), with the
// twiddle index t = i1 >> (L+1) independent of the column e because
// e < CH = 2^L_lo <= 2^L.  For m = 24 this cuts Taylor passes from 276
// to 65 and butterfly passes from 24 to 5: ~2.8x less transform
// traffic.  GF2C_TAYLOR_LV/GF2C_BFLY_LV = levels per pass; 5 => 64
// resp. 32 u64 of registers per thread (reduce to 4 if a target
// architecture spills).

#ifndef GF2C_TAYLOR_LV
#define GF2C_TAYLOR_LV 5
#endif
#ifndef GF2C_BFLY_LV
#define GF2C_BFLY_LV 5
#endif

template <int NS>
__global__ void k_taylor_reg(u64 *f, u64 CH, u64 total, int inverse) {
    u64 col = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; col < total; col += stride) {
        u64 win = col / CH, e = col - win * CH;
        u64 base = win * (u64)NS * CH + e;
        u64 v[NS];
        GF2C_UNROLL
        for (int s = 0; s < NS; s++) v[s] = f[base + (u64)s * CH];
        if (!inverse) {
            GF2C_UNROLL
            for (int B = NS; B >= 4; B >>= 1) {
                const int q = B >> 2;
                GF2C_UNROLL
                for (int o = 0; o < NS; o += B) {
                    GF2C_UNROLL
                    for (int i = 0; i < q; i++) {
                        v[o + 2 * q + i] ^= v[o + 3 * q + i];
                        v[o + q + i] ^= v[o + 2 * q + i];
                    }
                }
            }
        } else {
            GF2C_UNROLL
            for (int B = 4; B <= NS; B <<= 1) {
                const int q = B >> 2;
                GF2C_UNROLL
                for (int o = 0; o < NS; o += B) {
                    GF2C_UNROLL
                    for (int i = 0; i < q; i++) {
                        v[o + q + i] ^= v[o + 2 * q + i];
                        v[o + 2 * q + i] ^= v[o + 3 * q + i];
                    }
                }
            }
        }
        GF2C_UNROLL
        for (int s = 0; s < NS; s++) f[base + (u64)s * CH] = v[s];
    }
}

template <int NS>
__global__ void k_bfly_reg(u64 *f, const u64 *W, u64 CH, u64 total, int Llo,
                           int forward) {
    const int LG = NS == 2 ? 1 : NS == 4 ? 2 : NS == 8 ? 3
                 : NS == 16 ? 4 : NS == 32 ? 5 : 6;
    u64 col = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; col < total; col += stride) {
        u64 win = col / CH, e = col - win * CH;
        u64 winbase = win * (u64)NS * CH;
        u64 base = winbase + e;
        u64 v[NS];
        GF2C_UNROLL
        for (int s = 0; s < NS; s++) v[s] = f[base + (u64)s * CH];
        if (forward) {
            GF2C_UNROLL
            for (int dl = LG - 1; dl >= 0; dl--) {
                const int D = 1 << dl;
                const int sh = 1;   // shift = Llo + dl + 1, split below
                (void)sh;
                GF2C_UNROLL
                for (int s = 0; s < NS; s++) {
                    if (s & D) continue;
                    u64 t = (winbase + (u64)s * CH) >> (Llo + dl + 1);
                    bfly_fwd(v[s], v[s + D], W[t]);
                }
            }
        } else {
            GF2C_UNROLL
            for (int dl = 0; dl <= LG - 1; dl++) {
                const int D = 1 << dl;
                GF2C_UNROLL
                for (int s = 0; s < NS; s++) {
                    if (s & D) continue;
                    u64 t = (winbase + (u64)s * CH) >> (Llo + dl + 1);
                    bfly_inv(v[s], v[s + D], W[t]);
                }
            }
        }
        GF2C_UNROLL
        for (int s = 0; s < NS; s++) f[base + (u64)s * CH] = v[s];
    }
}

struct TaylorGroup { int NS; u64 CH; u64 nwin; };
struct BflyGroup { int NS; u64 CH; u64 nwin; int Llo; };

// Partition the NL = m-L-1 cascade levels of depth L (blocksize
// exponents b = NL+1 .. 2) into groups of <= GF2C_TAYLOR_LV levels,
// remainder at the TOP; returned top-to-bottom (forward order).
static inline void taylor_groups(int m, size_t n, int L,
                                 std::vector<TaylorGroup> &out) {
    out.clear();
    int NL = m - L - 1;
    if (NL <= 0) return;
    int b_hi = NL + 1;
    int rem = NL % GF2C_TAYLOR_LV;
    int first = rem ? rem : GF2C_TAYLOR_LV;
    int done = 0;
    while (done < NL) {
        int lv = (done == 0) ? first : GF2C_TAYLOR_LV;
        int b_lo = b_hi - lv + 1;
        TaylorGroup g;
        g.NS = 1 << (lv + 1);
        g.CH = ((u64)1 << (b_lo - 2)) << L;
        g.nwin = (u64)n / ((u64)g.NS * g.CH);
        out.push_back(g);
        done += lv;
        b_hi = b_lo - 1;
    }
}

// Butterfly depths m-1 .. 0 in groups of <= GF2C_BFLY_LV, remainder at
// the bottom; returned top-to-bottom (forward order).
static inline void bfly_groups(int m, size_t n, std::vector<BflyGroup> &out) {
    out.clear();
    int Lhi = m - 1;
    while (Lhi >= 0) {
        int lv = (Lhi + 1 >= GF2C_BFLY_LV) ? GF2C_BFLY_LV : Lhi + 1;
        int Llo = Lhi - lv + 1;
        BflyGroup g;
        g.NS = 1 << lv;
        g.CH = (u64)1 << Llo;
        g.nwin = (u64)n / ((u64)g.NS * g.CH);
        g.Llo = Llo;
        out.push_back(g);
        Lhi = Llo - 1;
    }
}

static inline void fused_dims(u64 total, int &blocks, int &threads) {
    threads = 256;
    u64 b = (total + 255) / 256;
    blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
}

static inline void launch_taylor_reg(u64 *df, int NS, u64 CH, u64 nwin,
                                     int inverse) {
    u64 total = nwin * CH;
    int blocks, threads;
    fused_dims(total, blocks, threads);
    switch (NS) {
    case 4:  GF2C_LAUNCH(k_taylor_reg<4>,  blocks, threads, df, CH, total, inverse); break;
    case 8:  GF2C_LAUNCH(k_taylor_reg<8>,  blocks, threads, df, CH, total, inverse); break;
    case 16: GF2C_LAUNCH(k_taylor_reg<16>, blocks, threads, df, CH, total, inverse); break;
    case 32: GF2C_LAUNCH(k_taylor_reg<32>, blocks, threads, df, CH, total, inverse); break;
    case 64: GF2C_LAUNCH(k_taylor_reg<64>, blocks, threads, df, CH, total, inverse); break;
    default: fprintf(stderr, "bad taylor NS %d\n", NS); exit(1);
    }
}

static inline void launch_bfly_reg(u64 *df, const u64 *W, int NS, u64 CH,
                                   u64 nwin, int Llo, int forward) {
    u64 total = nwin * CH;
    int blocks, threads;
    fused_dims(total, blocks, threads);
    switch (NS) {
    case 2:  GF2C_LAUNCH(k_bfly_reg<2>,  blocks, threads, df, W, CH, total, Llo, forward); break;
    case 4:  GF2C_LAUNCH(k_bfly_reg<4>,  blocks, threads, df, W, CH, total, Llo, forward); break;
    case 8:  GF2C_LAUNCH(k_bfly_reg<8>,  blocks, threads, df, W, CH, total, Llo, forward); break;
    case 16: GF2C_LAUNCH(k_bfly_reg<16>, blocks, threads, df, W, CH, total, Llo, forward); break;
    case 32: GF2C_LAUNCH(k_bfly_reg<32>, blocks, threads, df, W, CH, total, Llo, forward); break;
    default: fprintf(stderr, "bad bfly NS %d\n", NS); exit(1);
    }
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

    void taylor_fused(u64 *df, bool forward) const {
        std::vector<TaylorGroup> gs;
        if (forward) {
            for (int L = 0; L <= m - 2; L++) {
                taylor_groups(m, n, L, gs);
                for (size_t i = 0; i < gs.size(); i++)
                    launch_taylor_reg(df, gs[i].NS, gs[i].CH, gs[i].nwin, 0);
            }
        } else {
            for (int L = m - 2; L >= 0; L--) {
                taylor_groups(m, n, L, gs);
                for (size_t i = gs.size(); i-- > 0;)
                    launch_taylor_reg(df, gs[i].NS, gs[i].CH, gs[i].nwin, 1);
            }
        }
    }
    void bfly_fused(u64 *df, bool forward) const {
        std::vector<BflyGroup> gs;
        bfly_groups(m, n, gs);
        if (forward) {
            for (size_t i = 0; i < gs.size(); i++)
                launch_bfly_reg(df, dW, gs[i].NS, gs[i].CH, gs[i].nwin,
                                gs[i].Llo, 1);
        } else {
            for (size_t i = gs.size(); i-- > 0;)
                launch_bfly_reg(df, dW, gs[i].NS, gs[i].CH, gs[i].nwin,
                                gs[i].Llo, 0);
        }
    }

    void fwd(u64 *df) const {
        if (gf2c_use_fused) {
            taylor_fused(df, true);
            bfly_fused(df, true);
            return;
        }
        taylor_dir(df, true);
        int blocks, threads;
        launch_dims(n >> 1, blocks, threads);
        for (int L = m - 1; L >= 0; L--)
            k_bfly<<<blocks, threads>>>(df, dW, L, n >> 1, 1);
    }
    void inv(u64 *df) const {
        if (gf2c_use_fused) {
            bfly_fused(df, false);
            taylor_fused(df, false);
            return;
        }
        int blocks, threads;
        launch_dims(n >> 1, blocks, threads);
        for (int L = 0; L <= m - 1; L++)
            k_bfly<<<blocks, threads>>>(df, dW, L, n >> 1, 0);
        taylor_dir(df, false);
    }
};


#endif // GF2_CANTOR_CUDA_H

/*
Copyright (C) 2026  Gord Palameta
(actual author: Claude AI)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
*/

/* irred_gpu.cu -- the GPU backend of irred (built as irred_gpu).

   irred.c is the driver (command line, error checking, checkpoints, map
   check, selftest); compiled with -DIRRED_GPU it calls the functions below
   for the work polynomials, which stay in GPU memory.  The verified state
   and everything written to disk live in host memory, as in the CPU build.

   The modulus is x^r + x^smod + 1 with smod = min(s, r - s) (the CPU build
   uses the odd one of s, r - s): the squaring kernels need smod <= (r-1)/2.
   For odd s < r/2 both builds use the same modulus, so their states,
   residues and checkpoints are interchangeable.

   Kernels: gf2_trin_sqr.cuh.  At start the available squaring kernels are
   timed at the actual size and the fastest is used (--sq-kernel to choose).

   Built with nvcc, or with g++ under the CUDA emulation (cuda_emul.h,
   -DIRRED_EMUL), which runs the kernels as loops on the CPU: slow, but it
   validates the code without a GPU. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef IRRED_EMUL
#define TRSQ_EMUL 1
#include "cuda_emul.h"
#define cudaDeviceScheduleBlockingSync 4
static inline cudaError_t cudaSetDeviceFlags(unsigned) { return 0; }
#else
#include <cuda_runtime.h>
#endif
#include "gf2_trin_sqr.cuh"

typedef uint64_t u64;             /* as in irred.c (trsq_u64 has the same width) */
#define TQ(p) ((trsq_u64 *)(p))
#define TQC(p) ((const trsq_u64 *)(p))

static void gpu_fail(const char *what, cudaError_t e)
{
    fprintf(stderr, "GPU error in %s: %s (code %d)\n"
            "The last checkpoint (if any) holds the last verified state; "
            "rerun the same command to resume.\n",
            what, cudaGetErrorString(e), (int)e);
    exit(EXIT_FAILURE);
}
#define CK(call) do { cudaError_t e_ = (call); \
                      if (e_ != cudaSuccess) gpu_fail(#call, e_); } while (0)

static int g_dev = -1;
static TrsqParams g_P;
static int g_kern = TRSQ_K2;
static size_t g_nw = 0;
static cudaEvent_t g_ev[2];
static unsigned g_launches = 0;
static double g_tsq[TRSQ_NKERN];

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

__global__ void k_add(u64 *d, const u64 *s, size_t n)
{
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < n; w += stride) d[w] ^= s[w];
}

__global__ void k_linbug(u64 *p)                /* the "map" fault */
{
    if (blockIdx.x == 0 && threadIdx.x == 0) p[3] ^= p[1];
}

static void dims(size_t work, int &blocks, int &threads)
{
    threads = 256;
    size_t b = (work + threads - 1) / threads;
    blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
}

extern "C" {

/* Selects and initialises the device; prints one line about it.  Returns
   0, or -1 (with a message) if there is no usable GPU. */
int irred_gpu_open(int device, int verbose)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
        fprintf(stderr, "No CUDA GPU found (%s).  Use the CPU build (irred), "
                "or irred_gpu --cpu.\n",
                e != cudaSuccess ? cudaGetErrorString(e) : "0 devices");
        return -1;
    }
    if (device < 0 || device >= n) {
        fprintf(stderr, "--device %d: there are %d CUDA device(s) (0..%d)\n",
                device, n, n - 1);
        return -1;
    }
    CK(cudaSetDevice(device));
    /* wait for the GPU by sleeping, not spinning (frees the core for other
       work); not fatal if refused */
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    (void)cudaGetLastError();
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, device));
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    printf("GPU %d: %s (sm_%d%d, %d SMs, %zu MB)\n", device, p.name, p.major,
           p.minor, p.multiProcessorCount, tot >> 20);
    (void)verbose;
    CK(cudaEventCreateWithFlags(&g_ev[0], cudaEventBlockingSync |
                                cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&g_ev[1], cudaEventBlockingSync |
                                cudaEventDisableTiming));
    g_dev = device;
    return 0;
}

/* The trinomial (r odd, 1 <= smod <= (r-1)/2); may be called again */
void irred_gpu_setup(int r, int smod)
{
    g_P = trsq_params((trsq_u64)r, (trsq_u64)smod);
    g_nw = g_P.nw;
}

int irred_gpu_nkernels(void) { return TRSQ_NKERN; }
const char *irred_gpu_kernel_name(int k) { return trsq_names[k]; }
int irred_gpu_kernel(void) { return g_kern; }
void irred_gpu_set_kernel(int k) { g_kern = k; }

u64 *irred_gpu_alloc(void)
{
    u64 *p = NULL;
    CK(cudaMalloc((void **)&p, (g_nw + TRSQ_PAD) * sizeof(u64)));
    CK(cudaMemset(p, 0, (g_nw + TRSQ_PAD) * sizeof(u64)));
    return p;
}

void irred_gpu_release(u64 *p)
{
    cudaFree(p);
}

/* dst = src^2 mod T.  Asynchronous; the host is kept at most 64 squarings
   ahead of the GPU (and sleeps while it waits). */
void irred_gpu_square(u64 *src, u64 *dst)
{
    trsq_square(g_kern, TQC(src), TQ(dst), g_P);
    CK(cudaGetLastError());
    if ((++g_launches & 31) == 0) {
        int e = (g_launches >> 5) & 1;
        CK(cudaEventRecord(g_ev[e]));
        if (g_launches > 32) CK(cudaEventSynchronize(g_ev[e ^ 1]));
    }
}

void irred_gpu_copy(u64 *dst, const u64 *src)
{
    CK(cudaMemcpy(dst, src, g_nw * sizeof(u64), cudaMemcpyDeviceToDevice));
}

void irred_gpu_add(u64 *dst, const u64 *src)
{
    int b, t;
    dims(g_nw, b, t);
    TRSQ_LAUNCH(k_add, b, t, dst, src, g_nw);
    CK(cudaGetLastError());
}

/* canonical words 0..nw-1 to / from the host */
void irred_gpu_get(const u64 *p, u64 *host)
{
    CK(cudaMemcpy(host, p, g_nw * sizeof(u64), cudaMemcpyDeviceToHost));
}

void irred_gpu_put(u64 *p, const u64 *host)
{
    u64 top = host[g_nw - 1] & g_P.topmask;
    if (g_nw > 1)
        CK(cudaMemcpy(p, host, (g_nw - 1) * sizeof(u64), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(p + g_nw - 1, &top, sizeof(u64), cudaMemcpyHostToDevice));
}

u64 irred_gpu_word0(const u64 *p)
{
    u64 w = 0;
    CK(cudaMemcpy(&w, p, sizeof(u64), cudaMemcpyDeviceToHost));
    return w;
}

void irred_gpu_flip(u64 *p, size_t w, int bit)
{
    u64 x = 0;
    CK(cudaMemcpy(&x, p + w, sizeof(u64), cudaMemcpyDeviceToHost));
    x ^= (u64)1 << bit;
    CK(cudaMemcpy(p + w, &x, sizeof(u64), cudaMemcpyHostToDevice));
}

void irred_gpu_linbug(u64 *p)
{
    TRSQ_LAUNCH(k_linbug, 1, 1, p);
    CK(cudaGetLastError());
}

/* Times every squaring kernel at the current size, keeps the fastest,
   and writes a one-line summary into buf. */
void irred_gpu_tune(char *buf, size_t len)
{
    u64 *a = irred_gpu_alloc(), *b = irred_gpu_alloc();
    u64 *h = (u64 *)malloc(g_nw * sizeof(u64));
    u64 x = 0x9E3779B97F4A7C15ULL;
    /* (xorshift64*) */
    for (size_t i = 0; i < g_nw; i++) {        /* a random input */
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        h[i] = x * 0x2545F4914F6CDD1DULL;
    }
    irred_gpu_put(a, h);
    int best = -1;
    for (int k = 0; k < TRSQ_NKERN; k++) {
        trsq_square(k, TQC(a), TQ(b), g_P);        /* warm-up */
        CK(cudaDeviceSynchronize());
        double t0 = now_s(), dt = 0;
        int reps = 0;
        do {                                      /* >= 0.05 s, <= 200 reps */
            trsq_square(k, TQC(a), TQ(b), g_P);
            trsq_square(k, TQC(b), TQ(a), g_P);
            reps += 2;
            CK(cudaDeviceSynchronize());
            dt = now_s() - t0;
        } while (dt < 0.05 && reps < 200);
        CK(cudaGetLastError());
        g_tsq[k] = dt / reps;
        if (best < 0 || g_tsq[k] < g_tsq[best]) best = k;
    }
    g_kern = best;
    size_t n = (size_t)snprintf(buf, len, "%s %.3f ms (", trsq_names[best],
                                1e3 * g_tsq[best]);
    for (int k = 0, first = 1; k < TRSQ_NKERN && n < len; k++) {
        if (k == best) continue;
        n += (size_t)snprintf(buf + n, len - n, "%s%s %.3f", first ? "" : ", ",
                              trsq_names[k], 1e3 * g_tsq[k]);
        first = 0;
    }
    if (n < len) snprintf(buf + n, len - n, ")");
    free(h);
    irred_gpu_release(a);
    irred_gpu_release(b);
}

/* waits for the GPU and reports any error */
void irred_gpu_sync(void)
{
    CK(cudaDeviceSynchronize());
}

} /* extern "C" */

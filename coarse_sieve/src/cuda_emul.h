// functional single-threaded CUDA emulation for host validation of kernels
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorNotReady 600
static inline const char* cudaGetErrorString(cudaError_t) { return "emul"; }
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
static inline cudaError_t cudaMalloc(void* p, size_t n) { *(void**)p = malloc(n); return *(void**)p ? 0 : 1; }
static inline cudaError_t cudaFree(void* p) { free(p); return 0; }
static inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { memcpy(d, s, n); return 0; }
static inline cudaError_t cudaMemset(void* d, int v, size_t n) { memset(d, v, n); return 0; }
static inline cudaError_t cudaDeviceSynchronize() { return 0; }
static inline cudaError_t cudaGetLastError() { return 0; }
static inline cudaError_t cudaSetDevice(int) { return 0; }
// test hooks: EMUL_NDEV=n pretends to have n devices (all the same host
// memory); EMUL_FREE_MB=m reports m MB free (default 64 GB) so the
// memory-dependent paths (coarse_sieve buckets / logs tables) can be hit
static inline cudaError_t cudaGetDeviceCount(int* n) { *n = getenv("EMUL_NDEV") ? atoi(getenv("EMUL_NDEV")) : 1; return 0; }
static inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) {
    size_t m = getenv("EMUL_FREE_MB") ? (size_t)atol(getenv("EMUL_FREE_MB")) << 20 : (size_t)64 << 30;
    *f = m; *t = m; return 0; }
// launches run to completion synchronously, so streams and events are
// always complete
static inline cudaError_t cudaStreamQuery(int) { return 0; }
typedef void* cudaEvent_t;
#define cudaEventDefault 0
#define cudaEventBlockingSync 1
#define cudaEventDisableTiming 2
static inline cudaError_t cudaEventCreate(cudaEvent_t*) { return 0; }
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = nullptr; return 0; }
static inline cudaError_t cudaEventRecord(cudaEvent_t) { return 0; }
static inline cudaError_t cudaEventRecord(cudaEvent_t, int) { return 0; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
static inline cudaError_t cudaEventElapsedTime(float* f, cudaEvent_t, cudaEvent_t) { *f = 0; return 0; }
static inline cudaError_t cudaEventDestroy(cudaEvent_t) { return 0; }
struct dim3s { unsigned x, y, z; };
static thread_local dim3s blockIdx{0,0,0}, blockDim{1,1,1}, threadIdx{0,0,0}, gridDim{1,1,1};
#define __global__
#define __shared__
static inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v) { auto o=*a; if (v<o) *a=v; return o; }
static inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { auto o=*a; *a+=v; return o; }
static inline unsigned atomicAdd(unsigned* a, unsigned v) { unsigned o=*a; *a+=v; return o; }
static inline unsigned long long atomicXor(unsigned long long* a, unsigned long long v) { auto o=*a; *a^=v; return o; }
static inline unsigned long long atomicCAS(unsigned long long* a, unsigned long long c, unsigned long long v) { auto o=*a; if (o==c) *a=v; return o; }
// launch emulation: kernel<<<B,T>>>(args) is sed-rewritten to
// EMUL_LAUNCH(kernel, B, T)(args)
template <class F> struct Launcher {
    unsigned B, T; F f;
    template <class... A> void operator()(A&&... a) {
        gridDim = {B,1,1}; blockDim = {T,1,1};
        for (blockIdx.x = 0; blockIdx.x < B; blockIdx.x++)
            for (threadIdx.x = 0; threadIdx.x < T; threadIdx.x++)
                f(a...);
    }
};
template <class F> Launcher<F> make_launcher(unsigned B, unsigned T, F f) { return {B, T, f}; }
#define EMUL_LAUNCH(fn, B, T) make_launcher((unsigned)(B), (unsigned)(T), fn)
// templated-kernel launch interception (see GF2C_LAUNCH in gf2_cantor_cuda.h)
#define GF2C_LAUNCH(kern, B, T, ...) \
    make_launcher((unsigned)(B), (unsigned)(T), kern)(__VA_ARGS__)

// functional single-threaded CUDA emulation for host validation of kernels
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;
#define cudaSuccess 0
static inline const char* cudaGetErrorString(cudaError_t) { return "emul"; }
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
static inline cudaError_t cudaMalloc(void* p, size_t n) { *(void**)p = malloc(n); return *(void**)p ? 0 : 1; }
static inline cudaError_t cudaFree(void* p) { free(p); return 0; }
static inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { memcpy(d, s, n); return 0; }
static inline cudaError_t cudaMemset(void* d, int v, size_t n) { memset(d, v, n); return 0; }
static inline cudaError_t cudaDeviceSynchronize() { return 0; }
static inline cudaError_t cudaGetLastError() { return 0; }
static inline cudaError_t cudaSetDevice(int) { return 0; }
static inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return 0; }
static inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { *f = (size_t)64<<30; *t = (size_t)64<<30; return 0; }
typedef void* cudaEvent_t;
static inline cudaError_t cudaEventCreate(cudaEvent_t*) { return 0; }
static inline cudaError_t cudaEventRecord(cudaEvent_t) { return 0; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
static inline cudaError_t cudaEventElapsedTime(float* f, cudaEvent_t, cudaEvent_t) { *f = 0; return 0; }
struct dim3s { unsigned x, y, z; };
static dim3s blockIdx{0,0,0}, blockDim{1,1,1}, threadIdx{0,0,0}, gridDim{1,1,1};
#define __global__
#define __shared__
static inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v) { auto o=*a; if (v<o) *a=v; return o; }
static inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { auto o=*a; *a+=v; return o; }
static inline unsigned atomicAdd(unsigned* a, unsigned v) { unsigned o=*a; *a+=v; return o; }
static inline unsigned long long atomicXor(unsigned long long* a, unsigned long long v) { auto o=*a; *a^=v; return o; }
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

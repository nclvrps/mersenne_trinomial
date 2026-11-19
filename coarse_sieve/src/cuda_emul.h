// functional single-threaded CUDA emulation for host validation of kernels
// (round 7: optionally with lockstep warps, EMUL_LOCKSTEP_WARPS)
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
static inline cudaError_t cudaGetDevice(int* d) { *d = 0; return 0; }
// launch-shape queries: a small pretend GPU (4 SMs, 2 resident blocks per SM
// for any kernel, 64 KB of shared memory per block) -- only the work split
// changes, never the results
enum cudaDeviceAttr { cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
                      cudaDevAttrMultiProcessorCount = 16 };
static inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
    *v = a == cudaDevAttrMultiProcessorCount ? 4
       : a == cudaDevAttrMaxSharedMemoryPerBlockOptin ? 65536 : 0; return 0; }
template <class F> static inline cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, F, int, size_t) { *n = 2; return 0; }
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
template <class F> static inline cudaError_t
cudaFuncSetAttribute(F, cudaFuncAttribute, int) { return 0; }
#define __launch_bounds__(...)
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
// (round 8) a load hint and the cycle counter: a plain load, and a host clock
template <class T> static inline T __ldcg(const T* p) { return *p; }
static inline long long clock64() {
    unsigned lo, hi;
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
#else
    lo = hi = 0;
#endif
    return (long long)(((unsigned long long)hi << 32) | lo);
}
// ---------------------------------------------------------------------------
#if defined(EMUL_LOCKSTEP_WARPS)
// Lockstep warps (coarse_sieve round 7; a program opts in by defining
// EMUL_LOCKSTEP_WARPS before this header), for warp-synchronous intrinsics
// (emul_ballot_sync, emul_shfl_sync).  The lanes of a warp run as fibers on
// the one host thread: a lane runs until it calls an intrinsic (or ends);
// once every lane of the warp has called it, the results are computed and
// the lanes go on.  Every live lane must call the same intrinsic with the
// full mask -- warp-uniform control flow, which the device code needs too;
// anything else stops the program with a message (so the emulator checks
// that property).  Kernels without intrinsics run lane after lane, as
// before, at the cost of two fiber switches per thread.
#include <cstdio>
#include <sys/mman.h>
#if defined(__x86_64__)
extern "C" void emul_ctx_switch(void **save_sp, void *load_sp);
__asm__(".pushsection .text\n"
        ".weak emul_ctx_switch\n"
        ".type emul_ctx_switch,@function\n"
        "emul_ctx_switch:\n"
        "  pushq %rbp\n  pushq %rbx\n  pushq %r12\n  pushq %r13\n"
        "  pushq %r14\n  pushq %r15\n"
        "  movq %rsp, (%rdi)\n"
        "  movq %rsi, %rsp\n"
        "  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n"
        "  popq %rbx\n  popq %rbp\n"
        "  ret\n"
        ".size emul_ctx_switch, .-emul_ctx_switch\n"
        ".popsection\n");
#else
#include <ucontext.h>                    // portable, slower
#endif
enum { EMUL_OP_BALLOT = 1, EMUL_OP_SHFL = 2 };
struct EmulLane {
    void *sp; int state;                 // 0 runnable, 1 at an intrinsic, 2 done
    int op, src; unsigned mask;
    unsigned long long arg, res;
#if !defined(__x86_64__)
    ucontext_t uc;
#endif
};
struct EmulWarp {
    EmulLane lane[32];
    int n, cur;
    void *sched_sp;
#if !defined(__x86_64__)
    ucontext_t sched_uc;
#endif
    void (*body)(void *); void *ctx;
    bool active;
};
static EmulWarp g_ew;
#define EMUL_FIBER_STACK ((size_t)2 << 20)
static char *g_ew_stack[32];
static inline void emul_to_sched(EmulLane &L) {
#if defined(__x86_64__)
    emul_ctx_switch(&L.sp, g_ew.sched_sp);
#else
    swapcontext(&L.uc, &g_ew.sched_uc);
#endif
}
static void emul_fiber_entry() {
    g_ew.body(g_ew.ctx);
    EmulLane &L = g_ew.lane[g_ew.cur];
    L.state = 2;
    emul_to_sched(L);
    abort();                             // (never resumed)
}
template <class B> static void emul_call_body(void *p) { (*static_cast<B *>(p))(); }
static void emul_warp_fail(const char *what) {
    fprintf(stderr, "emulated warp (block %u, threads %u..): %s\n", blockIdx.x,
            threadIdx.x & ~31u, what);
    abort();
}
// run the n lanes of one warp, threads tbase..tbase+n-1 of the block
static void emul_run_warp(void (*body)(void *), void *ctx, unsigned tbase, int n) {
    g_ew.body = body; g_ew.ctx = ctx; g_ew.n = n; g_ew.active = true;
    for (int l = 0; l < n; l++) {
        if (!g_ew_stack[l]) {             // stacks with a guard page below
            char *m = (char *)mmap(nullptr, EMUL_FIBER_STACK + 4096,
                                   PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED) { perror("emul fiber stack"); abort(); }
            mprotect(m, 4096, PROT_NONE);
            g_ew_stack[l] = m + 4096;
        }
        EmulLane &L = g_ew.lane[l];
        L.state = 0;
#if defined(__x86_64__)
        // first switch: pops six registers, then "returns" into the entry
        void **top = (void **)(g_ew_stack[l] + EMUL_FIBER_STACK);
        top[-1] = nullptr;               // (the entry never returns)
        top[-2] = (void *)&emul_fiber_entry;
        for (int i = 3; i <= 8; i++) top[-i] = nullptr;
        L.sp = (void *)(top - 8);
#else
        getcontext(&L.uc);
        L.uc.uc_stack.ss_sp = g_ew_stack[l];
        L.uc.uc_stack.ss_size = EMUL_FIBER_STACK;
        L.uc.uc_link = nullptr;
        makecontext(&L.uc, (void (*)())emul_fiber_entry, 0);
#endif
    }
    for (;;) {
        for (int l = 0; l < n; l++) {
            if (g_ew.lane[l].state != 0) continue;
            g_ew.cur = l;
            threadIdx.x = tbase + (unsigned)l;
#if defined(__x86_64__)
            emul_ctx_switch(&g_ew.sched_sp, g_ew.lane[l].sp);
#else
            swapcontext(&g_ew.sched_uc, &g_ew.lane[l].uc);
#endif
        }
        int waiting = 0, op = 0;
        unsigned want = 0;
        for (int l = 0; l < n; l++) {
            const EmulLane &L = g_ew.lane[l];
            if (L.state != 1) continue;
            if (!waiting) { op = L.op; want = L.mask; }
            else if (L.op != op || L.mask != want)
                emul_warp_fail("lanes at different warp intrinsics");
            waiting++;
        }
        if (!waiting) break;
        const unsigned all = n == 32 ? ~0u : (1u << n) - 1;
        if ((want & all) != want) emul_warp_fail("intrinsic mask names lanes the warp lacks");
        for (int l = 0; l < n; l++)
            if (((want >> l) & 1) && g_ew.lane[l].state != 1)
                emul_warp_fail("a lane named in the mask has exited");
        if (op == EMUL_OP_BALLOT) {
            unsigned b = 0;
            for (int l = 0; l < n; l++)
                if (g_ew.lane[l].state == 1 && g_ew.lane[l].arg) b |= 1u << l;
            for (int l = 0; l < n; l++)
                if (g_ew.lane[l].state == 1) g_ew.lane[l].res = b & want;
        } else {
            for (int l = 0; l < n; l++) {
                EmulLane &L = g_ew.lane[l];
                if (L.state != 1) continue;
                const int s = L.src & 31;
                L.res = (s < n && g_ew.lane[s].state == 1) ? g_ew.lane[s].arg : L.arg;
            }
        }
        for (int l = 0; l < n; l++) if (g_ew.lane[l].state == 1) g_ew.lane[l].state = 0;
    }
    g_ew.active = false;
}
static inline unsigned long long emul_warp_op(int op, unsigned mask,
                                              unsigned long long arg, int src) {
    if (!g_ew.active) { fprintf(stderr, "warp intrinsic outside a kernel\n"); abort(); }
    EmulLane &L = g_ew.lane[g_ew.cur];
    L.op = op; L.mask = mask; L.arg = arg; L.src = src; L.state = 1;
    emul_to_sched(L);
    return L.res;
}
static inline unsigned emul_ballot_sync(unsigned mask, int pred) {
    return (unsigned)emul_warp_op(EMUL_OP_BALLOT, mask, pred != 0, 0);
}
static inline unsigned long long emul_shfl_sync(unsigned mask, unsigned long long v,
                                                int src) {
    return emul_warp_op(EMUL_OP_SHFL, mask, v, src);
}

// launch emulation: kernel<<<B,T>>>(args) is sed-rewritten to
// EMUL_LAUNCH(kernel, B, T)(args)
template <class F> struct Launcher {
    unsigned B, T; F f;
    template <class... A> void operator()(A&&... a) {
        gridDim = {B,1,1}; blockDim = {T,1,1};
        auto body = [&]() { f(a...); };
        for (blockIdx.x = 0; blockIdx.x < B; blockIdx.x++)
            for (unsigned w0 = 0; w0 < T; w0 += 32)
                emul_run_warp(&emul_call_body<decltype(body)>, &body, w0,
                              (int)(T - w0 < 32 ? T - w0 : 32));
    }
};
#else
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
#endif
template <class F> Launcher<F> make_launcher(unsigned B, unsigned T, F f) { return {B, T, f}; }
// (a third launch parameter -- dynamic shared memory -- is accepted and
// ignored: kernels only use shared memory in their device-only code)
#define EMUL_LAUNCH(...) EMUL_LAUNCH_(__VA_ARGS__, 0)
#define EMUL_LAUNCH_(fn, B, T, ...) make_launcher((unsigned)(B), (unsigned)(T), fn)
// templated-kernel launch interception (see GF2C_LAUNCH in gf2_cantor_cuda.h)
#define GF2C_LAUNCH(kern, B, T, ...) \
    make_launcher((unsigned)(B), (unsigned)(T), kern)(__VA_ARGS__)

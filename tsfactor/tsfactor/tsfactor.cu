// tsfactor.cu
// Synthesis of gf2x/apps/factor.cpp (Brent/Gaudry/Thome/Zimmermann,
// GPL) and trinomial_scan.cu: finds the smallest-degree irreducible
// factor of x^r + x^s + 1 over GF(2) for each survivor s read from a
// file, using factor.cpp's multi-level blocking DDF with the
// squarings and the block modmuls on the GPU.
//
// The algorithm (decoded from factor.cpp; see tsfactor_handoff.md):
// maintain h[j] = e_{j+1}(y_0..y_{m-1}), the elementary symmetric
// functions of the Frobenius iterates y_t = x^(2^(i+t)) mod T covering
// the current block of m degrees [i, i+m-1].  Then
//     C = x^m + h[0] x^(m-1) + ... + h[m-1]
//       = prod_t (x + x^(2^(i+t)))            (mod T)
// captures exactly the degrees of the block (an irreducible of degree
// d divides x + x^(2^c) iff d | c); advancing the block by m degrees
// is h[j] <- h[j]^(2^m), i.e. m squarings per h (Frobenius commutes
// with the e_k).  Per degree: m linear-time squarings + 1/m FFT-size
// modmuls -- the same arithmetic balance as factor.cpp, with both
// legs on the GPU.  One gcd(A, T) per interval of q blocks (q grows
// linearly as in factor -f 1) runs on a CPU thread pool, fully
// overlapped with speculative scanning; verdicts are consumed
// strictly in interval order.  A hit is resolved by factoring the
// small interval gcd (CanZass); oversized gcds trigger factor.cpp's
// fineDDF-style interval bisection first.
//
// Output lines (results file and stdout), matching factor.cpp:
//     s d p<hex>     least-degree factor, numerically least mask
//     s u            gave up (-z tripped, or --maxd exceeded)
//     s primitive    no factor to the Swan bound => irreducible =>
//                    primitive for Mersenne-exponent r  (loudly)
//     s rA-B         pathological: factor with degree in [A,B],
//                    interval gcd not capturable (hand to factor.cpp)
//
// Build:   make tsfactor          (CUDA; NTL auto-detected, required
//                                  for production r)
//          make emul-tsfactor     (CPU emulation of the kernels)
// Test:    ./tsfactor --selftest
//
// Error checking (ON by default; --no-ec disables): every block is
// verified on the fly -- the h-table by a Frobenius-linear Gerbicz
// chain, the Horner build / FFT product / reduction by cyclic folds,
// the h-table initialisation from a closed form, and every transfer
// and RAM snapshot by a fold on both sides.  A failed check restarts
// the scan from the newest verified state; see gf2_ec.h for the
// mechanisms and tsfactor_handoff.md (round 7) for the integration.
// Bench:   ./tsfactor --bench <r> [s]
// Run:     ./tsfactor <r> <survivors.txt> --skip 36 [options]
//
// This program is free software, GPL v3 or later, and incorporates
// algorithmic structure from gf2x/apps/factor.cpp (GPL).

#include "gf2_trinomial_cuda.h"
#include "gf2_gcd.h"
#include "gf2_ec.h"
#include <cinttypes>
#include <algorithm>
#include <atomic>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <signal.h>
#include <cstdlib>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>

// ---------------------------------------------------------------------------
// new kernels (all gather-form: every output word is produced by one
// thread from reads only -> race-free and exact under the sequential
// CPU emulation harness)

// C[w] = XOR_j (h_j << (m-1-j))[w], h_j = hbase + j*nw, for w < nwc.
// Word w of (v << sh): (v[w] << sh) | (v[w-1] >> (64-sh)); v[i>=nw]=0.
__global__ void k_horner_fused(const u64 *hbase, size_t nw, int m,
                               u64 *C, size_t nwc) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nwc; w += stride) {
        u64 acc = 0;
        for (int j = 0; j < m; j++) {
            const u64 *h = hbase + (size_t)j * nw;
            unsigned sh = (unsigned)(m - 1 - j);
            u64 a = (w < nw) ? h[w] : 0;
            u64 b = (w >= 1 && w - 1 < nw) ? h[w - 1] : 0;
            acc ^= sh ? ((a << sh) | (b >> (64 - sh))) : a;
        }
        C[w] = acc;
    }
}

// tail of the fused Horner: fold bits [r, r+m-2] of C down
// (x^(r+e) = x^(s+e) + x^e), add the leading x^m term, and leave all
// bits >= r zero.  Single thread; <= 63 sequential iterations.
// `fired_out` (may be null) receives the mask of the e that were folded:
// it is the quotient of the unreduced Horner sum by T, which the error
// checker needs to predict phi(C) (gf2_ec.h, ecf_horner_predict).
__global__ void k_horner_tail(u64 *C, u64 r, u64 s, int m, u64 *fired_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    u64 fired = 0;
    for (int e = 0; e <= m - 2; e++) {
        u64 p = r + (u64)e;
        u64 bit = (C[p >> 6] >> (p & 63)) & 1;
        if (bit) {
            fired |= 1ULL << e;
            C[p >> 6] ^= 1ULL << (p & 63);
            u64 t1 = s + (u64)e;
            C[t1 >> 6] ^= 1ULL << (t1 & 63);
            C[(u64)e >> 6] ^= 1ULL << (e & 63);
        }
    }
    C[(u64)m >> 6] ^= 1ULL << (m & 63);         // + x^m  (m < r)
    if (fired_out) *fired_out = fired;
}

// dst ^= (src * x) mod (x^r + x^s + 1); dst != src; both nw words with
// bits >= r zero.  Wrap: if src bit r-1 is set, the shifted x^r term
// becomes x^s + 1.
__global__ void k_mulx_xor(u64 *dst, const u64 *src, size_t nw,
                           u64 r, u64 s) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    u64 topw = (r - 1) >> 6;
    unsigned topb = (unsigned)((r - 1) & 63);
    for (; w < nw; w += stride) {
        u64 a = src[w];
        u64 b = w ? src[w - 1] : 0;
        u64 v = (a << 1) | (b >> 63);
        // clear the shifted-out x^r bit if present in this word
        if (w == (r >> 6)) v &= ~(1ULL << (r & 63));
        u64 wrap = (src[topw] >> topb) & 1;      // coefficient of x^(r-1)
        if (w == 0) v ^= wrap;                   // + 1
        if (w == (s >> 6)) v ^= wrap << (s & 63);// + x^s
        dst[w] ^= v;
    }
}

// out = in^2 mod (x^r + x^s + 1), fused spread + two fold passes.
// Derivation (tsfactor_handoff.md / verified vs the 3-kernel path):
// with SP the virtual bit-spread of `in` (SP[2i] = in[i]) of length
// 2r-1, and V(pos,lo,hi) the 64-bit window of SP at [pos,pos+64)
// clipped to SP-indices [lo,hi):
//   out_word(w) = V(b,      0,    r   )
//               ^ V(b+r,    r,    2r-1)
//               ^ V(b+r-s,  r,    2r-s)
//               ^ V(b+2r-s, 2r-s, 2r-1)
//               ^ V(b+2r-2s,2r-s, 2r-1),   b = 64w.
// Requires only 1 <= s <= (r-1)/2 (no parity constraint).
__device__ __forceinline__ u64 ts_spread32(u32 v) {
    u64 x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}
__device__ __forceinline__ u64 ts_in32(const u64 *in, size_t nw, u64 q) {
    // 32-bit window of `in` at bit q (bits beyond the array are 0)
    size_t w0 = (size_t)(q >> 6);
    unsigned sh = (unsigned)(q & 63);
    u64 a = (w0 < nw) ? in[w0] : 0;
    u64 b = (w0 + 1 < nw) ? in[w0 + 1] : 0;
    u64 v = sh ? ((a >> sh) | (b << (64 - sh))) : a;
    return v & 0xFFFFFFFFULL;
}
__device__ __forceinline__ u64 ts_vspread(const u64 *in, size_t nw,
                                          u64 pos, u64 lo, u64 hi) {
    // window of the virtual spread at [pos, pos+64), clipped to [lo,hi)
    u64 v;
    if ((pos & 1) == 0)
        v = ts_spread32(( u32)ts_in32(in, nw, pos >> 1));
    else
        v = ts_spread32((u32)ts_in32(in, nw, (pos + 1) >> 1)) << 1;
    u64 lc = (lo > pos) ? lo - pos : 0;
    u64 hc = (hi > pos) ? hi - pos : 0;
    if (lc >= 64 || hc == 0) return 0;
    if (hc > 64) hc = 64;
    u64 msk = (hc >= 64 ? ~0ULL : ((1ULL << hc) - 1));
    if (lc) msk &= ~((1ULL << lc) - 1);
    return v & msk;
}
__global__ void k_square_fused(const u64 *in, u64 *out, size_t nw,
                               u64 r, u64 s) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) {
        u64 b = (u64)w << 6;
        u64 v = ts_vspread(in, nw, b,               0,         r);
        v ^= ts_vspread(in, nw, b + r,              r,         2*r - 1);
        v ^= ts_vspread(in, nw, b + r - s,          r,         2*r - s);
        v ^= ts_vspread(in, nw, b + 2*r - s,        2*r - s,   2*r - 1);
        v ^= ts_vspread(in, nw, b + 2*(r - s),      2*r - s,   2*r - 1);
        out[w] = v;
    }
}

__global__ void k_xor_buf(u64 *dst, const u64 *src, size_t n) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < n; w += stride) dst[w] ^= src[w];
}
__global__ void k_set_zero(u64 *p, size_t nw) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) p[w] = 0;
}
__global__ void k_set_one(u64 *p, size_t nw) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) p[w] = (w == 0) ? 1ULL : 0;
}

// ---------------------------------------------------------------------------
// error checking: configuration, event log, statistics, fault injection
// (the mechanisms themselves are described at the top of gf2_ec.h)

enum { ECC_GERBICZ, ECC_PRODUCT, ECC_REDUCE, ECC_INIT, ECC_IDLE, ECC_XFER,
       ECC_SNAP, ECC_GCD, ECC_FACTOR, ECC_N };
static const char *ecc_name[ECC_N] = {
    "h-table chain", "A*C product / Horner", "reduction mod T",
    "h-table initialisation", "idle h-table", "device<->host transfer",
    "host snapshot", "gcd", "factor verification" };

struct EcCfg {
    bool on = true;            // --no-ec
    long window = 8;           // --ecw: blocks per h-table chain check
    bool init_check = true;    // --eci0 disables the init verification
    bool gcd_cert = false;     // the gcd is CERTIFIED (gf2_gcd.h).  main()
                               // turns this on for every run with checking
                               // unless --gcd-ntl / --ecg-dup ask otherwise;
                               // the selftest sets it per suite
    bool gcd_dup = false;      // --ecg-dup: every gcd computed twice, compared
    bool timing = false;       // --ect: cost breakdown (adds device syncs)
    int max_retry = 3;         // failed restarts from one point before giving up
    std::string log_path;      // PREFIX.ec.log; empty = stderr only
};
static EcCfg g_ec;
static bool g_ec_quiet = false;          // selftest: count events, do not print

struct EcStats {
    std::atomic<u64> fails[ECC_N];
    std::atomic<u64> restarts{0}, xfer_retries{0}, gcd_reruns{0},
        repairs{0};
    std::atomic<u64> blocks{0}, windows{0};
    std::atomic<u64> gcd_cert{0}, gcd_uncert{0};   // certified / fell back
    // --ect accounting (scan thread only)
    double t_block = 0, t_host = 0, t_window = 0, t_init = 0, t_xfer = 0,
           t_verify = 0;
    u64 n_init = 0, n_xfer = 0, n_verify = 0;
    EcStats() { for (auto &f : fails) f = 0; }
    u64 total() const {
        u64 t = 0;
        for (auto &f : fails) t += f.load();
        return t;
    }
};
static EcStats g_ecs;
static std::mutex g_ec_log_mu;
static std::atomic<u64> g_hold_restores{0};   // fine DDF stepped back (selftest)

// Report an error-checking event: cat >= 0 is a detection in that category
// (counted), cat < 0 an informational note.  Always goes to stderr the
// moment it happens, and to PREFIX.ec.log so that an unattended run leaves
// a record.
static void ec_event(int cat, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void ec_event(int cat, const char *fmt, ...) {
    char msg[640];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (cat >= 0) g_ecs.fails[cat]++;
    if (g_ec_quiet) return;
    std::lock_guard<std::mutex> lk(g_ec_log_mu);
    if (cat >= 0) fprintf(stderr, "EC FAIL [%s] %s\n", ecc_name[cat], msg);
    else fprintf(stderr, "EC: %s\n", msg);
    fflush(stderr);
    if (!g_ec.log_path.empty()) {
        FILE *f = fopen(g_ec.log_path.c_str(), "a");
        if (f) {
            time_t now = time(nullptr);
            char ts[32];
            strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", localtime(&now));
            fprintf(f, "%s  %-24s %s\n", ts,
                    cat >= 0 ? ecc_name[cat] : "note", msg);
            fclose(f);
        }
    }
}

// Fault injection, for validating the checker itself: --ec-fault SITE:N
// flips one bit at the N-th opportunity of the named site (once), and the
// selftest sweeps every site.  N = -1 fires at EVERY opportunity, which
// models a persistent hardware fault.
enum { ECI_NONE = 0, ECI_H, ECI_HMID, ECI_A, ECI_C, ECI_T, ECI_RED, ECI_IDLE,
       ECI_INIT, ECI_XFER_H, ECI_XFER_A, ECI_SNAP, ECI_HOLD, ECI_JOB,
       ECI_GCD, ECI_N };
static const char *eci_name[ECI_N] = {
    "none", "h", "hmid", "a", "c", "t", "red", "idle", "init", "xferh",
    "xfera", "snap", "hold", "job", "gcd" };
static const char *eci_desc[ECI_N] = {
    "", "h-table entry after a block's squarings",
    "h-table entry between two squaring steps",
    "accumulator A while idle between blocks",
    "block polynomial C before the product",
    "unreduced FFT product", "reduced product",
    "h-table while idle between intervals",
    "h-table during initialisation", "h-table snapshot in transfer",
    "accumulator in transfer", "h-table snapshot in host RAM",
    "bisection hold table in VRAM", "accumulator queued for the gcd",
    "gcd result" };
struct EcInject {
    int site = ECI_NONE;
    long at = 0;                         // N; -1 = every opportunity
    std::atomic<long> ctr{0};
    std::atomic<long> fired{0};
    u64 seed = 1;
    void arm(int s, long n, u64 sd) { site = s; at = n; ctr = 0; fired = 0;
                                      seed = sd; }
    void disarm() { site = ECI_NONE; }
};
static EcInject g_eci;
static bool ec_inject_now(int site) {
    if (g_eci.site != site) return false;
    long c = ++g_eci.ctr;
    if (g_eci.at >= 0 && (c != g_eci.at || g_eci.fired.load())) return false;
    g_eci.fired++;
    return true;
}
static u64 ec_inject_rnd(u64 n) {        // reproducible position in [0, n)
    u64 z = (g_eci.seed += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return n ? z % n : 0;
}
static void ec_inject_note(int site, u64 bit) {
    if (g_ec_quiet) return;
    fprintf(stderr, "EC INJECT: flipped bit %" PRIu64 " (%s, opportunity "
            "%ld)\n", bit, eci_desc[site], g_eci.ctr.load());
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// device context for the blocked scan

struct TsGPU {
    u64 r = 0, s = 0;
    size_t nw = 0;             // words for r bits
    size_t nw2 = 0;            // scratch words for 2r bits
    size_t nwc = 0;            // words for the unreduced Horner (r+m-1 bits)
    int m = 0;                 // block size (h slots); m <= 64
    int fftm = 0;
    CantorBasis Bb;
    CudaFFT F;
    u64 *dH[2] = {nullptr, nullptr};   // ping-pong, m*nw words each
    int hcur = 0;
    u64 *dHold = nullptr;      // fineDDF snapshot (allocated on demand)
    u64 *dA = nullptr;         // interval accumulator, nw
    u64 *dC = nullptr;         // block polynomial, nw+1 (Horner headroom)
    u64 *dTmp = nullptr;       // nw scratch (Pascal init)
    u64 *dS0 = nullptr;        // nw2 scratch
    u64 *dS1 = nullptr;        // nw2 scratch
    u64 *dFA = nullptr, *dFB = nullptr;
    bool legacy_sq = false;    // force 3-kernel squaring path
    double t_sq = 0, t_horner = 0, t_mul = 0;   // accumulated seconds

    // ---- error checking (gf2_ec.h) ----
    bool ec_on = false;        // checks active for the current scan
    u64 *ecD = nullptr;        // window accumulator: XOR of the digests S_z
    u64 *ecS0 = nullptr;       // digest at the start of the window
    u64 *ecX = nullptr, *ecY = nullptr, *ecZ = nullptr;   // nw scratch each
    u64 *ecPart = nullptr;     // partial folds (3 * EC_NBLK_MAX * EC_NF)
    u64 *ecOut = nullptr;      // small result arena the host reads back
    EcFold ecA;                // phi(dA), carried on the host block to block
    EcFold ecC;                // phi(C) as last predicted (diagnostics)
    long ec_w = 0;             // blocks accumulated in the current window
    enum { ECO_CU = 0,                  // fold of the unreduced Horner sum
           ECO_T = EC_NF,               // 3 segment folds of the product
           ECO_A = 4 * EC_NF,           // fold of the reduced product
           ECO_FIRED = 5 * EC_NF,       // Horner tail quotient mask
           ECO_FLAG = 5 * EC_NF + 1,    // full-width compare result
           ECO_TAB = 5 * EC_NF + 2,     // fold of a whole h-table
           ECO_WORDS = 6 * EC_NF + 2 };

    void init(u64 r_) {
        r = r_;
        nw = bits_to_words(r);
        nw2 = 2 * nw;
        Bb.build();
        if (!Bb.verify()) { fprintf(stderr, "Cantor basis failure\n"); exit(1); }
        size_t ca = (size_t)((r + 31) >> 5);
        fftm = 1;
        while (((size_t)1 << fftm) < 2 * ca - 1) fftm++;
        F.init(fftm, Bb);
        CUCHK(cudaMalloc(&dA, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dC, (nw + 1) * sizeof(u64)));
        CUCHK(cudaMalloc(&dTmp, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dS0, nw2 * sizeof(u64)));
        CUCHK(cudaMalloc(&dS1, nw2 * sizeof(u64)));
        CUCHK(cudaMalloc(&dFA, F.n * sizeof(u64)));
        CUCHK(cudaMalloc(&dFB, F.n * sizeof(u64)));
    }
    void set_m(int m_) {
        if (m_ < 2 || m_ > 64) { fprintf(stderr, "need 2 <= m <= 64\n"); exit(1); }
        if (m == m_) return;
        for (int i = 0; i < 2; i++)
            if (dH[i]) { cudaFree(dH[i]); dH[i] = nullptr; }
        m = m_;
        nwc = bits_to_words(r + (u64)m - 1);
        CUCHK(cudaMalloc(&dH[0], (size_t)m * nw * sizeof(u64)));
        CUCHK(cudaMalloc(&dH[1], (size_t)m * nw * sizeof(u64)));
    }
    void need_hold() {
        if (!dHold) CUCHK(cudaMalloc(&dHold, (size_t)m * nw * sizeof(u64)));
    }
    void set_s(u64 s_) {
        if (!(s_ >= 1 && 2 * s_ < r)) {
            fprintf(stderr, "need 1 <= s <= (r-1)/2 (got s=%" PRIu64 ")\n", s_);
            exit(1);
        }
        s = s_;
    }
    void fini() {
        for (int i = 0; i < 2; i++) if (dH[i]) cudaFree(dH[i]);
        if (dHold) cudaFree(dHold);
        cudaFree(dA); cudaFree(dC); cudaFree(dTmp);
        cudaFree(dS0); cudaFree(dS1);
        cudaFree(dFA); cudaFree(dFB);
        if (ecD) {
            cudaFree(ecD); cudaFree(ecS0); cudaFree(ecX); cudaFree(ecY);
            cudaFree(ecZ); cudaFree(ecPart); cudaFree(ecOut);
            ecD = nullptr;
        }
        F.fini();
    }
    u64 *hslot(int j) { return dH[hcur] + (size_t)j * nw; }
    u64 *hslot_o(int j) { return dH[1 - hcur] + (size_t)j * nw; }

    // dst <- src^2 mod T (src != dst, both nw words, bits >= r zero)
    void square_slot(const u64 *src, u64 *dst) {
        int blocks, threads;
        if (!legacy_sq) {
            CudaFFT::launch_dims(nw, blocks, threads);
            k_square_fused<<<blocks, threads>>>(src, dst, nw, r, s);
        } else {
            CudaFFT::launch_dims(nw2, blocks, threads);
            k_spread_square<<<blocks, threads>>>(src, dS0, nw2);
            size_t w1 = bits_to_words(r + s - 1);
            CudaFFT::launch_dims(w1, blocks, threads);
            k_fold_pass<<<blocks, threads>>>(dS0, nw2, dS1, w1, r, s,
                                             2 * r - 1);
            CudaFFT::launch_dims(nw, blocks, threads);
            k_fold_pass<<<blocks, threads>>>(dS1, w1, dst, nw, r, s,
                                             r + s - 1);
        }
    }
    // one Frobenius step of the whole h-table: all m slots squared once
    void square_all_step() {
        for (int j = 0; j < m; j++)
            square_slot(hslot(j), hslot_o(j));
        hcur = 1 - hcur;
    }
    // C <- x^m + sum_j h[j] x^(m-1-j)  (reduced mod T)
    void horner_C() {
        int blocks, threads;
        CudaFFT::launch_dims(nwc, blocks, threads);
        k_horner_fused<<<blocks, threads>>>(dH[hcur], nw, m, dC, nwc);
        k_horner_tail<<<1, 1>>>(dC, r, s, m,
                                ec_on ? ecOut + ECO_FIRED : nullptr);
    }
    // dS0 <- x * y (full 2r-1-bit product; x,y device residues < r bits)
    void mul_full(const u64 *dx, const u64 *dy) {
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
        k_overlap_add<<<blocks, threads>>>(dFA, 2 * ca - 1, dS0, nw2);
    }
    // reduce dS0 (bit length 2r-1) mod T into dst (nw words)
    void reduce_into(u64 *dst) {
        int blocks, threads;
        size_t w1 = bits_to_words(r + s - 1);
        CudaFFT::launch_dims(w1, blocks, threads);
        k_fold_pass<<<blocks, threads>>>(dS0, nw2, dS1, w1, r, s, 2 * r - 1);
        CudaFFT::launch_dims(nw, blocks, threads);
        k_fold_pass<<<blocks, threads>>>(dS1, w1, dst, nw, r, s, r + s - 1);
    }
    // A <- A * C mod T.  With error checking the unreduced product and the
    // reduced result are folded on the device (gf2_ec.h, mechanism 3); the
    // host compares them in ec_block_verify.
    void mul_A_by_C() {
        mul_full(dA, dC);
        if (ec_inject_now(ECI_T)) inject(dS0, 2 * r - 1, ECI_T);
        if (ec_on) ec_after_mul();
        reduce_into(dA);
        if (ec_inject_now(ECI_RED)) inject(dA, r, ECI_RED);
        if (ec_on) ec_after_reduce();
        if (ec_inject_now(ECI_A)) inject(dA, r, ECI_A);
    }
    void reset_A() {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_set_one<<<blocks, threads>>>(dA, nw);
        ecf_one(ecA);
    }

    // ---- error checking ------------------------------------------------
    void ec_setup(bool on) {
        ec_on = on;
        if (!on || ecD) return;
        CUCHK(cudaMalloc(&ecD, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecS0, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecX, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecY, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecZ, nw * sizeof(u64)));
        CUCHK(cudaMalloc(&ecPart,
                         (size_t)3 * EC_NBLK_MAX * EC_NF * sizeof(u64)));
        CUCHK(cudaMalloc(&ecOut, ECO_WORDS * sizeof(u64)));
        CUCHK(cudaMemset(ecOut, 0, ECO_WORDS * sizeof(u64)));
        ecf_one(ecA);
    }
    // --ect: isolate the device time of a group of checking kernels.  The
    // extra synchronisation is only paid when timing is requested.
    double ec_tic() const {
        if (!g_ec.timing) return 0;
        CUCHK(cudaDeviceSynchronize());
        return trin_now_s();
    }
    void ec_toc(double t0, double &acc) const {
        if (!g_ec.timing) return;
        CUCHK(cudaDeviceSynchronize());
        acc += trin_now_s() - t0;
    }
    // test hook: flip one bit of a device buffer holding `nbits` bits
    void inject(u64 *buf, u64 nbits, int site) {
        u64 bit = ec_inject_rnd(nbits);
        k_ec_flip_bit<<<1, 1>>>(buf, bit);
        ec_inject_note(site, bit);
    }
    void inject_h(u64 *base, int site) {
        int j = (int)ec_inject_rnd((u64)m);
        inject(base + (size_t)j * nw, r, site);
    }
    // fold of src[0..n) -> ecOut[off .. off+EC_NF)
    void ec_fold_dev(const u64 *src, size_t n, size_t off) {
        unsigned nb = ec_nblk_for(n);
        k_ec_fold_part<<<nb, EC_NF>>>(src, n, ecPart);
        k_ec_combine<<<1, EC_NF>>>(ecPart, nb, 1, ecOut + off);
    }
    void ec_read(size_t off, size_t n, u64 *dst) {
        CUCHK(cudaMemcpy(dst, ecOut + off, n * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    // fold of a whole h-table (m*nw words taken as one polynomial)
    void ec_table_fold(const u64 *base, EcFold &f) {
        ec_fold_dev(base, (size_t)m * nw, ECO_TAB);
        ec_read(ECO_TAB, EC_NF, f.w);
    }
    // per block, before the Horner build: D ^= S_z and the fold of the
    // unreduced Horner sum, in one pass over the table
    void ec_block_pre() {
        double t0 = ec_tic();
        unsigned nb = ec_nblk_for(nwc);
        k_ec_block_pass<<<nb, EC_NF>>>(dH[hcur], nw, m, ecD, nwc, ecPart);
        k_ec_combine<<<1, EC_NF>>>(ecPart, nb, 1, ecOut + ECO_CU);
        ec_toc(t0, g_ecs.t_block);
    }
    void ec_after_mul() {                  // dS0 = unreduced product
        double t0 = ec_tic();
        unsigned nb = ec_nblk_for(nw2);
        k_ec_fold3_part<<<nb, EC_NF>>>(dS0, nw2, r, 2 * r - s, ecPart);
        k_ec_combine<<<1, EC_NF>>>(ecPart, nb, 3, ecOut + ECO_T);
        ec_toc(t0, g_ecs.t_block);
    }
    void ec_after_reduce() {
        double t0 = ec_tic();
        ec_fold_dev(dA, nw, ECO_A);
        ec_toc(t0, g_ecs.t_block);
    }
    // host half of the per-block check.  Returns 0 if clean; bit 0: the
    // Horner build or the FFT product is wrong (phi(t) != phi(A) phi(C));
    // bit 1: the reduction is wrong.  Updates the carried phi(A).
    int ec_block_verify() {
        double t0 = ec_tic();
        u64 buf[ECO_FLAG];
        ec_read(0, ECO_FLAG, buf);
        double t1 = g_ec.timing ? trin_now_s() : 0;
        EcFold fCu, seg[3], fAn, fC, ft, fred, prod;
        memcpy(fCu.w, buf + ECO_CU, sizeof fCu.w);
        for (int g = 0; g < 3; g++)
            memcpy(seg[g].w, buf + ECO_T + g * EC_NF, sizeof seg[g].w);
        memcpy(fAn.w, buf + ECO_A, sizeof fAn.w);
        ecf_horner_predict(fC, fCu, buf[ECO_FIRED], m, r, s);
        ecf_reduce_predict(ft, fred, seg, r, s);
        ecf_mul(prod, ecA, fC);
        ecC = fC;
        int rc = 0;
        if (!ecf_eq(prod, ft)) rc |= 1;
        if (!ecf_eq(fred, fAn)) rc |= 2;
        ecA = fAn;
        g_ecs.blocks++;
        if (g_ec.timing) {
            double t2 = trin_now_s();
            g_ecs.t_block += t1 - t0;
            g_ecs.t_host += t2 - t1;
        }
        return rc;
    }
    // h-table chain: restart it from the table as it stands
    void ec_chain_reset() {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_ec_digest<<<blocks, threads>>>(dH[hcur], nw, m, ecS0);
        k_set_zero<<<blocks, threads>>>(ecD, nw);
        ec_w = 0;
    }
    // h-table chain check (gf2_ec.h, mechanism 1), compared at full width:
    //     sigma(D) + D == S_start + S_now,   sigma = Frobenius^m.
    // With no block since the last check (ec_w == 0, D == 0) it reduces to
    // S_now == S_start: "the idle table has not changed".  Either way the
    // next window starts from the digest taken here.  True if clean.
    bool ec_window_check() {
        double t0 = ec_tic();
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_ec_digest<<<blocks, threads>>>(dH[hcur], nw, m, ecX);
        const u64 *sig = ecD;
        if (ec_w > 0) {
            u64 *dst = ecY;
            for (int t = 0; t < m; t++) {
                square_slot(sig, dst);
                sig = dst;
                dst = (dst == ecY) ? ecZ : ecY;
            }
        }
        k_ec_set_word<<<1, 1>>>(ecOut + ECO_FLAG, 0);
        k_ec_or4<<<blocks, threads>>>(sig, ecD, ecS0, ecX, nw,
                                      ecOut + ECO_FLAG);
        u64 flag = 1;
        ec_read(ECO_FLAG, 1, &flag);
        std::swap(ecS0, ecX);
        k_set_zero<<<blocks, threads>>>(ecD, nw);
        ec_w = 0;
        g_ecs.windows++;
        ec_toc(t0, g_ecs.t_window);
        return flag == 0;
    }
    // Verify the table at block start k from first principles (gf2_ec.h,
    // mechanism 2):  (1 + y_k)(1 + S_k) == 1 + y_{k+m},  y_t = x^(2^t),
    // using 1 + y_t = (1 + x)^(2^t).  Costs k + m squarings of one residue
    // and one modmul -- about 1/m of building the table.  Leaves the chain
    // reset at this table.  True if the table is right.
    bool ec_verify_table_at(u64 k) {
        double t0 = ec_tic();
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        k_set_one<<<blocks, threads>>>(ecY, nw);
        k_xor_x<<<1, 1>>>(ecY);                        // 1 + x
        u64 *cur = ecY, *oth = ecZ;
        for (u64 t = 0; t < k; t++) {
            square_slot(cur, oth);
            std::swap(cur, oth);
        }
        CUCHK(cudaMemcpy(ecX, cur, nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));   // 1 + y_k
        for (int t = 0; t < m; t++) {
            square_slot(cur, oth);
            std::swap(cur, oth);
        }                                              // cur = 1 + y_{k+m}
        ec_chain_reset();                              // ecS0 = S_k
        CUCHK(cudaMemcpy(ecD, ecS0, nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));
        k_ec_flip_bit<<<1, 1>>>(ecD, 0);               // 1 + S_k
        mul_full(ecX, ecD);
        reduce_into(oth);
        k_ec_set_word<<<1, 1>>>(ecOut + ECO_FLAG, 0);
        k_ec_neq<<<blocks, threads>>>(oth, cur, nw, ecOut + ECO_FLAG);
        u64 flag = 1;
        ec_read(ECO_FLAG, 1, &flag);
        k_set_zero<<<blocks, threads>>>(ecD, nw);
        g_ecs.n_init++;
        ec_toc(t0, g_ecs.t_init);
        return flag == 0;
    }

    // h-table init at block start kstart: Pascal-triangle build of the
    // elementary symmetric functions of (x^(2^0) .. x^(2^(m-1))), then
    // kstart Frobenius steps.  Mirrors factor.cpp's init_h exactly.
    void init_h_at(u64 kstart) {
        int blocks, threads;
        CudaFFT::launch_dims(nw, blocks, threads);
        for (int j = 0; j < m; j++)
            k_set_zero<<<blocks, threads>>>(hslot(j), nw);
        for (int j = 1; j <= m; j++) {
            for (int i = j - 1; i >= 0; i--) {
                if (i > 0) {
                    square_slot(hslot(i - 1), dTmp);
                    CUCHK(cudaMemcpy(hslot(i - 1), dTmp, nw * sizeof(u64),
                                     cudaMemcpyDeviceToDevice));
                    k_mulx_xor<<<blocks, threads>>>(hslot(i), hslot(i - 1),
                                                    nw, r, s);
                } else {
                    k_xor_x<<<1, 1>>>(hslot(0));       // h[0] ^= x
                }
            }
        }
        if (ec_inject_now(ECI_INIT)) inject_h(dH[hcur], ECI_INIT);
        for (u64 t = 0; t < kstart; t++) {
            square_all_step();
            if (ec_inject_now(ECI_INIT)) inject_h(dH[hcur], ECI_INIT);
        }
    }

    void copy_h(u64 *dst_base, const u64 *src_base) {
        CUCHK(cudaMemcpy(dst_base, src_base, (size_t)m * nw * sizeof(u64),
                         cudaMemcpyDeviceToDevice));
    }
    void h_to_host(std::vector<u64> &out) {
        out.resize((size_t)m * nw);
        CUCHK(cudaMemcpy(out.data(), dH[hcur],
                         (size_t)m * nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void h_from_host(const std::vector<u64> &in) {
        CUCHK(cudaMemcpy(dH[hcur], in.data(),
                         (size_t)m * nw * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }
    void A_to_host(std::vector<u64> &out) {
        out.resize(nw);
        CUCHK(cudaMemcpy(out.data(), dA, nw * sizeof(u64),
                         cudaMemcpyDeviceToHost));
    }
    void A_from_host(const std::vector<u64> &in) {
        CUCHK(cudaMemcpy(dA, in.data(), nw * sizeof(u64),
                         cudaMemcpyHostToDevice));
        // the carried fold comes from the host copy, so a bad upload shows
        // up at the next product check
        ecf_fold_host(ecA, in.data(), in.size() < nw ? in.size() : nw);
    }

    // ---- verified transfers (gf2_ec.h, mechanism 4) ----
    // D2H of the current h-table; `f` receives its fold, computed on the
    // device and confirmed against what arrived.  A mismatch is re-read: a
    // transfer error is transient.  False if three reads all disagree.
    bool h_to_host_ec(std::vector<u64> &out, EcFold &f) {
        double t0 = ec_tic();
        ec_table_fold(dH[hcur], f);
        bool ok = false;
        for (int tr = 0; tr < 3 && !ok; tr++) {
            h_to_host(out);
            if (ec_inject_now(ECI_XFER_H)) {
                u64 bit = ec_inject_rnd((u64)out.size() * 64);
                out[bit >> 6] ^= 1ULL << (bit & 63);
                ec_inject_note(ECI_XFER_H, bit);
            }
            EcFold fh;
            ecf_fold_host(fh, out.data(), out.size());
            ok = ecf_eq(fh, f);
            if (!ok) g_ecs.xfer_retries++;
        }
        g_ecs.n_xfer++;
        ec_toc(t0, g_ecs.t_xfer);
        return ok;
    }
    // H2D of an h-table.  `expect` (may be null) is the fold recorded when
    // the host copy was captured.  Returns 0 if the device now holds exactly
    // that table (and restarts the chain from it); 1 if the host copy no
    // longer matches its recorded fold; 2 if the upload cannot be confirmed.
    int h_from_host_ec(const std::vector<u64> &in, const EcFold *expect) {
        double t0 = ec_tic();
        EcFold fh, fd;
        ecf_fold_host(fh, in.data(), in.size());
        int rc = 2;
        if (expect && !ecf_eq(fh, *expect)) {
            rc = 1;
        } else {
            for (int tr = 0; tr < 3 && rc; tr++) {
                h_from_host(in);
                ec_table_fold(dH[hcur], fd);
                if (ecf_eq(fd, fh)) rc = 0;
                else g_ecs.xfer_retries++;
            }
        }
        if (!rc) ec_chain_reset();
        g_ecs.n_xfer++;
        ec_toc(t0, g_ecs.t_xfer);
        return rc;
    }
    // D2H of A, confirmed against the fold carried from its last product
    // check: catches a bad transfer and a flip in the idle accumulator.
    bool A_to_host_ec(std::vector<u64> &out) {
        double t0 = ec_tic();
        bool ok = false;
        for (int tr = 0; tr < 3 && !ok; tr++) {
            A_to_host(out);
            if (ec_inject_now(ECI_XFER_A)) {
                u64 bit = ec_inject_rnd(r);
                out[bit >> 6] ^= 1ULL << (bit & 63);
                ec_inject_note(ECI_XFER_A, bit);
            }
            EcFold f;
            ecf_fold_host(f, out.data(), out.size());
            ok = ecf_eq(f, ecA);
            if (!ok) g_ecs.xfer_retries++;
        }
        g_ecs.n_xfer++;
        ec_toc(t0, g_ecs.t_xfer);
        return ok;
    }
};

// ---------------------------------------------------------------------------
// CPU GCD thread pool (carried over from trinomial_scan.cu): jobs are
// (id, A words) against a shared_ptr modulus; verdicts retrieved by id
// and consumed by the caller strictly in interval order; stale jobs
// are dropped without blocking after a hit.

struct GcdRes {
    u64 gdeg = 0;
    std::vector<u64> g;      // always captured (up to r bits; a few MB)
    double secs = 0;
    bool ec_bad = false;     // error checking: this verdict cannot be trusted
    const char *ec_why = "";
};

// gcd with full mask capture; an interval gcd is at most r bits and we
// always want it (the 4-Mbit default in gf2_gcd.h would drop deep-tail
// single-factor gcds that the shortcut in resolve_gcd needs verbatim)
static u64 g_keep_bits = ((u64)1 << 22);
static GpuMulHook *g_gm = nullptr;       // set when --gpu-gcd is active
static bool g_use_hybrid = false;
static bool g_hyb_xcheck = false;        // selftest: verify hybrid vs NTL
static u64 ts_gcd(const u64 *a, size_t aw, const u64 *b, size_t bw,
                  std::vector<u64> *gout) {
#ifdef HAVE_NTL
    if (g_use_hybrid) {
        u64 d = poly_gcd_hybrid(a, aw, b, bw, gout, g_keep_bits, g_gm);
        if (g_hyb_xcheck) {
            std::vector<u64> gref;
            u64 dr = poly_gcd_ntl(a, aw, b, bw, &gref, g_keep_bits);
            if (dr != d || (dr && gout && !gout->empty() && *gout != gref)) {
                fprintf(stderr, "HYBRID GCD MISMATCH: hyb %" PRIu64
                        " ntl %" PRIu64 " (aw %zu bw %zu) -- dumping\n",
                        d, dr, aw, bw);
                FILE *f = fopen("/tmp/hyb_mismatch.bin", "wb");
                if (f) {
                    u64 h[2] = {(u64)aw, (u64)bw};
                    fwrite(h, 8, 2, f);
                    fwrite(a, 8, aw, f);
                    fwrite(b, 8, bw, f);
                    fclose(f);
                }
                if (gout) *gout = gref;   // heal so the run continues
                d = dr;
            }
        }
        return d;
    }
    return poly_gcd_ntl(a, aw, b, bw, gout, g_keep_bits);
#else
    return poly_gcd_naive(a, aw, b, bw, gout);
#endif
}

// Certified gcd (gf2_gcd.h): fa / fb are the folds the inputs must have.
static u64 ts_gcd_cert(const u64 *a, size_t aw, const u64 *b, size_t bw,
                       std::vector<u64> *gout, const EcFold *fa,
                       const EcFold *fb, int *st) {
#ifdef HAVE_NTL
    return poly_gcd_certified(a, aw, b, bw, gout, g_keep_bits, g_gm, fa, fb,
                              st);
#else
    (void)fa; (void)fb;
    *st = 3;
    return poly_gcd_naive(a, aw, b, bw, gout);
#endif
}

static const char *gcd_backend() {
#ifdef HAVE_NTL
    if (!ntl_gf2x_backed())
        return "NTL (HalfGCD + CanZass) -- WARNING: NTL built WITHOUT "
               "gf2x; large-degree GCDs will be many times slower "
               "(rebuild NTL with NTL_GF2X_LIB=on)";
    if (g_ec.on && g_ec.gcd_cert)
        return hyb_cert_xgcd
                   ? "CERTIFIED: NTL extended gcd, Bezout identity "
                     "fold-verified (--ecg-xgcd)"
               : (g_gm || g_use_hybrid)
                   ? "CERTIFIED half-gcd chain (GPU-offloaded mults, every "
                     "step fold-verified) + verified XGCD finish"
                   : "CERTIFIED half-gcd chain (CPU mults via NTL/gf2x, "
                     "every step fold-verified) + verified XGCD finish";
    if (g_use_hybrid)
        return g_gm ? "hybrid HGCD (GPU-offloaded mults) + NTL finish"
                    : "hybrid HGCD (CPU mults) + NTL finish";
    return "NTL gf2x-backed (HalfGCD + CanZass)";
#else
    return "naive fallback -- NO NTL: factor resolution will fail for "
           "multi-factor interval gcds; NOT fit for production";
#endif
}

#ifdef HAVE_NTL
// GPU multiplication service for the hybrid HGCD: packs both operands
// into 32-bit chunks, runs the Cantor FFT convolution, and returns the
// full product.  Owns its device buffers (no sharing with the scan
// state) and lazily builds one CudaFFT plan per size class; a mutex
// serializes hook users, and the legacy default CUDA stream serializes
// device work against the concurrently running scan.
struct TsGpuMul : GpuMulHook {
    CantorBasis *Bb = nullptr;
    std::map<int, CudaFFT *> plans;
    u64 *dA = nullptr, *dB = nullptr, *dP = nullptr;
    u64 *dM = nullptr, *dR = nullptr;    // batched-op scratch/accumulator
    size_t cap_n = 0;
    bool alive = false;
    std::mutex mu;
    void init(CantorBasis *bb, int max_fm) {
        std::lock_guard<std::mutex> lk(mu);
        Bb = bb;
        cap_n = (size_t)1 << max_fm;
        CUCHK(cudaMalloc(&dA, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dB, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dM, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dR, cap_n * sizeof(u64)));
        CUCHK(cudaMalloc(&dP, (cap_n / 2 + 2) * sizeof(u64)));
        alive = true;
    }
    // Takes the hook mutex, so a call already in flight completes
    // first, and clears `alive`, so later calls decline (the engine
    // then multiplies on the CPU) instead of touching freed buffers.
    void fini() {
        std::lock_guard<std::mutex> lk(mu);
        alive = false;
        for (auto &kv : plans) { kv.second->fini(); delete kv.second; }
        plans.clear();
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dM) cudaFree(dM);
        if (dR) cudaFree(dR);
        if (dP) cudaFree(dP);
        dA = dB = dM = dR = dP = nullptr;
    }
    CudaFFT *plan_for(int fm) {
        auto it = plans.find(fm);
        if (it != plans.end()) return it->second;
        CudaFFT *F = new CudaFFT();
        F->init(fm, *Bb);
        plans[fm] = F;
        return F;
    }
    // pack a host operand and forward-transform it into dst
    void pack_fwd(CudaFFT *F, const u64 *w, size_t nw, u64 bits, u64 *dst) {
        size_t c = (size_t)((bits + 31) >> 5);
        size_t nu = bits_to_words(bits);      // ignore caller padding: a
        if (nu > nw) nu = nw;                 // trailing zero word must
        int blocks, threads;                  // not overrun the staging
        CUCHK(cudaMemcpy(dP, w, nu * sizeof(u64), cudaMemcpyHostToDevice));
        CudaFFT::launch_dims(F->n, blocks, threads);
        k_pack<<<blocks, threads>>>(dP, nu, dst, c, F->n);
        F->fwd(dst);
    }
    bool mat2_apply(const Op m[4], const Op &a, const Op &b,
                    std::vector<u64> &oa, std::vector<u64> &ob) override {
        size_t ca = (size_t)((a.bits + 31) >> 5);
        size_t cb = (size_t)((b.bits + 31) >> 5);
        size_t mxc = 0;
        u64 rb[2] = {0, 0};                    // result bits per row
        for (int i = 0; i < 4; i++) {
            if (!m[i].bits) continue;
            size_t cx = (i & 1) ? cb : ca;
            if (!cx) continue;
            size_t nch = (size_t)((m[i].bits + 31) >> 5) + cx - 1;
            if (nch > mxc) mxc = nch;
            u64 pb = m[i].bits + ((i & 1) ? b.bits : a.bits) - 1;
            if (pb > rb[i >> 1]) rb[i >> 1] = pb;
        }
        if (!mxc) { oa.clear(); ob.clear(); return true; }
        int fm = 5;
        while (((size_t)1 << fm) < mxc + 1) fm++;
        if (((size_t)1 << fm) > cap_n) return false;
        std::lock_guard<std::mutex> lk(mu);
        if (!alive) return false;
        CudaFFT *F = plan_for(fm);
        int blocks, threads;
        CudaFFT::launch_dims(F->n, blocks, threads);
        bool ua = (m[0].bits && a.bits) || (m[2].bits && a.bits);
        bool ub = (m[1].bits && b.bits) || (m[3].bits && b.bits);
        if (ua) pack_fwd(F, a.w, a.nw, a.bits, dA);
        if (ub) pack_fwd(F, b.w, b.nw, b.bits, dB);
        for (int row = 0; row < 2; row++) {
            std::vector<u64> &out = row ? ob : oa;
            const Op &e0 = m[2 * row], &e1 = m[2 * row + 1];
            bool h0 = e0.bits && a.bits, h1 = e1.bits && b.bits;
            if (!h0 && !h1) { out.clear(); continue; }
            if (h0) {
                pack_fwd(F, e0.w, e0.nw, e0.bits, dM);
                k_pointwise<<<blocks, threads>>>(dM, dA, F->n);
            }
            if (h1) {
                u64 *t = h0 ? dR : dM;
                pack_fwd(F, e1.w, e1.nw, e1.bits, t);
                k_pointwise<<<blocks, threads>>>(t, dB, F->n);
                if (h0) k_xor_buf<<<blocks, threads>>>(dM, dR, F->n);
            }
            F->inv(dM);
            size_t ow = bits_to_words(rb[row]);
            int b2, t2;
            CudaFFT::launch_dims(ow, b2, t2);
            k_overlap_add<<<b2, t2>>>(dM, mxc, dP, ow);
            out.resize(ow);
            CUCHK(cudaMemcpy(out.data(), dP, ow * sizeof(u64),
                             cudaMemcpyDeviceToHost));
        }
        return true;
    }
    bool mul(const u64 *a, size_t aw, u64 abits,
             const u64 *b, size_t bw, u64 bbits,
             std::vector<u64> &out) override {
        size_t ca = (size_t)((abits + 31) >> 5);
        size_t cb = (size_t)((bbits + 31) >> 5);
        size_t nch = ca + cb - 1;
        int fm = 5;
        while (((size_t)1 << fm) < nch + 1) fm++;
        if (((size_t)1 << fm) > cap_n) return false;   // too big: CPU
        size_t ow = bits_to_words(abits + bbits - 1);
        std::lock_guard<std::mutex> lk(mu);
        if (!alive) return false;
        CudaFFT *F = plan_for(fm);
        int blocks, threads;
        size_t au = bits_to_words(abits), bu = bits_to_words(bbits);
        if (au > aw) au = aw;
        if (bu > bw) bu = bw;
        CUCHK(cudaMemcpy(dP, a, au * sizeof(u64), cudaMemcpyHostToDevice));
        CudaFFT::launch_dims(F->n, blocks, threads);
        k_pack<<<blocks, threads>>>(dP, au, dA, ca, F->n);
        CUCHK(cudaMemcpy(dP, b, bu * sizeof(u64), cudaMemcpyHostToDevice));
        k_pack<<<blocks, threads>>>(dP, bu, dB, cb, F->n);
        F->fwd(dA);
        F->fwd(dB);
        k_pointwise<<<blocks, threads>>>(dA, dB, F->n);
        F->inv(dA);
        CudaFFT::launch_dims(ow, blocks, threads);
        k_overlap_add<<<blocks, threads>>>(dA, nch, dP, ow);
        out.resize(ow);
        CUCHK(cudaMemcpy(out.data(), dP, ow * sizeof(u64),
                         cudaMemcpyDeviceToHost));
        return true;
    }
};
static TsGpuMul g_ts_gm;
#endif // HAVE_NTL

class GcdPool {
    std::mutex mu;
    std::condition_variable cv_job, cv_res;
    struct Job {
        u64 id;
        std::shared_ptr<std::vector<u64>> acc;
        std::shared_ptr<const std::vector<u64>> mod;
        EcFold f;              // fold of acc, as verified on arrival
        EcFold fm;             // fold of the modulus
        bool has_f;
    };
    std::deque<Job> jobs;
    std::map<u64, GcdRes> results;
    std::map<u64, std::vector<GcdRes>> partial;  // --ecg: copies finished
    std::map<u64, int> live;                     // id -> copies queued/running
    std::set<u64> discard;
    bool stopf = false;
    std::vector<std::thread> ths;

    static bool same(const GcdRes &a, const GcdRes &b) {
        return !a.ec_bad && !b.ec_bad && a.gdeg == b.gdeg && a.g == b.g;
    }

    void worker() {
        for (;;) {
            std::unique_lock<std::mutex> lk(mu);
            cv_job.wait(lk, [&] { return stopf || !jobs.empty(); });
            if (jobs.empty()) { if (stopf) return; continue; }
            Job j = std::move(jobs.front());
            jobs.pop_front();
            bool stale = discard.count(j.id) > 0;   // verdict no longer wanted
            lk.unlock();
            GcdRes rres;
            if (!stale) {
                double t0 = trin_now_s();
                // The accumulator may have waited in host RAM behind other
                // gcds: confirm it is still what the device produced.
                bool okf = true;
                if (j.has_f) {
                    EcFold f;
                    ecf_fold_host(f, j.acc->data(), j.acc->size());
                    okf = ecf_eq(f, j.f);
                }
                if (!okf) {
                    rres.ec_bad = true;
                    rres.ec_why = "accumulator changed in host RAM while "
                                  "queued for its gcd";
                } else if (j.has_f && g_ec.gcd_cert) {
                    // certified gcd; a failed certificate means a fault
                    // DURING the gcd, so it is simply run again
                    int st = 0;
                    for (int tr = 0; tr < 2; tr++) {
                        rres.g.clear();
                        rres.gdeg = ts_gcd_cert(j.acc->data(), j.acc->size(),
                                                j.mod->data(), j.mod->size(),
                                                &rres.g, &j.f, &j.fm, &st);
                        if (st == 0 || st == 3) break;
                        ec_event(ECC_GCD, "gcd certificate failed (%s)%s",
                                 st == 1 ? "an input changed in host RAM"
                                         : "a step identity does not hold: "
                                           "fault during the gcd",
                                 tr == 0 ? " -- running the gcd again" : "");
                        if (tr == 0) g_ecs.gcd_reruns++;
                    }
                    if (st == 1 || st == 2) {
                        rres.ec_bad = true;
                        rres.ec_why = "gcd certificate failed twice";
                        rres.gdeg = 0;
                        rres.g.clear();
                    } else if (st == 3) {    // builds without NTL only
                        if (g_ecs.gcd_uncert++ == 0)
                            ec_event(-1, "this build has no certified gcd "
                                     "(no NTL): the gcd is NOT checked");
                    } else {
                        g_ecs.gcd_cert++;
                    }
                } else {
                    rres.gdeg = ts_gcd(j.acc->data(), j.acc->size(),
                                       j.mod->data(), j.mod->size(), &rres.g);
                    if (ec_inject_now(ECI_GCD)) {    // test hook
                        if (rres.gdeg) { rres.gdeg = 0; rres.g.clear(); }
                        else { rres.gdeg = 3; rres.g.assign(2, 0);
                               rres.g[0] = 0xB; }    // x^3 + x + 1
                        ec_inject_note(ECI_GCD, 0);
                    }
                }
                rres.secs = trin_now_s() - t0;
            }
            lk.lock();
            finish_locked(j, rres);
            cv_res.notify_all();
        }
    }

    // bookkeeping for one finished copy of a job (mu held)
    void finish_locked(const Job &j, GcdRes &rres) {
        int left = --live[j.id];
        if (discard.count(j.id)) {
            if (left <= 0) {
                discard.erase(j.id);
                live.erase(j.id);
                partial.erase(j.id);
            }
            return;
        }
        GcdRes fin;
        bool have = false;
        if (!g_ec.gcd_dup || rres.ec_bad) {
            fin = std::move(rres);
            have = true;
        } else {
            // --ecg: the gcd is computed twice from the same verified input
            // and the answers must agree; a third run breaks a tie
            auto &pv = partial[j.id];
            pv.push_back(std::move(rres));
            if (pv.size() == 2) {
                if (same(pv[0], pv[1])) {
                    fin = pv[0];
                    fin.secs = pv[0].secs > pv[1].secs ? pv[0].secs
                                                       : pv[1].secs;
                    have = true;
                } else {
                    ec_event(ECC_GCD, "two runs of one gcd disagree (degree %"
                             PRIu64 " vs %" PRIu64 ") -- running it a third "
                             "time", pv[0].gdeg, pv[1].gdeg);
                    g_ecs.gcd_reruns++;
                    jobs.push_front(j);
                    left = ++live[j.id];
                    cv_job.notify_one();
                }
            } else if (pv.size() >= 3) {
                if (same(pv[2], pv[0])) fin = pv[0];
                else if (same(pv[2], pv[1])) fin = pv[1];
                else {
                    fin.ec_bad = true;
                    fin.ec_why = "three runs of one gcd gave three answers";
                }
                fin.secs = pv[0].secs + pv[1].secs + pv[2].secs;
                have = true;
            }
        }
        if (have) {
            results[j.id] = std::move(fin);
            partial.erase(j.id);
            if (left <= 0) live.erase(j.id);
            else discard.insert(j.id);     // drop the copy still in flight
        }
    }

public:
    void start(int n) {
        for (int i = 0; i < n; i++) ths.emplace_back([this] { worker(); });
    }
    // `f` (may be null) is the fold of acc established on the device; the
    // worker re-checks it immediately before the gcd starts.
    void submit(u64 id, std::vector<u64> acc,
                std::shared_ptr<const std::vector<u64>> mod,
                const EcFold *f = nullptr) {
        Job j;
        j.id = id;
        j.acc = std::make_shared<std::vector<u64>>(std::move(acc));
        j.mod = std::move(mod);
        j.has_f = (f != nullptr);
        if (f) j.f = *f; else ecf_zero(j.f);
        ecf_fold_host(j.fm, j.mod->data(), j.mod->size());
        if (ec_inject_now(ECI_JOB) && !j.acc->empty()) {    // test hook
            u64 bit = ec_inject_rnd((u64)j.acc->size() * 64 - 64);
            (*j.acc)[bit >> 6] ^= 1ULL << (bit & 63);
            ec_inject_note(ECI_JOB, bit);
        }
        int copies = g_ec.gcd_dup ? 2 : 1;
        std::lock_guard<std::mutex> lk(mu);
        live[id] = copies;
        for (int c = 0; c < copies; c++) jobs.push_back(j);
        cv_job.notify_all();
    }
    bool try_get(u64 id, GcdRes &out) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = results.find(id);
        if (it == results.end()) return false;
        out = std::move(it->second);
        results.erase(it);
        return true;
    }
    GcdRes wait_get(u64 id) {
        std::unique_lock<std::mutex> lk(mu);
        cv_res.wait(lk, [&] { return results.count(id) > 0; });
        GcdRes rres = std::move(results[id]);
        results.erase(id);
        return rres;
    }
    void forget_all_pending() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto &kv : live) discard.insert(kv.first);
        results.clear();
        partial.clear();
    }
    // Block until nothing is queued or running, returning how many jobs
    // were still in flight on entry.  forget_all_pending() only MARKS
    // jobs for discard -- a worker executing one keeps running, and it
    // is still inside ts_gcd (hence possibly inside the GPU mult hook).
    // Anything a running gcd can touch must be drained before teardown.
    size_t drain() {
        std::unique_lock<std::mutex> lk(mu);
        size_t n = live.size();
        cv_res.wait(lk, [&] {
            return stopf || (jobs.empty() && live.empty());
        });
        return n;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stopf = true;
            jobs.clear();          // prompt exit: drop queued (not running)
        }
        cv_job.notify_all();
        for (auto &t : ths) t.join();
        ths.clear();
    }
    // signal-exit path: a worker may be inside a multi-minute GCD; any
    // checkpoint is already on disk, so don't wait -- detach and let
    // the caller _Exit before static teardown can race the workers.
    void stop_nowait() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stopf = true;
            jobs.clear();
        }
        cv_job.notify_all();
        for (auto &t : ths) t.detach();
        ths.clear();
    }
};

// ---------------------------------------------------------------------------
// checkpointing: header + CRC-protected sections, 3-generation rotation

static u32 crc32_tab[256];
static void crc32_init() {
    for (u32 i = 0; i < 256; i++) {
        u32 c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc32_tab[i] = c;
    }
}
static u32 crc32_of(const void *buf, size_t len, u32 seed = 0) {
    const unsigned char *p = (const unsigned char *)buf;
    u32 c = seed ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
        c = crc32_tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

#define TS_CKPT_MAGIC 0x31434653544ULL   // "TSFC1"-ish
struct CkptHdr {
    u64 magic, version;
    u64 r, s, skip;
    u64 m;
    u64 k_iv;          // start degree of the interval being processed
    u64 blk;           // blocks of that interval already folded into A
    u64 q;             // block count of that interval
    double f0;         // growth state (factor's f0)
    u64 has_A;         // 1 iff A section present (blk > 0)
    u64 nw;
    u32 crc_h, crc_a;  // section CRCs
    u32 crc_hdr;       // of all fields above
};

static bool write_file_atomic(const std::string &path, const void *hdr,
                              size_t hdrsz, const std::vector<u64> &h,
                              const std::vector<u64> &a) {
    std::string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(hdr, 1, hdrsz, f) == hdrsz;
    if (ok && !h.empty()) ok = fwrite(h.data(), 8, h.size(), f) == h.size();
    if (ok && !a.empty()) ok = fwrite(a.data(), 8, a.size(), f) == a.size();
    if (ok) ok = fflush(f) == 0 && fsync(fileno(f)) == 0;
    fclose(f);
    if (!ok) { remove(tmp.c_str()); return false; }
    return rename(tmp.c_str(), path.c_str()) == 0;
}

static void rotate_ckpts(const std::string &base) {
    std::string c0 = base, c1 = base + ".1", c2 = base + ".2";
    remove(c2.c_str());
    rename(c1.c_str(), c2.c_str());
    rename(c0.c_str(), c1.c_str());
}

static bool save_ckpt(const std::string &base, const CkptHdr &H,
                      const std::vector<u64> &h, const std::vector<u64> &a,
                      int verbose) {
    CkptHdr W = H;
    W.crc_h = h.empty() ? 0 : crc32_of(h.data(), h.size() * 8);
    W.crc_a = a.empty() ? 0 : crc32_of(a.data(), a.size() * 8);
    W.crc_hdr = 0;
    W.crc_hdr = crc32_of(&W, sizeof(W) - sizeof(u32));
    rotate_ckpts(base);
    bool ok = write_file_atomic(base, &W, sizeof(W), h, a);
    if (verbose)
        fprintf(stderr, "  [checkpoint %s: s=%" PRIu64 " k_iv=%" PRIu64
                " blk=%" PRIu64 "/%" PRIu64 " %s]\n",
                ok ? "written" : "FAILED", H.s, H.k_iv, H.blk, H.q,
                base.c_str());
    return ok;
}

static bool load_ckpt_one(const std::string &path, CkptHdr &H,
                          std::vector<u64> &h, std::vector<u64> &a) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&H, 1, sizeof(H), f) == sizeof(H);
    if (ok) {
        CkptHdr T = H;
        T.crc_hdr = 0;
        ok = H.magic == TS_CKPT_MAGIC &&
             crc32_of(&T, sizeof(T) - sizeof(u32)) == H.crc_hdr;
    }
    if (ok) {
        h.resize((size_t)H.m * H.nw);
        ok = fread(h.data(), 8, h.size(), f) == h.size() &&
             crc32_of(h.data(), h.size() * 8) == H.crc_h;
    }
    if (ok && H.has_A) {
        a.resize((size_t)H.nw);
        ok = fread(a.data(), 8, a.size(), f) == a.size() &&
             crc32_of(a.data(), a.size() * 8) == H.crc_a;
    } else if (ok) {
        a.clear();
    }
    fclose(f);
    return ok;
}

static bool load_ckpt(const std::string &base, CkptHdr &H,
                      std::vector<u64> &h, std::vector<u64> &a) {
    const char *sfx[3] = {"", ".1", ".2"};
    for (int i = 0; i < 3; i++) {
        std::string p = base + sfx[i];
        if (load_ckpt_one(p, H, h, a)) {
            if (i) fprintf(stderr, "note: fell back to checkpoint %s\n",
                           p.c_str());
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// fine-DDF standby state.
//
// localize() needs the h-table exactly as it stood at the START of the hit
// interval.  Rebuilding it from scratch costs ka*m squarings -- at
// r=136279841 with ka in the millions that is hours (26417 s observed at
// ka=4843958).  But the scan HAD that state when the interval began, so
// instead of rebuilding we keep it:
//   * in RAM, one snapshot per interval with an outstanding gcd
//     (Pend::h0, bounded by --hsnap-mb), which removes the rebuild
//     entirely in the normal case; and
//   * on disk, once fine DDF is actually entered, so that a crash or
//     reboot mid-localization does not force the rebuild either.  The
//     file also carries the current bisection window, so a resumed run
//     re-enters localize where it left off instead of rescanning the
//     (possibly multi-hour) hit interval.
// Degrees below the hit interval are always verified clear before
// localization starts -- verdicts are consumed in order and the first hit
// returns -- so this file is self-contained: resuming from it needs no
// other state.
#define TS_FINE_MAGIC 0x314e4653544ULL     // "TSFN1"-ish
struct FineHdr {
    u64 magic, version;
    u64 r, s, skip;
    u64 m, nw;
    u64 ka0, q0;        // hit interval as reported by the coarse gcd
    u64 kcur, qcur;     // current bisection window; the h below is at kcur
    u32 crc_h;
    u32 crc_hdr;
};

static std::string fine_path(const std::string &base) { return base + ".fine"; }

static bool save_fine(const std::string &base, FineHdr H,
                      const std::vector<u64> &h, int verbose) {
    H.magic = TS_FINE_MAGIC;
    H.version = 1;
    H.crc_h = h.empty() ? 0 : crc32_of(h.data(), h.size() * 8);
    H.crc_hdr = 0;
    H.crc_hdr = crc32_of(&H, sizeof(H) - sizeof(u32));
    std::vector<u64> none;
    bool ok = write_file_atomic(fine_path(base), &H, sizeof(H), h, none);
    if (verbose)
        fprintf(stderr, "  [fine checkpoint %s: s=%" PRIu64 " window %"
                PRIu64 "..%" PRIu64 "]\n", ok ? "written" : "FAILED", H.s,
                H.kcur, H.kcur + H.qcur * H.m - 1);
    return ok;
}

static bool load_fine(const std::string &base, FineHdr &H,
                      std::vector<u64> &h) {
    FILE *f = fopen(fine_path(base).c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&H, 1, sizeof(H), f) == sizeof(H);
    if (ok) {
        FineHdr T = H;
        T.crc_hdr = 0;
        ok = H.magic == TS_FINE_MAGIC &&
             crc32_of(&T, sizeof(T) - sizeof(u32)) == H.crc_hdr &&
             H.m && H.nw && H.qcur;
    }
    if (ok) {
        h.resize((size_t)H.m * H.nw);
        ok = fread(h.data(), 8, h.size(), f) == h.size() &&
             crc32_of(h.data(), h.size() * 8) == H.crc_h;
    }
    fclose(f);
    if (!ok) h.clear();
    return ok;
}

static void remove_fine(const std::string &base) {
    if (!base.empty()) remove(fine_path(base).c_str());
}

// ---------------------------------------------------------------------------
// interval schedule, replicating factor.cpp -q/-f semantics exactly:
//   f == 0: q = q0 always;  f == 1: q += q0;  f > 1: q = round(f0*q0),
//   f0 *= f.  Then clamp so the interval does not overshoot rhigh.
struct QSched {
    long q0 = 15;
    long q = 0;
    // Euler-optimal mode (--sched opt): interval widths follow the
    // stationarity recurrence of E = sum_j S(k_{j-1})(c_s*w_j + c_g)
    // under the empirical survival law S(k) = (16/9)/k, namely
    //   k_{j+1} = k_j^2/k_{j-1} - G,   G = c_g/c_s in degrees,
    // in width form  w' = w*k/(k-w) - G  which needs only the current
    // position and width -- both restored by checkpoint resume, so the
    // schedule continues exactly across kills with no header changes.
    // Widths are monotone non-decreasing (guard against the recurrence
    // degenerating when parameters are extreme) and snapped to blocks.
    bool opt = false;
    u64 optG = 4000;           // c_g/c_s, degrees (see --sched-G)
    double rho_min = 1.25;     // geometric floor: the stationarity
                               // recurrence's ratios decay by G/k each
                               // interval and can cross 1.0, which the
                               // monotone guard would freeze into a
                               // constant-width tail (observed in the
                               // field as "q capped at 473").  Deep
                               // runs want the ratio to settle, not
                               // die: floor it (see --sched-rho).
    long m = 1;                // block size, for snapping
    void grow(u64 k_end) {
        if (!opt) { q = q + q0; return; }    // linear growth
        u64 w = (u64)q * (u64)m;
        if (!w) {               // first interval: pick k1 by a small
            u64 k0 = k_end;     // argmin scan of the cost model
            double C = 16.0 / 9.0, bestE = 1e300;
            u64 bestk1 = 0;
            u64 lo = (u64)(1.02 * sqrt((double)k0 * (double)optG)) + 1;
            if (lo <= k0) lo = k0 + (u64)m;
            for (int i = 0; i < 96; i++) {
                u64 k1 = lo + (u64)((double)lo * 0.06 * i);
                double E = 0, kp = (double)k0, kc = (double)k1;
                for (int j = 0; j < 200 && kp < 4e18; j++) {
                    E += (C / kp) * ((kc - kp) + (double)optG);
                    double nx = kc * kc / kp - (double)optG;
                    double wn = nx - kc;
                    double wc = kc - kp;
                    if (wn < wc) wn = wc;
                    if (wn < (rho_min - 1.0) * kc) wn = (rho_min - 1.0) * kc;
                    kp = kc;
                    kc = kc + wn;
                    if (kp > 1e14) break;
                }
                if (E < bestE) { bestE = E; bestk1 = k1; }
            }
            w = bestk1 > k0 ? bestk1 - k0 : (u64)m;
        } else {
            // w' = w*k/(k-w) - G, monotone, geometric-floored
            double kd = (double)k_end, wd = (double)w;
            double wn = wd * kd / (kd - wd) - (double)optG;
            if (wn < wd) wn = wd;
            if (wn < (rho_min - 1.0) * kd) wn = (rho_min - 1.0) * kd;
            w = (u64)wn;
        }
        q = (long)((w + (u64)m - 1) / (u64)m);
        if (q < 1) q = 1;
    }
};

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

// ---------------------------------------------------------------------------
// scan parameters and outcome

struct SParams {
    u64 skip = 0;              // no factors of degree <= skip (REQUIRED)
    u64 maxd = 0;              // 0 = no cap
    long q0 = 0;               // 0 = auto (ceil(600/m))
    long zq = 0;               // -z: emit "u" when q would exceed
    long Zq = 0;               // -Z: cap q, keep scanning
    u64 canzass_max = 0;       // 0 = auto (2*r^(2/3))
    bool sched_opt = false;    // Euler-optimal interval schedule
    u64 sched_G = 0;           // c_g/c_s in degrees; 0 = default 4000
    double sched_rho = 0;      // geometric floor ratio; 0 = default 1.25
    int verbose = 0;
    int pend_max = 4;
    long hsnap_mb = 2048;      // RAM budget for fine-DDF h-table snapshots
    std::string ckpt_base;     // empty = no checkpointing
    double ckpt_secs = 15 * 60.0;
    long die_after_blocks = -1;   // test hook: simulate SIGTERM
    long die_in_fine = -1;        // test hook: SIGTERM inside localize
    // error checking: after an interval at least this long, download the
    // next interval's start table BEFORE waiting on the gcd (see scan)
    double ec_presnap_secs = 30.0;
};

struct ScanOut {
    enum Kind { FOUND, GAVE_UP, PRIMITIVE, RANGE, INTERRUPTED,
                EC_RETRY,      // internal: an error check failed, restart
                EC_FATAL       // persistent fault: no result for this s
              } kind;
    u64 d = 0;
    std::string hex;           // for FOUND
    u64 rlo = 0, rhi = 0;      // for RANGE
    u64 degrees_scanned = 0;
    double wall = 0;
    int ec_restarts = 0;       // error-check restarts this s needed
};

// resolve a captured interval gcd g: least factor degree + numerically
// least mask at that degree.  If deg(g) < 2*ka every factor exceeds
// half of deg(g), so g is a single irreducible (trinomials are
// squarefree): skip CanZass entirely.
static bool resolve_gcd(const std::vector<u64> &g, u64 gdeg, u64 ka,
                        u64 &d, std::vector<u64> &mask) {
    if (gdeg > 0 && gdeg < 2 * ka) {
        d = gdeg;
        mask = g;
        size_t need = bits_to_words(gdeg + 1);
        if (mask.size() > need) mask.resize(need);
        return true;
    }
#ifdef HAVE_NTL
    return factor_min_degree_ntl(g.data(), g.size(), d, mask);
#else
    // no-NTL fallback: least degree by squaring mod g; mask by
    // enumeration (tiny degrees only -- selftest territory)
    int64_t dg = poly_deg_v(g);
    if (dg <= 0) return false;
    size_t gw = (size_t)(dg >> 6) + 1;
    std::vector<u64> h(gw, 0);
    h[0] = 2;                                    // x mod g
    for (u64 k = 1; k <= (u64)dg; k++) {
        std::vector<u64> sq(mul_out_words((u64)dg, (u64)dg));
        gf2x_naive_mul(h.data(), (u64)dg, h.data(), (u64)dg, sq.data());
        // reduce mod g
        int64_t da = poly_deg_v(sq);
        while (da >= dg) {
            poly_xor_shift(sq, g, dg, (u64)(da - dg));
            da = poly_deg_v(sq);
        }
        sq.resize(gw);
        h = sq;
        std::vector<u64> hx = h;
        hx[0] ^= 2ULL;                           // h + x
        std::vector<u64> gg;
        u64 gd = poly_gcd_naive(hx.data(), hx.size(), g.data(), g.size(),
                                &gg);
        if (gd > 0) {
            d = k;
            return edf_least_enum(gg, k, mask);
        }
    }
    return false;
#endif
}

static std::string mask_hex(const std::vector<u64> &mask) {
    int64_t d = poly_deg_v(mask);
    std::string out;
    if (d < 0) return out;
    int c = 0;
    static const char *t = "0123456789abcdef";
    for (int64_t i = d; i >= 0; i--) {
        c = 2 * c + (int)((mask[i >> 6] >> (i & 63)) & 1);
        if ((i & 3) == 0) { out.push_back(t[c]); c = 0; }
    }
    return out;
}

// ---------------------------------------------------------------------------
// the per-s scan

// An h-table held in host RAM, with the fold taken on the device when it
// was captured.  The fold is checked again whenever the copy is used, so a
// bit flip in host RAM while it waits (possibly for hours) is caught.
struct HSnap {
    std::vector<u64> h;
    EcFold f;
    bool has_f = false;
};

struct ResumeState {
    bool valid = false;
    CkptHdr H;
    std::vector<u64> h, a;
    EcFold fh;                 // fold of h, when it came from RAM (EC restart)
    bool has_fh = false;
};

// Put a host h-table on the device.  With error checking the host copy is
// compared with the fold recorded at capture (if any), the upload is
// confirmed by a fold of the device table, and the h-table chain restarts
// from it.  False (and logged) if the table cannot be trusted.
static bool load_h(TsGPU &G, const std::vector<u64> &h, const EcFold *expect,
                   const char *what) {
    if (!G.ec_on) { G.h_from_host(h); return true; }
    int rc = G.h_from_host_ec(h, expect);
    if (rc == 1)
        ec_event(ECC_SNAP, "s=%" PRIu64 ": %s no longer matches the fold "
                 "recorded when it was captured (host RAM)", G.s, what);
    else if (rc == 2)
        ec_event(ECC_XFER, "s=%" PRIu64 ": upload of %s could not be "
                 "confirmed in 3 attempts", G.s, what);
    return rc == 0;
}

// D2H of the accumulator for a gcd.  With error checking it must match the
// fold carried from its last product check.
static bool fetch_A(TsGPU &G, std::vector<u64> &hostA) {
    if (!G.ec_on) { G.A_to_host(hostA); return true; }
    if (G.A_to_host_ec(hostA)) return true;
    ec_event(ECC_XFER, "s=%" PRIu64 ": accumulator A does not match its "
             "carried fold after 3 reads (changed on the device since its "
             "last check, or corrupted in transfer)", G.s);
    return false;
}

// Build the h-table at block start k.  With error checking it is verified
// from first principles (gf2_ec.h, mechanism 2) and a failed build is
// repeated; false if it keeps failing.
static bool init_h_checked(TsGPU &G, u64 k) {
    for (int tr = 0;; tr++) {
        G.init_h_at(k);
        if (!G.ec_on) return true;
        if (!g_ec.init_check) { G.ec_chain_reset(); return true; }
        if (G.ec_verify_table_at(k)) return true;
        ec_event(ECC_INIT, "s=%" PRIu64 ": h-table built at degree %" PRIu64
                 " fails (1+y_k)(1+S_k) = 1+y_(k+m)%s", G.s, k,
                 tr < g_ec.max_retry ? " -- rebuilding" : "");
        if (tr >= g_ec.max_retry) return false;
    }
}

// one block: C from h, A *= C, advance h by m degrees.  With error
// checking, returns nonzero if the block's fold checks failed (bit 0:
// Horner build or FFT product, bit 1: reduction); the h-table chain is
// checked separately, once per window (TsGPU::ec_window_check).
static int do_block(TsGPU &G, int verbose) {
    const bool tm = verbose >= 2;
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if (G.ec_on) G.ec_block_pre();
    if (tm) { CUCHK(cudaDeviceSynchronize()); t0 = trin_now_s(); }
    G.horner_C();
    if (ec_inject_now(ECI_C)) G.inject(G.dC, G.r, ECI_C);
    if (tm) { CUCHK(cudaDeviceSynchronize()); t1 = trin_now_s(); }
    G.mul_A_by_C();
    if (tm) { CUCHK(cudaDeviceSynchronize()); t2 = trin_now_s(); }
    for (int t = 0; t < G.m; t++) {
        G.square_all_step();
        if (t == G.m / 2 && ec_inject_now(ECI_HMID))
            G.inject_h(G.dH[G.hcur], ECI_HMID);
    }
    if (ec_inject_now(ECI_H)) G.inject_h(G.dH[G.hcur], ECI_H);
    if (tm) {
        CUCHK(cudaDeviceSynchronize());
        t3 = trin_now_s();
        G.t_horner += t1 - t0;
        G.t_mul += t2 - t1;
        G.t_sq += t3 - t2;
    }
    int rc = 0;
    if (G.ec_on) {
        rc = G.ec_block_verify();
        G.ec_w++;
    }
    return rc;
}

// One block plus the checks that go with it; kblk is the block's first
// degree (for the log).  The h-table chain is checked every --ecw blocks
// and whenever `close_window` is set (end of an interval: the table is
// about to sit idle and A is about to leave the device).  False, already
// logged, if any check tripped.
static bool run_block(TsGPU &G, const SParams &P, u64 kblk,
                      bool close_window) {
    int rc = do_block(G, P.verbose);
    if (!G.ec_on) return true;
    if (rc & 1) {
        ec_event(ECC_PRODUCT, "s=%" PRIu64 " block %" PRIu64 "..%" PRIu64
                 ": phi(A*C) != phi(A)*phi(C)", G.s, kblk, kblk + G.m - 1);
        return false;
    }
    if (rc & 2) {
        ec_event(ECC_REDUCE, "s=%" PRIu64 " block %" PRIu64 "..%" PRIu64
                 ": phi(A*C mod T) != phi(A*C) + phi(q)*phi(T)", G.s, kblk,
                 kblk + G.m - 1);
        return false;
    }
    if (G.ec_w >= g_ec.window || close_window) {
        long nb = G.ec_w;
        if (!G.ec_window_check()) {
            ec_event(ECC_GERBICZ, "s=%" PRIu64 ": sigma(D)+D != S0+S1 over "
                     "the %ld block(s) ending at degree %" PRIu64, G.s, nb,
                     kblk + G.m - 1);
            return false;
        }
    }
    return true;
}

// Does the claimed factor divide T = x^r + x^s + 1?  A wrong gcd or a
// wrong factorisation cannot pass this, so a reported factor is always a
// factor.  (Without NTL only the selftest's kernel suites run.)
static bool verify_factor(const std::vector<u64> &mask, u64 r, u64 s) {
#ifdef HAVE_NTL
    return trinomial_divisible_ntl(mask.data(), mask.size(), r, s);
#else
    (void)mask; (void)r; (void)s;
    return true;
#endif
}

// fineDDF-style localization: a factor is known to lie in
// [ka, ka + q*m - 1] but the interval gcd was too large to factor
// comfortably (deg >= canzass_max).  Rebuild h at ka and bisect,
// exactly like factor.cpp's fineDDF mode.  Returns the final captured
// gcd (all of whose factors lie in the final sub-interval), or sets
// range_only.  `ec_trip` is set (with false returned) when an error
// check failed: the caller then restarts from its last verified state.
static bool localize(TsGPU &G, GcdPool &pool, const SParams &P, u64 r,
                     std::shared_ptr<const std::vector<u64>> T,
                     u64 ka, long q, u64 &gseq,
                     std::vector<u64> &g_out, u64 &gdeg_out,
                     u64 &ka_out, u64 &kb_out,
                     const HSnap *h_at_ka, bool &ec_trip) {
    ec_trip = false;
    if (P.verbose) {
        printf("Entering fine DDF mode: localize in %" PRIu64 "..%" PRIu64
               "\n", ka, ka + (u64)q * G.m - 1);
        fflush(stdout);
    }
    u64 kcur = ka;
    long qcur = q;
    bool have_h = false;
    // (a) an interrupted localization of this very interval?
    if (!P.ckpt_base.empty()) {
        FineHdr FH;
        std::vector<u64> fh;
        if (load_fine(P.ckpt_base, FH, fh) && FH.r == r && FH.s == G.s &&
            FH.skip == P.skip && FH.m == (u64)G.m && FH.nw == (u64)G.nw &&
            FH.ka0 == ka && FH.q0 == (u64)q && FH.kcur >= ka &&
            FH.kcur + FH.qcur * FH.m <= ka + (u64)q * G.m) {
            if (!load_h(G, fh, nullptr, "the fine-DDF checkpoint")) {
                ec_trip = true;
                return false;
            }
            kcur = FH.kcur;
            qcur = (long)FH.qcur;
            have_h = true;
            if (P.verbose) {
                printf("Resuming fine DDF from its checkpoint at %" PRIu64
                       " (q %ld) -- re-exponentiation skipped\n",
                       kcur, qcur);
                fflush(stdout);
            }
        }
    }
    // (b) the snapshot the scan took when this interval began
    if (!have_h && h_at_ka && h_at_ka->h.size() == (size_t)G.m * G.nw) {
        if (!load_h(G, h_at_ka->h, h_at_ka->has_f ? &h_at_ka->f : nullptr,
                    "the interval-start h-table snapshot")) {
            ec_trip = true;
            return false;
        }
        have_h = true;
        if (P.verbose) {
            printf("Using the h-table saved at the interval start "
                   "-- re-exponentiation skipped\n");
            fflush(stdout);
        }
    }
    // (c) last resort: rebuild from scratch (ka*m squarings -- hours at
    //     production scale; only reachable with --hsnap-mb 0, with a
    //     snapshot budget too small for the pending queue, or after a
    //     resume with no fine checkpoint)
    if (!have_h) {
        double t0 = trin_now_s();
        if (!init_h_checked(G, ka)) {
            ec_trip = true;
            return false;
        }
        CUCHK(cudaDeviceSynchronize());
        if (P.verbose) {
            printf("Re-exponentiation took %f\n", trin_now_s() - t0);
            fflush(stdout);
        }
    }
    G.need_hold();
    std::vector<u64> hostA, hsave;
    EcFold hold_f;
    ecf_zero(hold_f);
    for (;;) {
        long qfirst = (qcur == 1) ? 1 : qcur - qcur / 2;
        // the table about to be held and saved must itself be verified
        if (G.ec_on && !G.ec_window_check()) {
            ec_event(ECC_IDLE, "s=%" PRIu64 ": h-table changed before the "
                     "fine-DDF window at %" PRIu64, G.s, kcur);
            ec_trip = true;
            return false;
        }
        G.copy_h(G.dHold, G.dH[G.hcur]);
        if (G.ec_on) G.ec_table_fold(G.dHold, hold_f);
        if (ec_inject_now(ECI_HOLD)) G.inject_h(G.dHold, ECI_HOLD);
        // mirror the window to disk before spending hours inside it
        if (!P.ckpt_base.empty()) {
            FineHdr FH;
            memset(&FH, 0, sizeof(FH));
            FH.r = r; FH.s = G.s; FH.skip = P.skip;
            FH.m = (u64)G.m; FH.nw = (u64)G.nw;
            FH.ka0 = ka; FH.q0 = (u64)q;
            FH.kcur = kcur; FH.qcur = (u64)qcur;
            if (G.ec_on) {
                EcFold f;
                if (!G.h_to_host_ec(hsave, f)) {
                    ec_event(ECC_XFER, "s=%" PRIu64 ": h-table download for "
                             "the fine-DDF checkpoint could not be confirmed",
                             G.s);
                    ec_trip = true;
                    return false;
                }
            } else {
                G.h_to_host(hsave);
            }
            save_fine(P.ckpt_base, FH, hsave, P.verbose);
            if (P.die_in_fine >= 0 && --((SParams &)P).die_in_fine < 0)
                g_stop = 1;                      // test hook
        }
        // With the window on disk, localization is now interruptible: a
        // resumed run picks it up here instead of restarting the hours of
        // rebuild + rescan that used to be required.
        if (g_stop && !P.ckpt_base.empty()) {
            ka_out = kcur;
            kb_out = kcur + (u64)qcur * G.m - 1;
            gdeg_out = 0;
            return false;                        // caller -> INTERRUPTED
        }
        G.reset_A();
        for (long b = 0; b < qfirst; b++)
            if (!run_block(G, P, kcur + (u64)b * G.m, b == qfirst - 1)) {
                ec_trip = true;
                return false;
            }
        CUCHK(cudaDeviceSynchronize());
        if (!fetch_A(G, hostA)) {
            ec_trip = true;
            return false;
        }
        u64 id = gseq++;
        pool.submit(id, hostA, T, G.ec_on ? &G.ecA : nullptr);
        GcdRes R = pool.wait_get(id);
        u64 kb = kcur + (u64)qfirst * G.m - 1;
        if (R.ec_bad) {
            ec_event(ECC_GCD, "s=%" PRIu64 " fineDDF gcd %" PRIu64 "..%"
                     PRIu64 ": %s", G.s, kcur, kb, R.ec_why);
            ec_trip = true;
            return false;
        }
        if (P.verbose) {
            printf("   fineDDF %" PRIu64 "..%" PRIu64 ": gcd took %f, "
                   "deg %" PRIu64 "\n", kcur, kb, R.secs, R.gdeg);
            fflush(stdout);
        }
        if (R.gdeg > 0) {
            bool captured = !R.g.empty();
            if (captured &&
                (R.gdeg < P.canzass_max || R.gdeg < 2 * kcur || qfirst == 1)) {
                g_out = std::move(R.g);
                gdeg_out = R.gdeg;
                ka_out = kcur;
                kb_out = kb;
                remove_fine(P.ckpt_base);
                return true;
            }
            if (qfirst == 1) {          // cannot refine further, uncaptured
                ka_out = kcur;
                kb_out = kb;
                remove_fine(P.ckpt_base);
                return false;
            }
            G.copy_h(G.dH[G.hcur], G.dHold);   // back to state at kcur
            g_hold_restores++;
            if (G.ec_on) {              // the hold table sat idle in VRAM
                EcFold f;
                G.ec_table_fold(G.dH[G.hcur], f);
                if (!ecf_eq(f, hold_f)) {
                    ec_event(ECC_IDLE, "s=%" PRIu64 ": fine-DDF hold table "
                             "for %" PRIu64 " changed while held in VRAM",
                             G.s, kcur);
                    ec_trip = true;
                    return false;
                }
                G.ec_chain_reset();
            }
            qcur = qfirst;
        } else {
            if (qcur == 1) {
                // The interval gcd saw a factor that no sub-interval shows:
                // one of those gcds is wrong.
                if (G.ec_on) {
                    ec_event(ECC_GCD, "s=%" PRIu64 ": fine DDF lost the "
                             "factor at %" PRIu64 " (gcds inconsistent)",
                             G.s, kcur);
                    remove_fine(P.ckpt_base);
                    ec_trip = true;
                    return false;
                }
                if (!g_ec_quiet) {
                    fprintf(stderr, "ERROR: fineDDF lost the factor at %"
                            PRIu64 "\n", kcur);
                    fflush(stderr);
                }
                ka_out = kcur;
                kb_out = kb;
                remove_fine(P.ckpt_base);
                return false;
            }
            kcur += (u64)qfirst * G.m;         // h is already there
            qcur = qcur / 2;
        }
    }
}

// One attempt at s.  Returns ScanOut::EC_RETRY when an error check failed:
// *resume then holds the newest verified state to restart from (valid), or
// is invalid if there is none in RAM.  scan_one_s (below) is the entry
// point and runs the restarts.
static ScanOut scan_one_s_once(TsGPU &G, GcdPool &pool, const SParams &P,
                               u64 r, u64 s, u64 &gseq, ResumeState *resume) {
    ScanOut out;
    double s_t0 = trin_now_s();
    G.set_s(s);
    G.ec_setup(g_ec.on);
    G.t_sq = G.t_horner = G.t_mul = 0;
    const bool ec = G.ec_on;

    auto T = std::make_shared<std::vector<u64>>(bits_to_words(r + 1), 0);
    (*T)[0] |= 1;
    (*T)[s >> 6] |= 1ULL << (s & 63);
    (*T)[r >> 6] |= 1ULL << (r & 63);

    // Swan bound (flip-invariant; computed on the original s)
    bool swan = (((r & 7) == 1) || ((r & 7) == 7)) &&
                ((((s & 1) == 0) && (((2 * r) % s) != 0)) ||
                 (((s & 1) == 1) && (((2 * r) % (r - s)) != 0)));
    u64 rhigh = swan ? (r / 3) : (r / 2);
    bool maxd_limited = false;
    if (P.maxd && P.maxd < rhigh) { rhigh = P.maxd; maxd_limited = true; }
    if (P.verbose && swan && !maxd_limited)
        printf("By Swan's theorem search at most to degree %" PRIu64 "\n",
               rhigh);
    if (P.verbose && maxd_limited)
        printf("Maximal degree checked reduced to %" PRIu64 "\n", rhigh);

    QSched Q;
    Q.q0 = P.q0;
    Q.opt = P.sched_opt;
    Q.optG = P.sched_G ? P.sched_G : 4000;
    if (P.sched_rho > 1.0) Q.rho_min = P.sched_rho;
    Q.m = (long)G.m;
    u64 k;                     // current interval start
    long blk0 = 0;             // blocks of it already folded into A

    // Error checking: the newest state known to be good, kept in RAM --
    // where a failed check restarts from.  It holds exactly what a disk
    // checkpoint would, so the restart reuses the resume path.  Captured
    // only at moments when every degree below it has a clear verdict.
    struct Rollback { bool valid = false; u64 k_iv = 0; long blk = 0, q = 0;
                      std::shared_ptr<HSnap> h; std::vector<u64> a; } rb;

    bool resumed = false;
    if (resume->valid && resume->H.s == s) {
        const CkptHdr &H = resume->H;
        k = H.k_iv;
        Q.q = (long)H.q;
        blk0 = (long)H.blk;
        if (!load_h(G, resume->h, resume->has_fh ? &resume->fh : nullptr,
                    "the saved h-table")) {
            resume->valid = false;         // unusable: restart without it
            out.kind = ScanOut::EC_RETRY;
            out.wall = trin_now_s() - s_t0;
            return out;
        }
        if (H.has_A) G.A_from_host(resume->a);
        else G.reset_A();
        if (ec) {                          // this state is the rollback point
            rb.valid = true;
            rb.k_iv = k; rb.blk = blk0; rb.q = Q.q;
            rb.h = std::make_shared<HSnap>();
            rb.h->h = std::move(resume->h);
            ecf_fold_host(rb.h->f, rb.h->h.data(), rb.h->h.size());
            rb.h->has_f = true;
            if (H.has_A) rb.a = std::move(resume->a);
        }
        resumed = true;
        resume->valid = false;
        if (P.verbose) {
            printf("Resumed s=%" PRIu64 " at interval %" PRIu64
                   " block %ld/%ld\n", s, k, blk0, Q.q);
            fflush(stdout);
        }
    } else {
        resume->valid = false;
        k = P.skip + 1;
        double t0 = trin_now_s();
        if (!init_h_checked(G, k)) {
            out.kind = ScanOut::EC_RETRY;
            out.wall = trin_now_s() - s_t0;
            return out;
        }
        CUCHK(cudaDeviceSynchronize());
        if (P.verbose) {
            printf("Exponentiation took %f\n", trin_now_s() - t0);
            fflush(stdout);
        }
        G.reset_A();
        Q.grow(k);
        if (P.Zq && Q.q > P.Zq) Q.q = P.Zq;
    }

    struct Pend { u64 seq, ka, kb; long q;
                  std::shared_ptr<HSnap> h0; };
    std::deque<Pend> pending;
    u64 vf = k;                // verified frontier: all degrees < vf clear

    // checkpoint candidate: h at the start of an interval whose
    // predecessors were all verified at snapshot time
    struct Cand { bool valid = false; u64 k_iv = 0; long q = 0; double f0 = 1;
                  std::shared_ptr<HSnap> h; } cand;
    // h-table snapshots kept so that a hit can be localized without
    // rebuilding the table (see FineHdr).  One per interval with an
    // outstanding gcd; bounded by --hsnap-mb.
    const double snap_mb = (double)G.m * G.nw * 8.0 / (1024 * 1024);
    long nsnap_max = (P.hsnap_mb <= 0) ? 0 : (long)(P.hsnap_mb / snap_mb);
    if (P.hsnap_mb > 0 && nsnap_max < 1) nsnap_max = 1;
    double last_ckpt = trin_now_s();
    long blocks_done_hook = 0;
    bool ck_trip = false;      // write_ckpt: the state failed verification

    std::vector<u64> hostA;
    // the NEXT interval's start table, saved before a long gcd wait (below)
    std::shared_ptr<HSnap> pre_h0;

    // Nothing unverified is ever saved: the mid-interval flavor first
    // closes the h-table chain window, and both the table and A are
    // downloaded through the fold-checked paths.
    auto write_ckpt = [&](bool exiting) -> void {
        ck_trip = false;
        if (P.ckpt_base.empty()) return;
        CkptHdr H;
        memset(&H, 0, sizeof(H));
        H.magic = TS_CKPT_MAGIC; H.version = 1;
        H.r = r; H.s = s; H.skip = P.skip; H.m = (u64)G.m; H.nw = (u64)G.nw;
        std::vector<u64> aa;
        if (vf == k) {           // current interval fully verified behind us
            H.k_iv = k; H.blk = 0; H.q = (u64)Q.q; H.f0 = 0;
            // mid-interval flavor if we are inside the interval
            // (caller ensures we are between blocks)
            auto snap = std::make_shared<HSnap>();
            long b = blocks_done_hook;
            if (ec) {
                long nb = G.ec_w;
                if (!G.ec_window_check()) {
                    ec_event(ECC_GERBICZ, "s=%" PRIu64 ": sigma(D)+D != S0+S1"
                             " over the %ld block(s) before the checkpoint at"
                             " degree %" PRIu64, s, nb, k + (u64)b * G.m);
                    ck_trip = true;
                } else if (!G.h_to_host_ec(snap->h, snap->f)) {
                    ec_event(ECC_XFER, "s=%" PRIu64 ": h-table download for "
                             "the checkpoint could not be confirmed", s);
                    ck_trip = true;
                } else {
                    snap->has_f = true;
                }
            } else {
                G.h_to_host(snap->h);
            }
            if (!ck_trip && b > 0) {
                H.blk = (u64)b; H.has_A = 1;
                if (!fetch_A(G, aa)) ck_trip = true;
            }
            if (ck_trip) {
                if (exiting)
                    fprintf(stderr, "  [state failed verification: NOT "
                            "checkpointed; the previous checkpoint, if any, "
                            "stands]\n");
                return;
            }
            save_ckpt(P.ckpt_base, H, snap->h, aa, P.verbose);
            if (ec) {
                rb.valid = true;
                rb.k_iv = k; rb.blk = b; rb.q = Q.q;
                rb.h = snap;
                rb.a = aa;
            }
        } else if (cand.valid && cand.h) {
            H.k_iv = cand.k_iv; H.blk = 0; H.q = (u64)cand.q; H.f0 = 0;
            save_ckpt(P.ckpt_base, H, cand.h->h, aa, P.verbose);
        } else if (exiting && P.verbose) {
            fprintf(stderr, "  [no verified state to checkpoint; s=%" PRIu64
                    " will restart from scratch on resume]\n", s);
        }
        last_ckpt = trin_now_s();
    };

    // resolve a localized hit into a result (shared by the verdict path
    // and by the fine-DDF resume path below).  Returns 1 when `out` is
    // filled, 3 when an error check failed (caller restarts).
    auto finish_hit = [&](bool have_g, std::vector<u64> &g, u64 gdeg,
                          u64 rka, u64 rkb) -> int {
        if (!have_g) {
            if (g_stop) {            // interrupted mid-localization; the
                out.kind = ScanOut::INTERRUPTED;   // fine checkpoint holds
                return 1;                          // the window
            }
            out.kind = ScanOut::RANGE;
            out.rlo = rka;
            out.rhi = rkb;
            return 1;
        }
        u64 d = 0;
        std::vector<u64> mask;
        double t0 = trin_now_s();
        bool ok = resolve_gcd(g, gdeg, rka, d, mask);
        if (!ok) {
            fprintf(stderr, "ERROR: could not factor interval gcd (deg %"
                    PRIu64 ") for s=%" PRIu64 "\n", gdeg, s);
            out.kind = ScanOut::RANGE;
            out.rlo = rka;
            out.rhi = rkb;
            return 1;
        }
        if (P.verbose) {
            printf("CanZass took %f, total degree %" PRIu64
                   ", small degree %" PRIu64 "\n",
                   trin_now_s() - t0, gdeg, d);
            fflush(stdout);
        }
        if (ec) {
            // A reported factor must divide T, and nothing a correct gcd of
            // this interval contains can have its least degree above it.
            double tv = trin_now_s();
            bool divides = verify_factor(mask, r, s);
            g_ecs.n_verify++;
            g_ecs.t_verify += trin_now_s() - tv;
            if (!divides) {
                ec_event(ECC_FACTOR, "s=%" PRIu64 ": degree-%" PRIu64
                         " factor from interval %" PRIu64 "..%" PRIu64
                         " does NOT divide the trinomial (gcd or "
                         "factorisation corrupted)", s, d, rka, rkb);
                return 3;
            }
            if (d > rkb) {
                ec_event(ECC_FACTOR, "s=%" PRIu64 ": least factor degree %"
                         PRIu64 " lies above its interval %" PRIu64 "..%"
                         PRIu64 " (gcd corrupted)", s, d, rka, rkb);
                return 3;
            }
        }
        if (!(d >= rka && d <= rkb)) {
            if (!g_ec_quiet)
                fprintf(stderr, "WARNING: s=%" PRIu64 " least degree %" PRIu64
                        " outside hit interval [%" PRIu64 ",%" PRIu64
                        "] -- input not a genuine survivor?\n", s, d, rka,
                        rkb);
            if (ec)
                ec_event(-1, "s=%" PRIu64 ": least factor degree %" PRIu64
                         " is BELOW its hit interval %" PRIu64 "..%" PRIu64
                         ": either the input has a factor of degree <= "
                         "--skip, or an earlier gcd missed it", s, d, rka,
                         rkb);
        }
        out.kind = ScanOut::FOUND;
        out.d = d;
        out.hex = mask_hex(mask);
        return 1;
    };

    // returns 1 if factor resolved (out filled), 0 for clear verdicts,
    // 3 if an error check failed (caller restarts from the rollback state)
    auto consume_verdict = [&](const Pend &pv, GcdRes &R) -> int {
        if (R.ec_bad) {
            ec_event(ECC_GCD, "s=%" PRIu64 " gcd %" PRIu64 "..%" PRIu64
                     ": %s", s, pv.ka, pv.kb, R.ec_why);
            return 3;
        }
        if (P.verbose) {
            printf("   gcd %" PRIu64 "..%" PRIu64 " took %f%s\n",
                   pv.ka, pv.kb, R.secs, R.gdeg ? "  ** HIT **" : "");
            fflush(stdout);
        }
        if (R.gdeg == 0) {
            vf = pv.kb + 1;
            return 0;
        }
        // hit: stop speculation, resolve
        pool.forget_all_pending();
        pending.clear();
        std::vector<u64> g;
        u64 gdeg = R.gdeg, rka = pv.ka, rkb = pv.kb;
        bool have_g = false;
        if (!R.g.empty() && (R.gdeg < P.canzass_max || R.gdeg < 2 * pv.ka)) {
            g = std::move(R.g);
            have_g = true;
        } else {
            bool ltrip = false;
            have_g = localize(G, pool, P, r, T, pv.ka, pv.q, gseq,
                              g, gdeg, rka, rkb, pv.h0.get(), ltrip);
            if (ltrip) return 3;
        }
        return finish_hit(have_g, g, gdeg, rka, rkb);
    };
// consume one verdict inside the scan loops
#define TS_CONSUME(pv, R) do {                 \
        int cv_ = consume_verdict(pv, R);      \
        if (cv_ == 3) goto ec_retry;           \
        if (cv_) goto done;                    \
    } while (0)

    // A fine-DDF localization interrupted by a crash, reboot or SIGTERM
    // resumes directly: every degree below the hit interval was verified
    // clear before it began, so nothing needs rescanning.
    bool fine_resumed = false;
    if (!P.ckpt_base.empty()) {
        FineHdr FH;
        std::vector<u64> fh;
        if (load_fine(P.ckpt_base, FH, fh) && FH.r == r && FH.s == s &&
            FH.skip == P.skip && FH.m == (u64)G.m && FH.nw == (u64)G.nw &&
            FH.q0 && FH.ka0 > P.skip) {
            fh.clear();
            fh.shrink_to_fit();          // localize reloads it itself
            if (P.verbose) {
                printf("Found an interrupted fine DDF for s=%" PRIu64
                       " in %" PRIu64 "..%" PRIu64 "; resuming it\n", s,
                       FH.ka0, FH.ka0 + FH.q0 * (u64)G.m - 1);
                fflush(stdout);
            }
            std::vector<u64> g;
            u64 gdeg = 0, rka = FH.ka0;
            u64 rkb = FH.ka0 + FH.q0 * (u64)G.m - 1;
            bool ltrip = false;
            bool have_g = localize(G, pool, P, r, T, FH.ka0, (long)FH.q0,
                                   gseq, g, gdeg, rka, rkb, nullptr, ltrip);
            if (ltrip) goto ec_retry;
            if (finish_hit(have_g, g, gdeg, rka, rkb) == 3) goto ec_retry;
            fine_resumed = true;
        }
    }

    // main interval loop
    if (!fine_resumed)
    for (;;) {
        if (k > rhigh) break;
        long q = Q.q;
        if (P.verbose) {
            printf("Current q %ld\n", q);
            fflush(stdout);
        }
        u64 k2 = k + (u64)q * G.m - 1;
        if (k2 > rhigh) {                       // clamp to cover rhigh
            q = (long)((rhigh - k + G.m) / G.m);
            k2 = k + (u64)q * G.m - 1;
            if (P.verbose) {
                printf("Reducing q to %ld\n", q);
                fflush(stdout);
            }
            Q.q = q;
        }
        if (P.zq && q > P.zq) {                 // -z: give up
            while (!pending.empty()) {          // a pending gcd may hit
                Pend pv = pending.front();
                pending.pop_front();
                GcdRes R = pool.wait_get(pv.seq);
                TS_CONSUME(pv, R);
            }
            out.kind = ScanOut::GAVE_UP;
            if (P.verbose) {
                printf("q > %ld so skipping further test\n", P.zq);
                fflush(stdout);
            }
            goto done;
        }
        if (P.verbose) {
            printf("Interval %" PRIu64 "..%" PRIu64 ":\n", k, k2);
            fflush(stdout);
        }
        // The table has been idle since the chain was last closed (through
        // the previous interval's gcd, under --pend-max 1): confirm it is
        // unchanged BEFORE it is snapshotted and built upon.
        if (ec && !G.ec_window_check()) {
            ec_event(ECC_IDLE, "s=%" PRIu64 ": h-table changed while idle "
                     "before interval %" PRIu64 "..%" PRIu64, s, k, k2);
            // saved before the wait?  Then put it back: no work is lost.
            if (!(pre_h0 && load_h(G, pre_h0->h, &pre_h0->f,
                                   "the pre-wait h-table snapshot")))
                goto ec_retry;
            g_ecs.repairs++;
            ec_event(-1, "s=%" PRIu64 ": h-table restored from the snapshot "
                     "taken before the gcd wait; continuing", s);
        }
        // h at the interval start: the state fine DDF would otherwise have
        // to rebuild, and (when everything behind us is verified) also the
        // checkpoint candidate.  blk0 != 0 means we resumed INTO this
        // interval, so the device h is past its start and must not be kept.
        std::shared_ptr<HSnap> h0;
        bool need_cand = (!resumed && vf == k);
        bool keep_h0 = (blk0 == 0 && nsnap_max > 0 &&
                        (long)pending.size() + 1 <= nsnap_max);
        if (pre_h0) {
            h0 = pre_h0;                 // already downloaded and verified
            pre_h0.reset();
        } else if (keep_h0 || need_cand) {
            h0 = std::make_shared<HSnap>();
            if (ec) {
                if (!G.h_to_host_ec(h0->h, h0->f)) {
                    ec_event(ECC_XFER, "s=%" PRIu64 ": h-table download at "
                             "the start of interval %" PRIu64 " could not "
                             "be confirmed", s, k);
                    goto ec_retry;
                }
                h0->has_f = true;
            } else {
                G.h_to_host(h0->h);
            }
        }
        if (need_cand) {
            cand.valid = true;
            cand.k_iv = k; cand.q = q; cand.f0 = 0;
            cand.h = h0;
            if (ec) {                    // verified, and nothing pending
                rb.valid = true;         // below it: a restart point
                rb.k_iv = k; rb.blk = 0; rb.q = q;
                rb.h = h0;
                rb.a.clear();
            }
        }
        if (h0 && ec_inject_now(ECI_SNAP)) {    // test hook: RAM rot
            u64 bit = ec_inject_rnd((u64)h0->h.size() * 64);
            h0->h[bit >> 6] ^= 1ULL << (bit & 63);
            ec_inject_note(ECI_SNAP, bit);
        }
        double iv_t0 = trin_now_s(), iv_poll = 0;
        for (long b = blk0; b < q; b++) {
            blocks_done_hook = b;
            // poll verdicts between blocks
            double p0 = trin_now_s();
            while (!pending.empty()) {
                GcdRes R;
                if (!pool.try_get(pending.front().seq, R)) break;
                Pend pv = pending.front();
                pending.pop_front();
                TS_CONSUME(pv, R);
            }
            iv_poll += trin_now_s() - p0;
            if (g_stop || (P.die_after_blocks >= 0 &&
                           --((SParams &)P).die_after_blocks < 0)) {
                CUCHK(cudaDeviceSynchronize());
                write_ckpt(true);
                out.kind = ScanOut::INTERRUPTED;
                goto done;
            }
            if (!P.ckpt_base.empty() &&
                trin_now_s() - last_ckpt > P.ckpt_secs) {
                CUCHK(cudaDeviceSynchronize());
                write_ckpt(false);
                if (ck_trip) goto ec_retry;
            }
            if (!run_block(G, P, k + (u64)b * G.m, b == q - 1))
                goto ec_retry;
        }
        blocks_done_hook = 0;
        CUCHK(cudaDeviceSynchronize());
        double iv_wall = trin_now_s() - iv_t0 - iv_poll;
        out.degrees_scanned += (u64)(q - blk0) * G.m;
        if (P.verbose) {
            double per = iv_wall / (double)((q - blk0) * G.m);
            printf("   squares/products took %.1f (wct %.1f), per term "
                   "%.3f (wct %.3f)\n", iv_wall, iv_wall, per, per);
            if (P.verbose >= 2)
                printf("   [components cumulative: sq %.1f  horner %.1f  mul %.1f]\n",
                       G.t_sq, G.t_horner, G.t_mul);
            fflush(stdout);
        }
        if (!fetch_A(G, hostA)) goto ec_retry;
        u64 id = gseq++;
        pool.submit(id, hostA, T, ec ? &G.ecA : nullptr);
        pending.push_back(Pend{id, k, k2, q,
                               keep_h0 ? h0 : std::shared_ptr<HSnap>()});
        G.reset_A();
        resumed = false;
        blk0 = 0;
        k = k2 + 1;
        // The table now sits idle through this interval's gcd, and a long
        // interval is expensive to redo: save the next interval's start
        // table first, so that an idle flip costs a reload, not a rescan.
        if (ec && k <= rhigh && (int)pending.size() >= P.pend_max &&
            iv_wall >= P.ec_presnap_secs) {
            pre_h0 = std::make_shared<HSnap>();
            if (!G.h_to_host_ec(pre_h0->h, pre_h0->f)) {
                ec_event(ECC_XFER, "s=%" PRIu64 ": h-table download after "
                         "interval %" PRIu64 "..%" PRIu64 " could not be "
                         "confirmed", s, pending.back().ka, k2);
                goto ec_retry;
            }
            pre_h0->has_f = true;
        }
        if (ec_inject_now(ECI_IDLE)) G.inject_h(G.dH[G.hcur], ECI_IDLE);
        if ((int)pending.size() >= P.pend_max) {   // bound speculation
            Pend pv = pending.front();
            pending.pop_front();
            GcdRes R = pool.wait_get(pv.seq);
            TS_CONSUME(pv, R);
        }
        Q.grow(k - 1);
        if (P.Zq && Q.q > P.Zq) Q.q = P.Zq;
    }
    // exhausted the degree range: drain all pending verdicts
    while (!pending.empty()) {
        Pend pv = pending.front();
        pending.pop_front();
        GcdRes R = pool.wait_get(pv.seq);
        TS_CONSUME(pv, R);
    }
    if (!fine_resumed)
        out.kind = maxd_limited ? ScanOut::GAVE_UP : ScanOut::PRIMITIVE;
    goto done;

ec_retry:
    // An error check failed.  Nothing computed since the rollback state is
    // trusted: drop the in-flight gcds and hand that state back.
    pool.forget_all_pending();
    pending.clear();
    resume->valid = false;
    resume->has_fh = false;
    if (rb.valid && rb.h) {
        CkptHdr &H = resume->H;
        memset(&H, 0, sizeof(H));
        H.magic = TS_CKPT_MAGIC; H.version = 1;
        H.r = r; H.s = s; H.skip = P.skip; H.m = (u64)G.m; H.nw = (u64)G.nw;
        H.k_iv = rb.k_iv; H.blk = (u64)rb.blk; H.q = (u64)rb.q;
        H.has_A = rb.blk > 0 ? 1 : 0;
        resume->h = rb.h->h;
        resume->fh = rb.h->f;
        resume->has_fh = rb.h->has_f;
        resume->a = rb.a;
        resume->valid = true;
    }
    out.kind = ScanOut::EC_RETRY;

done:
    if (out.kind != ScanOut::INTERRUPTED) pool.forget_all_pending();
    out.wall = trin_now_s() - s_t0;
    return out;
}
#undef TS_CONSUME

// The per-s entry point: scan s, restarting from the newest verified state
// whenever an error check fails.  A restart point that fails more than
// --ec-retries times in a row is a persistent fault, not a cosmic ray:
// give up (ScanOut::EC_FATAL) rather than report anything for this s.
static ScanOut scan_one_s(TsGPU &G, GcdPool &pool, const SParams &P,
                          u64 r, u64 s, u64 &gseq, ResumeState *resume) {
    ResumeState local;
    ResumeState *RS = resume ? resume : &local;
    double t0 = trin_now_s();
    u64 degs = 0, last_k = (u64)-1, last_blk = (u64)-1;
    int consec = 0, restarts = 0;
    for (;;) {
        ScanOut o = scan_one_s_once(G, pool, P, r, s, gseq, RS);
        degs += o.degrees_scanned;
        o.degrees_scanned = degs;
        o.ec_restarts = restarts;
        if (o.kind != ScanOut::EC_RETRY) {
            o.wall = trin_now_s() - t0;
            if (restarts && o.kind != ScanOut::INTERRUPTED)
                ec_event(-1, "s=%" PRIu64 ": completed after %d restart(s)",
                         s, restarts);
            return o;
        }
        // no state in RAM: fall back to the newest checkpoint on disk
        if (!RS->valid && !P.ckpt_base.empty()) {
            RS->valid = load_ckpt(P.ckpt_base, RS->H, RS->h, RS->a) &&
                        RS->H.r == r && RS->H.s == s &&
                        RS->H.skip == P.skip && RS->H.m == (u64)G.m &&
                        RS->H.nw == (u64)G.nw;
            RS->has_fh = false;
        }
        u64 rk = RS->valid ? RS->H.k_iv : 0;
        u64 rblk = RS->valid ? RS->H.blk : 0;
        consec = (rk == last_k && rblk == last_blk) ? consec + 1 : 1;
        last_k = rk;
        last_blk = rblk;
        if (consec > g_ec.max_retry) {
            ec_event(-1, "s=%" PRIu64 ": %d consecutive failures restarting "
                     "from the same point -- this is a persistent fault, "
                     "giving up on this s", s, consec);
            o.kind = ScanOut::EC_FATAL;
            o.wall = trin_now_s() - t0;
            return o;
        }
        restarts++;
        g_ecs.restarts++;
        if (RS->valid)
            ec_event(-1, "s=%" PRIu64 ": restarting from the verified state "
                     "at interval %" PRIu64 ", block %" PRIu64, s, rk, rblk);
        else
            ec_event(-1, "s=%" PRIu64 ": no verified state to restart from; "
                     "starting this s again", s);
    }
}

// ---------------------------------------------------------------------------
// CPU oracle (per-degree naive scan) for the selftest

struct OracleOut {
    bool survivor = false, found = false;
    u64 d = 0, gdeg = 0;
    std::vector<u64> g, mask;
    bool mask_ok = false;
};

static OracleOut oracle_scan(u64 r, u64 s, u64 skip, u64 maxd) {
    OracleOut o;
    std::vector<u64> T(bits_to_words(r + 1), 0);
    T[0] |= 1;
    T[s >> 6] |= 1ULL << (s & 63);
    T[r >> 6] |= 1ULL << (r & 63);
    std::vector<u64> Fh(bits_to_words(r), 0);
    Fh[0] = 2;
    for (u64 kk = 1; kk <= maxd; kk++) {
        cpu_square_mod(Fh, r, s);
        std::vector<u64> Fx = Fh;
        Fx[0] ^= 2ULL;
        std::vector<u64> g;
        u64 gd = poly_gcd_naive(Fx.data(), Fx.size(), T.data(), T.size(), &g);
        if (gd > 0) {
            if (kk <= skip) return o;                  // not a survivor
            o.survivor = o.found = true;
            o.d = kk;
            o.gdeg = gd;
            o.g = g;
            if (gd == kk) { o.mask = g; o.mask_ok = true; }
#ifdef HAVE_NTL
            else o.mask_ok = edf_least_ntl(g.data(), g.size(), kk, o.mask);
#else
            else if (kk <= 22) o.mask_ok = edf_least_enum(g, kk, o.mask);
#endif
            return o;
        }
    }
    o.survivor = true;                                 // "u"
    return o;
}

static u64 st_rng = 0x243F6A8885A308D3ULL;
static u64 st_rnd() {
    u64 x = st_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return st_rng = x;
}
static std::vector<u64> st_res(u64 r) {                // random residue < r
    std::vector<u64> v(bits_to_words(r));
    for (auto &w : v) w = st_rnd();
    unsigned tb = (unsigned)(r & 63);
    if (tb) v.back() &= (1ULL << tb) - 1;
    return v;
}

// host reference for k_mulx_xor
static void host_mulx_xor(std::vector<u64> &dst, const std::vector<u64> &src,
                          u64 r, u64 s) {
    std::vector<u64> t(src.size() + 1, 0);
    for (size_t i = 0; i < src.size(); i++) {
        t[i] ^= src[i] << 1;
        t[i + 1] ^= src[i] >> 63;
    }
    if ((t[r >> 6] >> (r & 63)) & 1) {
        t[r >> 6] ^= 1ULL << (r & 63);
        t[s >> 6] ^= 1ULL << (s & 63);
        t[0] ^= 1ULL;
    }
    for (size_t i = 0; i < dst.size(); i++) dst[i] ^= t[i];
}

static int selftest(int gcd_threads) {
    int bad = 0;
    printf("tsfactor selftest\n");
    printf("GCD backend: %s\n", gcd_backend());
    // error-checking counters at entry (the selftest may be repeated)
    const u64 ec0_fail = g_ecs.total(), ec0_blocks = g_ecs.blocks.load(),
              ec0_win = g_ecs.windows.load(),
              ec0_rst = g_ecs.restarts.load();
    (void)ec0_fail; (void)ec0_blocks; (void)ec0_win; (void)ec0_rst;

    // (1) squaring: fused vs legacy vs host, many (r, s) incl. edges
    {
        bool ok = true;
        struct RS { u64 r, s; };
        std::vector<RS> cases;
        u64 rs[] = {4423, 4159, 4097, 4229, 9689};
        for (u64 rr : rs) {
            u64 svals[] = {1, 2, 3, 63, 64, 65, 2 * 64 + 1, rr / 3,
                           (rr - 1) / 2 - 1, (rr - 1) / 2};
            for (u64 sv : svals)
                if (sv >= 1 && 2 * sv < rr) cases.push_back({rr, sv});
        }
        for (auto &cs : cases) {
            TsGPU G;
            G.init(cs.r);
            G.set_m(2);
            G.set_s(cs.s);
            for (int t = 0; t < 3 && ok; t++) {
                std::vector<u64> a = st_res(cs.r), gl(G.nw), gf(G.nw), hr = a;
                CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8,
                                 cudaMemcpyHostToDevice));
                G.legacy_sq = true;
                G.square_slot(G.dA, G.dTmp);
                CUCHK(cudaMemcpy(gl.data(), G.dTmp, G.nw * 8,
                                 cudaMemcpyDeviceToHost));
                G.legacy_sq = false;
                G.square_slot(G.dA, G.dTmp);
                CUCHK(cudaMemcpy(gf.data(), G.dTmp, G.nw * 8,
                                 cudaMemcpyDeviceToHost));
                cpu_square_mod(hr, cs.r, cs.s);
                if (gl != hr || gf != hr) {
                    printf("  squaring mismatch r=%" PRIu64 " s=%" PRIu64
                           " (legacy %s fused %s)\n", cs.r, cs.s,
                           gl == hr ? "ok" : "BAD", gf == hr ? "ok" : "BAD");
                    ok = false;
                }
            }
            G.fini();
        }
        printf("squaring: fused vs legacy vs host (%zu cases)      %s\n",
               cases.size(), ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }

    // (2) mulx kernel vs host
    {
        bool ok = true;
        u64 rr = 4423;
        u64 svals[] = {1, 5, 64, 2211};
        for (u64 sv : svals) {
            TsGPU G;
            G.init(rr);
            G.set_m(2);
            G.set_s(sv);
            std::vector<u64> a = st_res(rr), d0 = st_res(rr), dg(G.nw),
                             dh = d0;
            CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8,
                             cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.dTmp, d0.data(), G.nw * 8,
                             cudaMemcpyHostToDevice));
            int blocks, threads;
            CudaFFT::launch_dims(G.nw, blocks, threads);
            k_mulx_xor<<<blocks, threads>>>(G.dTmp, G.dA, G.nw, rr, sv);
            CUCHK(cudaMemcpy(dg.data(), G.dTmp, G.nw * 8,
                             cudaMemcpyDeviceToHost));
            host_mulx_xor(dh, a, rr, sv);
            if (dg != dh) { ok = false; printf("  mulx mismatch s=%" PRIu64
                                              "\n", sv); }
            G.fini();
        }
        printf("mul-by-x kernel vs host                            %s\n",
               ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }

    // (3) init_h + Horner vs direct product of (x + x^(2^(k+i))) mod T
    {
        bool ok = true;
        u64 rr = 4423, sv = 130;
        int ms[] = {2, 3, 5, 8};
        u64 ks[] = {1, 5, 17};
        for (int mi = 0; mi < 4 && ok; mi++)
            for (int ki = 0; ki < 3 && ok; ki++) {
                int m = ms[mi];
                u64 kst = ks[ki];
                TsGPU G;
                G.init(rr);
                G.set_m(m);
                G.set_s(sv);
                G.init_h_at(kst);
                G.horner_C();
                CUCHK(cudaDeviceSynchronize());
                std::vector<u64> Cg(G.nw);
                CUCHK(cudaMemcpy(Cg.data(), G.dC, G.nw * 8,
                                 cudaMemcpyDeviceToHost));
                // host: y_i = x^(2^(kst+i)) mod T; C = prod (x + y_i) mod T
                std::vector<u64> y(bits_to_words(rr), 0);
                y[0] = 2;
                for (u64 t = 0; t < kst; t++) cpu_square_mod(y, rr, sv);
                std::vector<u64> Ch(bits_to_words(rr), 0);
                Ch[0] = 1;
                for (int i = 0; i < m; i++) {
                    std::vector<u64> yx = y;
                    yx[0] ^= 2ULL;                      // x + y_i
                    std::vector<u64> pr(mul_out_words(rr, rr));
                    gf2x_naive_mul(Ch.data(), rr, yx.data(), rr, pr.data());
                    cpu_mod_trinomial(pr, 2 * rr - 1, rr, sv);
                    for (size_t w = 0; w < Ch.size(); w++) Ch[w] = pr[w];
                    cpu_square_mod(y, rr, sv);
                }
                if (Cg != Ch) {
                    printf("  init_h/horner mismatch m=%d kst=%" PRIu64 "\n",
                           m, kst);
                    ok = false;
                }
                G.fini();
            }
        printf("init_h + fused Horner vs host block product        %s\n",
               ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }

    // (3b) FFT variant equivalence: legacy vs fused configurations,
    //      forward byte-equality and inverse roundtrip, at m = 12
    {
        bool okk = true;
        CantorBasis bbf;
        bbf.build();
        CudaFFT Ff;
        Ff.init(12, bbf);
        size_t fn = Ff.n;
        u64 *dbuf;
        CUCHK(cudaMalloc(&dbuf, fn * 8));
        std::vector<u64> hin(fn), href(fn), hout(fn);
        u64 st = 0xabcdef12ULL;
        for (auto &w : hin) {
            st += 0x9E3779B97F4A7C15ULL;
            u64 z = st;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            w = z ^ (z >> 31);
        }
        bool sv_f = gf2c_use_fused;
        int sv_t = gf2c_taylor_lv, sv_b = gf2c_bfly_lv;
        u64 sv_c = gf2c_fuse_max_ch;
        gf2c_use_fused = false;
        CUCHK(cudaMemcpy(dbuf, hin.data(), fn * 8, cudaMemcpyHostToDevice));
        Ff.fwd(dbuf);
        CUCHK(cudaMemcpy(href.data(), dbuf, fn * 8, cudaMemcpyDeviceToHost));
        gf2c_use_fused = true;
        for (int tl : {3, 4, 5})
            for (int bl : {3, 5})
                for (u64 ch : std::vector<u64>{0, 64}) {
                    gf2c_taylor_lv = tl;
                    gf2c_bfly_lv = bl;
                    gf2c_fuse_max_ch = ch;
                    CUCHK(cudaMemcpy(dbuf, hin.data(), fn * 8,
                                     cudaMemcpyHostToDevice));
                    Ff.fwd(dbuf);
                    CUCHK(cudaMemcpy(hout.data(), dbuf, fn * 8,
                                     cudaMemcpyDeviceToHost));
                    if (hout != href) okk = false;
                    Ff.inv(dbuf);
                    CUCHK(cudaMemcpy(hout.data(), dbuf, fn * 8,
                                     cudaMemcpyDeviceToHost));
                    if (hout != hin) okk = false;
                }
        gf2c_use_fused = sv_f;
        gf2c_taylor_lv = sv_t;
        gf2c_bfly_lv = sv_b;
        gf2c_fuse_max_ch = sv_c;
        cudaFree(dbuf);
        Ff.fini();
        printf("FFT variants: legacy vs fused configs (m=12)   %s\n",
               okk ? "PASS" : "FAIL");
        if (!okk) bad++;
    }

    // (3c) error-checking kernels and fold algebra against host
    //      references.  NTL-free.  On a real GPU this is the suite that
    //      validates the checking kernels under true concurrency.
    {
        bool ok_fold = true, ok_blk = true, ok_neg = true, ok_chain = true,
             ok_init = true;
        struct RS { u64 r, s; };
        std::vector<RS> cases;
        u64 rs[] = {607, 3217, 3907, 4423, 9689, 19937};
        for (u64 rr : rs) {
            u64 svals[] = {1, 2, 63, 64, 65, rr / 3, (rr - 1) / 2 - 1,
                           (rr - 1) / 2};
            for (u64 sv : svals)
                if (sv >= 1 && 2 * sv < rr) cases.push_back({rr, sv});
        }
        const int ms[] = {2, 3, 8, 14, 33, 64};
        const bool sv_quiet = g_ec_quiet;
        g_ec_quiet = true;
        size_t ci = 0;
        for (auto &cs : cases) {
            int m = ms[ci++ % 6];
            TsGPU G;
            G.init(cs.r);
            G.set_m(m);
            G.set_s(cs.s);
            G.ec_setup(true);
            int blocks, threads;
            CudaFFT::launch_dims(G.nw, blocks, threads);
            std::vector<u64> hh((size_t)m * G.nw);
            for (int j = 0; j < m; j++) {
                std::vector<u64> v = st_res(cs.r);
                memcpy(&hh[(size_t)j * G.nw], v.data(), G.nw * 8);
            }
            std::vector<u64> a = st_res(cs.r);
            auto load = [&] {
                G.h_from_host(hh);
                G.A_from_host(a);
                k_set_zero<<<blocks, threads>>>(G.ecD, G.nw);
            };
            auto pull = [&](const u64 *d) {
                std::vector<u64> v(G.nw);
                CUCHK(cudaMemcpy(v.data(), d, G.nw * 8,
                                 cudaMemcpyDeviceToHost));
                return v;
            };
            // device folds == host folds (a whole table, one residue)
            load();
            {
                EcFold fd, fh;
                G.ec_table_fold(G.dH[G.hcur], fd);
                ecf_fold_host(fh, hh.data(), hh.size());
                if (!ecf_eq(fd, fh)) ok_fold = false;
                G.ec_fold_dev(G.dA, G.nw, TsGPU::ECO_A);
                G.ec_read(TsGPU::ECO_A, EC_NF, fd.w);
                ecf_fold_host(fh, a.data(), a.size());
                if (!ecf_eq(fd, fh)) ok_fold = false;
            }
            // one clean block: every prediction must match the real data
            G.ec_block_pre();
            G.horner_C();
            G.mul_A_by_C();
            if (G.ec_block_verify()) ok_blk = false;
            {
                std::vector<u64> dD = pull(G.ecD), dC = pull(G.dC),
                                 dAn = pull(G.dA), xs(G.nw, 0);
                for (int j = 0; j < m; j++)
                    for (size_t w = 0; w < G.nw; w++)
                        xs[w] ^= hh[(size_t)j * G.nw + w];
                EcFold fh;
                if (dD != xs) ok_blk = false;          // digest accumulate
                ecf_fold_host(fh, dC.data(), dC.size());
                if (!ecf_eq(fh, G.ecC)) ok_blk = false;  // Horner prediction
                ecf_fold_host(fh, dAn.data(), dAn.size());
                if (!ecf_eq(fh, G.ecA)) ok_blk = false;  // carried phi(A)
            }
            // one flipped bit at each stage must trip the matching check
            for (int site : {ECI_C, ECI_T, ECI_RED, ECI_NONE}) {
                load();
                if (site != ECI_NONE) g_eci.arm(site, 1, 77 + ci);
                G.ec_block_pre();
                if (site == ECI_NONE)      // table changes after the EC pass
                    G.inject_h(G.dH[G.hcur], ECI_H);
                G.horner_C();
                if (ec_inject_now(ECI_C)) G.inject(G.dC, G.r, ECI_C);
                G.mul_A_by_C();
                int rc = G.ec_block_verify();
                g_eci.disarm();
                int want = (site == ECI_RED) ? 2 : 1;
                if (!(rc & want)) ok_neg = false;
            }
            // h-table chain over 3 blocks: clean, then with one flip
            for (int flip = 0; flip < 2; flip++) {
                load();
                G.ec_chain_reset();
                for (int b = 0; b < 3; b++) {
                    G.ec_block_pre();
                    for (int t = 0; t < m; t++) {
                        G.square_all_step();
                        if (flip && b == 1 && t == m / 2)
                            G.inject_h(G.dH[G.hcur], ECI_HMID);
                    }
                    G.ec_w++;
                }
                if (G.ec_window_check() != (flip == 0)) ok_chain = false;
                // idle table: unchanged passes, one flip fails
                if (!G.ec_window_check()) ok_chain = false;
                G.inject_h(G.dH[G.hcur], ECI_IDLE);
                if (G.ec_window_check()) ok_chain = false;
            }
            // closed form of the table digest, on real tables
            if (ci % 3 == 0) {
                for (u64 kst : {(u64)0, (u64)1, (u64)9}) {
                    G.init_h_at(kst);
                    if (!G.ec_verify_table_at(kst)) ok_init = false;
                    G.inject_h(G.dH[G.hcur], ECI_INIT);
                    if (G.ec_verify_table_at(kst)) ok_init = false;
                }
            }
            G.fini();
        }
        g_ec_quiet = sv_quiet;
        printf("EC: device folds vs host (%zu cases)                 %s\n",
               cases.size(), ok_fold ? "PASS" : "FAIL");
        printf("EC: block predictions vs real C, A*C, A*C mod T    %s\n",
               ok_blk ? "PASS" : "FAIL");
        printf("EC: one flipped bit trips the matching check       %s\n",
               ok_neg ? "PASS" : "FAIL");
        printf("EC: h-table chain, clean / one flip / idle flip    %s\n",
               ok_chain ? "PASS" : "FAIL");
        printf("EC: closed-form table digest (init verification)   %s\n",
               ok_init ? "PASS" : "FAIL");
        if (!ok_fold) bad++;
        if (!ok_blk) bad++;
        if (!ok_neg) bad++;
        if (!ok_chain) bad++;
        if (!ok_init) bad++;
    }

    // (4) end-to-end vs oracle at r = 4423
#ifndef HAVE_NTL
    printf("end-to-end vs oracle                               SKIPPED (NTL required)\n");
    printf("fineDDF bisection path (forced)                    SKIPPED (NTL required)\n");
    printf("-z gives up / -Z caps and continues                SKIPPED (NTL required)\n");
    printf("checkpoint kill/resume equivalence                 SKIPPED (NTL required)\n");
    printf("EC: fault-injection sweeps                         SKIPPED (NTL required)\n");
    printf("NOTE: this binary cannot resolve factors from multi-factor\n"
           "      interval gcds.  Ensure ntl_check.cpp is present next to\n"
           "      the Makefile and libntl-dev is installed, then rebuild;\n"
           "      `make tsfactor` prints the NTL detection result.\n");
    (void)gcd_threads;
#else
    {
        const u64 rr = 4423, skip = 5, maxd = 200;
        std::vector<u64> surv;
        std::map<u64, OracleOut> want;
        for (u64 sv = 2; sv <= 350; sv++) {
            OracleOut o = oracle_scan(rr, sv, skip, maxd);
            if (o.survivor) { surv.push_back(sv); want[sv] = std::move(o); }
        }
        u64 nf = 0, nu = 0, nt = 0;
        for (auto &kv : want)
            if (kv.second.found) { nf++; if (kv.second.gdeg != kv.second.d) nt++; }
            else nu++;
        printf("oracle r=%" PRIu64 ": %zu survivors (%" PRIu64 " found, %"
               PRIu64 " u, %" PRIu64 " ties)\n", rr, surv.size(), nf, nu, nt);

        GcdPool pool;
        pool.start(gcd_threads);
        u64 gseq = 0;
        struct Cfg { int m; long q0; long Zq; bool legacy; const char *nm;
                     bool sopt; u64 sG; };
        Cfg cfgs[] = {{4, 3, 0, false, "m=4 q0=3", false, 0},
                      {8, 2, 0, false, "m=8 q0=2", false, 0},
                      {6, 1, 1, true,  "m=6 const-q legacy-sq", false, 0},
                      {14, 15, 0, false, "m=14 q0=15", false, 0},
                      {3, 2, 0, false, "m=3 q0=2", false, 0},
                      {6, 3, 0, false, "m=6 sched-opt G=60", true, 60}};
        for (auto &cf : cfgs) {
            TsGPU G;
            G.init(rr);
            G.set_m(cf.m);
            G.legacy_sq = cf.legacy;
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = cf.q0; P.Zq = cf.Zq;
            P.sched_opt = cf.sopt; P.sched_G = cf.sG;
            P.canzass_max = 2 * 270;   // 2*r^(2/3) ~ 540
            P.verbose = 0;
            u64 mism = 0;
            for (u64 sv : surv) {
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                const OracleOut &o = want[sv];
                bool okk;
                if (o.found) {
                    okk = res.kind == ScanOut::FOUND && res.d == o.d;
                    if (okk && o.mask_ok)
                        okk = res.hex == mask_hex(o.mask);
                } else {
                    okk = res.kind == ScanOut::GAVE_UP;
                }
                if (!okk) {
                    mism++;
                    printf("  MISMATCH cfg[%s] s=%" PRIu64 " kind=%d d=%"
                           PRIu64 " (want found=%d d=%" PRIu64 ")\n",
                           cf.nm, sv, (int)res.kind, res.d, (int)o.found,
                           o.d);
                }
            }
            printf("end-to-end vs oracle [%s]                %s\n", cf.nm,
                   mism ? "FAIL" : "PASS");
            if (mism) bad++;
            G.fini();
        }

        // (5) fineDDF forced on every hit (canzass_max = 1)
        {
            TsGPU G;
            G.init(rr);
            G.set_m(5);
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 4;
            P.canzass_max = 1;
            P.verbose = 0;
            u64 mism = 0;
            // both localize entry paths must agree: reusing the h-table
            // snapshot taken at the interval start, and rebuilding it
            for (int hs = 0; hs < 2; hs++) {
              P.hsnap_mb = hs ? 0 : 2048;
              for (size_t i = 0; i < surv.size(); i += 3) {
                u64 sv = surv[i];
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                const OracleOut &o = want[sv];
                bool okk = o.found ? (res.kind == ScanOut::FOUND &&
                                      res.d == o.d)
                                   : (res.kind == ScanOut::GAVE_UP);
                if (okk && o.found && o.mask_ok)
                    okk = res.hex == mask_hex(o.mask);
                if (!okk) mism++;
              }
            }
            printf("fineDDF bisection path (forced, both h paths)      %s\n",
                   mism ? "FAIL" : "PASS");
            if (mism) bad++;
            G.fini();
        }

        // (6) -z / -Z semantics on a known-deep s
        {
            u64 deep_s = 0, deep_d = 0;
            for (auto &kv : want)
                if (kv.second.found && kv.second.d > 60 &&
                    kv.second.d > deep_d) { deep_s = kv.first;
                                            deep_d = kv.second.d; }
            bool okk = true;
            if (deep_s) {
                TsGPU G;
                G.init(rr);
                G.set_m(4);
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 2;
                P.canzass_max = 540;
                P.verbose = 0;
                P.zq = 3;              // trips before reaching deep_d
                ScanOut r1 = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                        nullptr);
                okk = okk && r1.kind == ScanOut::GAVE_UP;
                P.zq = 0;
                P.Zq = 3;              // capped q but keeps going
                ScanOut r2 = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                        nullptr);
                okk = okk && r2.kind == ScanOut::FOUND && r2.d == deep_d;
                G.fini();
            }
            printf("-z gives up / -Z caps and continues                %s\n",
                   okk ? "PASS" : "FAIL");
            if (!okk) bad++;
        }

        // (7) checkpoint / resume equivalence (kill at various points)
        {
            u64 deep_s = 0, deep_d = 0;
            for (auto &kv : want)
                if (kv.second.found && kv.second.d > 100) {
                    deep_s = kv.first; deep_d = kv.second.d; break;
                }
            bool okk = true;
            if (deep_s) {
                TsGPU G;
                G.init(rr);
                G.set_m(5);
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 2;
                P.canzass_max = 540;
                P.verbose = 0;
                P.ckpt_base = "/tmp/tsfactor_test.ckpt";
                P.ckpt_secs = 1e9;
                long kills[] = {0, 1, 3, 6, 11};
                for (long ka : kills) {
                    remove("/tmp/tsfactor_test.ckpt");
                    remove("/tmp/tsfactor_test.ckpt.1");
                    remove("/tmp/tsfactor_test.ckpt.2");
                    SParams P2 = P;
                    P2.die_after_blocks = ka;
                    ScanOut r1 = scan_one_s(G, pool, P2, rr, deep_s, gseq,
                                            nullptr);
                    if (r1.kind != ScanOut::INTERRUPTED) {
                        // died late enough to finish: fine, just check
                        okk = okk && r1.kind == ScanOut::FOUND &&
                              r1.d == deep_d;
                        continue;
                    }
                    ResumeState RS;
                    RS.valid = load_ckpt(P.ckpt_base, RS.H, RS.h, RS.a);
                    ScanOut r2;
                    if (RS.valid && RS.H.s == deep_s)
                        r2 = scan_one_s(G, pool, P, rr, deep_s, gseq, &RS);
                    else            // no verified state: restart is correct
                        r2 = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                        nullptr);
                    okk = okk && r2.kind == ScanOut::FOUND && r2.d == deep_d;
                    if (!okk) printf("  ckpt kill@%ld: d=%" PRIu64
                                     " want %" PRIu64 "\n", ka, r2.d, deep_d);
                }
                G.fini();
            }
            printf("checkpoint kill/resume equivalence                 %s\n",
                   okk ? "PASS" : "FAIL");
            if (!okk) bad++;
        }

        // (7b) fine DDF interrupted mid-localization and resumed from its
        //      standby checkpoint -- the state that otherwise costs
        //      ka*m squarings (hours at production scale) to rebuild
        {
            const char *fb = "/tmp/tsfactor_fine.ckpt";
            auto wipe = [&] {
                remove(fb);
                remove((std::string(fb) + ".1").c_str());
                remove((std::string(fb) + ".2").c_str());
                remove_fine(fb);
            };
            TsGPU G;
            G.init(rr);
            G.set_m(5);
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 4;
            P.canzass_max = 1;                 // force the fineDDF path
            P.verbose = 0;
            P.ckpt_base = fb;
            P.ckpt_secs = 1e9;
            // find an s whose hit actually goes through localize
            u64 deep_s = 0, deep_d = 0;
            int tries = 0;
            for (auto &kv : want) {
                if (!kv.second.found || kv.second.d <= 40) continue;
                if (++tries > 40) break;
                wipe();
                SParams P2 = P;
                P2.die_in_fine = 0;
                ScanOut r1 = scan_one_s(G, pool, P2, rr, kv.first, gseq,
                                        nullptr);
                g_stop = 0;
                if (r1.kind == ScanOut::INTERRUPTED) {
                    deep_s = kv.first; deep_d = kv.second.d; break;
                }
            }
            bool okk = (deep_s != 0);
            u64 nres = 0;
            for (long kill = 0; okk && kill < 3; kill++) {
                wipe();
                SParams P2 = P;
                P2.die_in_fine = kill;
                ScanOut r1 = scan_one_s(G, pool, P2, rr, deep_s, gseq,
                                        nullptr);
                g_stop = 0;
                if (r1.kind != ScanOut::INTERRUPTED) {   // finished first
                    okk = okk && r1.kind == ScanOut::FOUND &&
                          r1.d == deep_d;
                    continue;
                }
                // fresh process would see only the standby file
                ScanOut r2 = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                        nullptr);
                g_stop = 0;
                nres++;
                okk = okk && r2.kind == ScanOut::FOUND && r2.d == deep_d;
                if (!okk)
                    printf("  fine kill@%ld: kind=%d d=%" PRIu64 " want %"
                           PRIu64 "\n", kill, (int)r2.kind, r2.d, deep_d);
            }
            wipe();
            G.fini();
            printf("fine DDF interrupt/resume (%" PRIu64 " resumed)       "
                   "      %s\n", nres, okk && nres ? "PASS" : "FAIL");
            if (!okk || !nres) bad++;
        }

        // (8) hybrid HGCD gcd vs NTL gcd: CPU-mult and GPU-hook paths,
        //     nonlinear and adversarial F2-linear (LFSR) data, planted
        //     common factors, degenerate shapes
        {
            bool okk = true;
            u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
            hyb_ntl_finish_bits = 64;    // force deep HGCD recursion
            hyb_gpu_min_bits = 2048;     // exercise the hook heavily
            hyb_reset_stats();
            CantorBasis bb8;
            bb8.build();
            TsGpuMul gm8;
            gm8.init(&bb8, 15);
            u64 rs8 = 0xC0FFEEULL;
            auto nl = [&]() { rs8 += 0x9E3779B97F4A7C15ULL; u64 z = rs8;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                return z ^ (z >> 31); };
            auto lf = [&]() { rs8 ^= rs8 << 13; rs8 ^= rs8 >> 7;
                rs8 ^= rs8 << 17; return rs8; };
            auto rvb = [&](u64 bits, bool lin) {
                std::vector<u64> v((size_t)((bits + 63) / 64));
                for (auto &w : v) w = lin ? lf() : nl();
                unsigned tb = (unsigned)((bits - 1) & 63);
                v.back() &= (tb == 63) ? ~0ULL : ((1ULL << (tb + 1)) - 1);
                v.back() |= 1ULL << tb;
                return v;
            };
            auto chk = [&](std::vector<u64> a, std::vector<u64> b) {
                std::vector<u64> g1, g2, g3;
                u64 d1 = poly_gcd_ntl(a.data(), a.size(), b.data(),
                                      b.size(), &g1, ~0ULL);
                u64 d2 = poly_gcd_hybrid(a.data(), a.size(), b.data(),
                                         b.size(), &g2, ~0ULL, &gm8);
                u64 d3 = poly_gcd_hybrid(a.data(), a.size(), b.data(),
                                         b.size(), &g3, ~0ULL, nullptr);
                bool o = d1 == d2 && d1 == d3 &&
                         (d1 == 0 || (g1 == g2 && g1 == g3));
                if (!o) okk = false;
            };
            for (u64 bits : std::vector<u64>{600, 5000, 60000}) {
                for (int lin = 0; lin < 2; lin++) {
                    chk(rvb(bits, lin), rvb(bits - 1 - nl() % 30, lin));
                    // planted common factor via the hook itself
                    u64 gd = 20 + nl() % (bits / 3);
                    auto g = rvb(gd + 1, false);
                    auto u = rvb(bits - gd, lin), w = rvb(bits - gd, lin);
                    std::vector<u64> pa, pb;
                    gm8.mul(g.data(), g.size(), gd + 1, u.data(), u.size(),
                            bits - gd, pa);
                    gm8.mul(g.data(), g.size(), gd + 1, w.data(), w.size(),
                            bits - gd, pb);
                    chk(pa, pb);
                }
                chk(rvb(bits, false), std::vector<u64>());
                { auto sa = rvb(bits, false); chk(sa, sa); }
                chk(rvb(bits, false), std::vector<u64>{1});
                chk(rvb(bits, false), rvb(9, false));
            }
            u64 g8, c8, f8;
            hyb_get_stats(g8, c8, f8);
            hyb_gpu_min_bits = sv_min;
            hyb_ntl_finish_bits = sv_fin;
            gm8.fini();
            printf("hybrid HGCD vs NTL (gpu-mults %" PRIu64 ", fb %" PRIu64
                   ")             %s\n", g8, f8, okk ? "PASS" : "FAIL");
            if (!okk) bad++;
        }

        // (9) end-to-end scan with the hybrid GPU gcd forced through the
        //     pool (thresholds shrunk so 4423-bit gcds take the path)
        {
            u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
            hyb_gpu_min_bits = 512;
            hyb_ntl_finish_bits = 256;
            CantorBasis bb9;
            bb9.build();
            TsGpuMul gm9;
            gm9.init(&bb9, 12);
            g_gm = &gm9;
            g_use_hybrid = true;
            g_hyb_xcheck = true;
            hyb_reset_stats();
            TsGPU G;
            G.init(rr);
            G.set_m(6);
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 3;
            P.canzass_max = 540;
            P.verbose = 0;
            u64 mism = 0;
            for (size_t i = 0; i < surv.size(); i += 4) {
                u64 sv = surv[i];
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                const OracleOut &o = want[sv];
                bool ok9 = o.found ? (res.kind == ScanOut::FOUND &&
                                      res.d == o.d)
                                   : (res.kind == ScanOut::GAVE_UP);
                if (ok9 && o.found && o.mask_ok)
                    ok9 = res.hex == mask_hex(o.mask);
                if (!ok9) mism++;
            }
            // A verdict ends a scan via forget_all_pending(), which does
            // NOT wait for the worker running that gcd -- it may still be
            // inside the hook.  Drain before freeing gm9/G.
            size_t stale9 = pool.drain();
            u64 g9, c9, f9;
            hyb_get_stats(g9, c9, f9);
            g_gm = nullptr;
            g_use_hybrid = false;
            g_hyb_xcheck = false;
            hyb_gpu_min_bits = sv_min;
            hyb_ntl_finish_bits = sv_fin;
            G.fini();
            gm9.fini();
            printf("end-to-end, hybrid gcd forced (gpu-mults %" PRIu64
                   ", fb %" PRIu64 ", drained %zu)   %s\n", g9, f9, stale9,
                   mism ? "FAIL" : "PASS");
            if (mism) bad++;
        }
        // (10) every scan above ran with error checking on: it must not
        //      have raised a single alarm on correct hardware-free data
        {
            u64 nd = g_ecs.total() - ec0_fail,
                nb = g_ecs.blocks.load() - ec0_blocks,
                nwn = g_ecs.windows.load() - ec0_win;
            bool okk = (nd == 0 && nb > 0 && nwn > 0 &&
                        g_ecs.restarts.load() == ec0_rst);
            printf("EC: no false alarms (%" PRIu64 " blocks, %" PRIu64
                   " windows)        %s\n", nb, nwn, okk ? "PASS" : "FAIL");
            if (!okk) bad++;
        }

        // (11) single-fault sweep.  One bit is flipped at the N-th
        //      opportunity of each site.  With checking on the answer must
        //      ALWAYS be right; the same fault with checking off shows
        //      what it would have cost.
        {
            u64 deep_s = 0, deep_d = 0;
            for (auto &kv : want)
                if (kv.second.found && kv.second.mask_ok &&
                    kv.second.d > 100 && kv.second.d > deep_d) {
                    deep_s = kv.first; deep_d = kv.second.d;
                }
            const std::string deep_hex =
                deep_s ? mask_hex(want[deep_s].mask) : std::string();
            const EcCfg sv_cfg = g_ec;
            const bool sv_quiet = g_ec_quiet;
            g_ec_quiet = true;
            // a second target whose hit goes through fine-DDF bisection
            // (with canzass_max = 1) AND makes it step back to its hold
            // table (an interval gcd holding more than one factor does)
            u64 fine_s = 0, fine_d = 0;
            std::string fine_hex;
            for (auto &kv : want) {
                if (!kv.second.found || !kv.second.mask_ok) continue;
                const u64 hr0 = g_hold_restores.load();
                TsGPU G;
                G.init(rr);
                G.set_m(5);
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 2;
                P.canzass_max = 1;
                scan_one_s(G, pool, P, rr, kv.first, gseq, nullptr);
                bool through = g_hold_restores.load() > hr0;
                pool.drain();
                G.fini();
                if (through) {
                    fine_s = kv.first; fine_d = kv.second.d;
                    fine_hex = mask_hex(kv.second.mask);
                    break;
                }
            }
            struct Site { int site; bool fine, dup, must; int nmax; };
            const Site sites[] = {
                {ECI_H, false, false, true, 36},
                {ECI_HMID, false, false, true, 36},
                {ECI_A, false, false, true, 36},
                {ECI_C, false, false, true, 36},
                {ECI_T, false, false, true, 36},
                {ECI_RED, false, false, true, 36},
                {ECI_IDLE, false, false, false, 5},
                {ECI_INIT, false, false, true, 7},
                {ECI_XFER_H, false, false, true, 5},
                {ECI_XFER_A, false, false, true, 5},
                {ECI_SNAP, true, false, false, 5},
                {ECI_HOLD, true, false, false, 5},
                {ECI_JOB, false, false, true, 5},
                {ECI_GCD, false, true, true, 5}};
            u64 t_fired = 0, t_det = 0, t_on_wrong = 0, t_undet = 0,
                t_off_fired = 0, t_off_wrong = 0;
            bool okk = (deep_s != 0 && fine_s != 0);
            for (const Site &S : sites) {
                u64 fired = 0, det = 0, on_wrong = 0, undet = 0,
                    off_fired = 0, off_wrong = 0;
                const u64 tgt_s = S.fine ? fine_s : deep_s;
                const u64 tgt_d = S.fine ? fine_d : deep_d;
                const std::string &tgt_hex = S.fine ? fine_hex : deep_hex;
                for (int N = 1; N <= S.nmax; N++) {
                    for (int pass = 0; pass < 2; pass++) {
                        g_ec = sv_cfg;
                        g_ec.on = (pass == 0);
                        g_ec.gcd_dup = S.dup && pass == 0;
                        g_ec.log_path.clear();
                        TsGPU G;
                        G.init(rr);
                        G.set_m(5);
                        SParams P;
                        P.skip = skip; P.maxd = maxd; P.q0 = 2;
                        P.canzass_max = S.fine ? 1 : 540;
                        P.verbose = 0;
                        // serial verdicts: every fault lands in work that
                        // counts, and the run is reproducible
                        P.pend_max = 1;
                        u64 e0 = g_ecs.total() + g_ecs.xfer_retries.load() +
                                 g_ecs.gcd_reruns.load();
                        g_eci.arm(S.site, N, 1000 * (u64)S.site + (u64)N);
                        ScanOut res = scan_one_s(G, pool, P, rr, tgt_s,
                                                 gseq, nullptr);
                        bool did = g_eci.fired.load() > 0;
                        g_eci.disarm();
                        pool.drain();
                        u64 e1 = g_ecs.total() + g_ecs.xfer_retries.load() +
                                 g_ecs.gcd_reruns.load();
                        bool right = res.kind == ScanOut::FOUND &&
                                     res.d == tgt_d && res.hex == tgt_hex;
                        if (pass == 0) {
                            if (!right) on_wrong++;
                            if (did) {
                                fired++;
                                if (e1 > e0) det++;
                                else if (S.must) undet++;
                            }
                        } else if (did) {
                            off_fired++;
                            if (!right) off_wrong++;
                        }
                        G.fini();
                    }
                }
                printf("  %-6s fired %2" PRIu64 "  detected %2" PRIu64
                       "  wrong WITH checking %" PRIu64
                       "  | wrong WITHOUT %2" PRIu64 " of %2" PRIu64 "\n",
                       eci_name[S.site], fired, det, on_wrong, off_wrong,
                       off_fired);
                if (on_wrong || undet || !fired) okk = false;
                if (S.fine && !det) okk = false;   // must be seen at least once
                t_fired += fired; t_det += det; t_on_wrong += on_wrong;
                t_undet += undet; t_off_fired += off_fired;
                t_off_wrong += off_wrong;
            }
            g_ec = sv_cfg;
            g_ec_quiet = sv_quiet;
            printf("EC: single-fault sweep: %" PRIu64 " faults, %" PRIu64
                   " detected, %" PRIu64 " wrong answers;\n"
                   "    the same faults unchecked: %" PRIu64 " of %" PRIu64
                   " wrong            %s\n", t_fired, t_det, t_on_wrong,
                   t_off_wrong, t_off_fired, okk ? "PASS" : "FAIL");
            if (!okk) bad++;

            // (11b) the same block-level faults with speculation on (default
            //       --pend-max): a fault may now land in work that a hit
            //       verdict discards, so only the answer is asserted
            {
                g_ec_quiet = true;
                u64 runs = 0, wrong = 0;
                for (int site : {ECI_H, ECI_A, ECI_C, ECI_T, ECI_RED,
                                 ECI_IDLE, ECI_JOB})
                    for (int N = 1; N <= 34; N += 3) {
                        TsGPU G;
                        G.init(rr);
                        G.set_m(5);
                        SParams P;
                        P.skip = skip; P.maxd = maxd; P.q0 = 2;
                        P.canzass_max = 540;
                        P.verbose = 0;
                        g_eci.arm(site, N, 31 * (u64)site + (u64)N);
                        ScanOut res = scan_one_s(G, pool, P, rr, deep_s,
                                                 gseq, nullptr);
                        g_eci.disarm();
                        pool.drain();
                        runs++;
                        if (!(res.kind == ScanOut::FOUND && res.d == deep_d &&
                              res.hex == deep_hex)) wrong++;
                        G.fini();
                    }
                g_ec_quiet = sv_quiet;
                printf("EC: faults under speculation (%" PRIu64 " runs, %"
                       PRIu64 " wrong)          %s\n", runs, wrong,
                       wrong ? "FAIL" : "PASS");
                if (wrong) bad++;
            }

            // (12) recovery paths
            {
                const EcCfg sv2 = g_ec;
                g_ec.log_path.clear();
                g_ec_quiet = true;
                bool ok_fatal, ok_mid, ok_rep;
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 2;
                P.canzass_max = 540;
                P.verbose = 0;
                auto right = [&](const ScanOut &o) {
                    return o.kind == ScanOut::FOUND && o.d == deep_d &&
                           o.hex == deep_hex;
                };
                {   // a fault at EVERY block is persistent: give up cleanly
                    TsGPU G;
                    G.init(rr);
                    G.set_m(5);
                    g_eci.arm(ECI_H, -1, 5);
                    ScanOut o = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                           nullptr);
                    g_eci.disarm();
                    pool.drain();
                    ok_fatal = (o.kind == ScanOut::EC_FATAL &&
                                o.ec_restarts == g_ec.max_retry);
                    G.fini();
                }
                {   // restart from a mid-interval state (checkpoint taken
                    // before every block), not from the interval start
                    const char *cb = "/tmp/tsfactor_ec.ckpt";
                    auto wipe = [&] {
                        remove(cb);
                        remove((std::string(cb) + ".1").c_str());
                        remove((std::string(cb) + ".2").c_str());
                        remove_fine(cb);
                    };
                    wipe();
                    TsGPU G;
                    G.init(rr);
                    G.set_m(5);
                    SParams P2 = P;
                    P2.ckpt_base = cb;
                    P2.ckpt_secs = -1;
                    u64 r0 = g_ecs.restarts.load();
                    g_eci.arm(ECI_H, 9, 6);
                    ScanOut o = scan_one_s(G, pool, P2, rr, deep_s, gseq,
                                           nullptr);
                    g_eci.disarm();
                    pool.drain();
                    ok_mid = right(o) && g_ecs.restarts.load() == r0 + 1;
                    wipe();
                    G.fini();
                }
                {   // idle flip during the gcd wait, table saved before the
                    // wait: repaired in place, nothing rescanned
                    TsGPU G;
                    G.init(rr);
                    G.set_m(5);
                    SParams P3 = P;
                    P3.ec_presnap_secs = 0;
                    P3.pend_max = 1;
                    u64 r0 = g_ecs.restarts.load(),
                        p0 = g_ecs.repairs.load();
                    g_eci.arm(ECI_IDLE, 1, 7);
                    ScanOut o = scan_one_s(G, pool, P3, rr, deep_s, gseq,
                                           nullptr);
                    g_eci.disarm();
                    pool.drain();
                    ok_rep = right(o) && g_ecs.restarts.load() == r0 &&
                             g_ecs.repairs.load() == p0 + 1;
                    G.fini();
                }
                g_ec = sv2;
                g_ec_quiet = sv_quiet;
                printf("EC: persistent fault gives up cleanly              "
                       "%s\n", ok_fatal ? "PASS" : "FAIL");
                printf("EC: restart from a mid-interval verified state     "
                       "%s\n", ok_mid ? "PASS" : "FAIL");
                printf("EC: idle flip repaired from the pre-wait snapshot  "
                       "%s\n", ok_rep ? "PASS" : "FAIL");
                if (!ok_fatal) bad++;
                if (!ok_mid) bad++;
                if (!ok_rep) bad++;
            }
        }

        // (13) certified gcd engine: agrees with NTL on the same battery as
        //      suite (8), CPU-mult and GPU-hook paths, always certified;
        //      then a fault at each verified step in turn must never yield
        //      a certified wrong answer
        {
            bool ok_eq = true, ok_flt = true;
            u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
            hyb_ntl_finish_bits = 64;    // force a long verified chain
            hyb_gpu_min_bits = 2048;
            hyb_cert_xgcd = false;
            CantorBasis bb13;
            bb13.build();
            TsGpuMul gm13;
            gm13.init(&bb13, 15);
            u64 rs13 = 0xFACADEULL;
            auto nl = [&]() { rs13 += 0x9E3779B97F4A7C15ULL; u64 z = rs13;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                return z ^ (z >> 31); };
            auto rvb = [&](u64 bits) {
                std::vector<u64> v((size_t)((bits + 63) / 64));
                for (auto &w : v) w = nl();
                unsigned tb = (unsigned)((bits - 1) & 63);
                v.back() &= (tb == 63) ? ~0ULL : ((1ULL << (tb + 1)) - 1);
                v.back() |= 1ULL << tb;
                return v;
            };
            u64 ncert = 0, nflt = 0, ndet = 0;
            auto chk = [&](const std::vector<u64> &a,
                           const std::vector<u64> &b) {
                std::vector<u64> g1;
                u64 d1 = poly_gcd_ntl(a.data(), a.size(), b.data(), b.size(),
                                      &g1, ~0ULL);
                EcFold fa, fb;
                ecf_fold_host(fa, a.data(), a.size());
                ecf_fold_host(fb, b.data(), b.size());
                for (int hook = 0; hook < 3; hook++) {
                    std::vector<u64> g2;
                    int st = -1;
                    hyb_fault_at = 0;
                    hyb_cert_xgcd = (hook == 2);        // 2: the XGCD mode
                    u64 d2 = poly_gcd_certified(a.data(), a.size(), b.data(),
                                                b.size(), &g2, ~0ULL,
                                                hook == 1 ? &gm13 : nullptr,
                                                &fa, &fb, &st);
                    hyb_cert_xgcd = false;
                    if (st != GCD_CERT_OK || d2 != d1 || (d1 && g2 != g1))
                        ok_eq = false;
                    ncert++;
                }
                {   // the chain declines: the answer must still come back
                    // certified (by the extended gcd), and be the same
                    std::vector<u64> g2;
                    int st = -1;
                    hyb_force_fallback = true;
                    u64 d2 = poly_gcd_certified(a.data(), a.size(), b.data(),
                                                b.size(), &g2, ~0ULL, nullptr,
                                                &fa, &fb, &st);
                    hyb_force_fallback = false;
                    if (st != GCD_CERT_OK || d2 != d1 || (d1 && g2 != g1))
                        ok_eq = false;
                    ncert++;
                }
                if (hyb_threads_ok()) {
                    // the products of each matrix step on three threads
                    std::vector<u64> g2;
                    int st = -1;
                    const int sv_t = hyb_threads;
                    const u64 sv_mb = hyb_mt_min_bits;
                    hyb_threads = 3;
                    hyb_mt_min_bits = 600;
                    u64 d2 = poly_gcd_certified(a.data(), a.size(), b.data(),
                                                b.size(), &g2, ~0ULL, nullptr,
                                                &fa, &fb, &st);
                    hyb_threads = sv_t;
                    hyb_mt_min_bits = sv_mb;
                    if (st != GCD_CERT_OK || d2 != d1 || (d1 && g2 != g1))
                        ok_eq = false;
                    ncert++;
                }
                // one fault per run, at step 1, 2, 3, ...: the engine must
                // answer "certificate failed" or the right gcd, never a
                // certified wrong one
                for (long N = 1; N <= 40; N++) {
                    std::vector<u64> g2;
                    int st = -1;
                    hyb_fault_at = N;
                    hyb_fault_reset();
                    u64 d2 = poly_gcd_certified(a.data(), a.size(), b.data(),
                                                b.size(), &g2, ~0ULL, nullptr,
                                                &fa, &fb, &st);
                    const bool fired = hyb_fault_count() >= N;
                    hyb_fault_at = 0;
                    nflt++;
                    if (st == GCD_CERT_FAIL) ndet++;
                    if (fired != (st == GCD_CERT_FAIL)) ok_flt = false;
                    if (st != GCD_CERT_FAIL &&
                        (st != GCD_CERT_OK || d2 != d1 || (d1 && g2 != g1)))
                        ok_flt = false;
                }
                // a wrong input fold is refused outright
                {
                    EcFold bad = fa;
                    bad.w[7] ^= 1;
                    std::vector<u64> g2;
                    int st = -1;
                    poly_gcd_certified(a.data(), a.size(), b.data(), b.size(),
                                       &g2, ~0ULL, nullptr, &bad, &fb, &st);
                    if (st != GCD_CERT_INPUT) ok_flt = false;
                }
            };
            for (u64 bits : std::vector<u64>{700, 5000, 40000}) {
                chk(rvb(bits), rvb(bits - 1 - nl() % 30));
                u64 gd = 20 + nl() % (bits / 3);
                auto g = rvb(gd + 1);
                auto u = rvb(bits - gd), w = rvb(bits - gd);
                std::vector<u64> pa, pb;
                gm13.mul(g.data(), g.size(), gd + 1, u.data(), u.size(),
                         bits - gd, pa);
                gm13.mul(g.data(), g.size(), gd + 1, w.data(), w.size(),
                         bits - gd, pb);
                chk(pa, pb);                       // planted common factor
                { auto sa = rvb(bits); chk(sa, sa); }
                chk(rvb(bits), rvb(9));
                // trinomial vs dense: the production shape
                {
                    u64 rr2 = bits | 1, ss2 = rr2 / 3;
                    std::vector<u64> T2(bits_to_words(rr2 + 1), 0);
                    T2[0] |= 1;
                    T2[ss2 >> 6] |= 1ULL << (ss2 & 63);
                    T2[rr2 >> 6] |= 1ULL << (rr2 & 63);
                    chk(rvb(rr2 - 1), T2);
                }
            }
            hyb_gpu_min_bits = sv_min;
            hyb_ntl_finish_bits = sv_fin;
            hyb_cert_xgcd = false;
            gm13.fini();
            printf("EC: certified gcd == NTL (%" PRIu64 " gcds)                "
                   "  %s\n", ncert, ok_eq ? "PASS" : "FAIL");
            printf("EC: certified gcd under faults (%" PRIu64 " runs, %" PRIu64
                   " caught) %s\n", nflt, ndet,
                   (ok_flt && ndet) ? "PASS" : "FAIL");
            if (!ok_eq) bad++;
            if (!ok_flt || !ndet) bad++;
        }

        // (14) end-to-end with the gcd certified (thresholds shrunk so the
        //      4423-bit gcds run a real chain), clean and with a fault
        //      inside the gcd of every scan
        {
            const EcCfg sv_cfg = g_ec;
            const bool sv_quiet = g_ec_quiet;
            u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
            hyb_gpu_min_bits = 512;
            hyb_ntl_finish_bits = 256;
            g_ec.gcd_cert = true;
            g_ec.log_path.clear();
            CantorBasis bb14;
            bb14.build();
            TsGpuMul gm14;
            gm14.init(&bb14, 12);
            u64 mism = 0, runs = 0;
            const u64 c0 = g_ecs.gcd_cert.load(), u0 = g_ecs.gcd_uncert.load(),
                      d0 = g_ecs.total(), rr0 = g_ecs.gcd_reruns.load();
            for (int hook = 0; hook < 3; hook++) {
                g_gm = hook == 1 ? &gm14 : nullptr;
                hyb_cert_xgcd = (hook == 2);        // 2: the XGCD mode
                TsGPU G;
                G.init(rr);
                G.set_m(6);
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 3;
                P.canzass_max = 540;
                P.verbose = 0;
                for (size_t i = hook; i < surv.size(); i += 4) {
                    u64 sv = surv[i];
                    ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq,
                                             nullptr);
                    const OracleOut &o = want[sv];
                    bool okk = o.found ? (res.kind == ScanOut::FOUND &&
                                          res.d == o.d)
                                       : (res.kind == ScanOut::GAVE_UP);
                    if (okk && o.found && o.mask_ok)
                        okk = res.hex == mask_hex(o.mask);
                    runs++;
                    if (!okk) mism++;
                }
                pool.drain();
                G.fini();
            }
            hyb_cert_xgcd = false;       // the fault runs below use the chain
            const u64 ncert = g_ecs.gcd_cert.load() - c0;
            bool ok_clean = (mism == 0 && ncert > 0 &&
                             g_ecs.gcd_uncert.load() == u0 &&
                             g_ecs.total() == d0);
            // now with one fault inside the gcd engine per scan
            g_ec_quiet = true;
            g_gm = nullptr;
            u64 fm = 0, fruns = 0;
            {
                TsGPU G;
                G.init(rr);
                G.set_m(6);
                SParams P;
                P.skip = skip; P.maxd = maxd; P.q0 = 3;
                P.canzass_max = 540;
                P.verbose = 0;
                P.pend_max = 1;
                for (size_t i = 0; i < surv.size(); i += 5) {
                    u64 sv = surv[i];
                    hyb_fault_at = 1 + (long)(i % 7);
                    hyb_fault_reset();
                    ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq,
                                             nullptr);
                    hyb_fault_at = 0;
                    const OracleOut &o = want[sv];
                    bool okk = o.found ? (res.kind == ScanOut::FOUND &&
                                          res.d == o.d)
                                       : (res.kind == ScanOut::GAVE_UP);
                    if (okk && o.found && o.mask_ok)
                        okk = res.hex == mask_hex(o.mask);
                    fruns++;
                    if (!okk) fm++;
                }
                pool.drain();
                G.fini();
            }
            const u64 nrr = g_ecs.gcd_reruns.load() - rr0;
            g_gm = nullptr;
            g_ec = sv_cfg;
            g_ec_quiet = sv_quiet;
            hyb_gpu_min_bits = sv_min;
            hyb_ntl_finish_bits = sv_fin;
            gm14.fini();
            printf("EC: scans with certified gcd (%" PRIu64 " scans, %" PRIu64
                   " gcds)    %s\n", runs, ncert, ok_clean ? "PASS" : "FAIL");
            printf("EC: faults inside the gcd (%" PRIu64 " scans, %" PRIu64
                   " re-runs)     %s\n", fruns, nrr,
                   (fm == 0 && nrr > 0) ? "PASS" : "FAIL");
            if (!ok_clean) bad++;
            if (fm || !nrr) bad++;
        }
        pool.stop();
    }
#endif // HAVE_NTL

#ifdef HAVE_NTL
    printf(bad ? "SELFTEST: %d FAILURE(S)\n" : "SELFTEST: all PASS\n", bad);
#else
    printf(bad ? "SELFTEST: %d FAILURE(S)\n"
               : "SELFTEST: kernel suites PASS; resolution suites SKIPPED "
                 "(built without NTL)\n", bad);
#endif
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// bench: component times at full scale + m recommendation

static int bench(u64 r, u64 s) {
    if (!s) s = (r >= 5) ? (r - 1) / 2 : 1;
    printf("bench r=%" PRIu64 " s=%" PRIu64 "\n", r, s);
    TsGPU G;
    G.init(r);
    G.set_m(24);
    G.set_s(s);
    std::vector<u64> a = st_res(r);
    CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.dH[0], a.data(), G.nw * 8, cudaMemcpyHostToDevice));
    auto timeit = [&](const char *nm, int reps, auto fn) -> double {
        fn();                                   // warm-up
        CUCHK(cudaDeviceSynchronize());
        double t0 = trin_now_s();
        for (int i = 0; i < reps; i++) fn();
        CUCHK(cudaDeviceSynchronize());
        double dt = (trin_now_s() - t0) / reps;
        printf("  %-28s %10.3f ms\n", nm, dt * 1e3);
        return dt;
    };
    G.legacy_sq = true;
    double t_leg = timeit("squaring (legacy 3-kernel)", 10,
                          [&] { G.square_slot(G.dA, G.dTmp); });
    G.legacy_sq = false;
    double t_fus = timeit("squaring (fused)", 10,
                          [&] { G.square_slot(G.dA, G.dTmp); });
    double t_hor = timeit("Horner (m=24)", 10, [&] { G.horner_C(); });
    double t_mul = timeit("modmul (FFT + reduce)", 4,
                          [&] { G.mul_A_by_C(); });
    {   // component breakdown of the modmul (drives the C2 FFT work)
        size_t ca = (size_t)((r + 31) >> 5);
        int blocks, threads;
        CudaFFT::launch_dims(G.F.n, blocks, threads);
        timeit("  pack (one operand)", 4, [&] {
            k_pack<<<blocks, threads>>>(G.dA, G.nw, G.dFA, ca, G.F.n); });
        timeit("  forward FFT (one of 2)", 4, [&] { G.F.fwd(G.dFA); });
        timeit("  pointwise GF(2^64)", 4, [&] {
            k_pointwise<<<blocks, threads>>>(G.dFA, G.dFB, G.F.n); });
        timeit("  inverse FFT", 4, [&] { G.F.inv(G.dFA); });
        int b2, t2;
        CudaFFT::launch_dims(G.nw2, b2, t2);
        timeit("  overlap-add", 4, [&] {
            k_overlap_add<<<b2, t2>>>(G.dFA, 2 * ca - 1, G.dS0, G.nw2); });
        timeit("  fold reduce (2 passes)", 4, [&] {
            G.reduce_into(G.dTmp); });
    }
    double t_sq = t_fus < t_leg ? t_fus : t_leg;
    // error-checking costs at full scale (m = 24 here; the h-table pass
    // and the chain check scale with m, the product folds do not)
    G.ec_setup(true);
    G.ec_chain_reset();
    printf("error checking (per block unless noted):\n");
    double te_pre = timeit("EC h-table pass (m=24)", 10,
                           [&] { G.ec_block_pre(); });
    double te_fold = timeit("EC product + result folds", 10,
                            [&] { G.ec_after_mul(); G.ec_after_reduce(); });
    double te_ver = timeit("EC readback + host algebra", 10,
                           [&] { G.ec_block_verify(); });
    double te_win = timeit("EC chain check (per window)", 4,
                           [&] { G.ec_w = 1; G.ec_window_check(); });
    G.ec_on = false;
    printf("per-degree projection (m*t_sq + (t_mul + t_horner(m))/m), and "
           "the error-checking overhead at --ecw %ld:\n", g_ec.window);
    int best_m = 8;
    double best_c = 1e100;
    for (int m : {8, 12, 16, 20, 24, 32, 40, 48, 56, 64}) {
        double c = m * t_sq + (t_mul + t_hor * m / 24.0) / m;
        double ce = (te_pre * m / 24.0 + te_fold + te_ver +
                     te_win * m / 24.0 / (double)g_ec.window) / m;
        printf("  m=%-3d  %8.2f ms/degree   + EC %6.3f ms (%.2f%%)\n", m,
               c * 1e3, ce * 1e3, 100.0 * ce / c);
        if (c < best_c) { best_c = c; best_m = m; }
    }
    int mo = (int)(sqrt(t_mul / t_sq) + 0.5) & ~1;
    if (mo < 8) mo = 8;
    if (mo > 64) mo = 64;
    {   // FFT autotune: verify + time transform variants; the winner is
        // left active (so a following --bench-gcd uses it) and printed
        // as environment settings that production runs pick up.
        printf("FFT autotune (fwd+inv per config; ~1 min on GPU):\n");
        size_t fn = G.F.n;
        std::vector<u64> hin(fn), href(fn), hout(fn);
        u64 st = 0xfeedfaceULL;
        for (auto &w : hin) {
            st += 0x9E3779B97F4A7C15ULL;
            u64 z = st;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            w = z ^ (z >> 31);
        }
        auto push = [&] {
            CUCHK(cudaMemcpy(G.dFA, hin.data(), fn * 8,
                             cudaMemcpyHostToDevice));
        };
        auto pull = [&](std::vector<u64> &h) {
            CUCHK(cudaDeviceSynchronize());
            CUCHK(cudaMemcpy(h.data(), G.dFA, fn * 8,
                             cudaMemcpyDeviceToHost));
        };
        auto timecfg = [&]() -> double {
            G.F.fwd(G.dFA); G.F.inv(G.dFA);        // warm-up
            CUCHK(cudaDeviceSynchronize());
            double t0 = trin_now_s();
            for (int i2 = 0; i2 < 3; i2++) { G.F.fwd(G.dFA); G.F.inv(G.dFA); }
            CUCHK(cudaDeviceSynchronize());
            return (trin_now_s() - t0) / 3;
        };
        bool sv_f = gf2c_use_fused;
        int sv_t = gf2c_taylor_lv, sv_b = gf2c_bfly_lv;
        u64 sv_c = gf2c_fuse_max_ch;
        gf2c_use_fused = false;
        push();
        G.F.fwd(G.dFA);
        pull(href);
        double t_legacy = timecfg();
        printf("  %-34s %8.1f ms\n", "legacy per-level", t_legacy * 1e3);
        double best_t = t_legacy;
        int best_tl = 0, best_bl = 0;
        u64 best_ch = 0;                            // 0/0/0 = legacy
        gf2c_use_fused = true;
        for (int tl : {2, 3, 4, 5}) {
            for (int bl : {2, 3, 4, 5}) {
                for (u64 ch : std::vector<u64>{0, (u64)1 << 16, (u64)1 << 12}) {
                    gf2c_taylor_lv = tl;
                    gf2c_bfly_lv = bl;
                    gf2c_fuse_max_ch = ch;
                    push();
                    G.F.fwd(G.dFA);
                    pull(hout);
                    bool okf = hout == href;
                    G.F.inv(G.dFA);
                    pull(hout);
                    bool okr = hout == hin;
                    double tc = timecfg();
                    char nm[64];
                    snprintf(nm, sizeof nm, "fused tlv=%d blv=%d maxch=%s",
                             tl, bl,
                             ch == 0 ? "off" : (ch == ((u64)1 << 16) ? "2^16"
                                                                     : "2^12"));
                    printf("  %-34s %8.1f ms%s\n", nm, tc * 1e3,
                           (okf && okr) ? "" : "  VERIFY-FAIL");
                    if (okf && okr && tc < best_t) {
                        best_t = tc;
                        best_tl = tl; best_bl = bl; best_ch = ch;
                    }
                }
            }
        }
        if (best_tl) {
            gf2c_use_fused = true;
            gf2c_taylor_lv = best_tl;
            gf2c_bfly_lv = best_bl;
            gf2c_fuse_max_ch = best_ch;
            printf("best: fused tlv=%d blv=%d maxch=%" PRIu64
                   " -- fwd+inv %.1f ms (%.2fx vs legacy)\n",
                   best_tl, best_bl, best_ch, best_t * 1e3,
                   t_legacy / best_t);
            printf("  production: export GF2C_FFT=fused GF2C_TAYLOR_LV=%d "
                   "GF2C_BFLY_LV=%d GF2C_FUSE_MAX_CH=%" PRIu64 "\n",
                   best_tl, best_bl, best_ch);
        } else {
            gf2c_use_fused = false;
            printf("best: legacy per-level -- fwd+inv %.1f ms\n",
                   t_legacy * 1e3);
            printf("  production: export GF2C_FFT=legacy\n");
        }
        double t_mul2 = 1.5 * best_t + 0.003;       // 3 transforms + misc
        double bm = 8, bc = 1e100;
        for (int mm2 = 8; mm2 <= 96; mm2 += 2) {
            double c = mm2 * t_sq + (t_mul2 + t_hor * mm2 / 24.0) / mm2;
            if (c < bc) { bc = c; bm = mm2; }
        }
        printf("projected with tuned FFT: modmul ~%.0f ms, m=%d at "
               "~%.1f ms/degree\n", t_mul2 * 1e3, (int)bm, bc * 1e3);
        (void)sv_f; (void)sv_t; (void)sv_b; (void)sv_c;
    }
        printf("recommended m: %d (analytic %d); expect ~%.1f ms/degree\n",
           best_m, mo, best_c * 1e3);
    G.fini();
    
return 0;
}

static int auto_m(TsGPU &G) {
    std::vector<u64> a = st_res(G.r);
    CUCHK(cudaMemcpy(G.dA, a.data(), G.nw * 8, cudaMemcpyHostToDevice));
    G.square_slot(G.dA, G.dTmp);
    CUCHK(cudaDeviceSynchronize());
    double t0 = trin_now_s();
    for (int i = 0; i < 6; i++) G.square_slot(G.dA, G.dTmp);
    CUCHK(cudaDeviceSynchronize());
    double t_sq = (trin_now_s() - t0) / 6;
    G.mul_A_by_C();
    CUCHK(cudaDeviceSynchronize());
    t0 = trin_now_s();
    for (int i = 0; i < 2; i++) G.mul_A_by_C();
    CUCHK(cudaDeviceSynchronize());
    double t_mul = (trin_now_s() - t0) / 2;
    int mo = (int)(sqrt(t_mul / t_sq) + 0.5) & ~1;
    if (mo < 8) mo = 8;
    if (mo > 64) mo = 64;
    fprintf(stderr, "auto-m: t_sq %.3f ms, t_mul %.1f ms -> m = %d\n",
            t_sq * 1e3, t_mul * 1e3, mo);
    return mo;
}

// ---------------------------------------------------------------------------
// main

static void usage() {
    fprintf(stderr,
        "Usage: tsfactor <r> <survivors.txt> --skip D [options]\n"
        "       tsfactor --selftest [--gcd-threads N]\n"
        "       tsfactor --bench <r> [s]\n"
        "Options:\n"
        "  --skip D        REQUIRED: inputs have no factor of degree <= D\n"
        "  --maxd D        do not scan degrees > D (emit \"s u\")\n"
        "  --m M           block size 2..64 (default: auto-benchmark)\n"
        "  --q0 N          blocks in the first interval (default 600/m)\n"
        "  -z Q            emit \"s u\" when q would exceed Q (factor -z)\n"
        "  -Z Q            cap q at Q but keep scanning\n"
        "  --out PREFIX    results -> PREFIX.results.txt (default ts)\n"
        "  --no-ckpt       disable checkpointing\n"
        "  --ckpt-mins N   periodic checkpoint cadence (default 15)\n"
        "  --gcd-threads N CPU GCD pool size\n"
        "  --device N      CUDA device index\n"
        "  --canzass-max D fineDDF threshold (default 2*r^(2/3))\n"
        "  --legacy-sq     use the 3-kernel squaring path\n"
        "  --pend-max N    max speculative intervals in flight (default 4)\n"
        "  --gpu-gcd       gcd engine with its big multiplications on the\n"
        "                  GPU: CPU load per gcd drops sharply (opt-in)\n"
        "  --gpu-gcd-min-bits N   offload mults with an operand >= N bits\n"
        "  --gcd-finish-bits N    gcd engine hands over to NTL below N bits\n"
        "                  (default 4194304)\n"
        "  --gcd-full-apply       gcd engine: apply each half-gcd matrix to\n"
        "                  the full operands, as before round EC (A/B\n"
        "                  timing; the default multiplies the low half only)\n"
        "  --bench-gcd     with --bench: also time one full-size gcd with\n"
        "                  NTL and with the certified engine (CPU mults, or\n"
        "                  GPU mults under --gpu-gcd)\n"
        "  --sched opt     Euler-optimal interval schedule from the\n"
        "                  empirical survival law S(k)=(16/9)/k: fewer,\n"
        "                  later GCDs (E[gcds]/survivor ~1.66 -> ~1.26;\n"
        "                  deep runs ~4-9x fewer); default: linear\n"
        "  --sched-G N     gcd cost / scan cost in degrees (default 4000\n"
        "                  ~ 63 s / 15.8 ms); tune from --bench numbers\n"
        "  --sched-rho R   geometric floor for the opt schedule's deep\n"
        "                  tail (default 1.25)\n"
        "  --hsnap-mb N    RAM for h-table snapshots (default 2048).  A\n"
        "                  snapshot lets a deep hit be localized without\n"
        "                  rebuilding the h-table (hours at r=136M); one\n"
        "                  is kept per interval with an outstanding gcd.\n"
        "                  0 disables them (restores the old rebuild)\n"
        "  --selftest-reps N   repeat --selftest N times (races in the\n"
        "                  GPU/pool interaction are intermittent; a\n"
        "                  single clean run proves less than a loop)\n"
        "  -v              verbose (repeat for more)\n"
        "Error checking (ON by default; detects hardware faults -- VRAM or\n"
        "RAM bit flips, marginal overclocks, bad transfers -- and restarts\n"
        "from the newest verified state; ~1%% overhead):\n"
        "  --no-ec         turn it off (testing / A-B timing only)\n"
        "  --ecw N         check the h-table chain every N blocks (default\n"
        "                  8; it is always checked at the end of an\n"
        "                  interval and before a checkpoint)\n"
        "  The gcd is CERTIFIED by default: every step of the half-gcd\n"
        "  reduction is verified by folds and the finish by a Bezout\n"
        "  identity, which PROVES \"no factor in this interval\" instead of\n"
        "  trusting a minute of unchecked CPU/RAM work.  The certified\n"
        "  engine is also the faster one (measure: make gcdbench).\n"
        "  --gcd-mt N      use N CPU threads inside each gcd (the products\n"
        "                  of a matrix step run side by side; default 1).\n"
        "                  Worth having when cores are idle during the gcd,\n"
        "                  as under --pend-max 1.  Total gcd threads =\n"
        "                  --gcd-threads x --gcd-mt.  CPU products only\n"
        "  --gcd-ntl       plain NTL gcd instead: NOT checked (A/B timing)\n"
        "  --ecg-xgcd      certify with one verified NTL extended gcd\n"
        "                  instead of the chain (~1.4x NTL's gcd; reference)\n"
        "  --ecg-dup       compute every gcd twice with NTL and compare,\n"
        "                  instead of certifying.  Free in wall time with\n"
        "                  --gcd-threads >= 2, else doubles the gcd time\n"
        "  --ecg           accepted for symmetry with factor_ec; certifying\n"
        "                  is already the default (with --ecg-dup: both)\n"
        "  --ect           print an error-checking cost breakdown at exit\n"
        "                  (adds device synchronisation: for measurement)\n"
        "  --eci0          skip the verification of the h-table\n"
        "                  initialisation (one modmul + k+m squarings per\n"
        "                  s: a fraction of a second)\n"
        "  --ec-retries N  give up on an s after N failed restarts from\n"
        "                  the same point (default 3; exit status 3)\n"
        "  --ec-fault SITE:N   TEST: flip one bit at the N-th opportunity\n"
        "                  of SITE (h hmid a c t red idle init xferh xfera\n"
        "                  snap hold job gcd) to watch the checker work\n");
    exit(1);
}

int main(int argc, char **argv) {
    crc32_init();
    u64 r = 0, s_bench = 0;
    std::string surv_path, out_prefix = "ts";
    SParams P;
    P.skip = (u64)-1;
    int gcd_threads = 0, device = -1, want_m = 0;
    bool do_selftest = false, do_bench = false, legacy = false,
         no_ckpt = false, allow_no_ntl = false, gpu_gcd = false,
         bench_gcd = false;
    u64 gcd_min_bits_opt = 0;
    int selftest_reps = 1;
    bool ecg_explicit = false, ecg_xgcd = false, gcd_ntl = false;
    int gcd_mt = 1;

    // Extra options from the environment (TSFACTOR_ARGS="--ecg,--ect";
    // commas or spaces separate): lets a wrapper script's configuration add
    // options without the script itself changing.  They are parsed after
    // the command line.
    std::vector<std::string> argstore(argv, argv + argc);
    if (const char *ex = getenv("TSFACTOR_ARGS")) {
        std::string w;
        for (const char *c = ex;; c++) {
            if (*c && *c != ',' && !isspace((unsigned char)*c)) {
                w.push_back(*c);
                continue;
            }
            if (!w.empty()) { argstore.push_back(w); w.clear(); }
            if (!*c) break;
        }
        if ((int)argstore.size() > argc)
            fprintf(stderr, "TSFACTOR_ARGS adds: %s\n", ex);
    }
    std::vector<char *> argvx;
    for (auto &t : argstore) argvx.push_back(&t[0]);
    argc = (int)argvx.size();
    argv = argvx.data();

    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *o) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", o);
                                 exit(1); }
            return argv[++i];
        };
        if (a == "--selftest") do_selftest = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--skip") P.skip = strtoull(need("--skip"), 0, 10);
        else if (a == "--maxd") P.maxd = strtoull(need("--maxd"), 0, 10);
        else if (a == "--m") want_m = atoi(need("--m"));
        else if (a == "--q0") P.q0 = atol(need("--q0"));
        else if (a == "-z") P.zq = atol(need("-z"));
        else if (a == "-Z") P.Zq = atol(need("-Z"));
        else if (a == "--out") out_prefix = need("--out");
        else if (a == "--no-ckpt") no_ckpt = true;
        else if (a == "--ckpt-mins")
            P.ckpt_secs = atof(need("--ckpt-mins")) * 60.0;
        else if (a == "--gcd-threads") gcd_threads = atoi(need("--gcd-threads"));
        else if (a == "--device") device = atoi(need("--device"));
        else if (a == "--canzass-max")
            P.canzass_max = strtoull(need("--canzass-max"), 0, 10);
        else if (a == "--legacy-sq") legacy = true;
        else if (a == "--allow-no-ntl") allow_no_ntl = true;
        else if (a == "--gpu-gcd") gpu_gcd = true;
        else if (a == "--selftest-reps")
            selftest_reps = atoi(need("--selftest-reps"));
        else if (a == "--sched") {
            std::string v = need("--sched");
            if (v == "opt") P.sched_opt = true;
            else if (v == "linear") P.sched_opt = false;
            else { fprintf(stderr, "bad --sched %s (opt|linear)\n",
                           v.c_str()); exit(1); }
        }
        else if (a == "--sched-G")
            P.sched_G = strtoull(need("--sched-G"), 0, 10);
        else if (a == "--sched-rho")
            P.sched_rho = atof(need("--sched-rho"));
        else if (a == "--gpu-gcd-min-bits")
            gcd_min_bits_opt = strtoull(need("--gpu-gcd-min-bits"), 0, 10);
        else if (a == "--bench-gcd") bench_gcd = true;
#ifdef HAVE_NTL
        else if (a == "--gcd-finish-bits")     // tuning/testing knob
            hyb_ntl_finish_bits = strtoull(need("--gcd-finish-bits"), 0, 10);
        else if (a == "--gcd-full-apply")      // A/B: the pre-EC engine
            hyb_split_apply = false;
#endif
        else if (a == "--pend-max") P.pend_max = atoi(need("--pend-max"));
        else if (a == "--hsnap-mb") P.hsnap_mb = atol(need("--hsnap-mb"));
        else if (a == "--die-after-blocks")
            P.die_after_blocks = atol(need("--die-after-blocks"));
        else if (a == "--no-ec") g_ec.on = false;
        else if (a == "--ecw") {
            g_ec.window = atol(need("--ecw"));
            if (g_ec.window < 1) { fprintf(stderr, "--ecw must be >= 1\n");
                                   exit(1); }
        }
        else if (a == "--ecg") ecg_explicit = true;
        else if (a == "--ecg-xgcd") { ecg_explicit = true; ecg_xgcd = true; }
        else if (a == "--ecg-dup") g_ec.gcd_dup = true;
        else if (a == "--gcd-ntl") gcd_ntl = true;
        else if (a == "--gcd-mt") gcd_mt = atoi(need("--gcd-mt"));
        else if (a == "--ect") g_ec.timing = true;
        else if (a == "--eci0") g_ec.init_check = false;
        else if (a == "--ec-retries") {
            g_ec.max_retry = atoi(need("--ec-retries"));
            if (g_ec.max_retry < 0) g_ec.max_retry = 0;
        }
        else if (a == "--ec-fault") {
            std::string v = need("--ec-fault");
            size_t c = v.find(':');
            std::string nm = v.substr(0, c);
            long n = (c == std::string::npos) ? 1 : atol(v.c_str() + c + 1);
            int site = ECI_NONE;
            for (int i = 1; i < ECI_N; i++)
                if (nm == eci_name[i]) site = i;
            if (site == ECI_NONE || n == 0) {
                fprintf(stderr, "bad --ec-fault %s (SITE:N; sites:",
                        v.c_str());
                for (int i = 1; i < ECI_N; i++)
                    fprintf(stderr, " %s", eci_name[i]);
                fprintf(stderr, ")\n");
                exit(1);
            }
            g_eci.arm(site, n, 0x5EEDULL);
        }
        else if (a == "-v") P.verbose++;
        else if (a == "-vv") P.verbose += 2;
        else if (a.size() && a[0] == '-' && !isdigit(a[1])) {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
        }
        else pos.push_back(a);
    }
    if (gcd_threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        gcd_threads = hc > 2 ? (int)std::min(3u, hc - 1) : 1;
    }

#ifdef HAVE_NTL
    if (gcd_mt > 1 && !hyb_threads_ok()) {
        fprintf(stderr, "WARNING: --gcd-mt ignored: this NTL was built "
                        "without NTL_THREADS\n");
        gcd_mt = 1;
    }
    hyb_threads = gcd_mt < 1 ? 1 : gcd_mt > 64 ? 64 : gcd_mt;
#else
    (void)gcd_mt;
#endif
    if (do_selftest) {
        if (device >= 0) cudaSetDevice(device);
        int rc = 0;
        for (int rep = 0; rep < selftest_reps; rep++) {
            if (selftest_reps > 1)
                printf("===== selftest run %d of %d =====\n", rep + 1,
                       selftest_reps);
            int one = selftest(gcd_threads);
            if (one) rc = one;
        }
        if (selftest_reps > 1)
            printf("===== %d runs complete: %s =====\n", selftest_reps,
                   rc ? "FAILURES SEEN" : "all runs PASS");
        return rc;
    }
    if (do_bench) {
        if (pos.empty()) usage();
        r = strtoull(pos[0].c_str(), 0, 10);
        if (pos.size() > 1) s_bench = strtoull(pos[1].c_str(), 0, 10);
        if (device >= 0) cudaSetDevice(device);
        int brc = bench(r, s_bench);
#ifdef HAVE_NTL
        if (!brc && bench_gcd) {
            u64 sb = s_bench ? s_bench : (r >= 5 ? (r - 1) / 2 : 1);
            fprintf(stderr, "GCD backend: %s\n", gcd_backend());
            size_t nwt = bits_to_words(r + 1);
            std::vector<u64> T(nwt, 0), A(bits_to_words(r), 0);
            T[0] |= 1;
            T[sb >> 6] |= 1ULL << (sb & 63);
            T[r >> 6] |= 1ULL << (r & 63);
            u64 st = 0x1234abcd;
            for (auto &w : A) {
                st += 0x9E3779B97F4A7C15ULL;
                u64 z = st;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                w = z ^ (z >> 31);
            }
            A[(r - 1) >> 6] &= (((r - 1) & 63) == 63)
                ? ~0ULL : ((1ULL << (((r - 1) & 63) + 1)) - 1);
            CantorBasis bb;
            if (gpu_gcd) {
                bb.build();
                size_t ca = (size_t)((r + 31) >> 5);
                int fm = 1;
                while (((size_t)1 << fm) < 2 * ca - 1) fm++;
                g_ts_gm.init(&bb, fm);
                g_gm = &g_ts_gm;
                g_use_hybrid = true;
                if (gcd_min_bits_opt) hyb_gpu_min_bits = gcd_min_bits_opt;
                hyb_reset_stats();
            }
            g_keep_bits = r + 1;
            // the reference: NTL's own gcd (what --gcd-ntl / --no-ec run)
            double t0 = trin_now_s();
            std::vector<u64> gg;
            u64 gd = poly_gcd_ntl(A.data(), A.size(), T.data(), T.size(), &gg,
                                  g_keep_bits);
            double dt = trin_now_s() - t0;
            printf("gcd bench (random A vs T, NTL): %.2f s (gdeg %" PRIu64
                   ")\n", dt, gd);
            fflush(stdout);
            if (g_use_hybrid) {          // the engine, uncertified (--no-ec)
                double th = trin_now_s();
                std::vector<u64> gh;
                u64 gdh = ts_gcd(A.data(), A.size(), T.data(), T.size(), &gh);
                double dth = trin_now_s() - th;
                u64 ng, nc, nf;
                hyb_get_stats(ng, nc, nf);
                printf("gcd bench (random A vs T, hybrid): %.2f s (gdeg %"
                       PRIu64 ", %+.1f%% vs NTL)\n", dth, gdh,
                       100.0 * (dth - dt) / dt);
                printf("  hybrid mults: gpu %" PRIu64 " cpu %" PRIu64
                       " fallbacks %" PRIu64 "\n", ng, nc, nf);
                if (gdh != gd) printf("  *** MISMATCH vs NTL ***\n");
                fflush(stdout);
            }
            if (g_ec.on && !gcd_ntl) {
                // the same gcd, certified (what a production run computes)
                hyb_cert_xgcd = ecg_xgcd;
                EcFold fa, ft;
                ecf_fold_host(fa, A.data(), A.size());
                ecf_fold_host(ft, T.data(), T.size());
                std::vector<u64> g2;
                int st = -1;
                hyb_reset_stats();
                double t1 = trin_now_s();
                u64 gd2 = ts_gcd_cert(A.data(), A.size(), T.data(), T.size(),
                                      &g2, &fa, &ft, &st);
                double dt2 = trin_now_s() - t1;
                u64 ng, nc, nf;
                hyb_get_stats(ng, nc, nf);
                printf("gcd bench, CERTIFIED (%s): %.2f s (gdeg %"
                       PRIu64 ", status %s, %+.1f%% vs NTL)\n",
                       ecg_xgcd ? "NTL XGCD" : g_gm ? "chain, GPU mults"
                                                    : "chain, CPU mults",
                       dt2, gd2,
                       st == 0 ? "certified" : st == 3 ? "NOT certified"
                       : "CERTIFICATE FAILED",
                       100.0 * (dt2 - dt) / dt);
                printf("  mults: gpu %" PRIu64 " cpu %" PRIu64 " fallbacks %"
                       PRIu64 "\n", ng, nc, nf);
                if (hyb_threads > 1 && !g_gm && !ecg_xgcd)
                    printf("  (with --gcd-mt %d: %d threads inside the gcd)\n",
                           hyb_threads, hyb_threads);
                if (gd2 != gd) printf("  *** MISMATCH vs NTL ***\n");
            }
            if (g_use_hybrid) g_ts_gm.fini();
        }
#endif
        return brc;
    }
    if (pos.size() != 2) usage();
    r = strtoull(pos[0].c_str(), 0, 10);
    surv_path = pos[1];
    if (!(r & 1) || r < 5) { fprintf(stderr, "r must be odd, >= 5\n");
                             exit(1); }
    if (P.skip == (u64)-1) {
        fprintf(stderr, "--skip is REQUIRED (36 or 37 for the r=136279841 "
                        "pipeline);\nscanning starts at skip+1 and factors "
                        "of degree <= skip would be MISSED.\n");
        exit(1);
    }
    if (P.zq && P.Zq) { fprintf(stderr, "-z and -Z are mutually "
                                        "exclusive\n"); exit(1); }
    if (!P.canzass_max) {
        u64 r3 = 0;
        while (r3 * r3 * r3 < r) r3++;
        P.canzass_max = 2 * r3 * r3;
    }
#ifdef HAVE_NTL
    if (gpu_gcd) g_use_hybrid = true;    // reflected in the banner
    if ((gcd_ntl && (ecg_explicit || gpu_gcd)) || (ecg_xgcd && gpu_gcd)) {
        fprintf(stderr, "ERROR: --gcd-ntl contradicts --ecg / --ecg-xgcd / "
                        "--gpu-gcd, and --ecg-xgcd contradicts --gpu-gcd\n");
        exit(1);
    }
    // With checking on, the gcd is CERTIFIED unless the plain NTL gcd is
    // asked for (--gcd-ntl; --ecg-dup alone means "NTL, twice").  The
    // certified chain is also the faster engine on CPU multiplications.
    if (!g_ec.on) g_ec.gcd_cert = g_ec.gcd_dup = false;
    else g_ec.gcd_cert = ecg_explicit || gpu_gcd ||
                         !(gcd_ntl || g_ec.gcd_dup);
    hyb_cert_xgcd = ecg_xgcd;
#else
    g_ec.gcd_cert = false;
    (void)ecg_explicit; (void)ecg_xgcd; (void)gcd_ntl;
#endif
#ifndef HAVE_NTL
    if (gpu_gcd) {
        fprintf(stderr, "ERROR: --gpu-gcd needs an NTL build (the hybrid "
                        "engine uses NTL for CPU-side mults and the "
                        "finishing gcd)\n");
        exit(1);
    }
#endif
    fprintf(stderr, "GCD backend: %s\n", gcd_backend());
#ifdef HAVE_NTL
    if (hyb_threads > 1)
        fprintf(stderr, "gcd: %d threads inside each gcd (--gcd-mt)%s\n",
                hyb_threads, (g_ec.on && g_ec.gcd_cert && !hyb_cert_xgcd &&
                              !gpu_gcd)
                    ? "" : " -- NOTE: only the certified chain on CPU "
                           "products uses them");
#endif
    g_ec.log_path = out_prefix + ".ec.log";
    if (g_ec.on) {
        fprintf(stderr, "error checking: ON -- h-table chain every %ld "
                "block(s), product/reduction/Horner folds every block, "
                "init %s, transfers and snapshots verified, reported "
                "factors verified; gcd %s\n", g_ec.window,
                g_ec.init_check ? "verified" : "NOT verified (--eci0)",
                g_ec.gcd_cert ? (g_ec.gcd_dup ? "certified and computed "
                                                "twice" : "certified")
                              : g_ec.gcd_dup ? "computed twice (--ecg-dup)"
                                             : "NOT checked");
        if (!g_ec.gcd_cert && !g_ec.gcd_dup)
            fprintf(stderr, "NOTE: the gcd itself (CPU, ordinary RAM) is not "
                    "checked in this configuration (--gcd-ntl, or a build "
                    "without NTL).  A fault there could still turn a hit "
                    "into a miss.\n");
        if (g_ec.gcd_dup && gcd_threads < 2)
            fprintf(stderr, "hint: --ecg-dup with --gcd-threads 1 runs the "
                    "two gcds one after the other (double the gcd time); "
                    "--gcd-threads 2 runs them side by side\n");
        fprintf(stderr, "error checking: detections are logged to %s\n",
                g_ec.log_path.c_str());
    } else {
        fprintf(stderr, "error checking: OFF (--no-ec) -- a hardware fault "
                "will silently produce a wrong result\n");
    }
    if (g_eci.site != ECI_NONE)
        fprintf(stderr, "TEST MODE: will flip one bit (%s) at opportunity "
                "%ld\n", eci_desc[g_eci.site], g_eci.at);
    if (gcd_threads == 1 && P.pend_max == 1)
        fprintf(stderr, "hint: with --gcd-threads 1, --pend-max 2 lets the "
                "scan run one interval\nahead of the single GCD worker at "
                "no extra CPU cost\n");
#ifndef HAVE_NTL
    if (!allow_no_ntl) {
        fprintf(stderr,
            "ERROR: refusing to scan survivors without NTL.  Every hit\n"
            "whose interval gcd holds more than one factor would come out\n"
            "as an unresolved range.  Ensure ntl_check.cpp sits next to\n"
            "the Makefile and libntl-dev (+libgmp-dev) is installed, then\n"
            "rebuild -- `make tsfactor` prints the NTL detection result\n"
            "and refuses silent no-NTL builds.  Override (degrees-only\n"
            "experiments): --allow-no-ntl.\n");
        exit(1);
    }
#else
    (void)allow_no_ntl;
#endif
    if (device >= 0) cudaSetDevice(device);

    // survivors
    std::vector<u64> svs;
    {
        FILE *f = fopen(surv_path.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", surv_path.c_str());
                  exit(1); }
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            char *p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '#' || *p == '\n' || !*p) continue;
            u64 sv = strtoull(p, 0, 10);
            if (!(sv >= 1 && 2 * sv < r)) {
                fprintf(stderr, "bad survivor s=%" PRIu64
                        " (need 1 <= s <= (r-1)/2)\n", sv);
                exit(1);
            }
            svs.push_back(sv);
        }
        fclose(f);
    }
    fprintf(stderr, "%zu survivors from %s\n", svs.size(), surv_path.c_str());

    // resume: completed s from the results file
    std::string res_path = out_prefix + ".results.txt";
    std::set<u64> done_s;
    {
        std::ifstream in(res_path);
        if (in) {
            u64 sv;
            for (;;) {
                if (in >> sv) {
                    if (sv) done_s.insert(sv);
                } else if (in.eof() || in.bad()) {
                    break;                 // nothing left, or an I/O error
                } else {
                    in.clear();            // line didn't start with a number
                }
                in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
            fprintf(stderr, "%zu already completed in %s\n", done_s.size(),
                    res_path.c_str());
        }
    }
    if (!no_ckpt) P.ckpt_base = out_prefix + ".ckpt";

    // checkpoint (for the in-progress s, if any)
    ResumeState RS;
    if (!no_ckpt) {
        RS.valid = load_ckpt(P.ckpt_base, RS.H, RS.h, RS.a);
        if (RS.valid && (RS.H.r != r || RS.H.skip != P.skip ||
                         done_s.count(RS.H.s))) {
            fprintf(stderr, "checkpoint ignored (r/skip mismatch or s "
                            "already done)\n");
            RS.valid = false;
        }
    }

    TsGPU G;
    G.init(r);
    G.legacy_sq = legacy;
#ifdef HAVE_NTL
    if (gpu_gcd) {
        g_ts_gm.init(&G.Bb, G.fftm);
        g_gm = &g_ts_gm;
        if (gcd_min_bits_opt) hyb_gpu_min_bits = gcd_min_bits_opt;
        fprintf(stderr, "GPU-assisted GCD: on (GPU mults >= %" PRIu64
                " bits; NTL finish <= %" PRIu64 " bits)\n",
                hyb_gpu_min_bits, hyb_ntl_finish_bits);
        hyb_reset_stats();
    }
#endif
    int m = want_m;
    if (RS.valid) {
        if (m && m != (int)RS.H.m)
            fprintf(stderr, "note: --m %d overridden by checkpoint m=%"
                    PRIu64 "\n", m, RS.H.m);
        m = (int)RS.H.m;
    }
    if (!m) {
        G.set_m(24);            // provisional (auto_m uses dC via horner)
        G.set_s((r - 1) / 2);
        m = auto_m(G);
    }
    G.set_m(m);
    {
        double mb = 1.0 / (1024 * 1024);
        double htab = 2.0 * G.m * G.nw * 8 * mb;
        double fft = (2.0 * G.F.n + G.F.n) * 8 * mb;   // dFA+dFB+twiddles
        double scr = (4.0 * G.nw + 2.0 * G.nw2 + G.nw) * 8 * mb;
        double ecm = g_ec.on ? 5.0 * G.nw * 8 * mb : 0;
        fprintf(stderr, "GPU memory (fixed at startup, independent of "
                "q/depth): h-tables %.0f MB, FFT %.0f MB, scratch %.0f MB, "
                "error checking %.0f MB%s\n", htab, fft, scr, ecm,
                gpu_gcd ? ", +gcd hook ~610 MB" : "");
        double snap = (double)G.m * G.nw * 8 * mb;
        long nsn = (P.hsnap_mb <= 0) ? 0 : (long)(P.hsnap_mb / snap);
        if (P.hsnap_mb > 0 && nsn < 1) nsn = 1;
        fprintf(stderr, "host memory: h-table snapshot %.0f MB each, up to "
                "%ld kept (--hsnap-mb %ld)%s\n", snap, nsn, P.hsnap_mb,
                nsn >= P.pend_max + 1 ? "" :
                "  [< pend-max+1: a deep hit may need the slow rebuild]");
    }
    if (P.q0 <= 0) P.q0 = (long)((600 + m - 1) / m);
    if (P.sched_opt)
        fprintf(stderr, "interval schedule: euler-optimal (G=%" PRIu64
                " degrees)\n", P.sched_G ? P.sched_G : 4000);
    fprintf(stderr, "r=%" PRIu64 " skip=%" PRIu64 " m=%d q0=%ld "
            "gcd-threads=%d pend-max=%d%s%s\n",
            r, P.skip, m, P.q0, gcd_threads, P.pend_max,
            P.zq ? " -z" : "", P.Zq ? " -Z" : "");

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    FILE *rf = fopen(res_path.c_str(), "a");
    if (!rf) { fprintf(stderr, "cannot append %s\n", res_path.c_str());
               exit(1); }

    GcdPool pool;
    pool.start(gcd_threads);
    g_keep_bits = r + 1;
    u64 gseq = 0, n_found = 0, n_u = 0;

    // process the checkpointed s first (if it is still pending)
    if (RS.valid) {
        auto it = std::find(svs.begin(), svs.end(), RS.H.s);
        if (it == svs.end() || done_s.count(RS.H.s)) RS.valid = false;
        else std::rotate(svs.begin(), it, it + 1);
    }

    bool interrupted = false, ec_fatal = false;
    u64 ec_fatal_s = 0;
    for (u64 sv : svs) {
        if (done_s.count(sv)) continue;
        if (g_stop) { interrupted = true; break; }
        if (P.verbose) printf("== s=%" PRIu64 " ==\n", sv);
        ScanOut o = scan_one_s(G, pool, P, r, sv, gseq,
                               RS.valid && RS.H.s == sv ? &RS : nullptr);
        if (o.kind == ScanOut::INTERRUPTED) { interrupted = true; break; }
        if (o.kind == ScanOut::EC_FATAL) { ec_fatal = true; ec_fatal_s = sv;
                                           break; }
        char buf[256];
        switch (o.kind) {
        case ScanOut::FOUND:
            snprintf(buf, sizeof(buf), "%" PRIu64 " %" PRIu64 " p", sv, o.d);
            fprintf(rf, "%s%s\n", buf, o.hex.c_str());
            printf("%s%s\n", buf, o.hex.c_str());
            n_found++;
            break;
        case ScanOut::GAVE_UP:
            fprintf(rf, "%" PRIu64 " u\n", sv);
            printf("%" PRIu64 " u\n", sv);
            n_u++;
            break;
        case ScanOut::PRIMITIVE:
            fprintf(rf, "%" PRIu64 " primitive\n", sv);
            printf("%" PRIu64 " primitive\n", sv);
            fprintf(stderr, "\n*** PRIMITIVE TRINOMIAL CANDIDATE: s=%"
                    PRIu64 " -- verify independently! ***\n\n", sv);
            break;
        case ScanOut::RANGE:
            fprintf(rf, "%" PRIu64 " r%" PRIu64 "-%" PRIu64 "\n", sv,
                    o.rlo, o.rhi);
            printf("%" PRIu64 " r%" PRIu64 "-%" PRIu64 "\n", sv, o.rlo,
                   o.rhi);
            break;
        default: break;
        }
        fflush(rf);
        fsync(fileno(rf));
        fflush(stdout);
        if (P.verbose)
            printf("== s=%" PRIu64 " done in %.1f s (%" PRIu64
                   " degrees) ==\n", sv, o.wall, o.degrees_scanned);
    }

    fclose(rf);
    if (g_ec.on) {
        u64 nd = g_ecs.total();
        fprintf(stderr, "error checking: %" PRIu64 " blocks and %" PRIu64
                " h-table chain windows verified; %" PRIu64 " detection(s)"
                ", %" PRIu64 " restart(s), %" PRIu64 " in-place repair(s), %"
                PRIu64 " transfer re-read(s), %" PRIu64 " gcd re-run(s)\n",
                g_ecs.blocks.load(), g_ecs.windows.load(), nd,
                g_ecs.restarts.load(), g_ecs.repairs.load(),
                g_ecs.xfer_retries.load(), g_ecs.gcd_reruns.load());
        if (g_ec.gcd_cert)
            fprintf(stderr, "error checking: %" PRIu64 " gcd(s) certified, %"
                    PRIu64 " not certified\n",
                    g_ecs.gcd_cert.load(), g_ecs.gcd_uncert.load());
        for (int c = 0; c < ECC_N; c++)
            if (g_ecs.fails[c].load())
                fprintf(stderr, "    %-26s %" PRIu64 "\n", ecc_name[c],
                        g_ecs.fails[c].load());
        if (nd || g_ecs.xfer_retries.load())
            fprintf(stderr, "  NOTE: detections mean this machine made "
                    "errors.  Results are still correct (every one was "
                    "caught and redone), but check cooling, clocks and "
                    "power; details in %s\n", g_ec.log_path.c_str());
        if (g_ec.timing) {
            double tot = g_ecs.t_block + g_ecs.t_host + g_ecs.t_window +
                         g_ecs.t_init + g_ecs.t_xfer + g_ecs.t_verify;
            u64 nb = g_ecs.blocks.load(), nwn = g_ecs.windows.load();
            fprintf(stderr, "error-checking cost breakdown (--ect)\n"
                    "  %-34s %10s %11s %10s\n", "category", "count",
                    "seconds", "ms each");
            auto row = [](const char *nm, u64 n, double t) {
                fprintf(stderr, "  %-34s %10" PRIu64 " %11.3f %10.3f\n", nm,
                        n, t, n ? 1e3 * t / (double)n : 0.0);
            };
            row("per block: device folds + readback", nb, g_ecs.t_block);
            row("per block: host fold algebra", nb, g_ecs.t_host);
            row("h-table chain check", nwn, g_ecs.t_window);
            row("h-table init verification", g_ecs.n_init, g_ecs.t_init);
            row("verified transfers", g_ecs.n_xfer, g_ecs.t_xfer);
            row("factor verification", g_ecs.n_verify, g_ecs.t_verify);
            fprintf(stderr, "  %-34s %10s %11.3f\n", "TOTAL", "", tot);
        }
    }
    if (ec_fatal) {
        fprintf(stderr,
            "\n*** ERROR CHECKING: PERSISTENT FAULT -- STOPPING ***\n"
            "s=%" PRIu64 " failed its checks %d times in a row from the "
            "same verified state.\nThat is not a cosmic ray; the GPU, its "
            "driver, RAM or power is unreliable right now.\nNo result was "
            "written for this s (it will be redone on the next run).\n"
            "Check temperatures and any overclock/undervolt, run "
            "`tsfactor --selftest`,\nand see %s.\n", ec_fatal_s,
            g_ec.max_retry + 1, g_ec.log_path.c_str());
        fflush(nullptr);
        pool.stop_nowait();
        std::_Exit(3);
    }
    if (interrupted) {
        fprintf(stderr, "interrupted: %" PRIu64 " found, %" PRIu64
                " u; exiting (in-flight GCDs abandoned)\n", n_found, n_u);
        fflush(nullptr);
        pool.stop_nowait();
        std::_Exit(0);             // workers may hold NTL state; skip
    }                              // static teardown entirely
    pool.stop();          // joins workers: nothing can be in the hook now
#ifdef HAVE_NTL
    g_ts_gm.fini();
#endif
    G.fini();
    fprintf(stderr, "done: %" PRIu64 " found, %" PRIu64 " u\n", n_found,
            n_u);
    return 0;
}

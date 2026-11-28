/*
Copyright (C) 2026  Gord Palameta
(actual author: Claude AI)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Some code may have been inspired by:
  GF2X library
  factor.cpp and other files in GF2X library apps subdirectory
  NTL library
*/

// ts_core.h
// The part of tsfactor that does not care where the arithmetic runs:
// the scan driver (intervals, speculation, verdicts, fine DDF), the gcd
// pool, error-checking bookkeeping and recovery, checkpoints, the interval
// schedule, the selftest suites that drive whole scans, and the command
// line.  It talks to the arithmetic through one interface, TsDev:
//
//     TsGPU (tsfactor.cu)  CUDA kernels; the production GPU program
//     TsCPU (ts_cpu.h)     the same operations on the CPU: BLZ squaring for
//                          both parities of s, gf2x products, word folds
//
// Both programs are built from this header:
//     tsfactor       nvcc tsfactor.cu       (GPU; also contains TsCPU, which
//                                            its selftest checks against TsGPU)
//     tsfactor_cpu   g++ tsfactor_cpu.cpp   (no CUDA anywhere)
// and a future GPU irred can reuse TsDev's squaring and the checkpoint and
// error-checking machinery here.  Round 10 split this out of tsfactor.cu;
// see tsfactor_handoff.md.
//
// Include order: the GPU build includes gf2_trinomial_cuda.h and gf2_ec.h
// (with its kernels) first; the CPU build defines GF2_EC_HOST_ONLY.

#ifndef TS_CORE_H
#define TS_CORE_H

#include "gf2_trin_host.h"
#include "gf2_gcd.h"
#include "gf2_ec.h"
#include <cerrno>
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
#include <cstddef>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/file.h>
#include <sys/stat.h>

// Version.  The fingerprint of a tuning record starts with this, so a
// checkpoint tuned by another round is re-tuned when resumed.
#define TS_ROUND 10
#define TS_ROUND_STR "round 11"

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
    bool gcd_cert = false;     // the gcd is CERTIFIED (gf2_gcd.h).  ts_main()
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
// TsDev: what the scan driver needs from the arithmetic.
//
// State that lives with the arithmetic (VRAM for TsGPU, RAM for TsCPU):
//   the h-table (m residues mod T, ping-ponged by the squaring), the
//   accumulator A, the block polynomial C, scratch, the error-checking
//   window state (D, S_start), and on demand a "hold" copy of the h-table
//   for fine DDF.  A residue is nw = ceil(r/64) words with bits >= r zero.
// Conventions every backend must follow (the selftest compares them):
//   - T = x^r + x^s + 1 with r odd and 1 <= s, 2s < r (set_s rejects
//     anything else); no flipping to the reciprocal trinomial, ever, so
//     both backends hold IDENTICAL h-tables and accumulators.
//   - host copies of an h-table are m*nw words, slot j at offset j*nw;
//     table_fold()/hold_fold() fold exactly that layout.
//   - fault-injection points sit where TsGPU has always had them (inside
//     init_h_at and mul_A_by_C; inject_* elsewhere), drawing from
//     ec_inject_rnd in the same order.
struct TsDev {
    u64 r = 0, s = 0;
    size_t nw = 0;             // words for r bits
    int m = 0;                 // block size (h slots); 2 <= m <= 64
    bool ec_on = false;        // checks active for the current scan
    EcFold ecA;                // phi(A), carried on the host block to block
    EcFold ecC;                // phi(C) as last predicted (diagnostics)
    long ec_w = 0;             // blocks accumulated in the current window
    double t_sq = 0, t_horner = 0, t_mul = 0;   // -v -v component seconds

    TsDev() { ecf_one(ecA); ecf_zero(ecC); }
    virtual ~TsDev() {}
    virtual const char *kind() const = 0;      // "GPU" or "CPU"
    virtual void init(u64 r_) = 0;             // buffers that depend on r
    virtual void set_m(int m_) = 0;            // (re)allocate the h-table;
                                               // drops any hold table
    virtual void set_s(u64 s_) = 0;            // exits unless 2s < r
    virtual void fini() = 0;
    virtual void sync() {}                     // wait for queued device work

    // ---- the scan ----
    // h-table at block start kstart: Pascal-triangle build of the
    // elementary symmetric functions of x^(2^0) .. x^(2^(m-1)), then
    // kstart Frobenius steps (factor.cpp's init_h)
    virtual void init_h_at(u64 kstart) = 0;
    virtual void reset_A() = 0;                // A <- 1 (and ecA)
    virtual void horner_C() = 0;               // C <- x^m + sum h[j] x^(m-1-j)
    virtual void mul_A_by_C() = 0;             // A <- A*C mod T (+ EC folds)
    virtual void square_all_step() = 0;        // every slot squared once
    // Every slot squared n times.  `mid`, if given, is called once part way
    // through (the driver's mid-block fault-injection opportunity).  The
    // default steps the whole table n times; TsCPU squares one slot n times
    // before the next, so that its working set is two residues instead of
    // the whole table (30-40% less per squaring where two fit in cache).
    virtual void square_steps(u64 n, void (*mid)(TsDev &) = nullptr) {
        for (u64 t = 0; t < n; t++) {
            square_all_step();
            if (mid && t == n / 2) mid(*this);
        }
    }

    // ---- error checking ----
    virtual void ec_setup(bool on) = 0;
    // per block, before horner_C: D ^= S_z, and the fold of the unreduced
    // Horner sum, recomputed independently of horner_C
    virtual void ec_block_pre() = 0;
    // the folds the per-block check needs, as left by the last block:
    // unreduced Horner sum, the 3 segment folds of the unreduced product,
    // the reduced product, and the Horner tail's quotient mask
    virtual void ec_read_block(EcFold &fCu, EcFold seg[3], EcFold &fAn,
                               u64 &fired) = 0;
    virtual void ec_chain_reset() = 0;         // S_start <- S(table), D <- 0
    // sigma(D) + D == S_start + S_now at full width (with no block since
    // the last check: S_now == S_start).  Restarts the window.  True if clean.
    virtual bool ec_window_check() = 0;
    // (1 + y_k)(1 + S_k) == 1 + y_(k+m); leaves the chain reset at this
    // table.  True if the table is right.
    virtual bool ec_verify_table_at(u64 k) = 0;
    virtual void table_fold(EcFold &f) = 0;    // current table, folded in place
    virtual void hold_fold(EcFold &f) = 0;     // the hold table

    // ---- copies ----
    virtual void h_to_host(std::vector<u64> &out) = 0;
    virtual void h_from_host(const std::vector<u64> &in) = 0;
    virtual void A_to_host(std::vector<u64> &out) = 0;
    virtual void A_from_host_raw(const std::vector<u64> &in) = 0;
    virtual void need_hold() = 0;
    virtual void hold_save() = 0;              // hold <- current table
    virtual void hold_restore() = 0;           // current table <- hold

    // ---- fault injection (--ec-fault, selftest) ----
    virtual void inject_C(int site) = 0;
    virtual void inject_h(int site) = 0;       // a random slot of the table
    virtual void inject_hold(int site) = 0;

    // ---- measurement, for the tuning record (seconds per operation) ----
    virtual double time_square(int reps) = 0;
    virtual double time_mul(int reps) = 0;

    // ---- selftest access ----
    virtual void C_to_host(std::vector<u64> &out) = 0;
    virtual void ecD_to_host(std::vector<u64> &out) = 0;
    virtual void ecD_zero() = 0;

    // ---- generic helpers built on the above ----
    double ec_tic() {          // --ect: isolate device time (syncs only then)
        if (!g_ec.timing) return 0;
        sync();
        return trin_now_s();
    }
    void ec_toc(double t0, double &acc) {
        if (!g_ec.timing) return;
        sync();
        acc += trin_now_s() - t0;
    }
    // Upload A.  The carried fold comes from the host copy, so a bad
    // upload shows up at the next product check.
    void A_from_host(const std::vector<u64> &in) {
        A_from_host_raw(in);
        ecf_fold_host(ecA, in.data(), in.size() < nw ? in.size() : nw);
    }
    // host half of the per-block check.  Returns 0 if clean; bit 0: the
    // Horner build or the product is wrong (phi(t) != phi(A) phi(C));
    // bit 1: the reduction is wrong.  Updates the carried phi(A).
    int ec_block_verify() {
        double t0 = ec_tic();
        EcFold fCu, seg[3], fAn, fC, ft, fred, prod;
        u64 fired = 0;
        ec_read_block(fCu, seg, fAn, fired);
        double t1 = g_ec.timing ? trin_now_s() : 0;
        ecf_horner_predict(fC, fCu, fired, m, r, s);
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
    // Download the current h-table; `f` receives its fold, taken where the
    // table lives and confirmed against what arrived.  A mismatch is
    // re-read (a transfer error is transient).  False if three reads all
    // disagree.
    bool h_to_host_ec(std::vector<u64> &out, EcFold &f) {
        double t0 = ec_tic();
        table_fold(f);
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
    // Upload an h-table.  `expect` (may be null) is the fold recorded when
    // the host copy was captured.  Returns 0 if the table now in place is
    // exactly that one (and restarts the chain from it); 1 if the host copy
    // no longer matches its recorded fold; 2 if the upload cannot be
    // confirmed.
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
                table_fold(fd);
                if (ecf_eq(fd, fh)) rc = 0;
                else g_ecs.xfer_retries++;
            }
        }
        if (!rc) ec_chain_reset();
        g_ecs.n_xfer++;
        ec_toc(t0, g_ecs.t_xfer);
        return rc;
    }
    // Download A, confirmed against the fold carried from its last product
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
static bool g_engine_cpu = false;        // uncertified gcds also use the
                                         // half-gcd engine (CPU products);
                                         // ts_main sets it unless --gcd-ntl
static bool g_hyb_xcheck = false;        // selftest: verify hybrid vs NTL
static u64 ts_gcd(const u64 *a, size_t aw, const u64 *b, size_t bw,
                  std::vector<u64> *gout) {
#ifdef HAVE_NTL
    if (g_use_hybrid || g_engine_cpu) {
        u64 d = poly_gcd_hybrid(a, aw, b, bw, gout, g_keep_bits,
                                g_use_hybrid ? g_gm : nullptr);
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
               : hyb_gf2x_direct()
                   ? "CERTIFIED half-gcd chain (CPU products, gf2x called "
                     "directly; every step fold-verified) + verified XGCD "
                     "finish"
                   : "CERTIFIED half-gcd chain (CPU products through NTL; "
                     "every step fold-verified) + verified XGCD finish";
    if (g_use_hybrid)
        return g_gm ? "hybrid HGCD (GPU-offloaded mults) + NTL finish"
                    : "hybrid HGCD (CPU mults) + NTL finish";
    if (g_engine_cpu)
        return "half-gcd engine (CPU products) + NTL finish -- NOT "
               "certified";
    return "NTL gf2x-backed (HalfGCD + CanZass)";
#else
    return "naive fallback -- NO NTL: factor resolution will fail for "
           "multi-factor interval gcds; NOT fit for production";
#endif
}

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

// The tuning record (round 10).  A run times the configurable parts of its
// arithmetic when it starts -- on the GPU the FFT variants, on both the
// squaring and the product that set the block size m -- and stamps the
// result, with a fingerprint of who measured it, into every checkpoint it
// writes.  A run that resumes a checkpoint whose fingerprint matches its
// own adopts the record instead of timing again; another round of the
// program, or another GPU/CPU, tunes afresh (--retune always does).
struct TsTune {
    char fp[176];          // "tsfactor round 10 on <device> (...)"
    u32 fft_fused;         // GPU: 1 = fused register-tile transform
    u32 fft_tlv, fft_blv;  // GPU: levels fused per Taylor / butterfly pass
    u32 measured;          // 1 = timed by a tune; 0 = defaults or pinned
    u64 fft_maxch;         // GPU: long-stride groups to the legacy kernels
    u64 m;                 // block size chosen
    double t_sq, t_mul;    // seconds per squaring / per modmul, as timed
    u64 valid;             // 1 if this record has been filled in
};

// Checkpoint format version 2 (round 10).  Version 1 files (rounds 1-9)
// are not read: the magic differs and they are ignored with a note.
#define TS_CKPT_MAGIC 0x32434653544ULL   // "TSFC2"-ish
struct CkptHdr {
    u64 magic, version;
    u64 r, s, skip;
    u64 m;
    u64 k_iv;          // start degree of the interval being processed
    u64 blk;           // blocks of that interval already folded into A
    u64 q;             // block count of that interval
    u64 has_A;         // 1 iff A section present (blk > 0)
    u64 nw;
    u64 sched_opt;     // informational: schedule of the run that wrote it
    double cost_gcd;   // the run's measured seconds per gcd (0 = none yet)
    double cost_deg;   // ... and per scanned degree
    TsTune tune;       // who tuned the run that wrote this, and what
    u64 reserved[6];
    u32 crc_h, crc_a;  // section CRCs
    u32 crc_hdr;       // of all fields above
    u32 pad;
};
static_assert(sizeof(TsTune) == 232, "TsTune layout");
static_assert(sizeof(CkptHdr) == 408, "CkptHdr layout");

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

// Checkpoint names (round 10): one set of files per (r, s),
//     <prefix>_r<r>_s<s>.ckpt  (+ .ckpt.1, .ckpt.2, and .ckpt.fine)
// so that an s can be set aside -- interrupted and left out of the next
// survivor list -- without its state being overwritten by the next s.
static std::string ckpt_base_for(const std::string &prefix, u64 r, u64 s) {
    char b[64];
    snprintf(b, sizeof b, "_r%" PRIu64 "_s%" PRIu64 ".ckpt", r, s);
    return prefix + b;
}

// the checkpoint and its two rotated generations
static void remove_ckpts(const std::string &base) {
    if (base.empty()) return;
    remove(base.c_str());
    remove((base + ".1").c_str());
    remove((base + ".2").c_str());
}

static void rotate_ckpts(const std::string &base) {
    std::string c0 = base, c1 = base + ".1", c2 = base + ".2";
    remove(c2.c_str());
    rename(c1.c_str(), c2.c_str());
    rename(c0.c_str(), c1.c_str());
}

static u32 ckpt_hdr_crc(const CkptHdr &H) {
    CkptHdr T = H;
    T.crc_hdr = 0;
    T.pad = 0;
    return crc32_of(&T, offsetof(CkptHdr, crc_hdr));
}

static bool save_ckpt(const std::string &base, const CkptHdr &H,
                      const std::vector<u64> &h, const std::vector<u64> &a,
                      int verbose) {
    CkptHdr W = H;
    W.magic = TS_CKPT_MAGIC;
    W.version = 2;
    W.crc_h = h.empty() ? 0 : crc32_of(h.data(), h.size() * 8);
    W.crc_a = a.empty() ? 0 : crc32_of(a.data(), a.size() * 8);
    W.pad = 0;
    W.crc_hdr = ckpt_hdr_crc(W);
    rotate_ckpts(base);
    bool ok = write_file_atomic(base, &W, sizeof(W), h, a);
    if (verbose)
        fprintf(stderr, "  [checkpoint %s: s=%" PRIu64 " k_iv=%" PRIu64
                " blk=%" PRIu64 "/%" PRIu64 " %s]\n",
                ok ? "written" : "FAILED", H.s, H.k_iv, H.blk, H.q,
                base.c_str());
    return ok;
}

// header only (cheap; used to decide about tuning before anything runs).
// 1 = a valid version-2 header; 0 = no such file; -1 = unreadable, or an
// older format.
static int read_ckpt_hdr(const std::string &path, CkptHdr &H) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    bool ok = fread(&H, 1, sizeof(H), f) == sizeof(H);
    fclose(f);
    if (!ok) return -1;
    if (H.magic != TS_CKPT_MAGIC || H.version != 2 ||
        ckpt_hdr_crc(H) != H.crc_hdr)
        return -1;
    H.tune.fp[sizeof H.tune.fp - 1] = '\0';
    return 1;
}

static bool load_ckpt_one(const std::string &path, CkptHdr &H,
                          std::vector<u64> &h, std::vector<u64> &a) {
    if (read_ckpt_hdr(path, H) != 1) return false;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = fseek(f, (long)sizeof(H), SEEK_SET) == 0 && H.m && H.nw &&
              H.m <= 64;
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

// One process per output prefix (round 10).  Two runs sharing a prefix
// would interleave their results files and could resume the same
// checkpoint twice; the lock makes the second one stop with a message.
// flock() is released when the process ends, however it ends, so a lock
// file left behind by a crash is harmless.  Round 11: a normal exit
// deletes the file.  That is safe because a process that gets the lock
// checks that the file it locked is still the one at the path (another
// process may have deleted it between the open and the flock) and tries
// again if not.
static int g_prefix_lock_fd = -1;
static std::string g_prefix_lock_path;
static bool lock_prefix(const std::string &prefix) {
    std::string p = prefix + ".lock";
    int fd = -1;
    for (int tries = 0;; tries++) {
        fd = open(p.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0) {
            fprintf(stderr, "warning: cannot create %s (%s); running without "
                    "the one-process-per-prefix lock\n", p.c_str(),
                    strerror(errno));
            return true;
        }
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            return false;
        }
        struct stat a, b;
        if (fstat(fd, &a) == 0 && stat(p.c_str(), &b) == 0 &&
            a.st_ino == b.st_ino && a.st_dev == b.st_dev)
            break;                         // the file at the path: ours
        close(fd);                         // deleted meanwhile: again
        if (tries > 100) return false;
    }
    char buf[32];
    int n = snprintf(buf, sizeof buf, "%ld\n", (long)getpid());
    if (ftruncate(fd, 0) == 0 && write(fd, buf, (size_t)n) == n) {}
    g_prefix_lock_fd = fd;
    g_prefix_lock_path = p;
    return true;
}

// at a normal exit, while still holding the lock
static void unlock_prefix() {
    if (g_prefix_lock_fd < 0) return;
    unlink(g_prefix_lock_path.c_str());
    close(g_prefix_lock_fd);
    g_prefix_lock_fd = -1;
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
#define TS_FINE_MAGIC 0x324e4653544ULL     // "TSFN2"-ish (round 10)
struct FineHdr {
    u64 magic, version;
    u64 r, s, skip;
    u64 m, nw;
    u64 ka0, q0;        // hit interval as reported by the coarse gcd
    u64 kcur, qcur;     // current bisection window; the h below is at kcur
    u64 reserved[4];
    u32 crc_h;
    u32 crc_hdr;
};

static std::string fine_path(const std::string &base) { return base + ".fine"; }

static bool save_fine(const std::string &base, FineHdr H,
                      const std::vector<u64> &h, int verbose) {
    H.magic = TS_FINE_MAGIC;
    H.version = 2;
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

// header only; true if it is a valid version-2 standby file
static bool read_fine_hdr(const std::string &base, FineHdr &H) {
    FILE *f = fopen(fine_path(base).c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&H, 1, sizeof(H), f) == sizeof(H);
    fclose(f);
    if (ok) {
        FineHdr T = H;
        T.crc_hdr = 0;
        ok = H.magic == TS_FINE_MAGIC &&
             crc32_of(&T, sizeof(T) - sizeof(u32)) == H.crc_hdr &&
             H.m && H.m <= 64 && H.nw && H.qcur;
    }
    return ok;
}

static bool load_fine(const std::string &base, FineHdr &H,
                      std::vector<u64> &h) {
    if (!read_fine_hdr(base, H)) return false;
    FILE *f = fopen(fine_path(base).c_str(), "rb");
    if (!f) return false;
    bool ok = fseek(f, (long)sizeof(H), SEEK_SET) == 0;
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
// interval schedule
//
// linear (--sched linear): factor.cpp's -f 1, q grows by q0 per interval.
//
// opt (--sched opt, the default since round 10): the schedule that
// minimizes the expected cost of a survivor,
//     E = sum_j S(k_{j-1}) (c_s w_j + c_g),     S(k) = (16/9) / k,
// over interval boundaries k_0 < k_1 < ... (k_0 = the first degree not yet
// verified clear, w_j = k_j - k_{j-1}, c_s per scanned degree, c_g per gcd),
// solved EXACTLY up to a fine grid by dynamic programming from the far end:
//     V(k) = min_{k' > k} [ (k' - k) + G + (k / k') V(k') ],  V(K+1) = 0,
// in units of c_s, with G = c_g / c_s and K the last degree to scan
// (rhigh: the Swan bound, r/2, or --maxd).  The next interval from any
// position k is the argmin above, so the schedule depends on (k, G, K)
// only: a resumed scan needs nothing but its position, and a change of G
// mid-run (G is measured as the run goes) simply re-solves -- consistent
// at every step.
//
// Why round 10 replaced the round-6 recurrence k_{j+1} = k_j^2/k_{j-1} - G:
// the recurrence is the DP's stationarity condition, but continuing it
// from a k_1 that was optimal for a DIFFERENT G (the default before the
// first gcd, then the measured value) runs away geometrically -- Gord's
// round-5 log: q = 36, 141, 543, a degree-5816 factor found by scanning to
// 21862.  The expected cost of that trajectory is 41% above the optimum;
// the recurrence was also 5-7% off at fixed G (its first-interval argmin
// summed a cost that diverges logarithmically without the upper bound K).
// The DP has neither problem.  (With today's ~17 s gcds, G ~ 1300, the
// optimal widths grow like sqrt(k) in the deep tail -- close to linear
// growth, which is why --sched linear q0=20 comes within 1% of it there.)
struct DPSched {
    u64 k0 = 0, K = 0;
    double G = 0;
    bool ready = false;
    std::vector<u64> pts;      // grid of interval starts; back() = K+1
    std::vector<double> V;     // expected remaining cost from pts[i]
    void solve(u64 k0_, u64 K_, double G_) {
        k0 = k0_; K = K_; G = G_;
        pts.clear();
        V.clear();
        ready = true;
        if (k0 > K) { pts.push_back(K + 1); V.push_back(0); return; }
        const double ratio = 1.002;
        for (u64 p = k0; p < K + 1;) {
            pts.push_back(p);
            u64 nx = (u64)((double)p * ratio);
            if (nx < p + 1) nx = p + 1;
            p = nx;
        }
        pts.push_back(K + 1);
        const size_t N = pts.size();
        V.assign(N, 0.0);
        for (size_t i = N - 1; i-- > 0;) {
            const double k = (double)pts[i];
            double best = 1e300;
            for (size_t j = i + 1; j < N; j++) {
                const double kp = (double)pts[j];
                const double c = (kp - k) + G + (k / kp) * V[j];
                if (c < best) best = c;
                if ((kp - k) + G >= best) break;   // later j only cost more
            }
            V[i] = best;
        }
    }
    // start of the interval that follows the one starting at k (k0 <= k <= K)
    u64 next_start(u64 k) const {
        size_t j = (size_t)(std::upper_bound(pts.begin(), pts.end(), k) -
                            pts.begin());
        if (j >= pts.size()) return K + 1;
        double best = 1e300;
        u64 arg = pts[j];
        for (; j < pts.size(); j++) {
            const double kp = (double)pts[j];
            const double c = (kp - (double)k) + G + ((double)k / kp) * V[j];
            if (c < best) { best = c; arg = pts[j]; }
            if ((kp - (double)k) + G >= best) break;
        }
        return arg;
    }
};

struct QSched {
    long q0 = 15;
    long q = 0;
    long m = 1;                // block size, for snapping
    bool opt = false;
    double G = 1500;           // gcd cost in degrees (opt)
    u64 K = 0;                 // last degree to scan (opt)
    DPSched dp;
    // choose q for the interval that starts at kstart
    void plan(u64 kstart) {
        if (!opt) { q = q + q0; return; }    // linear growth
        if (!dp.ready || dp.K != K || dp.k0 > kstart ||
            fabs(dp.G - G) > 0.02 * dp.G)
            dp.solve(kstart, K, G);
        u64 kn = dp.next_start(kstart);
        u64 w = kn > kstart ? kn - kstart : 1;
        q = (long)((w + (u64)m - 1) / (u64)m);
        if (q < 1) q = 1;
    }
};

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

// ---------------------------------------------------------------------------
// The schedule needs G = (cost of one gcd) / (cost of scanning one degree).
// Both are measured as the run goes (exponential averages over the whole
// process, so every survivor after the first starts well informed) unless
// --sched-G fixes G.  Round 10: they are also saved in every checkpoint, so
// a resumed run starts from the values it had (round 9 went back to the
// default until its first gcd -- Gord's round-5 observation).
static std::atomic<double> g_cost_gcd{0}, g_cost_deg{0};
static void cost_note(std::atomic<double> &c, double v) {
    if (!(v > 0)) return;
    double old = c.load();
    c.store(old > 0 ? 0.6 * old + 0.4 * v : v);
}

// ---------------------------------------------------------------------------
// scan parameters and outcome

struct SParams {
    u64 skip = 0;              // no factors of degree <= skip (REQUIRED)
    u64 maxd = 0;              // 0 = no cap
    long q0 = 0;               // 0 = auto (ceil(600/m)); linear schedule
    long zq = 0;               // -z: emit "u" when q would exceed
    long Zq = 0;               // -Z: cap q, keep scanning
    u64 canzass_max = 0;       // 0 = auto (2*r^(2/3))
    bool sched_opt = false;    // DP schedule (ts_main's default: on)
    u64 sched_G = 0;           // c_g/c_s in degrees; 0 = measured at run
                               // time (sched_G0 until the first gcd)
    u64 sched_G0 = 1500;       // the initial guess for that (per platform)
    int verbose = 0;
    int pend_max = 4;
    long hsnap_mb = 2048;      // RAM budget for fine-DDF h-table snapshots
    std::string ckpt_base;     // this s's checkpoint path; empty = none
    double ckpt_secs = 15 * 60.0;
    long die_after_blocks = -1;   // test hook: simulate SIGTERM
    long die_in_fine = -1;        // test hook: SIGTERM inside localize
    // error checking: after an interval at least this long, download the
    // next interval's start table BEFORE waiting on the gcd (see scan)
    double ec_presnap_secs = 2.0;
    TsTune tune;               // stamped into every checkpoint written
};

// G for the next interval: fixed by --sched-G, else measured
// Swan's theorem: for r = +-1 mod 8 (and s not dividing 2r in the way
// below) the trinomial has an odd number of irreducible factors, so if it
// is reducible its smallest factor has degree <= r/3 (otherwise <= r/2).
static bool swan_applies(u64 r, u64 s) {
    return (((r & 7) == 1) || ((r & 7) == 7)) &&
           ((((s & 1) == 0) && (((2 * r) % s) != 0)) ||
            (((s & 1) == 1) && (((2 * r) % (r - s)) != 0)));
}

// The 52 known Mersenne prime exponents (2^r - 1 prime) up to 136279841.
// For them an irreducible trinomial of degree r is primitive; for any
// other r the scan proves irreducibility only (round 11: the result line
// then says "irreducible", not "primitive").
static bool mersenne_exponent(u64 r) {
    static const u64 M[] = {
        2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
        2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209,
        44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787,
        1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583,
        25964951, 30402457, 32582657, 37156667, 42643801, 43112609,
        57885161, 74207281, 77232917, 82589933, 136279841 };
    for (u64 x : M) if (x == r) return true;
    return false;
}

static u64 sched_G_now(const SParams &P) {
    if (P.sched_G) return P.sched_G;
    double cg = g_cost_gcd.load(), cs = g_cost_deg.load();
    if (!(cg > 0) || !(cs > 0)) return P.sched_G0;
    double Gd = cg / cs;
    if (Gd < 1) Gd = 1;
    if (Gd > 1e8) Gd = 1e8;
    return (u64)Gd;
}

// Round 11.  When the whole scan of a survivor, skip+1 .. --maxd, is a
// single interval (the CPU stage: 56..279), whole blocks of the tuned m
// would end it up to m-1 degrees past maxd (m = 17: at 293).  A slightly
// smaller block size often ends it exactly.  Returns the m' in [3m/4, m]
// with the least modelled cost: the start, m'(k0 + m'/2) squarings, then
// ceil(L/m') blocks of m'^2 squarings and one product (timings from the
// tuning record).  Never larger than m, so memory never grows.
static int maxd_block_size(const SParams &P, u64 r, u64 s, int m) {
    if (!P.maxd || P.Zq || !(P.tune.t_sq > 0) || !(P.tune.t_mul > 0))
        return m;
    u64 rhigh = swan_applies(r, s) ? r / 3 : r / 2;
    if (P.maxd >= rhigh) return m;         // the Swan bound, not maxd, caps
    rhigh = P.maxd;
    const u64 k0 = P.skip + 1;
    if (rhigh < k0) return m;
    const u64 L = rhigh - k0 + 1;
    if (L % (u64)m == 0) return m;
    QSched Q;                              // as scan_one_s_once will plan it
    Q.opt = P.sched_opt;
    Q.q0 = P.q0 > 0 ? P.q0 : (long)((600 + m - 1) / m);
    Q.K = rhigh;
    Q.m = m;
    Q.G = (double)sched_G_now(P);
    Q.plan(k0);
    if ((u64)Q.q * (u64)m < L) return m;   // more than one interval
    const double ts = P.tune.t_sq, tm = P.tune.t_mul;
    auto cost = [&](int mm) {
        double nb = (double)((L + (u64)mm - 1) / (u64)mm);
        return ts * mm * ((double)k0 + mm / 2.0) + nb * (ts * mm * mm + tm);
    };
    int best = m;
    double cb = cost(m);
    for (int mm = (3 * m) / 4; mm < m; mm++) {
        if (mm < 2) continue;
        double c = cost(mm);
        if (c < 0.995 * cb) { cb = c; best = mm; }
    }
    return best;
}

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
static bool load_h(TsDev &G, const std::vector<u64> &h, const EcFold *expect,
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
static bool fetch_A(TsDev &G, std::vector<u64> &hostA) {
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
static bool init_h_checked(TsDev &G, u64 k) {
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
// checked separately, once per window (TsDev::ec_window_check).
static int do_block(TsDev &G, int verbose) {
    const bool tm = verbose >= 2;
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if (G.ec_on) G.ec_block_pre();
    if (tm) { G.sync(); t0 = trin_now_s(); }
    G.horner_C();
    if (ec_inject_now(ECI_C)) G.inject_C(ECI_C);
    if (tm) { G.sync(); t1 = trin_now_s(); }
    G.mul_A_by_C();
    if (tm) { G.sync(); t2 = trin_now_s(); }
    G.square_steps((u64)G.m, [](TsDev &D) {
        if (ec_inject_now(ECI_HMID)) D.inject_h(ECI_HMID);
    });
    if (ec_inject_now(ECI_H)) G.inject_h(ECI_H);
    if (tm) {
        G.sync();
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
static bool run_block(TsDev &G, const SParams &P, u64 kblk,
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
static bool localize(TsDev &G, GcdPool &pool, const SParams &P, u64 r,
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
        G.sync();
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
        G.hold_save();
        if (G.ec_on) G.hold_fold(hold_f);
        if (ec_inject_now(ECI_HOLD)) G.inject_hold(ECI_HOLD);
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
        G.sync();
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
            G.hold_restore();                  // back to state at kcur
            g_hold_restores++;
            if (G.ec_on) {              // the hold table sat idle in VRAM
                EcFold f;
                G.table_fold(f);
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
static ScanOut scan_one_s_once(TsDev &G, GcdPool &pool, const SParams &P,
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
    bool swan = swan_applies(r, s);
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
    Q.K = rhigh;
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
        // the measured costs of the run that wrote the checkpoint, unless
        // this process already measured its own
        bool seeded = false;
        if (!(g_cost_gcd.load() > 0) && H.cost_gcd > 0) {
            g_cost_gcd.store(H.cost_gcd);
            seeded = true;
        }
        if (!(g_cost_deg.load() > 0) && H.cost_deg > 0) {
            g_cost_deg.store(H.cost_deg);
            seeded = true;
        }
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
            if (seeded && P.sched_opt && !P.sched_G)
                printf("   [schedule costs from the checkpoint: gcd %.3g s, "
                       "%.3g ms/degree: G %" PRIu64 " degrees]\n",
                       g_cost_gcd.load(), 1e3 * g_cost_deg.load(),
                       sched_G_now(P));
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
        G.sync();
        if (P.verbose) {
            printf("Exponentiation took %f\n", trin_now_s() - t0);
            fflush(stdout);
        }
        G.reset_A();
        Q.G = (double)sched_G_now(P);
        Q.plan(k);
        if (P.Zq && Q.q > P.Zq) Q.q = P.Zq;
    }

    struct Pend { u64 seq, ka, kb; long q;
                  std::shared_ptr<HSnap> h0; };
    std::deque<Pend> pending;
    u64 vf = k;                // verified frontier: all degrees < vf clear

    // checkpoint candidate: h at the start of an interval whose
    // predecessors were all verified at snapshot time
    struct Cand { bool valid = false; u64 k_iv = 0; long q = 0;
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

    // the checkpoint header fields common to both flavors below
    auto ckpt_hdr = [&](CkptHdr &H) {
        memset(&H, 0, sizeof(H));
        H.magic = TS_CKPT_MAGIC; H.version = 2;
        H.r = r; H.s = s; H.skip = P.skip; H.m = (u64)G.m; H.nw = (u64)G.nw;
        H.sched_opt = P.sched_opt ? 1 : 0;
        H.cost_gcd = g_cost_gcd.load();
        H.cost_deg = g_cost_deg.load();
        H.tune = P.tune;
    };

    // Nothing unverified is ever saved: the mid-interval flavor first
    // closes the h-table chain window, and both the table and A are
    // downloaded through the fold-checked paths.
    auto write_ckpt = [&](bool exiting) -> void {
        ck_trip = false;
        if (P.ckpt_base.empty()) return;
        CkptHdr H;
        ckpt_hdr(H);
        std::vector<u64> aa;
        if (vf == k) {           // current interval fully verified behind us
            H.k_iv = k; H.blk = 0; H.q = (u64)Q.q;
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
            H.k_iv = cand.k_iv; H.blk = 0; H.q = (u64)cand.q;
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
        cost_note(g_cost_gcd, R.secs);
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
        if (read_fine_hdr(P.ckpt_base, FH) && FH.r == r && FH.s == s &&
            FH.skip == P.skip && FH.m == (u64)G.m && FH.nw == (u64)G.nw &&
            FH.q0 && FH.ka0 > P.skip) {
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
            if (P.verbose >= 2 && P.sched_opt && !P.sched_G)
                printf("   [schedule G %" PRIu64 " degrees: gcd %.1f s, "
                       "%.2f ms/degree]\n", sched_G_now(P),
                       g_cost_gcd.load(), 1e3 * g_cost_deg.load());
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
            cand.k_iv = k; cand.q = q;
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
                G.sync();
                write_ckpt(true);
                out.kind = ScanOut::INTERRUPTED;
                goto done;
            }
            if (!P.ckpt_base.empty() &&
                trin_now_s() - last_ckpt > P.ckpt_secs) {
                G.sync();
                write_ckpt(false);
                if (ck_trip) goto ec_retry;
            }
            if (!run_block(G, P, k + (u64)b * G.m, b == q - 1))
                goto ec_retry;
        }
        blocks_done_hook = 0;
        G.sync();
        double iv_wall = trin_now_s() - iv_t0 - iv_poll;
        out.degrees_scanned += (u64)(q - blk0) * G.m;
        if (q > blk0)
            cost_note(g_cost_deg, iv_wall / (double)((q - blk0) * G.m));
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
        if (ec_inject_now(ECI_IDLE)) G.inject_h(ECI_IDLE);
        if ((int)pending.size() >= P.pend_max) {   // bound speculation
            Pend pv = pending.front();
            pending.pop_front();
            GcdRes R = pool.wait_get(pv.seq);
            TS_CONSUME(pv, R);
        }
        Q.G = (double)sched_G_now(P);
        Q.plan(k);
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
        ckpt_hdr(H);
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
static ScanOut scan_one_s(TsDev &G, GcdPool &pool, const SParams &P,
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

// ---------------------------------------------------------------------------
// Selftest suites that run on any backend.  The environment says how to
// make a device and which multiplier (if any) the gcd engine may offload to.
struct TsTestEnv {
    const char *name = "";
    // a device initialised for r with block size m; variant 1 selects the
    // backend's alternate squaring path (GPU: the legacy 3-kernel one;
    // CPU: the generic spread-and-fold one instead of BLZ)
    std::unique_ptr<TsDev> (*make)(u64 r, int m, int variant) = nullptr;
    GpuMulHook *(*hook_new)(int max_fm) = nullptr;     // null: no device mults
    void (*hook_free)(GpuMulHook *) = nullptr;
};

// planted-factor helper: the product of two polynomials, by the device
// multiplier when there is one (that exercises it too), else on the host
static std::vector<u64> st_mul(GpuMulHook *hk, const std::vector<u64> &a,
                               u64 abits, const std::vector<u64> &b,
                               u64 bbits) {
    std::vector<u64> out;
    if (hk && hk->mul(a.data(), a.size(), abits, b.data(), b.size(), bbits,
                      out))
        return out;
    out.assign(mul_out_words(abits, bbits), 0);
    gf2x_naive_mul(a.data(), abits, b.data(), bbits, out.data());
    return out;
}

// suites (3) and (3c) of the old selftest, backend-independent: the
// h-table build and Horner product against the host, and the error-checking
// predictions/detections against real data
static int selftest_table_suites(TsTestEnv &E) {
    int bad = 0;
    // (3) init_h + Horner vs direct product of (x + x^(2^(k+i))) mod T
    {
        bool ok = true;
        const u64 rr = 4423;
        const u64 svs[] = {130, 131};             // both parities of s
        int ms[] = {2, 3, 5, 8};
        u64 ks[] = {1, 5, 17};
        for (u64 sv : svs)
        for (int mi = 0; mi < 4 && ok; mi++)
            for (int ki = 0; ki < 3 && ok; ki++) {
                int m = ms[mi];
                u64 kst = ks[ki];
                auto Gp = E.make(rr, m, 0);
                TsDev &G = *Gp;
                G.set_s(sv);
                G.init_h_at(kst);
                G.horner_C();
                G.sync();
                std::vector<u64> Cg;
                G.C_to_host(Cg);
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
                Cg.resize(Ch.size());
                if (Cg != Ch) {
                    printf("  init_h/horner mismatch s=%" PRIu64 " m=%d kst=%"
                           PRIu64 "\n", sv, m, kst);
                    ok = false;
                }
                G.fini();
            }
        printf("init_h + fused Horner vs host block product        %s\n",
               ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }
    // (3c) error-checking folds and predictions against host references
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
            auto Gp = E.make(cs.r, m, 0);
            TsDev &G = *Gp;
            G.set_s(cs.s);
            G.ec_setup(true);
            std::vector<u64> hh((size_t)m * G.nw);
            for (int j = 0; j < m; j++) {
                std::vector<u64> v = st_res(cs.r);
                memcpy(&hh[(size_t)j * G.nw], v.data(), G.nw * 8);
            }
            std::vector<u64> a = st_res(cs.r);
            auto load = [&] {
                G.h_from_host(hh);
                G.A_from_host(a);
                G.ecD_zero();
            };
            // folds taken where the data lives == host folds
            load();
            {
                EcFold fd, fh;
                G.table_fold(fd);
                ecf_fold_host(fh, hh.data(), hh.size());
                if (!ecf_eq(fd, fh)) ok_fold = false;
            }
            // one clean block: every prediction must match the real data
            G.ec_block_pre();
            G.horner_C();
            G.mul_A_by_C();
            if (G.ec_block_verify()) ok_blk = false;
            {
                std::vector<u64> dD, dC, dAn, xs(G.nw, 0);
                G.ecD_to_host(dD);
                G.C_to_host(dC);
                G.A_to_host(dAn);
                for (int j = 0; j < m; j++)
                    for (size_t w = 0; w < G.nw; w++)
                        xs[w] ^= hh[(size_t)j * G.nw + w];
                EcFold fh;
                dD.resize(G.nw);
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
                    G.inject_h(ECI_H);
                G.horner_C();
                if (ec_inject_now(ECI_C)) G.inject_C(ECI_C);
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
                            G.inject_h(ECI_HMID);
                    }
                    G.ec_w++;
                }
                if (G.ec_window_check() != (flip == 0)) ok_chain = false;
                // idle table: unchanged passes, one flip fails
                if (!G.ec_window_check()) ok_chain = false;
                G.inject_h(ECI_IDLE);
                if (G.ec_window_check()) ok_chain = false;
            }
            // closed form of the table digest, on real tables
            if (ci % 3 == 0) {
                for (u64 kst : {(u64)0, (u64)1, (u64)9}) {
                    G.init_h_at(kst);
                    if (!G.ec_verify_table_at(kst)) ok_init = false;
                    G.inject_h(ECI_INIT);
                    if (G.ec_verify_table_at(kst)) ok_init = false;
                }
            }
            G.fini();
        }
        g_ec_quiet = sv_quiet;
        printf("EC: table folds vs host (%zu cases)                  %s\n",
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
    return bad;
}

// the schedule: the DP must reproduce a brute-force optimum on a small
// range, never plan past K, and depend on the position only
static int selftest_sched() {
    bool ok = true;
    for (double G : {3.0, 40.0, 220.0}) {
        const u64 k0 = 20, K = 140;
        DPSched dp;
        dp.solve(k0, K, G);
        // brute force over every boundary set on the integers k0..K+1
        std::vector<double> W(K + 2, 0.0);
        for (u64 k = K + 1; k-- > k0;) {
            double best = 1e300;
            for (u64 kp = k + 1; kp <= K + 1; kp++) {
                double c = (double)(kp - k) + G +
                           ((double)k / (double)kp) * W[kp];
                if (c < best) best = c;
            }
            W[k] = best;
        }
        // the grid is the integers here (ratio steps < 1), so V must agree
        if (fabs(dp.V[0] - W[k0]) > 1e-9 * W[k0]) ok = false;
        for (u64 k = k0; k <= K; k++) {
            u64 kn = dp.next_start(k);
            if (kn <= k || kn > K + 1) ok = false;
        }
    }
    {   // a production-sized solve stays fast and monotone
        DPSched dp;
        double t0 = trin_now_s();
        dp.solve(263, 136279841ULL / 3, 1300.0);
        double dt = trin_now_s() - t0;
        u64 k = 263, prev_w = 0;
        int steps = 0;
        bool mono = true;
        while (k <= 136279841ULL / 3 && steps < 100000) {
            u64 kn = dp.next_start(k);
            if (kn <= k) { mono = false; break; }
            // widths grow (up to the grid's 0.2% steps and block snapping)
            if ((kn - k) + k / 200 + 64 < prev_w) mono = false;
            prev_w = kn - k;
            k = kn;
            steps++;
        }
        if (!mono || dt > 5.0) ok = false;
    }
    printf("schedule: DP == brute force, monotone at full size  %s\n",
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// suites (4)-(14) of the old selftest: whole scans against the per-degree
// oracle, fine DDF, -z/-Z, checkpoints, the gcd engines, and the
// error-checking fault sweeps.  NTL builds only.
static int selftest_scan_suites(TsTestEnv &E, int gcd_threads) {
    int bad = 0;
#ifndef HAVE_NTL
    (void)E;
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
    const u64 ec0_fail = g_ecs.total(), ec0_blocks = g_ecs.blocks.load(),
              ec0_win = g_ecs.windows.load(),
              ec0_rst = g_ecs.restarts.load();
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
    auto right_of = [&](u64 sv, const ScanOut &res) {
        const OracleOut &o = want[sv];
        bool okk = o.found ? (res.kind == ScanOut::FOUND && res.d == o.d)
                           : (res.kind == ScanOut::GAVE_UP);
        if (okk && o.found && o.mask_ok) okk = res.hex == mask_hex(o.mask);
        return okk;
    };
    {
        struct Cfg { int m; long q0; long Zq; int variant; const char *nm;
                     bool sopt; u64 sG; };
        Cfg cfgs[] = {{4, 3, 0, 0, "m=4 q0=3", false, 0},
                      {8, 2, 0, 0, "m=8 q0=2", false, 0},
                      {6, 1, 1, 1, "m=6 const-q alt-sq", false, 0},
                      {14, 15, 0, 0, "m=14 q0=15", false, 0},
                      {3, 2, 0, 0, "m=3 q0=2", false, 0},
                      {6, 3, 0, 0, "m=6 sched-opt G=60", true, 60},
                      {5, 3, 0, 0, "m=5 sched-opt G=7", true, 7}};
        for (auto &cf : cfgs) {
            auto Gp = E.make(rr, cf.m, cf.variant);
            TsDev &G = *Gp;
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = cf.q0; P.Zq = cf.Zq;
            P.sched_opt = cf.sopt; P.sched_G = cf.sG;
            P.canzass_max = 2 * 270;   // 2*r^(2/3) ~ 540
            P.verbose = 0;
            u64 mism = 0;
            for (u64 sv : surv) {
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                if (!right_of(sv, res)) {
                    mism++;
                    const OracleOut &o = want[sv];
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
    }

    // (5) fineDDF forced on every hit (canzass_max = 1)
    {
        auto Gp = E.make(rr, 5, 0);
        TsDev &G = *Gp;
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
            if (!right_of(sv, res)) mism++;
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
            auto Gp = E.make(rr, 4, 0);
            TsDev &G = *Gp;
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 2;
            P.canzass_max = 540;
            P.verbose = 0;
            P.zq = 3;              // trips before reaching deep_d
            ScanOut r1 = scan_one_s(G, pool, P, rr, deep_s, gseq, nullptr);
            okk = okk && r1.kind == ScanOut::GAVE_UP;
            P.zq = 0;
            P.Zq = 3;              // capped q but keeps going
            ScanOut r2 = scan_one_s(G, pool, P, rr, deep_s, gseq, nullptr);
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
            auto Gp = E.make(rr, 5, 0);
            TsDev &G = *Gp;
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 2;
            P.canzass_max = 540;
            P.verbose = 0;
            P.ckpt_base = "/tmp/tsfactor_test.ckpt";
            P.ckpt_secs = 1e9;
            long kills[] = {0, 1, 3, 6, 11};
            for (long ka : kills) {
                remove_ckpts(P.ckpt_base);
                SParams P2 = P;
                P2.die_after_blocks = ka;
                ScanOut r1 = scan_one_s(G, pool, P2, rr, deep_s, gseq,
                                        nullptr);
                if (r1.kind != ScanOut::INTERRUPTED) {
                    // died late enough to finish: fine, just check
                    okk = okk && r1.kind == ScanOut::FOUND && r1.d == deep_d;
                    continue;
                }
                ResumeState RS;
                RS.valid = load_ckpt(P.ckpt_base, RS.H, RS.h, RS.a);
                ScanOut r2;
                if (RS.valid && RS.H.s == deep_s)
                    r2 = scan_one_s(G, pool, P, rr, deep_s, gseq, &RS);
                else            // no verified state: restart is correct
                    r2 = scan_one_s(G, pool, P, rr, deep_s, gseq, nullptr);
                okk = okk && r2.kind == ScanOut::FOUND && r2.d == deep_d;
                if (!okk) printf("  ckpt kill@%ld: d=%" PRIu64 " want %"
                                 PRIu64 "\n", ka, r2.d, deep_d);
            }
            remove_ckpts(P.ckpt_base);
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
        const std::string fb = "/tmp/tsfactor_fine.ckpt";
        auto wipe = [&] { remove_ckpts(fb); remove_fine(fb); };
        auto Gp = E.make(rr, 5, 0);
        TsDev &G = *Gp;
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
            ScanOut r1 = scan_one_s(G, pool, P2, rr, kv.first, gseq, nullptr);
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
            ScanOut r1 = scan_one_s(G, pool, P2, rr, deep_s, gseq, nullptr);
            g_stop = 0;
            if (r1.kind != ScanOut::INTERRUPTED) {   // finished first
                okk = okk && r1.kind == ScanOut::FOUND && r1.d == deep_d;
                continue;
            }
            // fresh process would see only the standby file
            ScanOut r2 = scan_one_s(G, pool, P, rr, deep_s, gseq, nullptr);
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

    // (8) hybrid HGCD gcd vs NTL gcd: the device-mult (if any) and CPU-mult
    //     paths, nonlinear and adversarial F2-linear (LFSR) data, planted
    //     common factors, degenerate shapes
    {
        bool okk = true;
        u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
        hyb_ntl_finish_bits = 64;    // force deep HGCD recursion
        hyb_gpu_min_bits = 2048;     // exercise the hook heavily
        hyb_reset_stats();
        GpuMulHook *gm8 = E.hook_new ? E.hook_new(15) : nullptr;
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
                                     b.size(), &g2, ~0ULL, gm8);
            u64 d3 = poly_gcd_hybrid(a.data(), a.size(), b.data(),
                                     b.size(), &g3, ~0ULL, nullptr);
            bool o = d1 == d2 && d1 == d3 &&
                     (d1 == 0 || (g1 == g2 && g1 == g3));
            if (!o) okk = false;
        };
        for (u64 bits : std::vector<u64>{600, 5000, 60000}) {
            for (int lin = 0; lin < 2; lin++) {
                chk(rvb(bits, lin), rvb(bits - 1 - nl() % 30, lin));
                // planted common factor
                u64 gd = 20 + nl() % (bits / 3);
                auto g = rvb(gd + 1, false);
                auto u = rvb(bits - gd, lin), w = rvb(bits - gd, lin);
                chk(st_mul(gm8, g, gd + 1, u, bits - gd),
                    st_mul(gm8, g, gd + 1, w, bits - gd));
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
        if (gm8) E.hook_free(gm8);
        if (E.hook_new)
            printf("hybrid HGCD vs NTL (gpu-mults %" PRIu64 ", fb %" PRIu64
                   ")             %s\n", g8, f8, okk ? "PASS" : "FAIL");
        else
            printf("gcd engine vs NTL (CPU products)                  %s\n",
                   okk ? "PASS" : "FAIL");
        if (!okk) bad++;
    }

    // (9) end-to-end scan with the hybrid gcd forced through the pool
    //     (thresholds shrunk so 4423-bit gcds take the path; device mults
    //     when the build has them)
    {
        u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
        hyb_gpu_min_bits = 512;
        hyb_ntl_finish_bits = 256;
        GpuMulHook *gm9 = E.hook_new ? E.hook_new(12) : nullptr;
        g_gm = gm9;
        g_use_hybrid = true;
        g_hyb_xcheck = true;
        hyb_reset_stats();
        auto Gp = E.make(rr, 6, 0);
        TsDev &G = *Gp;
        SParams P;
        P.skip = skip; P.maxd = maxd; P.q0 = 3;
        P.canzass_max = 540;
        P.verbose = 0;
        u64 mism = 0;
        for (size_t i = 0; i < surv.size(); i += 4) {
            u64 sv = surv[i];
            ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
            if (!right_of(sv, res)) mism++;
        }
        // A verdict ends a scan via forget_all_pending(), which does
        // NOT wait for the worker running that gcd -- it may still be
        // inside the hook.  Drain before freeing the hook and the device.
        size_t stale9 = pool.drain();
        u64 g9, c9, f9;
        hyb_get_stats(g9, c9, f9);
        g_gm = nullptr;
        g_use_hybrid = false;
        g_hyb_xcheck = false;
        hyb_gpu_min_bits = sv_min;
        hyb_ntl_finish_bits = sv_fin;
        G.fini();
        if (gm9) E.hook_free(gm9);
        if (E.hook_new)
            printf("end-to-end, hybrid gcd forced (gpu-mults %" PRIu64
                   ", fb %" PRIu64 ", drained %zu)   %s\n", g9, f9, stale9,
                   mism ? "FAIL" : "PASS");
        else
            printf("end-to-end, gcd engine forced (drained %zu)         %s\n",
                   stale9, mism ? "FAIL" : "PASS");
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
            auto Gp = E.make(rr, 5, 0);
            TsDev &G = *Gp;
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
                    auto Gp = E.make(rr, 5, 0);
                    TsDev &G = *Gp;
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
                    ScanOut res = scan_one_s(G, pool, P, rr, tgt_s, gseq,
                                             nullptr);
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
                    auto Gp = E.make(rr, 5, 0);
                    TsDev &G = *Gp;
                    SParams P;
                    P.skip = skip; P.maxd = maxd; P.q0 = 2;
                    P.canzass_max = 540;
                    P.verbose = 0;
                    g_eci.arm(site, N, 31 * (u64)site + (u64)N);
                    ScanOut res = scan_one_s(G, pool, P, rr, deep_s, gseq,
                                             nullptr);
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
                auto Gp = E.make(rr, 5, 0);
                TsDev &G = *Gp;
                g_eci.arm(ECI_H, -1, 5);
                ScanOut o = scan_one_s(G, pool, P, rr, deep_s, gseq, nullptr);
                g_eci.disarm();
                pool.drain();
                ok_fatal = (o.kind == ScanOut::EC_FATAL &&
                            o.ec_restarts == g_ec.max_retry);
                G.fini();
            }
            {   // restart from a mid-interval state (checkpoint taken
                // before every block), not from the interval start
                const std::string cb = "/tmp/tsfactor_ec.ckpt";
                auto wipe = [&] { remove_ckpts(cb); remove_fine(cb); };
                wipe();
                auto Gp = E.make(rr, 5, 0);
                TsDev &G = *Gp;
                SParams P2 = P;
                P2.ckpt_base = cb;
                P2.ckpt_secs = -1;
                u64 r0 = g_ecs.restarts.load();
                g_eci.arm(ECI_H, 9, 6);
                ScanOut o = scan_one_s(G, pool, P2, rr, deep_s, gseq, nullptr);
                g_eci.disarm();
                pool.drain();
                ok_mid = right(o) && g_ecs.restarts.load() == r0 + 1;
                wipe();
                G.fini();
            }
            {   // idle flip during the gcd wait, table saved before the
                // wait: repaired in place, nothing rescanned
                auto Gp = E.make(rr, 5, 0);
                TsDev &G = *Gp;
                SParams P3 = P;
                P3.ec_presnap_secs = 0;
                P3.pend_max = 1;
                u64 r0 = g_ecs.restarts.load(), p0 = g_ecs.repairs.load();
                g_eci.arm(ECI_IDLE, 1, 7);
                ScanOut o = scan_one_s(G, pool, P3, rr, deep_s, gseq, nullptr);
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
    //      suite (8), CPU-mult and device-mult paths, always certified;
    //      then a fault at each verified step in turn must never yield
    //      a certified wrong answer
    {
        bool ok_eq = true, ok_flt = true;
        u64 sv_min = hyb_gpu_min_bits, sv_fin = hyb_ntl_finish_bits;
        hyb_ntl_finish_bits = 64;    // force a long verified chain
        hyb_gpu_min_bits = 2048;
        hyb_cert_xgcd = false;
        GpuMulHook *gm13 = E.hook_new ? E.hook_new(15) : nullptr;
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
        auto chk = [&](const std::vector<u64> &a, const std::vector<u64> &b) {
            std::vector<u64> g1;
            u64 d1 = poly_gcd_ntl(a.data(), a.size(), b.data(), b.size(),
                                  &g1, ~0ULL);
            EcFold fa, fb;
            ecf_fold_host(fa, a.data(), a.size());
            ecf_fold_host(fb, b.data(), b.size());
            for (int hook = 0; hook < 3; hook++) {
                if (hook == 1 && !gm13) continue;
                std::vector<u64> g2;
                int st = -1;
                hyb_fault_at = 0;
                hyb_cert_xgcd = (hook == 2);        // 2: the XGCD mode
                u64 d2 = poly_gcd_certified(a.data(), a.size(), b.data(),
                                            b.size(), &g2, ~0ULL,
                                            hook == 1 ? gm13 : nullptr,
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
            for (int hook = 0; hook < 2 && hyb_threads_ok(); hook++) {
                if (hook == 1 && !gm13) continue;
                // the products of each matrix step on three threads:
                // alone, and sharing the work with the device hook (which
                // takes the products above its size threshold)
                std::vector<u64> g2;
                int st = -1;
                const int sv_t = hyb_threads;
                const u64 sv_mb = hyb_mt_min_bits;
                hyb_threads = 3;
                hyb_mt_min_bits = 600;
                u64 d2 = poly_gcd_certified(a.data(), a.size(), b.data(),
                                            b.size(), &g2, ~0ULL,
                                            hook ? gm13 : nullptr,
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
                EcFold badf = fa;
                badf.w[7] ^= 1;
                std::vector<u64> g2;
                int st = -1;
                poly_gcd_certified(a.data(), a.size(), b.data(), b.size(),
                                   &g2, ~0ULL, nullptr, &badf, &fb, &st);
                if (st != GCD_CERT_INPUT) ok_flt = false;
            }
        };
        for (u64 bits : std::vector<u64>{700, 5000, 40000}) {
            chk(rvb(bits), rvb(bits - 1 - nl() % 30));
            u64 gd = 20 + nl() % (bits / 3);
            auto g = rvb(gd + 1);
            auto u = rvb(bits - gd), w = rvb(bits - gd);
            chk(st_mul(gm13, g, gd + 1, u, bits - gd),
                st_mul(gm13, g, gd + 1, w, bits - gd));   // planted factor
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
        if (gm13) E.hook_free(gm13);
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
        GpuMulHook *gm14 = E.hook_new ? E.hook_new(12) : nullptr;
        u64 mism = 0, runs = 0;
        const u64 c0 = g_ecs.gcd_cert.load(), u0 = g_ecs.gcd_uncert.load(),
                  d0 = g_ecs.total();
        const u64 rr0 = g_ecs.gcd_reruns.load();
        for (int hook = 0; hook < 3; hook++) {
            g_gm = hook == 1 ? gm14 : nullptr;
            hyb_cert_xgcd = (hook == 2);        // 2: the XGCD mode
            auto Gp = E.make(rr, 6, 0);
            TsDev &G = *Gp;
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 3;
            P.canzass_max = 540;
            P.verbose = 0;
            for (size_t i = hook; i < surv.size(); i += 4) {
                u64 sv = surv[i];
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                runs++;
                if (!right_of(sv, res)) mism++;
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
            auto Gp = E.make(rr, 6, 0);
            TsDev &G = *Gp;
            SParams P;
            P.skip = skip; P.maxd = maxd; P.q0 = 3;
            P.canzass_max = 540;
            P.verbose = 0;
            P.pend_max = 1;
            for (size_t i = 0; i < surv.size(); i += 5) {
                u64 sv = surv[i];
                hyb_fault_at = 1 + (long)(i % 7);
                hyb_fault_reset();
                ScanOut res = scan_one_s(G, pool, P, rr, sv, gseq, nullptr);
                hyb_fault_at = 0;
                fruns++;
                if (!right_of(sv, res)) fm++;
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
        if (gm14) E.hook_free(gm14);
        printf("EC: scans with certified gcd (%" PRIu64 " scans, %" PRIu64
               " gcds)    %s\n", runs, ncert, ok_clean ? "PASS" : "FAIL");
        printf("EC: faults inside the gcd (%" PRIu64 " scans, %" PRIu64
               " re-runs)     %s\n", fruns, nrr,
               (fm == 0 && nrr > 0) ? "PASS" : "FAIL");
        if (!ok_clean) bad++;
        if (fm || !nrr) bad++;
    }
    pool.stop();
#endif // HAVE_NTL
    return bad;
}

static void selftest_footer(int bad) {
#ifdef HAVE_NTL
    printf(bad ? "SELFTEST: %d FAILURE(S)\n" : "SELFTEST: all PASS\n", bad);
#else
    printf(bad ? "SELFTEST: %d FAILURE(S)\n"
               : "SELFTEST: kernel suites PASS; resolution suites SKIPPED "
                 "(built without NTL)\n", bad);
#endif
}

// ---------------------------------------------------------------------------
// what each program (GPU / CPU build) supplies to the common main

struct TsPlatform {
    virtual ~TsPlatform() {}
    virtual const char *prog() const = 0;          // "tsfactor", "tsfactor_cpu"
    virtual const char *build() const = 0;         // "GPU build", "CPU build"
    // who is measuring: program, round, device -- the tuning record's key
    virtual std::string fingerprint() = 0;
    virtual bool has_gpu_gcd() const { return false; }
    // options only this build knows; true if consumed (i advanced past
    // any value it took)
    virtual bool parse_opt(const std::string &a, int &i, int argc,
                           char **argv) {
        (void)a; (void)i; (void)argc; (void)argv;
        return false;
    }
    virtual const char *usage_extra() const { return ""; }
    virtual void start() {}                        // after option parsing
    virtual TsDev *make_dev() = 0;
    virtual void defaults(TsTune &T) = 0;          // compiled-in configuration
    // a configuration fixed on the command line (GPU: --fft); true if so
    virtual bool pinned(TsTune &T) { (void)T; return false; }
    // time the variants and the operations; fill T (configuration, t_sq,
    // t_mul, m) and leave T's configuration active
    virtual void tune(TsDev &G, TsTune &T, int verbose) = 0;
    virtual void apply(const TsTune &T) = 0;       // make T's config active
    virtual std::string describe(const TsTune &T) = 0;
    virtual int choose_m(double t_sq, double t_mul) = 0;
    virtual int selftest(int gcd_threads) = 0;
    virtual int bench(u64 r, u64 s, bool bench_gcd, bool gpu_gcd,
                      u64 gpu_min_bits, bool ecg_xgcd, bool gcd_ntl) = 0;
    virtual void gcd_hook_init(TsDev &G, u64 min_bits) {
        (void)G; (void)min_bits;
    }
    virtual void gcd_hook_fini() {}
    virtual u64 default_G0() const = 0;            // gcd cost in degrees
    virtual void default_threads(int &gcd_threads, int &pend_max) {
        (void)gcd_threads; (void)pend_max;
    }
    // threads inside one gcd when --gcd-mt is not given: 0 = automatic
    // (the cores the scan and the gcd workers leave over, at most 4)
    virtual int default_gcd_mt() const { return 0; }
    virtual void mem_report(TsDev &G, const SParams &P, bool gpu_gcd) = 0;
};

// measure t_sq and t_mul on G (any s, r as initialised) and choose m
static void ts_measure_m(TsPlatform &PL, TsDev &G, TsTune &T) {
    if (G.m == 0) G.set_m(8);
    G.set_s((G.r - 1) / 2);
    G.ec_setup(false);
    T.t_sq = G.time_square(6);
    T.t_mul = G.time_mul(2);
    T.m = (u64)PL.choose_m(T.t_sq, T.t_mul);
}

// --bench --bench-gcd: one full-size gcd with NTL, the uncertified engine
// (under --gpu-gcd) and the certified gcd the configuration would run
static void ts_bench_gcd(u64 r, u64 sb, bool ecg_xgcd, bool gcd_ntl) {
#ifdef HAVE_NTL
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
    g_keep_bits = r + 1;
    // the reference: NTL's own gcd (what --gcd-ntl runs)
    double t0 = trin_now_s();
    std::vector<u64> gg;
    u64 gd = poly_gcd_ntl(A.data(), A.size(), T.data(), T.size(), &gg,
                          g_keep_bits);
    double dt = trin_now_s() - t0;
    printf("gcd bench (random A vs T, NTL): %.2f s (gdeg %" PRIu64 ")\n",
           dt, gd);
    fflush(stdout);
    if (g_use_hybrid) {          // the engine, uncertified (--no-ec)
        double th = trin_now_s();
        std::vector<u64> gh;
        u64 gdh = ts_gcd(A.data(), A.size(), T.data(), T.size(), &gh);
        double dth = trin_now_s() - th;
        u64 ng, nc, nf;
        hyb_get_stats(ng, nc, nf);
        printf("gcd bench (random A vs T, hybrid): %.2f s (gdeg %" PRIu64
               ", %+.1f%% vs NTL)\n", dth, gdh, 100.0 * (dth - dt) / dt);
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
        int st2 = -1;
        hyb_reset_stats();
        double t1 = trin_now_s();
        u64 gd2 = ts_gcd_cert(A.data(), A.size(), T.data(), T.size(), &g2,
                              &fa, &ft, &st2);
        double dt2 = trin_now_s() - t1;
        u64 ng, nc, nf;
        hyb_get_stats(ng, nc, nf);
        printf("gcd bench, CERTIFIED (%s): %.2f s (gdeg %" PRIu64
               ", status %s, %+.1f%% vs NTL)\n",
               ecg_xgcd ? "NTL XGCD" : g_gm ? "chain, GPU mults"
                                            : "chain, CPU mults",
               dt2, gd2,
               st2 == 0 ? "certified" : st2 == 3 ? "NOT certified"
               : "CERTIFICATE FAILED",
               100.0 * (dt2 - dt) / dt);
        printf("  mults: gpu %" PRIu64 " cpu %" PRIu64 " fallbacks %"
               PRIu64 "\n", ng, nc, nf);
        if (hyb_threads > 1 && !ecg_xgcd)
            printf("  (--gcd-mt %d: %d threads inside the gcd%s)\n",
                   hyb_threads, hyb_threads,
                   g_gm ? ", for the products below the GPU's size "
                          "threshold" : "");
        if (gd2 != gd) printf("  *** MISMATCH vs NTL ***\n");
    }
#else
    (void)r; (void)sb; (void)ecg_xgcd; (void)gcd_ntl;
    printf("gcd bench: needs an NTL build\n");
#endif
}

static void ts_usage(TsPlatform &PL) {
    fprintf(stderr,
        "Usage: %s <r> <survivors.txt> --skip D [options]\n"
        "       %s --selftest [--gcd-threads N]\n"
        "       %s --bench <r> [s] [--bench-gcd]\n"
        "       %s --version\n"
        "Options:\n"
        "  --skip D        REQUIRED: inputs have no factor of degree <= D\n"
        "  --maxd D        do not scan degrees > D (emit \"s u\")\n"
        "  --m M           block size 2..64 (default: from the tuning)\n"
        "  --sched opt|linear   interval schedule (default opt: the expected-\n"
        "                  cost optimum by dynamic programming, from the\n"
        "                  survival law S(k)=(16/9)/k; linear = factor -f 1)\n"
        "  --sched-G N     gcd cost / scan cost in degrees.  Default: measured\n"
        "                  as the run goes, kept in checkpoints (a fixed N\n"
        "                  makes the schedule reproducible)\n"
        "  --q0 N          linear schedule: blocks in the first interval\n"
        "                  (default 600/m)\n"
        "  -z Q            emit \"s u\" when q would exceed Q (factor -z)\n"
        "  -Z Q            cap q at Q but keep scanning\n"
        "  --out PREFIX    results -> PREFIX.results.txt (default ts);\n"
        "                  checkpoints PREFIX_r<r>_s<s>.ckpt; one process\n"
        "                  per prefix (PREFIX.lock)\n"
        "  --no-ckpt       disable checkpointing\n"
        "  --ckpt-mins N   periodic checkpoint cadence (default 15)\n"
        "  --gcd-threads N CPU GCD pool size\n"
        "  --pend-max N    max speculative intervals in flight\n"
        "  --canzass-max D fineDDF threshold (default 2*r^(2/3))\n"
        "  --hsnap-mb N    RAM for h-table snapshots (default 2048); one per\n"
        "                  interval with an outstanding gcd, so a deep hit\n"
        "                  is localized without rebuilding the h-table\n"
        "                  (hours at r=136M).  0 restores the old rebuild\n"
        "  --gcd-mt N      CPU threads inside each gcd (default: automatic)\n"
        "  --gcd-finish-bits N    gcd engine hands over to NTL below N bits\n"
        "  --gcd-base-bits N      gcd engine: word-level base case size\n"
        "  --gcd-full-apply       gcd engine: full matrix application (A/B)\n"
        "  --gcd-ntl       plain NTL gcd instead: NOT checked (A/B timing)\n"
        "  --bench-gcd     with --bench: also time one full-size gcd\n"
        "  --selftest-reps N   repeat --selftest N times\n"
        "  -v              verbose (repeat for more)\n"
        "Tuning (round 10): the run times its configurable arithmetic when\n"
        "it starts and stamps the choice into its checkpoints; a run that\n"
        "resumes a checkpoint made by this binary on this device reuses it.\n"
        "  --retune        time again even when resuming\n"
        "  --no-tune       compiled-in defaults (block size still measured\n"
        "                  unless --m is given)\n"
        "Error checking (ON by default; detects hardware faults and restarts\n"
        "from the newest verified state; ~1%% overhead on a GPU, ~3%% on a\n"
        "CPU):\n"
        "  --no-ec         turn it off (testing / A-B timing only)\n"
        "  --ecw N         check the h-table chain every N blocks (default 8)\n"
        "  --ecg-xgcd      certify the gcd with one verified NTL extended gcd\n"
        "  --ecg-dup       compute every gcd twice with NTL and compare\n"
        "  --ecg           accepted; certifying is the default\n"
        "  --ect           print an error-checking cost breakdown at exit\n"
        "  --eci0          skip the verification of the h-table init\n"
        "  --ec-retries N  give up on an s after N failed restarts from the\n"
        "                  same point (default 3; exit status 3)\n"
        "  --ec-fault SITE:N   TEST: flip one bit at the N-th opportunity of\n"
        "                  SITE (h hmid a c t red idle init xferh xfera snap\n"
        "                  hold job gcd)\n"
        "%s", PL.prog(), PL.prog(), PL.prog(), PL.prog(), PL.usage_extra());
    exit(1);
}

// checkpoints <prefix>_r<r>_s<N>.ckpt in the prefix's directory, by s
static std::set<u64> list_ckpt_s(const std::string &prefix, u64 r) {
    std::set<u64> out;
    std::string dir = ".", base = prefix;
    size_t sl = prefix.rfind('/');
    if (sl != std::string::npos) {
        dir = prefix.substr(0, sl);
        if (dir.empty()) dir = "/";
        base = prefix.substr(sl + 1);
    }
    char mid[48];
    snprintf(mid, sizeof mid, "_r%" PRIu64 "_s", r);
    const std::string head = base + mid;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    while (struct dirent *e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() <= head.size() + 5 || n.compare(0, head.size(), head))
            continue;
        const char *p = n.c_str() + head.size();
        char *end = nullptr;
        unsigned long long sv = strtoull(p, &end, 10);
        if (end == p || strcmp(end, ".ckpt") != 0 || !sv) continue;
        out.insert((u64)sv);
    }
    closedir(d);
    return out;
}

static int ts_main(int argc, char **argv, TsPlatform &PL) {
    crc32_init();
    u64 r = 0, s_bench = 0;
    std::string surv_path, out_prefix = "ts";
    SParams P;
    P.skip = (u64)-1;
    P.sched_opt = true;
    P.sched_G0 = PL.default_G0();
    int gcd_threads = 0, want_m = 0;
    bool do_selftest = false, do_bench = false, no_ckpt = false,
         gpu_gcd = false, bench_gcd = false, retune = false, no_tune = false;
#ifndef HAVE_NTL
    bool allow_no_ntl = false;
#endif
    u64 gcd_min_bits_opt = 0, gcd_base_bits = 0;
    int selftest_reps = 1;
    bool ecg_explicit = false, ecg_xgcd = false, gcd_ntl = false;
    int gcd_mt = 0;            // 0 = automatic
    bool gcd_mt_auto = false;
    bool pend_set = false;

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
        else if (a == "--version") {
            printf("%s %s (%s)\n", PL.prog(), TS_ROUND_STR, PL.build());
            return 0;
        }
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
        else if (a == "--canzass-max")
            P.canzass_max = strtoull(need("--canzass-max"), 0, 10);
        else if (a == "--allow-no-ntl") {
#ifndef HAVE_NTL
            allow_no_ntl = true;
#endif
        }
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
        else if (a == "--sched-rho") {
            need("--sched-rho");
            fprintf(stderr, "note: --sched-rho is no longer used (the round-10 "
                    "schedule needs no geometric floor)\n");
        }
        else if (a == "--gpu-gcd-min-bits")
            gcd_min_bits_opt = strtoull(need("--gpu-gcd-min-bits"), 0, 10);
        else if (a == "--bench-gcd") bench_gcd = true;
        else if (a == "--retune") retune = true;
        else if (a == "--no-tune") no_tune = true;
#ifdef HAVE_NTL
        else if (a == "--gcd-finish-bits")     // tuning/testing knob
            hyb_ntl_finish_bits = strtoull(need("--gcd-finish-bits"), 0, 10);
        else if (a == "--gcd-full-apply")      // A/B: the pre-EC engine
            hyb_split_apply = false;
#endif
        else if (a == "--pend-max") { P.pend_max = atoi(need("--pend-max"));
                                      pend_set = true; }
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
        else if (a == "--gcd-base-bits")
            gcd_base_bits = strtoull(need("--gcd-base-bits"), 0, 10);
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
            for (int k = 1; k < ECI_N; k++)
                if (nm == eci_name[k]) site = k;
            if (site == ECI_NONE || n == 0) {
                fprintf(stderr, "bad --ec-fault %s (SITE:N; sites:",
                        v.c_str());
                for (int k = 1; k < ECI_N; k++)
                    fprintf(stderr, " %s", eci_name[k]);
                fprintf(stderr, ")\n");
                exit(1);
            }
            g_eci.arm(site, n, 0x5EEDULL);
        }
        else if (a == "-v") P.verbose++;
        else if (a == "-vv") P.verbose += 2;
        else if (PL.parse_opt(a, i, argc, argv)) {}
        else if (a.size() && a[0] == '-' && !isdigit(a[1])) {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            ts_usage(PL);
        }
        else pos.push_back(a);
    }
    if (gpu_gcd && !PL.has_gpu_gcd()) {
        fprintf(stderr, "ERROR: --gpu-gcd needs the GPU build (tsfactor); "
                "this is %s\n", PL.prog());
        exit(1);
    }
    if (gcd_threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        gcd_threads = hc > 2 ? (int)std::min(3u, hc - 1) : 1;
        int pm = P.pend_max;
        PL.default_threads(gcd_threads, pm);
        if (!pend_set) P.pend_max = pm;
    }
    PL.start();

#ifdef HAVE_NTL
    if (gcd_mt > 1 && !hyb_threads_ok()) {
        fprintf(stderr, "WARNING: --gcd-mt ignored: this NTL was built "
                        "without NTL_THREADS\n");
        gcd_mt = 1;
    }
    if (gcd_base_bits)
        hyb_base_bits = gcd_base_bits < 64 ? 64
                        : gcd_base_bits > 16320 ? 16320 : gcd_base_bits;
    if (gcd_mt <= 0 && PL.default_gcd_mt() > 0 && !do_selftest)
        gcd_mt = PL.default_gcd_mt();
    if (gcd_mt <= 0) {
        gcd_mt_auto = true;
        // automatic: the cores left over by the gcd workers and the scan
        // thread, at most 4 (a matrix step has 4 independent products;
        // more threads only help its 8-product sibling).  The selftest
        // stays on one thread unless told otherwise.
        unsigned hc = std::thread::hardware_concurrency();
        int per = hc > 1 ? (int)((hc - 1) / (unsigned)gcd_threads) : 1;
        gcd_mt = (do_selftest || !hyb_threads_ok()) ? 1
                 : per < 1 ? 1 : per > 4 ? 4 : per;
    }
    hyb_threads = gcd_mt > 64 ? 64 : gcd_mt;
#else
    (void)gcd_mt; (void)gcd_base_bits; (void)gcd_mt_auto;
#endif
    if (do_selftest) {
        int rc = 0;
        for (int rep = 0; rep < selftest_reps; rep++) {
            if (selftest_reps > 1)
                printf("===== selftest run %d of %d =====\n", rep + 1,
                       selftest_reps);
            int one = PL.selftest(gcd_threads);
            if (one) rc = one;
        }
        if (selftest_reps > 1)
            printf("===== %d runs complete: %s =====\n", selftest_reps,
                   rc ? "FAILURES SEEN" : "all runs PASS");
        return rc;
    }
    if (do_bench) {
        if (pos.empty()) ts_usage(PL);
        r = strtoull(pos[0].c_str(), 0, 10);
        if (pos.size() > 1) s_bench = strtoull(pos[1].c_str(), 0, 10);
        if (!(r & 1) || r < 5) { fprintf(stderr, "r must be odd, >= 5\n");
                                 exit(1); }
        if (!s_bench) s_bench = (r - 1) / 2;
        if (gpu_gcd) g_use_hybrid = true;
        return PL.bench(r, s_bench, bench_gcd, gpu_gcd, gcd_min_bits_opt,
                        ecg_xgcd, gcd_ntl);
    }
    if (pos.size() != 2) ts_usage(PL);
    r = strtoull(pos[0].c_str(), 0, 10);
    surv_path = pos[1];
    if (!(r & 1) || r < 5) { fprintf(stderr, "r must be odd, >= 5\n");
                             exit(1); }
    if (P.skip == (u64)-1) {
        fprintf(stderr, "--skip is REQUIRED (55 for the r=136279841 "
                        "pipeline after coarse_sieve);\nscanning starts at "
                        "skip+1 and factors of degree <= skip would be "
                        "MISSED.\n");
        exit(1);
    }
    if (P.zq && P.Zq) { fprintf(stderr, "-z and -Z are mutually "
                                        "exclusive\n"); exit(1); }
    if (!P.canzass_max) {
        u64 r3 = 0;
        while (r3 * r3 * r3 < r) r3++;
        P.canzass_max = 2 * r3 * r3;
    }
    if (retune && no_tune) {             // as coarse_sieve
        fprintf(stderr, "--retune conflicts with --no-tune\n");
        exit(1);
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
    // The half-gcd engine is the gcd backend, certified or not; plain NTL
    // only on request (--gcd-ntl) and for --ecg-dup's two NTL runs.
    g_engine_cpu = !gcd_ntl && !g_ec.gcd_dup;
#else
    g_ec.gcd_cert = false;
    (void)ecg_explicit; (void)ecg_xgcd; (void)gcd_ntl;
#endif
    fprintf(stderr, "%s %s (%s)\n", PL.prog(), TS_ROUND_STR, PL.build());
    fprintf(stderr, "GCD backend: %s\n", gcd_backend());
#ifdef HAVE_NTL
    if (hyb_threads > 1) {
        const bool engine = (g_ec.on && g_ec.gcd_cert && !hyb_cert_xgcd) ||
                            g_use_hybrid || g_engine_cpu;
        if (engine)
            fprintf(stderr, "gcd: %d threads inside each gcd (--gcd-mt %d%s)"
                    "\n", hyb_threads, hyb_threads,
                    gcd_mt_auto ? ", chosen automatically" : "");
        else if (!gcd_mt_auto)
            fprintf(stderr, "gcd: --gcd-mt has no effect on NTL's own gcd "
                    "(--gcd-ntl / --ecg-xgcd / --ecg-dup)\n");
    }
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
    if (PL.has_gpu_gcd() && gcd_threads == 1 && P.pend_max == 1)
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
            "rebuild -- `make` prints the NTL detection result\n"
            "and refuses silent no-NTL builds.  Override (degrees-only\n"
            "experiments): --allow-no-ntl.\n");
        exit(1);
    }
#endif
    // one process per output prefix
    if (!lock_prefix(out_prefix)) {
        fprintf(stderr, "ERROR: another %s is running with --out %s (%s.lock "
                "is held).\nUse a different --out for each concurrent run.\n",
                PL.prog(), out_prefix.c_str(), out_prefix.c_str());
        exit(1);
    }

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

    // Checkpoints of this list's pending s (round 10: one set per s).
    // They are processed first, in list order; the first one's tuning
    // record decides whether this run needs to tune.
    std::vector<u64> resumable;
    std::string kept_note;     // checkpoints of s not in the list
    TsTune ck_tune;
    memset(&ck_tune, 0, sizeof ck_tune);
    u64 ck_tune_s = 0;
    if (!no_ckpt) {
        std::set<u64> in_list(svs.begin(), svs.end());
        std::set<u64> seen;
        for (u64 sv : svs) {
            if (!seen.insert(sv).second) continue;
            std::string base = ckpt_base_for(out_prefix, r, sv);
            if (done_s.count(sv)) {        // finished, but the files stayed
                CkptHdr H;
                if (read_ckpt_hdr(base, H) != 0 ||
                    read_ckpt_hdr(base + ".1", H) != 0) {
                    remove_ckpts(base);
                    remove_fine(base);
                    fprintf(stderr, "note: removed the checkpoint of s=%"
                            PRIu64 " (already in %s)\n", sv,
                            res_path.c_str());
                }
                continue;
            }
            CkptHdr H;
            bool have = false;
            const char *sfx[3] = {"", ".1", ".2"};
            for (int g = 0; g < 3 && !have; g++) {
                int rc = read_ckpt_hdr(base + sfx[g], H);
                if (rc == 1 && H.r == r && H.s == sv && H.skip == P.skip)
                    have = true;
                else if (rc == 1 && g == 0)
                    fprintf(stderr, "note: %s is for skip %" PRIu64
                            ", not %" PRIu64 "; ignored\n", base.c_str(),
                            H.skip, P.skip);
                else if (rc == -1 && g == 0)
                    fprintf(stderr, "note: %s is unreadable or from an "
                            "earlier round (format 1); ignored\n",
                            base.c_str());
            }
            FineHdr FH;
            bool fine = read_fine_hdr(base, FH) && FH.r == r && FH.s == sv &&
                        FH.skip == P.skip;
            if (have || fine) resumable.push_back(sv);
            if (have && !ck_tune_s && H.tune.valid) {
                ck_tune = H.tune;
                ck_tune_s = sv;
            }
        }
        // set-aside checkpoints of s not in this list: say so (they are
        // kept; running a list that contains the s resumes it)
        std::set<u64> others;
        for (u64 sv : list_ckpt_s(out_prefix, r))
            if (!in_list.count(sv)) others.insert(sv);
        if (!others.empty()) {
            char b[128];
            snprintf(b, sizeof b, "%zu checkpoint(s) for s not in this list "
                     "are kept for later:", others.size());
            kept_note = b;
            int k = 0;
            for (u64 sv : others) {
                if (++k > 12) { kept_note += " ..."; break; }
                kept_note += " " + std::to_string(sv);
            }
            fprintf(stderr, "\n*** NOTE: %s ***\n\n", kept_note.c_str());
        }
        if (!resumable.empty())
            fprintf(stderr, "%zu survivor(s) of this list have checkpoints and "
                    "go first\n", resumable.size());
        // a checkpoint of rounds 1-9 (one file per output prefix): it cannot
        // be resumed, and it must not be mistaken for lost work silently
        for (const char *sfx : {".ckpt", ".ckpt.1", ".ckpt.2"}) {
            const std::string old = out_prefix + sfx;
            FILE *f = fopen(old.c_str(), "rb");
            if (!f) continue;
            u64 w[4] = {0, 0, 0, 0};       // magic, version, r, s
            bool rd = fread(w, sizeof(u64), 4, f) == 4;
            fclose(f);
            if (rd && w[0] == 0x31434653544ULL) {
                fprintf(stderr, "note: %s is a checkpoint of an earlier round "
                        "(r=%" PRIu64 " s=%" PRIu64 "), which this round "
                        "cannot resume.\n      Finish that survivor with "
                        "the earlier binary, or delete %s* to drop it.\n",
                        old.c_str(), w[2], w[3], (out_prefix + ".ckpt").c_str());
                break;
            }
        }
    }

    {
        size_t pend = 0;
        for (u64 sv : svs) if (!done_s.count(sv)) pend++;
        if (pend == 0) {
            fprintf(stderr, "done: nothing to do -- all %zu survivors of %s "
                    "are in %s\n", svs.size(), surv_path.c_str(),
                    res_path.c_str());
            unlock_prefix();
            return 0;
        }
    }

    std::unique_ptr<TsDev> Gp(PL.make_dev());
    TsDev &G = *Gp;
    G.init(r);
    if (gpu_gcd) PL.gcd_hook_init(G, gcd_min_bits_opt);

    // tuning: adopt, pin, default, or measure
    TsTune T;
    memset(&T, 0, sizeof T);
    const std::string fp = PL.fingerprint();
    {
        double tt0 = trin_now_s();
        if (PL.pinned(T)) {
            PL.apply(T);
            if (!want_m) {
                ts_measure_m(PL, G, T);
                PL.apply(T);
            } else {
                T.m = (u64)want_m;
            }
            fprintf(stderr, "tuning: pinned on the command line: %s\n",
                    PL.describe(T).c_str());
        } else if (!retune && ck_tune_s && ck_tune.valid &&
                   fp == ck_tune.fp && ck_tune.m >= 2 && ck_tune.m <= 64) {
            T = ck_tune;
            PL.apply(T);
            fprintf(stderr, "tuning: from the checkpoint of s=%" PRIu64
                    " (made by this binary on this device): %s\n",
                    ck_tune_s, PL.describe(T).c_str());
        } else if (no_tune) {
            PL.defaults(T);
            PL.apply(T);
            if (!want_m) ts_measure_m(PL, G, T);
            else T.m = (u64)want_m;
            fprintf(stderr, "tuning: compiled-in defaults (--no-tune): %s\n",
                    PL.describe(T).c_str());
        } else {
            if (ck_tune_s && !retune)
                fprintf(stderr, "tuning: the checkpoint of s=%" PRIu64
                        " was tuned by \"%s\"; this is \"%s\": tuning "
                        "again\n", ck_tune_s, ck_tune.fp, fp.c_str());
            else if (retune)
                fprintf(stderr, "tuning: --retune\n");
            PL.tune(G, T, P.verbose);
            if (want_m) T.m = (u64)want_m;
            fprintf(stderr, "tuning: %s (%.1f s)\n", PL.describe(T).c_str(),
                    trin_now_s() - tt0);
        }
        snprintf(T.fp, sizeof T.fp, "%s", fp.c_str());
        T.valid = 1;
    }
    int m = want_m ? want_m : (int)T.m;
    if (m < 2 || m > 64) { fprintf(stderr, "need 2 <= m <= 64\n"); exit(1); }
    G.set_m(m);
    P.tune = T;
    PL.mem_report(G, P, gpu_gcd);
    if (P.q0 <= 0) P.q0 = (long)((600 + m - 1) / m);
    if (P.sched_opt) {
        if (P.sched_G)
            fprintf(stderr, "interval schedule: expected-cost optimum (G=%"
                    PRIu64 " degrees, fixed by --sched-G)\n", P.sched_G);
        else
            fprintf(stderr, "interval schedule: expected-cost optimum, G = gcd "
                    "time / time per degree, measured as the run goes (%"
                    PRIu64 " until the first gcd or a checkpoint's value; "
                    "--sched-G N fixes it)\n", P.sched_G0);
    } else {
        fprintf(stderr, "interval schedule: linear (q += %ld)\n", P.q0);
    }
    fprintf(stderr, "r=%" PRIu64 " skip=%" PRIu64 " m=%d gcd-threads=%d "
            "pend-max=%d%s%s\n", r, P.skip, m, gcd_threads, P.pend_max,
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

    // the s with checkpoints first, then the rest, each in list order
    {
        std::vector<u64> ord = resumable;
        std::set<u64> rs(resumable.begin(), resumable.end());
        for (u64 sv : svs) if (!rs.count(sv)) ord.push_back(sv);
        svs.swap(ord);
    }

    bool interrupted = false, ec_fatal = false;
    u64 ec_fatal_s = 0;
    int last_mb_note = 0;
    for (u64 sv : svs) {
        if (done_s.count(sv)) continue;
        if (g_stop) { interrupted = true; break; }
        if (P.verbose) printf("== s=%" PRIu64 " ==\n", sv);
        SParams Ps = P;
        Ps.ckpt_base = no_ckpt ? std::string()
                               : ckpt_base_for(out_prefix, r, sv);
        ResumeState RS;
        int m_s = m;
        if (!no_ckpt) {
            RS.valid = load_ckpt(Ps.ckpt_base, RS.H, RS.h, RS.a) &&
                       RS.H.r == r && RS.H.s == sv && RS.H.skip == P.skip &&
                       RS.H.nw == (u64)G.nw && RS.H.m >= 2 && RS.H.m <= 64;
            FineHdr FH;
            if (RS.valid) m_s = (int)RS.H.m;
            else if (read_fine_hdr(Ps.ckpt_base, FH) && FH.r == r &&
                     FH.s == sv && FH.skip == P.skip && FH.nw == (u64)G.nw)
                m_s = (int)FH.m;
            if (m_s != m)
                fprintf(stderr, "note: s=%" PRIu64 " resumes with the "
                        "checkpoint's m=%d (this run's m is %d)\n", sv, m_s,
                        m);
        }
        if (m_s == m && !RS.valid && !want_m) {
            int mb = maxd_block_size(P, r, sv, m);
            if (mb != m) {
                if (mb != last_mb_note) {
                    u64 k0 = P.skip + 1, L = P.maxd - k0 + 1;
                    fprintf(stderr, "block size %d instead of %d for the scan "
                            "%" PRIu64 "..%" PRIu64 " (one interval): %" PRIu64
                            " blocks end at --maxd exactly (m=%d: at %" PRIu64
                            ")\n", mb, m, k0, P.maxd, (L + mb - 1) / mb, m,
                            k0 + (L + m - 1) / m * m - 1);
                    last_mb_note = mb;
                }
                m_s = mb;
            }
        }
        G.set_m(m_s);
        ScanOut o = scan_one_s(G, pool, Ps, r, sv, gseq,
                               RS.valid ? &RS : nullptr);
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
        case ScanOut::PRIMITIVE: {
            const bool prim = mersenne_exponent(r);
            const char *w = prim ? "primitive" : "irreducible";
            fprintf(rf, "%" PRIu64 " %s\n", sv, w);
            printf("%" PRIu64 " %s\n", sv, w);
            fprintf(stderr, "\n*** %s TRINOMIAL CANDIDATE: s=%" PRIu64
                    " -- verify independently (irred)! ***\n\n",
                    prim ? "PRIMITIVE" : "IRREDUCIBLE", sv);
            break;
        }
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
        done_s.insert(sv);
        // This s is on record.  Its checkpoints (up to three generations of
        // ~0.5 GB each at r=136279841) are of no further use.
        remove_ckpts(Ps.ckpt_base);
        remove_fine(Ps.ckpt_base);
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
            "same verified state.\nThat is not a cosmic ray; the %s, "
            "RAM or power is unreliable right now.\nNo result was "
            "written for this s (it will be redone on the next run).\n"
            "Check temperatures and any overclock/undervolt, run "
            "`%s --selftest`,\nand see %s.\n", ec_fatal_s,
            g_ec.max_retry + 1,
            strcmp(G.kind(), "GPU") == 0 ? "GPU (or its driver)" : "CPU",
            PL.prog(), g_ec.log_path.c_str());
        fflush(nullptr);
        pool.stop_nowait();
        unlock_prefix();
        std::_Exit(3);
    }
    if (interrupted) {
        fprintf(stderr, "interrupted: %" PRIu64 " found, %" PRIu64
                " u; exiting (in-flight GCDs abandoned)\n", n_found, n_u);
        if (!kept_note.empty())
            fprintf(stderr, "NOTE: %s\n", kept_note.c_str());
        fflush(nullptr);
        pool.stop_nowait();
        unlock_prefix();
        std::_Exit(0);             // workers may hold NTL state; skip
    }                              // static teardown entirely
    pool.stop();          // joins workers: nothing can be in the hook now
    PL.gcd_hook_fini();
    G.fini();
    fprintf(stderr, "done: %" PRIu64 " found, %" PRIu64 " u\n", n_found,
            n_u);
    if (!kept_note.empty())
        fprintf(stderr, "NOTE: %s\n", kept_note.c_str());
    unlock_prefix();
    return 0;
}

#endif // TS_CORE_H

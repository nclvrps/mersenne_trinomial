// trinomial_scan.cu
// Deep scan of coarse-sieve survivors: for each s in the survivors file
// (trinomials x^r + x^s + 1 over GF(2) with no irreducible factor of
// degree <= k0), find the least factor degree d in (k0, maxd] and the
// lexicographically least degree-d factor mask, or report "u" if no
// factor with degree <= maxd exists.  Implements Part C of
// deep_scan_handoff.md.
//
// Method: Frobenius iterates F_k = x^(2^k) mod T_s on the GPU (bit-
// spread squaring + two-pass trinomial fold), interval products
// ACC = prod (F_k + x) mod T_s via the Cantor additive-FFT multiplier,
// CPU GCDs (NTL HalfGCD; the same machinery as factor.cpp) from a
// thread pool, fully overlapped: the GPU keeps scanning speculatively
// while GCD verdicts are pending, and verdicts are processed strictly
// in interval order so out-of-order pool completions cannot misreport
// the FIRST hit.  Resolving a hit needs NO further full-size GCDs and
// no GPU work: the interval gcd g is a small polynomial whose
// irreducible factors ALL have degrees inside the hit interval (the C2
// theorem plus induction over the previously cleared intervals), so a
// single factorization of g yields both the least factor degree and
// its lexicographically least mask.
//
// Build (NTL auto-detected by the Makefile; REQUIRED for large r):
//     make trinomial_scan
// Run:
//     ./trinomial_scan <r> <survivors.txt> <k0> <maxd> <out_prefix>
//         [--interval Lmax] [--l0 L0] [--gcd-threads N] [--state FILE]
//         [--verbose 0..3]
// k0 MUST be the depth the coarse sieve actually reached (it prints
// this; 29 on an 8 GB card), or factors in the gap will be missed.
// Output <out_prefix>.results.txt, appended, resumable:
//     s d p<hex>    least-degree factor (mask lex-least at that degree)
//     s d g<file>   degree found, tie-break deferred (no NTL, d > 22):
//                   gcd polynomial dumped to <file> for offline EDF
//     s rA-B        pathological fallback: at least one factor with
//                   degree in [A, B], unresolved (interval gcd not
//                   capturable); rerun that range or hand to factor.cpp
//     s u           no factor of degree <= maxd (hand to factor.cpp)
// Selftest (CPU-oracle end-to-end at r = 4423, both backends):
//     ./trinomial_scan --selftest

#include "gf2_trinomial_cuda.h"
#include "gf2_gcd.h"
#include <cinttypes>
#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>

// ---------------------------------------------------------------------------
// CPU GCD thread pool.  Jobs are (id, ACC words); the modulus T is set
// per s (only while the pool is idle).  Results are retrieved by id;
// the caller enforces in-order handling.
struct GcdRes {
    u64 gdeg = 0;
    std::vector<u64> g;
};

class GcdPool {
    std::mutex mu;
    std::condition_variable cv_job, cv_res;
    struct Job {
        u64 id;
        std::vector<u64> acc;
        std::shared_ptr<const std::vector<u64>> mod;
    };
    std::deque<Job> jobs;
    std::map<u64, GcdRes> results;
    std::set<u64> outstanding, discard;
    bool stopf = false;
    std::vector<std::thread> ths;

    void worker() {
        for (;;) {
            std::unique_lock<std::mutex> lk(mu);
            cv_job.wait(lk, [&] { return stopf || !jobs.empty(); });
            if (jobs.empty()) { if (stopf) return; continue; }
            Job j = std::move(jobs.front());
            jobs.pop_front();
            lk.unlock();
            GcdRes r;
            r.gdeg = poly_gcd(j.acc.data(), j.acc.size(), j.mod->data(),
                              j.mod->size(), &r.g);
            lk.lock();
            outstanding.erase(j.id);
            if (discard.erase(j.id) == 0) results[j.id] = std::move(r);
            cv_res.notify_all();
        }
    }

public:
    void start(int n) {
        for (int i = 0; i < n; i++) ths.emplace_back([this] { worker(); });
    }
    // each job carries shared ownership of its modulus, so a new s can
    // start immediately while stale jobs from the previous s finish
    void submit(u64 id, std::vector<u64> acc,
                std::shared_ptr<const std::vector<u64>> mod) {
        std::lock_guard<std::mutex> lk(mu);
        outstanding.insert(id);
        jobs.push_back(Job{id, std::move(acc), std::move(mod)});
        cv_job.notify_one();
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
        GcdRes r = std::move(results[id]);
        results.erase(id);
        return r;
    }
    // NON-BLOCKING: mark everything still in flight or unread as stale;
    // workers drop stale results on completion.  (The old drain-based
    // version blocked for up to pending * one-full-GCD after each find.)
    void forget_all_pending() {
        std::lock_guard<std::mutex> lk(mu);
        for (u64 id : outstanding) discard.insert(id);
        results.clear();
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stopf = true;
        }
        cv_job.notify_all();
        for (auto &t : ths) t.join();
        ths.clear();
    }
};

// ---------------------------------------------------------------------------
// resume state: F at a verified frontier of the in-progress s
static void save_state(const char *path, u64 r, u64 s, u64 k,
                       const std::vector<u64> &F) {
    std::string tmp = std::string(path) + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    u64 hdr[5] = {0x4E435354ULL, r, s, k, (u64)F.size()};
    fwrite(hdr, 8, 5, f);
    fwrite(F.data(), 8, F.size(), f);
    fclose(f);
    rename(tmp.c_str(), path);
}
static bool load_state(const char *path, u64 r, u64 s, u64 &k,
                       std::vector<u64> &F, size_t nw) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    u64 hdr[5];
    bool ok = fread(hdr, 8, 5, f) == 5 && hdr[0] == 0x4E435354ULL &&
              hdr[1] == r && hdr[2] == s && hdr[4] == (u64)nw;
    if (ok) {
        F.resize(nw);
        ok = fread(F.data(), 8, nw, f) == nw;
    }
    fclose(f);
    if (ok) k = hdr[3];
    return ok;
}

// ---------------------------------------------------------------------------
struct ScanParams {
    u64 k0 = 0, maxd = 0;
    u64 L0 = 64, Lmax = 4096;
    int verbose = 1;
    const char *state_path = nullptr;
};

struct ScanResult {
    bool found = false;
    u64 d = 0;
    std::vector<u64> mask;   // valid iff mask_ok
    bool mask_ok = false;
    std::vector<u64> g;      // interval gcd; dump when !mask_ok
    u64 gdeg = 0;
    bool range_only = false; // pathological: factor exists in [rlo, rhi]
    u64 rlo = 0, rhi = 0;    // but g was not capturable for resolution
};

// Resolving a hit no longer touches T or the GPU at all: the interval
// gcd g is small and every irreducible factor of g has degree inside
// the hit interval (deep_scan_handoff.md C2), so one factorization of
// g yields the least degree AND its lexicographically least mask.
// This fallback (used without NTL, i.e. selftests / small r) finds the
// least degree by squaring mod g and the mask by enumeration when the
// degree is small enough; with NTL, factor_min_degree_ntl does both
// via a single CanZass.
static void poly_rem_inplace(std::vector<u64> &A, const std::vector<u64> &g) {
    int64_t dg = poly_deg_v(g), da = poly_deg_v(A);
    while (da >= dg && da >= 0) {
        poly_xor_shift(A, g, dg, (u64)(da - dg));
        da = poly_deg_v(A);
    }
}

static bool factor_min_degree_fallback(const std::vector<u64> &g, u64 kmax,
                                       u64 &min_deg, std::vector<u64> &mask,
                                       bool &mask_ok) {
    min_deg = 0;
    mask_ok = false;
    int64_t dg = poly_deg_v(g);
    if (dg <= 0) return false;
    size_t gw = (size_t)(dg >> 6) + 1;
    std::vector<u64> h(gw, 0);
    h[0] = 2;                                   // x mod g
    for (u64 k = 1; k <= kmax; k++) {
        std::vector<u64> sq(mul_out_words((u64)dg, (u64)dg));
        gf2x_naive_mul(h.data(), (u64)dg, h.data(), (u64)dg, sq.data());
        poly_rem_inplace(sq, g);
        sq.resize(gw);
        h = sq;
        std::vector<u64> hx = h;
        hx[0] ^= 2ULL;                          // h + x
        std::vector<u64> gg;
        u64 gd = poly_gcd_naive(hx.data(), hx.size(), g.data(), g.size(), &gg);
        if (gd > 0) {
            min_deg = k;
            mask_ok = edf_least_enum(gg, k, mask);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// scan one s to first factor or maxd
static ScanResult scan_one_s(TrinGPU &G, GcdPool &pool, const ScanParams &P,
                             u64 r, u64 s, std::vector<u64 *> &slot,
                             u64 &gseq) {
    ScanResult res;
    G.set_s(s);
    size_t nw = G.nw, nb = nw * sizeof(u64);

    auto T = std::make_shared<std::vector<u64>>(bits_to_words(r + 1), 0);
    (*T)[0] |= 1;
    (*T)[s >> 6] |= 1ULL << (s & 63);
    (*T)[r >> 6] |= 1ULL << (r & 63);

    std::vector<int> frees;
    for (int i = 0; i < (int)slot.size(); i++) frees.push_back(i);
    auto alloc_slot = [&]() { int i = frees.back(); frees.pop_back(); return i; };

    struct Pend { u64 seq, a, b; int ck; };
    std::deque<Pend> pending;

    // ramp (or resume at a verified frontier)
    u64 frontier = P.k0;
    std::vector<u64> hostF(nw);
    bool resumed = P.state_path && load_state(P.state_path, r, s, frontier, hostF, nw);
    int blocks, threads;
    CudaFFT::launch_dims(nw, blocks, threads);
    if (resumed) {
        if (frontier > P.maxd) frontier = P.maxd;
        CUCHK(cudaMemcpy(G.dA, hostF.data(), nb, cudaMemcpyHostToDevice));
        if (P.verbose >= 1)
            fprintf(stderr, "  s=%" PRIu64 ": resumed at k=%" PRIu64 "\n", s,
                    frontier);
    } else {
        k_set_x<<<blocks, threads>>>(G.dA, nw);
        for (u64 k = 0; k < P.k0; k++) G.square_A();
    }
    int cur = alloc_slot();
    CUCHK(cudaMemcpy(slot[cur], G.dA, nb, cudaMemcpyDeviceToDevice));
    G.reset_acc();
    u64 k = frontier, iv_start = frontier, L = P.L0;
    u64 &seqn = gseq;                    // ids unique across all s values
    double last_save = trin_now_s();
    std::vector<u64> hostA(nw);

    auto do_found = [&](const Pend &pv, GcdRes &R) {
        res.found = true;
        res.gdeg = R.gdeg;
        u64 dmin = 0;
        std::vector<u64> mask;
        bool mask_ok = false, resolved = false;
        if (!R.g.empty()) {
#ifdef HAVE_NTL
            resolved = factor_min_degree_ntl(R.g.data(), R.g.size(), dmin, mask);
            mask_ok = resolved;
#else
            resolved = factor_min_degree_fallback(R.g, pv.b, dmin, mask, mask_ok);
#endif
        }
        if (resolved) {
            res.d = dmin;
            res.mask = mask;
            res.mask_ok = mask_ok;
            res.g = std::move(R.g);
            if (!(res.d > pv.a && res.d <= pv.b))
                fprintf(stderr,
                        "  s=%" PRIu64 ": WARNING least degree %" PRIu64
                        " outside hit interval (%" PRIu64 ",%" PRIu64
                        "] -- input not a genuine survivor?\n",
                        s, res.d, pv.a, pv.b);
            else if (P.verbose >= 2)
                fprintf(stderr, "  s=%" PRIu64 ": hit in (%" PRIu64
                        ",%" PRIu64 "], deg(g)=%" PRIu64 ", resolved d=%"
                        PRIu64 " by factoring g\n",
                        s, pv.a, pv.b, res.gdeg, res.d);
        } else {
            // pathological: interval gcd not capturable/factorable --
            // report the known degree range for offline processing
            res.range_only = true;
            res.rlo = pv.a + 1;
            res.rhi = pv.b;
            res.g = std::move(R.g);
        }
        pool.forget_all_pending();       // non-blocking; stale jobs dropped
    };

    // handle the HEAD verdict (already popped); true if factor found
    auto head_verdict = [&](const Pend &pv, GcdRes &R) -> bool {
        if (R.gdeg > 0) {
            do_found(pv, R);
            return true;
        }
        frontier = pv.b;
        frees.push_back(pv.ck);
        if (P.verbose >= 2)
            fprintf(stderr, "  s=%" PRIu64 ": (%" PRIu64 ",%" PRIu64 "] clear\n",
                    s, pv.a, pv.b);
        if (P.state_path && trin_now_s() - last_save > 30.0) {
            int nslot = pending.empty() ? cur : pending.front().ck;
            CUCHK(cudaMemcpy(hostF.data(), slot[nslot], nb,
                             cudaMemcpyDeviceToHost));
            save_state(P.state_path, r, s, frontier, hostF);
            last_save = trin_now_s();
        }
        return false;
    };

    for (;;) {
        // consume any completed verdicts, strictly in interval order
        while (!pending.empty()) {
            GcdRes R;
            if (!pool.try_get(pending.front().seq, R)) break;
            Pend pv = pending.front();
            pending.pop_front();
            if (head_verdict(pv, R)) return res;
        }
        if (k == P.maxd || (int)pending.size() >= (int)slot.size() - 2) {
            if (k == P.maxd && pending.empty()) return res;    // "u"
            GcdRes R = pool.wait_get(pending.front().seq);     // block on head
            Pend pv = pending.front();
            pending.pop_front();
            if (head_verdict(pv, R)) return res;
            continue;
        }
        // advance one degree
        k++;
        G.square_A();
        G.accumulate();
        if (k - iv_start >= L || k == P.maxd) {
            CUCHK(cudaMemcpy(hostA.data(), G.dACC, nb, cudaMemcpyDeviceToHost));
            pool.submit(seqn, hostA, T);
            pending.push_back({seqn, iv_start, k, cur});
            seqn++;
            cur = alloc_slot();
            CUCHK(cudaMemcpy(slot[cur], G.dA, nb, cudaMemcpyDeviceToDevice));
            G.reset_acc();
            if (P.verbose >= 3)
                fprintf(stderr, "  s=%" PRIu64 ": submitted (%" PRIu64
                        ",%" PRIu64 "]\n", s, iv_start, k);
            iv_start = k;
            L = std::min(2 * L, P.Lmax);
        }
    }
}

// ---------------------------------------------------------------------------
// independent CPU oracle for the selftest: per-degree naive GCDs
struct OracleOut {
    bool survivor = false;
    bool found = false;
    u64 d = 0, gdeg = 0;
    std::vector<u64> g, mask;
    bool mask_ok = false;
};

static OracleOut oracle_scan(u64 r, u64 s, u64 k0, u64 maxd) {
    OracleOut o;
    size_t nw = bits_to_words(r);
    std::vector<u64> T(bits_to_words(r + 1), 0);
    T[0] |= 1;
    T[s >> 6] |= 1ULL << (s & 63);
    T[r >> 6] |= 1ULL << (r & 63);
    std::vector<u64> F(nw, 0);
    F[0] = 2;
    for (u64 k = 1; k <= maxd; k++) {
        cpu_square_mod(F, r, s);
        std::vector<u64> Fx = F;
        Fx[0] ^= 2ULL;
        std::vector<u64> g;
        u64 gd = poly_gcd_naive(Fx.data(), Fx.size(), T.data(), T.size(), &g);
        if (gd > 0) {
            if (k <= k0) return o;                 // not a survivor
            o.survivor = o.found = true;
            o.d = k;
            o.gdeg = gd;
            o.g = g;
            if (gd == k) { o.mask = g; o.mask_ok = true; }
            else o.mask_ok = edf_least_enum(g, k, o.mask);
            return o;
        }
    }
    o.survivor = true;                             // "u"
    return o;
}

static bool poly_eq(const std::vector<u64> &a, const std::vector<u64> &b) {
    return !poly_less(a, b) && !poly_less(b, a);
}

static u64 st_rng = 0x243F6A8885A308D3ULL;
static u64 st_rnd() {
    u64 x = st_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return st_rng = x;
}
static std::vector<u64> st_randpoly(u64 degree) {
    std::vector<u64> v(bits_to_words(degree + 1), 0);
    for (auto &w : v) w = st_rnd();
    v[degree >> 6] &= (~0ULL) >> (63 - (degree & 63));
    v[degree >> 6] |= 1ULL << (degree & 63);       // exact degree
    return v;
}

static int run_selftest(int gcd_threads) {
    int bad = 0;

    // (1) GCD backends: naive self-consistency, NTL cross-check
    {
        bool ok = true;
        for (int t = 0; t < 20 && ok; t++) {
            u64 gd = 5 + st_rnd() % 400, ud = 1 + st_rnd() % 1600,
                vd = 1 + st_rnd() % 1600;
            std::vector<u64> g = st_randpoly(gd), u = st_randpoly(ud),
                             v = st_randpoly(vd);
            std::vector<u64> a(mul_out_words(gd + 1, ud + 1)),
                             b(mul_out_words(gd + 1, vd + 1));
            gf2x_naive_mul(g.data(), gd + 1, u.data(), ud + 1, a.data());
            gf2x_naive_mul(g.data(), gd + 1, v.data(), vd + 1, b.data());
            std::vector<u64> G1;
            u64 d1 = poly_gcd_naive(a.data(), a.size(), b.data(), b.size(), &G1);
            if (d1 < gd || !poly_divides(g, G1)) ok = false;
#ifdef HAVE_NTL
            std::vector<u64> G2;
            u64 d2 = poly_gcd_ntl(a.data(), a.size(), b.data(), b.size(), &G2,
                                  GCD_KEEP_MAX_BITS);
            if (d2 != d1 || !poly_eq(G1, G2)) ok = false;
#endif
        }
#ifdef HAVE_NTL
        printf("gcd backends: naive consistency + NTL cross-check    %s\n",
               ok ? "PASS" : "FAIL");
#else
        printf("gcd backend: naive consistency (NTL not compiled)    %s\n",
               ok ? "PASS" : "FAIL");
#endif
        if (!ok) bad++;
    }

    // (2) end-to-end vs oracle at r = 4423
    {
        const u64 r = 4423, k0 = 5, maxd = 200;
        std::vector<u64> surv;
        std::map<u64, OracleOut> want;
        for (u64 s = 2; s <= 350; s++) {
            OracleOut o = oracle_scan(r, s, k0, maxd);
            if (o.survivor) { surv.push_back(s); want[s] = std::move(o); }
        }
        u64 nfound = 0, nu = 0, nties = 0;
        for (auto &kv : want)
            if (kv.second.found) { nfound++; if (kv.second.gdeg != kv.second.d) nties++; }
            else nu++;
        printf("oracle r=%" PRIu64 " k0=%" PRIu64 " maxd=%" PRIu64
               ": %zu survivors (%" PRIu64 " found, %" PRIu64 " u, %" PRIu64
               " multi-factor ties)\n", r, k0, maxd, surv.size(), nfound, nu,
               nties);

        TrinGPU G;
        G.init(r);
        std::vector<u64 *> slot(10);
        for (auto &p : slot) CUCHK(cudaMalloc(&p, G.nw * sizeof(u64)));
        GcdPool pool;
        pool.start(gcd_threads);
        u64 gseq = 0;

        struct Policy { u64 L0, Lmax; const char *name; };
        Policy pols[2] = {{32, 4096, "default intervals"},
                          {8, 8, "stress intervals L=8"}};
        for (auto &pol : pols) {
            ScanParams P;
            P.k0 = k0; P.maxd = maxd; P.L0 = pol.L0; P.Lmax = pol.Lmax;
            P.verbose = 0;
            u64 mism = 0;
            for (u64 s : surv) {
                ScanResult res = scan_one_s(G, pool, P, r, s, slot, gseq);
                const OracleOut &o = want[s];
                bool ok = (res.found == o.found);
                if (ok && res.found) {
                    ok = (res.d == o.d);
                    if (ok && o.mask_ok && res.mask_ok)
                        ok = poly_eq(res.mask, o.mask);
                    else if (ok && res.mask_ok)
                        ok = ((u64)poly_deg_v(res.mask) == res.d &&
                              poly_divides(res.mask, o.g));
                }
                if (!ok && ++mism <= 5)
                    fprintf(stderr, "MISMATCH s=%" PRIu64 " scan(found=%d d=%"
                            PRIu64 ") oracle(found=%d d=%" PRIu64 ")\n", s,
                            (int)res.found, res.d, (int)o.found, o.d);
            }
            printf("scan vs oracle, %-22s                %s (%" PRIu64
                   " mismatches)\n", pol.name, mism ? "FAIL" : "PASS", mism);
            if (mism) bad++;
        }
        pool.stop();
        for (auto &p : slot) cudaFree(p);
        G.fini();
    }

    printf(bad ? "\nSCAN SELFTEST: %d FAILURES\n" : "\nSCAN SELFTEST: ALL PASS\n",
           bad);
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest"))
        return run_selftest(2);

    if (argc < 6) {
        fprintf(stderr,
            "usage: %s <r> <survivors.txt> <k0> <maxd> <out_prefix>\n"
            "          [--interval Lmax] [--l0 L0] [--gcd-threads N]\n"
            "          [--state FILE]\n"
            "          [--verbose 0..3]\n"
            "       %s --selftest\n"
            "k0 must be the depth the coarse sieve actually reached.\n",
            argv[0], argv[0]);
        return 1;
    }
    u64 r = strtoull(argv[1], 0, 10);
    const char *survf = argv[2];
    ScanParams P;
    P.k0 = strtoull(argv[3], 0, 10);
    P.maxd = strtoull(argv[4], 0, 10);
    std::string prefix = argv[5];
    int gcd_threads = 2;
    std::string state_path;
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--interval") && i + 1 < argc)
            P.Lmax = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--l0") && i + 1 < argc)
            P.L0 = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--gcd-threads") && i + 1 < argc)
            gcd_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--state") && i + 1 < argc)
            state_path = argv[++i];
        else if (!strcmp(argv[i], "--verbose") && i + 1 < argc)
            P.verbose = atoi(argv[++i]);
        else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 1;
        }
    }
    if (P.Lmax < 1) P.Lmax = 1;
    if (P.L0 > P.Lmax) P.L0 = P.Lmax;
    if (!(P.k0 >= 1 && P.maxd > P.k0)) {
        fprintf(stderr, "need 1 <= k0 < maxd\n");
        return 1;
    }
#ifndef HAVE_NTL
    if (r > (1u << 20)) {
        fprintf(stderr,
                "r=%" PRIu64 " requires the NTL GCD backend (naive GCD is "
                "O(r^2)); rebuild with NTL installed -- see Makefile\n", r);
        return 1;
    }
#endif
    if (!state_path.empty()) P.state_path = state_path.c_str();

    // survivors
    std::vector<u64> surv;
    {
        FILE *f = fopen(survf, "r");
        if (!f) { perror(survf); return 1; }
        u64 s;
        while (fscanf(f, "%" SCNu64, &s) == 1) {
            if (s >= 1 && 2 * s < r) surv.push_back(s);
            else fprintf(stderr, "skipping s=%" PRIu64 " (need 1 <= s <= (r-1)/2)\n", s);
        }
        fclose(f);
    }

    // results (append; parse for resume)
    std::string rpath = prefix + ".results.txt";
    std::set<u64> done;
    {
        FILE *f = fopen(rpath.c_str(), "r");
        if (f) {
            char line[256];
            while (fgets(line, 250, f)) {
                u64 s;
                if (sscanf(line, "%" SCNu64, &s) == 1) done.insert(s);
            }
            fclose(f);
        }
    }
    FILE *fres = fopen(rpath.c_str(), "a");
    if (!fres) { perror(rpath.c_str()); return 1; }

    fprintf(stderr,
            "trinomial_scan r=%" PRIu64 " k0=%" PRIu64 " maxd=%" PRIu64
            ": %zu survivors (%zu already done), gcd backend: %s, "
            "%d gcd threads\n", r, P.k0, P.maxd, surv.size(), done.size(),
#ifdef HAVE_NTL
            "NTL",
#else
            "naive",
#endif
            gcd_threads);

    TrinGPU G;
    G.init(r);
    std::vector<u64 *> slot(10);
    for (auto &p : slot) CUCHK(cudaMalloc(&p, G.nw * sizeof(u64)));
    GcdPool pool;
    pool.start(gcd_threads);
    u64 gseq = 0;

    u64 nfound = 0, nu = 0;
    double t0 = trin_now_s();
    for (u64 s : surv) {
        if (done.count(s)) continue;
        double ts = trin_now_s();
        ScanResult res = scan_one_s(G, pool, P, r, s, slot, gseq);
        if (res.found) {
            nfound++;
            if (res.range_only) {
                fprintf(fres, "%" PRIu64 " r%" PRIu64 "-%" PRIu64 "\n", s,
                        res.rlo, res.rhi);
            } else if (res.mask_ok) {
                fprintf(fres, "%" PRIu64 " %" PRIu64 " p%s\n", s, res.d,
                        poly_hex(res.mask).c_str());
            } else {
                char gp[512];
                snprintf(gp, sizeof gp, "%s.gcd.s%" PRIu64 ".bin",
                         prefix.c_str(), s);
                FILE *fg = fopen(gp, "wb");
                if (fg) {
                    fwrite(res.g.data(), 8, res.g.size(), fg);
                    fclose(fg);
                }
                fprintf(fres, "%" PRIu64 " %" PRIu64 " g%s\n", s, res.d, gp);
            }
        } else {
            nu++;
            fprintf(fres, "%" PRIu64 " u\n", s);
        }
        fflush(fres);
        if (P.state_path) remove(P.state_path);
        if (P.verbose >= 1) {
            if (res.found && res.range_only)
                fprintf(stderr, "  s=%" PRIu64 ": factor in [%" PRIu64
                        ",%" PRIu64 "], unresolved (%.1fs)\n", s, res.rlo,
                        res.rhi, trin_now_s() - ts);
            else if (res.found)
                fprintf(stderr, "  s=%" PRIu64 ": d=%" PRIu64 " (%.1fs)\n", s,
                        res.d, trin_now_s() - ts);
            else
                fprintf(stderr, "  s=%" PRIu64 ": u (no factor <= %" PRIu64
                        ", %.1fs)\n", s, P.maxd, trin_now_s() - ts);
        }
    }
    fprintf(stderr, "done: %" PRIu64 " found, %" PRIu64 " u, %.1fs total\n",
            nfound, nu, trin_now_s() - t0);
    fclose(fres);
    pool.stop();
    for (auto &p : slot) cudaFree(p);
    G.fini();
    return 0;
}

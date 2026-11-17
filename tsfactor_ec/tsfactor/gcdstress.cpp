// gcdstress.cpp -- randomized cross-check of every CPU gcd path in
// gf2_gcd_ntl.cpp against NTL's GCD.  No GPU needed.
//
//     hybrid HGCD, split apply     (the engine, uncertified)
//     hybrid HGCD, full apply      (the engine before round EC)
//     CERTIFIED, NTL XGCD          (--ecg-xgcd)
//     CERTIFIED, chain             (tsfactor's default gcd)
//
// Each must return the same gcd (degree AND polynomial) as NTL on every
// input, and the certified ones must report GCD_CERT_OK.  The inputs are
// chosen to reach the engine's corner cases: quotients of every degree
// (built from a prescribed quotient sequence, including long runs of
// degree-1 quotients and occasional huge ones that straddle the half-gcd
// split point), planted common factors from tiny to almost everything,
// b | a, a == b, zero operands, and the production shape (a residue against
// a trinomial).  The NTL-finish threshold is varied from 64 bits up so the
// engine, not NTL, does the work; a third of the cases run the products of
// each matrix operation on three threads (tsfactor --gcd-mt).
//
//   make gcdstress && ./gcdstress            (400 cases up to 120000 bits:
//                                             about a minute)
//   ./gcdstress 1500 200000 7                (cases, max bits, seed)
//
// Exit status 0 iff every case agrees.

#include "gf2_gcd.h"
#define GF2_EC_HOST_ONLY
#include "gf2_ec.h"
#include <NTL/GF2X.h>
#include <cstdlib>

using NTL::GF2X;

static u64 rng_state = 1;
static u64 rnd() {
    rng_state += 0x9E3779B97F4A7C15ULL;
    u64 z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static u64 rnd_below(u64 n) { return n ? rnd() % n : 0; }

// random polynomial of EXACT degree d (d < 0: zero)
static GF2X rpoly(long d) {
    GF2X f;
    if (d < 0) return f;
    size_t nw = (size_t)d / 64 + 1;
    std::vector<u64> w(nw);
    for (auto &x : w) x = rnd();
    unsigned top = (unsigned)(d & 63);
    if (top != 63) w[nw - 1] &= (1ULL << (top + 1)) - 1;
    w[nw - 1] |= 1ULL << top;
    NTL::GF2XFromBytes(f, (const unsigned char *)w.data(), (long)(nw * 8));
    return f;
}
static std::vector<u64> words(const GF2X &f) {
    long nb = NTL::NumBytes(f);
    std::vector<u64> out(nb <= 0 ? 1 : ((size_t)nb + 7) / 8, 0);
    if (nb > 0) NTL::BytesFromGF2X((unsigned char *)out.data(), f, nb);
    return out;
}
static void trim(std::vector<u64> &v) {
    while (!v.empty() && v.back() == 0) v.pop_back();
}

// (a, b) with a prescribed quotient sequence on top of gcd g
static void from_quotients(GF2X &a, GF2X &b, const GF2X &g, long target_bits,
                           int flavour) {
    a = g;
    clear(b);
    while (deg(a) < target_bits) {
        long qd;
        switch (flavour) {
        case 0: qd = 1; break;                           // all degree 1
        case 1: qd = 1 + (long)rnd_below(3); break;      // typical
        case 2: qd = rnd_below(40) ? 1 + (long)rnd_below(2)
                                   : 50 + (long)rnd_below(3000); break;
        default: qd = rnd_below(6) ? 1
                    : 1 + (long)rnd_below((u64)target_bits / 3 + 1); break;
        }
        GF2X q = rpoly(qd), t;
        mul(t, q, a);
        add(t, t, b);
        b = a;
        a = t;
    }
}

int main(int argc, char **argv) {
    long cases = argc > 1 ? atol(argv[1]) : 400;
    long maxbits = argc > 2 ? atol(argv[2]) : 120000;
    rng_state = argc > 3 ? strtoull(argv[3], 0, 10) : 20261116;
    if (cases < 1 || maxbits < 256) {
        fprintf(stderr, "usage: gcdstress [cases [maxbits [seed]]]\n");
        return 2;
    }
    const u64 finish_choices[] = {64, 300, 4096, 20000, 65536};
    long bad = 0, nontrivial = 0, certs = 0, declined = 0, threaded = 0;
    u64 fb_before = 0, g0, c0;
    hyb_reset_stats();
    hyb_get_stats(g0, c0, fb_before);
    for (long it = 0; it < cases; it++) {
        long n = 200 + (long)rnd_below((u64)maxbits - 199);
        int kind = (int)rnd_below(12);
        GF2X a, b, g;
        set(g);                                  // g = 1
        if (rnd_below(3) == 0) {                 // plant a common factor
            long gd = rnd_below(4) ? 1 + (long)rnd_below(200)
                                   : 1 + (long)rnd_below((u64)n);
            g = rpoly(gd);
        }
        switch (kind) {
        case 0: case 1: case 2: {                // two random multiples of g
            GF2X u = rpoly(n), v = rpoly((long)rnd_below((u64)n + 1));
            mul(a, u, g);
            mul(b, v, g);
            break;
        }
        case 3: from_quotients(a, b, g, n, 0); break;
        case 4: from_quotients(a, b, g, n, 1); break;
        case 5: case 6: from_quotients(a, b, g, n, 2); break;
        case 7: from_quotients(a, b, g, n, 3); break;
        case 8: {                                // b divides a
            b = rpoly(n / 2 + 1);
            mul(a, b, rpoly(n / 2));
            break;
        }
        case 9: {                                // equal, or one zero
            a = rpoly(n);
            if (rnd_below(2)) b = a;
            break;
        }
        default: {                               // production shape
            long r = n | 1, s = 1 + (long)rnd_below((u64)(r / 2));
            GF2X T;
            SetCoeff(T, r); SetCoeff(T, s); SetCoeff(T, 0);
            // a residue sharing a factor with T now and then
            GF2X res = rpoly(r - 1);
            if (rnd_below(2)) {
                // multiply the residue by (x^(2^j) + x) mod T for a small j:
                // that picks up every factor of T of degree dividing j
                GF2X p;
                SetX(p);
                long j = 1 + (long)rnd_below(12);
                for (long t = 0; t < j; t++) SqrMod(p, p, T);
                GF2X xx; SetX(xx);
                add(p, p, xx);
                MulMod(res, res, p, T);
            }
            a = res;
            b = T;
            break;
        }
        }
        if (rnd_below(2)) swap(a, b);            // either order
        std::vector<u64> A = words(a), B = words(b);
        // pad with zero words now and then: callers pass fixed-width arrays
        if (rnd_below(3) == 0) A.resize(A.size() + 1 + rnd_below(3), 0);
        if (rnd_below(3) == 0) B.resize(B.size() + 1 + rnd_below(3), 0);

        hyb_ntl_finish_bits = finish_choices[rnd_below(5)];
        // a third of the cases with the products of each matrix operation
        // on three threads (threshold lowered so that it happens here)
        hyb_threads = (hyb_threads_ok() && rnd_below(3) == 0) ? 3 : 1;
        hyb_mt_min_bits = 600;
        if (hyb_threads > 1) threaded++;
        std::vector<u64> gref, gs, gf, gx, gc;
        u64 dref = poly_gcd_ntl(A.data(), A.size(), B.data(), B.size(), &gref,
                                ~0ULL);
        hyb_split_apply = true;
        u64 ds = poly_gcd_hybrid(A.data(), A.size(), B.data(), B.size(), &gs,
                                 ~0ULL, nullptr);
        hyb_split_apply = false;
        u64 df = poly_gcd_hybrid(A.data(), A.size(), B.data(), B.size(), &gf,
                                 ~0ULL, nullptr);
        hyb_split_apply = true;
        int stx = -1, stc = -1;
        hyb_cert_xgcd = true;
        u64 dx = poly_gcd_certified(A.data(), A.size(), B.data(), B.size(),
                                    &gx, ~0ULL, nullptr, nullptr, nullptr,
                                    &stx);
        hyb_cert_xgcd = false;
        u64 dc = poly_gcd_certified(A.data(), A.size(), B.data(), B.size(),
                                    &gc, ~0ULL, nullptr, nullptr, nullptr,
                                    &stc);
        // every 7th case: make the chain decline; the answer must then come
        // certified from the extended gcd, and be the same
        if (it % 7 == 3) {
            std::vector<u64> gd;
            int std_ = -1;
            u64 x0, x1, f0, f1;
            hyb_get_stats(x0, x1, f0);
            hyb_force_fallback = true;
            u64 dd = poly_gcd_certified(A.data(), A.size(), B.data(),
                                        B.size(), &gd, ~0ULL, nullptr,
                                        nullptr, nullptr, &std_);
            hyb_force_fallback = false;
            hyb_get_stats(x0, x1, f1);
            declined += (long)(f1 - f0);     // 0 if the pair was already small
            trim(gd);
            std::vector<u64> gr = gref;
            trim(gr);
            if (dd != dref || std_ != GCD_CERT_OK || (dref > 0 && gd != gr)) {
                bad++;
                printf("MISMATCH case %ld: declined chain gave %llu [st %d], "
                       "NTL %llu\n", it, (unsigned long long)dd, std_,
                       (unsigned long long)dref);
            }
        }
        trim(gref); trim(gs); trim(gf); trim(gx); trim(gc);
        bool ok = ds == dref && df == dref && dx == dref && dc == dref &&
                  stx == GCD_CERT_OK && stc == GCD_CERT_OK;
        if (ok && dref > 0)
            ok = gs == gref && gf == gref && gx == gref && gc == gref;
        if (stc == GCD_CERT_OK) certs++;
        if (dref > 0) nontrivial++;
        if (!ok) {
            bad++;
            if (bad <= 10)
                printf("MISMATCH case %ld kind %d: deg a %ld, deg b %ld, "
                       "finish %llu: NTL %llu, split %llu, full %llu, "
                       "xgcd %llu [st %d], chain %llu [st %d]\n",
                       it, kind, deg(a), deg(b),
                       (unsigned long long)hyb_ntl_finish_bits,
                       (unsigned long long)dref, (unsigned long long)ds,
                       (unsigned long long)df, (unsigned long long)dx, stx,
                       (unsigned long long)dc, stc);
        }
    }
    u64 g1, c1, fb_after;
    hyb_get_stats(g1, c1, fb_after);
    printf("gcdstress: %ld cases up to %ld bits (%ld with a nontrivial gcd), "
           "4 engines each vs NTL\n", cases, maxbits, nontrivial);
    printf("  chain certified in %ld of %ld; chain declined on its own %lld "
           "time(s); %ld forced declines re-certified by XGCD\n", certs,
           cases, (long long)(fb_after - fb_before) - (long long)declined,
           declined);
    printf("  %ld of the cases ran each gcd's products on 3 threads%s\n",
           threaded, hyb_threads_ok() ? "" : " (NTL built without "
           "NTL_THREADS: threading not tested)");
    printf("  mismatches: %ld   %s\n", bad, bad ? "*** FAIL ***" : "PASS");
    return bad ? 1 : 0;
}

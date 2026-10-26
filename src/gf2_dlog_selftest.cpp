// gf2_dlog_selftest.cpp -- host self-test for gf2_dlog.h (coarse_sieve, rounds 3-6)
//
//   g++ -O2 -march=native -std=c++17 -I. -Icommon -o gf2_dlog_selftest gf2_dlog_selftest.cpp
//   (or: make gf2_dlog_selftest)
//   ./gf2_dlog_selftest [dlo dhi [budget_MB [samples [tower_dmax [rr_dmax]]]]]
//                                      (default 2 55 1536 20000 44 47)
//
// 1. fast arithmetic vs the bit-serial reference in gf2_wide_field.h, d=2..55:
//    gf2f_mul / gf2f_sqr / gf2f_antilog vs gf_mul / gf_sqr / repeated
//    multiplies, Mersenne mulmod vs mulmod_u64, inverses by phi(M)-1 and by
//    Carmichael lambda(M)-1, exact-division tests
// 2. the necklace (rotation-minimality) test vs the all-rotations reference
// 3. FKM (Lyndon words in order) vs brute force: every word for d <= 22, and
//    stretches at every density for d <= 55
// 4. bit-sliced arithmetic vs scalar for D = 2..55: multiply, square,
//    selected multiply-by-x, g^u with per-slot u, X^E, transposes; and the
//    compile-time moduli vs find_primitive_poly
// 5. both discrete-log engines with the tables the sieve builds (tower: T by
//    log_h, classic or with the subfield tower inside K -- automatic choice,
//    and for d <= 36 every other subfield; PH: 16-byte slots), for the
//    bit-sliced planner (the batched kernel) and the scalar one (RR's
//    factor-base logs):
//    every element for d <= 20, `samples` random elements above; PH tables
//    limited to budget_MB (d = 31, 37, 41 then use baby-step/giant-step, as
//    on an 8 GB card), tower up to tower_dmax (default 44, as the tables
//    take a while to build on the CPU above that; the device test does all
//    even d <= 54, and tower_dmax = 54 does them here too: ~3 GB, minutes)
// 6. the whole per-class computation at every d up to 55: for random classes
//    u, the Zech log, j = u r^-1, the progression start s0 = v j^-1 (lambda
//    exponent) and the minimal polynomial f of g^j -- then checks directly
//    that f divides x^r + x^s0 + 1 (r = 136279841), by arithmetic mod f;
//    for even d <= tower_dmax also that the tower's log is the same; and
//    the reciprocal partner (round 6): the minimal polynomial of g^-j is f
//    reversed, Z(-u) = Z(u) - u, and f* divides x^r + x^s + 1 for
//    s = r - s0 (mod 2^d - 1)
// 7. the packed best[] key format for d <= 55 (order and round trip)
// 8. RR (round 5): the half-gcd at every d = 3..55 (r = t*y mod q by the
//    bit-serial reference multiply, deg r, deg t <= d/2, compile-time and
//    run-time moduli agree), the factor-base sieve against trial division
//    for d/2 <= 14, and RR logs end to end up to rr_dmax (default 47; d=49
//    is skipped -- its factor-base logs need ~2e10 giant steps, a GPU job)
//    with the factor-base logs by PH on the host
// 9. reciprocal pairs (round 6): for every class of degree d <= 22 the pair
//    role of it and of its partner rep(~u) are complementary (self-
//    reciprocal classes alone), the counts match I(d) and the number of
//    self-reciprocal irreducibles; I(d) for every d <= 55
//
// Exit status 0 iff everything matches.  The same code runs on the GPU in
// coarse_sieve --selftest; this is the CPU-side cross-check.
#include "gf2_dlog.h"
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

typedef unsigned long long ull;

static double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::mt19937_64 rng(20261020);
static u64 g_bad = 0;

// ---------------------------------------------------------------- part 1
static u64 test_arith(u64 &n) {
    u64 bad = 0;
    for (int d = 2; d <= 55; d++) {
        u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        if (!gf2f_usable(q, d)) { printf("d=%d: fast path not usable\n", d); bad++; continue; }
        u64 gp[64]; gp[0] = 2;
        for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
        u64 edge[] = {0, 1, 2, M, M - 1, M >> 1, (u64)1 << (d - 1), 0x5555555555555555ULL & M};
        for (u64 a : edge) for (u64 b : edge) {
            n++;
            if (gf2f_mul(a, b, ql, d) != gf_mul(a, b, q, d)) bad++;
        }
        for (int k = 0; k < 20000; k++) {
            u64 a = rng() & M, b = rng() & M;
            n += 3;
            if (gf2f_mul(a, b, ql, d) != gf_mul(a, b, q, d)) bad++;
            if (gf2f_sqr(a, ql, d) != gf_sqr(a, q, d)) bad++;
            u64 x = rng() % M, y = rng() % M;
            if (gf2f_mulmod_m(x, y, d, M) != mulmod_u64(x, y, M)) bad++;
        }
        for (int k = 0; k < 200; k++) {
            u64 i = rng() % M, ref = 1;
            n++;
            for (int t = 0; t < d; t++) if (i >> t & 1) ref = gf_mul(ref, gp[t], q, d);
            if (gf2f_antilog(i, ql, d, gp) != ref) bad++;
        }
        u64 phi = 1, lam = 1, mm = M;
        for (u64 p : factor_u64(M)) {
            u64 pk = 1;
            while (mm % p == 0) { mm /= p; pk *= p; }
            u64 ph = pk / p * (p - 1);
            phi *= ph;
            lam = lam / gcd_u64(lam, ph) * ph;
        }
        for (int k = 0; k < 2000; k++) {
            u64 a = 1 + rng() % (M > 1 ? M - 1 : 1);
            if (gcd_u64(a, M) != 1) continue;
            n += 2;
            if (gf2f_mulmod_m(a, gf2f_powmod_m(a, phi - 1, d, M), d, M) != 1 % M) bad++;
            if (gf2f_mulmod_m(a, gf2f_powmod_m(a, lam - 1, d, M), d, M) != 1 % M) bad++;
        }
    }
    for (u64 p : {3ULL, 7ULL, 31ULL, 127ULL, 5419ULL, 2099863ULL, 164511353ULL}) {
        OddDiv o = odd_div_make(p);
        for (int k = 0; k < 100000; k++) {
            u64 x = rng() >> (rng() % 40), qq = 0;
            n++;
            bool t = odd_div_test(x, o, &qq);
            if (t != (x % p == 0) || (t && qq != x / p)) bad++;
        }
    }
    return bad;
}

// ---------------------------------------------------------------- parts 2, 3
static u64 necklaces(int d) {           // (1/d) sum_{k|d} phi(k) 2^(d/k)
    u64 s = 0;
    for (int k = 1; k <= d; k++) {
        if (d % k) continue;
        int ph = 0;
        for (int t = 1; t <= k; t++) {
            int a = t, b = k;
            while (b) { int c = a % b; a = b; b = c; }
            ph += (a == 1);
        }
        s += (u64)ph << (d / k);
    }
    return s / d;
}

// odd u < 2^(n-1), least of its rotations and aperiodic
static bool is_lyndon_ref(u64 u, int n) {
    const u64 M = ((u64)1 << n) - 1;
    if (!(u & 1) || u >= ((u64)1 << (n - 1))) return false;
    u64 x = u;
    for (int i = 1; i < n; i++) {
        x = ((x << 1) | (x >> (n - 1))) & M;
        if (x <= u) return false;
    }
    return true;
}

static void fkm_range(u64 lo, u64 hi, int n, std::vector<u64> &out) {
    int p;
    u64 w = fkm_seek(lo, n, &p);
    while (w < hi) {
        if (p == n) out.push_back(w);
        w = fkm_next(w, n, &p);
    }
}

static u64 test_necklace(u64 &n) {
    u64 bad = 0;
    for (int d = 2; d <= 22; d++) {
        const u64 M = ((u64)1 << d) - 1;
        u64 reps = 0;
        std::vector<u64> ref, got;
        for (u64 u = 1; u < ((u64)1 << (d - 1)); u += 2) {
            bool f = gf2f_is_min_rotation(u, d, M), r = gf2f_min_rotation_ref(u, d, M);
            n++;
            bad += (f != r);
            reps += r;
            if (is_lyndon_ref(u, d)) ref.push_back(u);
        }
        if (reps != necklaces(d) - 2) { printf("d=%d: %llu reps\n", d, (ull)reps); bad++; }
        fkm_range(0, (u64)1 << (d - 1), d, got);
        n++;
        if (got != ref) { printf("d=%d: FKM %zu vs %zu Lyndon words\n", d, got.size(), ref.size()); bad++; }
        std::vector<u64> cuts = {0, (u64)1 << (d - 1)}, got2;
        for (int i = 0; i < 20; i++) cuts.push_back(rng() % ((u64)1 << (d - 1)));
        std::sort(cuts.begin(), cuts.end());
        for (size_t i = 0; i + 1 < cuts.size(); i++) fkm_range(cuts[i], cuts[i + 1], d, got2);
        n++;
        if (got2 != ref) { printf("d=%d: FKM over sub-ranges differs\n", d); bad++; }
    }
    for (int d = 23; d <= 55; d++) {
        const u64 M = ((u64)1 << d) - 1;
        for (int i = 0; i < 400000; i++) {
            int sh = (64 - (d - 1)) + (int)(rng() % (u64)(d - 1));
            u64 u = (rng() >> sh) | 1;
            n++;
            bad += (gf2f_is_min_rotation(u, d, M) != gf2f_min_rotation_ref(u, d, M));
        }
        for (int t = 0; t < 30; t++) {          // FKM stretches at every density
            int z = 1 + (int)(rng() % (u64)(d - 2));
            u64 lo = ((u64)1 << (d - 1 - z)) + (rng() & (((u64)1 << (d - 1 - z)) - 1));
            u64 hi = std::min<u64>(lo + ((u64)1 << (8 + rng() % 9)), (u64)1 << (d - 1));
            std::vector<u64> got, ref;
            fkm_range(lo, hi, d, got);
            for (u64 u = lo | 1; u < hi; u += 2) if (is_lyndon_ref(u, d)) ref.push_back(u);
            n++;
            if (got != ref) { printf("d=%d: FKM [%llx,%llx) differs\n", d, (ull)lo, (ull)hi); bad++; }
        }
    }
    return bad;
}

// ---------------------------------------------------------------- part 4
static u64 g_bs_n = 0, g_bs_bad = 0;
#define BSCHECK(c) do { g_bs_n++; if (!(c)) g_bs_bad++; } while (0)

template <int D> static void test_bs() {
    const u64 q = find_primitive_poly(D), ql = q ^ ((u64)1 << D), M = ((u64)1 << D) - 1;
    BSCHECK(ql == gf2_qlow(D));
    for (int rep = 0; rep < 24; rep++) {
        u64 a[32], b[32], r[32], u[32];
        for (int k = 0; k < 32; k++) { a[k] = rng() & M; b[k] = rng() & M; u[k] = (rng() & M) >> 1; }
        if (rep == 0) { a[0] = 0; a[1] = 1; a[2] = M; b[3] = 0; b[4] = 1; b[5] = M; u[0] = 0; u[1] = 1; u[2] = M >> 1; }
        u32 A[D], B[D], R[D], U[D], G[D];
        bs_load<D>(a, A); bs_load<D>(b, B);
        bs_store<D>(A, r);
        for (int k = 0; k < 32; k++) BSCHECK(r[k] == a[k]);
        for (int i = 0; i < D; i++) for (int k = 0; k < 32; k += 5)
            BSCHECK(((A[i] >> k) & 1) == ((a[k] >> i) & 1));
        bs_mul<D>(A, B, R); bs_store<D>(R, r);
        for (int k = 0; k < 32; k++) BSCHECK(r[k] == gf_mul(a[k], b[k], q, D));
        bs_sqr<D>(A, R); bs_store<D>(R, r);
        for (int k = 0; k < 32; k++) BSCHECK(r[k] == gf_sqr(a[k], q, D));
        const u32 sel = (u32)rng();
        for (int i = 0; i < D; i++) G[i] = A[i];
        bs_mulx_sel<D>(G, sel); bs_store<D>(G, r);
        for (int k = 0; k < 32; k++) BSCHECK(r[k] == ((sel >> k) & 1 ? gf_mulx(a[k], q, D) : a[k]));
        bs_load<D>(u, U); bs_pow_g<D>(U, G); bs_store<D>(G, r);
        for (int k = 0; k < 32; k++) BSCHECK(r[k] == gf_pow(2, u[k], q, D));
        u64 E = rep == 1 ? 1 : rep == 2 ? M - 1 : 1 + rng() % M;
        bs_pow<D>(A, E, R); bs_store<D>(R, r);
        for (int k = 0; k < 32; k++) BSCHECK(r[k] == gf_pow(a[k], E, q, D));
    }
}
template <int D> struct BSAll { static void go() { BSAll<D - 1>::go(); test_bs<D>(); } };
template <> struct BSAll<1> { static void go() {} };

// ---------------------------------------------------------------- part 5
struct PHTabs { std::vector<PHSlot> slots[PH_MAXPARTS]; };

static void ph_fill_host(const PHHost &H, u64 ql, int d, PHTabs &tb, PHDev &D) {
    D.nparts = H.nparts;
    D.chain = H.chain;
    for (int i = 0; i < H.nparts; i++) {
        const PHPartHost &p = H.part[i];
        const size_t S = (size_t)1 << p.sbits;
        tb.slots[i].assign(S, PHSlot{0, 0});
        u64 y = 1;
        for (u64 k = 0; k < p.N; k++, y = gf2f_mul(y, p.gamma, ql, d)) {
            u64 h = ph_slot(y, p.sbits);
            while (tb.slots[i][h].key) h = (h + 1) & (S - 1);
            tb.slots[i][h] = PHSlot{y, k};
        }
        D.part[i] = PHPart{p.P, p.E, p.K, p.N, p.G, p.delta, tb.slots[i].data(), p.sbits};
    }
}

struct TowerTabs { TowerHost W; std::vector<u64> T; TowerDev dev; };

// T by log_h(c), filled by the same coupon-collector sweep as k_tower_T;
// sub: -1 automatic (subfield tower when m is composite), 0 classic, n
static bool tower_fill_host(int d, u64 q, TowerTabs &tt, int sub = -1) {
    if (!tower_build_host(d, q, tt.W, sub)) return false;
    const u64 M = ((u64)1 << d) - 1, MK = tt.W.MK;
    tt.T.assign((size_t)MK + 1, ~0ULL);
    tt.dev = tower_dev(tt.W, tt.W.btab.data(), tt.W.LK.data(), tt.T.data(),
                       tt.W.LKn.data(), tt.W.TP.data());
    u64 left = MK + 1, x = 1;
    for (u64 u = 0; u < M && left; u++, x = gf_mulx(x, q, d)) {
        u64 lb, l;
        if (!tower_split(tower_coords(x, tt.dev), tt.dev, &lb, &l)) continue;
        if (tt.T[l] == ~0ULL) {
            u64 h = (MK + 2) * lb;
            tt.T[l] = u >= h ? u - h : u + M - h;
            left--;
        }
    }
    return left == 0;
}

static int test_dlog(int dlo, int dhi, size_t budget, u64 samples, int tower_dmax) {
    int fails = 0;
    for (int d = std::max(dlo, 2); d <= dhi; d++) {
        const u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        u64 gp[64]; gp[0] = 2;
        for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
        const bool exh = d <= 20;
        const u64 nchk = exh ? M : samples;
        for (int model = 0; model < 2; model++) {
            double t0 = now();
            PHHost H; PHTabs tb; PHDev D{};
            if (!ph_plan_host(d, q, budget, H, model ? kPHCostScalar : kPHCostBitsliced)) {
                printf("d=%2d PH (%s planner): no feasible plan%s\n", d, model ? "scalar" : "bit-sliced",
                       d == 49 ? " (expected: 2^49-1 has the prime factor 4432676798593)" : "  <-- FAIL");
                if (d != 49) fails++;
                continue;
            }
            ph_fill_host(H, ql, d, tb, D);
            u64 bad = 0, e = 1;
            for (u64 k = 0; k < nchk; k++) {
                u64 u = exh ? k : rng() % M;
                if (!exh) e = gf2f_antilog(u, ql, d, gp);
                if (ph_log(e, M, d, ql, D) != u) bad++;
                if (exh) e = gf_mulx(e, q, d);
            }
            printf("d=%2d PH    %s %8llu logs: %llu wrong | %-10s:", d, exh ? "all    " : "sampled",
                   (ull)nchk, (ull)bad, model ? "scalar" : "bit-sliced");
            for (int i = 0; i < H.nparts; i++)
                printf(" %llu(pc %d%s)", (ull)H.part[i].P, __builtin_popcountll(H.part[i].E),
                       H.part[i].G > 1 ? (", G=" + std::to_string(H.part[i].G)).c_str() : "");
            printf(", %.0f MB, %.1fs\n", H.table_bytes / 1048576.0, now() - t0);
            if (bad) fails++;
        }
        // towers: the classic one, the automatic subfield tower, and (for
        // d <= 36) every other inner subfield K_n, n | m
        std::vector<int> subs;
        if (d >= 4 && !(d & 1) && d <= tower_dmax) {
            subs.push_back(0);
            if (tower_sub_choice(d / 2, -1, nullptr)) subs.push_back(-1);
            if (d <= 36)
                for (int n = 2; n < d / 2; n++)
                    if ((d / 2) % n == 0 && n != tower_sub_choice(d / 2, -1, nullptr))
                        subs.push_back(n);
        }
        for (int sub : subs) {
            double t0 = now();
            TowerTabs tt;
            if (!tower_fill_host(d, q, tt, sub)) { printf("d=%2d tower build FAILED\n", d); fails++; continue; }
            u64 bad = 0, e = 1;
            for (u64 k = 0; k < nchk; k++) {
                u64 u = exh ? k : rng() % M;
                if (!exh) e = gf2f_antilog(u, ql, d, gp);
                if (tower_log(e, M, tt.dev) != u) bad++;
                if (exh) e = gf_mulx(e, q, d);
            }
            char how[48];
            if (tt.W.sub_k) snprintf(how, sizeof how, "inner GF(2^%d), k=%d%s", tt.W.sub_n,
                                     tt.W.sub_k, sub < 0 ? "" : " (forced)");
            else snprintf(how, sizeof how, "classic");
            printf("d=%2d tower %s %8llu logs: %llu wrong | m=%d, %s, %.1fs\n", d, exh ? "all    " : "sampled",
                   (ull)nchk, (ull)bad, tt.W.m, how, now() - t0);
            if (bad) fails++;
        }
        fflush(stdout);
    }
    return fails;
}

// ---------------------------------------------------------------- part 8
// RR: the half-gcd's contract, for random y at degree D
template <int D> static u64 rr_split_check(u64 &n) {
    const u64 q = find_primitive_poly(D), mask = ((u64)1 << D) - 1;
    u64 bad = 0;
    for (int i = 0; i < 3000; i++) {
        u64 y = rng() & mask;
        if (!y) y = 1;
        u64 r, r2;
        u32 t, t2;
        rr_split<D>(y, &r, &t);
        rr_split_rt(y, q, D, D / 2, &r2, &t2);
        n++;
        if (r != r2 || t != t2 || !t || (r >> (D / 2 + 1)) || (t >> (D / 2 + 1)) ||
            gf_mul(t, y, q, D) != r) bad++;
    }
    return bad;
}
template <int D> struct RRAll { static void go(u64 &n, u64 &b) { RRAll<D - 1>::go(n, b); b += rr_split_check<D>(n); } };
template <> struct RRAll<2> { static void go(u64 &, u64 &) {} };

static u32 poly_mod32(u32 a, u32 f) {
    const int df = 31 - __builtin_clz(f);
    for (int k = 31 - __builtin_clz(a | 1) - df; k >= 0; k--)
        if ((a >> (k + df)) & 1) a ^= f << k;
    return a;
}

// the sieve vs trial division: the irreducibles of degree <= dh in order, and
// spf[P] = the least (irreducible) divisor of every reducible P
static u64 test_rr_sieve(int dh, u64 &n) {
    std::vector<u32> spf, irr, ref;
    rr_sieve(dh, spf, irr);
    u64 bad = 0;
    for (u32 P = 2; P < (2u << dh); P++) {
        const int dp = 31 - __builtin_clz(P);
        u32 least = 0;
        for (u32 f = 2; 2 * (31 - __builtin_clz(f)) <= dp; f++)
            if (!poly_mod32(P, f)) { least = f; break; }
        if (!least) ref.push_back(P);
        if (spf[P] != least) bad++;
        n++;
    }
    if (ref != irr) bad++;
    return bad;
}

static int test_rr(int dlo, int dhi, size_t budget, u64 samples) {
    int fails = 0;
    for (int d = std::max(dlo, 3); d <= dhi; d++) {
        if (d == 49) { printf("d=49 RR    skipped (factor-base logs: a GPU job)\n"); continue; }
        const double t0 = now();
        const u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        const int dh = d / 2;
        u64 gp[64]; gp[0] = 2;
        for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
        PHHost H; PHTabs tb; PHDev D{};
        if (!ph_plan_host(d, q, budget, H, kPHCostScalar)) {
            printf("d=%2d RR    no PH plan for the factor-base logs  <-- FAIL\n", d);
            fails++;
            continue;
        }
        ph_fill_host(H, ql, d, tb, D);
        std::vector<u32> spf, irr;
        rr_sieve(dh, spf, irr);
        std::vector<u64> fbl(irr.size()), L;
        u64 badfb = 0;
        for (size_t i = 0; i < irr.size(); i++) {
            fbl[i] = ph_log(irr[i], M, d, ql, D);
            if (fbl[i] >= M || gf2f_antilog(fbl[i], ql, d, gp) != irr[i]) badfb++;
        }
        rr_fill(dh, M, spf, irr, fbl, L);
        const RRDev rr{L.data(), dh};
        const bool exh = d <= 20;
        const u64 nchk = exh ? M : samples;
        u64 bad = 0, e = 1;
        for (u64 k = 0; k < nchk; k++) {
            const u64 u = exh ? k : rng() % M;
            if (!exh) e = gf2f_antilog(u, ql, d, gp);
            if (rr_log(e, q, d, M, rr) != u) bad++;
            if (exh) e = gf_mulx(e, q, d);
        }
        printf("d=%2d RR    %s %8llu logs: %llu wrong | %zu irreducibles of degree <= %d "
               "(%llu bad), table %.0f MB, %.1fs\n", d, exh ? "all    " : "sampled",
               (ull)nchk, (ull)bad, irr.size(), dh, (ull)badfb, L.size() * 8 / 1048576.0,
               now() - t0);
        if (bad || badfb) fails++;
        fflush(stdout);
    }
    return fails;
}

// ---------------------------------------------------------------- part 6
// minimal polynomial of a (coefficient mask), as in coarse_sieve's kernels
static u64 minpoly_mask(u64 a, u64 q, int d) {
    u64 P[64] = {0};
    P[0] = 1;
    u64 root = a;
    for (int i = 0; i < d; i++) {
        for (int k = i + 1; k >= 1; k--) P[k] = P[k - 1] ^ gf_mul(root, P[k], q, d);
        P[0] = gf_mul(root, P[0], q, d);
        root = gf_sqr(root, q, d);
    }
    u64 mask = 0;
    for (int k = 0; k <= d; k++) {
        if (P[k] > 1) return 0;
        mask |= P[k] << k;
    }
    return mask;
}

// x^e mod f (f of degree d given as a mask with bit d set), bit-serial
static u64 xpow_mod(u64 e, u64 f, int d) {
    u64 r = 1, b = 2;                          // x
    for (; e; e >>= 1) {
        if (e & 1) r = gf_mul(r, b, f, d);
        b = gf_mul(b, b, f, d);
    }
    return r;
}

static int test_classes(int dlo, int dhi, size_t budget, int per_degree,
                        int tower_dmax) {
    const u64 r = 136279841;
    int fails = 0;
    for (int d = std::max(dlo, 3); d <= dhi; d++) {
        if (d == 49) continue;
        const u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        if (gcd_u64(r % M, M) != 1) continue;
        u64 gp[64]; gp[0] = 2;
        for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
        PHHost H; PHTabs tb; PHDev D{};
        if (!ph_plan_host(d, q, budget, H, kPHCostBitsliced)) { printf("d=%d: no PH plan\n", d); fails++; continue; }
        ph_fill_host(H, ql, d, tb, D);
        // even d: the production engine is the tower -- its log must agree
        TowerTabs tt;
        const bool tw = d >= 4 && !(d & 1) && d <= tower_dmax && tower_fill_host(d, q, tt);
        int twbad = 0;
        const u64 rinv = inv_mod(r % M, M);
        u64 lam = 1, mm = M;
        for (u64 p : factor_u64(M)) {
            u64 pk = 1;
            while (mm % p == 0) { mm /= p; pk *= p; }
            u64 ph = pk / p * (p - 1);
            lam = lam / gcd_u64(lam, ph) * ph;
        }
        int done = 0, bad = 0, pbad = 0, tries = 0;
        while (done < per_degree && tries++ < 100 * per_degree) {
            int p;
            u64 u = fkm_seek((rng() % ((u64)1 << (d - 1))) | 1, d, &p);
            while (p != d && u < ((u64)1 << (d - 1))) u = fkm_next(u, d, &p);
            if (u >= ((u64)1 << (d - 1)) || gcd_u64(u, M) != 1) continue;
            const u64 x = gf2f_antilog(u, ql, d, gp) ^ 1;     // g^u + 1
            const u64 v = ph_log(x, M, d, ql, D);
            if (tw && tower_log(x, M, tt.dev) != v) twbad++;
            const u64 j = gf2f_mulmod_m(u, rinv, d, M);
            const u64 s0 = gf2f_mulmod_m(v, gf2f_powmod_m(j, lam - 1, d, M), d, M);
            const u64 f = minpoly_mask(gf2f_antilog(j, ql, d, gp), q, d);
            // f must be irreducible of degree d with f | x^r + x^s0 + 1
            const u64 s = s0 ? s0 : M;
            const bool ok = f != 0 && (f >> d) == 1 &&
                            (xpow_mod(r, f, d) ^ xpow_mod(s, f, d) ^ 1) == 0;
            if (!ok) { if (bad < 3) printf("d=%d u=%llx: f=%llx s0=%llu does not divide\n", d, (ull)u, (ull)f, (ull)s0); bad++; }
            // the reciprocal partner (round 6): its class is that of -u, its
            // root g^-j has the minimal polynomial f* (f reversed), Z(-u) =
            // Z(u) - u, and it divides x^r + x^s + 1 for s = r - s0 (mod M)
            {
                const u64 fr = minpoly_mask(gf2f_antilog(M - j, ql, d, gp), q, d);
                u64 frev = 0;
                for (int k = 0; k <= d; k++) if ((f >> k) & 1) frev |= (u64)1 << (d - k);
                const u64 rm = r % M, sp0 = rm >= s0 ? rm - s0 : rm + M - s0;
                const u64 sp = sp0 ? sp0 : M;
                const u64 vn = ph_log(gf2f_antilog(M - u, ql, d, gp) ^ 1, M, d, ql, D);
                const bool okp = fr != 0 && fr == frev &&
                                 (xpow_mod(r, fr, d) ^ xpow_mod(sp, fr, d) ^ 1) == 0 &&
                                 vn == (v >= u ? v - u : v + M - u);
                if (!okp) {
                    if (pbad < 3) printf("d=%d u=%llx: partner f*=%llx s=%llu fails\n", d, (ull)u, (ull)fr, (ull)sp);
                    pbad++;
                }
            }
            done++;
        }
        if (bad || pbad || twbad || done < per_degree) fails++;
        printf("d=%2d classes: %d checked, f | x^r + x^s0 + 1 for %d, f* | x^r + x^(r-s0) + 1 for %d",
               d, done, done - bad, done - pbad);
        if (tw) printf("; tower log %s", twbad ? "DIFFERS" : "agrees");
        printf("\n");
        fflush(stdout);
    }
    return fails;
}

// ---------------------------------------------------------------- part 7
static ull key_pack_h(int d, u64 mask) {
    return d <= 47 ? ((ull)d << 48) | mask : (0xFull << 60) | ((ull)(d - 48) << 56) | mask;
}
static int test_keys() {
    int bad = 0;
    ull prev = 0;
    for (int d = 2; d <= 55; d++) {
        const u64 lo = ((u64)1 << d) | 1, hi = ((u64)2 << d) - 1;   // monic masks, ends
        const ull k0 = key_pack_h(d, lo), k1 = key_pack_h(d, hi);
        if (!(k0 > prev && k1 > k0 && k1 < ~0ULL)) bad++;
        prev = k1;
    }
    return bad;
}

// ---------------------------------------------------------------- part 9
// Reciprocal pairs (round 6): gf2f_pair_role must split the Lyndon words of
// length d into pairs {u, rep(~u)} with exactly one canonical member, plus
// the self-reciprocal classes as singletons; their numbers must match the
// formulas, and gf2_irreducible_count must count the classes.
static long long sr_count(int d) {          // self-reciprocal irreducibles
    if (d & 1) return d == 1 ? 1 : 0;
    const int m = d / 2;
    long long s = 0;
    for (int e = 1; e <= m; e += 2) {
        if (m % e) continue;
        int mu = 1, x = e;
        for (int p = 2; p * p <= x; p++) {
            if (x % p) continue;
            x /= p;
            if (x % p == 0) { mu = 0; break; }
            mu = -mu;
        }
        if (mu && x > 1) mu = -mu;
        s += mu * (long long)((u64)1 << (m / e));
    }
    return s / d;
}

static int test_pairs(int dmax_exh, u64 &nchecked) {
    int fails = 0;
    for (int d = 2; d <= dmax_exh; d++) {
        const u64 M = ((u64)1 << d) - 1;
        u64 nly = 0, n1 = 0, n0 = 0, n2 = 0, bad = 0;
        int p;
        for (u64 u = fkm_seek(1, d, &p); u < ((u64)1 << (d - 1)) || (d == 2 && u == 1);
             u = fkm_next(u, d, &p)) {
            if (p == d) {
                nly++;
                const int role = gf2f_pair_role(u, d, M);
                const u64 pr = gf2f_min_rotation(~u & M, d, M);
                const int prole = gf2f_pair_role(pr, d, M);
                if (role == 1) { n1++; if (prole != 0 || pr == u) bad++; }
                else if (role == 0) { n0++; if (prole != 1 || pr == u) bad++; }
                else { n2++; if (pr != u) bad++; }
            }
            if (d == 2) break;
        }
        nchecked += nly;
        if (bad || nly != gf2_irreducible_count(d) || n1 != n0 ||
            (long long)n2 != sr_count(d)) {
            printf("d=%d pairs: %llu classes (I(d) %llu), %llu canonical, %llu skipped, "
                   "%llu self-reciprocal (formula %lld), %llu bad  <-- FAIL\n", d,
                   (ull)nly, (ull)gf2_irreducible_count(d), (ull)n1, (ull)n0, (ull)n2,
                   sr_count(d), (ull)bad);
            fails++;
        }
    }
    // I(d) for all d <= 55 by the necklace identity sum_{e|d} e I(e) = 2^d
    for (int d = 1; d <= 55; d++) {
        unsigned __int128 t = 0;
        for (int e = 1; e <= d; e++)
            if (d % e == 0) t += (unsigned __int128)e * (e == 1 ? 2 : gf2_irreducible_count(e));
        if (t != ((unsigned __int128)1 << d)) { printf("I(%d) wrong  <-- FAIL\n", d); fails++; }
    }
    return fails;
}

int main(int argc, char **argv) {
    int dlo = argc > 1 ? atoi(argv[1]) : 2, dhi = argc > 2 ? atoi(argv[2]) : 55;
    size_t budget = (size_t)(argc > 3 ? atol(argv[3]) : 1536) << 20;
    u64 samples = argc > 4 ? strtoull(argv[4], 0, 10) : 20000;
    int tower_dmax = argc > 5 ? atoi(argv[5]) : 44;
    int rr_dmax = argc > 6 ? atoi(argv[6]) : 47;
    if (tower_dmax > 54) tower_dmax = 54;
    if (dlo < 2) dlo = 2;
    if (dhi > 55) dhi = 55;
    double t0 = now();
    u64 n1 = 0, n2 = 0;
    u64 b1 = test_arith(n1);
    printf("1. fast arithmetic vs reference, d=2..55: %llu checks, %llu mismatches\n", (ull)n1, (ull)b1);
    u64 b2 = test_necklace(n2);
    printf("2./3. necklace test and FKM vs brute force: %llu checks, %llu mismatches\n", (ull)n2, (ull)b2);
    BSAll<55>::go();
    printf("4. bit-sliced arithmetic D=2..55: %llu checks, %llu mismatches\n", (ull)g_bs_n, (ull)g_bs_bad);
    fflush(stdout);
    printf("5. discrete logs, d=%d..%d (PH tables <= %zu MB):\n", dlo, dhi, budget >> 20);
    int f5 = test_dlog(dlo, dhi, budget, samples, tower_dmax);
    printf("6. per-class computation vs direct division, d=%d..%d:\n", std::max(dlo, 3), dhi);
    int f6 = test_classes(dlo, dhi, budget, 40, tower_dmax);
    int f7 = test_keys();
    printf("7. packed keys d=2..55: %s\n", f7 ? "FAILED" : "ok");
    u64 n8 = 0, b8 = 0, n8s = 0, b8s = 0;
    RRAll<55>::go(n8, b8);
    for (int dh = 1; dh <= 14; dh++) b8s += test_rr_sieve(dh, n8s);
    printf("8. RR: half-gcd d=3..55: %llu checks, %llu bad; factor-base sieve "
           "d/2 <= 14: %llu checks, %llu bad; RR logs d=%d..%d:\n", (ull)n8,
           (ull)b8, (ull)n8s, (ull)b8s, std::max(dlo, 3), std::min(dhi, rr_dmax));
    fflush(stdout);
    int f8 = test_rr(dlo, std::min(dhi, rr_dmax), budget, samples);
    u64 n9 = 0;
    int f9 = test_pairs(22, n9);
    printf("9. reciprocal pairs: every class d=2..22 (%llu), pair roles and counts "
           "(I(d), self-reciprocal) %s; I(d) d<=55 %s\n", (ull)n9,
           f9 ? "FAILED" : "ok", f9 ? "(see above)" : "ok");
    bool ok = !b1 && !b2 && !g_bs_bad && !f5 && !f6 && !f7 && !b8 && !b8s && !f8 && !f9;
    printf("gf2_dlog_selftest: %s (%.0fs)\n", ok ? "ALL OK" : "FAILURES", now() - t0);
    (void)g_bad;
    return ok ? 0 : 1;
}

// gf2_dlog_selftest.cpp -- host self-test for gf2_dlog.h (coarse_sieve round 2)
//
//   g++ -O2 -march=native -I. -Icommon -o gf2_dlog_selftest gf2_dlog_selftest.cpp
//   ./gf2_dlog_selftest [dlo dhi [budget_MB [samples]]]     (default 2 47 1536 20000)
//
// 1. fast arithmetic vs the bit-serial reference in gf2_wide_field.h, d=2..47:
//    gf2f_mul / gf2f_sqr / gf2f_antilog vs gf_mul / gf_sqr / repeated
//    multiplies, Mersenne mulmod vs mulmod_u64, inverses by phi(M)-1 and by
//    Carmichael lambda(M)-1, exact-division tests
// 2. the necklace (rotation-minimality) test vs the all-rotations reference:
//    exhaustive over odd u < 2^(d-1) for d <= 22 (and the class count vs the
//    necklace formula), random u with spread leading-zero counts for d <= 47
// 3. both discrete-log engines: every element for d <= 20, `samples` random
//    elements above; PH tables are limited to budget_MB (so the large prime
//    factors at d = 31, 37, 41 run as baby-step/giant-step, as on an 8 GB card)
//
// Exit status 0 iff everything matches.  The same engines run on the GPU in
// coarse_sieve_wide --selftest-logs; this is the CPU-side cross-check.
#include "gf2_dlog.h"
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

static double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- part 1
static u64 test_arith(std::mt19937_64 &rng, u64 &n) {
    u64 bad = 0;
    for (int d = 2; d <= 47; d++) {
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

// ---------------------------------------------------------------- part 2
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

static u64 test_rotation(std::mt19937_64 &rng, u64 &n) {
    u64 bad = 0;
    for (int d = 2; d <= 22; d++) {
        const u64 M = ((u64)1 << d) - 1;
        u64 reps = 0;
        for (u64 u = 1; u < ((u64)1 << (d - 1)); u += 2) {
            bool f = gf2f_is_min_rotation(u, d, M), r = gf2f_min_rotation_ref(u, d, M);
            n++;
            bad += (f != r);
            reps += r;
        }
        // minimal odd rotations < 2^(d-1) = all necklaces but 0...0 and 1...1
        if (reps != necklaces(d) - 2) {
            printf("d=%d: %llu representatives, expected %llu\n", d,
                   (unsigned long long)reps, (unsigned long long)(necklaces(d) - 2));
            bad++;
        }
    }
    for (int d = 23; d <= 47; d++) {
        const u64 M = ((u64)1 << d) - 1;
        for (int i = 0; i < 1000000; i++) {
            int sh = (64 - (d - 1)) + (int)(rng() % (u64)(d - 1));    // spread z
            u64 u = (rng() >> sh) | 1;
            n++;
            bad += (gf2f_is_min_rotation(u, d, M) != gf2f_min_rotation_ref(u, d, M));
        }
    }
    return bad;
}

// ---------------------------------------------------------------- part 3
struct PHTabs { std::vector<u64> keys[PH_MAXPARTS]; std::vector<u32> vals[PH_MAXPARTS]; };

static void ph_fill_host(const PHHost &H, u64 ql, int d, PHTabs &tb, PHDev &D) {
    D.nparts = H.nparts;
    D.chain = H.chain;
    for (int i = 0; i < H.nparts; i++) {
        const PHPartHost &p = H.part[i];
        const size_t S = (size_t)1 << p.sbits;
        tb.keys[i].assign(S, 0);
        tb.vals[i].assign(S, 0);
        u64 y = 1;
        for (u64 k = 0; k < p.N; k++, y = gf2f_mul(y, p.gamma, ql, d)) {
            u64 h = ph_slot(y, p.sbits);
            while (tb.keys[i][h]) h = (h + 1) & (S - 1);
            tb.keys[i][h] = y;
            tb.vals[i][h] = (u32)k;
        }
        D.part[i] = PHPart{p.P, p.E, p.K, p.N, p.G, p.delta,
                           tb.keys[i].data(), tb.vals[i].data(), p.sbits};
    }
}

static int test_dlog(int dlo, int dhi, size_t budget, u64 samples, std::mt19937_64 &rng) {
    int fails = 0;
    for (int d = dlo; d <= dhi; d++) {
        const u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        u64 gp[64]; gp[0] = 2;
        for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
        const bool exh = d <= 20;
        const u64 nchk = exh ? M : samples;
        // Pohlig-Hellman
        double t0 = now();
        PHHost H; PHTabs tb; PHDev D{};
        if (!ph_plan_host(d, q, budget, H)) { printf("d=%2d PH plan FAILED\n", d); fails++; continue; }
        ph_fill_host(H, ql, d, tb, D);
        double tb_ph = now() - t0;
        u64 bad = 0, e = 1;
        for (u64 k = 0; k < nchk; k++) {
            u64 u = exh ? k : rng() % M;
            if (!exh) e = gf2f_antilog(u, ql, d, gp);
            if (ph_log(e, M, d, ql, D) != u) bad++;
            if (exh) e = gf_mulx(e, q, d);
        }
        printf("d=%2d PH    %s %8llu logs: %llu wrong |", d, exh ? "all    " : "sampled",
               (unsigned long long)nchk, (unsigned long long)bad);
        for (int i = 0; i < H.nparts; i++)
            printf(" %llu(pc %d%s)", (unsigned long long)H.part[i].P,
                   __builtin_popcountll(H.part[i].E),
                   H.part[i].G > 1 ? (", BSGS G=" + std::to_string(H.part[i].G)).c_str() : "");
        printf(" chain %d, %.0f MB, %.1fs\n", H.chain, H.table_bytes / 1048576.0, tb_ph);
        if (bad) fails++;
        // tower
        if (d >= 4 && !(d & 1) && d <= 46) {
            t0 = now();
            TowerHost W;
            if (!tower_build_host(d, q, W)) { printf("d=%2d tower build FAILED\n", d); fails++; continue; }
            std::vector<u64> T((size_t)W.MK + 1, ~0ULL);
            u64 left = W.MK + 1, x = 1;
            TowerDev TD{W.m, W.nb, W.MK, W.btab.data(), W.LK.data(), W.AK.data(), T.data()};
            for (u64 u = 0; u < M && left; u++, x = gf_mulx(x, q, d)) {   // coupon collector
                u64 pk = tower_coords(x, TD), ac = pk & W.MK, bc = pk >> W.m;
                if (!bc) continue;
                u64 lb = W.LK[bc], la = W.LK[ac];
                u64 cc = ac ? W.AK[la >= lb ? la - lb : la + W.MK - lb] : 0;
                if (T[cc] == ~0ULL) {
                    u64 t = (W.MK + 2) * lb;
                    T[cc] = u >= t ? u - t : u + M - t;
                    left--;
                }
            }
            double tb_tw = now() - t0;
            bad = 0; e = 1;
            for (u64 k = 0; k < nchk; k++) {
                u64 u = exh ? k : rng() % M;
                if (!exh) e = gf2f_antilog(u, ql, d, gp);
                if (tower_log(e, M, TD) != u) bad++;
                if (exh) e = gf_mulx(e, q, d);
            }
            printf("d=%2d tower %s %8llu logs: %llu wrong | m=%d, %.1fs\n", d,
                   exh ? "all    " : "sampled", (unsigned long long)nchk,
                   (unsigned long long)bad, W.m, tb_tw);
            if (bad || left) fails++;
        }
        fflush(stdout);
    }
    return fails;
}

int main(int argc, char **argv) {
    int dlo = argc > 1 ? atoi(argv[1]) : 2, dhi = argc > 2 ? atoi(argv[2]) : 47;
    size_t budget = (size_t)(argc > 3 ? atol(argv[3]) : 1536) << 20;
    u64 samples = argc > 4 ? strtoull(argv[4], 0, 10) : 20000;
    if (dlo < 2) dlo = 2;
    if (dhi > 47) dhi = 47;
    std::mt19937_64 rng(20261019);
    u64 n1 = 0, n2 = 0;
    u64 b1 = test_arith(rng, n1);
    printf("1. fast arithmetic vs reference, d=2..47: %llu checks, %llu mismatches\n",
           (unsigned long long)n1, (unsigned long long)b1);
    u64 b2 = test_rotation(rng, n2);
    printf("2. necklace test vs all rotations, d=2..47: %llu checks, %llu mismatches\n",
           (unsigned long long)n2, (unsigned long long)b2);
    printf("3. discrete logs, d=%d..%d (PH tables <= %zu MB):\n", dlo, dhi, budget >> 20);
    int f3 = test_dlog(dlo, dhi, budget, samples, rng);
    bool ok = !b1 && !b2 && !f3;
    printf("gf2_dlog_selftest: %s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}

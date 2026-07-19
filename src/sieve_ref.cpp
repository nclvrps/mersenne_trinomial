// sieve_ref.cpp
// CPU reference implementation of the GF(2) trinomial coarse sieve.
//
// For each s in [1, SMAX], finds the smallest-degree irreducible factor
// (ties broken by lexicographically least mask, which for equal degree
// equals numerically least mask) of  T_s(x) = x^r + x^s + 1  over GF(2),
// restricted to factor degrees d <= DEPTH.
//
// Algorithm (no big-polynomial arithmetic, no GCDs):
//   For each degree d:
//     - build GF(2^d) = GF(2)[x]/q_d with q_d a primitive polynomial, so
//       g = x generates the multiplicative group of order M = 2^d - 1;
//     - build antilog table A[i] = g^i and log table L[A[i]] = i;
//     - an irreducible p of degree d divides T_s  iff  T_s(beta) = 0 for
//       a root beta = g^j of p, i.e.  beta^r + beta^s = 1;
//     - in log space with u = j*r mod M: beta^r = g^u, and
//       beta^s = g^u + 1 = g^{Z(u)}  where Z is the Zech logarithm,
//       Z(u) = L[A[u] ^ 1]; so we need  j*s == Z(u) (mod M),
//       an arithmetic progression in s with modulus n = M / gcd(j, M);
//     - conjugate roots (j, 2j, 4j, ...) give the same p and the same
//       progression, so only the minimal j of each cyclotomic coset is
//       processed; elements of proper subfields are skipped (handled at
//       their own degree);
//     - mark the progression with packed key (d << 48) | p_mask, taking
//       min at each s: min over packed keys == (min degree, then
//       lexicographically least mask).
//
// Modes:
//   ./sieve_ref selftest
//        cross-checks against an independent brute-force oracle
//   ./sieve_ref run <r> <depth> <smax>
//        prints "s d p<hex>" for every s that got a factor
//   ./sieve_ref validate <r> <depth> <smax> <logfile>
//        validates against factor.cpp output lines "s d p<hex>"

#include "gf2_small_field.h"
#include "gf2_field_setup.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The sieve. best[s] is min-packed (d<<48)|mask; UINT64_MAX = no factor yet.
struct SieveStats { u64 classes = 0, marks = 0; };

static void sieve_degree(int d, u64 r, u64 smax, std::vector<u64> &best,
                         SieveStats &st) {
    const u64 M64 = ((u64)1 << d) - 1;
    const u32 M = (u32)M64;
    const u32 q = find_primitive_poly(d);

    // subfield moduli: beta = g^j lies in GF(2^e) iff (M/(2^e-1)) | j
    std::vector<u64> subq;
    for (u64 p : factor_u64((u64)d)) {
        int e = d / (int)p;
        subq.push_back(M64 / (((u64)1 << e) - 1));
    }

    // tables
    std::vector<u32> A((size_t)M64);
    std::vector<u32> L((size_t)M64 + 1);
    u32 t = 1;
    for (u64 i = 0; i < M64; i++) { A[i] = t; t = gf_mulx(t, q, d); }
    if (t != 1) { fprintf(stderr, "d=%d: generator order wrong\n", d); exit(1); }
    for (u64 i = 0; i < M64; i++) L[A[i]] = (u32)i;

    const u64 rmod = r % M64;
    if (gcd_u64(rmod ? rmod : M64, M64) != 1) {
        fprintf(stderr, "d=%d: gcd(r, 2^d-1) != 1; unsupported r\n", d);
        exit(1);
    }

    u32 P[40]; // minpoly scratch (d <= 39 supported here; we use d <= 31)

    for (u32 j = 1; j < M; j++) {
        // skip proper-subfield elements
        bool sub = false;
        for (u64 sq_ : subq) if (j % sq_ == 0) { sub = true; break; }
        if (sub) continue;
        // only minimal element of each cyclotomic coset
        u32 jj = j; bool minimal = true;
        for (int i = 1; i < d; i++) {
            jj = rotl_d(jj, d, M);
            if (jj < j) { minimal = false; break; }
        }
        if (!minimal) continue;

        st.classes++;

        u64 u = ((u64)j * rmod) % M64;
        if (u == 0) continue;                    // beta^r = 1: no solutions
        u32 v = L[A[u] ^ 1u];                    // Zech logarithm of u

        u64 gg = gcd_u64(j, M64);
        if ((u64)v % gg != 0) continue;          // j*s == v (mod M) unsolvable
        u64 n = M64 / gg;
        u64 s0 = (inv_mod((u64)j / gg, n) * ((u64)v / gg)) % n;

        u64 mask = minpoly_mask(j, d, M, q, A.data(), P);
        if (mask == 0 || !((mask >> d) & 1) || !(mask & 1)) {
            fprintf(stderr, "d=%d j=%u: minpoly sanity failure\n", d, j);
            exit(1);
        }
        u64 packed = ((u64)d << 48) | mask;

        for (u64 s = (s0 == 0 ? n : s0); s <= smax; s += n) {
            st.marks++;
            if (packed < best[s]) best[s] = packed;
        }
    }
}

static void run_sieve(u64 r, int depth, u64 smax, std::vector<u64> &best) {
    best.assign(smax + 1, UINT64_MAX);
    for (int d = 2; d <= depth; d++) {
        double t0 = now_s();
        SieveStats st;
        sieve_degree(d, r, smax, best, st);
        fprintf(stderr, "d=%2d done: %" PRIu64 " classes, %" PRIu64
                " marks, %.2fs\n", d, st.classes, st.marks, now_s() - t0);
    }
}

// ---------------------------------------------------------------------------
// Independent brute-force oracle: enumerate irreducibles directly and test
// divisibility by evaluating x^r + x^s + 1 in GF(2)[x]/p, iterating s so
// that x^s is maintained incrementally.  Shares only the primitive gf_mul
// with the sieve; the number theory (Zech logs, cosets, progressions,
// minpoly) is all exercised against this.
static void brute_force(u64 r, int depth, u64 smax, std::vector<u64> &best) {
    best.assign(smax + 1, UINT64_MAX);
    struct Irr { int d; u32 mask; };
    std::vector<Irr> irr;
    for (int d = 2; d <= depth; d++)
        for (u32 m = (1u << d) | 1u; m < (2u << d); m += 2)
            if (is_irreducible(m, d)) irr.push_back({d, m});
    // ascending (d, mask): first divisor found per s is the answer, but we
    // simply take min over all, identical result
    for (const Irr &p : irr) {
        u32 xr = gf_pow(2, r, p.mask, p.d);
        u32 xs = 1;
        u64 packed = ((u64)p.d << 48) | p.mask;
        for (u64 s = 1; s <= smax; s++) {
            xs = gf_mulx(xs, p.mask, p.d);
            if ((xr ^ xs ^ 1u) == 0 && packed < best[s]) best[s] = packed;
        }
    }
}

// ---------------------------------------------------------------------------
static std::string fmt_line(u64 s, u64 packed) {
    char buf[64];
    int d = (int)(packed >> 48);
    u64 mask = packed & 0xFFFFFFFFFFFFULL;
    snprintf(buf, sizeof buf, "%" PRIu64 " %d p%" PRIx64, s, d, mask);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 1; }

    if (!strcmp(argv[1], "selftest")) {
        struct Case { u64 r; int depth; u64 smax; } cases[] = {
            {136279841, 11, 30000},
            {74207281, 10, 20000},
            {101, 9, 100},          // tiny r exercises r < M paths
        };
        for (auto c : cases) {
            std::vector<u64> a, b;
            run_sieve(c.r, c.depth, c.smax, a);
            brute_force(c.r, c.depth, c.smax, b);
            u64 bad = 0;
            for (u64 s = 1; s <= c.smax; s++)
                if (a[s] != b[s] && ++bad <= 5)
                    fprintf(stderr, "MISMATCH r=%" PRIu64 " s=%" PRIu64
                            ": sieve=%s oracle=%s\n", c.r, s,
                            a[s] == UINT64_MAX ? "-" : fmt_line(s, a[s]).c_str(),
                            b[s] == UINT64_MAX ? "-" : fmt_line(s, b[s]).c_str());
            printf("selftest r=%" PRIu64 " depth=%d smax=%" PRIu64 ": %s"
                   " (%" PRIu64 " mismatches)\n",
                   c.r, c.depth, c.smax, bad ? "FAIL" : "PASS", bad);
            if (bad) return 1;
        }
        return 0;
    }

    if (!strcmp(argv[1], "run") && argc >= 5) {
        u64 r = strtoull(argv[2], 0, 10);
        int depth = atoi(argv[3]);
        u64 smax = strtoull(argv[4], 0, 10);
        std::vector<u64> best;
        run_sieve(r, depth, smax, best);
        for (u64 s = 1; s <= smax; s++)
            if (best[s] != UINT64_MAX) puts(fmt_line(s, best[s]).c_str());
        return 0;
    }

    if (!strcmp(argv[1], "validate") && argc >= 6) {
        u64 r = strtoull(argv[2], 0, 10);
        int depth = atoi(argv[3]);
        u64 smax = strtoull(argv[4], 0, 10);
        FILE *f = fopen(argv[5], "r");
        if (!f) { perror("logfile"); return 1; }

        std::vector<u64> best;
        double t0 = now_s();
        run_sieve(r, depth, smax, best);
        fprintf(stderr, "sieve total: %.2fs\n", now_s() - t0);

        // parse log lines: s d p<hex...>  (mask may be very long; we only
        // need full masks when d <= depth, which always fit in 48 bits)
        u64 checked_deep = 0, checked_shallow = 0, bad = 0;
        char *line = nullptr; size_t cap = 0;
        std::vector<char> mybuf;
        while (getline(&line, &cap, f) > 0) {
            u64 s; int d; int off = 0;
            if (sscanf(line, "%" SCNu64 " %d p%n", &s, &d, &off) < 2 || !off)
                continue;
            if (s < 1 || s > smax) continue;
            if (d <= depth) {
                checked_shallow++;
                u64 mask = strtoull(line + off, 0, 16);
                u64 packed = ((u64)d << 48) | mask;
                if (best[s] != packed) {
                    if (++bad <= 10)
                        fprintf(stderr, "MISMATCH s=%" PRIu64 ": log=%d p%"
                                PRIx64 " sieve=%s\n", s, d, mask,
                                best[s] == UINT64_MAX ? "(none)"
                                    : fmt_line(s, best[s]).c_str());
                }
            } else {
                checked_deep++;
                if (best[s] != UINT64_MAX) {
                    if (++bad <= 10)
                        fprintf(stderr, "FALSE POSITIVE s=%" PRIu64 ": log "
                                "says smallest degree %d, sieve found %s\n",
                                s, d, fmt_line(s, best[s]).c_str());
                }
            }
        }
        free(line);
        fclose(f);
        printf("validate r=%" PRIu64 " depth=%d: %" PRIu64 " lines with d<=%d"
               " matched exactly, %" PRIu64 " lines with d>%d confirmed "
               "unmarked, %" PRIu64 " mismatches => %s\n",
               r, depth, checked_shallow, depth, checked_deep, depth, bad,
               bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    }

    fprintf(stderr, "bad arguments\n");
    return 1;
}

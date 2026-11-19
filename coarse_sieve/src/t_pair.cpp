// unit test: gf2f_pair_role partitions the Lyndon words of length d into
// pairs {u, rep(~u)} with exactly one canonical member, plus self-reciprocal
// singletons; gf2_irreducible_count matches the enumeration; and (round 7)
// the roles equal round 6's (gf2f_pair_role_ref).
#include <cstdio>
#include "gf2_dlog.h"
int main() {
    int fails = 0;
    for (int d = 2; d <= 24; d++) {
        const u64 M = ((u64)1 << d) - 1;
        u64 nly = 0, n1 = 0, n0 = 0, n2 = 0, bad = 0;
        for (u64 u = 1; u < ((u64)1 << (d - 1)) || (d == 2 && u == 1); u += 2) {
            if (!gf2f_min_rotation_ref(u, d, M)) continue;
            // aperiodic?
            bool aper = true;
            for (int i = 1; i < d; i++) if (d % i == 0) {
                u64 x = ((u << i) | (u >> (d - i))) & M;
                if (x == u) { aper = false; break; }
            }
            if (!aper) continue;
            nly++;
            const int role = gf2f_pair_role(u, d, M);
            const u64 pr = gf2f_min_rotation(~u & M, d, M);   // partner's rep
            const int prole = gf2f_pair_role(pr, d, M);
            if (role != gf2f_pair_role_ref(u, d, M) || prole != gf2f_pair_role_ref(pr, d, M)) bad++;
            if (role == 1) { n1++; if (prole != 0 || pr == u) bad++; }
            else if (role == 0) { n0++; if (prole != 1 || pr == u) bad++; }
            else { n2++; if (pr != u) bad++; }
            if (d == 2) break;
        }
        const u64 I = gf2_irreducible_count(d);
        // self-reciprocal irreducibles of even degree 2m: (1/2m) sum_{e|m, e odd} mu(e) 2^(m/e)
        long long sr = 0;
        if (d % 2 == 0) {
            const int m = d / 2;
            for (int e = 1; e <= m; e += 2) {
                if (m % e) continue;
                int mu = 1, x = e;
                for (int p = 2; p * p <= x; p++) { if (x % p) continue; x /= p; if (x % p == 0) { mu = 0; break; } mu = -mu; }
                if (mu && x > 1) mu = -mu;
                sr += mu * (long long)((u64)1 << (m / e));
            }
            sr /= d;
        }
        const bool ok = !bad && nly == I && n1 == n0 && (long long)n2 == sr;
        if (!ok) fails++;
        printf("d=%2d Lyndon %8llu (I(d) %8llu) canonical %7llu skipped %7llu self-recip %4llu (formula %lld) %s\n",
               d, (unsigned long long)nly, (unsigned long long)I, (unsigned long long)n1,
               (unsigned long long)n0, (unsigned long long)n2, sr, ok ? "ok" : "FAIL");
    }
    // I(d) up to 55 against the necklace recursion sum_{e|d} e*I(e) = 2^d
    for (int d = 1; d <= 55; d++) {
        unsigned __int128 s = 0;
        for (int e = 1; e <= d; e++) if (d % e == 0) s += (unsigned __int128)e * (e == 1 ? 2 : gf2_irreducible_count(e));
        if (s != ((unsigned __int128)1 << d)) { printf("I(%d) recursion FAIL\n", d); fails++; }
    }
    printf("I(49) = %llu, I(50) = %llu\n", (unsigned long long)gf2_irreducible_count(49), (unsigned long long)gf2_irreducible_count(50));
    printf(fails ? "FAILURES\n" : "ALL OK\n");
    return fails != 0;
}

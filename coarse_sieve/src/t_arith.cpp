#include "gf2_dlog.h"
#include "gf2_wide_setup.h"
#include <random>
#include <cstdio>
int main() {
    std::mt19937_64 rng(11); unsigned long long n = 0, bad = 0;
    for (int d = 2; d <= 47; d++) {
        u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        if (!gf2f_usable(q, d)) { printf("d=%d not usable\n", d); bad++; continue; }
        u64 gp[64]; gp[0] = 2; for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t-1], q, d);
        u64 edge[] = {0, 1, 2, M, M - 1, M >> 1, (u64)1 << (d - 1), 0x5555555555555555ULL & M};
        for (u64 a : edge) for (u64 b : edge) { n++;
            if (gf2f_mul(a, b, ql, d) != gf_mul(a, b, q, d)) bad++; }
        for (int k = 0; k < 20000; k++) {
            u64 a = rng() & M, b = rng() & M; n += 3;
            if (gf2f_mul(a, b, ql, d) != gf_mul(a, b, q, d)) bad++;
            if (gf2f_sqr(a, ql, d) != gf_sqr(a, q, d)) bad++;
            u64 x = rng() % M, y = rng() % M;
            if (gf2f_mulmod_m(x, y, d, M) != mulmod_u64(x, y, M)) bad++;
        }
        for (int k = 0; k < 200; k++) { u64 i = rng() % M; n++;
            u64 ref = 1; for (int t = 0; t < d; t++) if (i >> t & 1) ref = gf_mul(ref, gp[t], q, d);
            if (gf2f_antilog(i, ql, d, gp) != ref) bad++; }
        // powmod: a^phi == 1 for a coprime to M (spot check via inverse)
        for (int k = 0; k < 50; k++) { u64 a = 1 + rng() % (M - 1); if (gcd_u64(a, M) != 1) continue;
            u64 ph = 1, mm = M; for (u64 p : factor_u64(M)) { u64 pk = 1; while (mm % p == 0) { mm /= p; pk *= p; } ph *= pk / p * (p - 1); }
            n++; if (gf2f_mulmod_m(a, gf2f_powmod_m(a, ph - 1, d, M), d, M) != 1 % M) bad++; }
        // Carmichael exponent (what the sieve uses): a * a^(lambda-1) == 1
        { u64 lam = 1, mm = M; for (u64 p : factor_u64(M)) { u64 pk = 1; while (mm % p == 0) { mm /= p; pk *= p; } u64 ph = pk / p * (p - 1); lam = lam / gcd_u64(lam, ph) * ph; }
          for (int k = 0; k < 2000; k++) { u64 a = 1 + rng() % (M > 1 ? M - 1 : 1); if (gcd_u64(a, M) != 1) continue;
            n++; if (gf2f_mulmod_m(a, gf2f_powmod_m(a, lam - 1, d, M), d, M) != 1 % M) bad++; } }
    }
    for (u64 p : {3ULL, 7ULL, 31ULL, 127ULL, 5419ULL, 2099863ULL, 164511353ULL}) {
        OddDiv o = odd_div_make(p);
        for (int k = 0; k < 100000; k++) { u64 x = rng() >> (rng() % 40), qq = 0; n++;
            bool t = odd_div_test(x, o, &qq);
            if (t != (x % p == 0) || (t && qq != x / p)) bad++; }
    }
    printf("fast arithmetic vs reference, d=2..47: %llu checks, %llu mismatches\n", n, bad);
    return bad != 0;
}

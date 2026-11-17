// proto_ph_dlp.cpp -- reference prototype (coarse_sieve round 2 roadmap).
// Discrete logs in GF(2^d) by Pohlig-Hellman: tables only as large as the
// largest prime factor of 2^d - 1.  Validates d = 42 (prime powers 3^2, 7^2),
// 43 and 45 on random samples.
// Build: g++ -O2 -std=c++17 -I. -o proto_ph_dlp proto_ph_dlp.cpp
//        (needs gf2_wide_field.h, gf2_wide_setup.h next to it)
#include "gf2_wide_field.h"
#include "gf2_wide_setup.h"
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdio>
static double now() { return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct PF { u64 p; int k; u64 pk; u64 gam, gi, giinv;   // gam: order-p generator
            std::vector<std::pair<u64,u32>> tab; };      // sorted (gam^t, t)
static u64 lookup(const PF &f, u64 y) {
    auto it = std::lower_bound(f.tab.begin(), f.tab.end(), std::make_pair(y, (u32)0));
    return (it != f.tab.end() && it->first == y) ? it->second : ~0ULL;
}
int main() {
    for (int d : {42, 43, 45}) {
        const u64 M = ((u64)1 << d) - 1, q = find_primitive_poly(d);
        std::vector<PF> fs; u64 rest = M;
        for (u64 p : factor_u64(M)) { PF f{}; f.p = p; f.k = 0; f.pk = 1;
            while (rest % p == 0) { rest /= p; f.k++; f.pk *= p; } fs.push_back(f); }
        double t0 = now(); u64 tabmax = 0;
        for (auto &f : fs) {
            f.gi = gf_pow(2, M / f.pk, q, d);                 // order p^k
            f.giinv = gf_pow(f.gi, f.pk - 1, q, d);
            f.gam = gf_pow(f.gi, f.pk / f.p, q, d);           // order p
            f.tab.resize(f.p); u64 y = 1;
            for (u64 t = 0; t < f.p; t++, y = gf_mul(y, f.gam, q, d)) f.tab[t] = {y, (u32)t};
            std::sort(f.tab.begin(), f.tab.end()); tabmax = std::max(tabmax, f.p);
        }
        double tb = now() - t0;
        auto dlog = [&](u64 e) -> u64 {
            u64 x = 0, mod = 1;                               // running CRT
            for (auto &f : fs) {
                u64 ei = gf_pow(e, M / f.pk, q, d), xi = 0, ppow = 1;
                for (int j = 0; j < f.k; j++) {               // base-p digits
                    // strip known digits: (ei * gi^-xi)^(p^(k-1-j)) has order p
                    u64 z = gf_mul(ei, gf_pow(f.giinv, xi, q, d), q, d);
                    u64 pw = 1; for (int t = 0; t < f.k - 1 - j; t++) pw *= f.p;
                    u64 dj = lookup(f, gf_pow(z, pw, q, d));
                    if (dj == ~0ULL) return ~0ULL;
                    xi += dj * ppow; ppow *= f.p;
                }
                // CRT: x == x mod 'mod', x == xi mod pk
                u64 inv = inv_mod(mod % f.pk, f.pk);
                u64 tt = mulmod_u64((xi + f.pk - x % f.pk) % f.pk, inv, f.pk);
                x += mod * tt; mod *= f.pk;
            }
            return x % M;
        };
        std::mt19937_64 rng(d); u64 gp[64]; gp[0] = 2;
        for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
        const int N = 2000; u64 bad = 0; double t1 = now();
        for (int i = 0; i < N; i++) {
            u64 u = rng() % M, e = 1, b = u; int t = 0;
            while (b) { if (b & 1) e = gf_mul(e, gp[t], q, d); b >>= 1; t++; }
            if (dlog(e) != u) bad++;
        }
        printf("d=%d  2^d-1 =", d);
        for (auto &f : fs) { printf(" %llu", (unsigned long long)f.p); if (f.k > 1) printf("^%d", f.k); }
        printf("\n      largest table %llu entries (built %.2fs); %d random logs: %llu wrong; "
               "%.0f us/log on 1 CPU core (bit-serial gf_mul)\n",
               (unsigned long long)tabmax, tb, N, (unsigned long long)bad, (now() - t1) / N * 1e6);
    }
}

// t_cubic.cpp: the cubic tower (round 9) against ground truth.
//   t_cubic [dmax] [samples]
// For every 6 | d in 12..dmax: build the cubic structure (cubic_build_host),
// fill T3 by an independent host Pohlig-Hellman (BSGS per prime; lazily, so
// only the entries the checked x need at large d), and check cubic_log(x) ==
// log x -- for every x at d <= 18, for `samples` random x = g^u above --
// and that the sieve's compile-time cubic_parts<d> agrees with the checks'
// cubic_parts_rt on every one of them.
// Build: g++ -O2 -march=native -std=c++17 -o t_cubic t_cubic.cpp
#include "gf2_dlog.h"
#include <random>
#include <cstdio>
#include <chrono>
#include <unordered_map>
static std::mt19937_64 rng(11);

// log_g(x) in GF(2^d) (g = x mod q) by Pohlig-Hellman over the prime powers
// of M, BSGS per prime -- shares nothing with the device code
struct HostPH {
    int d; u64 q, M;
    std::vector<u64> pp, pr;  // prime powers p^e || M, and p
    std::vector<int> ex;
    std::vector<std::unordered_map<u64, u32>> baby;   // per prime: gamma^j -> j
    std::vector<u64> gam, giant, bm;
    void init(int d_, u64 q_) {
        d = d_; q = q_; M = ((u64)1 << d) - 1;
        u64 rest = M;
        for (u64 p : factor_u64(M)) {
            int e = 0; u64 pe = 1;
            while (rest % p == 0) { rest /= p; e++; pe *= p; }
            pp.push_back(pe); pr.push_back(p); ex.push_back(e);
            const u64 gp = gf_pow(2, M / p, q, d);          // order p
            u64 m = 1; while (m * m < p) m++;
            std::unordered_map<u64, u32> b;
            u64 t = 1;
            for (u64 j = 0; j < m; j++) { b.emplace(t, (u32)j); t = gf_mul(t, gp, q, d); }
            baby.push_back(std::move(b));
            gam.push_back(gp);
            giant.push_back(gf_pow(gp, (p - m % p) % p, q, d));  // gp^(-m)
            bm.push_back(m);
        }
    }
    u64 dlog_p(size_t i, u64 h) {                      // h in <gam[i]>
        u64 y = h;
        for (u64 s = 0; s <= bm[i]; s++) {
            auto it = baby[i].find(y);
            if (it != baby[i].end()) return (s * bm[i] + it->second) % pr[i];
            y = gf_mul(y, giant[i], q, d);
        }
        return ~0ULL;
    }
    u64 log(u64 x) {
        u64 v = 0;
        for (size_t i = 0; i < pp.size(); i++) {
            const u64 pe = pp[i], p = pr[i], m = M / pe;
            const u64 xp = gf_pow(x, m, q, d), gp = gf_pow(2, m, q, d);   // order p^e
            u64 l = 0, pk = 1;
            for (int k = 0; k < ex[i]; k++) {
                // (xp * gp^-l)^(p^(e-1-k)) = gamma^digit
                u64 t = gf_mul(xp, gf_pow(gp, (pe - l % pe) % pe, q, d), q, d);
                u64 pw = 1; for (int z = 0; z < ex[i] - 1 - k; z++) pw *= p;
                t = gf_pow(t, pw, q, d);
                const u64 dg = dlog_p(i, t);
                if (dg == ~0ULL) return ~0ULL;
                l += dg * pk; pk *= p;
            }
            const u64 C = M / pe;                       // CRT
            v = (v + mulmod_u64(mulmod_u64(C, inv_mod(C % pe, pe), M), l, M)) % M;
        }
        return v;
    }
};

// the sieve's version (d a template parameter) at run-time d
static u64 parts_ct(int d, u64 x, const TowerDev &t, u64 *ir, u64 *it) {
    switch (d) {
    case 12: return cubic_parts<12>(x, t, ir, it);
    case 18: return cubic_parts<18>(x, t, ir, it);
    case 24: return cubic_parts<24>(x, t, ir, it);
    case 30: return cubic_parts<30>(x, t, ir, it);
    case 36: return cubic_parts<36>(x, t, ir, it);
    case 42: return cubic_parts<42>(x, t, ir, it);
    case 48: return cubic_parts<48>(x, t, ir, it);
    case 54: return cubic_parts<54>(x, t, ir, it);
    default: *ir = *it = ~0ULL; return ~0ULL;
    }
}

int main(int argc, char **argv) {
    const int dmax = argc > 1 ? atoi(argv[1]) : 54;
    const u64 samples = argc > 2 ? strtoull(argv[2], nullptr, 10) : 3000;
    u64 bad = 0, nchk = 0;
    for (int d = 12; d <= dmax; d += 6) {
        auto t0 = std::chrono::steady_clock::now();
        const u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        CubicHost C;
        if (!cubic_build_host(d, q, C)) { printf("d=%d: build FAILED\n", d); bad++; continue; }
        HostPH ph; ph.init(d, q);
        // T3, lazily: T3[l] = log(w + y'^l), T3[MK] = log w, T3[MK + 1] = 0
        std::vector<u64> T3((size_t)C.MK + 2, ~0ULL);
        T3[C.MK] = C.logw;
        T3[C.MK + 1] = 0;
        if (ph.log(C.w) != C.logw) { printf("d=%d: log w mismatch\n", d); bad++; continue; }
        if (gf_pow(C.kappa, C.c, q, d) != C.yp) { printf("d=%d: y' != kappa^c\n", d); bad++; continue; }
        auto fill = [&](u64 l) {
            if (T3[l] != ~0ULL) return;
            T3[l] = ph.log(C.w ^ gf_pow(C.yp, l, q, d));
        };
        const bool full = C.MK <= 4096;
        if (full) for (u64 l = 0; l < C.MK; l++) fill(l);
        TowerDev t = cubic_dev(C, C.btab.data(), T3.data(), C.kt.btab.data(),
                               C.kt.LK.data(), C.kT.data());
        u64 gp[64]; gp[0] = 2;
        for (int k = 1; k < d; k++) gp[k] = gf_sqr(gp[k - 1], q, d);
        const bool exh = d <= 18;
        const u64 cnt = exh ? M : samples;
        u64 b = 0, x = 1, nk = 0, bct = 0;
        for (u64 i = 0; i < cnt; i++) {
            const u64 u = exh ? i : rng() % M;
            if (!exh) x = gf2f_antilog(u, ql, d, gp);
            u64 ir, it, ir2, it2;
            const u64 lb = cubic_parts_rt(x, t, &ir, &it);
            nk += ir == C.MK + 1 && it == C.MK + 1;
            if (!full) { fill(ir); fill(it); }
            if (cubic_log(x, M, t) != u) b++;
            if (parts_ct(d, x, t, &ir2, &it2) != lb || ir2 != ir || it2 != it) bct++;
            if (exh) x = gf_mulx(x, q, d);
        }
        // and the elements of K* (x = phi(Y0), the inK branch) on a sample
        u64 bk = 0;
        for (int i = 0; i < 200; i++) {
            const u64 l = rng() % C.MK, u = mulmod_u64(C.H, l, M);
            const u64 xk = gf2f_antilog(u, ql, d, gp);
            if (cubic_log(xk, M, t) != u) bk++;
            u64 ir, it, ir2, it2;
            if (parts_ct(d, xk, t, &ir2, &it2) != cubic_parts_rt(xk, t, &ir, &it) ||
                ir2 != ir || it2 != it || ir != C.MK + 1 || it != C.MK + 1) bk++;
        }
        nchk += cnt + 200; bad += b + bk + bct;
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("d=%2d n=%2d c=%6llu T3=%7llu: %8llu logs (%s), %llu wrong, %llu with "
               "cubic_parts<d> != cubic_parts_rt; K* %llu wrong (%.1fs)\n",
               d, C.n, (unsigned long long)C.c, (unsigned long long)C.MK + 2,
               (unsigned long long)cnt, exh ? "all" : "sampled", (unsigned long long)b,
               (unsigned long long)bct, (unsigned long long)bk, s);
    }
    printf("%s: %llu logs, %llu wrong\n", bad ? "FAILURES" : "ALL OK",
           (unsigned long long)nchk, (unsigned long long)bad);
    return bad != 0;
}

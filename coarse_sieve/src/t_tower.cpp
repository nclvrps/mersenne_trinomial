// t_tower.cpp: classic and subfield towers vs ground truth, every n
#include "gf2_dlog.h"
#include <random>
#include <cstdio>
#include <chrono>
static std::mt19937_64 rng(7);

// fill T on the host exactly like k_tower_T (coupon collector over u)
static bool fill_T(int d, u64 q, TowerHost &W, std::vector<u64> &T, TowerDev &t) {
    const u64 M = ((u64)1 << d) - 1, MK = W.MK;
    T.assign((size_t)MK + 1, ~0ULL);
    t = tower_dev(W, W.btab.data(), W.LK.data(), T.data(), W.LKn.data(), W.TP.data());
    u64 left = MK + 1, x = 1;
    for (u64 u = 0; u < M && left; u++, x = gf_mulx(x, q, d)) {
        u64 lb, l;
        if (!tower_split(tower_coords(x, t), t, &lb, &l)) continue;
        if (T[l] == ~0ULL) {
            u64 h = (MK + 2) * lb;
            T[l] = u >= h ? u - h : u + M - h;
            left--;
        }
    }
    return left == 0;
}

int main(int argc, char **argv) {
    int dmax = argc > 1 ? atoi(argv[1]) : 26;
    int dmin = argc > 2 ? atoi(argv[2]) : 4;
    int allmax = argc > 3 ? atoi(argv[3]) : 32;    // all n choices up to here
    u64 bad = 0, nchk = 0;
    for (int d = dmin; d <= dmax; d += 2) {
        const int m = d / 2;
        const u64 q = find_primitive_poly(d), ql = q ^ ((u64)1 << d), M = ((u64)1 << d) - 1;
        u64 gp[64]; gp[0] = 2;
        for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
        std::vector<int> wants = {0, -1};
        if (d <= allmax) for (int n = 2; n < m; n++) if (m % n == 0) wants.push_back(n);
        for (int want : wants) {
            auto t0 = std::chrono::steady_clock::now();
            TowerHost W;
            if (!tower_build_host(d, q, W, want)) { printf("d=%d want=%d: build FAILED\n", d, want); bad++; continue; }
            if (want > 0 && W.sub_n != want) { printf("d=%d want=%d: got n=%d\n", d, want, W.sub_n); continue; }
            std::vector<u64> T; TowerDev t;
            if (!fill_T(d, q, W, T, t)) { printf("d=%d want=%d: T incomplete\n", d, want); bad++; continue; }
            const bool exh = d <= 20;
            const u64 cnt = exh ? M : 20000;
            u64 b = 0, e = 1;
            for (u64 i = 0; i < cnt; i++) {
                u64 u = exh ? i : rng() % M;
                if (!exh) e = gf2f_antilog(u, ql, d, gp);
                if (tower_log(e, M, t) != u) b++;
                if (exh) e = gf_mulx(e, q, d);
            }
            nchk += cnt; bad += b;
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("d=%2d %-9s n=%2d k=%2d TP=%7zu: %8llu logs, %llu wrong (%.1fs)\n", d,
                   want == 0 ? "classic" : want < 0 ? "auto" : "forced", W.sub_n, W.sub_k, W.TP.size(),
                   (unsigned long long)cnt, (unsigned long long)b, s);
        }
    }
    printf("%s: %llu logs, %llu wrong\n", bad ? "FAILURES" : "ALL OK", (unsigned long long)nchk, (unsigned long long)bad);
    return bad != 0;
}

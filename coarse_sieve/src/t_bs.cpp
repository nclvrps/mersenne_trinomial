// t_bs.cpp: unit tests for the round-3 pieces of gf2_dlog.h
//   bit-sliced arithmetic vs scalar, transposes, FKM vs brute force,
//   compile-time moduli vs find_primitive_poly, planner output
#include "gf2_dlog.h"
#include <random>
#include <cstdio>
#include <set>
#include <chrono>

static std::mt19937_64 rng(42);
static u64 nchk = 0, nbad = 0;
#define CHECK(c, ...) do { nchk++; if (!(c)) { if (nbad++ < 20) { printf(__VA_ARGS__); printf("\n"); } } } while (0)

template <int D> static void test_bs() {
    const u64 q = find_primitive_poly(D), ql = q ^ ((u64)1 << D), M = ((u64)1 << D) - 1;
    CHECK(ql == gf2_qlow(D), "D=%d: gf2_qlow 0x%llx != find_primitive_poly 0x%llx", D,
          (unsigned long long)gf2_qlow(D), (unsigned long long)ql);
    for (int rep = 0; rep < 40; rep++) {
        u64 a[32], b[32], r[32];
        for (int k = 0; k < 32; k++) { a[k] = rng() & M; b[k] = rng() & M; }
        if (rep == 0) { a[0] = 0; a[1] = 1; a[2] = M; b[3] = 0; b[4] = 1; b[5] = M; }
        u32 A[D], B[D], R[D];
        bs_load<D>(a, A); bs_load<D>(b, B);
        // load/store round trip and plane semantics
        bs_store<D>(A, r);
        for (int k = 0; k < 32; k++) CHECK(r[k] == a[k], "D=%d load/store k=%d", D, k);
        for (int i = 0; i < D; i++) for (int k = 0; k < 32; k += 7)
            CHECK(((A[i] >> k) & 1) == ((a[k] >> i) & 1), "D=%d plane %d slot %d", D, i, k);
        bs_mul<D>(A, B, R); bs_store<D>(R, r);
        for (int k = 0; k < 32; k++) CHECK(r[k] == gf_mul(a[k], b[k], q, D), "D=%d mul k=%d", D, k);
        bs_sqr<D>(A, R); bs_store<D>(R, r);
        for (int k = 0; k < 32; k++) CHECK(r[k] == gf_sqr(a[k], q, D), "D=%d sqr k=%d", D, k);
        u32 sel = (u32)rng(), Gm[D];
        for (int i = 0; i < D; i++) Gm[i] = A[i];
        bs_mulx_sel<D>(Gm, sel); bs_store<D>(Gm, r);
        for (int k = 0; k < 32; k++)
            CHECK(r[k] == ((sel >> k) & 1 ? gf_mulx(a[k], q, D) : a[k]), "D=%d mulx_sel k=%d", D, k);
        // g^u for u < 2^(D-1)
        u64 u[32];
        for (int k = 0; k < 32; k++) u[k] = (rng() & M) >> 1;
        if (rep == 1) { u[0] = 0; u[1] = 1; u[2] = M >> 1; }
        u32 U[D], G[D];
        bs_load<D>(u, U); bs_pow_g<D>(U, G); bs_store<D>(G, r);
        for (int k = 0; k < 32; k++) CHECK(r[k] == gf_pow(2, u[k], q, D), "D=%d pow_g k=%d u=%llx", D, k, (unsigned long long)u[k]);
        // X^E, uniform E
        u64 E = 1 + rng() % M;
        if (rep == 2) E = 1;
        if (rep == 3) E = M - 1;
        u32 P[D];
        bs_pow<D>(A, E, P); bs_store<D>(P, r);
        for (int k = 0; k < 32; k++) CHECK(r[k] == gf_pow(a[k], E, q, D), "D=%d pow k=%d", D, k);
    }
}

template <int D> struct RunAll { static void go() { RunAll<D - 1>::go(); test_bs<D>(); } };
template <> struct RunAll<1> { static void go() {} };

// brute-force: odd u < 2^(n-1) that is a minimal rotation and aperiodic
static bool is_lyndon_ref(u64 u, int n) {
    const u64 M = ((u64)1 << n) - 1;
    if (!(u & 1) || u >= ((u64)1 << (n - 1))) return false;
    u64 x = u;
    for (int i = 1; i < n; i++) {
        x = ((x << 1) | (x >> (n - 1))) & M;
        if (x <= u) return false;          // smaller, or equal (periodic)
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

static void test_fkm() {
    for (int n = 2; n <= 22; n++) {
        std::vector<u64> got, ref;
        fkm_range(0, (u64)1 << (n - 1), n, got);
        for (u64 u = 1; u < ((u64)1 << (n - 1)); u += 2) if (is_lyndon_ref(u, n)) ref.push_back(u);
        CHECK(got == ref, "FKM n=%d full: %zu vs %zu", n, got.size(), ref.size());
        // split into random sub-ranges: the union must be the same
        std::vector<u64> cuts = {0, (u64)1 << (n - 1)};
        for (int i = 0; i < 20; i++) cuts.push_back(rng() % ((u64)1 << (n - 1)));
        std::sort(cuts.begin(), cuts.end());
        std::vector<u64> got2;
        for (size_t i = 0; i + 1 < cuts.size(); i++) fkm_range(cuts[i], cuts[i + 1], n, got2);
        CHECK(got2 == ref, "FKM n=%d split", n);
    }
    for (int n = 23; n <= 55; n++) {
        for (int t = 0; t < 30; t++) {
            // ranges at every density: near 2^(n-1-z) for z = 1..n-2
            int z = 1 + (int)(rng() % (u64)(n - 2));
            u64 base = ((u64)1 << (n - 1 - z)) + (rng() & (((u64)1 << (n - 1 - z)) - 1));
            u64 span = (u64)1 << (8 + rng() % 9);
            u64 lo = base, hi = std::min<u64>(base + span, (u64)1 << (n - 1));
            std::vector<u64> got, ref;
            fkm_range(lo, hi, n, got);
            for (u64 u = lo | 1; u < hi; u += 2) if (is_lyndon_ref(u, n)) ref.push_back(u);
            CHECK(got == ref, "FKM n=%d [%llx,%llx): %zu vs %zu", n, (unsigned long long)lo,
                  (unsigned long long)hi, got.size(), ref.size());
        }
    }
}

int main(int argc, char **argv) {
    auto t0 = std::chrono::steady_clock::now();
    RunAll<55>::go();
    printf("bit-sliced arithmetic D=2..55: %llu checks, %llu bad\n", (unsigned long long)nchk, (unsigned long long)nbad);
    u64 c0 = nchk, b0 = nbad;
    test_fkm();
    printf("FKM vs brute force: %llu checks, %llu bad\n", (unsigned long long)(nchk - c0), (unsigned long long)(nbad - b0));
    if (argc > 1) {
        for (int d = 21; d <= 55; d++) {
            const u64 q = find_primitive_poly(d);
            for (int bs = 0; bs < 2; bs++) {
                PHHost H;
                auto ta = std::chrono::steady_clock::now();
                bool ok = ph_plan_host(d, q, (size_t)5500 << 20, H, bs ? kPHCostBitsliced : kPHCostScalar);
                double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
                printf("d=%2d %s %s", d, bs ? "bitsliced" : "scalar   ", ok ? "" : "INFEASIBLE");
                if (ok) {
                    for (int i = 0; i < H.nparts; i++)
                        printf(" %llu(pc %d top %d G %llu)", (unsigned long long)H.part[i].P,
                               __builtin_popcountll(H.part[i].E), 63 - __builtin_clzll(H.part[i].E),
                               (unsigned long long)H.part[i].G);
                    printf(" | %.0f MB, est %.0f", H.table_bytes / 1048576.0, H.cost);
                }
                printf("  (%.2fs)\n", tp);
            }
        }
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%s (%.1fs)\n", nbad ? "FAILURES" : "ALL OK", t);
    return nbad != 0;
}

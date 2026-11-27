/*
Copyright (C) 2026  Gord Palameta
(actual author: Claude AI)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Some code may have been inspired by:
  GF2X library
  factor.cpp and other files in GF2X library apps subdirectory
  NTL library
*/

// tsfactor_cpu.cpp
// tsfactor without a GPU (round 10): the same program -- the same command
// line, checkpoints, results files, interval schedule, certified gcd and
// error checking -- with the scan arithmetic on the CPU (ts_cpu.h).  Built
// with g++ alone; no CUDA headers or libraries anywhere.
//
// Meant for the CPU stage of the pipeline (coarse_sieve's survivors up to
// d ~ 279 on cheap cloud cores), where it replaces factor_ec: one process
// per core by default (--gcd-threads 1 --pend-max 1, one thread inside each
// gcd, one squaring thread), so N copies use N cores.  A single copy on an
// otherwise idle machine can use more: --threads N (squaring), --gcd-mt N,
// --pend-max 2.
//
// Build:   make tsfactor_cpu
// Test:    ./tsfactor_cpu --selftest
// Bench:   ./tsfactor_cpu --bench 136279841 588 [--bench-gcd]
// Run:     ./tsfactor_cpu 136279841 survivors.txt --skip 55 --maxd 279
//              --out stage2      (one command line)

#define GF2_EC_HOST_ONLY
#include "ts_core.h"
#include "ts_cpu.h"

static int g_cpu_threads = 1;

static std::unique_ptr<TsDev> make_cpu_dev(u64 r, int m, int variant) {
    std::unique_ptr<TsCPU> G(new TsCPU());
    G->threads = g_cpu_threads;
    G->init(r);
    G->set_m(m);
    G->force_generic = (variant == 1);
    return std::unique_ptr<TsDev>(G.release());
}

static std::string cpu_model_name() {
    std::string name = "unknown CPU";
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return name;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *c = strchr(line, ':');
            if (c) {
                c++;
                while (*c == ' ' || *c == '\t') c++;
                size_t n = strlen(c);
                while (n && (c[n - 1] == '\n' || c[n - 1] == ' ')) c[--n] = 0;
                name = c;
            }
            break;
        }
    }
    fclose(f);
    return name;
}

static int cpu_bench(u64 r, u64 s) {
    printf("bench r=%" PRIu64 " s=%" PRIu64 " (CPU build, %d squaring "
           "thread%s)\n", r, s, g_cpu_threads, g_cpu_threads > 1 ? "s" : "");
    TsCPU G;
    G.threads = g_cpu_threads;
    G.init(r);
    G.set_m(12);
    G.set_s(s);
    auto timeit = [&](const char *nm, int reps, auto fn) -> double {
        fn();                                   // warm-up
        double t0 = trin_now_s();
        for (int i = 0; i < reps; i++) fn();
        double dt = (trin_now_s() - t0) / reps;
        printf("  %-34s %10.3f ms\n", nm, dt * 1e3);
        return dt;
    };
    for (int j = 0; j < G.m; j++) {
        std::vector<u64> v = st_res(r);
        memcpy(G.slot[j], v.data(), G.nw * 8);
    }
    char nm[80];
    snprintf(nm, sizeof nm, "squaring (BLZ, s %s)",
             G.g.path == 1 ? "odd" : G.g.path == 2 ? "even" : "small: generic");
    double t_one = timeit(nm, 4, [&] {
        G.sq(G.slot[0], G.spare[0], G.T2);
        std::swap(G.slot[0], G.spare[0]); });
    {
        const int sv = G.g.path;
        G.g.path = 0;
        timeit("squaring (generic spread+fold)", 2, [&] {
            G.sq(G.slot[0], G.spare[0], G.T2);
            std::swap(G.slot[0], G.spare[0]); });
        G.g.path = sv;
    }
    double t_sq = timeit("h-table advance (m=12 x 12)", 1,
                         [&] { G.square_steps((u64)G.m, nullptr); }) /
                  ((double)G.m * G.m);
    printf("  %-34s %10.3f ms\n", "  = per squaring", t_sq * 1e3);
    (void)t_one;
    double t_hor = timeit("Horner (m=12)", 3, [&] { G.horner_C(); });
    std::vector<u64> a = st_res(r);
    memcpy(G.A, a.data(), G.nw * 8);
    double t_mul = timeit("modmul (gf2x product + reduce)", 2,
                          [&] { G.mul_A_by_C(); });
    timeit("  product only", 2, [&] { G.full_product(G.A, G.C, G.T2); });
    timeit("  fold reduce (2 passes)", 3, [&] {
        G.reduce(G.T2, G.W); });
    G.ec_setup(true);
    G.ec_chain_reset();
    printf("error checking (per block unless noted):\n");
    double te_pre = timeit("EC h-table pass (m=12)", 3,
                           [&] { G.ec_block_pre(); });
    double te_fold = timeit("EC product + result folds", 3, [&] {
        TsCPU::fold3(G.T2, 2 * G.nw, r, 2 * r - s, G.seg_);
        ecf_fold_host(G.fAn_, G.A, G.nw); });
    double te_win = timeit("EC chain check (per window)", 2,
                           [&] { G.ec_w = 1; G.ec_window_check(); });
    G.ec_on = false;
    printf("per-degree projection (m*t_sq + (t_mul + t_horner(m))/m), and "
           "the error-checking overhead at --ecw %ld:\n", g_ec.window);
    int best_m = 4;
    double best_c = 1e100;
    for (int m : {4, 6, 8, 10, 12, 14, 16, 20, 24, 32}) {
        double c = m * t_sq + (t_mul + t_hor * m / 12.0) / m;
        double ce = (te_pre * m / 12.0 + te_fold +
                     te_win * m / 12.0 / (double)g_ec.window) / m;
        printf("  m=%-3d  %8.2f ms/degree   + EC %6.3f ms (%.2f%%)\n", m,
               c * 1e3, ce * 1e3, 100.0 * ce / c);
        if (c < best_c) { best_c = c; best_m = m; }
    }
    int mo = (int)(sqrt(t_mul / t_sq) + 0.5);
    printf("recommended m: %d (analytic %d); expect ~%.1f ms/degree\n",
           best_m, mo, best_c * 1e3);
    G.fini();
    return 0;
}

struct CpuPlatform : TsPlatform {
    const char *prog() const override { return "tsfactor_cpu"; }
    const char *build() const override { return "CPU build"; }
    std::string fingerprint() override {
        char b[176];
        snprintf(b, sizeof b, "tsfactor_cpu %s on %s (%u CPUs, %d thread%s)",
                 TS_ROUND_STR, cpu_model_name().c_str(),
                 std::thread::hardware_concurrency(), g_cpu_threads,
                 g_cpu_threads > 1 ? "s" : "");
        return b;
    }
    bool parse_opt(const std::string &a, int &i, int argc,
                   char **argv) override {
        if (a == "--threads") {
            if (i + 1 >= argc) { fprintf(stderr, "--threads needs a value\n");
                                 exit(1); }
            g_cpu_threads = atoi(argv[++i]);
            if (g_cpu_threads < 1) g_cpu_threads = 1;
#ifndef _OPENMP
            if (g_cpu_threads > 1) {
                fprintf(stderr, "note: --threads ignored (built without "
                        "OpenMP)\n");
                g_cpu_threads = 1;
            }
#endif
            return true;
        }
        if (a == "--device" || a == "--fft" || a == "--legacy-sq") {
            fprintf(stderr, "%s: GPU option; this is the CPU build\n",
                    a.c_str());
            exit(1);
        }
        return false;
    }
    const char *usage_extra() const override {
        return "CPU build:\n"
               "  --threads N     threads for the h-table squarings (default\n"
               "                  1: run one process per core instead)\n"
               "  Defaults differ from the GPU build: --gcd-threads 1\n"
               "  --pend-max 1 and one thread inside each gcd, so that one\n"
               "  process uses one core.\n";
    }
    TsDev *make_dev() override {
        TsCPU *g = new TsCPU();
        g->threads = g_cpu_threads;
        return g;
    }
    void defaults(TsTune &T) override { T.measured = 0; }
    void tune(TsDev &G, TsTune &T, int verbose) override {
        (void)verbose;                  // ts_main prints the outcome
        ts_measure_m(*this, G, T);
        T.measured = 1;
    }
    void apply(const TsTune &T) override { (void)T; }
    std::string describe(const TsTune &T) override {
        char b[160];
        snprintf(b, sizeof b, "m=%" PRIu64 " (t_sq %.3f ms per slot, t_mul "
                 "%.1f ms)", T.m, T.t_sq * 1e3, T.t_mul * 1e3);
        return b;
    }
    int choose_m(double t_sq, double t_mul) override {
        int mo = (int)(sqrt(t_mul / t_sq) + 0.5);
        return mo < 4 ? 4 : mo > 64 ? 64 : mo;
    }
    int selftest(int gcd_threads) override {
        int bad = 0;
        printf("tsfactor_cpu selftest (%s, CPU build)\n", TS_ROUND_STR);
        printf("GCD backend: %s\n", gcd_backend());
        bad += cpu_kernel_suites();
        bad += selftest_sched();
        TsTestEnv E;
        E.name = "CPU";
        E.make = make_cpu_dev;
        bad += selftest_table_suites(E);
        bad += selftest_scan_suites(E, gcd_threads);
        selftest_footer(bad);
        return bad ? 1 : 0;
    }
    int bench(u64 r, u64 s, bool bench_gcd, bool gpu_gcd, u64 min_bits,
              bool ecg_xgcd, bool gcd_ntl) override {
        (void)gpu_gcd; (void)min_bits;
        int rc = cpu_bench(r, s);
        if (!rc && bench_gcd) ts_bench_gcd(r, s, ecg_xgcd, gcd_ntl);
        return rc;
    }
    u64 default_G0() const override { return 300; }
    void default_threads(int &gcd_threads, int &pend_max) override {
        gcd_threads = 1;
        pend_max = 1;
    }
    int default_gcd_mt() const override { return 1; }
    void mem_report(TsDev &Gd, const SParams &P, bool gpu_gcd) override {
        (void)gpu_gcd;
        TsCPU &G = static_cast<TsCPU &>(Gd);
        double mb = 1.0 / (1024 * 1024);
        double tab = (double)(G.m + G.nspare_alloc) * G.stride * 8 * mb;
        double scr = 14.0 * G.stride * 8 * mb;
        double snap = (double)G.m * G.nw * 8 * mb;
        long nsn = (P.hsnap_mb <= 0) ? 0 : (long)(P.hsnap_mb / snap);
        if (P.hsnap_mb > 0 && nsn < 1) nsn = 1;
        fprintf(stderr, "memory: h-table %.0f MB, scratch %.0f MB, snapshots "
                "%.0f MB each (up to %ld, --hsnap-mb %ld)\n", tab, scr, snap,
                nsn, P.hsnap_mb);
    }
};

int main(int argc, char **argv) {
    CpuPlatform PL;
    return ts_main(argc, argv, PL);
}

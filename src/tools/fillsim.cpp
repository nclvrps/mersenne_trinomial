// fillsim.cpp -- lockstep simulation of the batched kernel's fill loop
// (coarse_sieve round 7)
//
//   g++ -O2 -march=native -std=c++17 -I.. -o fillsim fillsim.cpp
//   ./fillsim d chunk [fill [warps [threads blocks [nopair]]]]
//     chunk    the chunk index, or -1 for the tuner's dense sample
//     fill     0 round 6 (each lane its own batch, runs t + k*stride),
//              1 the same with rotated runs, 2 pooled with rotated runs
//              (COARSE_FILL=0..2; 3, runs from a counter, depends on timing
//              and is not modelled)
//     warps    how many warps of the launch to simulate, drawn at random
//              (fixed seed) from all of it (default 64 of 640; evenly
//              spaced warps would see runs whose bits above the lane share
//              a pattern, and so a biased sample of the chunk)
//     threads blocks   the launch (default 512 x 40: tower and RR kernels
//              on a 40-SM card; PH runs 128 x 80)
//     nopair   1: every class processed, as in round 5
//
// Each warp's 32 lanes run k_logs_batch's fill loop in lockstep: an
// iteration is one pass of the loop body for every lane still filling, and
// the warp processes its batches together, once no lane is filling.  The
// counts, per class credited (a canonical class credits 2, a self-
// reciprocal one 1):
//   lane-steps   fill iterations x 32: what the warp spends on enumeration,
//                idle lanes included (FKM steps alone would be ~1-2)
//   slots used   batch slots holding a class (the rest is the batches'
//                padding: lanes with nothing to process, issuing anyway)
// and the warps' imbalance (each warp's fill iterations + 1000 x its batch
// calls, max over mean) and, for fills 0-1, how many iterations had a lane
// at a tie and at round 6's d-step tie-break loop.
#include "gf2_dlog.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

typedef unsigned long long ull;
#define LB 32

static u64 dense_stretch(int d, int L) {    // coarse_sieve's logs_dense_stretch
    const int hb = d - L;
    const int z = L / 2 < hb - 1 ? L / 2 : hb - 1;
    u64 H = 0;
    for (int b = hb - z - 1; b >= 0; b -= 2) H |= (u64)1 << b;
    return H << L;
}

struct Lane {
    bool have = false, done = false, pend = false;
    u64 w = 0, whi = 0, rr = 0, k = 0;
    int p = 0, n = 0;
};

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "fillsim d chunk [fill [warps [threads blocks [nopair]]]]\n");
        return 1;
    }
    const int d = atoi(argv[1]);
    const long long chunk_ix = atoll(argv[2]);
    const int fill = argc > 3 ? atoi(argv[3]) : 0;
    const int wsim = argc > 4 ? atoi(argv[4]) : 64;
    const int tpb = argc > 6 ? atoi(argv[5]) : 512;
    const int blocks = argc > 6 ? atoi(argv[6]) : 40;
    const bool nopair = argc > 7 && atoi(argv[7]) != 0;
    const bool rot = fill >= 1, pool = fill >= 2;
    const u64 M = ((u64)1 << d) - 1;
    const u64 total = (u64)1 << (d - 1);
    u64 run = 1u << 14;
    while (run > 64 && total / run < ((u64)1 << 17)) run >>= 1;
    u64 chunk = run * (u64)(1u << 17);
    if (chunk > total) chunk = total;
    u64 ulo, uhi;
    if (chunk_ix < 0) {
        const int lb = d - 4 < 27 ? d - 4 : 27;
        ulo = dense_stretch(d, lb);
        uhi = ulo + ((u64)1 << lb);
    } else {
        ulo = (u64)chunk_ix * chunk;
        uhi = ulo + chunk;
    }
    const u64 krun = run >= 512 ? run / 8 : run;
    const u64 nruns = (uhi - ulo + krun - 1) / krun;
    const u64 stride = (u64)tpb * blocks;
    const u64 nwarps = stride / 32;
    ull it_fill = 0, batches = 0, processed = 0, credited = 0, lyndon = 0,
        steps = 0, it_tie = 0, it_scan = 0;
    std::vector<double> wcost;
    auto run_at = [&](u64 rr, u64 k) -> u64 {          // logs_run_at
        if (!rot || (rr | 31) >= nruns) return rr;
        return (rr & ~(u64)31) | ((rr + k) & 31);
    };
    // one pass of the fill loop's body for a lane; returns the role of the
    // class it found (0: none)
    auto step = [&](Lane &L, bool &tie, bool &scan) -> int {
        if (!L.have) {
            if (L.rr >= nruns) { L.done = true; return 0; }
            const u64 lo = ulo + run_at(L.rr, L.k) * krun;
            L.whi = uhi - lo > krun ? lo + krun : uhi;
            L.rr += stride; L.k++;
            L.w = fkm_seek(lo, d, &L.p);
            L.have = true;
        }
        if (L.w >= L.whi) { L.have = false; return 0; }
        steps++;
        int role = 0;
        if (L.p == d) {
            lyndon++;
            role = nopair ? 2 : gf2f_pair_role(L.w, d, M);
            if (!nopair && 2 * gf2f_popc64(L.w) == d) {
                tie = true;
                const int z0 = gf2f_clz64(L.w) - (64 - d);
                int z1 = 0;
                for (u64 x = L.w; x; x &= x >> 1) z1++;
                if (z0 == z1) scan = true;
            }
            if (role) credited += role == 1 ? 2 : 1;
        }
        L.w = fkm_next(L.w, d, &L.p);
        return role;
    };
    std::vector<u64> pick(nwarps);
    for (u64 i = 0; i < nwarps; i++) pick[i] = i;
    std::mt19937_64 rng(0xF111517ULL + (u64)chunk_ix);
    std::shuffle(pick.begin(), pick.end(), rng);
    const int nsim = wsim < (int)nwarps ? wsim : (int)nwarps;
    for (int wi = 0; wi < nsim; wi++) {
        const u64 wid = pick[wi];
        const ull it0 = it_fill, b0 = batches;
        Lane ln[32];
        for (int l = 0; l < 32; l++) ln[l].rr = wid * 32 + l;
        for (;;) {
            if (!pool) {
                // every lane until its batch is full or its runs are done
                for (;;) {
                    bool any = false, tie = false, scan = false;
                    for (int l = 0; l < 32; l++) {
                        Lane &L = ln[l];
                        if (L.done || L.n >= LB) continue;
                        any = true;
                        if (step(L, tie, scan)) L.n++;
                    }
                    if (!any) break;
                    it_fill++;
                    if (tie) it_tie++;
                    if (scan) it_scan++;
                }
            } else {
                // pooled: the found classes go to the lanes in turn; a class
                // that finds no room waits with its lane
                int dealt = 0;
                for (;;) {
                    bool live = false, tie = false, scan = false;
                    int nf = 0;
                    for (int l = 0; l < 32; l++) {
                        Lane &L = ln[l];
                        if (!L.pend && !L.done && step(L, tie, scan)) L.pend = true;
                        if (!L.done) live = true;
                        if (L.pend) nf++;
                    }
                    it_fill++;
                    if (!nf) { if (!live) break; continue; }
                    const int room = 32 * LB - dealt;
                    const int take = nf < room ? nf : room;
                    int r = 0;
                    for (int l = 0; l < 32 && r < take; l++)
                        if (ln[l].pend) { ln[l].pend = false; ln[(dealt + r) & 31].n++; r++; }
                    dealt += take;
                    if (dealt == 32 * LB) break;
                }
            }
            bool anyb = false;
            for (int l = 0; l < 32; l++) if (ln[l].n) anyb = true;
            if (!anyb) break;
            batches++;
            for (int l = 0; l < 32; l++) { processed += ln[l].n; ln[l].n = 0; }
        }
        wcost.push_back((double)(it_fill - it0) + 1000.0 * (double)(batches - b0));
    }
    double mx = 0, sm = 0;
    for (double c : wcost) { sm += c; if (c > mx) mx = c; }
    printf("d=%d %s %lld, fill %d%s, %d of %llu warps: FKM steps %llu, classes %llu, "
           "processed %llu, credited %llu\n", d, chunk_ix < 0 ? "dense sample" : "chunk",
           chunk_ix, fill, nopair ? " (no pairs)" : "", nsim, (ull)nwarps, steps, lyndon,
           processed, credited);
    printf("  lane-steps per class %.3f; slots used %.1f%%; batch calls %llu; "
           "warp imbalance (max/mean) %.3f", credited ? 32.0 * it_fill / credited : 0.0,
           batches ? 100.0 * processed / (32.0 * LB * batches) : 0.0, batches,
           wcost.empty() ? 0.0 : mx / (sm / wcost.size()));
    if (!pool && it_fill)
        printf("; iterations with a tie %.1f%%, at round 6's d-step loop %.1f%%",
               100.0 * it_tie / it_fill, 100.0 * it_scan / it_fill);
    printf("\n");
    return 0;
}

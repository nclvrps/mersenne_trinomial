// cstat.cpp -- per-chunk class statistics of coarse_sieve's pass (round 7)
//
//   g++ -O2 -march=native -std=c++17 -I.. -o cstat cstat.cpp -lpthread
//   ./cstat d pos_lo pos_hi [threads]
//
// For the pass positions pos_lo..pos_hi-1 of degree d (the chunk geometry
// and the scrambled order of coarse_sieve.cu: chunk_at), enumerates every
// Lyndon word of the chunk by FKM and prints, per position, the chunk, the
// FKM steps, the classes (Lyndon words), the canonical ones (pair role 1),
// the ties (as many ones as zeros) and the ties that reach the run scan
// (leading zero run = longest run of ones).  Every class's pair role is
// computed both by gf2f_pair_role (round 7) and gf2f_pair_role_ref (round
// 6), and any difference is counted (exit status 1 if there is one).
//
// Used for: which chunks hold the work (68-73% of the chunks hold no class
// at d = 44-47; the profiling positions of the round-7 handoff), the
// class counts behind the warp simulation (fillsim.cpp), and the check of
// the round-7 tie-break against round 6's (all of d = 2..36, 4.5e10
// classes of d=44, 9.3e10 of d=47).
#include "gf2_dlog.h"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>

typedef unsigned long long ull;

// coarse_sieve.cu's pass order (round 6, unchanged in round 7)
#define CHUNK_BLOCK 8
static u64 chunk_at(int p, int nchunks) {
    if (nchunks <= CHUNK_BLOCK || nchunks % CHUNK_BLOCK) return (u64)p;
    const u64 nb = (u64)(nchunks / CHUNK_BLOCK);
    if (nb & (nb - 1)) return (u64)p;
    int k = 0;
    while (((u64)1 << k) < nb) k++;
    const u64 b = (u64)(p / CHUNK_BLOCK);
    u64 x = 0;
    for (int i = 0; i < k; i++) x |= ((b >> i) & 1) << (k - 1 - i);
    u64 y = x;
    for (int i = k - 1; i >= 0; i--) {
        u64 h = ((x >> (i + 1)) + 1) * 0x9E3779B97F4A7C15ULL + (u64)i * 0xD1B54A32D192ED03ULL;
        h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 29;
        y ^= (h >> 63) << i;
    }
    return y * CHUNK_BLOCK + (u64)(p % CHUNK_BLOCK);
}

// the number of rotations the round-7 tie-break compares (0: no scan)
static int scan_count(u64 u, int d) {
    if (2 * gf2f_popc64(u) != d) return 0;
    const int z0 = gf2f_clz64(u) - (64 - d);
    u64 y = gf2f_zero_runs(u, z0);
    if (!y || (y & (y >> 1))) return 0;
    return gf2f_popc64(y);
}

struct St {
    ull steps = 0, lyn = 0, r1 = 0, r0 = 0, r2 = 0, tie = 0, zeq = 0, bad = 0;
    ull cand[8] = {0};
};

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "cstat d pos_lo pos_hi [threads]\n"); return 1; }
    const int d = atoi(argv[1]);
    const int plo = atoi(argv[2]), phi = atoi(argv[3]);
    const int nth = argc > 4 ? atoi(argv[4]) : 2;
    const u64 M = ((u64)1 << d) - 1;
    // coarse_sieve.cu's logs_geometry
    const u64 total = (u64)1 << (d - 1);
    u64 run = 1u << 14;
    while (run > 64 && total / run < ((u64)1 << 17)) run >>= 1;
    u64 chunk = run * (u64)(1u << 17);
    if (chunk > total) chunk = total;
    const int nchunks = (int)((total + chunk - 1) / chunk);
    if (plo < 0 || phi > nchunks || plo >= phi) {
        fprintf(stderr, "positions %d..%d: d=%d has %d chunks\n", plo, phi - 1, d, nchunks);
        return 1;
    }
    std::vector<St> per(phi - plo);
    std::atomic<int> next(plo);
    auto work = [&]() {
        for (;;) {
            const int p = next++;
            if (p >= phi) return;
            St s;
            const u64 lo = chunk_at(p, nchunks) * chunk, hi = lo + chunk;
            int pp;
            u64 w = fkm_seek(lo, d, &pp);
            while (w < hi) {
                s.steps++;
                if (pp == d) {
                    s.lyn++;
                    const int rn = gf2f_pair_role(w, d, M);
                    if (rn != gf2f_pair_role_ref(w, d, M)) s.bad++;
                    if (rn == 1) s.r1++; else if (rn == 0) s.r0++; else s.r2++;
                    if (2 * gf2f_popc64(w) == d) {
                        s.tie++;
                        const int nc = scan_count(w, d);
                        if (nc) { s.zeq++; s.cand[nc < 7 ? nc : 7]++; }
                    }
                }
                if (w == M) break;
                w = fkm_next(w, d, &pp);
            }
            per[p - plo] = s;
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) th.emplace_back(work);
    for (auto &t : th) t.join();
    St T;
    for (int i = 0; i < phi - plo; i++) {
        const St &s = per[i];
        printf("pos %6d chunk %8llu steps %11llu lyndon %11llu canon %11llu tie %10llu zeq %9llu\n",
               plo + i, (ull)chunk_at(plo + i, nchunks), s.steps, s.lyn, s.r1, s.tie, s.zeq);
        T.steps += s.steps; T.lyn += s.lyn; T.r1 += s.r1; T.r0 += s.r0; T.r2 += s.r2;
        T.tie += s.tie; T.zeq += s.zeq; T.bad += s.bad;
        for (int k = 0; k < 8; k++) T.cand[k] += s.cand[k];
    }
    printf("TOTAL d=%d nchunks %d positions %d..%d: steps %llu lyndon %llu (%.3f steps/word) "
           "canon %llu skip %llu self %llu; tie %llu (%.4f of words), z0==z1 %llu (%.4f of ties); "
           "candidates:", d, nchunks, plo, phi - 1, T.steps, T.lyn,
           T.lyn ? (double)T.steps / T.lyn : 0.0, T.r1, T.r0, T.r2, T.tie,
           T.lyn ? (double)T.tie / T.lyn : 0.0, T.zeq, T.tie ? (double)T.zeq / T.tie : 0.0);
    for (int k = 1; k < 8; k++) printf(" %d:%llu", k, T.cand[k]);
    printf("; MISMATCHES round 7 / round 6 roles %llu\n", T.bad);
    return T.bad != 0;
}

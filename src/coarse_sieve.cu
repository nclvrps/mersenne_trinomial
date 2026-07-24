// coarse_sieve.cu
// GPU coarse sieve for smallest-degree irreducible factors of the
// trinomials  x^r + x^s + 1  over GF(2), all s in [1, smax], factor
// degrees d <= depth.
//
// This is a port of sieve_ref.cpp, which was validated two ways:
//   (1) against an independent brute-force oracle (direct irreducible
//       enumeration + per-polynomial evaluation) for three r values;
//   (2) against verified `factor` output for r = 136279841: all 999,999
//       lines of the s < 10^6 log reproduced exactly at depth 27
//       (935,227 factors matched in degree AND mask including every
//       tie-break; 64,772 deeper lines confirmed as sieve survivors).
// The device field arithmetic and the per-class number theory come from
// the same gf2_small_field.h header the reference uses.
//
// Method (see sieve_ref.cpp for the full derivation): for each degree d,
// work in GF(2^d) with a primitive polynomial q_d so g = x generates the
// multiplicative group of order M = 2^d - 1.  A degree-d irreducible p
// with root beta = g^j divides x^r + x^s + 1  iff  j*s == Z(j*r mod M)
// (mod M) where Z is the Zech logarithm -- an arithmetic progression in
// s of modulus n = M / gcd(j, M).  Only the minimal j of each cyclotomic
// coset is processed (conjugates give the same factor and progression);
// proper-subfield elements are skipped.  Each progression is marked into
// best[s] with atomicMin on the packed key (d << 48) | mask, which
// implements exactly "smallest degree, then lexicographically least
// mask" (equal-degree masks have equal hex length, so lexicographic
// order on the printed string is numeric order on the mask).
//
// Output: <prefix>.survivors.txt  s values with no factor of degree <=
//                                 depth (the deep-search worklist)
//         <prefix>.found.bin      (optional, --found) little-endian u64
//                                 packed keys for s = 1..smax, one per
//                                 s, UINT64_MAX where no factor found;
//                                 mask = key & 0xFFFFFFFFFFFF, d = key>>48
//
// Build:  nvcc -O3 -arch=native -I../common -o coarse_sieve coarse_sieve.cu
//         (narrow build: u32 field, fast, depths d <= 32)
// Wide:   nvcc -O3 -arch=native -DWIDE_FIELD -I../common
//              -o coarse_sieve_wide coarse_sieve.cu
//         (u64 field via gf2_wide_field.h, depths d <= 63; slower per op,
//          and needed for any d >= 33 since field elements and log values
//          no longer fit in 32 bits.  Use the narrow binary for d <= 32.)
// Run:    ./coarse_sieve <r> <depth> <out_prefix>   (smax defaults to floor(r/2))
//                        [--found] [--no-survivors]
//                        [--load <file.found.bin>] [--min-depth <D>]
//                        [--checkpoint <file>] [--checkpoint-mins <N>]
//                        [--resume <file>]
//                        [--selftest] [--selftest-lowmem [nbuckets]]
// Extend: --load seeds best[] from a prior run's .found.bin and --min-depth
//         starts the sieve at degree D, so you compute only the new
//         degrees.  Recommended for d>=33: run the FAST narrow binary to
//         depth 32 with --found, then run the wide binary for just d=33:
//           ./coarse_sieve      <r> 32 <smax> out32 --found
//           ./coarse_sieve_wide <r> 33 <smax> out33 --found \
//                               --load out32.found.bin --min-depth 33
//         This avoids re-doing d<=32 in the slow u64 build (a degree-d
//         factor only ever improves an s with no smaller-degree factor, so
//         seeding from the depth-32 result and adding d=33 is exact).
// Checkpoint/resume: for long runs on reclaimable (spot) instances, add
//         --checkpoint <file> to save progress on SIGTERM/SIGINT -- and,
//         with --checkpoint-mins <N>, every N minutes.  In low-memory mode
//         the save happens at bucket boundaries (mid-degree), so at most one
//         bucket is redone; on SIGTERM the in-flight bucket is abandoned
//         within ~100ms so the ~30s reclaim budget is met.  Resume with
//         --resume <file> (which keeps checkpointing to the same file):
//           ./coarse_sieve_wide <r> 34 <smax> out34 --found \
//               --load out33.found.bin --min-depth 34 \
//               --checkpoint out34.ckpt --checkpoint-mins 20
//           # on reclaim, just:
//           ./coarse_sieve_wide <r> 34 <smax> out34 --found --resume out34.ckpt
//         Low-memory runs print a per-bucket progress bar ('=' for buckets
//         already done on resume, '.' as each new bucket completes).
// Env:    COARSE_FORCE_BUCKETS=N forces the bucketed path with N buckets for
//         every degree (manual override / testing).
// Degrees: narrow build supports d <= 32 (elements fit u32, modulus u64);
//         the -DWIDE_FIELD build supports d <= 63 (u64 elements, 128-bit
//         products).  The survivor list uses a 32-bit counter, so d <= 37
//         until that is widened.  Memory in the wide build: at d=33 the
//         low-memory path fits ~5.5 GB on an 8 GB card (64 buckets) or
//         ~12.5 GB on a 16 GB card (8 buckets); d=34 needs 16 GB.
// Outputs: <prefix>.survivors.txt is written by default (--no-survivors
//         suppresses it); it is just the list of s with no factor of
//         degree <= depth, which is exactly the set of s missing from
//         <prefix>.found.bin, so it can be reconstructed from --found.
// Sizing: the dense per-degree tables need 8*2^d bytes at degree d
//         (8.6 GB at d=30, 17.2 GB at d=31) plus 8*(smax+1) bytes for
//         best[] (545 MB at smax = 68,139,920).  The program checks
//         free device memory before each degree and, once the dense
//         tables no longer fit, automatically switches to a low-memory
//         bucketed mode (see "LOW-MEMORY MODE" below) that trades
//         runtime for memory instead of simply stopping.
// Selftest: --selftest runs r=136279841 depth=20 smax=999999 entirely
//         on the GPU and compares every best[] entry against the CPU
//         reference computed in-process.  --selftest-lowmem does the
//         same but forces the bucketed low-memory path (with the given
//         number of buckets, default 4) for every degree regardless of
//         whether the dense tables would have fit, so the low-memory
//         kernels get exercised and checked even at small, fast depths.
//
// LOW-MEMORY MODE
// ----------------
// The dense path keeps two full-size tables resident per degree:
//   dA[i]  = g^i               (antilog table, size 2^d)
//   dL[v]  = log_g(v)           (Zech/discrete-log table, size 2^d)
// Both are only there to avoid recomputing g^i on demand. Two
// observations remove the need to size either of them as O(2^d):
//
//  1. dA is never actually needed as a stored array. Every place that
//     reads A[x] for a specific x can instead compute g^x directly
//     from the small (size d) gpow2[] chain (same technique k_antilog
//     already uses internally), and the per-class minimal-polynomial
//     root sequence A[j], A[2j mod M], A[4j mod M], ... is just
//     repeated squaring of a single seed value g^j (Frobenius: squaring
//     a field element doubles its discrete log). So k_build_lslice
//     and k_resolve_slice below compute antilogs on the fly from gpow2[]
//     (tiny, O(d) bytes) and never store dA at all.
//
//  2. dL is only read once per surviving class (the single Zech-log
//     lookup). That means it doesn't need to exist all at once: the
//     value range of L can be split into `nbuckets` slices, and each
//     slice built and consumed in its own pass:
//       For bucket b covering value range [lo, hi): build only dL's
//       slice over that range (k_build_lslice), then resolve exactly
//       the survivors belonging to that slice (k_resolve_slice).
//
//  3. The survivor list need not be stored at all, because the L slice
//     already indexes its own bucket's survivors. A class j targets
//     x(j) = antilog(j*rmod mod M) ^ 1, and each slice is forced
//     even-aligned, hence closed under XOR 1, so x(j) is in [lo,hi)
//     exactly when antilog(j*rmod mod M) is -- which is exactly the
//     condition under which lslice holds that element's log. Walking
//     the slice slots and mapping each stored log u back through
//     j = u * rmod^-1 mod M therefore enumerates this bucket's
//     survivors precisely: every survivor of the degree lands in
//     exactly one bucket, and each class is filtered exactly once
//     across all buckets. Total filter work is what the old separate
//     survivor pass cost, but nothing survivor-sized stays resident,
//     so peak memory now depends only on nbuckets -- and any degree
//     can be made to fit by using more buckets.
//
// Peak memory drops from 8*2^d bytes to just
//     sizeof(fw)*ceil(2^d / nbuckets)       (one L-slice, nothing else)
// so peak memory is bounded solely by the bucket count, at the cost of
// `nbuckets` full-domain passes to rebuild each L-slice (runtime scales
// roughly linearly in nbuckets, so use the fewest buckets that fit).
// run_gpu() picks the smallest nbuckets that fits free device memory
// automatically; --selftest-lowmem lets you force a specific value.

#if defined(WIDE_FIELD)
  #include "gf2_wide_field.h"
  #include "gf2_wide_setup.h"
#else
  #include "gf2_small_field.h"
  #include "gf2_field_setup.h"
#endif
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <vector>
#include <string>
#include <chrono>
#include <csignal>
#include <ctime>
#include <unistd.h>

// fw ("field word") is the type of everything sized like a field element
// or an index into the 2^d-element group: field elements, antilog/log
// values, class indices j, the key x, the mask M, and the L-table entries.
// Narrow build: u32 (d <= 32).  -DWIDE_FIELD build: u64 (d <= 63).  The
// modulus q is always u64.  Loop counters, grid/block dims, degree d,
// bucket counts and heavy-queue indices stay u32/unsigned.
#if defined(WIDE_FIELD)
typedef u64 fw;
#else
typedef u32 fw;
#endif

#define CUCHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
            __FILE__, __LINE__); exit(1); } } while (0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

typedef unsigned long long ull;

// heavy-progression record for the second-pass marking kernel
struct Heavy { u64 s_first, n, packed; };

// classes whose progressions have more than this many marks are deferred
// to k_mark_heavy (one thread block per record) for load balance
#define INLINE_MARKS 1024
#define HEAVY_CAP    (1u << 22)

// (a*b) mod m for index arithmetic.  In the narrow build a,b < 2^32 so the
// product fits u64 and a plain multiply is used; in the wide build a,b can
// reach 2^d-1 with d up to 63, so the product exceeds 64 bits and the
// double-and-add mulmod_u64 is required.
#if defined(WIDE_FIELD)
#define MULMOD(a, b, m) mulmod_u64((u64)(a), (u64)(b), (u64)(m))
#else
#define MULMOD(a, b, m) (((u64)(a) * (u64)(b)) % (u64)(m))
#endif

// ---------------------------------------------------------------------------
// kernels
// antilog table: A[i] = g^i computed from the precomputed squares
// gpow2[t] = g^(2^t); i's set bits select which squares to multiply.
__global__ void k_antilog(fw *A, u64 M64, u64 q, int d, const fw *gpow2) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < M64; i += stride) {
        fw v = 1;
        u64 bits = i;
        int t = 0;
        while (bits) {
            if (bits & 1) v = gf_mul(v, gpow2[t], q, d);
            bits >>= 1;
            t++;
        }
        A[i] = v;
    }
}

__global__ void k_logscatter(const fw *A, fw *L, u64 M64) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < M64; i += stride) L[A[i]] = (fw)i;
}

// one thread per candidate j; does the full per-class computation and
// either marks inline or defers to the heavy queue
__global__ void k_classes(int d, u64 M64, u64 q, u64 rmod, u64 smax,
                          const fw *A, const fw *L,
                          const u64 *subq, int nsub,
                          ull *best, Heavy *heavy, unsigned *nheavy,
                          unsigned *overflow) {
    const fw M = (fw)M64;
    u64 j64 = (u64)blockIdx.x * blockDim.x + threadIdx.x + 1;
    u64 stride = (u64)gridDim.x * blockDim.x;
    fw P[64];                                     // minpoly scratch

    for (; j64 < M64; j64 += stride) {
        fw j = (fw)j64;
        // skip proper-subfield elements
        bool skip = false;
        for (int k = 0; k < nsub; k++)
            if (j64 % subq[k] == 0) { skip = true; break; }
        if (skip) continue;
        // only the minimal element of each cyclotomic coset
        fw jj = j;
        bool minimal = true;
        for (int i = 1; i < d; i++) {
            jj = rotl_d(jj, d, M);
            if (jj < j) { minimal = false; break; }
        }
        if (!minimal) continue;

        u64 u = MULMOD(j64, rmod, M64);
        if (u == 0) continue;                     // beta^r = 1: no solutions
        fw v = L[A[u] ^ 1u];                       // Zech logarithm

        u64 gg = gcd_u64(j64, M64);
        if ((u64)v % gg != 0) continue;
        u64 n = M64 / gg;
        u64 s0 = MULMOD(inv_mod(j64 / gg, n), (u64)v / gg, n);

        u64 mask = minpoly_mask(j, d, M, q, A, P);
        // mask == 0 would mean non-GF(2) coefficients: impossible if the
        // tables are right; validated exhaustively on CPU.
        u64 packed = ((u64)d << 48) | mask;

        u64 s = (s0 == 0 ? n : s0);
        if (s > smax) continue;
        u64 count = (smax - s) / n + 1;
        if (count <= INLINE_MARKS) {
            for (; s <= smax; s += n) atomicMin(&best[s], (ull)packed);
        } else {
            unsigned idx = atomicAdd(nheavy, 1u);
            if (idx < HEAVY_CAP) heavy[idx] = {s, n, packed};
            else atomicAdd(overflow, 1u);
        }
    }
}

// one block per heavy record; threads stride the progression
__global__ void k_mark_heavy(const Heavy *heavy, unsigned nheavy, u64 smax,
                             ull *best) {
    unsigned rec = blockIdx.x;
    if (rec >= nheavy) return;
    u64 s0 = heavy[rec].s_first, n = heavy[rec].n;
    ull packed = (ull)heavy[rec].packed;
    for (u64 s = s0 + (u64)threadIdx.x * n; s <= smax; s += (u64)blockDim.x * n)
        atomicMin(&best[s], packed);
}

__global__ void k_fill(ull *p, u64 n, ull v) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = v;
}

__global__ void k_fill_fw(fw *p, u64 n, fw v) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = v;
}

// ---------------------------------------------------------------------------
// LOW-MEMORY (bucketed) path -- see the "LOW-MEMORY MODE" note at the top
// of the file. Never materializes a full antilog table; the log table is
// built and consumed one value-range slice at a time.

// one surviving class: its index j and its Zech-log query key x = A[u]^1.
// fw is u32 in the narrow build (8 bytes/survivor) and u64 in the wide
// build (16 bytes/survivor, since j and x range up to 2^d-1 for d up to 63).

// g^idx computed on the fly from the (tiny, size-d) gpow2 chain, i.e.
// gpow2[t] = g^(2^t). Identical arithmetic to k_antilog's inner loop,
// just usable for one arbitrary index instead of the whole table.
GF2_HD fw d_antilog(u64 idx, u64 q, int d, const fw *gpow2) {
    fw v = 1;
    u64 bits = idx;
    int t = 0;
    while (bits) {
        if (bits & 1) v = gf_mul(v, gpow2[t], q, d);
        bits >>= 1;
        t++;
    }
    return v;
}

// Same result as minpoly_mask(), but seeded with a single root value
// (g^j) and advanced by repeated squaring instead of indexing a stored
// antilog table -- squaring a field element doubles its discrete log,
// which is exactly the A[rotl_d(...)] step the table-based version used
// A[] for.
GF2_HD u64 d_minpoly_seeded(fw seed, int d, u64 q, fw *P) {
    for (int k = 0; k <= d; k++) P[k] = 0;
    P[0] = 1;
    fw root = seed;
    for (int i = 0; i < d; i++) {
        for (int k = i + 1; k >= 1; k--)
            P[k] = P[k - 1] ^ gf_mul(root, P[k], q, d);
        P[0] = gf_mul(root, P[0], q, d);
        root = gf_sqr(root, q, d);
    }
    u64 mask = 0;
    for (int k = 0; k <= d; k++) {
        if (P[k] > 1) return 0;
        mask |= (u64)P[k] << k;
    }
    return mask;
}

// Phase A: identical filtering to k_classes (subfield skip, cyclotomic-
// coset minimality), but stops at computing the Zech-log query key --
// no L table exists yet, and no A table is ever stored. One thread per
// candidate j; survivors are stream-compacted into `surv` via atomicAdd.

// Phase B, step 1: build only the L slice covering value range [lo, hi).
// Same computation as k_antilog+k_logscatter fused together, just
// filtered to one output range instead of writing a full-size table.
__global__ void k_build_lslice(int d, u64 M64, u64 q, const fw *gpow2,
                               u64 lo, u64 hi, fw *lslice) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < M64; i += stride) {
        fw Ai = d_antilog(i, q, d, gpow2);
        if ((u64)Ai >= lo && (u64)Ai < hi) lslice[(u64)Ai - lo] = (fw)i;
    }
}

// Phase B, step 2: resolve whichever stashed survivors have their query
// key x in the current bucket's range against this slice, then continue
// exactly as k_classes' tail does (gcd/inv_mod/minpoly/progression mark).
// Fused survivor-enumeration + resolve for one bucket.  This replaces both
// the stored survivor list and the old k_resolve_bucket scan.
//
// The key observation is that the L slice already IS the survivor index for
// its own bucket.  A class j is a survivor when it passes the subfield and
// coset-minimality filters, and its target is
//        x(j) = antilog(j * rmod mod M) ^ 1.
// Because every slice [lo,hi) is even-aligned (chunk is forced even), the
// range is closed under XOR 1, so x(j) lands in this bucket exactly when
// antilog(j * rmod mod M) does.  After k_build_lslice, that condition is
// precisely "lslice[e - lo] holds the log of e" for e in [lo,hi).
//
// So we can walk the slots instead of a list: slot t corresponds to element
// e = lo + t, u = lslice[t] is its discrete log, and j = u * rmod^-1 mod M
// recovers the unique class that could target this slot.  Testing that j
// reproduces exactly this bucket's survivors -- each survivor of the degree
// belongs to exactly one bucket, and each class is tested exactly once
// across all buckets, so total filter work is unchanged while the survivor
// list disappears completely.  x's own log, needed below, is the sibling
// slot t^1, since lo is even.
__global__ void k_resolve_slice(int d, u64 M64, u64 q, u64 rmodinv,
                                const u64 *subq, int nsub, const fw *gpow2,
                                u64 lo, u64 hi, u64 smax, const fw *lslice,
                                ull *best, Heavy *heavy, unsigned *nheavy,
                                unsigned *overflow, unsigned long long *nsurv) {
    const fw M = (fw)M64;
    const fw EMPTY = (fw)~(fw)0;
    u64 span = hi - lo;
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    fw P[64];

    for (; t < span; t += stride) {
        fw vu = lslice[t];
        if (vu == EMPTY) continue;            // element 0 has no discrete log
        u64 j64 = MULMOD((u64)vu, rmodinv, M64);
        if (j64 == 0) continue;               // u == 0 <-> j == 0, not a class

        bool skip = false;
        for (int k = 0; k < nsub; k++)
            if (j64 % subq[k] == 0) { skip = true; break; }
        if (skip) continue;
        fw j = (fw)j64;
        fw jj = j;
        bool minimal = true;
        for (int i = 1; i < d; i++) {
            jj = rotl_d(jj, d, M);
            if (jj < j) { minimal = false; break; }
        }
        if (!minimal) continue;
        atomicAdd(nsurv, 1ULL);               // counted where Phase A used to

        fw v = lslice[t ^ 1ULL];              // log of x = (lo+t)^1
        u64 gg = gcd_u64(j64, M64);
        if ((u64)v % gg != 0) continue;
        u64 n = M64 / gg;
        u64 s0 = MULMOD(inv_mod(j64 / gg, n), (u64)v / gg, n);

        u64 s = (s0 == 0 ? n : s0);
        if (s > smax) continue;

        fw seed = d_antilog(j64, q, d, gpow2);            // = A[j]
        u64 mask = d_minpoly_seeded(seed, d, q, P);
        u64 packed = ((u64)d << 48) | mask;

        u64 count = (smax - s) / n + 1;
        if (count <= INLINE_MARKS) {
            for (; s <= smax; s += n) atomicMin(&best[s], (ull)packed);
        } else {
            unsigned idx = atomicAdd(nheavy, 1u);
            if (idx < HEAVY_CAP) heavy[idx] = {s, n, packed};
            else atomicAdd(overflow, 1u);
        }
    }
}


// ---------------------------------------------------------------------------
static void launch_dims(u64 work, int &blocks, int &threads) {
    threads = 256;
    u64 b = (work + threads - 1) / threads;
    blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
}

// ---------------------------------------------------------------------------
// CHECKPOINT / RESUME
//
// Long low-memory runs (d >= 33 can take hours over `nbuckets` passes) need
// to survive a spot-instance reclaim (SIGTERM, ~30s notice) or a manual
// Ctrl-C (SIGINT), and resume without losing the buckets already done.
//
// The checkpoint records best[] plus (degree d, next_bucket): "all degrees
// < d are complete in best[], and for degree d buckets [0, next_bucket) are
// done." A host copy of best[] is refreshed after every completed bucket, so
// on a signal we can persist the last clean bucket boundary immediately and
// abandon the in-flight bucket (it re-runs on resume). The survivor list is
// NOT saved (it can be 4 GB); Phase A is simply recomputed on resume, which
// is deterministic and cheap relative to the bucket passes.
//
// The signal handler only sets a flag. All device queries and file writes
// happen on the main thread at poll points (every ~100 ms during a bucket,
// via cudaStreamQuery), so response is well under the 30s SIGTERM budget
// even though a single d=33 bucket runs a couple of minutes.
// DELTA CHECKPOINTS.  Writing all of best[] is 545 MB at smax=68139920,
// which on network-backed cloud storage took ~110 s -- far beyond the ~30 s
// spot-reclaim budget.  But a high-degree run started with --load changes
// almost nothing: at d=33 only ~112k of 68.1M entries became non-survivors.
// So the checkpoint stores the BASELINE FILE IDENTITY plus just the entries
// that differ from it -- a couple of MB, written in well under a second.
// If the delta ever grows past a quarter of the range (e.g. a from-scratch
// run over many degrees) the writer falls back to a full image
// automatically, so correctness never depends on the delta staying small.
struct CkptCtx {
    bool     enabled   = false;
    const char *path   = nullptr;
    double   period_s  = 0;          // periodic write interval (0 = signal-only)
    double   last_write = 0;
    u64      r = 0, smax = 0;
    int      depth = 0;
    int      keep = 3;               // checkpoint generations to retain
    ull     *d_best = nullptr;       // device best[] (source for snapshots)
    std::vector<ull> snap;           // host snapshot as of last completed bucket
    int      snap_d = 0, snap_next_bucket = 0, snap_nbuckets = 0;
    // baseline for delta encoding (the --load image), empty if none
    std::vector<ull> base;
    std::string base_path;
    int      base_kind = 0;          // 1 = .found.bin image, 2 = survivors list
    u64      base_hash = 0;
    volatile sig_atomic_t stop = 0;  // set by signal handler
    // test hook: force a stop after this many completed buckets (0 = off)
    long     test_stop_after = 0, test_buckets_done = 0;
};
static CkptCtx g_ckpt;

extern "C" void ckpt_signal_handler(int) { g_ckpt.stop = 1; }

static const char CKPT_MAGIC[8] = {'G','F','2','S','V','C','K','2'};

// cheap 64-bit hash to tie a delta checkpoint to the exact baseline image
static u64 ckpt_hash(const ull *p, u64 n) {
    u64 h = 1469598103934665603ULL;                  // FNV-1a offset basis
    for (u64 i = 0; i < n; i++) { h ^= (u64)p[i]; h *= 1099511628211ULL; }
    return h;
}

// Copy device best[] into the host snapshot and tag it with (d, next_bucket).
static void ckpt_snapshot(int d, int next_bucket, int nbuckets) {
    if (!g_ckpt.enabled) return;
    if (g_ckpt.snap.size() != g_ckpt.smax + 1) g_ckpt.snap.resize(g_ckpt.smax + 1);
    CUCHK(cudaMemcpy(g_ckpt.snap.data(), g_ckpt.d_best,
                     (g_ckpt.smax + 1) * sizeof(ull), cudaMemcpyDeviceToHost));
    g_ckpt.snap_d = d;
    g_ckpt.snap_next_bucket = next_bucket;
    g_ckpt.snap_nbuckets = nbuckets;
}

// Keep the previous generations: <path> -> <path>.1 -> ... -> <path>.<keep-1>,
// so a checkpoint corrupted by a hardware/storage glitch is not the only copy.
static void ckpt_rotate() {
    char a[1100], b[1100];
    if (g_ckpt.keep <= 1) return;
    snprintf(a, sizeof a, "%s.%d", g_ckpt.path, g_ckpt.keep - 1);
    remove(a);                                   // drop the oldest
    for (int i = g_ckpt.keep - 1; i >= 1; i--) {
        if (i == 1) snprintf(b, sizeof b, "%s", g_ckpt.path);
        else        snprintf(b, sizeof b, "%s.%d", g_ckpt.path, i - 1);
        snprintf(a, sizeof a, "%s.%d", g_ckpt.path, i);
        rename(b, a);                            // no-op if b doesn't exist
    }
}

// Write the current snapshot atomically (tmp + rename) so a crash mid-write
// can never corrupt a good checkpoint.  Uses a delta against the --load
// baseline when that is much smaller than a full image.
static void ckpt_write() {
    if (!g_ckpt.enabled || g_ckpt.snap.size() != g_ckpt.smax + 1) return;
    double t0 = now_s();

    // Build the delta against the baseline, if we have one.
    std::vector<u64> dsig;                       // interleaved (s, key) pairs
    bool use_delta = false;
    if (g_ckpt.base.size() == g_ckpt.smax + 1) {
        const u64 cap = g_ckpt.smax / 4 + 1;     // fall back to full beyond this
        dsig.reserve(4096);
        use_delta = true;
        for (u64 s = 1; s <= g_ckpt.smax; s++) {
            if (g_ckpt.snap[s] != g_ckpt.base[s]) {
                if ((u64)(dsig.size() / 2) >= cap) { use_delta = false; break; }
                dsig.push_back(s); dsig.push_back((u64)g_ckpt.snap[s]);
            }
        }
    }

    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", g_ckpt.path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { perror(tmp); return; }
    u32 bplen = use_delta ? (u32)g_ckpt.base_path.size() : 0;
    u32 hdr32[6] = { (u32)2 /*version*/, (u32)g_ckpt.snap_d,
                     (u32)g_ckpt.snap_next_bucket, (u32)g_ckpt.snap_nbuckets,
                     (u32)(use_delta ? g_ckpt.base_kind : 0), bplen };
    u64 ndelta = use_delta ? (u64)(dsig.size() / 2) : 0;
    u64 hdr64[6] = { g_ckpt.r, g_ckpt.smax, (u64)g_ckpt.depth,
                     (u64)(g_ckpt.period_s + 0.5),
                     use_delta ? g_ckpt.base_hash : 0, ndelta };
    bool ok = fwrite(CKPT_MAGIC, 1, 8, f) == 8
           && fwrite(hdr32, sizeof(u32), 6, f) == 6
           && fwrite(hdr64, sizeof(u64), 6, f) == 6;
    if (ok && bplen)
        ok = fwrite(g_ckpt.base_path.data(), 1, bplen, f) == bplen;
    if (ok) {
        if (use_delta)
            ok = dsig.empty() ||
                 fwrite(dsig.data(), sizeof(u64), dsig.size(), f) == dsig.size();
        else
            ok = fwrite(g_ckpt.snap.data() + 1, sizeof(ull), g_ckpt.smax, f)
                   == g_ckpt.smax;
    }
    if (fflush(f) != 0) ok = false;
    fclose(f);
    if (!ok) { fprintf(stderr, "checkpoint write failed (%s)\n", tmp); return; }
    ckpt_rotate();
    if (rename(tmp, g_ckpt.path) != 0) { perror("rename checkpoint"); return; }
    g_ckpt.last_write = now_s();
    if (use_delta)
        fprintf(stderr, "[checkpoint: d=%d bucket %d/%d, delta %" PRIu64
                " entries (%.2f MB) -> %s, %.2fs]\n",
                g_ckpt.snap_d, g_ckpt.snap_next_bucket, g_ckpt.snap_nbuckets,
                ndelta, (ndelta * 16 + 128) / 1048576.0, g_ckpt.path,
                now_s() - t0);
    else
        fprintf(stderr, "[checkpoint: d=%d bucket %d/%d, full image (%.0f MB)"
                " -> %s, %.2fs]\n", g_ckpt.snap_d, g_ckpt.snap_next_bucket,
                g_ckpt.snap_nbuckets, g_ckpt.smax * 8 / 1048576.0,
                g_ckpt.path, now_s() - t0);
}

static bool ckpt_periodic_due() {
    return g_ckpt.enabled && g_ckpt.period_s > 0
        && (now_s() - g_ckpt.last_write) >= g_ckpt.period_s;
}
static bool ckpt_stop_requested() { return g_ckpt.stop != 0; }

// Persist and exit cleanly after a stop signal (or test stop).
static void ckpt_stop_and_exit() {
    ckpt_write();
    fprintf(stderr, "stopping on request; resume with "
                    "--resume %s\n", g_ckpt.path ? g_ckpt.path : "<file>");
    exit(0);
}

// Read a whole .found.bin image (one ull per s, s = 1..smax) into `v`.
static bool read_found_image(const char *path, u64 smax, std::vector<ull> &v) {
    FILE *lf = fopen(path, "rb");
    if (!lf) { perror(path); return false; }
    fseek(lf, 0, SEEK_END);
    long bytes = ftell(lf);
    fseek(lf, 0, SEEK_SET);
    if ((u64)bytes != smax * sizeof(ull)) {
        fprintf(stderr, "%s: file has %ld bytes but smax=%" PRIu64 " expects %"
                PRIu64 " (smax must match the prior run)\n",
                path, bytes, smax, (u64)(smax * sizeof(ull)));
        fclose(lf); return false;
    }
    v.assign(smax + 1, ~0ULL);                   // index 0 unused
    bool ok = fread(v.data() + 1, sizeof(ull), smax, lf) == smax;
    fclose(lf);
    if (!ok) { fprintf(stderr, "%s: short read\n", path); return false; }
    return true;
}

// Build a best[] image from a survivors list (one s per line).  Listed s are
// open (~0); everything else gets sentinel 0, meaning "already has a factor of
// some smaller degree" -- which is all the sieve needs, since it only ever
// compares with atomicMin.
static bool read_survivors_image(const char *path, u64 smax,
                                 std::vector<ull> &v, u64 *nsurv_out,
                                 u64 *nline_out, u64 *noor_out) {
    FILE *sf = fopen(path, "r");
    if (!sf) { perror(path); return false; }
    v.assign(smax + 1, 0ULL);
    u64 nsurv = 0, line = 0, oor = 0;
    char buf[128];
    while (fgets(buf, sizeof buf, sf)) {
        line++;
        char *end = nullptr;
        u64 s = strtoull(buf, &end, 10);
        if (end == buf) continue;                 // blank/garbage line
        if (s < 1 || s > smax) { oor++; continue; }
        v[s] = ~0ULL; nsurv++;
    }
    fclose(sf);
    if (nsurv_out) *nsurv_out = nsurv;
    if (nline_out) *nline_out = line;
    if (noor_out)  *noor_out  = oor;
    return true;
}

// Read a checkpoint (delta or full) into `best`; returns resume coordinates.
// A delta checkpoint reloads its recorded baseline file and applies the diff.
static bool ckpt_read_one(const char *path, u64 r, u64 smax,
                          std::vector<ull> &best, int &resume_d,
                          int &resume_bucket, int &resume_nbuckets,
                          double &stored_period_s) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    u32 hdr32[6]; u64 hdr64[6];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, CKPT_MAGIC, 8) != 0) {
        fprintf(stderr, "%s: not a v2 checkpoint file\n", path);
        fclose(f); return false;
    }
    if (fread(hdr32, sizeof(u32), 6, f) != 6 ||
        fread(hdr64, sizeof(u64), 6, f) != 6) {
        fprintf(stderr, "%s: truncated header\n", path); fclose(f); return false;
    }
    if (hdr64[0] != r || hdr64[1] != smax) {
        fprintf(stderr, "%s: r/smax (%" PRIu64 "/%" PRIu64 ") do not match this "
                "run (%" PRIu64 "/%" PRIu64 ")\n", path, hdr64[0], hdr64[1],
                r, smax);
        fclose(f); return false;
    }
    int  base_kind = (int)hdr32[4];              // 0 none, 1 found.bin, 2 survivors
    bool is_delta = (base_kind != 0);
    u32  bplen    = hdr32[5];
    stored_period_s = (double)hdr64[3];
    std::string bpath(bplen, '\0');
    if (bplen && fread(&bpath[0], 1, bplen, f) != bplen) {
        fprintf(stderr, "%s: truncated baseline path\n", path);
        fclose(f); return false;
    }
    if (is_delta) {
        bool okbase = (base_kind == 1)
            ? read_found_image(bpath.c_str(), smax, best)
            : read_survivors_image(bpath.c_str(), smax, best, nullptr, nullptr, nullptr);
        if (!okbase) {
            fprintf(stderr, "%s: needs its baseline file '%s' (keep that file "
                    "alongside the checkpoint)\n", path, bpath.c_str());
            fclose(f); return false;
        }
        u64 h = ckpt_hash(best.data() + 1, smax);
        if (h != hdr64[4]) {
            fprintf(stderr, "%s: baseline '%s' does not match the one this "
                    "checkpoint was built from\n", path, bpath.c_str());
            fclose(f); return false;
        }
        u64 nd = hdr64[5];
        for (u64 k = 0; k < nd; k++) {
            u64 pair[2];
            if (fread(pair, sizeof(u64), 2, f) != 2) {
                fprintf(stderr, "%s: truncated delta\n", path);
                fclose(f); return false;
            }
            if (pair[0] < 1 || pair[0] > smax) {
                fprintf(stderr, "%s: delta index out of range\n", path);
                fclose(f); return false;
            }
            best[pair[0]] = (ull)pair[1];
        }
        // keep the baseline available so this run can write deltas too
        g_ckpt.base_path = bpath;
        g_ckpt.base_hash = hdr64[4];
        g_ckpt.base_kind = base_kind;
        bool okb = (base_kind == 1)
            ? read_found_image(bpath.c_str(), smax, g_ckpt.base)
            : read_survivors_image(bpath.c_str(), smax, g_ckpt.base, nullptr,
                                   nullptr, nullptr);
        if (!okb) g_ckpt.base.clear();
    } else {
        best.assign(smax + 1, ~0ULL);
        if (fread(best.data() + 1, sizeof(ull), smax, f) != smax) {
            fprintf(stderr, "%s: short read\n", path); fclose(f); return false;
        }
    }
    fclose(f);
    resume_d        = (int)hdr32[1];
    resume_bucket   = (int)hdr32[2];
    resume_nbuckets = (int)hdr32[3];
    // normalize: a completed degree resumes at the next one
    if (resume_bucket >= resume_nbuckets && resume_nbuckets > 0) {
        resume_d += 1; resume_bucket = 0; resume_nbuckets = 0;
    }
    return true;
}

// Try the newest checkpoint, then successively older generations, so a
// corrupted or truncated newest copy is not fatal.
static bool ckpt_read(const char *path, u64 r, u64 smax,
                      std::vector<ull> &best, int &resume_d,
                      int &resume_bucket, int &resume_nbuckets,
                      double &stored_period_s) {
    char gen[1100];
    for (int i = 0; i < 8; i++) {
        if (i == 0) snprintf(gen, sizeof gen, "%s", path);
        else        snprintf(gen, sizeof gen, "%s.%d", path, i);
        FILE *probe = fopen(gen, "rb");
        if (!probe) { if (i == 0) continue; else break; }
        fclose(probe);
        if (ckpt_read_one(gen, r, smax, best, resume_d, resume_bucket,
                          resume_nbuckets, stored_period_s)) {
            if (i) fprintf(stderr, "note: newest checkpoint unusable; fell back "
                           "to generation %s\n", gen);
            return true;
        }
        fprintf(stderr, "checkpoint %s unusable; trying an older generation\n", gen);
    }
    fprintf(stderr, "--resume %s: no usable checkpoint generation found\n", path);
    return false;
}

// Poll the default stream to completion, checking the stop flag every ~100 ms
// so a signal during a long kernel is noticed promptly. Returns false if a
// stop was requested before completion.
static bool poll_until_done() {
    for (;;) {
        if (ckpt_stop_requested()) return false;
        cudaError_t st = cudaStreamQuery(0);
        if (st == cudaSuccess) return true;
        if (st != cudaErrorNotReady) CUCHK(st);
        usleep(100000); // 100 ms
    }
}

// run the sieve for one degree; best[] stays resident on device
static void sieve_degree_gpu(int d, u64 r, u64 smax, ull *d_best,
                             Heavy *d_heavy, unsigned *d_nheavy,
                             unsigned *d_overflow) {
    const u64 M64 = ((u64)1 << d) - 1;
    const u64 q = find_primitive_poly(d);
    const u64 rmod = r % M64;
    if (gcd_u64(rmod ? rmod : M64, M64) != 1) {
        fprintf(stderr, "d=%d: gcd(r, 2^d-1) != 1; unsupported r\n", d);
        exit(1);
    }

    // subfield moduli
    std::vector<u64> subq;
    for (u64 p : factor_u64((u64)d)) {
        int e = d / (int)p;
        subq.push_back(M64 / (((u64)1 << e) - 1));
    }

    // g^(2^t) chain on host
    fw gpow2[64];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);

    double t0 = now_s();
    fw *dA, *dL, *dgp;
    u64 *dsub = nullptr;
    CUCHK(cudaMalloc(&dA, M64 * sizeof(fw)));
    CUCHK(cudaMalloc(&dL, (M64 + 1) * sizeof(fw)));
    CUCHK(cudaMalloc(&dgp, d * sizeof(fw)));
    CUCHK(cudaMemcpy(dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
    if (!subq.empty()) {
        CUCHK(cudaMalloc(&dsub, subq.size() * sizeof(u64)));
        CUCHK(cudaMemcpy(dsub, subq.data(), subq.size() * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }

    int blocks, threads;
    launch_dims(M64, blocks, threads);
    k_antilog<<<blocks, threads>>>(dA, M64, q, d, dgp);
    k_logscatter<<<blocks, threads>>>(dA, dL, M64);

    unsigned zero = 0;
    CUCHK(cudaMemcpy(d_nheavy, &zero, 4, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_overflow, &zero, 4, cudaMemcpyHostToDevice));
    k_classes<<<blocks, threads>>>(d, M64, q, rmod, smax, dA, dL, dsub,
                                   (int)subq.size(), d_best, d_heavy,
                                   d_nheavy, d_overflow);
    unsigned nheavy = 0, overflow = 0;
    CUCHK(cudaMemcpy(&nheavy, d_nheavy, 4, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(&overflow, d_overflow, 4, cudaMemcpyDeviceToHost));
    if (overflow) {
        fprintf(stderr, "d=%d: heavy queue overflow (%u); raise HEAVY_CAP\n",
                d, overflow);
        exit(1);
    }
    if (nheavy) {
        unsigned use = nheavy > HEAVY_CAP ? HEAVY_CAP : nheavy;
        k_mark_heavy<<<use, 256>>>(d_heavy, use, smax, d_best);
    }
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaGetLastError());

    cudaFree(dA); cudaFree(dL); cudaFree(dgp);
    if (dsub) cudaFree(dsub);
    fprintf(stderr, "d=%2d done: %u heavy progressions, %.2fs\n", d, nheavy,
            now_s() - t0);
}

// ---------------------------------------------------------------------------
// LOW-MEMORY (bucketed) path: host-side driver + sizing helpers.

// Upper bound on the number of surviving classes at degree d: every
// element outside every proper subfield has a Frobenius orbit of exactly
// size d (that's precisely what the subfield-skip + coset-minimality
// filters leave behind), so there are at most (2^d - 1)/d minimal coset
// representatives. Padded for safety margin.


// bytes needed for one degree's low-memory tables at a given bucket count.
// Only the L slice remains: survivors are reconstructed from the slice
// itself (see k_resolve_slice), so nothing here is independent of nbuckets
// and every degree can be made to fit by using more buckets.
static inline u64 lowmem_chunk(u64 M64, int nbuckets) {
    u64 domain = M64 + 1;
    u64 chunk = (domain + (u64)nbuckets - 1) / (u64)nbuckets;
    return (chunk + 1) & ~1ULL;      // even: each slice is closed under XOR 1
}

static inline size_t lowmem_need_bytes(u64 M64, int d, int nbuckets) {
    (void)d;
    return (size_t)lowmem_chunk(M64, nbuckets) * sizeof(fw)
         + 4096; // gpow2[]/subq[] etc., negligible
}

#define GPU_MEM_RESERVE (256ull << 20) // headroom for allocator/context overhead

// smallest power-of-two bucket count whose slice fits in `freeb` bytes.
static int pick_buckets(u64 M64, int d, size_t freeb) {
    size_t budget = freeb > GPU_MEM_RESERVE ? freeb - GPU_MEM_RESERVE : 0;
    for (int nb = 1; nb <= (1 << 20); nb <<= 1)
        if (lowmem_need_bytes(M64, d, nb) <= budget) return nb;
    return 0;
}

// run the sieve for one degree using the bucketed low-memory path.
// There is no separate survivor pass any more: each of the `nbuckets`
// passes builds one slice of L (k_build_lslice) and then reads its own
// survivors straight back out of that slice (k_resolve_slice). Nothing
// resident is independent of nbuckets, so any degree can be made to fit by
// raising the bucket count. See "LOW-MEMORY MODE" at the top of the file.
static void sieve_degree_gpu_lowmem(int d, u64 r, u64 smax, int nbuckets,
                                    ull *d_best, Heavy *d_heavy,
                                    unsigned *d_nheavy, unsigned *d_overflow,
                                    int start_bucket = 0) {
    const u64 M64 = ((u64)1 << d) - 1;
    const u64 q = find_primitive_poly(d);
    const u64 rmod = r % M64;
    if (gcd_u64(rmod ? rmod : M64, M64) != 1) {
        fprintf(stderr, "d=%d: gcd(r, 2^d-1) != 1; unsupported r\n", d);
        exit(1);
    }
    // rmod is invertible mod M64 (checked above); the inverse lets each
    // bucket recover the class j from a discrete log read out of its slice.
    const u64 rmodinv = inv_mod(rmod, M64);

    std::vector<u64> subq;
    for (u64 p : factor_u64((u64)d)) {
        int e = d / (int)p;
        subq.push_back(M64 / (((u64)1 << e) - 1));
    }

    fw gpow2[64];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);

    double t0 = now_s();
    fw *dgp;
    u64 *dsub = nullptr;
    CUCHK(cudaMalloc(&dgp, d * sizeof(fw)));
    CUCHK(cudaMemcpy(dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
    if (!subq.empty()) {
        CUCHK(cudaMalloc(&dsub, subq.size() * sizeof(u64)));
        CUCHK(cudaMemcpy(dsub, subq.data(), subq.size() * sizeof(u64),
                         cudaMemcpyHostToDevice));
    }
    unsigned zero = 0;

    // Baseline snapshot for this degree: best[] currently holds all degrees
    // < d (and, on resume, buckets [0, start_bucket) of degree d).
    ckpt_snapshot(d, start_bucket, nbuckets);

    // --- nbuckets passes, each building and then consuming one L slice ---
    u64 domain = M64 + 1;
    u64 chunk = lowmem_chunk(M64, nbuckets);   // even, so slices close under ^1
    fw *dlslice;
    CUCHK(cudaMalloc(&dlslice, (size_t)chunk * sizeof(fw)));
    unsigned long long *d_nsurv;
    CUCHK(cudaMalloc(&d_nsurv, 8));
    unsigned long long zero64 = 0;
    CUCHK(cudaMemcpy(d_nsurv, &zero64, 8, cudaMemcpyHostToDevice));

    int blocks, threads;
    launch_dims(M64, blocks, threads);

    // progress bar: one dot per completed bucket (dots for buckets already
    // done on resume are shown up front so the bar length always matches
    // nbuckets)
    fprintf(stderr, "d=%2d lowmem %d buckets: ", d, nbuckets);
    for (int b = 0; b < start_bucket; b++) fputc('=', stderr);
    fflush(stderr);

    const fw LSLICE_EMPTY = (fw)~(fw)0;   // all-ones sentinel (u32 or u64)
    unsigned total_heavy = 0;
    for (int b = start_bucket; b < nbuckets; b++) {
        // bucket boundary: snap holds buckets [0,b) -> safe to persist here
        if (ckpt_stop_requested() || (g_ckpt.test_stop_after &&
                g_ckpt.test_buckets_done >= g_ckpt.test_stop_after)) {
            fprintf(stderr, "\n");
            ckpt_stop_and_exit();
        }
        if (ckpt_periodic_due()) ckpt_write();

        u64 lo = (u64)b * chunk;
        u64 hi = lo + chunk; if (hi > domain) hi = domain;
        if (lo >= hi) continue;

        int fblocks, fthreads;
        launch_dims(hi - lo, fblocks, fthreads);
        k_fill_fw<<<fblocks, fthreads>>>(dlslice, hi - lo, LSLICE_EMPTY);
        k_build_lslice<<<blocks, threads>>>(d, M64, q, dgp, lo, hi, dlslice);

        CUCHK(cudaMemcpy(d_nheavy, &zero, 4, cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_overflow, &zero, 4, cudaMemcpyHostToDevice));
        int sblocks, sthreads;
        launch_dims(hi - lo, sblocks, sthreads);
        k_resolve_slice<<<sblocks, sthreads>>>(d, M64, q, rmodinv, dsub,
                                               (int)subq.size(), dgp, lo, hi,
                                               smax, dlslice, d_best, d_heavy,
                                               d_nheavy, d_overflow, d_nsurv);
        // wait with ~100ms polling so a signal mid-bucket is caught quickly;
        // on stop, snap still holds buckets [0,b) and this bucket re-runs
        if (!poll_until_done()) { fprintf(stderr, "\n"); ckpt_stop_and_exit(); }

        unsigned nheavy = 0, overflow = 0;
        CUCHK(cudaMemcpy(&nheavy, d_nheavy, 4, cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(&overflow, d_overflow, 4, cudaMemcpyDeviceToHost));
        if (overflow) {
            fprintf(stderr, "\nd=%d: heavy queue overflow (%u); raise "
                    "HEAVY_CAP\n", d, overflow);
            exit(1);
        }
        if (nheavy) {
            unsigned use = nheavy > HEAVY_CAP ? HEAVY_CAP : nheavy;
            k_mark_heavy<<<use, 256>>>(d_heavy, use, smax, d_best);
            if (!poll_until_done()) { fprintf(stderr, "\n"); ckpt_stop_and_exit(); }
        }
        total_heavy += nheavy;

        // bucket b fully done (inline + heavy marks): refresh snapshot to
        // (d, b+1) and tick the progress bar
        ckpt_snapshot(d, b + 1, nbuckets);
        g_ckpt.test_buckets_done++;
        fputc('.', stderr); fflush(stderr);
    }
    fprintf(stderr, "\n");
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaGetLastError());

    unsigned long long nsurv = 0;
    CUCHK(cudaMemcpy(&nsurv, d_nsurv, 8, cudaMemcpyDeviceToHost));
    cudaFree(dgp);
    if (dsub) cudaFree(dsub);
    cudaFree(d_nsurv);
    cudaFree(dlslice);
    fprintf(stderr, "d=%2d done (lowmem, %d buckets): %llu survivors, "
            "%u heavy progressions, %.2fs\n", d, nbuckets, nsurv,
            total_heavy, now_s() - t0);
}

static void run_gpu(u64 r, int depth, u64 smax, std::vector<ull> &best,
                    int force_lowmem_buckets = 0, int min_depth = 2,
                    const std::vector<ull> *preload = nullptr,
                    int resume_bucket = 0, int resume_nbuckets = 0) {
    ull *d_best;
    Heavy *d_heavy;
    unsigned *d_nheavy, *d_overflow;
    CUCHK(cudaMalloc(&d_best, (smax + 1) * sizeof(ull)));
    CUCHK(cudaMalloc(&d_heavy, HEAVY_CAP * sizeof(Heavy)));
    CUCHK(cudaMalloc(&d_nheavy, 4));
    CUCHK(cudaMalloc(&d_overflow, 4));
    g_ckpt.d_best = d_best;          // checkpoints snapshot from here
    if (preload) {
        // Resume/extend: seed best[] with results from a prior run (its
        // .found.bin or checkpoint) so we only sieve the new degrees.  A
        // degree-d factor can only improve an s that has no smaller-degree
        // factor, so atomicMin against the loaded state is exactly correct.
        CUCHK(cudaMemcpy(d_best, preload->data(), (smax + 1) * sizeof(ull),
                         cudaMemcpyHostToDevice));
    } else {
        int blocks, threads;
        launch_dims(smax + 1, blocks, threads);
        k_fill<<<blocks, threads>>>(d_best, smax + 1, ~0ULL);
    }

    for (int d = min_depth; d <= depth; d++) {
        // On a mid-degree resume, the first degree processed continues from
        // resume_bucket using the exact nbuckets the checkpoint was made
        // with (bucket value-ranges depend on nbuckets); later degrees start
        // fresh with the auto-picked count.
        int start_bucket = (d == min_depth) ? resume_bucket : 0;
        int forced_nb    = (d == min_depth) ? resume_nbuckets : 0;

        // dense path keeps two 2^d-element tables (A and L), each sizeof(fw)
        // bytes/element: 8*2^d in the narrow build, 16*2^d in the wide build.
        size_t need = (((size_t)2 * sizeof(fw)) << d) + 16;
        size_t freeb = 0, totb = 0;
        cudaMemGetInfo(&freeb, &totb);

        if (force_lowmem_buckets > 0) {
            sieve_degree_gpu_lowmem(d, r, smax, force_lowmem_buckets, d_best,
                                    d_heavy, d_nheavy, d_overflow, start_bucket);
        } else if (forced_nb > 0) {
            // resuming mid-degree: reuse the checkpoint's bucket count
            fprintf(stderr, "d=%d: resuming at bucket %d/%d\n", d,
                    start_bucket, forced_nb);
            sieve_degree_gpu_lowmem(d, r, smax, forced_nb, d_best, d_heavy,
                                    d_nheavy, d_overflow, start_bucket);
        } else if (need <= freeb) {
            sieve_degree_gpu(d, r, smax, d_best, d_heavy, d_nheavy, d_overflow);
        } else {
            // dense tables don't fit; fall back to the bucketed low-memory
            // path (see "LOW-MEMORY MODE" at the top of the file) instead of
            // giving up outright.
            u64 M64 = ((u64)1 << d) - 1;
            int nbuckets = pick_buckets(M64, d, freeb);
            if (nbuckets == 0) {
                fprintf(stderr, "d=%d needs %.2f GB tables but only %.2f GB"
                        " free, and even 2^20 buckets does not fit; stopping"
                        " at depth %d\n", d, need / 1073741824.0,
                        freeb / 1073741824.0, d - 1);
                break;
            }
            fprintf(stderr, "d=%d: dense tables need %.2f GB but only %.2f GB"
                    " free; switching to low-memory mode (%d buckets)\n", d,
                    need / 1073741824.0, freeb / 1073741824.0, nbuckets);
            sieve_degree_gpu_lowmem(d, r, smax, nbuckets, d_best, d_heavy,
                                    d_nheavy, d_overflow, start_bucket);
        }

        // Degree boundary: a stop during a dense degree (which can't
        // checkpoint mid-way) is honored here, and periodic saves land here
        // too. best[] is a clean "degrees <= d done" state.
        if (ckpt_stop_requested()) {
            ckpt_snapshot(d + 1, 0, 0);   // next degree, no buckets done
            ckpt_stop_and_exit();
        }
        if (ckpt_periodic_due()) { ckpt_snapshot(d + 1, 0, 0); ckpt_write(); }
    }

    best.resize(smax + 1);
    CUCHK(cudaMemcpy(best.data(), d_best, (smax + 1) * sizeof(ull),
                     cudaMemcpyDeviceToHost));
    cudaFree(d_best); cudaFree(d_heavy); cudaFree(d_nheavy);
    cudaFree(d_overflow);
}

// ---------------------------------------------------------------------------
// CPU reference (identical logic to sieve_ref.cpp) for --selftest
static void sieve_cpu(u64 r, int depth, u64 smax, std::vector<ull> &best) {
    best.assign(smax + 1, ~0ULL);
    for (int d = 2; d <= depth; d++) {
        const u64 M64 = ((u64)1 << d) - 1;
        const fw M = (fw)M64;
        const u64 q = find_primitive_poly(d);
        std::vector<u64> subq;
        for (u64 p : factor_u64((u64)d)) {
            int e = d / (int)p;
            subq.push_back(M64 / (((u64)1 << e) - 1));
        }
        std::vector<fw> A((size_t)M64), L((size_t)M64 + 1);
        fw t = 1;
        for (u64 i = 0; i < M64; i++) { A[i] = t; t = gf_mulx(t, q, d); }
        for (u64 i = 0; i < M64; i++) L[A[i]] = (fw)i;
        const u64 rmod = r % M64;
        fw P[64];
        for (u64 jj64 = 1; jj64 < M64; jj64++) {
            fw j = (fw)jj64;
            bool skip = false;
            for (u64 sq_ : subq) if (jj64 % sq_ == 0) { skip = true; break; }
            if (skip) continue;
            fw jr = j; bool minimal = true;
            for (int i = 1; i < d; i++) {
                jr = rotl_d(jr, d, M);
                if (jr < j) { minimal = false; break; }
            }
            if (!minimal) continue;
            u64 u = MULMOD(jj64, rmod, M64);
            if (u == 0) continue;
            fw v = L[A[u] ^ 1u];
            u64 gg = gcd_u64(jj64, M64);
            if ((u64)v % gg != 0) continue;
            u64 n = M64 / gg;
            u64 s0 = MULMOD(inv_mod(jj64 / gg, n), (u64)v / gg, n);
            u64 mask = minpoly_mask(j, d, M, q, A.data(), P);
            u64 packed = ((u64)d << 48) | mask;
            for (u64 s = (s0 == 0 ? n : s0); s <= smax; s += n)
                if (packed < best[s]) best[s] = packed;
        }
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc >= 2 && (!strcmp(argv[1], "--selftest") ||
                      !strcmp(argv[1], "--selftest-lowmem"))) {
        bool lowmem = !strcmp(argv[1], "--selftest-lowmem");
        int force_p = 0;
        if (lowmem) {
            force_p = (argc >= 3) ? atoi(argv[2]) : 4;
            if (force_p < 1) force_p = 4;
        }
        u64 r = 136279841, smax = 999999;
        int depth = 20;
        printf("selftest%s: r=%" PRIu64 " depth=%d smax=%" PRIu64
               " (GPU vs in-process CPU reference)\n",
               lowmem ? " [lowmem, forced]" : "", r, depth, smax);
        if (lowmem)
            printf("  forcing the bucketed low-memory path, %d buckets,"
                   " for every degree\n", force_p);
        std::vector<ull> g, c;
        run_gpu(r, depth, smax, g, force_p);
        double t0 = now_s();
        sieve_cpu(r, depth, smax, c);
        fprintf(stderr, "cpu reference: %.2fs\n", now_s() - t0);
        u64 bad = 0;
        for (u64 s = 1; s <= smax; s++)
            if (g[s] != c[s] && ++bad <= 10)
                fprintf(stderr, "MISMATCH s=%" PRIu64 " gpu=%" PRIx64
                        " cpu=%" PRIx64 "\n", s, (u64)g[s], (u64)c[s]);
        printf("selftest: %s (%" PRIu64 " mismatches)\n",
               bad ? "FAIL" : "PASS", bad);
        return bad ? 1 : 0;
    }

    if (argc < 4) {
        fprintf(stderr, "usage: %s <r> <depth> <out_prefix> [options]\n"
                        "       %s --selftest\n"
                        "       %s --selftest-lowmem [nbuckets]\n"
                        "options:\n"
                        "  --smax <S>          highest s to sieve "
                        "(default floor(r/2))\n"
                        "  --no-cert           suppress <out_prefix>.found.cert "
                        "(written by default)\n"
                        "  --no-survivors      suppress <out_prefix>.survivors.txt "
                        "(written by default)\n"
                        "  --found             also write the raw binary "
                        "<out_prefix>.found.bin\n"
                        "  --load <f.found.bin>        seed best[] from a prior "
                        "run's binary image\n"
                        "  --load-survivors <f.txt>    seed from a prior run's "
                        "survivors list instead\n"
                        "  --min-depth <D>     start the sieve at degree D "
                        "(compute only new degrees)\n"
                        "  --checkpoint <file> save progress on SIGTERM/SIGINT, "
                        "mid-degree\n"
                        "  --checkpoint-mins <N>       also save every N minutes\n"
                        "  --checkpoint-keep <G>       keep G checkpoint "
                        "generations (default 3)\n"
                        "  --resume <file>     continue from a checkpoint\n"
                        "\n"
                        "  <out_prefix>.found.cert lists one line per s that has a "
                        "factor, in the same\n"
                        "  's d p<hexmask>' format as GF2X factor.cpp, so per-degree "
                        "certs simply\n"
                        "  concatenate into a full record.  Example incremental run:\n"
                        "     %s <r> 35 out35 --load-survivors out34.survivors.txt "
                        "--min-depth 35 \\\n"
                        "         --checkpoint out35.ckpt --checkpoint-mins 20\n"
                        "  (the legacy form '<r> <depth> <smax> <out_prefix>' is "
                        "still accepted)\n",
                        argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    u64 r = strtoull(argv[1], 0, 10);
    int depth = atoi(argv[2]);
    // Positional smax is legacy: '<r> <depth> <smax> <out_prefix>'.  Detect it
    // by argv[3] being a pure integer (an out_prefix never is).
    bool legacy_smax = false;
    {
        const char *p = argv[3];
        bool alldig = *p != '\0';
        for (const char *c = p; *c; c++) if (*c < '0' || *c > '9') { alldig = false; break; }
        if (alldig && argc >= 5) legacy_smax = true;
    }
    u64 smax = legacy_smax ? strtoull(argv[3], 0, 10) : (r / 2);
    const char *prefix = legacy_smax ? argv[4] : argv[3];
    int first_flag = legacy_smax ? 5 : 4;
    if (legacy_smax)
        fprintf(stderr, "note: positional <smax> is deprecated; --smax <S> "
                "(default floor(r/2) = %" PRIu64 ") is preferred\n", r / 2);

    bool write_found = false, write_survivors = true;   // survivors default on
    bool write_cert = true;                             // cert default on
    const char *load_path = nullptr, *ckpt_path = nullptr, *resume_path = nullptr;
    const char *loadsurv_path = nullptr;
    int min_depth = 2, ckpt_keep = 3;
    double ckpt_mins = -1;                       // <0 = not given on command line
    for (int a = first_flag; a < argc; a++) {
        if (!strcmp(argv[a], "--found")) write_found = true;
        else if (!strcmp(argv[a], "--smax") && a + 1 < argc) smax = strtoull(argv[++a], 0, 10);
        else if (!strcmp(argv[a], "--cert")) write_cert = true;
        else if (!strcmp(argv[a], "--no-cert")) write_cert = false;
        else if (!strcmp(argv[a], "--no-survivors")) write_survivors = false;
        else if (!strcmp(argv[a], "--load") && a + 1 < argc) load_path = argv[++a];
        else if (!strcmp(argv[a], "--load-survivors") && a + 1 < argc) loadsurv_path = argv[++a];
        else if (!strcmp(argv[a], "--min-depth") && a + 1 < argc) min_depth = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--checkpoint") && a + 1 < argc) ckpt_path = argv[++a];
        else if (!strcmp(argv[a], "--checkpoint-mins") && a + 1 < argc) ckpt_mins = atof(argv[++a]);
        else if (!strcmp(argv[a], "--checkpoint-keep") && a + 1 < argc) ckpt_keep = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--resume") && a + 1 < argc) resume_path = argv[++a];
        else fprintf(stderr, "warning: ignoring unknown argument '%s'\n", argv[a]);
    }
    if (min_depth < 2) min_depth = 2;
    if (ckpt_keep < 1) ckpt_keep = 1;
    g_ckpt.keep = ckpt_keep;

    // test hook (deterministic checkpoint testing without a real signal):
    // COARSE_TEST_STOP_AFTER_BUCKETS=N sets the stop flag after N buckets.
    if (const char *e = getenv("COARSE_TEST_STOP_AFTER_BUCKETS"))
        g_ckpt.test_stop_after = atol(e);

    std::vector<ull> preload;
    bool have_preload = false;
    int resume_bucket = 0, resume_nbuckets = 0;

    double stored_period_s = 0;
    if (resume_path) {
        // Mid-degree resume: read best[] + (degree, bucket, nbuckets).
        if (!ckpt_read(resume_path, r, smax, preload, min_depth,
                       resume_bucket, resume_nbuckets, stored_period_s))
            return 1;
        have_preload = true;
        u64 pre_surv = 0;
        for (u64 s = 1; s <= smax; s++) if (preload[s] == ~0ULL) pre_surv++;
        fprintf(stderr, "resumed %s: %" PRIu64 " survivors carried in; "
                "continuing at degree %d, bucket %d/%d, up to depth %d\n",
                resume_path, pre_surv, min_depth, resume_bucket,
                resume_nbuckets, depth);
    } else if (load_path) {
        // Degree-boundary extend from a prior .found.bin.
        if (!read_found_image(load_path, smax, preload)) return 1;
        have_preload = true;
        // This image is also the baseline for delta checkpoints.
        g_ckpt.base = preload;
        g_ckpt.base_path = load_path;
        g_ckpt.base_kind = 1;
        g_ckpt.base_hash = ckpt_hash(preload.data() + 1, smax);
        u64 pre_surv = 0;
        for (u64 s = 1; s <= smax; s++) if (preload[s] == ~0ULL) pre_surv++;
        fprintf(stderr, "loaded %s: %" PRIu64 " survivors carried in; "
                "sieving degrees %d..%d only\n",
                load_path, pre_surv, min_depth, depth);
    } else if (loadsurv_path) {
        // Seed from a survivors list instead of a full .found.bin.  Only the
        // *degrees* in best[] matter for correctness when sieving degrees
        // >= min_depth: every s not listed already has a factor of degree
        // < min_depth, so any sentinel below (min_depth << 48) blocks all new
        // writes exactly as the real key would.  The polynomial data in a
        // .found.bin is never read by the sieve -- only compared against.
        u64 pre_surv = 0, line = 0, oor = 0;
        if (!read_survivors_image(loadsurv_path, smax, preload, &pre_surv,
                                  &line, &oor)) return 1;
        if (oor) fprintf(stderr, "warning: %" PRIu64 " survivor entries outside "
                         "1..%" PRIu64 " ignored\n", oor, smax);
        have_preload = true;
        g_ckpt.base = preload;                    // baseline for delta checkpoints
        g_ckpt.base_path = loadsurv_path;
        g_ckpt.base_kind = 2;
        g_ckpt.base_hash = ckpt_hash(preload.data() + 1, smax);
        fprintf(stderr, "loaded %s: %" PRIu64 " survivors carried in (%" PRIu64
                " lines); sieving degrees %d..%d only\n",
                loadsurv_path, pre_surv, line, min_depth, depth);
        fprintf(stderr, "note: with --load-survivors the output .found.bin "
                "records only the newly found degree-%d..%d factors; keep the "
                "earlier run's file for the lower degrees\n", min_depth, depth);
    } else if (min_depth > 2) {
        fprintf(stderr, "warning: --min-depth %d without --load/--resume means "
                "degrees 2..%d are not computed\n", min_depth, min_depth - 1);
    }

    // Configure checkpointing and install signal handlers.  A --resume run
    // keeps checkpointing to the same file (so it survives repeated spot
    // reclaims) unless --checkpoint names a different destination.
    if (resume_path && !ckpt_path) ckpt_path = resume_path;
    if (ckpt_path) {
        g_ckpt.enabled  = true;
        g_ckpt.path     = ckpt_path;
        // --checkpoint-mins on the command line wins; otherwise a resumed run
        // inherits the interval the checkpoint was written with, so you don't
        // silently lose periodic saves by omitting the flag on resume.
        g_ckpt.period_s = (ckpt_mins >= 0) ? ckpt_mins * 60.0 : stored_period_s;
        g_ckpt.last_write = now_s();
        g_ckpt.r = r; g_ckpt.smax = smax; g_ckpt.depth = depth;
        struct sigaction sa; memset(&sa, 0, sizeof sa);
        sa.sa_handler = ckpt_signal_handler;
        sigaction(SIGTERM, &sa, nullptr);
        sigaction(SIGINT,  &sa, nullptr);
        bool delta_ok = (g_ckpt.base.size() == smax + 1);
        fprintf(stderr, "checkpointing to %s (%s, keeping %d generation%s)",
                ckpt_path, delta_ok ? "delta vs baseline" : "full image",
                g_ckpt.keep, g_ckpt.keep == 1 ? "" : "s");
        if (g_ckpt.period_s > 0)
            fprintf(stderr, ", every %.1f min", g_ckpt.period_s / 60.0);
        fprintf(stderr, "; SIGTERM/SIGINT saves and exits\n");
        if (!delta_ok && !loadsurv_path)
            fprintf(stderr, "note: no --load baseline, so checkpoints write a "
                    "full %.0f MB image; on slow storage prefer --load\n",
                    smax * 8 / 1048576.0);
    }

    double t0 = now_s();
    std::vector<ull> best;
    int force_buckets = 0;
    if (const char *e = getenv("COARSE_FORCE_BUCKETS")) force_buckets = atoi(e);
    run_gpu(r, depth, smax, best, force_buckets, min_depth,
            have_preload ? &preload : nullptr, resume_bucket, resume_nbuckets);
    fprintf(stderr, "total sieve time: %.2fs\n", now_s() - t0);

    char path[1024];
    if (write_survivors) {
        snprintf(path, sizeof path, "%s.survivors.txt", prefix);
        FILE *fs = fopen(path, "w");
        if (!fs) { perror(path); return 1; }
        u64 nsurv = 0;
        for (u64 s = 1; s <= smax; s++)
            if (best[s] == ~0ULL) { fprintf(fs, "%" PRIu64 "\n", s); nsurv++; }
        fclose(fs);
        fprintf(stderr, "%" PRIu64 " survivors (no factor of degree <= %d) -> %s\n",
                nsurv, depth, path);
    } else {
        u64 nsurv = 0;
        for (u64 s = 1; s <= smax; s++) if (best[s] == ~0ULL) nsurv++;
        fprintf(stderr, "%" PRIu64 " survivors (no factor of degree <= %d) "
                "[survivors file suppressed via --no-survivors]\n", nsurv, depth);
    }

    if (write_cert) {
        // Text certificate in GF2X factor.cpp's own format: one line per s
        // that has a factor, "s d p<hexmask>", ascending in s.  Entries that
        // are survivors (~0) or dummies carried in by --load-survivors (0)
        // are skipped, so an incremental run's cert holds exactly the factors
        // that run discovered.  Concatenating per-degree certs (or sorting
        // them with `sort -n`) reconstitutes a complete record.
        snprintf(path, sizeof path, "%s.found.cert", prefix);
        FILE *fc = fopen(path, "w");
        if (!fc) { perror(path); return 1; }
        u64 ncert = 0;
        for (u64 s = 1; s <= smax; s++) {
            ull b = best[s];
            if (b == ~0ULL || b == 0ULL) continue;      // survivor / not ours
            unsigned dd = (unsigned)(b >> 48);
            u64 mask = (u64)b & 0xFFFFFFFFFFFFULL;
            fprintf(fc, "%" PRIu64 " %u p%" PRIx64 "\n", s, dd, mask);
            ncert++;
        }
        fclose(fc);
        fprintf(stderr, "%" PRIu64 " factors -> %s\n", ncert, path);
    }

    if (write_found) {
        snprintf(path, sizeof path, "%s.found.bin", prefix);
        FILE *ff = fopen(path, "wb");
        if (!ff) { perror(path); return 1; }
        fwrite(best.data() + 1, sizeof(ull), smax, ff);
        fclose(ff);
        fprintf(stderr, "packed keys for s=1..%" PRIu64 " -> %s\n", smax, path);
    }
    // Run finished normally: drop the checkpoint so it can't be resumed by
    // mistake later.
    if (g_ckpt.enabled && g_ckpt.path) remove(g_ckpt.path);
    return 0;
}

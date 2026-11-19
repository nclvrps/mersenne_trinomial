// coarse_sieve.cu  (round 9)
// GPU coarse sieve for the least-degree irreducible factors of the
// trinomials x^r + x^s + 1 over GF(2), all s in [1, floor(r/2)], factor
// degrees d <= 55.
//
// For each degree d the sieve finds every s whose trinomial has an
// irreducible factor of degree d and records the least such factor; s
// already resolved at a lower degree are never touched again.  Outputs:
//   <prefix>.found.cert     one line "s d p<hexmask>" per s that got its
//                           factor in this run, ascending in s (the format
//                           of GF2X factor.cpp; certs of consecutive runs
//                           concatenate into a full record)
//   <prefix>.survivors.txt  the s with no factor of degree <= the run's top
//                           degree: the next run's --load-survivors seed
//
// USAGE
//   coarse_sieve <r> <degrees> <out_prefix> [options]
//     <degrees>  D1-D2 for degrees D1..D2, or D for degree D alone.  A run
//                from scratch starts at 2; a run starting at D1 > 2 needs
//                --load-survivors with the survivors of the run that ended
//                at D1-1:
//                  coarse_sieve 136279841 2-46 out2-46
//                  coarse_sieve 136279841 47 out47
//                               --load-survivors out2-46.survivors.txt
//   options
//     --load-survivors <f>    the seed: survivors of a run through D1-1
//     --no-cert, --no-survivors   skip that output file
//     --checkpoint <f>        checkpoint file (default <out_prefix>.ckpt)
//     --checkpoint-mins <N>   save every N minutes (default 15; 0: on signals
//                             only)
//     --no-checkpoint         no checkpoint file at all
//     --fresh                 set an existing checkpoint aside (*.old) and
//                             start over
//     --gpus <N>              use N GPUs (default: all; chunks are dealt out,
//                             results merged by minimum; no peer access needed)
//     --no-tune               skip the per-degree timing of kernel variants
//     --watchdog-mins <N>     exit with status 3 when nothing progresses for N
//                             minutes, e.g. a hung GPU (default 30; 0: off;
//                             see WATCHDOG)
//   coarse_sieve --selftest [tower|ph|rr|cubic]
//
// Build:  nvcc -O3 -arch=native -std=c++17 -Xcompiler -march=native -I.
//              -o coarse_sieve coarse_sieve.cu -Xcompiler -pthread
//         (needs gf2_wide_field.h, gf2_wide_setup.h, gf2_dlog.h; the 116
//         kernel instantiations take ~6 minutes to compile).  Add
//         -DCOARSE_PHASE_TIMING for a diagnostic build that reports where
//         the batched kernel's time goes (see PHASE TIMING).
//
// METHOD
// Work in GF(2^d) = GF(2)[x]/(q_d), q_d primitive, so g = x generates the
// multiplicative group, of order M = 2^d - 1.  A degree-d irreducible P with
// root beta = g^j divides x^r + x^s + 1 iff beta^s = 1 + beta^r, i.e.
//     j s == Z(u)  (mod M),   u = j r mod M,   Z(u) = log_g(1 + g^u)
// (Z is the Zech logarithm): an arithmetic progression of s with modulus
// n = M / gcd(j, M), or nothing if gcd(j, M) does not divide Z(u).  The
// conjugate roots g^(2^i j) give the same P and the same progression, so
// each class -- a cyclotomic coset of u under u -> 2u mod M, i.e. bit
// rotation of u as a d-bit word -- is handled once, through its least
// rotation: a binary Lyndon word, odd and below 2^(d-1).  These are exactly
// the classes of degree d, I(d) = (1/d) sum_{e|d} mu(e) 2^(d/e) of them,
// which every degree checks at its end.
//
// Reciprocal pairs (round 6).  The class of -u (the d-bit complement of u)
// holds the roots g^-j of the reciprocal polynomial P*, and as
// 1 + g^-u = g^-u (1 + g^u),  Z(-u) = Z(u) - u:  that class solves
// -j s == Z(u) - u, i.e. s == r - s0 (mod n), with the same n.  So one Zech
// log serves both classes: the canonical member of each pair (fewer ones
// than zeros, ties broken exactly; gf2f_pair_role in gf2_dlog.h) marks both
// progressions and the other member is skipped; self-reciprocal classes,
// their own partners, are handled alone.  This halves the discrete logs,
// table reads, inversions and bit-sliced work of every degree.  (Round 7
// computes the tie-break of even d -- as many ones as zeros, ~11% of the
// classes -- from u's longest runs of ones instead of all d rotations of
// ~u: the same roles, without a d-step loop inside the enumeration.)
//
// The Zech log of each canonical class comes from one of four engines
// (gf2_dlog.h), whose tables have ~2^(d/2) entries rather than 2^d:
//   tower (even d = 2m)  GF(2^d) = K(w) with K = GF(2^m): e = b (c + w), so
//          log e = (2^m + 1) log_h(b) + T[log_h(c)], T of 2^m logs.  The logs
//          in K come from the same trick one level down, over a subfield of
//          K (the subfield tower), or from a table of all of K (the classic
//          tower; the only one when m is prime: d = 34, 38, 46).
//   PH, Pohlig-Hellman (odd d)  log x mod each coprime part P_i of M from
//          x^(M/P_i) (bit-sliced), by table lookup -- baby-step/giant-step
//          when a part's table does not fit -- then CRT.
//   RR, rational reconstruction (odd d >= 5)  half an extended Euclid on
//          (q, y) writes y = r/t with deg r, deg t <= d/2, so log y =
//          L[r] - L[t], L the logs of ALL 2^(d/2+1) polynomials of degree
//          <= d/2 (256 MB at d=49, 2 GB at d=55): no smoothness test, no
//          failure.  L is built from the logs of the irreducibles of degree
//          <= d/2 (by PH; 1.47M at d=49) by multiplicativity, and cached in
//          coarse_rr_d<d>.tbl when that took over two minutes.  RR is the
//          only engine at d=49 (2^49 - 1 = 127 * 4432676798593).
//   cubic tower (round 9; 6 | d = 3n)  GF(2^d) = K(w), K = GF(2^n), w^3 in
//          K: x = r/t with r, t LINEAR in w (RR over K), so log x is four
//          logs in K (the classic tower of K: tables of 2^(n/2) entries, in
//          shared memory) and two reads of T3, the 2^n logs of w + c -- 2 MB
//          at d=54, held in L2, where the quadratic tower reads a 1 GB T in
//          DRAM.  A tuner candidate at d = 42, 48, 54 (gf2_dlog.h).
// The batched kernel enumerates the Lyndon words of its range of u with
// FKM -- every lane of a warp its own runs, the canonical classes found
// dealt out to the lanes in turn (round 7) -- and handles them 32 per
// thread: g^u (and PH's powers)
// in bit-sliced form, then the logs, then the progression starts v j^-1 of
// all 32 with one modular inversion (Montgomery's trick).  Classes with
// gcd(u, M) > 1 join the same inversion through exact 2-adic quotients and
// a unit lift (round 5; see "Irregular classes").  A class marks best[s] --
// atomicMin of the key (d, mask), i.e. least degree, then least polynomial
// -- only for the s <= smax on its progression (rare: probability ~smax/n),
// and its minimal polynomial is computed only then.  Progressions with more
// than INLINE_MARKS marks (small d) go to a queue marked by k_mark_heavy.
//
// The pass over u in [0, 2^(d-1)) is cut into chunks, one kernel launch
// each, the checkpoint granularity; with several GPUs, each takes every
// NG-th chunk into its own best[], and the copies are merged by minimum.
// Classes concentrate at small u (a Lyndon word starts with its longest
// zero run, typically ~log2(d) bits): the first 1/64 of the chunks holds
// ~15-18% of the classes at d = 42..50.  So the pass takes the chunks in a
// scrambled order (blocks of 8 in a pseudo-random order, chunk_at), every
// stage of it a fair sample of the range: the progress bar's 64 dots come
// at a steady pace, and the time left -- shown once a few hundred chunks
// are done, and with every checkpoint -- is this run's time per class times
// the classes left (I(d) is known).
// Every log is range-checked, one per 8 batches is verified (g^v ==
// g^u + 1), the class count is checked against I(d), and any inconsistency
// stops the run before an output or checkpoint could include it.
//
// TUNING
// For d >= 40 (COARSE_TUNE_MIN) each degree first runs its candidate
// variants on one dense sample of ~2^26 classes on GPU 0 -- the subfield
// towers over the two largest proper divisors of m, the classic tower and
// (round 9, 6 | d) the cubic tower at even d, PH and RR at odd d; the
// tower, cubic and RR ones in each loop order: round 5's ("r5-order", the
// default since round 8), round 9's ("r9-order": a class's log times the
// prefix product stored as one value, 9-10 local-memory accesses per class
// instead of 12; it replaced round 6's) and the grouped one ("grouped":
// w and J of a group of 8 classes held in registers, one inversion per
// group; round 8, rewritten with rolled loops in round 9 --
// logs_batch_grouped); and at d = 48 and 52, where it decides the L1 size,
// the subfield tower with its table TP in global memory too ("TP-global";
// tower_tp_choice) -- and sieves with the fastest; the "[tune ...]" line
// reports ns per class for each.  All variants compute identical results.
// --no-tune (COARSE_TUNE=0) turns this off.  The choice is stored in
// checkpoints, so a resumed degree keeps it without re-timing.
//
// CHECKPOINTS (format v4, rounds 6-9)
// Every run saves to <out_prefix>.ckpt (or --checkpoint <file>) every 15
// minutes, at a chunk boundary, and on SIGTERM or SIGINT: the first signal
// saves and exits after the chunk in flight (seconds), a second exits at
// once.  Running the same command again continues from the file; it records
// the run (r, smax, the degree range, the seed file's path and a hash of its
// contents) and a different command line refuses to start rather than mix
// two runs.  --fresh sets the old file aside (*.old) and starts over;
// --no-checkpoint writes none; a finished run deletes it.  <file>.lock
// keeps a second run off the same file: what counts is the flock on it,
// which ends with its process however that ends, so a lock file left by a
// killed run is harmless and never needs deleting (round 8 notes one when
// it finds it, and unlinks its own on a second signal and on a watchdog
// exit too).  The file also records the position
// (degree, chunks done, the classes they covered) and the degree's engine,
// inner subfield, loop order and (round 8) TP placement, and holds best[]
// as a delta against the
// seed image when that is small (a few MB for a one-degree run), else as a
// full image (8 bytes per s).  Writes are atomic (tmp + rename); the two
// previous generations are kept as <file>.1 and <file>.2.  Round 7: it also
// records the time spent on the degree and on the run's earlier degrees, so
// a resumed degree's done line (and the total) gives the time of one
// uninterrupted run -- each run's time up to its last checkpoint; the table
// rebuild of a resumed run is not counted there -- with this run's own
// time beside it.  Round 6's and 7's binaries read round 8's checkpoints
// (they take the grouped order for their own pipelined one and ignore the
// TP flag: the same results), and round 8 reads theirs.  Round 9 reads
// them all (loop order 1, round 6's there, is round 9's order here: the
// same results), and rounds 6-8 read round 9's -- except in the middle of
// a degree sieved with the cubic tower (engine tag 4), which they refuse
// ("the checkpoint's engine is not possible here"); round 9 continues it.
//
// WATCHDOG (round 8)
// If no chunk, tuner sample, table-building launch or checkpoint write
// completes for --watchdog-mins (30), the run says so and exits with
// status 3, the last checkpoint intact -- for unattended runs, where a hung
// GPU would otherwise burn hours; a wrapper can restart it:
//     until ./coarse_sieve ...; do [ $? -eq 3 ] || break; sleep 60; done
//
// PHASE TIMING (round 8; a build option, -DCOARSE_PHASE_TIMING)
// Each degree (and a stop) prints the batched kernel's warp-cycles per log
// and lane by phase -- fill (enumeration and deal), g^u (bit-sliced),
// logs, inversion, marking, other -- and their shares.  The production
// build has none of this code.
//
// MEMORY
// best[] (8 bytes per s: 545 MB at r = 136279841) on every GPU, plus the
// degree's tables: tower 8 * 2^(d/2) bytes for T (1 GB at d=54); PH within
// the free memory (fewer giant steps the more fits); RR 8 * 2^(d/2+1)
// (2 GB at d=55) and, while building it at d=49, a baby-step table of half
// an 8 GB card for a minute or so; the cubic tower 8 * 2^(d/3) (2 MB at
// d=54) and 26 KB of shared memory, plus PH tables of 64 MB while T3 is
// built (under a second).
//
// SELFTEST
// --selftest checks the pass order (chunk_at: a permutation, spread out),
// sieves r = 136279841, d = 2..20, s < 10^6 on the GPU and compares every
// entry with the in-process CPU reference (sieve_cpu, the plain table
// method, no pairing); then on device 0 it checks the rotation
// test, the irregular-class lift and the reciprocal pairing (against
// independent direct computations), and every engine at every d = 21..55:
// log(g^u) == u on 2^20 exponents, and the batched kernel's enumeration and
// logs on a stretch dense in classes -- RR at odd d <= 47, once (d=41) with
// its factor-base logs forced through a many-giant-step plan like d=49's;
// the cubic tower (round 9) at every 6 | d beside the tower.
// "tower" / "ph" / "rr" / "cubic" force that engine wherever it applies.
// gf2_dlog_selftest.cpp is the host-side counterpart.
//
// TEST HOOKS (environment)
//   COARSE_METHOD=auto|tower|ph|rr|cubic   engine choice (rr: RR at odd
//                         d; cubic: the cubic tower at 6 | d, round 9)
//   COARSE_TOWER_SUBN=n   inner subfield GF(2^n) of the tower (0: classic)
//   COARSE_LOGS_MAX_MB=n  cap on the VRAM of the degree's tables
//   COARSE_LOGS_CHUNK_LOG2=k   chunks of at most 2^k u values (a resume
//                         must use the same k)
//   COARSE_TUNE=0, COARSE_TUNE_MIN=d, COARSE_TUNE_PICK=i (take candidate i)
//   COARSE_PIPE=0|1|2     fix the loop order instead of tuning it (0: round
//                         5's, the default; 1: round 9's (round 6's before
//                         round 9); 2: grouped -- tower, cubic and RR at
//                         d >= 40 on the GPU, other degrees fall back to 0)
//   COARSE_LDCG=0         the random table reads (tower T, RR L) through L1,
//                         as before round 8 (default 1: past L1, __ldcg)
//   COARSE_PREFETCH=0     no L1 prefetches of the local arrays (round 9:
//                         g^u's exponent planes two groups ahead, the batch
//                         arrays two classes ahead forward, four backward)
//   COARSE_SYNC=poll      wait for each chunk by polling its event every
//                         millisecond (round 9; default block: sleep in the
//                         driver -- which under WSL2 seems to spin a core)
//   COARSE_TOWER_TP=shared|global   fix where the subfield tower's TP lives
//                         (default auto: shared memory if it fits; the tuner
//                         also times global at d = 48, 52)
//   COARSE_TEST_TP_CHOICE=1   offer the TP-global candidates at every degree
//   COARSE_TEST_WATCHDOG_SECS=s   the watchdog's limit in seconds
//   COARSE_TEST_HANG_AFTER_CHUNKS=n   hang (no progress, stop requests
//                         ignored) once n chunks are done
//   COARSE_FILL=0|1|2|3   how the batches are filled (round 7, k_logs_batch):
//                         0 round 6's, 1 rotated runs, 2 pooled, 3 pooled
//                         with runs taken from a counter (the default)
//   COARSE_TEST_SMAX=n    sieve s <= n instead of floor(r/2)
//   COARSE_TEST_FAULT_EVERY=n   corrupt the log of classes with u == 1
//                         (mod n): the run must stop with consistency errors
//   COARSE_TEST_STOP_DEGREE=d   act as if degree d could not be sieved: write
//                         the outputs through d-1, keep the checkpoint, exit 2
//   COARSE_TEST_STOP_AFTER_CHUNKS=n   stop as on a signal after n chunks
//   COARSE_TEST_ETA=1     print the time-left estimate of every pass, in s
//   COARSE_RR_DIR=dir     where RR tables are cached (default .)
//   COARSE_RR_FB_BUDGET_MB=m   plan RR's factor-base logs with tables of at
//                         most m MB (forces the many-giant-step path)
//   COARSE_SELFTEST_LOGS_DMAX, COARSE_SELFTEST_RR_DMAX (default 47; 49 adds
//                         d=49's table, a few minutes), COARSE_SELFTEST_RR_BSGS_D
//
// HISTORY (details in the round handoffs): round 1 LFSR antilog sweeps and
// smaller buckets; round 2 the logs path (tower, PH); round 3 the batched
// kernel; round 4 the subfield tower; round 5 the irregular-class lift,
// RR and d=49, tuning, automatic checkpoints; round 6 reciprocal pairs, and
// removal of the u32 build, the dense and bucketed paths (the logs path is
// faster at every degree), the round-2 kernel, the round-4 irregular path,
// .found.bin files and the old checkpoint formats; round 7 full batches
// (the warps pool the classes they find), the faster pair tie-break, the
// time of all runs in checkpoints, and lockstep warps in the emulation;
// round 8 the watchdog and the lock-file notes, the grouped loop order and
// TP-global as tuner candidates, table reads past L1, round 5's order as
// the untuned default, and the phase-timing build; round 9 the cubic tower
// (d = 54), round 9's loop order (replacing round 6's), the grouped order
// with rolled loops and the fill's state parked across its batches, g^u's
// exponent planes loaded three steps ahead, and the inversion's check
// (inv * prod J == 1).

// the CPU emulation (make emul-coarse_sieve) runs the lanes of each warp in
// lockstep, for the batched kernel's warp intrinsics (cuda_emul.h)
#define EMUL_LOCKSTEP_WARPS 1
#include "gf2_wide_field.h"
#include "gf2_wide_setup.h"
#include <cuda_runtime.h>
#include "gf2_dlog.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <csignal>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <cerrno>
#include <atomic>
#include <pthread.h>

// fw ("field word"): field elements and everything indexed like them
typedef u64 fw;

#define CUCHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
            __FILE__, __LINE__); exit(1); } } while (0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

typedef unsigned long long ull;

// best[] keys: "smallest degree, then numerically least mask" as one u64
// compared by atomicMin.  The mask has d+1 bits (all coefficients of the
// minimal polynomial).
//   d <= 47:       (d << 48) | mask
//   48 <= d <= 55: (0xF << 60) | ((d - 48) << 56) | mask
// Keys increase with (d, mask), every d <= 47 key is below every d >= 48 key,
// and all are below the UINT64_MAX "no factor" marker (and above the 0 that
// --load-survivors uses for "factor of a lower degree").
#define KEY_MAX_DEGREE 55
GF2_HD ull key_pack(int d, u64 mask) {
    return d <= 47 ? ((ull)d << 48) | mask
                   : (0xFull << 60) | ((ull)(d - 48) << 56) | mask;
}
GF2_HD ull key_floor(int d) { return key_pack(d, 0); }  // least degree-d key
static inline int key_degree(ull k) {
    return (k >> 60) == 0xF ? 48 + (int)((k >> 56) & 0xF) : (int)(k >> 48);
}
static inline u64 key_mask(ull k) {
    return (k >> 60) == 0xF ? (u64)(k & ((1ull << 56) - 1))
                            : (u64)(k & 0xFFFFFFFFFFFFull);
}

// heavy-progression record for the second-pass marking kernel
struct Heavy { u64 s_first, n, packed; };

// classes whose progressions have more than this many marks (small d) are
// deferred to k_mark_heavy (one thread block per record) for load balance
#define INLINE_MARKS 1024
#define HEAVY_CAP    (1u << 22)

// (a*b) mod m for any m < 2^64 (double-and-add; rare or host-side callers)
#define MULMOD(a, b, m) mulmod_u64((u64)(a), (u64)(b), (u64)(m))

// True if some best[s'] on the progression s, s+n, ... <= smax can still be
// lowered by a degree-d key.  Every degree-d key is >= key_floor(d); entries
// already below that (a smaller-degree factor, or the 0 sentinel that
// --load-survivors uses for non-survivors) can never change, and best[] only
// ever decreases (atomicMin), so a stale read can only cause extra work,
// never a wrong skip.  Lets the caller skip the d^2-multiply minimal
// polynomial for the ~95% of in-range hits that land on s values which
// already have a smaller factor.
GF2_HD bool progression_open(const ull *best, u64 s, u64 n, u64 smax, int d) {
    const ull floor_d = key_floor(d);
    for (; s <= smax; s += n)
        if (best[s] >= floor_d) return true;
    return false;
}

// ---------------------------------------------------------------------------
// g^idx from the size-d gpow2 chain, gpow2[t] = g^(2^t): one field multiply
// per set bit of idx (rare per-class uses only: a marked class's root)
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

// The minimal polynomial of the root `seed` as a (d+1)-bit mask: the
// product of (x - seed^(2^i)), i < d, by repeated squaring of the root;
// 0 if a coefficient is not in GF(2) (an internal inconsistency).
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

// logs-path engines (also the checkpoint's engine tags); LM_CUBIC (round
// 9): the cubic tower, 6 | d (gf2_dlog.h, "cubic tower logs")
enum { LM_NONE = 0, LM_TOWER = 1, LM_PH = 2, LM_RR = 3, LM_CUBIC = 4 };

// ---------------------------------------------------------------------------
// THE LOGS PATH: no 2^d tables.  See METHOD in the header comment and
// gf2_dlog.h.  One pass over u in [0, 2^(d-1)):
//   * every cyclotomic class has exactly one coset-minimal u (minimal under
//     u -> 2u mod M, i.e. bit rotation), and that u is odd and < 2^(d-1);
//   * any representative gives the same factor and progression: with
//     u' = 2^i u and Z(2u) = 2 Z(u), the congruence j's == Z(u') reduces to
//     j s == Z(u) because 2 is invertible mod M;  gcd(j, M) = gcd(u, M) and
//     u is in a subfield iff j is, since j = u * r^-1 with r a unit;
//   * g^u comes from bit-sliced squarings, and Z(u) = log(g^u + 1) from the
//     tower, PH or RR engine -- one discrete log per reciprocal pair.
#define LOGS_MAXPF 16
// How the batched kernel fills its batches (round 7; see k_logs_batch):
//   FILL_LANE   as in round 6: each lane from its own runs (t, t + stride, ..)
//   FILL_ROT    the same, with the runs of each warp rotated among its lanes
//   FILL_POOLS  the warp's lanes pool the classes they find and deal them
//               out in turn; static runs as FILL_ROT
//   FILL_POOL   (default) pooled, and every lane takes its next run from a
//               counter of the launch (L.runctr): no lane runs out early
enum { FILL_LANE = 0, FILL_ROT = 1, FILL_POOLS = 2, FILL_POOL = 3 };

struct LogsParams {
    int d, method;                 // method: LM_TOWER, LM_PH or LM_RR
    u64 M, q, qlow, rmodinv, smax;
    u64 rmod;                      // r mod M (the partner's s0 = r - s0)
    int pair;                      // 1: one Zech log per reciprocal pair
    int pipe;                      // the loop order: 0 round 5's (round 8's
                                   // default), 1 round 9's (w = v * prefix
                                   // product; round 6's before), 2 grouped
                                   // (its own kernel, logs_batch_grouped)
    int fill;                      // round 7: the batches' fill (FILL_*)
    unsigned long long *runctr;    // FILL_POOL: the launch's next run (zeroed
                                   // before each launch)
    int ldcg;                      // round 8: the random table reads (tower T,
                                   // RR L) bypass L1 (__ldcg)
    int prefetch;                  // round 9: L1 prefetches of the local
                                   // arrays ahead of their loads (g^u's
                                   // planes, the batch arrays)
    unsigned long long *prof;      // round 8, -DCOARSE_PHASE_TIMING builds:
                                   // per-phase warp-cycles (PT_*), or null
    u64 lam_m1;                    // lambda(M) - 1: j^-1 = j^(lambda-1) mod M
    u64 fault;                     // test hook: corrupt the log of u = 1 mod fault
    int check;                     // selftest: compute log(g^u), count != u
    int nsub, npf;
    OddDiv sub[4];                 // subfield moduli M/(2^(d/p)-1)
    OddDiv pf[LOGS_MAXPF];         // prime factors of M ...
    int pk[LOGS_MAXPF];            // ... and their multiplicities
    TowerDev tw;
    PHDev ph;
    RRDev rr;
};

GF2_HD u64 logs_dlog(u64 x, const LogsParams &L) {
    return L.method == LM_TOWER ? tower_log(x, L.M, L.tw)
         : L.method == LM_RR    ? rr_log(x, L.q, L.d, L.M, L.rr)
         : L.method == LM_CUBIC ? cubic_log(x, L.M, L.tw)
                                : ph_log(x, L.M, L.d, L.qlow, L.ph);
}

// u (odd, < 2^(d-1)) is the representative of a degree-d class
GF2_HD bool logs_is_rep(u64 u, const LogsParams &L) {
    if (!gf2f_is_min_rotation(u, L.d, L.M)) return false;
    u64 qq;
    for (int k = 0; k < L.nsub; k++)
        if (odd_div_test(u, L.sub[k], &qq)) return false;
    return true;
}

// Rarely executed code stays out of line (round 5): the logs kernels are
// ~90-140 KB of SASS while Turing has a 16 KiB L0 and a 46 KiB L1
// instruction cache, so every hot loop should be contiguous code.
#if defined(__CUDACC__)
#define CS_COLD __device__ __noinline__
#else
#define CS_COLD static __attribute__((noinline))
#endif

// The progression s, s+n, ... <= smax (start s in [1, n]) of the class with
// root g^j: mark it unless nothing on it can improve.  Returns 1 on an
// internal inconsistency (no degree-d minimal polynomial), else 0.
CS_COLD int logs_mark_cold(u64 s, u64 n, u64 j, u64 smax, int d, u64 q,
                           const fw *gpow2, ull *best, Heavy *heavy,
                           unsigned *nheavy, unsigned *overflow, fw *P) {
    u64 count = (smax - s) / n + 1;
    if (count <= INLINE_MARKS && !progression_open(best, s, n, smax, d))
        return 0;
    fw seed = d_antilog(j, q, d, gpow2);        // root g^j
    u64 mask = d_minpoly_seeded(seed, d, q, P);
    if (mask == 0) return 1;
    u64 packed = key_pack(d, mask);
    if (count <= INLINE_MARKS) {
        for (; s <= smax; s += n) atomicMin(&best[s], (ull)packed);
    } else {
        unsigned idx = atomicAdd(nheavy, 1u);
        if (idx < HEAVY_CAP) heavy[idx] = {s, n, packed};
        else atomicAdd(overflow, 1u);
    }
    return 0;
}

// ... unless it is out of range (almost always: the probability is smax/n)
__device__ __forceinline__ void logs_mark(u64 s, u64 n, u64 j,
                                          const LogsParams &L,
                                          const fw *gpow2, ull *best,
                                          Heavy *heavy, unsigned *nheavy,
                                          unsigned *overflow, fw *P,
                                          unsigned long long *err) {
    if (s > L.smax) return;
    *err += logs_mark_cold(s, n, j, L.smax, L.d, L.q, gpow2, best, heavy,
                           nheavy, overflow, P);
}

// spot check of a log: g^v == g^u + 1?  (once per 8 batches)
CS_COLD bool logs_verify_bad(u64 v, u64 u, u64 qlow, int d, const fw *gpow2) {
    return gf2f_antilog(v, qlow, d, gpow2) != (gf2f_antilog(u, qlow, d, gpow2) ^ 1);
}

// log_h of an element of K for x in K* (b = 0; probability 2^-m per class)
CS_COLD u64 tower_klog_cold(u64 yc, TowerDev t) { return tower_klog(yc, t); }

// ---------------------------------------------------------------------------
// Irregular classes, round 5.  A class with gg = gcd(u, M) > 1 has a
// progression only if gg | v, then of modulus n = M/gg with start
// s0 = (v/gg) (j/gg)^-1 mod n.  Round 4 computed that inverse per class by
// extended Euclid (25-29 calls of the software 64-bit division) plus a
// double-and-add; at even d some lane of nearly every warp step had such a
// class (40-60% of the classes are irregular, ~15% need the inverse), and
// the whole warp waited: ~3000-3800 instructions per warp step against
// ~1200-1500 for everything else.  Instead these classes join the batch
// inversion:
//   * gg, its 2-adic inverse ginv and so the EXACT quotients n = M*ginv,
//     jg = j*ginv, vg = v*ginv (mod 2^64) come from the primes that divide
//     u -- one multiply-and-compare per prime;  gg | v  <=>  vg < n;
//   * J = jg + t*n with the least t >= 0 that makes J prime to the primes
//     of gg not dividing n (t = 0 most of the time) is a unit mod M with
//     J == jg (mod n), so J^-1 mod M reduces to jg^-1 mod n (n | M): J
//     enters the batch product like a regular j;
//   * s0 = (vg * J^-1 mod M) mod n, with x mod n = x - n*floor(x*gg/M) and
//     floor(./M) a shift and a compare (mod_divisor_m).
// About 100-300 instructions per warp step remain.
#define VB_SKIP  (~0ULL)        // vb[k]: nothing (more) to do for this class
#define VB_IRREG (1ULL << 63)   // vb[k] = VB_IRREG | vg, gg in xb[k]

// bit i set iff the i-th prime of M divides u (npf tests, no branches;
// npf = gf2_npf(D) is a constant in the kernels, so the loop unrolls)
GF2_HD u32 logs_divmask_n(u64 u, const LogsParams &L, int npf) {
    u32 m = 0;
    GF2D_UNROLL
    for (int i = 0; i < npf; i++) m |= (u32)(u * L.pf[i].pinv <= L.pf[i].plim) << i;
    return m;
}

template <int D>
__device__ __forceinline__ u32 logs_divmask(u64 u, const LogsParams &L) {
    return logs_divmask_n(u, L, gf2_npf(D));
}

// The lift for an irregular class (mask = logs_divmask(u) != 0):
// 0: joins the batch with *J, *vg, *gg;  1: gg does not divide v (no s);
// 2: no lift within 64 tries -- impossible (each prime of gg rules out at
// most one t, and M has at most 15 primes), so an internal error
__device__ __forceinline__ int logs_irreg_lift(u64 u, u64 j, u64 v, u32 mask,
                                               const LogsParams &L, u64 *J,
                                               u64 *vg, u64 *gg_out) {
    u64 gg = 1, ginv = 1;
    u32 full = 0;                          // primes of gg that do not divide n
    for (u32 mk = mask; mk; mk &= mk - 1) {
        const int i = gf2f_ctz32(mk);
        const u64 p = L.pf[i].p, pinv = L.pf[i].pinv;
        u64 t = u * pinv, q;               // u / p
        int e = 1;
        gg *= p; ginv *= pinv;
        while (e < L.pk[i] && odd_div_test(t, L.pf[i], &q)) {
            t = q; e++; gg *= p; ginv *= pinv;
        }
        if (e == L.pk[i]) full |= 1u << i;
    }
    const u64 n = L.M * ginv, w = v * ginv;
    if (w >= n) return 1;
    u64 x = j * ginv;                      // jg: prime to n
    for (int t = 0;; t++) {                // a good t < gg exists (CRT), and
        bool ok = true;                    // x < (t+1)*n <= M while t < gg
        for (u32 f = full; f; f &= f - 1) {
            const OddDiv &P = L.pf[gf2f_ctz32(f)];
            if (x * P.pinv <= P.plim) { ok = false; break; }
        }
        if (ok) break;
        if (t == 63) return 2;
        x += n;
    }
    *J = x; *vg = w; *gg_out = gg;
    return 0;
}

// selftest: the lift on pseudo-random classes with gcd(u, M) > 1, half of
// them with gg | v, against the definition: the lift must report "no s"
// exactly when gg does not divide v, and else give a unit J == j/gg (mod n)
// whose s0 = (vg J^-1 mod M) mod n solves j s0 == v (mod M) with s0 < n
// (checked with the plain double-and-add mulmod, independent of the kernel
// arithmetic).  bad[0] counts failures, bad[1] classes that joined a batch,
// bad[2] classes with no s, bad[3] lifts with t > 0.
__global__ void k_irreg_check(LogsParams L, u64 seed, u64 ntests,
                              unsigned long long *bad) {
    const u64 M = L.M;
    const int d = L.d;
    unsigned long long nb = 0, nj = 0, ns = 0, nt = 0;
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < ntests; i += stride) {
        u64 z = seed + i * 0x9E3779B97F4A7C15ULL;
        auto rnd = [&]() {                      // splitmix64
            u64 x = (z += 0x9E3779B97F4A7C15ULL);
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
        };
        u64 g0 = 1;                             // a random divisor of M
        for (int k = 0; k < L.npf; k++) {
            const u64 r = rnd();
            for (int e = 0; e < L.pk[k] && ((r >> (8 * e)) & 3) == 0; e++)
                g0 *= L.pf[k].p;
        }
        if (g0 == 1) g0 = L.pf[rnd() % L.npf].p;
        if (g0 >= M) continue;
        const u64 u = g0 * (1 + rnd() % (M / g0 - 1 > 0 ? M / g0 - 1 : 1));
        if (u == 0 || u >= M) continue;
        u64 v = rnd() % M;
        if (rnd() & 1) v = v / g0 * g0;         // often divisible
        const u64 j = gf2f_mulmod_m(u, L.rmodinv, d, M);
        const u32 mask = logs_divmask_n(u, L, L.npf);
        if (!mask) { nb++; continue; }          // g0 > 1 divides u: impossible
        const u64 gref = gcd_u64(u, M);         // the reference gcd
        u64 J = 0, wg = 0, gg = 0;
        const int st = logs_irreg_lift(u, j, v, mask, L, &J, &wg, &gg);
        if (v % gref) { ns++; if (st != 1) nb++; continue; }
        if (st != 0 || gg != gref) { nb++; continue; }
        nj++;
        const u64 n = M / gref;
        const u64 nn = M * inv_2adic(gg);
        const u64 Jinv = gf2f_powmod_m(J, L.lam_m1, d, M);
        const u64 s = mod_divisor_m(gf2f_mulmod_m(wg, Jinv, d, M), nn, gg, d, M);
        if (nn != n || wg != v / gref || J >= M || mulmod_u64(J, Jinv, M) != 1 % M)
            nb++;
        else if (s >= n || mulmod_u64(j, s, M) != v) nb++;      // j s0 == v
        else if (J % n != (j / gref) % n) nb++;                 // J == j/gg
        if (J != j / gref) nt++;
    }
    if (nb) atomicAdd(&bad[0], nb);
    if (nj) atomicAdd(&bad[1], nj);
    if (ns) atomicAdd(&bad[2], ns);
    if (nt) atomicAdd(&bad[3], nt);
}

// selftest: the reciprocal pairing against direct computation, on random
// classes u (least rotations of random odd words, aperiodic only).  For
// each: the roles of u and of its partner rep(~u) must be round 6's (the
// reference gf2f_pair_role_ref) and complementary (1/0 or 0/1; 2 only for
// a self-reciprocal class); the engine's log of
// g^-u + 1 must be Z(u) - u; and the partner's progression start computed
// from scratch (j' = -j, v' = Z(-u), s0' = v' j'^-1 mod n by the extended-
// Euclid inverse) must equal (r - s0) mod n.  bad[0] failures, bad[1] pairs
// checked, bad[2] self-reciprocal classes seen.
__global__ void k_pair_check(LogsParams L, const fw *gpow2, u64 seed,
                             u64 ntests, unsigned long long *bad) {
    const u64 M = L.M;
    const int d = L.d;
    unsigned long long nb = 0, np = 0, nsr = 0;
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (; i < ntests; i += stride) {
        u64 z = (seed + i) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        const u64 u = gf2f_min_rotation((z | 1) & M, d, M);
        if (u == 0 || u == M || !logs_is_rep(u, L)) continue;   // a class rep
        const u64 pr = gf2f_min_rotation(~u & M, d, M);
        const int ru = gf2f_pair_role(u, d, M), rp = gf2f_pair_role(pr, d, M);
        // round 7: the fast tie-break gives round 6's roles exactly
        if (ru != gf2f_pair_role_ref(u, d, M) || rp != gf2f_pair_role_ref(pr, d, M)) {
            nb++;
            continue;
        }
        if (pr == u) { nsr++; if (ru != 2) nb++; continue; }
        if (!((ru == 1 && rp == 0) || (ru == 0 && rp == 1))) { nb++; continue; }
        np++;
        const u64 x = gf2f_antilog(u, L.qlow, d, gpow2) ^ 1;    // g^u + 1
        const u64 v = logs_dlog(x, L);
        const u64 um = M - u;                                   // -u
        const u64 vp = logs_dlog(gf2f_antilog(um, L.qlow, d, gpow2) ^ 1, L);
        if (v >= M || vp >= M || vp != (v >= u ? v - u : v + M - u)) { nb++; continue; }
        const u64 j = gf2f_mulmod_m(u, L.rmodinv, d, M), jp = M - j;
        const u64 gg = gcd_u64(u, M), n = M / gg;
        if (v % gg) continue;                   // no progression for either
        const u64 s0 = MULMOD(inv_mod((j / gg) % n, n), v / gg, n);
        const u64 s0p = MULMOD(inv_mod((jp / gg) % n, n), vp / gg, n);
        const u64 rn = L.rmod % n;
        if (s0p != (rn >= s0 ? rn - s0 : rn + n - s0)) nb++;
    }
    if (nb) atomicAdd(&bad[0], nb);
    if (np) atomicAdd(&bad[1], np);
    if (nsr) atomicAdd(&bad[2], nsr);
}

// ---------------------------------------------------------------------------
// The batched kernel (round 3).  Each thread enumerates the Lyndon words of
// its runs with FKM -- exactly the class representatives, nothing else to
// test -- and handles the canonical ones of each reciprocal pair 32 at a
// time (bit-sliced, see gf2_dlog.h):
//   * g^u for all 32 by bs_pow_g (squarings and plane shifts only), then
//     x = g^u + 1;
//   * Pohlig-Hellman (odd d): x^(E_i) bit-sliced for each part, back to
//     scalars, one table lookup per class and part, CRT;
//     tower (even d): back to scalars, tower_log per class;
//   * progression starts v * j^-1 for all 32 with ONE modular exponentiation
//     (Montgomery's trick: invert the product of the j's, then peel off each
//     inverse with two multiplies).
//   * then each class's progression, and its partner's: s == r - s0 (mod n).
#define LB 32

// Per-phase timing of the batched kernel (round 8; a build option,
// -DCOARSE_PHASE_TIMING, so the production kernels are untouched): lane 0
// of every warp adds the clock64() cycles of each phase to L.prof[PT_*].
// The warp runs in lockstep, so these are wall-clock cycles of the warp,
// stalls included -- where a warp's time goes, not what each lane computes.
enum { PT_FILL = 0, PT_GU, PT_LOGS, PT_INV, PT_MARK, PT_REST, PT_NPH,
       PT_BATCH = PT_NPH, PT_ENTRY, PT_N };
#if defined(COARSE_PHASE_TIMING)
#define PT_PARAM , long long &pt_t
#define PT_ARG , pt_t
#define PT_INIT long long pt_t = clock64()
#define PT(ph) pt_mark(L.prof, pt_t, (ph))
#define PT_COUNT(i, v) do { if (L.prof) atomicAdd(&L.prof[i], (unsigned long long)(v)); } while (0)
__device__ __forceinline__ void pt_mark(unsigned long long *prof, long long &t,
                                        int ph) {
    const long long now = clock64();
    if (prof && (threadIdx.x & 31u) == 0)
        atomicAdd(&prof[ph], (unsigned long long)(now - t));
    t = now;
}
#else
#define PT_PARAM
#define PT_ARG
#define PT_INIT do {} while (0)
#define PT(ph) do {} while (0)
#define PT_COUNT(i, v) do {} while (0)
#endif

// A random read of a big table (tower T, RR L): with L.ldcg (round 8, the
// default) past L1, where its line would never be read again and would only
// evict the batch arrays' local memory -- 64-80% of the L1 traffic in the
// round-7 profiles.  COARSE_LDCG=0 reads through L1 as before.
__device__ __forceinline__ u64 logs_ld(const u64 *p, int cg) {
#if defined(__CUDA_ARCH__)
    return cg ? (u64)__ldcg((const unsigned long long *)p) : *p;
#else
    (void)cg;
    return *p;
#endif
}

// Stage A of a class's log (tower, RR, cubic): the table reads -- the
// tower's coordinates, inner logs and T[l]; RR's split and L[r], L[t]; the
// cubic tower's K-logs and its two reads of T3.  Stage V (logs_stage_v)
// combines them: tower h1*lb + tv, RR tv - tv2... see there.
template <int D, int METHOD>
__device__ __forceinline__ void logs_stage_a(u64 x, const LogsParams &L,
                                             const TowerDev &tw, u64 &lb,
                                             u64 &tv, u64 &tv2, bool &inK) {
    if constexpr (METHOD == LM_TOWER) {
        u64 l;
        const u64 pk = tower_coords(x, tw);
        inK = !tower_split(pk, tw, &lb, &l);
        if (inK) { lb = tower_klog_cold(pk & tw.MK, tw); tv = 0; }  // x in K*
        else tv = logs_ld(tw.T + l, L.ldcg);
        tv2 = 0;
    } else if constexpr (METHOD == LM_CUBIC) {  // round 9: x = r/t over K
        u64 ir, it;                             // (T3 slots; MK + 1 holds 0)
        lb = cubic_parts<D>(x, tw, &ir, &it);
        tv = logs_ld(tw.cb.T3 + ir, L.ldcg);
        tv2 = logs_ld(tw.cb.T3 + it, L.ldcg);
        inK = false;
    } else {                                    // RR: x = r/t, two reads
        u64 rr_r;
        u32 rr_t;
        rr_split<D>(x, &rr_r, &rr_t);
        lb = 0;
        tv = logs_ld(L.rr.L + rr_r, L.ldcg);
        tv2 = logs_ld(L.rr.L + rr_t, L.ldcg);
        inK = false;
    }
}

// v = the class's log from stage A's values: tower h1*lb + tv (h1*lb alone
// for x in K*); RR tv - tv2 (log r - log t); cubic lb + tv - tv2 (lb the
// K-part of log r - log t, tv and tv2 the T3 entries of r and t)
template <int METHOD>
__device__ __forceinline__ u64 logs_stage_v(u64 lb, u64 tv, u64 tv2, bool inK,
                                            u64 M, const TowerDev &tw) {
    if constexpr (METHOD == LM_TOWER) {
        const u64 h1 = tw.MK + 2;               // log_g(h) = 2^m + 1
        const u64 v = inK ? h1 * lb : h1 * lb + tv;
        return v >= M ? v - M : v;
    } else if constexpr (METHOD == LM_CUBIC) {
        u64 s = lb + tv;                        // < 2M
        if (s >= M) s -= M;
        return s >= tv2 ? s - tv2 : s + M - tv2;
    } else {
        (void)lb;
        return tv >= tv2 ? tv - tv2 : tv + M - tv2;    // RR: log r - log t
    }
}

// The rest of a class's forward step once its log v is known (round 5's
// order): the fault hook, the spot check, the irregular lift; sets vb[k],
// xb[k], ub[k] = J and multiplies J into acc (the caller sets tb[k], the
// prefix product before the class)
template <int D>
__device__ __forceinline__ void logs_fwd_tail(const LogsParams &L, int k,
                                              u64 u, u64 v, u64 j, u32 mask,
                                              bool verify, const fw *gpow2,
                                              u64 (&ub)[LB], u64 (&vb)[LB],
                                              u64 (&xb)[LB], u64 &acc,
                                              bool &any,
                                              unsigned long long *err) {
    const u64 M = L.M;
    if (L.fault && u % L.fault == 1) v ^= 1;        // fault injection (tests)
    if (k == 0 && verify && v < M && logs_verify_bad(v, u, L.qlow, D, gpow2))
        v = ~0ULL;                                  // spot check failed
    if (v >= M) { (*err)++; vb[k] = VB_SKIP; return; }     // inconsistent log
    u64 J = j;
    if (mask) {                                     // gcd(u, M) > 1
        u64 vg = 0, gg = 0;
        const int st = logs_irreg_lift(u, j, v, mask, L, &J, &vg, &gg);
        if (st) {                       // 1: no s, for u or its partner
            if (st == 2) (*err)++;      // (impossible; see the lift)
            vb[k] = VB_SKIP;
            return;
        }
        vb[k] = VB_IRREG | vg;
        xb[k] = gg;
    } else {
        vb[k] = v;
    }
    ub[k] = J;
    acc = gf2f_mulmod_m(acc, J, D, M);
    any = true;
}

// Round 9's order (L.pipe = 1), the same with two values stored where round
// 5's stores three.  Montgomery's backward step for class k takes
// jinv = inv * t_k (t_k the prefix product before k), s0 = v * jinv and
// inv <- inv * J, with v, J, t_k and gg loaded back from the batch arrays.
// Here the forward step keeps w = v * t_k in vb[k] (vg * t_k for an irregular
// class) and nothing in tb: the backward step's s0 = inv * w directly -- one
// multiply moves from the backward step to the forward one -- and it loads w
// and J (and gg where irregular classes are common, logs_carry_gg).  In the
// round-7 profiles these arrays are 64-80% of the L1 traffic, most of it
// missing L1 and much of it L2; a class makes 9 local accesses instead of
// 12 (10 instead of 12 with gg).
template <int D>
__device__ __forceinline__ void logs_fwd_tail9(const LogsParams &L, int k,
                                               u64 u, u64 v, u64 j, u32 mask,
                                               bool verify, const fw *gpow2,
                                               u64 (&ub)[LB], u64 (&vb)[LB],
                                               u64 (&xb)[LB], u64 &acc,
                                               bool &any,
                                               unsigned long long *err) {
    const u64 M = L.M;
    if (L.fault && u % L.fault == 1) v ^= 1;        // fault injection (tests)
    if (k == 0 && verify && v < M && logs_verify_bad(v, u, L.qlow, D, gpow2))
        v = ~0ULL;                                  // spot check failed
    if (v >= M) { (*err)++; vb[k] = VB_SKIP; return; }     // inconsistent log
    u64 J = j, w;
    if (mask) {                                     // gcd(u, M) > 1
        u64 vg = 0, gg = 0;
        const int st = logs_irreg_lift(u, j, v, mask, L, &J, &vg, &gg);
        if (st) {                       // 1: no s, for u or its partner
            if (st == 2) (*err)++;      // (impossible; see the lift)
            vb[k] = VB_SKIP;
            return;
        }
        w = VB_IRREG | gf2f_mulmod_m(vg, acc, D, M);    // vg * t_k
        xb[k] = gg;
    } else {
        w = gf2f_mulmod_m(v, acc, D, M);                // v * t_k
    }
    vb[k] = w;
    ub[k] = J;
    acc = gf2f_mulmod_m(acc, J, D, M);
    any = true;
}

// Whether the backward steps of round 9's orders carry gg with every class
// (irregular classes common: 3 | 2^D - 1 for even D, 7 | 2^D - 1 for 3 | D)
// or load it for an irregular class alone (rare: a dependent load, in ~0.1-
// 0.5% of the classes at d = 41..55 otherwise).
template <int D> __device__ __forceinline__ constexpr bool logs_carry_gg() {
    return D % 2 == 0 || D % 3 == 0;
}

// A class's marks, from S = v J^-1 mod M (regular; modulus M) or
// S = vg J^-1 mod M (irregular: modulus n = M/gg), and its partner's
// (round 6: s == r - s0, the same modulus)
template <int D>
__device__ __forceinline__ void logs_mark_pair(const LogsParams &L, bool irreg,
                                               u64 S, u64 J, u64 gg,
                                               bool paired, const fw *gpow2,
                                               ull *best, Heavy *heavy,
                                               unsigned *nheavy,
                                               unsigned *overflow, fw *P,
                                               unsigned long long *err) {
    const u64 M = L.M;
    // the partner (round 6): root g^-j, Z(-u) = Z(u) - u, so it solves
    // -j s == v - u, i.e. s == r - s0 (mod n): same modulus, no new log
    if (!irreg) {
        logs_mark(S == 0 ? M : S, M, J, L, gpow2, best, heavy, nheavy,
                  overflow, P, err);
        if (paired) {
            const u64 sp = L.rmod >= S ? L.rmod - S : L.rmod + M - S;
            logs_mark(sp == 0 ? M : sp, M, M - J, L, gpow2, best, heavy,
                      nheavy, overflow, P, err);
        }
    } else {                                        // modulus n = M/gg
        const u64 nn = M * inv_2adic(gg);
        const u64 s0 = mod_divisor_m(S, nn, gg, D, M);  // vg * jg^-1 mod n
        const u64 j = gg * mod_divisor_m(J, nn, gg, D, M);
        if ((s0 == 0 ? nn : s0) <= L.smax)
            logs_mark(s0 == 0 ? nn : s0, nn, j, L, gpow2, best, heavy,
                      nheavy, overflow, P, err);
        if (paired) {
            const u64 rn = mod_divisor_m(L.rmod, nn, gg, D, M);
            const u64 sp = rn >= s0 ? rn - s0 : rn + nn - s0;
            if ((sp == 0 ? nn : sp) <= L.smax)
                logs_mark(sp == 0 ? nn : sp, nn, M - j, L, gpow2, best,
                          heavy, nheavy, overflow, P, err);
        }
    }
}

// Round 9 (L.prefetch, COARSE_PREFETCH): L1 prefetches of the batch arrays ahead
// of their loads -- two classes ahead in the forward loops (each class's
// log takes ~600-1300 instructions), LOGS_PF_BACK in the backward ones
// (~100-200 per class), whose loads made 12-17% of the round-7 stall
// samples: the values were written a whole forward pass earlier, and are
// back in L2 or DRAM.
#define LOGS_PF_BACK 4

template <int D, int METHOD>
__device__ __forceinline__ void logs_batch(const LogsParams &L,
                                           const TowerDev &tw,
                                           const fw *gpow2, u64 (&ub)[LB],
                                           u64 (&vb)[LB], u64 (&tb)[LB],
                                           u64 (&xb)[LB], int n, u32 pairm,
                                           bool verify, ull *best,
                                           Heavy *heavy, unsigned *nheavy,
                                           unsigned *overflow, fw *P,
                                           unsigned long long *err PT_PARAM) {
    const u64 M = L.M;
    for (int k = n; k < LB; k++) ub[k] = 1;     // padding; results unused
    u32 X[D];
    {
        u32 U[D];
        bs_load<D>(ub, U);
        bs_pow_g<D>(U, X, L.prefetch != 0);     // g^u
    }
    if (!L.check) X[0] = ~X[0];                 // x = g^u + 1
    PT(PT_GU);
    if (METHOD == LM_PH) {
        for (int k = 0; k < LB; k++) vb[k] = 0;
        for (int i = 0; i < L.ph.nparts; i++) {
            const PHPart &pt = L.ph.part[i];
            {
                u32 A[D];
                bs_pow<D>(X, pt.E, A);          // in the order-P_i subgroup
                bs_store<D>(A, tb);
            }
            for (int k = 0; k < n; k++) {
                if (vb[k] == ~0ULL) continue;
                const u64 xi = ph_part_log(pt, tb[k], L.qlow, D);
                if (xi == ~0ULL) { vb[k] = ~0ULL; continue; }
                const u64 t = vb[k] + gf2f_mulmod_m(xi, pt.K, D, M);
                vb[k] = t >= M ? t - M : t;
            }
        }
    } else {
        bs_store<D>(X, vb);                     // x = g^u + 1; logs below
    }
    if (L.check) {                              // selftest: log(g^u) == u?
        if constexpr (METHOD == LM_PH) {
            for (int k = 0; k < n; k++) if (vb[k] != ub[k]) (*err)++;
        } else {                                // (through the sieve's stages)
            for (int k = 0; k < n; k++) {
                u64 lb = 0, tv = 0, tv2 = 0;
                bool inK = false;
                logs_stage_a<D, METHOD>(vb[k], L, tw, lb, tv, tv2, inK);
                if (logs_stage_v<METHOD>(lb, tv, tv2, inK, M, tw) != ub[k]) (*err)++;
            }
        }
        return;
    }
    // forward: ub[k] <- J, the class's member of the batch product (j itself
    // for a regular class); round 5 (L.pipe = 0): tb[k] = the product of the
    // members before k, vb[k] <- v, VB_IRREG | vg, or VB_SKIP; round 9
    // (L.pipe = 1): vb[k] <- w = v * tb[k] (see logs_fwd_tail9), tb unused.
    // An in-order GPU stalls a warp at the first use of a pending load, so
    // each iteration loads the next class's x and u first, and issues its
    // own table reads before the work that does not depend on them (j, the
    // divisibility mask).  (Round 6's order, the reads one class ahead, gave
    // nothing on the 2070 SUPER and was replaced by round 9's; the grouped
    // order, L.pipe = 2, is logs_batch_grouped below.)
    const bool r9 = L.pipe == 1;
    u64 acc = 1;
    bool any = false;
    {
        u64 xn = vb[0], un = ub[0];
        for (int k = 0; k < n; k++) {
            const u64 x = xn, u = un;
            if (k + 1 < n) { xn = vb[k + 1]; un = ub[k + 1]; }
            if (L.prefetch && k + 2 < n) { pf_local(&vb[k + 2]); pf_local(&ub[k + 2]); }
            if (!r9) tb[k] = acc;
            u64 v = x, lb = 0, tv = 0, tv2 = 0;     // PH: x is the log
            bool inK = false;
            if constexpr (METHOD != LM_PH)
                logs_stage_a<D, METHOD>(x, L, tw, lb, tv, tv2, inK);
            const u64 j = gf2f_mulmod_m(u, L.rmodinv, D, M);
            const u32 mask = logs_divmask<D>(u, L);
            if constexpr (METHOD != LM_PH)
                v = logs_stage_v<METHOD>(lb, tv, tv2, inK, M, tw);
            if (r9)
                logs_fwd_tail9<D>(L, k, u, v, j, mask, verify, gpow2, ub, vb,
                                  xb, acc, any, err);
            else
                logs_fwd_tail<D>(L, k, u, v, j, mask, verify, gpow2, ub, vb,
                                 xb, acc, any, err);
        }
    }
    PT(PT_LOGS);
    if (!any) return;
    u64 inv = gf2f_powmod_m(acc, L.lam_m1, D, M);   // (prod J)^-1
    PT(PT_INV);
    // backward (Montgomery): from (J_0 .. J_k)^-1, class k's inverse, then
    // inv <- inv * J_k; at the end inv == 1, which round 9 checks (a wrong
    // inversion stops the run like a wrong log)
    if (r9) {
        constexpr bool kGG = logs_carry_gg<D>();
        u64 wn = vb[n - 1], Jn = ub[n - 1], gn = kGG ? xb[n - 1] : 1;
        for (int k = n - 1; k >= 0; k--) {
            const u64 w = wn, J = Jn, g = gn;
            if (k > 0) {
                wn = vb[k - 1]; Jn = ub[k - 1];
                if (kGG) gn = xb[k - 1];
            }
            if (L.prefetch && k >= LOGS_PF_BACK) {    // (a short loop: further ahead)
                pf_local(&vb[k - LOGS_PF_BACK]); pf_local(&ub[k - LOGS_PF_BACK]);
                if (kGG) pf_local(&xb[k - LOGS_PF_BACK]);
            }
            if (w == VB_SKIP) continue;
            const u64 S = gf2f_mulmod_m(inv, w & ~VB_IRREG, D, M);  // v J^-1
            inv = gf2f_mulmod_m(inv, J, D, M);
            const bool irr = (w & VB_IRREG) != 0;
            logs_mark_pair<D>(L, irr, S, J, irr ? (kGG ? g : xb[k]) : 1,
                              (pairm >> k) & 1, gpow2, best, heavy, nheavy,
                              overflow, P, err);
        }
    } else {
        u64 vn = vb[n - 1], Jn = ub[n - 1], tn = tb[n - 1], gn = xb[n - 1];
        for (int k = n - 1; k >= 0; k--) {
            const u64 vv = vn, J = Jn, t = tn, gg = gn;
            if (k > 0) { vn = vb[k - 1]; Jn = ub[k - 1]; tn = tb[k - 1]; gn = xb[k - 1]; }
            if (L.prefetch && k >= LOGS_PF_BACK) {
                pf_local(&vb[k - LOGS_PF_BACK]); pf_local(&ub[k - LOGS_PF_BACK]);
                pf_local(&tb[k - LOGS_PF_BACK]); pf_local(&xb[k - LOGS_PF_BACK]);
            }
            if (vv == VB_SKIP) continue;
            const u64 jinv = gf2f_mulmod_m(inv, t, D, M);
            inv = gf2f_mulmod_m(inv, J, D, M);
            const u64 S = gf2f_mulmod_m(vv & ~VB_IRREG, jinv, D, M);
            logs_mark_pair<D>(L, (vv & VB_IRREG) != 0, S, J, gg,
                              (pairm >> k) & 1, gpow2, best, heavy, nheavy,
                              overflow, P, err);
        }
    }
    if (inv != 1) (*err)++;                     // inv * prod J == 1
    PT(PT_MARK);
}

// The grouped loop order, round 9 (L.pipe = 2; tower, RR and the cubic tower
// at d >= 40 on the GPU).  Round 8's grouped order kept everything a class
// carries from its log to its marking in registers, in groups of LG = 8
// classes with one inversion each -- and unrolled the group's loops to index
// them by constants.  Eight copies of a class's code (105-178 KB of SASS)
// cost it the instruction cache, and the pressure spilled the enumeration's
// state: it won by 1-3% at the RR degrees and lost at the towers.  Round 9
// keeps the group's w, J (and gg) in registers but its loops rolled: a class
// is written to, and read back from, its registers through a switch on its
// place in the group (grp_put / grp_get), which costs a few instructions and
// keeps one copy of the code.  So a class makes 5 local accesses (u: 3, x: 2)
// against round 5's 12, for four inversions per batch instead of one (~45-66
// instructions per class at d = 47-55, by the exponent lambda(M) - 1).
// Also checked: each group's inverse, inv * prod J == 1 after its marking.
#define LG 8

template <bool G3>
__device__ __forceinline__ void grp_put(int i, u64 w, u64 J, u64 g,
                                        u64 (&W)[LG], u64 (&Jg)[LG],
                                        u64 (&Gg)[LG]) {
    switch (i) {
#define GRP_PUT(s) case s: W[s] = w; Jg[s] = J; if (G3) Gg[s] = g; break;
    GRP_PUT(0) GRP_PUT(1) GRP_PUT(2) GRP_PUT(3)
    GRP_PUT(4) GRP_PUT(5) GRP_PUT(6) GRP_PUT(7)
#undef GRP_PUT
    default: break;
    }
}

template <bool G3>
__device__ __forceinline__ void grp_get(int i, u64 &w, u64 &J, u64 &g,
                                        const u64 (&W)[LG], const u64 (&Jg)[LG],
                                        const u64 (&Gg)[LG]) {
    switch (i) {
#define GRP_GET(s) case s: w = W[s]; J = Jg[s]; g = G3 ? Gg[s] : 1; break;
    GRP_GET(0) GRP_GET(1) GRP_GET(2) GRP_GET(3)
    GRP_GET(4) GRP_GET(5) GRP_GET(6) GRP_GET(7)
#undef GRP_GET
    default: w = VB_SKIP; J = 1; g = 1; break;
    }
}

template <int D, int METHOD>
__device__ __forceinline__ void logs_batch_grouped(const LogsParams &L,
                                                   const TowerDev &tw,
                                                   const fw *gpow2, u64 (&ub)[LB],
                                                   u64 (&vb)[LB], u64 (&xb)[LB],
                                                   int n, u32 pairm,
                                                   bool verify, ull *best,
                                                   Heavy *heavy, unsigned *nheavy,
                                                   unsigned *overflow, fw *P,
                                                   unsigned long long *err PT_PARAM) {
    static_assert(METHOD != LM_PH, "the grouped order is for tower, RR, cubic");
    const u64 M = L.M;
    for (int k = n; k < LB; k++) ub[k] = 1;     // padding; results unused
    u32 X[D];
    {
        u32 U[D];
        bs_load<D>(ub, U);
        bs_pow_g<D>(U, X, L.prefetch != 0);     // g^u
    }
    if (!L.check) X[0] = ~X[0];                 // x = g^u + 1
    PT(PT_GU);
    bs_store<D>(X, vb);
    if (L.check) {                              // selftest: log(g^u) == u?
        for (int k = 0; k < n; k++) {
            u64 lb = 0, tv = 0, tv2 = 0;
            bool inK = false;
            logs_stage_a<D, METHOD>(vb[k], L, tw, lb, tv, tv2, inK);
            if (logs_stage_v<METHOD>(lb, tv, tv2, inK, M, tw) != ub[k]) (*err)++;
        }
        return;
    }
    constexpr bool kGG = logs_carry_gg<D>();
    u64 xn = vb[0], un = ub[0];                 // one class ahead, across groups
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
    for (int g0 = 0; g0 < n; g0 += LG) {
        const int ng = n - g0 < LG ? n - g0 : LG;
        u64 Wg[LG], Jg[LG], Gg[LG];             // constant indices only
        GF2D_UNROLL
        for (int s = 0; s < LG; s++) { Wg[s] = VB_SKIP; Jg[s] = 1; Gg[s] = 1; }
        u64 acc = 1;
        bool any = false;
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
        for (int i = 0; i < ng; i++) {
            const int k = g0 + i;
            const u64 x = xn, u = un;
            if (k + 1 < n) { xn = vb[k + 1]; un = ub[k + 1]; }
            if (L.prefetch && k + 2 < n) { pf_local(&vb[k + 2]); pf_local(&ub[k + 2]); }
            u64 lb = 0, tv = 0, tv2 = 0;
            bool inK = false;
            logs_stage_a<D, METHOD>(x, L, tw, lb, tv, tv2, inK);
            const u64 j = gf2f_mulmod_m(u, L.rmodinv, D, M);
            const u32 mask = logs_divmask<D>(u, L);
            u64 v = logs_stage_v<METHOD>(lb, tv, tv2, inK, M, tw);
            if (L.fault && u % L.fault == 1) v ^= 1;        // fault injection
            if (k == 0 && verify && v < M && logs_verify_bad(v, u, L.qlow, D, gpow2))
                v = ~0ULL;                                  // spot check failed
            u64 w = VB_SKIP, J = 1, gg = 1;
            if (v >= M) {
                (*err)++;                                   // inconsistent log
            } else if (mask) {                              // gcd(u, M) > 1
                u64 vg = 0, Jl = j;
                const int st = logs_irreg_lift(u, j, v, mask, L, &Jl, &vg, &gg);
                if (st == 2) (*err)++;          // (impossible; see the lift)
                if (!st) {
                    J = Jl;
                    w = VB_IRREG | gf2f_mulmod_m(vg, acc, D, M);
                    if (!kGG) xb[k] = gg;       // (rare: not in the registers)
                }
            } else {
                J = j;
                w = gf2f_mulmod_m(v, acc, D, M);
            }
            if (w != VB_SKIP) { acc = gf2f_mulmod_m(acc, J, D, M); any = true; }
            grp_put<kGG>(i, w, J, gg, Wg, Jg, Gg);
        }
        PT(PT_LOGS);
        if (!any) continue;
        u64 inv = gf2f_powmod_m(acc, L.lam_m1, D, M);   // (prod J)^-1
        PT(PT_INV);
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
        for (int i = ng - 1; i >= 0; i--) {
            u64 w = VB_SKIP, J = 1, g = 1;
            grp_get<kGG>(i, w, J, g, Wg, Jg, Gg);
            if (w == VB_SKIP) continue;
            const int k = g0 + i;
            const u64 S = gf2f_mulmod_m(inv, w & ~VB_IRREG, D, M);  // v J^-1
            inv = gf2f_mulmod_m(inv, J, D, M);
            const bool irr = (w & VB_IRREG) != 0;
            logs_mark_pair<D>(L, irr, S, J, irr ? (kGG ? g : xb[k]) : 1,
                              (pairm >> k) & 1, gpow2, best, heavy, nheavy,
                              overflow, P, err);
        }
        if (inv != 1) (*err)++;                 // inv * prod J == 1
        PT(PT_MARK);
    }
}

#if defined(__CUDA_ARCH__)
// Copy the tower's small tables -- btab, and for the subfield tower LKn and
// (if t.shm_tp) TP -- into dynamic shared memory and point t at the copies.
// Every thread of the block must call this; __syncthreads() follows.
__device__ __forceinline__ void tower_stage_shared(TowerDev &t,
                                                   unsigned char *shm) {
    u64 *bt = (u64 *)shm;
    const int nbw = t.nb * 256;
    for (int i = threadIdx.x; i < nbw; i += blockDim.x) bt[i] = t.btab[i];
    t.btab = bt;
    unsigned char *p = shm + (size_t)nbw * sizeof(u64);
    if (t.sub_k) {
        const int nw = 1 << (t.sub_n - 1);      // 2^n u16 = 2^(n-1) u32
        u32 *lk = (u32 *)p;
        const u32 *src = (const u32 *)t.LKn;
        for (int i = threadIdx.x; i < nw; i += blockDim.x) lk[i] = src[i];
        t.LKn = (const uint16_t *)p;
        p += (size_t)nw * sizeof(u32);
        if (t.shm_tp) {
            u32 *tp = (u32 *)p;
            for (u32 i = threadIdx.x; i < t.tp_n; i += blockDim.x) tp[i] = t.TP[i];
            t.TP = tp;
        }
    }
}

// The cubic tower's (round 9): its byte tables, and the classic tower of
// K_n -- byte tables, LK (2^(n/2) u32), T (2^(n/2) + 1 u64) -- 26 KB at
// d=54 (cubic_shm_bytes); T3 stays in global memory (L2).
__device__ __forceinline__ void cubic_stage_shared(TowerDev &t,
                                                   unsigned char *shm) {
    u64 *bt = (u64 *)shm;                       // 8-byte entries first
    const int n3 = t.cb.nb * 256, nk = t.nb * 256, nT = (int)t.MK + 1;
    for (int i = threadIdx.x; i < n3; i += blockDim.x) bt[i] = t.cb.btab[i];
    for (int i = threadIdx.x; i < nk; i += blockDim.x) bt[n3 + i] = t.btab[i];
    for (int i = threadIdx.x; i < nT; i += blockDim.x) bt[n3 + nk + i] = t.T[i];
    u32 *lk = (u32 *)(bt + n3 + nk + nT);
    for (int i = threadIdx.x; i < nT; i += blockDim.x) lk[i] = t.LK[i];
    t.cb.btab = bt;
    t.btab = bt + n3;
    t.T = bt + n3 + nk;
    t.LK = lk;
}
#endif

// the shared bytes cubic_stage_shared needs
static inline u32 cubic_shm_bytes(const TowerDev &t) {
    return (u32)(((size_t)t.cb.nb * 256 + (size_t)t.nb * 256 + t.MK + 1) * 8 +
                 (t.MK + 1) * 4);
}

// warp-synchronous helpers (round 7): the real intrinsics under nvcc, the
// lockstep-warp emulation of cuda_emul.h otherwise
#if defined(__CUDACC__)
#define WARP_BALLOT(p)  __ballot_sync(0xffffffffu, (p))
#define WARP_SHFL(v, s) __shfl_sync(0xffffffffu, (unsigned long long)(v), (s))
#else
#define WARP_BALLOT(p)  emul_ballot_sync(0xffffffffu, (p))
#define WARP_SHFL(v, s) emul_shfl_sync(0xffffffffu, (unsigned long long)(v), (s))
#endif

// the position of the r-th set bit of m, counting from 0 (r < popc(m); any
// value in 0..31 otherwise)
GF2_HD int warp_select(u32 m, int r) {
    int pos = 0, c = gf2f_popc32(m & 0xffffu);
    if (r >= c) { r -= c; pos += 16; m >>= 16; }
    c = gf2f_popc32(m & 0xffu);
    if (r >= c) { r -= c; pos += 8; m >>= 8; }
    c = gf2f_popc32(m & 0xfu);
    if (r >= c) { r -= c; pos += 4; m >>= 4; }
    c = gf2f_popc32(m & 0x3u);
    if (r >= c) { r -= c; pos += 2; m >>= 2; }
    if (r >= (int)(m & 1u)) pos += 1;
    return pos;
}

// The k-th run of a thread, rr = t + k * stride (round 6's order).  With
// rot (FILL_ROT, FILL_POOLS) the runs of each aligned group of 32 -- one
// per lane of a warp, all at the same k -- are rotated by k, so a lane's
// runs take every place in the group in turn: in round 6's order all of a
// lane's runs share the 5 bits of u just above the run, so their classes
// differ systematically from lane to lane (some lanes had half the work of
// others, and idled through their warp's last batches).  A bijection on
// each group (stride is a multiple of 32); a partial last group stays.
GF2_HD u64 logs_run_at(u64 rr, u64 k, u64 nruns, bool rot) {
    if (!rot || (rr | 31) >= nruns) return rr;
    return (rr & ~(u64)31) | ((rr + k) & 31);
}

// Sweep the Lyndon words in [ulo, uhi), in runs of `run` u values; tower
// kernels run 512-thread blocks (one per SM, sharing one copy of the small
// tables in shared memory), PH kernels 128-thread blocks (up to ~250
// registers).  The batches (LB canonical classes per lane) are filled in
// one of the FILL_* ways (L.fill):
//   lane (round 6): each lane enumerates its own runs into its own batch.
//          The warp runs each batch when its slowest lane is full -- and
//          the lanes' supply of canonical classes differs a lot (the
//          canonical share depends on the ones in u's high bits), so most
//          enumeration steps had a third of the lanes active, and lanes
//          that finished their runs early sat out the warp's last batches:
//          27-38% of the batch slots were empty over a real pass (but 1-2%
//          in the tuner's dense sample), by a lockstep simulation of the
//          warps on sampled chunks of d = 40-47 (tools/fillsim.cpp).
//   pool (round 7): every lane steps its own enumeration once per
//          iteration, and the canonical classes the warp finds are dealt to
//          the lanes in turn (a ballot, then each lane fetches its class
//          from the finder with a shuffle), so the lanes fill together and
//          every batch is full but the last: by the same simulation ~2
//          lane-steps of the loop per class instead of 7.5-14, and no empty
//          slots.
//          A lane whose class finds no room keeps it for the next round.
//          FILL_POOL also takes each run from the launch's counter, so
//          lanes and warps run out of work together at the end.
// Round 9, the grouped order's kernels: the fill's state crosses the batch
// parked in local memory -- one store and one load of each value per batch
// of 32 classes -- instead of in registers, which the grouped batch needs
// for its groups (ptxas spilled the state instead, inside the fill loop,
// reloading and storing it at every enumeration step).  volatile, so the
// values do not stay in registers across the batch.
struct FillPark { u64 rr, kr, w, whi, pw, surv, pnb; u32 flags; };

template <int D, int METHOD, int GROUP = 0>
__global__ void __launch_bounds__(METHOD == LM_PH ? 128 : 512, 1)
k_logs_batch(LogsParams L, const fw *gpow2, u64 ulo, u64 uhi,
             u64 run, ull *best, Heavy *heavy,
             unsigned *nheavy, unsigned *overflow,
             unsigned long long *nsurv,
             unsigned long long *nerr) {
    PT_INIT;
    TowerDev tw = L.tw;
#if defined(__CUDA_ARCH__)
    if (METHOD == LM_TOWER && tw.shm_bytes) {
        extern __shared__ __align__(16) unsigned char logs_shm[];
        tower_stage_shared(tw, logs_shm);
        __syncthreads();
    }
    if (METHOD == LM_CUBIC && tw.cb.shm_bytes) {
        extern __shared__ __align__(16) unsigned char logs_shm[];
        cubic_stage_shared(tw, logs_shm);
        __syncthreads();
    }
#endif
    PT(PT_REST);
    const u64 nruns = (uhi - ulo + run - 1) / run;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    const bool dyn = L.fill == FILL_POOL;
    const bool rot = L.fill == FILL_ROT || L.fill == FILL_POOLS;
    u64 rr = dyn ? 0 : (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 kr = 0;                                 // runs taken (static order)
    u64 ub[LB], vb[LB], tb[LB], xb[LB];
    for (int k = 0; k < LB; k++) xb[k] = 1;     // read ahead before written
    fw P[64];
    unsigned long long my_surv = 0, my_err = 0;
    unsigned nb = 0;
    u64 w = 0, whi = 0;
    int p = 0;
    bool have = false;
    if (L.fill >= FILL_POOLS) {
        // ---- pooled fill ----
        const unsigned lane = threadIdx.x & 31u;
        const u32 below = (1u << lane) - 1;     // the lanes before this one
        u64 pw = 0;                             // the pending class; bit 63:
        bool pend = false, live = true;         //   it has a partner
        for (;;) {                              // rounds (warp-uniform)
            int n = 0;
            u32 pairm = 0;                      // bit k: ub[k] has a partner
            u32 total = 0;                      // classes dealt this round
            for (;;) {
                if (!pend && live) {            // this lane's next step
                    if (!have) {
                        if (dyn) rr = atomicAdd(L.runctr, 1ULL);
                        if (rr >= nruns) live = false;
                        else {
                            const u64 lo = ulo + logs_run_at(rr, kr, nruns, rot) * run;
                            whi = uhi - lo > run ? lo + run : uhi;
                            if (!dyn) { rr += stride; kr++; }
                            w = fkm_seek(lo, D, &p);
                            have = true;
                        }
                    }
                    if (have) {
                        if (w >= whi) have = false;
                        else {
                            if (p == D) {       // a Lyndon word: a class
                                const int role = L.pair ? gf2f_pair_role(w, D, L.M) : 2;
                                if (role) {
                                    pend = true;
                                    pw = w | (role == 1 ? (u64)1 << 63 : 0);
                                    my_surv += role == 1 ? 2 : 1;   // classes covered
                                }
                            }
                            w = fkm_next(w, D, &p);
                        }
                    }
                }
                const u32 found = WARP_BALLOT(pend);
                const u32 alive = WARP_BALLOT(live);
                if (!found) {
                    if (!alive) break;          // the warp's supply is gone
                    continue;
                }
                // deal: the found classes in lane order, to the lanes from
                // total mod 32 on, as far as there is room
                const u32 nf = (u32)gf2f_popc32(found), room = 32u * LB - total;
                const u32 take = nf < room ? nf : room;
                const u32 r = (lane - total) & 31u;     // the class this lane gets
                const u64 got = WARP_SHFL(pw, warp_select(found, (int)r));
                if (r < take) {
                    if (got >> 63) pairm |= 1u << n;
                    ub[n++] = got & ~((u64)1 << 63);
                }
                if (pend && (u32)gf2f_popc32(found & below) < take) pend = false;
                total += take;
                if (total == 32u * LB) break;   // every lane has LB
            }
            PT(PT_FILL);
            const bool last = total < 32u * LB; // this is the last round
            if (n) {
                if (lane == 0) PT_COUNT(PT_BATCH, 1);
                PT_COUNT(PT_ENTRY, n);
                const bool vf = (nb++ & 7) == 0;
                if constexpr (GROUP) {          // round 8: the grouped order
                    volatile FillPark fp;       // (round 9: parked, see above)
                    fp.rr = rr; fp.kr = kr; fp.w = w; fp.whi = whi; fp.pw = pw;
                    fp.surv = my_surv; fp.pnb = (u64)(u32)p | (u64)nb << 32;
                    fp.flags = (u32)have | (u32)pend << 1 | (u32)live << 2;
                    logs_batch_grouped<D, METHOD>(L, tw, gpow2, ub, vb, xb, n, pairm,
                                                  vf, best, heavy, nheavy, overflow,
                                                  P, &my_err PT_ARG);
                    rr = fp.rr; kr = fp.kr; w = fp.w; whi = fp.whi; pw = fp.pw;
                    my_surv = fp.surv;
                    const u64 pnb = fp.pnb;
                    p = (int)(u32)pnb; nb = (unsigned)(pnb >> 32);
                    const u32 fl = fp.flags;
                    have = fl & 1; pend = (fl >> 1) & 1; live = (fl >> 2) & 1;
                } else {
                    logs_batch<D, METHOD>(L, tw, gpow2, ub, vb, tb, xb, n, pairm,
                                          vf, best, heavy, nheavy, overflow, P,
                                          &my_err PT_ARG);
                }
            }
            if (last) break;                    // that was the last round
        }
    } else {
        // ---- per-lane fill (round 6; FILL_ROT rotates the runs) ----
        for (;;) {
            int n = 0;
            u32 pairm = 0;                      // bit k: ub[k] has a partner
            while (n < LB) {                    // fill: next 32 canonical classes
                if (!have) {
                    if (rr >= nruns) break;
                    const u64 lo = ulo + logs_run_at(rr, kr, nruns, rot) * run;
                    whi = uhi - lo > run ? lo + run : uhi;
                    rr += stride; kr++;
                    w = fkm_seek(lo, D, &p);
                    have = true;
                }
                if (w >= whi) { have = false; continue; }
                if (p == D) {                   // a Lyndon word: a class
                    const int role = L.pair ? gf2f_pair_role(w, D, L.M) : 2;
                    if (role) {
                        if (role == 1) pairm |= 1u << n;
                        my_surv += role == 1 ? 2 : 1;   // classes covered
                        ub[n++] = w;
                    }
                }
                w = fkm_next(w, D, &p);
            }
            PT(PT_FILL);
            if (n == 0) break;
            if ((threadIdx.x & 31u) == 0) PT_COUNT(PT_BATCH, 1);
            PT_COUNT(PT_ENTRY, n);
            const bool vf = (nb++ & 7) == 0;
            if constexpr (GROUP) {              // (round 9: parked, see above)
                volatile FillPark fp;
                fp.rr = rr; fp.kr = kr; fp.w = w; fp.whi = whi;
                fp.surv = my_surv; fp.pnb = (u64)(u32)p | (u64)nb << 32;
                fp.flags = (u32)have;
                logs_batch_grouped<D, METHOD>(L, tw, gpow2, ub, vb, xb, n, pairm,
                                              vf, best, heavy, nheavy, overflow,
                                              P, &my_err PT_ARG);
                rr = fp.rr; kr = fp.kr; w = fp.w; whi = fp.whi;
                my_surv = fp.surv;
                const u64 pnb = fp.pnb;
                p = (int)(u32)pnb; nb = (unsigned)(pnb >> 32);
                have = (fp.flags & 1) != 0;
            } else {
                logs_batch<D, METHOD>(L, tw, gpow2, ub, vb, tb, xb, n, pairm,
                                      vf, best, heavy, nheavy, overflow, P,
                                      &my_err PT_ARG);
            }
        }
    }
    if (my_surv) atomicAdd(nsurv, my_surv);
    if (my_err) atomicAdd(nerr, my_err);
    PT(PT_REST);
}

// selftest: count the representatives in [ulo, uhi) by the necklace test
// (the batched kernel's class count must match)
__global__ void k_rep_count(LogsParams L, u64 ulo, u64 uhi,
                            unsigned long long *count) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    unsigned long long c = 0;
    for (u64 u = (ulo | 1) + 2 * i; u < uhi; u += 2 * stride)
        c += logs_is_rep(u, L);
    if (c) atomicAdd(count, c);
}

// tower: T[log_h(c)] = log(c + w) (and T[MK] = log(w)), from every e = g^u
// with b != 0 in [u0, u0+count) (all writers of one slot write the same
// value, so plain stores suffice)
__global__ void k_tower_T(TowerDev t, u64 M, u64 q, u64 qlow, int d,
                          const fw *gpow2, u64 u0, u64 count, u64 run,
                          u64 *T) {
    const u64 nruns = (count + run - 1) / run;
    u64 rr = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; rr < nruns; rr += stride) {
        u64 u = u0 + rr * run;
        u64 uend = u + run; if (uend > u0 + count) uend = u0 + count;
        u64 e = gf2f_antilog(u, qlow, d, gpow2);
        for (; u < uend; u++, e = gf_mulx(e, q, d)) {
            u64 lb, l;
            if (!tower_split(tower_coords(e, t), t, &lb, &l)) continue;
            u64 h = (t.MK + 2) * lb;
            T[l] = u >= h ? u - h : u + M - h;
        }
    }
}

// Pohlig-Hellman: insert gamma^i -> i for i < N
__global__ void k_ph_build(PHSlot *slots, int sbits, u64 N, u64 gamma,
                           u64 qlow, int d, u64 run) {
    const u64 nruns = (N + run - 1) / run, mask = ((u64)1 << sbits) - 1;
    u64 rr = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; rr < nruns; rr += stride) {
        u64 i = rr * run, iend = i + run; if (iend > N) iend = N;
        u64 y = gf2f_pow(gamma, i, qlow, d);
        for (; i < iend; i++, y = gf2f_mul(y, gamma, qlow, d)) {
            for (u64 h = ph_slot(y, sbits);; h = (h + 1) & mask) {
                ull old = atomicCAS((ull *)&slots[h].key, 0ULL, (ull)y);
                if (old == 0) { slots[h].val = i; break; }
            }
        }
    }
}

// selftest: the fast rotation test against the all-rotations reference, on
// every odd u < min(2^(d-1), 2*count) and on `count` pseudo-random odd
// u < 2^(d-1) whose leading-zero count is spread over 1..d-1
__global__ void k_rep_check(int d, u64 count, unsigned long long *nbad) {
    const u64 M = ((u64)1 << d) - 1, top = (u64)1 << (d - 1);
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    unsigned long long bad = 0;
    for (; i < count; i += stride) {
        u64 u = 2 * i + 1;
        if (u < top && gf2f_is_min_rotation(u, d, M) !=
                       gf2f_min_rotation_ref(u, d, M)) bad++;
        u64 h = (i + 1) * 0x9E3779B97F4A7C15ULL;              // splitmix64
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
        h ^= h >> 31;
        u = ((h >> (65 - d + (int)(i % (u64)(d - 1)))) | 1) & (top - 1);
        if (u && gf2f_is_min_rotation(u, d, M) !=
                 gf2f_min_rotation_ref(u, d, M)) bad++;
    }
    if (bad) atomicAdd(nbad, bad);
}

// selftest: log(g^u) == u for every u in [ulo, uhi)
__global__ void k_dlog_check(LogsParams L, const fw *gpow2, u64 ulo, u64 uhi,
                             u64 run, unsigned long long *nbad) {
    const u64 nruns = (uhi - ulo + run - 1) / run;
    u64 rr = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    unsigned long long bad = 0;
    for (; rr < nruns; rr += stride) {
        u64 u = ulo + rr * run, uend = u + run; if (uend > uhi) uend = uhi;
        u64 e = gf2f_antilog(u, L.qlow, L.d, gpow2);
        for (; u < uend; u++, e = gf_mulx(e, L.q, L.d))
            if (logs_dlog(e, L) != u) bad++;
    }
    if (bad) atomicAdd(nbad, bad);
}
// ---------------------------------------------------------------------------
static void launch_dims(u64 work, int &blocks, int &threads) {
    threads = 256;
    u64 b = (work + threads - 1) / threads;
    blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
}

// ---------------------------------------------------------------------------
// MULTI-GPU
//
// Chunks are independent: each reads only the degree's tables and writes
// best[] with atomicMin.  Since min is associative and commutative, N devices
// can each keep a PRIVATE best[] and the elementwise minimum of those arrays
// is bit-identical to a single-device run -- no inter-device traffic during
// compute, no peer access or NVLink; one best[] readback per device when a
// checkpoint or the final result is needed.  Chunks are dealt out in rounds
// (round k gives chunk start+k*NG+g to device g) and rounds end with every
// device synchronized, so the completed set is always a contiguous prefix of
// chunks -- what the checkpoint records.
struct GpuCtx {
    int dev = 0;
    ull *d_best = nullptr;
    Heavy *d_heavy = nullptr;
    unsigned *d_nheavy = nullptr, *d_overflow = nullptr;
    unsigned long long *d_runctr = nullptr;  // FILL_POOL: the launch's next run
    // per-degree scratch: the g^(2^t) chain, class and error counters
    fw *dgp = nullptr;
    unsigned long long *d_nsurv = nullptr, *d_nerr = nullptr;
    // per-degree tables
    u64 *d_btab = nullptr, *d_T = nullptr;
    u32 *d_LK = nullptr, *d_TP = nullptr;
    uint16_t *d_LKn = nullptr;
    PHSlot *d_slots[4] = {nullptr, nullptr, nullptr, nullptr};
    u64 *d_rrL = nullptr;                // RR: logs of all small polynomials
    u64 *d_cbtab = nullptr, *d_T3 = nullptr;  // round 9: the cubic tower's
                                         // byte tables and T3 (its K-tower: d_btab, d_LK, d_T)
    unsigned long long *d_prof = nullptr;   // -DCOARSE_PHASE_TIMING (round 8)
};
static std::vector<GpuCtx> g_gpus;
static std::vector<ull> g_gather_tmp;

// Merge every device's best[] into `out` by elementwise minimum.
static void gather_best(ull *out, u64 n) {
    CUCHK(cudaSetDevice(g_gpus[0].dev));
    CUCHK(cudaMemcpy(out, g_gpus[0].d_best, n * sizeof(ull),
                     cudaMemcpyDeviceToHost));
    if (g_gpus.size() == 1) return;
    if (g_gather_tmp.size() != n) g_gather_tmp.resize(n);
    for (size_t g = 1; g < g_gpus.size(); g++) {
        CUCHK(cudaSetDevice(g_gpus[g].dev));
        CUCHK(cudaMemcpy(g_gather_tmp.data(), g_gpus[g].d_best,
                         n * sizeof(ull), cudaMemcpyDeviceToHost));
        for (u64 i = 0; i < n; i++)
            if (g_gather_tmp[i] < out[i]) out[i] = g_gather_tmp[i];
    }
}

// engine choice (COARSE_METHOD, --selftest [engine])
enum { FM_AUTO = 0, FM_TOWER = 1, FM_PH = 2, FM_RR = 3, FM_CUBIC = 4 };
static int g_force_method = FM_AUTO;
static bool g_logs_selfcheck = false;    // selftest: verify every log on device
static bool g_tune = true;               // --no-tune / COARSE_TUNE=0
static int g_tune_min = 40;              // COARSE_TUNE_MIN: least degree tuned
static int g_tower_sub = -1;             // COARSE_TOWER_SUBN: -1 auto, 0 classic, n
static bool g_test_eta = false;          // COARSE_TEST_ETA=1: always show the ETA
static int g_pipe = 0;                   // the loop order (COARSE_PIPE): 0 round
                                         // 5's (round 8: the default), 1 round
                                         // 6's, 2 grouped (round 8)
static int g_fill = FILL_POOL;           // COARSE_FILL=0..3 (round 7)
static int g_ldcg = 1;                   // COARSE_LDCG=0: table reads via L1
static int g_pf = 1;                     // COARSE_PREFETCH=0: no L1 prefetches
static bool g_sync_poll = false;         // COARSE_SYNC=poll: poll chunk events
static bool g_pipe_fixed = false;        // COARSE_PIPE set: the tuner keeps it

#define GPU_MEM_RESERVE (256ull << 20)  // headroom for allocator/context overhead

// ---------------------------------------------------------------------------
// CHECKPOINTS (format v4; see CHECKPOINTS in the header comment)
//
// The signal handler only sets a flag; the main thread acts on it at chunk
// boundaries (a chunk takes ~0.1-0.2 s at d >= 44), so a stop is prompt and
// the ~30 s budget of a spot-instance reclaim is easily met.  The snapshot
// is refreshed after every round of chunks: best[] merged from the devices
// plus (degree d, the first `next` chunks of its pass done, the classes they
// covered, and the degree's engine and inner subfield).
#define CKPT_DEFAULT_MINS 15
#define CKPT_KEEP 3                      // generations: <file>, .1, .2

struct CkptCtx {
    bool enabled = false;
    const char *path = nullptr;
    double period_s = 0, last_write = 0;
    // the run: r, smax, the degree range, the seed
    u64 r = 0, smax = 0;
    int d_lo = 0, d_hi = 0;
    std::string seed_path;               // "" = no seed (a run from scratch)
    u64 seed_hash = 0;
    std::vector<ull> base;               // the seed image: the delta baseline
    // the snapshot: best[] after the first `next` chunks of degree d's pass
    std::vector<ull> snap;
    int snap_d = 0, snap_next = 0, snap_nchunks = 0;
    int snap_method = LM_NONE, snap_sub = -1;   // the degree's engine, subfield
    int snap_pipe = 1;                   // and loop order (LogsParams::pipe)
    int snap_tpg = 0, cur_tpg = 0;       // round 8: TP from global memory (the
                                         // snapshot's; the current degree's)
    u64 snap_classes = 0;                // classes covered by those chunks
    double snap_deg_s = 0;               // time on degree snap_d so far
    // round 7: time in all runs, as one uninterrupted run would show it --
    // of the run's degrees before the current one (run_known: every run so
    // far recorded it; false after resuming a round-6 checkpoint)
    double run_before_s = 0, snap_run_s = 0;
    bool run_known = true, snap_run_known = true;
    double eta_s = -1;                   // the latest time-left estimate
    volatile sig_atomic_t stop = 0;      // set by the signal handler
    long test_stop_after = 0, test_chunks_done = 0;   // test hook
};
static CkptCtx g_ckpt;

// the watchdog's progress counter (round 8; see WATCHDOG below)
static std::atomic<unsigned long long> g_wd_progress{0};
static inline void wd_progress() { g_wd_progress.fetch_add(1, std::memory_order_relaxed); }

// The first SIGINT/SIGTERM asks for a stop at the next chunk boundary; the
// second kills at once, as before -- but (round 8) first unlinks the lock
// file (see ckpt_lock), so a run stopped by a double Ctrl-C leaves none
// behind.  unlink, signal and raise are async-signal-safe; the re-raised
// signal is delivered with the default action once the handler returns, so
// the exit status is the usual one of a run killed by that signal.
static char g_lock_cpath[1100];                 // the lock file, once held
extern "C" void ckpt_signal_handler(int sig) {
    if (!g_ckpt.stop) { g_ckpt.stop = 1; return; }
    if (g_lock_cpath[0]) unlink(g_lock_cpath);
    signal(sig, SIG_DFL);
    raise(sig);
}

// layout: magic; u32[CKPT_N32] = {4, d, next, nchunks, method, sub, flags,
// seed path length, d_lo, d_hi, loop order (pipe), run time}; u64[CKPT_N64]
// = {r, smax, period (s), seed hash, ndelta, time written, classes, degree
// time}; the seed path; then ndelta (s, key) pairs (a delta against the
// seed image) or best[1..smax].  The two times (round 7; round 6 wrote 0
// there and ignores them, and its readers ignore the flags it does not
// know): the degree's time so far as an IEEE double (flag CKPT_F_DEGTIME),
// the time of the run's earlier degrees as a float (CKPT_F_RUNTIME).
static const char CKPT_MAGIC[8] = {'G','F','2','S','V','C','K','4'};
#define CKPT_N32 12
#define CKPT_N64 8
#define CKPT_F_DELTA   1u
#define CKPT_F_SEED    2u
#define CKPT_F_DEGTIME 4u
#define CKPT_F_RUNTIME 8u
#define CKPT_F_TPGLOB  16u     // round 8: the degree's tower reads TP from global

struct CkptInfo {
    int d = 0, next = 0, nchunks = 0, method = LM_NONE, sub = -1, pipe = 1;
    int d_lo = 0, d_hi = 0;
    bool delta = false, seed = false;
    bool tpg = false;                           // round 8: CKPT_F_TPGLOB
    bool deg_known = false, run_known = false;  // the times below recorded
    std::string seed_path;
    u64 r = 0, smax = 0, period_s = 0, seed_hash = 0, ndelta = 0, written = 0,
        classes = 0;
    double deg_s = 0, run_s = 0;
};

// the header (and the seed path); *f_out is left at the payload if given
static bool ckpt_peek(const char *path, CkptInfo &ci, FILE **f_out = nullptr) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    u32 h32[CKPT_N32];
    u64 h64[CKPT_N64];
    bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, CKPT_MAGIC, 8) &&
              fread(h32, sizeof(u32), CKPT_N32, f) == CKPT_N32 &&
              fread(h64, sizeof(u64), CKPT_N64, f) == CKPT_N64 && h32[0] == 4;
    if (ok) {
        ci.d = (int)h32[1]; ci.next = (int)h32[2]; ci.nchunks = (int)h32[3];
        ci.method = (int)h32[4]; ci.sub = (int)(int32_t)h32[5];
        ci.delta = (h32[6] & CKPT_F_DELTA) != 0;
        ci.seed = (h32[6] & CKPT_F_SEED) != 0;
        ci.d_lo = (int)h32[8]; ci.d_hi = (int)h32[9]; ci.pipe = (int)h32[10];
        ci.r = h64[0]; ci.smax = h64[1]; ci.period_s = h64[2];
        ci.seed_hash = h64[3]; ci.ndelta = h64[4]; ci.written = h64[5];
        ci.classes = h64[6];
        ci.deg_known = (h32[6] & CKPT_F_DEGTIME) != 0;
        ci.run_known = (h32[6] & CKPT_F_RUNTIME) != 0;
        ci.tpg = (h32[6] & CKPT_F_TPGLOB) != 0;
        float fr;
        memcpy(&fr, &h32[11], sizeof fr);
        memcpy(&ci.deg_s, &h64[7], sizeof ci.deg_s);
        ci.run_s = fr;
        if (!(ci.deg_s >= 0 && ci.deg_s < 1e12)) { ci.deg_s = 0; ci.deg_known = false; }
        if (!(ci.run_s >= 0 && ci.run_s < 1e12)) { ci.run_s = 0; ci.run_known = false; }
        const u32 plen = h32[7];
        ci.seed_path.assign(plen, '\0');
        ok = plen <= 4096 && (!plen || fread(&ci.seed_path[0], 1, plen, f) == plen);
    }
    if (!ok || !f_out) fclose(f);
    else *f_out = f;
    return ok;
}

// cheap 64-bit hash (FNV-1a) tying a checkpoint to the exact seed image
static u64 ckpt_hash(const ull *p, u64 n) {
    u64 h = 1469598103934665603ULL;
    for (u64 i = 0; i < n; i++) { h ^= (u64)p[i]; h *= 1099511628211ULL; }
    return h;
}

// Copy device best[] into the host snapshot and tag it.  nchunks = 0 marks a
// degree boundary ("degree d, nothing of it done yet").  deg_s: the time on
// the degree so far (in all runs; < 0: unknown, after resuming a round-6
// checkpoint mid-degree).
static void ckpt_snapshot(int d, int next, int nchunks, int method, int sub,
                          int pipe, u64 classes, double deg_s) {
    if (!g_ckpt.enabled) return;
    if (g_ckpt.snap.size() != g_ckpt.smax + 1) g_ckpt.snap.resize(g_ckpt.smax + 1);
    gather_best(g_ckpt.snap.data(), g_ckpt.smax + 1);
    g_ckpt.snap_d = d;
    g_ckpt.snap_next = next;
    g_ckpt.snap_nchunks = nchunks;
    g_ckpt.snap_method = method;
    g_ckpt.snap_sub = sub;
    g_ckpt.snap_pipe = pipe;
    g_ckpt.snap_tpg = method == LM_TOWER ? g_ckpt.cur_tpg : 0;
    g_ckpt.snap_classes = classes;
    g_ckpt.snap_deg_s = deg_s;
    g_ckpt.snap_run_s = g_ckpt.run_before_s;
    g_ckpt.snap_run_known = g_ckpt.run_known;
}

static std::string ckpt_gen_name(const char *path, int i) {
    char b[1100];
    if (i == 0) snprintf(b, sizeof b, "%s", path);
    else        snprintf(b, sizeof b, "%s.%d", path, i);
    return b;
}

// <path> -> <path>.1 -> <path>.2 (the oldest dropped), so a checkpoint hit by
// a storage glitch is not the only copy
static void ckpt_rotate() {
    remove(ckpt_gen_name(g_ckpt.path, CKPT_KEEP - 1).c_str());
    for (int i = CKPT_KEEP - 1; i >= 1; i--)
        rename(ckpt_gen_name(g_ckpt.path, i - 1).c_str(),
               ckpt_gen_name(g_ckpt.path, i).c_str());
}

// "~25 min to go" / "~6.3 h to go"
static void eta_text(double s, char *b, size_t n) {
    if (s < 5400) snprintf(b, n, "~%.0f min to go", s / 60.0 < 1 ? 1.0 : s / 60.0);
    else snprintf(b, n, "~%.1f h to go", s / 3600.0);
}

// Write the snapshot atomically (tmp + rename), as a delta against the seed
// image when that is under a quarter of the range, else as a full image.
static void ckpt_write() {
    if (!g_ckpt.enabled || g_ckpt.snap.size() != g_ckpt.smax + 1) return;
    const double t0 = now_s();
    std::vector<u64> dsig;                       // interleaved (s, key) pairs
    bool use_delta = false;
    if (g_ckpt.base.size() == g_ckpt.smax + 1) {
        const u64 cap = g_ckpt.smax / 4 + 1;
        use_delta = true;
        for (u64 s = 1; s <= g_ckpt.smax; s++) {
            if (g_ckpt.snap[s] == g_ckpt.base[s]) continue;
            if ((u64)(dsig.size() / 2) >= cap) { use_delta = false; break; }
            dsig.push_back(s);
            dsig.push_back((u64)g_ckpt.snap[s]);
        }
    }
    const std::string tmp = std::string(g_ckpt.path) + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) { perror(tmp.c_str()); return; }
    const bool seed = !g_ckpt.seed_path.empty();
    const u32 plen = seed ? (u32)g_ckpt.seed_path.size() : 0;
    const u64 ndelta = use_delta ? (u64)(dsig.size() / 2) : 0;
    const bool deg_known = g_ckpt.snap_deg_s >= 0;
    const double deg_s = deg_known ? g_ckpt.snap_deg_s : 0.0;
    const float run_s = g_ckpt.snap_run_known ? (float)g_ckpt.snap_run_s : 0.0f;
    u32 run_bits;
    u64 deg_bits;
    memcpy(&run_bits, &run_s, sizeof run_bits);
    memcpy(&deg_bits, &deg_s, sizeof deg_bits);
    const u32 h32[CKPT_N32] = { 4u, (u32)g_ckpt.snap_d, (u32)g_ckpt.snap_next,
        (u32)g_ckpt.snap_nchunks, (u32)g_ckpt.snap_method,
        (u32)(int32_t)g_ckpt.snap_sub,
        (use_delta ? CKPT_F_DELTA : 0u) | (seed ? CKPT_F_SEED : 0u) |
        (deg_known ? CKPT_F_DEGTIME : 0u) |
        (g_ckpt.snap_run_known ? CKPT_F_RUNTIME : 0u) |
        (g_ckpt.snap_tpg ? CKPT_F_TPGLOB : 0u), plen,
        (u32)g_ckpt.d_lo, (u32)g_ckpt.d_hi, (u32)g_ckpt.snap_pipe, run_bits };
    const u64 h64[CKPT_N64] = { g_ckpt.r, g_ckpt.smax,
        (u64)(g_ckpt.period_s + 0.5), seed ? g_ckpt.seed_hash : 0, ndelta,
        (u64)time(nullptr), g_ckpt.snap_classes, deg_bits };
    bool ok = fwrite(CKPT_MAGIC, 1, 8, f) == 8 &&
              fwrite(h32, sizeof(u32), CKPT_N32, f) == CKPT_N32 &&
              fwrite(h64, sizeof(u64), CKPT_N64, f) == CKPT_N64 &&
              (!plen || fwrite(g_ckpt.seed_path.data(), 1, plen, f) == plen);
    if (ok) {
        if (use_delta)
            ok = dsig.empty() ||
                 fwrite(dsig.data(), sizeof(u64), dsig.size(), f) == dsig.size();
        else
            ok = fwrite(g_ckpt.snap.data() + 1, sizeof(ull), g_ckpt.smax, f) ==
                 g_ckpt.smax;
    }
    if (fflush(f) != 0) ok = false;
    fclose(f);
    if (!ok) { fprintf(stderr, "checkpoint write failed (%s)\n", tmp.c_str()); return; }
    ckpt_rotate();
    if (rename(tmp.c_str(), g_ckpt.path) != 0) { perror("rename checkpoint"); return; }
    g_ckpt.last_write = now_s();
    wd_progress();
    char eta[48] = "";
    if (g_ckpt.eta_s > 0) {
        snprintf(eta, sizeof eta, ", ");
        eta_text(g_ckpt.eta_s, eta + 2, sizeof eta - 2);
    }
    if (use_delta)
        fprintf(stderr, "[checkpoint: d=%d chunk %d/%d, delta %" PRIu64
                " entries (%.2f MB) -> %s, %.2fs%s]\n", g_ckpt.snap_d,
                g_ckpt.snap_next, g_ckpt.snap_nchunks, ndelta,
                (ndelta * 16 + 128) / 1048576.0, g_ckpt.path, now_s() - t0, eta);
    else
        fprintf(stderr, "[checkpoint: d=%d chunk %d/%d, full image (%.0f MB) -> "
                "%s, %.2fs%s]\n", g_ckpt.snap_d, g_ckpt.snap_next,
                g_ckpt.snap_nchunks, g_ckpt.smax * 8 / 1048576.0, g_ckpt.path,
                now_s() - t0, eta);
}

static bool ckpt_periodic_due() {
    return g_ckpt.enabled && g_ckpt.period_s > 0 &&
           (now_s() - g_ckpt.last_write) >= g_ckpt.period_s;
}
static bool ckpt_stop_requested() { return g_ckpt.stop != 0; }
// a stop signal, or the test hook's chunk count reached
static bool ckpt_stop_due() {
    return g_ckpt.stop != 0 || (g_ckpt.test_stop_after &&
                                g_ckpt.test_chunks_done >= g_ckpt.test_stop_after);
}

// persist and exit cleanly after a stop signal (or the test stop)
static void ckpt_stop_and_exit() {
    if (!g_ckpt.enabled) {                  // (only the test hook, e.g. to
        fprintf(stderr, "stopping after %ld chunks (COARSE_TEST_STOP_AFTER_"
                "CHUNKS); no checkpoint, nothing saved\n",   // profile a few)
                g_ckpt.test_chunks_done);
        exit(0);
    }
    ckpt_write();
    fprintf(stderr, "stopping on request; run the same command again to "
            "continue\n");
    exit(0);
}

// A best[] image from a survivors list (one s per line): listed s are open
// (~0), every other s gets the sentinel 0, "has a factor of a lower degree",
// which is all the sieve needs (it only ever compares with atomicMin).
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
        if (v[s] != ~0ULL) nsurv++;
        v[s] = ~0ULL;
    }
    fclose(sf);
    if (nsurv_out) *nsurv_out = nsurv;
    if (nline_out) *nline_out = line;
    if (noor_out)  *noor_out  = oor;
    return true;
}

// Why the checkpoint is NOT this run's ("" if it is).
static std::string ckpt_mismatch(const CkptInfo &ci, u64 r, u64 smax, int d_lo,
                                 int d_hi, bool seed, u64 seed_hash) {
    char b[512];
    if (ci.r != r || ci.smax != smax)
        snprintf(b, sizeof b, "it is for r=%" PRIu64 " (s <= %" PRIu64 "), this "
                 "run is for r=%" PRIu64 " (s <= %" PRIu64 ")", ci.r, ci.smax, r,
                 smax);
    else if (ci.d_lo != d_lo || ci.d_hi != d_hi)
        snprintf(b, sizeof b, "it was written by a run of degrees %d-%d, this run "
                 "is %d-%d", ci.d_lo, ci.d_hi, d_lo, d_hi);
    else if (ci.seed != seed || (seed && ci.seed_hash != seed_hash))
        snprintf(b, sizeof b, "it started from %s%s, this run from %s",
                 ci.seed ? "--load-survivors " : "no seed file",
                 ci.seed ? ci.seed_path.c_str() : "",
                 !seed ? "no seed file" : ci.seed ? "a file with different "
                 "contents" : "a seed file");
    else
        return "";
    return b;
}

static std::string ckpt_describe(const CkptInfo &ci) {
    char b[256], when[64] = "";
    if (ci.written) {
        time_t t = (time_t)ci.written;
        struct tm tmv;
        localtime_r(&t, &tmv);
        strftime(when, sizeof when, "written %Y-%m-%d %H:%M, ", &tmv);
    }
    if (ci.nchunks && ci.deg_known)
        snprintf(b, sizeof b, "%sdegree %d, chunk %d of %d, %.2fs into it", when,
                 ci.d, ci.next, ci.nchunks, ci.deg_s);
    else if (ci.nchunks)
        snprintf(b, sizeof b, "%sdegree %d, chunk %d of %d", when, ci.d, ci.next,
                 ci.nchunks);
    else
        snprintf(b, sizeof b, "%sstart of degree %d", when, ci.d);
    return b;
}

// Load the payload of a checkpoint whose header `ci` (file f, at the
// payload) belongs to this run: a delta is applied to the seed image
// `base`, a full image replaces it.
static bool ckpt_load_payload(FILE *f, const CkptInfo &ci, u64 smax,
                              const std::vector<ull> &base,
                              std::vector<ull> &best) {
    if (ci.delta) {
        if (base.size() != smax + 1) return false;
        best = base;
        for (u64 k = 0; k < ci.ndelta; k++) {
            u64 pair[2];
            if (fread(pair, sizeof(u64), 2, f) != 2) return false;
            if (pair[0] < 1 || pair[0] > smax) return false;
            best[pair[0]] = (ull)pair[1];
        }
        return true;
    }
    best.assign(smax + 1, ~0ULL);
    return fread(best.data() + 1, sizeof(ull), smax, f) == smax;
}

// The newest usable generation of `path` for this run.  1: loaded (ci, best,
// gen); 0: no checkpoint; -1: files exist but none is readable; -2: the
// newest readable one belongs to another run (*why says how).
static int ckpt_resume(const char *path, u64 r, u64 smax, int d_lo, int d_hi,
                       bool seed, u64 seed_hash, const std::vector<ull> &base,
                       CkptInfo &ci, std::vector<ull> &best, std::string &gen,
                       std::string &why) {
    bool any = false, checked = false;
    for (int i = 0; i < 8; i++) {
        const std::string g = ckpt_gen_name(path, i);
        if (access(g.c_str(), F_OK) != 0) { if (i == 0) continue; else break; }
        any = true;
        FILE *f = nullptr;
        CkptInfo c;
        if (!ckpt_peek(g.c_str(), c, &f)) {
            fprintf(stderr, "checkpoint %s: not a readable checkpoint (format "
                    "v4, rounds 6-9)\n", g.c_str());
            continue;
        }
        if (!checked) {                          // the newest readable one
            checked = true;
            why = ckpt_mismatch(c, r, smax, d_lo, d_hi, seed, seed_hash);
            if (!why.empty()) { fclose(f); gen = g; return -2; }
        } else if (!ckpt_mismatch(c, r, smax, d_lo, d_hi, seed, seed_hash).empty()) {
            fclose(f);
            continue;
        }
        const bool ok = ckpt_load_payload(f, c, smax, base, best);
        fclose(f);
        if (ok) {
            if (i) fprintf(stderr, "note: newer checkpoint generations were "
                           "unusable; continuing from %s\n", g.c_str());
            ci = c;
            gen = g;
            return 1;
        }
        fprintf(stderr, "checkpoint %s: truncated or damaged; trying an older "
                "generation\n", g.c_str());
    }
    return any ? -1 : 0;
}

// --fresh: move every generation of `path` to <name>.old
static void ckpt_set_aside(const char *path) {
    int n = 0;
    for (int i = 0; i < 8; i++) {
        const std::string g = ckpt_gen_name(path, i);
        if (access(g.c_str(), F_OK) != 0) continue;
        if (rename(g.c_str(), (g + ".old").c_str()) == 0) n++;
    }
    if (n) fprintf(stderr, "--fresh: set %d earlier checkpoint file%s aside as "
                   "%s*.old\n", n, n == 1 ? "" : "s", path);
}

static void ckpt_remove_all(const char *path) {
    for (int i = 0; i < 8; i++) remove(ckpt_gen_name(path, i).c_str());
}

// One run per checkpoint: an flock'ed <checkpoint>.lock.  The lock, not the
// file, is what counts: the kernel drops it when its process ends, however
// it ends (a second Ctrl-C, kill -9, a crash), so a file left behind is
// harmless -- the next run simply locks it again (round 8 says so).  Deleting
// it by hand is never needed, and is harmful only while the run that holds
// it is still alive (a hung one): a new run would not see that lock.  A
// clean exit unlinks the file (atexit), and round 8 also unlinks it on a
// second SIGINT/SIGTERM and on a watchdog exit.  --fresh leaves it alone: by
// then it is this run's own lock.
static std::string g_lock_path;                 // (and g_lock_cpath, above)
static void ckpt_unlock_atexit() { if (!g_lock_path.empty()) unlink(g_lock_path.c_str()); }
static bool ckpt_lock(const char *path) {
    const std::string lp = std::string(path) + ".lock";
    for (int attempt = 0; attempt < 16; attempt++) {
        bool existed = false;
        int fd = open(lp.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
        if (fd < 0 && errno == EEXIST) {
            existed = true;
            fd = open(lp.c_str(), O_RDWR);
            if (fd < 0 && errno == ENOENT) continue;    // unlinked meanwhile
        }
        if (fd < 0) {
            fprintf(stderr, "note: cannot create %s (no protection against a "
                    "second run on the same checkpoint)\n", lp.c_str());
            return true;
        }
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            fprintf(stderr, "another run is using checkpoint %s (lock %s held); "
                    "not starting a second one\n", path, lp.c_str());
            close(fd);
            return false;
        }
        // the locked file must still be the one at the path: a run that was
        // exiting may have unlinked it between our open and its lock's
        // release, and a third run would then lock a new file there
        struct stat sf, sp;
        if (fstat(fd, &sf) != 0 || stat(lp.c_str(), &sp) != 0 ||
            sf.st_dev != sp.st_dev || sf.st_ino != sp.st_ino) {
            close(fd);
            continue;
        }
        if (existed)
            fprintf(stderr, "note: %s was left by an earlier run that did not "
                    "exit normally (killed or crashed); harmless -- its lock "
                    "ended with it\n", lp.c_str());
        g_lock_path = lp;                        // fd stays open: lock held
        snprintf(g_lock_cpath, sizeof g_lock_cpath, "%s", lp.c_str());
        atexit(ckpt_unlock_atexit);
        return true;
    }
    fprintf(stderr, "note: could not settle the lock %s (it keeps being "
            "replaced); not starting\n", lp.c_str());
    return false;
}

// ---------------------------------------------------------------------------
// WATCHDOG (round 8)
// When the GPU or its driver hangs, the run waits forever inside a CUDA
// call: the GPU idle, the host thread blocked (or spinning) in the driver,
// and a stop request never acted on, since no chunk boundary comes.  A
// thread watches a progress counter, bumped whenever a round of chunks, a
// tuner candidate, a table-building launch or a checkpoint write completes.
// If it stays put for --watchdog-mins (default 30; 0: off) while degrees are
// being sieved, the watchdog says so, removes the lock file and ends the
// process with exit status 3.  The last checkpoint is intact (writes are
// atomic), so a wrapper can restart an unattended run:
//     until ./coarse_sieve ...; do [ $? -eq 3 ] || break; sleep 60; done
// It writes no checkpoint itself: best[] lives on the GPU, and a hung driver
// would not hand it back.  (A chunk takes seconds at most, a table build
// minutes at most, so 30 minutes without progress is not a slow run.)
// Test hooks: COARSE_TEST_WATCHDOG_SECS=s sets the limit in seconds;
// COARSE_TEST_HANG_AFTER_CHUNKS=n makes the main thread hang, ignoring stop
// requests as a hung driver call would, once n chunks are done.
static std::atomic<int> g_wd_armed{0};
#define WD_DEFAULT_MINS 30
static double g_wd_limit_s = WD_DEFAULT_MINS * 60.0;
static long g_test_hang_after = 0;

static void wd_fire(double idle_s) {
    char b[1400];
    const bool mins = idle_s >= 120;
    int n = snprintf(b, sizeof b, "\nwatchdog: no progress for %.0f %s -- the "
                     "GPU or its driver seems to hang; exiting with status 3.  ",
                     mins ? idle_s / 60 : idle_s, mins ? "minutes" : "seconds");
    if (n > 0 && n < (int)sizeof b) {
        if (g_ckpt.enabled && g_ckpt.path)
            n += snprintf(b + n, sizeof b - n, "The last checkpoint (%s) is "
                          "intact: the same command continues from it.\n",
                          g_ckpt.path);
        else
            n += snprintf(b + n, sizeof b - n, "This run had no checkpoint, "
                          "so its work is lost.\n");
    }
    if (n > 0) {
        const size_t len = n < (int)sizeof b ? (size_t)n : sizeof b - 1;
        ssize_t w = write(2, b, len);
        (void)w;
    }
    if (g_lock_cpath[0]) unlink(g_lock_cpath);
    _exit(3);
}

static void *wd_main(void *) {
    unsigned long long last = g_wd_progress.load();
    double since = now_s();
    const useconds_t step = g_wd_limit_s < 120 ? 100000 : 1000000;
    for (;;) {
        usleep(step);
        const unsigned long long p = g_wd_progress.load();
        const double t = now_s();
        if (p != last || !g_wd_armed.load()) {
            last = p;
            since = t;
            continue;
        }
        if (t - since >= g_wd_limit_s) wd_fire(t - since);
    }
    return nullptr;
}

static void wd_start() {
    if (g_wd_limit_s <= 0) return;
    pthread_t t;
    if (pthread_create(&t, nullptr, wd_main, nullptr) == 0) pthread_detach(t);
    else fprintf(stderr, "note: could not start the watchdog thread\n");
}

// the test hook's hang: from here on the run makes no progress and ignores
// stop requests, as if stuck in the driver (a second signal still kills)
static void wd_test_hang_check() {
    if (g_test_hang_after && g_ckpt.test_chunks_done >= g_test_hang_after) {
        fprintf(stderr, "\n[COARSE_TEST_HANG_AFTER_CHUNKS: hanging after %ld "
                "chunks]\n", g_ckpt.test_chunks_done);
        for (;;) pause();
    }
}

// ---------------------------------------------------------------------------
// THE LOGS PATH: host-side driver

#define LOGS_RUN_MAX        (1u << 14)   // u values per thread-run
#define LOGS_RUNS_PER_CHUNK (1u << 17)   // runs per kernel launch (chunk)
// Class representatives are concentrated at small u: a representative
// starts with its longest zero run (typically ~log2(d) bits), so most lie
// in [2^(d-8), 2^(d-4)) and nearly every odd u below ~2^(d-12) is one.  The
// first 1/64 of the chunks (u < 2^(d-7): a zero run of >= 7) hold ~15-18% of
// the classes at d = 42..50 -- P(longest run >= 7) ~ 1 - exp(-d/256) -- and
// more of the canonical ones.  Taken in order, the early chunks would be
// several times slower than the late ones (round 5's progress dots, and its
// time-left estimates, 7-10x too high); so the pass takes them in the
// scrambled order of chunk_at.

// chunk geometry depends only on d (and the test hook), so a checkpoint's
// position in the pass is stable; nchunks is a power of two
static void logs_geometry(int d, u64 &total, u64 &run, u64 &chunk,
                          int &nchunks) {
    total = (u64)1 << (d - 1);                  // class reps are < 2^(d-1)
    run = LOGS_RUN_MAX;
    while (run > 64 && total / run < ((u64)1 << 17)) run >>= 1;
    chunk = run * (u64)LOGS_RUNS_PER_CHUNK;
    // test hook: COARSE_LOGS_CHUNK_LOG2=k caps chunks at 2^k u values, so
    // small-degree runs exercise multi-chunk and checkpoint/resume logic
    // (a resume must use the same setting; the chunk count is checked)
    if (const char *e = getenv("COARSE_LOGS_CHUNK_LOG2")) {
        int k = atoi(e);
        if (k >= 1 && k < 63 && ((u64)1 << k) < chunk) chunk = (u64)1 << k;
    }
    if (chunk > total) chunk = total;
    nchunks = (int)((total + chunk - 1) / chunk);
}

// The order of the pass (round 6): position p -> chunk index.  Blocks of
// CHUNK_BLOCK consecutive chunks (the NG <= CHUNK_BLOCK chunks of a round
// stay neighbours of similar density, so the GPUs finish them together),
// the blocks in a scrambled bit-reversed order (Owen's nested scrambling of
// the van der Corput sequence): the first 2^j blocks of the pass hold one
// block of each 1/2^j of the range, at a pseudo-random place within it.
// So every stage of the pass is a fair, stratified sample of the range: the
// progress dots come at a steady pace, and the time left can be scaled from
// the classes done so far.  Plain bit reversal would not do: its first 2^j
// blocks are the multiples of 2^(k-j), whose u have k-j zero bits just
// above the block's own -- an internal zero run longer than most leading
// ones, so those blocks hold almost no classes.  (Any order gives the same
// result: best[] is merged by atomicMin.)
#define CHUNK_BLOCK 8
static u64 chunk_at(int p, int nchunks) {
    if (nchunks <= CHUNK_BLOCK || nchunks % CHUNK_BLOCK) return (u64)p;
    const u64 nb = (u64)(nchunks / CHUNK_BLOCK);
    if (nb & (nb - 1)) return (u64)p;           // (never: a power of two)
    int k = 0;
    while (((u64)1 << k) < nb) k++;
    const u64 b = (u64)(p / CHUNK_BLOCK);
    u64 x = 0;                                  // bit reversal of b
    for (int i = 0; i < k; i++) x |= ((b >> i) & 1) << (k - 1 - i);
    // flip bit i by a hash of the bits above it: a bijection, and blocks in
    // different dyadic intervals stay in different intervals
    u64 y = x;
    for (int i = k - 1; i >= 0; i--) {
        u64 h = ((x >> (i + 1)) + 1) * 0x9E3779B97F4A7C15ULL + (u64)i * 0xD1B54A32D192ED03ULL;
        h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 29;
        y ^= (h >> 63) << i;
    }
    return y * CHUNK_BLOCK + (u64)(p % CHUNK_BLOCK);
}

struct LogsPlan {
    int method = LM_NONE;
    PHHost ph;                                  // PH parameters (if ph_ok)
    size_t bytes = 0;                           // device table bytes
    // odd d (round 5): both engines are planned, the tuner may pick either
    bool ph_ok = false, rr_ok = false;
    PHHost ph_fb;                               // RR: PH plan of the factor-
    size_t rr_bytes = 0;                        //   base logs; the RR table
    int tp_global = 0;                          // round 8: the subfield tower's
                                                //   TP read from global memory
    // round 9: the cubic tower (6 | d), a tuner candidate or COARSE_METHOD=
    // cubic; its T3 is built by PH (plan ph_t3)
    bool cubic_ok = false;
    size_t cubic_bytes = 0;
    PHHost ph_t3;
};

// The batched kernel for (d, method), or nullptr: tower at even d >= 4, PH
// at odd d (and d=2), RR at odd d >= 5, the cubic tower (round 9) at
// 6 | d >= CUBIC_MIN_D.  One instantiation per degree, since the bit-sliced
// code needs d at compile time.  group (round 8): the kernel of the grouped
// loop order (L.pipe = 2), tower, cubic and RR at d >= GROUPED_MIN_D -- the
// tuned degrees on the GPU (each such kernel adds compile time), every
// degree in the emulated build (the tests).
typedef void (*LogsBatchFn)(LogsParams, const fw *, u64, u64, u64, ull *,
                            Heavy *, unsigned *, unsigned *,
                            unsigned long long *, unsigned long long *);
#if defined(__CUDACC__)
#define GROUPED_MIN_D 40
#define CUBIC_MIN_D 24          // round 9: the cubic tower's kernels, 6 | d
#else
#define GROUPED_MIN_D 5
#define CUBIC_MIN_D 12
#endif
// the cubic tower's kernel at d (6 | d, d >= CUBIC_MIN_D), else nullptr
template <int D> static LogsBatchFn cubic_batch_fn(bool group) {
    if constexpr (D % 6 == 0 && D >= CUBIC_MIN_D)
        return !group ? k_logs_batch<D, LM_CUBIC, 0>
             : D >= GROUPED_MIN_D ? k_logs_batch<D, LM_CUBIC, (D >= GROUPED_MIN_D)>
                                  : nullptr;
    else
        return nullptr;
}
static LogsBatchFn logs_batch_fn(int d, int method, bool group = false) {
#define LBF_G(D, MT) (!group ? k_logs_batch<D, MT, 0>                           \
                     : D >= GROUPED_MIN_D ? k_logs_batch<D, MT, (D >= GROUPED_MIN_D)> \
                                          : nullptr)
#define LBF_PH(D) case D: return method == LM_PH && !group ? k_logs_batch<D, LM_PH> : nullptr;
#define LBF_PR(D) case D: return method == LM_PH ? (group ? nullptr : k_logs_batch<D, LM_PH>) \
                               : method == LM_RR ? LBF_G(D, LM_RR) : nullptr;
#define LBF_TW(D) case D: return method == LM_TOWER ? LBF_G(D, LM_TOWER)          \
                               : method == LM_CUBIC ? cubic_batch_fn<D>(group) : nullptr;
    switch (d) {
    LBF_PH(2)  LBF_PH(3)  LBF_TW(4)  LBF_PR(5)  LBF_TW(6)  LBF_PR(7)
    LBF_TW(8)  LBF_PR(9)  LBF_TW(10) LBF_PR(11) LBF_TW(12) LBF_PR(13)
    LBF_TW(14) LBF_PR(15) LBF_TW(16) LBF_PR(17) LBF_TW(18) LBF_PR(19)
    LBF_TW(20) LBF_PR(21) LBF_TW(22) LBF_PR(23) LBF_TW(24) LBF_PR(25)
    LBF_TW(26) LBF_PR(27) LBF_TW(28) LBF_PR(29) LBF_TW(30) LBF_PR(31)
    LBF_TW(32) LBF_PR(33) LBF_TW(34) LBF_PR(35) LBF_TW(36) LBF_PR(37)
    LBF_TW(38) LBF_PR(39) LBF_TW(40) LBF_PR(41) LBF_TW(42) LBF_PR(43)
    LBF_TW(44) LBF_PR(45) LBF_TW(46) LBF_PR(47) LBF_TW(48) LBF_PR(49)
    LBF_TW(50) LBF_PR(51) LBF_TW(52) LBF_PR(53) LBF_TW(54) LBF_PR(55)
    default: break;
    }
#undef LBF_PH
#undef LBF_PR
#undef LBF_TW
#undef LBF_G
    return nullptr;
}

// VRAM the logs tables may use: free memory minus the usual reserve and some
// slack (min(free/8, 512 MB)): a table filling the card would only save a few
// giant steps over one half its size, and a near-full card risks a failed
// allocation (e.g. when the desktop grabs memory).  COARSE_LOGS_MAX_MB caps
// it further.
static size_t logs_table_budget(size_t freeb) {
    size_t slack = freeb / 8 < (512ull << 20) ? freeb / 8 : (512ull << 20);
    size_t b = freeb > GPU_MEM_RESERVE + slack ? freeb - GPU_MEM_RESERVE - slack : 0;
    if (const char *e = getenv("COARSE_LOGS_MAX_MB")) {
        size_t cap = (size_t)strtoull(e, nullptr, 10) << 20;
        if (cap && cap < b) b = cap;
    }
    return b;
}

// Pick the engine for degree d whose tables fit in `budget` bytes.
// want: FM_AUTO -- tower at even d, PH at odd d (RR where PH has no plan:
// d = 49); FM_TOWER -- tower at even d (or nothing); FM_PH -- PH at odd d
// (or nothing); FM_RR -- RR at odd d >= 5 (or nothing); each is FM_AUTO at
// the other degrees.  At odd d both PH and RR are planned when possible (the
// tuner may switch between them).
static bool logs_plan(int d, u64 q, int want, size_t budget, LogsPlan &pl) {
    pl = LogsPlan();
    if (!gf2f_usable(q, d)) return false;
    if ((q ^ ((u64)1 << d)) != gf2_qlow(d)) {   // the kernels' compiled modulus
        fprintf(stderr, "d=%d: primitive polynomial differs from the compiled "
                "table -- internal error\n", d);
        exit(1);
    }
    if (want == FM_TOWER && (d & 1)) want = FM_AUTO;
    if (want == FM_PH && !(d & 1) && d >= 4) want = FM_AUTO;
    if (want == FM_RR && (!(d & 1) || d < 5)) want = FM_AUTO;
    if (want == FM_CUBIC && !logs_batch_fn(d, LM_CUBIC)) want = FM_AUTO;
    const bool tower_ok = d >= 4 && !(d & 1) && d <= 54;
    if (tower_ok) {
        // T (8 B per K-element) and the byte tables, plus LK (4 B per
        // K-element) for the classic tower or LKn + TP for the subfield one
        u64 tp = 0;
        const int n = tower_sub_choice(d / 2, g_tower_sub, &tp);
        size_t need = ((size_t)8 << (d / 2)) + ((size_t)((d + 7) / 8) << 11) + 65536;
        need += n ? ((size_t)2 << n) + 4 * (size_t)tp : (size_t)4 << (d / 2);
        // round 9: the cubic tower at 6 | d: T3 (8 B per element of
        // GF(2^(d/3))), the byte tables, its K-tower; while T3 is built, PH
        // tables (freed before the sieve)
        if (logs_batch_fn(d, LM_CUBIC)) {
            const size_t cb = ((size_t)8 << (d / 3)) + ((size_t)((d + 7) / 8) << 11) + 65536;
            const size_t phb = budget > cb ? std::min(budget - cb, (size_t)64 << 20) : 0;
            if (cb <= budget && ph_plan_host(d, q, phb, pl.ph_t3, kPHCostScalar, 64, false)) {
                pl.cubic_ok = true;
                pl.cubic_bytes = cb;
            }
        }
        if (want == FM_CUBIC) {
            if (!pl.cubic_ok) return false;
            pl.method = LM_CUBIC;
            pl.bytes = pl.cubic_bytes;
            return true;
        }
        if (need <= budget) {
            pl.method = LM_TOWER;
            pl.bytes = need;
            return true;
        }
        if (pl.cubic_ok && want == FM_AUTO) {   // (little VRAM: d=54's T alone
            pl.method = LM_CUBIC;               //  is 1 GB, its T3 2 MB)
            pl.bytes = pl.cubic_bytes;
            return true;
        }
        return false;
    }
    pl.ph_ok = ph_plan_host(d, q, budget, pl.ph, kPHCostBitsliced);
    // RR (round 5): the table of 2^(dh+1) logs, and a PH plan for the
    // factor-base logs -- the sieve's own if it has one, else (d=49) one with
    // up to 2^22 giant steps per log.  Test hook: COARSE_RR_FB_BUDGET_MB=m
    // always plans the factor-base logs separately with tables of at most
    // m MB (many giant steps: d=49's path at small d).
    if (want != FM_PH && logs_batch_fn(d, LM_RR)) {
        const int dh = d / 2;
        const size_t tb = (size_t)8 << (dh + 1);
        const char *e = getenv("COARSE_RR_FB_BUDGET_MB");
        if (tb <= budget) {
            pl.rr_bytes = tb;
            if (pl.ph_ok && !e) {
                pl.ph_fb = pl.ph;
                pl.rr_ok = true;
            } else {
                // leave room for the factor-base arrays (~2^(dh+1)/dh
                // elements, 12 bytes each)
                const size_t fbb = (size_t)(1.2 * (double)((u64)1 << (dh + 1)) / dh) * 16;
                size_t b = budget > fbb ? budget - fbb : 0;
                if (e && ((size_t)strtoull(e, nullptr, 10) << 20) < b)
                    b = (size_t)strtoull(e, nullptr, 10) << 20;
                pl.rr_ok = ph_plan_host(d, q, b, pl.ph_fb, kPHCostScalar,
                                        (u64)1 << 22);
            }
        }
    }
    if (want == FM_RR) {
        if (!pl.rr_ok) return false;
        pl.method = LM_RR;
    } else if (pl.ph_ok) {
        pl.method = LM_PH;
    } else if (pl.rr_ok && want != FM_PH) {
        pl.method = LM_RR;                      // d = 49: the only engine
    } else {
        return false;
    }
    pl.bytes = pl.method == LM_RR ? pl.rr_bytes : pl.ph.table_bytes;
    return true;
}

static LogsParams logs_params_base(int d, u64 r, u64 smax, u64 q, int method) {
    LogsParams L;
    memset(&L, 0, sizeof L);
    L.d = d;
    L.method = method;
    L.M = ((u64)1 << d) - 1;
    L.q = q;
    L.qlow = q ^ ((u64)1 << d);
    L.rmodinv = inv_mod(r % L.M, L.M);
    L.rmod = r % L.M;
    L.pair = 1;
    L.pipe = g_pipe;
    L.fill = g_fill;
    L.ldcg = g_ldcg;
    L.prefetch = g_pf;
    L.smax = smax;
    // Carmichael lambda(M) = lcm of phi(p^k) over p^k || M: the exponent of
    // the unit group, so j^(lambda-1) = j^-1 for gcd(j, M) = 1.  It is much
    // shorter than phi(M) (31 vs 47 bits at d=47, 16 vs 41 at d=42), which
    // cuts the per-class modular inverse by a third or more.
    u64 rest = L.M, lam = 1;
    for (u64 p : factor_u64(L.M)) {
        int k = 0;
        u64 pk = 1;
        while (rest % p == 0) { rest /= p; k++; pk *= p; }
        L.pf[L.npf] = odd_div_make(p);
        L.pk[L.npf] = k;
        L.npf++;
        const u64 ph = pk / p * (p - 1);
        lam = lam / gcd_u64(lam, ph) * ph;
    }
    L.lam_m1 = lam - 1;
    if (L.npf != gf2_npf(d)) {                  // the kernels' loop bounds
        fprintf(stderr, "d=%d: %d primes in 2^d-1, table says %d -- internal "
                "error\n", d, L.npf, gf2_npf(d));
        exit(1);
    }
    for (u64 p : factor_u64((u64)d)) {
        int e = d / (int)p;
        L.sub[L.nsub++] = odd_div_make(L.M / (((u64)1 << e) - 1));
    }
    return L;
}

static u64 logs_run_for(u64 work) {             // ~2^16 threads per launch
    u64 run = 64;
    while (run < LOGS_RUN_MAX && work / run > ((u64)1 << 16)) run <<= 1;
    return run;
}

// Build this device's tables and fill L's device pointers.  The tower's T
// is computed on the first device and copied to the others via `Th`.
// Round 9: the cubic tower of the current degree, host side -- the
// structure (cubic_build_host) and T3 (cubic_build_T3, on GPU 0) -- built
// once per degree for the tuner, the pass and the devices
static int g_cubic_d = 0;
static CubicHost g_cubic;
static std::vector<u64> g_cubic_T3;

static void logs_setup_device(GpuCtx &G, int d, u64 q, const LogsPlan &pl,
                              const TowerHost &W, std::vector<u64> &Th,
                              LogsParams &L,
                              const std::vector<u64> *rrL = nullptr) {
    const u64 M = L.M;
    int blocks, threads;
    if (pl.method == LM_CUBIC) {                // round 9: tables from the host
        const CubicHost &C = g_cubic;
        if (g_cubic_d != d || g_cubic_T3.size() != C.MK + 2) {
            fprintf(stderr, "d=%d: cubic tower tables missing -- internal error\n", d);
            exit(1);
        }
        auto up = [](u64 **dp, const std::vector<u64> &h) {
            CUCHK(cudaMalloc(dp, h.size() * sizeof(u64)));
            CUCHK(cudaMemcpy(*dp, h.data(), h.size() * sizeof(u64), cudaMemcpyHostToDevice));
        };
        up(&G.d_cbtab, C.btab);
        up(&G.d_T3, g_cubic_T3);
        up(&G.d_btab, C.kt.btab);
        up(&G.d_T, C.kT);
        CUCHK(cudaMalloc(&G.d_LK, C.kt.LK.size() * sizeof(u32)));
        CUCHK(cudaMemcpy(G.d_LK, C.kt.LK.data(), C.kt.LK.size() * sizeof(u32),
                         cudaMemcpyHostToDevice));
        L.tw = cubic_dev(C, G.d_cbtab, G.d_T3, G.d_btab, G.d_LK, G.d_T);
        int optin = 48 << 10;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, G.dev);
        const u32 shm = cubic_shm_bytes(L.tw);
        L.tw.cb.shm_bytes = shm <= (u32)optin ? shm : 0;
        return;
    }
    if (pl.method == LM_RR) {                   // the table built by the host
        if (!rrL || rrL->size() != ((size_t)2 << (d / 2))) {
            fprintf(stderr, "d=%d: RR table missing -- internal error\n", d);
            exit(1);
        }
        CUCHK(cudaMalloc(&G.d_rrL, rrL->size() * sizeof(u64)));
        CUCHK(cudaMemcpy(G.d_rrL, rrL->data(), rrL->size() * sizeof(u64),
                         cudaMemcpyHostToDevice));
        L.rr = RRDev{G.d_rrL, d / 2};
        return;
    }
    if (pl.method == LM_TOWER) {
        CUCHK(cudaMalloc(&G.d_btab, W.btab.size() * sizeof(u64)));
        CUCHK(cudaMemcpy(G.d_btab, W.btab.data(), W.btab.size() * sizeof(u64),
                         cudaMemcpyHostToDevice));
        if (W.sub_k) {                          // subfield tower: LKn, TP
            CUCHK(cudaMalloc(&G.d_LKn, W.LKn.size() * sizeof(uint16_t)));
            CUCHK(cudaMemcpy(G.d_LKn, W.LKn.data(),
                             W.LKn.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice));
            CUCHK(cudaMalloc(&G.d_TP, W.TP.size() * sizeof(u32)));
            CUCHK(cudaMemcpy(G.d_TP, W.TP.data(), W.TP.size() * sizeof(u32),
                             cudaMemcpyHostToDevice));
        } else {                                // classic: LK
            CUCHK(cudaMalloc(&G.d_LK, W.LK.size() * sizeof(u32)));
            CUCHK(cudaMemcpy(G.d_LK, W.LK.data(), W.LK.size() * sizeof(u32),
                             cudaMemcpyHostToDevice));
        }
        const u64 slots = W.MK + 1;
        CUCHK(cudaMalloc(&G.d_T, slots * sizeof(u64)));
        L.tw = tower_dev(W, G.d_btab, G.d_LK, G.d_T, G.d_LKn, G.d_TP);
        // the batched kernel stages the small tables in shared memory: the
        // byte tables, and for the subfield tower LKn and, if it fits, TP
        {
            int optin = 48 << 10;
            cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   G.dev);
            size_t shm = W.btab.size() * sizeof(u64);
            if (W.sub_k) {
                shm += W.LKn.size() * sizeof(uint16_t);
                const size_t tpb = W.TP.size() * sizeof(u32);
                if (!pl.tp_global && shm + tpb <= (size_t)optin) {
                    shm += tpb;
                    L.tw.shm_tp = 1;
                }
            }
            L.tw.shm_bytes = shm <= (size_t)optin ? (u32)shm : 0;
        }
        if (Th.empty()) {                       // coupon-collector sweeps
            launch_dims(slots, blocks, threads);
            k_fill<<<blocks, threads>>>((ull *)G.d_T, slots, ~0ULL);
            Th.assign(slots, ~0ULL);
            u64 u0 = 0, missing = slots;
            while (missing && u0 < M) {
                u64 cnt = slots * 8;
                if (cnt > M - u0) cnt = M - u0;
                u64 run = logs_run_for(cnt);
                launch_dims((cnt + run - 1) / run, blocks, threads);
                k_tower_T<<<blocks, threads>>>(L.tw, M, q, L.qlow, d, G.dgp,
                                               u0, cnt, run, G.d_T);
                CUCHK(cudaDeviceSynchronize());
                wd_progress();
                CUCHK(cudaMemcpy(Th.data(), G.d_T, slots * sizeof(u64),
                                 cudaMemcpyDeviceToHost));
                missing = 0;
                for (u64 c = 0; c < slots; c++) missing += (Th[c] >= M);
                u0 += cnt;
            }
            if (missing) {
                fprintf(stderr, "d=%d: tower table T incomplete (%" PRIu64
                        " missing) -- internal error\n", d, missing);
                exit(1);
            }
        } else {
            CUCHK(cudaMemcpy(G.d_T, Th.data(), slots * sizeof(u64),
                             cudaMemcpyHostToDevice));
        }
    } else {
        L.ph.nparts = pl.ph.nparts;
        L.ph.chain = pl.ph.chain;
        for (int i = 0; i < pl.ph.nparts; i++) {
            const PHPartHost &p = pl.ph.part[i];
            const size_t S = (size_t)1 << p.sbits;
            CUCHK(cudaMalloc(&G.d_slots[i], S * sizeof(PHSlot)));
            CUCHK(cudaMemset(G.d_slots[i], 0, S * sizeof(PHSlot)));
            u64 run = logs_run_for(p.N);
            launch_dims((p.N + run - 1) / run, blocks, threads);
            k_ph_build<<<blocks, threads>>>(G.d_slots[i], p.sbits, p.N, p.gamma,
                                            L.qlow, d, run);
            L.ph.part[i] = PHPart{p.P, p.E, p.K, p.N, p.G, p.delta,
                                  G.d_slots[i], p.sbits};
        }
        CUCHK(cudaDeviceSynchronize());
    }
}

static void logs_free_device(GpuCtx &G) {
    cudaFree(G.d_btab); cudaFree(G.d_LK); cudaFree(G.d_T);
    cudaFree(G.d_LKn); cudaFree(G.d_TP); cudaFree(G.d_rrL);
    cudaFree(G.d_cbtab); cudaFree(G.d_T3);
    G.d_btab = G.d_T = nullptr; G.d_LK = G.d_TP = nullptr; G.d_LKn = nullptr;
    G.d_rrL = nullptr; G.d_cbtab = G.d_T3 = nullptr;
    for (int i = 0; i < 4; i++) {
        cudaFree(G.d_slots[i]);
        G.d_slots[i] = nullptr;
    }
}

// launch shape for the batched kernel: ONE full wave of resident blocks
// (from the occupancy calculator, so it follows whatever register count the
// compiler gave this instantiation), each thread grid-striding over its
// runs and carrying partial batches from run to run; fewer blocks if there
// are fewer runs.  A grid of several waves would leave its last wave partly
// idle: e.g. 3.2 waves of PH blocks take as long as 4.  PH: 128-thread
// blocks (register-heavy); tower: 512-thread blocks, one per SM, sharing
// the tables staged in `shm` bytes of shared memory.
static void logs_batch_dims(LogsBatchFn kfn, u64 nruns, int method,
                            const LogsParams &L, int &blocks, int &threads,
                            size_t &shm) {
    threads = method == LM_PH ? 128 : 512;
    shm = method == LM_TOWER ? L.tw.shm_bytes : method == LM_CUBIC ? L.tw.cb.shm_bytes : 0;
    if (shm > ((size_t)48 << 10))
        CUCHK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shm));
    int dev = 0, nsm = 0, per_sm = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) !=
            cudaSuccess || nsm < 1)
        nsm = 1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kfn, threads, shm)
            != cudaSuccess || per_sm < 1)
        per_sm = 1;
    const u64 wave = (u64)nsm * (u64)per_sm;
    const u64 need = (nruns + threads - 1) / threads;
    const u64 b = need < wave ? need : wave;
    blocks = (int)(b ? b : 1);
}

// ---------------------------------------------------------------------------
// RR table (round 5; the method is described in gf2_dlog.h): the logs of the
// irreducible polynomials of degree <= dh = d/2 by Pohlig-Hellman on GPU 0
// (plan pl.ph_fb), then all 2^(dh+1) logs by multiplicativity on the host.
// For d=49 the PH plan needs up to ~8,000-33,000 giant steps per log
// (2^49-1 = 127 * 4432676798593; the count depends on how large a baby-step
// table fits), ~2.4e10 for the 1.47M factor-base logs on an 8 GB card --
// random reads of a 4 GB table, so a minute or two by my estimate.  The
// table does not depend on r; a build that took over two minutes is cached
// in coarse_rr_d<d>.tbl (COARSE_RR_DIR=dir puts the files elsewhere).  Every factor-base log is verified (g^log == f) on the
// device, a sample of the table on the host.  Launches are kept near
// 0.1 s (a display GPU's watchdog) and a stop request is honored between
// them.

// one log per factor-base element, with work stealing: a lane that is done
// takes the next element, so logs of very different cost (0..G giant
// steps) keep every lane busy
__global__ void k_rr_fblogs(LogsParams L, const fw *gpow2, const u32 *fb,
                            u64 *out, u64 lo, u64 hi,
                            unsigned long long *next,
                            unsigned long long *nerr) {
    unsigned long long bad = 0;
    for (;;) {
        const u64 i = lo + atomicAdd(next, 1ULL);
        if (i >= hi) break;
        const u64 f = fb[i];
        const u64 v = ph_log(f, L.M, L.d, L.qlow, L.ph);
        if (v >= L.M || gf2f_antilog(v, L.qlow, L.d, gpow2) != f) { bad++; out[i] = 0; }
        else out[i] = v;
    }
    if (bad) atomicAdd(nerr, bad);
}

static bool g_rr_nocache = false;        // selftests: always build afresh

static std::string rr_cache_path(int d) {
    const char *dir = getenv("COARSE_RR_DIR");
    char b[64];
    snprintf(b, sizeof b, "coarse_rr_d%d.tbl", d);
    return std::string(dir ? dir : ".") + "/" + b;
}

static const char RR_MAGIC[8] = {'G','F','2','R','R','T','B','1'};

// a sample of the table, checked on the host: g^L[P] == P (bit-serial
// arithmetic, independent of the device code)
static bool rr_table_spotcheck(int d, u64 q, const std::vector<u64> &L,
                               int samples) {
    u64 z = 0x2545F4914F6CDD1DULL ^ (u64)d;
    for (int i = 0; i < samples; i++) {
        z ^= z << 13; z ^= z >> 7; z ^= z << 17;
        const u64 P = 1 + z % (L.size() - 1);
        if (gf_pow(2, L[P], q, d) != P) return false;
    }
    return true;
}

static bool rr_cache_load(int d, u64 q, std::vector<u64> &L) {
    if (g_rr_nocache) return false;
    const std::string path = rr_cache_path(d);
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    u64 h[5];
    const u64 n = (u64)2 << (d / 2);
    bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, RR_MAGIC, 8) &&
              fread(h, sizeof(u64), 5, f) == 5 && h[0] == (u64)d && h[1] == q &&
              h[2] == n;
    if (ok) {
        L.resize(n);
        ok = fread(L.data(), sizeof(u64), n, f) == n &&
             ckpt_hash((const ull *)L.data(), n) == h[3] &&
             rr_table_spotcheck(d, q, L, 4096);
    }
    fclose(f);
    if (!ok) {
        fprintf(stderr, "  [%s: not a valid RR table for d=%d; rebuilding]\n",
                path.c_str(), d);
        L.clear();
        return false;
    }
    fprintf(stderr, "  [RR table for d=%d loaded from %s]\n", d, path.c_str());
    return true;
}

static void rr_cache_save(int d, u64 q, const std::vector<u64> &L) {
    const std::string path = rr_cache_path(d), tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) { perror(tmp.c_str()); return; }
    u64 h[5] = {(u64)d, q, (u64)L.size(), ckpt_hash((const ull *)L.data(), L.size()), 0};
    bool ok = fwrite(RR_MAGIC, 1, 8, f) == 8 && fwrite(h, sizeof(u64), 5, f) == 5 &&
              fwrite(L.data(), sizeof(u64), L.size(), f) == L.size();
    if (fflush(f) != 0) ok = false;
    fclose(f);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "  [could not save the RR table to %s]\n", path.c_str());
        remove(tmp.c_str());
        return;
    }
    fprintf(stderr, "  [RR table saved to %s (%.0f MB); later runs at d=%d load "
            "it instead of rebuilding]\n", path.c_str(),
            L.size() * 8.0 / 1048576.0, d);
}

static bool rr_build_table(int d, u64 q, u64 r, const LogsPlan &pl,
                           std::vector<u64> &Lh) {
    const int dh = d / 2;
    const u64 M = ((u64)1 << d) - 1;
    if (rr_cache_load(d, q, Lh)) return true;
    const double t0 = now_s();
    std::vector<u32> spf, irr;
    rr_sieve(dh, spf, irr);
    const double t_sieve = now_s() - t0;
    GpuCtx &G = g_gpus[0];
    CUCHK(cudaSetDevice(G.dev));
    // the PH tables of the factor-base plan, and the field's antilog seeds
    LogsPlan pf = pl;
    pf.method = LM_PH;
    pf.ph = pl.ph_fb;
    LogsParams Lp = logs_params_base(d, r, 1, q, LM_PH);
    TowerHost W;
    std::vector<u64> Th;
    logs_setup_device(G, d, q, pf, W, Th, Lp);
    fw gpow2[64];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);
    fw *dgp;
    u32 *dfb;
    u64 *dout;
    unsigned long long *dnext, *dnerr, zero = 0, nerr = 0;
    const u64 n = irr.size();
    CUCHK(cudaMalloc(&dgp, d * sizeof(fw)));
    CUCHK(cudaMemcpy(dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc(&dfb, n * sizeof(u32)));
    CUCHK(cudaMemcpy(dfb, irr.data(), n * sizeof(u32), cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc(&dout, n * sizeof(u64)));
    CUCHK(cudaMalloc(&dnext, 8));
    CUCHK(cudaMalloc(&dnerr, 8));
    CUCHK(cudaMemcpy(dnerr, &zero, 8, cudaMemcpyHostToDevice));
    u64 maxg = 1;
    for (int i = 0; i < pf.ph.nparts; i++) maxg = std::max<u64>(maxg, pf.ph.part[i].G);
    fprintf(stderr, "  [RR table for d=%d: %" PRIu64 " irreducibles of degree "
            "<= %d (sieve %.1fs); their logs by PH, up to %" PRIu64
            " giant step%s each]\n", d, n, dh, t_sieve, maxg, maxg == 1 ? "" : "s");
    int nsm = 1, per = 1, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_rr_fblogs, 128, 0)
            != cudaSuccess || per < 1) per = 1;
    const int blocks = nsm * per, threads = 128;
    const double t1 = now_s();
    double last_note = t1;
    u64 lo = 0, slice = (u64)blocks * threads;   // grown to ~0.1 s per launch
    while (lo < n) {
        const u64 hi = std::min(n, lo + slice);
        CUCHK(cudaMemcpy(dnext, &zero, 8, cudaMemcpyHostToDevice));
        const double ts = now_s();
        k_rr_fblogs<<<blocks, threads>>>(Lp, dgp, dfb, dout, lo, hi, dnext, dnerr);
        CUCHK(cudaDeviceSynchronize());
        CUCHK(cudaGetLastError());
        wd_progress();
        const double dt = now_s() - ts;
        lo = hi;
        if (dt < 0.05 && slice < ((u64)1 << 30)) slice *= 2;
        else if (dt > 0.2 && slice > (u64)blocks * threads) slice /= 2;
        if (now_s() - last_note > 30 && lo < n) {
            const double el = now_s() - t1;
            fprintf(stderr, "  [RR table for d=%d: %.1f%% of the factor-base "
                    "logs, %.0fs, ~%.0fs to go]\n", d, 100.0 * lo / n, el,
                    el * (n - lo) / lo);
            last_note = now_s();
        }
        if (ckpt_stop_requested()) {
            fprintf(stderr, "  [RR table for d=%d: stopped while building]\n", d);
            ckpt_stop_and_exit();
        }
    }
    const double t_fb = now_s() - t1;
    std::vector<u64> fbl(n);
    CUCHK(cudaMemcpy(fbl.data(), dout, n * sizeof(u64), cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(&nerr, dnerr, 8, cudaMemcpyDeviceToHost));
    cudaFree(dgp); cudaFree(dfb); cudaFree(dout); cudaFree(dnext); cudaFree(dnerr);
    logs_free_device(G);
    if (nerr) {
        fprintf(stderr, "d=%d: %llu factor-base logs failed verification -- "
                "internal or hardware error\n", d, nerr);
        return false;
    }
    const double t2 = now_s();
    rr_fill(dh, M, spf, irr, fbl, Lh);
    if (!rr_table_spotcheck(d, q, Lh, 4096)) {
        fprintf(stderr, "d=%d: RR table failed its host spot check -- internal "
                "error\n", d);
        Lh.clear();
        return false;
    }
    fprintf(stderr, "  [RR table for d=%d: %.0f MB, factor-base logs %.1fs, fill "
            "%.1fs, verified]\n", d, Lh.size() * 8.0 / 1048576.0, t_fb,
            now_s() - t2);
    if (now_s() - t0 > 120 && !g_rr_nocache) rr_cache_save(d, q, Lh);
    return true;
}

// ---------------------------------------------------------------------------
// The cubic tower's T3 (round 9): T3[l] = log(w + y'^l) for l < 2^n - 1,
// T3[2^n - 1] = log w and T3[2^n] = 0 (see cubic_parts), the 2^n - 1 logs
// by Pohlig-Hellman on GPU 0 (plan pl.ph_t3; 2^18 logs at d=54, well under
// a second by my estimate), each verified on the device (g^log == w + y'^l)
// and a sample on the host.
__global__ void k_cubic_T3(LogsParams L, const fw *gpow2, u64 w, u64 yp,
                           u64 *out, u64 lo, u64 hi, unsigned long long *nerr) {
    unsigned long long bad = 0;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 l = lo + (u64)blockIdx.x * blockDim.x + threadIdx.x; l < hi; l += stride) {
        const u64 e = w ^ gf2f_pow(yp, l, L.qlow, L.d);
        const u64 v = ph_log(e, L.M, L.d, L.qlow, L.ph);
        if (v >= L.M || gf2f_antilog(v, L.qlow, L.d, gpow2) != e) { bad++; out[l] = 0; }
        else out[l] = v;
    }
    if (bad) atomicAdd(nerr, bad);
}

static bool cubic_build_T3(int d, u64 q, u64 r, const LogsPlan &pl,
                           const CubicHost &C, std::vector<u64> &T3) {
    const double t0 = now_s();
    GpuCtx &G = g_gpus[0];
    CUCHK(cudaSetDevice(G.dev));
    LogsPlan pf = pl;
    pf.method = LM_PH;
    pf.ph = pl.ph_t3;
    LogsParams Lp = logs_params_base(d, r, 1, q, LM_PH);
    TowerHost W;
    std::vector<u64> Th;
    logs_setup_device(G, d, q, pf, W, Th, Lp);
    fw gpow2[64];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);
    fw *dgp;
    u64 *dout;
    unsigned long long *dnerr, zero = 0, nerr = 0;
    const u64 n = C.MK;                          // entries by PH; T3[MK] = log w
    CUCHK(cudaMalloc(&dgp, d * sizeof(fw)));
    CUCHK(cudaMemcpy(dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc(&dout, (n + 1) * sizeof(u64)));
    CUCHK(cudaMalloc(&dnerr, 8));
    CUCHK(cudaMemcpy(dnerr, &zero, 8, cudaMemcpyHostToDevice));
    int nsm = 1, per = 1, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_cubic_T3, 128, 0)
            != cudaSuccess || per < 1) per = 1;
    const int blocks = nsm * per, threads = 128;
    u64 lo = 0, slice = (u64)blocks * threads;   // grown to ~0.1 s per launch
    while (lo < n) {
        const u64 hi = std::min(n, lo + slice);
        const double ts = now_s();
        k_cubic_T3<<<blocks, threads>>>(Lp, dgp, C.w, C.yp, dout, lo, hi, dnerr);
        CUCHK(cudaDeviceSynchronize());
        CUCHK(cudaGetLastError());
        wd_progress();
        const double dt = now_s() - ts;
        lo = hi;
        if (dt < 0.05 && slice < ((u64)1 << 30)) slice *= 2;
        else if (dt > 0.2 && slice > (u64)blocks * threads) slice /= 2;
        if (ckpt_stop_requested()) {
            fprintf(stderr, "  [cubic tower for d=%d: stopped while building T3]\n", d);
            ckpt_stop_and_exit();
        }
    }
    T3.assign(n + 2, 0);                         // T3[MK + 1] = 0: a factor in K
    CUCHK(cudaMemcpy(T3.data(), dout, n * sizeof(u64), cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(&nerr, dnerr, 8, cudaMemcpyDeviceToHost));
    cudaFree(dgp); cudaFree(dout); cudaFree(dnerr);
    logs_free_device(G);
    T3[n] = C.logw;
    if (nerr) {
        fprintf(stderr, "d=%d: %llu T3 logs failed verification -- internal or "
                "hardware error\n", d, nerr);
        return false;
    }
    // a host sample, independent of the device code (bit-serial arithmetic)
    u64 z = 0x2545F4914F6CDD1DULL ^ (u64)d;
    for (int i = 0; i < 256; i++) {
        z ^= z << 13; z ^= z >> 7; z ^= z << 17;
        const u64 l = z % (n + 1);
        const u64 e = l == n ? C.w : C.w ^ gf_pow(C.yp, l, q, d);
        if (gf_pow(2, T3[l], q, d) != e) {
            fprintf(stderr, "d=%d: T3 failed its host spot check -- internal "
                    "error\n", d);
            return false;
        }
    }
    fprintf(stderr, "  [cubic tower for d=%d over GF(2^%d): T3 of %" PRIu64 " logs "
            "(%.0f MB) by PH in %.1fs, verified]\n", d, C.n, n + 1,
            (n + 2) * 8.0 / 1048576.0, now_s() - t0);
    return true;
}

// the cubic tower of degree d in g_cubic / g_cubic_T3 (once per degree)
static bool cubic_prepare(int d, u64 q, u64 r, const LogsPlan &pl) {
    if (g_cubic_d == d && g_cubic_T3.size() == g_cubic.MK + 2) return true;
    g_cubic_d = 0;
    g_cubic_T3.clear();
    if (!pl.cubic_ok || !cubic_build_host(d, q, g_cubic)) {
        fprintf(stderr, "d=%d: cubic tower construction failed -- internal error\n", d);
        return false;
    }
    if (!cubic_build_T3(d, q, r, pl, g_cubic, g_cubic_T3)) return false;
    g_cubic_d = d;
    return true;
}

static void cubic_release() {
    g_cubic_d = 0;
    g_cubic = CubicHost();
    g_cubic_T3.clear();
    g_cubic_T3.shrink_to_fit();
}

// selftest: check log(g^u) == u for u in [0, min(M, 2^24)) on device 0
static void logs_selfcheck(int d, const LogsParams &L) {
    GpuCtx &G = g_gpus[0];
    CUCHK(cudaSetDevice(G.dev));
    u64 hi = L.M < ((u64)1 << 24) ? L.M : ((u64)1 << 24);
    unsigned long long zero = 0, bad = 0;
    unsigned long long *dbad;
    CUCHK(cudaMalloc(&dbad, 8));
    CUCHK(cudaMemcpy(dbad, &zero, 8, cudaMemcpyHostToDevice));
    u64 run = logs_run_for(hi);
    int blocks, threads;
    launch_dims((hi + run - 1) / run, blocks, threads);
    k_dlog_check<<<blocks, threads>>>(L, G.dgp, 0, hi, run, dbad);
    CUCHK(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost));
    cudaFree(dbad);
    fprintf(stderr, "  [d=%d %s logs: %" PRIu64 " checked, %llu wrong]\n", d,
            L.method == LM_TOWER ? "tower" : L.method == LM_RR ? "RR"
            : L.method == LM_CUBIC ? "cubic" : "PH",
            hi, bad);
    if (bad) { fprintf(stderr, "logs selfcheck FAILED at d=%d\n", d); exit(1); }
}

// selftest: the batched kernel in check mode over [ulo, uhi): every class
// it enumerates must get log(g^u) == u, and it must enumerate exactly the
// representatives that the necklace test finds there.  Returns failures.
static u64 logs_batch_check(GpuCtx &G, int d, const LogsPlan &pl,
                            LogsParams L, u64 ulo, u64 uhi, u64 *nclasses) {
    LogsBatchFn kfn = logs_batch_fn(d, pl.method, L.pipe == 2);
    if (!kfn) return 0;
    unsigned long long zero = 0, ns = 0, ne = 0, nc = 0;
    unsigned long long *dns, *dne, *dnc, *dctr;
    CUCHK(cudaMalloc(&dns, 8)); CUCHK(cudaMalloc(&dne, 8)); CUCHK(cudaMalloc(&dnc, 8));
    CUCHK(cudaMalloc(&dctr, 8));
    CUCHK(cudaMemcpy(dns, &zero, 8, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dne, &zero, 8, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dnc, &zero, 8, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dctr, &zero, 8, cudaMemcpyHostToDevice));
    L.check = 1;
    L.pair = 0;                                 // every class of the stretch
    L.runctr = dctr;
    const u64 run = 1024;
    int blocks, threads;
    size_t shm;
    logs_batch_dims(kfn, (uhi - ulo + run - 1) / run, pl.method, L, blocks,
                    threads, shm);
    kfn<<<blocks, threads, shm>>>(L, G.dgp, ulo, uhi, run, nullptr, nullptr,
                                  nullptr, nullptr, dns, dne);
    launch_dims((uhi - ulo) / 2, blocks, threads);
    k_rep_count<<<blocks, threads>>>(L, ulo, uhi, dnc);
    CUCHK(cudaMemcpy(&ns, dns, 8, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(&ne, dne, 8, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(&nc, dnc, 8, cudaMemcpyDeviceToHost));
    cudaFree(dns); cudaFree(dne); cudaFree(dnc); cudaFree(dctr);
    *nclasses = ns;
    return ne + (ns != nc ? 1 : 0);
}

// A stretch [H 2^L, (H+1) 2^L) of u that is dense in class representatives:
// H = 0^z 1 0 1 0 1 ... (d - L bits, z = min(L/2, d-L-1)) puts the longest
// zero run first, so most odd u there are their own least rotation (about
// half of the 2^L values are classes).  (A stretch at 2^(d-12) is not: its
// u have an internal zero run of d - 12 - L bits, longer than the leading
// one once d > 2L + 12, and then hold no representatives at all.)
static u64 logs_dense_stretch(int d, int L) {
    const int hb = d - L;
    const int z = L / 2 < hb - 1 ? L / 2 : hb - 1;
    u64 H = 0;
    for (int b = hb - z - 1; b >= 0; b -= 2) H |= (u64)1 << b;
    return H << L;
}

// A stretch of an ordinary density (round 7, the fill test): H = 0^z 1 and
// pseudo-random bits without a run of z zeros, z = 5 (a common leading run
// at d >= 30, smaller when H is short), so that the stretch holds classes
// -- fewer than the dense one, and with a canonical share that varies
// along it as in a real chunk.  (The dense stretch when H is too short.)
static u64 logs_typical_stretch(int d, int L, u64 seed) {
    const int hb = d - L;
    int z = 5;
    while (z > 1 && hb < z + 2) z--;
    if (hb < z + 2) return logs_dense_stretch(d, L);
    for (int tries = 0; tries < 1000; tries++) {
        u64 h = 0;
        for (int b = hb - z - 2; b >= 0; b--) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            h |= (seed >> 63) << b;
        }
        const u64 H = ((u64)1 << (hb - z - 1)) | h;     // 0^z 1 h
        bool ok = true;                 // no run of >= z zeros in "1 h"
        for (int b = 0; b + z <= hb - z && ok; b++)
            if (((H >> b) & (((u64)1 << z) - 1)) == 0) ok = false;
        if (ok) return H << L;
    }
    return logs_dense_stretch(d, L);
}

// --selftest, second part: the rotation test, the irregular-class lift, and
// at every degree d = 21..55 on device 0 (the sieve comparison only reaches
// d=20) the engines, the batched kernel and the reciprocal pairing.  want =
// FM_TOWER / FM_PH / FM_RR limits the engine checks to that engine.
// Returns the number of failures.
static u64 logs_wide_selftest(u64 r, int want) {
    GpuCtx &G = g_gpus[0];
    CUCHK(cudaSetDevice(G.dev));
#if defined(__CUDACC__)
    const u64 nrep = (u64)1 << 22, nlog = (u64)1 << 20, nbat = (u64)1 << 22,
              nirr = (u64)1 << 20, npair = (u64)1 << 16;
    int dmax_log = KEY_MAX_DEGREE;
    int dmax_rr = 47;                         // 49: its many-giant-step table
                                              // build (minutes), on request
    int d_rr_bsgs = 41;                       // RR with d=49's kind of plan
#else                                         // CPU emulation: keep it short
    const u64 nrep = (u64)1 << 12, nlog = (u64)1 << 9, nbat = (u64)1 << 12,
              nirr = (u64)1 << 13, npair = (u64)1 << 9;
    int dmax_log = 30;
    int dmax_rr = 30;
    int d_rr_bsgs = 27;
#endif
    if (const char *e = getenv("COARSE_SELFTEST_LOGS_DMAX")) {   // test hook
        int k = atoi(e);
        if (k >= 21 && k <= KEY_MAX_DEGREE) dmax_log = k;
    }
    if (const char *e = getenv("COARSE_SELFTEST_RR_DMAX")) dmax_rr = atoi(e);
    if (const char *e = getenv("COARSE_SELFTEST_RR_BSGS_D")) d_rr_bsgs = atoi(e);
    unsigned long long zero = 0, bad = 0, *dbad;
    CUCHK(cudaMalloc(&dbad, 8));
    u64 fails = 0;
    int blocks, threads;
    g_rr_nocache = true;                      // test the builder, not a file
    for (int d = 2; d <= KEY_MAX_DEGREE; d++) {
        CUCHK(cudaMemcpy(dbad, &zero, 8, cudaMemcpyHostToDevice));
        launch_dims(nrep, blocks, threads);
        k_rep_check<<<blocks, threads>>>(d, nrep, dbad);
        CUCHK(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost));
        if (bad) fprintf(stderr, "  d=%d: rotation test: %llu mismatches\n", d, bad);
        fails += bad;
    }
    fprintf(stderr, "  [rotation test vs all-rotations reference, d=2..%d, "
            "%" PRIu64 " u per degree: %s]\n", KEY_MAX_DEGREE, 2 * nrep,
            fails ? "FAILED" : "ok");

    {   // irregular classes: the round-5 batch lift against the definition
        unsigned long long ib[4], *dib;
        CUCHK(cudaMalloc(&dib, sizeof ib));
        u64 ifail = 0, nj = 0, ns = 0, nt = 0;
        for (int d = 4; d <= KEY_MAX_DEGREE; d++) {
            const u64 M = ((u64)1 << d) - 1;
            if (factor_u64(M).size() == 1 && factor_u64(M)[0] == M) continue;
            LogsParams L = logs_params_base(d, r, 1, 0, LM_PH);
            CUCHK(cudaMemset(dib, 0, sizeof ib));
            launch_dims(nirr, blocks, threads);
            k_irreg_check<<<blocks, threads>>>(L, 0x5EED0000ULL + (u64)d, nirr, dib);
            CUCHK(cudaMemcpy(ib, dib, sizeof ib, cudaMemcpyDeviceToHost));
            if (ib[0]) fprintf(stderr, "  d=%d: irregular lift: %llu failures\n",
                               d, ib[0]);
            if (ib[1] < nirr / 8) {             // the check must test something
                fprintf(stderr, "  d=%d: irregular lift: only %llu classes "
                        "joined\n", d, ib[1]);
                ifail++;
            }
            ifail += ib[0]; nj += ib[1]; ns += ib[2]; nt += ib[3];
        }
        cudaFree(dib);
        fprintf(stderr, "  [irregular classes, batch lift vs j s0 == v (mod M), "
                "d=4..%d: %" PRIu64 " joined (%" PRIu64 " with t > 0), %" PRIu64
                " without s: %s]\n", KEY_MAX_DEGREE, nj, nt, ns,
                ifail ? "FAILED" : "ok");
        fails += ifail;
    }

    size_t freeb = 0, totb = 0;
    CUCHK(cudaMemGetInfo(&freeb, &totb));
    const size_t budget = logs_table_budget(freeb);
    unsigned long long pb[3], *dpb;
    CUCHK(cudaMalloc(&dpb, sizeof pb));
    for (int d = 21; d <= dmax_log; d++) {
        const u64 q = find_primitive_poly(d), M = ((u64)1 << d) - 1;
        fw gpow2[64];
        gpow2[0] = 2;
        for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);
        // RR (round 5) at the odd degrees, and once with the factor-base
        // logs forced through a many-giant-step plan (d=49's case)
        const bool rr_here = (d & 1) && d <= dmax_rr;
        const bool rr_bsgs = d == d_rr_bsgs && (d & 1);
        for (int eng : {(int)FM_TOWER, (int)FM_CUBIC, (int)FM_PH, (int)FM_RR, FM_RR + 100}) {
            // (the cubic tower, round 9, with the towers: "tower" or "cubic")
            if (want != FM_AUTO && want != eng % 100 &&
                !(eng == FM_CUBIC && want == FM_TOWER)) continue;
            if (eng == FM_CUBIC && !logs_batch_fn(d, LM_CUBIC)) continue;   // 6 | d
            if (eng == FM_TOWER && (d & 1)) continue;          // even d only
            if (eng != FM_TOWER && eng != FM_CUBIC && !(d & 1)) continue;   // odd d only
            if (eng == FM_RR && !rr_here) continue;
            if (eng == FM_RR + 100 && !rr_bsgs) continue;
            if (eng == FM_RR + 100) setenv("COARSE_RR_FB_BUDGET_MB", "4", 1);
            LogsPlan pl;
            const bool ok_plan = logs_plan(d, q, eng % 100, budget, pl);
            if (eng == FM_RR + 100) unsetenv("COARSE_RR_FB_BUDGET_MB");
            if (!ok_plan) {
                if (eng == FM_PH)
                    fprintf(stderr, "  [d=%d PH    logs: no feasible plan "
                            "(expected only for d=49)]\n", d);
                else
                    fprintf(stderr, "  [d=%d %s logs: no feasible plan]\n", d,
                            eng == FM_TOWER ? "tower" : eng == FM_CUBIC ? "cubic" : "RR   ");
                if (!(eng == FM_PH && d == 49)) fails++;
                continue;
            }
            double t0 = now_s();
            CUCHK(cudaMalloc(&G.dgp, d * sizeof(fw)));
            CUCHK(cudaMemcpy(G.dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
            LogsParams L = logs_params_base(d, r, 1, q, pl.method);
            TowerHost W;
            if (pl.method == LM_TOWER && !tower_build_host(d, q, W, g_tower_sub)) {
                fprintf(stderr, "  d=%d: tower construction failed\n", d);
                fails++;
                cudaFree(G.dgp); G.dgp = nullptr;
                continue;
            }
            if (pl.method == LM_CUBIC && !cubic_prepare(d, q, r, pl)) {
                fails++;
                cudaFree(G.dgp); G.dgp = nullptr;
                continue;
            }
            std::vector<u64> Th, rrL;
            u64 rr_giant = 0;
            if (pl.method == LM_RR) {
                for (int i = 0; i < pl.ph_fb.nparts; i++)
                    rr_giant = std::max<u64>(rr_giant, pl.ph_fb.part[i].G);
                if (!rr_build_table(d, q, r, pl, rrL)) {
                    fprintf(stderr, "  d=%d: RR table construction failed\n", d);
                    fails++;
                    cudaFree(G.dgp); G.dgp = nullptr;
                    continue;
                }
                if (eng == FM_RR + 100 && rr_giant < 2) {
                    fprintf(stderr, "  d=%d: the forced RR factor-base plan has "
                            "no giant steps\n", d);
                    fails++;
                }
            }
            logs_setup_device(G, d, q, pl, W, Th, L, &rrL);
            // nlog consecutive exponents from a degree-dependent offset
            const u64 ulo = (0x9E3779B97F4A7C15ULL * (u64)d) % (M - nlog);
            CUCHK(cudaMemcpy(dbad, &zero, 8, cudaMemcpyHostToDevice));
            const u64 run = logs_run_for(nlog);
            launch_dims((nlog + run - 1) / run, blocks, threads);
            k_dlog_check<<<blocks, threads>>>(L, G.dgp, ulo, ulo + nlog, run, dbad);
            CUCHK(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost));
            // the batched kernel on a stretch dense in representatives
            u64 bfail = 0, ncls = 0;
            int lb = 0;
            while (((u64)2 << lb) <= nbat && lb + 1 <= d - 8) lb++;
            const u64 blo = logs_dense_stretch(d, lb);
            bfail = logs_batch_check(G, d, pl, L, blo, blo + ((u64)1 << lb), &ncls);
            int norders = 1;
            if (pl.method != LM_PH) {           // and in the other loop orders
                LogsParams L2 = L;
                L2.pipe = L.pipe == 1 ? 0 : 1;      // round 5's <-> round 9's
                u64 ncls2 = 0;
                bfail += logs_batch_check(G, d, pl, L2, blo, blo + ((u64)1 << lb),
                                          &ncls2);
                if (ncls2 != ncls) bfail++;
                norders++;
                if (logs_batch_fn(d, pl.method, true)) {    // round 8: grouped
                    L2.pipe = 2;
                    ncls2 = 0;
                    bfail += logs_batch_check(G, d, pl, L2, blo,
                                              blo + ((u64)1 << lb), &ncls2);
                    if (ncls2 != ncls) bfail++;
                    norders++;
                }
            }
            if (ncls < ((u64)1 << lb) / 8) {        // the check must test something
                fprintf(stderr, "  d=%d: batched check found only %" PRIu64
                        " classes in its stretch\n", d, ncls);
                bfail++;
            }
            // round 7: every way of filling the batches, on a stretch of an
            // ordinary density (sparser, the canonical share varying)
            u64 ntyp = 0;
            {
                const u64 tlo = logs_typical_stretch(d, lb, 0x51ED2700ULL + (u64)d);
                for (int f = FILL_LANE; f <= FILL_POOL; f++) {
                    LogsParams Lf = L;
                    Lf.fill = f;
                    u64 nf = 0;
                    bfail += logs_batch_check(G, d, pl, Lf, tlo, tlo + ((u64)1 << lb), &nf);
                    if (f == FILL_LANE) ntyp = nf;
                    else if (nf != ntyp) bfail++;
                }
                if (!ntyp) {
                    fprintf(stderr, "  d=%d: no classes in the fill test's stretch\n", d);
                    bfail++;
                }
            }
            // the reciprocal pairing against direct computation
            CUCHK(cudaMemset(dpb, 0, sizeof pb));
            launch_dims(npair, blocks, threads);
            k_pair_check<<<blocks, threads>>>(L, G.dgp, 0xC0FFEE00ULL + (u64)d,
                                              npair, dpb);
            CUCHK(cudaMemcpy(pb, dpb, sizeof pb, cudaMemcpyDeviceToHost));
            const u64 pfail = pb[0] + (pb[1] < npair / 8 ? 1 : 0);
            logs_free_device(G);
            cubic_release();
            cudaFree(G.dgp); G.dgp = nullptr;
            fprintf(stderr, "  [d=%d %-5s logs, %4.0f MB tables: %" PRIu64
                    " checked, %llu wrong; batched: %" PRIu64 " classes%s and %"
                    PRIu64 " (4 fills), %s; pairs: %llu, %s", d,
                    pl.method == LM_TOWER ? "tower" : pl.method == LM_RR ? "RR"
                    : pl.method == LM_CUBIC ? "cubic" : "PH",
                    pl.bytes / 1048576.0, nlog, bad, ncls,
                    norders == 3 ? " (3 loop orders)" : norders == 2 ? " (both loop orders)" : "",
                    ntyp,
                    bfail ? "FAILED" : "ok", pb[1], pfail ? "FAILED" : "ok");
            if (pl.method == LM_TOWER && W.sub_k)
                fprintf(stderr, "; over GF(2^%d)", W.sub_n);
            else if (pl.method == LM_TOWER)
                fprintf(stderr, "; classic");
            else if (pl.method == LM_CUBIC)
                fprintf(stderr, "; over GF(2^%d)", d / 3);
            else if (pl.method == LM_RR)
                fprintf(stderr, "; factor-base logs with up to %" PRIu64 " giant "
                        "step%s%s", rr_giant, rr_giant == 1 ? "" : "s",
                        eng == FM_RR + 100 ? " (forced, d=49's path)" : "");
            fprintf(stderr, ", %.2fs]\n", now_s() - t0);
            fails += bad + bfail + pfail;
        }
    }
    cudaFree(dpb);
    cudaFree(dbad);
    return fails;
}

// ---------------------------------------------------------------------------
// Auto-tuning (round 5; the sample fixed in round 6).  Which variant of the
// batched kernel is fastest depends on the GPU -- its L2 size (an RTX 4090's
// 72 MB holds the whole tower table T up to d=46, a 2070 SUPER's 4 MB holds
// none of it), shared memory, integer throughput -- so for every degree
// d >= g_tune_min the candidates run the same sample of classes on GPU 0
// first, and the fastest sieves the degree:
//   tower: the subfield towers over the two largest proper divisors n of m
//          (or the forced one), and the classic tower;
//   odd d: PH and RR (unless COARSE_METHOD fixes the engine).
// The sample is a stretch dense in classes (logs_dense_stretch: u with a
// long zero run in front, about half of them classes) of 2^27 u values,
// ~2^26 classes, doubled while it has fewer than 2^25.  (Round 5 started at
// 1/8 of the range, where at d >= 40 the first 2^(d-8) values hold no class
// at all -- their words have an internal zero run longer than the leading
// one -- and doubled into the dense part: samples of 9.2e9 classes at d=44,
// 4.7e11 at d=50, an hour of tuning.)  All candidates compute identical
// results; the sample's marks are simply repeated by the real pass
// (atomicMin) and its heavy progressions dropped (the real pass queues them
// again).  The report line gives ns per class (partners included) for each.
// --no-tune (COARSE_TUNE=0) disables it; COARSE_TOWER_SUBN fixes the
// subfield; test hook: COARSE_TUNE_PICK=i takes candidate i.
struct TuneCand { int method, sub_n, pipe; double ns; int tpg; };

static void tune_name(const TuneCand &c, char *b, size_t n) {
    char base[32];
    if (c.method == LM_TOWER && c.sub_n) snprintf(base, sizeof base, "GF(2^%d)", c.sub_n);
    else if (c.method == LM_TOWER) snprintf(base, sizeof base, "classic");
    else if (c.method == LM_CUBIC) snprintf(base, sizeof base, "cubic GF(2^%d)", c.sub_n);
    else snprintf(base, sizeof base, "%s", c.method == LM_RR ? "RR" : "PH");
    snprintf(b, n, "%s%s%s", base,
             c.method == LM_PH ? "" : c.pipe == 0 ? " r5-order" : c.pipe == 2 ? " grouped" : " r9-order",
             c.tpg ? " TP-global" : "");
}

// Round 8: the subfield tower stages its small tables -- btab, LKn and TP --
// in shared memory.  At d = 48 and 52 TP takes them past 32 KB, which on
// Turing forces the larger shared-memory carveout (64 KB) and halves L1
// (64 -> 32 KB), where the batch arrays' local memory is cached.  So such a
// candidate is also timed with TP read from global memory instead ("TP-
// global": cached in L1 and L2, 32 KB at d=52), unless COARSE_TOWER_TP fixes
// the choice (global or shared).
static int g_tower_tp = -1;              // COARSE_TOWER_TP: -1 auto, 0 shared, 1 global
static bool g_test_tp_choice = false;    // test hook COARSE_TEST_TP_CHOICE=1:
                                         // offer TP-global for every subfield
static bool tower_tp_choice(int d, int n, u64 tp) {
    if (!n) return false;
    if (g_test_tp_choice) return true;
    const size_t base = ((size_t)((d + 7) / 8) << 11) + ((size_t)2 << n);
    const size_t with = base + 4 * (size_t)tp;
    int optin = 48 << 10;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           g_gpus[0].dev);
    return base <= ((size_t)32 << 10) && with > ((size_t)32 << 10) &&
           with <= (size_t)optin;
}

// pl.method may change (odd d: PH <-> RR, unless COARSE_METHOD fixes it);
// sub_use receives the tower's subfield (0: classic), pipe_use the loop
// order, tp_use the TP placement (round 8); rrL the RR table if it was built
static void logs_tune(int d, u64 q, u64 r, u64 smax, LogsPlan &pl,
                      const fw *gpow2, std::vector<u64> &Th,
                      std::vector<u64> &rrL, int &sub_use, int &pipe_use,
                      int &tp_use) {
    GpuCtx &G = g_gpus[0];
    CUCHK(cudaSetDevice(G.dev));
    std::vector<TuneCand> cand;                 // the first is the default
    if (pl.method == LM_CUBIC) {                // forced (or the tower did not fit)
        cand.push_back({LM_CUBIC, d / 3, g_pipe, 0, 0});
    } else if (pl.method == LM_TOWER) {
        u64 tp;
        const int sub_auto = tower_sub_choice(d / 2, g_tower_sub, &tp);
        cand.push_back({LM_TOWER, sub_auto, g_pipe, 0, g_tower_tp > 0});
        if (g_tower_sub < 0) {                  // the variants, if they fit
            size_t freeb = 0, totb = 0;
            CUCHK(cudaMemGetInfo(&freeb, &totb));
            const size_t budget = logs_table_budget(freeb);
            const int m = d / 2;
            int second = 0;                     // the next smaller divisor
            for (int c = sub_auto - 1; c >= 2 && !second; c--) if (m % c == 0) second = c;
            std::vector<int> alts;
            if (second) alts.push_back(second);
            if (sub_auto) alts.push_back(0);    // classic, unless the default
            for (int n : alts) {
                u64 tpn = 0;
                if (n && tower_sub_choice(m, n, &tpn) != n) continue;
                size_t need = ((size_t)8 << m) + ((size_t)((d + 7) / 8) << 11) + 65536;
                need += n ? ((size_t)2 << n) + 4 * (size_t)tpn : (size_t)4 << m;
                if (need <= budget) cand.push_back({LM_TOWER, n, g_pipe, 0, g_tower_tp > 0});
            }
        }
        // round 9: the cubic tower at 6 | d (unless the engine or the
        // tower's subfield is fixed)
        if (pl.cubic_ok && g_force_method == FM_AUTO && g_tower_sub < 0)
            cand.push_back({LM_CUBIC, d / 3, g_pipe, 0, 0});
    } else {
        cand.push_back({pl.method, 0, g_pipe, 0, 0});
        const int other = pl.method == LM_PH ? LM_RR : LM_PH;
        if (g_force_method == FM_AUTO && (other == LM_RR ? pl.rr_ok : pl.ph_ok) &&
            logs_batch_fn(d, other))
            cand.push_back({other, 0, g_pipe, 0, 0});
    }
    // round 8: a subfield tower whose TP would take the shared tables past
    // 32 KB, also with TP from global memory (tower_tp_choice)
    if (g_tower_tp < 0) {
        std::vector<TuneCand> both;
        for (const TuneCand &c : cand) {
            both.push_back(c);
            u64 tpn = 0;
            if (c.method == LM_TOWER && c.sub_n &&
                tower_sub_choice(d / 2, c.sub_n, &tpn) == c.sub_n &&
                tower_tp_choice(d, c.sub_n, tpn))
                both.push_back({c.method, c.sub_n, c.pipe, 0, 1});
        }
        cand.swap(both);
    }
    // each tower/cubic/RR variant in every loop order -- round 5's (the
    // default), round 9's, and the grouped one where its kernel exists --
    // unless COARSE_PIPE fixes the order; a variant's orders one after
    // another, as they need no new tables
    if (!g_pipe_fixed) {
        std::vector<TuneCand> both;
        for (const TuneCand &c : cand) {
            if (c.method == LM_PH) { both.push_back(c); continue; }
            for (int o : {0, 1, 2}) {
                if (o == 2 && !logs_batch_fn(d, c.method, true)) continue;
                TuneCand t = c;
                t.pipe = o;
                both.push_back(t);
            }
        }
        cand.swap(both);
    }
    sub_use = cand[0].sub_n;
    pipe_use = cand[0].pipe;
    tp_use = cand[0].tpg;
    if (cand.size() < 2) return;

    u64 total, run, chunk;
    int nchunks;
    logs_geometry(d, total, run, chunk, nchunks);
    int lb = d - 4 < 27 ? d - 4 : 27;           // the sample: 2^lb dense u
    u64 ulo = logs_dense_stretch(d, lb);
    const u64 krun = run >= 512 ? run / 8 : run;
    unsigned long long zero64 = 0, cls = 0, nerr = 0;
    unsigned zero = 0;
    cudaEvent_t e0, e1;
    CUCHK(cudaEventCreate(&e0));
    CUCHK(cudaEventCreate(&e1));
    const double t0 = now_s();
    int cur_m = -1, cur_sub = -1, cur_tpg = -1;  // tables set up for these
    LogsParams Lcur;
    bool timed_any = false;
    for (size_t ci = 0; ci < cand.size(); ci++) {
        TuneCand &c = cand[ci];
        c.ns = -1;
        if (c.method != cur_m || c.sub_n != cur_sub || c.tpg != cur_tpg) {
            // (re)build the tables
            if (cur_m >= 0) {
                logs_free_device(G);
                cudaFree(G.dgp); G.dgp = nullptr;
                cudaFree(G.d_nsurv); G.d_nsurv = nullptr;
                cudaFree(G.d_nerr); G.d_nerr = nullptr;
            }
            cur_m = cur_sub = cur_tpg = -1;
            if (c.method == LM_RR && rrL.empty() && !rr_build_table(d, q, r, pl, rrL))
                continue;
            if (c.method == LM_CUBIC && !cubic_prepare(d, q, r, pl)) continue;
            TowerHost W;
            if (c.method == LM_TOWER && !tower_build_host(d, q, W, c.sub_n)) continue;
            LogsPlan pc = pl;
            pc.method = c.method;
            pc.tp_global = c.tpg;
            Lcur = logs_params_base(d, r, smax, q, c.method);
            CUCHK(cudaMalloc(&G.dgp, d * sizeof(fw)));
            CUCHK(cudaMemcpy(G.dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
            CUCHK(cudaMalloc(&G.d_nsurv, 8));
            CUCHK(cudaMalloc(&G.d_nerr, 8));
            logs_setup_device(G, d, q, pc, W, Th, Lcur, &rrL);
            cur_m = c.method;
            cur_sub = c.sub_n;
            cur_tpg = c.tpg;
        }
        const LogsBatchFn kfn = logs_batch_fn(d, c.method, c.pipe == 2);
        if (!kfn) continue;                     // (c.ns stays -1: n/a)
        Lcur.pipe = c.pipe;                     // (no new tables for this)
        Lcur.runctr = G.d_runctr;
        for (int rep = timed_any ? 1 : 0; rep < 2; rep++) {   // first: warm-up
            u64 uhi = ulo + ((u64)1 << lb); if (uhi > total) uhi = total;
            int blocks, threads;
            size_t shm;
            // round 7: runs of the sample short enough for ~16 per lane of
            // the whole wave (with runs of krun, 2^27 u held ~3.2 runs per
            // lane: some lanes had 4, the rest 3, and the sample's time
            // came out ~25% high -- ~10% for PH, whose wave has half the
            // lanes)
            logs_batch_dims(kfn, ~(u64)0 >> 2, c.method, Lcur, blocks, threads, shm);
            const u64 lanes = (u64)blocks * (u64)threads;
            u64 trun = (uhi - ulo + 16 * lanes - 1) / (16 * lanes);
            if (trun < 64) trun = 64;
            if (trun > krun) trun = krun;
            logs_batch_dims(kfn, (uhi - ulo + trun - 1) / trun, c.method, Lcur,
                            blocks, threads, shm);
            CUCHK(cudaMemcpy(G.d_nsurv, &zero64, 8, cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.d_nerr, &zero64, 8, cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.d_runctr, &zero64, 8, cudaMemcpyHostToDevice));
            CUCHK(cudaEventRecord(e0));
            kfn<<<blocks, threads, shm>>>(Lcur, G.dgp, ulo, uhi, trun, G.d_best,
                                          G.d_heavy, G.d_nheavy, G.d_overflow,
                                          G.d_nsurv, G.d_nerr);
            CUCHK(cudaEventRecord(e1));
            CUCHK(cudaEventSynchronize(e1));
            CUCHK(cudaGetLastError());
            wd_progress();
            float ms = 0;
            CUCHK(cudaEventElapsedTime(&ms, e0, e1));
            CUCHK(cudaMemcpy(&cls, G.d_nsurv, 8, cudaMemcpyDeviceToHost));
            CUCHK(cudaMemcpy(&nerr, G.d_nerr, 8, cudaMemcpyDeviceToHost));
            CUCHK(cudaMemcpy(G.d_nheavy, &zero, 4, cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.d_overflow, &zero, 4, cudaMemcpyHostToDevice));
            if (nerr) {
                fprintf(stderr, "d=%d: %llu discrete-log consistency errors "
                        "while tuning -- results are NOT trustworthy; "
                        "stopping\n", d, nerr);
                exit(1);
            }
            c.ns = cls ? ms * 1e6 / (double)cls : 0;
            // at least ~2^25 classes (a twice longer dense stretch)
            if (rep == 0 && cls < ((u64)1 << 25) && lb + 1 <= d - 4 && ms < 2000) {
                lb++;
                ulo = logs_dense_stretch(d, lb);
                rep--;
            }
        }
        timed_any = true;
    }
    if (cur_m >= 0) {
        logs_free_device(G);
        cudaFree(G.dgp); G.dgp = nullptr;
        cudaFree(G.d_nsurv); G.d_nsurv = nullptr;
        cudaFree(G.d_nerr); G.d_nerr = nullptr;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    size_t best = 0;
    for (size_t i = 1; i < cand.size(); i++)
        if (cand[i].ns >= 0 && (cand[best].ns < 0 || cand[i].ns < cand[best].ns))
            best = i;
    if (const char *e = getenv("COARSE_TUNE_PICK")) {    // test hook
        size_t k = (size_t)atoi(e);
        if (k < cand.size() && cand[k].ns >= 0) best = k;
    }
    if (cand[best].ns < 0) best = 0;            // nothing ran: keep the default
    pl.method = cand[best].method;
    pl.bytes = pl.method == LM_RR ? pl.rr_bytes
             : pl.method == LM_PH ? pl.ph.table_bytes
             : pl.method == LM_CUBIC ? pl.cubic_bytes : pl.bytes;
    sub_use = cand[best].sub_n;
    pipe_use = cand[best].pipe;
    tp_use = cand[best].tpg;
    if (pl.method != LM_RR) { rrL.clear(); rrL.shrink_to_fit(); }
    if (pl.method != LM_CUBIC) cubic_release();
    char nm[64];
    fprintf(stderr, "  [tune d=%d on GPU 0, %llu classes, %.1fs:", d, cls,
            now_s() - t0);
    for (size_t i = 0; i < cand.size(); i++) {
        tune_name(cand[i], nm, sizeof nm);
        if (cand[i].ns < 0) fprintf(stderr, "%s %s n/a", i ? "," : "", nm);
        else fprintf(stderr, "%s %s %.3f ns/class", i ? "," : "", nm, cand[i].ns);
    }
    tune_name(cand[best], nm, sizeof nm);
    fprintf(stderr, " -> %s]\n", nm);
}

// where a degree resumes: the first `next` of its nchunks done (in the
// order of chunk_at), covering `classes` classes, with that engine and
// tower subfield
struct ResumePos {
    int next = 0, nchunks = 0, method = LM_NONE, sub = -1, pipe = 1;
    int tpg = 0;                        // round 8: the tower's TP from global
    u64 classes = 0;
    double deg_s = -1;                  // time on the degree so far (< 0: unknown)
};

#if defined(COARSE_PHASE_TIMING)
// The phase timing of this run's chunks of degree d (round 8), summed over
// the devices, which it then frees: per log and lane, the warp-cycles of
// each phase -- a lane's time per log, as the 32 lanes of a warp run
// together (a warp-cycle serves all of them) -- and the shares.
static void pt_report(int d, int NG) {
    unsigned long long pt[PT_N] = {0};
    for (int g = 0; g < NG; g++) {
        GpuCtx &G = g_gpus[g];
        if (!G.d_prof) continue;
        CUCHK(cudaSetDevice(G.dev));
        CUCHK(cudaDeviceSynchronize());
        unsigned long long h[PT_N];
        CUCHK(cudaMemcpy(h, G.d_prof, sizeof h, cudaMemcpyDeviceToHost));
        for (int i = 0; i < PT_N; i++) pt[i] += h[i];
        cudaFree(G.d_prof);
        G.d_prof = nullptr;
    }
    static const char *const nm[PT_NPH] = {"fill", "g^u", "logs", "inversion",
                                           "marking", "other"};
    double tot = 0;
    for (int i = 0; i < PT_NPH; i++) tot += (double)pt[i];
    if (!pt[PT_ENTRY] || tot <= 0) {
        fprintf(stderr, "  [phase timing d=%d: no batches in this run]\n", d);
        return;
    }
    const double per = 32.0 / (double)pt[PT_ENTRY];   // per log and lane
    fprintf(stderr, "  [phase timing d=%d, warp-cycles per log and lane:", d);
    for (int i = 0; i < PT_NPH; i++)
        fprintf(stderr, "%s %s %.0f (%.1f%%)", i ? "," : "", nm[i],
                (double)pt[i] * per, 100.0 * (double)pt[i] / tot);
    fprintf(stderr, "; total %.0f; %llu logs in %llu warp batches, %.2f of 32 "
            "per lane]\n", tot * per, pt[PT_ENTRY], pt[PT_BATCH],
            pt[PT_BATCH] ? (double)pt[PT_ENTRY] / (32.0 * (double)pt[PT_BATCH]) : 0.0);
}
#endif

// Sieve degree d (any VRAM: tables only).  rs: resume position within the
// degree, or nullptr to start it.
static void sieve_degree_gpu_logs(int d, u64 r, u64 smax, const LogsPlan &pl_in,
                                  const ResumePos *rs) {
    LogsPlan pl = pl_in;                        // the tuner may switch engines
    const u64 M64 = ((u64)1 << d) - 1;
    const u64 q = find_primitive_poly(d);
    const u64 rmod = r % M64;
    if (gcd_u64(rmod ? rmod : M64, M64) != 1) {
        fprintf(stderr, "d=%d: gcd(r, 2^d-1) != 1; unsupported r\n", d);
        exit(1);
    }
    const int NG = (int)g_gpus.size();
    u64 total, run, chunk;
    int nchunks;
    logs_geometry(d, total, run, chunk, nchunks);
    const bool mid = rs && rs->nchunks > 0 && rs->next > 0;
    if (mid && rs->nchunks != nchunks) {
        fprintf(stderr, "d=%d: the checkpoint has %d chunks but this run has "
                "%d (a different COARSE_LOGS_CHUNK_LOG2?); cannot resume\n", d,
                rs->nchunks, nchunks);
        exit(1);
    }
    const int start_chunk = mid ? rs->next : 0;
    const u64 classes_before = mid ? rs->classes : 0;
    const u64 classes_all = gf2_irreducible_count(d);

    fw gpow2[64];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);

    double t0 = now_s();
    std::vector<u64> Th;                        // T, built once per degree
    std::vector<u64> rrL;                       // RR: the table, host copy
    int sub_use = g_tower_sub, pipe_use = g_pipe, tp_use = g_tower_tp > 0;
    if (mid && rs->method != LM_NONE) {
        // the degree's engine and subfield, as recorded: no re-tuning
        if (rs->method != pl.method) {
            const bool ok = rs->method == LM_RR ? pl.rr_ok
                          : rs->method == LM_PH ? pl.ph_ok
                          : rs->method == LM_CUBIC ? pl.cubic_ok
                          : rs->method == LM_TOWER ? d % 2 == 0 : false;
            if (!ok) {
                fprintf(stderr, "d=%d: the checkpoint's engine is not possible "
                        "here (less free VRAM?); cannot resume\n", d);
                exit(1);
            }
            pl.method = rs->method;
            pl.bytes = pl.method == LM_RR ? pl.rr_bytes
                     : pl.method == LM_CUBIC ? pl.cubic_bytes : pl.ph.table_bytes;
        }
        if (pl.method == LM_TOWER) sub_use = rs->sub;
        pipe_use = rs->pipe;
        tp_use = rs->tpg;
    } else if (g_tune && d >= g_tune_min) {
        logs_tune(d, q, r, smax, pl, gpow2, Th, rrL, sub_use, pipe_use, tp_use);
    }
    pl.tp_global = pl.method == LM_TOWER ? tp_use : 0;
    g_ckpt.cur_tpg = pl.tp_global;
    if (pipe_use == 2 && !logs_batch_fn(d, pl.method, true)) {
        // (COARSE_PIPE=2, or a checkpoint of the emulated build, below the
        // degrees that have the grouped kernels)
        fprintf(stderr, "d=%d: no grouped-order kernel at this degree; using "
                "the round-5 order\n", d);
        pipe_use = 0;
    }
    const LogsBatchFn kfn = logs_batch_fn(d, pl.method, pipe_use == 2);
    if (!kfn) {
        fprintf(stderr, "d=%d: no kernel for this engine -- internal error\n", d);
        exit(1);
    }
    const char *mname = pl.method == LM_TOWER ? "tower"
                      : pl.method == LM_CUBIC ? "cubic tower"
                      : pl.method == LM_RR ? "RR" : "PH";
    if (pl.method == LM_RR && rrL.empty() && !rr_build_table(d, q, r, pl, rrL)) {
        fprintf(stderr, "d=%d: RR table construction failed\n", d);
        exit(1);
    }
    if (pl.method == LM_CUBIC && !cubic_prepare(d, q, r, pl)) exit(1);
    TowerHost W;
    if (pl.method == LM_TOWER && !tower_build_host(d, q, W, sub_use)) {
        fprintf(stderr, "d=%d: tower construction failed -- internal error\n", d);
        exit(1);
    }
    const int sub_rec = pl.method == LM_TOWER ? (W.sub_k ? W.sub_n : 0)
                      : pl.method == LM_CUBIC ? d / 3 : -1;
    std::vector<LogsParams> Lg;
    unsigned long long zero64 = 0;
    for (int g = 0; g < NG; g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        CUCHK(cudaMalloc(&G.dgp, d * sizeof(fw)));
        CUCHK(cudaMemcpy(G.dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
        CUCHK(cudaMalloc(&G.d_nsurv, 8));
        CUCHK(cudaMemcpy(G.d_nsurv, &zero64, 8, cudaMemcpyHostToDevice));
        CUCHK(cudaMalloc(&G.d_nerr, 8));
        CUCHK(cudaMemcpy(G.d_nerr, &zero64, 8, cudaMemcpyHostToDevice));
        Lg.push_back(logs_params_base(d, r, smax, q, pl.method));
        Lg.back().pipe = pipe_use;
        Lg.back().runctr = G.d_runctr;
#if defined(COARSE_PHASE_TIMING)
        CUCHK(cudaMalloc(&G.d_prof, PT_N * sizeof(unsigned long long)));
        CUCHK(cudaMemset(G.d_prof, 0, PT_N * sizeof(unsigned long long)));
        Lg.back().prof = G.d_prof;
#endif
        // test hook: COARSE_TEST_FAULT_EVERY=n flips a bit of the log of
        // every class with u = 1 (mod n), to exercise the error stop
        if (const char *e = getenv("COARSE_TEST_FAULT_EVERY"))
            Lg.back().fault = strtoull(e, nullptr, 10);
        logs_setup_device(G, d, q, pl, W, Th, Lg.back(), &rrL);
    }
    rrL.clear(); rrL.shrink_to_fit();
    cubic_release();                            // (on the devices now)
    char inner[96] = "";                        // the inner tower, if any
    if (pl.method == LM_TOWER && W.sub_k)
        snprintf(inner, sizeof inner, " over GF(2^%d)", W.sub_n);
    else if (pl.method == LM_TOWER)
        snprintf(inner, sizeof inner, ", classic");
    else if (pl.method == LM_CUBIC)
        snprintf(inner, sizeof inner, " over GF(2^%d)", d / 3);
    if (pl.method != LM_PH)
        snprintf(inner + strlen(inner), sizeof inner - strlen(inner), "%s",
                 pipe_use == 0 ? ", round-5 loop order"
                 : pipe_use == 2 ? ", grouped loop order" : ", round-9 loop order");
    if (pl.method == LM_TOWER && pl.tp_global && Lg[0].tw.sub_k && !Lg[0].tw.shm_tp)
        snprintf(inner + strlen(inner), sizeof inner - strlen(inner),
                 ", TP from global memory");
    W = TowerHost();                            // host copies no longer needed
    Th.clear(); Th.shrink_to_fit();
    if (mid)
        fprintf(stderr, "d=%d: resuming at chunk %d/%d (%" PRIu64 " of %" PRIu64
                " classes done)\n", d, start_chunk, nchunks, classes_before,
                classes_all);
    fprintf(stderr, "d=%2d logs (%s%s, %.0f MB tables, built in %.1fs), %d "
            "chunk%s", d, mname, inner, pl.bytes / 1048576.0, now_s() - t0,
            nchunks, nchunks == 1 ? "" : "s");
    if (NG > 1) fprintf(stderr, " on %d GPUs", NG);
    fprintf(stderr, ": ");
    if (g_logs_selfcheck) { fprintf(stderr, "\n"); logs_selfcheck(d, Lg[0]); }

    // One blocking-sync event per device: the host thread sleeps in the
    // driver until its chunk is done and wakes at once -- no polling delay
    // between chunks and no busy-waiting CPU core.  (Under WSL2 the wait
    // appears to spin a core anyway; COARSE_SYNC=poll, round 9, polls the
    // event every millisecond instead: ~0.1% of a chunk.)
    std::vector<cudaEvent_t> ev(NG);
    for (int g = 0; g < NG; g++) {
        CUCHK(cudaSetDevice(g_gpus[g].dev));
        CUCHK(cudaEventCreateWithFlags(&ev[g], cudaEventBlockingSync |
                                               cudaEventDisableTiming));
    }

    const int dot_every = nchunks > 64 ? (nchunks + 63) / 64 : 1;
    for (int c = 0; c < start_chunk; c += dot_every) fputc('=', stderr);
    fflush(stderr);
    unsigned zero = 0;
    unsigned long long total_heavy = 0;
    u64 classes_run = 0;                        // covered by this run's chunks
    // time left: this run's time per class times the classes left (in the
    // order of chunk_at the classes done so far are a fair sample); shown
    // once, at the first dot after 256 chunks (or 1/8 of a short pass)
    const double t_pass = now_s();
    const int eta_after = nchunks / 8 < 256 ? (nchunks / 8 > 0 ? nchunks / 8 : 1) : 256;
    g_ckpt.eta_s = -1;
    bool eta_shown = false;
    // the degree's time in all runs, as one uninterrupted run would show it
    // (round 7): a fresh degree counts from t0 (tables, tuning, pass); a
    // resumed one adds this run's pass to the time recorded in the
    // checkpoint, not its table rebuild (< 0: unknown -- a round-6 checkpoint)
    const double deg_before = mid ? rs->deg_s : 0;
    const double t_count = mid ? t_pass : t0;
    auto deg_time = [&]() {
        return deg_before < 0 ? -1.0 : deg_before + (now_s() - t_count);
    };
    for (int base = start_chunk; base < nchunks; base += NG) {
        // round boundary: every chunk < base is complete on every device
        const u64 classes_now = classes_before + classes_run;
        if (ckpt_stop_due()) {
            fprintf(stderr, "\n");
#if defined(COARSE_PHASE_TIMING)
            pt_report(d, NG);
#endif
            ckpt_snapshot(d, base, nchunks, pl.method, sub_rec, pipe_use,
                          classes_now, deg_time());
            ckpt_stop_and_exit();
        }
        if (ckpt_periodic_due()) {
            ckpt_snapshot(d, base, nchunks, pl.method, sub_rec, pipe_use,
                          classes_now, deg_time());
            ckpt_write();
        }
        int used = 0;
        for (int g = 0; g < NG && base + g < nchunks; g++) {
            GpuCtx &G = g_gpus[g];
            CUCHK(cudaSetDevice(G.dev));
            u64 ulo = chunk_at(base + g, nchunks) * chunk;
            u64 uhi = ulo + chunk; if (uhi > total) uhi = total;
            CUCHK(cudaMemcpy(G.d_nheavy, &zero, 4, cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.d_overflow, &zero, 4, cudaMemcpyHostToDevice));
            CUCHK(cudaMemset(G.d_runctr, 0, 8));
            // runs of krun = run/8 u-values: ~50 per thread, so threads
            // finish together (the chunk itself is unchanged; with FILL_POOL
            // a lane takes its next run when it needs one)
            const u64 krun = run >= 512 ? run / 8 : run;
            int blocks, threads;
            size_t shm;
            logs_batch_dims(kfn, (uhi - ulo + krun - 1) / krun, pl.method,
                            Lg[g], blocks, threads, shm);
            kfn<<<blocks, threads, shm>>>(Lg[g], G.dgp, ulo, uhi, krun,
                                          G.d_best, G.d_heavy, G.d_nheavy,
                                          G.d_overflow, G.d_nsurv, G.d_nerr);
            CUCHK(cudaEventRecord(ev[g], 0));
            used++;
        }
        for (int g = 0; g < used; g++) {        // a chunk is seconds at most
            CUCHK(cudaSetDevice(g_gpus[g].dev));
            if (g_sync_poll) {
                cudaError_t qe;
                while ((qe = cudaEventQuery(ev[g])) == cudaErrorNotReady) usleep(1000);
                CUCHK(qe);
            } else {
                CUCHK(cudaEventSynchronize(ev[g]));
            }
        }
        classes_run = 0;
        for (int g = 0; g < NG; g++) {
            GpuCtx &G = g_gpus[g];
            CUCHK(cudaSetDevice(G.dev));
            unsigned nheavy = 0, overflow = 0;
            unsigned long long nbad = 0, ncls = 0;
            CUCHK(cudaMemcpy(&ncls, G.d_nsurv, 8, cudaMemcpyDeviceToHost));
            classes_run += ncls;
            if (g >= used) continue;
            CUCHK(cudaMemcpy(&nheavy, G.d_nheavy, 4, cudaMemcpyDeviceToHost));
            CUCHK(cudaMemcpy(&overflow, G.d_overflow, 4, cudaMemcpyDeviceToHost));
            CUCHK(cudaMemcpy(&nbad, G.d_nerr, 8, cudaMemcpyDeviceToHost));
            if (overflow) {
                fprintf(stderr, "\nd=%d: heavy queue overflow (%u); raise "
                        "HEAVY_CAP\n", d, overflow);
                exit(1);
            }
            // checked every round, so no checkpoint ever includes a chunk
            // in which an inconsistency was seen
            if (nbad) {
                fprintf(stderr, "\nd=%d: %llu discrete-log consistency "
                        "errors in chunk %" PRIu64 " (pass position %d, GPU %d) "
                        "-- results are NOT trustworthy; stopping (no output "
                        "written; any checkpoint predates this chunk)\n", d,
                        nbad, chunk_at(base + g, nchunks), base + g, g);
                exit(1);
            }
            // (queued ahead of the next chunk, which waits for it in-stream)
            if (nheavy) k_mark_heavy<<<nheavy, 256>>>(G.d_heavy, nheavy, smax,
                                                       G.d_best);
            total_heavy += nheavy;
        }
        g_ckpt.test_chunks_done += used;
        wd_progress();                          // (round 8: the watchdog)
        wd_test_hang_check();
        const u64 cov = classes_before + classes_run;
        if (classes_run && cov < classes_all)
            g_ckpt.eta_s = (now_s() - t_pass) * (double)(classes_all - cov) /
                           (double)classes_run;
        for (int c = base; c < base + used; c++)
            if ((c + 1) % dot_every == 0 || c + 1 == nchunks) {
                fputc('.', stderr);
                // the estimate, if over 10 min (test hook COARSE_TEST_ETA=1:
                // always, in seconds)
                if (!eta_shown && c + 1 < nchunks && c + 1 - start_chunk >= eta_after &&
                    g_ckpt.eta_s >= 0) {
                    eta_shown = true;
                    char b[48];
                    eta_text(g_ckpt.eta_s, b, sizeof b);
                    if (g_test_eta) fprintf(stderr, "[eta %.3f s]", g_ckpt.eta_s);
                    else if (g_ckpt.eta_s > 600) fprintf(stderr, "[%s]", b);
                }
            }
        fflush(stderr);
    }
    fprintf(stderr, "\n");
    g_ckpt.eta_s = -1;

#if defined(COARSE_PHASE_TIMING)
    pt_report(d, NG);
#endif
    unsigned long long nsurv = 0, nerr = 0;
    for (int g = 0; g < NG; g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        CUCHK(cudaDeviceSynchronize());
        CUCHK(cudaGetLastError());
        CUCHK(cudaEventDestroy(ev[g]));
        unsigned long long c = 0, ne = 0;     // (not 'e': CUCHK declares e)
        CUCHK(cudaMemcpy(&c, G.d_nsurv, 8, cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(&ne, G.d_nerr, 8, cudaMemcpyDeviceToHost));
        nsurv += c; nerr += ne;
        cudaFree(G.dgp); G.dgp = nullptr;
        cudaFree(G.d_nsurv); G.d_nsurv = nullptr;
        cudaFree(G.d_nerr); G.d_nerr = nullptr;
        logs_free_device(G);
    }
    if (nerr) {
        fprintf(stderr, "d=%d: %llu discrete-log consistency errors -- this "
                "degree's results are NOT trustworthy; stopping\n", d, nerr);
        exit(1);
    }
    // every class of the degree exactly once (partners included), across a
    // resume too
    if (classes_before + nsurv != classes_all) {
        fprintf(stderr, "d=%d: %" PRIu64 " classes covered, but there are %"
                PRIu64 " irreducible polynomials of degree %d -- internal "
                "error; results are NOT trustworthy\n", d,
                classes_before + (u64)nsurv, classes_all, d);
        exit(1);
    }
    // the time: of this run; of all runs, if resumed (Item 3 of round 7)
    const double t_this = now_s() - t0, t_all = deg_time();
    char tb[192];
    if (!mid)
        snprintf(tb, sizeof tb, "%.2fs", t_this);
    else if (t_all >= 0)
        snprintf(tb, sizeof tb, "%.2fs in all (this run %.2fs, resumed at chunk "
                 "%d of %d)", t_all, t_this, start_chunk, nchunks);
    else
        snprintf(tb, sizeof tb, "%.2fs in this run (resumed at chunk %d of %d; "
                 "the checkpoint, from round 6, has no time for the runs before)",
                 t_this, start_chunk, nchunks);
    fprintf(stderr, "d=%2d done (logs/%s, %d chunks): %" PRIu64 " classes, %llu "
            "heavy progressions, %s\n", d, mname, nchunks, classes_all,
            total_heavy, tb);
    if (t_all >= 0) g_ckpt.run_before_s += t_all;
    else g_ckpt.run_known = false;
}

// ---------------------------------------------------------------------------
// Sieve degrees d_from..d_hi into best[] (smax + 1 entries), starting from
// `start` (the seed, or a checkpoint's image; nullptr: nothing found yet);
// rs: where degree d_from resumes.  Returns the last degree fully sieved
// (d_from - 1 if none).
static int run_gpu(u64 r, int d_from, int d_hi, u64 smax, std::vector<ull> &best,
                   const std::vector<ull> *start, const ResumePos *rs) {
    // Every device gets a private best[]; the elementwise min across them is
    // the answer (see MULTI-GPU).  Seeding each with the same start keeps
    // that identity true from the beginning.
    for (size_t g = 0; g < g_gpus.size(); g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        CUCHK(cudaMalloc(&G.d_best, (smax + 1) * sizeof(ull)));
        CUCHK(cudaMalloc(&G.d_heavy, HEAVY_CAP * sizeof(Heavy)));
        CUCHK(cudaMalloc(&G.d_nheavy, 4));
        CUCHK(cudaMalloc(&G.d_overflow, 4));
        CUCHK(cudaMalloc(&G.d_runctr, 8));
        if (start) {
            CUCHK(cudaMemcpy(G.d_best, start->data(), (smax + 1) * sizeof(ull),
                             cudaMemcpyHostToDevice));
        } else {
            int blocks, threads;
            launch_dims(smax + 1, blocks, threads);
            k_fill<<<blocks, threads>>>(G.d_best, smax + 1, ~0ULL);
        }
    }
    CUCHK(cudaSetDevice(g_gpus[0].dev));
    g_wd_armed = 1;                             // (round 8: the watchdog)
    wd_progress();
    int reached = d_from - 1;
    // test hook: COARSE_TEST_STOP_DEGREE=n behaves as if degree n could not
    // be sieved here (exercises the early-stop handling in main)
    const char *tsd = getenv("COARSE_TEST_STOP_DEGREE");
    const int test_stop_degree = tsd ? atoi(tsd) : 0;
    for (int d = d_from; d <= d_hi; d++) {
        if (d == test_stop_degree) {
            fprintf(stderr, "d=%d: COARSE_TEST_STOP_DEGREE: stopping after "
                    "degree %d\n", d, d - 1);
            break;
        }
        // tables sized for the most constrained device
        size_t freeb = (size_t)-1;
        for (size_t g = 0; g < g_gpus.size(); g++) {
            size_t f = 0, t = 0;
            CUCHK(cudaSetDevice(g_gpus[g].dev));
            cudaMemGetInfo(&f, &t);
            if (f < freeb) freeb = f;
        }
        CUCHK(cudaSetDevice(g_gpus[0].dev));
        LogsPlan pl;
        const size_t budget = logs_table_budget(freeb);
        if (!logs_plan(d, find_primitive_poly(d), g_force_method, budget, pl)) {
            fprintf(stderr, "d=%d: no discrete-log engine fits in %.2f GB of free "
                    "VRAM%s; stopping after degree %d\n", d,
                    budget / 1073741824.0, d == 49 ? " (RR needs ~1.5 GB or "
                    "more at d=49)" : "", d - 1);
            break;
        }
        sieve_degree_gpu_logs(d, r, smax, pl, d == d_from ? rs : nullptr);
        // Degree boundary: best[] is a clean "degrees <= d done" state.
        if (ckpt_stop_due()) {
            ckpt_snapshot(d + 1, 0, 0, LM_NONE, -1, 1, 0, 0);
            ckpt_stop_and_exit();
        }
        reached = d;
        // a periodic save only if more degrees remain (else the outputs
        // follow at once)
        if (d < d_hi && ckpt_periodic_due()) {
            ckpt_snapshot(d + 1, 0, 0, LM_NONE, -1, 1, 0, 0);
            ckpt_write();
        }
    }
    if (reached < d_hi && g_ckpt.enabled) {
        // stopped early (a degree that cannot be done here): save the degree
        // boundary, so a resume -- e.g. with more free VRAM -- starts there
        ckpt_snapshot(reached + 1, 0, 0, LM_NONE, -1, 1, 0, 0);
        ckpt_write();
    }
    best.resize(smax + 1);
    gather_best(best.data(), smax + 1);
    for (size_t g = 0; g < g_gpus.size(); g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        cudaFree(G.d_best); cudaFree(G.d_heavy);
        cudaFree(G.d_nheavy); cudaFree(G.d_overflow); cudaFree(G.d_runctr);
        G.d_runctr = nullptr;
        G.d_best = nullptr;
    }
    g_wd_armed = 0;
    g_gather_tmp.clear(); g_gather_tmp.shrink_to_fit();
    return reached;
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
            u64 packed = key_pack(d, mask);
            for (u64 s = (s0 == 0 ? n : s0); s <= smax; s += n)
                if (packed < best[s]) best[s] = packed;
        }
    }
}

// ---------------------------------------------------------------------------
// --selftest [engine]: the GPU sieve against the CPU reference at
// r = 136279841, d = 2..20, s <= 999999, then the device checks of the
// logs path (logs_wide_selftest).  Exit status 0 iff everything passed.
static int selftest_main(const char *engine) {
    g_gpus.assign(1, GpuCtx());
    g_force_method = !engine ? FM_AUTO : !strcmp(engine, "tower") ? FM_TOWER
                   : !strcmp(engine, "ph") ? FM_PH : !strcmp(engine, "rr") ? FM_RR
                   : !strcmp(engine, "cubic") ? FM_CUBIC : -1;
    if (g_force_method < 0) {
        fprintf(stderr, "--selftest %s: expected tower, cubic, ph or rr\n", engine);
        return 1;
    }
    g_logs_selfcheck = true;
    const u64 r = 136279841, smax = 999999;
    const int depth = 20;
    printf("selftest: r=%" PRIu64 " degrees 2..%d s<=%" PRIu64 " (GPU vs "
           "in-process CPU reference), engines: %s\n", r, depth, smax,
           g_force_method == FM_TOWER ? "tower at even d"
           : g_force_method == FM_CUBIC ? "cubic tower at 6 | d"
           : g_force_method == FM_PH ? "PH at odd d"
           : g_force_method == FM_RR ? "RR at odd d >= 5" : "automatic");
    // the pass order: a permutation of the chunks for every geometry, whose
    // first 2^j blocks hold one block of each 1/2^j of the range
    u64 order_bad = 0;
    for (int lg = 0; lg <= 18; lg++) {
        const int nc = 1 << lg;
        std::vector<char> seen(nc, 0);
        for (int p = 0; p < nc; p++) {
            const u64 c = chunk_at(p, nc);
            if (c >= (u64)nc || seen[c]) order_bad++;
            else seen[c] = 1;
        }
        const int nb = nc / CHUNK_BLOCK;
        for (int j = 1; nb > 1 && (1 << j) <= nb; j++) {
            std::vector<char> hit(1 << j, 0);
            for (int q = 0; q < (1 << j); q++) {
                const u64 blk = chunk_at(q * CHUNK_BLOCK, nc) / CHUNK_BLOCK;
                const u64 iv = blk / (u64)(nb >> j);
                if (hit[iv]) order_bad++;
                hit[iv] = 1;
            }
        }
    }
    fprintf(stderr, "pass order (chunk_at, 1..2^18 chunks): %s\n",
            order_bad ? "FAIL" : "ok");
    std::vector<ull> g, c;
    run_gpu(r, 2, depth, smax, g, nullptr, nullptr);
    double t0 = now_s();
    sieve_cpu(r, depth, smax, c);
    fprintf(stderr, "cpu reference: %.2fs\n", now_s() - t0);
    u64 bad = 0;
    for (u64 s = 1; s <= smax; s++)
        if (g[s] != c[s] && ++bad <= 10)
            fprintf(stderr, "MISMATCH s=%" PRIu64 " gpu=%" PRIx64 " cpu=%" PRIx64
                    "\n", s, (u64)g[s], (u64)c[s]);
    fprintf(stderr, "logs-path checks at d <= %d (device 0):\n", KEY_MAX_DEGREE);
    const u64 wide_fails = logs_wide_selftest(r, g_force_method);
    if (wide_fails)
        printf("selftest: logs-path checks: %" PRIu64 " failures\n", wide_fails);
    const bool fail = bad || wide_fails || order_bad;
    printf("selftest: %s (%" PRIu64 " mismatches)\n", fail ? "FAIL" : "PASS", bad);
    return fail ? 1 : 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s <r> <degrees> <out_prefix> [options]\n"
        "       %s --selftest [tower|ph|rr]\n"
        "  <degrees>  D1-D2 (degrees D1..D2) or D (degree D alone); a run from\n"
        "             scratch starts at 2, a run starting at D1 > 2 needs\n"
        "             --load-survivors with the survivors of a run through D1-1\n"
        "options:\n"
        "  --load-survivors <file>  the seed: survivors of the run through D1-1\n"
        "  --no-cert              do not write <out_prefix>.found.cert\n"
        "  --no-survivors         do not write <out_prefix>.survivors.txt\n"
        "  --checkpoint <file>    checkpoint file (default <out_prefix>.ckpt); the\n"
        "                         same command continues from it after any stop\n"
        "  --checkpoint-mins <N>  save every N minutes (default %d; 0 = on signals\n"
        "                         only)\n"
        "  --no-checkpoint        no checkpoint file at all\n"
        "  --fresh                set an existing checkpoint aside (*.old), start over\n"
        "  --gpus <N>             use N GPUs (default: all present)\n"
        "  --no-tune              skip the per-degree timing of kernel variants\n"
        "  --watchdog-mins <N>    exit with status 3 if nothing progresses for N\n"
        "                         minutes, e.g. a hung GPU (default %d; 0 = off)\n"
        "examples:\n"
        "  %s 136279841 2-46 out2-46\n"
        "  %s 136279841 47 out47 --load-survivors out2-46.survivors.txt\n",
        prog, prog, CKPT_DEFAULT_MINS, WD_DEFAULT_MINS, prog, prog);
}

// "D" or "D1-D2", 2 <= D1 <= D2 <= KEY_MAX_DEGREE
static bool parse_degrees(const char *s, int &lo, int &hi) {
    char *end = nullptr;
    long a = strtol(s, &end, 10);
    if (end == s) return false;
    long b = a;
    if (*end == '-') {
        const char *t = end + 1;
        b = strtol(t, &end, 10);
        if (end == t) return false;
    }
    if (*end != '\0' || a < 2 || b < a || b > KEY_MAX_DEGREE) return false;
    lo = (int)a; hi = (int)b;
    return true;
}

int main(int argc, char **argv) {
    // COARSE_TUNE=0 turns the per-degree auto-tuning off (like --no-tune);
    // COARSE_TUNE_MIN=d tunes from degree d up (default 40)
    if (const char *e = getenv("COARSE_TUNE")) g_tune = atoi(e) != 0;
    if (const char *e = getenv("COARSE_TUNE_MIN")) g_tune_min = atoi(e);
    // COARSE_TOWER_SUBN=0 uses the classic tower; =n forces the subfield K_n
    // of the inner tower (tests); default: automatic
    if (const char *e = getenv("COARSE_TOWER_SUBN")) g_tower_sub = atoi(e);
    if (const char *e = getenv("COARSE_TEST_ETA")) g_test_eta = atoi(e) != 0;
    if (const char *e = getenv("COARSE_PIPE")) {
        char *end = nullptr;
        const long o = strtol(e, &end, 10);
        if (end == e || *end || o < 0 || o > 2) {
            fprintf(stderr, "COARSE_PIPE=%s: expected 0 (round 5's loop order, "
                    "the default), 1 (round 9's) or 2 (grouped)\n", e);
            return 1;
        }
        g_pipe = (int)o;
        g_pipe_fixed = true;
    }
    if (const char *e = getenv("COARSE_LDCG")) g_ldcg = atoi(e) != 0;
    if (const char *e = getenv("COARSE_PREFETCH")) g_pf = atoi(e) != 0;
    if (const char *e = getenv("COARSE_SYNC")) {
        if (!strcmp(e, "poll")) {
            g_sync_poll = true;
            fprintf(stderr, "COARSE_SYNC=poll: chunks are waited for by polling their "
                    "events every millisecond\n");
        } else if (strcmp(e, "block")) {
            fprintf(stderr, "COARSE_SYNC=%s: expected block (the default) or poll\n", e);
            return 1;
        }
    }
    if (const char *e = getenv("COARSE_TEST_TP_CHOICE")) g_test_tp_choice = atoi(e) != 0;
    if (const char *e = getenv("COARSE_TOWER_TP")) {
        if (!strcmp(e, "global")) g_tower_tp = 1;
        else if (!strcmp(e, "shared")) g_tower_tp = 0;
        else if (strcmp(e, "auto")) {
            fprintf(stderr, "COARSE_TOWER_TP=%s: expected auto, shared or global\n", e);
            return 1;
        }
    }
    if (const char *e = getenv("COARSE_FILL")) {
        char *end = nullptr;
        const long f = strtol(e, &end, 10);
        if (end == e || *end || f < FILL_LANE || f > FILL_POOL) {
            fprintf(stderr, "COARSE_FILL=%s: expected 0 (round 6), 1, 2 or 3 "
                    "(the default)\n", e);
            return 1;
        }
        g_fill = (int)f;
    }
    if (const char *e = getenv("COARSE_METHOD")) {
        if (!strcmp(e, "tower")) g_force_method = FM_TOWER;
        else if (!strcmp(e, "ph")) g_force_method = FM_PH;
        else if (!strcmp(e, "rr")) g_force_method = FM_RR;
        else if (!strcmp(e, "cubic")) g_force_method = FM_CUBIC;
        else if (strcmp(e, "auto")) {
            fprintf(stderr, "COARSE_METHOD=%s: expected auto, tower, cubic, ph or rr\n", e);
            return 1;
        }
    }
    if (argc >= 2 && !strcmp(argv[1], "--selftest"))
        return selftest_main(argc >= 3 ? argv[2] : nullptr);
    if (argc < 4) { usage(argv[0]); return 1; }

    char *rend = nullptr;
    const u64 r = strtoull(argv[1], &rend, 10);
    if (*rend || r < 3) {
        fprintf(stderr, "r = '%s': expected an integer >= 3\n", argv[1]);
        return 1;
    }
    int d_lo = 0, d_hi = 0;
    if (!parse_degrees(argv[2], d_lo, d_hi)) {
        fprintf(stderr, "degrees '%s': expected D or D1-D2 with 2 <= D1 <= D2 <= "
                "%d\n", argv[2], KEY_MAX_DEGREE);
        return 1;
    }
    const char *prefix = argv[3];
    u64 smax = r / 2;
    if (const char *e = getenv("COARSE_TEST_SMAX")) {   // tests: a short range
        smax = strtoull(e, nullptr, 10);
        if (smax < 1 || smax > r / 2) { fprintf(stderr, "bad COARSE_TEST_SMAX\n"); return 1; }
    }

    bool write_survivors = true, write_cert = true;
    const char *ckpt_path = nullptr, *loadsurv_path = nullptr;
    int want_gpus = 0;                           // 0 = use every device
    bool no_ckpt = false, fresh = false;
    double ckpt_mins = CKPT_DEFAULT_MINS;
    for (int a = 4; a < argc; a++) {
        const bool more = a + 1 < argc;
        if (!strcmp(argv[a], "--no-cert")) write_cert = false;
        else if (!strcmp(argv[a], "--no-survivors")) write_survivors = false;
        else if (!strcmp(argv[a], "--load-survivors") && more) loadsurv_path = argv[++a];
        else if (!strcmp(argv[a], "--checkpoint") && more) ckpt_path = argv[++a];
        else if (!strcmp(argv[a], "--checkpoint-mins") && more) ckpt_mins = atof(argv[++a]);
        else if (!strcmp(argv[a], "--gpus") && more) want_gpus = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--no-checkpoint")) no_ckpt = true;
        else if (!strcmp(argv[a], "--fresh")) fresh = true;
        else if (!strcmp(argv[a], "--no-tune")) g_tune = false;
        else if (!strcmp(argv[a], "--watchdog-mins") && more) {
            char *end = nullptr;
            const double m = strtod(argv[++a], &end);
            if (end == argv[a] || *end || m < 0) {
                fprintf(stderr, "--watchdog-mins %s: expected minutes >= 0\n", argv[a]);
                return 1;
            }
            g_wd_limit_s = m * 60.0;
        }
        else {
            fprintf(stderr, "unknown or incomplete option '%s'\n", argv[a]);
            usage(argv[0]);
            return 1;
        }
    }
    // the seed: required exactly when the run starts above degree 2
    if (d_lo > 2 && !loadsurv_path) {
        fprintf(stderr, "degrees %s: a run starting at degree %d needs "
                "--load-survivors <file> with the survivors of a run through "
                "degree %d (for a run from scratch, write 2-%d)\n", argv[2],
                d_lo, d_lo - 1, d_hi);
        return 1;
    }
    if (d_lo == 2 && loadsurv_path) {
        fprintf(stderr, "--load-survivors: a run from degree 2 has no lower "
                "degrees to take from a seed file\n");
        return 1;
    }
    if (no_ckpt && (ckpt_path || fresh)) {
        fprintf(stderr, "--no-checkpoint conflicts with --checkpoint / --fresh\n");
        return 1;
    }
    // test hook (deterministic checkpoint testing without a real signal):
    // COARSE_TEST_STOP_AFTER_CHUNKS=N acts as a stop signal after N chunks
    if (const char *e = getenv("COARSE_TEST_STOP_AFTER_CHUNKS"))
        g_ckpt.test_stop_after = atol(e);
    // test hooks of the watchdog (round 8): its limit in seconds, and a hang
    if (const char *e = getenv("COARSE_TEST_WATCHDOG_SECS")) g_wd_limit_s = atof(e);
    if (const char *e = getenv("COARSE_TEST_HANG_AFTER_CHUNKS"))
        g_test_hang_after = atol(e);

    std::vector<ull> seed;
    u64 seed_hash = 0;
    if (loadsurv_path) {
        u64 nsurv = 0, nlines = 0, noor = 0;
        if (!read_survivors_image(loadsurv_path, smax, seed, &nsurv, &nlines, &noor))
            return 1;
        if (noor) fprintf(stderr, "warning: %" PRIu64 " survivor entries "
                          "outside 1..%" PRIu64 " ignored\n", noor, smax);
        seed_hash = ckpt_hash(seed.data() + 1, smax);
        fprintf(stderr, "loaded %s: %" PRIu64 " survivors carried in (%" PRIu64
                " lines); sieving degrees %d..%d\n", loadsurv_path, nsurv, nlines,
                d_lo, d_hi);
        fprintf(stderr, "note: the output records only the newly found degree-"
                "%d..%d factors; keep the earlier runs' .found.cert for the "
                "lower degrees\n", d_lo, d_hi);
    }

    // checkpointing: the file, its lock, and an automatic resume
    std::string auto_path;
    if (!no_ckpt && !ckpt_path) {
        auto_path = std::string(prefix) + ".ckpt";
        ckpt_path = auto_path.c_str();
    }
    std::vector<ull> start;                      // best[] to start from
    bool have_start = false, resumed = false;
    int d_from = d_lo;
    ResumePos rs;
    if (ckpt_path) {
        if (!ckpt_lock(ckpt_path)) return 1;
        if (fresh) ckpt_set_aside(ckpt_path);
        else {
            CkptInfo ci;
            std::string gen, why;
            const int st = ckpt_resume(ckpt_path, r, smax, d_lo, d_hi,
                                       loadsurv_path != nullptr, seed_hash, seed,
                                       ci, start, gen, why);
            if (st == -2) {
                fprintf(stderr, "checkpoint %s belongs to a different run: %s.\n"
                        "  Run that command line to continue it; or add --fresh "
                        "to start this one over (the old file is kept as *.old); "
                        "or checkpoint elsewhere with --checkpoint <file>\n",
                        gen.c_str(), why.c_str());
                return 1;
            }
            if (st == -1) {
                fprintf(stderr, "checkpoint %s exists but no generation of it is "
                        "usable (an older format, or damaged); --fresh sets it "
                        "aside and starts over\n", ckpt_path);
                return 1;
            }
            if (st == 1) {
                have_start = true;
                d_from = ci.d;
                rs.next = ci.next; rs.nchunks = ci.nchunks;
                rs.method = ci.method; rs.sub = ci.sub; rs.pipe = ci.pipe;
                rs.tpg = ci.tpg ? 1 : 0;
                rs.classes = ci.classes;
                rs.deg_s = ci.deg_known ? ci.deg_s : -1;
                g_ckpt.run_before_s = ci.run_known ? ci.run_s : 0;
                g_ckpt.run_known = ci.run_known;
                resumed = true;
                fprintf(stderr, "found checkpoint %s (%s): continuing that run "
                        "(--fresh starts over)\n", gen.c_str(),
                        ckpt_describe(ci).c_str());
                if (d_from > d_hi) {             // finished, outputs not written
                    fprintf(stderr, "note: every degree is done already; "
                            "writing the outputs\n");
                }
            }
        }
        g_ckpt.enabled = true;
        g_ckpt.path = ckpt_path;
        g_ckpt.period_s = ckpt_mins > 0 ? ckpt_mins * 60.0 : 0;
        g_ckpt.last_write = now_s();
        g_ckpt.r = r; g_ckpt.smax = smax;
        g_ckpt.d_lo = d_lo; g_ckpt.d_hi = d_hi;
        if (loadsurv_path) {
            g_ckpt.seed_path = loadsurv_path;
            g_ckpt.seed_hash = seed_hash;
            g_ckpt.base = seed;                  // the delta baseline
        }
        // the first SIGINT/SIGTERM saves and exits at the next chunk
        // boundary; a second one kills at once (after unlinking the lock
        // file: ckpt_signal_handler) -- the checkpoint file is replaced
        // atomically, so that is safe too
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = ckpt_signal_handler;
        sa.sa_flags = 0;                         // (round 7: SA_RESETHAND)
        sigaction(SIGTERM, &sa, nullptr);
        sigaction(SIGINT, &sa, nullptr);
        fprintf(stderr, "checkpointing to %s (%s)", ckpt_path,
                loadsurv_path ? "delta vs the seed" : "full image");
        if (g_ckpt.period_s > 0)
            fprintf(stderr, " every %.1f min", g_ckpt.period_s / 60.0);
        fprintf(stderr, "; SIGTERM/SIGINT saves and exits (twice: exit now)");
        if (g_wd_limit_s > 0) {
            if (g_wd_limit_s >= 60) fprintf(stderr, "; watchdog %.0f min", g_wd_limit_s / 60);
            else fprintf(stderr, "; watchdog %.0f s", g_wd_limit_s);
        }
        fprintf(stderr, "\n");
    }
    wd_start();                                  // (round 8; --watchdog-mins)
    if (!have_start && loadsurv_path) { start.swap(seed); have_start = true; }
    if (d_from <= d_hi && d_from <= 49 && d_hi >= 49)
        fprintf(stderr, "note: degree 49 uses the RR engine: first a table of "
                "2^25 logs is built (a minute or two; kept in %s if that took "
                "over two), then the sieve\n", rr_cache_path(49).c_str());

    // ---- devices ----
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) ndev = 1;
    int use = (want_gpus > 0 && want_gpus < ndev) ? want_gpus : ndev;
    if (want_gpus > ndev)
        fprintf(stderr, "warning: --gpus %d requested but only %d device%s "
                "present; using %d\n", want_gpus, ndev, ndev == 1 ? "" : "s",
                ndev);
    g_gpus.assign(use, GpuCtx());
    for (int g = 0; g < use; g++) g_gpus[g].dev = g;
    if (use > 1)
        fprintf(stderr, "using %d GPUs (chunks dealt out, results merged by "
                "minimum)\n", use);

    double t0 = now_s();
    std::vector<ull> best;
    int reached = d_hi;
    if (d_from <= d_hi) {
        reached = run_gpu(r, d_from, d_hi, smax, best, have_start ? &start : nullptr,
                          &rs);
    } else {
        best.swap(start);                        // a finished checkpoint
    }
    start.clear(); start.shrink_to_fit();
    // resumed: also the time of all runs (the degrees' times, as on their
    // done lines, summed), when every run recorded its time
    if (resumed && g_ckpt.run_known)
        fprintf(stderr, "total sieve time: %.2fs in this run, %.2fs in all\n",
                now_s() - t0, g_ckpt.run_before_s);
    else
        fprintf(stderr, "total sieve time: %.2fs\n", now_s() - t0);
    if (reached < d_hi)
        fprintf(stderr, "WARNING: stopped after degree %d of %d; the outputs "
                "below cover degrees through %d only, and the checkpoint (if "
                "any) is kept.  Exit status 2.\n", reached, d_hi, reached);

    char path[1100];
    u64 nsurv = 0;
    for (u64 s = 1; s <= smax; s++) nsurv += best[s] == ~0ULL;
    if (write_survivors) {
        snprintf(path, sizeof path, "%s.survivors.txt", prefix);
        FILE *fs = fopen(path, "w");
        if (!fs) { perror(path); return 1; }
        for (u64 s = 1; s <= smax; s++)
            if (best[s] == ~0ULL) fprintf(fs, "%" PRIu64 "\n", s);
        if (fclose(fs) != 0) { perror(path); return 1; }
        fprintf(stderr, "%" PRIu64 " survivors (no factor of degree <= %d) -> "
                "%s\n", nsurv, reached, path);
    } else {
        fprintf(stderr, "%" PRIu64 " survivors (no factor of degree <= %d) "
                "[survivors file suppressed via --no-survivors]\n", nsurv, reached);
    }
    if (write_cert) {
        // One line per s that got its factor in this run, "s d p<hexmask>",
        // ascending in s.  Survivors (~0) and the seed's s with a lower-degree
        // factor (0) are skipped, so concatenating the certs of consecutive
        // runs (or sorting them with `sort -n`) gives a complete record.
        snprintf(path, sizeof path, "%s.found.cert", prefix);
        FILE *fc = fopen(path, "w");
        if (!fc) { perror(path); return 1; }
        u64 ncert = 0;
        for (u64 s = 1; s <= smax; s++) {
            const ull b = best[s];
            if (b == ~0ULL || b == 0ULL) continue;
            fprintf(fc, "%" PRIu64 " %u p%" PRIx64 "\n", s,
                    (unsigned)key_degree(b), key_mask(b));
            ncert++;
        }
        if (fclose(fc) != 0) { perror(path); return 1; }
        fprintf(stderr, "%" PRIu64 " factors -> %s\n", ncert, path);
    }
    if (reached < d_hi) return 2;
    // finished: drop the checkpoint so it can't be resumed by mistake later
    if (g_ckpt.enabled && g_ckpt.path) ckpt_remove_all(g_ckpt.path);
    return 0;
}

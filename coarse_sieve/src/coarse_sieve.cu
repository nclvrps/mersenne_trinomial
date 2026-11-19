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
// best[s] with atomicMin on the packed key (d << 48) | mask (d <= 47; see
// key_pack for d = 48..55), which
// implements exactly "smallest degree, then lexicographically least
// mask" (equal-degree masks have equal hex length, so lexicographic
// order on the printed string is numeric order on the mask).
//
// Output: <prefix>.survivors.txt  s values with no factor of degree <=
//                                 depth (the deep-search worklist)
//         <prefix>.found.bin      (optional, --found) little-endian u64
//                                 packed keys for s = 1..smax, one per
//                                 s, UINT64_MAX where no factor found;
//                                 d <= 47: mask = key & 0xFFFFFFFFFFFF,
//                                 d = key >> 48; if key >> 60 == 0xF:
//                                 d = 48 + ((key >> 56) & 0xF),
//                                 mask = key & 0xFFFFFFFFFFFFFF
//
// Build:  nvcc -O3 -arch=native -I../common -o coarse_sieve coarse_sieve.cu
//         (narrow build: u32 field, fast, depths d <= 32)
// Wide:   nvcc -O3 -arch=native -DWIDE_FIELD -I../common
//              -o coarse_sieve_wide coarse_sieve.cu    (needs gf2_dlog.h)
//         (u64 field via gf2_wide_field.h, depths d <= 63; slower per op,
//          and needed for any d >= 33 since field elements and log values
//          no longer fit in 32 bits.  Use the narrow binary for d <= 32.)
// Run:    ./coarse_sieve <r> <depth> <out_prefix>   (smax defaults to floor(r/2))
//                        [--found] [--no-survivors]
//                        [--load <file.found.bin>] [--min-depth <D>]
//                        [--load-survivors <file.survivors.txt>]
//                        [--checkpoint <file>] [--checkpoint-mins <N>]
//                        [--resume <file>] [--fresh] [--no-checkpoint]
//                        [--allow-degree-gap | --seed-covers-49] [--no-tune]
//                        [--selftest] [--selftest-lowmem [nbuckets]]
//                        [--selftest-logs [logs|tower|ph|rr]]
// Extend: --load seeds best[] from a prior run's .found.bin and --min-depth
//         starts the sieve at degree D, so you compute only the new
//         degrees.  Recommended for d>=33: run the FAST narrow binary to
//         depth 32 with --found, then run the wide binary for just d=33:
//           ./coarse_sieve      <r> 32 <smax> out32 --found
//           ./coarse_sieve_wide <r> 33 <smax> out33 --found
//                               --load out32.found.bin --min-depth 33
//         This avoids re-doing d<=32 in the slow u64 build (a degree-d
//         factor only ever improves an s with no smaller-degree factor, so
//         seeding from the depth-32 result and adding d=33 is exact).
// Checkpoint/resume (automatic since round 5): every run saves its progress
//         to <out_prefix>.ckpt every 15 minutes and on SIGTERM/SIGINT (a
//         second Ctrl-C exits at once), and running the same command again
//         continues from that file -- after checking that it was written by
//         the same run (r, smax, depth, --min-depth and the contents of the
//         --load/--load-survivors file); a different run refuses to start
//         rather than mix the two.  --fresh sets an old checkpoint aside
//         (*.old) and starts over; --no-checkpoint turns it all off;
//         --checkpoint <file> / --checkpoint-mins <N> (0 = on signals only)
//         change the file and interval; a <file>.lock keeps a second run
//         off the same checkpoint.  A finished run deletes its checkpoint.
//         The rest of this paragraph describes the mechanism, which is
//         unchanged.  In low-memory mode
//         the save happens at bucket boundaries (mid-degree), so at most one
//         bucket is redone; on SIGTERM the in-flight bucket is abandoned
//         within ~100ms so the ~30s reclaim budget is met.  In logs mode
//         saves happen at chunk boundaries, and a stop waits for the
//         in-flight chunk (seconds at most).  Explicit resume:
//         --resume <file> (which keeps checkpointing to the same file):
//           ./coarse_sieve_wide <r> 34 <smax> out34 --found
//               --load out33.found.bin --min-depth 34
//           # on reclaim, the same command again -- or explicitly:
//           ./coarse_sieve_wide <r> 34 <smax> out34 --found --resume out34.ckpt
//         Low-memory runs print a per-bucket progress bar ('=' for buckets
//         already done on resume, '.' as each new bucket completes).
// Env:    COARSE_FORCE_BUCKETS=N forces the bucketed path with N buckets for
//         every degree (manual override / testing).
//         COARSE_METHOD=auto|dense|buckets|logs|tower|ph|rr overrides the
//         per-degree method choice (A/B timing; "logs" = tower for even d,
//         Pohlig-Hellman or RR (tuned; RR at d=49) for odd d; "rr" = RR for
//         odd d, tower for even d).  COARSE_LOGS_MAX_MB=n caps the VRAM
//         used by logs-path tables (default: free memory minus the reserve
//         and min(free/8, 512 MB) of slack; less only means more giant
//         steps for the PH parts that do not fit whole).
//         COARSE_LOGS_KERNEL=batched|r2 selects the logs-path kernel: the
//         round-3 batched kernel (default) or the round-2 one (A/B timing,
//         fallback).  Checkpoints do not depend on it: either kernel can
//         resume the other's.
//         COARSE_TOWER_SUBN=0 uses the round-3 tower (one table of all logs
//         in K) even where the subfield tower applies (A/B timing).
//         COARSE_IRREG=r4 makes the batched kernel handle the classes with
//         gcd(u, 2^d-1) > 1 the round-4 way (a modular inverse by extended
//         Euclid per class) instead of in the batch inversion (A/B timing;
//         identical results).
//         Tests only: COARSE_TOWER_SUBN=n forces the inner subfield
//         GF(2^n) (any n | m, 2 <= n < m); COARSE_LOGS_CHUNK_LOG2=k caps
//         logs-path chunks at 2^k
//         values (a resume must use the same k); COARSE_SELFTEST_LOGS_DMAX
//         limits the engine checks of --selftest-logs, COARSE_SELFTEST_RR_DMAX
//         its RR checks (default 47; 49 adds the d=49 table and engine, a
//         few minutes more) and COARSE_SELFTEST_RR_BSGS_D the
//         degree of its RR check with a forced many-giant-step factor-base
//         plan (default 41); COARSE_RR_FB_BUDGET_MB=m forces such plans
//         (factor-base tables of at most m MB); COARSE_TUNE_PICK=i takes
//         tuning candidate i;
//         COARSE_TEST_FAULT_EVERY=n corrupts the log of every class with
//         u = 1 mod n (the run must then stop with "discrete-log
//         consistency errors"); COARSE_TEST_STOP_DEGREE=n acts as if
//         degree n could not be sieved (the run must then write outputs
//         for the degrees below n only, say so, and exit with status 2).
// Degrees: narrow build supports d <= 32 (elements fit u32, modulus u64);
//         the -DWIDE_FIELD build supports d <= 55: the packed best[] key
//         holds d <= 55 (key_pack) and the logs path's fast arithmetic
//         d <= 55 (main() enforces both limits).  Degrees 48..55 run only
//         in the logs path; d = 49 (2^49 - 1 = 127 * 4432676798593) only
//         with the RR engine (round 5; before that it could not be sieved
//         at all).  A run starting above 49 must say whether its seed file
//         covers degree 49: --seed-covers-49 (it came from a run through 49)
//         or --allow-degree-gap (it did not: a factor found at 50..55 is
//         then the LEAST-degree factor of its trinomial only where there is
//         no degree-49 factor).  Low-memory mode fits any degree <= 47 into
//         any card by raising the bucket count; see the cost note in
//         LOW-MEMORY MODE.
// Outputs: <prefix>.survivors.txt is written by default (--no-survivors
//         suppresses it); it is just the list of s with no factor of
//         degree <= depth, which is exactly the set of s missing from
//         <prefix>.found.bin, so it can be reconstructed from --found.
// Sizing: the dense per-degree tables need 8*2^d bytes at degree d in the
//         narrow build and 16*2^d in the wide build (at d=30: 8.6 GB and
//         17.2 GB), plus 8*(smax+1) bytes for
//         best[] (545 MB at smax = 68,139,920).  The program checks
//         free device memory before each degree and, once the dense
//         tables no longer fit, automatically switches -- in the wide
//         build -- to the LOGS path (see "LOGS MODE" below: tables of only
//         ~2^(d/2) entries, any VRAM), or if even those do not fit, to the
//         bucketed low-memory mode (see "LOW-MEMORY MODE").
// Selftest: --selftest runs r=136279841 depth=20 smax=999999 entirely
//         on the GPU and compares every best[] entry against the CPU
//         reference computed in-process.  --selftest-lowmem does the
//         same but forces the bucketed low-memory path (with the given
//         number of buckets, default 4) for every degree regardless of
//         whether the dense tables would have fit, so the low-memory
//         kernels get exercised and checked even at small, fast depths.
//         --selftest-logs [logs|tower|ph|rr] (wide build) forces the logs
//         path instead, and additionally checks log(g^u) == u on the device
//         for every u < min(2^d - 1, 2^24) at each degree; then, beyond the
//         sieve's depth, it checks the chosen engine(s) on 2^20 exponents at
//         every d = 21..55, the batched kernel end to end (log(g^u) == u for
//         every class it enumerates in a stretch of u, and its class count
//         against the necklace test's, with the round-5 irregular-class
//         handling checked against the round-4 one), and the necklace test
//         against the all-rotations reference at every d <= 55
//         (gf2_dlog_selftest.cpp is the CPU counterpart).  The RR engine
//         is checked the same way (2^20 exponents and the batched kernel)
//         at every odd d = 21..47, once (d = 41) with its factor-base logs
//         forced through a many-giant-step plan, the d = 49 kind; "logs"
//         (the default) checks all engines, "rr" sieves with RR at odd d
//         and checks only RR, "tower" / "ph" only those.
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
// automatically (any count, not just powers of two); --selftest-lowmem lets
// you force a specific value.
//
// COST NOTE.  Total work is ~nbuckets * 2^d sweep steps, and nbuckets itself
// grows like 2^d / VRAM, so each degree costs ~4x the previous one -- every
// instruction in the sweep's inner loop is multiplied by nbuckets.  The sweep
// therefore walks g^i by LFSR stepping (g = x: one shift + one conditional
// XOR per element, seeded once per run of ANTILOG_RUN_MAX elements).  It
// used to recompute each g^i from scratch with ~d/2 bit-serial field
// multiplies, which was ~1000x more expensive and accounted for essentially
// all of the measured d >= 33 runtime (e.g. 346 s per bucket at d=36 on an
// RTX 2070 SUPER = 5.0 ns per element per bucket).
//
// LOGS MODE (wide build; the default once dense tables do not fit)
// ----------------------------------------------------------------
// Buckets exist only because the method above materializes the full 2^d-
// entry log table.  The sieve needs much less: ONE Zech logarithm
// Z(u) = log(g^u + 1) per cyclotomic class.  The logs path computes that
// log directly, from tables of ~2^(d/2) entries (gf2_dlog.h):
//   tower (even d = 2m):  GF(2^d) = K(w), K = GF(2^m); a + b*w = b*(c + w),
//          c = a/b, so log = (2^m+1) log_K(b) + T[log_h(c)] with a 2^m-entry
//          table T (64 MB at d=46, 1 GB at d=54) and a GF(2)-linear split
//          (byte tables).  The two logs in K come from the same trick one
//          level down whenever m is composite (round 4): K is an extension
//          of a subfield K_n, n | m, and a log in K is one read of a table
//          of 2^n logs in K_n plus one of a table indexed by the projective
//          point Y/lambda (2^n entries for even m; 16512 / 1082400 /
//          262656 for m = 21 / 25 / 27).  Those tables are small enough to
//          sit in shared memory or cache, so a tower log costs one DRAM read
//          (T) instead of three.  For prime m (d = 34, 38, 46) K's logs come
//          from a 2^m-entry table LK as in round 3.
//   Pohlig-Hellman (odd d): M = 2^d - 1 = P_1 * ... * P_k (k <= 4) split
//          into coprime parts; x^E_i lands in the order-P_i subgroup, a hash
//          table gives log mod P_i, CRT combines.  A planner picks the split
//          and the exponents E_i (few set bits in a low window) for the least
//          estimated time per class, charging for table memory; a part too
//          large for the memory budget (d=31: M itself is prime; d=37:
//          616318177; d=41: 164511353 on an 8 GB card) uses baby-step/
//          giant-step.
//   RR, rational reconstruction (odd d, round 5): the extended Euclidean
//          algorithm on (q, y), stopped half way, writes y = r/t with r and
//          t both of degree <= d/2, so log y = L[r] - L[t] with L the logs
//          of ALL polynomials of degree <= d/2 (2^(d/2+1) entries: 256 MB
//          at d=49, 2 GB at d=55): one half-gcd and two reads per class, no
//          smoothness test, no failure.  L comes from the logs of the
//          irreducible polynomials of degree <= d/2 (1.47M at d=49) by
//          multiplicativity; those come from Pohlig-Hellman -- for d = 49,
//          whose M = 127 * 4432676798593 has no table-sized parts, with up
//          to ~8,000-33,000 giant steps per log (by card size), affordable
//          for 1.47M elements (~2.4e10 giant steps on an 8 GB card: a
//          minute or two, estimated) where it was not for 1.15e13 classes.
//          So degree 49 can now be sieved.  A table that took over two
//          minutes to build is cached in coarse_rr_d<d>.tbl (it does not
//          depend on r).  At the other odd degrees the tuner picks PH or RR,
//          whichever is faster on the card.
// Tuning (round 5): for d >= 40 each degree starts with a few seconds of
//         timing on GPU 0 -- the tower variants, PH vs RR, and the round-4
//         irregular-class path as a reference -- and runs with the fastest
//         (a line "[tune d=..: ... ns/class ... -> choice]" shows the
//         numbers).  --no-tune, COARSE_TUNE=0 / COARSE_TUNE_MIN=d.
// Classes: each class of degree d has exactly one representative u that is
// a Lyndon word (aperiodic and least among its bit rotations, hence odd and
// below 2^(d-1)); any representative yields the same factor and progression
// (see k_logs_sweep).  The batched kernel (round 3, the default) enumerates
// exactly those u with the FKM algorithm and handles them 32 per thread at
// a time: g^u and the PH powers x^E_i in bit-sliced form (32 elements per
// u32 per coefficient), and one modular inversion per batch (Montgomery's
// trick).  The round-2 kernel (COARSE_LOGS_KERNEL=r2) tests every odd u with
// the necklace test, steps g^u by LFSR and handles classes one at a time.
// The progression start needs j^-1 mod M, computed as j^(lambda(M)-1)
// (Carmichael exponent, much shorter than phi(M)); since round 5 the
// classes with gcd(u, M) > 1 join that batch inversion too (see "Irregular
// classes, round 5").  Work is O(2^d) per
// degree instead of O(4^d / VRAM); memory is
// best[] + the tables.  The pass is cut into chunks of u-values (kernel
// launches) for checkpointing; checkpoints record which method wrote them,
// and resuming re-enters the same method.  Every computed log is range-
// checked, 1 in 256 is re-verified (g^v == x), and any inconsistency stops
// the run at the end of that round of chunks, before any output or
// checkpoint could include it.

#if defined(WIDE_FIELD)
  #include "gf2_wide_field.h"
  #include "gf2_wide_setup.h"
#else
  #include "gf2_small_field.h"
  #include "gf2_field_setup.h"
#endif
#include <cuda_runtime.h>
#if defined(WIDE_FIELD)
  #include "gf2_dlog.h"
#endif
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

// best[] keys: "smallest degree, then numerically least mask" as one u64
// compared by atomicMin.  The mask has d+1 bits (all coefficients of the
// minimal polynomial).
//   d <= 47:       (d << 48) | mask   -- the original format, so .found.bin
//                                        files and checkpoints from earlier
//                                        rounds stay valid
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

// (a*b) mod M for the Mersenne-form modulus M = 2^d - 1 (d <= 63), with
// a, b < M.  Since 2^d == 1 (mod M) the 2d-bit product folds at bit d:
// p mod M == (p & M) + (p >> d).  About 20 instructions instead of the
// ~36-iteration double-and-add loop of mulmod_u64, which ran once per L-slice
// slot in k_resolve_slice.  (Used only where the modulus is M itself; the
// progression moduli n = M/gg keep MULMOD.)
GF2_HD u64 mulmod_mersenne(u64 a, u64 b, int d, u64 M) {
#if defined(__CUDA_ARCH__)
    u64 lo = a * b, hi = __umul64hi(a, b);
#else   // portable 64x64 -> 128 (host/emulation path; no __int128 needed)
    u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
    u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
    u64 lo = (mid << 32) | (u32)p00;
    u64 hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
    // p < 2^(2d), so p >> d < 2^d fits in 64 bits; for d <= 32 hi == 0
    u64 t = (lo & M) + ((lo >> d) | (d < 64 ? (hi << (64 - d)) : 0));
    t = (t & M) + (t >> d);
    if (t >= M) t -= M;
    return t;
}

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

// Elements per thread in the antilog sweeps (k_antilog, k_build_lslice).
// Each thread computes ONE seed g^i0 with d_antilog() (~d/2 field
// multiplies), then walks g^(i+1) = x * g^i with gf_mulx -- one shift and
// one conditional XOR -- since g = x for the primitive modulus.  The
// previous kernels recomputed every g^i from scratch (~d/2 bit-serial
// multiplies, ~20k GPU instructions per element at d=36), and the low-memory
// path did that for the whole 2^d domain once PER BUCKET, which was ~all of
// its runtime.
#define ANTILOG_RUN_MAX 16384
static u64 antilog_run_len(u64 M64) {
    u64 run = 64;   // keep >= ~2^20 runs for parallelism, amortize the seed
    while (run < ANTILOG_RUN_MAX && M64 / run > ((u64)1 << 20)) run <<= 1;
    return run;
}

// ---------------------------------------------------------------------------
// kernels
// g^idx computed on the fly from the (tiny, size-d) gpow2 chain, i.e.
// gpow2[t] = g^(2^t): one field multiply per set bit of idx.  Used only for
// run seeds and rare per-class lookups -- never per element of a sweep.
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

// antilog table: A[i] = g^i.  Each thread owns runs of `run` consecutive
// exponents: one d_antilog() seed per run, then gf_mulx steps (see
// ANTILOG_RUN_MAX above).
__global__ void k_antilog(fw *A, u64 M64, u64 q, int d, const fw *gpow2,
                          u64 run) {
    const u64 nruns = (M64 + run - 1) / run;
    u64 rr = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; rr < nruns; rr += stride) {
        u64 i = rr * run;
        u64 iend = i + run; if (iend > M64) iend = M64;
        fw a = d_antilog(i, q, d, gpow2);
        for (; i < iend; i++) {
            A[i] = a;
            a = gf_mulx(a, q, d);
        }
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
        // only the minimal element of each cyclotomic coset (cheap test
        // first: it rejects (d-1)/d of all j after ~2 rotations)
        fw jj = j;
        bool minimal = true;
        for (int i = 1; i < d; i++) {
            jj = rotl_d(jj, d, M);
            if (jj < j) { minimal = false; break; }
        }
        if (!minimal) continue;
        // skip proper-subfield elements
        bool skip = false;
        for (int k = 0; k < nsub; k++)
            if (j64 % subq[k] == 0) { skip = true; break; }
        if (skip) continue;

        u64 u = mulmod_mersenne(j64, rmod, d, M64);
        if (u == 0) continue;                     // beta^r = 1: no solutions
        fw v = L[A[u] ^ 1u];                       // Zech logarithm

        u64 gg = gcd_u64(j64, M64);
        if ((u64)v % gg != 0) continue;
        u64 n = M64 / gg;
        u64 s0 = MULMOD(inv_mod(j64 / gg, n), (u64)v / gg, n);

        u64 s = (s0 == 0 ? n : s0);
        if (s > smax) continue;
        u64 count = (smax - s) / n + 1;
        if (count <= INLINE_MARKS && !progression_open(best, s, n, smax, d))
            continue;                             // nothing left to improve

        // minimal polynomial only for in-range, still-improvable hits (it
        // was previously computed for every class, before the range test)
        u64 mask = minpoly_mask(j, d, M, q, A, P);
        // mask == 0 would mean non-GF(2) coefficients: impossible if the
        // tables are right; validated exhaustively on CPU.
        u64 packed = key_pack(d, mask);

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
// This sweeps the WHOLE 2^d domain once per bucket, so its per-element cost
// is multiplied by nbuckets: hence run-based LFSR stepping (one d_antilog
// seed per run, then gf_mulx), never a per-element d_antilog.
__global__ void k_build_lslice(int d, u64 M64, u64 q, const fw *gpow2,
                               u64 lo, u64 hi, fw *lslice, u64 run) {
    const u64 nruns = (M64 + run - 1) / run;
    const u64 span = hi - lo;
    u64 rr = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (; rr < nruns; rr += stride) {
        u64 i = rr * run;
        u64 iend = i + run; if (iend > M64) iend = M64;
        fw a = d_antilog(i, q, d, gpow2);
        for (; i < iend; i++) {
            u64 off = (u64)a - lo;              // wraps (huge) when a < lo
            if (off < span) lslice[off] = (fw)i;
            a = gf_mulx(a, q, d);
        }
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
    unsigned long long my_surv = 0;           // one atomic per thread, not
                                              // one per class (2^d/d of them)
    for (; t < span; t += stride) {
        fw vu = lslice[t];
        if (vu == EMPTY) continue;            // element 0 has no discrete log
        u64 j64 = mulmod_mersenne((u64)vu, rmodinv, d, M64);
        if (j64 == 0) continue;               // u == 0 <-> j == 0, not a class

        // coset-minimality first: it rejects (d-1)/d of all slots after ~2
        // rotations on average, before the 64-bit modulo of the subfield test
        fw j = (fw)j64;
        fw jj = j;
        bool minimal = true;
        for (int i = 1; i < d; i++) {
            jj = rotl_d(jj, d, M);
            if (jj < j) { minimal = false; break; }
        }
        if (!minimal) continue;
        bool skip = false;
        for (int k = 0; k < nsub; k++)
            if (j64 % subq[k] == 0) { skip = true; break; }
        if (skip) continue;
        my_surv++;                            // counted where Phase A used to

        fw v = lslice[t ^ 1ULL];              // log of x = (lo+t)^1
        u64 gg = gcd_u64(j64, M64);
        if ((u64)v % gg != 0) continue;
        u64 n = M64 / gg;
        u64 s0 = MULMOD(inv_mod(j64 / gg, n), (u64)v / gg, n);

        u64 s = (s0 == 0 ? n : s0);
        if (s > smax) continue;
        u64 count = (smax - s) / n + 1;
        if (count <= INLINE_MARKS && !progression_open(best, s, n, smax, d))
            continue;                         // nothing left to improve

        fw seed = d_antilog(j64, q, d, gpow2);            // = A[j]
        u64 mask = d_minpoly_seeded(seed, d, q, P);
        u64 packed = key_pack(d, mask);

        if (count <= INLINE_MARKS) {
            for (; s <= smax; s += n) atomicMin(&best[s], (ull)packed);
        } else {
            unsigned idx = atomicAdd(nheavy, 1u);
            if (idx < HEAVY_CAP) heavy[idx] = {s, n, packed};
            else atomicAdd(overflow, 1u);
        }
    }
    if (my_surv) atomicAdd(nsurv, my_surv);
}

// logs-path engines (also the checkpoint method tags)
enum { LM_NONE = 0, LM_TOWER = 1, LM_PH = 2, LM_RR = 3 };

#if defined(WIDE_FIELD)
// ---------------------------------------------------------------------------
// LOGS PATH (wide build): no 2^d tables and no buckets.  See "LOGS MODE" in
// the header comment and gf2_dlog.h.  One pass over u in [0, 2^(d-1)):
//   * every cyclotomic class has exactly one coset-minimal u (minimal under
//     u -> 2u mod M, i.e. bit rotation), and that u is odd and < 2^(d-1);
//   * any representative gives the same factor and progression: with
//     u' = 2^i u and Z(2u) = 2 Z(u), the congruence j's == Z(u') reduces to
//     j s == Z(u) because 2 is invertible mod M;  gcd(j, M) = gcd(u, M) and
//     u is in a subfield iff j is, since j = u * r^-1 with r a unit;
//   * g^u comes from the LFSR (g = x), and Z(u) = log(g^u + 1) from the
//     tower or Pohlig-Hellman engine -- one discrete log per class.
// Each thread walks runs of consecutive u; within a run it advances to its
// next class representative before processing it, so the warp processes
// ~32 classes together instead of idling 30 lanes behind each survivor.
#define LOGS_MAXPF 16
struct LogsParams {
    int d, method;                 // method: LM_TOWER or LM_PH
    u64 M, q, qlow, rmodinv, smax;
    u64 lam_m1;                    // lambda(M) - 1: j^-1 = j^(lambda-1) mod M
    u64 fault;                     // test hook: corrupt the log of u = 1 mod fault
    int check;                     // selftest: compute log(g^u), count != u
    int irreg_r4;                  // COARSE_IRREG=r4: round-4 irregular path
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

GF2_HD u64 logs_mulx2(u64 e, u64 q, int d) {
    return gf_mulx(gf_mulx(e, q, d), q, d);
}

// Rarely executed code stays out of line (round 5): the logs kernels are
// 140-180 KB of SASS while Turing has a 16 KiB L0 and a 46 KiB L1
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

// gcd(u, M) == 1?  (gcd(u, M) = gcd(j, M) as r is a unit)
GF2_HD bool logs_regular(u64 u, const LogsParams &L) {
    u64 qq;
    for (int i = 0; i < L.npf; i++)
        if (odd_div_test(u, L.pf[i], &qq)) return false;
    return true;
}

// a class with gg = gcd(u, M) > 1: j s == v (mod M) needs gg | v, and then
// has modulus n = M / gg
__device__ __forceinline__ void logs_irregular(u64 u, u64 j, u64 v,
                                               const LogsParams &L,
                                               const fw *gpow2, ull *best,
                                               Heavy *heavy, unsigned *nheavy,
                                               unsigned *overflow, fw *P,
                                               unsigned long long *err) {
    u64 jg = j, vg = v, n = L.M, t = u, qq;
    for (int i = 0; i < L.npf; i++) {
        for (int k = 0; k < L.pk[i]; k++) {
            if (!odd_div_test(t, L.pf[i], &qq)) break;
            t = qq;
            if (!odd_div_test(vg, L.pf[i], &qq)) return;   // gg does not divide v
            vg = qq;
            odd_div_test(jg, L.pf[i], &jg);
            odd_div_test(n, L.pf[i], &n);
        }
    }
    u64 s0 = MULMOD(inv_mod(jg, n), vg, n);
    logs_mark(s0 == 0 ? n : s0, n, j, L, gpow2, best, heavy, nheavy, overflow,
              P, err);
}

// ---------------------------------------------------------------------------
// Irregular classes, round 5.  logs_irregular above spends, per class that
// reaches its last line, an extended Euclid of ~25-29 iterations (each a
// call of the software 64-bit division, ~80 instructions in the SASS) plus a
// ~40-step double-and-add.  At even d some lane of nearly every warp step
// has such a class (40-60% of the classes are irregular, ~15% need the
// inverse), so the whole warp waits through it: ~3000-3800 instructions per
// warp step against ~1200-1500 for everything else (nvcc 13.3 SASS, sm_75).
// Instead the batched kernel lets those classes join the batch inversion:
//   * gg, its 2-adic inverse ginv and so the EXACT quotients n = M*ginv,
//     jg = j*ginv, vg = v*ginv (mod 2^64) come from the primes that divide
//     u -- one multiply-and-compare per prime;  gg | v  <=>  vg < n;
//   * J = jg + t*n with the least t >= 0 that makes J prime to the primes
//     of gg not dividing n (t = 0 most of the time) is a unit mod M with
//     J == jg (mod n), so J^-1 mod M reduces to jg^-1 mod n (n | M): J
//     enters the batch product like a regular j;
//   * s0 = (vg * J^-1 mod M) mod n, with x mod n = x - n*floor(x*gg/M) and
//     floor(./M) a shift and a compare (mod_divisor_m).
// About 100-300 instructions per warp step remain.  COARSE_IRREG=r4 selects
// the round-4 path (A/B timing; results are identical).
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
// 2: no lift found within 64 tries (never seen; caller takes the old path)
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

// selftest: the lift against the round-4 formulas (extended Euclid) on
// pseudo-random classes with gcd(u, M) > 1, half of them with gg | v.
// bad[0] counts mismatches, bad[1] classes that joined a batch, bad[2]
// classes skipped (gg does not divide v), bad[3] lifts with t > 0.
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
        // reference: logs_irregular's arithmetic
        u64 jg = j, vg = v, n = M, t = u, qq;
        bool ref_skip = false;
        for (int a = 0; a < L.npf && !ref_skip; a++)
            for (int k = 0; k < L.pk[a]; k++) {
                if (!odd_div_test(t, L.pf[a], &qq)) break;
                t = qq;
                if (!odd_div_test(vg, L.pf[a], &qq)) { ref_skip = true; break; }
                vg = qq;
                odd_div_test(jg, L.pf[a], &jg);
                odd_div_test(n, L.pf[a], &n);
            }
        u64 J = 0, wg = 0, gg = 0;
        const int st = logs_irreg_lift(u, j, v, mask, L, &J, &wg, &gg);
        if (ref_skip) { ns++; if (st != 1) nb++; continue; }
        if (st != 0) { nb++; continue; }
        nj++;
        const u64 ref = MULMOD(inv_mod(jg, n), vg, n);
        const u64 nn = M * inv_2adic(gg);
        const u64 Jinv = gf2f_powmod_m(J, L.lam_m1, d, M);
        const u64 s = mod_divisor_m(gf2f_mulmod_m(wg, Jinv, d, M), nn, gg, d, M);
        if (J >= M || gf2f_mulmod_m(J, Jinv, d, M) != 1 % M) nb++;   // a unit
        else if (nn != n || wg != vg || s != ref) nb++;
        else if (gg * mod_divisor_m(J, nn, gg, d, M) != j) nb++;    // j back
        if (J != jg) nt++;
    }
    if (nb) atomicAdd(&bad[0], nb);
    if (nj) atomicAdd(&bad[1], nj);
    if (ns) atomicAdd(&bad[2], ns);
    if (nt) atomicAdd(&bad[3], nt);
}

// round-2 kernel: everything after "u is a class representative, g^u = e"
__device__ __forceinline__ void logs_process(u64 u, u64 e, const LogsParams &L,
                                             const fw *gpow2, ull *best,
                                             Heavy *heavy, unsigned *nheavy,
                                             unsigned *overflow, fw *P,
                                             unsigned long long *err,
                                             bool verify) {
    const int d = L.d;
    const u64 M = L.M;
    const u64 x = e ^ 1;                        // g^u + 1 (nonzero: u != 0)
    u64 v = logs_dlog(x, L);                    // Zech log Z(u)
    if (L.fault && u % L.fault == 1) v ^= 1;    // fault injection (tests)
    if (v >= M) { (*err)++; return; }
    if (verify && gf2f_antilog(v, L.qlow, d, gpow2) != x) { (*err)++; return; }
    u64 j = gf2f_mulmod_m(u, L.rmodinv, d, M);
    if (!logs_regular(u, L)) {
        logs_irregular(u, j, v, L, gpow2, best, heavy, nheavy, overflow, P, err);
        return;
    }
    u64 s0 = gf2f_mulmod_m(v, gf2f_powmod_m(j, L.lam_m1, d, M), d, M);
    logs_mark(s0 == 0 ? M : s0, M, j, L, gpow2, best, heavy, nheavy, overflow,
              P, err);
}

// sweep u in [ulo, uhi) (ulo, run even), runs of `run` consecutive u
__global__ void k_logs_sweep(LogsParams L, const fw *gpow2, u64 ulo, u64 uhi,
                             u64 run, ull *best, Heavy *heavy,
                             unsigned *nheavy, unsigned *overflow,
                             unsigned long long *nsurv,
                             unsigned long long *nerr) {
    const u64 nruns = (uhi - ulo + run - 1) / run;
    u64 rr = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    fw P[64];
    unsigned long long my_surv = 0, my_err = 0;
    for (; rr < nruns; rr += stride) {
        u64 u = (ulo + rr * run) | 1;            // odd u only
        u64 uend = ulo + rr * run + run; if (uend > uhi) uend = uhi;
        if (u >= uend) continue;
        u64 e = gf2f_antilog(u, L.qlow, L.d, gpow2);
        for (;;) {
            while (u < uend && !logs_is_rep(u, L)) {
                u += 2;
                e = logs_mulx2(e, L.q, L.d);
            }
            if (u >= uend) break;
            my_surv++;
            logs_process(u, e, L, gpow2, best, heavy, nheavy, overflow, P,
                         &my_err, (my_surv & 255) == 1);   // spot-verify logs
            u += 2;
            e = logs_mulx2(e, L.q, L.d);
        }
    }
    if (my_surv) atomicAdd(nsurv, my_surv);
    if (my_err) atomicAdd(nerr, my_err);
}

// ---------------------------------------------------------------------------
// LOGS PATH, round 3: the batched kernel.  Each thread enumerates the Lyndon
// words of its runs with FKM -- exactly the class representatives, nothing
// else to test -- and handles them 32 at a time (bit-sliced, see gf2_dlog.h):
//   * g^u for all 32 by bs_pow_g (squarings and plane shifts only), then
//     x = g^u + 1;
//   * Pohlig-Hellman (odd d): x^(E_i) bit-sliced for each part, back to
//     scalars, one table lookup per class and part, CRT;
//     tower (even d): back to scalars, tower_log per class;
//   * progression starts v * j^-1 for all 32 with ONE modular exponentiation
//     (Montgomery's trick: invert the product of the j's, then peel off each
//     inverse with two multiplies).
// Against the round-2 kernel this removes per class ~12 necklace tests (and
// the warp idling between representatives), ~19 scalar multiplies and 27
// squarings of a PH log, and ~45 of the ~50 multiplies of j^-1.
#define LB 32

template <int D, int METHOD>
__device__ __forceinline__ void logs_batch(const LogsParams &L,
                                           const TowerDev &tw,
                                           const fw *gpow2, u64 (&ub)[LB],
                                           u64 (&vb)[LB], u64 (&tb)[LB],
                                           u64 (&xb)[LB], int n,
                                           bool verify, ull *best,
                                           Heavy *heavy, unsigned *nheavy,
                                           unsigned *overflow, fw *P,
                                           unsigned long long *err) {
    const u64 M = L.M;
    for (int k = n; k < LB; k++) ub[k] = 1;     // padding; results unused
    u32 X[D];
    {
        u32 U[D];
        bs_load<D>(ub, U);
        bs_pow_g<D>(U, X);                      // g^u
    }
    if (!L.check) X[0] = ~X[0];                 // x = g^u + 1
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
        for (int k = 0; k < n; k++) {
            const u64 v = METHOD == LM_TOWER ? tower_log(vb[k], M, tw)
                        : METHOD == LM_RR ? rr_log(vb[k], L.q, D, M, L.rr) : vb[k];
            if (v != ub[k]) (*err)++;
        }
        return;
    }
    // forward (round 5: fused with the tower logs): tb[k] = product of the
    // batch members before k; ub[k] <- J, the class's member (j itself for
    // a regular class); vb[k] <- v, VB_IRREG | vg, or VB_SKIP.  Each
    // iteration loads the next class's x and u first, and issues its T read
    // before the work that does not depend on it: on an in-order GPU every
    // first use of a pending load stalls the warp.
    u64 acc = 1;
    bool any = false;
    u64 xn = vb[0], un = ub[0];
    for (int k = 0; k < n; k++) {
        const u64 x = xn, u = un;
        if (k + 1 < n) { xn = vb[k + 1]; un = ub[k + 1]; }
        tb[k] = acc;
        u64 v, lb = 0, tv = 0;
        bool inK = false;
        if (METHOD == LM_TOWER) {
            u64 l;
            const u64 pk = tower_coords(x, tw);
            inK = !tower_split(pk, tw, &lb, &l);
            if (inK) lb = tower_klog_cold(pk & tw.MK, tw);   // x in K*
            else tv = tw.T[l];
        } else if (METHOD == LM_RR) {                   // x = r/t, two reads
            u64 rr_r;
            u32 rr_t;
            rr_split<D>(x, &rr_r, &rr_t);
            lb = L.rr.L[rr_r];
            tv = L.rr.L[rr_t];
        } else {
            v = x;                                      // PH: x is the log
        }
        const u64 j = gf2f_mulmod_m(u, L.rmodinv, D, M);
        const u32 mask = logs_divmask<D>(u, L);
        if (METHOD == LM_TOWER) {
            const u64 h1 = tw.MK + 2;                   // log_g(h) = 2^m + 1
            v = inK ? h1 * lb : h1 * lb + tv;
            if (v >= M) v -= M;
        } else if (METHOD == LM_RR) {
            v = lb >= tv ? lb - tv : lb + M - tv;       // log r - log t
        }
        if (L.fault && u % L.fault == 1) v ^= 1;        // fault injection (tests)
        if (k == 0 && verify && v < M && logs_verify_bad(v, u, L.qlow, D, gpow2))
            v = ~0ULL;                                  // spot check failed
        if (v >= M) { (*err)++; vb[k] = VB_SKIP; continue; }   // inconsistent log
        u64 J = j;
        if (mask) {                                     // gcd(u, M) > 1
            u64 vg = 0, gg = 0;
            const int st = L.irreg_r4 ? 2 : logs_irreg_lift(u, j, v, mask, L,
                                                            &J, &vg, &gg);
            if (st) {                                   // 1: no s at all
                if (st == 2)
                    logs_irregular(u, j, v, L, gpow2, best, heavy, nheavy,
                                   overflow, P, err);
                vb[k] = VB_SKIP;
                continue;
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
    if (!any) return;
    u64 inv = gf2f_powmod_m(acc, L.lam_m1, D, M);   // (prod J)^-1
    // backward: J^-1 = inv * tb[k], then inv <- inv * J (Montgomery)
    u64 vn = vb[n - 1], Jn = ub[n - 1], tn = tb[n - 1], gn = xb[n - 1];
    for (int k = n - 1; k >= 0; k--) {
        const u64 vv = vn, J = Jn, t = tn, gg = gn;
        if (k > 0) { vn = vb[k - 1]; Jn = ub[k - 1]; tn = tb[k - 1]; gn = xb[k - 1]; }
        if (vv == VB_SKIP) continue;
        const u64 jinv = gf2f_mulmod_m(inv, t, D, M);
        inv = gf2f_mulmod_m(inv, J, D, M);
        if (!(vv & VB_IRREG)) {
            const u64 s0 = gf2f_mulmod_m(vv, jinv, D, M);
            logs_mark(s0 == 0 ? M : s0, M, J, L, gpow2, best, heavy, nheavy,
                      overflow, P, err);
        } else {                                    // modulus n = M/gg
            const u64 nn = M * inv_2adic(gg);
            const u64 S = gf2f_mulmod_m(vv & ~VB_IRREG, jinv, D, M);
            u64 s = mod_divisor_m(S, nn, gg, D, M);  // vg * jg^-1 mod n
            if (s == 0) s = nn;
            if (s <= L.smax)                         // j = gg * (J mod n)
                logs_mark(s, nn, gg * mod_divisor_m(J, nn, gg, D, M), L,
                          gpow2, best, heavy, nheavy, overflow, P, err);
        }
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
#endif

// sweep the Lyndon words in [ulo, uhi), runs of `run` u-values per step of
// the grid-stride loop; partial batches carry over between runs.  Tower
// kernels run 512-thread blocks (one per SM, sharing one copy of the small
// tables in shared memory), PH kernels 128-thread blocks (up to ~250
// registers).
template <int D, int METHOD>
__global__ void __launch_bounds__(METHOD == LM_PH ? 128 : 512, 1)
k_logs_batch(LogsParams L, const fw *gpow2, u64 ulo, u64 uhi,
             u64 run, ull *best, Heavy *heavy,
             unsigned *nheavy, unsigned *overflow,
             unsigned long long *nsurv,
             unsigned long long *nerr) {
    TowerDev tw = L.tw;
#if defined(__CUDA_ARCH__)
    if (METHOD == LM_TOWER && tw.shm_bytes) {
        extern __shared__ __align__(16) unsigned char logs_shm[];
        tower_stage_shared(tw, logs_shm);
        __syncthreads();
    }
#endif
    const u64 nruns = (uhi - ulo + run - 1) / run;
    u64 rr = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    u64 ub[LB], vb[LB], tb[LB], xb[LB];
    for (int k = 0; k < LB; k++) xb[k] = 1;     // read ahead before written
    fw P[64];
    unsigned long long my_surv = 0, my_err = 0;
    unsigned nb = 0;
    u64 w = 0, whi = 0;
    int p = 0;
    bool have = false;
    for (;;) {
        int n = 0;
        while (n < LB) {                        // fill: next 32 Lyndon words
            if (!have) {
                if (rr >= nruns) break;
                const u64 lo = ulo + rr * run;
                whi = uhi - lo > run ? lo + run : uhi;
                rr += stride;
                w = fkm_seek(lo, D, &p);
                have = true;
            }
            if (w >= whi) { have = false; continue; }
            if (p == D) ub[n++] = w;
            w = fkm_next(w, D, &p);
        }
        if (n == 0) break;
        my_surv += n;
        logs_batch<D, METHOD>(L, tw, gpow2, ub, vb, tb, xb, n, (nb++ & 7) == 0,
                              best, heavy, nheavy, overflow, P, &my_err);
    }
    if (my_surv) atomicAdd(nsurv, my_surv);
    if (my_err) atomicAdd(nerr, my_err);
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
#endif // WIDE_FIELD

// ---------------------------------------------------------------------------
static void launch_dims(u64 work, int &blocks, int &threads) {
    threads = 256;
    u64 b = (work + threads - 1) / threads;
    blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
}

// ---------------------------------------------------------------------------
// MULTI-GPU
//
// Bucket passes are completely independent: each reads only the tiny gpow2
// chain plus its own L slice, and writes results into best[] with atomicMin.
// Since min is associative and commutative, N devices can each keep a
// PRIVATE best[] and the elementwise minimum of those arrays at the end is
// bit-identical to a single-device run.  So there is no inter-device traffic
// during compute at all -- no peer access, no NVLink requirement -- and the
// only communication is one best[] readback per device when a checkpoint or
// the final result is needed.
//
// Buckets are handed out in rounds: round k gives bucket start+k*NG+g to
// device g.  Rounds end with every device synchronized, so the completed
// set is always a contiguous prefix [start, start+(k+1)*NG) -- which is what
// the checkpoint format records and what resume needs.
struct GpuCtx {
    int dev = 0;
    ull *d_best = nullptr;
    Heavy *d_heavy = nullptr;
    unsigned *d_nheavy = nullptr, *d_overflow = nullptr;
    // per-degree scratch
    fw *dgp = nullptr; u64 *dsub = nullptr; fw *dlslice = nullptr;
    unsigned long long *d_nsurv = nullptr;
    // logs path (per-degree tables + error counter)
    unsigned long long *d_nerr = nullptr;
    u64 *d_btab = nullptr, *d_T = nullptr;
    u32 *d_LK = nullptr, *d_TP = nullptr;
    uint16_t *d_LKn = nullptr;
#if defined(WIDE_FIELD)
    PHSlot *d_slots[4] = {nullptr, nullptr, nullptr, nullptr};
    u64 *d_rrL = nullptr;                // RR: logs of all small polynomials
#endif
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
    // run identity recorded in v3 checkpoints (round 5): what an automatic
    // resume must match besides r, smax and the target depth
    int      min_depth = 2;
    bool     gap49 = false;
};
static CkptCtx g_ckpt;

extern "C" void ckpt_signal_handler(int) { g_ckpt.stop = 1; }

static const char CKPT_MAGIC[8] = {'G','F','2','S','V','C','K','2'};
// header word counts and flags of the v3 format (v2: 6 + 6 words)
#define CKPT3_N32 8
#define CKPT3_N64 8
#define CKPT_F_DELTA 1u
#define CKPT_F_GAP49 2u

// A checkpoint header, either version (v2 records the seed only for a
// delta payload, and no min_depth).
struct CkptInfo {
    int version = 0;
    int snap_d = 0, next_bucket = 0, nbuckets_enc = 0;
    int seed_kind = 0;               // 0 none, 1 --load image, 2 survivors list
    std::string seed_path;
    u64 seed_hash = 0;
    int min_depth = -1;              // -1: not recorded (v2)
    bool delta = false, gap49 = false;
    u64 r = 0, smax = 0, depth = 0, period_s = 0, ndelta = 0, written = 0;
};

// Read just the header (and the seed path) of a checkpoint; *f is left at
// the payload when f_out is given.
static bool ckpt_peek(const char *path, CkptInfo &ci, FILE **f_out = nullptr) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    u32 h32[CKPT3_N32]; u64 h64[CKPT3_N64];
    bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, CKPT_MAGIC, 8) == 0
           && fread(h32, sizeof(u32), 1, f) == 1;
    if (ok) {
        ci.version = (int)h32[0];
        const int n32 = ci.version == 2 ? 6 : CKPT3_N32;
        const int n64 = ci.version == 2 ? 6 : CKPT3_N64;
        ok = (ci.version == 2 || ci.version == 3)
          && fread(h32 + 1, sizeof(u32), n32 - 1, f) == (size_t)(n32 - 1)
          && fread(h64, sizeof(u64), n64, f) == (size_t)n64;
        if (ok) {
            ci.snap_d = (int)h32[1]; ci.next_bucket = (int)h32[2];
            ci.nbuckets_enc = (int)h32[3]; ci.seed_kind = (int)h32[4];
            ci.r = h64[0]; ci.smax = h64[1]; ci.depth = h64[2];
            ci.period_s = h64[3]; ci.seed_hash = h64[4]; ci.ndelta = h64[5];
            if (ci.version == 2) {
                ci.delta = ci.seed_kind != 0;    // v2: a seed only for deltas
            } else {
                ci.min_depth = (int)h32[6];
                ci.delta = (h32[7] & CKPT_F_DELTA) != 0;
                ci.gap49 = (h32[7] & CKPT_F_GAP49) != 0;
                ci.written = h64[6];
            }
            const u32 bplen = h32[5];
            ci.seed_path.assign(bplen, '\0');
            ok = bplen <= 4096 &&
                 (!bplen || fread(&ci.seed_path[0], 1, bplen, f) == bplen);
        }
    }
    if (!ok || !f_out) fclose(f);
    else *f_out = f;
    return ok;
}

// The checkpoint's bucket-count field also records WHICH mid-degree method
// wrote it (bits 24+: 0 = buckets, 1 = tower logs, 2 = Pohlig-Hellman logs;
// bits 0-23: bucket or chunk count), because bucket b and chunk b cover
// different work: resuming one as the other would silently skip classes.
// Old checkpoints (buckets) decode unchanged.
#define CKPT_METHOD_SHIFT 24
#if defined(WIDE_FIELD)
static inline int ckpt_enc(int count, int method) {
    return count | (method << CKPT_METHOD_SHIFT);
}
#endif
static inline int ckpt_count(int enc) { return enc & ((1 << CKPT_METHOD_SHIFT) - 1); }
static inline int ckpt_method(int enc) { return (int)((unsigned)enc >> CKPT_METHOD_SHIFT); }

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
    gather_best(g_ckpt.snap.data(), g_ckpt.smax + 1);
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
    // v3 (round 5): the seed identity (kind, path, hash of the --load /
    // --load-survivors image) is recorded even for a full image, with the
    // run's min_depth, so an automatic resume can tell its own checkpoint
    // from another run's.  The payload is a delta against the seed if
    // CKPT_F_DELTA.
    u32 bplen = g_ckpt.base_kind ? (u32)g_ckpt.base_path.size() : 0;
    u32 hdr32[CKPT3_N32] = { (u32)3 /*version*/, (u32)g_ckpt.snap_d,
                     (u32)g_ckpt.snap_next_bucket, (u32)g_ckpt.snap_nbuckets,
                     (u32)g_ckpt.base_kind, bplen, (u32)g_ckpt.min_depth,
                     (u32)((use_delta ? CKPT_F_DELTA : 0) |
                           (g_ckpt.gap49 ? CKPT_F_GAP49 : 0)) };
    u64 ndelta = use_delta ? (u64)(dsig.size() / 2) : 0;
    u64 hdr64[CKPT3_N64] = { g_ckpt.r, g_ckpt.smax, (u64)g_ckpt.depth,
                     (u64)(g_ckpt.period_s + 0.5),
                     g_ckpt.base_kind ? g_ckpt.base_hash : 0, ndelta,
                     (u64)time(nullptr), 0 };
    bool ok = fwrite(CKPT_MAGIC, 1, 8, f) == 8
           && fwrite(hdr32, sizeof(u32), CKPT3_N32, f) == CKPT3_N32
           && fwrite(hdr64, sizeof(u64), CKPT3_N64, f) == CKPT3_N64;
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
    const char *unit = ckpt_method(g_ckpt.snap_nbuckets) ? "chunk" : "bucket";
    if (use_delta)
        fprintf(stderr, "[checkpoint: d=%d %s %d/%d, delta %" PRIu64
                " entries (%.2f MB) -> %s, %.2fs]\n",
                g_ckpt.snap_d, unit, g_ckpt.snap_next_bucket,
                ckpt_count(g_ckpt.snap_nbuckets),
                ndelta, (ndelta * 16 + 128) / 1048576.0, g_ckpt.path,
                now_s() - t0);
    else
        fprintf(stderr, "[checkpoint: d=%d %s %d/%d, full image (%.0f MB)"
                " -> %s, %.2fs]\n", g_ckpt.snap_d, unit,
                g_ckpt.snap_next_bucket, ckpt_count(g_ckpt.snap_nbuckets),
                g_ckpt.smax * 8 / 1048576.0, g_ckpt.path, now_s() - t0);
}

static bool ckpt_periodic_due() {
    return g_ckpt.enabled && g_ckpt.period_s > 0
        && (now_s() - g_ckpt.last_write) >= g_ckpt.period_s;
}
static bool ckpt_stop_requested() { return g_ckpt.stop != 0; }

// Persist and exit cleanly after a stop signal (or test stop).
static void ckpt_stop_and_exit() {
    ckpt_write();
    fprintf(stderr, "stopping on request; run the same command again to "
                    "continue (or --resume %s)\n", g_ckpt.path ? g_ckpt.path : "<file>");
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
    CkptInfo ci;
    FILE *f = nullptr;
    if (!ckpt_peek(path, ci, &f)) {
        fprintf(stderr, "%s: not a readable v2/v3 checkpoint file\n", path);
        return false;
    }
    if (ci.r != r || ci.smax != smax) {
        fprintf(stderr, "%s: r/smax (%" PRIu64 "/%" PRIu64 ") do not match this "
                "run (%" PRIu64 "/%" PRIu64 ")\n", path, ci.r, ci.smax, r, smax);
        fclose(f); return false;
    }
    const int base_kind = ci.seed_kind;          // 0 none, 1 found.bin, 2 survivors
    const bool is_delta = ci.delta;
    const std::string &bpath = ci.seed_path;
    u64 hdr64[6] = { ci.r, ci.smax, ci.depth, ci.period_s, ci.seed_hash, ci.ndelta };
    stored_period_s = (double)ci.period_s;
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
        if (base_kind) {          // v3 full image: keep the seed identity
            g_ckpt.base_path = bpath;
            g_ckpt.base_hash = ci.seed_hash;
            g_ckpt.base_kind = base_kind;
            g_ckpt.base.clear();
        }
    }
    fclose(f);
    if (ci.min_depth >= 2) g_ckpt.min_depth = ci.min_depth;
    if (ci.gap49) g_ckpt.gap49 = true;
    resume_d        = ci.snap_d;
    resume_bucket   = ci.next_bucket;
    resume_nbuckets = ci.nbuckets_enc;
    // normalize: a completed degree resumes at the next one
    if (resume_bucket >= ckpt_count(resume_nbuckets) &&
        ckpt_count(resume_nbuckets) > 0) {
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

// ---- automatic checkpointing (round 5) ----
// Every run checkpoints to <out_prefix>.ckpt (or --checkpoint <file>) every
// CKPT_DEFAULT_MINS minutes and on SIGTERM/SIGINT, unless --no-checkpoint.
// Starting a run whose checkpoint file exists resumes it automatically when
// the file belongs to the same run -- same r, smax, depth, min-depth and
// seed (--load / --load-survivors content) -- and refuses otherwise, so a
// changed command line never silently mixes two runs.  --fresh sets an old
// checkpoint aside (*.old) and starts over.
#define CKPT_DEFAULT_MINS 15

// The newest generation of `path` that has a readable header: 1 (ci, gen
// filled), 0 if no generation exists, -1 if files exist but none is readable.
static int ckpt_find(const char *path, CkptInfo &ci, std::string &gen) {
    bool any = false;
    char g[1100];
    for (int i = 0; i < 8; i++) {
        if (i == 0) snprintf(g, sizeof g, "%s", path);
        else        snprintf(g, sizeof g, "%s.%d", path, i);
        if (access(g, F_OK) != 0) { if (i == 0) continue; else break; }
        any = true;
        if (ckpt_peek(g, ci)) { gen = g; return 1; }
    }
    return any ? -1 : 0;
}

// Why the checkpoint is NOT this run's (empty string: it is).
static std::string ckpt_mismatch(const CkptInfo &ci, u64 r, u64 smax,
                                 int depth, int min_depth, int seed_kind,
                                 u64 seed_hash) {
    char b[512];
    if (ci.r != r || ci.smax != smax) {
        snprintf(b, sizeof b, "it is for r=%" PRIu64 " smax=%" PRIu64 ", this run "
                 "has r=%" PRIu64 " smax=%" PRIu64, ci.r, ci.smax, r, smax);
        return b;
    }
    if (ci.depth != (u64)depth) {
        snprintf(b, sizeof b, "it was written by a run to depth %" PRIu64
                 ", this run goes to depth %d", ci.depth, depth);
        return b;
    }
    if (ci.min_depth >= 2 && ci.min_depth != min_depth) {
        snprintf(b, sizeof b, "it was written by a run with --min-depth %d, "
                 "this run has %d", ci.min_depth, min_depth);
        return b;
    }
    if (ci.min_depth < 2 && ci.snap_d < min_depth) {    // v2: no min-depth record
        snprintf(b, sizeof b, "it stopped at degree %d, below this run's "
                 "--min-depth %d", ci.snap_d, min_depth);
        return b;
    }
    const bool recorded = ci.version >= 3 || ci.delta;
    if (!recorded && seed_kind) {
        return "it is an older-format full image with no record of the "
               "--load/--load-survivors file it started from";
    }
    if (recorded && (ci.seed_kind != seed_kind ||
                     (seed_kind && ci.seed_hash != seed_hash))) {
        const char *was = ci.seed_kind == 1 ? "--load " : ci.seed_kind == 2
                          ? "--load-survivors " : "no seed file";
        snprintf(b, sizeof b, "it started from %s%s, this run from %s", was,
                 ci.seed_kind ? ci.seed_path.c_str() : "",
                 seed_kind == 0 ? "no seed file"
                 : ci.seed_kind == seed_kind ? "a file with different contents"
                                             : "a different kind of seed file");
        return b;
    }
    return "";
}

static std::string ckpt_describe(const CkptInfo &ci) {
    char b[256], when[64] = "";
    if (ci.written) {
        time_t t = (time_t)ci.written;
        struct tm tmv;
        localtime_r(&t, &tmv);
        strftime(when, sizeof when, "written %Y-%m-%d %H:%M, ", &tmv);
    }
    snprintf(b, sizeof b, "%sdegree %d, %s %d of %d", when, ci.snap_d,
             ckpt_method(ci.nbuckets_enc) ? "chunk" : "bucket", ci.next_bucket,
             ckpt_count(ci.nbuckets_enc));
    return b;
}

// --fresh: move every generation of `path` to <name>.old
static void ckpt_set_aside(const char *path) {
    char g[1100], o[1110];
    int n = 0;
    for (int i = 0; i < 8; i++) {
        if (i == 0) snprintf(g, sizeof g, "%s", path);
        else        snprintf(g, sizeof g, "%s.%d", path, i);
        if (access(g, F_OK) != 0) continue;
        snprintf(o, sizeof o, "%s.old", g);
        if (rename(g, o) == 0) n++;
    }
    if (n) fprintf(stderr, "--fresh: set %d earlier checkpoint file%s aside as "
                   "%s*.old\n", n, n == 1 ? "" : "s", path);
}

// One run per checkpoint: an flock'ed <checkpoint>.lock, unlinked at exit.
static std::string g_lock_path;
static void ckpt_unlock_atexit() { if (!g_lock_path.empty()) unlink(g_lock_path.c_str()); }
static bool ckpt_lock(const char *path) {
    std::string lp = std::string(path) + ".lock";
    int fd = open(lp.c_str(), O_CREAT | O_RDWR, 0644);
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
    g_lock_path = lp;                            // fd stays open: lock held
    atexit(ckpt_unlock_atexit);
    return true;
}

// Poll the default stream to completion, checking the stop flag every ~100 ms
// so a signal during a long kernel is noticed promptly. Returns false if a
// stop was requested before completion.
// Sleeps back off 1 ms -> 100 ms: short kernels are not padded out to a full
// 100 ms poll period (that used to cost ~0.1 s per bucket), long ones still
// cost almost nothing to watch.  honor_stop = false waits for completion
// even if a stop was requested (used where the in-flight work is short).
static bool poll_until_done(bool honor_stop = true) {
    useconds_t nap = 1000;
    for (;;) {
        if (honor_stop && ckpt_stop_requested()) return false;
        bool all_done = true;
        for (size_t g = 0; g < g_gpus.size(); g++) {
            CUCHK(cudaSetDevice(g_gpus[g].dev));
            cudaError_t st = cudaStreamQuery(0);
            if (st == cudaErrorNotReady) { all_done = false; break; }
            if (st != cudaSuccess) CUCHK(st);
        }
        if (all_done) return true;
        usleep(nap);
        if (nap < 100000) nap *= 2;
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
    const u64 run = antilog_run_len(M64);
    launch_dims((M64 + run - 1) / run, blocks, threads);
    k_antilog<<<blocks, threads>>>(dA, M64, q, d, dgp, run);
    launch_dims(M64, blocks, threads);
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

// smallest bucket count whose slice fits in `freeb` bytes.  Any count works
// (slices are just even-aligned value ranges), and every bucket costs one
// full-domain sweep, so rounding up to a power of two wasted up to ~2x
// (e.g. d=36 on an 8 GB card needs 90 buckets but used 128).
static int pick_buckets(u64 M64, int d, size_t freeb) {
    size_t budget = freeb > GPU_MEM_RESERVE ? freeb - GPU_MEM_RESERVE : 0;
    if (budget < 8192) return 0;
    u64 domain = M64 + 1;
    u64 per = (budget - 4096) / sizeof(fw);         // slice entries that fit
    u64 nb = per ? (domain + per - 1) / per : 0;
    if (nb == 0) nb = 1;
    for (; nb <= ((u64)1 << 20); nb++)              // absorb the even rounding
        if (lowmem_need_bytes(M64, d, (int)nb) <= budget) return (int)nb;
    return 0;
}

// run the sieve for one degree using the bucketed low-memory path.
// There is no separate survivor pass any more: each of the `nbuckets`
// passes builds one slice of L (k_build_lslice) and then reads its own
// survivors straight back out of that slice (k_resolve_slice). Nothing
// resident is independent of nbuckets, so any degree can be made to fit by
// raising the bucket count. See "LOW-MEMORY MODE" at the top of the file.
static void sieve_degree_gpu_lowmem(int d, u64 r, u64 smax, int nbuckets,
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
    const int NG = (int)g_gpus.size();

    std::vector<u64> subq;
    for (u64 p : factor_u64((u64)d)) {
        int e = d / (int)p;
        subq.push_back(M64 / (((u64)1 << e) - 1));
    }

    fw gpow2[64];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);

    double t0 = now_s();
    u64 domain = M64 + 1;
    u64 chunk = lowmem_chunk(M64, nbuckets);   // even, so slices close under ^1
    unsigned zero = 0;
    unsigned long long zero64 = 0;

    // per-device scratch for this degree
    for (int g = 0; g < NG; g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        CUCHK(cudaMalloc(&G.dgp, d * sizeof(fw)));
        CUCHK(cudaMemcpy(G.dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
        G.dsub = nullptr;
        if (!subq.empty()) {
            CUCHK(cudaMalloc(&G.dsub, subq.size() * sizeof(u64)));
            CUCHK(cudaMemcpy(G.dsub, subq.data(), subq.size() * sizeof(u64),
                             cudaMemcpyHostToDevice));
        }
        CUCHK(cudaMalloc(&G.dlslice, (size_t)chunk * sizeof(fw)));
        CUCHK(cudaMalloc(&G.d_nsurv, 8));
        CUCHK(cudaMemcpy(G.d_nsurv, &zero64, 8, cudaMemcpyHostToDevice));
    }

    // Baseline snapshot for this degree: best[] currently holds all degrees
    // < d (and, on resume, buckets [0, start_bucket) of degree d).
    ckpt_snapshot(d, start_bucket, nbuckets);

    int blocks, threads;
    const u64 run = antilog_run_len(M64);       // elements per thread-run
    launch_dims((M64 + run - 1) / run, blocks, threads);

    // progress bar: one dot per completed bucket (dots for buckets already
    // done on resume are shown up front so the bar length always matches
    // nbuckets)
    fprintf(stderr, "d=%2d lowmem %d buckets", d, nbuckets);
    if (NG > 1) fprintf(stderr, " on %d GPUs", NG);
    fprintf(stderr, ": ");
    for (int b = 0; b < start_bucket; b++) fputc('=', stderr);
    fflush(stderr);

    const fw LSLICE_EMPTY = (fw)~(fw)0;   // all-ones sentinel (u32 or u64)
    unsigned long long total_heavy = 0;

    // Buckets are issued in rounds of NG, so after every round the completed
    // set is the contiguous prefix the checkpoint format expects.
    for (int base = start_bucket; base < nbuckets; base += NG) {
        // round boundary: snap holds buckets [0,base) -> safe to persist here
        if (ckpt_stop_requested() || (g_ckpt.test_stop_after &&
                g_ckpt.test_buckets_done >= g_ckpt.test_stop_after)) {
            fprintf(stderr, "\n");
            ckpt_stop_and_exit();
        }
        if (ckpt_periodic_due()) ckpt_write();

        int used = 0;
        for (int g = 0; g < NG; g++) {
            int b = base + g;
            if (b >= nbuckets) break;
            u64 lo = (u64)b * chunk;
            u64 hi = lo + chunk; if (hi > domain) hi = domain;
            if (lo >= hi) continue;
            GpuCtx &G = g_gpus[g];
            CUCHK(cudaSetDevice(G.dev));

            int fblocks, fthreads;
            launch_dims(hi - lo, fblocks, fthreads);
            k_fill_fw<<<fblocks, fthreads>>>(G.dlslice, hi - lo, LSLICE_EMPTY);
            k_build_lslice<<<blocks, threads>>>(d, M64, q, G.dgp, lo, hi,
                                                G.dlslice, run);
            CUCHK(cudaMemcpy(G.d_nheavy, &zero, 4, cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.d_overflow, &zero, 4, cudaMemcpyHostToDevice));
            int sblocks, sthreads;
            launch_dims(hi - lo, sblocks, sthreads);
            k_resolve_slice<<<sblocks, sthreads>>>(d, M64, q, rmodinv, G.dsub,
                                                   (int)subq.size(), G.dgp,
                                                   lo, hi, smax, G.dlslice,
                                                   G.d_best, G.d_heavy,
                                                   G.d_nheavy, G.d_overflow,
                                                   G.d_nsurv);
            used++;
        }
        // wait with ~100ms polling so a signal mid-round is caught quickly;
        // on stop, snap still holds buckets [0,base) and this round re-runs
        if (!poll_until_done()) { fprintf(stderr, "\n"); ckpt_stop_and_exit(); }

        // drain each device's heavy queue
        for (int g = 0; g < used; g++) {
            GpuCtx &G = g_gpus[g];
            CUCHK(cudaSetDevice(G.dev));
            unsigned nheavy = 0, overflow = 0;
            CUCHK(cudaMemcpy(&nheavy, G.d_nheavy, 4, cudaMemcpyDeviceToHost));
            CUCHK(cudaMemcpy(&overflow, G.d_overflow, 4, cudaMemcpyDeviceToHost));
            if (overflow) {
                fprintf(stderr, "\nd=%d: heavy queue overflow (%u); raise "
                        "HEAVY_CAP\n", d, overflow);
                exit(1);
            }
            if (nheavy) {
                unsigned use = nheavy > HEAVY_CAP ? HEAVY_CAP : nheavy;
                k_mark_heavy<<<use, 256>>>(G.d_heavy, use, smax, G.d_best);
            }
            total_heavy += nheavy;
        }
        if (!poll_until_done()) { fprintf(stderr, "\n"); ckpt_stop_and_exit(); }

        // round complete (inline + heavy marks on every device): refresh the
        // snapshot to the new contiguous prefix and tick the progress bar
        int done_to = base + used; if (done_to > nbuckets) done_to = nbuckets;
        ckpt_snapshot(d, done_to, nbuckets);
        g_ckpt.test_buckets_done += used;
        for (int k = 0; k < used; k++) fputc('.', stderr);
        fflush(stderr);
    }
    fprintf(stderr, "\n");
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaGetLastError());

    unsigned long long nsurv = 0;
    for (int g = 0; g < NG; g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        unsigned long long c = 0;
        CUCHK(cudaMemcpy(&c, G.d_nsurv, 8, cudaMemcpyDeviceToHost));
        nsurv += c;
        cudaFree(G.dgp);   G.dgp = nullptr;
        if (G.dsub) { cudaFree(G.dsub); G.dsub = nullptr; }
        cudaFree(G.dlslice); G.dlslice = nullptr;
        cudaFree(G.d_nsurv); G.d_nsurv = nullptr;
    }
    fprintf(stderr, "d=%2d done (lowmem, %d buckets): %llu survivors, "
            "%llu heavy progressions, %.2fs\n", d, nbuckets, nsurv,
            total_heavy, now_s() - t0);
}

// Method override (COARSE_METHOD env var or --selftest-logs):
enum { FM_AUTO = 0, FM_TOWER = 1, FM_PH = 2, FM_LOGS = 3, FM_BUCKETS = 4,
       FM_DENSE = 5, FM_RR = 6 };          // FM_RR: RR for odd d, tower for even
static int g_force_method = FM_AUTO;
static bool g_logs_selfcheck = false;    // selftest: verify every log on device
static bool g_logs_r2 = false;           // COARSE_LOGS_KERNEL=r2: round-2 kernel
static bool g_irreg_r4 = false;          // COARSE_IRREG=r4: round-4 irregular path
static bool g_irreg_fixed = false;       // COARSE_IRREG given: do not tune it
static bool g_tune = true;               // --no-tune / COARSE_TUNE=0
static int g_tune_min = 40;              // COARSE_TUNE_MIN: least degree tuned
static int g_tower_sub = -1;             // COARSE_TOWER_SUBN: -1 auto, 0 classic, n

#if defined(WIDE_FIELD)
// ---------------------------------------------------------------------------
// LOGS path: host-side driver

#define LOGS_RUN_MAX        (1u << 14)   // u values per thread-run
#define LOGS_RUNS_PER_CHUNK (1u << 17)   // runs per kernel launch (chunk)
// Class representatives are concentrated at small u: a representative
// starts with its longest zero run (typically ~log2(d) bits), so most lie
// in [2^(d-8), 2^(d-4)) and nearly every odd u below ~2^(d-12) is one.  The
// first chunks are therefore ~12x denser than average (at d=47, 2^30 odd u
// per chunk, ~8% of them representatives on average, ~100% in chunk 0), so
// progress dots come slower at the start of a degree than at the end, and
// a stop request waits for at most one chunk (a few seconds at d=47).

// chunk geometry depends only on d, so a checkpoint's chunk index is stable
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

struct LogsPlan {
    int method = LM_NONE;
    bool batched = false;                       // round-3 (batched) kernel
    PHHost ph;                                  // PH parameters (if ph_ok)
    size_t bytes = 0;                           // device table bytes
    // odd d (round 5): both engines are planned, the tuner may pick either
    bool ph_ok = false, rr_ok = false;
    PHHost ph_fb;                               // RR: PH plan of the factor-
    size_t rr_bytes = 0;                        //   base logs; the RR table
};

// The round-3 kernel for (d, method), or nullptr: Pohlig-Hellman for odd d
// (and d=2), tower for even d >= 4.  One instantiation per degree, since the
// bit-sliced code needs d at compile time.
typedef void (*LogsBatchFn)(LogsParams, const fw *, u64, u64, u64, ull *,
                            Heavy *, unsigned *, unsigned *,
                            unsigned long long *, unsigned long long *);
static LogsBatchFn logs_batch_fn(int d, int method) {
#define LBF_PH(D) case D: return method == LM_PH ? k_logs_batch<D, LM_PH> : nullptr;
#define LBF_PR(D) case D: return method == LM_PH ? k_logs_batch<D, LM_PH>     \
                               : method == LM_RR ? k_logs_batch<D, LM_RR> : nullptr;
#define LBF_TW(D) case D: return method == LM_TOWER ? k_logs_batch<D, LM_TOWER> : nullptr;
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

// Pick the logs method for degree d whose tables fit in `budget` bytes.
// want: FM_TOWER / FM_PH force one engine, anything else = automatic
// (tower for even d, Pohlig-Hellman otherwise).  The batched kernel is used
// whenever there is one for (d, method), its compile-time modulus matches
// find_primitive_poly(d), and COARSE_LOGS_KERNEL=r2 does not ask otherwise;
// the PH plan is optimized for whichever kernel will run it (the round-2
// kernel gets round 2's plan: the most balanced pair of parts).
static bool logs_plan(int d, u64 q, int want, size_t budget, LogsPlan &pl) {
    pl = LogsPlan();
    if (!gf2f_usable(q, d)) return false;
    const bool qtab = (q ^ ((u64)1 << d)) == gf2_qlow(d);
    bool tower_ok = d >= 4 && !(d & 1) && d <= 54;
    if (want == FM_TOWER && !tower_ok) return false;
    if (want != FM_PH && tower_ok) {
        // T (8 B per K-element) and the byte tables, plus LK (4 B per
        // K-element) for the classic tower or LKn + TP for the subfield one
        u64 tp = 0;
        const int n = tower_sub_choice(d / 2, g_tower_sub, &tp);
        size_t need = ((size_t)8 << (d / 2)) + ((size_t)((d + 7) / 8) << 11) + 65536;
        need += n ? ((size_t)2 << n) + 4 * (size_t)tp : (size_t)4 << (d / 2);
        if (need <= budget) {
            pl.method = LM_TOWER;
            pl.bytes = need;
            pl.batched = !g_logs_r2 && qtab && logs_batch_fn(d, LM_TOWER);
            return true;
        }
        if (want == FM_TOWER) return false;
    }
    const bool batched = !g_logs_r2 && qtab && logs_batch_fn(d, LM_PH);
    pl.ph_ok = ph_plan_host(d, q, budget, pl.ph,
                            batched ? kPHCostBitsliced : kPHCostRound2);
    // RR (round 5, batched kernel only): the table of 2^(dh+1) logs, and a
    // PH plan for the factor-base logs -- the sieve's own if it has one,
    // else (d=49) one with up to 2^22 giant steps per log.  Test hook:
    // COARSE_RR_FB_BUDGET_MB=m always plans the factor-base logs separately
    // with tables of at most m MB (many giant steps: d=49's path at small d).
    if (want != FM_PH && !g_logs_r2 && qtab && logs_batch_fn(d, LM_RR)) {
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
    if (want == FM_RR && (pl.rr_ok || d >= 5)) {   // (d < 5: no RR kernel)
        if (!pl.rr_ok) return false;
        pl.method = LM_RR;
    } else if (pl.ph_ok) {
        pl.method = LM_PH;
    } else if (pl.rr_ok && want != FM_PH) {
        pl.method = LM_RR;                      // d = 49: the only engine
    } else {
        return false;
    }
    pl.batched = pl.method == LM_RR ? true : batched;
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
    L.irreg_r4 = g_irreg_r4;
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
static void logs_setup_device(GpuCtx &G, int d, u64 q, const LogsPlan &pl,
                              const TowerHost &W, std::vector<u64> &Th,
                              LogsParams &L,
                              const std::vector<u64> *rrL = nullptr) {
    const u64 M = L.M;
    int blocks, threads;
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
                if (shm + tpb <= (size_t)optin) { shm += tpb; L.tw.shm_tp = 1; }
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
    G.d_btab = G.d_T = nullptr; G.d_LK = G.d_TP = nullptr; G.d_LKn = nullptr;
    G.d_rrL = nullptr;
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
    shm = method == LM_TOWER ? L.tw.shm_bytes : 0;
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
            L.method == LM_TOWER ? "tower" : L.method == LM_RR ? "RR" : "PH",
            hi, bad);
    if (bad) { fprintf(stderr, "logs selfcheck FAILED at d=%d\n", d); exit(1); }
}

// selftest: the batched kernel in check mode over [ulo, uhi): every class
// it enumerates must get log(g^u) == u, and it must enumerate exactly the
// representatives that the necklace test finds there.  Returns failures.
static u64 logs_batch_check(GpuCtx &G, int d, const LogsPlan &pl,
                            LogsParams L, u64 ulo, u64 uhi, u64 *nclasses) {
    LogsBatchFn kfn = logs_batch_fn(d, pl.method);
    if (!kfn) return 0;
    unsigned long long zero = 0, ns = 0, ne = 0, nc = 0;
    unsigned long long *dns, *dne, *dnc;
    CUCHK(cudaMalloc(&dns, 8)); CUCHK(cudaMalloc(&dne, 8)); CUCHK(cudaMalloc(&dnc, 8));
    CUCHK(cudaMemcpy(dns, &zero, 8, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dne, &zero, 8, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dnc, &zero, 8, cudaMemcpyHostToDevice));
    L.check = 1;
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
    cudaFree(dns); cudaFree(dne); cudaFree(dnc);
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

// --selftest-logs, second part: the rotation test, FKM and the discrete-log
// engines at every degree up to 55 on device 0 (the sieve comparison only
// reaches d=20).  want = FM_TOWER / FM_PH limits it to one engine.
// Returns the number of failures.
static u64 logs_wide_selftest(u64 r, int want) {
    GpuCtx &G = g_gpus[0];
    CUCHK(cudaSetDevice(G.dev));
#if defined(__CUDACC__)
    const u64 nrep = (u64)1 << 22, nlog = (u64)1 << 20, nbat = (u64)1 << 22,
              nirr = (u64)1 << 20;
    int dmax_log = KEY_MAX_DEGREE;
    int dmax_rr = 47;                         // 49: its many-giant-step table
                                              // build (minutes), on request
    int d_rr_bsgs = 41;                       // RR with d=49's kind of plan
#else                                         // CPU emulation: keep it short
    const u64 nrep = (u64)1 << 12, nlog = (u64)1 << 9, nbat = (u64)1 << 12,
              nirr = (u64)1 << 13;
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

    {   // irregular classes: the round-5 batch lift vs the round-4 formulas
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
            if (ib[0]) fprintf(stderr, "  d=%d: irregular lift: %llu mismatches\n",
                               d, ib[0]);
            if (ib[1] < nirr / 8) {             // the check must test something
                fprintf(stderr, "  d=%d: irregular lift: only %llu classes "
                        "joined\n", d, ib[1]);
                ifail++;
            }
            ifail += ib[0]; nj += ib[1]; ns += ib[2]; nt += ib[3];
        }
        cudaFree(dib);
        fprintf(stderr, "  [irregular classes, batch lift vs extended Euclid, "
                "d=4..%d: %" PRIu64 " joined (%" PRIu64 " with t > 0), %" PRIu64
                " skipped: %s]\n", KEY_MAX_DEGREE, nj, nt, ns,
                ifail ? "FAILED" : "ok");
        fails += ifail;
    }

    size_t freeb = 0, totb = 0;
    CUCHK(cudaMemGetInfo(&freeb, &totb));
    const size_t budget = logs_table_budget(freeb);
    for (int d = 21; d <= dmax_log; d++) {
        const u64 q = find_primitive_poly(d), M = ((u64)1 << d) - 1;
        fw gpow2[64];
        gpow2[0] = 2;
        for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);
        // RR (round 5) at the odd degrees, and once with the factor-base
        // logs forced through a many-giant-step plan (d=49's case)
        const bool rr_here = (d & 1) && d <= dmax_rr && !g_logs_r2;   // batched only
        const bool rr_bsgs = d == d_rr_bsgs && d <= dmax_log && !g_logs_r2;
        for (int eng : {(int)FM_TOWER, (int)FM_PH, (int)FM_RR, FM_RR + 100}) {
            if (want != FM_LOGS && want != eng % 100) continue;
            if (eng == FM_RR && !rr_here) continue;
            if (eng == FM_RR + 100 && !rr_bsgs) continue;
            if (eng == FM_RR + 100) setenv("COARSE_RR_FB_BUDGET_MB", "4", 1);
            LogsPlan pl;
            const bool ok_plan = logs_plan(d, q, eng % 100, budget, pl);
            if (eng == FM_RR + 100) unsetenv("COARSE_RR_FB_BUDGET_MB");
            if (!ok_plan) {   // tower: even d only
                if (eng == FM_PH)
                    fprintf(stderr, "  [d=%d PH    logs: no feasible plan "
                            "(expected only for d=49)]\n", d);
                if (eng == FM_PH && d != 49) fails++;
                if (eng % 100 == FM_RR) {
                    fprintf(stderr, "  [d=%d RR    logs: no feasible plan]\n", d);
                    fails++;
                }
                continue;
            }
            if (eng % 100 == FM_RR && pl.method != LM_RR) continue;   // even d
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
            if (pl.batched) {
                int lb = 0;
                while (((u64)2 << lb) <= nbat && lb + 1 <= d - 8) lb++;
                const u64 blo = logs_dense_stretch(d, lb);
                bfail = logs_batch_check(G, d, pl, L, blo, blo + ((u64)1 << lb),
                                         &ncls);
                if (ncls < ((u64)1 << lb) / 8) {      // the check must test something
                    fprintf(stderr, "  d=%d: batched check found only %" PRIu64
                            " classes in its stretch\n", d, ncls);
                    bfail++;
                }
            }
            logs_free_device(G);
            cudaFree(G.dgp); G.dgp = nullptr;
            fprintf(stderr, "  [d=%d %-5s logs, %4.0f MB tables: %" PRIu64
                    " checked, %llu wrong", d,
                    pl.method == LM_TOWER ? "tower" : pl.method == LM_RR ? "RR" : "PH",
                    pl.bytes / 1048576.0, nlog, bad);
            if (pl.batched)
                fprintf(stderr, "; batched: %" PRIu64 " classes, %s", ncls,
                        bfail ? "FAILED" : "ok");
            if (pl.method == LM_TOWER && W.sub_k)
                fprintf(stderr, "; over GF(2^%d)", W.sub_n);
            else if (pl.method == LM_TOWER)
                fprintf(stderr, "; classic");
            else if (pl.method == LM_RR)
                fprintf(stderr, "; factor-base logs with up to %" PRIu64 " giant "
                        "step%s%s", rr_giant, rr_giant == 1 ? "" : "s",
                        eng == FM_RR + 100 ? " (forced, d=49's path)" : "");
            fprintf(stderr, ", %.2fs]\n", now_s() - t0);
            fails += bad + bfail;
        }
    }
    cudaFree(dbad);
    return fails;
}

// run the sieve for one degree with the logs path (any VRAM; tables only)
// ---------------------------------------------------------------------------
// Auto-tuning (round 5).  Which variant of the batched kernel is fastest
// depends on the GPU -- its L2 size (an RTX 4090's 72 MB holds the whole
// tower table T up to d=46, a 2070 SUPER's 4 MB holds none of it), shared
// memory, integer throughput -- so for every degree d >= g_tune_min the
// candidates run the same sample of classes on GPU 0 first and the fastest
// is used for the degree:
//   tower: inner subfield GF(2^n) for the largest two divisors n of m (or
//          the forced one), and the classic tower;
//   both:  the round-4 irregular path, with the default tower, where
//          classes with gcd(u, M) > 1 are not rare (a safety net, and the
//          A/B number for the log).
// All candidates compute identical results; the sample chunk's marks are
// simply repeated by the real pass (atomicMin), its heavy progressions
// dropped (the real pass queues them again).  Costs a few seconds per
// degree; the report line shows ns per class for every candidate.
// --no-tune (COARSE_TUNE=0) disables it; COARSE_TOWER_SUBN / COARSE_IRREG
// fix their knob; test hook: COARSE_TUNE_PICK=i takes candidate i.
struct TuneCand { int method, sub_n; bool r4; double ns; };

static void tune_name(const TuneCand &c, char *b, size_t n) {
    const char *ir = c.r4 ? "+r4-irregular" : "";
    if (c.method == LM_TOWER && c.sub_n) snprintf(b, n, "GF(2^%d)%s", c.sub_n, ir);
    else if (c.method == LM_TOWER) snprintf(b, n, "classic%s", ir);
    else snprintf(b, n, "%s%s", c.method == LM_RR ? "RR" : "PH", ir);
}

// pl.method may change (odd d: PH <-> RR, not when COARSE_METHOD forces
// one); rrL receives the RR table if it was built
static void logs_tune(int d, u64 q, u64 r, u64 smax, LogsPlan &pl,
                      const fw *gpow2, std::vector<u64> &Th,
                      std::vector<u64> &rrL, int &sub_use, bool &r4_use) {
    GpuCtx &G = g_gpus[0];
    CUCHK(cudaSetDevice(G.dev));
    // candidates; the first is the default
    std::vector<TuneCand> cand;
    int sub_auto = 0;
    if (pl.method == LM_TOWER) {
        u64 tp;
        sub_auto = tower_sub_choice(d / 2, g_tower_sub, &tp);
        cand.push_back({LM_TOWER, sub_auto, r4_use, 0});
        if (g_tower_sub < 0) {                  // the variants, if they fit
            size_t freeb = 0, totb = 0;
            CUCHK(cudaMemGetInfo(&freeb, &totb));
            const size_t budget = logs_table_budget(freeb);
            const int m = d / 2;
            int second = 0;
            for (int c = sub_auto - 1; c >= 2 && !second; c--) if (m % c == 0) second = c;
            for (int n : {second, 0}) {
                if (n == sub_auto || (n == 0 && sub_auto == 0) || (n && !second)) continue;
                u64 tpn = 0;
                if (n && tower_sub_choice(m, n, &tpn) != n) continue;
                size_t need = ((size_t)8 << m) + ((size_t)((d + 7) / 8) << 11) + 65536;
                need += n ? ((size_t)2 << n) + 4 * (size_t)tpn : (size_t)4 << m;
                if (need <= budget) cand.push_back({LM_TOWER, n, r4_use, 0});
            }
        }
    } else {
        cand.push_back({pl.method, 0, r4_use, 0});
        const bool free_choice = g_force_method == FM_AUTO || g_force_method == FM_LOGS;
        const int other = pl.method == LM_PH ? LM_RR : LM_PH;
        if (free_choice && (other == LM_RR ? pl.rr_ok : pl.ph_ok) &&
            logs_batch_fn(d, other))
            cand.push_back({other, 0, r4_use, 0});
    }
    if (!g_irreg_fixed) {                       // irregular classes common?
        LogsParams L0 = logs_params_base(d, r, smax, q, pl.method);
        double p2 = 0;                          // P(class needs an inverse)
        for (int i = 0; i < L0.npf; i++) p2 += 1.0 / ((double)L0.pf[i].p * L0.pf[i].p);
        if (1 - pow(1 - p2, 32) > 0.05)         // ... in some lane of a warp
            cand.insert(cand.begin() + 1, TuneCand{cand[0].method, sub_auto,
                                                   !r4_use, 0});
    }
    if (cand.size() < 2) return;

    // the sample: half a chunk at 1/8 of the range, doubled while too small
    u64 total, run, chunk;
    int nchunks;
    logs_geometry(d, total, run, chunk, nchunks);
    const int cs = nchunks / 8;
    const u64 ulo = (u64)cs * chunk;
    u64 span = chunk / 2 ? chunk / 2 : chunk;
    const u64 krun = run >= 512 ? run / 8 : run;
    unsigned long long zero64 = 0, cls = 0, nerr = 0;
    unsigned zero = 0;
    cudaEvent_t e0, e1;
    CUCHK(cudaEventCreate(&e0));
    CUCHK(cudaEventCreate(&e1));
    const double t0 = now_s();
    int cur_m = -1, cur_sub = -1;               // tables set up for these
    LogsParams Lcur;
    bool timed_any = false;
    for (size_t ci = 0; ci < cand.size(); ci++) {
        TuneCand &c = cand[ci];
        c.ns = -1;
        if (c.method != cur_m || c.sub_n != cur_sub) {   // (re)build the tables
            if (cur_m >= 0) {
                logs_free_device(G);
                cudaFree(G.dgp); G.dgp = nullptr;
                cudaFree(G.d_nsurv); G.d_nsurv = nullptr;
                cudaFree(G.d_nerr); G.d_nerr = nullptr;
            }
            cur_m = cur_sub = -1;
            if (c.method == LM_RR && rrL.empty() && !rr_build_table(d, q, r, pl, rrL))
                continue;
            TowerHost W;
            if (c.method == LM_TOWER && !tower_build_host(d, q, W, c.sub_n)) continue;
            LogsPlan pc = pl;
            pc.method = c.method;
            Lcur = logs_params_base(d, r, smax, q, c.method);
            CUCHK(cudaMalloc(&G.dgp, d * sizeof(fw)));
            CUCHK(cudaMemcpy(G.dgp, gpow2, d * sizeof(fw), cudaMemcpyHostToDevice));
            CUCHK(cudaMalloc(&G.d_nsurv, 8));
            CUCHK(cudaMalloc(&G.d_nerr, 8));
            logs_setup_device(G, d, q, pc, W, Th, Lcur, &rrL);
            cur_m = c.method;
            cur_sub = c.sub_n;
        }
        const LogsBatchFn kfn = logs_batch_fn(d, c.method);
        LogsParams L = Lcur;
        L.irreg_r4 = c.r4;
        for (int rep = timed_any ? 1 : 0; rep < 2; rep++) {   // first: warm-up
            u64 uhi = ulo + span; if (uhi > total) uhi = total;
            int blocks, threads;
            size_t shm;
            logs_batch_dims(kfn, (uhi - ulo + krun - 1) / krun, c.method, L,
                            blocks, threads, shm);
            CUCHK(cudaMemcpy(G.d_nsurv, &zero64, 8, cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.d_nerr, &zero64, 8, cudaMemcpyHostToDevice));
            CUCHK(cudaEventRecord(e0));
            kfn<<<blocks, threads, shm>>>(L, G.dgp, ulo, uhi, krun, G.d_best,
                                          G.d_heavy, G.d_nheavy, G.d_overflow,
                                          G.d_nsurv, G.d_nerr);
            CUCHK(cudaEventRecord(e1));
            CUCHK(cudaEventSynchronize(e1));
            CUCHK(cudaGetLastError());
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
            // a sample of at least ~2^25 classes (or the whole range)
            if (rep == 0 && cls < ((u64)1 << 25) && span < total - ulo &&
                ms < 2000) { span *= 2; rep--; }
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
    pl.batched = true;
    pl.bytes = pl.method == LM_RR ? pl.rr_bytes
             : pl.method == LM_PH ? pl.ph.table_bytes : pl.bytes;
    sub_use = cand[best].sub_n;
    r4_use = cand[best].r4;
    if (pl.method != LM_RR) { rrL.clear(); rrL.shrink_to_fit(); }
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

static void sieve_degree_gpu_logs(int d, u64 r, u64 smax, const LogsPlan &pl_in,
                                  int start_chunk = 0) {
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

    fw gpow2[64];
    gpow2[0] = 2;
    for (int t = 1; t < d; t++) gpow2[t] = gf_sqr(gpow2[t - 1], q, d);

    double t0 = now_s();
    std::vector<u64> Th;                        // T, built once per degree
    std::vector<u64> rrL;                       // RR: the table, host copy
    int sub_use = g_tower_sub;
    bool r4_use = g_irreg_r4;
    if (pl.batched && g_tune && d >= g_tune_min)
        logs_tune(d, q, r, smax, pl, gpow2, Th, rrL, sub_use, r4_use);
    const LogsBatchFn kfn = pl.batched ? logs_batch_fn(d, pl.method) : nullptr;
    const char *mname = pl.method == LM_TOWER ? "tower"
                      : pl.method == LM_RR ? "RR" : "PH";
    if (pl.method == LM_RR && rrL.empty() && !rr_build_table(d, q, r, pl, rrL)) {
        fprintf(stderr, "d=%d: RR table construction failed\n", d);
        exit(1);
    }
    TowerHost W;
    if (pl.method == LM_TOWER && !tower_build_host(d, q, W, sub_use)) {
        fprintf(stderr, "d=%d: tower construction failed -- internal error\n", d);
        exit(1);
    }
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
        Lg.back().irreg_r4 = r4_use;
        // test hook: COARSE_TEST_FAULT_EVERY=n flips a bit of the log of
        // every class with u = 1 (mod n), to exercise the error stop
        if (const char *e = getenv("COARSE_TEST_FAULT_EVERY"))
            Lg.back().fault = strtoull(e, nullptr, 10);
        logs_setup_device(G, d, q, pl, W, Th, Lg.back(), &rrL);
    }
    rrL.clear(); rrL.shrink_to_fit();
    char inner[48] = "";                        // the inner tower, if any
    if (pl.method == LM_TOWER && W.sub_k)
        snprintf(inner, sizeof inner, " over GF(2^%d)", W.sub_n);
    W = TowerHost();                            // host copies no longer needed
    Th.clear(); Th.shrink_to_fit();
    fprintf(stderr, "d=%2d logs (%s%s%s, %.0f MB tables, built in %.1fs), %d "
            "chunk%s", d, mname, inner,
            !kfn ? ", round-2 kernel" : r4_use ? ", batched, r4 irregular"
                                               : ", batched",
            pl.bytes / 1048576.0, now_s() - t0, nchunks, nchunks == 1 ? "" : "s");
    if (NG > 1) fprintf(stderr, " on %d GPUs", NG);
    fprintf(stderr, ": ");
    if (g_logs_selfcheck) { fprintf(stderr, "\n"); logs_selfcheck(d, Lg[0]); }

    // One blocking-sync event per device: the host thread sleeps in the
    // driver until its chunk is done and wakes at once -- no polling delay
    // between chunks (chunks are short; a 1..128 ms poll back-off would
    // idle the GPU ~10-25% of the time) and no busy-waiting CPU core.
    std::vector<cudaEvent_t> ev(NG);
    for (int g = 0; g < NG; g++) {
        CUCHK(cudaSetDevice(g_gpus[g].dev));
        CUCHK(cudaEventCreateWithFlags(&ev[g], cudaEventBlockingSync |
                                               cudaEventDisableTiming));
    }

    const int enc = ckpt_enc(nchunks, pl.method);
    const int dot_every = nchunks > 64 ? (nchunks + 63) / 64 : 1;
    for (int c = 0; c < start_chunk; c += dot_every) fputc('=', stderr);
    fflush(stderr);
    unsigned zero = 0;
    unsigned long long total_heavy = 0;
    const double t_pass = now_s();              // for the one time-left note
    bool eta_shown = false;
    for (int base = start_chunk; base < nchunks; base += NG) {
        // round boundary: every chunk < base is complete on every device
        if (ckpt_stop_requested() || (g_ckpt.test_stop_after &&
                g_ckpt.test_buckets_done >= g_ckpt.test_stop_after)) {
            fprintf(stderr, "\n");
            ckpt_snapshot(d, base, enc);
            ckpt_stop_and_exit();
        }
        if (ckpt_periodic_due()) { ckpt_snapshot(d, base, enc); ckpt_write(); }
        int used = 0;
        for (int g = 0; g < NG && base + g < nchunks; g++) {
            GpuCtx &G = g_gpus[g];
            CUCHK(cudaSetDevice(G.dev));
            u64 ulo = (u64)(base + g) * chunk;
            u64 uhi = ulo + chunk; if (uhi > total) uhi = total;
            CUCHK(cudaMemcpy(G.d_nheavy, &zero, 4, cudaMemcpyHostToDevice));
            CUCHK(cudaMemcpy(G.d_overflow, &zero, 4, cudaMemcpyHostToDevice));
            int blocks, threads;
            if (kfn) {
                // runs of krun = run/8 u-values: ~50 per thread, so threads
                // finish together (the chunk itself is unchanged)
                const u64 krun = run >= 512 ? run / 8 : run;
                size_t shm;
                logs_batch_dims(kfn, (uhi - ulo + krun - 1) / krun, pl.method,
                                Lg[g], blocks, threads, shm);
                kfn<<<blocks, threads, shm>>>(Lg[g], G.dgp, ulo, uhi, krun,
                                              G.d_best, G.d_heavy, G.d_nheavy,
                                              G.d_overflow, G.d_nsurv,
                                              G.d_nerr);
            } else {
                launch_dims((uhi - ulo + run - 1) / run, blocks, threads);
                k_logs_sweep<<<blocks, threads>>>(Lg[g], G.dgp, ulo, uhi, run,
                                                  G.d_best, G.d_heavy,
                                                  G.d_nheavy, G.d_overflow,
                                                  G.d_nsurv, G.d_nerr);
            }
            CUCHK(cudaEventRecord(ev[g], 0));
            used++;
        }
        for (int g = 0; g < used; g++) {        // a chunk is seconds at most
            CUCHK(cudaSetDevice(g_gpus[g].dev));
            CUCHK(cudaEventSynchronize(ev[g]));
        }
        for (int g = 0; g < used; g++) {
            GpuCtx &G = g_gpus[g];
            CUCHK(cudaSetDevice(G.dev));
            unsigned nheavy = 0, overflow = 0;
            unsigned long long nbad = 0;
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
                        "errors in chunks %d..%d (GPU %d) -- results are NOT "
                        "trustworthy; stopping (no output written; any "
                        "checkpoint predates these chunks)\n", d, nbad, base,
                        base + used - 1, g);
                exit(1);
            }
            // (queued ahead of the next chunk, which waits for it in-stream)
            if (nheavy) k_mark_heavy<<<nheavy, 256>>>(G.d_heavy, nheavy, smax,
                                                       G.d_best);
            total_heavy += nheavy;
        }
        g_ckpt.test_buckets_done += used;
        for (int c = base; c < base + used; c++)
            if ((c + 1) % dot_every == 0 || c + 1 == nchunks) {
                fputc('.', stderr);
                // after the first dot of a long pass (resumed: of this run's
                // chunks), one estimate of the time left
                if (!eta_shown && c + 1 < nchunks && c + 1 > start_chunk) {
                    eta_shown = true;
                    const double left = (now_s() - t_pass) *
                        (double)(nchunks - c - 1) / (double)(c + 1 - start_chunk);
                    if (left > 600)
                        fprintf(stderr, "[~%.1f h to go]", left / 3600.0);
                }
            }
        fflush(stderr);
    }
    fprintf(stderr, "\n");

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
    fprintf(stderr, "d=%2d done (logs/%s, %d chunks): %llu survivors, %llu "
            "heavy progressions, %.2fs\n", d, mname, nchunks, nsurv,
            total_heavy, now_s() - t0);
}
#endif // WIDE_FIELD

// Returns the last degree fully sieved: depth, unless some degree could not
// be done here (then best[] holds the results through the returned degree).
static int run_gpu(u64 r, int depth, u64 smax, std::vector<ull> &best,
                   int force_lowmem_buckets = 0, int min_depth = 2,
                   const std::vector<ull> *preload = nullptr,
                   int resume_bucket = 0, int resume_nbuckets = 0) {
    // Every device gets a private best[]; the elementwise min across them is
    // the answer (see the MULTI-GPU note above).  Seeding each with the same
    // preload keeps that identity true from the start.
    for (size_t g = 0; g < g_gpus.size(); g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        CUCHK(cudaMalloc(&G.d_best, (smax + 1) * sizeof(ull)));
        CUCHK(cudaMalloc(&G.d_heavy, HEAVY_CAP * sizeof(Heavy)));
        CUCHK(cudaMalloc(&G.d_nheavy, 4));
        CUCHK(cudaMalloc(&G.d_overflow, 4));
        if (preload) {
            // Resume/extend: seed best[] with results from a prior run so we
            // only sieve the new degrees.  A degree-d factor can only improve
            // an s that has no smaller-degree factor, so atomicMin against
            // the loaded state is exactly correct.
            CUCHK(cudaMemcpy(G.d_best, preload->data(),
                             (smax + 1) * sizeof(ull), cudaMemcpyHostToDevice));
        } else {
            int blocks, threads;
            launch_dims(smax + 1, blocks, threads);
            k_fill<<<blocks, threads>>>(G.d_best, smax + 1, ~0ULL);
        }
    }
    CUCHK(cudaSetDevice(g_gpus[0].dev));
    ull *d_best = g_gpus[0].d_best;              // dense path runs on device 0
    Heavy *d_heavy = g_gpus[0].d_heavy;
    unsigned *d_nheavy = g_gpus[0].d_nheavy, *d_overflow = g_gpus[0].d_overflow;

    int reached = min_depth - 1;
    // test hook: COARSE_TEST_STOP_DEGREE=n behaves as if degree n could not
    // be sieved here (exercises the early-stop handling in main)
    const char *tsd = getenv("COARSE_TEST_STOP_DEGREE");
    const int test_stop_degree = tsd ? atoi(tsd) : 0;
    for (int d = min_depth; d <= depth; d++) {
        if (d == test_stop_degree) {
            fprintf(stderr, "d=%d: COARSE_TEST_STOP_DEGREE: stopping at depth "
                    "%d\n", d, d - 1);
            break;
        }
        // On a mid-degree resume, the first degree processed continues from
        // resume_bucket using the exact nbuckets the checkpoint was made
        // with (bucket value-ranges depend on nbuckets); later degrees start
        // fresh with the auto-picked count.
        int start_bucket = (d == min_depth) ? resume_bucket : 0;
        int forced_nb    = (d == min_depth) ? resume_nbuckets : 0;

        // dense path keeps two 2^d-element tables (A and L), each sizeof(fw)
        // bytes/element: 8*2^d in the narrow build, 16*2^d in the wide build.
        size_t need = (((size_t)2 * sizeof(fw)) << d) + 16;
        // size for the most constrained device so every device fits
        size_t freeb = (size_t)-1;
        for (size_t g = 0; g < g_gpus.size(); g++) {
            size_t f = 0, t = 0;
            CUCHK(cudaSetDevice(g_gpus[g].dev));
            cudaMemGetInfo(&f, &t);
            if (f < freeb) freeb = f;
        }
        CUCHK(cudaSetDevice(g_gpus[0].dev));
        const int fm = g_force_method;
#if defined(WIDE_FIELD)
        // logs path: chosen (forced) engine or automatic; true if it ran
        auto try_logs = [&](int want, int start_chunk, bool must) -> bool {
            LogsPlan pl;
            const u64 qd = find_primitive_poly(d);
            const size_t lbudget = logs_table_budget(freeb);
            if (!logs_plan(d, qd, want, lbudget, pl)) {
                if (must) {
                    fprintf(stderr, "d=%d: logs path (%s) not possible here "
                            "(tables do not fit in %.2f GB, or unsupported d)\n",
                            d, want == FM_TOWER ? "tower" : want == FM_PH
                            ? "PH" : want == FM_RR ? "RR" : "auto",
                            lbudget / 1073741824.0);
                    exit(1);
                }
                return false;
            }
            if (fm == FM_AUTO)
                fprintf(stderr, "d=%d: dense tables need %.2f GB but only %.2f "
                        "GB free; using the logs path\n", d,
                        need / 1073741824.0, freeb / 1073741824.0);
            sieve_degree_gpu_logs(d, r, smax, pl, start_chunk);
            return true;
        };
#endif
        if (force_lowmem_buckets > 0) {
            sieve_degree_gpu_lowmem(d, r, smax, force_lowmem_buckets,
                                    start_bucket);
        } else if (forced_nb > 0) {
            int meth = ckpt_method(forced_nb), cnt = ckpt_count(forced_nb);
            if (meth == LM_NONE) {
                // resuming mid-degree: reuse the checkpoint's bucket count
                fprintf(stderr, "d=%d: resuming at bucket %d/%d\n", d,
                        start_bucket, cnt);
                sieve_degree_gpu_lowmem(d, r, smax, cnt, start_bucket);
            } else {
#if defined(WIDE_FIELD)
                u64 tt, rn, ch;
                int nch;
                logs_geometry(d, tt, rn, ch, nch);
                if (nch != cnt) {
                    fprintf(stderr, "d=%d: checkpoint has %d chunks but this "
                            "build uses %d; cannot resume\n", d, cnt, nch);
                    exit(1);
                }
                fprintf(stderr, "d=%d: resuming logs path at chunk %d/%d\n", d,
                        start_bucket, cnt);
                // (any logs engine covers the same chunks; this is the one
                // that wrote the checkpoint)
                try_logs(meth == LM_TOWER ? FM_TOWER : meth == LM_RR ? FM_RR
                                                                     : FM_PH,
                         start_bucket, true);
#else
                fprintf(stderr, "checkpoint was written by the logs path; "
                        "resume it with the -DWIDE_FIELD build\n");
                exit(1);
#endif
            }
        } else {
            bool done = false;
            if (fm == FM_DENSE || (fm == FM_AUTO && need <= freeb)) {
                sieve_degree_gpu(d, r, smax, d_best, d_heavy, d_nheavy,
                                 d_overflow);
                done = true;
            } else if (fm == FM_TOWER || fm == FM_PH || fm == FM_LOGS ||
                       fm == FM_RR) {
#if defined(WIDE_FIELD)
                done = try_logs(fm, 0, true);
#else
                fprintf(stderr, "the logs path needs the -DWIDE_FIELD build\n");
                exit(1);
#endif
            }
#if defined(WIDE_FIELD)
            else if (fm == FM_AUTO) done = try_logs(FM_LOGS, 0, false);
#endif
            if (!done && d > 47) {
                // beyond d=47 only the logs path is supported (a bucketed
                // pass would take years); d=49 has only RR (2^49-1 = 127 *
                // 4432676798593), which needs its 256 MB table plus a
                // baby-step table of a GB or more for the factor-base logs
                fprintf(stderr, "d=%d: no discrete-log method fits here (%s); "
                        "stopping at depth %d\n", d, d == 49 ? "RR needs "
                        "~1.5 GB or more of free VRAM at d=49" : "tables too "
                        "large for this card", d - 1);
                break;
            }
            if (!done) {
                // dense tables don't fit and no logs path: bucketed
                // low-memory path (see "LOW-MEMORY MODE")
                u64 M64 = ((u64)1 << d) - 1;
                int nbuckets = pick_buckets(M64, d, freeb);
                if (nbuckets == 0) {
                    fprintf(stderr, "d=%d needs %.2f GB tables but only %.2f "
                            "GB free, and even 2^20 buckets does not fit; "
                            "stopping at depth %d\n", d, need / 1073741824.0,
                            freeb / 1073741824.0, d - 1);
                    break;
                }
                fprintf(stderr, "d=%d: dense tables need %.2f GB but only "
                        "%.2f GB free; switching to low-memory mode (%d "
                        "buckets)\n", d, need / 1073741824.0,
                        freeb / 1073741824.0, nbuckets);
                sieve_degree_gpu_lowmem(d, r, smax, nbuckets, start_bucket);
            }
        }

        // Degree boundary: a stop during a dense degree (which can't
        // checkpoint mid-way) is honored here, and periodic saves land here
        // too. best[] is a clean "degrees <= d done" state.
        if (ckpt_stop_requested()) {
            ckpt_snapshot(d + 1, 0, 0);   // next degree, no buckets done
            ckpt_stop_and_exit();
        }
        // Only worth a periodic save if more degrees remain; after the last
        // one the run is about to write its real output anyway.
        reached = d;
        if (d < depth && ckpt_periodic_due()) {
            ckpt_snapshot(d + 1, 0, 0); ckpt_write();
        }
    }

    if (reached < depth && g_ckpt.enabled) {
        // stopped early (a degree that cannot be done here): save the degree
        // boundary, so a resume -- e.g. with more free VRAM -- starts there
        ckpt_snapshot(reached + 1, 0, 0);
        ckpt_write();
    }
    best.resize(smax + 1);
    gather_best(best.data(), smax + 1);
    for (size_t g = 0; g < g_gpus.size(); g++) {
        GpuCtx &G = g_gpus[g];
        CUCHK(cudaSetDevice(G.dev));
        cudaFree(G.d_best); cudaFree(G.d_heavy);
        cudaFree(G.d_nheavy); cudaFree(G.d_overflow);
        G.d_best = nullptr;
    }
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
int main(int argc, char **argv) {
    // selftests run on a single device
    g_gpus.assign(1, GpuCtx());
    // COARSE_METHOD=auto|dense|buckets|logs|tower|ph overrides the per-degree
    // method choice (testing / A-B timing); tower falls back to an error on
    // odd d, logs = tower for even d and Pohlig-Hellman for odd d.
    // COARSE_LOGS_KERNEL=r2 uses the round-2 logs kernel (one class per
    // thread, scalar arithmetic) instead of the batched one: A/B timing
    if (const char *e = getenv("COARSE_LOGS_KERNEL")) {
        if (!strcmp(e, "r2")) g_logs_r2 = true;
        else if (strcmp(e, "batched")) {
            fprintf(stderr, "COARSE_LOGS_KERNEL=%s: expected batched or r2\n", e);
            return 1;
        }
    }
    // COARSE_IRREG=r4 handles the classes with gcd(u, M) > 1 the round-4
    // way (extended Euclid per class) in the batched kernel: A/B timing
    if (const char *e = getenv("COARSE_IRREG")) {
        if (!strcmp(e, "r4")) g_irreg_r4 = true;
        else if (strcmp(e, "batch")) {
            fprintf(stderr, "COARSE_IRREG=%s: expected batch or r4\n", e);
            return 1;
        }
        g_irreg_fixed = true;
    }
    // COARSE_TUNE=0 turns the per-degree auto-tuning off (like --no-tune);
    // COARSE_TUNE_MIN=d tunes from degree d up (default 40)
    if (const char *e = getenv("COARSE_TUNE")) g_tune = atoi(e) != 0;
    if (const char *e = getenv("COARSE_TUNE_MIN")) g_tune_min = atoi(e);
#if !defined(WIDE_FIELD)
    // the logs-path knobs are parsed (so a bad value is still reported) but
    // only the wide build has a logs path
    if (g_logs_r2 || g_irreg_r4 || g_irreg_fixed || !g_tune || g_tune_min != 40 ||
        g_tower_sub >= 0 || g_logs_selfcheck)
        fprintf(stderr, "note: COARSE_LOGS_KERNEL, COARSE_IRREG, COARSE_TUNE*, "
                "COARSE_TOWER_SUBN affect only the wide build\n");
#endif
    // COARSE_TOWER_SUBN=0 uses the classic (round-3) tower; =n forces the
    // subfield K_n of the inner tower (tests); default: automatic
    if (const char *e = getenv("COARSE_TOWER_SUBN")) g_tower_sub = atoi(e);
    if (const char *e = getenv("COARSE_METHOD")) {
        if (!strcmp(e, "dense")) g_force_method = FM_DENSE;
        else if (!strcmp(e, "buckets")) g_force_method = FM_BUCKETS;
        else if (!strcmp(e, "logs")) g_force_method = FM_LOGS;
        else if (!strcmp(e, "tower")) g_force_method = FM_TOWER;
        else if (!strcmp(e, "ph")) g_force_method = FM_PH;
        else if (!strcmp(e, "rr")) g_force_method = FM_RR;
        else if (strcmp(e, "auto")) {
            fprintf(stderr, "COARSE_METHOD=%s: expected auto, dense, buckets, "
                    "logs, tower, ph or rr\n", e);
            return 1;
        }
    }
    if (argc >= 2 && (!strcmp(argv[1], "--selftest") ||
                      !strcmp(argv[1], "--selftest-lowmem") ||
                      !strcmp(argv[1], "--selftest-logs"))) {
        bool lowmem = !strcmp(argv[1], "--selftest-lowmem");
        bool logs = !strcmp(argv[1], "--selftest-logs");
        int force_p = 0;
        if (lowmem) {
            force_p = (argc >= 3) ? atoi(argv[2]) : 4;
            if (force_p < 1) force_p = 4;
        }
        if (logs) {
#if !defined(WIDE_FIELD)
            fprintf(stderr, "--selftest-logs needs the -DWIDE_FIELD build\n");
            return 1;
#endif
            const char *e = argc >= 3 ? argv[2] : "logs";
            g_force_method = !strcmp(e, "tower") ? FM_TOWER
                           : !strcmp(e, "ph") ? FM_PH
                           : !strcmp(e, "rr") ? FM_RR : FM_LOGS;
            g_logs_selfcheck = true;
        }
        const int logs_want = g_force_method;
        u64 r = 136279841, smax = 999999;
        int depth = 20;
        printf("selftest%s: r=%" PRIu64 " depth=%d smax=%" PRIu64
               " (GPU vs in-process CPU reference)\n",
               lowmem ? " [lowmem, forced]" : logs ? " [logs, forced]" : "",
               r, depth, smax);
        if (lowmem)
            printf("  forcing the bucketed low-memory path, %d buckets,"
                   " for every degree\n", force_p);
        if (logs)
            printf("  forcing the logs path (%s) for every degree, with an "
                   "on-device check of the logs\n",
                   g_force_method == FM_TOWER ? "tower: even d only, so the "
                   "run starts at d=4" : g_force_method == FM_PH
                   ? "Pohlig-Hellman" : g_force_method == FM_RR
                   ? "tower for even d, RR for odd d"
                   : "tower for even d, PH for odd d");
        std::vector<ull> g, c;
        if (g_force_method == FM_TOWER) {
            // tower needs even d >= 4: seed d=2,3 with the dense path first
            g_force_method = FM_DENSE;
            run_gpu(r, 3, smax, g, 0);
            g_force_method = FM_TOWER;
            for (int d = 4; d <= depth; d += 2) {
                std::vector<ull> pre = g;
                run_gpu(r, d, smax, g, 0, d, &pre);
                if (d + 1 <= depth) {         // odd degree: dense
                    g_force_method = FM_DENSE;
                    pre = g;
                    run_gpu(r, d + 1, smax, g, 0, d + 1, &pre);
                    g_force_method = FM_TOWER;
                }
            }
        } else {
            run_gpu(r, depth, smax, g, force_p);
        }
        double t0 = now_s();
        sieve_cpu(r, depth, smax, c);
        fprintf(stderr, "cpu reference: %.2fs\n", now_s() - t0);
        u64 bad = 0;
        for (u64 s = 1; s <= smax; s++)
            if (g[s] != c[s] && ++bad <= 10)
                fprintf(stderr, "MISMATCH s=%" PRIu64 " gpu=%" PRIx64
                        " cpu=%" PRIx64 "\n", s, (u64)g[s], (u64)c[s]);
        u64 wide_fails = 0;
#if defined(WIDE_FIELD)
        if (logs) {
            fprintf(stderr, "logs engines at d <= %d (device 0):\n", KEY_MAX_DEGREE);
            wide_fails = logs_wide_selftest(r, logs_want);
        }
#endif
        (void)logs_want;
        if (wide_fails)
            printf("selftest: logs engine checks at d <= %d: %" PRIu64
                   " failures\n", KEY_MAX_DEGREE, wide_fails);
        printf("selftest: %s (%" PRIu64 " mismatches)\n",
               bad || wide_fails ? "FAIL" : "PASS", bad);
        return bad || wide_fails ? 1 : 0;
    }

    if (argc < 4) {
        fprintf(stderr, "usage: %s <r> <depth> <out_prefix> [options]\n"
                        "       %s --selftest\n"
                        "       %s --selftest-lowmem [nbuckets]\n"
                        "       %s --selftest-logs [logs|tower|ph|rr]   "
                        "(wide build)\n"
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
                        "  --checkpoint <file> checkpoint file (default "
                        "<out_prefix>.ckpt): progress is\n"
                        "                      saved on SIGTERM/SIGINT and "
                        "periodically, and the same\n"
                        "                      command continues from it after "
                        "an interruption\n"
                        "  --checkpoint-mins <N>       save every N minutes "
                        "(default 15; 0 = signals only)\n"
                        "  --checkpoint-keep <G>       keep G checkpoint "
                        "generations (default 3)\n"
                        "  --resume <file>     continue from a checkpoint "
                        "(also another run's)\n"
                        "  --fresh             ignore an existing checkpoint "
                        "(kept as *.old), start over\n"
                        "  --no-checkpoint     no checkpoint file at all\n"
                        "  --no-tune           skip the per-degree timing of "
                        "kernel variants (d >= 40)\n"
                        "  --gpus <N>          use N GPUs (default: all "
                        "present; buckets split across\n"
                        "                      devices, no peer access or "
                        "NVLink needed)\n"
                        "  --seed-covers-49    --min-depth >= 50 with a seed file "
                        "from a run through 49\n"
                        "  --allow-degree-gap  --min-depth >= 50 without degree 49 "
                        "(factors found at\n"
                        "                      d >= 50 are least-degree only where "
                        "there is no degree-49\n"
                        "                      factor)\n"
                        "\n"
                        "  <out_prefix>.found.cert lists one line per s that has a "
                        "factor, in the same\n"
                        "  's d p<hexmask>' format as GF2X factor.cpp, so per-degree "
                        "certs simply\n"
                        "  concatenate into a full record.  Example incremental run:\n"
                        "     %s <r> 35 out35 --load-survivors out34.survivors.txt "
                        "--min-depth 35\n"
                        "  (interrupted? run the same command again: it "
                        "continues from out35.ckpt)\n"
                        "  (the legacy form '<r> <depth> <smax> <out_prefix>' is "
                        "still accepted)\n",
                        argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    u64 r = strtoull(argv[1], 0, 10);
    int depth = atoi(argv[2]);
    // best[] keys hold degrees up to KEY_MAX_DEGREE (see key_pack); degrees
    // above 47 exist only in the logs path (wide build)
    if (depth > KEY_MAX_DEGREE) {
        fprintf(stderr, "depth %d unsupported: the packed best[] key holds "
                "d <= %d\n", depth, KEY_MAX_DEGREE);
        return 1;
    }
#if !defined(WIDE_FIELD)
    if (depth > 32) {
        fprintf(stderr, "depth %d needs the -DWIDE_FIELD build "
                "(coarse_sieve_wide); this narrow build is u32, d <= 32\n",
                depth);
        return 1;
    }
#endif
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
    int min_depth = 2, ckpt_keep = 3, want_gpus = 0;   // 0 = use every device
    bool allow_gap = false;                      // --allow-degree-gap
    bool no_ckpt = false, fresh = false;         // --no-checkpoint, --fresh
    bool seed49 = false;                         // --seed-covers-49
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
        else if (!strcmp(argv[a], "--gpus") && a + 1 < argc) want_gpus = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--allow-degree-gap")) allow_gap = true;
        else if (!strcmp(argv[a], "--no-checkpoint")) no_ckpt = true;
        else if (!strcmp(argv[a], "--fresh")) fresh = true;
        else if (!strcmp(argv[a], "--no-tune")) g_tune = false;
        else if (!strcmp(argv[a], "--seed-covers-49")) seed49 = true;
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
    g_ckpt.min_depth = min_depth;

    // The seed image (--load wins over --load-survivors), read first: an
    // automatic resume compares its hash with the checkpoint's record.
    std::vector<ull> seed;
    int seed_kind = 0;
    const char *seed_path = nullptr;
    u64 seed_hash = 0, seed_surv = 0, seed_lines = 0, seed_oor = 0;
    if (load_path) {
        if (!read_found_image(load_path, smax, seed)) return 1;
        seed_kind = 1; seed_path = load_path;
        for (u64 s = 1; s <= smax; s++) if (seed[s] == ~0ULL) seed_surv++;
    } else if (loadsurv_path) {
        if (!read_survivors_image(loadsurv_path, smax, seed, &seed_surv,
                                  &seed_lines, &seed_oor)) return 1;
        seed_kind = 2; seed_path = loadsurv_path;
        if (seed_oor) fprintf(stderr, "warning: %" PRIu64 " survivor entries "
                              "outside 1..%" PRIu64 " ignored\n", seed_oor, smax);
    }
    if (seed_kind) seed_hash = ckpt_hash(seed.data() + 1, smax);

    // Checkpoint file: --checkpoint, else the --resume file, else (round 5)
    // <out_prefix>.ckpt unless --no-checkpoint.
    std::string auto_path;
    if (no_ckpt) {
        if (ckpt_path || resume_path) {
            fprintf(stderr, "--no-checkpoint conflicts with --checkpoint / "
                    "--resume\n");
            return 1;
        }
    } else if (!ckpt_path && !resume_path) {
        auto_path = std::string(prefix) + ".ckpt";
        ckpt_path = auto_path.c_str();
    }
    if (resume_path && !ckpt_path) ckpt_path = resume_path;
    if (ckpt_path && !ckpt_lock(ckpt_path)) return 1;
    if (fresh) {
        if (resume_path) {
            fprintf(stderr, "--fresh conflicts with --resume\n");
            return 1;
        }
        if (ckpt_path) ckpt_set_aside(ckpt_path);
    } else if (ckpt_path && !resume_path) {
        // an existing checkpoint of THIS run: continue it automatically
        CkptInfo ci;
        std::string gen;
        const int st = ckpt_find(ckpt_path, ci, gen);
        if (st < 0) {
            fprintf(stderr, "checkpoint %s exists but cannot be read.  Use "
                    "--resume %s to try its older generations, --fresh to set "
                    "it aside and start over, or --checkpoint <other file>\n",
                    ckpt_path, ckpt_path);
            return 1;
        }
        if (st > 0) {
            const std::string why = ckpt_mismatch(ci, r, smax, depth, min_depth,
                                                  seed_kind, seed_hash);
            if (!why.empty()) {
                fprintf(stderr, "checkpoint %s belongs to a different run: %s.\n"
                        "  To continue that run, use its command line or "
                        "--resume %s; to start this one over, add --fresh (the "
                        "old checkpoint is kept as *.old); or checkpoint "
                        "elsewhere with --checkpoint <file>\n", gen.c_str(),
                        why.c_str(), ckpt_path);
                return 1;
            }
            fprintf(stderr, "found checkpoint %s (%s): continuing that run "
                    "(--fresh starts over)\n", gen.c_str(), ckpt_describe(ci).c_str());
            resume_path = ckpt_path;
        }
    }

    double stored_period_s = 0;
    if (resume_path) {
        // Mid-degree resume: read best[] + (degree, bucket, nbuckets).
        if (!ckpt_read(resume_path, r, smax, preload, min_depth,
                       resume_bucket, resume_nbuckets, stored_period_s))
            return 1;
        have_preload = true;
        // the command line's seed can serve as the delta baseline
        if (seed_kind && g_ckpt.base.empty() &&
            (g_ckpt.base_kind == 0 || (g_ckpt.base_kind == seed_kind &&
                                       g_ckpt.base_hash == seed_hash))) {
            g_ckpt.base.swap(seed);
            g_ckpt.base_path = seed_path;
            g_ckpt.base_kind = seed_kind;
            g_ckpt.base_hash = seed_hash;
        }
        u64 pre_surv = 0;
        for (u64 s = 1; s <= smax; s++) if (preload[s] == ~0ULL) pre_surv++;
        fprintf(stderr, "resumed %s: %" PRIu64 " survivors carried in; "
                "continuing at degree %d, %s %d/%d, up to depth %d\n",
                resume_path, pre_surv, min_depth,
                ckpt_method(resume_nbuckets) ? "chunk" : "bucket",
                resume_bucket, ckpt_count(resume_nbuckets), depth);
    } else if (seed_kind == 1) {
        // Degree-boundary extend from a prior .found.bin.  This image is
        // also the baseline for delta checkpoints.
        preload = seed;
        have_preload = true;
        g_ckpt.base.swap(seed);
        g_ckpt.base_path = load_path;
        g_ckpt.base_kind = 1;
        g_ckpt.base_hash = seed_hash;
        fprintf(stderr, "loaded %s: %" PRIu64 " survivors carried in; "
                "sieving degrees %d..%d only\n",
                load_path, seed_surv, min_depth, depth);
    } else if (seed_kind == 2) {
        // Seed from a survivors list instead of a full .found.bin.  Only the
        // *degrees* in best[] matter for correctness when sieving degrees
        // >= min_depth: every s not listed already has a factor of degree
        // < min_depth, so any sentinel below key_floor(min_depth) blocks all new
        // writes exactly as the real key would.  The polynomial data in a
        // .found.bin is never read by the sieve -- only compared against.
        preload = seed;
        have_preload = true;
        g_ckpt.base.swap(seed);                   // baseline for delta checkpoints
        g_ckpt.base_path = loadsurv_path;
        g_ckpt.base_kind = 2;
        g_ckpt.base_hash = seed_hash;
        fprintf(stderr, "loaded %s: %" PRIu64 " survivors carried in (%" PRIu64
                " lines); sieving degrees %d..%d only\n",
                loadsurv_path, seed_surv, seed_lines, min_depth, depth);
        fprintf(stderr, "note: with --load-survivors the output records only "
                "the newly found degree-%d..%d factors; keep the earlier "
                "run's .found.cert for the lower degrees\n", min_depth, depth);
    } else if (min_depth > 2) {
        fprintf(stderr, "warning: --min-depth %d without --load/--resume means "
                "degrees 2..%d are not computed\n", min_depth, min_depth - 1);
    }

    // Degree 49 (2^49 - 1 = 127 * 4432676798593) has no Pohlig-Hellman plan;
    // since round 5 the RR engine sieves it (gf2_dlog.h), after a one-off
    // table build.  A run starting above 49 must say whether its seed file
    // covers degree 49 (--seed-covers-49) or not (--allow-degree-gap: then a
    // factor found at 50..55 is the least-degree factor only where there is
    // no degree-49 one).  The run's own --min-depth counts, not where a
    // resumed run continues.
    const int orig_min = g_ckpt.min_depth;
#if defined(WIDE_FIELD)
    if (orig_min <= 49 && depth >= 49)
        fprintf(stderr, "note: degree 49 uses the RR engine: first a table "
                "of 2^25 logs is built (minutes; kept in %s if that took "
                "over two), then the sieve\n", rr_cache_path(49).c_str());
#endif
    if (orig_min >= 50 && !allow_gap && !seed49 && !g_ckpt.gap49) {
        fprintf(stderr, "--min-depth %d: is degree 49 covered by the "
                "--load/--load-survivors file?  If it comes from a run through "
                "degree 49, add --seed-covers-49.  If not, a factor found at "
                "degree %d..%d is the least-degree factor of its trinomial only "
                "where there is no degree-49 factor; add --allow-degree-gap to "
                "sieve anyway (or run degree 49 first: --min-depth 49).\n",
                orig_min, orig_min, depth);
        return 1;
    }
    const bool gap49 = orig_min >= 50 && !seed49;
    if (gap49)
        fprintf(stderr, "note: --allow-degree-gap: degree 49 is NOT covered; "
                "factors of degree %d..%d found here are least-degree factors "
                "only where there is no degree-49 factor\n", orig_min, depth);

    // Configure checkpointing and install signal handlers.  A resumed run
    // keeps checkpointing to the same file (so it survives repeated spot
    // reclaims) unless --checkpoint names a different destination.
    if (ckpt_path) {
        g_ckpt.enabled  = true;
        g_ckpt.path     = ckpt_path;
        g_ckpt.gap49    = g_ckpt.gap49 || gap49;
        // --checkpoint-mins on the command line wins (0 = on signals only);
        // otherwise a resumed run inherits a recorded interval, and the
        // default is CKPT_DEFAULT_MINS.
        g_ckpt.period_s = ckpt_mins >= 0 ? ckpt_mins * 60.0
                        : stored_period_s > 0 ? stored_period_s
                        : CKPT_DEFAULT_MINS * 60.0;
        g_ckpt.last_write = now_s();
        g_ckpt.r = r; g_ckpt.smax = smax; g_ckpt.depth = depth;
        // the first SIGINT/SIGTERM saves and exits at the next chunk or
        // bucket boundary; a second one (SA_RESETHAND) kills at once -- the
        // checkpoint file is replaced atomically, so that is safe too
        struct sigaction sa; memset(&sa, 0, sizeof sa);
        sa.sa_handler = ckpt_signal_handler;
        sa.sa_flags = SA_RESETHAND;
        sigaction(SIGTERM, &sa, nullptr);
        sigaction(SIGINT,  &sa, nullptr);
        bool delta_ok = (g_ckpt.base.size() == smax + 1);
        fprintf(stderr, "checkpointing to %s (%s, keeping %d generation%s)",
                ckpt_path, delta_ok ? "delta vs baseline" : "full image",
                g_ckpt.keep, g_ckpt.keep == 1 ? "" : "s");
        if (g_ckpt.period_s > 0)
            fprintf(stderr, ", every %.1f min", g_ckpt.period_s / 60.0);
        fprintf(stderr, "; SIGTERM/SIGINT saves and exits (twice: exit now)\n");
        if (!delta_ok && !loadsurv_path)
            fprintf(stderr, "note: no --load baseline, so checkpoints write a "
                    "full %.0f MB image; on slow storage prefer --load\n",
                    smax * 8 / 1048576.0);
    }

    // ---- select devices ----
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
        fprintf(stderr, "using %d GPUs (buckets split across devices; results "
                "merged by minimum)\n", use);

    double t0 = now_s();
    std::vector<ull> best;
    int force_buckets = 0;
    if (const char *e = getenv("COARSE_FORCE_BUCKETS")) force_buckets = atoi(e);
    const int reached = run_gpu(r, depth, smax, best, force_buckets, min_depth,
                                have_preload ? &preload : nullptr, resume_bucket,
                                resume_nbuckets);
    fprintf(stderr, "total sieve time: %.2fs\n", now_s() - t0);
    if (reached < depth)
        fprintf(stderr, "WARNING: stopped after degree %d of %d; the outputs "
                "below cover degrees through %d only, and the checkpoint (if "
                "any) is kept.  Exit status 2.\n", reached, depth, reached);
    char covered[64];
    if (gap49) snprintf(covered, sizeof covered, "<= %d other than 49", reached);
    else snprintf(covered, sizeof covered, "<= %d", reached);

    char path[1024];
    if (write_survivors) {
        snprintf(path, sizeof path, "%s.survivors.txt", prefix);
        FILE *fs = fopen(path, "w");
        if (!fs) { perror(path); return 1; }
        u64 nsurv = 0;
        for (u64 s = 1; s <= smax; s++)
            if (best[s] == ~0ULL) { fprintf(fs, "%" PRIu64 "\n", s); nsurv++; }
        fclose(fs);
        fprintf(stderr, "%" PRIu64 " survivors (no factor of degree %s) -> %s\n",
                nsurv, covered, path);
    } else {
        u64 nsurv = 0;
        for (u64 s = 1; s <= smax; s++) if (best[s] == ~0ULL) nsurv++;
        fprintf(stderr, "%" PRIu64 " survivors (no factor of degree %s) "
                "[survivors file suppressed via --no-survivors]\n", nsurv, covered);
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
            unsigned dd = (unsigned)key_degree(b);
            u64 mask = key_mask(b);
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
    if (reached < depth) return 2;
    // Run finished normally: drop the checkpoint so it can't be resumed by
    // mistake later.
    if (g_ckpt.enabled && g_ckpt.path) {
        remove(g_ckpt.path);
        char gen[1100];
        for (int i = 1; i < 8; i++) {            // and every rotated generation
            snprintf(gen, sizeof gen, "%s.%d", g_ckpt.path, i);
            remove(gen);
        }
    }
    return 0;
}

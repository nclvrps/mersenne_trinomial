# coarse_sieve (-DWIDE_FIELD) -- review round 1: diagnosis, fix, roadmap

Baseline reviewed: `multi-gpu` branch, commit 1427bd4 (2026-07-25),
`coarse_sieve.cu` + `gf2_wide_field.h` + `gf2_wide_setup.h`.
Deliverables: `coarse_sieve.cu` (patched), `coarse_sieve_round1.patch`
(the same change as a git patch against 1427bd4 -- use it if your
production copy has moved on), `proto_tower_dlp.cpp`, `proto_ph_dlp.cpp`
(validated reference prototypes for round 2).  No other file changes.

## TL;DR

Low-memory mode spent essentially all of its time recomputing each
antilog g^i from scratch -- about d/2 bit-serial field multiplies,
~20,000 GPU instructions per element -- for the whole 2^d domain, once
per bucket.  Since g = x for the primitive modulus, g^(i+1) = x * g^i is
one shift plus one conditional XOR.  Switching the sweep to that LFSR
stepping is ~1000x cheaper per element; projected d=36 on the RTX 2070
SUPER: ~20-40 s instead of 12.3 h; d=37 ~1.5-2 min instead of ~2 days.
Outputs are unchanged (byte-identical in every A/B run).

VRAM still sets the bucket count (time ~ 1/VRAM), but at this cost per
bucket an 8 GB card reaches d = 40-41 in hours.  Beyond that, the
bucketing itself can go: discrete logs in GF(2^d) do not need a 2^d
table (tower decomposition for even d, Pohlig-Hellman for most d), both
validated here; that is round 2.  More CPU RAM / SSD / HDD would not
help, and hand-written PTX is not where the wins are (details below).

## 1. Diagnosis: where the time went

Evidence from the d=33..36 runs in `coarse_sieve.log` (2070 SUPER,
6.35 GiB free):

    d    buckets   total      per bucket
    33     16      593 s        37 s
    34     32     2494 s        78 s
    36    128    44324 s       346 s

* Per-bucket time scales exactly with 2^d (x2.1 per degree), while the
  slice size per bucket is constant (~5.4e8 slots at fixed VRAM).  So
  the cost is in the one kernel that walks the WHOLE domain each bucket:
  `k_build_lslice`.  `k_resolve_slice` (per-slot work) is negligible.
* Buckets double per degree too, hence the observed ~4.2x per degree:
  total work ~ nbuckets * 2^d ~ 4^d / VRAM.
* 346 s / 2^36 = 5.04 ns per element per bucket ~ 23,000 integer
  instructions on this card -- matching `d_antilog(i)`: popcount(i)
  ~ 18 calls to `gf_mul`, each a bit-serial 64x64 carry-less multiply
  (four 32-iteration loops) plus a 35-iteration bit-serial reduction,
  on 32-bit ALUs.
* CPU microbenchmark at d=36 (same code): `d_antilog` 2021 ns/element vs
  `gf_mulx` stepping 2.2 ns/element (903x), stepping == recompute.
  (`gf_mulx` already existed in `gf2_wide_field.h`, documented as "the
  antilog chain when the generator is x", but no kernel used it.)

## 2. What changed (coarse_sieve.cu only)

1. `k_build_lslice` and `k_antilog`: each thread owns runs of
   consecutive exponents (up to ANTILOG_RUN_MAX = 16384): one
   `d_antilog()` seed per run, then `gf_mulx` per element.  Run length
   adapts so small d still has parallelism (`antilog_run_len`).
   THE fix: ~1000x on the dominant kernel.
2. `pick_buckets`: smallest bucket count that fits, not the smallest
   power of two.  d=36 on your card needs 84, previously used 128;
   every bucket is a full sweep, so this is up to ~2x by itself.
3. `mulmod_mersenne`: (a*b) mod (2^d - 1) by folding at bit d
   (~20 instructions) replaces the ~36-iteration `mulmod_u64` loop for
   j = u * r^-1 mod M, which ran once per slice slot.  Tested against
   `mulmod_u64` for every d = 2..63 (12.4M cases incl. edge operands).
4. Coset-minimality test before the subfield test (it rejects (d-1)/d
   of slots after ~2 rotations, before any 64-bit modulo).
5. `progression_open`: skip the d^2-multiply minimal polynomial when no
   s on the progression can still be lowered (best[s] < d<<48).  Exact
   because best[] only decreases; a stale read only costs extra work.
   With `--load-survivors` ~95% of in-range hits land on non-survivors.
   The dense `k_classes` also computed the minimal polynomial for EVERY
   class before the range test (a fraction ~smax/M reaches s <= smax:
   ~1 in 8 at d=29, ~1 in 1000 at d=36); it now runs only for in-range,
   still-improvable hits.
6. Per-thread survivor counts with one atomic per thread, instead of
   one atomicAdd per class (2^d/d of them) on a single address.
7. Guards: depth > 47 is refused (the packed key (d<<48)|mask needs
   d+1 mask bits, so d = 48 spills into the degree field and silently
   corrupts both the ordering and the decoding); the narrow build
   refuses depth > 32 (it used to accept it and overflow u32).
   Header comments updated (the old "32-bit counter, d <= 37" note was
   obsolete; the real ceiling is the key format).

## 3. Validation done here (functional CUDA emulation, no GPU)

* `--selftest` and `--selftest-lowmem N` for N = 1, 3, 4, 7, 16, both
  narrow and wide builds: all PASS, 0 mismatches vs the in-process CPU
  reference (depth 20, s <= 999,999).
* A/B against the ORIGINAL binary, byte-for-byte:
  - scratch -> depth 22, 4 buckets: found.bin, cert, survivors
    identical (920,820 factors); 24.4 s -> 1.3 s emulated
  - `--load-survivors` (depth-18 list) degrees 19..23, 8 buckets:
    cert + survivors identical; 93.7 s -> 2.9 s emulated
  - degrees 19..22 of the incremental run identical to the from-scratch
    run's lines
* Automatic bucket counts (emulated 20 MB free): picks 1, 2, 4, 7 at
  d = 21..24; output identical to forced 4 buckets.
* Checkpoint stop after 3 and after 5 buckets (test hook) + `--resume`:
  identical to the original binary's uninterrupted run.
* `git apply` of the patch on 1427bd4 is clean and reproduces the file.
* Not testable here: real GPU timing, `__umul64hi` device path (the
  host path uses a portable 32-bit-halves product, tested above).

## 4. Hardware test plan (in this order)

    make coarse_sieve coarse_sieve_wide          # Makefile unchanged
    ./coarse_sieve --selftest
    ./coarse_sieve_wide --selftest
    ./coarse_sieve_wide --selftest-lowmem 7      # non-power-of-two path
    ./coarse_sieve_wide --selftest-lowmem 16

Regression + calibration against your existing certs (expect identical
files; note the times):

    ./coarse_sieve_wide 136279841 33 new33 --load-survivors out32.survivors.txt --min-depth 33
    cmp new33.found.cert out33.found.cert        # was 593 s; expect ~1-2 s
    ./coarse_sieve_wide 136279841 34 new34 --load-survivors out33.survivors.txt --min-depth 34
    cmp new34.found.cert out34.found.cert        # was 2494 s; expect ~2-5 s
    # optional: d=36 (was 12.3 h; expect well under a minute) and d=37

Then production d = 38, 39, 40 (each from the previous survivors list,
with `--checkpoint` as before).  Projections below are a cost model
calibrated to your measured 5.04 ns; the d=33/34 timings will pin the
real constant -- please send them.

    d   old (buckets)        new (buckets)
    36  12.3 h  (128)        ~20-40 s    (84)
    37  ~2 days (256)        ~1.5-2 min  (168)
    38  ~8 days              ~5-9 min    (336)
    39  ~33 days             ~20-35 min  (672)
    40  ~4 months            ~1.3-2.3 h  (1343)
    41  ~1.4 years           ~5-9 h      (2686)
    42                       ~1-1.5 days (5372)
    43                       ~4-6 days   -- use round 2 instead

Note: with `--checkpoint`, each bucket also copies best[] (545 MB) to
the host for the snapshot, ~45 ms on PCIe 3.0: ~15% at d=36, negligible
from d=38 up.  Not worth changing now.

## 5. Your questions

### Would large CPU RAM, SSD or HDD help?  No.

The only big object is the discrete-log table, and the GPU now
regenerates its contents (by LFSR stepping) at roughly 2-3e11 elements
per second.  Streaming a stored table instead: PCIe 3.0 x16 ~12 GB/s
= ~1.5e9 eight-byte entries/s (~150x slower), NVMe ~0.5-1e9/s, HDD
~2.5e7/s.  A precomputed table would also have to be BUILT first, which
needs the same value-ordered scatter that the buckets exist for (or
2^d * 8 B of RAM: 550 GB at d=36, 2.2 TB at d=38).  Recomputing beats
every storage tier, and the gap widens with the round-2 options.  (The
Zech table depends only on d, not on r -- only relevant if you ever
sieve several r.)

### Would hand-written PTX help?  Not meaningfully.

After the fix the hot loop is ~15 integer instructions per element
(64-bit shift, conditional XOR, range test) and nvcc already emits the
right SASS for that (funnel shifts SHF, LOP3, IADD3).  ptxas re-
schedules and re-allocates whatever PTX you write, so hand-PTX buys at
most a few percent.  The large "low-level" wins are algorithmic and
expressible in plain CUDA C++:
* bitslicing the sweep (32 elements per register bit-plane): the LFSR
  step becomes ~3 XORs per 32 elements and the bucket test ~log2(B)
  LOP3s per 32 elements -- est. 10-20x on the sweep term;
* templating the sweep on d (shift counts become immediates), a few %;
* keeping 64-bit division/modulo out of inner loops (GPUs emulate them
  in software, ~100 instructions each) -- done for the per-slot path.

### Where does this leave the crossover?

After d=37 there are ~3.23M survivors.  Each further degree eliminates
about survivors/d (~80-85k at d = 38-40; d=37 actually gave 88,522),
and each eliminated s saves one `factor` first-interval run, ~108 CPU-s
(your 1962620 log: 6.3 s exponentiation + 35.5 s squarings + 66.0 s
gcd).  That is ~2,500 CPU-hours per degree -- ~17 days of a 6-core box,
or ~$26 at your Turin estimate -- against minutes to hours of GPU time
for d = 38-41.  The bucketed sieve now pays off clearly through ~42.

## 6. Round 2 roadmap: make VRAM irrelevant

The bucketing exists only because the method builds a full 2^d-entry
log table.  The sieve does not need one: per class it needs one
discrete log, log(g^(jr) + 1), and both routes below give those from
small tables in O(2^d) total work per degree instead of
O(4^d / VRAM).  Both are validated by the included prototypes.

A. Tower decomposition (even d).  GF(2^d) = K(w), K = GF(2^(d/2)); every
   element with b != 0 is a + b*w = b*(c + w), c = a/b, so
       log e = (2^m + 1) * logK(b) + T[c],   T[c] = log(c + w),
   from tables of 2^(d/2) entries (the split e -> (a, b) is GF(2)-linear:
   ceil(d/8) byte-table lookups).  `proto_tower_dlp`: exhaustive
   0 errors at d = 12..24; d=36: 300,000 random logs and Zech logs,
   0 errors, 4 MB of tables (dense: 550 GB), built in 0.1 s.
   Cubic towers handle d = 3m with 2^(2d/3) tables (39: 512 MB).

B. Pohlig-Hellman (most d).  Tables only as large as the largest prime
   factor of 2^d - 1, which for the degrees ahead is small:
       38: 524287   39: 121369   40: 61681    41: 164511353 (1.3 GB)
       42: 5419     43: 2099863  44: 2113     45: 23311
       46: 2796203  47: 13264529
   (37: 616318177 -- done already, and ~2 min bucketed anyway.)
   `proto_ph_dlp`: d = 42 (incl. prime powers 3^2, 7^2), 43, 45 --
   2,000 random logs each, 0 wrong.

Design sketch (one new lowmem-free path): sweep u = 0..M-1 by LFSR runs
(as now), keep j = u * r^-1 mod M incrementally (j += r^-1 mod M),
coset-minimality test, and for the ~1/d surviving classes compute
v = log(e ^ 1) by A or B, then the existing gcd / s0 / progression code.
Memory: best[] + tables, no slices, no buckets.  Needs a faster GPU
gf_mul for B (4-bit-window clmul + sparse-modulus folding, est. ~200
instructions vs ~1400 now; squarings are GF(2)-linear -> byte tables).
Model estimates on the 2070 SUPER: PH ~0.5-15 min per degree for
38-43, ~1-3.5 h for 44-47 (~10 h for all of 38..47); tower cuts the
even degrees to minutes.  Going 37 -> 47 would remove ~1 - 37/47 = 21%
of the survivors (~690k s, ~20,000 CPU-hours of `factor`).

C. Bitsliced sweep for the bucketed path (Section 5): 10-20x on the
   sweep, keeps the current architecture; a cheaper intermediate step
   if round 2 is deferred.

## 7. Known limits and small issues

* Depth ceiling 47 (packed key) -- now enforced.  Widening, e.g.
  (d << 56) | mask, is a small change if round 2 makes d > 47 cheap.
* A minimal polynomial with a non-GF(2) coefficient returns mask 0 and
  would be written as "s d p0"; "impossible" by construction and
  check-ntl would reject it, but there is no in-kernel guard.
* `gcd_u64` / `inv_mod` use 64-bit division (~2-4k instructions per
  surviving class); fine now (~1 s at d=36), worth a binary-GCD or
  batch-inversion rewrite only in round 2.

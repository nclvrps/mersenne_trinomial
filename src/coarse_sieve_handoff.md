# coarse_sieve (-DWIDE_FIELD) -- round 2: discrete logs instead of buckets

Baseline: the round-1 `coarse_sieve.cu` (2026-10-18), which you validated
on hardware (selftests, d = 33..41, oracle data for smaller Mersenne r).

Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | round 2 (replaces round 1's) |
| `gf2_dlog.h` | NEW -- fast GF(2^d) arithmetic, necklace test, tower and Pohlig-Hellman log engines; put it next to `gf2_wide_field.h` (common/) |
| `gf2_dlog_selftest.cpp` | NEW -- host self-test of everything in `gf2_dlog.h` (sieve/ or anywhere on the vpath) |
| `Makefile` | `coarse_sieve_wide` depends on `gf2_dlog.h`; new targets `gf2_dlog_selftest`, `emul-coarse_sieve_wide` |
| `cuda_emul.h` | your version plus what coarse_sieve needs (`cudaStreamQuery`, events with flags, `atomicCAS`, optional `EMUL_NDEV` / `EMUL_FREE_MB` hooks); backward compatible |
| `coarse_sieve_round2.patch` | the same changes as a patch against round 1 (flat paths, like the round-1 patch; if your tree is sieve/ + common/, just copy the files) |

## TL;DR

* New **logs path** in the wide build.  Per degree: one sweep over the
  class representatives, and for each class ONE discrete log
  Z(u) = log(g^u + 1), computed directly from small tables -- tower
  decomposition for even d (2^(d/2)-entry tables, 128 MB at d=46),
  Pohlig-Hellman for odd d (<= 768 MB at d=47).  No 2^d table, no
  buckets: time grows ~2x per degree instead of ~4x, and VRAM no longer
  sets the speed.
* It is chosen automatically whenever the dense tables don't fit (on
  your card: d >= 29).  `COARSE_METHOD=buckets` gives round 1 exactly.
* Output is byte-identical to round 1 in every comparison made here
  (emulated: from scratch to d=28, then d = 29..32 incrementally with the
  r=136279841 survivor chain, plus the round-1 regression set; engine
  checks at every d up to 47).
* Projected on the 2070 SUPER: d=41 ~3-5 min (round 1: ~5 h), d=42 ~1-2
  min, ..., d=47 ~3-4.5 h; all of 42..47 in ~4.5-7 h, versus ~3 years
  bucketed (table in section 5).
* Your auto-checkpoint idea is deferred to round 3, as agreed (sketch in
  section 7).

## 1. How the logs path works

For the primitive modulus q of degree d, g = x, M = 2^d - 1, r^-1 mod M:

1. **Sweep** u over the odd integers below 2^(d-1), in runs per thread,
   with g^u by LFSR stepping (two `gf_mulx` per odd step).  Every
   cyclotomic class has exactly one coset-minimal representative (the
   least rotation of its d-bit string), and it is odd and < 2^(d-1).
   Any representative gives the same factor and progression: with
   u' = 2^i u, Z(u') = 2^i Z(u) and 2 is a unit mod M.
2. **Necklace test** (is u its own least rotation?): a rotation smaller
   than u must start in a zero run at least as long as u's leading run of
   z zeros; runs are found with log2(z) shift-ANDs, and only the few
   other runs of exactly z zeros need an explicit comparison.  ~50
   instructions instead of a data-dependent loop over all d-1 rotations
   (which kept warps busy ~30 iterations per u).  Subfield classes are
   rejected afterwards by exact-division tests.
3. **Zech log** v = log(g^u + 1):
   * tower (even d = 2m): GF(2^d) = K(w), K = GF(2^m), w = x; every
     element with b != 0 is a + b w = b (c + w), c = a/b, so
     log = (2^m + 1) log_K(b) + T[c].  The split e -> (a, b) is
     GF(2)-linear (ceil(d/8) byte tables); log_K / antilog_K and T are
     2^m-entry tables.  T is filled on the GPU by a coupon-collector
     sweep and checked complete.
   * Pohlig-Hellman (odd d, works for any d): M = P_1 * P_2 split into
     the most balanced coprime halves; x^(E_i) lies in the order-P_i
     subgroup, an open-addressing hash table gives its log mod P_i, CRT
     combines.  E_i = c_i M/P_i is chosen (over c_i and rotations) with
     few set bits in a short window, so one shared squaring chain plus
     ~19 multiplies does both parts (d=47: chain 27, popcounts 10 + 11).
     A part too big for the table budget runs baby-step/giant-step:
     d = 31 (M prime), 37 (616318177) and 41 (164511353, G = 2 on your
     card); every d >= 42 fits whole (largest part 13264529, at d=47).
4. **Progression**: j = u r^-1, gg = gcd(j, M) from M's factorization
   (exact-division tests, no 64-bit division), s0 = v * j^-1 mod M with
   j^-1 = j^(lambda(M)-1) (Carmichael exponent: 31 bits at d=47 instead
   of 47; 16 at d=42).  Range test and `progression_open` as in round 1;
   the minimal polynomial (d^2 work) only for in-range, improvable hits.
   Marks go into the same best[] by atomicMin, so results do not depend
   on order, chunking or device count.
5. **Fast arithmetic** (`gf2_dlog.h`): 32x32 carry-less products from
   integer multiplies on 4-bit-spaced bit classes (BearSSL style), one
   Karatsuba level to 64x64, two-fold reduction by the sparse low part of
   q (2-6 terms for every d <= 47), squaring by bit-spreading.  ~8x
   cheaper than the bit-serial `gf_mul` the other paths use.

**Chunks.**  The sweep is cut into kernel launches of 2^17 thread-runs
(d=47: 32768 chunks of 2^31 u-values).  Chunk geometry depends only on d,
so a checkpoint's chunk index is stable.  Representatives cluster at
small u (the least rotation starts with the longest zero run), so the
first chunks of a degree are ~12x denser than average: the progress dots
come slowly at first and quickly at the end.  The host waits on
blocking-sync events (no polling gap between chunks, no busy CPU core).

**Safety checks.**  Every log is range-checked and every 256th one per
thread is re-verified by exponentiation (g^v == g^u + 1); the tower table
must be complete; an impossible minimal polynomial is counted too.  The
count is read after every round of chunks, and any nonzero count stops
the run ("discrete-log consistency errors ... NOT trustworthy", exit 1),
so neither the output nor a checkpoint can include such a chunk.  The
sampling catches anything systematic (a bug, a bad table, a flaky card)
within the first chunk; a lone random bit flip in an unsampled log can
still produce one false factor, which check-ntl would reject.  Test hook:
`COARSE_TEST_FAULT_EVERY=n` corrupts the log of every class with
u = 1 (mod n); with n=3 at d=19 the run stops after the first chunk
(298 errors caught).

## 2. What changed

`gf2_dlog.h` (new): the arithmetic above, `gf2f_is_min_rotation` (+ the
all-rotations reference), Mersenne mulmod/powmod, exact-division tests,
`TowerDev`/`tower_log`, `PHDev`/`ph_log`, host-side table builders and the
PH planner (`ph_plan_host`: coprime split, exponent search, table sizes
within budget).

`coarse_sieve.cu`:
* kernels `k_logs_sweep`, `k_tower_T`, `k_ph_build`, `k_dlog_check`,
  `k_rep_check`; host driver `sieve_degree_gpu_logs` (setup per device,
  chunk rounds across devices, checkpoint/stop at chunk boundaries,
  heavy-progression drain as before, error abort);
* method selection in `run_gpu`: dense if its tables fit, else logs
  (tower for even d <= 46, PH otherwise), else buckets;
  `COARSE_METHOD=auto|dense|buckets|logs|tower|ph` overrides;
* checkpoints: same file format; the bucket-count field now carries a
  method tag in bits 24+ (old round-1 checkpoints decode unchanged and
  resume in the bucketed path; a logs checkpoint resumes in the logs path
  and checks the chunk count);
* logs-table budget: free VRAM minus the 256 MB reserve and
  min(free/8, 512 MB) slack; `COARSE_LOGS_MAX_MB` caps it further
  (smaller only means more giant steps for the PH parts that don't fit);
* `--selftest-logs [logs|tower|ph]` (below);
* bucket path: unchanged except the poll back-off (1 ms doubling to
  128 ms instead of a flat 100 ms; short buckets were padded out);
* header comments: Env / Sizing / Selftest / new "LOGS MODE" section.

## 3. Validation done here

No GPU here; everything ran under the functional CUDA emulation
(`cuda_emul.h`), plus host unit tests of the same code.

* `gf2_dlog_selftest` (host): fast multiply / square / antilog vs the
  bit-serial reference at every d = 2..47, Mersenne mulmod, inverses via
  phi and via lambda, exact division: 3.5M checks, 0 mismatches.
  Necklace test vs all rotations: exhaustive for d <= 22 (and the
  representative count equals the necklace formula), random u with spread
  leading-zero counts to d=47 -- 100M checks in a separate run, 0
  mismatches.  Both log engines: every element for d <= 20 (separately
  every element to d=24), 20,000-200,000 random elements per degree to
  d=47, incl. BSGS at 31/37/41 (G up to 32): 0 wrong.
* Full program, emulated, A/B against the round-1 binary (byte-for-byte
  `found.bin` / cert / survivors):
  - from scratch to d=22 through the logs path, 128 chunks at d=22;
  - `--load-survivors` d = 19..23; stop mid-degree (after 37 chunks) +
    `--resume`; automatic dense -> logs switch (emulated 400 MB card);
    2 emulated GPUs; forced buckets; a round-1 checkpoint resumed by the
    round-2 binary: all identical.
  - fault injection (above): the run stops with exit 1 and no output.
  - deep chain, r = 136279841, s <= 999,999: logs from scratch to d=28
    vs round 1 (937,442 factors), then d = 29 (PH), 30 (tower), 31 (PH,
    M prime, BSGS), 32 (tower) incrementally vs round-1 buckets:
    certs and survivor lists identical at every degree, and the class
    counts printed by both paths agree.
* All selftests PASS: `--selftest`, `--selftest-lowmem 7`,
  `--selftest-logs` (logs / tower / ph) in the wide build, `--selftest`
  and `--selftest-lowmem 3` in the narrow build.  The `--selftest-logs`
  engine checks ran in emulation for every d up to 47 (512 exponents
  per engine, with tables built by the device kernels): 0 wrong.
* Compile check: no nvcc here, but both builds compile as CUDA (device
  and host passes) with clang 18 against the CUDA 12.9 headers, and the
  device code assembles with NVIDIA's ptxas for sm_60/75/80/86, no
  warnings: `k_logs_sweep` 80 registers, no spills (nvcc's allocation
  will differ a little).
* Not testable here: GPU timing, nvcc itself, real multi-GPU.

## 4. Hardware test plan

    make coarse_sieve coarse_sieve_wide gf2_dlog_selftest
    ./gf2_dlog_selftest                    # CPU, ~1 min: "ALL OK"
    ./coarse_sieve --selftest
    ./coarse_sieve_wide --selftest
    ./coarse_sieve_wide --selftest-lowmem 7
    ./coarse_sieve_wide --selftest-logs          # + engine checks to d=47
    ./coarse_sieve_wide --selftest-logs tower
    ./coarse_sieve_wide --selftest-logs ph

Each prints `selftest: PASS`; the `--selftest-logs` runs also print one
line per engine and degree, e.g.
`[d=47 PH    logs,  768 MB tables: 1048576 checked, 0 wrong, ...]`.

Regression against your round-1 certs (logs is now the automatic choice
at these degrees; adjust names to your files):

    for d in $(seq 33 41); do
      ./coarse_sieve_wide 136279841 $d r2_$d \
          --load-survivors out$((d-1)).survivors.txt --min-depth $d
      cmp r2_$d.found.cert out$d.found.cert && \
      cmp r2_$d.survivors.txt out$d.survivors.txt && echo "d=$d identical"
    done

Expect seconds per degree through d=40 (about a minute at d=39) and a
few minutes at d=41: the whole loop ~10 min, vs ~7.5 h bucketed.  The
same loop with `COARSE_METHOD=buckets` reproduces round 1 if you ever
want an A/B.  Your oracle runs for the smaller Mersenne r are another
cheap cross-check.

Production, one degree at a time as before:

    ./coarse_sieve_wide 136279841 42 out42 --load-survivors out41.survivors.txt \
        --min-depth 42 --checkpoint out42.ckpt --checkpoint-mins 20

and so on to 47 (or one run `... 47 out47 ... --min-depth 42` for all
six; the cert then holds degrees 42..47).  The first lines show the
engine and tables, e.g. `d=42 logs (tower, 32 MB tables, built in 0.4s),
1024 chunks:`.  Please send the `done` lines for d=41..43: they pin the
real per-class cost, and d+2 then takes ~4x as long as d.

## 5. Projections (RTX 2070 SUPER)

From an instruction-count model of the compiled kernels (PTX weighted
for 64-bit ops, 4.5e12 issue slots/s, 65-100% efficiency); the round-1
column is your measurements up to d=40 and the calibrated model above.

| d | engine | classes | round 2 (est.) | round 1 (buckets) |
|---|---|---|---|---|
| 33-36 | PH / tower | 0.26-1.9e9 | ~1-4 s each | 9-59 s |
| 37 | PH (BSGS) | 3.7e9 | ~15-25 s | 146 s |
| 38 | tower | 7.2e9 | ~5-10 s | 422 s |
| 39 | PH | 1.4e10 | ~1 min | 25 min |
| 40 | tower | 2.7e10 | ~20-30 s | 1.5 h |
| 41 | PH (BSGS) | 5.4e10 | ~3-5 min | ~5 h |
| 42 | tower | 1.0e11 | ~1-2 min | ~20 h |
| 43 | PH | 2.0e11 | ~13-20 min | ~3.3 days |
| 44 | tower | 4.0e11 | ~5-8 min | ~13 days |
| 45 | PH | 7.8e11 | ~50-75 min | ~7 weeks |
| 46 | tower | 1.5e12 | ~25-40 min | ~7 months |
| 47 | PH | 3.0e12 | ~3-4.5 h | ~2.3 years |

Plus ~1-5 s of table setup per degree.  Odd degrees cost more: a PH log
is ~12,000 instructions (27 squarings, 19 multiplies, 2 lookups) against
~200 plus a few table reads for the tower.  The per-class inverse
(~3,000) and the sweep (~70 per odd u) are the rest.  Sieving 42..47
removes about 1 - 41/47 = 13% of the survivors (~370k s), i.e. roughly
11,000 CPU-hours of `factor` first-interval work at ~108 CPU-s each.

## 6. Operational notes

* Memory: best[] (545 MB at full smax) + tables: tower 16 * 2^(d/2) bytes
  (32 MB at d=42, 128 MB at d=46); PH 192-768 MB for d = 43..47, up to
  the budget for d = 31/37/41.  Nothing else scales with d.
* Stop / checkpoint: SIGINT/SIGTERM are honoured at the next chunk
  boundary -- at most a few seconds at d=47 (the densest chunks come
  first), within the ~30 s spot-reclaim budget.  Checkpoints are written
  at chunk boundaries; resume with `--resume <file>` exactly as before.
  A logs checkpoint must be resumed by a wide build of this round (the
  narrow build says so).
* Multi-GPU: chunks go round-robin like buckets; each device builds its
  own tables (sized for the smallest card).
* "survivors" in the `done` line is, as in round 1, the number of degree-d
  classes examined (e.g. 69,273,666 at d=31); it must match between paths.
* Depth ceiling stays 47 (packed key, and the fast arithmetic's two-fold
  reduction).

## 7. Round 3 candidates (not done)

1. **Auto-checkpoint** (your PS): default `<out_prefix>.ckpt`, default
   interval 15 min, and if that file exists at start, resume from it
   automatically.  To be safe it needs the run's identity in the
   checkpoint (r, depth, min-depth, smax, a hash of the loaded survivors
   / found file) so a stale file from a different command line is refused
   with a clear message instead of being resumed; plus `--no-checkpoint`,
   and removing (or marking complete) the file after a successful run.
   Small change; checkpoints are ~1 MB, so 15 min costs nothing.
2. Batch (Montgomery) inversion of j, ~8 classes per thread: ~-40% on the
   tower degrees (the inverse dominates there), ~-15% on PH degrees.
3. Bit-sliced PH exponentiation (32 classes per thread as bit-planes):
   squarings become nearly free and multiplies ~4x cheaper; est. 2-3x on
   odd d (d=47 ~1.5 h).  Register pressure is the risk; a real rewrite.
4. d > 47: needs a wider packed key and a three-fold reduction; each
   extra degree ~2x the previous and removes ~1/d of the survivors.

## Appendix: round 1 (2026-10-18), for reference

The bucketed low-memory path spent essentially all its time recomputing
each antilog g^i from scratch (~d/2 bit-serial field multiplies, ~20,000
GPU instructions per element) for the whole 2^d domain, once per bucket.
Round 1 replaced that with LFSR stepping (g = x: one shift + conditional
XOR), ~1000x cheaper, and added: smallest-fitting (not power-of-two)
bucket counts; Mersenne folding mulmod; coset test before the subfield
test; `progression_open` (skip the minimal polynomial when nothing on
the progression can still improve); per-thread survivor counters; depth
guards (47 wide, 32 narrow).  Byte-identical output in all A/B runs.

Your hardware results: selftests PASS; d = 33..37 certs identical to the
earlier version's; d = 38..41 with no false positives / negatives against
`factor` ranges; oracle data for smaller Mersenne r matched.  Times:
d=33 8.7 s, 34 17 s, 35 34 s, 36 59 s, 37 146 s, 38 422 s, 39 1525 s,
40 5398 s (2.96 ps per element per bucket + ~0.76 s fixed per bucket --
the per-bucket slice resolve, which is what made d = 33/34 slower than
estimated).

Round-1 answers that still stand: more CPU RAM / SSD / HDD would not
help (recomputing beats every storage tier, and round 2 needs no big
table at all); hand-written PTX would not either (ptxas re-schedules
anyway; the wins are algorithmic).

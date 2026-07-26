# tsfactor: synthesis of factor.cpp + trinomial_scan — handoff document

Round 1 of the "synthesize factor + trinomial_scan" effort.  This file
is the durable cross-session reference; update it every round.

Pipeline context (unchanged):

    coarse_sieve_wide (GPU, all factors of degree <= 36/37 found)
        -> survivors file (s values, one per line)
    tsfactor (GPU+CPU, THIS PROGRAM)          [replaces trinomial_scan]
        -> results file: "s d p<hex>" | "s u"
    factor.cpp (CPU, gf2x/NTL)  for anything tsfactor punts on

Target: r = 136279841, s in [1, floor(r/2)], smallest-degree factor of
x^r + x^s + 1 over GF(2), ties broken by numerically least mask.
Reference hardware: RTX 2070 SUPER (8 GB, CC 7.5, ~448 GB/s), 6-core
Zen 2, and cloud T4s; future: community consumer GPUs.

---------------------------------------------------------------------------
## Part A. What factor.cpp actually does (decoded from source)

This is the information the earlier chat lacked, and it changes the
design completely.  Source: gf2x/apps/factor.cpp (v2.01, GPL) +
halfgcd.cpp, now in hand and built successfully in the sandbox
(`apt install libntl-dev libgf2x-dev libgmp-dev`, then
Makefile_gf2x_apps_factor).

### A1. The double-blocking algorithm ("multi-level blocking DDF",
Brent-Zimmermann 2008, implemented in the k > k0 branch)

Per candidate-degree block of m consecutive degrees [i, i+m-1], factor
does NOT compute one Frobenius iterate + one modmul per degree the way
trinomial_scan does.  Instead it maintains an **h-table of m residues**

    h[j] = e_{j+1}( y_0, ..., y_{m-1} ),   y_t = x^(2^(i+t)) mod T,

the elementary symmetric functions of the m Frobenius iterates covering
the block.  Then:

1. **Block polynomial by Horner, no FFT muls:**
       C = x^m + h[0] x^(m-1) + ... + h[m-1]
         = prod_{t=0}^{m-1} (x + x^(2^(i+t)))  (mod T)
   built with m mul-by-x-mod-T (1-bit shift + 2-bit fixup, linear) and
   m XORs.  An irreducible f, deg f = d, divides (x + x^(2^c)) iff
   d | c, so C captures exactly degrees [i, i+m-1].
2. **Advance the whole block by m degrees with squarings only:**
   y_t -> y_t^(2^m) is Frobenius, and e_k(y^(2^m)) = e_k(y)^(2^m), so
       h[j] <- h[j]^(2^m)   (m fastsqr each, m^2 per block).
3. **One FFT-sized modmul per block** to fold C into the interval
   accumulator A <- A*C mod T (factor materializes q block C's and
   multiplies them with an OpenMP product tree; the count is the same:
   q big muls per interval of q*m degrees).
4. One FastGCD(A, T) per interval (interval = q blocks; q grows
   linearly, q_j = j*q0 with -f 1), then CanZass on the (small)
   interval gcd to extract the least-degree, lex-least factor.
   "Lex-least" = numerically least mask; the flip machinery only
   compensates for factor's internal reciprocal-trinomial trick.

**Per-degree cost = m squarings + 1/m modmuls** (+ trivial linear ops
+ amortized GCD).  fastsqr (BLZ 2003 in-place reduce+interleave, needs
r,s odd, ~60 MB traffic) is ~12 ms at r=136279841 on Zen 2; gf2x's
modmul is ~0.9 s there.  With Gord's -m 14 -q 15: 14*12ms + 0.9/14 =
0.168 + 0.064 ~= 0.233 s/degree, matching the production logs
(0.17-0.26 s/term; first-interval 0.169 = squarings only, later
intervals ~0.21-0.24 as the modmul share stabilizes).  m_opt =
sqrt(t_mul/t_sqr) ~= 8.7 on that CPU; factor's own -m auto-calibration
does exactly this measurement (see choose_m in main()).

### A2. Other facts worth keeping

- **Exponentiation** (the 6.28 s line) = init_h: Pascal-triangle build
  of the m elementary symmetric functions of (x^(2^0..2^(m-1))), then
  k*m squarings to shift to the first block.  m(m-1)/2 + k*m fastsqr.
- **Interval schedule:** q_j = j*q0 (-f 1), interval length q_j*m,
  memory = q * r/8 bytes because factor stores all q block C's for the
  OpenMP tree (this is what -z limits: -z 720 ~ 16 GB).  We accumulate
  incrementally on GPU, so this memory wall does not exist for us.
- **-z q1:** if the next interval's q would exceed q1, print "s u" and
  stop.  Gord wants an additional -Z q1 = cap q, keep scanning.
- **fineDDF:** if an interval gcd has degree >= 2*r^(2/3) (~530k for
  our r) and >= 2k and q > 1, factor bisects the interval (halving q,
  restoring h from a saved copy `hold`) before CanZass, to keep
  CanZass cheap.  Rare; must be replicated for robustness.
- **Swan:** for r ≡ ±1 (mod 8) and (s even, s ∤ 2r) or (s odd,
  (r−s) ∤ 2r), the trinomial has an odd number of irreducible factors,
  hence a factor of degree <= r/3: rhigh = r/3 (both sample logs show
  45426613 = floor(r/3)).  Else rhigh = r/2.  Condition is
  flip-invariant.
- **Flip:** factor flips to the reciprocal trinomial when 2s > r
  (cheap NTL rem) and/or to make s odd (fastsqr requirement), and
  un-flips the output mask by printing it reversed.  Working directly
  on the original trinomial with a parity-agnostic squaring kernel
  gives byte-identical output with no flip logic at all (our GPU
  spread+fold squaring has no parity constraint, and input s <= r/2
  always).  Keep this in mind if the BLZ-style odd-s squaring kernel
  is ever ported for its 2-3x traffic advantage: that would reintroduce
  flip for even s.
- **Small-degree phase** (k <= k0 <= 64): gcd(x^(2^k)+x, T) computed as
  gcd(x^K+1, x^(r mod K)+x^(s mod K)+1), K = 2^k−1 — tiny.  Irrelevant
  for us (survivors are pre-filtered; we require --skip and start
  blocking at skip+1) but useful for selftests.
- **halfgcd.cpp FastGCD**: subquadratic HGCD over gf2x mul_gen.  NTL's
  own GCD (what gf2_gcd_ntl.cpp calls) is also HalfGCD-based and
  measured comparable (62 s vs 65 s at r=136279841 on Zen 2).  Either
  backend is fine; we use NTL's via the existing glue.

### A3. The apples-to-oranges correction

The earlier chat's verdict "gf2x 0.183 s/term beats GPU modmul 0.492 s"
compared factor's **per-degree** cost (mostly squarings) against the
GPU's **per-modmul** cost.  factor performs only ~1/14 modmul per
degree.  Once tsfactor adopts the same blocking, the GPU modmul is
amortized identically and the comparison becomes: GPU squaring
(0.69 ms measured, ~0.15 ms achievable fused) vs CPU fastsqr (12 ms),
and GPU modmul (492 ms today) vs gf2x (~0.9 s single-core).  The GPU
wins both legs on current kernels; the earlier program simply lacked
the blocking.

---------------------------------------------------------------------------
## Part B. tsfactor design (implemented this round)

One new source file `tsfactor.cu` reusing the validated headers
(gf2_cantor_core/engine/cuda.h, gf2_trinomial_cuda.h, gf2_gcd.h,
gf2_gcd_ntl.cpp, cuda_emul.h).  trinomial_scan.cu remains untouched
for comparison.

### B1. Scan core (per s)

    state: h[0..m-1] device residues (17 MB each), accumulator A,
           block poly C, spread/fold scratch, FFT buffers
    init:  Pascal-triangle e_j of (x^2^0 .. x^2^(m-1)), then
           (skip+1) batched squaring steps        [init_h equivalent]
    per block (m degrees):
           C <- fused Horner over h (single gather kernel + tail fixup)
           A <- A * C mod T          (Cantor FFT modmul, 1 per block)
           h[j] <- h[j]^(2^m) all j  (m batched fused-squaring steps)
    per interval (q blocks, q_next = q + q0 by default):
           D2H copy of A (17 MB) -> GCD thread pool (NTL), speculative
           continue; verdicts consumed strictly in order
    hit:   CanZass on interval gcd -> least degree + least mask
           (existing factor_min_degree_ntl); if deg(gcd) >= fineDDF
           threshold, bisect the interval first (B4)
    stop:  factor found | k > rhigh (Swan-aware) | maxd | -z trip

### B2. Kernels (new this round, all gather-form, emulation-safe)

- `k_horner_fused`: C = x^m XOR_j (h[j] << (m-1-j)), one pass, m*17 MB
  read + 17 MB write; degree can reach r+m-2, so a tiny tail kernel
  folds the top m-1 bits (x^(r+e) = x^(s+e) + x^e) and sets bit m.
  Requires m <= 64 (single-word shifts).
- `k_sqspread_multi` / fold passes batched over the m h-slots (grid.y
  = slot): the m Frobenius chains are independent, so each of the m
  sequential steps squares all m residues in one launch (3 launches
  per step vs 3m unbatched).
- `k_square_fused_multi`: OPTIONAL single-pass composition of
  spread+fold1+fold2 (each output word gathers <= ~9 shifted spread
  words, each spread word = Morton expansion of half an input word);
  ~34 MB traffic vs ~165 MB for the 3-kernel path.  Selftest A/Bs it
  against the 3-kernel path; `--legacy-sq` forces the old path.
  [STATUS: see Part E]
- mul path unchanged: k_pack, CudaFFT fwd/inv, k_pointwise,
  k_overlap_add, k_fold_pass reduction.

### B3. GCD pipeline

GcdPool from trinomial_scan (in-order verdict consumption, speculative
scan-ahead, non-blocking discard of stale jobs on hit, shared_ptr
modulus ownership) carried over intact.  Pinned host staging for the
17 MB A copies.  Pool size --gcd-threads (default: min(4, cores-1)).
Backends: NTL GCD (production) or naive Euclid (selftests, no NTL).

### B4. fineDDF / huge-gcd localization (rare path)

If deg(interval gcd) >= max(2*m*q0, 2*r^(2/3)) [--canzass-max to
override]: recompute h at the hit interval's start (init_h cost, GPU),
then replicate factor's bisection: hold = h copy; scan first ceil(q/2)
blocks, gcd; if hit recurse into it (h <- hold) else advance (k +=,
q = floor(q/2)); stop when deg(gcd) < threshold or q == 1; then
CanZass.  Extra transient memory: one hold set (m x 17 MB).  Steady
state keeps NO h snapshots.

### B5. Checkpoint / resume

- Results file (append + fsync per s) is the source of truth for
  COMPLETED s; resume skips s already present.
- Checkpoint = state of the in-progress s at a **verified** point (all
  earlier intervals' gcds returned trivial):
      header {magic, version, r, s, m, q-schedule state, k_iv, blk},
      h[0..m-1], and (iff blk > 0) A; CRC32 per section.
- Every interval start, h is snapshotted device->host (async) into a
  small ring tagged (k_iv, q); a ring entry becomes checkpointable
  when the in-order verdict frontier reaches k_iv.  The periodic
  writer (--ckpt-mins N, default 15) prefers a mid-interval snapshot
  {h_now, A_now, blk} when the current interval's predecessors are
  already verified (usual case: gcd latency 65 s << late-interval
  scan time), else falls back to the newest checkpointable ring entry.
- SIGINT/SIGTERM: finish the current block, write a checkpoint (same
  rules; no gcd drain by default — losing the speculative tail is
  bounded by pipeline depth), then exit.  Rotation .ckpt -> .ckpt.1 ->
  .ckpt.2, write-tmp-fsync-rename, resume tries newest-first with CRC
  validation, matching the coarse_sieve conventions Gord relies on.
- Mid-interval resume restarts at (k_iv, blk) exactly; the q growth
  sequence state (q, f0) is stored so schedules continue identically.

### B6. CLI (differences from both parents)

    tsfactor <r> <survivors.txt> [options]
      --skip D        REQUIRED: no factors of degree <= D exist (36/37)
      --maxd D        stop scanning above D (emit "s u")
      --m M           block size (default: auto-benchmark, even, 8..64)
      --q0 N          first-interval block count (default ceil(600/m))
      --f {0|1}       interval growth: 0 = constant q0, 1 = linear (dflt)
      -z Q | -Z Q     factor-compatible: give up at q > Q ("s u") vs
                      cap q at Q and continue (mutually exclusive)
      --out PREFIX    PREFIX.results.txt (+ .ckpt*)  [default: ts]
      --ckpt-mins N   periodic checkpoint interval    [default: 15]
      --gcd-threads N
      --device N      CUDA device (multi-GPU: run one process per GPU
                      on a split survivor file; nothing shared)
      --canzass-max D fineDDF threshold override
      --legacy-sq     3-kernel squaring path (A/B, fallback)
      -v[v[v]]        factor-style timing lines per interval
      --selftest | --bench [r [s]]

  Verbose output mirrors factor's:
      Current q 15 / Interval 37..246: / squares/products took ... per
      term ... / gcd took ...   with gcd lines arriving asynchronously
      (tagged by interval) plus a per-s summary.

### B7. Performance model (2070 SUPER numbers)

    per-degree ~= m * t_sqr + t_mul / m        [+ Horner ~0.05 ms]
    t_sqr: 0.69 ms measured (3-kernel), ~0.15-0.2 ms fused (predicted)
    t_mul: 492 ms measured (unfused Cantor FFT, n = 2^24)

    m_opt = sqrt(t_mul/t_sqr):  ~27 today -> per-degree ~37 ms
                                ~50 fused -> per-degree ~20 ms
    factor.cpp on the 6-core Zen 2: 0.233 s/degree single-core; the
    squaring 72% is serial per s, so a single deep s runs at ~0.19-0.23
    s/degree regardless of cores.  GPU speedup on the deep tail: ~6-10x
    per box today, ~2x more with fused squaring, ~2-3x more again if
    the FFT traffic fusion ever lands.
    Easy-survivor throughput is gcd-bound either way (~1.0-1.6 full
    GCDs/survivor floor, 62-65 s each on one core); the GPU frees all
    cores for GCDs and the speculative pipeline hides their latency.
    Roughly half the total remaining work (in degree-mass) sits in the
    d > ~4*10^4 tail, which is where the GPU advantage is largest.

    VRAM at m=24: h 408 MB + scratch/FFT ~570 MB + A/C ~35 MB ~= 1.0 GB
    (hold transient +408 MB).  8 GB cards can host m = 64 if the bench
    ever favors it.

---------------------------------------------------------------------------
## Part C. Optimization roadmap (post-v1, in priority order)

1. **Fused squaring kernel** (single-pass gather; 4-5x traffic cut on
   the dominant cost).  Implemented behind a flag this round; needs
   hardware A/B: `tsfactor --bench 136279841` prints both.
2. **Shared-memory FFT passes** (radix-2^g tiles with __syncthreads,
   ~24 levels in 3-5 memory passes).  The register-tile Taylor-cascade
   fusion REGRESSED 5-9% on hardware (register spill / DRAM locality;
   hybrid routing keeps legacy kernels for large strides) — do NOT
   resurrect it as-is; shared-memory tiling is the correct successor.
   Requires upgrading cuda_emul.h to phase-synchronous block emulation
   (run threads of a block in lockstep segments between barriers) or
   validating those kernels on hardware only.  Payoff: t_mul 492 ->
   plausibly 100-150 ms; moves m_opt up and per-degree toward ~8-12 ms.
3. **Cross-s pipelining**: round-robin 2-4 s values on one GPU so easy
   survivors' gcd latency is hidden by other s's scan work (h-sets are
   400 MB each — fits).  Big win only for the easy masses; measure
   first whether gcd CPU throughput, not latency, is the binding
   constraint on the target box (6-core: likely yes -> this buys less).
4. **Hybrid GPU HGCD** (host recursion, device muls above ~10^5 bits):
   est. 8-15 s/gcd vs 62 s, but contends with scanning for the GPU.
   Only worth it if boxes are CPU-starved (community machines with
   8 GB GPUs often have 6-8 cores; revisit with real telemetry).
5. **BLZ odd-s squaring port** (~60 MB traffic, needs flip logic for
   even s): only if (1) underdelivers; the fused gather already gets
   most of the win without parity constraints.
6. Overlap h-update chain with the block modmul on separate streams
   (both saturate bandwidth, so expect <10%; free to try).

---------------------------------------------------------------------------
## Part D. Validation assets

- `factor` builds in the sandbox against apt NTL/gf2x; NTL here is NOT
  gf2x-backed (slow muls, correct results) — fine for oracles, do not
  benchmark it.
- 136279841_00M.log: full reference for s < 1.5e6 (951208 entries with
  d <= 36; 46998 with 37 <= d <= 1000; 1714 with d <= 2e4; 79 above).
  Spot-checked masks re-verified by an independent Python divisibility
  check (x^r + x^s + 1 ≡ 0 mod f) — 3/3 pass.
- Planned/implemented selftests (CPU emulation, no GPU needed):
  (a) blocked scanner vs per-degree naive oracle, r=4423, many s,
      multiple (m, q0), interval-edge hits, "u" cases, tie cases;
  (b) exact-output diff vs the real factor binary at r=44497 over an
      s range (same skip), including flip-parity s values;
  (c) checkpoint/resume equivalence: kill mid-interval, resume, diff;
  (d) fineDDF forced via tiny --canzass-max;
  (e) fused-vs-legacy squaring bit-identity across many (r, s) parities;
  (f) Horner-fused vs mulx/XOR reference bit-identity.
- Hardware validation (Gord): `--bench 136279841 <s>` component times;
  selftest on GPU; then a slice of real survivors re-checked against
  factor and/or verify_gf2_trinomial.py.

---------------------------------------------------------------------------
## Part E. Round-1 status inventory  [UPDATE EVERY ROUND]

DONE this round:
- factor.cpp fully decoded (Part A); factor built + reference runs.
- tsfactor.cu written (~1790 lines): blocked scan core, fused Horner,
  legacy 3-kernel squaring, fused single-pass squaring, GCD pool +
  speculation, interval schedule with -z/-Z, Swan/rhigh, CanZass
  resolution, fineDDF bisection, verified-state checkpointing with
  3-generation rotation + CRC + signals, factor-style verbose timing,
  selftest + bench modes; Makefile targets tsfactor / emul-tsfactor.
- VALIDATION (all in CPU emulation, g++ 13 / NTL 11.5.1):
  * --selftest: ALL PASS.  Covers: fused-vs-legacy-vs-host squaring
    (50 (r,s) cases incl. parity and word-boundary edges), mul-by-x
    kernel, init_h + fused Horner vs host block product (m in
    {2,3,5,8}, kstart in {1,5,17}), end-to-end vs per-degree oracle at
    r=4423 (104 survivors; 102 found / 2 u / 3 equal-degree ties)
    across 5 configs incl. f=0, f=2, legacy-sq, forced fineDDF
    (--canzass-max 1), -z / -Z semantics, and checkpoint kill/resume
    equivalence at kill points {0,1,3,6,11}.
  * vs the REAL factor binary at r=44497 (s = 2..499, skip 8, 106
    survivors, degrees up to 603): byte-identical results for both
    (m=8,q0=4) and (m=14,q0=2) -- including all 50 even-s cases, which
    empirically confirms the no-flip reciprocal-bijection argument.
  * real SIGTERM mid-run at r=44497 then resume: 83 + 23 survivors, 0
    duplicate lines, combined output byte-identical to factor.
  * production-scale smoke at r=136279841 (emulated, m=4): Swan bound
    45426613 printed correctly, Exponentiation 6.4 s, mid-interval
    checkpoint written (85 MB, k_iv=37 blk=2/3), resume continued the
    interval and grew q correctly, interrupted exit is fast (~19 s)
    even with a 136M-bit GCD in flight (stop_nowait + _Exit path).
  * expected first result on real hardware: "474 41 p3991ef16129"
    (from the independently re-verified 00M reference log).

NOT yet done / needs Gord:
- Any real-GPU run (no GPU in sandbox): `--selftest` on device, then
  `--bench 136279841 474` for t_sqr fused vs legacy and t_mul (auto-m
  uses these), then a survivors slice diffed against the 00M log
  and/or verify_gf2_trinomial.py.
- NTL thread-safety on Gord's build is assumed (pool default 3
  threads); if anything misbehaves, --gcd-threads 1.
- Multi-round: optimization items C1-C4 as measurements dictate.

Known judgment calls (flag if you disagree):
- No flip logic: we always work on the original trinomial (s <= r/2 by
  input contract), so masks print directly; factor-output equivalence
  is by the reciprocal-bijection argument (A2) and tested in (b).
- Checkpoints only at verified states; kill loses <= speculative tail.
- -maxd exhaustion prints "s u" (factor would print "irreducible",
  which is misleading under -maxd).

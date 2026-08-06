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

INCIDENT (round 1a): first hardware selftest FAILED -- every factor
resolution errored "could not factor interval gcd" while all kernel
suites passed and every reported gcd degree matched a correct scan.
Root cause: the binary was built WITHOUT NTL.  The Makefile's NTL
detection try-links ntl_check.cpp; that file was not among the round-1
deliverables, so in a build directory assembled from them the probe
silently failed, NTL_OK=0, and the build proceeded cleanly on the
naive-GCD fallback, whose mask enumeration cannot handle degrees
beyond ~24.  Multi-factor interval gcds (the cases needing CanZass)
then failed while single-factor hits resolved through the
gdeg < 2*ka shortcut -- which is exactly why -z/-Z passed and why the
tight-window m=6 config showed only 2 failures.  Proven by rebuilding
the emulation with NTL_OK=0: all 55 ERROR lines (s, deg) match the
hardware log exactly.  Hardening now in place: ntl_check.cpp ships
with the deliverables; `make tsfactor` prints the NTL detection
result and HARD-FAILS on a no-NTL build unless ALLOW_NO_NTL=1;
tsfactor prints "GCD backend: ..." in the selftest banner and at scan
startup; scanning survivors without NTL is refused unless
--allow-no-ntl; a no-NTL selftest marks resolution suites
"SKIPPED (NTL required)" instead of FAIL and exits 0 when the kernel
suites pass.  Conclusion: no GPU bug exists; the device-side pipeline
is fully validated by his hardware log.

Known judgment calls (flag if you disagree):
- No flip logic: we always work on the original trinomial (s <= r/2 by
  input contract), so masks print directly; factor-output equivalence
  is by the reciprocal-bijection argument (A2) and tested in (b).
- Checkpoints only at verified states; kill loses <= speculative tail.
- -maxd exhaustion prints "s u" (factor would print "irreducible",
  which is misleading under -maxd).

## Round 2 (hybrid GPU-assisted GCD + hardening)

HARDWARE RESULTS FROM GORD (RTX 2070 SUPER, 8 GB):
- bench: sqr legacy 0.368 ms, sqr fused 0.252 ms, Horner(m=24)
  1.177 ms, modmul 506.4 ms; recommended m=48 at ~22.7 ms/degree.
  Production auto-m picked 46 and achieved 19 ms/degree.
- Validation: s in {96, 843, 2331, 4478, 9249, 19985, 29216} match the
  factor 00M log exactly (d from 94 to 70774); >1500 NEW factors for
  s in 2.0-2.3M all check-ntl verified; records d=568473 (s=2273693)
  and d=285602 (s=2247915).  GPU memory steady at 4.0 GB.
- Multithread defaults validated: s=7458 (d=30775) with gcd-threads 3 /
  pend-max 4 matches the 00M log; gcds (67-69 s) overlap the scan.
- Gord runs --gcd-threads 1 --pend-max 1 to keep CPU cores free for
  "factor"-based prefiltering (d 38..247), then tsfactor --skip 247.
  NOTE: --pend-max 2 with gcd-threads 1 is strictly better (scan runs
  one interval ahead of the single worker, no extra CPU); tsfactor now
  prints this hint.

T4 / CLOUD "HANG" ROOT CAUSE: the gcd is pure CPU work plus one ~17 MB
transfer -- CPU<->GPU bandwidth is not involved.  Either that VM's NTL
lacks the gf2x backend (an order of magnitude slower at 136M bits;
looks hung) or its vCPU is simply far slower than the Zen 2 baseline.
tsfactor now prints the gf2x status in every banner (NTL_GF2X_LIB
detection via NTL/config.h) and --bench --bench-gcd times one real gcd
so the imbalance is visible in seconds, not hours.

MAKEFILE (adopted Gord's approach): NTLLIB is now hardcoded to
"-lntl -lgf2x -lgmp" by default (correct for both /usr/local source
builds and distro packages, and gf2x is effectively required for
production anyway).  The ntl_check.cpp try-link remains as a VERIFIER:
`make tsfactor` prints "== NTL: OK/FAILED ... ==" with a specific
diagnosis and refuses to build a no-NTL binary unless ALLOW_NO_NTL=1.
`make ntl-diag` shows the raw compiler error on link failure.
Gord's simplified Makefile dropped the NTL_OK plumbing that the
emul-tsfactor target needs; the merged version restores it.

C4 DELIVERED -- HYBRID HGCD WITH GPU MULTIPLICATION OFFLOAD (opt-in):
- Engine: gf2_gcd_ntl.cpp, poly_gcd_hybrid(): classical polynomial
  half-gcd recursion (m = ceil(n/2) split, one division step across
  the straddle, second recursion at k = 2m - l), base case 4096 bits,
  tiny multiplies (<= 512-bit operand) via direct shift-xor, mid-size
  via NTL/gf2x, large via a GpuMulHook.  Postconditions checked at
  every level; any violation or non-progress falls back to plain
  poly_gcd_ntl on the ORIGINAL inputs (slow, never wrong).  Captured g
  is normalized to from_gf2x's format (byte-identical to the NTL path).
- Hook: tsfactor.cu TsGpuMul -- packs operands to 32-bit chunks, runs
  the existing Cantor-FFT convolution with lazily built per-size plans
  (shares the scan's CantorBasis, owns separate device buffers dA/dB/dP
  = at most ~335 MB at r=136M), mutex-serialized; the legacy default
  CUDA stream serializes device work against the concurrent scan.
- Flags: --gpu-gcd (enable), --gpu-gcd-min-bits N (default 2^21),
  --bench-gcd (with --bench: time one full gcd, print backend + mult
  counts).  hyb_ntl_finish_bits (2^22) hands the reduced pair to NTL.
- VALIDATION: standalone battery 119/119 byte-exact vs NTL (70 bits to
  400k bits; random / planted gcds / b=0 / a=a / unit b / huge degree
  gaps / trinomial-vs-dense), zero fallbacks; exact at 8M and 33M bits.
  Selftest suite (8): CPU-mult and GPU-hook paths vs NTL on nonlinear
  AND adversarial F2-linear (xorshift/LFSR) data -- the LFSR data is a
  deliberate stress case: consecutive xorshift outputs form one linear
  recurrence, so remainder sequences collapse via a giant quotient to
  the generator's ~4096-bit linear complexity (this initially looked
  like an engine bug; it is a property of F2-linear test data.  Use
  SplitMix64-style nonlinear generators for realistic benchmarks).
  Suite (9): full end-to-end scan with the hybrid forced through the
  pool (thresholds shrunk for r=4423) plus a ts_gcd cross-check mode
  (g_hyb_xcheck) that verifies every hybrid gcd against NTL and dumps
  operands on mismatch.
- HONEST PERFORMANCE EXPECTATIONS: HGCD's total multiplication work is
  Theta(M(n) log n) spread across ~15 recursion levels; offloading
  mults >= 2^21 bits captures roughly the top quarter-to-third of the
  mass, and the GPU's advantage per mult over gf2x-on-Zen2 is modest
  at mid sizes (per-call floor: transfers + ~7 launches + small-plan
  transform).  First-generation projection for the 67 s gcd: roughly
  45-60 s with a few seconds of CPU freed per gcd -- measure with
  --bench-gcd, and A/B a survivors slice before adopting.  The main
  wins today: weak-CPU hosts (T4-class) and partial CPU relief.
  Round-3 path to more: batch the 4 products of a mat_apply (shared
  operands: 6 forward + 2 inverse transforms instead of 12), pinned
  staging + smaller per-call overhead to take the mid-level mass, and
  trimming the hybrid's remaining ~25% CPU-constant vs NTL (mat_mul
  composition avoidance at more levels).
- EMULATION FIX (delivered cuda_emul.h): blockIdx/threadIdx/gridDim/
  blockDim are now thread_local.  They were shared statics, so
  emulated kernels launched from pool worker threads (the hybrid's
  hook) raced against the scan thread's kernels -- emulation-only
  nondeterminism; real CUDA has per-thread builtins and was never
  affected.  Suite (9) is what exposed this.

## Round 3 (batched frequency-domain matrix ops + FFT bench breakdown)

BATCHED 2x2 OPERATIONS (delivered, automatic under --gpu-gcd): the four
products of an HGCD matrix application share their operands, so
GpuMulHook gained mat2_apply(m[4], a, b -> oa, ob): the hook transforms
a and b once, streams each matrix entry through one scratch spectrum,
combines spectra by XOR (GF(2^64) pointwise sums = sums of products),
and inverse-transforms once per row -- 6 forward + 2 inverse transforms
instead of 12.  mat_mul routes column-wise through the same primitive
(column j of A*B is A applied to (B[0][j], B[1][j])): 16 transforms
instead of 24.  Engine falls back to per-product mul() whenever the
hook declines (size over plan cap, or zero batch), so behavior is
identical, just cheaper.  TsGpuMul grew two scratch buffers (dM, dR);
with --gpu-gcd at r=136279841 the hook now holds five full-size
device buffers (~603 MB total) plus lazily built plans -- still
comfortable next to the scan's 4.0 GB on an 8 GB card.  Zero-entry
matrix rows/columns (e.g. right after an elementary step) are handled
by bits==0 skipping.  Validation: selftest suites (8) and (9) exercise
the batched path (GPU-op counts dropped 616 -> 446 and 408 -> 188 with
identical results); all suites PASS, zero fallbacks.

MODMUL COMPONENT BREAKDOWN (delivered): --bench now splits the modmul
into pack / forward FFT / pointwise GF(2^64) / inverse FFT /
overlap-add / fold-reduce, each timed with device syncs.  ACTION FOR
GORD: send the breakdown from `./tsfactor --bench 136279841 474` on
the 2070 SUPER -- it decides where C2 effort goes (transform-internal
memory behavior vs. the GF(2^64) pointwise multiply vs. reduction).

MULTI-GPU NOTE: tsfactor parallelizes across survivors at the process
level today: run one instance per GPU with --device N on disjoint
survivor files (checkpoints are per-s and independent).  No code
needed; within-s parallelism is blocked by the sequential Frobenius
chain.

ROUND-4 CANDIDATES: (a) C2 FFT work from the breakdown numbers;
(b) pinned host staging + per-call overhead cuts so hyb_gpu_min_bits
can drop toward the mid-level mult mass; (c) trim the hybrid's
remaining CPU constant vs NTL (56.8 s vs 45.4 s at 8M bits on
identical data, sandbox vCPU); (d) C5 BLZ squaring variant if
squaring stays the largest per-degree term at the recommended m.

## Round 4 (C2: transform tuning infrastructure from the breakdown)

GORD'S ROUND-3 HARDWARE DATA (2070 SUPER): selftest all PASS on
hardware including both hybrid suites (fb 0).  Breakdown of the 510 ms
modmul: forward FFT 177.9 ms (x2), inverse FFT 152.5 ms, everything
else < 2 ms combined -- the transforms are 99.6% of the modmul.  gcd
bench: NTL 63.0 s vs hybrid 63.57 s (1440 GPU mults, 136M tiny CPU
mults, 0 fallbacks) -- wall-time neutral, as projected for the
first-generation offload.  --gpu-gcd production A/B: all 37 survivors
with d >= 1000 (s < 20000) match "factor" exactly.  GPU memory 4.9 GB
with --gpu-gcd vs 4.0 GB without.  NEW RECORD: s=2601234 factor of
degree 1,440,452 (~9 h wall with "factor" prefilters sharing the CPU).

DEPLOYMENT FINDING (Gord): apt libgf2x-dev is ~10x slower than
source-built-and-tuned gf2x on a c2-standard-4 ("factor": 710 s + 710 s
vs 45 s + 60 s).  Likely the original T4 story.  RULE: on cloud hosts,
build gf2x from source with its tuning step, then NTL against it.

ANALYSIS: the transform cost model is purely memory-bound and matches
measurement.  Legacy path: the Taylor cascade is m(m-1) ~ 552 streaming
passes (~55 GB) plus 24 butterfly passes (~8 GB) => ~63 GB/transform
=> ~165-178 ms on a 2070S.  The fused register-tile path (DEFAULT ON;
Gord's 178 ms measured it) should cut traffic to ~19 GB (~50 ms ideal)
but at GF2C_TAYLOR_LV=5 each thread holds 64 u64 in registers: spill +
collapsed occupancy strangle achieved bandwidth to ~30% of peak, which
is the previously observed "fusion regression".  Long-stride (large CH)
groups additionally defeat DRAM locality.

SHIPPED (gf2_cantor_cuda.h + tsfactor.cu):
- Runtime knobs: gf2c_taylor_lv / gf2c_bfly_lv (fused levels per pass;
  register cost 2^(tlv+1) resp. 2^blv u64 per thread) and
  gf2c_fuse_max_ch (groups with column stride >= this run the legacy
  streaming kernels for exactly those levels; 0 = no limit).  Defaults
  reproduce the previous behavior bit-for-bit.
- Environment overrides read at CudaFFT::init (production picks up the
  tuned config with no new flags): GF2C_FFT=legacy|fused,
  GF2C_TAYLOR_LV, GF2C_BFLY_LV, GF2C_FUSE_MAX_CH.
- --bench FFT autotune: verifies every candidate ON DEVICE against the
  legacy transform (forward byte-equality + inverse roundtrip), times
  fwd+inv for legacy and 27 fused configs (tlv, blv in {3,4,5}; maxch
  in {off, 2^16, 2^12}), leaves the best active (so a following
  --bench-gcd uses it), prints the winner as export lines, and projects
  the tuned modmul and per-degree cost.  ~1 minute on GPU.
- Selftest suite (3b): FFT variant equivalence at m=12 (legacy vs 12
  fused configs, forward byte-equality + roundtrip); NTL-free, runs in
  all build modes.
EXPECTATION: if a config reaches memory-bound, fwd+inv ~100-130 ms
(from 330 ms), modmul ~510 -> ~160-200 ms, per-degree ~21.3 -> ~12-15
ms at a lower optimal m -- and the hybrid gcd's 1440 GPU mults shrink
proportionally.  ACTION FOR GORD: run --bench 136279841 474 and adopt
the printed exports if a fused config wins and verifies.

OPEN (round 5+): probdist-driven interval scheduling (Gord supplied
p(d) ~ (16/9)/d^2 empirics from 74207281/57885161 -- geometric interval
growth should cut deep-survivor gcd counts by an order of magnitude vs
the current linear growth; needs a careful expected-cost derivation and
an opt-in growth mode + selftest config); lowering hyb_gpu_min_bits
once transforms are faster; C5 BLZ squaring if squaring becomes
dominant after FFT tuning.

## Round 5 (reliability: teardown race in the GPU gcd hook)

SYMPTOM (Gord, repeated --selftest runs): intermittent failures at the
END of the run -- "CUDA error invalid argument at tsfactor.cu:505",
"... illegal memory access at :505/:558", SIGFPE ("Floating point
exception"), or a hang with no GPU activity.  Rates: ~3/100 with
default --gcd-threads 3, ~1/100 with --gcd-threads 1, present in
rounds 2, 3 and 4 (i.e. since the hook was introduced).

ROOT CAUSE (one bug, all four symptoms): tearing down the GPU
multiplication hook while a pool worker was still inside it.  A scan
ends at a verdict via GcdPool::forget_all_pending(), which only MARKS
outstanding jobs for discard -- the worker executing one keeps running
and is still inside ts_gcd -> poly_gcd_hybrid -> the hook.  Selftest
suite (9) then did g_gm = nullptr; G.fini(); gm9.fini(), freeing dP /
dA / dB / dM / dR and deleting the CudaFFT plans.  The still-running
worker then dereferenced freed device pointers:
  - line 505 = the H2D cudaMemcpy into dP in pack_fwd -> "invalid
    argument"; line 558 = the D2H cudaMemcpy out of dP in mat2_apply;
  - a kernel launched with freed pointers -> "illegal memory access";
  - CudaFFT::fwd on a DELETED plan -> garbage m/n -> taylor_groups
    computes CH = (1 << (b_lo-2)) << L from garbage, CH = 0, then
    nwin = n / (NS*CH) -> integer divide by zero -> SIGFPE;
  - a worker stuck in a corrupted context -> the hang in pool.stop()'s
    join, GPU idle.
More gcd threads = more chances to be mid-hook = higher rate, matching
the observed 3/100 vs 1/100.
EMPIRICAL CONFIRMATION: injecting a delay into the hook (test build
only) and draining at teardown reports "drained 2" -- two jobs were
still running inside the hook at that exact point, every run.

PRODUCTION WAS NOT AFFECTED.  main() already tears down in the correct
order: pool.stop() JOINS all workers before G.fini().  The bug was
reachable only through the selftest's suite-9 teardown.  It also
cannot corrupt a computed gcd: the crash is a use-after-free of device
buffers at shutdown, and every hook call is mutex-serialized on
operands the worker owns.  No result, degree, or "smallest factor"
claim from any production run is in question.

FIXES (defence in depth, all shipped):
- GcdPool::drain(): blocks until nothing is queued or running and
  returns how many jobs were in flight; suite (9) drains before
  freeing anything and prints the count ("drained N").
- TsGpuMul::fini() now takes the hook mutex (a call already in flight
  completes first) and clears a new `alive` flag; mul() and
  mat2_apply() check `alive` under the lock and DECLINE when torn down,
  so the engine falls back to CPU multiplication instead of touching
  freed memory.  Either mechanism alone is sufficient; both are in.
- Staging copies are clamped to bits_to_words(bits) words, so a caller
  vector with trailing zero words cannot overrun dP.
- main() frees the hook explicitly after pool.stop() (provably safe).
- New flag --selftest-reps N repeats the whole selftest in-process;
  intermittent races need loops, not single runs.

FFT AUTOTUNE RESULTS (Gord, 2070 SUPER, 5 repeats, very consistent):
  legacy per-level          320-325 ms      (fwd+inv)
  fused tlv=5 blv=5 (OLD DEFAULT)  353-363 ms   <-- slower than legacy
  fused tlv=3 blv=3 maxch=off     157-169 ms   <-- best, ~2.0x legacy
  fused tlv=4 blv=*               297-345 ms   (sharp cliff at 32+
                                                registers per thread)
  maxch=2^16 / 2^12         always WORSE (226-294 ms): routing
    long-stride groups to the legacy streaming kernels HURTS -- the
    old "large strides defeat the register tile" belief is disproven
    at these sizes; keep GF2C_FUSE_MAX_CH=0.
ACTIONS TAKEN: defaults changed to GF2C_TAYLOR_LV_DEFAULT=3 and
GF2C_BFLY_LV_DEFAULT=3 (so a plain rebuild gets the 2x with no env
vars), and the sweep now also tests lv=2 -- the win was still growing
at the old lower bound, so 2 may beat 3.
EXPECTED PRODUCTION EFFECT: modmul 510 -> ~240-250 ms, per-degree
~21-23 -> ~15.6-16.4 ms at a lower optimal m (30-32 rather than 48):
roughly a 30% throughput gain on the squares/products phase.

PROBDIST ANALYSIS (for round 6, from Gord's r=74207281 and r=57885161
files, 37.1M and 28.9M trinomials): the small-degree values are exact
rationals (p(2)=1/3, p(3)=4/21, p(4)=2/21 to 5 decimals in both), and
the SURVIVAL function fits S(k) = P(no factor of degree <= k) =
(16/9)/k to 3-4 significant figures over four decades and in both
files:
    k        37       247      1000     10000    100000
    S(k)   0.04746  0.00719  0.00177  0.00018  0.00002
    16/9k  0.04805  0.00720  0.00178  0.00018  0.00002
CONSEQUENCE for scheduling: expected scan cost to depth K is
proportional to K, but the number of GCDs under the current LINEAR
interval growth is ~sqrt(2K/(m*q0)) (~375 to full depth at
r=136279841), whereas GEOMETRIC growth (interval k -> rho*k) needs
only log_rho(K/k0) (~30 at rho=1.5).  With c_g ~ 63 s per gcd and
c_s ~ 16-21 ms per degree, the optimum rho balances (rho-1)/ln(rho)
expected scan overshoot against rho/(rho-1) gcd density; S(k)=C/k
makes the expectation integrable in closed form.  Deliver as an opt-in
growth mode with its own selftest config, then A/B on a survivor slice.

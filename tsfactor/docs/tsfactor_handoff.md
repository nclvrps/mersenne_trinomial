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

## Round 6 (probdist-driven interval scheduling: --sched opt)

ROUND-5 CONFIRMATIONS (Gord, 2070 SUPER): selftest 100/100 clean both
with default threads and --gcd-threads 1 (teardown race fixed).  New
3/3 FFT defaults verified: modmul 510 -> 228-231 ms (fwd 76.6, inv
76.9), best per-degree m=32 at 15.8 ms/degree in bench and 13-14
ms/degree in production.  Sweep at lv=2 confirms 3/3 remains best
(tlv=2 is worse: fewer levels fused = more passes; the optimum is
bracketed).  BENCH REDUNDANCY: --bench --bench-gcd --gpu-gcd is a
superset of the other two variants; run the plain --bench when only
FFT numbers are wanted.

PEND-MAX A/B (Gord, two survivors with d ~ 24.1k/24.6k, --gpu-gcd,
--gcd-threads 1, "factor" instances loading the CPU): with pend-max 2
the gcd latency inflates 75 -> 132 s as intervals lengthen (the
hybrid's GPU mults queue behind continuous scan kernels), and since
the single gcd worker is the bottleneck stream at these depths
(gcds ~73 s vs interval scans 8-85 s), the inflation ate the overlap:
wall 1068 s (pend 2, incl. one speculative interval) vs 1034 s
(pend 1).  CONCLUSION: at shallow-to-mid depths with 1 gcd thread the
run is GCD-STREAM-BOUND either way; pend-max 2 only wins deep (scan >
gcd per interval) or with more gcd threads.  Gord's home-vs-cloud
framing adopted: home users may prefer pend-max 1 (energy/heat, same
wall time in this regime); cloud users should verify pend-max 2
actually wins on their instance before paying for it.  The real fix
for the gcd-stream regime is fewer gcds -- which is this round.

SCHEDULING THEORY (from the probdist survival law S(k) = (16/9)/k,
verified to 3-4 significant figures over four decades in both of
Gord's files): minimize E = sum_j S(k_{j-1}) (c_s w_j + c_g).  The
stationarity condition is k_{j+1} = k_j^2/k_{j-1} - G with
G = c_g/c_s in degrees (~4000 at 63 s / 15.8 ms) -- ratios start
large and settle toward ~1.3.  Numeric study at c_s=13.5 ms,
c_g=73 s, skip=247, K=r/3:
                       E[cost|surv]  E[gcds]  gcds to 24.6k/1.44M/K
  linear (current)        170.5 s     1.66        9 / 68 / 384
  pure geometric (best)   181.6 s     1.46       (worse than linear!)
  euler-optimal           153-157 s   1.26-1.3    4-5 / 12-29 / 19-80
KEY INSIGHTS: (a) conditioned on surviving skip=247, 71% of factors
land in the FIRST interval, so E[gcds] is small under any schedule
and pure geometric LOSES (its big first interval overshoots the near
field); (b) the euler shape = gentle near field + geometric far
field; (c) the big structural wins are E[gcds] -24% (CPU freed for
"factor" prefilters) and deep-run gcd counts cut 4-9x (matters for
wall on deep/Swan-capped runs, for weak-CPU hosts, and for the
gcd-stream serialization above); E[wall] improves ~8-10%.

IMPLEMENTATION (--sched opt, default remains linear):
- QSched gained an opt mode: first interval end k1 chosen by a 96-
  point argmin scan of the closed-form cost model (guarded above
  sqrt(k0*G), where the recurrence would otherwise degenerate); then
  widths follow w' = w*k/(k-w) - G (the stationarity recurrence in
  width form), monotone non-decreasing and snapped to blocks.  The
  width form needs only the current position and width -- exactly what
  checkpoint resume restores -- so the schedule continues bit-exactly
  across kills with NO checkpoint format change (verified: kill/
  resume at r=44497 under --sched opt is byte-identical to the
  uninterrupted run).
- --sched-G N overrides G (default 4000; retune from --bench: G =
  gcd_seconds / per_degree_seconds).  -z/-Z caps still apply after
  growth.  Banner prints the mode.
- Selftest: new end-to-end config "m=6 sched-opt G=60" vs the oracle
  (all 104 survivors identical); full selftest all PASS.
GUIDANCE: use --sched opt for production; expect the same factors
(any in-order interval partition finds the same smallest factor),
~24% fewer gcds in expectation, and dramatically fewer on deep runs.
DYNAMIC ADAPTATION (Gord's suggestion -- adjust G from observed
runtimes mid-run): deliberately deferred; a static per-run schedule
keeps resume deterministic and testable.  A v2 could re-derive
FUTURE widths from an EWMA of measured c_g/c_s (the width-form
recurrence makes this a one-line change), at the cost of resume
reproducing decisions from the checkpointed width rather than the
adapted history -- still correct, mildly nondeterministic.

## Round 6b (schedule saturation fix, --f retirement, memory accounting)

FIELD REPORT (Gord): --sched opt A/B on three survivors, factors
byte-identical.  Shallow pair (d ~ 24k): 940 s -> 685 s (-27%; the
schedule's 5 gcds vs linear's 9 in a gcd-stream-bound regime).  Deep
(d=185321): 4167 s -> 3661 s, but q CAPPED AT 473 -- and that cap was
a BUG, not memory:

SATURATION BUG: the stationarity recurrence's ratios decrease by G/k
each interval (rho' = rho - G/k).  For k1 choices whose ratio budget
rho_1 - G*sum(1/k_j) crosses 1.0, the ratio decays below 1, the
monotone-width guard freezes the width, and the schedule degenerates
into a constant-width tail (reproduced exactly from the log: widths
freeze once w*k/(k-w) - G < w).  E[cost] barely notices (survival is
tiny out there) which is why the k1 argmin happily picks such
trajectories -- but deep-run WALL pays ~(D/w_frozen) gcds: a d=1.44M
run would have paid ~100 gcds.  FIX: a geometric floor,
w' >= (rho_min - 1)*k, rho_min default 1.25 (--sched-rho), applied in
both the recurrence and the argmin's internal model.  The floor
matches the depth-conditional optimum rho*ln^2(rho) =
2*ln(D/k1)*c_g/(D*c_s), which spans ~1.2 (D=1.44M) to ~1.65 (D=185k):
1.25 covers deep runs at negligible shallow cost.  Same trajectory now
reaches d=1.44M in ~22 gcds.

--f RETIRED (per Gord; no external users yet): the factor.cpp-inherited
growth modes (f=0 constant, f=1 additive, f>1 crude geometric) are
gone; non-opt growth is plain linear (q += q0), --sched opt is the
sophisticated path.  CkptHdr keeps the f0 field as reserved (written
0) so the checkpoint FORMAT is unchanged; resuming an old checkpoint
under the new binary continues with linear growth from the stored
(k, q).  Selftest configs reworked: constant-q coverage now via the
-Z cap (P.Zq=1); all 16 suites PASS.

--sched-G GUIDANCE (Gord's measurement): idle ~4000, loaded ~5000.
The expectation is flat across that range; either value is fine, and
the schedule is deterministic per run either way.

GPU MEMORY ACCOUNTING (Gord's question; also now printed at startup):
allocations are FIXED at init and independent of q/depth --
  h-tables: 2*m*ceil(r/64)*8 B (dH ping-pong; ~1.02 GB at r=136.3M,
    m=30; ~1.56 GB at m=46) + the same again lazily for dHold on the
    first fineDDF localize;
  FFT: dFA+dFB (2 * 2^24 * 8 = 268 MB) + twiddle table (~134 MB);
  scan scratch: dA/dC/dTmp/dS0/dS1 (~120 MB);
  --gpu-gcd hook: five buffers ~610 MB + small lazy per-size plans;
  plus CUDA context (~300-500 MB) and whatever the GUI holds.
The q=473 cap had nothing to do with memory; there is no memory-based
q logic anywhere.  On a 16 GB T4 usage is identical (depends on r and
m, not the card).  The old log's slow creep to ~4.2 GB was most
plausibly dHold materializing plus lazily built gcd-hook plans.

## Optimization roadmap: status and what remains

DONE (original Part C and successors):
  C1 fused squaring        -- delivered R1; 0.36 -> 0.26 ms measured.
  C4 hybrid GPU HGCD       -- delivered R2-R3 (opt-in --gpu-gcd);
                              wall-neutral vs NTL on a loaded Zen 2,
                              wins = CPU relief + weak-CPU hosts;
                              batched mat ops (R3) cut its transforms.
  C2 FFT                   -- delivered R4-R5: runtime knobs + autotune;
                              tlv=3/blv=3 default; transforms 2.16x
                              faster; 19-21 -> 13-14 ms/degree in
                              production.
  scheduling               -- delivered R6/R6b: --sched opt (+ rho
                              floor); -27% on shallow survivors,
                              4-9x fewer deep gcds.
  reliability              -- R5: hook teardown race fixed (drain +
                              alive-guard); --selftest-reps; 100/100.
  C3 cross-s pipelining    -- RETIRED: superseded by Gord's
                              factor-based CPU prefilter workflow.

REMAINING CANDIDATES, in rough value order:
  1. FFT achieved-bandwidth work: at tlv=3 the ~29 GB of fused traffic
     runs at only ~45% of streaming peak -- the strided register-tile
     gathers are the suspect.  Needs Nsight Compute profiles from the
     2070S (memory workload analysis on k_taylor_reg NS=16) before
     touching code; potential is another ~1.5-2x on transforms.
  2. C5 BLZ squaring: m*t_sq is now ~7.8 of 15.5 ms/degree -- the
     LARGEST per-degree term post-FFT-tuning.  A ~30-40% squaring cut
     is ~2-3 ms/degree.
  3. Hybrid gcd mass: with faster small-plan transforms, lower
     hyb_gpu_min_bits toward the mid-level mult mass; add pinned host
     staging to cut the per-call floor.  Target: 63 s -> 35-45 s
     hybrid with most mass on GPU.
  4. Dynamic schedule adaptation (Gord's idea): EWMA of measured
     c_g/c_s feeding the width recurrence; one-line change, mildly
     nondeterministic on resume -- do when static tuning feels stale.
  5. Two-GPU single-s: hand the h-table to a second GPU each interval
     (~1 GB transfer, ~100 ms over PCIe) so GPU2 scans interval j+1
     while GPU1 finishes j -- speculative but architecturally cheap;
     only worth it once single-GPU per-degree cost stops improving.
  6. Primitive-trinomial mode polish: for full-depth campaigns (the
     three-irreducibles hunt), the "u"/primitive paths plus sched-opt's
     ~20-30 gcds to r/3 make exhaustive runs ~178 h -> dominated
     purely by scan; any FFT/squaring win transfers 1:1.

## Round 6c (schedule behavior confirmed; C3 re-evaluated; release notes)

Q-PLATEAU IS BY DESIGN, NOT A GLITCH.  All three of Gord's observed q
sequences (G=4000/5000/6000, m=30, k0=248) were reproduced EXACTLY
from the schedule arithmetic.  The opt schedule has three regimes and
takes the max of three rules each step:
  1. recurrence   w' = w*k/(k-w) - G     (ratios decay by ~G/k)
  2. monotone     w' >= w                (never shrink an interval)
  3. geometric    w' >= (rho_min-1)*k    (floor, default rho 1.25)
Early on rule 1 governs; its ratio decays until it drops below the
current width; for ONE OR TWO intervals rule 2 holds the width flat
(the plateau) until the growing floor 0.25*k overtakes it, after which
rule 3 governs and widths grow at exactly 1.25x forever.  Traced for
G=5000: recurrence for 4 intervals, one monotone hold at q=250
(rec=5613, floor=6392, both < w=7500), then geometric floor from
q=276 on.  The plateau is the handoff between regimes, costs
essentially nothing, and cannot recur (the floor only grows).

--sched-G EMPIRICS (Gord, s=2469396, d=185321, moderate load):
  G=4000 (default)  3602.6 s   207750 degrees scanned
  G=5000            3568.6 s   197100 degrees   <- best
  G=6000            3966.1 s   265110 degrees   (11% worse: big early
                                ratios overshoot the answer)
GUIDANCE: G in 4000-5000; measure as gcd_seconds/per_degree_seconds
and do not round up.  Larger G buys fewer gcds at the price of
overshoot, and past ~5000 the overshoot wins.

C3 CROSS-S PIPELINING: RE-EVALUATED, STILL NOT WORTH REVIVING -- but
the overlap it was designed to capture IS now available for free.
Under --sched opt the intervals grow geometrically, so scan time per
interval overtakes gcd time: for s=2469396 (G=5000) scan > gcd in
12 of 14 intervals.  Cost model vs observation: serial (pend-max 1)
predicts 3557 s, observed 3569 s; ideal one-deep overlap predicts
2743 s, i.e. ~814 s / 23% available from --pend-max 2 ALONE.  This is
exactly what the earlier pend-max A/B could not show, because under
LINEAR growth intervals stayed short and the gcd stream was the
bottleneck; geometric growth inverts that.  So:
  - FIRST re-run the pend-max 1 vs 2 A/B under --sched opt on a deep
    survivor.  If gcd latency inflation (the 75->132 s effect from
    GPU contention) eats the win, the targeted fix is STREAM
    SEPARATION -- give the gcd hook its own non-blocking CUDA stream
    (or build with nvcc --default-stream per-thread, which gives each
    pool worker its own stream; the hook's blocking D2H already
    synchronizes before the mutex is released, and buffers are
    disjoint) -- NOT cross-s pipelining.
  - C3's only remaining niche is shallow survivors whose single s has
    too little scan work to hide its own gcd.  That is precisely the
    regime Gord's CPU "factor" prefilter already owns, and C3 would
    cost a second GPU h-table (~1 GB at m=30), multi-s checkpoint
    state, and verdict routing across s -- large complexity for a
    shrinking niche.
VERDICT: leave C3 retired.  If the pend-max 2 A/B disappoints, add
"gcd hook on its own stream" to the roadmap at position 2 (cheap,
contained, same benefit); C3 stays off the list.

CODE CLEANLINESS: after the --f removal the whole translation unit
compiles warning-free under -Wall -Wextra (g++ 13/15).  No dead code
or refactoring debt found worth churning before release.

RELEASE READINESS / PARTICIPANT CAMPAIGN NOTES:
  - Recommended production invocation:
      ./tsfactor <r> <survivors> --skip <d> --gpu-gcd --sched opt \
        --gcd-threads 1 --pend-max 2 --ckpt-mins 60 --out <prefix>
    (drop --gpu-gcd if the host CPU has spare cores and gf2x-backed
    NTL; use --pend-max 1 to minimise GPU energy at equal wall time
    in the shallow regime.)
  - Each participant should run --selftest once (expect 16 PASS) and
    --bench <r> 474 once, adopting the printed GF2C_* exports if their
    card's best config differs from the 3/3 default.
  - Verify every returned factor with check-ntl before merging results.
  - ERROR CHECKING matters MORE for a distributed campaign than for a
    single operator: check-ntl proves a REPORTED factor divides the
    trinomial, but it cannot detect a MISSED factor caused by a silent
    hardware error -- a false negative looks like a clean "no factor
    below d" result and would silently corrupt the campaign's coverage
    claim.  The separate error-checking analysis in this project found
    a Gerbicz-style accumulator that is DETERMINISTIC in
    characteristic 2 (Frobenius is additive, so the check is pure XOR)
    at ~2-5% overhead, plus Bezout-cofactor certification for the gcd.
    That work is the natural next priority if the campaign is to make
    a defensible completeness claim over the full s range.

## Round 7 (fine DDF: no more re-exponentiation)

PROBLEM (Gord's large-degree log): entering fine DDF cost a full rebuild
of the h-table at the hit interval's start -- init_h_at(ka) ends with ka
Frobenius steps, i.e. ka*m squarings.  Observed: 2756 s at ka=501728,
6845 s at ka=1250318, and 26417 s (7.3 h) at ka=4843958 for s=5408270,
whose localization was then lost entirely to a reboot.

ROOT INSIGHT: that state is not missing, it is discarded.  The scan HELD
exactly the h-table at ka when it began that interval; it was simply
overwritten as the scan moved on.  factor.cpp had to rebuild because it
kept nothing, and tsfactor inherited the behavior.

FIX (three layers, all shipped):
1. RAM snapshot per outstanding interval.  Each interval start copies
   the h-table to host (~m*nw*8 B: 511 MB at r=136279841, m=30) into a
   shared_ptr carried by its Pend entry, so the snapshot stays alive
   exactly as long as that interval's gcd is unresolved -- note
   consume_verdict copies the Pend before clearing the queue, so the
   buffer survives into localize.  localize() then does one H2D instead
   of ka*m squarings.  Budget: --hsnap-mb (default 2048), one snapshot
   per interval with an outstanding gcd; the startup banner prints the
   per-snapshot size, how many fit, and warns when the budget is below
   pend-max+1.  --hsnap-mb 0 restores the old rebuild.
   NOT snapshotted: an interval we resumed INTO mid-way (blk>0), where
   the device h is already past the interval start -- that one interval
   still rebuilds if it hits.
2. On-disk standby file <ckpt-base>.fine, written when fine DDF is
   actually entered and refreshed at each bisection step: (r, s, skip,
   m, nw, ka0, q0, kcur, qcur) + the h-table at kcur, CRC'd and written
   atomically through the existing write_file_atomic.  Removed on every
   localize exit.  So a crash or reboot mid-localization loses neither
   the rebuild nor the bisection progress.
3. Resume straight into localization: scan_one_s checks for a matching
   .fine before the interval loop and, if found, re-enters localize at
   the saved window and resolves from there -- no rescan of the (up to
   multi-hour) hit interval.  This is sound because verdicts are
   consumed in order and the first hit returns, so every degree below
   ka0 was already verified clear when the file was written.
BONUS: localization is now INTERRUPTIBLE.  localize checks g_stop right
after refreshing the standby file, so SIGINT/SIGTERM during a
multi-hour bisection now exits cleanly and resumes where it stopped;
previously the signal was ignored until localization finished.

TESTS: selftest is now 17 suites, all PASS.  Suite 5 runs the forced
fineDDF battery TWICE -- once reusing the snapshot, once with
--hsnap-mb 0 forcing the rebuild -- and both must match the oracle.
New suite 7b picks an s whose hit genuinely goes through localize,
interrupts it inside the bisection (die_in_fine hook, at 3 different
points), and re-runs with only the standby file on disk, requiring the
same factor degree.  CLI check at r=4423: both paths give byte-identical
results, and the .fine file is created during bisection (7 refreshes)
and gone after completion.

EXPECTED EFFECT on Gord's cases: the 2756 s / 6845 s / 26417 s rebuilds
become a single ~0.1 s host-to-device copy.  The bisection scan itself
is unchanged (it is bounded by the interval width, which is work
already budgeted); the pure waste was the rebuild, which was
proportional to the ENTIRE depth reached so far, not to the interval.

REJECTED ALTERNATIVE (so it is not re-proposed): doing the localization
modulo the interval gcd g instead of modulo T.  g is ~40x smaller than
T, but T is a sparse trinomial whose reduction is a near-free fold,
while g is dense and needs Barrett/Newton reduction -- roughly two FFT
multiplications per squaring.  At r=136M that is ~0.26 ms per squaring
mod T versus ~20 ms mod a 3.5M-degree dense g: about 85x WORSE.
Bisection modulo T is the right algorithm.

MINOR NOTES: a .fine file for an s that is never re-run stays on disk
(it only matches its own r/s/skip/m/nw, so it is inert -- just delete
it); at m=30 it is ~511 MB, refreshed once per bisection step.

---------------------------------------------------------------------------
## Round 8 (error checking on by default; certified gcd)

[Round 9 note.  This section is kept as written, with two exceptions marked
"R9:".  Everything it says about gcd SPEED was measured against the
distribution's gf2x and did not survive contact with a tuned one -- see
Round 9.  Its "not yet run on a GPU" caveats are obsolete: Gord ran the whole
test plan on the real card and it passed (Round 9, first paragraph).
Defaults that Round 9 changed: --gcd-mt (now automatic), --no-ec's gcd
backend (now the engine, uncertified), --sched-G (now measured), and the
pre-wait snapshot threshold (now 2 s).]

Written in the error-checking conversation, against the tsfactor.cu dated
Sep 16 (the one with the fine-DDF standby state and --hsnap-mb).  The
mathematics and its history are in errorcheck_handoff.md; this section is
the tsfactor-specific record: what was added, where it sits, how it was
tested, and what still needs a real GPU.

STATUS AT A GLANCE
  - Error checking is implemented and ON BY DEFAULT.  --no-ec restores the
    old program exactly.
  - The gcd is CERTIFIED by default, by a new engine path that is also
    faster than NTL's gcd on CPU multiplications (in the authoring sandbox:
    -24% at 4 Mbit, -27% at 16 Mbit, -32% at 136 Mbit).  --gcd-ntl restores
    the plain NTL gcd.  THIS IS THE ONE DEFAULT THAT CHANGES RUN TIME; confirm
    it on the production machine with `make gcdbench && ./gcdbench
    136279841 588` before trusting the sandbox ratio.
  - --gcd-mt N (opt-in) runs the independent products of each gcd on N CPU
    threads.  Verified correct; 1.29x from 2 threads on the sandbox's two
    virtual CPUs, unmeasured on real cores.
  - Validated under CPU emulation only (no nvcc in the authoring sandbox):
    selftest 32/32 PASS (15 new suites), 20 consecutive repeats clean,
    ASan + UBSan clean (selftest and the CLI regression), 33 CLI
    configuration runs byte-identical to the ORIGINAL binary, 11 injected
    faults caught and recovered from on the command line, one run at the
    production exponent r=136279841 (see VALIDATION).
  - NOT yet compiled by nvcc, NOT yet run on a GPU, NOT yet timed there.
    The first thing to do is `make tsfactor && ./tsfactor --selftest`.

FILES
  gf2_ec.h         NEW.  Host fold algebra (EcFold, ecf_*) + the device
                   kernels (k_ec_*).  Header-only; define GF2_EC_HOST_ONLY
                   to get the host part alone (the gcd engine does).
  tsfactor.cu      2844 -> 4816 lines: EC state in TsGPU, verified
                   transfers, restart logic in the scan, options, selftest
                   suites.  The stale --f line is gone from the usage text.
  gf2_gcd.h/.cpp   certified gcd (poly_gcd_certified), split application of
                   half-gcd matrices (hyb_split_apply), factor verification
                   (trinomial_divisible_ntl).  poly_gcd_ntl is unchanged;
                   poly_gcd_hybrid returns the same answers, faster.
  gcdbench.cpp     NEW.  CPU-only timing of the gcd paths; `make gcdbench`.
  gcdstress.cpp    NEW.  CPU-only randomized cross-check of every gcd path
                   against NTL; `make gcdstress`.
  Makefile         gf2_ec.h added to the tsfactor / emul-tsfactor /
                   emul-trinomial_scan rules; gcdbench, gcdstress targets.
  (devkit) tests/ts_regress.sh   CLI regression, see VALIDATION.

WHAT IS CHECKED, AND BY WHAT (numbers refer to the header of gf2_ec.h)

  h-table squarings      Mechanism 1, the Frobenius-linear Gerbicz chain.
  + idle h-table         S = XOR_j h[j];  sigma(D) + D == S_first + S_last
                         over a window, compared at FULL WIDTH on the
                         device (k_ec_or4 -> one flag word).  One extra
                         pass over the table per block (k_ec_block_pass)
                         plus m squarings per window (--ecw, default 8).
                         With no block since the last check the same call
                         reduces to "S unchanged", which is how an idle
                         table is checked: at the start of every interval
                         (i.e. after the gcd wait under --pend-max 1),
                         before every checkpoint, before every fine-DDF
                         window.
  h-table initialisation Mechanism 2, NEW MATHEMATICS THIS ROUND: the
                         digest has a closed form,
                             1 + S_k = (1+x)^(2^k (2^m - 1))  mod T,
                         so (1+y_k)(1+S_k) == 1+y_(k+m), y_t = x^(2^t).
                         TsGPU::ec_verify_table_at: k+m squarings of ONE
                         residue + one modmul.  A failed init is rebuilt
                         (init_h_checked).  --eci0 skips it.
                         COST IN PERSPECTIVE: tsfactor's init is squarings
                         only, about m(k + m/2) of them -- roughly half a
                         second at skip 55, m ~ 28 on a 2070 SUPER.  The
                         check is one modmul (~0.25 s) plus k+m squarings.
                         Neither matters next to a 60 s gcd, so there is no
                         case for making --eci0 the default.  (On a deep
                         resume the init is k*m squarings and the check k:
                         there it really is 1/m.)  R9: factor_ec's init
                         is squarings-only too (882 of them at m=14,
                         k=56); what made it the largest EC cost there
                         was running it twice.  Round 9 gives factor_ec
                         this same check.
  Horner build of C      Mechanism 3.  The per-block table pass also folds
  FFT product A*C        the unreduced Horner sum, independently of
  reduction mod T        k_horner_fused; k_horner_tail reports its
                         quotient mask; the host predicts phi(C).  The
                         unreduced product is folded in three segments by
                         ONE pass (k_ec_fold3_part), which yields phi(t)
                         and the quotient fold; the reduced A is folded
                         once.  Host: phi(t) == phi(A)*phi(C) and
                         phi(A') == phi(t) + phi(q)*phi(T).  phi(A) is
                         CARRIED on the host from block to block (never
                         recomputed), so a flip in the idle accumulator
                         breaks the next product check, and A's download
                         is checked against it for free.
  transfers, snapshots   Mechanism 4.  h_to_host_ec / h_from_host_ec /
                         A_to_host_ec: fold on the device vs fold on the
                         host, re-read up to 3 times.  Every host snapshot
                         (HSnap) carries its fold and is re-verified when
                         used, so RAM rot while it waits is caught.
  the gcd                see GCD below.
  reported factors       trinomial_divisible_ntl: x^r mod f by repeated
                         squaring; a factor that does not divide T, or
                         whose degree lies above its interval, is an EC
                         event, never a result.

  Fold length: EC_NF = 61 words (3904 bits), deliberately not a power of
  two.  With 4096 bits an error that repeats with a power-of-two period (a
  stuck bus lane; a constant pattern over a power-of-two FFT sub-block)
  folds onto itself an even number of times and cancels.  Do not "tidy"
  this to 64.

  tsfactor enforces 2s < r (set_s).  The reduction prediction
  (ecf_reduce_predict) is valid ONLY there; for 2s > r the quotient has
  more terms.  factor_ec lives in the other regime and uses a different
  reduction check for that reason.

RECOVERY
  A failed check never patches anything in place (one exception below).
  scan_one_s_once returns ScanOut::EC_RETRY with *resume filled from `rb`,
  the newest state known good, kept in RAM:
      - the interval-start snapshot, captured when everything below it has
        a clear verdict (same condition as the checkpoint candidate), or
      - the mid-interval state written by the last periodic checkpoint
        (verified BEFORE it is saved: the chain window is closed first),
      - or the state the scan was resumed from.
  scan_one_s (the old name, now a wrapper) re-enters through the ordinary
  resume path; with no RAM state it falls back to the disk checkpoint, then
  to restarting the s.  The same restart point failing more than
  --ec-retries (3) times in a row is a persistent fault: ScanOut::EC_FATAL,
  no result line for that s, loud message, exit status 3.
  The exception: after an interval that took >= 30 s (SParams::
  ec_presnap_secs) the NEXT interval's start table is downloaded before the
  gcd wait; an idle flip during the wait is then repaired by reloading it
  ("in-place repair" in the summary) instead of rescanning the interval.
  Nothing unverified is ever written to a checkpoint or fine-DDF file.

GCD
  The gcd is about a minute of CPU work in ordinary RAM per interval and,
  under --pend-max 1, most of the wall time of a shallow survivor.  A fault
  there can turn a hit into a miss, and nothing downstream would notice.

  THE CERTIFICATE.  The engine (hyb::hgcd in gf2_gcd_ntl.cpp, the one
  behind --gpu-gcd) reduces the pair by steps of two kinds,
      (a,b) <- M (a,b)      and      (a,b) <- (b, a - q b).
  Each is verified on the host with folds carried from the step that
  produced an operand to the step that consumes it (hyb::Cert), and the
  finish is an NTL XGCD of the small remaining pair whose Bezout identity
  is verified the same way.  Every step is an exact GF(2)[x]-linear
  combination, so every common divisor of (A, T) divides every later pair
  and the result: a returned degree of 0 PROVES coprimality.  No
  determinant check and no cofactor accumulation are needed, and only the
  depth-0 transformations need checking: the recursive calls only produce
  a matrix, and ANY matrix applied through a verified step preserves the
  invariant.  The fold work is linear; certification cost between 1% and
  9% in the gcdbench runs below (the larger figures include timing noise).

  THE SPEEDUP (split application).  hgcd recurses on the top half of the
  pair, IN PLACE on a copy, and gets back a matrix R.  The old code then
  applied R to the full-length operands.  But
      R (a, b) = x^k * R (a >> k, b >> k)  +  R (a mod x^k, b mod x^k)
  and the first term is exactly the pair the recursion already holds.  So
  only the low k bits are multiplied (mat_apply_split): the engine's
  largest products shrink from (n/4 x n) and (n/4 x 3n/4) to (n/4 x n/2)
  and (n/4 x n/4).  Counting balanced products of size n/4 per level:
      old engine  36 (28 at the top)    NTL HalfGCD  24 (28)    now  20 (12)
  which is why the old engine was 30-40% slower than NTL on the CPU (the
  "remaining ~25% CPU-constant" of Round 2) and the new one is faster.
  hyb_split_apply = false / --gcd-full-apply restores the old behaviour.

  Measured in the sandbox (one core of a 2.1 GHz Xeon, distro gf2x 1.3.0,
  UNTUNED; r, finish threshold as shown; `gcdbench`):
                                   4.0 Mbit / 131072    16 Mbit / 524288
      NTL GCD                          5.43 s               39.06 s
      CERTIFIED, NTL XGCD              7.63 s  +40.6%       56.00 s  +43.4%
      engine, full apply (old)         7.08 s  +30.5%       55.02 s  +40.9%
      engine, split apply              3.95 s  -27.2%       28.38 s  -27.3%
      CERTIFIED chain (the default)    4.15 s  -23.5%       28.61 s  -26.7%
                                   136 Mbit / 4194304 (the production size and threshold)
      NTL GCD                          972.0 s
      engine, split apply              605.8 s  -37.7%
      CERTIFIED chain (the default)    657.7 s  -32.3%
  (the 136 Mbit rows were timed while other tests used the second core)
  The ratio is a property of the multiplication counts, so it should carry
  over to a tuned gf2x; it has NOT been measured there.

  THREADS INSIDE ONE GCD (--gcd-mt N, default 1).  The four products of a
  matrix application are independent, and so are the eight of a matrix
  product; with N > 1 they run side by side (par_run / cpu_mat2 / mat_mul
  in gf2_gcd_ntl.cpp) whenever both operands have >= hyb_mt_min_bits
  (32768) bits.  The two recursive calls of a half-gcd are inherently
  sequential, as are the division steps, the base cases and the folds, so
  the gain is bounded well below N.  MEASURED ONLY ON THE SANDBOX'S TWO
  VIRTUAL CPUS: 31.1 s -> 24.1 s at 16 Mbit, 1.29x from 2 threads (the NTL
  gcd took 44.4 s in that run).  Two vCPUs that may share a core say little
  about 4-6 real ones: `./gcdbench 136279841 588 4194304 quick 4` prints
  the number that matters.  Under --pend-max 1 the scan waits for the
  gcd, so idle cores are otherwise wasted during it.  CPU products only:
  it does nothing under --gpu-gcd, --gcd-ntl or --ecg-xgcd.  Needs a
  thread-safe NTL (NTL_THREADS, the default) and gf2x >= 1.2; total gcd
  threads = --gcd-threads x --gcd-mt.  Results are identical by
  construction (same products, same order of additions); gcdstress runs a
  third of its cases on three threads, clean under ThreadSanitizer.

  MODES
    default                the certified chain on CPU multiplications.
    --gpu-gcd              the same chain with its large products on the
                           GPU (certified as before; the hook's
                           multiplications are covered by the same folds).
                           Was wall-neutral vs NTL (63.0 vs 63.6 s) with
                           the OLD full application and before the FFT
                           tuning: both changes favour it, neither has been
                           measured.  Under --pend-max 1 the GPU is idle
                           during every gcd.
    --gcd-ntl              plain NTL GCD.  The input is verified on arrival
                           and a reported factor is verified; the gcd
                           itself is NOT checked.  The banner says so.
    --ecg-xgcd             certify with one NTL XGCD of the whole problem,
                           Bezout identity fold-verified.  ~1.4x NTL's
                           gcd.  An independent reference for the chain.
    --ecg-dup              NTL, twice, compared; third run on a tie.
    --ecg                  accepted, redundant (factor_ec's option name).
    --no-ec                NTL (or the uncertified engine under --gpu-gcd),
                           as before this round.
  A failed certificate is a fault DURING the gcd: it is rerun once; twice
  failed is an EC event and a restart.  If the chain ever declines (a
  half-gcd postcondition fails: never observed) the gcd is certified by the
  XGCD instead, so "not certified" cannot happen in an NTL build.
  Below --gcd-finish-bits (4 Mbit) the pair goes to the verified XGCD; at
  small r that is the whole gcd.

OPTIONS (all new)
  --no-ec  --ecw N  --ect  --eci0  --ec-retries N  --ec-fault SITE:N
  --gcd-mt N  --gcd-ntl  --ecg-xgcd  --ecg-dup  --ecg
  --gcd-finish-bits N  --gcd-full-apply
  --bench R S --bench-gcd now times NTL, then (under --gpu-gcd) the
  uncertified engine, then the certified gcd the configuration would run.
  Environment: TSFACTOR_ARGS="--ect,--ecw,4" (comma or space separated) is
  appended to the command line -- for wrapper scripts whose configuration
  can set environment variables but not arguments (mersenne_trinomial.py's
  tsfactor_env line splits on whitespace: use commas).
  Detections go to stderr immediately and to PREFIX.ec.log (created only
  if something happens).  The exit summary counts blocks, windows,
  detections by category, restarts, repairs, transfer re-reads, gcd
  re-runs; --ect adds a cost table (it inserts device synchronisation, so
  use it to measure, not in production).
  Exit status 3 = persistent fault (EC_FATAL).  mersenne_trinomial.py
  relaunches tsfactor regardless of exit status; the devkit's
  mersenne_trinomial.ec.patch makes it stop on 3.

COST OF THE SCAN CHECKS (ESTIMATES -- the emulator cannot time a GPU)
  per block     1 table pass (about one Horner: ~1.2 ms at m=24 measured
                in R2) + 51 MB of streaming folds + a 2.4 KB readback
  per window    m squarings + 1 digest pass
  per s         k+m squarings + 1 modmul (init verification), ~0.25 s
  => expected 0.5-1% of scan time at m ~ 30, --ecw 8.  `--bench R S` now
  prints the measured EC kernel times and a projected percentage per m.
  VRAM: 5 residues + small = 81 MB at r=136279841.  Host RAM: one extra
  h-table copy while a mid-interval rollback state is held.

VALIDATION (emulation)
  selftest, new suites:
    EC: device folds vs host (48 (r,s,m) cases)
    EC: block predictions vs real C, A*C, A*C mod T
    EC: one flipped bit trips the matching check
    EC: h-table chain, clean / one flip / idle flip
    EC: closed-form table digest (init verification)
    EC: no false alarms (~6500 blocks, ~4100 windows across suites 4-9)
    EC: single-fault sweep -- 14 sites, 216 faults: 0 wrong answers with
        checking; 81 of 206 wrong without
    EC: faults under speculation (84 runs, answers only)
    EC: persistent fault gives up cleanly (EC_FATAL)
    EC: restart from a mid-interval verified state
    EC: idle flip repaired from the pre-wait snapshot
    EC: certified gcd == NTL (75 gcds: chain/CPU, chain/GPU hook, XGCD,
        declined chain, chain on 3 threads)
    EC: certified gcd under faults (600 runs: every fired fault caught)
    EC: scans with certified gcd; faults inside the gcd (reruns)
  The same selftest built with -DEC_NBLK_MAX=1 (every fold kernel takes
  many strides per thread, as at production size) passes.
  The fault sites (also --ec-fault): h hmid a c t red idle init xferh
  xfera snap hold job gcd.

  gcdstress: 3100 randomized inputs up to 200000 bits (prescribed quotient
  sequences incl. all-degree-1 and huge straddling quotients, planted
  common factors, b | a, a == b, zero operands, the production shape; NTL
  finish threshold from 64 bits up), four engines each: 0 mismatches,
  0 spontaneous declines.

  tests/ts_regress.sh NEW ORIG: r = 3217, 4423, 9689 (294 survivors) in
  eleven configurations each, results identical to the ORIGINAL
  (pre-round) binary; options from the environment; eleven --ec-fault
  runs (ten caught with identical results, one landed in a snapshot that
  was never read again); persistent fault -> exit 3; three interruptions
  and resumes; a fault after a resume.  All PASS, also under ASan + UBSan.

  Production exponent, emulated (m=8, q0=4, s=320, skip 55; the oracle
  says a factor of degree 74):
    default     (final binary) result line identical to the oracle, 4
                blocks + 2 windows verified, 0 detections, the gcd
                certified by the chain in 660 s
    --ecg-xgcd  (earlier binary, when that mode was spelled --ecg) the
                same, the gcd certified by the XGCD in 1360 s
  (NTL's plain gcd on this input size and machine: 972 s.)
  These are the only tests in which the fold kernels and the certified
  chain ran inside tsfactor at production size.

  What the sweep shows about consequences (unchecked column):
    h, hmid, init, idle  ~always wrong: the table stays wrong forever
    a, c, t, red         wrong only if the fault lands in the hit interval
                         at or after the factor's block (1 in 30 here)
    gcd                  a hit turned into a miss is silent; a clear
                         turned into a bogus hit is REPORTED AS A FACTOR
                         without the divisibility check

KNOWN GAPS
  1. Not compiled by nvcc; not run on hardware.  Emulation is sequential:
     a race in a kernel would not show.  The EC kernels are gather-form
     with one writer per output word (k_ec_or4/k_ec_neq have many writers
     of the SAME value), which is why a race is unlikely -- not proof.
  2. The gcd timings are sandbox timings on an untuned gf2x.
  3. Two faults in ONE window with identical error polynomials cancel in
     the XOR accumulator.  The realistic instance is a stuck bit hitting
     the same cell twice; it cancels in about half the windows, is caught
     in the others, and then fails every retry -> EC_FATAL.  --ecw 1
     removes the window.
  4. Control flow (loop counters, k, q) is not covered by any identity.
  5. CanZass / the fine-DDF resolution of a NONTRIVIAL gcd are checked only
     through the final "does the reported factor divide T, and is its
     degree in the interval" test.  That makes a wrong factor impossible
     to report; it does not prove that the reported factor is the one of
     LEAST degree if the equal-degree split itself were faulted.

WHAT TO RUN ON THE GPU FIRST  (details in TESTPLAN_EC.md)
  1. make tsfactor && ./tsfactor --selftest          expect 32 PASS
  2. ./tsfactor --selftest --selftest-reps 20
  3. make gcdbench gcdstress && ./gcdstress && ./gcdbench 136279841 588
  4. ./tsfactor --bench 136279841 588 --bench-gcd [--gpu-gcd]
     (EC kernel times + projected overhead; NTL vs certified)
  5. one known survivor with and without --no-ec, -v -v -v --ect:
     identical result line, 0 detections, and the --ect total against the
     wall-clock difference
  6. the same survivor with --ec-fault h:200, then idle:1, then t:50
  7. tests/ts_regress.sh ./tsfactor

OPEN QUESTIONS / NEXT STEPS
  - gcdbench on the production machine: is the certified chain faster than
    NTL there too?  And is --gpu-gcd now faster still?  If the latter, the
    recommended invocation gains --gpu-gcd.
  - Frequency-domain batching on the CPU.  The four products of a matrix
    application share operands (6 forward + 2 inverse transforms instead of
    12) and a 2x2 matrix product needs 8 + 4 instead of 24.  The GPU hook
    already does this (mat2_apply); on the CPU it needs gf2x's FFT
    interface (gf2x-fft.h) instead of NTL's mul.  Estimated: a further
    25-35% off the CPU gcd.  This is the next gcd speedup if the gcd is
    still the bottleneck.
  - Strassen for the 2x2 matrix product (7 products instead of 8, as
    gf2x/apps/halfgcd.cpp does): about 4% of the gcd.  Small and safe.
  - Fuse k_ec_block_pass into k_horner_fused's read stream, or the digest
    into the last squaring's store, if --bench shows the table pass matters.
  - Control-flow guards (redundant counters; assert k2 == k + q*m - 1).
  - Port the closed-form init check to factor_ec (there it replaces a
    duplicate init costing ~7.6 s per s with ~2 s).  R9: done.

---------------------------------------------------------------------------
## Round 9 (the gcd's base case; threads by default; a schedule that measures itself)

Written in the error-checking conversation (its round 5), from Gord's logs
of the Round 8 test plan on the real hardware.

WHAT GORD'S RUN OF ROUND 8 SHOWED  (RTX 2070 SUPER, 6-core Zen 2, tuned gf2x)
  - nvcc compiles it (5 unused-variable warnings); --selftest 32/32, 21
    consecutive runs; ts_regress.sh ./tsfactor all PASS; four injected
    faults at production size caught with the same result line; interrupt
    and resume fine.  THE ERROR CHECKING WORKS ON THE REAL CARD.
  - scan checks: --bench projects 0.6% (m = 24..32); a real survivor
    (s=12337659, skip 262) measured +1.0% (124.0 -> 125.2 s of scanning).
  - but the survivor took 305.5 s instead of 280.2 s, because each of its
    three gcds took 59 s instead of 51 s: Round 8's default gcd, measured
    27-32% faster than NTL in the sandbox, was 6% SLOWER on the real machine
    (gcdbench: NTL 51.9 s, certified chain 55.1 s; 59 s inside tsfactor).
  - the gcd was 58% of that survivor's wall time (177 of 305 s).
  - --gcd-mt 4: 35.1 s, only 1.57x (2: 42.1 s, 3: 39.7 s, 5: 35.1 s).

DIAGNOSIS
  The sandbox's gf2x was the distribution package (no PCLMUL): its products
  are ~10x slower than a source-built gf2x, so there the gcd was nothing
  but large products.  With fast products the rest of the engine showed.
  1.57x from 4 threads means half the time was outside the threaded
  products, and --bench-gcd had been printing why all along:
      mults: gpu 0 cpu 135960386
  136 million multiplications per gcd -- the base case.  Below 4096 bits
  hyb::hgcd did Euclid one division step at a time on std::vector
  operands, each step followed by two bit-level shift-xor "products" to
  maintain the matrix, each of those allocating, each taking a mutex to
  bump a statistics counter.  A profile confirmed it: ~90% of the engine's
  own instructions were in that loop.
  LESSON, recorded so it is not relearned: gcd timings from a build with
  the distro gf2x say nothing about production.  The sandbox now builds
  gf2x from Gord's source tree (hwdir x86_64_pclmul, untuned thresholds);
  with it NTL's 136 Mbit gcd takes 52-53 s there, the same as on his
  machine, and the round-8 engine reproduces his +6%.

WHAT CHANGED

1. gf2_hgcd_base.h (NEW): the word-level base case.
   The polynomial form of Lehmer's method.  Take the top 64 bits of the
   pair (same alignment); run Euclid on those two words, accumulating a 2x2
   matrix of word-sized polynomials (hgb::euclid64: shifts and xors); apply
   it to the operands and to the accumulated matrix with carry-less word
   products (hgb::lin2, PCLMULQDQ through ecf_cl64) -- about 31 quotient
   bits per pass.  Fixed stack arrays, no allocation, no library call.
   EXACTNESS: no carries in GF(2)[x], so the rule is clean.  With the
   dividend's leading bit at bit 63 of the window, a division step whose
   divisor has window degree v is the TRUE step iff 2v >= 63.  So the
   window Euclid runs while the divisor's degree is >= 32 (the ordinary
   half-gcd rule for a 64-bit pair) or until the true degree would drop
   below the caller's target, and the result is not approximate: quotient
   sequence, matrix and reduced pair are IDENTICAL to the old loop's.  A
   quotient of degree >= 32 (rare; it happens with sparse inputs) takes one
   plain multiword division step.
   The old loop is kept as hyb::hgcd_base_ref: it is the reference
   (hyb_base_selfcheck compares the two word for word) and the fallback if
   the word-level routine ever declines (size bound; never observed).
   Hand-over size: hyb_base_bits, default 2048 (--gcd-base-bits; anything
   from 512 to 8192 measures the same within noise; maximum 16320).
   In certified mode a top-level base case is verified as ONE matrix step
   (cert_matstep), like every other step of the chain.

2. Products call gf2x directly.
   cpu_mul now calls gf2x_mul on the engine's word arrays instead of
   converting both operands to NTL::GF2X and the result back.  Compile-time
   switch -DHAVE_GF2X, set by the Makefile when gf2x_check.cpp (NEW) links;
   without it the NTL path remains.  gf2x_mul is re-entrant (it allocates
   its own pool when called with none), which the threads rely on.
   The statistics counters are atomics; the mutex is gone.

3. Division steps: a quotient shorter than 64 bits is done by shift-xor
   passes at any operand size (it was converted to NTL above 65536 bits).

4. Threads.
   --gcd-mt now defaults to AUTOMATIC: (cores - 1) / gcd-threads, at most 4
   (the selftest stays on 1).  hyb_mt_min_bits 32768 -> 65536: a thread is
   spawned per product, ~50 us, and a 32 kbit product now takes ~20 us.
   Threads and the GPU hook work TOGETHER: under --gpu-gcd the products
   below the hook's size threshold run on the threads (they used to run
   serially whenever a hook was present).

5. The engine is the gcd backend whether or not checking is on.
   --no-ec used to mean NTL's gcd; it now means the engine, uncertified
   (g_engine_cpu).  --gcd-ntl selects NTL in either case.

6. The interval schedule measures its own G.
   --sched opt needs G = (seconds per gcd) / (seconds per scanned degree).
   It was the constant 4000 (63 s / 15.8 ms).  It is now an exponential
   average of the measured gcd time over the measured scan rate
   (g_cost_gcd, g_cost_deg, sched_G_now), process-wide, so every survivor
   after the first starts informed; 3000 until the first gcd finishes.
   --sched-G N fixes it, as before, and is what the selftest and the
   regression use (a measured G makes the interval boundaries depend on
   timing; results never do).  -v -v prints the G in use.
   This matters now: with the gcd several times cheaper the optimal
   intervals are several times shorter.

7. Checkpoints are deleted when their s is finished.
   PREFIX.ckpt, .ckpt.1, .ckpt.2 (0.5 GB each at r=136279841) used to stay
   on disk after the result line was written.  remove_ckpts runs right
   after the result file's fsync.

8. Pre-wait snapshot threshold 30 s -> 2 s (SParams::ec_presnap_secs).
   Gord's idle:1 demo restarted the first interval (16 s of scanning plus
   a second 59 s gcd) because that interval was under the 30 s threshold.
   The snapshot is taken while the scan would be waiting for the gcd
   anyway, so it is free in wall time; now any interval worth 2 s gets one
   and an idle flip costs a reload.

9. nvcc's five warnings: the bench autotune now restores the FFT
   configuration it changed (it had been leaving its last trial setting in
   place for the gcd bench that follows); allow_no_ntl exists only in
   builds without NTL.

10. Tools.  gcdbench shows every generation of the engine side by side
    (round 3: full apply; round 4: split apply; now: word-level base case),
    the certified chain on one thread, and on as many as tsfactor would
    pick; built with -DHYB_PROF it prints where the time goes.  gcdstress
    adds the base-case-vs-reference comparison (124,000 small pairs of
    every shape and 3100 stress cases at seven hand-over sizes: identical).
    ts_regress.sh checks that no checkpoint file is left behind.

NUMBERS  (136 Mbit, `gcdbench 136279841 588`)
                                  Gord, Round 8 code     sandbox, Round 9 code
                                  (tuned gf2x)           (gf2x from the same source)
    NTL GCD                          51.9 s                 53.4 s
    CERTIFIED, NTL XGCD              60.9 s  +17%           63.4 s  +19%
    engine as of round 3             78.2 s  +51%           70.8 s  +33%
    engine as of round 4             54.8 s   +6%           47.2 s  -12%
    engine now, uncertified                                 37.7 s  -30%
    CERTIFIED chain, 1 thread        55.1 s   +6%           38.8 s  -28%
    CERTIFIED chain, 2 threads       42.1 s  -19%           24.6 s  -54%  (22.7 s in another run)
    CERTIFIED chain, 4 threads       35.1 s  -32%           (two cores only)
  (The sandbox's "round 3/4" rows are those algorithms inside today's code,
  so they already benefit from the direct gf2x calls.)
  Where the time goes now, one thread: products >= 64 kbit 87%, smaller
  products 6.5%, base case 2%, everything else 4.5%.  The serial fraction
  fell from ~50% to ~13%, so four threads should give about 3x instead of
  1.57x: expect the certified gcd near 40 s on one thread and 12-15 s on
  four on Gord's machine.  NOT MEASURED THERE YET.
  Production-size emulated scan (s=320, the oracle's degree-74 factor):
  same result line, gcd certified in 23 s on two threads (Round 8: 660 s on
  the distro gf2x).

MEASURED AND NOT ADOPTED: frequency-domain batching on the CPU
  Round 8 proposed it as the next gcd speedup (est. 25-35%).  gf2x exports
  the interface (gf2x-ternary-fft.h: dft / addcompose_n / ift), and the
  devkit's experiments/fftbatch_bench.cpp times it against separate
  gf2x_mul calls: 0% at 130 kbit operands, 12-25% from 2 Mbit up for the
  ternary FFT, and the Cantor FFT loses everywhere.  That is ~10% of a
  single-threaded gcd.  It does not combine with threads (6 transforms,
  then 4 pointwise passes, then 2 inverses is slower on four cores than 4
  whole products side by side), and the distro libgf2x does not export the
  interface at all.  Threads give 3x; this would give 1.1x; not adopted.
  The GPU hook, which does batch (cheap pointwise products in the additive
  FFT), is the place for it.
  Also not done: Strassen for the 2x2 product (4% on one thread, nothing
  on four).

OPTIONS
  new:      --gcd-base-bits N
  changed:  --gcd-mt N      default automatic (was 1)
            --sched-G N     default measured (was 4000)
            --no-ec         gcd = the engine, uncertified (was NTL)
  Makefile: GF2X_OK / -DHAVE_GF2X (automatic), gf2x_check.cpp,
            gf2_hgcd_base.h in the tsfactor and emulation rules.

VALIDATION (emulation and CPU tools; g++ 13.3; gf2x from Gord's source)
  - tsfactor --selftest 32/32 (certified-gcd suite now 90 gcds: + threads
    sharing the work with the GPU hook); repeated; ASan + UBSan.
  - ts_regress.sh against the ORIGINAL (pre-Round-8) binary: all PASS.
  - gcdstress: base case == reference (124,000 small pairs + 3100 cases),
    all engines == NTL, a third of the cases threaded; ASan + UBSan;
    ThreadSanitizer; products through NTL (GF2X_OK=0); portable carry-less
    multiply (-mno-pclmul).
  - one emulated scan at r=136279841.

NOT VERIFIED
  - Round 9 has not been through nvcc or run on a GPU.  All its changes are
    host code.
  - The gcd on more than two cores; on the tuned gf2x; --gpu-gcd with
    threads at production size (emulated at small sizes only).
  - The self-measuring schedule over a long production run.

WHAT TO RUN FIRST  (details: TESTPLAN_R5.md)
  1. make tsfactor gcdbench gcdstress      (the first line printed should
     end ", gf2x called directly")
  2. ./gcdstress ; ./gcdbench 136279841 588
  3. ./tsfactor --selftest
  4. ./tsfactor --bench 136279841 588 --bench-gcd   (and with --gpu-gcd)
  5. the survivor of Round 8's step 5, default options, -v -v -v

OPEN QUESTIONS / NEXT STEPS
  - The numbers above on the real machine.  If --gpu-gcd beats the threaded
    default there, the recommended invocation gains --gpu-gcd; if not, the
    GPU hook is only worth keeping for machines short of cores.
  - --gpu-gcd-min-bits (2 Mbit) was tuned against a slower CPU side.
  - With the gcd at ~13 s the scan is ~80% of a survivor again: the next
    improvement has to come from the scan (FFT transform, squaring), not
    from the gcd.
  - Control-flow guards; fusing the EC table pass into the Horner kernel
    (1.36 ms vs 1.08 ms per block: that is most of the 0.6%).

---------------------------------------------------------------------------
## Round 10 (one driver, two backends: tsfactor_cpu; tuning at start; per-survivor checkpoints; the optimal schedule)

Written in the error-checking conversation (its round 6), from Gord's
message with the TESTPLAN_R5 logs, the news below, the public repository
(github.com/nclvrps/mersenne_trinomial) and four requests.

THE PRIMITIVE TRINOMIAL
  x^136279841 + x^12338357 + 1 is PRIMITIVE.  tsfactor found no factor
  (result line "12338357 primitive", i.e. none up to the Swan bound r/3),
  and irred confirmed irreducibility; 2^136279841 - 1 is prime, so
  irreducible means primitive.  An independent recheck by a second program
  on other hardware is still worth having before any announcement (a GPU
  irred, ~8 h instead of days, would be the fast way: see the irred
  handoff).

WHAT GORD'S RUN OF ROUND 9 SHOWED  (TESTPLAN_R5: RTX 2070 SUPER, 6-core Zen 2)
  - nvcc builds warning-free; --selftest all PASS; gcdstress PASS.
  - gcdbench 136279841 588 (several runs, spread under 1%):
        NTL GCD                      51.0 s
        CERTIFIED chain, 1 thread    36.3 s   (-29%; predicted ~40 s)
        --gcd-mt 2                   23.7 s   (1.53x)
        --gcd-mt 4                   17.0 s   (2.14x; predicted 12-15 s)
        --gcd-mt 6                   17.1 s   (2.13x)
    One thread came out as predicted; the threads saturate at ~2.1x from 4
    on (the largest products are memory-bound; four of them at once fill
    the memory system).  Recorded so it is not re-predicted.
  - --bench-gcd inside tsfactor: CPU certified 21.7 s, --gpu-gcd 19.1 s.
    --gpu-gcd stays in the recommended invocation (Gord uses it).
  - The step-5 survivor (s=12337659, skip 262): same result line, three
    gcds at 17 s -- but 342 s in all (Round 8: 305 s), because the third
    interval, 5573..21862, scanned 16,000 degrees past the factor at 5816.
    His question: "after resuming, G is reset to the default of 3000 --
    intentional?"  No.  G was measured in-process only; it is now saved in
    the checkpoint (5. below).  And the planning itself was the bigger
    problem (5. below): with the Round 10 schedule this survivor would
    take about 177 s.
  - Results identical across the three step-5 runs and to Round 4.

REQUESTS AND WHAT WAS DONE
  a. Tune automatically at the start of a run, not when resuming unless
     the version or platform changed; --retune as in coarse_sieve; no more
     GF2C_* environment variables.                      -> 3.
  b. r and s in the checkpoint names (as irred), so one s can be set aside
     while others continue.                             -> 4.
  c. Merge factor_modified_20260830.cpp into factor_ec. -> factor_ec 3.01
     (factor_ec/BUILD_ec.md).
  d. factor_ec lags tsfactor (manual -m/-q, no checkpoints, old gcd): port
     the improvements, or unify the CPU and GPU code.   -> unified: 1, 2.
  e. irred: no more reading of 0.20 checkpoints.        -> irred devkit r3.

1. ONE DRIVER, TWO BACKENDS
   tsfactor.cu was 4941 lines, all of it CUDA-compiled.  Now:
     ts_core.h (NEW, ~4300 lines)  everything that is not arithmetic on
                       residues: survivor loop, schedule, gcd pool and the
                       certified gcd, checkpoints, the error-checking logic
                       (the fold algebra), fine DDF, the selftest suites,
                       option parsing and main (ts_main).  No CUDA.
     struct TsDev (in ts_core.h)   the interface.  A backend owns the
                       h-table (m residues), A, C and the checking state
                       (digest D, folds) and implements: init / set_m /
                       set_s / fini; init_h_at, reset_A, horner_C,
                       mul_A_by_C, square_all_step, square_steps; the EC
                       hooks (ec_setup, ec_block_pre, ec_read_block,
                       ec_chain_reset, ec_window_check, ec_verify_table_at,
                       table_fold, hold_fold); transfers of h and A; the
                       hold table; fault injection; time_square/time_mul.
     struct TsPlatform (ts_core.h) what differs between the programs: name,
                       tuning fingerprint, extra options, tune/apply/
                       describe, choose_m, bench, selftest, defaults.
     tsfactor.cu (1495 lines)      the kernels (unchanged), TsGPU, the GPU
                       multiplier hook, the FFT sweep, gpu_bench,
                       GpuPlatform, main.
     ts_cpu.h (NEW)    TsCPU.
     tsfactor_cpu.cpp (NEW)        CpuPlatform and main; g++ only.
     gf2_trin_host.h (NEW)         the host bit-level oracles, moved out of
                       gf2_trinomial_cuda.h so the CPU build includes no
                       CUDA header.
   The state is bit-identical between backends, so a checkpoint written by
   one build resumes in the other (tested both ways, 6. below).

   Evidence that the port changed nothing: ts_regress.sh against the
   ORIGINAL (pre-Round-8) binary, every configuration identical, for both
   builds; selftest all PASS for both; and a new GPU-build suite, "CPU
   backend == GPU backend, step by step", runs TsCPU and TsGPU side by side
   on 10 (r, s, m, k) cases and compares h-table, A, C, D, the table folds
   and the block folds after the init and after every block.
   Latent bug fixed in the port: TsGPU::set_m did not free the hold table,
   so a change of m would have used one of the old size.  A change of m
   can now happen (a resumed survivor keeps its checkpoint's m).

2. tsfactor_cpu
   The same program -- command line, results file, checkpoints, schedule,
   certified gcd, error checking, fine DDF -- with the arithmetic on the CPU
   (ts_cpu.h).  Meant for the CPU stage: coarse_sieve's survivors (skip 55)
   up to d ~ 279, on cheap cloud cores, where it replaces factor_ec.

   Squaring.  BLZ in place, as factor.cpp's fastsqr (factor_modified's
   version of it), for s odd and > 128.  factor.cpp flips an even s to the
   reciprocal trinomial (r - s is odd), and every residue with it -- the
   reason tsfactor's and factor's states could never be compared.
   tsfactor_cpu does not flip: s even gets its own in-place reduction,
   derived in ts_cpu.h's header.  In the scrambled order position
   p >= alpha+1 stands for x^(2p) = x^(2p-r) + x^(2p-r+s).  The first term
   is odd and stays at p.  For s even the second is odd too, at position
   p + s/2, final -- no cascade -- except for the top s/2 positions, where
   2p-r+s >= r and it reduces once more to two even terms, at positions
   p-r+s/2 and p-r+s, both below s.  So: XOR the top s/2 bits into [0, s/2)
   and into [s/2, s); then bits [alpha+1, r-s/2) into themselves shifted up
   by s/2, top down; then the usual interleave.  Odd s <= 128 (64 of the 68
   million) use a generic spread-and-fold squaring (4x slower).
   Products: gf2x_mul, then the two linear fold passes, as word runs.

   Speed.  The first version was 1.6x slower than factor_ec per degree.
   Two causes, both fixed: (a) the hot loops read the geometry through a
   struct reference that stores could alias, so nothing vectorised -- the
   squaring took 8.6 ms against factor_modified's 2.7 ms on the same data;
   rewritten with plain values and __restrict__, 2.6 ms; (b) the table was
   advanced one squaring of every slot at a time (the GPU's order), so each
   step streamed all m residues; TsCPU now squares one slot m times before
   the next (TsDev::square_steps; the GPU keeps the default), working set
   two residues: another 30-40% here.  The reduction became word runs (23
   -> 8 ms).  Numbers (one core of the sandbox's 2.1 GHz Xeon, gf2x built
   from Gord's tree, r = 136279841):
       squaring 2.9-3.4 ms     product 1.1-1.25 s     reduction 8 ms
       m = 19-20 (tuned)       ~0.13 s per degree with checking
       error checking +3.3% (per block: table pass 26 ms, product folds
       18 ms; chain check 66 ms per window)
   Production shape, s = 588, skip 55, maxd 279, run one after another:
       factor_ec 3.01          84.3 s  one interval 56..279, scan 30.3 s
                                       (0.135 s/degree), NTL gcd 48.1 s
       factor_modified 2.02    83.2 s  scan 0.131 s/degree, gcd 49.2 s
       tsfactor_cpu, final     84.7 s  tuning 3.7 s (once per process),
                                       init 5.6 s, one interval 56..293
                                       (m = 17) at 0.130 s/degree, certified
                                       gcd 43.9 s
       tsfactor_cpu, G0 200    135.5 s the same scan rate, but the first
                                       interval was planned before any gcd
                                       was timed: two intervals, two gcds
   So the scan runs at factor's rate, with checking.  With G0 = 200 the
   schedule split 56..279 in two (a coin toss in expectation at G ~ 200,
   a clear loss at the measured G ~ 360 here, ~280 with Gord's 36 s gcd);
   the CPU default is now G0 = 300, which takes one interval.  Expected per
   survivor on Gord's machine: about factor_ec's time minus 15 s (certified
   gcd 36 s against NTL's 51 s), plus the tuning once per process.
   Full-size cross-check: six random s (skip 1, maxd 40), tsfactor_cpu and
   factor_ec 3.01 give identical result lines (factors of degree 2 to 16,
   both parities of s), and again after the speed work.

   Defaults: one process per core (--gcd-threads 1, --pend-max 1, one
   thread inside each gcd, --threads 1); schedule G0 300.  --threads N
   splits the slots between N squaring threads (OpenMP; 2 threads halve
   the table step here).  Memory at r = 136279841, m = 19 (MiB, as the
   start-up line prints them): h-table 325, scratch 227, the hold table
   (m residues, ~310) when an interval has more than one block, and one
   h-table snapshot (309) per interval with an outstanding gcd: about
   1.2 GB per process.  GPU-only options
   (--device, --fft, --legacy-sq, --gpu-gcd) are refused with a message.

   A NEAR MISS, RECORDED.  The word-run reduction first assumed that the
   scratch beyond its logical end was zero.  It is not after the tuning,
   which runs at s = (r-1)/2 and leaves words there; a later, smaller s
   then read them.  The small-r selftests reuse a device only with s
   increasing, so they passed; the first production-size run failed its
   init verification four times in a row and stopped with exit status 3,
   "persistent fault", writing nothing -- the error checking did its job
   on our own bug.  Fixed (reduce() clears the words pass two reads), and
   the product test now reuses one device with s DECREASING (verified to
   fail without the fix).

3. TUNING AT START
   A run times its configurable arithmetic before it scans.  The GPU build
   sweeps the FFT configurations (fused transform, 2-4 Taylor x 2-4
   butterfly levels, plus legacy; seconds) and then times t_sq and t_mul at
   the winner to choose m (even, 8..64); the CPU build times t_sq and t_mul
   (m = round(sqrt(t_mul/t_sq)), 4..64; 4 s at full size).  The record --
   TsTune: FFT configuration, m, t_sq, t_mul and a fingerprint
       "tsfactor round 10 on <GPU name> (sm_XY, N SMs)"
       "tsfactor_cpu round 10 on <CPU model> (N CPUs, T threads)"
   -- goes into every checkpoint.  A run whose first resumable checkpoint
   carries its own fingerprint adopts the record ("tuning: from the
   checkpoint of s=..."); another round or another device tunes again
   (saying why); --retune always tunes; --no-tune takes the compiled-in
   defaults (m still timed unless --m); --fft T,B[,C] | legacy pins the
   FFT (what the GF2C_* variables did); --m pins m.  --retune with
   --no-tune is refused, as in coarse_sieve.  There is no --tune: tuning
   is what a run does unless told otherwise (coarse_sieve has none either).
   GF2C_FFT, GF2C_TAYLOR_LV, GF2C_BFLY_LV, GF2C_FUSE_MAX_CH are no longer
   read (gf2c_read_env is gone; gf2c_clamp_config keeps values in range).

4. PER-SURVIVOR CHECKPOINTS
   PREFIX_r<r>_s<s>.ckpt (+ .1, .2) and PREFIX_r<r>_s<s>.fine (the fine-DDF
   standby file), one set per survivor.  At start the run looks for them
   for every pending s of ITS list; those s go first (list order), the rest
   follow.  Checkpoints of s not in the list are kept and listed ("kept for
   later"); running a list that contains such an s resumes it.  Files of an
   s already in the results file are deleted.  A skip mismatch or an
   earlier format is reported and ignored; a round 1-9 checkpoint
   (PREFIX.ckpt, one per prefix) is recognised and reported with its s
   ("finish it with the earlier binary, or delete it") -- the new format is
   not backward compatible, as agreed.
   One process per prefix: PREFIX.lock is flock()ed for the life of the
   process; a second run with the same --out stops with a message (exit 1).
   The lock file stays behind (deleting it at exit would race; harmless).
   Format version 2: magic "TSFC2", 408-byte header (static_assert-pinned)
   with the tuning record, the measured costs (5.) and reserved space.

5. THE SCHEDULE: EXPECTED-COST OPTIMUM BY DYNAMIC PROGRAMMING
   --sched opt (the default since Round 9) planned with the stationarity
   recurrence k_{j+1} = k_j^2 / k_{j-1} - G, which carries the previous two
   points: when G changes mid-run (measured: 3000 until the first gcd,
   then ~1300) the trajectory keeps the old G's shape.  Expected cost
   relative to the optimum (skip 262, K = r/3, survival S(k) = (16/9)/k;
   simulation reproduces Gord's q sequences exactly):
       Round 9 code, G 3000 then 1300 (his step 5)     +40.9%
       Round 9 code, G fixed at 1300                    +7.0%
       Round 9 code, G fixed at 4000 (Round 6-8)        +5.3%
       linear, q0 = 20                                  +0.7%
       Round 10 (dynamic programming)                    0
   Round 10 solves V(k) = min_{k'>k} [(k'-k) + G + (k/k') V(k')], V(K+1) = 0,
   on a geometric grid (ratio 1.002, ~6000 points to r/3, 11 ms) and
   re-solves when G moves by more than 2%; it plans from wherever the scan
   is, so a resume, a G change or an -Z cap cannot derail it.  For Gord's
   step-5 survivor (G0 1500, then 1300): 263..1062, 1063..2598, 2599..5062,
   5063..8550 -- four gcds and 8288 degrees, ~177 s, instead of 342 s.
   The measured costs (seconds per gcd, per degree) go into every
   checkpoint, and a resumed run starts from them (-v prints "[schedule
   costs from the checkpoint: ...]").  Selftest: DP == brute force on all
   integer partitions for three G, and a full-size solve that is monotone
   and fast.  --sched-rho is accepted and ignored (the DP needs no floor).
   CPU regime (56..279, G ~ 300): one interval, matching Gord's factor_ec
   setting (-q 16 -m 14 covers 56..279 exactly).

6. TESTS AND TOOLS
   selftest: the CPU build runs every scan/EC/gcd suite on TsCPU plus its
   own kernel suites (squaring 3934 cases of both parities and the generic
   path vs the bit-level oracle; product, mul-by-x and squaring on one
   device with s decreasing; word-run fold vs windowed fold, 77 cases); the
   GPU build runs those too, plus the cross-backend suite.
   tests/ts_regress.sh: detects the CPU build ("CPU build" in --version)
   and skips the --gpu-gcd row there; new "linear" row; leftover pattern
   now PREFIX_r*_s*.ckpt*.
   tests/ts_ckpt10.sh NEW (either build; optionally the other build as a
   second argument): per-survivor names; a survivor set aside and resumed
   later; two interrupted survivors resumed first by one run; tuning
   measured at start / adopted on resume / --retune; the schedule's costs
   restored on resume; the prefix lock; a checkpoint of one build resumed
   by the other (tunes again, same result).
   Makefile: `make` builds everything this machine can (the CUDA build is
   skipped with a note when nvcc is missing); `make cpu`, `make cuda`;
   `make emultest` (both selftests, no GPU); `make regress` (both scripts
   on tsfactor_cpu, ORIG=... to compare with an older binary).  Removed:
   the emultest dependency on cantor_cuda (its source is gone: that target
   would have waited on stdin), the dead generic emul-% rule, the undefined
   CPU variable.

VALIDATION (sandbox: g++ 13.3, NTL 11.6 + gf2x from Gord's tree, 2 cores)
  - selftest all PASS: GPU build (emulated) and CPU build (also with
    --threads 2); both under ASan + UBSan.
  - ts_regress.sh against the ORIGINAL binary: all PASS, both builds; the
    CPU build also under ASan + UBSan.
  - ts_ckpt10.sh: all PASS, CPU->GPU and GPU->CPU; ASan for the CPU build.
  - full size: the six-survivor cross-check with factor_ec (identical);
    the production-shape timing above; TsCPU squaring and product checked
    against NTL's SqrMod/MulMod for s = 588, 12338357, 68139920, 200, 101.

NOT VERIFIED
  - Not compiled by nvcc.  The kernels are unchanged and the new host code
    uses nothing beyond C++14, but the CUDA file was restructured: build it
    first (TESTPLAN_R6 step 1).
  - The start-up FFT sweep's choice on the real card (it should pick 3/3,
    the Round 5 winner) and its duration.
  - tsfactor_cpu on Zen 2 or cloud CPUs: the slot-at-a-time order helps
    where two residues (34 MB) fit in the cache; on a Zen 2 CCX (16 MB L3)
    they do not, so expect less gain there than here.

WHAT TO RUN FIRST  (details: TESTPLAN_R6.md)
  1. make; ./tsfactor --selftest; ./tsfactor_cpu --selftest
  2. make regress (and ts_regress.sh ./tsfactor)
  3. ./tsfactor_cpu --bench 136279841 588 against factor_ec on one survivor
  4. the step-5 survivor again (expect ~177 s instead of 342 s)
  5. suspend/resume one survivor of a list (ts_ckpt10.sh does it small)

OPEN QUESTIONS / NEXT STEPS
  - The orchestrator: tsfactor_cpu could replace factor_ec in
    mersenne_trinomial.py's CPU mode (command line in TESTPLAN_R6 step 6);
    not changed here.
  - GPU irred on the TsDev pattern (the irred handoff has the sketch).
  - On the GPU the scan is ~80% of a survivor again: FFT bandwidth and the
    fused EC table pass remain the scan-side items.
  - CPU: at the tuned m the squarings and the products cost the same,
    sqrt(t_sq t_mul) per degree each, so halving either buys ~29%.  The
    squaring now matches factor_modified's; the product is gf2x's.

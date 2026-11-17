## Round EC (error checking on by default; certified, faster gcd) — APPEND THIS SECTION TO tsfactor_handoff.md

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
    selftest 32/32 PASS (16 new suites), 20 consecutive repeats clean,
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
                         there it really is 1/m.)  This is unlike
                         factor_ec, whose init is built from modmuls and
                         was the largest EC cost.
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
  - Port the closed-form init check to factor_ec (its init is modmuls, so
    there it replaces a duplicate init costing ~7.6 s per s with ~2 s).

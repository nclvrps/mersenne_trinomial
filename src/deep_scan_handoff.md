# Deep scan of coarse-sieve survivors: evaluation + revised handoff

Scope note on names. What the Gemini conversation calls a "coarse
sieve" (testing survivor trinomials for factors of degree up to maxd ~
10^4..10^5) is called the DEEP SCAN here, to distinguish it from the
existing `coarse_sieve` program (log-table periodicity sieve to d <=
29..31). The pipeline is:

    coarse_sieve (GPU, d <= 29..31)
        -> survivors file
    trinomial_scan (GPU, this document: k0 < d <= maxd)   [to be written]
        -> found factors + "u" records
    factor.cpp (CPU, gf2x/NTL)  for the rare survivors past maxd

Everything in Part C is specified to implementation level so a future
session (or you) can write `trinomial_scan.cu` directly from it.

---------------------------------------------------------------------------
## Part A. Evaluation of Gemini's analysis

Gemini's overall verdict was "infeasible without a massive effort to
port a GF(2) FFT library." Point by point:

A1. "Writing a GF(2) FFT in CUDA is a massive undertaking, often a PhD
    thesis." OVERSTATED, now empirically: the Cantor-basis additive FFT
    in this package is ~700 lines across three headers, was validated
    against four independent oracles on CPU (13.0 s for a full
    136,279,841 x 136,279,841-bit product on one core), and its GPU
    port passed transform selftests bit-for-bit on your CC 7.5 card on
    the first hardware run. The one hardware bug found was a write race
    in the overlap-add epilogue (scatter form), not anything
    FFT-theoretic; it is fixed by the gather-form k_overlap_add.
    Verdict: the "missing piece" existed ~48 hours after being named.

A2. "Critical mathematical boundary at d = 27/28 because 2^d - 1
    crosses r." WRONG AS STATED. The periodicity method needs only
    arithmetic mod M = 2^d - 1; whether M < r or M > r is irrelevant
    (for M > r the reduction r mod M = r and the sieve is unchanged --
    your own hardware run did d = 28 and d = 29). The real boundary is
    the Theta(2^d) memory/time of the log tables: 8*2^d bytes, i.e.
    8.6 GB at d = 30. The crossover to squarings-based testing is an
    economic one around d ~ 30-35, not a mathematical wall at 28.

A3. "PCIe bottleneck: 17 MB x 100,000 degrees = 1.7 TB per s."
    TRUE ONLY FOR THE ARCHITECTURE GEMINI ASSUMED (per-degree residues
    shipped to a CPU that also does the products). With products
    accumulated ON the GPU -- which is precisely what the FFT enables
    and what trinomial_stage2.cu already does -- the bus carries one
    17 MB accumulator per GCD interval per s. At the interval policy
    below that is ~10 transfers of 17 MB per survivor, ~2 ms each on
    PCIe 3.0 x16. The PCIe argument evaporates.

A4. "~400 polynomial instances fit in 8 GB." IGNORES WORKING SET. Each
    in-flight s needs the running residue F_k, the accumulator ACC, and
    one interval-start checkpoint of F (for localization): ~51 MB.
    Shared FFT scratch at n = 2^24 is ~0.4 GB (two element buffers +
    twiddles + fold ping-pong). Realistic batch on 8 GB: ~130 s values,
    which is ample (see C4: products serialize on the shared FFT
    anyway).

A5. "s > LIMIT needed so reductions don't overlap; __syncthreads
    barriers." UNNECESSARY under the gather formulation used in
    trinomial_stage2.cu: reduction mod x^r + x^s + 1 is exactly two
    fold passes valid for ALL 1 <= s <= (r-1)/2 (which for odd r is
    exactly the search space s <= floor(r/2)); each destination word is
    computed by one thread from source words only, so there are no
    intra-pass hazards and no barriers. No restriction on s is needed.

A6. What Gemini got right: squarings are cheap, linear-time,
    GPU-friendly; block DDF (interval products + one GCD per interval
    + binary-search localization) is the correct algorithmic skeleton;
    the output semantics (smallest d, then lexicographically least
    mask -- your inserted clarification) and the CLI shape are
    sensible. The analysis errs mainly in treating the multiplier as
    unattainable and in siting the products on the CPU.

---------------------------------------------------------------------------
## Part B. The GCD question ("can it all be done on the GPU?")

Short answer: yes in theory, and the sane form is a HYBRID (host
recursion, GPU multiplies) -- but it is the wrong place to spend effort
first, because the GCD can be engineered off the critical path
entirely. The stage-2 CPU handoff in trinomial_stage2.cu was a
scheduling decision, not a feasibility judgment.

B1. Theoretical feasibility. FastGCD/HGCD computes gcd of two r-bit
    polynomials in O(M(r) log r): a divide-and-conquer over 2x2
    polynomial matrices in which essentially all work is polynomial
    MULTIPLICATION at geometrically decreasing sizes, bottoming out in
    O(1)-size classical Euclidean steps. We now have the multiplication
    primitive on the GPU. The natural implementation is: host code owns
    the HGCD recursion and control flow; every multiplication above a
    threshold (~10^5 bits) is dispatched to the GPU Cantor multiplier;
    sub-threshold work runs on the host. Because the cost is a
    geometric series dominated by the top levels, this captures most of
    the attainable speedup for perhaps 500-1000 lines of host code plus
    the existing kernels. A monolithic "GCD kernel" (all control flow
    on-device) is neither necessary nor sensible: deep data-dependent
    recursion and tiny divergent base cases are exactly what GPUs do
    badly, and nothing forces them on-device.

B2. Practical necessity: mostly none, because of frequency
    engineering. A GCD is needed once per interval per s, plus
    O(log interval) during localization of a hit. With geometric
    intervals (C5) the expected number of full-size GCDs per survivor
    is ~10. Each has 17 MB of input; the CPU side (NTL GF2X GCD, built
    on gf2x -- i.e., exactly the machinery inside your factor.cpp) runs
    them from a small thread pool, fully overlapped with GPU compute.
    Budget check with the cost model of B3: the GPU spends ~10-100 s of
    product work per interval per s; one CPU core spends a comparable
    time per GCD; so a handful of cores keeps pace with one GPU, with
    PCIe traffic in the milliseconds. If a future box is CPU-starved,
    B1 is the upgrade path -- the hard part (the multiplier) is already
    written and now hardware-validated.

B3. The actual bottleneck, quantified: it is the per-degree modular
    product, not the GCD. Let k0 be the coarse depth reached (29-31).
    Empirically the least-factor-degree density is ~ (16/9)/d^2, so
    conditional on surviving the coarse sieve,
        P(least degree > k | survivor) ~ k0 / k,
    hence expected scanned degrees to maxd:
        E[#degrees] ~ k0 * ln(maxd / k0)   (~ 243 for k0 = 30, maxd = 1e5)
    and only a fraction k0/maxd (~0.03%) of survivors reach maxd at
    all. Each scanned degree costs one squaring (spread + two folds,
    ~0.2 GB traffic, ~0.5 ms -- negligible) and one accumulation
    modmul: 3 transforms at n = 2^24 plus pointwise and folds, ~200 GB
    of memory traffic in the current (unfused) kernels. On a 448 GB/s
    card (2070 SUPER) that is ~0.4-0.5 s per modmul; ~0.1-0.15 s on a
    ~1 TB/s card (4090-class); proportionally less on HBM parts.
    Per-survivor GPU time (incl. ~2x localization overhead on hits):
    roughly 300-500 products => ~2-4 min (2070S), ~30-60 s (4090).
    Calibrate all of this with one command after the race fix:
        ./cantor_cuda bench          (full-scale modmul components)
        ./trinomial_stage2 bench 136279841 <s>   (squaring + product)
    Comparison point: factor.cpp needs the same NUMBER of products; its
    unit cost is gf2x's M(r) on however many cores you give it. Rough
    expectation: an 8 GB 2070S lands near parity with a good 16-core
    CPU; a modern high-bandwidth card wins by ~5-15x; the biggest
    single software lever is fusing the Taylor cascade levels in shared
    memory (the m^2/4 XOR-pass term dominates transform traffic;
    2-3x reduction is available) -- an optimization, not a correctness
    issue.

    MEASURED CALIBRATION (user's CC 7.5, 8 GB card): cantor_cuda bench
    = 489 ms per full 136,279,841-bit product, squarely inside the
    predicted 0.4-0.5 s band, and equal to ~410 GB/s effective traffic
    -- the kernels are memory-bandwidth-saturated, confirming both the
    cost model and that Taylor-cascade fusion (traffic reduction) is
    the right next optimization.  Derived: ~0.5 s per accumulation
    modmul; E[products/survivor] ~ 29 ln(10^5/29) + localization ~
    300-500 => ~2.5-4 min per survivor; a ~63K-survivor s < 10^6 slice
    ~ 4-5 GPU-months on this card, scaling with memory bandwidth on
    newer parts.  Run `./trinomial_stage2 bench 136279841 <s>` for the
    fold-inclusive per-op numbers.

    Feasibility verdict for the deep scan: YES -- clearly worthwhile on
    >= 12 GB modern cards, and roughly CPU-parity (still useful as an
    offload) on the 8 GB CC 7.5 card.

---------------------------------------------------------------------------
## Part C. Design: trinomial_scan (the program to write next)

C1. CLI and files.

    trinomial_scan <r> <survivors.txt> <k0> <maxd> <out_prefix>
        [--batch N]        max s values in flight (default: auto from VRAM)
        [--interval L]     max GCD interval length (default 4096)
        [--gcd-threads N]  CPU GCD pool size (default: hw_concurrency/2)
        [--state DIR]      checkpoint directory (resume if present)
        [--mem GB]         VRAM budget override
        [--verbose 0..3]

    IMPORTANT: k0 must be the depth the coarse sieve actually REACHED
    (29 on your 8 GB card -- it prints this), not the depth requested;
    scanning must start at k0+1 or factors in the gap are missed.

    Input: one s per line (out.survivors.txt from coarse_sieve).
    Output <out_prefix>.results.txt, one line per s, factor.cpp-style:
        s d p<hex>     least-degree factor found (mask lex-least at that d)
        s u            no factor with degree <= maxd  (hand to factor.cpp)
    Optional <out_prefix>.gcds/: raw gcd polynomial dumps when the
    equal-degree tie-break is deferred (see C6).

C2. Correctness backbone (why first hit = least degree). The coarse
    sieve guarantees no factor of degree <= k0. Scan degrees k = k0+1,
    k0+2, ... in order; a "hit at k" means gcd(prod over the interval
    containing k of (F_j + x), T) is nontrivial, localized to the least
    such k. Since a factor of degree d divides x^(2^k) + x iff d | k,
    the first hit k* satisfies: some factor degree d | k*, d > k0. If
    d < k*, then d in (k0, k*) would itself have produced an earlier
    hit at k = d; contradiction. So d = k*: THE FIRST-HIT DEGREE IS THE
    LEAST FACTOR DEGREE, and moreover every factor dividing
    gcd(F_{k*} + x, T) has degree exactly k* (any proper divisor of k*
    in (k0, k*) is excluded by the same argument). Consequence: the
    final gcd g is a product of one or more degree-k* irreducibles, and
    the tie-break needs only EQUAL-degree factorization of a
    small-degree g (C6). No distinct-degree machinery is needed.

C3. Per-s state machine.

    INIT      F <- x; k <- 0
    RAMP      k0 squarings, no products (microseconds of work)
    SCAN      for k = k0+1 .. maxd:
                F <- F^2 mod T                  (spread + 2 folds)
                ACC <- ACC * (F + x) mod T      (FFT modmul)
                at interval end: snapshot ACC to host ring buffer,
                enqueue GCD job; ACC <- 1; also snapshot F as the
                interval-start checkpoint for possible localization
    on GCD nontrivial for interval (a, b]:
    LOCALIZE  restore F from the checkpoint at a; binary split: scan
              (a, m] rebuilding the sub-product, one GCD; recurse into
              the half containing the first hit; cost ~2x the interval
              in modmuls + ceil(log2 L) GCDs. (The other batch members
              keep streaming; localization is just more work items.)
    FOUND     g = gcd(F_{k*} + x, T)  [one more GCD]; emit via C6.
    EXHAUST   k > maxd: emit "s u".

C4. Memory and scheduling (8 GB, CC 7.5 baseline).
    Per in-flight s: F (17 MB) + ACC (17 MB) + checkpoint F (17 MB)
    = 51 MB. Shared: FFT buffers 2 x 134 MB + twiddles 67 MB + fold
    ping-pong 2 x 34 MB ~ 0.40 GB. Batch B: 0.4 + 0.051*B <= 7.2 GB
    -> B ~ 130. Compute capability 7.5 is fully sufficient (the only
    "exotic" op anywhere is atomicMin(u64) in the coarse sieve, sm_35+).
    Schedule: for each degree k, one batched squaring pass over all B
    residues (a batched spread/fold kernel indexed by (s_slot, word) --
    trivial extension of the stage-2 kernels), then the B accumulation
    modmuls run sequentially through the shared grid-wide FFT (a 2^24
    transform already saturates the device; batching FFTs would only
    shave launch overhead). GCD jobs go to a host thread pool over a
    ring of pinned 17 MB buffers; pool never blocks the GPU (if the
    ring fills, deepen it or add cores -- see B2 budget).

C5. Interval policy. Geometric growth L_i = min(32 * 2^i, --interval),
    resetting after each localization. Rationale: early degrees carry
    most of the hit probability (P ~ k0/k^2 density), so short early
    intervals localize cheaply; long late intervals amortize GCDs where
    hits are rare. Expected full-size GCDs per survivor ~ log2(stop/k0)
    + log2(L_hit) ~ 10. OPTIONAL v2: divisibility covering sets --
    cover all degrees d in (a, b] by testing only k in a set K where
    every d has a multiple in K (e.g. primes in (b/2, b] force |K| >~
    pi(b) - pi(a) for K bounded near maxd); saves up to ~ln(maxd)x in
    products at real bookkeeping cost, and localization must then
    search divisors of the hit k. Correctness of "least degree" is
    preserved only if K is processed so that min-covered-degree order
    is respected; specify carefully before implementing. Not needed for
    v1.

C6. Tie-break / mask extraction. deg(g) = m * k* with m >= 1 (C2).
    If m = 1: mask is g itself; emit s, k*, hex(g).
    If m > 1 (rare): lex-least mask at fixed degree = numerically least
    mask. v1: dump g to <prefix>.gcds/s_<s>.bin and emit a deferred
    marker; a 20-line NTL post-processor (GF2X, EDF via
    CanZass/EDFSplit on a degree-(m*k*) input -- tiny) picks the least
    factor; this matches your existing check-ntl tooling. v2: build the
    same EDF in-process against NTL. Do NOT hand-roll EDF on the GPU;
    g is small (typically k* or 2k* bits... coefficients, i.e. a few
    KB at most) and this is pure CPU convenience territory.

C7. Checkpoint/restart. Reuse the stage-2 state format generalized to
    a batch: per s record {magic, r, s, k, interval_start_k, F words,
    ACC words, checkpoint-F words}; write round-robin every ~1024
    degrees of progress per s; resume = load records, rebuild schedule.
    (51 MB/s-value on disk; a full 130-batch checkpoint ~ 6.6 GB --
    acceptable, or store F + ACC only (34 MB) and accept re-scanning
    the current interval on resume, which is the simpler and
    recommended v1.)

C8. Selftest plan (all runnable under `make emultest`-style emulation
    AND on hardware).
    (a) small-r end-to-end: r = 4423 (and one more odd prime r ~ 10^4),
        synthetic survivor lists; CPU oracle = naive squarings + naive
        Euclid GCD + ascending-mask divisibility scan for the
        tie-break; compare full results files. Construct cases hitting:
        immediate factor at k0+1; factor mid-interval; factor exactly
        at an interval boundary; no factor <= maxd ("u"); at least one
        multi-factor-equal-degree tie (the oracle will find them
        naturally given enough s values).
    (b) localization stress: force --interval 4096 with a known hit at
        a random k inside; verify k* and mask.
    (c) full-scale internal consistency: spread-square == FFT-square
        (already in stage-2 bench) plus ACC recomputation invariance
        across a checkpoint/resume cycle at r = 136279841.
    (d) hardware race audit: every kernel is gather-form or atomic
        (keep it that way; the k_unpack lesson is the reason).

C9. Reuse map (what already exists and is validated).
    - squarings + trinomial reduction kernels: trinomial_stage2.cu
      (k_spread_square, k_fold_pass, window) -- hardware-passed.
    - FFT modmul: gf2_cantor_cuda.h + mul_dev pattern in stage2
      (post-fix; re-verify on hardware).
    - GCD: NTL GF2X GCD (subquadratic, gf2x-backed) via the same
      linkage as factor.cpp; naive Euclid fallback for selftests.
    - packed-key output conventions and hex formatting: sieve_ref.cpp.
    What is genuinely new in trinomial_scan: the batch scheduler, the
    interval/localization state machine, the GCD thread pool, and the
    results/checkpoint plumbing. No new mathematics and no new
    GPU primitives are required.

---------------------------------------------------------------------------
## Part D. State inventory for resumption

Validated on hardware (your CC 7.5, 8 GB): coarse_sieve selftest +
production to d = 29 (check-ntl confirmed; d = 29 stop = memory guard,
by design). Validated on hardware EXCEPT the overlap-add race, now
fixed and emulation-revalidated, pending your re-run: cantor_cuda,
trinomial_stage2. Fully CPU-validated: sieve_ref (oracle + full-log
depth 27, 0 mismatches), cantor_ref (four oracles + 13.0 s full-scale
multiply + residue checks). Files: see README.md table; the race fix
touched gf2_cantor_cuda.h (k_overlap_add), cantor_cuda.cu and
trinomial_stage2.cu (call sites). Build: `make cpu`, `make cuda`,
`make emultest`. Next implementation step: trinomial_scan.cu per Part C
(reuse map C9), selftests C8(a) first, on emulation, before hardware.

---------------------------------------------------------------------------
## UPDATE 2 (after first production runs on real hardware)

Measured on the user's 6-core Zen 2 + CC 7.5 8 GB GPU, r = 136279841:
squaring+reduction 0.69 ms; product+reduction 492 ms (s-independent);
one NTL/gf2x GCD ~62 s; factor.cpp reference ~0.183 s per term
single-core with one GCD per ~210-630-degree interval.

1. LOCALIZATION REDESIGN (obsoletes the C3 LOCALIZE step and most of
   the practical case for B1).  The binary-search localization spent
   one full-size GCD per level plus a final extraction GCD -- 6-8
   serialized 62 s GCDs per find with the GPU idle (observed: silent
   fans, --gcd-threads irrelevant).  All of it is unnecessary: by C2
   plus induction over the previously cleared intervals, EVERY
   irreducible factor of the interval gcd g has degree inside the hit
   interval, so factoring the small g (one CanZass; or squaring mod g
   without NTL) yields the least degree AND the lex-least mask
   directly, in microseconds.  Implemented; discarding stale
   speculative jobs is now non-blocking too (the old drain waited up
   to pending x 62 s after each find; jobs now own their modulus via
   shared_ptr so the next s starts immediately).  Verified: results
   byte-identical to the pre-rewrite scanner, to the CPU oracle, and
   to sieve_ref on the overlapping degree range.  Pathological
   fallback when g is not capturable: emit "s rA-B" (a factor with
   degree in [A, B], unresolved) -- the user's suggested range-only
   mode, kept as the automatic escape hatch.

2. HONEST PERFORMANCE VERDICT, revised with measurements.  gf2x's
   tuned CPU multiplication (0.183 s/term) beats the unfused GPU FFT
   (0.492 s) ~2.7x on this pairing, and a 6-core box can run several
   single-threaded factor.cpp instances.  Even after fix (1), the deep
   scan here costs ~1.5-2.5 min per found survivor (scan to the hit
   interval end + one 62 s GCD, partly overlapped) versus factor.cpp's
   ~100 s per ~210-degree interval per instance times N cores:
   factor.cpp wins the deep scan on this hardware today.  The GPU path
   becomes competitive-to-winning with (a) Taylor-cascade fusion
   (2-3x traffic reduction -> 0.16-0.25 s/modmul on this card), and
   (b) high-bandwidth GPUs (~1 TB/s puts current code at ~0.22 s and a
   fused version at ~0.07-0.1 s/modmul).  The coarse sieve (d <= 31)
   remains an unambiguous GPU win regardless.  Next engineering
   priority if the deep scan matters: the fusion.

3. Interval sizing after (1): larger intervals amortize the 62 s GCDs
   against product overshoot past d*; break-even ~126 products per GCD
   on this pairing.  Defaults now L0 = 64 doubling to Lmax = 4096
   (--l0 / --interval to tune).

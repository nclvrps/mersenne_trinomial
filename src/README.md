# GPU coarse sieve + Cantor FFT stage-2 for x^r + x^s + 1 over GF(2)

Target: r = 136279841 (and any prime r with gcd(r, 2^d - 1) = 1 in the
sieved range, which holds for both 136279841 and 74207281 for all
d <= 48). Output semantics match `factor`: smallest-degree irreducible
factor, ties broken by lexicographically least mask.

## The assessment you asked for

I could not read the shared Gemini conversation: the link serves only a
JavaScript application shell to non-browser clients, so none of the
actual chat content was retrievable. The assessment below is therefore
of the claim as you relayed it (a CUDA coarse sieve is infeasible
without a "PhD-thesis-level" CUDA GF(2) FFT multiplier).

Two things are true at once:

1. **The premise is architecturally wrong for a coarse sieve.** A
   coarse sieve does not need large-polynomial arithmetic at all. For
   each candidate degree d, working in GF(2^d) with a primitive
   polynomial, beta = g^j is a root of a divisor of x^r + x^s + 1
   exactly when j*s = Z(j*r mod M) (mod M), Z the Zech logarithm,
   M = 2^d - 1. That is an arithmetic progression in s of modulus
   M/gcd(j, M): one table build plus one pass over cyclotomic-coset
   representatives per degree marks *every* s simultaneously. Cost is
   ~O(2^(depth+1)) per run, nearly independent of smax. The CPU
   reference here does the entire s < 10^6 slice to depth 27 in ~2
   minutes on one core; the CUDA port exists to push depth to 30-31 and
   smax to (r-1)/2, not because the math is heavy.

2. **The Cantor multiplier is still the right "missing piece" -- for
   stage 2.** Deep-testing the survivors (squares-and-products, as in
   `factor`) genuinely needs fast multiplication mod T_s, and a
   GPU-friendly GF(2)[x] FFT is the enabling component. But it is a
   few hundred lines, not a thesis: GF(2^64) is a Cantor field
   (GF(2^(2^6))), so the full basis chain b1 = 1,
   b_{i+1}^2 + b_{i+1} = b_i of length 64 exists -- constructed and
   verified here by Gaussian elimination -- and the Gao-Mateer
   recursion on that basis needs no twiddle scaling, no bit reversal,
   and reduces to contiguous range-XORs plus one butterfly table.
   GCDs stay on the CPU (gf2x FastGCD, your existing factor.cpp flow);
   only squarings and products belong on the GPU.

## Files

    common/gf2_small_field.h    GF(2^d) ops, d <= 31 (host + device)
    common/gf2_field_setup.h    irreducibility/primitivity, primitive poly search
    common/gf2_cantor_core.h    GF(2^64) mod x^64+x^4+x^3+x+1: portable
                                clmul (masked integer mults) + PCLMUL path,
                                butterflies (host + device)
    common/gf2_cantor_engine.h  Cantor basis, forward/inverse additive FFT,
                                chunked GF(2)[x] multiply, test oracles
    common/gf2_cantor_cuda.h    CUDA kernels + orchestration for the FFT
    common/cuda_emul.h          single-threaded CUDA emulation (validation
                                without a GPU; see `make emultest`)
    sieve/sieve_ref.cpp         CPU sieve: selftest / run / validate
    sieve/coarse_sieve.cu       CUDA sieve (best[] resident on device,
                                atomicMin packed keys, two-level marking)
    cantor/cantor_ref.cpp       FFT validation driver: selftest / bigtest
    cantor/cantor_cuda.cu       CUDA FFT driver: selftest / bench
    stage2/trinomial_stage2.cu  squares-and-products accelerator:
                                selftest / run / bench
    common/gf2_trinomial_cuda.h shared trinomial kernels + GPU context
                                (spread squaring, fold reduction, FFT
                                modmul), used by stage2 and the scanner
    common/gf2_gcd.h            host GCD layer: naive backend + helpers
    common/gf2_gcd_ntl.cpp      NTL backend (subquadratic GCD, CanZass
                                EDF); auto-detected by the Makefile
    scan/trinomial_scan.cu      DEEP SCAN of coarse-sieve survivors:
                                least factor degree in (k0, maxd] +
                                lex-least mask, or "u"; speculative
                                interval GCDs on a CPU pool; hits are
                                resolved by factoring the SMALL interval
                                gcd (no further full-size GCDs, no GPU
                                work); resume support (design:
                                deep_scan_handoff.md Part C + updates)
    common/ntl_check.cpp        tiny link check for NTL auto-detection
    Makefile                    cpu / cuda / emultest targets

## Validation record (everything run in this container)

CPU sieve (`sieve_ref`):
- selftest vs an independent brute-force oracle (direct irreducible
  enumeration, per-trinomial evaluation): r = 136279841 (d <= 11,
  s <= 30000), r = 74207281 (d <= 10), r = 101 (d <= 9) -- 0 mismatches.
- `validate` against your verified factor log for r = 136279841,
  all 999,999 lines: depth 20 -> 913,104 lines matched exactly (degree
  AND mask, including every tie-break), 86,895 deeper lines confirmed
  unmarked; depth 27 -> 935,227 matched exactly, 64,772 confirmed
  survivors; 0 mismatches at both depths. Depth-27 run: ~118 s, one core.
  Reproduce with:  ./sieve_ref validate 136279841 27 999999 <log>

CPU FFT (`cantor_ref`):
- gf64 multiply: PCLMUL == portable(masked-int) == bit-serial, 10^5 rand.
- Cantor basis: chain length 64 (exactly the theoretical value for
  GF(2^(2^6))), chain relations + linear independence verified.
- round-trip + linearity m = 1..12; FFT == naive schoolbook (300 random
  <= 2500 b); FFT == independent cl64 word-comb (0.1M..1M bits);
  associativity at 200k bits.
- full-scale: 136,279,841-bit x 136,279,841-bit multiply, FFT 2^24,
  **13.0 s on one CPU core**, verified by residues modulo three
  independent degree-61 irreducibles.

Deep scanner (`trinomial_scan`, emulation + this container's NTL):
- GCD backends: naive vs NTL cross-check on constructed common-factor
  instances (validates the word<->GF2X byte conversions empirically).
- end-to-end vs an independent CPU oracle (per-degree naive GCDs) at
  r = 4423, k0 = 5, maxd = 200: 104 survivors, 102 factors + 2 "u",
  including 3 multi-factor equal-degree ties resolved identically by
  NTL CanZass (scan) and ascending-mask enumeration (oracle); 0
  mismatches under both the default and a stress (L = 8) interval
  policy, exercising deep localization.
- pipeline cross-validation: sieve_ref survivors (Zech-log method)
  agree exactly with the oracle's survivor set, and scan results with
  d <= 11 are line-for-line identical (degrees AND masks) to a
  depth-11 sieve_ref run -- two disjoint algorithms, same output.

CUDA programs (no GPU in this container -- see caveats):
- all three validated by *functional* single-threaded emulation of the
  CUDA runtime (`make emultest`), which executes the actual kernel
  code with real launch geometry sequentially:
  - cantor_cuda selftest: transforms m = 1..20 and end-to-end
    multiplies 1 b .. 8 Mb, all bit-identical to the host engine.
  - coarse_sieve --selftest (r = 136279841, depth 20, s <= 999999):
    best[] byte-identical to the in-process CPU reference across all
    999,999 entries -- this exercises the GPU-only antilog construction
    (per-thread product over g^(2^t)) and the two-level marking.
  - trinomial_stage2 selftest: 40 squarings at r=31,s=3 vs an
    independent u64 oracle plus the known-answer Frobenius identity
    F_31 == x; 24 squarings and an interval product at r=4423 vs a
    naive bit-serial oracle.

## Build and first runs

    make cpu                 # sieve_ref, cantor_ref (g++ only)
    make cuda                # coarse_sieve, cantor_cuda, trinomial_stage2
    make emultest            # run all three CUDA selftests without a GPU

On the GPU box, in order:

    ./cantor_cuda selftest           # GPU vs host engine, many sizes
    ./coarse_sieve --selftest        # full sieve GPU-vs-CPU, 10^6 s values
    ./trinomial_stage2 selftest
    ./trinomial_scan --selftest      # end-to-end vs CPU oracle at r=4423
    ./cantor_cuda bench              # 136M-bit multiply timing + residues

Production coarse sieve (s up to (r-1)/2 = 68,139,920):

    ./coarse_sieve 136279841 30 68139920 out --found

- Device memory: 8*2^d bytes of tables at degree d (8.6 GB at d = 30,
  17.2 GB at d = 31; the program checks free memory and stops cleanly
  at the deepest degree that fits) + 8*(smax+1) bytes for best[]
  (545 MB at full smax).
- `out.survivors.txt`: the deep-search worklist (s with no factor of
  degree <= depth). `out.found.bin` (optional): u64 packed key per s,
  little-endian, key = (d << 48) | mask, UINT64_MAX = survivor.
- Expected time: table build is O(2^d) with ~popcount(i) field mults
  per element; the class pass is ~M/d minpoly computations of O(d^2)
  mults each. At d = 30 this is a few times 10^10 32-bit clmul-free
  field mults -- minutes on a modern card, versus CPU-hours.

Deep scan of the survivors (the factor.cpp-style search for degrees
k0 < d <= maxd; see deep_scan_handoff.md for design and cost model):

    ./trinomial_scan 136279841 out.survivors.txt 29 100000 deep \
        --gcd-threads 8 --state deep.state
    # k0 MUST be the depth coarse_sieve actually reached (it prints it).
    # deep.results.txt: "s d p<hex>" / "s u" (plus rare "s rA-B" =
    # unresolved factor with degree in [A,B]); append-mode, resumable.
    # --l0 / --interval set the geometric interval lengths (default 64
    # doubling to 4096); one CPU GCD per interval, overlapped.
    # Production r requires the NTL GCD backend: install libntl-dev
    # (the Makefile auto-detects it), ideally NTL built with gf2x
    # (NTL_GF2X_LIB=on) exactly as for factor.cpp.

Stage 2 primitives standalone (per s):

    ./trinomial_stage2 run 136279841 <s> <k0> <k1> out_s [state.bin]
    # then on CPU: gcd(out_s.acc as an r-bit polynomial, T_s) with
    # gf2x/NTL FastGCD; nontrivial => a factor with degree in (k0, k1]
    # divides T_s; narrow the interval to isolate k; the factor product
    # for that k is gcd(F_k + x, T_s) with F_k from out_s.fk.

  State checkpointing every 1024 squarings when a state file is given
  (resume by rerunning the same command). `bench` mode cross-checks
  spread-squaring against FFT-squaring at full scale on-device and
  reports per-op timings.

## Caveats, honestly stated

- Hardware field report (CC 7.5, 8 GB): coarse_sieve --selftest passed
  and a full production run produced factors through d = 29, confirmed
  valid with check-ntl (the stop at 29 is the memory guard working as
  designed: d = 30 needs 8.59 GB of tables; a >= 12 GB card reaches 30,
  a 24 GB card reaches 31). The first hardware run of cantor_cuda /
  trinomial_stage2 exposed a write race between adjacent odd chunks in
  the scatter-form overlap-add -- exactly the class of bug sequential
  emulation cannot exhibit. Fixed by reformulating as a gather
  (k_overlap_add: one thread owns each output word); re-run
  `./cantor_cuda selftest` and `./trinomial_stage2 selftest` on
  hardware after this fix. All other kernels were audited for the same
  pattern: every remaining write is either to thread-owned locations
  (gather form, bijective index maps) or through atomics.
- Measured on the same card: `cantor_cuda bench` = 489 ms for a full
  136,279,841 x 136,279,841-bit product (FFT 2^24), residues PASS.
  That is ~200 GB of kernel memory traffic in 0.489 s -- an effective
  ~410 GB/s, i.e. the transform is already running at the card's
  practical memory-bandwidth ceiling.  Consequences: (a) the 13.0 s
  single-core CPU multiply maps to a 26.6x single-GPU speedup; (b) the
  remaining optimization lever is traffic reduction (fusing Taylor
  cascade levels in shared memory, est. 2-3x), not occupancy tuning;
  (c) deep-scan calibration: ~0.5 s per accumulation modmul => roughly
  2.5-4 min per survivor at maxd = 10^5 on this class of card (see
  deep_scan_handoff.md B3).
- Toolchain notes (GCC 13+, incl. g++ 15.2 / Ubuntu 26.04, CUDA hosts):
  nvcc's host pass needs the ISA flags for the PCLMUL/SSE4.1
  intrinsics, so NVFLAGS now defaults to `-Xcompiler -march=native`
  (or use -mpclmul,-msse4.1); without them gf2_cantor_core.h now falls
  back cleanly to the bit-identical portable carry-less multiply
  instead of a hard always_inline error.  A -Wstringop-overflow false
  positive from const-propagated clones of poly_gcd_naive (g++ 15.2)
  is fixed with an explicit provable size clamp; the header compiles
  clean at -O3 -Wstringop-overflow=4.
- NTL specifics from a local-install field report: detection now
  TRY-LINKS common/ntl_check.cpp against "-lntl -lgmp", then
  "-lntl -lgf2x -lgmp" (an NTL built with NTL_GF2X_LIB=on references
  gf2x and needs it spelled out at link time), then with -lpthread;
  the try-compile also handles /usr/local installs.  Overrides:
  make NTL_OK=0, or make NTLLIB="-L/opt/ntl/lib -lntl -lgf2x -lgmp".
- Deep-scan performance, measured honestly (6-core Zen 2 + the CC 7.5
  GPU): one NTL/gf2x GCD at r bits is ~62 s; a gf2x CPU modmul is
  ~0.183 s/term single-core versus 0.492 s on this GPU, so tuned gf2x
  multiplication currently BEATS the unfused GPU FFT ~2.7x per product
  on this pairing.  The first scanner build additionally serialized
  6-8 full-size GCDs per find inside a binary-search localization;
  that design is now removed -- hits resolve by factoring the small
  interval gcd (all its factors provably have degrees inside the hit
  interval), cutting per-found-survivor wall from the observed 5.5-13
  min to an expected ~1.5-2.5 min.  Straight verdict: even after the
  fix, multi-instance factor.cpp remains the stronger deep-scan tool
  on that CPU/GPU pairing; the GPU path wins after Taylor-cascade
  fusion (2-3x product-traffic reduction) and/or on high-bandwidth
  cards, while coarse_sieve (d <= 31) remains an unambiguous GPU win.
  Zen 2's slow PDEP/PEXT is irrelevant: nothing in this codebase (or
  gf2x's configuration) uses BMI2.
- Marking order is irrelevant by construction (atomicMin over packed
  keys is commutative/idempotent), so the emulation's serial execution
  does not mask ordering bugs in best[].
- Tunables: INLINE_MARKS / HEAVY_CAP in coarse_sieve.cu; FFT butterfly
  passes are memory-bound and could be fused/shared-memory-tiled for
  another constant factor; k_classes could compact coset-minimal
  survivors before the minpoly stage to remove divergence. None of
  these affect correctness.
- sieve depth is capped at d <= 31 by the u32 element representation
  (and by memory: d = 31 already needs 17.2 GB). Past experience with
  the s < 10^6 log says depth 27 already eliminates ~93.5% of s; the
  marginal yield per extra degree decays like ~1/d.

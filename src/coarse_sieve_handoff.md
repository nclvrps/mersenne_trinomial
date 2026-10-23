# coarse_sieve (-DWIDE_FIELD), round 5: auto-checkpoint, the hidden floor, and degree 49

**Baseline: round 4 (2026-10-22), which you validated on hardware.**
- All selftests PASS.
- d=36 4.40 s, d=40 45.42 s, d=42 150.14 s, d=44 668.65 s.
- A/B at d=44: classic tower 774.48 s vs subfield tower 666.77 s, a 1.16x gain against the ~2.5x I had predicted.

## Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | Round 5. Replaces round 4's file. |
| `gf2_dlog.h` | Round 5. Adds the exact-quotient helpers used for the irregular classes, and the RR engine (the half-gcd split and the table builders). |
| `gf2_dlog_selftest.cpp` | New part 8 tests RR: the split at every d up to 55, the factor-base sieve, and the logs. New optional 6th argument `rr_dmax` (default 47). |
| `coarse_sieve_round5.patch` | The same changes as a patch against round 4 (flat paths). |

The `Makefile`, `cuda_emul.h` and the other headers are unchanged from round 4.

## TL;DR

* **Auto-checkpoint (as you asked).**
  - Every run now saves to `<out_prefix>.ckpt` every 15 minutes and on Ctrl-C or SIGTERM.
  - Running the same command again continues from that file.
  - The file records what run wrote it: r, smax, depth, `--min-depth`, and a hash of the contents of the `--load`/`--load-survivors` file. A different command line refuses to start rather than mix two runs.
  - `--fresh` starts over and keeps the old file as `*.old`. `--no-checkpoint` turns checkpointing off.
  - Round-4 checkpoints are continued too. Details in section 4.
* **Why round 4 gained only 1.16x.**
  - At even d, 3 divides 2^d − 1, so about half of all classes have gcd(u, 2^d − 1) > 1, and ~15% of all classes need a modular inverse.
  - Round 4 computed that inverse per class by extended Euclid. In the real nvcc 13.3 SASS, each of its ~25-29 steps calls the software 64-bit division (~80 instructions), and a bit-serial multiply follows.
  - That is ~3,000-3,800 warp-instructions per warp-step, against ~1,200-1,500 for all the other work.
  - In 94-98% of warp-steps at least one of the 32 lanes needs it, so the whole warp waits through it.
  - Round 4 removed DRAM reads, but this was the real floor (section 1).
* **The fix.**
  - Those classes now join the batch inversion that all the other classes already use. It needs exact division by multiplication, a "unit lift" and a reduction without division (section 2).
  - The irregular cost drops to ~100-300 instructions per warp-step, about **2.9x fewer instructions in all** at even d.
  - **Estimate: d=44 in ~4-7 min (round 4: 11.1 min).**
  - `COARSE_IRREG=r4` restores the old path, and the tuner times both anyway.
* **Degree 49 can now be sieved.** It uses a new engine, RR (rational reconstruction):
  - Half an extended Euclid writes every field element as y = r/t, with deg r and deg t both ≤ 24. Then log y = L[r] − L[t].
  - L is a table of the logs of *all* 2^25 polynomials of degree ≤ 24 (256 MB). There is no smoothness test and no failure case.
  - Building L needs the logs of only the 1.47M irreducibles of degree ≤ 24. Those are the expensive many-giant-step logs that made d=49 impossible for 1.15e13 classes. For 1.47M of them they are affordable.
  - **Estimates on your card: table build ~1-2 min, sieve ~4-6 h.**
  - After that, degrees 50-55 run without the gap caveat (`--seed-covers-49`). Section 6.
* **Auto-tuner, for your 2070 SUPER and for newer GPUs.**
  - At each degree ≥ 40, the candidate kernels first run on the same sample of ~2^25 classes, taking seconds. The fastest one runs the degree.
  - Candidates: the tower variants; PH vs RR at odd d; the round-4 irregular path.
  - The `[tune ...]` line prints ns/class for every candidate, so every run doubles as an A/B.
  - RR may also beat PH at d = 41..47 on your card; those lines will show it. Section 5.
* **Smaller changes:**
  - About 33 KB of rarely run code per kernel moved out of the hot loops.
  - A one-time "[~x h to go]" note on long passes.
  - `--selftest-logs rr`.
  - The header documentation is brought up to date.
  - Output files and their formats are unchanged, so nothing downstream (tsfactor's inputs) needs to change.
* **Validation done here:**
  - 67 emulated regression checks, all byte-identical or passing.
  - Host selftest ALL OK.
  - RR at d=49 end to end in the emulator: PASS. RR against the round-4 paths at d = 30..33: byte-identical (section 7).
  - nvcc 13.3 sm_75: both builds warning-free, no spills.
* Your PS questions are in section 10, and my opinion on "last round?" is in section 9.

## 1. What held round 4 back

The question was why removing two of three DRAM reads gained only 16%. To answer it I assembled the real CUDA 13.3 toolchain here (nvcc, ptxas, nvdisasm from the PyPI wheels). Then I read the d=44 kernel's SASS with line info.

In the dominant loop, regular classes cost ~1,200-1,500 warp-instructions per warp-step (32 classes, one per lane), including the subfield-tower log.

Classes with gg = gcd(u, M) > 1 took `logs_irregular`. When gg | v, that path computes (j/gg)^-1 mod n by extended Euclid, ~25-29 iterations. Each iteration calls `__cuda_sm20_div_s64`, the software 64-bit division, ~80 instructions. A ~40-step double-and-add multiply follows.

At even d, about half of all classes are irregular (3 | 2^d − 1, and 5 | 2^d − 1 when 4 | d), and ~15% need the inverse. A simulation over random warps, using the SASS costs, gives:

| d | warp-steps where some lane needs the inverse | max Euclid iterations | irregular warp-instr / warp-step |
|---|---|---|---|
| 40 | 96% | 24.7 | ~3,060 |
| 42 | 94% | 25.0 | ~3,180 |
| 44 | 98% | 27.8 | ~3,570 |
| 46 | 98% | 28.9 | ~3,790 |
| 48 | 95% | 28.4 | ~3,790 |

So at d=44 more than 2/3 of the issued instructions served ~15% of the classes. That is why round 4's change, which cut memory traffic, moved the time so little.

The odd degrees are almost unaffected: 3 does not divide 2^d − 1. d=45 and d=51 have 7 | M, so about half their warp-steps take the path. At d=47, 49, 53 the path is rare.

## 2. Irregular classes, done in the batch (round 5)

Setup: j = u·r^-1 mod M and v = Z(u). The class's progression exists iff gg | v, and then

    s0 = (v/gg) · (j/gg)^-1  mod n,   n = M/gg.

Round 5 computes this without any division:

1. **Exact quotients.**
   - gg is odd, so dividing by it exactly is multiplying by its inverse mod 2^64 (ginv).
   - That gives n = M·ginv, jg = j·ginv and vg = v·ginv, each mod 2^64.
   - The divisibility test comes free: gg | v iff vg < n (the classic divisibility test by multiplication; see Hacker's Delight, ch. 10).
   - gg and ginv are built from the primes of M that divide u. Those primes are found by one multiply-and-compare each, branch-free.
2. **Unit lift.**
   - J = jg + t·n, with the least t ≥ 0 that makes J prime to the primes of gg that do not divide n. Usually t = 0.
   - J ≡ jg (mod n) and J is a unit mod M. So J enters the Montgomery batch product like a regular j, and J^-1 mod M reduced mod n is jg^-1 mod n.
3. **Reduction without division.**
   - x mod n = x − n·floor(x·gg/M).
   - floor(·/M) for M = 2^d − 1 is two shifts and a carry: write x = A·2^d + B, then floor(x/M) = A + [A + B ≥ M].

About 100-300 instructions per warp-step remain for the irregular classes. Results are identical to round 4, and the selftest compares the two paths class by class.

## 3. Cold code out of line

The logs kernels were 140-180 KB of SASS, and the code their inner loop actually touches was ~46 KB. For scale, Jia et al.'s Turing microbenchmarks measure ~16 KB of L0 instruction cache per partition and ~46 KB of L1 instruction cache per SM.

The parts that almost never run are now `__noinline__` functions:
- marking heavy progressions,
- the verification-failure path,
- the tower's cold branch.

That is ~33 KB less per kernel. The tower kernels are now ~108 KB, PH 130-142 KB and RR 84-92 KB, with registers unchanged: tower ≤ 128, RR 115-121, PH up to 250, and no spills. I expect only a few percent from this, and it is folded into the estimates.

## 4. Automatic checkpointing

    ./coarse_sieve_wide 136279841 49 r5_49 --load-survivors S48.survivors.txt --min-depth 49
    # ... Ctrl-C, a reboot, a spot reclaim ...
    ./coarse_sieve_wide 136279841 49 r5_49 --load-survivors S48.survivors.txt --min-depth 49
    found checkpoint r5_49.ckpt (written 2026-10-24 03:24, degree 49, chunk 25600 of 131072):
      continuing that run (--fresh starts over)

* **Where and when.**
  - The file is `<out_prefix>.ckpt`, or `--checkpoint <file>`.
  - It is saved every 15 minutes (`--checkpoint-mins N`; 0 means on signals only), on SIGTERM, and on the first SIGINT.
  - A second Ctrl-C exits at once.
  - In logs mode a stop waits for the chunk in flight (seconds).
  - As before, saves are atomic and `--checkpoint-keep` generations are kept (default 3).
  - A `--load-survivors` run writes small delta files (0.25 MB in the test here). A from-scratch run writes full images (8 bytes × smax, so up to 545 MB per generation).
* **Same run or not.** When the file exists, the run compares it with its own r, smax, depth, `--min-depth` and the contents of its seed file:
  - If they match, it continues.
  - If not, it says why and exits with status 1, leaving the file alone. For example: `checkpoint K.ckpt belongs to a different run: it was written by a run to depth 23, this run goes to depth 24`, followed by what to do.
  - `--resume <file>` still takes any checkpoint explicitly, as before.
* **Lock.** `<file>.lock` (flock) stops a second run on the same checkpoint.
* **Cleanup.** A finished run deletes its checkpoint and the lock.
* **Round-4 checkpoints (v2)** are continued automatically. They carry less identity information, so they are checked on r, smax, depth and what they do record about the seed. An old full-image file from a seeded run is refused with a pointer to `--resume`.
* **One behavior change to know about:** rerunning a command to recompute from scratch now resumes instead, if a checkpoint is lying around. Use `--fresh` for that.

Tested here (K1-K8 in the regression):
- stop and resume,
- refusal on four kinds of mismatch,
- `--fresh`, `--no-checkpoint`,
- a round-4 binary's checkpoint resumed by round 5,
- the lock,
- a real SIGINT with two successive continuations, byte-identical to an uninterrupted run,
- the narrow build's bucketed path.

## 5. The auto-tuner

Which variant is fastest depends on the card: L2 size (an RTX 4090's 72 MB holds tables your 4 MB L2 cannot), shared memory and integer throughput. So for every degree d ≥ 40:
- each candidate runs the same sample of ≥ 2^25 classes on GPU 0 (a few seconds),
- the fastest one runs the degree,
- a line like this shows the numbers:

      [tune d=44 on GPU 0, N classes, T s: GF(2^11) x ns/class,
       GF(2^11)+r4-irregular y ns/class, GF(2^2) w ns/class, classic z ns/class -> GF(2^11)]

The candidates are:
- **Tower degrees:** the subfield tower over the largest two divisors n of d/2, and the classic tower.
- **Odd degrees:** PH and RR.
- **Where irregular classes are common:** the round-4 irregular path too, kept as a safety net and as the A/B number.

All candidates compute identical results; the sample's marks are simply repeated by the real pass. `--no-tune` (or `COARSE_TUNE=0`) turns tuning off and `COARSE_TUNE_MIN=d` moves the threshold. `COARSE_TOWER_SUBN`, `COARSE_IRREG` and `COARSE_METHOD` pin their knob, as before.

d=49 has only one candidate (RR), so it prints no tune line. It gets the ETA note instead.

On the "auto-configure for modern GPUs" idea: nothing else in the code is specific to the 2070 SUPER. Launch shapes come from the occupancy API and table budgets from `cudaMemGetInfo`. `-arch=native` compiles for whatever card is present, and multi-GPU already splits chunks.

## 6. Degree 49: the RR engine

**The idea.**
- Work in GF(2^d) = GF(2)[x]/q. For y ≠ 0, run the extended Euclidean algorithm on (q, y) and stop at the first remainder r of degree ≤ dh = ⌊d/2⌋.
- Then r ≡ t·y (mod q) with deg t ≤ d − 1 − dh ≤ dh. So y = r/t and

      log y = L[r] − L[t]  (mod 2^d − 1),

  where L[P] = log_g P for every nonzero P of degree ≤ dh.
- Every polynomial of degree ≤ dh is in the table, so there is no smoothness condition and no failure case.
- This is index calculus with a *complete* factor base. The y = r/t step is Blake, Fuji-Hara, Mullin and Vanstone's 1984 trick for GF(2^127), where it raised the chance that both halves are smooth. At d ≤ 55 the entire factor base, and every product of its members up to degree dh, fits in memory, so nothing is left to chance.

**Per class.**
- One half-gcd, ~25 shift-and-xor steps, and two random reads in L: 256 MB at d=49, 2 GB at d=55.
- No PH exponentiations.
- The bit-sliced g^u, FKM enumeration, batch inversion and progressions are shared with the other engines.

**Building L.**
- First the logs of the irreducibles of degree ≤ dh are computed by Pohlig-Hellman on GPU 0, one log per lane with work stealing (`k_rr_fblogs`).
- Then L[f·Q] = L[f] + L[Q] fills in everything else, via a smallest-factor sieve on the host.
- For d = 49, M = 127 × 4432676798593. On an 8 GB card PH's plan uses a 2^27-entry baby-step table (4 GB), which means up to 33,027 giant steps per log.
- Over 1.47M logs that is ~2.4e10 giant steps. Each is a random read of the 4 GB table, so a minute or so at the read rates your card showed (my estimate).
- Host side, measured here: the sieve takes 3.0 s and the fill 2.3 s.
- Every factor-base log is verified on the device (g^log = f), and 4096 table entries are checked on the host. During the sieve, 1 log in 256 is re-verified as always.
- A build that took over two minutes is cached in `coarse_rr_d49.tbl` (`COARSE_RR_DIR=dir` moves it). On your card the build probably won't take that long.
- The progress lines look like `[RR table for d=49: 1465020 irreducibles of degree <= 24 (sieve 2.9s); their logs by PH, up to 33027 giant steps each]`, then percentages, then `[RR table for d=49: 256 MB, factor-base logs ..s, fill ..s, verified]`.

**Memory.**
- During the sieve: best[] (545 MB) plus L (256 MB).
- During the build only: the 4 GB baby-step table, freed before the sieve.
- Host: ~0.5 GB extra.

**Time.** 1.15e13 classes at two random reads each. Your round-3 tower runs sustained 1.4-1.55 G random reads/s, which gives ~1.3-1.45 ns per class, so **~4.2-4.6 h, up to ~6 h** if the half-gcd does not hide behind the reads. The `[~x h to go]` note after the first 1/64 of the pass will say within minutes.

**Options.**
- d=49 runs whenever it is in [min-depth, depth]; no flag is needed.
- A run that *starts* above 49 must say what its seed covers:
  - `--seed-covers-49`: the seed comes from a run through 49.
  - `--allow-degree-gap`: the old meaning. A factor found at 50..55 is the least-degree factor only where there is no degree-49 factor.

**Other degrees.**
- At the other odd degrees RR is a candidate for the tuner.
- At d=47 my rough estimate is RR ~1.1-1.6 h vs PH's 1.5 h (your round-3 5,411 s), so it is close either way. The tune line decides.
- At even d RR would need two reads where the tower needs one, so it is not offered there.

## 7. Validation done here

* **Host, `gf2_dlog_selftest`: ALL OK (4-6 min here).** Part 8:
  - the half-gcd contract (r = t·y, both degrees ≤ d/2, and the fast and generic splits agree) at every d = 3..55, 159,000 checks;
  - the factor-base sieve against trial division for d/2 ≤ 14;
  - RR logs at every d = 3..47 (every element up to d=20, 20,000 sampled elements above that): 0 wrong.
* **Emulated program, `regress.sh`: 67 checks, all PASS. Output files are byte-identical wherever compared.**
  - **Earlier rounds' checks:**
    - all selftest variants, including `--selftest-logs rr`;
    - both kernels against the round-1 oracle, from scratch to 22 and seeded 19..23;
    - mid-degree stop and resume; 2 emulated GPUs; fault injection; buckets;
    - every inner subfield at d=24;
    - cross-version checkpoints.
  - **New in round 5:**
    - the irregular path against round 4's: scratch→22, seeded 19..23, and d = 24..28;
    - every tuner pick (4 at d = 20..26, plus a tuned run stopped and continued with a different pick);
    - RR + tower against the round-1 oracle at 19..23;
    - RR = PH at d = 24..29 (found.cert and survivors);
    - each PH/RR pick at d = 21..25;
    - an RR run stopped and continued;
    - K1-K8 checkpoint tests (section 4).
  - `--selftest-logs` itself now also checks the irregular lift against extended Euclid at every d = 4..55.
* **d = 49 in the emulator — PASSED.** This was `--selftest-logs rr` with the RR checks raised to 49 and the 4 GB plan an 8 GB card gets.
  - The table was built through the real d=49 BSGS plan: 1,465,020 factor-base logs with up to 33,027 giant steps each, every one verified, then the fill (2.4 s) and the host spot check.
  - The engine: 512 consecutive exponents with log(g^u) = u, 0 wrong. The batched kernel's own check of the 1,970 classes in a stretch: ok.
  - The same run also passed the rotation test (d = 2..55), the irregular-lift check (163,014 lifted classes, 16,925 of them needing t > 0) and the depth-20 sieve against the CPU reference with RR at odd d.
  - The factor-base logs took 81 min single-threaded in the emulator, ~2.4e10 giant steps at ~0.2 µs each. That is CPU emulation; the GPU should need a minute or so.
* **d = 30..33, seeded, smax = 299,999 — PASSED.** RR + tower against PH + tower with the round-4 irregular path, which are entirely different code paths for every class: found.cert (3,401 factors) and survivors byte-identical.
* **nvcc 13.3, sm_75:** both builds warning-free (`-Xcompiler -march=native`), no spills, registers as in section 3.
* All of the above were rerun on the final sources, the files you have.
* **What the emulator cannot tell me:**
  - real timings (every number in sections 6 and 8 is an estimate);
  - the shared-memory staging of the tower;
  - your card's read rate on a 256 MB table.

## 8. Hardware test plan

    make coarse_sieve coarse_sieve_wide gf2_dlog_selftest
    ./gf2_dlog_selftest                     # ~5 min: part 8 is new; "ALL OK"
    ./coarse_sieve --selftest
    ./coarse_sieve_wide --selftest
    ./coarse_sieve_wide --selftest-lowmem 7
    ./coarse_sieve_wide --selftest-logs     # all engines; adds RR at odd d = 21..47
                                            # and the irregular-lift check
    COARSE_SELFTEST_RR_DMAX=49 ./coarse_sieve_wide --selftest-logs rr
                                            # RR only, through d=49: builds d=49's
                                            # table and checks the engine on the GPU
                                            # (a few minutes; do it before step 4)
    ./coarse_sieve_wide --selftest-logs tower
    ./coarse_sieve_wide --selftest-logs ph
    COARSE_LOGS_KERNEL=r2 ./coarse_sieve_wide --selftest-logs

1. **Regression against your round-4 files**, as last time:

       for d in 36 40 42 44; do
         ./coarse_sieve_wide 136279841 $d r5_$d \
             --load-survivors r4_$((d-1)).survivors.txt --min-depth $d
         cmp r5_$d.found.cert r4_$d.found.cert && \
         cmp r5_$d.survivors.txt r4_$d.survivors.txt && echo "d=$d identical"
       done

   Please send the `done` lines and the `[tune ...]` lines of d = 40, 42, 44. In the d=44 tune line, the ratio of the `+r4-irregular` number to the default one is the round-5 gain on your card.
2. **Odd degrees, if you have time.** d=41 or 43 against your round-3 files. The tuner may pick RR there, and identical files are then the check of RR on hardware.
3. **Checkpoint.** Start d=44, press Ctrl-C once partway through ("stopping on request"), then run the same command. It should report "found checkpoint", and `cmp` should show the files identical to step 1's.
4. **d = 49:**

       ./coarse_sieve_wide 136279841 49 r5_49 --load-survivors S48.survivors.txt --min-depth 49

   Here `S48.survivors.txt` stands for your d=48 survivors file (from round 3). The table builds first, in minutes, then the sieve, with an ETA after ~1/64 of the pass. Then d = 50.. can follow with `--seed-covers-49`.
5. **The strongest independent check of RR, optional.** Run r = 74207281 at d=47 with `COARSE_METHOD=rr` against Brent et al.'s certificates, as you did for d = 45..48 in round 3. If you have the GPU time, d=49 for that r against the certificates' degree-49 lines is the final word.

## 9. Should this be the last round? My view: yes, once you've run this one

* **Coverage is complete.** Every degree through 55 can be sieved. Degree 49, the one hole, is closed, so nothing that matters is blocked any more.
* **Diminishing returns are built in.**
  - The cost doubles per degree, so even a perfect 2x speedup buys exactly one more degree.
  - Each degree resolves only ~1/d of the remaining survivors, ~2%, about 50K of the ~2.4M survivors at d ≈ 49.
  - So the useful depth is now set by economics, not by code. Each further sieve degree saves tsfactor one degree step on every survivor and removes ~2% of them outright. Compare that with the GPU-hours per degree (table below).
  - What's left in the code is ≤ 1.5x per item: Karatsuba bit-slicing for PH, local reads of T, a division in K for d=34/38/46.
* **Risk grows with every round.** 67 regression checks and a dozen knobs are a lot of surface for small gains.

A round 6 would make sense if:
1. your round-5 hardware numbers miss these estimates badly, or anything fails (I'd want to know why);
2. a new Mersenne prime appears (the program needs no change for a new r; it checks gcd(r, 2^d − 1) = 1 at every degree);
3. a cleanup round, once round 5 is trusted: dropping the round-2 kernel, the round-4 irregular path and other A/B scaffolding (~1,000 lines).

Projections, RTX 2070 SUPER, round 5. All are estimates from the read rates and SASS counts above, except the "last measured" column.

| d | engine | classes | last measured | round 5 (est.) |
|---|---|---|---|---|
| 44 | tower/GF(2^11) | 4.0e11 | 669 s (r4) | ~4-7 min |
| 45 | PH or RR | 7.8e11 | 1,702 s (r3) | ~15-30 min |
| 46 | tower classic | 1.5e12 | 2,921 s (r3) | unchanged: already read-bound (3 reads/class) |
| 47 | PH or RR | 3.0e12 | 5,411 s (r3) | ~1.1-1.5 h |
| 48 | tower/GF(2^12) | 5.9e12 | 12,711 s (r3) | ~1.2-2 h |
| 49 | RR | 1.15e13 | (impossible) | ~4-6 h + 1-2 min table |
| 50 | tower/GF(2^5) | 2.3e13 | -- | ~5-9 h |
| 51 | RR or PH | 4.4e13 | -- | ~16-24 h |
| 52 | tower/GF(2^13) | 8.7e13 | -- | ~17-29 h |
| 53 | RR | 1.7e14 | -- | ~3-4 days |
| 54 | tower/GF(2^9) | 3.3e14 | -- | ~3-5 days |
| 55 | RR (2 GB table) | 6.6e14 | -- | ~11-15 days |

## 10. Your PS questions

**Would index calculus for d=49 beat 2-9 GPU-days on newer hardware?** It turned out better than that on your current card.
- RR is that idea taken to its extreme. At d=49 the factor base can be *everything* of degree ≤ 24. There is no relation collection and no linear algebra, and the 1.47M factor-base logs come from plain PH/BSGS in a minute or so. What remains is ~4-6 h on your 2070 SUPER, not GPU-days.
- On newer cards the sieve is bound by random reads in a 256 MB table, so it scales with random-access DRAM throughput: roughly 2-3x faster on an RTX 4090 (~1.5-2.5 h), and more on an RTX 5090 or H100.
- The table is larger than any NVIDIA L2 today (4090: 72 MB, H100: 50 MB). A card whose last-level cache held 256 MB would turn d=49 into an instruction-bound job.

**Would these ideas interest other projects?** In decreasing order of fit:
1. **Anyone hunting primitive trinomials of large degree.** For a new Mersenne exponent, this sieve to d ≈ 49 settles ~96% of the s values in GPU-hours and hands the rest to a factoring program with a guaranteed lower bound on their smallest factor. The same goes for projects tabulating factorization statistics of trinomials. Your survivor counts follow the c/d law from round 4.
2. **Anything that needs ~10^11-10^14 discrete or Zech logs in GF(2^d) with d up to ~56.** The three engines are textbook ideas made GPU-friendly: the tower with subfield normalization, bit-sliced PH, and RR with a complete factor base. One example is finding low-weight multiples x^a + x^b + 1 of an LFSR feedback polynomial for correlation attacks on stream ciphers. "P divides x^a + x^b + 1" is exactly the Zech-log progression condition used here.
3. **The integer analogues** (k·2^n ± 1 sieves such as srsieve and PrimeGrid's) already do one discrete log per prime by BSGS. The big lever here, one field GF(2^d) holding all ~2^d/d irreducibles of degree d so that one table serves them all, has no integer counterpart: each prime is its own field.

**Why did Brent et al. rely on `factor`?** Not for lack of awareness, as far as I can tell from their papers.
- They use exactly this progression structure for small factors. In the multi-level-blocking DDF paper, a factor of degree d < ½·log2 r is found by reducing the exponents mod 2^d − 1 and taking gcd(x^(r mod D) + x^(s mod D) + 1, x^D − 1) (§2.5). They note that the cost of small factors is "negligible".
- In the 2003 searches, a per-trinomial "sieve" by GCDs up to degree 24-26 eliminated ~93% of the trinomials in 5-8% of the total time.
- Their time went into the survivors: the blocked DDF and the irreducibility tests. A deep GPU sieve would therefore have saved them only a few percent. 2016 GPUs could have run it; it just wasn't where their time went.
- For your pipeline the trade-off is different: the sieve's value is in giving tsfactor only the hard ~4%, already known to have no factor of degree ≤ 49.

**Two by-products that apply beyond this program:**
- For d up to ~56, the half-gcd split gives a discrete log in GF(2^d) from two reads of a table of 2^(d/2+1) entries (at most a few GB), with no failure case. That makes it a general tool for bulk Zech-log work at those sizes.
- The division-free treatment of non-unit classes (section 2) works for any batched-inversion kernel over Z/(2^d − 1) that meets non-units.

## Appendix: earlier rounds, for reference

**Round 4 (2026-10-22).** Subfield tower: a tower log needs only one DRAM read, of T; the inner logs come from shared-memory tables over a subfield of K. Your results: selftests PASS; d=36 4.40 s, 40 45.42 s, 42 150.14 s, 44 668.65 s; A/B at d=44 774.48 s → 666.77 s.

**Round 3 (2026-10-20).** Batched kernel: FKM enumeration of exactly the class representatives, 32 classes per thread in bit-sliced form, Montgomery batch inversion, a multi-part PH planner, and depth limit 55. Your results: d=45 1,702 s, 46 2,921 s, 47 5,411 s, 48 12,711 s, consistent with the oracle and with the ranges already done.

**Round 2 (2026-10-19).** The logs path: one Zech log per class from tables of ~2^(d/2) entries. Your times: d=41 399 s, 42 222 s, 43 1,646 s, 44 945 s, 45 6,380 s, 46 4,022 s, 47 21,242 s.

**Round 1 (2026-10-18).** LFSR stepping in the bucketed path (~1000x), smallest-fitting bucket counts, other fixes.

Sources:
- [Brent, Larvala, Zimmermann, *A fast algorithm for testing reducibility of trinomials mod 2* (Math. Comp. 2003)](https://maths-people.anu.edu.au/~brent/pd/rpb199.pdf)
- [Brent, Larvala, Zimmermann, *A primitive trinomial of degree 6972593* (2003)](https://maths-people.anu.edu.au/~brent/pd/rpb214tr.pdf)
- [Brent, Zimmermann, *A multi-level blocking distinct degree factorization algorithm* (2007)](https://arxiv.org/abs/0710.4410)
- [Brent, Zimmermann, *Twelve new primitive binary trinomials* (2016)](https://arxiv.org/abs/1605.09213)
- [Jia, Maggioni, Smith, Scarpazza, *Dissecting the NVidia Turing T4 GPU via Microbenchmarking* (2019)](https://arxiv.org/abs/1903.07486)

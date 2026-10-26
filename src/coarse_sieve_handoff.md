# coarse_sieve, round 6: one log for a polynomial and its reciprocal, an honest ETA, quick tuning, and a cleanup

**Baseline: round 5 (2026-10-24), which you validated on hardware.**
- All selftests PASS, including `COARSE_SELFTEST_RR_DMAX=49 --selftest-logs rr`.
- Regressions at d = 36, 40-44 identical; checkpoint interrupt/resume at d=44 identical.
- r = 136279841: d=49 14,185 s and d=50 34,627 s, consistent with the ranges already done with `factor`.
- r = 74207281: d=47 (RR) and d=49 match Brent et al.'s certificates.
- r = 57885161: d = 2-46 match the certificates.

## Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | Round 6. One build now: `coarse_sieve` (the former `coarse_sieve_wide`). |
| `gf2_dlog.h` | Round 6: the reciprocal-pair rule (`gf2f_pair_role`), the class count I(d), and the round-2 planner removed. |
| `gf2_dlog_selftest.cpp` | New part 9 (reciprocal pairs); part 6 also checks the partner's factor by polynomial division. |
| `Makefile` | One `coarse_sieve` target (C++17, the u64 field headers); `emul-coarse_sieve`. |
| `gf2_wide_field.h`, `gf2_wide_setup.h` | Comments only (they no longer mention `-DWIDE_FIELD`). |
| `coarse_sieve_round6.patch` | All of the above as a patch against round 5 (flat paths). |

`cuda_emul.h` and the other headers are unchanged. `gf2_small_field.h` and `gf2_field_setup.h` are no longer used by coarse_sieve, but sieve_ref and trinomial_stage2 still use them, so keep them.

## TL;DR

* **Your three observations, diagnosed and fixed.**
  - **The "[~x h to go]" estimate.** Round 5 multiplied the time of the first 1/64 of the pass by 63. But the classes are packed at small u: in your d = 42-50 runs that first 1/64 took 13-17% of the pass, so the estimate came out 9-13x the real remainder. Now the pass takes its chunks in a scrambled, stratified order, so every stage of it is a fair sample of the whole range. The 64 dots come at a steady pace, and the estimate is this run's time per class times the classes left. In the emulator it lands within 0.9-1.06x of the actual remainder at d = 30-32, with one outlier (1.27x) where the first dot took 0.09 s. On your card at d ≥ 46, where the first dot takes minutes, I expect it within a few percent (estimate). Section 1.
  - **The 66-minute tuning at d=50.** This was a bug in the tuner's sample, not a symptom of anything else; results were never affected. The sample started at a point with no classes at all and doubled until it ran into the dense part, so it grew to 4.7e11 classes at d=50, where ~3e7 were meant. It also timed the classic tower twice. Tuning should now take seconds (estimate; section 2).
  - **Re-tuning on resume.** The checkpoint (format v4) now records the degree's engine, inner subfield and loop order. A resumed degree uses them and prints no tune line (section 3).
* **New: one Zech log serves a polynomial and its reciprocal.**
  - The class of −u holds the roots of the reciprocal polynomial P*, and Z(−u) = Z(u) − u. Its progression is s ≡ r − s0 (mod n), with the same n.
  - So each log now marks two progressions, and only one class per reciprocal pair is computed: half the logs, table reads, inversions and bit-sliced work.
  - **Estimate: ~1.6-1.9x faster at every degree**, e.g. d=50 ~4.5-5.3 h instead of 9.6 h. Section 4.
* **Also new, unmeasured: the tables' DRAM reads are issued one class ahead** (section 6). Both loop orders are kept and the tuner times both at every degree ≥ 40, so the slower one is not used there; the tune lines will show the difference on your card.
* **Cleanup, as you suggested** (section 5):
  - one binary, `coarse_sieve`;
  - the degree range is now the second argument: `33-46` or `47`;
  - removed: `--smax`, `--found`, `--load`, `--min-depth`, `--allow-degree-gap`, `--seed-covers-49`, `--resume`, `--checkpoint-keep`;
  - removed code: the u32 build, the dense and bucketed paths, the round-2 kernel, the round-4 irregular path;
  - `--selftest [tower|ph|rr]` replaces the three selftest options.
  - coarse_sieve.cu shrinks from 4,289 to 3,303 lines. The outputs (`.found.cert`, `.survivors.txt`) are unchanged.

## 1. The time-left estimate

**What went wrong.**
- Each class is handled at its least rotation u, a Lyndon word: odd, below 2^(d−1), and starting with its longest run of zeros.
- So classes pile up at small u. The first 1/64 of the pass covers u < 2^(d−7), the words whose longest zero run is ≥ 7, and that is about 1 − e^(−d/256) ≈ 15-18% of the classes at d = 42-50.
- Round 5 took the chunks in order and scaled the first dot's time by 63. From your logs, the first dot took 13-17% of each pass, so the estimate was 9-13x the real remainder (7-10x of the whole run, as you measured it).

**The fix: a scrambled, stratified order.**
- The chunks are grouped in blocks of 8. The NG chunks of one round stay within a block, so several GPUs still get neighbouring chunks of similar density and finish together.
- The blocks run in bit-reversed order with Owen's nested scrambling: the first 2^j blocks of the pass hold exactly one block from each 1/2^j of the range, at a pseudo-random place inside it.
- Plain bit reversal was my first version, and it failed in the emulator. Its first 1/64 is every 64th block, and in those blocks u has six zero bits just above the block's own bits: an internal zero run longer than most leading runs, so the blocks are nearly empty. The estimate came out 5-15x too *low*. The scrambling fixes exactly that.
- The estimate is this run's time per class times the classes left (I(d) is known exactly). It is shown once, at the first dot after 256 chunks (1/64 of the pass at d ≥ 46), when more than 10 minutes remain, and in every checkpoint line, e.g. `[checkpoint: d=51 chunk 81920/524288, delta 20112 entries (0.31 MB) -> r6_51.ckpt, 0.02s, ~7.2 h to go]`. Below 1.5 h it is given in minutes.
- Results don't depend on the order: best[] is merged by atomicMin. The selftest checks that the order is a permutation and that it is stratified.

**Measured in the emulator** (d = 30-32 at s ≤ 299,999, 16-65K chunks; `dottime.py` timestamps every dot):

| | d=30 | d=31 (RR) | d=32 |
|---|---|---|---|
| round 5: in order, chunk-scaled | 8.9x | 8.5x (PH) | 8.0x |
| bit-reversed (first version) | 0.07x | 0.14x | 0.20x |
| scrambled + per class (round 6) | 0.90x, 1.27x | 0.95x, 0.97x | 0.97x, 1.06x |

(Estimate after the first dot ÷ actual remaining time; two chunk sizes each.) The dots now come at a steady pace: the middle 80% of the gaps between dots lie within −20%/+30% of the median, where in round 5's order the 90th percentile was 70-100 times the median.

## 2. The 66-minute tuning at d=50

**It was a bug in the tuner's sample, and nothing else.**
- Round 5's sample started at u = 2^(d−4), 1/8 of the range, with half a chunk, and doubled until it held 2^25 classes.
- But every u in [2^(d−4), 2^(d−4) + 2^(d−8)) is 0001 0000…: an internal zero run of four or more after a leading run of three. None of them is a Lyndon word, so that stretch holds no classes at all.
- So the doubling ran until the sample was ~2^(d−8) wide and had reached the dense part: 9.2e9 classes at d=44 and 4.7e11 at d=50.
- Each candidate then ran that sample, the first one twice (warm-up): 4.69e11 classes × (2 × 1.438 + 1.794 + 1.864 + 1.871) ns = 3,942 s, against the 3,945 s in your tune line.
- Results were never affected. All candidates compute identical marks, and the real pass repeats them.
- A second, smaller bug: when m = d/2 has only one proper divisor (d = 18, 50), the classic tower was listed twice. That is the "classic 1.864, classic 1.871" in your d=50 line.

**Now:**
- The sample is a fixed stretch of 2^27 u right after a long leading zero run, where about half the u are classes: ~2^26 classes, almost all canonical (section 4).
- Tuning should take a few seconds at every degree (estimate), plus the table builds of the candidates.
- The candidate list has no duplicates; the regression checks the lists at d = 18, 20, 24.

## 3. Checkpoints, format v4

What a checkpoint now records:
- **The run:** r, smax, the degree range D1-D2, and the seed file's path and a hash of its contents. A different command line refuses to start, as before.
- **The position:** the degree, the chunks done (in the pass order of section 1), and the classes they covered, so the I(d) count at the end of the degree still holds across a resume.
- **The degree's configuration:** the engine (tower, PH or RR), the tower's inner subfield, and the loop order (section 6).

A resumed degree uses the recorded configuration and prints no tune line. The degrees after it are tuned as usual. Stopping on a degree boundary records only the next degree, which is then tuned when it starts.

Older checkpoints (rounds 4-5) are refused with `checkpoint X exists but no generation of it is usable (an older format, or damaged); --fresh sets it aside and starts over`.

On resume, a different chunk count (`COARSE_LOGS_CHUNK_LOG2`) is refused. The number of GPUs may differ between the stop and the resume; that is tested both ways.

## 4. Reciprocal pairs: one Zech log for P and P*

**The identity.**
- A class u (the class of β^r, β a root of P, u = j·r) gives s ≡ s0 = Z(u)/j (mod n), with Z(u) = log_g(1 + g^u).
- The class of −u, the d-bit complement of u, is the class of β^(−r): the roots of the reciprocal polynomial P*.
- 1 + g^(−u) = g^(−u)·(1 + g^u), so **Z(−u) = Z(u) − u**.
- P*'s condition is (−j)·s ≡ Z(u) − u, so s ≡ (u − Z(u))/j = **r − s0 (mod n)**, since u/j = r. gcd(−j, M) = gcd(j, M), so n is the same, and the irregular (gcd > 1) classes work the same way.
- In plain terms: P | x^r + x^s + 1 if and only if P* | x^r + x^(r−s) + 1, the reciprocal trinomial. The sieve uses this for both progressions in the range, not for the mirror half.

**What the kernel does.**
- Of each pair, only the member with fewer ones (popcount < d/2) is computed. Ties at even d are broken exactly: first the leading zero run against the longest run of ones, then u against the least rotation of ~u (`gf2f_pair_role`).
- That member marks its own progression and its partner's (j' = M − j, s' = r − s0).
- Self-reciprocal classes (P = P*, only at even d) are their own partners and are handled alone. There are SR(2m) = (1/2m)·Σ_{e | m, e odd} μ(e)·2^(m/e) of them, about 2^(d/2)/d.
- The partner skip costs a popcount per class. Every class is still enumerated (FKM), but the logs, table reads, bit-sliced powers and inversions are halved.
- The class count credited at the end of each degree includes the partners and must equal I(d) exactly, as before.

**What it should buy.** The emulator runs the same code on the CPU, one thread. There, round 6 takes 0.57-0.62 of round 5's time at d = 30 and 32 (section 7). On the GPU the halved part, the logs with their DRAM reads, is a larger share of the time, so I expect **~1.6-1.9x per degree** (estimate).

**Checked independently of the kernel:**
- host selftest part 9: every class up to d=22, partners complementary, the counts equal I(d) and SR(d);
- host part 6: for sampled classes at every degree 3-55 (49 excepted), the minimal polynomial of g^(−j) is f reversed and divides x^r + x^(r−s0) + 1 (arithmetic mod f*), and a direct log gives Z(−u) = Z(u) − u;
- device selftest: 2^16 random classes per engine and degree 21-55, Z(−u) computed directly equals Z(u) − u, and the partner's s0 computed directly equals r − s0 mod n;
- the sieve outputs, byte-identical to round 5's dense and bucketed paths, which know nothing of pairs.

**An aside: the other symmetries.**
- Z also satisfies Z(Z(u)) = u, so one log fixes the Zech logs of six classes: u, Z(u), −u, Z(u) − u, −Z(u), u − Z(u). This is the S3 orbit of x ↦ 1 + x, 1/x.
- But only the reciprocal pair can be skipped without knowing Z(u) first.
- The other four are reachable only from field elements: enumerate x in a normal basis, where Frobenius is a rotation and x ↦ 1 + x is a complement, and test the orbit by inversion. I estimated that at no better than ~15% over round 6, for a much bigger change, so I left it.

## 5. Cleanup and the new command line

    coarse_sieve <r> <degrees> <out_prefix> [options]
    coarse_sieve --selftest [tower|ph|rr]

`<degrees>` is `D1-D2` or `D`. A run from scratch starts at 2 and takes no seed. A run starting at D1 > 2 needs `--load-survivors` with the survivors of a run through D1−1, and the program refuses to start without one.

| round 5 | round 6 |
|---|---|
| `coarse_sieve 57885161 32 out2-32` (u32 build) | `coarse_sieve 57885161 2-32 out2-32` |
| `coarse_sieve_wide 57885161 46 out33-46 --load-survivors out2-32.survivors.txt --min-depth 33` | `coarse_sieve 57885161 33-46 out33-46 --load-survivors out2-32.survivors.txt` |
| `coarse_sieve_wide 57885161 47 out47 --load-survivors out46.survivors.txt --min-depth 47` | `coarse_sieve 57885161 47 out47 --load-survivors out46.survivors.txt` |
| `--smax S` | gone; the test hook `COARSE_TEST_SMAX=S` remains for short test runs |
| `--found`, `--load f.found.bin` | gone (.found.bin files) |
| `--allow-degree-gap`, `--seed-covers-49` | gone: 49 is an ordinary degree, and a seed always covers every degree below D1 |
| `--resume f`, `--checkpoint-keep G` | gone: the same command resumes by itself, `--checkpoint f` names the file, and 3 generations are kept |
| `--selftest`, `--selftest-logs [e]`, `--selftest-lowmem N` | `--selftest [tower\|ph\|rr]`: the sieve against the CPU reference to d=20, then the logs-path checks to d=55 |
| `COARSE_METHOD=dense\|buckets\|logs`, `COARSE_LOGS_KERNEL`, `COARSE_IRREG`, `COARSE_FORCE_BUCKETS` | gone; `COARSE_METHOD=auto\|tower\|ph\|rr` stays |
| `COARSE_TEST_STOP_AFTER_BUCKETS` | `COARSE_TEST_STOP_AFTER_CHUNKS` |

Unchanged: `--load-survivors`, `--no-cert`, `--no-survivors`, `--checkpoint`, `--checkpoint-mins`, `--no-checkpoint`, `--fresh`, `--gpus`, `--no-tune`, and the output formats. The done line now also gives the degree's class count (I(d)).

Exit status: 0 done; 1 error or refusal; 2 stopped early on a degree that cannot be sieved on this card (e.g. RR at d=49 needs ~1.5 GB free). In the last case the outputs cover the degrees finished, and the checkpoint is kept.

**Removed code** (the file goes from 4,289 to 3,303 lines, new code included):
- the u32 build (`coarse_sieve` without `-DWIDE_FIELD`, on gf2_small_field.h);
- the dense-table and bucketed paths, including low-memory mode. The logs path covers every degree; d = 2-32 take seconds on a GPU (estimate), and small degrees with many marks go through the heavy-progression queue as before;
- the round-2 kernel and the round-4 irregular path, kept in round 5 as A/B references;
- .found.bin output and the old checkpoint formats.

## 6. Also new, and unmeasured: the table reads one class ahead

**The change.**
- Each class's log needs one or two random DRAM reads: the tower's T[l], or RR's L[r] and L[t].
- Round 5 issued them and used them 20-90 instructions later. A warp then waits out most of the DRAM latency (~500+ cycles), and only the other 15 warps of the SM can cover for it.
- Round 6 splits a class's forward step into stage A, the reads (the tower's coordinates and inner logs and then T[l]; RR's half-gcd and then its two reads), and stage B, everything else.
- Stage A runs one class ahead, with two alternating sets of registers. Class k's reads are issued before class k−1's stage B and first used in class k's own stage B, after class k+1's stage A. That gives them a few hundred instructions to arrive.
- The two sets matter. My first version was a plain rolled loop. Its SASS showed the compiler copying the loaded value at the top of the next iteration, and a copy of a pending load waits just like a use, so it lost half the window.
- In the final SASS (d=44 tower, d=45 RR), a read's first use comes ~460-1,000 instructions after it along the code (static count, cold branches included). In the round-5 order it comes after 22 instructions (tower) or 88 (RR).

**Why I think it can matter.** Your round-5 tune lines show the classic tower, which adds a level of dependent DRAM reads per class, running 0.5-0.8 ns/class slower than the subfield tower at d = 40-44 (1.313 vs 0.824, 1.524 vs 0.899, 1.695 vs 0.897). That is memory latency, not instructions. But this is a guess about your card that I can't time here.

**So it is a tuner candidate, not a commitment.**
- Both loop orders are in every tower and RR kernel. At d ≥ 40 the tuner times both, and the old one appears as "r5-order", for example:

      [tune d=44 on GPU 0, N classes, T s: GF(2^11) a ns/class, GF(2^11) r5-order a' ns/class,
       GF(2^2) b ns/class, GF(2^2) r5-order b' ns/class, classic c ns/class, classic r5-order c' ns/class -> ...]

- a'/a is the gain on your card. If it is below 1, the tuner picks the round-5 order and nothing is lost.
- The choice goes into the checkpoint. `COARSE_PIPE=0|1` fixes the order instead.
- ptxas (CUDA 13.3, sm_75): the tower kernels use 128 registers (round 5: 124-128) and the RR kernels 125-128 (round 5: 115-121), so occupancy is unchanged at 512 threads per SM. At d ≥ 34 the tower kernels spill one 8-byte value, outside the per-class loop: once per batch of 32 classes.

## 7. Validation done here

* **Host, `gf2_dlog_selftest`: ALL OK.** New part 9 checks every class up to d=22, and the I(d) identity up to 55. Part 6 adds the partner checks of section 4 at every d = 3-55 (49 excepted).
* **Unit tests:** arithmetic, rotation, bit-sliced arithmetic, tower, and `t_pair` (the pairing partition, every class of d = 2-24).
* **Emulated program, `regress6.sh`: 77 checks, all PASS.** Wherever outputs were compared they are byte-identical.
  - **Oracles:** round 5's dense-table and bucketed paths. They share no kernel code with round 6 and know nothing of pairs.
    - from scratch to 22, and seeded 19-23, at s ≤ 10^6;
    - 24-29 and 30-33 at s ≤ 3·10^5, each also with RR at the odd degrees;
    - **the full range, s ≤ 68,139,920:** from scratch to 28, then 29-32.
  - **Both loop orders** of section 6, for tower + PH and tower + RR (19-23, 24-29).
  - 2 emulated GPUs; fault injection; every inner subfield at d=24.
  - **Tuner:** every pick at d = 18-26, including the loop-order twins, gives identical output; the candidate lists have no duplicates; `COARSE_PIPE` fixes the order.
  - **Command line:** the refusals, malformed degree ranges, removed options, and the early stop (exit status 2) with its continuation.
  - **Checkpoints (K1-K14):**
    - stop and resume mid-degree and on degree boundaries;
    - refusal of four kinds of mismatch; `--fresh`; `--no-checkpoint`; a round-5 checkpoint refused; the lock;
    - a real SIGINT with two continuations;
    - the tuned engine, subfield and loop order kept on resume without re-tuning;
    - damaged newest generations skipped;
    - a finished checkpoint (outputs only);
    - 1 → 2 and 2 → 1 GPUs;
    - a from-scratch, full-image checkpoint against the dense oracle.
* **Pass order and estimate:** section 1's table (the bit-reversed first attempt failed exactly there).
* **Per-class time, emulator:** round 6 takes 0.57-0.62 of round 5's time at the tower degrees d = 30, 32 (same seed and range, identical output).
* **nvcc 13.3, sm_75:** warning-free, 90 kernels, 3 min 11 s. Registers: tower 128 and RR 125-128, the limit for 512-thread blocks; PH 117-248, as before. The only spill is the one value of section 6, outside the loop.
* All of the above were rerun on the final sources, the files you have.
* **What the emulator cannot tell me:** GPU timings (every time in sections 4, 6 and 9 is an estimate), and the gain of section 6.

## 8. Hardware test plan

    make coarse_sieve gf2_dlog_selftest
    ./gf2_dlog_selftest                    # ~7 min; part 9 is new; "ALL OK"
    ./coarse_sieve --selftest              # all engines, both loop orders, pairs
    ./coarse_sieve --selftest tower
    ./coarse_sieve --selftest ph
    ./coarse_sieve --selftest rr
    COARSE_SELFTEST_RR_DMAX=49 ./coarse_sieve --selftest rr   # builds d=49's RR table

1. **Regression against your round-5 files**, same seeds as last time (the old prefixes r4_/r5_ are only examples):

       for d in 40 41 42 43 44; do
         ./coarse_sieve 136279841 $d r6_$d --load-survivors r4_$((d-1)).survivors.txt
         cmp r6_$d.found.cert r5_$d.found.cert && \
         cmp r6_$d.survivors.txt r5_$d.survivors.txt && echo "d=$d identical"
       done

   Please send the `done` lines and the `[tune ...]` lines. Three numbers to look at:
   - the d=44 done line against round 5's 401 s;
   - the tune line's time, now seconds;
   - each "r5-order" ns/class against its partner's, which is the gain of section 6 on your card.
2. **The estimate.** A pass needs more than 10 minutes to show it, so use d = 46 or 47 (or prefix any run with `COARSE_TEST_ETA=1`, which always prints `[eta N s]`). Compare the estimate, plus the time of the first 4-8 dots, with the done line.
3. **Checkpoint with tuning.** Start d=44, wait for the tune line and a few dots, press Ctrl-C once, then run the same command again. Expect `found checkpoint ... degree 44, chunk N of 4096`, `d=44: resuming at chunk N/4096`, no tune line, and files identical to step 1's.
4. **Certificates, as you did in round 5.** r = 57885161 at 2-32 and 33-46, and r = 74207281 at 47 and 49, against Brent et al.'s data. In the new syntax:

       ./coarse_sieve 57885161 2-32 out2-32
       ./coarse_sieve 57885161 33-46 out33-46 --load-survivors out2-32.survivors.txt
       ./coarse_sieve 74207281 47 out47 --load-survivors <your d=46 survivors>
       ./coarse_sieve 74207281 49 out49 --load-survivors <your d=48 survivors>

5. **Then production:** `./coarse_sieve 136279841 51 r6_51 --load-survivors <your d=50 survivors>`, ~8-10 h (estimate).
6. **Optional, for a next round: one profile.** If Nsight Compute (`ncu`, part of the CUDA toolkit) is installed:

       sudo ncu --set full -k regex:k_logs_batch --launch-skip 20 --launch-count 1 -o d44 \
            ./coarse_sieve 136279841 44 prof --load-survivors r4_43.survivors.txt --no-checkpoint --no-tune
       ncu --import d44.ncu-rep --page details > d44.txt

   Then send d44.txt, and the same for d=47 if you like. Section 9 explains why.

## 9. Projections, and where a next 2x might come from

RTX 2070 SUPER. The round-6 column is the round-5 rate divided by 1.6-1.9 for the pairs, and does not count section 6. All of it is estimate except the "round 5" column.

| d | engine | classes | round 5 (your runs) | round 6 (est.) |
|---|---|---|---|---|
| 44 | tower/GF(2^11) | 4.0e11 | 401 s (60 s of it tuning) | ~3-4 min |
| 47 | RR | 3.0e12 | 3,648 s | ~32-38 min |
| 49 | RR | 1.15e13 | 14,185 s | ~2.1-2.5 h |
| 50 | tower/GF(2^5) | 2.25e13 | 34,627 s (3,945 s tuning) | ~4.5-5.3 h |
| 51 | RR (or PH) | 4.4e13 | — | ~8-10 h |
| 52 | tower/GF(2^13) | 8.7e13 | — | ~11-15 h |
| 53 | RR | 1.7e14 | — | ~32-38 h |
| 54 | tower/GF(2^9) | 3.3e14 | — | ~2-2.7 days |
| 55 | RR (2 GB table) | 6.6e14 | — | ~5.4-6.4 days |

**Where the time goes, as far as I can tell from here.**
- In round 5's SASS, a warp spent ~1,200-1,500 instructions on each step of 32 classes in the hot loop. With the bit-sliced powers and the enumeration that is roughly 50-60 issue slots per class, ~0.2 ns per class at your card's issue rate (40 SMs × 4 schedulers × ~1.8 GHz). Your tune lines show 0.82-0.90 ns per class at the best tower degrees, so the kernel runs at ~25% of its instruction bound: stalls dominate, not arithmetic.
- Three candidates for the stalls:
  - the DRAM read(s) of each log, which section 6 attacks;
  - the batch arrays: four arrays of 32 u64 per thread, 1.5-1.7 KB of stack per thread with the rest. They are indexed arrays, so they live in local memory whatever the register count. 512 threads × 1.6 KB is 0.8 MB per SM against an L1 of 32-64 KB, so those accesses go to L2 and DRAM;
  - the shared-memory tables are read through generic pointers (`LD.E` in the SASS, not `LDS`).
- A profile (step 6 of the test plan) would say which one it is in one run, and I'd rather aim the next change at that than guess.
- If it is the batch arrays, the fix is structural: one kernel for enumeration and the bit-sliced powers, writing x and u to a buffer, and a second, light one-class-per-thread kernel for the logs. That one would have few registers and no arrays, so 2-4x the warps per SM to hide latency. It would be a sizeable change.
- Smaller, known items: RR at the even degrees whose m is prime (d = 34, 38, 46, where only the classic tower exists: your d=46 is 1.78 ns/class vs RR's ~1.24 at d=45, but those degrees are behind you); and PH's per-part lookups, which could be pipelined like section 6 (PH loses to RR at every degree ≥ 41 on your card anyway).

**On the economics** (as in round 5): each degree removes ~1/d of the survivors, ~2%, and costs twice the previous one. Going from 50 to 55 removes ~9% of the remaining survivors (1 − 50/55) for ~9-12 GPU-days (estimate). Whether that pays depends on tsfactor's cost per survivor, which you know better than I do.

## Appendix: earlier rounds, for reference

**Round 5 (2026-10-24).** The irregular classes joined the batch inversion (1.97x at d=44: 669 s → 401 s, including 60 s of tuning). Also RR and degree 49, auto-checkpointing, and tuning. Your results: all selftests PASS; regressions at d = 36, 40-44 identical; d=47 (RR) 3,648 s, d=49 14,185 s, d=50 34,627 s; the certificates of r = 74207281 (47, 49) and 57885161 (2-46) matched.

**Round 4 (2026-10-22).** Subfield tower: a tower log needs only one DRAM read, of T, and the inner logs come from shared-memory tables over a subfield of K. Your results: d=36 4.40 s, 40 45.42 s, 42 150.14 s, 44 668.65 s.

**Round 3 (2026-10-20).** Batched kernel: FKM enumeration of exactly the class representatives, 32 classes per thread in bit-sliced form, Montgomery batch inversion, a multi-part PH planner, and depth limit 55. Your results: d=45 1,702 s, 46 2,921 s, 47 5,411 s, 48 12,711 s.

**Round 2 (2026-10-19).** The logs path: one Zech log per class from tables of ~2^(d/2) entries. Your times: d=41 399 s, 42 222 s, 43 1,646 s, 44 945 s, 45 6,380 s, 46 4,022 s, 47 21,242 s.

**Round 1 (2026-10-18).** LFSR stepping in the bucketed path (~1000x), smallest-fitting bucket counts, other fixes.

Sources:
- [Brent, Larvala, Zimmermann, *A fast algorithm for testing reducibility of trinomials mod 2* (Math. Comp. 2003)](https://maths-people.anu.edu.au/~brent/pd/rpb199.pdf)
- [Brent, Zimmermann, *Twelve new primitive binary trinomials* (2016)](https://arxiv.org/abs/1605.09213)
- [*ART-Owen Scrambling* (2023), for Owen's nested scrambling of digital sequences, used in section 1](https://arxiv.org/pdf/2311.11664)

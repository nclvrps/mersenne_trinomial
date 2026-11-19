# coarse_sieve, round 10 (final): the results, a last small version, and a handoff for the next Mersenne prime

**Baseline: round 9 (2026-10-29), which you validated on hardware.**
- Every selftest passed, and every output of steps 1-3 matched your earlier files.
- d=53 was finished with round 9, d=54 took 30.4 h, and d=55 is running.

This handoff is written to be read years from now as well as today:
- §1-§3 are this round;
- §4-§10 are the reference: the statistics, how the program works, how to run it for a new exponent, what lies beyond d = 55, and the history.

## Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | Round 10 (§3). |
| `cuda_emul.h` | `cudaGetDeviceProperties` for the emulated build. |
| `gf2_dlog.h` | A comment only: the prefetch's default. |
| `coarse_sieve_round10.patch` | The three files above as a patch against round 9 (flat paths). |
| `coarse_sieve_devkit_final.tar.gz` | The final dev kit (§10). It holds the sources of rounds 5-10, the test suites, the oracle generator, the analysis tools of every round, and all ten handoffs. |
| `coarse_sieve_handoff.md` | This file, also the project doc `claude/coarse_sieve_handoff.md`. |

`gf2_dlog_selftest.cpp`, the field headers, the unit tests and the `Makefile` are unchanged.

## TL;DR

* **Your questions** (§1).
  - Continuing in this conversation was fine: the workspace was intact.
  - Full logs are the best format.
  - Nothing essential was missing.
* **Round 9 on your hardware** (§2).
  - **d=53:** the part sieved by round 9 ran at 0.525 ns/class, against 0.595 for round 8's part: 12% less time.
  - **d=54:** the cubic tower took 30.4 h (0.328 ns/class). By the tune line the quadratic tower would have taken ~37-41 h.
  - **d = 40-45:** RR's grouped order runs 4-14% faster than round 8's best. At the tower degrees, round 9's loop order is the best one.
  - **The prefetches** cost PH and most round-5- and round-9-order candidates 2-5%, and gained nothing anywhere, so round 10 turns them off.
  - **Polling cost 2.2%**: it woke ~0.8 ms after each chunk had finished.
  - **The T4 and the L4 are power-capped.** Under load they ran at a median of 465 MHz and ~630 MHz, against 1,590 and 2,040 MHz peak. That is why their passes run 15-23% slower than their tune lines.
* **Round 10** (§3).
  - **`--load`** replaces `--load-survivors`; the old name still works.
  - **Re-tuning on resume.** Checkpoints now record which round and GPU chose the degree's engine and loop order. A degree resumed by another round, or on another GPU, is tuned again; `--retune` forces it.
  - **The tuner** times each loop order three times, interleaved, and keeps the median.
  - **Under WSL2, polling is the default.** It now waits once per round of up to 8 chunks queued back to back. A simulation of d=55's chunk sizes puts its cost at ~0.1-0.2%, at the same 1-3% CPU.
  - **Also:** the L1 prefetches are off by default, and the done line shows ns/class.
  - **Validated** in the emulator (255 checks, all PASS) and built warning-free by nvcc 13.3 for sm_75, with every kernel's registers and spills as in round 9. It has not run on a GPU yet.
  - **Moving the running d=55 to round 10 is optional** (§3.4).
* **The data against theory** (§4).
  - At every degree, the share of survivors that gain a factor of degree d matches 1 − e^(−1/d): χ² = 12.3 on 11 degrees.
  - The 842 s with both a degree-52 and a degree-53 factor: 833.5 ± 28.9 expected.
  - **d=55, predicted:** 40,100 ± 200 factors, ~2,185,500 survivors.
  - **Expect about 3 primitive trinomials** with s ≤ r/2 (Poisson: P(none) ≈ 4%).
* **For a new Mersenne prime** (§6): the same program and commands. On a 2070 SUPER d ≤ 55 takes ~7.3 days, whatever r is; three quarters of that is d = 54-55.
* **Beyond d = 55** (§7):
  - what limits the program to d ≤ 55 (the key format);
  - an engine for each degree up to 63, with its table size;
  - a quintic tower for d=55 whose 32 MB table would fit the L2 of large modern GPUs.

## 1. Your questions

**Continuing in this conversation instead of a new one.** It was not a mistake, and it worked: the workspace was still there, with every build and oracle, so no setup was needed.
- **Why I had suggested a new conversation per round:**
  1. **Context.** A conversation keeps everything said and every file read. When it grows too long, the older part is replaced by a summary, which keeps the decisions but loses detail: exact numbers, code I had read. That happened at the start of this round. I could recover what I needed from the files on disk, but a fresh start has nothing to recover.
  2. **Usage.** Every step re-reads the conversation so far, so steps in a long conversation cost more.
  3. **The workspace.** It is reclaimed after a period of inactivity. Then setup.sh has to run anyway.
- **setup.sh's 45 minutes.** Those minutes are mostly spent waiting for compilers and oracle runs. As far as I can tell, waiting on a long command uses little of your usage, since little text goes back and forth. What grows usage is the length of the conversation each step re-reads.
- **So:** after a short gap, continuing is fine. After a long gap, or with a large new task, a new conversation with the dev kit is cheaper and loses nothing. For a round years from now, the dev kit and this handoff are the way (§10).

**Full logs.** They are the best format.
- **No real drawbacks:** I read them in pieces, and the progress dots and checkpoint lines cost little.
- **What a curated excerpt would have lost:** the d=53 resume history, the T4 and L4 clock tables, the L4's machine details, and the d=54 per-run times (§2.4).

**What was missing.** Nothing I needed. These would complete the record:
- the d=55 done line, when it comes;
- an `nvidia-smi` clock log during a cubic-tower pass on the 2070 SUPER (§2.5, optional);
- the survivor counts after d = 48, 49 and 50. §4 uses d = 40-47 and 51-54.

## 2. Round 9 on your hardware

### 2.1 The new loop orders at d = 40-45 (step 1)

All outputs were identical to your files. Times are the done lines; ns per class = time / I(d).

| d | engine | before round 9 (best) | round 9's order | grouped | change |
|---|---|---|---|---|---|
| 40 | tower GF(2^10) | 10.30 s (r7) | **10.02 s** (0.365) | 12.72 s (0.463) | −3% |
| 41 | RR | 26.12 s (r8 grouped) | 24.96 s (0.465) | **22.52 s** (0.420) | −14% |
| 42 | tower GF(2^7) | 39.60 s (r7) | 40.82 s (0.390) | 43.91 s (0.419) | +3% |
| 42 | cubic GF(2^14) | — | r5-order **30.73 s** (0.294); r9 32.06 s; grouped 38.10 s | | **−22%** against the tower |
| 43 | RR | 108.40 s (r7) | 107.02 s (0.523) | **97.65 s** (0.477) | −10% |
| 44 | tower GF(2^11) | 150.09 s (r8 tuned) | **150.08 s** (0.375) | 156.24 s (0.391) | 0% |
| 45 | RR | 428.06 s (r8 grouped) | 436.23 s (0.558) | **408.94 s** (0.523) | −4.5% |

- **What it shows:**
  - RR: the rewritten grouped order wins at every odd degree.
  - The towers: round 9's order wins, and the grouped one loses 4-27%.
  - The cubic tower beats the quadratic tower by 22% even at d=42.
- **The tuner's call at the tower degrees.** At d=44 it put the grouped order 3-4% ahead in step 2 and 3% behind in step 4, while the pass ran it 4% slower. Round 10's repeated timings are for exactly this (§3.1).

### 2.2 The prefetches (step 2)

The `COARSE_PREFETCH=0` tune lines against the `=1` ones, candidate by candidate:

| d | candidates | pf=0 / pf=1, mean | range | the grouped RR candidate |
|---|---|---|---|---|
| 44 | 9 | 0.974 | 0.952-0.996 | — |
| 45 | 4 | 0.988 | 0.971-1.009 | 0.519 → 0.516 |
| 47 | 4 | 0.976 | 0.957-0.996 | 0.525 → 0.523 |
| 48 | 15 | 0.986 | 0.957-1.013 | — |

- **Without the prefetches,** PH and most of the round-5- and round-9-order candidates of the towers and RR ran 2-5% faster. The prefetched lines were evidently evicted again before their loads.
- **The grouped candidates** moved by less than ±1.6%, either way.
- **The cubic tower** was 0.4-1.3% faster *with* them (0.234 vs 0.236 at d=48, for instance), within the noise.
- **The candidate that matters now**, RR grouped, gains ~0.5% without them.
- **Round 10 turns them off by default;** `COARSE_PREFETCH=1` brings them back.

### 2.3 Phase timing (step 3)

Warp-cycles per log and lane:

| | fill | g^u | logs | inversion | marking | total |
|---|---|---|---|---|---|---|
| d=47 RR, r5-order | 5,826 | 5,377 | 22,083 | 451 | 11,934 | 45,671 |
| d=47 RR, r9-order | 5,465 | 5,198 | 20,971 | 541 | 10,608 | 42,783 |
| d=47 RR, grouped | 9,077 | 5,353 | 21,633 | 1,472 | 1,689 | **39,224** |
| d=47 RR, grouped, pf=0 | 8,907 | 5,354 | 21,345 | 1,527 | 1,649 | **38,782** |
| round 8's grouped, for comparison | 10,318 | 4,249 | 22,509 | 2,583 | 3,876 | 43,535 |
| d=48 cubic, r5-order | 9,067 (40%) | 2,230 | 8,265 | 347 | 3,010 | **22,922** |
| d=48 cubic, r9-order | 9,058 | 2,262 | 8,858 | 343 | 2,847 | 23,371 |
| d=48 cubic, grouped | 11,395 | 3,059 | 12,712 | 1,399 | 5,359 | 33,926 |

- **The grouped rewrite** cut RR's total by 10% against round 8's grouped order:
  - the marking went from 3,876 to 1,689;
  - the inversion went from 2,583 to 1,472;
  - the parked fill went from 10,318 to 9,077.
- **The grouped fill is still the cost of that order:** 9,077 against r5-order's 5,826, ~3,200 warp-cycles per log.
- **In the cubic tower the fill is now the largest phase**, 40%. At d=54 this is where any further gain would be.

### 2.4 Production

| d | engine | where | time | ns/class (tune) | survivors after | factors |
|---|---|---|---|---|---|---|
| 51 | RR, r5-order | 2070 SUPER, round 7 | 25,772 s (7.2 h), reconstructed | 0.584 (0.580) | 2,354,612 | 46,598 |
| 51 again | RR, r5-order | T4, round 8 | 43,010.68 s (11.9 h), 0 heavy, files identical | 0.974 (0.821) | | |
| 52 | tower | T4, round 8 | 61,535 s (17.1 h) | 0.711 (0.577) | 2,310,017 | 44,595 |
| 53 | RR grouped | 2070 SUPER, round 8 → 9 | 97,038.07 s (27.0 h) | round 8's part 0.595 (0.564); round 9's part **0.525** (0.518) | 2,266,973 | 43,044 |
| 54 | cubic, r9-order | 2070 SUPER, round 9 | 109,445.49 s (30.4 h) | **0.328** (0.262) | 2,225,592 | 41,381 |
| 55 | RR grouped | 2070 SUPER, round 9 | ~96 h (running since 2026-11-01 12:45) | ~0.528 (0.517) | ~2,185,500 predicted | ~40,100 predicted |

- **The d=53 parts.** Each part's rate is its time over its classes, from your resume lines:
  - round 8: 66,823.84 s for 112,348,952,209,822 classes;
  - round 9: 30,233.43 s for 57,598,203,540,008.

  Round 9 needed 12% less time per class, more than its tune line promised (8%). With round 9 throughout, d=53 would have taken ~24.8 h.
- **d=53's certificate correction.** Your d=53 run started from d=51's survivors, in parallel with d=52, so its 43,886 certificates included 842 s that also have a degree-52 factor. Removing those leaves 43,044, consistent with chance (§4). The survivors after d=53, 2,266,973, are what d=54 loaded, so the chain is consistent.
- **d=54's six runs** each ran at 0.327-0.330 ns/class: steady from the first checkpoint to the last.

### 2.5 The pass against the tune

The tuner times each candidate for a fraction of a second. The pass runs for hours. Their ratio, pass ns/class over tune ns/class:

| GPU | engine | pass / tune |
|---|---|---|
| 2070 SUPER | RR (d = 45, 53, 55) | 1.01-1.02 |
| 2070 SUPER | tower, r9-order (d=44) | 0.99-1.00 |
| 2070 SUPER | cubic tower (d=54) | **1.25** |
| T4 | RR (d=51), tower (d=52) | 1.19, 1.23 |
| L4 | RR (d=55, first 5 minutes) | 1.15 (round 9), 1.16 (round 8) |

- **The T4 and the L4 are power-capped.** Both clock logs give `SwPowerCap` (0x4) as the active reason in every sample:
  - **T4**, during the d=51 pass: median 465 MHz (mean 501, max 1,350), median 66 W, 76-78 °C. Its peak is 1,590 MHz and its base 585 MHz, so it ran below base most of the time.
  - **L4**: 555-1,095 MHz, mostly ~630, at ~72 W, 74-78 °C. Its peak is 2,040 MHz.
  - The tuner's samples are too short for the cap to bite, so they see higher clocks. That is the whole gap.
- **The 2070 SUPER's cubic pass, 25% slower than its tune.** The cause is open. Its d=54 runs held 0.327-0.330 from start to end, so it is not slow drift.
  - **The likely cause** is the same effect: the cubic tower is the most compute-bound engine, with its tables in L2 and shared memory, so it draws the most power. It could hit the card's power limit where the memory-bound engines don't.
  - **How to check:** log `nvidia-smi` during a few minutes of a cubic pass and of an RR pass, e.g. d=48 with `COARSE_METHOD=cubic`.
  - **It didn't change the choice:** the tower's tune rate at d=54 was 54% higher.

### 2.6 The L4, the T4 and the price

- **Random reads:** the L4 reads a 1 GB table at 3.26 G/s, against 2.41 for the 2070 SUPER. Its 48 MB L2 makes it 5.7 G/s on 64 MB.
- **RR at d=55:** ~0.43 ns/class in the pass, ~78 h, against ~96 h on the 2070 SUPER and ~2.2× the T4's speed (d=51: 0.974).
  - At ~3× the T4's price, it costs ~35% more per degree.
  - Your conclusion stands: the T4 is the cheaper cloud card for this, and your own card the cheapest.
- **Round 8 vs round 9 on the L4:** round 8 was ~2% faster (0.368 against 0.376), where on the 2070 SUPER round 9 was 8% faster.
  - The likeliest cause: round 9's prefetches, on by default there, which round 10 turns off.
  - The L4 was never tuned with `COARSE_PREFETCH=0`, so this is a guess.

### 2.7 Polling under WSL2 (step 4)

- **The measurement:** `COARSE_SYNC=poll` took the CPU from ~100% to 1-3%, and d=44 from 151.35 s to 154.61 s, +2.2%.
- **Why it cost that much:**
  - Each GPU has one chunk in flight, so the GPU waits idle from the end of a chunk until the host notices.
  - Polling every millisecond notices ~0.5-1 ms late.
  - Most chunks are short. At d=55 about five in six hold no class at all: their 24-bit prefix cannot start a Lyndon word, e.g. it has an internal zero run longer than its leading one. The rest hold ~6× the mean.
- **A simulation of that size distribution** (Lyndon words counted per chunk at a scaled-down degree, then replayed in the pass order) puts 1 ms polling at 0.3-2.3% lost. The range depends on how long an empty chunk takes, which I don't know.

## 3. Round 10

### 3.1 What changed

- **`--load <file>`** is the new name of `--load-survivors`, which still works. Messages and the usage text now say `--load`.
- **Re-tuning on resume** (§3.2): when the checkpoint's engine and loop order were chosen by another round or on another GPU. `--retune` forces it. `--retune` with `--no-tune` is refused.
- **The tuner's timings.**
  - The candidates that share tables (a variant's three loop orders) are timed in turn, three times over (`COARSE_TUNE_REPS`, 1-99), and each keeps its median.
  - Before the first timing, the first candidate runs for ~0.5 s (at most 8 times), so that the GPU's clocks have left idle.
  - The candidates, their order and the tune line's format are unchanged. A tune now takes ~1-3 s more at d ≥ 40.
- **Polling** (§3.3): the default under WSL2, where it now waits once per round of up to 8 chunks.
  - `COARSE_SYNC=auto` is the default: poll under WSL2, block elsewhere.
  - `block` and `poll` force either one.
  - When polling is on, the run's first line says so.
- **`COARSE_PREFETCH`** now defaults to 0, off. `=1` turns the prefetches back on.
- **The done line** ends with the degree's rate from d=30 up, comparable with the tune line:

      d=54 done (logs/cubic tower, 4194304 chunks): 333599969907456 classes, 0 heavy progressions, 109445.49s in all (this run 26758.30s, resumed at chunk 3171173 of 4194304), 0.328 ns/class

The kernels are unchanged, apart from the prefetch flag's default.

### 3.2 Re-tuning on resume

- **The record.** A checkpoint now ends with a short trailer after best[], recording who chose the degree's engine and loop order:

      round 10 on NVIDIA GeForce RTX 2070 SUPER (sm_75, 40 SMs)

  That is GPU 0's name, compute capability and SM count. The tuner runs on GPU 0 only, so GPU 0 is what counts.
- **Compatibility.** Rounds 6-9 read the file as before and never see the trailer, so every combination still resumes. Tested: round 10's checkpoints under rounds 6-9, and theirs under round 10.
- **On resume, at a degree the tuner covers (d ≥ 40):**
  - **Same round, same GPU:** the recorded choice is kept, as before.
  - **Another round or GPU, or no record** (rounds 6-9 write none; also a damaged trailer): the run says so and tunes again:

        d=55: the checkpoint's choice, RR grouped, was made by an earlier round (6-9); tuning again (--no-tune keeps it)
        d=55: the checkpoint's choice, RR grouped, was made by round 10 on NVIDIA Tesla T4 (sm_75, 40 SMs); this is round 10 on NVIDIA GeForce RTX 2070 SUPER (sm_75, 40 SMs): tuning again (--no-tune keeps it)

  - **`--retune`:** tunes again in any case.
  - **`--no-tune`:** keeps the recorded choice in any case.
- **Why it is safe at any point of a pass:**
  - Every variant computes the same results.
  - The tuner's sample marks only what the pass marks anyway (atomicMin).
  - The new choice goes into the next checkpoint, with the new record.
- **The cost:** one tune, 1-45 s. Most of that is building the tables, which a resume does anyway.

### 3.3 Polling under WSL2

- **Detection:** WSL2 is recognized by its kernel's name in `/proc/sys/kernel/osrelease` (e.g. `5.15.167.4-microsoft-standard-WSL2`).
- **What polling does now:**
  - With one GPU, a round queues up to 8 chunks back to back, each its own launch with its own run counter, and the host polls once for the lot.
  - With several GPUs, rounds stay one chunk per GPU, as before.
  - Each launch is still one chunk. Under Windows a display GPU's watchdog, 2 s by default, can kill a long launch, and WSL2 goes through Windows' driver.
- **The estimate:** in the simulation of §2.7, rounds of 8 lose 0.09-0.19% instead of 0.3-2.3%, at the same ~1,000 polls a second. This is not measured yet.
- **What else changes in poll mode:**
  - Checkpoints and stops come at round boundaries, so mostly at multiples of 8 chunks: ~0.3 s apart at d=55, at most a few seconds.
  - A consistency error names the round's chunks instead of one chunk.
  - A run may switch between poll and block at any resume.
- **My first design for this, now discarded:** it slept through 90% of the previous chunk's time, then polled finely. The simulation showed it worse than plain polling (1.7-5.6% lost), because neighbouring chunks differ too much.

### 3.4 Your d=55 run

Moving it to round 10 is optional.
- **What it would gain:**
  - ~0.5% from the prefetches being off. Round 9 with `COARSE_PREFETCH=0` gets the same.
  - Under WSL2, a free CPU core for the remaining days, at ~0.2% by the estimate above, if you let round 10 poll.
- **What it would risk:** round 10's new host code has run only in the emulator. It touches the chunk loop, the checkpoint trailer and the tuner.
- **If you want it:**
  1. Run its selftest and steps 1-2 of §3.6.
  2. Then Ctrl-C the d=55 run, and run the same command with round 10.
  3. It will print the "earlier round (6-9)" note, tune again (~45 s, picking RR grouped by a wide margin), and continue. It also accepts `--load-survivors`.
- **If anything looks wrong,** go back to round 9: it resumes round 10's checkpoints.

### 3.5 Validation done here

- **Emulated program:**
  - `tests/regress.sh`, round 10's suite: `ALL REGRESSIONS PASS`, in 55 minutes. That is 139 checks of its own (round 9: 120) plus the 116 of the checkpoint suite (round 9: 87): 255 in all, every one PASS. Wherever outputs were compared, they are byte-identical.
  - **New in it:**
    - `--load`, and the old name as an alias;
    - `COARSE_PREFETCH=1` against the oracles;
    - polling in rounds of 8 against oracles OA, OB and OJ (every loop order, the cubic tower, 2 emulated GPUs, fault injection naming the round's chunks);
    - the WSL2 default (test hook `COARSE_TEST_OSRELEASE`), and `COARSE_SYNC=block` overriding it;
    - the tuner with 1, 3 and 5 timings: the same candidates and picks; bad values refused;
    - the rate on the done line from d=30, and its absence at d=29.
- **Checkpoints, K26-K34:**
  - **K26:** round 8's checkpoints are tuned again, or kept with `--no-tune`.
  - **K28:** round 9's too; below `COARSE_TUNE_MIN` nothing changes.
  - **K29:** the trailer's exact text and place; another GPU (test hook `COARSE_TEST_GPU_NAME`) tunes again, and its next checkpoint records that GPU.
  - **K30:** `--retune`, and its conflicts.
  - **K31:** round 9 resumes round 10's file, and round 10 then tunes again.
  - **K32:** a cut trailer: the file is still used, and the degree tuned again.
  - **K33:** `--load` and `--load-survivors` share a checkpoint.
  - **K34:** polling's stops at round boundaries, and switching between poll and block across resumes.
- **The unit tests:** `t_arith`, `t_rot`, `t_bs`, `t_tower`, `t_pair`, `t_cubic` ALL OK.
- **nvcc 13.3.73 for sm_75:**
  - warning-free, 116 kernels;
  - every kernel's registers, stack and spills are identical to round 9's (the kernel code is unchanged but for the prefetch flag's default, a runtime value);
  - the phase-timing build compiles warning-free too.
- **The dev kit, tested as you'd use it:**
  - unpacked into an empty directory, `FULL=0 ./setup.sh` built every binary and made OA, OB, OJ and OBIG, all matching their checksums (31 minutes);
  - then `FULL=0 bash tests/regress.sh`: 248 checks (the full run's 255 without the seven production-range ones), all PASS, in 40 minutes.
- **What I can't test here:** anything about speed or real asynchrony, i.e. polling's actual cost and the tuner's medians on a GPU.

### 3.6 Hardware test plan

    make coarse_sieve gf2_dlog_selftest
    ./gf2_dlog_selftest && ./coarse_sieve --selftest      # also: tower ph rr cubic

1. **Polling against blocking** (~6 min). Each `cmp` should print nothing:

       ./coarse_sieve 136279841 44 a44 --load out43.survivors.txt --no-checkpoint                     # polls under WSL2
       COARSE_SYNC=block ./coarse_sieve 136279841 44 b44 --load out43.survivors.txt --no-checkpoint   # blocks (spins under WSL2)
       cmp a44.found.cert out44.found.cert; cmp b44.found.cert out44.found.cert

   - The first run's first line should read `WSL2: chunks are waited for by polling their events (COARSE_SYNC=block: in the driver)`. The second has no such line.
   - Both done lines should end in `ns/class`.
   - Please send the two done lines, and the CPU of each in `top`. Round 9 gave 151.35 s blocking and 154.61 s polling.
2. **Re-tuning on resume** (~8 min):

       COARSE_TEST_STOP_AFTER_CHUNKS=100 ./coarse_sieve 136279841 45 c45 --load out44.survivors.txt
       ./coarse_sieve 136279841 45 c45 --load out44.survivors.txt --retune
       cmp c45.found.cert out45.found.cert

   - The second run should say `d=45: --retune: tuning again (the checkpoint's choice: RR grouped)`, or whichever the first run chose, and print a tune line.
   - Its done line gives d=45 under round 10, against round 9's 408.94 s.
3. **Production:** when d=55 is done, please send its done line.

## 4. The sieve's statistics

**The heuristic.**
- For a random polynomial of large degree, the number of irreducible factors of degree d is about Poisson with mean 1/d, independently across degrees.
- So a survivor of degrees < d gains a degree-d factor with probability 1 − e^(−1/d).
- Your counts (survivors after d−1, factors found at d):

| d | survivors before | found | share | 1 − e^(−1/d) | ratio | z |
|---|---|---|---|---|---|---|
| 40 | 3,070,345 | 75,672 | 2.4646% | 2.4690% | 0.998 | −0.50 |
| 41 | 2,994,673 | 72,237 | 2.4122% | 2.4095% | 1.001 | +0.30 |
| 42 | 2,922,436 | 69,183 | 2.3673% | 2.3528% | 1.006 | +1.63 |
| 43 | 2,853,253 | 65,323 | 2.2894% | 2.2988% | 0.996 | −1.05 |
| 44 | 2,787,930 | 62,471 | 2.2408% | 2.2471% | 0.997 | −0.71 |
| 45 | 2,725,459 | 59,933 | 2.1990% | 2.1977% | 1.001 | +0.15 |
| 46 | 2,665,526 | 57,723 | 2.1655% | 2.1505% | 1.007 | +1.70 |
| 47 | 2,607,803 | 54,587 | 2.0932% | 2.1052% | 0.994 | −1.35 |
| 52 | 2,354,612 | 44,595 | 1.8939% | 1.9047% | 0.994 | −1.21 |
| 53 | 2,310,017 | 43,044 | 1.8634% | 1.8691% | 0.997 | −0.64 |
| 54 | 2,266,973 | 41,381 | 1.8254% | 1.8348% | 0.995 | −1.06 |

- **The fit:** χ² = 12.3 on 11 degrees (p ≈ 0.34). Pooled: 646,149 found against 646,698 expected (z = −0.7).
- **The overlap:** 842 s have both a degree-52 and a degree-53 factor, against 44,595 × (1 − e^(−1/53)) = 833.5 ± 28.9 expected.
- **The verdict:** at this scale the sieve's output is statistically indistinguishable from the heuristic: no excess, no deficit, no sign of missed or spurious factors.
- **d=55:** 2,225,592 × (1 − e^(−1/55)) = 40,100 ± 200 factors expected, leaving ~2,185,500 survivors.

**How many primitive trinomials to expect.** For r = 136279841, a Mersenne exponent, irreducible and primitive are the same thing.
- **The survivors are 3.17× what a random polynomial would give.** After d=54 they are 3.266% of all s ≤ r/2. A random polynomial would be free of factors of degree ≤ 54 with probability ∏_{k ≤ 54} (1 − 2^−k)^I(k) = 1.030%.
  - The excess comes from the small degrees: a trinomial is never divisible by x or x + 1, for instance.
- **A random polynomial** of degree r with no factor of degree ≤ 54 is irreducible with probability (1/r) / 1.030%. Summed over the survivors: 2,225,592 × 97.07 / r ≈ **1.6**.
- **Swan's theorem doubles that.** For r ≡ ±1 (mod 8) (here r ≡ 1), x^r + x^s + 1 has an odd number of irreducible factors for every s other than 2 and r − 2. Irreducible means one factor, so knowing that the count is odd doubles the chance: **≈ 3.2 expected**.
  - As Poisson: P(none) ≈ 4%, P(at least one) ≈ 96%.
- **The historical record agrees roughly.** These are the Mersenne exponents 6,972,593 to 74,207,281 that Swan's theorem doesn't rule out, from Brent's table. Counts for s ≤ r/2: 1, 2, 4, 1, 3, 5, 4, 0, 3, mean 2.6.
  - r = 57885161 was the first Mersenne exponent not ruled out by Swan's theorem that turned out to have none.

  The estimate is mine, a heuristic, not a theorem.

## 5. How coarse_sieve works

The header comment of `coarse_sieve.cu` is the detailed reference. This section is the map.

**The problem.** For every s in [1, floor(r/2)], find the least-degree irreducible factor of x^r + x^s + 1 over GF(2) up to degree 55, or report that there is none (a survivor).
- s and r − s give reciprocal trinomials with the same factor degrees, so s ≤ r/2 covers both.
- The outputs:
  - `<prefix>.found.cert`: one line `s d p<hexmask>` per s that got its factor in this run (the format of GF2X's `factor.cpp`);
  - `<prefix>.survivors.txt`: the next run's seed.

**The method: sieve over the factors, not over s.**
- **The condition.** Work in GF(2^d) with a primitive modulus, g = x, M = 2^d − 1. An irreducible P of degree d with root β = g^j divides x^r + x^s + 1 iff β^s = 1 + β^r, i.e.

      j·s ≡ Z(u)  (mod M),   u = j·r mod M,   Z(u) = log_g(1 + g^u), the Zech logarithm.

  The solutions s form one arithmetic progression with modulus ord(β). That modulus is ~2^d, so almost always it puts at most one s in range.
- **One computation per class.** The conjugates of β give the same progression, so each class of exponents under u → 2u (mod M), i.e. under rotation of u as a d-bit word, is handled once, through its least rotation: a Lyndon word.
  - There are I(d) = (1/d) Σ_{e|d} μ(e) 2^(d/e) of them, 6.55e14 at d=55.
  - The work per degree is therefore ~2^d/d, **independent of r**. Only best[] (8 bytes per s) and the marking depend on r.
- **Reciprocal pairs** (round 6). The class of −u holds the roots of the reciprocal polynomial, and Z(−u) = Z(u) − u, so one Zech log serves both. That halves everything.
- **The discrete logs** use tables of ~2^(d/2) entries or fewer, never 2^d. There are four engines (`gf2_dlog.h`):

| engine | where | how | tables at the top degrees |
|---|---|---|---|
| quadratic tower | even d | GF(2^d) = K(w), K = GF(2^(d/2)); log(b(c + w)) = (2^(d/2)+1)·log_K(b) + T[log_K(c)]; the logs in K by the same trick over a subfield of K, or by a table of K | T: 8·2^(d/2) bytes (1 GB at d=54) |
| cubic tower (round 9) | 6 \| d | K = GF(2^(d/3)), w³ ∈ K; x = r/t with r, t linear in w (a rational reconstruction over K that never fails); four logs in K (shared memory) and two reads of T3 | T3: 8·2^(d/3) bytes (2 MB at d=54, held in L2) |
| RR, rational reconstruction | odd d | half an extended Euclid writes y = r/t with deg r, t ≤ d/2; log y = L[r] − L[t], L the logs of all polynomials of degree ≤ d/2 | L: 8·2^(d/2+1) bytes (2 GB at d=55) |
| Pohlig-Hellman | odd d | log mod each prime-power part of M, by table or baby-step/giant-step, then CRT | within free memory; slower than RR wherever RR fits |

**The batched kernel** (`k_logs_batch<D, METHOD, GROUP>`), per thread:
- it enumerates its range of u with FKM, the warp's lanes pooling the classes they find (round 7);
- it computes g^u for 32 classes at a time in bit-sliced form (one u32 per coefficient);
- it takes their logs;
- it inverts all 32 with one modular inversion (Montgomery's trick; the classes with gcd(u, M) > 1 join through a unit lift, round 5);
- it marks best[s] by atomicMin of a key (degree, then the polynomial) for the rare s that fall in range.
- Three loop orders differ in how the batch's state moves through local memory: r5-order, r9-order, and grouped (8 classes held in registers).
- Every log is range-checked and one batch in 8 is verified (g^v = g^u + 1). The inversion is checked (inv · ∏J = 1), and the class count against I(d). Any inconsistency stops the run before an output or checkpoint could include it.

**The pass over u**:
- **Chunks.** The u range is cut into chunks of 2^31 u values, one launch each, which is also the checkpoint granularity.
- **The order.** Classes crowd at small u, and about five in six chunks at d=55 hold none. So the pass takes the chunks in a scrambled, stratified order (blocks of 8), which keeps the progress dots steady and the time-left estimate honest.
- **Multi-GPU (`--gpus N`).** The chunks are dealt out in rounds; each GPU keeps its own best[], merged by minimum.

**Around it:**
- **The tuner** (round 5; repeated timings in round 10). At d ≥ 40 it times each engine, inner subfield and loop order on a dense sample on GPU 0, and the fastest sieves the degree.
- **Checkpoints** (format v4, rounds 6-10). Every 15 min and on SIGINT/SIGTERM, atomic, three generations. They are tied to the run's command line and seed, and resume across rounds 6-10 and across GPU counts.
- **A lock file** keeps two runs off the same checkpoint.
- **A watchdog** (round 8) exits with status 3 after 30 min without progress, e.g. a hung GPU.
- **Selftests** check every engine at every d ≤ 55 against direct computation. **The phase-timing build** (`coarse_sieve_prof`) reports where the kernel's time goes.

**Files** (`src/r10/`):

| file | what |
|---|---|
| `coarse_sieve.cu` | the program: kernels, tuner, pass, checkpoints, selftest |
| `gf2_dlog.h` | GF(2^d) arithmetic, bit-sliced arithmetic, FKM, the four log engines, the reciprocal-pair rule |
| `gf2_wide_field.h`, `gf2_wide_setup.h` | u64 field arithmetic and primitive moduli |
| `gf2_dlog_selftest.cpp` | the host selftest of `gf2_dlog.h` |
| `t_*.cpp` | unit tests (arithmetic, rotations, bit-slicing, towers, pairs, cubic tower) |
| `cuda_emul.h` | the CPU emulation of the CUDA runtime, with lockstep warps |
| `Makefile` | `make coarse_sieve coarse_sieve_prof gf2_dlog_selftest` (and your other tools) |

## 6. Running it for a new Mersenne prime

**Build and check** (CUDA 12 or later; `-arch=native` builds for the card it runs on, ~6-12 min):

    make coarse_sieve gf2_dlog_selftest        # or the nvcc line in coarse_sieve.cu's header
    ./gf2_dlog_selftest
    for e in "" tower ph rr cubic; do ./coarse_sieve --selftest $e; done      # each ends "selftest: PASS"

**The runs.** Degrees 2-46 together (minutes to an hour), then one degree per run, each from the previous run's survivors:

    R=<the exponent>
    ./coarse_sieve $R 2-46 out2-46
    ./coarse_sieve $R 47 out47 --load out2-46.survivors.txt
    ./coarse_sieve $R 48 out48 --load out47.survivors.txt
    ...
    ./coarse_sieve $R 55 out55 --load out54.survivors.txt

- **Stopping and resuming.** Ctrl-C (or SIGTERM) saves and exits; the same command continues, with any later round's binary too.
- **The record.** Keep every run's `.found.cert`: together they are the complete record, and `sort -n` merges them.
- **The final check, the one you made:** every s in [1, floor(r/2)] should appear exactly once across all the certificates and the last survivors file:

      cat out*.found.cert | cut -d' ' -f1 | cat - out55.survivors.txt | sort -n | uniq -d | wc -l   # 0: no s twice
      cat out*.found.cert | cut -d' ' -f1 | cat - out55.survivors.txt | wc -l                      # floor(r/2)

**The options** (`coarse_sieve` with no arguments prints them; the header of `coarse_sieve.cu` has every environment switch):

| option | what |
|---|---|
| `--load <file>` | the seed: the survivors of the run through D1−1 (round 10; `--load-survivors` still works) |
| `--checkpoint <file>`, `--checkpoint-mins <N>`, `--no-checkpoint`, `--fresh` | where and how often to checkpoint (default `<prefix>.ckpt`, 15 min); `--fresh` sets an old checkpoint aside |
| `--gpus <N>` | use N GPUs (default: all) |
| `--no-tune`, `--retune` | skip the tuner; tune again on resume |
| `--watchdog-mins <N>` | exit with status 3 after N minutes without progress (default 30) |
| `--no-cert`, `--no-survivors` | skip an output file |
| `COARSE_SYNC=auto\|block\|poll` | how the host waits for chunks (auto: poll under WSL2) |
| `COARSE_METHOD=tower\|cubic\|rr\|ph`, `COARSE_PIPE=0\|1\|2` | force an engine or a loop order (the tuner then times only those) |

**Two degrees on two machines at once,** e.g. d=52 on one and d=53 on another, both from out51's survivors. Then merge:

    comm -12 <(sort out52.survivors.txt) <(sort out53.survivors.txt) | sort -n > out53m.survivors.txt
    awk 'NR==FNR {f[$1]; next} !($1 in f)' out52.found.cert out53.found.cert > out53m.found.cert

- The merged survivors are the s with no factor of degree ≤ 53.
- The merged certificate drops the s that have a degree-52 factor. Your 842.

**Time, on an RTX 2070 SUPER with round 10's code.** This is per degree and does not depend on r. A faster GPU divides it.

| d | engine (the tuner's choice) | classes | ns/class | time |
|---|---|---|---|---|
| ≤ 45 | | | | ~15 min in all |
| 46 | classic tower | 1.53e12 | ~0.75 (a guess) | ~20 min |
| 47 | RR grouped | 2.99e12 | 0.525 | 26 min |
| 48 | cubic | 5.86e12 | ~0.29 | 28 min |
| 49 | RR grouped (the only engine) | 1.15e13 | 0.525 | 1.7 h |
| 50 | tower | 2.25e13 | ~0.5 (a guess) | ~3 h |
| 51 | RR grouped | 4.42e13 | 0.525 | 6.4 h |
| 52 | tower | 8.66e13 | ~0.43 (from the T4's) | ~10 h |
| 53 | RR grouped | 1.70e14 | 0.525 | 24.8 h |
| 54 | cubic | 3.34e14 | 0.328 | 30.4 h |
| 55 | RR grouped | 6.55e14 | 0.528 | ~96 h |
| | | | **total** | **~7.3 days** |

**Memory.**
- **best[]:** 4·r bytes on every GPU (545 MB at r = 136279841).
- **The degree's tables:** up to 2 GB (RR at d=55), 1 GB for the quadratic tower at d=54 if it were chosen. Building d=49's RR table takes a baby-step table of half the card for a minute or two; the result is cached in `coarse_rr_d49.tbl`.
- So an 8 GB card has room for r up to ~1e9.

**What does change with r:**
- best[] and the survivors files grow with r;
- the share of survivors depends a little on r through its residues (§4: 3.17× the random rate here);
- the program refuses an r with gcd(r, 2^d − 1) > 1. For a prime r that happens only if r divides some 2^d − 1 with d ≤ 55 (e.g. 164511353, a factor of 2^41 − 1). No Mersenne exponent does so far.

**Hardware notes:**
- **Random reads matter most.** The kernel is bound by random 8-byte reads and local-memory traffic; `tools/randread.cu` measures a card's random-read rate. More L2 helps the tower and cubic tables, and more memory bandwidth helps everything.
- **Cloud cards are often power-capped** (the T4 at 70 W, the L4 at 72 W): their passes run 15-25% slower than their tune lines. A clock log (`nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv -l 60`) shows it.
- **Under WSL2,** round 10 polls by default, so a core stays free.
- **Unattended cloud runs** can loop on the watchdog's exit status:

      until ./coarse_sieve ...; do [ $? -eq 3 ] || break; sleep 60; done

## 7. Beyond d = 55

**What stops the program at 55.** The best[] key is one u64 holding the factor's degree and its d+1 coefficients, ordered by degree, then by polynomial:
- (d << 48) | mask up to d = 47;
- 0xF << 60 | (d − 48) << 56 | mask for 48-55.

A degree-56 mask needs 57 bits.

**What d ≥ 56 would need:**
1. **A new key.** The leading and constant coefficients are always 1, so d − 1 bits suffice: (d << 58) | the middle bits works to d = 59. Single-degree runs (the norm for d ≥ 48) could drop the degree from the key, which works to d = 63. The cert writer and the seed sentinels (0 and ~0) carry over.
2. **New kernel instantiations,** D = 56 and up. Each degree adds compile time, and the bit-sliced arrays of D words per thread add registers; the spill picture needs a fresh look.
3. **A log engine per degree.** The number to watch is the table size against VRAM and L2 (8-byte entries). PH needs the largest prime factor p of 2^d − 1 to be small.

| d | 2^d − 1, largest prime | engines (table) | 2070 SUPER at 0.5 ns/class |
|---|---|---|---|
| 56 | 15,790,321 | RR (4 GB); quadratic over GF(2^28) (2 GB); **degree-7 tower over GF(2^8) (130 MB)** | 7.4 days |
| 57 | 1,212,847 | RR (4 GB); **cubic over GF(2^19) (4 MB, plus K's logs: a 2 MB table)**; PH | 14.6 days |
| 58 | 3,033,169 | RR (8 GB); **quadratic over GF(2^29) (4 GB; 2 GB with the norm trick below), K's logs by RR in K**; PH | 29 days |
| 59 | 3,203,431,780,337 | **RR only (8 GB)**: prime d, and PH is out | 57 days |
| 60 | 1,321 | **cubic over GF(2^20) (8 MB; K by its classic tower, as at d=54)**; quintic over GF(2^12) (128 MB); PH (very smooth) | 111 days |
| 61 | 2^61 − 1 is prime | RR only (16 GB) | 219 days |
| 62 | 2,147,483,647 | quadratic over GF(2^31) (16 GB; 8 GB with the norm trick), K's logs by RR | 431 days |
| 63 | 649,657 | **cubic over GF(2^21) (16 MB, plus K's logs: an 8 MB table)**; PH | 2.3 years |

So d = 56-58 are within reach of one consumer GPU in weeks; beyond 60 it takes many GPUs or many years.

**The general tower** (the cubic one is its m = 3 case). Let m | d, K = GF(2^n), n = d/m, L = K(w) of degree m over K, and k = ⌈(m−1)/2⌉.
- **The split.** For any x in L there is a nonzero t of degree ≤ k in w with r = x·t also of degree ≤ k. The coefficients of w^(k+1)..w^(m−1) in x·t give m − 1 − k ≤ k linear equations in k + 1 unknowns, so a solution always exists and it never fails.
- **The logs.** log x = log r − log t. A polynomial of degree k in w, made monic, is looked up in a table indexed by the logs in K of its k other coefficients: (2^n + 1)^k entries, plus smaller tables for lower degrees. On top come 2(k + 1) logs in K per class.
- **The sizes:**
  - m = 3 (k = 1): tables of 2^(d/3), the cubic tower;
  - m = 5 (k = 2): 2^(2d/5), **32 MB at d=55 (n = 11)**;
  - m = 7 (k = 3): 2^(3d/7), 130 MB at d=56.
- **The case worth trying first: d=55 on a GPU with a large L2.**
  - The quintic table fits the L2 of an L4 (48 MB) and of today's large consumer cards. Its reads would then come from L2, where RR's come from DRAM.
  - On the 2070 SUPER (4 MB L2) its reads would still go to DRAM, so no gain.
  - It needs a non-binomial modulus over GF(2^11): 5 ∤ 2^11 − 1, so w^5 = c is never irreducible there. So it is more arithmetic per class than the cubic tower's.
  - This is the one real speedup left that I can see. It is a round's work, and it matters only for a future run at d=55 on such a card.

**The norm trick (halving T's entries).** For L ⊃ K, the norm N(x) = x^H with H = (|L|−1)/(|K|−1) satisfies log_L(x) ≡ log_K(N(x)) (mod |K|−1).
- When gcd(H, |K|−1) = 1, always so for the quadratic tower, a table can store log_L(x) mod H only, and get the rest from one more log in K.
- For quadratic towers with |K| ≤ 2^31 that gives 4-byte entries: half the memory, the same number of reads.

**Smaller ideas, not done:**
- **Skip empty chunks on the host.** About five in six of d=55's chunks hold no class: those whose 24-bit prefix is not a prenecklace (1.47M of the 8.4M prefixes are). The saving is unknown, since an empty chunk is fast already.
- **Keep a second chunk queued in blocking mode**, so that the GPU never waits for the host's bookkeeping: ~0.1-0.3% at d=55.
- **Shrink the grouped order's fill.** It costs ~3,200 warp-cycles per log more than round 5's order (§2.3), ~8% of a grouped RR pass.
- **A tune sample closer to the pass.** It would matter only if the cubic tower's 25% gap (§2.5) were not clocks.

**Is going deeper worth it?** Sieving degree d removes ~1/d of the survivors (~38,700 at d=56) and costs I(d) classes (7.4 GPU-days at d=56 on a 2070 SUPER, doubling with each degree).
- So the coarse sieve stops paying where a survivor costs the next stage less than (that GPU time)/(survivors removed): ~16 GPU-seconds each at d=56, ~33 at d=57, ~65 at d=58.
- The next stage's per-survivor cost is the number that decides. You know it better than I do (§8).

## 8. After the coarse sieve

d=55 leaves ~2.19 million s.
- **What remains for each:** either find its least factor (degree 56 and up) or prove it irreducible. Your stage-2 tools (`trinomial_stage2`, `tsfactor` and `cantor_cuda` in the Makefile, which I haven't seen) are for that.
- **Brent and Zimmermann's figures for r = 74207281** give the scale: squaring a polynomial of degree r took ~2 ms, multiplying two took ~700 ms, and checking all the certificates of that exponent took 22 minutes on one core. Their largest least factor had degree 19,865,299.
- **Where the cost lies.** At r = 136279841 a proof of irreducibility is ~r squarings, days of CPU at ~4 ms each.
  - The heuristic of §4 says ~3 survivors will need it.
  - The rest need squarings up to their least factor's degree, plus the gcds that detect it. Their least degree is spread like 55/x: half are found below ~110, 5% only above ~1,100, 0.5% above ~11,000.
- **For the record, the arithmetic:**
  - a coarse-sieve survivor's expected number of squarings until its least factor is ~55·(1 + ln(r/55)) ≈ 860, so ~1.9e9 squarings for all of them: ~80 core-days at ~4 ms, before the gcds;
  - the gcds and the products that batch them, not the squarings, are what Brent and Zimmermann's multi-level blocking was designed to make cheap.

## 9. History

| round | date | what changed | effect |
|---|---|---|---|
| 1 | 2026-10-18 | LFSR stepping in the bucketed sieve (one shift and XOR per element instead of a d/2-multiply power) | ~1000× per element (d=36 had taken 12.3 h) |
| 2 | 10-19 | the logs path: one Zech log per class from tables of 2^(d/2) entries (tower at even d, PH at odd d); no buckets | 2× per degree instead of 4×; VRAM no longer sets the speed |
| 3 | 10-20 | the batched kernel: FKM enumeration, 32 classes per thread bit-sliced, Montgomery batch inversion; the PH planner; depth 47 → 55 | d=47 5,411 s (3.9×) |
| 4 | 10-22 | the subfield tower: K's logs from a subfield, one DRAM read per tower log | d=44 668.65 s (1.16×) |
| 5 | 10-24 | irregular classes join the batch inversion; RR and degree 49; auto-checkpoints; the tuner | 1.97× at d=44; d=47 (RR) 3,648 s; d=49 14,185 s; d=50 34,627 s |
| 6 | 10-25 | one Zech log per reciprocal pair; the scrambled pass order and honest ETA; a quick tuner; one build, old paths removed | d=44 268 s, d=47 2,142 s, d=49 8,369 s |
| 7 | 10-26 | full batches (warps pool their classes); a cheaper pair tie-break; times of all runs in checkpoints; lockstep warps in the emulator | d=44 150 s, d=45 434 s (1.3-1.9×); d=51 production |
| 8 | 10-28 | the watchdog and lock-file notes; the grouped loop order and TP-global as candidates; table reads past L1; the phase-timing build | d=52 on a T4 for $1.98; profiles showed local-memory traffic as the main wait |
| 9 | 10-29 | the cubic tower; round 9's loop order; the grouped order rewritten (rolled, fill parked); g^u's planes three steps ahead; the inversion check; prefetches (undone in 10) | d=41-45 4-14% faster; d=53 12%; d=54 in 30.4 h |
| 10 | 11-01 | `--load`; re-tuning on resume, with the tuning record; repeated tuner timings; polling by default under WSL2, in rounds of 8; prefetches off; ns/class on the done line | robustness and a free CPU core under WSL2 |

**From round 2's 5.9 h to round 9's ~26 min (estimated from its tune line), d=47 sped up ~13×**:
- the tables instead of buckets;
- bit-slicing and batch inversion;
- RR;
- the reciprocal pairs (2×);
- full batches;
- the local-memory work of rounds 8-9.

At d=44, round 4's 669 s became 150 s.

## 10. The final dev kit

`coarse_sieve_devkit_final.tar.gz` unpacks to `coarse_sieve_devkit/`:

| path | what |
|---|---|
| `src/r10/` | the current sources (§5) |
| `src/r5/` - `src/r9/` | earlier rounds, for the checkpoint-compatibility tests; round 5 also has the dense and bucketed sieves that make the oracles |
| `setup.sh` | builds the emulated binaries (round 10's `cs` and `csprof`, rounds 5-9), then the oracles (~45 min on 2 cores; `FULL=0` skips the two big ones) |
| `tests/regress.sh`, `tests/ckpt_tests.sh` | the suites (~50 and ~4 min); they end `ALL REGRESSIONS PASS` and `ALL CHECKPOINT TESTS PASS` |
| `fixtures/` | the seed, the oracle generator and the oracles' checksums |
| `nvcc/build_check.sh` | the real nvcc build, for warnings and per-kernel registers and spills |
| `tools/` | the analysis programs of rounds 7-10, including `randread.cu` and, new, `analysis/` (this round's statistics, tower survey, factorizations and the polling simulation) |
| `docs/` | this handoff and those of rounds 1-9 |

**To continue in a new conversation, attach:**
- the dev kit;
- this handoff (also the project doc `claude/coarse_sieve_handoff.md`);
- your logs.

The kit's README says how to rebuild the workspace (`./setup.sh`) and check it (`bash tests/regress.sh`). In a workspace without CUDA, nvcc comes from pip; the README has the line.

Thank you for all the careful testing, round after round. The logs, the checksums and your d=53 correction made this work.

Sources:
- [R. P. Brent, table of primitive trinomials of Mersenne exponent degree](https://maths-people.anu.edu.au/brent/ftp/trinom/table.txt) (the counts in §4)
- [Brent and Zimmermann, *Twelve new primitive binary trinomials* (arXiv:1605.09213)](https://arxiv.org/pdf/1605.09213) (the costs in §8; r = 57885161 has none)
- [Brent, Larvala and Zimmermann, *A fast algorithm for testing reducibility of trinomials mod 2* (Math. Comp. 2003)](https://maths-people.anu.edu.au/~brent/pd/rpb199.pdf)
- [R. P. Brent, primitive trinomials page](https://maths-people.anu.edu.au/~brent/trinom.html)

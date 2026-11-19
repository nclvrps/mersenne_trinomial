# coarse_sieve, round 8: a watchdog, a clearer lock, and three experiments on the memory stalls

**Baseline: round 7 (2026-10-26), which you validated on hardware.**
- All selftests passed. Regressions at d = 40-44 identical; the d=51 production run matched the known ranges, with no false positives or negatives.
- Round 7 against round 6, measured: d=44 150 s (268 s), d=45 434 s (568 s), 1.3-1.9x over d = 40-45. At d=44 the fills went 242 → 215 → 174 → 151 s (`COARSE_FILL` 0 → 3).
- The resumed done line showed the time of all runs, as asked.

## Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | Round 8. |
| `cuda_emul.h` | `__ldcg` and `clock64()` for the emulated build. |
| `Makefile` | `coarse_sieve` links with `-pthread` (the watchdog thread). New target `coarse_sieve_prof`: the phase-timing build (section 6). |
| `coarse_sieve_round8.patch` | All of the above as a patch against round 7 (flat paths). |
| `coarse_sieve_devkit_r8.tar.gz` | The dev kit for continuing in a new conversation (section 12). |

`gf2_dlog.h`, `gf2_dlog_selftest.cpp`, `gf2_wide_field.h`, `gf2_wide_setup.h` and `tools/` are unchanged.

## TL;DR

* **Your questions:**
  - The `.lock` file is only there to hold a lock. A leftover one is harmless, and `--fresh` is right to leave it alone (section 7).
  - The hang was environmental. The first Ctrl-C couldn't act on it, because stops are taken between chunks.
  - Long threads: yes, start a new conversation per round. The dev kit makes that easy.
* **The lock file** (section 7).
  - A leftover lock file now gets a note saying it is harmless.
  - A double Ctrl-C removes the run's lock file.
  - A small race in taking the lock is closed.
* **A watchdog** (section 7). If nothing progresses for `--watchdog-mins` (default 30), the run says so and exits with status 3. The checkpoint is intact, so a wrapper loop can restart an unattended cloud run.
* **Round 7's pass now runs at the tuner's rate** (section 1). The fill fix closed the whole 1.2-1.9x gap. My predicted 20% drop in the tune lines did not happen (3-6%).
* **What the profiles say** (section 2).
  - All three kernels are latency-bound: 19-30% of issue slots used.
  - The batch arrays in local memory are 64-83% of the L1 traffic, and 90% of their loads miss L1. RR at d=47 waits on memory 70% of the time.
  - My hypothesis: the batch arrays' round trips to L2 and DRAM are the main stall.
* **Round 8 tests that hypothesis three ways.** The tuner or a simple A/B run measures each:
  1. **A grouped loop order** (section 3), a new tuner candidate at d ≥ 40: groups of 8 classes in registers, four inversions per batch instead of one.
  2. **Table reads that bypass L1** (section 4), the default; `COARSE_LDCG=0` for the A/B.
  3. **TP in global memory** at d = 48 and 52 (section 5), a tuner candidate. It trades shared memory for twice the L1, and decides d=52 on the T4.
* **A phase-timing build**, `make coarse_sieve_prof` (section 6). It shows where each loop order's time goes.
* **I can't predict which way these land.** Each has a real cost against its saving. The tune lines will tell.
* **Before production with round 8,** please validate the grouped kernels on your GPU (step 1 of section 10, ~15 min). They are new code that the emulator has checked but your GPU hasn't run. `COARSE_PIPE=0` keeps the tuner away from them.

## 1. Round 7 on your card

The pass time per class, against the tune line's (your round-7 log; the pass includes the table build and tuning, a second or two):

| d | engine | pass, ns/class | tune, ns/class | ratio (round 6's) |
|---|---|---|---|---|
| 40 | tower GF(2^10) | 0.374 | 0.342 | 1.09 (1.94) |
| 41 | RR | 0.501 | 0.479 | 1.05 (1.34) |
| 42 | tower GF(2^7) | 0.378 | 0.364 | 1.04 (1.80) |
| 43 | RR | 0.530 | 0.533 | 0.99 (1.24) |
| 44 | tower GF(2^11) | 0.376 | 0.380 | 0.99 (1.75) |
| 45 | RR | 0.555 | 0.548 | 1.01 (1.25) |

- **What this shows.** The tuner's dense sample and the real pass now run at the same rate per class: the real pass has full batches.
- **My side prediction failed.** I expected the tune lines themselves to drop ~20%, from the shorter runs in the tuner's sample. They fell 3-6%: the run quantization I blamed in round 7 was a small effect. The gap was the fill.
- **d=51:** the tune line said 0.577 ns/class, i.e. ~7.1 h for 4.4e13 classes, and the first dot's estimate was 7.0 h. If you still have the done line, I'd like it for the projections.

## 2. What the profiles say, and what they don't

**Round 6's fill against round 7's, d=44, the same chunk (131: 852M classes):**

| | fill 0 (round 6) | fill 3 (round 7) |
|---|---|---|
| kernel time | 630.7 ms | 494.3 ms (1.28x) |
| active threads per warp | 14.7 | 20.8 |
| instructions | 50.2e9 | 42.3e9 |
| issue slots busy | 27.9% | 30.2% |

**Where the time goes**, in all three profiles (d44_fill0, d44_fill3, d47_rr):
- **Latency-bound.** With 4 warps per scheduler (occupancy 50%, limited by 126-128 registers), the schedulers find no eligible warp 70-81% of the time.
- **The top stall is waiting on memory** ("L1TEX scoreboard"): 45% of each warp's time at d=44, 70% for RR at d=47.
- **Local memory dominates L1.** It is 64% (RR) to 80-83% (tower) of all L1 sectors, and 90% of its loads miss L1. That traffic is the batch arrays: the 32 classes per lane with their u, x, log, prefix product and gg, ~1 KB per thread, so 512 KB per SM against a 64 KB L1. It is about half of all L2 traffic, and L2's hit rate is only 37% for RR, 61-63% for the tower, so much of it goes on to DRAM.
- **The table reads are random**, using 8 of every 32 bytes fetched. That's inherent: one log per read.
- **The cost per log** (one log serves a class and its partner): RR at d=47 runs ~1,530 thread-instructions and ~40,000 warp-cycles per log and lane. The schedulers issue in about one cycle in five.

**What I counted in the SASS.** Per class, the round-5 order makes ~12 local-memory accesses:
- one store in the fill;
- one load and one store in the bit-slice transposes;
- in the forward loop, two loads and three stores;
- in the backward loop, four loads.

**What I can't tell from the summary pages:** *which* instructions the warps wait at. Two exports from your existing reports would show it per SASS instruction, without re-profiling:

    ncu --import d47_rr.ncu-rep --page source --print-source sass --csv > d47_rr_sass.csv
    ncu --import d44_fill3.ncu-rep --page source --print-source sass --csv > d44_fill3_sass.csv
    nvcc --version

With your nvcc version I can build the same SASS here and map the addresses back to the code. The phase timing of section 6 answers the coarser question of which phase gets the time.

## 3. The grouped loop order (`grouped`, `COARSE_PIPE=2`)

**Why.** The round-5 and round-6 orders carry each class's log, unit J, prefix product and gg from the forward loop to the backward one through the batch arrays: three stores, then four loads, tens of thousands of instructions apart, in a working set L1 can't hold. Montgomery's trick needs all 32 classes' prefix products before its one inversion, so those arrays can't simply shrink.

**What the grouped order does.**
- It cuts each batch into four groups of 8 classes, each with its own inversion.
- The loops over a group are unrolled, so the group's v, J, prefix product and gg are indexed by constants and live in registers.
- Only u (from the fill) and x = g^u + 1 (from the bit-sliced phase) pass through local memory: written once, read once.
- In the SASS, the group loops have no local stores at all and two loads per class: u and x.

**What it costs.**
- **Four inversions per batch instead of one:** by my count ~5-8% more instructions, depending on the exponent λ(M) − 1 (19 bits at d=52, 29 at d=51, 33 at d=53).
- **Eight copies of a class's code.** That is the main risk.
  - **RR (odd d):** a group loop of ~6,700 SASS instructions (105 KB). By my reading its hot part is ~200 instructions per class, ~26 KB for the eight, since the irregular classes are rare. That no longer fits the 16 KB L0 instruction cache, but should fit L1's 46 KB.
  - **Tower (even d):** ~11,400 instructions (178 KB). About half the classes are irregular (3 always divides 2^d − 1), so their lift and marking code is hot too: ~600 instructions per class, ~75 KB for the group. That overflows L1's instruction cache, and **I expect the grouped order to lose at even d**, unless the local-memory savings outweigh the instruction misses.
  - The round-6 order's slowdown at d=51 (1.25 and 2.30 against 0.58 ns/class in your tune lines) may be this same kind of effect. I don't know.
- **Small spills, partly in hot code.** ptxas reports 52-68 bytes of spill stores in the grouped kernels (the other orders have none). In the SASS of all 16:
  - the group loops reload one 4-byte value per group of 8 classes, and 4-6 accesses come once per batch;
  - but the enumeration loop of the pooled fill (the default), one pass per step of the enumeration, has 9 (RR) or 11 (tower) spill loads and stores. A few values of the enumeration's state, such as the end of the lane's current run, are reloaded at every step and stored when they change.
  - That adds latency to the fill phase: ~128 steps per batch, each waiting on a reload. By my estimate that is 1% of a batch's time if the reloads hit L1, several % if they miss. The phase timing's fill figure will show it.
  - If the grouped order wins, this is the first thing to fix. A first try, not in this round's code, didn't do it: counting the classes once per deal instead of at each find left 9 spill accesses in that loop, and ptxas then spilled more overall (84 bytes). The next options are a smaller group or a register limit above 128 for these kernels, each with a cost of its own.
- **Compile time.** Grouped kernels exist only for d ≥ 40 (16 kernels), which makes the build ~8 minutes instead of 5.5.

**A check the other orders lack.** The first inverse of each group's backward step is multiplied back (J · J⁻¹ ≡ 1 mod M), and a mismatch stops the run as a consistency error.

**In the tuner.** Every tower and RR variant at d ≥ 40 is now timed in three loop orders, labelled `r5-order`, `r6-order` (round 6's, previously unlabelled) and `grouped`.
- Round 5's is now the default when nothing is tuned (d < 40, `--no-tune`). It won every one of your round-7 tune lines.
- `COARSE_PIPE=0|1|2` fixes the order, as before.
- The checkpoint records it. Round 7's binary continues a grouped checkpoint in its own pipelined order: the same results, tested.

## 4. Table reads that bypass L1 (`COARSE_LDCG`)

- **What changed.** The random reads of the big tables (tower T, RR's L) now go through L2 only (`__ldcg`).
- **Why.** Each read fills an L1 sector that is never read again. A tower log reads T once per class and RR reads L twice, and each such fill can evict a batch-array line that will be read again.
- **What it should buy.** A modest gain at best (estimate): the batch arrays' working set is 8x L1 either way, so a better L1 hit rate can only help a little. It can't make a table read slower: those miss L1 at these sizes regardless.
- **The A/B.** `COARSE_LDCG=0` restores reads through L1. Section 10's step 3 compares the two.

## 5. TP in global memory at d = 48 and 52 (`TP-global`, `COARSE_TOWER_TP`)

- **The trade-off.** The subfield tower stages its small tables in shared memory: the byte tables, the inner logs LKn, and the projective-line logs TP.
  - At d=52 (inner GF(2^13)) they total 62 KB, of which TP is 32 KB.
  - Past 32 KB, a Turing SM must switch to its larger shared-memory carveout, which halves L1 from 64 to 32 KB. That is where the batch arrays are cached.
  - With TP in global memory instead (cached in L1 and L2), the shared tables need 30 KB and L1 stays at 64 KB. But TP's two reads per class then go to L1 or L2 instead of shared memory.
- **What decides it.** I can't tell which wins. So when staging TP takes the shared tables past 32 KB, which happens at d=48 and d=52 only, the tuner times both: `GF(2^13) r5-order TP-global`, and so on. The degree line then says `TP from global memory`.
- **The checkpoint** records the choice (flag 16). Round 7 ignores the flag, with the same results.
- **Forcing it.** `COARSE_TOWER_TP=shared|global` fixes the placement.

## 6. Phase timing (`make coarse_sieve_prof`)

- **What it is.** A diagnostic build (`-DCOARSE_PHASE_TIMING`); the production build has none of this code. Lane 0 of every warp adds the `clock64()` cycles of each phase of the batched kernel to six counters. Each degree, or a stop, then prints:

      [phase timing d=47, warp-cycles per log and lane: fill 3700 (9.2%), g^u 3000 (7.4%), logs 24300 (60.1%), inversion 760 (1.9%), marking 8400 (20.8%), other 240 (0.6%); total 40400; 9709438 logs in 303420 warp batches, 31.99 of 32 per lane]

  (Illustrative numbers.)
- **The phases:**
  - fill: the enumeration and the deal;
  - g^u: the bit-sliced exponentiation;
  - logs: the forward step, including the table reads;
  - inversion;
  - marking: the backward step;
  - other: start-up and tail.
- **How to read it.** These are wall-clock cycles of the warp, stalls included, per log of each lane, since a warp's cycles serve its 32 lanes at once. So a phase's number shows where the time goes, not how much work it does. Where a phase's share is far above its share of instructions, that phase waits on memory.
- **Its own perturbation.** The timing needs a few registers. The tower kernels then spill ~40 bytes (none in the production build) and the grouped ones 80-110, so its numbers run a little pessimistic.

## 7. The lock file and the watchdog

**The lock** (`<checkpoint>.lock`). What counts is the `flock` on it, and the kernel drops that the moment its process ends.
- **A leftover file is harmless.** Round 8 says so when it finds one: `note: K.ckpt.lock was left by an earlier run that did not exit normally (killed or crashed); harmless -- its lock ended with it`.
- **A second SIGINT/SIGTERM** now unlinks the file before the process dies (round 7 left it). The exit status is still that of a run killed by the signal, 130 for SIGINT.
- **A race fixed.** A run that exits unlinks its lock file. A second run could open the file just before that, and a third create a new one just after; both could then lock "the" file. Round 8 re-checks after locking that the file at the path is the one it locked.
- **Deleting a lock file, by hand or by `--fresh`, is never needed.** It would be harmful only while the run that holds it is still alive, such as a hung run not yet killed: a new run would then lock a new file at the same path, and if the old run came back to life, both could write the same checkpoint.

**The watchdog.**
- **When it fires.** If nothing progresses for `--watchdog-mins` minutes (default 30; `0` turns it off), it acts. Progress means a round of chunks, a tuner sample, a table-building launch or a checkpoint write.
- **What it does.** It prints `watchdog: no progress for 30 minutes -- the GPU or its driver seems to hang; exiting with status 3.  The last checkpoint (out52.ckpt) is intact: the same command continues from it.`, removes the lock file and exits with status 3.
- **Why 30 minutes is safe.** A chunk takes a second or so and a table build minutes, so a slow run is not a hang.
- **It writes no checkpoint itself.** best[] lives on the GPU, and a hung driver would not give it back. The work lost is at most the checkpoint interval plus the watchdog's limit.
- **For an unattended run**, e.g. on the T4:

      until ./coarse_sieve 136279841 52 out52 --load-survivors out51.survivors.txt; do
          rc=$?; [ $rc -eq 3 ] || { echo "stopped with status $rc"; break; }
          echo "watchdog restart"; sleep 60
      done

  A hang like your WSL2 one, if it outlasts the process, would make this loop restart again and again, each time losing the work since the last checkpoint, so watch for repeated restarts.
- **The startup line** shows the setting: `...; SIGTERM/SIGINT saves and exits (twice: exit now); watchdog 30 min`.

## 8. Smaller changes

- **Tune labels.** Every tower and RR candidate now carries its loop order: `r5-order`, `r6-order`, `grouped`. In round 7 the round-6 order had no label.
- **The degree line names the order:** `round-5 loop order`, `round-6 loop order` or `grouped loop order`.
- **Values are checked.** `COARSE_PIPE` accepts only 0, 1, 2, and `COARSE_TOWER_TP` only auto, shared, global.
- **The checkpoint-format message** now says "rounds 6-8".

## 9. Validation done here

* **Host, `gf2_dlog_selftest`:** it and `gf2_dlog.h` are unchanged since round 7, where it gave ALL OK.
* **Emulated program, `regress8.sh`: 156 checks, all PASS** (round 7 had 100). That is 83 of its own, plus the 73 of the checkpoint suite below; the unit tests also give ALL OK. Wherever outputs were compared, they are byte-identical. On top of round 7's suite:
  - **Every loop order against the oracles:** 19-23 for tower + PH and tower + RR, the grouped order on 2 emulated GPUs, 24-29, 30-33, and the full range s ≤ floor(r/2) for 2-28 and 29-32. The full range includes heavy progressions.
  - **Fault injection** stops the grouped order too.
  - **Each new switch gives identical output:** `COARSE_LDCG=0`, `COARSE_TOWER_TP=global` (with the grouped order too), and the phase-timing build. Its phase lines appear in the documented format and their shares add up to 100%.
  - **The tuner:** the candidates in their documented order at d = 18-26 (all 3 loop orders, the TP-global twins via a test hook), and every pick tested gives identical results.
  - **The command line:** bad `--watchdog-mins` and `COARSE_PIPE` values refused, and the watchdog shown or off.
  - **Selftest:** the batched kernel in 3 loop orders at d = 21-30 (on your GPU: the grouped order at 40-55).
* **Checkpoints, `ckpt8_tests.sh`: 73 checks, all PASS.** Round 7's K1-K21, plus:
  - K19b: round 7's real checkpoint resumed by round 8, its times carried;
  - K22-K23: a resumed degree keeps the grouped order, or TP-global (the flag checked in the file), without re-tuning;
  - K24: round 7's binary continues both kinds of round-8 checkpoint, with identical results;
  - L1-L3, the lock: a leftover file noted and removed; `--fresh` with one; a hung run that `--watchdog-mins 0` keeps alive, that a first SIGINT can't stop, and that a second kills (status 130) without leaving its lock;
  - W1-W3, the watchdog: status 3 after 2 s without progress, lock removed, checkpoint kept, and the same command continuing to identical results; without a checkpoint, it says the work is lost; a progressing run under a 4 s limit completes.
* **nvcc 13.3, sm_75: warning-free, 106 kernels** (90 + the 16 grouped ones).
  - The 90 round-7 kernels: 81 identical in registers, stack and spills; RR at d = 39, 49 and 53 uses 2-4 registers fewer. Overall: tower 127-128, RR 124-128, PH 132-252 registers, and no spills at all.
  - A correction: round 7's handoff said it had no spills. In fact six RR kernels at d ≤ 31 spilled 4 bytes there. Round 8's don't.
  - The 16 grouped kernels: 128 registers, stack ~1.25 KB (from ~1.7 KB: the prefix-product and gg arrays are gone), 52-68 bytes of spills, some of them in the fill's enumeration loop (section 3).
  - The phase-timing build compiles too, with the small spills of section 6.
* **All of the above ran on the final sources**, the files you have.
* **The dev kit** (section 12), tested as you'd use it:
  - unpacked into an empty directory, built as `setup.sh` does, the oracles checked against their checksums, then `FULL=0 tests/regress.sh`: 152 checks, all PASS, 31 minutes in all;
  - its oracle generator, run in a second copy, made OA, OB, OJ and OBIG, all matching their checksums;
  - the archive you have differs from the one tested only by README wording and the added `tools/sass_locals.py`.
* **What I can't test here:** anything about speed. Every number in sections 3-5 is an estimate or a question for your hardware.

## 10. Hardware test plan

    make coarse_sieve coarse_sieve_prof gf2_dlog_selftest     # ~16 min for the two CUDA builds
    ./gf2_dlog_selftest
    ./coarse_sieve --selftest; ./coarse_sieve --selftest tower; ./coarse_sieve --selftest ph; ./coarse_sieve --selftest rr

At d ≥ 40 the tower and RR lines should now say `batched: N classes (3 loop orders) and M (4 fills), ok`.

1. **The grouped kernels against your files** (~15 min). This is what makes them safe to tune:

       for d in 40 41 42 43 44 45; do
         COARSE_PIPE=2 ./coarse_sieve 136279841 $d g_$d --load-survivors out$((d-1)).survivors.txt --no-checkpoint
         cmp g_$d.found.cert out$d.found.cert && cmp g_$d.survivors.txt out$d.survivors.txt && echo "d=$d grouped identical"
       done

   - Their done lines are also the first pass timings of the grouped order, against round 7's r5-order times (10.3, 26.9, 39.6, 108.4, 150.3, 433.8 s).
   - If anything differs, please send the log, and use `COARSE_PIPE=0` until I've fixed it.
2. **The tuned default** at one tower and one RR degree: `./coarse_sieve 136279841 44 r8_44 --load-survivors out43.survivors.txt`, the same for 45, and `cmp` with your out44/out45. Please send the tune lines: they hold the A/B of the three orders.
3. **Reads past L1, A/B:** the same two runs with `COARSE_LDCG=0` in front. The done lines against step 2's give the effect.
4. **Phase timing** (~10 min). `COARSE_METHOD=rr` matters here: with `--no-tune`, d=47 would otherwise use PH.

       for o in 0 1 2; do
         COARSE_METHOD=rr COARSE_PIPE=$o COARSE_TEST_STOP_AFTER_CHUNKS=3000 ./coarse_sieve_prof 136279841 47 pt47 \
             --load-survivors out46.survivors.txt --no-checkpoint --no-tune
       done
       for o in 0 1 2; do
         COARSE_PIPE=$o COARSE_TEST_STOP_AFTER_CHUNKS=600 ./coarse_sieve_prof 136279841 44 pt44 \
             --load-survivors out43.survivors.txt --no-checkpoint --no-tune
       done

   Please send the six `[phase timing ...]` lines. The first 3,000 (or 600) positions of the scrambled pass are a fair sample of the degree.
5. **The two ncu exports and `nvcc --version`** (section 2).
6. **Optional, the watchdog on hardware** (~1 min). `COARSE_TEST_WATCHDOG_SECS=20 COARSE_TEST_HANG_AFTER_CHUNKS=200 ./coarse_sieve 136279841 44 wd --load-survivors out43.survivors.txt --checkpoint-mins 0.05` should:
   - exit with status 3 after ~20 s without progress (`echo $?`);
   - leave no `wd.ckpt.lock`.

   The same command without the two variables then continues from wd.ckpt.
7. **Production.**
   - **On the T4:** time d=44 from out43 first and `cmp` with your out44 files. That checks the machine and gives its speed factor. Then d=52 with the wrapper of section 7.
   - **What d=52's tuner will compare:** twelve candidates. GF(2^13) in the three orders, each with TP in shared and in global memory; then GF(2^2) and the classic tower, each in the three orders.
   - **On your card,** d=53 can run at the same time (RR; the two-machine merge recipe is in round 7's handoff, section 6).

## 11. Projections

The degrees left, on your RTX 2070 SUPER, calibrated by the ones you've run. None of round 8's experiments is in these numbers; your tune lines will say whether any of them pays. On a T4, multiply by ~1.4-1.8 (estimate; your d=44 timing there will replace it).

| d | engine | classes | time |
|---|---|---|---|
| 44 | tower GF(2^11) | 4.0e11 | 150 s (measured, round 7) |
| 45 | RR | 7.8e11 | 434 s (measured, round 7) |
| 51 | RR | 4.4e13 | ~7.0 h (your run's own estimate, round 7) |
| 52 | tower GF(2^13) | 8.7e13 | 8.5-11 h (est.) |
| 53 | RR | 1.7e14 | ~27 h (est., at d=51's 0.58 ns/class) |
| 54 | tower GF(2^9) | 3.3e14 | 1.7-2.1 days (est.) |
| 55 | RR, 2 GB table | 6.6e14 | ~4.6 days (est., at 0.6 ns/class) |

- **RR** I now scale by the class count alone. Its ns/class barely grew from d=45 to d=51 (0.55 → 0.58), though its table grew 8x.
- **The tower at d = 52 and 54** (T tables of 512 MB and 1 GB) hasn't been timed at such sizes since round 5's d=50, so its rate there is the least certain number in this table. The T4's tune line at d=52 will show it.

## 12. Continuing in a new conversation

`coarse_sieve_devkit_r8.tar.gz` has what a fresh workspace needs to carry on:
- the current sources, and those of rounds 5-7;
- the emulated build and the two suites;
- the seed and the oracles' checksums, with a generator for the oracles from round 5's independent sieves;
- the nvcc check build and the analysis tools, among them `sass_locals.py`, which maps a kernel's local-memory accesses loop by loop (section 3's spill check).

Its README says how to use it: `./setup.sh` (~45 min, mostly one oracle), then `bash tests/regress.sh` (~35 min).

To start round 9 in a new conversation in this project, attach:
- the dev kit;
- this handoff (also the project doc `claude/coarse_sieve_handoff.md`);
- your logs and results, as you have been doing.

## Appendix: earlier rounds, for reference

**Round 7 (2026-10-26).** Full batches (the warps pool the classes they find), the cheaper even-d tie-break, the time of all runs in checkpoints, lockstep warps in the emulator. Your results: pass = tune rate; d=44 150 s, d=45 434 s; the d=51 production run matched the known ranges.

**Round 6 (2026-10-25).** One Zech log per reciprocal pair; the scrambled, stratified pass order and an honest time-left estimate; a quick tuner sample; the checkpoint records the engine; the command-line cleanup. Your results: d=44 268 s, d=47 2,142 s, d=49 8,369 s.

**Round 5 (2026-10-24).** The irregular classes joined the batch inversion (1.97x at d=44). Also RR and degree 49, auto-checkpointing, and tuning. Your results: d=47 (RR) 3,648 s, d=49 14,185 s, d=50 34,627 s.

**Round 4 (2026-10-22).** Subfield tower: one DRAM read per tower log. Your results: d=44 668.65 s.

**Round 3 (2026-10-20).** Batched kernel: FKM enumeration, 32 classes per thread bit-sliced, Montgomery batch inversion, the PH planner, depth 55.

**Round 2 (2026-10-19).** The logs path: one Zech log per class from tables of ~2^(d/2) entries.

**Round 1 (2026-10-18).** LFSR stepping in the bucketed path (~1000x), other fixes.

Sources:
- [NVIDIA Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) (`--page source`, `--print-source`, for section 2's exports)
- [Brent, Larvala, Zimmermann, *A fast algorithm for testing reducibility of trinomials mod 2* (Math. Comp. 2003)](https://maths-people.anu.edu.au/~brent/pd/rpb199.pdf)

# coarse_sieve, round 7: full batches, the time of all runs, a cheaper tie-break

**Baseline: round 6 (2026-10-25), which you validated on hardware.**
- Regressions at d = 40-44 identical; all the certificates (r = 74207281 at 47 and 49, r = 57885161) matched Brent et al.'s data.
- 64 dots at a steady pace; the time-left estimates held (d=46: 1,484 s estimated, 1,565 s actual; d=49: 2.3 h vs 2.32 h).
- Times: d=44 268 s, d=47 (RR) 2,142 s, d=49 8,369 s. The tuner picked round 5's loop order nearly everywhere.

## Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | Round 7. |
| `gf2_dlog.h` | The pair tie-break computed from u's longest runs (same roles); `gf2f_pair_role_ref` keeps round 6's version as the reference. |
| `gf2_dlog_selftest.cpp` | Part 9 also compares the roles with round 6's, on every class to d=22 and on samples to d=55. |
| `cuda_emul.h` | Lockstep warps for the emulated build (opt-in, so the other programs' emulated builds are unchanged). |
| `tools/cstat.cpp`, `tools/fillsim.cpp`, `tools/passavg.py`, `tools/estimate.py` | The analysis behind sections 2-3: class counts per chunk, the lockstep warp simulation, its pass averages, and the estimate. Host programs; build lines in their headers. |
| `coarse_sieve_round7.patch` | All of the above as a patch against round 6 (flat paths). |

`Makefile`, `gf2_wide_field.h` and `gf2_wide_setup.h` are unchanged.

## TL;DR

* **Your Item 3, done.** The checkpoint now also records the time spent on the degree so far and on the run's earlier degrees. A resumed degree's done line gives the time an uninterrupted run would have shown, with this run's own time beside it:

      d=44 done (logs/tower, 4096 chunks): 399822314775 classes, 0 heavy progressions, 268.10s in all (this run 251.89s, resumed at chunk 288 of 4096)
      total sieve time: 252.10s in this run, 268.31s in all

  (Illustrative numbers.) A round-6 checkpoint still resumes, and its done line says that the earlier time isn't known. Section 1.
* **New: the batches are now full.** I simulated the warps of the batched kernel, lane by lane, on real chunks of the pass (section 2). In round 6, **27-38% of the batch slots were empty**: lanes that had run out of work sat idle through their warp's last batches. The enumeration loop ran with only one lane in six to ten doing work. In the tuner's dense sample both are close to ideal, which is why your passes ran 1.2-1.9x slower per class than the tune lines said. Round 7 pools the classes each warp finds and deals them out to its lanes in turn, and the lanes take their runs from a shared counter. **Estimate: 1.6-1.9x faster at the subfield-tower degrees (40, 42, 44, 50, 52, ...), 1.2-1.5x at the RR and classic-tower ones.** That is a model, not a measurement, so `COARSE_FILL=0` restores round 6's fill for an A/B test (section 8).
* **The even-degree tie-break now costs 1-2 rotations instead of d.** It is the same rule, so it gives the same roles: checked on every Lyndon word of length ≤ 36 and on 4.5e10 of length 44. Section 4.
* **The ncu run, explained** (as in my message): the d=47 run hadn't hung; and both profiles caught chunks that hold no classes at all. Section 5 has corrected commands.
* **The T4 question** (as in my message): feasible, with negligible host-device traffic, ~1.4-1.8x slower than your card (estimate). Section 6, plus how to run two degrees on two machines and merge them.

## 1. The time of all runs (your Item 3)

**What the checkpoint records now** (format v4 unchanged; two fields that round 6 left zero, with two flag bits saying they are set):
- the time spent on the current degree so far, in all runs (a double, in the last u64 header slot);
- the time of the run's earlier degrees, in all runs (a float, in the last u32 slot).

**What a run prints.**
- A degree that started in this run prints its time as before: `..., 268.45s`.
- A resumed degree prints the time of one uninterrupted run, then this run's own time:
  `..., 268.10s in all (this run 251.89s, resumed at chunk 288 of 4096)`.
- The total line of a resumed run gives both: `total sieve time: 252.10s in this run, 268.31s in all`. The "in all" figure is the degrees' times summed.
- The "found checkpoint" line also gives the degree's time so far: `(written 2026-10-26 14:46, degree 44, chunk 288 of 4096, 16.21s into it)`.

**What "in all" counts.**
- Each run's time up to its last checkpoint. A stop by signal, or the 15-minute save, writes the checkpoint at a chunk boundary, so nothing between the save and the exit is counted twice.
- If a run dies without saving (power cut, kill -9), its work after the last save is redone by the next run, and only the redone time is counted.
- A resumed run rebuilds the degree's tables, which an uninterrupted run doesn't do. That rebuild is not in "in all", only in "this run". So "in all" is comparable with an uninterrupted run's done line.

**Compatibility.**
- Round 6's binary reads round 7's checkpoints (it ignores the new fields), and round 7 reads round 6's. Both directions are tested, with real checkpoints written by the other binary.
- A degree resumed from a round-6 checkpoint prints `..., 251.89s in this run (resumed at chunk 288 of 4096; the checkpoint, from round 6, has no time for the runs before)`, and the total line then shows this run's time only.
- So a d=51 run started with round 6 can be finished with round 7.

**Also:** a `COARSE_TEST_STOP_AFTER_CHUNKS` stop under `--no-checkpoint` now says `no checkpoint, nothing saved` instead of "run the same command again to continue".

## 2. Why the passes ran slower than the tune lines

**The puzzle, from your round-6 log.** Per class, the real pass was 1.2-1.9x slower than the tuner's sample:

| d | engine | tune, ns/class | pass, ns/class | ratio |
|---|---|---|---|---|
| 40 | tower GF(2^10) | 0.363 | 0.704 | 1.94 |
| 42 | tower GF(2^7) | 0.382 | 0.689 | 1.80 |
| 44 | tower GF(2^11) | 0.384 | 0.671 | 1.75 |
| 46 | tower, classic | 0.830 | 1.039 | 1.25 |
| 41 | RR | 0.494 | 0.662 | 1.34 |
| 43 | RR | 0.553 | 0.687 | 1.24 |
| 45 | RR | 0.582 | 0.726 | 1.25 |
| 47 | RR | 0.586 | 0.715 | 1.22 |
| 49 | RR | 0.603 | 0.728 | 1.21 |

(The pass column is the done line's time over I(d), so it includes the tables and tuning, a second or two.) I first suspected the even-degree tie-break (section 4). It was part of the cost, but too small to explain this.

**What a lane does.**
- Each lane of a warp enumerates its own runs of u (FKM) into its own batch of 32 canonical classes. A batch is then processed in one go: g^u bit-sliced, the logs, one inversion.
- The warp runs in lockstep. A batch starts when the slowest lane has filled its batch, and the warp keeps processing batches until its last lane has run out of runs.

**The simulation** (`tools/fillsim.cpp`).
- It replays the kernel's fill loop exactly: the grid-stride runs, FKM, the pair roles, 32 lanes in lockstep.
- It does this for real chunks of the pass. `tools/cstat.cpp` counted the classes of the first 256-1,024 positions of each pass at d = 40-47, a stratified sample by the design of the pass order. The simulation then ran the non-empty chunks among them, 16 warps each, drawn at random from the launch's 640.
- Pass averages (steps per class credited, idle lanes included):

| d | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | dense sample |
|---|---|---|---|---|---|---|---|---|---|
| batch slots holding a class, round 6 | 66% | 66% | 67% | 62% | 70% | 73% | 73% | 69% | 98-99% |
| enumeration steps per class, round 6 | 13.6 | 10.5 | 13.2 | 11.2 | 12.2 | 7.5 | 9.7 | 8.9 | 1.1-1.2 |
| batch slots holding a class, round 7 | 99.8% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | |
| enumeration steps per class, round 7 | 2.2 | 2.2 | 2.2 | 2.2 | 2.2 | 1.8 | 2.0 | 2.0 | |

(An enumeration step is one iteration of the fill loop for one lane, busy or not; FKM itself needs ~1.2 steps per class. Round 7's rows are for COARSE_FILL=2, pooled with rotated runs, which the simulation models exactly. The default, 3, takes its runs from a counter in an order that depends on timing, which it can't model; it should do at least as well.)

**Why the lanes differ so much.**
- A lane's runs were t, t + stride, t + 2·stride, ..., and the stride is a multiple of 32. So all of lane l's runs share the 5 bits of u just above the run: they equal l.
- Those bits change both how many Lyndon words a run holds (lane 0's 00000 is an internal zero run longer than most leading runs) and how many of them are canonical (lane 31's 11111 adds five ones).
- So some lanes had half the work of others. They finished their 51 runs early and idled through the warp's remaining batches. In the dense sample every lane sees the same kind of words, so the tune never showed this.
- Round 5 had a milder form of it: 81% of the slots used at d=44 in the simulation, where every class was processed. The pairs made it worse, because the canonical share depends on the same lane bits. That is part of why the pairs gave d=44 only 1.27x, against 1.7x at the RR degrees.

**Why it hurts the subfield towers most.** A padded slot costs a full slot of instructions, since the warp issues them for its idle lanes too. But it costs no memory traffic, because idle lanes issue no loads. The subfield tower is mostly instructions (its inner logs come from shared memory), so it lost the most. RR (two DRAM reads per class) and the classic tower (two dependent reads) are more latency-bound, and lost less. That matches the two groups in the table above.

## 3. The fix: pooled batches (`COARSE_FILL`)

**What each lane does now, each iteration** (FILL_POOL, the default):
1. **One enumeration step**, unless it already holds a class waiting for room.
2. **A ballot.** The warp learns which lanes found a canonical class.
3. **The deal.** The found classes go to the lanes in turn, starting where the last deal stopped. Each lane fetches its class from the finder with one shuffle (`warp_select` picks the source lane). So the lanes' batches fill together, and every batch is full except the warp's last.
4. **Runs from a counter.** When a lane needs a new run, it takes the next one from a counter of the launch (one atomicAdd per run of 2048 u values). So lanes, and warps, run out of work together at the end of the chunk.

A class that finds no room stays with its lane for the next batch. The loop's exits are warp-uniform, as the full-mask intrinsics require.

**The other settings,** for A/B tests (any of them gives identical output):

| `COARSE_FILL` | fill | runs |
|---|---|---|
| 0 | round 6: each lane its own batch | t, t + stride, ... |
| 1 | each lane its own batch | rotated: at its k-th run a lane takes position (l + k) mod 32 of its warp's group of 32, so all 32 positions in turn |
| 2 | pooled | rotated, as 1 |
| 3 (default) | pooled | from the counter |

The simulation's pass averages for 1 are 91-93% of the slots used and 8-13 steps per class (rotation evens out the lanes' totals, not their steps); for 2, 99.8-99.9% and 1.8-2.2. The fill is not part of the checkpoint: a run stopped with one fill can be resumed with another (tested). Nothing in the output lines changes but the times.

**What it should buy (estimate).**
- The number of batch calls per class falls by 27-38%, and the enumeration steps by 4-6x.
- **How I estimated** (`tools/estimate.py`): per class, a cost per enumeration step times the steps, plus a cost per batch slot times the slots. The slot cost comes from your tune line (the dense sample, nearly full batches); the step cost then follows from your done line and the simulated counts. The extra work of a pooled step (two ballots, a shuffle, a 5-step bit select: ~40 instructions, ~1.9x an old step) is included.
- **Subfield towers: ~1.6-1.9x.** The model fits d = 40, 42, 44 with step costs of 0.009-0.020 ns. The range depends on whether the tune sample's time was 25% high (see below).
- **RR and the classic tower: ~1.2-1.5x.** Here the model fits only if the tune sample's time was high. Taken as measured, a padded slot must cost these engines much less than a full one, which fits section 2's argument about memory traffic. 1.4-1.6x if the tune was high, less if padding was cheap.
- I can't time any of this here. **The A/B run in section 8 is the real test.**

**Also in the tuner.** The tune sample was cut into runs of 2048 u, about 3.2 runs per lane, so a fifth of the lanes (8 SMs' worth) had 4 runs and the rest 3. The kernel's time was that of those 8 SMs, so the sample's time per class came out up to 25% high for tower and RR, ~10% for PH, whose wave has half the lanes. (Less than 25% for a memory-bound engine, since the 8 SMs run faster once the others are idle.) It now uses runs short enough for ~16 per lane. The ranking hardly changes. But your new tune lines should read up to ~20% below round 6's, and by how much is a check of this paragraph and of the RR estimate above.

**Testing a warp-cooperative kernel without a GPU.**
- The emulator used to run threads one after another, which can't execute a ballot.
- `cuda_emul.h` now runs the 32 lanes of each warp as fibers on the one host thread (a hand-written x86-64 context switch of 15 instructions; other CPUs fall back to ucontext, which I haven't tested). A lane runs until it reaches a warp intrinsic; when all 32 have, the result is computed and they go on.
- It also stops the program if lanes reach different intrinsics, or if a lane named in the mask has exited. So it checks the warp-uniformity that the device needs.
- coarse_sieve opts in with `#define EMUL_LOCKSTEP_WARPS`. Other programs built with `make emul-%` keep the old launcher.

## 4. The even-degree tie-break

- When u has as many ones as zeros, round 6 compared the leading zero run with the longest run of ones. If those were equal, it compared u with the least rotation of ~u, computed by a d-step loop.
- At d=44, 10.9% of the classes tie and 26% of those reach the loop: 2.8% of all classes. In round 6's fill, 4-7% of a warp's enumeration steps had a lane in that 43-step loop (simulated on two median chunks of d=44), and each such step cost the whole warp ~250 instructions.
- The least rotation of ~u starts at one of ~u's longest zero runs, that is, at one of u's longest runs of ones. Those positions are the set bits of u AND (u >> 1) AND ... (z0 terms), so round 7 compares only those rotations. There are 1 or 2 of them in 92% of the cases at d=44.
- The longest-run comparison now uses the same bit trick (log2(z0) shift-ANDs, as the rotation test does) instead of a loop.
- **It is the same rule, so it gives the same roles**: no change to checkpoints or results.
- Checked against round 6's function (`gf2f_pair_role_ref`):
  - every Lyndon word of length 2-36: all 1.9e9 at d=36, 3.9e9 in total (`tools/cstat.cpp`);
  - 4.5e10 Lyndon words of length 44 (512 chunks of the pass) and 9.3e10 of length 47;
  - host selftest part 9 (every class to d=22, stretches to d=55), `t_pair` (d ≤ 24), and the device selftest (2^16 random classes per engine and degree 21-55).
- Its own effect on time is part of section 3's estimate at even d. By the simulation, the old loop added 20-40% to round 6's enumeration loop at d=44, a few percent of the pass (estimate).

## 5. Profiling: what happened at d=47, and commands that profile real work

**The d=47 run had not hung.**
- With `--no-tune`, d=47 used the default engine, PH, instead of the RR the tuner picks (0.82 vs 0.59 ns/class in your log).
- So after the profiled kernel came an ordinary PH pass of about an hour. Its first dot printed `[~64 min to go]`, and you stopped it at 34 of 64 dots.
- The profile itself was complete (`100% - 34 passes`).

**Both profiles caught empty chunks, my fault.**
- I suggested `--launch-skip 20`. In round 6's pass order, position 20 is chunk 3052 of 4096 at d=44 and chunk 12892 of 32768 at d=47.
- `tools/cstat.cpp` finds no class in either. All their words have an inner zero run longer than the leading one, so none is a Lyndon word.
- So the kernels ran ~50 µs of start-up work: copying the tower's tables into shared memory and initialising the batch arrays.
- The dynamic figures in d44.txt and d47.txt (memory, stalls, issue rate) are about that start-up. The static ones hold:
  - round 6's d=44 tower: 128 registers, 50% occupancy;
  - PH at d=47: 222 registers, 25% occupancy;
  - both: a ~1.7 KB stack, which holds the batch arrays.
- Such chunks are common: 68% of the chunks I sampled at d=44 and 73% at d=47 hold no classes.

**Commands for round 7.**
- Each profiles a chunk at the median by work, in the variant the tuner picks, then stops right after it.
- The first pair compares round 6's fill with round 7's on the same chunk. The difference should show in "Avg. Active Threads Per Warp" and in the issue-slot figures.

      COARSE_FILL=0 COARSE_PIPE=0 COARSE_TEST_STOP_AFTER_CHUNKS=124 ncu --set full -k regex:k_logs_batch \
          --launch-skip 123 --launch-count 1 -o d44_fill0 ./coarse_sieve 136279841 44 prof \
          --load-survivors out43.survivors.txt --no-checkpoint --no-tune
      COARSE_PIPE=0 COARSE_TEST_STOP_AFTER_CHUNKS=124 ncu --set full -k regex:k_logs_batch \
          --launch-skip 123 --launch-count 1 -o d44_fill3 ./coarse_sieve 136279841 44 prof \
          --load-survivors out43.survivors.txt --no-checkpoint --no-tune
      COARSE_METHOD=rr COARSE_PIPE=0 COARSE_TEST_STOP_AFTER_CHUNKS=322 ncu --set full -k regex:k_logs_batch \
          --launch-skip 321 --launch-count 1 -o d47_rr ./coarse_sieve 136279841 47 prof \
          --load-survivors out46.survivors.txt --no-checkpoint --no-tune
      ncu --import d44_fill0.ncu-rep --page details > d44_fill0.txt     # and the same for the others

- Position 123 at d=44 is chunk 131: 852M classes, 9.8% of them ties (the average is 10.9%). Position 321 at d=47 is chunk 1361: 796M classes.
- `--no-tune` makes the kernel launches the pass positions exactly. `COARSE_PIPE=0` is the loop order the tuner chose on your card.
- The profiled kernel should take about half a second per replay (estimate).

## 6. T4s, clouds, and two degrees at once

(As in my message, for the record.)
- **Host-device traffic is negligible.**
  - The GPU enumerates, computes the logs from tables in its own memory and marks best[] there.
  - Per chunk, the host launches a kernel and reads back ~30 bytes of counters, ~0.1 ms. That is about a minute over d=51's 524,288 chunks.
  - best[] (545 MB) goes up once and comes back only at the checkpoints (every 15 min) and at the end.
- **A T4 is your card's chip** (TU104, 40 SMs, sm_75, the same build) with 16 GB, capped at 70 W. Measured on Google Cloud, it sustained ~1.24-1.27 GHz once warm, against your card's ~1.8 GHz. **Expect ~1.4-1.8x slower** (estimate).
  - Before a long run, time d=44 from out43 on it (4.5 min on your card with round 6) and compare with your out44 files. That gives the real factor and checks the machine.
- **Spot or preemptible instances are fine.**
  - Checkpoints come every 15 min, and SIGTERM saves within seconds.
  - To resume on another machine, bring the checkpoint, the seed file and the same command line. The checkpoint checks the seed's path and contents. Keep them on a persistent disk.
- **Two degrees on two machines, from the same seed:** e.g. d=51 on your card and d=52 on a T4, both with `--load-survivors out50.survivors.txt`. Then merge:

      comm -12 <(sort out51.survivors.txt) <(sort out52.survivors.txt) | sort -n > out52m.survivors.txt
      awk 'NR==FNR {f[$1]; next} !($1 in f)' out51.found.cert out52.found.cert > out52m.found.cert

  - The merged files are exactly what d=52 seeded from d=51's survivors would give. I checked the recipe on small degrees: d=20 and 21 each run from d=19's survivors and merged gave files identical to d=21 run after d=20.
  - The d=52 run costs the same either way, since the work doesn't depend on how many s are still open.

## 7. Validation done here

* **Host, `gf2_dlog_selftest`: ALL OK.**
  - Part 9 now also compares the roles with round 6's: every class to d=22, and 6.2M classes on stretches at d = 23-55 (24,058 of them through the run scan).
* **The tie-break against round 6's, separately** (`tools/cstat.cpp`): no difference on any Lyndon word of length 2-36 (3.9e9 words), nor on the 4.5e10 of 512 sampled chunks at d=44 or the 9.3e10 of 1,024 at d=47.
* **Unit tests:** arithmetic, rotation, bit-sliced arithmetic, tower, and `t_pair` (now also against round 6's roles).
* **Emulated program, `regress7.sh`: 100 checks, all PASS (round 6 had 77).** Wherever outputs were compared they are byte-identical. With the pooled fill as the default:
  - **Round 6's suite, rerun:** round 5's dense and bucketed oracles, both loop orders, 2 emulated GPUs, fault injection, every subfield, every tuner pick, the command line, and checkpoints K1-K14.
  - **Each fill:** `COARSE_FILL` 0, 1, 2 at 19-23 (tower + PH, tower + RR, 2 GPUs) and at 24-29 (0, and 1 with RR), all identical to the oracles; `COARSE_FILL=4` refused.
  - **Selftest:** every engine and degree checks the batched kernel with all four fills on a stretch of ordinary density (here d = 21-30; on your GPU 21-55).
  - **Checkpoints K15-K21:**
    - the time display after a resume: its format, and the exact carry-over of times patched to 1,000 s and 500 s;
    - a checkpoint without times, and a real one written by round 6's binary;
    - round 6's binary continuing a round-7 checkpoint;
    - a degree-boundary checkpoint;
    - the `--no-checkpoint` stop message;
    - a change of fill between stop and resume.
  - Every intrinsic call in all of these went through the emulator's warp-uniformity check without a violation.
* **nvcc 13.3, sm_75:** warning-free, 90 kernels.
  - Registers: tower 127-128 and RR 126-128 (the limit for 512-thread blocks, as before); PH 132-252 (still 2 blocks of 128 per SM).
  - No spills at all. Round 6 had one 8-byte spill in the tower kernels at d ≥ 34. Stack ~1.7 KB, as before.
  - In the SASS, the fill loop has its two `VOTE.ANY` and a 64-bit `SHFL.IDX`, and the counter's atomicAdd became a warp-aggregated atomic.
* All of the above were rerun on the final sources, the files you have.
* **What the emulator cannot tell me:** GPU times. Every time in sections 3 and 9 is an estimate.

## 8. Hardware test plan

    make coarse_sieve gf2_dlog_selftest
    ./gf2_dlog_selftest                    # "ALL OK"; part 9 now compares the roles with round 6's
    ./coarse_sieve --selftest              # each engine line now ends "... and N (4 fills), ok"
    ./coarse_sieve --selftest tower
    ./coarse_sieve --selftest ph
    ./coarse_sieve --selftest rr

1. **The A/B timing of the fill: the measurement that matters** (~25 minutes). One tower degree and one RR degree, round 6's fill against round 7's, on the same seeds as your outNN files:

       for f in 0 3; do
         COARSE_FILL=$f ./coarse_sieve 136279841 44 f${f}_44 --load-survivors out43.survivors.txt --no-checkpoint
         COARSE_FILL=$f ./coarse_sieve 136279841 45 f${f}_45 --load-survivors out44.survivors.txt --no-checkpoint
       done
       for d in 44 45; do for f in 0 3; do
         cmp f${f}_$d.found.cert out$d.found.cert && cmp f${f}_$d.survivors.txt out$d.survivors.txt && echo "d=$d fill $f identical"
       done; done

   - Please send the done lines and tune lines.
   - `COARSE_FILL=0` should reproduce round 6's times (268 s and 568 s), give or take the cheaper tie-break at d=44 and the tuner's new sample.
   - The default should land near the estimates of section 3. If you have time, `COARSE_FILL=1` and `2` at d=44 separate the effects of the rotated runs, the pooling and the counter.
2. **Regression against your round-6 files**, as last time: `for d in 40 41 42 43 44; do ./coarse_sieve 136279841 $d r7_$d --load-survivors out$((d-1)).survivors.txt; cmp ...; done`.
3. **The time of all runs (your Item 3).**
   - Start d=44, press Ctrl-C after a few dots, and run the same command again.
   - The found-checkpoint line should say `..., N.NNs into it`. The done line should give `X s in all (this run Y s, resumed at chunk N of 4096)`, with X close to step 2's uninterrupted d=44 time. The total line should say `... in this run, ... in all`.
   - Files identical to step 2's.
4. **Optional: the profiles of section 5**, d44_fill0 against d44_fill3, and d47_rr.
5. **Production.** d=51 if it isn't running already, then d=52. A d=51 started with round 6 can be finished with round 7. Two machines can take two degrees at once (section 6).

## 9. Projections

RTX 2070 SUPER. The round-6 column is your times where you have them (round 6's estimates otherwise, marked ~). The round-7 column applies section 3's factors: 1.6-1.9x for the subfield towers, 1.2-1.5x for RR and the classic tower. Everything but your times is an estimate. On a T4, multiply by ~1.4-1.8.

| d | engine | classes | round 6 | round 7 (est.) |
|---|---|---|---|---|
| 44 | tower GF(2^11) | 4.0e11 | 268 s | 140-170 s |
| 45 | RR | 7.8e11 | 568 s | 380-470 s |
| 46 | tower, classic | 1.5e12 | 1,589 s | 1,060-1,320 s |
| 47 | RR | 3.0e12 | 2,142 s | 1,430-1,790 s |
| 49 | RR | 1.15e13 | 8,369 s | 1.55-1.95 h |
| 50 | tower GF(2^5) | 2.25e13 | ~6.7 h (round 5: 9.6 h) | 3.5-4.2 h |
| 51 | RR | 4.4e13 | ~9-10 h | 6-8.5 h |
| 52 | tower GF(2^13) | 8.7e13 | ~16-18 h | 8.5-11 h |
| 53 | RR | 1.7e14 | ~35 h | 23-29 h |
| 54 | tower GF(2^9) | 3.3e14 | ~3.3 days | 1.7-2.1 days |
| 55 | RR, 2 GB table | 6.6e14 | ~6 days | 4-5 days |

**Where a next gain might come from.**
- **The profiles of section 5.** The RR and classic-tower degrees gained least from the pairs and will gain least from the fill (by the argument at the end of section 2), and the cause is probably memory latency. The d47_rr profile would show it as long-scoreboard stalls. Then the two RR reads of each class might be issued for several classes at once.
- **The batch arrays** (~1.7 KB of local memory per thread, section 9 of round 6's handoff) still go to L2 and DRAM. With full batches each array access now serves a real class, but the traffic is the same per class.
- **The enumeration's seeks.** Every run of 2048 u starts with a 30-45-step fkm_seek, about one per 1,000 classes. That is small now; it was hidden by the idle lanes before.

**On the economics** (as before): each degree removes ~1/d of the survivors (~2%) and costs about twice the previous one. Going from 51 to 55 would take ~7-9 days of your card with round 7 (estimate), or a fraction of that spread over rented T4s with the two-degree recipe of section 6.

## Appendix: earlier rounds, for reference

**Round 6 (2026-10-25).** One Zech log per reciprocal pair; the scrambled, stratified pass order and an honest time-left estimate; a quick tuner sample; the checkpoint records the engine; the command-line cleanup. Your results: regressions identical, certificates matched, d=44 268 s, d=47 2,142 s, d=49 8,369 s; the estimates held.

**Round 5 (2026-10-24).** The irregular classes joined the batch inversion (1.97x at d=44). Also RR and degree 49, auto-checkpointing, and tuning. Your results: d=47 (RR) 3,648 s, d=49 14,185 s, d=50 34,627 s.

**Round 4 (2026-10-22).** Subfield tower: one DRAM read per tower log. Your results: d=44 668.65 s.

**Round 3 (2026-10-20).** Batched kernel: FKM enumeration, 32 classes per thread bit-sliced, Montgomery batch inversion, the PH planner, depth 55.

**Round 2 (2026-10-19).** The logs path: one Zech log per class from tables of ~2^(d/2) entries.

**Round 1 (2026-10-18).** LFSR stepping in the bucketed path (~1000x), other fixes.

Sources:
- [Brent, Larvala, Zimmermann, *A fast algorithm for testing reducibility of trinomials mod 2* (Math. Comp. 2003)](https://maths-people.anu.edu.au/~brent/pd/rpb199.pdf)
- [*Don't Assume Your Cloud T4 Runs at Spec* (sustained T4 clocks on Google Cloud), for section 6](https://www.neura.market/directories/deepseek/blog/devto-4863183)
- [ServeTheHome, *NVIDIA Tesla T4 AI Inferencing GPU Benchmarks and Review* (74 W at full load)](https://www.servethehome.com/nvidia-tesla-t4-ai-inferencing-gpu-review/5)

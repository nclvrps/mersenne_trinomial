# coarse_sieve, round 9: less local-memory traffic per class, and a cubic tower for d=54

**Baseline: round 8 (2026-10-28), which you validated on hardware.**
- All selftests passed. The grouped kernels gave identical outputs at d = 40-45, and the T4 ran d=52 for $1.98.
- As predicted, the grouped order lost at the tower degrees (+12-35%) and about tied at RR (−3 to +2%). The tuner picked it at d=45 and d=53.
- `COARSE_LDCG=0` made no difference (149.95 vs 150.09 s at d=44, 430.54 vs 430.18 s at d=45).

## Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | Round 9. |
| `gf2_dlog.h` | The cubic tower; the exponent-plane prefetch in g^u. |
| `t_cubic.cpp` | New unit test: the cubic tower against an independent host Pohlig-Hellman. |
| `cuda_emul.h` | `cudaEventQuery` for the emulated build. |
| `Makefile` | A comment only. |
| `randread.cu` | New microbenchmark (dev kit `tools/`): your GPU's random-read rate, alone and beside streaming traffic. |
| `coarse_sieve_round9.patch` | All of the above as a patch against round 8 (flat paths). |
| `coarse_sieve_devkit_r9.tar.gz` | The dev kit for round 10 (section 10). |

`gf2_dlog_selftest.cpp`, `gf2_wide_field.h`, `gf2_wide_setup.h` and the other unit tests are unchanged.

## TL;DR

* **Your round-8 data, read** (section 1). The phase timing shows where the grouped order won and lost.
  - It cut g^u's time by 63% and marking's by 59%, with the same code and work. Less local-memory traffic made every wait shorter.
  - It gave all of that back in the fill (spilled state reloaded at every enumeration step) and in the logs (eight copies of the code, so instruction-cache misses).
* **The SASS exports** (section 2). At d=47, three places in the code take 65% of RR's stall samples:
  - the two RR table reads, 24%;
  - one local load in g^u, 24%;
  - the backward loop's loads, 17%.

  At d=44, add 23% for instruction-cache misses.
* **Round 9 keeps what won and removes what lost** (section 3):
  1. **g^u's exponent planes** are loaded three steps ahead (round 8: ~25 instructions), with L1 prefetches two groups ahead.
  2. **A new loop order, `r9-order`**, replaces round 6's: 9 local accesses per class instead of round 5's 12.
  3. **The grouped order, rewritten:** rolled loops (one copy of the code, not eight), and the fill's state parked across each batch. The RR grouped kernels are now spill-free, except at d=45 and 51.
  4. **L1 prefetches of the batch arrays**, on by default; `COARSE_PREFETCH=0` turns them off for the A/B.
* **A cubic tower for d=54** (section 4).
  - It treats GF(2^54) as a cubic extension of GF(2^18).
  - Per class: four logs in GF(2^18) from 26 KB of shared tables, and two reads of a 2 MB table that L2 can hold. The quadratic tower instead makes a DRAM read of its 1 GB T.
  - A tuner candidate at d = 42, 48 and 54.
* **None of this has run on a GPU.**
  - The tuner times every engine and loop order; the A/Bs of section 8 measure the rest.
  - My guess, and no more than that: a few % at RR, more at d=54 if the cubic tower's table stays in L2. Each change could also lose.
* **Your questions** (section 6):
  - **WSL2's 100% CPU** is real CPU time, but WSL2's doing, not the sieve's. It costs no GPU speed. `COARSE_SYNC=poll` is there to test it.
  - **ncu on the T4:** not needed. To see whether its pass is throttled, log `nvidia-smi` instead.
  - **Files sent while I work** reach me at my next step.
  - **Multi-GPU** still works with round 9. It adds throughput only, and a mixed pair runs at the slower card's pace.
* **d=51, reconstructed:** ≈25,772 s, i.e. 0.584 ns/class against the tune line's 0.580 (section 1).

## 1. Round 8 on your cards

**The grouped order's passes**, against round 7's (round 5's order):

| d | engine | round 7, r5-order | round 8, grouped | ratio |
|---|---|---|---|---|
| 40 | tower GF(2^10) | 10.3 s | 13.39 s | 1.30 |
| 41 | RR | 26.9 s | 26.12 s | 0.97 |
| 42 | tower GF(2^7) | 39.6 s | 53.56 s | 1.35 |
| 43 | RR | 108.4 s | 110.03 s | 1.02 |
| 44 | tower GF(2^11) | 150.3 s | 168.01 s | 1.12 |
| 45 | RR | 433.8 s | 428.06 s | 0.99 |

**Tuned:** d=44 chose GF(2^11) r5-order at 0.377 ns/class (150.09 s); d=45 chose RR grouped at 0.534 ns/class (430.18 s, 1% under round 7).

**The phase timing** (warp-cycles per log and lane; "other" ~0 throughout):

| d=47, RR | fill | g^u | logs | inversion | marking | total |
|---|---|---|---|---|---|---|
| r5-order | 5,097 | 11,344 | 17,306 | 370 | 9,441 | 43,558 |
| r6-order | | | | | | 43,209 |
| grouped | 10,318 | 4,249 | 22,509 | 2,583 | 3,876 | 43,535 |

At d=44 (tower) the totals were 28,198 (r5), 29,031 (r6) and 31,941 (grouped).

**What it shows.** The grouped order did what it was meant to do, then lost it again:
- **g^u fell 63%.** That code is identical in both orders, so this is not less work. It is shorter waits: the grouped kernel makes fewer local-memory accesses, so its exponent-plane loads come back sooner.
- **Marking fell 59%:** its values came from registers instead of local memory.
- **That −12,700 was lost again:**
  - the fill, +5,200: the spilled enumeration state, reloaded at every step (round 8's section 3 named this risk);
  - the logs, +5,200: eight copies of a class's code, too big for the instruction cache;
  - the inversions, +2,200: four per batch instead of one.

  This is the strongest evidence yet that local-memory traffic is what the kernel waits on. Round 9 is built on it (section 3).

**The T4.**
- **d=44:** 260.59 s, i.e. 0.652 ns/class, 1.74x the 2070 SUPER's 0.375.
- **d=52:** 61,535 s, i.e. 0.711 ns/class, against its tune line's 0.577: the pass ran 1.23x slower than the tune.
  - On your 2070 SUPER the two agree within 1%: d=51 1.006, d = 40-45 within 1-9% (round 8, section 1).
  - My guess: the T4, a 70 W passively cooled card, clocks down under hours of load, where the tuner's seconds-long samples see its boost clock. Section 6 says how to check.
- **$1.98 for d=52** against ~$14 with `factor`: very nice.

**d=51, reconstructed.** Your d=49 calibration puts ~2.7 s outside the done line: wall clock 14,187.5 s, done line 14,184.78 s. That time goes to reading the seed and setting up. Applied to d=51, the line would have read about:

    d=51 done (logs/RR, 524288 chunks): 44152937520670 classes, ? heavy progressions, 25772s

- **The time:** 25,775.05 − 2.7 ≈ 25,772 s, ±1 s for `date`'s truncation.
- **The rate:** I(51) = 44,152,937,520,670 classes, so 0.5837 ns/class, against the tune's 0.580: the pass ran at the tuner's rate (1.006).
- **The heavy count** can't be reconstructed. Unlike at d=49, degree-51 classes can have small moduli (orders like 103 · 2143), so it need not be 0.

## 2. What the SASS exports show

Your two exports carry every SASS instruction of round 7's kernels with its stall samples. I located the hot instructions by reading the SASS against round 7's source: its loops, loads and table reads.

**RR, d=47.** Three places take 65% of all stall samples:
- the two RR table reads, 24%;
- one local load in the bit-sliced g^u loop, 24%. It fetches the exponent plane `U[b]` ~25 instructions before its use. Earlier rounds assumed it hit L1; it doesn't.
- the backward (marking) loop's local loads, 17%.

**Tower, d=44.** The same pattern:
- the T read, 17%;
- `U[b]`, 14%;
- the backward loop, 12%;
- instruction-cache misses (`no_instruction`), another 23%. The tower's per-class loops are ~20 KB of hot code.

**The memory system is congested:**
- local loads take 2,000-9,000 cycles to return;
- local memory is ~70-80% of the DRAM bytes, and DRAM runs at 46-55% of peak;
- the random table reads arrive at ~1.75-1.9 G/s (RR, classic tower), 1.33 G/s for the subfield tower at d=44.

**The conclusion.**
- Every load waits behind the batch arrays' traffic.
- Hiding latency gained nothing: round 6's order read the tables a class ahead.
- Cutting traffic gained a lot: the grouped order's g^u and marking.

`randread.cu` (dev kit `tools/`) measures your card's ceiling for random 8-byte reads, alone and beside streaming traffic shaped like local memory. With it, section 8's numbers can be read against the hardware's limits.

## 3. Less local-memory traffic, and shorter waits

### 3.1 g^u's exponent planes (all orders)

- **The loads.** Four registers take the planes in turn, each reloaded right after its use. In the source that is three steps ahead.
  - In the SASS, ptxas sinks the four loads to the end of the loop body. At d=53 that leaves 55, 163, 273 and 410 instructions between a plane's load and its use. Round 8 had ~25.
- **Plus L1 prefetches** (`CCTLL.PF1`: no register, nothing waits for it) of the planes two groups ahead.
  - ptxas sinks those too, next to the loads. That still puts each line ~550 instructions ahead of its load, which then hits L1, if the line survives there.

### 3.2 Round 9's loop order (`r9-order`, `COARSE_PIPE=1`)

**What changed.** Round 9's order replaces round 6's, which never won a tune.
- **Forward step:** it stores w = v · t_k, the class's log times the product of the J before it, where round 5's stores v and t_k separately.
- **Backward step:** it then needs only w and J: S = inv · w = v · J_k⁻¹.
- **Cost:** one product moves from the backward loop to the forward one.

**Local-memory accesses per class**, 9 against round 5's 12:
- the fill: one store;
- the bit-slice transposes: one load and one store;
- the forward loop: two loads (x, u) and two stores (w, J);
- the backward loop: two loads.

**The irregular classes' gg:** carried with every class where irregular classes are common (even d, or 3 | d); elsewhere loaded only for an irregular class.

**A new check in every order:** at the end of each backward pass inv · ΠJ must be 1. A wrong inversion now stops the run like a wrong log.

### 3.3 The grouped order, rewritten (`grouped`, `COARSE_PIPE=2`)

- **Rolled loops.** Round 8 unrolled each group's loops, so that the group's values were indexed by constants and lived in registers. That made eight copies of a class's code, which the instruction cache couldn't hold.
  - Round 9 keeps the group's w, J (and gg) in registers but rolls the loops. A class writes its values to, and reads them back from, the right registers through a switch on its place in the group. That costs a few instructions per class and keeps one copy of the code.
  - Per class: 5 local accesses (u: 3, x: 2).
- **The fill's state, parked.** The grouped batch needs the registers the enumeration's state occupies. Round 8's ptxas spilled that state inside the fill loop, reloading it at every step (the +5,200 of section 1).
  - Round 9 stores it to local memory before each batch and loads it back after: 8 accesses per 32 classes.
  - ptxas -v, round 8 → round 9:

    | grouped kernels | round 8 | round 9 |
    |---|---|---|
    | RR, d = 41-55 except 45 and 51 | 52 B | **0** |
    | RR, d = 45 and 51 (they carry gg) | 52 B | 264 B stored, 116 loaded |
    | tower, d = 40-54 | 68 B | 188 B stored, 116 loaded |
    | cubic, d = 42, 48, 54 | — | 188 B stored, 116 loaded |

  - In the tower and cubic kernels, 4 of the 16 group values still spill: about one 8-byte store and one load per class, outside the fill loop. Before the parking these kernels spilled 414 B. So at the tower degrees the grouped order now makes about as many local accesses per class as round 9's order; at RR it makes 5 against 9.
- **Its four inversions per batch** stay. In round 8's d=47 profile they took 2,583 warp-cycles per log and lane, against 370 for round 5's one.

### 3.4 Prefetches of the batch arrays (`COARSE_PREFETCH`, default on)

- **Where:**
  - the forward loops prefetch the class two ahead, one class (~700-1,400 instructions) before its loads;
  - the backward loops, four ahead, ~1,200 instructions before its loads: their values were written a whole forward pass earlier and are back in L2 or DRAM.
- **The risk:** a line prefetched into L1 can be evicted again before its load, wasting the fetch. `COARSE_PREFETCH=0` turns off these and section 3.1's for the A/B.

## 4. The cubic tower (d = 54)

**The idea.**
- GF(2^54) = K(w), with K = GF(2^18) and w³ = y′ ∈ K.
- Write x = Y0 + Y1·w + Y2·w², a linear split by byte tables, and t = Y1 + Y2·w. Then r = t·x = (Y0Y1 + y′Y2²) + (Y1² + Y0Y2)·w: the w² terms cancel.
- So x = r/t with r and t both *linear* in w. It is RR's rational reconstruction, over K instead of GF(2), and it never fails.
- A linear a + b·w is b·(w + a/b). Its log is log b, a log in K, plus T3[log(a/b)], where T3 holds the logs of w + c for all c in K: 2^18 entries, 2 MB.
- The logs in K come from the classic tower of K: tables of 2^9 entries, in shared memory.

**Per class:**
- seven byte-table lookups and two products in K;
- four logs in K, six shared-memory lookups each;
- one product mod 2^18 − 1;
- two reads of T3.

The quadratic tower at d=54 instead makes one DRAM read of its 1 GB T per class (plus the subfield's logs). The cubic tower makes no DRAM-random read at all, *if* L2 keeps T3. L2 is 4 MB on both your cards, but the batch arrays stream through it.

**The tables.**
- T3 is built by Pohlig-Hellman on GPU 0 at the start of the degree: 2^18 logs, under a second by my estimate, each verified on the device and a sample on the host. The line is `[cubic tower for d=54 over GF(2^18): T3 of 262144 logs (2 MB) by PH in Xs, verified]`.
- Shared memory: 26 KB.

**Code size.**
- My first version had a 5,440-instruction forward loop: four inlined general tower logs, each with the subfield tower's loops. It would have thrashed the instruction cache.
- The final one fixes d and the field's modulus at compile time and keeps only the classic path. Its forward loop is 1,236 instructions (tower 1,371, RR 727).
- It also indexes T3 by log_y instead of log_kappa, which saves three products per class.

**In the tuner.**
- At d = 42, 48 and 54 the tuner times it in all three orders: `cubic GF(2^18) r5-order`, etc.
- `COARSE_METHOD=cubic` forces it.

**The checkpoint** records it as engine 4.
- Round 9 resumes such a degree.
- Rounds 6-8 refuse a checkpoint written in the middle of a cubic degree, and leave it alone. Round 8's message is misleading: "the checkpoint's engine is not possible here (less free VRAM?)".

**Verification.**
- t_cubic: every x at d = 12 and 18, and 3,000 random x at each 6 | d up to 54. Each log is checked against an independent host Pohlig-Hellman, and the sieve's compile-time version against the run-time one.
- The emulated sieve against the oracles at d = 12, 18, 24 and 30, in all orders and fills.
- The emulated selftest, with its degree limit raised to 54, runs the batched kernel at every 6 | d from 24 to 54, in the three orders and the four fills, T3 included (built in 1.9 s on the CPU at d=54): all ok.
- On your GPU, the selftest does that by default (section 8).

**What I can't tell:**
- whether T3 stays in L2;
- what the shared-memory bank conflicts of 31 lookups per class cost;
- whether ~400 instructions per class are cheaper than the tower's DRAM read.

The tune line at d=54 will say.

## 5. Smaller changes

- **Selftests:** `--selftest cubic`; the default selftest also checks the cubic tower beside the tower at every 6 | d.
- **`COARSE_SYNC=poll`:** waits for each chunk by polling its event every millisecond (~0.1% of a chunk) instead of sleeping in the driver. It is for WSL2 (section 6).
- **Labels:**
  - `r9-order`, and `round-9 loop order` on the degree line;
  - `COARSE_PIPE=1` now means round 9's order. Round 8's checkpoints of `r6-order` degrees resume in it, with the same results.
- **The checkpoint-format message** says "rounds 6-9".
- **The new environment switches are checked:** `COARSE_SYNC` takes only block or poll, and `COARSE_METHOD` now accepts `cubic`.
- **A bug fixed before it shipped:** the cubic tower's check path read `T3[ir]` in the same expression as the call that sets `ir`. nvcc warned, and the order is unspecified in C++. It is fixed.

## 6. Your questions

**(a) No CPU on the T4, ~100% on WSL2: an artifact?**
- **It is real CPU time, but WSL2's doing.** Both runs wait for each chunk on an event created with `cudaEventBlockingSync`. On native Linux (the T4) the host thread sleeps in the driver until the GPU signals; under WSL2 the same wait evidently spins.
  - The GPU's work is the same either way, so it costs no speed: your 2070 SUPER's passes run at the tuner's rate.
  - What it does cost: one busy core for days, and its power.
- **NVIDIA's WSL user guide** lists its known limitations (unified memory, pinned memory, NVML queries), but nothing about blocking waits. So this is my inference, not documented behaviour.
- **To test it:** run d=44 with `COARSE_SYNC=poll` and watch `top`. If the CPU drops to a few % and the done line's time doesn't move, polling is the better mode under WSL2, and I'll make it that platform's default.
- **ncu on the T4** wouldn't show this. ncu profiles the GPU kernels, which are the same Turing kernels as on your 2070 SUPER.
- **For the T4's pass/tune gap** (section 1), log its clocks during a run:

      nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 60 > t4_clocks.csv &

  A falling `clocks.sm`, with a power or thermal reason, would confirm it.

**(b) Sending files while I work.**
- **It is possible.** In a session like this one, a message you send while I'm working is queued and handed to me at my next step, between tool calls. It doesn't stop the work, and I can use what it brings for the rest of the round.
- **What I'd suggest:**
  - Put whatever could change the plan in the first message, as you did this round; I make the big decisions early.
  - Send mid-work anything I've asked for, or a correction, with a line on what it is. If I'm inside a long step, such as a 45-minute test run, it waits until that step ends.
- In round 8 the exports were requested in the handoff, so this round's kickoff was exactly the right time.

**(c) Multi-GPU (`--gpus N`).**
- **How it works.**
  - Each GPU holds its own best[] (545 MB) and its own copy of the degree's tables, built once and copied.
  - The pass deals the chunks in rounds of N, neighbouring chunks in the scrambled order, and waits for the round's slowest GPU.
  - The tuner runs on GPU 0 only, and its choice applies to all.
  - At the end, and for each checkpoint, the copies of best[] are merged by minimum on the host. Since that merged copy is what's saved, a run can resume on a different number of GPUs.
- **Benefits:** near-linear throughput for identical GPUs, with no peer access needed (any PCIe topology, cloud VMs with 2-8 T4s).
- **Drawbacks:**
  - **A mixed pair runs at the slower card's pace.** A 2070 SUPER with a T4 would do ~2 chunks per T4 chunk-time, ~1.15x a 2070 SUPER alone, where dynamic dealing could give ~1.57x.
  - **VRAM is not pooled:** every GPU needs best[] plus the full tables, ~3 GB at d=55.
  - **One tune for all GPUs:** a different model might prefer another variant.
- **Round 9 compatibility:** yes.
  - Everything round 9 adds is per device: the parking, the prefetch flag, the cubic tower's tables (uploaded to each GPU), the run counter.
  - The emulated 2-GPU runs match the oracles for every engine and order: the cubic tower, RR grouped, the fills. A checkpoint written on 1 GPU resumes on 2 and the other way around.
- **What the emulator can't test:** real concurrency, i.e. that the per-device launches overlap.
- **For your setup** (two machines, one GPU each), separate degrees per machine with the merge recipe of round 7's handoff, section 6, remain the better use.

**(d) Missing information, and artifacts.**
- **I had what I needed:** the dev kit rebuilt the workspace exactly, and the d=51 notes were enough to reconstruct the line.
- **What would help next time:**
  - the full d=53 tune line (all candidates);
  - its done line when it finishes;
  - `nvidia-smi` logs from the T4.
- **Artifacts:** the project doc and the dev kit are the right carriers between rounds. A per-degree run log as a project doc, with each degree's command, tune line, done line and dates, would replace the scrollback that lost d=51. I can start one in round 10 if you'd like.

## 7. Validation done here

* **Unit tests:** `t_arith`, `t_rot`, `t_bs`, `t_tower`, `t_pair` ALL OK. New, `t_cubic`: ALL OK, 285,838 logs, 0 wrong, and the compile-time and run-time versions agree on every one.
* **Emulated program, `tests/regress.sh`: 207 checks, all PASS** (round 8 had 156): 120 of its own, plus the 87 of the checkpoint suite. Wherever outputs were compared, they are byte-identical. On top of round 8's suite:
  - **The cubic tower against the oracles** wherever a degree has 6 | d:
    - scratch 2-22 (d = 12, 18), in all three orders and with fill 1;
    - 24-29 (d=24), in all three orders, with fill 0, and on 2 emulated GPUs;
    - 30-33 (d=30);
    - the full range s ≤ floor(r/2), 2-28 and 29-32.

    At d=24 it equals every inner subfield of the quadratic tower. The phase-timing build and `COARSE_LDCG=0` give the same results with it, and fault injection stops it.
  - **Round 9's order** in place of round 6's: 19-23 for tower + PH and tower + RR, 24-29 for RR, and fault injection.
  - **The grouped order** with every fill (the parked state), and the new switches: `COARSE_PREFETCH=0`; `COARSE_SYNC=poll` (bad values refused); `COARSE_METHOD=cubic` at degrees without 6 | d, which keeps the usual engines.
  - **The tuner:**
    - the candidates in their documented order, the cubic tower's included: 9 at d=18, 12 at d=24;
    - the picks that choose it, with identical results;
    - `COARSE_PIPE`, `COARSE_TOWER_SUBN` and `COARSE_METHOD=cubic` narrowing the list.
  - **Selftests:** `--selftest cubic`, and the default one checking the cubic tower beside the tower.
  - **Checkpoints, 87 checks** (round 8: 73):
    - K25: a cubic degree, forced or tuned, stopped and resumed by round 9, and refused by round 8 with the file left alone;
    - K26-K27: round 8's checkpoints under round 9 (its r6-order as round 9's order, its grouped order as round 9's) and round 9's under round 8;
    - K21 and K24 now resume round-9 checkpoints under rounds 6 and 7.
  - The full run took 54 minutes, with another job on the second core.
* **Added after that run:** the cubic selftest with its degree limit at 54 (section 4), as one more check in `regress.sh`. It passes, run on its own (7 s) here and in the kit test below.
* **The dev kit** (section 10), tested as you'd use it:
  - unpacked into an empty directory, `FULL=0 ./setup.sh` built every binary and made OA, OB, OJ and OBIG, all matching their checksums (31 minutes);
  - then `FULL=0 tests/regress.sh`: 201 checks (the full run's 207 without the six full-range ones), all PASS, in 38 minutes.
* **nvcc 13.3.73 (your version, from pip), sm_75: warning-free, 116 kernels** (round 8's 106, 9 cubic ones, the T3 builder). The tower, RR and cubic kernels use ≤ 128 registers; the PH ones 140-254, as before.
  - Non-grouped kernels: no spills.
  - Grouped kernels: as in section 3.3's table.
  - The phase-timing build compiles warning-free too. Its non-grouped kernels have no spills either (round 8's spilled ~40 B in the tower kernels); its grouped ones spill 12-368 B.
* **What I can't test here:** anything about speed. Every number in sections 3-4 that isn't a count of instructions or bytes is an estimate or a question for your hardware.

## 8. Hardware test plan

    make coarse_sieve coarse_sieve_prof gf2_dlog_selftest     # ~12 min for the two CUDA builds
    ./gf2_dlog_selftest
    g++ -O2 -march=native -std=c++17 -o t_cubic t_cubic.cpp && ./t_cubic       # ~10 s, ends "ALL OK"
    for e in "" tower ph rr cubic; do ./coarse_sieve --selftest $e; done
    nvcc -O3 -arch=native -o randread randread.cu && ./randread                # < 1 min

At d ≥ 40 the selftest's lines should say `batched: N classes (3 loop orders) and M (4 fills), ok`, the cubic ones at 6 | d too: `[d=54 cubic logs, 2 MB tables: ... 0 wrong; batched: ... ok; pairs: ..., ok; over GF(2^18), ...]`. Please send `randread`'s output.

1. **The new code against your files** (~30 min): round 9's order and the rewritten grouped order at d = 40-45, and the cubic tower at d=42. `COARSE_METHOD` keeps each degree on its usual engine, since `--no-tune` would otherwise use PH at odd d.

       for d in 40 41 42 43 44 45; do
         m=$([ $((d % 2)) = 1 ] && echo rr || echo tower)
         for o in 1 2; do
           COARSE_METHOD=$m COARSE_PIPE=$o ./coarse_sieve 136279841 $d p${o}_$d \
               --load-survivors out$((d-1)).survivors.txt --no-checkpoint --no-tune
           cmp p${o}_$d.found.cert out$d.found.cert && cmp p${o}_$d.survivors.txt out$d.survivors.txt \
               && echo "d=$d order $o identical"
         done
       done
       for o in 0 1 2; do
         COARSE_METHOD=cubic COARSE_PIPE=$o ./coarse_sieve 136279841 42 c${o}_42 \
             --load-survivors out41.survivors.txt --no-checkpoint --no-tune
         cmp c${o}_42.found.cert out42.found.cert && cmp c${o}_42.survivors.txt out42.survivors.txt \
             && echo "d=42 cubic order $o identical"
       done

   - The done lines double as pass timings, against round 8's (r5-order: 150.09 s at d=44; grouped: 428.06 s at d=45, and so on).
   - If anything differs, please send the log, and keep round 8's binary for production until I've fixed it.
2. **The tune lines, with and without the prefetches** (~10 min). `COARSE_TEST_STOP_AFTER_CHUNKS=1` stops right after the tuner and the first chunk:

       for pf in 1 0; do
         for d in 44 45 47 48; do
           COARSE_PREFETCH=$pf COARSE_TEST_STOP_AFTER_CHUNKS=1 ./coarse_sieve 136279841 $d t$d \
               --load-survivors out$((d-1)).survivors.txt --no-checkpoint 2>&1 | grep "tune d="
         done
       done

   - Eight lines, holding every candidate's ns/class:
     - d=44 and 45 against round 8's tune lines (0.377 for GF(2^11) r5-order, 0.534 for RR grouped). The same order in both rounds isolates the g^u loads; r9-order and the rewritten grouped order are new.
     - d=47 for RR;
     - d=48 for the cubic tower against the quadratic ones.
     - The `pf 0` lines against the `pf 1` lines: what the prefetches do, candidate by candidate.
3. **Phase timing** (~12 min), as in round 8, plus the cubic tower:

       for o in 0 1 2; do
         COARSE_METHOD=rr COARSE_PIPE=$o COARSE_TEST_STOP_AFTER_CHUNKS=3000 ./coarse_sieve_prof 136279841 47 pt47 \
             --load-survivors out46.survivors.txt --no-checkpoint --no-tune
         COARSE_METHOD=cubic COARSE_PIPE=$o COARSE_TEST_STOP_AFTER_CHUNKS=600 ./coarse_sieve_prof 136279841 48 pt48 \
             --load-survivors out47.survivors.txt --no-checkpoint --no-tune
       done
       COARSE_PREFETCH=0 COARSE_METHOD=rr COARSE_PIPE=2 COARSE_TEST_STOP_AFTER_CHUNKS=3000 ./coarse_sieve_prof 136279841 47 pt47 \
           --load-survivors out46.survivors.txt --no-checkpoint --no-tune

   Please send the seven `[phase timing ...]` lines.
4. **WSL2's CPU** (section 6a): d=44 once with `COARSE_SYNC=poll`. Watch `top` and compare the done line.
5. **The T4's clocks** (section 6a): the `nvidia-smi` log during its next long run.
6. **Production.**
   - **d=53, while it runs:** steps 1-3 need the GPU, so they can't run alongside it, but the run can pause. Ctrl-C saves a checkpoint and exits; the same command continues later, with either binary.
     - Under round 9 the degree keeps its tuned engine and order, RR grouped, now in round 9's grouped order. That combination was tested from a round-8 checkpoint (K26).
     - Switching is worth it only if step 2 shows round 9's grouped order faster at d=45 or 47.
   - **d=54**, after d=53: `./coarse_sieve 136279841 54 out54 --load-survivors out53.survivors.txt`.
     - Its tuner will compare 12 or more candidates: GF(2^9), GF(2^3), the classic tower and the cubic tower, each in three orders, with TP-global twins where they apply.
     - Please send the tune line.
     - If step 1 or 2 showed a problem, run d=54 with round 8's binary instead.
   - **d=55** (RR, 2 GB table) from d=54's survivors.
     - Or in parallel on the T4 from d=53's survivors, with round 7's merge recipe.
     - On the T4 I'd expect ~1.0 ns/class, about a week and ~$20 at d=52's rate: an estimate.

## 9. Projections

On your RTX 2070 SUPER. The ranges cover what round 9's changes might do; your tune lines will narrow them.

| d | engine | classes | time |
|---|---|---|---|
| 53 | RR grouped (round 8) | 1.70e14 | ~26.8 h (tune 0.564 ns/class; running) |
| 54 | tower or cubic tower | 3.34e14 | 1.7-2.5 days (0.45-0.65 ns/class) |
| 55 | RR, 2 GB table | 6.55e14 | 3.4-4.6 days (0.45-0.60 ns/class) |

- **RR's rate** has stayed flat as its table grew: 0.550 ns/class at d=45 (64 MB), 0.584 at d=51 (512 MB), 0.564 tuned at d=53 (1 GB). So d=55's 2 GB table should not change it much.
- **d=54** is the least certain: the quadratic tower's 1 GB T has never been timed on your card, and the cubic tower has never run at all.

## 10. Continuing in a new conversation

`coarse_sieve_devkit_r9.tar.gz` carries on from round 8's kit:
- the sources of rounds 5-9;
- the emulated build and the two suites, now for round 9;
- the seed and the oracles' checksums, with the oracle generator;
- the nvcc check build;
- the tools, now with `randread.cu`.

Its README says how to use it: `./setup.sh` (~45 min, mostly one oracle), then `bash tests/regress.sh` (~50 min).

To start round 10 in a new conversation in this project, attach:
- the dev kit;
- this handoff (also the project doc `claude/coarse_sieve_handoff.md`);
- your logs and results, as you have been doing.

## Appendix: earlier rounds, for reference

**Round 8 (2026-10-28).**
- What changed: the watchdog and the lock-file notes; the grouped loop order and TP-global as tuner candidates; table reads past L1; round 5's order as the untuned default; the phase-timing build.
- Your results: the grouped order lost at tower degrees and tied at RR (picked at d=45 and d=53); `COARSE_LDCG` had no effect; d=52 on the T4 in 61,535 s.

**Round 7 (2026-10-26).** Full batches (the warps pool the classes they find), the cheaper even-d tie-break, the time of all runs in checkpoints, lockstep warps in the emulator. Your results: pass = tune rate; d=44 150 s, d=45 434 s; the d=51 production run matched the known ranges.

**Round 6 (2026-10-25).** One Zech log per reciprocal pair; the scrambled, stratified pass order and an honest time-left estimate; a quick tuner sample; the checkpoint records the engine; the command-line cleanup. Your results: d=44 268 s, d=47 2,142 s, d=49 8,369 s.

**Round 5 (2026-10-24).** The irregular classes joined the batch inversion (1.97x at d=44). Also RR and degree 49, auto-checkpointing, and tuning. Your results: d=47 (RR) 3,648 s, d=49 14,185 s, d=50 34,627 s.

**Round 4 (2026-10-22).** Subfield tower: one DRAM read per tower log. Your results: d=44 668.65 s.

**Round 3 (2026-10-20).** Batched kernel: FKM enumeration, 32 classes per thread bit-sliced, Montgomery batch inversion, the PH planner, depth 55.

**Round 2 (2026-10-19).** The logs path: one Zech log per class from tables of ~2^(d/2) entries.

**Round 1 (2026-10-18).** LFSR stepping in the bucketed path (~1000x), other fixes.

Sources:
- [NVIDIA, CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) (section 6a: its known limitations, none about blocking waits)
- [NVIDIA forums, "CPU spins while waiting for GPU to finish computation"](https://forums.developer.nvidia.com/t/cpu-spins-while-waiting-for-gpu-to-finish-computation/241672/5) (the spin, yield and blocking-sync modes)
- [NVIDIA Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) (the per-instruction exports of section 2)
- [Brent, Larvala, Zimmermann, *A fast algorithm for testing reducibility of trinomials mod 2* (Math. Comp. 2003)](https://maths-people.anu.edu.au/~brent/pd/rpb199.pdf)

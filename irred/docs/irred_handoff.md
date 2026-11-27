# irred — handoff (complete record, Rounds 1–3)

Fork of R. P. Brent's `irred` 3.15 (2003), now in
`github.com/nclvrps/mersenne_trinomial`, subdirectory `irred/`.
The program tests one trinomial x^r + x^s + 1 over GF(2) by computing
x^(2^r) mod (x^r + x^sodd + 1) with r squarings. The trinomial is
irreducible iff the result is x. Here sodd is whichever of s, r−s is odd;
using the reciprocal trinomial when s is even is harmless, because it is
irreducible iff the original is.

This file supersedes the earlier versions of `claude/irred_handoff.md` in
the Brent trinomials project (Round 1, 2026-11-17; Rounds 1–2, 2026-11-19).

---------------------------------------------------------------------------
## 0. Where things stand

| version | date | what |
|---|---|---|
| 3.15 | 2003 | Brent's original; `reference/irred315_orig.c` (+ Gord's compile patch) |
| 0.14 | 2026-05 | Gord's fork, cleaned up; **has a reduction bug** (§2.3); `reference/irred_0.14.c` |
| 0.20 | 2026-11-17 | Round 1: CLI `r s`, 64-bit residue, checkpoints, bug fixed, ~2–4x faster; `reference/irred_0.20.c` |
| 0.21 | 2026-11-19 | Round 2: Gerbicz error checking, map check, `--threads`, `--nt` |
| **0.22** | **2026-11-25** | **Round 3: 0.20 checkpoints no longer read (and never overwritten), `--nt` automatic, `--stop-after`; `irred/irred.c`** |

* **Round 3 (0.22, §4a)**, from Gord's round-2 test plan on his Zen 2:
  version 1 (0.20) checkpoints are no longer read — 0.22 refuses to start
  over one rather than overwrite it; `--nt` turns itself on when the
  squaring buffers do not fit in the L3 cache (his measurement: 2.40 →
  1.76 ms per squaring on Zen 2); `--stop-after N` for timing runs and tests.
  x^136279841 + x^12338357 + 1 was confirmed irreducible (hence primitive)
  by irred on his machine.
* 0.21 is validated in the sandbox (§5): the selftest, a 5,124-case
  comparison with an independent reference, exhaustive fault-injection
  sweeps (8,606 faults: all detected, all results right), checkpoint and
  compatibility tests, an end-to-end interrupted run at r = 6972593, and
  reinsertion of the real 0.14 bug.
* 0.21 continued 0.20's runs (it read version 1 checkpoints); 0.22 does
  not: finish any remaining 0.20 run with 0.21.
* Measured on Gord's 6-core Zen 2 (round 2 test plan, r = 136279841): the
  selftests pass; 2.40 ms per squaring on one thread, 1.76 with `--nt`,
  1.52 with `--nt --threads 2`, no gain beyond 2 threads (§4).

---------------------------------------------------------------------------
## 1. The program

### 1.1 Usage (0.22)

    irred [options] r s

    -v, --verbose                 --progress N        (interim residue every N)
    --no-checkpoint, --no-ckpt    --checkpoint-mins M, --ckpt-mins M (default 30)
    --threads N   (OpenMP; default 1; used for r > ~5e6)
    --nt, --no-nt (streaming stores in the SIMD interleave; default: on
                   when the two squaring buffers, r/4 bytes, exceed the L3)
    --stop-after N (stop after N squarings of this run, as on SIGTERM)
    --isa auto|avx512|avx2|generic      --selftest      -h, --help
    error checking: --no-ec  --ec-block L  --ec-window N  --ec-retries N
                    --map-checks N  --seed N  --ec-fault SITE:N (testing)

The result line keeps the old log format:

    r s irreducible (may be primitive) LOG
    r s x0123456789abcdef LOG     (s odd)      r s y... LOG   (s even: reciprocal)

The residue is the low 64 bits of x^(2^r) mod (x^r + x^sodd + 1); its low
8 hex digits are the 32-bit residue of 3.15 and 0.14 (except where 0.14 was
wrong, §2.3). Before the result, the program (since 0.21) prints whether
all r squarings were error-checked.

Exit status: 0 finished, 1 error (including: a 0.20 checkpoint for this
r, s is present), 2 stopped by SIGINT/SIGTERM or `--stop-after` (after a
checkpoint, unless `--no-ckpt`), 3 error check failed persistently or map
check failed.

Limits: r odd, r ≥ 257, r − sodd ≥ 128 (i.e. s ≥ 128 if s is even,
r − s ≥ 128 if s is odd). Small odd s is fine.

### 1.2 Brent's algorithm, as implemented

A polynomial of degree < r is held in words 0..q11 (q11 = (r−1) div 64).
One squaring is two passes:

1. **reducep** (in place). It reduces the implicit square Σ a_i x^(2i),
   leaving it in the "scrambled" order: bit i ≤ alpha holds x^(2i), and bit
   i ≥ alpha+1 holds x^(2i−r) (odd exponents), with alpha = (r−1)/2. For
   bit i ≥ alpha+1, x^(2i) = x^(2i−r) + x^(2i−r+sodd): the first term stays
   at bit i and the second goes to bit i − delta, with delta = (r−sodd)/2.
   Going down from the top, target word T receives words T+deltaw and
   T+deltaw+1. Those words must already have received their own bits, so
   the reduction cascades and has to run downwards.
2. **interleave** into the other buffer. Output bit 2m comes from bit m and
   output bit 2m+1 from bit alpha+1+m, which restores the normal order.

The parameters (setup_params) are q11, q4 = alpha div 64,
q41 = (alpha+1) div 64, s1_x, delta = 64·deltaw + deltaq, mask1 and mask2.

### 1.3 Speed and the source of each gain (2.1 GHz Xeon, AVX-512, 260 MB L3, gcc 13)

| r | 0.14 | 0.20/0.21 1 thread | 0.21 2 threads |
|---|---|---|---|
| 756839 | 15.4 s | 3.3 s (3.6 s with checking) | (below threshold) |
| 30402457 | — | 0.41 ms/sq | 0.19 ms/sq |
| 136279841 | 3.7 ms/sq (5.9 days) | 1.8 ms/sq (2.9 days) | 0.95–1.05 ms/sq (1.5–1.7 days) |

* The interleave is ~72% of single-thread time; the AVX2 and AVX-512BW
  versions (nibble lookup by vpshufb, byte unpack, lane permute) are chosen
  at run time. The portable C version is not vectorised by gcc, because of
  the interleaved stores.
* The reduction runs in blocks of ≤ deltaw target words. Such a block never
  reads its own targets, so `restrict` is true and gcc vectorises it.
  `target_clones` gives AVX2/AVX-512 versions without `-march`.
* Interleaving forward only: alternating directions (3.15's trick) was ~10%
  slower.
* Tried and rejected: BMI2 PDEP (slower than AVX2, and every CPU with BMI2
  has AVX2). Non-temporal stores were 2x slower here, where the data fit in
  the L3 cache. That is not Gord's case, so 0.21 offered them as `--nt` (off
  by default) to be measured on Zen 2; 0.22 turns them on by itself when
  the squaring buffers exceed the L3 cache (§4a).

---------------------------------------------------------------------------
## 2. Round 1 (0.20, 2026-11-17)

### 2.1 Requested changes

* The CLI is just `r s`; the restart file, log file, time limit, skip file
  and stdin input are gone.
* 64-bit residue.
* Checkpoints `irred_r<r>_s<s>.ckpt` (+ `.ckpt.1`, `.ckpt.2`), every 30
  minutes and on SIGINT/SIGTERM. They have magic `IRREDCKP`, CRC-32 on the
  header and on the data, and are written as tmp file, fsync, rotate,
  rename. The run resumes automatically from the newest valid file for the
  same (r, s); other files are ignored with a message. They are deleted on
  completion. `irred_r<r>_s<s>.lock` (flock) keeps a second process off the
  same (r, s). r and s are in the file name so that runs for different s in
  one directory cannot clobber each other.
* `--progress N`.

### 2.2 Decisions

* With Swan's theorem (r ≡ ±3 mod 8), 3.15/0.14 fell into "Illegal
  parameters"; the program now notes that the trinomial is reducible and
  computes the residue anyway.
* The limit was relaxed from r − sodd ≥ 384 to ≥ 128, which is the true
  requirement of the code (deltaw ≥ 1), verified against the reference
  including small sodd.

### 2.3 The 0.14 bug

When deltaq = 0, i.e. r − sodd ≡ 0 (mod 128), 0.14's `xor_shift_zero` call
folded source words q4..q11−1 instead of q4+1..q11. It printed wrong
residues, and could have reported an irreducible trinomial as reducible.
3.15 was correct. For r = 136279841 (≡ 33 mod 128) the affected s are
s ≡ 33 (s odd) and s ≡ 0 (s even) mod 128, about 1 in 64. Any 0.14
result for those s must be redone. `tests/bug014_test.sh` re-inserts the
bug and reproduces 0.14's residues exactly (§3.6).

0.14 also declared the overlapping source and target ranges of the in-place
reduction `__restrict__`, which is undefined behaviour. Fixed by the
blocking in §1.3.

---------------------------------------------------------------------------
## 3. Round 2 (0.21, 2026-11-19): error checking

### 3.1 The check

The chain is u(t+1) = u(t)^2 mod T with u(0) = x. Over GF(2), squaring is
the Frobenius map F, which is **additive**: F(u+v) = F(u) + F(v). As in
tsfactor's h-table check (errorcheck_handoff §4.2), the Gerbicz accumulator
can therefore be an XOR instead of a product.

A window starts at a verified state u(0), kept as V, and has n blocks of L
squarings. After each block that does not end the window, the current u is
XORed into D, which starts as u(0):

    D = u(0) + u(L) + ... + u((n-1)L)
    F^L(D) = u(L) + ... + u(nL)          =>   F^L(D) = D + u(0) + u(nL)

Closing the window costs L squarings of a copy of D, plus one comparison
pass over D, V, u and the result.

**Detection is deterministic.** Suppose a fault changes u by e ≠ 0 at step
t of a window. Because squaring is linear, every later state is off by
F^(t'−t)(e). The off-by terms telescope, and the two sides of the identity
differ by exactly F^(jL−t)(e), where jL is the first block end after t.
That is never 0, because F is injective (T is squarefree). So a single
fault anywhere in the window — in a squaring, or damage to u while it sits
in memory — is always detected. There is no probability and no tuning
knob.

Faults in D, or in the check's own squarings, cause a false alarm, which
costs one retry. Damage δ to D goes unseen only if F^L(δ) = δ, and then it
is harmless, because D is never used for the result. Two faults whose
effects cancel exactly are the only miss; independent random faults never
do that.

On failure the window is redone from V. After more than `--ec-retries` (3)
consecutive failures at the same point the program exits with status 3: a
persistent fault.

**Implementation note (a bug I made and fixed).** u(nL), the state that
closes the window, must not be XORed into D. The code decides whether to
close before it accumulates. The first version accumulated first, and every
check failed.

### 3.2 What the chain cannot see, and the map check

The chain proves only that each squaring agrees with *the code's own*
squaring map. A deterministic bug that makes the code compute some other
**GF(2)-linear** map g ≠ F leaves the chain consistent with g, so every
window passes. The 0.14 bug was exactly that kind (an XOR fold of the
wrong word range). Integer Gerbicz (PRP tests) does not have this blind
spot, because it checks squarings against a separately coded
multiplication. The XOR version gives that up in exchange for its nearly
free accumulator.

So at start-up 0.21 compares the fast squaring with **`ref_square`**,
which shares no code with it: a byte-table spread to 2r bits, then a
word-wise top-down reduction of the double-length square in normal order.
Inputs are `--map-checks` (16) random polynomials plus one sparse one, on
the actual (r, s), buffers and SIMD path. A wrong linear map disagrees with
F on at least half of all inputs, because its difference from F has a
proper kernel. So 16 inputs miss it with probability ≤ 2^−16. The check
costs about 0.5 s at r = 136279841 and is instantaneous for small r.

A bug that is *not* linear (reads of uninitialised memory, races) breaks
additivity, and the chain catches it.

### 3.3 Protecting the verified state

The chain verifies how the state evolved, not where it started. A copy that
becomes a starting point — V, a checkpoint, a resume — must therefore be
protected separately.

* While comparing, `ec_close` takes the CRC-32 (slice-by-8) of the
  very words of u that it compared. V is copied from u, and the copy's CRC
  must equal that value: the copy is retried up to 3 times, or the check
  counts as failed. D is then re-initialised from V.
* The next window's chain starts from u in the work buffer, while D and the
  check use V. Damage to u between windows is therefore just an error at
  t = 0 of the next window, and is caught (the discrepancy is F^L(e)).
* Before V is used for a rollback or written as a checkpoint, its CRC is
  checked. A checkpoint is written **only** from V, i.e. only from verified
  states. If V's CRC is bad, the checkpoint is not written, and the rollback
  falls back to the newest valid checkpoint on disk, or to the beginning
  (x) if there is none.
* At the end, the result comes from V, or from the work buffer if V was
  damaged and the work buffer still matches the verified CRC.
* A checkpoint file's data CRC is the same CRC, so disk corruption is caught
  on resume, as in 0.20.

### 3.4 Schedule, defaults, cost

* L = max(32, ⌊√r / 16⌋), which is 729 at r = 136279841, and N = L blocks
  per window, so a window is 531,441 squarings (about 16 minutes there).
* A window also closes early for a due checkpoint, a signal, or the tail
  (fewer than L squarings left: the last window then uses a block of the
  remaining length, so k = r always ends a block).
* On SIGINT/SIGTERM the program finishes the current block, closes the
  window, writes the verified checkpoint and exits 2. That takes at most
  2L squarings, about 2.7 s at r = 136279841.
* Measured cost of one check at r = 136279841: 1.37 s, i.e. L squarings +
  0.03 s. With 16-minute windows that is **≈ 0.15%**. The XOR accumulate
  every L squarings is in the noise. At small r the relative cost is higher
  (r = 756839: 3.6 s vs 3.37 s, +7%), because the blocks are short
  (L = 54); this does not matter at that size.
* Memory: six buffers of r/8 bytes (work pair, V, D, two scratch for the
  check) = 100 MB at r = 136279841.

### 3.5 Checkpoint format version 2, and 0.20 compatibility

The version 2 header adds `unchecked` (how many of the k squarings were done
without error checking), `ec_checks`, `ec_failures` and 5 reserved words.
0.21 reads **version 1 (0.20) files**: their k squarings count as unchecked,
a note is printed on resume, and the final report says
"Note: N of the r squarings were NOT error-checked". `--no-ec` segments are
counted the same way.

0.20 cannot read version 2 files. It would report "unknown format version",
fall back to any version 1 generation still present, or start afresh.

*0.22 (round 3):* version 1 files are no longer read, and a version 1 file
for the same r and s stops 0.22 with exit status 1 instead of being
overwritten (§4a).

### 3.6 Validation

**Exhaustive fault sweeps** (`tests/ec_sweep.sh`, `logs/ec_sweep*.log`): one
bit flip per run, at every opportunity of a site.

| site | runs | detected | result right | unchecked (`--no-ec`) |
|---|---|---|---|---|
| sq, r=1279 s=500 (residue) | 1279 | 1279 | 1279 | 1275 silently wrong, 4 harmless |
| sq, r=1279 s=418 (irreducible) | 1279 | 1279 | 1279 | **1279 silently wrong** ("not irreducible") |
| sq, r=1279, L=8, N=4 | 1279 | 1279 | 1279 | |
| sq, r=4423 s=2098 (irreducible), portable path | 4423 | 4423 | 4423 | 4423 silently wrong |
| acc (D), two configurations | 252 | 252 | 252 | |
| chk (inside a check) | 47 | 47 | 47 | |
| ver (V after verification) | 47 | 47 | 47 | |

Also: under ASan+UBSan (subset), with 2 threads forced on, and the same
results.

**The real 0.14 bug** (`tests/bug014_test.sh`): re-inserted into a scratch
copy of 0.21. With the map check off, every window passes and the residues
are wrong, with the same low 32 bits that 0.14 printed:

    4423 3015: true x95868bc52f115a6e | 0.14 xe25996b6 | chain only: x0423c7a7e25996b6 | map check: exit 3

With the map check on, it exits with status 3. Cases with deltaq ≠ 0 are
unaffected.

The selftest (`--selftest`) runs all the known cases checked and
unchecked, one fault per site, and the linear-bug demonstration.

---------------------------------------------------------------------------
## 4. Threads (`--threads N`, Round 2)

There is one OpenMP parallel region per squaring. Each reduction block is
split between the threads (its target words never overlap its source words),
with a barrier between blocks; reduction blocks are only split if
deltaw ≥ 2048, otherwise one thread does the reduction. Then the interleave
is split by j (disjoint outputs). Threads are used only if q41 ≥ 40000
(r ≳ 5.1 million): below that the per-squaring synchronisation costs more
than it saves (r = 1398269: 2x slower with 2 threads). Measured here
(2 vCPUs):

| r | 1 thread | 2 threads |
|---|---|---|
| 3021377 | 0.0172 ms | 0.0287 ms (forced) |
| 6972593 | 0.0496 ms | 0.0414 ms |
| 13466917 | 0.143 ms | 0.0889 ms |
| 30402457 | 0.41 ms | 0.193 ms |
| 136279841 | 1.80 ms | 0.95–1.05 ms |

The residues are identical for any thread count (`tests/cmp.sh` runs 2 and 3
threads with `--mt-min-words 1`, which forces threads on for small r).
Built without `-fopenmp`, `--threads` is ignored with a note. The pieces of
the interleave start at multiples of 8 words (whole cache lines, and the
alignment `--nt` needs).

`--nt` uses streaming stores (`vmovntdq`) in the AVX2/AVX-512 interleave,
with an `sfence` at the end. The output buffer is 64-byte aligned, and j0 is
0 or a multiple of 8. It is correct on every path (selftest, `tests/cmp.sh`).
On this Xeon it is much slower, because its 260 MB L3 holds the whole
working set and streaming stores bypass it. On a machine where
r = 136279841 is DRAM-bound (Zen 2: 16 MB of L3 per CCX against a 34 MB
working set) it saves the read-for-ownership of the 17 MB output, about a
quarter of the traffic. Measured in round 3 on Gord's Zen 2: 27% faster on
one thread (2.40 → 1.76 ms); 0.22 turns it on automatically there (§4a).

---------------------------------------------------------------------------
## 4a. Round 3 (0.22, 2026-11-25)

**What Gord's run of the round-2 test plan showed** (6-core Zen 2, AVX2,
Ubuntu 26.04, gcc 15): `make` and `make selftest` clean; at r = 136279841,
ms per squaring (`--progress 2000`, 30000 squarings):

| threads | default | `--nt` |
|---|---|---|
| 1 | 2.40 | 1.76 |
| 2 | 1.86 | 1.52 |
| 3 | 1.83 | 1.49 |
| 4 | 1.69 | 1.50 |
| 6 | 1.68 | 1.57 |

So on Zen 2 the streaming stores win (27% on one thread), as predicted for a
DRAM-bound machine, and threads beyond 2 buy nothing once they are on.

**The test plan's missing output was not irred's fault.** The loop piped
`timeout -s TERM 90 irred ... | grep '^iter' | tail -1` and printed nothing;
without `tail -1` the lines appeared. irred line-buffers stdout and flushes
every progress line. GNU `timeout` puts itself in a process group of its
own and signals only that group; Ubuntu 26.04's Rust coreutils (uutils)
`timeout` evidently signals the pipeline's whole group, so `grep` and `tail`
were killed with irred before `tail` could print (bash's "Terminated" for the
job is the tell: with GNU timeout the job's last process, `tail`, exits
normally). With SIGINT the shell also aborted the loop, which is what bash
does when a foreground job dies of SIGINT. Robust forms: write irred's output
to a file and look at it afterwards, or (new) `--stop-after N`, which needs
no `timeout` at all.

**Changes:**
1. **Version 1 checkpoints are no longer read.** `struct ckpt_hdr_v1` and the
   conversion are gone (Gord: no compatibility needed). A version 1 file is
   reported and ignored; if one of them belongs to the same r and s and no
   valid version 2 checkpoint exists, 0.22 **refuses to start** (exit 1:
   "finish the run with irred 0.21, or delete the files"). Starting fresh
   would rotate those files away and delete them at the end, losing days of
   0.20 work to a version mix-up.
2. **`--nt` is automatic.** If neither `--nt` nor `--no-nt` is given, the
   streaming stores are used when the two squaring buffers (2 × r/8 bytes,
   33 MB at r = 136279841) exceed the L3 cache one core sees: the size from
   `/sys/devices/system/cpu/cpu0/cache/index*/` (level 3, i.e. one Zen CCX,
   16 MB on Zen 2), else `sysconf(_SC_LEVEL3_CACHE_SIZE)`, else off. It says
   so when it turns them on. On the sandbox's Xeon (260 MB L3) they stay off,
   where they were 2x slower. `--selftest --nt` is now in `make selftest`.
3. **`--stop-after N`** stops after N squarings of the current run exactly as
   SIGTERM would at that point: with checking on, the block in progress is
   finished and verified (up to 729 more squarings at r = 136279841), a
   checkpoint is written unless `--no-ckpt`, exit status 2. For timing runs:
   `irred --no-ckpt --progress 2000 --stop-after 30000 r s`.
4. Version 0.22, 2026-11-25; header comment and usage updated.

**Validation (sandbox):** `make selftest` (now 8 lines including `--nt`);
`tests/cmp.sh` 5,124 comparisons with the naive reference and the 3.15
comparison, 0 failures; `make sweep` all faults detected, all results right;
`make bug014`; `tests/ckpt_test.sh` 22 checks (the 0.20 section now checks
the refusal and that the 0.20 files are untouched; three new `--stop-after`
checks: stop with a verified checkpoint and exit 2, resume to the right
result, and a `--no-ckpt` timing run that leaves no files).  At
r = 136279841, s = 12338357, the residues after 2000, 4000 and 6000
squarings agree with Gord's 0.21 run on the Zen 2 (`logs/round3/`).

---------------------------------------------------------------------------
## 5. Verification ledger

**Verified in the sandbox** (Xeon @ 2.1 GHz, AVX-512, 2 vCPUs, gcc 13.3,
clang 18):

* `--selftest` passes for native, `--isa avx2`, `--isa generic`, no-`-march`,
  ASan+UBSan, and 2 and 3 threads; clang build too.
* `tests/cmp.sh`: 5,124 comparisons with `tools/ref.c` (naive, bit at a
  time), r ∈ {257, 383, 1151, 1279, 2203, 4423, 9689}. The cases are
  boundary s, every deltaq = 0 s, small sodd and random s; the builds are
  every path, `--no-ec`, threads, `--nt` and ASan. Plus the low 32 bits
  against 3.15 at r = 44497 and 132049 (`logs/cmp.log`).
* Fault sweeps and the 0.14 bug test (§3.6).
* `tests/ckpt_test.sh`: 20 checks in 0.21 (22 in 0.22, §4a). They cover
  SIGINT/SIGTERM (exit 2, verify-before-stop), resume equivalence (final
  result and every progress residue), damaged/mismatched/foreign files, the
  lock, `--ckpt-mins 0`, **0.20 → 0.21 continuation** (in 0.22: refusal),
  mixed `--no-ec`, and faults in a checkpointed run (including fallback to
  the disk checkpoint when V is damaged).
* `tests/e2e_mid.sh`: r = 6972593, 2 threads + checking + checkpoints every
  30 s, SIGINT after 120 s (the block was finished and verified first:
  stopped at 2592049, checkpoint at 2592150), then resumed. Against a
  single-thread unchecked run: same final residue (xe724e828890fa31e) and
  every progress residue identical; 263 windows verified, checking cost
  0.62% (`logs/e2e_r6972593.log`).
* r = 136279841: speed with and without checking, per-check cost,
  2 threads, `--nt`; residues at matching iterations agree across variants
  (`logs/bench.log`).

**Verified on Gord's machine** (6-core Zen 2, round-2 test plan, 0.21):
build, `make selftest`; production-size timings for 1–6 threads with and
without `--nt` (§4a); irred confirmed x^136279841 + x^12338357 + 1
irreducible.

**Round 3 (0.22), verified in the sandbox:** §4a.

**Not verified:**

* 0.22 on Gord's machine, in particular the automatic `--nt` (it should turn
  itself on there and print the line saying so).
* Non-x86 builds (the portable path is tested via `--isa generic`, but not
  on another architecture).

---------------------------------------------------------------------------
## 6. Open items / ideas for a next round (rough order of value)

1. **GPU irred** (Gord: probably a later round; ~8–10 h instead of 2–3
   days, and a fast independent recheck of a primitive trinomial).
   tsfactor round 10 split its program into a driver and backends behind an
   interface (`TsDev`, `ts_core.h`), with a CPU backend (`ts_cpu.h`) and a GPU
   one; a GPU irred would follow the same pattern with a much smaller
   interface: init(r, s), square_n(n), xor_into(D), equal(D, V), copy to/from
   host, plus the start-up tuning record and fingerprint, the per-(r, s)
   checkpoint names and the prefix lock that `ts_core.h` already has (worth
   moving into a small shared header when irred uses them). The squaring
   exists twice already: `k_square_fused` on the GPU (any parity, middle
   exponent ≤ (r−1)/2) and, new in round 10, an even-s in-place BLZ
   reduction on the CPU (`ts_cpu.h`), so a CPU irred could work modulo the
   original trinomial for even s too. The residue question below then has a
   clean answer: both programs can compute x^(2^r) modulo
   x^r + x^min(s, r−s) + 1 (a new residue letter, say `z`), keeping today's
   `x`/`y` residue as the CPU default for continuity with the old logs.
   The original notes:
   tsfactor already has validated GPU squaring for these
   trinomials (`gf2_trinomial_cuda.h`: `k_spread_square` + two
   `k_fold_pass`; tsfactor's fused single-pass squaring measured
   **0.25 ms at r = 136279841** on the RTX 2070 SUPER). r squarings would
   take ~10 hours, against ~1.7–3 days on CPUs. The XOR chain of §3.1 maps
   directly onto it: one XOR kernel per block, the check on the device, and
   only a flag back, as tsfactor does. The map check would compare the GPU
   squaring against `ref_square` on the host.
   **Design point.** The fold passes need s ≤ (r−1)/2. To produce
   residues compatible with the CPU program, the GPU must reduce modulo
   x^r + x^sodd + 1, which needs about ⌈r/(r−sodd)⌉ passes: 2 for
   sodd ≤ r/2, but very many when s is small and even (then sodd = r − s).
   Options: use min(s, r−s) and label the residue differently (the verdict
   is the same, but CPU cross-checks need `tools/ref.c` adapted to any
   middle exponent), or add passes only when 2 < ⌈r/(r−sodd)⌉ ≤ some
   bound. tsfactor's emulation (`cuda_emul.h`) allows a sandbox-validated
   first version, as in tsfactor rounds 4–5.
2. ~~Measure on Zen 2~~ (done, §4a). `--nt` is now automatic. `--threads`
   stays 1 by default: on Zen 2 two threads give 14% (1.76 → 1.52 ms) and
   more give nothing, and one process per core is the better use of a
   machine running several tests; `--threads 2` is worth it for a single
   urgent test.
3. **Fuse the reduction and the interleave** (one pass, ~15% less memory
   traffic): interleave word j right after the reduction front passes it.
   It is complicated by the cascade and the special word q4. Zen 2 is
   DRAM-bound at r = 136279841 (the `--nt` gain, §4a), so this is now worth
   trying there.
4. **Verify the unchecked prefix of a run that 0.21 continued from 0.20**
   without redoing it. There is no shortcut: any check needs the squarings.
   The cheap procedure was step 4 of round 2's TESTPLAN: a fresh checked
   run with `--progress K` must print, at K, the residue that the resume
   line showed. Only a *reducible* verdict needs this: an undetected error
   turns x^(2^r) mod T into x only with probability about 2^−r, so an
   "irreducible" verdict (such as s = 12338357's) stands without it.
5. README of `mersenne_trinomial` (Gord: "incomplete and outdated").

---------------------------------------------------------------------------
## 7. Devkit file map

See `README_DEVKIT.md`.

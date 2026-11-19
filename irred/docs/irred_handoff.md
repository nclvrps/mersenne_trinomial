# irred — handoff (complete record, Rounds 1–2)

Fork of R. P. Brent's `irred` 3.15 (2003), now in
`github.com/nclvrps/mersenne_trinomial`, subdirectory `irred/`.
The program tests one trinomial x^r + x^s + 1 over GF(2) by computing
x^(2^r) mod (x^r + x^sodd + 1) with r squarings. The trinomial is
irreducible iff the result is x. Here sodd is whichever of s, r−s is odd;
using the reciprocal trinomial when s is even is harmless, because it is
irreducible iff the original is.

This file supersedes the Round 1 note (`claude/irred_handoff.md` in the
Brent trinomials project, 2026-11-17).

---------------------------------------------------------------------------
## 0. Where things stand

| version | date | what |
|---|---|---|
| 3.15 | 2003 | Brent's original; `reference/irred315_orig.c` (+ Gord's compile patch) |
| 0.14 | 2026-05 | Gord's fork, cleaned up; **has a reduction bug** (§2.3); `reference/irred_0.14.c` |
| 0.20 | 2026-11-17 | Round 1: CLI `r s`, 64-bit residue, checkpoints, bug fixed, ~2–4x faster; `reference/irred_0.20.c` |
| **0.21** | **2026-11-19** | **Round 2: Gerbicz error checking, map check, `--threads`, `--nt`; `irred/irred.c`** |

* 0.21 is validated in the sandbox (§5): the selftest, a 5,124-case
  comparison with an independent reference, exhaustive fault-injection
  sweeps (8,606 faults: all detected, all results right), checkpoint and
  compatibility tests, an end-to-end interrupted run at r = 6972593, and
  reinsertion of the real 0.14 bug.
* **0.21 continues a run started by 0.20.** It reads 0.20's version 1
  checkpoints, and reports at the end how many squarings were done without
  checking. Gord has a 0.20 run in progress (2026-11-19); `docs/TESTPLAN.md`
  says how to move it over.
* Not verified: anything on Gord's hardware (6-core Zen 2, no AVX-512).
  In particular `--threads` beyond 2 cores and the speed on a DRAM-bound
  machine are unmeasured.

---------------------------------------------------------------------------
## 1. The program

### 1.1 Usage (0.21)

    irred [options] r s

    -v, --verbose                 --progress N        (interim residue every N)
    --no-checkpoint, --no-ckpt    --checkpoint-mins M, --ckpt-mins M (default 30)
    --threads N   (OpenMP; default 1; used for r > ~5e6)
    --nt          (streaming stores in the SIMD interleave; measure)
    --isa auto|avx512|avx2|generic      --selftest      -h, --help
    error checking: --no-ec  --ec-block L  --ec-window N  --ec-retries N
                    --map-checks N  --seed N  --ec-fault SITE:N (testing)

The result line keeps the old log format:

    r s irreducible (may be primitive) LOG
    r s x0123456789abcdef LOG     (s odd)      r s y... LOG   (s even: reciprocal)

The residue is the low 64 bits of x^(2^r) mod (x^r + x^sodd + 1); its low
8 hex digits are the 32-bit residue of 3.15 and 0.14 (except where 0.14 was
wrong, §2.3). Before the result, 0.21 prints whether all r squarings were
error-checked.

Exit status: 0 finished, 1 error, 2 stopped by SIGINT/SIGTERM after a
checkpoint, 3 error check failed persistently or map check failed.

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
  the L3 cache. That is not Gord's case, so 0.21 offers them as `--nt` (off
  by default) to be measured on Zen 2.

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
quarter of the traffic. Unmeasured.

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
* `tests/ckpt_test.sh`: 20 checks. They cover SIGINT/SIGTERM (exit 2,
  verify-before-stop), resume equivalence (final result and every progress
  residue), damaged/mismatched/foreign files, the lock, `--ckpt-mins 0`,
  **0.20 → 0.21 continuation**, mixed `--no-ec`, and faults in a
  checkpointed run (including fallback to the disk checkpoint when V is
  damaged).
* `tests/e2e_mid.sh`: r = 6972593, 2 threads + checking + checkpoints every
  30 s, SIGINT after 120 s (the block was finished and verified first:
  stopped at 2592049, checkpoint at 2592150), then resumed. Against a
  single-thread unchecked run: same final residue (xe724e828890fa31e) and
  every progress residue identical; 263 windows verified, checking cost
  0.62% (`logs/e2e_r6972593.log`).
* r = 136279841: speed with and without checking, per-check cost,
  2 threads, `--nt`; residues at matching iterations agree across variants
  (`logs/bench.log`).

**Not verified:**

* Gord's machine (Zen 2: AVX2 path at production size, DRAM-bound
  behaviour, `--threads` 3–6). The AVX2 path is verified for correctness
  here (`--isa avx2`).
* A complete r = 136279841 run with 0.21 (only the 0.20 one Gord is running).
* Non-x86 builds (the portable path is tested via `--isa generic`, but not
  on another architecture).

---------------------------------------------------------------------------
## 6. Open items / ideas for a next round (rough order of value)

1. **GPU irred.** tsfactor already has validated GPU squaring for these
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
2. **Measure on Zen 2.** Measure `--threads 1..6`, each with and without
   `--nt`, at r = 136279841 (TESTPLAN step 2). Then decide the defaults:
   should `--threads` default to some number of cores above r ≈ 5e6, and
   should `--nt` turn itself on when the working set exceeds the L3?
3. **Fuse the reduction and the interleave** (one pass, ~15% less memory
   traffic): interleave word j right after the reduction front passes it.
   It is complicated by the cascade and the special word q4; worth it only
   if 2. shows a DRAM bound.
4. **Verify the unchecked prefix of a resumed 0.20 run** without redoing it.
   There is no shortcut: any check needs the squarings. The cheap
   procedure is in TESTPLAN.
5. README of `mersenne_trinomial` (Gord: "incomplete and outdated").

---------------------------------------------------------------------------
## 7. Devkit file map

See `README_DEVKIT.md`.

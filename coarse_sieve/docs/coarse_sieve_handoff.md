# coarse_sieve (-DWIDE_FIELD) -- round 4: one DRAM read per tower log

Baseline: round 3 (2026-10-20), which you validated on hardware: all
selftests PASS, d = 34..41 identical to round 2, d = 45..48 for
r = 74207281 matching the Brent et al. certificates and for r = 136279841
consistent with the already-done ranges, no spills in `-Xptxas -v`.

Deliverables

| file | what |
|---|---|
| `coarse_sieve.cu` | round 4 (replaces round 3's) |
| `gf2_dlog.h` | round 4: the subfield tower (inner tower over a subfield of K) |
| `gf2_dlog_selftest.cpp` | tests every tower variant; cross-checks tower logs against PH logs end to end |
| `cuda_emul.h` | adds stubs for `cudaFuncSetAttribute`, the shared-memory attribute, `__launch_bounds__`, and accepts a third launch parameter; backward compatible |
| `coarse_sieve_round4.patch` | the same changes as a patch against round 3 (flat paths) |

The `Makefile` is unchanged from round 3.

## TL;DR

* **Why the tower degrees came in slow in round 3.**  Your PH timings
  were at the upper end of my estimate (d=47: 5,411 s, 3.9x faster than
  round 2), but the tower ones were not (d=46: 2,921 s, only 1.38x; d=48:
  12,711 s).  The tower is bound by random DRAM reads, not instructions:
  round 2 did 4 table reads per class at d=46 in 2.63 ns, round 3 did 3 in
  1.91 ns -- both about 1.55 G reads/s (1.38 G/s at d=48, whose tables are
  twice as large).  My round-3 model had assumed the card sustains several
  times that, so cutting instructions bought little.
* **Round 4 removes two of the three reads.**  A tower log needs the logs
  of two elements a, b of K = GF(2^(d/2)); round 3 read both from a table
  LK of all 2^(d/2) elements.  Now they come from the same tower trick
  applied once more, inside K, over a subfield K_n: a few reads of tables
  with ~2^n entries (2^12 at d=48), which sit in shared memory.  Only the
  read of T remains.  This is item 3 of the round-3 list done more
  thoroughly: no division in K is needed, and both LK reads go.
* **Where it applies**: every tower degree with d/2 composite -- 36, 40,
  42, 44, 48, 50, 52, 54.  d = 34, 38, 46 (d/2 prime: K has no subfield)
  keep the round-3 tower.  `COARSE_TOWER_SUBN=0` selects the round-3 tower
  everywhere, for A/B.
* **Estimated**: d=48 ~1.1-1.5 h (round 3: 3.5 h), d=44 ~5 min; and if you
  go past 49, d=50 ~6-9 h, d=52 ~18-23 h, d=54 ~3.3-4.3 days.  These rest
  on the read rate measured above, so they should be closer than last
  time; the d=48 / A-B lines will tell (section 4).
* LK is no longer on the device either: 64 MB less at d=48, 512 MB at
  d=54.
* Output byte-identical in every comparison made here (section 3).
* The `totb` warning is gone.
* Your PS questions: section 6.

## 1. The subfield tower

For d = 2m, round 3 already writes e = a + b*w = b*(c + w) with a, b, c in
K = GF(2^m), so log e = (2^m+1) log_h(b) + T[log_h(a) - log_h(b)]
(h = g^(2^m+1) generates K*).  If n | m, 2 <= n < m, K is a degree-k
extension of K_n = GF(2^n), k = m/n, with basis 1, h, ..., h^(k-1).  Any
Y in K is then lambda*Y', lambda the last nonzero coordinate of Y (an
element of K_n) and Y' = (y_0/lambda, ..., y_(j-1)/lambda, 1, 0, ...):

    log_h Y = (2^m-1)/(2^n-1) * LKn[lambda] + TP[Y']

where LKn holds the 2^n logs in K_n (base hn = h^((2^m-1)/(2^n-1))), and
TP the logs of the normalized points Y' -- one per point of the projective
space P^(k-1)(K_n), indexed by the logs of Y's coordinates relative to
lambda.  For even m (k = 2) this is literally the tower again: TP has 2^n
entries.  Per class it costs ~7 byte-table reads (now giving the K_n
coordinates of a and b directly) and ~6 reads of LKn / TP instead of two
DRAM reads of LK.

| d | m | inner field | k | TP entries | small tables |
|---|---|---|---|---|---|
| 36, 40, 44, 48, 52 | 18 ... 26 | GF(2^(m/2)) | 2 | 2^(m/2) | shared memory (d=52: 62 KB) |
| 42 | 21 | GF(2^7) | 3 | 16,512 | TP in cache (64 KB) |
| 50 | 25 | GF(2^5) | 5 | 1,082,400 | TP in L2/DRAM (4.3 MB): smaller gain |
| 54 | 27 | GF(2^9) | 3 | 262,656 | TP in L2 (1 MB) |
| 34, 38, 46 | prime | -- | -- | -- | round-3 tower (LK) |

Device side: tower kernels now run 512-thread blocks (one per SM,
`__launch_bounds__(512, 1)`, so <= 128 registers -- the count nvcc already
chose in your log), and copy the byte tables, LKn and (when it fits) TP
into shared memory at kernel start.  T is still filled on the device by the
same coupon-collector sweep, now through the subfield-tower logs.  The
per-degree header names the inner field, e.g.
`d=48 logs (tower over GF(2^12), batched, 128 MB tables, ...)`.

Also: within a chunk the batched kernel now uses runs 8x shorter (~50 per
thread instead of ~6), so threads finish together; chunk boundaries are
unchanged, so checkpoints are compatible both ways.

## 2. What changed

`gf2_dlog.h`: `TowerDev` gains the subfield fields; `tower_sub_log`,
`tower_klog`, `tower_split` (shared by `tower_log` and the T fill);
`tower_build_host` builds LKn/TP from a labelled RREF of the basis
h^i * hn^j (and keeps LK/AK on the host only); `tower_sub_choice` picks n
(the largest proper divisor of m); `tower_dev` assembles a `TowerDev`.

`coarse_sieve.cu`: device tables (LKn/TP instead of LK), the shared-memory
staging and launch shape, `COARSE_TOWER_SUBN`, the header and selftest
lines, the shorter runs, and the `totb` fix (the unused variables are gone).

## 3. Validation done here

* Host: every tower variant against ground truth -- classic, the automatic
  choice, and for d <= 36 every other inner field (k = 2..9): every
  element for d <= 20, 20,000 random elements per degree up to d = 54.
  End to end, for 40 random classes at every even d up to 54, the tower's
  log equals the PH log, whose progression start s0 is in turn checked by
  verifying f | x^r + x^s0 + 1 directly.  0 wrong anywhere.
* Emulated program: the full regression set (both kernels), plus the
  classic tower, plus d=24 sieved through each inner field GF(2^2),
  GF(2^3), GF(2^4), GF(2^6) and classic: all byte-identical.  Deep chain,
  r = 136279841, s <= 999,999: from scratch to d=28, then d = 29..32 (d=30
  through GF(2^5) with k = 3, d=32 through GF(2^8)) identical to round 1.
  `--selftest-logs tower` to d = 54 with tables built by the device code:
  0 wrong at every degree.
* Compile: clang 18 + ptxas 12.9 (sm_75), both builds, no warnings, no
  spills; tower kernels 107-128 registers.
* What the emulator cannot run: the shared-memory staging (device-only
  code).  On the GPU it is exercised by `--selftest-logs` (the batched
  check at every tower degree), and in production by the usual spot
  verification of 1 log in 256.

## 4. Hardware test plan

    make coarse_sieve coarse_sieve_wide gf2_dlog_selftest
    ./gf2_dlog_selftest                     # ~5 min: "ALL OK"
    ./coarse_sieve --selftest
    ./coarse_sieve_wide --selftest
    ./coarse_sieve_wide --selftest-lowmem 7
    ./coarse_sieve_wide --selftest-logs     # tower lines end "; over GF(2^n)"
    ./coarse_sieve_wide --selftest-logs tower
    ./coarse_sieve_wide --selftest-logs ph
    COARSE_LOGS_KERNEL=r2 ./coarse_sieve_wide --selftest-logs

Regression on tower degrees that changed, against the files of your
round-3 regression (r3_34 .. r3_41; adjust names):

    for d in 36 40; do
      ./coarse_sieve_wide 136279841 $d r4_$d \
          --load-survivors r3_$((d-1)).survivors.txt --min-depth $d
      cmp r4_$d.found.cert r3_$d.found.cert && \
      cmp r4_$d.survivors.txt r3_$d.survivors.txt && echo "d=$d identical"
    done

d=42 (the k = 3 case) and d=44 the same way from whatever d=41 / d=43
survivors files you have.  Then the A/B that pins the gain, same binary,
one tower degree (d=44: minutes; S43 = your d=43 survivors file):

    COARSE_TOWER_SUBN=0 ./coarse_sieve_wide 136279841 44 ab_old \
        --load-survivors S43 --min-depth 44
    ./coarse_sieve_wide 136279841 44 ab_new --load-survivors S43 --min-depth 44
    cmp ab_old.found.cert ab_new.found.cert

Please send the two `done` lines, and a d=48 one if you rerun it.  If the
gain is well short of ~2.5x, the next suspect is the reads of T itself
(see section 7).

## 5. Projections (RTX 2070 SUPER)

Tower: ~0.7-0.9 ns per class -- one random read of T (~0.65-0.75 ns at
the rates your round-3 runs showed) overlapped with ~900 instructions.
PH degrees are unchanged from round 3 (your d=45/47 matched the upper end
of that estimate, so the upper end is used here).

| d | engine | classes | round 3 (measured / est.) | round 4 (est.) |
|---|---|---|---|---|
| 44 | tower/GF(2^11) | 4.0e11 | (round 2: 945 s) | ~5 min |
| 46 | tower (classic) | 1.5e12 | 2,921 s | unchanged |
| 47 | PH | 3.0e12 | 5,411 s | unchanged |
| 48 | tower/GF(2^12) | 5.9e12 | 12,711 s | ~1.1-1.5 h |
| 50 | tower/GF(2^5) | 2.3e13 | ~14 h | ~6-9 h |
| 51 | PH | 4.4e13 | ~1.5 days | unchanged |
| 52 | tower/GF(2^13) | 8.7e13 | ~2.4 days | ~18-23 h |
| 53 | PH | 1.7e14 | ~6 days | unchanged |
| 54 | tower/GF(2^9) | 3.3e14 | ~10 days | ~3.3-4.3 days |
| 55 | PH | 6.6e14 | ~19 days | unchanged |

(Degrees 50..55 still need `--allow-degree-gap`; see section 6 on why
they may be worth it if reducibility is what matters.)

## 6. Your PS questions

**Are smallest factors valuable in themselves?**  Not individually, as
far as I know: their role is the one you describe -- a canonical choice
that makes certificates unique and comparable with `cmp`.  Collectively
they carry some signal: your survivor counts follow
P(no factor of degree <= d) ~ c/d remarkably well (d=32: 3,731,849 /
68,139,920 * 32 = 1.753; d=38: 1.756), the Mertens-type law of the random
polynomial model.  The constant is lower than for random polynomials
without linear factors (~2.2), roughly as a simple model in which a
trinomial is divisible by a given irreducible of degree k with probability
1/(2^k - 1) instead of 2^-k predicts (~1.84) -- the kind of statistics
Brent and Zimmermann used to predict search costs.

**Non-traditional ways to find (not necessarily smallest) factors.**  The
analogue of P-1 exists: x^E mod T for a smooth E, then gcd(x^E - 1, T)
finds the factors f with ord(x mod f) | E.  But ord(x mod f) divides
2^d - 1 (d = deg f), x usually has nearly full order mod f, and so P-1
succeeds only where 2^d - 1 is itself nearly B-smooth -- which beyond
d ~ 100 essentially never happens (it carries the primitive prime factors
of every 2^k - 1, k | d, and those are typically large).  So it finds
nothing tsfactor would care about.  ECM analogues (curves over
GF(2)[x]/T) need group orders of size ~2^d to be smooth -- hopeless at
d >= 272.  The underlying reason is that polynomials have something
integers lack: distinct-degree factorization finds *every* factor of
degree d at the cost of about one squaring per degree, with no luck
needed, so nothing based on group-order smoothness can compete -- and
"smallest first" falls out of DDF simply because small degrees are the
cheapest to look at.  The one non-traditional method that does work is
the coarse sieve itself, which finds factors by enumerating irreducibles
rather than by testing trinomials; with `--allow-degree-gap` at d = 50..55
it produces exactly reducibility certificates that are not guaranteed
minimal.  For a primitivity search that is worth having: each s it
resolves needs neither `factor` nor an irreducibility test.  Its cost
doubles per degree, though (section 5).

**Proving irreducibility without finding a factor.**  For prime r and
trinomials (no linear factors), T is irreducible iff x^(2^r) = x mod T --
the Ben-Or/Rabin test collapses to one condition, which is what `irred`
computes: r squarings, the polynomial counterpart of Lucas-Lehmer.
Modular-composition methods are asymptotically faster in theory, but
their memory and constants rule them out at r ~ 1.4e8, and squaring mod a
trinomial is already linear time; I know of no practical shortcut.
Swan's theorem gives the parity of the number of factors for free, but for
r = 136279841 = 1 mod 8 it only rules out s = 2 (and r - 2).  So, as you
say, finding a factor is almost always the cheaper proof of reducibility,
and the full test is only worth it for the rare survivor.

## 7. Next candidates (not done)

1. **Auto-checkpoint** (your earlier idea; still pending): default
   `<out_prefix>.ckpt`, auto-resume, run identity stored and checked.
2. **Tower: reads of T made local.**  If the tower is now bound by the one
   remaining random read, bucketing the classes of a chunk by the high
   bits of their T index (write ~24-byte records, then finish each bucket
   against an L2-resident slice of T) turns random reads into streaming
   ones -- perhaps another 1.5x on even degrees, at the cost of a
   two-pass kernel structure.  Worth it only if you go well past 49.
3. **PH: Karatsuba bit-sliced products** -- ~15-20% fewer instructions per
   product at d >= 45, perhaps 10% on odd degrees; register pressure (the
   PH kernels are at 228-248 registers in your log) is the risk.
4. d = 34, 38, 46 (m prime): a bit-sliced division in K would drop one of
   their three reads (~1.5x); only matters for a new r.

## Appendix: earlier rounds, for reference

**Round 3 (2026-10-20)**: batched kernel -- FKM enumeration of exactly the
class representatives, 32 classes per thread in bit-sliced form (g^u, PH
powers), Montgomery batch inversion, multi-part PH planner, 16-byte PH
slots, T indexed by log; depth limit 55 with a backward-compatible key;
degree 49 impossible (2^49-1 = 127 * 4432676798593), so runs including it
are refused and 50..55 need `--allow-degree-gap`.  Your results: d=45
1,702 s, 46 2,921 s, 47 5,411 s, 48 12,711 s (both r values alike),
outputs consistent with the oracle and with done ranges.

**Round 2 (2026-10-19)**: the logs path -- one Zech log per class from
tables of ~2^(d/2) entries (tower for even d, Pohlig-Hellman otherwise),
replacing the bucketed path; your times d=41 399 s, 42 222 s, 43 1,646 s,
44 945 s, 45 6,380 s, 46 4,022 s, 47 21,242 s.

**Round 1 (2026-10-18)**: LFSR stepping instead of recomputing each
antilog in the bucketed path (~1000x), plus smallest-fitting bucket
counts and other fixes; byte-identical outputs.  Answers that still stand:
more RAM or disk would not help, nor would hand-written PTX.

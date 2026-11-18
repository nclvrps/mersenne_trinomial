# `factor.cpp` error-checking modifications — hunk-by-hunk

Deliverables: `factor_ec.cpp` (full modified source) and `factor_ec.patch`
(unified diff against the original `factor.cpp`).

> **Status (round 5): compiled, linked, and run** against NTL 11.5.1, GMP and
> gf2x 1.3.0 with `g++ -O2 -fopenmp`. Output is identical to the original
> `factor` on 5781 trinomials in 20 configurations (`regress.sh`); 108/108
> injected single-bit faults and 4/4 corrupted initialisations are caught
> (`sweep.sh`). Rounds 2–3 were also run by Gord at r = 136279841; round 5 has
> not been yet. See `errorcheck_handoff.md` §13 for the full ledger.

Options (round 5: checking is ON by default):

```
-noec      turn error checking off
-ec        accepted; it is the default
-ecw w     run the Gerbicz check every w blocks (default 8)
-ect       per-category cost breakdown at exit
-eci0      skip the check of the h-table initialisation
-ecg       also run each gcd twice and compare
-ecfault N flip one bit at the N-th injection opportunity (testing only)
-ecfaultinit   flip one bit of the first h-table after init (testing only)
```

---

## Hunk 1 — infrastructure (after `NTL_CLIENT`)

Adds the cyclic fold `phi_n` (`ECFOLDW = 61` words = 3904 bits; it was 64 until round 4, see `errorcheck_handoff.md` §5.1), its helpers
(`ec_fold`, `ec_fold_eq`, `ec_fold_add`, `ec_fold_shl1`, `ec_fold_to_gf2x`,
`ec_fold_mul`), the independent reducer `ec_reduce_ref`, and the global flags.

`ec_fold` is a straight word XOR — it is valid *because* `n` is a multiple of
`NTL_BITS_PER_LONG`. `ec_fold_shl1` is a 1-bit rotate, i.e. multiplication by
`x` in `GF(2)[x]/(x^n+1)`.

Round 1's `ec_reduce_ref` is gone; see hunk 2. What replaces it lives in
`ec_words.h` (`ec_reverse_w`, `ec_reduce_lowdiff`, `ec_divides_trinomial`),
with GF2X bridging (`ec_to_words`, `ec_xor_words`, `ec_alloc`) here. The
scratch buffers are `thread_local` because `accumulate` has an OpenMP pragma
over the product tree.

## Hunk 2 — `fastmulmod`

Two checks around the existing body:

1. `phi(t) == phi(a)*phi(b)` on the **unreduced** product from `mul_gen`. An
   exact ring-hom identity, no `mod T` involved, covering the dominant-cost
   operation.
2. `T | (t + c)` together with `deg c < r`, which is equivalent to
   `c == t mod T`.

**Check 2 replaces round 1's independent re-reduction, which was the bug that
hung the program.** The naive reduction loses `(r - s)` degrees per pass, and
production always runs with `r - s` small, so it needed 231769 passes. The
divisibility test moves to the reciprocal trinomial when `s > r/2`, where the
exponent is small again, and is then always 2 passes. It is also strictly
stronger: an error `e` in `c` is missed only if `T | e`, and `deg e < r` forces
`e = 0`. Handoff §5.2 has the derivation.

The unreduced product is stashed in a plain word buffer rather than a `GF2X`
copy, which avoids an allocation; `fastrem` reduces in place, so the copy
itself is unavoidable. The word primitives live in `ec_words.h`, shared with
`ecbench` and `ectest2` so that the thing benchmarked and the thing tested are
the same code.

## Hunk 3 — `usage()`, option parsing, and the `choose_m` guard

Six options now. One thing here is not cosmetic: `choose_m` times `fastmulmod`
in a loop to estimate M/S and pick the block size. With checking on it would
have measured an inflated M and chosen the wrong `m` — a silent
performance-and-correctness trap, since `m` also sets the block structure the
Gerbicz check runs over. EC is suspended across the calibration and restored
after. Your production runs pass `-m` explicitly and never enter it, but a run
without `-m` would have.

## Hunk 4 — `init_h`: closed-form check (round 5; was duplicate-and-compare)

The Gerbicz check certifies how the h-table *evolves*, not where it starts.
Rounds 2–4 therefore ran `init_h` twice and compared, which your logs priced
at 5.4–7.6 s per `s` — 69% of all checking cost at `-skip 55`.

Round 5 replaces that with `ec_init_closed_form`. The table holds, in
`h[j]`, the sum of `y^e` over the exponents `e < 2^m` of Hamming weight `j+1`
(`y = y_k = x^(2^k) mod T`), so the sum `S` of all its entries is the
geometric series `y + y^2 + ... + y^(2^m - 1)`, and

```
        (1 + y_k) (1 + S)  =  1 + y_k^(2^m)  =  1 + y_(k+m)      mod T.
```

Both sides are computed from scratch: `k + m` squarings of one residue and one
`fastmulmod` (itself fold-checked), against the `m (k + m/2)` squarings of a
second `init_h`. A table that fails is rebuilt and checked again; five
failures in a row abort, as elsewhere. An error in the check's own squarings
makes the identity fail too, which costs a rebuild and cannot make a bad
table pass.

Round 3 of this document said a cheap check "only covers one of the `m`
entries" (it looked at `h[0]` alone). The sum of all entries is what has the
closed form.

## Hunk 5 — retry wrapper around the interval body

The block loop, `assert`, and `accumulate` are wrapped in
`for (ec_attempt = 0; ; ec_attempt++)`, with `hchk`/`achk` capturing the
interval-start state beforehand. On `ec_fail`, `h` and `a` are restored and the
interval is redone; after `ec_maxretry = 5` consecutive failures the program
stops with a clear message rather than spinning.

Points to check when you compile:

* `GF2XVec C` is now declared **inside** the retry loop so each attempt gets a
  clean vector. (`GF2XVec::SetSize` rejects reinitialisation, hence the move.)
* The pre-existing `goto gcds` now jumps *out of* the retry loop. That is legal
  C++ (leaving scopes is fine; only jumping *into* a scope past a non-trivial
  initialisation is not), but it is the thing most likely to draw a compiler
  complaint. If it does, replace `goto gcds` with a flag test.
* Rollback costs one extra h-table copy (`m*r/8` bytes ≈ 780 MB at
  `r = 136279841`, `m = 46`) **only when `-ec` is given**. If that is
  unaffordable, reuse the existing `hold` (it already holds an interval-start
  copy whenever `q > 1 && deg(a) == 0`) and skip `hchk` in that case.

## Hunk 6 — fold shadow on the Horner construction of `C[z]`

Maintains a predicted `phi(C[z])` through the Horner recurrence:

```
phi(C*x mod T) = x*phi(C) + wrap*phi(T)
```

where `wrap = coeff(C, r-1)` sampled *before* `fastmulbyxmod`. The predicted
fold is compared against a direct fold of the finished `C[z]`.

**The correction term is `phi(T)`, not `phi(x^s+1)`.** `C*x mod T` equals
`(C<<1) + wrap*(x^r + x^s + 1)`, and the `x^r` term does not vanish under
folding. Getting this wrong produces a checker that fires on every correct
block — a safe failure mode, but a baffling one. (Ask me how I know.)

## Hunk 7 — the Frobenius-linear Gerbicz check

The heart of it. Around the existing h-update loop:

* `add(ec_D, ec_D, ec_S)` before the table advances (window accumulator).
* `add(ec_Sn, ec_Sn, h[j])` after each `h[j]` finishes its `m` squarings.
* Every `ec_window` blocks (and always on the last block of the interval),
  compute `sigma(D)` with `m` squarings and test

  ```
  sigma(D) + D  ==  S_{z0} + S_{z1}
  ```

The `sigma` loop mirrors the existing pairing of `fastsqr` calls so it works on
both the `usefastsqr` and `SqrMod` paths.

Overhead: `m` squarings per `ec_window` blocks out of `m^2` per block, i.e.
`1/(w*m)` of the squaring work — 2.2% at `w=1, m=46`, 0.27% at `w=8`. The `m`
XOR-accumulates per block are the other ~2%, and are the part worth fusing into
the squaring's output store if you care.

## Hunk 8 — gcd repeat (`-ecg`)

A stopgap: run `FastGCD` twice, compare, take a third opinion on disagreement.
Catches transient faults, not a persistently bad input.

**This is not the recommended production check.** The right one is a Bezout
certificate — see handoff §6. `HalfGCD2` in `halfgcd.cpp` already takes an
`extended` flag that produces the transformation matrix; `FastGCD` deliberately
passes `extended = 0` to avoid the `strassen_mul` cost. Plumbing the cofactors
out gives `u*a + v*T = g`, verifiable with four folds, which *proves*
`gcd(a,T) = 1` instead of merely agreeing with itself. Batch it (one extended
GCD per `s` value, §6.2) and the cost disappears.

## Hunk 9 — banner and summary

Reports the EC configuration at startup and the detection count at exit.

---

## Not implemented (deliberately)

* **Bezout GCD certificate** — needs `halfgcd.cpp` work, see above. Biggest
  remaining win.
* **Control-flow guards** (handoff §9.3) — redundant copies of `k`, `q`, `z`,
  and assertions that `k2 == k + q*m - 1` and that the block count matches `q`.
  Cheap, and covers a hole no algebraic check can see. Should go in.
* **Fused `S_z` accumulation** — currently a separate `add`. Fusing it into
  `fastsqr`'s output loop would take the h-check overhead from ~4% to under 1%.
* **`phi(q)` instrumentation of `fastsqr`** (handoff §5.2b) — only needed if
  you want per-squaring deterministic checking instead of per-block. The
  Gerbicz check already covers this ground more cheaply.

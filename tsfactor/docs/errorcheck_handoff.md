# Low-cost error checking for trinomial factor search over GF(2)
## Analysis + handoff document — Round 4

> **Round 4 implemented the scheme in `tsfactor` (CUDA), on by default**, and
> in doing so found four things that change this document:
>
> 1. **The h-table digest has a closed form** (§4.7). The initialisation can be
>    verified from first principles with `k + m` squarings and one modmul.
>    Round 3's conclusion that no cheap `init_h` check exists (old §12) was
>    **wrong**.
> 2. **The gcd can be certified without computing cofactors** (§6.3): verifying
>    each step of the half-gcd reduction chain with folds is enough to prove
>    "no factor here", at a cost of a few percent of the gcd.
> 3. **The gcd engine got faster than NTL's** (§6.4): a half-gcd matrix only
>    has to be multiplied into the *low* half of the operands. Certified and
>    24–32% faster than the unchecked NTL gcd in the sandbox (32% at
>    136 Mbit); it is now tsfactor's default.
> 4. **The fold length must not be a power of two** (§5.1). It is now 61 words
>    in both programs.
>
> §13 remains the ledger of what is verified by execution and what is not.
> For the tsfactor integration itself read `tsfactor_handoff_EC.md`.

**Status: implemented in both programs.** `factor_ec` (CPU, opt-in `-ec`):
measured 1.8% of wall clock at r = 136279841 excluding the duplicate
`init_h`. `tsfactor` (GPU, on by default, gcd certified by default): validated
under CPU emulation including fault injection at 14 sites and one run at the
production exponent; **not yet compiled by nvcc or run on a GPU**.

The core fact is unchanged: in characteristic 2 the squaring map is
**additive** as well as multiplicative, so the Gerbicz "product of
checkpoints" becomes an **XOR of checkpoints**, cheaper than the integer
version and deterministic for single faults.

---

## 0. Reading guide / file map

| File | What it is |
|---|---|
| `errorcheck_handoff.md` | This document: the mathematics, the measurements, the status ledger. Start here. |
| `tsfactor_handoff_EC.md` | The tsfactor integration: where each check sits, recovery, options, tests. Append it to `tsfactor_handoff.md`. |
| `TESTPLAN_EC.md` | What to run on the real GPU, in order, and what to send back. |
| **tsfactor** | |
| `gf2_ec.h` | Fold algebra (host) + checking kernels (device). Header-only. Replaces round 3's `ec_cuda.cuh`. |
| `tsfactor.cu`, `gf2_gcd.h`, `gf2_gcd_ntl.cpp` | The implementation; the certified gcd lives in the last two. |
| `gcdbench.cpp` | CPU-only timing of plain / certified / old-engine gcd. `make gcdbench` |
| `gcdstress.cpp` | CPU-only randomized cross-check of every gcd path against NTL. `make gcdstress` |
| `tests/ts_regress.sh` | tsfactor command-line regression: 33 configuration runs vs the original binary, injected faults, interrupt/resume. |
| **factor_ec** | |
| `factor_ec.cpp`, `factor_ec.patch`, `factor_ec.patch.md` | Patched `factor.cpp`, the diff, and hunk-by-hunk commentary. Round 4 changed one line: the fold is 61 words. |
| `ec_words.h` | Word-level primitives shared by `factor_ec`, `ecbench`, `ectest2`. |
| `ectest.cpp`, `ectest2.cpp` | Unit tests of the word primitives against bit-level references. |
| `ecbench.cpp` | Per-primitive cost at any (r,s), no NTL. |
| `regress.sh`, `sweep.sh` | factor_ec vs factor on 2700 trinomials; the 150-opportunity fault sweep. |
| **model** | |
| `gecsim_core.h`, `gecsim.cpp`, `gecsim_results.txt` | Dependency-free model of the algorithm with a fault injector (round 1). |
| **superseded, not shipped** | `tsfactor_ec_design.md`, `ec_cuda.cuh`, `ectest_gpu.cpp`, `eccuda_emul_test.cpp` (round 3's design for a frozen tsfactor: 64-word fold, shift-fold quotient). The tsfactor selftest now covers everything they tested. |


Terminology used throughout:

* `T = x^r + x^s + 1`, the trinomial; `R = GF(2)[x]/(T)` (a **ring**, not a
  field — `T` is generally reducible, that is the whole point).
* `Frob(u) = u^2`; `sigma = Frob^m : u -> u^(2^m)`.
* `h[j]`, `j = 0..m-1`: the h-table, `h[j] = e_{j+1}(x^(2^K), ..., x^(2^(K+m-1)))`,
  the (j+1)-st elementary symmetric function of `m` consecutive Frobenius
  powers of `x`. `K = k + z*m` for block `z`.
* `phi_n : GF(2)[x] -> GF(2)[x]/(x^n + 1)`, the **cyclic fold** checksum. With
  `n` a multiple of 64 this is a pure word-wise XOR fold: `out[i] ^= in[i mod (n/64)]`.
  Both programs use `n = 3904` (61 words); §5.1 says why not 4096.
* `y_t = x^(2^t) mod T`. Note `1 + y_t = (1 + x)^(2^t)`.

---

## 1. What the algorithm actually is (needed to see where checks attach)

Decoded from `factor.cpp` `main()` + `init_h()` + `accumulate()`:

```
h[j] <- e_{j+1}(x^(2^k), ..., x^(2^(k+m-1)))          [init_h, one-time]

for each interval [k, k2], k2 = k + q*m - 1:
    for z = 0 .. q-1:                                  # block z covers m degrees
        C[z] <- 1                                      # Horner:
        for j = 0 .. m-1: C[z] <- C[z]*x mod T ; C[z] += h[j]
        #  => C[z] == prod_{i<m} (x + x^(2^(K_z+i))) mod T,  K_z = k + z*m
        for j = 0 .. m-1:                              # advance the h-table
            h[j] <- h[j]^(2^m)                         # m squarings each
    a <- a * prod_z C[z]  mod T                        # product tree
    g <- gcd(a, T)                                     # FastGCD
    if deg g > 0: factor found in [k, k2] -> refine (fineDDF/CanZass)
```

Two facts that drive everything:

**(F1)** The h-table evolves by a single map applied componentwise:
`h[j]^(z+1) = sigma(h[j]^(z))` with `sigma = Frob^m`. It is an *iterated
Frobenius chain*, structurally identical to the `u(t+1) = u(t)^2` chain of a
Proth/PRP test. This is where Gerbicz applies.

**(F2)** `a = prod_{d=k}^{k2} (x^(2^d) + x) mod T`. The whole DDF is just a
product over degrees of `(Frobenius power) + x`, so *everything* is downstream
of the Frobenius chain. If the chain is right and the products are right, the
answer is right.

Cost split, using the program's own tuning rule `m ~= sqrt(M/S)` (`M` = cost of
a modmul, `S` = cost of a modsqr — see the `choose_m` code):

* h-table squarings: `q*m^2` squarings per interval `= q*M` in cost units.
* product tree: `q` modmuls per interval `= q*M`.
* So **squarings and products cost about the same**, roughly half each, with
  the GCD amortised on top. A check that covers only one of them covers half
  the exposure.

---

## 2. Threat model — and why this matters more than it looks

A run either (i) reports a factor, or (ii) reports "irreducible"/"no factor
below D". Case (i) is **self-verifying**: a claimed factor `P` is checked by one
division. So a corrupted run that *invents* a factor is harmless — it gets
caught downstream for free.

The dangerous case is the opposite one. A bit flip in `h[j]` makes the h-table
algebraically meaningless from that block onward. `a` becomes an essentially
random element of `R`, and a random element of `R` is coprime to `T` with
overwhelming probability. So the GCD returns 1, the program says "nothing
here", and **marches on silently for the rest of the run**. The output is a
false negative — "no factor of degree < D" — which is precisely the claim that
is expensive to double check and easy to believe.

This is confirmed experimentally rather than assumed. Sweeping every single
possible bit-flip position in a small run (experiment 3):

```
   squaring (h-table)       faults injected = 144
      unchecked: 128 SILENT wrong (factor missed), 0 self-evident, 16 harmless
   idle h-table memory      faults injected = 36
      unchecked: 32 SILENT wrong (factor missed), 0 self-evident, 4 harmless
```

**Zero** of the consequential faults produced a self-evidently bogus answer.
Every single one produced a silent missed factor. So the value of error
checking here is not "avoid wasted work" — it is "avoid publishing a wrong
negative result", which for a search whose entire output is negative results is
the whole ball game.

---

## 3. The obstruction you would hit first (and why the usual trick fails)

The standard cheap check for modular arithmetic is a **homomorphic shadow**:
carry the computation in a small quotient ring alongside the big one. For
integers mod `N` you shadow mod a small prime. The polynomial analogue would be
a ring homomorphism `R = GF(2)[x]/(T) -> S` with `S` small.

Such a homomorphism exists iff `S` contains a root of `T`, i.e. iff you know a
factor of `T` of small degree. **That is exactly the thing the program is
trying to compute.** Concretely: `GF(2)[x]/(T, x^n+1) = GF(2)[x]/gcd(T, x^n+1)`,
and since `T` has no small factors the gcd is 1 and the shadow ring collapses
to zero.

So: no homomorphic shadow, ever, for any small modulus. Anyone attacking this
problem will waste a day on that idea; don't. What works instead is

* **Gerbicz-type self-consistency inside `R`** (§4), for the Frobenius chain, and
* **linear/fold checks with explicit quotient bookkeeping** (§5), for the
  operations where the reduction mod `T` is materialised and therefore
  correctable.

---

## 4. The core result: a Frobenius-linear Gerbicz check on the h-table

### 4.1 Classical Gerbicz, for reference

For `u(t+1) = u(t)^2 mod N`, with block length `L`, Gerbicz accumulates
`d(t) = u(0)*u(L)*...*u(tL)` and uses

```
d(t+1) = d(t)^(2^L) * u(0)          [multiplicative telescoping]
```

Cost: `n/L` mulmods to maintain `d`, plus `L` squarings per check. Detection is
essentially certain because a corrupted `u` enters the product and cannot
cancel.

### 4.2 The characteristic-2 version

In `R`, `sigma(u) = u^(2^m)` is a ring **endomorphism**, so it is multiplicative
*and additive*:

```
sigma(u + v) = (u+v)^(2^m) = u^(2^m) + v^(2^m) = sigma(u) + sigma(v)
```

That means the Gerbicz accumulator can be an **XOR** instead of a product.
Define the per-block digest

```
S_z := XOR_{j=0}^{m-1}  h[j]^(z)
```

Then, because every component advances by the same `sigma`:

```
        sigma(S_z) = S_{z+1}                                  (*)
```

and telescoped over a window of `w` blocks with `D := XOR_{z=z0}^{z1-1} S_z`
(`z1 = z0 + w`):

```
        sigma(D) + D = S_{z0} + S_{z1}                        (**)
```

Identity (*) was verified numerically for `m = 2,3,4,6,8` over 10 consecutive
blocks each — see experiment 1.

### 4.3 Why this is *better* than the integer case, not just different

1. **The accumulator costs XORs, not multiplies.** Classical Gerbicz pays
   `n/L` mulmods to maintain `d`. Here `S_z` costs `m` XOR-accumulates per
   block against `m^2` squarings — and if you fuse the accumulate into the
   store of the final squaring of each `h[j]` (trivial in a CUDA kernel,
   easy in `fastsqr`'s output loop), it is nearly free.
2. **Detection is deterministic, not probabilistic.** Suppose a fault during
   block `z_e` leaves `h[j_e]` wrong by `eps != 0`. Then `S_{z_e+1}` is wrong by
   exactly `eps`, while `sigma(S_{z_e})` is computed from an uncorrupted
   `S_{z_e}` and is right. In the telescoped form the discrepancy works out to
   exactly `eps` as well, independent of *where in the window* the fault landed.
   A single fault is therefore **always** caught. There is no random
   coefficient vector, no `2^-t` miss probability, no tuning knob.

   (Note the one thing you must not do: compress the sum with a *random*
   0/1 vector `c_j` instead of all-ones. That is the natural instinct and it
   costs you a factor of 2 in detection per fault, for no saving. All-ones is
   both cheaper and strictly better. The only faults all-ones misses are two or
   more faults whose error polynomials cancel exactly — which requires
   identical `eps` in different components in the same window.)
3. **It also catches idle memory corruption.** If `h[j]` rots in DRAM/VRAM
   between blocks, the next block's `S` picks up `sigma(e)` while
   `sigma(S_prev)` does not. Experiment 3, row "idle h-table memory": 36/36
   caught. This matters for consumer GPUs without ECC.

### 4.4 Cost

Per block: `m` XOR-accumulates (fusible, ~0). Per window of `w` blocks: one
`sigma` = `m` squarings, plus 2 XORs of the digests.

```
overhead  ~=  1 / (w * m)   of the squaring work
```

For the production configuration (`m ~= 46`): `w=1` gives 2.2%, `w=8` gives
0.27%, `w=32` gives 0.07%. Experiment 4 confirms detection stays at 100% as `w`
grows — the telescoping is free in detection terms, it only lengthens the
rollback distance.

### 4.5 Memory

Two extra polynomials of degree `< r` (`D` and the window-start digest), i.e.
`2*r/8` bytes = **34 MB at r=136279841**. Not `m` extra polynomials — the
all-ones compression is what buys this. Compare: the h-table itself is
`m*r/8` = 780 MB.

Rollback state is the only heavy part: to redo a window you need `h` as of the
window start (`m` polynomials, 780 MB). Options, in order of preference:

* **Roll back to the interval start.** `factor.cpp` already keeps `hold` (the
  `copy_h` for fineDDF). Zero extra memory; you redo up to one interval. Since
  faults are rare, redoing an interval occasionally is free in expectation.
* **Dedicated window checkpoint in host memory** (GPU case): copy `h` to host
  every `w` blocks. At `w=8`, `m=46`, ~0.9 s/block, a 780 MB PCIe copy is well
  under 1% and gives short rollbacks.
* Never checkpoint `h` to disk on the check path; it is not needed.

### 4.6 What it does *not* cover

The chain certifies that the h-table **evolved** correctly from wherever it
started. It says nothing about the starting table, and nothing about a table
that is *restored* from a corrupted copy: a wrong table that evolves correctly
passes forever. So the chain must be anchored (§4.7) and every copy that can
become a starting point — snapshot, checkpoint, upload — must be verified when
it is made and when it is used (tsfactor does both with folds).

Two faults in the same window whose error polynomials are *identical* cancel
in the XOR. Independent faults never do this; a **stuck bit** can (the same
cell forced the same way in two blocks). It then cancels in windows where it
manifested an even number of times and is caught in the others, so a stuck
fault is still found almost at once — and it fails every retry, which is how
it is told apart from a transient. `w = 1` removes the window altogether.

### 4.7 The digest has a closed form (round 4)

`S` is the sum of the elementary symmetric functions `e_1..e_m` of
`y_k, ..., y_{k+m-1}`, and `sum_{j=0}^{m} e_j = prod_i (1 + y_i)`. In
characteristic 2, `1 + y_i = 1 + x^(2^i) = (1 + x)^(2^i)`, so

```
        1 + S_k  =  prod_{i<m} (1+x)^(2^(k+i))  =  (1 + x)^(2^k (2^m - 1))     mod T
```

and therefore

```
        (1 + y_k) * (1 + S_k)  =  1 + y_{k+m}                         (***)
```

Everything on both sides is computable without the h-table: `1 + y_k` is
`k` squarings of the residue `1 + x`, `1 + y_{k+m}` is `m` more, and the left
side is one modular multiplication. So the table at any block start `k` can be
verified **from first principles** for

```
        k + m squarings of ONE residue  +  1 modmul
```

against `k*m + m^2/2` squarings to build it: about `1/m` plus one modmul.
(Verified numerically for 60 random `(r, s, m, k)`, and in the tsfactor
selftest on real tables, including that one flipped bit breaks it.)

Is a check of the *sum* enough for a table of `m` entries? For a single fault,
yes. The Pascal-triangle build is Frobenius-semilinear, so an error `eps`
injected anywhere propagates as `E(Z) = eps * P(Z)` in the generating
function, and the digest sees `E(1)`, which is `eps` times a product of
factors `(1+x)^(2^i)` — units, because `T(1) = 1`. A nonzero `eps` cannot
vanish.

This replaces "run `init_h` twice". How much it saves depends on what a
modmul costs relative to a squaring, because the one modmul dominates:

* **factor_ec** (`m = 14`, `k = 56`): a modmul is ~200 squarings, the init
  ~870 squarings, the check ~270: **~30% instead of 100%**, i.e. roughly 2 s
  instead of the 5.5–7.6 s measured for the duplicate. factor_ec still runs
  the init twice (`-eci0` turns that off) — porting (***) there is a small,
  self-contained job, and the one place where it buys real time.
* **tsfactor** (`m ~ 30`): the GPU squaring is a single fused kernel
  (~0.26 ms) and a modmul is ~900 of them. The init is squarings only,
  `m(k + m/2)` of them — about **half a second** at `k = 56` — and the check
  is ~0.25 s: 46% of the init at `k = 56`, 14% at `k = 280`, tending to `1/m`
  on a deep resume, where the init takes minutes. Either way it is noise
  next to a gcd, so the init is simply not the issue in tsfactor that it was
  in factor_ec, and `--eci0` has no claim to be the default.

**Correction.** Round 3 examined a different identity,
`H + y_m G = H^2 + y_0 G^2`, which needs two modmuls, concluded that "any
check that touches even one modular multiply loses to simply running `init_h`
twice", and recommended against it. That conclusion was wrong: (***) needs one
modmul and no squaring of table-sized objects, and wins by 2x to `m`x.

(***) also means the digest chain is nothing but the Frobenius orbit of the
fixed element `(1+x)^(2^m - 1)`. An *absolute* check is therefore possible at
any time during the scan by keeping one extra residue `v = 1 + S` advanced by
`m` squarings per block (`1/m` of the squaring work) and comparing it with the
table's digest. tsfactor does not do this — the window check costs several
times less on the critical path — but under `--pend-max 1` the GPU idles through every
gcd and that chain could be advanced then, for free. Noted as an option; it
would also make stuck-bit faults deterministic to catch.

---

## 5. The products substep: cyclic-fold checks with quotient bookkeeping

`fastmulmod` does `mul_gen(t, a, b)` (full, unreduced, degree `< 2r`) and then
`fastrem(t)` (reduce in place). Two separately checkable stages.

### 5.1 The multiplication (the expensive half)

`phi_n` is a genuine ring homomorphism on `GF(2)[x]` — the obstruction of §3
only bites when you try to push it through `mod T`. On the **unreduced**
product there is no `mod T`, so:

```
        phi(t)  ==  phi(a) * phi(b)      in GF(2)[x]/(x^n + 1)
```

is an exact identity, checkable with 3 streaming XOR-folds plus one tiny
`n`-bit cyclic multiply. Against an `r = 1.4e8`-bit gf2x/Cantor-FFT multiply,
that is noise (< 1–2%).

Detection: the check misses an error `e` iff `(x^n + 1) | e`. Since
`(x^n+1)*g = g + x^n*g` has support spanning at least `n+1` positions, **any
nonzero error confined to a window of fewer than `n` consecutive bit positions
is caught with certainty.** Single bit flips, single corrupted words, single
corrupted cache lines: all caught, deterministically, for any `n >= 4096`.
Experiment 5: 0 survivors out of 100000 single-bit errors and 100000
single-word bursts, at every `n` tested.

Choose `n` a multiple of 64 so the fold is a pure word XOR, **and not a power
of two**: both programs use `n = 3904 = 64 * 61`. Rounds 1–3 used 4096 and
treated the choice as cosmetic. It is not. With `n = 2^12`, `x^n + 1 =
(x+1)^4096`, and any error whose pattern repeats with a power-of-two period
folds onto itself an even number of times and **cancels exactly**:

* a stuck or marginal bus lane flips the same bit of every 8th (or 2^k-th)
  word. Under a 64-word fold those land in only 8 residue classes, and each
  survives only as a parity: a 1-in-256 chance of total cancellation;
* an additive-FFT fault in a sub-transform of 2^L points can add the *same*
  value across a power-of-two block of coefficients, which a 4096-bit fold
  sums to zero whenever the block is longer than the fold.

With 61 words (61 is prime, and 2 is a primitive root mod 61, so
`x^3904 + 1 = ((x+1) * Phi_61)^64` with `Phi_61` irreducible of degree 60) a
power-of-two-periodic pattern spreads over all 61 words and has to cancel in
every one of them. The burst guarantee above is unchanged.

### 5.2 The reduction — divisibility, not re-reduction

**Round 1 got this wrong and it is worth understanding why, because the same
trap is waiting in `tsfactor`.**

`t mod T` is linear-time, an order of magnitude cheaper than the multiply that
precedes it, so the obvious check is to reduce a second time with an
independent routine and compare. That is what round 1 shipped, and it does not
work. The naive reduction folds the top block down,

```
        w = w_H x^r + w_L   ->   w_L + w_H x^s + w_H
```

which lowers the degree by exactly `(r - s)` per pass. That is 2 passes when
`2s <= r`, and `r/(r-s)` passes otherwise. **Production is always in the second
regime**: `factor` flips `s -> r-s` whenever `r` is odd and `s` is even so that
`fastsqr` becomes available (`main`, the "flip again if necessary" branch), and
that flip lands on `s > r/2` by construction. For `r = 136279841, s = 588` the
working exponent is `136279253`, `r - s = 588`, and the loop needs **231769
passes per `fastmulmod`** — roughly 54 minutes for a single call.

`gecsim` never caught this because it ran at `r = 1279, s = 345`, where
`r - s = 934` and the loop terminates in 2 passes. The bug is invisible at any
`(r,s)` whose *shape* differs from production, and the shape, not the size, is
what matters. **Test the regime, not just the arithmetic.**

The replacement does not re-reduce at all. It tests divisibility:

```
        c == t mod T    <=>    T | (t + c)    and    deg c < r
```

and when `s > r/2` it moves the test to the reciprocal, where the exponent is
small again. Writing `rev_D(f) = x^D f(1/x)` for `deg f <= D`:

```
        T | w    <=>    T* | rev_D(w),      T* = x^r + x^(r-s) + 1
```

exact in **both** directions, because `rev_D` is an involution on degree-`<=D`
polynomials, it is graded-multiplicative (`rev_{a+b}(fg) = rev_a(f) rev_b(g)`),
and `rev_r(T*) = T`. `D` may be padded freely to a word multiple, since that
only multiplies by a power of `x` and `T*` has nonzero constant term. Reversal
is then a pure word operation — reverse the word order, reverse the bits in
each word, no cross-word shifting — and the reduction mod `T*` is always 2
passes.

This is also **strictly stronger** than the check it replaces. An error `e` in
`c` is missed only if `T | e`, and `deg e < r = deg T` forces `e = 0`. So
*every* error in the reduction step is caught, deterministically — not merely
single bit flips.

Measured at `r = 136279841, s = 136279253` (`ecbench`):

```
        bit-reverse, 2r bits                    9.77 ms
        ec_divides_trinomial (reverse+reduce)  37.52 ms
        whole EC cost of one fastmulmod        59.9  ms
```

against roughly 1.4 s for the multiply itself.

**The closed-form quotient of the old §5.2(b) has the same defect.** The
derivation `q = t_H + (t_H >> (r-s))` assumes the 2-pass regime; for `s > r/2`
it is only a partial quotient. If you want the fold-with-quotient formulation
(for `fastsqr`, which never materialises the unreduced square), derive it in
the reciprocal domain, or do not use it — §4 already covers the squarings
better and more cheaply.

### 5.3 The Horner step

`C[z] <- C[z]*x mod T ; C[z] += h[j]`. Multiply-by-x wraps exactly one
coefficient, and `C*x mod T = (C<<1) + wrap*T` **exactly**, so

```
        phi(C*x mod T) = x*phi(C) + wrap*phi(T)
```

Maintain the predicted `phi(C[z])` through the whole Horner recurrence at a cost
of `m` tiny `n`-bit operations plus `m` folds of the `h[j]` (which you want
anyway, see §7), then compare against a direct fold of `C[z]` at the end.
Deterministic for the usual burst classes. Experiment 3: 72/72 caught.

*(Implementation warning, learned the hard way while writing `gecsim`: the
correction term is `phi(T)`, not `phi(x^s + 1)`. Getting this wrong produces a
checker that fires constantly on correct data — which is at least a safe
failure mode, but a confusing one.)*

---

## 6. The GCD substep: a complete certificate

The gcd looked like the opaque part. It is the one substep that can be
**proved** correct rather than merely checked, and round 4 found a way to do
it that needs no cofactors at all.

### 6.1 What has to be proved

Only one direction is dangerous. `gcd(A, T) != 1` is self-verifying: the
factor it leads to either divides `T` or it does not (tsfactor now checks
this for every reported factor, with `x^r mod f` by repeated squaring). The
claim that needs a proof is `gcd(A, T) = 1` — "no factor in this interval" —
because a gcd that goes wrong part-way degenerates into the gcd of two
unrelated polynomials, which is 1.

### 6.2 The Bezout certificate (rounds 1–3), and what it costs

An extended gcd gives `u*A + v*T = d`, an exact identity in `GF(2)[x]`, so

```
        phi(u)*phi(A) + phi(v)*phi(T)  ==  phi(d)
```

is checkable with folds, and `d = 1` proves coprimality. This is what
`tsfactor --ecg-xgcd` does, with NTL's `XGCD` as the untrusted prover.
Measured (sandbox, NTL 11.5.1 + distro gf2x, 0.5–16 Mbit): `XGCD` takes
**1.41–1.43x** the time of `GCD`. The verification itself is four folds. It
is kept as an independent reference, and as the safety net if the chain of
§6.3 ever declines.

### 6.3 Certifying the reduction chain instead (round 4)

A half-gcd computation transforms the working pair by a sequence of steps of
just two kinds,

```
        (a, b) <- M * (a, b)            M a 2x2 polynomial matrix
        (a, b) <- (b, a - q*b)          one division step
```

and each is an exact `GF(2)[x]`-linear combination with no `mod` anywhere, so
each can be verified with folds:

```
        phi(a') == phi(M00) phi(a) + phi(M01) phi(b)      (and likewise b')
        phi(rem) == phi(a) + phi(q) phi(b)
```

**Lemma.** If every step satisfies its identity, every common divisor of the
inputs divides every later pair. Hence `gcd(A, T)` divides the final result.

That is all that is needed, and three things follow that were not obvious:

1. **No determinant check.** The steps do not have to be unimodular. A
   corrupted `M`, applied through a *verified* step, can only add spurious
   factors to the result; it cannot remove a true one. Spurious factors do
   not divide `T`, so the downstream factor check rejects them.
2. **No cofactor accumulation.** The product of the `M`s is never formed.
   This is where an extended gcd spends its extra time.
3. **Only the top level needs checking.** The recursive calls of a half-gcd
   work on truncated copies and exist only to produce `M`. By (1), `M` may be
   *anything*. So the recursion — nearly all of the code and the small-operand
   arithmetic — is outside the trusted base. Only the handful of full-size
   transformations of the real `(a, b)` are verified.

The folds are **carried**: `phi(a)` is computed once, when `a` is produced,
and reused as the input fold of the step that consumes it. It is never
recomputed in between, so an operand that changes in RAM between two steps
breaks the next identity. The finish (the hybrid engine hands a pair of a few
Mbit to NTL) is certified by §6.2 on that small pair, and the final answer is
re-folded against its carried fold before it is returned.

Because the big products are verified on the host whoever computed them, this
also covers the **GPU multiplications** of `--gpu-gcd`, which had no
protection at all.

Measured: certification adds between 1% and 9% to the engine in the runs of
§6.4 (the folds are linear-time and the extended finish is on a pair a small
fraction of the size; the larger figures include timing noise). Selftest: 75
certified gcds equal to NTL's; 600 runs with one fault at step N = 1..40 —
every fault that fired produced "certificate failed", never a certified wrong
answer.

### 6.4 Split application: the certified gcd is also the faster one (round 4)

The engine (tsfactor's `hyb::hgcd`, written in an earlier round for
`--gpu-gcd`) was 30–40% *slower* than NTL's gcd on CPU multiplications. The
reason turned out to be one line. `hgcd` recurses on the top half of the
pair — in place, on a copy — and gets back a matrix `R`; it then applied `R`
to the full-length operands. But with `a = (a >> k) x^k + (a mod x^k)`,

```
        R (a, b)  =  x^k * R (a >> k, b >> k)  +  R (a mod x^k, b mod x^k)
```

and the first term is exactly the pair the recursion has just reduced. So only
the low `k` bits need multiplying. The engine's largest products shrink from
`(n/4 x n)` and `(n/4 x 3n/4)` to `(n/4 x n/2)` and `(n/4 x n/4)`. In balanced
products of size `n/4` per recursion level:

```
        old engine        36  (28 at the top level)
        NTL's HalfGCD     24  (28 at the top level)
        split application 20  (12 at the top level)
```

(`gf2x/apps/halfgcd.cpp`, which factor_ec uses, already works this way.) The
identity verified by §6.3 is that of the *whole* step, however `a'` was
assembled, so the certificate is unaffected.

Measured, sandbox (one core, 2.1 GHz Xeon, distro gf2x 1.3.0, **untuned**):

```
                                  4.0 Mbit              16 Mbit
   NTL GCD                         5.43 s               39.06 s
   CERTIFIED, NTL XGCD             7.63 s  +40.6%       56.00 s  +43.4%
   engine, full apply (old)        7.08 s  +30.5%       55.02 s  +40.9%
   engine, split apply             3.95 s  -27.2%       28.38 s  -27.3%
   CERTIFIED chain                 4.15 s  -23.5%       28.61 s  -26.7%

                                  136 Mbit (production size, finish 4 Mbit)
   NTL GCD                         972.0 s
   engine, split apply             605.8 s  -37.7%
   CERTIFIED chain                 657.7 s  -32.3%
```

The ratio follows from the multiplication counts and should carry over to a
tuned gf2x, but that is an inference: `gcdbench` measures it on the machine
at hand. Correctness: `gcdstress`, 3100 randomized inputs chosen to reach the
corner cases (prescribed quotient sequences, planted factors, degenerate
pairs), four engines against NTL, no mismatch.

### 6.5 Which to use

| tsfactor | what runs | vs NTL's gcd (sandbox) |
|---|---|---|
| default | certified chain, CPU multiplications | −24% to −32% (−32% at 136 Mbit) |
| `--gcd-mt N` | the same, with the 4 products of a matrix application and the 8 of a matrix product on N threads | correct (cross-checked, ThreadSanitizer clean); 1.29x from 2 threads on the sandbox's two virtual CPUs, unmeasured on real cores — the recursion, division steps and small products stay sequential |
| `--gpu-gcd` | certified chain, large products on the GPU | unmeasured since the split; was neutral before it and before the FFT tuning |
| `--ecg-xgcd` | one verified NTL `XGCD` | +42% |
| `--ecg-dup` | NTL twice, compared | +100% (free in wall time with a spare core); transient faults only |
| `--gcd-ntl` | NTL, **unchecked** | 0 |

For `factor_ec` the same lemma applies to `halfgcd.cpp`: `FastGCD` applies its
matrices to the operands in two places per level, and those are the steps to
verify. Not implemented; `-ecg` there still repeats the gcd.

The round-1 idea of **batching** (one extended gcd of `prod_i A_i` per `s`
instead of one per interval) is superseded: with certification this cheap
there is nothing left to amortise, and most survivors are resolved by their
first gcd anyway.

**Next speedup, not done.** The four products of a matrix application share
operands, and so do the eight of a matrix product: in the frequency domain
that is 6 + 2 transforms instead of 12, and 8 + 4 instead of 24. tsfactor's
GPU hook already batches this way; on the CPU it needs gf2x's FFT interface
(`gf2x-fft.h`) rather than NTL's `mul`. Estimated at a further 25–35%.

---

## 7. Configuration as implemented

| Substep | Check | Cost | Coverage |
|---|---|---|---|
| h-table initialisation | closed form (***), §4.7 (tsfactor); run twice (factor_ec) | `k+m` squarings + 1 modmul (~0.25 s on a GPU); or +100% of init | any single fault |
| h-table squarings, idle table | Frobenius-linear Gerbicz chain, §4, full-width compare | 1 table pass per block + `m` squarings per window | deterministic, single fault |
| Horner build of `C` | fold of an independent recomputation, checked through the product | shares the table pass | error `e` with `phi(e) != 0` |
| product `A*C` | `phi(t) = phi(A) phi(C)` on the unreduced product, `phi(A)` carried | 2r bits of streaming fold | same; also a flip in the idle `A` |
| reduction mod `T` | tsfactor (`2s < r`): closed-form quotient fold. factor_ec (`s > r/2`): divisibility in the reciprocal domain, §5.2 | one fold; or ~1/2 of the fastmulmod EC cost | tsfactor: `phi(e) != 0`; factor_ec: **every** error |
| transfers, snapshots, checkpoints | fold on both sides; fold stored with the copy and rechecked on use | linear | `phi(e) != 0` |
| gcd | §6: certified chain (tsfactor default, `--gpu-gcd`), verified `XGCD` (`--ecg-xgcd`), or repeated (`--ecg-dup`, factor_ec `-ecg`); plain NTL (`--gcd-ntl`) is **unchecked** | a few % of the gcd, on an engine 27–38% faster than NTL's / +42% / +100% | proof / proof / transient faults |
| reported factor | `f | T` by `x^r mod f`; degree within its interval | milliseconds | exact |

Recovery: a failed check restarts from the newest state that was verified
when it was saved. The same restart point failing repeatedly (3 times in
tsfactor, 5 in factor_ec) is a persistent fault: stop and say so, write no
result. That is Gerbicz's own advice ("if we would roll back too much ... then
we just restart completely").

---

## 8. Experimental validation

`gecsim` implements the exact algorithm above at `r = 1279, s = 345` (smallest
factor degree 36), `m = 4, q = 3`, and injects single bit flips at every
operation position in turn. Full output in `gecsim_results.txt`.

**Every single injected fault was caught, and no run with checking enabled ever
produced a wrong answer.**

```
=== 3. Exhaustive single-fault sweep ===
   squaring (h-table)       faults injected = 144
      unchecked: 128 SILENT wrong (factor missed), 0 self-evident, 16 harmless
      checked  : 144 detected+recovered, 0 not flagged, 0 wrong
   modular multiply         faults injected = 18
      unchecked: 6 SILENT wrong, 0 self-evident, 12 harmless
      checked  : 18 detected+recovered, 0 not flagged, 0 wrong
   Horner (build C[z])      faults injected = 72
      unchecked: 8 SILENT wrong, 0 self-evident, 64 harmless
      checked  : 72 detected+recovered, 0 not flagged, 0 wrong
   gcd result               faults injected = 3
      unchecked: 2 SILENT wrong, 0 self-evident, 1 harmless
      checked  : 3 detected+recovered, 0 not flagged, 0 wrong
   idle h-table memory      faults injected = 36
      unchecked: 32 SILENT wrong, 0 self-evident, 4 harmless
      checked  : 36 detected+recovered, 0 not flagged, 0 wrong

=== 6. Random-rate model (faults in every substep simultaneously) ===
   p=0.001  faulty runs 62/300:  unchecked wrong = 44,  checked wrong = 0
   p=0.003  faulty runs 151/300: unchecked wrong = 104, checked wrong = 0
   p=0.01   faulty runs 271/300: unchecked wrong = 238, checked wrong = 0
```

Caveats on what this does and does not establish: `gecsim` uses schoolbook
arithmetic and a bit-serial reducer, so it validates the *algebra of the
checking scheme* and the *fault-propagation model*, not the numerical behaviour
of gf2x's FFT paths or of CUDA kernels. It also injects faults only into
polynomial data, not into loop counters, degree variables, or pointers (see
§9.3).

---

## 9. Caveats, edge cases, and things that could still bite

**9.1 `T` is always squarefree here, so Frobenius is injective.** (Earlier
rounds hedged on the case `r`, `s` both odd; the hedge was unnecessary.) For
`r` odd and `s` even, `T' = x^(r-1)` and `T(0) = 1`. For `r`, `s` both odd,
`T' = x^(s-1) (x^(r-s) + 1)`; an irreducible `g` dividing `T` and `x^(r-s)+1`
would give `x^r = x^s` mod `g`, hence `T = 1` mod `g`. So `gcd(T, T') = 1`
whenever `r` is odd, the ring `R` is a product of fields, `sigma` is an
automorphism, and a nonzero error can never be squared away.

**9.2 Cancellation in the XOR accumulator.** Two faults in one window with
*identical* error polynomials cancel. Independent faults do not do this. A
stuck bit can; §4.6 explains why it is still caught and how it shows up
(repeated failure of the same restart point). A second digest over a random
subset of the components, or `w = 1`, are the belt-and-braces options.

**9.3 Control-flow faults are not covered by any of this.** A flipped bit in
`k`, `q`, `z`, `m`, or a loop bound silently skips degrees, and no algebraic
identity will notice. Cheap mitigations: keep a redundant copy of the loop
counters and compare; assert `k2 == k + q*m - 1` and that the number of blocks
processed equals `q`; assert `deg(a) < r` and `deg(h[j]) < r`. These cost
nothing and close a real hole.

**9.4 The checker can itself be faulted**, producing a false alarm. That is the
safe direction — you redo work you did not need to redo. Just make sure the
rollback path is idempotent and the "too many consecutive failures" counter
exists (§7).

**9.5 `fastrem` writes in place**, so if you want the §5.1 fold check on the
unreduced product you must fold `t` (and grab `t_H`) *before* calling it.

**9.6 `n` and word alignment.** The word-XOR fold is only valid when `n` is a
multiple of the word size. If you use `n` not a multiple of 64, the fold needs
bit shifts and the cost goes up several-fold for no benefit.

---

## 10. `tsfactor` (CUDA): as implemented

The full record is `tsfactor_handoff_EC.md`. What differs from factor_ec, and
why:

* **Opposite reduction regime.** tsfactor enforces `2s < r`; factor.cpp flips
  to `s > r/2`. The closed-form quotient `q = t_H + (t >> (2r-s))` is valid in
  the first and wrong in the second (selftest suite and round 3's `ectest_gpu`
  both pin this). So tsfactor predicts `phi(t mod T)` from three segment folds
  of `t` taken in one pass, and never reduces twice.
* **Nothing is recomputed that can be carried.** `phi(A)` lives on the host
  from one block to the next; `phi(C)` is predicted from the table, not
  measured from `C`. Each carried value turns a "was this computed right"
  check into "and has it stayed right since".
* **One table pass per block does two jobs**: it adds the digest into the
  window accumulator and folds the unreduced Horner sum, independently of the
  Horner kernel.
* **The chain is compared at full width on the device**; only a flag comes
  back. Folds are used where an identity has to survive a multiplication.
* **Rollback is a restart through the resume path**, from a RAM copy of what a
  checkpoint would hold. No second h-table in VRAM.
* **Verify before save.** A snapshot or checkpoint is taken only after the
  chain window is closed on the table it captures; otherwise a restart would
  happily begin from a corrupted table.

* **The gcd is certified by default**, by the chain of §6.3 on the engine of
  §6.4. `--gcd-ntl` goes back to the unchecked NTL gcd.

Cost of the scan checks is estimated at 0.5–1% of scan time (m ~ 30, window
8) and 81 MB of VRAM; the emulator cannot time a GPU, and `--bench` now
reports the real figures.

---

## 11. Relationship to Gerbicz–Li / Atnashev, and an open direction

What was borrowed: the telescoping-checkpoint idea from Robert Gerbicz's
Proth/Pepin check (identities [2] and [3] in the forum post), transplanted from
`(Z/NZ)^*` with `u -> u^2` to `R = GF(2)[x]/(T)` with `u -> u^(2^m)`, and
simplified because `sigma` is additive here.

What was *not* used: the Li–Gallot / Atnashev **proof** layer (arXiv
2209.15623), which the problem statement set aside as unnecessary since factors
are self-verifying. That reasoning is right about *positive* results and worth
revisiting for *negative* ones, because §2 shows the negative results are the
fragile output:

> The claim "`x^r + x^s + 1` has no irreducible factor of degree `<= D`" is
> currently double-checkable only by re-running the whole search. The Atnashev
> scheme makes an analogous claim about a squaring chain verifiable in `1/s` of
> the work.

The transplant is partially clean and partially open:

* **Clean.** `sigma` is multiplicative as well as additive, so the Atnashev
  construction carries over verbatim to the Frobenius chain `u_i = x^(2^i)`:
  with checkpoints `r_i = u_{iM}` and random 64-bit tags `x_i`, one has
  `sigma(prod r_i^{x_i}) = prod r_{i+1}^{x_i}`, and a verifier redoes `M`
  squarings instead of `n`. That certifies the chain — roughly half the work.
* **Open.** The DDF accumulator `a = prod_d (u_d + x)` has no self-similar
  recurrence (`Frob(prod (x + u_d)) = prod (x^2 + u_{d+1})`, which is a
  *different* family — note the `x^2`). So a verifier holding a certified chain
  still has to redo the products. Net: verification at ~50% of original cost,
  not ~5%. Making the product accumulation cheaply certifiable is the open
  problem, and it is the interesting one.

Not needed for error checking. Flagged because it is a real, tractable-looking
research question sitting adjacent to this work, and because a 2x cheaper
double-check for negative results has obvious value to the wider Mersenne
community.

---

## 12. Round status / what a future round should pick up

Round 1: feasibility analysis, the Frobenius-linear Gerbicz identity, `gecsim`.

Round 2: diagnosed and fixed the `ec_reduce_ref` divergence (§5.2); compiled,
linked and ran the patch; 2700-trinomial regression; 108/108 injected faults
caught. Gord then confirmed it at production size — see below.

Round 3: measured overhead reconciled against Gord's production log; two
defects found and fixed from that data; the scheme mapped onto a then-frozen
`tsfactor` with kernels written and validated under emulation.

Round 4 (this one): implemented in `tsfactor`, on by default, with rollback
and restart; closed form for the table digest (§4.7); certified gcd (§6.3)
on a faster engine (§6.4), now the default gcd; fold length 61 in both
programs (§5.1); `factor_ec` otherwise unchanged.

### `factor_ec` at production size, r = 136279841

Round 2 run (s = 588, m = 14, q = 15), `-ect` against the unchecked baseline:

```
  exponentiation      5.65 ->  11.09 s   (+5.44, the duplicate init_h)
  squares/products  127.10 -> 131.60 s   (+4.50)
  gcd               110.89 -> 110.20 s   (-0.69, noise; EC does nothing here)

  EC on intervals only          4.50 s = 3.5% of squares/products = 1.8% of total
  EC including init_h           9.94 s                            = 4.1% of total
```

Result byte-identical to the unchecked run, zero false alarms.

Round 3 build, rerun by Gord at `-skip 55 -k 55 -m 14 -q 16` (his current
sieving configuration):

```
  EC total (-ect)               11.07 s   of which the duplicate init_h  7.64 s
  EC total with -eci0            3.38 s
  copy t for check               2.6 ms per call   (was 7.9: the round-3 fix worked)
  digest XOR                     1.60 s over 675 calls   (was untimed)
  ecbench                        ~38 ms of checking per fastmulmod, 1.3% projected
```

Both round-3 defects are confirmed fixed. The duplicate `init_h` is now 69%
of all checking cost in this configuration — which is exactly what (***) of
§4.7 would cut to about 2 s. That port is the one worthwhile change left in
`factor_ec`; it was not done this round because `factor_ec` now only does
shallow sieving on ECC machines and tsfactor was the priority.

Note that `mersenne_trinomial.py` starts `factor_ec` **without** `-ec`: the
sieving stage currently runs unchecked. Adding `-ec -eci0` to that command
line would cost about 2% there.

### Open items, in rough order of value

- [ ] **Run tsfactor on a real GPU** (`TESTPLAN_EC.md`). Everything in §13
      under "emulation" needs that confirmation.
- [ ] **`gcdbench` on the production machine.** The default gcd changed; the
      sandbox says −32% at production size; a tuned gf2x has not been asked.
- [ ] `--gcd-mt N` timed on a machine with more than two cores
      (`gcdbench r s finish quick N`); then decide whether tsfactor should
      default to more than one thread per gcd under `--pend-max 1`.
- [ ] `--gpu-gcd` re-timed: the split application and the FFT tuning both
      postdate the last measurement (63.0 vs 63.6 s).
- [ ] CPU frequency-domain batching of the matrix products (§6.5, end).
- [ ] Port (***) to `factor_ec` (`init_h` check: ~2 s instead of ~7.6 s).
- [ ] Certify `factor_ec`'s gcd the same way (two verified steps per level in
      `halfgcd.cpp`'s `FastGCD`); `-ecg` there still repeats the gcd.
- [ ] Control-flow guards of §9.3, in both programs.
- [ ] Fuse the digest XOR into the squaring's output store (factor_ec: 1.6 s
      of 3.4 s interval cost; tsfactor: only if `--bench` shows the table pass
      matters).
- [ ] The absolute digest chain of §4.7 (last paragraph), advanced while the
      GPU idles through a gcd: makes stuck-bit faults deterministic to catch.

---

## 13. What is actually verified, and what is not

Round 1 shipped an untested patch on the grounds that the analysis was the
deliverable. The analysis was right and the patch still cost 45 minutes of
machine time. The distinction is therefore kept explicit.

### `factor_ec` — verified by execution

| Claim | How |
|---|---|
| Compiles and links | g++ 13.3, NTL 11.5.1, distro gf2x 1.3.0 (here); NTL 11.6.0 + patched gf2x (Gord) |
| No false alarms | 2700 trinomials here (rerun with the 61-word fold); production runs at r = 136279841 (64-word fold) |
| Output unchanged | byte-identical to unchecked `factor`, both scales |
| Faults are caught | 108/108 injected single-bit faults, 0 wrong answers (rerun with the 61-word fold) |
| Faults matter | 60/108 give a wrong answer unchecked, 56 of them a false `irreducible` |
| Overhead at production size | 1.8% of wall clock excluding `init_h`, 4.1% including it (64-word fold; the 61-word fold does the same work) |
| Word primitives | `ectest2`, 54400 checks against bit-level references |

Not verified: the 61-word fold at production size (one changed constant; the
regression and the sweep pass with it); behaviour above r = 136279841;
multi-threaded operation (`-t > 1`) — the scratch buffers are `thread_local`
but that path is unexercised, and the `-ect` counters are plain globals that
will race harmlessly; the `-ecg` repeated gcd, which is a stopgap.

### `tsfactor` — verified by execution, under CPU emulation

The real kernel bodies run through the project's `cuda_emul.h`
(`make emul-tsfactor`), sequentially.

| Claim | How |
|---|---|
| Compiles warning-free | g++ 13.3, `-Wall -Wextra`; also without NTL (same warnings as before the round) |
| Old behaviour intact | the 16 pre-existing selftest suites pass; `--no-ec` and ten checking configurations give results byte-identical to the **original binary** on 294 survivors at r = 3217, 4423, 9689 (`ts_regress.sh`) |
| Kernels compute what the host algebra predicts | selftest: 48 `(r, s, m)` cases of device folds vs host; block predictions vs the real `C`, `A*C`, `A*C mod T` |
| No false alarms | ~6500 blocks and ~4100 chain windows in the selftest, ~30000 blocks in the regression; twenty consecutive selftest repeats |
| Faults are caught | selftest sweep, 14 sites, 216 faults: 0 wrong answers with checking, 81 of 206 wrong without; 84 more under speculation; 11 on the command line |
| Recovery works | restart from the interval start, from a mid-interval verified state, from a disk checkpoint; in-place repair of an idle flip; persistent fault → exit status 3 and no result line |
| Interrupt/resume intact | three interruptions and resumes, then a fault after a resume, results identical |
| Certified gcd is right | 75 gcds in the selftest (chain on CPU products, on the GPU hook, on three threads, XGCD, declined chain); 600 fault runs: fired ⇔ certificate failed; `gcdstress`: 3100 inputs × 4 engines vs NTL, 0 mismatches |
| Memory safety | ASan + UBSan clean: selftest, `ts_regress.sh`, `gcdstress` |
| Strided fold loops | selftest rebuilt with `-DEC_NBLK_MAX=1` (many strides per thread, as at production size) |
| Production size | emulated scans at r = 136279841 (s = 320), once with each certified gcd mode: result equal to the oracle, all block and chain checks passing, gcd certified (chain: 660 s; XGCD: 1360 s; NTL's plain gcd on that machine: 972 s) |

### `tsfactor` — NOT verified

* **Nothing has been through `nvcc`; nothing has run on a GPU.** There is no
  CUDA toolkit in the authoring sandbox.
* **Concurrency.** Emulation is sequential. The checking kernels are
  gather-form with one writer per output word (two have many writers of the
  *same* value), which is why a race is unlikely; unlikely is not tested.
* **Every GPU cost figure** is an estimate. `--bench` and `--ect` report the
  real ones.
* **The gcd speedup on a tuned gf2x.** Sandbox only.
* **The 2 GB-scale paths under real memory pressure** (snapshots, rollback
  state) — exercised once, at production size, in emulation.

# Low-cost error checking for trinomial factor search over GF(2)
## Analysis + handoff document — Round 7

> **Round 7 in one paragraph.** Gord's run of the round-6 test plan: nvcc
> builds round 10, both selftests pass (the GPU one ten times running), the
> regression passes for both builds, and the step-5 survivor took 192.8 s
> with the new schedule (342 s before). The round's main item is **GPU
> irred** (`irred/docs/irred_handoff.md` §4b): the single-chain XOR Gerbicz
> check of irred (one polynomial, F^L(D) = D + V + u) now runs with the
> squarings on the GPU. The design keeps the verified state V in host
> memory: at a window close the check's L squarings run on the GPU, then
> x, D and u are downloaded and compared with V in the same host loop that
> takes u's CRC, so V and every checkpoint are copied from the words that
> were compared; a corrupted transfer in either direction surfaces as a
> failed check, never as a wrong verified state. Validated under the CUDA
> emulation by the full fault sweeps (8,606 injected faults, all detected,
> all results right). tsfactor got small fixes (round 11 in its handoff).

> **Round 6 in one paragraph.** Gord's run of the round-5 test plan: the
> certified gcd is 29% faster than NTL's on his machine on one thread
> (36.3 s against 51.0 s), and 2.1x faster again on 4 threads, where it
> saturates (17.0 s on 4, 17.1 s on 6). With the gcd settled, round 6 is
> structural: the checking now lives in a backend-independent driver
> (`ts_core.h`), so the same checks run in **`tsfactor_cpu`**, the CPU build
> of tsfactor, and are verified there by the same suites (§10, §13). On the
> CPU they cost about 3% of the scan (one table pass, the product folds and
> the chain check against ~0.13 s per degree), against ~15% for `factor_ec`'s
> NTL-object implementation. `factor_ec` 3.01 takes factor_modified's faster
> squaring; its checks are unchanged. And the error checking caught a bug of
> ours at production size (§12, round 6): exactly what it is for.
>
> **Round 5 in one paragraph.** Gord compiled the round-4 `tsfactor` with nvcc
> and ran the whole test plan on the real GPU: everything passed, and the
> scan checks cost about 1% as estimated (§12). His logs also showed that the
> gcd was 58% of a survivor's wall time and that the certified gcd engine,
> "27% faster than NTL" in the sandbox, was 6% *slower* on his machine: half
> of every gcd was the engine's bit-level base case, hidden in the sandbox by
> a gf2x ten times slower than his. Round 5 replaces that base case (§6.5),
> which roughly halves the single-threaded gcd and lets `--gcd-mt` scale;
> ports the closed-form init check to `factor_ec`; and turns checking on by
> default there too.
>
> What rounds 4 and 5 changed in this document:
>
> 1. **The h-table digest has a closed form** (§4.7): the initialisation is
>    verified with `k + m` squarings and one product. Now in both programs.
> 2. **The gcd can be certified without computing cofactors** (§6.3).
> 3. **Split application** of half-gcd matrices (§6.4) and a **word-level
>    base case** (§6.5): together they make the certified gcd the fastest
>    gcd available here, and the default.
> 4. **The fold length must not be a power of two** (§5.1): 61 words.
> 5. **CPU frequency-domain batching was measured and not adopted** (§6.5).
>
> §13 is the ledger of what is verified by execution, where, and what is not.
> For the tsfactor integration itself read `tsfactor_handoff.md` (Rounds 8
> and 9 there).

**Status: implemented in all three programs (tsfactor, tsfactor_cpu,
factor_ec), on by default in all; and in irred and irred_gpu (irred's own
single-chain version).** `tsfactor` round 10 (= this document's round 6)
is verified on the real GPU (Gord, R6 test plan); round 11 (round 7 here)
changes no checking code. `factor_ec` 3.01 is verified at small r and in a
production-shape run in the sandbox. irred_gpu is verified under emulation
and has not been through nvcc.

The core fact is unchanged: in characteristic 2 the squaring map is
**additive** as well as multiplicative, so the Gerbicz "product of
checkpoints" becomes an **XOR of checkpoints**, cheaper than the integer
version and deterministic for single faults.

---

## 0. Reading guide / file map

| File | What it is |
|---|---|
| `errorcheck_handoff.md` | This document: the mathematics, the measurements, the status ledger. Start here. |
| `tsfactor_handoff.md` | The complete tsfactor record. Round 8 there is the error-checking integration (where each check sits, recovery, options, tests); Round 9 is the gcd work of round 5; Round 10 the driver/backend split and `tsfactor_cpu` (round 6). |
| `TESTPLAN_R7.md` | What to run on the real machine now (tsfactor round 11), in order, and what to send back; GPU irred: `irred/docs/TESTPLAN.md`. |
| `irred/docs/irred_handoff.md` | irred and irred_gpu; §3 is irred's single-chain version of the check, §4b its GPU build. |
| **tsfactor** | |
| `gf2_ec.h` | Fold algebra (host) + checking kernels (device). Header-only. Replaces round 3's `ec_cuda.cuh`. |
| `ts_core.h` | The driver (round 6): the checking logic, chain bookkeeping, restarts, checkpoints; backend-independent. |
| `tsfactor.cu` / `ts_cpu.h` + `tsfactor_cpu.cpp` | The GPU backend and program / the CPU backend and program; each supplies the folds of its own buffers. |
| `gf2_gcd.h`, `gf2_gcd_ntl.cpp` | The gcd engine and the certified gcd. |
| `gf2_hgcd_base.h` | the word-level (Lehmer) base case of the gcd engine, §6.5 |
| `gcdbench.cpp` | CPU-only timing of every gcd path and engine generation. `make gcdbench` |
| `gcdstress.cpp` | CPU-only randomized cross-check of every gcd path against NTL, and of the word-level base case against the reference one. `make gcdstress` |
| `experiments/fftbatch_bench.cpp` | the measurement behind "CPU batching: not adopted" (§6.5) |
| `tests/ts_regress.sh` | tsfactor command-line regression (either build): configuration runs vs the original binary, injected faults, interrupt/resume. |
| `tests/ts_ckpt10.sh` | per-survivor checkpoints, tuning adoption, the prefix lock, cross-build resume (round 6). |
| **factor_ec** | |
| `factor_ec.cpp`, `factor_ec.patch`, `factor_ec.patch.md` | Patched `factor.cpp`, the diff, and hunk-by-hunk commentary. Round 5: checking on by default, closed-form init check. 3.01 (round 6): factor_modified's squaring and `-Z`. |
| `ec_words.h` | Word-level primitives shared by `factor_ec`, `ecbench`, `ectest2`. |
| `ectest.cpp`, `ectest2.cpp` | Unit tests of the word primitives against bit-level references. |
| `ecbench.cpp` | Per-primitive cost at any (r,s), no NTL. |
| `regress.sh`, `sweep.sh` | factor_ec vs factor on 5781 trinomials in 20 configurations; the 150-opportunity fault sweep plus corrupted initialisations. |
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

One qualification (Gord, round 5). There is a last line of defence outside
these programs: a trinomial that survived everything would be re-verified
with an independent, slower program before anything was announced — `irred`,
or `factor`'s own irreducibility mode, which computes a residue by `r`
squarings and would take a few weeks at r = 136279841. So a false "no
factor" would not end up *published*. What it costs is everything spent
scanning deeper on an `s` that should have been retired, and then that
verification run; and it would be found out at the worst possible moment.
The checks exist to keep the wrong result from arising in the first place.

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

The left side is also a geometric series, which is the quickest way to
believe it: the table's entries are the sums of `y^e` over exponents `e < 2^m`
of each Hamming weight, so `S = y + y^2 + ... + y^(2^m - 1)` with `y = y_k`,
and `(1 + y)(1 + S) = 1 + y^(2^m)`.

This replaces "run `init_h` twice". In both programs the table is built from
squarings only, `m (k + m/2)` of them; the check is `k + m` squarings and one
product, so what it saves depends on what a product costs in squarings:

* **factor_ec** (`m = 14`, `k = 56`): the init is 882 squarings, the check 69
  squarings and a product worth ~200: roughly **2 s instead of the 5.4–7.6 s**
  measured for the duplicate, which was 69% of all checking at `-skip 55`.
  Implemented in round 5 (`ec_init_closed_form`); `-eci0` skips it.
* **tsfactor** (`m ~ 30`): a product is ~900 fused GPU squarings. The init
  is about half a second at `k = 56` and 1.6 s at `k = 263`; the check
  measured **0.28 s** on the RTX 2070 SUPER. Noise next to a gcd, which is why
  `--eci0` has no claim to be the default.

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

Measured: certification adds 0–4% to the engine (Gord's machine: 55.1 s
certified against 54.8 s uncertified; the folds are linear-time and the
extended finish is on a pair a small fraction of the size). Selftest: 75
certified gcds equal to NTL's; 600 runs with one fault at step N = 1..40 —
every fault that fired produced "certificate failed", never a certified wrong
answer.

### 6.4 Split application (round 4)

The engine (tsfactor's `hyb::hgcd`, written in an earlier round for
`--gpu-gcd`) recurses on the top half of the pair — in place, on a copy — and
gets back a matrix `R`; it then applied `R` to the full-length operands. But
with `a = (a >> k) x^k + (a mod x^k)`,

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

### 6.5 The base case was half the gcd (round 5) — and a lesson about benchmarks

Round 4 measured the split engine 27–32% **faster** than NTL's gcd in the
sandbox and made it tsfactor's default. On Gord's machine it was 6% **slower**
(55.1 s against 51.9 s). Both measurements were right. The sandbox used the
distribution's gf2x, built without PCLMUL: its multiplications are about ten
times slower than those of a source-built gf2x, so there the gcd *was* its
large products and the product count decided everything. With fast products
the rest of the engine shows, and the rest was large: `--gcd-mt 4` only
reached 1.57x, which by Amdahl's law says **half of the time was not in the
parallel products at all**. `tsfactor --bench-gcd` printed the reason in
passing: `cpu 135960386` — 136 million multiplications per gcd.

Those were the bottom of the recursion. Below 4096 bits the engine did Euclid
one division step at a time on `std::vector` operands, with a bit-at-a-time
shift-xor product per quotient to maintain the matrix: about 68 million
steps, each allocating.

**The replacement (`gf2_hgcd_base.h`) is the polynomial form of Lehmer's
method.** Take the top 64 bits of the pair, same alignment for both; run
Euclid on those two machine words, accumulating a 2x2 matrix of word-sized
polynomials (shifts and xors only); then apply that matrix to the full
operands and to the accumulated matrix with carry-less word multiplications —
about 31 quotient bits per pass. No allocation, no library call.

Over `GF(2)[x]` the usual Lehmer caveat ("the quotient computed from leading
digits may be off by one") does not exist, because there are no carries. The
rule is exact: with the dividend's leading bit at bit 63 of the window, a
division step whose divisor has window degree `v` is the true step **iff
`2v >= 63`**. (The true remainder is `window_remainder * x^j + e` with
`deg e < j + 63 - v'`, `v'` the previous divisor's window degree; the quotient
depends only on the top `u - v + 1` bits of dividend and divisor.) So the
window Euclid runs while the divisor's degree is at least 32 — which is the
ordinary half-gcd stopping rule applied to a 64-bit pair — and what it
produces is not an approximation: the quotient sequence, the matrix and the
reduced pair are **identical** to the old loop's. That is how it is tested
(`hyb_base_selfcheck`, `gcdstress`): the old loop is kept as the reference and
the two are compared word for word: 124,000 small pairs of every shape, and
3100 stress cases at seven hand-over sizes, without a difference.

Two smaller changes went with it: the engine's CPU products call `gf2x_mul`
directly on its word arrays instead of converting to and from `NTL::GF2X`
(`-DHAVE_GF2X`, detected by the Makefile), and the per-product statistics
counter no longer takes a mutex.

Where one production-size gcd's time goes now (`gcdbench` built with
`-DHYB_PROF`, single thread, sandbox with the source-built gf2x):

```
    products >= 1 Mbit            24.8 s   62%
    products 64 kbit - 1 Mbit     10.0 s   25%
    products 4 - 64 kbit           2.0 s    5%
    word-level base case           0.9 s    2%
    products < 4 kbit              0.6 s    1.5%
    division steps, shifts, adds   0.8 s    2%
    everything else                1.0 s    2.5%
```

93% products, and 87% in products large enough to be worth a thread: the
serial fraction fell from about 50% to about 13%.

Measured. Left: Gord's machine (6-core Zen 2, tuned gf2x), round-4 code, his
`gcdbench 136279841 588`. Right: the sandbox with gf2x built from his source
(its NTL gcd time happens to match his), round-5 code.

```
                                    round 4 (Gord)      round 5 (sandbox, same NTL time)
   NTL GCD                             51.9 s              53.4 s
   CERTIFIED, NTL XGCD                 60.9 s  +17%        63.4 s  +19%
   engine as of round 3                78.2 s  +51%        70.8 s  +33%
   engine as of round 4                54.8 s   +6%        47.2 s  -12%
   engine now, uncertified                                 37.7 s  -30%
   CERTIFIED chain, 1 thread           55.1 s   +6%        38.8 s  -28%
   CERTIFIED chain, 2 threads          42.1 s  -19%        24.6 s  -54%  (22.7 s in another run)
   CERTIFIED chain, 4 threads          35.1 s  -32%        (two cores only)
```

(The sandbox's "round 3" and "round 4" rows are those algorithms inside
today's code, so they already benefit from the direct gf2x calls; that is why
round 4 reads -12% there and +6% on Gord's machine.)
The round-5 column on four real cores is the number still missing; with a
13% serial fraction and the 4x scaling Gord's machine showed on the products,
it should land near 12–15 s.

**Frequency-domain batching on the CPU: measured, not adopted.** Round 4
proposed it as "the next speedup" and estimated 25–35%. The four products of a
matrix application share operands (6 forward + 2 inverse transforms instead of
12), and gf2x exports exactly the interface for it (`gf2x-ternary-fft.h`,
`addcompose_n`). Measured against four separate `gf2x_mul` calls
(`experiments/fftbatch_bench.cpp`), single thread:

```
     operand words        matrix x vector          matrix x matrix
        2048 (130 kbit)        0%                       0%
        8192                  -2% .. -16%               0%
       32768 (2 Mbit)        -12% .. -25%             -21%
      131072                 -15% .. -20%             -25%
      524288 (34 Mbit)       -12% .. -19%             -19%
     Cantor FFT instead: slower than separate products at every size.
```

So about 15–20% on the products above 2 Mbit, nothing below 0.5 Mbit: roughly
10% of a single-threaded gcd. In a Schönhage-type FFT the pointwise products
are themselves recursive multiplications and are not shared, which is why the
gain is so much smaller than on the GPU's additive FFT. And it does not
combine with threads: four whole products run side by side on four cores,
whereas the batched form needs six transforms, then four pointwise passes,
then two inverses — on four cores that is *slower* in wall time than the
unbatched products, and it only pulls ahead with six or more. Since the
threads are worth 3–4x and batching 1.1x, the threads won; and the GPU hook,
which does batch, remains available for the largest products.

### 6.6 Which to use

| tsfactor | what runs | notes |
|---|---|---|
| default | certified chain, CPU products, `--gcd-mt` threads (automatic: cores left over, at most 4) | the fastest CPU configuration and the checked one |
| `--gcd-mt 1` | the same on one thread | when the cores are needed elsewhere |
| `--gpu-gcd` | certified chain, products above 2 Mbit on the GPU, the rest on the threads | the GPU idles during a gcd under `--pend-max 1`; worth timing against the default (`--bench-gcd`) |
| `--ecg-xgcd` | one verified NTL `XGCD` | +17–20% over NTL's gcd; independent reference |
| `--ecg-dup` | NTL twice, compared | transient faults only |
| `--gcd-ntl` | NTL, **unchecked** | A/B timing |
| `--no-ec` | the engine, **uncertified** | (round 4: NTL) |

For `factor_ec` the same lemma applies to `halfgcd.cpp`: `FastGCD` applies its
matrices to the operands in two places per level, and those are the steps to
verify. Not implemented; `-ecg` there still repeats the gcd.

The round-1 idea of **batching gcds** (one extended gcd of `prod_i A_i` per `s`
instead of one per interval) is superseded: with certification this cheap
there is nothing left to amortise, and most survivors are resolved by their
first gcd anyway.

---

## 7. Configuration as implemented

| Substep | Check | Cost | Coverage |
|---|---|---|---|
| h-table initialisation | closed form (***), §4.7, in both programs | `k+m` squarings + 1 product (0.28 s on the GPU; ~2 s in factor_ec) | any single fault |
| h-table squarings, idle table | Frobenius-linear Gerbicz chain, §4, full-width compare | 1 table pass per block + `m` squarings per window | deterministic, single fault |
| Horner build of `C` | fold of an independent recomputation, checked through the product | shares the table pass | error `e` with `phi(e) != 0` |
| product `A*C` | `phi(t) = phi(A) phi(C)` on the unreduced product, `phi(A)` carried | 2r bits of streaming fold | same; also a flip in the idle `A` |
| reduction mod `T` | tsfactor (`2s < r`): closed-form quotient fold. factor_ec (`s > r/2`): divisibility in the reciprocal domain, §5.2 | one fold; or ~1/2 of the fastmulmod EC cost | tsfactor: `phi(e) != 0`; factor_ec: **every** error |
| transfers, snapshots, checkpoints | fold on both sides; fold stored with the copy and rechecked on use | linear | `phi(e) != 0` |
| gcd | §6: certified chain (tsfactor default, `--gpu-gcd`), verified `XGCD` (`--ecg-xgcd`), or repeated (`--ecg-dup`, factor_ec `-ecg`); plain NTL (`--gcd-ntl`) and `--no-ec` are **unchecked** | 0–4% of the gcd, on an engine faster than NTL's (§6.5) / +17% / +100% | proof / proof / transient faults |
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

The full record is `tsfactor_handoff.md` (Rounds 8, 9 and 10). What differs from
factor_ec, and why:

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
  §6.4–6.5, on several threads. `--gcd-ntl` goes back to the unchecked NTL
  gcd.

Cost of the scan checks, measured on an RTX 2070 SUPER: 0.6% of scan time
projected by `--bench` (m = 24..32, window 8), +1.0% on a real survivor, and
81 MB of VRAM.

**Round 10 (round 6 here): the same checks on the CPU.** The checking logic
moved into the backend-independent driver `ts_core.h`; a backend supplies
only the folds of its own buffers (table pass, three-segment product fold,
fold of the reduced result, digest, window check, closed-form init check,
table and hold-table folds). `tsfactor_cpu` (backend `TsCPU`, `ts_cpu.h`)
implements them with word loops that produce exactly the GPU kernels'
values, so everything above holds for it unchanged — including the
`2s < r` regime: tsfactor_cpu does not flip even s (it has its own even-s
squaring), so it has none of factor_ec's reciprocal-regime machinery.
Measured cost on one sandbox core at r = 136279841: ~3.3% of scan time
(table pass 26 ms and product folds 18 ms per block, chain check 66 ms per
window of 8 blocks, against ~2.4 s of scanning per block of m = 20 degrees).
The checks' share is larger than on the GPU because the CPU's squaring is
relatively cheap next to its folds' memory traffic, not because anything
extra is checked.

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
caught. Gord then confirmed it at production size.

Round 3: measured overhead reconciled against Gord's production log; two
defects found and fixed from that data; the scheme mapped onto a then-frozen
`tsfactor`.

Round 4: implemented in `tsfactor`, on by default, with rollback and restart;
closed form for the table digest (§4.7); certified gcd (§6.3) with split
application (§6.4); fold length 61 in both programs (§5.1).

Round 5: Gord compiled round 4 with nvcc and ran the whole test
plan on the RTX 2070 SUPER — everything passed (below). His logs showed the
gcd to be 58% of a survivor's wall time and half of each gcd to be the
engine's base case; that was rewritten (§6.5). `--gcd-mt` became automatic,
the interval schedule now measures its own gcd/scan cost ratio, finished
survivors no longer leave checkpoint files behind. `factor_ec`: closed-form
init check implemented, checking on by default.

Round 6 (tsfactor round 10): Gord's run of round 5 — nvcc clean,
selftest and gcdstress PASS, the certified gcd 36.3 s on one thread against
NTL's 51.0 s, 17.0 s on four threads (2.1x, saturating), `--gpu-gcd` ahead
inside tsfactor (19.1 against 21.7 s). The tsfactor code was split into a
driver and two backends; the checks are in the driver (`ts_core.h`: fold
algebra, chain bookkeeping, restart logic, certified gcd) and each backend
supplies the folds of its own buffers (`TsDev::ec_block_pre`,
`ec_read_block`, `ec_window_check`, `ec_verify_table_at`, `table_fold`,
`hold_fold`). The CPU backend computes exactly the values the GPU kernels
compute: the GPU build's selftest compares the two after every block. On the
CPU the scan checks cost ~3.3% (bench at r = 136279841: table pass 26 ms and
product folds 18 ms per block, chain check 66 ms per window, at ~0.12 s per
degree).

*The checking caught one of our bugs.* A word-run version of the CPU
reduction assumed zeros beyond its logical end in a scratch buffer; after the
start-up tuning (run at s = (r−1)/2) a smaller s found stale words there. The
small-r tests reuse a device only with s increasing and passed. The first
production-size run then failed the closed-form init check (§4.7) four times
from the same state and stopped with exit status 3 ("persistent fault"),
writing no result. That is the designed behaviour, and it is how the bug was
found within minutes instead of as a wrong "u" weeks later. Fixed, with a
selftest that reuses one device with s decreasing (and fails without the
fix).

`factor_ec` 3.01: factor_modified 2.02's squaring (vectorisable reduce loop,
constants precomputed per r) and `-Z`; the checks are unchanged and pass the
same regression and fault sweep.

Round 7 (tsfactor round 11, irred 0.23): Gord's run of round 6 on the real
GPU — nvcc builds (two warnings, fixed), `tsfactor --selftest` all PASS ten
times in a row, `tsfactor_cpu --selftest` all PASS, `ts_regress.sh` all PASS
for both builds, the step-5 survivor in 192.8 s with every check on (0
detections). The new item is the GPU build of irred, whose check is the
single-chain identity of irred_handoff §3.1 with the squarings on the GPU
and the comparison and V on the host (paragraph at the top). Its map check
compares the GPU squaring with irred's `ref_square`; its selftest compares
every GPU kernel with `ref_square` at every mask boundary (5,168 cases) and
GPU chains with the CPU backend; the full `ec_sweep_all.sh` runs on the
emulated GPU build with all faults detected and all results right.

### `tsfactor` on the real GPU (Gord, round-4 code, RTX 2070 SUPER, 6-core Zen 2)

```
  make tsfactor (nvcc)            compiles; 5 unused-variable warnings (fixed in round 5)
  --selftest, --selftest-reps 20  32 PASS lines each, 21 runs
  ts_regress.sh ./tsfactor        all PASS (33 configurations, 11 faults, fatal, resume)
  --bench: scan-check kernels     h-table pass 1.36 ms/block (m=24), folds 0.24 ms,
                                  readback 0.10 ms, chain check 5.6 ms/window
                                  => 0.6% of scan time at m = 24..32, --ecw 8   (estimate was 0.5-1%)
  real survivor (s=12337659, skip 262, 3 intervals, factor of degree 5816):
     --no-ec                      280.2 s   scan 124.0 s   gcd 3 x 51 s
     default                      305.5 s   scan 125.2 s   gcd 3 x 59 s
     scan overhead                +1.0%  (--ect table: 3.8 s, of which 2.4 s are the
                                  7 verified h-table transfers at 0.34 s each)
     gcd                          +16%: the certified chain was SLOWER than NTL there (§6.5)
  --ec-fault h:200, idle:1, t:50, xferh:1
                                  each caught, restarted or re-read, same result line
  interrupt + resume              same result line
```

So the checking itself costs about 1% of the scan, as estimated. What the
round-4 default cost in practice was 8 s per gcd, from a gcd engine that was
slower than believed; round 5 removes that and then some.

### `factor_ec` at production size, r = 136279841

Round 2 run (s = 588, m = 14, q = 15), `-ect` against the unchecked baseline:

```
  exponentiation      5.65 ->  11.09 s   (+5.44, the duplicate init_h)
  squares/products  127.10 -> 131.60 s   (+4.50)
  gcd               110.89 -> 110.20 s   (-0.69, noise; EC does nothing here)

  EC on intervals only          4.50 s = 3.5% of squares/products = 1.8% of total
```

Round 3 build, rerun by Gord at `-skip 55 -k 55 -m 14 -q 16`:

```
  EC total (-ect)               11.07 s   of which the duplicate init_h  7.64 s
  EC total with -eci0            3.38 s
```

Round 5 replaces the duplicate `init_h` (882 squarings) by the closed-form
check (69 squarings and one product): expected about 2 s instead of 7.6 s,
**not yet measured at production size**. `init_h` in factor.cpp is built from
squarings only, exactly as in tsfactor (an earlier version of this document
said it used modular products; it does not).

### Open items, in rough order of value

- [x] Round-5 numbers on Gord's machine (above, round 6).
- [x] nvcc compile of the round-5 tsfactor.cu (clean).
- [x] nvcc compile of the round-10 tsfactor.cu and its selftest on the card
      (Gord, R6: clean apart from two warnings, fixed in round 11).
- [ ] **nvcc compile of irred_gpu** and its selftest, tests and full-size
      timings on the card (irred TESTPLAN steps 1–3).
- [ ] tsfactor_cpu's check cost and throughput on Zen 2 / cloud CPUs, six
      processes at once (memory bandwidth).
- [ ] `factor_ec -ect` at production size with the closed-form init check.
- [ ] `--gpu-gcd` against the threaded default; `--gpu-gcd-min-bits` may want
      re-tuning now that the CPU side is faster.
- [ ] Strassen for the 2x2 matrix product on one thread (7 products instead
      of 8): about 4% of a single-threaded gcd, nothing with 4+ threads.
- [ ] Certify `factor_ec`'s gcd the same way (two verified steps per level in
      `halfgcd.cpp`'s `FastGCD`); `-ecg` there still repeats the gcd.
- [ ] Control-flow guards of §9.3, in both programs.
- [ ] Fuse the digest XOR into the squaring's output store (factor_ec: 1.6 s
      of 3.4 s interval cost; tsfactor: the table pass is 1.36 ms against a
      1.08 ms Horner — fusing them would roughly halve the 0.6%).
- [ ] The absolute digest chain of §4.7 (last paragraph), advanced while the
      GPU idles through a gcd: makes stuck-bit faults deterministic to catch.

---

## 13. What is actually verified, and what is not

Round 1 shipped an untested patch on the grounds that the analysis was the
deliverable. The analysis was right and the patch still cost 45 minutes of
machine time. Round 4 shipped a gcd "27% faster than NTL" measured on a
library ten times slower than the one in production; on the real machine it
was 6% slower. The distinction between what was measured where is therefore
kept explicit.

### `factor_ec` — verified by execution

| Claim | How |
|---|---|
| Compiles and links | g++ 13.3, NTL 11.5.1, distro gf2x 1.3.0 (here); NTL 11.6.0 + patched gf2x (Gord, rounds 2–3) |
| No false alarms, output unchanged | round 5: 5781 trinomials in 20 configurations (default, three windows, `-noec`, `-eci0`, `-mt 2`, the production option set with s on stdin), identical to `factor`; production runs at r = 136279841 (rounds 2–3) |
| Faults are caught | 108/108 injected single-bit faults, 0 wrong answers; 60/108 give a wrong answer unchecked, 56 of them a false `irreducible` |
| Closed-form init check | a corrupted fresh h-table is caught and rebuilt in 4/4 runs; the same corruption unchecked gives 4/4 wrong answers |
| Overhead at production size | 1.8% of wall clock excluding `init_h` (round 2, 64-word fold) |
| Word primitives | `ectest2`, 54400 checks against bit-level references |

Round 6 (3.01, sandbox): regression and fault sweep as above, plus four `-Z`
rows; at r = 136279841 the production shape (s = 588, `-skip 55 -k 55 -m 14
-q 16 -maxd 279`) ran in 84 s with 0 detections, and six survivors (skip 1)
gave the same result lines as tsfactor_cpu.

Not verified: the `-ect` cost table at production size with the closed-form
init check; behaviour above r = 136279841; the `-ecg` repeated gcd, which is
a stopgap.

### `tsfactor` — verified on the real GPU (Gord, round-4 code)

| Claim | How |
|---|---|
| Compiles with nvcc | CUDA on Ubuntu 26.04, g++ 15.2 host compiler |
| Kernels correct under real concurrency | `--selftest` 32/32, 21 consecutive runs |
| Old behaviour intact, faults caught, recovery works | `ts_regress.sh ./tsfactor`: all PASS |
| Production size | s = 12337659: identical result line in 8 runs (checking off, on, with timing, four injected faults, interrupt/resume) |
| Scan-check cost | 0.6% projected by `--bench`, +1.0% measured on the survivor |

### `tsfactor` — verified under CPU emulation (round-5 code)

The real kernel bodies run through `cuda_emul.h` (`make emul-tsfactor`),
sequentially. Round 5 changed only host code.

| Claim | How |
|---|---|
| Compiles warning-free | g++ 13.3, `-Wall -Wextra` |
| Selftest | 32/32; repeated; under ASan + UBSan |
| Old behaviour intact | `ts_regress.sh` against the **original (pre-round-4) binary**: 33 configuration runs identical, faults caught, resume, no checkpoint left behind |
| Word-level base case == reference | `gcdstress`: 124,000 small pairs of every shape and 3100 stress cases at seven hand-over sizes, reduced pair and matrix identical word for word; also under ASan + UBSan and with the portable (non-PCLMUL) multiply |
| Every gcd path == NTL | `gcdstress`: 3100 cases, four engines each, a third of them on three threads; ThreadSanitizer clean; also with products routed through NTL (`GF2X_OK=0`) |
| Certified gcd | selftest: 90 gcds (chain on CPU products, on the GPU hook, on three threads, threads + hook together, XGCD, declined chain); 600 fault runs: fired ⇔ certificate failed |
| Production size | emulated scan at r = 136279841 (s = 320): result equal to the oracle, gcd certified in 23 s on two threads |

### `tsfactor` / `tsfactor_cpu` — verified here, round-10 code (round 6)

| Claim | How |
|---|---|
| Driver/backend split changed nothing | `ts_regress.sh` against the original binary: all PASS for the emulated GPU build and for `tsfactor_cpu` (also under ASan + UBSan) |
| CPU backend == GPU backend | GPU-build selftest suite: h-table, A, C, D, table and block folds compared after init and after every block, 10 (r, s, m, k) cases |
| Checks work on the CPU backend | `tsfactor_cpu --selftest`: every EC suite (single-fault sweep 216 faults, 0 wrong answers; faults under speculation; persistent fault; certified gcd under faults), also with `--threads 2`, also under ASan + UBSan |
| CPU arithmetic correct | squaring (3934 cases, both parities of s, generic path) and products against bit-level oracles; at r = 136279841 against NTL's SqrMod/MulMod for five s; six survivors at r = 136279841 identical to `factor_ec` |
| Checkpoint/tuning/lock logic | `ts_ckpt10.sh`, both builds, both directions of cross-build resume |
| Persistent-fault path at production size | the bug above: init check failed 4 times, exit 3, nothing written |

### `tsfactor` — verified on the real GPU, round-10 code (Gord, R6 test plan)

nvcc build; `--selftest` all PASS (ten consecutive runs); `ts_regress.sh` all
PASS (GPU and CPU builds); the step-5 survivor (192.8 s, 4 certified gcds,
261 blocks and 37 chain windows verified, 0 detections); setting a survivor
aside and resuming it; checkpoint and lock behaviour (`logs/TESTPLAN_R6/`).

### `irred_gpu` — verified under emulation (round 7), NOT on a GPU

Selftest (kernels vs `ref_square`, GPU vs CPU chains, every fault site, the
linear-bug demonstration); `tests/gpu_test.sh` (144 results vs the naive
reference, CPU agreement, checkpoints between the builds, the full fault
sweeps); ASan+UBSan; the residues after 2000 and 4000 squarings at
r = 136279841, s = 12338357 equal the Zen 2's. Not compiled by nvcc, not
run on a GPU.

### `tsfactor` — NOT verified

* ~~Round 10 has not been through nvcc~~ (it has: above). Round 11's
  changes are host code in `ts_core.h` (verified here: selftest, regress)
  and one format string in tsfactor.cu.
* (Measured since: the round-5 gcd on Gord's six cores and tuned gf2x —
  36.3 s on one thread, 17.0 s on four, §12.)
* **Concurrency of the checking kernels beyond what the selftest exercises.**
* ~~The round-10 interval schedule in production~~: the step-5 survivor ran
  it (192.8 s, intervals as planned).
* **tsfactor_cpu on Zen 2 or cloud CPUs**, alone or six at once.

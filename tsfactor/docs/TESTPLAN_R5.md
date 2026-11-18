# Round 5 — what to run, in order

Round 5 changed host code only (the gcd engine, a few options, scheduling,
checkpoint cleanup) plus `factor_ec`. Nothing in the CUDA kernels changed.
It was tested here under emulation and with the CPU tools, against a gf2x
built from your source tree — which gives the same NTL gcd time as your
machine — but on two cores. The questions for your machine are therefore:
**does it still compile under nvcc, and how fast is the gcd on real cores?**

Steps 1, 2 and 5 matter most. Step 2 needs no GPU.

## 0. Build

    make tsfactor gcdbench gcdstress

The first line printed should read

    == NTL: OK (-lntl -lgf2x -lgmp), gf2x called directly ==

If it says "gf2x.h is not usable" instead, the gcd still works but through
NTL's `mul` (slower): tell me, and check that `gf2x.h` is where the compiler
looks (`/usr/local/include` for your source build).

New files: `gf2_hgcd_base.h`, `gf2x_check.cpp`. The five nvcc warnings from
last time should be gone.

## 1. gcd correctness (CPU, well under a minute with your gf2x)

    ./gcdstress

Three things must hold, and the last line must say `PASS`:

    gcdstress: word-level base case == reference on 16000 small pairs (...)
    ...
      word-level base case == reference on all 400 pairs, at 7 hand-over sizes
      mismatches: 0   PASS

## 2. gcd speed (CPU, about six minutes) — the number I most want

    ./gcdbench 136279841 588

Last time this printed, on your machine,

      NTL GCD                          51.95 s
      hybrid HGCD, full apply (old)    78.23 s
      hybrid HGCD, split apply         54.82 s
      CERTIFIED, chain (the default)   55.11 s
      CERTIFIED, chain, --gcd-mt 4     35.07 s

The same rows now have new names, and there is one new engine row:

      NTL GCD (--gcd-ntl)
      CERTIFIED, NTL XGCD
      engine as of round 3          (full apply, bit-level base case)
      engine as of round 4          (split apply, bit-level base case)
      engine now, uncertified       (word-level base case)
      CERTIFIED chain, 1 thread
      CERTIFIED chain, --gcd-mt N   N = what tsfactor picks by itself: 4 on 6 cores

In the sandbox the "CERTIFIED chain, 1 thread" row went from +6% slower than
NTL to about 24% faster, and two threads gave 1.7–1.8x on top. If your
machine behaves like the sandbox did on one thread, expect roughly **40 s on
one thread and 12–15 s on four**, against the 55 s and 35 s above.

    ./gcdbench 136279841 588 4194304 quick 6     # last row on 6 threads
    ./gcdbench 136279841 588 4194304 quick 2

are worth a minute each if you have it: a matrix step has 4 independent
products and its sibling 8, so 6 threads should help a little beyond 4.

## 3. Selftest on the GPU (seconds)

    ./tsfactor --selftest

32 PASS lines as before. (Last time I wrote "16 beginning with `EC:`"; it is
15. The count of 32 was right.) The line `EC: certified gcd == NTL` now says
90 gcds.

## 4. Bench

    ./tsfactor --bench 136279841 588 --bench-gcd
    ./tsfactor --bench 136279841 588 --bench-gcd --gpu-gcd

Compare the three gcd lines at the end with last time's:

    gcd bench (random A vs T, NTL): 50.37 s
    gcd bench, CERTIFIED (chain, CPU mults): 59.15 s   (+17.4% vs NTL)
    gcd bench, CERTIFIED (chain, GPU mults): 47.68 s   (-5.6% vs NTL)

The CPU line now uses the automatic thread count (it says so). Under
`--gpu-gcd` the products too small for the GPU now run on the threads as
well, so that line should also drop a long way. Whichever of the two is
faster is the configuration to use; I expect them to be close.

## 5. The survivor from last time

    echo 12337659 > one.txt
    ./tsfactor 136279841 one.txt --skip 262 --sched opt --no-ckpt \
        --gcd-threads 1 --pend-max 1 -v -v -v --out r5

Last time: 305.5 s with checking (280.2 s without), three gcds at 59 s each.
What to look at:

* the result line (`12337659 5816 p11a...`) — unchanged;
* `gcd ... took` — should be near the step-2 thread row;
* the banner line `interval schedule: euler-optimal, G = gcd time / time per
  degree, measured as the run goes` and, with `-v -v`, lines like
  `[schedule G 950 degrees: gcd 12.4 s, 13.05 ms/degree]`. With a cheaper gcd
  the schedule takes shorter intervals and more of them, so the interval
  boundaries will differ from last time's 263..1492, 1493..4522, 4523..9712;
* total time. `--sched-G 4000` reproduces the old boundaries if you want to
  separate the two effects.

Then once more without `--no-ckpt`, to see that `r5.ckpt*` is gone after the
result is written.

## 6. Normal use

Nothing to change in `mersenne_trinomial.config`. `TSFACTOR_ARGS=--gcd-mt,4`
is now what tsfactor does by itself on your machine; keeping it is harmless.
If step 4 shows `--gpu-gcd` ahead, add that.

If other work shares the CPU, `--gcd-mt 1` (or 2) gives the cores back.

## 7. factor_ec

Checking is now **on by default** (`-noec` turns it off; `-ec` is still
accepted), the default window is 8, and the duplicate `init_h` is replaced by
the closed-form check.

    make factor factor_ec        # in factor_ec/
    ./regress.sh                 # needs ./factor; under a minute
    ./sweep.sh                   # fault sweep incl. corrupted init; seconds

and one production-size run with the cost table:

    ./factor_ec -ect -v -skip 55 -k 55 -s0 588 -s1 589 -m 14 -q 16 136279841

Last time the table said `EC total 11.07 s`, of which `init_h duplicate
7.64 s`. The row is now called `init_h closed-form check` and should be
about 2 s, for a total near 5.5 s.

`mersenne_trinomial.py` needs no change for this: its factor_ec command line
has no `-ec`, and now needs none.

# factor_ec: building and testing

Naming, as fixed in round 3: **`factor`** is the unmodified program from the
gf2x `apps/` directory; **`factor_ec`** is that program with error checking
(`factor_ec.cpp` = `factor.cpp` + `factor_ec.patch`). Without `-ec`,
`factor_ec` behaves exactly like `factor`.

## What changed in round 4

One line: the fold is now 61 words (3904 bits) instead of 64 (4096 bits) —
`#define ECFOLDW 61`. A power-of-two fold length is blind to errors that
repeat with a power-of-two period (`errorcheck_handoff.md` §5.1). Nothing else
in `factor_ec` changed; the round went into `tsfactor`.

The banner now says `fold modulus x^3904+1`.

## The build that was validated

    g++ 13.3.0, NTL 11.5.1, GMP 6.3.0, gf2x 1.3.0 (Ubuntu packages; real gf2x,
    not the round-2 schoolbook shim), -fopenmp

at r = 1279, 3217, 4423: `regress.sh` (2700 trinomials, output identical to
`factor`, 0 false alarms) and `sweep.sh` (108 injected faults, 0 wrong answers
with checking). Not rerun at r = 136279841 with the 61-word fold; your
round-3 production runs used the 64-word one.

## Build

Keep `ec_words.h` beside `factor_ec.cpp`, then build as for `factor`:

    g++ -O2 -fopenmp -o factor_ec factor_ec.cpp halfgcd.cpp \
        -lntl -lgf2x -lgmp -lm

or `make factor factor_ec ecbench ectest2` with the Makefile here.

## Check it before trusting it

    ./ectest2                     # word primitives: all tests PASS -- 54400 checks
    ./regress.sh                  # needs ./factor and ./factor_ec; a minute
    ./sweep.sh                    # the fault sweep; a few minutes
    ./ecbench 136279841 136279253 # per-call cost of every check at your r, s

The `s` to pass `ecbench` is the *working* s after the program's flip, not the
one on the command line: r odd and s even sends s to r − s so that `fastsqr`
can be used. For `-s0 588` at r = 136279841 that is 136279253.

## A real run

    ./factor_ec -ec -ecw 8 -ect -v -skip 55 -k 55 -s0 588 -s1 589 \
                -m 14 -q 16 136279841

`-ect` prints a per-category cost table at exit. With the 61-word fold the
table should look like your round-3 one (EC total ~11 s, of which the
duplicate `init_h` ~7.6 s; ~3.4 s with `-eci0`).

For the production sieving command, checking is added like this:

    ./factor_ec -ec -ecw 8 -eci0 -v -p -f 1 -s0 101 -s1 100 -k 55 -skip 55 \
                -m 14 -q 16 -maxd 279 136279841

(`mersenne_trinomial.py` currently starts `factor_ec` without `-ec`.) `-eci0`
skips the duplicate `init_h`, which is 69% of the checking cost at this
depth; what remains is about 2%. The closed-form init check of
`errorcheck_handoff.md` §4.7 would let that protection back in for ~2 s per
`s` instead of ~7.6 s; it is implemented in tsfactor, not yet here.

## Options

    -ec        enable error checking
    -ecw w     Gerbicz check every w blocks (default 1; 8 is a good start)
    -ect       per-category cost breakdown at exit
    -eci0      skip the duplicate init_h (saves one exponentiation per s)
    -ecg       also recompute each gcd and compare
    -ecfault N flip one bit at the N-th injection opportunity (testing only)

`-ecfault` exists so you can confirm the checker works on your own hardware:

    ./factor_ec     -ecfault 8 -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217
    ./factor_ec -ec -ecfault 8 -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217

The first should report `640 irreducible` (wrong). The second should report
`640 119 pa2e1...` (right), with an `EC FAIL` line on stderr.

## Files no longer needed

`tsfactor_ec_design.md`, `ec_cuda.cuh`, `ectest_gpu.cpp` and
`eccuda_emul_test.cpp` were round 3's design for a then-frozen tsfactor.
tsfactor now has its own implementation (`gf2_ec.h`) and selftest; those four
files are superseded.

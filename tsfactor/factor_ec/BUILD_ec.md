# factor_ec: building and testing

Naming: **`factor`** is the unmodified program from the gf2x `apps/`
directory; **`factor_ec`** is that program with error checking and, since
3.01, factor_modified's faster squaring and `-Z` (`factor_ec.cpp` =
`factor.cpp` + `factor_ec.patch`).

## What changed in 3.01 (round 6)

factor_modified 2.02 (2026-08-30) merged in, without its whitespace changes:

1. **The faster squaring** for CPUs without fast BMI2 (Zen 1, Zen 2): the
   constants that depend only on r are computed once per run (`SqrConstR`),
   the per-s ones once per call; the reduce loop is split into helpers that
   the compiler can vectorise — the word-aligned case in runs of at most
   `deltaw` words, never with `__restrict__` on the overlapping windows (the
   comments say why); the interleave is a plain loop. `fastsqr`, `init_h`
   and the error-checking code that squares (the closed-form init check and
   the Gerbicz `sigma(D)`) all take the constants. In the sandbox the
   squaring at r = 136279841 is 2.5–2.7 ms.
2. **`-Z q1`**: cap the number of blocks per interval at q1 and keep going
   (`-z q1` still gives up with "u"; the last of the two given applies). In
   3.01 `-Z` caps q at q1. 2.02 kept the previous q instead, which breaks
   when q0 > q1: with `-f 1` the first interval gets 0 blocks and the
   program crashes (segmentation fault, seen at r = 4423 with `-q 8 -Z 3`),
   and with `-f 0` the intervals are skipped without being scanned and
   wrong answers come out (`103 irreducible` at r = 4423, where the truth is
   a factor of degree 56). Production command lines (`-q 16` with a larger
   `-Z`, or none) never met this case.
3. The get_certificate fix (your patch; also in 2.02).
4. Version 3.01; `certificate` initialised (silences the one g++ warning).

`regress.sh` gained four `-Z` rows (same answers as without `-Z`, including
`q0 > q1` and `-f 0`); the Makefile has `make test` and header
dependencies, and no longer names the two tools whose sources are gone.
`factor_ec.patch` is regenerated against `factor.cpp`.

Validated: `regress.sh` (all rows identical to `factor`, 0 false alarms),
`sweep.sh` (108 faults, 0 wrong answers with checking), a BMI2 build
(`-DGF2X_HAVE_BMI2_SUPPORT`) identical at r = 4423, and r = 136279841:
six survivors (skip 1, maxd 40) identical to tsfactor_cpu, and the
production shape (`-skip 55 -k 55 -m 14 -q 16 -maxd 279`, s = 588) in 84 s
with 0.135 s per degree (factor_modified 2.02: 83 s, 0.131 s per degree).

For the CPU stage there is now also **tsfactor_cpu** (`../tsfactor`), the
same program as tsfactor with the arithmetic on the CPU: automatic m, the
optimal interval schedule, the certified gcd, per-survivor checkpoints.
factor_ec stays as the factor-compatible reference.

## What changed in round 5

1. **Error checking is ON by default.** `-noec` turns it off; with `-noec`
   `factor_ec` behaves exactly like `factor`. `-ec` is still accepted and does
   nothing, so existing command lines keep working.
2. **The default Gerbicz window is 8** (was 1): `-ecw 8` is what the
   production measurements used.
3. **The duplicate `init_h` is gone.** The freshly built h-table is checked
   against the closed form of its digest instead:

       (1 + y_k) (1 + S)  =  1 + y_(k+m)   mod T,     S = sum of the table,  y_t = x^(2^t)

   which costs `k + m` squarings of one residue and one modular product
   instead of a second `init_h` (`m (k + m/2)` squarings). At `-skip 55 -m 14`
   that is 69 squarings and a product against 882 squarings: about 2 s instead
   of the 7.6 s you measured, which was 69% of all checking cost. A table that
   fails is rebuilt. `-eci0` skips the check.
4. New test option `-ecfaultinit`: flips one bit of the first h-table right
   after it is built.

Round 4's only change was the fold length (61 words instead of 64).

## The build that was validated

    g++ 13.3.0, NTL 11.5.1, GMP 6.3.0, gf2x 1.3.0 (Ubuntu packages), -fopenmp

at small r: `regress.sh` (5781 trinomials in 20 configurations, output
identical to `factor`, 0 false alarms) and `sweep.sh` (108 injected faults and
4 corrupted initialisations, 0 wrong answers with checking). **Round 5 has not
been run at r = 136279841**; your production runs used the round-3 build.

## Build

Keep `ec_words.h` beside `factor_ec.cpp`, then build as for `factor`:

    g++ -O2 -fopenmp -o factor_ec factor_ec.cpp halfgcd.cpp \
        -lntl -lgf2x -lgmp -lm

or `make factor factor_ec ecbench ectest2` with the Makefile here.

## Check it before trusting it

    ./ectest2                     # word primitives: all tests PASS -- 54400 checks
    ./regress.sh                  # needs ./factor, ./factor_ec, surv9689.txt
    ./sweep.sh                    # the fault sweep
    ./ecbench 136279841 136279253 # per-call cost of every check at your r, s

`regress.sh` compares `factor_ec` with `factor` on every s in [100, 400) at
r = 1279, 3217, 4423 in six configurations each (default, `-ecw 1/4/16`,
`-noec`, `-eci0`), then with `-mt 2`, then with the production sieving option
set (`-p -f 1 ... -maxd 279`, s values on stdin).

The `s` to pass `ecbench` is the *working* s after the program's flip, not the
one on the command line: r odd and s even sends s to r − s so that `fastsqr`
can be used. For `-s0 588` at r = 136279841 that is 136279253.

## A real run

    ./factor_ec -ect -v -skip 55 -k 55 -s0 588 -s1 589 -m 14 -q 16 136279841

`-ect` prints a per-category cost table at exit. Your round-3 table said
`EC total 11.07 s`, of which `init_h duplicate 7.64 s`. That row is now
`init_h closed-form check` and should be about 2 s.

The production sieving command needs nothing added:

    ./factor_ec -v -p -f 1 -s0 101 -s1 100 -k 55 -skip 55 \
                -m 14 -q 16 -maxd 279 136279841

## Options

    -Z q1      cap the number of blocks per interval at q1, keep scanning
    -noec      turn error checking off
    -ec        accepted; checking is the default
    -ecw w     Gerbicz check every w blocks (default 8)
    -ect       per-category cost breakdown at exit
    -eci0      skip the check of the h-table initialisation
    -ecg       also recompute each gcd and compare
    -ecfault N flip one bit at the N-th injection opportunity (testing only)
    -ecfaultinit   flip one bit of the first h-table after init (testing only)

`-ecfault` and `-ecfaultinit` exist so you can confirm the checker works on
your own hardware:

    ./factor_ec -noec -ecfault 8   -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217
    ./factor_ec       -ecfault 8   -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217
    ./factor_ec -noec -ecfaultinit -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217
    ./factor_ec       -ecfaultinit -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217

The `-noec` runs report `640 irreducible` (wrong). The others report
`640 119 pa2e1...` (right), with an `EC FAIL` line on stderr.

## Files no longer needed

`tsfactor_ec_design.md`, `ec_cuda.cuh`, `ectest_gpu.cpp` and
`eccuda_emul_test.cpp` were round 3's design for a then-frozen tsfactor.
tsfactor has its own implementation (`gf2_ec.h`) and selftest; those four
files are superseded.

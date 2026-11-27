# Round 6 (tsfactor round 10) — what to run, in order

This round restructures tsfactor into one driver (`ts_core.h`) with two
backends: the GPU program as before, and **`tsfactor_cpu`**, the same program
on the CPU (g++ only). It also tunes at start, writes one checkpoint per
survivor, replaces the interval schedule by the exact expected-cost optimum,
and merges factor_modified into **factor_ec 3.01**.

Everything was tested here under emulation and on the CPU, against a gf2x
built from your source tree, on two cores. The CUDA kernels did not change,
but `tsfactor.cu` was rewritten around them and **has not been through
nvcc**: step 1 is the important one.

Old checkpoints: round 10 cannot resume a round-9 checkpoint (`PREFIX.ckpt`).
It says so when it finds one. Finish a survivor that is half done with the
round-9 binary first, or accept that it restarts.

## 1. Build and selftests

    cd tsfactor
    make

`make` builds `tsfactor` (nvcc), `tsfactor_cpu`, `gcdbench` and `gcdstress`;
the NTL line should end `gf2x called directly` (twice). On a machine without
nvcc it builds the CPU programs and says it skipped the CUDA one.

    ./tsfactor --selftest
    ./tsfactor_cpu --selftest

Both must end `SELFTEST: all PASS`. New lines in the GPU build:

    CPU squaring: BLZ odd/even + generic vs host (3954) PASS
    CPU product mod T and mul-by-x vs host             PASS
    CPU fold pass: word runs vs windowed (77)          PASS
    CPU backend == GPU backend, step by step (10 cases) PASS
    schedule: DP == brute force, monotone at full size  PASS

The CPU build prints the same scan, EC and gcd suites (on the CPU backend)
without the GPU-only ones. If you have time, `./tsfactor --selftest
--selftest-reps 10` (the cross-backend suite exercises the real kernels a
lot).

## 2. Regression (CPU, a few minutes)

    cd ../tests
    ./ts_regress.sh ../tsfactor/tsfactor
    ./ts_regress.sh ../tsfactor/tsfactor_cpu
    ./ts_ckpt10.sh ../tsfactor/tsfactor ../tsfactor/tsfactor_cpu
    ./ts_ckpt10.sh ../tsfactor/tsfactor_cpu ../tsfactor/tsfactor

Each ends `all PASS`. (`ts_regress.sh` takes an optional second argument, an
older tsfactor, for its reference results — e.g. your round-9 binary.) The
`ts_ckpt10.sh` lines worth a look:

    a list without s=84: the other two done, its checkpoint kept and reported
    tuning: timed at a fresh start, taken from the checkpoint on resume, timed again with --retune
    schedule: [schedule costs from the checkpoint: gcd ... s, ... ms/degree: G ... degrees]
    lock: a second run with the same --out refuses (status 1)
    cross-build resume: tsfactor -> tsfactor_cpu: tuned again, state carried over, results identical

## 3. The step-5 survivor again (the number I most want from the GPU)

    echo 12337659 > one.txt
    ./tsfactor 136279841 one.txt --skip 262 --gpu-gcd --gcd-threads 1 \
        --pend-max 1 -v -v -v --out r6

What to look at:

* the start: `tuning: FFT fused tlv=3 blv=3 maxch=0, m=... (t_sq ..., t_mul
  ...) (N s)` — the sweep should pick 3/3 as your round-5 bench did, and m
  near 30; tell me N (seconds);
* the intervals. Round 9 took `263..1342`, `1343..5572`, `5573..21862` and
  scanned 16,000 degrees past the factor. Round 10 should take something
  like `263..1062`, `1063..2598`, `2599..5062`, `5063..8550` (the exact ends
  depend on the measured G and the tuned m);
* the result line, `12337659 5816 p11a...`, unchanged;
* total time: about **180 s** instead of 342 s.

Then once more with checkpoints, interrupted (Ctrl-C) in the third or fourth
interval, and resumed with the same command. The resumed run should say

    tuning: from the checkpoint of s=12337659 (made by this binary on this device): ...
    Resumed s=12337659 at interval ... block .../...
       [schedule costs from the checkpoint: gcd ... s, ... ms/degree: G ... degrees]

and finish with the same result line and no `r6_r136279841_s12337659.ckpt*`
left behind. `--retune` on the resume makes it time again instead.

## 4. Setting a survivor aside (optional at full size; step 2 did it small)

    printf "12337659\n<another survivor>\n" > two.txt
    ./tsfactor 136279841 two.txt --skip 262 --gpu-gcd --out r6b ...   # Ctrl-C during the first
    echo <another survivor> > b.txt
    ./tsfactor 136279841 b.txt --skip 262 --gpu-gcd --out r6b ...     # "kept for later: 12337659"
    ./tsfactor 136279841 one.txt --skip 262 --gpu-gcd --out r6b ...   # resumes 12337659

## 5. tsfactor_cpu against factor_ec on your Zen 2

    ./tsfactor_cpu --bench 136279841 588

prints the squaring, product, reduction and checking times and the
projected cost per degree. In the sandbox: squaring 3.0–3.4 ms, product
1.1 s, m ≈ 20, ~0.12 s per degree plus 3% for checking. (Round 1's notes had
the original factor.cpp's fastsqr at ~12 ms on your Zen 2; factor_modified
made it faster there; tsfactor_cpu's squaring runs at factor_modified's speed
on the same data here.)

Then one production-shape survivor with each program, one after the other:

    cd ../factor_ec && make factor_ec && cd ../tsfactor
    time ../factor_ec/factor_ec -v -k 55 -skip 55 -m 14 -q 16 -maxd 279 -s0 588 -s1 589 136279841
    echo 588 > s588.txt
    time ./tsfactor_cpu 136279841 s588.txt --skip 55 --maxd 279 --out c588 -v -v

In the sandbox: factor_ec 84.3 s (scan 0.135 s/degree, NTL gcd 48 s);
tsfactor_cpu 84.7 s including 3.7 s of tuning (scan 0.130 s/degree,
certified gcd 44 s, one interval). Both say the same thing about 588
(factor_ec prints `588 irreducible` under `-maxd`, tsfactor `588 u`). On
your machine the certified gcd is 36 s against NTL's 51 s, so tsfactor_cpu
should come out ~15 s ahead per survivor, plus ~4 s of tuning once per
process.

**The deployment number:** the same pair of runs with all six cores busy —
six copies of each program on six different survivors at once (one process
per core is tsfactor_cpu's default). Memory bandwidth, not the cores, may
decide this, and it is what the cloud cost depends on. Each tsfactor_cpu
process needs about 1.2 GB at r = 136279841.

## 6. If you want tsfactor_cpu in the pipeline

`mersenne_trinomial.py`'s factor_ec mode would need a variant command line
(I have not changed your script):

    tsfactor_cpu 136279841 /dev/stdin --skip <k> --maxd <maxd> --out <prefix> -v

with the batch on stdin as now. The `Interval a..b:` lines and the result
lines (`s d p<hex>`, `s u`) have factor's format; the gcd line differs
(`   gcd a..b took 47.1`); a `u` means no factor up to the last interval's end,
which may be a few degrees past `--maxd` (whole blocks). Use one `--out`
prefix per worker (each prefix is locked to one process). Happy to make that
change next round if you want it.

## 7. factor_ec 3.01

    cd ../factor_ec
    make            # factor, factor_ec and the tools; `make test` runs both scripts
    ./regress.sh    # now also four -Z rows
    ./sweep.sh

`regress.sh` must end `all regressions identical to the original factor`,
with the new rows

    r=4423 -Z 3                         300 trinomials  OK, 0 false alarms
    r=4423 -q 8 -Z 3 (q0 > q1)          300 trinomials  OK, 0 false alarms
    r=4423 -f 0 -q 8 -Z 3               300 trinomials  OK, 0 false alarms
    r=4423 -z 2 -Z 3 (-Z wins)          300 trinomials  OK, 0 false alarms

One production-size run with the checking cost table (still an open item
from round 5):

    ./factor_ec -ect -v -skip 55 -k 55 -s0 588 -s1 589 -m 14 -q 16 136279841

The squaring is now factor_modified's: compare the `squares/products` line
with factor_modified 2.02 on the same command line.

## 8. irred

See the irred devkit's `docs/TESTPLAN.md` (round 3, irred 0.22).

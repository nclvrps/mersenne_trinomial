# Round 7 (tsfactor round 11) — what to run, in order

A small tsfactor round: the R6 fixes (nvcc warnings, the lock file, the
"kept for later" note, no tuning when there is nothing to do), exact
`--maxd` in the CPU stage, and "irreducible" instead of "primitive" for
non-Mersenne r. The round's main item is GPU irred: `irred/docs/TESTPLAN.md`.

## 1. Build and selftests

    cd tsfactor/tsfactor
    make
    ./tsfactor --selftest | tail -1        # SELFTEST: all PASS
    ./tsfactor_cpu --selftest | tail -1    # SELFTEST: all PASS

nvcc should now be silent: no warning about `allow_no_ntl`, and none about
`snprintf` in `GpuPlatform::fingerprint`.

## 2. Regression (a few minutes)

    make regress
    cd ../tests && ./ts_regress.sh ../tsfactor/tsfactor && cd ../tsfactor

Both end in `all PASS`.

## 3. The CPU stage, exact maxd (3 minutes)

    echo 588 > s588.txt
    time ./tsfactor_cpu 136279841 s588.txt --skip 55 --maxd 279 --out c588 -v -v

Expect a line like

    block size 16 instead of 17 for the scan 56..279 (one interval): 14 blocks end at --maxd exactly (m=17: at 293)

then `Interval 56..279:` and `== s=588 done in ... (224 degrees) ==`, about
2 s less than round 10's 72.0 s. After the run, `c588.lock` is gone.

Run the same command again: it must say

    done: nothing to do -- all 1 survivors of s588.txt are in c588.results.txt

at once, without tuning.

**Still the most useful CPU number** (TESTPLAN_R6 step 5, second part): six
tsfactor_cpu processes on six different survivors at once, each with its own
`--out`, against six factor_ec processes the same way. Memory bandwidth may
decide the per-core speed when all cores are busy, and the cloud cost
depends on that.

## 4. The "kept for later" note

As in TESTPLAN_R6 step 4 (set a survivor aside, run another list): the note
now stands out between blank lines,

    *** NOTE: 1 checkpoint(s) for s not in this list are kept for later: 12337659 ***

and is repeated as the last line of the run.

## 5. What to send back

The output of step 3 (the block-size line, the interval and the time), and
anything unexpected from steps 1–2.

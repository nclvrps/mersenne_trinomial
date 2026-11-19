# irred 0.21 — what to run on the real machine

Everything here uses the devkit's Makefile (gcc with OpenMP). Times are for
the sandbox's 2-core Xeon.

## 1. Build and selftest (1 minute)

    cd irred_devkit_r2
    make                 # build/irred (+ test programs)
    make selftest        # seven lines "selftest passed"

`build/irred --selftest` covers the known cases checked and unchecked, one
injected fault at every site, and the linear-bug demonstration. Every line
must say `ok`.

Optional, the full suite (about 15 minutes): `make test`. It runs the
reference comparison, the fault sweeps, the checkpoint tests and the 0.14
bug test, and must end without "FAIL" or "MISMATCH". Compare with `logs/`.

## 2. Measure threads at production size (stop the 0.20 run first)

Two runs competing for memory bandwidth would make both measurements
meaningless. So first stop the running 0.20 process with Ctrl-C (or
`kill -INT`). It writes a checkpoint and exits with status 2.

Then, in a **scratch directory** (not the run's directory), with the s of
the real run:

    for nt in "" --nt; do
      for n in 1 2 3 4 6; do
        echo "threads $n $nt"
        timeout -s TERM 90 /path/to/build/irred --no-ckpt --progress 2000 \
            --threads $n $nt 136279841 S | grep '^iter' | tail -1
      done
    done

Note the ms/iter of each (about 15 minutes in all). On Zen 2 the interleave
is AVX2 (`-v` shows "AVX2 interleave"). `--nt` (streaming stores) was much
slower here, where everything fits in the L3, but may help on Zen 2, where
it does not. 0.20 does not need measuring: it runs at the
`--threads 1` speed of 0.21, minus about 0.2%.

## 3. Continue the 0.20 run with 0.21

In the run's directory, the same command as before, plus the best
`--threads`:

    /path/to/build/irred --threads N [--nt] [--progress P] 136279841 S

Expected at start:

    Resuming from irred_r136279841_sS.ckpt (written ...): K/136279841 squarings done (...) in ..., residue HHHH
      (a version 1 checkpoint from irred 0.20: its K squarings were done without error checking)
    ...
    Error checking: Gerbicz XOR chain, blocks of 729 squarings, windows of up to 729 blocks; map check passed

and at the end, before the result line:

    Error checking: ... windows verified
    Note: K of the 136279841 squarings were NOT error-checked

From now on, do not start 0.20 on this directory again. It cannot read the
version 2 checkpoints that 0.21 writes.

Ctrl-C now takes a few seconds (the block is finished and verified first).
A second Ctrl-C kills at once and keeps the previous checkpoints.

## 4. Optional: confirm the unchecked first K squarings

The resume line prints the 64-bit residue of the 0.20 state at K. A fresh
0.21 run computes the same state independently, and with checking:

    mkdir verify && cd verify
    /path/to/build/irred --no-ckpt --progress K 136279841 S

It prints `iter K/136279841 ... residue HHHH` after K squarings; compare
with the resume line, then stop it. Equal residues mean the 0.20 prefix was
right (a wrong state matching in 64 bits by chance is 2^-64). This costs K
squarings of CPU, so it is only worth it if the final result matters before
an independent confirmation (e.g. a factor) is available. Run it after the
main run, or on another machine, not alongside it.

## 5. What to send back

* The ms/iter of step 2.
* The start and end of the run's output (the lines quoted in step 3) and
  any "ERROR CHECK FAILED" lines. Each such line is a real hardware fault
  that would otherwise have spoiled the result, so it is worth knowing about.

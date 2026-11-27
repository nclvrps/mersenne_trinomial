# irred 0.22 — what to run on the real machine (round 3)

Everything here uses the devkit's Makefile (gcc with OpenMP). Round 3 is
small: version 1 (0.20) checkpoints are no longer read, `--nt` chooses
itself, and `--stop-after N` replaces `timeout` in timing runs.

## 1. Build and selftest (1 minute)

    cd irred_devkit_r3
    make
    make selftest        # eight lines "selftest passed" (new: --nt)

Optional, the full suite (about 15 minutes): `make test`. It must end
without "FAIL" or "MISMATCH". `make ckpt` alone runs the 22 checkpoint
checks in two minutes.

## 2. The automatic `--nt` (1 minute)

    build/irred --no-ckpt --progress 2000 --stop-after 30000 136279841 12338357

On your Zen 2 the second or third line should read

    Streaming stores: on (squaring buffers 33 MB > 16 MB L3 cache; --no-nt turns them off)

and the `iter` lines should show ~1.76 ms/iter, your `--nt` figure of last
round. The run stops by itself after 30000 squarings plus the rest of the
block (exit status 2): no `timeout` needed, so nothing kills `grep` or
`tail` in a pipeline. (That was what happened last time: Ubuntu 26.04's Rust
`timeout` signals the whole pipeline, so `tail` died before printing; irred
was flushing every line.)

If you want the whole table again, with output safe in a pipeline:

    for nt in --no-nt --nt; do
      for n in 1 2 4; do
        echo "threads $n $nt"
        build/irred --no-ckpt --progress 2000 --stop-after 30000 \
            --threads $n $nt 136279841 12338357 | grep '^iter' | tail -1
      done
    done

## 3. A 0.20 checkpoint is refused, not overwritten

Only if you still have a directory with a 0.20 run in it: 0.22 must stop
with

    ERROR: a checkpoint of irred 0.20 (format version 1) for this r and s is here.

and exit status 1, leaving the files alone. Finish such a run with 0.21.

## 4. What to send back

* The line of step 2 (streaming stores on or off, and the ms/iter).
* Anything unexpected from `make test`.

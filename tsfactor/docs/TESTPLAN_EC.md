# tsfactor error checking — what to run on the real machine

Everything in this round was validated under CPU emulation. Nothing has been
through `nvcc` or run on a card. The steps below are ordered so that each one
is cheap and tells you whether the next is worth running. Please send back the
output of whichever you run; **steps 1, 2, 4 and 5 matter most.**

Build exactly as before (`gf2_ec.h`, `gcdbench.cpp` and `gcdstress.cpp` are
the only new source files):

    make tsfactor
    make gcdbench gcdstress      # CPU only, need NTL + gf2x

If `nvcc` rejects something, send the error; the checking kernels are plain
gather loops written in the style of the existing ones, so it should be minor.

## 1. Selftest (seconds)

    ./tsfactor --selftest

Expect `SELFTEST: all PASS` and **32** PASS lines (16 old + 16 beginning with
`EC:`). The `EC:` kernel suites are the first time those kernels run with real
concurrency, so this is the test that matters most. The table in the middle

      h      fired 30  detected 30  wrong WITH checking 0  | wrong WITHOUT 29 of 30

is a fault-injection sweep: one bit flipped at the N-th opportunity of each
site, with checking on and off. `wrong WITH checking` must be 0 everywhere.

    ./tsfactor --selftest --selftest-reps 20

## 2. The gcd — the one default that changes run time (CPU only)

    ./gcdstress                      # about a minute; must end in PASS
    ./gcdbench 136279841 588         # five gcds; roughly 5 x your NTL gcd time

`gcdbench` prints

      NTL GCD                          what tsfactor used until now (unchecked)
      CERTIFIED, NTL XGCD (--ecg-xgcd)
      hybrid HGCD, full apply (old)    the engine as it was
      hybrid HGCD, split apply         the engine now
      CERTIFIED, chain (the default)   the engine now, every step verified

tsfactor's default gcd is now the last row instead of the first. In the
sandbox (untuned distro gf2x) the last row was 24–27% **faster** than the
first at 4 and 16 Mbit, and 32% faster at 136 Mbit (972 s against 658 s — a
slow machine). Your gf2x is tuned and mine was not, so **this is the
number I most want to see.** If the last row is slower than the first on
your machine, `--gcd-ntl` gives the old gcd back (unchecked), and tell me.

`./gcdbench 136279841 588 4194304 quick` skips the two slow reference rows.

**And with more than one core for the gcd:**

    ./gcdbench 136279841 588 4194304 quick 4

adds a row for the certified chain with the independent products of each
matrix step on 4 threads (tsfactor `--gcd-mt 4`). The sandbox has two
virtual CPUs: enough to verify that this is correct, not to say how fast it
is (2 threads gave 1.29x there). The recursion itself, the division steps
and the smallest products stay on one thread, so do not expect 4x from 4.
Under `--pend-max 1` the scan waits for the gcd, so if you have cores idle at
that moment this is where they pay off. Try 2, 4 and 6 if you have the time.

## 3. The emulated build still passes on your machine (optional, a minute)

    make emul-tsfactor && ./build_emul/tsfactor --selftest
    cd tests && ./ts_regress.sh ../build_emul/tsfactor

## 4. What the checking costs on your card

    ./tsfactor --bench 136279841 588 --bench-gcd
    ./tsfactor --bench 136279841 588 --bench-gcd --gpu-gcd

New lines: four `EC ...` timings, and next to every `m=` line of the
projection `+ EC x.xxx ms (y.yy%)`. I expect well under 1% at m ~ 30. If the
`EC h-table pass` is much more than the `Horner` time above it, tell me: that
is the kernel to fuse.

`--bench-gcd` now times NTL first and then the certified gcd; with
`--gpu-gcd` also the engine with its large products on the GPU (it idles
during every gcd under `--pend-max 1`). The engine does about half the
multiplication work it did when you measured 63.0 s vs 63.6 s, so this
comparison is open again.

## 5. A real survivor, checking on vs off

Pick a survivor whose answer you know and that takes a few minutes. Three
runs, all with `-v -v -v`:

    echo <s> > one.txt
    C="136279841 one.txt --skip <d> --sched opt --no-ckpt --gcd-threads 1 --pend-max 1 -v -v -v"
    ./tsfactor $C --out noec  --no-ec           # the old program
    ./tsfactor $C --out ec                      # the new default
    ./tsfactor $C --out ect   --ect             # with the cost table

Wanted:
* the same result line in all three `*.results.txt`;
* `error checking: N blocks and M h-table chain windows verified; 0 detection(s)`
  and `K gcd(s) certified, 0 not certified`;
* `squares/products took` from the first two runs: their difference is the
  honest scan overhead (`--ect` adds device synchronisation, so its table is
  an upper bound);
* the `gcd ... took` lines from the first two runs: NTL vs the certified
  chain, in production conditions.

`Exponentiation took` now includes the init verification (one modmul plus
k+m squarings: about a quarter of a second). `--eci0` skips it.

## 6. Watch it catch something

Same survivor, one deliberately flipped bit:

    ./tsfactor $C --out f1 --ec-fault h:200        # h-table, after the 200th block
    ./tsfactor $C --out f2 --ec-fault idle:1       # h-table while the first gcd runs
    ./tsfactor $C --out f3 --ec-fault t:50         # the FFT product of block 50
    ./tsfactor $C --out f4 --ec-fault xferh:1      # an h-table snapshot in transit

Each should print `EC INJECT`, then `EC FAIL [...]`, then a restart
(`restarting from the verified state at interval ...`) — or, for `xferh`, a
silent re-read (`1 transfer re-read(s)` in the summary) — and end with the
**same result line as step 5**. `f1.ec.log` records it. (If the survivor is
resolved by its first gcd, `idle:1` lands in a table that is never read
again: nothing to detect, nothing wrong. Use a survivor that needs two or
more intervals.) Add `--no-ec` to `h:200` to see what the same flip does
unchecked: almost certainly a missed factor.

`tests/ts_regress.sh ./tsfactor` runs eleven such faults at small exponents,
plus eleven configurations against `--no-ec`, a persistent fault, and
interrupt/resume — on the real card that takes a few minutes and is worth
doing once.

## 7. Checkpoint and resume with checking on

Run step 5's second command without `--no-ckpt`, interrupt it with Ctrl-C in
the middle of an interval, run it again. It should print `Resumed s=...` and
finish with the same result. (The checkpoint is now written only after the
state passes its checks; if it ever fails them at exit you will see
`state failed verification: NOT checkpointed`.)

## 8. Using it

Nothing to change: checking is on by default, the gcd is certified by
default, and `mersenne_trinomial.py` works unmodified. Three optional things:

* Extra options can be passed through the environment, which the script's
  `tsfactor_env` config line can already set (commas instead of spaces,
  because that line is split on whitespace):

      tsfactor_env: TSFACTOR_ARGS=--gcd-mt,4         (or --gpu-gcd, --gcd-ntl, ...)

  `--gcd-mt N` (threads inside each gcd) and `--gpu-gcd` (the gcd's large
  products on the GPU) are the two candidates for a faster gcd; steps 2 and
  4 tell you which, if either, is worth turning on.

* `mersenne_trinomial.ec.patch` makes the script stop when tsfactor exits
  with status 3 (a persistent hardware fault) instead of relaunching it
  forever. Seven lines; apply it or not as you see fit.

* The script runs `factor_ec` **without** `-ec`, so the sieving stage is
  unchecked. `-ec -eci0` there costs about 2%.

Detections during production runs land in `<out>.ec.log` (for the script:
`remnants.ec.log`), which is only created if something happens. An occasional
entry over weeks is the system working. Repeated entries mean the card, its
cooling, or its power needs attention.

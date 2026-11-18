# tsfactor / factor_ec devkit — round 5

Everything needed to continue this work — in the same conversation or a new
one — and everything Gord needs to try it.

## State of the work

* **Error checking works on the real GPU.** Gord compiled the round-4
  `tsfactor` with nvcc and ran the whole test plan on an RTX 2070 SUPER:
  selftest, regression, four injected faults at production size,
  interrupt/resume — all passed. The scan checks cost about 1%.
* **Round 5 is about the gcd**, which his logs showed to be 58% of a
  survivor's wall time and slower than believed. The engine's base case was
  rewritten (word-level, exact); `--gcd-mt` is automatic; the interval
  schedule measures its own gcd/scan ratio. In the sandbox, with gf2x built
  from Gord's source, the certified gcd went from 6% slower than NTL to 28%
  faster on one thread, and the serial fraction from ~50% to ~13%.
* **Round 5 changed host code only** and has **not** been through nvcc or run
  on a GPU, nor on more than two cores.
* `factor_ec`: checking on by default, closed-form init check.
* Documents: `docs/errorcheck_handoff.md` (the mathematics, measurements and
  the ledger of what is verified where — §13), `docs/tsfactor_handoff.md`
  (the complete tsfactor record, Rounds 1–9), `docs/TESTPLAN_R5.md` (what to
  run now).

## Layout

    README_DEVKIT.md
    docs/
      errorcheck_handoff.md   mathematics, measurements, status ledger (§13)
      tsfactor_handoff.md     COMPLETE: Gord's file (through Round 7) with
                              Round 8 (error checking) and Round 9 (this
                              round) appended.  Replaces tsfactor_handoff.md
                              and the earlier tsfactor_handoff_EC.md.
      TESTPLAN_R5.md          what to run on the real machine, in order
    tsfactor/                 the complete source tree
      tsfactor.cu
      gf2_hgcd_base.h         NEW  word-level base case of the gcd engine
      gf2x_check.cpp          NEW  Makefile probe: is gf2x.h usable?
      gf2_gcd.h, gf2_gcd_ntl.cpp   the gcd engine
      gf2_ec.h                error-checking folds and kernels (unchanged)
      gcdbench.cpp            CPU-only gcd timing, every engine generation
      gcdstress.cpp           CPU-only gcd cross-checks
      Makefile
      gf2_cantor_*.h, gf2_trinomial_cuda.h, cuda_emul.h, ntl_check.cpp   unchanged
    tsfactor_r4/              the files round 5 changed, as they were in round 4
    tsfactor_r5.diff          unified diff round 4 -> round 5
    tsfactor_orig/            the pre-error-checking sources (Sep 16), for
                              building the reference binary of ts_regress.sh
    tests/
      ts_regress.sh           CLI regression (configurations, faults, resume)
      surv3217.txt surv4423.txt surv9689.txt, mksurv.py
    factor_ec/                the CPU program: sources, patch, tools, scripts
      BUILD_ec.md             start here for factor_ec
    experiments/
      fftbatch_bench.cpp      the measurement behind "CPU frequency-domain
                              batching: not adopted" (needs a gf2x that
                              exports its FFT interface, i.e. a source build)
    orchestrator/
      mersenne_trinomial.ec.patch   stop on tsfactor exit status 3 (Gord has
                              applied it)
    logs/                     outputs of the validation runs quoted in the docs

## Rebuilding the validation from scratch (no GPU)

**Use a source-built gf2x.** The distribution's libgf2x has no PCLMUL code
and multiplies about ten times slower; gcd timings taken with it are
meaningless for production (round 4 learned this the hard way). From Gord's
gf2x tree and patches:

    cd gf2x && for p in ../gf2x-patches/*.patch; do patch -p1 < $p; done
    autoreconf --install && ./configure --prefix=$HOME/gf2x_inst && make -j && make install

The packaged NTL links `libgf2x.so.3` while the source build is
`libgf2x.so.4`; the ABI of `gf2x_mul` is the same, so a directory of
symlinks is enough to make everything use the fast one:

    mkdir fastlib && cd fastlib
    for n in libgf2x.so libgf2x.so.3 libgf2x.so.4; do
        ln -s $HOME/gf2x_inst/lib/libgf2x.so.4.0.0 $n; done
    export LD_LIBRARY_PATH=$PWD

(`sudo apt-get install g++ make libntl-dev libgf2x-dev libgmp-dev autoconf
libtool` provides the rest; `libgf2x-dev` is still wanted for `gf2x.h`.)
A quick check that it took: `./gcdbench 16000057 5000001 524288 quick` should
show an NTL gcd of 3–4 s, not 40.

    cd tsfactor
    make emul-tsfactor gcdbench gcdstress   # "== NTL: OK (...), gf2x called directly =="
    ./build_emul/tsfactor --selftest        # 32 PASS lines
    ./gcdstress                             # ends in PASS
    ./gcdbench 136279841 588                # ~6 min; compare with the docs

    cd ../tsfactor_orig                     # reference binary, untouched sources
    cp ../tsfactor/cuda_emul.h ../tsfactor/ntl_check.cpp .
    make emul-tsfactor
    cd ../tests
    ./ts_regress.sh ../tsfactor/build_emul/tsfactor ../tsfactor_orig/build_emul/tsfactor

Variants that were run before shipping:

    make CXXFLAGS="-O1 -g -march=native -fsanitize=address,undefined \
         -fno-omit-frame-pointer -fno-sanitize-recover=undefined" emul-tsfactor
    make NTL_OK=0 ALLOW_NO_NTL=1 emul-tsfactor         # no NTL at all
    make GF2X_OK=0 gcdstress                           # products through NTL
    g++ ... -fsanitize=thread ... gcdstress.cpp gf2_gcd_ntl.cpp ...
    g++ -O2 -mno-pclmul ... gcdstress.cpp ...          # portable carry-less multiply

Where one gcd's time goes (single thread):

    g++ -O2 -march=native -DHYB_PROF -DHAVE_NTL -DHAVE_GF2X -o gcdbench_prof \
        gcdbench.cpp gf2_gcd_ntl.cpp -lntl -lgf2x -lgmp -pthread
    ./gcdbench_prof 136279841 588 4194304 quick 1

`emul-tsfactor` rewrites `kernel<<<blocks, threads>>>(...)` launches into
sequential loops (`cuda_emul.h`). It exercises the real kernel bodies and all
host logic. It cannot show a data race and it cannot time a GPU.

The production-size emulated run (3 minutes, 1.5 GB):

    echo 320 > s320.txt
    ./build_emul/tsfactor 136279841 s320.txt --skip 55 --m 8 --q0 4 \
        --out p320 --gcd-threads 1 --pend-max 1 --ect -v -v
    # expect:  320 74 p4855d483a53d52f8989

## Conventions worth keeping

* **`factor`** = the unmodified gf2x `apps/factor.cpp`; **`factor_ec`** = the
  checked version.
* Gord is not a mathematician and does not have the sandbox: he compiles and
  benchmarks locally (6-core Zen 2, RTX 2070 SUPER 8 GB, Ubuntu 26.04, tuned
  source-built gf2x). Deliverables are sources + a self-test + a bench tool +
  build notes, and **every new code path gets a test that can be run without
  reading the code**.
* tsfactor options use `--long` form; factor_ec keeps `-short` form.
* Claims are sorted into "verified on the real machine" / "verified here" /
  "not verified" in `errorcheck_handoff.md` §13. Keep that ledger honest, and
  say on which gf2x build a timing was taken.
* The fold length is 61 words on purpose. Do not round it to 64.
* tsfactor works in the regime `2s < r`; factor_ec in `s > r/2`. The
  reduction checks differ for that reason and are not interchangeable.
* Production invocation of tsfactor (via `mersenne_trinomial.py`):
  `--sched opt --ckpt-mins 30 --gcd-threads 1 --pend-max 1 -v -v -v`,
  survivors on stdin, `--skip` = the depth already sieved. `--pend-max 1`
  is deliberate (no speculative work on a home PC's card). Extra options go
  in through `tsfactor_env: TSFACTOR_ARGS=...` (commas, no spaces).
* A purported primitive trinomial would be verified independently (`irred`,
  weeks of CPU) before anything is announced; see §2 of the handoff.
* tsfactor will be GPL (derived in spirit from `factor`); NTL is LGPL, gf2x
  LGPL/GPL: borrowing code from either is acceptable if it helps.

## Where to pick up

`errorcheck_handoff.md` §12 and the end of Round 9 in `tsfactor_handoff.md`
have the ordered lists. The first items need Gord's machine: the nvcc
build, `gcdbench` on real cores, `--bench-gcd` with and without `--gpu-gcd`.
After that the gcd is probably no longer the bottleneck — at ~13 s per gcd
the scan is ~80% of a survivor again — and the next gain has to come from
the scan itself.

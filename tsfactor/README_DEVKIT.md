# Error-checking devkit (round 4): tsfactor + factor_ec

Everything needed to continue this work — in the same conversation or a new
one — and everything Gord needs to try it. Nothing here needs a GPU except
the real `make tsfactor`.

## State of the work in five lines

* `tsfactor` has error checking **on by default** and a **certified gcd by
  default** that is also faster than NTL's gcd on CPU multiplications, and
  can use several cores per gcd (`--gcd-mt N`, opt-in).
* All of it is validated under CPU emulation (selftest 32/32, regression vs
  the original binary, fault injection, sanitizers, one production-size run).
* **None of it has been compiled by nvcc or run on a GPU.** That is step one.
* `factor_ec` changed by one constant (fold length 61).
* The mathematics is in `docs/errorcheck_handoff.md`; the tsfactor record is
  `docs/tsfactor_handoff_EC.md`; what to run is `docs/TESTPLAN_EC.md`.

## Layout

    README_DEVKIT.md          this file
    docs/
      errorcheck_handoff.md   the mathematics, measurements, status ledger (§13)
      tsfactor_handoff_EC.md  tsfactor integration record; APPEND to tsfactor_handoff.md
      TESTPLAN_EC.md          what to run on the real machine, in order
    tsfactor/                 the complete source tree (drop-in for the Sep-16 one)
      tsfactor.cu             2844 -> 4816 lines
      gf2_ec.h                NEW  fold algebra (host) + checking kernels (device)
      gf2_gcd.h, gf2_gcd_ntl.cpp   certified gcd, split application
      gcdbench.cpp            NEW  CPU-only gcd timing        (make gcdbench)
      gcdstress.cpp           NEW  CPU-only gcd cross-check   (make gcdstress)
      Makefile                gf2_ec.h dependencies, two new targets; the
                              emul-tsfactor rule no longer copies
                              gf2_small_field.h / gf2_field_setup.h (tsfactor
                              does not include them; without them in the
                              directory the rule printed two harmless errors)
      gf2_cantor_*.h, gf2_trinomial_cuda.h    unchanged
      cuda_emul.h, ntl_check.cpp              unchanged (needed to build; not in
                              the tsfactor tarball: cuda_emul.h is the copy
                              from factor_ec_Round3.tar.gz, ntl_check.cpp
                              from the earlier tsfactor devkit)
    tsfactor_orig/            the Sep-16 sources exactly as received
    tsfactor_ec.diff          unified diff tsfactor_orig -> tsfactor
    tests/
      ts_regress.sh           CLI regression (configurations, faults, resume)
      surv3217.txt surv4423.txt surv9689.txt    survivor lists for it
      mksurv.py               how those lists were made (brute force)
    factor_ec/                the CPU program: sources, patch, tools, scripts
      BUILD_ec.md             start here for factor_ec
    orchestrator/
      mersenne_trinomial.ec.patch   optional: stop on tsfactor exit status 3
    logs/                     outputs of the validation runs quoted in the docs

## Rebuilding the validation from scratch (no GPU)

    sudo apt-get install g++ make libntl-dev libgf2x-dev libgmp-dev

    # tsfactor/ is the complete tree.  (Or: copy tsfactor_orig/ and run
    # `patch -p1 < ../tsfactor_ec.diff` inside the copy -- same result.)
    cd tsfactor
    make emul-tsfactor                      # -> build_emul/tsfactor (CPU emulation)
    ./build_emul/tsfactor --selftest        # 32 PASS lines, ~15 s
    make gcdbench gcdstress
    ./gcdstress                             # ~1 min, must end in PASS
    ./gcdbench 16000057 5000001 524288      # ~4 min; compare with the docs

    # reference binary from the untouched sources
    cd ../tsfactor_orig
    cp ../tsfactor/cuda_emul.h ../tsfactor/ntl_check.cpp .
    make emul-tsfactor

    cd ../tests
    ./ts_regress.sh ../tsfactor/build_emul/tsfactor ../tsfactor_orig/build_emul/tsfactor

    # variants that were run before shipping
    cd ../tsfactor
    make CXXFLAGS="-O1 -g -march=native -fsanitize=address,undefined \
         -fno-omit-frame-pointer -fno-sanitize-recover=undefined" emul-tsfactor
    make CXXFLAGS="-O2 -march=native -Wall -Wextra -DEC_NBLK_MAX=1" emul-tsfactor
    make NTL_OK=0 ALLOW_NO_NTL=1 emul-tsfactor

`emul-tsfactor` rewrites `kernel<<<blocks, threads>>>(...)` launches into
sequential loops (`cuda_emul.h`). It exercises the real kernel bodies and all
host logic. It cannot show a data race and it cannot time a GPU.

The production-size emulated run (14 minutes on the sandbox, 1.5 GB):

    echo 320 > s320.txt
    ./build_emul/tsfactor 136279841 s320.txt --skip 55 --m 8 --q0 4 \
        --out p320 --no-ckpt --gcd-threads 1 --pend-max 1 --ect -v -v
    # expect:  320 74 p4855d483a53d52f8989

## Conventions worth keeping

* **`factor`** = the unmodified gf2x `apps/factor.cpp`; **`factor_ec`** = the
  checked version. (Older text sometimes says "factor" for both.)
* Gord is not a mathematician and does not have the sandbox: he compiles and
  benchmarks locally (6-core Zen 2, RTX 2070 SUPER 8 GB). Deliverables are
  sources + a self-test + a bench tool + build notes, and **every new code
  path gets a test that can be run without reading the code**.
* tsfactor options use `--long` form; factor_ec keeps `-short` form.
* Claims are sorted into "verified by execution" / "by algebra" / "not
  verified" in `errorcheck_handoff.md` §13. Keep that ledger honest: round 1
  shipped an untested patch and it cost 45 minutes of machine time.
* The fold length is 61 words on purpose. Do not round it to 64.
* tsfactor works in the regime `2s < r`; factor_ec in `s > r/2`. The
  reduction checks differ for that reason and are not interchangeable.
* Production invocation of tsfactor (via `mersenne_trinomial.py`):
  `--sched opt --ckpt-mins 30 --gcd-threads 1 --pend-max 1 -v -v -v`,
  survivors on stdin, `--skip` = the depth already sieved. Extra options go
  in through `tsfactor_env: TSFACTOR_ARGS=...` (commas, no spaces).

## What was NOT in the round-4 input and is therefore not updated

* The current `tsfactor_handoff.md`. The copy available ends at Round 6c and
  predates the fine-DDF standby / `--hsnap-mb` round, so
  `tsfactor_handoff_EC.md` is written as a section to append, not as a
  replacement.
* `README.md` of the tsfactor package.

## Where to pick up

`errorcheck_handoff.md` §12 has the ordered list. The first three items all
need Gord's machine: the GPU selftest, `gcdbench` on the tuned gf2x, and
`--bench-gcd --gpu-gcd`. After that the largest remaining technical item is
CPU frequency-domain batching in the gcd engine (§6.5).

# tsfactor / tsfactor_cpu / factor_ec devkit — round 6 (tsfactor round 10)

Everything needed to continue this work — in the same conversation or a new
one — and everything Gord needs to try it. The tree is laid out like the
`tsfactor/` directory of github.com/nclvrps/mersenne_trinomial.

## State of the work

* **x^136279841 + x^12338357 + 1 is primitive**: tsfactor found no factor,
  irred confirmed irreducibility (Gord, round 5 of the test plans).
* **One driver, two backends.** `tsfactor.cu` is split into `ts_core.h` (the
  whole program apart from the arithmetic on residues: schedule, certified
  gcd, checkpoints, error checking, fine DDF, selftest, main) and backends
  behind an interface (`TsDev`): the GPU one in `tsfactor.cu` and a new CPU
  one in `ts_cpu.h`. **`tsfactor_cpu`** is the same program with the CPU
  backend, built with g++ alone: for the CPU stage (survivors of
  coarse_sieve up to d ≈ 279) on cheap cores, replacing factor_ec there. It
  scans at factor_modified's speed per degree and adds the automatic m, the
  certified gcd (29% faster than NTL's on Gord's machine), checkpoints and
  the schedule. The state is bit-identical between backends: a checkpoint of
  one build resumes in the other.
* **Tuning at start** (FFT configuration on the GPU, block size m on both),
  stamped into checkpoints with a fingerprint; a resume on the same binary
  and device reuses it; `--retune`, `--no-tune`, `--fft`, `--m` override.
  The GF2C_* environment variables are gone.
* **One checkpoint per survivor**: `PREFIX_r<r>_s<s>.ckpt`; a survivor can be
  set aside and resumed later; one process per prefix (`PREFIX.lock`).
  Round-9 checkpoints cannot be resumed (and are reported if found).
* **The interval schedule is now the exact expected-cost optimum** (dynamic
  programming over the survival law); round 9's recurrence was 41% above it
  in Gord's step-5 run (that survivor: ~177 s instead of 342 s). The
  measured gcd/scan costs are saved in checkpoints.
* **factor_ec 3.01**: factor_modified 2.02 merged in (faster squaring, `-Z`;
  `-Z` fixed for q0 > q1).
* **Not compiled by nvcc** (no CUDA in the sandbox). The kernels are
  unchanged; `tsfactor.cu` was rewritten around them. Everything was tested
  under emulation and on the CPU, with gf2x built from Gord's source tree.
* Documents: `docs/tsfactor_handoff.md` (complete record, Rounds 1–10),
  `docs/errorcheck_handoff.md` (the mathematics and the ledger, §13),
  `docs/TESTPLAN_R6.md` (what to run now).

## Layout

    README_DEVKIT.md
    docs/
      tsfactor_handoff.md     COMPLETE record, Rounds 1-10 (Round 10 at the end)
      errorcheck_handoff.md   mathematics, measurements, status ledger (§13)
      TESTPLAN_R6.md          what to run on the real machine, in order
      TESTPLAN_R5.md          last round's (results in logs/TESTPLAN_R5)
    tsfactor/                 the complete source tree
      ts_core.h               NEW  driver: everything but the arithmetic
      ts_cpu.h                NEW  CPU backend (TsCPU) + its kernel selftests
      tsfactor_cpu.cpp        NEW  the CPU program (g++ only)
      gf2_trin_host.h         NEW  host bit-level oracles (no CUDA)
      tsfactor.cu             kernels (unchanged), TsGPU, GPU gcd hook, FFT
                              sweep, bench, main
      gf2_trinomial_cuda.h    (oracles moved out)   gf2_cantor_cuda.h (no env vars)
      gf2_gcd.h, gf2_gcd_ntl.cpp, gf2_hgcd_base.h   the gcd engine (unchanged)
      gf2_ec.h                error-checking folds and kernels (unchanged)
      gcdbench.cpp, gcdstress.cpp   CPU-only gcd timing / cross-checks
      Makefile                make | make cpu | make cuda | make emultest | make regress
      gf2_cantor_core.h, gf2_cantor_engine.h, cuda_emul.h, ntl_check.cpp, gf2x_check.cpp
    tsfactor_r10.diff         unified diff round 9 (the repository) -> round 10,
                              whole devkit tree (supersedes tsfactor_r5.diff)
    tsfactor_orig/            the pre-error-checking sources (Sep 16), for
                              building the reference binary of ts_regress.sh
    tests/
      ts_regress.sh           CLI regression (configurations, faults, resume);
                              either build
      ts_ckpt10.sh            NEW  per-survivor checkpoints, set-aside, tuning
                              adopt/retune, saved schedule costs, the lock,
                              cross-build resume
      surv3217.txt surv4423.txt surv9689.txt, mksurv.py
    factor_ec/                the CPU reference program (3.01), tools, scripts
      BUILD_ec.md             start here for factor_ec
    experiments/
      fftbatch_bench.cpp      the measurement behind "CPU frequency-domain
                              batching: not adopted"
    orchestrator/
      mersenne_trinomial.ec.patch   (applied by Gord in round 5)
    logs/
      round10/                this round's validation runs (quoted in the docs)
      TESTPLAN_R5/            Gord's results of the last test plan
      *.log                   round 9's validation runs

## Rebuilding the validation from scratch (no GPU)

**Use a source-built gf2x** (the distribution's has no PCLMUL code and
multiplies ~10x slower; timings taken with it say nothing about production).
The sandbox builds GMP, Gord's gf2x tree (with his thresholds copied into
`already_tuned/tuned/`) and NTL (`NTL_GF2X_LIB=on NTL_THREADS=on`) into
/usr/local.

    cd tsfactor
    make cpu emul-tsfactor        # "== NTL: OK (...), gf2x called directly =="
    ./build_emul/tsfactor --selftest       # SELFTEST: all PASS
    ./tsfactor_cpu --selftest              # SELFTEST: all PASS (also --threads 2)
    ./gcdstress                            # ends in PASS

    cd ../tsfactor_orig                    # reference binary, untouched sources
    cp ../tsfactor/cuda_emul.h ../tsfactor/ntl_check.cpp .
    make emul-tsfactor
    cd ../tests
    ./ts_regress.sh ../tsfactor/build_emul/tsfactor ../tsfactor_orig/build_emul/tsfactor
    ./ts_regress.sh ../tsfactor/tsfactor_cpu    ../tsfactor_orig/build_emul/tsfactor
    ./ts_ckpt10.sh  ../tsfactor/tsfactor_cpu ../tsfactor/build_emul/tsfactor
    ./ts_ckpt10.sh  ../tsfactor/build_emul/tsfactor ../tsfactor/tsfactor_cpu

(The round-9 binary works as the reference too; this round used it.)
Sanitizers, as run before shipping:

    g++ -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
        -fno-omit-frame-pointer -fopenmp -I. -DHAVE_NTL -DHAVE_GF2X \
        -o tsfactor_cpu_asan tsfactor_cpu.cpp gf2_gcd_ntl.cpp -lntl -lgf2x -lgmp -pthread
    # then --selftest, --selftest --threads 2, ts_regress.sh and ts_ckpt10.sh on it;
    # the same flags on build_emul/tsfactor.cpp for the emulated GPU build

Production size on the CPU (a few minutes each):

    ./tsfactor_cpu --bench 136279841 588
    echo 588 > s588.txt
    ./tsfactor_cpu 136279841 s588.txt --skip 55 --maxd 279 --out c588 -v -v

`emul-tsfactor` rewrites `kernel<<<blocks, threads>>>(...)` launches into
sequential loops (`cuda_emul.h`). It exercises the real kernel bodies and all
host logic. It cannot show a data race and it cannot time a GPU.

## Conventions worth keeping

* **`factor`** = the unmodified gf2x `apps/factor.cpp`; **`factor_ec`** = the
  checked version (3.01: plus factor_modified's squaring and `-Z`).
  **`tsfactor`** = the GPU build, **`tsfactor_cpu`** = the CPU build of the
  same program.
* Gord is not a mathematician and does not have the sandbox: he compiles and
  benchmarks locally (6-core Zen 2, RTX 2070 SUPER 8 GB, Ubuntu 26.04 with
  g++ 15.2 and Rust coreutils, tuned source-built gf2x). Deliverables are
  sources + a self-test + a bench tool + build notes, and **every new code
  path gets a test that can be run without reading the code**.
* Keep the two builds in step: anything a backend computes must be bit-for-
  bit what the other computes (the cross-backend selftest enforces it).
  New driver logic goes in `ts_core.h`, not in a backend.
* tsfactor options use `--long` form; factor_ec keeps `-short` form.
* Claims are sorted into "verified on the real machine" / "verified here" /
  "not verified" in `errorcheck_handoff.md` §13. Keep that ledger honest, and
  say on which gf2x build a timing was taken.
* The fold length is 61 words on purpose. Do not round it to 64.
* tsfactor (both builds) works in the regime `2s < r`, any parity of s;
  factor_ec flips to `s > r/2`. The reduction checks differ for that reason.
* Production invocation of tsfactor (via `mersenne_trinomial.py`):
  `--sched opt --gpu-gcd --ckpt-mins 30 --gcd-threads 1 --pend-max 1 -v -v -v`,
  survivors on stdin, `--skip` = the depth already sieved. Extra options go
  in through `tsfactor_env: TSFACTOR_ARGS=...` (commas, no spaces).
  tsfactor_cpu defaults to one process per core.
* A purported primitive trinomial is verified independently (irred) before
  anything is announced.
* GPL (the repository's licence); NTL is LGPL, gf2x LGPL/GPL.

## Where to pick up

The end of Round 10 in `tsfactor_handoff.md` and `errorcheck_handoff.md` §12
have the ordered lists. First: the nvcc build and selftest, the step-5
survivor under the new schedule, and tsfactor_cpu against factor_ec on Gord's
cores (six processes at once). After that: tsfactor_cpu in the orchestrator
if Gord wants it, and a GPU irred on the same driver/backend pattern.

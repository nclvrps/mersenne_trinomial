# coarse_sieve dev kit (final, round 10)

`coarse_sieve` is a GPU sieve for the least-degree irreducible factors, up to degree 55, of the trinomials x^r + x^s + 1 over GF(2), for every s ≤ r/2. It was built for r = 136279841, the exponent of the 52nd Mersenne prime, over ten rounds in October-November 2026.

The kit serves two purposes:
- **Running the sieve** on a GPU, e.g. for a new Mersenne exponent. See `docs/coarse_sieve_handoff.md`, §6.
- **Developing it further** in a workspace with no GPU. It holds the sources, an emulated (CPU-only) build, the regression and checkpoint suites, the oracles they compare against, and the analysis tools.

Start with `docs/coarse_sieve_handoff.md`, the final handoff. §5 says how the program works, §6 how to run it, §7 what lies beyond d = 55.

## Layout

| path | what |
|---|---|
| `src/r10/` | The current sources: `coarse_sieve.cu`, `gf2_dlog.h`, `gf2_dlog_selftest.cpp`, `cuda_emul.h`, the field headers, `Makefile`, and the unit tests `t_*.cpp`. |
| `src/r5/` - `src/r9/` | Earlier rounds. Rounds 6-9 are for the checkpoint-compatibility tests: each side resumes the other's checkpoints. Round 5 is for those tests too, and it still has the dense-table and bucketed sieves that make the oracles. |
| `setup.sh` | Builds the emulated binaries into `build/` (round 10's `cs` and `csprof`, rounds 5-9 for the tests), then the oracles into `oracles/`. |
| `fixtures/` | The seed `base18.survivors.txt`, the oracle generator `gen_oracles.sh`, and `oracles.sha256`. |
| `tests/regress.sh` | The regression suite: unit tests, selftests, and sieve outputs against the oracles for every engine, loop order, fill, GPU count, sync mode and inner subfield. It also checks the tuner, the phase-timing build and the command line, then runs `ckpt_tests.sh`. |
| `tests/ckpt_tests.sh` | Checkpoints, the lock file, the watchdog, re-tuning on resume, and compatibility with rounds 5-9. |
| `tests/build_emul.sh` | One emulated build: `<src dir> <out dir> <name>`, with extra flags in `EXTRA`. |
| `nvcc/build_check.sh` | The real nvcc build, run for its diagnostics: warnings, and registers, stack and spills per kernel. |
| `tools/` | The analysis programs: `cstat.cpp`, `fillsim.cpp`, `passavg.py`, `estimate.py` (round 7), `sass_locals.py` (round 8), `randread.cu` (round 9; the GPU's random-read rate, GPU only), and `analysis/` (round 10). Each has its build line or usage in its header. |
| `tools/analysis/` | Round 10's tools: `sieve_stats.py` (the counts against the Poisson heuristic, the expected number of primitive trinomials), `factor_mersenne.py` (2^d − 1), `tower_survey.py` (table sizes per degree and engine), `pollsim/` (the polling simulation). |
| `docs/` | `coarse_sieve_handoff.md`, the final handoff, and `history/` with the handoffs of rounds 1-9. |

## Use on a GPU

    cd src/r10
    make coarse_sieve gf2_dlog_selftest           # CUDA 12+, ~6-12 min for the 116 kernels
    ./gf2_dlog_selftest && ./coarse_sieve --selftest
    ./coarse_sieve <r> 2-46 out2-46               # then one degree per run:
    ./coarse_sieve <r> 47 out47 --load out2-46.survivors.txt

The handoff's §6 has the whole procedure, the times per degree and the final consistency check.

## Use without a GPU (development)

    ./setup.sh                  # ~45 min on 2 cores, ~30 of them for one oracle (OBIG); FULL=0 skips the two
                                #   production-range oracles (~10 min, ~1 GB)
    bash tests/regress.sh       # ~50 min (FULL=0: ~40); ends "ALL REGRESSIONS PASS"
    bash tests/ckpt_tests.sh    # ~4 min (also run by regress.sh); ends "ALL CHECKPOINT TESTS PASS"

- **The emulated build** runs every CUDA kernel on the CPU. Each warp's 32 lanes run as fibers in lockstep, so the warp intrinsics (`__ballot_sync`, `__shfl_sync`) behave as on the GPU, and the emulator checks that their use is warp-uniform. Inline PTX (the optional L1 prefetches) is compiled out there.
- **What it can't show** is GPU timing. Speed has to be measured on hardware, with the tune lines, the done lines (which end with ns/class) and the phase-timing build (`make coarse_sieve_prof`).
- **Don't edit `tests/*.sh` while a suite is running.** bash reads a script as it goes, and an edit before its current line makes it misread the rest.
- **The oracles** come from round 5's dense and bucketed sieves, which share no code with the logs kernel of rounds 6-10.
  - `gen_oracles.sh` checks each one against `oracles.sha256`.
  - Two of them cover the production range s ≤ floor(r/2) up to d = 32 and take ~1 GB of disk. `FULL=0` skips them, and the regression's F tests with them.
- **For nvcc in a workspace without CUDA,** pip has it (CUDA 13 wheels; for 12.x the same names end in `-cu12`):

      pip install --target cu13 nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt nvidia-cuda-runtime \
          nvidia-cuda-cccl nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm
      C=$PWD/cu13/nvidia/cu13; export PATH=$C/bin:$C/nvvm/bin:$PATH
      NVLIB=-L$C/lib nvcc/build_check.sh

  `cuobjdump -sass -fun <kernel> build/nvcc/coarse_sieve` then shows a kernel's SASS. With `-lineinfo` added, `nvdisasm -gi` maps each instruction to its source line.

## To continue in a new conversation

Attach this kit, `docs/coarse_sieve_handoff.md`, and your logs. On the project's claude.ai page, the doc `claude/coarse_sieve_handoff.md` carries the same handoff.

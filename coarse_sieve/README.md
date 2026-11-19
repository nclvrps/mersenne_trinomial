# coarse_sieve dev kit (round 9)

Everything needed to continue developing `coarse_sieve` in a fresh workspace with no GPU: the sources, an emulated (CPU-only) build, the regression and checkpoint suites, the oracle data they compare against, and the analysis tools. It is meant to be attached to a new conversation together with the latest handoff and logs.

## Layout

| path | what |
|---|---|
| `src/r9/` | Round 9, the current sources: `coarse_sieve.cu`, `gf2_dlog.h`, `gf2_dlog_selftest.cpp`, `cuda_emul.h`, the field headers, `Makefile`, and the unit tests `t_*.cpp` (round 9 adds `t_cubic.cpp`, the cubic tower against an independent host Pohlig-Hellman). |
| `src/r5/` - `src/r8/` | Earlier rounds, for two purposes. Rounds 6-8 are for the checkpoint compatibility tests: each side resumes the other's checkpoints, and round 8 must refuse a cubic-tower checkpoint. Round 5 is for those tests too, and it still has the dense-table and bucketed sieves that make the oracles. |
| `setup.sh` | Builds the emulated binaries into `build/` (round 9's `cs` and `csprof`, rounds 5-8 for the tests), then the oracles into `oracles/`. |
| `fixtures/` | The seed `base18.survivors.txt`, the oracle generator `gen_oracles.sh`, and `oracles.sha256`, the oracles' checksums (from round 6). |
| `tests/regress.sh` | The regression suite. It covers the unit tests and the selftests. It compares sieve outputs with the oracles, for every engine (the cubic tower included), loop order, fill, GPU count and inner subfield. It also checks the tuner, the phase-timing build, the prefetch and sync switches, and the command line, then runs `ckpt_tests.sh`. |
| `tests/ckpt_tests.sh` | Checkpoints, the lock file, the watchdog, and compatibility with rounds 5-8. |
| `tests/build_emul.sh` | One emulated build: `<src dir> <out dir> <name>`, with extra flags in `EXTRA`. |
| `nvcc/build_check.sh` | The real nvcc build, run only for its diagnostics: warnings, and registers, stack and spills per kernel. |
| `tools/` | Analysis programs from round 7: `cstat.cpp` (class counts per chunk), `fillsim.cpp` (lockstep warp simulation), `passavg.py`, `estimate.py`; from round 8, `sass_locals.py` (where a kernel's local-memory accesses sit in its SASS, loop by loop); from round 9, `randread.cu` (the GPU's random-read rate, alone and beside streaming traffic; GPU only). Each has its build line or usage in its header. |

## Use

    ./setup.sh                  # ~45 min on 2 cores, ~30 of them for one oracle (OBIG); FULL=0 skips the two
                                #   production-range oracles (~10 min, ~1 GB)
    bash tests/regress.sh       # ~50 min (FULL=0: ~40, without the full-range tests); ends "ALL REGRESSIONS PASS"
    bash tests/ckpt_tests.sh    # ~3 min (also run by regress.sh)

- **The emulated build** runs every CUDA kernel on the CPU. Each warp's 32 lanes run as fibers in lockstep, so the warp intrinsics (`__ballot_sync`, `__shfl_sync`) behave as on the GPU, and the emulator checks that their use is warp-uniform. Inline PTX (round 9's L1 prefetches) is compiled out there.
- **What it can't show** is GPU timing. Speed has to be measured on hardware, with the tune lines, done lines and the phase-timing build (`make coarse_sieve_prof`).
- **Don't edit `tests/*.sh` while a suite is running:** bash reads a script as it goes, and an edit before its current line makes it misread the rest.
- **The oracles** come from round 5's dense and bucketed sieves, which share no code with the logs kernel of rounds 6-9. `gen_oracles.sh` checks each one against `oracles.sha256`. Two of them cover the production range s <= floor(r/2) up to d = 32 and take ~1 GB of disk; `FULL=0` skips them, and the regression's F tests with them.
- **For nvcc in a workspace without CUDA,** pip has it (CUDA 13 wheels; for 12.x the same names end in `-cu12`):

      pip install --target cu13 nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt nvidia-cuda-runtime \
          nvidia-cuda-cccl nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm
      C=$PWD/cu13/nvidia/cu13; export PATH=$C/bin:$C/nvvm/bin:$PATH
      NVLIB=-L$C/lib nvcc/build_check.sh

  `cuobjdump -sass -fun <kernel> build/nvcc/coarse_sieve` then shows a kernel's SASS (~6 minutes to compile all 116 kernels). With `-lineinfo` added, `nvdisasm -gi` maps each instruction to its source line.

## The state of things

The handoff of the latest round has the details. On the project's claude.ai page, the doc `claude/coarse_sieve_handoff.md` carries the current state.

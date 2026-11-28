# irred 0.23 — what to run on the real machine (round 4: GPU irred)

Round 4 adds the GPU build `irred_gpu` (the same program; the squarings run
on the GPU). The CPU build `irred` behaves as before. Everything below runs
in `irred/` of the repository.

## 1. Build and selftests (a few minutes)

    cd irred
    make                 # irred etc., and build/irred_gpu (nvcc is found)
    make selftest        # eight lines "selftest passed" (the CPU build)
    build/irred_gpu --selftest | grep -E 'GPU|passed|FAILED'

The GPU selftest must print

    selftest GPU squaring kernels vs reference (4 kernels, 5168 cases): ok
    selftest GPU == CPU, 300 squarings, 4 cases x 4 kernels: ok
    selftest passed

**This is the first time the GPU code meets nvcc** (it was validated under
the CUDA emulation only). Please send any compiler warning or error.
If `make` does not find nvcc: `make gpu NVCC=/usr/local/cuda/bin/nvcc`.

## 2. The GPU tests (about 5 minutes)

    make gputest          # = tests/gpu_test.sh build/irred_gpu build

It compares 144 results with the naive reference, checks agreement with the
CPU build, passes checkpoints between the two builds and runs the fault
sweeps on the GPU. It must end with `gpu tests: 8 passed, 0 failed`.

## 3. Kernel timings and the first residues at full size (about 2 minutes)

    build/irred_gpu --no-ckpt --progress 2000 --stop-after 30000 136279841 12338357

Please send the whole output. The interesting lines:

    GPU 0: NVIDIA GeForce RTX 2070 SUPER (sm_75, 40 SMs, ... MB)
    GPU squaring kernel: k? 0.1?? ms (fused 0.19?, ...)

`fused` is tsfactor's kernel (0.195 ms in tsfactor's tuning); the k-kernels
are new and should be faster. The `iter` lines must show these residues,
which your Zen 2 printed in round 3 (s = 12338357 is odd and below r/2, so
both builds compute the same thing):

    iter 2000  ... residue 524cef6916ae5278
    iter 4000  ... residue d61458d6bb7b0384
    iter 6000  ... residue 48d654b6eef5b870
    iter 10000 ... residue 13636a0f50a7db2c
    iter 20000 ... residue 6289c745dbd0ce06
    iter 30000 ... residue 97b488763a1f6c36

and ms/iter is the speed of the whole run (the check adds ~0.2%).

## 4. The full recheck of s = 12338357 (hours)

    build/irred_gpu --progress 1000000 136279841 12338357 > irred_gpu_12338357.log 2>&1

Expect between 4 and 8 hours (step 3's ms/iter × 136279841). It
checkpoints every 30 minutes to `irred_r136279841_s12338357.ckpt`; Ctrl-C
(or SIGTERM) finishes the block, verifies and writes a checkpoint; the same
command resumes. The last lines should be

    All 136279841 squarings were error-checked
    136279841 12338357 irreducible (may be primitive) LOG

That is the third independent result (tsfactor: no factor to r/3; CPU
irred: x^(2^r) = x), this time on other hardware with other code.

Do not run it in the same directory as a CPU irred run of the same s:
they would share the checkpoint file names (they can resume each other's
checkpoints for this s, but the lock lets only one run at a time).

## 5. What to send back

* The output of step 3 (kernel timings, residues, ms/iter).
* Anything unexpected from steps 1–2.
* Step 4's log when it is done.

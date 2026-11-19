# irred devkit — round 2 (irred 0.21, 2026-11-19)

Everything needed to continue work on `irred` — in the same conversation or
a new one — and for Gord to build, test and run it.

## State of the work

* **irred 0.21** (`irred/irred.c`) adds **Gerbicz-style error checking**.
  Squaring over GF(2) is additive, so the accumulator is an XOR and a single
  fault is always detected. A start-up **map check** against an independent
  squaring implementation covers the one thing the chain cannot see:
  deterministic linear bugs like 0.14's. Version 0.21 also adds
  `--threads N` (OpenMP; 1.7x with 2 threads at r = 136279841 in the
  sandbox), `--nt` (streaming stores, for memory-bound machines; to be
  measured there) and checkpoint format version 2. **0.21 continues runs started
  by 0.20.**
* It is validated in the sandbox only (Xeon, AVX-512, 2 vCPUs). Nothing has
  run on Gord's Zen 2 yet: see `docs/TESTPLAN.md`.
* Documents: `docs/irred_handoff.md` (the complete record of rounds 1 and 2:
  algorithm, the 0.14 bug, error-checking mathematics, measurements,
  verification ledger, next steps) and `docs/TESTPLAN.md` (what to run on
  the real machine, including moving the running 0.20 job to 0.21).

## Layout

    README_DEVKIT.md
    Makefile                  make / make test / make selftest / cmp / sweep /
                              ckpt / bug014 / bench  (builds into build/)
    irred/irred.c             irred 0.21 (the deliverable; single file)
    docs/
      irred_handoff.md        complete record, rounds 1-2
      TESTPLAN.md             steps for the real machine
    tools/
      ref.c                   naive O(r^2) bit-at-a-time reference: prints the
                              64-bit residue in irred's format ("r s x..."),
                              shares no code with irred
    reference/
      irred315_orig.c         Brent's 3.15 (2003), untouched
      irred315_compile.patch  Gord's minimal patch to compile it today; the
                              Makefile also disables its sieve so it prints a
                              residue for every s ("r s s+1" on stdin)
      irred_0.14.c            Gord's 0.14 (has the deltaq = 0 bug; kept for
                              tests/bug014_test.sh)
      irred_0.20.c            round 1's version (writes version 1 checkpoints;
                              used by the 0.20 -> 0.21 compatibility test)
      irred_0.20_repo_move.patch   Gord's patch for the repository move
    tests/
      cmp.sh                  all builds/paths vs tools/ref.c (+ 3.15 at larger r)
      ec_sweep.sh             exhaustive one-fault-per-run sweep for one site
      ec_sweep_all.sh         the sweeps quoted in the handoff
      ckpt_test.sh            checkpoint / resume / signal / lock / 0.20 compat /
                              faults in checkpointed runs (20 checks)
      bug014_test.sh          re-inserts 0.14's bug: chain blind, map check not
      e2e_mid.sh              r = 6972593, threads + checking + interruption
      bench.sh                timings, with/without checking, r = 136279841
    logs/                     outputs of the validation runs quoted in the docs

## Rebuilding the validation from scratch

Needs gcc (13 used; clang 18 also builds it), make, bash, python3 (two
tests), bc (bench).

    make                # build/irred, irred_generic, irred_asan, ref, irred315, irred020
    make selftest       # 7 x "selftest passed"
    make test           # cmp + sweep + ckpt + bug014, ~15 minutes on 2 cores
    make bench          # ~3 minutes
    tests/e2e_mid.sh build/irred    # ~12 minutes

Build variants: `make CC=clang`, or `CFLAGS="-O3"` (no `-march`: run-time
dispatch still picks AVX2/AVX-512). Without `-fopenmp`, `--threads` is
ignored with a note.

## Notes for a future round

* **Start here:** `docs/irred_handoff.md` §6 (open items). The biggest one
  is a GPU irred built on tsfactor's validated squaring kernels; the design
  point about the residue convention (s ≤ (r−1)/2 for the fold passes) is
  written down there.
* The tsfactor devkit (`errorcheck_handoff.md` §4) has the h-table version
  of the XOR Gerbicz check; irred's single-chain version is simpler and is
  derived in irred_handoff §3.1.
* The sandbox can only show compute behaviour (260 MB L3 holds everything).
  Ask Gord for Zen 2 numbers before tuning for memory bandwidth.
* `irred --selftest` and `make test` are the regression gates. Every change
  to the squaring code must also pass `tests/cmp.sh` and keep
  `tests/bug014_test.sh` reporting "caught by the map check".

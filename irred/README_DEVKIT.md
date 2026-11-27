# irred devkit — round 3 (irred 0.22, 2026-11-25)

Everything needed to continue work on `irred` — in the same conversation or
a new one — and for Gord to build, test and run it.

## State of the work

* **irred 0.22** (`irred/irred.c`, round 3, from Gord's round-2 test plan on
  his Zen 2):
  * checkpoints of 0.20 (format version 1) are no longer read; a 0.20
    checkpoint for the same r and s makes 0.22 stop rather than overwrite it;
  * `--nt` is automatic: streaming stores when the two squaring buffers do
    not fit in the L3 cache (Zen 2 at r = 136279841: 2.40 → 1.76 ms per
    squaring, his measurement); `--no-nt` / `--nt` override;
  * `--stop-after N` stops after N squarings as SIGTERM would — for timing
    runs without `timeout` (whose Rust version on Ubuntu 26.04 killed the
    rest of the test plan's pipeline, not irred's output).
* 0.21 (round 2) added Gerbicz error checking, the start-up map check,
  `--threads` and `--nt`. It confirmed x^136279841 + x^12338357 + 1
  irreducible on Gord's machine.
* Documents: `docs/irred_handoff.md` (the complete record, rounds 1–3:
  algorithm, the 0.14 bug, error-checking mathematics, measurements,
  verification ledger, next steps — GPU irred is the big one) and
  `docs/TESTPLAN.md` (round 3: three short steps).

## Layout

    README_DEVKIT.md
    Makefile                  make / make test / make selftest / cmp / sweep /
                              ckpt / bug014 / bench  (builds into build/)
    irred/irred.c             irred 0.22 (the deliverable; single file)
    irred_0.21_to_0.22.diff   this round's change
    docs/
      irred_handoff.md        complete record, rounds 1-3
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
                              tests/ckpt_test.sh checks that 0.22 refuses them)
      irred_0.20_repo_move.patch   Gord's patch for the repository move
    tests/
      cmp.sh                  all builds/paths vs tools/ref.c (+ 3.15 at larger r)
      ec_sweep.sh             exhaustive one-fault-per-run sweep for one site
      ec_sweep_all.sh         the sweeps quoted in the handoff
      ckpt_test.sh            checkpoint / resume / signal / lock / --stop-after /
                              0.20 files refused / faults in checkpointed runs
                              (22 checks)
      bug014_test.sh          re-inserts 0.14's bug: chain blind, map check not
      e2e_mid.sh              r = 6972593, threads + checking + interruption
      bench.sh                timings, with/without checking, r = 136279841
    logs/                     outputs of the validation runs quoted in the docs

## Rebuilding the validation from scratch

Needs gcc (13 used; clang 18 also builds it), make, bash, python3 (two
tests), bc (bench).

    make                # build/irred, irred_generic, irred_asan, ref, irred315, irred020
    make selftest       # 8 x "selftest passed"
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
* The sandbox can only show compute behaviour (260 MB L3 holds everything;
  the automatic `--nt` stays off here). Gord's Zen 2 numbers are in
  irred_handoff §4a.
* `irred --selftest` and `make test` are the regression gates. Every change
  to the squaring code must also pass `tests/cmp.sh` and keep
  `tests/bug014_test.sh` reporting "caught by the map check".

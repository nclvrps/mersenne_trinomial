#!/bin/bash
# Builds everything the tests need, CPU only: the emulated binaries of round
# 9 (cs; csprof, the phase-timing build) and of rounds 5-8 (the checkpoint
# compatibility tests and the oracles), then the oracle files.  FULL=0 skips
# the two big oracles (see fixtures/gen_oracles.sh).
ROOT=$(cd "$(dirname "$0")" && pwd); B=$ROOT/build
T=$ROOT/tests/build_emul.sh
bash $T $ROOT/src/r9 $B/r9 cs || exit 1
EXTRA=-DCOARSE_PHASE_TIMING bash $T $ROOT/src/r9 $B/r9 csprof || exit 1
bash $T $ROOT/src/r8 $B/r8 cs || exit 1
bash $T $ROOT/src/r7 $B/r7 cs || exit 1
bash $T $ROOT/src/r6 $B/r6 cs || exit 1
EXTRA="-DWIDE_FIELD -Wno-unused-parameter -Wno-unused-function" bash $T $ROOT/src/r5 $B/r5 cs_wide || exit 1
EXTRA="-Wno-unused-parameter -Wno-unused-function" bash $T $ROOT/src/r5 $B/r5 cs_narrow || exit 1
bash $ROOT/fixtures/gen_oracles.sh

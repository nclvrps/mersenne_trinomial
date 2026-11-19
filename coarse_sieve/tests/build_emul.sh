#!/bin/bash
# The emulated (CPU-only) build of a coarse_sieve source tree, as the
# Makefile's emul-coarse_sieve does it, with -Wall -Wextra:
#   build_emul.sh <source dir> <output dir> <binary name>   (EXTRA: more flags)
# Prints BUILD-OK <binary> on success; any compiler warning is shown.
SRC=$1; OUT=$2; NAME=$3
[ -n "$SRC" ] && [ -n "$OUT" ] && [ -n "$NAME" ] || { echo "usage: $0 <src> <out> <name>"; exit 1; }
mkdir -p "$OUT" || exit 1
cp "$SRC"/cuda_emul.h "$OUT"/ || exit 1
for h in "$SRC"/gf2_*.h; do cp "$h" "$OUT"/; done
printf '#pragma once\n#include "cuda_emul.h"\n' > "$OUT"/cuda_runtime.h
sed -E 's/([A-Za-z_][A-Za-z0-9_]*)<<<([^>]*)>>>/EMUL_LAUNCH(\1, \2)/g; s/__device__ __forceinline__/static inline/' \
    "$SRC"/coarse_sieve.cu > "$OUT"/coarse_sieve.cpp || exit 1
rm -f "$OUT/$NAME"
( cd "$OUT" && g++ -O2 -march=native -Wall -Wextra -std=c++17 -x c++ -I. -pthread $EXTRA \
    -o "$NAME" coarse_sieve.cpp 2>&1 | grep -E "error|warning" | grep -v "multi-line" | head -30 )
[ -x "$OUT/$NAME" ] && echo "BUILD-OK $OUT/$NAME"

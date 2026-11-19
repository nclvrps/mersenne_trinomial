#!/bin/bash
# The nvcc build, for its diagnostics (no GPU needed): warnings, and per
# kernel the registers, stack and spills ptxas reports -- the batched
# kernels must stay at <= 128 registers (512-thread blocks, one per SM) and,
# but for the grouped tower and cubic ones' ~190 bytes (and RR's at d=45,
# 51, which carry gg), spill-free.
#   NVCC=/path/to/nvcc ARCH=sm_75 NVLIB=-L/path/to/cuda/lib \
#       nvcc/build_check.sh [extra nvcc flags, e.g. -DCOARSE_PHASE_TIMING]
# (A workspace without CUDA: see the README for nvcc from pip.)
ROOT=$(cd "$(dirname "$0")/.." && pwd); S=$ROOT/src/r10; O=$ROOT/build/nvcc
NVCC=${NVCC:-nvcc}; ARCH=${ARCH:-sm_75}
mkdir -p $O && cd $O || exit 1
cp $S/coarse_sieve.cu $S/gf2_dlog.h $S/gf2_wide_field.h $S/gf2_wide_setup.h .
( time $NVCC -O3 -arch=$ARCH -std=c++17 -Xcompiler -march=native -Xptxas -v "$@" -I. \
    $NVLIB -o coarse_sieve coarse_sieve.cu -Xcompiler -pthread ) > build.full 2>&1
echo "nvcc exit $?"; tail -3 build.full
grep -E "warning|error" build.full | grep -v "^ptxas info" | head
python3 - build.full <<'PYEOF'
import re, sys
t = open(sys.argv[1]).read()
ents = re.findall(r"Compiling entry function '(\S+)' for '\S+'\nptxas info\s*: Function properties for \S+\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers", t)
print(len(ents), "kernels")
meth = {'1': 'tower', '2': 'PH', '3': 'RR', '4': 'cubic'}
for name, stack, ss, sl, regs in ents:
    m = re.match(r'_Z12k_logs_batchILi(\d+)ELi(\d)E(?:Li(\d)E)?', name)
    if m and (int(m.group(1)) >= 40 or ss != '0' or sl != '0'):
        print("k_logs_batch d=%s %-5s%s: %s registers, stack %s B, spills %s/%s B" % (
            m.group(1), meth[m.group(2)], " grouped" if m.group(3) == '1' else "", regs, stack, ss, sl))
other = [e for e in ents if not e[0].startswith('_Z12k_logs_batch') and (e[2] != '0' or e[3] != '0')]
for e in other: print("spills in", e[0], e[2], e[3])
PYEOF

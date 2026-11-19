#!/usr/bin/env python3
# Where a kernel's local-memory accesses sit, from its SASS:
#
#   cuobjdump -sass -fun <mangled kernel name> build/nvcc/coarse_sieve > k.sass
#   python3 tools/sass_locals.py k.sass
#
# (the names: cuobjdump -sass build/nvcc/coarse_sieve | grep Function).
# Prints every loop (a backward BRA) of at least MIN instructions (argument 2,
# default 300) with the local loads and stores inside it, then each scalar
# access at a fixed offset from R1, the stack pointer, with the size of the
# innermost loop around it: those in small inner loops run most often.  Such
# accesses are mostly spill slots (the batch arrays move as .128 or through a
# base register), but an array element at a constant index looks the same;
# ptxas -v's spill counts say how many bytes are spills.
# Round 8 used this on the grouped kernels (handoff section 3): in the copy
# of the batch code that has SHFL.IDX (the pooled fill's deal), the deal loop
# reloads spilled values at every enumeration step.
import re, sys

path = sys.argv[1]
MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ins = []
for line in open(path):
    m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\*', line)
    if m:
        ins.append((int(m.group(1), 16), m.group(2).strip()))
if not ins:
    sys.exit("no SASS instructions found in " + path)
loops = []
for a, t in ins:
    m = re.search(r'\bBRA\b.*?(0x[0-9a-f]+)', t)
    if m and int(m.group(1), 16) < a:
        loops.append((int(m.group(1), 16), a))
loops.sort()
loc = [(a, t) for a, t in ins if re.search(r'\b(LDL|STL)', t)]
print("%d instructions, %d local accesses (%d stores, %d loads); SHFL.IDX at %s"
      % (len(ins), len(loc), sum('STL' in t for a, t in loc), sum('LDL' in t for a, t in loc),
         ", ".join(hex(a) for a, t in ins if 'SHFL.IDX' in t) or "none"))
print("loops of >= %d instructions:" % MIN)
for t0, a0 in loops:
    if (a0 - t0) // 16 >= MIN:
        n = sum(t0 <= a <= a0 for a, t in loc)
        print("  %8s .. %-8s %6d instructions, %3d local accesses" % (hex(t0), hex(a0), (a0 - t0) // 16, n))
print("scalar accesses at R1 + offset (mostly spill slots), with the innermost loop around each:")
for a, t in loc:
    if not re.search(r'\[R1(\+0x[0-9a-f]+)?\]', t) or '.128' in t:
        continue
    around = [(t0, a0) for t0, a0 in loops if t0 <= a <= a0]
    if around:
        t0, a0 = min(around, key=lambda x: x[1] - x[0])
        where = "loop %s..%s (%d instructions)" % (hex(t0), hex(a0), (a0 - t0) // 16)
    else:
        where = "no loop"
    print("  %8s  %-34s %s" % (hex(a), t, where))

#!/usr/bin/env python3
# passavg.py -- pass averages of the fill simulation (coarse_sieve round 7)
#
#   ./cstat 44 0 512 2 > cstat44.txt          # classes of the first 512 positions
#   python3 passavg.py 44 cstat44.txt 0,1,2 16
#
# Runs ./fillsim on every non-empty chunk listed in a cstat output (the first
# positions of the pass are a stratified sample of it, by the design of the
# pass order), with each of the given fills and `warps` warps per chunk, and
# sums the counts: lane-steps per class credited, the share of batch slots
# holding a class, and batch calls (lane-batches) per 1000 classes credited.
import re
import subprocess
import sys

d = int(sys.argv[1])
statf = sys.argv[2]
fills = [int(x) for x in sys.argv[3].split(',')]
warps = sys.argv[4] if len(sys.argv) > 4 else '16'
chunks = []
for line in open(statf):
    if line.startswith('pos'):
        f = line.split()
        if int(f[7]) > 0:
            chunks.append(int(f[3]))
print("d=%d: %d non-empty chunks in the sample" % (d, len(chunks)))
for fill in fills:
    it = proc = cred = batch = 0.0
    for c in chunks:
        out = subprocess.run(['./fillsim', str(d), str(c), str(fill), warps],
                             capture_output=True, text=True).stdout
        l1 = re.search(r'processed (\d+), credited (\d+)', out)
        l2 = re.search(r'lane-steps per class ([0-9.]+); slots used [0-9.]+%; batch calls (\d+)', out)
        p, cr = int(l1.group(1)), int(l1.group(2))
        ls, b = float(l2.group(1)), int(l2.group(2))
        it += ls * cr / 32.0          # fill iterations
        proc += p; cred += cr; batch += b
    if not cred or not batch:
        print("fill %d: no classes in the simulated warps (more warps?)" % fill)
        continue
    print("fill %d: lane-steps per class %.3f, batch slots used %.1f%%, "
          "lane-batches per 1000 classes %.3f" %
          (fill, 32.0 * it / cred, 100.0 * proc / (batch * 32 * 32),
           1000.0 * batch * 32 / cred))

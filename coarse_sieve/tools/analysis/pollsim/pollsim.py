#!/usr/bin/env python3
"""Round 10's polling simulation.  lyn.c counts the canonical Lyndon words per
chunk at a scaled-down degree (n=33, chunks of 2^13: 2^19 chunks, like d=51):
    gcc -O2 -o lyn lyn.c && ./lyn 33 13 > cnt.bin && python3 pollsim.py cnt.bin
Each chunk's time is taken proportional to its classes (mean 41 ms, as at
d=55) plus a fixed cost for an empty chunk; the pass visits the chunks in
coarse_sieve's order (chunk_at).  It prints the time lost to waking up late:
round 9's 1 ms polling per chunk, round 10's one wait per round of 8 queued
chunks, and the discarded predictive sleep."""
import numpy as np, sys
cnt = np.fromfile(sys.argv[1], dtype=np.uint64).astype(np.float64)
nch = len(cnt); CB = 8; M64 = (1 << 64) - 1
def chunk_at(p, nchunks):          # coarse_sieve.cu's pass order
    if nchunks <= CB or nchunks % CB: return p
    nb = nchunks // CB; k = 0
    while (1 << k) < nb: k += 1
    b = p // CB; x = 0
    for i in range(k): x |= ((b >> i) & 1) << (k - 1 - i)
    y = x
    for i in range(k - 1, -1, -1):
        h = (((x >> (i + 1)) + 1) * 0x9E3779B97F4A7C15 + i * 0xD1B54A32D192ED03) & M64
        h ^= h >> 31; h = (h * 0xBF58476D1CE4E5B9) & M64; h ^= h >> 29
        y ^= (h >> 63) << i
    return y * CB + p % CB
c = cnt[np.array([chunk_at(p, nch) for p in range(nch)])]
print("chunks %d; %.1f%% hold no class" % (nch, 100 * np.mean(c == 0)))
for empty_ms in (0.05, 0.2, 0.5, 1.0):
    base = empty_ms * 1e-3; dur = base + c / c.mean() * (0.041 - base); T = dur.sum(); wo = 0.06e-3
    def lost(d, iv):
        k = np.ceil(d / (iv + wo)); return 100 * (k * (iv + wo) - d).sum() / T, k.sum() / T
    l9, p9 = lost(dur, 1e-3)
    l10, p10 = lost(dur.reshape(-1, CB).sum(axis=1), 1e-3)
    print("empty chunk %.2f ms: round 9 (1 ms polls per chunk) %.2f%% lost, %.0f polls/s;"
          " round 10 (one wait per 8 chunks) %.2f%%, %.0f polls/s" % (empty_ms, l9, p9, l10, p10))

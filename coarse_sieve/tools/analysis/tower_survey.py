#!/usr/bin/env python3
"""Table sizes of the log engines per degree (8-byte entries), for planning
degrees beyond 55 (handoff of round 10, section 7).  The general tower: m | d,
K = GF(2^(d/m)), k = ceil((m-1)/2); x = r/t with r, t of degree <= k in w
over K; a table of (2^n + 1)^k entries (m = 3: the cubic tower)."""
import sys
def I(n):
    def mob(n):
        r, p, m = 1, 2, n
        while p * p <= m:
            if m % p == 0:
                m //= p
                if m % p == 0: return 0
                r = -r
            p += 1
        return -r if m > 1 else r
    return sum(mob(n // e) * 2**e for e in range(1, n + 1) if n % e == 0) // n
def fb(b):
    for u, s in ((1 << 40, 'TB'), (1 << 30, 'GB'), (1 << 20, 'MB'), (1 << 10, 'KB')):
        if b >= u: return ("%.0f %s" if b / u >= 10 else "%.1f %s") % (b / u, s)
    return "%d B" % b
NS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5      # ns per class
for d in range(40, 65):
    o = ["RR %s" % fb(8 << (d // 2 + 1))]
    if d % 2 == 0: o.append("quadratic/GF(2^%d) %s" % (d // 2, fb(8 << (d // 2))))
    for m in range(3, d + 1):
        if d % m: continue
        n = d // m; k = m // 2
        ent = (2**n + 1) ** k
        if ent * 8 <= (64 << 30): o.append("deg-%d/GF(2^%d) %s" % (m, n, fb(ent * 8)))
    print("d=%d  I(d)=%.3g  %.1f days at %.2f ns/class:  %s" % (d, I(d), I(d) * NS * 1e-9 / 86400, NS, "; ".join(o)))

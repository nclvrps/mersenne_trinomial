import sys, collections
def load(fn):
    D = collections.defaultdict(list)
    for line in open(fn):
        r, s, d = map(int, line.split())
        D[r].append((s, d))
    return D
def S(lst, k):            # fraction with no factor of degree <= k
    return sum(1 for s, d in lst if d == 0 or d > k) / len(lst)
ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
def table(D, label):
    print(label)
    print("   r  r%8  N      " + " ".join(f"{k:>6}" for k in ks) + "   irr")
    for r in sorted(D):
        lst = D[r]
        row = []
        for k in ks:
            row.append(f"{k*S(lst,k):6.3f}" if k <= r // 3 else "     -")
        nirr = sum(1 for s, d in lst if d == 0)
        print(f"{r:6d} {r%8:3d} {len(lst):5d}  " + " ".join(row) + f"  {nirr:4d}")
def pooled(D, cls, label):
    lst = [x for r in D if r % 8 in cls for x in D[r]]
    rs = [r for r in D if r % 8 in cls]
    if not lst: return
    row = []
    for k in ks:
        if k <= min(rs) // 3:
            row.append(f"{k*S(lst,k):6.3f}")
        else:
            row.append("     -")
    print(f"{label:28s} N={len(lst):6d}  " + " ".join(row))
M = load(sys.argv[1]); N = load(sys.argv[2]) if len(sys.argv) > 2 else {}
table(M, "Mersenne exponents: k*S(k)")
if N: table(N, "other primes: k*S(k)")
print()
print("pooled k*S(k) at k =        " + " ".join(f"{k:>6}" for k in ks))
pooled(M, {1, 7}, "Mersenne, r = +-1 mod 8")
pooled(N, {1, 7}, "other primes, r = +-1 mod 8")
pooled(M, {3, 5}, "Mersenne, r = +-3 mod 8")
pooled(N, {3, 5}, "other primes, r = +-3 mod 8")

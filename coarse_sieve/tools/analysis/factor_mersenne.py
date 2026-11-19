#!/usr/bin/env python3
"""Factor 2^d - 1 for the degrees given (default 40..64): PH needs the largest
prime factor small; the others decide the irregular classes."""
import math, random, sys
def is_prime(n):
    if n < 2: return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True
def rho(n):
    if n % 2 == 0: return 2
    while True:
        c = random.randrange(1, n); x = y = random.randrange(2, n); g = 1
        while g == 1:
            x = (x * x + c) % n; y = (y * y + c) % n; y = (y * y + c) % n; g = math.gcd(abs(x - y), n)
        if g != n: return g
def factor(n, out):
    if n == 1: return
    if is_prime(n): out.append(n); return
    for p in range(2, 1000):
        if n % p == 0: out.append(p); factor(n // p, out); return
    g = rho(n); factor(g, out); factor(n // g, out)
ds = [int(a) for a in sys.argv[1:]] or list(range(40, 65))
for d in ds:
    f = []; factor(2**d - 1, f); f.sort()
    print("%d  largest %d  = %s" % (d, max(f), " x ".join(map(str, f))))

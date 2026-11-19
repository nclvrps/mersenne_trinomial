#!/usr/bin/env python3
"""The coarse sieve's counts against the Poisson heuristic (round 10).

A survivor of degrees < d gains a degree-d factor with probability
1 - exp(-1/d).  This prints that comparison for the r = 136279841 runs, the
d=55 prediction, the d=52/d=53 overlap check, and the heuristic number of
irreducible (= primitive, r a Mersenne exponent) trinomials x^r + x^s + 1
with s <= r/2: survivors * (1/r) / P(random poly has no factor of degree
<= D), doubled by Swan's theorem's parity for r = +-1 (mod 8).
Edit SURV (survivors after each degree) for another exponent.
"""
import math
R = 136279841
SURV = {39: 3070345, 40: 2994673, 41: 2922436, 42: 2853253, 43: 2787930, 44: 2725459,
        45: 2665526, 46: 2607803, 47: 2553216, 51: 2354612, 52: 2310017, 53: 2266973, 54: 2225592}

def mobius(n):
    r, p, m = 1, 2, n
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0: return 0
            r = -r
        p += 1
    return -r if m > 1 else r

def I(n):  # irreducible polynomials of degree n over GF(2)
    return sum(mobius(n // e) * 2**e for e in range(1, n + 1) if n % e == 0) // n

smax = R // 2
print(" d  survivors before   found   share     1-e^(-1/d)  ratio    z")
chi2 = n = 0
for d in sorted(SURV):
    if d - 1 not in SURV: continue
    before = SURV[d - 1]; f = before - SURV[d]; p = 1 - math.exp(-1 / d)
    z = (f - before * p) / math.sqrt(before * p * (1 - p)); chi2 += z * z; n += 1
    print("%2d %12d %10d  %.4f%%   %.4f%%   %.3f  %+.2f" % (d, before, f, 100 * f / before, 100 * p, f / (before * p), z))
print("chi^2 = %.2f on %d degrees" % (chi2, n))
top = max(SURV); p = 1 - math.exp(-1 / (top + 1)); e = SURV[top] * p
print("d=%d predicted: %.0f +- %.0f factors, %.0f survivors" % (top + 1, e, math.sqrt(e * (1 - p)), SURV[top] - e))
if 51 in SURV and 52 in SURV:
    f52 = SURV[51] - SURV[52]; p53 = 1 - math.exp(-1 / 53)
    print("s with factors of degree 52 and 53: %.1f +- %.1f expected (842 observed)" % (f52 * p53, math.sqrt(f52 * p53)))
P = math.exp(sum(I(k) * math.log1p(-2.0 ** -k) for k in range(1, top + 1)))
e1 = SURV[top] / R / P
print("random polynomial: P(no factor of degree <= %d) = %.4f%%; these trinomials: %.4f%% (x%.2f)" % (top, 100 * P, 100 * SURV[top] / smax, SURV[top] / smax / P))
par = 2 if R % 8 in (1, 7) else 1
print("expected irreducible trinomials (s <= r/2): %.2f%s; Poisson P(none) = %.3f" % (par * e1, " (with Swan's parity doubling)" if par == 2 else "", math.exp(-par * e1)))

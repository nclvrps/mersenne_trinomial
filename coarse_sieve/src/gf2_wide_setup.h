// gf2_wide_setup.h
// Wide-field companion to gf2_field_setup.h: same host-side setup
// (trial-division factorization, irreducibility / primitivity tests,
// primitive-polynomial search) but with u64 field elements, for the
// -DWIDE_FIELD build.  Requires gf2_wide_field.h to be included first.
#ifndef GF2_WIDE_SETUP_H
#define GF2_WIDE_SETUP_H
#include "gf2_wide_field.h"
#include <vector>
#include <cstdio>
#include <cstdlib>

// factor m by trial division (m <= 2^63 fully factored by primes <= 2^32)
static inline std::vector<u64> factor_u64(u64 m) {
    std::vector<u64> ps;
    for (u64 p = 2; p <= 4294967296ULL && p * p <= m; p += (p == 2 ? 1 : 2)) {
        if (m % p == 0) { ps.push_back(p); while (m % p == 0) m /= p; }
    }
    if (m > 1) ps.push_back(m);
    return ps;
}

// irreducibility of q (degree d): x^(2^d) == x mod q, and
// x^(2^(d/p)) != x for each prime p | d
static inline bool is_irreducible(u64 q, int d) {
    std::vector<u64> chain(d + 1);
    u64 t = 2; // x
    chain[0] = t;
    for (int i = 1; i <= d; i++) { t = gf_sqr(t, q, d); chain[i] = t; }
    if (chain[d] != 2) return false;
    for (u64 p : factor_u64((u64)d))
        if (chain[d / p] == 2) return false;
    return true;
}

// is x primitive mod q? (order of x == M)
static inline bool x_is_primitive(u64 q, int d, u64 M) {
    for (u64 p : factor_u64(M))
        if (gf_pow(2, M / p, q, d) == 1) return false;
    return true;
}

// smallest primitive polynomial of degree d (so that g = x is a generator)
static inline u64 find_primitive_poly(int d) {
    u64 M = ((u64)1 << d) - 1;
    u64 lo = ((u64)1 << d) | 1u;
    u64 hi = (u64)2 << d;                 // = 2^(d+1)  (d <= 62 here)
    for (u64 q = lo; q < hi; q += 2)
        if (is_irreducible(q, d) && x_is_primitive(q, d, M))
            return q;
    fprintf(stderr, "no primitive polynomial of degree %d?!\n", d);
    exit(1);
}
#endif

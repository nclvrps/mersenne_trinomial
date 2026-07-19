// gf2_gcd_ntl.cpp
// NTL backend for gf2_gcd.h: subquadratic GCD (HalfGCD) and CanZass
// factorization for the equal-degree tie-break.  Compile with
// -DHAVE_NTL and link -lntl -lgmp; for production-scale r, build NTL
// with the gf2x multiplication backend (NTL_GF2X_LIB=on), exactly as
// for factor.cpp.
//
// Word <-> GF2X conversion relies on: x86-64 little-endian u64 words
// viewed as bytes give monotone bit order, and NTL's GF2XFromBytes /
// BytesFromGF2X use little-endian bytes with bit i of byte j as
// coefficient 8j + i.  The naive-vs-NTL cross-check in
// `trinomial_scan --selftest` validates this empirically.

#ifdef HAVE_NTL
#include "gf2_gcd.h"
#include <NTL/GF2X.h>
#include <NTL/GF2XFactoring.h>

using namespace NTL;

static void to_gf2x(GF2X &f, const u64 *w, size_t nw) {
    GF2XFromBytes(f, (const unsigned char *)w, (long)(nw * 8));
}

static void from_gf2x(std::vector<u64> &out, const GF2X &f) {
    long nb = NumBytes(f);
    if (nb <= 0) { out.assign(1, 0); return; }
    out.assign(((size_t)nb + 7) / 8 + 1, 0);
    BytesFromGF2X((unsigned char *)out.data(), f, nb);
}

u64 poly_gcd_ntl(const u64 *a, size_t aw, const u64 *b, size_t bw,
                 std::vector<u64> *gout, u64 keep_max_bits) {
    GF2X A, B, G;
    to_gf2x(A, a, aw);
    to_gf2x(B, b, bw);
    GCD(G, A, B);
    long d = deg(G);
    if (d <= 0) return 0;                      // gcd is 0 or 1: trivial
    if (gout && (u64)d + 1 <= keep_max_bits) from_gf2x(*gout, G);
    return (u64)d;
}

bool edf_least_ntl(const u64 *g, size_t gw, u64 target,
                   std::vector<u64> &out) {
    GF2X G;
    to_gf2x(G, g, gw);
    if (deg(G) <= 0) return false;
    vec_pair_GF2X_long fac;
    CanZass(fac, G);                           // g is squarefree (T is)
    bool have = false;
    std::vector<u64> best;
    for (long i = 0; i < fac.length(); i++) {
        if ((u64)deg(fac[i].a) != target) continue;
        std::vector<u64> w;
        from_gf2x(w, fac[i].a);
        if (!have || poly_less(w, best)) { best = w; have = true; }
    }
    if (have) out = best;
    return have;
}
#endif // HAVE_NTL

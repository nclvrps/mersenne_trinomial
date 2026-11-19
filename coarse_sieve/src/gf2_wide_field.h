// gf2_wide_field.h
// Wide-field companion to gf2_small_field.h: arithmetic in GF(2^d) for
// d <= 63, field elements packed as u64 bitmasks (bit i = coefficient of
// x^i, so an element has degree <= d-1 <= 62 and fits in u64).  The
// modulus q has degree d (top term at bit d <= 63) and also fits in u64.
//
// Same public API and identical results as gf2_small_field.h on the
// overlapping range d <= 31 (verified by cross-check), but every element
// / index / log value is u64 here, and the carry-less product of two
// elements can reach degree 2d-2 <= 124, so it is carried in a 128-bit
// {hi,lo} pair.  No __int128 or PCLMUL is used, so this compiles and runs
// identically on host and CUDA device.
//
// Selected by coarse_sieve.cu when built with -DWIDE_FIELD; the default
// (narrow) build keeps gf2_small_field.h for its faster u32 path (d<=32).

#ifndef GF2_WIDE_FIELD_H
#define GF2_WIDE_FIELD_H

#include <stdint.h>

#if defined(__CUDACC__)
#define GF2_HD __host__ __device__ __forceinline__
#else
#define GF2_HD static inline
#endif

typedef uint32_t u32;
typedef uint64_t u64;

// 32x32 -> 64 carry-less multiply (portable shift/xor); building block for
// the 64-bit product below.  (Kept local so this header is self-contained
// and doesn't depend on gf2_small_field.h being included first.)
GF2_HD u64 gf2w_clmul32(u32 a, u32 b) {
    u64 r = 0, aa = a;
    while (b) {
        r ^= aa * (u64)(b & 1);
        aa <<= 1;
        b >>= 1;
    }
    return r;
}

// Carry-less product of two 64-bit polynomials -> 128-bit result {hi,lo}.
// Split each operand into 32-bit halves and combine four 32x32 products;
// because the products are carry-less, the cross term folds in by XOR with
// no carries.
//   a*b = ll  ^ (m << 32) ^ (hh << 64),   m = al*bh ^ ah*bl
GF2_HD void gf2w_clmul64(u64 a, u64 b, u64 *hi, u64 *lo) {
    u32 al = (u32)a, ah = (u32)(a >> 32);
    u32 bl = (u32)b, bh = (u32)(b >> 32);
    u64 ll = gf2w_clmul32(al, bl);
    u64 hh = gf2w_clmul32(ah, bh);
    u64 m  = gf2w_clmul32(al, bh) ^ gf2w_clmul32(ah, bl);
    *lo = ll ^ (m << 32);
    *hi = hh ^ (m >> 32);
}

// Reduce a 128-bit product {phi,plo} of degree <= 2d-2 (<= 124) modulo q
// (degree d, bit d set).  Bit-by-bit from the top down to bit d; each set
// high bit b clears by XOR-ing q shifted left by (b-d).  Result has degree
// < d <= 63 and so fits in u64.  Branch-free inner body via a 0/-1 mask.
GF2_HD u64 gf2_reduce_wide(u64 phi, u64 plo, u64 q, int d) {
    for (int b = 2 * d - 2; b >= d; b--) {
        u64 have = (b >= 64) ? ((phi >> (b - 64)) & 1u)
                             : ((plo >> b) & 1u);
        u64 msk = 0ULL - have;             // all-ones iff bit b set
        int sh = b - d;                    // 0 .. d-2 (<= 61)
        if (sh == 0) {
            plo ^= q & msk;
        } else {
            plo ^= (q << sh) & msk;
            phi ^= (q >> (64 - sh)) & msk; // sh in [1,63] so 64-sh in [1,63]
        }
    }
    return plo;
}

// Field multiply in GF(2)[x]/q
GF2_HD u64 gf_mul(u64 a, u64 b, u64 q, int d) {
    u64 hi, lo;
    gf2w_clmul64(a, b, &hi, &lo);
    return gf2_reduce_wide(hi, lo, q, d);
}

GF2_HD u64 gf_sqr(u64 a, u64 q, int d) { return gf_mul(a, a, q, d); }

// Multiply by x (antilog chain when the generator is x).  a has degree
// <= d-1 <= 62, so a << 1 has degree <= d <= 63 and fits in u64; the
// single possible overflow bit (bit d) is folded by q.
GF2_HD u64 gf_mulx(u64 a, u64 q, int d) {
    u64 t = a << 1;
    u64 have = (t >> d) & 1u;
    return t ^ q * have;
}

// x^e mod q by left-to-right binary exponentiation (element base)
GF2_HD u64 gf_pow(u64 base, u64 e, u64 q, int d) {
    u64 result = 1;
    for (int b = 63; b >= 0; b--) {
        result = gf_sqr(result, q, d);
        if ((e >> b) & 1) result = gf_mul(result, base, q, d);
    }
    return result;
}

// Rotate-left of a d-bit value: j*2 mod (2^d - 1) for 0 < j < 2^d - 1.
// Now a u64 rotate within d bits (d <= 63): the "<< 1" can reach bit d,
// which & M drops, and the wrapped bit re-enters at position 0.
GF2_HD u64 rotl_d(u64 j, int d, u64 M) {
    return ((j << 1) | (j >> (d - 1))) & M;
}

GF2_HD u64 gcd_u64(u64 a, u64 b) {
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

// (a * b) mod m without a 128-bit type, for m <= 2^63-1 (d <= 63).  At
// d >= 33 the index products j*rmod and inv*(v/gg) exceed 64 bits, so the
// plain "(a*b) % m" used in the narrow build would overflow; this
// double-and-add stays within u64 because a,b < m < 2^63 keeps every
// partial sum below 2^64.
GF2_HD u64 mulmod_u64(u64 a, u64 b, u64 m) {
    a %= m;
    u64 r = 0;
    while (b) {
        if (b & 1) { r += a; if (r >= m) r -= m; }
        a <<= 1; if (a >= m) a -= m;
        b >>= 1;
    }
    return r;
}

// Modular inverse of a mod m (gcd(a,m)=1), extended Euclid.  The
// intermediate product qq*newt stays bounded by 2m (it equals the
// difference of two coefficients each bounded by m), so int64 is safe for
// m < 2^62; d <= 33 is far inside that.
GF2_HD u64 inv_mod(u64 a, u64 m) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m, newr = (int64_t)(a % m);
    while (newr != 0) {
        int64_t qq = r / newr;
        int64_t tmp = t - qq * newt; t = newt; newt = tmp;
        tmp = r - qq * newr; r = newr; newr = tmp;
    }
    if (t < 0) t += (int64_t)m;
    return (u64)t;
}

// Minimal polynomial mask of beta = g^j given the antilog table A
// (A[i] = g^i).  Coefficients live in GF(2); returns the (d+1)-bit mask,
// or 0 on internal inconsistency.  P is caller scratch of >= d+1 u64s.
GF2_HD u64 minpoly_mask(u64 j, int d, u64 M, u64 q, const u64 *A, u64 *P) {
    for (int k = 0; k <= d; k++) P[k] = 0;
    P[0] = 1;
    u64 rt = j;
    for (int i = 0; i < d; i++) {
        u64 root = A[rt];
        for (int k = i + 1; k >= 1; k--)
            P[k] = P[k - 1] ^ gf_mul(root, P[k], q, d);
        P[0] = gf_mul(root, P[0], q, d);
        rt = rotl_d(rt, d, M);
    }
    u64 mask = 0;
    for (int k = 0; k <= d; k++) {
        if (P[k] > 1) return 0;
        mask |= (u64)P[k] << k;
    }
    return mask;
}

#endif // GF2_WIDE_FIELD_H

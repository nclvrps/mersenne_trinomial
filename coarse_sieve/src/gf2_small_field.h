// gf2_small_field.h
// Arithmetic in GF(2^d) for d <= 32, elements packed as u32 bitmasks
// (bit i = coefficient of x^i, so a field element has degree <= d-1 <= 31
// and fits in u32).  The modulus q has degree d and its top term is bit d;
// at d = 32 that bit is 32, so q itself is carried as a u64.  (For d <= 31
// q also fits in u32, and every result below is bit-identical to the
// earlier u32-modulus version -- only the parameter width changed.)
//
// This header is compiled both by the CPU reference (where it is
// validated against independent oracles and against verified factor.cpp
// output) and by the CUDA kernels, so the exact arithmetic that runs on
// the GPU is the arithmetic that was tested on the CPU.

#ifndef GF2_SMALL_FIELD_H
#define GF2_SMALL_FIELD_H

#include <stdint.h>

#if defined(__CUDACC__)
#define GF2_HD __host__ __device__ __forceinline__
#else
#define GF2_HD static inline
#endif

typedef uint32_t u32;
typedef uint64_t u64;

// Carry-less multiply of two <=32-bit polynomials, 64-bit result.
// Portable shift/xor version; branch-free inner step.
GF2_HD u64 gf2_clmul32_portable(u32 a, u32 b) {
    u64 r = 0, aa = a;
    while (b) {
        r ^= aa * (u64)(b & 1);   // (b&1) in {0,1}: multiply keeps it branch-free
        aa <<= 1;
        b >>= 1;
    }
    return r;
}

#if !defined(__CUDA_ARCH__) && (defined(__PCLMUL__) || defined(__x86_64__))
#include <wmmintrin.h>
static inline u64 gf2_clmul32_x86(u32 a, u32 b) {
    __m128i va = _mm_cvtsi64_si128((u64)a);
    __m128i vb = _mm_cvtsi64_si128((u64)b);
    return (u64)_mm_cvtsi128_si64(_mm_clmulepi64_si128(va, vb, 0x00));
}
#define GF2_CLMUL32(a, b) gf2_clmul32_x86((a), (b))
#else
#define GF2_CLMUL32(a, b) gf2_clmul32_portable((a), (b))
#endif

// Reduce a product of degree <= 2d-2 modulo q (degree d, bit d set).
// q is u64 so the degree-d top term (bit d, = bit 32 at d = 32) is
// representable; the shifted modulus q << (b-d) reaches bit 2d-2 <= 62,
// which still fits in u64.  The reduced result has degree < d <= 32 and
// so fits back into u32.
GF2_HD u32 gf2_reduce(u64 t, u64 q, int d) {
    for (int b = 2 * d - 2; b >= d; b--) {
        // branch-free: xor (q << (b-d)) iff bit b of t is set
        u64 have = (t >> b) & 1u;
        t ^= (q << (b - d)) * have;
    }
    return (u32)t;
}

// Field multiply in GF(2)[x]/q
GF2_HD u32 gf_mul(u32 a, u32 b, u64 q, int d) {
    return gf2_reduce(GF2_CLMUL32(a, b), q, d);
}

GF2_HD u32 gf_sqr(u32 a, u64 q, int d) { return gf_mul(a, a, q, d); }

// Multiply by x (used for the antilog chain when the generator is x)
GF2_HD u32 gf_mulx(u32 a, u64 q, int d) {
    u64 t = (u64)a << 1;
    u64 have = (t >> d) & 1u;
    return (u32)(t ^ q * have);
}

// x^e mod q by left-to-right binary exponentiation (element base)
GF2_HD u32 gf_pow(u32 base, u64 e, u64 q, int d) {
    u32 result = 1;
    for (int b = 63; b >= 0; b--) {
        result = gf_sqr(result, q, d);
        if ((e >> b) & 1) result = gf_mul(result, base, q, d);
    }
    return result;
}

// Rotate-left of a d-bit value: j*2 mod (2^d - 1) for 0 < j < 2^d - 1.
GF2_HD u32 rotl_d(u32 j, int d, u32 M) {
    return ((j << 1) | (j >> (d - 1))) & M;
}

GF2_HD u64 gcd_u64(u64 a, u64 b) {
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

// Modular inverse of a mod m (gcd(a,m)=1), extended Euclid.
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

// Minimal polynomial mask of beta = g^j in GF(2^d) given the antilog
// table A (A[i] = g^i).  Returns the (d+1)-bit mask, or 0 on internal
// inconsistency (coefficients not landing in GF(2) => caller bug).
// P is caller-provided scratch of at least d+1 u32s.
GF2_HD u64 minpoly_mask(u32 j, int d, u32 M, u64 q, const u32 *A, u32 *P) {
    for (int k = 0; k <= d; k++) P[k] = 0;
    P[0] = 1;
    u32 rt = j;
    for (int i = 0; i < d; i++) {
        u32 root = A[rt];
        // multiply P (degree i) by (x + root)
        for (int k = i + 1; k >= 1; k--)
            P[k] = P[k - 1] ^ gf_mul(root, P[k], q, d);
        P[0] = gf_mul(root, P[0], q, d);
        rt = rotl_d(rt, d, M);
    }
    u64 mask = 0;
    for (int k = 0; k <= d; k++) {
        if (P[k] > 1) return 0;           // coefficients must be in GF(2)
        mask |= (u64)P[k] << k;
    }
    return mask;
}

#endif // GF2_SMALL_FIELD_H

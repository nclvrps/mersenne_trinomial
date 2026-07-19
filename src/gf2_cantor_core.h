// gf2_cantor_core.h
// Core arithmetic for GF(2)[x] multiplication via an additive FFT over
// GF(2^64) on a Cantor basis.  Everything in this header is compiled by
// BOTH the CPU reference (where it is validated) and the CUDA kernels,
// so the arithmetic on the GPU is bit-identical to what was tested.
//
// Field: GF(2^64) = GF(2)[x] / (x^64 + x^4 + x^3 + x + 1)
//
// Multiplication strategy on devices without a carry-less multiply
// instruction: 32x32 clmul via sixteen masked integer multiplies (bits
// of each masked operand are 4 apart, so column sums <= 8 < 16 never
// carry into the next used position), assembled into 64x64 by Karatsuba.

#ifndef GF2_CANTOR_CORE_H
#define GF2_CANTOR_CORE_H

#include <stdint.h>

#if defined(__CUDACC__)
#define GF2C_HD __host__ __device__ __forceinline__
#else
#define GF2C_HD static inline
#endif

typedef uint32_t u32;
typedef uint64_t u64;

// ---------------------------------------------------------------------------
// carry-less 32x32 -> 64 via masked integer multiplies (portable, GPU-safe)
GF2C_HD u64 cl32_masked(u32 a, u32 b) {
    const u64 M0 = 0x1111111111111111ULL;
    const u64 M1 = M0 << 1, M2 = M0 << 2, M3 = M0 << 3;
    u64 a0 = a & 0x11111111u, a1 = a & 0x22222222u,
        a2 = a & 0x44444444u, a3 = a & 0x88888888u;
    u64 b0 = b & 0x11111111u, b1 = b & 0x22222222u,
        b2 = b & 0x44444444u, b3 = b & 0x88888888u;
    u64 r0 = (a0 * b0) ^ (a1 * b3) ^ (a2 * b2) ^ (a3 * b1);
    u64 r1 = (a0 * b1) ^ (a1 * b0) ^ (a2 * b3) ^ (a3 * b2);
    u64 r2 = (a0 * b2) ^ (a1 * b1) ^ (a2 * b0) ^ (a3 * b3);
    u64 r3 = (a0 * b3) ^ (a1 * b2) ^ (a2 * b1) ^ (a3 * b0);
    return (r0 & M0) ^ (r1 & M1) ^ (r2 & M2) ^ (r3 & M3);
}

// carry-less 64x64 -> 128 (hi, lo) via Karatsuba over cl32
struct cl128 { u64 lo, hi; };

GF2C_HD cl128 cl64(u64 a, u64 b) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32);
    u32 b0 = (u32)b, b1 = (u32)(b >> 32);
    u64 lo = cl32_masked(a0, b0);
    u64 hi = cl32_masked(a1, b1);
    u64 mid = cl32_masked(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;
    cl128 r;
    r.lo = lo ^ (mid << 32);
    r.hi = hi ^ (mid >> 32);
    return r;
}

// reduce 128-bit carry-less product modulo x^64 + x^4 + x^3 + x + 1
GF2C_HD u64 gf64_reduce(cl128 t) {
    u64 h = t.hi;
    u64 lo = t.lo ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4);
    u64 h2 = (h >> 63) ^ (h >> 61) ^ (h >> 60);   // spill bits, < 2^4
    lo ^= h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4);
    return lo;
}

GF2C_HD u64 gf64_mul_portable(u64 a, u64 b) { return gf64_reduce(cl64(a, b)); }

// On x86 hosts, use PCLMULQDQ for the reference/host path (validated
// against the portable path at startup by the test program).
#if !defined(__CUDA_ARCH__) && defined(__x86_64__)
#include <immintrin.h>
static inline cl128 cl64_x86(u64 a, u64 b) {
    __m128i p = _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long)a),
                                     _mm_cvtsi64_si128((long long)b), 0x00);
    cl128 r;
    r.lo = (u64)_mm_cvtsi128_si64(p);
    r.hi = (u64)_mm_extract_epi64(p, 1);
    return r;
}
static inline u64 gf64_mul(u64 a, u64 b) { return gf64_reduce(cl64_x86(a, b)); }
#else
GF2C_HD u64 gf64_mul(u64 a, u64 b) { return gf64_mul_portable(a, b); }
#endif

GF2C_HD u64 gf64_sqr(u64 a) { return gf64_mul(a, a); }

// ---------------------------------------------------------------------------
// Forward/inverse butterflies.  Points at pair index t use twiddle W[t]
// (independent of depth; see cantor_fft notes).
GF2C_HD void bfly_fwd(u64 &e1, u64 &e2, u64 tw) {
    u64 g = e1, h = e2;
    u64 e = g ^ gf64_mul(tw, h);
    e1 = e;
    e2 = e ^ h;
}
GF2C_HD void bfly_inv(u64 &e1, u64 &e2, u64 tw) {
    u64 h = e1 ^ e2;
    u64 g = e1 ^ gf64_mul(tw, h);
    e1 = g;
    e2 = h;
}

#endif // GF2_CANTOR_CORE_H

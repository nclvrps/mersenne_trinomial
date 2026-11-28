/*
Copyright (C) 2026  Gord Palameta
(actual author: Claude AI)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
*/

/* gf2_trin_sqr.cuh -- squaring modulo the trinomial x^r + x^s + 1 over
   GF(2) on a GPU, 1 <= s <= (r-1)/2, r odd.

   Self-contained (no other project header); used by irred_gpu.cu and
   written so that tsfactor.cu can include it too.  Compiles with nvcc, or
   with g++ under the CUDA emulation (cuda_emul.h, -DTRSQ_EMUL).

   Polynomials: nw = ceil(r/64) words, canonical (bits >= r zero), followed
   by TRSQ_PAD zero words that the kernels read but never write.

   The square of a is SQ = sum a_i x^(2i) (degree <= 2r-2).  Reducing
   x^k = x^(k-r) + x^(k-r+s) twice gives, for output bit t (0 <= t < r),

     out(t) = SQ[t] + SQ[t+r] + SQ[t+r-s]   [t >= s]
                    + SQ[t+2r-s]            (non-zero only for t < s-1)
                    + SQ[t+2r-2s]           [t >= s] (non-zero only for t < 2s-1)

   where SQ[k] = 0 for k > 2r-2 (a has no bits >= r), so only two of the
   bounds need a mask.  A 64-bit window of SQ at an offset c is the spread
   of a 32-bit window of a at bit ceil(c/2), shifted up by one when c is
   odd.  So one 64-bit window of a gives two output words of one term.

   Kernels:
     trsq_fused   tsfactor's k_square_fused (round 4), unchanged: one thread
                  per output word, five clipped 32-bit windows (10 loads).
     trsq_k2/4/8  one thread per K output words: per term one funnel-shifted
                  window of K/2 + 1 words, masks only where a bound falls.
                  About 2.5 loads per output word for K = 2, 2 for K = 4.
   All produce identical results; irred_gpu times them at start and uses
   the fastest (the first two terms read the input twice and the third
   once more, so the floor is about 2.5 reads + 1 write of a polynomial). */

#ifndef GF2_TRIN_SQR_CUH
#define GF2_TRIN_SQR_CUH

#include <stddef.h>

typedef unsigned long long trsq_u64;

#ifdef TRSQ_EMUL
#define TRSQ_D static inline
#define TRSQ_RESTRICT __restrict__
#define TRSQ_UNROLL
#define TRSQ_LAUNCH(kern, blocks, threads, ...) \
    make_launcher((unsigned)(blocks), (unsigned)(threads), kern)(__VA_ARGS__)
#else
#define TRSQ_D __device__ __forceinline__
#define TRSQ_RESTRICT __restrict__
#define TRSQ_UNROLL _Pragma("unroll")
#define TRSQ_LAUNCH(kern, blocks, threads, ...) \
    kern<<<(blocks), (threads)>>>(__VA_ARGS__)
#endif

#define TRSQ_PAD 8              /* zero words after each polynomial */

/* one term: SQ window at offset c <-> a window at bit h = ceil(c/2) */
struct TrsqTerm {
    unsigned ow;                /* h div 64 */
    unsigned sh;                /* h mod 64 */
    unsigned odd;               /* c mod 2: spread to the odd positions */
};

struct TrsqParams {
    trsq_u64 r, s;
    size_t nw;                  /* words of a polynomial */
    trsq_u64 topmask;           /* valid bits of word nw-1 */
    TrsqTerm t[5];              /* c = 0, r, r-s, 2r-s, 2r-2s */
};

static inline TrsqParams trsq_params(trsq_u64 r, trsq_u64 s)
{
    TrsqParams P;
    P.r = r;
    P.s = s;
    P.nw = (size_t)((r + 63) / 64);
    unsigned topbits = (unsigned)(r - 64 * (P.nw - 1));      /* 1..64 */
    P.topmask = topbits == 64 ? ~0ULL : ((1ULL << topbits) - 1);
    const trsq_u64 c[5] = { 0, r, r - s, 2 * r - s, 2 * r - 2 * s };
    for (int i = 0; i < 5; i++) {
        trsq_u64 h = (c[i] + 1) / 2;
        P.t[i].ow = (unsigned)(h / 64);
        P.t[i].sh = (unsigned)(h % 64);
        P.t[i].odd = (unsigned)(c[i] & 1);
    }
    return P;
}

TRSQ_D trsq_u64 trsq_spread32(trsq_u64 v)
{
    trsq_u64 x = v & 0xFFFFFFFFULL;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

/* mask of the bits t >= T of the word whose bit 0 is t = t0 */
TRSQ_D trsq_u64 trsq_ge(trsq_u64 t0, trsq_u64 T)
{
    if (t0 >= T) return ~0ULL;
    if (t0 + 64 <= T) return 0;
    return ~0ULL << (unsigned)(T - t0);
}

/* ---- tsfactor's fused kernel (verbatim apart from names) ---------------- */

TRSQ_D trsq_u64 trsq_in32(const trsq_u64 *in, size_t nw, trsq_u64 q)
{
    size_t w0 = (size_t)(q >> 6);
    unsigned sh = (unsigned)(q & 63);
    trsq_u64 a = (w0 < nw) ? in[w0] : 0;
    trsq_u64 b = (w0 + 1 < nw) ? in[w0 + 1] : 0;
    trsq_u64 v = sh ? ((a >> sh) | (b << (64 - sh))) : a;
    return v & 0xFFFFFFFFULL;
}

TRSQ_D trsq_u64 trsq_vspread(const trsq_u64 *in, size_t nw, trsq_u64 pos,
                             trsq_u64 lo, trsq_u64 hi)
{
    trsq_u64 v;
    if ((pos & 1) == 0)
        v = trsq_spread32(trsq_in32(in, nw, pos >> 1));
    else
        v = trsq_spread32(trsq_in32(in, nw, (pos + 1) >> 1)) << 1;
    trsq_u64 lc = (lo > pos) ? lo - pos : 0;
    trsq_u64 hc = (hi > pos) ? hi - pos : 0;
    if (lc >= 64 || hc == 0) return 0;
    if (hc > 64) hc = 64;
    trsq_u64 msk = (hc >= 64 ? ~0ULL : ((1ULL << hc) - 1));
    if (lc) msk &= ~((1ULL << lc) - 1);
    return v & msk;
}

__global__ void trsq_fused(const trsq_u64 *in, trsq_u64 *out, TrsqParams P)
{
    const trsq_u64 r = P.r, s = P.s;
    const size_t nw = P.nw;
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) {
        trsq_u64 b = (trsq_u64)w << 6;
        trsq_u64 v = trsq_vspread(in, nw, b,               0,         r);
        v ^= trsq_vspread(in, nw, b + r,              r,         2*r - 1);
        v ^= trsq_vspread(in, nw, b + r - s,          r,         2*r - s);
        v ^= trsq_vspread(in, nw, b + 2*r - s,        2*r - s,   2*r - 1);
        v ^= trsq_vspread(in, nw, b + 2*(r - s),      2*r - s,   2*r - 1);
        out[w] = v;
    }
}

/* ---- the wide kernels ---------------------------------------------------- */

/* K/2 consecutive 64-bit windows of a starting at word w, bit sh: needs
   words w .. w + K/2 */
template <int K>
TRSQ_D void trsq_windows(const trsq_u64 *TRSQ_RESTRICT a, size_t w,
                         unsigned sh, trsq_u64 *A)
{
    trsq_u64 x[K / 2 + 1];
TRSQ_UNROLL
    for (int i = 0; i <= K / 2; i++) x[i] = a[w + i];
    if (sh == 0) {
TRSQ_UNROLL
        for (int i = 0; i < K / 2; i++) A[i] = x[i];
    }
    else {
TRSQ_UNROLL
        for (int i = 0; i < K / 2; i++)
            A[i] = (x[i] >> sh) | (x[i + 1] << (64 - sh));
    }
}

/* v[0..K) ^= the K output words of one term, each masked by mask(t0) */
template <int K>
TRSQ_D void trsq_term(const trsq_u64 *TRSQ_RESTRICT a, size_t base,
                      const TrsqTerm &T, trsq_u64 *v)
{
    trsq_u64 A[K / 2];
    trsq_windows<K>(a, base + T.ow, T.sh, A);
TRSQ_UNROLL
    for (int q = 0; q < K / 2; q++) {
        v[2 * q]     ^= trsq_spread32(A[q]) << T.odd;
        v[2 * q + 1] ^= trsq_spread32(A[q] >> 32) << T.odd;
    }
}

template <int K>
TRSQ_D void trsq_term_ge(const trsq_u64 *TRSQ_RESTRICT a, size_t base,
                         const TrsqTerm &T, trsq_u64 b0, trsq_u64 lim,
                         trsq_u64 *v)
{
    trsq_u64 A[K / 2];
    trsq_windows<K>(a, base + T.ow, T.sh, A);
TRSQ_UNROLL
    for (int q = 0; q < K / 2; q++) {
        const trsq_u64 t0 = b0 + 128 * (trsq_u64)q;
        v[2 * q]     ^= (trsq_spread32(A[q]) << T.odd) & trsq_ge(t0, lim);
        v[2 * q + 1] ^= (trsq_spread32(A[q] >> 32) << T.odd) &
                        trsq_ge(t0 + 64, lim);
    }
}

template <int K>
TRSQ_D void trsq_wide_body(const trsq_u64 *TRSQ_RESTRICT a,
                           trsq_u64 *TRSQ_RESTRICT o, const TrsqParams &P)
{
    const size_t nthr = (P.nw + K - 1) / K;
    const trsq_u64 s = P.s;
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < nthr;
         j += (size_t)gridDim.x * blockDim.x) {
        const size_t base = j * (K / 2);            /* a word of term 0 */
        const trsq_u64 b0 = (trsq_u64)j * (64 * K); /* first output bit */
        trsq_u64 v[K];
TRSQ_UNROLL
        for (int i = 0; i < K; i++) v[i] = 0;
        trsq_term<K>(a, base, P.t[0], v);           /* SQ[t] */
        trsq_term<K>(a, base, P.t[1], v);           /* SQ[t+r] */
        if (b0 + 64 * K > s)                        /* SQ[t+r-s], t >= s */
            trsq_term_ge<K>(a, base, P.t[2], b0, s, v);
        if (b0 < s)                                 /* SQ[t+2r-s] */
            trsq_term<K>(a, base, P.t[3], v);
        if (b0 < 2 * s && b0 + 64 * K > s)          /* SQ[t+2r-2s], t >= s */
            trsq_term_ge<K>(a, base, P.t[4], b0, s, v);
TRSQ_UNROLL
        for (int i = 0; i < K; i++) {
            const size_t w = j * K + i;
            if (w + 1 < P.nw) o[w] = v[i];
            else if (w + 1 == P.nw) o[w] = v[i] & P.topmask;
        }
    }
}

__global__ void trsq_k2(const trsq_u64 *TRSQ_RESTRICT a,
                        trsq_u64 *TRSQ_RESTRICT o, TrsqParams P)
{ trsq_wide_body<2>(a, o, P); }

__global__ void trsq_k4(const trsq_u64 *TRSQ_RESTRICT a,
                        trsq_u64 *TRSQ_RESTRICT o, TrsqParams P)
{ trsq_wide_body<4>(a, o, P); }

__global__ void trsq_k8(const trsq_u64 *TRSQ_RESTRICT a,
                        trsq_u64 *TRSQ_RESTRICT o, TrsqParams P)
{ trsq_wide_body<8>(a, o, P); }

/* ---- launching ----------------------------------------------------------- */

enum { TRSQ_FUSED, TRSQ_K2, TRSQ_K4, TRSQ_K8, TRSQ_NKERN };
static const char *const trsq_names[TRSQ_NKERN] = { "fused", "k2", "k4", "k8" };
static const int trsq_width[TRSQ_NKERN] = { 1, 2, 4, 8 };

/* out = in^2 mod (x^r + x^s + 1); in != out, both canonical with the pad */
static inline void trsq_square(int kern, const trsq_u64 *in, trsq_u64 *out,
                               const TrsqParams &P)
{
    const int threads = 256;
    size_t nthr = (P.nw + trsq_width[kern] - 1) / trsq_width[kern];
    size_t b = (nthr + threads - 1) / threads;
    int blocks = (int)(b > 65535 ? 65535 : (b ? b : 1));
    switch (kern) {
    case TRSQ_FUSED: TRSQ_LAUNCH(trsq_fused, blocks, threads, in, out, P); break;
    case TRSQ_K2:    TRSQ_LAUNCH(trsq_k2, blocks, threads, in, out, P); break;
    case TRSQ_K4:    TRSQ_LAUNCH(trsq_k4, blocks, threads, in, out, P); break;
    default:         TRSQ_LAUNCH(trsq_k8, blocks, threads, in, out, P); break;
    }
}

#endif /* GF2_TRIN_SQR_CUH */

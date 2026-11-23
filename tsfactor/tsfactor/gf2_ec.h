/*
Copyright (C) 2026  Gord Palameta
(actual author: Claude AI)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Some code may have been inspired by:
  GF2X library
  factor.cpp and other files in GF2X library apps subdirectory
  NTL library
*/

// gf2_ec.h
// Error-checking layer for tsfactor: detects hardware faults (VRAM / RAM
// bit flips, marginal overclocks, transfer corruption) in the blocked
// distinct-degree scan, cheaply enough to leave on by default.
//
// Theory, derivations and the measured behaviour are in
// errorcheck_handoff.md; the tsfactor-specific design is in
// tsfactor_handoff.md (round 7).  In brief, four independent mechanisms:
//
//  1. h-table (the squarings, ~all of the state that persists):
//     with S = XOR_j h[j] and sigma = Frobenius^m, the block update
//     h[j] <- h[j]^(2^m) gives sigma(S_z) = S_{z+1}, because Frobenius is
//     ADDITIVE in characteristic 2.  Summed over a window of blocks,
//         sigma(D) + D = S_first + S_last,   D = XOR of the S_z,
//     which is checked at full width (no compression): any single fault
//     in any squaring, or any bit flip in the idle table, is detected
//     with certainty.  Cost: one streaming pass per block plus m
//     squarings per window.
//
//  2. the table's absolute value: S has a closed form,
//         1 + S_k = (1 + x)^(2^k (2^m - 1))   mod T,
//     so  (1 + y_k)(1 + S_k) = 1 + y_{k+m}  with y_t = x^(2^t).  One
//     single-slot Frobenius chain and one modmul verify the whole h-table
//     initialisation from first principles (no duplicate init).
//
//  3. products A <- A*C mod T and the Horner build of C: a cyclic fold
//         phi : GF(2)[x] -> GF(2)[x] / (x^n + 1)
//     is a ring homomorphism, so phi(t) = phi(A) phi(C) for the UNREDUCED
//     product t, and the reduction satisfies phi(t mod T) = phi(t) +
//     phi(q) phi(T) with the quotient known in closed form when 2s < r
//     (which tsfactor enforces).  phi(A) is carried on the host from one
//     block to the next and phi(C) is predicted from an independent pass
//     over the h-table, so idle corruption of A or C is caught as well.
//
//  4. transfers and snapshots: the same fold computed on both sides.
//
// The fold length is EC_NF = 61 words (n = 3904 bits), deliberately NOT a
// power of two: with n = 2^12 an error that repeats with a power-of-two
// period (a stuck bus lane; a constant pattern across a power-of-two FFT
// sub-block) folds onto itself an even number of times and cancels.  61
// is prime, so such patterns spread over all 61 words instead.
//
// Everything here is header-only and has no dependency beyond
// gf2_cantor_core.h, so the fold algebra can be unit-tested on the host.

#ifndef GF2_EC_H
#define GF2_EC_H

#include "gf2_cantor_core.h"
#include <stddef.h>
#include <string.h>

#define EC_NF 61                          // fold words (prime, see above)
#define EC_NBITS ((u64)EC_NF * 64)        // n = 3904
#ifndef EC_NBLK_MAX
#define EC_NBLK_MAX 512                   // blocks of EC_NF threads per fold
#endif

// ===========================================================================
// host side: arithmetic in the fold ring GF(2)[x] / (x^n + 1)

struct EcFold { u64 w[EC_NF]; };

static inline void ecf_zero(EcFold &f) { memset(f.w, 0, sizeof f.w); }
static inline void ecf_one(EcFold &f) { ecf_zero(f); f.w[0] = 1; }
static inline bool ecf_eq(const EcFold &a, const EcFold &b) {
    return memcmp(a.w, b.w, sizeof a.w) == 0;
}
static inline bool ecf_is_zero(const EcFold &a) {
    for (int j = 0; j < EC_NF; j++) if (a.w[j]) return false;
    return true;
}
static inline void ecf_xor(EcFold &d, const EcFold &a) {
    for (int j = 0; j < EC_NF; j++) d.w[j] ^= a.w[j];
}
// toggle the coefficient of x^pos (pos taken mod n)
static inline void ecf_flip(EcFold &f, u64 pos) {
    pos %= EC_NBITS;
    f.w[pos >> 6] ^= 1ULL << (pos & 63);
}
// d = a * x^k (k any integer mod n): a rotation.  d must not alias a.
static inline void ecf_rot(EcFold &d, const EcFold &a, u64 k) {
    k %= EC_NBITS;
    unsigned kw = (unsigned)(k >> 6), kb = (unsigned)(k & 63);
    for (unsigned j = 0; j < EC_NF; j++) {
        u64 lo = a.w[(j + EC_NF - kw) % EC_NF];
        u64 hi = a.w[(j + 2 * EC_NF - kw - 1) % EC_NF];
        d.w[j] = kb ? ((lo << kb) | (hi >> (64 - kb))) : lo;
    }
}
// d = a * x^(-k)
static inline void ecf_rot_neg(EcFold &d, const EcFold &a, u64 k) {
    ecf_rot(d, a, EC_NBITS - (k % EC_NBITS));
}

static inline cl128 ecf_cl64(u64 a, u64 b) {
#if !defined(__CUDA_ARCH__) && defined(__x86_64__) && \
    defined(__PCLMUL__) && defined(__SSE4_1__)
    return cl64_x86(a, b);
#else
    return cl64(a, b);
#endif
}
// d = a * b in the fold ring (schoolbook: 3721 carry-less word products,
// ~10 us with PCLMUL, ~0.1 ms without -- once per block either way)
static inline void ecf_mul(EcFold &d, const EcFold &a, const EcFold &b) {
    u64 acc[2 * EC_NF];
    memset(acc, 0, sizeof acc);
    for (int i = 0; i < EC_NF; i++) {
        if (!a.w[i]) continue;
        for (int j = 0; j < EC_NF; j++) {
            cl128 p = ecf_cl64(a.w[i], b.w[j]);
            acc[i + j] ^= p.lo;
            acc[i + j + 1] ^= p.hi;
        }
    }
    for (int j = 0; j < EC_NF; j++) d.w[j] = acc[j] ^ acc[j + EC_NF];
}
// d = a * phi(x^r + x^s + 1): three rotations, no general product
static inline void ecf_mul_trinomial(EcFold &d, const EcFold &a, u64 r,
                                     u64 s) {
    EcFold t;
    ecf_rot(d, a, r);
    ecf_rot(t, a, s);
    ecf_xor(d, t);
    ecf_xor(d, a);
}
// fold of a host buffer
static inline void ecf_fold_host(EcFold &f, const u64 *src, size_t nw) {
    ecf_zero(f);
    size_t w = 0;
    for (; w + EC_NF <= nw; w += EC_NF)
        for (int j = 0; j < EC_NF; j++) f.w[j] ^= src[w + j];
    for (int j = 0; w < nw; w++, j++) f.w[j] ^= src[w];
}

// ---------------------------------------------------------------------------
// predictions used by the per-block checks (pure functions of folds)

// Fold of the block polynomial C as left in dC by k_horner_fused +
// k_horner_tail, given the fold of the UNREDUCED Horner sum
//     Cu = XOR_j (h[j] << (m-1-j))                     (r+m-1 bits)
// and the mask `fired` of the top bits the tail kernel folded down
// (bit e set <=> x^(r+e) was present and x^e * T was added):
//     C = Cu + x^m + XOR_{e in fired} x^e (x^r + x^s + 1).
static inline void ecf_horner_predict(EcFold &fC, const EcFold &fCu,
                                      u64 fired, int m, u64 r, u64 s) {
    fC = fCu;
    for (int e = 0; e < 64; e++)
        if ((fired >> e) & 1) {
            ecf_flip(fC, r + (u64)e);
            ecf_flip(fC, s + (u64)e);
            ecf_flip(fC, (u64)e);
        }
    ecf_flip(fC, (u64)m);
}

// From the three segment folds of the unreduced product t (deg <= 2r-2),
//     seg[0] = bits [0, r)   seg[1] = bits [r, 2r-s)   seg[2] = bits >= 2r-s
// (each at its original position), produce phi(t) and the predicted
// phi(t mod T).  Valid for 2s < r, where t mod T = t + q*T with
//     q = t_H + u_H,   t_H = t >> r,   u_H = t >> (2r - s)
// (the two fold passes of reduce_into).  For 2s > r the quotient has
// more terms and this prediction is WRONG; tsfactor's set_s rejects it.
static inline void ecf_reduce_predict(EcFold &ft, EcFold &fred,
                                      const EcFold seg[3], u64 r, u64 s) {
    EcFold hi, q, uh, qT;
    ft = seg[0];
    ecf_xor(ft, seg[1]);
    ecf_xor(ft, seg[2]);
    hi = seg[1];
    ecf_xor(hi, seg[2]);                  // phi(x^r * t_H)
    ecf_rot_neg(q, hi, r);                // phi(t_H)
    ecf_rot_neg(uh, seg[2], 2 * r - s);   // phi(u_H)
    ecf_xor(q, uh);
    ecf_mul_trinomial(qT, q, r, s);
    fred = ft;
    ecf_xor(fred, qT);
}

// Translation units that only need the host-side fold algebra (the gcd
// engine) define GF2_EC_HOST_ONLY before including this header.
#ifndef GF2_EC_HOST_ONLY

// ===========================================================================
// device side.  Fold kernels are launched <<<nblk, EC_NF>>>: thread t of
// block b owns fold slot t and visits words (b + i*nblk)*EC_NF + t, so
// every partial result has exactly one writer (no atomics, exact under
// the sequential emulation) and each block reads EC_NF consecutive words.

static inline unsigned ec_nblk_for(size_t nw) {
    size_t b = (nw + EC_NF - 1) / EC_NF;
    if (b < 1) b = 1;
    if (b > EC_NBLK_MAX) b = EC_NBLK_MAX;
    return (unsigned)b;
}

// mask of the bits of the word covering [wbit, wbit+64) that lie in [lo, hi)
GF2C_HD u64 ec_mask_range(u64 wbit, u64 lo, u64 hi) {
    u64 a = lo > wbit ? lo - wbit : 0;
    u64 b = hi > wbit ? hi - wbit : 0;
    if (a >= 64 || b == 0) return 0;
    if (b > 64) b = 64;
    if (a >= b) return 0;
    u64 m = (b >= 64) ? ~0ULL : ((1ULL << b) - 1);
    if (a) m &= ~((1ULL << a) - 1);
    return m;
}

// part[b*EC_NF + t] = XOR of src[w], w = (b + i*nblk)*EC_NF + t < nw
__global__ void k_ec_fold_part(const u64 *src, size_t nw, u64 *part) {
    const size_t t = threadIdx.x;
    const size_t step = (size_t)gridDim.x * EC_NF;
    u64 acc = 0;
    for (size_t w = (size_t)blockIdx.x * EC_NF + t; w < nw; w += step)
        acc ^= src[w];
    part[(size_t)blockIdx.x * EC_NF + t] = acc;
}

// three folds in one pass: bits [0,c1), [c1,c2), [c2,inf) of src, each
// folded at its original position.  part holds 3 * nblk * EC_NF words.
__global__ void k_ec_fold3_part(const u64 *src, size_t nw, u64 c1, u64 c2,
                                u64 *part) {
    const size_t t = threadIdx.x;
    const size_t step = (size_t)gridDim.x * EC_NF;
    u64 a0 = 0, a1 = 0, a2 = 0;
    for (size_t w = (size_t)blockIdx.x * EC_NF + t; w < nw; w += step) {
        u64 v = src[w];
        u64 wb = (u64)w << 6;
        u64 m0 = ec_mask_range(wb, 0, c1);
        u64 m1 = ec_mask_range(wb, c1, c2);
        a0 ^= v & m0;
        a1 ^= v & m1;
        a2 ^= v & ~(m0 | m1);
    }
    const size_t o = (size_t)blockIdx.x * EC_NF + t;
    const size_t seg = (size_t)gridDim.x * EC_NF;
    part[o] = a0;
    part[seg + o] = a1;
    part[2 * seg + o] = a2;
}

// out[g*EC_NF + t] = XOR_b part[(g*nblk + b)*EC_NF + t], g < nseg.
// Launch <<<1, EC_NF>>>.
__global__ void k_ec_combine(const u64 *part, unsigned nblk, unsigned nseg,
                             u64 *out) {
    if (blockIdx.x != 0) return;
    const size_t t = threadIdx.x;
    for (unsigned g = 0; g < nseg; g++) {
        u64 acc = 0;
        for (unsigned b = 0; b < nblk; b++)
            acc ^= part[((size_t)g * nblk + b) * EC_NF + t];
        out[(size_t)g * EC_NF + t] = acc;
    }
}

// The one extra pass over the h-table per block.  For every word w < nwc:
//   x  = XOR_j h[j][w]                      -> D[w] ^= x   (window digest)
//   cu = XOR_j (h[j] << (m-1-j))[w]         -> folded into part
// The second quantity is the unreduced Horner sum, recomputed here
// independently of k_horner_fused (different kernel, different
// thread-to-word mapping), so a fault in either is exposed when the
// product check compares phi(A*C) with phi(A) * phi(C predicted).
__global__ void k_ec_block_pass(const u64 *hbase, size_t nw, int m, u64 *D,
                                size_t nwc, u64 *part) {
    const size_t t = threadIdx.x;
    const size_t step = (size_t)gridDim.x * EC_NF;
    u64 acc = 0;
    for (size_t w = (size_t)blockIdx.x * EC_NF + t; w < nwc; w += step) {
        u64 x = 0, cu = 0;
        for (int j = 0; j < m; j++) {
            const u64 *h = hbase + (size_t)j * nw;
            unsigned sh = (unsigned)(m - 1 - j);
            u64 a = (w < nw) ? h[w] : 0;
            u64 b = (w >= 1 && w - 1 < nw) ? h[w - 1] : 0;
            x ^= a;
            cu ^= sh ? ((a << sh) | (b >> (64 - sh))) : a;
        }
        if (w < nw) D[w] ^= x;
        acc ^= cu;
    }
    part[(size_t)blockIdx.x * EC_NF + t] = acc;
}

// out[w] = XOR_j h[j][w]   (the digest S, full width)
__global__ void k_ec_digest(const u64 *hbase, size_t nw, int m, u64 *out) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; w < nw; w += stride) {
        u64 x = 0;
        for (int j = 0; j < m; j++) x ^= hbase[(size_t)j * nw + w];
        out[w] = x;
    }
}

// *flag = 1 if (a ^ b ^ c ^ d) is nonzero anywhere.  Every writer stores
// the same value, so the race between writers is benign.
__global__ void k_ec_or4(const u64 *a, const u64 *b, const u64 *c,
                         const u64 *d, size_t nw, u64 *flag) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    u64 bad = 0;
    for (; w < nw; w += stride) bad |= a[w] ^ b[w] ^ c[w] ^ d[w];
    if (bad) *flag = 1;
}
__global__ void k_ec_neq(const u64 *a, const u64 *b, size_t nw, u64 *flag) {
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    u64 bad = 0;
    for (; w < nw; w += stride) bad |= a[w] ^ b[w];
    if (bad) *flag = 1;
}
// single-thread helpers
__global__ void k_ec_set_word(u64 *p, u64 v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *p = v;
}
__global__ void k_ec_flip_bit(u64 *p, u64 bit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) p[bit >> 6] ^= 1ULL << (bit & 63);
}

#endif // GF2_EC_HOST_ONLY

#endif // GF2_EC_H

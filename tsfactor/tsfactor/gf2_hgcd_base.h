// gf2_hgcd_base.h -- word-level half-gcd over GF(2) for small operands.
//
// The bottom of the gcd engine's recursion (gf2_gcd_ntl.cpp, hyb::hgcd).
// Until round 5 that bottom was a bit-at-a-time shift-xor loop on
// std::vector operands with one NTL-mediated product per quotient: at
// r = 136279841 it performed 136 million tiny multiplications and was
// HALF of every gcd.  This replaces it with the polynomial form of Lehmer's
// method:
//
//   * take the top 64 bits of the pair (same alignment for both);
//   * run Euclid on those two machine words, accumulating a 2x2 matrix of
//     word-sized polynomials -- shifts and xors only;
//   * apply that matrix to the full operands and to the accumulated matrix
//     with carry-less word multiplications, about 31 quotient bits at a time.
//
// EXACTNESS.  Over GF(2)[x] there are no carries, so the rule is clean: with
// the dividend's leading bit at bit 63 of the window, a division step whose
// divisor has window degree v is the TRUE step if and only if 2v >= 63.
// (The true remainder is  window_remainder * x^j + e  with deg e < j + (63 -
// v'), v' the previous divisor's window degree, and the quotient depends
// only on the top (u - v + 1) bits of dividend and divisor.)  So the window
// Euclid runs while the divisor's degree is >= 32 -- which is the ordinary
// half-gcd stopping rule for a 64-bit pair -- and everything it produces is
// exact, not approximate: the resulting quotient sequence, hence the matrix
// and the reduced pair, are IDENTICAL to what the old loop produced.  That
// is how it is tested: hgcd_base_ref below is the old algorithm, kept as the
// reference, and gcdstress compares the two on every input.
//
// No allocation, no library calls; needs only a 64x64 carry-less multiply
// (ecf_cl64 from gf2_ec.h: PCLMULQDQ when the compiler flags allow it, the
// portable version otherwise).
#pragma once
#include <cstdint>
#include <cstring>

namespace hgb {

static const int MAX_BITS = 16384;               // largest supported operand
static const int MAXW = MAX_BITS / 64 + 3;       // operand words
static const int MAXM = MAX_BITS / 128 + 4;      // matrix-entry words

static inline int deg64(u64 x) { return 63 - __builtin_clzll(x); }

// degree of x[0..n-1], -1 if zero; *top receives the index of the top word
static inline int64_t mdeg(const u64 *x, int n, int *top = nullptr) {
    for (int i = n - 1; i >= 0; i--)
        if (x[i]) {
            if (top) *top = i;
            return (int64_t)i * 64 + deg64(x[i]);
        }
    if (top) *top = -1;
    return -1;
}

// bits j .. j+63 of x[0..n-1]
static inline u64 window(const u64 *x, int n, int64_t j) {
    int wi = (int)(j >> 6), bo = (int)(j & 63);
    u64 lo = wi < n ? x[wi] : 0;
    if (!bo) return lo;
    u64 hi = wi + 1 < n ? x[wi + 1] : 0;
    return (lo >> bo) | (hi << (64 - bo));
}

// dst[0..n] = m0 * x[0..n-1] + m1 * y[0..n-1]   (m0, m1: single words)
static inline void lin2(u64 *dst, u64 m0, const u64 *x, u64 m1, const u64 *y,
                        int n) {
    u64 carry = 0;
    for (int i = 0; i < n; i++) {
        cl128 p = ecf_cl64(m0, x[i]), q = ecf_cl64(m1, y[i]);
        dst[i] = p.lo ^ q.lo ^ carry;
        carry = p.hi ^ q.hi;
    }
    dst[n] = carry;
}

// dst[0..] ^= src[0..n-1] << sh ; returns the number of words touched
static inline int xor_shl(u64 *dst, const u64 *src, int n, int64_t sh) {
    int wo = (int)(sh >> 6), bo = (int)(sh & 63);
    if (!bo) {
        for (int i = 0; i < n; i++) dst[i + wo] ^= src[i];
        return n + wo;
    }
    u64 carry = 0;
    for (int i = 0; i < n; i++) {
        dst[i + wo] ^= (src[i] << bo) | carry;
        carry = src[i] >> (64 - bo);
    }
    dst[n + wo] ^= carry;
    return n + wo + 1;
}

// Euclid on two words.  On entry (a, b) is the window pair; on exit
//     a_out = m[0] a + m[1] b,   b_out = m[2] a + m[3] b,
// after `return value` division steps, each taken only while the divisor
// has degree >= stop.  A step with deg a < deg b is the swap (quotient 0).
static inline int euclid64(u64 a, u64 b, int stop, u64 m[4]) {
    u64 m0 = 1, m1 = 0, m2 = 0, m3 = 1;
    int steps = 0;
    while (b) {
        int db = deg64(b);
        if (db < stop) break;
        while (a) {
            int sh = deg64(a) - db;
            if (sh < 0) break;
            a ^= b << sh;
            m0 ^= m2 << sh;
            m1 ^= m3 << sh;
        }
        u64 t;
        t = a; a = b; b = t;
        t = m0; m0 = m2; m2 = t;
        t = m1; m1 = m3; m3 = t;
        steps++;
    }
    m[0] = m0; m[1] = m1; m[2] = m2; m[3] = m3;
    return steps;
}

// The half-gcd of (a, b):  nw words each (zero-padded), deg a = n <= MAX_BITS.
// Performs Euclidean division steps while deg b >= mm and returns the
// reduced pair in a, b (nw words each) and, if M != nullptr, the transform
//     (a_out, b_out) = [[M0, M1], [M2, M3]] (a_in, b_in)
// as four arrays of *mw words (M must point to 4*MAXM words; entry e lives
// at M + e*MAXM).  *nsteps receives the number of division steps; 0 means
// the pair was already reduced and M is the identity.
// Returns false only if a size bound would be exceeded (never observed;
// the caller then uses the reference routine).
static bool hgcd_base(u64 *a_io, u64 *b_io, int nw, u64 mm, u64 *M, int *mw,
                      long *nsteps) {
    if (nw + 2 > MAXW) return false;
    u64 bufA[2][MAXW], bufB[2][MAXW], bufM[2][4][MAXM];
    int curM = 0;
    u64 *a = bufA[0], *b = bufB[0];
    memcpy(a, a_io, (size_t)nw * 8);
    memcpy(b, b_io, (size_t)nw * 8);
    a[nw] = a[nw + 1] = b[nw] = b[nw + 1] = 0;
    int mlen = 1;                                // words in use per entry
    if (M) {
        // invariant: in both matrix buffers every word at index >= mlen
        // is zero (the plain division step shifts rows into that space)
        int mcap = nw / 2 + 5 < MAXM ? nw / 2 + 5 : MAXM;
        for (int s2 = 0; s2 < 2; s2++)
            for (int e = 0; e < 4; e++)
                memset(bufM[s2][e], 0, (size_t)mcap * 8);
        bufM[0][0][0] = 1;
        bufM[0][3][0] = 1;
    }
    long steps = 0;
    int ta, tb, hi = nw - 1;                     // hi: bound on the top word
    for (;;) {
        int64_t da = mdeg(a, hi + 1, &ta), db = mdeg(b, hi + 1, &tb);
        if (db < 0 || (u64)db < mm) break;
        hi = ta > tb ? ta : tb;
        int len = (ta > tb ? ta : tb) + 1;
        u64 m[4];
        int st = 0;
        if (da >= db) {
            int64_t j = da >= 63 ? da - 63 : 0;
            int stop = 32;
            if (j == 0) stop = (int)mm;          // exact: nothing truncated
            else if ((int64_t)mm - j > 32) stop = (int)((int64_t)mm - j);
            st = euclid64(window(a, nw, j), window(b, nw, j), stop, m);
        }
        if (st == 0) {
            // quotient of degree >= 32 (or deg a < deg b): one plain
            // division step on the full operands
            if (M && mlen + (int)((da > db ? da - db : 0) >> 6) + 2 > MAXM)
                return false;
            const int rl = mlen;                 // extent of matrix row 1
            while (da >= db) {
                int64_t sh = da - db;
                xor_shl(a, b, tb + 1, sh);
                if (M) {
                    int w0 = xor_shl(bufM[curM][0], bufM[curM][2], rl, sh);
                    int w1 = xor_shl(bufM[curM][1], bufM[curM][3], rl, sh);
                    int w = w0 > w1 ? w0 : w1;
                    if (w > mlen) mlen = w;
                }
                da = mdeg(a, nw, &ta);
            }
            u64 *t = a; a = b; b = t;            // swap the pair ...
            if (M)                               // ... and the matrix rows
                for (int c = 0; c < 2; c++)
                    for (int i = 0; i < mlen; i++) {
                        u64 x = bufM[curM][c][i];
                        bufM[curM][c][i] = bufM[curM][2 + c][i];
                        bufM[curM][2 + c][i] = x;
                    }
            steps++;
            // trim mlen
            if (M)
                while (mlen > 1 && !(bufM[curM][0][mlen - 1] |
                                     bufM[curM][1][mlen - 1] |
                                     bufM[curM][2][mlen - 1] |
                                     bufM[curM][3][mlen - 1])) mlen--;
            continue;
        }
        steps += st;
        // (a, b) <- m (a, b)
        u64 *na = (a == bufA[0] || a == bufB[0]) ? bufA[1] : bufA[0];
        u64 *nb = (a == bufA[0] || a == bufB[0]) ? bufB[1] : bufB[0];
        lin2(na, m[0], a, m[1], b, len);
        lin2(nb, m[2], a, m[3], b, len);
        for (int i = len + 1; i < nw + 2; i++) na[i] = nb[i] = 0;
        a = na;
        b = nb;
        // M <- m M
        if (M) {
            if (mlen + 2 > MAXM) return false;
            int o = curM ^ 1;
            for (int c = 0; c < 2; c++) {
                lin2(bufM[o][c], m[0], bufM[curM][c], m[1],
                     bufM[curM][2 + c], mlen);
                lin2(bufM[o][2 + c], m[2], bufM[curM][c], m[3],
                     bufM[curM][2 + c], mlen);
            }
            curM = o;
            mlen++;
            while (mlen > 1 && !(bufM[curM][0][mlen - 1] |
                                 bufM[curM][1][mlen - 1] |
                                 bufM[curM][2][mlen - 1] |
                                 bufM[curM][3][mlen - 1])) mlen--;
            // (lin2 wrote index mlen-1 at most plus a carry word that the
            // trim just proved zero wherever it was dropped)
        }
    }
    memcpy(a_io, a, (size_t)nw * 8);
    memcpy(b_io, b, (size_t)nw * 8);
    if (M) {
        for (int e = 0; e < 4; e++)
            memcpy(M + (size_t)e * MAXM, bufM[curM][e], (size_t)mlen * 8);
        *mw = mlen;
    }
    *nsteps = steps;
    return true;
}

} // namespace hgb

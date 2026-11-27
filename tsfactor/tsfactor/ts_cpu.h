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

// ts_cpu.h
// TsCPU: the scan arithmetic of tsfactor on the CPU (round 10).  Same
// interface and the same state as the GPU backend, bit for bit -- the GPU
// build's selftest compares the two step by step -- so the whole driver in
// ts_core.h (schedule, certified gcd, checkpoints, error checking, fine
// DDF) runs unchanged on a machine without a GPU: tsfactor_cpu.
//
// Squaring modulo T = x^r + x^s + 1 (r odd, 2s < r) uses the in-place
// reduce-and-interleave algorithm of Brent, Larvala and Zimmermann (Math.
// Comp. 72 (2003), 1443-1452) as in factor.cpp's fastsqr, which needs s
// odd: factor.cpp flips an even s to the reciprocal trinomial (r - s is
// odd), and with it every residue, which is why its states could never be
// compared with tsfactor's.  Here the even case has its own reduction
// instead (derived below), so both parities work on T itself:
//
//   The square of a = sum a_i x^i, held in the input array itself, is read
//   in "scrambled" order: position p <= alpha = (r-1)/2 means x^(2p);
//   position p >= alpha+1 stands for x^(2p) = x^(2p-r) * x^r and is
//   rewritten with x^r = x^s + 1.  The term x^(2p-r) is odd and is kept at
//   position p (odd exponents 2p-r live in the upper half); the remaining
//   term x^(2p-r+s) is where the parities differ.
//   s odd:  x^(2p-r+s) is EVEN -- position p - (r-s)/2 -- and may itself
//           need the same rewrite: an in-place cascade, top down (BLZ).
//   s even: x^(2p-r+s) is ODD -- position p + s/2 -- final, no cascade --
//           unless 2p-r+s >= r, i.e. p >= r - s/2, where it reduces once
//           more to the even terms x^(2p-2r+s) and x^(2p-2r+2s), positions
//           p-r+s/2 and p-r+s (both below s/2 + s/2 = s <= alpha).
//           So: XOR the top s/2 bits into [0, s/2) and [s/2, s), then the
//           bits [alpha+1, r-s/2) into themselves shifted up by s/2 (top
//           down, so every source is read before it is written).
//   Both then interleave: even exponents from the lower half, odd ones
//   from the upper half.
// Verified against the bit-level oracle on thousands of (r, s) of both
// parities (cpu_kernel_suites below, and the GPU cross-check).  The odd
// path needs s > 128 (BLZ's boundary words); odd s <= 128 and the
// --variant 1 test path use a generic spread-and-fold squaring.
//
// Products: gf2x_mul (or NTL without -DHAVE_GF2X), then the two linear
// fold passes of the GPU's reduction, as word runs (tc_fold_pass_fast).
// Error-checking folds: word loops producing exactly the values the GPU
// kernels produce.
//
// Speed (r = 136279841, one core of the sandbox's 2.1 GHz Xeon, gf2x built
// from Gord's tree): a squaring 2.6-3.4 ms as the scan does them -- the
// same as factor_modified's fastsqr on the same machine -- a product
// 1.15-1.25 s, the reduction 8 ms; the tuning picks m ~ 19, and the scan
// runs at ~0.13 s per degree with checking, factor_ec's rate.  Two things
// matter for that.  The hot loops take plain values and __restrict__
// pointers, so they vectorise (the first version read the geometry
// through a struct reference, did not vectorise, and was 3x slower).  And
// the table advances one slot at a time (square_steps: m squarings of one
// slot, ping-ponging with one spare buffer), so the working set is two
// residues rather than the whole table (30-40% less per squaring here).

#ifndef TS_CPU_H
#define TS_CPU_H

#include "ts_core.h"
#ifdef HAVE_GF2X
#include <gf2x.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// word-level arithmetic modulo x^r + x^s + 1

// 64-bit window of src (nw words) at bit position pos, with source bits
// outside [lo, hi) cleared (host twin of gf2_trinomial_cuda.h's window)
static inline u64 tc_window(const u64 *src, size_t nw, u64 pos, u64 lo,
                            u64 hi) {
    size_t w0 = (size_t)(pos >> 6);
    unsigned sh = (unsigned)(pos & 63);
    u64 a = (w0 < nw) ? src[w0] : 0;
    u64 b = (w0 + 1 < nw) ? src[w0 + 1] : 0;
    u64 v = sh ? ((a >> sh) | (b << (64 - sh))) : a;
    u64 lc = (lo > pos) ? lo - pos : 0;
    u64 hc = (hi > pos) ? hi - pos : 0;
    if (lc >= 64 || hc == 0) return 0;
    if (hc > 64) hc = 64;
    u64 msk = (hc >= 64 ? ~0ULL : ((1ULL << hc) - 1));
    if (lc) msk &= ~((1ULL << lc) - 1);
    return v & msk;
}
// mask of the bits of the word at bit wbit that lie in [lo, hi)
static inline u64 tc_rmask(u64 wbit, u64 lo, u64 hi) {
    u64 a = lo > wbit ? lo - wbit : 0, b = hi > wbit ? hi - wbit : 0;
    if (a >= 64 || b == 0 || a >= b) return 0;
    if (b > 64) b = 64;
    u64 msk = b >= 64 ? ~0ULL : ((1ULL << b) - 1);
    if (a) msk &= ~((1ULL << a) - 1);
    return msk;
}

// one fold pass (gather form, as k_fold_pass): dst[0..dw) =
//   src[0, r) ^ (src[r, L) >> r) ^ (src[r, L) >> (r - s))
static void tc_fold_pass(const u64 *src, size_t snw, u64 *dst, size_t dw,
                         u64 r, u64 s, u64 L) {
    for (size_t w = 0; w < dw; w++) {
        u64 pos = (u64)w << 6;
        dst[w] = tc_window(src, snw, pos, 0, r) ^
                 tc_window(src, snw, pos + r, r, L) ^
                 tc_window(src, snw, pos + (r - s), r, L);
    }
}

struct SqGeom {
    u64 r = 0, s = 0;
    size_t nw = 0, q1 = 0;
    u64 alpha = 0, mask1 = 0;
    size_t q5 = 0;
    unsigned s1 = 0, s2 = 0;
    int path = 0;              // 1 odd-s BLZ, 2 even-s BLZ, 0 generic
    u64 delta = 0;             // odd: (r-s)/2
    size_t deltaw = 0, q4 = 0;
    unsigned deltaq = 0, deltaqc = 0;
    u64 mask2 = 0;
    u64 h = 0;                 // even: s/2
    void init(u64 r_, u64 s_) {
        r = r_; s = s_;
        nw = bits_to_words(r);
        q1 = (size_t)((r - 1) >> 6);
        alpha = r >> 1;
        mask1 = ~0ULL >> (63 - ((r - 1) & 63));
        q5 = (size_t)((alpha + 1) >> 6);
        s1 = (unsigned)((alpha + 1) & 63);
        s2 = 63 - s1;
        if ((s & 1) && s > 128 && r - s > 128 && r >= 257) {
            path = 1;
            delta = (r - s) >> 1;
            deltaw = (size_t)(delta >> 6);
            deltaq = (unsigned)(delta & 63);
            deltaqc = 64 - deltaq;
            q4 = (size_t)(alpha >> 6);
            mask2 = (~(u64)1) << (alpha & 63);
        } else if (!(s & 1) && s >= 2 && 2 * s < r && r >= 257) {
            path = 2;
            h = s >> 1;
        } else {
            path = 0;
        }
    }
};

static const u64 TC_IC0 = 0x00000000FFFFFFFFull, TC_IC1 = 0x0000FFFF0000FFFFull,
                 TC_IC2 = 0x00FF00FF00FF00FFull, TC_IC3 = 0x0F0F0F0F0F0F0F0Full,
                 TC_IC4 = 0x3333333333333333ull, TC_IC5 = 0x2222222222222222ull;
// spread (lo, hi) into two interleaved words: bp[0] even/odd bits of the
// low halves, bp[1] of the high halves (factor.cpp's interleave step)
static inline void tc_spread_pair(u64 lo, u64 hi, u64 *bp) {
    u64 t = lo & TC_IC0, u = lo >> 32, v = hi & TC_IC0, w = hi >> 32;
    u = (u | u << 16) & TC_IC1;  t = (t | t << 16) & TC_IC1;
    w = (w | w << 16) & TC_IC1;  v = (v | v << 16) & TC_IC1;
    u = (u | u << 8)  & TC_IC2;  t = (t | t << 8)  & TC_IC2;
    w = (w | w << 8)  & TC_IC2;  v = (v | v << 8)  & TC_IC2;
    u = (u | u << 4)  & TC_IC3;  t = (t | t << 4)  & TC_IC3;
    w = (w | w << 4)  & TC_IC3;  v = (v | v << 4)  & TC_IC3;
    u = (u | u << 2)  & TC_IC4;  t = (t | t << 2)  & TC_IC4;
    w = (w | w << 2)  & TC_IC4;  v = (v | v << 2)  & TC_IC4;
    u += u & TC_IC5;  t += t & TC_IC5;
    w += w & TC_IC5;  v += v & TC_IC5;
    bp[1] = u | (w << 1);
    bp[0] = t | (v << 1);
}

// The hot loops of the squaring, as separate functions taking plain values
// (not the SqGeom by reference: stores through a u64* could alias its
// size_t members, and the compiler would then reload them every iteration
// and not vectorise -- that cost a factor of 3 against factor_modified's
// fastsqr, which these loops now match).

// dst[j] ^= (src[j] >> dq) | (src[j+1] << dqc), j = count .. 0 (top down),
// where src[count+1] counts as 0.  src and dst are two windows into the
// SAME array and ALWAYS overlap (that is the BLZ cascade), so no
// __restrict__ here.  Returns the last src word read.
static u64 tc_cascade(const u64 *src, u64 *dst, long count, unsigned dq,
                      unsigned dqc) {
    u64 newer = 0;
    for (long j = count; j >= 0; j--) {
        u64 older = newer;
        newer = src[j];
        dst[j] ^= (newer >> dq) | (older << dqc);
    }
    return newer;
}
// dst[j] ^= src[j]: the caller guarantees the two runs do not overlap
static void tc_xor_run(const u64 *__restrict__ src, u64 *__restrict__ dst,
                       long n) {
    for (long j = 0; j < n; j++) dst[j] ^= src[j];
}
// dst[j] ^= (src[j] << hb) | (src[j-1] >> hc), j = 0 .. n-1, 0 < hb < 64;
// reads src[-1 .. n-1], which the caller keeps clear of dst[0 .. n-1]
static void tc_xor_run_shl(const u64 *__restrict__ src, u64 *__restrict__ dst,
                           long n, unsigned hb, unsigned hc) {
    for (long j = 0; j < n; j++) dst[j] ^= (src[j] << hb) | (src[j - 1] >> hc);
}
// dst[j] ^= the 64 bits of src starting at bit sh of src[j], j = 0 .. n-1
// (reads src[0 .. n]); the caller keeps src and dst apart
static void tc_xor_funnel(const u64 *__restrict__ src, u64 *__restrict__ dst,
                          long n, unsigned sh) {
    if (sh == 0) { tc_xor_run(src, dst, n); return; }
    const unsigned shc = 64 - sh;
    for (long j = 0; j < n; j++) dst[j] ^= (src[j] >> sh) | (src[j + 1] << shc);
}
// even exponents from ap[0 .. q5], odd ones from the upper half, spread
// into bp[0 .. 2 q5 + 1].  (hn << 1) << s2 keeps every shift below 64.
static void tc_interleave(const u64 *__restrict__ ap, u64 *__restrict__ bp,
                          long q5, unsigned s1, unsigned s2) {
    for (long j = 0; j <= q5; j++) {
        u64 lo = ap[j], hc = ap[j + q5], hn = ap[j + q5 + 1];
        u64 hi = (hc >> s1) | ((hn << 1) << s2);
        tc_spread_pair(lo, hi, bp + 2 * j);
    }
}

// One fold pass, word-run form (what reduce() uses):
//   dst[0..dw) = src[0, r) ^ (src[r, L) >> r) ^ (src[r, L) >> (r - s))
// REQUIRES the bits of src from L on to be zero and its words readable up
// to (L >> 6) + 2: true of the product and spread buffers (2 nw words
// written, more allocated, the rest never written) and of U (pass one
// writes w1 words, the rest stays zero).  Each term then only runs over
// the destination words its source can reach, and only the first word of
// the third term needs a mask (its source bits below r belong to the first
// term).  tc_fold_pass above is the general (windowed) form; the selftest
// compares the two.
static void tc_fold_pass_fast(const u64 *src, u64 *dst, size_t dw, u64 r,
                              u64 s, u64 L) {
    const size_t rw = (size_t)(r >> 6);
    const unsigned rb = (unsigned)(r & 63);
    const size_t nc = dw < rw ? dw : rw;
    memcpy(dst, src, nc * sizeof(u64));
    if (rw < dw) {
        dst[rw] = rb ? (src[rw] & ((1ULL << rb) - 1)) : 0;
        for (size_t w = rw + 1; w < dw; w++) dst[w] = 0;
    }
    if (L <= r) return;
    size_t n2 = bits_to_words(L - r);              // dst bits [0, L - r)
    if (n2 > dw) n2 = dw;
    tc_xor_funnel(src + rw, dst, (long)n2, rb);
    size_t e3 = bits_to_words(L - r + s);          // dst bits [s, L - r + s)
    if (e3 > dw) e3 = dw;
    const size_t w0 = (size_t)(s >> 6);
    if (w0 < e3) {
        const u64 p0 = r - s + ((u64)w0 << 6);
        const unsigned sh = (unsigned)(p0 & 63);
        const u64 *sp = src + (p0 >> 6);
        u64 v = sh ? ((sp[0] >> sh) | (sp[1] << (64 - sh))) : sp[0];
        if (s & 63) v &= ~((1ULL << (s & 63)) - 1);
        dst[w0] ^= v;
        if (e3 > w0 + 1) {
            const u64 p1 = p0 + 64;
            tc_xor_funnel(src + (p1 >> 6), dst + w0 + 1, (long)(e3 - w0 - 1),
                          (unsigned)(p1 & 63));
        }
    }
}

// b <- a^2 mod T by BLZ (paths 1 and 2).  a: nw+2 words, DESTROYED (the
// reduction runs in place); b: nw+2 words, distinct from a.  Bits >= r of
// a are ignored.
static void tc_square_blz(const SqGeom &g, u64 *ap, u64 *bp) {
    const size_t nw = g.nw, q1 = g.q1;
    ap[q1] &= g.mask1;
    ap[nw] = 0;
    ap[nw + 1] = 0;
    if (g.path == 1) {
        // s odd: the cascade ap[p - delta] ^= ap[p], p = r-1 .. alpha+1
        const long q4 = (long)g.q4, dw = (long)g.deltaw;
        if (g.deltaq == 0) {
            // word-aligned: the loop-carried dependence distance is exactly
            // deltaw words, so runs of <= deltaw words are independent;
            // walk them from the top (factor.cpp's fastsqr_reduce_aligned)
            long hi = (long)q1;
            while (hi > q4) {
                long n = hi - q4;
                if (n > dw) n = dw;
                tc_xor_run(ap + hi - n + 1, ap + hi - n + 1 - dw, n);
                hi -= n;
            }
            ap[q4 - dw] ^= ap[q4] & g.mask2;
        } else {
            const unsigned dq = g.deltaq, dqc = g.deltaqc;
            u64 carry = tc_cascade(ap + q4 + 1, ap + q4 + 1 - dw,
                                   (long)q1 - q4 - 1, dq, dqc);
            u64 nex = ap[q4] & g.mask2;
            ap[q4 - dw] ^= (nex >> dq) | (carry << dqc);
            ap[q4 - 1 - dw] ^= nex << dqc;
        }
    } else {
        // s even (see the header): first the top h bits into [0,h) and
        // [h,2h), then bits [alpha+1, r-h) shifted up by h, top down
        const u64 r = g.r, h = g.h, alpha = g.alpha;
        // Bit x of [0, h) gets bit r-h+x, bit x of [h, 2h) bit r-2h+x: both
        // are word runs read at a fixed bit offset.  The sources lie in
        // [r-h, r), above the targets (2h = s < r-h), so the runs are
        // disjoint; source bits >= r are zero (masked above), so no target
        // bit >= h (first run) or >= 2h (second) changes, and only the
        // second run's first word needs a mask (its bits below h).
        {
            const u64 b1 = r - h;
            tc_xor_funnel(ap + (b1 >> 6), ap, (long)bits_to_words(h),
                          (unsigned)(b1 & 63));
            const long d0 = (long)(h >> 6), d1 = (long)bits_to_words(2 * h);
            const u64 b2 = r - 2 * h + ((u64)d0 << 6);   // source of word d0
            const unsigned sh2 = (unsigned)(b2 & 63);
            const u64 *src2 = ap + (b2 >> 6);
            u64 first = sh2 ? ((src2[0] >> sh2) | (src2[1] << (64 - sh2)))
                            : src2[0];
            if (h & 63) first &= ~((1ULL << (h & 63)) - 1);
            ap[d0] ^= first;
            if (d1 > d0 + 1)
                tc_xor_funnel(src2 + 1, ap + d0 + 1, d1 - d0 - 1, sh2);
        }
        const u64 tlo = alpha + 1 + h;          // first target bit
        if (tlo < r) {
            const long wtop = (long)((r - 1) >> 6), wlo = (long)(tlo >> 6);
            const long hw = (long)(h >> 6);
            const unsigned hb = (unsigned)(h & 63);
            auto edge = [&](long w) {
                u64 wb = (u64)w << 6;
                u64 msk = tc_rmask(wb, tlo, r);
                if (!msk) return;
                u64 v = (wb >= h) ? tc_window(ap, nw, wb - h, 0, ~0ULL)
                                  : (tc_window(ap, nw, 0, 0, ~0ULL)
                                     << (h - wb));
                ap[w] ^= v & msk;
            };
            // the top word and the lowest target word may be partial; the
            // words strictly between are whole targets whose sources
            // (target - h) all lie inside [alpha+1, r-h).  Word w reads
            // words w-hw and w-hw-1 (just w-hw when hb == 0), all below
            // w: so walking down in runs of at most hw (hb == 0) or hw-1
            // words, no run reads a word it writes, and every source is
            // read before it is written (the runs go top down).
            edge(wtop);
            long hi = wtop - 1;
            const long lo = wlo + 1;
            const long run = (hb == 0) ? hw : hw - 1;
            if (run >= 1) {
                while (hi >= lo) {
                    long n = hi - lo + 1;
                    if (n > run) n = run;
                    long w0 = hi - n + 1;
                    if (hb == 0)
                        tc_xor_run(ap + w0 - hw, ap + w0, n);
                    else
                        tc_xor_run_shl(ap + w0 - hw, ap + w0, n, hb, 64 - hb);
                    hi -= n;
                }
            } else {                    // h < 128: plain loop, top down
                for (long w = hi; w >= lo; w--) {
                    u64 a0 = ap[w - hw];
                    u64 a1 = (w - hw - 1 >= 0) ? ap[w - hw - 1] : 0;
                    ap[w] ^= hb ? ((a0 << hb) | (a1 >> (64 - hb))) : a0;
                }
            }
            if (wlo < wtop) edge(wlo);
        }
    }
    const long q5 = (long)g.q5;
    tc_interleave(ap, bp, q5, g.s1, g.s2);
    for (size_t j = 2 * (size_t)q5 + 2; j < nw + 2; j++) bp[j] = 0;
    bp[nw] = 0;
    bp[nw + 1] = 0;
    bp[q1] &= g.mask1;
}

// ---------------------------------------------------------------------------
// aligned word buffers

static u64 *tc_alloc(size_t words) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, (words ? words : 1) * sizeof(u64)) != 0) {
        fprintf(stderr, "out of memory (%zu MB)\n", words * 8 >> 20);
        exit(1);
    }
    memset(p, 0, words * sizeof(u64));
    return (u64 *)p;
}

struct TsCPU : TsDev {
    SqGeom g;
    bool force_generic = false;   // selftest variant 1: generic squaring
    int threads = 1;              // squaring threads (OpenMP builds)
    size_t stride = 0;            // words per residue buffer (>= nw + 2)
    size_t nwc = 0;               // words of the unreduced Horner sum
    u64 *pool = nullptr;          // m table slots + one spare per thread
    std::vector<u64 *> slot, spare;
    u64 *holdbuf = nullptr;       // m residues, slot j at j*stride
    u64 *A = nullptr, *C = nullptr, *T2 = nullptr, *U = nullptr;
    u64 *D = nullptr, *S0 = nullptr, *X = nullptr, *Y = nullptr,
        *Z = nullptr, *W = nullptr;
    EcFold fCu_, seg_[3], fAn_;
    u64 fired_ = 0;
    int nspare_alloc = 0;

    TsCPU() {
        ecf_zero(fCu_); ecf_zero(fAn_);
        for (auto &f : seg_) ecf_zero(f);
    }
    ~TsCPU() override { fini(); }          // fini() may already have run
    const char *kind() const override { return "CPU"; }

    void init(u64 r_) override {
        r = r_;
        nw = bits_to_words(r);
        stride = (nw + 2 + 7) & ~(size_t)7;
        A = tc_alloc(stride);
        C = tc_alloc(stride + 2);
        T2 = tc_alloc(2 * stride + 4);
        U = tc_alloc(stride + bits_to_words(r) + 4);
        D = tc_alloc(stride);
        S0 = tc_alloc(stride);
        X = tc_alloc(stride);
        Y = tc_alloc(stride);
        Z = tc_alloc(stride);
        W = tc_alloc(stride);
    }
    void set_m(int m_) override {
        if (m_ < 2 || m_ > 64) { fprintf(stderr, "need 2 <= m <= 64\n"); exit(1); }
        int nsp = threads > 1 ? threads : 1;
        if (m == m_ && nspare_alloc == nsp) return;
        free(pool);
        free(holdbuf);
        holdbuf = nullptr;
        m = m_;
        nwc = bits_to_words(r + (u64)m - 1);
        pool = tc_alloc((size_t)(m + nsp) * stride);
        slot.assign(m, nullptr);
        for (int j = 0; j < m; j++) slot[j] = pool + (size_t)j * stride;
        spare.assign(nsp, nullptr);
        for (int t = 0; t < nsp; t++)
            spare[t] = pool + (size_t)(m + t) * stride;
        nspare_alloc = nsp;
    }
    void set_s(u64 s_) override {
        if (!(s_ >= 1 && 2 * s_ < r)) {
            fprintf(stderr, "need 1 <= s <= (r-1)/2 (got s=%" PRIu64 ")\n", s_);
            exit(1);
        }
        s = s_;
        g.init(r, s);
        if (force_generic) g.path = 0;
    }
    void fini() override {
        free(pool); free(holdbuf);
        free(A); free(C); free(T2); free(U);
        free(D); free(S0); free(X); free(Y); free(Z); free(W);
        pool = holdbuf = A = C = T2 = U = D = S0 = X = Y = Z = W = nullptr;
        slot.clear(); spare.clear();
        m = 0;
        nspare_alloc = 0;
    }

    // ---- arithmetic ----
    // t (bits < tbits <= 2r-1, tnw words) mod T -> out (nw words)
    // t (2nw words, bits beyond its degree zero, see tc_fold_pass_fast)
    // mod T -> out (nw words)
    void reduce(const u64 *t, u64 *out) {
        size_t w1 = bits_to_words(r + s - 1);
        tc_fold_pass_fast(t, U, w1, r, s, 2 * r - 1);
        // pass two reads U a little past w1: clear what an earlier, larger
        // s may have left there (tuning runs at s = (r-1)/2)
        U[w1] = U[w1 + 1] = U[w1 + 2] = 0;
        tc_fold_pass_fast(U, out, nw, r, s, r + s - 1);
    }
    // out <- in^2 mod T; `in` may be destroyed; both stride words
    void sq(u64 *in, u64 *out, u64 *scratch2) {
        if (g.path) { tc_square_blz(g, in, out); return; }
        // generic: spread into 2nw words, then the two fold passes
        in[nw - 1] &= ((r & 63) ? ((1ULL << (r & 63)) - 1) : ~0ULL);
        for (size_t i = 0; i < nw; i++)
            tc_spread_pair(in[i], 0, scratch2 + 2 * i);
        reduce(scratch2, out);
        out[nw] = out[nw + 1] = 0;
    }
    // dst ^= src * x mod T (k_mulx_xor on the host)
    void mulx_xor_exact(u64 *dst, const u64 *src) {
        // dst ^= ((src << 1) without the x^r term) ^ wrap*(x^s + 1)
        u64 wrap = (src[(r - 1) >> 6] >> ((r - 1) & 63)) & 1;
        u64 carry = 0;
        for (size_t w = 0; w < nw; w++) {
            u64 v = src[w];
            u64 o = (v << 1) | carry;
            carry = v >> 63;
            if (w == (r >> 6)) o &= ~(1ULL << (r & 63));
            dst[w] ^= o;
        }
        if (wrap) {
            dst[0] ^= 1;
            dst[s >> 6] ^= 1ULL << (s & 63);
        }
    }
    void square_all_step() override {
        if (threads > 1 && g.path) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int j = 0; j < m; j++) {
                int t = omp_get_thread_num();
                sq(slot[j], spare[t], nullptr);
                std::swap(slot[j], spare[t]);
            }
            return;
#endif
        }
        for (int j = 0; j < m; j++) {
            sq(slot[j], spare[0], T2);
            std::swap(slot[j], spare[0]);
        }
    }
    // n squarings of each slot in turn (see TsDev::square_steps); with
    // --threads, the slots are shared out between the threads
    void square_steps(u64 n, void (*mid)(TsDev &)) override {
        if (!n) return;
        if (threads > 1 && g.path) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
            for (int j = 0; j < m; j++) {
                int t = omp_get_thread_num();
                for (u64 i = 0; i < n; i++) {
                    sq(slot[j], spare[t], nullptr);
                    std::swap(slot[j], spare[t]);
                }
            }
            if (mid) mid(*this);
            return;
#endif
        }
        for (int j = 0; j < m; j++) {
            for (u64 i = 0; i < n; i++) {
                sq(slot[j], spare[0], T2);
                std::swap(slot[j], spare[0]);
            }
            if (mid && j == (m - 1) / 2) mid(*this);
        }
    }
    // C = x^m + sum_j h[j] x^(m-1-j), reduced; records the tail's quotient
    void horner_C() override {
        for (size_t w = 0; w < nwc; w++) {
            u64 acc = 0;
            for (int j = 0; j < m; j++) {
                const u64 *hj = slot[j];
                unsigned sh = (unsigned)(m - 1 - j);
                u64 a = (w < nw) ? hj[w] : 0;
                u64 b = (w >= 1 && w - 1 < nw) ? hj[w - 1] : 0;
                acc ^= sh ? ((a << sh) | (b >> (64 - sh))) : a;
            }
            C[w] = acc;
        }
        for (size_t w = nwc; w < stride + 2; w++) C[w] = 0;
        u64 fired = 0;
        for (int e = 0; e <= m - 2; e++) {
            u64 p = r + (u64)e;
            if ((C[p >> 6] >> (p & 63)) & 1) {
                fired |= 1ULL << e;
                C[p >> 6] ^= 1ULL << (p & 63);
                u64 t1 = s + (u64)e;
                C[t1 >> 6] ^= 1ULL << (t1 & 63);
                C[(u64)e >> 6] ^= 1ULL << (e & 63);
            }
        }
        C[(u64)m >> 6] ^= 1ULL << (m & 63);
        if (ec_on) fired_ = fired;
    }
    void full_product(const u64 *a, const u64 *b, u64 *out) {
#ifdef HAVE_GF2X
        gf2x_mul((unsigned long *)out, (const unsigned long *)a,
                 (long)nw, (const unsigned long *)b, (long)nw);
#elif defined(HAVE_NTL)
        NTL::GF2X ga, gb, gc;
        ga.SetLength(0); gb.SetLength(0);
        {
            std::vector<unsigned char> bytes(nw * 8);
            memcpy(bytes.data(), a, nw * 8);
            NTL::GF2XFromBytes(ga, bytes.data(), (long)bytes.size());
            memcpy(bytes.data(), b, nw * 8);
            NTL::GF2XFromBytes(gb, bytes.data(), (long)bytes.size());
        }
        NTL::mul(gc, ga, gb);
        std::vector<unsigned char> ob(2 * nw * 8, 0);
        long nb = NTL::NumBytes(gc);
        if (nb > (long)ob.size()) nb = (long)ob.size();
        NTL::BytesFromGF2X(ob.data(), gc, nb);
        memcpy(out, ob.data(), 2 * nw * 8);
#else
        std::vector<u64> o(mul_out_words(r, r));
        gf2x_naive_mul(a, r, b, r, o.data());
        memcpy(out, o.data(), 2 * nw * 8);
#endif
    }
    void mul_A_by_C() override {
        full_product(A, C, T2);
        if (ec_inject_now(ECI_T)) inject_buf(T2, 2 * r - 1, ECI_T);
        if (ec_on) {                    // three segment folds, one pass
            double t0 = ec_tic();
            fold3(T2, 2 * nw, r, 2 * r - s, seg_);
            ec_toc(t0, g_ecs.t_block);
        }
        reduce(T2, A);
        if (ec_inject_now(ECI_RED)) inject_buf(A, r, ECI_RED);
        if (ec_on) {
            double t0 = ec_tic();
            ecf_fold_host(fAn_, A, nw);
            ec_toc(t0, g_ecs.t_block);
        }
        if (ec_inject_now(ECI_A)) inject_buf(A, r, ECI_A);
    }
    void reset_A() override {
        memset(A, 0, stride * sizeof(u64));
        A[0] = 1;
        ecf_one(ecA);
    }
    void init_h_at(u64 kstart) override {
        for (int j = 0; j < m; j++) memset(slot[j], 0, stride * sizeof(u64));
        for (int j = 1; j <= m; j++) {
            for (int i = j - 1; i >= 0; i--) {
                if (i > 0) {
                    sq(slot[i - 1], spare[0], T2);
                    std::swap(slot[i - 1], spare[0]);
                    mulx_xor_exact(slot[i], slot[i - 1]);
                } else {
                    slot[0][0] ^= 2ULL;             // h[0] ^= x
                }
            }
        }
        if (ec_inject_now(ECI_INIT)) inject_h(ECI_INIT);
        if (g_eci.site == ECI_INIT) {   // test mode: one opportunity per step
            for (u64 t = 0; t < kstart; t++) {
                square_all_step();
                if (ec_inject_now(ECI_INIT)) inject_h(ECI_INIT);
            }
        } else {
            square_steps(kstart, nullptr);
        }
    }

    // ---- error checking ----
    // three folds of src: bits [0,c1), [c1,c2), [c2,inf), each in place
    static void fold3(const u64 *src, size_t n, u64 c1, u64 c2, EcFold f[3]) {
        for (int g3 = 0; g3 < 3; g3++) ecf_zero(f[g3]);
        size_t w1 = (size_t)(c1 >> 6), w2 = (size_t)(c2 >> 6);
        for (size_t w = 0; w < n; w++) {
            u64 v = src[w];
            int j = (int)(w % EC_NF);
            if (w < w1) { f[0].w[j] ^= v; continue; }
            if (w > w2) { f[2].w[j] ^= v; continue; }
            u64 wb = (u64)w << 6;
            u64 m0 = tc_rmask(wb, 0, c1), m1 = tc_rmask(wb, c1, c2);
            f[0].w[j] ^= v & m0;
            f[1].w[j] ^= v & m1;
            f[2].w[j] ^= v & ~(m0 | m1);
        }
    }
    void ec_setup(bool on) override {
        ec_on = on;
        ecf_one(ecA);
    }
    // D ^= S (digest) and the fold of the unreduced Horner sum, in one pass
    // over the table, independently of horner_C
    void ec_block_pre() override {
        double t0 = ec_tic();
        ecf_zero(fCu_);
        for (size_t w = 0; w < nwc; w++) {
            u64 x = 0, cu = 0;
            for (int j = 0; j < m; j++) {
                const u64 *hj = slot[j];
                unsigned sh = (unsigned)(m - 1 - j);
                u64 a = (w < nw) ? hj[w] : 0;
                u64 b = (w >= 1 && w - 1 < nw) ? hj[w - 1] : 0;
                x ^= a;
                cu ^= sh ? ((a << sh) | (b >> (64 - sh))) : a;
            }
            if (w < nw) D[w] ^= x;
            fCu_.w[w % EC_NF] ^= cu;
        }
        ec_toc(t0, g_ecs.t_block);
    }
    void ec_read_block(EcFold &fCu, EcFold seg[3], EcFold &fAn,
                       u64 &fired) override {
        fCu = fCu_;
        for (int i = 0; i < 3; i++) seg[i] = seg_[i];
        fAn = fAn_;
        fired = fired_;
    }
    void digest(u64 *out) {
        for (size_t w = 0; w < nw; w++) {
            u64 x = 0;
            for (int j = 0; j < m; j++) x ^= slot[j][w];
            out[w] = x;
        }
    }
    void ec_chain_reset() override {
        digest(S0);
        memset(D, 0, stride * sizeof(u64));
        ec_w = 0;
    }
    bool ec_window_check() override {
        double t0 = ec_tic();
        digest(X);
        const u64 *sig = D;
        if (ec_w > 0) {
            memcpy(Y, D, stride * sizeof(u64));
            for (int t = 0; t < m; t++) {
                sq(Y, Z, T2);
                std::swap(Y, Z);
            }
            sig = Y;
        }
        u64 bad = 0;
        for (size_t w = 0; w < nw; w++) bad |= sig[w] ^ D[w] ^ S0[w] ^ X[w];
        std::swap(S0, X);
        memset(D, 0, stride * sizeof(u64));
        ec_w = 0;
        g_ecs.windows++;
        ec_toc(t0, g_ecs.t_window);
        return bad == 0;
    }
    bool ec_verify_table_at(u64 k) override {
        double t0 = ec_tic();
        memset(Y, 0, stride * sizeof(u64));
        Y[0] = 3;                                       // 1 + x
        for (u64 t = 0; t < k; t++) { sq(Y, Z, T2); std::swap(Y, Z); }
        memcpy(X, Y, stride * sizeof(u64));             // 1 + y_k
        for (int t = 0; t < m; t++) { sq(Y, Z, T2); std::swap(Y, Z); }
        ec_chain_reset();                               // S0 = S_k
        memcpy(D, S0, stride * sizeof(u64));
        D[0] ^= 1;                                      // 1 + S_k
        full_product(X, D, T2);
        reduce(T2, W);
        u64 bad = 0;
        for (size_t w = 0; w < nw; w++) bad |= W[w] ^ Y[w];
        memset(D, 0, stride * sizeof(u64));
        g_ecs.n_init++;
        ec_toc(t0, g_ecs.t_init);
        return bad == 0;
    }
    // fold of m residues laid out as one m*nw-word polynomial
    void fold_table(u64 *const *base_or_null, const u64 *contig, EcFold &f) {
        ecf_zero(f);
        for (int j = 0; j < m; j++) {
            const u64 *p = base_or_null ? base_or_null[j]
                                        : contig + (size_t)j * stride;
            size_t idx = ((size_t)j * nw) % EC_NF;
            for (size_t w = 0; w < nw; w++) {
                f.w[idx] ^= p[w];
                if (++idx == EC_NF) idx = 0;
            }
        }
    }
    void table_fold(EcFold &f) override { fold_table(slot.data(), nullptr, f); }
    void hold_fold(EcFold &f) override { fold_table(nullptr, holdbuf, f); }

    // ---- copies ----
    void h_to_host(std::vector<u64> &out) override {
        out.resize((size_t)m * nw);
        for (int j = 0; j < m; j++)
            memcpy(&out[(size_t)j * nw], slot[j], nw * sizeof(u64));
    }
    void h_from_host(const std::vector<u64> &in) override {
        for (int j = 0; j < m; j++) {
            memcpy(slot[j], &in[(size_t)j * nw], nw * sizeof(u64));
            for (size_t w = nw; w < stride; w++) slot[j][w] = 0;
        }
    }
    void A_to_host(std::vector<u64> &out) override {
        out.assign(A, A + nw);
    }
    void A_from_host_raw(const std::vector<u64> &in) override {
        memset(A, 0, stride * sizeof(u64));
        memcpy(A, in.data(), (in.size() < nw ? in.size() : nw) * sizeof(u64));
    }
    void need_hold() override {
        if (!holdbuf) holdbuf = tc_alloc((size_t)m * stride);
    }
    void hold_save() override {
        for (int j = 0; j < m; j++)
            memcpy(holdbuf + (size_t)j * stride, slot[j], stride * sizeof(u64));
    }
    void hold_restore() override {
        for (int j = 0; j < m; j++)
            memcpy(slot[j], holdbuf + (size_t)j * stride, stride * sizeof(u64));
    }

    // ---- fault injection ----
    void inject_buf(u64 *buf, u64 nbits, int site) {
        u64 bit = ec_inject_rnd(nbits);
        buf[bit >> 6] ^= 1ULL << (bit & 63);
        ec_inject_note(site, bit);
    }
    void inject_C(int site) override { inject_buf(C, r, site); }
    void inject_h(int site) override {
        int j = (int)ec_inject_rnd((u64)m);
        inject_buf(slot[j], r, site);
    }
    void inject_hold(int site) override {
        int j = (int)ec_inject_rnd((u64)m);
        inject_buf(holdbuf + (size_t)j * stride, r, site);
    }

    // ---- measurement ----
    // the cost of one squaring as the scan does them (square_steps: one
    // slot at a time, the slots shared between the squaring threads)
    double time_square(int reps) override {
        for (int j = 0; j < m; j++) {
            std::vector<u64> v = st_res(r);
            memcpy(slot[j], v.data(), nw * sizeof(u64));
        }
        square_steps(1, nullptr);
        u64 steps = reps < 1 ? 1 : (u64)reps;
        double t0 = trin_now_s();
        square_steps(steps, nullptr);
        return (trin_now_s() - t0) / ((double)steps * m);
    }
    double time_mul(int reps) override {
        std::vector<u64> a = st_res(r), c = st_res(r);
        memcpy(A, a.data(), nw * 8);
        memcpy(C, c.data(), nw * 8);
        const bool sv = ec_on;
        ec_on = false;
        mul_A_by_C();
        double t0 = trin_now_s();
        for (int i = 0; i < reps; i++) mul_A_by_C();
        ec_on = sv;
        return (trin_now_s() - t0) / reps;
    }

    // ---- selftest access ----
    void C_to_host(std::vector<u64> &out) override { out.assign(C, C + nw); }
    void ecD_to_host(std::vector<u64> &out) override { out.assign(D, D + nw); }
    void ecD_zero() override { memset(D, 0, stride * sizeof(u64)); }
};

// ---------------------------------------------------------------------------
// CPU kernel suite: the squaring paths, the reduction and the product
// against the bit-level oracles (run by both builds' --selftest)
static int cpu_kernel_suites() {
    int bad = 0;
    {
        long ncase = 0, nbad = 0;
        int npath[3] = {0, 0, 0};
        const u64 rs[] = {257, 263, 607, 1279, 2203, 3217, 4097, 4159, 4229,
                          4423, 9689, 19937};
        for (u64 rr : rs) {
            std::vector<u64> ss;
            for (u64 sv = 1; sv <= 70 && 2 * sv < rr; sv++) ss.push_back(sv);
            for (u64 sv : {126, 127, 128, 129, 130, 131, 190, 191, 192, 255,
                           256, 257})
                if (2 * sv < rr) ss.push_back(sv);
            for (u64 sv = (rr - 1) / 2; sv > 0 && sv + 40 > (rr - 1) / 2; sv--)
                ss.push_back(sv);
            for (int k = 0; k < 24; k++) ss.push_back(1 + st_rnd() % ((rr - 1) / 2));
            for (u64 sv : ss) {
                for (int variant = 0; variant < 2; variant++) {
                    TsCPU G;
                    G.init(rr);
                    G.set_m(2);
                    G.force_generic = (variant == 1);
                    G.set_s(sv);
                    if (variant == 0) npath[G.g.path]++;
                    else if (sv % 7) { G.fini(); continue; }   // sample
                    for (int t = 0; t < 2; t++) {
                        std::vector<u64> a = st_res(rr), hr = a;
                        if (t == 1) {
                            for (auto &w : a) w = ~0ULL;
                            if (rr & 63) a.back() &= (1ULL << (rr & 63)) - 1;
                            hr = a;
                        }
                        u64 *in = G.slot[0], *out = G.slot[1];
                        memset(in, 0, G.stride * 8);
                        memcpy(in, a.data(), G.nw * 8);
                        G.sq(in, out, G.T2);
                        cpu_square_mod(hr, rr, sv);
                        ncase++;
                        if (memcmp(out, hr.data(), G.nw * 8) ||
                            out[G.nw] || out[G.nw + 1]) {
                            if (++nbad < 6)
                                printf("  CPU squaring mismatch r=%" PRIu64
                                       " s=%" PRIu64 " path %d\n", rr, sv,
                                       G.g.path);
                        }
                    }
                    G.fini();
                }
            }
        }
        printf("CPU squaring: BLZ odd/even + generic vs host (%ld) %s\n",
               ncase, nbad ? "FAIL" : "PASS");
        if (nbad) bad++;
        (void)npath;
    }
    {
        // one device per r, the s from large to small: what a run does
        // after its tuning (at s = (r-1)/2), so stale scratch would show
        bool ok = true;
        const u64 rs[] = {607, 4423, 9689};
        for (u64 rr : rs) {
            TsCPU G;
            G.init(rr);
            G.set_m(3);
            for (u64 sv : {(rr - 1) / 2, rr / 3, (u64)129, (u64)101,
                           (u64)64, (u64)2, (u64)1}) {
                G.set_s(sv);
                std::vector<u64> a = st_res(rr), c = st_res(rr);
                std::vector<u64> pr(mul_out_words(rr, rr));
                gf2x_naive_mul(a.data(), rr, c.data(), rr, pr.data());
                cpu_mod_trinomial(pr, 2 * rr - 1, rr, sv);
                memcpy(G.A, a.data(), G.nw * 8);
                memcpy(G.C, c.data(), G.nw * 8);
                G.mul_A_by_C();
                if (memcmp(G.A, pr.data(), G.nw * 8)) ok = false;
                // mul by x
                std::vector<u64> d0 = st_res(rr), hx(G.nw + 1, 0), dd = d0;
                for (size_t i = 0; i < G.nw; i++) {
                    hx[i] ^= a[i] << 1;
                    hx[i + 1] ^= a[i] >> 63;
                }
                if ((hx[rr >> 6] >> (rr & 63)) & 1) {
                    hx[rr >> 6] ^= 1ULL << (rr & 63);
                    hx[sv >> 6] ^= 1ULL << (sv & 63);
                    hx[0] ^= 1;
                }
                for (size_t i = 0; i < G.nw; i++) dd[i] ^= hx[i];
                memcpy(G.W, d0.data(), G.nw * 8);
                G.mulx_xor_exact(G.W, a.data());
                if (memcmp(G.W, dd.data(), G.nw * 8)) ok = false;
                // and a squaring on the same device (the generic path
                // shares the reduction's scratch)
                std::vector<u64> q = st_res(rr), hq = q;
                memcpy(G.slot[0], q.data(), G.nw * 8);
                G.sq(G.slot[0], G.slot[1], G.T2);
                cpu_square_mod(hq, rr, sv);
                if (memcmp(G.slot[1], hq.data(), G.nw * 8)) ok = false;
            }
            G.fini();
        }
        printf("CPU product mod T and mul-by-x vs host             %s\n",
               ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }
    {   // the word-run fold pass against the general (windowed) one
        bool ok = true;
        long ncase = 0;
        for (u64 rr : {(u64)257, (u64)1279, (u64)4423, (u64)19937}) {
            const size_t nw2 = bits_to_words(rr);
            std::vector<u64> ss = {1, 2, 63, 64, 65, 127, 128, 129, 191, 192,
                                   rr / 3, (rr - 1) / 2};
            for (int k = 0; k < 8; k++) ss.push_back(1 + st_rnd() % ((rr - 1) / 2));
            for (u64 sv : ss) {
                if (!(sv >= 1 && 2 * sv < rr)) continue;
                std::vector<u64> t(2 * nw2 + 8, 0), u1(2 * nw2 + 8, 0),
                    u2(2 * nw2 + 8, 0), o1(nw2 + 2, 0), o2(nw2 + 2, 0);
                for (size_t w = 0; w < 2 * nw2; w++) t[w] = st_rnd();
                const u64 L = 2 * rr - 1;              // product length
                t[(L - 1) >> 6] &= ~0ULL >> (63 - ((L - 1) & 63));
                for (size_t w = ((L - 1) >> 6) + 1; w < t.size(); w++) t[w] = 0;
                const size_t w1 = bits_to_words(rr + sv - 1);
                tc_fold_pass(t.data(), t.size(), u1.data(), w1, rr, sv, L);
                tc_fold_pass_fast(t.data(), u2.data(), w1, rr, sv, L);
                tc_fold_pass(u1.data(), w1, o1.data(), nw2, rr, sv, rr + sv - 1);
                tc_fold_pass_fast(u2.data(), o2.data(), nw2, rr, sv,
                                  rr + sv - 1);
                if (u1 != u2 || o1 != o2) ok = false;
                ncase++;
            }
        }
        printf("CPU fold pass: word runs vs windowed (%ld)         %s\n", ncase,
               ok ? "PASS" : "FAIL");
        if (!ok) bad++;
    }
    return bad;
}

#endif // TS_CPU_H

/* The original version (2019) of this file is part of the gf2x library.

   Copyright 2007, 2008, 2009, 2013, 2016, 2019
   Richard Brent, Pierrick Gaudry, Emmanuel Thome', Paul Zimmermann

   Copyright (C) 2026  Gord Palameta
   (actual author: Claude AI)

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 3 of the License, or (at your
   option) any later version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
   more details.

   You should have received a copy of the GNU General Public License along
   with this program; see the file COPYING.  If not, write to the Free
   Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
   02111-1307, USA.
*/

/*
Versions of this modified file (not official gf2x versions):
  3.00  Gerbicz-style error checking (2026)
  3.01  merged factor_modified 2.02 (2026-08-30): faster squaring without
        BMI2, for older machines like Zen 1 and Zen 2 (constants depending
        only on r computed once, vectorisable reduce loop, simpler
        interleave); the -Z option; no segmentation fault in
        get_certificate when the trinomial is primitive (also fixed in 2.02)
*/

/* Finds smallest irreducible factor of trinomial over GF(2) */

#define GF2X_FACTOR_VERSION 3.01

#include <stdio.h>
#include <stdlib.h>
#include <limits.h> /* for CHAR_BIT */
#include <NTL/GF2XFactoring.h>
#include <NTL/ZZXFactoring.h>
#include <NTL/version.h>
#include <NTL/GF2XVec.h>
#include <signal.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <sys/time.h>
#ifdef GF2X_HAVE_BMI2_SUPPORT
#include <x86intrin.h> /* for _pdep_u64 */
#endif

#include "halfgcd.hpp"

NTL_CLIENT

/* ==================================================================== *
 *  ERROR CHECKING  (see errorcheck_handoff.md)
 *
 *  Three independent mechanisms:
 *    (1) a Frobenius-linear Gerbicz check on the h-table evolution,
 *    (2) cyclic-fold (phi_n) checks on the products and on the Horner
 *        construction of C[z], plus an independent re-reduction,
 *    (3) optional repeated GCD.
 *  On any failure the current interval is recomputed from a checkpoint.
 * ==================================================================== */

static int ec_enabled   = 1;    /* ON by default since round 5; -noec */
static int ec_window    = 8;    /* -ecw w           */
static int ec_gcdrep    = 0;    /* -ecg             */
static long ec_maxretry = 5;
static volatile int ec_fail = 0;      /* set by any failing check */
static long ec_detections = 0;
static int  ec_timing   = 0;    /* -ect: per-category cost accounting  */
static int  ec_initchk  = 1;    /* -eci0 to disable the init_h check   */
static int  ec_fault_init = 0;  /* -ecfaultinit: corrupt the first h-table
                                   right after it is built (testing only) */

/* ---- fault injection, for validating the checker itself (-ecfault N).
   Flips one coefficient at the N-th injection opportunity.  Fires once, so a
   rollback retry re-does the work cleanly and the run should still produce
   the correct answer -- which is the property we actually want to prove. ---- */
static long ec_fault_at = 0, ec_fault_ctr = 0, ec_fault_done = 0;

static void ec_maybe_inject (NTL::GF2X& a, const char *site)
{
  if (!ec_fault_at || ec_fault_done) return;
  if (++ec_fault_ctr != ec_fault_at) return;
  long d = deg (a);
  if (d < 1) return;
  long pos = d / 2;
  SetCoeff (a, pos, !(rep (coeff (a, pos))));
  ec_fault_done = 1;
  fprintf (stderr, "EC INJECT: flipped bit %ld of a polynomial at site '%s'"
           " (opportunity %ld)\n", pos, site, ec_fault_ctr);
  fflush (stderr);
}

/* ---- cost accounting.  Round 1 shipped with none of this, which is why a
   pathological check could burn 45 minutes without saying anything. ---- */

enum { ECT_MULFOLD, ECT_DIVIS, ECT_COPY, ECT_HORNER, ECT_DIGEST,
       ECT_SIGMA, ECT_INIT, ECT_GCD, ECT_N };
static const char *ec_tname[ECT_N] = {
  "fold check on product", "reduction divisibility", "copy t for check",
  "horner fold shadow", "gerbicz digest xor", "gerbicz sigma(D)",
  "init_h closed-form check", "gcd repeat" };
static double ec_tsec[ECT_N];
static long   ec_tcnt[ECT_N];

static inline double ec_now (void)
{
  struct timeval tv; gettimeofday (&tv, NULL);
  return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}
#define EC_TIC(slot) double _ectic = ec_timing ? ec_now () : 0.0
#define EC_TOC(slot) do { if (ec_timing) {                       \
      ec_tsec[slot] += ec_now () - _ectic; ec_tcnt[slot]++; } } while (0)

/* Every check reports the instant it trips, with a category and a location.
   Silence is the one failure mode we cannot afford twice. */
#define EC_FAIL(what, ...)                                              \
  do {                                                                  \
    ec_fail = 1;                                                        \
    fprintf (stderr, "EC FAIL [%s] " what "\n", __func__, ##__VA_ARGS__); \
    fflush (stderr);                                                    \
  } while (0)

/* ---- cyclic fold checksum  phi_n : GF(2)[x] -> GF(2)[x]/(x^n + 1) ----
   n = ECFOLDW * NTL_BITS_PER_LONG bits, so the fold is a pure word XOR.  */

#include "ec_words.h"                   /* fold / reverse / divisibility */

#if NTL_BITS_PER_LONG != 64
#error "The EC layer assumes 64-bit words (ecword == _ntl_ulong)."
#endif

/* 61 words = 3904 bits.  Deliberately NOT a power of two (it was 64 until
   round 4): with n = 2^12 an error that repeats with a power-of-two period
   -- a stuck bus lane, a constant pattern across a power-of-two FFT
   sub-block -- folds onto itself an even number of times and cancels.
   61 is prime, so such patterns spread over all 61 words instead. */
#define ECFOLDW 61

typedef struct { _ntl_ulong w[ECFOLDW]; } ecfold;

static void ec_fold (ecfold *f, const GF2X& a)
{
  memset (f->w, 0, sizeof f->w);
  long n = a.xrep.length ();
  const _ntl_ulong *p = a.xrep.elts ();
  long i = 0, j;
  for (; i + ECFOLDW <= n; i += ECFOLDW)
    for (j = 0; j < ECFOLDW; j++)
      f->w[j] ^= p[i + j];
  for (j = 0; i + j < n; j++)
    f->w[j] ^= p[i + j];
}

static int ec_fold_eq (const ecfold *a, const ecfold *b)
{
  return memcmp (a->w, b->w, sizeof a->w) == 0;
}

static void ec_fold_add (ecfold *c, const ecfold *a, const ecfold *b)
{
  for (long j = 0; j < ECFOLDW; j++) c->w[j] = a->w[j] ^ b->w[j];
}

/* c = x * a  in GF(2)[x]/(x^n + 1) : a rotate by one bit */
static void ec_fold_shl1 (ecfold *c, const ecfold *a)
{
  const int TOP = NTL_BITS_PER_LONG - 1;
  _ntl_ulong carry = a->w[ECFOLDW - 1] >> TOP;
  for (long j = ECFOLDW - 1; j > 0; j--)
    c->w[j] = (a->w[j] << 1) | (a->w[j - 1] >> TOP);
  c->w[0] = (a->w[0] << 1) | carry;
}

static void ec_fold_to_gf2x (GF2X& g, const ecfold *f)
{
  unsigned char buf[ECFOLDW * (NTL_BITS_PER_LONG / 8)];
  for (long j = 0; j < ECFOLDW; j++)
    for (long b = 0; b < NTL_BITS_PER_LONG / 8; b++)
      buf[j * (NTL_BITS_PER_LONG / 8) + b] =
        (unsigned char) ((f->w[j] >> (8 * b)) & 0xff);
  GF2XFromBytes (g, buf, (long) sizeof buf);
}

/* c = a * b  in GF(2)[x]/(x^n + 1).  Tiny (4096-bit) -- cost is noise. */
static void ec_fold_mul (ecfold *c, const ecfold *a, const ecfold *b)
{
  GF2X ga, gb, gc;
  ec_fold_to_gf2x (ga, a);
  ec_fold_to_gf2x (gb, b);
  mul (gc, ga, gb);
  ec_fold (c, gc);
}

/* ================= reduction check: divisibility by T =================
 *
 * ROUND 1 BUG, and why this file no longer contains a second reduction.
 *
 * The old ec_reduce_ref reduced by repeated block folding:
 *     w = w_H x^r + w_L  ->  w_L + w_H x^s + w_H
 * whose degree drops by (r - s) per pass.  That is 2 passes when 2s <= r,
 * and r/(r-s) passes otherwise.  factor *deliberately* flips to s > r/2 to
 * enable fastsqr, so the production case r=136279841, s=136279253 needed
 * 231769 passes -- about 54 minutes for a single fastmulmod.  gecsim never
 * saw it because r=1279, s=345 is the 2-pass regime.
 *
 * The replacement does not re-reduce at all.  It tests divisibility:
 *     c == t mod T   <=>   T | (t + c)   and   deg c < r.
 * When s > r/2 the test is moved to the reciprocal, where the exponent is
 * small again:
 *     T | w   <=>   T* | rev_D(w),     T* = x^r + x^(r-s) + 1,
 * exact in both directions because rev is graded-multiplicative and
 * rev_r(T*) = T.  Reversal is one linear pass; the reduction mod T* is then
 * always 2 passes.  Measured 37 ms at r=136279841 (see ecbench).
 *
 * This is also *stronger* than the old check.  An error e in c is missed
 * only if T | e, and deg e < r = deg T forces e = 0.  So every error in the
 * reduction step is caught, deterministically -- not just single bit flips.
 */

/* thread_local: accumulate() has an OpenMP pragma over the product tree, so
   with -t > 1 several threads are inside fastmulmod at once and would
   otherwise share these scratch buffers. */
static thread_local ecword *ec_wbuf = 0, *ec_wsc = 0, *ec_wsc2 = 0;
static thread_local long    ec_nw2  = 0;

static void ec_alloc (long r)
{
  long need = (2 * r + ECW_BITS - 1) / ECW_BITS + 1;
  if (ec_nw2 >= need) return;
  free (ec_wbuf); free (ec_wsc); free (ec_wsc2);
  ec_nw2  = need;
  ec_wbuf = (ecword *) malloc ((size_t) need * sizeof (ecword));
  ec_wsc  = (ecword *) malloc ((size_t) need * sizeof (ecword));
  ec_wsc2 = (ecword *) malloc ((size_t) need * sizeof (ecword));
  if (!ec_wbuf || !ec_wsc || !ec_wsc2)
    { fprintf (stderr, "EC: out of memory\n"); exit (1); }
}

/* dst = a, zero-padded to nw words */
static void ec_to_words (ecword *dst, long nw, const GF2X& a)
{
  long n = a.xrep.length ();
  if (n > nw) n = nw;
  /* memcpy the body and zero only the tail.  Doing memset(nw) first and then
     copying writes the whole buffer twice; t is nearly full length here, so
     that doubled the cost of this step. */
  if (n > 0) memcpy (dst, a.xrep.elts (), (size_t) n * sizeof (ecword));
  if (n < nw) memset (dst + n, 0, (size_t) (nw - n) * sizeof (ecword));
}

/* dst ^= a */
static void ec_xor_words (ecword *dst, long nw, const GF2X& a)
{
  long n = a.xrep.length ();
  if (n > nw) n = nw;
  const _ntl_ulong *p = a.xrep.elts ();
  for (long j = 0; j < n; j++) dst[j] ^= (ecword) p[j];
}

double
WctTime (void)
{
    struct timeval tv[1];
    gettimeofday (tv, NULL);
    return (double)tv->tv_sec + (double)tv->tv_usec*1.0e-6;
}

/* Usage: factor [options] -s0 s0 -s1 s1 r [see usage() function for options]

   Check every trinomial x^r+x^s+1 for s0 <= s < s1, with 0 < s0 and s1 <= r,
   and "s k px..y" is output if it has a factor of degree k
   (the factor is x..y in hex encoding);
   otherwise "s irreducible" (or "s primitive") is output.
   If the "-z" option is used then "s u" may be output (in this case
   we need to check the trinomial with another program, e.g. irred).
*/

/* Squaring constants (factor_modified 2.02): those depending only on r
   are computed once per run, those depending on s once per call. */
struct SqrConstR {
    unsigned long alpha;
    unsigned long q1;
    unsigned long q4;
    unsigned long q5;      // == q41 in irred: (alpha+1)/64
    unsigned long s1;      // (alpha+1) mod 64
    unsigned long s2;      // 63 - s1
    unsigned long smax;    // ceil(r/64)
    _ntl_ulong   mask1;
};

static SqrConstR make_sqr_const_r(unsigned long r)
{
    SqrConstR c;
    const unsigned long wlen = 64, wlenm = 63, wd = 6;
    c.smax  = (r + wlen - 1) >> wd;
    c.alpha = r >> 1;
    c.q1    = (r - 1) >> wd;           // == smax - 1
    c.q4    = c.alpha >> wd;
    c.q5    = (c.alpha + 1) >> wd;
    c.s1    = (c.alpha + 1) & wlenm;
    c.s2    = wlenm - c.s1;
    c.mask1 = UINT64_MAX >> (wlenm - ((r - 1) & wlenm));
    return c;
}

struct SqrConstS {
    unsigned long delta;
    unsigned long deltaw;
    unsigned long deltaq;
    unsigned long deltaqc;
    _ntl_ulong   mask2;
};

static SqrConstS make_sqr_const_s(unsigned long r, unsigned long s,
        const SqrConstR& cr)
{
    SqrConstS c;
    const unsigned long wlenm = 63, wd = 6, wlen = 64;
    c.delta   = (r - s) >> 1;
    c.deltaw  = c.delta >> wd;
    c.deltaq  = c.delta & wlenm;
    c.deltaqc = wlen - c.deltaq;
    c.mask2   = (~(_ntl_ulong)1) << (cr.alpha & wlenm);
    return c;
}

#define BITS_PER_LONG2 (2*NTL_BITS_PER_LONG)

// Reduce helper: barrel-shift XOR, reads from ap[src..], writes to ap[dst..].
//
// WARNING: src and dst are two windows into the SAME array ap, and they
// ALWAYS overlap.  src spans ap[q4+1 .. q1] and dst spans that window shifted
// down by deltaw, so they overlap in ap[q4+1 .. q1-deltaw] whenever
// deltaw <= q1-q4-1, which holds for every s permitted here.  The overlap is
// not incidental -- it is what makes the reduction cascade: a word written on
// one iteration is re-read (and reduced again) deltaw iterations later.
// delta >= 2*wlen only guarantees deltaw >= 2 elementwise; it says nothing
// about the windows as a whole.
//
// Therefore these pointers MUST NOT be marked __restrict__.  Doing so lets the
// vectoriser drop the runtime alias check and reorder loads across stores,
// silently breaking the cascade for deltaq == 0.

// The boundary needs the carry out of the main reduce loop.
// Cleanest solution: return it from the helper.
    static _ntl_ulong
fastsqr_reduce(const _ntl_ulong * src,
        _ntl_ulong * dst,
        long count, unsigned long deltaq, unsigned long deltaqc)
    // Computes dst[j] ^= (src[j] >> deltaq) | (src[j+1] << deltaqc)
    // for j = count downto 0, where src[j+1] is the "carry" from j+1.
    // Note: caller must handle the boundary (masked) iterations separately.
    // Returns the final value of "newer" (the carry into the boundary step).
{
    _ntl_ulong newer = 0;
    for (long j = count; j >= 0; j--) {
        _ntl_ulong older = newer;
        newer = src[j];
        dst[j] ^= (newer >> deltaq) | (older << deltaqc);
    }
    return newer;  // caller uses this for the masked boundary iterations
}

// Word-aligned special case (deltaq == 0): simple XOR, fully vectorisable.
    static void
fastsqr_xor_block(const _ntl_ulong * __restrict__ src,
        _ntl_ulong * __restrict__ dst, long n)
{
    for (long j = 0; j < n; j++)
        dst[j] ^= src[j];
}

// Blocked aligned reduce.  ap[g-deltaw] ^= ap[g] for g = q1 downto q4+1.
// Source and destination ALWAYS overlap; the loop-carried dependence
// distance is exactly deltaw words, so any run of <= deltaw consecutive
// g values is internally independent and may be vectorised, provided the
// blocks themselves are walked from high g to low g.
    static void
fastsqr_reduce_aligned(_ntl_ulong * ap, long q1, long q4, long deltaw)
{
    long hi = q1;
    while (hi > q4) {
        long n = hi - q4;
        if (n > deltaw) n = deltaw;
        fastsqr_xor_block(ap + hi - n + 1, ap + hi - n + 1 - deltaw, n);
        hi -= n;
    }
}

// 64-bit constants for bit-spread (interleave step).
static const _ntl_ulong IC0 = 0x00000000FFFFFFFFull;
static const _ntl_ulong IC1 = 0x0000FFFF0000FFFFull;
static const _ntl_ulong IC2 = 0x00FF00FF00FF00FFull;
static const _ntl_ulong IC3 = 0x0F0F0F0F0F0F0F0Full;
static const _ntl_ulong IC4 = 0x3333333333333333ull;
static const _ntl_ulong IC5 = 0x2222222222222222ull;

// Spread one pair of 64-bit source words (lo, hi) into two interleaved
// output words at bp[2*j] and bp[2*j+1].
    static inline void
spread_word_pair(_ntl_ulong lo, _ntl_ulong hi,
        _ntl_ulong * __restrict__ bp, long j)
{
    _ntl_ulong t = lo & IC0, u = lo >> 32;
    _ntl_ulong v = hi & IC0, w = hi >> 32;
    u = (u | u<<16) & IC1;  t = (t | t<<16) & IC1;
    w = (w | w<<16) & IC1;  v = (v | v<<16) & IC1;
    u = (u | u<<8)  & IC2;  t = (t | t<<8)  & IC2;
    w = (w | w<<8)  & IC2;  v = (v | v<<8)  & IC2;
    u = (u | u<<4)  & IC3;  t = (t | t<<4)  & IC3;
    w = (w | w<<4)  & IC3;  v = (v | v<<4)  & IC3;
    u = (u | u<<2)  & IC4;  t = (t | t<<2)  & IC4;
    w = (w | w<<2)  & IC4;  v = (v | v<<2)  & IC4;
    u += u & IC5;           t += t & IC5;
    w += w & IC5;           v += v & IC5;
    bp[2*j+1] = u | (w << 1);
    bp[2*j]   = t | (v << 1);
}

// Interleave: reads ap[0..q5] (lo half) and ap[q5..2*q5+1] (hi half),
// writes interleaved result to bp[0..2*q5+1].
// Uses simplified direct indexing (no Sparc prefetch chain).
    static void
fastsqr_interleave(const _ntl_ulong * __restrict__ ap,
        _ntl_ulong * __restrict__ bp,
        long q5, unsigned long s1, unsigned long s2)
{
    for (long j = 0; j <= q5; j++) {
        _ntl_ulong lo     = ap[j];
        _ntl_ulong hi_cur = ap[j + q5];
        _ntl_ulong hi_nxt = ap[j + q5 + 1];
        /* the two-step shift keeps every count below 64, so s1 == 0
           (s2 == 63) is safe: that term is then 0, as it must be */
        _ntl_ulong hi = (hi_cur >> s1) | ((hi_nxt << 1) << s2);
        spread_word_pair(lo, hi, bp, j);
    }
}


void fastsqr_old(GF2X& b, GF2X& a, _ntl_ulong r, _ntl_ulong s,
        const SqrConstR& cr)
    /* Returns b = a*a mod (x^r + x^s + 1)

       Here a and b represent polynomials (in GF(2)[x]) of degree < r,
       using the natural representation (as used in NTL and irred) of
       one bit per coefficient (least significant bit of word 0 is the
       constant term).

       The input a is destroyed and must not overlap the output b.

       r and s must be odd. In addition, this implementation assumes that
       min(s, r-s) > 2*wlen, where wlen is the wordlength in bits (assumed
       to be 32 or 64).

       The algorithm is as described in Section 4 of Brent, Larvala and
       Zimmermann, Math. Comp. 72 (2003), 1443--1452 [rpb199], but without
       any overlap of the arrays.

       The algorithm has two steps (reduce and interleave). The first step
       reduces the implicit square of polynomial a of degree < r "in place"
       given x^r = x^s + 1 (mod 2), with the result left in scrambled order
       0 2 4 ... r-3 r-1 1 3 5 ... r-4 r-2 in a.  The second step then
       "interleaves" the bits to give the correct order in b.

       Uses about (2*load + 1.5*store)*r/wlen memory operations, which is
       better than the obvious "square and reduce" algorithm as in
       NTL's SqrMod. The saving is because operations are not performed
       on bits that are known to be zero.

       RPB, 20070318 (based on reducep, reducer and interlvf in irred.c). */

    /* As original, but:
       - r-only constants taken from pre-computed struct cr
       - hot loops extracted into helpers (see aliasing note above)
       - ~0L / ~1L replaced with UINT64_MAX / (~(_ntl_ulong)1)
       - 32-bit path removed
       - Sparc prefetch chain removed from interleave
     */
{
#if (NTL_BITS_PER_LONG != 64)
#error "fastsqr_old: only 64-bit supported in this version"
#endif

    assert(r & 1);
    assert(s & 1);

    // Per-s quantities
    const SqrConstS cs = make_sqr_const_s(r, s, cr);

    // Ensure a has smax words, zero-padding if needed
    long sa = a.xrep.length();
    if (sa < (long)cr.smax)
        a.xrep.SetLength(cr.smax);
    _ntl_ulong *ap = a.xrep.elts();
    for (long j = sa; j < (long)cr.smax; j++)
        ap[j] = 0;

    // Allocate b with two words of headroom
    b.xrep.SetLength(cr.smax + 2);
    _ntl_ulong *bp = b.xrep.elts();
    bp[cr.smax]   = 0;
    bp[cr.smax+1] = 0;

    // Mask irrelevant high bits of a
    ap[cr.q1] &= cr.mask1;

    /* --- Reduce step --- */
    if (cs.deltaq == 0) {
        // Word-aligned: plain XOR in blocks of <= deltaw words (source
        // and destination overlap; see fastsqr_reduce_aligned)
        fastsqr_reduce_aligned(ap, (long)cr.q1, (long)cr.q4, (long)cs.deltaw);
        ap[cr.q4 - cs.deltaw] ^= ap[cr.q4] & cs.mask2;
    } else {
        // General case: barrel-shift XOR
        // Source: ap[q4+1 .. q1], dest: ap[q4+1-deltaw .. q1-deltaw];
        // the two overlap (that is the cascade): see fastsqr_reduce
        _ntl_ulong carry = fastsqr_reduce(
            ap + cr.q4 + 1,
            ap + cr.q4 + 1 - cs.deltaw,
            (long)(cr.q1 - cr.q4 - 1),
            cs.deltaq, cs.deltaqc);
        // Boundary iterations (masked)
        _ntl_ulong nex = ap[cr.q4] & cs.mask2;
        ap[cr.q4   - cs.deltaw]     ^= (nex >> cs.deltaq) | (carry << cs.deltaqc);
        ap[cr.q4-1 - cs.deltaw]     ^=  nex << cs.deltaqc;
    }

    /* --- Interleave step --- */
    fastsqr_interleave(ap, bp, (long)cr.q5, cr.s1, cr.s2);

    bp[cr.smax+1] = 0;
    bp[cr.smax]   = 0;
    bp[cr.q1]    &= cr.mask1;
    b.normalize();
}

void
print_gf2x (GF2X& a)
{
  _ntl_ulong *ap = a.xrep.elts();
  for (long j = 0; j < 64 * a.xrep.length(); j++)
    if (ap[j / 64] >> (j % 64) & 1)
      printf ("+x^%lu", j);
  printf ("\n");
}

#ifdef GF2X_HAVE_BMI2_SUPPORT
/* b <- a^2 mod (x^r + x^s + 1)
   Assumes r and s are odd, and NTL_BITS_PER_LONG = 64.
   Assumes r-s >= 128.
   Destroys the value in a.
 */
void
fastsqr_pdep (GF2X& b, GF2X& a, _ntl_ulong r, _ntl_ulong s)
{
  assert (r & 1);
  assert (s & 1);
  assert (NTL_BITS_PER_LONG == 64);

  unsigned long smax = (r + 63) / 64; /* ceil(r/64) */

  unsigned long j, sa = a.xrep.length ();    /* size in words */
  if (sa < smax)
    a.xrep.SetLength (smax);	/* Always use length smax for a */

  _ntl_ulong *ap = a.xrep.elts();
  for (j = sa; j < smax; j++)
    ap[j] = 0;  	/* Clear high words of a if necessary */

  b.xrep.SetLength (smax);

  unsigned long alpha = (r - 1) / 2;
  unsigned long delta = (r - s) / 2;

  assert (delta >= 64); /* to avoid word-overlap between b_j and b_{j+delta} */

  /* first loop (Section 4.1 of "A fast algorithm ...", Math. Comp., 2002):

     for j <- r-1 downto alpha+1 do
        b_{j - delta} <- b_{j - delta} xor b_j

     for j <- r-1-delta downto alpha+1-delta do
        b_j <- b_j xor b_{j+delta}
  */
  unsigned long jmin, jmax, deltaq, deltar;
  jmin = (alpha + 1 - delta) / 64;  /* floor((alpha+1-delta)/64 */
  jmax = (r - 1 - delta) / 64;      /* floor((r-1-delta)/64 */
  deltaq = delta / 64;
  deltar = delta % 64;
  assert (jmax + deltaq < smax);
  assert (jmin + deltaq >= 1);
  /* separate case for j = jmax, since jmax + deltaq + 1 might
     equal smax */
  ap[jmax] ^= ap[jmax + deltaq] >> deltar;
  if (jmax + deltaq + 1 < smax && deltar != 0)
    ap[jmax] ^= ap[jmax + deltaq + 1] << (64 - deltar);
  if (deltar != 0)
    {
      for (j = jmax - 1; j > jmin; j--)
	{
	  /* add the 64 - deltar least significant bits of ap[j + deltaq],
	     after shift right by deltar */
	  ap[j] ^= ap[j + deltaq] >> deltar;
	  /* add the deltar least significant bits of ap[j + deltaq + 1],
	     after shift left by 64-deltar */
	  ap[j] ^= ap[j + deltaq + 1] << (64 - deltar);
	}
    }
  else /* special case for deltar = 0 */
    {
      for (j = jmax - 1; j > jmin; j--)
	ap[j] ^= ap[j + deltaq];
    }
  /* separate case for j = jmin, we should only modify bits x..64 where
     x = (alpha+1-delta) % 64 */
  unsigned long x = (alpha + 1 - delta) % 64;
  _ntl_ulong mask = (~0UL >> x) << x;
  ap[jmin] ^= mask & (ap[jmin + deltaq] >> deltar);
  if (deltar != 0)
    ap[jmin] ^= mask & (ap[jmin + deltaq + 1] << (64 - deltar));

  /* truncate the upper bits */
  ap[r / 64] &= (1UL << (r % 64)) - 1UL;

  /* now we have the bits of the result interleaved in 'a' (cf "A fast ..."):
     b_0 b_2 b_4 b_6 ... b_{r-1} b_1 b_3 b_5 b_7 ... b_{r-2} */
  jmax = (alpha + 1) / 64; /* floor(((r-1)/2+1)/64) */
  deltar = (alpha + 1) % 64;
  _ntl_ulong *bp = b.xrep.elts(), even, odd;
  if (deltar != 0)
    {
      for (j = 0; j < jmax; j++)
	{
	  even = ap[j];
	  odd = (ap[j + jmax] >> deltar) ^ (ap[j + jmax + 1] << (64 - deltar));
	  bp[2 * j] = _pdep_u64 (even, 0x5555555555555555) ^
	    _pdep_u64 (odd, 0xaaaaaaaaaaaaaaaa);
	  bp[2 * j + 1] = _pdep_u64 (even >> 32, 0x5555555555555555) ^
	    _pdep_u64 (odd >> 32, 0xaaaaaaaaaaaaaaaa);
	}
    }
  else /* case deltar = 0 */
    {
      for (j = 0; j < jmax; j++)
	{
	  even = ap[j];
	  odd = ap[j + jmax];
	  bp[2 * j] = _pdep_u64 (even, 0x5555555555555555) ^
	    _pdep_u64 (odd, 0xaaaaaaaaaaaaaaaa);
	  bp[2 * j + 1] = _pdep_u64 (even >> 32, 0x5555555555555555) ^
	    _pdep_u64 (odd >> 32, 0xaaaaaaaaaaaaaaaa);
	}
    }
  /* special case for j = jmax */
  even = ap[jmax];
  odd = (2 * jmax < smax) ? ap[2 * jmax] >> deltar : 0;
  if (2 * jmax + 1 < smax && deltar != 0)
    odd ^= ap[2 * jmax + 1] << (64 - deltar);
  if (2 * jmax < smax)
    bp[2 * jmax] = _pdep_u64 (even, 0x5555555555555555) ^
      _pdep_u64 (odd, 0xaaaaaaaaaaaaaaaa);
  if (2 * jmax + 1 < smax)
      bp[2 * jmax + 1] = _pdep_u64 (even >> 32, 0x5555555555555555) ^
	_pdep_u64 (odd >> 32, 0xaaaaaaaaaaaaaaaa);
  /* truncate the upper bits */
  bp[r / 64] &= (1UL << (r % 64)) - 1UL;
  b.normalize();
}
#endif

void
fastsqr (GF2X& b, GF2X& a, _ntl_ulong r, _ntl_ulong s, const SqrConstR& cr)
{
#if 0 /* debug code */
  GF2X copy_a = a, c;

  fastsqr_old (b, a, r, s, cr);
  a = copy_a;
  fastsqr_pdep (c, a, r, s);
  if (b != c)
    {
      printf ("error for r=%lu s=%lu\n", r, s);
      printf ("a="); print_gf2x (copy_a);
      printf ("fastsqr_old:  "); print_gf2x (b);
      printf ("fastsqr_pdep: "); print_gf2x (c);
      exit (1);
    }
#else
#ifdef GF2X_HAVE_BMI2_SUPPORT
  (void) cr;
  return fastsqr_pdep (b, a, r, s);
#else
  return fastsqr_old (b, a, r, s, cr);
#endif
#endif
}

void fastrem (GF2X& a, _ntl_ulong r, _ntl_ulong s)

// Sets a := a mod (x^r + x^s + 1) using a word operations where possible.
// This is much faster than NTL's rem if s > r/2.
// Assumes 0 < s < r and 32-bit or 64-bit words.	RPB 20070321

{
  long d, jw, krb, krw, ksb, ksw;
  long apr, aps, lastr, lasts, sh1, sh2;
  unsigned long rs, j, wlenr, jb;

#if (NTL_BITS_PER_LONG == 64)
				/* Assume wordlength is 32 or 64 bits */
  				/* This is for 64 bits */
  long wd = 6;
  unsigned long wlen = 64;	/* wordlength (bits) = 2^wd */
  unsigned long wlenm = 63;	/* wlen - 1 */
#else
  long wd = 5;			/* This is for 32 bits */
  unsigned long wlen = 32;	/* wordlength (bits) = 2^wd */
  unsigned long wlenm = 31;	/* wlen - 1 */
#endif

  _ntl_ulong high;
  int slow;

  _ntl_ulong *ap = a.xrep.elts();

  rs = r - s;

  if ((s <= 0) || (rs <= 0))
    {
      printf("Error in fastrem: r = %lu and s = %lu must satisfy r > s > 0\n",
	     r, s);
      exit (1);
    }

  d = deg(a);
  if (d < (long) r)
    return;

  slow = ((s <= 2*wlen) || (rs <= 2*wlen));	// Avoid word ops if slow
  wlenr = wlen + r;

  for (j = d; j >= r;)
    {
    jw = j>>wd;			// (j div wlen) = word index
    jb = j&wlenm;		// (j mod wlen) = bit index in word jw
    if (slow || (jb != wlenm) || (j < wlenr))
      {
      if (rep (coeff (a, j)) != 0)		// This is slow but only
        {					// executed a few times
        SetCoeff(a, j, 0);
        SetCoeff(a, j-r,  !(rep (coeff (a, j-r))));
        SetCoeff(a, j-rs, !(rep (coeff (a, j-rs))));
        }
      j--;					// Process next bit
      }
    else					// Process by words
      {						// Loop precomputation
      krb = (j-r)&wlenm;			// bit index in [0..wlen)
      krw = (j-r)>>wd;				// word index > 0
      ksb = (j-rs)&wlenm;			// bit index in [0..wlen)
      ksw = (j-rs)>>wd;				// word index > 0
      sh1 = wlenm - krb;			// first shift amount
      sh2 = wlenm - ksb;			// second shift amount

      for (lastr = lasts = 0; krw > 0; )
        {					// Loop to process by words
        high = ap[jw];
        apr  = ap[krw];				// In this version we avoid
        aps  = ap[ksw];				// the " ^= " operator in
        ap[jw--]  = 0;				// the inner loop.
        ap[krw--] = (apr^lastr)^(high >> sh1);	// This reduced dependencies.
        ap[ksw--] = (aps^lasts)^(high >> sh2);
	high  = high << 1;
        lastr = high << krb;			// krb == wlenm possible
        lasts = high << ksb;			// ksb == wlenm possible
        }
      ap[krw] ^= lastr;
      ap[ksw] ^= lasts;
      j = r + krb;				// krb bits remain
      }
    }
  a.normalize(); 	     			/* Normalize output */
}

int lex_less (GF2X a, GF2X b, int flip)

/* Returns true if
  (deg(a) < deg(b)) or ((deg(a) = deg(b)) and (a < b in lex ordering))
  except if flip is true when this is changed appropriately. */

{
  long da = deg (a);
  long db = deg (b);
  long i;
  if (da < db) return 1;
  if (da > db) return 0;
  if (flip)
    for (i = 0; i <= da; i++)
      {
      if (rep (coeff (a, i)) < rep (coeff (b, i)))
        return 1;
      if (rep (coeff (a, i)) > rep (coeff (b, i)))
        return 0;
      }
  else
    for (i = da; i >= 0; i--)
      {
      if (rep (coeff (a, i)) < rep (coeff (b, i)))
        return 1;
      if (rep (coeff (a, i)) > rep (coeff (b, i)))
        return 0;
      }
  return 0;		/* Equal here, return false */

}

void print_hex (char *s, GF2X a, int flip)

/* prints a in hexadecimal (leading bits first unless flip true) */

{
  long d = deg (a);
  long i, j, c;
  char t[17] = "0123456789abcdef";

  c = 0;
  for (i = d; i >= 0; i--)
    {
      j = (flip) ? (d-i) : i;
      c = (2 * c) + rep (coeff (a, j));
      if (i % 4 == 0)
        {
          *s++ = t[c];
          c = 0;
        }
    }
  *s = '\0';
}

_ntl_ulong
get_certificate (GF2X a)
{
  // avoid segmentation fault when it's irreducible
  if (a.xrep.length() != 0)
    return a.xrep.elts()[0];
  else
    return (_ntl_ulong)0; // value is irrelevant, won't be used
}

void
fastmulmod (GF2X& c, const GF2X& a, const GF2X& b, const GF2XModulus& F,
	         _ntl_ulong r, _ntl_ulong s)

// Returns c = a*b mod (x^r + x^s + 1)

{
  if ((s <= 0) || (r <= s))
    {
      printf("Error in fastmulmod: r = %lu and s = %lu must satisfy r > s > 0\n",
	     r, s);
      exit (1);
    }

   GF2X t;
   mul_gen(t, a, b);

   /* --- EC: phi is a ring hom on GF(2)[x], and t is the *unreduced*
      product, so phi(t) == phi(a)*phi(b) exactly.  Catches any error
      confined to fewer than n consecutive bit positions.            --- */
   if (ec_enabled)
     {
       { EC_TIC (ECT_MULFOLD);
         ecfold fa, fb, ft, fp;
         ec_fold (&fa, a); ec_fold (&fb, b); ec_fold (&ft, t);
         ec_fold_mul (&fp, &fa, &fb);
         if (!ec_fold_eq (&ft, &fp))
           EC_FAIL ("phi(t) != phi(a)*phi(b), multiply corrupted");
         EC_TOC (ECT_MULFOLD); }

       /* stash the unreduced product in our own word buffer: fastrem
          destroys t, and this avoids a GF2X allocation as well */
       { EC_TIC (ECT_COPY);
         ec_alloc ((long) r);
         ec_to_words (ec_wbuf, ec_nw2, t);
         EC_TOC (ECT_COPY); }
     }

   if ((r&s&1) && (s > BITS_PER_LONG2) && ((r-s) > BITS_PER_LONG2))
     {
       fastrem(t, r, s);
       c = t;
     }
   else
     rem(c, t, F);        // This is slow if s > r/2

   /* --- EC: c == t mod T  <=>  T | (t + c) and deg c < r.  No second
      reduction: see the note above ec_alloc for why that was fatal.  --- */
   ec_maybe_inject (c, "fastmulmod result");

   if (ec_enabled)
     {
       EC_TIC (ECT_DIVIS);
       if (deg (c) >= (long) r)
         EC_FAIL ("deg(c) = %ld >= r = %lu after reduction", deg (c), r);
       else
         {
           ec_xor_words (ec_wbuf, ec_nw2, c);          /* w = t + c */
           if (!ec_divides_trinomial (ec_wbuf, ec_nw2, (long) r, (long) s,
                                      ec_wsc, ec_wsc2))
             EC_FAIL ("T does not divide t + c, reduction corrupted");
         }
       EC_TOC (ECT_DIVIS);
     }
}

void fastmulbyxmod (GF2X& c, const GF2X& a, _ntl_ulong r, _ntl_ulong s)

// Returns c = a*x mod (x^r + x^s + 1)

{
  if ((s <= 0) || (r <= s))
    {
      printf("Error in fastmulbyxmod: r = %lu and s = %lu must satisfy r > s > 0\n",
	     r, s);
      exit (1);
    }

   MulByX(c, a);
   if (deg(c) >= (long) r)
     {
     SetCoeff(c, r, 0);
     SetCoeff(c, s,  !(rep (coeff (c, s))));
     SetCoeff(c, 0,  !(rep (coeff (c, 0))));
     c.normalize();
     }
}

void copy_h (vec_GF2X& hold, vec_GF2X& h, long m)

/* copy h[0..m-1] into hold[0..m-1] */

{
  for (long i = 0; i < m; i++)
    hold[i] = h[i];
}

void fastmulmod (GF2X& c, const GF2X& a, const GF2X& b, const GF2XModulus& F,
                 _ntl_ulong r, _ntl_ulong s);

/* --- EC (round 5): closed-form check of a freshly built h-table.

   h[j] is the sum of y^e over the exponents e < 2^m of Hamming weight j+1,
   with y = y_k = x^(2^k) mod T.  So the sum S of the whole table is
   y + y^2 + ... + y^(2^m - 1), a geometric series:

        (1 + y_k) (1 + S)  =  1 + y_k^(2^m)  =  1 + y_(k+m)      mod T.

   Both sides are computable from scratch without the table: k + m squarings
   of ONE residue and one modular product, against the m (k + m/2) squarings
   that building the table a second time costs (which is what rounds 2-4 did:
   5.4 - 7.6 s per s at r = 136279841, 69% of all checking at -skip 55).

   An error in the table changes S; an error in this routine's own squarings
   makes the identity fail too, which costs a rebuild and nothing else.
   Returns 1 if the identity holds.                                      --- */
static int
ec_init_closed_form (const vec_GF2X& h, long k, long m, const GF2XModulus& F,
                     _ntl_ulong r, _ntl_ulong s, int usefs,
                     const SqrConstR& cr)
{
  GF2X y, t, yk, S, one, u, v, lhs;
  SetX (y);
  set (one);
  for (long i = 0; i < k + m; i++)
    {
      if (i == k)
        yk = y;
      if (usefs)
        {
          fastsqr (t, y, r, s, cr);
          swap (y, t);
        }
      else
        SqrMod (y, y, F);
    }
  clear (S);
  for (long j = 0; j < m; j++)
    add (S, S, h[j]);
  add (u, yk, one);
  add (v, S, one);
  add (y, y, one);                      /* 1 + y_(k+m) */
  /* the product is checked by fastmulmod's own folds; keep its verdict
     apart from the scan's, and keep the test injector's opportunity count
     where it was before this check existed */
  int fail_save = ec_fail;
  long inj_save = ec_fault_at;
  ec_fail = 0;
  ec_fault_at = 0;
  fastmulmod (lhs, u, v, F, r, s);
  int mul_bad = ec_fail;
  ec_fail = fail_save;
  ec_fault_at = inj_save;
  return !mul_bad && lhs == y;
}

void init_h (vec_GF2X& h, long k, long m, const GF2XModulus& F,
 	_ntl_ulong r, _ntl_ulong s, int usefs, const SqrConstR& cr)

/* initialize h[j] to sum ((x^(2^k))^s, 0 <= s < 2^m, hw(s)=j+1),
   for 0 <= j < m */

{
  long i, j;
  GF2X a, x, htemp;

  /* we use the following Pascal-triangle algorithm:
     if S(m,j) := { 0 <= s < 2^m, hw(s) = j }
     then an element s of S(m,j) is either in S(m-1,j) if s < 2^(m-1),
     or s = 2^(m-1) + t where t is in S(m-1,j-1):
        S(m,j) = S(m-1,j) union [S(m-1,j-1) + 2^(m-1)]
     Or equivalently:
        S(m,j) = {2s, s in S(m-1,j)} union {1+2s, s in S(m-1,j-1)}.
     With S(m,0) = { 0 } and S(m,m) = { 2^m-1 }.
  */
  clear(htemp);
  SetX (x);
  for (j = 1; j <= m; j++)
    {
      h[j-1] = 0;
      for (i = j-1; i >= 0; i--)
	{
	  if (i > 0)
	    {
	     if (usefs)
	       {
	       fastsqr (htemp, h[i-1], r, s, cr);
	       h[i-1] = htemp;
	       }
	     else
	       SqrMod (h[i-1], h[i-1], F);	// Avoid if possible as slow
             fastmulbyxmod (a, h[i-1], r, s);
	    }
	  else
	    a = x;
	  add (h[i], h[i], a);
	}
    }
  for (j = 0; j < m; j++)
    {
    if (usefs)
      {
      for (i = 0; i < k/2; i++)
        {
        fastsqr (htemp, h[j], r, s, cr);
        fastsqr (h[j], htemp, r, s, cr);
        }
      if (k&1)		// last iteration if k odd
        {
        fastsqr (htemp, h[j], r, s, cr);
        h[j] = htemp;	// copy result
        }
      }
    else
      for (i = 0; i < k; i++)
        SqrMod (h[j], h[j], F);	// Avoid if possible as slow
    }
}

void
usage ()
{
  fprintf (stderr, "Usage: factor [options] r\n");
  fprintf (stderr, "Options:\n");
  fprintf (stderr, "   -b d    - omit hex encoding of factor if degree < d\n");
  fprintf (stderr, "   -f f    - use interval bounds of f^j (default is f=1);\n");
  fprintf (stderr, "             if f=1 then intervals increase linearly in size\n");
  fprintf (stderr, "             if f=0 then intervals do not increase\n");
  fprintf (stderr, "   -m m    - use double-blocking size of m\n");
  				// default is as for s if r small, or try to
  				// estimate if r large
  fprintf (stderr, "   -p      - assume 2^r-1 is a Mersenne prime so any irreducible\n");
  fprintf (stderr, "             trinomial is primitive\n");
  fprintf (stderr, "   -q q0   - start with q0 blocks of length m (default is q0=4)\n");
  fprintf (stderr, "   -k k0   - use double-blocking strategy only for degree > k0\n");
  fprintf (stderr, "             (default is <= largest k0 such that 2^k0 < r); assume k0 > 0\n");
  fprintf (stderr, "   -skip d - skip degrees <= d (input trinomials are known to have no\n");
  fprintf (stderr, "             factor of degree <= d)\n");
  fprintf (stderr, "   -maxd d - skip degrees > d\n");
  fprintf (stderr, "   -v      - verbose\n");
  fprintf (stderr, "   -z q1   - output 'u' if number of blocks of length m would be > q1\n");
  fprintf (stderr, "             (default is 'infinity')\n");
  fprintf (stderr, "   -Z q1   - cap the number of blocks at q1, but continue\n");
  fprintf (stderr, "             (-z and -Z: the last one given applies)\n");
  fprintf (stderr, "   -s0 s0  - start with x^r + x^s0 + 1, default 1\n");
  fprintf (stderr, "   -s1 s1  - ends with x^r + x^(s1-1) + 1, default floor(r/2)+1\n");
  fprintf (stderr, "   -mt n   - use n threads (default 1)\n");
  fprintf (stderr, "   Hardware error checking (Gerbicz chain on the h-table, cyclic-fold\n");
  fprintf (stderr, "   checks on the products, closed-form check of init_h) is ON by default:\n");
  fprintf (stderr, "   -noec   - turn it off (for timing comparisons)\n");
  fprintf (stderr, "   -ec     - accepted; it is the default\n");
  fprintf (stderr, "   -ecw w  - run the Gerbicz check every w blocks (default 8);\n");
  fprintf (stderr, "             larger w is cheaper but rolls back further\n");
  fprintf (stderr, "   -ecg    - also recompute each gcd and compare\n");
  fprintf (stderr, "   -ect    - report per-category error-checking cost at exit\n");
  fprintf (stderr, "   -eci0   - skip the check of the h-table initialisation\n");
  fprintf (stderr, "             (k+m squarings and one product per s value)\n");
  fprintf (stderr, "   -ecfault N / -ecfaultinit - flip one bit (testing the checker)\n");
  exit (1);
}

/* a <- C[0] * C[1] * ... * C[z-1] mod F := (x^r+x^s+1) */
static void
accumulate (GF2X& a, GF2XVec& C, const GF2XModulus& F, long r, long s)
{
  long z = C.length();

  /* invariant: z is the number of cells C[0]...C[z-1] to accumulate in a */
  while (z > 1)
    {
      /* uncomment this pragma when only one trinomial has to be checked */
#pragma omp parallel for schedule(static)
      for (long j = 0; j < z/2; j++)
	fastmulmod (C[j], C[j], C[z-1-j], F, r, s);
      z = (z + 1) / 2;
    }
  fastmulmod (a, a, C[0], F, r, s);
}

int
main (int argc, char *argv[])
{
  long r; /* degree of trinomials */
  long maxd = LONG_MAX; /* skip degrees > maxd */
  long m = 0, k0 = 0, b0 = 2, q0 = 4, q1 = 0;
  bool q1_continue_flag = false;	/* -Z rather than -z */
  long skipd = 0, fineDDFtol;
  double f = 1.0; /* scaling factor, default 1.0 */
  long s0 = 1, s1 = 1;

  int verbose = 0, primitive = 0, choose_m = 0;

  int mt = 1;

  /* print whole command line */
  for (int i = 0; i < argc; i++)
    fprintf (stderr, "%s ", argv[i]);
  fprintf (stderr, "\n");

  fprintf (stderr, "This is factor version %1.2f\n", GF2X_FACTOR_VERSION);
  fprintf (stderr, "based on NTL version %s\n", NTL_VERSION);
  /* deferred: ec status is printed after option parsing, below */

  while (argc > 1 && argv[1][0] == '-')
    {
      if (strcmp (argv[1], "-v") == 0)
        {
          verbose ++;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-p") == 0)
        {
          primitive = 1;
          argc --;
          argv ++;
        }
      else if (argc > 2 && strcmp (argv[1], "-m") == 0)
        {
          m = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-f") == 0)
        {
          f = atof (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-k") == 0)
        {
          k0 = atoi (argv[2]);
	  if (k0 > NTL_BITS_PER_LONG)
	    {
	      fprintf (stderr, "Error, k0 should be <= NTL_BITS_PER_LONG\n");
	      exit (1);
	    }
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-b") == 0)
        {
          b0 = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-q") == 0)
        {
          q0 = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-z") == 0)
        {
          q1 = atoi (argv[2]);
          q1_continue_flag = false;
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-Z") == 0)
        {
          q1 = atoi (argv[2]);
          q1_continue_flag = true;
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-skip") == 0)
        {
          skipd = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-maxd") == 0)
        {
          maxd = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-s0") == 0)
        {
          s0 = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-s1") == 0)
        {
          s1 = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (argc > 2 && strcmp (argv[1], "-mt") == 0)
        {
          mt = atoi (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (strcmp (argv[1], "-noec") == 0)
        {
          ec_enabled = 0;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-ecfaultinit") == 0)
        {
          ec_fault_init = 1;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-ec") == 0)
        {
          ec_enabled = 1;		/* the default; accepted for old scripts */
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-ecg") == 0)
        {
          ec_gcdrep = 1;
          argc --;
          argv ++;
        }
      else if (argc > 2 && strcmp (argv[1], "-ecfault") == 0)
        {
          ec_fault_at = atol (argv[2]);
          argv += 2;
          argc -= 2;
        }
      else if (strcmp (argv[1], "-ect") == 0)
        {
          ec_timing = 1;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-eci0") == 0)
        {
          ec_initchk = 0;
          argc --;
          argv ++;
        }
      else if (argc > 2 && strcmp (argv[1], "-ecw") == 0)
        {
          ec_window = atoi (argv[2]);
          if (ec_window < 1)
            {
              fprintf (stderr, "Error, -ecw must be >= 1\n");
              exit (1);
            }
          argv += 2;
          argc -= 2;
        }
      else
        {
          fprintf (stderr, "Error, invalid option: %s\n", argv[1]);
          exit (1);
        }
    }

  if (argc != 2)
    usage ();

  if (ec_enabled)
    fprintf (stderr, "Error checking ENABLED: Gerbicz window %d block(s), "
	     "fold modulus x^%d+1%s%s%s\n", ec_window,
	     ECFOLDW * NTL_BITS_PER_LONG, ec_gcdrep ? ", repeated gcd" : "",
	     ec_initchk ? "" : ", init_h check off",
	     ec_timing  ? ", timing on" : "");
  else
    fprintf (stderr, "Error checking DISABLED (-noec): a hardware fault "
	     "will silently produce a wrong result\n");

  r = atol (argv[1]);

  if (r < 2)
    {
      fprintf (stderr, "Error, r < 2\n");
      exit (1);
    }

  const SqrConstR cr = make_sqr_const_r (r);

  /* check s0 and s1 */
  assert (0 < s0 && s0 < r);
  assert (0 < s1 && s1 <= r);

  if ((primitive) && (!ProbPrime(r, 2)))
    {
      fprintf (stderr,
	       "Error: -p option and r = %lu is not prime\n", r);
      exit (1);
    }

  if (k0 == 0)
    {
      for (; (2 << k0) < r; k0++)
	;
      if (k0 > 20)
	k0 = (k0+20)/2;		// Empirical optimum, could check at runtime
    }

  if (m == 0)
    {
      choose_m = (r > 1000);	 // Later try to find optimal m
      for ( ; (2 << m) < r; m++) // Reasonable though probably not optimal.
	;
    }
  /* we cannot determine m at runtime if mt > 1 */
  assert (choose_m == 0 || mt == 1);

  if (r < 17) 			// Tiny case, this avoids a problem
    m = 1;			// with e.g. r = 11, s = 1.

  if (r < 1000)			// Small case, this avoids a problem
    q0 = 1;			// with e.g. r = 521, s = 62, failing
    				// to enter fineDDF mode (it was OK with
    				// V1.29, but not with V1.30 .. V1.32)
  if (m > r / 2)
    {
      fprintf (stderr, "Error, m = %ld greater than r/2\n", m);
      exit (1);
    }

  if (q0 < 1)
    {
      fprintf (stderr, "Error, q0 < 1\n");
      exit (1);
    }

  if (q1 == 0)
    q1 = r;				// Effectively infinity

  long r3;
  for (r3 = 0; r3*r3*r3 < r; r3++) ;	// r3 = r^(1/3)
  fineDDFtol = 2*r3*r3;			// 2*r^(2/3)

  if (verbose)
    {
      printf ("Test separately degrees up to k0=%ld\n", k0);
      if (skipd > 0)
        printf ("Skip degrees up to %ld\n", skipd);
      printf ("Using block size of m=%ld\n", m);
      printf ("Start with q0=%ld block(s)\n", q0);
      if (q1 < r)
        {
          if (q1_continue_flag)
            printf ("Cap the number of blocks at q1=%ld\n", q1);
          else
            printf ("Stop when number of blocks is > q1=%ld\n", q1);
        }
      printf ("Use fine DDF above %ld\n", fineDDFtol);
      fflush (stdout);
    }

  /* since all degree k factors divide x^(2^k)-x, if x^r+x^s+1 has a degree
     k factor, then so has x^r+x^(s+2^k-1)+1, thus if 2^k-1 <= floor(r/2)-1,
     i.e. 2^(k+1) <= r, then we can save some computations */

  /* NTL is thread-safe up from version 7 */
  if (mt > 1 && NTL_MAJOR_VERSION < 7)
    {
      fprintf (stderr, "Error, use NTL version >= 7 with multiple threads\n");
      exit (1);
    }

  /* prepare inputs */
  long *todo, ntodo = 0;
  todo = (long*) malloc (r * sizeof (long));
  /* fill todo[] with entries in [s0, s1-1] */
  for (long s = s0; s < s1; s++)
    todo[ntodo++] = s;
  /* read entries from stdin (only when s0 >= s1) */
  if (ntodo == 0)
    while (1)
      {
	long s;
	int ret = scanf ("%ld", &s);
	if (ret != 1)
	  break;
	assert (0 < s && s < r);
	todo[ntodo++] = s;
	assert (ntodo <= r);
      }

  omp_set_num_threads (mt);
#pragma omp parallel
#pragma omp master
  printf ("Using %d thread(s)\n", omp_get_num_threads ());

  /* uncomment this pragma when several trinomials are checked in parallel */
  // #pragma omp parallel for schedule(dynamic)
  for (long idx = 0; idx < ntodo; idx++)
        {
	  long input_s = todo[idx];
	  int divisible, skip;
	  long s = input_s, j, l;
	  long ss, rhigh, k = 0, k1, k2, q, q2 = 0;
	  int fineDDF, flip, swan;
	  char *fact;
	  double f0;
	  int usefastsqr;
	  _ntl_ulong certificate = 0;

	  fact = (char*) malloc ((2 + (r / 4)) * sizeof (char));

	  skip = 0;

	  GF2X p;
	  SetCoeff(p, r);
	  SetCoeff(p, 0);

	  if (s < 1)
	    {
	    fprintf (stderr, "Error, s < 1\n");
	      exit (1);
	    }
	  if (s >= r)
	    {
	      fprintf (stderr, "Error, s = %ld >= r\n", s);
	      exit (1);
	    }

// We want to flip (i.e. consider reciprocal trinomial x^r + x^(r-s) + 1)
// if s > r/2, because the NTL mod operation is slow in this case.
// However, we also need to flip if s is even, so that we can use
// the fast squaring algorithm (fastsqr). Thus sometimes we work with
// s > r/2 (but then SqrMod is avoided where possible).

	  ss = s;		// Save s (in case flipped)
	  flip = 0;
	  if (2*s > r)
	    {
	    flip = 1;		// Use reciprocal trinomial
	    s = r - s;		// Original s saved in ss
	    }

	  usefastsqr = 0;
	  if ((r&1) && ((s&1) == 0) && (s > BITS_PER_LONG2)
	  			    && ((r-s) > BITS_PER_LONG2))
	    {
	    flip = !flip;	// Flip (again if necessary)
	    s = r - s;		// so we can use fastsqr
	    }
	  usefastsqr = ((r&s&1) && (s > BITS_PER_LONG2)
	  			&& ((r-s) > BITS_PER_LONG2));

	  if (verbose && flip)
	    printf ("Using reciprocal trinomial\n");

	  if (verbose && usefastsqr)
	    printf ("Using fastsqr\n");

	  // If r = +-1 mod 8,
	  // and s even and s does not divide 2r,
	  // or s odd and (r-s) does not divide 2r,
	  // then x^r + x^s + 1 had an odd number of irreducible factors
	  // by Swan's theorem.

	  swan = (((r&7) == 1) || ((r&7) == 7)) &&
	    	 ((((s&1)==0) && (((2*r)%s) != 0)) ||
	    	  (((s&1)==1) && (((2*r)%(r-s)) != 0)));

	  rhigh = (swan) ? (r/3) : (r/2);

          /* in case the -skip d option is used, and d >= rhigh, stop at d+1 */
          if (skipd >= rhigh)
            {
              rhigh = skipd + 1;
              if (rhigh > r)
                rhigh = r;
            }
          else if (verbose && swan)
	    printf ("By Swan's theorem search at most to degree %ld\n", rhigh);

          if (maxd < rhigh)
            {
              rhigh = maxd;
              if (verbose)
                printf ("Maximal degree checked reduced to %ld\n", rhigh);
            }

	  SetCoeff(p, s, 1);
	  GF2X a, c, htemp, htemp2;
	  GF2XModulus F;
          int Finit = 0; // non zero if F is build
	  clear(htemp);
	  double st;

	  if (choose_m)  			// Try to estimate optimal m
	    {
	    long jt, jtt, mopt;
	    double times, timem;

            if ((Finit == 0) && (!usefastsqr))
              {
                build (F, p);
                Finit = 1;
              }

	    choose_m = 0;			// Only try once
	    /* EC: the calibration below times fastmulmod in a loop.  Leaving
	       the checks on would inflate the measured M and make choose_m
	       pick the wrong block size, so suspend EC for the duration. */
	    int ec_save = ec_enabled;
	    ec_enabled = 0;
	    SetCoeff (htemp, 0, 1);
	    SetCoeff (htemp, 1, 1);
	    if (usefastsqr)
	      {
	      for (jt = 0; jt < 50; jt++) 	// Warm up cache
	        {				// and make htemp dense
	        fastsqr (htemp2, htemp, r, s, cr);
	        fastsqr (htemp, htemp2, r, s, cr);
	        }
	      times = GetTime ();
	      for (jt = 0; (GetTime() - times) < (double)1.0 ; jt += 100)
	        for (jtt = 0; jtt < 50; jtt++)	// Avoid many GetTime calls
	          {
	          fastsqr (htemp2, htemp, r, s, cr);
	          fastsqr (htemp, htemp2, r, s, cr);
	          }
	      }
	    else
	      {
	      for (jt = 0; jt < 100; jt++)
	        SqrMod (htemp, htemp, F);	// Warm up cache
	      times = GetTime ();
	      for (jt = 0; (GetTime() - times) < (double)1.0; jt += 100)
	        for (jtt = 0; jtt < 100; jtt++)
	          SqrMod (htemp, htemp, F);
	      }
	    times = (GetTime () - times)/(double)jt;
	    fastmulmod (htemp, htemp, htemp, F, r, s);
	    timem = GetTime ();
	    for (jt = 0; (GetTime() - timem) < (double)1.0; jt++)
	      fastmulmod (htemp, htemp, htemp2, F, r, s);
	    timem = (GetTime () - timem)/(double)jt;
	    // Prefer even m to avoid copying overhead
	    mopt = 2*(long)((double)0.5*sqrt(timem/times) + (double)0.5);
	    fprintf (stderr, "M %f msec, S %f msec, M/S %f, sqrt(M/S) %f\n",
	      1000.0* timem, 1000.0*times, timem/times, sqrt(timem/times));
	    ec_enabled = ec_save;		// EC back on after calibration
	    fprintf (stderr, "Estimate of best m %ld\n", mopt);
 	    if ((mopt > 1) && (mopt <= 100))
 	      {
 	      m = mopt;
 	      if (verbose) printf ("Using optimal block size of m=%ld\n", m);
 	      }
	    }

	  vec_GF2X h(INIT_SIZE, m);
	  vec_GF2X hold(INIT_SIZE, m); 	/* copy of h for fine DDF */

	  f0 = (double) 1.0;
	  q  = 0;
	  k2 = 1; /* invariant: k2 is last degree checked */
	  fineDDF = 0;  /* 0 as long as GCDs are trivial */
	  for (a = 1; (deg (a) == 0 && k2 < rhigh) || fineDDF != 0;)
	    {
	      if (fineDDF == 0)
		k = k2 + 1; /* next degree to check */
	      else /* fineDDF mode */
		{
		  if (deg (a) > 0) /* smallest factor in current interval */
		    {
		      a = 1; /* reset a to 1 */
		      q2 = q / 2;
		      q = q - q2;
		      copy_h (h, hold, m); /* revert to initial h */
		    }
		  else /* smallest factor in next interval */
		    {
		      k += q * m;
		      q = q2;
		    }
		  k2 = k + q * m - 1;
		}

	      if (k <= k0) /* sieve only one degree at a time */
		{
		  unsigned long K = (1 << k) - 1;

		  if (verbose)
                    printf ("Degree %ld: ", k);

                  if (k <= skipd)
                    {
                      if (verbose)
                        printf ("skip\n");
                    }
                  else
                    {
                      a = 1;
                      SetCoeff (a, K);
                      c = 1;
                      SetCoeff (c, r % K, coeff (c, r % K) == 0);
                      SetCoeff (c, s % K, coeff (c, s % K) == 0);
                      st = GetTime ();
                      GCD (a, a, c);
                      if (verbose)
                        {
                          printf ("gcd took %f sec\n", GetTime () - st);
                          fflush (stdout);
                        }
                    }
		  k2 = k;
		}
	      else /* use double-blocking strategy */
		{
                  if ((Finit == 0) && (!usefastsqr))
                    {
                      build (F, p);
                      Finit = 1;
                    }
		  /* initialize h-table */
		  if (k == k0 + 1)
		    {
		      st = GetTime ();
		      init_h (h, k, m, F, r, s, usefastsqr, cr);
		      /* --- EC: the Gerbicz check certifies how the table
			 *evolves*, not where it starts.  The start is checked
			 against the closed form of its digest (see
			 ec_init_closed_form); a table that fails is rebuilt. --- */
		      if (ec_fault_init)
			{		/* testing: a bad table must not pass */
			  ec_fault_init = 0;
			  long pos = deg (h[m / 2]) / 2;
			  SetCoeff (h[m / 2], pos, !(rep (coeff (h[m / 2], pos))));
			  fprintf (stderr, "EC INJECT: flipped bit %ld of "
				   "h[%ld] right after init_h\n", pos, m / 2);
			}
		      if (ec_enabled && ec_initchk)
			{
			  EC_TIC (ECT_INIT);
			  for (long tries = 0;
			       !ec_init_closed_form (h, k, m, F, r, s,
						     usefastsqr, cr); tries++)
			    {
			      fprintf (stderr, "EC FAIL [init_h] the h-table "
				       "for s=%ld fails its closed-form check"
				       " -- rebuilding\n", input_s);
			      fflush (stderr);
			      ec_detections++;
			      if (tries + 1 >= ec_maxretry)
				{
				  fprintf (stderr, "EC: %ld consecutive "
					   "failures at the same point -- this "
					   "is not a cosmic ray. Aborting.\n",
					   ec_maxretry);
				  exit (1);
				}
			      init_h (h, k, m, F, r, s, usefastsqr, cr);
			    }
			  EC_TOC (ECT_INIT);
			}
		      if (verbose)
			printf ("Exponentiation took %f\n", GetTime () - st);
		    }

		  if (fineDDF == 0)
		    {
		      if (f == 0.0)
			q = q0; /* no increase */
		      else if (f > (double) 1.0)
		 	q = (long) (f0 * (double) q0 + (double) 0.5); // round
		      else
		        q = q + q0;		// linear increase if f = 1
		      /* -Z: cap the number of blocks and keep going.  (2.02
		         kept the previous q instead, which left q = 0 -- an
		         empty interval -- when q0 > q1, and with -f 0 did not
		         scan the interval at all.) */
		      if (q1_continue_flag && q > q1)
		        q = q1;
		      f0 = f * f0;
		      k2 = k + q * m - 1;	// q blocks of size m
		    }

		  if (verbose)
                    {
                      printf ("Current q %ld\n", q);
                      fflush (stdout);
                    }

		  /* k2 should not exceed rhigh + m - 1 < r */
		  if (k2 > rhigh) /* should not happen in fine DDF mode */
		    {
		      q = (rhigh - k + m) / m;	// Round up
		      k2 = k + q * m - 1;
		      if (verbose)
			printf ("Reducing q to %ld\n", q);
		    }

		  /* we need to save h whenever q > 1 and the current gcd=1 */
		  if (q > 1 && deg (a) == 0) /* save h for fineDDF */
                    copy_h (hold, h, m);

		  if (verbose)
                    {
                      printf ("Interval %ld..%ld:\n", k, k2);
                      fflush (stdout);
                    }

		  if (q > q1 && !q1_continue_flag) /* Output "u" and skip here */
		    {
		    if (verbose)
		      printf ("q > q1 so skipping further test\n");
		    skip = 1;
		    divisible = 0;
		    k2 = r;
		    }
		  else
		    {
		    st = GetTime ();
		    double wct = WctTime ();
		    long i;

		    /* --- EC: checkpoint the state this interval starts from,
		       so that a failed check can simply redo the interval. --- */
		    vec_GF2X hchk;
		    GF2X achk, ec_S, ec_D, ec_S0, ec_Sn, ec_sD, ec_rhs;
		    long ec_attempt;
		    if (ec_enabled)
		      {
			hchk.SetLength (m);
			copy_h (hchk, h, m);
			achk = a;
		      }

		  for (ec_attempt = 0; ; ec_attempt++)
		   {
		    ec_fail = 0;
		    if (ec_enabled)
		      {
			/* S_0 = XOR_j h[j], the block digest at interval start */
			clear (ec_S0);
			for (j = 0; j < m; j++) add (ec_S0, ec_S0, h[j]);
			ec_S = ec_S0;
			clear (ec_D);
		      }
		    long ec_w = 0;

		    /* we can't easily parallelize this loop since the h[j]
		       are updated at each loop on i */
		    GF2XVec C;
		    C.SetSize ((k2 - k + m) / m,
			       (r + NTL_BITS_PER_LONG - 1) / NTL_BITS_PER_LONG);
		    long z = 0;
		    for (i = k; i + m - 1 <= k2 ; i += m, z++) /* i .. i+m-1 */
		      {
                        if (i + m - 1 <= skipd)
                          {
                            if (verbose >= 2)
                              printf ("   skip degree interval %ld..%ld\n",
                                      i, i + m - 1);
                          }
                        else
			  {
			    C[z] = 1; /* s_{0,m} = 1 */
			    ecfold fC, fT, fh, ftmp;
			    if (ec_enabled)
			      {
				GF2X one; set (one);
				ec_fold (&fC, one);
				ec_fold (&fT, p);
			      }
			    /* first compute the polynomial p_m(X,x) from head of
			       page 51 in "A multi-level blocking distinct-degree
			       factorization algorithm" */
			    for (j = 0; j < m; j++)
			      {
				/* EC: C*x mod T == (C<<1) + wrap*T exactly, so
				   phi(C*x mod T) = x*phi(C) + wrap*phi(T).
				   NOTE the correction is phi(T), *not*
				   phi(x^s+1) -- getting that wrong makes the
				   checker fire on correct data. */
				int ec_wrap = ec_enabled &&
				              (rep (coeff (C[z], r - 1)) != 0);
				fastmulbyxmod (C[z], C[z], r, s);
				add (C[z], C[z], h[j]);
				if (ec_enabled)
				  {
				    ec_fold_shl1 (&ftmp, &fC);
				    if (ec_wrap) ec_fold_add (&ftmp, &ftmp, &fT);
				    ec_fold (&fh, h[j]);
				    ec_fold_add (&fC, &ftmp, &fh);
				  }
			      }
			    if (ec_enabled)
			      {
				EC_TIC (ECT_HORNER);
				ec_fold (&ftmp, C[z]);
				if (!ec_fold_eq (&fC, &ftmp))
				  EC_FAIL ("horner fold mismatch at z=%ld", z);
				EC_TOC (ECT_HORNER);
			      }
			  }

			/* EC: accumulate the telescoping window digest
			   D <- D + S_z  *before* the table advances. */
			if (ec_enabled)
			  { EC_TIC (ECT_DIGEST); add (ec_D, ec_D, ec_S);
			    EC_TOC (ECT_DIGEST); }

			/* update the h[j] values */
			if (ec_enabled)
			  clear (ec_Sn);
			for (j = 0; j < m; j++)
			  {
			    if (usefastsqr)
			      {			// Can use fastsqr here
				for (l = 0; l < m/2; l++)
				  {
				    fastsqr (htemp, h[j], r, s, cr);
				    fastsqr (h[j], htemp, r, s, cr);
				  }
				if (m&1)		// Last square if m odd
				  {
				    fastsqr (htemp, h[j], r, s, cr);
				    h[j] = htemp;	// Copy result to h[j]
				  }
			      }
			    else		// Have to use SqrMod here
			      for (l = 0; l < m; l++)
			        SqrMod (h[j], h[j], F);
			    /* EC: S_{z+1} = XOR_j h[j]^(z+1).  In a GPU port
			       this XOR fuses into the store of the last
			       squaring and costs essentially nothing. */
			    ec_maybe_inject (h[j], "h-table entry");
			    if (ec_enabled)
			      { EC_TIC (ECT_DIGEST); add (ec_Sn, ec_Sn, h[j]);
			        EC_TOC (ECT_DIGEST); }
			  }

			/* ---- EC: Frobenius-linear Gerbicz check -------------
			   sigma = Frob^m is additive in characteristic 2, so
			      sigma(S_z) = S_{z+1}
			   and, telescoped over a window of w blocks,
			      sigma(D) + D = S_{z0} + S_{z1}.
			   Cost: one sigma (m squarings) per w blocks, i.e.
			   1/(w*m) of the squaring work.  Detects any single
			   fault in the table update -- deterministically, not
			   probabilistically.                              --- */
			if (ec_enabled)
			  {
			    ec_w++;
			    if (ec_w >= ec_window || i + m + m - 1 > k2)
			      {
				EC_TIC (ECT_SIGMA);
				ec_sD = ec_D;
				if (usefastsqr)
				  {
				    for (l = 0; l < m/2; l++)
				      {
					fastsqr (htemp, ec_sD, r, s, cr);
					fastsqr (ec_sD, htemp, r, s, cr);
				      }
				    if (m&1)
				      {
					fastsqr (htemp, ec_sD, r, s, cr);
					ec_sD = htemp;
				      }
				  }
				else
				  for (l = 0; l < m; l++)
				    SqrMod (ec_sD, ec_sD, F);
				add (ec_sD, ec_sD, ec_D);      /* sigma(D)+D */
				add (ec_rhs, ec_S0, ec_Sn);    /* S_z0 + S_z1 */
				if (ec_sD != ec_rhs)
				  EC_FAIL ("gerbicz: sigma(D)+D != S_z0+S_z1, "
				           "h-table corrupted in the last %d block(s)", ec_window);
				EC_TOC (ECT_SIGMA);
				clear (ec_D);
				ec_S0 = ec_Sn;
				ec_w = 0;
			      }
			    ec_S = ec_Sn;
			  }
		      }

		    assert (z == (k2 - k + m) / m);

		    if (k2 <= skipd)
		      {
                        if (verbose)
                          {
                            printf ("   skip products\n");
                            fflush (stdout);
                          }
			goto gcds;
		      }

		    /* accumulate the C[z] values */
		    accumulate (a, C, F, r, s);

		    /* --- EC: everything checked out?  Leave the retry loop. --- */
		    if (!ec_enabled || !ec_fail)
		      break;
		    ec_detections++;
		    fprintf (stderr,
			     "EC: check FAILED for s=%ld interval %ld..%ld "
			     "(attempt %ld) -- recomputing\n",
			     input_s, k, k2, ec_attempt + 1);
		    fflush (stderr);
		    if (ec_attempt + 1 >= ec_maxretry)
		      {
			fprintf (stderr, "EC: %ld consecutive failures at the "
				 "same point -- this is not a cosmic ray. "
				 "Aborting.\n", ec_maxretry);
			exit (1);
		      }
		    copy_h (h, hchk, m);        /* roll back the h-table */
		    a = achk;                   /* and the accumulator    */
		   }
                    st = GetTime () - st;

		    if (verbose)
                      {
			double wt = WctTime () - wct;
                        printf ("   squares/products took %.1f (wct %.1f), per term %.3f (wct %.3f)\n",
                                st, wt, st/(double)(q*m), wt/(double)(q*m));
                        fflush (stdout);
                      }

		    gcds:
                    if (k2 <= skipd)
                      {
                        if (verbose)
                          {
                            printf ("   skip gcd\n");
                            fflush (stdout);
                          }
                      }
                    else
                      {
                        st = GetTime ();
			if (m == 1 && q == 1 && f == 0.0 && skipd == r-1)
			  /* mimic irred */
			  certificate = get_certificate (a);
			/* --- EC: the cheap stopgap.  The proper check is a
			   Bezout certificate u*a + v*T = g, verified with four
			   folds -- it *proves* gcd(a,T)=1 rather than merely
			   agreeing with itself -- but that needs cofactors out
			   of HalfGCD2(..., extended=1).  See handoff §6.  --- */
			GF2X ec_agcd;
			if (ec_enabled && ec_gcdrep)
			  ec_agcd = a;
                        FastGCD(a, a, p);
			if (ec_enabled && ec_gcdrep)
			  {
			    GF2X ec_g2;
			    FastGCD (ec_g2, ec_agcd, p);
			    if (ec_g2 != a)
			      {
				fprintf (stderr, "EC: gcd disagreed with itself "
					 "for s=%ld at degree %ld\n", input_s, k);
				ec_detections++;
				FastGCD (a, ec_agcd, p);   /* third opinion */
			      }
			  }
                        if (verbose)
                          {
                            printf ("   gcd took %f\n", GetTime () - st);
			    fflush (stdout);
			  }
			if (deg(a) > 0)
			  {
			    /* check for GCD error */
			    if (deg(a) < k)
			      {
			      printf ("Error: GCD degree %ld<%ld for s=%ld\n",
				      deg(a), k, input_s);
			      exit (1);
			      }
			    /* exit the loop if we isolated a single factor
			       or q = 1 (we cannot refine any more)
			       or deg(a) < 10*sqrt(r) so CanZass cheap */
			    if (deg (a) < 2 * k || q == 1 ||
			        deg (a) < fineDDFtol)
			        break;
			    fineDDF = 1; /* enter fineDDF mode */
			    if (verbose)
			      printf ("Entering fine DDF mode\n");
			  }
                      }
		    }
		}
	    }

	  k1 = deg (a);
          /* if k1 = r, the trinomial is irreducible */
          if (rhigh < r)
            divisible = k1 != 0;
          else /* case rhigh = r: either we found a factor of degree < r,
                  or we went up to degree r */
            divisible = k1 < r;
	  if (divisible)
	    {
	      if (k1 >= 2 * k)
		{
		  vec_pair_GF2X_long factors;
		  st = GetTime ();
		  CanZass (factors, a, 0);
		  long i0 = 0;
		  // find lexicographically least factor of smallest degree
		  // (except if flipped)
		  for (long i = 1; i < factors.length(); i++)
		    if (lex_less(factors[i].a, factors[i0].a, flip))
		      i0 = i;
		  a = factors[i0].a; // the lex least factor of smallest
		  		     // degree (unless flipped)
		  if (verbose)
		    printf ("CanZass took %f, total degree %ld, small degree %ld\n",
			    GetTime () - st, k1, deg(a));
		  k1 = deg(a);       // the smallest degree
		  if (k1 < k)
		    {
		    printf ("Error: GCD degree %ld < %ld for s=%ld\n",
			    k1, k, input_s);
		    exit (1);
		    }
		}
              if (k1 != 0) /* k1=0 can happen with -skip option */
                print_hex (fact, a, flip);
	    }

#pragma omp critical
	  {
          if (!divisible || k1 == r)
            {
              if (skip)
                printf ("%ld u\n", ss);
              else if (primitive)
                printf ("%ld primitive\n", ss);
              else
	        printf ("%ld irreducible\n", ss);
            }
          else
            {
              if (k1 != 0)
                printf ("%ld %ld", ss, k1);
              else
                printf ("%ld non-irreducible (LOG %lx)", ss,
			(certificate ^ 0x2) & 0xFFFFFFFF);
	      if (k1 >= b0)
	        printf (" p%s", fact);
	      printf("\n");
            }
	  }
	  fflush (stdout);
	  free (fact);
        }

  free (todo);

  if (ec_enabled)
    fprintf (stderr, "Error checking: %ld detection(s) / recovery(ies)\n",
	     ec_detections);

  if (ec_enabled && ec_timing)
    {
      double tot = 0;
      for (int i = 0; i < ECT_N; i++) tot += ec_tsec[i];
      fprintf (stderr, "\nError-checking cost breakdown\n");
      fprintf (stderr, "  %-26s %10s %12s %10s\n",
	       "category", "calls", "seconds", "ms/call");
      for (int i = 0; i < ECT_N; i++)
	if (ec_tcnt[i])
	  fprintf (stderr, "  %-26s %10ld %12.3f %10.3f\n",
		   ec_tname[i], ec_tcnt[i], ec_tsec[i],
		   1e3 * ec_tsec[i] / (double) ec_tcnt[i]);
      fprintf (stderr, "  %-26s %10s %12.3f\n", "TOTAL", "", tot);
      fprintf (stderr, "\nIf any ms/call above looks absurd, that check is the\n"
	       "problem -- compare against ecbench for the same r and s.\n");
    }

  return 0;
}


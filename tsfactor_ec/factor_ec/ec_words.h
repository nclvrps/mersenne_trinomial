/* ec_words.h -- word-level primitives for the error-checking layer.
 *
 * Self-contained: no NTL, no gf2x, no CUDA.  Compiled into ecbench (timing)
 * and ectest (correctness), and mirrored by the GF2X-level code in factor_ec.
 * Keeping one definition means the thing that is timed and the thing that is
 * tested are the same thing.
 *
 * Convention throughout: a GF(2) polynomial is an array of `ecword` (64-bit),
 * bit i of the polynomial = bit (i % 64) of word (i / 64).  Little-endian
 * within the word, ascending word order.  This matches NTL's GF2X::xrep.
 */

#ifndef EC_WORDS_H
#define EC_WORDS_H

#include <stdint.h>
#include <string.h>

typedef uint64_t ecword;
#define ECW_BITS 64

/* ---------------------------------------------------------------- *
 *  1. cyclic fold  phi_n : GF(2)[x] -> GF(2)[x]/(x^n + 1)
 *     n = nfold * 64 bits, so the fold is a pure word XOR.
 * ---------------------------------------------------------------- */

static inline void
ec_fold_w (ecword *f, long nfold, const ecword *a, long na)
{
  memset (f, 0, (size_t) nfold * sizeof (ecword));
  long i = 0, j;
  for (; i + nfold <= na; i += nfold)
    for (j = 0; j < nfold; j++)
      f[j] ^= a[i + j];
  for (j = 0; i + j < na; j++)
    f[j] ^= a[i + j];
}

/* c = x * a  in GF(2)[x]/(x^n + 1): rotate left by one bit */
static inline void
ec_fold_shl1_w (ecword *c, long nfold, const ecword *a)
{
  const int TOP = ECW_BITS - 1;
  ecword carry = a[nfold - 1] >> TOP;
  for (long j = nfold - 1; j > 0; j--)
    c[j] = (a[j] << 1) | (a[j - 1] >> TOP);
  c[0] = (a[0] << 1) | carry;
}

/* ---------------------------------------------------------------- *
 *  2. bit reversal within an exact multiple of 64 bits
 *
 *     rev_D(f) = x^D * f(1/x),  D = 64*nw - 1.
 *     Coefficient at position D-i of the output = coefficient i of the
 *     input, i.e. reverse the whole 64*nw-bit string.  Because D+1 is a
 *     word multiple this is exactly "reverse the word order, reverse the
 *     bits in each word" -- no cross-word shifting, no alignment fixup.
 *
 *     rev_D is an involution on polynomials of degree <= D, and it is
 *     graded-multiplicative: rev_{a+b}(fg) = rev_a(f) * rev_b(g).
 * ---------------------------------------------------------------- */

static inline ecword
ec_revword (ecword v)
{
  v = ((v >> 1)  & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
  v = ((v >> 2)  & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
  v = ((v >> 4)  & 0x0f0f0f0f0f0f0f0fULL) | ((v & 0x0f0f0f0f0f0f0f0fULL) << 4);
  v = ((v >> 8)  & 0x00ff00ff00ff00ffULL) | ((v & 0x00ff00ff00ff00ffULL) << 8);
  v = ((v >> 16) & 0x0000ffff0000ffffULL) | ((v & 0x0000ffff0000ffffULL) << 16);
  return (v >> 32) | (v << 32);
}

/* out[0..nw-1] = rev_{64*nw-1}(in[0..nw-1]).  out and in may not alias. */
static inline void
ec_reverse_w (ecword *out, const ecword *in, long nw)
{
  for (long j = 0; j < nw; j++)
    out[nw - 1 - j] = ec_revword (in[j]);
}

/* ---------------------------------------------------------------- *
 *  3. reduction mod x^r + x^s + 1 by repeated block folding
 *
 *     One pass:  w = w_H * x^r + w_L  ->  w_L + w_H * x^s + w_H.
 *     The degree drops by (r - s) per pass, so this is only usable when
 *     (r - s) is a large fraction of r.  With deg w <= 2r-2 and 2s <= r
 *     it terminates in exactly 2 passes.
 *
 *     *** This is the routine whose misuse cost us a 45-minute hang. ***
 *     At r=136279841, s=136279253 (the working trinomial after factor's
 *     flip-to-enable-fastsqr) it needs 231769 passes, not 2.  Callers must
 *     route s > r/2 through the reciprocal -- see ec_divides_trinomial.
 * ---------------------------------------------------------------- */

/* Highest set bit index of w[0..nw-1], or -1 if zero. */
static inline long
ec_degree_w (const ecword *w, long nw)
{
  for (long t = nw - 1; t >= 0; t--)
    if (w[t])
      {
        ecword v = w[t];
        long b = ECW_BITS - 1;
        while (((v >> b) & 1) == 0) b--;
        return t * ECW_BITS + b;
      }
  return -1;
}

/* Reduce w (nw words) in place mod x^r + x^s + 1.  `hi` is caller-supplied
   scratch of nw words.  Returns the number of passes taken. */
static inline long
ec_reduce_lowdiff (ecword *w, long nw, long r, long s, ecword *hi)
{
  const long rw = r / ECW_BITS, rb = r % ECW_BITS;
  const long sw = s / ECW_BITS, sb = s % ECW_BITS;
  const long guard = 4 * (r / (r - s) + 2);        /* safety net */
  long passes = 0;

  for (;;)
    {
      long d = ec_degree_w (w, nw);
      if (d < r) return passes;

      const long hw = (d - r) / ECW_BITS + 1;      /* words of hi */

      /* hi = w >> r */
      if (rb == 0)
        memcpy (hi, w + rw, (size_t) hw * sizeof (ecword));
      else
        for (long j = 0; j < hw; j++)
          hi[j] = (w[rw + j] >> rb)
                | ((rw + j + 1 < nw) ? (w[rw + j + 1] << (ECW_BITS - rb)) : 0);

      /* w = lo = w mod x^r : keep bits 0..r-1, clear the rest */
      for (long j = rw + 1; j < nw; j++) w[j] = 0;
      w[rw] &= rb ? ((((ecword) 1) << rb) - 1) : (ecword) 0;

      /* w ^= hi   (from x^r = x^s + 1, the "+1" half) */
      for (long j = 0; j < hw; j++) w[j] ^= hi[j];

      /* w ^= hi << s   (the "x^s" half) */
      for (long j = 0; j < hw; j++)
        {
          if (hi[j] == 0) continue;
          const long dst = sw + j;
          if (dst < nw)            w[dst]     ^= sb ? (hi[j] << sb) : hi[j];
          if (sb && dst + 1 < nw)  w[dst + 1] ^= hi[j] >> (ECW_BITS - sb);
        }

      if (++passes > guard) return passes;
    }
}

/* ---------------------------------------------------------------- *
 *  4. divisibility test: does x^r + x^s + 1 divide w?
 *
 *     deg w < 2r.  Returns 1 if divisible.
 *
 *     For 2s <= r, reduce directly (2 passes).
 *     For 2s >  r, reverse and reduce modulo the reciprocal trinomial
 *     T* = x^r + x^(r-s) + 1, which then has the small exponent.  This is
 *     exact in both directions:
 *         T | w   <=>   T* | rev_D(w)      for any D >= deg w,
 *     because rev is graded-multiplicative and rev_r(T*) = T.
 * ---------------------------------------------------------------- */

static inline int
ec_divides_trinomial (const ecword *w, long nw, long r, long s,
                      ecword *scratch, ecword *scratch2)
{
  long j;
  for (j = 0; j < nw; j++) if (w[j]) break;
  if (j == nw) return 1;                     /* 0 is divisible by anything */

  const long sd = (2 * s <= r) ? s : (r - s);   /* the small exponent */

  if (2 * s <= r)
    memcpy (scratch, w, (size_t) nw * sizeof (ecword));
  else
    ec_reverse_w (scratch, w, nw);           /* rev_{64*nw-1} */

  ec_reduce_lowdiff (scratch, nw, r, sd, scratch2);

  for (j = 0; j < nw; j++) if (scratch[j]) return 0;
  return 1;
}


/* ---------------------------------------------------------------- *
 *  5. arithmetic inside the fold ring GF(2)[x]/(x^n + 1)
 *
 *     Needed by the tsfactor checks, where the shadow of a shift or of
 *     a multiplication by T has to be predicted from folds alone.
 *     phi is a ring hom on GF(2)[x], so for 0 <= k < n:
 *         phi(f * x^k) = phi(f) * x^k = rotate(phi(f), k)
 * ---------------------------------------------------------------- */

/* dst = a * x^k in GF(2)[x]/(x^n+1), n = nfold*64.  dst != a. */
static inline void
ec_fold_rot (ecword *dst, long nfold, const ecword *a, long k)
{
  const long n = nfold * ECW_BITS;
  k = ((k % n) + n) % n;
  const long kw = k / ECW_BITS, kb = k % ECW_BITS;
  for (long j = 0; j < nfold; j++)
    {
      const ecword lo = a[(j - kw + 2 * nfold) % nfold];
      const ecword hi = a[(j - kw - 1 + 2 * nfold) % nfold];
      dst[j] = kb ? ((lo << kb) | (hi >> (ECW_BITS - kb))) : lo;
    }
}

/* dst = a * phi(x^r + x^s + 1) -- three rotations, no general multiply */
static inline void
ec_fold_mul_trinomial (ecword *dst, long nfold, const ecword *a,
                       long r, long s)
{
  ecword t1[64], t2[64];                 /* nfold <= 64 by construction */
  ec_fold_rot (t1, nfold, a, r);
  ec_fold_rot (t2, nfold, a, s);
  for (long j = 0; j < nfold; j++) dst[j] = t1[j] ^ t2[j] ^ a[j];
}

/* dst = a * b in GF(2)[x]/(x^n+1).  Schoolbook over nfold words; at
   nfold = 64 that is 4096 word products, microseconds on a host CPU and
   negligible against a block.  Production may swap in PCLMUL. */
static inline void
ec_fold_mul_w (ecword *dst, long nfold, const ecword *a, const ecword *b)
{
  ecword acc[128];
  for (long j = 0; j < 2 * nfold; j++) acc[j] = 0;
  for (long i = 0; i < nfold; i++)
    {
      if (!a[i]) continue;
      for (long j = 0; j < nfold; j++)
        {
          if (!b[j]) continue;
          /* carry-less product of two words, bit by bit */
          ecword x = a[i], y = b[j], lo = 0, hi = 0;
          for (int t = 0; t < ECW_BITS; t++)
            if ((y >> t) & 1)
              {
                lo ^= (t ? (x << t) : x);
                if (t) hi ^= x >> (ECW_BITS - t);
              }
          acc[i + j]     ^= lo;
          acc[i + j + 1] ^= hi;
        }
    }
  /* fold x^n = 1 */
  for (long j = 0; j < nfold; j++) dst[j] = acc[j] ^ acc[j + nfold];
}

#endif /* EC_WORDS_H */

/* ectest2.cpp -- correctness tests for the new EC word primitives.
 *
 *   g++ -O2 -o ectest2 ectest2.cpp && ./ectest2
 *
 * Tests, all against an independent bit-level reference:
 *   1. ec_revword / ec_reverse_w   -- reversal is an involution and matches
 *                                     a naive bit-by-bit reversal
 *   2. ec_reduce_lowdiff           -- matches naive bit-serial reduction, and
 *                                     terminates in 2 passes when 2s <= r
 *   3. ec_divides_trinomial        -- agrees with naive division for both
 *                                     s < r/2 and s > r/2, on divisible and
 *                                     non-divisible inputs
 *   4. single-bit corruption       -- perturbing a divisible w by one bit is
 *                                     always detected
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include "ec_words.h"

typedef std::vector<unsigned char> bits;   /* reference rep: one byte per bit */

static bits from_words (const ecword *w, long nw)
{
  bits b ((size_t) nw * ECW_BITS, 0);
  for (long i = 0; i < nw * ECW_BITS; i++)
    b[i] = (w[i / ECW_BITS] >> (i % ECW_BITS)) & 1;
  return b;
}

static void to_words (ecword *w, long nw, const bits& b)
{
  memset (w, 0, (size_t) nw * sizeof (ecword));
  for (size_t i = 0; i < b.size () && (long) i < nw * ECW_BITS; i++)
    if (b[i]) w[i / ECW_BITS] |= ((ecword) 1) << (i % ECW_BITS);
}

static long ref_deg (const bits& a)
{
  for (long i = (long) a.size () - 1; i >= 0; i--) if (a[i]) return i;
  return -1;
}

/* naive bit-serial reduction mod x^r + x^s + 1, top down (fastrem's shape) */
static bits ref_reduce (bits a, long r, long s)
{
  for (long j = ref_deg (a); j >= r; j--)
    if (a[j]) { a[j] = 0; a[j - r] ^= 1; a[j - (r - s)] ^= 1; }
  return a;
}

/* schoolbook multiply */
static bits ref_mul (const bits& a, const bits& b)
{
  bits c (a.size () + b.size (), 0);
  for (size_t i = 0; i < a.size (); i++) if (a[i])
    for (size_t j = 0; j < b.size (); j++) if (b[j]) c[i + j] ^= 1;
  return c;
}

int main ()
{
  std::mt19937_64 rng (20260808);
  long fails = 0, checks = 0;

  /* ---- 1. reversal ---------------------------------------------------- */
  {
    long bad = 0;
    for (int t = 0; t < 20000; t++)
      {
        ecword v = rng (), rv = ec_revword (v);
        for (int b = 0; b < 64; b++)
          if (((v >> b) & 1) != ((rv >> (63 - b)) & 1)) bad++;
        if (ec_revword (rv) != v) bad++;
        checks++;
      }
    const long nw = 7;
    ecword a[nw], b[nw], c[nw];
    for (int t = 0; t < 20000; t++)
      {
        for (long j = 0; j < nw; j++) a[j] = rng ();
        ec_reverse_w (b, a, nw);
        ec_reverse_w (c, b, nw);
        for (long j = 0; j < nw; j++) if (a[j] != c[j]) bad++;
        bits ba = from_words (a, nw), bb = from_words (b, nw);
        for (size_t i = 0; i < ba.size (); i++)
          if (ba[i] != bb[ba.size () - 1 - i]) bad++;
        checks++;
      }
    printf ("1. reversal (involution + matches naive)      : %s (%ld bad)\n",
            bad ? "FAIL" : "PASS", bad);
    fails += bad;
  }

  /* ---- 2/3/4. reduction, divisibility, corruption --------------------- */
  {
    /* (r,s) pairs covering both regimes, including the production shape:
       r-s tiny (s > r/2) is exactly the case that hung factor.            */
    struct { long r, s; } cases[] = {
      {101, 7}, {101, 53}, {101, 97}, {127, 63}, {127, 64},
      {257, 12}, {257, 251}, {401, 399}, {401, 5}, {521, 489},
      {1279, 345}, {1279, 1275}, {4099, 4091}, {4099, 9},
    };
    long badred = 0, baddiv = 0, badcorr = 0, worstpass = 0;

    for (auto& C : cases)
      {
        const long r = C.r, s = C.s;
        const long nw = (2 * r + ECW_BITS - 1) / ECW_BITS + 1;
        std::vector<ecword> w (nw), sc (nw), sc2 (nw);

        for (int t = 0; t < 300; t++)
          {
            /* --- 2. reduction agrees with the naive reference --- */
            if (2 * s <= r)
              {
                bits bw ((size_t) nw * ECW_BITS, 0);
                for (long i = 0; i < 2 * r - 1; i++) bw[i] = rng () & 1;
                to_words (w.data (), nw, bw);
                long p = ec_reduce_lowdiff (w.data (), nw, r, s, sc.data ());
                if (p > 2) badred++;                 /* must be 2 passes */
                if (p > worstpass) worstpass = p;
                bits got = from_words (w.data (), nw);
                bits want = ref_reduce (bw, r, s);
                want.resize (got.size (), 0);
                if (got != want) badred++;
                checks++;
              }

            /* --- 3. divisibility, both directions --- */
            bits T ((size_t) r + 1, 0);
            T[0] = 1; T[s] = 1; T[r] = 1;
            bits q ((size_t) (r - 1), 0);
            for (long i = 0; i < r - 1; i++) q[i] = rng () & 1;

            bits prod = ref_mul (q, T);              /* deg <= 2r-3 */
            prod.resize ((size_t) nw * ECW_BITS, 0);
            to_words (w.data (), nw, prod);
            if (!ec_divides_trinomial (w.data (), nw, r, s,
                                       sc.data (), sc2.data ()))
              baddiv++;                              /* false alarm */
            checks++;

            /* --- 4. one flipped bit must always be caught --- */
            long pos = (long) (rng () % (ecword) (2 * r - 2));
            bits pert = prod;
            pert[pos] ^= 1;
            to_words (w.data (), nw, pert);
            if (ec_divides_trinomial (w.data (), nw, r, s,
                                      sc.data (), sc2.data ()))
              badcorr++;                             /* missed */
            checks++;

            /* --- 3b. a random non-multiple is (almost surely) rejected --- */
            bits rnd ((size_t) nw * ECW_BITS, 0);
            for (long i = 0; i < 2 * r - 2; i++) rnd[i] = rng () & 1;
            to_words (w.data (), nw, rnd);
            bits red = ref_reduce (rnd, r, s);
            bool truly_div = (ref_deg (red) < 0);
            bool said_div = ec_divides_trinomial (w.data (), nw, r, s,
                                                  sc.data (), sc2.data ());
            if (truly_div != said_div) baddiv++;
            checks++;
          }
      }
    printf ("2. reduce_lowdiff vs naive (2s<=r, <=2 passes): %s (%ld bad, "
            "max %ld passes)\n", badred ? "FAIL" : "PASS", badred, worstpass);
    printf ("3. divides_trinomial vs naive, both regimes   : %s (%ld bad)\n",
            baddiv ? "FAIL" : "PASS", baddiv);
    printf ("4. single-bit corruption of a multiple caught : %s (%ld missed)\n",
            badcorr ? "FAIL" : "PASS", badcorr);
    fails += badred + baddiv + badcorr;
  }

  printf ("\n%s -- %ld checks, %ld failures\n",
          fails ? "*** FAILURES ***" : "all tests PASS", checks, fails);
  return fails ? 1 : 0;
}

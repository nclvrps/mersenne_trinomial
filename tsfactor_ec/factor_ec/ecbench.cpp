/* ecbench.cpp -- time every error-checking primitive at production size.
 *
 *   g++ -O2 -o ecbench ecbench.cpp && ./ecbench [r] [s] [m] [window]
 *
 * Needs no NTL, no gf2x, no CUDA -- it only touches raw word arrays, which is
 * all the EC layer ever does.  Run this before a full factor run to find out
 * what the checking layer will cost, instead of discovering it 45 minutes in.
 *
 * Defaults to r = 136279841, s = 136279253: the *working* trinomial for the
 * -s0 588 run after factor's flip-to-enable-fastsqr.  Note s > r/2 -- that is
 * the normal case for this search, not an edge case, and it is what made the
 * round-1 ec_reduce_ref diverge.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <random>
#include "ec_words.h"

static double now (void)
{
  struct timeval tv;
  gettimeofday (&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

static const int REPS = 20;

int main (int argc, char **argv)
{
  const long r = (argc > 1) ? atol (argv[1]) : 136279841;
  const long s = (argc > 2) ? atol (argv[2]) : 136279841 - 588;
  const long m = (argc > 3) ? atol (argv[3]) : 14;
  const long W = (argc > 4) ? atol (argv[4]) : 8;

  const long nw1 = (r + ECW_BITS - 1) / ECW_BITS;
  const long nw2 = (2 * r + ECW_BITS - 1) / ECW_BITS + 1;
  const long NF  = 64;

  printf ("r = %ld, s = %ld  (r-s = %ld; s > r/2: %s)\n",
          r, s, r - s, (2 * s > r) ? "YES -> reciprocal path" : "no");
  printf ("m = %ld, Gerbicz window = %ld block(s)\n", m, W);
  printf ("residue %.1f MB (%ld words), product %.1f MB (%ld words)\n\n",
          nw1 * 8.0 / 1048576.0, nw1, nw2 * 8.0 / 1048576.0, nw2);

  ecword *a  = (ecword *) malloc (nw2 * sizeof (ecword));
  ecword *b  = (ecword *) malloc (nw2 * sizeof (ecword));
  ecword *sc = (ecword *) malloc (nw2 * sizeof (ecword));
  ecword *s2 = (ecword *) malloc (nw2 * sizeof (ecword));
  ecword  fold[64];
  if (!a || !b || !sc || !s2) { printf ("out of memory\n"); return 1; }

  /* Touch every page of every buffer before timing anything.  Without this,
     the first loop to write a fresh mmap pays the entire first-touch fault
     cost and reports ~7x slower than the steady state. */
  std::mt19937_64 rng (1);
  for (long j = 0; j < nw2; j++)
    { a[j] = rng (); b[j] = rng (); sc[j] = 1; s2[j] = 1; }

  double t0, t_fold2, t_fold1, t_copy, t_xor2, t_xor1, t_rev, t_div;

  t0 = now (); for (int i=0;i<REPS;i++) ec_fold_w (fold, NF, a, nw2);
  t_fold2 = (now()-t0)/REPS;
  t0 = now (); for (int i=0;i<REPS;i++) ec_fold_w (fold, NF, a, nw1);
  t_fold1 = (now()-t0)/REPS;
  t0 = now (); for (int i=0;i<REPS;i++) memcpy (b, a, nw2*sizeof(ecword));
  t_copy = (now()-t0)/REPS;
  t0 = now (); for (int i=0;i<REPS;i++)
    for (long j=0;j<nw2;j++) sc[j] = a[j] ^ b[j];
  t_xor2 = (now()-t0)/REPS;
  t0 = now (); for (int i=0;i<REPS;i++)
    for (long j=0;j<nw1;j++) sc[j] ^= a[j];
  t_xor1 = (now()-t0)/REPS;
  t0 = now (); for (int i=0;i<REPS;i++) ec_reverse_w (sc, a, nw2);
  t_rev = (now()-t0)/REPS;

  /* Build a genuine multiple w = q*(x^r + x^s + 1), else the reduction exits
     early on a short degree and the timing means nothing. */
  {
    memset (b, 0, nw2 * sizeof (ecword));
    for (long j = 0; j < nw1; j++) b[j] = rng ();            /* q, deg < r */
    memset (a, 0, nw2 * sizeof (ecword));
    for (long j = 0; j < nw1; j++) a[j] ^= b[j];
    const long rw = r/64, rb = r%64, sw = s/64, sb = s%64;
    for (long j = 0; j < nw1; j++)
      {
        if (rw+j   < nw2) a[rw+j]   ^= rb ? (b[j] << rb) : b[j];
        if (rb && rw+j+1 < nw2) a[rw+j+1] ^= b[j] >> (64-rb);
        if (sw+j   < nw2) a[sw+j]   ^= sb ? (b[j] << sb) : b[j];
        if (sb && sw+j+1 < nw2) a[sw+j+1] ^= b[j] >> (64-sb);
      }
  }

  int ok = 1;
  t0 = now ();
  for (int i=0;i<REPS;i++) ok &= ec_divides_trinomial (a, nw2, r, s, sc, s2);
  t_div = (now()-t0)/REPS;

  a[nw2/3] ^= 1;
  int caught = !ec_divides_trinomial (a, nw2, r, s, sc, s2);
  a[nw2/3] ^= 1;

  printf ("primitive costs\n");
  printf ("  fold phi(.), 2r bits                  %8.2f ms\n", t_fold2*1e3);
  printf ("  fold phi(.),  r bits                  %8.2f ms\n", t_fold1*1e3);
  printf ("  copy t before fastrem (2r bits)       %8.2f ms\n", t_copy*1e3);
  printf ("  xor, 2r bits                          %8.2f ms\n", t_xor2*1e3);
  printf ("  xor-accumulate, r bits                %8.2f ms\n", t_xor1*1e3);
  printf ("  bit-reverse, 2r bits                  %8.2f ms\n", t_rev*1e3);
  printf ("  ec_divides_trinomial (reverse+reduce) %8.2f ms   %s\n",
          t_div*1e3, ok ? "[multiple accepted]" : "[*** FALSE ALARM ***]");
  printf ("  same, one bit flipped                 %8s       %s\n", "",
          caught ? "[rejected, correct]" : "[*** MISSED ***]");

  const double per_mulmod = t_fold2 + 2*t_fold1 + t_copy + t_xor2 + t_div;
  const double per_block_digest = m * t_xor1;

  printf ("\naggregate\n");
  printf ("  per fastmulmod  %6.1f ms   (fold t,a,b + copy + xor + divisibility)\n",
          per_mulmod*1e3);
  printf ("  per block       %6.1f ms   (%ld digest xor-accumulates)\n",
          per_block_digest*1e3, m);

  printf ("\nprojection for -s0 588 -m 14 -q 15, interval 38..247\n");
  printf ("  measured unchecked: 41.9 s squares/products, 56.1 s gcd\n");
  const double ec_mul = 15 * per_mulmod;          /* ~15 calls in accumulate */
  const double ec_blk = 15 * per_block_digest;    /* 15 blocks               */
  const double ec_sig = (15.0 / W) * m * 0.0071;  /* m squarings @ ~7.1 ms   */
  printf ("  EC on ~15 fastmulmod calls   %6.2f s\n", ec_mul);
  printf ("  EC digest, 15 blocks         %6.2f s\n", ec_blk);
  printf ("  EC sigma, 15/%-2ld windows      %6.2f s\n", W, ec_sig);
  printf ("  -------------------------------------\n");
  printf ("  total                        %6.2f s  = %.1f%% of squares/products\n",
          ec_mul+ec_blk+ec_sig, 100.0*(ec_mul+ec_blk+ec_sig)/41.9);
  printf ("                                        = %.1f%% of interval wall clock\n",
          100.0*(ec_mul+ec_blk+ec_sig)/(41.9+56.1));

  printf ("\nround-1 comparison\n");
  const long badpasses = (2*s > r) ? (r-1)/(r-s)+1 : 2L;
  printf ("  old ec_reduce_ref took %ld reduction passes per fastmulmod\n",
          badpasses);
  printf ("  at ~%.0f ms per pass that is ~%.0f minutes for ONE call\n",
          (t_div-t_rev)*1e3/2, badpasses * ((t_div-t_rev)/2) / 60.0);
  printf ("  (and NTL's RightShift/LeftShift reallocate each pass, so the\n");
  printf ("   real figure was worse than this word-array estimate)\n");

  free (a); free (b); free (sc); free (s2);
  return 0;
}

/* irred.c                                      irred version forked from 3.15

***********************************************************
Forked version (2026):
    https://github.com/nclvrps/mersenne_trinomial irred subdirectory

Cleaned up with the assistance of Claude (chatbot)
to create a version suited for 64-bit machines
and modern optimizing compilers,
and to remove functionality now better performed by another program.

This forked version removes all sieving for polynomial factors,
and just does the irreducibility test for a single trinomial
x^r + x^s + 1, given as two command-line arguments r and s.

A 64-bit hexadecimal residue is printed if the trinomial is not
irreducible (version 0.14 and earlier printed only the low 32 bits;
the low 8 hex digits of the new residue are the old residue).

Sieving was removed because the GF2X package has the program "factor"
in the apps directory of its source code,
which is much better at finding polynomial factors.
(Not to be confused with the common GNU utility for factoring integers,
which is also called "factor")

For very large r where the "factor" program
has not found any polynomial factor for a trinomial with some value of s
after searching candidate polynomials up to a very large degree d,
it will be faster to use this program to do an irreducibility test
rather than continuing to seek a factor.

Nonetheless, if this test does confirm non-irreducibility, it is desirable
to take the time (eventually) to find a factor, for ease of verification.

For r equal to any Mersenne prime exponent up to 74207281,
all of the irreducible trinomials have already been found by Brent et al.
in 2016 and earlier.

For all non-irreducible trinomials (millions of them),
polynomials factors were found and recorded in certificate files.

Those certificate files are linked from:
https://maths-people.anu.edu.au/~brent/trinom.html

The check-ntl program is linked from the same page,
and it can be used to check non-irreducibility very quickly
by verifying the known factors listed in the certificate files.

Richard Brent's page about the original version of this program is at:
https://maths-people.anu.edu.au/~brent/irred.html

The original version 3.15 of irred can be downloaded from:
https://maths-people.anu.edu.au/~brent/ftp/trinom/irred315.tar.gz

***********************************************************

==========================================================================
|                                                                        |
|  Copyright (C) 2003 R. P. Brent.                                       |
|                                                                        |
|  This program is free software; you can redistribute it and/or         |
|  modify it under the terms of the GNU General Public License,          |
|  version 2, June 1991, as published by the Free Software Foundation.   |
|  For details see http://www.gnu.org/copyleft/gpl.html .                |
|                                                                        |
==========================================================================

Author: Richard P. Brent (rpb@comlab.ox.ac.uk)
        with assistance and contributions from
        Samuli Larvala and Paul Zimmermann

Dates:  20000528..20030328 (version 3.15)

Summary:

   An efficient test for irreducibility of trinomials
   x^r + x^s + 1 (mod 2) where r is a prime,
   or primitivity in the case that 2^r-1 is a Mersenne prime,
   using 64-bit word-length bit operations.

   The test computes x^(2^r) mod A(x) by r squarings; A(x) is irreducible
   iff the result is x.  If s is even, the reciprocal trinomial
   x^r + x^(r-s) + 1 (irreducible iff the original one is) is used
   instead, so that the middle exponent "sodd" is always odd.

Restrictions, Assumptions and Limitations:

   Assumes r odd and r - sodd >= LIM = 2*64 = 128, where sodd is whichever
   of s, r-s is odd; i.e. s >= 128 if s is even, r - s >= 128 if s is odd.
   (Version 0.14 and earlier required r - sodd >= 384.)

   It is assumed that r is prime - if not true then a warning is printed.
   There is no check on whether or not 2^r-1 is prime.

   When output says A(x) is "irreducible" this actually means that
   x^(2^r) = x mod A(x). If r is not prime, we should also check
   that GCD(x^(2^d) - x, A(x)) = 1 for divisors d of r, but the program
   does not do this.

   Similarly, if r is prime but 2^r-1 is not prime, then an
   irreducible A(x) might not be primitive, because x^((2^r-1)/p) mod A(x)
   is not checked for any prime divisors p of 2^r-1. (In this case we
   can not complete the primitivity test without factoring 2^r-1.)

Usage:
        irred [options] r s

        Tests x^r + x^s + 1.  Options (anywhere on the command line):

        -v, --verbose            more informative output
        --progress N             print an interim residue every N squarings
        --no-checkpoint, --no-ckpt
                                 no checkpoint files (and no resume)
        --checkpoint-mins M, --ckpt-mins M
                                 checkpoint every M minutes (default 30;
                                 0 = only when stopped by a signal)
        --selftest               test small known cases, with and without
                                 injected faults, and exit
        --threads N              threads per squaring (default 1).  Needs
                                 a build with -fopenmp; used only for
                                 r > about 5 million.  Measure 1..cores:
                                 1.7x with 2 at r = 136279841 on a Xeon.
        --nt                     non-temporal (streaming) stores in the
                                 AVX2/AVX-512 interleave.  2x slower where
                                 the data fit in the L3 cache; may help where
                                 they do not (Zen 2 at r = 136279841?):
                                 measure.
        --isa auto|avx512|avx2|generic
                                 which interleave code to use; the default
                                 picks the best the CPU has (for testing)
        -h, --help               usage

        Error checking (see "Error checking" below):
        --no-ec                  no Gerbicz check (as in 0.20)
        --ec-block L             squarings per block (default sqrt(r)/16,
                                 at least 32)
        --ec-window N            blocks per window (default L); windows
                                 also close at every checkpoint and signal
        --ec-retries N           consecutive failed checks allowed at the
                                 same point before giving up (default 3)
        --map-checks N           random inputs for the map check at start
                                 (default 16; 0 = none)
        --seed N                 seed for the map check's inputs
        --ec-fault SITE:N        TEST: flip one bit at the N-th opportunity
                                 of SITE: sq (after a squaring), acc (the
                                 accumulator), chk (inside a check), ver
                                 (the verified state), or map (a linear
                                 bug in every squaring, which the chain
                                 cannot see but the map check can)

Output:

        One result line, in the format of the old log files:

          r s irreducible (may be primitive) LOG
          r s xHHHHHHHHHHHHHHHH LOG      (not irreducible, s odd)
          r s yHHHHHHHHHHHHHHHH LOG      (not irreducible, s even)

        where HHHHHHHHHHHHHHHH is the low 64 bits (coefficients of
        x^63 .. x^0) of x^(2^r) mod (x^r + x^sodd + 1), in hex.  "y" marks
        that the reciprocal trinomial x^r + x^(r-s) + 1 was used.

        With --progress N, after every N squarings a line
          iter k/r ... residue HHHHHHHHHHHHHHHH
        gives the low 64 bits of x^(2^k) mod (x^r + x^sodd + 1).

        Before the result, a line says whether all r squarings were
        error-checked (a run resumed from a 0.20 checkpoint, or run with
        --no-ec, reports how many were not).

        Exit status: 0 = finished, 1 = error, 2 = stopped by SIGINT or
        SIGTERM after writing a checkpoint (run the same command to resume),
        3 = an error check failed persistently, or the map check failed.

Error checking:

        Over GF(2), squaring is additive: (u + v)^2 = u^2 + v^2.  So the
        Gerbicz accumulator can be a plain XOR: a window starting at a
        verified state u(0) is made of n blocks of L squarings, D is the XOR
        of u(0), u(L), ..., u((n-1)L), and the window is correct iff
        D^(2^L) = D + u(0) + u(nL), which costs L squarings to test.  Any
        single fault in a squaring, or damage to the state in memory, is
        always detected (not just with high probability); the window is
        then redone from u(0).  Checkpoints are only ever written from
        verified states, and the verified state's CRC, taken from the words
        that were compared, is checked again before it is written or used.
        Cost: about 1/L + 1/N of the squarings (0.3% at r = 136279841).

        The chain only shows that each squaring agrees with the code's own
        idea of squaring.  A deterministic bug that computes another
        GF(2)-linear map (like the reduction bug of 0.14) passes it.  So at
        start the fast code is also compared with a simple independent
        implementation on random inputs (the map check): a wrong linear map
        disagrees on at least half of all inputs, so 16 inputs miss it with
        probability at most 2^-16.

Checkpoints:

        Unless disabled, the state is saved to irred_r<r>_s<s>.ckpt in the
        current directory every 30 minutes (--checkpoint-mins) and when the
        program receives SIGINT or SIGTERM (a second signal kills it at
        once).  The two previous checkpoints are kept as .ckpt.1 and
        .ckpt.2 (.ckpt is always the newest).  On start, the program looks
        for these files and resumes from the newest valid one that is for
        the same r and s; files for another (r, s), from another program,
        or failing their CRC check are ignored with a message.  So the
        command that started a run also resumes it.  When the run finishes,
        its checkpoint files are deleted.  irred_r<r>_s<s>.lock keeps a
        second process from working on the same (r, s) in the same
        directory at the same time.  With error checking (the default), a
        checkpoint holds the last verified state, and a signal first
        finishes and verifies the current block.  Checkpoints written by
        0.20 (format version 1) are read too; 0.20 cannot read the version 2
        files written by this version.

Comments on timing and memory requirements:

        Each trinomial takes time O(r^2).
        Space requirement is about 3r/4 bytes (six buffers of r/8 bytes:
        two for the squarings, one for the verified state, three for the
        error check), 100 MB at r = 136279841; 3r/8 bytes with --no-ec.

References:

        R. P. Brent, Search for primitive trinomials (mod 2),
        https://maths-people.anu.edu.au/~brent/trinom.html

        R. P. Brent, S. Larvala and P. Zimmermann, A fast algorithm for
        testing irreducibility of trinomials mod 2 (preliminary report),
        Report PRG-TR-13-00, Oxford University Computing Laboratory,
        30 Dec 2000. Revision in Mathematics of Computation 72 (2003),
        1443-1452.
        See https://maths-people.anu.edu.au/~brent/pub/pub199.html

        S. W. Golomb, Shift register sequences, Holden-Day, San Francisco,
        1967. Revised edition, Aegean Park Press, 1982.

        J. W. Heringa, H. W. J. Blöte and A. Compagner, New primitive
        trinomials of Mersenne-exponent degrees for random-number generation,
        Int. Journal of Modern Physics C 3 (1992), 561-564.

        D. E. Knuth, The Art of Computer Programming, Vol. 2, third ed.,
        Sec. 3.2.2.

        T. Kumada, H. Leeb, Y. Kurita and M. Matsumoto, New primitive
        t-nomials (t = 3, 5) over GF(2) whose degree is a Mersenne exponent,
        Mathematics of Computation 69 (2000), 811-814.
        Corrigenda: ibid 71 (2002), 1337-1338.

        N. Zierler, Primitive trinomials whose degree is a Mersenne exponent,
        Information and Control 15 (1969), 67-69.

Comments on different versions:

    The history of versions up to 3.15 has been moved to a separate file:
    OLD_VERSION_HISTORY, because much of it is no longer relevant.

    Forked version 0.20 (2026-11-17):
      - command line is just "irred [options] r s" (restart file, log
        file, time limit, skip file and stdin input removed)
      - 64-bit residue
      - checkpointing (see above) and --progress
      - fixed the reduction when (r - sodd) = 0 (mod 128): versions up to
        0.14 folded the wrong range of words in that case and printed
        wrong residues (and could have missed an irreducible trinomial)
      - the in-place reduction is done in blocks that are free of
        read-after-write overlap, so the compiler may vectorise it
        legitimately (0.14 declared overlapping pointers __restrict__)
      - AVX-512BW and AVX2 versions of the interleave, chosen at run time
        (--isa); the interleave always runs forwards now.  (A BMI2 PDEP
        version was tried: faster than portable C, slower than AVX2, and
        every CPU with BMI2 has AVX2, so it was not kept.)
      - on a 2.1 GHz Xeon (AVX-512, 260 MB L3), compiled with
        gcc 13 -O3 -march=native: r = 756839 takes 3.3 s instead of 15.4 s;
        r = 136279841 takes about 1.8 ms per squaring (2.9 days) instead
        of 3.7 ms (5.9 days)
      - removed the timing run in fastmem, the r = 127 (mod 128) warning
        (that case is handled correctly) and other obsolete code

    Forked version 0.21 (2026-11-19):
      - Gerbicz-style error checking (XOR accumulator, see above), on by
        default, with fault injection (--ec-fault) and exit status 3
      - map check of the squaring code against an independent
        implementation at start
      - checkpoint format version 2 (error-checking bookkeeping); version
        1 checkpoints from 0.20 are still read, so a 0.20 run continues
      - checkpoints are written only from verified states; on SIGINT or
        SIGTERM the program finishes the current block and verifies first
        (a few seconds at r = 136279841)
      - slice-by-8 CRC for the polynomial data
      - --threads N: each squaring split between N threads (OpenMP)
      - --nt: optional streaming stores in the SIMD interleave
      - github location moved to nclvrps/mersenne_trinomial (irred/)

Compile with, e.g.,
        gcc -O3 -march=native -fopenmp -o irred irred.c -lm
(-fopenmp only matters for --threads; without it the program is the same,
single-threaded.)
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define IRRED_VERSION "0.21"
#define IRRED_DATE    "20261119"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_X86_SIMD 1
#include <immintrin.h>
#else
#define HAVE_X86_SIMD 0
#endif

#define WLEN  64                /* bits in a word */
#define WLENM (WLEN-1)
#define WD    6                 /* WLEN = 2^WD */
#define LIM   (2*WLEN)          /* need r - sodd >= LIM */

#define CKPT_DEFAULT_MINS 30.0

/* ------------------------------------------------------------------------ */
/* Parameters depending on r and sodd (set by setup_params) */

static int r, s, sodd;
static int alpha;               /* (r-1)/2 */
static int q11;                 /* (r-1) div WLEN: index of the top word */
static int q4;                  /* alpha div WLEN */
static int q41;                 /* (alpha+1) div WLEN */
static int s1_x, s2_x;          /* (alpha+1) mod WLEN, WLENM - s1_x */
static int delta, deltaw, deltaq, deltaqc;  /* (r-sodd)/2, div, mod WLEN */
static uint64_t mask1;          /* valid bits of word q11 */
static uint64_t mask2;          /* bits >= alpha+1 of word q4 */
static size_t nwords;           /* q11 + 1: words in a polynomial */
static size_t bufwords;         /* words per work buffer (excluding pad) */

#define FRONTPAD 8              /* words before each buffer (a[-1] safe) */

static bool verbose = false;

static void setup_params(void)
  {
  sodd = (s & 1) ? s : r - s;           /* whichever of s, r-s is odd */
  alpha = r >> 1;
  q11 = (r - 1) >> WD;
  q4 = alpha >> WD;
  q41 = (alpha + 1) >> WD;
  s1_x = (alpha + 1) & WLENM;
  s2_x = WLENM - s1_x;
  delta = (r - sodd) >> 1;
  deltaw = delta >> WD;
  deltaq = delta & WLENM;
  deltaqc = WLEN - deltaq;              /* WLEN if deltaq == 0 (special) */
  mask1 = UINT64_MAX >> (WLENM - ((r - 1) & WLENM));
  mask2 = (~(uint64_t)1) << (alpha & WLENM);
  nwords = (size_t)q11 + 1;
  size_t need = (size_t)q11 + 3;        /* reducep zeroes q11+1, q11+2 */
  if ((size_t)(2 * q41 + 2) > need) need = 2 * (size_t)q41 + 2;
  bufwords = (need + 7) & ~(size_t)7;
  }

/* ------------------------------------------------------------------------ */
/* Squaring modulo the trinomial.

   A polynomial a of degree < r is held in words a[0..q11].  One squaring
   step is reducep (which reduces the implicit square of a, leaving the
   result in the "scrambled" order 0 2 4 ... r-1 1 3 ... r-2) followed by an
   interleave into the other buffer, which restores the normal order.

   In the scrambled order, bit i >= alpha+1 stands for x^(2i), which is
   reduced to x^(2i-r) (the same bit position) plus x^(2i-r+sodd), at bit
   i - delta.  Going down from the top, word T receives the bits of words
   T+deltaw and T+deltaw+1 -- which must already have received theirs (the
   reduction cascades), so the work runs downwards and any block of at most
   deltaw consecutive target words reads only words outside the block.
   fold_* work on one such block, so their restrict qualifiers are true. */

/* With GCC/clang on x86-64 Linux, also compile AVX2 and AVX-512 clones of
   these two loops, selected at load time (matters only when not compiling
   with -march=native or similar). */
#if HAVE_X86_SIMD && defined(__linux__) && !defined(NO_CLONES)
#define FOLD_ATTR __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define FOLD_ATTR __attribute__((noinline))
#endif

FOLD_ATTR
static void fold_shift(uint64_t *restrict d, const uint64_t *restrict src,
                       long n, int sh)
/* d[i] ^= bits of src[i..i+1] shifted down by sh (0 < sh < WLEN), i < n */
  {
  const int shc = WLEN - sh;
  for (long i = 0; i < n; i++)
    d[i] ^= (src[i] >> sh) | (src[i + 1] << shc);
  }

FOLD_ATTR
static void fold_word(uint64_t *restrict d, const uint64_t *restrict src,
                      long n)
/* d[i] ^= src[i], i < n */
  {
  for (long i = 0; i < n; i++)
    d[i] ^= src[i];
  }

static inline void reduce_fold(uint64_t *a, long t0, long n)
/* one piece of a block: target words t0 .. t0+n-1 */
  {
  if (n <= 0) return;
  if (deltaq != 0)
    fold_shift(a + t0, a + t0 + deltaw, n, deltaq);
  else
    fold_word(a + t0, a + t0 + deltaw, n);
  }

static inline void reduce_tail(uint64_t *a)
/* The last source word q4 is special: only its bits >= alpha+1 count */
  {
  uint64_t w = a[q4] & mask2;
  if (deltaq != 0) {
    a[q4 - deltaw] ^= (w >> deltaq) | (a[q4 + 1] << deltaqc);
    a[q4 - deltaw - 1] ^= w << deltaqc;   /* index may be -1: then w << deltaqc
                                             is 0 (mask2) and FRONTPAD absorbs it */
    }
  else
    a[q4 - deltaw] ^= w;
  }

static void reducep(uint64_t *a)

/* Reduces implicit square of polynomial a of degree < r "in place"
   given x^r = x^sodd + 1 (mod 2), result left in scrambled order
   0 2 4 ... 1 3 5 ...
   Assumes ODD(r), ODD(sodd), r - sodd >= 2*WLEN (so deltaw >= 1).
   RPB, 20000815; blocked 2026. */

  {
  a[q11] &= mask1;              /* mask any irrelevant high bits */
  a[q11 + 1] = 0;
  a[q11 + 2] = 0;

  /* targets T = q11-deltaw down to q4+1-deltaw, in blocks of <= deltaw */
  long top = (long)q11 - deltaw;
  const long low = (long)q4 + 1 - deltaw;
  while (top >= low) {
    long n = top - low + 1;
    if (n > deltaw) n = deltaw;
    long t0 = top - n + 1;
    reduce_fold(a, t0, n);
    top = t0 - 1;
    }
  reduce_tail(a);
  }

/* Spread the low 32 bits of x to the even bit positions of the result */
static inline uint64_t spread32(uint64_t x)
  {
  x &= 0x00000000FFFFFFFFULL;
  x = (x | x << 16) & 0x0000FFFF0000FFFFULL;
  x = (x | x << 8)  & 0x00FF00FF00FF00FFULL;
  x = (x | x << 4)  & 0x0F0F0F0F0F0F0F0FULL;
  x = (x | x << 2)  & 0x3333333333333333ULL;
  x = (x | x << 1)  & 0x5555555555555555ULL;
  return x;
  }

/* One step of the interleave: bits 0, 2, 4, ... , r-3, r-1, 1, 3, 5, ...,
   r-4, r-2 in a are moved to b in the correct order.  The "hi" word is
   built with two left shifts because s2_x may be 63. */
#define ILV_STEP(j)                                                       \
  do {                                                                    \
    const uint64_t lo = a[(j)];                                           \
    const uint64_t hi = (a[(j) + h] >> s1) |                              \
                        ((a[(j) + h + 1] << 1) << s2);                    \
    b[2 * (j)]     = spread32(lo)       | (spread32(hi)       << 1);      \
    b[2 * (j) + 1] = spread32(lo >> 32) | (spread32(hi >> 32) << 1);      \
  } while (0)

static void interlvf(const uint64_t *restrict a, uint64_t *restrict b,
                     long j0, long j1)
/* Interleave, loop index runs up (RPB, 20000907).  The reverse-direction
   version interlvr was dropped in 0.20: alternating directions is slower
   on current machines (about 10% at r = 756839 and at r = 136279841). */
  {
  const long h = q41;
  const int s1 = s1_x, s2 = s2_x;
  for (long j = j0; j <= j1; j++)
    ILV_STEP(j);
  }

#if HAVE_X86_SIMD
/* AVX2 version of the interleave, four words of lo/hi at a time.  Each
   byte is split into nibbles, a nibble is spread to the even (lo) or odd
   (hi) bits of a byte by table lookup (vpshufb), and the bytes of the two
   results are interleaved (vpunpck[lh]bw), which within each 128-bit lane
   gives the output for one 64-bit input word per 16 bytes. */

#define ILV_AVX2_BLOCK(j, ST)                                             \
  do {                                                                    \
    const __m256i L  = _mm256_loadu_si256((const __m256i *)(a + (j)));    \
    const __m256i H0 = _mm256_loadu_si256((const __m256i *)(a + (j) + h));\
    const __m256i H1 = _mm256_loadu_si256((const __m256i *)(a + (j) + h + 1)); \
    const __m256i H  = _mm256_or_si256(_mm256_srl_epi64(H0, c1),          \
                                       _mm256_sll_epi64(H1, c2));         \
    const __m256i Ll = _mm256_and_si256(L, nib);                          \
    const __m256i Lh = _mm256_and_si256(_mm256_srli_epi16(L, 4), nib);    \
    const __m256i Hl = _mm256_and_si256(H, nib);                          \
    const __m256i Hh = _mm256_and_si256(_mm256_srli_epi16(H, 4), nib);    \
    const __m256i A  = _mm256_or_si256(_mm256_shuffle_epi8(T0, Ll),       \
                                       _mm256_shuffle_epi8(T1, Hl));      \
    const __m256i B  = _mm256_or_si256(_mm256_shuffle_epi8(T0, Lh),       \
                                       _mm256_shuffle_epi8(T1, Hh));      \
    const __m256i U  = _mm256_unpacklo_epi8(A, B);                        \
    const __m256i V  = _mm256_unpackhi_epi8(A, B);                        \
    ST((__m256i *)(b + 2 * (j)), _mm256_permute2x128_si256(U, V, 0x20));  \
    ST((__m256i *)(b + 2 * (j) + 4), _mm256_permute2x128_si256(U, V, 0x31)); \
  } while (0)

#define ILV_AVX2_SETUP                                                    \
  const long h = q41;                                                     \
  const int s1 = s1_x, s2 = s2_x;                                         \
  const __m128i c1 = _mm_cvtsi32_si128(s1);                               \
  const __m128i c2 = _mm_cvtsi32_si128(s2 + 1);  /* 64 gives 0: fine */   \
  const __m256i nib = _mm256_set1_epi8(0x0F);                             \
  const __m256i T0 = _mm256_setr_epi8(                                    \
      0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,                     \
      0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,                     \
      0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,                     \
      0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55);                    \
  const __m256i T1 = _mm256_add_epi8(T0, T0)     /* odd bit positions */

#define DEFINE_ILV_AVX2(NAME, ST, FENCE)                                  \
__attribute__((target("avx2")))                                           \
static void NAME(const uint64_t *restrict a, uint64_t *restrict b,        \
                 long j0, long j1)                                        \
  {                                                                       \
  ILV_AVX2_SETUP;                                                         \
  long j = j0;                                                            \
  for (; j + 3 <= j1; j += 4)                                             \
    ILV_AVX2_BLOCK(j, ST);                                                \
  for (; j <= j1; j++)                                                    \
    ILV_STEP(j);                                                          \
  FENCE;                                                                  \
  }
DEFINE_ILV_AVX2(interlvf_avx2, _mm256_storeu_si256, (void)0)
/* --nt: non-temporal (streaming) stores; b + 2j is 32-byte aligned because
   b is 64-byte aligned and j0 is even (0, or a multiple of 8 with threads) */
DEFINE_ILV_AVX2(interlvf_avx2_nt, _mm256_stream_si256, _mm_sfence())

#endif

#if HAVE_X86_SIMD
/* AVX-512BW version: the same method, eight words of lo/hi at a time.  The
   byte interleave works within 128-bit lanes, so a final cross-lane
   permutation puts the eight 128-bit results in order. */
__attribute__((target("avx512f,avx512bw"), always_inline))
static inline void ilv512_body(const uint64_t *restrict a, uint64_t *restrict b,
                               long j0, long j1, const bool nt)
  {
  const long h = q41;
  const int s1 = s1_x, s2 = s2_x;
  const __m128i c1 = _mm_cvtsi32_si128(s1);
  const __m128i c2 = _mm_cvtsi32_si128(s2 + 1);  /* 64 gives 0: fine */
  const __m512i nib = _mm512_set1_epi8(0x0F);
  const __m512i T0 = _mm512_broadcast_i32x4(_mm_setr_epi8(
      0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
      0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55));
  const __m512i T1 = _mm512_add_epi8(T0, T0);    /* odd bit positions */
  const __m512i P0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
  const __m512i P1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
  long j = j0;
  for (; j + 7 <= j1; j += 8) {
    const __m512i L  = _mm512_loadu_si512((const void *)(a + j));
    const __m512i H0 = _mm512_loadu_si512((const void *)(a + j + h));
    const __m512i H1 = _mm512_loadu_si512((const void *)(a + j + h + 1));
    const __m512i H  = _mm512_or_si512(_mm512_srl_epi64(H0, c1),
                                       _mm512_sll_epi64(H1, c2));
    const __m512i Ll = _mm512_and_si512(L, nib);
    const __m512i Lh = _mm512_and_si512(_mm512_srli_epi16(L, 4), nib);
    const __m512i Hl = _mm512_and_si512(H, nib);
    const __m512i Hh = _mm512_and_si512(_mm512_srli_epi16(H, 4), nib);
    const __m512i A  = _mm512_or_si512(_mm512_shuffle_epi8(T0, Ll),
                                       _mm512_shuffle_epi8(T1, Hl));
    const __m512i B  = _mm512_or_si512(_mm512_shuffle_epi8(T0, Lh),
                                       _mm512_shuffle_epi8(T1, Hh));
    const __m512i U  = _mm512_unpacklo_epi8(A, B);
    const __m512i V  = _mm512_unpackhi_epi8(A, B);
    const __m512i O0 = _mm512_permutex2var_epi64(U, P0, V);
    const __m512i O1 = _mm512_permutex2var_epi64(U, P1, V);
    if (nt) {             /* b + 2j is 64-byte aligned: j0 % 8 == 0 */
      _mm512_stream_si512((void *)(b + 2 * j), O0);
      _mm512_stream_si512((void *)(b + 2 * j + 8), O1);
      }
    else {
      _mm512_storeu_si512((void *)(b + 2 * j), O0);
      _mm512_storeu_si512((void *)(b + 2 * j + 8), O1);
      }
    }
  for (; j <= j1; j++)
    ILV_STEP(j);
  }

__attribute__((target("avx512f,avx512bw")))
static void interlvf_avx512(const uint64_t *restrict a, uint64_t *restrict b,
                            long j0, long j1)
  {
  ilv512_body(a, b, j0, j1, false);
  }

__attribute__((target("avx512f,avx512bw")))
static void interlvf_avx512_nt(const uint64_t *restrict a, uint64_t *restrict b,
                               long j0, long j1)
  {
  ilv512_body(a, b, j0, j1, true);
  _mm_sfence();
  }
#endif

typedef void (*ilv_fn)(const uint64_t *restrict, uint64_t *restrict,
                       long, long);
static ilv_fn ilv = interlvf;
static bool ilv_nt = false;             /* --nt */
static const char *ilv_name = "portable C";

static bool comparex(const uint64_t *a)
/* Returns true if poly a of degree < r is x */
  {
  if (a[0] != 2) return false;
  for (size_t j = 1; j < (size_t)q11; j++)
    if (a[j] != 0) return false;
  if (q11 > 0 && (a[q11] & mask1) != 0) return false;
  return true;
  }

/* Choose the interleave: isa is "auto", "avx512", "avx2" or "generic".
   Returns false if the requested one is not available. */
static bool choose_interleave(const char *isa)
  {
  bool want512 = !strcmp(isa, "auto") || !strcmp(isa, "avx512");
  bool want2 = !strcmp(isa, "auto") || !strcmp(isa, "avx2");
  if (!want512 && !want2 && strcmp(isa, "generic") != 0) return false;
#if HAVE_X86_SIMD
  __builtin_cpu_init();
  if (want512 && __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw")) {
    ilv = ilv_nt ? interlvf_avx512_nt : interlvf_avx512;
    ilv_name = ilv_nt ? "AVX-512BW (streaming stores)" : "AVX-512BW";
    return true;
    }
  if (want2 && __builtin_cpu_supports("avx2") && strcmp(isa, "avx512") != 0) {
    ilv = ilv_nt ? interlvf_avx2_nt : interlvf_avx2;
    ilv_name = ilv_nt ? "AVX2 (streaming stores)" : "AVX2";
    return true;
    }
#endif
  return strcmp(isa, "auto") == 0 || strcmp(isa, "generic") == 0;
  }

/* ------------------------------------------------------------------------ */
/* Utilities */

static double wall_now(void)
  {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
  }

static double cpu_now(void)
  {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
  }

static void fmt_duration(double sec, char *buf, size_t n)
  {
  if (sec < 0) sec = 0;
  if (sec < 120)
    snprintf(buf, n, "%.1f s", sec);
  else if (sec < 7200)
    snprintf(buf, n, "%.1f min", sec / 60);
  else if (sec < 2 * 86400)
    snprintf(buf, n, "%.2f h", sec / 3600);
  else
    snprintf(buf, n, "%.2f days", sec / 86400);
  }

static bool prime(int n)
/* Returns true if n is prime. Simple and not intended to be efficient. */
  {
  if (n <= 1) return false;
  if (n <= 3) return true;
  if ((n & 1) == 0) return false;
  for (long j = 3; j <= n / j; j += 2)
    if (n % j == 0) return false;
  return true;
  }

static uint64_t *alloc_buffer(void)
/* One zeroed work buffer of bufwords words with FRONTPAD words before it */
  {
  size_t bytes = (bufwords + FRONTPAD + 8) * sizeof(uint64_t);
  size_t align = bytes >= (4u << 20) ? (2u << 20) : 64;  /* huge pages */
  bytes = (bytes + align - 1) & ~(align - 1);
  void *p = NULL;
  if (posix_memalign(&p, align, bytes) != 0) {
    fprintf(stderr, "unable to allocate %zu bytes\n", bytes);
    exit(EXIT_FAILURE);
    }
#ifdef MADV_HUGEPAGE
  if (align > 64) madvise(p, bytes, MADV_HUGEPAGE);
#endif
  memset(p, 0, bytes);
  return (uint64_t *)p + FRONTPAD;
  }

/* ------------------------------------------------------------------------ */
/* CRC-32 (IEEE, as in zlib).  Byte-wise for headers; slice-by-8, one 64-bit
   word at a time, for polynomials (identical results on little-endian
   machines, checked by --selftest). */

static uint32_t crc_tab[8][256];

static void crc_init(void)
  {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++)
      c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    crc_tab[0][i] = c;
    }
  for (int k = 1; k < 8; k++)
    for (int i = 0; i < 256; i++)
      crc_tab[k][i] = (crc_tab[k - 1][i] >> 8) ^ crc_tab[0][crc_tab[k - 1][i] & 0xFF];
  }

static uint32_t crc32_buf(const void *buf, size_t len)
  {
  const unsigned char *p = buf;
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++)
    c = crc_tab[0][(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
  }

/* One little-endian 64-bit word into a running (inverted) CRC state */
static inline uint32_t crc_word(uint32_t c, uint64_t w)
  {
  uint64_t x = w ^ c;
  return crc_tab[7][x & 0xFF] ^ crc_tab[6][(x >> 8) & 0xFF] ^
         crc_tab[5][(x >> 16) & 0xFF] ^ crc_tab[4][(x >> 24) & 0xFF] ^
         crc_tab[3][(x >> 32) & 0xFF] ^ crc_tab[2][(x >> 40) & 0xFF] ^
         crc_tab[1][(x >> 48) & 0xFF] ^ crc_tab[0][x >> 56];
  }

/* CRC of a polynomial in canonical form (words 0..q11, top word masked):
   the same value as the checkpoint data CRC */
static uint32_t poly_crc(const uint64_t *a)
  {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < (size_t)q11; i++)
    c = crc_word(c, a[i]);
  c = crc_word(c, a[q11] & mask1);
  return c ^ 0xFFFFFFFFu;
  }

static void poly_copy(uint64_t *restrict dst, const uint64_t *restrict src)
/* canonical copy: words 0..q11, top word masked */
  {
  memcpy(dst, src, (size_t)q11 * sizeof(uint64_t));
  dst[q11] = src[q11] & mask1;
  }

/* ------------------------------------------------------------------------ */
/* Fault injection, for testing the error checks (--ec-fault SITE:N) */

enum { F_SQ, F_ACC, F_CHK, F_VER, F_MAP, F_NSITES };
static const char *const fault_names[F_NSITES] =
  { "sq", "acc", "chk", "ver", "map" };
static int fault_site = -1;             /* none */
static uint64_t fault_n = 0;            /* at the fault_n-th opportunity */
static uint64_t fault_seen[F_NSITES];

static void maybe_fault(int site, uint64_t *p)
/* Flips one bit of polynomial p at the fault_n-th call for this site */
  {
  if (site != fault_site || ++fault_seen[site] != fault_n) return;
  size_t w = (size_t)((fault_n * 2654435761u) % (uint64_t)q11);  /* < q11 */
  int bit = (int)((fault_n * 40503u) % WLEN);
  p[w] ^= (uint64_t)1 << bit;
  printf("[ec-fault: flipped bit %d of word %zu at %s #%" PRIu64 "]\n",
         bit, w, fault_names[site], fault_n);
  }

/* One squaring: src (consumed) -> dst.  The "map" fault simulates a
   deterministic, LINEAR bug in the squaring code (like the one in 0.14):
   the Gerbicz check cannot see such a bug, the map check can. */
/* Threads (--threads N, needs OpenMP).  One parallel region per squaring:
   each block of the reduction is split between the threads (a block's
   target words never overlap its source words), with a barrier between
   blocks; then the interleave, split by j.  Used only when the polynomial
   is big enough for it to pay. */
static int nthreads = 1;
static long mt_min_words = 40000;       /* threads only above this q41 */
static long mt_min_block = 2048;        /* reduction blocks in parallel above */

#ifdef _OPENMP
static void square_mt(uint64_t *a, uint64_t *b)
  {
  a[q11] &= mask1;
  a[q11 + 1] = 0;
  a[q11 + 2] = 0;
  const long low = (long)q4 + 1 - deltaw;
  const bool par_reduce = deltaw >= mt_min_block;
#pragma omp parallel num_threads(nthreads)
  {
    const int tt = omp_get_thread_num(), nt = omp_get_num_threads();
    if (par_reduce) {
      long top = (long)q11 - deltaw;
      while (top >= low) {
        long n = top - low + 1;
        if (n > deltaw) n = deltaw;
        long t0 = top - n + 1;
        long lo = n * tt / nt, hi = n * (tt + 1) / nt;
        reduce_fold(a, t0 + lo, hi - lo);
        top = t0 - 1;
#pragma omp barrier
        }
#pragma omp single
      reduce_tail(a);
      }
    else {
#pragma omp single
      {
      long top = (long)q11 - deltaw;
      while (top >= low) {
        long n = top - low + 1;
        if (n > deltaw) n = deltaw;
        long t0 = top - n + 1;
        reduce_fold(a, t0, n);
        top = t0 - 1;
        }
      reduce_tail(a);
      }
      }
    /* (implicit barrier at the end of each single) */
    /* pieces start at multiples of 8 (aligned stores, whole cache lines) */
    long n = (long)q41 + 1;
    long lo = (n * tt / nt) & ~7L;
    long hi = (tt == nt - 1) ? n : (n * (tt + 1) / nt) & ~7L;
    if (hi > lo) ilv(a, b, lo, hi - 1);
  }
  }
#endif

static inline void square(uint64_t *src, uint64_t *dst)
  {
#ifdef _OPENMP
  if (nthreads > 1 && (long)q41 >= mt_min_words)
    square_mt(src, dst);
  else
#endif
    {
    reducep(src);
    ilv(src, dst, 0, q41);
    }
  if (fault_site == F_MAP) dst[3] ^= dst[1];
  }

/* ------------------------------------------------------------------------ */
/* Map check: the fast squaring (reducep + interleave, with whatever SIMD
   code was chosen) against an independent, simple implementation (byte
   table spread, then word-wise reduction of the double-length square from
   the top), on random polynomials.  The squaring map is GF(2)-linear, so a
   deterministic bug makes it a different linear map, which disagrees with
   the true one on at least half of all inputs: n random inputs miss it with
   probability at most 2^-n. */

static inline void xor_at(uint64_t *c, uint64_t w, long pos)
/* c ^= w * x^pos */
  {
  long wi = pos >> WD;
  int sh = (int)(pos & WLENM);
  c[wi] ^= w << sh;
  if (sh) c[wi + 1] ^= w >> (WLEN - sh);
  }

static void ref_square(const uint64_t *u, uint64_t *c)
/* c = u^2 mod (x^r + x^sodd + 1); c needs 2*nwords+1 words; canonical u */
  {
  static uint16_t sp8[256];
  if (sp8[255] == 0)
    for (int i = 0; i < 256; i++) {
      uint16_t v = 0;
      for (int b = 0; b < 8; b++)
        if (i >> b & 1) v |= (uint16_t)(1u << (2 * b));
      sp8[i] = v;
      }
  for (size_t i = 0; i < nwords; i++) {
    uint64_t w = (i == (size_t)q11) ? u[i] & mask1 : u[i];
    uint64_t lo = 0, hi = 0;
    for (int b = 0; b < 4; b++) {
      lo |= (uint64_t)sp8[(w >> (8 * b)) & 0xFF] << (16 * b);
      hi |= (uint64_t)sp8[(w >> (32 + 8 * b)) & 0xFF] << (16 * b);
      }
    c[2 * i] = lo;
    c[2 * i + 1] = hi;
    }
  c[2 * nwords] = 0;
  const long rw = r >> WD, rb = r & WLENM;          /* r is odd: rb > 0 */
  for (long i = (2L * r - 2) >> WD; i > rw; i--) {  /* whole words >= x^r */
    uint64_t w = c[i];
    if (w == 0) continue;
    c[i] = 0;
    long e0 = WLEN * i - r;                         /* x^(64i) = x^e0 x^r */
    xor_at(c, w, e0);
    xor_at(c, w, e0 + sodd);
    }
  uint64_t w = c[rw] >> rb;                         /* bits r, r+1, ... */
  c[rw] &= ((uint64_t)1 << rb) - 1;
  xor_at(c, w, 0);
  xor_at(c, w, sodd);
  }

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void)
  {
  uint64_t x = rng_state;               /* xorshift64* */
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
  rng_state = x;
  return x * 0x2545F4914F6CDD1DULL;
  }

static bool map_check(uint64_t *A, uint64_t *B, int n)
/* A, B: work buffers (contents destroyed).  Returns true if all agree. */
  {
  uint64_t *in = malloc(nwords * sizeof(uint64_t));
  uint64_t *ref = malloc((2 * nwords + 2) * sizeof(uint64_t));
  if (in == NULL || ref == NULL) {
    fprintf(stderr, "unable to allocate memory for the map check\n");
    exit(EXIT_FAILURE);
    }
  bool ok = true;
  for (int t = 0; t < n && ok; t++) {
    memset(A, 0, bufwords * sizeof(uint64_t));
    for (size_t i = 0; i < nwords; i++)
      A[i] = rng_next();
    if (t == 0) {                       /* one sparse input: x^(r-1) + x */
      memset(A, 0, nwords * sizeof(uint64_t));
      A[0] = 2;
      A[q11] = (uint64_t)1 << ((r - 1) & WLENM);
      }
    A[q11] &= mask1;
    memcpy(in, A, nwords * sizeof(uint64_t));
    square(A, B);
    ref_square(in, ref);
    for (size_t i = 0; i < (size_t)q11; i++)
      if (B[i] != ref[i]) ok = false;
    if (((B[q11] ^ ref[q11]) & mask1) != 0) ok = false;
    }
  free(in);
  free(ref);
  return ok;
  }

/* ------------------------------------------------------------------------ */
/* Checkpoints

   File irred_r<r>_s<s>.ckpt: a header, then the polynomial x^(2^k) mod A(x)
   in normal order, words 0..q11 (top word masked).  Header and data each
   carry a CRC-32.  Written to <file>.tmp, fsync'ed, the older generations
   rotated (.ckpt -> .ckpt.1 -> .ckpt.2), then renamed into place.

   Format version 2 (0.21) adds error-checking bookkeeping to the header.
   Version 1 files (0.20) are still read, so a run started by 0.20 can be
   continued by this version. */

#define CKPT_MAGIC   "IRREDCKP"         /* 8 bytes, no terminating NUL used */
#define CKPT_VERSION 2u

struct ckpt_hdr_v1 {                    /* irred 0.20 */
  char     magic[8];
  uint32_t version;
  uint32_t hdr_bytes;
  uint64_t r, s, sodd;
  uint64_t k;
  uint64_t nwords;
  double   wall;
  double   cpu;
  int64_t  written;
  uint32_t data_crc;
  uint32_t hdr_crc;
};

struct ckpt_hdr {                       /* version 2 */
  char     magic[8];
  uint32_t version;
  uint32_t hdr_bytes;           /* sizeof(struct ckpt_hdr) */
  uint64_t r, s, sodd;
  uint64_t k;                   /* squarings done */
  uint64_t nwords;              /* words of data that follow */
  double   wall;                /* wall-clock seconds in all runs so far */
  double   cpu;                 /* CPU seconds in all runs so far */
  int64_t  written;             /* time(NULL) when written */
  uint64_t unchecked;           /* of the k squarings, how many were done
                                   without error checking (0.20, --no-ec) */
  uint64_t ec_checks;           /* windows verified, all runs */
  uint64_t ec_failures;         /* failed checks (recovered), all runs */
  uint64_t reserved[5];
  uint32_t data_crc;
  uint32_t hdr_crc;             /* CRC of all preceding header bytes */
};

static bool ckpt_enabled = true;
static double ckpt_secs = CKPT_DEFAULT_MINS * 60.0;
static char ckpt_name[3][128];  /* .ckpt, .ckpt.1, .ckpt.2 */
static char ckpt_tmp[128], lock_name[128];
static volatile sig_atomic_t stop_requested = 0;

static void on_signal(int sig)
  {
  if (!stop_requested) {        /* first signal: stop at the next safe point */
    stop_requested = sig;
    return;
    }
  signal(sig, SIG_DFL);         /* second signal: die at once */
  raise(sig);
  }

static void install_signals(void)
  {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGTERM, &sa, NULL);
  }

static void ckpt_setup_names(void)
  {
  snprintf(ckpt_name[0], sizeof ckpt_name[0], "irred_r%d_s%d.ckpt", r, s);
  snprintf(ckpt_name[1], sizeof ckpt_name[1], "irred_r%d_s%d.ckpt.1", r, s);
  snprintf(ckpt_name[2], sizeof ckpt_name[2], "irred_r%d_s%d.ckpt.2", r, s);
  snprintf(ckpt_tmp, sizeof ckpt_tmp, "irred_r%d_s%d.ckpt.tmp", r, s);
  snprintf(lock_name, sizeof lock_name, "irred_r%d_s%d.lock", r, s);
  }

static void ckpt_lock(void)
/* Keep a second process off the same (r, s) in this directory.  The flock
   ends with the process however it ends, so a stale lock file is harmless. */
  {
  int fd = open(lock_name, O_RDWR | O_CREAT, 0644);
  if (fd < 0) {
    fprintf(stderr, "Warning: cannot create lock file %s (%s)\n",
            lock_name, strerror(errno));
    return;
    }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    fprintf(stderr, "Another irred process is already testing r = %d, s = %d "
            "in this directory (%s is locked)\n", r, s, lock_name);
    exit(EXIT_FAILURE);
    }
  /* fd deliberately kept open for the life of the process */
  }

/* Bookkeeping carried from run to run in the checkpoint header */
struct history {
  double wall, cpu;             /* earlier runs */
  uint64_t unchecked;           /* squarings done without error checking */
  uint64_t ec_checks, ec_failures;
};

/* Writes polynomial a (state after k squarings).  If want_crc is nonzero,
   a must have that CRC (computed when it was verified); otherwise a was
   damaged in memory since, and nothing is written. */
static bool ckpt_write(const uint64_t *a, uint64_t k, uint32_t want_crc,
                       bool check_crc, const struct history *h)
  {
  double t0 = wall_now();
  struct ckpt_hdr H;
  memset(&H, 0, sizeof H);
  memcpy(H.magic, CKPT_MAGIC, 8);
  H.version = CKPT_VERSION;
  H.hdr_bytes = sizeof H;
  H.r = r; H.s = s; H.sodd = sodd;
  H.k = k;
  H.nwords = nwords;
  H.wall = h->wall;
  H.cpu = h->cpu;
  H.written = (int64_t)time(NULL);
  H.unchecked = h->unchecked;
  H.ec_checks = h->ec_checks;
  H.ec_failures = h->ec_failures;
  H.data_crc = poly_crc(a);
  if (check_crc && H.data_crc != want_crc) {
    fprintf(stderr, "[checkpoint NOT written: the verified state was damaged "
            "in memory after it was verified (CRC %08x, expected %08x)]\n",
            H.data_crc, want_crc);
    return false;
    }
  H.hdr_crc = crc32_buf(&H, offsetof(struct ckpt_hdr, hdr_crc));

  uint64_t top = a[q11] & mask1;
  FILE *f = fopen(ckpt_tmp, "wb");
  if (f == NULL) {
    fprintf(stderr, "[checkpoint FAILED: cannot create %s: %s]\n",
            ckpt_tmp, strerror(errno));
    return false;
    }
  bool ok = fwrite(&H, sizeof H, 1, f) == 1 &&
            fwrite(a, sizeof(uint64_t), nwords - 1, f) == nwords - 1 &&
            fwrite(&top, sizeof top, 1, f) == 1;
  ok = ok && fflush(f) == 0 && fsync(fileno(f)) == 0;
  if (fclose(f) != 0) ok = false;
  if (!ok) {
    fprintf(stderr, "[checkpoint FAILED: error writing %s: %s]\n",
            ckpt_tmp, strerror(errno));
    remove(ckpt_tmp);
    return false;
    }
  remove(ckpt_name[2]);                 /* rotate the older generations */
  rename(ckpt_name[1], ckpt_name[2]);
  rename(ckpt_name[0], ckpt_name[1]);
  if (rename(ckpt_tmp, ckpt_name[0]) != 0) {
    fprintf(stderr, "[checkpoint FAILED: cannot rename %s to %s: %s]\n",
            ckpt_tmp, ckpt_name[0], strerror(errno));
    return false;
    }
  int dfd = open(".", O_RDONLY);        /* make the renames durable */
  if (dfd >= 0) { fsync(dfd); close(dfd); }
  fprintf(stderr, "[checkpoint: %" PRIu64 "/%d squarings (%.2f%%)%s -> %s, "
          "%.2f s]\n", k, r, 100.0 * (double)k / r,
          check_crc ? ", verified" : "", ckpt_name[0], wall_now() - t0);
  return true;
  }

/* Try to load one checkpoint file into a.  Returns true if it is a valid
   checkpoint for this (r, s); otherwise prints why it is not used.  Reads
   format versions 1 and 2; H is filled in as version 2. */
static bool ckpt_load_one(const char *name, uint64_t *a, struct ckpt_hdr *H)
  {
  FILE *f = fopen(name, "rb");
  if (f == NULL) return false;          /* silently: no such file */
  unsigned char raw[sizeof(struct ckpt_hdr)];
  struct ckpt_hdr_v1 H1;
  memset(H, 0, sizeof *H);
  size_t got = fread(raw, 1, 16, f);
  if (got != 16 || memcmp(raw, CKPT_MAGIC, 8) != 0) {
    printf("Ignoring %s: not an irred checkpoint file\n", name);
    fclose(f);
    return false;
    }
  uint32_t ver, hb;
  memcpy(&ver, raw + 8, 4);
  memcpy(&hb, raw + 12, 4);
  bool ok = false;
  if (ver == 1 && hb == sizeof H1 &&
      fread(raw + 16, 1, sizeof H1 - 16, f) == sizeof H1 - 16) {
    memcpy(&H1, raw, sizeof H1);
    if (crc32_buf(&H1, offsetof(struct ckpt_hdr_v1, hdr_crc)) == H1.hdr_crc) {
      memcpy(H->magic, H1.magic, 8);
      H->version = 1;
      H->r = H1.r; H->s = H1.s; H->sodd = H1.sodd;
      H->k = H1.k; H->nwords = H1.nwords;
      H->wall = H1.wall; H->cpu = H1.cpu; H->written = H1.written;
      H->unchecked = H1.k;              /* 0.20 had no error checking */
      H->data_crc = H1.data_crc;
      ok = true;
      }
    }
  else if (ver == CKPT_VERSION && hb == sizeof *H &&
           fread(raw + 16, 1, sizeof *H - 16, f) == sizeof *H - 16) {
    memcpy(H, raw, sizeof *H);
    ok = crc32_buf(H, offsetof(struct ckpt_hdr, hdr_crc)) == H->hdr_crc;
    }
  if (!ok) {
    printf("Ignoring %s: unknown format version or damaged header\n", name);
    fclose(f);
    return false;
    }
  if (H->r != (uint64_t)r || H->s != (uint64_t)s) {
    printf("Ignoring %s: it is for r = %" PRIu64 ", s = %" PRIu64
           ", not r = %d, s = %d\n", name, H->r, H->s, r, s);
    fclose(f);
    return false;
    }
  if (H->sodd != (uint64_t)sodd || H->nwords != nwords || H->k > (uint64_t)r ||
      H->unchecked > H->k) {
    printf("Ignoring %s: inconsistent header\n", name);
    fclose(f);
    return false;
    }
  memset(a, 0, bufwords * sizeof(uint64_t));
  ok = fread(a, sizeof(uint64_t), nwords, f) == nwords &&
       poly_crc(a) == H->data_crc;
  fclose(f);
  if (!ok) {
    printf("Ignoring %s: data truncated or damaged (CRC mismatch)\n", name);
    memset(a, 0, bufwords * sizeof(uint64_t));
    return false;
    }
  return true;
  }

/* Loads the newest valid checkpoint for this (r, s) into a.  Returns false
   if there is none.  Fills *k and *h. */
static bool ckpt_resume(uint64_t *a, uint64_t *k, struct history *h)
  {
  struct ckpt_hdr H;
  for (int g = 0; g < 3; g++) {
    if (ckpt_load_one(ckpt_name[g], a, &H)) {
      char when[64] = "", took[32];
      time_t t = (time_t)H.written;
      struct tm tmv;
      if (localtime_r(&t, &tmv))
        strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", &tmv);
      fmt_duration(H.wall, took, sizeof took);
      printf("Resuming from %s (written %s): %" PRIu64 "/%d squarings done "
             "(%.2f%%) in %s, residue %016" PRIx64 "\n", ckpt_name[g], when,
             H.k, r, 100.0 * (double)H.k / r, took, a[0]);
      if (H.version == 1)
        printf("  (a version 1 checkpoint from irred 0.20: its %" PRIu64
               " squarings were done without error checking)\n", H.k);
      else if (H.unchecked > 0)
        printf("  (%" PRIu64 " of these squarings were done without error "
               "checking)\n", H.unchecked);
      *k = H.k;
      h->wall = H.wall;
      h->cpu = H.cpu;
      h->unchecked = H.unchecked;
      h->ec_checks = H.ec_checks;
      h->ec_failures = H.ec_failures;
      return true;
      }
    }
  return false;
  }

static void ckpt_remove_all(void)
  {
  for (int g = 0; g < 3; g++) remove(ckpt_name[g]);
  remove(ckpt_tmp);
  remove(lock_name);
  }

/* ------------------------------------------------------------------------ */
/* The test, with Gerbicz-style error checking

   The squarings form the chain u(t+1) = u(t)^2 mod A(x), u(0) = x.  Over
   GF(2), squaring is the Frobenius map F, which is additive as well as
   multiplicative: F(u + v) = F(u) + F(v).  So the Gerbicz accumulator can
   be a sum (XOR) instead of a product.  A window starts at a verified
   state u(0) (kept as V) and consists of n blocks of L squarings; after
   each block the current u is added into D, which starts as u(0):

       D = u(0) + u(L) + ... + u((n-1)L)
       F^L(D) = u(L) + ... + u(nL),  so  F^L(D) = D + u(0) + u(nL).

   Closing the window costs L squarings of a copy of D and one comparison.
   If a fault changes u by e at some step t of the window, every later u is
   off by the corresponding power of F applied to e (squaring is linear),
   the sum telescopes, and the two sides differ by F^(jL-t)(e) for the first
   block end jL after t: never zero, wherever the fault was.  A single fault
   is always detected; faults in D or in the check itself cause a false
   alarm, which costs a retry.  On failure the window is redone from V.

   What it cannot see: a deterministic bug that makes the squaring code
   compute some other LINEAR map.  The chain is then consistent with that
   map and every window passes.  The map check above covers that case. */

enum { RES_IRRED, RES_RESIDUE, RES_STOPPED, RES_ECFAIL };

static bool ec_enabled = true;
static long ec_block = 0;               /* L; 0 = automatic */
static long ec_window = 0;              /* blocks per window; 0 = automatic */
static int ec_retries = 3;              /* consecutive failures allowed */
static int map_checks = 16;

static uint64_t *Vb, *Db, *S1, *S2;     /* verified state, accumulator, scratch */
static uint32_t crcV;                   /* CRC of V when it was verified */

struct ecstat {
  uint64_t checks, failures, sq_check;
  double t_check;
};
static struct ecstat ecs;

static void poly_xor(uint64_t *restrict d, const uint64_t *restrict s)
  {
  for (size_t i = 0; i <= (size_t)q11; i++)
    d[i] ^= s[i];
  }

/* Closes a window: checks F^Lb(D) == D + V + u.  On success sets V = u
   (and D = V), so the next window starts from it. */
static bool ec_close(const uint64_t *u, long Lb)
  {
  double t0 = wall_now();
  ecs.checks++;
  ecs.sq_check += (uint64_t)Lb;
  poly_copy(S1, Db);
  uint64_t *x = S1, *y = S2;
  for (long i = 0; i < Lb; i++) {
    square(x, y);
    uint64_t *t = x; x = y; y = t;
    if (i == 0) maybe_fault(F_CHK, x);
    }
  uint64_t diff = 0;
  uint32_t c = 0xFFFFFFFFu;
  for (size_t j = 0; j < (size_t)q11; j++) {
    const uint64_t w = u[j];
    diff |= x[j] ^ Db[j] ^ Vb[j] ^ w;
    c = crc_word(c, w);
    }
  const uint64_t w = u[q11] & mask1;
  diff |= (x[q11] ^ Db[q11] ^ Vb[q11] ^ w) & mask1;
  c = crc_word(c, w);
  const uint32_t crc_u = c ^ 0xFFFFFFFFu;
  bool ok = (diff == 0);
  if (ok) {
    /* V = the state just verified; its CRC was taken from the very words
       that were compared, so a damaged copy is caught here. */
    int tries = 0;
    do {
      poly_copy(Vb, u);
    } while (poly_crc(Vb) != crc_u && ++tries < 3);
    if (tries == 3) ok = false;
    else {
      crcV = crc_u;
      poly_copy(Db, Vb);
      maybe_fault(F_VER, Vb);
      }
    }
  ecs.t_check += wall_now() - t0;
  return ok;
  }

static long auto_block(void)
  {
  long L = (long)(sqrt((double)r) / 16.0);
  return L < 32 ? 32 : L;
  }

/* Runs squarings from state V (after k0 squarings; CRC crcV) to r, with
   error checking, checkpoints and progress output as configured.  Returns
   RES_* and the residue in *res (from the verified final state). */
static int run_test(uint64_t *buf[2], uint64_t k0, long progress,
                    struct history *h, uint64_t *res)
  {
  uint64_t *a = buf[0], *b = buf[1];
  const double tstart = wall_now(), cstart = cpu_now();
  double last_ckpt = tstart, last_prog_t = tstart;
  uint64_t last_prog_k = k0;
  const struct history h0 = *h;
  const long L = ec_block > 0 ? ec_block : auto_block();
  const long N = ec_window > 0 ? ec_window : L;
  bool stop_msg = false;

  poly_copy(a, Vb);
  if (ec_enabled) poly_copy(Db, Vb);

  uint64_t k = k0, kv = k0;     /* squarings done; at V */
  long Lw = (r - k < (uint64_t)L) ? (long)(r - k) : L;  /* this window's block */
  long pos = 0, nblk = 0;       /* squarings into the block; blocks done */
  int fails_here = 0;           /* consecutive failures */

  /* history as it would be written now (state V after kv squarings) */
#define HIST_NOW(hh) do { (hh) = h0; double now_ = wall_now();              \
    (hh).wall = h0.wall + (now_ - tstart);                                   \
    (hh).cpu = h0.cpu + (cpu_now() - cstart);                                \
    (hh).ec_checks = h0.ec_checks + ecs.checks - ecs.failures;               \
    (hh).ec_failures = h0.ec_failures + ecs.failures; } while (0)

  while (k < (uint64_t)r) {
    square(a, b);
    uint64_t *t = a; a = b; b = t;
    k++;
    maybe_fault(F_SQ, a);

    if (progress > 0 && k % (uint64_t)progress == 0) {
      double now = wall_now();
      double per = (k > last_prog_k) ? (now - last_prog_t) / (double)(k - last_prog_k)
                                     : 0;
      char eta[32];
      fmt_duration(per * (double)((uint64_t)r - k), eta, sizeof eta);
      printf("iter %" PRIu64 "/%d (%6.2f%%)  %.3g ms/iter  ETA %s  "
             "residue %016" PRIx64, k, r, 100.0 * (double)k / r,
             1e3 * per, eta, a[0]);
      if (ec_enabled) printf("  (verified to %" PRIu64 ")", kv);
      printf("\n");
      last_prog_t = now;
      last_prog_k = k;
      }

    if (!ec_enabled) {          /* 0.20 behaviour: state is not verified */
      bool due = ckpt_enabled && ckpt_secs > 0 && (k & 63) == 0 &&
                 k < (uint64_t)r && wall_now() - last_ckpt >= ckpt_secs;
      if (stop_requested || due) {
        if (stop_requested)
          printf("Received signal %d after %" PRIu64 "/%d squarings\n",
                 (int)stop_requested, k, r);
        struct history hh;
        HIST_NOW(hh);
        hh.unchecked = h0.unchecked + (k - k0);
        bool w = ckpt_enabled && ckpt_write(a, k, 0, false, &hh);
        last_ckpt = wall_now();
        if (stop_requested) {
          if (w) printf("Checkpoint written; run the same command again to "
                        "continue\n");
          return RES_STOPPED;
          }
        }
      continue;
      }

    if (stop_requested && !stop_msg) {
      printf("Received signal %d after %" PRIu64 "/%d squarings: finishing "
             "the block and verifying before stopping\n",
             (int)stop_requested, k, r);
      stop_msg = true;
      }

    if (++pos < Lw) continue;

    /* end of a block: close the window, or add u(k) to D and go on */
    pos = 0;
    nblk++;
    bool due = ckpt_enabled && ckpt_secs > 0 &&
               wall_now() - last_ckpt >= ckpt_secs;
    bool tail = (uint64_t)r - k < (uint64_t)Lw;   /* includes k == r */
    if (!(nblk >= N || tail || due || stop_requested)) {
      poly_xor(Db, a);
      maybe_fault(F_ACC, Db);
      continue;
      }

    if (!ec_close(a, Lw)) {
      ecs.failures++;
      fails_here++;
      printf("ERROR CHECK FAILED for squarings %" PRIu64 "..%" PRIu64
             " (%ld blocks of %ld): hardware or software fault; %s\n",
             kv + 1, k, nblk, Lw,
             fails_here > ec_retries ? "giving up" : "redoing them");
      if (fails_here > ec_retries) {
        printf("The check failed %d times in a row at the same point: "
               "persistent fault.\nThe last checkpoint (if any) holds the "
               "last verified state.\n", fails_here);
        return RES_ECFAIL;
        }
      if (poly_crc(Vb) != crcV) {
        /* the verified state itself was damaged in memory */
        printf("The verified state at %" PRIu64 " was damaged in memory; "
               "falling back to the last checkpoint\n", kv);
        struct history hh;
        uint64_t kk = 0;
        if (ckpt_enabled && ckpt_resume(Vb, &kk, &hh))
          kv = kk;
        else {
          printf("No checkpoint to fall back on: restarting from the "
                 "beginning\n");
          memset(Vb, 0, bufwords * sizeof(uint64_t));
          Vb[0] = 2;
          kv = 0;
          }
        crcV = poly_crc(Vb);
        }
      poly_copy(a, Vb);
      poly_copy(Db, Vb);
      k = kv;
      nblk = 0;
      Lw = ((uint64_t)r - k < (uint64_t)L) ? (long)((uint64_t)r - k) : L;
      continue;
      }

    /* window verified: V = u(k) */
    if (verbose)
      printf("[verified squarings %" PRIu64 "..%" PRIu64 " (%ld x %ld)]\n",
             kv + 1, k, nblk, Lw);
    kv = k;
    fails_here = 0;
    nblk = 0;
    if ((uint64_t)r - k < (uint64_t)L) Lw = (long)((uint64_t)r - k);

    if (k < (uint64_t)r && (due || stop_requested)) {
      struct history hh;
      HIST_NOW(hh);
      bool w = ckpt_enabled && ckpt_write(Vb, kv, crcV, true, &hh);
      last_ckpt = wall_now();
      if (stop_requested) {
        if (w) printf("Checkpoint written; run the same command again to "
                      "continue\n");
        return RES_STOPPED;
        }
      }
    }
#undef HIST_NOW

  /* the result comes from the verified copy (or from the working buffer,
     which holds the same words, if the copy was damaged since) */
  const uint64_t *fin = ec_enabled ? Vb : a;
  if (ec_enabled && poly_crc(Vb) != crcV) {
    if (poly_crc(a) == crcV) {
      printf("The copy of the verified final state was damaged in memory; "
             "using the working copy, which is intact\n");
      fin = a;
      }
    else {
      printf("The verified final state was damaged in memory; run again\n");
      return RES_ECFAIL;
      }
    }
  if (comparex(fin)) return RES_IRRED;
  *res = fin[0];
  return RES_RESIDUE;
  }

/* ------------------------------------------------------------------------ */

static int check_params(void)
/* 0 if (r, s) can be tested, otherwise prints a message and returns 1 */
  {
  if (r < 2 * LIM + 1 || !(r & 1)) {
    printf("r must be odd and at least %d\n", 2 * LIM + 1);
    return 1;
    }
  if (s < 1 || s >= r) {
    printf("s must satisfy 0 < s < r\n");
    return 1;
    }
  int so = (s & 1) ? s : r - s;
  if (r - so < LIM) {
    printf("Need r - sodd >= %d, where sodd = %d is whichever of s, r-s is "
           "odd\n(i.e. s >= %d if s is even, r - s >= %d if s is odd); "
           "use another program for this case\n", LIM, so, LIM, LIM);
    return 1;
    }
  return 0;
  }

static void alloc_all(uint64_t *buf[2])
  {
  buf[0] = alloc_buffer();
  buf[1] = alloc_buffer();
  Vb = alloc_buffer();
  Db = S1 = S2 = NULL;
  if (ec_enabled) {
    Db = alloc_buffer();
    S1 = alloc_buffer();
    S2 = alloc_buffer();
    }
  }

static void free_all(uint64_t *buf[2])
  {
  uint64_t *p[6] = { buf[0], buf[1], Vb, Db, S1, S2 };
  for (int i = 0; i < 6; i++)
    if (p[i]) free(p[i] - FRONTPAD);
  Vb = Db = S1 = S2 = NULL;
  }

/* Runs one complete small test in memory (no checkpoints) */
static int run_small(int rr, int ss, uint64_t *res)
  {
  r = rr; s = ss;
  setup_params();
  uint64_t *buf[2];
  alloc_all(buf);
  memset(&ecs, 0, sizeof ecs);
  int rc = map_checks > 0 && !map_check(buf[0], buf[1], map_checks)
           ? RES_ECFAIL : RES_RESIDUE;
  if (rc != RES_ECFAIL) {
    Vb[0] = 2;
    crcV = poly_crc(Vb);
    struct history h;
    memset(&h, 0, sizeof h);
    rc = run_test(buf, 0, 0, &h, res);
    }
  free_all(buf);
  return rc;
  }

/* Small known cases (computed with an independent bit-by-bit program and
   consistent with irred 3.15).  r - sodd = 0 mod 128 for 4423 3015.  Then
   every fault-injection site, once each. */
static int selftest(void)
  {
  static const struct { int r, s; bool irred; uint64_t res; } T[] = {
    {  607,  105, true,  0 },
    { 1279,  418, true,  0 },
    { 1279,  500, false, 0x86946f14b9e6414cULL },
    { 1279,  438, false, 0x0000000000000000ULL },
    { 4423, 1000, false, 0x6b0463063ed3a0e8ULL },
    { 4423, 3015, false, 0x95868bc52f115a6eULL },
    { 4423, 2247, false, 0xd7951e1a96d918fcULL },
    { 4423, 2098, true,  0 },
    { 9689, 4187, true,  0 },
  };
  int fails = 0;
  const int nT = (int)(sizeof T / sizeof T[0]);

  /* the word-wise CRC must agree with the byte-wise one */
  {
    uint64_t w[5] = { 0x0123456789abcdefULL, 0, ~0ULL, 0x8000000000000001ULL,
                      0xdeadbeefcafef00dULL };
    uint32_t c = 0xFFFFFFFFu;
    for (int i = 0; i < 5; i++) c = crc_word(c, w[i]);
    bool ok = (c ^ 0xFFFFFFFFu) == crc32_buf(w, sizeof w);
    printf("selftest CRC slice-by-8: %s\n", ok ? "ok" : "FAILED");
    if (!ok) fails++;
  }

  const bool ec_saved = ec_enabled;
  for (int pass = 0; pass < 2; pass++) {
    ec_enabled = (pass == 0);
    for (int i = 0; i < nT; i++) {
      uint64_t res = 0;
      int rc = run_small(T[i].r, T[i].s, &res);
      bool ok = T[i].irred ? rc == RES_IRRED
                           : (rc == RES_RESIDUE && res == T[i].res);
      printf("selftest r=%5d s=%5d %s: %s\n", T[i].r, T[i].s,
             ec_enabled ? "checked  " : "unchecked", ok ? "ok" : "FAILED");
      if (!ok) fails++;
      }
    }
  ec_enabled = true;

  /* fault injection: the result must still be right, and the fault seen */
  static const struct { int site; uint64_t n; int ti; } F[] = {
    { F_SQ, 100, 8 }, { F_SQ, 4000, 4 }, { F_SQ, 4423, 4 },
    { F_ACC, 3, 8 }, { F_CHK, 2, 4 }, { F_VER, 2, 8 },
  };
  for (size_t i = 0; i < sizeof F / sizeof F[0]; i++) {
    fault_site = F[i].site;
    fault_n = F[i].n;
    memset(fault_seen, 0, sizeof fault_seen);
    uint64_t res = 0;
    int ti = F[i].ti;
    int rc = run_small(T[ti].r, T[ti].s, &res);
    bool right = T[ti].irred ? rc == RES_IRRED
                             : (rc == RES_RESIDUE && res == T[ti].res);
    bool ok = right && ecs.failures >= 1;
    printf("selftest fault %s:%" PRIu64 " (r=%d s=%d): %s (%" PRIu64
           " check failure%s, result %s)\n", fault_names[F[i].site], F[i].n,
           T[ti].r, T[ti].s, ok ? "ok" : "FAILED", ecs.failures,
           ecs.failures == 1 ? "" : "s", right ? "right" : "WRONG");
    if (!ok) fails++;
    }

  /* a linear squaring bug: invisible to the chain, caught by the map check */
  fault_site = F_MAP;
  {
    uint64_t res = 0;
    int saved = map_checks;
    map_checks = 0;
    int rc = run_small(T[4].r, T[4].s, &res);
    bool chain_blind = (rc == RES_RESIDUE && res != T[4].res &&
                        ecs.failures == 0);
    map_checks = saved > 0 ? saved : 16;
    rc = run_small(T[4].r, T[4].s, &res);
    bool caught = (rc == RES_ECFAIL);
    map_checks = saved;
    bool ok = chain_blind && caught;
    printf("selftest linear bug: %s (Gerbicz chain %s it, map check %s it)\n",
           ok ? "ok" : "FAILED", chain_blind ? "cannot see" : "?SAW?",
           caught ? "catches" : "MISSES");
    if (!ok) fails++;
  }
  fault_site = -1;
  ec_enabled = ec_saved;

  printf("%s\n", fails ? "SELFTEST FAILED" : "selftest passed");
  return fails ? EXIT_FAILURE : EXIT_SUCCESS;
  }

static void usage(FILE *f)
  {
  fprintf(f,
    "Usage: irred [options] r s\n"
    "  Tests x^r + x^s + 1 over GF(2) for irreducibility.\n"
    "Options:\n"
    "  -v, --verbose              more informative output\n"
    "  --progress N               print an interim residue every N squarings\n"
    "  --no-checkpoint, --no-ckpt no checkpoint files (and no resume)\n"
    "  --checkpoint-mins M, --ckpt-mins M\n"
    "                             checkpoint every M minutes (default %g;\n"
    "                             0 = only on SIGINT/SIGTERM)\n"
    "  --selftest                 run small known cases (with injected faults)\n"
    "  --threads N                threads per squaring (default 1; needs a\n"
    "                             build with -fopenmp; used for r > ~5 million)\n"
    "  --nt                       streaming stores in the interleave (try on\n"
    "                             machines where r is memory-bound; measure)\n"
    "  --isa auto|avx512|avx2|generic\n"
    "                             interleave code to use (default auto)\n"
    "  -h, --help                 this message\n"
    "Error checking (on by default):\n"
    "  --no-ec                    no Gerbicz check\n"
    "  --ec-block L               squarings per block (default sqrt(r)/16, >= 32)\n"
    "  --ec-window N              blocks per window (default L)\n"
    "  --ec-retries N             failed checks allowed at one point (default 3)\n"
    "  --map-checks N             random inputs for the start-up map check (16)\n"
    "  --seed N                   seed for the map check\n"
    "  --ec-fault SITE:N          TEST: inject a fault (sq acc chk ver map)\n"
    "Checkpoint files: irred_r<r>_s<s>.ckpt[.1|.2] in the current directory;\n"
    "rerunning the same command resumes.  Exit status 2: stopped by a signal\n"
    "after writing a checkpoint; 3: error check failed.\n", CKPT_DEFAULT_MINS);
  }

static bool parse_long(const char *str, long lo, long hi, long *out)
  {
  char *end;
  errno = 0;
  long v = strtol(str, &end, 10);
  if (errno || end == str || *end != '\0' || v < lo || v > hi) return false;
  *out = v;
  return true;
  }

static bool parse_minutes(const char *str, double *out)
  {
  char *end;
  errno = 0;
  double v = strtod(str, &end);
  if (errno || end == str || *end != '\0' || !(v >= 0) || v > 1e7) return false;
  *out = v;
  return true;
  }

int main(int argc, char *argv[])
  {
  long progress = 0;
  const char *pos[2];
  int npos = 0;
  bool do_selftest = false;
  const char *isa = "auto";
  uint64_t seed = 0;
  bool have_seed = false;

  setvbuf(stdout, NULL, _IOLBF, 0);
  crc_init();

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    const char *val = NULL;
    char opt[64];
    if (arg[0] == '-' && arg[1] == '-' && strchr(arg, '=') != NULL) {
      size_t n = (size_t)(strchr(arg, '=') - arg);  /* --opt=value form */
      if (n >= sizeof opt) n = sizeof opt - 1;
      memcpy(opt, arg, n);
      opt[n] = '\0';
      val = arg + n + 1;
      arg = opt;
      }
#define NEED_VALUE() \
    do { if (val == NULL) { \
           if (i + 1 >= argc) { \
             fprintf(stderr, "%s needs a value\n", arg); return EXIT_FAILURE; } \
           val = argv[++i]; } } while (0)

    if (!strcmp(arg, "-v") || !strcmp(arg, "--verbose"))
      verbose = true;
    else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      usage(stdout);
      return EXIT_SUCCESS;
      }
    else if (!strcmp(arg, "--no-checkpoint") || !strcmp(arg, "--no-ckpt"))
      ckpt_enabled = false;
    else if (!strcmp(arg, "--checkpoint-mins") || !strcmp(arg, "--ckpt-mins")) {
      double m;
      NEED_VALUE();
      if (!parse_minutes(val, &m)) {
        fprintf(stderr, "bad value for %s: %s\n", arg, val);
        return EXIT_FAILURE;
        }
      ckpt_secs = 60.0 * m;
      }
    else if (!strcmp(arg, "--progress")) {
      NEED_VALUE();
      if (!parse_long(val, 0, 0x7FFFFFFF, &progress)) {
        fprintf(stderr, "bad value for --progress: %s\n", val);
        return EXIT_FAILURE;
        }
      }
    else if (!strcmp(arg, "--selftest"))
      do_selftest = true;
    else if (!strcmp(arg, "--isa")) {
      NEED_VALUE();
      isa = val;
      }
    else if (!strcmp(arg, "--no-ec"))
      ec_enabled = false;
    else if (!strcmp(arg, "--nt"))
      ilv_nt = true;
    else if (!strcmp(arg, "--threads") || !strcmp(arg, "--mt-min-words")) {
      long v;
      NEED_VALUE();
      if (!parse_long(val, 1, 1024, &v) && strcmp(arg, "--threads") == 0) {
        fprintf(stderr, "bad value for --threads: %s\n", val);
        return EXIT_FAILURE;
        }
      if (!strcmp(arg, "--threads")) nthreads = (int)v;
      else {
        if (!parse_long(val, 1, 0x7FFFFFFF, &v)) {
          fprintf(stderr, "bad value for %s: %s\n", arg, val);
          return EXIT_FAILURE;
          }
        mt_min_words = mt_min_block = v;        /* testing */
        }
      }
    else if (!strcmp(arg, "--ec-block") || !strcmp(arg, "--ec-window") ||
             !strcmp(arg, "--ec-retries") || !strcmp(arg, "--map-checks")) {
      long v;
      NEED_VALUE();
      bool isblk = !strcmp(arg, "--ec-block"), iswin = !strcmp(arg, "--ec-window");
      if (!parse_long(val, (isblk || iswin) ? 1 : 0, 0x7FFFFFFF, &v)) {
        fprintf(stderr, "bad value for %s: %s\n", arg, val);
        return EXIT_FAILURE;
        }
      if (isblk) ec_block = v;
      else if (iswin) ec_window = v;
      else if (!strcmp(arg, "--ec-retries")) ec_retries = (int)v;
      else map_checks = (int)v;
      }
    else if (!strcmp(arg, "--ec-fault")) {
      NEED_VALUE();
      const char *colon = strchr(val, ':');
      fault_site = -1;
      for (int f = 0; colon && f < F_NSITES; f++)
        if ((size_t)(colon - val) == strlen(fault_names[f]) &&
            !strncmp(val, fault_names[f], (size_t)(colon - val)))
          fault_site = f;
      long n;
      if (fault_site < 0 || !parse_long(colon + 1, 1, 0x7FFFFFFF, &n)) {
        fprintf(stderr, "bad value for --ec-fault: %s (SITE:N, SITE one of "
                "sq acc chk ver map)\n", val);
        return EXIT_FAILURE;
        }
      fault_n = (uint64_t)n;
      }
    else if (!strcmp(arg, "--seed")) {
      long v;
      NEED_VALUE();
      if (!parse_long(val, 0, 0x7FFFFFFF, &v)) {
        fprintf(stderr, "bad value for --seed: %s\n", val);
        return EXIT_FAILURE;
        }
      seed = (uint64_t)v;
      have_seed = true;
      }
    else if (arg[0] == '-' && !(arg[1] >= '0' && arg[1] <= '9')) {
      fprintf(stderr, "unknown option %s\n", argv[i]);
      usage(stderr);
      return EXIT_FAILURE;
      }
    else if (npos < 2)
      pos[npos++] = arg;
    else {
      fprintf(stderr, "too many arguments: %s\n", arg);
      usage(stderr);
      return EXIT_FAILURE;
      }
#undef NEED_VALUE
    }

  printf("This is irred (forked github.com/nclvrps/mersenne_trinomial/ version %s, %s)\n",
         IRRED_VERSION, IRRED_DATE);

  if (!choose_interleave(isa)) {
    fprintf(stderr, "--isa %s: not available on this CPU/build (use auto, "
            "avx512, avx2 or generic)\n", isa);
    return EXIT_FAILURE;
    }
#ifndef _OPENMP
  if (nthreads > 1) {
    printf("Note: this irred was built without OpenMP (-fopenmp), so "
           "--threads is ignored\n");
    nthreads = 1;
    }
#endif
  if (do_selftest) {
    printf("Interleave: %s%s\n", ilv_name,
           nthreads > 1 ? " (threads forced on for small r)" : "");
    if (nthreads > 1) mt_min_words = mt_min_block = 1;
    ckpt_enabled = false;
    return selftest();
    }

  long rl, sl;
  if (npos != 2) {
    usage(stderr);
    return EXIT_FAILURE;
    }
  if (!parse_long(pos[0], 1, 0x7FFFFFF0L, &rl)) {
    fprintf(stderr, "bad r: %s\n", pos[0]);
    return EXIT_FAILURE;
    }
  if (!parse_long(pos[1], 1, 0x7FFFFFF0L, &sl)) {
    fprintf(stderr, "bad s: %s\n", pos[1]);
    return EXIT_FAILURE;
    }
  r = (int)rl;
  s = (int)sl;

  if (check_params()) return EXIT_FAILURE;
  printf("Testing x^%d + x^%d + 1", r, s);
  if (!(s & 1)) printf(" (via its reciprocal x^%d + x^%d + 1)", r, r - s);
  printf("\n");
  if (!prime(r))
    printf("Warning: r = %d is not prime, so irreducibility test "
           "is incomplete\n", r);
  if ((r % 8 == 3 || r % 8 == 5) && s != 2 && s != r - 2)
    printf("Note: r = +-3 (mod 8), so by Swan's theorem this trinomial is "
           "reducible; testing anyway to compute the residue\n");

  setup_params();
  if (verbose)
    printf("sodd %d, delta %d (%d words + %d bits), %zu words per polynomial"
           ", %s interleave\n", sodd, delta, deltaw, deltaq, nwords, ilv_name);
  if (nthreads > 1)
    printf("Threads: %d%s\n", nthreads, (long)q41 >= mt_min_words ? "" :
           " requested, but r is too small for them to help: using 1");

  uint64_t *buf[2];
  alloc_all(buf);

  /* the squaring code against an independent implementation */
  if (!have_seed)
    seed = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
  rng_state ^= seed * 0x9E3779B97F4A7C15ULL;
  if (rng_state == 0) rng_state = 1;
  if (map_checks > 0) {
    double tm = wall_now();
    if (!map_check(buf[0], buf[1], map_checks)) {
      printf("MAP CHECK FAILED: the fast squaring code disagrees with the "
             "reference implementation for r = %d, s = %d (%s interleave).\n"
             "This is a bug (or a persistent hardware fault); nothing was "
             "computed.\n", r, s, ilv_name);
      free_all(buf);
      return 3;
      }
    if (verbose)
      printf("Map check: fast squaring agrees with the reference on %d "
             "inputs (%.2f s)\n", map_checks, wall_now() - tm);
    }

  uint64_t k0 = 0;
  struct history h;
  memset(&h, 0, sizeof h);
  memset(Vb, 0, bufwords * sizeof(uint64_t));

  if (ckpt_enabled) {
    ckpt_setup_names();
    ckpt_lock();
    if (!ckpt_resume(Vb, &k0, &h)) {
      memset(Vb, 0, bufwords * sizeof(uint64_t));
      k0 = 0;
      }
    install_signals();
    if (ckpt_secs > 0)
      printf("Checkpointing to %s every %g minutes\n", ckpt_name[0],
             ckpt_secs / 60);
    else
      printf("Checkpointing to %s only when stopped by a signal\n",
             ckpt_name[0]);
    }
  else
    printf("Checkpointing disabled\n");
  if (k0 == 0)
    Vb[0] = 2;                          /* start from x */
  crcV = poly_crc(Vb);

  const long L = ec_block > 0 ? ec_block : auto_block();
  const long N = ec_window > 0 ? ec_window : L;
  if (ec_enabled)
    printf("Error checking: Gerbicz XOR chain, blocks of %ld squarings, "
           "windows of up to %ld blocks%s\n", L, N,
           map_checks > 0 ? "; map check passed" : "");
  else
    printf("Error checking DISABLED%s\n",
           map_checks > 0 ? " (map check passed)" : "");

  double t0 = wall_now(), c0 = cpu_now();
  uint64_t res = 0;
  memset(&ecs, 0, sizeof ecs);
  int rc = run_test(buf, k0, progress, &h, &res);
  double wall = wall_now() - t0, cpu = cpu_now() - c0;

  if (rc == RES_STOPPED || rc == RES_ECFAIL) {
    free_all(buf);
    return rc == RES_STOPPED ? 2 : 3;
    }

  uint64_t unchecked = h.unchecked + (ec_enabled ? 0 : (uint64_t)r - k0);
  if (ec_enabled) {
    printf("Error checking: %" PRIu64 " windows verified", ecs.checks -
           ecs.failures + h.ec_checks);
    if (ecs.failures + h.ec_failures > 0)
      printf(", %" PRIu64 " check failures (recovered)",
             ecs.failures + h.ec_failures);
    printf("\n");
    }
  if (unchecked == 0)
    printf("All %d squarings were error-checked\n", r);
  else
    printf("Note: %" PRIu64 " of the %d squarings were NOT error-checked\n",
           unchecked, r);

  if (rc == RES_IRRED) {
    printf("%d %d irreducible (primitive if 2^%d-1 is prime)\n", r, s, r);
    printf("%d %d irreducible (may be primitive) LOG\n", r, s);
    }
  else {
    if (verbose)
      printf("Not irreducible/primitive, residue %016" PRIx64 " (hex)\n", res);
    printf("%d %d %c%016" PRIx64 " LOG\n", r, s, (s == sodd) ? 'x' : 'y', res);
    }

  if (ckpt_enabled) ckpt_remove_all();
  free_all(buf);

  if (verbose) {
    char w1[32], w2[32];
    fmt_duration(wall, w1, sizeof w1);
    double done = (double)r - (double)k0;     /* squarings this run */
    printf("Time this run: %s wall, %.2f s CPU = %.3g r^2 nsec\n", w1, cpu,
           done > 0 ? 1e9 * cpu / ((double)r * done) : 0.0);
    if (ec_enabled && done > 0)
      printf("Error checking cost: %" PRIu64 " extra squarings (%.2f%%), "
             "%.2f s in checks (%.2f%% of wall time)\n", ecs.sq_check,
             100.0 * (double)ecs.sq_check / done, ecs.t_check,
             wall > 0 ? 100.0 * ecs.t_check / wall : 0.0);
    if (k0 > 0) {
      fmt_duration(h.wall + wall, w2, sizeof w2);
      printf("Total over all runs: %s wall, %.2f s CPU\n", w2, h.cpu + cpu);
      }
    }
  return EXIT_SUCCESS;
  }

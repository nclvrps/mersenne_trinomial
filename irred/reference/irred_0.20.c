/* irred.c                                      irred version forked from 3.15

***********************************************************
Forked version (2026):
    https://github.com/nclvrps/irred

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
        --selftest               test a few small known cases and exit
        --isa auto|avx512|avx2|generic
                                 which interleave code to use; the default
                                 picks the best the CPU has (for testing)
        -h, --help               usage

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

        Exit status: 0 = finished, 1 = error, 2 = stopped by SIGINT or
        SIGTERM after writing a checkpoint (run the same command to resume).

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
        directory at the same time.

Comments on timing and memory requirements:

        Each trinomial takes time O(r^2).
        Space requirement is about r/4 bytes (two buffers of r/8 bytes).

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

Compile with, e.g.,
        gcc -O3 -march=native -o irred irred.c
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

#define IRRED_VERSION "0.20"
#define IRRED_DATE    "20261117"

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
    if (deltaq != 0)
      fold_shift(a + t0, a + t0 + deltaw, n, deltaq);
    else
      fold_word(a + t0, a + t0 + deltaw, n);
    top = t0 - 1;
    }

  /* The last source word q4 is special: only its bits >= alpha+1 count */
  uint64_t w = a[q4] & mask2;
  if (deltaq != 0) {
    a[q4 - deltaw] ^= (w >> deltaq) | (a[q4 + 1] << deltaqc);
    a[q4 - deltaw - 1] ^= w << deltaqc;   /* index may be -1: then w << deltaqc
                                             is 0 (mask2) and FRONTPAD absorbs it */
    }
  else
    a[q4 - deltaw] ^= w;
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

static void interlvf(const uint64_t *restrict a, uint64_t *restrict b)
/* Interleave, loop index runs up (RPB, 20000907).  The reverse-direction
   version interlvr was dropped in 0.20: alternating directions is slower
   on current machines (about 10% at r = 756839 and at r = 136279841). */
  {
  const long h = q41;
  const int s1 = s1_x, s2 = s2_x;
  for (long j = 0; j <= h; j++)
    ILV_STEP(j);
  }

#if HAVE_X86_SIMD
/* AVX2 version of the interleave, four words of lo/hi at a time.  Each
   byte is split into nibbles, a nibble is spread to the even (lo) or odd
   (hi) bits of a byte by table lookup (vpshufb), and the bytes of the two
   results are interleaved (vpunpck[lh]bw), which within each 128-bit lane
   gives the output for one 64-bit input word per 16 bytes. */

#define ILV_AVX2_BLOCK(j)                                                 \
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
    _mm256_storeu_si256((__m256i *)(b + 2 * (j)),                         \
                        _mm256_permute2x128_si256(U, V, 0x20));           \
    _mm256_storeu_si256((__m256i *)(b + 2 * (j) + 4),                     \
                        _mm256_permute2x128_si256(U, V, 0x31));           \
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

__attribute__((target("avx2")))
static void interlvf_avx2(const uint64_t *restrict a, uint64_t *restrict b)
  {
  ILV_AVX2_SETUP;
  long j = 0;
  for (; j + 3 <= h; j += 4)
    ILV_AVX2_BLOCK(j);
  for (; j <= h; j++)
    ILV_STEP(j);
  }

#endif

#if HAVE_X86_SIMD
/* AVX-512BW version: the same method, eight words of lo/hi at a time.  The
   byte interleave works within 128-bit lanes, so a final cross-lane
   permutation puts the eight 128-bit results in order. */
__attribute__((target("avx512f,avx512bw")))
static void interlvf_avx512(const uint64_t *restrict a, uint64_t *restrict b)
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
  long j = 0;
  for (; j + 7 <= h; j += 8) {
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
    _mm512_storeu_si512((void *)(b + 2 * j),
                        _mm512_permutex2var_epi64(U, P0, V));
    _mm512_storeu_si512((void *)(b + 2 * j + 8),
                        _mm512_permutex2var_epi64(U, P1, V));
    }
  for (; j <= h; j++)
    ILV_STEP(j);
  }
#endif

typedef void (*ilv_fn)(const uint64_t *restrict, uint64_t *restrict);
static ilv_fn ilv = interlvf;
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
    ilv = interlvf_avx512;
    ilv_name = "AVX-512BW";
    return true;
    }
  if (want2 && __builtin_cpu_supports("avx2") && strcmp(isa, "avx512") != 0) {
    ilv = interlvf_avx2;
    ilv_name = "AVX2";
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
/* Checkpoints

   File irred_r<r>_s<s>.ckpt: a header, then the polynomial x^(2^k) mod A(x)
   in normal order, words 0..q11 (top word masked).  Header and data each
   carry a CRC-32.  Written to <file>.tmp, fsync'ed, the older generations
   rotated (.ckpt -> .ckpt.1 -> .ckpt.2), then renamed into place. */

#define CKPT_MAGIC   "IRREDCKP"         /* 8 bytes, no terminating NUL used */
#define CKPT_VERSION 1u

struct ckpt_hdr {
  char     magic[8];
  uint32_t version;
  uint32_t hdr_bytes;           /* sizeof(struct ckpt_hdr) */
  uint64_t r, s, sodd;
  uint64_t k;                   /* squarings done */
  uint64_t nwords;              /* words of data that follow */
  double   wall;                /* wall-clock seconds in all runs so far */
  double   cpu;                 /* CPU seconds in all runs so far */
  int64_t  written;             /* time(NULL) when written */
  uint32_t data_crc;
  uint32_t hdr_crc;             /* CRC of all preceding header bytes */
};

static uint32_t crc_table[256];

static void crc_init(void)
  {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++)
      c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    crc_table[i] = c;
    }
  }

static uint32_t crc32_buf(const void *buf, size_t len)
  {
  const unsigned char *p = buf;
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++)
    c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
  }

static bool ckpt_enabled = true;
static double ckpt_secs = CKPT_DEFAULT_MINS * 60.0;
static char ckpt_name[3][128];  /* .ckpt, .ckpt.1, .ckpt.2 */
static char ckpt_tmp[128], lock_name[128];
static volatile sig_atomic_t stop_requested = 0;

static void on_signal(int sig)
  {
  if (!stop_requested) {        /* first signal: stop at the next squaring */
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

static bool ckpt_write(const uint64_t *a, uint64_t k, double wall, double cpu)
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
  H.wall = wall;
  H.cpu = cpu;
  H.written = (int64_t)time(NULL);

  /* data: words 0..q11, top word masked (canonical form) */
  uint64_t top = a[q11] & mask1;
  H.data_crc = crc32_buf(a, (nwords - 1) * sizeof(uint64_t));
  {
    /* fold in the masked top word: CRC of the concatenation */
    uint32_t c = H.data_crc ^ 0xFFFFFFFFu;
    const unsigned char *p = (const unsigned char *)&top;
    for (size_t i = 0; i < sizeof top; i++)
      c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    H.data_crc = c ^ 0xFFFFFFFFu;
  }
  H.hdr_crc = crc32_buf(&H, offsetof(struct ckpt_hdr, hdr_crc));

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
  fprintf(stderr, "[checkpoint: %" PRIu64 "/%d squarings (%.2f%%) -> %s, "
          "%.2f s]\n", k, r, 100.0 * (double)k / r, ckpt_name[0],
          wall_now() - t0);
  return true;
  }

/* Try to load one checkpoint file into a.  Returns true if it is a valid
   checkpoint for this (r, s); otherwise prints why it is not used. */
static bool ckpt_load_one(const char *name, uint64_t *a, struct ckpt_hdr *H)
  {
  FILE *f = fopen(name, "rb");
  if (f == NULL) return false;          /* silently: no such file */
  bool ok = fread(H, sizeof *H, 1, f) == 1;
  if (!ok || memcmp(H->magic, CKPT_MAGIC, 8) != 0) {
    printf("Ignoring %s: not an irred checkpoint file\n", name);
    fclose(f);
    return false;
    }
  if (H->version != CKPT_VERSION || H->hdr_bytes != sizeof *H ||
      crc32_buf(H, offsetof(struct ckpt_hdr, hdr_crc)) != H->hdr_crc) {
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
  if (H->sodd != (uint64_t)sodd || H->nwords != nwords || H->k > (uint64_t)r) {
    printf("Ignoring %s: inconsistent header\n", name);
    fclose(f);
    return false;
    }
  memset(a, 0, bufwords * sizeof(uint64_t));
  ok = fread(a, sizeof(uint64_t), nwords, f) == nwords &&
       crc32_buf(a, nwords * sizeof(uint64_t)) == H->data_crc;
  fclose(f);
  if (!ok) {
    printf("Ignoring %s: data truncated or damaged (CRC mismatch)\n", name);
    memset(a, 0, bufwords * sizeof(uint64_t));
    return false;
    }
  return true;
  }

/* Returns the number of squarings already done (0 if starting afresh) and
   leaves the state in a; adds earlier runs' times to *wall0, *cpu0. */
static uint64_t ckpt_resume(uint64_t *a, double *wall0, double *cpu0)
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
             "(%.2f%%) in %s\n", ckpt_name[g], when, H.k, r,
             100.0 * (double)H.k / r, took);
      *wall0 = H.wall;
      *cpu0 = H.cpu;
      return H.k;
      }
    }
  return 0;
  }

static void ckpt_remove_all(void)
  {
  for (int g = 0; g < 3; g++) remove(ckpt_name[g]);
  remove(ckpt_tmp);
  remove(lock_name);
  }

/* ------------------------------------------------------------------------ */
/* The test */

enum { RES_IRRED, RES_RESIDUE, RES_STOPPED };

/* Runs squarings k0..r-1 starting from the polynomial in buf[0], with
   checkpoints and progress output as configured.  Returns RES_* and the
   residue in *res. */
static int run_test(uint64_t *buf[2], uint64_t k0, long progress,
                    double wall0, double cpu0, uint64_t *res)
  {
  uint64_t *a = buf[0], *b = buf[1];
  const double tstart = wall_now(), cstart = cpu_now();
  double last_ckpt = tstart, last_prog_t = tstart;
  uint64_t last_prog_k = k0;

  for (uint64_t k = k0; k < (uint64_t)r; ) {
    reducep(a);
    ilv(a, b);
    uint64_t *t = a; a = b; b = t;
    k++;

    if (progress > 0 && k % (uint64_t)progress == 0) {
      double now = wall_now();
      double per = (k > last_prog_k) ? (now - last_prog_t) / (double)(k - last_prog_k)
                                     : 0;
      char eta[32];
      fmt_duration(per * (double)((uint64_t)r - k), eta, sizeof eta);
      printf("iter %" PRIu64 "/%d (%6.2f%%)  %.3g ms/iter  ETA %s  "
             "residue %016" PRIx64 "\n", k, r, 100.0 * (double)k / r,
             1e3 * per, eta, a[0]);
      fflush(stdout);
      last_prog_t = now;
      last_prog_k = k;
      }

    if (stop_requested) {
      printf("Received signal %d after %" PRIu64 "/%d squarings\n",
             (int)stop_requested, k, r);
      fflush(stdout);
      if (ckpt_enabled) {
        double now = wall_now();
        if (ckpt_write(a, k, wall0 + (now - tstart),
                       cpu0 + (cpu_now() - cstart))) {
          printf("Checkpoint written; run the same command again to "
                 "continue\n");
          fflush(stdout);
          }
        }
      return RES_STOPPED;
      }

    if (ckpt_enabled && ckpt_secs > 0 && k < (uint64_t)r && (k & 63) == 0) {
      double now = wall_now();
      if (now - last_ckpt >= ckpt_secs) {
        ckpt_write(a, k, wall0 + (now - tstart), cpu0 + (cpu_now() - cstart));
        last_ckpt = wall_now();
        }
      }
    }

  if (comparex(a)) return RES_IRRED;
  *res = a[0];
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

/* Small known cases (computed with an independent bit-by-bit program and
   consistent with irred 3.15).  r - sodd = 0 mod 128 for 4423 3015. */
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
  for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
    r = T[i].r; s = T[i].s;
    setup_params();
    uint64_t *buf[2] = { alloc_buffer(), alloc_buffer() };
    buf[0][0] = 2;                      /* x */
    uint64_t res = 0;
    int rc = run_test(buf, 0, 0, 0, 0, &res);
    bool ok = T[i].irred ? rc == RES_IRRED
                         : (rc == RES_RESIDUE && res == T[i].res);
    printf("selftest r=%5d s=%5d: %s\n", r, s, ok ? "ok" : "FAILED");
    if (!ok) fails++;
    free(buf[0] - FRONTPAD);
    free(buf[1] - FRONTPAD);
    }
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
    "  --selftest                 run a few small known cases and exit\n"
    "  --isa auto|avx512|avx2|generic\n"
    "                             interleave code to use (default auto)\n"
    "  -h, --help                 this message\n"
    "Checkpoint files: irred_r<r>_s<s>.ckpt[.1|.2] in the current directory;\n"
    "rerunning the same command resumes.  Exit status 2: stopped by a signal\n"
    "after writing a checkpoint.\n", CKPT_DEFAULT_MINS);
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

  printf("This is irred (forked github.com/nclvrps/irred version %s, %s)\n",
         IRRED_VERSION, IRRED_DATE);

  if (!choose_interleave(isa)) {
    fprintf(stderr, "--isa %s: not available on this CPU/build (use auto, "
            "avx512, avx2 or generic)\n", isa);
    return EXIT_FAILURE;
    }
  if (do_selftest) {
    printf("Interleave: %s\n", ilv_name);
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

  uint64_t *buf[2] = { alloc_buffer(), alloc_buffer() };
  uint64_t k0 = 0;
  double wall0 = 0, cpu0 = 0;

  if (ckpt_enabled) {
    ckpt_setup_names();
    ckpt_lock();
    k0 = ckpt_resume(buf[0], &wall0, &cpu0);
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
    buf[0][0] = 2;                      /* start from a = x */

  double t0 = wall_now(), c0 = cpu_now();
  uint64_t res = 0;
  int rc = run_test(buf, k0, progress, wall0, cpu0, &res);
  double wall = wall_now() - t0, cpu = cpu_now() - c0;

  if (rc == RES_STOPPED) {
    free(buf[0] - FRONTPAD);
    free(buf[1] - FRONTPAD);
    return 2;
    }

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
  free(buf[0] - FRONTPAD);
  free(buf[1] - FRONTPAD);

  if (verbose) {
    char w1[32], w2[32];
    fmt_duration(wall, w1, sizeof w1);
    double done = (double)r - (double)k0;     /* squarings this run */
    printf("Time this run: %s wall, %.2f s CPU = %.3g r^2 nsec\n", w1, cpu,
           done > 0 ? 1e9 * cpu / ((double)r * done) : 0.0);
    if (k0 > 0) {
      fmt_duration(wall0 + wall, w2, sizeof w2);
      printf("Total over all runs: %s wall, %.2f s CPU\n", w2, cpu0 + cpu);
      }
    }
  return EXIT_SUCCESS;
  }

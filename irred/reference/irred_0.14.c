/* irred.c	 					irred version forked from 3.15
 

***********************************************************
Forked version (2026):
    https://github.com/nclvrps/irred

Cleaned up with the assistance of Claude (chatbot)
to create a version suited for 64-bit machines
and modern optimizing compilers,
and to remove functionality now better performed by another program.

This forked version removes all sieving for polynomial factors,
and just does the irreducibility test.

As before, a 32-bit hexadecimal residue is printed
if a trinomial is not irreducible.

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

Nonetheless, if this test does confirm non-reducibility, it is desirable
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
It is somewhat lacking in documentation.

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
|  External assembler routines (see ASM and comments below) are covered  |
|  by the same copyright conditions as this C code.                      |
|                                                                        |
==========================================================================
 
Author: Richard P. Brent (rpb@comlab.ox.ac.uk)
      	with assistance and contributions from
   	Samuli Larvala and Paul Zimmermann

Dates:  20000528..20030328

Summary:

   An efficient test for irreducibility of trinomials
   x^r + x^s +1 (mod 2) where r is a prime, 
   or primitivity in the case that 2^r-1 is a Mersenne prime,
   using (32-bit or 64-bit) word-length bit operations.
   Written in C (with optional assembler routines for some machines).

Restrictions, Assumptions and Limitations:

   Assumes 0 < s < r, s >= LIM, r-s >= LIM,
   where LIM is usually twice the wordlength in bits,
   or six times the wordlength for ULTRA option,
   or ten times the wordlength for UNROLL option,
   or 256 for IBMPC ASM option
   (other cases can be handled by less efficient programs,
   e.g. irred version 1.35 or earlier versions,
     or irred version 1.50 with ODDS false).

   It is assumed that r is prime - if not true then a warning is printed.
   There is no check on whether or not 2^r-1 is prime.
   
   When output says A(x) is "irreducible" this actually means that
   x^(2^r) = x mod A(x). If r is not prime, we should also check
   that GCD(x^(2^d) - x, A(x)) = 1 for divisors d of r, but the program
   does not do this (it would be easy enough to add code to perform this
   check if desired).

   Similarly, if r is prime but 2^r-1 is not prime, then an
   irreducible A(x) might not be primitive, because x^((2^r-1)/p) mod A(x) 
   is not checked for any prime divisors p of 2^r-1. (In this case we
   can not complete the primitivity test without factoring 2^r-1.)

Usage:
   	  irred [-v] restart-file log-file time-limit skip-file
   	  					(all arguments optional)
Arguments:
          -v flag for VERBOSE output

          A restart file can be specified as first command line argument.
          If no file is specified, input data is read from stdin.

            Format of restart file or stdin is one line
              r s1 s2
            meaning to start next run at this (r,s1) and go to (r,s2-1)
            [or, if s1 > s2, go from (r,s1) to (r,s2+1)].
            If s1 = s2 then run is finished.

	  A log file can be specified as second command line argument.
	  If no file is specified, then log output -> stdout with an
	  identifier "LOG" (can be extracted using grep).

 	  Max minutes to run (m) can be specified as 3rd command line argument.
	
	    Maximum runtime is m minutes (infinite if m = 0).
	
	    The parameters m and s2 may be omitted (in which case they
	    assumed to be zero).
	
	  A skip file can be specified as the fourth command line argument.
	  This file is a list of pairs (u, v) (one per line) indicating that
	  s in the interval [u, v] should not be tested. This is useful if
	  certain intervals are already done or reserved for another machine.
	  If a skip file is desired, the parameter m must be specified
	  (possibly zero).
	  
	  For a "parallel" version using MPI, see irredpar.c

Comments on timing and memory requirements:

	Sieving quickly discards about 93% of polynomials.
	(NOTE: This forked version no longer does sieving.)
	Each polynomial not discarded by sieving takes time O(r^2)
	but there may be a slow-down as r increases (this is a cache effect).

	Space requirement is 7r/16 + (constant) bytes
	(temporarily increasing to 3r.FASTTRY/16 if FASTTRY > 2, see below).
	
	The working set after sieving is 3r/16 + (constant) bytes.

References:

        R. P. Brent, Search for primitive trinomials (mod 2),
        https://maths-people.anu.edu.au/~brent/trinom.html

        R. P. Brent, S. Larvala and P. Zimmermann, A fast algorithm for
        testing irreducibility of trinomials mod 2 (preliminary report),
        Report PRG-TR-13-00, Oxford University Computing Laboratory,
        30 Dec 2000. Revision to appear in Mathematics of Computation
        (posted electronically 18 Dec 2002).     
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

*/

#include <stdio.h>
#include <stdlib.h>		/* For malloc */
#include <stdint.h>
#include <stdbool.h>
#include <time.h>   		/* For clock */
#include <string.h>     /* For memset */

/* verbose, CONTINUE, GNU determine program behaviour */

/* this was formerly #define VERBOSE true */
bool verbose = false;   /* command-line option */

#define CONTINUE false		/* If true, continue after finding a
                                   primitive trinomial (useful for small r) */

#define GNU false		/* Determines if copyright notice is printed
				   (but the program is copyright anyway) */

		     /* compile irred.c with
		      gcc -O2 -fomit-frame-pointer -funroll-loops
		      	-o irred irred.c
	 	      */
/*

Comments on different versions:

    The history of versions up to 3.15
    has been moved to a separate file: OLD_VERSION_HISTORY
    because much of it is no longer relevant to this forked version.
*/

 #define LIM (6*WLEN) 

#define CLEAR true			   /* To clear allocated space */

/* Some constants */

#define TIMEOUT 10		/* Seconds to timeout when opening files */
#define CPUTOL 0.01		/* Times less than CPUTOL sec are negligible */

  /* Number of a0 to try in fastmem.
     Set to 1 if trivial selection desired. */

  #define FASTTRY 1

#define SMALLR  200000		/* Trivial selection if r < SMALLR */
#define CPUTEST 1		/* Seconds for each trial in fastmem.
				   Assume that CPUTOL is much less than 
				   CPUTEST. */

/* Useful macro definitions */

#define ODD(i)		((i)&1)

#define WLEN  (64)	/* Bits in a long word, always assume 64-bit */
#define WLENM (WLEN-1)		/* Ditto less 1, i.e. 31 or 63 */
#define WD    (6)		/* 5 or 6, so WLEN = 2^WD */

	
#define NMAX (WLENM)	/* Sieve over x^kn - 1 for kn = 2^n-1,
				   2^n < r, n <= NMAX (but see also NEXTRA) */

#define MC 256			/* Max string length */

/* Struct for the skip list (a singly-linked list of pairs) */

struct skip {
  int low, high;
  struct skip *next;
  };

/* Global variables */

  int r, q1, s, sodd;			/* Main parameter is r */
  int alpha, delta;		/* Could be global */
  int deltaw, deltaq, deltaqc;	/* ditto */
  int q11, q4;			/* ditto */
  uint64_t mask1, mask2;		/* ditto */
  int q41;
  int s1_x, s2_x;
  uint64_t c0, c1, c2, c3, c4, c5;

double CPUtime;				/* For timer */
clock_t cstart;

	/* 32 bits should suffice for these counters */

unsigned int syscalls = 0;			/* Counts calls to system/sleep */
unsigned int clockcalls = 0;			/* Counts calls to clock */
unsigned int space = 0;				/* Counts bytes allocated by malloc */

/* Routines start here */

void initialize_r_s_globals(void) {

    // these depend only on r
  alpha = r >> 1;		/* alpha = (r-1)/2 */
  q1  = r >> WD;			/* q1 > 0 */
  q11 = (r-1) >> WD;		/* q11 = (r-1) div WLEN */
  q4 = alpha >> WD;		/* q4 = alpha div WLEN */
  q41 = (alpha+1) >> WD;			/* q41 = (alpha+1) div WLEN */
  s1_x = (alpha+1) & WLENM;		/* s1_x = (alpha+1) mod WLEN */
  s2_x = WLENM - s1_x;			/* In [0, WLEN) */

    // these depend on r and s (sodd = either s or r-s)
  sodd = ODD(s) ? s: r-s;		/* whichever of s, r-s is odd */
  delta = (r - sodd) >> 1;	/* delta = (r - sodd)/2 */
  deltaw = delta >> WD;		/* deltaw = delta div WLEN */
  deltaq = delta & WLENM;	/* deltaq = delta mod WLEN */
  deltaqc = WLEN - deltaq;	/* Special case if deltaq is zero */

  c0 = 0x00000000FFFFFFFFUL;		/* Some 64-bit constants */
  c1 = 0x0000FFFF0000FFFFUL;
  c2 = 0x00FF00FF00FF00FFUL;
  c3 = 0x0F0F0F0F0F0F0F0FUL;
  c4 = 0x3333333333333333UL;
  c5 = 0x2222222222222222UL;
  
  mask1 = UINT64_MAX >> (WLENM - ((r-1) & WLENM));
  				/* mask1 has WLEN-1 - ((r-1) mod WLEN)
  				   zero bits in high positions */
  mask2 = (~(uint64_t)1) << (alpha & WLENM);
  				/* mask2 has (alpha mod WLEN) + 1 zero bits
  				   in low positions */
}

double clockd(clock_t *starta, bool first)

/* If first = true, initialises *starta and returns zero.
   If first = false, returns time in sec since last call and updates starta. */ 
  
  {
  clock_t stop = clock();
  clockcalls++;
  if (first) *starta = stop;
  clock_t diff = stop - *starta;
  *starta = stop;
  return (double)diff/CLOCKS_PER_SEC;
  }

void *mymalloc(int size)

/* Same as malloc but gives error exit if can not allocate space.
   Optionally clears space allocated. */

  {
  space += size;
  void *ptr = malloc(size);
  if (ptr == NULL) {
    printf("malloc unable to allocate %u bytes\n", space);
    exit(EXIT_FAILURE);
    }
#if CLEAR
  memset(ptr, 0, size);
#endif
  return ptr;
  }

static inline __attribute__((always_inline))
void reducer(uint64_t * __restrict__ a, uint64_t * __restrict__ b,
             int kt, int shift, uint64_t *prev)
{
    const int shiftc = WLEN - shift;
    uint64_t newer = *prev;

    for (int j = kt; j >= 0; j--) {
        uint64_t older = newer;
        newer = a[j];
        b[j] ^= (newer >> shift) | (older << shiftc);
    }
    *prev = newer;
}
  
static inline __attribute__((always_inline))
void xor_shift_zero(uint64_t * __restrict__ dst,
                            const uint64_t * __restrict__ src,
                            int count)
/* XORs count+1 elements: dst[j] ^= src[j] for j = 0..count.
   dst and src are guaranteed non-overlapping (caller ensures this
   via r - sodd >= LIM). */
{
    for (int j = count; j >= 0; j--)
        dst[j] ^= src[j];
}

void reducep(uint64_t *a)	

/* Reduces implicit square of polynomial a of degree < r "in place"
   given x^r = x^sodd + 1 (mod 2), result left in scrambled order
   0 2 4 ... 1 3 5 ...
   
   Assumes 0 < sodd < r && r-sodd >= 2*WLEN && ODD(r) && ODD(sodd).
    
   Uses about (2.load + 1.store)r/(2.WLEN) memory ops.
   
   RPB, 20000815. */

  {

  /* To be safe, mask any high bits of a which are irrelevant */

  a[q11] &= mask1;
  a[q11+1] = 0;			/* In case reducemx called */
  a[q11+2] = 0;			/* Ditto */
  uint64_t newer = 0;			/* Assumed by reducemx */
  
  if (deltaq == 0) {		/* Special case deltaqc == WLEN */
    xor_shift_zero(a + q4 - deltaw, a + q4, q1 - q4 - 1); 
    a[q4-deltaw] ^= a[q4] & mask2;
    }

  else {			/* Usual case, deltaqc < WLEN */

    reducer (a+q4+1, a+q4+1-deltaw, q11-q4-1, deltaq, &newer);

    /* The last two iterations are special as need to mask some bits */

    uint64_t temp = newer;
    newer = a[q4] & mask2;
    a[q4-deltaw]   ^= (newer >> deltaq) | (temp << deltaqc);
    a[q4-deltaw-1] ^=  newer << deltaqc;
    }
  }
  
bool comparex(const uint64_t *a)

/* Returns true if poly a of degree r-1 is x */

  {
  if (a[0] != 2UL) return false;
  for (int j = q1-1; j > 0; j--) {
    if (a[j] != 0UL) return false;
    }
  if ((a[q1] & mask1) != 0) return false;
  return true;
  }
  
void setupx(uint64_t *a)

/* Sets up a = x (polynomial of degree r-1, all mod 2) */

  {
  memset(&a[1], 0, q1 * sizeof(uint64_t));
  a[0] = 2UL;			/* 0...010 represents x */
  }

void interlvf(uint64_t * __restrict__ a, uint64_t * __restrict__ b)

/* 64-bit version of interleave. Loop index runs up (compare interlvr).

   If bits 0, 2, 4, ... , r-3, r-1, 1, 3, 5, ..., r-4, r-2 in a,
   moves them to b in correct order.

   Work is about (1.load + 1.store + 32.ops)r/WLEN.

   RPB, 20000907 */

  {

  for (int j = 0; j <= q41; j++) {

        uint64_t lo     = a[j];
        uint64_t hi_cur = a[j + q41];
        uint64_t hi_nxt = a[j + q41 + 1];
    /*  Might have to special-case s1_x == 0,
        which occurs when r % 128 = 127,
        and presumably set hi = 0 if s1_x == 0 ?
        However, this modified code seems to give
        the same results as the original code.
     */
        uint64_t hi = (hi_cur >> s1_x) | ((hi_nxt << 1) << s2_x);

        uint64_t t = lo & c0,  u = lo >> 32;
        uint64_t v = hi & c0,  w = hi >> 32;

    u = (u | u<<16) & c1;		 /* Operations on t,u,v,w   */
    t = (t | t<<16) & c1;		 /* can be done in parallel */
    w = (w | w<<16) & c1;
    v = (v | v<<16) & c1;

    u = (u | u<<8) & c2;
    t = (t | t<<8) & c2;
    w = (w | w<<8) & c2;
    v = (v | v<<8) & c2;

    u = (u | u<<4) & c3;
    t = (t | t<<4) & c3;
    w = (w | w<<4) & c3;
    v = (v | v<<4) & c3;

    u = (u | u<<2) & c4;
    t = (t | t<<2) & c4;
    w = (w | w<<2) & c4;
    v = (v | v<<2) & c4;

    u += u & c5;
    t += t & c5;
    w += w & c5;
    v += v & c5;

    b[2*j+1] = u | (w << 1);
    b[2*j]   = t | (v << 1);
    }
  }
  
void interlvr(uint64_t * __restrict__ a, uint64_t * __restrict__ b)

/* 64-bit version of interleave. Loop index runs down (compare interlvf).

   If bits 0, 2, 4, ... , r-3, r-1, 1, 3, 5, ..., r-4, r-2 in a,
   moves them to b in correct order.

   Work is about (1.load + 1.store + 32.ops)r/WLEN.

   Because of a loop optimisation, it should be valid to access a[-1].

   RPB, 20000907 */

  {

  for (int j = q41; j >= 0; j--) {

        uint64_t lo     = a[j];
        uint64_t hi_cur = a[j + q41];
        uint64_t hi_nxt = a[j + q41 + 1];
    /*  Might have to special-case s1_x == 0,
        which occurs when r % 128 = 127,
        and presumably set hi = 0 if s1_x == 0 ?
        However, this modified code seems to give
        the same results as the original code.
     */
        uint64_t hi = (hi_cur >> s1_x) | ((hi_nxt << 1) << s2_x);

        uint64_t t = lo & c0,  u = lo >> 32;
        uint64_t v = hi & c0,  w = hi >> 32;

    u = (u | u<<16) & c1;		 /* Operations on t,u,v,w   */
    t = (t | t<<16) & c1;		 /* can be done in parallel */
    w = (w | w<<16) & c1;
    v = (v | v<<16) & c1;

    u = (u | u<<8) & c2;
    t = (t | t<<8) & c2;
    w = (w | w<<8) & c2;
    v = (v | v<<8) & c2;

    u = (u | u<<4) & c3;
    t = (t | t<<4) & c3;
    w = (w | w<<4) & c3;
    v = (v | v<<4) & c3;

    u = (u | u<<2) & c4;
    t = (t | t<<2) & c4;
    w = (w | w<<2) & c4;
    v = (v | v<<2) & c4;

    u += u & c5;
    t += t & c5;
    w += w & c5;
    v += v & c5;

    b[2*j+1] = u | (w << 1);
    b[2*j]   = t | (v << 1);
    }
  }

FILE *myfopen(const char *fname, const char *flag)

/* Attempts to open file fname. If not successful at first, keeps trying
   for TIMEOUT seconds, then error exits.
*/
   
  {
  FILE *fp;
  clock_t openstart;
  double time = clockd(&openstart, true);	/* Independent timer here */
  
  for (;;) {
    fp = fopen(fname, flag);
    if (fp != NULL) break;
    if ((time += clockd(&openstart, false)) > (double)TIMEOUT) {
      printf("Could not open %s after trying for %d seconds\n", 
	fname, TIMEOUT);
      exit(EXIT_FAILURE);
      }
    }
  return fp;
  }

bool skips(struct skip *skiplist)

/* Returns true if s is in the skip list */

  {
  struct skip *skiprec = skiplist;
  while (skiprec != NULL) {
    if ((skiprec->low <= s) && (s <= skiprec->high))
      return true;
    skiprec = skiprec->next;
    }  
  return false;
  }

uint64_t *fastmem(int sizeah, double *CPUest)

/* Selects and returns a "good" pointer a0 by performing some timing runs  -
   it is not obvious why this works (perhaps due to real <-> virtual page
   mapping, perhaps due to conflict with system tables in L2 cache), but it
   does work on Portland/bsp0/tosca for r = 3021377. Seems to have little
   effect on booth (P-III running Solaris) or on non-IBM machines. 
   
   Also (for use in computing sieve cutoff) returns estimated overall time
   in *CPUest. 
*/

  {
  clock_t cstart;
  double CPUbest, CPUworst, CPUtime;
  int nkt, kt, bestkt, nits;
  uint64_t *a0, *a1, *a;
  uint64_t *savea0[FASTTRY+1];

  nits = ((FASTTRY <= 0) || (r < SMALLR)) ? 1 : FASTTRY;

  bestkt = 1;		/* Index of best a0 yet */
  CPUworst = 0;
  CPUbest = 1;
  for (kt = nits; kt > 0; kt--) {
    CPUtime = clockd(&cstart, true);
    a0 = (uint64_t *)mymalloc(3*sizeah*(int)sizeof(uint64_t));
    savea0[kt] = a0; 	/* Save for later free or return */
    /* Changed 2 to 6 in version 3.15 to avoid out of bounds problem 
       (sizeah also increased by 4 before call to fastmem) */
    a0 += 6;		/* For optimisation in interlvr, may access a[-1] */	
    a1 = a0 + sizeah;
    a = a1;		/* Will cycle through a1, a0 */

    setupx (a);	
    for (nkt = 0; CPUtime < CPUTEST; nkt++) {
      reducep(a);		/* Timing run for about CPUTEST seconds */
      if (a == a1) {
        interlvf(a1, a0);		/* Forward interleave */
        a = a0;				/* Adjust pointer to data */
        }
      else {
        interlvr(a0, a1);		/* Reverse interleave */
        a = a1;
        }
      if ((nkt & 0x7F) == 0)		/* Reduce overhead of clockd calls */
        CPUtime += clockd(&cstart, false);
      }
    CPUtime = CPUtime/(double)nkt;	/* Normalise CPU time */
    if ((nkt == 0) || (CPUtime < CPUbest)) {
      CPUbest = CPUtime;		/* Save best result yet */
      bestkt = kt;
      }
    if (CPUtime > CPUworst)
      CPUworst = CPUtime;  		/* and (for comparison) worst too */
    }
  for (kt = nits; kt > 0; kt--) {	/* Free all but the "best" space */
    if (kt != bestkt) 	
      free(savea0[kt]);
    }
  CPUtime = ((double)r)*CPUtime;	/* Estimate of overall time */
  if (verbose) {
  if (nits > 1)
    printf("Worst/best ratio in fastmem %1.2f\n", CPUworst/CPUbest);
  printf("Estimated CPU time (not sieving) %2.2f sec = %.3g r^2 nsec\n\n",
  	CPUtime, 1.0e9*CPUtime/r/r);
  fflush(stdout); 	
  }
  *CPUest = CPUtime;			/* Return CPU time estimate */
  return savea0[bestkt];		/* and the "best" pointer */
  }

bool prime(int n)

/* Returns true if n is prime. Simple and not intended to be efficient. */

  {
  long j;
  if (n <= 1) return false;		/* n < 2 */
  if (n <= 3) return true;		/* n = 2 or 3 */
  if ((n & 1) == 0) return false;	/* n even, not 2 */
  for (j = 3; j <= (n/j); j += 2) {	/* n > 3 */
    if ((n%j) == 0) return false;
    }
  return true;
  }

int main(int argc, char *argv[])
  {
  FILE *fp;
  double CPUtime1, CPUest;
  double CPUtotal = 0;
  double CPUtotal1 = 0;
  int minutes = 0;		/* minutes before stopping */
  s = -1;
  int s1 = 0, s2 = 0;
  int sizeah, sizep;
  int skt = 0;
  uint64_t *a;			/* For polynomial of degree (r-1) */
  uint64_t *a0, *a1;		/* a = a0 or a1 */

  char line[MC];		/* Line buffer */
  char log[] = " LOG";		/* Identifier for log 
  				   (if no log file specified) */
  char str8[9];			/* 8 characters plus null */
  struct skip *skiplist;	/* The head of the skip list */
  struct skip *skiprec;
  int slow, shigh;
  bool done, found, g, swan;
  
  printf("\nThis is irred (forked github.com/nclvrps/irred version 0.14)\n");	  /* Date 20260531 */
  
#if GNU
  printf("\nCopyright (C) 2003 R. P. Brent.\n");
  printf("irred comes with absolutely no warranty. This is free software,\n");
  printf("and you are welcome to redistribute it under the conditions of\n");
  printf("the GNU General Public License, version 2, June 1991.\n");
  printf("See http://www.gnu.org/copyleft/gpl.html for further details.\n\n");
#endif

  CPUtime = clockd(&cstart, true); 		/* Initialise timer */
  a0 = NULL;
  skiplist = NULL;
  found = false;
  r = 0;
  line[0] = '\0';

  verbose = false;
  while (argc > 1 && argv[1][0] == '-')
    {
      if (strcmp (argv[1], "-v") == 0)
        {
          verbose = true;
          argc --;
          argv ++;
        }
    }

  printf("Options ");				  /* Print relevant options */
  if (FASTTRY > 0) printf("FASTTRY = %d, ", FASTTRY);

  printf("\n");
      
  if (argc > 4) {				/* Process skip file */	
    if ((fp = fopen(argv[4], "r")) == NULL)
      printf("Warning - can not open skip file\n"); 
    else {					/* Ignore if can't open */
      while (fgets(line, MC, fp) != NULL) {
        skiprec = (struct skip *)mymalloc((int)sizeof(struct skip));
        slow = shigh = 0;  
        if (sscanf(line, "%d %d", &slow, &shigh) < 2) break;
        if ((slow <= 0) || (slow > shigh)) break; /* >= changed to > 20010307 */
#if false				  	   /* Print skip list */
        printf("Will skip s in [%d, %d]\n", slow, shigh);
#endif      
        skiprec->low = slow;
        skiprec->high = shigh;
        skiprec->next = skiplist;
        skiplist = skiprec;
        }
      fclose(fp);	
      }
    }  

  if (argc > 3) {   /* Run-time bound if nonzero, ignored if zero */
    if (sscanf(argv[3], "%d", &minutes) <= 0) minutes = 0;
    }
    	  
  if (argc > 1) {
    fp = myfopen(argv[1], "r");
    fgets(line, MC, fp);   
    fclose(fp);	
    }
  else {
    printf ("r, s1, s2 ?\n"); 	  	  /* Interactive input */
    fgets(line, MC, stdin);
    }
  int rv = sscanf(line, "%d %d %d", &r, &s1, &s2);
  if (rv < 3) s2 = 0;
  if (rv < 2) s1 = 0;
  if (rv < 1) r = 0;
  if (! prime(r)) {			  /* See comments above */
    printf("\nWarning: r = %d is not prime, so irreducibility test", r);
    printf(" is incomplete\n\n");
    }	
  if (r % 128 == 127) {
    printf("\n\n**** Warning: r = 127 (mod 128), the reducer code might not work correctly!\n\n");
    }
    					  /* Searches up from initial s1
    					     to s2-1 (or down to s2+1) */ 
  if (s1 < 0) exit(EXIT_SUCCESS);
  if (s1 >= r) s1 = r/2;		  /* Reasonable defaults */
  if (s2 < 0) s2 = 0;
  if (s2 > r) s2 = (s1 > r/2) ? r - 64: r/2 + 1;  

  /* Swan's theorem says trinomial is reducible if r = +- 3 (mod 8)
     and s is not 2 or r-2 */
     
  swan = ((r%8) == 3) || ((r%8) == 5);
  if (swan) {
    printf("Swan's theorem applies, only need to test s = 2\n");
    s1 = 2;
    s2 = 1;
    }   

  if (s1 < s2)
    printf("Searching from %d to %d\n", s1, s2-1);
  else if (s1 > s2)
    printf("Searching from %d down to %d\n", s1, s2+1);
  else
    printf("Empty search interval at %d\n", s1);
  if (minutes != 0) printf("Time limit %d minutes\n", minutes);
  fflush(stdout);  
  while ((s1 != s2) && (! found) && 
    ((minutes == 0) || (CPUtotal < 60*minutes))) {

    done = false;
    for (;;) {		/* Look for next s in range but not in skiplist */
      s = s1; 
      if (s1 == s2) {
        done = true;
        break;
        }
      if (s1 < s2)
        s1++;
      else
        s1--;
      if (! skips(skiplist)) break;
      }   
    if (done) break;  

    /* Set up global variables depending only on r, or on r and s */
    initialize_r_s_globals();

    if (verbose) {
    printf ("\nr %d, s %d, r-s %d\n", r, s, r-s);
    }

    if ((sodd <= 0) || (r <= sodd) || 
      (! ODD(r)) || (! ODD(sodd)) || ((r-sodd) < LIM)) {
        printf ("Illegal parameters\n");
        if (! ODD(s))
          printf("For small even s try version 1.35 or earlier\n");
        exit(EXIT_FAILURE);
        }            

    CPUtime += clockd(&cstart, false);
    CPUtotal += CPUtime;		/* CPUtotal is for all (r, s) */
    CPUtime = 0;			/* CPUtime is for current (r, s) */
    CPUtime1 = 0;			/* CPUtime1 is for current sieving */
    skt++;

    /* These local variables depend only on r */

    sizeah = (q1 >> 1) + 12;		/* r/(2*WLEN) + 4 = half size of
    					   a/b array plus safety margin */
    sizep  = q1 + 4;			/* sizep = r/WLEN + 4
    					   = size of p/q array */
    	
    sizeah += sizeah & 1;		/* Round up to even values */
    sizep  += sizep  & 1;		/* to preserve 8-byte boundaries */					   		

    sizeah = (((sizeah+7)>>3)<<3);	/* Make sizeah = 0 (mod 8)
    					   because 8 4-byte words per cache
    					   line on IBM PC (added v. 2.71)
    					   (does this help ?) */

    if (a0 == NULL) {	/* Allocate space for a, b arrays */

      CPUtime += clockd(&cstart, false);
      CPUtotal += CPUtime;
      CPUtime = 0;			/* Don't count in non-sieving time */

      a0 = fastmem(sizeah, &CPUest); /* Find a "good" a0 on the heap 
      					   	 and estimate CPU time */
      					   	 
      CPUtime += clockd(&cstart, false);/* Count call to fastmem in total */
      CPUtotal += CPUtime;		/* but not in sieving time */
      CPUtime = 0;		

      a0 += 2;		/* For optimisation in interlvr, may access a[-1] */	
      			/* See comments in fastmem re version 3.15 change */
      			
      /* Following is to align a0, a1 on 8-byte boundary */

      a1 = a0 + sizeah;
      a = a1;		/* Will cycle through a1, a0 */
      }

    g = true;

    CPUtime += clockd(&cstart, false);

    if (g) { /* Phase 1 sieve no longer done */

      setupx (a);

      for (int k = 0; g && (k < r); k++) {

      reducep(a);			/* Reduce (square of) a */

      /* Interleave in forward/reverse directions alternately */

      if (a == a1) {
        interlvf(a1, a0);		/* Forward interleave */
        a = a0;				/* Adjust pointer to data */
        }
      else {
        interlvr(a0, a1);		/* Reverse interleave */
        a = a1;
        }

	if (! g) break;  
        }

      CPUtime += clockd(&cstart, false);

      if (g) {       
       if (comparex(a)) {
        found = (! CONTINUE);
        printf("%d %d irreducible (primitive if 2^%d-1 is prime)\n", r, s, r);
        fflush(stdout);
        if (argc > 2) {
          fp = myfopen(argv[2], "a");
	  fprintf(fp, "%d %d irreducible (may be primitive)\n", r, s);
 	  fclose(fp);
          }
        else {
	  printf("%d %d irreducible (may be primitive)%s\n", r, s, log);
	  fflush(stdout);
	  }
        }
      else {

        snprintf(str8, 8+1, "%08x", (unsigned int)((a[0]<<32)>>32));
        if (verbose) {
        printf("Not irreducible/primitive, low word %s (hex)\n", str8); 
        }
        if (argc > 2) {
          fp = myfopen(argv[2], "a");
          if (s == sodd)
   	    fprintf(fp, "%d %d x%s\n", r, s, str8);
 	  else
   	    fprintf(fp, "%d %d y%s\n", r, s, str8);
 	  fclose(fp);
          }
        else {
          if (s == sodd)
   	    printf("%d %d x%s%s\n", r, s, str8, log);
	  else
   	    printf("%d %d y%s%s\n", r, s, str8, log);
 	  fflush(stdout);
 	  }
        }
      }
      
      if (verbose) {
      if (CPUtime > CPUTOL) {  
        printf("CPU time (not sieving) %.2f sec = ", CPUtime);
        printf("%.3g r^2 nsec\n", 1.0e9*CPUtime/r/r);
        }
      if (CPUtime1 > CPUTOL) {
	printf("CPU time (sieving)     %.2f sec", CPUtime1);
	if (CPUtime > CPUtime1)
          printf(", ratio %2.0f", CPUtime/CPUtime1);
        printf("\n");   
        }
        }
      CPUtotal1 += CPUtime1;  
      CPUtotal += CPUtime;
      CPUtime = 0;
      if (verbose) {
      if (CPUtotal > CPUTOL) {
        printf("\nOverall time %.2f sec = ", CPUtotal);
        printf("%.4f sec per trinomial\n", CPUtotal/skt);
        if ((CPUtotal1 > CPUTOL) && (CPUtotal1 < CPUtotal))
          printf("Overall %1.2f percent of time spent sieving\n", 
        	100.0*CPUtotal1/CPUtotal); 
	printf("\n");
        }
      fflush(stdout);
      }
      }
  
    /* Try again until irreducible/primitive trinomial found
       (or until all s checked if CONTINUE is true) */

    if (argc > 1) {
      fp = myfopen(argv[1], "w");
      fprintf(fp, "%d %d %d\n", r, s1, s2);
      fclose(fp);
      }  

    if (swan) break;
    }

  if (argc > 1) {
    fp = myfopen(argv[1], "w");
    fprintf(fp, "%d %d %d\n", r, s1, s2);
    fclose(fp);
    }

  if ((! found) && (s != -1))
    printf("Searched to %d\n", s);
  if (verbose) {
  printf("Working set after sieving about %d bytes\n", (3*r)>>4);
  printf("Dynamic storage allocation %d bytes\n", space);
  printf("%d clock calls\n", clockcalls);
  if (syscalls > 0) printf("%d system/sleep calls\n", syscalls);
  }
  return EXIT_SUCCESS;
  }

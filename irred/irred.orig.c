/* irred.c	 					irred version 3.15
 
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
     or irred version 1.50 with ODDS FALSE).

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
   	  irred restart-file log-file time-limit skip-file maxusers 
   	  	loadtolr loadtols
   	  					(all arguments optional)
Arguments:

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
	  
	  A maximum number of interactive users can be specified as the
	  fifth command line argument to override the default MAXUSERS.

	  The sixth and seventh command-line arguments can be specified
	  to override the defaults LOADTOLR and LOADTOLS.

	  For a "parallel" version using MPI, see irredpar.c

Comments on timing and memory requirements:

	Sieving quickly discards about 93% of polynomials.
	Each polynomial not discarded by sieving takes time O(r^2)
	but there may be a slow-down as r increases (this is a cache effect).

	For a 300 MHz P-II with 512KB L2 cache, times range from
	1.07r^2 nsec (for r = 44497) to 1.72r^2 nsec (for r = 3021377).

	For a 500 MHz P-III with 512KB L2 cache, times range from
	0.67r^2 nsec (for r = 44497) to 0.80r^2 nsec (for r = 3021377).

	If we measure time in units of cycles, then the time on IBM PCs
	is about 0.33r^2 if the problem is small enough for the L1 cache,
	degrading to about 0.6r^2 if the L2 cache is half as large as
	necessary (e.g. 512KB for r = 6972593).

	Detailed timing results for different versions and (r,s)
	are in the separate file times.dat.

	Space requirement is 7r/16 + (constant) bytes
	(temporarily increasing to 3r.FASTTRY/16 if FASTTRY > 2, see below).
	
	The working set after sieving is 3r/16 + (constant) bytes.

References:

        R. P. Brent, Search for primitive trinomials (mod 2),
        http://www.comlab.ox.ac.uk/oucl/work/richard.brent/trinom.html

        R. P. Brent, S. Larvala and P. Zimmermann, A fast algorithm for
        testing irreducibility of trinomials mod 2 (preliminary report),
        Report PRG-TR-13-00, Oxford University Computing Laboratory,
        30 Dec 2000. Revision to appear in Mathematics of Computation
        (posted electronically 18 Dec 2002).     
        See http://www.comlab.ox.ac.uk/oucl/work/richard.brent/pub/pub199.html  

        S. W. Golomb, Shift register sequences, Holden-Day, San Francisco,
        1967. Revised edition, Aegean Park Press, 1982.

   	J. W. Heringa, H. W. J. Bl\"ote and A. Compagner, New primitive 
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
#include <time.h>   		/* For clock */

#define TRUE 1
#define FALSE 0
#define AND &&			/* Logical AND */
#define OR ||			/* Logical OR  */
#define NOT !			/* Logical NOT */

/* BOOTH, VERBOSE, CONTINUE, GNU, MAXUSERS determine program behaviour */

#define BOOTH FALSE		/* Shortcut for my default machine.
				   Set TRUE to be nice to other users on
				   IBM PCs running Solaris, FALSE otherwise. */

#define VERBOSE TRUE		/* If true give more informative output */

#define CONTINUE FALSE		/* If true, continue after finding a
                                   primitive trinomial (useful for small r) */

#define GNU TRUE		/* Determines if copyright notice is printed
				   (but the program is copyright anyway) */

#if (NOT BOOTH)			/* Change following four defines as required */

/* ===>>> Set exactly one of the following machine types to TRUE <<<=== */

  #define IBMPC FALSE /* 32-bit CISC, little-endian, e.g. Pentium.

		      With gcc 2.95.2, compile irred.c with
		      gcc -O1 -fomit-frame-pointer -funroll-loops
		      	-o irred irred.c [irred.o]

		      With other versions of gcc, e.g. 2.7.2.3, try
		      gcc -O1 -fomit-frame-pointer
		        -o irred irred.c [irred.o]

		      where the assembler routines (if any) have been
		      assembled to give relocatables irred.o
	 	      */

  #define SPARC TRUE  /* 32-bit RISC, e.g. Old (32-bit) Sun Sparc. 
  			 Compile with gcc -O6 or (better) cc -Ofast */

  #define ALPHA FALSE /* 64-bit RISC, e.g. DEC Alpha. Compile with
			   gcc -O4 -mcpu=ev6 -Wa,-arch -Wa,ev6 -funroll-loops 
			   or cc -fast (on Compaq alphaserver).
			   Also recommended on recent Sparc processors 
			   with 64-bit options: 
			   cc -Ofast -xtarget=ultra -xarch=v9 */

  #define SGI FALSE   /* 64-bit SGI R10000. Compile with cc -64 -Ofast */

#endif	

#define ULTRA (SPARC OR SGI)	/* Try on Sparc Ultra and MIPS R12000 */  	
#define UNROLL TRUE		/* Best to set FALSE on Sun-Blade 1000,
				   but TRUE on Ultra-80 */
#define CPUINFO	TRUE  		/* If TRUE and IBMPC, give detailed CPU info */

#if BOOTH

  #define IBMPC TRUE		/* Don't change this or next three lines */
  #define SPARC FALSE		/* Don't change */
  #define SGI FALSE		/* Don't change */
  #define ALPHA FALSE		/* Don't change */

  #define MAXUSERS -3		/* Recommend 0 or -2 here, -3 for testing */

#else

  #define MAXUSERS -3		/* If MAXUSERS GE 0,  sleep while number of
				   users exceeds MAXUSERS.

			           If MAXUSERS EQ -1, sleep when load
			             exceeds LOADTOLR and until load
			             decreases below LOADTOLS.

				   If MAXUSERS EQ -2, sleep while number
				   of users exceeds zero OR load exceeds
				   LOADTOLR (i.e. conditions 0 OR -1).

  				   If MAXUSERS LE -3, never sleep
  				   (this is the recommended default).

				   The default may be overridden
				   by the fifth command-line argument */

#endif

#define LOADTOLR 1.2		/* Some value greater than the number of
				   processors (assumed to be one here) */
#define LOADTOLS 0.9		/* Some value less than LOADTOLR */

/* Following are only relevant on IBM PCs */

#if BOOTH
  #define LINUX FALSE
#else
  #define LINUX TRUE		/* Set TRUE if running under Linux,
				   FALSE otherwise. Relevant to choice
				   of FASTTRY (see below). */
#endif

/*

Comments on different versions:

   Version 3.15 has
   
   1) Extra 4 words allocated before arrays p and q in main
      and a0 in main/fastmem to avoid (probably harmless) valgrind 
      diagnostics - thanks to Julian Seward and Andrew Tridgell for 
      pointing out the problem.

   Version 3.14 has
   
   1) UNROLL flag to permit 4-way unrolled version of reducer
      (good on Sparc Ultra-80).

   Version 3.13 has
   
   1) ULTRA flag to permit another version of reducer
      (good on Sparc Ultra-80 and MIPS R12000).

   Version 3.12 has
   
   1) Argument of prime(n) is type int instead of ULONG
      to avoid diagnostic on 64-bit (e.g. SGI) machines.

   Version 3.11 has
   
   1) Better documentation, no significant changes to the code.

   Version 3.10 has

   1) Explicit reference to the GNU General Public License (previously
      this information was communicated separately). 

   2) Test for primality of r (a warning is printed if r is composite).
      The irreducibility test assumes that r is prime, and is incomplete
      if r is composite (see comments above). Earlier versions assumed that
      r was prime without any check.

   Version 3.07 has
   
   1) More accurate output (only claims irreducible rather than primitive).
      The earlier versions assume r is a Mersenne exponent (so 2^r-1 prime),
      this version only assumes that r is prime and tests if the trinomial
      is irreducible. In general, if 2^r-1 is composite, further tests 
      depending on the factors of 2^r-1 are necessary to check primitivity.
      If 2^r-1 is prime, no further checks are necessary.
      
   Version 3.06 has
   
   1) Skip list handling fixed so can skip a single item 
   (previously low and high bounds equal terminated list processing).

   Versions 3.04 and 3.05 have

   1) New CPU info for booth (1-35, 51-55) and tosca.

   Version 3.03 has

   1) Option to modify load tolerances using command-line arguments.

   Version 3.02 has

   1) Improved version of sysload, no longer limited to range [0, 2.55].

   2) SLEEPTIME becomes two different constants SLEEPU, SLEEPL as the load
      is likely to change more often than the number of users.

   3) Option -2 for MAXUSERS (checks both userkt and sysload).

   Version 3.01 has

   1) More efficient versions of userkt and sysload, avoiding use
      of a temporary file.

   Version 3.00 has

   1) uptime instead of who|wc to count interactive users (this can
      give different results if one user has several sessions).

   2) Counters for clock and system calls.

   3) Better error handling for system calls (e.g. will not bomb if
      the file /tmp/irred.tmp is not writeable).

   4) Better error handling for command-line arguments, e.g.  skip file
      (so can use 0 or /dev/null if don't desire skip file (4th argument)
      but need 5th argument). All sscanf calls are now checked to see
      if the expected number of fields was found.

   5) Name changed: irredg -> irred (as it used to be).

   Version 2.93 has

   1) Flag BOOTH as shortcut for options on boothxx machines.

   2) option to sleep while load average is high (set MAXUSERS = -1).
      
   Versions 2.92 and 2.91 are experimental variants on the "sleeping" theme.   

   Version 2.90 has 
   
   1) option to go to sleep while the number of interactive users exceeds
      a tolerance (set MAXUSERS to a nonnegative value).

   Version 2.82 has
   
   1) Changed format of log file - hex numbers now have
      leading zeros instead of blanks. For example, 
      "x   12345" -> "x00012345".
      
      Old log files can be checked and converted to the
      format used by version 2.82 with the separate program "fixlog".

   Version 2.81 has
   
   1) Changed format of log file - if sodd = r-s > r/2 use
      r s yabcdefgh instead of r sodd xabcdefgh

   Version 2.80 has
   
   1) Dynamic choice of sieving cutoff, only use NEXTRA for upper bound.
      This should improve overall efficiency because the cutoff can 
      adapt to different machines/compilers and also to exceptional
      (r, s) values, e.g. if s is small or close to r/2.
   
   2) Printing of skip list commented out (as may be too long).

   Version 2.76 has

   1) Constant NEXTRA reduced from 4 to 3 in some cases, including
      (ASM AND IBMPC). This should save about 3% on average for time 
      per trinomial on IBM P-III such as booth, at the expense of a 
      small inconsistency in the log files produced on different machines.

   2) Introduction of LINUX flag to distinguish between Linux and Solaris
      (only use FASTTRY > 1 under Linux).

   Version 2.75 has

   1) Optional empirical choice of a "good" memory location for data arrays
      by timing of some dummy iterations. Perhaps surprisingly, this helps
      on Portland (P-II) when r is large. May depend on the operating system
      (seems less effective under Solaris than Linux).
      See FASTTRY and fastmem below.

   Version 2.72 has

   1) Memory allocation on stack instead of heap (this turned out to be
      no faster, so reverted to heap in later versions).

   Version 2.71 has
   
   1) Option CLEAR to clear space allocated by malloc before use
      (may also have effect of clearing L1 and L2 caches).

   2) Rounding up of size of data arrays to multiple of cache line
      size (32 bytes) for IBM PC (this did not help, so omitted in
      later versions).

   Version 2.70 has
   
   1) New MMX routine reducemx, similar to reducemr but making
      different alignment assumptions.
      
   2) Interface routine reduceml to call reducemr/reducemx as
      appropriate, ensuring the correct data alignment.
      
   3) prefetch instructions removed in reducer to speed small cases
      on P-II machines.  [Restored but commented out in version 2.76]

   Version 2.65 has

   1) MMX routine reducemr used instead of reducer in some case.

   Version 2.61 has
   
   1) Call to reducer if shift zero but (IBMPC AND ASM)
      (saves about 5% in the special case of shift zero,
       which should occur about 1 in 32 cases).

   Version 2.60 has 
   
   1) interlvr restored as versions without it were 
      sometimes slower than versions with it (at least this is
      true for r = 3021377 on booth, where 3r/16 bytes is very 
      close to the cache size - see comments re version 2.40 below).

   2) Space requirement decreased to 7r/16 bytes and
      working set after sieving decreased to 3r/16 bytes.

   Versions 2.51-2.54 have minor variations in interlvf MMX code.

   Version 2.50 has

   1) interlvr abolished, instead we use a cycle of length three
      involving interlvr to get better cache performance in reducer.

   2) Space requirement increased to r/2 bytes (formerly 7r/16 bytes),
      working set after sieving increased to r/4 bytes (formerly 3r/16 bytes).

      Version 2.50 seems about 10% slower than version 2.40 on booth, 
      for r = 3021377.  Slightly faster for r = 6972593 so may revert 
      to version 2.50 in the future.

   Version 2.40 has

   1) Call to MMX routine reducemr removed as non-MMX reducer is faster.
      [This may depend on data alignment. The problem is that we can't ensure 
       that data is aligned on 8-byte boundaries in calls to reducemr.]

   Version 2.34 has

   1) Improved reducer loop in irred.s (non-MMX code).

   2) Call to reducer instead of reducemr.

   Version 2.33 has

   1) Unrolled reducer restored from version 2.00 because faster
      on SPARC (no change in MMX version).

   Version 2.32 has

   1) Reordered loops in MMX code (version 2.31 is slightly different)

   Version 2.30 has

   1) Combined align and interleave, alternating forward and back.
      This gives improved performance on IBM PC, not much difference
      on other machines.

   2) Space requirement increased to 7r/16 bytes (formerly 3r/8 bytes)
      but working set after sieving is the same, 3r/16 bytes.

   3) Options FORWARD, MMX, TABLE, UNROLL eliminated as now redundant.

   Version 2.20 has
   
   1) FORWARD parameter for optional forward loop in alignmr[f] (MMX version)
      and aligner (C non-unrolled version).
      This improves cache performance in some cases.

   Version 2.12 has
   
   1) Faster (at least on Portland PC) interlvm with shorter inner loop
      (may be slower on booth)
	
   Version 2.11 has
   
   1) Replacement of some shifts and ors by adds in interlvm

   Version 2.10 has
   
   1) Option to use ASM without MMX, and new non-MMX IBM assembler
      routines interlv2, interlv3

   Version 2.01 has

   1) Improved comments.

   2) Improved loop control in reducemr, alignmr (now OK to call with
      kt negative).

   Version 2.00 has
   
   1) Improved algorithm with reduced memory references (C only so far).
      This algorithm is faster overall than the one used in earlier versions
      although sieving is slightly slower.
      
   2) Restriction that min(s, r-s) can not be too small (previously
      only r-s was restricted) and that r is odd. Use version 1.35 or 
      earlier if necessary to circumvent these restrictions.
      
   3) Because even s is replaced by r-s, log files produced by version 2.00
      are slightly different from those produced by version 1.35 and earlier.
      
   4) PRELOAD and TABLEZ flags abolished.   
   
   Version 1.50
     
   1) Optionally uses odd s (useful for checking version 2.00 and later),
      otherwise the same as version 1.35.

   Version 1.35 has
      
   1) Better detection of illegal parameters (if ASM AND IBMPC)
         
   Version 1.34 has
   
   1) More informative output re CPU type (if IBM PC).

   Version 1.33 has
   
   1) Faster code in case that r-s is divisible by 32 and IBM PC assembler
      is available (previously this case did not call assembler).

   2) ASM flag reinstated to replace MMX and MMX2 flags.

   Version 1.32 has
   
   1) More informative output - times for last sieve iteration
      and fraction of time spent sieving.

   2) Bug in versions 1.30 & 1.31 (which gave wrong "n" for log on stdout
      if printed in phase 2) fixed.
      
   3) Storage allocation bug (which caused segmentation fault if only one
      command-line argument on some machines) fixed.
        
   Version 1.31 has
   
   1) Unrolled option in reducer (may help on Sun etc).

   Version 1.30 has 
   
   1) Faster relprime calling reducer for inner loop
      (with assembler version for IBM PC).
   2) Introduction of NEXTRA and abolition of RATIO.
   3) Storage requirement reduced to 3r/8 + O(1) bytes.

   Version 1.25 has
   
   1) More CPU information in case of IBM PC.
   2) Two assembler versions - 
      one (irrednas.s) uses prefetching (works on PIII and above)
      the other (irredasm.s) without prefetching.
      (Note: later irredasm.s was abolished and irrednas.s renamed irred.s)

   Version 1.24 has
   
   1) MMX2 option to allow reduce2.s in case q1 and q1-q2 both odd.

   Version 1.23 has
   
   1) Alignment of most 8-byte accesses on 8-byte boundaries.
   2) Abolition of PORTLAND, ASM, OLDGCC, OLDR options.
   3) Merging of MMXR and MMXS options to MMX.

   Version 1.22 has
   
   1) Larvala's new MMX routines (renamed squarem and reducem)

   Version 1.21 has
   
   1) Alternative version of MMX (reduce2.s) modified RPB  

   Version 1.20 has

   1) MMXR and MMXS options for use of Multi Media extension (mmx) 
      instructions on Pentium etc (written by Samuli Larvala).

   Version 1.12 has
   
   1) Slightly faster reducea (though still some room for improvement ?)
      Set OLDR to use the old version of reducea (may be faster on machines
      other than IBM PC).
      
   2) Comments and output improved (2^p is never prime for p > 1 !).   

   Version 1.11 has
       
   1) ASM flag added and square.s implemented (IBM PC version only)
   2) comments improved 

   Compared to the previous versions 1.0 and earlier (unnumbered),
   version 1.1 has

   1) a fix to avoid timer overflow for large r.
   2) memory requirements for sieving reduced by about 46%.
   3) addition of a "skip file" as optional 4th command-line argument.
   4) various small enhancements and improvements in the comments.

Compilation flags:

   Set following flags to optimise performance depending on machine type
   and compiler.
   
   On an IBM PC which can execute MMX instructions it is best 
   to set ASM = TRUE and use the assembler routines in irred.s
   (version 3.10 or higher). 
   
*/

#define ASM	TRUE			   /* Use external assembler routines.
					      At present these are only
					      available for the IBM PC and 
					      use MMX instructions. In this 
					      case we estimate the inner loops
					      take 0.58*r^2 instructions and 
					      at least 0.23*r^2 cycles.
					      For problems which fit in the
					      L1 cache, the observed time
					      is about 0.33*r^2 cycles. */

#define W64   	(ALPHA OR SGI)		   /* For 64-bit version. Faster than
					      32-bit version if available. */

#if IBMPC
 #define REGISTER			   /* Omit register declaration on PC */
#else
 #define REGISTER register		   /* Include it on other machines */
#endif

#if (ASM AND IBMPC)
 #define LIM 256			   /* Lower bound on s, r-s */
#elif UNROLL
  #define LIM (10*WLEN)
#elif ULTRA
 #define LIM (6*WLEN)
#else			
 #define LIM (2*WLEN) 
#endif

#define CLEAR TRUE			   /* To clear allocated space */

/* Some constants */

#define TIMEOUT 10		/* Seconds to timeout when opening files */
#define SLEEPU  10		/* Seconds to sleep if too many users */
#define SLEEPL  1		/* Seconds to sleep if load too high */
#define CPUTOL 0.01		/* Times less than CPUTOL sec are negligible */
#define SMALLKT 400		/* Tolerance for small kt in reduceml
				   determined empirically on a P-II */
#if (IBMPC AND LINUX)				

  #define FASTTRY 16		/* Number of a0 to try in fastmem.
				   Set to 1 if trivial selection desired. */
#else

  #define FASTTRY 1		/* Selection of a0 only seems useful on IBM PC
  				   running under Linux (not Solaris),
  				   but in any case do one timing run to 
  				   estimate non-sieving time and thus 
  				   determine a good cutoff for sieving. */
#endif

#define SMALLR  200000		/* Trivial selection if r < SMALLR */
#define CPUTEST 1		/* Seconds for each trial in fastmem.
				   Assume that CPUTOL is much less than 
				   CPUTEST. */

#define EQ ==
#define NE !=
#define GE >=
#define GT >
#define LE <=
#define LT <
#define ERROR 1
#define OK 0

/* Useful macro definitions */

#define ABS(i)          (((i)>0)   ? (i) : (-(i)))
#define MAX(i,j)        (((i)>(j)) ? (i) : (j))
#define MIN(i,j)        (((i)<(j)) ? (i) : (j))
#define RDUP(i,j)       ((((i)+(j)-1)/(j))*(j))
#define CEIL(i,j)       (((i)+(j)-1)/(j))
#define ODD(i)		((i)&1)

/* Type definitions */

typedef char BOOLEAN;
typedef unsigned char  UCHAR;	/* Assumed to be 8 bits */
typedef unsigned short USHORT;  /* Assume 16 bits - no longer required */
typedef unsigned int   UINT;	/* Assumed to be (at least) 32 bits */  	
typedef unsigned long  ULONG;	/* Should be 32 or 64 bits */

#define WLEN  (8*sizeof(ULONG))	/* Bits in a long word, should be 32 or 64 */
#define WLENM (WLEN-1)		/* Ditto less 1, i.e. 31 or 63 */
#define WD    (W64 + 5)		/* 5 or 6, so WLEN = 2^WD */

	
#define	NEXTRA 10		/* Do up to NEXTRA GCD computations for
				   2^(degree) GE r. Usually 3 or 4 is optimal.
				   Since version 2.80 the cutoff has been 
				   estimated at runtime (see DYNAMIC below)
				   and NEXTRA is only used as an upper bound.
				   This can result in minor differences in 
				   the log files produced in different runs
				   or on different machines. */

#define NMAX (WLENM + NEXTRA)	/* Sieve over x^kn - 1 for kn = 2^n-1,
				   2^n < r, n LE NMAX (but see also NEXTRA) */

#define DYNAMIC TRUE		/* If TRUE, dynamically determine sieving
				   cutoff with NEXTRA as upper bound (see
				   above). If FALSE, use exactly NEXTRA. */

#define MC 256			/* Max string length */

/* Struct for the skip list (a singly-linked list of pairs) */

struct skip {
  int low, high;
  struct skip *next;
  };

/* Global variables */

int r, q1, sodd;			/* Main parameter is r */

double CPUtime;				/* For timer */
clock_t cstart;

	/* 32 bits should suffice for these counters */

UINT slept = 0;				/* Counts seconds slept */
UINT syscalls = 0;			/* Counts calls to system/sleep */
UINT clockcalls = 0;			/* Counts calls to clock */
UINT space = 0;				/* Counts bytes allocated by malloc */
UINT userkt0 = 0;			/* Counts calls to userkt */
UINT userkt1 = 0;			/* Cumulative user count */

int maxusers = MAXUSERS;		/* Default may be overridden by
  				           fifth command-line argument */
float loadtolr = LOADTOLR;		/* Default may be overrridden by
					   sixth command-line argument */
float loadtols = LOADTOLS;		/* Default may be overridden by
					   seventh command-line argument */
/* Routines start here */

double clockd(starta, first)
clock_t *starta;
BOOLEAN first;

/* If first = TRUE, initialises *starta and returns zero.
   If first = FALSE, returns time in sec since last call and updates starta. 
   Note that this routine should be called at least each 4290 sec on
   IBM PC (under Linux) since CLOCKS_PER_SEC = 1000000 and 32-bit integer
   overflow may occur if called infrequently */
  
  {
  clock_t stop;
  ULONG diff;
  stop = (clock_t) clock();
  clockcalls++;
  if (first) *starta = stop;
  diff = (ULONG)(stop - *starta);	/* May wrap around in 2^32/10^6 sec */
  *starta = stop;
  return ((double)diff*((double)1/(double)CLOCKS_PER_SEC));
  }


char *mymalloc(size)
int size;

/* Same as malloc but gives error exit if can not allocate space.
   Optionally clears space allocated. */

  {
  char *ptr;
#if CLEAR
  int j;
#endif
  space += size;
  ptr = (char *)malloc(size);
  if (ptr EQ NULL) {
    printf("malloc unable to allocate %d bytes\n", space);
    exit(ERROR);
    }
#if CLEAR
  for (j = 0; j < size; j++) ptr[j] = (char)0;
#endif
  return(ptr);
  }

UINT mysleep(n)

UINT n;

/* Sleeps for (up to) n seconds and returns the time slept.
   Assume sleep(n) returns (n - (seconds slept)). */

  {
  syscalls++;			/* Count call to sleep as a system call */
  return (UINT)(n - sleep(n));
  }

UINT userkt()

/* Returns the number of interactive users using call to uptime,
   and updates some counters. Assumes the number of users is < 256.

   Assumes that uptime returns something like

   12:11pm  up 1 day, 50 min,  5 users,  load average: 0.31, 0.80, 0.59
                               ^                       ^^^^  ^^^^  ^^^^
   (awk field)                 NF-6                    NF-2  NF-1   NF

   To avoid using a temporary file (as in version 3.00) we return the
   result from uptime|awk using exit. Note that exit shifts its return
   value (mod 256) left 8, so we have to shift it right 8. */

  {
  int rv;
  syscalls++;
  rv = system("exit `uptime|awk '{print $(NF-6)}'`") >> 8;
  userkt0++;
  userkt1 += rv;
  return (UINT)rv;
  }

UINT checkusers(userstol)

int userstol;

/* Sleeps while number of users exceeds userstol,
   returns the total number of seconds slept */

  {
  UINT slept;
  slept = 0;
  while (userkt() > userstol)
    slept += mysleep(SLEEPU);		/* Sleep and increment counter */
  return slept;
  }

BOOLEAN sysload(tol)			/* BOOLEAN since version 3.02 */
float tol;

/* Returns TRUE if (system load) > tol, FALSE otherwise,
   using a call to uptime, and increments the syscalls counter.

   Assumes that uptime returns something like

   12:11pm  up 1 day, 50 min,  5 users,  load average: 0.31, 0.80, 0.59
                                                       ^^^^  ^^^^  ^^^^
   (awk field)                                         NF-2  NF-1   NF

   where the last three fields are the load averages. We use the left-most
   load average (i.e. the average over the last 60 seconds).

   We use a variant of the method used in userkt.
   awk processes the output of uptime. We use sprintf
   to encode tol in the string passed to sh via a system call. 
   In awk's notation, $(NF-2) is something like "0.31,". 
   We remove the comma using substr and compare with tol. 
   The Boolean result is returned by exit.
   exit shifts the return value (mod 256) left 8 so we shift right 8.
   
   Probably more efficient (but system-dependent) methods: 
     under Linux, read /proc/loadavg;
     under Solaris, call getloadavg.
*/
  {
  char str[MC];				/* 95 is enough */

#if (MC < 95)
  printf("Increase MC from %d to at least 95\n", MC);
  exit(ERROR);
#endif  

  sprintf(str, "%s%1.2f%s", 
    "exit `uptime|awk '{if (substr($(NF-2),0,length($(NF-2))-1) > ", tol, 
			  ") {print 1} else {print 0}}'`");
  syscalls++;
  return (BOOLEAN)(system(str) >> 8);
 }

UINT checkload()

/* Sleeps if average system load exceeds loadtolr (including this process),
   wakes after load diminishes to loadtols. One a single-processor machine 
   the constants should satisfy 0 < loadtols < 1 < loadtolr.

   Returns the total number of seconds slept */

  {
  UINT slept;
  slept = 0;
  if (sysload(loadtolr)) {      /* Running tolerance loadtolr here */
    while (sysload(loadtols)) { /* Sleeping tolerance loadtols here */
      slept += mysleep(SLEEPL);
      }
    }
  return slept;
  }

BOOLEAN sysload2(tol)
float tol;

/* Slight variant on sysload. 
   Returns TRUE if (userkt > 0) OR ((system load) > tol), FALSE otherwise,
   using a call to uptime, and increments the syscalls counter.
   Equivalent to
   
     (userkt() GT 0) OR sysload(tol)

   but more efficient.
*/

  {
  char str[MC];				/* 127 is enough */

#if (MC < 127)
  printf("Increase MC from %d to at least 127\n", MC);
  exit(ERROR);
#endif  

  sprintf(str, "%s%s%1.2f%s", 
    "exit `uptime|awk '{if ($(NF-6) > 0) {print 1}",
    " else if (substr($(NF-2),0,length($(NF-2))-1) > ", tol, 
			  ") {print 1} else {print 0}}'`");
  syscalls++;
  return (BOOLEAN)(system(str) >> 8);
 }

UINT checkload2()

/* Sleeps if any interactive users OR 
   average system load exceeds loadtolr (including this process),
   wakes after load diminishes to loadtols and no interactive users. 
   One a single-processor machine the constants should satisfy 
   0 < loadtols < 1 < loadtolr.

   Returns the total number of seconds slept */

  {
  UINT slept;
  slept = 0;
  if (sysload2(loadtolr)) {      /* Running tolerance loadtolr here */
    while (sysload2(loadtols)) { /* Sleeping tolerance loadtols here */
      slept += mysleep(SLEEPU);  /* Assume SLEEPL LE SLEEPU */	 
      }
    }
  return slept;
  }

BOOLEAN comparex(a)

/* Returns TRUE if poly a of degree r-1 is x */

ULONG *a;
  {
  ULONG mask1;
  int j;
  if (a[0] NE 2L) return(FALSE);
  for (j = q1-1; j > 0; j--) {
    if (a[j] NE 0) return(FALSE);
    }
  mask1 = (ULONG)(~0L) >> (WLENM - ((r-1) & WLENM));  
  if ((a[q1] & mask1) NE 0) return(FALSE);
  return(TRUE);
  }
  
#if (ASM AND IBMPC)

extern void cpuinfo(int flag, int* result);
				/* Returns info from cpuid instruction */
extern BOOLEAN checkmmx();	/* Returns TRUE if MMX instruction available */

#elif UNROLL

void reducer(a, b, kt, shift, prev)

/* 4-way unrolled version of reducer, optimised for Sparc Ultra-80.
   Note that LIM must be at least 10*WLEN. 
   Called by relprime and reducep. Assumes 0 < shift < WLEN. */

ULONG *a, *b;
int kt, shift;
ULONG *prev;		/* Previous -> last value of new */

  {
  REGISTER int j, shiftc;
  REGISTER ULONG bj, bj2, bj3, bj4, old, new, old2, new2, old4, new4;
  new = *prev;
  shiftc = WLEN - shift;
  for (j = kt; (j GE 0) AND ((j & 3) NE 3); j--) { /* Up to 3 iterations */
    old = new;
    new = a[j];
    b[j] ^= (new >> shift) | (old << shiftc);
    }

  /* Now j+1 is divisible by 4 */
  
  old2 = a[j];			/* Preload some registers */
  new2 = a[j-1];  
  old4 = a[j-2];
  new4 = a[j-3];
  bj   = b[j];
  bj2  = b[j-1];
  bj3  = b[j-2];
  bj4  = b[j-3];
  				/* Main loop.  Try to load ahead as far
  				   as possible. This will involve some
  				   harmless "out of bound" loads */
  for (; j GE 0; j -= 4) {
    old  = old2;
    old2 = a[j-4];
    b[j] = bj ^ ((old >> shift) | (new << shiftc));
    bj   = b[j-4];
    new  = new2;
    new2 = a[j-5];
    b[j-1] = bj2 ^ ((new >> shift) | (old << shiftc));
    bj2  = b[j-5];
    old  = old4;
    old4 = a[j-6];
    b[j-2] = bj3 ^ ((old >> shift) | (new << shiftc));
    bj3  = b[j-6];
    new  = new4;
    new4 = a[j-7];
    b[j-3] = bj4 ^ ((new >> shift) | (old << shiftc));
    bj4  = b[j-7];
    }
  *prev = new;
  return;
  }

#elif ULTRA

void reducer(a, b, kt, shift, prev)

/* Unrolled version of reducer, good on Sparc Ultra-80 and R12000.
   Called by relprime and reducep. Assumes 0 < shift < WLEN. */

ULONG *a, *b;
int kt, shift;
ULONG *prev;		/* Previous -> last value of new */

  {
  REGISTER int j, shiftc;
  REGISTER ULONG bj, bj2, old, new, old2, new2;
  new = *prev;
  shiftc = WLEN - shift;
  for (j = kt; (j GE 0) AND ((j & 1) EQ 0); j--) { /* One iteration if
  						      kt even */
    old = new;
    new = a[j];
    b[j] ^= (new >> shift) | (old << shiftc);
    }

  old2 = a[j];			/* Preload some registers */
  new2 = a[j-1];  
  bj   = b[j];
  bj2  = b[j-1];
  				/* Main loop.  Try to load ahead as far
  				   as possible. This will involve some
  				   harmless "out of bound" loads */
  for (; j GE 0; j -= 2) {
    old = old2;
    old2 = a[j-2];
    b[j] = bj ^ ((old >> shift) | (new << shiftc));
    bj = b[j-2];
    new = new2;
    new2 = a[j-3];
    b[j-1] = bj2 ^ ((new >> shift) | (old << shiftc));
    bj2 = b[j-3];
    }
  *prev = new;
  return;
  }

#else			

void reducer(a, b, kt, shift, prev)

/* Unrolled version.
   Simplified version of reducea, called by relprime and reducep.
   Assumes 0 < shift < WLEN. */

ULONG *a, *b;
int kt, shift;
ULONG *prev;		/* Previous -> last value of new */

  {
  REGISTER int j, shiftc;
  REGISTER ULONG old, new;
  new = *prev;
  shiftc = WLEN - shift;
  for (j = kt; (j GE 0) AND ((j & 1) EQ 0); j--) { /* One iteration if
  						      kt even */
    old = new;
    new = a[j];
    b[j] ^= (new >> shift) | (old << shiftc);
    }
  for (; j GE 0; j -= 2) {
    old = a[j];					/* old <-> new here ! */
    b[j] ^= (old >> shift) | (new << shiftc);
    new = a[j-1];
    b[j-1] ^= (new >> shift) | (old << shiftc);
    }
  *prev = new;
  return;
  }

#endif

#if (ASM AND IBMPC)

extern void reducer(ULONG* a, ULONG* b, int kt, int shift, ULONG* prev);
				/* non-MMX version */

extern void reducemr(ULONG* a, ULONG* b, int kt, int shift, ULONG* prev);
extern void reducemx(ULONG* a, ULONG* b, int kt, int shift, ULONG* prev);
				/* MMX versions */

void reduceml(a, b, kt, shift, prev)

/* Interface to reducer/reducemr/reducemx. Ensures proper data alignment. */

ULONG *a, *b;
int kt, shift;
ULONG *prev;

  {
  if (kt LT SMALLKT)			
      reducer(a, b, kt, shift, prev);	/* Faster in small cases */
      
  /* Tests for the four cases below are (1279, 418), (3217, 576),
     (2281, 1029) and (23209, 9739). For testing, reduce SMALLKT to 1. */ 

  else if (((ULONG)b & 4L) EQ 0) {	/* Alignment of b is OK here */
    if (((ULONG)a & 4L) EQ 0)
      reducemr(a, b, kt, shift, prev);	/* Alignment of a is OK here */
    else
      reducemx(a, b, kt, shift, prev);	/* Here a is mis-aligned */
    }
  else {				/* Here b is mis-aligned */
    if (((ULONG)a & 4L) EQ 0) 
      reducemx(a+1, b+1, kt-1, shift, prev);	/* a, b+1 aligned */
    else
      reducemr(a+1, b+1, kt-1, shift, prev);	/* a+1, b+1 aligned */
    reducer (a, b, 0, shift, prev);	/* Final iteration */
    }
  }

#endif

void reducep(a)	

/* Reduces implicit square of polynomial a of degree < r "in place"
   given x^r = x^sodd + 1 (mod 2), result left in scrambled order
   0 2 4 ... 1 3 5 ...
   
   Assumes 0 < sodd < r AND r-sodd GE 2*WLEN AND ODD(r) AND ODD(sodd).
    
   Uses about (2.load + 1.store)r/(2.WLEN) memory ops.
   
   RPB, 20000815. */

ULONG *a;

  {
  ULONG new;
  REGISTER ULONG temp;
  REGISTER int j;
  int alpha, delta;		/* Could be global */
  int deltaw, deltaq, deltaqc;	/* ditto */
  int q1, q4;			/* ditto */
  ULONG mask1, mask2;		/* ditto */
  
  alpha = r >> 1;		/* alpha = (r-1)/2 */
  delta = (r - sodd) >> 1;	/* delta = (r - sodd)/2 */
  deltaw = delta >> WD;		/* deltaw = delta div WLEN */
  deltaq = delta & WLENM;	/* deltaq = delta mod WLEN */
  deltaqc = WLEN - deltaq;	/* Special case if deltaq is zero */
  q1 = (r-1) >> WD;		/* q1 = (r-1) div WLEN */
  q4 = alpha >> WD;		/* q4 = alpha div WLEN */
  
  mask1 = (ULONG)(~0L) >> (WLENM - ((r-1) & WLENM));
  				/* mask1 has WLEN-1 - ((r-1) mod WLEN)
  				   zero bits in high positions */
  mask2 = (~1L) << (alpha & WLENM);
  				/* mask2 has (alpha mod WLEN) + 1 zero bits
  				   in low positions */

  /* To be safe, mask any high bits of a which are irrelevant */

  a[q1] &= mask1;
  a[q1+1] = 0;			/* In case reducemx called */
  a[q1+2] = 0;			/* Ditto */
  new = 0;			/* Assumed by reducemx */
  
  if (deltaq EQ 0) {		/* Special case deltaqc EQ WLEN */
    
#if (ASM AND IBMPC)
    reduceml(a+q4+1, a+q4+1-deltaw, q1-q4-1, deltaq, &new);
#else
    for (j = q1; j > q4; j--)	/* C reducer does not work here */
      a[j-deltaw] ^= a[j];
#endif
    a[q4-deltaw] ^= a[q4] & mask2;
    }

  else {			/* Usual case, deltaqc LT WLEN */

#if (ASM AND IBMPC)    
				/* Could call reduceml or reducer */

    reduceml (a+q4+1, a+q4+1-deltaw, q1-q4-1, deltaq, &new);
#else    
    reducer (a+q4+1, a+q4+1-deltaw, q1-q4-1, deltaq, &new);
#endif
    /* The last two iterations are special as need to mask some bits */

    temp = new;
    new = a[q4] & mask2;
    a[q4-deltaw]   ^= (new >> deltaq) | (temp << deltaqc);
    a[q4-deltaw-1] ^=  new << deltaqc;
    }
  }
  
BOOLEAN relprime (p, np, q, nq)

/* Returns TRUE if polynomials p and q (of degrees np and nq)
   are relatively prime (mod 2). Destroys p and q in the process.
   Returns FALSE if np < 0 or nq < 0.
   Time is O(np.nq). */
   
ULONG *p, *q;
int np, nq;
    
  {
  int j, k, del, delq, delr, delrc, high;
  int wkt = 0;
  REGISTER ULONG new;
  ULONG *t;
  ULONG prev;

  if ((np LT 0) OR (nq LT 0)) return(FALSE);
  
  while (TRUE) {
    if ((wkt -= nq) < 0) {
      CPUtime += clockd(&cstart, FALSE);
      					/* Avoid timer overflow by calling */
      wkt = 0x40000000; 		/* clockd occasionally */

      if (maxusers EQ -2) 		/* Check load and users here */
	slept += checkload2();
      else if (maxusers EQ -1)		/* Check load here */
        slept += checkload();		/* Sleep while load high */
      else if (maxusers GE 0)		/* Check users here */
        slept += checkusers(maxusers);  /* Sleep while users high */
      } 
    if (np LT nq) { 			/* p <-> q etc */
      k = np; np = nq; nq = k;
      t = p;  p = q; q = t;
      }  
    del = np - nq;
    delq = del >> WD;			/* delq = del/WLEN */			
    delr = del & WLENM;			/* delr = del%WLEN */			
    if (delr EQ 0) {			/* Simple case, word aligned */
      for (j = (nq >> WD); j GE 0; j--) 
        p[j+delq] ^= q[j];  
      }
    else {  				/* Need bit-alignment here */
      delrc = WLEN - delr;		/* in 1..(WLEN-1) since delr NE 0 */
      high = (np >> WD) - delq;
      prev = q[high];			
      reducer(q, p+delq+1, high-1, delrc, &prev); 	
      p[delq] ^= (prev << delr);
      }    
    for (j = np; j GE 0; j--) { 	/* Find new deg(p) */
      new = p[j >> WD];
      if (new EQ 0L) {			/* Speed up search over zero words */
        np -= j & WLENM;
        j-= j & WLENM;		
	}
      else
        if ((new >> (j & WLENM)) &1L) break;
      np--;
      }
    for (j = nq; j GE 0; j--) {	/* and deg(q) (not usually necessary) */
      if ((q[j >> WD] >> (j & WLENM)) &1L) break;
      nq--;
      }
    if (np LT 0) return (nq EQ 0);
    }
  }    

void setupx(a)

/* Sets up a = x (polynomial of degree r-1, all mod 2) */

ULONG *a;
  {
  int j;
  for (j = q1; j GT 0; j--)
    a[j] = 0L;
  a[0] = 2L;			/* 0...010 represents x */
  }

#if (ASM AND IBMPC)

/* Here use assembler versions of interlv which use the MMX
   instructions (available on most Pentiums and above, but not 486 machines).
*/

extern void interlvf(ULONG* a, ULONG* b, int r);
extern void interlvr(ULONG* a, ULONG* b, int r);

#elif W64

void interlvf(a, b, r)

/* 64-bit version of interleave. Loop index runs up.

   If bits 0, 2, 4, ... , r-3, r-1, 1, 3, 5, ..., r-4, r-2 in a,
   moves them to b in correct order.

   Work is about (1.load + 1.store + 32.ops)r/WLEN.

   RPB, 20000907 */

ULONG *a, *b;
int r;

  {
  REGISTER int j, s1, s2, q4;
  REGISTER ULONG t, u, v, w, next1, next2, old, new;
  REGISTER ULONG c0, c1, c2, c3, c4, c5;
  int alpha;

  c0 = 0x00000000FFFFFFFFL;		/* Some 64-bit constants */
  c1 = 0x0000FFFF0000FFFFL;
  c2 = 0x00FF00FF00FF00FFL;
  c3 = 0x0F0F0F0F0F0F0F0FL;
  c4 = 0x3333333333333333L;
  c5 = 0x2222222222222222L;

  alpha = r >> 1;			/* alpha = (r-1)/2 */
  q4 = (alpha+1) >> WD;			/* q4 = (alpha+1) div WLEN */
  s1 = (alpha+1) & WLENM;		/* s1 = (alpha+1) mod WLEN */
  s2 = WLENM - s1;			/* In [0, WLEN) */
  
  next1 = a[0];
  old = a[q4];
  new = a[q4+1];

  for (j = 0; j LE q4; j++) {

    next2 = (old >> s1) | ((new << 1) << s2);	/* Beware case s2 EQ 63 */

    u = next1>>32;			/* High order 32 bits low part of a */
    w = next2>>32;			/* Ditto high part of a */
    t = next1 & c0;			/* Low order 32 bits low part of a */
    v = next2 & c0;			/* Ditto high part of a */

    next1 = a[j+1];
    old   = new;
    new   = a[j+q4+2];

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
  
void interlvr(a, b, r)

/* 64-bit version of interleave. Loop index runs down (compare interlvf).

   If bits 0, 2, 4, ... , r-3, r-1, 1, 3, 5, ..., r-4, r-2 in a,
   moves them to b in correct order.

   Work is about (1.load + 1.store + 32.ops)r/WLEN.

   Because of a loop optimisation, it should be valid to access a[-1].

   RPB, 20000907 */

ULONG *a, *b;
int r;

  {
  REGISTER int j, s1, s2;
  REGISTER ULONG t, u, v, w, next1, next2, old, new;
  REGISTER ULONG c0, c1, c2, c3, c4, c5;
  int q4, alpha;

  c0 = 0x00000000FFFFFFFFL;		/* Some 64-bit constants */
  c1 = 0x0000FFFF0000FFFFL;
  c2 = 0x00FF00FF00FF00FFL;
  c3 = 0x0F0F0F0F0F0F0F0FL;
  c4 = 0x3333333333333333L;
  c5 = 0x2222222222222222L;

  alpha = r >> 1;			/* alpha = (r-1)/2 */
  q4 = (alpha+1) >> WD;			/* q4 = (alpha+1) div WLEN */
  s1 = (alpha+1) & WLENM;		/* s1 = (alpha+1) mod WLEN */
  s2 = WLENM - s1;			/* In [0, WLEN) */
  
  next1 = a[q4];
  old = a[2*q4+1];
  new = a[2*q4];

  for (j = q4; j GE 0; j--) {

    next2 = (new >> s1) | ((old << 1) << s2);	/* Beware case s2 EQ 63 */

    u = next1>>32;			/* High order 32 bits low part of a */
    w = next2>>32;			/* Ditto high part of a */
    t = next1 & c0;			/* Low order 32 bits low part of a */
    v = next2 & c0;			/* Ditto high part of a */

    next1 = a[j-1];
    old   = new;
    new   = a[j+q4-1];

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

  

#else

void interlvf(a, b, r)

/* 32-bit interleave. Loop index runs up.

   If bits 0, 2, 4, ... , r-3, r-1, 1, 3, 5, ..., r-4, r-2 in a,
   moves them to b in correct order.

   Work is about (1.load + 1.store + 26.ops)r/WLEN.

   RPB, 20000906 */

ULONG *a, *b;
int r;

  {
  REGISTER int j, s1, s2, q4;
  REGISTER ULONG t, u, v, w, next1, next2, old, new;
  REGISTER ULONG c1, c2, c3, c4, c5;
  int alpha;
  
  c1 = 0x0000FFFFL;			/* Some 32-bit constants */
  c2 = 0x00FF00FFL;
  c3 = 0x0F0F0F0FL;
  c4 = 0x33333333L;
  c5 = 0x22222222L;
  
  alpha = r >> 1;			/* alpha = (r-1)/2 */
  q4 = (alpha+1) >> WD;			/* q4 = (alpha+1) div WLEN */
  s1 = (alpha+1) & WLENM;		/* s1 = (alpha+1) mod WLEN */
  s2 = WLENM - s1;
 
  next1 = a[0];  
  old = a[q4];
  new = a[q4+1];
  
  for (j = 0; j LE q4; j++) {	

    next2 = (old >> s1) | ((new << 1) << s2); /* Can't combine left shifts
    						 in the case s1 EQ 0 */

    u = next1>>16;			/* High order 16 bits low part of a */
    w = next2>>16;			/* Ditto of high part of a */
    t = next1 & c1;			/* Low order 16 bits low part of a */
    v = next2 & c1;			/* Ditto of high part of a */

    next1 = a[j+1];
    old   = new;
    new   = a[j+q4+2];

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
  
void interlvr(a, b, r)

/* 32-bit interleave. Loop index runs down (compare interlvf).

   If bits 0, 2, 4, ... , r-3, r-1, 1, 3, 5, ..., r-4, r-2 in a,
   moves them to b in correct order.
      
   Work is about (1.load + 1.store + 26.ops)r/WLEN.

   Because of a loop optimisation, it should be valid to access a[-1]. 
   
   RPB, 20000906 */

ULONG *a, *b;
int r;

  {
  REGISTER int j, s1, s2;
  REGISTER ULONG t, u, v, w, next1, next2, old, new;
  REGISTER ULONG c1, c2, c3, c4, c5;
  int q4, alpha;
  
  c1 = 0x0000FFFFL;			/* Some 32-bit constants */
  c2 = 0x00FF00FFL;
  c3 = 0x0F0F0F0FL;
  c4 = 0x33333333L;
  c5 = 0x22222222L;
  
  alpha = r >> 1;			/* alpha = (r-1)/2 */
  q4 = (alpha+1) >> WD;			/* q4 = (alpha+1) div WLEN */
  s1 = (alpha+1) & WLENM;		/* s1 = (alpha+1) mod WLEN */
  s2 = WLENM - s1; 			/* In [0,WLEN) */
 
  next1 = a[q4];  
  old = a[2*q4+1];
  new = a[2*q4];
  
  for (j = q4; j GE 0; j--) {	

    next2 = (new >> s1) | ((old << 1) << s2); /* Can't combine left shifts
	     					 in the case s1 EQ 0 */
     				
    u = next1>>16;			/* High order 16 bits low part of a */
    w = next2>>16;			/* Ditto of high part of a */
    t = next1 & c1;			/* Low order 16 bits low part of a */
    v = next2 & c1;			/* Ditto of high part of a */

    next1 = a[j-1];
    old   = new;
    new   = a[j+q4-1];

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

#endif

FILE *myfopen(fname, flag)

/* Attempts to open file fname. If not successful at first, keeps trying
   for TIMEOUT seconds, then error exits.
*/
   
char *fname, *flag;

  {
  FILE *fp, *fopen();
  clock_t openstart;
  double time;
  time = clockd(&openstart, TRUE);	/* Independent timer here */
  
  for (;;) {
    fp = fopen(fname, flag);
    if (fp NE NULL) break;
    if ((time += clockd(&openstart, FALSE)) GT TIMEOUT) {
      printf("Could not open %s after trying for %d seconds\n", 
	fname, TIMEOUT);
      exit(ERROR);
      }
    }
  return(fp);
  }

BOOLEAN skips(skiplist, s)
struct skip *skiplist;
int s;

/* Returns TRUE if s is in the skip list */

  {
  struct skip *skiprec = skiplist;
  while (skiprec NE NULL) {
    if ((skiprec->low LE s) AND (s LE skiprec->high))
      return(TRUE);
    skiprec = skiprec->next;
    }  
  return(FALSE);
  }

ULONG *fastmem(r, sodd, sizeah, CPUest)

int r, sodd, sizeah;
double *CPUest;

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
  ULONG *a0, *a1, *a;
  ULONG *savea0[FASTTRY+1];

  nits = ((FASTTRY LE 0) OR (r LT SMALLR)) ? 1 : FASTTRY;

  bestkt = 1;		/* Index of best a0 yet */
  CPUworst = 0;
  CPUbest = 1;
  for (kt = nits; kt GT 0; kt--) {
    CPUtime = clockd(&cstart, TRUE);
    a0 = (ULONG *)mymalloc(3*sizeah*(int)sizeof(ULONG));
    savea0[kt] = a0; 	/* Save for later free or return */
    /* Changed 2 to 6 in version 3.15 to avoid out of bounds problem 
       (sizeah also increased by 4 before call to fastmem) */
    a0 += 6;		/* For optimisation in interlvr, may access a[-1] */	
#if IBMPC
    if ((((UINT)a0) & 4) NE 0) a0 += 1;	/* Align if necessary */
#endif
    a1 = a0 + sizeah;
    a = a1;		/* Will cycle through a1, a0 */

    setupx (a);	
    for (nkt = 0; CPUtime LT CPUTEST; nkt++) {
      reducep(a);		/* Timing run for about CPUTEST seconds */
      if (a EQ a1) {
        interlvf(a1, a0, r);		/* Forward interleave */
        a = a0;				/* Adjust pointer to data */
        }
      else {
        interlvr(a0, a1, r);		/* Reverse interleave */
        a = a1;
        }
      if ((nkt & 0x7F) EQ 0)		/* Reduce overhead of clockd calls */
        CPUtime += clockd(&cstart, FALSE);
      }
    CPUtime = CPUtime/(double)nkt;	/* Normalise CPU time */
    if ((nkt EQ 0) OR (CPUtime LT CPUbest)) {
      CPUbest = CPUtime;		/* Save best result yet */
      bestkt = kt;
      }
    if (CPUtime GT CPUworst)
      CPUworst = CPUtime;  		/* and (for comparison) worst too */
    }
  for (kt = nits; kt GT 0; kt--) {	/* Free all but the "best" space */
    if (kt NE bestkt) 	
      free(savea0[kt]);
    }
  CPUtime = ((double)r)*CPUtime;	/* Estimate of overall time */
#if VERBOSE
  if (nits > 1)
    printf("Worst/best ratio in fastmem %1.2f\n", CPUworst/CPUbest);
  printf("Estimated CPU time (not sieving) %2.2f sec = %2.2f r^2 nsec\n\n",
  	CPUtime, 1.0e9*CPUtime/r/r);
  fflush(stdout); 	
#endif
  *CPUest = CPUtime;			/* Return CPU time estimate */
  return savea0[bestkt];		/* and the "best" pointer */
  }

void hex8(n, str)
UINT n;
char *str;

/* Return str as 8-character hex string representing n,
   with leading zeros rather than blanks */

  {
  int j;
  sprintf(str, "%8x", n);
  for (j = 0; j < 8; j++) {
    if (str[j] EQ ' ') str[j] = '0';	/* Overwrite blanks by zeros */
    }  
  str[8] = '\0';			/* End with null */  
  }

BOOLEAN prime(n)
int n;

/* Returns TRUE if n is prime. Simple and not intended to be efficient. */

  {
  long j;
  if (n LE 1) return(FALSE);		/* n < 2 */
  if (n LE 3) return(TRUE);		/* n = 2 or 3 */
  if ((n & 1) EQ 0) return(FALSE);	/* n even, not 2 */
  for (j = 3; j LE (n/j); j += 2) {	/* n > 3 */
    if ((n%j) EQ 0) return(FALSE);
    }
  return(TRUE);
  }

int main(argc, argv)
int argc;
char *argv[];
  {
  FILE *fp;
  double CPUtime1, CPUlast, CPUest;
  double CPUtotal = 0;
  double CPUtotal1 = 0;
  int minutes = 0;		/* minutes before stopping */
  int j, k, rv;
  int s = -1;
  int s1 = 0, s2 = 0;
  int n, nsv, nm, kn, pn, qn, rk, sk, sizeah, sizep;
  int kt1 = 0, kt2 = 0, skt = 0;
  UINT temp;
  ULONG *a;			/* For polynomial of degree (r-1) */
  ULONG *a0, *a1;		/* a = a0 or a1 */
  ULONG *p, *q;			/* For sieving */
  ULONG new;

  int sieved[NMAX+1];		/* Count sieve successes */
  int sievemnz;			/* Index of highest nonzero in sieved[] */
  char line[MC];		/* Line buffer */
  char log[] = " LOG";		/* Identifier for log 
  				   (if no log file specified) */
  char str8[9];			/* 8 characters plus null */
  struct skip *skiplist;	/* The head of the skip list */
  struct skip *skiprec;
  int slow, shigh;
  BOOLEAN done, found, g, swan, sievemore;
  
#if (ASM AND IBMPC)
  int cpumax;
  int cpudata[5];		/* For results of cpuinfo */
#endif

  printf("\nThis is irred version 3.15\n");	  /* Date 20030328 */
  
#if GNU
  printf("\nCopyright (C) 2003 R. P. Brent.\n");
  printf("irred comes with absolutely no warranty. This is free software,\n");
  printf("and you are welcome to redistribute it under the conditions of\n");
  printf("the GNU General Public License, version 2, June 1991.\n");
  printf("See http://www.gnu.org/copyleft/gpl.html for further details.\n\n");
#endif

  printf("Options ");				  /* Print relevant options */
  if (BOOTH) printf("BOOTH, ");
  if (IBMPC) printf("IBMPC, ");
  if (ALPHA) printf("ALPHA, ");
  if (SPARC) printf("SPARC, ");
  if (SGI)   printf("SGI, ");
  if ((IBMPC + ALPHA + SPARC + SGI) NE 1) {
    printf("\nPlease set one machine type to TRUE, ");
    printf("others to FALSE, and recompile\n");
    return(ERROR);
    }
  if (W64)   printf("W64, ");
  if (UNROLL) 
    printf("UNROLL, ");
  else if (ULTRA) 
    printf("ULTRA, ");
  if (ASM AND IBMPC) {
    printf("ASM, ");
    }
  if (MAXUSERS GE 0) 
    printf("MAXUSERS = %d, ", MAXUSERS);
  else if (MAXUSERS EQ -2)
    printf("MAXUSERS = 0, ");  
  if ((MAXUSERS EQ -1) OR (MAXUSERS EQ -2))
    printf("LOADTOL = {%1.2f, %1.2f}, ", LOADTOLS, LOADTOLR);  
  if (FASTTRY GT 0) printf("FASTTRY = %d, ", FASTTRY);
  if (DYNAMIC) 
    printf("DYNAMIC\n");
  else
    printf("NEXTRA = %d\n", NEXTRA);

#if (ASM AND IBMPC)
  cpuinfo (0, cpudata);
  cpumax = cpudata[0];
  cpudata[4] = 0;

  #if CPUINFO
    printf("\nCPU info: %12s\n", (char *)(cpudata+1));
    for (j = 1; j LE cpumax; j++) {
      cpuinfo(j, cpudata);
      printf("%8x %x %8x %x\n",
        cpudata[0], cpudata[1], cpudata[2], cpudata[3]);
      }
  #endif

  cpuinfo(1, cpudata);
  if (cpudata[0] EQ 0x633)
    printf("P-II, possibly booth51-55\n");
  if (cpudata[0] EQ 0x634)
    printf("P-II, possibly portland\n");
  else if (cpudata[0] EQ 0x642)
    printf("P-II or AMD, possibly (old) bsp or lamaxe\n");
  else if (cpudata[0] EQ 0x673)
    printf("P-III, possibly booth1-35\n");
  else if (cpudata[0] EQ 0x683) {
    if (cpudata[2] EQ 0x383f9ff)   printf("P-III, possibly bsp\n");
    else           /* 0x383fbff */ printf("P-III, possibly tosca\n");
    }
  cpuinfo(2, cpudata);
  if ((cpudata[2] & 0xF) EQ 2) printf("256KB L2 cache\n");
  if ((cpudata[2] & 0xF) EQ 3) printf("512KB L2 cache\n");
#endif

  printf("\n");
      
#if W64

  if (sizeof(ULONG) NE 8) {	/* This is a 64-bit version */
    printf("sizeof(ULONG) %d, should be 8\n", (int)sizeof(ULONG));
    printf("recompile with 64-bit options\n");
    return(ERROR);
    }

#else

  if (sizeof(ULONG) NE 4) {	/* This is a 32-bit version */
    printf("sizeof(ULONG) %d, should be 4\n", (int)sizeof(ULONG));
    printf("recompile with 32-bit options\n");
    return(ERROR);
    }
#endif

  CPUtime = clockd(&cstart, TRUE); 		/* Initialise timer */
  slept = 0;
  a0 = NULL;
  skiplist = NULL;
  found = FALSE;
  r = 0;
  line[0] = '\0';

  for (j = NMAX; j GE 0; j--) 
    sieved[j] = 0;
  sievemnz = 0;  

#if (ASM AND IBMPC)
  if (NOT checkmmx()) {
    printf ("Sorry, your PC can not execute MMX instructions\n");
    printf ("Set ASM to FALSE and recompile\n");
    return(ERROR);
    }
#endif   

  if (argc GT 7)			/* Load tolerance while sleeping */
    sscanf(argv[7], "%f", &loadtols);	

  if (argc GT 6) {			/* Load tolerance while running */
    sscanf(argv[6], "%f", &loadtolr);
    if (loadtols LT 0.0) loadtols = 0.0;	   /* Ensure that new values */
    if (loadtolr LT loadtols) loadtolr = loadtols; /* are reasonable */ 
    }
    
  if (argc GT 5) {			/* Max number of interactive users */
    if (sscanf(argv[5], "%d", &maxusers) GT 0) {	
      if (maxusers GE 0)
        printf("Will sleep while more than %d interactive users\n", maxusers);
      if (maxusers EQ -2)
        printf("Will sleep while any interactive users\n");
      if ((maxusers EQ -1) OR (maxusers EQ -2))
        printf("Will sleep if load exceeds %1.2f until down to %1.2f\n", 
          loadtolr, loadtols);
      }      
    }

  if (argc GT 4) {				/* Process skip file */	
    if ((fp = fopen(argv[4], "r")) EQ NULL)
      printf("Warning - can not open skip file\n"); 
    else {					/* Ignore if can't open */
      while (fgets(line, MC, fp) NE NULL) {
        skiprec = (struct skip *)mymalloc((int)sizeof(struct skip));
        slow = shigh = 0;  
        if (sscanf(line, "%d %d", &slow, &shigh) LT 2) break;
        if ((slow LE 0) OR (slow GT shigh)) break; /* GE -> GT 20010307 */
#if FALSE				  	   /* Print skip list */
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

  if (argc GT 3) {   /* Run-time bound if nonzero, ignored if zero */
    if (sscanf(argv[3], "%d", &minutes) LE 0) minutes = 0;
    }
    	  
  if (argc GT 1) {
    fp = myfopen(argv[1], "r");
    fgets(line, MC, fp);   
    fclose(fp);	
    }
  else {
    printf ("r, s1, s2 ?\n"); 	  	  /* Interactive input */
    fgets(line, MC, stdin);
    }
  rv = sscanf(line, "%d %d %d", &r, &s1, &s2);
  if (rv LT 3) s2 = 0;
  if (rv LT 2) s1 = 0;
  if (rv LT 1) r = 0;
  if (NOT prime(r)) {			  /* See comments above */
    printf("\nWarning: r = %d is not prime, so irreducibility test", r);
    printf(" is incomplete\n\n");
    }	
    					  /* Searches up from initial s1
    					     to s2-1 (or down to s2+1) */ 
  if (s1 LT 0) return(OK);
  if (s1 GE r) s1 = r/2;		  /* Reasonable defaults */
  if (s2 LT 0) s2 = 0;
  if (s2 GT r) s2 = (s1 GT r/2) ? r - 64: r/2 + 1;  

  /* Swan's theorem says trinomial is reducible if r = +- 3 (mod 8)
     and s is not 2 or r-2 */
     
  swan = ((r%8) EQ 3) OR ((r%8) EQ 5);
  if (swan) {
    printf("Swan's theorem applies, only need to test s = 2\n");
    s1 = 2;
    s2 = 1;
    }   

  if (s1 LT s2)
    printf("Searching from %d to %d\n", s1, s2-1);
  else if (s1 GT s2)
    printf("Searching from %d down to %d\n", s1, s2+1);
  else
    printf("Empty search interval at %d\n", s1);
  if (minutes NE 0) printf("Time limit %d minutes\n", minutes);
  fflush(stdout);  
  while ((s1 NE s2) AND (NOT found) AND 
    ((minutes EQ 0) OR (CPUtotal < 60*minutes))) {

    done = FALSE;
    for (;;) {		/* Look for next s in range but not in skiplist */
      s = s1; 
      if (s1 EQ s2) {
        done = TRUE;
        break;
        }
      if (s1 LT s2)
        s1++;
      else
        s1--;
      if (NOT skips(skiplist, s)) break;		  
      }   
    if (done) break;  

    sodd = ODD(s) ? s: r-s;		/* whichever of s, r-s is odd */

#if VERBOSE       
    printf ("\nr %d, s %d, r-s %d\n", r, s, r-s);
#endif

    if ((sodd LE 0) OR (r LE sodd) OR 
      (NOT ODD(r)) OR (NOT ODD(sodd)) OR ((r-sodd) LT LIM)) {
        printf ("Illegal parameters\n");
        if (NOT ODD(s))
          printf("For small even s try version 1.35 or earlier\n");
        return(ERROR);
        }            

    CPUtime += clockd(&cstart, FALSE);
    CPUtotal += CPUtime;		/* CPUtotal is for all (r, s) */
    CPUtime = 0;			/* CPUtime is for current (r, s) */
    CPUtime1 = 0;			/* CPUtime1 is for current sieving */
    CPUlast = 0;
    skt++;

    /* Set up variables depending only on r */

    q1  = r >> WD;			/* q1 > 0 */

    sizeah = (q1 >> 1) + 12;		/* r/(2*WLEN) + 4 = half size of
    					   a/b array plus safety margin */
    sizep  = q1 + 4;			/* sizep = r/WLEN + 4
    					   = size of p/q array */
    	
    sizeah += sizeah & 1;		/* Round up to even values */
    sizep  += sizep  & 1;		/* to preserve 8-byte boundaries */					   		

#if IBMPC
    sizeah = (((sizeah+7)>>3)<<3);	/* Make sizeah = 0 (mod 8)
    					   because 8 4-byte words per cache
    					   line on IBM PC (added v. 2.71)
    					   (does this help ?) */
#endif

    /* Sieve to discard polynomials which are easily found to be reducible */

    kn = 1;
    for (nm = n = 2; n LE (NMAX-NEXTRA); n++) { /* Don't need to consider
    						   n = 1 for a trinomial */
      kn = 2*kn + 1;			/* kn = 2^n - 1 */
      if (kn GE r) break;		/* choose nm so 2^nm < r */
      nm = n;
      }

    if (a0 EQ NULL) {	/* Allocate space for a, b, p and q arrays */

      CPUtime += clockd(&cstart, FALSE);
      CPUtotal += CPUtime;
      CPUtime = 0;			/* Don't count in non-sieving time */

      a0 = fastmem(r, sodd, sizeah, &CPUest); /* Find a "good" a0 on the heap 
      					   	 and estimate CPU time */
      					   	 
      CPUtime += clockd(&cstart, FALSE);/* Count call to fastmem in total */
      CPUtotal += CPUtime;		/* but not in sieving time */
      CPUtime = 0;		

      /* The "+ 4"s were added in version 3.15 to avoid valgrind diagnostics
         caused by reading p[-1] or q[-1] */
         
      p  = (ULONG *)mymalloc((sizep + 4)*(int)sizeof(ULONG)) + 4;
      q  = (ULONG *)mymalloc((sizep + 4)*(int)sizeof(ULONG)) + 4;
      	
      a0 += 2;		/* For optimisation in interlvr, may access a[-1] */	
      			/* See comments in fastmem re version 3.15 change */
      			
      /* Following is to align a0, a1 on 8-byte boundary */

#if (IBMPC)
      if ((((UINT)a0) & 4) NE 0) a0 += 1;	/* Align if necessary */
#endif
      a1 = a0 + sizeah;
      a = a1;		/* Will cycle through a1, a0 */
      }

    kn = 1;
    g = TRUE;
    for (n = 2; n LE nm; n++) {	/* n EQ 1 never succeeds for trinomial, */
      nsv = n;			/* for n > 1 prob. of success about 1/(n+1) */
      kn = 2*kn + 1;		/* 2^n - 1 */
      for (j = kn >> WD; j GE 0; j--)
        p[j] = 0;
      p[0] = 1;
      p[kn >> WD] ^= 1L << (kn & WLENM);	/* p = x^kn + 1 */
      pn = kn;
      rk = r%kn;
      sk = sodd%kn;
      /* Following is intended to speed up relprime */
      if (rk < sk){			/* ensure that rk = max (rk, sk) */
        qn = rk; rk = sk; sk = qn;
        }
      if (rk < 2*sk) sk = rk - sk;	/* so now 0 LE 2*sk LE rk */
      qn = rk;	  			/* degree of q (even if rk = sk = 0) */
      for (j = qn >> WD; j GE 0; j--)
        q[j] = 0;
      q[0] = 1;
      q[rk >> WD] ^= 1L << (rk & WLENM);
      q[sk >> WD] ^= 1L << (sk & WLENM); /* q = x^rk + x^sk + 1 */
      g = relprime(p, pn, q, qn);
      if (NOT g) break;
      CPUtime += clockd(&cstart, FALSE);
      }

    if (NOT g) {
      kt2++;
#if VERBOSE      
      printf("Reducible by test with n %d\n", nsv);
      fflush(stdout);
#endif      
      sieved[nsv]++;
      if (sievemnz LT nsv) sievemnz = nsv;
      if (argc GT 2) {
        fp = myfopen(argv[2], "a");
        fprintf(fp, "%d %d %d\n", r, s, nsv);
        fclose(fp);
        }
      else {
        printf("%d %d %d%s\n", r, s, nsv, log);
        fflush(stdout);
        }
      }

    CPUtime += clockd(&cstart, FALSE);
    sievemore = TRUE;

    if (g) { /* Phase 1 sieve did not decide reducibility */

      setupx (a);
      for (k = 0; g AND (k < r); k++) {

      reducep(a);			/* Reduce (square of) a */

      /* Interleave in forward/reverse directions alternately */

      if (a EQ a1) {
        interlvf(a1, a0, r);		/* Forward interleave */
        a = a0;				/* Adjust pointer to data */
        }
      else {
        interlvr(a0, a1, r);		/* Reverse interleave */
        a = a1;
        }

      if ((k & 0x7FFFL) EQ 0) { 	 	/* Avoid timer overflow by */
        CPUtime += clockd(&cstart, FALSE);	/* calling clockd sometimes */
        }
        
      if (sievemore OR ((k & 2047L) EQ 0)) { /* Check every few seconds  */
	if (maxusers EQ -2) 		     /* Check load and users here */
	  slept += checkload2();
	else if (maxusers EQ -1)	     /* Check load here */
	  slept += checkload();		     /* Sleep while load high */
	else if (maxusers GE 0)		     /* Check users here */
	  slept += checkusers(maxusers);     /* Sleep while users high */
        }
        
      /* Phase 2 sieve: compute GCD(a(x)+x, trinomial) for small k */

        if (sievemore AND (k GE nm) AND (k LT (nm+NEXTRA)) AND (k LT (r-1))) {
	  for (j = q1; j GE 0; j--)
            p[j] = a[j];		/* Copy as relprime destroys p, q */
	  p[0] ^= 2L;			/* Add x (mod 2) */
	  pn = r;			/* Upper bound on degree */
	  for (j = pn; j GE 0; j--) {	/* Find exact degree */
	    new = p[j >> WD];
      	    if (new EQ 0L) {		/* Speed up search over zero words */
	      pn -= j & WLENM;
 	      j-= j & WLENM;		
	      }
	    else
	      if ((new >> (j & WLENM)) &1L) break;
      	    pn--;
	    }
	  qn = r;
	  for (j = qn >> WD; j GE 0; j--) q[j] = 0;
	  q[0] = 1;
	  q[r >> WD] ^= 1L << (r & WLENM);
	  q[sodd >> WD] ^= 1L << (sodd & WLENM); /* q = x^r + x^sodd + 1 */

	  g = relprime(p, pn, q, qn);

	  if (k LT (nm + NEXTRA)) {
	    CPUtime += clockd(&cstart, FALSE);
	    CPUlast = CPUtime;
	    CPUtime1 += CPUtime;	/* CPUtime1 is for sieving */
	    CPUtotal += CPUtime;
	    CPUtime = 0;		/* Non-sieving from now on */
#if DYNAMIC
	 /* Compute sieve cutoff based on probability of success for next
	    higher degree and estimated time for complete test.
	    Current degree is k+1, next degree is k+2 = d say,
	    and probability of success sieving with degree d
	    is approximately 1/(d+1) = 1/(k+3). We assume that
	    sieving with degree d takes about the same time as
	    the total sieving time for degrees less than d
	    (i.e. assume sieving time double when degree increments). */

	    sievemore = (((double)(k+3)*CPUtime1) < CPUest);
#endif
	    }
	    
	  if (g) {
	    if ((NOT sievemore) OR (k EQ (nm + (NEXTRA - 1)))) { 		
	      kt1++;			/* Unsuccessful sieve count */
	      sieved[0]++;
#if VERBOSE
	      if (CPUtime1 GT CPUTOL) {
		printf("Sieving to degree %d unsuccessful after %.2f sec", 
		  k+1, CPUtime1); 
		if (CPUlast GT CPUTOL) 
		  printf(", degree %d took %.2f sec", k+1, CPUlast);
		printf("\n");
		}
	      fflush(stdout);
#endif	      
	      }
            }
          else {
	    kt2++;
#if VERBOSE      
	    printf("Reducible by test with n %d\n", k+1);
#endif      
	    sieved[k+1]++;
	    if (sievemnz LE k) sievemnz = k+1;
	    if (argc GT 2) {
	      fp = myfopen(argv[2], "a");
	      fprintf(fp, "%d %d %d\n", r, s, k+1);
	      fclose(fp);
	      }
	    else {
	      printf("%d %d %d%s\n", r, s, k+1, log); /* bug in 1.31 fixed */
	      fflush(stdout);
	      }
	    }
	  }
	if (NOT g) break;  
        }

      CPUtime += clockd(&cstart, FALSE);

      if (g) {       
       if (comparex(a)) {
        found = (NOT CONTINUE);
        printf("%d %d irreducible (primitive if 2^%d-1 is prime)\n", r, s, r);
        fflush(stdout);
        if (argc GT 2) {
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
#if W64
	temp = (UINT)((a[0]<<32)>>32);
#else      
        temp = (UINT)a[0];
#endif
        hex8(temp, str8);
#if VERBOSE
        printf("Not irreducible/primitive, low word %s (hex)\n", str8); 
#endif
        if (argc GT 2) {
          fp = myfopen(argv[2], "a");
          if (s EQ sodd)
   	    fprintf(fp, "%d %d x%s\n", r, s, str8);
 	  else
   	    fprintf(fp, "%d %d y%s\n", r, s, str8);
 	  fclose(fp);
          }
        else {
          if (s EQ sodd)
   	    printf("%d %d x%s%s\n", r, s, str8, log);
	  else
   	    printf("%d %d y%s%s\n", r, s, str8, log);
 	  fflush(stdout);
 	  }
        }
      }
      
#if VERBOSE        
      if (CPUtime GT CPUTOL) {  
        printf("CPU time (not sieving) %.2f sec = ", CPUtime);
        printf("%2.2f r^2 nsec\n", 1.0e9*CPUtime/r/r);
        }
      if (CPUtime1 GT CPUTOL) {
	printf("CPU time (sieving)     %.2f sec", CPUtime1);
	if (CPUtime GT CPUtime1)
          printf(", ratio %2.0f", CPUtime/CPUtime1);
        printf("\n");   
        }
#endif
      CPUtotal1 += CPUtime1;  
      CPUtotal += CPUtime;
      CPUtime = 0;
#if VERBOSE
      if (CPUtotal GT CPUTOL) {
        printf("\nOverall time %.2f sec = ", CPUtotal);
        printf("%.4f sec per trinomial\n", CPUtotal/skt);
        if ((CPUtotal1 GT CPUTOL) AND (CPUtotal1 LT CPUtotal))
          printf("Overall %1.2f percent of time spent sieving\n", 
        	100.0*CPUtotal1/CPUtotal); 
        if (slept GT CPUTOL)
          printf("Slept for a total of %d seconds, %1.2f percent of time\n",
            slept, (double)100*(double)slept/((double)slept+CPUtotal));
	printf("Sieving: success %d, failure %d, ", kt2, kt1);
	printf("rate %1.2f percent\n", 100.0*(double)kt1/(kt1+kt2));
	for (j = 1; j LE sievemnz; j++) printf ("%d ", sieved[j]);
	printf("\n");
        }
      fflush(stdout);
#endif        
      }
  
    /* Try again until irreducible/primitive trinomial found
       (or until all s checked if CONTINUE is true) */

    if (argc GT 1) {
      fp = myfopen(argv[1], "w");
      fprintf(fp, "%d %d %d\n", r, s1, s2);
      fclose(fp);
      }  

    if (swan) break;
    }

  if (argc GT 1) {
    fp = myfopen(argv[1], "w");
    fprintf(fp, "%d %d %d\n", r, s1, s2);
    fclose(fp);
    }

  if ((NOT found) AND (p NE NULL) AND (s NE -1))
    printf("Searched to %d\n", s);
#if VERBOSE
  printf("Working set after sieving about %d bytes\n", (3*r)>>4);
  printf("Dynamic storage allocation %d bytes\n", space);
  printf("%d clock calls\n", clockcalls);
  if (syscalls GT 0) printf("%d system/sleep calls\n", syscalls);
  if (userkt0 GT 0) printf("Mean number of users %1.2f\n",
    (double)userkt1/(double)userkt0);
#endif
  return(OK);				/* Normal exit */
  }

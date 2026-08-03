## Overview ##

This repository contains programs (mostly written by AI) that find (smallest) polynomial factors for trinomials over GF(2).

These programs use CUDA to run on GPUs with greater throughput than the longstanding CPU-based **factor** program that has been used so far for this purpose.

(Note: **factor** should not be confused with the GNU coreutils utility of the same name)

The programs in this repository, as well as the **factor** program itself, make use of the NTL and GF2X libraries (which in turn rely on the GMP library) except as noted otherwise.

## Background ##

In GF(2) the coefficients of all terms of all polynomials are either 0 or 1.

If a polynomial has no factors, it is *irreducible*.
If *r* is the exponent of a Mersenne prime,
then irreducible trinomials x<sup>r</sup>&nbsp;+ x<sup>s</sup>&nbsp;+&nbsp;1 over GF(2) are in fact primitive.
See Richard Brent's [Search for Primitive Trinomials](https://maths-people.anu.edu.au/~brent/trinom.html) page.

As of this writing, the largest known Mersenne prime is 2<sup>136279841</sup>&nbsp;−&nbsp;1

We focus on *r*&nbsp;=&nbsp;136279841 because trinomials corresponding to all smaller Mersenne prime exponents have already been fully investigated
by Brent et al. for the entire range of *s* values. See for example:
[arXiv:1605.09213 [math.NT]](https://arxiv.org/abs/1605.09213)

## Theory and empirical observations ##

For any given&nbsp;*r*, Brent et al. found between zero to five values of&nbsp;*s* that produce primitive trinomials.

By symmetry between x<sup>r</sup>&nbsp;+ x<sup>s</sup>&nbsp;+&nbsp;1 and x<sup>r</sup>&nbsp;+ x<sup>r−s</sup>&nbsp;+&nbsp;1,
we (and they) only consider the range 1&nbsp;&le; *s*&nbsp;&le; floor(*r*/2)&nbsp;=&nbsp;68139920

By Swan's theorem, if *r*&nbsp;≠&nbsp;±1&nbsp;(mod&nbsp;8), then the only value of *s* that could produce a primitive trinomial is *s*&nbsp;=&nbsp;2,
and the search is trivial.

By Swan's theorem, polynomial factors can only have degree 2&nbsp;&le;&nbsp;*d*&nbsp;&le;&nbsp;floor(*r*/3)&nbsp;=&nbsp;45426613

In practice, pathologically large factors are extremely rare, and factors of small degree are very common.

When *r* is a Mersenne prime exponent smaller than 136279841, and only considering the polynomial factor of smallest degree for each trinomial, we empirically observe the following histogram probabilities for small values of&nbsp;*d* as tallied over the full range of *s* for non-irreducible trinomials:
* p(2) = 1/3 = 3255/9765
* p(3) = 4/21 = 1860/9765
* p(4) = 2/21 = 930/9765
* p(5) = 16/217 = 720/9765
* p(6) = 400/9765

Interestingly, 9765 = 3 × 7 × 15 × 31, the product of 2<sup>p</sup>&minus;1 for p&nbsp;=&nbsp;2,&nbsp;3,&nbsp;4,&nbsp;5

This is not strictly monotonic, since we also find that p(9)&nbsp;>&nbsp;p(8)

For larger *d*, we empirically observe that the histogram probability is asymptotically fit by the curve p(*d*)&nbsp;=&nbsp;*C*&nbsp;/&nbsp;*d*<sup>2</sup>,
where the constant *C* is close to 16/9.

## Certificates ##

By convention, programs that search for factors over GF(2) store their results in a certificate file, where almost every line is of the format:<br>
s d `p`&lt;bitmask&gt;<br>
where &lt;bitmask&gt; is a string of lowercase hexadecimal digits that represent the coefficients of the smallest polynomial over GF(2) that factors x<sup>r</sup>&nbsp;+ x<sup>s</sup>&nbsp;+&nbsp;1, and *d* is the degree of that polynomial. Therefore the hexadecimal number is always of size d+1 bits.

For example, for *r*&nbsp;=&nbsp;136279841, the line<br>
`2 5 p29`<br>
means that x<sup>136279841</sup>&nbsp;+ x<sup>2</sup>&nbsp;+&nbsp;1 over GF(2) has a degree-5 factor
whose coefficients are given by hexadecimal 29 or the binary string 101001 of length 5+1,
namely x<sup>5</sup>&nbsp;+&nbsp;x<sup>3</sup>&nbsp;+&nbsp;1

A very small number of lines will instead have the format:<br>
s `primitive`<br>
when there are no factors, or:<br>
s `u`<br>
when a search was terminated prior to finding a factor, but it has not been confirmed that there are no factors.

Brent's website contains [links to certificate files](https://maths-people.anu.edu.au/~brent/trinomlg.html)
for all Mersenne exponents *r*&nbsp;≡&nbsp;±1&nbsp;(mod&nbsp;8) less than 136279841

For each *s*, only the polynomial factor of smallest degree is recorded. If there are two or more of the same degree, then tiebreaks are settled by lexicographically comparing the hexadecimal bitmask strings.

These certificates can be verified by the C program **check-ntl** that is linked at Brent's website, proving that the factors found are valid.
This program uses the NTL and GF2X libraries to verify all the millions of lines of a certificate file very quickly.

This repository also contains **verify_gf2_trinomial.py**, a Python script that uses only GMP via the gmpy2 package, but does not use the NTL or GF2X libraries.
It does an independent double-check that the factors in a certificate file are indeed valid, using slow but "obviously correct" algorithms.
It generally takes a couple of days to do what **check-ntl** does in a couple of minutes.

## Programs ##

The longstanding **factor** program in [the apps subdirectory of the GF2X library repository](https://gitlab.inria.fr/gf2x/gf2x/-/tree/master/apps?ref_type=heads)
(not to be confused with the GNU coreutils utility of the same name) has been used for this purpose, but it is a CPU-based program.
These newer AI-written programs use CUDA to run on GPUs with greater throughput.

These programs are used as follows:
* **coarse_sieve_wide** finds all factors of degree M or less globally across the entire range of s in a single run.
It is run iteratively, with the survivors of a degree-M run being used as input to the degree-M+1 run.
This only needs to be done once by one well-equipped person, since the feasibility of large M depends on the amount of GPU memory available.
At the moment, degree 37 has been achieved: the incremental run from degree 36 to 37 took about 38 hours on a T4 with 16 GB.
* **factor** (the previously existing CPU-based program) can be used with `-skip M` and `-maxd N` command-line settings
to perform a single cycle of squares/products and GCD.
Note that **factor** enters a special mode if the `-s0` parameter is greater than the `-s1` parameter:
it ignores them both and simply reads s values from standard input instead.
* **tsfactor** is then used to find the smallest factor for any value of *s* that survived the preliminary **coarse_sieve_wide** and **factor** filters.

# VERY preliminary and minimal instructions:

These programs were written by Claude Fable and Claude Opus, except as noted. Human authorship of AI code is claimed only for purposes of GPL licensing.

## global coarse sieve

`make coarse_sieve coarse_sieve_wide`

* This program needs a GPU
* The difference between coarse_sieve and coarse_sieve_wide is internal word sizes being 32-bit or 64-bit
* (In practice it needs to go to coarse_sieve_wide relatively soon)

`./coarse_sieve 136279841 29 out29`<br>
`./coarse_sieve_wide 136279841 30 out30 --min-depth 30 --load-survivors out29.survivors.txt`<br>
`./coarse_sieve_wide 136279841 31 out31 --min-depth 31 --load-survivors out30.survivors.txt`

and so on iteratively until the runtime becomes prohibitive on the current GPU (works better if there is a lot of GPU memory)

Each iteration creates outNN.survivors.txt and outNN.found.cert, and reads the survivors file created by the previous iteration.

If multiple GPUs are present, it should use them all (but this is not yet tested).

**Note:** These files only need to be generated once by one very well-equipped person to as high a degree as possible,
and then shared with everyone else.
I myself have reached NN=37 in 40 hours on an NVIDIA T4. It should be possible to do considerably better with more GPU memory.

## processing survivors of the sieve

`make tsfactor`

* This program needs a GPU
* (might not run well or at all in the cloud because of heavy data traffic between CPU and GPU)
* (for example: zero throughput on a T4 in the cloud)
* (works OK on a home PC with graphics card using a standard PCIe connection)
* (might work in the cloud if the GPU has a high-bandwidth NVLink connection, but this has not been tested)

`./tsfactor 136279841 out31.survivors.txt --skip 31 --f 1 --out MYDATA --ckpt-mins 20 --gcd-threads 1 --pend-max 1 -v -v -v`

The above example assumes that the highest available outNN.survivors.txt file was done up to degree 31. In practice it will be a lot higher.

* It writes certificate data to a file called MYDATA.results.txt, and writes verbose logging data to the terminal (stdout).
* It writes checkpoint files every 20 minutes. These are very large. They preserve data and allow resumption after interruption.
* If there is a high-bandwidth (NVLink) connection, it may make sense to increase the --gcd-threads parameter. But it doesn't help on a home PC.
* If --pend-max is more than 1 (and the default is 4!), it will speculatively do squares/products steps ahead of time, but...
this will most often just wastefully burn a lot of energy and overuse the GPU to create data that just gets thrown away.

## validating polynomial factors

`make -f Makefile.extra check-ntl`

This is just a copy of Richard Brent's check-ntl program from https://maths-people.anu.edu.au/~brent/trinom.html provided for convenience

`cat out31.found.cert | ./check-ntl 136279841`<br>
`cat MYDATA.results.txt | ./check-ntl 136279841`

(this verifies that the polynomial factors generated are valid)

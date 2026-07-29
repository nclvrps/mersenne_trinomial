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

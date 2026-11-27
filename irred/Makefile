# irred devkit Makefile.  Everything builds into build/.
#
#   make            irred (native) + the test programs
#   make test       all the tests below (a few minutes)
#   make selftest   irred --selftest for every build
#   make cmp        compare every build with the naive reference (many r, s)
#   make sweep      exhaustive fault-injection sweep
#   make ckpt       checkpoint / resume / signal / --stop-after tests, and
#                   0.20's checkpoints refused (never overwritten)
#   make bug014     re-insert 0.14's bug: the chain misses it, the map check not
#   make bench      timing, with and without error checking and threads
#
# Needs gcc (or CC=clang), GNU make, bash, python3 (only for two tests).

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp
WARN    = -Wall -Wextra
B       = build

all: $(B)/irred $(B)/irred_generic $(B)/irred_asan $(B)/ref $(B)/irred315 $(B)/irred020

$(B):
	mkdir -p $(B)

# production build
$(B)/irred: irred/irred.c | $(B)
	$(CC) $(CFLAGS) $(WARN) -o $@ $< -lm

# no -march: the run-time dispatch must still pick AVX2/AVX-512 code
$(B)/irred_generic: irred/irred.c | $(B)
	$(CC) -O3 -fopenmp $(WARN) -o $@ $< -lm

# address + undefined-behaviour sanitizers
$(B)/irred_asan: irred/irred.c | $(B)
	$(CC) -O1 -g -fopenmp -fsanitize=address,undefined -fno-sanitize-recover=all \
	    -fno-omit-frame-pointer -o $@ $< -lm

# naive O(r^2) bit-at-a-time reference (prints the 64-bit residue)
$(B)/ref: tools/ref.c | $(B)
	$(CC) -O2 -o $@ $<

# Brent's irred 3.15 with the 2026 compile patch, sieving disabled so that it
# prints a (32-bit) residue for every s.  Input on stdin: "r s s+1".
$(B)/irred315: reference/irred315_orig.c reference/irred315_compile.patch | $(B)
	cp reference/irred315_orig.c $(B)/irred.c
	cd $(B) && patch -s -p0 < ../reference/irred315_compile.patch
	sed -i 's/for (n = 2; n LE nm; n++) {\t\/\* n EQ 1/for (n = 2; n LE 1; n++) {\t\/* n EQ 1/' $(B)/irred.c
	grep -q 'for (n = 2; n LE 1; n++)' $(B)/irred.c
	mv $(B)/irred.c $(B)/irred315.c
	$(CC) -O1 -w -o $@ $(B)/irred315.c

# irred 0.20 (writes version 1 checkpoints, which 0.22 must refuse)
$(B)/irred020: reference/irred_0.20.c | $(B)
	$(CC) -O3 -march=native -w -o $@ $< -lm

selftest: $(B)/irred $(B)/irred_generic $(B)/irred_asan
	$(B)/irred --selftest | tail -1
	$(B)/irred --selftest --isa avx2 | tail -1
	$(B)/irred --selftest --isa generic | tail -1
	$(B)/irred --selftest --nt | tail -1
	$(B)/irred_generic --selftest | tail -1
	$(B)/irred_asan --selftest | tail -1
	$(B)/irred --selftest --threads 2 | tail -1
	$(B)/irred_asan --selftest --threads 3 | tail -1

cmp: all
	tests/cmp.sh $(B)

sweep: $(B)/irred
	tests/ec_sweep_all.sh $(B)/irred

ckpt: $(B)/irred $(B)/irred020
	tests/ckpt_test.sh $(B)/irred $(B)/irred020

bench: $(B)/irred
	tests/bench.sh $(B)/irred

bug014: $(B)/ref
	tests/bug014_test.sh

test: selftest cmp sweep ckpt bug014

clean:
	rm -rf $(B)

.PHONY: all selftest cmp sweep ckpt bench bug014 test clean

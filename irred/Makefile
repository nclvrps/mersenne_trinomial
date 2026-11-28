# irred Makefile.  Everything builds into build/.
#
#   make            irred (native) + the test programs, and irred_gpu when
#                   nvcc is found (make gpu NVCC=/path/to/nvcc otherwise)
#   make gpu        irred_gpu: the same program with the GPU backend
#   make emul       irred_gpu_emul: irred_gpu under the CUDA emulation
#                   (kernels run as loops on the CPU; no GPU needed)
#   make gputest    tests/gpu_test.sh on irred_gpu (GPU_BIN=... to choose)
#   make emultest   tests/gpu_test.sh on irred_gpu_emul
#   make test       all the CPU tests below (a few minutes)
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
CXX     ?= g++
CFLAGS  ?= -O3 -march=native -fopenmp
WARN    = -Wall -Wextra
B       = build
NVCC    ?= nvcc
NVFLAGS ?= -O3 -arch=native
# cuda_emul.h (the CUDA emulation) lives with tsfactor
EMUL_H  ?= ../tsfactor/tsfactor/cuda_emul.h
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)

CPU_ALL = $(B)/irred $(B)/irred_generic $(B)/irred_asan $(B)/ref $(B)/irred315 $(B)/irred020

all: $(CPU_ALL) $(if $(HAVE_NVCC),gpu,nocuda)

nocuda:
	@echo "== $(NVCC) not found: built the CPU programs only (make gpu NVCC=/path/to/nvcc for irred_gpu) =="

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

# GPU build: irred.c (the driver, with -DIRRED_GPU) + irred_gpu.cu (backend)
gpu: $(B)/irred_gpu

$(B)/irred_gpu.o: irred/irred.c | $(B)
	$(CC) $(CFLAGS) $(WARN) -DIRRED_GPU -c -o $@ $<

$(B)/irred_gpu: $(B)/irred_gpu.o irred/irred_gpu.cu irred/gf2_trin_sqr.cuh
	$(NVCC) $(NVFLAGS) -Iirred -o $@ irred/irred_gpu.cu $(B)/irred_gpu.o \
	    -lm -Xcompiler -fopenmp

# the same under the CUDA emulation (g++ only)
emul: $(B)/irred_gpu_emul

$(B)/irred_gpu_emul.o: irred/irred_gpu.cu irred/gf2_trin_sqr.cuh $(EMUL_H) | $(B)
	$(CXX) -O2 -march=native $(WARN) -x c++ -DIRRED_EMUL -Iirred \
	    -I$(dir $(EMUL_H)) -c -o $@ irred/irred_gpu.cu

$(B)/irred_gpu_emul: $(B)/irred_gpu.o $(B)/irred_gpu_emul.o
	$(CXX) -fopenmp -o $@ $^ -lm

gputest: $(B)/irred $(B)/ref
	tests/gpu_test.sh $(or $(GPU_BIN),$(B)/irred_gpu) $(B)

emultest: $(B)/irred_gpu_emul $(B)/irred $(B)/ref
	tests/gpu_test.sh $(B)/irred_gpu_emul $(B)

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

.PHONY: all gpu emul nocuda gputest emultest selftest cmp sweep ckpt bench bug014 test clean

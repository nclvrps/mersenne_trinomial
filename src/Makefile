# GF(2) trinomial factor search: CPU references + CUDA implementations
# CPU tools need only g++; CUDA tools need nvcc (CUDA 11+, sm_60+).
# Works both from the source tree (sieve/ cantor/ stage2/ common/) and
# from a flat directory containing all files.

CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O2 -march=native -Wall -Wextra
NVFLAGS  ?= -O3 -arch=native
INC       = -I. -Icommon

vpath %.cpp . sieve cantor
vpath %.cu  . sieve cantor stage2
vpath %.h   . common

CPU  = sieve_ref cantor_ref
CUDA = coarse_sieve cantor_cuda trinomial_stage2

all: cpu
cpu: $(CPU)
cuda: $(CUDA)

sieve_ref: sieve_ref.cpp gf2_small_field.h gf2_field_setup.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

cantor_ref: cantor_ref.cpp gf2_cantor_core.h gf2_cantor_engine.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

coarse_sieve: coarse_sieve.cu gf2_small_field.h gf2_field_setup.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

cantor_cuda: cantor_cuda.cu gf2_cantor_cuda.h gf2_cantor_core.h gf2_cantor_engine.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

trinomial_stage2: trinomial_stage2.cu gf2_cantor_cuda.h gf2_small_field.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

# Functional single-threaded CPU emulation of the CUDA programs (no GPU
# needed): kernel launches become sequential loops via cuda_emul.h.
# Useful for validating changes before spending GPU time.
SEDLAUNCH = sed -E 's/([A-Za-z_][A-Za-z0-9_]*)<<<([^>]*)>>>/EMUL_LAUNCH(\1, \2)/g; s/__device__ __forceinline__/static inline/'

emul-%:
	mkdir -p build_emul
	for h in gf2_small_field.h gf2_field_setup.h gf2_cantor_core.h gf2_cantor_engine.h cuda_emul.h; do \
	  cp $$(ls common/$$h 2>/dev/null || ls $$h) build_emul/; done
	$(SEDLAUNCH) $$(ls common/gf2_cantor_cuda.h 2>/dev/null || ls gf2_cantor_cuda.h) > build_emul/gf2_cantor_cuda.h
	printf '#pragma once\n#include "cuda_emul.h"\n' > build_emul/cuda_runtime.h
	$(SEDLAUNCH) $$(find . sieve cantor stage2 -maxdepth 1 -name '$*.cu' 2>/dev/null | head -1) > build_emul/$*.cpp
	$(CXX) $(CXXFLAGS) -x c++ -Ibuild_emul -o build_emul/$* build_emul/$*.cpp

emultest: emul-coarse_sieve emul-cantor_cuda emul-trinomial_stage2
	./build_emul/cantor_cuda selftest
	./build_emul/coarse_sieve --selftest
	./build_emul/trinomial_stage2 selftest

clean:
	rm -rf $(CPU) $(CUDA) build_emul
.PHONY: all cpu cuda clean emultest

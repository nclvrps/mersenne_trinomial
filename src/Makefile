# GF(2) trinomial factor search: CPU references + CUDA implementations
# CPU tools need only g++; CUDA tools need nvcc (CUDA 11+, sm_60+).
# Works both from the source tree (sieve/ cantor/ stage2/ common/) and
# from a flat directory containing all files.

CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O2 -march=native -Wall -Wextra
# -Xcompiler -march=native: newer GCC (13+) hard-errors on the PCLMUL/
# SSE4.1 always_inline intrinsics in gf2_cantor_core.h unless host ISA
# flags are enabled (alternative: -Xcompiler -mpclmul,-msse4.1).
# Without them the code now compiles anyway via the slower portable
# carry-less-multiply fallback.
NVFLAGS  ?= -O3 -arch=native -Xcompiler -march=native
INC       = -I. -Icommon

vpath %.cpp . sieve cantor
vpath %.cu  . sieve cantor stage2 scan
vpath %.h   . common

# NTL provides the subquadratic GCD backend (effectively REQUIRED for
# production r; build NTL with NTL_GF2X_LIB=on so multiplication uses
# gf2x, exactly as for factor.cpp).  Detection TRY-LINKS ntl_check.cpp
# against plausible library sets, which handles /usr/local installs and
# NTL builds that need -lgf2x pulled in explicitly.  Overrides:
#   make NTL_OK=0                              (force naive backend)
#   make NTLLIB="-L/opt/ntl/lib -lntl -lgf2x -lgmp"   (custom paths)
NTLCHK := $(firstword $(wildcard common/ntl_check.cpp ntl_check.cpp))
ifeq ($(origin NTLLIB),undefined)
NTLLIB := $(shell for L in "-lntl -lgmp" "-lntl -lgf2x -lgmp" \
  "-lntl -lgf2x -lgmp -lpthread"; do \
  if $(CXX) $(NTLCHK) $$L -o /tmp/ntl_chk_$$$$ >/dev/null 2>&1; \
  then echo $$L; break; fi; done; rm -f /tmp/ntl_chk_$$$$)
endif
NTL_OK ?= $(if $(NTLLIB),1,0)
ifeq ($(NTL_OK),1)
NTLDEF = -DHAVE_NTL
NTLSRC = $(firstword $(wildcard common/gf2_gcd_ntl.cpp gf2_gcd_ntl.cpp))
NTLSRC_EMUL = build_emul/gf2_gcd_ntl.cpp
endif

CPU  = sieve_ref cantor_ref
CUDA = coarse_sieve coarse_sieve_wide cantor_cuda trinomial_stage2 trinomial_scan

all: cpu cuda
cpu: $(CPU)
cuda: $(CUDA)

sieve_ref: sieve_ref.cpp gf2_small_field.h gf2_field_setup.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

cantor_ref: cantor_ref.cpp gf2_cantor_core.h gf2_cantor_engine.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

coarse_sieve: coarse_sieve.cu gf2_small_field.h gf2_field_setup.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

coarse_sieve_wide: coarse_sieve.cu gf2_wide_field.h gf2_wide_setup.h
	$(NVCC) $(NVFLAGS) -DWIDE_FIELD $(INC) -o $@ $<

cantor_cuda: cantor_cuda.cu gf2_cantor_cuda.h gf2_cantor_core.h gf2_cantor_engine.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

trinomial_stage2: trinomial_stage2.cu gf2_trinomial_cuda.h gf2_cantor_cuda.h gf2_small_field.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

trinomial_scan: trinomial_scan.cu gf2_trinomial_cuda.h gf2_gcd.h
	$(NVCC) $(NVFLAGS) $(INC) $(NTLDEF) -o $@ $< $(NTLSRC) $(NTLLIB) -Xcompiler -pthread

check_ref: check_ref.cpp
	g++ -O2 -std=c++17 -o $@ $<

# Functional single-threaded CPU emulation of the CUDA programs (no GPU
# needed): kernel launches become sequential loops via cuda_emul.h.
# Useful for validating changes before spending GPU time.
SEDLAUNCH = sed -E 's/([A-Za-z_][A-Za-z0-9_]*)<<<([^>]*)>>>/EMUL_LAUNCH(\1, \2)/g; s/__device__ __forceinline__/static inline/'

emul-%:
	mkdir -p build_emul
	for h in gf2_small_field.h gf2_field_setup.h gf2_cantor_core.h gf2_cantor_engine.h cuda_emul.h; do \
	  cp $$(ls common/$$h 2>/dev/null || ls $$h) build_emul/; done
	$(SEDLAUNCH) $$(ls common/gf2_cantor_cuda.h 2>/dev/null || ls gf2_cantor_cuda.h) > build_emul/gf2_cantor_cuda.h
	$(SEDLAUNCH) $$(ls common/gf2_trinomial_cuda.h 2>/dev/null || ls gf2_trinomial_cuda.h) > build_emul/gf2_trinomial_cuda.h
	printf '#pragma once\n#include "cuda_emul.h"\n' > build_emul/cuda_runtime.h
	$(SEDLAUNCH) $$(find . sieve cantor stage2 -maxdepth 1 -name '$*.cu' 2>/dev/null | head -1) > build_emul/$*.cpp
	$(CXX) $(CXXFLAGS) -x c++ -Ibuild_emul -o build_emul/$* build_emul/$*.cpp

emul-trinomial_scan:
	mkdir -p build_emul
	for h in gf2_small_field.h gf2_field_setup.h gf2_cantor_core.h gf2_cantor_engine.h cuda_emul.h gf2_gcd.h; do \
	  cp $$(ls common/$$h 2>/dev/null || ls $$h) build_emul/; done
	$(SEDLAUNCH) $$(ls common/gf2_cantor_cuda.h 2>/dev/null || ls gf2_cantor_cuda.h) > build_emul/gf2_cantor_cuda.h
	$(SEDLAUNCH) $$(ls common/gf2_trinomial_cuda.h 2>/dev/null || ls gf2_trinomial_cuda.h) > build_emul/gf2_trinomial_cuda.h
	printf '#pragma once\n#include "cuda_emul.h"\n' > build_emul/cuda_runtime.h
	$(SEDLAUNCH) $$(find . sieve cantor stage2 scan -maxdepth 1 -name 'trinomial_scan.cu' 2>/dev/null | head -1) > build_emul/trinomial_scan.cpp
	@if [ "$(NTL_OK)" = "1" ]; then cp $$(ls common/gf2_gcd_ntl.cpp 2>/dev/null || ls gf2_gcd_ntl.cpp) build_emul/; fi
	$(CXX) $(CXXFLAGS) -x c++ -Ibuild_emul $(NTLDEF) -pthread -o build_emul/trinomial_scan build_emul/trinomial_scan.cpp $(NTLSRC_EMUL) $(NTLLIB)

emultest: emul-coarse_sieve emul-cantor_cuda emul-trinomial_stage2 emul-trinomial_scan
	./build_emul/cantor_cuda selftest
	./build_emul/coarse_sieve --selftest
	./build_emul/trinomial_stage2 selftest
	./build_emul/trinomial_scan --selftest

clean:
	rm -rf $(CPU) $(CUDA) build_emul
.PHONY: all cpu cuda clean emultest

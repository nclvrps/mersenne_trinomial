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

# NTL provides the subquadratic GCD backend and CanZass resolution --
# REQUIRED for production; build or install NTL with the gf2x
# multiplication backend (NTL_GF2X_LIB=on / libntl-dev + libgf2x-dev),
# exactly as for factor.cpp: NTL without gf2x is many times slower on
# large-degree polynomials (libntl.org/doc/tour-gf2x.html).
#
# Default link line is hardcoded (simplest and correct on both source
# builds in /usr/local and distro packages); a try-link of ntl_check.cpp
# VERIFIES it before building tsfactor and fails loudly with specifics
# instead of silently producing a crippled binary.  Overrides:
#   make NTLLIB="-L/opt/ntl/lib -lntl -lgf2x -lgmp"   (custom paths)
#   make NTL_OK=0                                     (force naive; test only)
NTLLIB ?= -lntl -lgf2x -lgmp
NTLCHK := $(firstword $(wildcard common/ntl_check.cpp ntl_check.cpp))
ifeq ($(origin NTL_OK),undefined)
NTL_OK := $(shell if [ -n "$(NTLCHK)" ] && $(CXX) $(NTLCHK) $(NTLLIB) \
  -o /tmp/ntl_chk_$$$$ >/dev/null 2>&1; then echo 1; else echo 0; fi; \
  rm -f /tmp/ntl_chk_$$$$)
endif
ifeq ($(NTL_OK),1)
NTLDEF = -DHAVE_NTL
NTLSRC = $(firstword $(wildcard common/gf2_gcd_ntl.cpp gf2_gcd_ntl.cpp))
NTLSRC_EMUL = build_emul/gf2_gcd_ntl.cpp
NTLMSG = NTL: OK ($(NTLLIB))
else
ifeq ($(NTLCHK),)
NTLMSG = NTL: FAILED -- ntl_check.cpp is MISSING from this directory, so the link check could not run
else
NTLMSG = NTL: FAILED -- '$(CXX) $(NTLCHK) $(NTLLIB)' does not link.  Install libntl-dev and libgf2x-dev (or set NTLLIB=... for custom paths); run 'make ntl-diag' for the full compiler error
endif
endif

CPU  = sieve_ref cantor_ref
CUDA = coarse_sieve coarse_sieve_wide cantor_cuda trinomial_stage2 trinomial_scan tsfactor

all: cpu cuda
cpu: $(CPU)
cuda: $(CUDA)

ntl-diag:
	@echo "== link check: $(CXX) $(NTLCHK) $(NTLLIB) =="
	@$(CXX) $(NTLCHK) $(NTLLIB) -o /tmp/ntl_diag_bin && \
	  echo "OK: NTL links and runs" && /tmp/ntl_diag_bin && \
	  rm -f /tmp/ntl_diag_bin || true

sieve_ref: sieve_ref.cpp gf2_small_field.h gf2_field_setup.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

coarse_sieve: coarse_sieve.cu gf2_small_field.h gf2_field_setup.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

coarse_sieve_wide: coarse_sieve.cu gf2_wide_field.h gf2_wide_setup.h
	$(NVCC) $(NVFLAGS) -DWIDE_FIELD $(INC) -o $@ $<

cantor_ref: cantor_ref.cpp gf2_cantor_core.h gf2_cantor_engine.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

cantor_cuda: cantor_cuda.cu gf2_cantor_cuda.h gf2_cantor_core.h gf2_cantor_engine.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

trinomial_stage2: trinomial_stage2.cu gf2_trinomial_cuda.h gf2_cantor_cuda.h gf2_small_field.h
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

trinomial_scan: trinomial_scan.cu gf2_trinomial_cuda.h gf2_gcd.h
	$(NVCC) $(NVFLAGS) $(INC) $(NTLDEF) -o $@ $< $(NTLSRC) $(NTLLIB) -Xcompiler -pthread

tsfactor: tsfactor.cu gf2_trinomial_cuda.h gf2_cantor_cuda.h gf2_gcd.h gf2_gcd_ntl.cpp
	@echo "== $(NTLMSG) =="
	@if [ "$(NTL_OK)" != "1" ] && [ "$(ALLOW_NO_NTL)" != "1" ]; then \
	  echo "ERROR: tsfactor needs NTL for factor resolution (CanZass)."; \
	  echo "  Kernel-testing-only override: make ALLOW_NO_NTL=1 tsfactor"; \
	  exit 1; fi
	$(NVCC) $(NVFLAGS) $(INC) $(NTLDEF) -o $@ tsfactor.cu $(NTLSRC) $(NTLLIB) -Xcompiler -pthread

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

emul-tsfactor:
	@echo "== $(NTLMSG) =="
	mkdir -p build_emul
	for h in gf2_small_field.h gf2_field_setup.h gf2_cantor_core.h gf2_cantor_engine.h cuda_emul.h gf2_gcd.h; do \
	  cp $$(ls common/$$h 2>/dev/null || ls $$h) build_emul/; done
	$(SEDLAUNCH) $$(ls common/gf2_cantor_cuda.h 2>/dev/null || ls gf2_cantor_cuda.h) > build_emul/gf2_cantor_cuda.h
	$(SEDLAUNCH) $$(ls common/gf2_trinomial_cuda.h 2>/dev/null || ls gf2_trinomial_cuda.h) > build_emul/gf2_trinomial_cuda.h
	printf '#pragma once\n#include "cuda_emul.h"\n' > build_emul/cuda_runtime.h
	$(SEDLAUNCH) tsfactor.cu > build_emul/tsfactor.cpp
	@if [ "$(NTL_OK)" = "1" ]; then cp $$(ls common/gf2_gcd_ntl.cpp 2>/dev/null || ls gf2_gcd_ntl.cpp) build_emul/; fi
	$(CXX) $(CXXFLAGS) -x c++ -Ibuild_emul $(NTLDEF) -pthread -o build_emul/tsfactor build_emul/tsfactor.cpp $(NTLSRC_EMUL) $(NTLLIB)

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

emultest: emul-coarse_sieve emul-cantor_cuda emul-trinomial_stage2 emul-trinomial_scan emul-tsfactor
	./build_emul/cantor_cuda selftest
	./build_emul/coarse_sieve --selftest
	./build_emul/trinomial_stage2 selftest
	./build_emul/trinomial_scan --selftest
	./build_emul/tsfactor --selftest

clean:
	rm -rf $(CPU) $(CUDA) build_emul
.PHONY: all cpu cuda clean emultest ntl-diag

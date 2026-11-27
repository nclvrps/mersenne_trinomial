# tsfactor: the trinomial factor scan, in two builds of the same program
#   tsfactor       CUDA (nvcc, CUDA 11+, sm_60+)
#   tsfactor_cpu   CPU only (g++), round 10
# plus two CPU tools for the gcd engine (gcdbench, gcdstress) and a CPU
# emulation of the CUDA build for testing without a GPU (emul-tsfactor).
#
#   make                everything this machine can build (the CUDA build
#                       is skipped, with a note, when nvcc is not found)
#   make cpu | cuda     one side
#   make emultest       --selftest of the emulated CUDA build and of
#                       tsfactor_cpu (no GPU needed; about 20 s)
#   make regress        ../tests/ts_regress.sh and ts_ckpt10.sh on
#                       tsfactor_cpu (ORIG=/path/to/older/tsfactor to
#                       compare against an older binary)

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

# gf2x itself, called directly by the gcd engine for its CPU products (no
# NTL object conversions).  Needs gf2x.h; detected by a try-link of
# gf2x_check.cpp with the same libraries.  Without it everything still
# works through NTL::mul, with a slower gcd.   make GF2X_OK=0 forces that.
GF2XCHK := $(firstword $(wildcard common/gf2x_check.cpp gf2x_check.cpp))
ifeq ($(origin GF2X_OK),undefined)
GF2X_OK := $(shell if [ -n "$(GF2XCHK)" ] && $(CXX) $(GF2XCHK) $(NTLLIB) \
  -o /tmp/gf2x_chk_$$$$ >/dev/null 2>&1; then echo 1; else echo 0; fi; \
  rm -f /tmp/gf2x_chk_$$$$)
endif
ifeq ($(NTL_OK)$(GF2X_OK),11)
NTLDEF += -DHAVE_GF2X
NTLMSG := $(NTLMSG), gf2x called directly
else
ifeq ($(NTL_OK),1)
NTLMSG := $(NTLMSG) -- but gf2x.h is not usable, so gcd products go through NTL (slower gcd)
endif
endif

CPU  = tsfactor_cpu gcdbench gcdstress
CUDA = tsfactor
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)

all: cpu $(if $(HAVE_NVCC),cuda,nocuda)
cpu: $(CPU)
cuda: $(CUDA)
nocuda:
	@echo "== $(NVCC) not found: built the CPU programs only (make cuda NVCC=/path/to/nvcc for the GPU build) =="

ntl-diag:
	@echo "== link check: $(CXX) $(NTLCHK) $(NTLLIB) =="
	@$(CXX) $(NTLCHK) $(NTLLIB) -o /tmp/ntl_diag_bin && \
	  echo "OK: NTL links and runs" && /tmp/ntl_diag_bin && \
	  rm -f /tmp/ntl_diag_bin || true

tsfactor: tsfactor.cu ts_core.h ts_cpu.h gf2_trin_host.h gf2_trinomial_cuda.h gf2_cantor_cuda.h gf2_gcd.h gf2_gcd_ntl.cpp gf2_ec.h gf2_hgcd_base.h
	@echo "== $(NTLMSG) =="
	@if [ "$(NTL_OK)" != "1" ] && [ "$(ALLOW_NO_NTL)" != "1" ]; then \
	  echo "ERROR: tsfactor needs NTL for factor resolution (CanZass)."; \
	  echo "  Kernel-testing-only override: make ALLOW_NO_NTL=1 tsfactor"; \
	  exit 1; fi
	$(NVCC) $(NVFLAGS) $(INC) $(NTLDEF) -o $@ tsfactor.cu $(NTLSRC) $(NTLLIB) -Xcompiler -pthread

# tsfactor without a GPU (round 10): the same program with the CPU backend
# (ts_cpu.h).  g++ only -- no CUDA.  -O3 for the squaring loops; OpenMP for
# --threads (optional: without -fopenmp the option is ignored).
CPUFLAGS ?= -O3 -march=native -Wall -Wextra -fopenmp
tsfactor_cpu: tsfactor_cpu.cpp ts_core.h ts_cpu.h gf2_trin_host.h gf2_gcd.h gf2_gcd_ntl.cpp gf2_ec.h gf2_hgcd_base.h gf2_cantor_core.h gf2_cantor_engine.h
	@echo "== $(NTLMSG) =="
	@if [ "$(NTL_OK)" != "1" ] && [ "$(ALLOW_NO_NTL)" != "1" ]; then \
	  echo "ERROR: tsfactor_cpu needs NTL for factor resolution (CanZass)."; \
	  echo "  Kernel-testing-only override: make ALLOW_NO_NTL=1 tsfactor_cpu"; \
	  exit 1; fi
	$(CXX) $(CPUFLAGS) $(INC) $(NTLDEF) -o $@ tsfactor_cpu.cpp $(NTLSRC) $(NTLLIB) -pthread

# CPU-only timing of the gcd paths on this machine's CPU and gf2x build:
# NTL's gcd against tsfactor's default (the certified chain), plus the
# reference rows.  No GPU.   ./gcdbench 136279841 588 [finish] [quick] [threads]
gcdbench: gcdbench.cpp gf2_gcd.h gf2_gcd_ntl.cpp gf2_ec.h gf2_hgcd_base.h gf2_cantor_core.h
	@echo "== $(NTLMSG) =="
	$(CXX) $(CXXFLAGS) $(INC) $(NTLDEF) -o $@ gcdbench.cpp $(firstword $(wildcard common/gf2_gcd_ntl.cpp gf2_gcd_ntl.cpp)) $(NTLLIB) -pthread

# CPU-only: every gcd path against NTL on randomized corner-case inputs
gcdstress: gcdstress.cpp gf2_gcd.h gf2_gcd_ntl.cpp gf2_ec.h gf2_hgcd_base.h gf2_cantor_core.h
	$(CXX) $(CXXFLAGS) $(INC) $(NTLDEF) -o $@ gcdstress.cpp $(firstword $(wildcard common/gf2_gcd_ntl.cpp gf2_gcd_ntl.cpp)) $(NTLLIB) -pthread

# Functional single-threaded CPU emulation of the CUDA programs (no GPU
# needed): kernel launches become sequential loops via cuda_emul.h.
# Useful for validating changes before spending GPU time.
SEDLAUNCH = sed -E 's/([A-Za-z_][A-Za-z0-9_]*)<<<([^>]*)>>>/EMUL_LAUNCH(\1, \2)/g; s/__device__ __forceinline__/static inline/'

emul-tsfactor:
	@echo "== $(NTLMSG) =="
	mkdir -p build_emul
	for h in gf2_cantor_core.h gf2_cantor_engine.h cuda_emul.h gf2_gcd.h gf2_ec.h gf2_hgcd_base.h gf2_trin_host.h ts_core.h ts_cpu.h; do \
	  cp $$(ls common/$$h 2>/dev/null || ls $$h) build_emul/; done
	$(SEDLAUNCH) $$(ls common/gf2_cantor_cuda.h 2>/dev/null || ls gf2_cantor_cuda.h) > build_emul/gf2_cantor_cuda.h
	$(SEDLAUNCH) $$(ls common/gf2_trinomial_cuda.h 2>/dev/null || ls gf2_trinomial_cuda.h) > build_emul/gf2_trinomial_cuda.h
	printf '#pragma once\n#include "cuda_emul.h"\n' > build_emul/cuda_runtime.h
	$(SEDLAUNCH) tsfactor.cu > build_emul/tsfactor.cpp
	@if [ "$(NTL_OK)" = "1" ]; then cp $$(ls common/gf2_gcd_ntl.cpp 2>/dev/null || ls gf2_gcd_ntl.cpp) build_emul/; fi
	$(CXX) $(CXXFLAGS) -x c++ -Ibuild_emul $(NTLDEF) -pthread -o build_emul/tsfactor build_emul/tsfactor.cpp $(NTLSRC_EMUL) $(NTLLIB)

emultest: emul-tsfactor tsfactor_cpu
	./build_emul/tsfactor --selftest
	./tsfactor_cpu --selftest

regress: tsfactor_cpu
	cd ../tests && ./ts_regress.sh ../tsfactor/tsfactor_cpu $(ORIG)
	cd ../tests && ./ts_ckpt10.sh ../tsfactor/tsfactor_cpu

clean:
	rm -rf $(CPU) $(CUDA) build_emul
.PHONY: all cpu cuda nocuda clean emultest regress ntl-diag emul-tsfactor

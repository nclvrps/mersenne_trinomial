# factor_ec: factor with hardware error checking, plus the original factor
# (the reference that regress.sh compares against) and the small error-
# checking simulators.  CPU only; needs NTL, gf2x and GMP.
#
#   make              everything
#   make factor_ec    just the program
#   make test         regress.sh + sweep.sh (about a minute)
#
# Add -DGF2X_HAVE_BMI2_SUPPORT to CXXFLAGS for the pdep squaring on CPUs
# with fast BMI2 (Intel Haswell+, AMD Zen 3+); without it the BLZ squaring
# is used, which is the faster one on Zen 1 and Zen 2.

CXXFLAGS=-O4 -march=native -fomit-frame-pointer -funroll-loops -fopenmp
LIBS=-lntl -lgf2x -lgmp

all: factor factor_ec gecsim ecbench ectest ectest2

factor: factor.o halfgcd.o
	g++ $(CXXFLAGS) -Wall -W -o $@ $^ $(LIBS)

factor.o: factor.cpp
	g++ $(CXXFLAGS) -DHAVE_CONFIG_H -I.     -Wall -W -MT $@ -MD -MP -c -o $@ $<

factor_ec: factor_ec.o halfgcd.o
	g++ $(CXXFLAGS) -Wall -W -o $@ $^ $(LIBS)

factor_ec.o: factor_ec.cpp
	g++ $(CXXFLAGS) -DHAVE_CONFIG_H -I.     -Wall -W -MT $@ -MD -MP -c -o $@ $<

halfgcd.o: halfgcd.cpp
	g++ $(CXXFLAGS) -DHAVE_CONFIG_H -I.     -Wall -W -MT $@ -MD -MP -c -o $@ $<

gecsim: gecsim.cpp gecsim_core.h
	g++ -O2 -o $@ $<

ecbench: ecbench.cpp ec_words.h
	g++ -O2 -o $@ $<

ectest: ectest.cpp
	g++ -O2 -o $@ $<

ectest2: ectest2.cpp ec_words.h
	g++ -O2 -o $@ $<

test: factor factor_ec
	./regress.sh
	./sweep.sh

clean:
	rm -f factor factor_ec gecsim ecbench ectest ectest2 *.o *.d

.PHONY: all test clean

# header dependencies written by -MD (ec_words.h, halfgcd.hpp)
-include $(wildcard *.d)

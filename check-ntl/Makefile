all: check-ntl

CC       ?= cc
CXX      ?= g++
CFLAGS   ?= -O2 -march=native
CXXFLAGS ?= -O2 -march=native

check-ntl: check-ntl.c
	$(CXX) $(CXXFLAGS) -DHAVE_CONFIG_H -I.     -Wall -W -fopenmp  -o $@ $< -lntl -lgf2x -lgmp

clean:
	rm check-ntl

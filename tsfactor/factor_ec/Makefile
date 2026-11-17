all: factor factor_ec gecsim ecbench ectest ectest2

CXXFLAGS=-O4 -march=native -fomit-frame-pointer -funroll-loops -fopenmp

factor: factor.o halfgcd.o
	g++ $(CXXFLAGS) -Wall -W -o $@ $^ -lntl -lgf2x -lgmp

factor.o: factor.cpp
	g++ $(CXXFLAGS) -DHAVE_CONFIG_H -I.     -Wall -W -MT $@ -MD -MP -c -o $@ $<

factor_ec: factor_ec.o halfgcd.o
	g++ $(CXXFLAGS) -Wall -W -o $@ $^ -lntl -lgf2x -lgmp

factor_ec.o: factor_ec.cpp
	g++ $(CXXFLAGS) -DHAVE_CONFIG_H -I.     -Wall -W -MT $@ -MD -MP -c -o $@ $<

halfgcd.o: halfgcd.cpp
	g++ $(CXXFLAGS) -DHAVE_CONFIG_H -I.     -Wall -W -MT $@ -MD -MP -c -o $@ $<

gecsim: gecsim.cpp
	g++ -O2 -o $@ $<

ecbench: ecbench.cpp
	g++ -O2 -o $@ $<

ectest: ectest.cpp
	g++ -O2 -o $@ $<

ectest2: ectest2.cpp
	g++ -O2 -o $@ $<

ectest_gpu: ectest_gpu.cpp
	g++ -O2 -std=c++11 -o $@ $<

eccuda_emul_test: eccuda_emul_test.cpp
	g++ -O2 -std=c++14 -I. -o $@ $<

clean:
	rm factor factor_ec gecsim ecbench ectest ectest2 ectest_gpu *.o *.d

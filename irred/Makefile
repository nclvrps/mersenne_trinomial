all: irred

CC       ?= cc
CFLAGS   ?= -O3 -march=native

irred: irred.c
	$(CC) $(CFLAGS) -Wall -Wextra -o $@ $<

irred.nearly_orig: irred.nearly_orig.c
	$(CC) $(CFLAGS) -Wall -Wextra -o $@ $<

clean:
	rm -f irred irred.nearly_orig

all: irred/irred build/irred315

CC       ?= cc
CFLAGS   ?= -O3 -march=native

irred/irred: irred/irred.c
	$(CC) $(CFLAGS) -Wall -Wextra -o $@ $<

build/irred315: build/irred315.c
	$(CC) $(CFLAGS) -Wno-old-style-definition -o $@ $<

clean:
	rm -f irred/irred build/irred315

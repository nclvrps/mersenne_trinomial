#!/bin/bash
# Timing with and without error checking.  r = 756839 runs to the end;
# r = 136279841 runs for 40 s each and reports ms per squaring.
B=$(readlink -f ${1:-build/irred})
cd /tmp
t() { local s=$(date +%s.%N); "$@" | tail -1; local e=$(date +%s.%N); echo "   $(echo "$e - $s" | bc) s  ($*)"; }
t $B --no-ckpt 756839 100001
t $B --no-ckpt --no-ec 756839 100001
t $B --no-ckpt --isa avx2 756839 100001
t $B --no-ckpt --isa generic 756839 100001
for ec in "" "--no-ec" "--threads 2" "--threads 2 --nt" "--nt"; do
  echo "r = 136279841 $ec:"
  ( timeout -s TERM 40 $B --no-ckpt $ec --progress 1000 136279841 12345 | grep '^iter' | tail -2 ) ; true
done

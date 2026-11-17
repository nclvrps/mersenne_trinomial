#!/bin/bash
# factor_ec single-fault sweep: one bit flipped at opportunity N, checking off vs on
TRUTH=$(./factor -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217 2>/dev/null | grep "^640")
offwrong=0; offbenign=0; onwrong=0; oncorrect=0; nofire=0
: > /tmp/wrongs.txt
for N in $(seq 1 150); do
  OFF=$(./factor_ec -ecfault $N -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217 2>/tmp/off.err | grep "^640")
  grep -q "EC INJECT" /tmp/off.err || { nofire=$((nofire+1)); continue; }
  ON=$(./factor_ec -ec -ecw 4 -ecfault $N -skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217 2>/dev/null | grep "^640")
  if [ "$OFF" = "$TRUTH" ]; then offbenign=$((offbenign+1)); else offwrong=$((offwrong+1)); echo "${OFF:-<no output>}" >> /tmp/wrongs.txt; fi
  if [ "$ON" = "$TRUTH" ]; then oncorrect=$((oncorrect+1)); else onwrong=$((onwrong+1)); fi
done
echo "single-fault sweep: r=3217, s=640 (flips to working s with r-s=640), m=8, q=4"
echo "  truth: $TRUTH"
echo "  injection opportunities probed: 150   (never reached: $nofire)"
echo "  checking OFF : $offwrong WRONG, $offbenign benign"
echo "  checking ON  : $onwrong wrong, $oncorrect correct"
echo "  distribution of wrong answers:"; sort /tmp/wrongs.txt | uniq -c | sed 's/^/    /'
[ $onwrong = 0 ]

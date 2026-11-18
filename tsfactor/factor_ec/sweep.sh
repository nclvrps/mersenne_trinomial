#!/bin/bash
# factor_ec single-fault sweep: one bit flipped at opportunity N, with
# checking off (-noec) and on (the default, window 4 here), then the same
# for a corrupted h-table initialisation (-ecfaultinit).
W=$(mktemp -d fec_sweep.XXXXXX)
A="-skip 60 -k 60 -m 8 -q 4 -s0 640 -s1 641 3217"
TRUTH=$(./factor $A 2>/dev/null | grep "^640")
offwrong=0; offbenign=0; onwrong=0; oncorrect=0; nofire=0
: > $W/wrongs.txt
for N in $(seq 1 150); do
  OFF=$(./factor_ec -noec -ecfault $N $A 2>$W/off.err | grep "^640")
  grep -q "EC INJECT" $W/off.err || { nofire=$((nofire+1)); continue; }
  ON=$(./factor_ec -ecw 4 -ecfault $N $A 2>/dev/null | grep "^640")
  if [ "$OFF" = "$TRUTH" ]; then offbenign=$((offbenign+1)); else offwrong=$((offwrong+1)); echo "${OFF:-<no output>}" >> $W/wrongs.txt; fi
  if [ "$ON" = "$TRUTH" ]; then oncorrect=$((oncorrect+1)); else onwrong=$((onwrong+1)); fi
done
echo "single-fault sweep: r=3217, s=640 (flips to working s with r-s=640), m=8, q=4"
echo "  truth: $TRUTH"
echo "  injection opportunities probed: 150   (never reached: $nofire)"
echo "  checking OFF : $offwrong WRONG, $offbenign benign"
echo "  checking ON  : $onwrong wrong, $oncorrect correct"
echo "  distribution of wrong answers:"; sort $W/wrongs.txt | uniq -c | sed 's/^/    /'
# the initialisation: a flipped table bit must be caught by the closed-form
# check and the table rebuilt; unchecked it ruins the answer
initbad=0; initoff=0
for S in 640 700 1000 1500; do
  B="-skip 60 -k 60 -m 8 -q 4 -s0 $S -s1 $((S+1)) 3217"
  T=$(./factor $B 2>/dev/null | grep "^$S")
  ON=$(./factor_ec -ecfaultinit $B 2>$W/on.err | grep "^$S")
  OFF=$(./factor_ec -noec -ecfaultinit $B 2>/dev/null | grep "^$S")
  { [ "$ON" = "$T" ] && grep -q "EC FAIL \[init_h\]" $W/on.err; } || initbad=$((initbad+1))
  [ "$OFF" = "$T" ] || initoff=$((initoff+1))
done
echo "  corrupted init_h, 4 runs: checking ON caught and repaired $((4-initbad)); checking OFF gave $initoff wrong answers"
rc=0; [ $onwrong = 0 ] && [ $initbad = 0 ] || rc=1
[ $rc = 0 ] && rm -rf $W
exit $rc

#!/bin/bash
# factor_ec regression: the checked program's output must equal the original
# factor's, with zero false alarms.  Needs ./factor, ./factor_ec and
# surv9689.txt.
#   1. every s in [100,400) at r = 1279, 3217, 4423, in four checking
#      configurations: the default (window 8) and windows 1, 4, 16
#   2. the default and -noec against each other
#   3. two threads (-mt 2)
#   4. the production sieving command line, s values on stdin
#   5. -Z (3.01): the same answers as without it
W=$(mktemp -d fec_regress.XXXXXX)
fail=0
cmp_run() {   # label, reference file, factor_ec args...
  local label=$1 ref=$2; shift 2
  ./factor_ec "$@" 2>$W/re.err | grep -v thread > $W/re.txt
  local nfail=$(grep -c "EC FAIL" $W/re.err)
  if diff -q $ref $W/re.txt >/dev/null && [ "$nfail" = "0" ]; then
    printf "  %-34s %4d trinomials  OK, 0 false alarms\n" "$label" $(wc -l < $ref)
  else printf "  %-34s MISMATCH (%s false alarms)\n" "$label" $nfail; fail=1; fi
}
for R in 1279 3217 4423; do
  ./factor -s0 100 -s1 400 $R 2>/dev/null | grep -v thread > $W/ro.txt
  cmp_run "r=$R default"        $W/ro.txt            -s0 100 -s1 400 $R
  for Wn in 1 4 16; do
    cmp_run "r=$R -ecw $Wn"     $W/ro.txt -ecw $Wn   -s0 100 -s1 400 $R
  done
  cmp_run "r=$R -noec"          $W/ro.txt -noec      -s0 100 -s1 400 $R
  cmp_run "r=$R -eci0"          $W/ro.txt -eci0      -s0 100 -s1 400 $R
done
./factor -mt 2 -m 8 -s0 100 -s1 400 4423 2>/dev/null | grep -v thread > $W/ro.txt
cmp_run "r=4423 -mt 2 -m 8" $W/ro.txt -mt 2 -m 8 -s0 100 -s1 400 4423
# the production sieving shape (r = 136279841 uses -k 55 -skip 55 -m 14 -q 16
# -maxd 279): the same options at small r, s values on stdin (-s0 > -s1).
# surv9689.txt lists the s < r/2 with no factor of degree <= 5.
OPTS="-p -f 1 -s0 101 -s1 100 -k 5 -skip 5 -m 6 -q 4 -maxd 279"
./factor $OPTS 9689 < surv9689.txt 2>/dev/null | grep -v thread > $W/ro.txt
cmp_run "r=9689 production options, stdin" $W/ro.txt $OPTS 9689 < surv9689.txt
# -Z q1 (3.01) caps the number of blocks but never gives up, so the answers
# are the same as without it -- also when q0 > q1 and with -f 0, the two
# cases factor_modified 2.02 got wrong
./factor -s0 100 -s1 400 4423 2>/dev/null | grep -v thread > $W/ro.txt
cmp_run "r=4423 -Z 3"                $W/ro.txt -Z 3           -s0 100 -s1 400 4423
cmp_run "r=4423 -q 8 -Z 3 (q0 > q1)" $W/ro.txt -q 8 -Z 3      -s0 100 -s1 400 4423
cmp_run "r=4423 -f 0 -q 8 -Z 3"      $W/ro.txt -f 0 -q 8 -Z 3 -s0 100 -s1 400 4423
cmp_run "r=4423 -z 2 -Z 3 (-Z wins)" $W/ro.txt -z 2 -Z 3      -s0 100 -s1 400 4423
[ $fail = 0 ] && echo "  ---- all regressions identical to the original factor ----" && rm -rf $W
[ $fail = 0 ] || echo "  ---- FAILURES; files kept in $W ----"
exit $fail

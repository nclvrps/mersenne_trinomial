#!/bin/bash
# factor_ec regression: checked output must equal the original factor's
fail=0
for R in 1279 3217 4423; do
  for W in 1 4 16; do
    ./factor               -s0 100 -s1 400 $R 2>/dev/null | grep -v thread > /tmp/ro.txt
    ./factor_ec -ec -ecw $W -s0 100 -s1 400 $R 2>/tmp/re.err | grep -v thread > /tmp/re.txt
    nfail=$(grep -c "EC FAIL" /tmp/re.err)
    if diff -q /tmp/ro.txt /tmp/re.txt >/dev/null && [ "$nfail" = "0" ]; then
      printf "  r=%-5s window=%-3s  %4d trinomials  OK, 0 false alarms\n" $R $W $(wc -l < /tmp/ro.txt)
    else printf "  r=%-5s window=%-3s  MISMATCH (%s false alarms)\n" $R $W $nfail; fail=1; fi
  done
done
[ $fail = 0 ] && echo "  ---- all regressions identical to unchecked output ----"
exit $fail

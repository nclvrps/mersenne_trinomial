#!/bin/bash
# All the fault-injection sweeps quoted in docs/irred_handoff.md
B=${1:-build/irred}
T=$(dirname "$0")
rc=0
$T/ec_sweep.sh $B 1279 500 sq 1279 || rc=1
$T/ec_sweep.sh $B 1279 418 sq 1279 || rc=1
$T/ec_sweep.sh $B 1279 500 sq 1279 --ec-block 8 --ec-window 4 || rc=1
$T/ec_sweep.sh $B 4423 2098 sq 4423 --isa generic || rc=1
for site in acc chk ver; do
  $T/ec_sweep.sh $B 1279 500 $site 200 --ec-block 8 --ec-window 4 || rc=1
  $T/ec_sweep.sh $B 4423 1000 $site 200 || rc=1
done
[ $rc -eq 0 ] && echo "fault sweep: all injected faults detected, all results right"
exit $rc

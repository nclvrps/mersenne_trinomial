#!/bin/bash
# The regression oracles: made by round 5's binaries through its dense-table
# and bucketed sieves, paths independent of the logs kernel that rounds 6-8
# use (round 6 removed them).  ~40 min, ~30 of them for OBIG (the bucketed
# path at d = 30-33); ~1 GB (OF28 alone is 944 MB).  FULL=0 skips the two
# production-range ones (OF28, OF32), which only the regression's F tests
# need.  Checked against oracles.sha256.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/build/r5/cs_wide; NW=$ROOT/build/r5/cs_narrow; O=$ROOT/oracles
[ -x $W ] && [ -x $NW ] || { echo "build the round-5 binaries first (setup.sh)"; exit 1; }
mkdir -p $O && cd $O || exit 1
cp -f $ROOT/fixtures/base18.survivors.txt .
# A: scratch -> 22, s <= 999999 (dense)
COARSE_METHOD=dense $W 136279841 22 OA --smax 999999 --no-checkpoint 2> gen_OA.log
# B: 19..23 from base18 (dense)
COARSE_METHOD=dense $W 136279841 23 OB --smax 999999 --load-survivors base18.survivors.txt --min-depth 19 --no-checkpoint 2> gen_OB.log
# J: 24..29 from base18, s <= 299999 (narrow: dense to 28, buckets at 29)
EMUL_FREE_MB=3000 $NW 136279841 29 OJ --smax 299999 --load-survivors base18.survivors.txt --min-depth 24 --no-checkpoint 2> gen_OJ.log
# BIG: 30..33 from base18, s <= 299999 (wide, forced buckets)
EMUL_FREE_MB=3000 COARSE_METHOD=buckets $W 136279841 33 OBIG --smax 299999 --load-survivors base18.survivors.txt --min-depth 30 --no-checkpoint 2> gen_OBIG.log
if [ "${FULL:-1}" = 1 ]; then
  # the production range s <= floor(r/2): scratch -> 28 (narrow dense), then 29..32 (narrow buckets)
  EMUL_FREE_MB=4000 $NW 136279841 28 OF28 --no-checkpoint 2> gen_OF28.log
  EMUL_FREE_MB=3000 $NW 136279841 32 OF32 --load-survivors OF28.survivors.txt --min-depth 29 --no-checkpoint 2> gen_OF32.log
fi
if [ "${FULL:-1}" = 1 ]; then sha256sum -c $ROOT/fixtures/oracles.sha256
else grep -v "OF28\|OF32" $ROOT/fixtures/oracles.sha256 | sha256sum -c; fi

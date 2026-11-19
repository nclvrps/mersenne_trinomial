#!/bin/bash
# round-10 checkpoint tests (emulated build).  Degrees 19-23 from base18 at
# s <= 999999 with chunks of 2^16 u: 4, 8, 16, 32, 64 chunks (124 in all).
# Round 8 added the lock file (L1-L3), the watchdog (W1-W3), and checkpoints
# of the grouped loop order and of TP from global memory (K22-K24); round 9
# checkpoints of the cubic tower (K25: resumed by round 9, refused mid-degree
# by round 8) and round 8's checkpoints under round 9 and back (K26-K27).
# Round 10: the tuning record (who chose a degree's engine and order) and
# re-tuning on resume (K28-K32), --load (K33), polling's queued rounds with
# checkpoints (K34); round 8's checkpoints are now re-tuned (K26).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=$ROOT/build/r10/cs; R5=$ROOT/build/r5/cs_wide; OR=$ROOT/oracles
R6B=$ROOT/build/r6/cs; R7B=$ROOT/build/r7/cs; R8B=$ROOT/build/r8/cs; R9B=$ROOT/build/r9/cs
mkdir -p $ROOT/work/ckpt && cd $ROOT/work/ckpt || exit 1
cp -f $ROOT/fixtures/base18.survivors.txt .
fail=0
ck2() { if cmp -s "$1.found.cert" "$2.found.cert" && cmp -s "$1.survivors.txt" "$2.survivors.txt"; then echo "PASS  $3"; else echo "FAIL  $3"; fail=1; fi; }
yes_() { echo "PASS  $1"; }; no_() { echo "FAIL  $1"; fail=1; }
export COARSE_TEST_SMAX=999999 COARSE_LOGS_CHUNK_LOG2=16
ARGS="136279841 19-23 K --load-survivors base18.survivors.txt"
clean() { rm -f K.* K_*.log; }
REF=$OR/OB

# K1: stop mid-degree (d=22, 9 of 32 chunks) -> K.ckpt; the same command continues
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_1a.log 2>&1; rc=$?
[ $rc = 0 ] && [ -s K.ckpt ] && grep -q "checkpoint: d=22 chunk 9/32, delta" K_1a.log && grep -q "run the same command again" K_1a.log && yes_ "K1 stop at d=22 chunk 9/32 writes K.ckpt (delta vs the seed)" || no_ "K1 stop (rc=$rc)"
$N $ARGS > K_1b.log 2>&1; rc=$?
grep -q "found checkpoint K.ckpt (written .*degree 22, chunk 9 of 32, [0-9.]*s into it)" K_1b.log && grep -q "^d=22: resuming at chunk 9/32" K_1b.log && yes_ "K1 same command resumes at d=22 chunk 9/32" || no_ "K1 resume"
ck2 K $REF "K1 resumed run == uninterrupted reference"
[ $rc = 0 ] && [ ! -e K.ckpt ] && [ ! -e K.ckpt.lock ] && yes_ "K1 finished run removed K.ckpt and its lock" || no_ "K1 cleanup (rc=$rc)"

# K2: a different command line refuses and leaves the checkpoint alone
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_2a.log 2>&1
cp K.ckpt K_saved.ckpt
$N 136279841 19-24 K --load-survivors base18.survivors.txt > K_2b.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "belongs to a different run: it was written by a run of degrees 19-23, this run is 19-24" K_2b.log && yes_ "K2 other degree range refused (exit 1)" || no_ "K2 range (rc=$rc)"
head -n 1000 base18.survivors.txt > other.txt
$N 136279841 19-23 K --load-survivors other.txt > K_2c.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "a file with different contents" K_2c.log && yes_ "K2 other seed contents refused" || no_ "K2 seed (rc=$rc)"
COARSE_TEST_SMAX=899999 $N $ARGS > K_2d.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "it is for r=136279841 (s <= 999999), this run is for r=136279841 (s <= 899999)" K_2d.log && yes_ "K2 other s range refused" || no_ "K2 smax (rc=$rc)"
$N 136279841 2-23 K > K_2e.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "degrees 19-23, this run is 2-23" K_2e.log && yes_ "K2 a run from scratch on a seeded run's checkpoint refused" || no_ "K2 scratch (rc=$rc)"
cmp -s K_saved.ckpt K.ckpt && yes_ "K2 refused runs left the checkpoint untouched" || no_ "K2 untouched"

# K3: --fresh sets the old checkpoint aside and starts over
$N 136279841 19-24 K --load-survivors base18.survivors.txt --fresh > K_3.log 2>&1; rc=$?
[ $rc = 0 ] && [ -s K.ckpt.old ] && grep -q "set 1 earlier checkpoint file aside" K_3.log && yes_ "K3 --fresh: old checkpoint kept as K.ckpt.old, the new run completes" || no_ "K3 --fresh (rc=$rc)"

# K4: --no-checkpoint writes nothing, even on a stop
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS --no-checkpoint > K_4.log 2>&1
[ ! -e K.ckpt ] && [ ! -e K.ckpt.lock ] && yes_ "K4 --no-checkpoint: no files" || no_ "K4 --no-checkpoint"

# K5: a round-5 (v3) checkpoint is refused; --fresh sets it aside
clean
COARSE_METHOD=logs COARSE_TEST_STOP_AFTER_BUCKETS=37 $R5 136279841 23 K --smax 999999 --load-survivors base18.survivors.txt --min-depth 19 > K_5a.log 2>&1
if [ -s K.ckpt ]; then
  $N $ARGS > K_5b.log 2>&1; rc=$?
  [ $rc = 1 ] && grep -q "not a readable checkpoint (format v4, rounds 6-10)" K_5b.log && grep -q "no generation of it is usable" K_5b.log && yes_ "K5 round-5 checkpoint refused (exit 1)" || no_ "K5 v3 refusal (rc=$rc)"
  $N $ARGS --fresh > K_5c.log 2>&1; rc=$?
  [ $rc = 0 ] && [ -s K.ckpt.old ] && yes_ "K5 --fresh sets it aside and runs" || no_ "K5 --fresh (rc=$rc)"
  ck2 K $REF "K5 that run == reference"
else no_ "K5 the round-5 binary wrote no checkpoint"; fi

# K6: a second run on the same checkpoint is refused while the first holds it
clean
( flock -x K.ckpt.lock -c "sleep 4" ) & sleep 0.5
$N $ARGS > K_6.log 2>&1; rc=$?
wait
[ $rc = 1 ] && grep -q "another run is using checkpoint K.ckpt" K_6.log && yes_ "K6 concurrent run refused" || no_ "K6 lock (rc=$rc)"
rm -f K.ckpt.lock

# K7: a real SIGINT mid-run saves; the same command continues (twice)
clean; rm -f KR.*
A7="136279841 24-29 K --load-survivors base18.survivors.txt"
$N 136279841 24-29 KR --load-survivors base18.survivors.txt --no-checkpoint > K_7r.log 2>&1
$N $A7 > K_7a.log 2>&1 & pid=$!          # SIGINT once degree 27 is under way
for i in $(seq 1 1200); do grep -q "^d=27 logs" K_7a.log && break; sleep 0.05; done
sleep 0.3; kill -INT $pid; wait $pid; rc=$?
if grep -q "stopping on request" K_7a.log && [ -s K.ckpt ]; then
  COARSE_TEST_STOP_AFTER_CHUNKS=100 $N $A7 > K_7b.log 2>&1
  $N $A7 > K_7c.log 2>&1
  grep -q "found checkpoint" K_7b.log && grep -q "found checkpoint" K_7c.log && yes_ "K7 SIGINT save, then two automatic continuations" || no_ "K7 continuations"
  ck2 K KR "K7 SIGINT-interrupted run == uninterrupted run (d=24..29)"
else
  no_ "K7 (no stop on SIGINT; rc=$rc)"; cp K_7a.log K_7a.failed.log
fi

# K8: a resumed degree keeps the engine / subfield the tuner chose (no re-tune)
clean
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=20 $N $ARGS > K_8a.log 2>&1
grep -q "tune d=21.* -> RR r5-order\]" K_8a.log && yes_ "K8 first run: d=21 tuned to RR (pick 1), stopped at chunk 8/16" || no_ "K8 first run"
COARSE_TUNE_MIN=19 $N $ARGS > K_8b.log 2>&1
grep -q "^d=21: resuming at chunk 8/16" K_8b.log && grep -q "^d=21 logs (RR, round-5 loop order" K_8b.log && ! grep -q "tune d=21" K_8b.log && grep -q "tune d=23" K_8b.log && yes_ "K8 resumed d=21 kept RR without re-tuning; d=23 tuned afresh" || no_ "K8 RR kept"
ck2 K $REF "K8 == reference"
clean
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=3 COARSE_TEST_STOP_AFTER_CHUNKS=6 $N $ARGS > K_8c.log 2>&1
COARSE_TUNE_MIN=19 $N $ARGS > K_8d.log 2>&1
grep -q "tune d=20.* -> GF(2^2) r5-order\]" K_8c.log && grep -q "^d=20 logs (tower over GF(2^2)" K_8d.log && ! grep -q "tune d=20" K_8d.log && yes_ "K8 resumed d=20 kept the GF(2^2) tower without re-tuning" || no_ "K8 subfield kept"
ck2 K $REF "K8 (subfield) == reference"
clean
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=6 $N $ARGS > K_8e.log 2>&1
COARSE_TUNE_MIN=19 $N $ARGS > K_8f.log 2>&1
grep -q "tune d=20.* -> GF(2^5) r9-order\]" K_8e.log && grep -q "^d=20 logs (tower over GF(2^5), round-9 loop order" K_8f.log && ! grep -q "tune d=20" K_8f.log && yes_ "K8 resumed d=20 kept round 9's loop order without re-tuning" || no_ "K8 loop order kept"
ck2 K $REF "K8 (loop order) == reference"

# K9: resuming with another chunk size is refused
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_9a.log 2>&1
COARSE_LOGS_CHUNK_LOG2=15 $N $ARGS > K_9b.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "the checkpoint has 32 chunks but this run has 64" K_9b.log && yes_ "K9 other chunk size refused" || no_ "K9 chunks (rc=$rc)"

# K10: damaged newest generations fall back to older ones
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS --checkpoint-mins 0.0001 > K_10a.log 2>&1
if [ -s K.ckpt ] && [ -s K.ckpt.1 ] && [ -s K.ckpt.2 ]; then
  head -c 100 K.ckpt.1 > t && mv t K.ckpt.1                    # header cut
  sz=$(stat -c %s K.ckpt); head -c $((sz / 2)) K.ckpt > t && mv t K.ckpt   # payload cut
  $N $ARGS --checkpoint-mins 0.0001 > K_10b.log 2>&1
  grep -q "checkpoint K.ckpt: truncated or damaged" K_10b.log && grep -q "checkpoint K.ckpt.1: not a readable checkpoint (format v4, rounds 6-10)" K_10b.log && grep -q "continuing from K.ckpt.2" K_10b.log && yes_ "K10 two damaged generations skipped, continued from K.ckpt.2" || no_ "K10 fallback"
  ck2 K $REF "K10 == reference"
else no_ "K10 three generations not written"; fi

# K11: a stop right after the last degree: the rerun only writes the outputs
clean
COARSE_TEST_STOP_AFTER_CHUNKS=124 $N $ARGS > K_11a.log 2>&1
$N $ARGS > K_11b.log 2>&1; rc=$?
grep -q "start of degree 24" K_11b.log && grep -q "every degree is done already" K_11b.log && [ $rc = 0 ] && [ ! -e K.ckpt ] && yes_ "K11 finished checkpoint: outputs written, checkpoint removed" || no_ "K11 finished (rc=$rc)"
ck2 K $REF "K11 == reference"

# K12: a stop at a degree boundary (after d=19's 4 chunks)
clean
COARSE_TEST_STOP_AFTER_CHUNKS=4 $N $ARGS > K_12a.log 2>&1
$N $ARGS > K_12b.log 2>&1
grep -q "start of degree 20" K_12b.log && ! grep -q "^d=19 logs" K_12b.log && yes_ "K12 degree-boundary checkpoint continues at d=20" || no_ "K12 boundary"
ck2 K $REF "K12 == reference"

# K13: the number of GPUs may change between stop and resume
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_13a.log 2>&1
EMUL_NDEV=2 $N $ARGS > K_13b.log 2>&1
ck2 K $REF "K13 stopped on 1 GPU, resumed on 2 == reference"
clean
EMUL_NDEV=2 COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_13c.log 2>&1
grep -q "checkpoint: d=22 chunk 10/32" K_13c.log && yes_ "K13 2 GPUs stop on a round boundary (chunk 10/32)" || no_ "K13 2-GPU stop"
$N $ARGS > K_13d.log 2>&1
ck2 K $REF "K13 stopped on 2 GPUs, resumed on 1 == reference"

# K14: a run from scratch checkpoints a full image (no seed); resumed == oracle
clean; rm -f KF.*
COARSE_LOGS_CHUNK_LOG2=14 COARSE_TEST_STOP_AFTER_CHUNKS=100 $N 136279841 2-22 KF > K_14a.log 2>&1
grep -q "checkpoint: d=21 chunk 24/64, full image" K_14a.log && yes_ "K14 scratch run: full-image checkpoint at d=21 chunk 24/64" || no_ "K14 full image"
COARSE_LOGS_CHUNK_LOG2=14 $N 136279841 2-22 KF > K_14b.log 2>&1
ck2 KF $OR/OA "K14 resumed scratch run == round-5 dense oracle"

# K15 (round 7): the time of all runs.  The resumed degree's done line gives
# the time in all runs and this run's; the total line both.  Exactly: with
# the checkpoint's times set to 1000 s (degree 22 so far) and 500 s (the
# degrees before it), d=22 shows 1000 s + this run's pass, the total 500 s +
# the degrees' times.
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_15a.log 2>&1
cp K.ckpt K_15saved.ckpt
$N $ARGS > K_15b.log 2>&1
grep -q "^d=22 done (logs/[A-Za-z]*, 32 chunks): [0-9]* classes, [0-9]* heavy progressions, [0-9.]*s in all (this run [0-9.]*s, resumed at chunk 9 of 32)$" K_15b.log && grep -q "^d=23 done (logs/[A-Za-z]*, 64 chunks): [0-9]* classes, [0-9]* heavy progressions, [0-9.]*s$" K_15b.log && grep -q "^total sieve time: [0-9.]*s in this run, [0-9.]*s in all$" K_15b.log && yes_ "K15 resumed: d=22 done line with the time in all and this run's; the total line both" || no_ "K15 format"
ck2 K $REF "K15 == reference"
cp K_15saved.ckpt K.ckpt
python3 - K.ckpt 1000.0 500.0 <<'PYEOF'
import struct, sys
p, deg, run = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
b = bytearray(open(p, 'rb').read())
assert b[:8] == b'GF2SVCK4'
flags = struct.unpack_from('<I', b, 8 + 6 * 4)[0]
assert flags & 12 == 12, flags          # both times recorded
struct.pack_into('<f', b, 8 + 11 * 4, run)
struct.pack_into('<d', b, 8 + 48 + 7 * 8, deg)
open(p, 'wb').write(b)
PYEOF
$N $ARGS > K_15c.log 2>&1
python3 - K_15c.log <<'PYEOF' && yes_ "K15 the recorded times carry over exactly (1000 s + this pass; 500 s + the degrees)" || no_ "K15 arithmetic"
import re, sys
t = open(sys.argv[1]).read()
m22 = re.search(r'^d=22 done .* ([0-9.]+)s in all \(this run ([0-9.]+)s, resumed at chunk 9 of 32\)$', t, re.M)
m23 = re.search(r'^d=23 done .* ([0-9.]+)s$', t, re.M)
mt = re.search(r'^total sieve time: ([0-9.]+)s in this run, ([0-9.]+)s in all$', t, re.M)
a22, this22, a23 = float(m22.group(1)), float(m22.group(2)), float(m23.group(1))
proc, allt = float(mt.group(1)), float(mt.group(2))
ok = 1000.0 <= a22 <= 1000.0 + this22 + 0.02 and abs(allt - (500.0 + a22 + a23)) <= 0.03 and proc < 100
print("d22 %.2f (this %.2f), d23 %.2f, total %.2f in all, %.2f this run" % (a22, this22, a23, allt, proc))
sys.exit(0 if ok else 1)
PYEOF
ck2 K $REF "K15 (times patched) == reference"

# K16: a round-6 checkpoint (no times recorded): the resumed degree says so,
# the total line gives this run's time alone
clean
cp K_15saved.ckpt K.ckpt
python3 - K.ckpt <<'PYEOF'
import struct, sys
p = sys.argv[1]
b = bytearray(open(p, 'rb').read())
o = 8 + 6 * 4
struct.pack_into('<I', b, o, struct.unpack_from('<I', b, o)[0] & ~12)
struct.pack_into('<I', b, 8 + 11 * 4, 0)
struct.pack_into('<Q', b, 8 + 48 + 7 * 8, 0)
open(p, 'wb').write(b)
PYEOF
$N $ARGS > K_16.log 2>&1
grep -q "found checkpoint K.ckpt (written .*degree 22, chunk 9 of 32)" K_16.log && grep -q "^d=22 done .* [0-9.]*s in this run (resumed at chunk 9 of 32; the checkpoint, from round 6, has no time for the runs before)$" K_16.log && grep -q "^total sieve time: [0-9.]*s$" K_16.log && yes_ "K16 round-6 checkpoint: this run's time only, said so" || no_ "K16 no times"
ck2 K $REF "K16 == reference"

# K17: a degree-boundary checkpoint carries the earlier degrees' time
clean
COARSE_TEST_STOP_AFTER_CHUNKS=4 $N $ARGS > K_17a.log 2>&1
python3 - K.ckpt <<'PYEOF' && yes_ "K17 boundary checkpoint: degree time 0, run time recorded (> 0)" || no_ "K17 header"
import struct, sys
b = open(sys.argv[1], 'rb').read()
flags = struct.unpack_from('<I', b, 8 + 6 * 4)[0]
run = struct.unpack_from('<f', b, 8 + 11 * 4)[0]
deg = struct.unpack_from('<d', b, 8 + 48 + 7 * 8)[0]
sys.exit(0 if flags & 12 == 12 and deg == 0.0 and run > 0 else 1)
PYEOF
$N $ARGS > K_17b.log 2>&1
grep -q "start of degree 20" K_17b.log && grep -q "^d=20 done .* [0-9.]*s$" K_17b.log && grep -q "^total sieve time: [0-9.]*s in this run, [0-9.]*s in all$" K_17b.log && yes_ "K17 continued at d=20: plain done lines, the total in all" || no_ "K17 boundary resume"

# K18: --no-checkpoint with the test stop says that nothing was saved
clean
COARSE_TEST_STOP_AFTER_CHUNKS=5 $N $ARGS --no-checkpoint > K_18.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "stopping after 5 chunks (COARSE_TEST_STOP_AFTER_CHUNKS); no checkpoint, nothing saved" K_18.log && ! grep -q "run the same command again" K_18.log && [ ! -e K.ckpt ] && yes_ "K18 --no-checkpoint test stop: nothing saved, said so" || no_ "K18 (rc=$rc)"

# K19: a real round-6 checkpoint (written by the round-6 binary) continues
# under round 9, with the same results
if [ -x $R6B ]; then
  clean
  COARSE_TEST_STOP_AFTER_CHUNKS=37 $R6B $ARGS > K_19a.log 2>&1
  $N $ARGS > K_19b.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=22: resuming at chunk 9/32" K_19b.log && grep -q "the checkpoint, from round 6, has no time for the runs before" K_19b.log && yes_ "K19 round-6 checkpoint resumed by round 10" || no_ "K19 (rc=$rc)"
  ck2 K $REF "K19 == reference"
else no_ "K19 no round-6 binary at $R6B"; fi

# K19b: a round-7 checkpoint (written by the round-7 binary) continues
# under round 9, with its times
if [ -x $R7B ]; then
  clean
  COARSE_TEST_STOP_AFTER_CHUNKS=37 $R7B $ARGS > K_19c.log 2>&1
  $N $ARGS > K_19d.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=22: resuming at chunk 9/32" K_19d.log && grep -q "^d=22 done .* [0-9.]*s in all (this run [0-9.]*s, resumed at chunk 9 of 32)$" K_19d.log && yes_ "K19b round-7 checkpoint resumed by round 10 (times carried)" || no_ "K19b (rc=$rc)"
  ck2 K $REF "K19b == reference"
else no_ "K19b no round-7 binary at $R7B"; fi

# K20: the fill (COARSE_FILL) may change between stop and resume
clean
COARSE_FILL=0 COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_20a.log 2>&1
$N $ARGS > K_20b.log 2>&1
grep -q "^d=22: resuming at chunk 9/32" K_20b.log && yes_ "K20 stopped with round 6's fill, resumed with the pooled one" || no_ "K20 resume"
ck2 K $REF "K20 == reference"

# K21: and the other way: round 6's binary continues a round-9 checkpoint
if [ -x $R6B ]; then
  clean
  COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_21a.log 2>&1
  $R6B $ARGS > K_21b.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=22: resuming at chunk 9/32" K_21b.log && yes_ "K21 round-10 checkpoint resumed by round 6" || no_ "K21 (rc=$rc)"
  ck2 K $REF "K21 == reference"
fi

# K22 (round 8): a resumed degree keeps the grouped loop order without
# re-tuning (d=20: pick 2 = GF(2^5) grouped)
clean
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=2 COARSE_TEST_STOP_AFTER_CHUNKS=6 $N $ARGS > K_22a.log 2>&1
COARSE_TUNE_MIN=19 $N $ARGS > K_22b.log 2>&1
grep -q "tune d=20.* -> GF(2^5) grouped\]" K_22a.log && grep -q "^d=20: resuming at chunk 2/8" K_22b.log && grep -q "^d=20 logs (tower over GF(2^5), grouped loop order" K_22b.log && ! grep -q "tune d=20" K_22b.log && yes_ "K22 resumed d=20 kept the grouped loop order without re-tuning" || no_ "K22 grouped order kept"
ck2 K $REF "K22 == reference"

# K23: ... and TP from global memory (the test hook offers it at d=20:
# pick 3 = GF(2^5) r5-order TP-global); the checkpoint's flag says so
clean
COARSE_TEST_TP_CHOICE=1 COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=3 COARSE_TEST_STOP_AFTER_CHUNKS=6 $N $ARGS > K_23a.log 2>&1
python3 - K.ckpt <<'PYEOF' && yes_ "K23 the checkpoint records TP from global memory (flag 16)" || no_ "K23 flag"
import struct, sys
b = open(sys.argv[1], 'rb').read()
flags = struct.unpack_from('<I', b, 8 + 6 * 4)[0]
sys.exit(0 if flags & 16 else 1)
PYEOF
COARSE_TUNE_MIN=19 $N $ARGS > K_23b.log 2>&1
grep -q "tune d=20.* -> GF(2^5) r5-order TP-global\]" K_23a.log && grep -q "^d=20 logs (tower over GF(2^5), round-5 loop order, TP from global memory" K_23b.log && ! grep -q "tune d=20" K_23b.log && yes_ "K23 resumed d=20 kept TP from global memory without re-tuning" || no_ "K23 TP kept"
ck2 K $REF "K23 == reference"

# K24: round 7's binary continues round-9 checkpoints of the grouped order
# and of TP from global memory (it reads the order as its own pipelined one
# and ignores the flag): the same results
if [ -x $R7B ]; then
  clean
  COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=2 COARSE_TEST_STOP_AFTER_CHUNKS=6 $N $ARGS > K_24a.log 2>&1
  COARSE_TUNE_MIN=19 $R7B $ARGS > K_24b.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=20: resuming at chunk 2/8" K_24b.log && yes_ "K24 round 7 resumed a grouped-order checkpoint" || no_ "K24 grouped (rc=$rc)"
  ck2 K $REF "K24 (grouped) == reference"
  clean
  COARSE_TEST_TP_CHOICE=1 COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=3 COARSE_TEST_STOP_AFTER_CHUNKS=6 $N $ARGS > K_24c.log 2>&1
  COARSE_TUNE_MIN=19 $R7B $ARGS > K_24d.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=20: resuming at chunk 2/8" K_24d.log && yes_ "K24 round 7 resumed a TP-global checkpoint" || no_ "K24 TP (rc=$rc)"
  ck2 K $REF "K24 (TP) == reference"
else no_ "K24 no round-7 binary at $R7B"; fi

# K25 (round 9): a degree sieved with the cubic tower (d=24 of 24-25 from
# OB's survivors, 128 chunks): stopped mid-degree, the same command -- even
# without COARSE_METHOD -- resumes it with the cubic tower; round 8's binary
# refuses that checkpoint (it has no such engine) and leaves it alone
AC="136279841 24-25 KC --load-survivors $OR/OB.survivors.txt"
rm -f KC.* KCR.*
$N 136279841 24-25 KCR --load-survivors $OR/OB.survivors.txt --no-checkpoint > K_25r.log 2>&1
COARSE_METHOD=cubic COARSE_TEST_STOP_AFTER_CHUNKS=40 $N $AC > K_25a.log 2>&1; rc=$?
[ $rc = 0 ] && [ -s KC.ckpt ] && grep -q "checkpoint: d=24 chunk 40/128" K_25a.log && grep -q "^d=24 logs (cubic tower over GF(2^8)" K_25a.log && yes_ "K25 cubic tower stopped at d=24 chunk 40/128" || no_ "K25 stop (rc=$rc)"
cp KC.ckpt K_25saved.ckpt
if [ -x $R8B ]; then
  $R8B $AC > K_25b.log 2>&1; rc=$?
  [ $rc = 1 ] && grep -q "d=24: the checkpoint's engine is not possible here" K_25b.log && cmp -s KC.ckpt K_25saved.ckpt && yes_ "K25 round 8 refuses the cubic checkpoint (exit 1), leaves it alone" || no_ "K25 round 8 refusal (rc=$rc)"
else no_ "K25 no round-8 binary at $R8B"; fi
$N $AC > K_25c.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "^d=24: resuming at chunk 40/128" K_25c.log && grep -q "^d=24 logs (cubic tower over GF(2^8), round-5 loop order" K_25c.log && grep -q "^d=25 logs (PH" K_25c.log && [ ! -e KC.ckpt ] && yes_ "K25 round 10 resumes it with the cubic tower (d=25: the usual engine)" || no_ "K25 resume (rc=$rc)"
ck2 KC KCR "K25 == uninterrupted run (tower at d=24)"
rm -f KC.*
COARSE_TUNE_MIN=24 COARSE_TUNE_PICK=11 COARSE_TEST_STOP_AFTER_CHUNKS=40 $N $AC > K_25d.log 2>&1
COARSE_TUNE_MIN=24 $N $AC > K_25e.log 2>&1
grep -q "tune d=24.* -> cubic GF(2^8) grouped\]" K_25d.log && grep -q "^d=24: resuming at chunk 40/128" K_25e.log && grep -q "^d=24 logs (cubic tower over GF(2^8), grouped loop order" K_25e.log && ! grep -q "tune d=24" K_25e.log && grep -q "tune d=25" K_25e.log && yes_ "K25 a tuned cubic tower (grouped) is kept on resume without re-tuning" || no_ "K25 tuned cubic kept"
ck2 KC KCR "K25 (tuned) == uninterrupted run"
rm -f KC.*

# K26: round 8's checkpoints continue under round 10: its loop order 1
# (round 6's there) as round 9's, its grouped order as the grouped one --
# kept with --no-tune; without it a tuned degree is tuned again (round 10:
# the checkpoint does not say who chose them)
if [ -x $R8B ]; then
  clean
  COARSE_TEST_STOP_AFTER_CHUNKS=37 $R8B $ARGS > K_26a.log 2>&1
  $N $ARGS > K_26b.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=22: resuming at chunk 9/32" K_26b.log && grep -q "^d=22 done .* [0-9.]*s in all (this run [0-9.]*s, resumed at chunk 9 of 32)$" K_26b.log && yes_ "K26 round-8 checkpoint resumed by round 10 (times carried)" || no_ "K26 (rc=$rc)"
  ck2 K $REF "K26 == reference"
  clean
  COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=6 $R8B $ARGS > K_26c.log 2>&1
  cp K.ckpt K_26saved.ckpt
  COARSE_TUNE_MIN=19 $N $ARGS --no-tune > K_26d.log 2>&1; rc=$?
  grep -q "tune d=20.* -> GF(2^5) r6-order\]" K_26c.log && grep -q "^d=20: resuming at chunk 2/8" K_26d.log && grep -q "^d=20 logs (tower over GF(2^5), round-9 loop order" K_26d.log && ! grep -q "tune d=20" K_26d.log && yes_ "K26 round 8's r6-order degree resumed in round 9's order (--no-tune)" || no_ "K26 r6-order (rc=$rc)"
  ck2 K $REF "K26 (r6-order) == reference"
  rm -f K.found.cert K.survivors.txt; cp K_26saved.ckpt K.ckpt
  COARSE_TUNE_MIN=19 COARSE_TUNE_REPS=1 $N $ARGS > K_26g.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=20: the checkpoint's choice, GF(2^5) r9-order, was made by an earlier round (6-9); tuning again (--no-tune keeps it)$" K_26g.log && grep -q "tune d=20" K_26g.log && grep -q "^d=20: resuming at chunk 2/8" K_26g.log && yes_ "K26 without --no-tune the round-8 degree is tuned again, said so" || no_ "K26 re-tune (rc=$rc)"
  ck2 K $REF "K26 (re-tuned) == reference"
  clean
  COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=2 COARSE_TEST_STOP_AFTER_CHUNKS=6 $R8B $ARGS > K_26e.log 2>&1
  COARSE_TUNE_MIN=19 $N $ARGS --no-tune > K_26f.log 2>&1; rc=$?
  grep -q "^d=20: resuming at chunk 2/8" K_26f.log && grep -q "^d=20 logs (tower over GF(2^5), grouped loop order" K_26f.log && yes_ "K26 round 8's grouped degree resumed in the grouped order (--no-tune)" || no_ "K26 grouped (rc=$rc)"
  ck2 K $REF "K26 (grouped) == reference"

# K27: and round 8's binary continues round 10's checkpoints of the other
# engines and orders (round 9's order as its round-6 one; the tuning record
# after best[] is never read)
  clean
  COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_27a.log 2>&1
  $R8B $ARGS > K_27b.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=22: resuming at chunk 9/32" K_27b.log && grep -q "^d=22 logs (tower, classic, round-6 loop order" K_27b.log && yes_ "K27 round-10 checkpoint (round 9's order) resumed by round 8" || no_ "K27 (rc=$rc)"
  ck2 K $REF "K27 == reference"
else no_ "K26-K27 no round-8 binary at $R8B"; fi

# K28 (round 10): a degree tuned by round 9 and resumed by round 10 is
# tuned again ("an earlier round"); with --no-tune it keeps round 9's choice
if [ -x $R9B ]; then
  clean
  COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=20 $R9B $ARGS > K_28a.log 2>&1
  cp K.ckpt K_28saved.ckpt
  COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=2 $N $ARGS > K_28b.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "tune d=21.* -> RR r5-order\]" K_28a.log && grep -q "^d=21: the checkpoint's choice, RR r5-order, was made by an earlier round (6-9); tuning again (--no-tune keeps it)$" K_28b.log && grep -q "tune d=21.* -> RR r9-order\]" K_28b.log && grep -q "^d=21: resuming at chunk 8/16" K_28b.log && grep -q "^d=21 logs (RR, round-9 loop order" K_28b.log && yes_ "K28 round 9's d=21 tuned again by round 10 (pick 2: RR r9-order)" || no_ "K28 re-tune (rc=$rc)"
  ck2 K $REF "K28 (re-tuned) == reference"
  rm -f K.found.cert K.survivors.txt; cp K_28saved.ckpt K.ckpt
  COARSE_TUNE_MIN=19 $N $ARGS --no-tune > K_28c.log 2>&1; rc=$?
  [ $rc = 0 ] && ! grep -q "tune d=" K_28c.log && ! grep -q "tuning again" K_28c.log && grep -q "^d=21 logs (RR, round-5 loop order" K_28c.log && yes_ "K28 with --no-tune round 9's choice is kept" || no_ "K28 --no-tune (rc=$rc)"
  ck2 K $REF "K28 (kept) == reference"
  # an untuned degree (below COARSE_TUNE_MIN) of a round-9 checkpoint: no tuning, no note
  clean
  COARSE_TEST_STOP_AFTER_CHUNKS=37 $R9B $ARGS > K_28d.log 2>&1
  $N $ARGS > K_28e.log 2>&1; rc=$?
  [ $rc = 0 ] && ! grep -q "tune d=\|tuning again" K_28e.log && grep -q "^d=22: resuming at chunk 9/32" K_28e.log && yes_ "K28 below COARSE_TUNE_MIN a round-9 degree resumes as before" || no_ "K28 untuned (rc=$rc)"
  ck2 K $REF "K28 (untuned) == reference"
else no_ "K28 no round-9 binary at $R9B"; fi

# K29: the record -- the trailer after best[] -- names this round and GPU 0;
# the same binary on the same GPU keeps the choice (as K8), another GPU
# (test hook) tunes again, and its checkpoints then record that GPU
clean
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=20 $N $ARGS > K_29a.log 2>&1
python3 - K.ckpt <<'PYEOF' && yes_ "K29 the checkpoint's trailer: 'round 10 on emulated GPU (sm_75, 4 SMs)'" || no_ "K29 trailer"
import sys
b = open(sys.argv[1], 'rb').read()
i = b.rfind(b'GF2SVFP1')
n = int.from_bytes(b[i + 8:i + 12], 'little')
sys.exit(0 if i > 0 and b[i + 12:i + 12 + n] == b'round 10 on emulated GPU (sm_75, 4 SMs)' and i + 12 + n == len(b) else 1)
PYEOF
cp K.ckpt K_29saved.ckpt
COARSE_TEST_GPU_NAME="NVIDIA Tesla T4" COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=3 COARSE_TEST_STOP_AFTER_CHUNKS=4 $N $ARGS > K_29b.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "^d=21: the checkpoint's choice, RR r5-order, was made by round 10 on emulated GPU (sm_75, 4 SMs); this is round 10 on NVIDIA Tesla T4 (sm_75, 4 SMs): tuning again (--no-tune keeps it)$" K_29b.log && grep -q "tune d=21.* -> RR grouped\]" K_29b.log && grep -q "checkpoint: d=21 chunk 12/16" K_29b.log && yes_ "K29 another GPU: tuned again (pick 3: RR grouped), stopped at chunk 12/16" || no_ "K29 other GPU (rc=$rc)"
COARSE_TEST_GPU_NAME="NVIDIA Tesla T4" COARSE_TUNE_MIN=19 $N $ARGS > K_29c.log 2>&1; rc=$?
[ $rc = 0 ] && ! grep -q "tune d=21\|tuning again" K_29c.log && grep -q "^d=21: resuming at chunk 12/16" K_29c.log && grep -q "^d=21 logs (RR, grouped loop order" K_29c.log && yes_ "K29 ... whose checkpoint then records it: kept on that GPU" || no_ "K29 record updated (rc=$rc)"
ck2 K $REF "K29 == reference"

# K30: --retune tunes a resumed degree even when the record matches;
# --retune with --no-tune (or COARSE_TUNE=0) is refused
clean
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=20 $N $ARGS > K_30a.log 2>&1
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=3 COARSE_TUNE_REPS=1 $N $ARGS --retune > K_30b.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "^d=21: --retune: tuning again (the checkpoint's choice: RR r5-order)$" K_30b.log && grep -q "tune d=21.* -> RR grouped\]" K_30b.log && grep -q "^d=21 logs (RR, grouped loop order" K_30b.log && yes_ "K30 --retune: the resumed d=21 tuned again" || no_ "K30 --retune (rc=$rc)"
ck2 K $REF "K30 == reference"
$N $ARGS --retune --no-tune > K_30c.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "^--retune conflicts with --no-tune (or COARSE_TUNE=0)$" K_30c.log && yes_ "K30 --retune with --no-tune refused (exit 1)" || no_ "K30 conflict (rc=$rc)"
COARSE_TUNE=0 $N $ARGS --retune > K_30d.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "conflicts with --no-tune" K_30d.log && yes_ "K30 --retune with COARSE_TUNE=0 refused" || no_ "K30 conflict env (rc=$rc)"

# K31: round 9 continues round 10's checkpoints (the trailer is ignored);
# the files round 9 then writes carry none, so round 10 tunes again
if [ -x $R9B ]; then
  clean
  COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=20 $N $ARGS > K_31a.log 2>&1
  COARSE_TUNE_MIN=19 COARSE_TEST_STOP_AFTER_CHUNKS=4 $R9B 136279841 19-23 K --load-survivors base18.survivors.txt > K_31b.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=21: resuming at chunk 8/16" K_31b.log && ! grep -q "tune d=21" K_31b.log && grep -q "checkpoint: d=21 chunk 12/16" K_31b.log && yes_ "K31 round 9 resumed round 10's checkpoint (its choice kept)" || no_ "K31 round 9 (rc=$rc)"
  COARSE_TUNE_MIN=19 COARSE_TUNE_REPS=1 $N $ARGS > K_31c.log 2>&1; rc=$?
  [ $rc = 0 ] && grep -q "^d=21: the checkpoint's choice, RR r5-order, was made by an earlier round (6-9); tuning again" K_31c.log && grep -q "^d=21: resuming at chunk 12/16" K_31c.log && yes_ "K31 ... and round 10 tunes again after round 9's checkpoint" || no_ "K31 back (rc=$rc)"
  ck2 K $REF "K31 == reference"
else no_ "K31 no round-9 binary at $R9B"; fi

# K32: a damaged trailer (cut short) leaves the checkpoint usable, the
# choice of unknown origin: tuned again
clean
COARSE_TUNE_MIN=19 COARSE_TUNE_PICK=1 COARSE_TEST_STOP_AFTER_CHUNKS=20 $N $ARGS > K_32a.log 2>&1
sz=$(stat -c %s K.ckpt); head -c $((sz - 5)) K.ckpt > t && mv t K.ckpt
COARSE_TUNE_MIN=19 COARSE_TUNE_REPS=1 $N $ARGS > K_32b.log 2>&1; rc=$?
[ $rc = 0 ] && ! grep -q "truncated or damaged" K_32b.log && grep -q "^d=21: resuming at chunk 8/16" K_32b.log && grep -q "was made by an earlier round (6-9); tuning again" K_32b.log && yes_ "K32 a cut trailer: checkpoint used, the degree tuned again" || no_ "K32 (rc=$rc)"
ck2 K $REF "K32 == reference"

# K33: --load is --load-survivors (round 10): the same checkpoint serves
# both spellings, and the refusal names --load
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N 136279841 19-23 K --load base18.survivors.txt > K_33a.log 2>&1
$N 136279841 19-23 K --load-survivors base18.survivors.txt > K_33b.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "^d=22: resuming at chunk 9/32" K_33b.log && yes_ "K33 a run started with --load continues with --load-survivors" || no_ "K33 (rc=$rc)"
ck2 K $REF "K33 == reference"
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N 136279841 19-23 K --load base18.survivors.txt > K_33c.log 2>&1
$N 136279841 19-23 K > K_33d.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "degrees 19-23: a run starting at degree 19 needs --load <file> with the survivors of a run through degree 18" K_33d.log && yes_ "K33 a seeded range without --load: refused, names --load" || no_ "K33 needs --load (rc=$rc)"
$N 136279841 19-24 K --load base18.survivors.txt > K_33e.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "belongs to a different run: it was written by a run of degrees 19-23" K_33e.log && yes_ "K33 refusal of another range (with --load)" || no_ "K33 range (rc=$rc)"
rm -f K.*

# K34: polling (COARSE_SYNC=poll; under WSL2 the default) with one GPU
# queues rounds of up to 8 chunks: stops land on round boundaries, a stop
# written in rounds of one continues in rounds of 8 and back, same results
clean
COARSE_SYNC=poll COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_34a.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "checkpoint: d=22 chunk 16/32" K_34a.log && yes_ "K34 polling: the stop after 37 chunks lands at d=22 chunk 16/32 (rounds of 8)" || no_ "K34 stop (rc=$rc)"
COARSE_TEST_STOP_AFTER_CHUNKS=3 $N $ARGS > K_34b.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "^d=22: resuming at chunk 16/32" K_34b.log && grep -q "checkpoint: d=22 chunk 19/32" K_34b.log && yes_ "K34 continued blocking, in rounds of one: stop at 19/32" || no_ "K34 block (rc=$rc)"
COARSE_TEST_OSRELEASE=5.15.167.4-microsoft-standard-WSL2 $N $ARGS > K_34c.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "^WSL2: chunks are waited for by polling their events (COARSE_SYNC=block: in the driver)$" K_34c.log && grep -q "^d=22: resuming at chunk 19/32" K_34c.log && yes_ "K34 under (pretend) WSL2 polling by default, resumed at 19/32" || no_ "K34 wsl (rc=$rc)"
ck2 K $REF "K34 == reference"

# L1 (round 8): a lock file left by a killed run is noted as harmless; the
# run proceeds, and its clean exit removes the file
clean
touch K.ckpt.lock
$N $ARGS > K_L1.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "note: K.ckpt.lock was left by an earlier run that did not exit normally (killed or crashed); harmless -- its lock ended with it" K_L1.log && [ ! -e K.ckpt.lock ] && yes_ "L1 leftover lock file: noted, run completes, file removed" || no_ "L1 (rc=$rc)"
ck2 K $REF "L1 == reference"

# L2: --fresh with a leftover lock file sets the checkpoint aside, keeps the lock
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_L2a.log 2>&1
touch K.ckpt.lock
$N $ARGS --fresh > K_L2b.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "harmless -- its lock ended with it" K_L2b.log && grep -q "set 1 earlier checkpoint file aside" K_L2b.log && [ ! -e K.ckpt.lock ] && yes_ "L2 --fresh with a leftover lock file" || no_ "L2 (rc=$rc)"
ck2 K $REF "L2 == reference"

# L3: a run killed by a second SIGINT removes its lock file (round 7 left it);
# here it hangs first (the test hook), as a run stuck in the driver would,
# so the first SIGINT cannot stop it
clean
COARSE_TEST_HANG_AFTER_CHUNKS=10 $N $ARGS --watchdog-mins 0 > K_L3.log 2>&1 & pid=$!
for i in $(seq 1 600); do grep -q "hanging after" K_L3.log && break; sleep 0.05; done
sleep 1.5
alive1=0; kill -0 $pid 2>/dev/null && alive1=1        # --watchdog-mins 0: still there
kill -INT $pid; sleep 0.7
alive2=0; kill -0 $pid 2>/dev/null && alive2=1        # the first SIGINT: no chunk boundary comes
lock2=0; [ -e K.ckpt.lock ] && lock2=1
kill -INT $pid; wait $pid; rc=$?
[ $alive1 = 1 ] && [ $alive2 = 1 ] && [ $lock2 = 1 ] && [ $rc = 130 ] && [ ! -e K.ckpt.lock ] && yes_ "L3 hung run: --watchdog-mins 0 keeps it, the first SIGINT can't stop it, the second kills it (130) and removes the lock" || no_ "L3 (alive $alive1 $alive2, lock $lock2, rc=$rc)"

# W1: the watchdog ends a hung run with status 3 (lock removed, the last
# checkpoint intact); the same command continues from that checkpoint
clean
COARSE_TEST_STOP_AFTER_CHUNKS=37 $N $ARGS > K_W1a.log 2>&1
COARSE_TEST_WATCHDOG_SECS=2 COARSE_TEST_HANG_AFTER_CHUNKS=12 $N $ARGS > K_W1b.log 2>&1; rc=$?
[ $rc = 3 ] && grep -q "watchdog: no progress for [0-9]* seconds -- the GPU or its driver seems to hang; exiting with status 3.  The last checkpoint (K.ckpt) is intact: the same command continues from it." K_W1b.log && [ ! -e K.ckpt.lock ] && [ -s K.ckpt ] && grep -q "SIGTERM/SIGINT saves and exits (twice: exit now); watchdog 2 s$" K_W1b.log && yes_ "W1 watchdog: exit 3 after 2 s without progress, lock removed, checkpoint kept" || no_ "W1 watchdog (rc=$rc)"
$N $ARGS > K_W1c.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "^d=22: resuming at chunk 9/32" K_W1c.log && yes_ "W1 the same command continues from the checkpoint" || no_ "W1 resume (rc=$rc)"
ck2 K $REF "W1 == reference"

# W2: without a checkpoint the watchdog says the work is lost
clean
COARSE_TEST_WATCHDOG_SECS=2 COARSE_TEST_HANG_AFTER_CHUNKS=3 $N $ARGS --no-checkpoint > K_W2.log 2>&1; rc=$?
[ $rc = 3 ] && grep -q "exiting with status 3.  This run had no checkpoint, so its work is lost." K_W2.log && yes_ "W2 watchdog without a checkpoint: exit 3, said so" || no_ "W2 (rc=$rc)"

# W3: a run that progresses is never stopped by a short watchdog (the
# emulated chunks take well under a second; 4 s limit)
clean
COARSE_TEST_WATCHDOG_SECS=4 $N $ARGS > K_W3.log 2>&1; rc=$?
[ $rc = 0 ] && ! grep -q "watchdog: no progress" K_W3.log && yes_ "W3 a progressing run under a 4 s watchdog completes" || no_ "W3 (rc=$rc)"
ck2 K $REF "W3 == reference"

[ $fail = 0 ] && echo "ALL CHECKPOINT TESTS PASS" || echo "CHECKPOINT TEST FAILURES"

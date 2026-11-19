#!/bin/bash
# round-9 regression (emulated build): unit tests, selftests, sieve outputs
# against oracles made by round 5's dense and bucketed paths (oracles/O*:
# fixtures/gen_oracles.sh), the checkpoint, tuner, watchdog and command-line
# behaviour.  FULL=0 skips the production-range (s <= floor(r/2))
# comparisons, which take a few minutes (and the two big oracles).
# ~45 min on 2 cores.  Run setup.sh once first.
# Round 9 adds the cubic tower (t_cubic; COARSE_METHOD=cubic against every
# oracle that has a degree 6 | d; the tuner's cubic candidates; its fault
# injection, 2-GPU and phase-timing runs), round 9's loop order in place of
# round 6's, the grouped order's fills, COARSE_PREFETCH=0 and COARSE_SYNC.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
S=$ROOT/src/r9; B=$ROOT/build/r9; OR=$ROOT/oracles; R9=$ROOT/work
mkdir -p $B $R9 && cd $S || exit 1
g++ -O2 -march=native -o $B/t_arith t_arith.cpp && $B/t_arith | tail -1 || exit 1
g++ -O2 -march=native -o $B/t_rot t_rot.cpp && $B/t_rot | tail -1 || exit 1
g++ -O2 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -o $B/t_bs t_bs.cpp && $B/t_bs | tail -1 || exit 1
g++ -O2 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -o $B/t_tower t_tower.cpp && $B/t_tower 30 4 30 | tail -1 || exit 1
g++ -O2 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -o $B/t_pair t_pair.cpp && $B/t_pair | tail -1 || exit 1
g++ -O2 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -o $B/t_cubic t_cubic.cpp && $B/t_cubic 54 3000 | tail -1 || exit 1
W=$(bash $ROOT/tests/build_emul.sh $S $B cs 2>&1 | grep -E "error|warning"); [ -n "$W" ] && { echo "$W"; exit 1; }
W=$(EXTRA=-DCOARSE_PHASE_TIMING bash $ROOT/tests/build_emul.sh $S $B csprof 2>&1 | grep -E "error|warning"); [ -n "$W" ] && { echo "$W"; exit 1; }
N=$B/cs; NP=$B/csprof
fail=0
ck()  { if cmp -s "$1" "$2"; then echo "PASS  $3"; else echo "FAIL  $3"; fail=1; fi; }
ck2() { if cmp -s "$1.found.cert" "$2.found.cert" && cmp -s "$1.survivors.txt" "$2.survivors.txt"; then echo "PASS  $3"; else echo "FAIL  $3"; fail=1; fi; }
yes_() { echo "PASS  $1"; }; no_() { echo "FAIL  $1"; fail=1; }

for e in "" tower ph rr cubic; do
  $N --selftest $e > $R9/st_$e.log 2>&1; grep -q "^selftest: PASS" $R9/st_$e.log && yes_ "selftest $e" || no_ "selftest $e"
done
grep -q "batched: [0-9]* classes (3 loop orders) and [0-9]* (4 fills), ok" $R9/st_tower.log && grep -q "RR    logs.*batched: [0-9]* classes (3 loop orders)" $R9/st_rr.log && grep -q "^  \[d=24 cubic logs.*batched: [0-9]* classes (3 loop orders) and [0-9]* (4 fills), ok" $R9/st_cubic.log && yes_ "selftest: the batched kernel in 3 loop orders (tower, RR, cubic)" || no_ "selftest 3 loop orders"
grep -q "^  \[d=30 cubic logs.*0 wrong; batched.*ok; pairs: [0-9]*, ok; over GF(2^10)" $R9/st_.log && grep -q "^  \[d=24 tower logs" $R9/st_.log && yes_ "selftest: the default one checks the cubic tower beside the tower (d=24, 30)" || no_ "selftest cubic beside tower"
COARSE_SELFTEST_LOGS_DMAX=54 $N --selftest cubic > $R9/st_cubic54.log 2>&1; [ "$(grep -c "^  \[d=[2-5][0-9] cubic logs, .* 0 wrong; batched: [0-9]* classes (3 loop orders) and [0-9]* (4 fills), ok; pairs: [0-9]*, ok; over GF" $R9/st_cubic54.log)" = 6 ] && grep -q "^  \[cubic tower for d=54 over GF(2^18): T3 of 262144 logs (2 MB) by PH in [0-9.]*s, verified\]" $R9/st_cubic54.log && grep -q "^selftest: PASS" $R9/st_cubic54.log && yes_ "selftest cubic up to d=54: d = 24..54 in 3 loop orders and 4 fills (T3 of 2^18 logs at d=54)" || no_ "selftest cubic to d=54"

T=$ROOT/work/regress; rm -rf $T; mkdir -p $T && cd $T && cp $ROOT/fixtures/base18.survivors.txt .
export COARSE_TEST_SMAX=999999
# --- sieve outputs against the round-5 dense oracles (s <= 999999) ---
COARSE_LOGS_CHUNK_LOG2=14 $N 136279841 2-22 A --no-checkpoint 2>A.log; ck2 A $OR/OA "A scratch 2-22 (128 chunks at d=22) == round-5 dense"
COARSE_LOGS_CHUNK_LOG2=16 $N 136279841 19-23 B --load-survivors base18.survivors.txt --no-checkpoint 2>B.log; ck2 B $OR/OB "B 19-23 seeded == round-5 dense"
$N 136279841 19 B19 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null
$N 136279841 20-23 B20 --load-survivors B19.survivors.txt --no-checkpoint 2>/dev/null
sort -n B19.found.cert B20.found.cert > B1.found.cert; cp B20.survivors.txt B1.survivors.txt
ck2 B1 $OR/OB "B 19, then 20-23 from its survivors: certs concatenate to the 19-23 run's"
grep -q "covered, but there are" A.log B.log && no_ "class counts" || yes_ "class counts == I(d) at every degree (A, B)"
EMUL_NDEV=2 COARSE_LOGS_CHUNK_LOG2=15 $N 136279841 19-23 E --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 E $OR/OB "E 19-23 on 2 emulated GPUs"
rm -f H.*; COARSE_TEST_FAULT_EVERY=3 COARSE_LOGS_CHUNK_LOG2=16 $N 136279841 19-23 H --load-survivors base18.survivors.txt > H.log 2>&1; rc=$?
if [ $rc != 0 ] && grep -q "discrete-log consistency errors" H.log && [ ! -e H.found.cert ]; then yes_ "H injected log faults stop the run (exit $rc, no output)"; else no_ "H fault injection (rc=$rc)"; fi
rm -f HG.*; COARSE_PIPE=2 COARSE_TEST_FAULT_EVERY=3 COARSE_LOGS_CHUNK_LOG2=16 $N 136279841 19-23 HG --load-survivors base18.survivors.txt > HG.log 2>&1; rc=$?
if [ $rc != 0 ] && grep -q "discrete-log consistency errors" HG.log && [ ! -e HG.found.cert ]; then yes_ "H injected log faults stop the grouped order too (exit $rc, no output)"; else no_ "H fault injection, grouped (rc=$rc)"; fi
rm -f H9.*; COARSE_PIPE=1 COARSE_METHOD=rr COARSE_TEST_FAULT_EVERY=3 COARSE_LOGS_CHUNK_LOG2=16 $N 136279841 19-23 H9 --load-survivors base18.survivors.txt > H9.log 2>&1; rc=$?
if [ $rc != 0 ] && grep -q "discrete-log consistency errors" H9.log && [ ! -e H9.found.cert ]; then yes_ "H injected log faults stop round 9's order too (RR; exit $rc, no output)"; else no_ "H fault injection, round 9's order (rc=$rc)"; fi
rm -f HC.*; COARSE_METHOD=cubic COARSE_TEST_FAULT_EVERY=3 $N 136279841 24 HC --load-survivors $OR/OB.survivors.txt > HC.log 2>&1; rc=$?
if [ $rc != 0 ] && grep -q "discrete-log consistency errors" HC.log && grep -q "^d=24 logs (cubic tower" HC.log && [ ! -e HC.found.cert ]; then yes_ "H injected log faults stop the cubic tower (d=24; exit $rc, no output)"; else no_ "H fault injection, cubic (rc=$rc)"; fi
COARSE_METHOD=rr COARSE_LOGS_CHUNK_LOG2=16 $N 136279841 19-23 R1 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 R1 $OR/OB "R RR (odd d) + tower, 19-23 == round-5 dense"
COARSE_METHOD=ph $N 136279841 19-23 R2 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 R2 $OR/OB "R PH (odd d) + tower, 19-23 == round-5 dense"
# every loop order (round 9: 0 = round 5's, the default; 1 = round 9's; 2 = grouped)
for o in 0 1 2; do
  COARSE_PIPE=$o COARSE_LOGS_CHUNK_LOG2=16 $N 136279841 19-23 P$o --load-survivors base18.survivors.txt --no-checkpoint 2>P$o.log; ck2 P$o $OR/OB "P loop order $o: tower + PH, 19-23 == round-5 dense"
  COARSE_PIPE=$o COARSE_METHOD=rr $N 136279841 19-23 PR$o --load-survivors base18.survivors.txt --no-checkpoint 2>PR$o.log; ck2 PR$o $OR/OB "P loop order $o: tower + RR, 19-23 == round-5 dense"
  COARSE_PIPE=$o COARSE_METHOD=cubic COARSE_LOGS_CHUNK_LOG2=14 $N 136279841 2-22 PC$o --no-checkpoint 2>PC$o.log; ck2 PC$o $OR/OA "P loop order $o: cubic tower (d=12, 18), scratch 2-22 == round-5 dense"
done
grep -q "^d=20 logs (tower over GF(2^5), grouped loop order" P2.log && grep -q "^d=21 logs (RR, grouped loop order" PR2.log && grep -q "^d=20 logs (tower over GF(2^5), round-5 loop order" P0.log && grep -q "^d=20 logs (tower over GF(2^5), round-9 loop order" P1.log && grep -q "^d=18 logs (cubic tower over GF(2^6), round-9 loop order" PC1.log && grep -q "^d=12 logs (cubic tower over GF(2^4), grouped loop order" PC2.log && yes_ "P the degree lines name the engine and the loop order" || no_ "P loop order names"
grep -q "^  \[cubic tower for d=18 over GF(2^6): T3 of 64 logs (0 MB) by PH in [0-9.]*s, verified\]" PC0.log && yes_ "P the cubic tower's T3 line (d=18: 64 logs)" || no_ "P T3 line"
COARSE_PIPE=2 EMUL_NDEV=2 COARSE_METHOD=rr $N 136279841 19-23 PG2 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 PG2 $OR/OB "P grouped order, tower + RR on 2 emulated GPUs, 19-23 == round-5 dense"
COARSE_PIPE=3 $N 136279841 19-23 X --load-survivors base18.survivors.txt > P3.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "COARSE_PIPE=3: expected 0 (round 5's loop order, the default), 1 (round 9's) or 2 (grouped)" P3.log && yes_ "P COARSE_PIPE=3 refused" || no_ "P COARSE_PIPE=3 (rc=$rc)"
COARSE_METHOD=cubic $N 136279841 19-23 X --load-survivors base18.survivors.txt > PCX.log 2>&1; ck2 X $OR/OB "P COARSE_METHOD=cubic where no degree has 6 | d: the usual engines, == round-5 dense"
# round 7: every way of filling the batches gives the same sieve (the
# default, 3, is in every other test); rounds 8-9: also with the grouped
# order (round 9 parks the fill's state across its batches)
for f in 0 1 2; do COARSE_FILL=$f COARSE_LOGS_CHUNK_LOG2=16 $N 136279841 19-23 Q$f --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 Q$f $OR/OB "Q fill $f: tower + PH, 19-23 == round-5 dense"; done
COARSE_FILL=0 COARSE_METHOD=rr $N 136279841 19-23 QR0 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 QR0 $OR/OB "Q fill 0: tower + RR, 19-23 == round-5 dense"
COARSE_FILL=2 COARSE_METHOD=rr EMUL_NDEV=2 $N 136279841 19-23 QR2 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 QR2 $OR/OB "Q fill 2: tower + RR on 2 emulated GPUs, 19-23 == round-5 dense"
for f in 0 1 2; do COARSE_FILL=$f COARSE_PIPE=2 COARSE_METHOD=rr $N 136279841 19-23 QG$f --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 QG$f $OR/OB "Q fill $f with the grouped order: tower + RR, 19-23 == round-5 dense"; done
COARSE_FILL=1 COARSE_PIPE=1 COARSE_METHOD=cubic COARSE_LOGS_CHUNK_LOG2=14 $N 136279841 2-22 QC1 --no-checkpoint 2>/dev/null; ck2 QC1 $OR/OA "Q fill 1, round 9's order, cubic tower: scratch 2-22 == round-5 dense"
COARSE_FILL=4 $N 136279841 19-23 X --load-survivors base18.survivors.txt > Q4.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "COARSE_FILL=4: expected 0 (round 6), 1, 2 or 3" Q4.log && yes_ "Q COARSE_FILL=4 refused" || no_ "Q COARSE_FILL=4 (rc=$rc)"
# round 8: the table reads through L1, and TP from global memory, give the same sieve
COARSE_LDCG=0 COARSE_METHOD=rr $N 136279841 19-23 LD0 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 LD0 $OR/OB "LD COARSE_LDCG=0 (tables through L1): tower + RR, 19-23 == round-5 dense"
COARSE_LDCG=0 COARSE_METHOD=cubic COARSE_LOGS_CHUNK_LOG2=14 $N 136279841 2-22 LDC --no-checkpoint 2>/dev/null; ck2 LDC $OR/OA "LD COARSE_LDCG=0: cubic tower (T3 through L1), scratch 2-22 == round-5 dense"
# round 9: no prefetches, and chunks waited for by polling, give the same sieve
COARSE_PREFETCH=0 COARSE_METHOD=rr $N 136279841 19-23 PF0 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 PF0 $OR/OB "PF COARSE_PREFETCH=0: tower + RR, 19-23 == round-5 dense"
COARSE_PREFETCH=0 COARSE_PIPE=2 COARSE_METHOD=cubic COARSE_LOGS_CHUNK_LOG2=14 $N 136279841 2-22 PFC --no-checkpoint 2>/dev/null; ck2 PFC $OR/OA "PF COARSE_PREFETCH=0: grouped order, cubic tower, scratch 2-22 == round-5 dense"
COARSE_SYNC=poll $N 136279841 19-23 SP --load-survivors base18.survivors.txt --no-checkpoint 2>SP.log; ck2 SP $OR/OB "SY COARSE_SYNC=poll, 19-23 == round-5 dense"
grep -q "^COARSE_SYNC=poll: chunks are waited for by polling their events every millisecond$" SP.log && yes_ "SY COARSE_SYNC=poll announced" || no_ "SY poll note"
COARSE_SYNC=spin $N 136279841 19-23 X --load-survivors base18.survivors.txt > SY.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "COARSE_SYNC=spin: expected block (the default) or poll" SY.log && yes_ "SY COARSE_SYNC=spin refused" || no_ "SY bad value (rc=$rc)"
COARSE_TOWER_TP=global $N 136279841 19-23 TPG --load-survivors base18.survivors.txt --no-checkpoint 2>TPG.log; ck2 TPG $OR/OB "TP COARSE_TOWER_TP=global, 19-23 == round-5 dense"
grep -q "^d=20 logs (tower over GF(2^5), round-5 loop order, TP from global memory" TPG.log && yes_ "TP the degree line says TP from global memory" || no_ "TP degree line"
COARSE_TOWER_TP=global COARSE_PIPE=2 $N 136279841 19-23 TPG2 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 TPG2 $OR/OB "TP TP from global, grouped order, 19-23 == round-5 dense"
COARSE_TOWER_TP=both $N 136279841 19-23 X --load-survivors base18.survivors.txt > TPX.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "COARSE_TOWER_TP=both: expected auto, shared or global" TPX.log && yes_ "TP COARSE_TOWER_TP=both refused" || no_ "TP bad value (rc=$rc)"
# round 8: the phase-timing build gives the same sieve and reports every degree
COARSE_LOGS_CHUNK_LOG2=16 $NP 136279841 19-23 PT --load-survivors base18.survivors.txt --no-checkpoint 2>PT.log; ck2 PT $OR/OB "PT phase-timing build, 19-23 == round-5 dense"
[ "$(grep -c "^  \[phase timing d=[0-9]*, warp-cycles per log and lane: fill [0-9]* ([0-9.]*%), g^u [0-9]* ([0-9.]*%), logs [0-9]* ([0-9.]*%), inversion [0-9]* ([0-9.]*%), marking [0-9]* ([0-9.]*%), other [0-9]* ([0-9.]*%); total [0-9]*; [0-9]* logs in [0-9]* warp batches, [0-9.]* of 32 per lane\]$" PT.log)" = 5 ] && yes_ "PT a phase line per degree, in the documented format" || no_ "PT phase lines"
python3 - PT.log <<'PYEOF' && yes_ "PT logs counted == classes - partners (I(d) checked), shares sum to 100%" || no_ "PT counts"
import re, sys
t = open(sys.argv[1]).read()
ok = True
for m in re.finditer(r'phase timing d=(\d+).*?\(([0-9.]+)%\).*?\(([0-9.]+)%\).*?\(([0-9.]+)%\).*?\(([0-9.]+)%\).*?\(([0-9.]+)%\).*?\(([0-9.]+)%\); total \d+; (\d+) logs', t):
    sh = sum(float(m.group(i)) for i in range(2, 8))
    if abs(sh - 100.0) > 0.4: ok = False
    if int(m.group(8)) <= 0: ok = False
sys.exit(0 if ok else 1)
PYEOF
COARSE_PIPE=2 COARSE_METHOD=rr COARSE_LOGS_CHUNK_LOG2=16 $NP 136279841 19-23 PTG --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 PTG $OR/OB "PT phase-timing build, grouped order, tower + RR, 19-23 == round-5 dense"
COARSE_PIPE=1 COARSE_METHOD=cubic COARSE_LOGS_CHUNK_LOG2=14 $NP 136279841 2-22 PTC --no-checkpoint 2>PTC.log; ck2 PTC $OR/OA "PT phase-timing build, round 9's order, cubic tower: scratch 2-22 == round-5 dense"
grep -q "^  \[phase timing d=18, " PTC.log && yes_ "PT the cubic degrees report their phases" || no_ "PT cubic phase line"
COARSE_LOGS_CHUNK_LOG2=12 COARSE_TEST_STOP_AFTER_CHUNKS=40 $NP 136279841 21 PTS --load-survivors base18.survivors.txt --no-checkpoint > PTS.log 2>&1; grep -q "phase timing d=21" PTS.log && grep -q "stopping after 40 chunks" PTS.log && yes_ "PT a test stop still reports the phases" || no_ "PT stop report"
# every inner subfield gives the same sieve (d=24: m=12, n = 2, 3, 4, 6, classic), and so does the cubic tower
for n in 2 3 4 6 0; do COARSE_TOWER_SUBN=$n $N 136279841 24 N$n --load-survivors $OR/OB.survivors.txt --no-checkpoint 2>/dev/null; done
for n in 2 3 4 0; do ck2 N$n N6 "N d=24 inner $( [ $n = 0 ] && echo classic || echo "GF(2^$n)") == GF(2^6)"; done
COARSE_METHOD=cubic $N 136279841 24 NC --load-survivors $OR/OB.survivors.txt --no-checkpoint 2>/dev/null; ck2 NC N6 "N d=24 cubic tower over GF(2^8) == tower over GF(2^6)"
# the auto-tuner (forced to small degrees) changes nothing; nor does any
# pick.  Candidates (round 9): each tower/cubic/RR variant in the three
# loop orders r5 (the default), r9, grouped: d=18 (m=9) GF(2^3), classic,
# cubic GF(2^6); d=20 (m=10) GF(2^5), GF(2^2), classic; d=24 (m=12)
# GF(2^6), GF(2^4), classic, cubic GF(2^8); d=22, 26 (m prime) classic;
# odd d PH (alone), RR
$N 136279841 2-17 S17 --no-checkpoint 2>/dev/null
COARSE_TUNE=0 $N 136279841 18-26 T0 --load-survivors S17.survivors.txt --no-checkpoint 2>/dev/null
for k in 0 1 2 3 4 5 6 9 11; do COARSE_TUNE_PICK=$k COARSE_TUNE_MIN=18 $N 136279841 18-26 T$((k+1)) --load-survivors S17.survivors.txt --no-checkpoint 2> T$((k+1)).log; ck2 T$((k+1)) T0 "T tuner pick $k (d=18..26) == untuned"; done
L18=$(grep "tune d=18 " T1.log)
[ "$(echo "$L18" | grep -o "ns/class" | wc -l)" = 9 ] && echo "$L18" | grep -q "GF(2^3) r5-order [0-9.]* ns/class, GF(2^3) r9-order [0-9.]* ns/class, GF(2^3) grouped [0-9.]* ns/class, classic r5-order [0-9.]* ns/class, classic r9-order [0-9.]* ns/class, classic grouped [0-9.]* ns/class, cubic GF(2^6) r5-order [0-9.]* ns/class, cubic GF(2^6) r9-order [0-9.]* ns/class, cubic GF(2^6) grouped [0-9.]* ns/class -> GF(2^3) r5-order\]" && yes_ "T d=18: nine candidates in order (3 loop orders each, the cubic tower last), no duplicate" || no_ "T d=18 candidates: $L18"
grep -q "tune d=18.* -> GF(2^3) r9-order\]" T2.log && grep -q "tune d=19.*PH .*, RR r5-order .*, RR r9-order .*, RR grouped .* -> RR r5-order\]" T2.log && grep -q "tune d=22.*classic r5-order .*, classic r9-order .*, classic grouped .* -> classic r9-order\]" T2.log && grep -q "^d=22 logs (tower, classic, round-9 loop order" T2.log && yes_ "T tuner pick 1 applied (incl. round 9's loop order)" || no_ "T tuner pick 1"
grep -q "tune d=18.* -> GF(2^3) grouped\]" T3.log && grep -q "tune d=19.* -> RR r9-order\]" T3.log && grep -q "^d=18 logs (tower over GF(2^3), grouped loop order" T3.log && grep -q "tune d=20.* -> GF(2^2) r5-order\]" T4.log && grep -q "tune d=24.* -> GF(2^4) r9-order\]" T5.log && grep -q "tune d=24.*GF(2^6) r5-order .*, GF(2^6) r9-order .*, GF(2^6) grouped .*, GF(2^4) r5-order .*, GF(2^4) r9-order .*, GF(2^4) grouped .*, classic r5-order .*, classic r9-order .*, classic grouped .*, cubic GF(2^8) r5-order .*, cubic GF(2^8) r9-order .*, cubic GF(2^8) grouped .* -> GF(2^4) grouped\]" T6.log && yes_ "T tuner picks 2-5 applied" || no_ "T tuner picks 2-5"
grep -q "tune d=18.* -> cubic GF(2^6) r5-order\]" T7.log && grep -q "^d=18 logs (cubic tower over GF(2^6), round-5 loop order" T7.log && grep -q "tune d=24.* -> classic r5-order\]" T7.log && grep -q "tune d=24.* -> cubic GF(2^8) r5-order\]" T10.log && grep -q "^d=24 logs (cubic tower over GF(2^8), round-5 loop order" T10.log && grep -q "tune d=24.* -> cubic GF(2^8) grouped\]" T12.log && grep -q "^d=24 logs (cubic tower over GF(2^8), grouped loop order" T12.log && yes_ "T the cubic tower's picks applied (d=18 pick 6, d=24 picks 9 and 11)" || no_ "T cubic picks"
COARSE_PIPE=1 COARSE_TUNE_MIN=18 $N 136279841 18 T7p --load-survivors S17.survivors.txt --no-checkpoint 2> T7p.log
[ "$(grep "tune d=18 " T7p.log | grep -o "ns/class" | wc -l)" = 3 ] && grep -q "tune d=18.*GF(2^3) r9-order .*, classic r9-order .*, cubic GF(2^6) r9-order .* -> GF(2^3) r9-order\]" T7p.log && yes_ "T COARSE_PIPE fixes the loop order (one candidate per variant)" || no_ "T COARSE_PIPE"
COARSE_TOWER_SUBN=3 COARSE_TUNE_MIN=18 $N 136279841 18 T7s --load-survivors S17.survivors.txt --no-checkpoint 2> T7s.log
[ "$(grep "tune d=18 " T7s.log | grep -o "ns/class" | wc -l)" = 3 ] && ! grep -q "tune d=18.*cubic" T7s.log && yes_ "T COARSE_TOWER_SUBN fixes the tower: no cubic candidate" || no_ "T COARSE_TOWER_SUBN"
COARSE_METHOD=cubic COARSE_TUNE_MIN=18 $N 136279841 18 T7c --load-survivors S17.survivors.txt --no-checkpoint 2> T7c.log
[ "$(grep "tune d=18 " T7c.log | grep -o "ns/class" | wc -l)" = 3 ] && grep -q "tune d=18.*cubic GF(2^6) r5-order .*, cubic GF(2^6) r9-order .*, cubic GF(2^6) grouped .* -> " T7c.log && yes_ "T COARSE_METHOD=cubic: the cubic tower in its 3 loop orders only" || no_ "T COARSE_METHOD=cubic"
# round 8: the TP-global candidates (forced by the test hook at these small d)
COARSE_TEST_TP_CHOICE=1 COARSE_TUNE_MIN=18 $N 136279841 18-26 T8 --load-survivors S17.survivors.txt --no-checkpoint 2> T8.log; ck2 T8 T0 "T TP-global candidates offered (test hook): == untuned"
grep -q "tune d=20.*GF(2^5) r5-order .*, GF(2^5) r9-order .*, GF(2^5) grouped .*, GF(2^5) r5-order TP-global .*, GF(2^5) r9-order TP-global .*, GF(2^5) grouped TP-global .*, GF(2^2) r5-order .*, classic grouped .* ns/class -> " T8.log && yes_ "T TP-global twins follow their variants, in the 3 loop orders" || no_ "T TP-global candidates"
for k in 3 5; do COARSE_TEST_TP_CHOICE=1 COARSE_TUNE_PICK=$k COARSE_TUNE_MIN=18 $N 136279841 18-26 T9_$k --load-survivors S17.survivors.txt --no-checkpoint 2> T9_$k.log; ck2 T9_$k T0 "T TP-global pick $k == untuned"; done
grep -q "^d=20 logs (tower over GF(2^5), round-5 loop order, TP from global memory" T9_3.log && grep -q "^d=20 logs (tower over GF(2^5), grouped loop order, TP from global memory" T9_5.log && yes_ "T TP-global picks applied" || no_ "T TP-global picks"
# --- 24..29 and 30..33 at s <= 299999 against round 5's dense/bucketed paths ---
export COARSE_TEST_SMAX=299999
$N 136279841 24-29 J --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 J $OR/OJ "J 24-29 == round-5 dense/buckets"
COARSE_METHOD=rr $N 136279841 24-29 JR --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JR $OR/OJ "J RR 24-29 == round-5 dense/buckets"
COARSE_PIPE=1 COARSE_METHOD=rr $N 136279841 24-29 JP --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JP $OR/OJ "J RR, round-9 loop order, 24-29 == round-5 dense/buckets"
COARSE_PIPE=2 $N 136279841 24-29 JG --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JG $OR/OJ "J grouped order (tower + PH), 24-29 == round-5 dense/buckets"
COARSE_PIPE=2 COARSE_METHOD=rr $N 136279841 24-29 JGR --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JGR $OR/OJ "J grouped order (tower + RR), 24-29 == round-5 dense/buckets"
COARSE_FILL=0 $N 136279841 24-29 JF0 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JF0 $OR/OJ "J fill 0 (round 6's), 24-29 == round-5 dense/buckets"
COARSE_FILL=1 COARSE_METHOD=rr $N 136279841 24-29 JF1 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JF1 $OR/OJ "J fill 1, RR, 24-29 == round-5 dense/buckets"
for o in 0 1 2; do COARSE_PIPE=$o COARSE_METHOD=cubic $N 136279841 24-29 JC$o --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JC$o $OR/OJ "J cubic tower (d=24), loop order $o, 24-29 == round-5 dense/buckets"; done
COARSE_FILL=0 COARSE_PIPE=2 COARSE_METHOD=cubic $N 136279841 24-29 JCF --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JCF $OR/OJ "J cubic tower, fill 0, grouped order, 24-29 == round-5 dense/buckets"
EMUL_NDEV=2 COARSE_PIPE=1 COARSE_METHOD=cubic $N 136279841 24-29 JC2 --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 JC2 $OR/OJ "J cubic tower on 2 emulated GPUs, 24-29 == round-5 dense/buckets"
EMUL_FREE_MB=1900 $N 136279841 30-33 G --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 G $OR/OBIG "G 30-33 == round-5 buckets"
EMUL_FREE_MB=1900 COARSE_METHOD=rr $N 136279841 30-33 GR --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 GR $OR/OBIG "G RR 30-33 == round-5 buckets"
EMUL_FREE_MB=1900 COARSE_PIPE=2 COARSE_METHOD=rr $N 136279841 30-33 GG --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 GG $OR/OBIG "G grouped order, tower + RR, 30-33 == round-5 buckets"
EMUL_FREE_MB=1900 COARSE_METHOD=cubic $N 136279841 30-33 GC --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 GC $OR/OBIG "G cubic tower (d=30), 30-33 == round-5 buckets"
EMUL_FREE_MB=1900 COARSE_PIPE=2 COARSE_METHOD=cubic $N 136279841 30-33 GCG --load-survivors base18.survivors.txt --no-checkpoint 2>/dev/null; ck2 GCG $OR/OBIG "G cubic tower, grouped order, 30-33 == round-5 buckets"
unset COARSE_TEST_SMAX
# --- the production range (s <= floor(r/2)): small degrees, heavy progressions ---
if [ "${FULL:-1}" = 1 ]; then
  $N 136279841 2-28 F28 --no-checkpoint 2>F28.log; ck2 F28 $OR/OF28 "F full range, scratch 2-28 == round-5 dense"
  COARSE_PIPE=2 $N 136279841 2-28 F28G --no-checkpoint 2>/dev/null; ck2 F28G $OR/OF28 "F full range, grouped order, scratch 2-28 == round-5 dense"
  COARSE_PIPE=1 COARSE_METHOD=cubic $N 136279841 2-28 F28C --no-checkpoint 2>/dev/null; ck2 F28C $OR/OF28 "F full range, cubic tower (d=12, 18, 24), round 9's order, scratch 2-28 == round-5 dense"
  EMUL_FREE_MB=1900 $N 136279841 29-32 F32 --load-survivors F28.survivors.txt --no-checkpoint 2>/dev/null; ck2 F32 $OR/OF32 "F full range, 29-32 == round-5 buckets"
  EMUL_FREE_MB=1900 COARSE_PIPE=2 COARSE_METHOD=rr $N 136279841 29-32 F32G --load-survivors F28.survivors.txt --no-checkpoint 2>/dev/null; ck2 F32G $OR/OF32 "F full range, grouped order, tower + RR, 29-32 == round-5 buckets"
  EMUL_FREE_MB=1900 COARSE_METHOD=cubic $N 136279841 29-32 F32C --load-survivors F28.survivors.txt --no-checkpoint 2>/dev/null; ck2 F32C $OR/OF32 "F full range, cubic tower (d=30), 29-32 == round-5 buckets"
fi
export COARSE_TEST_SMAX=999999
# --- command line ---
$N 136279841 47 X > X1.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "needs --load-survivors" X1.log && yes_ "CLI degree 47 without a seed refused" || no_ "CLI no seed (rc=$rc)"
$N 136279841 2-20 X --load-survivors base18.survivors.txt > X2.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "no lower" X2.log && yes_ "CLI seed with a run from 2 refused" || no_ "CLI seed from 2 (rc=$rc)"
bad=0; for dg in 20-19 1-5 56 2-56 abc 20- -20 3x; do $N 136279841 $dg X > X3.log 2>&1; [ $? = 1 ] && grep -q "expected D or D1-D2" X3.log || bad=1; done; [ $bad = 0 ] && yes_ "CLI malformed degree ranges refused" || no_ "CLI degree ranges"
$N 136279841 19-23 X --load-survivors base18.survivors.txt --smax 5 > X4.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "unknown or incomplete option '--smax'" X4.log && yes_ "CLI removed option --smax refused" || no_ "CLI --smax (rc=$rc)"
bad=0; for v in -1 abc 5x ""; do $N 136279841 19-23 X --load-survivors base18.survivors.txt --watchdog-mins "$v" > X5.log 2>&1; [ $? = 1 ] && grep -q "\-\-watchdog-mins $v: expected minutes >= 0" X5.log || bad=1; done; [ $bad = 0 ] && yes_ "CLI bad --watchdog-mins values refused" || no_ "CLI --watchdog-mins"
$N 136279841 19-20 X5 --load-survivors base18.survivors.txt --watchdog-mins 0.5 > X6.log 2>&1; rc=$?; [ $rc = 0 ] && grep -q "SIGTERM/SIGINT saves and exits (twice: exit now); watchdog 30 s$" X6.log && yes_ "CLI --watchdog-mins 0.5 accepted and shown" || no_ "CLI --watchdog-mins 0.5 (rc=$rc)"
$N 136279841 19-20 X7 --load-survivors base18.survivors.txt --watchdog-mins 0 > X7.log 2>&1; rc=$?; [ $rc = 0 ] && grep -q "SIGTERM/SIGINT saves and exits (twice: exit now)$" X7.log && yes_ "CLI --watchdog-mins 0: off, not shown" || no_ "CLI --watchdog-mins 0 (rc=$rc)"
COARSE_METHOD=cube $N 136279841 19-20 X > X8.log 2>&1; rc=$?; [ $rc = 1 ] && grep -q "COARSE_METHOD=cube" X8.log && yes_ "CLI COARSE_METHOD=cube refused" || no_ "CLI COARSE_METHOD=cube (rc=$rc)"
rm -f S.*; COARSE_TEST_STOP_DEGREE=21 $N 136279841 19-23 S --load-survivors base18.survivors.txt > S1.log 2>&1; rc=$?
[ $rc = 2 ] && [ -s S.ckpt ] && [ -s S.survivors.txt ] && grep -q "stopped after degree 20" S1.log && yes_ "S unsieveable degree: exit 2, outputs through 20, checkpoint kept" || no_ "S stop degree (rc=$rc)"
$N 136279841 19-23 S --load-survivors base18.survivors.txt > S2.log 2>&1; grep -q "found checkpoint S.ckpt (written .*start of degree 21)" S2.log && yes_ "S rerun continues at degree 21" || no_ "S rerun"
ck2 S $OR/OB "S continued run == uninterrupted"
# --- checkpoints, the lock and the watchdog ---
bash $ROOT/tests/ckpt_tests.sh > $R9/ckpt.out 2>&1; grep -E "^(PASS|FAIL)" $R9/ckpt.out; grep -q "ALL CHECKPOINT TESTS PASS" $R9/ckpt.out || fail=1
[ $fail = 0 ] && echo "ALL REGRESSIONS PASS" || echo "REGRESSION FAILURES"

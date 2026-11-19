#!/bin/bash
# Checkpoint, resume, signal and compatibility tests.
# usage: tests/ckpt_test.sh path/to/irred path/to/irred020
B=$(readlink -f ${1:-build/irred}); OLD=$(readlink -f ${2:-build/irred020})
W=$(mktemp -d /tmp/irred_ckpt_XXXX); cd $W
R=1398269; S=300001; F=irred_r${R}_s${S}
pass=0; failn=0
ok()  { echo "PASS: $*"; pass=$((pass+1)); }
bad() { echo "FAIL: $*"; failn=$((failn+1)); }
res() { awk '/ LOG$/{print $3}' "$1"; }
# interrupt a run after $1 seconds with signal $2; remaining args: the command
interrupt() { local t=$1 sig=$2; shift 2; "$@" > int.out 2> int.err & local p=$!
              sleep $t; kill -$sig $p; wait $p; echo $?; }

echo "== reference run (no checkpoints)"
$B --no-ckpt --progress 100000 $R $S > ref.out; WANT=$(res ref.out); echo "   result $WANT"
grep '^iter' ref.out | awk '{for(i=1;i<=NF;i++) if($i=="residue") print $2, $(i+1)}' | sort -u > want_prog.txt

echo "== SIGINT, then resume"
rc=$(interrupt 4 INT $B --ckpt-mins 0.02 --progress 100000 $R $S); cp int.out run1.out
[ "$rc" = 2 ] && ok "exit status 2 after SIGINT" || bad "exit status $rc after SIGINT"
grep -q 'finishing the block and verifying' run1.out && ok "signal: block finished and verified first" || bad "no verify-before-stop message"
grep -q 'verified -> ' int.err && ok "checkpoints are of verified states" || bad "checkpoint not marked verified"
[ -f $F.ckpt ] && [ -f $F.ckpt.1 ] && [ -f $F.ckpt.2 ] && ok "three generations" || bad "missing generations: $(ls)"
$B --ckpt-mins 0.02 --progress 100000 $R $S > run2.out 2>/dev/null
grep -q '^Resuming' run2.out && ok "resumed" || bad "did not resume"
[ "$(res run2.out)" = "$WANT" ] && ok "resumed result matches" || bad "resumed result $(res run2.out) != $WANT"
grep -q "All $R squarings were error-checked" run2.out && ok "all squarings checked" || bad "unchecked count wrong"
cat run1.out run2.out | grep '^iter' | awk '{for(i=1;i<=NF;i++) if($i=="residue") print $2, $(i+1)}' | sort -u > got_prog.txt
[ -z "$(comm -23 got_prog.txt want_prog.txt)" ] && ok "progress residues match the uninterrupted run" || bad "progress residues differ"
[ -z "$(ls irred_* 2>/dev/null)" ] && ok "files removed at the end" || bad "left: $(ls irred_*)"

echo "== corrupted newest checkpoint -> falls back to .ckpt.1"
interrupt 4 TERM $B --ckpt-mins 0.02 $R $S >/dev/null
python3 -c "
f=open('$F.ckpt','r+b'); f.seek(5000); b=f.read(1); f.seek(5000); f.write(bytes([b[0]^1])); f.close()"
$B $R $S > c.out 2>/dev/null
grep -q "Ignoring $F.ckpt: data truncated or damaged" c.out && grep -q "Resuming from $F.ckpt.1" c.out \
  && ok "damaged .ckpt skipped, .ckpt.1 used" || bad "fallback"
[ "$(res c.out)" = "$WANT" ] && ok "result after fallback matches" || bad "result after fallback"

echo "== mismatched and foreign files are ignored"
rm -f irred_*
interrupt 3 INT $B --ckpt-mins 0.02 $R 300003 >/dev/null
cp irred_r${R}_s300003.ckpt $F.ckpt; head -c 100000 /dev/urandom > $F.ckpt.1; printf 'TSFC1 not irred' > $F.ckpt.2
interrupt 2 INT $B --ckpt-mins 0 $R $S >/dev/null
grep -q "it is for r = $R, s = 300003" int.out && grep -q "$F.ckpt.1: not an irred" int.out \
  && grep -q "$F.ckpt.2: not an irred" int.out && ! grep -q '^Resuming' int.out \
  && ok "mismatch / foreign files ignored, fresh start" || bad "mismatch handling"
[ $(grep -c 'checkpoint:' int.err) = 1 ] && ok "--ckpt-mins 0: only the signal checkpoint" || bad "--ckpt-mins 0"
rm -f irred_r${R}_s300003*

echo "== lock"
$B $R $S > /dev/null 2>&1 & P=$!; sleep 1
$B $R $S > l.out 2> l.err; lrc=$?
kill -INT $P; wait $P
[ $lrc = 1 ] && grep -q 'already testing' l.err && ok "second process refused" || bad "lock (rc=$lrc)"
rm -f irred_*

echo "== a run started by irred 0.20 (version 1 checkpoint) continues"
interrupt 4 INT $OLD --ckpt-mins 0.02 $R $S >/dev/null
K1=$(grep -o 'after [0-9]*' int.out | awk '{print $2}')
$B --progress 100000 $R $S > v1.out 2>/dev/null
grep -q 'version 1 checkpoint from irred 0.20' v1.out && ok "v1 checkpoint recognised" || bad "v1 not recognised"
[ "$(res v1.out)" = "$WANT" ] && ok "0.20 -> 0.21 result matches" || bad "0.20 -> 0.21 result $(res v1.out)"
grep -q "Note: $K1 of the $R squarings were NOT error-checked" v1.out && ok "reports the $K1 unchecked squarings" || bad "unchecked report: $(grep -i 'checked' v1.out)"

echo "== --no-ec segment, then checked"
rm -f irred_*
interrupt 3 INT $B --no-ec --ckpt-mins 0.02 $R $S >/dev/null
K2=$(grep -o 'after [0-9]*' int.out | awk '{print $2}')
$B $R $S > m.out 2>/dev/null
[ "$(res m.out)" = "$WANT" ] && grep -q "Note: $K2 of the $R squarings were NOT error-checked" m.out \
  && ok "--no-ec squarings counted ($K2)" || bad "mixed run: $(grep -i checked m.out)"

echo "== faults in a checkpointed run"
rm -f irred_*
$B --ckpt-mins 0.01 --ec-fault sq:700000 $R $S > f1.out 2>/dev/null
[ "$(res f1.out)" = "$WANT" ] && grep -q 'ERROR CHECK FAILED' f1.out && ok "sq fault: detected, redone, result right" || bad "sq fault"
$B --ckpt-mins 0.02 --ec-fault ver:100 $R $S > f2.out 2> f2.err
[ "$(res f2.out)" = "$WANT" ] && grep -q -E 'damaged in memory|NOT written' f2.out f2.err \
  && ok "damaged verified state: detected, result right" || bad "ver fault"
grep -h -E 'damaged|NOT written|restarting' f2.out f2.err | sed 's/^/     /'

cd /; rm -rf $W
echo "checkpoint tests: $pass passed, $failn failed"
[ $failn = 0 ]

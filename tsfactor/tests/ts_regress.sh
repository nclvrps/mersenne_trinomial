#!/bin/bash
# ts_regress.sh -- command-line regression for tsfactor's error checking.
#
#   ./ts_regress.sh NEW [ORIG]
#
#   NEW   the tsfactor binary under test (e.g. ./build_emul/tsfactor, or the
#         real ./tsfactor on a GPU)
#   ORIG  optional: a tsfactor built from the sources BEFORE the error-
#         checking round.  If given, it supplies the reference results;
#         otherwise NEW --no-ec does.
#
# Three small exponents (r = 3217, 4423, 9689), a few hundred survivors in
# all, each scanned in eleven configurations ("ec" is the default; at these
# sizes the gcd chain only runs when --gcd-finish-bits is lowered, which the
# "chain" rows do; "mt" checks the option only -- products this small stay on
# one thread, and gcdstress is what exercises the threads).  Every
# configuration must give
# the SAME results file as the reference and log zero detections.  Then:
# options from the environment, eleven injected faults that must be caught
# and recovered from with the same results, a persistent fault that must
# stop the program with status 3, and interrupt-and-resume.
#
# Run it from the directory that holds surv3217.txt, surv4423.txt and
# surv9689.txt (the devkit's tests/).  Under a minute under emulation.

T=${1:?usage: ts_regress.sh NEW_TSFACTOR [ORIG_TSFACTOR]}
ORIG=$2
[ -x "$T" ] || { echo "not executable: $T"; exit 2; }
# round 10: the same script tests the CPU build (tsfactor_cpu); its rows
# that need the GPU multiplier are skipped
CPUBUILD=0
"$T" --version 2>/dev/null | grep -q "CPU build" && CPUBUILD=1
W=$(mktemp -d ts_regress.XXXXXX)
fail=0
note() { echo "$*"; }
bad()  { echo "  *** FAIL: $*"; fail=$((fail + 1)); }

run() {   # run NAME args...   -> $W/NAME.results.txt, .out, .err
    local name=$1; shift
    "$T" "$@" --out "$W/$name" > "$W/$name.out" 2> "$W/$name.err"
}
detections() {   # detections reported in NAME.err ("" when EC is off)
    sed -n 's/^error checking: .* verified; \([0-9]*\) detection.*/\1/p' \
        "$W/$1.err"
}

for R in 3217 4423 9689; do
    S=surv$R.txt
    [ -f "$S" ] || { echo "missing $S"; exit 2; }
    C="$R $S --skip 5 --maxd 600 --m 6 --q0 3 --no-ckpt --gcd-threads 2"
    if [ -n "$ORIG" ]; then
        "$ORIG" $C --out "$W/ref_$R" > "$W/ref_$R.out" 2> "$W/ref_$R.err"
    else
        run ref_$R $C --no-ec
    fi
    n=$(grep -c . "$W/ref_$R.results.txt")
    note "r=$R: $n reference results ($( [ -n "$ORIG" ] && echo original binary || echo --no-ec ))"
    i=0
    while read -r name opts; do
        i=$((i + 1))
        if [ $CPUBUILD = 1 ] && [ $name = gpugcd ]; then
            printf "  %-9s skipped (CPU build)\n" "$name"; continue
        fi
        run ${name}_$R $C $opts
        st=$?
        d=$(detections ${name}_$R)
        if ! cmp -s "$W/ref_$R.results.txt" "$W/${name}_$R.results.txt"; then
            bad "$name r=$R: results differ from the reference"
        elif [ $st -ne 0 ]; then
            bad "$name r=$R: exit status $st"
        elif [ "$name" != noec ] && [ "$d" != 0 ]; then
            bad "$name r=$R: '$d' detections on a clean run"
        else
            printf "  %-9s identical  %s\n" "$name" \
                "$(sed -n 's/^error checking: \([0-9]* blocks.*verified\).*/\1/p' "$W/${name}_$R.err")"
        fi
    done <<EOF
noec      --no-ec
ec
chain     --gcd-finish-bits 256
chainold  --gcd-finish-bits 256 --gcd-full-apply
ecw1      --ecw 1 --pend-max 1 --gcd-finish-bits 256
xgcd      --ecg-xgcd
ntl       --gcd-ntl
dup       --ecg-dup
gpugcd    --gpu-gcd --gpu-gcd-min-bits 512 --gcd-finish-bits 256
opt       --sched opt --sched-G 40 --eci0 --ect --pend-max 1
linear    --sched linear --pend-max 2
mt        --gcd-mt 3 --gcd-finish-bits 256
EOF
done

# ---- environment-variable options (what a wrapper script would use)
R=4423; S=surv$R.txt
C="$R $S --skip 5 --maxd 600 --m 6 --q0 3 --no-ckpt --gcd-threads 2"
TSFACTOR_ARGS="--ecg-xgcd,--ecw,4" run env_$R $C
if cmp -s "$W/ref_$R.results.txt" "$W/env_$R.results.txt" &&
   grep -q "ecg-xgcd" "$W/env_$R.err" &&
   grep -q "chain every 4 block" "$W/env_$R.err"; then
    note "TSFACTOR_ARGS: options taken from the environment, results identical"
else
    bad "TSFACTOR_ARGS"
fi

# ---- injected faults: detected, recovered, same results
caught() {   # detections + transfer re-reads + in-place repairs in NAME.err
    sed -n 's/^error checking: .* verified; \([0-9]*\) detection(s), [0-9]* restart(s), \([0-9]*\) in-place repair(s), \([0-9]*\) transfer re-read(s).*/\1 \2 \3/p' \
        "$W/$1.err" | awk '{print $1 + $2 + $3}'
}
for f in h:40 hmid:25 t:60 red:33 a:17 c:21 init:2 idle:4 xferh:2 xfera:5 snap:2; do
    site=${f%%:*}
    name=fault_$site
    run $name $C --pend-max 1 --ec-fault $f
    st=$?
    d=$(caught $name)
    inj=$(grep -c "EC INJECT" "$W/$name.err")
    if ! cmp -s "$W/ref_$R.results.txt" "$W/$name.results.txt"; then
        bad "--ec-fault $f: WRONG RESULTS"
    elif [ $st -ne 0 ]; then
        bad "--ec-fault $f: exit status $st"
    elif [ "$inj" = 0 ]; then
        note "  --ec-fault $f: site not reached in this run (no fault injected)"
    elif [ -n "$d" ] && [ "$d" != 0 ]; then
        printf "  --ec-fault %-9s injected, caught (%s), results identical\n" "$f" \
            "$(grep -m1 -o 'EC FAIL \[[^]]*\]' "$W/$name.err" || echo 're-read')"
    elif [ $site = idle ] || [ $site = snap ]; then
        # a flip in a table or snapshot that is never read again (its s was
        # resolved by the gcd it was waiting for) is harmless and unseen
        printf "  --ec-fault %-9s injected into state that was never read again: harmless\n" "$f"
    else
        bad "--ec-fault $f: injected but NOT detected"
    fi
done

# the same flip with checking off: shows the checker is what saved it
wrong=0
for n in 3 9 40 77; do
    "$T" $C --pend-max 1 --ec-fault h:$n --no-ec --out "$W/unchk$n" \
        > "$W/unchk$n.out" 2> "$W/unchk$n.err"
    cmp -s "$W/ref_$R.results.txt" "$W/unchk$n.results.txt" || wrong=$((wrong + 1))
done
note "  the same kind of flip with --no-ec (4 runs): $wrong gave wrong results, silently"

# ---- a persistent fault must stop with status 3 and no bogus result
"$T" $R $S --skip 5 --maxd 600 --m 6 --q0 3 --no-ckpt --gcd-threads 2 \
    --pend-max 1 --ec-fault h:-1 --out "$W/fatal" > "$W/fatal.out" 2> "$W/fatal.err"
st=$?
if [ $st -eq 3 ] && ! grep -v -x -f "$W/ref_$R.results.txt" "$W/fatal.results.txt" | grep -q .; then
    note "  persistent fault: exit status 3, no wrong line written"
else
    bad "persistent fault: exit status $st (want 3) or a wrong result line"
fi

# ---- interrupt and resume with checking on (--die-after-blocks N is the
#      program's own test hook: it behaves like SIGTERM after N blocks)
R=9689; S=surv$R.txt
C="$R $S --skip 5 --maxd 2500 --m 6 --q0 3 --gcd-threads 1 --pend-max 1 --ckpt-mins 0.002 -v"
if [ -n "$ORIG" ]; then
    "$ORIG" $C --no-ckpt --out "$W/kref" > "$W/kref.out" 2> "$W/kref.err"
else
    "$T" $C --no-ec --no-ckpt --out "$W/kref" > "$W/kref.out" 2> "$W/kref.err"
fi
for n in 400 700 150; do
    "$T" $C --out "$W/kill" --die-after-blocks $n >> "$W/kill.out" 2>> "$W/kill.err"
done
"$T" $C --out "$W/kill" >> "$W/kill.out" 2>> "$W/kill.err"
st=$?
nres=$(grep -c "^Resumed" "$W/kill.out")
left=$(ls "$W"/kill_r*.ckpt* 2>/dev/null | wc -l)
if cmp -s "$W/kref.results.txt" "$W/kill.results.txt" && [ $st -eq 0 ] && [ "$nres" -ge 1 ] && [ "$left" -eq 0 ]; then
    note "interrupt/resume: stopped 3 times, $nres mid-interval resume(s) from checkpoint, results identical, no checkpoint file left behind"
elif [ "$left" -ne 0 ]; then
    bad "interrupt/resume: $left checkpoint file(s) left behind after the list was finished"
else
    bad "interrupt/resume: status $st, $nres resume(s), results $(cmp -s "$W/kref.results.txt" "$W/kill.results.txt" && echo identical || echo DIFFER)"
fi

# ---- a fault while resuming: the checkpointed state is re-verified
"$T" $C --out "$W/kf" --die-after-blocks 400 > "$W/kf.out" 2> "$W/kf.err"
"$T" $C --out "$W/kf" --ec-fault h:30 >> "$W/kf.out" 2>> "$W/kf.err"
if cmp -s "$W/kref.results.txt" "$W/kf.results.txt" && grep -q "EC FAIL" "$W/kf.err"; then
    note "fault after a resume: detected, results identical"
else
    bad "fault after a resume"
fi

echo
if [ $fail -eq 0 ]; then
    echo "ts_regress: all PASS"
    rm -rf "$W"
else
    echo "ts_regress: $fail FAILURE(S); logs kept in $W"
fi
exit $fail

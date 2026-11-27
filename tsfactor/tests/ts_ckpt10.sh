#!/bin/bash
# ts_ckpt10.sh -- round 10's per-survivor checkpoints, start-up tuning,
# output-prefix lock and saved schedule costs, end to end.
#
#   ./ts_ckpt10.sh TSFACTOR [OTHER_BUILD]
#
#   TSFACTOR     tsfactor or tsfactor_cpu (either build)
#   OTHER_BUILD  optional: the other build.  Then a checkpoint written by
#                one is resumed by the other (the tuning fingerprint
#                differs, so it must tune again; the state carries over).
#
# Run it from the devkit's tests/ directory (it needs surv9689.txt).
# Well under a minute under emulation.

T=${1:?usage: ts_ckpt10.sh TSFACTOR [OTHER_BUILD]}
T2=$2
[ -x "$T" ] || { echo "not executable: $T"; exit 2; }
[ -z "$T2" ] || [ -x "$T2" ] || { echo "not executable: $T2"; exit 2; }
W=$(mktemp -d ts_ckpt10.XXXXXX)
fail=0
note() { echo "$*"; }
bad()  { echo "  *** FAIL: $*"; fail=$((fail + 1)); }

R=9689
S=surv$R.txt
[ -f "$S" ] || { echo "missing $S"; exit 2; }
C="$R --skip 5 --maxd 2500 --gcd-threads 1 --pend-max 1 --ckpt-mins 0.002"

# reference results for the whole list; then the three survivors whose
# least factor is deepest (or none below maxd), so that a run can be
# interrupted in the middle of each
"$T" $C $S --m 6 --no-ckpt --out "$W/ref" > "$W/ref.out" 2> "$W/ref.err"
awk '{ d = ($2 == "u") ? 99999 : $2; print d, $1 }' "$W/ref.results.txt" |
    sort -k1,1nr -k2,2n | head -3 | awk '{ print $2 }' > "$W/L3.txt"
mapfile -t SV < "$W/L3.txt"
[ ${#SV[@]} -eq 3 ] || { echo "could not pick survivors"; exit 2; }
printf "%s\n" "${SV[1]}" "${SV[2]}" > "$W/L23.txt"
printf "%s\n" "${SV[0]}" > "$W/L1.txt"
grep -E "^(${SV[0]}|${SV[1]}|${SV[2]}) " "$W/ref.results.txt" | sort \
    > "$W/ref3.txt"
note "survivors used: ${SV[*]}"
sorted_eq() { sort "$1" | cmp -s - "$2"; }
ck() { echo "$W/$1_r${R}_s$2.ckpt"; }

# ---- 1. per-survivor checkpoint names; setting one survivor aside
"$T" $C "$W/L3.txt" --m 6 --out "$W/a" --die-after-blocks 150 -v \
    > "$W/a1.out" 2> "$W/a1.err"
if [ -f "$(ck a "${SV[0]}")" ] && ! [ -s "$W/a.results.txt" ]; then
    note "interrupted s=${SV[0]}: checkpoint $(basename "$(ck a "${SV[0]}")")"
else
    bad "no per-survivor checkpoint after the interruption"
    ls "$W"
fi
"$T" $C "$W/L23.txt" --m 6 --out "$W/a" -v > "$W/a2.out" 2> "$W/a2.err"
if grep -q "for s not in this list are kept for later: ${SV[0]}" "$W/a2.err" &&
   [ -f "$(ck a "${SV[0]}")" ] && [ "$(grep -c . "$W/a.results.txt")" -eq 2 ] &&
   ! grep -q "^Resumed" "$W/a2.out"; then
    note "a list without s=${SV[0]}: the other two done, its checkpoint kept and reported"
else
    bad "setting s=${SV[0]} aside"
fi
"$T" $C "$W/L1.txt" --m 6 --out "$W/a" -v > "$W/a3.out" 2> "$W/a3.err"
if grep -q "^Resumed s=${SV[0]} " "$W/a3.out" && sorted_eq "$W/a.results.txt" "$W/ref3.txt" &&
   [ -z "$(ls "$W"/a_r*.ckpt* 2>/dev/null)" ]; then
    note "the set-aside s=${SV[0]} resumed later: results identical, no checkpoint left"
else
    bad "resuming the set-aside survivor"
fi

# ---- 2. two survivors interrupted, resumed in one run (checkpoints first)
"$T" $C "$W/L1.txt" --m 6 --out "$W/b" --die-after-blocks 120 > /dev/null 2>&1
printf "%s\n" "${SV[2]}" > "$W/L3only.txt"
"$T" $C "$W/L3only.txt" --m 6 --out "$W/b" --die-after-blocks 90 > /dev/null 2>&1
"$T" $C "$W/L3.txt" --m 6 --out "$W/b" -v > "$W/b.out" 2> "$W/b.err"
if [ "$(grep -c "^Resumed" "$W/b.out")" -eq 2 ] &&
   grep -q "2 survivor(s) of this list have checkpoints" "$W/b.err" &&
   sorted_eq "$W/b.results.txt" "$W/ref3.txt"; then
    note "two interrupted survivors resumed by one run (first), results identical"
else
    bad "resuming two survivors in one run"
fi

# ---- 3. tuning: measured at a fresh start, adopted on resume, --retune
"$T" $C "$W/L1.txt" --out "$W/t" --die-after-blocks 150 > /dev/null 2> "$W/t1.err"
"$T" $C "$W/L1.txt" --out "$W/t" > /dev/null 2> "$W/t2.err"
"$T" $C "$W/L1.txt" --out "$W/u" --die-after-blocks 150 > /dev/null 2>&1
"$T" $C "$W/L1.txt" --out "$W/u" --retune > /dev/null 2> "$W/u2.err"
tline() { grep -m1 "^tuning: " "$1"; }
if tline "$W/t1.err" | grep -qv "from the checkpoint" &&
   tline "$W/t2.err" | grep -q "from the checkpoint of s=${SV[0]}" &&
   tline "$W/u2.err" | grep -q -- "--retune" &&
   sorted_eq "$W/t.results.txt" <(grep "^${SV[0]} " "$W/ref3.txt") &&
   sorted_eq "$W/u.results.txt" <(grep "^${SV[0]} " "$W/ref3.txt"); then
    note "tuning: timed at a fresh start, taken from the checkpoint on resume, timed again with --retune"
else
    bad "tuning on start/resume"
    tline "$W/t1.err"; tline "$W/t2.err"; tline "$W/u2.err"
fi

# ---- 4. the schedule's measured costs survive a resume
"$T" $C "$W/L1.txt" --m 6 --out "$W/g" --die-after-blocks 250 -v -v \
    > "$W/g1.out" 2>&1
"$T" $C "$W/L1.txt" --m 6 --out "$W/g" -v -v > "$W/g2.out" 2>&1
if grep -A1 "^Resumed" "$W/g2.out" | grep -q "schedule costs from the checkpoint" &&
   sorted_eq "$W/g.results.txt" <(grep "^${SV[0]} " "$W/ref3.txt"); then
    note "schedule: $(grep -m1 "schedule costs from the checkpoint" "$W/g2.out" | sed 's/^ *//')"
else
    bad "the measured schedule costs were not taken from the checkpoint"
fi

# ---- 5. one process per output prefix
if command -v flock > /dev/null; then
    flock "$W/l.lock" sleep 3 &
    sleep 0.5
    "$T" $C "$W/L1.txt" --m 6 --out "$W/l" > /dev/null 2> "$W/l.err"
    st=$?
    wait
    if [ $st -eq 1 ] && grep -q "another .* is running with --out" "$W/l.err" &&
       ! [ -s "$W/l.results.txt" ]; then
        note "lock: a second run with the same --out refuses (status 1)"
    else
        bad "lock: status $st"
    fi
    "$T" $C "$W/L1.txt" --m 6 --out "$W/l" > /dev/null 2>&1
    [ "$(grep -c . "$W/l.results.txt")" -eq 1 ] ||
        bad "lock: the run after the lock was released"
else
    note "lock: skipped (no flock(1) on this system)"
fi

# ---- 6. a checkpoint resumed by the other build
if [ -n "$T2" ]; then
    "$T" $C "$W/L1.txt" --out "$W/x" --die-after-blocks 150 > /dev/null 2>&1
    "$T2" $C "$W/L1.txt" --out "$W/x" -v > "$W/x.out" 2> "$W/x.err"
    if grep -q "tuning again" "$W/x.err" && grep -q "^Resumed" "$W/x.out" &&
       sorted_eq "$W/x.results.txt" <(grep "^${SV[0]} " "$W/ref3.txt"); then
        note "cross-build resume: $(basename "$T") -> $(basename "$T2"): tuned again, state carried over, results identical"
    else
        bad "cross-build resume"
    fi
fi

if [ $fail -eq 0 ]; then
    rm -rf "$W"
    echo "ts_ckpt10: all PASS"
else
    echo "ts_ckpt10: $fail FAILURE(S); files kept in $W"
    exit 1
fi

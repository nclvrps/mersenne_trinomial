#!/usr/bin/env bash
# check_vs_factor.sh -- byte-exact comparison of tsfactor against
# Brent/Zimmermann's factor over an s range.
#
#   ./check_vs_factor.sh <r> <skip> <s0> <s1> [path/to/factor] [path/to/tsfactor]
#
# Runs factor on every s in [s0, s1], collects the survivors (d > skip
# or u), runs tsfactor on exactly those s with the same skip, and diffs
# the result lines.  Small r (e.g. 44497, skip 8) finishes in seconds;
# do NOT point this at r=136279841 unless you have days.
set -euo pipefail
r=$1 skip=$2 s0=$3 s1=$4
factor=${5:-./factor} tsf=${6:-./tsfactor}
wd=$(mktemp -d)
trap 'rm -rf "$wd"' EXIT

"$factor" -skip 0 -k "$skip" -s0 "$s0" -s1 "$((s1 + 1))" "$r" \
    | awk 'NF>=2 && $1 ~ /^[0-9]+$/' > "$wd/ref.txt"

awk -v k="$skip" '($2 == "u") || ($2+0 > k) {print $1}' "$wd/ref.txt" \
    > "$wd/surv.txt"
n=$(wc -l < "$wd/surv.txt")
echo "factor done: $n survivors of degree > $skip in [$s0, $s1]"

"$tsf" "$r" "$wd/surv.txt" --skip "$skip" --out "$wd/ts" --no-ckpt \
    > "$wd/ts.log" 2>&1 || { tail -5 "$wd/ts.log"; exit 1; }

python3 - "$wd" <<'EOF'
import sys
wd = sys.argv[1]
ref = {l.split()[0]: ' '.join(l.split()[1:]) for l in open(f'{wd}/ref.txt')}
mine = {l.split()[0]: ' '.join(l.split()[1:])
        for l in open(f'{wd}/ts.results.txt')}
surv = [l.strip() for l in open(f'{wd}/surv.txt') if l.strip()]
bad = 0
for s in surv:
    if ref.get(s) != mine.get(s):
        print(f"MISMATCH s={s}: factor='{ref.get(s)}' tsfactor='{mine.get(s)}'")
        bad += 1
print(f"{len(surv)} survivors compared: {bad} mismatches")
sys.exit(1 if bad else 0)
EOF

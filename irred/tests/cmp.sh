#!/bin/bash
# Compare irred builds with the naive reference (tools/ref.c) on many (r, s):
# boundary values of s, every s with r - sodd = 0 (mod 128) (the case 0.14
# got wrong), small sodd, r = 127 (mod 128) (383, 1151, 1279), random s.
# Then the low 32 bits against irred 3.15 at two larger r.
# usage: tests/cmp.sh BUILD_DIR
D=${1:-build}
cd "$D" || exit 1
fail=0; n=0
BINS=("irred" "irred --isa avx2" "irred --isa generic" "irred --no-ec"
      "irred --threads 2 --mt-min-words 1" "irred --threads 3 --mt-min-words 1"
      "irred --nt" "irred --nt --isa avx2 --threads 3 --mt-min-words 1"
      "irred_generic" "irred_asan" "irred_asan --isa generic"
      "irred_asan --threads 2 --mt-min-words 1")
check() {   # r s
  local r=$1 s=$2 want got b
  want=$(./ref $r $s | awk '{print $3}')
  for b in "${BINS[@]}"; do
    got=$(./$b --no-ckpt $r $s 2>&1 | awk '/ LOG$/{print $3} /runtime error|ERROR:/{print "SANITIZER"}')
    n=$((n+1))
    if [ "$got" != "$want" ]; then
      echo "MISMATCH $b r=$r s=$s got=$got want=$want"; fail=$((fail+1)); fi
  done
}
for r in 257 383 1151 1279 2203 4423 9689; do
  S="128 129 $((r-128)) $((r-129)) $((r/2)) $((r/2+1)) 1 3 $((r-1)) $((r-3))"
  for so in $(seq 1 2 $r); do
    if [ $(( (r-so) % 128 )) -eq 0 ] && [ $((r-so)) -ge 128 ]; then S="$S $so $((r-so))"; fi
  done
  cnt=12; [ $r -gt 5000 ] && cnt=4
  S="$S $(python3 -c "import random;random.seed($r);print(' '.join(str(random.randint(1,$r-1)) for _ in range($cnt)))")"
  for s in $S; do
    so=$(( s % 2 ? s : r - s ))
    [ $((r - so)) -lt 128 ] && continue
    check $r $s
  done
done
echo "reference comparison: checks=$n failures=$fail"

# low 32 bits against 3.15 at larger r (includes r - sodd = 0 mod 128 cases)
f2=0
for r in 44497 132049; do
  for s in 8575 21034 12345 12346 $((r-12800)) $((r-12928)) 7000 33912 54454 60001; do
    [ $s -ge $r ] && continue
    a=$(echo "$r $s $((s+1))" | ./irred315 2>/dev/null | awk '/ LOG$/{print $3}')
    b=$(./irred --no-ckpt $r $s | awk '/ LOG$/{print $3}')
    if [ "$a" = irreducible ]; then [ "$b" = irreducible ] || { echo "BAD $r $s 3.15=$a new=$b"; f2=$((f2+1)); }
    else [ "${a:0:1}${a:1}" = "${b:0:1}${b:9}" ] || { echo "BAD $r $s 3.15=$a new=$b"; f2=$((f2+1)); }; fi
  done
done
echo "3.15 comparison (low 32 bits, r = 44497, 132049): failures=$f2"
[ $fail -eq 0 ] && [ $f2 -eq 0 ]

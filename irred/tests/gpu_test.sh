#!/bin/bash
# Tests of the GPU build (irred_gpu) or of the same under the CUDA emulation
# (irred_gpu_emul): selftest, results against the naive reference for many
# (r, s) -- including the s the CPU build cannot do --, agreement with the
# CPU build, checkpoints passed between the two builds, and the fault sweeps.
# usage: tests/gpu_test.sh path/to/irred_gpu [BUILD_DIR] [--full]
#   BUILD_DIR has the CPU build (irred) and the reference (ref).
#   --full adds the first 4000 squarings at r = 136279841, s = 12338357,
#   compared with the CPU build's residues (Zen 2 log of round 3).
G=$(readlink -f ${1:-build/irred_gpu}); D=$(readlink -f ${2:-build}); FULL=$3
CPU=$D/irred; REF=$D/ref; T=$(readlink -f $(dirname "$0"))
pass=0; failn=0
ok()  { echo "PASS: $*"; pass=$((pass+1)); }
bad() { echo "FAIL: $*"; failn=$((failn+1)); }
res() { awk '/ LOG$/{print $3}' "$@"; }

echo "== selftest"
$G --selftest > st.out 2>&1
grep -E 'GPU|reference chain' st.out | sed 's/^/   /'
tail -1 st.out | grep -q '^selftest passed' && ok "selftest" || bad "selftest: $(tail -1 st.out)"
rm -f st.out

echo "== results against the naive reference (modulus x^r + x^min(s, r-s) + 1)"
n=0; f=0; nc=0; fc=0
for r in 257 263 383 1279 2203 4423; do
  S="1 2 3 4 63 64 65 127 128 129 $(((r-1)/2)) $(((r+1)/2)) $((r-1)) $((r-2)) $((r-3)) $((r-64)) $((r-128)) $((r-129))"
  S="$S $(python3 -c "import random;random.seed($r);print(' '.join(str(random.randint(1,$r-1)) for _ in range(6)))")"
  for s in $S; do
    want=$($REF $r $s gpu | awk '{print $3}')
    got=$($G --no-ckpt $r $s 2>&1 | res)
    n=$((n+1))
    [ "$got" = "$want" ] || { echo "   MISMATCH r=$r s=$s got=$got want=$want"; f=$((f+1)); }
    # the CPU build, where it can run: the same verdict; the same residue
    # when both use the same modulus (s odd, s < r/2)
    so=$(( s % 2 ? s : r - s ))
    if [ $((r - so)) -ge 128 ]; then
      c=$($CPU --no-ckpt $r $s 2>&1 | res)
      nc=$((nc+1))
      if [ $(( s % 2 )) = 1 ] && [ $((2 * s)) -lt $r ]; then
        [ "$c" = "$got" ] || { echo "   CPU/GPU differ r=$r s=$s cpu=$c gpu=$got"; fc=$((fc+1)); }
      else
        [ "$c" = irreducible ] && [ "$got" != irreducible ] && { echo "   verdicts differ r=$r s=$s"; fc=$((fc+1)); }
        [ "$c" != irreducible ] && [ "$got" = irreducible ] && { echo "   verdicts differ r=$r s=$s"; fc=$((fc+1)); }
      fi
    fi
  done
done
[ $f = 0 ] && ok "$n results equal the reference" || bad "$f of $n results differ from the reference"
[ $fc = 0 ] && ok "$nc comparisons with the CPU build agree" || bad "$fc of $nc comparisons with the CPU build differ"

echo "== checkpoints passed between the CPU and GPU builds"
W=$(mktemp -d /tmp/irred_gpu_XXXX); cd $W
R=44497; S=8575                       # s odd, s < r/2: the same modulus
WANT=$($CPU --no-ckpt $R $S | res)
$CPU --ckpt-mins 0 --stop-after 15000 $R $S > a1.out 2>/dev/null; rc=$?
$G --ckpt-mins 0 $R $S > a2.out 2>/dev/null
[ $rc = 2 ] && grep -q '^Resuming' a2.out && [ "$(res a2.out)" = "$WANT" ] \
  && ok "CPU checkpoint resumed by the GPU build, result $WANT" || bad "CPU -> GPU (rc=$rc, $(res a2.out))"
$G --ckpt-mins 0 --stop-after 15000 $R $S > b1.out 2>/dev/null; rc=$?
$CPU --ckpt-mins 0 $R $S > b2.out 2>/dev/null
[ $rc = 2 ] && grep -q '^Resuming' b2.out && [ "$(res b2.out)" = "$WANT" ] \
  && ok "GPU checkpoint resumed by the CPU build" || bad "GPU -> CPU (rc=$rc, $(res b2.out))"
[ -z "$(ls irred_* 2>/dev/null)" ] && ok "no files left" || bad "left: $(ls irred_*)"
S=12346                               # s even: different moduli
$CPU --ckpt-mins 0 --stop-after 2000 $R $S > c1.out 2>/dev/null
$G --ckpt-mins 0 --stop-after 2000 $R $S > c2.out 2>/dev/null; rc=$?
grep -q 'it holds powers of x modulo' c2.out && ! grep -q '^Resuming' c2.out && [ $rc = 2 ] \
  && ok "s even: the CPU checkpoint is recognised and not used" || bad "s even: $(grep -i -E 'ignor|resum' c2.out)"
rm -rf $W

echo "== fault sweeps (as tests/ec_sweep_all.sh)"
$T/ec_sweep_all.sh $G > sw.out 2>&1; rc=$?
sed 's/^/   /' sw.out; rm -f sw.out
[ $rc = 0 ] && ok "fault sweeps" || bad "fault sweeps"

if [ "$FULL" = "--full" ]; then
  echo "== r = 136279841, s = 12338357: residues at 2000 and 4000 (CPU build: Zen 2 log)"
  $G --no-ckpt --progress 2000 --stop-after 4000 136279841 12338357 > full.out 2>&1
  grep -E '^GPU|kernel|^iter' full.out | sed 's/^/   /'
  grep -q 'iter 2000/.*residue 524cef6916ae5278' full.out && grep -q 'iter 4000/.*residue d61458d6bb7b0384' full.out \
    && ok "full size: residues equal the CPU build's" || bad "full size residues"
  rm -f full.out
fi

echo "gpu tests: $pass passed, $failn failed"
[ $failn = 0 ]

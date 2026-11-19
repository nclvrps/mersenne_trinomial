#!/bin/bash
# Re-inserts the real reduction bug of irred 0.14 (deltaq == 0, i.e.
# r - sodd = 0 mod 128) into a scratch copy of irred.c and shows:
#   * the Gerbicz chain alone cannot see it (every window passes, wrong
#     residue -- the same low 32 bits that 0.14 printed),
#   * the start-up map check catches it (exit status 3),
#   * cases with deltaq != 0 are unaffected.
# (The bug goes into the single-threaded reducep, which is what runs at
#  these sizes.)
# usage: tests/bug014_test.sh   (from the devkit root; needs build/ref,
#        build/irred014 is built here from reference/irred_0.14.c)
set -e
mkdir -p build
python3 - <<'EOF'
t = open('irred/irred.c').read()
anchor = '''  a[q11 + 2] = 0;

  /* targets T = q11-deltaw down to q4+1-deltaw'''
bug = '''  a[q11 + 2] = 0;
  if (deltaq == 0) {           /* irred 0.14's code for this case */
    for (long j = q11 - q4 - 1; j >= 0; j--) a[q4 - deltaw + j] ^= a[q4 + j];
    a[q4 - deltaw] ^= a[q4] & mask2;
    return;
    }

  /* targets T = q11-deltaw down to q4+1-deltaw'''
assert t.count(anchor) == 1
open('build/irred_bug014.c', 'w').write(t.replace(anchor, bug))
EOF
gcc -O2 -w -o build/irred_bug014 build/irred_bug014.c -lm
gcc -O2 -w -o build/irred014 reference/irred_0.14.c
set +e
fail=0
for s in 3015 2247 1991; do          # r - sodd = 0 (mod 128)
  want=$(build/ref 4423 $s | awk '{print $3}')
  old=$(echo "4423 $s $((s+1))" | build/irred014 | awk '/ LOG$/{print $3}')
  out=$(build/irred_bug014 --no-ckpt --map-checks 0 4423 $s)
  got=$(awk '/ LOG$/{print $3}' <<<"$out")
  chain_ok=$(grep -c 'ERROR CHECK FAILED' <<<"$out")
  build/irred_bug014 --no-ckpt 4423 $s > /dev/null; rc=$?
  echo "4423 $s: true $want | 0.14 $old | bug + chain only: $got ($chain_ok check failures) | with map check: exit $rc"
  [ "$got" != "$want" ] && [ "${got:9}" = "${old:1}" ] && [ $chain_ok = 0 ] && [ $rc = 3 ] || fail=1
done
for s in 1000 2098; do               # deltaq != 0: unaffected
  want=$(build/ref 4423 $s | awk '{print $3}')
  got=$(build/irred_bug014 --no-ckpt 4423 $s | awk '/ LOG$/{print $3}')
  echo "4423 $s (deltaq != 0): true $want, bug copy $got"
  [ "$got" = "$want" ] || fail=1
done
[ $fail = 0 ] && echo "0.14 bug: invisible to the chain, caught by the map check" || echo "UNEXPECTED"
exit $fail

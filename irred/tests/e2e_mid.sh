#!/bin/bash
# End-to-end at a mid-size r: 2 threads + error checking + checkpoints,
# interrupted after 120 s and resumed; compared with a single-threaded run
# without error checking.  About 12 minutes on a 2-core machine.
B=$(readlink -f ${1:-build/irred}); R=${2:-6972593}; S=${3:-100001}
W=$(mktemp -d /tmp/irred_e2e_XXXX); cd $W
$B --threads 2 --ckpt-mins 0.5 --progress 1000000 $R $S > a1.out 2> a1.err & P=$!
sleep 120; kill -INT $P; wait $P; echo "first part exit $? ($(grep -o 'after [0-9]*' a1.out))"
$B --threads 2 --ckpt-mins 0.5 --progress 1000000 -v $R $S > a2.out 2> a2.err
grep -E 'Resuming|Error checking|squarings were|LOG' a2.out
grep -c 'checkpoint:' a1.err a2.err
$B --no-ckpt --no-ec --threads 1 --progress 1000000 $R $S > b.out
grep LOG b.out
A=$(awk '/ LOG$/{print $3}' a2.out); Bv=$(awk '/ LOG$/{print $3}' b.out)
cat a1.out a2.out | grep '^iter' | awk '{for(i=1;i<=NF;i++) if($i=="residue") print $2, $(i+1)}' | sort -u > pa
grep '^iter' b.out | awk '{for(i=1;i<=NF;i++) if($i=="residue") print $2, $(i+1)}' | sort -u > pb
[ "$A" = "$Bv" ] && [ -z "$(comm -3 pa pb)" ] && echo "E2E PASS: $A, progress residues identical" || echo "E2E FAIL: $A vs $Bv"
cd /; rm -rf $W

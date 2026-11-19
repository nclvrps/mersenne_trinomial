#!/bin/bash
# Exhaustive fault-injection sweep: one bit flip per run, at every
# opportunity of a site.  Checked runs must detect every injected fault and
# still give the right answer; unchecked (--no-ec) runs show what the check
# prevents.
# usage: ec_sweep.sh IRRED r s site maxN [extra irred options...]
B=$1; r=$2; s=$3; site=$4; maxN=$5; shift 5; X="$@"
want=$($B --no-ckpt --map-checks 0 $X $r $s | awk '/ LOG$/{print $3}')
inj=0; det=0; right=0; bad=0; ne_wrong=0; ne_right=0
for n in $(seq 1 $maxN); do
  out=$($B --no-ckpt --map-checks 0 --ec-fault $site:$n $X $r $s 2>&1)
  grep -q '^\[ec-fault' <<<"$out" || continue
  inj=$((inj+1))
  got=$(awk '/ LOG$/{print $3}' <<<"$out")
  grep -q -E 'ERROR CHECK FAILED|damaged in memory' <<<"$out" && det=$((det+1))
  if [ "$got" = "$want" ]; then right=$((right+1)); else bad=$((bad+1)); echo "  WRONG/NO RESULT at $site:$n: '$got'"; fi
  if [ "$site" = sq ]; then
    g2=$($B --no-ckpt --map-checks 0 --no-ec --ec-fault $site:$n $X $r $s 2>&1 | awk '/ LOG$/{print $3}')
    if [ "$g2" = "$want" ]; then ne_right=$((ne_right+1)); else ne_wrong=$((ne_wrong+1)); fi
  fi
done
echo "r=$r s=$s site=$site opts='$X': injected=$inj detected=$det right=$right wrong=$bad" \
     "$( [ $site = sq ] && echo "| unchecked: silently wrong=$ne_wrong, harmless=$ne_right" )"
[ $bad -eq 0 ] && [ $det -eq $inj ]

# estimate.py -- per-degree estimate of round 7's gain (coarse_sieve round 7)
#
#   python3 estimate.py        (in the directory of passavg.py's outputs,
#                               pavg2_<d>.txt: fills 0,1,2 of each degree)
#
# From the simulated counts and the round-6 tune/pass times (the round-6
# hardware log); see the round-7 handoff, section 3.  Model, per class
# credited: c_fill * (enumeration lane-steps) + c_batch * (batch slots,
# relative to full batches); c_batch from the tune line (the dense sample,
# ~full batches), scaled by q = 1 (as measured) or 0.8 (the sample's runs:
# some lanes had 4, most 3), c_fill from the pass's time.  A pooled step
# costs OVER times a round-6 one.
import re
tune={40:0.363,41:0.494,42:0.382,43:0.553,44:0.384,45:0.582,46:0.830,47:0.586}
real={40:0.704,41:0.662,42:0.689,43:0.687,44:0.671,45:0.726,46:1.039,47:0.715}
eng={40:'tower',41:'RR',42:'tower',43:'RR',44:'tower',45:'RR',46:'classic',47:'RR'}
rows={}
for d in tune:
    try: t=open('pavg2_%d.txt'%d).read()
    except: continue
    for m in re.finditer(r'fill (\d): lane-steps per class ([0-9.]+), batch slots used ([0-9.]+)%, lane-batches per 1000 classes ([0-9.]+)',t):
        rows[(d,int(m.group(1)))]=(float(m.group(2)),float(m.group(3)),float(m.group(4)))
OVER=1.9   # a pooled enumeration iteration costs ~1.9x a round-6 one (~40 more instructions)
for d in sorted(tune):
    if (d,0) not in rows or (d,2) not in rows: continue
    S6,_,B6=rows[(d,0)]; S7,_,B7=rows[(d,2)]
    ratio=B6/B7
    out=[]
    for q in (1.0,0.8):
        cb=q*tune[d]
        cf=(real[d]-cb*ratio)/S6
        if cf>0:
            t7=cf*OVER*S7+cb
            out.append("q=%.1f: c_fill %.4f, gain %.2fx"%(q,cf,real[d]/t7))
        else:
            out.append("q=%.1f: does not fit (padding cheaper than a full slot)"%q)
    print("d=%d %-7s batches r6/r7 %.2f, steps %.1f -> %.1f; "%(d,eng[d],ratio,S6,S7)+"; ".join(out))

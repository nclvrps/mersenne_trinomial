# survivors for small r: s with no factor of degree <= skip
import sys
r=int(sys.argv[1]); skip=int(sys.argv[2]); smax=int(sys.argv[3])
def pmod(a,b):
    db=b.bit_length()
    while a.bit_length()>=db: a^=b<<(a.bit_length()-db)
    return a
def pgcd(a,b):
    while b: a,b=b,pmod(a,b)
    return a
def mulmod(a,b,T):
    c=0
    while b:
        if b&1: c^=a
        b>>=1; a<<=1
    return pmod(c,T)
out=[]
for s in range(1,smax+1):
    if 2*s>=r: break
    T=(1<<r)|(1<<s)|1
    y=2; ok=True
    for d in range(1,skip+1):
        y=mulmod(y,y,T)
        if pgcd(T,y^2)!=1: ok=False; break
    if ok: out.append(s)
print("\n".join(map(str,out)))

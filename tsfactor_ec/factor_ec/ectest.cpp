/* standalone test of the word-level fold helpers copied verbatim from factor_ec.cpp */
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
typedef uint64_t _ntl_ulong;
#define NTL_BITS_PER_LONG 64
#define ECFOLDW 8                      /* small for testing: n = 512 bits */
typedef struct { _ntl_ulong w[ECFOLDW]; } ecfold;

/* ---- verbatim from factor_ec.cpp (GF2X access replaced by raw words) ---- */
static void ec_fold_words (ecfold *f, const _ntl_ulong *p, long n)
{
  memset (f->w, 0, sizeof f->w);
  long i = 0, j;
  for (; i + ECFOLDW <= n; i += ECFOLDW)
    for (j = 0; j < ECFOLDW; j++) f->w[j] ^= p[i + j];
  for (j = 0; i + j < n; j++) f->w[j] ^= p[i + j];
}
static void ec_fold_shl1 (ecfold *c, const ecfold *a)
{
  const int TOP = NTL_BITS_PER_LONG - 1;
  _ntl_ulong carry = a->w[ECFOLDW - 1] >> TOP;
  for (long j = ECFOLDW - 1; j > 0; j--)
    c->w[j] = (a->w[j] << 1) | (a->w[j - 1] >> TOP);
  c->w[0] = (a->w[0] << 1) | carry;
}
/* ---- independent reference: bit-level ---- */
static bool bit(const std::vector<uint64_t>&v,long i){ size_t w=i>>6; return w<v.size() && ((v[w]>>(i&63))&1); }
int main(){
  std::mt19937_64 rng(1);
  const long n = ECFOLDW*64;
  int bad=0;
  for(int t=0;t<20000;t++){
    long nw = 1 + (long)(rng()%40);
    std::vector<uint64_t> a(nw);
    for(auto&x:a) x=rng();
    /* reference fold: coefficient i of phi(a) = XOR of bits i, i+n, i+2n,... */
    std::vector<uint64_t> ref(ECFOLDW,0);
    for(long i=0;i<nw*64;i++) if(bit(a,i)) ref[((i%n)>>6)] ^= (uint64_t)1<<((i%n)&63);
    ecfold f; ec_fold_words(&f,a.data(),nw);
    if(memcmp(f.w,ref.data(),sizeof f.w)) {bad++; if(bad<3)printf("fold mismatch t=%d\n",t);}
    /* reference x*phi(a) mod x^n+1: rotate bits left by 1 */
    std::vector<uint64_t> rot(ECFOLDW,0);
    for(long i=0;i<n;i++) if(bit(ref,i)) rot[(((i+1)%n)>>6)] ^= (uint64_t)1<<(((i+1)%n)&63);
    ecfold s; ec_fold_shl1(&s,&f);
    if(memcmp(s.w,rot.data(),sizeof s.w)) {bad++; if(bad<6)printf("shl1 mismatch t=%d\n",t);}
  }
  printf("fold/shl1 word-level tests: %s (%d mismatches / 40000 checks)\n", bad?"FAIL":"PASS", bad);
  return bad!=0;
}

#pragma once
/* gecsim.cpp -- dependency-free reference model + fault-injection harness for
 * error checking of the multi-level-blocking DDF trinomial factor search.
 *
 * Mirrors the algorithm of factor.cpp (Brent/Gaudry/Thome/Zimmermann) at small
 * r so that the *checking scheme* can be validated exhaustively, independent of
 * NTL / gf2x / CUDA.
 *
 * Build:  g++ -O2 -o gecsim gecsim.cpp
 *
 * The point of this program is NOT speed.  It is to answer, with measured
 * numbers rather than hand-waving:
 *   1. Is the Frobenius-linear Gerbicz identity  sigma(sum_j h_j^(z)) ==
 *      sum_j h_j^(z+1)  actually an identity for the h-table factor.cpp uses?
 *   2. Does it detect injected single-bit faults in the squaring substep?
 *   3. Do the cyclic-fold checks detect faults in the product substep?
 *   4. Does the Bezout certificate detect faults in the GCD substep?
 *   5. Without checking, how often does a fault cause a *silent wrong answer*
 *      (a missed factor) as opposed to a self-evidently bogus one?
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

typedef std::vector<uint64_t> Poly;   /* little-endian words, bit i of word w = x^(64w+i) */

/* ------------------------------------------------------------------ *
 *  Basic GF(2)[x] arithmetic.  Schoolbook; correctness over speed.
 * ------------------------------------------------------------------ */

static void norm(Poly &a) { while (!a.empty() && a.back() == 0) a.pop_back(); }

static long deg(const Poly &a)
{
  for (long i = (long)a.size() - 1; i >= 0; i--)
    if (a[i]) return i * 64 + 63 - __builtin_clzll(a[i]);
  return -1;
}

static bool iszero(const Poly &a) { return deg(a) < 0; }
static bool isone(const Poly &a)  { return deg(a) == 0; }

static bool getbit(const Poly &a, long i)
{
  if (i < 0) return false;
  size_t w = (size_t)(i >> 6);
  if (w >= a.size()) return false;
  return (a[w] >> (i & 63)) & 1;
}

static void flipbit(Poly &a, long i)
{
  size_t w = (size_t)(i >> 6);
  if (w >= a.size()) a.resize(w + 1, 0);
  a[w] ^= (uint64_t)1 << (i & 63);
  norm(a);
}

static void setbit(Poly &a, long i, bool v)
{
  if (getbit(a, i) != v) flipbit(a, i);
}

static void addto(Poly &a, const Poly &b)          /* a ^= b */
{
  if (a.size() < b.size()) a.resize(b.size(), 0);
  for (size_t i = 0; i < b.size(); i++) a[i] ^= b[i];
  norm(a);
}

static Poly padd(const Poly &a, const Poly &b) { Poly c = a; addto(c, b); return c; }

static bool equal(const Poly &a, const Poly &b)
{
  size_t n = std::max(a.size(), b.size());
  for (size_t i = 0; i < n; i++) {
    uint64_t x = i < a.size() ? a[i] : 0, y = i < b.size() ? b[i] : 0;
    if (x != y) return false;
  }
  return true;
}

/* a ^= (b << k) */
static void addshifted(Poly &a, const Poly &b, long k)
{
  if (iszero(b)) return;
  long wk = k >> 6, bk = k & 63;
  size_t need = b.size() + (size_t)wk + 1;
  if (a.size() < need) a.resize(need, 0);
  for (size_t i = 0; i < b.size(); i++) {
    a[i + wk] ^= b[i] << bk;
    if (bk) a[i + wk + 1] ^= b[i] >> (64 - bk);
  }
  norm(a);
}

static Poly shl(const Poly &a, long k) { Poly c; addshifted(c, a, k); return c; }

/* c = a >> k  (drop low k coefficients) */
static Poly shr(const Poly &a, long k)
{
  Poly c;
  long d = deg(a);
  if (d < k) return c;
  c.resize((size_t)((d - k) / 64 + 1), 0);
  for (long i = k; i <= d; i++)
    if (getbit(a, i)) c[(size_t)((i - k) >> 6)] |= (uint64_t)1 << ((i - k) & 63);
  norm(c);
  return c;
}

/* full (unreduced) product */
static Poly mul(const Poly &a, const Poly &b)
{
  Poly c;
  long db = deg(b);
  if (db < 0 || deg(a) < 0) return c;
  c.resize(a.size() + b.size() + 1, 0);
  for (long i = 0; i <= db; i++)
    if (getbit(b, i)) {
      long wk = i >> 6, bk = i & 63;
      for (size_t j = 0; j < a.size(); j++) {
        c[j + wk] ^= a[j] << bk;
        if (bk) c[j + wk + 1] ^= a[j] >> (64 - bk);
      }
    }
  norm(c);
  return c;
}

/* full (unreduced) square: spread bits to even positions */
static Poly sqr_raw(const Poly &a)
{
  Poly c;
  long d = deg(a);
  if (d < 0) return c;
  c.resize((size_t)(2 * d / 64 + 2), 0);
  for (long i = 0; i <= d; i++)
    if (getbit(a, i)) c[(size_t)((2 * i) >> 6)] |= (uint64_t)1 << ((2 * i) & 63);
  norm(c);
  return c;
}

/* long division: a = q*b + rem */
static void divrem(Poly &q, Poly &rem, const Poly &a, const Poly &b)
{
  q.clear(); rem = a;
  long db = deg(b);
  if (db < 0) { fprintf(stderr, "divide by zero\n"); exit(1); }
  for (long d = deg(rem); d >= db; d = deg(rem)) {
    long k = d - db;
    flipbit(q, k);
    addshifted(rem, b, k);
  }
}

static Poly gcd_poly(Poly a, Poly b)
{
  while (!iszero(b)) { Poly q, r; divrem(q, r, a, b); a = b; b = r; }
  return a;
}

/* extended gcd: returns g, and u,v with u*a + v*b = g   (exact, unreduced) */
static Poly xgcd_poly(const Poly &a0, const Poly &b0, Poly &u, Poly &v)
{
  Poly a = a0, b = b0;
  Poly u0, u1, v0, v1;
  setbit(u0, 0, 1);            /* u0 = 1 */
  setbit(v1, 0, 1);            /* v1 = 1 */
  while (!iszero(b)) {
    Poly q, r; divrem(q, r, a, b);
    a = b; b = r;
    Poly nu = padd(u0, mul(q, u1)); u0 = u1; u1 = nu;
    Poly nv = padd(v0, mul(q, v1)); v0 = v1; v1 = nv;
  }
  u = u0; v = v0;
  return a;
}

/* ------------------------------------------------------------------ *
 *  Fault injection
 * ------------------------------------------------------------------ */

struct FaultModel {
  std::mt19937_64 rng;
  double p_sqr   = 0.0;   /* per squaring   */
  double p_mul   = 0.0;   /* per modmul     */
  double p_horn  = 0.0;   /* per Horner op  */
  double p_gcd   = 0.0;   /* per gcd call   */
  double p_idle  = 0.0;   /* per h-word per block (VRAM bit rot) */
  long   injected = 0;
  /* deterministic single-fault mode: fire on the n-th op of a given kind */
  long   n_sqr = 0, n_mul = 0, n_horn = 0, n_gcd = 0, n_idle = 0;
  long   target_sqr = -1, target_mul = -1, target_horn = -1,
         target_gcd = -1, target_idle = -1;
  bool   fire_sqr()  { return (target_sqr  >= 0) ? (n_sqr++  == target_sqr)  : coin(p_sqr); }
  bool   fire_mul()  { return (target_mul  >= 0) ? (n_mul++  == target_mul)  : coin(p_mul); }
  bool   fire_horn() { return (target_horn >= 0) ? (n_horn++ == target_horn) : coin(p_horn); }
  bool   fire_gcd()  { return (target_gcd  >= 0) ? (n_gcd++  == target_gcd)  : coin(p_gcd); }
  bool   fire_idle() { return (target_idle >= 0) ? (n_idle++ == target_idle) : coin(p_idle); }
  FaultModel(uint64_t seed = 12345) : rng(seed) {}
  bool coin(double p) { if (p <= 0) return false;
    return std::uniform_real_distribution<double>(0,1)(rng) < p; }
  /* flip one random bit in the low `nbits` positions of p */
  void corrupt(Poly &a, long nbits) {
    long d = deg(a);
    long lim = (d >= 0 ? d + 1 : 1);
    if (nbits > 0) lim = nbits;
    long i = (long)(rng() % (uint64_t)lim);
    flipbit(a, i);
    injected++;
  }
};

/* ------------------------------------------------------------------ *
 *  The trinomial ring  R = GF(2)[x]/(x^r + x^s + 1)
 * ------------------------------------------------------------------ */

struct Ring {
  long r, s;
  Poly T;
  FaultModel *fm = nullptr;

  Ring(long r_, long s_) : r(r_), s(s_) {
    setbit(T, r, 1); setbit(T, s, 1); setbit(T, 0, 1);
  }

  /* reduce in place mod T, bit-serial from the top (mirrors fastrem's
     mathematical effect).  Also (optionally) returns the quotient. */
  void reduce(Poly &a, Poly *quot = nullptr) const {
    if (quot) quot->clear();
    for (long j = deg(a); j >= r; j--)
      if (getbit(a, j)) {
        flipbit(a, j);
        flipbit(a, j - r);
        flipbit(a, j - (r - s));
        if (quot) flipbit(*quot, j - r);
      }
  }

  Poly rem(const Poly &a) const { Poly c = a; reduce(c); return c; }

  Poly mulmod(const Poly &a, const Poly &b) const {
    Poly t = mul(a, b);
    Poly c = t; reduce(c);
    return c;
  }

  Poly sqrmod(const Poly &a) const {
    Poly t = sqr_raw(a);
    reduce(t);
    return t;
  }

  Poly mulxmod(const Poly &a) const {   /* a*x mod T */
    Poly c = shl(a, 1);
    if (deg(c) >= r) { flipbit(c, r); flipbit(c, s); flipbit(c, 0); }
    return c;
  }
};

/* ------------------------------------------------------------------ *
 *  Cyclic fold checksum  phi_n : GF(2)[x] -> GF(2)[x]/(x^n + 1)
 *  n a multiple of 64  =>  pure word-wise XOR fold.
 * ------------------------------------------------------------------ */

struct Fold {
  long nw;                      /* n/64 words */
  explicit Fold(long nbits) : nw(nbits / 64) {}
  long nbits() const { return nw * 64; }

  std::vector<uint64_t> of(const Poly &a) const {
    std::vector<uint64_t> o((size_t)nw, 0);
    for (size_t i = 0; i < a.size(); i++) o[i % (size_t)nw] ^= a[i];
    return o;
  }
  Poly topoly(const std::vector<uint64_t> &f) const {
    Poly p(f.begin(), f.end()); norm(p); return p;
  }
  /* product in GF(2)[x]/(x^n+1) */
  std::vector<uint64_t> mulf(const std::vector<uint64_t> &a,
                             const std::vector<uint64_t> &b) const {
    Poly pa = topoly(a), pb = topoly(b);
    return of(mul(pa, pb));
  }
  std::vector<uint64_t> addf(const std::vector<uint64_t> &a,
                             const std::vector<uint64_t> &b) const {
    std::vector<uint64_t> c = a;
    for (size_t i = 0; i < c.size(); i++) c[i] ^= b[i];
    return c;
  }
  static bool eq(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b) {
    return a == b;
  }
  static bool isnz(const std::vector<uint64_t> &a) {
    for (uint64_t w : a) if (w) return true;
    return false;
  }
};

/* ------------------------------------------------------------------ *
 *  Instrumented substeps
 * ------------------------------------------------------------------ */

struct Stats {
  long faults_injected = 0;
  long faults_caught   = 0;
  long rollbacks       = 0;
  long sqr_ops = 0, mul_ops = 0;
  long check_sqr_ops = 0, check_mul_ops = 0, fold_ops = 0;
};

struct Engine {
  Ring   R;
  Fold   F;
  FaultModel *fm;
  Stats  st;
  bool   checks_on;

  Engine(long r, long s, long nbits, FaultModel *f, bool on)
    : R(r, s), F(nbits), fm(f), checks_on(on) {}

  /* --- squaring, with optional injected fault --- */
  Poly sqrmod(const Poly &a) {
    st.sqr_ops++;
    Poly c = R.sqrmod(a);
    if (fm && fm->fire_sqr()) fm->corrupt(c, R.r);
    return c;
  }
  Poly sqrmod_check(const Poly &a) {          /* used inside the checker itself */
    st.check_sqr_ops++;
    return R.sqrmod(a);
  }

  /* --- multiply mod T, with fold verification of the *unreduced* product
         and independent re-reduction ------------------------------------ */
  Poly mulmod(const Poly &a, const Poly &b, bool *ok) {
    st.mul_ops++;
    Poly t = mul(a, b);
    if (fm && fm->fire_mul()) fm->corrupt(t, 2 * R.r);   /* fault in mult */
    if (checks_on) {
      st.fold_ops += 3;
      auto fa = F.of(a), fb = F.of(b), ft = F.of(t);
      if (!Fold::eq(ft, F.mulf(fa, fb))) { *ok = false; return Poly(); }
    }
    Poly c = t; R.reduce(c);
    if (fm && fm->fire_mul()) fm->corrupt(c, R.r);       /* fault in reduce */
    if (checks_on) {
      /* independent re-reduction (a different, simpler code path in a real
         implementation) and full comparison */
      Poly c2 = t; R.reduce(c2);
      if (!equal(c, c2)) { *ok = false; return Poly(); }
    }
    *ok = true;
    return c;
  }
};

/* ------------------------------------------------------------------ *
 *  h-table (elementary symmetric functions of Frobenius powers)
 * ------------------------------------------------------------------ */

/* h[j] = e_{j+1}( x^(2^k), ..., x^(2^(k+m-1)) )  mod T */
static std::vector<Poly> init_h(Engine &E, long k, long m)
{
  std::vector<Poly> h((size_t)m);
  Poly x; setbit(x, 1, 1);
  for (long j = 1; j <= m; j++) {
    h[(size_t)(j - 1)].clear();
    for (long i = j - 1; i >= 0; i--) {
      Poly a;
      if (i > 0) { h[(size_t)(i-1)] = E.R.sqrmod(h[(size_t)(i-1)]);
                   a = E.R.mulxmod(h[(size_t)(i-1)]); }
      else       { a = x; }
      addto(h[(size_t)i], a);
    }
  }
  for (long j = 0; j < m; j++)
    for (long i = 0; i < k; i++) h[(size_t)j] = E.R.sqrmod(h[(size_t)j]);
  return h;
}

/* C = prod_{i<m} (x + x^(2^(K+i)))  via Horner on the h-table */
static Poly horner_C(Engine &E, const std::vector<Poly> &h, long m, bool *ok)
{
  Poly C; setbit(C, 0, 1);
  std::vector<uint64_t> fC = E.F.of(C);
  /* C*x mod T = (C<<1) + wrap*T  exactly, so phi(C*x mod T) = x*phi(C) + wrap*phi(T) */
  std::vector<uint64_t> fT = E.F.of(E.R.T);
  for (long j = 0; j < m; j++) {
    bool wrap = getbit(C, E.R.r - 1);
    C = E.R.mulxmod(C);
    if (E.fm && E.fm->fire_horn()) E.fm->corrupt(C, E.R.r);
    addto(C, h[(size_t)j]);
    if (E.fm && E.fm->fire_horn()) E.fm->corrupt(C, E.R.r);
    if (E.checks_on) {
      /* predicted fold: phi(C*x mod T) = x*phi(C) + wrap*phi(x^s+1) */
      Poly fp = E.F.topoly(fC);
      std::vector<uint64_t> pred = E.F.of(shl(fp, 1));
      if (wrap) pred = E.F.addf(pred, fT);
      pred = E.F.addf(pred, E.F.of(h[(size_t)j]));
      fC = pred;
      E.st.fold_ops++;
    }
  }
  if (E.checks_on && !Fold::eq(fC, E.F.of(C))) { *ok = false; return Poly(); }
  *ok = true;
  return C;
}

/* ------------------------------------------------------------------ *
 *  The scan: mirrors factor.cpp's main loop (simplified: fixed q, no
 *  fineDDF -- fineDDF only re-runs the same machinery on a sub-interval).
 * ------------------------------------------------------------------ */

struct ScanResult {
  long  found_degree = -1;   /* degree of gcd found, -1 if none up to maxd */
  long  restarts     = 0;
  bool  aborted      = false;
  long  detections   = 0;    /* how many times a check fired */
};

static ScanResult scan(Engine &E, long k0, long m, long q, long maxd,
                       bool gerbicz, long window)
{
  ScanResult res;
  Ring &R = E.R;
  std::vector<Poly> h = init_h(E, k0 + 1, m);

  long k = k0 + 1;
  const long MAX_RESTART = 200;

  while (k <= maxd) {
    /* ---- checkpoint at interval start ---- */
    std::vector<Poly> h_ckpt = h;
    long k_ckpt = k;
    bool redo;
    Poly a;
    do {
      redo = false;
      h = h_ckpt; k = k_ckpt;
      a.clear(); setbit(a, 0, 1);
      std::vector<Poly> C;

      /* Gerbicz accumulators */
      Poly S;                       /* S_z = sum_j h_j^(z) */
      for (long j = 0; j < m; j++) addto(S, h[(size_t)j]);
      Poly D;                       /* telescoping window accumulator */
      Poly S_win0 = S;
      long wpos = 0;

      for (long z = 0; z < q && !redo; z++) {
        bool ok = true;
        Poly Cz = horner_C(E, h, m, &ok);
        if (!ok) { redo = true; break; }
        C.push_back(Cz);

        if (gerbicz) addto(D, S);

        /* ---- update the h-table: h[j] <- h[j]^(2^m) ---- */
        Poly Snext;
        for (long j = 0; j < m; j++) {
          for (long l = 0; l < m; l++) h[(size_t)j] = E.sqrmod(h[(size_t)j]);
          /* idle-memory corruption model */
          if (E.fm && E.fm->fire_idle()) E.fm->corrupt(h[(size_t)j], R.r);
          addto(Snext, h[(size_t)j]);
        }

        /* ---- Gerbicz check ---- */
        if (gerbicz) {
          wpos++;
          if (wpos == window || z == q - 1) {
            /* sigma(D) + D == S_win0 + S_next   (telescoped Gerbicz) */
            Poly sD = D;
            for (long l = 0; l < m; l++) sD = E.sqrmod_check(sD);
            Poly lhs = padd(sD, D);
            Poly rhs = padd(S_win0, Snext);
            if (!equal(lhs, rhs)) { redo = true; break; }
            D.clear(); S_win0 = Snext; wpos = 0;
          }
        }
        S = Snext;
      }
      if (redo) { E.st.rollbacks++; res.detections++; res.restarts++;
                  if (res.restarts > MAX_RESTART) { res.aborted = true; return res; }
                  continue; }

      /* ---- accumulate the products ---- */
      bool ok = true;
      std::vector<Poly> W = C;
      long zz = (long)W.size();
      while (zz > 1 && ok) {
        for (long j = 0; j < zz / 2; j++)
          W[(size_t)j] = E.mulmod(W[(size_t)j], W[(size_t)(zz - 1 - j)], &ok);
        zz = (zz + 1) / 2;
      }
      if (ok) a = E.mulmod(a, W[0], &ok);
      if (!ok) { redo = true; E.st.rollbacks++; res.detections++; res.restarts++;
                 if (res.restarts > MAX_RESTART) { res.aborted = true; return res; }
                 continue; }

      /* ---- GCD with Bezout certificate ---- */
      Poly u, v;
      Poly g = xgcd_poly(a, R.T, u, v);
      if (E.fm && E.fm->fire_gcd()) E.fm->corrupt(g, R.r);
      if (E.checks_on) {
        /* certificate: u*a + v*T == g exactly, verified by folding */
        auto lhs = E.F.addf(E.F.mulf(E.F.of(u), E.F.of(a)),
                            E.F.mulf(E.F.of(v), E.F.of(R.T)));
        E.st.fold_ops += 4;
        if (!Fold::eq(lhs, E.F.of(g))) { redo = true; E.st.rollbacks++;
                                         res.detections++; res.restarts++;
                                         if (res.restarts > MAX_RESTART)
                                           { res.aborted = true; return res; }
                                         continue; }
      }
      if (deg(g) > 0) { res.found_degree = deg(g); return res; }
    } while (redo);

    k += q * m;
  }
  return res;
}

/* ------------------------------------------------------------------ *
 *  Reference: smallest-degree factor, by direct DDF (slow but obvious)
 * ------------------------------------------------------------------ */
static long reference_smallest(const Ring &R, long maxd)
{
  Poly xp; setbit(xp, 1, 1);        /* x^(2^0) */
  Poly x = xp;
  for (long d = 1; d <= maxd; d++) {
    xp = R.sqrmod(xp);              /* x^(2^d) */
    Poly t = padd(xp, x);
    Poly g = gcd_poly(t, R.T);
    if (deg(g) > 0) return d;
  }
  return -1;
}


// gf2_gcd.h
// Host-side GCD and small-factor utilities for GF(2)[x] bit arrays
// (little-endian u64 words, bit i of word w = coefficient 64w + i).
//
// Two backends:
//   - naive: self-contained binary-polynomial Euclid, O(d^2/64).
//     Always available; fine for selftests and r up to ~10^6 bits.
//   - NTL (define HAVE_NTL, link gf2_gcd_ntl.cpp with -lntl -lgmp):
//     subquadratic HalfGCD + CanZass equal-degree factorization --
//     the same machinery factor.cpp uses.  REQUIRED for production r.
//
// Degree convention: return value is deg(gcd); 0 means gcd == 1
// (trivial).  gcd(0, b) = b.

#ifndef GF2_GCD_H
#define GF2_GCD_H
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

typedef uint64_t u64;

static inline int64_t poly_deg_w(const u64 *w, size_t nw) {
    for (size_t i = nw; i-- > 0;)
        if (w[i]) return (int64_t)(i * 64 + 63 - __builtin_clzll(w[i]));
    return -1;
}
static inline int64_t poly_deg_v(const std::vector<u64> &v) {
    return poly_deg_w(v.data(), v.size());
}

// A ^= B << sh  (A must be large enough; caller guarantees)
static inline void poly_xor_shift(std::vector<u64> &A, const std::vector<u64> &B,
                                  int64_t bdeg, u64 sh) {
    size_t ws = (size_t)(sh >> 6);
    unsigned bs = (unsigned)(sh & 63);
    size_t bw = (size_t)(bdeg >> 6) + 1;
    for (size_t j = bw; j-- > 0;) {
        u64 v = B[j];
        A[j + ws] ^= v << bs;
        if (bs && j + ws + 1 < A.size()) A[j + ws + 1] ^= v >> (64 - bs);
    }
}

// naive Euclid; returns deg(gcd); writes gcd words into *gout if given
static inline u64 poly_gcd_naive(const u64 *a, size_t aw,
                                 const u64 *b, size_t bw,
                                 std::vector<u64> *gout) {
    // Word counts beyond SIZE_MAX/16 are impossible for real inputs;
    // the explicit clamp also gives the compiler a provable range for
    // aw*8 and bw*8, which silences a -Wstringop-overflow false
    // positive that GCC >= 13 emits from const-propagated clones of
    // this function (observed with g++ 15.2 on Ubuntu 26.04).
    if (aw > (SIZE_MAX >> 4) || bw > (SIZE_MAX >> 4)) return 0;
    size_t nw = (aw > bw ? aw : bw) + 1;
    std::vector<u64> A(nw, 0), B(nw, 0);
    if (aw) memcpy(A.data(), a, aw * sizeof(u64));
    if (bw) memcpy(B.data(), b, bw * sizeof(u64));
    int64_t da = poly_deg_v(A), db = poly_deg_v(B);
    if (da < db) { A.swap(B); std::swap(da, db); }
    while (db >= 0) {
        while (da >= db) {
            poly_xor_shift(A, B, db, (u64)(da - db));
            da = poly_deg_v(A);
            if (da < 0) break;
        }
        A.swap(B);
        std::swap(da, db);
    }
    if (da < 0) return 0;              // gcd(0,0); treat as trivial
    if (gout) { gout->assign(A.begin(), A.begin() + (size_t)(da >> 6) + 1); }
    return (u64)da;
}

// does m divide g?  (gcd(m, g) == m)
static inline bool poly_divides(const std::vector<u64> &m,
                                const std::vector<u64> &g) {
    std::vector<u64> d;
    u64 dd = poly_gcd_naive(m.data(), m.size(), g.data(), g.size(), &d);
    return (int64_t)dd == poly_deg_v(m) && dd > 0;
}

// numeric (== lexicographic-at-equal-degree) comparison
static inline bool poly_less(const std::vector<u64> &x,
                             const std::vector<u64> &y) {
    size_t n = (x.size() > y.size() ? x.size() : y.size());
    for (size_t i = n; i-- > 0;) {
        u64 xv = i < x.size() ? x[i] : 0, yv = i < y.size() ? y[i] : 0;
        if (xv != yv) return xv < yv;
    }
    return false;
}

static inline std::string poly_hex(const std::vector<u64> &v) {
    int64_t d = poly_deg_v(v);
    if (d < 0) return "0";
    std::string s;
    for (int64_t n = d / 4; n >= 0; n--) {
        unsigned dig = (unsigned)((v[(size_t)(n >> 4)] >> ((n & 15) * 4)) & 0xF);
        s += "0123456789abcdef"[dig];
    }
    return s;
}

#ifdef HAVE_NTL
// defined in gf2_gcd_ntl.cpp
u64 poly_gcd_ntl(const u64 *a, size_t aw, const u64 *b, size_t bw,
                 std::vector<u64> *gout, u64 keep_max_bits);
// least (numeric) irreducible factor of g with degree exactly `target`;
// returns false if none (should not happen when preconditions hold)
bool edf_least_ntl(const u64 *g, size_t gw, u64 target,
                   std::vector<u64> &out);
// least factor degree of g AND its lex-least mask, via one CanZass
bool factor_min_degree_ntl(const u64 *g, size_t gw, u64 &min_deg,
                           std::vector<u64> &least_mask);
// does f divide x^r + x^s + 1?  (x^r mod f by repeated squaring: a few
// dozen modular squarings at deg f, independent of how f was obtained)
bool trinomial_divisible_ntl(const u64 *f, size_t fw, u64 r, u64 s);
#endif

#define GCD_KEEP_MAX_BITS ((u64)1 << 22)

// dispatch: NTL when compiled in, else naive
static inline u64 poly_gcd(const u64 *a, size_t aw, const u64 *b, size_t bw,
                           std::vector<u64> *gout) {
#ifdef HAVE_NTL
    return poly_gcd_ntl(a, aw, b, bw, gout, GCD_KEEP_MAX_BITS);
#else
    return poly_gcd_naive(a, aw, b, bw, gout);
#endif
}

// Fallback equal-degree "least factor" by ascending-mask enumeration.
// Valid because every irreducible factor of g has degree exactly
// `target` (see deep_scan_handoff.md C2): any degree-`target` divisor
// of g is then necessarily one of those irreducible factors.  Only
// usable for small target (2^(target-1) candidates).
static inline bool edf_least_enum(const std::vector<u64> &g, u64 target,
                                  std::vector<u64> &out) {
    if (target > 22) return false;
    for (u64 m = ((u64)1 << target) | 1; m < ((u64)2 << target); m += 2) {
        std::vector<u64> mv{m};
        if (poly_divides(mv, g)) { out = mv; return true; }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Hybrid HGCD gcd with pluggable large-multiplication offload (C4).
// The engine lives in gf2_gcd_ntl.cpp (needs NTL for CPU-side mults and
// the finishing gcd); callers provide a GpuMulHook whose mul() computes
// the full carry-less product out = a * b over GF(2)[x] -- typically on
// a GPU -- and return false to decline (engine then multiplies on the
// CPU).  All sizes in bits; word arrays are little-endian u64 as
// elsewhere in this codebase.
struct GpuMulHook {
    virtual bool mul(const u64 *a, size_t aw, u64 abits,
                     const u64 *b, size_t bw, u64 bbits,
                     std::vector<u64> &out) = 0;
    // Batched 2x2 matrix-vector product over GF(2)[x]:
    //   oa = m[0]*a + m[1]*b,  ob = m[2]*a + m[3]*b
    // A frequency-domain implementation transforms a and b once and
    // combines spectra (6 forward + 2 inverse transforms instead of the
    // 12 of four independent products).  bits == 0 marks a zero
    // operand.  Default declines; callers then compose from mul().
    struct Op { const u64 *w; size_t nw; u64 bits; };
    virtual bool mat2_apply(const Op m[4], const Op &a, const Op &b,
                            std::vector<u64> &oa, std::vector<u64> &ob) {
        (void)m; (void)a; (void)b; (void)oa; (void)ob;
        return false;
    }
    virtual ~GpuMulHook() {}
};

#ifdef HAVE_NTL
// gcd(a, b) with the same result/capture contract as poly_gcd_ntl.
// Degrees are reduced by recursive HGCD (multiplications >=
// hyb_gpu_min_bits offloaded through gm when non-null) until both fall
// to <= hyb_ntl_finish_bits, then NTL finishes and captures.  Any
// postcondition violation or lack of progress falls back to plain
// poly_gcd_ntl on the ORIGINAL inputs: the failure mode is slower,
// never wrong.
u64 poly_gcd_hybrid(const u64 *a, size_t aw, const u64 *b, size_t bw,
                    std::vector<u64> *gout, u64 keep_max_bits,
                    GpuMulHook *gm);
// CERTIFIED gcd (error checking; theory in errorcheck_handoff.md).  Same
// engine, but every transformation of the working pair,
//     (a, b) <- M (a, b)     and     (a, b) <- (b, a - q b),
// is verified on the host with cyclic folds carried from one step to the
// next, and the NTL finish is replaced by an extended gcd whose Bezout
// identity is verified the same way.  Each step being an exact
// GF(2)[x]-linear combination, every common divisor of the inputs divides
// every later pair, hence the result:  gcd(a, b) | g_returned.  So a
// returned degree of 0 PROVES the inputs coprime -- a fault anywhere in the
// gcd (CPU, RAM, or the GPU multiplications) cannot turn a hit into a
// miss.  (A fault could at worst add a spurious factor to g; the caller's
// check that a reported factor divides T rejects that.)  Cost: linear-time
// folds plus the extended finish of the last hyb_ntl_finish_bits -- a few
// percent of the gcd.
//   gm may be null: the chain then runs on CPU multiplications.
//   fa_expect / fb_expect (may be null): folds the inputs must match.
//   *status receives one of the GCD_CERT_ codes.
struct EcFold;
enum { GCD_CERT_OK = 0,      // certified
       GCD_CERT_INPUT = 1,   // an input does not match its expected fold
       GCD_CERT_FAIL = 2,    // a step identity failed: fault during the gcd
       GCD_CERT_NONE = 3 };  // not certified (builds without NTL only: if
                             // the chain declines, the gcd is certified by
                             // one verified XGCD instead)
u64 poly_gcd_certified(const u64 *a, size_t aw, const u64 *b, size_t bw,
                       std::vector<u64> *gout, u64 keep_max_bits,
                       GpuMulHook *gm, const EcFold *fa_expect,
                       const EcFold *fb_expect, int *status);
// Certify with ONE verified NTL extended gcd of the whole problem instead of
// the chain (tsfactor --ecg-xgcd).  About 1.4x NTL's plain gcd; kept as an
// independent reference and for A/B timing (gcdbench).
extern bool hyb_cert_xgcd;
// Apply each half-gcd matrix to the LOW part of the operands only, reusing
// the high part the recursion already reduced (default on; roughly halves
// the engine's multiplication work).  false restores the full application,
// for A/B timing (tsfactor --gcd-full-apply, gcdbench).
extern bool hyb_split_apply;
extern bool hyb_force_fallback;  // TEST: make the chain decline
// Threads for the independent products of ONE gcd (CPU multiplications
// only): the four products of a matrix application and the eight of a
// matrix product run side by side when both operands have at least
// hyb_mt_min_bits bits.  Default 1.  Needs a thread-safe NTL (NTL_THREADS,
// the default configuration; hyb_threads_ok() reports it) and gf2x >= 1.2.
extern int hyb_threads;
extern u64 hyb_mt_min_bits;
bool hyb_threads_ok();
// The bottom of the half-gcd recursion.  Pairs of at most hyb_base_bits bits
// are reduced by the word-level routine of gf2_hgcd_base.h (the polynomial
// form of Lehmer's method: exact, allocation-free, ~31 quotient bits per
// pass).  hyb_base_ref = true selects the old bit-level loop instead -- the
// reference the word-level routine must reproduce exactly (gcdstress).
extern u64 hyb_base_bits;        // default 2048; at most 16320
extern bool hyb_base_ref;
void hyb_get_base_stats(u64 &calls, u64 &division_steps);
// TEST (single-threaded): half-gcd of (a, b) with the word-level base case
// and with the reference one; true iff reduced pair and transform agree
// exactly and satisfy the half-gcd contract.
bool hyb_base_selfcheck(const u64 *a, size_t aw, const u64 *b, size_t bw);
// true if the engine calls gf2x directly for its CPU products (built with
// -DHAVE_GF2X); otherwise they go through NTL::mul with conversions
bool hyb_gf2x_direct();
extern long hyb_fault_at;        // TEST: corrupt the N-th verified step
void hyb_fault_reset();          // TEST: restart the step counter
long hyb_fault_count();          // TEST: verified steps seen so far
extern u64 hyb_gpu_min_bits;     // offload mults with an operand >= this
extern u64 hyb_ntl_finish_bits;  // hand the pair to NTL below this
void hyb_get_stats(u64 &gpu_mults, u64 &cpu_mults, u64 &fallbacks);
void hyb_reset_stats();
bool ntl_gf2x_backed();          // NTL built with NTL_GF2X_LIB?
#endif

#endif // GF2_GCD_H

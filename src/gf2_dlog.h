// gf2_dlog.h
// Table-light discrete logarithms in GF(2^d), d <= 55, for coarse_sieve
// (its only sieve path since round 6).  The sieve needs, per cyclotomic
// class, one Zech logarithm log(g^u + 1); instead of a 2^d-entry log table
// (or its bucketed slices) it computes that one log directly:
//
//   tower  (even d = 2m): GF(2^d) = K(w), K = GF(2^m), w = x.  Every element
//          is a + b*w with a, b in K; for b != 0 it equals b*(c + w),
//          c = a/b, so  log e = (2^m+1)*logK(b) + T[logK(c)].  The split
//          e -> (a,b) is GF(2)-linear (byte tables); T has 2^m entries.
//          logK itself is either a 2^m-entry table (classic, prime m) or
//          (round 4) the same construction one level down, over a subfield
//          K_n of K, with tables of ~2^n entries: then a log costs one read
//          of a large table (T) instead of three.
//   ph     (any d): Pohlig-Hellman over a coprime split M = P_1*...*P_n of
//          M = 2^d - 1 (n <= 4): e^(E_i) lies in the order-P_i subgroup and
//          a hash table of 16-byte slots (with baby-step/giant-step when P_i
//          exceeds the table budget) gives v mod P_i; CRT gives v.  The
//          exponents E_i = c_i * M/P_i are chosen (over c_i coprime to P_i,
//          and rotations) to have few set bits in a short window: few
//          general multiplies, short squaring chain.  The planner picks the
//          split by estimated cost per class, for the kernel that will run
//          it (cost models below).
//   rr     (odd d >= 5, round 5): rational reconstruction -- half an
//          extended Euclid writes e = r/t with deg r, deg t <= d/2, so
//          log e = L[r] - L[t] from a table of the logs of all polynomials
//          of degree <= d/2 (see "RR logs" below).
//
// Also here: carry-less arithmetic that is ~8x cheaper than gf_mul's
// bit-serial loops (integer-multiply clmul with 4-bit holes, BearSSL-style;
// two-fold reduction by the sparse low part of the modulus); bit-sliced
// arithmetic for 32 elements at once (round 3: one u32 per coefficient,
// the modulus fixed at compile time); the FKM enumeration of class
// representatives (Lyndon words) and their reciprocal pairing (round 6,
// gf2f_pair_role); and the u64 helpers the sweep needs (Mersenne
// mulmod/powmod, exact-division tests).
// Everything is plain host+device code with no warp intrinsics, so
// cuda_emul.h validates it exactly (coarse_sieve's batched kernel copies the
// small tower tables to shared memory first -- device-only code).

#ifndef GF2_DLOG_H
#define GF2_DLOG_H
#include "gf2_wide_field.h"
#include <utility>        // std::integer_sequence (bit-sliced arithmetic)

// ---------------------------------------------------------------------------
// fast field arithmetic (d <= 55, modulus q = x^d + qlow, deg(qlow) <= 16)

// 32x32 -> 64 carry-less product via integer multiplies.  Each operand is
// split into 4 bit-classes (bits = k mod 4); an integer product of two
// classes has at most 8 terms per output position, so no carry ever
// crosses into the next position of the same class, and bit p of the
// product is the parity of its count.  XOR the four products landing in
// class k, then mask class k.
GF2_HD u64 gf2f_clmul32(u32 x, u32 y) {
    const u32 m0 = 0x11111111u, m1 = 0x22222222u, m2 = 0x44444444u,
              m3 = 0x88888888u;
    u32 x0 = x & m0, x1 = x & m1, x2 = x & m2, x3 = x & m3;
    u32 y0 = y & m0, y1 = y & m1, y2 = y & m2, y3 = y & m3;
    u64 z0 = ((u64)x0 * y0) ^ ((u64)x1 * y3) ^ ((u64)x2 * y2) ^ ((u64)x3 * y1);
    u64 z1 = ((u64)x0 * y1) ^ ((u64)x1 * y0) ^ ((u64)x2 * y3) ^ ((u64)x3 * y2);
    u64 z2 = ((u64)x0 * y2) ^ ((u64)x1 * y1) ^ ((u64)x2 * y0) ^ ((u64)x3 * y3);
    u64 z3 = ((u64)x0 * y3) ^ ((u64)x1 * y2) ^ ((u64)x2 * y1) ^ ((u64)x3 * y0);
    return (z0 & 0x1111111111111111ULL) | (z1 & 0x2222222222222222ULL) |
           (z2 & 0x4444444444444444ULL) | (z3 & 0x8888888888888888ULL);
}

// 64x64 -> 128 carry-less product, one Karatsuba level over 32-bit halves.
GF2_HD void gf2f_clmul64(u64 a, u64 b, u64 *hi, u64 *lo) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
    u64 l = gf2f_clmul32(a0, b0), h = gf2f_clmul32(a1, b1);
    u64 m = gf2f_clmul32(a0 ^ a1, b0 ^ b1) ^ l ^ h;
    *lo = l ^ (m << 32);
    *hi = h ^ (m >> 32);
}

GF2_HD int gf2f_ctz64(u64 x) {              // x != 0
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}

GF2_HD int gf2f_ctz32(u32 x) {              // x != 0
#if defined(__CUDA_ARCH__)
    return __ffs((int)x) - 1;
#else
    return __builtin_ctz(x);
#endif
}

GF2_HD int gf2f_clz64(u64 x) {              // x != 0
#if defined(__CUDA_ARCH__)
    return __clzll((long long)x);
#else
    return __builtin_clzll(x);
#endif
}

// p * qlow for p < 2^(64 - deg qlow): XOR of shifted copies, one per set
// bit of qlow (2..6 for every d <= 55; uniform across threads).  qlow has
// degree <= 8 here (< 2^25 whenever gf2f_usable holds), so the loop runs on
// 32-bit control: ~10 instead of ~18 instructions per term on the GPU.
GF2_HD u64 gf2f_mul_low(u64 p, u64 qlow) {
    u64 t = 0;
    for (u32 ql = (u32)qlow; ql; ql &= ql - 1) t ^= p << gf2f_ctz32(ql);
    return t;
}

// Reduce hi:lo (degree <= 2d-2) mod x^d + qlow.  Two folds suffice when
// 2*deg(qlow) - 2 < d and d + deg(qlow) <= 64 (checked by gf2f_usable).
GF2_HD u64 gf2f_reduce(u64 hi, u64 lo, u64 qlow, int d) {
    const u64 mask = ((u64)1 << d) - 1;
    u64 ph = (lo >> d) | (hi << (64 - d));      // P >> d, degree <= d-2
    u64 t = gf2f_mul_low(ph, qlow);             // degree <= d-2+deg(qlow)
    u64 th = t >> d;                            // degree <= deg(qlow)-2
    return (lo & mask) ^ (t & mask) ^ gf2f_mul_low(th, qlow);
}

GF2_HD u64 gf2f_mul(u64 a, u64 b, u64 qlow, int d) {
    u64 hi, lo;
    gf2f_clmul64(a, b, &hi, &lo);
    return gf2f_reduce(hi, lo, qlow, d);
}

// bit-spread of a 32-bit value: bit i -> bit 2i
GF2_HD u64 gf2f_spread32(u32 v) {
    u64 x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

GF2_HD u64 gf2f_sqr(u64 a, u64 qlow, int d) {
    return gf2f_reduce(gf2f_spread32((u32)(a >> 32)),
                       gf2f_spread32((u32)a), qlow, d);
}

// g^idx for g = x, from gpow2[t] = g^(2^t) (fast multiply)
GF2_HD u64 gf2f_antilog(u64 idx, u64 qlow, int d, const u64 *gpow2) {
    u64 v = 1;
    for (int t = 0; idx; idx >>= 1, t++)
        if (idx & 1) v = gf2f_mul(v, gpow2[t], qlow, d);
    return v;
}

// base^e (fast multiply/square)
GF2_HD u64 gf2f_pow(u64 base, u64 e, u64 qlow, int d) {
    u64 r = 1;
    for (; e; e >>= 1) {
        if (e & 1) r = gf2f_mul(r, base, qlow, d);
        if (e > 1) base = gf2f_sqr(base, qlow, d);
    }
    return r;
}

// host check: can the fast path handle this modulus?  (d <= 55: the
// packed best[] key's limit; the two-fold reduction holds there for the
// smallest primitive polynomials, whose low part has degree <= 8)
static inline bool gf2f_usable(u64 q, int d) {
    if (d < 2 || d > 55) return false;
    u64 ql = q ^ ((u64)1 << d);
    if (!ql) return false;
    int dl = 63 - __builtin_clzll(ql);
    return 2 * dl - 2 < d && d + dl <= 64 && dl < 32;   // dl < 32: mul_low
}

// ---------------------------------------------------------------------------
// u64 helpers for exponent arithmetic modulo M = 2^d - 1

// (a*b) mod M, a,b < M, M = 2^d - 1 (d <= 63): fold the product at bit d.
GF2_HD u64 gf2f_mulmod_m(u64 a, u64 b, int d, u64 M) {
#if defined(__CUDA_ARCH__)
    u64 lo = a * b, hi = __umul64hi(a, b);
#else
    u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
    u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
    u64 lo = (mid << 32) | (u32)p00;
    u64 hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
    u64 t = (lo & M) + ((lo >> d) | (hi << (64 - d)));
    t = (t & M) + (t >> d);
    if (t >= M) t -= M;
    return t;
}

// a^e mod M (left-to-right; the exponent is uniform across threads)
GF2_HD u64 gf2f_powmod_m(u64 a, u64 e, int d, u64 M) {
    if (e == 0) return 1 % M;
    u64 r = a;
    for (int b = 62 - gf2f_clz64(e); b >= 0; b--) {   // below the top bit
        r = gf2f_mulmod_m(r, r, d, M);
        if ((e >> b) & 1) r = gf2f_mulmod_m(r, a, d, M);
    }
    return r;
}

// Exact-division test for an odd divisor p: with pinv = p^-1 mod 2^64 and
// plim = floor((2^64-1)/p),  p | x  <=>  x*pinv <= plim,  and then
// x/p == x*pinv.  No 64-bit division (software-emulated on GPUs).
struct OddDiv { u64 p, pinv, plim; };

static inline OddDiv odd_div_make(u64 p) {
    OddDiv o;
    o.p = p;
    u64 inv = p;                                // Newton: 5 steps -> 64 bits
    for (int i = 0; i < 5; i++) inv *= 2 - p * inv;
    o.pinv = inv;
    o.plim = ~0ULL / p;
    return o;
}

GF2_HD bool odd_div_test(u64 x, const OddDiv &o, u64 *quot) {
    u64 qq = x * o.pinv;
    if (qq > o.plim) return false;
    *quot = qq;
    return true;
}

// Round 5: exact arithmetic for the irregular classes (see coarse_sieve.cu,
// "irregular classes").  No division anywhere.

// 2-adic inverse of an odd g: g^-1 mod 2^64 (g*g == 1 mod 8 gives 3 bits,
// each Newton step doubles them: 3, 6, 12, 24, 48, 96).  For g | x the
// exact quotient x/g is then simply x * g^-1 (mod 2^64).
GF2_HD u64 inv_2adic(u64 g) {
    u64 x = g;
    for (int i = 0; i < 5; i++) x *= 2 - g * x;
    return x;
}

// floor(a*b / M) for M = 2^d - 1, a < 2^d, b <= M: with a*b = A*2^d + B,
// a*b = A*M + (A + B), and A < b <= M, B <= M, so the floor is A or A + 1.
GF2_HD u64 floor_mul_div_m(u64 a, u64 b, int d, u64 M) {
#if defined(__CUDA_ARCH__)
    const u64 lo = a * b, hi = __umul64hi(a, b);
#else
    const u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
    const u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
    const u64 lo = (mid << 32) | (u32)p00;
    const u64 hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
    const u64 A = (lo >> d) | (hi << (64 - d));     // a*b < 2^(2d): fits
    const u64 B = lo & M;
    return A + (A + B >= M ? 1 : 0);
}

// x mod n for x < M, n = M/g (so x/n = x*g/M exactly): x - n*floor(x*g/M)
GF2_HD u64 mod_divisor_m(u64 x, u64 n, u64 g, int d, u64 M) {
    return x - n * floor_mul_div_m(x, g, d, M);
}

// omega(2^d - 1), the number of distinct primes of M, for d < 64 (compile-
// time loop bounds for the per-class divisibility tests; the host checks
// it against its own factorization)
GF2_HD constexpr int gf2_npf(int d) {
    return (int)"\0\0\1\1\2\1\2\1\3\2\3\2\4\1\3\3\4\1\4\1\5\3\4\2\6\3\3\3\6\3\6\1"
                "\5\4\3\4\10\2\3\4\7\2\6\3\7\6\4\3\11\2\7\5\7\3\6\6\10\4\6\2\13\1\3\6"[d];
}

// ---------------------------------------------------------------------------
// Is u the least of its d-bit rotations?  Requires u odd and u < 2^(d-1)
// (the only candidates the sweep offers).
//
// Let z >= 1 be the number of leading zeros of u as a d-bit string.  As u is
// odd, no zero run of the cyclic string passes through bit 0, so the runs
// are ordinary runs of ones in w = ~u & M.  Any rotation smaller than u
// starts with >= z zeros: it starts either in a run of >= z+1 zeros (then it
// is below 2^(d-1-z) <= u outright) or at another run of exactly z zeros
// (then it has to be compared).  zero_runs(w, k) has bit b set iff bits
// b..b+k-1 of u are all zero; it takes log2(k) shift-ANDs.
//
// This replaces a loop over all d-1 rotations whose data-dependent trip
// count kept warps busy for ~30 iterations per u, most of the sweep's time
// at large d; the result is the same for every such u (checked against
// gf2f_min_rotation_ref exhaustively for d <= 24 and on random u for every
// d <= 55 by t_rot.cpp / gf2_dlog_selftest.cpp, and on the device by
// --selftest-logs).
GF2_HD u64 gf2f_zero_runs(u64 w, int k) {    // k >= 1
    int m = 1;
    while (2 * m <= k) { w &= w >> m; m *= 2; }
    if (k > m) w &= w >> (k - m);
    return w;
}

GF2_HD bool gf2f_is_min_rotation(u64 u, int d, u64 M) {
    const int z = gf2f_clz64(u) - (64 - d);    // leading zeros, 1..d-1
    u64 y = gf2f_zero_runs(~u & M, z);         // starts of runs of >= z zeros
    if (y & (y >> 1)) return false;            // some run has >= z+1 zeros
    y &= ~((u64)1 << (d - z));                 // u's own leading run
    for (; y; y &= y - 1) {                    // the other runs of exactly z
        const int i = d - z - gf2f_ctz64(y);   // rotation putting it on top
        if ((((u << i) | (u >> (d - i))) & M) < u) return false;
    }
    return true;
}

// reference: compare with all d-1 rotations
GF2_HD bool gf2f_min_rotation_ref(u64 u, int d, u64 M) {
    u64 x = u;
    for (int i = 1; i < d; i++) {
        x = ((x << 1) | (x >> (d - 1))) & M;
        if (x < u) return false;
    }
    return true;
}

GF2_HD int gf2f_popc64(u64 x) {
#if defined(__CUDA_ARCH__)
    return __popcll((unsigned long long)x);
#else
    return __builtin_popcountll(x);
#endif
}

GF2_HD int gf2f_popc32(u32 x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}

// the least of the d-bit rotations of w (reference loop; rare callers only)
GF2_HD u64 gf2f_min_rotation(u64 w, int d, u64 M) {
    u64 best = w, x = w;
    for (int i = 1; i < d; i++) {
        x = ((x << 1) | (x >> (d - 1))) & M;
        if (x < best) best = x;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Reciprocal pairs (round 6).  The class of -u = ~u & M (the complement)
// holds the roots g^-j of the reciprocal polynomial, and since
// 1 + g^-u = g^-u (1 + g^u),  Z(-u) = Z(u) - u:  one Zech log serves both
// classes.  For a Lyndon word u (a class representative):
//   1: u is its pair's canonical member: process u and its partner;
//   0: the partner rep(~u) is canonical: skip u (the partner marks it);
//   2: u is its own partner (a self-reciprocal class): process u alone.
// Canonical: fewer ones than zeros (complementing swaps the counts, so for
// odd d exactly one member qualifies); with equal counts (even d), the
// smaller of u and rep(~u).  A Lyndon word starts with its longest zero run
// and rep(~u) with u's longest run of ones, so comparing those decides
// nearly every tie; the rest compare with the least rotation of ~u.
//
// Round 7, the same rule computed faster: the least rotation of ~u starts
// at one of its longest zero runs, i.e. at one of u's longest runs of ones
// (of length z1 = z0 here), so only those rotations are compared -- one or
// two, typically -- instead of all d.  Round 6's version below (the
// reference) ran a d-step loop for ~3% of the words at even d, inside the
// enumeration loop, where a warp waits for its slowest lane (in 4-7% of the
// loop's iterations at d=44, by tools/fillsim.cpp).  Identical roles:
// checked against the reference on
// every Lyndon word of length <= 36 and on 4.5e10 of length 44 (t_pair.cpp,
// gf2_dlog_selftest.cpp part 9, and on the device by --selftest).
GF2_HD int gf2f_pair_role(u64 u, int d, u64 M) {
    const int ones = gf2f_popc64(u);
    if (2 * ones < d) return 1;
    if (2 * ones > d) return 0;
    const int z0 = gf2f_clz64(u) - (64 - d);    // u's leading = longest 0-run
    // bit b set iff bits b..b+z0-1 of u are ones (u starts with 0 and ends
    // with 1: no run of ones wraps around)
    u64 y = gf2f_zero_runs(u, z0);
    if (!y) return 1;                           // u's longest 1-run < z0
    if (y & (y >> 1)) return 0;                 // ... or > z0
    // equal: rep(~u) starts at one of the runs of exactly z0 ones (one bit
    // of y each, at its low end); rotate it to the top (s in 1..d-z0)
    const u64 w = ~u & M;
    u64 rc = M;
    for (; y; y &= y - 1) {
        const int s = d - z0 - gf2f_ctz64(y);
        const u64 t = ((w << s) | (w >> (d - s))) & M;
        if (t < rc) rc = t;
    }
    return u < rc ? 1 : u > rc ? 0 : 2;
}

// round 6's computation of the same role (the reference for the tests)
GF2_HD int gf2f_pair_role_ref(u64 u, int d, u64 M) {
    const int ones = gf2f_popc64(u);
    if (2 * ones < d) return 1;
    if (2 * ones > d) return 0;
    const int z0 = gf2f_clz64(u) - (64 - d);    // u's leading = longest 0-run
    int z1 = 0;                                 // u's longest 1-run (u starts
    for (u64 x = u; x; x &= x >> 1) z1++;       // with 0: no wrap-around)
    if (z0 != z1) return z0 > z1 ? 1 : 0;
    const u64 rc = gf2f_min_rotation(~u & M, d, M);
    return u < rc ? 1 : u > rc ? 0 : 2;
}

// the number of degree-d classes = irreducible polynomials of degree d over
// GF(2) = binary Lyndon words of length d: (1/d) sum_{e|d} mu(e) 2^(d/e)
static inline u64 gf2_irreducible_count(int d) {
    long long s = 0;
    for (int e = 1; e <= d; e++) {
        if (d % e) continue;
        int mu = 1, x = e;                      // Moebius function of e
        for (int p = 2; p * p <= x; p++) {
            if (x % p) continue;
            x /= p;
            if (x % p == 0) { mu = 0; break; }
            mu = -mu;
        }
        if (mu && x > 1) mu = -mu;
        s += (long long)mu * (long long)((u64)1 << (d / e));
    }
    return (u64)(s / d);
}

// ---------------------------------------------------------------------------
// Lyndon words in increasing order (Fredricksen-Kessler-Maiorana).
//
// The degree-d classes are exactly the binary Lyndon words of length n = d
// (aperiodic, and least among their rotations), read MSB first; as numbers
// they are the odd representatives u < 2^(d-1).  FKM visits the
// pre-necklaces of length n in lexicographic = numeric order; a pre-
// necklace has a "period" p, and the Lyndon words are those with p == n.
// There are about twice as many pre-necklaces as Lyndon words, so a sweep
// costs ~2 cheap steps per class -- instead of a test of every odd u
// (~12 per class at d=47, most of them where there are almost no classes:
// half of all u start with "01" and hold only the classes without "00").
//
// The top n bits of the infinite repetition of the j-bit word P:
GF2_HD u64 fkm_extend(u64 P, int j, int n) {
    if (j >= n) return P;
    u64 X = P << (64 - j);
    for (int f = j; f < n; f *= 2) X |= X >> f;
    return X >> (64 - n);
}

// The least pre-necklace >= lo (lo < 2^n); *p receives its period.
// FKM's recurrence a_i >= a_(i-p) fails at the first i with a_i < a_(i-p);
// the least pre-necklace above then repeats a_1..a_p (its prefix up to i-1
// already does).
GF2_HD u64 fkm_seek(u64 lo, int n, int *p) {
    int pp = 1;
    for (int i = 2; i <= n; i++) {
        const int ai = (int)((lo >> (n - i)) & 1);
        const int aq = (int)((lo >> (n - i + pp)) & 1);
        if (ai < aq) { *p = pp; return fkm_extend(lo >> (n - pp), pp, n); }
        if (ai > aq) pp = i;
    }
    *p = pp;
    return lo;
}

// The next pre-necklace after w (w < 2^n - 1); *p receives its period:
// raise the last 0 to 1 and repeat the prefix up to it.
GF2_HD u64 fkm_next(u64 w, int n, int *p) {
    const int b = gf2f_ctz64(~w);              // last 0, as a bit index (< n)
    *p = n - b;
    return fkm_extend((w >> b) | 1, n - b, n);
}

// (device pass only: the host compiler of an nvcc build would warn about
// the unknown pragma under -Wall)
#if defined(__CUDA_ARCH__)
#define GF2D_UNROLL _Pragma("unroll")
#else
#define GF2D_UNROLL
#endif

// ---------------------------------------------------------------------------
// The moduli as compile-time constants: q_d = x^d + gf2_qlow(d) is the
// smallest primitive polynomial of degree d, i.e. exactly what
// find_primitive_poly(d) returns (the host code checks this before using
// the bit-sliced kernels, which need the taps at compile time).
GF2_HD constexpr u64 gf2_qlow(int d) {
    switch (d) {
    case  2: return 0x3;   case  3: return 0x3;   case  4: return 0x3;
    case  5: return 0x5;   case  6: return 0x3;   case  7: return 0x3;
    case  8: return 0x1d;  case  9: return 0x11;  case 10: return 0x9;
    case 11: return 0x5;   case 12: return 0x53;  case 13: return 0x1b;
    case 14: return 0x2b;  case 15: return 0x3;   case 16: return 0x2d;
    case 17: return 0x9;   case 18: return 0x27;  case 19: return 0x27;
    case 20: return 0x9;   case 21: return 0x5;   case 22: return 0x3;
    case 23: return 0x21;  case 24: return 0x1b;  case 25: return 0x9;
    case 26: return 0x47;  case 27: return 0x27;  case 28: return 0x9;
    case 29: return 0x5;   case 30: return 0x53;  case 31: return 0x9;
    case 32: return 0xaf;  case 33: return 0x53;  case 34: return 0xe7;
    case 35: return 0x5;   case 36: return 0x77;  case 37: return 0x3f;
    case 38: return 0x63;  case 39: return 0x11;  case 40: return 0x39;
    case 41: return 0x9;   case 42: return 0x3f;  case 43: return 0x59;
    case 44: return 0x65;  case 45: return 0x1b;  case 46: return 0x12f;
    case 47: return 0x21;  case 48: return 0xb7;  case 49: return 0x71;
    case 50: return 0x1d;  case 51: return 0x4b;  case 52: return 0x9;
    case 53: return 0x47;  case 54: return 0x7d;  case 55: return 0x47;
    default: return 0;
    }
}

GF2_HD constexpr int gf2_deg(u64 x) {
    int d = -1;
    while (x) { x >>= 1; d++; }
    return d;
}

// ---------------------------------------------------------------------------
// Bit-sliced arithmetic: 32 field elements at once, one u32 per coefficient
// ("plane"): bit k of plane i is coefficient i of element k.  A product is
// then sum_(i+j=m) a_i & b_j per output plane -- one 3-input LOP3 per term,
// d^2 for 32 products, against ~250 instructions for ONE scalar product --
// and reduction mod q_D only XORs whole planes.  A squaring is a fixed
// permutation of planes plus that reduction: nearly free.  D is a template
// parameter so every plane index is a compile-time constant and the planes
// live in registers (3D+ registers for a product: the PH kernels use ~160-
// 250 registers per thread, i.e. 8-12 warps per SM, which is plenty for
// LOP3 streams).

// Every plane index must be a compile-time constant for the planes to stay
// in registers.  Loops would need to be fully unrolled for that, and the
// unrollers' size limits are exceeded at large D (clang gives up at D = 55),
// so the plane computations are expanded by folds over index sequences
// instead: no loops at all in a product or a square.
template <int D> struct BSRed {               // reduction by q_D = x^D + QL
    static constexpr u64 QL = gf2_qlow(D);
    static constexpr int T = gf2_deg(QL);       // taps: x^D = QL, deg QL = T
    static constexpr int NH = T > 1 ? T - 1 : 1;  // pending planes D..D+T-2
};

// x^M = x^(M-D) * QL (M >= D): tap S sends plane M to plane M - D + S
// (the helpers take plain pointers: with array references, GCC's identical-
// code folding of the empty taps across D produces bogus -Warray-bounds
// warnings; on the device everything is inlined and indexed by constants)
template <int D, int M, int S>
GF2_HD void bs_tap(u32 t, u32 *r, u32 *hp) {
    if constexpr ((BSRed<D>::QL >> S) & 1) {
        constexpr int g = M - D + S;
        if constexpr (g >= D) hp[g - D] ^= t; else r[g] ^= t;
    }
}

template <int D, int M, int... S>
GF2_HD void bs_taps(u32 t, u32 *r, u32 *hp, std::integer_sequence<int, S...>) {
    (bs_tap<D, M, S>(t, r, hp), ...);
}

// fold plane M (value t) into the result; a plane M >= D first collects
// what higher planes folded into it
template <int D, int M>
GF2_HD void bs_fold_plane(u32 t, u32 *r, u32 *hp) {
    if constexpr (M >= D) {
        if constexpr (M - D < BSRed<D>::T - 1) t ^= hp[M - D];
        bs_taps<D, M>(t, r, hp, std::make_integer_sequence<int, BSRed<D>::T + 1>{});
    } else {
        r[M] ^= t;
    }
}

// plane M of a*b: XOR of a[i] & b[M-i] over I0 <= i <= I0 + #J - 1
template <int D, int M, int I0, int... J>
GF2_HD u32 bs_conv(const u32 *a, const u32 *b, std::integer_sequence<int, J...>) {
    return (0u ^ ... ^ (a[I0 + J] & b[M - I0 - J]));
}

template <int D, int M>
GF2_HD void bs_mul_plane(const u32 *a, const u32 *b, u32 *r, u32 *hp) {
    constexpr int I0 = M >= D ? M - D + 1 : 0, I1 = M < D ? M : D - 1;
    bs_fold_plane<D, M>(
        bs_conv<D, M, I0>(a, b, std::make_integer_sequence<int, I1 - I0 + 1>{}),
        r, hp);
}

template <int D, int... K>
GF2_HD void bs_mul_planes(const u32 *a, const u32 *b, u32 *r, u32 *hp,
                          std::integer_sequence<int, K...>) {
    (bs_mul_plane<D, 2 * D - 2 - K>(a, b, r, hp), ...);   // M = 2D-2 .. 0
}

// r = a * b  (r must not alias a or b)
template <int D>
GF2_HD void bs_mul(const u32 (&a)[D], const u32 (&b)[D], u32 (&r)[D]) {
    u32 hp[BSRed<D>::NH];
    GF2D_UNROLL
    for (int k = 0; k < BSRed<D>::NH; k++) hp[k] = 0;
    GF2D_UNROLL
    for (int k = 0; k < D; k++) r[k] = 0;
    bs_mul_planes<D>(a, b, r, hp, std::make_integer_sequence<int, 2 * D - 1>{});
}

// plane M of a^2 is a[M/2] for even M and 0 for odd M -- but an odd M >= D
// still has to pass down what higher planes folded into it
template <int D, int M>
GF2_HD void bs_sqr_plane(const u32 *a, u32 *r, u32 *hp) {
    if constexpr (M & 1) {
        if constexpr (M >= D) bs_fold_plane<D, M>(0u, r, hp);
    } else {
        bs_fold_plane<D, M>(a[M / 2], r, hp);
    }
}

template <int D, int... K>
GF2_HD void bs_sqr_planes(const u32 *a, u32 *r, u32 *hp,
                          std::integer_sequence<int, K...>) {
    (bs_sqr_plane<D, 2 * D - 2 - K>(a, r, hp), ...);       // M = 2D-2 .. 0
}

// r = a^2  (r must not alias a)
template <int D>
GF2_HD void bs_sqr(const u32 (&a)[D], u32 (&r)[D]) {
    u32 hp[BSRed<D>::NH];
    GF2D_UNROLL
    for (int k = 0; k < BSRed<D>::NH; k++) hp[k] = 0;
    GF2D_UNROLL
    for (int k = 0; k < D; k++) r[k] = 0;
    bs_sqr_planes<D>(a, r, hp, std::make_integer_sequence<int, 2 * D - 1>{});
}

// g = g * x in the slots selected by sel, unchanged elsewhere
template <int D>
GF2_HD void bs_mulx_sel(u32 (&g)[D], u32 sel) {
    constexpr u64 QL = gf2_qlow(D);
    const u32 top = g[D - 1] & sel;
    GF2D_UNROLL
    for (int k = D - 1; k >= 1; k--) g[k] = (g[k - 1] & sel) | (g[k] & ~sel);
    g[0] &= ~sel;
    GF2D_UNROLL
    for (int s = 0; s < D; s++)
        if ((QL >> s) & 1) g[s] ^= top;
}

// one step of g^u: G = G^2, then G = G * x in the slots selected by sel
template <int D>
GF2_HD void bs_step_g(u32 (&G)[D], u32 sel) {
    u32 R[D];
    bs_sqr<D>(G, R);
    GF2D_UNROLL
    for (int k = 0; k < D; k++) G[k] = R[k];
    bs_mulx_sel<D>(G, sel);
}

// Round 9: a hint to bring the cache line of a thread's local-memory
// address (a local array element) into L1 ahead of its load -- CCTLL.PF1:
// no register, nothing waits for it.  A no-op on the host.
GF2_HD void pf_local(const void *p) {
#if defined(__CUDA_ARCH__)
    asm volatile("{\n\t.reg .u64 a;\n\tcvta.to.local.u64 a, %0;\n\t"
                 "prefetch.local.L1 [a];\n\t}" :: "l"(p));
#else
    (void)p;
#endif
}

// G = g^u per slot, g = x; U holds the exponents as planes (bit k of U[b] =
// bit b of u_k), every u_k < 2^(D-1).  Left to right: square, then multiply
// by x where the bit is set -- a plane shift, so no general multiplies.
// The loop over b stays rolled on purpose: unrolled it would be ~15k
// instructions of straight-line code per kernel, far beyond the instruction
// cache; rolled, U[b] lives in local memory (the batched kernels' U).
// Round 9: the planes are loaded three steps ahead.  Round 8's version
// loaded U[b] ~25 instructions before its use, and the round-7 ncu exports
// put 24% (RR, d=47) and 14% (tower, d=44) of all the kernels' stall samples
// on that one wait: the load misses L1 (the batch arrays evict it) and
// queues behind the DRAM traffic.  Four registers in turn, each reloaded
// right after its use (a copy of a pending load would wait for it), so a
// plane has three steps (~300 instructions) to arrive -- in the source:
// ptxas sinks the four loads to the end of the loop body, which leaves the
// first plane of each group ~55 instructions (sm_75, d=53).  So with pf
// (the sieve's default, COARSE_PREFETCH) each group also prefetches into
// L1 the planes of the group after next -- ptxas sinks those too, next to
// the loads, which puts them one group (~470 instructions) ahead of the
// loads of those planes.
template <int D>
GF2_HD void bs_pow_g(const u32 (&U)[D], u32 (&G)[D], bool pf = false) {
    GF2D_UNROLL
    for (int k = 0; k < D; k++) G[k] = 0;
    G[0] = ~0u;                                  // 1 in every slot
    // the D-1 exponent bits, padded at the top to a multiple of four with
    // zero bits (steps that only square the 1 they start from), so that one
    // loop body of four steps does them all
    constexpr int N = D - 1, NP = (N + 3) / 4 * 4;
    u32 s0 = NP - 1 < N ? U[NP - 1 < N ? NP - 1 : 0] : 0u;
    u32 s1 = NP - 2 < N ? U[NP - 2 < N ? NP - 2 : 0] : 0u;
    u32 s2 = NP - 3 < N ? U[NP - 3 < N ? NP - 3 : 0] : 0u;
    u32 s3 = U[NP - 4];
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
    for (int b = NP - 1; b >= 3; b -= 4) {
        if (pf && b >= 11) {                    // the group after next
            pf_local(&U[b - 8]); pf_local(&U[b - 9]);
            pf_local(&U[b - 10]); pf_local(&U[b - 11]);
        }
        // each register is reloaded with the plane four steps on (plane 0,
        // unused, in the last group)
        bs_step_g<D>(G, s0); s0 = U[b - 4 >= 0 ? b - 4 : 0];
        bs_step_g<D>(G, s1); s1 = U[b - 5 >= 0 ? b - 5 : 0];
        bs_step_g<D>(G, s2); s2 = U[b - 6 >= 0 ? b - 6 : 0];
        bs_step_g<D>(G, s3); s3 = U[b - 7 >= 0 ? b - 7 : 0];
    }
}

// A = X^E in every slot (E >= 1, the same for all slots)
template <int D>
GF2_HD void bs_pow(const u32 (&X)[D], u64 E, u32 (&A)[D]) {
    GF2D_UNROLL
    for (int k = 0; k < D; k++) A[k] = X[k];
    for (int b = 62 - gf2f_clz64(E); b >= 0; b--) {
        u32 R[D];
        bs_sqr<D>(A, R);
        if ((E >> b) & 1) {
            bs_mul<D>(R, X, A);
        } else {
            GF2D_UNROLL
            for (int k = 0; k < D; k++) A[k] = R[k];
        }
    }
}

// 32x32 bit transpose in place: afterwards bit k of w[i] is the former bit
// i of w[k]
GF2_HD void bs_transpose32(u32 (&w)[32]) {
    GF2D_UNROLL
    for (int jj = 4; jj >= 0; jj--) {           // j = 16, 8, 4, 2, 1
        const int j = 1 << jj;
        const u32 m = j == 16 ? 0x0000FFFFu : j == 8 ? 0x00FF00FFu
                    : j == 4 ? 0x0F0F0F0Fu : j == 2 ? 0x33333333u : 0x55555555u;
        GF2D_UNROLL
        for (int k = 0; k < 32; k++) {
            if (k & j) continue;
            const u32 t = ((w[k] >> j) ^ w[k | j]) & m;
            w[k | j] ^= t;
            w[k] ^= t << j;
        }
    }
}

// 32 scalars (bits 0..D-1 used) -> D planes
template <int D>
GF2_HD void bs_load(const u64 *v, u32 (&P)[D]) {
    constexpr int D0 = D < 32 ? D : 32;
    u32 w[32];
    GF2D_UNROLL
    for (int k = 0; k < 32; k++) w[k] = (u32)v[k];
    bs_transpose32(w);
    GF2D_UNROLL
    for (int i = 0; i < D0; i++) P[i] = w[i];
    if (D > 32) {
        GF2D_UNROLL
        for (int k = 0; k < 32; k++) w[k] = (u32)(v[k] >> 32);
        bs_transpose32(w);
        GF2D_UNROLL
        for (int i = 32; i < D; i++) P[i] = w[i - 32];
    }
}

// D planes -> 32 scalars
template <int D>
GF2_HD void bs_store(const u32 (&P)[D], u64 *v) {
    constexpr int D0 = D < 32 ? D : 32;
    u32 w[32];
    GF2D_UNROLL
    for (int i = 0; i < D0; i++) w[i] = P[i];
    GF2D_UNROLL
    for (int i = D0; i < 32; i++) w[i] = 0;
    bs_transpose32(w);
    GF2D_UNROLL
    for (int k = 0; k < 32; k++) v[k] = w[k];
    if (D > 32) {
        GF2D_UNROLL
        for (int i = 32; i < D; i++) w[i - 32] = P[i];
        GF2D_UNROLL
        for (int i = (D > 32 ? D - 32 : 0); i < 32; i++) w[i] = 0;
        bs_transpose32(w);
        GF2D_UNROLL
        for (int k = 0; k < 32; k++) v[k] |= (u64)w[k] << 32;
    }
}

// ---------------------------------------------------------------------------
// tower logs (d = 2m).  All pointers are device pointers; structs are
// passed to kernels by value.
//
// log e = (2^m+1) log_h(b) + T[log_h(a) - log_h(b)] for e = a + b w, so a
// log needs log_h (h = g^(2^m+1), generating K*) of two elements of K:
//   classic (round 3): one table LK of all 2^m elements of K -- two random
//          DRAM reads per class, plus the read of T;
//   subfield tower (round 4, whenever m has a divisor n with 2 <= n < m):
//          the same trick once more, inside K.  K is a degree-k extension of
//          K_n = GF(2^n) (k = m/n), with basis 1, h, ..., h^(k-1).  Writing
//          Y = lambda * Y', lambda the last nonzero coordinate of Y, gives
//          log_h Y = (2^m-1)/(2^n-1) * log_hn(lambda) + TP[Y'], where
//          hn = h^((2^m-1)/(2^n-1)) generates K_n*, LKn (2^n entries) holds
//          log_hn, and TP the logs of the normalized Y' (one per point of
//          the projective space P^(k-1)(K_n), (2^m-1)/(2^n-1) of them).
//          For even m (k = 2) both are tiny: 2^n entries.  So the two K logs
//          cost a few reads of small (cached / shared-memory) tables, and
//          the class one DRAM read (T) instead of three.
//          Normalized points are indexed by the logs of their coordinates:
//          Y' = (y'_0, ..., y'_(j-1), 1, 0, ..., 0) has digits
//          l_i = log_hn(y'_i), or 2^n - 1 for y'_i = 0, at bits n*i of the
//          index, after the 2^n + ... + 2^(n(j-1)) entries for smaller j.
//
// The cubic tower (round 9; see "cubic tower logs" below) keeps its own
// fields in TowerDev::cb, and uses the rest of its TowerDev as the classic
// tower of K = GF(2^(d/3)).
struct CubicDev {
    int n, nb;            // n = d/3 (even); nb = ceil(d/8) byte tables
    u64 MK, H;            // 2^n - 1; H = (2^d - 1)/(2^n - 1): kappa = g^H
    u64 c;                // phi(y) = kappa^c: Lk(Y) = c log_y(Y) mod MK
    u64 qlow;             // K_n = GF(2)[y]/(y^n + qlow)
    const u64 *btab;      // [nb][256]: x -> Y0 | Y1 << n | Y2 << 2n
    const u64 *T3;        // [MK + 1]: log(w + kappa^l); T3[MK] = log w
    u32 shm_bytes;        // device: btab and the K-tower's tables staged
};

struct TowerDev {
    int m, nb;            // nb = ceil(d/8) byte tables
    u64 MK;               // 2^m - 1 (also the coordinate mask)
    const u64 *btab;      // [nb][256]: packed coords coord(a) | coord(b)<<m
    const u32 *LK;        // classic: [2^m] coordinate -> log base h
    const u64 *T;         // [2^m]   T[l] = log_g(h^l + w) for l < MK, and
                          //         T[MK] = log_g(w)  (c = 0)
    int sub_n, sub_k;     // subfield tower (sub_k = 0: classic); coord(y)
                          // is then k digits of n bits: y's K_n-coordinates
                          // in the basis hn^0..hn^(n-1) (digit i: coeff of h^i)
    u64 sub_mult;         // (2^m - 1) / (2^n - 1)
    const uint16_t *LKn;  // [2^n]   n-bit K_n element -> log base hn
    const u32 *TP;        // logs of the normalized points (see above)
    u32 tp_n;             // number of TP entries
    u32 shm_bytes;        // device: stage btab, LKn (and TP if shm_tp) in
    int shm_tp;           //   shared memory (0: read them from global)
    CubicDev cb;          // round 9: the cubic tower's own fields (cb.n = 0:
                          //   not a cubic tower)
};

GF2_HD u64 tower_coords(u64 x, const TowerDev &t) {
    u64 pk = 0;
    for (int k = 0; k < t.nb; k++) pk ^= t.btab[k * 256 + ((x >> (8 * k)) & 255)];
    return pk;
}

// log_h(Y) for Y in K given coord(Y) (subfield tower); ~0 for Y = 0
GF2_HD u64 tower_sub_log(u64 yc, const TowerDev &t) {
    const int n = t.sub_n;
    const u32 MKn = (1u << n) - 1;
    int j = t.sub_k - 1;
    u32 lam;
    while ((lam = (u32)(yc >> (n * j)) & MKn) == 0)
        if (--j < 0) return ~0ULL;
    const u32 ll = t.LKn[lam];
    u64 v = t.sub_mult * ll;                    // log_h(lambda) < MK
    if (j > 0) {
        u64 idx = 0, off = 0;
        for (int i = 0; i < j; i++) {
            const u32 yi = (u32)(yc >> (n * i)) & MKn;
            u32 li = MKn;                       // digit for a zero coordinate
            if (yi) {
                const u32 l = t.LKn[yi];
                li = l >= ll ? l - ll : l + MKn - ll;   // log_hn(yi / lambda)
            }
            idx |= (u64)li << (n * i);
            if (i) off += (u64)1 << (n * i);    // entries of smaller j
        }
        v += t.TP[off + idx];                   // < MK
        if (v >= t.MK) v -= t.MK;
    }
    return v;
}

GF2_HD u64 tower_klog(u64 yc, const TowerDev &t) {   // log_h, Y != 0
    return t.sub_k ? tower_sub_log(yc, t) : (u64)t.LK[yc];
}

// e -> (lb = log_h(b), l = T index) for b != 0; false if b == 0
GF2_HD bool tower_split(u64 pk, const TowerDev &t, u64 *lb, u64 *l) {
    const u64 ac = pk & t.MK, bc = pk >> t.m;
    if (!bc) return false;
    *lb = tower_klog(bc, t);
    *l = t.MK;                                  // c = a/b = 0: slot MK
    if (ac) {
        const u64 la = tower_klog(ac, t);
        *l = la >= *lb ? la - *lb : la + t.MK - *lb;    // log_h(c)
    }
    return true;
}

// log_g(x) for x != 0.  T is indexed by log_h(c) rather than by c, so the
// quotient c = a/b is never formed.
GF2_HD u64 tower_log(u64 x, u64 M, const TowerDev &t) {
    const u64 pk = tower_coords(x, t);
    const u64 h1 = t.MK + 2;                    // 2^m + 1 = log_g(h)
    u64 lb, l;
    if (!tower_split(pk, t, &lb, &l))           // x in K*;  < M
        return h1 * tower_klog(pk & t.MK, t);
    const u64 v = h1 * lb + t.T[l];             // h1*lb < M, T < M
    return v >= M ? v - M : v;
}

// ---------------------------------------------------------------------------
// Cubic tower logs (round 9), for d = 3n with n even (6 | d; d = 54 is the
// one that matters).  GF(2^d) = K(w) with K = GF(2^n) and [GF(2^d):K] = 3:
// w^3 = y' for y' a root, in K, of the modulus q_n = y^n + qlow of the field
// K_n = GF(2)[y]/(q_n) -- so phi: K_n -> K, y -> y', is an isomorphism and
// "times y'" is a shift in K_n.  K* = <kappa>, kappa = g^H with
// H = (2^d - 1)/(2^n - 1), and y' = kappa^c.  (w = g^(cH/3): 3 | H as n is
// even, and w^3 = y' is no cube in K, so w is cubic over K.)
// Every x != 0 is x = Y0 + Y1 w + Y2 w^2 with Yi = phi of n-bit polynomials
// (a GF(2)-linear split, byte tables), and with
//     t = Y1 + Y2 w,     r = t x = (Y0 Y1 + y Y2^2) + (Y1^2 + Y0 Y2) w
// (the w^2 terms cancel, w^3 = y'), x = r/t with r and t LINEAR in w: the
// rational reconstruction of RR, over K instead of GF(2).  (x in K*: r = Y0,
// t = 1.)  With ly(Y) = log_y(Y) in K_n, phi(Y) = y'^ly(Y) = kappa^(c ly(Y)),
// so a linear a + b w with b != 0 is phi(b) (w + y'^(ly(a) - ly(b))) and
//     log(a + b w) = H (c ly(b) mod MK) + T3[(ly(a) - ly(b)) mod MK]
//     log(a)       = H (c ly(a) mod MK)                       (b = 0)
// where T3[l] = log(w + y'^l) for l < MK, T3[MK] = log w (a = 0), and
// T3[MK + 1] = 0, the slot of a factor in K (b = 0): every class reads T3
// twice, without branches.  T3 has 2^n + 1 entries -- 2 MB at d=54, a
// table L2 keeps, against the 1 GB T of the quadratic tower (a DRAM read
// per class); the ly come from the classic tower of K_n (tables of 2^(n/2)
// entries, in shared memory), four per class, and the K-parts of log r and
// log t need ONE product by c: H ((c (ly(lead r) - ly(lead t))) mod MK).
// Verified for every x at d = 6, 12, 18 by a prototype, by t_cubic, and in
// the selftests (log(g^u) == u) at every 6 | d <= 54.
//
// Two versions: cubic_parts<D>, the sieve's (n, the moduli and every loop
// bound compile-time constants, branch-free; ~400 instructions where a
// version on the general tower code took ~4700 -- four inlined tower logs,
// each with the subfield tower's loops -- and blew the instruction cache),
// and cubic_parts_rt, the checks' (runtime d, the general tower_log of K_n).

// p * QL for a compile-time QL: one shift and XOR per tap
template <u64 QL>
GF2_HD u64 gf2c_mul_low(u64 p) {
    u64 t = 0;
    GF2D_UNROLL
    for (int s = 0; s <= gf2_deg(QL); s++)
        if ((QL >> s) & 1) t ^= p << s;
    return t;
}

// p (degree <= 2N-1) mod y^N + gf2_qlow(N): two folds when 2 deg(qlow) <= N
template <int N>
GF2_HD u64 gf2c_reduce(u64 p) {
    constexpr u64 QL = gf2_qlow(N), mask = ((u64)1 << N) - 1;
    static_assert(2 * gf2_deg(QL) <= N && N <= 31, "gf2c_reduce: two folds");
    const u64 t = gf2c_mul_low<QL>(p >> N);         // degree <= N - 1 + deg QL
    return (p & mask) ^ (t & mask) ^ gf2c_mul_low<QL>(t >> N);
}

// ly(Y) = log_y(Y) in K_N, Y != 0, by the classic tower of K_N (t: its
// byte tables, LK and T): tower_log's arithmetic with only the classic path
// and every constant known.  Y = 0 returns a value that is never used.
template <int N>
GF2_HD u64 cubic_ly(u64 Y, const TowerDev &t) {
    constexpr int m = N / 2, nb = (N + 7) / 8;
    constexpr u64 MKk = ((u64)1 << m) - 1, MKn = ((u64)1 << N) - 1;
    constexpr u64 h1 = MKk + 2;                     // log_y(h) = 2^m + 1
    u64 pk = 0;
    GF2D_UNROLL
    for (int k = 0; k < nb; k++) pk ^= t.btab[k * 256 + ((Y >> (8 * k)) & 255)];
    const u64 ac = pk & MKk, bc = pk >> m;          // Y = a + b z over GF(2^m)
    const u64 la = t.LK[ac], lb = t.LK[bc];         // (LK[0]: unused)
    const u64 l = ac ? (la >= lb ? la - lb : la + MKk - lb) : MKk;
    const u64 v = bc ? h1 * lb + t.T[l] : h1 * la;  // b = 0: Y in GF(2^m)
    return v >= MKn ? v - MKn : v;
}

// Stage A of a cubic log: returns the K-part H (c (ly(lead r) - ly(lead t))
// mod MK) of log r - log t, and the T3 slots of r and t (MK + 1 for a
// factor in K)
template <int D>
GF2_HD u64 cubic_parts(u64 x, const TowerDev &t, u64 *ir, u64 *it) {
    constexpr int N = D / 3, nb = (D + 7) / 8;
    constexpr u64 MK = ((u64)1 << N) - 1;
    constexpr u64 H = ((u64)1 << (2 * N)) + ((u64)1 << N) + 1;    // M / MK
    static_assert(D % 6 == 0 && D >= 12 && D <= 54, "cubic_parts: 6 | D <= 54");
    const CubicDev &C = t.cb;
    u64 pk = 0;
    GF2D_UNROLL
    for (int k = 0; k < nb; k++) pk ^= C.btab[k * 256 + ((x >> (8 * k)) & 255)];
    const u64 Y0 = pk & MK, Y1 = (pk >> N) & MK, Y2 = pk >> (2 * N);
    const bool inK = !(Y1 | Y2);
    // r = r0 + r1 w = t x, t = t0 + t1 w = Y1 + Y2 w; x in K*: r = Y0, t = 1
    u64 r0 = gf2c_reduce<N>(gf2f_clmul32((u32)Y0, (u32)Y1) ^
                            (gf2f_spread32((u32)Y2) << 1));        // + y Y2^2
    const u64 r1 = gf2c_reduce<N>(gf2f_spread32((u32)Y1) ^
                                  gf2f_clmul32((u32)Y0, (u32)Y2));
    if (inK) r0 = Y0;                               // (r1 = 0 then)
    const u64 t0 = inK ? 1 : Y1, t1 = Y2;
    // four independent K-logs (one of a zero coefficient is never used)
    const u64 l0 = cubic_ly<N>(r0, t), l1 = cubic_ly<N>(r1, t);
    const u64 m0 = cubic_ly<N>(t0, t), m1 = cubic_ly<N>(t1, t);
    const u64 lr = r1 ? l1 : l0, lt = t1 ? m1 : m0; // the leading coefficients'
    *ir = !r1 ? MK + 1 : !r0 ? MK : (l0 >= l1 ? l0 - l1 : l0 + MK - l1);
    *it = !t1 ? MK + 1 : !t0 ? MK : (m0 >= m1 ? m0 - m1 : m0 + MK - m1);
    u64 p = C.c * (lr >= lt ? lr - lt : lr + MK - lt);    // < 2^(2N)
    p = (p & MK) + (p >> N);
    p = (p & MK) + (p >> N);
    if (p >= MK) p -= MK;
    return H * p;                                   // < H MK = M
}

// The same with d at run time (the checks and selftests' logs_dlog, the
// host): the general tower_log of K_n for ly
GF2_HD u64 cubic_parts_rt(u64 x, const TowerDev &t, u64 *ir, u64 *it) {
    const CubicDev &C = t.cb;
    const int n = C.n;
    const u64 MK = C.MK;
    u64 pk = 0;
    for (int k = 0; k < C.nb; k++) pk ^= C.btab[k * 256 + ((x >> (8 * k)) & 255)];
    const u64 Y0 = pk & MK, Y1 = (pk >> n) & MK, Y2 = pk >> (2 * n);
    u64 r0 = Y0, r1 = 0, t0 = 1, t1 = 0;            // x in K*: r = Y0, t = 1
    if (Y1 | Y2) {
        const u64 qn = ((u64)1 << n) | C.qlow;
        const u64 s2 = gf2f_sqr(Y2, C.qlow, n);
        const u64 ys2 = (s2 << 1) ^ (((s2 >> (n - 1)) & 1) ? qn : 0);     // y Y2^2
        r0 = gf2f_reduce(0, gf2f_clmul32((u32)Y0, (u32)Y1), C.qlow, n) ^ ys2;
        r1 = gf2f_sqr(Y1, C.qlow, n) ^
             gf2f_reduce(0, gf2f_clmul32((u32)Y0, (u32)Y2), C.qlow, n);
        t0 = Y1; t1 = Y2;
    }
    auto ly = [&](u64 Y) { return tower_log(Y, MK, t); };
    const u64 lr = ly(r1 ? r1 : r0), lt = ly(t1 ? t1 : t0);
    *ir = *it = MK + 1;
    if (r1) {
        const u64 la = r0 ? ly(r0) : 0;
        *ir = r0 ? (la >= lr ? la - lr : la + MK - lr) : MK;
    }
    if (t1) {
        const u64 la = t0 ? ly(t0) : 0;
        *it = t0 ? (la >= lt ? la - lt : la + MK - lt) : MK;
    }
    const u64 p = mulmod_u64(C.c, lr >= lt ? lr - lt : lr + MK - lt, MK);
    return C.H * p;
}

// the whole log (selftests, the table check): x != 0
GF2_HD u64 cubic_log(u64 x, u64 M, const TowerDev &t) {
    u64 ir, it;
    u64 v = cubic_parts_rt(x, t, &ir, &it);     // (sets ir and it first)
    v += t.cb.T3[ir];
    if (v >= M) v -= M;
    const u64 tt = t.cb.T3[it];
    return v >= tt ? v - tt : v + M - tt;
}

// ---------------------------------------------------------------------------
// Pohlig-Hellman logs
#define PH_MAXPARTS 4

// One hash slot: key and value side by side, so a probe is one 16-byte
// read (one memory sector) instead of two.
struct alignas(16) PHSlot { u64 key, val; };

struct PHPart {
    u64 P;                // order of the subgroup this part resolves
    u64 E;                // exponent: x^E = gamma^(log x mod P)
    u64 K;                // CRT constant: log x = sum (log x mod P_i)*K_i mod M
    u64 N, G;             // baby table holds gamma^0..gamma^(N-1); G = ceil(P/N)
    u64 delta;            // gamma^(-N): one giant step
    const PHSlot *slots;  // 2^sbits slots, key 0 = empty (subgroup elems != 0)
    int sbits;
};

struct PHDev {
    int nparts, chain;    // chain = highest set bit over all E_i
    PHPart part[PH_MAXPARTS];
};

GF2_HD u64 ph_slot(u64 key, int sbits) {
    return (key * 0x9E3779B97F4A7C15ULL) >> (64 - sbits);
}

// index i < N with gamma^i == y, or ~0
GF2_HD u64 ph_lookup(const PHPart &p, u64 y) {
    const u64 mask = ((u64)1 << p.sbits) - 1;
    for (u64 h = ph_slot(y, p.sbits);; h = (h + 1) & mask) {
        const PHSlot e = p.slots[h];
        if (e.key == y) return e.val;
        if (e.key == 0) return ~0ULL;
    }
}

// log of the subgroup element y (= x^E) mod P, by baby-step/giant-step if
// the table holds fewer than P entries; ~0 if not found (an error)
GF2_HD u64 ph_part_log(const PHPart &p, u64 y, u64 qlow, int d) {
    for (u64 s = 0; s < p.G; s++) {
        u64 idx = ph_lookup(p, y);
        if (idx != ~0ULL) {
            idx += s * p.N;
            return idx < p.P ? idx : ~0ULL;
        }
        y = gf2f_mul(y, p.delta, qlow, d);
    }
    return ~0ULL;
}

// log_g(x) for x != 0, or ~0 on an internal inconsistency (never expected;
// the caller counts it as an error)
GF2_HD u64 ph_log(u64 x, u64 M, int d, u64 qlow, const PHDev &ph) {
    u64 acc[PH_MAXPARTS];
    GF2D_UNROLL
    for (int i = 0; i < PH_MAXPARTS; i++) acc[i] = 0;    // 0 = no factor yet
    u64 cur = x;                                // x^(2^t)
    for (int t = 0; t <= ph.chain; t++) {
        GF2D_UNROLL
        for (int i = 0; i < PH_MAXPARTS; i++)
            if (i < ph.nparts && ((ph.part[i].E >> t) & 1))
                acc[i] = acc[i] ? gf2f_mul(acc[i], cur, qlow, d) : cur;
        if (t < ph.chain) cur = gf2f_sqr(cur, qlow, d);
    }
    u64 v = 0;
    GF2D_UNROLL
    for (int i = 0; i < PH_MAXPARTS; i++) {
        if (i >= ph.nparts) continue;
        const PHPart &p = ph.part[i];
        u64 xi = ph_part_log(p, acc[i] ? acc[i] : 1, qlow, d);
        if (xi == ~0ULL) return ~0ULL;
        v += gf2f_mulmod_m(xi, p.K, d, M);
        if (v >= M) v -= M;
    }
    return v;
}

// ---------------------------------------------------------------------------
// RR logs (round 5): rational reconstruction against a table of the logs of
// ALL polynomials of degree <= dh = floor(d/2).
//
// The extended Euclidean algorithm on (q, y), stopped at the first remainder
// r of degree <= dh, gives r = t*y (mod q) with deg t = d - deg(previous
// remainder) <= d - dh - 1 <= dh.  So every y != 0 is r/t with both of
// degree <= dh, and log y = L[r] - L[t] with L[P] = log_g(P) for the
// 2^(dh+1) polynomials P (field elements, as deg P < d).  No smoothness
// test, no failure: one half-gcd and two reads of a table of 2^(dh+1) logs
// (2^25 entries = 256 MB at d=49) per class.  L is built from the logs of
// the irreducible polynomials of degree <= dh (~2^(dh+1)/dh of them) by
// multiplicativity; those come from Pohlig-Hellman -- for d=49, where M =
// 127 * 4432676798593 has no table-sized parts, with many giant steps per
// log, which is affordable for 1.47M factor-base elements but not for the
// 1.15e13 classes.  The cofactors t never exceed degree dh <= 27, so they
// are u32.
struct RRDev {
    const u64 *L;          // [2^(dh+1)] logs; L[0] unused
    int dh;
};

// the half-gcd: r = t * y mod q (q = x^d + qlow), deg r, deg t <= dh
GF2_HD void rr_split_rt(u64 y, u64 q, int d, int dh, u64 *r_out, u32 *t_out) {
    u64 a = q, b = y;
    u32 ta = 0, tb = 1;
    int da = d, db = 63 - gf2f_clz64(b);
    while (db > dh) {
        const int s = da - db;                  // reduce a by b once
        a ^= b << s;
        ta ^= tb << s;
        da = 63 - gf2f_clz64(a);                // a != 0: gcd(q, y) = 1
        const bool sw = da < db;                // keep deg a >= deg b
        const u64 na = sw ? b : a, nb = sw ? a : b;
        const u32 nta = sw ? tb : ta, ntb = sw ? ta : tb;
        const int nda = sw ? db : da, ndb = sw ? da : db;
        a = na; b = nb; ta = nta; tb = ntb; da = nda; db = ndb;
    }
    *r_out = b;
    *t_out = tb;
}

template <int D>
GF2_HD void rr_split(u64 y, u64 *r_out, u32 *t_out) {
    rr_split_rt(y, ((u64)1 << D) | gf2_qlow(D), D, D / 2, r_out, t_out);
}

GF2_HD u64 rr_log(u64 y, u64 q, int d, u64 M, const RRDev &rr) {
    u64 r;
    u32 t;
    rr_split_rt(y, q, d, rr.dh, &r, &t);
    const u64 lr = rr.L[r], lt = rr.L[t];
    return lr >= lt ? lr - lt : lr + M - lt;
}

// ---------------------------------------------------------------------------
// host-side builders (one-off per degree; bit-serial gf_mul is fine here)
#include "gf2_wide_setup.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>

// ---- RR tables, host side ----
// Polynomials of degree <= dh are the integers [1, 2^(dh+1)).  spf[P] is the
// least irreducible factor of P (0 for P irreducible, or P = 1): a sieve as
// Eratosthenes', ~2^(dh+1) * H(dh) carry-less products (3 s at dh = 24;
// ~1e9, perhaps half a minute, at dh = 27), and P is irreducible iff
// nothing marked it before P.
static void rr_sieve(int dh, std::vector<u32> &spf, std::vector<u32> &irr) {
    const u32 n = 1u << (dh + 1);
    spf.assign(n, 0);
    irr.clear();
    for (u32 P = 2; P < n; P++) {
        if (spf[P]) continue;
        irr.push_back(P);
        const int dp = 31 - __builtin_clz(P);
        const u32 qn = 1u << (dh + 1 - dp);      // multiples P*Q, deg <= dh
        for (u32 Q = 2; Q < qn; Q++) {
            const u32 m = (u32)gf2f_clmul32(P, Q);
            if (!spf[m]) spf[m] = P;
        }
    }
}

// P / f for f | P (carry-less long division)
static u32 rr_divexact(u32 P, u32 f) {
    const int df = 31 - __builtin_clz(f);
    u32 q = 0;
    for (int k = 31 - __builtin_clz(P) - df; k >= 0; k--)
        if ((P >> (k + df)) & 1) { P ^= f << k; q |= 1u << k; }
    return q;
}

// L[P] = log P for every P of degree <= dh from the logs of the
// irreducibles (irr[i] has log irr_log[i]): L[f*Q] = L[f] + L[Q] (mod M),
// in increasing order so L[Q] (deg Q < deg P) is always ready
static void rr_fill(int dh, u64 M, const std::vector<u32> &spf,
                    const std::vector<u32> &irr,
                    const std::vector<u64> &irr_log, std::vector<u64> &L) {
    const u32 n = 1u << (dh + 1);
    L.assign(n, 0);
    for (size_t i = 0; i < irr.size(); i++) L[irr[i]] = irr_log[i];
    L[1] = 0;
    for (u32 P = 2; P < n; P++) {
        const u32 f = spf[P];
        if (!f) continue;                        // irreducible: set above
        const u64 s = L[f] + L[rr_divexact(P, f)];
        L[P] = s >= M ? s - M : s;
    }
}
#include <cmath>

struct TowerHost {
    int m = 0, nb = 0;
    u64 MK = 0;
    std::vector<u64> btab;               // nb*256
    std::vector<u32> LK, AK;             // 2^m, 2^m - 1 (host; LK also on the
                                         // device for the classic tower)
    int sub_n = 0, sub_k = 0;            // subfield tower, 0 = classic
    u64 sub_mult = 0;
    std::vector<uint16_t> LKn;                // 2^n
    std::vector<u32> TP;                 // sum_{j=1}^{k-1} 2^(n j)
};

// The subfield K_n used for the inner tower: the largest proper divisor n of
// m with 2 <= n (so k = m/n >= 2 is smallest and TP smallest), or 0 if m is
// prime (or m = 2): then the classic tower.  want: -1 automatic, 0 classic,
// n > 0 a specific divisor (tests).  *tp_entries: size of TP.
static int tower_sub_choice(int m, int want, u64 *tp_entries) {
    int n = 0;
    if (want > 0) {
        if (want >= 2 && want < m && m % want == 0) n = want;
    } else if (want < 0) {
        for (int c = m / 2; c >= 2; c--) if (m % c == 0) { n = c; break; }
    }
    u64 tp = 0;
    if (n) for (int j = 1; j < m / n; j++) tp += (u64)1 << (n * j);
    if (tp > ((u64)1 << 26)) n = 0, tp = 0;     // keep TP <= 256 MB
    if (tp_entries) *tp_entries = tp;
    return n;
}

// Byte tables for e -> (coord(a) | coord(b) << m), and the logs in K: the
// table LK (classic), or LKn/TP (subfield tower, sub_want != 0 and m has a
// divisor 2 <= n < m).  T (the 2^m logs log(h^l + w)) is filled on the
// device (k_tower_T).  AK and LK stay on the host either way (TP is built
// from LK).  Returns false if any internal consistency check fails.
static bool tower_build_host(int d, u64 q, TowerHost &W, int sub_want = -1) {
    if (d < 4 || (d & 1) || d > 54) return false;
    const int m = d / 2;
    const u64 M = ((u64)1 << d) - 1, MK = ((u64)1 << m) - 1;
    std::vector<u64> sig(d);                    // sigma(x^i) = x^(i*2^m)
    for (int i = 0; i < d; i++) {
        u64 y = (u64)1 << i;
        for (int k = 0; k < m; k++) y = gf_sqr(y, q, d);
        sig[i] = y;
    }
    auto lin_sigma = [&](u64 e) {
        u64 r = 0;
        for (int i = 0; i < d; i++) if (e >> i & 1) r ^= sig[i];
        return r;
    };
    u64 den = 2 ^ lin_sigma(2);                 // w + sigma(w), w = x
    if (!den) return false;
    u64 inv_den = gf_pow(den, M - 1, q, d);
    // basis of K: 1, h, ..., h^(m-1), h = g^(2^m+1); full RREF
    u64 h = gf_pow(2, MK + 2, q, d);
    std::vector<u64> rows;
    u64 y = 1;
    for (int k = 0; k < m; k++, y = gf_mul(y, h, q, d)) {
        u64 v = y;
        for (u64 r : rows) { int p = 63 - __builtin_clzll(r); if (v >> p & 1) v ^= r; }
        if (!v) return false;
        rows.push_back(v);
        std::sort(rows.begin(), rows.end(), std::greater<u64>());
    }
    for (size_t i = 0; i < rows.size(); i++) {
        int p = 63 - __builtin_clzll(rows[i]);
        for (size_t j = 0; j < rows.size(); j++)
            if (j != i && (rows[j] >> p & 1)) rows[j] ^= rows[i];
    }
    std::vector<int> piv(m);
    for (int k = 0; k < m; k++) piv[k] = 63 - __builtin_clzll(rows[k]);
    auto coordK = [&](u64 e) {
        u64 c = 0;
        for (int k = 0; k < m; k++) c |= (e >> piv[k] & 1) << k;
        return c;
    };
    auto embedK = [&](u64 c) {
        u64 e = 0;
        for (int k = 0; k < m; k++) if (c >> k & 1) e ^= rows[k];
        return e;
    };
    W = TowerHost();
    W.m = m; W.MK = MK; W.nb = (d + 7) / 8;
    // K* = <h>: step by h in coordinates (a linear map) -> AK, LK
    std::vector<u64> hcol(m);
    for (int k = 0; k < m; k++) {
        u64 t = gf_mul(h, rows[k], q, d);
        hcol[k] = coordK(t);
        if (embedK(hcol[k]) != t) return false;
    }
    const int mb = (m + 7) / 8;
    std::vector<u64> hb((size_t)mb * 256, 0);
    for (int kb = 0; kb < mb; kb++)
        for (int byte = 0; byte < 256; byte++) {
            u64 r = 0;
            for (int bit = 0; bit < 8; bit++)
                if ((byte >> bit & 1) && 8 * kb + bit < m) r ^= hcol[8 * kb + bit];
            hb[(size_t)kb * 256 + byte] = r;
        }
    W.LK.assign((size_t)MK + 1, 0xFFFFFFFFu);
    W.AK.assign((size_t)MK, 0);
    {
        u64 c = coordK(1);
        for (u64 k = 0; k < MK; k++) {
            if (!c || c > MK || W.LK[c] != 0xFFFFFFFFu) return false;   // h not of order MK?!
            W.AK[k] = (u32)c;
            W.LK[c] = (u32)k;
            u64 nc = 0;
            for (int kb = 0; kb < mb; kb++) nc ^= hb[(size_t)kb * 256 + ((c >> (8 * kb)) & 255)];
            c = nc;
        }
        if (c != coordK(1)) return false;
    }
    // coordinates of the elements of K handed to the device
    std::function<u64(u64)> coord = coordK;
    std::vector<u64> fv, lab;                   // subfield tower: labelled RREF
    std::vector<int> fpiv;
    const int n = tower_sub_choice(m, sub_want, nullptr);
    if (n) {
        const int k = m / n;
        const u64 MKn = ((u64)1 << n) - 1, mult = MK / MKn;
        const u64 hn = gf_pow(h, mult, q, d);   // generates K_n*
        // basis of K over GF(2): h^i * hn^jj (i < k, jj < n); label bit
        // n*i + jj, so a vector's label is its packed coordinate
        u64 hi = 1;
        for (int i = 0; i < k; i++, hi = gf_mul(hi, h, q, d)) {
            u64 z = hi;
            for (int jj = 0; jj < n; jj++, z = gf_mul(z, hn, q, d)) {
                fv.push_back(z);
                lab.push_back((u64)1 << (n * i + jj));
            }
        }
        for (int r = 0; r < m; r++) {           // full reduction, labels along
            int best = -1;
            for (int t = r; t < m; t++)
                if (fv[t] && (best < 0 || fv[t] > fv[best])) best = t;
            if (best < 0) return false;         // not a basis
            std::swap(fv[r], fv[best]); std::swap(lab[r], lab[best]);
            const int p = 63 - __builtin_clzll(fv[r]);
            for (int t = 0; t < m; t++)
                if (t != r && (fv[t] >> p & 1)) { fv[t] ^= fv[r]; lab[t] ^= lab[r]; }
            fpiv.push_back(p);
        }
        coord = [&fv, &lab, &fpiv, m](u64 e) {
            u64 c = 0, r = 0;
            for (int t = 0; t < m; t++)
                if (e >> fpiv[t] & 1) { c ^= lab[t]; r ^= fv[t]; }
            return r == e ? c : ~0ULL;          // ~0: e not in K
        };
        // LKn: n-bit coordinates of hn^l (a K_n element: digit 0 only)
        W.LKn.assign((size_t)MKn + 1, 0xFFFF);
        std::vector<u64> hnp(MKn);
        u64 z = 1;
        for (u64 l = 0; l < MKn; l++, z = gf_mul(z, hn, q, d)) {
            const u64 c = coord(z);
            if (c == 0 || c > MKn || W.LKn[c] != 0xFFFF) return false;
            W.LKn[c] = (uint16_t)l;
            hnp[l] = z;
        }
        if (z != 1) return false;
        // TP: log_h of each normalized point, by its digit index
        std::vector<u64> hp(k + 1);
        hp[0] = 1;
        for (int i = 1; i <= k; i++) hp[i] = gf_mul(hp[i - 1], h, q, d);
        std::vector<std::vector<u64>> prod(k, std::vector<u64>((size_t)MKn + 1, 0));
        for (int i = 0; i < k; i++)              // prod[i][MKn] = 0 (zero digit)
            for (u64 l = 0; l < MKn; l++) prod[i][l] = gf_mul(hnp[l], hp[i], q, d);
        for (int j = 1; j < k; j++) {
            const u64 cnt = (u64)1 << (n * j);
            for (u64 idx = 0; idx < cnt; idx++) {
                u64 e = hp[j];
                for (int i = 0; i < j; i++) e ^= prod[i][(idx >> (n * i)) & MKn];
                const u64 ce = coordK(e);
                if (embedK(ce) != e || !ce) return false;
                W.TP.push_back(W.LK[ce]);
            }
        }
        W.sub_n = n; W.sub_k = k; W.sub_mult = mult;
    }
    W.btab.assign((size_t)W.nb * 256, 0);
    for (int kk = 0; kk < W.nb; kk++)
        for (int byte = 0; byte < 256; byte++) {
            u64 e = ((u64)byte << (8 * kk)) & M;
            u64 b = gf_mul(e ^ lin_sigma(e), inv_den, q, d);
            u64 a = e ^ gf_mul(b, 2, q, d);
            u64 ca = coord(a), cb = coord(b);
            if (ca == ~0ULL || cb == ~0ULL) return false;
            if (!n && (embedK(ca) != a || embedK(cb) != b)) return false;   // a,b in K
            W.btab[(size_t)kk * 256 + byte] = ca | (cb << m);
        }
    return true;
}

// A TowerDev over the given table pointers (device or host memory)
static inline TowerDev tower_dev(const TowerHost &W, const u64 *btab,
                                 const u32 *LK, const u64 *T,
                                 const uint16_t *LKn, const u32 *TP) {
    TowerDev t;
    memset(&t, 0, sizeof t);
    t.m = W.m; t.nb = W.nb; t.MK = W.MK;
    t.btab = btab; t.LK = LK; t.T = T;
    t.sub_n = W.sub_n; t.sub_k = W.sub_k; t.sub_mult = W.sub_mult;
    t.LKn = LKn; t.TP = TP; t.tp_n = (u32)W.TP.size();
    return t;
}

// ---- the cubic tower (round 9), host side ----
struct CubicHost {
    int d = 0, n = 0, nb = 0;
    u64 M = 0, MK = 0, H = 0, c = 0, qlow = 0, logw = 0;
    u64 w = 0, kappa = 0, yp = 0;        // elements of GF(2^d) (yp = y' =
                                         // kappa^c), for T3
    std::vector<u64> btab;               // [nb][256]: packed coordinates
    TowerHost kt;                        // the classic tower of K_n ...
    std::vector<u64> kT;                 // ... and its T (2^(n/2) + 1 logs)
};

// Everything but T3 (built on the device by Pohlig-Hellman, see
// coarse_sieve.cu): y' and c, w, the coordinates x -> (Y0, Y1, Y2) and the
// tower of K_n.  False for d without the structure (6 does not divide d,
// or d < 12) or on any failed consistency check.
static inline bool cubic_build_host(int d, u64 q, CubicHost &C) {
    C = CubicHost();
    if (d % 6 || d < 12 || d > 54) return false;
    const int n = d / 3;
    const u64 M = ((u64)1 << d) - 1, MK = ((u64)1 << n) - 1, H = M / MK;
    if (H % 3) return false;
    const u64 qn = find_primitive_poly(n);
    if ((qn ^ ((u64)1 << n)) != gf2_qlow(n)) return false;
    const u64 kappa = gf_pow(2, H, q, d);        // generates K*
    auto qn_at = [&](u64 z) {                    // q_n(z) in GF(2^d), Horner
        u64 acc = 0;
        for (int i = n; i >= 0; i--) {
            acc = gf_mul(acc, z, q, d);
            if ((qn >> i) & 1) acc ^= 1;
        }
        return acc;
    };
    // y' = kappa^c, the first root of q_n among the powers of kappa (all n
    // roots are generators of K*, as q_n is primitive)
    u64 c = 0, z = 1;
    for (u64 cc = 1; cc < MK; cc++) {
        z = gf_mul(z, kappa, q, d);
        if (gcd_u64(cc, MK) == 1 && !qn_at(z)) { c = cc; break; }
    }
    if (!c) return false;
    const u64 yp = z;
    const u64 logw = mulmod_u64(H / 3, c, M);
    const u64 w = gf_pow(2, logw, q, d);
    if (gf_mul(gf_mul(w, w, q, d), w, q, d) != yp) return false;
    // basis yp^i w^j (i < n, j < 3) of GF(2^d) over GF(2), fully reduced,
    // each vector labelled with its coordinate bit j*n + i
    std::vector<u64> fv, lab;
    std::vector<int> fpiv;
    u64 wj = 1;
    for (int j = 0; j < 3; j++, wj = gf_mul(wj, w, q, d)) {
        u64 e = wj;
        for (int i = 0; i < n; i++, e = gf_mul(e, yp, q, d)) {
            fv.push_back(e);
            lab.push_back((u64)1 << (j * n + i));
        }
    }
    for (int r = 0; r < d; r++) {
        int best = -1;
        for (int t = r; t < d; t++)
            if (fv[t] && (best < 0 || fv[t] > fv[best])) best = t;
        if (best < 0) return false;              // not a basis
        std::swap(fv[r], fv[best]); std::swap(lab[r], lab[best]);
        const int p = 63 - __builtin_clzll(fv[r]);
        for (int t = 0; t < d; t++)
            if (t != r && (fv[t] >> p & 1)) { fv[t] ^= fv[r]; lab[t] ^= lab[r]; }
        fpiv.push_back(p);
    }
    auto coord = [&](u64 e) {
        u64 cc = 0, rr = 0;
        for (int t = 0; t < d; t++)
            if (e >> fpiv[t] & 1) { cc ^= lab[t]; rr ^= fv[t]; }
        return rr == e ? cc : ~0ULL;
    };
    C.nb = (d + 7) / 8;
    C.btab.assign((size_t)C.nb * 256, 0);
    for (int kk = 0; kk < C.nb; kk++)
        for (int byte = 0; byte < 256; byte++) {
            const u64 cc = coord(((u64)byte << (8 * kk)) & M);
            if (cc == ~0ULL) return false;
            C.btab[(size_t)kk * 256 + byte] = cc;
        }
    // the coordinates reproduce x: x = phi(Y0) + phi(Y1) w + phi(Y2) w^2
    {
        auto phi = [&](u64 Y) {
            u64 e = 0, zz = 1;
            for (int i = 0; i < n; i++, zz = gf_mul(zz, yp, q, d)) if (Y >> i & 1) e ^= zz;
            return e;
        };
        u64 s = 0x9E3779B97F4A7C15ULL ^ (u64)d;
        for (int i = 0; i < 64; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            const u64 x = s & M;
            u64 pk = 0;
            for (int kk = 0; kk < C.nb; kk++) pk ^= C.btab[(size_t)kk * 256 + ((x >> (8 * kk)) & 255)];
            const u64 e = phi(pk & MK) ^ gf_mul(phi((pk >> n) & MK), w, q, d) ^
                          gf_mul(phi(pk >> (2 * n)), gf_mul(w, w, q, d), q, d);
            if (e != x) return false;
        }
    }
    if (!tower_build_host(n, qn, C.kt, 0)) return false;    // classic
    // its T, T[l] = log_y(h^l + y) and T[2^m - 1] = log_y(y) = 1 (h =
    // y^(2^m + 1), m = n/2: tower_log's definitions with g = w = y), from a
    // full log table of K_n (host only, 2^n entries)
    {
        std::vector<u32> lgn((size_t)MK + 1, 0xFFFFFFFFu);
        u64 e = 1;
        for (u64 i = 0; i < MK; i++) { lgn[e] = (u32)i; e = gf_mul(e, 2, qn, n); }
        if (e != 1) return false;
        const int mk = n / 2;
        const u64 MKk = ((u64)1 << mk) - 1;
        const u64 hk = gf_pow(2, MKk + 2, qn, n);
        C.kT.assign((size_t)MKk + 1, 0);
        u64 hl = 1;
        for (u64 l = 0; l < MKk; l++, hl = gf_mul(hl, hk, qn, n)) {
            const u32 lg = lgn[hl ^ 2];
            if (lg == 0xFFFFFFFFu) return false;
            C.kT[l] = lg;
        }
        C.kT[MKk] = 1;
        // the K_n tower against the full table, on a sample
        TowerDev kt = tower_dev(C.kt, C.kt.btab.data(), C.kt.LK.data(),
                                C.kT.data(), nullptr, nullptr);
        for (u64 x = 1; x <= MK; x += 1 + MK / 4096)
            if (tower_log(x, MK, kt) != lgn[x]) return false;
    }
    C.d = d; C.n = n; C.M = M; C.MK = MK; C.H = H; C.c = c;
    C.qlow = qn ^ ((u64)1 << n); C.logw = logw; C.w = w; C.kappa = kappa;
    C.yp = yp;
    return true;
}

// The cubic tower's TowerDev: the tower of K_n over the given pointers,
// plus the cubic fields (T3 filled in later where it is built)
static inline TowerDev cubic_dev(const CubicHost &C, const u64 *btab3,
                                 const u64 *T3, const u64 *kbtab,
                                 const u32 *kLK, const u64 *kT) {
    TowerDev t = tower_dev(C.kt, kbtab, kLK, kT, nullptr, nullptr);
    t.cb.n = C.n; t.cb.nb = C.nb;
    t.cb.MK = C.MK; t.cb.H = C.H; t.cb.c = C.c; t.cb.qlow = C.qlow;
    t.cb.btab = btab3; t.cb.T3 = T3;
    t.cb.shm_bytes = 0;
    return t;
}

struct PHPartHost { u64 P = 0, E = 0, K = 0, N = 0, G = 1, delta = 1, gamma = 1; int sbits = 1; };
struct PHHost {
    int nparts = 0, chain = 0;
    PHPartHost part[PH_MAXPARTS];
    size_t table_bytes = 0;
    double cost = 0;                     // model estimate, instructions/class
};

// slot count 2^s >= 2N (load <= 1/2), one PHSlot (16 bytes) each
static inline size_t ph_table_bytes(u64 N, int *sbits_out) {
    int s = 1;
    while (((u64)1 << s) < 2 * N) s++;
    if (sbits_out) *sbits_out = s;
    return ((size_t)1 << s) * sizeof(PHSlot);
}

// Cost of one Pohlig-Hellman log, in rough GPU instructions per class:
//   scalar (one log per thread: RR's factor-base logs): a multiply costs ~2
//     squarings and all parts share one squaring chain;
//   bit-sliced (the batched kernel): a multiply ~73, a squaring ~3, each
//     part its own chain, plus a transpose per part.
// A lookup in a table larger than ~1 MB is a DRAM access (~300 equivalent
// instructions); smaller tables stay in cache.  A giant step is a scalar
// multiply plus another lookup.  Table memory is charged 1 per MB, so a
// plan only takes gigabytes for a real gain (and small tables cache better).
struct PHCostModel {
    double mul, sqr, lookup_dram, lookup_cached, giant, per_part;
    bool shared_chain;
};
static const PHCostModel kPHCostScalar    = {250, 125, 300, 60, 550, 0, true};
static const PHCostModel kPHCostBitsliced = {73, 3, 300, 60, 550, 60, false};

// The exponent for one part: E = c * (M/P) mod M over c coprime to P (and
// its rotations c * 2^-r), minimizing the model cost of x^E.
static u64 ph_best_exponent(int d, u64 M, u64 P, const std::vector<u64> &primes,
                            const PHCostModel &cm) {
    const u64 C = M / P;
    std::vector<u64> pf;
    for (u64 p : primes) if (P % p == 0) pf.push_back(p);
    const u64 lim = std::min<u64>(P, (u64)1 << 22);
    u64 E = 0, bestE = C;
    double best = 1e300;
    for (u64 c = 1; c < lim; c++) {
        E += C; if (E >= M) E -= M;
        const int pc = __builtin_popcountll(E);
        if ((pc - 1) * cm.mul >= best) continue;
        bool cop = true;
        for (u64 p : pf) if (c % p == 0) { cop = false; break; }
        if (!cop) continue;
        u64 Er = E;
        for (int r = 0; r < d; r++) {            // rotations: same popcount
            const int top = 63 - __builtin_clzll(Er);
            const double cost = (pc - 1) * cm.mul + top * cm.sqr;
            if (cost < best) { best = cost; bestE = Er; }
            Er = (Er >> 1) | ((Er & 1) << (d - 1));      // * 2^-1 mod M
        }
    }
    return bestE;
}

// Table sizes for parts P[0..n): every N_i = P_i if that fits the budget,
// else the largest tables are halved (baby-step/giant-step).  False if even
// that fails, or would need more than max_giant giant steps somewhere.
static bool ph_fit_tables(const u64 *P, int n, size_t budget, u64 max_giant,
                          u64 *N, size_t *bytes) {
    for (int i = 0; i < n; i++) N[i] = P[i];
    auto total = [&]() {
        size_t t = 0;
        for (int i = 0; i < n; i++) t += ph_table_bytes(N[i], nullptr);
        return t;
    };
    while (total() > budget) {
        int big = 0;
        for (int i = 1; i < n; i++) if (N[i] > N[big]) big = i;
        int sb;
        ph_table_bytes(N[big], &sb);
        if (sb <= 12) return false;
        N[big] = (u64)1 << (sb - 2);             // half the slots, load 1/2
    }
    for (int i = 0; i < n; i++)
        if ((P[i] + N[i] - 1) / N[i] > max_giant) return false;
    *bytes = total();
    return true;
}

// Plan Pohlig-Hellman: a coprime split of M into 1..4 parts, table sizes
// within budget_bytes, the cheapest exponents, CRT constants -- the split
// with the least model cost over all splits whose tables fit (each distinct
// part gets one exponent search, ~4M candidates at most).  False if none
// fits (e.g. d=49: 2^49-1 = 127 * 4432676798593 would need ~8,000-33,000
// giant steps per log with any table that fits on a card).
// search_exponent = false (round 9: a plan for a few thousand logs, the
// cubic tower's T3) takes E = M/P as it is: the search costs seconds per
// distinct part, half a minute at d=48 (nine prime powers in 2^48 - 1).
static bool ph_plan_host(int d, u64 q, size_t budget_bytes, PHHost &H,
                         const PHCostModel &cm = kPHCostScalar,
                         u64 max_giant = 64, bool search_exponent = true) {
    const u64 M = ((u64)1 << d) - 1;
    std::vector<u64> primes = factor_u64(M), pp;   // prime powers p^k || M
    u64 rest = M;
    for (u64 p : primes) {
        u64 pk = 1;
        while (rest % p == 0) { rest /= p; pk *= p; }
        pp.push_back(pk);
    }
    const int n = (int)pp.size();
    std::vector<std::pair<u64, u64>> ecache;        // P -> best exponent
    auto exponent_for = [&](u64 P) {
        if (!search_exponent) return M / P;
        for (auto &e : ecache) if (e.first == P) return e.second;
        u64 E = ph_best_exponent(d, M, P, primes, cm);
        ecache.push_back({P, E});
        return E;
    };
    bool have = false;
    std::vector<int> g(n, 0);
    // every split into k <= 4 groups, each once (restricted growth strings)
    std::function<void(int, int)> rec = [&](int i, int used) {
        if (i < n) {
            for (int j = 0; j <= used && j < PH_MAXPARTS; j++) {
                g[i] = j;
                rec(i + 1, std::max(used, j + 1));
            }
            return;
        }
        u64 P[PH_MAXPARTS], N[PH_MAXPARTS];
        for (int j = 0; j < used; j++) P[j] = 1;
        for (int j = 0; j < n; j++) P[g[j]] *= pp[j];
        size_t bytes;
        if (!ph_fit_tables(P, used, budget_bytes, max_giant, N, &bytes)) return;
        PHHost T;
        T.nparts = used;
        double cost = bytes / 1048576.0;            // 1 per MB
        for (int j = 0; j < used; j++) {
            PHPartHost &pt = T.part[j];
            pt.P = P[j];
            pt.E = exponent_for(P[j]);
            pt.N = N[j];
            pt.G = (pt.P + pt.N - 1) / pt.N;
            const int top = 63 - __builtin_clzll(pt.E);
            T.chain = std::max(T.chain, top);
            const size_t tb = ph_table_bytes(pt.N, &pt.sbits);
            cost += (__builtin_popcountll(pt.E) - 1) * cm.mul + cm.per_part
                  + (cm.shared_chain ? 0 : top * cm.sqr)
                  + (tb <= ((size_t)1 << 20) ? cm.lookup_cached : cm.lookup_dram)
                  + (pt.G - 1) / 2.0 * cm.giant;
        }
        if (cm.shared_chain) cost += T.chain * cm.sqr;
        T.table_bytes = bytes;
        T.cost = cost;
        if (!have || cost < H.cost) { H = T; have = true; }
    };
    rec(0, 0);
    if (!have) return false;
    for (int j = 0; j < H.nparts; j++) {             // CRT constants, tables
        PHPartHost &pt = H.part[j];
        const u64 C = M / pt.P;
        pt.K = mulmod_u64(C, pt.P == 1 ? 0 : inv_mod(C % pt.P, pt.P), M);
        pt.gamma = gf_pow(2, pt.E, q, d);
        pt.delta = (pt.N >= pt.P) ? 1 : gf_pow(pt.gamma, pt.P - pt.N, q, d);
    }
    return true;
}

#endif // GF2_DLOG_H

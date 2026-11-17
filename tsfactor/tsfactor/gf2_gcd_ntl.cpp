// gf2_gcd_ntl.cpp
// NTL backend for gf2_gcd.h: subquadratic GCD (HalfGCD) and CanZass
// factorization for the equal-degree tie-break.  Compile with
// -DHAVE_NTL and link -lntl -lgmp; for production-scale r, build NTL
// with the gf2x multiplication backend (NTL_GF2X_LIB=on), exactly as
// for factor.cpp.
//
// Word <-> GF2X conversion relies on: x86-64 little-endian u64 words
// viewed as bytes give monotone bit order, and NTL's GF2XFromBytes /
// BytesFromGF2X use little-endian bytes with bit i of byte j as
// coefficient 8j + i.  The naive-vs-NTL cross-check in
// `trinomial_scan --selftest` validates this empirically.

#ifdef HAVE_NTL
#include "gf2_gcd.h"
#include <NTL/GF2X.h>
#include <NTL/GF2XFactoring.h>

using namespace NTL;

static void to_gf2x(GF2X &f, const u64 *w, size_t nw) {
    GF2XFromBytes(f, (const unsigned char *)w, (long)(nw * 8));
}

static void from_gf2x(std::vector<u64> &out, const GF2X &f) {
    long nb = NumBytes(f);
    if (nb <= 0) { out.assign(1, 0); return; }
    out.assign(((size_t)nb + 7) / 8 + 1, 0);
    BytesFromGF2X((unsigned char *)out.data(), f, nb);
}

u64 poly_gcd_ntl(const u64 *a, size_t aw, const u64 *b, size_t bw,
                 std::vector<u64> *gout, u64 keep_max_bits) {
    GF2X A, B, G;
    to_gf2x(A, a, aw);
    to_gf2x(B, b, bw);
    GCD(G, A, B);
    long d = deg(G);
    if (d <= 0) return 0;                      // gcd is 0 or 1: trivial
    if (gout && (u64)d + 1 <= keep_max_bits) from_gf2x(*gout, G);
    return (u64)d;
}

bool edf_least_ntl(const u64 *g, size_t gw, u64 target,
                   std::vector<u64> &out) {
    GF2X G;
    to_gf2x(G, g, gw);
    if (deg(G) <= 0) return false;
    vec_pair_GF2X_long fac;
    CanZass(fac, G);                           // g is squarefree (T is)
    bool have = false;
    std::vector<u64> best;
    for (long i = 0; i < fac.length(); i++) {
        if ((u64)deg(fac[i].a) != target) continue;
        std::vector<u64> w;
        from_gf2x(w, fac[i].a);
        if (!have || poly_less(w, best)) { best = w; have = true; }
    }
    if (have) out = best;
    return have;
}
#endif // HAVE_NTL

// One CanZass over the interval gcd g: every irreducible factor of g
// has degree inside the hit interval (deep_scan_handoff.md C2), so the
// least factor degree of T in that interval and its lexicographically
// least mask are read off directly -- no further full-size GCDs.
bool factor_min_degree_ntl(const u64 *g, size_t gw, u64 &min_deg,
                           std::vector<u64> &least_mask) {
    GF2X G;
    to_gf2x(G, g, gw);
    if (deg(G) <= 0) return false;
    vec_pair_GF2X_long fac;
    CanZass(fac, G);
    long best_d = -1;
    std::vector<u64> best;
    for (long i = 0; i < fac.length(); i++) {
        long d = deg(fac[i].a);
        if (d <= 0) continue;
        std::vector<u64> w;
        from_gf2x(w, fac[i].a);
        if (best_d < 0 || d < best_d ||
            (d == best_d && poly_less(w, best))) {
            best_d = d;
            best = w;
        }
    }
    if (best_d < 0) return false;
    min_deg = (u64)best_d;
    least_mask = best;
    return true;
}

#ifdef HAVE_NTL
// f | x^r + x^s + 1 ?   Used to confirm every reported factor: a wrong gcd
// or a wrong factorisation cannot survive this test.
bool trinomial_divisible_ntl(const u64 *f, size_t fw, u64 r, u64 s) {
    GF2X Fp;
    to_gf2x(Fp, f, fw);
    if (deg(Fp) <= 0) return false;
    GF2XModulus F;
    build(F, Fp);
    GF2X xr, xs;
    PowerXMod(xr, conv<ZZ>((unsigned long)r), F);
    PowerXMod(xs, conv<ZZ>((unsigned long)s), F);
    add(xr, xr, xs);
    add(xr, xr, GF2X(1L));
    return IsZero(xr);
}
#endif // HAVE_NTL

// ===========================================================================
// Hybrid HGCD (handoff item C4): reduce a full-size gcd to a small one
// with the classical polynomial half-gcd recursion, offloading every
// large multiplication through a GpuMulHook, then let NTL finish.  The
// polynomial case has none of the integer HGCD fixup subtleties, but we
// still verify the straddle postcondition at every level and fall back
// to plain NTL gcd on the original inputs if anything looks off.
#ifdef HAVE_NTL
#include <mutex>
#include <atomic>
#include <thread>
#include <NTL/config.h>
#define GF2_EC_HOST_ONLY
#include "gf2_ec.h"

u64 hyb_gpu_min_bits   = (u64)1 << 21;   // >= 2 Mbit -> offer to the GPU
u64 hyb_ntl_finish_bits = (u64)1 << 22;  // <= 4 Mbit -> NTL finishes
static u64 hyb_n_gpu = 0, hyb_n_cpu = 0, hyb_n_fb = 0;
static std::mutex hyb_stat_mu;

void hyb_get_stats(u64 &g, u64 &c, u64 &f) {
    std::lock_guard<std::mutex> lk(hyb_stat_mu);
    g = hyb_n_gpu; c = hyb_n_cpu; f = hyb_n_fb;
}
void hyb_reset_stats() {
    std::lock_guard<std::mutex> lk(hyb_stat_mu);
    hyb_n_gpu = hyb_n_cpu = hyb_n_fb = 0;
}

bool ntl_gf2x_backed() {
#ifdef NTL_GF2X_LIB
    return true;
#else
    return false;
#endif
}

long hyb_fault_at = 0;                   // test hook, see gf2_gcd.h
bool hyb_cert_xgcd = false;              // see gf2_gcd.h
bool hyb_force_fallback = false;         // test hook, see gf2_gcd.h
bool hyb_split_apply = true;             // see gf2_gcd.h
int hyb_threads = 1;                     // see gf2_gcd.h
u64 hyb_mt_min_bits = (u64)1 << 15;      // see gf2_gcd.h
bool hyb_threads_ok() {
#ifdef NTL_THREADS
    return true;
#else
    return false;
#endif
}
static std::atomic<long> hyb_fault_ctr{0};

namespace hyb {

using V = std::vector<u64>;

// ---- certification state: the folds of the CURRENT working pair, carried
// from the step that produced each operand to the step that consumes it.
// They are never recomputed in between, so an operand that changes in RAM
// between two steps breaks the next identity.
struct Cert {
    EcFold fa, fb;
    bool ok = true;
    u64 n = 0;               // identities verified
};
static inline void vfold(EcFold &f, const V &v) {
    ecf_fold_host(f, v.data(), v.size());
}
static bool fault_now() {                // test hook
    if (!hyb_fault_at) return false;
    return ++hyb_fault_ctr == hyb_fault_at;
}
static void flip_mid(V &v) {
    if (v.empty()) v.push_back(0);
    v[v.size() / 2] ^= 1ULL << 17;
}
static void flip_low(V &v) {             // a bit in the low quarter
    if (v.empty()) v.push_back(0);
    v[v.size() / 4] ^= 1ULL << 5;
}

static inline int64_t vdeg(const V &a) {
    for (size_t i = a.size(); i-- > 0;)
        if (a[i]) return (int64_t)(i * 64 + 63 - __builtin_clzll(a[i]));
    return -1;
}
static inline void vtrim(V &a) {
    size_t n = a.size();
    while (n && a[n - 1] == 0) n--;
    a.resize(n);
}
static inline bool vzero(const V &a) {
    for (u64 w : a) if (w) return false;
    return true;
}

// dst ^= src << k bits
static void vxor_shl(V &dst, const V &src, u64 k) {
    if (src.empty()) return;
    size_t wo = (size_t)(k >> 6);
    unsigned bo = (unsigned)(k & 63);
    size_t need = src.size() + wo + 1;
    if (dst.size() < need) dst.resize(need, 0);
    if (!bo) {
        for (size_t i = 0; i < src.size(); i++) dst[i + wo] ^= src[i];
    } else {
        u64 carry = 0;
        for (size_t i = 0; i < src.size(); i++) {
            dst[i + wo] ^= (src[i] << bo) | carry;
            carry = src[i] >> (64 - bo);
        }
        dst[src.size() + wo] ^= carry;
    }
}

static V vshr(const V &a, u64 k) {       // a >> k bits (drop low k)
    int64_t d = vdeg(a);
    if (d < (int64_t)k) return V();
    size_t wo = (size_t)(k >> 6);
    unsigned bo = (unsigned)(k & 63);
    size_t on = (size_t)((u64)d - k) / 64 + 1;
    V out(on, 0);
    for (size_t i = 0; i < on; i++) {
        u64 lo = a[i + wo];
        u64 hi = (i + wo + 1 < a.size()) ? a[i + wo + 1] : 0;
        out[i] = bo ? ((lo >> bo) | (hi << (64 - bo))) : lo;
    }
    vtrim(out);
    return out;
}

static V vlow(const V &a, u64 k) {       // a mod x^k (the low k bits)
    size_t nw = (size_t)((k + 63) >> 6);
    if (nw > a.size()) nw = a.size();
    V out(a.begin(), a.begin() + nw);
    if ((k & 63) && nw == (size_t)((k + 63) >> 6))
        out[nw - 1] &= (1ULL << (k & 63)) - 1;
    vtrim(out);
    return out;
}

// CPU product via NTL (routes through gf2x when NTL is so built)
static V cpu_mul(const V &a, const V &b) {
    NTL::GF2X A, B, C;
    to_gf2x(A, a.data(), a.size());
    to_gf2x(B, b.data(), b.size());
    mul(C, A, B);
    V out;
    from_gf2x(out, C);
    vtrim(out);
    return out;
}

// direct shift-xor product for small operands: avoids NTL conversion
// overhead on the millions of tiny matrix-update multiplies
static V small_mul(const V &a, const V &b) {
    V out(a.size() + b.size() + 1, 0);
    for (size_t i = 0; i < a.size(); i++) {
        u64 w = a[i];
        while (w) {
            int bit = __builtin_ctzll(w);
            w &= w - 1;
            vxor_shl(out, b, (u64)i * 64 + (unsigned)bit);
        }
    }
    vtrim(out);
    return out;
}

static V vmul(const V &a, const V &b, GpuMulHook *gm) {
    int64_t da = vdeg(a), db = vdeg(b);
    if (da < 0 || db < 0) return V();
    u64 ab = (u64)da + 1, bb = (u64)db + 1;
    u64 mn = ab < bb ? ab : bb;
    if (mn <= 512) {                     // tiny operand: direct shift-xor
        { std::lock_guard<std::mutex> lk(hyb_stat_mu); hyb_n_cpu++; }
        return (ab <= bb) ? small_mul(a, b) : small_mul(b, a);
    }
    if (gm && (ab >= hyb_gpu_min_bits || bb >= hyb_gpu_min_bits)) {
        V out;
        if (gm->mul(a.data(), a.size(), ab, b.data(), b.size(), bb, out)) {
            { std::lock_guard<std::mutex> lk(hyb_stat_mu); hyb_n_gpu++; }
            vtrim(out);
            return out;
        }
    }
    { std::lock_guard<std::mutex> lk(hyb_stat_mu); hyb_n_cpu++; }
    return cpu_mul(a, b);
}

static V vadd(V a, const V &b) {
    vxor_shl(a, b, 0);
    vtrim(a);
    return a;
}

// One full division step: (a, b) <- (b, a mod b); returns quotient.
// Shift-xor for small operands; NTL DivRem (subquadratic) for large.
// With C, the step is verified:  rem = a + q*b  in the fold ring.
static void cert_divstep(Cert &C, const V &q, V &rem) {
    if (fault_now()) flip_mid(rem);
    EcFold fq, fr, t;
    vfold(fq, q);
    vfold(fr, rem);
    ecf_mul(t, fq, C.fb);
    ecf_xor(t, C.fa);
    if (!ecf_eq(t, fr)) { C.ok = false; return; }
    C.fa = C.fb;
    C.fb = fr;
    C.n++;
}
static V divstep(V &a, V &b, Cert *C = nullptr) {
    int64_t da = vdeg(a), db = vdeg(b);
    if ((u64)da + 1 > ((u64)1 << 16)) {
        NTL::GF2X A, B, Q, R;
        to_gf2x(A, a.data(), a.size());
        to_gf2x(B, b.data(), b.size());
        DivRem(Q, R, A, B);
        V q, r;
        from_gf2x(q, Q);
        from_gf2x(r, R);
        vtrim(q); vtrim(r);
        a = b;
        b = r;
        if (C) cert_divstep(*C, q, b);
        return q;
    }
    V q;
    while (da >= db) {
        u64 sh = (u64)(da - db);
        size_t qw = (size_t)(sh >> 6);
        if (q.size() <= qw) q.resize(qw + 1, 0);
        q[qw] ^= 1ULL << (sh & 63);
        vxor_shl(a, b, sh);
        da = vdeg(a);
    }
    vtrim(a);
    std::swap(a, b);           // a = old b, b = remainder
    if (C) cert_divstep(*C, q, b);
    return q;
}

struct Mat2 {
    V m[2][2];
    bool ident = true;
};

static inline u64 vbits(const V &v) {
    int64_t d = vdeg(v);
    return d < 0 ? 0 : (u64)d + 1;
}

// (o0, o1) = (r0c0*x + r0c1*y, r1c0*x + r1c1*y) via the hook's batched
// frequency-domain path when any product is offload-sized; false if
// the hook declines or the batch does not qualify.
static bool try_mat2(const V &r0c0, const V &r0c1,
                     const V &r1c0, const V &r1c1,
                     const V &x, const V &y,
                     V &o0, V &o1, GpuMulHook *gm) {
    if (!gm) return false;
    u64 xb = vbits(x), yb = vbits(y);
    u64 eb[4] = {vbits(r0c0), vbits(r0c1), vbits(r1c0), vbits(r1c1)};
    u64 mx = 0;
    for (int i = 0; i < 4; i++) {
        u64 ob = eb[i] ? eb[i] + ((i & 1) ? yb : xb) : 0;
        if (ob > mx) mx = ob;
    }
    if (mx < hyb_gpu_min_bits) return false;
    GpuMulHook::Op m[4] = {
        {r0c0.data(), r0c0.size(), eb[0]}, {r0c1.data(), r0c1.size(), eb[1]},
        {r1c0.data(), r1c0.size(), eb[2]}, {r1c1.data(), r1c1.size(), eb[3]}};
    GpuMulHook::Op ox{x.data(), x.size(), xb}, oy{y.data(), y.size(), yb};
    if (!gm->mat2_apply(m, ox, oy, o0, o1)) return false;
    { std::lock_guard<std::mutex> lk(hyb_stat_mu); hyb_n_gpu += 2; }
    vtrim(o0);
    vtrim(o1);
    return true;
}

// Run job(0..n-1), on up to hyb_threads threads (the caller is one of them).
// The jobs are the independent products of one matrix operation: each reads
// shared operands and writes its own result.
template <class F> static void par_run(int n, F job) {
    int nt = hyb_threads < n ? hyb_threads : n;
    if (nt <= 1) {
        for (int i = 0; i < n; i++) job(i);
        return;
    }
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            int i = next.fetch_add(1);
            if (i >= n) break;
            job(i);
        }
    };
    std::vector<std::thread> th;
    th.reserve((size_t)nt - 1);
    for (int t = 1; t < nt; t++) th.emplace_back(worker);
    worker();
    for (auto &t : th) t.join();
}

// worth a thread: both operands of at least hyb_mt_min_bits
static inline bool mt_worth(const V &a, const V &b) {
    return vbits(a) >= hyb_mt_min_bits && vbits(b) >= hyb_mt_min_bits;
}

// (o0, o1) = (m00 x + m01 y, m10 x + m11 y) on CPU multiplications, the four
// products side by side when hyb_threads > 1 and they are large enough
static void cpu_mat2(const Mat2 &M, const V &x, const V &y, V &o0, V &o1,
                     GpuMulHook *gm) {
    const V *ml[4] = {&M.m[0][0], &M.m[0][1], &M.m[1][0], &M.m[1][1]};
    const V *op[4] = {&x, &y, &x, &y};
    V p[4];
    bool par = hyb_threads > 1 && !gm;
    if (par) {
        int big = 0;
        for (int i = 0; i < 4; i++) big += mt_worth(*ml[i], *op[i]);
        par = big >= 2;
    }
    if (par) par_run(4, [&](int i) { p[i] = vmul(*ml[i], *op[i], nullptr); });
    else for (int i = 0; i < 4; i++) p[i] = vmul(*ml[i], *op[i], gm);
    o0 = vadd(std::move(p[0]), p[1]);
    o1 = vadd(std::move(p[2]), p[3]);
}

// (a, b) <- M (a, b).  With C the two products are verified against the
// carried folds, whoever computed them (NTL/gf2x on the CPU or the GPU
// hook):  phi(a') = phi(M00) phi(a) + phi(M01) phi(b), likewise b'.
static void mat_apply(const Mat2 &M, V &a, V &b, GpuMulHook *gm,
                      Cert *C = nullptr) {
    if (M.ident) return;
    if (C && fault_now()) flip_mid(a);       // test: operand rots in RAM
    V na, nb;
    if (!try_mat2(M.m[0][0], M.m[0][1], M.m[1][0], M.m[1][1],
                  a, b, na, nb, gm))
        cpu_mat2(M, a, b, na, nb, gm);
    if (C) {
        if (fault_now()) flip_mid(nb);       // test: a product is wrong
        EcFold m00, m01, m10, m11, fna, fnb, t1, t2;
        vfold(m00, M.m[0][0]); vfold(m01, M.m[0][1]);
        vfold(m10, M.m[1][0]); vfold(m11, M.m[1][1]);
        vfold(fna, na);
        vfold(fnb, nb);
        ecf_mul(t1, m00, C->fa);
        ecf_mul(t2, m01, C->fb);
        ecf_xor(t1, t2);
        bool ok = ecf_eq(t1, fna);
        ecf_mul(t1, m10, C->fa);
        ecf_mul(t2, m11, C->fb);
        ecf_xor(t1, t2);
        ok = ok && ecf_eq(t1, fnb);
        if (!ok) C->ok = false;
        else { C->fa = fna; C->fb = fnb; C->n++; }
    }
    a = std::move(na);
    b = std::move(nb);
}

// (a, b) <- M (a, b), where the caller already holds
//     (ah, bh) = M (a >> k, b >> k)
// -- the pair the recursion that PRODUCED M reduced in place.  Writing
// a = ah0 x^k + al gives  M (a, b) = x^k (ah, bh) + M (al, bl), so only the
// low k bits have to be multiplied: the products are (deg M) x k instead of
// (deg M) x (deg a), about half the size at every level of the recursion.
// (gf2x/apps/halfgcd.cpp does the same; NTL's own HalfGCD does not.)
// The verification, when C is given, is exactly that of mat_apply: it is
// the identity for the WHOLE step, however a' and b' were assembled.
static void mat_apply_split(const Mat2 &M, V &a, V &b, const V &ah,
                            const V &bh, u64 k, GpuMulHook *gm,
                            Cert *C = nullptr) {
    if (M.ident) return;                     // then (ah, bh) is unchanged too
    if (C && fault_now()) flip_low(a);       // test: operand rots in RAM
    V al = vlow(a, k), bl = vlow(b, k);
    V na, nb;
    if (!try_mat2(M.m[0][0], M.m[0][1], M.m[1][0], M.m[1][1],
                  al, bl, na, nb, gm))
        cpu_mat2(M, al, bl, na, nb, gm);
    vxor_shl(na, ah, k);
    vxor_shl(nb, bh, k);
    vtrim(na);
    vtrim(nb);
    if (C) {
        if (fault_now()) flip_mid(nb);       // test: a product is wrong
        EcFold m00, m01, m10, m11, fna, fnb, t1, t2;
        vfold(m00, M.m[0][0]); vfold(m01, M.m[0][1]);
        vfold(m10, M.m[1][0]); vfold(m11, M.m[1][1]);
        vfold(fna, na);
        vfold(fnb, nb);
        ecf_mul(t1, m00, C->fa);
        ecf_mul(t2, m01, C->fb);
        ecf_xor(t1, t2);
        bool ok = ecf_eq(t1, fna);
        ecf_mul(t1, m10, C->fa);
        ecf_mul(t2, m11, C->fb);
        ecf_xor(t1, t2);
        ok = ok && ecf_eq(t1, fnb);
        if (!ok) C->ok = false;
        else { C->fa = fna; C->fb = fnb; C->n++; }
    }
    a = std::move(na);
    b = std::move(nb);
}

// M <- [[0,1],[1,q]] M   (the elementary matrix of one division step)
static void mat_lstep(Mat2 &M, const V &q, GpuMulHook *gm) {
    if (M.ident) {
        M.m[0][0] = V(); M.m[0][1] = V{1};
        M.m[1][0] = V{1}; M.m[1][1] = q;
        vtrim(M.m[1][1]);
        M.ident = false;
        return;
    }
    V n10 = vadd(vmul(q, M.m[1][0], gm), M.m[0][0]);
    V n11 = vadd(vmul(q, M.m[1][1], gm), M.m[0][1]);
    M.m[0][0] = std::move(M.m[1][0]);
    M.m[0][1] = std::move(M.m[1][1]);
    M.m[1][0] = std::move(n10);
    M.m[1][1] = std::move(n11);
}

static Mat2 mat_mul(const Mat2 &A, const Mat2 &B, GpuMulHook *gm) {
    if (A.ident) return B;
    if (B.ident) return A;
    Mat2 C;
    C.ident = false;
    if (hyb_threads > 1 && !gm && mt_worth(A.m[1][1], B.m[1][1])) {
        // the eight products of C = A B, side by side
        V p[8];
        par_run(8, [&](int t) {
            int i = t >> 2, j = (t >> 1) & 1, k = t & 1;
            p[t] = vmul(A.m[i][k], B.m[k][j], nullptr);
        });
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                C.m[i][j] = vadd(std::move(p[i * 4 + j * 2]),
                                 p[i * 4 + j * 2 + 1]);
        return C;
    }
    for (int j = 0; j < 2; j++) {
        // column j of C is a batched matrix-vector product with A
        if (try_mat2(A.m[0][0], A.m[0][1], A.m[1][0], A.m[1][1],
                     B.m[0][j], B.m[1][j], C.m[0][j], C.m[1][j], gm))
            continue;
        C.m[0][j] = vadd(vmul(A.m[0][0], B.m[0][j], gm),
                         vmul(A.m[0][1], B.m[1][j], gm));
        C.m[1][j] = vadd(vmul(A.m[1][0], B.m[0][j], gm),
                         vmul(A.m[1][1], B.m[1][j], gm));
    }
    return C;
}

static const u64 HGCD_BASE_BITS = 4096;

// In-place half-gcd: with n = deg a and mm = ceil(n/2), reduces (a, b)
// by unimodular steps until deg a >= mm > deg b, returning the transform
// (composed only when need_matrix).  `ok` is cleared if the classical
// postcondition ever fails -- caller then falls back to plain NTL.
// C (certification) is passed only by the top-level caller and is NOT
// handed down the recursion: the recursive calls work on truncated copies
// whose only product is the matrix, and ANY matrix applied through a
// verified mat_apply preserves "gcd(inputs) divides the pair".
static Mat2 hgcd(V &a, V &b, GpuMulHook *gm, bool need_matrix, bool &ok,
                 Cert *C = nullptr) {
    Mat2 M;
    int64_t n = vdeg(a);
    if (n <= 0) return M;
    u64 mm = ((u64)n + 1) / 2;
    if (vzero(b) || (u64)vdeg(b) < mm) return M;
    if ((u64)n <= HGCD_BASE_BITS) {
        int guard = 0;
        while (!vzero(b) && (u64)vdeg(b) >= mm) {
            V q = divstep(a, b, C);
            mat_lstep(M, q, gm);
            if (++guard > (int)HGCD_BASE_BITS + 4) { ok = false; break; }
        }
        return M;
    }
    // first recursion on the top halves
    {
        V a1 = vshr(a, mm), b1 = vshr(b, mm);
        Mat2 R = hgcd(a1, b1, gm, true, ok);
        if (!ok) return M;
        if (hyb_split_apply) mat_apply_split(R, a, b, a1, b1, mm, gm, C);
        else mat_apply(R, a, b, gm, C);
        M = std::move(R);
    }
    if (vzero(b) || (u64)vdeg(b) < mm) {
        if ((u64)vdeg(a) < mm) ok = false;
        return M;
    }
    // one division step across the straddle
    {
        V q = divstep(a, b, C);
        mat_lstep(M, q, gm);
    }
    if (vzero(b) || (u64)vdeg(b) < mm) {
        if ((u64)vdeg(a) < mm) ok = false;
        return M;
    }
    // second recursion on the adjusted top parts
    int64_t l = vdeg(a);
    if (l < (int64_t)mm || l >= 2 * (int64_t)mm) { ok = false; return M; }
    u64 k = (u64)(2 * (int64_t)mm - l);
    {
        V a3 = vshr(a, k), b3 = vshr(b, k);
        Mat2 S = hgcd(a3, b3, gm, true, ok);
        if (!ok) return M;
        if (hyb_split_apply) mat_apply_split(S, a, b, a3, b3, k, gm, C);
        else mat_apply(S, a, b, gm, C);
        if (need_matrix) M = mat_mul(S, M, gm);
    }
    if (!((u64)vdeg(a) >= mm && (vzero(b) || (u64)vdeg(b) < mm)))
        ok = false;
    return M;
}

} // namespace hyb

// Shared driver.  cert == nullptr: the original hybrid gcd.  Otherwise the
// certified one; *status as documented in gf2_gcd.h.
static u64 hyb_gcd_core(const u64 *aw_, size_t awn, const u64 *bw_, size_t bwn,
                        std::vector<u64> *gout, u64 keep_max_bits,
                        GpuMulHook *gm, hyb::Cert *C, int *status,
                        bool xgcd_only = false) {
    using namespace hyb;
    V a(aw_, aw_ + awn), b(bw_, bw_ + bwn);
    vtrim(a); vtrim(b);
    if (vdeg(a) < vdeg(b)) {
        std::swap(a, b);
        if (C) std::swap(C->fa, C->fb);
    }
    bool ok = true;
    int guard = 0;
    // xgcd_only (certified mode only): skip the reduction chain and go
    // straight to the verified extended gcd of the whole problem -- NTL's
    // XGCD, about 1.4x its GCD.  That is the reference certificate
    // (hyb_cert_xgcd) and the safety net when the chain declines.
    const bool chain = !(C && xgcd_only);
    while (chain && !vzero(b) && (u64)vdeg(b) > hyb_ntl_finish_bits && ok &&
           (!C || C->ok)) {
        if (hyb_force_fallback) { ok = false; break; }   // test hook
        if (vdeg(a) == vdeg(b)) {
            divstep(a, b, C);
            continue;
        }
        int64_t before = vdeg(a);
        hgcd(a, b, gm, false, ok, C);
        if (!ok) break;
        if (C && !C->ok) break;
        if (!vzero(b) && (u64)vdeg(b) > hyb_ntl_finish_bits)
            divstep(a, b, C);    // step past the straddle: strict halving
        if (vdeg(a) >= before) ok = false;
        if (++guard > 300) ok = false;
    }
    if (C && !C->ok) {                   // a step identity failed
        *status = GCD_CERT_FAIL;
        return 0;
    }
    if (!ok) {
        { std::lock_guard<std::mutex> lk(hyb_stat_mu); hyb_n_fb++; }
        if (C) {                         // the caller certifies another way
            *status = GCD_CERT_NONE;
            return 0;
        }
        return poly_gcd_ntl(aw_, awn, bw_, bwn, gout, keep_max_bits);
    }
    if (vzero(b)) {
        int64_t d = vdeg(a);
        if (C) {                         // the answer itself, as read now
            EcFold f;
            vfold(f, a);
            if (!ecf_eq(f, C->fa)) { *status = GCD_CERT_FAIL; return 0; }
        }
        if (d <= 0) return 0;
        if (gout && (u64)d + 1 <= keep_max_bits) {
            vtrim(a);
            a.push_back(0);      // match from_gf2x: minimal words + one 0
            *gout = std::move(a);
        }
        return (u64)d;
    }
    if (!C)
        return poly_gcd_ntl(a.data(), a.size(), b.data(), b.size(), gout,
                            keep_max_bits);
    // Certified finish: an extended gcd of the reduced pair.  NTL is the
    // (untrusted) prover; the Bezout identity  u a + v b = d  is checked
    // against the carried folds, so d is a combination of the inputs.
    GF2X A, B, D, U, W;
    to_gf2x(A, a.data(), a.size());
    to_gf2x(B, b.data(), b.size());
    XGCD(D, U, W, A, B);
    V dv, uv, wv;
    from_gf2x(dv, D);
    from_gf2x(uv, U);
    from_gf2x(wv, W);
    if (fault_now()) flip_mid(dv);       // test hook
    EcFold fd, fu, fw, t1, t2;
    vfold(fd, dv);
    vfold(fu, uv);
    vfold(fw, wv);
    ecf_mul(t1, fu, C->fa);
    ecf_mul(t2, fw, C->fb);
    ecf_xor(t1, t2);
    if (!ecf_eq(t1, fd)) { *status = GCD_CERT_FAIL; return 0; }
    C->n++;
    int64_t d = vdeg(dv);
    if (d <= 0) return 0;
    if (gout && (u64)d + 1 <= keep_max_bits) *gout = std::move(dv);
    return (u64)d;
}

u64 poly_gcd_hybrid(const u64 *aw_, size_t awn, const u64 *bw_, size_t bwn,
                    std::vector<u64> *gout, u64 keep_max_bits,
                    GpuMulHook *gm) {
    return hyb_gcd_core(aw_, awn, bw_, bwn, gout, keep_max_bits, gm, nullptr,
                        nullptr);
}

u64 poly_gcd_certified(const u64 *aw_, size_t awn, const u64 *bw_, size_t bwn,
                       std::vector<u64> *gout, u64 keep_max_bits,
                       GpuMulHook *gm, const EcFold *fa_expect,
                       const EcFold *fb_expect, int *status) {
    hyb::Cert C;
    ecf_fold_host(C.fa, aw_, awn);
    ecf_fold_host(C.fb, bw_, bwn);
    int st = GCD_CERT_OK;
    if ((fa_expect && !ecf_eq(C.fa, *fa_expect)) ||
        (fb_expect && !ecf_eq(C.fb, *fb_expect))) {
        if (status) *status = GCD_CERT_INPUT;
        return 0;
    }
    if (gout) gout->clear();
    const EcFold fa0 = C.fa, fb0 = C.fb;
    u64 d = hyb_gcd_core(aw_, awn, bw_, bwn, gout, keep_max_bits, gm, &C,
                         &st, hyb_cert_xgcd);
    if (st == GCD_CERT_NONE) {
        // The chain declined (a postcondition of the half-gcd did not hold:
        // never observed, but the engine is written to notice).  Certify
        // the original problem with one verified extended gcd instead, so
        // the answer is still a proof.
        C = hyb::Cert();
        C.fa = fa0;
        C.fb = fb0;
        st = GCD_CERT_OK;
        if (gout) gout->clear();
        d = hyb_gcd_core(aw_, awn, bw_, bwn, gout, keep_max_bits, nullptr,
                         &C, &st, true);
    }
    if (status) *status = st;
    return d;
}
void hyb_fault_reset() { hyb_fault_ctr = 0; }
long hyb_fault_count() { return hyb_fault_ctr.load(); }
#endif // HAVE_NTL

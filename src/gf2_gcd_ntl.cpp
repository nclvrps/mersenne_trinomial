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

// ===========================================================================
// Hybrid HGCD (handoff item C4): reduce a full-size gcd to a small one
// with the classical polynomial half-gcd recursion, offloading every
// large multiplication through a GpuMulHook, then let NTL finish.  The
// polynomial case has none of the integer HGCD fixup subtleties, but we
// still verify the straddle postcondition at every level and fall back
// to plain NTL gcd on the original inputs if anything looks off.
#ifdef HAVE_NTL
#include <mutex>

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

namespace hyb {

using V = std::vector<u64>;

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
static V divstep(V &a, V &b) {
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

// (a, b) <- M (a, b)
static void mat_apply(const Mat2 &M, V &a, V &b, GpuMulHook *gm) {
    if (M.ident) return;
    V na, nb;
    if (try_mat2(M.m[0][0], M.m[0][1], M.m[1][0], M.m[1][1],
                 a, b, na, nb, gm)) {
        a = std::move(na);
        b = std::move(nb);
        return;
    }
    na = vadd(vmul(M.m[0][0], a, gm), vmul(M.m[0][1], b, gm));
    nb = vadd(vmul(M.m[1][0], a, gm), vmul(M.m[1][1], b, gm));
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
static Mat2 hgcd(V &a, V &b, GpuMulHook *gm, bool need_matrix, bool &ok) {
    Mat2 M;
    int64_t n = vdeg(a);
    if (n <= 0) return M;
    u64 mm = ((u64)n + 1) / 2;
    if (vzero(b) || (u64)vdeg(b) < mm) return M;
    if ((u64)n <= HGCD_BASE_BITS) {
        int guard = 0;
        while (!vzero(b) && (u64)vdeg(b) >= mm) {
            V q = divstep(a, b);
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
        mat_apply(R, a, b, gm);
        M = std::move(R);
    }
    if (vzero(b) || (u64)vdeg(b) < mm) {
        if ((u64)vdeg(a) < mm) ok = false;
        return M;
    }
    // one division step across the straddle
    {
        V q = divstep(a, b);
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
        mat_apply(S, a, b, gm);
        if (need_matrix) M = mat_mul(S, M, gm);
    }
    if (!((u64)vdeg(a) >= mm && (vzero(b) || (u64)vdeg(b) < mm)))
        ok = false;
    return M;
}

} // namespace hyb

u64 poly_gcd_hybrid(const u64 *aw_, size_t awn, const u64 *bw_, size_t bwn,
                    std::vector<u64> *gout, u64 keep_max_bits,
                    GpuMulHook *gm) {
    using namespace hyb;
    V a(aw_, aw_ + awn), b(bw_, bw_ + bwn);
    vtrim(a); vtrim(b);
    if (vdeg(a) < vdeg(b)) std::swap(a, b);
    bool ok = true;
    int guard = 0;
    while (!vzero(b) && (u64)vdeg(b) > hyb_ntl_finish_bits && ok) {
        if (vdeg(a) == vdeg(b)) {
            divstep(a, b);
            continue;
        }
        int64_t before = vdeg(a);
        hgcd(a, b, gm, false, ok);
        if (!ok) break;
        if (!vzero(b) && (u64)vdeg(b) > hyb_ntl_finish_bits)
            divstep(a, b);       // step past the straddle: strict halving
        if (vdeg(a) >= before) ok = false;
        if (++guard > 300) ok = false;
    }
    if (!ok) {
        { std::lock_guard<std::mutex> lk(hyb_stat_mu); hyb_n_fb++; }
        return poly_gcd_ntl(aw_, awn, bw_, bwn, gout, keep_max_bits);
    }
    if (vzero(b)) {
        int64_t d = vdeg(a);
        if (d <= 0) return 0;
        if (gout && (u64)d + 1 <= keep_max_bits) {
            vtrim(a);
            a.push_back(0);      // match from_gf2x: minimal words + one 0
            *gout = std::move(a);
        }
        return (u64)d;
    }
    return poly_gcd_ntl(a.data(), a.size(), b.data(), b.size(), gout,
                        keep_max_bits);
}
#endif // HAVE_NTL

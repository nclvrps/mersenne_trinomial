// proto_tower_dlp.cpp -- reference prototype (coarse_sieve round 2 roadmap).
// Discrete logs in GF(2^d), d even, from 2^(d/2)-entry tables via the
// quadratic tower GF(2^d) = K(w), K = GF(2^(d/2)).  Validates exhaustively
// at d = 12..24 and by sampling (logs + Zech logs) at d = 36.
// Build: g++ -O2 -std=c++17 -I. -o proto_tower_dlp proto_tower_dlp.cpp
//        (needs gf2_wide_field.h, gf2_wide_setup.h next to it)
#include <algorithm>
#include <functional>
// All maps L -> K-coordinates are GF(2)-linear => byte-table lookups on a GPU.
#include "gf2_wide_field.h"
#include "gf2_wide_setup.h"
#include <vector>
#include <chrono>
#include <random>
#include <cstdio>
static double now() { return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
static u64 antilog(u64 idx, u64 q, int d, const u64 *gp) {
    u64 v = 1; int t = 0;
    while (idx) { if (idx & 1) v = gf_mul(v, gp[t], q, d); idx >>= 1; t++; }
    return v;
}
struct Tower {
    int d, m; u64 q, M, MK;           // MK = 2^m - 1
    u64 inv_den;                      // 1/(w + sigma(w))
    std::vector<u64> sig_col;         // sigma(x^i): Frobenius^m, a linear map
    std::vector<int> piv;             // pivot bit positions: K -> m-bit coords
    std::vector<u32> AK, LK;          // antilog/log of K* w.r.t. h = g^(2^m+1)
    std::vector<u64> T;               // T[c] = log_g(c + w)
    // byte tables: packed coords (aK | bK << m) of each byte's contribution
    std::vector<u64> btab;            // [nbytes][256]
    u64 lin_sigma(u64 e) const {      // sigma(e) = e^(2^m), GF(2)-linear
        u64 r = 0;
        for (int i = 0; i < d; i++)
            if (e >> i & 1) r ^= sig_col[i];
        return r;
    }
    u64 coordK(u64 y) const {         // y in K (as a d-bit element) -> m bits
        u64 c = 0;
        for (int k = 0; k < m; k++) c |= (y >> piv[k] & 1) << k;
        return c;
    }
    void split_slow(u64 e, u64 &a, u64 &b) const {   // reference split
        b = gf_mul(e ^ lin_sigma(e), inv_den, q, d);
        a = e ^ gf_mul(b, 2, q, d);
    }
    u64 coords_fast(u64 e) const {   // the GPU form: ceil(d/8) lookups + XORs
        u64 r = 0; for (int k = 0; 8 * k < d; k++) r ^= btab[k * 256 + (e >> (8*k) & 255)];
        return r;
    }
    u64 log(u64 e) const {            // e != 0
        u64 pk = coords_fast(e), ac = pk & MK_mask(), bc = pk >> m;
        if (bc == 0) return mulmod_u64(M / MK, LK[ac], M);
        u64 lb = LK[bc];
        u64 cc = ac ? AK[(LK[ac] + MK - lb) % MK] : 0;
        return (mulmod_u64(M / MK, lb, M) + T[cc]) % M;
    }
    u64 MK_mask() const { return ((u64)1 << m) - 1; }
};
// LK/AK need an index for each m-bit coordinate; LK[0] is unused.
static bool build(Tower &W, int d) {
    W.d = d; W.m = d / 2; W.q = find_primitive_poly(d);
    W.M = ((u64)1 << d) - 1; W.MK = ((u64)1 << W.m) - 1;
    const u64 q = W.q; u64 gp[64]; gp[0] = 2;
    for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], q, d);
    // Frobenius^m as a matrix: sigma(y) = y^(2^m)
    W.sig_col.resize(d);
    for (int i = 0; i < d; i++) {
        u64 y = (u64)1 << i;
        for (int k = 0; k < W.m; k++) y = gf_sqr(y, q, d);
        W.sig_col[i] = y;
    }
    u64 den = 2 ^ W.lin_sigma(2);                       // w + sigma(w), w = x
    W.inv_den = gf_pow(den, W.M - 1, q, d);
    // K basis: h^0..h^(m-1)?  Use powers of h until rank m, then RREF pivots.
    u64 h = gf_pow(2, W.M / W.MK, q, d);              // generator of K*
    std::vector<u64> rows; u64 y = 1;
    for (int k = 0; k < 4 * W.m && (int)rows.size() < W.m; k++, y = gf_mul(y, h, q, d)) {
        u64 v = y; for (u64 r : rows) { int p = 63 - __builtin_clzll(r); if (v >> p & 1) v ^= r; }
        if (v) { rows.push_back(v); std::sort(rows.begin(), rows.end(), std::greater<u64>()); }
    }
    if ((int)rows.size() != W.m) return false;
    // full RREF so each row has a unique pivot bit not present in the others
    for (size_t i = 0; i < rows.size(); i++) { int p = 63 - __builtin_clzll(rows[i]);
        for (size_t j = 0; j < rows.size(); j++) if (j != i && (rows[j] >> p & 1)) rows[j] ^= rows[i]; }
    W.piv.clear(); for (u64 r : rows) W.piv.push_back(63 - __builtin_clzll(r));
    // byte tables for e -> (coordK(a) | coordK(b) << m), linear in e
    int nb = (d + 7) / 8; W.btab.assign((size_t)nb * 256, 0);
    for (int k = 0; k < nb; k++) for (int byte = 0; byte < 256; byte++) {
        u64 e = (u64)byte << (8 * k); if (8 * k + 8 > d) e &= W.M;
        u64 a, b; W.split_slow(e, a, b);
        W.btab[k * 256 + byte] = W.coordK(a) | (W.coordK(b) << W.m); }
    // K* log/antilog in coordinates
    W.AK.assign(W.MK, 0); W.LK.assign(W.MK + 1, 0); y = 1;
    for (u64 k = 0; k < W.MK; k++, y = gf_mul(y, h, q, d)) {
        u64 c = W.coordK(y); W.AK[k] = (u32)c; W.LK[c] = (u32)k; }
    // T[c] = log(c + w): scan g^u until every c is hit (coupon collector)
    W.T.assign(W.MK + 1, ~0ULL); u64 left = W.MK + 1, e = 1;
    for (u64 u = 0; u < W.M && left; u++, e = gf_mulx(e, q, d)) {
        u64 pk = W.coords_fast(e), ac = pk & W.MK_mask(), bc = pk >> W.m;
        if (!bc) continue;
        u64 lb = W.LK[bc], cc = ac ? W.AK[(W.LK[ac] + W.MK - lb) % W.MK] : 0;
        if (W.T[cc] == ~0ULL) { W.T[cc] = (u + W.M - mulmod_u64(W.M / W.MK, lb, W.M)) % W.M; left--; }
    }
    return left == 0;
}
int main() {
    for (int d : {12, 16, 20, 24}) {                    // exhaustive
        Tower W; double t0 = now(); bool ok = build(W, d);
        u64 e = 1, bad = 0;
        for (u64 u = 0; u < W.M; u++, e = gf_mulx(e, W.q, d)) if (W.log(e) != u) bad++;
        printf("d=%2d exhaustive: %s, %llu/%llu logs wrong, tables 2x2^%d (%.2fs)\n", d,
               ok ? "built" : "BUILD FAIL", (unsigned long long)bad,
               (unsigned long long)W.M, W.m, now() - t0);
    }
    {   const int d = 36; Tower W; double t0 = now(); bool ok = build(W, d);
        double tb = now() - t0;
        u64 gp[64]; gp[0] = 2; for (int t = 1; t < d; t++) gp[t] = gf_sqr(gp[t - 1], W.q, d);
        std::mt19937_64 rng(1); u64 bad = 0, badz = 0; const int N = 300000;
        double tl = 0;
        for (int i = 0; i < N; i++) {
            u64 u = rng() % W.M, e = antilog(u, W.q, d, gp);
            double a = now(); u64 L = W.log(e); u64 Z = W.log(e ^ 1); tl += now() - a;
            if (L != u) bad++;
            if (antilog(Z, W.q, d, gp) != (e ^ 1)) badz++;   // Zech: g^Z(u) = g^u + 1
        }
        printf("d=36 sampled: %s in %.2fs, %d random u: %llu log errors, %llu Zech errors; "
               "%.0f ns per log on 1 CPU core; table memory %.1f MB (dense table: 550 GB)\n",
               ok ? "tables built" : "BUILD FAIL", tb, N, (unsigned long long)bad,
               (unsigned long long)badz, tl / (2.0 * N) * 1e9,
               ((W.AK.size() + W.LK.size()) * 4 + W.T.size() * 8 + W.btab.size() * 8) / 1048576.0);

    }
}

// fftbatch_bench.cpp -- is frequency-domain batching of the gcd engine's 2x2
// products worth it on the CPU?  (Round 5's answer: no.  See
// errorcheck_handoff.md section 6.5.)
//
// Compares, per operand size:
//   matrix x vector:  4 gf2x_mul   vs   6 dft + 2 addcompose_n(2) + 2 ift
//   matrix x matrix:  8 gf2x_mul   vs   8 dft + 4 addcompose_n(2) + 4 ift
// with gf2x's ternary (Schoenhage) FFT and with its Cantor FFT.
//
// Needs a gf2x that EXPORTS its FFT interface, i.e. a source build (the
// Ubuntu libgf2x does not):
//   g++ -O2 -march=native -I$PREFIX/include -o fftbatch_bench \
//       fftbatch_bench.cpp -L$PREFIX/lib -lgf2x
//   ./fftbatch_bench 2          (argument: repetitions, best time is kept)
//
// Sandbox result (gf2x from Gord's source, PCLMUL, untuned thresholds),
// seconds, one thread:
//    M words    x words |     4 mul   ternary    cantor |     8 mul   ternary    cantor
//        512        512 |    0.0001    0.0001    0.0003 |    0.0001    0.0002    0.0005
//       2048       2048 |    0.0009    0.0009    0.0017 |    0.0018    0.0018    0.0026
//       2048       4096 |    0.0011    0.0011    0.0028 |
//       8192       8192 |    0.0063    0.0062    0.0083 |    0.0121    0.0121    0.0131
//       8192      16384 |    0.0091    0.0076    0.0139 |
//      32768      32768 |    0.0308    0.0272    0.0436 |    0.0618    0.0491    0.0684
//      32768      65536 |    0.0512    0.0385    0.0720 |
//     131072     131072 |    0.1694    0.1362    0.2000 |    0.3430    0.2561    0.3192
//     131072     262144 |    0.2256    0.1913    0.3260 |
//     524288     524288 |    0.7089    0.6235    0.8678 |    1.4160    1.1466    1.3967
//     524288    1048576 |    1.1931    0.9724    1.4776 |
// i.e. 12-25% on products of 2 Mbit and more, nothing below 0.5 Mbit, and
// only on one thread: four whole products run side by side on four cores,
// the batched form cannot.
#include <cstring>
#include <cstdlib>
#include <gf2x.h>
#include <gf2x/gf2x-impl-export.h>
#include <gf2x/gf2x-ternary-fft.h>
#include <gf2x/gf2x-cantor-fft.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
typedef unsigned long ul;
static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static void rnd(std::vector<ul> &v) {
    static ul s = 12345;
    for (auto &w : v) { s = s * 6364136223846793005UL + 1442695040888963407UL; w = s ^ (s >> 29); }
}

#define BATCH(PFX, NAME)                                                     \
static double NAME(const std::vector<ul> M[4], const std::vector<ul> X[4],   \
                   int nx, std::vector<ul> out[4], bool &ok) {               \
    /* out[i*(nx/2)+j] = M[i][0] X[0][j] + M[i][1] X[1][j]; nx = 2 or 4 */   \
    size_t mw = M[0].size(), xw = X[0].size();                               \
    double t0 = now();                                                       \
    PFX##_info_t o;                                                          \
    if (PFX##_info_init(o, mw * 64, xw * 64) != 0) { ok = false; return 0; } \
    size_t sz[3];                                                            \
    PFX##_info_get_alloc_sizes(o, sz);                                       \
    int cols = nx / 2;                                                       \
    PFX##_ptr tm = PFX##_alloc(o, 4), tx = PFX##_alloc(o, nx),               \
              tc = PFX##_alloc(o, 1);                                        \
    PFX##_ptr t1 = (PFX##_ptr)malloc(sz[1]), t2 = (PFX##_ptr)malloc(sz[2]);  \
    for (int i = 0; i < 4; i++)                                              \
        PFX##_dft(o, PFX##_get(o, tm, i), M[i].data(), mw * 64, t1);         \
    for (int i = 0; i < nx; i++)                                             \
        PFX##_dft(o, PFX##_get(o, tx, i), X[i].data(), xw * 64, t1);         \
    for (int i = 0; i < 2; i++)                                              \
        for (int j = 0; j < cols; j++) {                                     \
            PFX##_srcptr ta[2] = {PFX##_get_const(o, tm, i * 2),             \
                                  PFX##_get_const(o, tm, i * 2 + 1)};        \
            PFX##_srcptr tb[2] = {PFX##_get_const(o, tx, j),                 \
                                  PFX##_get_const(o, tx, cols + j)};         \
            PFX##_zero(o, tc, 1);                                            \
            PFX##_addcompose_n(o, tc, ta, tb, 2, t2, t1);                    \
            out[i * cols + j].assign(mw + xw, 0);                            \
            PFX##_ift(o, out[i * cols + j].data(), (mw + xw) * 64, tc, t1);  \
        }                                                                    \
    free(t1); free(t2);                                                      \
    PFX##_free(o, tm, 4); PFX##_free(o, tx, nx); PFX##_free(o, tc, 1);       \
    PFX##_info_clear(o);                                                     \
    return now() - t0;                                                       \
}
BATCH(gf2x_ternary_fft, batch_ternary)
BATCH(gf2x_cantor_fft, batch_cantor)

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 1;
    printf("%10s %10s | %9s %9s %9s | %9s %9s %9s\n", "M words", "x words",
           "4 mul", "ternary", "cantor", "8 mul", "ternary", "cantor");
    for (size_t mw : {512, 2048, 8192, 32768, 131072, 524288}) {
        for (int shape = 0; shape < 2; shape++) {
            size_t xw = shape ? 2 * mw : mw;      // mat x vec uses (n/4, n/2) and (n/4, n/4)
            std::vector<ul> M[4], X[4], ref[4], o1[4], o2[4];
            for (int i = 0; i < 4; i++) { M[i].resize(mw); rnd(M[i]); X[i].resize(xw); rnd(X[i]); }
            double tv = 1e9, tvt = 1e9, tvc = 1e9, tm = 1e9, tmt = 1e9, tmc = 1e9;
            bool okt = true, okc = true, same = true;
            for (int r = 0; r < reps; r++) {
                // matvec: X[0], X[1] are the vector
                double t0 = now();
                std::vector<ul> p(mw + xw);
                for (int i = 0; i < 2; i++) {
                    ref[i].assign(mw + xw, 0);
                    for (int k = 0; k < 2; k++) {
                        gf2x_mul(p.data(), M[i * 2 + k].data(), mw, X[k].data(), xw);
                        for (size_t q = 0; q < mw + xw; q++) ref[i][q] ^= p[q];
                    }
                }
                double d = now() - t0; if (d < tv) tv = d;
                d = batch_ternary(M, X, 2, o1, okt); if (d < tvt) tvt = d;
                d = batch_cantor(M, X, 2, o2, okc); if (d < tvc) tvc = d;
                for (int i = 0; i < 2; i++) {
                    if (okt && o1[i] != ref[i]) same = false;
                    if (okc && o2[i] != ref[i]) same = false;
                }
                if (shape == 0) {
                    // matmul: 2x2 times 2x2 (columns X[0..1] and X[2..3])
                    t0 = now();
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++) {
                            ref[i * 2 + j].assign(mw + xw, 0);
                            for (int k = 0; k < 2; k++) {
                                gf2x_mul(p.data(), M[i * 2 + k].data(), mw, X[k * 2 + j].data(), xw);
                                for (size_t q = 0; q < mw + xw; q++) ref[i * 2 + j][q] ^= p[q];
                            }
                        }
                    d = now() - t0; if (d < tm) tm = d;
                    d = batch_ternary(M, X, 4, o1, okt); if (d < tmt) tmt = d;
                    d = batch_cantor(M, X, 4, o2, okc); if (d < tmc) tmc = d;
                    for (int i = 0; i < 4; i++) {
                        if (okt && o1[i] != ref[i]) same = false;
                        if (okc && o2[i] != ref[i]) same = false;
                    }
                }
            }
            if (shape == 0)
                printf("%10zu %10zu | %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f %s\n", mw, xw,
                       tv, okt ? tvt : -1, okc ? tvc : -1, tm, okt ? tmt : -1, okc ? tmc : -1,
                       same ? "" : "*** MISMATCH");
            else
                printf("%10zu %10zu | %9.4f %9.4f %9.4f | %s\n", mw, xw, tv,
                       okt ? tvt : -1, okc ? tvc : -1, same ? "" : "*** MISMATCH");
            fflush(stdout);
        }
    }
    return 0;
}

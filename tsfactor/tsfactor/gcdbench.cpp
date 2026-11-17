// gcdbench.cpp -- time the CPU gcd paths on one input, no GPU needed:
//     NTL GCD             tsfactor --gcd-ntl (and --no-ec): not checked
//     CERTIFIED, XGCD     tsfactor --ecg-xgcd: one NTL extended gcd whose
//                         Bezout identity is fold-verified
//     hybrid HGCD         tsfactor's own engine on CPU multiplications
//                         (with the old full and the new split application)
//     CERTIFIED, chain    tsfactor's DEFAULT: that engine with every step
//                         fold-verified and a verified finish
// The input is a random residue against the trinomial x^r + x^s + 1, the
// production shape.  The point is the RATIO "CERTIFIED, chain" / "NTL GCD"
// on your CPU with your gf2x build: below 1, the certified gcd is also the
// faster one; above 1, --gcd-ntl buys speed at the price of an unchecked gcd.
//
//   make gcdbench        (or: g++ -O2 -march=native -DHAVE_NTL -o gcdbench
//                         gcdbench.cpp gf2_gcd_ntl.cpp -lntl -lgf2x -lgmp
//                         -pthread)
//   ./gcdbench 136279841 588          (about six gcds' worth of time)
//   ./gcdbench 8000001                (a quick look at a smaller size)
//
// Optional third argument: the NTL-finish threshold in bits (default 2^22).
// Optional fourth argument "quick": skip the two slow reference rows (NTL
// XGCD and the old full-apply engine); anything else: all rows.
// Optional fifth argument N: also time the certified chain with the
// products of each matrix operation on N threads (tsfactor --gcd-mt N).

#include "gf2_gcd.h"
#define GF2_EC_HOST_ONLY
#include "gf2_ec.h"
#include <chrono>
#include <cstdlib>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    u64 r = argc > 1 ? strtoull(argv[1], 0, 10) : 8000001;
    u64 s = argc > 2 ? strtoull(argv[2], 0, 10) : r / 3;
    if (argc > 3) hyb_ntl_finish_bits = strtoull(argv[3], 0, 10);
    const bool quick = argc > 4 && argv[4][0] == 'q';
    const int mt = argc > 5 ? atoi(argv[5]) : 0;
    if (r < 100 || !(s >= 1 && s < r)) {
        fprintf(stderr, "usage: gcdbench r [s] [finish_bits] [quick|full] "
                        "[threads]\n");
        return 1;
    }
    printf("gcd of a random residue with x^%llu + x^%llu + 1; NTL %s gf2x; "
           "NTL finish below %llu bits\n", (unsigned long long)r,
           (unsigned long long)s, ntl_gf2x_backed() ? "WITH" : "WITHOUT",
           (unsigned long long)hyb_ntl_finish_bits);
    fflush(stdout);
    size_t nwt = (size_t)((r + 1 + 63) >> 6), nwa = (size_t)((r + 63) >> 6);
    std::vector<u64> T(nwt, 0), A(nwa, 0);
    T[0] |= 1;
    T[s >> 6] |= 1ULL << (s & 63);
    T[r >> 6] |= 1ULL << (r & 63);
    u64 st = 0x1234abcd;
    for (auto &w : A) {
        st += 0x9E3779B97F4A7C15ULL;
        u64 z = st;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        w = z ^ (z >> 31);
    }
    if (r & 63) A.back() &= (1ULL << (r & 63)) - 1;

    std::vector<u64> g1, g2, g3;
    double t0 = now_s();
    u64 d1 = poly_gcd_ntl(A.data(), A.size(), T.data(), T.size(), &g1, ~0ULL);
    double t_ntl = now_s() - t0;
    printf("  NTL GCD                      %9.2f s   (gcd degree %llu)\n",
           t_ntl, (unsigned long long)d1);
    fflush(stdout);

    auto what = [](int st) {
        return st == GCD_CERT_OK ? "certified"
             : st == GCD_CERT_NONE ? "NOT certified: engine fell back"
                                   : "CERTIFICATE FAILED";
    };
    EcFold fa, ft;
    std::vector<u64> g4 = g1, g5 = g1;
    int st_x = GCD_CERT_OK, st_c = -1;
    u64 d4 = d1, d5 = d1;
    ecf_fold_host(fa, A.data(), A.size());
    ecf_fold_host(ft, T.data(), T.size());
    if (!quick) {
        g4.clear();
        g5.clear();
        hyb_cert_xgcd = true;
        t0 = now_s();
        d4 = poly_gcd_certified(A.data(), A.size(), T.data(), T.size(), &g4,
                                ~0ULL, nullptr, &fa, &ft, &st_x);
        double t_x = now_s() - t0;
        hyb_cert_xgcd = false;
        printf("  CERTIFIED, NTL XGCD (--ecg-xgcd) %5.2f s   %+6.1f%% vs NTL   "
               "[%s]\n", t_x, 100.0 * (t_x - t_ntl) / t_ntl, what(st_x));
        fflush(stdout);

        // the engine as it was before round EC (every matrix applied to
        // the full operands), for comparison with the split application
        hyb_split_apply = false;
        t0 = now_s();
        d5 = poly_gcd_hybrid(A.data(), A.size(), T.data(), T.size(), &g5,
                             ~0ULL, nullptr);
        double t_full = now_s() - t0;
        printf("  hybrid HGCD, full apply (old) %8.2f s   %+6.1f%% vs NTL\n",
               t_full, 100.0 * (t_full - t_ntl) / t_ntl);
        fflush(stdout);
        hyb_split_apply = true;
    }

    hyb_reset_stats();
    t0 = now_s();
    u64 d2 = poly_gcd_hybrid(A.data(), A.size(), T.data(), T.size(), &g2,
                             ~0ULL, nullptr);
    double t_hyb = now_s() - t0;
    printf("  hybrid HGCD, split apply     %9.2f s   %+6.1f%% vs NTL\n",
           t_hyb, 100.0 * (t_hyb - t_ntl) / t_ntl);
    fflush(stdout);

    t0 = now_s();
    u64 d3 = poly_gcd_certified(A.data(), A.size(), T.data(), T.size(), &g3,
                                ~0ULL, nullptr, &fa, &ft, &st_c);
    double t_cert = now_s() - t0;
    printf("  CERTIFIED, chain (the default) %7.2f s   %+6.1f%% vs NTL, %+.1f%% "
           "vs hybrid   [%s]\n", t_cert, 100.0 * (t_cert - t_ntl) / t_ntl,
           100.0 * (t_cert - t_hyb) / t_hyb, what(st_c));
    bool same = d1 == d2 && d1 == d3 && d1 == d4 && d1 == d5 &&
                (d1 == 0 || (g1 == g2 && g1 == g3 && g1 == g4 && g1 == g5));
    fflush(stdout);
    if (mt > 1 && !hyb_threads_ok()) {
        printf("  (NTL was built without NTL_THREADS: no threaded row)\n");
    } else if (mt > 1) {
        std::vector<u64> g6;
        int st_m = -1;
        hyb_threads = mt;
        t0 = now_s();
        u64 d6 = poly_gcd_certified(A.data(), A.size(), T.data(), T.size(),
                                    &g6, ~0ULL, nullptr, &fa, &ft, &st_m);
        double t_mt = now_s() - t0;
        hyb_threads = 1;
        printf("  CERTIFIED, chain, --gcd-mt %-2d  %7.2f s   %+6.1f%% vs NTL, "
               "%.2fx as fast as 1 thread   [%s]\n", mt, t_mt,
               100.0 * (t_mt - t_ntl) / t_ntl, t_cert / t_mt, what(st_m));
        same = same && d6 == d1 && st_m == GCD_CERT_OK &&
               (d1 == 0 || g6 == g1);
    }
    printf("  all agree: %s\n", same ? "yes" : "*** NO ***");
    return same && st_x == GCD_CERT_OK && st_c == GCD_CERT_OK ? 0 : 1;
}

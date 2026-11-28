// Smallest irreducible factor degree of x^r + x^s + 1 over GF(2), every
// s in [1, (r-1)/2], by incremental distinct-degree factorisation (NTL).
// usage: survlaw r [r ...]   ->  lines "r s d" (d = 0: no factor <= r/2)
#include <NTL/GF2X.h>
#include <NTL/GF2XFactoring.h>
#include <cstdio>
#include <cstdlib>
using namespace NTL;
int main(int argc, char **argv) {
    for (int ai = 1; ai < argc; ai++) {
        long r = atol(argv[ai]);
        for (long s = 1; 2 * s < r; s++) {
            GF2X f; SetCoeff(f, r); SetCoeff(f, s); SetCoeff(f, 0);
            GF2XModulus F(f);
            GF2X h, x; SetX(x); h = x;
            long d = 0, B = 8;
            long found = 0;
            while (!found && d < r / 2) {
                GF2X h0 = h, acc; set(acc);
                long d0 = d;
                for (long i = 0; i < B && d < r / 2; i++) {
                    SqrMod(h, h, F); d++;
                    MulMod(acc, acc, h + x, F);
                }
                if (deg(GCD(acc, f)) > 0) {          // refine inside the block
                    h = h0;
                    for (long dd = d0 + 1; dd <= d; dd++) {
                        SqrMod(h, h, F);
                        if (deg(GCD(h + x, f)) > 0) { found = dd; break; }
                    }
                }
                if (B < 64) B *= 2;
            }
            printf("%ld %ld %ld\n", r, s, found);
        }
        fflush(stdout);
    }
    return 0;
}

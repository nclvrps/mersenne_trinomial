// minimal gf2x compile+link check used by the Makefile's auto-detection:
// is <gf2x.h> present and does gf2x_mul link?  If so the gcd engine calls
// gf2x directly (-DHAVE_GF2X) instead of going through NTL::mul.
#include <gf2x.h>
int main() {
    unsigned long a[1] = {3}, b[1] = {5}, c[2] = {0, 0};
    gf2x_mul(c, a, 1, b, 1);
    return c[0] == 15 ? 0 : 1;
}

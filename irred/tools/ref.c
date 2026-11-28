/* Naive reference: low 64 bits of x^(2^r) mod (x^r + x^m + 1) over GF(2).
   Bit-at-a-time, O(r^2); for testing only.

     ref r s        m = sodd, whichever of s, r-s is odd (as irred on a CPU)
     ref r s gpu    m = min(s, r-s)                      (as irred_gpu)
     ref r s M      m = M (M = s or r - s)

   Prints "r s x<16hex>" (m = s) / "r s y<16hex>" (m = r - s, the
   reciprocal trinomial) or "r s irreducible". */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ref r s [gpu|m]\n"); return 1; }
    int r = atoi(argv[1]), s = atoi(argv[2]);
    int so = (s & 1) ? s : r - s;
    if (argc > 3) so = strcmp(argv[3], "gpu") == 0 ? (s < r - s ? s : r - s)
                                                 : atoi(argv[3]);
    if (so != s && so != r - s) { fprintf(stderr, "m must be s or r - s\n"); return 1; }
    unsigned char *a = calloc(r, 1), *c = calloc(2 * r, 1);
    a[1] = 1;
    for (int k = 0; k < r; k++) {
        memset(c, 0, 2 * r);
        for (int i = 0; i < r; i++) c[2 * i] = a[i];
        for (int e = 2 * r - 2; e >= r; e--)
            if (c[e]) { c[e] = 0; c[e - r + so] ^= 1; c[e - r] ^= 1; }
        memcpy(a, c, r);
    }
    int isx = a[1];
    for (int i = 0; i < r; i++) if (i != 1 && a[i]) isx = 0;
    if (isx) { printf("%d %d irreducible\n", r, s); return 0; }
    uint64_t w = 0;
    for (int i = 0; i < 64 && i < r; i++) w |= (uint64_t)a[i] << i;
    printf("%d %d %c%016llx\n", r, s, s == so ? 'x' : 'y', (unsigned long long)w);
    return 0;
}

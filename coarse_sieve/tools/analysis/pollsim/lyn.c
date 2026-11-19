// count Lyndon words of length n (as n-bit integers, MSB first) per chunk of 2^k
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int n, k; int a[64]; uint64_t *cnt;
int main(int argc, char **argv) {
    n = atoi(argv[1]); k = atoi(argv[2]);
    uint64_t nch = (1ULL << (n - 1)) >> k;   // reps < 2^(n-1)
    cnt = calloc(nch + 1, 8);
    // FKM: generate necklaces in lex order; Lyndon when n % p == 0 and p == n
    for (int i = 0; i <= n; i++) a[i] = 0;
    int i = 1; uint64_t tot = 0;
    // standard iterative FKM
    int p = 1;
    a[0] = 0;
    while (1) {
        if (p == n) {   // Lyndon word a[1..n]
            uint64_t u = 0, ones = 0;
            for (int j = 1; j <= n; j++) { u = (u << 1) | a[j]; ones += a[j]; }
            if (u < (1ULL << (n - 1)) && 2 * ones < (uint64_t)n) { cnt[u >> k]++; tot++; }
        }
        // next pre-necklace
        i = n;
        while (i >= 1 && a[i] == 1) i--;
        if (i == 0) break;
        a[i] = 1;
        for (int j = i + 1; j <= n; j++) a[j] = a[j - i];
        p = i;
    }
    fprintf(stderr, "n=%d chunks=%llu canonical-ish Lyndon=%llu\n", n, (unsigned long long)nch, (unsigned long long)tot);
    fwrite(cnt, 8, nch, stdout);
    return 0;
}

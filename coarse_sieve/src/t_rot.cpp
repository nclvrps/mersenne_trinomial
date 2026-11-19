// t_rot.cpp: gf2f_is_min_rotation vs the all-rotations reference
//   exhaustive over odd u < 2^(d-1) for d = 2..24, then random u (with the
//   leading-zero count spread over 1..d-1) for d = 25..47; also checks the
//   class count against the necklace formula for the exhaustive range.
#include <cstdio>
#include <cstdint>
#include <chrono>
typedef uint64_t u64; typedef uint32_t u32;
#define GF2_HD static inline
#include "gf2_wide_field.h"
#include "gf2_dlog.h"

static u64 sm(u64 &s) {                        // splitmix64
    u64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// number of binary necklaces of length d with at least one 1 and one 0,
// minus nothing: count of minimal-rotation odd u < 2^(d-1) is
// (necklaces(d) - 2) for d >= 2 (all-zero and all-one excluded)
static u64 necklaces(int d) {
    // (1/d) sum_{k | d} phi(k) 2^(d/k)
    u64 s = 0;
    for (int k = 1; k <= d; k++) {
        if (d % k) continue;
        int ph = 0;
        for (int t = 1; t <= k; t++) {
            int a = t, b = k; while (b) { int c = a % b; a = b; b = c; }
            if (a == 1) ph++;
        }
        s += (u64)ph << (d / k);
    }
    return s / d;
}

int main() {
    u64 bad = 0, checked = 0;
    for (int d = 2; d <= 24; d++) {
        const u64 M = ((u64)1 << d) - 1;
        u64 reps = 0;
        for (u64 u = 1; u < ((u64)1 << (d - 1)); u += 2) {
            bool f = gf2f_is_min_rotation(u, d, M), r = gf2f_min_rotation_ref(u, d, M);
            checked++;
            if (f != r) { if (bad < 10) printf("d=%d u=%llx fast=%d ref=%d\n", d, (unsigned long long)u, f, r); bad++; }
            reps += r;
        }
        if (reps != necklaces(d) - 2) { printf("d=%d: %llu reps, necklaces-2 = %llu\n", d, (unsigned long long)reps, (unsigned long long)(necklaces(d) - 2)); bad++; }
    }
    u64 s = 12345;
    for (int d = 25; d <= 47; d++) {
        const u64 M = ((u64)1 << d) - 1;
        for (int i = 0; i < 4000000; i++) {
            int sh = (64 - (d - 1)) + (int)(sm(s) % (u64)(d - 1));   // z spread
            u64 u = (sm(s) >> sh) | 1;
            bool f = gf2f_is_min_rotation(u, d, M), r = gf2f_min_rotation_ref(u, d, M);
            checked++;
            if (f != r) { if (bad < 10) printf("d=%d u=%llx fast=%d ref=%d\n", d, (unsigned long long)u, f, r); bad++; }
        }
        // the extremes: u = 1, all-ones below the top bit, alternating, top-run patterns
        u64 extra[] = {1, M >> 1, (M >> 1) & 0x5555555555555555ULL, ((u64)1 << (d - 2)) | 1,
                       (M >> 2) | 1, (M >> 3), 3, 5, 7};
        for (u64 u : extra) {
            if (!(u & 1) || u >= ((u64)1 << (d - 1))) continue;
            bool f = gf2f_is_min_rotation(u, d, M), r = gf2f_min_rotation_ref(u, d, M);
            checked++;
            if (f != r) { printf("d=%d u=%llx fast=%d ref=%d (extra)\n", d, (unsigned long long)u, f, r); bad++; }
        }
    }
    // rough host timing on d=47 uniform odd u (the sweep's actual mix)
    const int d = 47; const u64 M = ((u64)1 << d) - 1;
    volatile u64 sink = 0;
    for (int pass = 0; pass < 2; pass++) {
        auto t0 = std::chrono::steady_clock::now();
        u64 cnt = 0, st = 777 + sink;
        for (u64 k = 0; k < 20000000; k++) {          // uniform odd u < 2^(d-1)
            u64 u = ((sm(st) & (M >> 1)) | 1) ^ (cnt & 2);
            cnt += pass ? gf2f_min_rotation_ref(u, d, M) : gf2f_is_min_rotation(u, d, M);
        }
        sink += cnt;
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("%s: %.1f ns/u (host)\n", pass ? "ref " : "fast", t / 20e6 * 1e9);
    }
    printf("%llu checked, %llu mismatches: %s\n", (unsigned long long)checked,
           (unsigned long long)bad, bad ? "FAIL" : "ALL OK");
    return bad != 0;
}

/*
Copyright (C) 2026  Gord Palameta
(actual author: Claude AI)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Some code may have been inspired by:
  GF2X library
  factor.cpp and other files in GF2X library apps subdirectory
  NTL library
*/

// gf2_trin_host.h
// Host-only helpers shared by the GPU and the CPU builds of tsfactor: a
// monotonic clock and the naive bit-level oracles for arithmetic modulo
// x^r + x^s + 1 that the selftests compare against.  (These used to live
// in gf2_trinomial_cuda.h; they moved here so that a build without CUDA
// headers can use them.)
#ifndef GF2_TRIN_HOST_H
#define GF2_TRIN_HOST_H

#include "gf2_cantor_engine.h"     // u64, bits_to_words, gf2x_naive_mul
#include <vector>
#include <chrono>

static inline double trin_now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// v (bits < `bits`) reduced in place modulo x^r + x^s + 1, bit at a time
static void cpu_mod_trinomial(std::vector<u64> &v, u64 bits, u64 r, u64 s) {
    for (u64 e = bits; e-- > r;) {
        if (!((v[e >> 6] >> (e & 63)) & 1)) continue;
        v[e >> 6] ^= 1ULL << (e & 63);
        u64 e1 = e - r, e2 = e - r + s;
        v[e1 >> 6] ^= 1ULL << (e1 & 63);
        v[e2 >> 6] ^= 1ULL << (e2 & 63);
    }
}
// a <- a^2 mod x^r + x^s + 1 (naive product, then the bit-level reduction)
static void cpu_square_mod(std::vector<u64> &a, u64 r, u64 s) {
    size_t ow = mul_out_words(r, r);
    std::vector<u64> sq(ow);
    gf2x_naive_mul(a.data(), r, a.data(), r, sq.data());
    cpu_mod_trinomial(sq, 2 * r - 1, r, s);
    a.assign(bits_to_words(r), 0);
    for (size_t i = 0; i < a.size(); i++) a[i] = sq[i];
}

#endif // GF2_TRIN_HOST_H

// cantor_ref.cpp
// Validation driver for the additive-FFT GF(2)[x] multiplier.
//
//   ./cantor_ref selftest          fast correctness suite
//   ./cantor_ref bigtest [bits]    one full-scale multiply (default
//                                  136279841-bit operands), verified by
//                                  residues modulo independent degree-61
//                                  irreducibles, with timing breakdown
//
// Oracles used (independent of the FFT machinery):
//   - bit-serial field multiply           (validates gf64_mul paths)
//   - naive bit-serial polynomial mult    (small sizes)
//   - word-comb schoolbook via cl64       (medium sizes)
//   - residues mod random irreducibles    (full scale)

#include "gf2_cantor_engine.h"
#include <chrono>
#include <cinttypes>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// xorshift PRNG (deterministic tests)
static u64 rng_state = 0x9E3779B97F4A7C15ULL;
static u64 rnd() {
    u64 x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}
static void rand_bits(std::vector<u64> &v, u64 bits) {
    v.assign(bits_to_words(bits), 0);
    for (auto &w : v) w = rnd();
    if (bits & 63) v.back() &= (~0ULL) >> (64 - (bits & 63));
}

static bool eq_bits(const u64 *a, const u64 *b, u64 bits) {
    size_t full = (size_t)(bits >> 6);
    for (size_t i = 0; i < full; i++) if (a[i] != b[i]) return false;
    if (bits & 63) {
        u64 m = (~0ULL) >> (64 - (bits & 63));
        if ((a[full] & m) != (b[full] & m)) return false;
    }
    return true;
}

static int fails = 0;
static void check(bool ok, const char *what) {
    printf("%-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) fails++;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "selftest";

    // --- primitive layer -------------------------------------------------
    {
        bool ok = true;
        for (int i = 0; i < 100000 && ok; i++) {
            u64 a = rnd(), b = rnd();
            u64 r0 = gf64_mul_bitref(a, b);
            if (gf64_mul(a, b) != r0) ok = false;
            if (gf64_mul_portable(a, b) != r0) ok = false;
        }
        // fixed edge cases
        ok = ok && gf64_mul(0, rnd()) == 0 && gf64_mul(1, 12345) == 12345
                && gf64_mul(~0ULL, ~0ULL) == gf64_mul_bitref(~0ULL, ~0ULL);
        check(ok, "gf64_mul: pclmul == portable == bit-serial (1e5 rand)");
    }

    // --- basis -----------------------------------------------------------
    CantorBasis B;
    B.build();
    printf("Cantor basis chain length: %d\n", B.len);
    check(B.len >= 28, "basis long enough for n = 2^28");
    check(B.verify(), "basis verifies (chain relations + independence)");

    if (!strcmp(mode, "selftest")) {
        // --- FFT round-trips ---------------------------------------------
        {
            bool ok = true;
            for (int m = 1; m <= 12 && ok; m++) {
                CantorFFT F; F.init(m, B);
                std::vector<u64> f(F.n), g;
                for (auto &x : f) x = rnd();
                g = f;
                F.fwd(g.data());
                F.inv(g.data());
                ok = (g == f);
                // linearity spot check: fwd(a^b) == fwd(a)^fwd(b)
                std::vector<u64> a(F.n), b2(F.n), s(F.n);
                for (size_t i = 0; i < F.n; i++) { a[i] = rnd(); b2[i] = rnd(); s[i] = a[i] ^ b2[i]; }
                F.fwd(a.data()); F.fwd(b2.data()); F.fwd(s.data());
                for (size_t i = 0; i < F.n; i++) if (s[i] != (a[i] ^ b2[i])) ok = false;
            }
            check(ok, "FFT round-trip + linearity, m = 1..12");
        }

        // --- known tiny product ------------------------------------------
        {
            u64 a = 3, b = 3, out[4];                 // (x+1)^2 = x^2+1
            gf2x_fft_mul(&a, 2, &b, 2, out, B);
            check(out[0] == 5, "(x+1)*(x+1) == x^2 + 1");
        }

        // --- small vs naive ----------------------------------------------
        {
            bool ok = true;
            for (int t = 0; t < 300 && ok; t++) {
                u64 abits = 1 + rnd() % 2500, bbits = 1 + rnd() % 2500;
                std::vector<u64> a, b;
                rand_bits(a, abits); rand_bits(b, bbits);
                size_t ow = mul_out_words(abits, bbits);
                std::vector<u64> c1(ow), c2(ow);
                gf2x_fft_mul(a.data(), abits, b.data(), bbits, c1.data(), B);
                gf2x_naive_mul(a.data(), abits, b.data(), bbits, c2.data());
                ok = eq_bits(c1.data(), c2.data(), abits + bbits - 1);
            }
            check(ok, "FFT mul == naive schoolbook (300 random <=2500b)");
        }

        // --- medium vs word comb -----------------------------------------
        {
            bool ok = true;
            u64 sizes[][2] = {{100000, 100000}, {262144, 262144},
                              {313371, 129003}, {1048576, 1048576}};
            for (auto &sz : sizes) {
                u64 abits = sz[0], bbits = sz[1];
                std::vector<u64> a, b;
                rand_bits(a, abits); rand_bits(b, bbits);
                size_t ow = mul_out_words(abits, bbits);
                std::vector<u64> c1(ow), c2(ow);
                gf2x_fft_mul(a.data(), abits, b.data(), bbits, c1.data(), B);
                gf2x_comb_mul(a.data(), abits, b.data(), bbits, c2.data());
                if (!eq_bits(c1.data(), c2.data(), abits + bbits - 1)) ok = false;
            }
            check(ok, "FFT mul == cl64 word-comb (0.1M..1M bit operands)");
        }

        // --- associativity / distributivity at 200k bits ------------------
        {
            u64 nb = 200000;
            std::vector<u64> a, b, c;
            rand_bits(a, nb); rand_bits(b, nb); rand_bits(c, nb);
            size_t ow2 = mul_out_words(nb, nb), ow3 = mul_out_words(2 * nb, nb);
            std::vector<u64> ab(ow2), bc(ow2), ab_c(ow3), a_bc(ow3);
            gf2x_fft_mul(a.data(), nb, b.data(), nb, ab.data(), B);
            gf2x_fft_mul(b.data(), nb, c.data(), nb, bc.data(), B);
            gf2x_fft_mul(ab.data(), 2 * nb, c.data(), nb, ab_c.data(), B);
            gf2x_fft_mul(a.data(), nb, bc.data(), 2 * nb, a_bc.data(), B);
            check(eq_bits(ab_c.data(), a_bc.data(), 3 * nb - 2), "(a*b)*c == a*(b*c) at 200k bits");
        }

        printf(fails ? "\nSELFTEST: %d FAILURES\n" : "\nSELFTEST: ALL PASS\n", fails);
        return fails ? 1 : 0;
    }

    if (!strcmp(mode, "bigtest")) {
        u64 nb = argc > 2 ? strtoull(argv[2], 0, 10) : 136279841ULL;
        printf("bigtest: multiplying two random %" PRIu64 "-bit operands\n", nb);

        std::vector<u64> a, b;
        rand_bits(a, nb); rand_bits(b, nb);
        size_t ca = (size_t)((nb + 31) >> 5);
        int m = 1;
        while (((size_t)1 << m) < 2 * ca - 1) m++;
        printf("chunks per operand: %zu, FFT size: 2^%d\n", ca, m);

        CantorFFT F;
        double t0 = now_s();
        F.init(m, B);
        printf("twiddle table build:      %7.2fs (%zu MB)\n", now_s() - t0,
               (F.n >> 1) * 8 >> 20);

        std::vector<u64> FA(F.n), FB(F.n);
        size_t ow = mul_out_words(nb, nb);
        std::vector<u64> out(ow);

        double tpack = now_s();
        size_t wa = bits_to_words(nb);
        for (size_t k = 0; k < F.n; k++) FA[k] = (k < ca) ? get_chunk(a.data(), wa, k) : 0;
        for (size_t k = 0; k < F.n; k++) FB[k] = (k < ca) ? get_chunk(b.data(), wa, k) : 0;
        double t1 = now_s();
        printf("pack:                     %7.2fs\n", t1 - tpack);
        F.fwd(FA.data());
        double t2 = now_s();
        printf("forward FFT (A):          %7.2fs\n", t2 - t1);
        F.fwd(FB.data());
        double t3 = now_s();
        printf("forward FFT (B):          %7.2fs\n", t3 - t2);
        for (size_t k = 0; k < F.n; k++) FA[k] = gf64_mul(FA[k], FB[k]);
        double t4 = now_s();
        printf("pointwise:                %7.2fs\n", t4 - t3);
        F.inv(FA.data());
        double t5 = now_s();
        printf("inverse FFT:              %7.2fs\n", t5 - t4);
        memset(out.data(), 0, ow * sizeof(u64));
        for (size_t k = 0; k < 2 * ca - 1; k++) {
            u64 e = FA[k];
            size_t pos = 32 * k, w = pos >> 6, sh = pos & 63;
            out[w] ^= e << sh;
            if (sh) out[w + 1] ^= e >> 32;
        }
        double t6 = now_s();
        printf("overlap-add:              %7.2fs\n", t6 - t5);
        printf("TOTAL multiply:           %7.2fs\n", t6 - tpack);

        // residue verification with 3 independent irreducibles
        bool ok = true;
        u64 seeds[3] = {0x1234567ULL, 0xDEADBEEFULL, 0xC0FFEEULL};
        for (int i = 0; i < 3; i++) {
            ModP p = find_irreducible_modp(61, seeds[i]);
            u64 ra = p.residue(a.data(), nb);
            u64 rb = p.residue(b.data(), nb);
            u64 rc = p.residue(out.data(), 2 * nb - 1);
            bool good = (p.mul(ra, rb) == rc);
            printf("residue check mod q=0x%" PRIx64 " (deg 61): %s\n", p.q,
                   good ? "PASS" : "FAIL");
            ok &= good;
        }
        printf(ok ? "\nBIGTEST: ALL PASS\n" : "\nBIGTEST: FAILURES\n");
        return ok ? 0 : 1;
    }

    fprintf(stderr, "unknown mode %s\n", mode);
    return 1;
}

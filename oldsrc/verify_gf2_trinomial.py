#!/usr/bin/env python3
"""
verify_gf2_trinomial.py

Independently verify, for each line of a CSV-like input file, that the
binary polynomial given by `mask` divides x^r + x^s + 1 over GF(2)[x].

This is meant as an independent double-check of files produced by the
GF2X/NTL-based verification tool used for Mersenne-prime "known factor"
certificates -- same math, deliberately different (simpler, unoptimized)
implementation, so that a bug in one is unlikely to be mirrored in the
other.

INPUT FORMAT
------------
Each line normally has three whitespace-separated fields:

    s d mask

  s    : integer, 1 <= s <= floor(r/2)
  d    : integer, 2 <= d <= floor(r/3)  (degree of the polynomial)
  mask : a string "p" followed by a hexadecimal number. Interpreted as a
         bit vector of length d+1 (bit i <-> coefficient of x^i), this is
         the polynomial P(x) of degree d we're told divides x^r+x^s+1.
         mask is always odd (constant term 1) and (per the file format)
         its bit-length should always be exactly d+1.

A handful of lines instead have exactly this reduced format (any fields
after the second are ignored):

    s u
    s primitive

These lines are simply skipped -- they don't describe a factor to check.

Any other line shape is treated as a malformed-input error.

VERIFICATION
------------
For each "s d mask" line, letting P(x) be the polynomial described by
mask (using its *actual* bit length, not the declared d -- see the
warning below), we check whether

    x^r + x^s + 1 (mod 2)   is divisible by   P(x) (mod 2)

by computing (x^r mod P(x)) XOR (x^s mod P(x)) XOR 1 in the ring
GF(2)[x]/(P(x)) and checking that it is the zero polynomial.

If mask's actual bit length differs from the declared d+1, a warning is
printed (to stderr) but verification proceeds using mask's actual degree.
If any line fails divisibility, or the file is malformed, an error is
printed (to stderr) and the program exits with status 1. On complete
success, the program exits with status 0 and (by default) prints
nothing to stdout, per the "silent success" requirement -- use
--progress for periodic stderr status if you want reassurance during a
long run.

ALGORITHM / COMPLEXITY NOTES
-----------------------------
Polynomials over GF(2) are represented as plain Python (or gmpy2) big
integers: bit i of the integer is the coefficient of x^i. Addition in
GF(2)[x] is then just XOR.

Multiplication (gf2_mul) cannot be done with ordinary integer
multiplication, because that has carries and GF(2) polynomial
multiplication must not. Instead we use a textbook Karatsuba-style
recursive multiply with XOR standing in for +/-: this is a standard,
easily-checked algebraic identity (the same one used for ordinary
Karatsuba multiplication), and gives roughly O(n^1.585) instead of
O(n^2).

Reduction modulo P(x) (gf2_mod) is done by the direct, "obviously
correct" method: repeatedly XOR a shifted copy of P(x) into the
remainder to cancel its current leading term, exactly mirroring
schoolbook polynomial long division. This is simple to verify by
inspection against the mathematical definition, but its cost grows
roughly quadratically with deg(P). For the overwhelming majority of
lines (d is small -- the problem states the distribution of d falls off
like ~(16/9)/d^2) this is instant. If the input file ever contains a
pathologically large d (many millions of bits) this step could become
slow (potentially minutes to hours for a single such line). Given this
tool is meant purely as an occasional, independent double-check
(clarity/correctness prioritized over raw speed), that tradeoff was
made deliberately; a genuinely sub-quadratic reduction is possible using
a Newton-iteration-based reciprocal ("Barrett reduction" for
polynomials) built on top of the same gf2_mul, but that algorithm is
considerably more intricate and easier to get subtly wrong, so it was
left out of this version.

r is expected to be an odd prime (in the cases of interest, the exponent
of a Mersenne prime), but nothing here relies on that beyond it being a
positive integer.
"""

import sys
import argparse

# --------------------------------------------------------------------------
# Optional gmpy2 acceleration (purely an implementation detail -- gmpy2's
# mpz supports the same bit_length()/<</>>/^/& operations we use on plain
# Python ints, so the rest of the code doesn't need to know which one it's
# using).
# --------------------------------------------------------------------------
try:
    import gmpy2
    _HAVE_GMPY2 = True
except ImportError:
    _HAVE_GMPY2 = False


def _parse_hex(hexpart):
    """Parse a hex string into a big integer, using gmpy2 if available."""
    if _HAVE_GMPY2:
        return gmpy2.mpz(hexpart, 16)
    return int(hexpart, 16)


# --------------------------------------------------------------------------
# GF(2)[x] polynomial arithmetic
# --------------------------------------------------------------------------

def gf2_mul(a, b, base_bits=256):
    """
    Multiply two GF(2)[x] polynomials, represented as integers (bit i <->
    coefficient of x^i). Standard Karatsuba split, with XOR in place of
    +/- (valid in any commutative ring, GF(2) included):

        a = ah*X^m + al,   b = bh*X^m + bl
        a*b = ah*bh*X^2m ^ [(ah^al)*(bh^bl) ^ ah*bh ^ al*bl]*X^m ^ al*bl

    Below some size threshold we fall back to a simple "for each set bit
    of the shorter operand, XOR in a shifted copy of the other operand"
    schoolbook multiply, which is easy to verify directly against the
    definition of polynomial multiplication.
    """
    if a == 0 or b == 0:
        return 0

    la, lb = a.bit_length(), b.bit_length()

    if la <= base_bits or lb <= base_bits:
        # Schoolbook base case.
        if lb < la:
            a, b = b, a
            la, lb = lb, la
        result = 0
        shift = 0
        aa = a
        while aa:
            if aa & 1:
                result ^= (b << shift)
            aa >>= 1
            shift += 1
        return result

    m = max(la, lb) // 2
    mask = (1 << m) - 1
    ah, al = a >> m, a & mask
    bh, bl = b >> m, b & mask

    z0 = gf2_mul(al, bl, base_bits)
    z2 = gf2_mul(ah, bh, base_bits)
    z1 = gf2_mul(al ^ ah, bl ^ bh, base_bits) ^ z0 ^ z2

    return z0 ^ (z1 << m) ^ (z2 << (2 * m))


def gf2_mod(a, p, d):
    """
    Reduce polynomial `a` modulo the monic polynomial `p` of degree `d`
    (i.e. p.bit_length() == d+1), returning a polynomial of degree < d.

    Directly mirrors schoolbook long division: while deg(a) >= d, cancel
    a's current leading term by XORing in p shifted up to match it.
    """
    while True:
        la = a.bit_length()
        if la <= d:
            return a
        shift = (la - 1) - d
        a ^= (p << shift)


def modpow_x(exp, p, d):
    """
    Compute x^exp mod p(x), where p is monic of degree d, via standard
    right-to-left binary exponentiation ("square and multiply") in the
    ring GF(2)[x]/(p(x)).
    """
    result = 1                   # x^0 = 1
    base = gf2_mod(0b10, p, d)   # x^1, reduced (no-op unless d < 1)
    e = exp
    while e > 0:
        if e & 1:
            result = gf2_mod(gf2_mul(result, base), p, d)
        e >>= 1
        if e:
            base = gf2_mod(gf2_mul(base, base), p, d)
    return result


def divides_trinomial(p, d, r, s):
    """
    Return True iff P(x) (monic, degree d, given by integer p) divides
    x^r + x^s + 1 over GF(2)[x].
    """
    xr = modpow_x(r, p, d)
    xs = modpow_x(s, p, d)
    return (xr ^ xs ^ 1) == 0


# --------------------------------------------------------------------------
# Self-test: a handful of small, hand-verifiable cases exercising the same
# code paths used on the real (huge) data, so correctness doesn't have to
# be taken purely on faith.
# --------------------------------------------------------------------------

def _run_selftest():
    failures = []

    def check(desc, cond):
        status = "ok" if cond else "FAIL"
        print(f"[{status}] {desc}")
        if not cond:
            failures.append(desc)

    # --- gf2_mul sanity checks -------------------------------------------
    # (x^2+1)*(x+1) = x^3+x^2+x+1  ->  0b101 * 0b11 == 0b1111
    check("gf2_mul((x^2+1),(x+1)) == x^3+x^2+x+1",
          gf2_mul(0b101, 0b11) == 0b1111)

    # (x^2+x+1)^2 = x^4+x^2+1  (Frobenius: squaring is linear in char 2)
    check("gf2_mul(x^2+x+1, x^2+x+1) == x^4+x^2+1",
          gf2_mul(0b111, 0b111) == 0b10101)

    # --- gf2_mod sanity checks --------------------------------------------
    # x^3+1 = (x+1)(x^2+x+1)  ->  remainder 0 when dividing by x+1
    check("(x^3+1) mod (x+1) == 0",
          gf2_mod(0b1001, 0b11, 1) == 0)

    # x^3+x+1 is irreducible of degree 3 -> not divisible by x+1 (remainder 1)
    check("(x^3+x+1) mod (x+1) == 1",
          gf2_mod(0b1011, 0b11, 1) == 1)

    # --- mask parsing sanity check (matches worked example in the prompt) -
    m = _parse_hex("29")
    check("hex 'p29' decodes to x^5+x^3+1 (degree 5)",
          int(m) == 0b101001 and int(m).bit_length() - 1 == 5)

    # --- divisibility: x^2+x+1 divides x^4+x^2+1 (it's the square of it) --
    check("x^2+x+1 divides x^4+x^2+1  (r=4, s=2)",
          divides_trinomial(0b111, 2, 4, 2) is True)

    # --- non-divisibility: x^2+x+1 does NOT divide x^6+x^3+1 --------------
    check("x^2+x+1 does NOT divide x^6+x^3+1  (r=6, s=3)",
          divides_trinomial(0b111, 2, 6, 3) is False)

    # --- a slightly bigger, still hand-checkable case ----------------------
    # x^3+x+1 is irreducible with x a primitive element mod it (order 7,
    # since 2^3-1=7 is prime), so x^7 == 1, hence x^7+x^1+1 == 1+x+1 == x,
    # which is not zero -- a non-divisibility check with a larger exponent.
    check("x^3+x+1 does NOT divide x^7+x^1+1  (r=7, s=1)",
          divides_trinomial(0b1011, 3, 7, 1) is False)

    # A genuine larger-exponent positive case: x^2+x+1 has order 3, so
    # x^300 == 1 and x^201 == 1 (both multiples of 3) -> x^300+x^201+1
    # == 1^1^1 == 1 (not zero). Instead use exponents 301 (301%3==1 -> x)
    # and 302 (302%3==2 -> x+1): x ^ (x+1) ^ 1 == 0 -> should divide.
    check("x^2+x+1 divides x^301+x^302+1  (r=302, s=301)",
          divides_trinomial(0b111, 2, 302, 301) is True)

    print()
    if failures:
        print(f"SELF-TEST FAILED ({len(failures)} failure(s))", file=sys.stderr)
        return False
    print("All self-tests passed.")
    return True


# --------------------------------------------------------------------------
# Main file-processing loop
# --------------------------------------------------------------------------

def process_file(r, path, progress_every=0):
    checked = 0
    skipped = 0

    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2 and parts[1] in ("u", "primitive"):
                skipped += 1
                continue

            if len(parts) < 3:
                sys.exit(
                    f"ERROR: line {lineno}: expected 'S D MASK' (or 'S u' / "
                    f"'S primitive'), got: {line!r}"
                )

            s_str, d_str, mask_str = parts[0], parts[1], parts[2]

            try:
                s = int(s_str)
            except ValueError:
                sys.exit(f"ERROR: line {lineno}: invalid integer for s: {s_str!r}")

            try:
                d = int(d_str)
            except ValueError:
                sys.exit(f"ERROR: line {lineno}: invalid integer for d: {d_str!r}")

            if not mask_str.startswith("p"):
                sys.exit(
                    f"ERROR: line {lineno}: mask field {mask_str!r} does not "
                    f"start with 'p'"
                )
            hexpart = mask_str[1:]
            try:
                p = _parse_hex(hexpart)
            except ValueError:
                sys.exit(
                    f"ERROR: line {lineno}: mask field {mask_str!r} is not "
                    f"valid hex after the leading 'p'"
                )

            p_int = int(p)  # normalize to a plain int for bit_length etc.
            if p_int <= 0 or (p_int & 1) == 0:
                sys.exit(
                    f"ERROR: line {lineno}: mask decodes to {p_int}, which "
                    f"is not a positive odd number (constant term must be 1)"
                )

            actual_bitlen = p_int.bit_length()
            if actual_bitlen != d + 1:
                print(
                    f"WARNING: line {lineno}: declared d={d} implies bit "
                    f"length {d + 1}, but mask has bit length {actual_bitlen} "
                    f"(using actual value)",
                    file=sys.stderr,
                )
            actual_d = actual_bitlen - 1

            if not divides_trinomial(p_int, actual_d, r, s):
                sys.exit(
                    f"ERROR: line {lineno}: polynomial from mask "
                    f"({mask_str}) does NOT divide x^{r} + x^{s} + 1 "
                    f"(s={s}, d={d})"
                )

            checked += 1
            if progress_every and checked % progress_every == 0:
                print(
                    f"... {checked} lines verified, {skipped} skipped "
                    f"(line {lineno})",
                    file=sys.stderr,
                )

    if progress_every:
        print(
            f"Done: {checked} lines verified, {skipped} lines skipped.",
            file=sys.stderr,
        )
    # Silent on success (per spec) unless --progress was requested above.


def main():
    parser = argparse.ArgumentParser(
        description="Independently verify that each mask polynomial in a "
                     "file divides x^r + x^s + 1 over GF(2)."
    )
    parser.add_argument("r", nargs="?", type=int,
                         help="the constant exponent r (e.g. 136279841)")
    parser.add_argument("input_file", nargs="?",
                         help="path to the input file")
    parser.add_argument("--progress", type=int, default=0, metavar="N",
                         help="print a status line to stderr every N "
                              "verified lines (default: silent)")
    parser.add_argument("--selftest", action="store_true",
                         help="run built-in self-tests and exit")
    args = parser.parse_args()

    if args.selftest:
        ok = _run_selftest()
        sys.exit(0 if ok else 1)

    if args.r is None or args.input_file is None:
        parser.error("r and input_file are required unless --selftest is given")

    process_file(args.r, args.input_file, progress_every=args.progress)


if __name__ == "__main__":
    main()

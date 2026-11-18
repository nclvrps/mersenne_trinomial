# irred 0.20 — handoff (2026-11-17)

Fork of Brent's irred 3.15 (github.com/nclvrps/irred): r squarings computing x^(2^r) mod (x^r + x^sodd + 1); irreducible iff the result is x.

## Usage
`irred [options] r s` — options: `-v`, `--progress N`, `--no-checkpoint|--no-ckpt`, `--checkpoint-mins|--ckpt-mins M` (default 30, 0 = only on signal), `--selftest`, `--isa auto|avx512|avx2|generic`.
Result line (old log format): `r s irreducible (may be primitive) LOG` or `r s x|y<16 hex> LOG` (low 64 bits; low 8 hex digits = old 32-bit residue; y = reciprocal used because s is even).
Exit status: 0 done, 1 error, 2 stopped by SIGINT/SIGTERM after checkpointing.
Limit: r − sodd ≥ 128 (s ≥ 128 if s even, r − s ≥ 128 if s odd).

## Checkpoints
`irred_r<r>_s<s>.ckpt` (+ `.ckpt.1`, `.ckpt.2`), magic `IRREDCKP`, CRC-32 on the header and the data, written via tmp file, then fsync, then rename. On start, resumes from the newest valid file for the same (r, s); others are ignored with a message. Deleted on completion. `irred_r<r>_s<s>.lock` (flock) prevents two runs on the same (r, s) in one directory. Size is about r/8 bytes (17 MB at r = 136279841, about 0.07 s to write).

## Bug found in 0.14 (fixed in 0.20)
When (r − sodd) ≡ 0 (mod 128) (deltaq == 0), 0.14's `xor_shift_zero` call folded words q4..q11−1 instead of q4+1..q11, giving wrong residues. It could also miss an irreducible trinomial. The 2003 code was correct. For r = 136279841 (≡ 33 mod 128) the affected s are s ≡ 33 or s ≡ 0 (mod 128). Any 0.14 result for those s should be rerun.

## Performance (2.1 GHz Xeon, AVX-512, 260 MB L3, gcc 13 -O3 -march=native)
- r = 756839: 0.14 took 15.4 s; 0.20 takes 3.3 s.
- r = 136279841: 0.14 took about 3.7 ms per squaring (5.9 days); 0.20 takes about 1.8 ms (2.9 days). This size is memory-bound.
- Changes: AVX2/AVX-512BW pshufb interleave, chosen at run time; the reduction is split into overlap-free blocks so its `restrict` is valid and it vectorises; forward-only interleave (alternating was about 10% slower); target_clones for the reduction loops.
- Tried and rejected: PDEP (slower than AVX2), non-temporal stores (2x slower when the data fits in L3).

## Verification
- Naive bit-by-bit reference over about 3,600 (r, s) cases: boundaries, all deltaq == 0 s, small sodd, r ≡ 127 (mod 128), and builds for AVX-512/AVX2/generic, gcc/clang and ASan+UBSan.
- Low 32 bits agree with the patched 2003 program at r = 44497 and r = 132049.
- Checkpoint interrupt/resume equivalence at r = 1398269 and r = 136279841.

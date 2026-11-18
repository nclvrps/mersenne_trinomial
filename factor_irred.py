#!/usr/bin/env python3
import os
import sys
import argparse

def positive_int(value):
    """Custom type checker for argparse to ensure a positive integer."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")
    
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' must be a positive integer (greater than 0).")
    return ivalue

def main():
    parser = argparse.ArgumentParser(
        description="""
Wrapper for factor_ec to output 32-bit residues like irred.

[Bug in original "factor" (not the GNU utility) and factor_ec:
if x^r + x^s + 1 is irreducible, then
factor 2.01 will crash with signal 11 (SIGSEGV, segmentation fault)
instead of correctly printing the residue, which is equal to 1.
This is fixed in my non-official version 2.02]
"""
    )

    parser.add_argument('-s0', required=True, type=positive_int, 
        help="r > s0 > 0"
    )

    parser.add_argument('-s1', required=False, type=positive_int, default=None,
        help="r > s1 > 0, s1 != s0, default = s0 + 1"
    )

    parser.add_argument('r', type=positive_int, 
        help="r > 0"
    )

    parser.add_argument('--path', '-p', type=str, default="tsfactor/factor_ec/factor_ec",
        help="path to the 'factor_ec' program"
    )

    args = parser.parse_args()

    s0_value = args.s0
    r_value = args.r

    if args.s1 is None:
        s1_value = s0_value + 1
    else:
        s1_value = args.s1

    if s0_value >= r_value:
        parser.error(f"-s0 ({s0_value}) must be less than r ({r_value}).")
        
    if s1_value >= r_value:
        parser.error(f"-s1 ({s1_value}) must be less than r ({r_value}).")

    if s1_value == s0_value:
        parser.error(f"-s1 ({s1_value}) must be different from ({s0_value}).")

    #path_to_factor = "./factor"
    path_to_factor = args.path

    hardcoded_args = [
        "-m", "1",
        "-q", "1",
        "-f", "0",
        "-skip", f"{r_value - 1}"
    ]

    # Reconstruct the exact string arguments to pass to 'foo'
    new_argv = [
        path_to_factor,
        *hardcoded_args,
        "-s0", str(s0_value),
        "-s1", str(s1_value),
        str(r_value)
    ]

    try:
        os.execv(path_to_factor, new_argv)
    except OSError as e:
        print(f"Error executing {path_to_factor}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

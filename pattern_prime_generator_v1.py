#!/usr/bin/env python3

import argparse
import math
from typing import List, Tuple


# Deterministic Miller-Rabin for 64-bit integers.
# Good for all n < 2^64.
def is_prime(n: int) -> bool:
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0

    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic bases for 64-bit range.
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in bases:
        if a >= n:
            continue

        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        ok = False

        for _ in range(s - 1):
            x = (x * x) % n

            if x == n - 1:
                ok = True
                break

        if not ok:
            return False

    return True


def parse_pattern(pattern: str) -> List[int]:
    # Accepts things like:
    # quad_0_6_84_90
    # 0_6_84_90
    parts = pattern.split("_")

    offsets = []

    for part in parts:
        try:
            offsets.append(int(part))
        except ValueError:
            pass

    return offsets


def build_wheel(offsets: List[int], wheel_primes: List[int]) -> Tuple[int, List[int]]:
    modulus = 1

    for p in wheel_primes:
        modulus *= p

    allowed = []

    for r in range(modulus):
        ok = True

        for q in wheel_primes:
            for off in offsets:
                if (r + off) % q == 0:
                    ok = False
                    break

            if not ok:
                break

        if ok:
            allowed.append(r)

    return modulus, allowed


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--pattern", required=True,
                        help="Pattern like quad_0_6_84_90 or 0_6_84_90")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit", type=int, default=10000000,
                        help="How many numbers above start to search")
    parser.add_argument("--max-results", type=int, default=20)

    args = parser.parse_args()

    offsets = parse_pattern(args.pattern)

    if not offsets or offsets[0] != 0:
        print("Pattern must contain offset 0, e.g. quad_0_6_84_90")
        return

    # Keep this modest first. Larger wheel = fewer candidates but more setup.
    wheel_primes = [2, 3, 5, 7, 11, 13]

    modulus, allowed_residues = build_wheel(offsets, wheel_primes)

    print("Pattern:", args.pattern)
    print("Offsets:", offsets)
    print("Wheel primes:", wheel_primes)
    print("Wheel modulus:", modulus)
    print("Allowed residues:", len(allowed_residues), "out of", modulus)
    print("Candidate reduction:", f"{100.0 * len(allowed_residues) / modulus:.4f}% survive")
    print()

    start = args.start
    end = args.start + args.limit

    base_k = start // modulus

    found = 0
    checked_candidates = 0

    print("Searching range:")
    print("[", start, ",", end, ")")
    print()

    for k in range(base_k, (end // modulus) + 2):
        base = k * modulus

        for r in allowed_residues:
            p = base + r

            if p < start or p >= end:
                continue

            # p must be odd and above small-prime issues.
            if p < 3:
                continue

            checked_candidates += 1

            values = [p + off for off in offsets]

            if all(is_prime(x) for x in values):
                found += 1
                print("FOUND", found, "base p =", p)
                print("  values:", ", ".join(str(x) for x in values))

                if found >= args.max_results:
                    print()
                    print("Stopped after max results.")
                    print("Checked wheel candidates:", checked_candidates)
                    return

    print()
    print("Done.")
    print("Found:", found)
    print("Checked wheel candidates:", checked_candidates)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


# Deterministic Miller-Rabin for 64-bit integers.
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

    # Deterministic bases for all 64-bit integers.
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in bases:
        if a >= n:
            continue

        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        probably_prime = False

        for _ in range(s - 1):
            x = (x * x) % n

            if x == n - 1:
                probably_prime = True
                break

        if not probably_prime:
            return False

    return True


def parse_pattern(pattern: str) -> List[int]:
    # Accepts:
    #   quad_0_6_84_90
    #   0_6_84_90
    offsets = []

    for part in pattern.split("_"):
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


def append_summary_csv(path: Path, row: dict) -> None:
    exists = path.exists()

    fields = [
        "label",
        "pattern",
        "offsets",
        "range_start",
        "range_end",
        "range_size",
        "wheel_primes",
        "wheel_modulus",
        "allowed_residues",
        "residue_survival_percent",
        "candidates_checked",
        "total_hits",
        "hit_rate_per_candidate",
        "hits_per_million_numbers",
        "first_hit",
        "last_hit",
    ]

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--pattern", required=True,
                        help="Pattern like quad_0_6_84_90 or 0_6_84_90")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--max-print", type=int, default=20,
                        help="Print only this many hits but keep counting the full range")
    parser.add_argument("--summary-csv", default="pattern_count_summary.csv")
    parser.add_argument("--hits-csv", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--wheel-primes", default="2,3,5,7,11,13")

    args = parser.parse_args()

    offsets = parse_pattern(args.pattern)

    if not offsets or offsets[0] != 0:
        print("ERROR: pattern must include offset 0, e.g. quad_0_6_84_90")
        return

    wheel_primes = [int(x.strip()) for x in args.wheel_primes.split(",") if x.strip()]

    modulus, allowed_residues = build_wheel(offsets, wheel_primes)

    start = args.start
    end = args.start + args.limit

    label = args.label if args.label else args.pattern

    print("Pattern:", args.pattern)
    print("Label:", label)
    print("Offsets:", offsets)
    print("Wheel primes:", wheel_primes)
    print("Wheel modulus:", modulus)
    print("Allowed residues:", len(allowed_residues), "out of", modulus)
    print("Candidate reduction:", f"{100.0 * len(allowed_residues) / modulus:.4f}% survive")
    print()
    print("Searching full range:")
    print("[", start, ",", end, ")")
    print()

    hits_writer = None
    hits_file = None

    if args.hits_csv:
        hits_file = Path(args.hits_csv).open("w", newline="")
        fields = ["pattern", "base_p"] + [f"value_{i}" for i in range(len(offsets))]
        hits_writer = csv.DictWriter(hits_file, fieldnames=fields)
        hits_writer.writeheader()

    checked_candidates = 0
    total_hits = 0
    first_hit = ""
    last_hit = ""

    base_k = start // modulus

    for k in range(base_k, (end // modulus) + 2):
        base = k * modulus

        for r in allowed_residues:
            p = base + r

            if p < start or p >= end:
                continue

            if p < 3:
                continue

            checked_candidates += 1

            values = [p + off for off in offsets]

            if all(is_prime(v) for v in values):
                total_hits += 1

                if first_hit == "":
                    first_hit = str(p)

                last_hit = str(p)

                if total_hits <= args.max_print:
                    print("FOUND", total_hits, "base p =", p)
                    print("  values:", ", ".join(str(v) for v in values))

                if hits_writer is not None:
                    row = {
                        "pattern": args.pattern,
                        "base_p": p,
                    }

                    for i, v in enumerate(values):
                        row[f"value_{i}"] = v

                    hits_writer.writerow(row)

    if hits_file is not None:
        hits_file.close()

    hit_rate = total_hits / checked_candidates if checked_candidates else 0.0
    hits_per_million = total_hits / (args.limit / 1_000_000.0)

    print()
    print("FULL RANGE SUMMARY")
    print("pattern:", args.pattern)
    print("range_start:", start)
    print("range_end:", end)
    print("range_size:", args.limit)
    print("candidates_checked:", checked_candidates)
    print("total_hits:", total_hits)
    print("hit_rate_per_candidate:", hit_rate)
    print("hits_per_million_numbers:", hits_per_million)
    print("first_hit:", first_hit)
    print("last_hit:", last_hit)

    row = {
        "label": label,
        "pattern": args.pattern,
        "offsets": "_".join(str(x) for x in offsets),
        "range_start": start,
        "range_end": end,
        "range_size": args.limit,
        "wheel_primes": "_".join(str(x) for x in wheel_primes),
        "wheel_modulus": modulus,
        "allowed_residues": len(allowed_residues),
        "residue_survival_percent": 100.0 * len(allowed_residues) / modulus,
        "candidates_checked": checked_candidates,
        "total_hits": total_hits,
        "hit_rate_per_candidate": hit_rate,
        "hits_per_million_numbers": hits_per_million,
        "first_hit": first_hit,
        "last_hit": last_hit,
    }

    append_summary_csv(Path(args.summary_csv), row)

    print()
    print("Saved summary to:", args.summary_csv)

    if args.hits_csv:
        print("Saved hits to:", args.hits_csv)


if __name__ == "__main__":
    main()

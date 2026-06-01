#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict


def parse_pattern(pattern):
    nums = []
    for part in pattern.split("_"):
        try:
            nums.append(int(part))
        except ValueError:
            pass
    return nums


def primes_upto(n):
    sieve = bytearray(b"\x01") * (n + 1)

    if n >= 0:
        sieve[0] = 0
    if n >= 1:
        sieve[1] = 0

    r = int(math.isqrt(n))

    for p in range(2, r + 1):
        if sieve[p]:
            start = p * p
            sieve[start:n + 1:p] = b"\x00" * (((n - start) // p) + 1)

    return [i for i in range(2, n + 1) if sieve[i]]


def singular_series(offsets, primes):
    # Hardy-Littlewood k-tuple singular series approximation:
    # product over primes q of:
    # (1 - nu(q)/q) / (1 - 1/q)^k
    #
    # nu(q) = number of distinct forbidden residues mod q.
    k = len(offsets)
    log_s = 0.0

    for q in primes:
        residues = set()

        for h in offsets:
            residues.add((-h) % q)

        nu = len(residues)

        if nu >= q:
            return 0.0

        factor = (1.0 - nu / q) / ((1.0 - 1.0 / q) ** k)

        if factor <= 0:
            return 0.0

        log_s += math.log(factor)

    return math.exp(log_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="top4_10M_summary.csv")
    parser.add_argument("--prime-limit", type=int, default=200000)
    args = parser.parse_args()

    primes = primes_upto(args.prime_limit)

    rows = []
    series_cache = {}

    with open(args.summary, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pattern = row["pattern"]
            offsets = parse_pattern(pattern)

            if pattern not in series_cache:
                series_cache[pattern] = singular_series(offsets, primes)

            s = series_cache[pattern]

            start = int(row["range_start"])
            size = int(row["range_size"])
            midpoint = start + size // 2

            actual_hits = int(row["total_hits"])
            actual_hpm = float(row["hits_per_million_numbers"])

            # Expected count ≈ S(H) * L / log(x)^k
            k = len(offsets)
            expected_hits = s * size / (math.log(midpoint) ** k)
            expected_hpm = expected_hits / (size / 1_000_000.0)

            ratio = actual_hits / expected_hits if expected_hits > 0 else 0.0

            rows.append({
                "pattern": pattern,
                "start": start,
                "actual_hits": actual_hits,
                "actual_hpm": actual_hpm,
                "singular_series": s,
                "expected_hits": expected_hits,
                "expected_hpm": expected_hpm,
                "actual_expected": ratio,
            })

    print()
    print("PER-RANGE ACTUAL VS THEORY")
    print("=" * 130)
    print(
        f"{'pattern':25s} {'start':>15s} {'actual':>8s} "
        f"{'exp':>10s} {'A/E':>8s} {'actual/M':>10s} "
        f"{'exp/M':>10s} {'singular':>12s}"
    )
    print("-" * 130)

    for r in sorted(rows, key=lambda x: (x["pattern"], x["start"])):
        print(
            f"{r['pattern']:25s} "
            f"{r['start']:15d} "
            f"{r['actual_hits']:8d} "
            f"{r['expected_hits']:10.2f} "
            f"{r['actual_expected']:8.3f} "
            f"{r['actual_hpm']:10.2f} "
            f"{r['expected_hpm']:10.2f} "
            f"{r['singular_series']:12.4f}"
        )

    by_pattern = defaultdict(list)

    for r in rows:
        by_pattern[r["pattern"]].append(r)

    print()
    print("AVERAGE ACTUAL / THEORY RANKING")
    print("=" * 95)
    print(
        f"{'rank':>4s} {'pattern':25s} {'avg_A/E':>10s} "
        f"{'avg_actual/M':>14s} {'avg_exp/M':>12s} {'singular':>12s} {'ranges':>8s}"
    )
    print("-" * 95)

    summary = []

    for pattern, vals in by_pattern.items():
        avg_ratio = sum(v["actual_expected"] for v in vals) / len(vals)
        avg_actual_hpm = sum(v["actual_hpm"] for v in vals) / len(vals)
        avg_expected_hpm = sum(v["expected_hpm"] for v in vals) / len(vals)
        singular = vals[0]["singular_series"]

        summary.append((avg_ratio, avg_actual_hpm, avg_expected_hpm, singular, pattern, len(vals)))

    for rank, (avg_ratio, avg_actual_hpm, avg_expected_hpm, singular, pattern, count) in enumerate(sorted(summary, reverse=True), 1):
        print(
            f"{rank:4d} {pattern:25s} "
            f"{avg_ratio:10.3f} "
            f"{avg_actual_hpm:14.2f} "
            f"{avg_expected_hpm:12.2f} "
            f"{singular:12.4f} "
            f"{count:8d}"
        )

    print()
    print("Interpretation:")
    print("  A/E near 1.0 means the pattern behaves close to Hardy-Littlewood expectation.")
    print("  A/E consistently above 1.2 or 1.5 would be more surprising.")
    print("  A high singular-series value means theory already predicts that pattern should be high-yield.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import math
import subprocess
from pathlib import Path


DEFAULT_PATTERNS = [
    "quad_0_6_126_336",
    "quad_0_6_96_210",
    "quad_0_6_66_360",
    "quad_0_6_120_336",
    "quad_0_6_66_150",
    "quad_0_6_156_300",
    "quad_0_6_90_330",
    "quad_0_6_126_270",
    "quad_0_6_132_336",
    "quad_0_6_66_90",
    "quad_0_6_132_210",
    "quad_0_6_90_120",
    "quad_0_6_126_360",
    "quad_0_6_126_246",
    "quad_0_6_132_252",
    "quad_0_6_84_90",
]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
    k = len(offsets)
    log_s = 0.0

    for q in primes:
        residues = {(-h) % q for h in offsets}
        nu = len(residues)

        if nu >= q:
            return 0.0

        factor = (1.0 - nu / q) / ((1.0 - 1.0 / q) ** k)

        if factor <= 0:
            return 0.0

        log_s += math.log(factor)

    return math.exp(log_s)


def generate_family(max_offset):
    patterns = []

    for a in range(12, max_offset + 1, 6):
        for b in range(a + 6, max_offset + 1, 6):
            patterns.append(f"quad_0_6_{a}_{b}")

    return patterns


def read_patterns(args):
    patterns = []

    if args.patterns:
        for p in args.patterns.split(","):
            p = p.strip()
            if p:
                patterns.append(p)

    if args.patterns_file:
        path = Path(args.patterns_file)

        with path.open() as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                for part in line.replace(",", " ").split():
                    part = part.strip()

                    if part:
                        patterns.append(part)

    if args.full_family:
        patterns.extend(generate_family(args.max_offset))

    if not patterns:
        patterns = list(DEFAULT_PATTERNS)

    return sorted(set(patterns))


def append_csv(path, row, fields):
    exists = Path(path).exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def run_generator(pattern, start, limit, tmp_summary):
    tmp = Path(tmp_summary)

    if tmp.exists():
        tmp.unlink()

    cmd = [
        "python3",
        "pattern_prime_generator_v2.py",
        "--pattern",
        pattern,
        "--start",
        str(start),
        "--limit",
        str(limit),
        "--label",
        f"ae_{start}_{pattern}",
        "--summary-csv",
        str(tmp),
        "--max-print",
        "0",
    ]

    subprocess.run(cmd, check=True)

    with tmp.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No output row produced for {pattern} at {start}")

    return rows[-1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--patterns", default="")
    parser.add_argument("--patterns-file", default="")
    parser.add_argument("--full-family", action="store_true")
    parser.add_argument("--max-offset", type=int, default=360)

    parser.add_argument("--starts", default="1000000000000,2000000000000,5000000000000,10000000000000")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=0)

    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--prime-limit", type=int, default=200000)
    parser.add_argument("--threshold", type=float, default=1.2)

    parser.add_argument("--results-csv", default="ae_hunter_results.csv")
    parser.add_argument("--hits-csv", default="ae_hunter_hits.csv")
    parser.add_argument("--tmp-summary", default=".ae_tmp_summary.csv")

    args = parser.parse_args()

    patterns = read_patterns(args)
    base_starts = parse_int_list(args.starts)
    theory_primes = primes_upto(args.prime_limit)

    fields = [
        "iteration",
        "pattern",
        "offsets",
        "range_start",
        "range_end",
        "range_size",
        "actual_hits",
        "hits_per_million",
        "candidates_checked",
        "hit_rate_per_candidate",
        "singular_series",
        "expected_hits",
        "expected_per_million",
        "actual_expected_ratio",
        "status",
    ]

    print("A/E Hunter")
    print("patterns:", len(patterns))
    print("starts:", base_starts)
    print("iterations:", args.iterations)
    print("limit:", args.limit)
    print("threshold:", args.threshold)
    print()

    for iteration in range(args.iterations):
        for base_start in base_starts:
            start = base_start + iteration * args.start_step

            for pattern in patterns:
                offsets = parse_pattern(pattern)

                if len(offsets) != 4:
                    print("Skipping non-quad pattern:", pattern)
                    continue

                print("=" * 80)
                print(f"Testing iteration={iteration} start={start} pattern={pattern}")
                print("=" * 80)

                generator_row = run_generator(
                    pattern=pattern,
                    start=start,
                    limit=args.limit,
                    tmp_summary=args.tmp_summary,
                )

                actual_hits = int(generator_row["total_hits"])
                hits_per_million = float(generator_row["hits_per_million_numbers"])
                candidates_checked = int(generator_row["candidates_checked"])
                hit_rate = float(generator_row["hit_rate_per_candidate"])

                midpoint = start + args.limit // 2
                singular = singular_series(offsets, theory_primes)

                expected_hits = singular * args.limit / (math.log(midpoint) ** len(offsets))
                expected_per_million = expected_hits / (args.limit / 1_000_000.0)
                ae_ratio = actual_hits / expected_hits if expected_hits > 0 else 0.0

                status = "HIT" if ae_ratio >= args.threshold else "normal"

                result = {
                    "iteration": iteration,
                    "pattern": pattern,
                    "offsets": "_".join(str(x) for x in offsets),
                    "range_start": start,
                    "range_end": start + args.limit,
                    "range_size": args.limit,
                    "actual_hits": actual_hits,
                    "hits_per_million": hits_per_million,
                    "candidates_checked": candidates_checked,
                    "hit_rate_per_candidate": hit_rate,
                    "singular_series": singular,
                    "expected_hits": expected_hits,
                    "expected_per_million": expected_per_million,
                    "actual_expected_ratio": ae_ratio,
                    "status": status,
                }

                append_csv(args.results_csv, result, fields)

                print(
                    f"{pattern} start={start} "
                    f"actual={actual_hits} expected={expected_hits:.2f} "
                    f"A/E={ae_ratio:.3f} status={status}"
                )

                if status == "HIT":
                    append_csv(args.hits_csv, result, fields)

                    print()
                    print("!!! A/E HIT SAVED !!!")
                    print(f"pattern={pattern}")
                    print(f"start={start}")
                    print(f"A/E={ae_ratio:.3f}")
                    print(f"saved_to={args.hits_csv}")
                    print()

    print()
    print("Done.")
    print("All results:", args.results_csv)
    print("A/E hits:", args.hits_csv)


if __name__ == "__main__":
    main()

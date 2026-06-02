#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_pattern(pattern):
    nums = []

    for part in pattern.split("_"):
        try:
            nums.append(int(part))
        except ValueError:
            pass

    return nums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="ae_hunter_results.csv")
    parser.add_argument("--prefix", default="ae")
    parser.add_argument("--top", type=int, default=12)

    args = parser.parse_args()

    rows = []

    with open(args.results, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pattern = row["pattern"]
            offsets = parse_pattern(pattern)

            if len(offsets) != 4:
                continue

            rows.append({
                "pattern": pattern,
                "a_offset": offsets[2],
                "b_offset": offsets[3],
                "start": int(row["range_start"]),
                "actual_hits": int(row["actual_hits"]),
                "hits_per_million": float(row["hits_per_million"]),
                "singular_series": float(row["singular_series"]),
                "expected_hits": float(row["expected_hits"]),
                "expected_per_million": float(row["expected_per_million"]),
                "ae": float(row["actual_expected_ratio"]),
            })

    if not rows:
        print("No rows found.")
        return

    by_pattern = defaultdict(list)

    for row in rows:
        by_pattern[row["pattern"]].append(row)

    summary = []

    for pattern, vals in by_pattern.items():
        avg_ae = sum(v["ae"] for v in vals) / len(vals)
        avg_hpm = sum(v["hits_per_million"] for v in vals) / len(vals)
        avg_exp = sum(v["expected_per_million"] for v in vals) / len(vals)
        singular = vals[0]["singular_series"]
        offsets = parse_pattern(pattern)

        summary.append({
            "pattern": pattern,
            "a_offset": offsets[2],
            "b_offset": offsets[3],
            "avg_ae": avg_ae,
            "avg_hpm": avg_hpm,
            "avg_expected_per_million": avg_exp,
            "singular_series": singular,
            "ranges": len(vals),
        })

    summary.sort(key=lambda r: r["avg_ae"], reverse=True)

    top_csv = f"{args.prefix}_top_patterns.csv"

    with open(top_csv, "w", newline="") as f:
        fields = [
            "pattern",
            "a_offset",
            "b_offset",
            "avg_ae",
            "avg_hpm",
            "avg_expected_per_million",
            "singular_series",
            "ranges",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in summary:
            writer.writerow(row)

    print("Saved:", top_csv)

    # Scatter: singular-series vs A/E
    plt.figure(figsize=(10, 7))

    x = [r["singular_series"] for r in summary]
    y = [r["avg_ae"] for r in summary]

    plt.scatter(x, y)

    for r in summary[:args.top]:
        plt.annotate(
            r["pattern"].replace("quad_0_6_", ""),
            (r["singular_series"], r["avg_ae"]),
            fontsize=8,
        )

    plt.axhline(1.0, linestyle="--")
    plt.axhline(1.2, linestyle="--")
    plt.xlabel("Singular-series score")
    plt.ylabel("Average actual / expected")
    plt.title("Pattern anomaly check: A/E vs singular-series score")
    plt.tight_layout()

    scatter_path = f"{args.prefix}_scatter_ae_vs_singular.png"
    plt.savefig(scatter_path, dpi=180)
    plt.close()

    print("Saved:", scatter_path)

    # Timeline for top patterns
    top_patterns = [r["pattern"] for r in summary[:min(args.top, len(summary))]]

    plt.figure(figsize=(12, 7))

    for pattern in top_patterns:
        vals = sorted(by_pattern[pattern], key=lambda r: r["start"])
        starts = [v["start"] for v in vals]
        aes = [v["ae"] for v in vals]

        plt.plot(starts, aes, marker="o", label=pattern.replace("quad_0_6_", ""))

    plt.axhline(1.0, linestyle="--")
    plt.axhline(1.2, linestyle="--")
    plt.xlabel("Range start")
    plt.ylabel("Actual / expected")
    plt.title("A/E over tested ranges for top patterns")
    plt.legend(fontsize=8)
    plt.tight_layout()

    timeline_path = f"{args.prefix}_timeline_top_patterns.png"
    plt.savefig(timeline_path, dpi=180)
    plt.close()

    print("Saved:", timeline_path)

    # Heatmap-style scatter in pattern space: x=third offset, y=fourth offset
    plt.figure(figsize=(10, 8))

    xs = [r["a_offset"] for r in summary]
    ys = [r["b_offset"] for r in summary]
    cs = [r["avg_ae"] for r in summary]

    sc = plt.scatter(xs, ys, c=cs, s=80)
    plt.colorbar(sc, label="Average actual / expected")

    for r in summary[:args.top]:
        plt.annotate(
            r["pattern"].replace("quad_0_6_", ""),
            (r["a_offset"], r["b_offset"]),
            fontsize=8,
        )

    plt.xlabel("Third offset")
    plt.ylabel("Fourth offset")
    plt.title("Pattern-space anomaly map for quad_0_6_6a_6b")
    plt.tight_layout()

    heat_path = f"{args.prefix}_pattern_space_ae.png"
    plt.savefig(heat_path, dpi=180)
    plt.close()

    print("Saved:", heat_path)

    # Hits/M pattern space
    plt.figure(figsize=(10, 8))

    cs = [r["avg_hpm"] for r in summary]

    sc = plt.scatter(xs, ys, c=cs, s=80)
    plt.colorbar(sc, label="Average hits per million")

    for r in sorted(summary, key=lambda x: x["avg_hpm"], reverse=True)[:args.top]:
        plt.annotate(
            r["pattern"].replace("quad_0_6_", ""),
            (r["a_offset"], r["b_offset"]),
            fontsize=8,
        )

    plt.xlabel("Third offset")
    plt.ylabel("Fourth offset")
    plt.title("Pattern-space yield map for quad_0_6_6a_6b")
    plt.tight_layout()

    yield_path = f"{args.prefix}_pattern_space_hits_per_million.png"
    plt.savefig(yield_path, dpi=180)
    plt.close()

    print("Saved:", yield_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Set


@dataclass
class Candidate:
    kind: str              # "offset" or "gap"
    source: str
    chunk: int
    name: str              # pattern name or gap motif name
    rank: int
    max_z: float
    top_count: int
    mean: float
    stddev: float
    range_start: int
    range_end: int


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def run_prime2(exe: Path, outdir: Path, start: int, range_size: int,
               width: int, height: int, label: str) -> bool:
    outdir.mkdir(parents=True, exist_ok=True)

    stdout_path = outdir / "run_stdout.txt"
    stderr_path = outdir / "run_time.txt"

    cmd = [
        "/usr/bin/time",
        "-v",
        str(exe),
        str(start),
        str(range_size),
        "1",
        str(width),
        str(height),
    ]

    print(f"[RUN] {label}: {' '.join(cmd)}")

    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        result = subprocess.run(cmd, cwd=outdir, stdout=out, stderr=err)

    ok = result.returncode == 0 and (outdir / "iter_000_top_patterns.csv").exists()

    if not ok:
        print(f"[ERROR] {label} failed.")
        print(f"stdout: {stdout_path}")
        print(f"stderr: {stderr_path}")

    return ok


def read_offset_candidates(path: Path, chunk: int, source: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    if not path.exists():
        return candidates

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row.get("pattern", "")

            candidates.append(Candidate(
                kind="offset",
                source=source,
                chunk=chunk,
                name=name,
                rank=safe_int(row.get("rank", 0)),
                max_z=safe_float(row.get("max_z", 0)),
                top_count=safe_int(row.get("top_count", 0)),
                mean=safe_float(row.get("mean", 0)),
                stddev=safe_float(row.get("stddev", 0)),
                range_start=safe_int(row.get("top_range_start", 0)),
                range_end=safe_int(row.get("top_range_end", 0)),
            ))

    return candidates


def read_gap_candidates(path: Path, chunk: int, source: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    if not path.exists():
        return candidates

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row.get("motif", "")

            candidates.append(Candidate(
                kind="gap",
                source=source,
                chunk=chunk,
                name=name,
                rank=safe_int(row.get("rank", 0)),
                max_z=safe_float(row.get("z", 0)),
                top_count=safe_int(row.get("top_count", 0)),
                mean=safe_float(row.get("mean", 0)),
                stddev=safe_float(row.get("stddev", 0)),
                range_start=safe_int(row.get("range_start", 0)),
                range_end=safe_int(row.get("range_end", 0)),
            ))

    return candidates


def collect_candidates(outdir: Path, chunk: int, source: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    candidates.extend(read_offset_candidates(
        outdir / "iter_000_top_patterns.csv",
        chunk,
        source,
    ))

    candidates.extend(read_gap_candidates(
        outdir / "iter_000_gap_motifs_len3.csv",
        chunk,
        source,
    ))

    return candidates


def parse_offsets(name: str) -> Tuple[str, Tuple[int, ...]]:
    parts = name.split("_")
    family = parts[0] if parts else ""

    nums = []

    for p in parts[1:]:
        try:
            nums.append(int(p))
        except Exception:
            pass

    return family, tuple(nums)


def related_offset(a: str, b: str) -> bool:
    fa, oa = parse_offsets(a)
    fb, ob = parse_offsets(b)

    if fa != fb:
        return False

    if fa != "quad":
        return a == b

    return len(set(oa).intersection(set(ob))) >= 3


def related_gap(a: str, b: str) -> bool:
    fa, oa = parse_offsets(a)
    fb, ob = parse_offsets(b)

    if fa != "gap" or fb != "gap":
        return False

    return len(set(oa).intersection(set(ob))) >= 2


def related(a: Candidate, b: Candidate) -> bool:
    if a.kind != b.kind:
        return False

    if a.name == b.name:
        return True

    if a.kind == "offset":
        return related_offset(a.name, b.name)

    if a.kind == "gap":
        return related_gap(a.name, b.name)

    return False


def is_interesting(c: Candidate) -> bool:
    if c.kind == "offset":
        if not c.name.startswith("quad_"):
            return False

        if c.top_count >= 7:
            return True

        if c.top_count >= 6 and c.max_z >= 12.0:
            return True

        if c.top_count >= 5 and c.max_z >= 20.0:
            return True

        return False

    if c.kind == "gap":
        if c.top_count >= 5:
            return True

        if c.top_count >= 4 and c.max_z >= 15.0:
            return True

        if c.top_count >= 3 and c.max_z >= 25.0:
            return True

        return False

    return False


def score(c: Candidate):
    # Gap motifs get a small boost because they describe consecutive prime gaps,
    # which is mathematically stronger than raw offset constellations.
    kind_bonus = 2 if c.kind == "gap" else 0
    return (c.top_count + kind_bonus, c.max_z, -c.mean)


def write_candidates(path: Path, candidates: List[Candidate]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "kind",
            "source",
            "chunk",
            "name",
            "rank",
            "max_z",
            "top_count",
            "mean",
            "stddev",
            "range_start",
            "range_end",
        ])

        for c in candidates:
            writer.writerow([
                c.kind,
                c.source,
                c.chunk,
                c.name,
                c.rank,
                c.max_z,
                c.top_count,
                c.mean,
                c.stddev,
                c.range_start,
                c.range_end,
            ])


def append_strong_candidate(path: Path, c: Candidate, reason: str) -> None:
    exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "kind",
                "source",
                "chunk",
                "name",
                "rank",
                "max_z",
                "top_count",
                "mean",
                "stddev",
                "range_start",
                "range_end",
                "reason",
            ])

        writer.writerow([
            c.kind,
            c.source,
            c.chunk,
            c.name,
            c.rank,
            c.max_z,
            c.top_count,
            c.mean,
            c.stddev,
            c.range_start,
            c.range_end,
            reason,
        ])


def matching_candidates(candidates: List[Candidate],
                        target: Candidate) -> Tuple[List[Candidate], List[Candidate]]:
    exact = []
    rel = []

    for c in candidates:
        if c.kind != target.kind:
            continue

        if c.name == target.name:
            exact.append(c)
        elif related(c, target):
            rel.append(c)

    exact.sort(key=score, reverse=True)
    rel.sort(key=score, reverse=True)

    return exact, rel


def confirmation_check(target: Candidate,
                       broad_all: List[Candidate],
                       zoom_all: List[Candidate]) -> Tuple[bool, str]:
    exact_broad = [
        c for c in broad_all
        if c.kind == target.kind
        and c.name == target.name
        and c.top_count >= max(4, target.top_count - 1)
    ]

    exact_zoom = [
        c for c in zoom_all
        if c.kind == target.kind
        and c.name == target.name
        and c.top_count >= 3
    ]

    related_zoom = [
        c for c in zoom_all
        if c.kind == target.kind
        and c.name != target.name
        and related(c, target)
        and c.top_count >= 3
    ]

    status = (
        f"exact_broad={len(exact_broad)}, "
        f"exact_zoom={len(exact_zoom)}, "
        f"related_zoom={len(related_zoom)}"
    )

    # Gap motifs are stronger because they are consecutive-prime structures.
    if target.kind == "gap":
        if len(exact_broad) >= 2 and len(exact_zoom) >= 1 and target.top_count >= 4:
            return True, f"VERY STRONG GAP CANDIDATE: repeated broadly and survived zoom. {status}"

        if len(exact_zoom) >= 2 and target.top_count >= 4:
            return True, f"STRONG GAP CANDIDATE: exact gap motif survived multiple zooms. {status}"

        if target.top_count >= 6 and target.max_z >= 15 and len(exact_zoom) >= 1:
            return True, f"STRONG GAP CANDIDATE: high count gap motif survived zoom. {status}"

    # Offset constellations need stricter proof because we test many of them.
    if target.kind == "offset":
        if len(exact_broad) >= 3:
            return True, f"VERY STRONG OFFSET CANDIDATE: same exact pattern appeared in {len(exact_broad)} broad chunks. {status}"

        if len(exact_broad) >= 2 and len(exact_zoom) >= 2:
            return True, f"STRONG OFFSET CANDIDATE: repeated broadly and survived zoom multiple times. {status}"

        if target.top_count >= 10 and len(exact_zoom) >= 2:
            return True, f"STRONG OFFSET CANDIDATE: top_count >= 10 and exact zoom survival. {status}"

        if len(exact_zoom) >= 4:
            return True, f"STRONG OFFSET CANDIDATE: exact pattern survived many zoom checks. {status}"

        if len(exact_zoom) >= 1 and len(related_zoom) >= 100 and target.top_count >= 9:
            return True, f"STRONG OFFSET FAMILY: exact survivor plus large related family. {status}"

    return False, f"not confirmed yet: {status}"


def chase_candidate(exe: Path,
                    root: Path,
                    target: Candidate,
                    chunk: int,
                    broad_all: List[Candidate],
                    zoom_all: List[Candidate],
                    log,
                    max_depth: int,
                    zoom_width: int,
                    zoom_height: int,
                    initial_zoom_range: int) -> Tuple[bool, str]:
    current_target = target
    current_range = initial_zoom_range

    for depth in range(max_depth):
        midpoint = (current_target.range_start + current_target.range_end) // 2
        zoom_start = midpoint - current_range // 2

        if zoom_start < 3:
            zoom_start = 3

        safe_name = current_target.name.replace("/", "_")
        zoom_dir = root / f"zoom_chunk_{chunk:03d}_depth_{depth}_{current_target.kind}_{safe_name}"

        log("")
        log("------------------------------------------------------------")
        log(f"CHASE depth {depth}: {current_target.kind}:{current_target.name}")
        log(f"top_count={current_target.top_count}, z={current_target.max_z:.3f}")
        log(f"zoom range=[{zoom_start}, {zoom_start + current_range})")
        log("------------------------------------------------------------")

        ok = run_prime2(
            exe=exe,
            outdir=zoom_dir,
            start=zoom_start,
            range_size=current_range,
            width=zoom_width,
            height=zoom_height,
            label=f"zoom_depth_{depth}_{current_target.kind}_{current_target.name}",
        )

        if not ok:
            return False, "zoom run failed"

        candidates = collect_candidates(
            zoom_dir,
            chunk,
            f"zoom_depth_{depth}_{current_target.kind}_{current_target.name}",
        )

        zoom_all.extend(candidates)
        write_candidates(root / "all_zoom_candidates.csv", zoom_all)

        exact_original, related_original = matching_candidates(candidates, target)
        exact_current, related_current = matching_candidates(candidates, current_target)

        log(f"Exact matches to original target in zoom: {len(exact_original)}")
        for c in exact_original[:10]:
            log(
                f"  [EXACT ORIGINAL] {c.kind}:{c.name} "
                f"top_count={c.top_count} z={c.max_z:.3f} "
                f"range={c.range_start}-{c.range_end}"
            )

        log(f"Related matches to original target in zoom: {len(related_original)}")
        for c in related_original[:10]:
            log(
                f"  [RELATED ORIGINAL] {c.kind}:{c.name} "
                f"top_count={c.top_count} z={c.max_z:.3f} "
                f"range={c.range_start}-{c.range_end}"
            )

        confirmed, reason = confirmation_check(target, broad_all, zoom_all)
        log(f"Confirmation check: {reason}")

        if confirmed:
            return True, reason

        # Chase the strongest survivor deeper.
        next_options = []

        for c in exact_current:
            if c.top_count >= 2:
                next_options.append(c)

        for c in related_current:
            if c.top_count >= 3:
                next_options.append(c)

        next_options.sort(key=score, reverse=True)

        if not next_options:
            log("No strong enough survivor to chase deeper.")
            return False, "no strong survivor in zoom"

        current_target = next_options[0]
        current_range = max(current_range // 5, 500_000)

    log("Reached max chase depth without final confirmation.")
    return False, "max chase depth reached"


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--start", type=int, default=1_000_000_000_000)
    parser.add_argument("--broad-range", type=int, default=200_000_000)
    parser.add_argument("--zoom-range", type=int, default=10_000_000)

    parser.add_argument("--broad-width", type=int, default=256)
    parser.add_argument("--broad-height", type=int, default=256)
    parser.add_argument("--zoom-width", type=int, default=256)
    parser.add_argument("--zoom-height", type=int, default=256)

    parser.add_argument("--max-zooms-per-chunk", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--out", type=str, default="")

    # Default is now to KEEP GOING after strong candidates.
    # Use this only if you want old behaviour.
    parser.add_argument("--stop-on-strong", action="store_true")

    args = parser.parse_args()

    base_dir = Path.cwd()
    exe = base_dir / "prime2"

    if not exe.exists():
        print(f"ERROR: {exe} does not exist. Run: module load cuda/auto && make")
        return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.out) if args.out else base_dir / f"auto_explore_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    log_path = root / "auto_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log("============================================================")
    log("AUTO PRIME PATTERN EXPLORER V3")
    log("Tracks: offset constellations + consecutive gap motifs")
    log("Strong candidates are saved and the search continues.")
    log("============================================================")
    log(f"Output folder: {root}")
    log(f"Executable: {exe}")
    log(f"Chunks: {args.chunks}")
    log(f"Broad range: {args.broad_range}")
    log(f"Zoom range: {args.zoom_range}")
    log(f"Max chase depth: {args.max_depth}")
    log(f"Stop on strong: {args.stop_on_strong}")
    log("")

    broad_all: List[Candidate] = []
    zoom_all: List[Candidate] = []
    strong_keys: Set[Tuple[str, str, int, int]] = set()

    for chunk in range(args.chunks):
        broad_start = args.start + chunk * args.broad_range
        broad_dir = root / f"broad_chunk_{chunk:03d}"

        log("============================================================")
        log(f"BROAD CHUNK {chunk}")
        log(f"Range: [{broad_start}, {broad_start + args.broad_range})")
        log("============================================================")

        ok = run_prime2(
            exe=exe,
            outdir=broad_dir,
            start=broad_start,
            range_size=args.broad_range,
            width=args.broad_width,
            height=args.broad_height,
            label=f"broad_chunk_{chunk:03d}",
        )

        if not ok:
            return 1

        candidates = collect_candidates(
            broad_dir,
            chunk,
            f"broad_chunk_{chunk:03d}",
        )

        broad_all.extend(candidates)
        write_candidates(root / "all_broad_candidates.csv", broad_all)

        interesting = [c for c in candidates if is_interesting(c)]
        interesting.sort(key=score, reverse=True)

        offset_count = sum(1 for c in interesting if c.kind == "offset")
        gap_count = sum(1 for c in interesting if c.kind == "gap")

        log(f"Interesting candidates: {len(interesting)}")
        log(f"  offset candidates: {offset_count}")
        log(f"  gap motif candidates: {gap_count}")

        for c in interesting[:15]:
            log(
                f"  {c.kind}:{c.name} top_count={c.top_count} "
                f"z={c.max_z:.3f} range={c.range_start}-{c.range_end}"
            )

        zooms_done = 0

        for target in interesting:
            if zooms_done >= args.max_zooms_per_chunk:
                break

            found, reason = chase_candidate(
                exe=exe,
                root=root,
                target=target,
                chunk=chunk,
                broad_all=broad_all,
                zoom_all=zoom_all,
                log=log,
                max_depth=args.max_depth,
                zoom_width=args.zoom_width,
                zoom_height=args.zoom_height,
                initial_zoom_range=args.zoom_range,
            )

            zooms_done += 1

            if found:
                key = (target.kind, target.name, target.range_start, target.range_end)

                if key not in strong_keys:
                    strong_keys.add(key)

                    append_strong_candidate(
                        root / "strong_candidates.csv",
                        target,
                        reason
                    )

                    log("")
                    log("============================================================")
                    log("STRONG CANDIDATE SAVED")
                    log("============================================================")
                    log(f"kind={target.kind}")
                    log(f"name={target.name}")
                    log(f"top_count={target.top_count}")
                    log(f"z={target.max_z:.3f}")
                    log(f"range={target.range_start}-{target.range_end}")
                    log(f"reason={reason}")
                    log(f"Saved to: {root / 'strong_candidates.csv'}")
                    log("Continuing search.")
                    log("============================================================")

                all_candidates = broad_all + zoom_all
                all_candidates.sort(key=score, reverse=True)
                write_candidates(root / "all_candidates_sorted.csv", all_candidates)

                if args.stop_on_strong:
                    log("Stopping because --stop-on-strong was used.")
                    return 0

        log("")

    log("============================================================")
    log("AUTO RUN COMPLETE")
    log("============================================================")

    all_candidates = broad_all + zoom_all
    all_candidates.sort(key=score, reverse=True)

    write_candidates(root / "all_zoom_candidates.csv", zoom_all)
    write_candidates(root / "all_candidates_sorted.csv", all_candidates)

    log("Best candidates overall:")
    for c in all_candidates[:30]:
        log(
            f"{c.kind}:{c.name} source={c.source} "
            f"top_count={c.top_count} z={c.max_z:.3f} "
            f"range={c.range_start}-{c.range_end}"
        )

    strong_file = root / "strong_candidates.csv"

    if strong_file.exists():
        log(f"Strong candidates saved to: {strong_file}")
    else:
        log("No strong candidates reached the threshold this run.")

    log(f"Output folder: {root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
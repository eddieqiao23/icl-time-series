#!/usr/bin/env python3
"""Create paper-facing aggregate tables from per-pool behavioral results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    stderr = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "mean_mse": mean, "stderr": stderr,
        "ci95_low": max(mean - 1.96 * stderr, 0.0),
        "ci95_high": mean + 1.96 * stderr, "num_pools": len(array),
    }


def write_table(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=HERE / "artifacts" / "summaries" / "mse_by_position.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "artifacts" / "summaries")
    parser.add_argument("--headline-start", type=int, default=40,
                        help="First task position included in the late-context headline table.")
    args = parser.parse_args()
    with args.input.open() as handle:
        source = list(csv.DictReader(handle))

    per_position = defaultdict(list)
    late_by_pool = defaultdict(list)
    for row in source:
        if not row["mse"]:
            continue
        condition = (int(row["T"]), int(row["K"]), int(row["N"]), int(row["d"]),
                     float(row["noise_std"]), row["method"])
        position = int(row["position"])
        pool = int(row["pool_index"])
        value = float(row["mse"])
        per_position[condition + (position,)].append(value)
        if position >= args.headline_start:
            late_by_pool[condition + (pool,)].append(value)

    position_rows = []
    for key, values in sorted(per_position.items()):
        T, K, N, d, noise, method, position = key
        position_rows.append({
            "T": T, "K": K, "N": N, "d": d, "noise_std": noise,
            "method": method, "position": position, **summarize(values),
        })

    headline_groups = defaultdict(list)
    for key, values in late_by_pool.items():
        headline_groups[key[:-1]].append(float(np.mean(values)))
    headline_rows = []
    for key, values in sorted(headline_groups.items()):
        T, K, N, d, noise, method = key
        headline_rows.append({
            "T": T, "K": K, "N": N, "d": d, "noise_std": noise,
            "method": method, "position_start": args.headline_start,
            "position_end": N - 1, **summarize(values),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(position_rows, args.output_dir / "mse_summary.csv")
    write_table(headline_rows, args.output_dir / "headline_summary.csv")
    print(f"Wrote {len(position_rows)} position summaries and "
          f"{len(headline_rows)} headline summaries")


if __name__ == "__main__":
    main()

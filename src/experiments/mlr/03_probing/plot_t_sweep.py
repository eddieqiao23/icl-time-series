#!/usr/bin/env python3
"""Plot coefficient decodability for each trained support-count checkpoint."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=HERE / "artifacts" / "t_sweep" / "probe_scores_by_t.csv")
    parser.add_argument("--output", type=Path,
                        default=HERE / "figures" / "beta_r2_by_t.png")
    args = parser.parse_args()

    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("T", "pool_index", "layer", "position"):
            row[field] = int(row[field])
        row["score"] = float(row["score"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for T in sorted({row["T"] for row in rows}):
        layers = sorted({row["layer"] for row in rows if row["T"] == T})
        means, errors = [], []
        for layer in layers:
            selected = [row for row in rows if row["T"] == T
                        and row["layer"] == layer and row["position"] >= 30]
            by_pool = defaultdict(list)
            for row in selected:
                by_pool[row["pool_index"]].append(row["score"])
            pool_means = np.asarray([np.mean(values) for values in by_pool.values()])
            means.append(float(pool_means.mean()))
            errors.append(float(pool_means.std(ddof=1) / np.sqrt(len(pool_means)))
                          if len(pool_means) > 1 else 0.0)
        ax.errorbar(layers, means, yerr=errors, marker="o", capsize=3,
                    label=f"T = {T}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Representation stage", ylabel="Coefficient probe R²")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Support pairs")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

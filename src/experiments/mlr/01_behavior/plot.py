#!/usr/bin/env python3
"""Plot the tracked long-form behavioral summaries."""

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
                        default=HERE / "artifacts" / "summaries" / "mse_by_position.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))

    grouped = defaultdict(list)
    for row in rows:
        if row["mse"]:
            key = (int(row["K"]), float(row["noise_std"]), int(row["T"]), row["method"])
            grouped[key].append((int(row["position"]), int(row["pool_index"]), float(row["mse"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for K, noise in sorted({(key[0], key[1]) for key in grouped}):
        fig, ax = plt.subplots(figsize=(8, 5))
        for T in sorted({key[2] for key in grouped if key[:2] == (K, noise)}):
            values = grouped.get((K, noise, T, "transformer"), [])
            if not values:
                continue
            positions = sorted({value[0] for value in values})
            matrix = np.array([[mse for n, _, mse in values if n == position]
                               for position in positions])
            mean = matrix.mean(axis=1)
            stderr = (matrix.std(axis=1, ddof=1) / np.sqrt(matrix.shape[1])
                      if matrix.shape[1] > 1 else np.zeros_like(mean))
            line, = ax.plot(positions, mean, label=f"T={T}")
            ax.fill_between(positions, np.maximum(mean - stderr, 1e-8), mean + stderr,
                            color=line.get_color(), alpha=0.18)
        ax.set(xlabel="Task position", ylabel="MSE",
               title=f"Transformer behavior (K={K}, noise={noise:g})")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        tag = "noiseless" if noise == 0 else "noisy"
        output = args.output_dir / f"transformer_mse_K{K}_{tag}.png"
        fig.savefig(output, dpi=200)
        plt.close(fig)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

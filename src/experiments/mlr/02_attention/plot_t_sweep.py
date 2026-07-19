#!/usr/bin/env python3
"""Plot final-query attention as layer-by-T task-index bar charts."""

import argparse
import csv
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
                        default=HERE / "artifacts" / "t_sweep" / "final_query_attention_by_t.csv")
    parser.add_argument("--output", type=Path,
                        default=HERE / "figures" / "final_query_attention_by_t.png")
    args = parser.parse_args()

    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("T", "layer", "key_task", "pool_index"):
            row[field] = int(row[field])
        row["attention"] = float(row["attention"])

    layers = sorted({row["layer"] for row in rows})
    support_counts = sorted({row["T"] for row in rows})
    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    subfigures = fig.subfigures(3, 2)
    for layer, subfigure in zip(layers, subfigures.flat):
        subfigure.suptitle(f"Layer {layer + 1}", fontweight="bold")
        axes = subfigure.subplots(2, 2, sharex=True, sharey=True)
        layer_means = {}
        for T in support_counts:
            layer_means[T] = [
                np.mean([row["attention"] for row in rows
                         if row["T"] == T and row["layer"] == layer
                         and row["key_task"] == task])
                for task in range(50)
            ]
        upper = max(max(values) for values in layer_means.values()) * 1.08
        for ax, T in zip(axes.flat, support_counts):
            ax.bar(range(50), layer_means[T], width=0.85)
            ax.set_title(f"T = {T}")
            ax.set_ylim(0, upper)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.grid(True, axis="y", alpha=0.25)
        for ax in axes[-1]:
            ax.set_xlabel("Run index")
        for ax in axes[:, 0]:
            ax.set_ylabel("Attention weight")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

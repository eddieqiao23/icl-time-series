#!/usr/bin/env python3
"""Aggregate pool-level probe scores and create layerwise figures."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=HERE / "artifacts" / "summaries" / "probe_scores.csv")
    parser.add_argument("--summary", type=Path,
                        default=HERE / "artifacts" / "summaries" / "probe_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    with args.input.open() as handle: rows = list(csv.DictReader(handle))
    for row in rows:
        row["layer"] = int(row["layer"]); row["position"] = int(row["position"])
        row["pool_index"] = int(row["pool_index"]); row["score"] = float(row["score"])
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_type"], row["layer"], row["position"],
                 row["metric"], row["control"])].append(row["score"])
    summary = []
    for key, values in sorted(grouped.items()):
        mean = float(np.mean(values)); se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        critical = float(student_t.ppf(0.975, len(values) - 1))
        summary.append({
            "model_type": key[0], "layer": key[1], "position": key[2],
            "metric": key[3], "control": key[4], "mean": mean,
            "stderr": se, "ci95_low": mean - critical * se,
            "ci95_high": mean + critical * se, "num_pools": len(values),
        })
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, chance in (("beta_r2", "Coefficient probe R²", 0.0),
                                    ("component_accuracy", "Component probe accuracy", 0.5)):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for model_type, control, label in (
            ("trained", "actual", "Trained"), ("untrained", "actual", "Untrained"),
            ("trained", "shuffled", "Shuffled labels"),
        ):
            layers = sorted({row["layer"] for row in summary if row["model_type"] == model_type
                             and row["metric"] == metric and row["control"] == control})
            means, errors = [], []
            for layer in layers:
                selected = [row for row in rows if row["model_type"] == model_type
                            and row["metric"] == metric and row["control"] == control
                            and row["layer"] == layer and row["position"] >= 30]
                by_pool = defaultdict(list)
                for row in selected: by_pool[row["pool_index"]].append(row["score"])
                pool_means = np.asarray([np.mean(values) for values in by_pool.values()])
                means.append(float(pool_means.mean()))
                errors.append(float(pool_means.std(ddof=1) / np.sqrt(len(pool_means))))
            ax.errorbar(layers, means, yerr=errors, marker="o", capsize=3, label=label)
        raw = [row["mean"] for row in summary if row["model_type"] == "raw_input"
               and row["metric"] == metric and row["control"] == "actual" and row["position"] >= 30]
        ax.axhline(np.mean(raw), linestyle="--", label="Raw packed input")
        ax.axhline(chance, color="black", linewidth=0.8)
        ax.set(xlabel="Representation stage", ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(True, alpha=0.3)
        ax.legend(); fig.tight_layout(); fig.savefig(args.output_dir / f"{metric}_by_layer.png", dpi=200)
        plt.close(fig)

    trained = [row for row in summary if row["model_type"] == "trained"
               and row["metric"] == "beta_r2" and row["control"] == "actual"]
    layers = sorted({row["layer"] for row in trained}); positions = sorted({row["position"] for row in trained})
    matrix = np.array([[next(row["mean"] for row in trained if row["layer"] == layer
                             and row["position"] == position) for position in positions]
                       for layer in layers])
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set(xlabel="Task position", ylabel="Representation stage", title="Coefficient decodability")
    ax.set_xticks(range(len(positions)), positions); ax.set_yticks(range(len(layers)), layers)
    fig.colorbar(image, ax=ax, label="R²"); fig.tight_layout()
    fig.savefig(args.output_dir / "beta_r2_layer_position.png", dpi=200); plt.close(fig)
    print(f"Wrote {len(summary)} summary rows and three figures")


if __name__ == "__main__": main()

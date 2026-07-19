#!/usr/bin/env python3
"""Summarize and plot attention-routing results."""

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


def write_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=HERE / "artifacts" / "summaries" / "attention_by_head.csv")
    parser.add_argument("--profile-input", type=Path,
                        default=HERE / "artifacts" / "summaries" / "final_query_attention_by_task.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--summary-dir", type=Path, default=HERE / "artifacts" / "summaries")
    args = parser.parse_args()
    with args.input.open() as handle:
        source = list(csv.DictReader(handle))
    for row in source:
        for field in ("pool_cosine", "same_attention", "different_attention", "selectivity",
                      "input_attention", "output_attention", "self_attention"):
            row[field] = float(row[field])
        for field in ("pool_index", "layer", "head", "source_position"):
            row[field] = int(row[field])
    with args.profile_input.open() as handle:
        profiles = list(csv.DictReader(handle))
    for row in profiles:
        for field in ("pool_cosine", "attention", "input_attention", "output_attention"):
            row[field] = float(row[field])
        for field in ("same_attention", "different_attention"):
            row[field] = float(row[field]) if row[field] else np.nan
        for field in ("pool_index", "layer", "head", "source_task", "key_task"):
            row[field] = int(row[field])
        row["is_self"] = row["is_self"] == "True"

    grouped = defaultdict(list)
    for row in source:
        grouped[(row["model_type"], row["pool_index"], row["layer"])].append(row)
    pool_layer_rows = []
    for (model_type, pool, layer), rows in sorted(grouped.items()):
        late = [row for row in rows if row["source_position"] >= 30]
        pool_layer_rows.append({
            "model_type": model_type, "pool_index": pool, "layer": layer,
            "pool_cosine": rows[0]["pool_cosine"],
            "selectivity": float(np.mean([row["selectivity"] for row in late])),
            "same_attention": float(np.mean([row["same_attention"] for row in late])),
            "different_attention": float(np.mean([row["different_attention"] for row in late])),
            "input_attention": float(np.mean([row["input_attention"] for row in late])),
            "output_attention": float(np.mean([row["output_attention"] for row in late])),
        })
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pool_layer_rows, args.summary_dir / "attention_summary.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_type in ("trained", "untrained"):
        means = []
        layers = sorted({row["layer"] for row in pool_layer_rows})
        for layer in layers:
            values = [row["selectivity"] for row in pool_layer_rows
                      if row["model_type"] == model_type and row["layer"] == layer]
            means.append(np.mean(values))
        ax.plot(layers, means, marker="o", label=model_type.capitalize())
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Layer", ylabel="Same-component attention selectivity")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "selectivity_by_layer.png", dpi=200); plt.close(fig)

    trained_rows = [row for row in source if row["model_type"] == "trained"
                    and row["source_position"] >= 30]
    layers = sorted({row["layer"] for row in trained_rows})
    heads = sorted({row["head"] for row in trained_rows})
    heatmap = np.zeros((len(layers), len(heads)))
    for layer in layers:
        for head in heads:
            heatmap[layer, head] = np.mean([row["selectivity"] for row in trained_rows
                                            if row["layer"] == layer and row["head"] == head])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bound = max(abs(heatmap.min()), abs(heatmap.max()))
    image = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set(xlabel="Head", ylabel="Layer", title="Late-position component selectivity")
    ax.set_xticks(heads); ax.set_yticks(layers)
    fig.colorbar(image, ax=ax, label="Selectivity"); fig.tight_layout()
    fig.savefig(args.output_dir / "selectivity_by_head.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer in layers:
        points = [row for row in pool_layer_rows if row["model_type"] == "trained"
                  and row["layer"] == layer]
        ax.scatter([row["pool_cosine"] for row in points],
                   [row["selectivity"] for row in points], label=f"Layer {layer}", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Coefficient cosine similarity", ylabel="Selectivity")
    ax.grid(True, alpha=0.3); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(args.output_dir / "selectivity_vs_similarity.png", dpi=200); plt.close(fig)

    # Average heads inside each coefficient pool before computing uncertainty,
    # so the error band reflects the independent pool replication unit.
    profile_groups = defaultdict(list)
    for row in profiles:
        profile_groups[(row["model_type"], row["pool_index"], row["layer"],
                        row["key_task"])].append(row)
    profile_pool_rows = []
    for (model_type, pool, layer, key_task), rows in sorted(profile_groups.items()):
        profile_pool_rows.append({
            "model_type": model_type, "pool_index": pool, "layer": layer,
            "key_task": key_task, "is_self": rows[0]["is_self"],
            "attention": float(np.mean([row["attention"] for row in rows])),
            "input_attention": float(np.mean([row["input_attention"] for row in rows])),
            "output_attention": float(np.mean([row["output_attention"] for row in rows])),
            "same_attention": float(np.nanmean([row["same_attention"] for row in rows]))
            if not rows[0]["is_self"] else np.nan,
            "different_attention": float(np.nanmean([row["different_attention"] for row in rows]))
            if not rows[0]["is_self"] else np.nan,
        })
    write_csv(profile_pool_rows, args.summary_dir / "final_query_attention_summary.csv")

    layers = sorted({row["layer"] for row in profile_pool_rows})
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey=True)
    for layer, ax in zip(layers, axes.flat):
        for model_type in ("trained", "untrained"):
            means, tasks = [], []
            for key_task in sorted({row["key_task"] for row in profile_pool_rows}):
                values = [row["attention"] for row in profile_pool_rows
                          if row["model_type"] == model_type and row["layer"] == layer
                          and row["key_task"] == key_task]
                tasks.append(key_task); means.append(np.mean(values))
            ax.plot(tasks, means, label=model_type.capitalize())
        ax.axvline(49, color="0.5", linewidth=0.8, linestyle="--")
        ax.set_title(f"Layer {layer}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Key task")
    for ax in axes[:, 0]:
        ax.set_ylabel("Attention probability")
    axes[0, 0].legend()
    fig.suptitle("Final query attention by task")
    fig.tight_layout()
    fig.savefig(args.output_dir / "final_query_attention_by_task.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey=True)
    for layer, ax in zip(layers, axes.flat):
        layer_rows = [row for row in profile_pool_rows
                      if row["model_type"] == "trained" and row["layer"] == layer
                      and not row["is_self"]]
        for field, label in (("same_attention", "Same component"),
                             ("different_attention", "Different component")):
            means, tasks = [], []
            for key_task in sorted({row["key_task"] for row in layer_rows}):
                values = [row[field] for row in layer_rows if row["key_task"] == key_task]
                tasks.append(key_task); means.append(np.mean(values))
            ax.plot(tasks, means, label=label)
        ax.set_title(f"Layer {layer}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Previous task")
    for ax in axes[:, 0]:
        ax.set_ylabel("Attention probability")
    axes[0, 0].legend()
    fig.suptitle("Final query attention by component relation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "final_query_attention_by_relation.png", dpi=200)
    plt.close(fig)

    final_layer = max(layers)
    token_rows = [row for row in profile_pool_rows if row["model_type"] == "trained"
                  and row["layer"] == final_layer and not row["is_self"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for field, label in (("input_attention", "Task input token"),
                         ("output_attention", "Task output token")):
        tasks, means = [], []
        for key_task in sorted({row["key_task"] for row in token_rows}):
            values = [row[field] for row in token_rows if row["key_task"] == key_task]
            tasks.append(key_task); means.append(np.mean(values))
        ax.plot(tasks, means, label=label)
    ax.set(xlabel="Previous task", ylabel="Attention probability")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "final_query_attention_by_token_type.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {len(pool_layer_rows)} selectivity rows, {len(profile_pool_rows)} "
          "profile summary rows, and six figures")


if __name__ == "__main__":
    main()

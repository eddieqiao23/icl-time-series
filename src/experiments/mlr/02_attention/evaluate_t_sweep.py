#!/usr/bin/env python3
"""Measure final-query task-level attention across trained T checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy  # Import before torch to keep the macOS OpenMP runtime ordering stable.
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(HERE.parent)]

from common.results import build_manifest, write_json  # noqa: E402
from common.runtime import (  # noqa: E402
    forward_internals, load_condition, normalized_pool, sample_with_pool,
)


def write_csv(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_pool(model, *, pool: torch.Tensor, T: int, N: int,
                  noise_std: float, num_prompts: int, batch_size: int,
                  seed: int, device: torch.device) -> list[dict]:
    # Accumulate across prompts and heads in vectorized tensors. Each previous
    # task receives the sum of its packed-input and output-token attention.
    layer_sums = None
    self_sums = None
    observation_count = 0
    for start in range(0, num_prompts, batch_size):
        size = min(batch_size, num_prompts - start)
        xs, ys, _ = sample_with_pool(
            pool=pool, T=T, N=N, noise_std=noise_std,
            batch_size=size, seed=seed + start,
        )
        outputs = forward_internals(model, xs, ys, device=device, attentions=True)
        source = 2 * (N - 1)
        batch_layers = []
        batch_self = []
        for attention in outputs.attentions:
            if attention is None:
                raise RuntimeError("Backbone did not return attention weights")
            attention = attention.detach().cpu()
            batch_layers.append(
                attention[:, :, source, 0:source:2]
                + attention[:, :, source, 1:source:2]
            )
            batch_self.append(attention[:, :, source, source])
        task_attention = torch.stack(batch_layers).sum(dim=(1, 2))
        self_attention = torch.stack(batch_self).sum(dim=(1, 2))
        layer_sums = task_attention if layer_sums is None else layer_sums + task_attention
        self_sums = self_attention if self_sums is None else self_sums + self_attention
        observation_count += size * outputs.attentions[0].shape[1]

    rows = []
    for layer in range(layer_sums.shape[0]):
        for key_task in range(N - 1):
            rows.append({
                "T": T, "layer": layer, "key_task": key_task,
                "attention": float(layer_sums[layer, key_task] / observation_count),
                "is_self": False, "num_prompts": num_prompts,
            })
        rows.append({
            "T": T, "layer": layer, "key_task": N - 1,
            "attention": float(self_sums[layer] / observation_count),
            "is_self": True, "num_prompts": num_prompts,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "artifacts" / "t_sweep")
    parser.add_argument("--T", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--num-pools", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "final_query_attention_by_t.csv"
    rows = []
    records = []
    started = time.monotonic()
    completed = 0
    total = len(args.T) * args.num_pools
    for T in args.T:
        model, _, record = load_condition(
            args.models_root, T=T, K=args.K, N=args.N,
            noise_std=args.noise, device=device,
        )
        records.append(asdict(record))
        for pool_index in range(args.num_pools):
            pool = normalized_pool(args.K, 4, args.seed + 10_000 + pool_index)
            pool_rows = evaluate_pool(
                model, pool=pool, T=T, N=args.N, noise_std=args.noise,
                num_prompts=args.num_prompts, batch_size=args.batch_size,
                seed=args.seed + pool_index * 1_000, device=device,
            )
            for row in pool_rows:
                row["pool_index"] = pool_index
            rows.extend(pool_rows)
            write_csv(rows, csv_path)
            completed += 1
            elapsed = time.monotonic() - started
            remaining = elapsed / completed * (total - completed)
            print(f"T={T} pool {pool_index + 1}/{args.num_pools}: "
                  f"estimated remaining {remaining / 60:.1f} min", flush=True)

    manifest_path = args.output_dir / "run_manifest.json"
    write_json(build_manifest(
        repo_root=REPO_ROOT, experiment="02_final_query_attention_t_sweep",
        command=sys.argv,
        parameters=vars(args) | {"models_root": str(args.models_root),
                                 "output_dir": str(args.output_dir)},
        checkpoints=records,
        seeds={"base_seed": args.seed, "pool_offset": 10_000},
        outputs=[str(csv_path), str(manifest_path)],
    ), manifest_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()

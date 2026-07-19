#!/usr/bin/env python3
"""Measure task-level attention routing and component selectivity."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(HERE.parent)]

from common.results import build_manifest, write_json  # noqa: E402
from common.runtime import (  # noqa: E402
    forward_internals, load_condition, normalized_pool, sample_with_pool,
    untrained_control,
)


def write_csv(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def pool_metrics(model, *, pool: torch.Tensor, pool_index: int, model_type: str,
                 T: int, N: int, noise_std: float, num_prompts: int,
                 batch_size: int, source_positions: list[int], seed: int,
                 device: torch.device) -> tuple[list[dict], list[dict]]:
    accumulators = {}
    profile_accumulators = {}
    for start in range(0, num_prompts, batch_size):
        size = min(batch_size, num_prompts - start)
        xs, ys, ids = sample_with_pool(
            pool=pool, T=T, N=N, noise_std=noise_std, batch_size=size,
            seed=seed + start,
        )
        outputs = forward_internals(model, xs, ys, device=device, attentions=True)
        if not outputs.attentions or outputs.attentions[0] is None:
            raise RuntimeError("Backbone did not return attention weights")

        for layer, attention in enumerate(outputs.attentions):
            attention = attention.detach().cpu()
            heads = attention.shape[1]
            for position in source_positions:
                source = 2 * position  # prediction is read from the current input token
                input_keys = torch.arange(0, source, 2)
                output_keys = torch.arange(1, source, 2)
                input_attention = attention[:, :, source, input_keys]
                output_attention = attention[:, :, source, output_keys]
                same_task = ids[:, :position] == ids[:, position, None]
                different_task = ~same_task
                for head in range(heads):
                    key = (layer, head, position)
                    acc = accumulators.setdefault(key, {
                        "same_sum": 0.0, "same_count": 0, "different_sum": 0.0,
                        "different_count": 0, "input_sum": 0.0, "output_sum": 0.0,
                        "token_count": 0, "self_sum": 0.0, "prompt_count": 0,
                    })
                    head_input = input_attention[:, head]
                    head_output = output_attention[:, head]
                    same_tokens = same_task.sum().item() * 2
                    different_tokens = different_task.sum().item() * 2
                    acc["same_sum"] += float((head_input[same_task] .sum() + head_output[same_task].sum()).item())
                    acc["same_count"] += int(same_tokens)
                    acc["different_sum"] += float((head_input[different_task].sum() + head_output[different_task].sum()).item())
                    acc["different_count"] += int(different_tokens)
                    acc["input_sum"] += float(head_input.sum().item())
                    acc["output_sum"] += float(head_output.sum().item())
                    acc["token_count"] += int(size * position)
                    acc["self_sum"] += float(attention[:, head, source, source].sum().item())
                    acc["prompt_count"] += size

            # Preserve the absolute final-query profile instead of reducing it
            # immediately to a selectivity score. Input/output weights are
            # summed to one task-level value for every previous task.
            target_position = N - 1
            source = 2 * target_position
            input_attention = attention[:, :, source, 0:source:2]
            output_attention = attention[:, :, source, 1:source:2]
            same_task = ids[:, :target_position] == ids[:, target_position, None]
            for head in range(heads):
                for key_task in range(target_position):
                    key = (layer, head, key_task)
                    acc = profile_accumulators.setdefault(key, {
                        "total_sum": 0.0, "input_sum": 0.0, "output_sum": 0.0,
                        "same_sum": 0.0, "same_count": 0,
                        "different_sum": 0.0, "different_count": 0,
                        "prompt_count": 0,
                    })
                    values = input_attention[:, head, key_task] + output_attention[:, head, key_task]
                    same = same_task[:, key_task]
                    different = ~same
                    acc["total_sum"] += float(values.sum().item())
                    acc["input_sum"] += float(input_attention[:, head, key_task].sum().item())
                    acc["output_sum"] += float(output_attention[:, head, key_task].sum().item())
                    acc["same_sum"] += float(values[same].sum().item())
                    acc["same_count"] += int(same.sum().item())
                    acc["different_sum"] += float(values[different].sum().item())
                    acc["different_count"] += int(different.sum().item())
                    acc["prompt_count"] += size

                self_key = (layer, head, target_position)
                self_acc = profile_accumulators.setdefault(self_key, {
                    "total_sum": 0.0, "input_sum": 0.0, "output_sum": 0.0,
                    "same_sum": 0.0, "same_count": 0,
                    "different_sum": 0.0, "different_count": 0,
                    "prompt_count": 0,
                })
                self_values = attention[:, head, source, source]
                self_acc["total_sum"] += float(self_values.sum().item())
                self_acc["input_sum"] += float(self_values.sum().item())
                self_acc["prompt_count"] += size
        del outputs

    cosine = float(torch.dot(pool[0], pool[1]).item()) if len(pool) == 2 else np.nan
    rows = []
    for (layer, head, position), acc in sorted(accumulators.items()):
        same = acc["same_sum"] / acc["same_count"]
        different = acc["different_sum"] / acc["different_count"]
        rows.append({
            "model_type": model_type, "pool_index": pool_index,
            "pool_cosine": cosine, "layer": layer, "head": head,
            "source_position": position, "same_attention": same,
            "different_attention": different,
            "selectivity": (same - different) / (same + different + 1e-12),
            "input_attention": acc["input_sum"] / acc["token_count"],
            "output_attention": acc["output_sum"] / acc["token_count"],
            "self_attention": acc["self_sum"] / acc["prompt_count"],
            "num_prompts": num_prompts,
        })
    profile_rows = []
    for (layer, head, key_task), acc in sorted(profile_accumulators.items()):
        profile_rows.append({
            "model_type": model_type, "pool_index": pool_index,
            "pool_cosine": cosine, "layer": layer, "head": head,
            "source_task": N - 1, "key_task": key_task,
            "is_self": key_task == N - 1,
            "attention": acc["total_sum"] / acc["prompt_count"],
            "input_attention": acc["input_sum"] / acc["prompt_count"],
            "output_attention": acc["output_sum"] / acc["prompt_count"],
            "same_attention": (acc["same_sum"] / acc["same_count"]
                               if acc["same_count"] else ""),
            "different_attention": (acc["different_sum"] / acc["different_count"]
                                    if acc["different_count"] else ""),
            "same_count": acc["same_count"],
            "different_count": acc["different_count"],
            "num_prompts": num_prompts,
        })
    return rows, profile_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "summaries")
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--num-pools", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--positions", type=int, nargs="+", default=[5, 10, 20, 30, 40, 49])
    parser.add_argument("--seed", type=int, default=31_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    trained, config, record = load_condition(
        args.models_root, T=args.T, K=args.K, N=args.N,
        noise_std=args.noise, device=device,
    )
    random_model = untrained_control(config, device, seed=args.seed + 999)
    rows = []
    profile_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "attention_by_head.csv"
    started = time.monotonic()
    for model_type, model in (("trained", trained), ("untrained", random_model)):
        for pool_index in range(args.num_pools):
            pool_started = time.monotonic()
            pool = normalized_pool(args.K, 4, args.seed + 10_000 + pool_index)
            pool_rows, pool_profiles = pool_metrics(
                model, pool=pool, pool_index=pool_index, model_type=model_type,
                T=args.T, N=args.N, noise_std=args.noise,
                num_prompts=args.num_prompts, batch_size=args.batch_size,
                source_positions=args.positions, seed=args.seed + pool_index * 1_000,
                device=device,
            )
            rows.extend(pool_rows)
            profile_rows.extend(pool_profiles)
            write_csv(rows, csv_path)
            write_csv(profile_rows, args.output_dir / "final_query_attention_by_task.csv")
            completed = (0 if model_type == "trained" else args.num_pools) + pool_index + 1
            remaining = 2 * args.num_pools - completed
            rate = (time.monotonic() - started) / completed
            print(f"{model_type} pool {pool_index + 1}/{args.num_pools}: "
                  f"{time.monotonic() - pool_started:.1f}s; "
                  f"estimated remaining {rate * remaining / 60:.1f} min", flush=True)

    manifest_path = args.output_dir / "run_manifest.json"
    manifest = build_manifest(
        repo_root=REPO_ROOT, experiment="02_attention_component_selectivity",
        command=sys.argv,
        parameters=vars(args) | {"models_root": str(args.models_root),
                                 "output_dir": str(args.output_dir)},
        checkpoints=[asdict(record)],
        seeds={"base_seed": args.seed, "pool_offset": 10_000,
               "untrained_model_seed": args.seed + 999},
        outputs=[str(csv_path),
                 str(args.output_dir / "final_query_attention_by_task.csv"),
                 str(manifest_path)],
    )
    write_json(manifest, manifest_path)
    print(f"Wrote {len(rows)} selectivity rows and {len(profile_rows)} profile rows")


if __name__ == "__main__":
    main()

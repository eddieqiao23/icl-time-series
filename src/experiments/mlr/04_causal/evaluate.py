#!/usr/bin/env python3
"""Run head, context, support-block, and activation-patching interventions."""

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
)


def atomic_csv(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def predict_from_embeddings(model, embeddings, *, head_mask=None):
    with torch.no_grad():
        hidden = model._backbone(
            inputs_embeds=embeddings, head_mask=head_mask, return_dict=True
        ).last_hidden_state
        return model._read_out(hidden)[:, ::2, 0]


def mse_at(prediction, target, positions):
    return float(torch.square(prediction[:, positions].cpu() - target[:, positions]).mean().item())


def replace_context(embeddings, ids, position, relation):
    modified = embeddings.clone()
    input_mean = embeddings[:, 0::2].mean(dim=(0, 1))
    output_mean = embeddings[:, 1::2].mean(dim=(0, 1))
    earlier = ids[:, :position]
    if relation == "same": mask = earlier == ids[:, position, None]
    elif relation == "different": mask = earlier != ids[:, position, None]
    elif relation == "all": mask = torch.ones_like(earlier, dtype=torch.bool)
    else: raise ValueError(relation)
    for batch in range(len(ids)):
        tasks = torch.nonzero(mask[batch], as_tuple=False).flatten()
        modified[batch, 2 * tasks] = input_mean
        modified[batch, 2 * tasks + 1] = output_mean
    return modified


def support_variants(xs, T, d):
    variants, names = [], []
    for support in range(T):
        modified = xs.clone(); start = support * (d + 1); stop = start + d + 1
        modified[..., start:stop] = xs[..., start:stop].mean(dim=(0, 1))
        variants.append(modified); names.append(f"support_{support}")
    modified = xs.clone()
    stop = T * (d + 1)
    modified[..., :stop] = xs[..., :stop].mean(dim=(0, 1))
    variants.append(modified); names.append("all_supports")
    return names, variants


def patch_query_stage(model, corrupt_embeddings, clean_states, *, stage, source):
    """Patch the query residual at embedding/block-boundary stages 0..L."""
    if stage == 0:
        # Stage zero is the explicit corrupt baseline. hidden_states[0]
        # already contains positional embeddings, so injecting it as an
        # inputs_embeds value would add position information twice.
        return predict_from_embeddings(model, corrupt_embeddings)

    if stage < len(model._backbone.h):
        def pre_hook(_module, inputs):
            hidden = inputs[0].clone()
            hidden[:, source] = clean_states[stage][:, source]
            return (hidden,) + inputs[1:]
        handle = model._backbone.h[stage].register_forward_pre_hook(pre_hook)
    else:
        def final_hook(_module, _inputs, output):
            hidden = output.clone()
            hidden[:, source] = clean_states[stage][:, source]
            return hidden
        handle = model._backbone.ln_f.register_forward_hook(final_hook)
    try:
        return predict_from_embeddings(model, corrupt_embeddings)
    finally:
        handle.remove()


def evaluate_pool(model, *, pool, pool_index, T, K, N, noise_std,
                  num_prompts, positions, seed, device):
    xs, ys, ids = sample_with_pool(
        pool=pool, T=T, N=N, noise_std=noise_std,
        batch_size=num_prompts, seed=seed,
    )
    xs_device, ys_device = xs.to(device), ys.to(device)
    with torch.no_grad():
        embeddings = model._read_in(model._combine(xs_device, ys_device))
    baseline_prediction = predict_from_embeddings(model, embeddings)
    late_positions = [position for position in positions if position >= 30]
    baseline_late = mse_at(baseline_prediction, ys, late_positions)
    n_layers = len(model._backbone.h); n_heads = model._backbone.config.n_head

    head_rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            mask = torch.ones(n_layers, n_heads, device=device); mask[layer, head] = 0
            prediction = predict_from_embeddings(model, embeddings, head_mask=mask)
            mse = mse_at(prediction, ys, late_positions)
            head_rows.append({
                "pool_index": pool_index, "layer": layer, "head": head,
                "baseline_mse": baseline_late, "ablated_mse": mse,
                "delta_mse": mse - baseline_late,
                "relative_delta": (mse - baseline_late) / (baseline_late + 1e-12),
                "num_prompts": num_prompts,
            })

    context_rows = []
    for position in positions:
        variants = []
        for relation in ("same", "different", "all"):
            variants.append(replace_context(embeddings, ids, position, relation))
        predictions = predict_from_embeddings(model, torch.cat(variants, dim=0))
        base_mse = mse_at(baseline_prediction, ys, [position])
        for index, relation in enumerate(("same", "different", "all")):
            prediction = predictions[index * num_prompts:(index + 1) * num_prompts]
            mse = mse_at(prediction, ys, [position])
            context_rows.append({
                "pool_index": pool_index, "position": position,
                "condition": relation, "baseline_mse": base_mse,
                "intervention_mse": mse, "delta_mse": mse - base_mse,
                "num_prompts": num_prompts,
            })

    support_rows = []
    names, variants = support_variants(xs_device, T, pool.shape[1])
    with torch.no_grad():
        variant_predictions = model(
            torch.cat(variants), torch.cat([ys_device] * len(variants))
        ).cpu()
    for index, name in enumerate(names):
        prediction = variant_predictions[index * num_prompts:(index + 1) * num_prompts]
        mse = mse_at(prediction, ys, late_positions)
        support_rows.append({
            "pool_index": pool_index, "condition": name,
            "baseline_mse": baseline_late, "intervention_mse": mse,
            "delta_mse": mse - baseline_late, "num_prompts": num_prompts,
        })

    target_position = max(positions); source = 2 * target_position
    clean_outputs = forward_internals(model, xs, ys, device=device, hidden_states=True)
    corrupt_embeddings = replace_context(embeddings, ids, target_position, "same")
    corrupt_prediction = predict_from_embeddings(model, corrupt_embeddings)
    clean_mse = mse_at(baseline_prediction, ys, [target_position])
    corrupt_mse = mse_at(corrupt_prediction, ys, [target_position])
    patch_rows = []
    for stage in range(n_layers + 1):
        patched = patch_query_stage(
            model, corrupt_embeddings, clean_outputs.hidden_states,
            stage=stage, source=source,
        )
        patched_mse = mse_at(patched, ys, [target_position])
        patch_rows.append({
            "pool_index": pool_index, "stage": stage,
            "position": target_position, "clean_mse": clean_mse,
            "corrupt_mse": corrupt_mse, "patched_mse": patched_mse,
            "recovery": (corrupt_mse - patched_mse) / (corrupt_mse - clean_mse + 1e-12),
            "num_prompts": num_prompts,
        })
    return head_rows, context_rows, support_rows, patch_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "summaries")
    parser.add_argument("--T", type=int, default=3); parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--N", type=int, default=50); parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--num-pools", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--positions", type=int, nargs="+", default=[10, 20, 30, 40, 49])
    parser.add_argument("--seed", type=int, default=59_000); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    model, _config, record = load_condition(
        args.models_root, T=args.T, K=args.K, N=args.N,
        noise_std=args.noise, device=device,
    )
    tables = {"head_ablation": [], "context_ablation": [],
              "support_ablation": [], "activation_patching": []}
    args.output_dir.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    for pool_index in range(args.num_pools):
        pool = normalized_pool(args.K, 4, args.seed + 10_000 + pool_index)
        outputs = evaluate_pool(
            model, pool=pool, pool_index=pool_index, T=args.T, K=args.K,
            N=args.N, noise_std=args.noise, num_prompts=args.num_prompts,
            positions=args.positions, seed=args.seed + pool_index * 1_000,
            device=device,
        )
        for name, rows in zip(tables, outputs):
            tables[name].extend(rows); atomic_csv(tables[name], args.output_dir / f"{name}.csv")
        elapsed = time.monotonic() - started
        remaining = elapsed / (pool_index + 1) * (args.num_pools - pool_index - 1)
        print(f"pool {pool_index + 1}/{args.num_pools}: {elapsed:.1f}s elapsed; "
              f"estimated remaining {remaining / 60:.1f} min", flush=True)
    manifest_path = args.output_dir / "run_manifest.json"
    write_json(build_manifest(
        repo_root=REPO_ROOT, experiment="04_causal_interventions", command=sys.argv,
        parameters=vars(args) | {"models_root": str(args.models_root), "output_dir": str(args.output_dir)},
        checkpoints=[asdict(record)], seeds={"base_seed": args.seed, "pool_offset": 10_000},
        outputs=[str(args.output_dir / f"{name}.csv") for name in tables] + [str(manifest_path)],
    ), manifest_path)
    print("Completed causal intervention suite")


if __name__ == "__main__": main()

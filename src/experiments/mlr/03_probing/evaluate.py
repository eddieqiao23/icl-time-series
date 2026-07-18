#!/usr/bin/env python3
"""Grouped-CV probes for active coefficient and component identity."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def collect(model, *, pool: torch.Tensor, T: int, N: int, noise_std: float,
            num_prompts: int, batch_size: int, positions: list[int], seed: int,
            device: torch.device):
    layers = []
    raw, beta, component, groups, position_values = [], [], [], [], []
    for start in range(0, num_prompts, batch_size):
        size = min(batch_size, num_prompts - start)
        xs, ys, ids = sample_with_pool(
            pool=pool, T=T, N=N, noise_std=noise_std, batch_size=size,
            seed=seed + start,
        )
        outputs = forward_internals(model, xs, ys, device=device, hidden_states=True)
        selected = []
        for hidden in outputs.hidden_states:  # embedding plus one state per block
            selected.append(hidden[:, [2 * p for p in positions]].detach().cpu().numpy())
        layers.append(np.stack(selected, axis=2))  # (B, P, L, H)
        raw.append(xs[:, positions].numpy())
        ids_selected = ids[:, positions]
        beta.append(pool[ids_selected].numpy())
        component.append(ids_selected.numpy())
        prompt_ids = np.arange(start, start + size)[:, None]
        groups.append(np.broadcast_to(prompt_ids, (size, len(positions))))
        position_values.append(np.broadcast_to(np.asarray(positions)[None], (size, len(positions))))
    return {
        "hidden": np.concatenate(layers).reshape(-1, layers[0].shape[2], layers[0].shape[3]),
        "raw": np.concatenate(raw).reshape(-1, raw[0].shape[-1]),
        "beta": np.concatenate(beta).reshape(-1, pool.shape[1]),
        "component": np.concatenate(component).reshape(-1),
        "groups": np.concatenate(groups).reshape(-1),
        "position": np.concatenate(position_values).reshape(-1),
    }


def probe_representation(X: np.ndarray, data: dict, *, model_type: str,
                         pool_index: int, layer: int, folds: int,
                         seed: int) -> list[dict]:
    groups = data["groups"]
    positions = data["position"]
    beta = data["beta"]
    component = data["component"]
    rng = np.random.default_rng(seed)
    beta_shuffled = beta[rng.permutation(len(beta))]
    component_shuffled = component[rng.permutation(len(component))]
    splitter = GroupKFold(n_splits=folds)
    predictions = {
        "beta": np.zeros_like(beta), "beta_shuffled": np.zeros_like(beta),
        "component": np.zeros_like(component),
        "component_shuffled": np.zeros_like(component),
    }
    for train, test in splitter.split(X, groups=groups):
        beta_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        beta_model.fit(X[train], beta[train]); predictions["beta"][test] = beta_model.predict(X[test])
        beta_null = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        beta_null.fit(X[train], beta_shuffled[train])
        predictions["beta_shuffled"][test] = beta_null.predict(X[test])
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1_000, random_state=seed)
        )
        classifier.fit(X[train], component[train])
        predictions["component"][test] = classifier.predict(X[test])
        classifier_null = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1_000, random_state=seed)
        )
        classifier_null.fit(X[train], component_shuffled[train])
        predictions["component_shuffled"][test] = classifier_null.predict(X[test])

    rows = []
    for position in sorted(np.unique(positions)):
        mask = positions == position
        for control, target, prediction, metric in (
            ("actual", beta, predictions["beta"], "beta_r2"),
            ("shuffled", beta_shuffled, predictions["beta_shuffled"], "beta_r2"),
        ):
            rows.append({
                "model_type": model_type, "pool_index": pool_index, "layer": layer,
                "position": int(position), "metric": metric, "control": control,
                "score": float(r2_score(target[mask], prediction[mask], multioutput="variance_weighted")),
                "num_prompts": int(mask.sum()),
            })
        for control, target, prediction, metric in (
            ("actual", component, predictions["component"], "component_accuracy"),
            ("shuffled", component_shuffled, predictions["component_shuffled"], "component_accuracy"),
        ):
            rows.append({
                "model_type": model_type, "pool_index": pool_index, "layer": layer,
                "position": int(position), "metric": metric, "control": control,
                "score": float(accuracy_score(target[mask], prediction[mask])),
                "num_prompts": int(mask.sum()),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "summaries")
    parser.add_argument("--T", type=int, default=3); parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--N", type=int, default=50); parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--num-pools", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--positions", type=int, nargs="+", default=[5, 10, 20, 30, 40, 49])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    trained, config, record = load_condition(
        args.models_root, T=args.T, K=args.K, N=args.N,
        noise_std=args.noise, device=device,
    )
    models = (("trained", trained),
              ("untrained", untrained_control(config, device, args.seed + 999)))
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "probe_scores.csv"
    started = time.monotonic()
    for pool_index in range(args.num_pools):
        pool = normalized_pool(args.K, 4, args.seed + 10_000 + pool_index)
        for model_type, model in models:
            data = collect(
                model, pool=pool, T=args.T, N=args.N, noise_std=args.noise,
                num_prompts=args.num_prompts, batch_size=args.batch_size,
                positions=args.positions, seed=args.seed + pool_index * 1_000,
                device=device,
            )
            for layer in range(data["hidden"].shape[1]):
                rows.extend(probe_representation(
                    data["hidden"][:, layer], data, model_type=model_type,
                    pool_index=pool_index, layer=layer, folds=args.folds,
                    seed=args.seed + pool_index * 100 + layer,
                ))
            if model_type == "trained":
                rows.extend(probe_representation(
                    data["raw"], data, model_type="raw_input", pool_index=pool_index,
                    layer=-1, folds=args.folds, seed=args.seed + pool_index * 100 - 1,
                ))
        write_csv(rows, csv_path)
        elapsed = time.monotonic() - started
        remaining = elapsed / (pool_index + 1) * (args.num_pools - pool_index - 1)
        print(f"pool {pool_index + 1}/{args.num_pools}: {elapsed:.1f}s elapsed; "
              f"estimated remaining {remaining / 60:.1f} min", flush=True)

    manifest_path = args.output_dir / "run_manifest.json"
    write_json(build_manifest(
        repo_root=REPO_ROOT, experiment="03_grouped_representation_probes",
        command=sys.argv,
        parameters=vars(args) | {"models_root": str(args.models_root),
                                 "output_dir": str(args.output_dir)},
        checkpoints=[asdict(record)],
        seeds={"base_seed": args.seed, "pool_offset": 10_000,
               "untrained_model_seed": args.seed + 999},
        outputs=[str(csv_path), str(manifest_path)],
    ), manifest_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate transformer and analytic baselines on identical fixed MLR pools."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(HERE.parent)]

from common.checkpoints import discover_checkpoints, select_checkpoint  # noqa: E402
from common.results import build_manifest, write_json  # noqa: E402
from baselines import (  # noqa: E402
    em_ridge_history, known_pool_bayes, ridge_current, ridge_history, unpack_tokens,
)
from models import build_model  # noqa: E402
from samplers import MLRSampler  # noqa: E402

METHODS = ("transformer", "known_pool", "em_ridge", "ridge_current", "ridge_history")


class ModelArgs:
    def __init__(self, values: dict):
        self.__dict__.update(values)
        self.predict_vector = values.get("predict_vector", False)


def normalized_pool(K: int, d: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    pool = torch.randn(K, d, generator=generator)
    return pool / pool.norm(dim=1, keepdim=True)


def load_model(run_dir: Path, device: torch.device):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = build_model(ModelArgs(config["model"])).to(device)
    try:
        state = torch.load(run_dir / "state.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "state.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval(), config


def evaluate_condition(*, record, K: int, noise_std: float, num_pools: int,
                       batch_size: int, methods: tuple[str, ...], seed_base: int,
                       device: torch.device) -> list[dict]:
    model, config = load_model(Path(record.run_dir), device)
    kwargs = config["training"]["task_kwargs"]
    T, N, d = record.T, record.N, record.d
    sampler = MLRSampler(
        n_dims=record.D, num_mixture_models=K, num_batches_per_sample=N,
        batch_size_per_task=T, regressor_dim=d, normalize_coeffs=True,
        regenerate_pool=False, noise_std=noise_std, use_gpu=False,
        device=torch.device("cpu"),
    )
    rows = []
    for pool_index in range(num_pools):
        pool_seed = seed_base + 10_000 + pool_index
        prompt_seed = seed_base + pool_index
        pool = normalized_pool(K, d, pool_seed)
        sampler.coefficient_pool = pool
        torch.manual_seed(prompt_seed)
        xs = sampler.sample_xs(N, batch_size)
        targets = sampler.current_ys.numpy().astype(np.float64)
        X, y, query = unpack_tokens(xs.numpy().astype(np.float64), T, d)
        predictions = {}
        if "transformer" in methods:
            with torch.no_grad():
                predictions["transformer"] = model(
                    xs.to(device), sampler.current_ys.to(device)
                ).cpu().numpy()
        if "known_pool" in methods:
            predictions["known_pool"] = known_pool_bayes(X, y, query, pool.numpy(), noise_std)
        if "ridge_current" in methods:
            predictions["ridge_current"] = ridge_current(X, y, query)
        if "ridge_history" in methods:
            predictions["ridge_history"] = ridge_history(X, y, query)
        if "em_ridge" in methods:
            predictions["em_ridge"] = em_ridge_history(
                X, y, query, components=K, noise_std=noise_std,
                regularization=max(noise_std ** 2 * d, 0.01),
                seed=seed_base + 20_000 + pool_index,
            )

        for method, prediction in predictions.items():
            squared_error = np.square(prediction - targets)
            for position in range(N):
                values = squared_error[:, position]
                valid = values[~np.isnan(values)]
                rows.append({
                    "T": T, "K": K, "N": N, "d": d, "noise_std": noise_std,
                    "method": method, "pool_index": pool_index,
                    "position": position, "num_examples": len(valid),
                    "mse": float(valid.mean()) if len(valid) else "",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "summaries")
    parser.add_argument("--T", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--K", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--min-step", type=int, default=500_000)
    parser.add_argument("--num-pools", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=17_000)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = discover_checkpoints(args.models_root, read_checkpoints=True)
    selected, rows = [], []
    combo_index = 0
    for K in args.K:
        for noise in args.noise:
            for T in args.T:
                record = select_checkpoint(records, T=T, K=K, N=args.N,
                                           noise_std=noise, min_step=args.min_step)
                selected.append(record)
                print(f"Evaluating {record.condition} at step {record.train_step}")
                rows.extend(evaluate_condition(
                    record=record, K=K, noise_std=noise, num_pools=args.num_pools,
                    batch_size=args.batch_size, methods=tuple(args.methods),
                    # Reuse the same pools and assignment seed across T within
                    # each (K, noise) comparison.
                    seed_base=args.seed + 100_000 * combo_index, device=device,
                ))
            combo_index += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mse_by_position.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = build_manifest(
        repo_root=REPO_ROOT, experiment="01_behavior_mse_by_position",
        command=sys.argv, parameters=vars(args) | {"models_root": str(args.models_root),
                                                   "output_dir": str(args.output_dir)},
        checkpoints=[asdict(record) for record in selected],
        seeds={"base_seed": args.seed, "condition_stride": 100_000,
               "pool_offset": 10_000}, outputs=[str(csv_path), str(manifest_path)],
    )
    write_json(manifest, manifest_path)
    print(f"Wrote {len(rows)} aggregate rows to {csv_path}")


if __name__ == "__main__":
    main()

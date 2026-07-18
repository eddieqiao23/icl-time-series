#!/usr/bin/env python3
"""Evaluate transformer and analytic baselines on identical fixed MLR pools."""

from __future__ import annotations

import argparse
import csv
import sys
import time
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
    model = None
    if "transformer" in methods:
        model, _ = load_model(Path(record.run_dir), device)
    T, N, d = record.T, record.N, record.d
    sampler = MLRSampler(
        n_dims=record.D, num_mixture_models=K, num_batches_per_sample=N,
        batch_size_per_task=T, regressor_dim=d, normalize_coeffs=True,
        regenerate_pool=False, noise_std=noise_std, use_gpu=False,
        device=torch.device("cpu"),
    )
    rows = []
    for pool_index in range(num_pools):
        pool_started = time.monotonic()
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
        if "em_ridge" in methods:
            print(f"  pool {pool_index + 1}/{num_pools}: "
                  f"{time.monotonic() - pool_started:.1f}s", flush=True)
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    """Atomically checkpoint a long-form result table."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "summaries")
    parser.add_argument("--T", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--K", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--min-step", type=int, default=0,
                        help="Optional checkpoint floor. Early-stopped final checkpoints are accepted by default.")
    parser.add_argument("--num-pools", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=17_000)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--append", action="store_true",
                        help="Replace the selected methods in an existing result CSV and preserve other methods.")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = discover_checkpoints(args.models_root, read_checkpoints=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mse_by_position.csv"
    base_rows, rows, completed_conditions = [], [], set()
    if args.append and csv_path.exists():
        with csv_path.open() as handle:
            existing_rows = list(csv.DictReader(handle))
        base_rows = [row for row in existing_rows if row["method"] not in args.methods]
        selected_rows = [row for row in existing_rows if row["method"] in args.methods]
        expected = args.num_pools * args.N
        counts = {}
        for row in selected_rows:
            key = (int(row["T"]), int(row["K"]), float(row["noise_std"]), row["method"])
            counts[key] = counts.get(key, 0) + 1
        for T in args.T:
            for K in args.K:
                for noise in args.noise:
                    if all(counts.get((T, K, noise, method), 0) == expected
                           for method in args.methods):
                        completed_conditions.add((T, K, noise))
        rows = [row for row in selected_rows
                if (int(row["T"]), int(row["K"]), float(row["noise_std"]))
                in completed_conditions]

    selected = []
    combo_index = 0
    total_conditions = len(args.T) * len(args.K) * len(args.noise)
    finished = len(completed_conditions)
    run_started = time.monotonic()
    for K in args.K:
        for noise in args.noise:
            for T in args.T:
                record = select_checkpoint(records, T=T, K=K, N=args.N,
                                           noise_std=noise, min_step=args.min_step)
                selected.append(record)
                if (T, K, noise) in completed_conditions:
                    print(f"Skipping completed {record.condition}", flush=True)
                    continue
                condition_started = time.monotonic()
                print(f"Evaluating {record.condition} at step {record.train_step}", flush=True)
                condition_rows = evaluate_condition(
                    record=record, K=K, noise_std=noise, num_pools=args.num_pools,
                    batch_size=args.batch_size, methods=tuple(args.methods),
                    # Reuse the same pools and assignment seed across T within
                    # each (K, noise) comparison.
                    seed_base=args.seed + 100_000 * combo_index, device=device,
                )
                rows.extend(condition_rows)
                write_rows(base_rows + rows, csv_path)
                finished += 1
                elapsed = time.monotonic() - run_started
                remaining_seconds = elapsed / max(finished - len(completed_conditions), 1) * (
                    total_conditions - finished
                )
                print(f"Completed in {time.monotonic() - condition_started:.1f}s; "
                      f"estimated remaining {remaining_seconds / 60:.1f} min", flush=True)
            combo_index += 1

    final_rows = base_rows + rows
    write_rows(final_rows, csv_path)
    manifest_suffix = "_" + "_".join(args.methods) if args.append else ""
    manifest_path = args.output_dir / f"run_manifest{manifest_suffix}.json"
    manifest = build_manifest(
        repo_root=REPO_ROOT, experiment="01_behavior_mse_by_position",
        command=sys.argv, parameters=vars(args) | {"models_root": str(args.models_root),
                                                   "output_dir": str(args.output_dir)},
        checkpoints=[asdict(record) for record in selected],
        seeds={"base_seed": args.seed, "condition_stride": 100_000,
               "pool_offset": 10_000}, outputs=[str(csv_path), str(manifest_path)],
    )
    write_json(manifest, manifest_path)
    print(f"Wrote {len(final_rows)} aggregate rows to {csv_path}")


if __name__ == "__main__":
    main()

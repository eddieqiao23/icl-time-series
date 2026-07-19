#!/usr/bin/env python3
"""Compare coefficient decodability across trained support-count checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(HERE), str(HERE.parent)]

from common.results import build_manifest, write_json  # noqa: E402
from common.runtime import load_condition, normalized_pool  # noqa: E402
from evaluate import collect  # noqa: E402


def write_csv(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def probe_beta(hidden: np.ndarray, data: dict, *, T: int, pool_index: int,
               layer: int, folds: int) -> list[dict]:
    predictions = np.zeros_like(data["beta"])
    splitter = GroupKFold(n_splits=folds)
    for train, test in splitter.split(hidden, groups=data["groups"]):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(hidden[train], data["beta"][train])
        predictions[test] = model.predict(hidden[test])

    rows = []
    for position in sorted(np.unique(data["position"])):
        mask = data["position"] == position
        rows.append({
            "T": T, "pool_index": pool_index, "layer": layer,
            "position": int(position), "score": float(r2_score(
                data["beta"][mask], predictions[mask],
                multioutput="variance_weighted",
            )),
            "num_prompts": int(mask.sum()),
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
    parser.add_argument("--positions", type=int, nargs="+",
                        default=[5, 10, 20, 30, 40, 49])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows: list[dict] = []
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "probe_scores_by_t.csv"
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
            data = collect(
                model, pool=pool, T=T, N=args.N, noise_std=args.noise,
                num_prompts=args.num_prompts, batch_size=args.batch_size,
                positions=args.positions, seed=args.seed + pool_index * 1_000,
                device=device,
            )
            for layer in range(data["hidden"].shape[1]):
                rows.extend(probe_beta(
                    data["hidden"][:, layer], data, T=T,
                    pool_index=pool_index, layer=layer, folds=args.folds,
                ))
            write_csv(rows, csv_path)
            completed += 1
            elapsed = time.monotonic() - started
            remaining = elapsed / completed * (total - completed)
            print(f"T={T} pool {pool_index + 1}/{args.num_pools}: "
                  f"estimated remaining {remaining / 60:.1f} min", flush=True)

    manifest_path = args.output_dir / "run_manifest.json"
    write_json(build_manifest(
        repo_root=REPO_ROOT, experiment="03_coefficient_probe_t_sweep",
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

"""Checkpoint discovery and validation for MLR experiments.

Model files are deliberately kept outside Git.  This module turns an arbitrary
``models/mlr`` directory into a reproducible, reviewable inventory without
assuming UUIDs or experiment-name suffixes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml


@dataclass(frozen=True)
class CheckpointRecord:
    run_name: str
    run_id: str
    run_dir: str
    checkpoint: str
    train_step: Optional[int]
    T: Optional[int]
    K: Optional[int]
    N: Optional[int]
    d: Optional[int]
    D: Optional[int]
    noise_std: Optional[float]
    config_valid: bool
    checkpoint_readable: bool
    error: str = ""

    @property
    def condition(self) -> str:
        noise = "noiseless" if self.noise_std == 0 else "noisy"
        return f"T{self.T}_K{self.K}_N{self.N}_{noise}"


def _load_state(path: Path) -> dict:
    """Load a trusted local checkpoint across supported PyTorch versions."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def inspect_run(run_dir: Path, read_checkpoint: bool = True) -> CheckpointRecord:
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "state.pt"
    run_name = run_dir.parent.name
    error_parts = []
    values = {"T": None, "K": None, "N": None, "d": None, "D": None,
              "noise_std": None}
    config_valid = False

    try:
        config = yaml.safe_load(config_path.read_text())
        training = config["training"]
        kwargs = training["task_kwargs"]
        if training["task"] != "linear_regression_mixture":
            raise ValueError(f"task is {training['task']!r}")
        values.update(
            T=int(kwargs["batch_size_per_task"]),
            K=int(kwargs["num_mixture_models"]),
            N=int(kwargs["num_batches_per_sample"]),
            d=int(kwargs["regressor_dim"]),
            D=int(config["model"]["n_dims"]),
            noise_std=float(kwargs.get("noise_std", 0.0)),
        )
        expected_d = values["T"] * (values["d"] + 1) + values["d"]
        if values["D"] != expected_d:
            raise ValueError(f"model.n_dims={values['D']}, expected {expected_d}")
        if int(config["model"]["n_positions"]) < values["N"]:
            raise ValueError("model has fewer positions than the evaluation prompt")
        config_valid = True
    except Exception as exc:  # inventory should report malformed runs, not abort
        error_parts.append(f"config: {exc}")

    checkpoint_readable = checkpoint_path.is_file()
    named_steps = []
    for path in run_dir.glob("model_*.pt"):
        try:
            named_steps.append(int(path.stem.removeprefix("model_")))
        except ValueError:
            pass
    train_step = max(named_steps, default=None)
    if checkpoint_readable and read_checkpoint:
        try:
            state = _load_state(checkpoint_path)
            if "model_state_dict" not in state:
                raise ValueError("missing model_state_dict")
            train_step = max(train_step or -1, int(state.get("train_step", -1)))
        except Exception as exc:
            checkpoint_readable = False
            error_parts.append(f"checkpoint: {exc}")

    return CheckpointRecord(
        run_name=run_name,
        run_id=run_dir.name,
        run_dir=str(run_dir.resolve()),
        checkpoint=str(checkpoint_path.resolve()),
        train_step=train_step,
        config_valid=config_valid,
        checkpoint_readable=checkpoint_readable,
        error="; ".join(error_parts),
        **values,
    )


def discover_checkpoints(models_root: Path, read_checkpoints: bool = True) -> list[CheckpointRecord]:
    if not models_root.is_dir():
        raise FileNotFoundError(f"Model root does not exist: {models_root}")
    run_dirs = sorted({
        p.parent for p in models_root.rglob("config.yaml")
        if "wandb" not in p.relative_to(models_root).parts
    })
    return [inspect_run(path, read_checkpoint=read_checkpoints) for path in run_dirs]


def select_checkpoint(records: Iterable[CheckpointRecord], *, T: int, K: int,
                      N: int, noise_std: float, min_step: int = 0) -> CheckpointRecord:
    matches = [
        record for record in records
        if record.config_valid and record.checkpoint_readable
        and (record.T, record.K, record.N) == (T, K, N)
        and record.noise_std == noise_std
        and (record.train_step or -1) >= min_step
    ]
    if not matches:
        raise LookupError(
            f"No valid checkpoint for T={T}, K={K}, N={N}, "
            f"noise_std={noise_std}, min_step={min_step}"
        )
    return max(matches, key=lambda record: (record.train_step or -1, record.run_name))


def write_inventory(records: Iterable[CheckpointRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = asdict(record) | {"condition": record.condition}
        # Inventories are portable: local model-root prefixes belong in the
        # run manifest, not in tracked summaries.
        row["run_dir"] = f"{record.run_name}/{record.run_id}"
        row["checkpoint"] = f"{row['run_dir']}/state.pt"
        rows.append(row)
    fields = list(rows[0]) if rows else [field.name for field in __import__("dataclasses").fields(CheckpointRecord)]
    with (output_dir / "checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "checkpoints.json").write_text(json.dumps(rows, indent=2) + "\n")

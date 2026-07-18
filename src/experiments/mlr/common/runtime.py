"""Shared model loading, prompt generation, and internal-state extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml

from models import build_model
from samplers import MLRSampler

from .checkpoints import CheckpointRecord, discover_checkpoints, select_checkpoint


class ModelArgs:
    def __init__(self, values: dict):
        self.__dict__.update(values)
        self.predict_vector = values.get("predict_vector", False)


def load_checkpoint(record: CheckpointRecord, device: torch.device) -> tuple[torch.nn.Module, dict]:
    run_dir = Path(record.run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = build_model(ModelArgs(config["model"])).to(device)
    try:
        state = torch.load(run_dir / "state.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "state.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval(), config


def load_condition(models_root: Path, *, T: int, K: int, N: int,
                   noise_std: float, device: torch.device) -> tuple[torch.nn.Module, dict, CheckpointRecord]:
    records = discover_checkpoints(models_root, read_checkpoints=True)
    record = select_checkpoint(records, T=T, K=K, N=N, noise_std=noise_std, min_step=0)
    model, config = load_checkpoint(record, device)
    return model, config, record


def untrained_control(config: dict, device: torch.device, seed: int) -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return build_model(ModelArgs(config["model"])).to(device).eval()


def normalized_pool(K: int, d: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    pool = torch.randn(K, d, generator=generator)
    return pool / pool.norm(dim=1, keepdim=True)


def sample_with_pool(*, pool: torch.Tensor, T: int, N: int, noise_std: float,
                     batch_size: int, seed: int,
                     assignments: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate packed prompts with a fixed pool and optional assignments."""
    K, d = pool.shape
    D = T * (d + 1) + d
    if assignments is None:
        sampler = MLRSampler(
            n_dims=D, num_mixture_models=K, num_batches_per_sample=N,
            batch_size_per_task=T, regressor_dim=d, normalize_coeffs=True,
            regenerate_pool=False, noise_std=noise_std, use_gpu=False,
            device=torch.device("cpu"),
        )
        sampler.coefficient_pool = pool.clone()
        torch.manual_seed(seed)
        xs = sampler.sample_xs(N, batch_size)
        return xs, sampler.current_ys, sampler.current_coefficient_ids

    if assignments.shape != (batch_size, N):
        raise ValueError(f"assignments has shape {assignments.shape}; expected {(batch_size, N)}")
    generator = torch.Generator().manual_seed(seed)
    x_all = torch.randn(batch_size, N, T + 1, d, generator=generator)
    betas = pool[assignments]
    y_all = (x_all * betas.unsqueeze(2)).sum(dim=-1)
    if noise_std:
        y_all += noise_std * torch.randn(batch_size, N, T + 1, generator=generator)
    pairs = torch.cat((x_all[..., :T, :], y_all[..., :T, None]), dim=-1)
    xs = torch.cat((pairs.reshape(batch_size, N, T * (d + 1)), x_all[..., T, :]), dim=-1)
    return xs, y_all[..., T], assignments.clone()


def forward_internals(model: torch.nn.Module, xs: torch.Tensor, ys: torch.Tensor,
                      *, device: torch.device, attentions: bool = False,
                      hidden_states: bool = False):
    with torch.no_grad():
        zs = model._combine(xs.to(device), ys.to(device))
        embeddings = model._read_in(zs)
        return model._backbone(
            inputs_embeds=embeddings,
            output_attentions=attentions,
            output_hidden_states=hidden_states,
            return_dict=True,
        )


def predictions_from_backbone(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    return model._read_out(hidden)[:, ::2, 0]

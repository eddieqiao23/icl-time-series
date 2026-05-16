"""Inspect a trained MLR checkpoint on fresh batches.

Checks:
  (1) per-position MSE: does the model do better with more in-context batches?
  (2) classification accuracy: of the two pool vectors, does the model pick the right one?
  (3) sample-level inspection: a few example (target, pred, mixture_id) tuples
  (4) accuracy as a function of context length
"""
import argparse
from pathlib import Path

import torch
import yaml

from samplers import get_data_sampler
from models import build_model


class _ModelArgs:
    def __init__(self, cfg_model):
        for k, v in cfg_model.items():
            setattr(self, k, v)
        if not hasattr(self, "predict_vector"):
            self.predict_vector = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/eqiao/icl-time-series/models/mlr/T2_K2_N30/6da52fd4-9da5-4c3c-9fa9-2fa34fe88835")
    ap.add_argument("--num-samples", type=int, default=512, help="batch elements for MSE-by-position stat")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
    tk = cfg["training"]["task_kwargs"]
    K = tk["num_mixture_models"]
    N = tk["num_batches_per_sample"]
    T = tk["batch_size_per_task"]
    d = tk["regressor_dim"]
    D = T * (d + 1) + d
    print(f"Loaded config: T={T}, K={K}, N={N}, d={d}, D={D}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build & load model
    model_args = _ModelArgs(cfg["model"])
    model = build_model(model_args).to(device)
    state = torch.load(model_dir / "state.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint at train_step={state['train_step']}")

    sampler = get_data_sampler(
        "linear_regression_mixture",
        n_dims=D,
        num_mixture_models=K,
        num_batches_per_sample=N,
        batch_size_per_task=T,
        regressor_dim=d,
        normalize_coeffs=True,
        regenerate_pool=True,
        use_gpu=(device.type == "cuda"),
        device=device,
    )

    B = args.num_samples
    torch.manual_seed(42)
    xs = sampler.sample_xs(n_points=N, b_size=B, n_dims_truncated=D)
    ys = sampler.current_ys
    pool_all = []  # one pool per sample-in-batch -- in our sampler, the pool is shared per sample_xs call
    pool = sampler.current_coefficient_pool  # (K, d) -- shared across this whole sample_xs call
    ids = sampler.current_coefficient_ids  # (B, N)

    with torch.no_grad():
        out = model(xs.to(device), ys.to(device)).cpu()

    # === (1) MSE per position ===
    # out shape (B, N), ys shape (B, N). Position i = the i-th batch in the prompt.
    # At position i, the model has seen i full prior batches (input+output) and the current batch's input.
    se = (out - ys) ** 2
    mse_per_pos = se.mean(dim=0)  # (N,)
    print("\n=== MSE per context position ===")
    print(f"  (position i = the i-th batch in the prompt; the model has seen {{0..i-1}} full prior batches)")
    print(f"  Overall MSE = {se.mean().item():.4f}")
    print(f"  Position 0:     MSE = {mse_per_pos[0].item():.4f}   (no prior context)")
    print(f"  Position 5:     MSE = {mse_per_pos[5].item():.4f}")
    print(f"  Position 10:    MSE = {mse_per_pos[10].item():.4f}")
    print(f"  Position 15:    MSE = {mse_per_pos[15].item():.4f}")
    print(f"  Position 20:    MSE = {mse_per_pos[20].item():.4f}")
    print(f"  Position 25:    MSE = {mse_per_pos[25].item():.4f}")
    print(f"  Position {N-1}:    MSE = {mse_per_pos[-1].item():.4f}   (most context)")

    # === (2) Classification accuracy ===
    # For each (sample, batch), compute y_hat under each pool vector. The "predicted" mixture
    # component is the one whose <b_k, x_query> is closest to the model's prediction.
    #
    # But careful: each sample_xs call uses a single pool. To compare against pool vectors,
    # we need pool_all (B, K, d). Since the pool is shared across B in our sampler, we can use
    # the same pool for all B.
    # Extract x_query for each (b, i): it's the last d slots of token xs[b, i].
    x_query = xs[..., T * (d + 1) : T * (d + 1) + d]  # (B, N, d)
    # Per-pool predictions: <b_k, x_query>
    pool_preds = torch.einsum("kd,bnd->bnk", pool, x_query)  # (B, N, K)
    # Which pool vector gives the closest prediction to the model's output?
    diffs = (pool_preds - out.unsqueeze(-1)).abs()  # (B, N, K)
    predicted_id = diffs.argmin(dim=-1)  # (B, N)
    correct = (predicted_id == ids).float()
    acc_per_pos = correct.mean(dim=0)  # (N,)
    print("\n=== Mixture-component classification accuracy ===")
    print("  (does the model's pred match <b_k, x_query> for the correct k?)")
    print(f"  Overall accuracy: {correct.mean().item():.4f}")
    print(f"  Position 0:  {acc_per_pos[0].item():.4f}")
    print(f"  Position 5:  {acc_per_pos[5].item():.4f}")
    print(f"  Position 10: {acc_per_pos[10].item():.4f}")
    print(f"  Position 15: {acc_per_pos[15].item():.4f}")
    print(f"  Position 20: {acc_per_pos[20].item():.4f}")
    print(f"  Position 25: {acc_per_pos[25].item():.4f}")
    print(f"  Position {N-1}: {acc_per_pos[-1].item():.4f}")

    # === (3) Few concrete examples ===
    print("\n=== Sample predictions (batch=0, all N positions) ===")
    print(f"  Pool b_0: {pool[0].cpu().tolist()}")
    print(f"  Pool b_1: {pool[1].cpu().tolist()}")
    print(f"\n  {'pos':>3s}  {'true_id':>7s}  {'target':>9s}  {'pred':>9s}  "
          f"{'<b0,xq>':>9s}  {'<b1,xq>':>9s}  {'guess':>5s}  {'err':>9s}")
    for i in range(N):
        target = ys[0, i].item()
        pred = out[0, i].item()
        true_id = ids[0, i].item()
        p0 = pool_preds[0, i, 0].item()
        p1 = pool_preds[0, i, 1].item()
        guess = predicted_id[0, i].item()
        err = (pred - target) ** 2
        marker = "*" if guess != true_id else ""
        print(f"  {i:>3d}  {true_id:>7d}  {target:+9.4f}  {pred:+9.4f}  "
              f"{p0:+9.4f}  {p1:+9.4f}  {guess:>5d}{marker}  {err:9.4f}")

    # === (4) Decomposition: when the model picks the right component, how accurate is it? ===
    print("\n=== Error decomposition ===")
    correct_mask = (predicted_id == ids)
    if correct_mask.any():
        mse_when_correct = se[correct_mask].mean().item()
        print(f"  MSE when classification is correct: {mse_when_correct:.6f}  ({correct_mask.float().mean().item()*100:.1f}% of cases)")
    if (~correct_mask).any():
        mse_when_wrong = se[~correct_mask].mean().item()
        print(f"  MSE when classification is wrong:   {mse_when_wrong:.6f}  ({(~correct_mask).float().mean().item()*100:.1f}% of cases)")


if __name__ == "__main__":
    main()

"""Manually inspect MLR model outputs.

Checks:
  (1) shapes line up: input (B, N, D), output (B, N), targets (B, N).
  (2) loss computed by hand matches loss_func.
  (3) causal masking: changing xs[b, j] for j > i does NOT change pred at position i.
  (4) prediction quality grows with context (later positions should be no worse).
"""
import torch

from samplers import get_data_sampler
from models import build_model
from tasks import get_task_sampler, mean_squared_error


class _ModelArgs:
    def __init__(self, n_dims, n_positions):
        self.family = "gpt2"
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = 64
        self.n_layer = 3
        self.n_head = 2
        self.predict_vector = False


def main():
    torch.manual_seed(0)

    K, N, T, d = 2, 5, 2, 4
    D = T * (d + 1) + d
    B = 4

    # Build a small untrained model on CPU
    model_args = _ModelArgs(n_dims=D, n_positions=N)
    model = build_model(model_args)
    model.eval()

    # Build sampler
    sampler = get_data_sampler(
        "linear_regression_mixture",
        n_dims=D,
        num_mixture_models=K,
        num_batches_per_sample=N,
        batch_size_per_task=T,
        regressor_dim=d,
        normalize_coeffs=True,
        regenerate_pool=True,
        use_gpu=False,
        device=torch.device("cpu"),
    )

    xs = sampler.sample_xs(n_points=N, b_size=B, n_dims_truncated=D)
    ys = sampler.current_ys  # (B, N)
    pool = sampler.current_coefficient_pool  # (K, d)
    ids = sampler.current_coefficient_ids  # (B, N)

    # === (1) shapes ===
    print("=== Shapes ===")
    print(f"  xs:   {tuple(xs.shape)}   expected ({B}, {N}, {D})")
    print(f"  ys:   {tuple(ys.shape)}   expected ({B}, {N})")
    print(f"  pool: {tuple(pool.shape)} expected ({K}, {d})")

    with torch.no_grad():
        out = model(xs, ys)
    print(f"  out:  {tuple(out.shape)} expected ({B}, {N})")
    assert out.shape == (B, N)

    # === (2) loss-by-hand vs loss_func ===
    print("\n=== Loss check ===")
    loss_hand = ((out - ys) ** 2).mean().item()
    loss_func = mean_squared_error(out, ys).item()
    print(f"  hand MSE:     {loss_hand:.6f}")
    print(f"  loss_func:    {loss_func:.6f}")
    print(f"  match: {abs(loss_hand - loss_func) < 1e-7}")

    # === (3) causal masking probe ===
    # Pick batch b=0. For each "current" position i, perturb xs[b, j] for j > i and
    # confirm the prediction at position i is unchanged.
    print("\n=== Causal-mask check (batch 0) ===")
    with torch.no_grad():
        out_orig = model(xs, ys)
    pred_orig_b0 = out_orig[0].clone()  # (N,)

    fail = 0
    for i in range(N):
        # Perturb everything strictly after position i
        xs_pert = xs.clone()
        ys_pert = ys.clone()
        # touch xs[0, i+1:, :] and ys[0, i+1:]
        if i + 1 < N:
            xs_pert[0, i + 1:, :] = torch.randn_like(xs_pert[0, i + 1:, :])
            ys_pert[0, i + 1:] = torch.randn_like(ys_pert[0, i + 1:])

        # The token at position i in the internal interleaved sequence is at index 2i.
        # All later positions (2i+1, 2i+2, ...) are "future" tokens that causal attention
        # must not let leak into the prediction at internal index 2i.
        # ALSO: the y_i target itself sits at internal index 2i+1 -- we keep it intact in ys_pert
        # because perturbing it would corrupt the output token at position 2i+1 (which is also
        # to the right of the prediction position 2i). It's already covered by ys_pert[0, i+1:].
        # Important: leave xs_pert[0, :i+1, :] and ys_pert[0, :i+1] untouched.

        with torch.no_grad():
            out_new = model(xs_pert, ys_pert)
        diff = (out_new[0, i] - pred_orig_b0[i]).abs().item()
        ok = diff < 1e-5
        marker = "OK" if ok else "LEAKAGE"
        print(f"  pos {i}: |pred_new - pred_orig| = {diff:.2e}  [{marker}]")
        if not ok:
            fail += 1
    print(f"  causal mask {'holds' if fail == 0 else 'FAILED'} for {N - fail}/{N} positions")

    # === (4) prediction variance vs target variance ===
    print("\n=== Output stats (untrained model) ===")
    print(f"  ys mean/std:    {ys.mean().item():+.4f} / {ys.std().item():.4f}")
    print(f"  pred mean/std:  {out.mean().item():+.4f} / {out.std().item():.4f}")
    print(f"  MSE per position (across batch):")
    per_pos_mse = ((out - ys) ** 2).mean(dim=0)
    for i, m in enumerate(per_pos_mse.tolist()):
        print(f"    pos {i}: {m:.4f}")

    # === (5) Dump one full example so the user can eyeball ===
    print("\n=== One concrete example (batch=0) ===")
    print(f"  Pool b_0: {pool[0].tolist()}  (norm={pool[0].norm().item():.4f})")
    print(f"  Pool b_1: {pool[1].tolist()}  (norm={pool[1].norm().item():.4f})")
    print(f"  Assignments per batch: {ids[0].tolist()}")
    print(f"  Target ys[0]:    {[f'{v:+.4f}' for v in ys[0].tolist()]}")
    print(f"  Pred   out[0]:   {[f'{v:+.4f}' for v in out[0].tolist()]}")

    # === (6) Verify packed token: in-token pairs satisfy y_t = <beta, x_t> ===
    print("\n=== Re-verify packed token layout (batch=0, position=0) ===")
    token = xs[0, 0]
    beta = pool[ids[0, 0]]
    for t in range(T):
        offset = t * (d + 1)
        x_t = token[offset : offset + d]
        y_t = token[offset + d].item()
        y_t_should = torch.dot(beta, x_t).item()
        print(f"  pair {t+1}: y={y_t:+.4f}  <beta,x>={y_t_should:+.4f}  diff={abs(y_t - y_t_should):.2e}")
    x_query = token[T * (d + 1) : T * (d + 1) + d]
    y_query_target = ys[0, 0].item()
    y_query_should = torch.dot(beta, x_query).item()
    print(f"  query: target_y={y_query_target:+.4f}  <beta,x_query>={y_query_should:+.4f}  diff={abs(y_query_target - y_query_should):.2e}")


if __name__ == "__main__":
    main()

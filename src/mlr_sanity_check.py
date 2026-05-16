"""Sanity check for the MLR sampler. Verifies:
  - shapes
  - within-token interleaving layout: [x_1, y_1, x_2, y_2, ..., x_T, y_T, x_{T+1}]
  - y_{i,t} = <beta_i, x_{i,t}> for the in-token pairs
  - current_ys[i] = <beta_i, x_{i,T+1}> (the held-out query target)
"""
import torch

from samplers import get_data_sampler


def main():
    torch.manual_seed(0)

    K, N, T, d = 2, 5, 2, 4
    D = T * (d + 1) + d
    assert D == 14

    B = 3
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
    ys = sampler.current_ys
    pool = sampler.current_coefficient_pool
    ids = sampler.current_coefficient_ids

    print(f"xs shape: {tuple(xs.shape)}  (expected ({B},{N},{D}))")
    print(f"ys shape: {tuple(ys.shape)}  (expected ({B},{N}))")
    print(f"pool shape: {tuple(pool.shape)}  (expected ({K},{d}))")
    print(f"ids shape: {tuple(ids.shape)}  (expected ({B},{N}))")
    print(f"pool norms: {pool.norm(dim=1).tolist()}  (expected ~1.0)")

    assert xs.shape == (B, N, D)
    assert ys.shape == (B, N)
    assert pool.shape == (K, d)
    assert ids.shape == (B, N)
    assert torch.allclose(pool.norm(dim=1), torch.ones(K), atol=1e-5)

    # Unpack one batch element and verify token layout.
    b = 0
    print("\nPool b_1, b_2:")
    for k in range(K):
        print(f"  b_{k} = {pool[k].tolist()}")

    print(f"\nSample b={b}, assignments per batch: {ids[b].tolist()}")

    fails = 0
    for i in range(N):
        token = xs[b, i]
        beta = pool[ids[b, i]]
        # Layout: T blocks of (x_t, y_t) of size (d+1), then a final x_{T+1} of size d
        for t in range(T):
            offset = t * (d + 1)
            x_t = token[offset : offset + d]
            y_t = token[offset + d].item()
            y_t_recomputed = torch.dot(beta, x_t).item()
            ok = abs(y_t - y_t_recomputed) < 1e-5
            if not ok:
                fails += 1
                print(f"  [batch {i} pair {t}] y_t={y_t:.4f} vs <beta,x>={y_t_recomputed:.4f}  MISMATCH")
        # Query: last d scalars
        x_query = token[T * (d + 1) : T * (d + 1) + d]
        y_query_target = ys[b, i].item()
        y_query_recomputed = torch.dot(beta, x_query).item()
        ok = abs(y_query_target - y_query_recomputed) < 1e-5
        marker = "OK" if ok else "MISMATCH"
        print(f"  batch {i} (beta_idx={ids[b, i].item()}): "
              f"target y_query={y_query_target:.4f}  recomputed=<beta,x_query>={y_query_recomputed:.4f}  [{marker}]")
        if not ok:
            fails += 1

    # Inspect raw layout of one token visually
    print(f"\nRaw token xs[{b},0]:")
    for t in range(T):
        offset = t * (d + 1)
        print(f"  slots {offset}..{offset+d-1} = x_{t+1}: {xs[b,0,offset:offset+d].tolist()}")
        print(f"  slot {offset+d} = y_{t+1}: {xs[b,0,offset+d].item():.4f}")
    print(f"  slots {T*(d+1)}..{D-1} = x_{T+1}: {xs[b,0,T*(d+1):D].tolist()}")
    print(f"  (held out) y_{T+1} target = {ys[b,0].item():.4f}")

    if fails == 0:
        print("\nALL CHECKS PASSED")
    else:
        print(f"\n{fails} MISMATCHES")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Causal baselines for mixture-of-linear-regression prompts."""

from __future__ import annotations

import numpy as np


def unpack_tokens(xs: np.ndarray, T: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack ``[x1,y1,...,xT,yT,x_query]`` tokens."""
    batch, tasks, width = xs.shape
    expected = T * (d + 1) + d
    if width != expected:
        raise ValueError(f"Token width is {width}; expected {expected}")
    pairs = xs[:, :, : T * (d + 1)].reshape(batch, tasks, T, d + 1)
    return pairs[..., :d], pairs[..., d], xs[:, :, T * (d + 1):]


def _ridge(X: np.ndarray, y: np.ndarray, regularization: float) -> np.ndarray:
    return np.linalg.solve(
        X.T @ X + regularization * np.eye(X.shape[1]), X.T @ y
    )


def ridge_current(X: np.ndarray, y: np.ndarray, query: np.ndarray,
                  regularization: float = 1.0) -> np.ndarray:
    batch, tasks, _, _ = X.shape
    predictions = np.empty((batch, tasks))
    for b in range(batch):
        for n in range(tasks):
            predictions[b, n] = query[b, n] @ _ridge(X[b, n], y[b, n], regularization)
    return predictions


def ridge_history(X: np.ndarray, y: np.ndarray, query: np.ndarray,
                  regularization: float = 1.0) -> np.ndarray:
    """Fit all completed tasks; position zero is undefined (NaN)."""
    batch, tasks, _, d = X.shape
    predictions = np.full((batch, tasks), np.nan)
    gram = np.zeros((batch, d, d))
    moment = np.zeros((batch, d))
    for n in range(tasks):
        if n:
            for b in range(batch):
                beta = np.linalg.solve(gram[b] + regularization * np.eye(d), moment[b])
                predictions[b, n] = query[b, n] @ beta
        gram += np.einsum("bti,btj->bij", X[:, n], X[:, n])
        moment += np.einsum("bti,bt->bi", X[:, n], y[:, n])
    return predictions


def known_pool_bayes(X: np.ndarray, y: np.ndarray, query: np.ndarray,
                     pool: np.ndarray, noise_std: float) -> np.ndarray:
    """Posterior predictive mean when the true coefficient pool is known."""
    variance = max(float(noise_std) ** 2, 1e-6)
    component_support = np.einsum("bntd,kd->bntk", X, pool)
    log_weights = -0.5 * np.square(y[..., None] - component_support).sum(axis=2) / variance
    log_weights -= log_weights.max(axis=-1, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=-1, keepdims=True)
    component_query = np.einsum("bnd,kd->bnk", query, pool)
    return (weights * component_query).sum(axis=-1)


def em_ridge_history(X: np.ndarray, y: np.ndarray, query: np.ndarray, *,
                     components: int, noise_std: float, regularization: float,
                     iterations: int = 20, initializations: int = 5,
                     seed: int = 0) -> np.ndarray:
    """Causal multi-start EM ridge fit to completed tasks only.

    Each task's support pairs share a latent component. At position ``n``, EM
    learns the component regressors from tasks ``0..n-1`` and uses the current
    task's support pairs only to infer its component responsibility.
    """
    batch, tasks, _, d = X.shape
    variance = max(float(noise_std) ** 2, 1e-6)
    predictions = np.full((batch, tasks), np.nan)
    rng = np.random.default_rng(seed)
    identity = np.eye(d)[None, None]

    for n in range(1, tasks):
        X_hist, y_hist = X[:, :n], y[:, :n]
        task_grams = np.einsum("bnti,bntj->bnij", X_hist, X_hist)
        task_moments = np.einsum("bnti,bnt->bni", X_hist, y_hist)
        best_score = np.full(batch, -np.inf)
        best_betas = np.zeros((batch, components, d))
        best_priors = np.full((batch, components), 1.0 / components)

        for _ in range(initializations):
            betas = rng.normal(scale=0.1, size=(batch, components, d))
            priors = np.full((batch, components), 1.0 / components)
            for _ in range(iterations):
                support_predictions = np.einsum("bnti,bki->bnkt", X_hist, betas)
                log_likelihood = -0.5 * np.square(
                    y_hist[:, :, None, :] - support_predictions
                ).sum(axis=-1) / variance
                log_likelihood += np.log(priors[:, None, :] + 1e-12)
                log_likelihood -= log_likelihood.max(axis=-1, keepdims=True)
                responsibilities = np.exp(log_likelihood)
                responsibilities /= responsibilities.sum(axis=-1, keepdims=True)
                priors = responsibilities.mean(axis=1)
                grams = np.einsum("bnk,bnij->bkij", responsibilities, task_grams)
                moments = np.einsum("bnk,bni->bki", responsibilities, task_moments)
                betas = np.linalg.solve(
                    grams + regularization * identity, moments[..., None]
                )[..., 0]

            support_predictions = np.einsum("bnti,bki->bnkt", X_hist, betas)
            component_ll = -0.5 * np.square(
                y_hist[:, :, None, :] - support_predictions
            ).sum(axis=-1) / variance
            component_ll += np.log(priors[:, None, :] + 1e-12)
            maximum = component_ll.max(axis=-1, keepdims=True)
            score = (maximum[..., 0] + np.log(
                np.exp(component_ll - maximum).sum(axis=-1)
            )).sum(axis=-1)
            improved = score > best_score
            best_score[improved] = score[improved]
            best_betas[improved] = betas[improved]
            best_priors[improved] = priors[improved]

        current_predictions = np.einsum("bti,bki->bkt", X[:, n], best_betas)
        current_ll = -0.5 * np.square(
            y[:, n, None, :] - current_predictions
        ).sum(axis=-1) / variance
        current_ll += np.log(best_priors + 1e-12)
        current_ll -= current_ll.max(axis=-1, keepdims=True)
        weights = np.exp(current_ll)
        weights /= weights.sum(axis=-1, keepdims=True)
        component_query = np.einsum("bi,bki->bk", query[:, n], best_betas)
        predictions[:, n] = (weights * component_query).sum(axis=-1)
    return predictions

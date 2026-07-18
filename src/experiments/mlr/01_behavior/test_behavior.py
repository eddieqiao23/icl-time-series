import unittest

import numpy as np

from baselines import em_ridge_history, known_pool_bayes, ridge_current, ridge_history, unpack_tokens


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.pool = np.array([[1.0, 0.0], [0.0, 1.0]])
        # B=1, N=2, T=2, d=2; each task reveals its coefficient exactly.
        self.xs = np.array([[[1, 0, 1, 0, 1, 0, 1, 0],
                             [1, 0, 0, 0, 1, 1, 1, 1]]], dtype=float)
        self.targets = np.array([[1.0, 1.0]])

    def test_unpack_layout(self):
        X, y, query = unpack_tokens(self.xs, T=2, d=2)
        self.assertEqual(X.shape, (1, 2, 2, 2))
        np.testing.assert_array_equal(y[0, 0], [1, 0])
        np.testing.assert_array_equal(query[0, 1], [1, 1])

    def test_known_pool_is_exact_for_noiseless_identifiable_tasks(self):
        X, y, query = unpack_tokens(self.xs, T=2, d=2)
        prediction = known_pool_bayes(X, y, query, self.pool, 0.0)
        np.testing.assert_allclose(prediction, self.targets, atol=1e-8)

    def test_ridge_shapes_and_causality(self):
        X, y, query = unpack_tokens(self.xs, T=2, d=2)
        self.assertEqual(ridge_current(X, y, query).shape, self.targets.shape)
        history = ridge_history(X, y, query)
        self.assertTrue(np.isnan(history[0, 0]))
        self.assertTrue(np.isfinite(history[0, 1]))

    def test_em_is_causal_and_finite_after_first_task(self):
        X, y, query = unpack_tokens(self.xs, T=2, d=2)
        prediction = em_ridge_history(
            X, y, query, components=2, noise_std=0.2,
            regularization=0.1, iterations=2, initializations=2, seed=3,
        )
        self.assertTrue(np.isnan(prediction[0, 0]))
        self.assertTrue(np.isfinite(prediction[0, 1]))


if __name__ == "__main__":
    unittest.main()

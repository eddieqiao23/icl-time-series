import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


class ARWarmupSampler(DataSampler):
    """
    AR(q) time series sampler for warmup experiments.
    Generates actual AR(q) sequences with known coefficients.
    """
    def __init__(self, n_dims, lag=3, base_sampler=None, noise_std=0.2, **kwargs):
        super().__init__(n_dims)
        self.lag = lag
        self.noise_std = noise_std
        # Use GaussianSampler as default base sampler if none provided
        # For AR tasks, always use 1-dimensional base sampler
        if base_sampler is None:
            self.base_sampler = GaussianSampler(1, **kwargs)  # Always 1D for AR
        else:
            self.base_sampler = base_sampler
        self.current_coefficients = None # Gets set in train.py
    
    def generate_bounded_coefficients(self, batch_size):
        """
        Generate AR coefficients that prevent explosive growth.
        Ensures the AR process remains bounded/stationary.
        
        Uses normalization by lag: x_t = (1/d) * sum(x_{t-i} * a_i) where a_i ~ N(0, 1)
        This is more natural and increases stability compared to scaling down variance.
        """
        # Generate coefficients from N(0, 1) and normalize by 1/lag
        coeffs = torch.randn(batch_size, self.lag) / self.lag
        return coeffs
    
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        Generate AR(q) sequences with known coefficients and extract lagged features.
        
        Args:
            n_points: Number of time points to generate
            b_size: Batch size
            n_dims_truncated: Target output dimension (should match model.n_dims)
            seeds: Random seeds for reproducibility
            
        Returns:
            xs: (b_size, n_points, n_dims_truncated) tensor of lagged features
        """        
        # Generate AR sequences using these coefficients
        T = n_points + self.lag
        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        ys_b = torch.zeros(b_size, n_points)  # Store actual noisy next values
        
        if seeds is not None:
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                # Generate initial lag values randomly
                z = torch.zeros(T)
                z[:self.lag] = torch.randn(self.lag, generator=generator)
                
                # Finish the remaining N-q values for AR(q) process
                # Might be slow to parallelize?
                for t in range(self.lag, T):
                    # x_t = (1/d) * sum(x_{t-i} * a_i) + ε_t, where ε_t ~ N(0, noise_std^2)
                    z[t] = sum(self.current_coefficients[i, j] * z[t-1-j] for j in range(self.lag))
                    z[t] += torch.randn(1, generator=generator).item() * self.noise_std
                
                # Create lagged features from this AR sequence
                lagged_features = self._create_lagged_features(z)
                # Pad or crop so that it reaches n_dims dimensions
                if self.lag <= self.n_dims:
                    xs_b[i, :, :self.lag] = lagged_features
                else:
                    xs_b[i] = lagged_features[:, :self.n_dims]
                
                # Store the actual next values (the noisy values, not predictions)
                # xs[i] contains lagged features for predicting z[lag+i]
                # so ys[i] should be the actual value z[lag+i]
                ys_b[i, :] = z[self.lag:self.lag+n_points]
        else:
            for i in range(b_size):
                # Generate initial lag values randomly
                z = torch.zeros(T)
                z[:self.lag] = torch.randn(self.lag)
                
                # Finish the remaining N-q values for AR(q) process
                # Might be slow to parallelize?
                for t in range(self.lag, T):
                    # x_t = (1/d) * sum(x_{t-i} * a_i) + ε_t, where ε_t ~ N(0, noise_std^2)
                    z[t] = sum(self.current_coefficients[i, j] * z[t-1-j] for j in range(self.lag))
                    z[t] += torch.randn(1).item() * self.noise_std
                
                # Create lagged features from this AR sequence
                lagged_features = self._create_lagged_features(z)
                # Pad or truncate to match n_dims
                if self.lag <= self.n_dims:
                    xs_b[i, :, :self.lag] = lagged_features
                else:
                    xs_b[i] = lagged_features[:, :self.n_dims]
                
                # Store the actual next values (the noisy values, not predictions)
                # xs[i] contains lagged features for predicting z[lag+i]
                # so ys[i] should be the actual value z[lag+i]
                ys_b[i, :] = z[self.lag:self.lag+n_points]
        
        if n_dims_truncated is not None:
            xs_b = xs_b[:, :, :n_dims_truncated]
        
        # Store ys_b in the sampler so it can be accessed
        self.current_ys = ys_b
        
        return xs_b
    
    def _create_lagged_features(self, z):
        """
        Create lagged features from a time series.
        
        Args:
            z: (T, 1) time series
            
        Returns:
            lagged: (n_points, lag) lagged features

        e.g. z is (19,) and lagged is (11, 8) if lag = 8
        """
        # Go from (T, 1) -> (T,)
        if z.dim() == 2:
            z = z[:, 0]
        
        # Create lagged indices
        t_idx = torch.arange(self.lag, z.shape[0])  # (n_points,)
        lags = torch.arange(1, self.lag + 1)  # (lag,)
        correct_idx = (t_idx[:, None] - lags[None, :])  # (n_points, lag)

        return z[correct_idx]


class ARMixtureSampler(ARWarmupSampler):
    """
    Mixture of AR(q) models. Fix a coefficient pool throughout training, so we are sampling
    coefficients from it during both training and testing.
    """
    def __init__(self, n_dims, lag=3, base_sampler=None, noise_std=0.2, num_mixture_models=5, **kwargs):
        super().__init__(n_dims, lag, base_sampler, noise_std, **kwargs)
        self.num_mixture_models = num_mixture_models
        self.coefficient_pool = self.generate_bounded_coefficients(num_mixture_models)
    
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        coeff_indices = torch.randint(0, self.num_mixture_models, (b_size,))
        self.current_coefficients = self.coefficient_pool[coeff_indices]
        self.current_coefficient_ids = coeff_indices
        return super().sample_xs(n_points, b_size, n_dims_truncated, seeds)


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "ar_warmup": ARWarmupSampler,
        "ar_mixture": ARMixtureSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

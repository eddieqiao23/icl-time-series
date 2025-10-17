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
    Generates lagged features for autoregressive modeling.
    """
    def __init__(self, n_dims, lag=3, base_sampler=None, **kwargs):
        super().__init__(n_dims)
        self.lag = lag
        # Use GaussianSampler as default base sampler if none provided
        # For AR tasks, always use 1-dimensional base sampler
        if base_sampler is None:
            self.base_sampler = GaussianSampler(1, **kwargs)  # Always 1D for AR
        else:
            self.base_sampler = base_sampler
    
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        Generate AR(q) lagged features.
        
        Args:
            n_points: Number of time points to generate
            b_size: Batch size
            n_dims_truncated: Target output dimension (should match model.n_dims)
            seeds: Random seeds for reproducibility
            
        Returns:
            xs: (b_size, n_points, n_dims_truncated) tensor of lagged features
        """
        T = n_points + self.lag
        
        # Generate base time series using the underlying sampler
        if seeds is not None:
            generator = torch.Generator()
            assert len(seeds) == b_size
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                z = torch.randn(T, 1, generator=generator)  # Always use 1D for AR
                lagged_features = self._create_lagged_features(z)
                # Pad so that it reaches n_dims dimensions
                if self.lag <= self.n_dims:
                    xs_b[i, :, :self.lag] = lagged_features
                    # Remaining dimensions are zero (already initialized)
                else:
                    xs_b[i] = lagged_features[:, :self.n_dims]
        else: # No seed provided
            z = self.base_sampler.sample_xs(T, b_size, 1, seeds)  # Always use 1D for AR
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            for i in range(b_size):
                lagged_features = self._create_lagged_features(z[i])
                # Pad or truncate to match n_dims
                if self.lag <= self.n_dims:
                    xs_b[i, :, :self.lag] = lagged_features
                    # Remaining dimensions are zero (already initialized)
                else:
                    xs_b[i] = lagged_features[:, :self.n_dims]
        
        # Apply truncation if specified
        if n_dims_truncated is not None:
            xs_b = xs_b[:, :, :n_dims_truncated]
        
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
        
        # print("z.shape: ", z.shape)
        # print("z: ", z)
        # print("z[correct_idx].shape: ", z[correct_idx].shape)
        # print("z[correct_idx]: ", z[correct_idx])
        return z[correct_idx]  # (n_points, lag)


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "ar_warmup": ARWarmupSampler,
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

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

        Uses L2 normalization to fix the total energy of coefficients.
        This provides better signal-to-noise ratio while maintaining stability.
        """
        # Generate coefficients from N(0, 1) and normalize L2 norm to 0.5
        coeffs = torch.randn(batch_size, self.lag)
        coeffs = coeffs / coeffs.norm(dim=1, keepdim=True) * 0.5
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
                    # x_t = (1/d) * sum(x_{t-i} * a_i) + ε_t, where eps_t ~ N(0, noise_std^2)
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

    Generates multi-run samples where each sample contains multiple AR sequences,
    each generated from a randomly selected model from the coefficient pool.
    """
    def __init__(self, n_dims, lag=3, base_sampler=None, noise_std=0.2, num_mixture_models=5, num_runs=3, use_gpu=True, device=None, **kwargs):
        super().__init__(n_dims, lag, base_sampler, noise_std, **kwargs)
        self.num_mixture_models = num_mixture_models
        self.num_runs = num_runs
        self.use_gpu = use_gpu # Generate everything at the start more quickly
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.coefficient_pool = self.generate_bounded_coefficients(num_mixture_models)

        # Move coefficient pool to GPU if using GPU sampling
        if self.use_gpu and self.device.type in ['cuda', 'mps']:
            self.coefficient_pool = self.coefficient_pool.to(self.device)

    def _generate_single_run(self, run_length, coefficients, seed=None):
        """
        Generate a single AR sequence.

        Args:
            run_length: Number of time points to generate
            coefficients: (lag,) tensor of AR coefficients
            seed: Optional random seed

        Returns:
            sequence: (run_length + 1,) tensor where sequence[i] is the value to predict sequence[i+1]
        """
        T = run_length + 1 + self.lag  # Need extra points for initialization and next value
        z = torch.zeros(T)

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            z[:self.lag] = torch.randn(self.lag, generator=generator)

            for t in range(self.lag, T):
                z[t] = sum(coefficients[j] * z[t-1-j] for j in range(self.lag))
                z[t] += torch.randn(1, generator=generator).item() * self.noise_std
        else:
            z[:self.lag] = torch.randn(self.lag)

            for t in range(self.lag, T):
                z[t] = sum(coefficients[j] * z[t-1-j] for j in range(self.lag))
                z[t] += torch.randn(1).item() * self.noise_std

        # Return sequence starting from the first predictable point
        return z[self.lag:self.lag + run_length + 1]

    def _sample_xs_gpu(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        GPU-parallelized AR mixture sampling. Within a batch, every sequence is generated
        in parallel.

        Args:
            n_points: Length of each run (run_length)
            b_size: Batch size
            n_dims_truncated: Should be 2*num_runs - 1
            seeds: Random seeds for reproducibility

        Returns:
            xs: (b_size, n_points, 2*num_runs - 1) tensor in progressive format
        """
        run_length = n_points
        expected_dims = 2 * self.num_runs - 1

        # n of the inputs, n-1 of the "answers"
        assert n_dims_truncated is not None and n_dims_truncated == expected_dims

        total_sequences = b_size * self.num_runs

        if seeds is not None:
            torch.manual_seed(seeds[0])

        # Generate coefficient pool with L2 normalization
        batch_coefficient_pool = torch.randn(self.num_mixture_models, self.lag, device=self.device)
        batch_coefficient_pool = batch_coefficient_pool / batch_coefficient_pool.norm(dim=1, keepdim=True) * 0.5

        coeff_indices = torch.randint(0, self.num_mixture_models, (total_sequences,), device=self.device)
        all_coefficients = batch_coefficient_pool[coeff_indices]

        self.current_coefficient_pool = batch_coefficient_pool.cpu()
        self.current_coefficient_ids = coeff_indices.cpu()

        T = run_length + 1 + self.lag
        z_batch = torch.zeros(total_sequences, T, device=self.device)

        if seeds is not None:
            for i in range(total_sequences):
                batch_idx = i // self.num_runs
                run_idx = i % self.num_runs
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(seeds[batch_idx]) + run_idx)
                z_batch[i, :self.lag] = torch.randn(self.lag, generator=generator, device=self.device)
        else:
            z_batch[:, :self.lag] = torch.randn(total_sequences, self.lag, device=self.device)

        if seeds is not None:
            noise_batch = torch.zeros(total_sequences, T - self.lag, device=self.device)
            for i in range(total_sequences):
                batch_idx = i // self.num_runs
                run_idx = i % self.num_runs
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(seeds[batch_idx]) + run_idx + 1000)
                noise_batch[i] = torch.randn(T - self.lag, generator=generator, device=self.device) * self.noise_std
        else:
            noise_batch = torch.randn(total_sequences, T - self.lag, device=self.device) * self.noise_std

        for t in range(self.lag, T):
            lagged_vals = torch.stack([z_batch[:, t-1-j] for j in range(self.lag)], dim=1)
            z_batch[:, t] = (all_coefficients * lagged_vals).sum(dim=1) + noise_batch[:, t - self.lag]

        sequences = z_batch[:, self.lag:self.lag + run_length + 1]

        xs_b = torch.zeros(b_size, run_length, expected_dims, device=self.device)
        ys_b = torch.zeros(b_size, run_length, device=self.device)

        for batch_idx in range(b_size):
            for run_idx in range(self.num_runs):
                seq_idx = batch_idx * self.num_runs + run_idx
                sequence = sequences[seq_idx]

                if run_idx < self.num_runs - 1:
                    val_col = 2 * run_idx
                    out_col = 2 * run_idx + 1
                    xs_b[batch_idx, :, val_col] = sequence[:run_length]
                    xs_b[batch_idx, :, out_col] = sequence[1:run_length+1]
                else:
                    val_col = 2 * run_idx
                    xs_b[batch_idx, :, val_col] = sequence[:run_length]
                    ys_b[batch_idx, :] = sequence[1:run_length+1]

        xs_b = xs_b.cpu()
        ys_b = ys_b.cpu()

        self.current_ys = ys_b

        return xs_b

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        Generate multi-run AR mixture samples.

        Args:
            n_points: Length of each run (run_length)
            b_size: Batch size
            n_dims_truncated: Should be 2*num_runs - 1
            seeds: Random seeds for reproducibility

        Returns:
            xs: (b_size, n_points, 2*num_runs - 1) tensor in progressive format
                Each row: [run1_val, run1_out, run2_val, run2_out, ..., runN_val]
        """
        # Use GPU-parallelized version if enabled and device supports it
        if self.use_gpu and self.device.type in ['cuda', 'mps']:
            return self._sample_xs_gpu(n_points, b_size, n_dims_truncated, seeds)

        run_length = n_points
        expected_dims = 2 * self.num_runs - 1

        if n_dims_truncated is not None and n_dims_truncated != expected_dims:
            print(f"Warning: n_dims_truncated={n_dims_truncated} but expected {expected_dims} for {self.num_runs} runs")

        # Generate coefficient pool with L2 normalization
        batch_coefficient_pool = torch.randn(self.num_mixture_models, self.lag)
        batch_coefficient_pool = batch_coefficient_pool / batch_coefficient_pool.norm(dim=1, keepdim=True) * 0.5
        self.current_coefficient_pool = batch_coefficient_pool

        xs_b = torch.zeros(b_size, run_length, expected_dims)
        ys_b = torch.zeros(b_size, run_length)
        coeff_indices = []

        for batch_idx in range(b_size):
            for run_idx in range(self.num_runs):
                coeff_idx = torch.randint(0, self.num_mixture_models, (1,)).item()
                coeff_indices.append(coeff_idx)
                coefficients = batch_coefficient_pool[coeff_idx]

                # Generate AR sequence for this run
                seed = seeds[batch_idx] + run_idx if seeds is not None else None
                sequence = self._generate_single_run(run_length, coefficients, seed)

                # Fill in the matrix columns
                # Format: [run1_val, run1_out, run2_val, run2_out, ..., runN_val]
                if run_idx < self.num_runs - 1:
                    # For runs 1 to N-1: both val and out columns
                    val_col = 2 * run_idx
                    out_col = 2 * run_idx + 1
                    xs_b[batch_idx, :, val_col] = sequence[:run_length]  # z0, z1, z2, ...
                    xs_b[batch_idx, :, out_col] = sequence[1:run_length+1]  # z1, z2, z3, ...
                else:
                    # For final run: only val column
                    val_col = 2 * run_idx
                    xs_b[batch_idx, :, val_col] = sequence[:run_length]  # v0, v1, v2, ...
                    ys_b[batch_idx, :] = sequence[1:run_length+1]  # v1, v2, v3, ... (targets)

        self.current_ys = ys_b
        self.current_coefficient_ids = torch.tensor(coeff_indices)

        return xs_b


class ARMixtureTransposedSampler(ARMixtureSampler):
    """
    Transposed version where columns become tokens.
    Token 0: [z_(1,0), z_(1,1), ..., z_(1,19)]
    Token 1: [z_(1,1), z_(1,2), ..., z_(1,20)]
    Returns xs: (batch, 2*num_runs-1, n_points)
    """
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_original = super().sample_xs(n_points, b_size, n_dims_truncated, seeds)
        xs_transposed = xs_original.transpose(1, 2)

        ys_vectors = torch.zeros(b_size, 2 * self.num_runs - 1, n_points)

        for run_idx in range(self.num_runs - 1):
            val_col = 2 * run_idx
            out_col = 2 * run_idx + 1
            ys_vectors[:, val_col, :] = xs_transposed[:, out_col, :]

        final_val_col = 2 * (self.num_runs - 1)
        ys_vectors[:, final_val_col, :] = self.current_ys

        self.current_ys_vectors = ys_vectors

        return xs_transposed


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "ar_warmup": ARWarmupSampler,
        "ar_mixture": ARMixtureSampler,
        "ar_mixture_transposed": ARMixtureTransposedSampler,
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

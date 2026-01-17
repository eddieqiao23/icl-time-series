"""
AR Coefficient Generation Methods

Implements 4 different approaches for generating AR coefficients:
1. L2 normalization (current baseline)
2. Scaled normal distribution
3. Scaled normal with stability resampling
4. Root-based generation from characteristic polynomial
"""

import torch
import numpy as np
from typing import Tuple, Optional


def generate_coeffs_l2(lag: int, num_models: int, l2_norm: float = 0.5) -> torch.Tensor:
    """
    L2 Normalization
    
    Sample from N(0,1) and normalize L2 norm to fixed value.
    
    Args:
        lag: AR order (number of coefficients)
        num_models: Number of coefficient vectors to generate
        l2_norm: Target L2 norm for each coefficient vector
        
    Returns:
        coeffs: (num_models, lag) tensor of coefficients
    """
    coeffs = torch.randn(num_models, lag)
    coeffs = coeffs / coeffs.norm(dim=1, keepdim=True) * l2_norm
    return coeffs


def generate_coeffs_scaled_normal(lag: int, num_models: int, scale_factor: float = 0.3) -> torch.Tensor:
    """
    Scaled Normal Distribution
    
    Sample from N(0, σ²) where σ = scale_factor / sqrt(lag).
    Intuition: As lag increases, individual coefficients should be smaller.
    
    Args:
        lag: AR order
        num_models: Number of coefficient vectors to generate
        scale_factor: Controls overall magnitude (higher = stronger signal)
        
    Returns:
        coeffs: (num_models, lag) tensor of coefficients
    """
    sigma = scale_factor / np.sqrt(lag)
    coeffs = torch.randn(num_models, lag) * sigma
    return coeffs


def is_stable(coeffs: torch.Tensor) -> bool:
    """
    Check if AR process with given coefficients is stable.
    
    Args:
        coeffs: (lag,) tensor of AR coefficients [c1, c2, ..., cp]
        
    Returns:
        True if stable, False otherwise
    """
    lag = len(coeffs)
    
    # Handle edge cases
    if lag == 0:
        return True
    if torch.any(torch.isnan(coeffs)) or torch.any(torch.isinf(coeffs)):
        return False
    
    poly_coeffs = np.concatenate([-coeffs.numpy()[::-1], [1.0]])
    
    try:
        # Check if all roots outside unit circle
        roots = np.roots(poly_coeffs)
        return np.all(np.abs(roots) > 1.0)
    except:
        # If root finding fails, assume unstable
        return False


def generate_coeffs_scaled_with_resampling(
    lag: int, 
    num_models: int, 
    scale_factor: float = 0.3,
    max_attempts: int = 100
) -> torch.Tensor:
    """
    Scaled Normal with Resampling
    
    Sample from N(0, σ²) and reject if unstable. Guarantees all coefficients
    produce stable AR processes.
    
    Args:
        lag: AR order
        num_models: Number of coefficient vectors to generate
        scale_factor: Controls overall magnitude
        max_attempts: Maximum resampling attempts per model
        
    Returns:
        coeffs: (num_models, lag) tensor of stable coefficients
    """
    sigma = scale_factor / np.sqrt(lag)
    coeffs_list = []
    rejection_counts = []
    
    for _ in range(num_models):
        for attempt in range(max_attempts):
            coeff = torch.randn(lag) * sigma
            
            if is_stable(coeff):
                coeffs_list.append(coeff)
                rejection_counts.append(attempt)
                break
        else:
            # If max attempts reached
            coeff = torch.tensor([0.5 ** (i + 1) for i in range(lag)], dtype=torch.float32)
            coeffs_list.append(coeff)
            rejection_counts.append(max_attempts)
    
    coeffs = torch.stack(coeffs_list)
    
    # avg_rejections = np.mean(rejection_counts)
    # rejection_rate = sum(1 for r in rejection_counts if r > 0) / len(rejection_counts)
    
    return coeffs


def generate_coeffs_from_roots(
    lag: int,
    num_models: int,
    radius_range: Tuple[float, float] = (0.7, 0.95)
) -> torch.Tensor:
    """
    Root-Based Generation
    
    Sample roots uniformly in the unit circle. Use these roots to compute
    the characteristic polynomial and extract the AR coefficients.
    
    Args:
        lag: AR order (number of roots/coefficients)
        num_models: Number of coefficient vectors to generate
        radius_range: (min, max) radii for roots 
        
    Returns:
        coeffs: (num_models, lag) tensor of stable coefficients
    """
    coeffs_list = []
    
    for _ in range(num_models):
        radii = torch.rand(lag) * (radius_range[1] - radius_range[0]) + radius_range[0]
        angles = torch.rand(lag) * 2 * np.pi
        roots = radii * torch.exp(1j * angles)
        
        # Roots must come in pairs. If odd, make one root real.
        if lag % 2 == 1:
            roots[0] = roots[0].real.type(torch.complex64)
            for i in range(1, lag, 2):
                roots[i+1] = torch.conj(roots[i])
        else:
            for i in range(0, lag, 2):
                roots[i+1] = torch.conj(roots[i])
        
        roots_np = roots.numpy()
        poly = np.poly(roots_np) # [1, a_1, a_2, ..., a_p]
        ar_coeffs = -torch.from_numpy(poly[1:].real).float()
        
        coeffs_list.append(ar_coeffs[:lag])
    
    return torch.stack(coeffs_list)


def compute_stability_rate(coeffs: torch.Tensor) -> float:
    """
    Compute percentage of coefficient vectors that produce stable AR processes.
    
    Args:
        coeffs: (num_models, lag) tensor of coefficients
        
    Returns:
        Stability rate as percentage (0-100)
    """
    num_stable = sum(is_stable(c) for c in coeffs)
    return 100.0 * num_stable / len(coeffs)


def simulate_ar_sequence(
    coeffs: torch.Tensor,
    noise_std: float,
    n_points: int,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Simulate an AR sequence with given coefficients.
    
    Args:
        coeffs: (lag,) tensor of AR coefficients
        noise_std: Standard deviation of noise term
        n_points: Number of time points to generate
        seed: Random seed for reproducibility
        
    Returns:
        sequence: (n_points,) tensor of AR values
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    lag = len(coeffs)
    T = n_points + lag
    
    # Initialize sequence
    z = torch.zeros(T)
    z[:lag] = torch.randn(lag)
    
    # Generate AR process
    for t in range(lag, T):
        z[t] = sum(coeffs[i] * z[t-1-i] for i in range(lag))
        z[t] += torch.randn(1).item() * noise_std
    
    return z[lag:]


def compute_snr(coeffs: torch.Tensor, noise_std: float, n_points: int = 1000) -> float:
    """
    Compute signal-to-noise ratio for an AR process.
    
    SNR = Var(signal) / Var(noise)
    
    Args:
        coeffs: (lag,) tensor of AR coefficients
        noise_std: Standard deviation of noise term
        n_points: Number of points to simulate for variance estimation
        
    Returns:
        SNR value
    """
    if not is_stable(coeffs):
        return np.inf
    
    lag = len(coeffs)
    
    # Same process with and without noise
    z_with_noise = simulate_ar_sequence(coeffs, noise_std, n_points)
    z_without_noise = torch.zeros(n_points + lag)
    z_without_noise[:lag] = z_with_noise[:lag].clone()  # Same initial conditions
    
    for t in range(lag, n_points + lag):
        z_without_noise[t] = sum(coeffs[i] * z_without_noise[t-1-i] for i in range(lag))
    
    # Variance of signal (deterministic part)
    signal_var = z_without_noise[lag:].var()
    noise_var = noise_std ** 2
    
    snr = signal_var / noise_var
    return snr.item()

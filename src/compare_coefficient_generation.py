"""
Compare AR Coefficient Generation Methods

Generates coefficients using different methods and prints metrics.
"""

import torch
import numpy as np
from typing import Dict

from coefficient_generators import (
    generate_coeffs_l2,
    generate_coeffs_scaled_normal,
    generate_coeffs_scaled_with_resampling,
    generate_coeffs_from_roots,
    compute_stability_rate,
    compute_snr,
    is_stable,
)


def generate_all_methods(lag: int, num_models: int) -> Dict[str, torch.Tensor]:
    """Generate coefficients using all methods."""
    methods = {}
    
    methods['L2_Norm'] = generate_coeffs_l2(lag, num_models, l2_norm=0.5)
    methods['Scaled_Normal'] = generate_coeffs_scaled_normal(lag, num_models, scale_factor=1.0)
    methods['Scaled_Resampled'] = generate_coeffs_scaled_with_resampling(lag, num_models, scale_factor=1.0)
    methods['Root_Based'] = generate_coeffs_from_roots(lag, num_models, radius_range=(0, 0.95))
    methods['Root_Based_High_SNR'] = generate_coeffs_from_roots(lag, num_models, radius_range=(0.7, 0.95))
    
    return methods


def analyze_methods(methods: Dict[str, torch.Tensor], noise_std: float = 0.2, n_points: int = 500):
    """Analyze and print metrics for all methods."""
    
    print("=" * 65)
    print(f"{'Method':<20} {'Stability':>10} {'SNR':>18}")
    print("=" * 65)
    
    for name, coeffs in methods.items():
        stability_rate = compute_stability_rate(coeffs)
        stable_coeffs = [c for c in coeffs if is_stable(c)]
        if stable_coeffs:
            snr_values = [compute_snr(c, noise_std, n_points) for c in stable_coeffs[:20]]
            snr_values = [s for s in snr_values if not np.isinf(s)]
            snr_mean = np.mean(snr_values) if snr_values else 0
            snr_std = np.std(snr_values) if snr_values else 0
            snr_str = f"{snr_mean:.2f} ± {snr_std:.2f}"
        else:
            snr_str = "N/A"
        
        print(f"{name:<20} {stability_rate:>9.1f}% {snr_str:>18}")
    
    print("=" * 65)


def main():
    lag = 15
    num_models = 50
    noise_std = 0.2
    
    print(f"Coefficient Generation Comparison")
    print(f"AR({lag}), {num_models} models, noise_std={noise_std}")
    print()
    
    methods = generate_all_methods(lag=lag, num_models=num_models)
    analyze_methods(methods, noise_std=noise_std)

if __name__ == "__main__":
    main()

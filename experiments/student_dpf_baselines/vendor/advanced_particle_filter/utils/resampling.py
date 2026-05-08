"""
Resampling algorithms for particle filters.
"""

import numpy as np
from numpy.random import Generator


def systematic_resample(weights: np.ndarray, rng: Generator) -> np.ndarray:
    """
    Systematic resampling.
    
    Deterministic spacing with single random offset. Low variance.
    
    Args:
        weights: [N] Normalized weights (must sum to 1)
        rng: NumPy random generator
        
    Returns:
        indices: [N] Resampled particle indices
    """
    N = len(weights)
    
    # Cumulative sum
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # Ensure exactly 1.0 to avoid numerical issues
    
    # Systematic positions
    u0 = rng.uniform(0.0, 1.0 / N)
    u = u0 + np.arange(N) / N
    
    # Find indices
    indices = np.searchsorted(cdf, u, side='left')
    indices = np.minimum(indices, N - 1)
    
    return indices


def stratified_resample(weights: np.ndarray, rng: Generator) -> np.ndarray:
    """
    Stratified resampling.
    
    Independent random draw within each stratum. Slightly higher variance
    than systematic but still good.
    
    Args:
        weights: [N] Normalized weights (must sum to 1)
        rng: NumPy random generator
        
    Returns:
        indices: [N] Resampled particle indices
    """
    N = len(weights)
    
    # Cumulative sum
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    
    # Stratified positions: uniform in each stratum [i/N, (i+1)/N)
    u = (np.arange(N) + rng.uniform(0.0, 1.0, N)) / N
    
    # Find indices
    indices = np.searchsorted(cdf, u, side='left')
    indices = np.minimum(indices, N - 1)
    
    return indices


def multinomial_resample(weights: np.ndarray, rng: Generator) -> np.ndarray:
    """
    Multinomial resampling.
    
    Standard resampling with replacement. Higher variance than systematic.
    
    Args:
        weights: [N] Normalized weights (must sum to 1)
        rng: NumPy random generator
        
    Returns:
        indices: [N] Resampled particle indices
    """
    N = len(weights)
    return rng.choice(N, size=N, replace=True, p=weights)


def residual_resample(weights: np.ndarray, rng: Generator) -> np.ndarray:
    """
    Residual resampling.
    
    Deterministic replication of floor(N * w_i), then multinomial on residuals.
    
    Args:
        weights: [N] Normalized weights (must sum to 1)
        rng: NumPy random generator
        
    Returns:
        indices: [N] Resampled particle indices
    """
    N = len(weights)
    
    # Deterministic part
    n_copies = np.floor(N * weights).astype(int)
    
    # Indices from deterministic part
    indices = np.repeat(np.arange(N), n_copies)
    
    # Residual weights
    n_residual = N - len(indices)
    if n_residual > 0:
        residual_weights = (N * weights - n_copies)
        residual_weights = residual_weights / residual_weights.sum()
        
        # Multinomial on residuals
        residual_indices = rng.choice(N, size=n_residual, replace=True, p=residual_weights)
        indices = np.concatenate([indices, residual_indices])
    
    return indices.astype(int)


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size (ESS).
    
    ESS = 1 / sum(w_i^2), where weights are normalized.
    
    Args:
        weights: [N] Normalized weights (must sum to 1)
        
    Returns:
        ESS value in [1, N]
    """
    return 1.0 / np.sum(weights ** 2)


def normalize_log_weights(log_weights: np.ndarray) -> tuple:
    """
    Normalize log weights to get normalized weights.
    
    Args:
        log_weights: [N] Unnormalized log weights
        
    Returns:
        weights: [N] Normalized weights (sum to 1)
        log_normalizer: Log of the normalizing constant
    """
    # Log-sum-exp for numerical stability
    max_log = np.max(log_weights)
    log_sum = max_log + np.log(np.sum(np.exp(log_weights - max_log)))
    
    # Normalized weights
    weights = np.exp(log_weights - log_sum)
    
    return weights, log_sum

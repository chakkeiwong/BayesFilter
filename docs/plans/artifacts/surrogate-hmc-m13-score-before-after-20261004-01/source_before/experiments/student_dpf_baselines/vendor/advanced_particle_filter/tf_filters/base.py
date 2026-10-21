"""
TensorFlow filter base classes and result containers.

Mirrors: filters/base.py (NumPy version)
"""

import tensorflow as tf
from typing import Optional


class TFFilterResult:
    """
    Container for TF filter outputs.

    All arrays are tf.Tensor. Provides .numpy() conversion for
    comparison against NumPy baselines.

    Attributes:
        means: [T+1, nx] Filtered state means
        covariances: [T+1, nx, nx] Filtered state covariances (optional)
        particles: [T+1, N, nx] Particle history (optional)
        weights: [T+1, N] Weight history (optional)
        ess: [T] Effective sample size (optional)
        log_likelihood: Total log marginal likelihood (scalar)
        log_likelihood_increments: [T] Per-step log likelihood (optional)
        resampled: [T] Boolean mask of resampling events (optional)
    """

    def __init__(
        self,
        means: tf.Tensor,
        covariances: Optional[tf.Tensor] = None,
        particles: Optional[tf.Tensor] = None,
        weights: Optional[tf.Tensor] = None,
        ess: Optional[tf.Tensor] = None,
        log_likelihood: Optional[tf.Tensor] = None,
        log_likelihood_increments: Optional[tf.Tensor] = None,
        resampled: Optional[tf.Tensor] = None,
    ):
        self.means = means
        self.covariances = covariances
        self.particles = particles
        self.weights = weights
        self.ess = ess
        self.log_likelihood = log_likelihood
        self.log_likelihood_increments = log_likelihood_increments
        self.resampled = resampled

    @property
    def T(self) -> int:
        """Number of time steps (observations)."""
        return self.means.shape[0] - 1

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.means.shape[1]

    def to_numpy(self):
        """Convert all tensors to numpy for comparison."""
        from advanced_particle_filter.filters.base import FilterResult
        import numpy as np

        def _np(t):
            return t.numpy() if t is not None else None

        return FilterResult(
            means=_np(self.means),
            covariances=_np(self.covariances),
            particles=_np(self.particles),
            weights=_np(self.weights),
            ess=_np(self.ess),
            log_likelihood=float(self.log_likelihood) if self.log_likelihood is not None else None,
            log_likelihood_increments=_np(self.log_likelihood_increments),
            resampled=_np(self.resampled),
        )

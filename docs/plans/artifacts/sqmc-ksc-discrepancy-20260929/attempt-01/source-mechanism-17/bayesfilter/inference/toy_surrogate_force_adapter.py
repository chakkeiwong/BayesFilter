"""Toy Potential Surrogate-Force Adapter for Mechanics Testing.

This module implements a simple quadratic potential for testing surrogate-force
HMC mechanics in isolation from filter complexity.

Mathematical Foundation
-----------------------
U(θ) = 0.5 * θᵀ Σ⁻¹ θ

where Σ = diag([1, 4, 9]) for anisotropy testing.

True posterior: N(0, Σ)

Surrogate-Force Construction
----------------------------
- Value (for MH acceptance): exact U(θ)
- Force (for leapfrog): damped gradient = damping_scale * ∇U(θ)

Per Corollary 5.2, the chain samples exp(-U) exactly regardless of force quality,
as long as:
1. Force is deterministic function of θ
2. Same U evaluated at trajectory endpoints
3. Proper MH correction applied

Purpose
-------
Phase 2 of ledh-surrogate-force-hmc-master-program-2026-09-04.md:
Verify surrogate-force mechanics work correctly before applying to LEDH filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf


@dataclass
class BatchValueScoreResult:
    """Result from log_prob_and_grad."""
    value: tf.Tensor  # [batch] log probability
    score: tf.Tensor  # [batch, P] gradient (force for HMC)


class ToyPotentialAdapter:
    """Simple quadratic potential for mechanics testing.

    U(θ) = 0.5 * θᵀ Σ⁻¹ θ
    ∇U(θ) = Σ⁻¹ θ

    Parameters
    ----------
    sigma_inv : np.ndarray or tf.Tensor
        Inverse covariance matrix [P, P]
    damping_scale : float, default 1.0
        Scaling factor for gradient. 1.0 = exact, <1.0 = damped
    dtype : tf.DType, default tf.float64
        Tensor dtype for computations
    """

    def __init__(
        self,
        sigma_inv: np.ndarray | tf.Tensor,
        damping_scale: float = 1.0,
        dtype: tf.DType = tf.float64,
    ):
        self.sigma_inv = tf.constant(sigma_inv, dtype=dtype)
        self.damping_scale = tf.constant(damping_scale, dtype=dtype)
        self.dtype = dtype

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute log probability and gradient.

        Parameters
        ----------
        theta : tf.Tensor, shape [P] or [batch, P]
            Parameter values

        Returns
        -------
        value : tf.Tensor, shape [] or [batch]
            Log probability = -U(θ)
        grad : tf.Tensor, shape [P] or [batch, P]
            Gradient = damping_scale * (-∇U(θ))
        """
        theta = tf.cast(theta, self.dtype)

        # Handle both scalar and batch inputs
        is_scalar = theta.shape.rank == 1
        if is_scalar:
            theta = theta[tf.newaxis, :]  # [P] -> [1, P]

        # U(θ) = 0.5 * θᵀ Σ⁻¹ θ
        # Vectorized: [batch, P] @ [P, P] @ [P, batch] -> [batch, batch] diag -> [batch]
        quadratic = tf.einsum("bp,pq,bq->b", theta, self.sigma_inv, theta)
        value = -0.5 * quadratic  # log prob = -U

        # ∇U(θ) = Σ⁻¹ θ
        # Vectorized: [batch, P] @ [P, P] -> [batch, P]
        grad_u = tf.einsum("bp,pq->bq", theta, self.sigma_inv)
        grad = self.damping_scale * (-grad_u)  # gradient of log prob = -∇U

        if is_scalar:
            value = tf.squeeze(value, axis=0)  # [1] -> []
            grad = tf.squeeze(grad, axis=0)    # [1, P] -> [P]

        return value, grad


class DualAdapterSurrogateForce:
    """Dual-adapter for surrogate-force HMC on toy potential.

    This adapter returns:
    - Exact value (from exact_adapter) for MH acceptance
    - Damped force (from damped_adapter) for leapfrog integration

    Parameters
    ----------
    sigma_inv : np.ndarray
        Inverse covariance matrix [P, P]
    damping_scale : float
        Damping scale for force (1.0 = exact, 0.1 = heavily damped)
    dtype : tf.DType, default tf.float64
        Tensor dtype

    Example
    -------
    >>> sigma_inv = np.diag([1.0, 0.25, 1.0/9.0])  # Σ = diag([1, 4, 9])
    >>> adapter = DualAdapterSurrogateForce(sigma_inv, damping_scale=0.1)
    >>> result = adapter.log_prob_and_grad(theta)
    >>> # result.value is exact, result.score is damped by 0.1
    """

    def __init__(
        self,
        sigma_inv: np.ndarray,
        damping_scale: float,
        dtype: tf.DType = tf.float64,
    ):
        self.exact_adapter = ToyPotentialAdapter(sigma_inv, damping_scale=1.0, dtype=dtype)
        self.damped_adapter = ToyPotentialAdapter(sigma_inv, damping_scale=damping_scale, dtype=dtype)
        self.dtype = dtype

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute exact value and damped force.

        Parameters
        ----------
        theta : tf.Tensor, shape [P] or [batch, P]
            Parameter values

        Returns
        -------
        value : tf.Tensor, shape [] or [batch]
            Exact log probability
        force : tf.Tensor, shape [P] or [batch, P]
            Damped gradient
        """
        value, _ = self.exact_adapter.log_prob_and_grad(theta)
        _, force = self.damped_adapter.log_prob_and_grad(theta)

        return value, force


__all__ = [
    "ToyPotentialAdapter",
    "DualAdapterSurrogateForce",
    "BatchValueScoreResult",
]

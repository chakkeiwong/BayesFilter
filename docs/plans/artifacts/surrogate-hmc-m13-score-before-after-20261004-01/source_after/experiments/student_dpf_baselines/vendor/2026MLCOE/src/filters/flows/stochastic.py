"""
Stochastic Flows Module
=======================
This module implements the Stochastic Particle Flow (SPF) framework based on
Theorem 2.1. It is primarily used for systems where numerical stiffness
validation is required throughout the homotopy migration.

Theoretical Foundation:
-----------------------
The flow represents a stochastic differential equation (SDE) where the
drift and diffusion are designed to transform the prior density into the
posterior. By monitoring the condition number of the Hessian matrix (M),
this module provides a diagnostic tool for assessing the stability of the
homotopy path under small measurement noise (High-Stiffness).

Author: Joowon Lee
Date: 2026-02-26
"""

import tensorflow as tf
import numpy as np

# Use high-precision for stiffness validation and spectral analysis
DTYPE = tf.float64
JITTER = 1e-6

class StochasticParticleFlow:
    """
    Implementation of Theorem 2.1 Stochastic Particle Flow.

    This class enables schedule-based particle migration while providing
    real-time monitoring of the Hessian's spectral properties. It is the
    preferred choice for validating if a given homotopy schedule (beta)
    successfully navigates ill-conditioned regions of the state space.
    """

    def __init__(self, model, n_steps=200):
        """
        Initializes the SPF engine with stiffness monitoring capabilities.

        Detailed Parameter Description:
        -------------------------------
        :param model: The State-Space Model (SSM) instance. The SPF engine
            requires the model to expose `P_prior` (prediction covariance),
            `R_inv_filter` (measurement precision), and a ground truth
            reference `x_truth` for stiffness validation in research settings.
        :param n_steps (int): The number of integration steps for the SDE.
            Because SPF often involves monitoring condition numbers, a
            finer discretization (default 200) is used compared to standard
            deterministic flows.

        Internal State:
        ---------------
        - self.Q_fixed: A small, fixed diffusion matrix (1e-8) that acts as
          the stochastic component of the flow, preventing the ensemble
          from collapsing into a single point in the absence of noise.
        """
        self.m = model
        self.n_steps = n_steps
        self.dim = model.state_dim
        self.Q_fixed = tf.eye(self.dim, dtype=DTYPE) * 1e-8

    @tf.function
    def compute_flow_with_schedule(self, z, X_init, beta_val, dot_beta_val):
        """
        Migrates the ensemble according to Theorem 2.1 velocity fields.

        This function calculates two separate gain matrices, K1 and K2,
        which decompose the drift into a component following the prior
        gradient and a component following the likelihood gradient.

        Detailed Logic:
        ---------------
        1. **Hessian Construction (M)**: At each step, we calculate
           M = P_inv + beta * H^T * R_inv * H. This matrix represents the
           local curvature of the log-posterior.
        2. **Spectral Analysis**: We perform a Singular Value Decomposition (SVD)
           on M to extract the condition number (max_eigen / min_eigen). This
           is recorded in `cond_history` for post-run stiffness evaluation.
        3. **Drift Integration**: The particles are shifted by the weighted
           combination of prior and likelihood gradients, scaled by the
           temporal derivative of the schedule (dot_beta).

        Parameters:
            z (tf.Tensor): Current observation [Obs_Dim, 1].
            X_init (tf.Tensor): Initial ensemble [N, Dim].
            beta_val (tf.Tensor): Pre-computed homotopy schedule array.
            dot_beta_val (tf.Tensor): Pre-computed time-derivative of beta.

        Returns:
            tuple: (X_final, cond_history_tensor)
        """
        X = tf.cast(X_init, DTYPE)
        dt = tf.cast(1.0 / self.n_steps, DTYPE)
        cond_history = tf.TensorArray(DTYPE, size=self.n_steps)

        P_inv = tf.linalg.inv(self.m.P_prior)
        R_inv = self.m.R_inv_filter

        # Precision-safe constants
        JIT = tf.cast(JITTER, DTYPE)

        for k in range(self.n_steps):
            beta = tf.cast(beta_val[k], DTYPE)
            dot_beta = tf.cast(dot_beta_val[k], DTYPE)

            # 1. Measurement Jacobian evaluation at the reference state
            H = self.m.jacobian_h(self.m.x_truth)

            # 2. M(lambda) calculation (Equation 22 from the project specs)
            M = P_inv + beta * (tf.transpose(H) @ R_inv @ H)

            # 3. Spectral Norm Condition Number Calculation
            # SVD ensures robust eigenvalue extraction even near singular regions.
            s = tf.linalg.svd(M, compute_uv=False)
            cond_num = tf.reduce_max(s) / tf.reduce_min(s)
            cond_history = cond_history.write(k, cond_num)

            # 4. Gain Matrices K1 and K2 (Theorem 2.1)
            M_inv = tf.linalg.inv(M + tf.eye(self.dim, dtype=DTYPE) * JIT)
            K2 = -dot_beta * M_inv
            K1 = 0.5 * self.Q_fixed + 0.5 * dot_beta * M_inv @ (tf.transpose(H) @ R_inv @ H) @ M_inv

            # 5. Local Gradient Calculation
            # innov = [h(x) - z]
            innov = tf.reshape(z, [1, -1]) - self.m.h_func(X)
            grad_h = tf.matmul(innov, R_inv)
            # grad_p = gradient of the log-prior
            grad_p = -tf.matmul(X - self.m.x_prior, P_inv) + beta * grad_h

            # 6. Drift Combination: Combining Prior and Measurement influences
            drift = tf.matmul(grad_p, K1, transpose_b=True) + tf.matmul(grad_h, K2, transpose_b=True)

            # Final particle update via Euler-Maruyama (deterministic part)
            X = X + (drift * dt)

        return X, cond_history.stack()

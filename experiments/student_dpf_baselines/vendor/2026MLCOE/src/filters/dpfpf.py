"""
Differentiable Particle Flow Particle Filter (dPFPF)
====================================================
Implements a modular, differentiable Sequential Monte Carlo (SMC) pipeline.
This filter serves as the likelihood engine for gradient-based parameter
inference methods (e.g., PHMC).

By utilizing deterministic flows and entropy-regularized optimal transport
for resampling, this implementation ensures that the marginal likelihood
estimate is a differentiable function of the model parameters (theta).

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Modular imports from your project structure
from .classical import DTYPE
from src.filters.flows.EDH import EDHSolver
from src.filters.resampling.optimal_transport import SinkhornResampler
from src.filters.resampling.soft import soft_resample

tfd = tfp.distributions

class DifferentiablePFPF:
    """
    Modular Differentiable Particle Flow Particle Filter.

    This class orchestrates the interaction between a state-space model,
    a homotopy flow solver (e.g., EDH), and a differentiable resampler
    (e.g., Sinkhorn).
    """

    def __init__(self, model, solver=None, resampler=None, N=30, n_steps=20):
        """
        Initializes the dPFPF engine.

        Args:
            model: The State-Space Model (SSM) containing propagation
                   and observation logic.
            solver: The ODE solver used to compute particle drift
                    and divergence (Defaults to EDHSolver).
            resampler: The strategy used to recombine particles
                       differentiably (Defaults to SinkhornResampler).
            N (int): Number of particles in the ensemble.
            n_steps (int): Number of integration steps for the homotopy flow.
        """
        self.m = model
        self.N = N
        self.n_steps = n_steps

        # Dependency Injection: Allows swapping solvers/resamplers without
        # modifying the filter core.
        self.solver = solver if solver else EDHSolver()
        self.resampler = resampler if resampler else SinkhornResampler(epsilon=0.1)

        # Homotopy step-size schedule:
        # Uses a log-linear progression to handle the 'stiffness' of the flow
        # as lambda approaches 1 (the posterior).
        eps_np = np.logspace(-2, 0, n_steps)
        self.eps = tf.constant(eps_np / eps_np.sum(), dtype=DTYPE)

    @tf.function
    def step(self, z, X_prev, W_prev, P_prev):
        """
        Performs a single recursive filtering step (t -> t+1).

        Args:
            z: The current observation tensor.
            X_prev: Particle ensemble from the previous timestep [N, Dim].
            W_prev: Normalized weights from the previous timestep [N].
            P_prev: Estimated state covariance from the previous timestep.

        Returns:
            X_final: Resampled particle ensemble.
            W_final: Reset (uniform) particle weights.
            new_P: Updated state covariance for the next timestep.
        """

        # 1. Prediction (State Transition)
        # -------------------------------
        # Propagate particles through the model dynamics and add process noise.
        # This represents the prior p(x_t | y_{1:t-1}).
        noise_dist = tfd.Normal(tf.cast(0, DTYPE), tf.sqrt(self.m.sigma_v_sq))
        eta = self.m.propagate(X_prev) + noise_dist.sample((self.N, 1))

        # Initialize tracking for volume change (Log-Determinant of the flow Jacobian).
        log_det_total = tf.zeros(self.N, dtype=DTYPE)
        beta_vals = tf.cumsum(self.eps)

        # 2. Modular Homotopy Flow Integration
        # ------------------------------------
        # Gradually migrate particles from the prior to the posterior.
        # Instead of discrete weighting, we solve an ODE that 'pushes' particles
        # toward high-likelihood regions.
        for k in range(self.n_steps):
            dl, lam = self.eps[k], beta_vals[k]

            # The solver computes the 'drift' (velocity field) and the 'div' (divergence).
            drift, div = self.solver(z, eta, P_prev, lam, self.m)

            # Update particle positions (Euler integration step).
            eta += dl * drift

            # Accumulate divergence to adjust weights for the volume change
            # induced by the flow.
            log_det_total += dl * div

        # 3. Importance Weighting
        # -----------------------
        # Compute likelihood for the flow-transformed particles.
        innov_f = tf.cast(z, DTYPE) - self.m.h_func(eta)
        log_lik = -0.5 * tf.reduce_sum(innov_f @ self.m.R_inv_filter * innov_f, axis=1)

        # Combine prior weights, current log-likelihood, and flow log-determinants.
        # Softmax ensures weights sum to 1 while maintaining differentiability.
        W_new = tf.nn.softmax(tf.math.log(W_prev + 1e-30) + log_lik + log_det_total)

        # 4. Modular Resampling
        # ---------------------
        # Execute differentiable resampling to combat particle degeneracy.
        # Using Sinkhorn-EOT (Entropy-regularized Optimal Transport) allows
        # gradients to pass through the resampling step.
        lw = tf.math.log(W_new + 1e-30)
        X_final, _ = self.resampler.resample(eta, lw)

        # Post-resampling: Weights are reset to uniform.
        W_final = tf.ones(self.N, dtype=DTYPE) / tf.cast(self.N, DTYPE)

        # Update empirical covariance for use in the flow solver of the next timestep.
        # A small diagonal jitter (1e-4) is added to ensure positive-definiteness.
        new_P = tfp.stats.covariance(X_final) + tf.eye(self.m.state_dim, dtype=DTYPE)*1e-4

        return X_final, W_final, new_P

"""
Stochastic Flow Filters Module
==============================
This module implements particle flow-based filters for high-dimensional and
non-linear state estimation. It specifically focuses on Stochastic Particle
Flow (SPF) and Localized Exact Daum-Huang (LEDH) filters utilizing optimized
integration schedules based on the Dai (2022) formulation.

Unlike traditional Sequential Monte Carlo (SMC), these methods transport
particles from the prior to the posterior via a continuous-time flow,
significantly mitigating particle impoverishment and the curse of
dimensionality in complex systems.

Author: Joowon Lee
Date: 2026-02-26
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Preserve global precision and jitter settings for numerical consistency
DTYPE = tf.float64
JITTER = 1e-6
tfd = tfp.distributions


class SPF:
    def __init__(self, model, N=100, n_steps=200):
        self.m = model
        self.N = N
        self.n_steps = n_steps
        self.dim = model.state_dim
        self.Q_fixed = tf.eye(self.dim, dtype=DTYPE) * 1e-8

    @tf.function
    def step_with_schedule(self, z, X_init, beta_val, dot_beta_val):
        X = tf.cast(X_init, DTYPE)
        dt = tf.cast(1.0 / self.n_steps, DTYPE)
        cond_history = tf.TensorArray(DTYPE, size=self.n_steps)
        P_inv = tf.linalg.inv(self.m.P_prior)
        R_inv = self.m.R_inv_filter

        # High-precision constants
        ONE, TWO, EPS, JIT = [tf.cast(v, DTYPE) for v in [1.0, 2.0, 1e-8, 1e-6]]

        for k in range(self.n_steps):
            beta, dot_beta = tf.cast(beta_val[k], DTYPE), tf.cast(dot_beta_val[k], DTYPE)

            # 1. Correct Jacobian at the truth
            H = self.m.jacobian_h(self.m.x_truth)

            # 2. Correct M(lambda) calculation per Equation 22
            # M = -grad_grad_log_p0 - beta * grad_grad_log_h
            # For Gaussian: -grad_grad_log_p0 = P_prior^-1
            # -grad_grad_log_h = H^T R^-1 H
            M = P_inv + beta * (tf.transpose(H) @ R_inv @ H)

            # 3. Use the Spectral Norm (L2) for the condition number calculation
            # s is already sorted from svd
            s = tf.linalg.svd(M, compute_uv=False)
            cond_history = cond_history.write(k, tf.reduce_max(s) / tf.reduce_min(s))
            # Theorem 2.1 Drift
            M_inv = tf.linalg.inv(M + tf.eye(self.dim, dtype=DTYPE) * JIT)
            K2 = -dot_beta * M_inv
            K1 = 0.5 * self.Q_fixed + 0.5 * dot_beta * M_inv @ (tf.transpose(H) @ R_inv @ H) @ M_inv

            innov = tf.reshape(z, [1, -1]) - self.m.h_func(X)
            grad_h = tf.matmul(innov, R_inv)
            grad_p = -tf.matmul(X - self.m.x_prior, P_inv) + beta * grad_h
            drift = tf.matmul(grad_p, K1, transpose_b=True) + tf.matmul(grad_h, K2, transpose_b=True)

            X = X + (drift * dt) # Deterministic drift for stiffness validation

        return X, cond_history.stack()



class LEDH_OPT:
    def __init__(self, model, n_steps=100, mu=0.2):
        self.m = model
        self.n_steps = n_steps
        self.mu = mu
        self.lambdas = np.linspace(0, 1, n_steps)
        self.dt = 1.0 / (n_steps - 1)

    def solve_dai_schedule(self, x_pred_mean):
        """Solves the Dai22 BVP for the current state geometry."""
        P_inv = tf.linalg.inv(self.m.P_prior)
        H = self.m.jacobian_h(x_pred_mean)
        R_inv = self.m.R_inv_filter
        H_hess = tf.transpose(H) @ R_inv @ H

        def get_accel(b_val):
            b = tf.cast(tf.clip_by_value(b_val, 0.0, 1.1), DTYPE)
            M = P_inv + b * H_hess
            M_inv = tf.linalg.inv(M)
            # d_kappa/d_beta for nuclear norm
            t1 = tf.linalg.trace(H_hess) * tf.linalg.trace(M_inv)
            t2 = tf.linalg.trace(M) * tf.linalg.trace(M_inv @ H_hess @ M_inv)
            return -self.mu * (t1 + t2)

        def simulate(v0):
            beta = np.zeros(self.n_steps)
            v = v0
            for i in range(self.n_steps - 1):
                beta[i+1] = beta[i] + v * self.dt
                v += get_accel(beta[i+1]).numpy() * self.dt
            return beta

        # Shooting method
        low, high = 1.0, 30.0
        for _ in range(20):
            v_mid = (low + high) / 2
            if simulate(v_mid)[-1] < 1.0: low = v_mid
            else: high = v_mid

        beta_opt = simulate(low)
        dot_beta_opt = np.gradient(beta_opt, self.lambdas)
        return tf.cast(beta_opt, DTYPE), tf.cast(dot_beta_opt, DTYPE)

    def filter_step(self, z, x_particles):
        """Performs one step of LEDH with optimized Dai schedule."""
        N = tf.shape(x_particles)[0]
        x_pred_mean = tf.reduce_mean(x_particles, axis=0)

        # 1. Compute the optimized schedule for this specific observation
        beta_opt, dot_beta_opt = self.solve_dai_schedule(x_pred_mean)

        # 2. Particle Flow with importance weight tracking
        log_det_jac = tf.zeros(N, dtype=DTYPE)
        x = x_particles

        for k in range(self.n_steps - 1):
            beta = beta_opt[k]
            # LEDH f computation
            # Note: We scale the drift by dot_beta to follow the Dai schedule
            f, grad_f = self.compute_ledh_drift(x, z, beta)

            # Update particles (Deterministic part of the flow)
            x = x + f * dot_beta_opt[k] * self.dt

            # Update log-determinant of Jacobian for weight update
            log_det_jac += tf.linalg.trace(grad_f) * dot_beta_opt[k] * self.dt

        return x, log_det_jac

    def compute_ledh_drift(self, x, z, beta):
        """Standard LEDH Drift logic using Local Exact Daum-Huang."""
        # This mirrors the Exact Flow solution from Daum (10)
        # Applied locally at each particle as suggested by Li (17)
        P = self.m.P_prior
        H = self.m.jacobian_h(tf.reduce_mean(x, axis=0)) # Local approximation
        R = self.m.R_filter

        # Solve for f: A f + b = 0 where A is the flow Jacobian
        # For simplicity in this example, we use the global version
        # but apply it to individual particles.
        S = H @ P @ tf.transpose(H) + R
        K = P @ tf.transpose(H) @ tf.linalg.inv(beta * H @ P @ tf.transpose(H) + R)

        # Drift calculation
        f = -K @ (H @ tf.transpose(x) - tf.reshape(z, (-1, 1)))
        grad_f = -K @ H

        return tf.transpose(f), grad_f

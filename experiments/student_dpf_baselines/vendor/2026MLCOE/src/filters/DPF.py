"""
Differentiable Particle Filter (DPF) & Modular Resampling Framework
==================================================================
This module implements a framework for differentiable Sequential Monte Carlo (SMC)
methods. It decouples the resampling logic from the filtering step, providing
specialized classes for Neural (Transformer-based), Soft, and Optimal Transport
(Sinkhorn) resampling.

Architecture:
1. Resamplers: Independent modules that transform a weighted particle set into
   an unweighted (or re-weighted) set with higher effective sample size.
2. DPF: The core filtering engine that orchestrates transition, observation,
   and conditional resampling steps.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf

# FIX: Import the global DTYPE for M1 Ultra precision matching
from .classical import DTYPE
from .resampling.soft import SoftResampler
from .resampling.optimal_transport import SinkhornResampler
from .resampling.transformer import TransformerResampler


# =============================================================================
# 3. DIFFERENTIABLE PARTICLE FILTER
# =============================================================================

class DPF(tf.Module):
    """
    Differentiable Particle Filter (DPF) Core
    -----------------------------------------
    Orchestrates filtering using modular resampling strategies.
    """
    def __init__(self, transition_fn, observation_fn, num_particles):
        super().__init__()
        self.transition_fn, self.observation_fn, self.N = transition_fn, observation_fn, num_particles
        self.transformer = TransformerResampler(num_particles)

        # Note: Sinkhorn initialization takes epsilon and n_iter directly.
        self.sinkhorn = SinkhornResampler(epsilon=0.05, n_iter=10)
        self.soft = SoftResampler(num_particles)

    def filter_step(self, x_p, h_p, lw_p, y, method='transformer', **kwargs):
        """
        Executes one step of the particle filter.
        """
        noise = tf.random.normal(tf.shape(x_p), dtype=DTYPE)
        x_c, h_c = self.transition_fn(x_p, h_p, noise)

        log_lik = tf.cast(self.observation_fn(x_c, y), DTYPE)
        lw_c = tf.nn.log_softmax(lw_p + log_lik)

        ess = 1.0 / (tf.reduce_sum(tf.square(tf.exp(lw_c))) + 1e-12)

        if ess < tf.cast(self.N, DTYPE) / 2.0:
            if method == 'transformer':
                return self.transformer(x_c, h_c, tf.exp(lw_c))
            elif method in ['soft', 'soft_orig']:
                return self.soft(x_c, h_c, tf.exp(lw_c), alpha=kwargs.get('alpha'))
            elif method == 'sinkhorn':
                # FIX: Call the .resample() method instead of __call__
                x_res, h_res = self.sinkhorn.resample(x_c, lw_c, h=h_c)
                # DPF expects a 3-tuple return: (particles, hidden_states, reset_weights)
                lw_out = tf.fill((self.N,), -tf.math.log(tf.cast(self.N, DTYPE)))
                return x_res, h_res, lw_out

        return x_c, h_c, lw_c

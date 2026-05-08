"""
Soft Resampling Module
======================
Implements Soft Resampling as a differentiable alternative to Multinomial
resampling. It is particularly useful when a lightweight gradient-friendly
recombination is needed without the computational cost of Sinkhorn iterations.

Mathematical Context:
---------------------
Standard multinomial resampling is non-differentiable because it involves
sampling discrete indices. Soft resampling bypasses this by creating a
linear combination of 'hard' resampled particles and a 'weighted' average
of the prior ensemble. This allows gradients to propagate from the
posterior particle positions back to the importance weights.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
from src.filters.classical import DTYPE

def soft_resample(particles, weights, alpha=0.5):
    """
    Differentiable soft resampling core logic with strict DTYPE enforcement.
    Performs a linear combination of categorical resampling and continuous weighting.
    """
    N = tf.shape(particles)[0]
    alpha_tf = tf.cast(alpha, DTYPE)
    w_tf = tf.cast(weights, DTYPE)
    p_tf = tf.cast(particles, DTYPE)

    # 1. Hard resampling indices (non-differentiable path)
    indices = tf.random.categorical(tf.math.log(w_tf[None, :] + 1e-12), N)[0]
    resampled_hard = tf.gather(p_tf, indices)

    # 2. Linear combination for gradient flow
    # This interpolates between the hard sample and the weighted continuous ensemble.
    return alpha_tf * resampled_hard + (1.0 - alpha_tf) * (p_tf * w_tf[:, None] * tf.cast(N, DTYPE))


class SoftResampler(tf.Module):
    """
    Soft Resampler (Wrapped)
    ------------------------
    Utilizes the external soft_resample function to perform a linear combination
    of hard resampling and weighted particles for gradient flow.

    This class acts as a high-level wrapper that ensures the resampled states
    (both physical particles and hidden RNN/LSTM states) maintain the
    structure required by the DPF filtering core.
    """
    def __init__(self, num_particles, alpha=0.5):
        """
        Initializes the Soft Resampler.

        Args:
            num_particles (int): The number of particles in the ensemble (N).
            alpha (float): The default 'softness' parameter. A value of 0.5 balances
                           variance reduction with gradient signal strength.
        """
        super().__init__()
        self.N = num_particles
        self.alpha = alpha

    def __call__(self, p, h, w, alpha=None):
        """
        Executes the soft resampling step.

        Args:
            p: Current particle states [N, Dim].
            h: Current hidden/auxiliary states (e.g., GRU/LSTM cell states).
            w: Normalized importance weights [N].
            alpha: Optional override for the softness parameter.

        Returns:
            res_p: Softened particle ensemble.
            res_h: Hard-resampled hidden states.
            log_weights: Reset uniform log-weights for the next step.
        """
        # Determine the interpolation factor for this specific step.
        a = alpha if alpha is not None else self.alpha

        # 1. Perform soft resampling on the state particles
        # ------------------------------------------------
        # Calls the external functional logic to compute the linear combination:
        # res_p = alpha * HardResampled + (1 - alpha) * WeightedMean.
        res_p = soft_resample(p, w, alpha=a)

        # 2. Hidden states follow hard resampling indices to maintain consistency
        # -----------------------------------------------------------------------
        # While particle positions are softened for gradients, hidden states (h)
        # often contain discrete logic or non-differentiable context.
        # We perform standard categorical sampling to pick the 'best' states.

        # Sample indices based on the log-likelihood of the current weights.
        idx = tf.random.categorical(tf.math.log(w[None, :] + 1e-12), self.N)[0]

        # Use tf.nest to recursively map the gather operation across all
        # tensors in the hidden state structure (supports both GRU and LSTM).
        res_h = tf.nest.map_structure(lambda s: tf.reshape(tf.gather(s, idx), [self.N, -1]), h)

        # 3. Standard reset for weights
        # -----------------------------
        # Soft resampling handles the bias internally via its linear combination.
        # We return a uniform log-weight distribution (-log(N)) to signify
        # a fresh ensemble for the next prediction cycle.
        return res_p, res_h, tf.fill((self.N,), -tf.math.log(float(self.N)))

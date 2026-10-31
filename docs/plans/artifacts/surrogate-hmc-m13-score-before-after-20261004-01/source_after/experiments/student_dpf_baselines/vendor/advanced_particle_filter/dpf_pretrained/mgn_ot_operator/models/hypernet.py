"""
Hypernet: maps cloud embedding z (plus log epsilon) to M-MGN parameters.

Output layout (flat vector, then reshaped):
    V:  (d, d)      -- for V^T V term
    a:  (d,)        -- global bias
    W:  (K, h, d)   -- per-module weights
    b:  (K, h)      -- per-module biases

Critical design choice: the final projection layer is initialized so that at
step 0, regardless of z, the hypernet outputs a "good init" MGN:

    V = I_d / sqrt(d)     (so V^T V = I_d / d, a mild positive-definite term)
    a = 0
    W_k = small Gaussian   (std 0.01)
    b_k = 0

Mechanism: set the final Dense kernel to zero, and the final Dense bias to the
flattened good-init parameter vector. Output = 0 * z + bias = bias.

This ensures the network starts near the identity map (after normalization),
and training learns deviations from it. This is the Chang et al. (2019) principle
for hypernet initialization.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .mgn import param_count


def make_good_init_params(d, K, h, W_std=None, V_scale=1.0):
    """Build the flat 'good init' parameter vector that the hypernet biases toward.

    V defaults to V_scale * I (so V^T V linear term equals x at V_scale=1).
    W defaults to cubic-root scaling derived from M-MGN structure:
        module output ~ h^2 * W_std^3 * sqrt(d)
        set W_std = 1 / (h^2 * sqrt(d))^(1/3)  ->  module contribution O(1)
    This keeps the MGN well-conditioned at init without exploding module terms.

    Returns:
        theta_init: (P,) numpy array where P = param_count(d, K, h).
    """
    if W_std is None:
        W_std = 1.0 / (h * h * np.sqrt(d)) ** (1.0 / 3.0)
    V0 = V_scale * np.eye(d)                          # (d, d)
    a0 = np.zeros(d)                                  # (d,)
    # Deterministic random W -- the bias vector is FIXED.
    rng = np.random.default_rng(42)
    W0 = rng.normal(0, W_std, size=(K, h, d))         # (K, h, d)
    b0 = np.zeros((K, h))                             # (K, h)
    return np.concatenate([V0.flatten(), a0.flatten(),
                           W0.flatten(), b0.flatten()]).astype(np.float32)


class Hypernet(layers.Layer):
    """MLP that emits MGN parameters, conditioned on cloud embedding z and log eps.

    Args:
        d: state dimension.
        K: number of M-MGN modules.
        h: M-MGN hidden dim per module.
        hidden: list of hidden layer sizes for the MLP.
    """
    def __init__(self, d, K, h, hidden=(256, 256), **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.K = K
        self.h = h
        self.P = param_count(d, K, h)

        self.trunk = tf.keras.Sequential([
            layers.Dense(hidden[0], activation='gelu'),
            layers.Dense(hidden[1], activation='gelu'),
        ])
        # Final projection: zero-init kernel, good-init bias.
        self.final = layers.Dense(
            self.P,
            kernel_initializer='zeros',
            bias_initializer=tf.keras.initializers.Constant(
                make_good_init_params(d, K, h)
            ),
        )

    def call(self, z_aug):
        """
        Args:
            z_aug: (B, D+1) cloud embedding concatenated with log-epsilon.
        Returns:
            theta_flat: (B, P) flat MGN parameter vector.
        """
        h = self.trunk(z_aug)
        theta = self.final(h)
        return theta

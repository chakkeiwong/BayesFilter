"""
End-to-end monotone gradient operator for amortized OT resampling.

Pipeline:
    1. Normalize:  center and scale x by weighted mean / std of the cloud.
    2. Encode:     Set Transformer on (x_tilde, log w) -> z.
    3. Condition:  concatenate log eps -> z_aug.
    4. Emit:       hypernet(z_aug) -> flat MGN params -> (V, a, W, b).
    5. Apply:      mmgn_forward(x_tilde, V, a, W, b) -> y_tilde.
    6. Denormalize: y = scale * y_tilde + center.

All steps are differentiable. @tf.function-compatible with fixed shapes.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from .mgn import mmgn_forward, unpack_params
from .set_encoder import SetEncoder
from .hypernet import Hypernet


class AmortizedOTOperator(Model):
    """Learns mu -> Brenier(mu -> uniform) as an operator.

    Call signature:
        y = operator(x, w, log_eps)
    where:
        x: (B, N, d) input particle positions
        w: (B, N)    input particle weights (sum to 1 per row)
        log_eps: (B,) log of regularization parameter

    Output:
        y: (B, N, d) pushforward positions (uniform weights implied)
    """
    def __init__(self, d, K=4, h=32,
                 d_model=64, d_embed=128,
                 num_heads=4, num_inducing=16, num_isab=2, num_seeds=4,
                 hypernet_hidden=(256, 256), **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.K = K
        self.h = h

        # Per-particle input features: d (position) + 1 (log weight) = d+1
        self.encoder = SetEncoder(
            d_in=d + 1,
            d_model=d_model,
            d_out=d_embed,
            num_heads=num_heads,
            num_inducing=num_inducing,
            num_isab=num_isab,
            num_seeds=num_seeds,
        )
        self.hypernet = Hypernet(d=d, K=K, h=h, hidden=hypernet_hidden)

    # ---- Normalization -----------------------------------------------------
    @staticmethod
    def _normalize(x, w):
        """Center and scale x using weighted mean and std of the cloud.

        Returns:
            x_tilde: (B, N, d) normalized positions.
            center: (B, 1, d)
            scale:  (B, 1, 1)
        """
        w_exp = w[:, :, tf.newaxis]                                  # (B, N, 1)
        center = tf.reduce_sum(w_exp * x, axis=1, keepdims=True)     # (B, 1, d)
        centered = x - center                                        # (B, N, d)
        # Weighted variance (sum over all coords and particles)
        var = tf.reduce_sum(w_exp * tf.square(centered), axis=[1, 2], keepdims=True)  # (B,1,1)
        # Add a small floor to avoid division by zero for degenerate clouds.
        scale = tf.sqrt(var + 1e-8)                                  # (B, 1, 1)
        x_tilde = centered / scale                                    # (B, N, d)
        return x_tilde, center, scale

    # ---- Forward (normalized-space MGN output) ----------------------------
    def forward_normalized(self, inputs, training=False):
        """Core forward pass; returns MGN output BEFORE denormalization.

        This is the quantity the training loss compares against y_tilde_target.

        Args:
            inputs: tuple (x, w, log_eps) where
                x: (B, N, d), w: (B, N), log_eps: (B,)
        Returns:
            y_tilde: (B, N, d) in normalized space.
            center:  (B, 1, d)
            scale:   (B, 1, 1)
        """
        x, w, log_eps = inputs

        # 1. Normalize.
        x_tilde, center, scale = self._normalize(x, w)

        # 2. Per-particle features: concatenate (x_tilde, log w).
        log_w = tf.math.log(w + 1e-30)[:, :, tf.newaxis]             # (B, N, 1)
        particle_feats = tf.concat([x_tilde, log_w], axis=-1)        # (B, N, d+1)

        # 3. Set encoder.
        z = self.encoder(particle_feats, training=training)          # (B, d_embed)

        # 4. Condition on log eps.
        log_eps_col = log_eps[:, tf.newaxis]                         # (B, 1)
        z_aug = tf.concat([z, log_eps_col], axis=-1)                 # (B, d_embed + 1)

        # 5. Hypernet -> MGN params.
        theta_flat = self.hypernet(z_aug)                            # (B, P)
        V, a, W, b = unpack_params(theta_flat, self.d, self.K, self.h)

        # 6. Apply MGN (in normalized space).
        y_tilde = mmgn_forward(x_tilde, V, a, W, b)                  # (B, N, d)
        return y_tilde, center, scale

    # ---- Forward (world-space output, for inference) ----------------------
    def call(self, inputs, training=False):
        """Full forward: returns y in world coordinates.

        Use forward_normalized() instead if you need the normalized output
        (e.g. to compute loss against y_tilde_target without double-denorm).
        """
        y_tilde, center, scale = self.forward_normalized(inputs, training=training)
        y = scale * y_tilde + center
        return y

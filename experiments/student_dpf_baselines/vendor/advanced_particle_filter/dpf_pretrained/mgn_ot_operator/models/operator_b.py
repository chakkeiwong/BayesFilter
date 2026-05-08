"""
Option B: Coupling-predictor operator.

Drops the M-MGN / gradient-of-convex structural constraint. Instead:

  1. Normalize the weighted input cloud.
  2. Encode per-particle features via an equivariant Set Transformer.
  3. Predict pairwise coupling logits via query/key attention.
  4. Column-softmax to get a coupling pi with exact target marginal = 1/N.
  5. Aggregate: y_tilde_j = N * sum_i pi_{ij} x_tilde_i (barycentric projection).
  6. Denormalize to world space.

Contrast with AmortizedOTOperator (the M-MGN version):
  - M-MGN emitted parameters of a monotone gradient map via a hypernet.
    The output was guaranteed to be gradient-of-convex (a Brenier map).
  - Option B directly predicts the coupling matrix. No structural OT
    guarantees, but also no class mismatch with ETPF targets.

API matches AmortizedOTOperator: call((x, w, log_eps)) -> y.
Also provides forward_normalized returning (y_tilde, center, scale, pi).
"""

import tensorflow as tf
from tensorflow.keras import Model

from .set_encoder import EquivariantSetEncoder
from .coupling_head import CouplingHead


class CouplingOperator(Model):
    """Learns mu -> ETPF-like resampling via a learned coupling matrix.

    Call signature:
        y = operator((x, w, log_eps))
    where:
        x: (B, N, d) input particle positions
        w: (B, N)    input particle weights (sum to 1 per row)
        log_eps: (B,) log of regularization parameter

    Output:
        y: (B, N, d) pushforward positions (uniform weights implied)
    """
    def __init__(self, d,
                 d_model=64, num_heads=4, num_inducing=16, num_isab=2,
                 d_head=64, condition_on_log_eps=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.d = d

        # Per-particle input features: d (position) + 1 (log weight) = d+1.
        self.encoder = EquivariantSetEncoder(
            d_in=d + 1,
            d_model=d_model,
            num_heads=num_heads,
            num_inducing=num_inducing,
            num_isab=num_isab,
        )
        self.coupling = CouplingHead(
            d_head=d_head,
            condition_on_log_eps=condition_on_log_eps,
        )

    # ---- Normalization ----------------------------------------------------
    @staticmethod
    def _normalize(x, w, eps_floor=1e-8):
        w_exp = w[:, :, tf.newaxis]                                  # (B, N, 1)
        center = tf.reduce_sum(w_exp * x, axis=1, keepdims=True)     # (B, 1, d)
        centered = x - center
        var = tf.reduce_sum(w_exp * tf.square(centered),
                            axis=[1, 2], keepdims=True)              # (B,1,1)
        scale = tf.sqrt(var + eps_floor)
        x_tilde = centered / scale
        return x_tilde, center, scale

    # ---- Forward (normalized-space output, with coupling) -----------------
    def forward_normalized(self, inputs, training=False):
        """
        Args:
            inputs: tuple (x, w, log_eps)

        Returns:
            y_tilde: (B, N, d) in normalized space.
            center:  (B, 1, d)
            scale:   (B, 1, 1)
            pi:      (B, N, N) coupling matrix
        """
        x, w, log_eps = inputs

        # 1. Normalize.
        x_tilde, center, scale = self._normalize(x, w)

        # 2. Per-particle features.
        log_w = tf.math.log(w + 1e-30)[:, :, tf.newaxis]
        particle_feats = tf.concat([x_tilde, log_w], axis=-1)   # (B, N, d+1)

        # 3. Set Transformer (equivariant, returns per-particle features).
        f = self.encoder(particle_feats, training=training)    # (B, N, d_model)

        # 4. Coupling head: attention -> column-softmax -> aggregate.
        y_tilde, pi = self.coupling(f, x_tilde, log_eps=log_eps,
                                    training=training)         # (B, N, d)

        return y_tilde, center, scale, pi

    # ---- Forward (world-space output) -------------------------------------
    def call(self, inputs, training=False):
        y_tilde, center, scale, _ = self.forward_normalized(
            inputs, training=training)
        y = scale * y_tilde + center
        return y

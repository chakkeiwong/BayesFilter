"""
Option B coupling head.

Given per-particle contextualized features {f_i} (shape (B, N, d_feat)) and
optional scalar log eps, predicts a coupling matrix pi and aggregates the
input positions into output positions in normalized space.

Pipeline:
    f_i         (contextualized feature, B, N, d_feat)
    +  log eps  (broadcast scalar)
    -> q_i = W_q f'_i,  k_j = W_k f'_j         (B, N, d_head)
    -> L_{ij} = q_i . k_j / sqrt(d_head)       (B, N, N)
    -> pi_{ij} = (1/N) * softmax_i(L_{ij})     (B, N, N), column-softmax
    -> y_tilde_j = N * sum_i pi_{ij} x_tilde_i (B, N, d)

The column-softmax gives target marginal `sum_i pi_{ij} = 1/N` exactly.
The source marginal `sum_j pi_{ij} = w_i` is only approximately satisfied;
this is acceptable since downstream resampling uses the output positions
directly, not pi.

No Brenier-class constraint. Any coupling representable by
`exp(q.k / sqrt(d))` is reachable, which is a much larger family than
gradients of convex functions.
"""

import tensorflow as tf
from tensorflow.keras import layers


class CouplingHead(layers.Layer):
    """Attention-based coupling predictor + aggregation.

    Args:
        d_head: dimension of query/key projection.
        condition_on_log_eps: if True, concatenates log_eps scalar to per-particle
            features before the QK projection.
    """
    def __init__(self, d_head=64, condition_on_log_eps=True, **kwargs):
        super().__init__(**kwargs)
        self.d_head = d_head
        self.condition_on_log_eps = condition_on_log_eps
        self.W_q = layers.Dense(d_head, use_bias=False, name='W_q')
        self.W_k = layers.Dense(d_head, use_bias=False, name='W_k')

    def call(self, f, x_tilde, log_eps=None, training=False):
        """
        Args:
            f:       (B, N, d_feat)  contextualized per-particle features
            x_tilde: (B, N, d)       normalized particle positions (to aggregate)
            log_eps: (B,) or None    scalar log-epsilon per cloud

        Returns:
            y_tilde: (B, N, d) output positions in normalized space
            pi:      (B, N, N) coupling matrix (for diagnostics)
        """
        if self.condition_on_log_eps:
            if log_eps is None:
                raise ValueError("log_eps must be provided when "
                                 "condition_on_log_eps=True")
            # Broadcast log_eps from (B,) to (B, N, 1) and concat.
            N = tf.shape(f)[1]
            le = tf.tile(log_eps[:, tf.newaxis, tf.newaxis], [1, N, 1])
            f = tf.concat([f, le], axis=-1)                # (B, N, d_feat+1)

        # Queries and keys.
        q = self.W_q(f)                                    # (B, N, d_head)
        k = self.W_k(f)                                    # (B, N, d_head)

        # Pairwise logits, scaled.
        scale = tf.cast(tf.math.sqrt(float(self.d_head)), f.dtype)
        # logits[b, i, j] = q[b, i, :] . k[b, j, :] / scale
        logits = tf.matmul(q, k, transpose_b=True) / scale  # (B, N, N)

        # Column-softmax: softmax over i for each j.
        # This gives sum_i pi_{ij} = 1/N exactly after scaling.
        N_f = tf.cast(tf.shape(f)[1], f.dtype)
        col_softmax = tf.nn.softmax(logits, axis=1)         # (B, N, N), sums to 1 along i
        pi = col_softmax / N_f                              # (B, N, N), sums to 1/N along i

        # Aggregate: y_tilde_j = N * sum_i pi_{ij} x_tilde_i
        # pi has shape (B, N_src, N_tgt). Transpose so the N_src axis contracts
        # with x_tilde. Equivalent to N * pi^T @ x_tilde.
        y_tilde = N_f * tf.matmul(pi, x_tilde, transpose_a=True)  # (B, N, d)

        return y_tilde, pi

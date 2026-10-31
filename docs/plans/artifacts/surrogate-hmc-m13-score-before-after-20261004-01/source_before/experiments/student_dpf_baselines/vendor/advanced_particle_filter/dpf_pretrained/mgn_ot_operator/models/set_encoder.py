"""
Set Transformer encoder (Lee et al., 2019), permutation-invariant.

Architecture:

    Per-particle embedding  phi:  (x_i, log w_i) -> h_i in R^{d_model}
    ISAB  x 2                     # Induced Set Attention Blocks, O(N m)
    PMA                           # Pooling by Multihead Attention, k seed vectors
    Linear projection             # -> z in R^D

Notes:
  - Designed for BATCHED input: (B, N, d) x + (B, N) log-weights.
  - Uses standard Keras Dense + MultiHeadAttention, all shape-static except
    for the particle count N, which is dynamic. Safe under @tf.function as
    long as N per batch is consistent (we keep it fixed during training).
  - ISAB is implemented with inducing points as trainable parameters (m, d_model).
"""

import tensorflow as tf
from tensorflow.keras import layers


class MAB(layers.Layer):
    """Multihead Attention Block: X, Y -> LayerNorm(X + MHA(X, Y, Y)); then + MLP.

    This follows Set Transformer paper closely.
    """
    def __init__(self, d_model, num_heads, mlp_hidden=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=d_model // num_heads)
        self.ln1 = layers.LayerNormalization()
        mlp_hidden = mlp_hidden or d_model
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_hidden, activation='gelu'),
            layers.Dense(d_model),
        ])
        self.ln2 = layers.LayerNormalization()

    def call(self, X, Y, training=False):
        # MHA takes (query, value, key) -- we use Y as both key & value.
        attn = self.mha(query=X, value=Y, key=Y, training=training)
        H = self.ln1(X + attn)
        H = self.ln2(H + self.mlp(H))
        return H


class ISAB(layers.Layer):
    """Induced Set Attention Block.

    Given input X in R^{N x d_model}, uses m learnable inducing points I:
        H = MAB(I, X)     # shape (m, d_model)
        Y = MAB(X, H)     # shape (N, d_model)
    Cost O(N m) instead of O(N^2).
    """
    def __init__(self, d_model, num_heads, num_inducing, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_inducing = num_inducing
        self.mab1 = MAB(d_model, num_heads)
        self.mab2 = MAB(d_model, num_heads)

    def build(self, input_shape):
        # Inducing points: shape (m, d_model). Shared across batch.
        self.I = self.add_weight(
            name='inducing',
            shape=(self.num_inducing, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, X, training=False):
        B = tf.shape(X)[0]
        # Broadcast inducing points to batch: (B, m, d_model)
        I_b = tf.broadcast_to(self.I[tf.newaxis, :, :],
                              (B, self.num_inducing, self.d_model))
        H = self.mab1(I_b, X, training=training)   # (B, m, d_model)
        Y = self.mab2(X, H, training=training)     # (B, N, d_model)
        return Y


class PMA(layers.Layer):
    """Pooling by Multihead Attention.

    Uses k learnable seed vectors as queries; attends over the set.
    Output shape: (B, k, d_model).
    """
    def __init__(self, d_model, num_heads, num_seeds, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_seeds = num_seeds
        self.mab = MAB(d_model, num_heads)

    def build(self, input_shape):
        self.S = self.add_weight(
            name='seeds',
            shape=(self.num_seeds, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, X, training=False):
        B = tf.shape(X)[0]
        S_b = tf.broadcast_to(self.S[tf.newaxis, :, :],
                              (B, self.num_seeds, self.d_model))
        return self.mab(S_b, X, training=training)  # (B, k, d_model)


class SetEncoder(layers.Layer):
    """Full encoder: per-particle embed -> ISAB x n_isab -> PMA -> flatten -> Linear.

    Args:
        d_in: input per-particle feature dim (usually d+1 for (position, log w)).
        d_model: transformer hidden dim.
        d_out: final cloud embedding dim.
        num_heads: attention heads.
        num_inducing: number of ISAB inducing points.
        num_isab: number of stacked ISAB blocks.
        num_seeds: number of PMA seed vectors.
    """
    def __init__(self, d_in, d_model=64, d_out=128,
                 num_heads=4, num_inducing=16,
                 num_isab=2, num_seeds=4, **kwargs):
        super().__init__(**kwargs)
        self.embed = tf.keras.Sequential([
            layers.Dense(d_model, activation='gelu'),
            layers.Dense(d_model),
        ])
        self.isab_blocks = [
            ISAB(d_model, num_heads, num_inducing, name=f'isab_{i}')
            for i in range(num_isab)
        ]
        self.pma = PMA(d_model, num_heads, num_seeds)
        self.out_proj = layers.Dense(d_out)

    def call(self, particle_feats, training=False):
        """
        Args:
            particle_feats: (B, N, d_in)
        Returns:
            z: (B, d_out)
        """
        H = self.embed(particle_feats)                         # (B, N, d_model)
        for isab in self.isab_blocks:
            H = isab(H, training=training)                     # (B, N, d_model)
        pooled = self.pma(H, training=training)                # (B, k, d_model)
        B = tf.shape(pooled)[0]
        flat = tf.reshape(pooled, (B, -1))                     # (B, k*d_model)
        z = self.out_proj(flat)                                # (B, d_out)
        return z


class EquivariantSetEncoder(layers.Layer):
    """Per-particle encoder for Option B: embed + ISAB stack, no pooling.

    Returns contextualized features (B, N, d_model) rather than a single
    cloud embedding. Each f_i carries cloud context via ISAB attention
    while remaining permutation-equivariant in the particle index.

    Args:
        d_in: input per-particle feature dim (usually d+1).
        d_model: transformer hidden dim.
        num_heads, num_inducing, num_isab: same as SetEncoder.
    """
    def __init__(self, d_in, d_model=64, num_heads=4,
                 num_inducing=16, num_isab=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embed = tf.keras.Sequential([
            layers.Dense(d_model, activation='gelu'),
            layers.Dense(d_model),
        ])
        self.isab_blocks = [
            ISAB(d_model, num_heads, num_inducing, name=f'isab_{i}')
            for i in range(num_isab)
        ]

    def call(self, particle_feats, training=False):
        """
        Args:
            particle_feats: (B, N, d_in)
        Returns:
            f: (B, N, d_model), contextualized per-particle features
        """
        H = self.embed(particle_feats)                         # (B, N, d_model)
        for isab in self.isab_blocks:
            H = isab(H, training=training)                     # (B, N, d_model)
        return H

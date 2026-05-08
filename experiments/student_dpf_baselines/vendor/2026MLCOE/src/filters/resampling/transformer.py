"""
Particle Transformer Module
===========================
Implements a Transformer-based architecture for Differentiable Particle Filtering (DPF).

Mathematical Context:
---------------------
This module treats the resampling step in Sequential Monte Carlo (SMC) as a permutation-invariant
set-to-set transformation. Traditional resampling (e.g., multinomial) is non-differentiable
due to discrete index sampling. The Particle Transformer circumvents this by using a
cross-attention mechanism where learnable 'seed vectors' (queries) attend to the weighted
prior particle ensemble (keys/values).

By incorporating the importance weights directly into the attention score calculation
(Weighted Multi-Head Attention), the model ensures that particles with higher posterior
likelihood exert a proportionally stronger influence on the generation of the new,
unweighted posterior ensemble. This provides a robust, fully differentiable gradient path.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
from src.filters.classical import DTYPE

class WeightedMultiHeadAttention(tf.keras.layers.Layer):
    """
    Weighted Multi-Head Attention (WMHA)
    ------------------------------------
    An extension of standard Multi-Head Attention that incorporates importance
    weights (w) into the attention mechanism. This ensures that particles with
    higher posterior probability have a proportional influence on the generated
    resampled states.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads, self.head_size = num_heads, head_size
        self.q_proj = tf.keras.layers.Dense(num_heads * head_size)
        self.k_proj = tf.keras.layers.Dense(num_heads * head_size)
        self.v_proj = tf.keras.layers.Dense(num_heads * head_size)
        self.out_proj = tf.keras.layers.Dense(num_heads * head_size)

    def call(self, q, k, v, w=None):
        batch_size = tf.shape(q)[0]

        # Linear projections and reshape for heads
        q_h = tf.transpose(tf.reshape(self.q_proj(q), [batch_size, -1, self.num_heads, self.head_size]), [0, 2, 1, 3])
        k_h = tf.transpose(tf.reshape(self.k_proj(k), [batch_size, -1, self.num_heads, self.head_size]), [0, 2, 1, 3])
        v_h = tf.transpose(tf.reshape(self.v_proj(v), [batch_size, -1, self.num_heads, self.head_size]), [0, 2, 1, 3])

        # Scaled Dot-Product
        score = tf.matmul(q_h, k_h, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_size, tf.float32))

        if w is not None:
            # Inject importance weights into the score
            w_expanded = tf.reshape(w, [batch_size, 1, 1, -1])
            exp_score = tf.math.exp(score) * w_expanded
            attn_weights = exp_score / (tf.reduce_sum(exp_score, axis=-1, keepdims=True) + 1e-12)
        else:
            attn_weights = tf.nn.softmax(score, axis=-1)

        context = tf.matmul(attn_weights, v_h)
        context = tf.reshape(tf.transpose(context, [0, 2, 1, 3]), [batch_size, -1, self.num_heads * self.head_size])
        return self.out_proj(context)


# =============================================================================
# 2. MODULAR RESAMPLERS
# =============================================================================

class TransformerResampler(tf.keras.Model):
    """
    Particle Transformer Resampler
    ------------------------------
    A neural resampling module that treats resampling as a set-to-set
    transformation. It uses seed vectors (queries) to attend to the prior
    particles (keys/values) weighted by their likelihood. This produces a
    new set of particles that are naturally unweighted and differentiable.
    """
    def __init__(self, num_particles, latent_dim=128, num_heads=4):
        super().__init__()
        self.N = num_particles
        self.latent_dim = latent_dim
        self.head_size = latent_dim // num_heads

        # Learnable seed vectors representing the 'new' particle set
        self.seed_vectors = self.add_weight(
            shape=(num_particles, latent_dim),
            initializer="random_normal",
            trainable=True
        )

        self.enc_linear = tf.keras.layers.Dense(latent_dim)
        self.enc_blocks = [
            (WeightedMultiHeadAttention(num_heads, self.head_size),
             tf.keras.Sequential([tf.keras.layers.Dense(latent_dim, activation='relu'),
                                  tf.keras.layers.Dense(latent_dim)]))
            for _ in range(2)
        ]

        self.dec_blocks = [
            (WeightedMultiHeadAttention(num_heads, self.head_size),
             WeightedMultiHeadAttention(num_heads, self.head_size),
             tf.keras.Sequential([tf.keras.layers.Dense(latent_dim, activation='relu'),
                                  tf.keras.layers.Dense(latent_dim)]))
            for _ in range(2)
        ]
        self.dec_out_linear = tf.keras.layers.Dense(1)

    def call(self, particles, h_states, weights):
        # Normalization and input prep
        x = tf.expand_dims(tf.expand_dims(particles, -1), 0)
        w = tf.expand_dims(weights, 0)
        p_min, p_max = tf.reduce_min(x, axis=1, keepdims=True), tf.reduce_max(x, axis=1, keepdims=True)
        x_scaled = 2.0 * (x - p_min) / (p_max - p_min + 1e-12) - 1.0

        # Encoder: Contextualize particles
        h = self.enc_linear(x_scaled)
        for attn, ff in self.enc_blocks:
            h = h + attn(h, h, h, w) + ff(h)

        # Decoder: Generate new particles from seeds
        z = tf.expand_dims(self.seed_vectors, 0)
        for s_attn, w_attn, ff in self.dec_blocks:
            z = z + s_attn(z, z, z) + w_attn(z, h, h, w) + ff(z)

        out = (self.dec_out_linear(z) + 1.0) / 2.0 * (p_max - p_min + 1e-12) + p_min

        # Categorical fallback for hidden states (non-differentiable part of h_states)
        indices = tf.cast(tf.squeeze(tf.random.categorical(tf.math.log(weights[None, :] + 1e-12), self.N)), tf.int32)
        res_h = tf.nest.map_structure(lambda s: tf.reshape(tf.gather(s, indices), [self.N, -1]), h_states)

        return tf.squeeze(out), res_h, tf.fill((self.N,), -tf.math.log(float(self.N)))

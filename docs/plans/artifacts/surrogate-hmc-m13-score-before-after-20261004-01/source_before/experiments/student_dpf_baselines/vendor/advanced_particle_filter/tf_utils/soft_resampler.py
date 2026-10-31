"""
Soft resampling for differentiable particle filtering — TensorFlow.

Implementation of the straight-through (ST) soft resampler from Jonschkowski
et al. (2018) and Corenflos et al. (2021). This is the low-memory alternative
to Sinkhorn-OT resampling: forward uses sampled ancestor indices (discrete);
backward sees only a weighted mean (differentiable).

Memory profile
--------------
OT resampling retains O(N_s * B * N^2) activations per PF step during
backprop because autodiff through tf.while_loop saves per-iteration
intermediates. Soft resampling retains only O(B * N * D) per step — the
single weighted mean tensor. At typical scales (B=8, N=100, N_s=30, T=75,
D=2, float64) this is a ~3000x memory reduction.

See docs/memory_compute_analysis.pdf for the full derivation.

Interface
---------
Matches tf_utils/sinkhorn.py so swapping resamplers is a one-line change:

    from tf_utils.soft_resampler import soft_resample
    particles_r, log_w_r = soft_resample(particles, log_w, alpha, rng)

JIT compatibility
-----------------
Per the TF migration guideline this module follows these rules:
  - @tf.function(reduce_retracing=True) on the public entry point
  - No Python for-loops over dynamic iteration counts
  - tf.random.Generator passed explicitly (no global state)
  - No .numpy() calls, no Python control flow on tensor values
  - Static shapes preserved through tf.stack/tf.gather; no tf.reshape
    with a dynamic shape
"""

import tensorflow as tf
import tensorflow_probability as tfp

from typing import Tuple

tfd = tfp.distributions


@tf.function(reduce_retracing=True)
def soft_resample(
    particles: tf.Tensor,
    log_w_norm: tf.Tensor,
    alpha: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Differentiable resampling via straight-through estimator on a
    (weight, uniform) mixture.

    Forward (non-differentiable part):
        w_soft = alpha * w + (1 - alpha) * uniform
        ancestors ~ Categorical(w_soft)       per batch row
        particles_hard = gather(particles, ancestors)

    Differentiable surrogate (used in backward pass only):
        x_interp = sum_i w_soft_i * particles_i    [B, D]  per batch row

    Straight-through trick: the returned particles equal particles_hard
    in value but carry gradients as if they were broadcast copies of
    x_interp. This gives gradient flow through the resampling step with
    memory O(B * N * D) rather than O(N_s * B * N^2) as for Sinkhorn.

    Importance-weight correction:
        log w_new^i = log w^{a_i} - log w_soft^{a_i}
    followed by normalisation. This keeps the estimator unbiased for the
    forward log-evidence estimate (up to the usual particle-filter
    consistency) while the backward path feeds the smooth weighted mean.

    Args:
        particles:  [B, N, D] input particles.
        log_w_norm: [B, N] normalised log-weights (log-probabilities).
        alpha:      scalar tensor in (0, 1]. Controls weight-vs-uniform
                    mixing. alpha=1 recovers pure categorical resampling
                    (but loses gradient smoothness); alpha=0 is pure
                    uniform (ignores weights). Typical: 0.5.
        rng:        tf.random.Generator.

    Returns:
        resampled:  [B, N, D] straight-through resampled particles.
        new_log_w:  [B, N] normalised log-weights after importance
                    correction.
    """
    dtype = particles.dtype
    B = tf.shape(particles)[0]
    N = tf.shape(particles)[1]
    N_f = tf.cast(N, dtype)

    # --- 1. Build soft weights: mixture with uniform ---
    w_norm = tf.exp(log_w_norm)                             # [B, N]
    w_uniform = tf.ones_like(w_norm) / N_f                  # [B, N]
    w_soft = alpha * w_norm + (tf.constant(1.0, dtype=dtype) - alpha) * w_uniform
    log_w_soft = tf.math.log(w_soft + tf.constant(1e-30, dtype=dtype))

    # --- 2. Sample ancestors from Categorical(w_soft), independently per
    # batch row. We draw a single seed from rng and use it in a stateless
    # categorical so the graph has no hidden RNG state that would require
    # retracing. tfd.Categorical reduces to tf.random.stateless_categorical
    # under the hood when given a seed; here we use rng.uniform trick for
    # clarity and JIT-safety.
    ancestors = _sample_categorical(log_w_soft, N, rng)     # [B, N]

    # --- 3. Gather particles at ancestor indices (non-differentiable path) ---
    # Build [B, N, 2] indices of (batch, ancestor) pairs.
    batch_idx = tf.broadcast_to(
        tf.range(B, dtype=ancestors.dtype)[:, tf.newaxis], [B, N]
    )
    gather_idx = tf.stack([batch_idx, ancestors], axis=-1)  # [B, N, 2]
    particles_hard = tf.gather_nd(particles, gather_idx)    # [B, N, D]

    # --- 4. Differentiable surrogate: weighted mean of source particles ---
    # x_interp[b] = sum_i w_soft[b, i] * particles[b, i]   [B, 1, D] (broadcast)
    x_interp = tf.reduce_sum(
        w_soft[:, :, tf.newaxis] * particles, axis=1, keepdims=True
    )                                                       # [B, 1, D]
    x_interp = tf.broadcast_to(x_interp, tf.shape(particles))  # [B, N, D]

    # --- 5. Straight-through: value = hard, gradient = interp ---
    resampled = tf.stop_gradient(particles_hard - x_interp) + x_interp

    # --- 6. Importance-weight correction ---
    # log_w_ancestors: values of log_w_norm at the sampled ancestor indices
    log_w_ancestors = tf.gather_nd(log_w_norm, gather_idx)          # [B, N]
    log_w_soft_ancestors = tf.gather_nd(log_w_soft, gather_idx)     # [B, N]
    new_log_w_unnorm = log_w_ancestors - log_w_soft_ancestors       # [B, N]

    # Normalise per batch row
    new_log_w = new_log_w_unnorm - tf.math.reduce_logsumexp(
        new_log_w_unnorm, axis=-1, keepdims=True
    )

    return resampled, new_log_w


@tf.function(reduce_retracing=True)
def _sample_categorical(
    log_probs: tf.Tensor,
    n_samples: int,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """
    Draw `n_samples` per batch row from Categorical(log_probs).

    Uses the Gumbel-max trick in log-space — fully JIT-compatible:
        argmax_i (log_probs_i + G_i)   with  G_i ~ Gumbel(0, 1)

    This avoids tf.random.categorical's quirks under tf.function and keeps
    all randomness inside the passed-in tf.random.Generator.

    Args:
        log_probs: [B, K] unnormalised log-probabilities per batch row.
        n_samples: int number of samples per row.
        rng:       tf.random.Generator.

    Returns:
        samples: [B, n_samples] int32 ancestor indices.
    """
    dtype = log_probs.dtype
    B = tf.shape(log_probs)[0]
    K = tf.shape(log_probs)[1]

    # Draw Gumbel noise of shape [B, n_samples, K]
    u = rng.uniform(
        shape=[B, n_samples, K],
        minval=tf.constant(1e-20, dtype=dtype),
        maxval=tf.constant(1.0,   dtype=dtype),
        dtype=dtype,
    )
    g = -tf.math.log(-tf.math.log(u))                       # [B, n_samples, K]

    # Broadcast log_probs and add noise, then argmax over K
    scored = log_probs[:, tf.newaxis, :] + g                # [B, n_samples, K]
    return tf.cast(tf.argmax(scored, axis=-1), tf.int32)    # [B, n_samples]

"""
Sinkhorn OT resampling for differentiable particle filtering — TensorFlow.

XLA-fused Sinkhorn iterations via Python-level unrolling inside a
jit_compile=True tf.function. Accepts a leading batch dimension so the
same kernel serves both (a) parallel MC replicas for variance reduction
and (b) parallel HMC chains.

Design note: Option-C optimization
----------------------------------
Earlier versions used tf.while_loop over the Sinkhorn iterations, letting
the outer filter function wrap everything in a single XLA-compiled graph.
That approach failed: the backward pass through tf.while_loop builds a
TensorList to save per-iteration activations for gradient computation,
and TensorList cannot cross the XLA/TF boundary in the current TF/XLA
implementation. Result: forward compiled fine; backward threw
'Support for TensorList crossing the XLA/TF boundary is not implemented'.

Fix: Unroll the Sinkhorn iterations at Python-trace time by using a
Python `for` loop instead of tf.while_loop. The outer filter function
stays as a plain @tf.function with tf.while_loop over time steps (no
JIT at that level — gradients need the TensorList and that requires
the non-XLA path). But this inner function is JIT-compiled and XLA
fuses all num_iters rounds of logsumexp into a single kernel.

num_iters thus becomes a Python int (not tf.Tensor). Changing it
triggers a retrace + recompile. In our HMC pipeline it is set once at
filter construction and never changes, so this is fine.

Shape convention:
    particles:  [B, N, D]
    log_w:      [B, N]
    returned:   [B, N, D], [B, N]
"""

import tensorflow as tf


def sinkhorn_resample(
    particles: tf.Tensor,
    log_w: tf.Tensor,
    epsilon: tf.Tensor,
    num_iters: int,
):
    """
    Differentiable OT resampling via entropy-regularised Sinkhorn.

    Args:
        particles: [B, N, D] input particles (float32 or float64)
        log_w:     [B, N] normalised log-weights
        epsilon:   scalar tensor, entropic regularisation strength
        num_iters: Python int, number of Sinkhorn iterations. Must be a
                   Python int (not tf.Tensor) so the iteration loop
                   unrolls at trace time into straight-line XLA code.

    Returns:
        resampled: [B, N, D] resampled particles
        new_log_w: [B, N] uniform log-weights (all equal to -log N)
    """
    if not isinstance(num_iters, int):
        raise TypeError(
            f"num_iters must be a Python int for XLA fusion, got "
            f"{type(num_iters).__name__}. Changing num_iters at runtime "
            "would require recompilation; pin it at filter construction."
        )
    return _sinkhorn_resample_impl(particles, log_w, epsilon, num_iters)


@tf.function(reduce_retracing=True, jit_compile=True)
def _sinkhorn_resample_impl(
    particles: tf.Tensor,
    log_w: tf.Tensor,
    epsilon: tf.Tensor,
    num_iters: int,
):
    """
    XLA-compiled implementation. The Python `for` loop unrolls into
    num_iters sequential ops at trace time, which XLA then fuses into
    a single kernel — no tf.while_loop, no TensorList, no boundary issue.
    """
    dtype = particles.dtype
    N = tf.shape(particles)[1]
    N_f = tf.cast(N, dtype)

    # Cost matrix: squared Euclidean per batch
    diff = particles[:, :, tf.newaxis, :] - particles[:, tf.newaxis, :, :]
    C = tf.reduce_sum(diff * diff, axis=-1)            # [B, N, N]

    # Gibbs kernel in log-space
    log_K = -C / epsilon                                # [B, N, N]

    # Target marginal: uniform
    log_b = tf.fill(tf.shape(log_w), -tf.math.log(N_f))

    # Sinkhorn iterations — Python loop, unrolled at trace time.
    # XLA will fuse all num_iters rounds into a single kernel so wall
    # time is roughly constant w.r.t. num_iters in the kernel-launch
    # sense (though total FLOPs still scale linearly).
    log_u = tf.zeros_like(log_w)
    log_v = tf.zeros_like(log_b)

    for _ in range(num_iters):
        # log_u_new = log_a - logsumexp_j(log_K + log_v[newaxis])
        log_u = log_w - tf.math.reduce_logsumexp(
            log_K + log_v[:, tf.newaxis, :], axis=-1
        )
        # log_v_new = log_b - logsumexp_i(log_K + log_u_new[newaxis])
        log_v = log_b - tf.math.reduce_logsumexp(
            log_K + log_u[:, :, tf.newaxis], axis=1
        )

    # Transport plan: T_{ij} = exp(log_u_i + log_K_{ij} + log_v_j)
    log_T = log_u[:, :, tf.newaxis] + log_K + log_v[:, tf.newaxis, :]
    T = tf.exp(log_T)                                   # [B, N, N]

    # Barycentric projection: x_new_i = N * sum_j T_{j,i} x_j
    transform = N_f * tf.transpose(T, perm=[0, 2, 1])   # [B, N, N]
    resampled = tf.matmul(transform, particles)         # [B, N, D]

    new_log_w = tf.fill(tf.shape(log_w), -tf.math.log(N_f))

    return resampled, new_log_w


@tf.function(reduce_retracing=True)
def batched_normalize_log_weights(log_w: tf.Tensor) -> tf.Tensor:
    """
    Batched version of normalize_log_weights.

    Args:
        log_w: [B, N] unnormalised log-weights

    Returns:
        log_w_norm:   [B, N] normalised log-weights
        log_evidence: [B]    log of the normalising constant per batch
    """
    log_sum = tf.math.reduce_logsumexp(log_w, axis=-1, keepdims=True)
    log_w_norm = log_w - log_sum
    log_evidence = tf.squeeze(log_sum, axis=-1)
    return log_w_norm, log_evidence


@tf.function(reduce_retracing=True)
def batched_effective_sample_size(log_w_norm: tf.Tensor) -> tf.Tensor:
    """
    Batched ESS from normalised log-weights.

    Args:
        log_w_norm: [B, N]

    Returns:
        ess: [B]
    """
    return tf.exp(-tf.math.reduce_logsumexp(2.0 * log_w_norm, axis=-1))

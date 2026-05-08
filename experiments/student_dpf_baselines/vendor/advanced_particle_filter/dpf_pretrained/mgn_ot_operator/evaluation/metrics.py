"""
Evaluation metrics and utilities for the trained operator.

Metrics:
  - MSE vs ETPF target
  - Per-particle L2 error
  - Transport cost: E_w[||y - x||^2]
  - Set-level Sinkhorn divergence between predicted and target (index-free)
  - Wall-clock speedup vs direct Sinkhorn
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import tensorflow as tf

from data.sinkhorn_targets import etpf_targets, sinkhorn_log, pairwise_sq_dist


def sinkhorn_divergence(x, y, eps=0.05, n_iters=100):
    """Sinkhorn divergence between two uniformly-weighted empirical measures.

    SD(x, y) = 2 W_eps(x, y) - W_eps(x, x) - W_eps(y, y)

    where W_eps is the entropic OT cost. Both x, y have shape (B, N, d)
    with implied uniform weights 1/N.

    Returns:
        SD: (B,) divergence per cloud pair.
    """
    B = tf.shape(x)[0]
    N = tf.shape(x)[1]
    N_f = tf.cast(N, x.dtype)
    u = tf.fill((B, N), 1.0 / N_f)
    eps_b = tf.fill((B,), tf.constant(eps, dtype=x.dtype))

    def ot_cost(a, b):
        C = tf.reduce_sum(
            tf.square(a[:, :, tf.newaxis, :] - b[:, tf.newaxis, :, :]),
            axis=-1,
        )
        T = sinkhorn_log(C, u, u, eps_b, n_iters=n_iters)
        return tf.reduce_sum(T * C, axis=[1, 2])  # (B,)

    return 2.0 * ot_cost(x, y) - ot_cost(x, x) - ot_cost(y, y)


def evaluate_on_split(operator, split_dict, batch_size=32,
                      sinkhorn_div_eps=0.1, sinkhorn_div_iters=100):
    """Compute standard metrics on a precomputed split.

    All position-based metrics are in NORMALIZED space, matching how the
    targets are stored.

    Returns dict of per-metric mean values (scalar floats).
    """
    x_all = split_dict['x']
    w_all = split_dict['w']
    y_tilde_all = split_dict['y_tilde']
    log10_eps_grid = split_dict['log10_eps_grid']
    n_clouds, n_eps = y_tilde_all.shape[:2]

    mses, l2s, tcs_pred, tcs_target, sds = [], [], [], [], []
    for e_idx, log10_eps in enumerate(log10_eps_grid):
        log_eps = np.float32(log10_eps * np.log(10.0))
        for start in range(0, n_clouds, batch_size):
            end = min(start + batch_size, n_clouds)
            x = tf.constant(x_all[start:end])
            w = tf.constant(w_all[start:end])
            y_tilde_tgt = tf.constant(y_tilde_all[start:end, e_idx])
            le = tf.fill((end - start,), log_eps)

            # Operator output in normalized space.
            y_tilde_pred, center, scale = operator.forward_normalized(
                (x, w, le), training=False)
            x_tilde = (x - center) / scale

            mses.append(float(tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_tgt))))
            l2s.append(float(tf.reduce_mean(tf.norm(y_tilde_pred - y_tilde_tgt, axis=-1))))

            tcs_pred.append(float(tf.reduce_mean(
                tf.reduce_sum(w[:, :, None] * tf.square(y_tilde_pred - x_tilde),
                              axis=[1, 2]))))
            tcs_target.append(float(tf.reduce_mean(
                tf.reduce_sum(w[:, :, None] * tf.square(y_tilde_tgt - x_tilde),
                              axis=[1, 2]))))

            sd = sinkhorn_divergence(y_tilde_pred, y_tilde_tgt,
                                     eps=sinkhorn_div_eps,
                                     n_iters=sinkhorn_div_iters)
            sds.append(float(tf.reduce_mean(sd)))

    return {
        'mse': float(np.mean(mses)),
        'per_particle_l2': float(np.mean(l2s)),
        'transport_cost_pred': float(np.mean(tcs_pred)),
        'transport_cost_target': float(np.mean(tcs_target)),
        'sinkhorn_divergence': float(np.mean(sds)),
    }


def benchmark_speedup(operator, N=1000, d=2, eps=0.5,
                      n_warmup=5, n_trials=50, batch_size=32,
                      sinkhorn_iters=1000):
    """Compare operator forward time vs direct Sinkhorn (in normalized space).

    Default eps=0.5 matches Corenflos's DPF default. Returns dict with mean
    times in ms and speedup ratio.
    """
    from data.sinkhorn_targets import etpf_targets_normalized

    rng = np.random.default_rng(0)
    x_np = rng.normal(size=(batch_size, N, d)).astype(np.float32)
    w_np = rng.random((batch_size, N)).astype(np.float32)
    w_np = w_np / w_np.sum(axis=-1, keepdims=True)
    x = tf.constant(x_np)
    w = tf.constant(w_np)
    le = tf.fill((batch_size,), np.float32(np.log(eps)))
    eps_b = tf.fill((batch_size,), np.float32(eps))

    @tf.function
    def op_fwd(x, w, le):
        return operator((x, w, le), training=False)

    def sinkhorn_fwd(x, w, eps_b):
        # Not @tf.function because etpf_targets_normalized has Python-level
        # dtype casts that are awkward inside tf.function.
        y, _ = etpf_targets_normalized(x, w, eps_b, n_iters=sinkhorn_iters)
        return y

    for _ in range(n_warmup):
        _ = op_fwd(x, w, le)
        _ = sinkhorn_fwd(x, w, eps_b)

    t0 = time.time()
    for _ in range(n_trials):
        y = op_fwd(x, w, le)
    _ = y.numpy()
    t_op = (time.time() - t0) / n_trials * 1000

    t0 = time.time()
    for _ in range(n_trials):
        y = sinkhorn_fwd(x, w, eps_b)
    _ = y.numpy()
    t_sk = (time.time() - t0) / n_trials * 1000

    return {
        'operator_ms': t_op,
        'sinkhorn_ms': t_sk,
        'speedup': t_sk / t_op,
        'batch_size': batch_size,
        'N': N,
        'd': d,
        'eps': eps,
    }

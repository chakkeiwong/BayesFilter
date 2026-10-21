"""
Modular Monotone Gradient Network (M-MGN) forward pass.

Implements Chaudhari et al. (2023), Section 4.2:

    M-MGN(x) = a + V^T V x + sum_{k=1..K} s_k(z_k) * W_k^T * sigma_k(z_k)
    where z_k = W_k x + b_k

with s_k = sum_j log cosh(z_j)  and  sigma_k = d/dz s_k = tanh(z).
These satisfy Prop. 2: the Jacobian w.r.t. x is PSD everywhere.

This module does NOT own (V, a, W_k, b_k) as tf.Variables. They are
provided at call time -- emitted by a hypernet. This is what enables the
outer operator to be a function of the input cloud mu.

Shapes (single cloud):
    x:      (N, d)
    V:      (d, d)         -> V^T V term
    a:      (d,)
    W_k:    (h, d)         for k = 0..K-1, stacked as (K, h, d)
    b_k:    (h,)           stacked as (K, h)
Returns:
    y:      (N, d)

Shapes (batched over clouds):
    x:      (B, N, d)
    V:      (B, d, d)
    a:      (B, d)
    W:      (B, K, h, d)
    b:      (B, K, h)
Returns:
    y:      (B, N, d)

All tensors are float32 unless specified.
"""

import tensorflow as tf


def mmgn_forward(x, V, a, W, b):
    """M-MGN forward pass, batched over clouds.

    Args:
        x: (B, N, d) particle positions, one cloud per batch element.
        V: (B, d, d) emitted by hypernet.
        a: (B, d) global bias.
        W: (B, K, h, d) per-module weight matrices.
        b: (B, K, h) per-module biases.

    Returns:
        y: (B, N, d) pushforward positions.
    """
    # --- V^T V x term -------------------------------------------------------
    # VtV shape (B, d, d). Symmetric, PSD.
    VtV = tf.matmul(V, V, transpose_a=True)  # (B, d, d)
    # Apply to each particle:  (B, N, d) @ (B, d, d) -> (B, N, d)
    linear_term = tf.matmul(x, VtV, transpose_b=True)  # V^T V symmetric, but keep explicit

    # --- Module contributions -----------------------------------------------
    # For each of the K modules, we need:
    #   z_k = W_k x + b_k          shape (B, N, h)
    #   s_k(z_k) = sum_j log cosh(z_kj)   shape (B, N, 1)
    #   sigma_k(z_k) = tanh(z_k)   shape (B, N, h)
    #   contribution = s_k(z_k) * sigma_k(z_k) @ W_k      shape (B, N, d)
    #
    # We vectorize the K dimension via einsum.

    # z has shape (B, K, N, h).  From: x (B, N, d), W (B, K, h, d), b (B, K, h).
    # z_{bknh} = sum_d W_{bkhd} x_{bnd}  + b_{bkh}
    z = tf.einsum('bkhd,bnd->bknh', W, x) + b[:, :, tf.newaxis, :]

    # sigma_k(z) = tanh(z), elementwise.
    sigma = tf.math.tanh(z)  # (B, K, N, h)

    # s_k(z) = sum_j log cosh(z_j), scalar per (b, k, n).
    # log cosh is computed stably as |z| + softplus(-2|z|) - log 2.
    s_per_component = _log_cosh(z)  # (B, K, N, h)
    s = tf.reduce_sum(s_per_component, axis=-1, keepdims=True)  # (B, K, N, 1)

    # scaled_sigma = s * sigma, shape (B, K, N, h)
    scaled_sigma = s * sigma

    # contribution_k = scaled_sigma_k @ W_k^T           gives (B, K, N, d)
    # = einsum 'bknh,bkhd->bknd'
    contributions = tf.einsum('bknh,bkhd->bknd', scaled_sigma, W)

    # Sum over modules k.
    module_term = tf.reduce_sum(contributions, axis=1)  # (B, N, d)

    # --- Assemble -----------------------------------------------------------
    y = a[:, tf.newaxis, :] + linear_term + module_term  # (B, N, d)
    return y


def _log_cosh(z):
    """Numerically stable log(cosh(z)).

    log cosh(z) = |z| + log(1 + exp(-2|z|)) - log 2
                = |z| + softplus(-2|z|) - log 2
    """
    abs_z = tf.abs(z)
    return abs_z + tf.math.softplus(-2.0 * abs_z) - tf.math.log(2.0)


def param_count(d, K, h):
    """Number of parameters per cloud that the hypernet must emit."""
    n_V = d * d
    n_a = d
    n_W = K * h * d
    n_b = K * h
    return n_V + n_a + n_W + n_b


def unpack_params(theta_flat, d, K, h):
    """Unpack a flat parameter vector into (V, a, W, b) with batch dim.

    Args:
        theta_flat: (B, P) where P = param_count(d, K, h).
        d, K, h: dimension hyperparameters.

    Returns:
        V: (B, d, d)
        a: (B, d)
        W: (B, K, h, d)
        b: (B, K, h)
    """
    B = tf.shape(theta_flat)[0]
    n_V = d * d
    n_a = d
    n_W = K * h * d
    n_b = K * h

    V_flat = theta_flat[:, :n_V]
    a = theta_flat[:, n_V:n_V + n_a]
    W_flat = theta_flat[:, n_V + n_a:n_V + n_a + n_W]
    b_flat = theta_flat[:, n_V + n_a + n_W:]

    V = tf.reshape(V_flat, (B, d, d))
    W = tf.reshape(W_flat, (B, K, h, d))
    b = tf.reshape(b_flat, (B, K, h))
    return V, a, W, b

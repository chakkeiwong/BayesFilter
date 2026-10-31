"""
Batched Sinkhorn in log-domain, implemented in pure TensorFlow.

We solve, for each cloud independently:

    T* = argmin_{T >= 0}  sum_{ij} T_ij || x_i - x_j ||^2   +  eps * KL(T || w (x) u)
    s.t.  T 1 = w            (row marginals = weighted input)
          T^T 1 = u = (1/N) 1 (column marginals = uniform target)

The barycentric projection gives the target positions:

    y_j = (1/u_j) * sum_i T_{ij} x_i  =  N * (T^T x)_j

This is the Ensemble Transform Particle Filter (ETPF) target of Reich (2013),
with entropic regularization -- exactly what Corenflos et al. (2021) use in
differentiable particle filters.

Design notes:
  - Log-domain Sinkhorn is essential at small eps (say eps < 1e-2).
  - Fully batched across clouds:  x has shape (B, N, d), w has shape (B, N).
  - We support per-cloud epsilon: eps has shape (B,).
  - Returned T has shape (B, N, N); targets y have shape (B, N, d).
  - Not wrapped in @tf.function because we call this during offline precompute
    with varying shapes / counts and don't need JIT speed here.
"""

import tensorflow as tf


def pairwise_sq_dist(x):
    """Squared Euclidean distance matrix.

    Args:
        x: (B, N, d)
    Returns:
        C: (B, N, N), C[b, i, j] = ||x[b,i] - x[b,j]||^2
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq_norm = tf.reduce_sum(x * x, axis=-1, keepdims=True)  # (B, N, 1)
    inner = tf.matmul(x, x, transpose_b=True)               # (B, N, N)
    C = sq_norm + tf.linalg.matrix_transpose(sq_norm) - 2.0 * inner
    # Numerical: clip tiny negatives from floating-point error.
    return tf.maximum(C, 0.0)


@tf.function(reduce_retracing=True)
def _sinkhorn_log_loop(minus_C_over_eps, eps_b, log_a, log_b, n_iters):
    """The hot inner loop: n_iters Sinkhorn updates in log-domain.

    Wrapped in @tf.function so the whole loop becomes a compiled graph
    rather than n_iters separate eager ops. Python-side dispatch
    overhead drops from O(n_iters) to O(1).
    """
    B = tf.shape(minus_C_over_eps)[0]
    N = tf.shape(minus_C_over_eps)[1]
    M = tf.shape(minus_C_over_eps)[2]
    dtype = minus_C_over_eps.dtype

    f = tf.zeros((B, N), dtype=dtype)
    g = tf.zeros((B, M), dtype=dtype)

    # Use tf.while_loop so n_iters can be a tensor and the loop is truly
    # compiled. A Python for-loop with @tf.function works too but is
    # slower to trace for large n_iters.
    def cond(i, f_, g_):
        return i < n_iters

    def body(i, f_, g_):
        M_mat = minus_C_over_eps + g_[:, tf.newaxis, :] / eps_b[:, tf.newaxis, tf.newaxis]
        lse_j = tf.reduce_logsumexp(M_mat, axis=2)
        f_new = eps_b[:, tf.newaxis] * (log_a - lse_j)

        M_mat2 = minus_C_over_eps + f_new[:, :, tf.newaxis] / eps_b[:, tf.newaxis, tf.newaxis]
        lse_i = tf.reduce_logsumexp(M_mat2, axis=1)
        g_new = eps_b[:, tf.newaxis] * (log_b - lse_i)
        return i + 1, f_new, g_new

    _, f_final, g_final = tf.while_loop(
        cond, body,
        loop_vars=[tf.constant(0), f, g],
        parallel_iterations=1,
    )
    return f_final, g_final


def sinkhorn_log(C, a, b, eps, n_iters=200, tol=1e-6):
    """Log-domain Sinkhorn, batched.

    Args:
        C: (B, N, M) cost matrix.
        a: (B, N) source marginals (>=0, sum to 1 along last axis).
        b: (B, M) target marginals (>=0, sum to 1 along last axis).
        eps: scalar or (B,) regularization parameter.
        n_iters: maximum number of iterations.
        tol: stop when marginals match within this tolerance.

    Returns:
        T: (B, N, M) the optimal coupling.
    """
    B = tf.shape(C)[0]

    # Eager-side dtype/shape handling: tf.rank check is done on a Python
    # level, so it's safe here (sinkhorn_log itself is NOT @tf.function).
    if hasattr(eps, 'shape') and len(eps.shape) > 0:
        eps_b = eps
    else:
        eps_b = tf.fill([B], eps)
    eps_b = tf.cast(eps_b, C.dtype)

    log_a = tf.math.log(a + 1e-30)
    log_b = tf.math.log(b + 1e-30)
    minus_C_over_eps = -C / eps_b[:, tf.newaxis, tf.newaxis]

    # Hot loop: compiled via @tf.function.
    n_iters_t = tf.constant(n_iters)
    f, g = _sinkhorn_log_loop(minus_C_over_eps, eps_b, log_a, log_b, n_iters_t)

    # Assemble T from dual variables.
    log_T = (f[:, :, tf.newaxis] + g[:, tf.newaxis, :] - C) / eps_b[:, tf.newaxis, tf.newaxis]
    T = tf.exp(log_T)
    return T


def marginal_residuals(T, a, b):
    """L1 residuals of Sinkhorn coupling marginals.

    For a coupling T that should satisfy T 1 = a and T^T 1 = b, returns

        row_res[b]  = sum_i | sum_j T[b,i,j] - a[b,i] |       # L1 over rows
        col_res[b]  = sum_j | sum_i T[b,i,j] - b[b,j] |       # L1 over columns

    These should both be near zero for converged Sinkhorn. Sum >> tol means
    the Sinkhorn iteration did not converge at the given eps for this cloud.

    Args:
        T: (B, N, M) coupling.
        a: (B, N) source marginals.
        b: (B, M) target marginals.

    Returns:
        row_res: (B,)
        col_res: (B,)
    """
    row_sum = tf.reduce_sum(T, axis=2)                         # (B, N)
    col_sum = tf.reduce_sum(T, axis=1)                         # (B, M)
    row_res = tf.reduce_sum(tf.abs(row_sum - a), axis=1)       # (B,)
    col_res = tf.reduce_sum(tf.abs(col_sum - b), axis=1)       # (B,)
    return row_res, col_res


def weighted_normalize(x, w, eps_floor=1e-8):
    """Normalize a weighted particle cloud to weighted-mean 0, weighted-std 1.

    This matches the normalization inside AmortizedOTOperator._normalize so the
    training targets we compute are in the same coordinate frame the model
    operates on.

    Args:
        x: (B, N, d)
        w: (B, N), non-negative, sum to 1 along last axis.
        eps_floor: numerical floor on scale.

    Returns:
        x_tilde: (B, N, d) normalized positions.
        center:  (B, 1, d)
        scale:   (B, 1, 1)
    """
    w_exp = w[:, :, tf.newaxis]                                  # (B, N, 1)
    center = tf.reduce_sum(w_exp * x, axis=1, keepdims=True)     # (B, 1, d)
    centered = x - center
    var = tf.reduce_sum(w_exp * tf.square(centered), axis=[1, 2], keepdims=True)
    scale = tf.sqrt(var + eps_floor)                             # (B, 1, 1)
    x_tilde = centered / scale
    return x_tilde, center, scale


def etpf_targets(x, w, eps, n_iters=200, return_residuals=False,
                 use_float64=True):
    """Compute ETPF targets via batched Sinkhorn in UN-NORMALIZED space.

    Source marginals:  w            (B, N)  (arbitrary weights, sum to 1)
    Target marginals:  u = 1/N      (B, N)  (uniform)
    Support of target: the same support as source (empirical x).

    This is Reich's ETPF: reweight mass across the existing particle set
    rather than creating new support locations. The OUTPUT positions y_j
    are a weighted barycentric projection of x through the coupling.

    NOTE: ETPF is sensitive to the scale of X at fixed eps (see Corenflos
    et al. 2021, Section 3.2, and Proposition 4.2 remark). For scale-
    independent behavior, use `etpf_targets_normalized` instead. This
    function is kept for backward compatibility and diagnostic use.

    Args:
        x: (B, N, d) particle positions.
        w: (B, N) source weights.
        eps: scalar or (B,) regularization parameter.
        n_iters: Sinkhorn iterations.
        return_residuals: if True, also return (row_res, col_res).
        use_float64: cast to float64 for the Sinkhorn solve (recommended).

    Returns:
        y: (B, N, d) target positions (uniform weights implied).
        T: (B, N, N) optimal coupling.
        (optional) row_res, col_res: (B,) each.
    """
    out_dtype = x.dtype
    if use_float64:
        x_ = tf.cast(x, tf.float64)
        w_ = tf.cast(w, tf.float64)
        eps_ = tf.cast(eps, tf.float64)
    else:
        x_, w_, eps_ = x, w, eps

    B = tf.shape(x_)[0]
    N = tf.shape(x_)[1]
    N_f = tf.cast(N, x_.dtype)

    C = pairwise_sq_dist(x_)
    u = tf.fill((B, N), tf.cast(1.0, x_.dtype) / N_f)
    T = sinkhorn_log(C, w_, u, eps_, n_iters=n_iters)

    y = N_f * tf.matmul(T, x_, transpose_a=True)

    if return_residuals:
        row_res, col_res = marginal_residuals(T, w_, u)
        return (tf.cast(y, out_dtype),
                tf.cast(T, out_dtype),
                tf.cast(row_res, out_dtype),
                tf.cast(col_res, out_dtype))
    return tf.cast(y, out_dtype), tf.cast(T, out_dtype)


def etpf_targets_normalized(x, w, eps, n_iters=1000, return_residuals=False,
                            return_scale_center=False):
    """ETPF in NORMALIZED space. This is the semantic match to Corenflos et al.

    Pipeline:
      1. Normalize: x_tilde = (x - center) / scale,  scale = weighted std.
      2. Run Sinkhorn between (x_tilde, w) and (x_tilde, uniform).
      3. Target y_tilde = N * T^T x_tilde, in normalized space.

    Here eps is the Corenflos-style regularization: it interacts with the
    normalized cost matrix (max C ~ O(1)-O(10) depending on cloud shape),
    making eps=0.25..0.75 the natural range, independent of cloud scale.

    The returned y_tilde is directly comparable to what the operator's
    normalized-space MGN output predicts. Training loss should compare
    in normalized space.

    Args:
        x: (B, N, d) particle positions (arbitrary scale).
        w: (B, N) source weights.
        eps: scalar or (B,) regularization parameter in Corenflos units.
        n_iters: Sinkhorn iterations (default 1000 is conservative).
        return_residuals: include marginal residuals for diagnostics.
        return_scale_center: also return (center, scale) so caller can
            denormalize if needed.

    Dtype note: runs entirely in float32. We verified earlier that fp32
    and fp64 produce identical row/col residuals for normalized clouds at
    eps >= 0.1 (the kernel-form vs log-form experiment). fp32 saturates
    A100's tensor cores and gives ~4x throughput over fp64.

    Returns:
        y_tilde: (B, N, d) target in NORMALIZED space. uniform weights.
        T:       (B, N, N) coupling.
        (optional) row_res, col_res.
        (optional) center, scale (in original dtype).
    """
    out_dtype = x.dtype
    # Normalize in original dtype (cheap, stable).
    x_tilde, center, scale = weighted_normalize(x, w)

    # Cast everything to float32 for Sinkhorn solve.
    x_tilde_f = tf.cast(x_tilde, tf.float32)
    w_f = tf.cast(w, tf.float32)
    eps_f = tf.cast(eps, tf.float32)

    B = tf.shape(x_tilde_f)[0]
    N = tf.shape(x_tilde_f)[1]
    N_f = tf.cast(N, tf.float32)

    C_tilde = pairwise_sq_dist(x_tilde_f)
    u = tf.fill((B, N), tf.cast(1.0, tf.float32) / N_f)
    T = sinkhorn_log(C_tilde, w_f, u, eps_f, n_iters=n_iters)

    y_tilde_f = N_f * tf.matmul(T, x_tilde_f, transpose_a=True)
    y_tilde = tf.cast(y_tilde_f, out_dtype)

    extras = []
    if return_residuals:
        row_res, col_res = marginal_residuals(T, w_f, u)
        extras.extend([tf.cast(row_res, out_dtype),
                       tf.cast(col_res, out_dtype)])
    if return_scale_center:
        extras.extend([center, scale])

    if extras:
        return (y_tilde, tf.cast(T, out_dtype), *extras)
    return y_tilde, tf.cast(T, out_dtype)


# ---------------------------------------------------------------------------
# Convergence diagnostic for precompute
# ---------------------------------------------------------------------------
def summarize_convergence(row_res_all, col_res_all, tol=1e-4):
    """Summary statistics of marginal residuals across many clouds.

    Args:
        row_res_all: 1-D numpy array of row residuals (one per cloud).
        col_res_all: 1-D numpy array of column residuals.
        tol: threshold above which a cloud is flagged as "not converged".

    Returns:
        dict with keys: max_row, max_col, mean_row, mean_col,
                        n_not_converged, frac_not_converged.
    """
    import numpy as np
    total_res = np.asarray(row_res_all) + np.asarray(col_res_all)
    n_total = len(total_res)
    n_bad = int(np.sum(total_res > tol))
    return {
        'max_row': float(np.max(row_res_all)),
        'max_col': float(np.max(col_res_all)),
        'mean_row': float(np.mean(row_res_all)),
        'mean_col': float(np.mean(col_res_all)),
        'n_total': n_total,
        'n_not_converged': n_bad,
        'frac_not_converged': n_bad / max(n_total, 1),
        'tol': tol,
    }

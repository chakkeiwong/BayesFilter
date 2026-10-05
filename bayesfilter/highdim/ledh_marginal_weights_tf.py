"""Exact Gaussian marginal importance ratios and analytical total tangents.

Optional method extension for one-child-per-ancestor affine LEDH proposals.
This is not an admission authority and does not change the default weight law.
No random label is sampled here: the caller must retain its outer ancestor
weight. Full-rank Gaussian transitions and innovation-independent affine maps
are prerequisites. Covariances are checked, never silently regularized.
"""
from __future__ import annotations

import math
import tensorflow as tf

import bayesfilter.highdim.ledh_small_matrix_tf as small_matrix
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

MARGINAL_WEIGHT_POLICY_ID = "ledh_exact_gaussian_marginal_weights_v1"


def gaussian_mixture_log_density_tangent(
    points, d_points, means, d_means, covariances, d_covariances,
    log_weights, d_log_weights,
):
    """Log sum_j w_j N(x_i;mu_j,S_j), with all input dependence included.

    Inputs have shapes [N,d], [N,d,d], [N]. A single analytical direction is
    carried at a time, like the shared LEDH executor. Exact equal row/column
    tiles obey the repository transport chunk policy. The returned validity
    flags refer to the complete mixture, including every component.
    """
    n, d = int(points.shape[0]), int(points.shape[1])
    policy = select_transport_chunks(n)
    # The selector is evaluated outside tracing, with no caller override.
    k = policy.row_chunk_size
    tiles = n // k
    dtype = points.dtype
    valid, chol = safe_cholesky(covariances, "marginal_component")
    precision = small_matrix.cholesky_solve(
        chol, tf.broadcast_to(tf.eye(d, dtype=dtype), [n, d, d]))
    normalizers = (-tf.constant(.5*d*math.log(2*math.pi), dtype)
                   - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1))
    trace_term = tf.einsum("nij,nji->n", precision, d_covariances)
    value_valid = (tf.reduce_all(valid) & tf.reduce_all(tf.math.is_finite(points))
                   & tf.reduce_all(tf.math.is_finite(means))
                   & tf.reduce_all(tf.math.is_finite(log_weights)))
    force_valid = (value_valid & tf.reduce_all(tf.math.is_finite(d_points))
                   & tf.reduce_all(tf.math.is_finite(d_means))
                   & tf.reduce_all(tf.math.is_finite(d_covariances))
                   & tf.reduce_all(tf.math.is_finite(d_log_weights)))
    out = tf.TensorArray(dtype, size=tiles, element_shape=[k])
    dout = tf.TensorArray(dtype, size=tiles, element_shape=[k])

    def row_body(row, out, dout):
        x = tf.slice(points, [row*k, 0], [k, d])
        dx = tf.slice(d_points, [row*k, 0], [k, d])

        def column_body(col, total, dtotal):
            start = col*k
            # Explicit constant tile extents are required by XLA. Dynamic
            # Python slice endpoints may compile to an N-sized slice.
            component_means = tf.slice(means, [start, 0], [k, d])
            component_d_means = tf.slice(d_means, [start, 0], [k, d])
            component_precision = tf.slice(precision, [start, 0, 0], [k, d, d])
            component_d_cov = tf.slice(d_covariances, [start, 0, 0], [k, d, d])
            component_log_weights = tf.slice(log_weights, [start], [k])
            component_normalizers = tf.slice(normalizers, [start], [k])
            component_d_log_weights = tf.slice(d_log_weights, [start], [k])
            component_trace = tf.slice(trace_term, [start], [k])
            residual = x[:, None, :] - component_means[None]
            d_residual = dx[:, None, :] - component_d_means[None]
            u = tf.einsum("cij,rcj->rci", component_precision, residual)
            log_terms = (component_log_weights[None] + component_normalizers[None]
                         - .5*tf.reduce_sum(residual*u, axis=-1))
            d_terms = (component_d_log_weights[None]
                       - tf.reduce_sum(u*d_residual, axis=-1)
                       + .5*tf.einsum("rci,cij,rcj->rc", u, component_d_cov, u)
                       - .5*component_trace[None])
            block = tf.reduce_logsumexp(log_terms, axis=1)
            dblock = tf.reduce_sum(tf.nn.softmax(log_terms, axis=1)*d_terms, axis=1)
            joined = tf.reduce_logsumexp(tf.stack([total, block]), axis=0)
            djoined = tf.exp(total-joined)*dtotal + tf.exp(block-joined)*dblock
            return col+1, joined, djoined

        _, values, tangents = tf.while_loop(
            lambda col, *_: col < tiles, column_body,
            (tf.constant(0), tf.fill([k], tf.cast(-float("inf"), dtype)), tf.zeros([k], dtype)),
            parallel_iterations=1)
        return row+1, out.write(row, values), dout.write(row, tangents)

    _, out, dout = tf.while_loop(lambda row, *_: row < tiles, row_body,
                                (tf.constant(0), out, dout), parallel_iterations=1)
    values, tangents = out.concat(), dout.concat()
    value_valid &= tf.reduce_all(tf.math.is_finite(values))
    force_valid &= value_valid & tf.reduce_all(tf.math.is_finite(tangents))
    return values, tangents, value_valid, force_valid


def marginal_prior_ratio_tangent(
    points, d_points, anchors, d_anchors, process_covariance, d_process_covariance,
    proposal_means, d_proposal_means, affine_matrix, d_affine_matrix,
    log_weights, d_log_weights,
):
    """Return log p_N(x)-log q_N(x), its total tangent, and validity.

    ``proposal_means`` is the auxiliary endpoint F_j(m_j) computed by the
    *same* flow loop. The pushed Gaussian covariance is B_j Q B_j'.
    """
    n, d = int(points.shape[0]), int(points.shape[1])
    q = tf.broadcast_to(process_covariance, [n, d, d])
    dq = tf.broadcast_to(d_process_covariance, [n, d, d])
    b, db = affine_matrix, d_affine_matrix
    bq = tf.linalg.matmul(b, q)
    s = tf.linalg.matmul(bq, b, transpose_b=True)
    ds = (tf.linalg.matmul(tf.linalg.matmul(db, q) + tf.linalg.matmul(b, dq), b, transpose_b=True)
          + tf.linalg.matmul(bq, db, transpose_b=True))
    p, dp, pv, pf = gaussian_mixture_log_density_tangent(
        points, d_points, anchors, d_anchors, q, dq, log_weights, d_log_weights)
    proposal, dproposal, qv, qf = gaussian_mixture_log_density_tangent(
        points, d_points, proposal_means, d_proposal_means, s, ds, log_weights, d_log_weights)
    return p-proposal, dp-dproposal, pv & qv, pf & qf

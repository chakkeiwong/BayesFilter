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

IMPORTANCE_WEIGHT_POLICIES = ("ancestor", "marginal_mixture")
MARGINAL_WEIGHT_POLICY_ID = "ledh_exact_gaussian_marginal_weights_v1"


def gaussian_mixture_log_density_tangent(
    points, d_points, means, d_means, covariances, d_covariances,
    log_weights, d_log_weights, *, component_policy="marginal_mixture",
):
    """Return log density, its total tangent, and value/force validity.

    Inputs are [N,d], [N,d,d] and [N]. ``ancestor`` selects component i for
    point i; ``marginal_mixture`` selects every component for every point.
    Weights need not be uniform. Their tangents are included in both modes.
    """
    if component_policy not in IMPORTANCE_WEIGHT_POLICIES:
        raise ValueError(f"unsupported importance weight policy: {component_policy}")
    n, d = int(points.shape[0]), int(points.shape[1])
    dtype = points.dtype
    valid, chol = safe_cholesky(covariances, "importance_component")
    precision = small_matrix.cholesky_solve(chol, tf.broadcast_to(tf.eye(d, dtype=dtype), [n,d,d]))
    normalizers = (-tf.constant(.5*d*math.log(2*math.pi), dtype)
                   -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1))
    trace = tf.einsum("nij,nji->n", precision, d_covariances)
    value_valid = (tf.reduce_all(valid) & tf.reduce_all(tf.math.is_finite(points))
                   & tf.reduce_all(tf.math.is_finite(means))
                   & tf.reduce_all(tf.math.is_finite(log_weights)))
    force_valid = (value_valid & tf.reduce_all(tf.math.is_finite(d_points))
                   & tf.reduce_all(tf.math.is_finite(d_means))
                   & tf.reduce_all(tf.math.is_finite(d_covariances))
                   & tf.reduce_all(tf.math.is_finite(d_log_weights)))

    def reduce_components(x, dx, mu, dmu, precision, dcov, lw, dlw, norm, trace):
        residual, d_residual = x-mu, dx-dmu
        u = tf.linalg.matvec(precision, residual)
        terms = lw+norm-.5*tf.reduce_sum(residual*u, axis=-1)
        dterms = (dlw-tf.reduce_sum(u*d_residual, axis=-1)
                  +.5*tf.reduce_sum(u*tf.linalg.matvec(dcov,u), axis=-1)-.5*trace)
        return (tf.reduce_logsumexp(terms, axis=-1),
                tf.reduce_sum(tf.nn.softmax(terms, axis=-1)*dterms, axis=-1))

    if component_policy == "ancestor":
        value, tangent = reduce_components(
            points[:,None], d_points[:,None], means[:,None], d_means[:,None],
            precision[:,None], d_covariances[:,None], log_weights[:,None],
            d_log_weights[:,None], normalizers[:,None], trace[:,None])
    else:
        k = select_transport_chunks(n).row_chunk_size
        tiles = n//k
        out = tf.TensorArray(dtype, size=tiles, element_shape=[k])
        dout = tf.TensorArray(dtype, size=tiles, element_shape=[k])

        def row_body(row, out, dout):
            x = tf.slice(points, [row*k,0], [k,d])[:,None]
            dx = tf.slice(d_points, [row*k,0], [k,d])[:,None]

            def column_body(col, total, dtotal):
                start = col*k
                block, dblock = reduce_components(x, dx,
                    tf.slice(means,[start,0],[k,d])[None],
                    tf.slice(d_means,[start,0],[k,d])[None],
                    tf.slice(precision,[start,0,0],[k,d,d])[None],
                    tf.slice(d_covariances,[start,0,0],[k,d,d])[None],
                    tf.slice(log_weights,[start],[k])[None],
                    tf.slice(d_log_weights,[start],[k])[None],
                    tf.slice(normalizers,[start],[k])[None],
                    tf.slice(trace,[start],[k])[None])
                joined = tf.stack([total,block], axis=-1)
                updated = tf.reduce_logsumexp(joined, axis=-1)
                dupdated = tf.reduce_sum(tf.nn.softmax(joined, axis=-1)
                                         *tf.stack([dtotal,dblock], axis=-1), axis=-1)
                return col+1, updated, dupdated

            _, value, tangent = tf.while_loop(lambda col,*_:col<tiles, column_body,
                (tf.constant(0), tf.fill([k],tf.constant(-math.inf,dtype)), tf.zeros([k],dtype)))
            return row+1, out.write(row,value), dout.write(row,tangent)

        _, out, dout = tf.while_loop(lambda row,*_:row<tiles, row_body, (tf.constant(0),out,dout))
        value, tangent = out.concat(), dout.concat()
    value_valid &= tf.reduce_all(tf.math.is_finite(value))
    force_valid &= value_valid & tf.reduce_all(tf.math.is_finite(tangent))
    return value, tangent, value_valid, force_valid


def marginal_prior_ratio_tangent(
    points, d_points, anchors, d_anchors, process_covariance, d_process_covariance,
    proposal_means, d_proposal_means, affine_matrix, d_affine_matrix,
    log_weights, d_log_weights, *, component_policy="marginal_mixture",
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
        points, d_points, anchors, d_anchors, q, dq, log_weights, d_log_weights,
        component_policy=component_policy)
    proposal, dproposal, qv, qf = gaussian_mixture_log_density_tangent(
        points, d_points, proposal_means, d_proposal_means, s, ds, log_weights, d_log_weights,
        component_policy=component_policy)
    return p-proposal, dp-dproposal, pv & qv, pf & qf

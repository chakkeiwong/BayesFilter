"""Exact Gaussian-mixture analytical scores through weighted moments.

The pairwise log densities have shape [B,K,K]. Parameter directions enter
only component features and their weighted sums, never a [P,B,K,K] tensor.
This is the nonshared-covariance part of the same finite mixture, not a
different approximation. See the 2026-10-21 optimization plan for derivation.
"""
import math

import tensorflow as tf

from bayesfilter.highdim.analytical_jet_tf import Jet, logsumexp, stack
from bayesfilter.highdim.covariance_proposal_score_tf import tile
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim import ledh_small_matrix_tf as sm
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks


def nonshared_gaussian_mixture(x, means, covariance, log_weights=None):
    """Optional differentiable log masses; caller owns normalization."""
    batch, n, d = x.v.shape
    p = x.t.shape[0]
    dtype = x.v.dtype
    k = select_transport_chunks(n).row_chunk_size
    blocks = n//k
    valid, factor = safe_cholesky(covariance.v, "importance_component")
    precision = sm.cholesky_solve(factor,
        tf.broadcast_to(tf.eye(d, dtype=dtype), tf.shape(factor)))
    normal = (-tf.constant(.5*d*math.log(2*math.pi)+math.log(n), dtype)
              -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)), -1))
    if log_weights is not None:
        if log_weights.v.shape != x.v.shape[:2]:
            raise ValueError("component log masses must have shape [B,N]")
        normal = normal + tf.constant(math.log(n), dtype) + log_weights.v
    # Fixed common origins cancel in every difference and its derivative.
    xc = x.v-x.v[:, :1]
    mc = means.v-x.v[:, :1]
    dx = x.t-x.t[:, :, :1]
    dm = means.t-x.t[:, :, :1]
    pm = tf.linalg.matvec(precision, mc)
    pdm = tf.linalg.matvec(precision[None], dm)
    h = tf.linalg.matmul(precision[None],
        tf.linalg.matmul(covariance.t, precision[None]), transpose_a=True)
    h = .5*(h+tf.linalg.matrix_transpose(h))
    hm = tf.linalg.matvec(h, mc[None])
    trace = tf.einsum("bnij,pbnji->pbn", precision, covariance.t)

    def pack(t):
        return tf.reshape(tf.transpose(t, [1, 2, 0, *range(3, t.shape.rank)]),
                          [batch, n, -1])

    # Features depend on components, not on evaluation points or tile pairs.
    features = tf.concat([
        tf.reshape(precision, [batch, n, d*d]), pm,
        pack(pdm), pack(tf.reduce_sum(mc[None]*pdm, -1)),
        pack(h), pack(hm),
        pack(tf.reduce_sum(mc[None]*hm, -1)-trace)], -1)
    if log_weights is not None:
        features = tf.concat([features, pack(log_weights.t)], -1)
    out = tf.TensorArray(dtype, blocks, element_shape=[batch, k])
    dout = tf.TensorArray(dtype, blocks, element_shape=[p, batch, k])

    def rows(i, out, dout):
        xv = tile(xc, i, k, 1)
        xt = tile(dx, i, k, 2)
        xx = tf.reshape(xv[..., :, None]*xv[..., None, :], [batch, k, d*d])

        def cols(j, total, dtotal):
            mu = tile(mc, j, k, 1)
            pr = tile(precision, j, k, 1)
            # Direct differences avoid cancellation in expanded quadratic
            # distances. Stream state coordinates instead of retaining K*K*D.
            def outer(a, quadratic):
                da = xv[:, :, a, None]-mu[:, None, :, a]
                def inner(b, u):
                    db = xv[:, :, b, None]-mu[:, None, :, b]
                    return b+1, u+pr[:, None, :, a, b]*db
                _, u = tf.while_loop(lambda b, _: b < d, inner,
                    (0, tf.zeros([batch, k, k], dtype)), parallel_iterations=1)
                return a+1, quadratic+da*u
            _, quadratic = tf.while_loop(lambda a, _: a < d, outer,
                (0, tf.zeros([batch, k, k], dtype)), parallel_iterations=1)
            terms = tile(normal, j, k, 1)[:, None]-.5*quadratic
            value = tf.reduce_logsumexp(terms, -1)
            weights = tf.nn.softmax(terms, -1)
            moments = tf.linalg.matmul(weights, tile(features, j, k, 1))
            cursor = 0
            def take(width):
                nonlocal cursor
                value = moments[..., cursor:cursor+width]
                cursor += width
                return value
            ep = tf.reshape(take(d*d), [batch, k, d, d])
            epm = take(d)
            def directions(width):
                return tf.transpose(tf.reshape(take(p*width), [batch, k, p, width]),
                                    [2, 0, 1, 3])
            epdm = directions(d)
            empdm = directions(1)[..., 0]
            eh = directions(d*d)
            ehm = directions(d)
            emhm_trace = directions(1)[..., 0]
            derivative = (-tf.reduce_sum(xt*(tf.linalg.matvec(ep, xv)-epm)[None], -1)
                          +tf.reduce_sum(xv[None]*epdm, -1)-empdm
                          +.5*(tf.reduce_sum(xx[None]*eh, -1)
                                -2*tf.reduce_sum(xv[None]*ehm, -1)+emhm_trace))
            if log_weights is not None:
                derivative = derivative + directions(1)[..., 0]
            joined = logsumexp(stack([Jet(total, dtotal), Jet(value, derivative)], -1), -1)
            return j+1, joined.v, joined.t

        _, v, t = tf.while_loop(lambda j, *_: j < blocks, cols,
            (0, tf.fill([batch, k], tf.constant(-math.inf, dtype)),
             tf.zeros([p, batch, k], dtype)), parallel_iterations=1)
        return i+1, out.write(i, v), dout.write(i, t)

    _, out, dout = tf.while_loop(lambda i, *_: i < blocks, rows,
        (0, out, dout), parallel_iterations=1)
    value = tf.reshape(tf.transpose(out.stack(), [1, 0, 2]), [batch, n])
    tangent = tf.reshape(tf.transpose(dout.stack(), [1, 2, 0, 3]), [p, batch, n])
    valid = (tf.reduce_all(valid, 1) & tf.reduce_all(tf.math.is_finite(value), 1)
             & tf.reduce_all(tf.math.is_finite(tangent), [0, 2])
             & tf.reduce_all(tf.math.is_finite(x.v), [1, 2])
             & tf.reduce_all(tf.math.is_finite(means.v), [1, 2]))
    return Jet(value, tangent), valid

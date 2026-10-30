"""Exact tiled finite-Sinkhorn barycentres and analytical total tangents.

This is the memory-blocked evaluation of covariance_proposal_tf's dense reset,
with identical finite iteration order and actual column normalization. It does
not use the residual-injection Contract-E reset or an implicit OT derivative.
For L=log(sum(exp(a))), dL=sum(softmax(a)*da); combining block log sums uses
this same identity. Thus every finite row/column update is differentiated.
"""
import math
import tensorflow as tf
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks


def tiled_sinkhorn_barycentres(x, dx, logw, dlogw, controls):
    """Return y, dy, row-mass error, and normalized-column error, using KxK tiles."""
    n, d = x.shape
    k = select_transport_chunks(n).row_chunk_size
    blocks = n // k
    dtype = x.dtype
    epsilon = tf.cast(controls.reset_epsilon, dtype)

    def tile(i, j):
        xi, xj = tf.slice(x, [i*k, 0], [k, d]), tf.slice(x, [j*k, 0], [k, d])
        dxi, dxj = tf.slice(dx, [i*k, 0], [k, d]), tf.slice(dx, [j*k, 0], [k, d])
        delta, ddelta = xi[:, None]-xj[None], dxi[:, None]-dxj[None]
        return (-tf.reduce_sum(delta*delta, -1)/epsilon,
                -2*tf.reduce_sum(delta*ddelta, -1)/epsilon)

    def reduce_tiles(potential, dpotential, rows):
        values = tf.TensorArray(dtype, size=blocks, element_shape=[k])
        derivs = tf.TensorArray(dtype, size=blocks, element_shape=[k])

        def outer(i, values, derivs):
            def inner(j, total, dtotal):
                cost, dcost = tile(i, j) if rows else tile(j, i)
                p = tf.slice(potential, [j*k], [k])
                dp = tf.slice(dpotential, [j*k], [k])
                terms = cost + (p[None] if rows else p[:, None])
                dterms = dcost + (dp[None] if rows else dp[:, None])
                axis = 1 if rows else 0
                part = tf.reduce_logsumexp(terms, axis)
                dpart = tf.reduce_sum(tf.nn.softmax(terms, axis=axis)*dterms, axis)
                joined = tf.stack([total, part], -1)
                return (j+1, tf.reduce_logsumexp(joined, -1),
                        tf.reduce_sum(tf.nn.softmax(joined, -1)*tf.stack([dtotal, dpart], -1), -1))

            _, value, derivative = tf.while_loop(
                lambda j, *_: j < blocks, inner,
                (tf.constant(0), tf.fill([k], tf.constant(-math.inf, dtype)), tf.zeros([k], dtype)),
                parallel_iterations=1)
            return i+1, values.write(i, value), derivs.write(i, derivative)

        _, values, derivs = tf.while_loop(lambda i, *_: i < blocks, outer,
                                        (tf.constant(0), values, derivs), parallel_iterations=1)
        return tf.ensure_shape(values.concat(), [n]), tf.ensure_shape(derivs.concat(), [n])

    logn = tf.math.log(tf.cast(n, dtype))
    zero = tf.zeros([n], dtype)

    def iteration(t, gamma, dgamma, alpha, dalpha):
        rows, drows = reduce_tiles(gamma, dgamma, True)
        alpha, dalpha = logw-rows, dlogw-drows
        columns, dcolumns = reduce_tiles(alpha, dalpha, False)
        return t+1, -logn-columns, -dcolumns, alpha, dalpha

    _, gamma, dgamma, alpha, dalpha = tf.while_loop(
        lambda t, *_: t < controls.reset_steps, iteration,
        (tf.constant(0), zero, zero, zero, zero), parallel_iterations=1)
    # gamma is constant within a column and cancels from the ACTUAL column
    # normalization, including its tangent. Do not assume convergence or 1/N
    # column mass; compute the normalizer for the finite final alpha explicitly.
    columns, dcolumns = reduce_tiles(alpha, dalpha, False)
    rows, _ = reduce_tiles(gamma, dgamma, True)
    row_error = tf.reduce_max(tf.abs(tf.exp(alpha+rows)-tf.exp(logw)))
    ys = tf.TensorArray(dtype, size=blocks, element_shape=[k, d])
    dys = tf.TensorArray(dtype, size=blocks, element_shape=[k, d])

    def column(j, ys, dys, column_error):
        normalizer = tf.slice(columns, [j*k], [k])
        dnormalizer = tf.slice(dcolumns, [j*k], [k])

        def row(i, y, dy, mass):
            cost, dcost = tile(i, j)
            a, da = tf.slice(alpha, [i*k], [k]), tf.slice(dalpha, [i*k], [k])
            trans = tf.exp(cost+a[:, None]-normalizer[None])
            dtrans = trans*(dcost+da[:, None]-dnormalizer[None])
            xi, dxi = tf.slice(x, [i*k, 0], [k, d]), tf.slice(dx, [i*k, 0], [k, d])
            y += tf.einsum('ij,id->jd', trans, xi)
            dy += tf.einsum('ij,id->jd', dtrans, xi)+tf.einsum('ij,id->jd', trans, dxi)
            return i+1, y, dy, mass+tf.reduce_sum(trans, 0)

        _, y, dy, mass = tf.while_loop(lambda i, *_: i < blocks, row,
            (tf.constant(0), tf.zeros([k, d], dtype), tf.zeros([k, d], dtype), tf.zeros([k], dtype)),
            parallel_iterations=1)
        return j+1, ys.write(j, y), dys.write(j, dy), tf.maximum(column_error, tf.reduce_max(tf.abs(mass-1)))

    _, ys, dys, column_error = tf.while_loop(lambda j, *_: j < blocks, column,
        (tf.constant(0), ys, dys, tf.constant(0., dtype)), parallel_iterations=1)
    return tf.ensure_shape(ys.concat(), [n, d]), tf.ensure_shape(dys.concat(), [n, d]), row_error, column_error

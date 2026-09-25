"""Bounded full-coefficient Gaussian projection reference, d<=4; not a runtime route.

All numerics use TensorFlow. The full (degree+1)^(2*d) coefficient tensor is
deliberately diagnostic. A04 derives the Gaussian amplitude recurrence.
"""
from functools import lru_cache
import tensorflow as tf

D = tf.float64


@lru_cache(None)
def coefficient_kernel(coordinates, degree=3, jit_compile=True):
    """Hermite amplitude coefficients by total-degree recurrence; one traced loop."""
    n = degree + 1
    count = n ** coordinates
    strides = tf.constant([n**k for k in range(coordinates-1, -1, -1)], tf.int32)
    index = tf.range(count, dtype=tf.int32)
    alpha = (index[:, None] // strides[None, :]) % n
    total = tf.reduce_sum(alpha, axis=1)
    pivot = tf.argmax(tf.cast(alpha > 0, tf.int32), axis=1, output_type=tf.int32)
    beta = alpha - tf.one_hot(pivot, coordinates, dtype=tf.int32)
    parent = tf.maximum(index - tf.gather(strides, pivot), 0)
    grandparent = tf.maximum(parent[:, None] - strides[None, :], 0)
    denominator = tf.cast(tf.maximum(tf.gather(alpha, pivot, batch_dims=1), 1), D)
    multiplier = tf.sqrt(tf.cast(tf.maximum(beta, 0), D) / denominator[:, None])

    @tf.function(input_signature=[tf.TensorSpec([coordinates], D),
                                  tf.TensorSpec([coordinates, coordinates], D)],
                 jit_compile=jit_compile, autograph=False)
    def coefficients(mean, covariance):
        identity = tf.eye(coordinates, dtype=D)
        precision = tf.linalg.solve(covariance, identity)
        V = 2. * tf.linalg.solve(precision + identity, identity)
        h = .5 * tf.linalg.matvec(V, tf.linalg.matvec(precision, mean))
        logM = (-.25*tf.linalg.logdet(covariance) + .5*tf.linalg.logdet(V)
                -.25*tf.tensordot(mean, tf.linalg.matvec(precision, mean), 1)
                +.25*tf.tensordot(h, tf.linalg.matvec(precision+identity, h), 1))
        weights = tf.gather(V-identity, pivot) * multiplier
        linear = tf.gather(h, pivot)/tf.sqrt(denominator)
        e = tf.one_hot(0, count, dtype=D)
        def body(k, values):
            update = linear*tf.gather(values, parent) + tf.reduce_sum(
                weights*tf.gather(values, grandparent), axis=1)
            return k+1, tf.where(total == k, update, values)
        _, e = tf.while_loop(lambda k, _: k <= coordinates*degree, body,
                            (tf.constant(1), e), parallel_iterations=1)
        return tf.reshape(tf.exp(logM)*e, [n]*coordinates), logM
    return coefficients


def paired_gaussian(model, current, condition, *, predictive=False,
                    guide_current=None, guide_condition=None):
    """Mean/covariance in interleaved current/previous marginal coordinates."""
    A = model.transition
    guide_current = current if guide_current is None else guide_current
    guide_condition = condition if guide_condition is None else guide_condition
    P = guide_condition.factor @ tf.transpose(guide_condition.factor)
    S = A @ P @ tf.transpose(A) + model.sigma**2*tf.eye(model.dimension, dtype=D)
    K = tf.transpose(tf.linalg.solve(S, A @ P))
    B = P-K @ S @ tf.transpose(K)
    qmean = tf.linalg.matvec(A, guide_condition.mean) if predictive else guide_current.mean
    qcov = S if predictive else guide_current.factor @ tf.transpose(guide_current.factor)
    zmean = guide_condition.mean + tf.linalg.matvec(K, qmean-tf.linalg.matvec(A, guide_condition.mean))
    means = tf.concat([qmean, zmean], axis=0)
    covariance = tf.concat([tf.concat([qcov, qcov @ tf.transpose(K)], axis=1),
        tf.concat([K @ qcov, B + K @ qcov @ tf.transpose(K)], axis=1)], axis=0)
    zeros = tf.zeros_like(current.factor)
    L = tf.concat([tf.concat([current.factor, zeros], axis=1),
                   tf.concat([zeros, condition.factor], axis=1)], axis=0)
    mean = tf.linalg.triangular_solve(L, (means-tf.concat([current.mean, condition.mean], 0))[:, None])[:, 0]
    inverse = tf.linalg.triangular_solve(L, tf.eye(2*model.dimension, dtype=D))
    covariance = inverse @ covariance @ tf.transpose(inverse)
    order = [j for i in range(model.dimension) for j in (i, i+model.dimension)]
    return tf.gather(mean, order), tf.gather(tf.gather(covariance, order), order, axis=1)


def reconstruct(cores):
    result = cores[0][0]
    for core in cores[1:]:
        result = tf.tensordot(result, core, [[-1], [0]])
    return result[..., 0]


def pair_svd(coefficients, rank=3):
    """Diagnostic TT-SVD of a full interleaved coefficient tensor."""
    dimension = len(coefficients.shape)//2
    n = int(coefficients.shape[0])
    rest, previous, cores, spectra = coefficients, 1, [], []
    for axis in range(dimension-1):
        matrix = tf.reshape(rest, [previous*n*n, -1])
        singular, left, right = tf.linalg.svd(matrix, full_matrices=False)
        keep = min(rank, int(singular.shape[0]))
        cores.append(tf.reshape(left[:, :keep], [previous, n, n, keep]))
        spectra.append(singular[:keep])
        rest = singular[:keep, None]*tf.transpose(right[:, :keep])
        previous = keep
    cores.append(tf.reshape(rest, [previous, n, n, 1]))
    residual = tf.reduce_sum(tf.square(coefficients-reconstruct(cores)))
    return tuple(cores), residual, spectra

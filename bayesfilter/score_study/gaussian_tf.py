"""Analytical all-parameter Gaussian references and score-study baselines.

These are explicit independent reference/comparator providers. They do not
implement or replace canonical LEDH. Parameters, initial moments and both
noise covariances carry total directional sensitivities. No autodiff or pfor
is used by the value/score kernels.
"""
from __future__ import annotations

from functools import lru_cache
import tensorflow as tf


def symmetric(matrix):
    return (matrix + tf.linalg.matrix_transpose(matrix)) * tf.cast(0.5, matrix.dtype)


def chol_tangent(cholesky, covariance_tangent):
    left = tf.linalg.triangular_solve(cholesky, covariance_tangent)
    whitened = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(cholesky, tf.linalg.matrix_transpose(left)))
    lower = tf.linalg.band_part(whitened, -1, 0)
    lower -= tf.linalg.diag(tf.linalg.diag_part(lower) * tf.cast(0.5, lower.dtype))
    return tf.linalg.matmul(cholesky, lower)


def gaussian_log_density_and_tangent(residual, dresidual, covariance, dcovariance):
    """Residual [...,o], tangent [p,...,o], covariance [...,o,o]."""
    factor = tf.linalg.cholesky(symmetric(covariance))
    inverse_residual = tf.linalg.cholesky_solve(factor, residual[..., None])[..., 0]
    logdet = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)), axis=-1)
    dim = tf.cast(tf.shape(residual)[-1], residual.dtype)
    log_two_pi = tf.math.log(tf.cast(2.0, residual.dtype) * tf.acos(tf.cast(-1.0, residual.dtype)))
    value = -tf.cast(0.5, residual.dtype) * dim * log_two_pi
    value -= tf.cast(0.5, residual.dtype) * (logdet + tf.reduce_sum(residual * inverse_residual, axis=-1))
    trace = tf.linalg.trace(tf.linalg.cholesky_solve(factor, dcovariance))
    quadratic = tf.einsum("...i,p...ij,...j->p...", inverse_residual, dcovariance, inverse_residual)
    tangent = -tf.reduce_sum(dresidual * inverse_residual, axis=-1) - 0.5 * trace + 0.5 * quadratic
    return value, tangent


def affine_prediction(mean, dmean, covariance, dcovariance, A, dA, Q, dQ, *, unscented=False):
    if unscented:
        # Positive spherical rule, alpha=1, kappa=0, beta=0. It is exact for
        # affine first/second moments; this choice is a Gaussian fixture only.
        dimension = mean.shape[-1]
        factor = tf.linalg.cholesky(symmetric(covariance))
        dfactor = chol_tangent(factor, dcovariance)
        directions = tf.concat([tf.eye(dimension, dtype=mean.dtype), -tf.eye(dimension, dtype=mean.dtype)], 0)
        directions *= tf.sqrt(tf.cast(dimension, mean.dtype))
        nodes = mean[None, :] + tf.einsum("ij,nj->ni", factor, directions)
        dnodes = dmean[:, None, :] + tf.einsum("pij,nj->pni", dfactor, directions)
        values = tf.einsum("ij,nj->ni", A, nodes)
        dvalues = tf.einsum("pij,nj->pni", dA, nodes) + tf.einsum("ij,pnj->pni", A, dnodes)
        predicted = tf.reduce_mean(values, axis=0)
        dpredicted = tf.reduce_mean(dvalues, axis=1)
        centered = values - predicted
        dcentered = dvalues - dpredicted[:, None, :]
        count = tf.cast(2 * dimension, mean.dtype)
        P = tf.einsum("ni,nj->ij", centered, centered) / count + Q
        dP = (tf.einsum("pni,nj->pij", dcentered, centered) +
              tf.einsum("ni,pnj->pij", centered, dcentered)) / count + dQ
        return predicted, dpredicted, symmetric(P), symmetric(dP)
    predicted = tf.linalg.matvec(A, mean)
    dpredicted = tf.einsum("pij,j->pi", dA, mean) + tf.einsum("ij,pj->pi", A, dmean)
    P = A @ covariance @ tf.transpose(A) + Q
    dP = dA @ covariance @ tf.transpose(A) + A @ dcovariance @ tf.transpose(A) + A @ covariance @ tf.linalg.matrix_transpose(dA) + dQ
    return predicted, dpredicted, symmetric(P), symmetric(dP)


def condition(mean, dmean, covariance, dcovariance, y, H, dH, R, dR):
    residual = y - tf.linalg.matvec(H, mean)
    dresidual = -tf.einsum("pij,j->pi", dH, mean) - tf.einsum("ij,pj->pi", H, dmean)
    S = symmetric(H @ covariance @ tf.transpose(H) + R)
    dS = symmetric(dH @ covariance @ tf.transpose(H) + H @ dcovariance @ tf.transpose(H) + H @ covariance @ tf.linalg.matrix_transpose(dH) + dR)
    factor = tf.linalg.cholesky(S)
    cross = covariance @ tf.transpose(H)
    dcross = dcovariance @ tf.transpose(H) + covariance @ tf.linalg.matrix_transpose(dH)
    gain = tf.transpose(tf.linalg.cholesky_solve(factor, tf.transpose(cross)))
    dgain = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(factor, tf.linalg.matrix_transpose(dcross - gain @ dS)))
    value, score = gaussian_log_density_and_tangent(residual, dresidual, S, dS)
    updated = mean + tf.linalg.matvec(gain, residual)
    dupdated = dmean + tf.einsum("pij,j->pi", dgain, residual) + tf.einsum("ij,pj->pi", gain, dresidual)
    P = covariance - gain @ S @ tf.transpose(gain)
    dP = dcovariance - dgain @ S @ tf.transpose(gain) - gain @ dS @ tf.transpose(gain) - gain @ S @ tf.linalg.matrix_transpose(dgain)
    return updated, dupdated, symmetric(P), symmetric(dP), value, score


def model_signature(dimension, observation_dimension, parameters, dtype):
    d, o, p = dimension, observation_dimension, parameters
    return [tf.TensorSpec(shape, dtype) for shape in (
        (None, o), (d, d), (p, d, d), (o, d), (p, o, d),
        (d,), (p, d), (d, d), (p, d, d), (d, d), (p, d, d), (o, o), (p, o, o))]


@lru_cache(maxsize=32)
def make_gaussian_kernel(dimension, observation_dimension, parameters, dtype_name="float64", jit_compile=True, unscented=False):
    dtype = tf.as_dtype(dtype_name)

    @tf.function(input_signature=model_signature(dimension, observation_dimension, parameters, dtype), jit_compile=jit_compile)
    def kernel(observations, A, dA, H, dH, mean, dmean, covariance, dcovariance, Q, dQ, R, dR):
        def step(t, m, dm, P, dP, value, score, minimum):
            m, dm, P, dP = affine_prediction(m, dm, P, dP, A, dA, Q, dQ, unscented=unscented)
            m, dm, P, dP, term, derivative = condition(m, dm, P, dP, observations[t], H, dH, R, dR)
            return t + 1, m, dm, P, dP, value + term, score + derivative, tf.minimum(minimum, tf.reduce_min(tf.linalg.eigvalsh(P)))
        result = tf.while_loop(lambda t, *_: t < tf.shape(observations)[0], step,
            (tf.constant(0), mean, dmean, covariance, dcovariance, tf.zeros([], dtype),
             tf.zeros([parameters], dtype), tf.reduce_min(tf.linalg.eigvalsh(covariance))), parallel_iterations=1)
        return result[5], result[6], result[1], result[3], result[7]
    return kernel


def parameterized_model(theta, dimension=2, observation_dimension=1):
    """Six directions: A, log sigma_Q, log sigma_R, H, m0, log sigma_P0.

    The general kernels accept arbitrary tangent tensors; this concrete fixture
    exercises each source of dependence, including a non-diagonal covariance.
    """
    dtype = theta.dtype
    d, o = dimension, observation_dimension
    identity = tf.eye(d, dtype=dtype)
    shift = tf.linalg.diag(tf.fill([max(0, d - 1)], tf.cast(0.08, dtype)), k=1, num_rows=d, num_cols=d) if d > 1 else tf.zeros([1, 1], dtype)
    A = theta[0] * identity + shift
    Q = tf.exp(2 * theta[1]) * tf.linalg.diag(tf.linspace(tf.cast(1., dtype), tf.cast(1.3, dtype), d))
    R = tf.exp(2 * theta[2]) * tf.eye(o, dtype=dtype)
    base_H = tf.eye(o, d, dtype=dtype) + tf.fill([o, d], tf.cast(0.15, dtype))
    H = theta[3] * base_H
    direction = tf.linspace(tf.cast(0.7, dtype), tf.cast(1.1, dtype), d)
    mean = theta[4] * direction
    covariance = tf.exp(2 * theta[5]) * (identity + tf.fill([d, d], tf.cast(0.15, dtype)))
    basis = tf.eye(6, dtype=dtype)
    return (A, basis[:, 0, None, None] * identity, H, basis[:, 3, None, None] * base_H,
            mean, basis[:, 4, None] * direction, covariance, 2 * basis[:, 5, None, None] * covariance,
            Q, 2 * basis[:, 1, None, None] * Q, R, 2 * basis[:, 2, None, None] * R)


@lru_cache(maxsize=32)
def make_data_kernel(dimension, observation_dimension, horizon, dtype_name="float64", jit_compile=True):
    dtype = tf.as_dtype(dtype_name)
    d, o, T = dimension, observation_dimension, horizon
    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([2], tf.int32)], jit_compile=jit_compile)
    def kernel(theta, seed):
        A, _, H, _, mean, _, P, _, Q, _, R, _ = parameterized_model(theta, d, o)
        initial_seed = tf.random.experimental.stateless_fold_in(seed, 0)
        x = mean + tf.linalg.matvec(tf.linalg.cholesky(P), tf.random.stateless_normal([d], initial_seed, dtype=dtype))
        process = tf.random.stateless_normal([T, d], tf.random.experimental.stateless_fold_in(seed, 1), dtype=dtype)
        noise = tf.random.stateless_normal([T, o], tf.random.experimental.stateless_fold_in(seed, 2), dtype=dtype)
        LQ, LR = tf.linalg.cholesky(Q), tf.linalg.cholesky(R)
        observations = tf.zeros([T, o], dtype)
        def step(t, x, ys):
            x = tf.linalg.matvec(A, x) + tf.linalg.matvec(LQ, process[t])
            y = tf.linalg.matvec(H, x) + tf.linalg.matvec(LR, noise[t])
            return t + 1, x, tf.tensor_scatter_nd_update(ys, tf.reshape(t, [1, 1]), y[None, :])
        return tf.while_loop(lambda t, *_: t < T, step, (0, x, observations), parallel_iterations=1)[2]
    return kernel


@lru_cache(maxsize=32)
def make_particle_kernel(dimension, observation_dimension, particles, horizon, dtype_name="float64", jit_compile=True, adapted=False, resampling=False):
    """Prior/conditional proposal with optional multinomial resampling.

    The derivative holds discrete ancestor labels locally fixed. It is the
    almost-everywhere derivative of this finite random program, NOT an unbiased
    derivative of its expectation. Common uniforms couple perturbed runs while
    preserving each marginal multinomial law. No-resampling is an explicit SIS
    comparator. Neither baseline is an LEDH variant.
    """
    dtype = tf.as_dtype(dtype_name)
    d, o, N, T = dimension, observation_dimension, particles, horizon
    signature = [tf.TensorSpec([6], dtype), tf.TensorSpec([T, o], dtype),
                 tf.TensorSpec([N, d], dtype), tf.TensorSpec([T, N, d], dtype)]
    if resampling:
        signature.append(tf.TensorSpec([T, N], dtype))
    @tf.function(input_signature=signature, jit_compile=jit_compile)
    def kernel(theta, observations, initial_noise, noises, *uniforms):
        A, dA, H, dH, mean, dmean, P0, dP0, Q, dQ, R, dR = parameterized_model(theta, d, o)
        L0 = tf.linalg.cholesky(P0)
        dL0 = chol_tangent(L0, dP0)
        cloud = mean + tf.einsum("ij,nj->ni", L0, initial_noise)
        dcloud = dmean[:, None, :] + tf.einsum("pij,nj->pni", dL0, initial_noise)
        LQ, dLQ = tf.linalg.cholesky(Q), chol_tangent(tf.linalg.cholesky(Q), dQ)
        # The conditional covariance is shared by all ancestors in this model.
        zeros = tf.zeros([d], dtype)
        dzeros = tf.zeros([6, d], dtype)
        _, _, conditional_P, dconditional_P, _, _ = condition(zeros, dzeros, Q, dQ, tf.zeros([o], dtype), H, dH, R, dR)
        LC = tf.linalg.cholesky(conditional_P)
        dLC = chol_tangent(LC, dconditional_P)
        S = symmetric(H @ Q @ tf.transpose(H) + R)
        dS = symmetric(dH @ Q @ tf.transpose(H) + H @ dQ @ tf.transpose(H) + H @ Q @ tf.linalg.matrix_transpose(dH) + dR)
        LS = tf.linalg.cholesky(S)
        K = tf.transpose(tf.linalg.cholesky_solve(LS, H @ Q))
        dK = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(LS, tf.linalg.matrix_transpose(dQ @ tf.transpose(H) + Q @ tf.linalg.matrix_transpose(dH) - K @ dS)))
        def step(t, x, dx, log_weights, dlog_weights, logZ, dlogZ, ess):
            predicted = tf.einsum("ij,nj->ni", A, x)
            dpredicted = tf.einsum("pij,nj->pni", dA, x) + tf.einsum("ij,pnj->pni", A, dx)
            if adapted:
                residual = observations[t] - tf.einsum("ij,nj->ni", H, predicted)
                dresidual = -tf.einsum("pij,nj->pni", dH, predicted) - tf.einsum("ij,pnj->pni", H, dpredicted)
                x = predicted + tf.einsum("ij,nj->ni", K, residual) + tf.einsum("ij,nj->ni", LC, noises[t])
                dx = dpredicted + tf.einsum("pij,nj->pni", dK, residual) + tf.einsum("ij,pnj->pni", K, dresidual) + tf.einsum("pij,nj->pni", dLC, noises[t])
                terms, dterms = gaussian_log_density_and_tangent(residual, dresidual, S, dS[:, None, :, :])
            else:
                x = predicted + tf.einsum("ij,nj->ni", LQ, noises[t])
                dx = dpredicted + tf.einsum("pij,nj->pni", dLQ, noises[t])
                residual = observations[t] - tf.einsum("ij,nj->ni", H, x)
                dresidual = -tf.einsum("pij,nj->pni", dH, x) - tf.einsum("ij,pnj->pni", H, dx)
                terms, dterms = gaussian_log_density_and_tangent(residual, dresidual, R, dR[:, None, :, :])
            log_weights, dlog_weights = log_weights + terms, dlog_weights + dterms
            weights = tf.nn.softmax(log_weights)
            ess = tf.minimum(ess, 1 / tf.reduce_sum(weights**2))
            if resampling:
                logZ += tf.reduce_logsumexp(log_weights) - tf.math.log(tf.cast(N, dtype))
                dlogZ += tf.reduce_sum(dlog_weights * weights, axis=1)
                # Setting the final cumulative mass to one only fixes floating
                # summation error; the uniforms have support [0,1).
                cdf = tf.concat([tf.cumsum(weights)[:-1], tf.ones([1], dtype)], axis=0)
                ancestors = tf.searchsorted(cdf, uniforms[0][t], side="right")
                x, dx = tf.gather(x, ancestors), tf.gather(dx, ancestors, axis=1)
                log_weights, dlog_weights = tf.zeros_like(log_weights), tf.zeros_like(dlog_weights)
            return t + 1, x, dx, log_weights, dlog_weights, logZ, dlogZ, ess
        result = tf.while_loop(lambda t, *_: t < T, step,
            (0, cloud, dcloud, tf.zeros([N], dtype), tf.zeros([6, N], dtype),
             tf.zeros([], dtype), tf.zeros([6], dtype), tf.cast(N, dtype)), parallel_iterations=1)
        weights = tf.nn.softmax(result[3])
        value = result[5] + tf.reduce_logsumexp(result[3]) - tf.math.log(tf.cast(N, dtype))
        score = result[6] + tf.reduce_sum(result[4] * weights, axis=1)
        ess = result[7]
        return value, score, ess
    return kernel

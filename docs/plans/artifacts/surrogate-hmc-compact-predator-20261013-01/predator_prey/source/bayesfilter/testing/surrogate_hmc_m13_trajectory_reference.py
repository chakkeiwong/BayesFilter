"""Independent diagnostic integration of the M13 latent position trajectory.

The Gaussian approximation is a proposal. Every estimate includes the exact
prior, wrapped observation likelihood, and full defensive-mixture density.
No Laplace likelihood is substituted for the model likelihood.
"""
import math

import tensorflow as tf

F64 = tf.float64
MIXTURE_PRIOR = 1. / 16.
COVARIANCE_EXPANSION = 1.2


def trajectory_prior(fixture, horizon):
    """Exact position marginal: mean, initial covariance, unit-scale Q covariance."""
    a = tf.convert_to_tensor(fixture.A, F64)
    h = tf.eye(4, dtype=F64)[:2]
    powers = [tf.eye(4, dtype=F64)]
    for _ in range(horizon):
        powers.append(tf.linalg.matmul(powers[-1], a))
    initial = tf.concat([tf.linalg.matmul(h, powers[t]) for t in range(1, horizon+1)], axis=0)
    zero = tf.zeros([2, 4], F64)
    noise = tf.concat([tf.concat([
        tf.linalg.matmul(h, powers[t-s]) if s <= t else zero
        for s in range(1, horizon+1)], axis=1)
        for t in range(1, horizon+1)], axis=0)
    q_chol = tf.linalg.cholesky(tf.convert_to_tensor(fixture.Q, F64))
    blocks = tf.reshape(noise, [2*horizon, horizon, 4])
    whitened = tf.reshape(tf.linalg.matmul(blocks, q_chol), [2*horizon, 4*horizon])
    c0 = tf.linalg.matmul(tf.linalg.matmul(initial, fixture.P0), initial, transpose_b=True)
    cq = tf.linalg.matmul(whitened, whitened, transpose_b=True)
    return tf.linalg.matvec(initial, fixture.m0), c0, cq


def gaussian_log_density(samples, mean, chol):
    """Normalized multivariate Gaussian density for an arbitrary sample batch."""
    dimension = mean.shape[0]
    flat = tf.reshape(samples-mean, [-1, dimension])
    whitened = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(flat)))
    values = (-.5*tf.reduce_sum(whitened*whitened, axis=-1)
              - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
              - .5*tf.cast(dimension, F64)*tf.constant(math.log(2*math.pi), F64))
    return tf.reshape(values, tf.shape(samples)[:-1])


def defensive_log_weights(samples, prior_mean, prior_chol, proposal_mean,
                          proposal_chol, log_likelihood):
    prior = gaussian_log_density(samples, prior_mean, prior_chol)
    proposal = gaussian_log_density(samples, proposal_mean, proposal_chol)
    mixture = tf.reduce_logsumexp(tf.stack([
        proposal+tf.constant(math.log1p(-MIXTURE_PRIOR), F64),
        prior+tf.constant(math.log(MIXTURE_PRIOR), F64)], axis=0), axis=0)
    return prior+log_likelihood-mixture


def trajectory_importance_reference(fixture, particles, replicates=16, *, full_support=False):
    """Full-sequence Gaussian proposal, with exact likelihood integration."""
    from bayesfilter.testing.surrogate_hmc_m13_reference import observation_log_density
    if full_support:
        from bayesfilter.testing.surrogate_hmc_m13_wrapped_reference import (
            observation_log_density, angular_terms,
        )

    n, b = int(particles), int(replicates)
    horizon = int(fixture.observations.shape[0])
    if horizon < 2:
        raise ValueError("M13 trajectory comparison requires T>=2")
    r = tf.convert_to_tensor(fixture.R, F64)
    if not bool(tf.reduce_all(r == tf.linalg.diag(tf.linalg.diag_part(r))).numpy()):
        raise ValueError("independent reference requires diagonal R")
    r_base = tf.linalg.diag_part(r)
    observations = tf.convert_to_tensor(fixture.observations, F64)
    priors = {t: trajectory_prior(fixture, t) for t in (1, horizon)}
    pi = tf.constant(math.pi, F64)
    log_n = tf.math.log(tf.cast(n, F64))

    def calculate(theta, seed, length):
        mean, c0, cq = priors[length]
        dimension = 2*length
        covariance = c0+tf.exp(2*theta[0])*cq
        prior_chol = tf.linalg.cholesky(covariance)
        identity = tf.eye(dimension, dtype=F64)
        precision = tf.linalg.cholesky_solve(prior_chol, identity)
        r_variance = r_base*tf.exp(2*theta[1])
        sigma = tf.sqrt(r_variance)
        observed = observations[:length]
        scale_valid = (tf.reduce_all(tf.math.is_finite(covariance))
                       & tf.reduce_all(tf.linalg.diag_part(prior_chol) > 0.)
                       & tf.reduce_all(tf.math.is_finite(sigma))
                       & tf.reduce_all(sigma > 0.))
        if not full_support:
            scale_valid &= sigma[1] <= .080000000000001
        block_selector = tf.eye(length, dtype=F64)[:, :, None, None]

        def objective(z):
            difference = z-mean
            return (.5*tf.reduce_sum(difference*tf.linalg.matvec(precision, difference))
                    - tf.reduce_sum(observation_log_density(tf.reshape(z, [length, 2]),
                                                            observed, sigma)))

        def geometry(z):
            position = tf.reshape(z, [length, 2])
            radius2 = tf.reduce_sum(position*position, axis=-1)
            valid = tf.reduce_all(tf.math.is_finite(radius2) & (radius2 > 1e-20))
            safe = tf.where(radius2 > 1e-20, radius2, tf.ones_like(radius2))
            radius = tf.sqrt(safe+1e-12)
            radial_jacobian = position/radius[:, None]
            angle_jacobian = tf.stack([-position[:, 1], position[:, 0]], axis=-1)/safe[:, None]
            jacobian = tf.stack([radial_jacobian, angle_jacobian], axis=1)
            residual_angle = tf.math.floormod(
                observed[:, 1]-tf.atan2(position[:, 1], position[:, 0])+pi, 2*pi)-pi
            if full_support:
                _, angular_score, angular_valid = angular_terms(residual_angle, sigma[1])
                valid &= tf.reduce_all(angular_valid)
                natural_residual = tf.stack([(observed[:, 0]-radius)/r_variance[0],
                                             -angular_score], axis=1)
                score = tf.linalg.matvec(jacobian, natural_residual, transpose_a=True)
            else:
                aliases = residual_angle[:, None]+2*pi*tf.cast(tf.range(-2, 3), F64)
                alias_weights = tf.nn.softmax(-.5*aliases*aliases/r_variance[1], axis=-1)
                residual = tf.stack([observed[:, 0]-radius,
                                     tf.reduce_sum(alias_weights*aliases, axis=-1)], axis=1)
                score = tf.linalg.matvec(jacobian, residual/r_variance, transpose_a=True)
            gradient = tf.linalg.matvec(precision, z-mean)-tf.reshape(score, [dimension])
            blocks = tf.linalg.matmul(jacobian, jacobian/r_variance[None, :, None], transpose_a=True)
            observation_precision = tf.reshape(tf.transpose(
                block_selector*blocks[:, None, :, :], [0, 2, 1, 3]), [dimension, dimension])
            hessian = precision+observation_precision
            chol = tf.linalg.cholesky(hessian)
            step = tf.squeeze(tf.linalg.cholesky_solve(chol, gradient[:, None]), axis=1)
            decrement = tf.reduce_sum(gradient*step)
            valid = (valid & tf.reduce_all(tf.math.is_finite(chol))
                     & tf.math.is_finite(decrement) & (decrement >= 0.))
            return step, decrement, chol, valid

        def fit(initial):
            def body(iteration, z, done, valid):
                step, decrement, _, good = geometry(z)
                converged = decrement <= 1e-8
                current = objective(z)
                def move():
                    alpha = tf.constant(1., F64)
                    proposal = z-step
                    value = objective(proposal)
                    def backtrack(k, alpha, proposal, value):
                        alpha = .5*alpha
                        proposal = z-alpha*step
                        return k+1, alpha, proposal, objective(proposal)
                    _, alpha, proposal, value = tf.while_loop(
                        lambda k, alpha, proposal, value: (k < 16) & (
                            ~tf.math.is_finite(value) | (value > current-1e-4*alpha*decrement)),
                        backtrack, (tf.constant(0), alpha, proposal, value),
                        parallel_iterations=1)
                    accepted = tf.math.is_finite(value) & (value <= current-1e-4*alpha*decrement)
                    return proposal, accepted
                proposal, accepted = tf.cond(converged,
                    lambda: (z, tf.constant(True)), move)
                return iteration+1, proposal, converged, valid & good & accepted
            iterations, center, done, valid = tf.while_loop(
                lambda i, z, done, valid: (i < 60) & ~done & valid, body,
                (tf.constant(0), initial, tf.constant(False), scale_valid),
                parallel_iterations=1)
            _, decrement, curvature_chol, geometry_ok = geometry(center)
            valid = valid & geometry_ok & (decrement <= 1e-8)
            return center, curvature_chol, valid, iterations, decrement, objective(center)

        observed_positions = tf.stack([observed[:, 0]*tf.cos(observed[:, 1]),
                                       observed[:, 0]*tf.sin(observed[:, 1])], axis=1)
        center, curvature_chol, valid, iterations, decrement, value = fit(mean)
        other, _, other_valid, other_iterations, other_decrement, other_value = fit(
            tf.reshape(observed_positions, [dimension]))
        objective_difference = tf.abs(value-other_value)
        valid = valid & other_valid & (objective_difference <= 1e-7)
        proposal_covariance = COVARIANCE_EXPANSION*tf.linalg.cholesky_solve(curvature_chol, identity)
        proposal_chol = tf.linalg.cholesky(proposal_covariance)
        noise = tf.random.stateless_normal([b, n, dimension], seed, dtype=F64)
        select_seed = tf.stack([seed[0], seed[1]+1])
        select_prior = tf.random.stateless_uniform([b, n, 1], select_seed, dtype=F64) < MIXTURE_PRIOR
        prior_sample = mean+tf.linalg.matmul(noise, prior_chol, transpose_b=True)
        proposal_sample = center+tf.linalg.matmul(noise, proposal_chol, transpose_b=True)
        samples = tf.where(select_prior, prior_sample, proposal_sample)
        observation_values = observation_log_density(
            tf.reshape(samples, [b, n, length, 2]), observed, sigma)
        log_likelihood = tf.reduce_sum(observation_values, axis=-1)
        weights = defensive_log_weights(samples, mean, prior_chol, center, proposal_chol,
                                        log_likelihood)
        log_sum = tf.reduce_logsumexp(weights, axis=1)
        log_estimate = log_sum-log_n
        ess = tf.exp(2*log_sum-tf.reduce_logsumexp(2*weights, axis=1))
        maximum = tf.exp(tf.reduce_max(weights, axis=1)-log_sum)
        valid = (tf.fill([b], valid) & tf.math.is_finite(log_estimate)
                 & tf.reduce_all(tf.math.is_finite(samples), axis=[1, 2])
                 & tf.reduce_all(tf.math.is_finite(weights), axis=1)
                 & tf.math.is_finite(ess) & (ess > 0.))
        diagnostics = tf.stack([tf.cast(iterations, F64), decrement, value,
            tf.cast(other_iterations, F64), other_decrement, other_value,
            objective_difference, tf.reduce_max(tf.abs(center-other)),
            tf.reduce_min(tf.linalg.diag_part(prior_chol)),
            tf.reduce_min(tf.linalg.diag_part(proposal_chol))])
        return log_estimate, ess, maximum, valid, diagnostics, center

    @tf.function(input_signature=[tf.TensorSpec([2], F64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def evaluate(theta, seed):
        first = calculate(theta, seed, 1)
        full = calculate(theta, tf.stack([seed[0], seed[1]+2]), horizon)
        return dict(first=first[0], log_likelihood=full[0],
                    minimum_ess=tf.minimum(first[1], full[1]),
                    valid=first[3] & full[3],
                    proposal_first_diagnostics=first[4],
                    proposal_full_diagnostics=full[4],
                    proposal_first_center=first[5], proposal_full_center=full[5],
                    proposal_maximum_normalized_weight=tf.maximum(first[2], full[2]))
    return evaluate

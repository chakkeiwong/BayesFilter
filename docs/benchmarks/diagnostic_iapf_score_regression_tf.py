"""Independent diagnostic fitting extension; never a production or Eq15 route.

Fit the backward target's coordinate score by a symmetric affine field after
full empirical whitening, then KL-project its Gaussian to diagonal covariance.
"""
from functools import lru_cache
import math

import tensorflow as tf


def gaussian_log(residual, covariance):
    d = residual.shape[-1]
    factor = tf.linalg.cholesky(covariance)
    standardized = tf.linalg.triangular_solve(factor, tf.transpose(residual))
    return (-.5*tf.reduce_sum(standardized**2, axis=0)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))
            -.5*tf.constant(d*math.log(2*math.pi), residual.dtype))


def target_log_score(x, y, A, H, Q, R, next_center=None, next_cov=None, next_floor=None):
    residual = y-tf.einsum('ij,nj->ni', H, x)
    value = gaussian_log(residual, R)
    score = tf.transpose(tf.matmul(H, tf.linalg.solve(R, tf.transpose(residual)), transpose_a=True))
    if next_center is not None:
        future = next_center-tf.einsum('ij,nj->ni', A, x)
        covariance = Q+next_cov
        log_gaussian = gaussian_log(future, covariance)
        log_mixture = tf.reduce_logsumexp(tf.stack([log_gaussian, tf.fill(tf.shape(log_gaussian), next_floor)]), axis=0)
        probability = tf.exp(log_gaussian-log_mixture)
        score += probability[:, None]*tf.transpose(tf.matmul(A, tf.linalg.solve(covariance, tf.transpose(future)), transpose_a=True))
        value += log_mixture
    return value, score


def fit_score(x, score):
    """Finite placeholders on rejected branches are never an accepted guide."""
    dtype = x.dtype; N, d = x.shape
    if N <= d:
        raise ValueError('Score regression requires N >= d+1')
    n = tf.constant(N, dtype); identity = tf.eye(d, dtype=dtype)
    tolerance = tf.constant(64*2**-52*max(N,d), dtype)
    mean = tf.reduce_mean(x, axis=0); centered = x-mean
    covariance = tf.matmul(centered, centered, transpose_a=True)/n
    finite_cloud = tf.reduce_all(tf.math.is_finite(covariance))
    covariance = tf.where(finite_cloud, covariance, identity)
    eigenvalues = tf.linalg.eigvalsh(covariance)
    cloud_margin = tf.math.divide_no_nan(tf.reduce_min(eigenvalues),tf.reduce_max(tf.abs(eigenvalues)))
    cloud_valid = finite_cloud & (cloud_margin > tolerance)
    factor = tf.linalg.cholesky(tf.where(cloud_valid, covariance, identity))
    z = tf.transpose(tf.linalg.triangular_solve(factor, tf.transpose(centered)))
    transformed = tf.matmul(score, factor)
    intercept = tf.reduce_mean(transformed, axis=0)
    raw_precision = -tf.matmul(transformed-intercept, z, transpose_a=True)/n
    precision = .5*(raw_precision+tf.transpose(raw_precision))
    finite_precision = tf.reduce_all(tf.math.is_finite(precision))
    precision = tf.where(finite_precision, precision, identity)
    eigenvalues = tf.linalg.eigvalsh(precision)
    precision_margin = tf.math.divide_no_nan(tf.reduce_min(eigenvalues),tf.reduce_max(tf.abs(eigenvalues)))
    precision_valid = finite_precision & (precision_margin > tolerance)
    precision_factor = tf.linalg.cholesky(tf.where(precision_valid, precision, identity))
    whitened_cov = tf.linalg.cholesky_solve(precision_factor, identity)
    center = mean+tf.linalg.matvec(factor, tf.linalg.matvec(whitened_cov, intercept))
    full_cov = tf.matmul(tf.matmul(factor, whitened_cov), factor, transpose_b=True)
    fitted_cov = tf.linalg.diag(tf.linalg.diag_part(full_cov))
    residual = transformed-intercept+tf.matmul(z, precision, transpose_b=True)
    info = dict(valid=cloud_valid & precision_valid & tf.reduce_all(tf.math.is_finite(center))
                & tf.reduce_all(tf.math.is_finite(full_cov)),
        cloud_margin=cloud_margin, precision_margin=precision_margin, guard_tolerance=tolerance,
        whitening_error=tf.reduce_max(tf.abs(tf.matmul(z,z,transpose_a=True)/n-identity)),
        skew_norm=tf.linalg.norm(raw_precision-tf.transpose(raw_precision)),
        score_residual=tf.reduce_mean(tf.reduce_sum(residual**2,axis=1)),
        full_covariance=full_cov)
    return center, fitted_cov, info


@lru_cache(None)
def make_score_fit(d, N):
    @tf.function(input_signature=[tf.TensorSpec([N,d],tf.float64),tf.TensorSpec([N,d],tf.float64)], jit_compile=True)
    def fit(x, score):
        return fit_score(x, score)
    return fit


@lru_cache(None)
def make_recursive_score_fit(d, N, T, covariance_mode='diagonal', floor_mode='positive'):
    """Explicit diagnostic factorial; original default arithmetic is preserved."""
    if covariance_mode not in ('diagonal', 'full') or floor_mode not in ('positive', 'negligible'):
        raise ValueError('Unknown diagnostic covariance or floor mode')
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    @tf.function(input_signature=[tf.TensorSpec([6],tf.float64),tf.TensorSpec([T,d],tf.float64),
                                  tf.TensorSpec([T,N,d],tf.float64)], jit_compile=True)
    def fit(theta, observations, clouds):
        A,_,H,_,_,_,_,_,Q,_,R,_ = parameterized_model(theta,d,d)
        centers=[];covariances=[];floors=[];details=[];valid=tf.constant(True)
        center=cov=floor=None
        for t in reversed(range(T)):
            _, score = target_log_score(clouds[t], observations[t], A,H,Q,R,center,cov,floor)
            center,cov,info = fit_score(clouds[t],score)
            if covariance_mode == 'full':
                cov = info['full_covariance']
                log_det = 2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(cov))))
            else:
                log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cov)))
            log_ratio = tf.math.log(tf.cast(.01,tf.float64)) if floor_mode == 'positive' else tf.constant(-1000.,tf.float64)
            floor = log_ratio-.5*tf.constant(d*math.log(2*math.pi),tf.float64)-.5*log_det
            valid &= info['valid']
            centers.append(center);covariances.append(cov);floors.append(floor)
            details.append(tf.stack([tf.cast(info['valid'],tf.float64),info['cloud_margin'],info['precision_margin'],
                                     info['guard_tolerance'],info['whitening_error'],info['skew_norm'],info['score_residual']]))
        return (tf.stack(centers[::-1]),tf.stack(covariances[::-1]),tf.stack(floors[::-1]),valid,
                tf.stack(details[::-1]))
    return fit


@lru_cache(None)
def make_recursive_diagonal_qr_fit(d, N, T):
    """Use the existing bounded initialization primitive without descent."""
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit
    @tf.function(input_signature=[tf.TensorSpec([6],tf.float64),tf.TensorSpec([T,d],tf.float64),
                                  tf.TensorSpec([T,N,d],tf.float64)], jit_compile=True)
    def fit(theta, observations, clouds):
        A,_,H,_,_,_,_,_,Q,_,R,_=parameterized_model(theta,d,d)
        centers=[];covariances=[];floors=[];details=[];valid=tf.constant(True);converged=tf.constant(True)
        center=cov=floor=None
        for t in reversed(range(T)):
            target,_=target_log_score(clouds[t],observations[t],A,H,Q,R,center,cov,floor)
            center,cov,floor,info=bounded_density_fit(clouds[t],target,mean_bound=4.,sd_lower=.2,sd_upper=4.,
                max_steps=0,max_backtracks=30,tolerance=1e-7,floor_ratio=.01,initialization='log_quadratic')
            valid &= info['valid'];converged &= info['converged']
            centers.append(center);covariances.append(cov);floors.append(floor)
            details.append(tf.stack([tf.cast(info['initialization_valid'],tf.float64),info['initialization_rank_margin'],
                tf.cast(info['initialization_clipped'],tf.float64),info['initial_shape_residual']]))
        return (tf.stack(centers[::-1]),tf.stack(covariances[::-1]),tf.stack(floors[::-1]),valid,converged,tf.stack(details[::-1]))
    return fit


@lru_cache(None)
def make_oracle(d, T):
    """Exact full backward Gaussian and observation-only guide; reference only."""
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    @tf.function(input_signature=[tf.TensorSpec([6],tf.float64),tf.TensorSpec([T,d],tf.float64)], jit_compile=True)
    def oracle(theta, observations):
        A,_,H,_,_,_,_,_,Q,_,R,_ = parameterized_model(theta,d,d)
        observation_precision=tf.matmul(H,tf.linalg.solve(R,H),transpose_a=True)
        observation_cov=tf.linalg.inv(observation_precision)
        one_step=tf.transpose(tf.matmul(observation_cov,tf.matmul(H,tf.linalg.solve(R,tf.transpose(observations)),transpose_a=True)))
        centers=[];covariances=[];constants=[];center=cov=None
        for t in reversed(range(T)):
            precision=observation_precision
            linear=tf.linalg.matvec(H,tf.linalg.solve(R,observations[t,:,None])[:,0],transpose_a=True)
            if center is not None:
                precision+=tf.matmul(A,tf.linalg.solve(Q+cov,A),transpose_a=True)
                linear+=tf.linalg.matvec(A,tf.linalg.solve(Q+cov,center[:,None])[:,0],transpose_a=True)
            current_cov=tf.linalg.inv(precision); current_center=tf.linalg.matvec(current_cov,linear)
            residual=observations[t]-tf.linalg.matvec(H,current_center)
            constant=gaussian_log(residual[None,:],R)[0]-gaussian_log(tf.zeros([1,d],tf.float64),current_cov)[0]
            if center is not None:
                constant+=gaussian_log((center-tf.linalg.matvec(A,current_center))[None,:],Q+cov)[0]
            center,cov=current_center,current_cov
            centers.append(center);covariances.append(cov);constants.append(constant)
        return tf.stack(centers[::-1]),tf.stack(covariances[::-1]),tf.stack(constants[::-1]),one_step,observation_cov
    return oracle

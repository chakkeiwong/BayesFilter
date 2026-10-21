"""Independent diagnostic importance reference for additive Gaussian dynamics.

Requires a linear, parameter-independent observation matrix with zero offset,
Gaussian process/observation noise, and parameter-independent N(initial_mean,I).
The linearized Gaussian smoother is only a proposal. Exact transition-density
ratios correct it. No production inference, LEDH derivative, or oracle claim.
See docs/plans/sir-fresh-point-reference-2026-10-14.md for the derivation.
"""
import tensorflow as tf

from bayesfilter.testing.nonlinear_bootstrap_fisher_reference_tf import (
    complete_increment, gaussian_log_density)


def make_trajectory_fisher_kernel(model_factory, initial_mean, parameter_count,
                                  observation_dimension, horizon, particles,
                                  *, jit_compile=True):
    mean = tf.convert_to_tensor(initial_mean)
    dtype, d, p, n, length = mean.dtype, mean.shape[0], parameter_count, particles, horizon
    if min(d, p, n, length, observation_dimension) < 1:
        raise ValueError('positive dimensions are required')

    def symmetric(matrix):
        return .5*(matrix+tf.transpose(matrix))

    @tf.function(input_signature=[tf.TensorSpec([p], dtype),
                                  tf.TensorSpec([length, observation_dimension], dtype),
                                  tf.TensorSpec([], tf.int32)],
                 autograph=False, jit_compile=jit_compile)
    def evaluate(theta, observations, seed):
        model, _ = model_factory(theta, tf.zeros_like(theta))
        q, r = model.process_covariance, model.observation_covariance
        h = model.observation_jacobian_fn(mean[None, :])[0]
        identity = tf.eye(d, dtype=dtype)
        filtered_means = tf.TensorArray(dtype, size=length+1).write(0, mean)
        filtered_covs = tf.TensorArray(dtype, size=length+1).write(0, identity)
        predicted_means = tf.TensorArray(dtype, size=length)
        predicted_covs = tf.TensorArray(dtype, size=length)
        matrices = tf.TensorArray(dtype, size=length)
        offsets = tf.TensorArray(dtype, size=length)
        min_factor = tf.constant(float('inf'), dtype)

        def filtering(t, m, cov, ell, means, covs, preds, predcovs, fs, bs, margin):
            # All state directions share a batch; theta direction is zero.
            points = tf.broadcast_to(m, [d,d])
            f = tf.transpose(model.transition_mean_tangent_fn(theta, points, identity))
            prediction = model.transition_mean_fn(theta, m[None,:])[0]
            offset = prediction-tf.linalg.matvec(f,m)
            pcov = symmetric(tf.matmul(tf.matmul(f,cov),f,transpose_b=True)+q)
            innovation = symmetric(tf.matmul(tf.matmul(h,pcov),h,transpose_b=True)+r)
            chol = tf.linalg.cholesky(innovation)
            gain = tf.transpose(tf.linalg.cholesky_solve(chol,tf.matmul(h,pcov)))
            residual = observations[t]-tf.linalg.matvec(h,prediction)
            new_mean = prediction+tf.linalg.matvec(gain,residual)
            # Joseph form is algebraically identical and preserves symmetry.
            a = identity-tf.matmul(gain,h)
            new_cov = symmetric(tf.matmul(tf.matmul(a,pcov),a,transpose_b=True)
                                +tf.matmul(tf.matmul(gain,r),gain,transpose_b=True))
            contribution = gaussian_log_density(residual[None,:],innovation)[0]
            margin = tf.minimum(margin,tf.reduce_min(tf.linalg.diag_part(chol)))
            return (t+1,new_mean,new_cov,ell+contribution,means.write(t+1,new_mean),
                    covs.write(t+1,new_cov),preds.write(t,prediction),predcovs.write(t,pcov),
                    fs.write(t,f),bs.write(t,offset),margin)

        _,_,_,approx_ell,filtered_means,filtered_covs,predicted_means,predicted_covs,matrices,offsets,min_factor = tf.while_loop(
            lambda t,*_: t < length, filtering,
            (0,mean,identity,tf.zeros([],dtype),filtered_means,filtered_covs,
             predicted_means,predicted_covs,matrices,offsets,min_factor),parallel_iterations=1)
        terminal_chol = tf.linalg.cholesky(filtered_covs.read(length))
        min_factor = tf.minimum(min_factor,tf.reduce_min(tf.linalg.diag_part(terminal_chol)))
        final = filtered_means.read(length)+tf.matmul(
            tf.random.stateless_normal([n,d],[seed,0],dtype=dtype),terminal_chol,transpose_b=True)
        paths = tf.TensorArray(dtype,size=length+1).write(length,final)

        def backward(t, following, paths, margin):
            cov = filtered_covs.read(t)
            f = matrices.read(t)
            pcov = predicted_covs.read(t)
            pc = tf.linalg.cholesky(pcov)
            gain = tf.transpose(tf.linalg.cholesky_solve(pc,tf.matmul(f,cov)))
            conditional_cov = symmetric(cov-tf.matmul(tf.matmul(gain,pcov),gain,transpose_b=True))
            cc = tf.linalg.cholesky(conditional_cov)
            conditional_mean = filtered_means.read(t)+tf.matmul(
                following-predicted_means.read(t),gain,transpose_b=True)
            current = conditional_mean+tf.matmul(
                tf.random.stateless_normal([n,d],[seed,t+1],dtype=dtype),cc,transpose_b=True)
            margin = tf.minimum(margin,tf.minimum(tf.reduce_min(tf.linalg.diag_part(pc)),
                                                tf.reduce_min(tf.linalg.diag_part(cc))))
            return t-1,current,paths.write(t,current),margin

        _,_,paths,min_factor = tf.while_loop(lambda t,*_: t >= 0,backward,
             (length-1,final,paths,min_factor),parallel_iterations=1)

        def weighting(t, logw, scores):
            parents, states = paths.read(t),paths.read(t+1)
            exact_mean = model.transition_mean_fn(theta,parents)
            approximate_mean = tf.matmul(parents,matrices.read(t),transpose_b=True)+offsets.read(t)
            ratio = (gaussian_log_density(states-exact_mean,q)
                     -gaussian_log_density(states-approximate_mean,q))
            increment = complete_increment(model_factory,theta,parents,states,observations[t])
            return t+1,logw+ratio,scores+increment

        _,logw,scores = tf.while_loop(lambda t,*_: t < length,weighting,
            (0,tf.zeros([n],dtype),tf.zeros([n,p],dtype)),parallel_iterations=1)
        logsum = tf.reduce_logsumexp(logw)
        weights = tf.exp(logw-logsum)
        score = tf.einsum('n,np->p',weights,scores)
        ell = approx_ell+logsum-tf.math.log(tf.cast(n,dtype))
        ess = 1/tf.reduce_sum(tf.square(weights))
        valid = (tf.reduce_all(tf.math.is_finite(logw)) & tf.reduce_all(tf.math.is_finite(scores))
                 & tf.math.is_finite(ell) & tf.math.is_finite(min_factor) & (min_factor > 0))
        return dict(log_likelihood=ell,score=score,importance_ess=ess,
                    maximum_weight=tf.reduce_max(weights),minimum_cholesky_diagonal=min_factor,
                    approximate_log_likelihood=approx_ell,log_weight_min=tf.reduce_min(logw),
                    log_weight_max=tf.reduce_max(logw),valid=valid)

    return evaluate

"""Scalar Gaussian controlled SMC derived from Heng et al. Algorithms 1--3.

For psi(x)=exp(-a*x*x-b*x-c), completing the square gives precision
1/v+2a, variance k=1/(1/v+2a), mean k*(m/v-b), and log integral
.5*log(k/v)-c+.5*k*(m/v-b)^2-.5*m*m/v. Appendix I.3 gives the
neuroscience specialization. Backward residuals use differences of Gaussian
log integrals. No author implementation is incorporated.

Independent repetitions use native batched linear algebra in the same kernel.
The scalar interface adds/removes a size-one repetition dimension. This is
scalar-state affine-Gaussian dynamics, NOT a multivariate implementation.
"""
from __future__ import annotations
import math
import tensorflow as tf


def gaussian_twist(mean, variance, coefficients):
    """Tensor square-completion identity; invalid precision is flagged."""
    variance = tf.convert_to_tensor(variance, dtype=mean.dtype)
    a, b, c = tf.unstack(coefficients, axis=-1)
    precision = tf.math.reciprocal(variance) + 2*a
    valid = tf.logical_and(variance > 0, precision > 0)
    valid = tf.logical_and(valid, tf.logical_and(
        tf.math.is_finite(variance), tf.math.is_finite(precision)))
    k = tf.where(valid, tf.math.reciprocal(precision),
                 tf.cast(float('nan'), mean.dtype))
    h = mean/variance-b
    log_integral = .5*tf.math.log(k/variance)-c+.5*k*h*h-.5*mean*mean/variance
    return k*h, k, log_integral, valid


def systematic_ancestors(weights, uniform):
    """One uniform per repetition; only the last (particle) axis is scanned."""
    n = tf.shape(weights)[-1]
    positions = (tf.cast(tf.range(n), weights.dtype) +
                 tf.expand_dims(uniform, -1))/tf.cast(n, weights.dtype)
    cdf = tf.cumsum(weights, axis=-1)
    cdf = tf.concat([cdf[..., :-1], tf.ones_like(cdf[..., -1:])], axis=-1)
    return tf.searchsorted(cdf, positions, side='left', out_type=tf.int32)


def binomial_logistic_observation(trials):
    """Normalized binomial observation density in stable log arithmetic."""
    def log_density(y, x):
        count = tf.cast(trials, x.dtype)
        return (tf.math.lgamma(count+1)-tf.math.lgamma(y+1)-tf.math.lgamma(count-y+1)
                -y*tf.nn.softplus(-x)-(count-y)*tf.nn.softplus(x))
    return log_density


class ScalarGaussianControlledSMC:
    """Fixed-shape graph factory with optional independent repetition batches.

    replicas=None preserves the scalar interface [time, particles]. Explicit
    replicas=B uses [time, B, particles], [time, B, 3] policy and [time, B]
    observations/uniforms. Scalar and batched endpoints call identical kernels.
    Every operation normalizes over particles within each independent replica.

    source_scalar_projection explicitly selects the source .001 precision
    margin. It is not a universal numerical default. Invalid equality and
    deficient regressions fail closed.
    """
    def __init__(self, *, observations, particles, alpha, process_variance,
                 initial_mean, initial_variance, observation_log_density,
                 precision_policy, replicas=None, jit_compile=True):
        if precision_policy != 'source_scalar_projection':
            raise ValueError('Select the declared source-comparison precision policy explicitly')
        if observations < 2 or particles < 3 or process_variance <= 0 or initial_variance <= 0:
            raise ValueError('Invalid scalar Gaussian configuration')
        if replicas is not None and (int(replicas) != replicas or replicas < 1):
            raise ValueError('replicas must be a positive integer')
        self.h = int(observations)
        self.n = int(particles)
        self.b = 1 if replicas is None else int(replicas)
        self.scalar_interface = replicas is None
        self.alpha = float(alpha)
        self.q = float(process_variance)
        self.m0 = float(initial_mean)
        self.v0 = float(initial_variance)
        self.observation_log_density = observation_log_density
        self.dtype = tf.float64
        h, n, b, d = self.h, self.n, self.b, self.dtype
        prefix = [h] if self.scalar_interface else [h, b]
        self._policy = tf.TensorSpec(prefix+[3], d)
        self._cloud = tf.TensorSpec(prefix+[n], d)
        self._data = tf.TensorSpec(prefix, d)
        self.forward = tf.function(self._forward, input_signature=[
            self._policy, self._data, self._cloud, self._data],
            jit_compile=jit_compile, autograph=False)
        self.cloud_weights = tf.function(self._cloud_weights, input_signature=[
            self._policy, self._data, self._cloud], jit_compile=jit_compile, autograph=False)
        self.original_filtering = tf.function(self._original_filtering, input_signature=[
            self._policy, self._data, self._cloud], jit_compile=jit_compile, autograph=False)
        self.fit = tf.function(self._fit, input_signature=[
            self._policy, self._cloud, self._cloud], jit_compile=jit_compile, autograph=False)

    def _return_layout(self, result):
        if self.scalar_interface:
            return {key: value if key == 'valid' else tf.squeeze(value, axis=1)
                    for key, value in result.items()}
        return result

    def _cloud_weights(self, policy, y, x):
        if self.scalar_interface:
            policy, y, x = policy[:, None, :], y[:, None], x[:, None, :]
        return self._return_layout(self._cloud_weights_kernel(policy, y, x))

    def _cloud_weights_kernel(self, policy, y, x):
        # G_t^psi = g_t M_{t+1}(psi_{t+1}) / psi_t; M at T is 1.
        _, _, lookahead, valid = gaussian_twist(
            self.alpha*x[:-1], self.q, policy[1:, :, None, :])
        lookahead = tf.concat([
            lookahead, tf.zeros([1, self.b, self.n], self.dtype)], axis=0)
        potential = self.observation_log_density(y[:, :, None], x)+lookahead
        potential += (policy[:, :, 0, None]*x*x +
                      policy[:, :, 1, None]*x+policy[:, :, 2, None])
        _, _, initial, initial_valid = gaussian_twist(
            tf.fill([self.b], tf.constant(self.m0, self.dtype)), self.v0, policy[0])
        increments = tf.reduce_logsumexp(potential, axis=-1)-tf.math.log(tf.cast(self.n, self.dtype))
        logz = tf.cumsum(increments, axis=0)+initial[None, :]
        weights = tf.nn.softmax(potential, axis=-1)
        return dict(log_incremental_weight=potential, logZ=logz,
                    ess=1/tf.reduce_sum(weights*weights, axis=-1),
                    valid=tf.logical_and(
                        tf.logical_and(tf.reduce_all(valid), tf.reduce_all(initial_valid)),
                        tf.reduce_all(tf.math.is_finite(potential))))

    def _original_filtering_kernel(self, policy, y, x):
        """Remove the remaining look-ahead twist from the prefix marginal.

        gamma_t^psi = gamma_t M_{t+1}(psi_{t+1}), with final integral one.
        Self-normalized estimates target filtering, not smoothing. Learned
        policies may use future data; this is not an online-latency claim.
        """
        result = self._cloud_weights_kernel(policy, y, x)
        _, _, lookahead, valid = gaussian_twist(
            self.alpha*x[:-1], self.q, policy[1:, :, None, :])
        lookahead = tf.concat([lookahead, tf.zeros([1, self.b, self.n], self.dtype)], axis=0)
        logw = result['log_incremental_weight']
        adjusted = logw-lookahead
        weights = tf.nn.softmax(adjusted, axis=-1)
        mean = tf.reduce_sum(weights*x, axis=-1)
        variance = tf.reduce_sum(weights*(x-mean[..., None])**2, axis=-1)
        return dict(mean=mean, variance=variance,
                    logZ=result['logZ']+tf.reduce_logsumexp(adjusted, axis=-1)
                         -tf.reduce_logsumexp(logw, axis=-1),
                    ess=1/tf.reduce_sum(weights**2, axis=-1),
                    valid=result['valid'] & tf.reduce_all(valid))

    def _original_filtering(self, policy, y, x):
        if self.scalar_interface:
            policy, y, x = policy[:, None, :], y[:, None], x[:, None, :]
        return self._return_layout(self._original_filtering_kernel(policy, y, x))

    def _forward(self, policy, y, normal, uniform):
        if self.scalar_interface:
            policy, y = policy[:, None, :], y[:, None]
            normal, uniform = normal[:, None, :], uniform[:, None]
        return self._return_layout(self._forward_kernel(policy, y, normal, uniform))

    def _forward_kernel(self, policy, y, normal, uniform):
        h, n, b, d = self.h, self.n, self.b, self.dtype
        clouds = tf.TensorArray(d, size=h, element_shape=[b, n])
        ancestry = tf.TensorArray(tf.int32, size=h, element_shape=[b, n])
        def step(t, previous, clouds, ancestry):
            mean = tf.cond(t == 0, lambda: tf.fill([b, n], tf.constant(self.m0, d)),
                           lambda: self.alpha*previous)
            variance = tf.where(t == 0, tf.constant(self.v0, d), tf.constant(self.q, d))
            controlled_mean, k, _, _ = gaussian_twist(mean, variance, policy[t, :, None, :])
            x = controlled_mean+tf.sqrt(k)*normal[t]
            def next_integral():
                return gaussian_twist(self.alpha*x, self.q, policy[t+1, :, None, :])[2]
            lookahead = tf.cond(t < h-1, next_integral, lambda: tf.zeros([b, n], d))
            logw = (self.observation_log_density(y[t, :, None], x)+lookahead +
                    policy[t, :, 0, None]*x*x+policy[t, :, 1, None]*x+policy[t, :, 2, None])
            ancestors = systematic_ancestors(tf.nn.softmax(logw, axis=-1), uniform[t])
            return (t+1, tf.gather(x, ancestors, batch_dims=1),
                    clouds.write(t, x), ancestry.write(t, ancestors))
        _, _, clouds, ancestry = tf.while_loop(
            lambda t, *_: t < h, step,
            (tf.constant(0), tf.zeros([b, n], d), clouds, ancestry), parallel_iterations=1)
        x = clouds.stack()
        result = self._cloud_weights_kernel(policy, y, x)
        result.update(x=x, ancestors=ancestry.stack())
        result['valid'] = tf.logical_and(result['valid'], tf.reduce_all(
            tf.logical_and(uniform >= 0, uniform < 1)))
        result['valid'] = tf.logical_and(result['valid'], tf.reduce_all(tf.math.is_finite(x)))
        return result

    def _fit(self, policy, x, log_incremental_weight):
        if self.scalar_interface:
            policy = policy[:, None, :]
            x, log_incremental_weight = x[:, None, :], log_incremental_weight[:, None, :]
        return self._return_layout(self._fit_kernel(policy, x, log_incremental_weight))

    def _fit_kernel(self, policy, x, log_incremental_weight):
        h, n, b, d = self.h, self.n, self.b, self.dtype
        rows = tf.TensorArray(d, size=h, element_shape=[b, 3])
        ratios = tf.TensorArray(d, size=h, element_shape=[b])
        residuals = tf.TensorArray(d, size=h, element_shape=[b])
        projected = tf.TensorArray(tf.bool, size=h, element_shape=[b])
        def step(t, next_policy, rows, ratios, residuals, projected):
            def correction():
                mean = self.alpha*x[t]
                return (gaussian_twist(mean, self.q, next_policy[:, None, :])[2] -
                        gaussian_twist(mean, self.q, policy[t+1, :, None, :])[2])
            target = -log_incremental_weight[t]-tf.cond(
                t < h-1, correction, lambda: tf.zeros([b, n], d))
            design = tf.stack([x[t]*x[t], x[t], tf.ones([b, n], d)], axis=-1)
            gram = tf.matmul(design, design, transpose_a=True)
            eigenvalues = tf.linalg.eigvalsh(gram)
            ratio = eigenvalues[:, 0]/eigenvalues[:, -1]
            rhs = tf.linalg.matvec(design, target, transpose_a=True)
            delta = tf.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
            new = policy[t]+delta
            base_precision = tf.where(t == 0, tf.constant(1/self.v0, d), tf.constant(1/self.q, d))
            projected_here = new[:, 0] < -.5*base_precision
            a = tf.where(projected_here, -.5*base_precision+tf.constant(.001, d), new[:, 0])
            new = tf.stack([a, new[:, 1], new[:, 2]], axis=-1)
            healthy = tf.logical_and(
                ratio > tf.constant(64*math.ulp(1.0), d), base_precision+2*a > 0)
            new = tf.where(healthy[:, None], new, tf.fill([b, 3], tf.constant(float('nan'), d)))
            residual = tf.reduce_max(tf.abs(tf.linalg.matvec(design, delta)-target), axis=-1)
            return (t-1, new, rows.write(t, new), ratios.write(t, ratio),
                    residuals.write(t, residual), projected.write(t, projected_here))
        _, _, rows, ratios, residuals, projected = tf.while_loop(
            lambda t, *_: t >= 0, step,
            (tf.constant(h-1), tf.zeros([b, 3], d), rows, ratios, residuals, projected),
            parallel_iterations=1)
        fitted = rows.stack()
        return dict(policy=fitted, valid=tf.reduce_all(tf.math.is_finite(fitted)),
                    gram_eigenvalue_ratio=ratios.stack(), max_residual=residuals.stack(),
                    projected=projected.stack())

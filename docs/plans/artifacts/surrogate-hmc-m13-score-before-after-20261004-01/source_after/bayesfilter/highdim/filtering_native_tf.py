"""Native dense filtering and Gaussian TT-fit histories for the public wrapper.

The dense quadrature and fixed ALS programs retain their existing definitions.
The latter is a local extension, not the author's adaptive TTIRT/TTSIRT route
(Zhao--Cui Algorithm 2; author models/full_sol.m:73--125).
"""

from collections import OrderedDict
from functools import lru_cache
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.highdim.fixed_tt_native_fit_tf import NativeFixedTTFit
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import squared_marginal

D = tf.float64
_DENSE_CACHE = OrderedDict()
_FIT_CACHE = OrderedDict()


@lru_cache(maxsize=16)
def constant_core_program(count, rank, width):
    """Compile padded constant-polynomial initialization for a fixed TT schema."""
    @tf.function(input_signature=[], jit_compile=True, autograph=False)
    def run():
        degree = tf.one_hot(0, width, dtype=D)[None, None, :, None]
        rest = tf.eye(rank, dtype=D)[None, :, None, :] * degree
        first = (tf.one_hot(0, rank, dtype=D)[None, :, None, None]
                 * tf.one_hot(0, rank, dtype=D)[None, None, None, :] * degree)
        return tf.where(tf.range(count)[:, None, None, None] == 0, first, rest)

    return run


def scalar_dense_program(model, observation_shape, grid_size, *, jit_compile=True):
    """Cache model schema; observations, parameters and grid remain tensor inputs."""
    key = (id(model), tuple(observation_shape), grid_size, bool(jit_compile))
    if key in _DENSE_CACHE:
        _DENSE_CACHE.move_to_end(key)
        return _DENSE_CACHE[key][1]
    dates = observation_shape[0]

    @tf.function(input_signature=[
        tf.TensorSpec([model.parameter_dim()], D),
        tf.TensorSpec(observation_shape, D),
        tf.TensorSpec([grid_size, 1], D),
        tf.TensorSpec([grid_size], D),
        tf.TensorSpec([grid_size], D),
    ], jit_compile=jit_compile, autograph=False)
    def run(theta, observations, points, weights, logdet):
        previous_points = tf.tile(points, [grid_size, 1])
        next_points = tf.repeat(points, grid_size, axis=0)
        posterior = tf.zeros([grid_size], D)
        history = tf.zeros([dates, grid_size], D)
        moments = tf.zeros([dates, 4], D)

        def advance(t, posterior, history, moments):
            def predict():
                transition = tf.reshape(model.transition_log_density(
                    theta, previous_points, next_points, t=t), [grid_size, grid_size])
                terms = (tf.math.log(weights)[None] + logdet[None]
                         + posterior[None] + transition)
                return tf.reduce_logsumexp(terms, axis=1)

            predictive = tf.cond(t == 0, lambda: model.initial_log_density(theta, points), predict)
            unnormalized = predictive + model.observation_log_density(
                theta, points, observations[t], t=t)
            increment = tf.reduce_logsumexp(unnormalized + logdet + tf.math.log(weights))
            posterior = unnormalized - increment
            mass = weights * tf.exp(posterior + logdet)
            total = tf.reduce_sum(mass)
            mean = tf.reduce_sum(mass * points[:, 0]) / total
            second = tf.reduce_sum(mass * tf.square(points[:, 0])) / total
            variance = tf.maximum(second - tf.square(mean), tf.constant(0.0, D))
            index = tf.reshape(t, [1, 1])
            return (t + 1, posterior,
                    tf.tensor_scatter_nd_update(history, index, posterior[None]),
                    tf.tensor_scatter_nd_update(moments, index,
                        tf.stack([increment, mean, variance, total])[None]))

        _, _, history, moments = tf.while_loop(
            lambda t, *_: t < dates, advance,
            (tf.constant(0), posterior, history, moments),
            maximum_iterations=dates, parallel_iterations=1,
        )
        return {"posterior": history, "moments": moments,
                "log_likelihood": tf.reduce_sum(moments[:, 0])}

    _DENSE_CACHE[key] = (model, run)
    if len(_DENSE_CACHE) > 8:
        _DENSE_CACHE.popitem(last=False)
    return run


@lru_cache(maxsize=16)
def gaussian_history_program(dates, width, observation_dim, *, jit_compile=True):
    from bayesfilter.highdim.filtering import _compiled_linear_gaussian_moment_history

    shapes = ([dates, observation_dim], [width], [width, width], [width, width],
              [width, width], [observation_dim, width], [observation_dim, observation_dim],
              [width], [observation_dim])

    @tf.function(input_signature=[tf.TensorSpec(shape, D) for shape in shapes],
                 jit_compile=jit_compile, autograph=False)
    def run(y, mean, covariance, transition, process, observation, noise, offset, obs_offset):
        model = SimpleNamespace(
            initial_mean=mean, initial_covariance=covariance,
            transition_matrix=transition, transition_covariance=process,
            observation_matrix=observation, observation_covariance=noise,
            transition_offset=offset, observation_offset=obs_offset,
            state_dim=lambda: width, observation_dim=lambda: observation_dim,
        )
        return _compiled_linear_gaussian_moment_history(model, y, jit_compile=False)

    return run


class GaussianFitProgram:
    """Fixed preparation plus compiled fits, followed by host-only records."""

    def __init__(self, config, dates, width, jit_compile):
        from bayesfilter.highdim import filtering as old

        basis = config.product_basis
        if basis is None or config.fit_config is None or basis.dimension != width:
            raise ValueError("tt_artifacts: INVALID_SHAPE")
        self.points, self.weights = old._tensor_product_reference_quadrature(
            basis, config.fit_quadrature_order)
        self.physical, self.logdet = old._coordinate_map_for_config(config, width).forward(self.points)
        initial = config.initial_cores or old._default_initial_cores(basis, config.fit_config)
        self.report_fit = NativeFixedTTFit(basis, self.points, self.weights,
            config.fit_config, initial, jit_compile=jit_compile, _report_only=True)
        self.core_values = tuple(core.values for core in initial)
        self.defensive = old.TensorProductReferenceDensity(basis, config.measure_convention)
        count = self.points.shape[0]

        @tf.function(input_signature=[
            tf.TensorSpec([dates, width], D), tf.TensorSpec([dates, width, width], D),
            tf.TensorSpec([count, width], D), tf.TensorSpec([count], D),
            tf.TensorSpec([count, width], D), tf.TensorSpec([count], D),
            tuple(tf.TensorSpec(value.shape, D) for value in self.core_values),
        ], jit_compile=jit_compile, autograph=False)
        def run(means, covariances, points, weights, physical, logdet, core_values):
            native = NativeFixedTTFit(basis, points, weights, config.fit_config,
                tuple(TTCore(value) for value in core_values), jit_compile=jit_compile)
            log_reference = old._log_uniform_reference_weight_density(basis)
            defensive_z = self.defensive.normalizer(config.measure_convention.mass_measure)

            def date_row(t):
                log_target = old._gaussian_log_density(physical, means[t], covariances[t])
                log_target = log_target + logdet - log_reference
                target = tf.exp(0.5 * (log_target - tf.reduce_max(log_target)))
                fit = native(target, native.initial)
                normalizer = squared_marginal(native.unpack(fit["cores"]), basis, (),
                    tf.zeros([1, 0], D))[0] + tf.constant(config.density_tau, D) * defensive_z
                return {"log_target": log_target, "target": target,
                        "fit": fit, "normalizer": normalizer}

            first = date_row(tf.constant(0))
            histories = tf.nest.map_structure(lambda value: tf.TensorArray(
                value.dtype, dates, element_shape=value.shape).write(0, value), first)

            def advance(t, histories):
                row = date_row(t)
                return t + 1, tf.nest.map_structure(
                    lambda buffer, value: buffer.write(t, value), histories, row)

            _, histories = tf.while_loop(lambda t, _: t < dates, advance,
                (tf.constant(1), histories), maximum_iterations=dates - 1, parallel_iterations=1)
            return tf.nest.map_structure(lambda value: value.stack(), histories)

        self.call = run

    def __call__(self, means, covariances):
        return self.call(means, covariances, self.points, self.weights,
                         self.physical, self.logdet, self.core_values)


def gaussian_fit_program(config, dates, width, *, jit_compile=True):
    key = (id(config), dates, width, bool(jit_compile))
    if key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        return _FIT_CACHE[key][1]
    program = GaussianFitProgram(config, dates, width, jit_compile)
    _FIT_CACHE[key] = (config, program)
    if len(_FIT_CACHE) > 8:
        _FIT_CACHE.popitem(last=False)
    return program

"""Diagnostic independent-pilot calibration through the existing proposal maps."""

import tensorflow as tf

from bayesfilter.highdim.covariance_proposal_tf import (
    build_maps, component_densities, draw_from_maps, observation_density,
)


def make_context_pilot(spec, particles, horizon, controls, guide=None,
                       dtype=tf.float64, jit_compile=True):
    dimension = spec.dimension
    observed_dimension = getattr(spec, "observation_dimension", dimension)
    signature = [tf.TensorSpec([spec.parameter_count], dtype),
                 tf.TensorSpec([horizon, observed_dimension], dtype),
                 tf.TensorSpec([horizon, particles, dimension], dtype),
                 tf.TensorSpec([horizon, particles, dimension], dtype),
                 tf.TensorSpec([horizon, particles, 2], dtype)]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def evaluate(theta, observations, ancestors, noise, uniforms):
        model, _ = spec.model(theta, tf.zeros_like(theta))
        equal = tf.fill([3], tf.constant(1 / 3, dtype))
        components = tf.TensorArray(dtype, size=horizon, element_shape=[particles, 3])
        targets = tf.TensorArray(dtype, size=horizon, element_shape=[particles])
        validity = tf.TensorArray(tf.bool, size=horizon, element_shape=[])
        diagnostics = tf.TensorArray(dtype, size=horizon, element_shape=[2])

        def step(time, components, targets, validity, diagnostics):
            maps = build_maps(model, theta, ancestors[time], tf.zeros_like(ancestors[time]),
                              observations[time], controls, backward_guide=guide, time=time)
            states, tangent = draw_from_maps(maps, equal, uniforms[time], noise[time])
            densities, _, density_valid = component_densities(states, tangent, *maps[:4])
            target = observation_density(model, theta, states, tangent, observations[time])[0] + densities[:, 0]
            sampling_density = tf.reduce_logsumexp(densities, axis=-1) - tf.math.log(tf.constant(3., dtype))
            weights = tf.nn.softmax(target - sampling_density)
            valid = maps[12] & density_valid & tf.reduce_all(tf.math.is_finite(densities)) & tf.reduce_all(tf.math.is_finite(target))
            diagnostic = tf.stack([1 / tf.reduce_sum(tf.square(weights)), tf.reduce_max(weights)])
            return (time + 1, components.write(time, densities), targets.write(time, target),
                    validity.write(time, valid), diagnostics.write(time, diagnostic))

        _, components, targets, validity, diagnostics = tf.while_loop(
            lambda time, *_: time < horizon, step,
            (0, components, targets, validity, diagnostics), parallel_iterations=1)
        return dict(log_components=components.stack(), log_target=targets.stack(),
                    valid=validity.stack(), diagnostics=diagnostics.stack())

    return evaluate

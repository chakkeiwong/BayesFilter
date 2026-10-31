"""Diagnostic wrappers for the shared mixture; not canonical LEDH admission."""

import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenMixtureBackwardGuide, FrozenQuadraticBackwardGuide)
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec


def make_diagnostic_programs(particles, horizon, controls, *, spec=None,
                             guide_family="quadratic", mixture_components=None,
                             jit_compile=True):
    spec = LGSSMSpec("diagonal_ar", 2) if spec is None else spec
    dimension = spec.dimension
    parameter_count = spec.parameter_count
    observation_dimension = getattr(spec, "observation_dimension", dimension)
    dtype = tf.float64
    signature = [
        tf.TensorSpec([parameter_count], dtype),
        tf.TensorSpec([horizon, observation_dimension], dtype),
        tf.TensorSpec([particles, dimension], dtype),
        tf.TensorSpec([horizon, particles, dimension], dtype),
        tf.TensorSpec([horizon, particles, 2], dtype),
        tf.TensorSpec([3], dtype),
    ]
    if guide_family == "quadratic":
        signature += [tf.TensorSpec([horizon, dimension, dimension], dtype),
                      tf.TensorSpec([horizon, dimension], dtype)]
        guide_type = FrozenQuadraticBackwardGuide
    elif guide_family == "mixture" and isinstance(mixture_components, int) and mixture_components > 0:
        signature += [tf.TensorSpec([horizon, mixture_components], dtype),
                      tf.TensorSpec([horizon, mixture_components, dimension], dtype),
                      tf.TensorSpec([horizon, mixture_components, dimension, dimension], dtype),
                      tf.TensorSpec([horizon], tf.bool)]
        guide_type = FrozenMixtureBackwardGuide
    else:
        raise ValueError("unsupported guide family or missing positive component count")

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def evaluate(theta, observations, initial, innovations, uniforms, beta, *guide_parameters):
        guide = guide_type(*guide_parameters)
        program = core.make_filter(spec, particles, horizon, controls, dtype,
                                   jit_compile=jit_compile, backward_guide=guide)
        first = program(theta, tf.one_hot(0, parameter_count, dtype=dtype), observations,
                        initial, innovations, uniforms, beta)
        scores = tf.TensorArray(dtype, size=parameter_count).write(0, first["score"])
        validities = tf.TensorArray(tf.bool, size=parameter_count).write(0, first["valid"])
        traces = tf.TensorArray(dtype, size=parameter_count).write(0, first["trace"])

        def step(coordinate, scores, validities, traces):
            result = program(theta, tf.one_hot(coordinate, parameter_count, dtype=dtype),
                             observations, initial, innovations, uniforms, beta)
            return (coordinate + 1, scores.write(coordinate, result["score"]),
                    validities.write(coordinate, result["valid"]), traces.write(coordinate, result["trace"]))

        _, scores, validities, traces = tf.while_loop(
            lambda coordinate, *_: coordinate < parameter_count, step,
            (1, scores, validities, traces), parallel_iterations=1)
        return {"value": first["value"], "score": scores.stack(),
                "valid": tf.reduce_all(validities.stack()), "trace": first["trace"],
                "direction_valid": validities.stack(), "direction_trace": traces.stack(),
                "ancestor_history": first["ancestor_history"]}

    @tf.function(input_signature=signature + [
        tf.TensorSpec([horizon, particles, dimension], dtype),
        tf.TensorSpec([horizon, particles, 2], dtype),
    ], jit_compile=jit_compile, autograph=False)
    def pilot(theta, observations, initial, innovations, uniforms, beta,
              *auxiliary):
        guide = guide_type(*auxiliary[:-2])
        pilot_normals, pilot_uniforms = auxiliary[-2:]
        program = core.make_filter(spec, particles, horizon, controls, dtype,
                                   jit_compile=jit_compile, backward_guide=guide)
        forward = program(theta, tf.zeros_like(theta), observations, initial,
                          innovations, uniforms, beta)
        model, _ = spec.model(theta, tf.zeros_like(theta))
        components = tf.TensorArray(dtype, size=horizon, element_shape=[particles, 3])
        targets = tf.TensorArray(dtype, size=horizon, element_shape=[particles])

        def step(time, components, targets, valid):
            ancestors = forward["ancestor_history"][time]
            maps = core.build_maps(model, theta, ancestors, tf.zeros_like(ancestors),
                                   observations[time], controls, backward_guide=guide, time=time)
            child, tangent = core.draw_from_maps(
                maps, tf.fill([3], tf.constant(1.0 / 3.0, dtype)),
                pilot_uniforms[time], pilot_normals[time])
            density, _, density_valid = core.component_densities(child, tangent, *maps[:4])
            likelihood, _ = core.observation_density(model, theta, child, tangent, observations[time])
            return (time + 1, components.write(time, density),
                    targets.write(time, likelihood + density[:, 0]),
                    valid & density_valid & maps[12])

        _, components, targets, valid = tf.while_loop(
            lambda time, *_: time < horizon, step,
            (0, components, targets, forward["valid"]), parallel_iterations=1)
        return {"log_components": components.stack(), "log_target": targets.stack(),
                "valid": valid}

    return evaluate, pilot

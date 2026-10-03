"""Tensor-bound C2 Student sampling through the shared proposal authority."""

from functools import partial

import tensorflow as tf

from bayesfilter.highdim.c2_transformed_observation_student_proposal_tf import _student_tensor_view
from bayesfilter.ops.stateless_gamma_tf import philox_gamma_float64
from bayesfilter.ops.stateless_random_tf import philox_normal_float64

D = tf.float64


def pack_student_proposals(proposals):
    """Pack existing validated tensor fields, specializing only on dimension/nu."""
    layouts = tuple((proposal.dimension, float(proposal.nu)) for proposal in proposals)
    configurations = tuple(dict.fromkeys(layouts))
    indices = tf.constant([configurations.index(layout) for layout in layouts], tf.int32)
    groups = []
    for configuration in configurations:
        template = proposals[layouts.index(configuration)]
        selected = tuple(proposal if layout == configuration else template
                         for proposal, layout in zip(proposals, layouts, strict=True))
        rows = tuple((p.transition_matrix, p.gain, p.transformed_observation, p.chol)
                     for p in selected)
        groups.append(tf.nest.map_structure(lambda *values: tf.stack(values), *rows))
    operands = (indices, tuple(groups))
    specs = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, value.dtype), operands)
    return configurations, operands, specs


def sample_student(configuration, payload, count, parents, seed):
    dimension, nu = configuration
    view = _student_tensor_view(nu, payload)
    normal = philox_normal_float64([count, dimension], seed)
    chi_square = philox_gamma_float64([count], seed + [0, 1],
        tf.constant(nu / 2., D), tf.constant(.5, D))
    return view._transform_core(parents, normal, chi_square)


def student_step(configurations, count):
    def sample(time, parents, seed, theta, operands):
        del theta
        indices, groups = operands

        def execute(configuration, packed):
            payload = tf.nest.map_structure(lambda value: value[time-1], packed)
            return sample_student(configuration, payload, count, parents, seed)

        functions = tuple(partial(execute, configuration, packed)
                          for configuration, packed in zip(configurations, groups, strict=True))
        result = tf.switch_case(indices[time-1], branch_fns=functions)
        return result['physical_points'], result['physical_log_density'], {'finite': result['finite']}
    return sample

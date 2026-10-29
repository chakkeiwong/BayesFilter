"""Tensor-bound independent C2 proposal sampling through shared authorities."""

from functools import partial

import tensorflow as tf

from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    GaussianHermiteRetainedProposal, _random_inputs_program,
)
from bayesfilter.ops.stateless_random_tf import philox_normal_float64

D = tf.float64


def proposal_configuration(proposal):
    """Static configuration only; numerical payloads are separate operands."""
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import FrozenGaussianStateProposal
    if isinstance(proposal, FrozenGaussianStateProposal):
        return ('gaussian', proposal.dimension)
    if isinstance(proposal, GaussianHermiteRetainedProposal):
        return ('hermite', proposal.dimension,
                tuple(tuple(core.shape) for core in proposal.prefix_core_values),
                proposal.defensive_nu)
    raise TypeError('unsupported independent transition proposal')


def proposal_payload(proposal):
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import FrozenGaussianStateProposal
    if isinstance(proposal, FrozenGaussianStateProposal):
        return (proposal.mean, proposal.chol)
    return (proposal.prefix_core_values, proposal.suffix_gram, proposal.z_h,
            proposal.tau_abs, proposal.coordinate_offset, proposal.coordinate_matrix)


def pack_proposals(proposals):
    """Collect validated tensor leaves by static topology; no numerical work."""
    layouts = tuple(proposal_configuration(proposal) for proposal in proposals)
    configurations = tuple(dict.fromkeys(layouts))
    indices = tf.constant([configurations.index(layout) for layout in layouts], tf.int32)
    time_indices = tf.constant([int(proposal.time_index) for proposal in proposals], tf.int32)
    groups = []
    for configuration in configurations:
        template = proposals[layouts.index(configuration)]
        rows = tuple(proposal_payload(proposal if layout == configuration else template)
                     for proposal, layout in zip(proposals, layouts, strict=True))
        groups.append(tf.nest.map_structure(lambda *values: tf.stack(values), *rows))
    operands = (indices, time_indices, tuple(groups))
    specs = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, value.dtype), operands)
    return configurations, operands, specs


def diagnostic_specs():
    return {'finite': tf.TensorSpec([], tf.bool),
        'time_index_valid': tf.TensorSpec([], tf.bool),
        'selected_polynomial_count': tf.TensorSpec([], tf.int32),
        'polynomial_probability': tf.TensorSpec([], D),
        'maximum_inverse_cdf_residual': tf.TensorSpec([], D),
        'minimum_conditional_mass': tf.TensorSpec([], D),
        'minimum_endpoint_margin': tf.TensorSpec([], D),
        'cdf_bracket_valid': tf.TensorSpec([], tf.bool)}


def _hermite_tensor_view(configuration, payload):
    """Private traced view of validated tensors, without a public identity."""
    view = object.__new__(GaussianHermiteRetainedProposal)
    object.__setattr__(view, 'prefix_core_values', payload[0])
    object.__setattr__(view, 'suffix_gram', payload[1])
    object.__setattr__(view, 'z_h', payload[2])
    object.__setattr__(view, 'tau_abs', payload[3])
    object.__setattr__(view, 'coordinate_offset', payload[4])
    object.__setattr__(view, 'coordinate_matrix', payload[5])
    object.__setattr__(view, 'defensive_nu', configuration[3])
    return view


def _sample_configuration(configuration, payload, count, seed):
    if configuration[0] == 'gaussian':
        from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import _gaussian_transform_core
        mean, chol = payload
        result = _gaussian_transform_core(philox_normal_float64([count, configuration[1]], seed), mean, chol)
        row = {'finite': result['finite'], 'selected_polynomial_count': tf.constant(0, tf.int32),
               'polynomial_probability': tf.constant(0., D), 'maximum_inverse_cdf_residual': tf.constant(0., D),
               'minimum_conditional_mass': tf.constant(0., D), 'minimum_endpoint_margin': tf.constant(0., D),
               'cdf_bracket_valid': tf.constant(True)}
    else:
        # A private tensor view reuses the sole retained-proposal mathematical
        # authority. Its inputs come from validated public proposals. It has
        # no proposal_id and cannot issue a public manifest or admission claim.
        view = _hermite_tensor_view(configuration, payload)
        random_inputs = _random_inputs_program(count, configuration[1], configuration[3])(seed)
        result = view.sample_physical(*random_inputs)
        row = {'finite': result['finite'],
               'selected_polynomial_count': tf.reduce_sum(tf.cast(result['selected_polynomial'], tf.int32)),
               'polynomial_probability': result['polynomial_probability'],
               'maximum_inverse_cdf_residual': result['maximum_inverse_cdf_residual'],
               'minimum_conditional_mass': result['minimum_conditional_mass'],
               'minimum_endpoint_margin': result['minimum_endpoint_margin'],
               'cdf_bracket_valid': result['cdf_bracket_valid']}
    return result['physical_points'], result['physical_log_density'], row


def independent_step(configurations, count):
    def sample(time, parents, seed, theta, operands):
        del parents, theta
        indices, time_indices, groups = operands
        def execute(configuration, packed):
            payload = tf.nest.map_structure(lambda value: value[time-1], packed)
            return _sample_configuration(configuration, payload, count, seed)
        functions = tuple(partial(execute, configuration, packed)
                          for configuration, packed in zip(configurations, groups, strict=True))
        points, density, row = tf.switch_case(indices[time-1], branch_fns=functions)
        row['time_index_valid'] = time_indices[time-1] == time
        return points, density, row
    return sample

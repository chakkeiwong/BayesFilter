"""Compiled fixed-HMC source-route evaluation of prepared transports.

Frozen targets and transports adapt models/full_sol.m:32--40, 76--84,
98--99, 123 and 129--130 (paper section 4.1). This preserves the existing
fitting algorithm and cannot establish source-faithfulness of grid-CDF fits.
Object and manifest construction stays outside the numerical program.
"""

from collections import OrderedDict

import tensorflow as tf

from bayesfilter.highdim import source_route_numerics_tf as numerics
from bayesfilter.ops.compiled_tensor_program_tf import tensor_program as _tensor_program

D = tf.float64
_PROGRAMS = OrderedDict()
_MARGINALS = OrderedDict()


def _query_program(function, shape, jit_compile):
    return _tensor_program(function, [tf.TensorSpec(shape, D)], jit_compile)


def retained_values(target, transport, reference):
    """Complete tensor calculation for one existing retained-sample step."""
    from bayesfilter.highdim.source_route import (
        effective_sample_size_from_log_weights,
        normalize_log_weights,
        source_route_proposal_log_weights,
    )

    reference = numerics.finite(reference, "reference_samples")
    local = transport.inverse_transport(reference)
    physical = target.physical_points_from_reference(local)
    proposal = transport.proposal_log_density(local_points=local, reference_points=reference)
    target_log = target.log_target_density(local)
    correction = source_route_proposal_log_weights(
        log_target_density=target_log, log_proposal_density=proposal)
    weights = normalize_log_weights(correction)
    ess = effective_sample_size_from_log_weights(weights)
    return physical, proposal, target_log, correction, weights, ess, transport.log_normalizer()


def retained_program(target, transport, shape, *, jit_compile=True):
    key = ("retained", id(target), id(transport), tuple(shape), bool(jit_compile))
    if key not in _PROGRAMS:
        program = _query_program(lambda query: retained_values(target, transport, query), shape, jit_compile)
        _PROGRAMS[key] = (target, transport, program)
        if len(_PROGRAMS) > 16:
            _PROGRAMS.popitem(last=False)
    _PROGRAMS.move_to_end(key)
    return _PROGRAMS[key][-1]


def prepare_previous_marginal(previous, keep):
    """Prepare only immutable transport schema, independent of query tensors."""
    from bayesfilter.highdim.source_route import SourceRouteTransportProtocol

    key = (id(previous), tuple(keep))
    if key not in _MARGINALS:
        # First use during tracing must not construct an eager-only metadata
        # object inside the numerical graph. No query value enters this scope.
        with tf.init_scope():
            protocol = SourceRouteTransportProtocol(previous.transport_object)
            marginal = protocol.marginalize(keep)
        _MARGINALS[key] = (previous, marginal)
        if len(_MARGINALS) > 16:
            _MARGINALS.popitem(last=False)
    _MARGINALS.move_to_end(key)
    return _MARGINALS[key][-1]


def previous_marginal_values(frame, marginal, keep, points):
    """Previous SIRT prefix density with the unchanged affine Jacobian."""
    from bayesfilter.highdim.source_route import _source_route_eval_marginal_pdf

    points = numerics.finite(points, "physical_points")
    mu = tf.gather(frame.mu, keep)
    matrix = tf.gather(tf.gather(frame.matrix, keep, axis=0), keep, axis=1)
    local = tf.linalg.solve(matrix, points - mu[:, None])
    density = numerics.positive(_source_route_eval_marginal_pdf(marginal, local),
                                "previous_marginal_eval_pdf")
    return local, tf.math.log(density) - numerics.log_abs_det(matrix)


def previous_marginal_program(previous, keep, shape, *, jit_compile=True):
    marginal = prepare_previous_marginal(previous, keep)
    key = ("previous", id(previous), tuple(keep), tuple(shape), bool(jit_compile))
    if key not in _PROGRAMS:
        program = _query_program(lambda query: previous_marginal_values(
            previous.coordinate_frame, marginal, keep, query), shape, jit_compile)
        _PROGRAMS[key] = (previous, marginal, program)
        if len(_PROGRAMS) > 16:
            _PROGRAMS.popitem(last=False)
    _PROGRAMS.move_to_end(key)
    return _PROGRAMS[key][-1], marginal

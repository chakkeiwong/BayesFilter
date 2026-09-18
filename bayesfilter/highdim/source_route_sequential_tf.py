"""Native date evaluation for the existing frozen source-route replay.

This fixed-HMC adaptation consumes predeclared transports; it does not fit or
retune them. The previous density is the preceding frozen transport's prefix
marginal (paper section 4.1; author models/full_sol.m:76--84, 123, 129--130).
Python constructs heterogeneous callable/output schemas. TensorFlow evaluates
dates, and host code subsequently assembles the retained-object provenance.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim import source_route_numerics_tf as numerics
from bayesfilter.highdim.source_route_runtime_tf import (
    _tensor_program,
    prepare_previous_marginal,
    previous_marginal_values,
)

D = tf.float64
_PROGRAMS = OrderedDict()


@dataclass(frozen=True)
class _SquaredMarginalEvaluator:
    """Schema-only view; evaluate paired-core moments inside the date graph."""

    density: object
    keep: tuple

    def eval_pdf(self, local):
        return self.density.normalized_marginal_density_values(self.keep, tf.transpose(local))


def _marginal_evaluator(protocol, keep):
    from bayesfilter.highdim.transport import FixedTTSIRTTransport

    transport = protocol.transport_object
    if isinstance(transport, FixedTTSIRTTransport):
        if not keep or keep != tuple(range(len(keep))) or keep[-1] >= transport.dimension:
            raise ValueError("source previous marginalization requires valid prefix keep axes")
        return _SquaredMarginalEvaluator(transport.density, tuple(keep))
    # Caller-owned contract test doubles retain their own preparation contract;
    # the BayesFilter-owned fixed-TTSIRT path above does no numerical work here.
    return protocol.marginalize(keep)


def density_values(points, time_index, parameter_dim, state_dim, prior,
                   transition, likelihood):
    """The unchanged prior/previous + transition + observation density."""
    from bayesfilter.highdim.source_route import _finite_vector

    axes = tuple(range(parameter_dim)) + tuple(range(parameter_dim + state_dim, parameter_dim + 2 * state_dim))
    prior_values = _finite_vector("prior_log_density", prior(tf.gather(points, axes, axis=0)))
    transition_values = _finite_vector("transition_log_density", transition(points, time_index))
    likelihood_values = _finite_vector("likelihood_log_density", likelihood(points, time_index))
    if prior_values.shape != transition_values.shape or prior_values.shape != likelihood_values.shape:
        raise ValueError("source sequential density: INVALID_SHAPE")
    return -prior_values - transition_values - likelihood_values


def physical_density_program(time_index, parameter_dim, state_dim, transition, likelihood,
                             prior, previous, shape, *, jit_compile=True):
    """Fixed-signature boundary for direct sequential target callbacks."""
    key = ("physical", time_index, parameter_dim, state_dim, id(transition), id(likelihood),
           id(prior), id(previous), tuple(shape), bool(jit_compile))
    if key not in _PROGRAMS:
        if time_index == 1:
            prior_function = prior
        else:
            keep = tuple(range(parameter_dim + state_dim))
            marginal = prepare_previous_marginal(previous, keep)

            def prior_function(query):
                return previous_marginal_values(previous.coordinate_frame, marginal, keep, query)[1]

        def numerical(query):
            return density_values(query, time_index, parameter_dim, state_dim,
                                  prior_function, transition, likelihood)

        program = _tensor_program(numerical, [tf.TensorSpec(shape, D)], jit_compile)
        _PROGRAMS[key] = (transition, likelihood, prior, previous, program)
        if len(_PROGRAMS) > 8:
            _PROGRAMS.popitem(last=False)
    _PROGRAMS.move_to_end(key)
    return _PROGRAMS[key][-1]


def _prepare(specs, index):
    from bayesfilter.highdim.source_route import (
        SourceRouteSequentialDensityComponents,
        SourceRouteTarget,
    )

    spec = specs[index]
    components = spec.density_components
    if (getattr(components.negative_log_physical_density, "__func__", None)
            is not SourceRouteSequentialDensityComponents.negative_log_physical_density):
        raise TypeError("compiled sequential replay requires the declared density-component formula")
    if index == 0:
        prior = components.prior_log_density_fn
        marginal = None
        if not callable(prior):
            raise TypeError("t=1 requires callable prior_log_density_fn")
    else:
        previous = specs[index - 1]
        if components.prior_log_density_fn is not None:
            raise ValueError("t>1 uses previous retained marginal, not prior_log_density_fn")
        keep = tuple(range(components.parameter_dim + components.state_dim))
        density_marginal = _marginal_evaluator(previous.transport, keep)
        marginal = (density_marginal if keep == spec.previous_marginal_keep_axes else
                    _marginal_evaluator(previous.transport, spec.previous_marginal_keep_axes))

        def prior(query):
            return previous_marginal_values(previous.target.coordinate_frame,
                                             density_marginal, keep, query)[1]

    def physical_density(points):
        expected = components.parameter_dim + 2 * components.state_dim
        if points.shape.rank != 2 or points.shape[0] != expected:
            raise ValueError("physical_points: INVALID_SHAPE")
        return density_values(points, spec.time_index, components.parameter_dim,
            components.state_dim, prior, components.transition_log_density_fn,
            components.likelihood_log_density_fn)

    target = SourceRouteTarget(negative_log_physical_density_fn=physical_density,
        coordinate_frame=spec.target.coordinate_frame, shift_constant=spec.target.shift_constant,
        time_index=spec.time_index, target_family=spec.target.target_family,
        source_terms=spec.target.source_terms, log_abs_det_policy=spec.target.log_abs_det_policy)
    rows, dimension = spec.reference_samples.shape[1], spec.target.coordinate_frame.dimension
    prefix = 0 if index == 0 else len(spec.previous_marginal_keep_axes)
    shapes = ((dimension, rows), (rows,), (rows,), (rows,), (rows,), (), (),
              (prefix, rows), (prefix, rows), (rows,))
    return target, marginal, shapes


def _transport_schema(specs):
    """Unique immutable transport/shape combinations, independent of dates."""
    groups, indices = {}, []
    for spec in specs:
        key = (id(spec.transport), tuple(spec.reference_samples.shape))
        if key not in groups:
            groups[key] = (len(groups), spec)
        indices.append(groups[key][0])
    return tuple(value[1] for value in groups.values()), tuple(indices)


def sequential_program(specs, *, jit_compile=True):
    """Compile all date calculations; immutable callback schemas stay explicit."""
    specs = tuple(specs)
    # Query values are explicit program inputs. Rebuilding a step specification
    # with new queries must reuse the same immutable numerical schema.
    key = (tuple((id(spec.target), id(spec.transport), id(spec.density_components),
                  spec.previous_marginal_keep_axes, spec.previous_marginal_input_axes,
                  tuple(spec.reference_samples.shape)) for spec in specs), bool(jit_compile))
    if key not in _PROGRAMS:
        with tf.init_scope():
            prepared = tuple(_prepare(specs, index) for index in range(len(specs)))
        lengths = tuple(sum(math.prod(shape) for shape in row[2]) for row in prepared)
        width = max(lengths)
        transport_specs, transport_ids = _transport_schema(specs)
        query_width = max(math.prod(spec.reference_samples.shape) for spec in transport_specs)
        transport_width = max(math.prod(spec.reference_samples.shape) + spec.reference_samples.shape[1] + 1
                              for spec in transport_specs)

        def numerical(*queries):
            from bayesfilter.highdim.source_route import (
                effective_sample_size_from_log_weights,
                normalize_log_weights,
                source_route_proposal_log_weights,
            )

            packed_queries = tf.stack(tuple(tf.pad(tf.reshape(query, [-1]),
                [[0, query_width - math.prod(query.shape)]]) for query in queries))

            def step(index):
                query_row = packed_queries[index]

                def transport_branch(spec):
                    def evaluate(query_row):
                        shape = spec.reference_samples.shape
                        reference = numerics.finite(tf.reshape(query_row[:math.prod(shape)], shape),
                                                    "reference_samples")
                        local = spec.transport.inverse_transport(reference)
                        proposal = spec.transport.proposal_log_density(local_points=local, reference_points=reference)
                        log_z = spec.transport.log_normalizer()
                        packed = tf.concat([tf.reshape(local, [-1]), proposal, tf.reshape(log_z, [1])], axis=0)
                        return tf.pad(packed, [[0, transport_width - math.prod(shape) - shape[1] - 1]])
                    local_program = _tensor_program(evaluate, [tf.TensorSpec([query_width], D)], False)
                    return lambda: local_program.python_function(query_row)

                # Reusing one frozen transport at several dates does not copy
                # its CDF/bisection graph into each target branch.
                transport_branches = tuple(transport_branch(spec) for spec in transport_specs)
                transported = tf.switch_case(tf.gather(tf.constant(transport_ids), index), transport_branches)

                def target_branch(date):
                    # Keep the complete marginal/date pullback in its own
                    # Case arm; heterogeneous branches must not exchange loop
                    # tapes. The enclosing date program owns XLA compilation.
                    local_program = _tensor_program(lambda row: target_values(date, row),
                                                   [tf.TensorSpec([transport_width], D)], False)
                    return lambda: local_program.python_function(transported)

                def target_values(index, transported):
                    spec = specs[index]
                    target, marginal, _ = prepared[index]
                    shape = spec.reference_samples.shape
                    local = tf.reshape(transported[:math.prod(shape)], shape)
                    proposal = transported[math.prod(shape):math.prod(shape) + shape[1]]
                    log_z = transported[math.prod(shape) + shape[1]]
                    physical = target.physical_points_from_reference(local)
                    target_log = target.log_target_density(local)
                    correction = source_route_proposal_log_weights(
                        log_target_density=target_log, log_proposal_density=proposal)
                    weights = normalize_log_weights(correction)
                    ess = effective_sample_size_from_log_weights(weights)
                    values = physical, proposal, target_log, correction, weights, ess, log_z
                    if index == 0:
                        previous_physical = tf.zeros([0, shape[1]], D)
                        previous_local = previous_physical
                        previous_log = tf.zeros([shape[1]], D)
                    else:
                        previous_physical = tf.gather(values[0], spec.previous_marginal_input_axes, axis=0)
                        previous_local, previous_log = previous_marginal_values(
                            specs[index-1].target.coordinate_frame, marginal,
                            spec.previous_marginal_keep_axes, previous_physical)
                    fields = (*values, previous_physical, previous_local, previous_log)
                    packed = tf.concat(tuple(tf.reshape(value, [-1]) for value in fields), axis=0)
                    return tf.pad(packed, [[0, width - lengths[index]]])

                branches = tuple(target_branch(date) for date in range(len(specs)))
                return tf.switch_case(index, branches)

            return tf.map_fn(step, tf.range(len(specs)),
                fn_output_signature=tf.TensorSpec([width], D), parallel_iterations=1)

        signatures = tuple(tf.TensorSpec(spec.reference_samples.shape, D) for spec in specs)
        program = _tensor_program(numerical, signatures, jit_compile)
        _PROGRAMS[key] = (specs, program, prepared, lengths)
        if len(_PROGRAMS) > 8:
            _PROGRAMS.popitem(last=False)
    _PROGRAMS.move_to_end(key)
    _, program, prepared, lengths = _PROGRAMS[key]
    return program, prepared, lengths


def unpack_step(packed, shapes, length):
    """Restore the fixed heterogeneous result schema after numerical execution."""
    fields = tf.split(packed[:length], tuple(math.prod(shape) for shape in shapes))
    return tuple(tf.reshape(value, shape) for value, shape in zip(fields, shapes, strict=True))


@numerics.compiled
def total_log_normalizer(log_normalizers, shifts):
    return tf.reduce_sum(log_normalizers - shifts)

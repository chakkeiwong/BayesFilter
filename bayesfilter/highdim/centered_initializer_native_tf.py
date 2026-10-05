"""Native contractions for the existing centered additive/pair initializers.

The initializer remains a repository extension. Forward and reverse TT
messages preserve the original feature integrals, including the all-ones
coefficient function on unselected axes of the pair-feature construction.
"""

from collections import OrderedDict
from dataclasses import fields
from functools import wraps
from inspect import signature

import tensorflow as tf

from bayesfilter.highdim import centered_tt_native_tf as tt

D = tf.float64
_PROGRAMS = OrderedDict()
_INITIALIZERS = OrderedDict()


def compiled_initializer(result_type):
    """Compile complete fixed-parent initialization and restore its host record."""
    static_names = ("ridge_fraction", "global_score_weight", "prefix_weight")
    output_names = tuple(field.name for field in fields(result_type) if field.name not in static_names)

    def decorate(function):
        api = signature(function)

        @wraps(function)
        def dispatch(*args, **kwargs):
            if tf.inside_function():
                return function(*args, **kwargs)
            bound = api.bind(*args, **kwargs)
            bound.apply_defaults()
            parent = bound.arguments["parent"]
            static = tuple((name, value) for name, value in bound.arguments.items()
                           if name != "parent" and (name in static_names or value is None))
            names = tuple(name for name, value in bound.arguments.items()
                          if name != "parent" and name not in static_names and value is not None)
            inputs = tuple(tf.convert_to_tensor(bound.arguments[name], D) for name in names)
            specifications = tuple(tf.TensorSpec(value.shape, D) for value in inputs)
            key = (function, id(parent), static, specifications)
            if key not in _INITIALIZERS:
                def numerical(*values):
                    record = function(parent=parent, **dict(static), **dict(zip(names, values, strict=True)))
                    return tuple(getattr(record, name) for name in output_names)

                forward = tf.function(numerical, input_signature=specifications, jit_compile=True, autograph=False)
                structure = forward.get_concrete_function().structured_outputs
                output_specs = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, D), structure)
                input_count = len(specifications)

                @tf.function(input_signature=[*specifications, *tf.nest.flatten(output_specs)],
                             jit_compile=True, autograph=False)
                def backward(*values):
                    query, cotangents = values[:input_count], values[input_count:]
                    with tf.GradientTape() as tape:
                        tape.watch(query)
                        outputs = tf.nest.flatten(numerical(*query))
                    return tape.gradient(outputs, query, output_gradients=cotangents,
                                         unconnected_gradients=tf.UnconnectedGradients.ZERO)

                @tf.custom_gradient
                def differentiable(*values):
                    def pullback(*cotangents):
                        return backward(*values, *cotangents)
                    outputs = forward(*tf.nest.map_structure(tf.stop_gradient, values))
                    return tuple(tf.nest.flatten(outputs)), pullback

                differentiable.compiled = forward
                _INITIALIZERS[key] = (parent, differentiable, structure)
                if len(_INITIALIZERS) > 16:
                    _INITIALIZERS.popitem(last=False)
            _INITIALIZERS.move_to_end(key)
            _, evaluate, structure = _INITIALIZERS[key]
            outputs = tf.nest.pack_sequence_as(structure, evaluate(*inputs))
            metadata = {name: bound.arguments[name] for name in static_names if name in bound.arguments}
            return result_type(**metadata, **dict(zip(output_names, outputs, strict=True)))

        return dispatch

    return decorate


def _integrals(basis, width):
    def branch(local):
        return lambda: tf.pad(local.integral_vector(basis.convention.mass_measure),
                              [[0, width-local.basis_dim]])

    if all(local is basis.bases[0] for local in basis.bases):
        return tf.broadcast_to(branch(basis.bases[0])(), [len(basis.bases), width])
    branches = tuple(branch(local) for local in basis.bases)
    return tf.map_fn(lambda axis: tf.switch_case(axis, branches), tf.range(len(basis.bases)),
                     fn_output_signature=tf.TensorSpec([width], D), parallel_iterations=1)


def _feature_values(basis, packed, points):
    dimension, _, width, _ = packed.shape
    values = tt._basis_table(basis, points, width, paired=False)
    additive = tf.reshape(tf.transpose(values, [1, 0, 2]), [points.shape[0], dimension * width])
    pair_width = basis.bases[0].basis_dim
    pairs = tf.einsum("pnl,pnm->pnlm", values[::2, :, :pair_width], values[1::2, :, :pair_width])
    pairs = tf.reshape(tf.transpose(pairs, [1, 0, 2, 3]), [points.shape[0], -1])
    return tf.concat([_select_columns(basis, additive, width), pairs], axis=1)


def _select_columns(basis, values, width):
    # Fixed heterogeneous basis widths describe the output schema.
    indices = tuple(axis * width + index for axis, local in enumerate(basis.bases)
                    for index in range(local.basis_dim))
    return tf.gather(values, tf.constant(indices, tf.int32), axis=1)


def _cross_features(basis, packed, points, *, include_pairs):
    dimension, rank, width, _ = packed.shape
    rows, prefix = points.shape
    full_points = tf.pad(points, [[0, 0], [0, dimension-prefix]])
    values = tt._basis_table(basis, full_points, width, paired=False)
    masses = tt._basis_table(basis, tf.zeros([1, 0], D), width, paired=True)[:, 0]
    observed = tf.range(dimension) < prefix
    pair_table = tf.where(observed[:, None, None, None],
        tf.einsum("dnl,dnm->dnlm", values, values), masses[:, None])
    if include_pairs:
        # The original feature TTs use all-ones coefficients on other axes.
        # Keep that exact finite function even for a non-cardinal basis.
        weights = tf.reduce_sum(pair_table, axis=-1)
    else:
        weights = tf.where(observed[:, None, None], values, _integrals(basis, width)[:, None])
    transfer = tf.einsum("dni,dlir->dnlr", weights, packed)
    initial = tf.broadcast_to(tf.one_hot(0, rank, dtype=D), [rows, rank])
    forward = tf.scan(lambda state, matrix: tf.einsum("nl,nlr->nr", state, matrix),
                      transfer, initializer=initial, parallel_iterations=1)
    backward = tf.scan(lambda state, matrix: tf.einsum("nlr,nr->nl", matrix, state),
                       transfer, initializer=initial, reverse=True, parallel_iterations=1)
    left = tf.concat([initial[None], forward[:-1]], axis=0)
    right = tf.concat([backward[1:], initial[None]], axis=0)
    single = tf.einsum("dnl,dlir,dnij,dnr->dnj", left, packed, pair_table, right)
    additive = tf.reshape(tf.transpose(single, [1, 0, 2]), [rows, dimension * width])
    additive = _select_columns(basis, additive, width)
    if not include_pairs:
        return additive
    pair_width = basis.bases[0].basis_dim
    pairs = tf.einsum("pnx,pxiy,pnij,pyhz,pnhk,pnz->pnjk",
        left[::2], packed[::2], pair_table[::2, :, :, :pair_width],
        packed[1::2], pair_table[1::2, :, :, :pair_width], right[1::2])
    pairs = tf.reshape(tf.transpose(pairs, [1, 0, 2, 3]), [rows, -1])
    return tf.concat([additive, pairs], axis=1)


def _numerical(basis, operation, packed, points):
    if operation == "values":
        return _feature_values(basis, packed, points)
    return _cross_features(basis, packed, points, include_pairs=operation == "pair_cross")


def _program(basis, operation, specifications):
    key = (id(basis), operation, specifications)
    if key in _PROGRAMS:
        _PROGRAMS.move_to_end(key)
        return _PROGRAMS[key][1]

    def evaluate(packed, points):
        return _numerical(basis, operation, packed, points)

    forward = tf.function(evaluate, input_signature=specifications, jit_compile=True, autograph=False)
    output = forward.get_concrete_function().output_shapes

    @tf.function(input_signature=[*specifications, tf.TensorSpec(output, D)],
                 jit_compile=True, autograph=False)
    def backward(packed, points, cotangent):
        with tf.GradientTape() as tape:
            tape.watch((packed, points))
            result = evaluate(packed, points)
        return tape.gradient(result, (packed, points), output_gradients=cotangent,
                             unconnected_gradients=tf.UnconnectedGradients.ZERO)

    @tf.custom_gradient
    def differentiable(packed, points):
        def pullback(cotangent):
            return backward(packed, points, cotangent)
        return forward(tf.stop_gradient(packed), tf.stop_gradient(points)), pullback

    differentiable.compiled = forward
    _PROGRAMS[key] = (basis, differentiable)
    if len(_PROGRAMS) > 16:
        _PROGRAMS.popitem(last=False)
    return differentiable


def feature_integrals(cores, basis, points=None, *, include_pairs=False):
    packed = tt.pack_components((cores,))[0]
    query = tf.zeros([1, 0], D) if points is None else tf.convert_to_tensor(points, D)
    operation = "pair_cross" if include_pairs else "additive_cross"
    result = dispatch(basis, operation, packed, query)
    return result[0] if points is None else result


def feature_values(basis, points):
    values = tf.convert_to_tensor(points, D)
    width = max(local.basis_dim for local in basis.bases)
    return dispatch(basis, "values", tf.zeros([len(basis.bases), 1, width, 1], D), values)


def dispatch(basis, operation, packed, points):
    if tf.inside_function():
        return _numerical(basis, operation, packed, points)
    specifications = (tf.TensorSpec(packed.shape, D), tf.TensorSpec(points.shape, D))
    return _program(basis, operation, specifications)(packed, points)

"""Compiled numerical kernels for the existing UKF-guided TT initializer.

The initializer remains an extension_or_invention, not Zhao-Cui admission.
Only fixed basis/rank schemas are packed on the host; frame inputs stay live.
"""

from collections import OrderedDict

import tensorflow as tf

from bayesfilter.highdim.bases import LegendreBasis1D, _legendre_values
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.ops.fixed_signature_tf import fixed_signature_function

_PROJECTION_CACHE = OrderedDict()
_EMBED_CACHE = OrderedDict()
_INITIALIZER_CACHE = OrderedDict()
D = tf.float64


@fixed_signature_function(floating_dtype=D)
def stabilize_covariance(covariance, abs_floor, rel_floor):
    matrix = 0.5 * (covariance + tf.transpose(covariance))
    raw, eigenvectors = tf.linalg.eigh(matrix)
    floor = tf.maximum(abs_floor, rel_floor * tf.reduce_max(tf.abs(raw)))
    floored = tf.maximum(raw, floor)
    stabilized = eigenvectors @ tf.linalg.diag(floored) @ tf.transpose(eigenvectors)
    return 0.5 * (stabilized + tf.transpose(stabilized)), eigenvectors, raw, floored, floor


@fixed_signature_function(floating_dtype=D)
def local_frame(covariance, gamma, abs_floor, rel_floor):
    stabilized = stabilize_covariance.python_function(covariance, abs_floor, rel_floor)
    linear_map = gamma * stabilized[1] @ tf.linalg.diag(tf.sqrt(stabilized[3]))
    return linear_map, stabilized


def projection_program(product_basis, order, *, jit_compile=True):
    key = (id(product_basis), order, bool(jit_compile))
    if key in _PROJECTION_CACHE:
        _PROJECTION_CACHE.move_to_end(key)
        return _PROJECTION_CACHE[key][1]
    dimension = product_basis.dimension
    widths = tuple(int(basis.basis_dim) for basis in product_basis.bases)
    width = max(widths)
    bounds = tf.stack(tuple((basis.domain.left, basis.domain.right) for basis in product_basis.bases))
    measure = product_basis.convention.mass_measure
    if measure not in (MassMeasure.REFERENCE_MEASURE, MassMeasure.REFERENCE_LEBESGUE):
        raise ValueError("unsupported UKF projection mass measure")
    legendre = all(isinstance(basis, LegendreBasis1D) for basis in product_basis.bases)

    @tf.function(input_signature=[tf.TensorSpec([], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension, dimension], D)], jit_compile=jit_compile, autograph=False)
    def evaluate(gamma, center, linear_map, reference_offset, reference_matrix):
        from bayesfilter.highdim.ukf_initializer import _legendre_gauss_nodes_weights

        nodes, weights = _legendre_gauss_nodes_weights(order)
        endpoints = bounds
        half_length = 0.5 * (endpoints[:, 1] - endpoints[:, 0])
        midpoint = 0.5 * (endpoints[:, 0] + endpoints[:, 1])
        points = midpoint[:, None] + half_length[:, None] * nodes[None, :]
        scale = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(linear_map), axis=1)), tf.constant(1e-15, D))
        physical = reference_offset[:, None] + tf.linalg.diag_part(reference_matrix)[:, None] * points
        standardized = (physical - center[:, None]) / scale[:, None]
        active_weights = tf.broadcast_to(0.5 * weights, [dimension, order])
        if measure is MassMeasure.REFERENCE_LEBESGUE:
            active_weights = half_length[:, None] * weights[None, :]
        squared = tf.square(gamma) * tf.square(standardized)
        normalizer = tf.reduce_sum(active_weights * tf.exp(-0.5 * squared), axis=1)
        values = tf.exp(-0.25 * squared) / tf.sqrt(normalizer[:, None])
        if legendre:
            xi = 2.0 * (points - endpoints[:, :1]) / (endpoints[:, 1:] - endpoints[:, :1]) - 1.0
            basis_values = _legendre_values(xi, width - 1) * tf.sqrt(tf.cast(2 * tf.range(width) + 1, D))
            mask = tf.sequence_mask(widths, width, dtype=D)
            basis_values *= mask[:, None, :]
            diagonal = tf.ones([dimension, width], D)
            if measure is MassMeasure.REFERENCE_LEBESGUE:
                diagonal *= (2.0 * half_length)[:, None]
            masses = tf.linalg.diag(diagonal)
        else:
            def bind_basis(axis):
                basis = product_basis.bases[axis]
                count = widths[axis]

                def constants():
                    basis_values = tf.pad(basis.evaluate(points[axis]), [[0, 0], [0, width - count]])
                    mass = tf.pad(basis.mass_matrix(measure), [[0, width - count], [0, width - count]])
                    mass += tf.linalg.diag(tf.cast(tf.range(width) >= count, D))
                    return basis_values, mass

                return constants

            branches = tuple(bind_basis(axis) for axis in range(dimension))
            basis_values, masses = tf.map_fn(
                lambda axis: tf.switch_case(axis, branches), tf.range(dimension),
                fn_output_signature=(tf.TensorSpec([order, width], D), tf.TensorSpec([width, width], D)),
                parallel_iterations=1,
            )
        rhs = tf.reduce_sum(active_weights[:, :, None] * values[:, :, None] * basis_values, axis=1)
        coefficients = tf.linalg.solve(masses, rhs[:, :, None])[:, :, 0]
        valid = tf.reduce_all(normalizer > 0.0) & tf.reduce_all(tf.math.is_finite(coefficients))
        return coefficients, valid

    _PROJECTION_CACHE[key] = (product_basis, evaluate)
    if len(_PROJECTION_CACHE) > 16:
        _PROJECTION_CACHE.popitem(last=False)
    return evaluate


def embedding_program(widths, ranks, *, jit_compile=True):
    key = (widths, ranks, bool(jit_compile))
    if key in _EMBED_CACHE:
        _EMBED_CACHE.move_to_end(key)
        return _EMBED_CACHE[key]
    dimension, width, rank = len(widths), max(widths), max(ranks)

    @tf.function(input_signature=[tf.TensorSpec([dimension, width], D), tf.TensorSpec([], D)],
                 jit_compile=jit_compile, autograph=False)
    def evaluate(coefficients, epsilon):
        zero = tf.one_hot(0, rank, dtype=D)
        values = coefficients[:, None, :, None] * zero[None, :, None, None] * zero[None, None, None, :]
        axis = tf.range(dimension)[:, None]
        channel = tf.range(rank)[None, :]
        sizes = tf.constant(widths)[:, None]
        basis_index = tf.where(sizes > 1, 1 + tf.math.floormod(axis + channel - 1, tf.maximum(sizes - 1, 1)), 0)
        basis = tf.one_hot(basis_index, width, dtype=D)
        left = tf.constant(ranks[:-1])[:, None]
        right = tf.constant(ranks[1:])[:, None]
        diagonal = tf.cast((channel > 0) & (channel < tf.minimum(left, right)), D)
        values += basis[:, :, :, None] * tf.eye(rank, dtype=D)[None, :, None, :] * diagonal[:, :, None, None]
        first = tf.cast((axis == 0) & (channel > 0) & (channel < right), D)
        scale = tf.maximum(tf.abs(coefficients[0, 0]), tf.constant(1e-300, D)) * epsilon
        values += zero[None, :, None, None] * tf.transpose(basis * first[:, :, None], [0, 2, 1])[:, None, :, :] * scale
        last = tf.cast((axis == dimension - 1) & (channel > 0) & (channel < left), D)
        values += basis[:, :, :, None] * last[:, :, None, None] * zero[None, None, None, :]
        return values

    _EMBED_CACHE[key] = evaluate
    if len(_EMBED_CACHE) > 16:
        _EMBED_CACHE.popitem(last=False)
    return evaluate


def initializer_program(product_basis, ranks, order, *, jit_compile=True):
    key = (id(product_basis), ranks, order, bool(jit_compile))
    if key in _INITIALIZER_CACHE:
        _INITIALIZER_CACHE.move_to_end(key)
        return _INITIALIZER_CACHE[key][1]
    dimension = product_basis.dimension
    widths = tuple(int(basis.basis_dim) for basis in product_basis.bases)
    project = projection_program(product_basis, order, jit_compile=jit_compile).python_function
    embed = embedding_program(widths, ranks, jit_compile=jit_compile).python_function

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension, dimension], D),
        tf.TensorSpec([], D), tf.TensorSpec([], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([], D)],
        jit_compile=jit_compile, autograph=False)
    def evaluate(center, covariance, gamma, abs_floor, rel_floor, offset, reference_matrix, epsilon_per_channel):
        linear_map, stabilized = local_frame.python_function(covariance, gamma, abs_floor, rel_floor)
        coefficients, valid = project(gamma, center, linear_map, offset, reference_matrix)
        cores = embed(coefficients, epsilon_per_channel)
        return coefficients, cores, linear_map, stabilized, valid

    _INITIALIZER_CACHE[key] = (product_basis, evaluate)
    if len(_INITIALIZER_CACHE) > 16:
        _INITIALIZER_CACHE.popitem(last=False)
    return evaluate

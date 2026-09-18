"""Compiled numerical programs for the public squared-TT density interface.

The paired-core contraction and the existing 33-point suffix trapezoid rule
are unchanged. In particular the conditional rule is a local grid method,
not a replacement by the paper's algebraic conditional construction.
"""

from collections import OrderedDict

import tensorflow as tf

from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import prefix_rows, squared_marginal
from bayesfilter.ops.fixed_signature_tf import fixed_signature_function

_DENSITY_CACHE = OrderedDict()


@fixed_signature_function(floating_dtype=tf.float64)
def reference_volume(bounds):
    """Ordered product of interval lengths, including the empty product."""
    return tf.foldl(
        lambda volume, bound: volume * (bound[1] - bound[0]),
        bounds,
        initializer=tf.constant(1.0, tf.float64),
        parallel_iterations=1,
    )


def density_program(density, operation, *, points=None, keep_axes=(), axis=None,
                    prefix=None, grid=None, chunk_size=None, jit_compile=True):
    """Return a bounded fixed-signature program and its numerical arguments.

    The immutable defensive callable and basis schema are captured. Core
    coefficients and mixture weight are inputs, including on cache reuse.
    Host validity checks consume the returned values and normalizers.
    """
    from bayesfilter.highdim.squared_tt import (
        TensorProductReferenceDensity,
        _points_with_prefix_axis_grid,
        _tensor_product_grid_and_weights,
        _trapezoid_integral,
    )

    basis = density.sqrt_tt.product_basis
    cores = density.sqrt_tt.cores
    defensive = density.defensive_density
    shapes = tuple(tuple(core.values.shape) for core in cores)
    extra = (prefix, grid) if grid is not None else (() if points is None else (points,))
    if operation == "retained_chunked":
        if points.shape[0] < 1 or chunk_size is None or chunk_size < 1:
            raise ValueError("retained chunks require positive rows and chunk size")
        chunk_size = min(int(chunk_size), int(points.shape[0]))
    key = (id(basis), id(defensive), shapes, operation, tuple(keep_axes), axis, chunk_size,
           tuple(tuple(value.shape) for value in extra), bool(jit_compile))
    if key in _DENSITY_CACHE:
        _DENSITY_CACHE.move_to_end(key)
        program = _DENSITY_CACHE[key][2]
    else:
        specs = (tuple(tf.TensorSpec(shape, tf.float64) for shape in shapes),
                 tf.TensorSpec([], tf.float64),
                 *(tf.TensorSpec(value.shape, tf.float64) for value in extra))

        @tf.function(input_signature=specs, jit_compile=jit_compile, autograph=False)
        def program(values, tau, *queries):
            if operation in ("marginal", "normalized_marginal", "defensive_marginal"):
                query = queries[0]
                if keep_axes == tuple(range(len(cores))):
                    defensive_values = tf.exp(defensive.log_density(query))
                elif isinstance(defensive, TensorProductReferenceDensity):
                    if not keep_axes:
                        raise ValueError("ProductBasis requires at least one basis")
                    volume = tf.constant(1.0, tf.float64)
                    if basis.convention.mass_measure is MassMeasure.REFERENCE_LEBESGUE:
                        bounds = tuple((part.domain.left, part.domain.right)
                                       for index, part in enumerate(basis.bases)
                                       if index not in keep_axes)
                        volume = reference_volume.python_function(tf.reshape(tf.stack(bounds), [-1, 2]))
                    defensive_values = tf.fill([query.shape[0]], (1.0 + defensive.floor) * volume)
                else:
                    raise NotImplementedError("source-style defensive marginal requires tensor-product reference density")
                if operation == "defensive_marginal":
                    return defensive_values

            typed = tuple(TTCore(value) for value in values)
            square_z = squared_marginal(typed, basis, (), tf.zeros([1, 0], tf.float64))[0]
            if operation == "sqrt_normalizer":
                return square_z
            z = square_z + tau * defensive.normalizer(basis.convention.mass_measure)
            if operation == "normalizer":
                return z

            def unnormalized(query):
                h = prefix_rows(typed, basis, query)[0][:, 0]
                return tf.square(h) + tau * tf.exp(defensive.log_density(query))

            def log_values(query):
                return tf.math.log(unnormalized(query)) - tf.math.log(z)

            if operation == "unnormalized":
                return unnormalized(queries[0])
            if operation in ("log_density", "normalized_retained"):
                raw = unnormalized(queries[0])
                result = tf.math.log(raw) - tf.math.log(z)
                if operation == "normalized_retained":
                    result = tf.exp(result)
                return result, z, raw
            if operation == "retained_chunked":
                query = queries[0]
                count = query.shape[0]
                blocks = (count + chunk_size - 1) // chunk_size
                outputs = tf.TensorArray(tf.float64, blocks, element_shape=[chunk_size])
                axes = tuple(range(len(cores)))

                def step(index, outputs):
                    # Repeat the final valid row to keep the last block XLA-static.
                    indices = tf.minimum(index * chunk_size + tf.range(chunk_size), count - 1)
                    block = tf.gather(query, indices)
                    numerator = squared_marginal(typed, basis, axes, block)
                    numerator += tau * tf.exp(defensive.log_density(block))
                    return index + 1, outputs.write(index, numerator / z)

                _, outputs = tf.while_loop(lambda index, _: index < blocks, step,
                    (tf.constant(0), outputs), maximum_iterations=blocks, parallel_iterations=1)
                return tf.reshape(outputs.stack(), [-1])[:count], z
            if operation in ("marginal", "normalized_marginal"):
                query = queries[0]
                numerator = squared_marginal(typed, basis, keep_axes, query) + tau * defensive_values
                if operation == "normalized_marginal":
                    return numerator / z, z
                return numerator, z

            if operation not in ("prefix", "conditional"):
                raise ValueError("unknown squared-TT operation")
            prefix_values, grid_values = queries
            dimension = len(cores)
            if axis == dimension - 1:
                query = _points_with_prefix_axis_grid(dimension, axis, prefix_values, grid_values, None)
                rows = tf.exp(log_values(query))
            else:
                # Pack heterogeneous domain endpoints; all grid arithmetic and
                # row evaluations execute in native tensor control flow.
                bounds = tf.stack(tuple((part.domain.left, part.domain.right)
                                        for part in basis.bases[axis+1:]))
                grids = tf.transpose(tf.linspace(bounds[:, 0], bounds[:, 1], 33))
                suffix_points, suffix_weights = _tensor_product_grid_and_weights(tf.unstack(grids))
                count = grid_values.shape[0]
                output = tf.TensorArray(tf.float64, size=count, element_shape=[])

                def step(index, output):
                    query = _points_with_prefix_axis_grid(
                        dimension, axis, prefix_values, grid_values[index:index+1], suffix_points)
                    row = tf.reduce_sum(tf.exp(log_values(query)) * suffix_weights)
                    return index + 1, output.write(index, row)

                _, output = tf.while_loop(lambda index, _: index < count, step,
                    (tf.constant(0), output), maximum_iterations=count, parallel_iterations=1)
                rows = output.stack()
            if operation == "prefix":
                return rows, z
            integral = _trapezoid_integral(grid_values, rows)
            return rows / integral, integral, z

        _DENSITY_CACHE[key] = (basis, defensive, program)
        if len(_DENSITY_CACHE) > 16:
            _DENSITY_CACHE.popitem(last=False)
    return program, (tuple(core.values for core in cores), density.tau, *extra)


def evaluate_density(density, operation, **kwargs):
    program, arguments = density_program(density, operation, **kwargs)
    evaluate = program.python_function if tf.inside_function() else program
    return evaluate(*arguments)

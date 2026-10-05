"""Shared coordinate graph for the existing grid-CDF TTSIRT extension.

Tensor masks select retained coordinates inside one paired-core recurrence.
The contraction, trapezoid, interpolation and bisection rules are unchanged.
Python constructs only the heterogeneous basis/domain and tensor schemas.
"""

import tensorflow as tf

from bayesfilter.highdim.bases import (
    BoundedInterval,
    LegendreBasis1D,
    ProductBasis,
    _legendre_values,
)
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.squared_tt import TensorProductReferenceDensity
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import _basis_axis_program, basis_masses
from bayesfilter.ops.compiled_tensor_program_tf import tensor_program

D = tf.float64


def _shared_legendre_marginal(basis, cores, points, keep, rows):
    """The same paired-core recurrence for equal-shaped Legendre cores.

Bounds are tensors, so axes can have distinct intervals and retain their
captured derivatives. Only the common degree/core schema is specialized.
"""
    dimension = len(cores)
    width = cores[0].basis_dim
    packed = tf.stack(tuple(core.values for core in cores))
    bounds = tf.stack(tuple((part.domain.left, part.domain.right) for part in basis.bases))
    scales = tf.sqrt(tf.cast(2 * tf.range(width, dtype=tf.int32) + 1, D))

    def basis_values(coordinate, left, right):
        # LegendreBasis1D.evaluate and BoundedInterval.to_reference, with
        # their original operation order and shared polynomial recurrence.
        xi = 2.0 * (coordinate - left) / (right - left) - 1.0
        return _legendre_values(xi, width - 1) * scales

    # Keep the polynomial recurrence and its complete pullback in one common
    # boundary. A bare recurrence leaks variant tapes through the outer Cond.
    basis_program = tensor_program(basis_values,
        [tf.TensorSpec([rows], D), tf.TensorSpec([], D), tf.TensorSpec([], D)], False)

    def step(axis, state):
        core = packed[axis]
        length = bounds[axis, 1] - bounds[axis, 0]

        def retained():
            phi = basis_program.python_function(points[:, axis], bounds[axis, 0], bounds[axis, 1])
            return tf.einsum("nl,nm,alb,AmB->naAbB", phi, phi, core, core)

        def integrated():
            mass = tf.eye(width, dtype=D)
            if basis.convention.mass_measure is MassMeasure.REFERENCE_LEBESGUE:
                mass *= length
            paired = tf.einsum("alb,AmB,lm->aAbB", core, core, mass)
            return tf.broadcast_to(paired[None], [rows, *paired.shape])

        paired = tf.cond(keep[axis], retained, integrated)
        matrix = tf.reshape(paired, [rows, 1, 1])
        return axis + 1, tf.einsum("na,nab->nb", state, matrix)

    start = tf.one_hot(tf.zeros([rows], tf.int32), 1, dtype=D)
    _, state = tf.while_loop(lambda axis, _: axis < dimension, step,
        (tf.constant(0), start), maximum_iterations=dimension, parallel_iterations=1)
    return state


def masked_marginal_program(density, rows, *, jit_compile=True):
    """Evaluate nonempty transport marginals with a fixed full-query shape."""
    basis = density.sqrt_tt.product_basis
    shapes = tuple(tuple(core.values.shape) for core in density.sqrt_tt.cores)
    dimension = len(shapes)
    width = max(shape[1] for shape in shapes)
    rank = max(max(shape[0], shape[2]) for shape in shapes) ** 2
    shared_legendre = (type(basis) is ProductBasis and rank == 1
        and all(shape == shapes[0] for shape in shapes)
        and all(type(part) is LegendreBasis1D and type(part.domain) is BoundedInterval
                and part.basis_dim == width for part in basis.bases))
    defensive = density.defensive_density
    if dimension > 1 and not isinstance(defensive, TensorProductReferenceDensity):
        raise NotImplementedError("source-style defensive marginal requires tensor-product reference density")
    signature = (*tuple(tf.TensorSpec(shape, D) for shape in shapes),
                 tf.TensorSpec([], D), tf.TensorSpec([rows, dimension], D),
                 tf.TensorSpec([dimension], tf.bool))

    def evaluate(*arguments):
        cores = tuple(TTCore(value) for value in arguments[:dimension])
        tau, points, keep = arguments[dimension:]
        masses = None if shared_legendre else basis_masses(basis, cores)

        def branch(axis):
            core = cores[axis]

            def retained():
                local = _basis_axis_program(basis, axis, rows, width, D)
                phi = local.python_function(points[:, axis])[:, :core.basis_dim]
                return tf.einsum("nl,nm,alb,AmB->naAbB", phi, phi, core.values, core.values)

            def integrated():
                paired = tf.einsum("alb,AmB,lm->aAbB", core.values, core.values, masses[axis])
                return tf.broadcast_to(paired[None], [rows, *paired.shape])

            def matrix():
                paired = tf.cond(keep[axis], retained, integrated)
                value = tf.reshape(paired, [rows, core.left_rank**2, core.right_rank**2])
                return tf.pad(value, [[0, 0], [0, rank-core.left_rank**2],
                                      [0, rank-core.right_rank**2]])

            return matrix

        if shared_legendre:
            state = _shared_legendre_marginal(basis, cores, points, keep, rows)
        else:
            branches = tuple(branch(axis) for axis in range(dimension))
            start = tf.one_hot(tf.zeros([rows], tf.int32), rank, dtype=D)

            def step(axis, state):
                return axis+1, tf.einsum("na,nab->nb", state, tf.switch_case(axis, branches))

            _, state = tf.while_loop(lambda axis, _: axis < dimension, step,
                (tf.constant(0), start), maximum_iterations=dimension, parallel_iterations=1)

        if isinstance(defensive, TensorProductReferenceDensity):
            volume = tf.constant(1.0, D)
            if basis.convention.mass_measure is MassMeasure.REFERENCE_LEBESGUE:
                bounds = tf.stack(tuple((part.domain.left, part.domain.right) for part in basis.bases))
                _, volume = tf.while_loop(lambda axis, _: axis < dimension,
                    lambda axis, value: (axis+1, tf.cond(keep[axis], lambda: value,
                        lambda: value * (bounds[axis, 1]-bounds[axis, 0]))),
                    (tf.constant(0), volume), maximum_iterations=dimension, parallel_iterations=1)
            # Both expressions are finite on valid inputs. Keep the volume
            # loop outside this selection so Case gradients need no unused
            # branch TensorList/FakeParam output across the XLA boundary.
            defensive_values = tf.where(tf.reduce_all(keep),
                tf.exp(defensive.log_density(points)),
                tf.fill([rows], (1.0 + defensive.floor) * volume))
        else:
            # A one-dimensional transport has only its full numerator. The
            # empty denominator is the unit density and never calls this path.
            defensive_values = tf.exp(defensive.log_density(points))
        return state[:, 0] + tau * defensive_values

    return tensor_program(evaluate, signature, jit_compile)


def coordinate_program(transport, mode, count, *, suffix=False, jit_compile=True):
    """One coordinate body, retaining the complete local recomputed pullback."""
    from bayesfilter.highdim.transport import _interp_rows
    from bayesfilter.highdim.ttsirt_native_tf import _status

    dimension = transport.dimension
    config = transport.cdf_config
    grid_size = config.grid_size
    shapes = tuple(tuple(core.values.shape) for core in transport.density.sqrt_tt.cores)
    signature = (*tuple(tf.TensorSpec(shape, D) for shape in shapes),
                 *(tf.TensorSpec([], D) for _ in range(4)),
                 tf.TensorSpec([dimension, count], D), tf.TensorSpec([count], D),
                 tf.TensorSpec([], tf.int32))
    numerator = masked_marginal_program(transport.density, count*grid_size, jit_compile=False)
    denominator = masked_marginal_program(transport.density, count, jit_compile=False)

    def axis_call(function, axis, values):
        branches = tuple(lambda index=index: function(index, values) for index in range(dimension))
        return tf.switch_case(axis, branches)

    def evaluate(*arguments):
        cores = arguments[:dimension]
        tau, normalizer_floor, denominator_floor, z, state, targets, axis = arguments[dimension:]
        # FixedTTSIRTTransport defines one reference interval and grid size for
        # every coordinate; _axis_grid deliberately ignores its axis argument.
        grid = transport._axis_grid(0)
        physical_grid = axis_call(transport._axis_reference_to_domain, axis, grid)
        coordinates = tf.range(dimension)
        numerator_keep = coordinates >= axis if suffix else coordinates <= axis
        denominator_keep = coordinates > axis if suffix else coordinates < axis
        query = tf.where((coordinates == axis)[None, None, :],
                         physical_grid[None, :, None], tf.transpose(state)[:, None, :])
        raw = numerator.python_function(*cores, tau,
            tf.reshape(query, [count*grid_size, dimension]), numerator_keep)
        numerator_values = tf.reshape(raw/z, [count, grid_size])
        denominator_values = tf.cond(tf.reduce_any(denominator_keep),
            lambda: denominator.python_function(*cores, tau, tf.transpose(state), denominator_keep)/z,
            lambda: tf.ones([count], D))
        code = _status(tf.constant(0), tf.math.is_finite(z), 1)
        code = _status(code, z > normalizer_floor, 2)
        code = _status(code, tf.reduce_all(tf.math.is_finite(denominator_values)
                                         & (denominator_values > denominator_floor)), 3)
        conditional = numerator_values/denominator_values[:, None]
        if mode != "inverse" and not suffix:
            code = _status(code, tf.reduce_all(tf.math.is_finite(conditional)), 1)
        increments = 0.5*(conditional[:, 1:]+conditional[:, :-1])*(grid[1:]-grid[:-1])[None]
        cdf = tf.concat([tf.zeros([count, 1], D), tf.cumsum(increments, axis=1)], axis=1)
        totals = cdf[:, -1]
        if mode != "inverse" and not suffix:
            code = _status(code, tf.reduce_all(tf.math.is_finite(totals)), 1)
        code = _status(code, tf.reduce_all(tf.math.is_finite(cdf))
                       & tf.reduce_all(totals > config.denominator_floor), 3)
        cdf /= totals[:, None]
        minimum = tf.reduce_min(cdf[:, 1:]-cdf[:, :-1])
        code = _status(code, tf.math.is_finite(minimum), 1)
        code = _status(code, minimum >= -config.monotonicity_tolerance, 4)
        if mode != "inverse":
            reference = axis_call(transport._axis_domain_to_reference, axis, targets)
            if mode == "forward":
                return _interp_rows(reference, grid, cdf), code
            if suffix:
                table = conditional / totals[:, None]
                right = tf.clip_by_value(tf.searchsorted(grid, reference, side="right"), 1, grid_size-1)
                left = right-1
                rows = tf.range(count)
                y0 = tf.gather_nd(table, tf.stack([rows, left], 1))
                y1 = tf.gather_nd(table, tf.stack([rows, right], 1))
                value = y0 + (reference-tf.gather(grid, left))/(tf.gather(grid, right)-tf.gather(grid, left))*(y1-y0)
            else:
                value = _interp_rows(reference, grid, conditional)/totals
            jacobian = axis_call(transport._axis_domain_to_reference_jacobian, axis, targets)
            return tf.math.log(value * jacobian), code
        lo, hi = tf.fill([count], grid[0]), tf.fill([count], grid[-1])

        def bisect(index, lo, hi, mid):
            mid = 0.5*(lo+hi)
            choose_right = _interp_rows(mid, grid, cdf) < targets
            return index+1, tf.where(choose_right, mid, lo), tf.where(choose_right, hi, mid), mid

        _, _, _, midpoint = tf.while_loop(lambda index, *_: index < config.bisection_steps,
            bisect, (tf.constant(0), lo, hi, 0.5*(lo+hi)),
            maximum_iterations=config.bisection_steps, parallel_iterations=1)
        return axis_call(transport._axis_reference_to_domain, axis, midpoint), code

    return tensor_program(evaluate, signature, jit_compile)

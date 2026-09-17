"""Native execution of the existing grid-CDF TTSIRT numerical operations.

This preserves the local trapezoid/bisection extension, its coordinate order,
and its veto thresholds. It does not implement the author's algebraic CDF.
"""

from collections import OrderedDict

import tensorflow as tf

from bayesfilter.highdim.diagnostics import HighDimStatus
from bayesfilter.highdim.squared_tt_density_native_tf import density_program

_PROGRAMS = OrderedDict()
_ERRORS = (
    HighDimStatus.OK,
    HighDimStatus.NONFINITE_VALUE,
    HighDimStatus.NORMALIZER_FLOOR_EXCEEDED,
    HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED,
    HighDimStatus.CDF_MONOTONICITY_FAILURE,
    HighDimStatus.INVERSE_BRACKET_FAILURE,
)


def _status(current, valid, code):
    return tf.where((current == 0) & ~valid, tf.constant(code, tf.int32), current)


def transport_arguments(transport):
    density = transport.density
    return (tuple(core.values for core in density.sqrt_tt.cores), density.tau,
            density.normalizer_floor, density.denominator_floor)


def transport_program(transport, operation, count, conditioning_dimension=0, *, jit_compile=True):
    """Bind a stable enclosing program with all core values as inputs."""
    from bayesfilter.highdim.transport import _interp_rows

    key = (id(transport), operation, count, conditioning_dimension, bool(jit_compile))
    if key in _PROGRAMS:
        _PROGRAMS.move_to_end(key)
        return _PROGRAMS[key][1]
    dimension, dx = transport.dimension, conditioning_dimension
    dr = dimension - dx
    suffix = operation.endswith("_suffix")
    mode = operation.removesuffix("_suffix")
    axes = tuple(range(dr-1, -1, -1)) if suffix else tuple(range(dx, dimension))
    config = transport.cdf_config
    grid_size = config.grid_size
    if mode in ("inverse", "forward", "log_jacobian"):
        for axis in axes:
            report = transport.batch_working_set_estimate(axis=axis, sample_count=count)
            if report["estimated_bytes"] > config.max_batch_working_bytes:
                raise ValueError("KR batch working set exceeds max_batch_working_bytes")

    def marginal_kernel(axes, rows):
        return density_program(transport.density, "marginal", keep_axes=axes,
            points=tf.TensorSpec([rows, len(axes)], tf.float64), jit_compile=jit_compile)[0].python_function

    full_pdf = density_program(transport.density, "log_density",
        points=tf.TensorSpec([count, dimension], tf.float64), jit_compile=jit_compile)[0].python_function
    condition_axes = tuple(range(dr, dimension)) if suffix else tuple(range(dx))
    prefix_pdf = marginal_kernel(condition_axes, count) if dx else None
    shapes = tuple(core.values.shape for core in transport.density.sqrt_tt.cores)
    specs = (tuple(tf.TensorSpec(shape, tf.float64) for shape in shapes),
             tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.float64),
             tf.TensorSpec([dx, count], tf.float64), tf.TensorSpec([dr, count], tf.float64))

    def reference_density(points, selected_axes):
        def branch(axis):
            return lambda: transport._axis_reference_measure_density(axis, points[axis])
        branches = tuple(branch(axis) for axis in selected_axes)
        if not branches:
            return tf.ones([count], tf.float64)
        _, product = tf.while_loop(lambda index, _: index < len(branches),
            lambda index, value: (index+1, value*tf.switch_case(index, branches)),
            (tf.constant(0), tf.ones([count], tf.float64)), maximum_iterations=len(branches), parallel_iterations=1)
        return product

    def coordinate_branch(axis):
        numerator_axes = tuple(range(axis, dimension)) if suffix else tuple(range(axis+1))
        denominator_axes = tuple(range(axis+1, dimension)) if suffix else tuple(range(axis))
        numerator_kernel = marginal_kernel(numerator_axes, count*grid_size)
        denominator_kernel = marginal_kernel(denominator_axes, count) if denominator_axes else None

        def calculate(cores, tau, normalizer_floor, denominator_floor, state, targets):
            grid = transport._axis_grid(axis)
            known = tf.transpose(state[axis+1:] if suffix else state[:axis])
            physical_grid = transport._axis_reference_to_domain(axis, grid)
            tiled_known = tf.repeat(known[:, None, :], repeats=grid_size, axis=1)
            tiled_grid = tf.broadcast_to(physical_grid[None, :, None], [count, grid_size, 1])
            parts = [tiled_grid, tiled_known] if suffix else [tiled_known, tiled_grid]
            query = tf.reshape(tf.concat(parts, axis=2), [count*grid_size, len(numerator_axes)])
            numerator, z = numerator_kernel(cores, tau, query)
            numerator = tf.reshape(numerator/z, [count, grid_size])
            denominator = tf.ones([count], tf.float64)
            if denominator_axes:
                raw_denominator, _ = denominator_kernel(cores, tau, known)
                denominator = raw_denominator/z
            code = _status(tf.constant(0), tf.math.is_finite(z), 1)
            code = _status(code, z > normalizer_floor, 2)
            code = _status(code, tf.reduce_all(tf.math.is_finite(denominator) & (denominator > denominator_floor)), 3)
            conditional = numerator/denominator[:, None]
            if mode != "inverse" and not suffix:
                code = _status(code, tf.reduce_all(tf.math.is_finite(conditional)), 1)
            increments = 0.5*(conditional[:, 1:]+conditional[:, :-1])*(grid[1:]-grid[:-1])[None]
            cdf = tf.concat([tf.zeros([count, 1], tf.float64), tf.cumsum(increments, axis=1)], axis=1)
            totals = cdf[:, -1]
            if mode != "inverse" and not suffix:
                code = _status(code, tf.reduce_all(tf.math.is_finite(totals)), 1)
            code = _status(code, tf.reduce_all(tf.math.is_finite(cdf)) & tf.reduce_all(totals > config.denominator_floor), 3)
            cdf /= totals[:, None]
            minimum = tf.reduce_min(cdf[:, 1:]-cdf[:, :-1])
            code = _status(code, tf.math.is_finite(minimum), 1)
            code = _status(code, minimum >= -config.monotonicity_tolerance, 4)
            if mode != "inverse":
                reference = transport._axis_domain_to_reference(axis, targets)
                if mode == "forward":
                    return _interp_rows(reference, grid, cdf), code
                if suffix:
                    # The existing suffix Jacobian extrapolates its endpoint
                    # density, although its CDF clips. Preserve that behavior.
                    table = conditional / totals[:, None]
                    right = tf.clip_by_value(tf.searchsorted(grid, reference, side="right"), 1, grid_size-1)
                    left = right-1
                    rows = tf.range(count)
                    y0 = tf.gather_nd(table, tf.stack([rows, left], 1))
                    y1 = tf.gather_nd(table, tf.stack([rows, right], 1))
                    value = y0 + (reference-tf.gather(grid, left))/(tf.gather(grid, right)-tf.gather(grid, left))*(y1-y0)
                else:
                    value = _interp_rows(reference, grid, conditional)/totals
                return tf.math.log(value * transport._axis_domain_to_reference_jacobian(axis, targets)), code
            lo, hi = tf.fill([count], grid[0]), tf.fill([count], grid[-1])

            def bisect(index, lo, hi, mid):
                mid = 0.5*(lo+hi)
                choose_right = _interp_rows(mid, grid, cdf) < targets
                return index+1, tf.where(choose_right, mid, lo), tf.where(choose_right, hi, mid), mid

            _, _, _, midpoint = tf.while_loop(lambda index, *_: index < config.bisection_steps,
                bisect, (tf.constant(0), lo, hi, 0.5*(lo+hi)),
                maximum_iterations=config.bisection_steps, parallel_iterations=1)
            return transport._axis_reference_to_domain(axis, midpoint), code
        return calculate

    coordinate_branches = tuple(coordinate_branch(axis) for axis in axes) if mode in ("inverse", "forward", "log_jacobian") else ()

    @tf.function(input_signature=specs, jit_compile=jit_compile, autograph=False)
    def evaluate(cores, tau, normalizer_floor, denominator_floor, condition, values):
        code = _status(tf.constant(0), tf.reduce_all(tf.math.is_finite(condition)) & tf.reduce_all(tf.math.is_finite(values)), 1)
        if coordinate_branches:
            if mode == "inverse":
                code = _status(code, tf.reduce_all((values >= 0.0) & (values <= 1.0)), 5)
                generated = tf.zeros([dr, count], tf.float64)
            else:
                generated = values
            state = tf.concat([generated, condition] if suffix else [condition, generated], axis=0)
            outputs = tf.zeros([dr, count], tf.float64)

            def step(index, state, outputs, code):
                value_index = dr-1-index if suffix else index
                axis = value_index if suffix else dx+index
                branches = tuple(lambda fn=fn: fn(cores, tau, normalizer_floor, denominator_floor,
                    state, values[value_index]) for fn in coordinate_branches)
                following, new_code = tf.switch_case(index, branches)
                if mode == "inverse":
                    state = tf.tensor_scatter_nd_update(state, [[axis]], [following])
                outputs = tf.tensor_scatter_nd_update(outputs, [[value_index]], [following])
                return index+1, state, outputs, tf.where(code == 0, new_code, code)

            _, _, outputs, code = tf.while_loop(lambda index, *_: index < dr, step,
                (tf.constant(0), state, outputs, code), maximum_iterations=dr, parallel_iterations=1)
            if mode == "log_jacobian":
                if suffix:
                    outputs = tf.foldl(lambda total, row: total+row, tf.reverse(outputs, [0]),
                        initializer=tf.zeros([count], tf.float64), parallel_iterations=1)
                    code = _status(code, tf.reduce_all(tf.math.is_finite(outputs)), 1)
                else:
                    outputs = tf.reduce_sum(outputs, axis=0)
            return outputs, code
        if dr == 0 and mode in ("inverse", "forward", "log_jacobian"):
            return (tf.zeros([count], tf.float64) if mode == "log_jacobian" else values), code
        joint = tf.concat([values, condition] if suffix else [condition, values], axis=0)
        log_relative, z, raw = full_pdf(cores, tau, tf.transpose(joint))
        code = _status(code, tf.math.is_finite(z) & tf.reduce_all(tf.math.is_finite(raw)), 1)
        code = _status(code, z > normalizer_floor, 2)
        pdf = tf.exp(log_relative)*reference_density(joint, tuple(range(dimension)))
        if mode == "pdf":
            return pdf, code
        if mode != "conditional_logpdf":
            raise ValueError("unknown TTSIRT numerical operation")
        raw_prefix, _ = prefix_pdf(cores, tau, tf.transpose(condition))
        prefix = (raw_prefix/z)*reference_density(joint, condition_axes)
        result = tf.math.log(pdf)-tf.math.log(prefix)
        code = _status(code, tf.reduce_all(tf.math.is_finite(result)), 1)
        return result, code

    _PROGRAMS[key] = (transport, evaluate)
    if len(_PROGRAMS) > 32:
        _PROGRAMS.popitem(last=False)
    return evaluate


def evaluate_transport(transport, operation, condition, values, *, jit_compile=True):
    program = transport_program(transport, operation, values.shape[1], condition.shape[0], jit_compile=jit_compile)
    result, code = program(*transport_arguments(transport), condition, values)
    check_transport_status(code)
    return result


def check_transport_status(code):
    """Raise at the host boundary after the complete compiled calculation."""
    status = int(code)
    if status:
        raise ValueError(_ERRORS[status].value)

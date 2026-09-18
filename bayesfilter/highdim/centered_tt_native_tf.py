"""Native contractions for the existing centered squared-TT extension.

The component family remains an extension, not a Zhao-Cui source-faithfulness
claim. Axis products preserve the existing amplitude / mass algebra (paper
section 3.1, (13)--(14); author @TTFun/int_reference.m:22--29 and ttdot.m:20--23).
Python specializes heterogeneous tuple schemas; TensorFlow iterates axes and
batches component pairs. The public numerical boundary defaults to XLA.
"""

from collections import OrderedDict
from functools import wraps
from inspect import signature

import tensorflow as tf

from bayesfilter.ops.fixed_signature_tf import fixed_signature_function

D = tf.float64
_PROGRAMS = OrderedDict()
_METHOD_PROGRAMS = OrderedDict()


def pack_components(components):
    """Pack fixed heterogeneous core shapes; numerical values remain inputs."""
    shapes = tuple(tuple(tuple(core.shape) for core in component) for component in components)
    if not shapes or not shapes[0] or any(len(row) != len(shapes[0]) for row in shapes):
        raise ValueError("components must have the same positive axis count")
    rank = max(max(shape[0], shape[2]) for row in shapes for shape in row)
    width = max(shape[1] for row in shapes for shape in row)
    return tf.stack(tuple(tf.stack(tuple(tf.pad(tf.convert_to_tensor(core, D),
        ((0, rank-shape[0]), (0, width-shape[1]), (0, rank-shape[2])))
        for core, shape in zip(component, row, strict=True)))
        for component, row in zip(components, shapes, strict=True)))


def _homogeneous_basis_table(basis, points, width, *, paired):
    """Evaluate a shared immutable basis across all observed coordinates."""
    local = basis.bases[0]
    count, prefix, rows = len(basis.bases), points.shape[1], points.shape[0]
    if prefix:
        values = local.evaluate(tf.reshape(points, [-1]))
        values = tf.pad(values, ((0, 0), (0, width-local.basis_dim)))
        values = tf.transpose(tf.reshape(values, [rows, prefix, width]), [1, 0, 2])
        if not paired:
            return values
        observed = tf.einsum("anl,anm->anlm", values, values)
    else:
        observed = tf.zeros([0, rows, width, width], D)
    mass = local.mass_matrix(basis.convention.mass_measure)
    padding = width - local.basis_dim
    mass = tf.pad(mass, ((0, padding), (0, padding)))
    integrated = tf.broadcast_to(mass, [count-prefix, rows, width, width])
    return tf.concat([observed, integrated], axis=0)


def _basis_table(basis, points, width, *, paired):
    if all(local is basis.bases[0] for local in basis.bases):
        return _homogeneous_basis_table(basis, points, width, paired=paired)
    count, prefix = len(basis.bases), points.shape[1]
    rows = points.shape[0]

    def local_values(axis, query):
        local = basis.bases[axis]
        values = tf.pad(basis.evaluate_axis(axis, query), ((0, 0), (0, width-local.basis_dim)))
        return tf.einsum("nl,nm->nlm", values, values) if paired else values

    def branch(axis, query):
        def evaluate():
            local = basis.bases[axis]
            padding = width - local.basis_dim
            if axis < prefix:
                return local_values(axis, query[:, axis])
            mass = local.mass_matrix(basis.convention.mass_measure)
            mass = tf.pad(mass, ((0, padding), (0, padding)))
            return tf.broadcast_to(mass, [rows, width, width])
        return evaluate

    # Fixed basis types/domains are static callable branches, never Python
    # iteration over numerical rows or contraction states.
    shape = [rows, width, width] if paired else [rows, width]

    @tf.custom_gradient
    def evaluate(query):
        branches = tuple(branch(axis, query) for axis in range(count))
        result = tf.map_fn(lambda axis: tf.switch_case(axis, branches), tf.range(count),
            fn_output_signature=tf.TensorSpec(shape, D), parallel_iterations=1)

        def pullback(cotangent):
            # Recompute each heterogeneous basis's VJP inside its own branch.
            # Nested polynomial-loop TensorLists must not cross a Case boundary.
            def derivative_branch(axis):
                def derivative():
                    if axis >= prefix:
                        return tf.zeros([rows], D)
                    coordinate = query[:, axis]
                    local = basis.bases[axis]
                    if callable(getattr(local, "derivative", None)):
                        derivative = tf.pad(local.derivative(coordinate),
                            ((0, 0), (0, width-local.basis_dim)))
                        if paired:
                            value = tf.pad(basis.evaluate_axis(axis, coordinate),
                                ((0, 0), (0, width-local.basis_dim)))
                            derivative = (tf.einsum("nl,nm->nlm", derivative, value)
                                          + tf.einsum("nl,nm->nlm", value, derivative))
                            return tf.reduce_sum(cotangent[axis] * derivative, axis=[1, 2])
                        return tf.reduce_sum(cotangent[axis] * derivative, axis=1)
                    with tf.GradientTape() as tape:
                        tape.watch(coordinate)
                        value = local_values(axis, coordinate)
                    return tape.gradient(value, coordinate, output_gradients=cotangent[axis],
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return derivative

            derivatives = tuple(derivative_branch(axis) for axis in range(count))
            gradient = tf.map_fn(lambda axis: tf.switch_case(axis, derivatives), tf.range(count),
                fn_output_signature=tf.TensorSpec([rows], D), parallel_iterations=1)
            return tf.transpose(gradient[:prefix])

        return result, pullback

    return evaluate(points)


def _values(basis, packed, points):
    components, count, rank, width, _ = packed.shape
    if points.shape[1] != count:
        raise ValueError("points must include every TT axis")
    table = _basis_table(basis, points, width, paired=False)
    initial = tf.broadcast_to(tf.one_hot(0, rank, dtype=D), [components, points.shape[0], rank])

    def step(axis, state):
        matrices = tf.einsum("nl,calb->cnab", table[axis], packed[:, axis])
        return axis + 1, tf.einsum("cna,cnab->cnb", state, matrices)

    _, state = tf.while_loop(lambda axis, _: axis < count, step, (tf.constant(0), initial),
                             maximum_iterations=count, parallel_iterations=1)
    return tf.transpose(state[:, :, 0])


def _cross(basis, left, right, points):
    left_count, count, left_rank, width, _ = left.shape
    right_count, right_axes, right_rank, right_width, _ = right.shape
    if (right_axes, right_width) != (count, width) or points.shape[1] > count:
        raise ValueError("cross components must have compatible axes and bases")
    table = _basis_table(basis, points, width, paired=True)
    rank = left_rank * right_rank
    rows = points.shape[0]
    initial = tf.broadcast_to(tf.one_hot(0, rank, dtype=D), [left_count, right_count, rows, rank])

    def step(axis, state):
        paired = tf.einsum("ialb,jcmd,nlm->ijnacbd", left[:, axis], right[:, axis], table[axis])
        matrix = tf.reshape(paired, [left_count, right_count, rows, rank, rank])
        return axis + 1, tf.einsum("ijna,ijnab->ijnb", state, matrix)

    _, state = tf.while_loop(lambda axis, _: axis < count, step, (tf.constant(0), initial),
                             maximum_iterations=count, parallel_iterations=1)
    return tf.transpose(state[:, :, :, 0], [2, 0, 1])


def _program(basis, operation, specs, *, jit_compile=True):
    key = (id(basis), operation, tuple(specs), bool(jit_compile))
    if key in _PROGRAMS:
        _PROGRAMS.move_to_end(key)
        return _PROGRAMS[key][1]
    core = _values if operation == "values" else _cross

    def numerical(*inputs):
        return core(basis, *inputs)

    forward = tf.function(numerical, input_signature=specs, jit_compile=jit_compile, autograph=False)
    output = forward.get_concrete_function().output_shapes

    @tf.function(input_signature=[*specs, tf.TensorSpec(output, D)],
                 jit_compile=jit_compile, autograph=False)
    def backward(*inputs):
        values, cotangent = inputs[:-1], inputs[-1]
        with tf.GradientTape() as tape:
            tape.watch(values)
            result = numerical(*values)
        return tape.gradient(result, values, output_gradients=cotangent,
                             unconnected_gradients=tf.UnconnectedGradients.ZERO)

    @tf.custom_gradient
    def differentiable(*inputs):
        def pullback(cotangent):
            return backward(*inputs, cotangent)
        # Detach only the internal compiled invocation, whose loop-state lists
        # cannot cross XLA. The custom pullback above restores every input's
        # complete derivative by recomputing the same finite numerical program.
        return forward(*tf.nest.map_structure(tf.stop_gradient, inputs)), pullback

    differentiable.compiled = forward
    _PROGRAMS[key] = (basis, differentiable)
    if len(_PROGRAMS) > 16:
        _PROGRAMS.popitem(last=False)
    return differentiable


def _dispatch(basis, operation, *inputs):
    if tf.inside_function():
        return (_values if operation == "values" else _cross)(basis, *inputs)
    specs = tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)
    return _program(basis, operation, specs)(*inputs)


def evaluate_components(components, basis, points):
    return _dispatch(basis, "values", pack_components(components), tf.convert_to_tensor(points, D))


def cross_components(left, right, basis, points=None):
    query = tf.zeros([1, 0], D) if points is None else tf.convert_to_tensor(points, D)
    result = _dispatch(basis, "cross", pack_components(left), pack_components(right), query)
    return result[0] if points is None else result


def compiled_child_method(method):
    """Compile complete immutable-child endpoints with a full input pullback.

    Child cores/bases are frozen model configuration. Theta and query tensors
    are explicit inputs; the bounded cache retains at most 16 specializations.
    Recomputing the pullback keeps numerical loop state inside XLA, including
    when an external eager tape differentiates a returned analytical score.
    """
    api = signature(method)

    @wraps(method)
    def dispatch(child, *args, **kwargs):
        if tf.inside_function():
            return method(child, *args, **kwargs)
        bound = api.bind(child, *args, **kwargs)
        bound.apply_defaults()
        names = tuple(bound.arguments)[1:]
        inputs = tuple(tf.convert_to_tensor(bound.arguments[name], D) for name in names)
        specs = tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)
        key = (id(child), method, specs)
        if key not in _METHOD_PROGRAMS:
            input_count = len(specs)
            def numerical(*values):
                return method(child, **dict(zip(names, values, strict=True)))

            forward = tf.function(numerical, input_signature=specs, jit_compile=True, autograph=False)
            structure = forward.get_concrete_function().structured_outputs
            output_specs = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, value.dtype), structure)

            @tf.function(input_signature=[*specs, *tf.nest.flatten(output_specs)],
                         jit_compile=True, autograph=False)
            def backward(*values):
                query, cotangent = values[:input_count], values[input_count:]
                with tf.GradientTape() as tape:
                    tape.watch(query)
                    result = tf.nest.flatten(numerical(*query))
                return tape.gradient(result, query, output_gradients=cotangent,
                                     unconnected_gradients=tf.UnconnectedGradients.ZERO)

            @tf.custom_gradient
            def differentiable(*values):
                def pullback(*cotangent):
                    return backward(*values, *cotangent)
                result = forward(*tf.nest.map_structure(tf.stop_gradient, values))
                return tuple(tf.nest.flatten(result)), pullback

            differentiable.compiled = forward
            _METHOD_PROGRAMS[key] = (child, differentiable, structure)
            if len(_METHOD_PROGRAMS) > 16:
                _METHOD_PROGRAMS.popitem(last=False)
        _METHOD_PROGRAMS.move_to_end(key)
        _, evaluate, structure = _METHOD_PROGRAMS[key]
        result = tf.nest.pack_sequence_as(structure, evaluate(*inputs))
        # XLA may discard in-graph Assert ops. Preserve a host-side finite
        # output veto at this public boundary after the compiled calculation.
        tf.nest.map_structure(lambda value: tf.debugging.assert_all_finite(
            value, "nonfinite centered child endpoint"), result)
        return result

    return dispatch


@fixed_signature_function(floating_dtype=D, tensor_dtypes={"families": tf.int32,
    "left_indices": tf.int32, "right_indices": tf.int32})
def feature_values_and_jacobian(theta, families, left_indices, right_indices):
    """Linear, quadratic and interaction features batched over theta rows."""
    left = tf.gather(theta, left_indices, axis=1)
    right = tf.gather(theta, right_indices, axis=1)
    linear = families[None, :] == 0
    values = tf.where(linear, left, left * right)
    left_eye = tf.one_hot(left_indices, theta.shape[1], dtype=theta.dtype)
    right_eye = tf.one_hot(right_indices, theta.shape[1], dtype=theta.dtype)
    tangent = tf.where(linear[:, :, None], left_eye[None, :, :],
        right[:, :, None] * left_eye[None, :, :] + left[:, :, None] * right_eye[None, :, :])
    return values, tangent

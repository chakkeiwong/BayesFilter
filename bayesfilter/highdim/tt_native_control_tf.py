"""Native tensor recurrences over heterogeneous, statically declared TT cores.

Python handles only the fixed tuple schema and branch construction. Numerical
axis contractions run in TensorFlow loops. Padding stores heterogeneous cores;
all additional entries are zero and are removed at the tuple boundary.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache

import tensorflow as tf

_PRIMITIVE_CACHE = OrderedDict()


def functional_tt_primitive(product_basis, cores, *, points=None, integrate_axes=None,
                            measure=None, jit_compile=True):
    """Stable enclosing primitive for a fixed heterogeneous TT schema.

    Tensor values are inputs; only basis types, shapes and integrated axes are
    specialized. Python tuple packing handles the heterogeneous return schema.
    """
    from bayesfilter.highdim.tt import TTCore

    shapes = tuple(tuple(core.values.shape) for core in cores)
    axes = None if integrate_axes is None else tuple(integrate_axes)
    key = (id(product_basis), shapes, None if points is None else tuple(points.shape), axes, measure, bool(jit_compile))
    if key in _PRIMITIVE_CACHE:
        _PRIMITIVE_CACHE.move_to_end(key)
        program = _PRIMITIVE_CACHE[key][1]
    else:
        core_specs = tuple(tf.TensorSpec(shape, tf.float64) for shape in shapes)
        if axes is None:
            @tf.function(input_signature=[tf.TensorSpec(points.shape, tf.float64), core_specs],
                         jit_compile=jit_compile, autograph=False)
            def program(query, values):
                typed = tuple(TTCore(value) for value in values)
                return prefix_rows(typed, product_basis, query)[0][:, 0]
        else:
            @tf.function(input_signature=[core_specs], jit_compile=jit_compile, autograph=False)
            def program(values):
                typed = tuple(TTCore(value) for value in values)
                return contract_core_axes(typed, product_basis, axes, measure)
        _PRIMITIVE_CACHE[key] = (product_basis, program)
        if len(_PRIMITIVE_CACHE) > 16:
            _PRIMITIVE_CACHE.popitem(last=False)
    values = tuple(core.values for core in cores)
    return program(points, values) if axes is None else program(values)


def functional_tt_jvp(product_basis, cores, dot_cores, *, points=None, core_index=None,
                      jit_compile=True):
    """Explicit product-rule JVP with a bounded tensor-signature cache."""
    from bayesfilter.highdim.tt import TTCore

    shapes = tuple(tuple(core.values.shape) for core in cores)
    if shapes != tuple(tuple(core.values.shape) for core in dot_cores):
        raise ValueError("TT core tangent shapes must match primal cores")
    key = ("jvp", id(product_basis), shapes, None if points is None else tuple(points.shape),
        core_index, bool(jit_compile))
    if key in _PRIMITIVE_CACHE:
        _PRIMITIVE_CACHE.move_to_end(key)
        program = _PRIMITIVE_CACHE[key][1]
    else:
        core_specs = tuple(tf.TensorSpec(shape, tf.float64) for shape in shapes)
        signature = [core_specs, core_specs]
        if points is not None:
            signature.append(tf.TensorSpec(points.shape, tf.float64))

        @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
        def program(values, directions, *queries):
            typed = tuple(TTCore(value) for value in values)
            dotted = tuple(TTCore(value) for value in directions)
            if not queries:
                masses = basis_masses(product_basis, typed)
                return tf.reshape(gram_chain(typed, masses, dot_cores=dotted)[1], [])
            if core_index is None:
                return prefix_rows(typed, product_basis, queries[0], dot_cores=dotted)[1][:, 0]
            matrices = core_matrices(product_basis, queries[0], typed)
            dot_matrices = core_matrices(product_basis, queries[0], dotted)
            left, dot_left = row_environment_jvp(matrices, dot_matrices)
            right, dot_right = row_environment_jvp(matrices, dot_matrices, reverse=True)
            phi = product_basis.evaluate_axis(core_index, queries[0][:, core_index])
            blocks = tf.einsum("na,nl,nb->nalb", dot_left[core_index], phi, right[core_index])
            blocks += tf.einsum("na,nl,nb->nalb", left[core_index], phi, dot_right[core_index])
            return tf.reshape(blocks, [queries[0].shape[0], math.prod(shapes[core_index])])

        _PRIMITIVE_CACHE[key] = (product_basis, program)
        if len(_PRIMITIVE_CACHE) > 16:
            _PRIMITIVE_CACHE.popitem(last=False)
    arguments = (tuple(core.values for core in cores), tuple(core.values for core in dot_cores))
    return program(*arguments, points) if points is not None else program(*arguments)


def row_environment_jvp(matrices, dot_matrices, *, reverse=False):
    """Native left/right product-rule recurrence preserving update order."""
    count = len(matrices)
    rank = max(max(matrix.shape[1:]) for matrix in matrices)

    def pack(values):
        return tf.stack(tuple(tf.pad(value, (
            (0, 0), (0, rank-value.shape[1]), (0, rank-value.shape[2]),
        )) for value in values))

    packed, dots = pack(matrices), pack(dot_matrices)
    start = tf.one_hot(tf.zeros([tf.shape(packed)[1]], tf.int32), rank, dtype=packed.dtype)
    history = tf.TensorArray(packed.dtype, count, element_shape=start.shape)
    dot_history = tf.TensorArray(packed.dtype, count, element_shape=start.shape)

    def step(index, state, tangent, history, dot_history):
        axis = count-1-index if reverse else index
        history, dot_history = history.write(axis, state), dot_history.write(axis, tangent)
        if reverse:
            tangent = tf.einsum("nab,nb->na", dots[axis], state) + tf.einsum("nab,nb->na", packed[axis], tangent)
            state = tf.einsum("nab,nb->na", packed[axis], state)
        else:
            tangent = tf.einsum("na,nab->nb", tangent, packed[axis]) + tf.einsum("na,nab->nb", state, dots[axis])
            state = tf.einsum("na,nab->nb", state, packed[axis])
        return index+1, state, tangent, history, dot_history

    _, _, _, history, dot_history = tf.while_loop(
        lambda i, *_: i < count, step,
        (tf.constant(0), start, tf.zeros_like(start), history, dot_history),
        maximum_iterations=count, parallel_iterations=1,
    )
    history, dot_history = history.stack(), dot_history.stack()
    return (
        tuple(history[axis, :, :matrix.shape[2 if reverse else 1]] for axis, matrix in enumerate(matrices)),
        tuple(dot_history[axis, :, :matrix.shape[2 if reverse else 1]] for axis, matrix in enumerate(matrices)),
    )


def contract_core_axes(cores, product_basis, axes, measure):
    """Integrate a declared subset in original axis order using padded state."""
    count = len(cores)
    rank = max(max(core.left_rank, core.right_rank) for core in cores)
    width = max(core.basis_dim for core in cores)
    packed = tf.stack(tuple(tf.pad(core.values, (
        (0, rank-core.left_rank), (0, width-core.basis_dim), (0, rank-core.right_rank),
    )) for core in cores))
    masks = tf.constant(tuple(axis in axes for axis in range(count)))
    right_ranks = tf.constant(tuple(core.right_rank for core in cores))
    kept = tuple(axis for axis in range(count) if axis not in axes)

    def integral_branch(axis):
        def evaluate():
            value = product_basis.bases[axis].integral_vector(measure)
            return tf.pad(value, [[0, width-cores[axis].basis_dim]])
        return evaluate

    branches = tuple(integral_branch(axis) for axis in range(count))
    integrals = tf.map_fn(
        lambda axis: tf.switch_case(axis, branches), tf.range(count),
        fn_output_signature=tf.TensorSpec([width], packed.dtype), parallel_iterations=1,
    )
    start = tf.linalg.diag(tf.one_hot(0, rank, dtype=packed.dtype))

    def step(axis, pending, outputs):
        def integrated():
            matrix = tf.einsum("l,alb->ab", integrals[axis], packed[axis])
            return tf.linalg.matmul(pending, matrix), outputs

        def retained():
            value = tf.einsum("ea,alb->elb", pending, packed[axis])
            identity = tf.linalg.diag(tf.cast(tf.range(rank) < right_ranks[axis], packed.dtype))
            return identity, tf.tensor_scatter_nd_update(outputs, [[axis]], value[None])

        pending, outputs = tf.cond(masks[axis], integrated, retained)
        return axis+1, pending, outputs

    _, final, outputs = tf.while_loop(
        lambda axis, *_: axis < count, step,
        (tf.constant(0), start, tf.zeros_like(packed)),
        maximum_iterations=count, parallel_iterations=1,
    )
    if not kept:
        return (), final[0, 0]
    last = tf.einsum("elb,bc->elc", outputs[kept[-1]], final)
    outputs = tf.tensor_scatter_nd_update(outputs, [[kept[-1]]], last[None])
    result = tuple(outputs[axis,
        :1 if index == 0 else cores[kept[index-1]].right_rank,
        :cores[axis].basis_dim,
        :1 if index == len(kept)-1 else cores[axis].right_rank,
    ] for index, axis in enumerate(kept))
    return result, tf.constant(0.0, packed.dtype)


def _fixed_basis_rows(product_basis, points, shapes):
    width = max(shape[1] for shape in shapes)

    def branch(axis):
        return lambda: tf.pad(
            product_basis.evaluate_axis(axis, points[:, axis]),
            ((0, 0), (0, width - shapes[axis][1])),
        )

    branches = tuple(branch(axis) for axis in range(len(shapes)))
    return tf.map_fn(
        lambda axis: tf.switch_case(axis, branches),
        tf.range(len(shapes)),
        fn_output_signature=tf.TensorSpec([points.shape[0], width], points.dtype),
        parallel_iterations=1,
    )


def fixed_basis_program(product_basis, point_shape, shapes, *, jit_compile=True):
    """Bounded preparation cache; numerical query values remain inputs."""
    key = ("basis", id(product_basis), tuple(point_shape), tuple(shapes), bool(jit_compile))
    if key in _PRIMITIVE_CACHE:
        _PRIMITIVE_CACHE.move_to_end(key)
        return _PRIMITIVE_CACHE[key][1]
    program = tf.function(lambda query: _fixed_basis_rows(product_basis, query, shapes),
        input_signature=[tf.TensorSpec(point_shape, tf.float64)],
        jit_compile=jit_compile, autograph=False)
    _PRIMITIVE_CACHE[key] = (product_basis, program)
    if len(_PRIMITIVE_CACHE) > 16:
        _PRIMITIVE_CACHE.popitem(last=False)
    return program


def fixed_basis_rows(product_basis, points, cores, *, jit_compile=True):
    """Compile standalone basis preparation; inline into enclosing graphs."""
    shapes = tuple(tuple(core.values.shape) for core in cores)
    if tf.inside_function():
        return _fixed_basis_rows(product_basis, points, shapes)
    return fixed_basis_program(product_basis, points.shape, shapes,
        jit_compile=jit_compile)(points)


def fixed_core_matrices(basis_rows, cores):
    """Batched contraction of fixed basis rows with heterogeneous padded cores."""
    packed, _ = pack_tensors(tuple(core.values for core in cores))
    matrices = tf.einsum("cnl,calb->cnab", basis_rows, packed)
    return tuple(
        matrices[axis, :, : core.left_rank, : core.right_rank]
        for axis, core in enumerate(cores)
    )


def make_fixed_squared_marginal(product_basis, initial_cores, keep_axes, points, *, jit_compile=True):
    """Prepare fixed basis moments; the returned core recurrence is tensor-only.

    This boundary is for parameter-independent quadrature/fit points. Moving
    their basis evaluation outside the score tape avoids differentiating
    unrelated basis-recursion state in TensorFlow conditional gradients.
    """
    count = len(initial_cores)
    rank = max(max(core.left_rank, core.right_rank) for core in initial_cores)
    width = max(core.basis_dim for core in initial_cores)
    columns = {axis: column for column, axis in enumerate(keep_axes)}

    @tf.function(input_signature=[tf.TensorSpec(points.shape, points.dtype)],
                 jit_compile=jit_compile, autograph=False)
    def prepare(query):
        def branch(axis):
            def evaluate():
                core = initial_cores[axis]
                if axis in columns:
                    phi = product_basis.evaluate_axis(axis, query[:, columns[axis]])
                    mass = tf.zeros([core.basis_dim, core.basis_dim], query.dtype)
                else:
                    mass = product_basis.bases[axis].mass_matrix(
                        product_basis.convention.mass_measure
                    )
                    phi = tf.zeros([query.shape[0], core.basis_dim], query.dtype)
                return (
                    tf.pad(phi, ((0, 0), (0, width - core.basis_dim))),
                    tf.pad(
                        mass, ((0, width - core.basis_dim), (0, width - core.basis_dim))
                    ),
                )

            return evaluate

        branches = tuple(branch(axis) for axis in range(count))
        return tf.map_fn(
            lambda axis: tf.switch_case(axis, branches),
            tf.range(count),
            fn_output_signature=(
                tf.TensorSpec([query.shape[0], width], query.dtype),
                tf.TensorSpec([width, width], query.dtype),
            ),
            parallel_iterations=1,
        )
    basis_rows, masses = prepare.python_function(points) if tf.inside_function() else prepare(points)
    retained = tf.constant(tuple(axis in columns for axis in range(count)))

    def contract(cores):
        packed = tf.stack(
            tuple(
                tf.pad(
                    core.values,
                    (
                        (0, rank - core.left_rank),
                        (0, width - core.basis_dim),
                        (0, rank - core.right_rank),
                    ),
                )
                for core in cores
            )
        )
        unit = tf.one_hot(
            tf.zeros([points.shape[0]], tf.int32), rank, dtype=points.dtype
        )
        state = unit[:, :, None] * unit[:, None, :]

        def step(axis, state):
            value = packed[axis]

            # Preserve the original contraction order at retained coordinates:
            # evaluate each square-root core before taking its outer product.
            # Forming phi*phi first can catastrophically cancel near zero.
            def retained_matrix():
                matrix = tf.einsum("nl,alb->nab", basis_rows[axis], value)
                return tf.einsum("nab,nAB->naAbB", matrix, matrix)

            def integrated_matrix():
                matrix = tf.einsum("alb,AmB,lm->aAbB", value, value, masses[axis])
                return tf.broadcast_to(
                    matrix[None], [points.shape[0], rank, rank, rank, rank]
                )

            paired = tf.cond(retained[axis], retained_matrix, integrated_matrix)
            return axis + 1, tf.einsum("naA,naAbB->nbB", state, paired)

        _, state = tf.while_loop(
            lambda axis, _: axis < count,
            step,
            (tf.constant(0), state),
            maximum_iterations=count,
            parallel_iterations=1,
        )
        return state[:, 0, 0]

    contract.preparation_program = prepare
    return contract


def _basis_rows_with_pullback(product_basis, keep_axes, points, width):
    """Evaluate heterogeneous bases before differentiating core contractions.

    The full query pullback recomputes each local derivative in its own Case
    branch. Polynomial-loop TensorLists therefore never cross a Case boundary.
    Core coefficients remain ordinary inputs to the subsequent contractions.
    """
    count, rows = len(product_basis.bases), points.shape[0]
    columns = {axis: column for column, axis in enumerate(keep_axes)}

    def branch(axis, query):
        def evaluate():
            if axis not in columns:
                return tf.zeros([rows, width], points.dtype)
            local = product_basis.bases[axis]
            value = product_basis.evaluate_axis(axis, query[:, columns[axis]])
            return tf.pad(value, ((0, 0), (0, width - local.basis_dim)))
        return evaluate

    @tf.custom_gradient
    def evaluate(query):
        branches = tuple(branch(axis, query) for axis in range(count))
        result = tf.map_fn(lambda axis: tf.switch_case(axis, branches), tf.range(count),
            fn_output_signature=tf.TensorSpec([rows, width], points.dtype),
            parallel_iterations=1)

        def pullback(cotangent):
            def derivative_branch(axis):
                def derivative():
                    if axis not in columns:
                        return tf.zeros([rows], points.dtype)
                    local = product_basis.bases[axis]
                    coordinate = query[:, columns[axis]]
                    local_cotangent = cotangent[axis, :, :local.basis_dim]
                    if callable(getattr(local, "derivative", None)):
                        return tf.reduce_sum(local_cotangent * local.derivative(coordinate), axis=1)
                    with tf.GradientTape() as tape:
                        tape.watch(coordinate)
                        value = product_basis.evaluate_axis(axis, coordinate)
                    return tape.gradient(value, coordinate, output_gradients=local_cotangent,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return derivative

            derivatives = tuple(derivative_branch(axis) for axis in range(count))
            gradient = tf.map_fn(lambda axis: tf.switch_case(axis, derivatives), tf.range(count),
                fn_output_signature=tf.TensorSpec([rows], points.dtype), parallel_iterations=1)
            return tf.transpose(tf.gather(gradient, tf.constant(keep_axes, tf.int32)))

        return result, pullback

    return evaluate(points)


def squared_marginal(cores, product_basis, keep_axes, points):
    """Contract the paired cores in the original axis order with native control."""
    count = len(cores)
    rank = max(max(core.left_rank, core.right_rank) for core in cores) ** 2
    width = max(core.basis_dim for core in cores)
    columns = {axis: column for column, axis in enumerate(keep_axes)}
    basis_rows = _basis_rows_with_pullback(product_basis, keep_axes, points, width)
    masses = basis_masses(product_basis, cores)

    def branch(axis):
        def matrix():
            core = cores[axis]
            if axis in columns:
                phi = basis_rows[axis, :, :core.basis_dim]
                paired = tf.einsum(
                    "nl,nm,alb,AmB->naAbB", phi, phi, core.values, core.values
                )
            else:
                gram = masses[axis]
                paired = tf.einsum("alb,AmB,lm->aAbB", core.values, core.values, gram)
                paired = tf.broadcast_to(paired[None], [points.shape[0], *paired.shape])
            values = tf.reshape(
                paired, [points.shape[0], core.left_rank**2, core.right_rank**2]
            )
            return tf.pad(
                values,
                [[0, 0], [0, rank - core.left_rank**2], [0, rank - core.right_rank**2]],
            )

        return matrix

    branches = tuple(branch(axis) for axis in range(count))
    state = tf.one_hot(tf.zeros([points.shape[0]], tf.int32), rank, dtype=points.dtype)

    def step(axis, state):
        return axis + 1, tf.einsum("na,nab->nb", state, tf.switch_case(axis, branches))

    _, state = tf.while_loop(
        lambda axis, _: axis < count,
        step,
        (tf.constant(0), state),
        maximum_iterations=count,
        parallel_iterations=1,
    )
    return state[:, 0]


@lru_cache(maxsize=16)
def random_core_start_program(shapes, date_count, *, jit_compile=True):
    """Compile both date/core recurrences with the existing Philox draws."""
    from bayesfilter.ops.stateless_random_tf import philox_normal_float64

    maximum = max(math.prod(shape) for shape in shapes)

    @tf.function(input_signature=[tf.TensorSpec([date_count], tf.int32),
                                  tf.TensorSpec([], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def generate(dates, seed):
        def at_date(date):
            def branch(axis):
                def draw():
                    value = 0.3 * philox_normal_float64(
                        shapes[axis], tf.stack((seed, 7000 + 31 * date + axis)))
                    return tf.pad(tf.reshape(value, [-1]),
                                  [[0, maximum - math.prod(shapes[axis])]])

                return draw

            branches = tuple(branch(axis) for axis in range(len(shapes)))
            return tf.map_fn(
                lambda axis: tf.switch_case(axis, branches),
                tf.range(len(shapes)),
                fn_output_signature=tf.TensorSpec([maximum], tf.float64),
                parallel_iterations=1,
            )
        return tf.map_fn(
            at_date, dates,
            fn_output_signature=tf.TensorSpec([len(shapes), maximum], tf.float64),
            parallel_iterations=1,
        )
    return generate


def random_core_starts(shapes, dates, seed, *, jit_compile=True):
    """Prepare fixed TT starts under XLA without changing the seeded stream."""
    packed = random_core_start_program(tuple(shapes), dates.shape[0],
        jit_compile=jit_compile)(dates, tf.convert_to_tensor(seed, tf.int32))
    return tuple(
        tf.reshape(packed[:, axis, : math.prod(shape)], [dates.shape[0], *shape])
        for axis, shape in enumerate(shapes)
    )


def pack_tensors(values):
    """Pack a fixed heterogeneous tuple, without numerical iteration."""
    shapes = tuple(tuple(value.shape) for value in values)
    extents = tuple(max(shape[i] for shape in shapes) for i in range(len(shapes[0])))
    packed = tf.stack(
        tuple(
            tf.pad(value, tuple((0, top - dim) for top, dim in zip(extents, shape)))
            for value, shape in zip(values, shapes)
        )
    )
    return packed, shapes


def add_core_tensors(left, right):
    """One tensor addition for fixed heterogeneous core tuples."""
    packed_left, shapes = pack_tensors(left)
    packed_right, right_shapes = pack_tensors(right)
    if shapes != right_shapes:
        raise ValueError("core cotangent shapes differ")
    total = packed_left + packed_right
    return tuple(total[index, :shape[0], :shape[1], :shape[2]]
                 for index, shape in enumerate(shapes))


def core_matrices(product_basis, points, cores):
    rank = max(max(core.left_rank, core.right_rank) for core in cores)
    width = max(core.basis_dim for core in cores)
    basis_rows = _basis_rows_with_pullback(
        product_basis, tuple(range(len(cores))), points, width)

    # Each branch is a static heterogeneous basis/core type specialization.
    # tf.map_fn drives the numerical axis iteration with one traced body.
    def branch(axis):
        def evaluate():
            values = basis_rows[axis, :, :cores[axis].basis_dim]
            matrix = tf.einsum("nl,alb->nab", values, cores[axis].values)
            return tf.pad(
                matrix,
                (
                    (0, 0),
                    (0, rank - cores[axis].left_rank),
                    (0, rank - cores[axis].right_rank),
                ),
            )

        return evaluate

    branches = tuple(branch(axis) for axis in range(len(cores)))
    packed = tf.map_fn(
        lambda axis: tf.switch_case(axis, branches),
        tf.range(len(cores)),
        fn_output_signature=tf.TensorSpec([points.shape[0], rank, rank], points.dtype),
        parallel_iterations=1,
    )
    return tuple(
        packed[axis, :, : core.left_rank, : core.right_rank]
        for axis, core in enumerate(cores)
    )


def row_environments(matrices, *, reverse=False):
    count = len(matrices)
    rank = max(max(matrix.shape[1:]) for matrix in matrices)
    packed = tf.stack(
        tuple(
            tf.pad(
                matrix,
                ((0, 0), (0, rank - matrix.shape[1]), (0, rank - matrix.shape[2])),
            )
            for matrix in matrices
        )
    )
    state = tf.one_hot(
        tf.zeros([tf.shape(packed)[1]], tf.int32), rank, dtype=packed.dtype
    )
    history = tf.TensorArray(packed.dtype, count, element_shape=state.shape)

    def body(index, state, history):
        axis = count - 1 - index if reverse else index
        history = history.write(axis, state)
        state = (
            tf.einsum("nab,nb->na", packed[axis], state)
            if reverse
            else tf.einsum("na,nab->nb", state, packed[axis])
        )
        return index + 1, state, history

    _, _, history = tf.while_loop(
        lambda i, *_: i < count,
        body,
        (tf.constant(0), state, history),
        maximum_iterations=count,
        parallel_iterations=1,
    )
    history = history.stack()
    return tuple(
        history[axis, :, : matrix.shape[2 if reverse else 1]]
        for axis, matrix in enumerate(matrices)
    )


def basis_masses(product_basis, cores, *, axis_offset=0, measure=None):
    """Native evaluation of heterogeneous basis mass matrices."""
    active_measure = product_basis.convention.mass_measure if measure is None else measure
    width = max(core.basis_dim for core in cores)

    def branch(axis):
        def evaluate():
            mass = product_basis.bases[axis_offset + axis].mass_matrix(active_measure)
            return tf.pad(mass, [[0, width-cores[axis].basis_dim], [0, width-cores[axis].basis_dim]])
        return evaluate

    branches = tuple(branch(axis) for axis in range(len(cores)))
    packed = tf.map_fn(lambda axis: tf.switch_case(axis, branches), tf.range(len(cores)),
        fn_output_signature=tf.TensorSpec([width, width], tf.float64), parallel_iterations=1)
    return tuple(packed[axis, :core.basis_dim, :core.basis_dim] for axis, core in enumerate(cores))


def gram_chain(cores, masses, *, reverse=False, dot_cores=None):
    core_values = tuple(core.values for core in cores)
    count = len(cores)
    rank = max(max(core.left_rank, core.right_rank) for core in cores)
    basis_dim = max(core.basis_dim for core in cores)

    def pack(values):
        return tf.stack(
            tuple(
                tf.pad(
                    value,
                    (
                        (0, rank - value.shape[0]),
                        (0, basis_dim - value.shape[1]),
                        (0, rank - value.shape[2]),
                    ),
                )
                for value in values
            )
        )

    packed = pack(core_values)
    mass, _ = pack_tensors(masses)
    dot = (
        tf.zeros_like(packed)
        if dot_cores is None
        else pack(tuple(core.values for core in dot_cores))
    )
    unit = tf.one_hot(0, rank, dtype=packed.dtype)
    state = unit[:, None] * unit[None, :]
    dot_state = tf.zeros_like(state)
    equation = "akb,AlB,kl,bB->aA" if reverse else "akb,AlB,kl,aA->bB"

    def body(index, state, dot_state):
        axis = count - 1 - index if reverse else index
        core, dcore, gram = packed[axis], dot[axis], mass[axis]
        new_state = tf.einsum(equation, core, core, gram, state)
        if dot_cores is not None:
            dot_state = (
                tf.einsum(equation, dcore, core, gram, state)
                + tf.einsum(equation, core, dcore, gram, state)
                + tf.einsum(equation, core, core, gram, dot_state)
            )
        return index + 1, new_state, dot_state

    _, state, dot_state = tf.while_loop(
        lambda i, *_: i < count,
        body,
        (tf.constant(0), state, dot_state),
        maximum_iterations=count,
        parallel_iterations=1,
    )
    final_rank = cores[0].left_rank if reverse else cores[-1].right_rank
    return state[:final_rank, :final_rank], dot_state[:final_rank, :final_rank]


def prefix_rows(cores, product_basis, points, dot_cores=None):
    matrices = core_matrices(product_basis, points, cores)
    rank = max(max(matrix.shape[1:]) for matrix in matrices)

    def pack(values):
        return tf.stack(
            tuple(
                tf.pad(
                    value,
                    ((0, 0), (0, rank - value.shape[1]), (0, rank - value.shape[2])),
                )
                for value in values
            )
        )

    packed = pack(matrices)
    dot = (
        tf.zeros_like(packed)
        if dot_cores is None
        else pack(core_matrices(product_basis, points, dot_cores))
    )
    state = tf.one_hot(
        tf.zeros([tf.shape(points)[0]], tf.int32), rank, dtype=points.dtype
    )
    dot_state = tf.zeros_like(state)

    def body(axis, state, dot_state):
        new_state = tf.einsum("na,nab->nb", state, packed[axis])
        if dot_cores is not None:
            dot_state = tf.einsum("na,nab->nb", dot_state, packed[axis]) + tf.einsum(
                "na,nab->nb", state, dot[axis]
            )
        return axis + 1, new_state, dot_state

    _, state, dot_state = tf.while_loop(
        lambda i, *_: i < len(cores),
        body,
        (tf.constant(0), state, dot_state),
        maximum_iterations=len(cores),
        parallel_iterations=1,
    )
    return state[:, : cores[-1].right_rank], dot_state[:, : cores[-1].right_rank]


def gram_adjoint(cores, masses, bar_gram, *, reverse=False):
    """Manual transpose of the Gram recurrence; no autodiff or axis unrolling."""
    count = len(cores)
    rank = max(max(core.left_rank, core.right_rank) for core in cores)
    basis_dim = max(core.basis_dim for core in cores)
    packed = tf.stack(
        tuple(
            tf.pad(
                core.values,
                (
                    (0, rank - core.left_rank),
                    (0, basis_dim - core.basis_dim),
                    (0, rank - core.right_rank),
                ),
            )
            for core in cores
        )
    )
    mass, _ = pack_tensors(masses)
    unit = tf.one_hot(0, rank, dtype=packed.dtype)
    state = unit[:, None] * unit[None, :]
    history = tf.TensorArray(packed.dtype, size=count, element_shape=[rank, rank])
    equation = "akb,AlB,kl,bB->aA" if reverse else "akb,AlB,kl,aA->bB"

    def forward(i, state, history):
        axis = count - 1 - i if reverse else i
        history = history.write(axis, state)
        state = tf.einsum(equation, packed[axis], packed[axis], mass[axis], state)
        return i + 1, state, history

    _, _, history = tf.while_loop(
        lambda i, *_: i < count,
        forward,
        (tf.constant(0), state, history),
        maximum_iterations=count,
        parallel_iterations=1,
    )
    bar = tf.pad(
        bar_gram, ((0, rank - bar_gram.shape[0]), (0, rank - bar_gram.shape[1]))
    )
    gradients = tf.TensorArray(
        packed.dtype, size=count, element_shape=[rank, basis_dim, rank]
    )

    def backward(i, bar, gradients):
        axis = i if reverse else count - 1 - i
        core, gram, value = packed[axis], mass[axis], history.read(axis)
        if reverse:
            gradient = tf.einsum(
                "aA,AlB,kl,bB->akb", bar, core, gram, value
            ) + tf.einsum("aA,akb,kl,bB->AlB", bar, core, gram, value)
            bar = tf.einsum("aA,akb,AlB,kl->bB", bar, core, core, gram)
        else:
            gradient = tf.einsum(
                "bB,AlB,kl,aA->akb", bar, core, gram, value
            ) + tf.einsum("bB,akb,kl,aA->AlB", bar, core, gram, value)
            bar = tf.einsum("bB,akb,AlB,kl->aA", bar, core, core, gram)
        return i + 1, bar, gradients.write(axis, gradient)

    _, _, gradients = tf.while_loop(
        lambda i, *_: i < count,
        backward,
        (tf.constant(0), bar, gradients),
        maximum_iterations=count,
        parallel_iterations=1,
    )
    result = gradients.stack()
    return tuple(
        result[axis, : core.left_rank, : core.basis_dim, : core.right_rank]
        for axis, core in enumerate(cores)
    )


def row_chain_adjoint(
    cores, basis, points, *, bar_rows=None, core_index=None, bar_design=None
):
    """Manual transpose of prefix rows or an ALS design's two environments."""
    count = len(cores)
    matrices = core_matrices(basis, points, cores)
    rank = max(max(matrix.shape[1:]) for matrix in matrices)
    degree = max(core.basis_dim for core in cores)

    def pack_rows(values):
        return tf.stack(
            tuple(
                tf.pad(value, ((0, 0), (0, rank - value.shape[1]))) for value in values
            )
        )

    left = pack_rows(row_environments(matrices))
    right = pack_rows(row_environments(matrices, reverse=True))
    packed = tf.stack(
        tuple(
            tf.pad(
                matrix,
                ((0, 0), (0, rank - matrix.shape[1]), (0, rank - matrix.shape[2])),
            )
            for matrix in matrices
        )
    )

    def basis_branch(axis):
        def evaluate():
            value = basis.evaluate_axis(axis, points[:, axis])
            return tf.pad(value, ((0, 0), (0, degree - value.shape[1])))

        return evaluate

    branches = tuple(basis_branch(axis) for axis in range(count))
    phi = tf.map_fn(
        lambda axis: tf.switch_case(axis, branches),
        tf.range(count),
        fn_output_signature=tf.TensorSpec([points.shape[0], degree], points.dtype),
        parallel_iterations=1,
    )
    gradients = tf.zeros([count, rank, degree, rank], points.dtype)
    if core_index is None:
        stop_left = count
        bar_left = tf.pad(bar_rows, ((0, 0), (0, rank - bar_rows.shape[1])))
        bar_right = tf.zeros_like(bar_left)
        start_right = count
    else:
        core = cores[core_index]
        blocks = tf.reshape(
            bar_design,
            [points.shape[0], core.left_rank, core.basis_dim, core.right_rank],
        )
        blocks = tf.pad(
            blocks,
            (
                (0, 0),
                (0, rank - core.left_rank),
                (0, degree - core.basis_dim),
                (0, rank - core.right_rank),
            ),
        )
        bar_left = tf.einsum(
            "nalb,nl,nb->na", blocks, phi[core_index], right[core_index]
        )
        bar_right = tf.einsum(
            "nalb,na,nl->nb", blocks, left[core_index], phi[core_index]
        )
        stop_left, start_right = core_index, core_index + 1

    def reverse_left(index, bar, gradients):
        axis = stop_left - 1 - index
        gradient = tf.einsum("na,nk,nb->akb", left[axis], phi[axis], bar)
        bar = tf.einsum("nb,nab->na", bar, packed[axis])
        return (
            index + 1,
            bar,
            tf.tensor_scatter_nd_update(gradients, [[axis]], [gradient]),
        )

    _, _, gradients = tf.while_loop(
        lambda i, *_: i < stop_left,
        reverse_left,
        (tf.constant(0), bar_left, gradients),
        maximum_iterations=stop_left,
        parallel_iterations=1,
    )

    def reverse_right(axis, bar, gradients):
        gradient = tf.einsum("na,nk,nb->akb", bar, phi[axis], right[axis])
        bar = tf.einsum("na,nab->nb", bar, packed[axis])
        return (
            axis + 1,
            bar,
            tf.tensor_scatter_nd_update(gradients, [[axis]], [gradient]),
        )

    _, _, gradients = tf.while_loop(
        lambda i, *_: i < count,
        reverse_right,
        (tf.constant(start_right), bar_right, gradients),
        maximum_iterations=count - start_right,
        parallel_iterations=1,
    )
    return tuple(
        gradients[axis, : core.left_rank, : core.basis_dim, : core.right_rank]
        for axis, core in enumerate(cores)
    )

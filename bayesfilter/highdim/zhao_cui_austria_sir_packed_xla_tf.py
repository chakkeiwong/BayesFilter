"""Packed TensorFlow/XLA kernels for the Austria SIR Lane-B TT replay."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf


DTYPE = tf.float64
AXIS_COUNT = 36
MAX_RANK = 4
BASIS_DIM = 5
PACKED_SHAPE = (AXIS_COUNT, MAX_RANK, BASIS_DIM, MAX_RANK)
MATERIAL_REPLAY_ATOL = 5e-12
MATERIAL_REPLAY_RTOL = 5e-6
MATERIAL_REPLAY_POLICY_ID = "elementwise_atol5e12_rtol5e6_v1"
PACKED_XLA_POLICY_ID = "packed_rank4_tf_while_loop_xla_v1"


def pack_cores(cores: Sequence[tf.Tensor]) -> tf.Tensor:
    """Pack setup-static heterogeneous TT cores into one masked rank-4 tensor."""

    if len(cores) != AXIS_COUNT:
        raise ValueError("packed Lane-B TT requires 36 cores")
    rows = []
    for axis, core in enumerate(cores):
        value = tf.convert_to_tensor(core, DTYPE)
        if value.shape.rank != 3 or value.shape[1] != BASIS_DIM:
            raise ValueError(f"invalid Lane-B core shape at axis {axis}")
        left = int(value.shape[0])
        right = int(value.shape[2])
        if left > MAX_RANK or right > MAX_RANK:
            raise ValueError(f"Lane-B core rank exceeds {MAX_RANK} at axis {axis}")
        rows.append(
            tf.pad(value, [[0, MAX_RANK - left], [0, 0], [0, MAX_RANK - right]])
        )
    return tf.ensure_shape(tf.stack(rows), PACKED_SHAPE)


def unpack_cores(packed: tf.Tensor, shapes: Sequence[tf.TensorShape]) -> tuple[tf.Tensor, ...]:
    """Unpack a numerical result at the artifact boundary."""

    value = tf.ensure_shape(tf.convert_to_tensor(packed, DTYPE), PACKED_SHAPE)
    if len(shapes) != AXIS_COUNT:
        raise ValueError("unpack requires 36 target shapes")
    return tuple(
        tf.ensure_shape(
            value[axis, : int(shape[0]), :, : int(shape[2])], shape
        )
        for axis, shape in enumerate(shapes)
    )


def pack_tangent_banks(
    banks: Sequence[Sequence[tf.Tensor]],
) -> tf.Tensor:
    """Pack axis-major three-parameter tangent banks as ``[3,36,4,5,4]``."""

    if len(banks) != AXIS_COUNT or any(len(bank) != 3 for bank in banks):
        raise ValueError("tangent banks must have shape [36,3,...]")
    return tf.ensure_shape(
        tf.stack(
            [pack_cores(tuple(banks[axis][parameter] for axis in range(AXIS_COUNT)))
             for parameter in range(3)]
        ),
        [3, *PACKED_SHAPE],
    )


def packed_core_mask(shapes: Sequence[tf.TensorShape]) -> tf.Tensor:
    """Return the setup-static active-entry mask for packed boundary ranks."""

    if len(shapes) != AXIS_COUNT:
        raise ValueError("packed mask requires 36 core shapes")
    rows = []
    for axis, shape in enumerate(shapes):
        if shape.rank != 3 or int(shape[1]) != BASIS_DIM:
            raise ValueError(f"invalid mask shape at axis {axis}")
        left = int(shape[0])
        right = int(shape[2])
        active = tf.ones([left, BASIS_DIM, right], DTYPE)
        rows.append(
            tf.pad(active, [[0, MAX_RANK - left], [0, 0], [0, MAX_RANK - right]])
        )
    return tf.ensure_shape(tf.stack(rows), PACKED_SHAPE)


def precompute_basis_values(basis: object, points: tf.Tensor) -> tf.Tensor:
    """Evaluate setup-static per-axis basis tables with the claim XLA backend."""

    values = tf.convert_to_tensor(points, DTYPE)
    if values.shape.rank != 2 or values.shape[1] != AXIS_COUNT:
        raise ValueError("Lane-B basis points must have shape [sample,36]")

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(active_values: tf.Tensor) -> tf.Tensor:
        return tf.ensure_shape(
            tf.stack(
                [
                    basis.evaluate_axis(axis, active_values[:, axis])
                    for axis in range(AXIS_COUNT)
                ],
                axis=1,
            ),
            [values.shape[0], AXIS_COUNT, BASIS_DIM],
        )

    return compiled(values)


def precompute_mass_matrices(basis: object) -> tf.Tensor:
    """Stack setup-static Lane-B reference-measure mass matrices."""

    measure = basis.convention.mass_measure
    return tf.ensure_shape(
        tf.stack([basis.bases[axis].mass_matrix(measure) for axis in range(AXIS_COUNT)]),
        [AXIS_COUNT, BASIS_DIM, BASIS_DIM],
    )


def packed_amplitude(cores: tf.Tensor, basis_values: tf.Tensor) -> tf.Tensor:
    """Contract all TT axes using TensorFlow control flow only."""

    active = tf.ensure_shape(tf.convert_to_tensor(cores, DTYPE), PACKED_SHAPE)
    tables = tf.convert_to_tensor(basis_values, DTYPE)
    vector = tf.einsum("nl,lb->nb", tables[:, 0, :], active[0, 0, :, :])

    def condition(axis, _vector):
        return axis < AXIS_COUNT - 1

    def body(axis, current):
        matrix = tf.einsum("nl,alb->nab", tables[:, axis, :], active[axis])
        return axis + 1, tf.einsum("na,nab->nb", current, matrix)

    _, result = tf.while_loop(
        condition,
        body,
        (tf.constant(1, tf.int32), vector),
        parallel_iterations=1,
        maximum_iterations=AXIS_COUNT - 2,
    )
    final_matrix = tf.einsum(
        "nl,al->na", tables[:, -1, :], active[-1, :, :, 0]
    )
    return tf.einsum("na,na->n", result, final_matrix)


def packed_square_mass(cores: tf.Tensor, mass_matrices: tf.Tensor) -> tf.Tensor:
    """Contract the squared-TT mass using TensorFlow control flow only."""

    active = tf.ensure_shape(tf.convert_to_tensor(cores, DTYPE), PACKED_SHAPE)
    masses = tf.ensure_shape(
        tf.convert_to_tensor(mass_matrices, DTYPE),
        [AXIS_COUNT, BASIS_DIM, BASIS_DIM],
    )
    first = active[0, 0, :, :]
    first_pair = tf.einsum("lb,mB,lm->bB", first, first, masses[0])
    vector = tf.reshape(first_pair, [MAX_RANK * MAX_RANK])

    def condition(axis, _vector):
        return axis < AXIS_COUNT - 1

    def body(axis, current):
        core = active[axis]
        pair = tf.einsum("alb,AmB,lm->aAbB", core, core, masses[axis])
        matrix = tf.reshape(pair, [MAX_RANK * MAX_RANK, MAX_RANK * MAX_RANK])
        return axis + 1, tf.einsum("a,ab->b", current, matrix)

    _, result = tf.while_loop(
        condition,
        body,
        (tf.constant(1, tf.int32), vector),
        parallel_iterations=1,
        maximum_iterations=AXIS_COUNT - 2,
    )
    final = active[-1, :, :, 0]
    final_pair = tf.einsum("al,Am,lm->aA", final, final, masses[-1])
    return tf.einsum("aA,aA->", tf.reshape(result, [MAX_RANK, MAX_RANK]), final_pair)


def packed_prefix_square_numerator(
    cores: tf.Tensor,
    basis_values: tf.Tensor,
    mass_matrices: tf.Tensor,
    *,
    prefix_dim: int = 18,
) -> tf.Tensor:
    """Evaluate a squared-TT prefix marginal before the defensive term."""

    if int(prefix_dim) <= 0 or int(prefix_dim) > AXIS_COUNT:
        raise ValueError("prefix_dim must be in [1,36]")
    active = tf.ensure_shape(tf.convert_to_tensor(cores, DTYPE), PACKED_SHAPE)
    tables = tf.convert_to_tensor(basis_values, DTYPE)
    masses = tf.ensure_shape(
        tf.convert_to_tensor(mass_matrices, DTYPE),
        [AXIS_COUNT, BASIS_DIM, BASIS_DIM],
    )
    sample_count = tf.shape(tables)[0]
    first = active[0, 0, :, :]
    first_values = tables[:, 0, :]
    first_pair = tf.einsum(
        "nl,nm,lb,mB->nbB", first_values, first_values, first, first
    )
    vector = tf.reshape(first_pair, [sample_count, MAX_RANK * MAX_RANK])

    def condition(axis, _vector):
        return axis < AXIS_COUNT - 1

    def body(axis, current):
        core = active[axis]

        def evaluated_matrix():
            evaluated = tables[:, axis, :]
            pair = tf.einsum(
                "nl,nm,alb,AmB->naAbB", evaluated, evaluated, core, core
            )
            return tf.reshape(
                pair,
                [sample_count, MAX_RANK * MAX_RANK, MAX_RANK * MAX_RANK],
            )

        def integrated_matrix():
            pair = tf.einsum(
                "alb,AmB,lm->aAbB", core, core, masses[axis]
            )
            matrix = tf.reshape(
                pair, [MAX_RANK * MAX_RANK, MAX_RANK * MAX_RANK]
            )
            return tf.broadcast_to(
                matrix[tf.newaxis, :, :],
                [sample_count, MAX_RANK * MAX_RANK, MAX_RANK * MAX_RANK],
            )

        matrix = tf.cond(
            axis < tf.constant(prefix_dim, tf.int32),
            evaluated_matrix,
            integrated_matrix,
        )
        return axis + 1, tf.einsum("na,nab->nb", current, matrix)

    _, result = tf.while_loop(
        condition,
        body,
        (tf.constant(1, tf.int32), vector),
        parallel_iterations=1,
        maximum_iterations=AXIS_COUNT - 2,
    )
    final = active[-1, :, :, 0]
    if prefix_dim == AXIS_COUNT:
        final_values = tables[:, -1, :]
        final_pair = tf.einsum(
            "nl,nm,al,Am->naA", final_values, final_values, final, final
        )
    else:
        integrated = tf.einsum("al,Am,lm->aA", final, final, masses[-1])
        final_pair = tf.broadcast_to(
            integrated[tf.newaxis, :, :], [sample_count, MAX_RANK, MAX_RANK]
        )
    return tf.einsum(
        "naA,naA->n", tf.reshape(result, [sample_count, MAX_RANK, MAX_RANK]), final_pair
    )


def packed_per_core_regularizers(
    cores: tf.Tensor, mask: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Reproduce the authority's per-core L1/L2 reduction tree."""

    active = tf.ensure_shape(tf.convert_to_tensor(cores, DTYPE), PACKED_SHAPE)
    active_mask = tf.ensure_shape(tf.convert_to_tensor(mask, DTYPE), PACKED_SHAPE)
    masked = active * active_mask
    per_core_l1 = tf.concat(
        (
            tf.reshape(tf.reduce_sum(tf.abs(masked[0, 0, :, :])), [1]),
            tf.reduce_sum(tf.abs(masked[1:-1]), axis=(1, 2, 3)),
            tf.reshape(tf.reduce_sum(tf.abs(masked[-1, :, :, 0])), [1]),
        ),
        axis=0,
    )
    per_core_l2 = tf.concat(
        (
            tf.reshape(tf.reduce_sum(tf.square(masked[0, 0, :, :])), [1]),
            tf.reduce_sum(tf.square(masked[1:-1]), axis=(1, 2, 3)),
            tf.reshape(tf.reduce_sum(tf.square(masked[-1, :, :, 0])), [1]),
        ),
        axis=0,
    )
    return (
        tf.raw_ops.AddN(inputs=tf.unstack(per_core_l1, num=AXIS_COUNT)),
        tf.raw_ops.AddN(inputs=tf.unstack(per_core_l2, num=AXIS_COUNT)),
    )


def packed_tuple_global_norm(gradients: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Return the norm of 36 logical core gradients, excluding padding."""

    active = tf.ensure_shape(tf.convert_to_tensor(gradients, DTYPE), PACKED_SHAPE)
    active_mask = tf.ensure_shape(tf.convert_to_tensor(mask, DTYPE), PACKED_SHAPE)
    masked = active * active_mask
    per_core_half_square = tf.concat(
        (
            tf.reshape(tf.nn.l2_loss(masked[0, 0, :, :]), [1]),
            tf.constant(0.5, DTYPE)
            * tf.reduce_sum(tf.square(masked[1:-1]), axis=(1, 2, 3)),
            tf.reshape(tf.nn.l2_loss(masked[-1, :, :, 0]), [1]),
        ),
        axis=0,
    )
    return tf.sqrt(tf.constant(2.0, DTYPE) * tf.reduce_sum(per_core_half_square))


def packed_normalized_density(
    cores: tf.Tensor,
    basis_values: tf.Tensor,
    mass_matrices: tf.Tensor,
    tau: tf.Tensor,
) -> tf.Tensor:
    """Return normalized defensive full-density values."""

    amplitude = packed_amplitude(cores, basis_values)
    normalizer = packed_square_mass(cores, mass_matrices) + tau
    return (tf.square(amplitude) + tau) / normalizer


def packed_normalized_prefix_density(
    cores: tf.Tensor,
    basis_values: tf.Tensor,
    mass_matrices: tf.Tensor,
    tau: tf.Tensor,
    *,
    prefix_dim: int = 18,
) -> tf.Tensor:
    """Return normalized defensive prefix-marginal values."""

    numerator = packed_prefix_square_numerator(
        cores, basis_values, mass_matrices, prefix_dim=prefix_dim
    ) + tau
    normalizer = packed_square_mass(cores, mass_matrices) + tau
    return numerator / normalizer


def packed_adam_apply_gradients(
    cores: tf.Tensor,
    momentums: tf.Tensor,
    velocities: tf.Tensor,
    gradients: tf.Tensor,
    mask: tf.Tensor,
    *,
    step: tf.Tensor,
    learning_rate: tf.Tensor,
    gradient_clip_norm: tf.Tensor,
    beta_1: float,
    beta_2: float,
    epsilon: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply the frozen Keras-3 Adam operation order to a packed TT tensor."""

    active_mask = tf.ensure_shape(tf.convert_to_tensor(mask, DTYPE), PACKED_SHAPE)
    active_cores = tf.ensure_shape(tf.convert_to_tensor(cores, DTYPE), PACKED_SHAPE)
    active_m = tf.ensure_shape(tf.convert_to_tensor(momentums, DTYPE), PACKED_SHAPE)
    active_v = tf.ensure_shape(tf.convert_to_tensor(velocities, DTYPE), PACKED_SHAPE)
    active_g = tf.ensure_shape(tf.convert_to_tensor(gradients, DTYPE), PACKED_SHAPE)
    active_g = active_g * active_mask
    global_norm = packed_tuple_global_norm(active_g, active_mask)
    scale = gradient_clip_norm * tf.math.minimum(
        tf.math.reciprocal(global_norm),
        tf.math.reciprocal(gradient_clip_norm),
    )
    scale = scale + (global_norm - global_norm)
    gradient = active_g * scale
    beta1 = tf.cast(tf.constant(beta_1, tf.float32), DTYPE)
    beta2 = tf.cast(tf.constant(beta_2, tf.float32), DTYPE)
    local_step = tf.cast(step, DTYPE)
    alpha = learning_rate * tf.sqrt(1.0 - tf.pow(beta2, local_step)) / (
        1.0 - tf.pow(beta1, local_step)
    )
    next_m = active_m + (gradient - active_m) * tf.constant(1.0 - beta_1, DTYPE)
    next_v = active_v + (tf.square(gradient) - active_v) * tf.constant(
        1.0 - beta_2, DTYPE
    )
    next_cores = active_cores - alpha * next_m / (
        tf.sqrt(next_v) + tf.constant(epsilon, DTYPE)
    )
    return (
        next_cores * active_mask,
        next_m * active_mask,
        next_v * active_mask,
    )


def material_replay_metrics(
    replayed: tf.Tensor,
    parent: tf.Tensor,
    mask: tf.Tensor,
    *,
    atol: float = MATERIAL_REPLAY_ATOL,
    rtol: float = MATERIAL_REPLAY_RTOL,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return pass, maximum absolute residual, and normalized residual."""

    active_mask = tf.ensure_shape(tf.convert_to_tensor(mask, DTYPE), PACKED_SHAPE)
    observed = tf.ensure_shape(tf.convert_to_tensor(replayed, DTYPE), PACKED_SHAPE)
    reference = tf.ensure_shape(tf.convert_to_tensor(parent, DTYPE), PACKED_SHAPE)
    residual = tf.abs(observed - reference) * active_mask
    threshold = (
        tf.constant(atol, DTYPE) + tf.constant(rtol, DTYPE) * tf.abs(reference)
    )
    normalized = tf.math.divide_no_nan(residual, threshold) * active_mask
    maximum_absolute = tf.reduce_max(residual)
    maximum_normalized = tf.reduce_max(normalized)
    passed = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(observed)),
        maximum_normalized <= tf.constant(1.0, DTYPE),
    )
    return passed, maximum_absolute, maximum_normalized


def material_positive_value_metrics(
    observed: tf.Tensor,
    reference: tf.Tensor,
    *,
    atol: float = MATERIAL_REPLAY_ATOL,
    rtol: float = MATERIAL_REPLAY_RTOL,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compare positive functional values under the five-digit mixed rule."""

    candidate = tf.convert_to_tensor(observed, DTYPE)
    authority = tf.convert_to_tensor(reference, DTYPE)
    residual = tf.abs(candidate - authority)
    threshold = tf.constant(atol, DTYPE) + tf.constant(rtol, DTYPE) * tf.abs(
        authority
    )
    normalized = tf.math.divide_no_nan(residual, threshold)
    positive_finite = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(candidate)),
        tf.logical_and(
            tf.reduce_all(tf.math.is_finite(authority)),
            tf.logical_and(
                tf.reduce_all(candidate > 0.0), tf.reduce_all(authority > 0.0)
            ),
        ),
    )
    passed = tf.logical_and(
        positive_finite, tf.reduce_max(normalized) <= tf.constant(1.0, DTYPE)
    )
    maximum_log_residual = tf.reduce_max(
        tf.abs(tf.math.log(candidate) - tf.math.log(authority))
    )
    return passed, tf.reduce_max(residual), tf.reduce_max(normalized), maximum_log_residual


__all__ = [
    "MATERIAL_REPLAY_ATOL",
    "MATERIAL_REPLAY_POLICY_ID",
    "MATERIAL_REPLAY_RTOL",
    "PACKED_SHAPE",
    "PACKED_XLA_POLICY_ID",
    "material_replay_metrics",
    "material_positive_value_metrics",
    "pack_cores",
    "pack_tangent_banks",
    "packed_adam_apply_gradients",
    "packed_amplitude",
    "packed_core_mask",
    "packed_normalized_density",
    "packed_normalized_prefix_density",
    "packed_prefix_square_numerator",
    "packed_square_mass",
    "precompute_basis_values",
    "precompute_mass_matrices",
    "unpack_cores",
]

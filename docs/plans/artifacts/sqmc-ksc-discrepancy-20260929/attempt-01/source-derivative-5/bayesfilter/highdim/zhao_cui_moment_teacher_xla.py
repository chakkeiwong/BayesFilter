"""Graph-native padded fixed-ALS mechanics for the moment teacher.

This is an opt-in ``extension_or_invention`` candidate.  It deliberately does
not import the setup-static reference implementation: all axis, environment,
fit, and tangent recurrences in the runtime function are TensorFlow control
flow.  Ranks and basis sizes are padded before tracing and are never selected
from a parameter or runtime residual.

The candidate requires a strictly positive ridge.  With zero design columns
for inactive padded entries, the positive ridge makes those coefficients zero
and leaves the active fixed-rank normal equations unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


ROUTE_ID = "zhao_cui_fixed_als_padded_xla_value_jvp_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
_FLOAT64_EPS = 2.220446049250313e-16


@dataclass(frozen=True)
class PaddedALSSetup:
    """Immutable setup tensors for a padded fixed-branch ALS replay.

    ``basis_values`` is ``[axis, row, padded_basis]`` and ``active_mask`` is
    ``[axis, padded_rank, padded_basis, padded_rank]``.  The schedule is a
    vector of axis indices, usually the repeated fixed sweep order.
    """

    basis_values: tf.Tensor
    active_mask: tf.Tensor
    schedule: tf.Tensor
    ridge: float
    column_scale_floor: float = 2.220446049250313e-16
    condition_number_veto: float = 1.0e14
    residual_veto: float = 1.0e-7

    def __post_init__(self) -> None:
        basis_values = tf.convert_to_tensor(self.basis_values, tf.float64)
        active_mask = tf.convert_to_tensor(self.active_mask, tf.float64)
        schedule = tf.convert_to_tensor(self.schedule, tf.int32)
        if basis_values.shape.rank != 3:
            raise ValueError("basis_values must have shape [axis,row,basis]")
        if active_mask.shape.rank != 4:
            raise ValueError("active_mask must have shape [axis,rank,basis,rank]")
        if schedule.shape.rank != 1 or schedule.shape[0] is None:
            raise ValueError("schedule must be a nonempty static vector")
        if basis_values.shape[0] != active_mask.shape[0]:
            raise ValueError("basis/mask axis count mismatch")
        if basis_values.shape[2] != active_mask.shape[2]:
            raise ValueError("basis/mask padded basis mismatch")
        if int(schedule.shape[0]) < 1:
            raise ValueError("schedule must be nonempty")
        if not bool(tf.reduce_all(tf.math.is_finite(basis_values)).numpy()):
            raise ValueError("basis_values must be finite")
        if not bool(tf.reduce_all(tf.math.is_finite(active_mask)).numpy()):
            raise ValueError("active_mask must be finite")
        if bool(tf.reduce_any((active_mask < 0.0) | (active_mask > 1.0)).numpy()):
            raise ValueError("active_mask must be in [0,1]")
        if not bool(tf.reduce_all(tf.math.is_finite(tf.cast(schedule, tf.float64))).numpy()):
            raise ValueError("schedule must be finite")
        if bool(tf.reduce_any((schedule < 0) | (schedule >= tf.shape(basis_values)[0])).numpy()):
            raise ValueError("schedule axis is out of range")
        if float(self.ridge) <= 0.0:
            raise ValueError("padded fixed ALS requires a strictly positive ridge")
        if float(self.column_scale_floor) <= 0.0:
            raise ValueError("column_scale_floor must be positive")
        if float(self.condition_number_veto) <= 0.0 or float(self.residual_veto) <= 0.0:
            raise ValueError("condition and residual vetoes must be positive")
        object.__setattr__(self, "basis_values", basis_values)
        object.__setattr__(self, "active_mask", active_mask)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "ridge", float(self.ridge))
        object.__setattr__(self, "column_scale_floor", float(self.column_scale_floor))
        object.__setattr__(self, "condition_number_veto", float(self.condition_number_veto))
        object.__setattr__(self, "residual_veto", float(self.residual_veto))

    @property
    def axis_count(self) -> int:
        return int(self.basis_values.shape[0])

    @property
    def row_count(self) -> int:
        return int(self.basis_values.shape[1])

    @property
    def padded_basis(self) -> int:
        return int(self.basis_values.shape[2])

    @property
    def padded_rank(self) -> int:
        return int(self.active_mask.shape[1])


def pad_tt_cores(
    cores: tf.Tensor,
    active_mask: tf.Tensor,
    *,
    padded_rank: int,
    padded_basis: int,
) -> tf.Tensor:
    """Embed a variable-rank core stack in the declared padded shape."""

    values = tf.convert_to_tensor(cores, tf.float64)
    mask = tf.convert_to_tensor(active_mask, tf.float64)
    if values.shape.rank != 4 or mask.shape.rank != 4:
        raise ValueError("cores and active_mask must be rank four")
    if values.shape[0] != mask.shape[0]:
        raise ValueError("core/mask axis mismatch")
    padded = tf.pad(
        values,
        [[0, 0], [0, padded_rank - int(values.shape[1])],
         [0, padded_basis - int(values.shape[2])],
         [0, padded_rank - int(values.shape[3])]],
    )
    return padded * mask


def _core_matrices(cores: tf.Tensor, basis_values: tf.Tensor) -> tf.Tensor:
    """Evaluate every padded core at every prepared row."""

    return tf.einsum("knl,kalb->knab", basis_values, cores)


def _environments(
    matrices: tf.Tensor,
    dot_matrices: tf.Tensor,
    axis: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return left/right environments and their JVPs for one update axis."""

    axis_count = tf.shape(matrices)[0]
    row_count = tf.shape(matrices)[1]
    rank = tf.shape(matrices)[2]
    first = tf.one_hot(0, rank, dtype=matrices.dtype)
    left = tf.broadcast_to(first[tf.newaxis, :], [row_count, rank])
    dot_left = tf.zeros_like(left)

    def left_cond(index, *_):
        return index < axis

    def left_body(index, value, dot_value):
        matrix = matrices[index]
        dot_matrix = dot_matrices[index]
        return (
            index + 1,
            tf.einsum("ma,mab->mb", value, matrix),
            tf.einsum("ma,mab->mb", dot_value, matrix)
            + tf.einsum("ma,mab->mb", value, dot_matrix),
        )

    _, left, dot_left = tf.while_loop(
        left_cond,
        left_body,
        (tf.zeros([], tf.int32), left, dot_left),
        parallel_iterations=1,
    )

    last = tf.one_hot(0, rank, dtype=matrices.dtype)
    right = tf.broadcast_to(last[tf.newaxis, :], [row_count, rank])
    dot_right = tf.zeros_like(right)

    def right_cond(index, *_):
        return index > axis

    def right_body(index, value, dot_value):
        matrix = matrices[index]
        dot_matrix = dot_matrices[index]
        return (
            index - 1,
            tf.einsum("mab,mb->ma", matrix, value),
            tf.einsum("mab,mb->ma", dot_matrix, value)
            + tf.einsum("mab,mb->ma", matrix, dot_value),
        )

    _, right, dot_right = tf.while_loop(
        right_cond,
        right_body,
        (axis_count - 1, right, dot_right),
        parallel_iterations=1,
    )
    return left, dot_left, right, dot_right


def _design_and_jvp(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    basis_values: tf.Tensor,
    active_mask: tf.Tensor,
    axis: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    matrices = _core_matrices(cores, basis_values)
    dot_matrices = _core_matrices(dot_cores, basis_values)
    left, dot_left, right, dot_right = _environments(matrices, dot_matrices, axis)
    basis = basis_values[axis]
    design = tf.einsum("ma,ml,mb->malb", left, basis, right)
    dot_design = tf.einsum("ma,ml,mb->malb", dot_left, basis, right) + tf.einsum(
        "ma,ml,mb->malb", left, basis, dot_right
    )
    flat_mask = tf.reshape(active_mask[axis], [-1])
    return tf.reshape(design, [tf.shape(design)[0], -1]) * flat_mask, tf.reshape(
        dot_design, [tf.shape(dot_design)[0], -1]
    ) * flat_mask


def _padded_paired_step(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    operator: tf.Tensor,
    basis: tf.Tensor,
    use_points: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Build one query or integrated paired-core transfer without host loops."""

    core = cores
    dot_core = dot_cores
    point_pair = tf.einsum(
        "ql,qm,alb,AmB->qaAbB", basis, basis, core, core
    )
    dot_point_pair = tf.einsum(
        "ql,qm,alb,AmB->qaAbB", basis, basis, dot_core, core
    ) + tf.einsum("ql,qm,alb,AmB->qaAbB", basis, basis, core, dot_core)
    integrated = tf.einsum("alb,AmB,lm->aAbB", core, core, operator)
    dot_integrated = tf.einsum(
        "alb,AmB,lm->aAbB", dot_core, core, operator
    ) + tf.einsum("alb,AmB,lm->aAbB", core, dot_core, operator)
    count = tf.shape(basis)[0]
    integrated = tf.broadcast_to(integrated[tf.newaxis], tf.shape(point_pair))
    dot_integrated = tf.broadcast_to(dot_integrated[tf.newaxis], tf.shape(point_pair))
    paired = tf.where(use_points, point_pair, integrated)
    dot_paired = tf.where(use_points, dot_point_pair, dot_integrated)
    return (
        tf.reshape(paired, [count, tf.shape(core)[0] ** 2, tf.shape(core)[2] ** 2]),
        tf.reshape(dot_paired, [count, tf.shape(core)[0] ** 2, tf.shape(core)[2] ** 2]),
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def padded_squared_tt_normalized_marginal_jvp_xla(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    query_basis_values: tf.Tensor,
    keep_mask: tf.Tensor,
    mass_operators: tf.Tensor,
    tau: tf.Tensor,
    dot_tau: tf.Tensor,
    defensive_values: tf.Tensor,
    dot_defensive_values: tf.Tensor,
    defensive_mass: tf.Tensor,
    dot_defensive_mass: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Graph-native normalized retained-marginal value and JVP."""

    cores = tf.convert_to_tensor(cores)
    dtype = cores.dtype
    dot_cores = tf.convert_to_tensor(dot_cores, dtype)
    query_basis_values = tf.convert_to_tensor(query_basis_values, dtype)
    keep_mask = tf.cast(tf.convert_to_tensor(keep_mask), tf.bool)
    mass_operators = tf.convert_to_tensor(mass_operators, dtype)
    query_count = tf.shape(query_basis_values)[1]
    rank = tf.shape(cores)[1]
    pair_rank = rank * rank
    initial = tf.one_hot(0, pair_rank, dtype=dtype)
    vector = tf.broadcast_to(initial[tf.newaxis], [query_count, pair_rank])
    dot_vector = tf.zeros_like(vector)
    mass_vector = tf.broadcast_to(initial[tf.newaxis], [1, pair_rank])
    dot_mass_vector = tf.zeros_like(mass_vector)

    def cond(axis, *_):
        return axis < tf.shape(cores)[0]

    def body(axis, vector, dot_vector, mass_vector, dot_mass_vector):
        operator = mass_operators[axis]
        basis = query_basis_values[axis]
        paired, dot_paired = _padded_paired_step(
            cores[axis], dot_cores[axis], operator, basis, keep_mask[axis]
        )
        vector_next = tf.einsum("qa,qab->qb", vector, paired)
        dot_vector_next = tf.einsum("qa,qab->qb", dot_vector, paired) + tf.einsum(
            "qa,qab->qb", vector, dot_paired
        )
        mass_basis = tf.ones([1, tf.shape(basis)[1]], dtype=dtype)
        mass_pair, dot_mass_pair = _padded_paired_step(
            cores[axis], dot_cores[axis], operator, mass_basis, tf.constant(False)
        )
        mass_next = tf.einsum("qa,qab->qb", mass_vector, mass_pair)
        dot_mass_next = tf.einsum("qa,qab->qb", dot_mass_vector, mass_pair) + tf.einsum(
            "qa,qab->qb", mass_vector, dot_mass_pair
        )
        return axis + 1, vector_next, dot_vector_next, mass_next, dot_mass_next

    _, vector, dot_vector, mass_vector, dot_mass_vector = tf.while_loop(
        cond,
        body,
        (
            tf.zeros([], tf.int32),
            vector,
            dot_vector,
            mass_vector,
            dot_mass_vector,
        ),
        parallel_iterations=1,
    )
    sqrt_numerator = vector[:, 0]
    dot_sqrt_numerator = dot_vector[:, 0]
    sqrt_mass = mass_vector[0, 0]
    dot_sqrt_mass = dot_mass_vector[0, 0]
    tau = tf.cast(tau, dtype)
    dot_tau = tf.cast(dot_tau, dtype)
    defensive_values = tf.cast(defensive_values, dtype)
    dot_defensive_values = tf.cast(dot_defensive_values, dtype)
    defensive_mass = tf.cast(defensive_mass, dtype)
    dot_defensive_mass = tf.cast(dot_defensive_mass, dtype)
    numerator = sqrt_numerator + tau * defensive_values
    dot_numerator = dot_sqrt_numerator + dot_tau * defensive_values + tau * dot_defensive_values
    normalizer = sqrt_mass + tau * defensive_mass
    dot_normalizer = dot_sqrt_mass + dot_tau * defensive_mass + tau * dot_defensive_mass
    value = numerator / normalizer
    tangent = (dot_numerator * normalizer - numerator * dot_normalizer) / tf.square(normalizer)
    return value, tangent, normalizer, dot_normalizer


def _paired_transfer_batch(
    core: tf.Tensor,
    dot_core: tf.Tensor,
    operators: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    paired = tf.einsum("alb,AmB,olm->oaAbB", core, core, operators)
    dot_paired = tf.einsum(
        "alb,AmB,olm->oaAbB", dot_core, core, operators
    ) + tf.einsum("alb,AmB,olm->oaAbB", core, dot_core, operators)
    pair_rank = tf.shape(core)[0] ** 2
    next_pair_rank = tf.shape(core)[2] ** 2
    return (
        tf.reshape(paired, [tf.shape(operators)[0], pair_rank, next_pair_rank]),
        tf.reshape(dot_paired, [tf.shape(operators)[0], pair_rank, next_pair_rank]),
    )


def _padded_separable_observables_jvp(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    operator_powers: tf.Tensor,
    power_vectors: tf.Tensor,
    tau: tf.Tensor,
    dot_tau: tf.Tensor,
    defensive_power_moments: tf.Tensor,
    defensive_mass: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Contract a batch of separable monomials and their core/tau JVPs."""

    dtype = cores.dtype
    observable_count = tf.shape(power_vectors)[0]
    rank = tf.shape(cores)[1]
    pair_rank = rank * rank
    initial = tf.one_hot(0, pair_rank, dtype=dtype)
    vector = tf.broadcast_to(initial[tf.newaxis], [observable_count, pair_rank])
    dot_vector = tf.zeros_like(vector)

    def cond(axis, *_):
        return axis < tf.shape(cores)[0]

    def body(axis, vector, dot_vector):
        powers = power_vectors[:, axis]
        operators = tf.gather(operator_powers[axis], powers)
        paired, dot_paired = _paired_transfer_batch(
            cores[axis], dot_cores[axis], operators
        )
        next_vector = tf.einsum("oa,oab->ob", vector, paired)
        next_dot = tf.einsum("oa,oab->ob", dot_vector, paired) + tf.einsum(
            "oa,oab->ob", vector, dot_paired
        )
        return axis + 1, next_vector, next_dot

    _, vector, dot_vector = tf.while_loop(
        cond,
        body,
        (tf.zeros([], tf.int32), vector, dot_vector),
        parallel_iterations=1,
    )
    sqrt_values = vector[:, 0]
    dot_sqrt_values = dot_vector[:, 0]
    axis_ids = tf.broadcast_to(
        tf.range(tf.shape(power_vectors)[1], dtype=tf.int32)[tf.newaxis, :],
        tf.shape(power_vectors),
    )
    defensive_indices = tf.stack([axis_ids, power_vectors], axis=-1)
    defensive_values = tf.reduce_prod(
        tf.gather_nd(defensive_power_moments, defensive_indices), axis=1
    )
    mass_powers = tf.zeros([1, tf.shape(power_vectors)[1]], tf.int32)
    mass_vector = tf.one_hot(0, pair_rank, dtype=dtype)[tf.newaxis, :]
    dot_mass_vector = tf.zeros_like(mass_vector)

    def mass_body(axis, mass_vector, dot_mass_vector):
        operators = operator_powers[axis, 0][tf.newaxis, ...]
        paired, dot_paired = _paired_transfer_batch(
            cores[axis], dot_cores[axis], operators
        )
        next_mass = tf.einsum("oa,oab->ob", mass_vector, paired)
        next_dot_mass = tf.einsum(
            "oa,oab->ob", dot_mass_vector, paired
        ) + tf.einsum("oa,oab->ob", mass_vector, dot_paired)
        return axis + 1, next_mass, next_dot_mass

    _, mass_vector, dot_mass_vector = tf.while_loop(
        lambda axis, *_: axis < tf.shape(cores)[0],
        mass_body,
        (tf.zeros([], tf.int32), mass_vector, dot_mass_vector),
        parallel_iterations=1,
    )
    del mass_powers
    sqrt_mass = mass_vector[0, 0]
    dot_sqrt_mass = dot_mass_vector[0, 0]
    normalizer = sqrt_mass + tau * defensive_mass
    dot_normalizer = dot_sqrt_mass + dot_tau * defensive_mass
    numerator = sqrt_values + tau * defensive_values
    dot_numerator = dot_sqrt_values + dot_tau * defensive_values
    values = numerator / normalizer
    tangents = (
        dot_numerator * normalizer - numerator * dot_normalizer
    ) / tf.square(normalizer)
    return values, tangents, normalizer, dot_normalizer


def _integer_power_jvp(
    value: tf.Tensor,
    tangent: tf.Tensor,
    exponent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    exponent_value = tf.cast(exponent, value.dtype)
    primal = tf.pow(value, exponent_value)
    safe_exponent = tf.maximum(exponent - 1, 0)
    derivative = exponent_value * tf.pow(value, tf.cast(safe_exponent, value.dtype)) * tangent
    return primal, derivative


def _padded_affine_moments_jvp(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    operator_powers: tf.Tensor,
    defensive_power_moments: tf.Tensor,
    first_coefficients: tf.Tensor,
    dot_first_coefficients: tf.Tensor,
    first_offsets: tf.Tensor,
    dot_first_offsets: tf.Tensor,
    first_powers: tf.Tensor,
    second_coefficients: tf.Tensor,
    dot_second_coefficients: tf.Tensor,
    second_offsets: tf.Tensor,
    dot_second_offsets: tf.Tensor,
    second_powers: tf.Tensor,
    tau: tf.Tensor,
    dot_tau: tf.Tensor,
    defensive_mass: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Contract batched affine-product moments of total declared degree <= 4."""

    dtype = cores.dtype
    max_degree = 4
    side = max_degree + 1
    degree = tf.range(side, dtype=tf.int32)
    out_first, out_second = tf.meshgrid(degree, degree, indexing="ij")
    state_first = tf.reshape(out_first, [-1])
    state_second = tf.reshape(out_second, [-1])
    in_first = state_first[tf.newaxis, :]
    in_second = state_second[tf.newaxis, :]
    out_first = state_first[:, tf.newaxis]
    out_second = state_second[:, tf.newaxis]
    delta_first = out_first - in_first
    delta_second = out_second - in_second
    valid_state = (state_first + state_second) <= max_degree
    valid_transition = (
        (delta_first >= 0)
        & (delta_second >= 0)
        & valid_state[:, tf.newaxis]
        & valid_state[tf.newaxis, :]
    )
    safe_delta_first = tf.maximum(delta_first, 0)
    safe_delta_second = tf.maximum(delta_second, 0)
    combination = tf.exp(
        tf.math.lgamma(tf.cast(out_first + 1, dtype))
        - tf.math.lgamma(tf.cast(in_first + 1, dtype))
        - tf.math.lgamma(tf.cast(safe_delta_first + 1, dtype))
        + tf.math.lgamma(tf.cast(out_second + 1, dtype))
        - tf.math.lgamma(tf.cast(in_second + 1, dtype))
        - tf.math.lgamma(tf.cast(safe_delta_second + 1, dtype))
    )
    combination = tf.where(valid_transition, combination, tf.zeros_like(combination))
    observable_count = tf.shape(first_coefficients)[0]
    pair_rank = tf.shape(cores)[1] ** 2
    first_initial, dot_first_initial = _integer_power_jvp(
        first_offsets[:, tf.newaxis],
        dot_first_offsets[:, tf.newaxis],
        state_first[tf.newaxis, :],
    )
    second_initial, dot_second_initial = _integer_power_jvp(
        second_offsets[:, tf.newaxis],
        dot_second_offsets[:, tf.newaxis],
        state_second[tf.newaxis, :],
    )
    initial_scalar = first_initial * second_initial
    dot_initial_scalar = (
        dot_first_initial * second_initial + first_initial * dot_second_initial
    )
    boundary = tf.one_hot(0, pair_rank, dtype=dtype)
    state = initial_scalar[:, :, tf.newaxis] * boundary[tf.newaxis, tf.newaxis, :]
    dot_state = dot_initial_scalar[:, :, tf.newaxis] * boundary[tf.newaxis, tf.newaxis, :]
    defensive_state = initial_scalar
    dot_defensive_state = dot_initial_scalar

    def body(axis, state, dot_state, defensive_state, dot_defensive_state):
        first = first_coefficients[:, axis]
        dot_first = dot_first_coefficients[:, axis]
        second = second_coefficients[:, axis]
        dot_second = dot_second_coefficients[:, axis]
        first_factor, dot_first_factor = _integer_power_jvp(
            first[:, tf.newaxis, tf.newaxis],
            dot_first[:, tf.newaxis, tf.newaxis],
            safe_delta_first[tf.newaxis, :, :],
        )
        second_factor, dot_second_factor = _integer_power_jvp(
            second[:, tf.newaxis, tf.newaxis],
            dot_second[:, tf.newaxis, tf.newaxis],
            safe_delta_second[tf.newaxis, :, :],
        )
        coefficient = combination[tf.newaxis, :, :] * first_factor * second_factor
        dot_coefficient = combination[tf.newaxis, :, :] * (
            dot_first_factor * second_factor + first_factor * dot_second_factor
        )
        local_power = tf.minimum(safe_delta_first + safe_delta_second, max_degree)
        operators = tf.gather(operator_powers[axis], local_power)
        paired = tf.einsum("alb,AmB,vulm->vuaAbB", cores[axis], cores[axis], operators)
        dot_paired = tf.einsum(
            "alb,AmB,vulm->vuaAbB", dot_cores[axis], cores[axis], operators
        ) + tf.einsum(
            "alb,AmB,vulm->vuaAbB", cores[axis], dot_cores[axis], operators
        )
        paired = tf.reshape(paired, [side * side, side * side, pair_rank, pair_rank])
        dot_paired = tf.reshape(
            dot_paired, [side * side, side * side, pair_rank, pair_rank]
        )
        local = coefficient[:, :, :, tf.newaxis, tf.newaxis] * paired[tf.newaxis, ...]
        dot_local = (
            dot_coefficient[:, :, :, tf.newaxis, tf.newaxis] * paired[tf.newaxis, ...]
            + coefficient[:, :, :, tf.newaxis, tf.newaxis] * dot_paired[tf.newaxis, ...]
        )
        next_state = tf.einsum("oup,ovupq->ovq", state, local)
        next_dot_state = tf.einsum("oup,ovupq->ovq", dot_state, local) + tf.einsum(
            "oup,ovupq->ovq", state, dot_local
        )
        defensive_local = coefficient * tf.gather(defensive_power_moments[axis], local_power)[tf.newaxis, ...]
        dot_defensive_local = dot_coefficient * tf.gather(
            defensive_power_moments[axis], local_power
        )[tf.newaxis, ...]
        next_defensive = tf.einsum("ou,ovu->ov", defensive_state, defensive_local)
        next_dot_defensive = tf.einsum(
            "ou,ovu->ov", dot_defensive_state, defensive_local
        ) + tf.einsum("ou,ovu->ov", defensive_state, dot_defensive_local)
        return axis + 1, next_state, next_dot_state, next_defensive, next_dot_defensive

    _, state, dot_state, defensive_state, dot_defensive_state = tf.while_loop(
        lambda axis, *_: axis < tf.shape(cores)[0],
        body,
        (
            tf.zeros([], tf.int32),
            state,
            dot_state,
            defensive_state,
            dot_defensive_state,
        ),
        parallel_iterations=1,
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape([None, 25, None]),
            tf.TensorShape([None, 25, None]),
            tf.TensorShape([None, 25]),
            tf.TensorShape([None, 25]),
        ),
    )
    output_indices = first_powers * side + second_powers
    row_indices = tf.range(observable_count, dtype=tf.int32)
    gather_indices = tf.stack([row_indices, output_indices], axis=1)
    sqrt_values = tf.gather_nd(state[:, :, 0], gather_indices)
    dot_sqrt_values = tf.gather_nd(dot_state[:, :, 0], gather_indices)
    defensive_values = tf.gather_nd(defensive_state, gather_indices)
    dot_defensive_values = tf.gather_nd(dot_defensive_state, gather_indices)
    zero_powers = tf.zeros([1, tf.shape(cores)[0]], tf.int32)
    _, _, normalizer, dot_normalizer = _padded_separable_observables_jvp(
        cores,
        dot_cores,
        operator_powers,
        zero_powers,
        tau,
        dot_tau,
        defensive_power_moments,
        defensive_mass,
    )
    numerator = sqrt_values + tau * defensive_values
    dot_numerator = (
        dot_sqrt_values
        + dot_tau * defensive_values
        + tau * dot_defensive_values
    )
    values = numerator / normalizer
    tangents = (
        dot_numerator * normalizer - numerator * dot_normalizer
    ) / tf.square(normalizer)
    return values, tangents


@tf.function(jit_compile=True, reduce_retracing=True)
def padded_squared_tt_affine_moments_jvp_xla(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    operator_powers: tf.Tensor,
    defensive_power_moments: tf.Tensor,
    first_coefficients: tf.Tensor,
    dot_first_coefficients: tf.Tensor,
    first_offsets: tf.Tensor,
    dot_first_offsets: tf.Tensor,
    first_powers: tf.Tensor,
    second_coefficients: tf.Tensor,
    dot_second_coefficients: tf.Tensor,
    second_offsets: tf.Tensor,
    dot_second_offsets: tf.Tensor,
    second_powers: tf.Tensor,
    tau: tf.Tensor,
    dot_tau: tf.Tensor,
    defensive_mass: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Public graph-native batch of fourth-order affine moments and JVPs."""

    cores = tf.convert_to_tensor(cores)
    dtype = cores.dtype
    return _padded_affine_moments_jvp(
        cores,
        tf.convert_to_tensor(dot_cores, dtype),
        tf.convert_to_tensor(operator_powers, dtype),
        tf.convert_to_tensor(defensive_power_moments, dtype),
        tf.convert_to_tensor(first_coefficients, dtype),
        tf.convert_to_tensor(dot_first_coefficients, dtype),
        tf.convert_to_tensor(first_offsets, dtype),
        tf.convert_to_tensor(dot_first_offsets, dtype),
        tf.convert_to_tensor(first_powers, tf.int32),
        tf.convert_to_tensor(second_coefficients, dtype),
        tf.convert_to_tensor(dot_second_coefficients, dtype),
        tf.convert_to_tensor(second_offsets, dtype),
        tf.convert_to_tensor(dot_second_offsets, dtype),
        tf.convert_to_tensor(second_powers, tf.int32),
        tf.cast(tau, dtype),
        tf.cast(dot_tau, dtype),
        tf.cast(defensive_mass, dtype),
    )


def _cholesky_jvp(chol: tf.Tensor, matrix_tangent: tf.Tensor) -> tf.Tensor:
    left = tf.linalg.triangular_solve(chol, matrix_tangent)
    inner = tf.linalg.triangular_solve(chol, tf.transpose(left))
    lower = tf.linalg.band_part(inner, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(lower))
    return chol @ phi


@tf.function(jit_compile=True, reduce_retracing=True)
def padded_squared_tt_shape_targets_jvp_xla(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    operator_powers: tf.Tensor,
    defensive_power_moments: tf.Tensor,
    state_offset: tf.Tensor,
    dot_state_offset: tf.Tensor,
    state_matrix: tf.Tensor,
    dot_state_matrix: tf.Tensor,
    pair_indices: tf.Tensor,
    tau: tf.Tensor,
    dot_tau: tf.Tensor,
    defensive_mass: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return diagonal and declared pairwise standardized shape targets/JVPs."""

    cores = tf.convert_to_tensor(cores)
    dtype = cores.dtype
    dot_cores = tf.convert_to_tensor(dot_cores, dtype)
    operator_powers = tf.convert_to_tensor(operator_powers, dtype)
    defensive_power_moments = tf.convert_to_tensor(defensive_power_moments, dtype)
    offset = tf.convert_to_tensor(state_offset, dtype)
    dot_offset = tf.convert_to_tensor(dot_state_offset, dtype)
    matrix = tf.convert_to_tensor(state_matrix, dtype)
    dot_matrix = tf.convert_to_tensor(dot_state_matrix, dtype)
    pair_indices = tf.convert_to_tensor(pair_indices, tf.int32)
    dimension = tf.shape(cores)[0]
    state_dimension = tf.shape(offset)[0]
    mean_powers = tf.one_hot(tf.range(dimension), dimension, dtype=tf.int32)
    row = tf.repeat(tf.range(dimension, dtype=tf.int32), dimension)
    column = tf.tile(tf.range(dimension, dtype=tf.int32), [dimension])
    second_powers = tf.one_hot(row, dimension, dtype=tf.int32) + tf.one_hot(
        column, dimension, dtype=tf.int32
    )
    raw_powers = tf.concat([mean_powers, second_powers], axis=0)
    raw_values, raw_tangents, _, _ = _padded_separable_observables_jvp(
        cores,
        dot_cores,
        operator_powers,
        raw_powers,
        tf.cast(tau, dtype),
        tf.cast(dot_tau, dtype),
        defensive_power_moments,
        tf.cast(defensive_mass, dtype),
    )
    mean = raw_values[:dimension]
    mean_tangent = raw_tangents[:dimension]
    raw_second = tf.reshape(raw_values[dimension:], [dimension, dimension])
    raw_second_tangent = tf.reshape(
        raw_tangents[dimension:], [dimension, dimension]
    )
    covariance_raw = raw_second - mean[:, tf.newaxis] * mean[tf.newaxis, :]
    covariance = 0.5 * (covariance_raw + tf.transpose(covariance_raw))
    covariance_tangent_raw = (
        raw_second_tangent
        - mean_tangent[:, tf.newaxis] * mean[tf.newaxis, :]
        - mean[:, tf.newaxis] * mean_tangent[tf.newaxis, :]
    )
    covariance_tangent = 0.5 * (
        covariance_tangent_raw + tf.transpose(covariance_tangent_raw)
    )
    state_mean = offset + tf.linalg.matvec(matrix, mean)
    state_mean_tangent = (
        dot_offset
        + tf.linalg.matvec(dot_matrix, mean)
        + tf.linalg.matvec(matrix, mean_tangent)
    )
    state_covariance_raw = matrix @ covariance @ tf.transpose(matrix)
    state_covariance = 0.5 * (
        state_covariance_raw + tf.transpose(state_covariance_raw)
    )
    state_covariance_tangent_raw = (
        dot_matrix @ covariance @ tf.transpose(matrix)
        + matrix @ covariance_tangent @ tf.transpose(matrix)
        + matrix @ covariance @ tf.transpose(dot_matrix)
    )
    state_covariance_tangent = 0.5 * (
        state_covariance_tangent_raw + tf.transpose(state_covariance_tangent_raw)
    )
    chol = tf.linalg.cholesky(state_covariance)
    dot_chol = _cholesky_jvp(chol, state_covariance_tangent)
    standardized_matrix = tf.linalg.triangular_solve(chol, matrix)
    dot_standardized_matrix = tf.linalg.triangular_solve(
        chol, dot_matrix - dot_chol @ standardized_matrix
    )
    standardized_offset = tf.linalg.triangular_solve(
        chol, (offset - state_mean)[:, tf.newaxis]
    )[:, 0]
    dot_standardized_offset = tf.linalg.triangular_solve(
        chol,
        (dot_offset - state_mean_tangent)[:, tf.newaxis]
        - dot_chol @ standardized_offset[:, tf.newaxis],
    )[:, 0]
    pair_count = tf.shape(pair_indices)[0]
    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    first_coefficients = tf.concat(
        [
            standardized_matrix,
            standardized_matrix,
            tf.gather(standardized_matrix, left),
            tf.gather(standardized_matrix, left),
        ],
        axis=0,
    )
    dot_first_coefficients = tf.concat(
        [
            dot_standardized_matrix,
            dot_standardized_matrix,
            tf.gather(dot_standardized_matrix, left),
            tf.gather(dot_standardized_matrix, left),
        ],
        axis=0,
    )
    first_offsets = tf.concat(
        [
            standardized_offset,
            standardized_offset,
            tf.gather(standardized_offset, left),
            tf.gather(standardized_offset, left),
        ],
        axis=0,
    )
    dot_first_offsets = tf.concat(
        [
            dot_standardized_offset,
            dot_standardized_offset,
            tf.gather(dot_standardized_offset, left),
            tf.gather(dot_standardized_offset, left),
        ],
        axis=0,
    )
    zeros_coefficients = tf.zeros_like(first_coefficients[: 2 * state_dimension])
    zeros_offsets = tf.zeros_like(first_offsets[: 2 * state_dimension])
    second_coefficients = tf.concat(
        [
            zeros_coefficients,
            tf.gather(standardized_matrix, right),
            tf.gather(standardized_matrix, right),
        ],
        axis=0,
    )
    dot_second_coefficients = tf.concat(
        [
            zeros_coefficients,
            tf.gather(dot_standardized_matrix, right),
            tf.gather(dot_standardized_matrix, right),
        ],
        axis=0,
    )
    second_offsets = tf.concat(
        [
            zeros_offsets,
            tf.gather(standardized_offset, right),
            tf.gather(standardized_offset, right),
        ],
        axis=0,
    )
    dot_second_offsets = tf.concat(
        [
            zeros_offsets,
            tf.gather(dot_standardized_offset, right),
            tf.gather(dot_standardized_offset, right),
        ],
        axis=0,
    )
    first_powers = tf.concat(
        [
            tf.fill([state_dimension], 3),
            tf.fill([state_dimension], 4),
            tf.fill([pair_count], 2),
            tf.fill([pair_count], 2),
        ],
        axis=0,
    )
    second_powers = tf.concat(
        [
            tf.zeros([2 * state_dimension], tf.int32),
            tf.ones([pair_count], tf.int32),
            tf.fill([pair_count], 2),
        ],
        axis=0,
    )
    values, tangents = _padded_affine_moments_jvp(
        cores,
        dot_cores,
        operator_powers,
        defensive_power_moments,
        first_coefficients,
        dot_first_coefficients,
        first_offsets,
        dot_first_offsets,
        first_powers,
        second_coefficients,
        dot_second_coefficients,
        second_offsets,
        dot_second_offsets,
        second_powers,
        tf.cast(tau, dtype),
        tf.cast(dot_tau, dtype),
        tf.cast(defensive_mass, dtype),
    )
    skew = values[:state_dimension]
    kurtosis = values[state_dimension : 2 * state_dimension]
    co_skew = values[2 * state_dimension : 2 * state_dimension + pair_count]
    co_kurtosis = values[2 * state_dimension + pair_count :]
    skew_tangent = tangents[:state_dimension]
    kurtosis_tangent = tangents[state_dimension : 2 * state_dimension]
    co_skew_tangent = tangents[
        2 * state_dimension : 2 * state_dimension + pair_count
    ]
    co_kurtosis_tangent = tangents[2 * state_dimension + pair_count :]
    return (
        skew,
        kurtosis,
        co_skew,
        co_kurtosis,
        skew_tangent,
        kurtosis_tangent,
        co_skew_tangent,
        co_kurtosis_tangent,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def padded_fixed_teacher_recursion_marginal_xla(
    basis_values: tf.Tensor,
    active_mask: tf.Tensor,
    schedule: tf.Tensor,
    base_log_targets: tf.Tensor,
    dot_base_log_targets: tf.Tensor,
    weights: tf.Tensor,
    dot_weights: tf.Tensor,
    initial_cores: tf.Tensor,
    initial_dot_cores: tf.Tensor,
    scale_shift_indices: tf.Tensor,
    defensive_weights: tf.Tensor,
    dot_defensive_weights: tf.Tensor,
    query_basis_values: tf.Tensor,
    keep_mask: tf.Tensor,
    mass_operators: tf.Tensor,
    defensive_marginal_values: tf.Tensor,
    dot_defensive_marginal_values: tf.Tensor,
    defensive_mass: tf.Tensor,
    dot_defensive_mass: tf.Tensor,
    ridge: tf.Tensor,
    column_scale_floor: tf.Tensor,
    condition_number_veto: tf.Tensor,
    residual_veto: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Graph-native fixed-branch teacher recursion through carried marginals.

    The return values are final cores/tangents, per-time marginal values and
    tangents, per-time normalizers and a scalar validity bit.  The particle
    likelihood increment is intentionally not an input or output: it remains
    owned by the particle lane.
    """

    dtype = tf.convert_to_tensor(initial_cores).dtype
    base_log_targets = tf.convert_to_tensor(base_log_targets, dtype)
    dot_base_log_targets = tf.convert_to_tensor(dot_base_log_targets, dtype)
    scale_shift_indices = tf.convert_to_tensor(scale_shift_indices, tf.int32)
    defensive_weights = tf.convert_to_tensor(defensive_weights, dtype)
    dot_defensive_weights = tf.convert_to_tensor(dot_defensive_weights, dtype)
    time_count = tf.shape(base_log_targets)[0]
    query_count = tf.shape(query_basis_values)[1]
    row_count = int(base_log_targets.shape[1])
    marginal_values = tf.TensorArray(dtype, size=time_count, element_shape=[row_count])
    marginal_tangents = tf.TensorArray(dtype, size=time_count, element_shape=[row_count])
    normalizers = tf.TensorArray(dtype, size=time_count, element_shape=[])
    valid = tf.constant(True)
    previous_values = tf.ones([query_count], dtype)
    previous_tangent = tf.zeros_like(previous_values)

    def cond(index, *_):
        return index < time_count

    def body(index, cores, dot_cores, previous_values, previous_tangent, marginal_values, marginal_tangents, normalizers, valid):
        log_target = base_log_targets[index]
        dot_log_target = dot_base_log_targets[index]
        has_previous = index > 0
        safe_previous = tf.maximum(previous_values, tf.constant(1e-30, dtype))
        log_target = tf.where(
            has_previous,
            log_target + tf.math.log(safe_previous),
            log_target,
        )
        dot_log_target = tf.where(
            has_previous,
            dot_log_target + previous_tangent / safe_previous,
            dot_log_target,
        )
        shift_index = scale_shift_indices[index]
        shift = log_target[shift_index]
        dot_shift = dot_log_target[shift_index]
        target = tf.exp(0.5 * (log_target - shift))
        dot_target = 0.5 * target * (dot_log_target - dot_shift)
        tau_scale = tf.exp(-shift)
        tau = tau_scale * defensive_weights[index]
        dot_tau = tau_scale * (dot_defensive_weights[index] - defensive_weights[index] * dot_shift)
        cores, dot_cores, diagnostics, step_valid = padded_fixed_als_value_jvp_xla(
            basis_values,
            active_mask,
            schedule,
            target,
            dot_target,
            weights,
            dot_weights,
            cores,
            dot_cores,
            ridge,
            column_scale_floor,
            condition_number_veto,
            residual_veto,
        )
        del diagnostics
        values, tangent, normalizer, dot_normalizer = padded_squared_tt_normalized_marginal_jvp_xla(
            cores,
            dot_cores,
            query_basis_values,
            keep_mask,
            mass_operators,
            tau,
            dot_tau,
            defensive_marginal_values[index],
            dot_defensive_marginal_values[index],
            defensive_mass,
            dot_defensive_mass,
        )
        step_valid = step_valid & tf.reduce_all(tf.math.is_finite(values)) & tf.reduce_all(tf.math.is_finite(tangent))
        valid = valid & step_valid
        marginal_values = marginal_values.write(index, values)
        marginal_tangents = marginal_tangents.write(index, tangent)
        normalizers = normalizers.write(index, normalizer)
        del dot_normalizer
        return index + 1, cores, dot_cores, values, tangent, marginal_values, marginal_tangents, normalizers, valid

    _, cores, dot_cores, _, _, marginal_values, marginal_tangents, normalizers, valid = tf.while_loop(
        cond,
        body,
        (
            tf.zeros([], tf.int32),
            tf.convert_to_tensor(initial_cores, dtype) * tf.cast(active_mask, dtype),
            tf.convert_to_tensor(initial_dot_cores, dtype) * tf.cast(active_mask, dtype),
            previous_values,
            previous_tangent,
            marginal_values,
            marginal_tangents,
            normalizers,
            valid,
        ),
        parallel_iterations=1,
    )
    poison = tf.constant(float("nan"), dtype)
    cores = tf.where(valid, cores, tf.fill(tf.shape(cores), poison))
    dot_cores = tf.where(valid, dot_cores, tf.fill(tf.shape(dot_cores), poison))
    return cores, dot_cores, marginal_values.stack(), marginal_tangents.stack(), normalizers.stack(), valid


@tf.function(jit_compile=True)
def padded_fixed_teacher_recursion_shape_xla(
    basis_values: tf.Tensor,
    active_mask: tf.Tensor,
    schedule: tf.Tensor,
    base_log_targets: tf.Tensor,
    dot_base_log_targets: tf.Tensor,
    weights: tf.Tensor,
    dot_weights: tf.Tensor,
    initial_cores: tf.Tensor,
    initial_dot_cores: tf.Tensor,
    scale_shift_indices: tf.Tensor,
    defensive_weights: tf.Tensor,
    dot_defensive_weights: tf.Tensor,
    query_basis_values: tf.Tensor,
    keep_mask: tf.Tensor,
    mass_operators: tf.Tensor,
    defensive_marginal_values: tf.Tensor,
    dot_defensive_marginal_values: tf.Tensor,
    defensive_mass: tf.Tensor,
    dot_defensive_mass: tf.Tensor,
    operator_powers: tf.Tensor,
    defensive_power_moments: tf.Tensor,
    state_offset: tf.Tensor,
    dot_state_offset: tf.Tensor,
    state_matrix: tf.Tensor,
    dot_state_matrix: tf.Tensor,
    pair_indices: tf.Tensor,
    ridge: tf.Tensor,
    column_scale_floor: tf.Tensor,
    condition_number_veto: tf.Tensor,
    residual_veto: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    """Run one graph-native teacher recursion including per-time TT shapes.

    This is the complete TT lane: every time step fits the square-root TT,
    emits shape targets from those same cores, and carries its normalized
    marginal.  It still owns no particle likelihood scalar and performs no
    Contract E particle correction.
    """

    cores = tf.convert_to_tensor(initial_cores)
    dtype = cores.dtype
    dot_cores = tf.convert_to_tensor(initial_dot_cores, dtype)
    if cores.shape.rank != 4 or cores.shape[1] is None:
        raise ValueError("teacher recursion requires a static padded core shape")
    padded_core_shape = cores.shape
    cores = tf.ensure_shape(cores, padded_core_shape)
    dot_cores = tf.ensure_shape(dot_cores, padded_core_shape)
    active_mask = tf.convert_to_tensor(active_mask, dtype)
    base_log_targets = tf.convert_to_tensor(base_log_targets, dtype)
    dot_base_log_targets = tf.convert_to_tensor(dot_base_log_targets, dtype)
    scale_shift_indices = tf.convert_to_tensor(scale_shift_indices, tf.int32)
    defensive_weights = tf.convert_to_tensor(defensive_weights, dtype)
    dot_defensive_weights = tf.convert_to_tensor(dot_defensive_weights, dtype)
    state_offset = tf.convert_to_tensor(state_offset, dtype)
    dot_state_offset = tf.convert_to_tensor(dot_state_offset, dtype)
    state_matrix = tf.convert_to_tensor(state_matrix, dtype)
    dot_state_matrix = tf.convert_to_tensor(dot_state_matrix, dtype)
    pair_indices = tf.convert_to_tensor(pair_indices, tf.int32)
    if base_log_targets.shape[1] is None or state_offset.shape[0] is None:
        raise ValueError("teacher recursion requires static fit/state dimensions")
    if query_basis_values.shape[1] != base_log_targets.shape[1]:
        raise ValueError("carried-marginal query rows must match teacher fit rows")
    if pair_indices.shape[0] is None:
        raise ValueError("teacher recursion requires a static pair set")
    time_count = tf.shape(base_log_targets)[0]
    row_count = int(base_log_targets.shape[1])
    state_dimension = int(state_offset.shape[0])
    pair_count = int(pair_indices.shape[0])
    previous_values = tf.ones([row_count], dtype)
    previous_tangent = tf.zeros_like(previous_values)
    marginal_values = tf.TensorArray(dtype, size=time_count, element_shape=[row_count])
    marginal_tangents = tf.TensorArray(dtype, size=time_count, element_shape=[row_count])
    normalizers = tf.TensorArray(dtype, size=time_count, element_shape=[])
    skew = tf.TensorArray(dtype, size=time_count, element_shape=[state_dimension])
    kurtosis = tf.TensorArray(dtype, size=time_count, element_shape=[state_dimension])
    co_skew = tf.TensorArray(dtype, size=time_count, element_shape=[pair_count])
    co_kurtosis = tf.TensorArray(dtype, size=time_count, element_shape=[pair_count])
    dot_skew = tf.TensorArray(dtype, size=time_count, element_shape=[state_dimension])
    dot_kurtosis = tf.TensorArray(dtype, size=time_count, element_shape=[state_dimension])
    dot_co_skew = tf.TensorArray(dtype, size=time_count, element_shape=[pair_count])
    dot_co_kurtosis = tf.TensorArray(dtype, size=time_count, element_shape=[pair_count])

    def cond(index, *_):
        return index < time_count

    def body(
        index,
        cores,
        dot_cores,
        previous_values,
        previous_tangent,
        marginal_values,
        marginal_tangents,
        normalizers,
        skew,
        kurtosis,
        co_skew,
        co_kurtosis,
        dot_skew,
        dot_kurtosis,
        dot_co_skew,
        dot_co_kurtosis,
        valid,
    ):
        log_target = base_log_targets[index]
        dot_log_target = dot_base_log_targets[index]
        has_previous = index > 0
        safe_previous = tf.maximum(previous_values, tf.constant(1e-30, dtype))
        log_target = tf.where(
            has_previous, log_target + tf.math.log(safe_previous), log_target
        )
        dot_log_target = tf.where(
            has_previous,
            dot_log_target + previous_tangent / safe_previous,
            dot_log_target,
        )
        shift_index = scale_shift_indices[index]
        shift = log_target[shift_index]
        dot_shift = dot_log_target[shift_index]
        target = tf.exp(0.5 * (log_target - shift))
        dot_target = 0.5 * target * (dot_log_target - dot_shift)
        tau_scale = tf.exp(-shift)
        tau = tau_scale * defensive_weights[index]
        dot_tau = tau_scale * (
            dot_defensive_weights[index] - defensive_weights[index] * dot_shift
        )
        cores, dot_cores, _, fit_valid = padded_fixed_als_value_jvp_xla(
            basis_values,
            active_mask,
            schedule,
            target,
            dot_target,
            weights,
            dot_weights,
            cores,
            dot_cores,
            ridge,
            column_scale_floor,
            condition_number_veto,
            residual_veto,
        )
        cores = tf.ensure_shape(cores, padded_core_shape)
        dot_cores = tf.ensure_shape(dot_cores, padded_core_shape)
        values, tangent, normalizer, _ = padded_squared_tt_normalized_marginal_jvp_xla(
            cores,
            dot_cores,
            query_basis_values,
            keep_mask,
            mass_operators,
            tau,
            dot_tau,
            defensive_marginal_values[index],
            dot_defensive_marginal_values[index],
            defensive_mass,
            dot_defensive_mass,
        )
        shape_values = padded_squared_tt_shape_targets_jvp_xla(
            cores,
            dot_cores,
            operator_powers,
            defensive_power_moments,
            state_offset,
            dot_state_offset,
            state_matrix,
            dot_state_matrix,
            pair_indices,
            tau,
            dot_tau,
            defensive_mass,
        )
        shape_finite = tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(shape_values[0], [-1]),
                        tf.reshape(shape_values[1], [-1]),
                        tf.reshape(shape_values[2], [-1]),
                        tf.reshape(shape_values[3], [-1]),
                        tf.reshape(shape_values[4], [-1]),
                        tf.reshape(shape_values[5], [-1]),
                        tf.reshape(shape_values[6], [-1]),
                        tf.reshape(shape_values[7], [-1]),
                    ],
                    axis=0,
                )
            )
        )
        valid = (
            valid
            & fit_valid
            & tf.reduce_all(tf.math.is_finite(values))
            & tf.reduce_all(tf.math.is_finite(tangent))
            & shape_finite
        )
        marginal_values = marginal_values.write(index, values)
        marginal_tangents = marginal_tangents.write(index, tangent)
        normalizers = normalizers.write(index, normalizer)
        skew = skew.write(index, shape_values[0])
        kurtosis = kurtosis.write(index, shape_values[1])
        co_skew = co_skew.write(index, shape_values[2])
        co_kurtosis = co_kurtosis.write(index, shape_values[3])
        dot_skew = dot_skew.write(index, shape_values[4])
        dot_kurtosis = dot_kurtosis.write(index, shape_values[5])
        dot_co_skew = dot_co_skew.write(index, shape_values[6])
        dot_co_kurtosis = dot_co_kurtosis.write(index, shape_values[7])
        return (
            index + 1,
            cores,
            dot_cores,
            values,
            tangent,
            marginal_values,
            marginal_tangents,
            normalizers,
            skew,
            kurtosis,
            co_skew,
            co_kurtosis,
            dot_skew,
            dot_kurtosis,
            dot_co_skew,
            dot_co_kurtosis,
            valid,
        )

    outputs = tf.while_loop(
        cond,
        body,
        (
            tf.zeros([], tf.int32),
            cores * active_mask,
            dot_cores * active_mask,
            previous_values,
            previous_tangent,
            marginal_values,
            marginal_tangents,
            normalizers,
            skew,
            kurtosis,
            co_skew,
            co_kurtosis,
            dot_skew,
            dot_kurtosis,
            dot_co_skew,
            dot_co_kurtosis,
            tf.constant(True),
        ),
        parallel_iterations=1,
    )
    valid = outputs[-1]
    poison = tf.constant(float("nan"), dtype)
    final_cores = tf.where(valid, outputs[1], tf.fill(tf.shape(outputs[1]), poison))
    final_dot_cores = tf.where(
        valid, outputs[2], tf.fill(tf.shape(outputs[2]), poison)
    )
    return (
        final_cores,
        final_dot_cores,
        outputs[5].stack(),
        outputs[6].stack(),
        outputs[7].stack(),
        outputs[8].stack(),
        outputs[9].stack(),
        outputs[10].stack(),
        outputs[11].stack(),
        outputs[12].stack(),
        outputs[13].stack(),
        outputs[14].stack(),
        outputs[15].stack(),
        valid,
    )


def _condition_from_spd(matrix: tf.Tensor, *, square_root: bool = False) -> tf.Tensor:
    """Return the spectral SPD condition estimate used by the gates."""

    eigenvalues = tf.linalg.eigvalsh(matrix)
    finite = tf.reduce_all(tf.math.is_finite(eigenvalues))
    positive = tf.reduce_min(eigenvalues) > 0.0
    ratio = tf.reduce_max(eigenvalues) / tf.reduce_min(eigenvalues)
    value = tf.math.sqrt(ratio) if square_root else ratio
    return tf.where(finite & positive, value, tf.constant(float("inf"), matrix.dtype))


@tf.function(jit_compile=True, reduce_retracing=True)
def padded_fixed_als_value_jvp_xla(
    basis_values: tf.Tensor,
    active_mask: tf.Tensor,
    schedule: tf.Tensor,
    target_values: tf.Tensor,
    dot_target_values: tf.Tensor,
    weights: tf.Tensor,
    dot_weights: tf.Tensor,
    initial_cores: tf.Tensor,
    initial_dot_cores: tf.Tensor,
    ridge: tf.Tensor,
    column_scale_floor: tf.Tensor,
    condition_number_veto: tf.Tensor,
    residual_veto: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Run fixed padded ALS and its manual total directional derivative.

    The return values are final cores, final core tangents, per-update
    diagnostics `[scaled_condition, unscaled_condition, value_solve_residual,
    jvp_solve_residual, fit_residual, fit_jvp_residual, finite, valid]`, and a
    scalar validity bit.  This is the score-bearing graph path; it does not use
    TensorFlow autodiff and does not compute a likelihood.
    """

    cores = tf.convert_to_tensor(initial_cores)
    dtype = cores.dtype
    if dtype not in (tf.float32, tf.float64):
        raise TypeError("padded fixed ALS requires float32 or float64")
    basis_values = tf.convert_to_tensor(basis_values, dtype)
    active_mask = tf.convert_to_tensor(active_mask, dtype)
    schedule = tf.convert_to_tensor(schedule, tf.int32)
    target_values = tf.convert_to_tensor(target_values, dtype)
    dot_target_values = tf.convert_to_tensor(dot_target_values, dtype)
    weights = tf.convert_to_tensor(weights, dtype)
    dot_weights = tf.convert_to_tensor(dot_weights, dtype)
    cores = cores * active_mask
    dot_cores = tf.convert_to_tensor(initial_dot_cores, dtype) * active_mask
    ridge = tf.convert_to_tensor(ridge, dtype)
    column_scale_floor = tf.convert_to_tensor(column_scale_floor, dtype)
    condition_number_veto = tf.convert_to_tensor(condition_number_veto, dtype)
    residual_veto = tf.convert_to_tensor(residual_veto, dtype)
    n_columns = tf.shape(cores)[1] * tf.shape(cores)[2] * tf.shape(cores)[3]
    diagnostics = tf.TensorArray(dtype, size=tf.shape(schedule)[0], element_shape=[8])
    valid = (
        tf.reduce_all(tf.math.is_finite(basis_values))
        & tf.reduce_all(tf.math.is_finite(target_values))
        & tf.reduce_all(tf.math.is_finite(dot_target_values))
        & tf.reduce_all(tf.math.is_finite(weights))
        & tf.reduce_all(tf.math.is_finite(dot_weights))
        & tf.reduce_all(tf.math.is_finite(cores))
        & tf.reduce_all(tf.math.is_finite(dot_cores))
        & tf.reduce_all(weights >= 0.0)
        & (tf.reduce_sum(weights) > 0.0)
        & (ridge > 0.0)
    )

    def cond(index, *_):
        return index < tf.shape(schedule)[0]

    def body(index, cores, dot_cores, diagnostics, valid):
        axis = schedule[index]
        design, dot_design = _design_and_jvp(
            cores, dot_cores, basis_values, active_mask, axis
        )
        weighted_design = design * weights[:, tf.newaxis]
        normal = tf.matmul(design, weighted_design, transpose_a=True) + ridge * tf.eye(
            n_columns, dtype=dtype
        )
        rhs = tf.linalg.matvec(design, weights * target_values, transpose_a=True)
        dot_normal = (
            tf.matmul(dot_design, design * weights[:, tf.newaxis], transpose_a=True)
            + tf.matmul(design, dot_design * weights[:, tf.newaxis], transpose_a=True)
            + tf.matmul(design, design * dot_weights[:, tf.newaxis], transpose_a=True)
        )
        dot_rhs = (
            tf.linalg.matvec(dot_design, weights * target_values, transpose_a=True)
            + tf.linalg.matvec(design, weights * dot_target_values, transpose_a=True)
            + tf.linalg.matvec(design, dot_weights * target_values, transpose_a=True)
        )
        raw_norms = tf.sqrt(tf.reduce_sum(weights[:, tf.newaxis] * tf.square(design), axis=0))
        scale_floor = tf.maximum(
            tf.sqrt(
                tf.constant(
                    _FLOAT64_EPS if dtype == tf.float64 else 1.1920928955078125e-7,
                    dtype,
                )
            )
            * tf.reduce_max(raw_norms),
            column_scale_floor,
        )
        scales = tf.maximum(raw_norms, scale_floor)
        scaled_design = design / scales[tf.newaxis, :]
        scaled_normal = tf.matmul(
            scaled_design, scaled_design * weights[:, tf.newaxis], transpose_a=True
        ) + tf.linalg.diag(ridge / tf.square(scales))
        scaled_rhs = tf.linalg.matvec(
            scaled_design, weights * target_values, transpose_a=True
        )
        scaled_cholesky = tf.linalg.cholesky(scaled_normal)
        normal_cholesky = tf.linalg.cholesky(normal)
        z = tf.linalg.cholesky_solve(scaled_cholesky, scaled_rhs[:, tf.newaxis])[:, 0]
        solution = z / scales
        dot_solution = tf.linalg.cholesky_solve(
            normal_cholesky,
            (dot_rhs - tf.linalg.matvec(dot_normal, solution))[:, tf.newaxis],
        )[:, 0]
        value_system_residual = tf.linalg.norm(tf.linalg.matvec(normal, solution) - rhs)
        jvp_system_residual = tf.linalg.norm(
            tf.linalg.matvec(normal, dot_solution)
            + tf.linalg.matvec(dot_normal, solution)
            - dot_rhs
        )
        fit_residual = tf.sqrt(
            tf.reduce_sum(weights * tf.square(tf.linalg.matvec(design, solution) - target_values))
            / tf.reduce_sum(weights)
        )
        fit_jvp_residual = tf.sqrt(
            tf.reduce_sum(
                weights
                * tf.square(
                    tf.linalg.matvec(design, dot_solution)
                    + tf.linalg.matvec(dot_design, solution)
                    - dot_target_values
                )
            )
            / tf.reduce_sum(weights)
        )
        scaled_condition = _condition_from_spd(scaled_normal, square_root=True)
        unscaled_condition = _condition_from_spd(normal, square_root=False)
        finite = tf.reduce_all(
            tf.math.is_finite(
                tf.stack(
                    [
                        scaled_condition,
                        unscaled_condition,
                        value_system_residual,
                        jvp_system_residual,
                        fit_residual,
                        fit_jvp_residual,
                    ]
                )
            )
        )
        update_valid = finite & (scaled_condition <= condition_number_veto) & (
            unscaled_condition <= condition_number_veto
        ) & (value_system_residual <= residual_veto) & (
            jvp_system_residual <= residual_veto
        )
        valid = valid & update_valid
        solution_core = tf.reshape(solution, tf.shape(cores[axis])) * active_mask[axis]
        dot_solution_core = tf.reshape(dot_solution, tf.shape(dot_cores[axis])) * active_mask[axis]
        core_indices = tf.reshape(tf.stack([axis]), [1, 1])
        cores = tf.tensor_scatter_nd_update(cores, core_indices, solution_core[tf.newaxis, ...])
        dot_cores = tf.tensor_scatter_nd_update(
            dot_cores, core_indices, dot_solution_core[tf.newaxis, ...]
        )
        diagnostics = diagnostics.write(
            index,
            tf.stack(
                [
                    scaled_condition,
                    unscaled_condition,
                    value_system_residual,
                    jvp_system_residual,
                    fit_residual,
                    fit_jvp_residual,
                    tf.cast(finite, dtype),
                    tf.cast(valid, dtype),
                ]
            ),
        )
        return index + 1, cores, dot_cores, diagnostics, valid

    _, cores, dot_cores, diagnostics, valid = tf.while_loop(
        cond,
        body,
        (tf.zeros([], tf.int32), cores, dot_cores, diagnostics, valid),
        parallel_iterations=1,
    )
    poison = tf.constant(float("nan"), dtype)
    cores = tf.where(valid, cores, tf.fill(tf.shape(cores), poison))
    dot_cores = tf.where(valid, dot_cores, tf.fill(tf.shape(dot_cores), poison))
    return cores, dot_cores, diagnostics.stack(), valid


__all__ = [
    "ROUTE_ID",
    "ROUTE_CLASSIFICATION",
    "PaddedALSSetup",
    "pad_tt_cores",
    "padded_fixed_als_value_jvp_xla",
    "padded_squared_tt_normalized_marginal_jvp_xla",
    "padded_squared_tt_affine_moments_jvp_xla",
    "padded_squared_tt_shape_targets_jvp_xla",
    "padded_fixed_teacher_recursion_marginal_xla",
    "padded_fixed_teacher_recursion_shape_xla",
]

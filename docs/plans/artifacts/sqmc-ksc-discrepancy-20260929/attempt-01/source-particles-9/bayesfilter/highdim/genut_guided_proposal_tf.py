"""Experimental exact/defensive guided proposals for the diagonal LGSSM.

The likelihood uses the exact importance correction. The score is the repository's
standard target-model backward-kernel filtering score; it is not obtained by
differentiating proposal sampling, importance weights, transport, or reset code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim.dual_cap_genut_primal_tf import dual_cap_genut_primal
from bayesfilter.highdim import ledh_contract_e_reset_tf as contract_e_reset
from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as standard_score
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.highdim.transport_chunk_policy import validate_transport_chunks


Tensor = tf.Tensor
_LOG_TWO_PI = math.log(2.0 * math.pi)


def _pairwise_squared_tile(query: Tensor, key: Tensor) -> Tensor:
    """Evaluate one dense-equivalent distance tile without retaining all pairs."""

    differences = query[:, None, :] - key[None, :, :]
    return tf.reduce_sum(tf.square(differences), axis=-1)


def _static_row_slice(value: Tensor, start: Tensor, size: int) -> Tensor:
    width = value.shape[1:]
    if any(item is None for item in width):
        raise ValueError("streaming row slices require static trailing dimensions")
    begin = tf.concat([[start], tf.zeros([len(width)], tf.int32)], axis=0)
    extent = tf.constant([size, *[int(item) for item in width]], tf.int32)
    result = tf.slice(value, begin, extent)
    return tf.ensure_shape(result, [size, *width])


def _streaming_pairwise_cost_scale(
    particles: Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> Tensor:
    particle_count = int(particles.shape[0])
    row_blocks = particle_count // row_chunk_size
    col_blocks = particle_count // col_chunk_size
    row_step = tf.constant(row_chunk_size, tf.int32)
    col_step = tf.constant(col_chunk_size, tf.int32)

    def row_body(row_index: Tensor, total: Tensor) -> tuple[Tensor, Tensor]:
        row_start = row_index * row_step
        query = _static_row_slice(particles, row_start, row_chunk_size)

        def col_body(col_index: Tensor, subtotal: Tensor) -> tuple[Tensor, Tensor]:
            col_start = col_index * col_step
            key = _static_row_slice(particles, col_start, col_chunk_size)
            return col_index + 1, subtotal + tf.reduce_sum(
                _pairwise_squared_tile(query, key)
            )

        _, subtotal = tf.while_loop(
            lambda col_index, _subtotal: col_index < col_blocks,
            col_body,
            (tf.zeros([], tf.int32), tf.zeros([], particles.dtype)),
            parallel_iterations=1,
        )
        return row_index + 1, total + subtotal

    _, total = tf.while_loop(
        lambda row_index, _total: row_index < row_blocks,
        row_body,
        (tf.zeros([], tf.int32), tf.zeros([], particles.dtype)),
        parallel_iterations=1,
    )
    count = tf.cast(particle_count, particles.dtype)
    return tf.maximum(
        total / tf.square(count), tf.constant(1.0e-3, particles.dtype)
    )


def _streaming_kernel_action(
    particles: Tensor,
    vector: Tensor,
    cost_scale: Tensor,
    epsilon: float,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
    transpose: bool,
) -> Tensor:
    particle_count = int(particles.shape[0])
    row_blocks = particle_count // row_chunk_size
    col_blocks = particle_count // col_chunk_size
    row_step = tf.constant(row_chunk_size, tf.int32)
    col_step = tf.constant(col_chunk_size, tf.int32)
    denominator = cost_scale * tf.cast(epsilon, particles.dtype)
    if transpose:
        output_blocks = tf.TensorArray(
            particles.dtype,
            size=col_blocks,
            element_shape=tf.TensorShape([col_chunk_size]),
        )

        def outer_body(
            col_index: Tensor, blocks: tf.TensorArray
        ) -> tuple[Tensor, tf.TensorArray]:
            col_start = col_index * col_step
            key = _static_row_slice(particles, col_start, col_chunk_size)

            def inner_body(
                row_index: Tensor, subtotal: Tensor
            ) -> tuple[Tensor, Tensor]:
                row_start = row_index * row_step
                query = _static_row_slice(particles, row_start, row_chunk_size)
                row_vector = _static_row_slice(
                    vector[:, None], row_start, row_chunk_size
                )[:, 0]
                kernel = tf.exp(-_pairwise_squared_tile(query, key) / denominator)
                return row_index + 1, subtotal + tf.linalg.matvec(
                    tf.transpose(kernel), row_vector
                )

            _, subtotal = tf.while_loop(
                lambda row_index, _subtotal: row_index < row_blocks,
                inner_body,
                (tf.zeros([], tf.int32), tf.zeros([col_chunk_size], particles.dtype)),
                parallel_iterations=1,
            )
            return col_index + 1, blocks.write(col_index, subtotal)

        _, output_blocks = tf.while_loop(
            lambda col_index, _blocks: col_index < col_blocks,
            outer_body,
            (tf.zeros([], tf.int32), output_blocks),
            parallel_iterations=1,
        )
    else:
        output_blocks = tf.TensorArray(
            particles.dtype,
            size=row_blocks,
            element_shape=tf.TensorShape([row_chunk_size]),
        )

        def outer_body(
            row_index: Tensor, blocks: tf.TensorArray
        ) -> tuple[Tensor, tf.TensorArray]:
            row_start = row_index * row_step
            query = _static_row_slice(particles, row_start, row_chunk_size)

            def inner_body(
                col_index: Tensor, subtotal: Tensor
            ) -> tuple[Tensor, Tensor]:
                col_start = col_index * col_step
                key = _static_row_slice(particles, col_start, col_chunk_size)
                col_vector = _static_row_slice(
                    vector[:, None], col_start, col_chunk_size
                )[:, 0]
                kernel = tf.exp(-_pairwise_squared_tile(query, key) / denominator)
                return col_index + 1, subtotal + tf.linalg.matvec(kernel, col_vector)

            _, subtotal = tf.while_loop(
                lambda col_index, _subtotal: col_index < col_blocks,
                inner_body,
                (tf.zeros([], tf.int32), tf.zeros([row_chunk_size], particles.dtype)),
                parallel_iterations=1,
            )
            return row_index + 1, blocks.write(row_index, subtotal)

        _, output_blocks = tf.while_loop(
            lambda row_index, _blocks: row_index < row_blocks,
            outer_body,
            (tf.zeros([], tf.int32), output_blocks),
            parallel_iterations=1,
        )
    return tf.reshape(output_blocks.stack(), [particle_count])


def _streaming_sinkhorn_barycentric_value(
    particles: Tensor,
    weights: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, Tensor]:
    particle_count = int(particles.shape[0])
    state_dimension = int(particles.shape[1])
    row_blocks = particle_count // row_chunk_size
    col_blocks = particle_count // col_chunk_size
    # The active exact-divisor policy selects K=N for N <= 3000. Reuse the
    # dense arithmetic in that one-block case so the streaming label does not
    # introduce an avoidable XLA reduction-order change into the trust reset.
    if row_blocks == 1 and col_blocks == 1:
        return _dense_sinkhorn_barycentric_value(
            particles,
            weights,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
        )
    row_step = tf.constant(row_chunk_size, tf.int32)
    col_step = tf.constant(col_chunk_size, tf.int32)
    cost_scale = _streaming_pairwise_cost_scale(
        particles,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    uniform = tf.fill(
        [particle_count],
        tf.cast(1.0 / float(particle_count), particles.dtype),
    )
    uniform_value = tf.cast(1.0 / float(particle_count), particles.dtype)
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    tiny = tf.cast(1.0e-7, particles.dtype)

    def sinkhorn_body(
        index: Tensor, left_value: Tensor, right_value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        left_denominator = _streaming_kernel_action(
            particles,
            right_value,
            cost_scale,
            epsilon,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            transpose=False,
        )
        left_new = uniform / (left_denominator + tiny)
        right_denominator = _streaming_kernel_action(
            particles,
            left_new,
            cost_scale,
            epsilon,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            transpose=True,
        )
        right_new = weights / (right_denominator + tiny)
        return index + 1, left_new, right_new

    _, left, right = tf.while_loop(
        lambda index, *_: index
        < tf.cast(sinkhorn_steps + balance_steps, tf.int32),
        sinkhorn_body,
        (tf.zeros([], tf.int32), left, right),
        parallel_iterations=1,
    )
    denominator = cost_scale * tf.cast(epsilon, particles.dtype)
    row_mass_blocks = tf.TensorArray(
        particles.dtype,
        size=row_blocks,
        element_shape=tf.TensorShape([row_chunk_size]),
    )
    numerator_blocks = tf.TensorArray(
        particles.dtype,
        size=row_blocks,
        element_shape=tf.TensorShape([row_chunk_size, state_dimension]),
    )
    def row_body(
        row_index: Tensor,
        mass_blocks: tf.TensorArray,
        carried_blocks: tf.TensorArray,
    ) -> tuple[Tensor, tf.TensorArray, tf.TensorArray]:
        row_start = row_index * row_step
        query = _static_row_slice(particles, row_start, row_chunk_size)
        left_block = _static_row_slice(left[:, None], row_start, row_chunk_size)[:, 0]

        def col_body(
            col_index: Tensor,
            mass: Tensor,
            numerator: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            col_start = col_index * col_step
            key = _static_row_slice(particles, col_start, col_chunk_size)
            right_block = _static_row_slice(
                right[:, None], col_start, col_chunk_size
            )[:, 0]
            kernel = tf.exp(-_pairwise_squared_tile(query, key) / denominator)
            coupling = left_block[:, None] * kernel * right_block[None, :]
            return (
                col_index + 1,
                mass + tf.reduce_sum(coupling, axis=1),
                numerator + tf.linalg.matmul(coupling, key),
            )

        _, mass, numerator = tf.while_loop(
            lambda col_index, *_: col_index < col_blocks,
            col_body,
            (
                tf.zeros([], tf.int32),
                tf.zeros([row_chunk_size], particles.dtype),
                tf.zeros([row_chunk_size, state_dimension], particles.dtype),
            ),
            parallel_iterations=1,
        )
        return (
            row_index + 1,
            mass_blocks.write(row_index, mass),
            carried_blocks.write(row_index, numerator),
        )

    _, row_mass_blocks, numerator_blocks = tf.while_loop(
        lambda row_index, *_: row_index < row_blocks,
        row_body,
        (
            tf.zeros([], tf.int32),
            row_mass_blocks,
            numerator_blocks,
        ),
        parallel_iterations=1,
    )
    row_mass = tf.reshape(row_mass_blocks.stack(), [particle_count])
    numerator = tf.reshape(
        numerator_blocks.stack(), [particle_count, state_dimension]
    )
    barycentric = numerator / row_mass[:, None]
    column_mass_blocks = tf.TensorArray(
        particles.dtype,
        size=col_blocks,
        element_shape=tf.TensorShape([2, col_chunk_size]),
    )

    def quotient_col_body(
        col_index: Tensor, blocks: tf.TensorArray
    ) -> tuple[Tensor, tf.TensorArray]:
        col_start = col_index * col_step
        key = _static_row_slice(particles, col_start, col_chunk_size)
        right_block = _static_row_slice(right[:, None], col_start, col_chunk_size)[:, 0]

        def quotient_row_body(
            row_index: Tensor, raw_subtotal: Tensor, quotient_subtotal: Tensor
        ) -> tuple[Tensor, Tensor, Tensor]:
            row_start = row_index * row_step
            query = _static_row_slice(particles, row_start, row_chunk_size)
            left_block = _static_row_slice(
                left[:, None], row_start, row_chunk_size
            )[:, 0]
            mass_block = _static_row_slice(
                row_mass[:, None], row_start, row_chunk_size
            )[:, 0]
            kernel = tf.exp(-_pairwise_squared_tile(query, key) / denominator)
            coupling = left_block[:, None] * kernel * right_block[None, :]
            quotient = uniform_value * coupling / mass_block[:, None]
            return (
                row_index + 1,
                raw_subtotal + tf.reduce_sum(coupling, axis=0),
                quotient_subtotal + tf.reduce_sum(quotient, axis=0),
            )

        _, raw_subtotal, quotient_subtotal = tf.while_loop(
            lambda row_index, _raw, _quotient: row_index < row_blocks,
            quotient_row_body,
            (
                tf.zeros([], tf.int32),
                tf.zeros([col_chunk_size], particles.dtype),
                tf.zeros([col_chunk_size], particles.dtype),
            ),
            parallel_iterations=1,
        )
        return col_index + 1, blocks.write(
            col_index, tf.stack([raw_subtotal, quotient_subtotal], axis=0)
        )

    _, column_mass_blocks = tf.while_loop(
        lambda col_index, _blocks: col_index < col_blocks,
        quotient_col_body,
        (tf.zeros([], tf.int32), column_mass_blocks),
        parallel_iterations=1,
    )
    stacked_column_mass = column_mass_blocks.stack()
    raw_column_mass = tf.reshape(stacked_column_mass[:, 0, :], [particle_count])
    quotient_column_mass = tf.reshape(
        stacked_column_mass[:, 1, :], [particle_count]
    )
    quotient_column_residual = quotient_column_mass - weights
    return {
        "barycentric": barycentric,
        "row_mass": row_mass,
        "raw_column_mass": raw_column_mass,
        "quotient_column_residual": quotient_column_residual,
        "cost_scale": cost_scale,
    }


def _dense_sinkhorn_barycentric_value(
    particles: Tensor,
    weights: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
) -> dict[str, Tensor]:
    deltas = particles[:, None, :] - particles[None, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=-1)
    cost_scale = tf.maximum(
        tf.reduce_mean(cost), tf.constant(1.0e-3, particles.dtype)
    )
    kernel = tf.exp(-cost / (cost_scale * tf.cast(epsilon, particles.dtype)))
    count = tf.shape(particles)[0]
    uniform = tf.fill(
        [count], tf.cast(1.0, particles.dtype) / tf.cast(count, particles.dtype)
    )
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    tiny = tf.cast(1.0e-7, particles.dtype)

    def sinkhorn_body(index, left_value, right_value):
        left_new = uniform / (tf.linalg.matvec(kernel, right_value) + tiny)
        right_new = weights / (
            tf.linalg.matvec(tf.transpose(kernel), left_new) + tiny
        )
        return index + 1, left_new, right_new

    _, left, right = tf.while_loop(
        lambda index, *_: index
        < tf.cast(sinkhorn_steps + balance_steps, tf.int32),
        sinkhorn_body,
        (tf.zeros([], tf.int32), left, right),
        parallel_iterations=1,
    )
    coupling = left[:, None] * kernel * right[None, :]
    row_mass = tf.reduce_sum(coupling, axis=1)
    barycentric = (coupling @ particles) / row_mass[:, None]
    raw_column_mass = tf.reduce_sum(coupling, axis=0)
    quotient_column_mass = tf.reduce_sum(
        uniform[:, None] * coupling / row_mass[:, None], axis=0
    )
    return {
        "barycentric": barycentric,
        "row_mass": row_mass,
        "raw_column_mass": raw_column_mass,
        "quotient_column_residual": quotient_column_mass - weights,
        "cost_scale": cost_scale,
    }


@dataclass(frozen=True)
class LGSSMGuidedProposalSpec:
    """Repository-owned Phase 1 proposal settings for the three-state LGSSM."""

    observation_matrix: Tensor
    rho: float
    transition_before_first_observation: bool = True
    proposal_family: str = "defensive_exact_conditional_lgssm_v1"

    def __post_init__(self) -> None:
        matrix = tf.convert_to_tensor(self.observation_matrix)
        if matrix.shape != (3, 3):
            raise ValueError("observation_matrix must have shape (3, 3)")
        if not math.isfinite(self.rho) or not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be finite and lie in [0, 1]")
        if not self.transition_before_first_observation:
            raise ValueError(
                "Phase 1 LGSSM guidance requires transition before first observation"
            )
        if self.proposal_family != "defensive_exact_conditional_lgssm_v1":
            raise ValueError("unsupported proposal family")
        expected_matrix = standard_score.lgssm._observation_matrix(matrix.dtype)
        tf.debugging.assert_near(
            matrix,
            expected_matrix,
            message="Phase 1 score provider requires the canonical LGSSM observation matrix",
        )


def _validate_theta(theta: Tensor) -> None:
    if theta.shape != (5,):
        raise ValueError("theta must have shape (5,)")


def _normal_log_density_from_cholesky(
    values: Tensor, mean: Tensor, cholesky: Tensor
) -> Tensor:
    """Evaluate row-wise multivariate normal log densities."""

    values = tf.convert_to_tensor(values)
    mean = tf.convert_to_tensor(mean, dtype=values.dtype)
    cholesky = tf.convert_to_tensor(cholesky, dtype=values.dtype)
    residual = values - mean
    standardized = tf.linalg.triangular_solve(
        cholesky, tf.transpose(residual), lower=True
    )
    dimension = tf.cast(tf.shape(values)[-1], values.dtype)
    log_determinant = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky)))
    return -0.5 * (
        dimension * tf.cast(_LOG_TWO_PI, values.dtype)
        + 2.0 * log_determinant
        + tf.reduce_sum(tf.square(standardized), axis=0)
    )


def defensive_mixture_log_density(
    rho: Tensor,
    base_log_density: Tensor,
    guided_log_density: Tensor,
) -> Tensor:
    """Evaluate the full defensive mixture density for an interior mixture weight."""

    rho = tf.convert_to_tensor(rho, dtype=base_log_density.dtype)
    tf.debugging.assert_all_finite(rho, "rho must be finite")
    tf.debugging.assert_greater(rho, tf.zeros([], rho.dtype))
    tf.debugging.assert_less(rho, tf.ones([], rho.dtype))
    return tf.reduce_logsumexp(
        tf.stack(
            [
                tf.math.log(rho) + base_log_density,
                tf.math.log1p(-rho) + guided_log_density,
            ],
            axis=0,
        ),
        axis=0,
    )


def exact_lgssm_conditional_moments(
    theta: Tensor,
    parents: Tensor,
    observation: Tensor,
    observation_matrix: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return exact ``p(x_t | x_(t-1), y_t)`` means and covariance factor."""

    theta = tf.convert_to_tensor(theta)
    _validate_theta(theta)
    parents = tf.convert_to_tensor(parents, dtype=theta.dtype)
    observation = tf.convert_to_tensor(observation, dtype=theta.dtype)
    matrix = tf.convert_to_tensor(observation_matrix, dtype=theta.dtype)
    if parents.shape.rank != 2 or parents.shape[-1] != 3:
        raise ValueError("parents must have shape (N, 3)")
    if observation.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("observation and observation_matrix must have shapes (3,) and (3, 3)")

    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    tf.debugging.assert_positive(q_scale)
    tf.debugging.assert_positive(r_scale)
    q_precision = tf.math.reciprocal(tf.square(q_scale))
    r_precision = tf.math.reciprocal(tf.square(r_scale))
    precision = q_precision * tf.eye(3, dtype=theta.dtype)
    precision += r_precision * tf.linalg.matmul(matrix, matrix, transpose_a=True)
    covariance_cholesky = tf.linalg.cholesky(precision)
    covariance = tf.linalg.cholesky_solve(
        covariance_cholesky, tf.eye(3, dtype=theta.dtype)
    )
    transition_mean = parents * phi[None, :]
    information = q_precision * transition_mean
    information += r_precision * tf.broadcast_to(
        tf.linalg.matvec(matrix, observation, transpose_a=True)[None, :],
        tf.shape(transition_mean),
    )
    conditional_mean = tf.linalg.matmul(information, covariance, transpose_b=True)
    return conditional_mean, tf.linalg.cholesky(covariance)


def _transition_log_density(
    theta: Tensor, parents: Tensor, children: Tensor
) -> Tensor:
    transition_mean = parents * theta[:3][None, :]
    factor = theta[3] * tf.eye(3, dtype=theta.dtype)
    return _normal_log_density_from_cholesky(children, transition_mean, factor)


def _observation_log_density(
    theta: Tensor, children: Tensor, observation: Tensor, observation_matrix: Tensor
) -> Tensor:
    predicted = tf.linalg.matmul(children, observation_matrix, transpose_b=True)
    observation_rows = tf.broadcast_to(observation[None, :], tf.shape(predicted))
    factor = theta[4] * tf.eye(3, dtype=theta.dtype)
    return _normal_log_density_from_cholesky(
        observation_rows, predicted, factor
    )


def guided_lgssm_step(
    spec: LGSSMGuidedProposalSpec,
    theta: Tensor,
    parents: Tensor,
    previous_weights: Tensor,
    observation: Tensor,
    base_noise: Tensor,
    component_uniforms: Tensor,
) -> dict[str, Tensor]:
    """Sample one corrected defensive proposal step and normalize its weights."""

    theta = tf.convert_to_tensor(theta)
    _validate_theta(theta)
    parents = tf.convert_to_tensor(parents, dtype=theta.dtype)
    previous_weights = tf.convert_to_tensor(previous_weights, dtype=theta.dtype)
    observation = tf.convert_to_tensor(observation, dtype=theta.dtype)
    base_noise = tf.convert_to_tensor(base_noise, dtype=theta.dtype)
    component_uniforms = tf.convert_to_tensor(component_uniforms, dtype=theta.dtype)
    matrix = tf.convert_to_tensor(spec.observation_matrix, dtype=theta.dtype)
    if parents.shape.rank != 2 or parents.shape[-1] != 3:
        raise ValueError("parents must have shape (N, 3)")
    if base_noise.shape != parents.shape:
        raise ValueError("base_noise must match parents")
    if previous_weights.shape.rank != 1 or component_uniforms.shape.rank != 1:
        raise ValueError("weights and component_uniforms must have shape (N,)")
    if (
        parents.shape[0] is not None
        and previous_weights.shape[0] is not None
        and parents.shape[0] != previous_weights.shape[0]
    ):
        raise ValueError("weights must match the particle count")
    if (
        parents.shape[0] is not None
        and component_uniforms.shape[0] is not None
        and parents.shape[0] != component_uniforms.shape[0]
    ):
        raise ValueError("component_uniforms must match the particle count")

    tf.debugging.assert_all_finite(theta, "theta must be finite")
    tf.debugging.assert_all_finite(parents, "parents must be finite")
    tf.debugging.assert_all_finite(base_noise, "base noise must be finite")
    tf.debugging.assert_positive(theta[3])
    tf.debugging.assert_positive(theta[4])
    tf.debugging.assert_non_negative(previous_weights)
    tf.debugging.assert_near(
        tf.reduce_sum(previous_weights), tf.ones([], theta.dtype)
    )
    tf.debugging.assert_greater_equal(component_uniforms, tf.zeros_like(component_uniforms))
    tf.debugging.assert_less(component_uniforms, tf.ones_like(component_uniforms))

    transition_mean = parents * theta[:3][None, :]
    transition_children = transition_mean + theta[3] * base_noise
    conditional_mean, conditional_cholesky = exact_lgssm_conditional_moments(
        theta, parents, observation, matrix
    )
    conditional_children = conditional_mean + tf.linalg.matmul(
        base_noise, conditional_cholesky, transpose_b=True
    )
    if spec.rho == 1.0:
        children = transition_children
    elif spec.rho == 0.0:
        children = conditional_children
    else:
        use_transition = component_uniforms < tf.cast(spec.rho, theta.dtype)
        children = tf.where(use_transition[:, None], transition_children, conditional_children)

    transition_log_density = _transition_log_density(theta, parents, children)
    observation_log_density = _observation_log_density(
        theta, children, observation, matrix
    )
    conditional_log_density = _normal_log_density_from_cholesky(
        children, conditional_mean, conditional_cholesky
    )
    if spec.rho == 1.0:
        proposal_log_density = transition_log_density
        log_weight_correction = tf.zeros_like(transition_log_density)
    elif spec.rho == 0.0:
        proposal_log_density = conditional_log_density
        log_weight_correction = transition_log_density - conditional_log_density
    else:
        proposal_log_density = defensive_mixture_log_density(
            tf.cast(spec.rho, theta.dtype),
            transition_log_density,
            conditional_log_density,
        )
        log_weight_correction = transition_log_density - proposal_log_density
    log_unnormalized_weights = (
        tf.math.log(previous_weights)
        + observation_log_density
        + log_weight_correction
    )
    increment = tf.reduce_logsumexp(log_unnormalized_weights)
    normalized_weights = tf.exp(log_unnormalized_weights - increment)
    centered_log_weights = log_unnormalized_weights - tf.reduce_mean(
        log_unnormalized_weights
    )
    return {
        "particles": children,
        "normalized_weights": normalized_weights,
        "increment": increment,
        "transition_log_density": transition_log_density,
        "observation_log_density": observation_log_density,
        "conditional_log_density": conditional_log_density,
        "proposal_log_density": proposal_log_density,
        "log_unnormalized_weights": log_unnormalized_weights,
        "ess": tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights))),
        "maximum_normalized_weight": tf.reduce_max(normalized_weights),
        "log_weight_variance": tf.reduce_mean(tf.square(centered_log_weights)),
    }


def _restore_cloud_primal(
    particles: Tensor,
    weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
    reset_policy: str = "contract_e",
    dual_cap_enabled: bool = False,
    dual_cap_diagonal_steps: int = 4,
    dual_cap_diagonal_strength: float = 0.2,
    dual_cap_pairwise_steps: int = 4,
    dual_cap_pairwise_strength: float = 0.02,
    dual_cap_pairwise_particle_rms_cap: float = 2.0,
    dual_cap_coordinate_cap: float = 0.98,
    dual_cap_coordinate_cap_power: int = 8,
    trust_region_enabled: bool = False,
    trust_region_lm_damping: float = 1.0e-2,
    trust_region_lm_scale_floor: float = 1.0e-4,
    trust_region_radius: float = 0.5,
    transport_plan_mode: str = "dense",
    transport_row_chunk_size: int | None = None,
    transport_col_chunk_size: int | None = None,
    marginal_tolerance: float = 1.0e-4,
) -> dict[str, Tensor]:
    """Apply the existing Sinkhorn/Contract-E reset without derivative work."""

    if (
        epsilon <= 0.0
        or sinkhorn_steps <= 0
        or balance_steps < 0
        or marginal_tolerance <= 0.0
    ):
        raise ValueError("epsilon and Sinkhorn counts must be valid")
    if reset_policy not in ("contract_e", "ot_only", "none"):
        raise ValueError(f"unsupported reset policy: {reset_policy}")
    if transport_plan_mode not in ("dense", "streaming"):
        raise ValueError("transport_plan_mode must be 'dense' or 'streaming'")
    particle_count = particles.shape[0]
    if particle_count is None:
        raise ValueError("reset transport requires a static particle count")
    if transport_plan_mode == "streaming":
        if transport_row_chunk_size is None or transport_col_chunk_size is None:
            raise ValueError("streaming transport requires both chunk sizes")
        chunk_selection = validate_transport_chunks(
            int(particle_count),
            row_chunk_size=transport_row_chunk_size,
            col_chunk_size=transport_col_chunk_size,
        )
    else:
        if transport_row_chunk_size is not None or transport_col_chunk_size is not None:
            raise ValueError("dense transport does not accept chunk sizes")
        chunk_selection = None
    if dual_cap_enabled and reset_policy != "contract_e":
        raise ValueError("dual cap requires the Contract-E reset")
    if trust_region_enabled and not dual_cap_enabled:
        raise ValueError("trust-region dual cap requires the dual-cap reset")
    if trust_region_enabled and (
        trust_region_lm_damping <= 0.0
        or trust_region_lm_scale_floor <= 0.0
        or trust_region_radius <= 0.0
    ):
        raise ValueError("trust-region controls must be positive")
    if reset_policy == "none":
        return {
            "particles": particles,
            "mean_residual": tf.zeros([], particles.dtype),
            "minimum_gap_eigenvalue": tf.zeros([], particles.dtype),
            "gap_condition_proxy": tf.zeros([], particles.dtype),
            "target_condition_proxy": tf.zeros([], particles.dtype),
            "injected_condition_proxy": tf.zeros([], particles.dtype),
            "affine_norm": tf.zeros([], particles.dtype),
            "output_covariance": tf.zeros([tf.shape(particles)[1], tf.shape(particles)[1]], particles.dtype),
            "gap_valid": tf.constant(True),
            "reset_valid": tf.reduce_all(tf.math.is_finite(particles)) & tf.reduce_all(tf.math.is_finite(weights)),
            "minimum_row_mass": tf.ones([], particles.dtype),
            "maximum_raw_row_residual": tf.zeros([], particles.dtype),
            "maximum_raw_column_residual": tf.zeros([], particles.dtype),
            "maximum_post_quotient_column_residual": tf.zeros([], particles.dtype),
            "post_quotient_column_tv_error": tf.zeros([], particles.dtype),
            "marginal_valid": tf.constant(True),
            "dual_cap_valid": tf.constant(True),
            "dual_cap_mean_residual": tf.zeros([], particles.dtype),
            "dual_cap_covariance_residual": tf.zeros([], particles.dtype),
            "maximum_pairwise_pre_cap_particle_rms": tf.zeros([], particles.dtype),
            "maximum_pairwise_post_cap_particle_rms": tf.zeros([], particles.dtype),
            "minimum_pairwise_particle_cap_scale": tf.ones([], particles.dtype),
            "maximum_coordinatewise_pre_cap_absolute": tf.zeros([], particles.dtype),
            "maximum_coordinatewise_post_cap_absolute": tf.zeros([], particles.dtype),
            "mean_coordinatewise_cap_displacement": tf.zeros([], particles.dtype),
            "fraction_coordinatewise_cap_active": tf.zeros([], particles.dtype),
            "minimum_coordinatewise_cap_derivative": tf.ones([], particles.dtype),
            "reset_route_id": tf.constant(0, tf.int32),
            "maximum_diagonal_scaled_system_condition": tf.zeros([], particles.dtype),
            "maximum_diagonal_pre_cap_particle_rms": tf.zeros([], particles.dtype),
            "maximum_diagonal_post_cap_particle_rms": tf.zeros([], particles.dtype),
            "transport_plan_id": tf.constant(0, tf.int32),
            "transport_row_chunk_size": tf.constant(int(particle_count), tf.int32),
            "transport_col_chunk_size": tf.constant(int(particle_count), tf.int32),
        }
    count = tf.shape(particles)[0]
    uniform = tf.fill(
        [count], tf.cast(1.0, particles.dtype) / tf.cast(count, particles.dtype)
    )
    tiny = tf.cast(1.0e-7, particles.dtype)
    if transport_plan_mode == "streaming":
        streamed = _streaming_sinkhorn_barycentric_value(
            particles,
            weights,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            row_chunk_size=chunk_selection.row_chunk_size,
            col_chunk_size=chunk_selection.col_chunk_size,
        )
        row_mass = streamed["row_mass"]
        barycentric = streamed["barycentric"]
        raw_column_mass = streamed["raw_column_mass"]
        quotient_column_residual = streamed["quotient_column_residual"]
    else:
        dense = _dense_sinkhorn_barycentric_value(
            particles,
            weights,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
        )
        row_mass = dense["row_mass"]
        barycentric = dense["barycentric"]
        raw_column_mass = dense["raw_column_mass"]
        quotient_column_residual = dense["quotient_column_residual"]
    post_quotient_column_tv_error = 0.5 * tf.reduce_sum(
        tf.abs(quotient_column_residual)
    )
    marginal_valid = (
        tf.reduce_all(tf.math.is_finite(row_mass))
        & tf.reduce_all(row_mass > tiny)
        & tf.reduce_all(tf.math.is_finite(barycentric))
        & (
            post_quotient_column_tv_error
            <= tf.cast(marginal_tolerance, particles.dtype)
        )
    )

    target_mean = tf.reduce_sum(weights[:, None] * particles, axis=0)
    centered_source = particles - target_mean[None, :]
    target_covariance = tf.einsum(
        "n,ni,nj->ij", weights, centered_source, centered_source
    )
    centered_transport = barycentric - tf.reduce_mean(
        barycentric, axis=0, keepdims=True
    )
    transport_covariance = tf.einsum(
        "ni,nj->ij", centered_transport, centered_transport
    ) / tf.cast(count, particles.dtype)
    covariance_gap = 0.5 * (
        target_covariance
        - transport_covariance
        + tf.transpose(target_covariance - transport_covariance)
    )
    minimum_gap_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(covariance_gap))
    gap_valid = tf.math.is_finite(minimum_gap_eigenvalue) & (
        minimum_gap_eigenvalue + tf.cast(ridge, particles.dtype) > 0.0
    )
    pre_reset_valid = marginal_valid & gap_valid
    if reset_policy == "none":
        output_particles = particles
        reset_valid = (
            tf.reduce_all(tf.math.is_finite(output_particles))
            & tf.reduce_all(tf.math.is_finite(weights))
            & tf.reduce_all(weights >= 0.0)
        )
        mean_residual = tf.zeros([], particles.dtype)
        minimum_gap_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(covariance_gap))
        gap_condition_proxy = tf.zeros([], particles.dtype)
        target_condition_proxy = tf.zeros([], particles.dtype)
        injected_condition_proxy = tf.zeros([], particles.dtype)
        affine_norm = tf.zeros([], particles.dtype)
        output_covariance = target_cov
    else:
        safe_barycentric = tf.where(
            marginal_valid,
            barycentric,
            tf.broadcast_to(target_mean[None, :], tf.shape(barycentric)),
        )
        if reset_policy == "ot_only":
            output_particles = safe_barycentric
            reset_valid = marginal_valid & tf.reduce_all(
                tf.math.is_finite(output_particles)
            )
            mean_residual = tf.reduce_max(
                tf.abs(tf.reduce_mean(output_particles, axis=0) - target_mean)
            )
            minimum_gap_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(covariance_gap))
            gap_condition_proxy = tf.zeros([], particles.dtype)
            target_condition_proxy = tf.zeros([], particles.dtype)
            injected_condition_proxy = tf.zeros([], particles.dtype)
            affine_norm = tf.zeros([], particles.dtype)
            output_covariance = tf.einsum(
                "ni,nj->ij",
                output_particles - tf.reduce_mean(output_particles, axis=0),
                output_particles - tf.reduce_mean(output_particles, axis=0),
            ) / tf.cast(count, particles.dtype)
        else:
            forward = contract_e_reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
                particles[None, :, :],
                weights[None, :],
                safe_barycentric[None, :, :],
                design[None, :, :],
                tf.constant([ridge], particles.dtype),
            )
            output_particles = forward["particles"][0]
            reset_valid = (
                pre_reset_valid
                & forward["finite"][0]
                & forward["factor_diagonal_positive"][0]
            )
            mean_residual = tf.reduce_max(tf.abs(forward["mean_residual"]))
            minimum_gap_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(covariance_gap))
            gap_condition_proxy = forward["gap_condition_proxy"][0]
            target_condition_proxy = forward["target_condition_proxy"][0]
            injected_condition_proxy = forward["injected_condition_proxy"][0]
            affine_norm = tf.linalg.norm(forward["affine"][0])
            output_covariance = forward["output_cov"][0]
    dual_cap_valid = tf.constant(True)
    dual_cap_mean_residual = tf.zeros([], particles.dtype)
    dual_cap_covariance_residual = tf.zeros([], particles.dtype)
    maximum_pairwise_pre_cap_particle_rms = tf.zeros([], particles.dtype)
    maximum_pairwise_post_cap_particle_rms = tf.zeros([], particles.dtype)
    minimum_pairwise_particle_cap_scale = tf.ones([], particles.dtype)
    maximum_coordinatewise_pre_cap_absolute = tf.zeros([], particles.dtype)
    maximum_coordinatewise_post_cap_absolute = tf.zeros([], particles.dtype)
    mean_coordinatewise_cap_displacement = tf.zeros([], particles.dtype)
    fraction_coordinatewise_cap_active = tf.zeros([], particles.dtype)
    minimum_coordinatewise_cap_derivative = tf.ones([], particles.dtype)
    maximum_diagonal_scaled_system_condition = tf.zeros([], particles.dtype)
    maximum_diagonal_pre_cap_particle_rms = tf.zeros([], particles.dtype)
    maximum_diagonal_post_cap_particle_rms = tf.zeros([], particles.dtype)
    reset_route_id = tf.constant(0, tf.int32)
    if dual_cap_enabled:
        if trust_region_enabled:
            count = tf.shape(particles)[0]
            dimension = tf.shape(particles)[1]
            source_tangent = tf.zeros(
                [count, dimension, 1], dtype=particles.dtype
            )
            weight_tangent = tf.zeros([count, 1], dtype=particles.dtype)
            point_tangent = tf.zeros(
                [count, dimension, 1], dtype=particles.dtype
            )
            trust = higher_moment_shape_jvp(
                particles,
                weights,
                source_tangent,
                weight_tangent,
                output_particles,
                point_tangent,
                correction_steps=dual_cap_diagonal_steps,
                strength=dual_cap_diagonal_strength,
                floor=1.0e-5,
                diagonal_lm_damping=trust_region_lm_damping,
                diagonal_lm_scale_floor=trust_region_lm_scale_floor,
                diagonal_trust_radius=trust_region_radius,
                pairwise_correction_steps=dual_cap_pairwise_steps,
                pairwise_strength=dual_cap_pairwise_strength,
                pairwise_floor=1.0e-5,
                pairwise_particle_rms_cap=dual_cap_pairwise_particle_rms_cap,
                coordinatewise_standardized_cap=dual_cap_coordinate_cap,
                coordinatewise_standardized_cap_power=dual_cap_coordinate_cap_power,
            )
            target_mean_for_trust = tf.reduce_sum(
                weights[:, None] * particles, axis=0
            )
            target_centered_for_trust = particles - target_mean_for_trust[None, :]
            target_covariance_for_trust = tf.einsum(
                "n,ni,nj->ij",
                weights,
                target_centered_for_trust,
                target_centered_for_trust,
            )
            output_mean_for_trust = tf.reduce_mean(trust["particles"], axis=0)
            output_centered_for_trust = (
                trust["particles"] - output_mean_for_trust[None, :]
            )
            output_covariance_for_trust = tf.einsum(
                "ni,nj->ij",
                output_centered_for_trust,
                output_centered_for_trust,
            ) / tf.cast(count, particles.dtype)
            dual_cap = {
                "particles": trust["particles"],
                "valid": trust["valid"],
                "mean_residual": tf.reduce_max(
                    tf.abs(output_mean_for_trust - target_mean_for_trust)
                ),
                "covariance_residual": tf.reduce_max(
                    tf.abs(
                        output_covariance_for_trust - target_covariance_for_trust
                    )
                ),
                "maximum_pairwise_pre_cap_particle_rms": trust[
                    "maximum_pairwise_pre_cap_particle_rms"
                ],
                "maximum_pairwise_post_cap_particle_rms": trust[
                    "maximum_pairwise_post_cap_particle_rms"
                ],
                "minimum_pairwise_particle_cap_scale": trust[
                    "minimum_pairwise_particle_cap_scale"
                ],
                "maximum_coordinatewise_pre_cap_absolute": trust[
                    "maximum_coordinatewise_pre_cap_absolute"
                ],
                "maximum_coordinatewise_post_cap_absolute": trust[
                    "maximum_coordinatewise_post_cap_absolute"
                ],
                "mean_coordinatewise_cap_displacement": trust[
                    "mean_coordinatewise_cap_displacement"
                ],
                "fraction_coordinatewise_cap_active": trust[
                    "fraction_coordinatewise_cap_active"
                ],
                "minimum_coordinatewise_cap_derivative": trust[
                    "minimum_coordinatewise_cap_derivative"
                ],
            }
            maximum_diagonal_scaled_system_condition = trust[
                "maximum_diagonal_scaled_system_condition"
            ]
            maximum_diagonal_pre_cap_particle_rms = trust[
                "maximum_diagonal_pre_cap_particle_rms"
            ]
            maximum_diagonal_post_cap_particle_rms = trust[
                "maximum_diagonal_post_cap_particle_rms"
            ]
            reset_route_id = tf.constant(1, tf.int32)
        else:
            dual_cap = dual_cap_genut_primal(
                particles,
                weights,
                output_particles,
                diagonal_steps=dual_cap_diagonal_steps,
                diagonal_strength=dual_cap_diagonal_strength,
                pairwise_steps=dual_cap_pairwise_steps,
                pairwise_strength=dual_cap_pairwise_strength,
                pairwise_particle_rms_cap=dual_cap_pairwise_particle_rms_cap,
                coordinate_cap=dual_cap_coordinate_cap,
                coordinate_cap_power=dual_cap_coordinate_cap_power,
            )
        output_particles = dual_cap["particles"]
        dual_cap_valid = dual_cap["valid"]
        dual_cap_mean_residual = dual_cap["mean_residual"]
        dual_cap_covariance_residual = dual_cap["covariance_residual"]
        maximum_pairwise_pre_cap_particle_rms = dual_cap[
            "maximum_pairwise_pre_cap_particle_rms"
        ]
        maximum_pairwise_post_cap_particle_rms = dual_cap[
            "maximum_pairwise_post_cap_particle_rms"
        ]
        minimum_pairwise_particle_cap_scale = dual_cap[
            "minimum_pairwise_particle_cap_scale"
        ]
        maximum_coordinatewise_pre_cap_absolute = dual_cap[
            "maximum_coordinatewise_pre_cap_absolute"
        ]
        maximum_coordinatewise_post_cap_absolute = dual_cap[
            "maximum_coordinatewise_post_cap_absolute"
        ]
        mean_coordinatewise_cap_displacement = dual_cap[
            "mean_coordinatewise_cap_displacement"
        ]
        fraction_coordinatewise_cap_active = dual_cap[
            "fraction_coordinatewise_cap_active"
        ]
        minimum_coordinatewise_cap_derivative = dual_cap[
            "minimum_coordinatewise_cap_derivative"
        ]
        output_mean = tf.reduce_mean(output_particles, axis=0)
        output_centered = output_particles - output_mean[None, :]
        output_covariance = tf.einsum(
            "ni,nj->ij", output_centered, output_centered
        ) / tf.cast(count, particles.dtype)
        reset_valid &= dual_cap_valid
        mean_residual = tf.maximum(mean_residual, dual_cap_mean_residual)
    return {
        "particles": output_particles,
        "mean_residual": mean_residual,
        "minimum_gap_eigenvalue": minimum_gap_eigenvalue,
        "gap_condition_proxy": gap_condition_proxy,
        "target_condition_proxy": target_condition_proxy,
        "injected_condition_proxy": injected_condition_proxy,
        "affine_norm": affine_norm,
        "output_covariance": output_covariance,
        "gap_valid": gap_valid,
        "reset_valid": reset_valid,
        "minimum_row_mass": tf.reduce_min(row_mass),
        "maximum_raw_row_residual": tf.reduce_max(tf.abs(row_mass - uniform)),
        "maximum_raw_column_residual": tf.reduce_max(
            tf.abs(raw_column_mass - weights)
        ),
        "maximum_post_quotient_column_residual": tf.reduce_max(
            tf.abs(quotient_column_residual)
        ),
        "post_quotient_column_tv_error": post_quotient_column_tv_error,
        "marginal_valid": marginal_valid,
        "dual_cap_valid": dual_cap_valid,
        "dual_cap_mean_residual": dual_cap_mean_residual,
        "dual_cap_covariance_residual": dual_cap_covariance_residual,
        "maximum_pairwise_pre_cap_particle_rms": maximum_pairwise_pre_cap_particle_rms,
        "maximum_pairwise_post_cap_particle_rms": maximum_pairwise_post_cap_particle_rms,
        "minimum_pairwise_particle_cap_scale": minimum_pairwise_particle_cap_scale,
        "maximum_coordinatewise_pre_cap_absolute": maximum_coordinatewise_pre_cap_absolute,
        "maximum_coordinatewise_post_cap_absolute": maximum_coordinatewise_post_cap_absolute,
        "mean_coordinatewise_cap_displacement": mean_coordinatewise_cap_displacement,
        "fraction_coordinatewise_cap_active": fraction_coordinatewise_cap_active,
        "minimum_coordinatewise_cap_derivative": minimum_coordinatewise_cap_derivative,
        "reset_route_id": reset_route_id,
        "maximum_diagonal_scaled_system_condition": maximum_diagonal_scaled_system_condition,
        "maximum_diagonal_pre_cap_particle_rms": maximum_diagonal_pre_cap_particle_rms,
        "maximum_diagonal_post_cap_particle_rms": maximum_diagonal_post_cap_particle_rms,
        "trust_region_solver_id": tf.constant(
            1 if trust_region_enabled else 0, tf.int32
        ),
        "transport_plan_id": tf.constant(
            1 if transport_plan_mode == "streaming" else 0, tf.int32
        ),
        "transport_row_chunk_size": tf.constant(
            chunk_selection.row_chunk_size
            if chunk_selection is not None
            else int(particle_count),
            tf.int32,
        ),
        "transport_col_chunk_size": tf.constant(
            chunk_selection.col_chunk_size
            if chunk_selection is not None
            else int(particle_count),
            tf.int32,
        ),
    }


def finite_value_standard_score_guided_proposal(
    spec: LGSSMGuidedProposalSpec,
    theta: Tensor,
    observations: Tensor,
    initial_noise: Tensor,
    process_noise: Tensor,
    component_uniforms: Tensor,
    design: Tensor,
    *,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return corrected likelihood and the standard target-model filtering score.

    The likelihood uses the exact ``f * g / q`` correction. Score marks use the
    established analytical LGSSM density scores and pairwise backward kernel from
    ``ledh_contract_e_tp_lgssm_tf``. Proposal and reset derivatives are deliberately
    absent because they are not terms in the Fisher/Poyiadjis score recursion.
    """

    theta = tf.convert_to_tensor(theta, dtype=initial_noise.dtype)
    _validate_theta(theta)
    observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
    initial_noise = tf.convert_to_tensor(initial_noise, dtype=theta.dtype)
    process_noise = tf.convert_to_tensor(process_noise, dtype=theta.dtype)
    component_uniforms = tf.convert_to_tensor(component_uniforms, dtype=theta.dtype)
    design = tf.convert_to_tensor(design, dtype=theta.dtype)
    if observations.shape.rank != 2 or observations.shape[-1] != 3:
        raise ValueError("observations must have shape (T, 3)")
    horizon_static = observations.shape[0]
    if horizon_static is None:
        raise ValueError("guided XLA core requires a static observation horizon")
    if initial_noise.shape.rank != 2 or initial_noise.shape[-1] != 3:
        raise ValueError("initial_noise must have shape (N, 3)")
    if process_noise.shape.rank != 3 or process_noise.shape[-1] != 3:
        raise ValueError("process_noise must have shape (T, N, 3)")
    if component_uniforms.shape.rank != 2:
        raise ValueError("component_uniforms must have shape (T, N)")
    if design.shape.rank not in (2, 3):
        raise ValueError("design must have shape (N, 3) or (T, N, 3)")

    tf.debugging.assert_positive(theta[3])
    tf.debugging.assert_positive(theta[4])
    tf.debugging.assert_less(tf.abs(theta[:3]), tf.ones([3], theta.dtype))
    stationary_scale = theta[3] / tf.sqrt(1.0 - tf.square(theta[:3]))
    particles = initial_noise * stationary_scale[None, :]
    n_static = particles.shape[0]
    if n_static is None:
        raise ValueError("guided XLA core requires a static particle dimension")
    weights = tf.fill(
        [n_static], tf.cast(1.0, theta.dtype) / tf.cast(n_static, theta.dtype)
    )
    score_marks = standard_score._initial_target_model_score_marks(  # noqa: SLF001
        theta, particles
    )
    horizon = tf.shape(observations)[0]
    increments = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    score_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=(5,)
    )
    ess = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    maximum_weight = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    log_weight_variance = tf.TensorArray(theta.dtype, size=horizon, element_shape=())

    def body(
        time_index,
        particles_value,
        weights_value,
        score_marks_value,
        total,
        score,
        valid,
        incs,
        scores,
        esses,
        maxes,
        variances,
    ):
        step = guided_lgssm_step(
            spec,
            theta,
            particles_value,
            weights_value,
            observations[time_index],
            process_noise[time_index],
            component_uniforms[time_index],
        )
        log_parent_weights = tf.math.log(weights_value)
        current_marks = standard_score._target_model_progressive_score_marks(  # noqa: SLF001
            theta,
            particles_value,
            log_parent_weights,
            score_marks_value,
            step["particles"],
            observations[time_index],
        )
        current_score = tf.einsum(
            "n,np->p", step["normalized_weights"], current_marks
        )
        current_design = design if design.shape.rank == 2 else design[time_index]
        restored = _restore_cloud_primal(
            step["particles"],
            step["normalized_weights"],
            current_design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
        )
        restored_marks = standard_score._target_model_progressive_score_marks(  # noqa: SLF001
            theta,
            particles_value,
            log_parent_weights,
            score_marks_value,
            restored["particles"],
            observations[time_index],
        )
        step_valid = (
            restored["reset_valid"]
            & tf.reduce_all(tf.math.is_finite(step["particles"]))
            & tf.reduce_all(tf.math.is_finite(step["normalized_weights"]))
            & tf.math.is_finite(step["increment"])
            & tf.reduce_all(tf.math.is_finite(current_marks))
            & tf.reduce_all(tf.math.is_finite(current_score))
            & tf.reduce_all(tf.math.is_finite(restored_marks))
        )
        uniform = tf.fill(
            [n_static], tf.cast(1.0, theta.dtype) / tf.cast(n_static, theta.dtype)
        )
        return (
            time_index + 1,
            tf.where(step_valid, restored["particles"], particles_value),
            uniform,
            tf.where(step_valid, restored_marks, score_marks_value),
            total + tf.where(step_valid, step["increment"], tf.zeros_like(total)),
            tf.where(step_valid, current_score, score),
            valid & step_valid,
            incs.write(time_index, step["increment"]),
            scores.write(time_index, current_score),
            esses.write(time_index, step["ess"]),
            maxes.write(time_index, step["maximum_normalized_weight"]),
            variances.write(time_index, step["log_weight_variance"]),
        )

    (
        _,
        particles,
        weights,
        score_marks,
        total,
        score,
        valid,
        increments,
        score_history,
        ess,
        maximum_weight,
        log_weight_variance,
    ) = tf.while_loop(
        lambda time_index, *_: time_index < horizon,
        body,
        (
            tf.zeros([], tf.int32),
            particles,
            weights,
            score_marks,
            tf.zeros([], theta.dtype),
            tf.zeros([5], theta.dtype),
            tf.constant(True),
            increments,
            score_history,
            ess,
            maximum_weight,
            log_weight_variance,
        ),
        parallel_iterations=1,
        maximum_iterations=horizon_static,
    )
    nan = tf.constant(float("nan"), theta.dtype)
    return (
        tf.where(valid, total, nan),
        tf.where(valid, score, tf.fill([5], nan)),
        {
            "program_valid": valid,
            "value_increments": increments.stack(),
            "filtering_score_history": score_history.stack(),
            "ess": ess.stack(),
            "maximum_normalized_weight": maximum_weight.stack(),
            "log_weight_variance": log_weight_variance.stack(),
            "final_particles": particles,
            "final_weights": weights,
            "final_score_marks": score_marks,
        },
    )



__all__ = [
    "LGSSMGuidedProposalSpec",
    "defensive_mixture_log_density",
    "exact_lgssm_conditional_moments",
    "finite_value_standard_score_guided_proposal",
    "guided_lgssm_step",
]

"""Independent online O(N^2) value/score teacher for latent pre-clipping SIR.

The score is the Poyiadjis-style backward-kernel filtering-score estimate. It is
not the derivative of the reported bootstrap-particle likelihood scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import LatentPreclipSIRSSM


DTYPE = tf.float64
PARAMETER_COUNT = 3
TEACHER_ID = "latent_preclip_sir_online_n2_filtering_score_teacher_v1"
_LOG_TWO_PI = tf.math.log(tf.constant(6.283185307179586476925286766559, DTYPE))


@dataclass(frozen=True)
class SIRTeacherStaticSpec:
    state_dimension: int
    observation_dimension: int
    compartments: int
    substeps: int
    step: tf.Tensor
    zhao_cui_rk4_variant: bool
    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    process_covariance: tf.Tensor
    observation_covariance: tf.Tensor
    base_kappa: tf.Tensor
    base_nu: tf.Tensor
    adjacency: tf.Tensor
    neighbor_degree: tf.Tensor


def static_spec_from_model(model: LatentPreclipSIRSSM) -> SIRTeacherStaticSpec:
    base = model.physical_model.base_model
    adjacency_rows = []
    for neighbors in base.neighbor_sets:
        neighbor_set = set(int(index) for index in neighbors)
        adjacency_rows.append(
            [1.0 if index in neighbor_set else 0.0 for index in range(len(base.neighbor_sets))]
        )
    return SIRTeacherStaticSpec(
        state_dimension=model.state_dim(),
        observation_dimension=model.observation_dim(),
        compartments=model.observation_dim(),
        substeps=int(base._rk4_substeps),
        step=tf.convert_to_tensor(
            base.delta / tf.cast(base._rk4_substeps, DTYPE), DTYPE
        ),
        zhao_cui_rk4_variant=base.rk4_variant == "zhao_cui_sir_step",
        initial_mean=tf.convert_to_tensor(base.initial_mean, DTYPE),
        initial_covariance=tf.convert_to_tensor(base.initial_covariance, DTYPE),
        process_covariance=tf.convert_to_tensor(base.process_covariance, DTYPE),
        observation_covariance=tf.convert_to_tensor(base.observation_covariance, DTYPE),
        base_kappa=tf.convert_to_tensor(base.kappa, DTYPE),
        base_nu=tf.convert_to_tensor(base.nu, DTYPE),
        adjacency=tf.constant(adjacency_rows, DTYPE),
        neighbor_degree=tf.constant(
            [float(len(neighbors)) for neighbors in base.neighbor_sets], DTYPE
        ),
    )


def _components(theta: tf.Tensor, spec: SIRTeacherStaticSpec) -> Mapping[str, tf.Tensor]:
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_COUNT])
    kappa = spec.base_kappa * tf.exp(theta[0])
    nu = spec.base_nu * tf.exp(theta[1])
    observation_covariance = spec.observation_covariance * tf.exp(2.0 * theta[2])
    zeros = tf.zeros_like(kappa)
    return {
        "kappa": kappa,
        "nu": nu,
        "d_kappa": tf.stack([kappa, zeros, zeros], axis=-1),
        "d_nu": tf.stack([zeros, nu, zeros], axis=-1),
        "observation_covariance": observation_covariance,
    }


def _physical_state(latent: tf.Tensor, time_index: tf.Tensor | int) -> tf.Tensor:
    susceptible = latent[..., 0::2]
    infectious = latent[..., 1::2]
    clipped = tf.reshape(
        tf.stack([tf.maximum(susceptible, 0.0), infectious], axis=-1),
        tf.shape(latent),
    )
    return tf.cond(
        tf.equal(tf.cast(time_index, tf.int32), 0),
        lambda: latent,
        lambda: clipped,
    )


def _rhs_and_tangent(
    state: tf.Tensor,
    tangent: tf.Tensor,
    components: Mapping[str, tf.Tensor],
    spec: SIRTeacherStaticSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    susceptible = state[:, 0::2]
    infectious = state[:, 1::2]
    d_susceptible = tangent[:, 0::2, :]
    d_infectious = tangent[:, 1::2, :]
    susceptible_neighbor = (
        tf.linalg.matmul(susceptible, spec.adjacency, transpose_b=True)
        - susceptible * spec.neighbor_degree[None, :]
    )
    infectious_neighbor = (
        tf.linalg.matmul(infectious, spec.adjacency, transpose_b=True)
        - infectious * spec.neighbor_degree[None, :]
    )
    d_susceptible_neighbor = (
        tf.einsum("mjp,kj->mkp", d_susceptible, spec.adjacency)
        - d_susceptible * spec.neighbor_degree[None, :, None]
    )
    d_infectious_neighbor = (
        tf.einsum("mjp,kj->mkp", d_infectious, spec.adjacency)
        - d_infectious * spec.neighbor_degree[None, :, None]
    )
    kappa = components["kappa"]
    nu = components["nu"]
    infection = kappa[None, :] * susceptible * infectious
    d_infection = (
        components["d_kappa"][None, :, :]
        * susceptible[:, :, None]
        * infectious[:, :, None]
        + kappa[None, :, None]
        * (
            d_susceptible * infectious[:, :, None]
            + susceptible[:, :, None] * d_infectious
        )
    )
    rhs_s = -infection + 0.5 * susceptible_neighbor
    rhs_i = infection - nu[None, :] * infectious + 0.5 * infectious_neighbor
    d_rhs_s = -d_infection + 0.5 * d_susceptible_neighbor
    d_rhs_i = (
        d_infection
        - components["d_nu"][None, :, :] * infectious[:, :, None]
        - nu[None, :, None] * d_infectious
        + 0.5 * d_infectious_neighbor
    )
    return (
        tf.reshape(tf.stack([rhs_s, rhs_i], axis=2), tf.shape(state)),
        tf.reshape(tf.stack([d_rhs_s, d_rhs_i], axis=2), tf.shape(tangent)),
    )


def _transition_mean_and_parameter_tangent(
    previous_latent: tf.Tensor,
    theta: tf.Tensor,
    time_index: tf.Tensor | int,
    spec: SIRTeacherStaticSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    state = _physical_state(previous_latent, tf.cast(time_index, tf.int32) - 1)
    tangent = tf.zeros(
        [tf.shape(state)[0], spec.state_dimension, PARAMETER_COUNT], DTYPE
    )
    components = _components(theta, spec)
    k4_factor = tf.constant(0.5 if spec.zhao_cui_rk4_variant else 1.0, DTYPE)

    def body(index, current, current_tangent):
        k1, d_k1 = _rhs_and_tangent(current, current_tangent, components, spec)
        k2, d_k2 = _rhs_and_tangent(
            current + 0.5 * spec.step * k1,
            current_tangent + 0.5 * spec.step * d_k1,
            components,
            spec,
        )
        k3, d_k3 = _rhs_and_tangent(
            current + 0.5 * spec.step * k2,
            current_tangent + 0.5 * spec.step * d_k2,
            components,
            spec,
        )
        k4, d_k4 = _rhs_and_tangent(
            current + k4_factor * spec.step * k3,
            current_tangent + k4_factor * spec.step * d_k3,
            components,
            spec,
        )
        return (
            index + 1,
            current + (spec.step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            current_tangent
            + (spec.step / 6.0) * (d_k1 + 2.0 * d_k2 + 2.0 * d_k3 + d_k4),
        )

    _, state, tangent = tf.while_loop(
        lambda index, *_: index < spec.substeps,
        body,
        (tf.constant(0, tf.int32), state, tangent),
        maximum_iterations=spec.substeps,
    )
    return state, tangent


def _gaussian_log_density(residual: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residual))
    )
    dimension = tf.cast(tf.shape(residual)[-1], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (
        dimension * _LOG_TWO_PI
        + logdet
        + tf.reduce_sum(residual * solved, axis=-1)
    )


def initial_log_density_and_score(
    theta: tf.Tensor,
    latent: tf.Tensor,
    *,
    spec: SIRTeacherStaticSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the initial density and its explicit theta-independent score."""

    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_COUNT])
    latent = tf.reshape(
        tf.convert_to_tensor(latent, DTYPE), [-1, spec.state_dimension]
    )
    del theta
    return (
        _gaussian_log_density(latent - spec.initial_mean, spec.initial_covariance),
        tf.zeros([tf.shape(latent)[0], PARAMETER_COUNT], DTYPE),
    )


def transition_log_density_and_score(
    theta: tf.Tensor,
    previous_latent: tf.Tensor,
    current_latent: tf.Tensor,
    *,
    time_index: tf.Tensor | int,
    spec: SIRTeacherStaticSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    mean, d_mean = _transition_mean_and_parameter_tangent(
        previous_latent, theta, time_index, spec
    )
    residual = current_latent - mean
    chol = tf.linalg.cholesky(spec.process_covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residual))
    )
    score = tf.einsum("mdp,md->mp", d_mean, solved)
    return _gaussian_log_density(residual, spec.process_covariance), score


def observation_log_density_and_score(
    theta: tf.Tensor,
    latent: tf.Tensor,
    observation: tf.Tensor,
    *,
    time_index: tf.Tensor | int,
    spec: SIRTeacherStaticSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    components = _components(theta, spec)
    physical = _physical_state(latent, time_index)
    predicted = physical[:, 1::2]
    residual = tf.reshape(observation, [1, spec.observation_dimension]) - predicted
    log_density = _gaussian_log_density(
        residual, components["observation_covariance"]
    )
    chol = tf.linalg.cholesky(components["observation_covariance"])
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residual))
    )
    quadratic = tf.reduce_sum(residual * solved, axis=1)
    zeros = tf.zeros_like(quadratic)
    score = tf.stack(
        [zeros, zeros, quadratic - tf.cast(spec.observation_dimension, DTYPE)],
        axis=1,
    )
    return log_density, score


def _stateless_normals(
    seeds: tf.Tensor, shape: tuple[int, int], domain: tf.Tensor
) -> tf.Tensor:
    return tf.map_fn(
        lambda seed: tf.random.stateless_normal(
            shape,
            seed=tf.stack([seed, tf.cast(domain, tf.int32)]),
            dtype=DTYPE,
        ),
        seeds,
        fn_output_signature=tf.TensorSpec(shape, DTYPE),
    )


def _systematic_resample(
    normalized_weights: tf.Tensor, seeds: tf.Tensor, time_index: tf.Tensor
) -> tf.Tensor:
    particle_count = tf.shape(normalized_weights)[1]
    scale = tf.cast(particle_count, DTYPE)
    offsets = tf.map_fn(
        lambda seed: tf.random.stateless_uniform(
            [],
            seed=tf.stack([seed, tf.constant(2000, tf.int32) + time_index]),
            minval=tf.constant(0.0, DTYPE),
            maxval=tf.math.reciprocal(scale),
            dtype=DTYPE,
        ),
        seeds,
        fn_output_signature=DTYPE,
    )
    positions = offsets[:, None] + tf.cast(tf.range(particle_count), DTYPE)[None, :] / scale
    cumulative = tf.math.cumsum(normalized_weights, axis=1)
    cumulative = tf.concat(
        [cumulative[:, :-1], tf.ones([tf.shape(cumulative)[0], 1], DTYPE)], axis=1
    )
    return tf.searchsorted(cumulative, positions, side="right", out_type=tf.int32)


def online_sir_value_and_score_teacher(
    model: LatentPreclipSIRSSM,
    theta: tf.Tensor,
    observations: tf.Tensor,
    seeds: tf.Tensor,
    *,
    num_particles: int,
    stop_previous_marks: bool = False,
    stop_transition_score: bool = False,
) -> Mapping[str, tf.Tensor | str]:
    """Run independent replicated bootstrap filters and online score recursions."""

    spec = static_spec_from_model(model)
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_COUNT])
    observations = tf.convert_to_tensor(observations, DTYPE)
    seed_values = tf.reshape(tf.convert_to_tensor(seeds, tf.int32), [-1])
    replicate_count = seed_values.shape[0]
    observation_count = observations.shape[0]
    particle_count = int(num_particles)
    if replicate_count is None or observation_count is None:
        raise ValueError("replicate and observation counts must be static")
    if particle_count < 2:
        raise ValueError("num_particles must be at least two")
    if observations.shape != (observation_count, spec.observation_dimension):
        raise ValueError("observations have the wrong latent-SIR shape")

    initial_noise = _stateless_normals(
        seed_values, (particle_count, spec.state_dimension), tf.constant(100, tf.int32)
    )
    initial_chol = tf.linalg.cholesky(spec.initial_covariance)
    particles = spec.initial_mean[None, None, :] + tf.einsum(
        "rnj,ij->rni", initial_noise, initial_chol
    )
    flat_particles = tf.reshape(particles, [-1, spec.state_dimension])
    _, initial_density_score = initial_log_density_and_score(
        theta, flat_particles, spec=spec
    )
    initial_log_weight, initial_observation_score = observation_log_density_and_score(
        theta,
        flat_particles,
        observations[0],
        time_index=0,
        spec=spec,
    )
    log_weight = tf.reshape(initial_log_weight, [replicate_count, particle_count])
    marks = tf.reshape(
        initial_density_score + initial_observation_score,
        [replicate_count, particle_count, PARAMETER_COUNT],
    )
    log_normalizer = tf.reduce_logsumexp(log_weight, axis=1)
    normalized_weights = tf.exp(log_weight - log_normalizer[:, None])
    log_likelihood = log_normalizer - tf.math.log(tf.cast(particle_count, DTYPE))
    score = tf.reduce_sum(normalized_weights[:, :, None] * marks, axis=1)
    minimum_ess = tf.math.reciprocal(
        tf.reduce_sum(tf.square(normalized_weights), axis=1)
    )
    finite = (
        tf.math.is_finite(log_likelihood)
        & tf.reduce_all(tf.math.is_finite(score), axis=1)
        & tf.reduce_all(tf.math.is_finite(particles), axis=[1, 2])
    )
    score_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[replicate_count, PARAMETER_COUNT]
    ).write(0, score)
    increment_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[replicate_count]
    ).write(0, log_likelihood)
    backward_error_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[replicate_count]
    ).write(0, tf.zeros([replicate_count], DTYPE))
    process_chol = tf.linalg.cholesky(spec.process_covariance)

    def body(
        time_index,
        previous_particles,
        previous_weights,
        previous_marks,
        previous_score,
        total_log_likelihood,
        minimum_ess,
        finite,
        score_history,
        increment_history,
        backward_error_history,
    ):
        ancestors = _systematic_resample(previous_weights, seed_values, time_index)
        parent_particles = tf.gather(previous_particles, ancestors, batch_dims=1)
        flat_parents = tf.reshape(parent_particles, [-1, spec.state_dimension])
        means, _ = _transition_mean_and_parameter_tangent(
            flat_parents, theta, time_index, spec
        )
        process_noise = _stateless_normals(
            seed_values,
            (particle_count, spec.state_dimension),
            tf.constant(1000, tf.int32) + time_index,
        )
        current_particles = tf.reshape(means, tf.shape(parent_particles)) + tf.einsum(
            "rnj,ij->rni", process_noise, process_chol
        )

        current_expanded = current_particles[:, :, None, :]
        previous_expanded = previous_particles[:, None, :, :]
        pair_shape = [replicate_count * particle_count * particle_count, spec.state_dimension]
        pair_previous = tf.reshape(
            tf.broadcast_to(
                previous_expanded,
                [replicate_count, particle_count, particle_count, spec.state_dimension],
            ),
            pair_shape,
        )
        pair_current = tf.reshape(
            tf.broadcast_to(
                current_expanded,
                [replicate_count, particle_count, particle_count, spec.state_dimension],
            ),
            pair_shape,
        )
        pair_log_transition, pair_transition_score = transition_log_density_and_score(
            theta,
            pair_previous,
            pair_current,
            time_index=time_index,
            spec=spec,
        )
        pair_log_transition = tf.reshape(
            pair_log_transition, [replicate_count, particle_count, particle_count]
        )
        pair_transition_score = tf.reshape(
            pair_transition_score,
            [replicate_count, particle_count, particle_count, PARAMETER_COUNT],
        )
        backward_logits = tf.math.log(previous_weights)[:, None, :] + pair_log_transition
        backward_log_normalizer = tf.reduce_logsumexp(backward_logits, axis=2)
        backward = tf.exp(backward_logits - backward_log_normalizer[:, :, None])
        backward_error = tf.reduce_max(
            tf.abs(tf.reduce_sum(backward, axis=2) - 1.0), axis=1
        )
        carried_marks = (
            tf.zeros_like(previous_marks) if stop_previous_marks else previous_marks
        )
        local_transition_score = (
            tf.zeros_like(pair_transition_score)
            if stop_transition_score
            else pair_transition_score
        )
        predictive_marks = tf.reduce_sum(
            backward[:, :, :, None]
            * (carried_marks[:, None, :, :] + local_transition_score),
            axis=2,
        )
        flat_current = tf.reshape(current_particles, [-1, spec.state_dimension])
        observation_log_weight, observation_score = observation_log_density_and_score(
            theta,
            flat_current,
            observations[time_index],
            time_index=time_index,
            spec=spec,
        )
        observation_log_weight = tf.reshape(
            observation_log_weight, [replicate_count, particle_count]
        )
        observation_score = tf.reshape(
            observation_score, [replicate_count, particle_count, PARAMETER_COUNT]
        )
        current_marks = predictive_marks + observation_score
        log_normalizer = tf.reduce_logsumexp(observation_log_weight, axis=1)
        current_weights = tf.exp(
            observation_log_weight - log_normalizer[:, None]
        )
        increment = log_normalizer - tf.math.log(tf.cast(particle_count, DTYPE))
        current_score = tf.reduce_sum(
            current_weights[:, :, None] * current_marks, axis=1
        )
        ess = tf.math.reciprocal(tf.reduce_sum(tf.square(current_weights), axis=1))
        step_finite = (
            tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(current_score), axis=1)
            & tf.reduce_all(tf.math.is_finite(current_particles), axis=[1, 2])
            & tf.math.is_finite(backward_error)
        )
        score_history = score_history.write(time_index, current_score)
        increment_history = increment_history.write(time_index, increment)
        backward_error_history = backward_error_history.write(time_index, backward_error)
        return (
            time_index + 1,
            current_particles,
            current_weights,
            current_marks,
            current_score,
            total_log_likelihood + increment,
            tf.minimum(minimum_ess, ess),
            finite & step_finite,
            score_history,
            increment_history,
            backward_error_history,
        )

    result = tf.while_loop(
        lambda time_index, *_: time_index < observation_count,
        body,
        (
            tf.constant(1, tf.int32),
            particles,
            normalized_weights,
            marks,
            score,
            log_likelihood,
            minimum_ess,
            finite,
            score_history,
            increment_history,
            backward_error_history,
        ),
        maximum_iterations=max(0, observation_count - 1),
        parallel_iterations=1,
    )
    score_history_tensor = tf.transpose(result[8].stack(), [1, 0, 2])
    increment_history_tensor = tf.transpose(result[9].stack(), [1, 0])
    return {
        "teacher_id": TEACHER_ID,
        "score_semantics": "online_backward_kernel_filtering_score_not_value_autodiff",
        "log_likelihood": result[5],
        "score": result[4],
        "score_history": score_history_tensor,
        "increment_score_history": tf.concat(
            [
                score_history_tensor[:, :1, :],
                score_history_tensor[:, 1:, :] - score_history_tensor[:, :-1, :],
            ],
            axis=1,
        ),
        "increment_history": increment_history_tensor,
        "minimum_ess": result[6],
        "maximum_backward_row_sum_error": tf.reduce_max(result[10].stack(), axis=0),
        "finite": result[7],
        "seed": seed_values,
        "num_particles": tf.fill([replicate_count], particle_count),
        "initial_particles": particles,
        "initial_normalized_weights": normalized_weights,
        "initial_marks": marks,
        "stop_previous_marks": tf.constant(stop_previous_marks),
        "stop_transition_score": tf.constant(stop_transition_score),
    }


def make_online_sir_teacher(
    model: LatentPreclipSIRSSM,
    observations: tf.Tensor,
    seeds: tf.Tensor,
    *,
    num_particles: int,
    jit_compile: bool = True,
):
    """Bind the teacher with the repository-default XLA JIT policy."""

    observations = tf.convert_to_tensor(observations, DTYPE)
    seeds = tf.convert_to_tensor(seeds, tf.int32)

    @tf.function(
        input_signature=[tf.TensorSpec([PARAMETER_COUNT], DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def teacher(theta: tf.Tensor):
        result = online_sir_value_and_score_teacher(
            model,
            theta,
            observations,
            seeds,
            num_particles=num_particles,
        )
        return {
            key: value
            for key, value in result.items()
            if tf.is_tensor(value)
        }

    return teacher


__all__ = [
    "DTYPE",
    "PARAMETER_COUNT",
    "SIRTeacherStaticSpec",
    "TEACHER_ID",
    "initial_log_density_and_score",
    "make_online_sir_teacher",
    "observation_log_density_and_score",
    "online_sir_value_and_score_teacher",
    "static_spec_from_model",
    "transition_log_density_and_score",
]

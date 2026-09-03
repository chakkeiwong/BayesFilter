"""Initialization-design harness for a fixed LEDH-PFPF-GenUT recursion.

The likelihood uses the LEDH proposal and exact PFPF density correction. The
score uses repository analytical local-density scores in the standard backward
filtering recursion; it is not a derivative of the finite likelihood, flow, or
reset implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as lgssm_score
from bayesfilter.highdim import sir_online_score_teacher_tf as sir_score
from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
from bayesfilter.highdim.sqmc_tf import (
    hilbert_permutation,
    inverse_cdf_ancestor_indices,
)
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    batched_ledh_flow_core_tf,
)


Tensor = tf.Tensor
_LOG_TWO_PI = math.log(2.0 * math.pi)
ANCESTRY_POLICIES = (
    "existing_one_to_one",
    "hilbert_one_to_one",
    "hilbert_inverse_cdf",
)


@dataclass(frozen=True)
class InitializationModelSpec:
    model_id: str
    state_dimension: int
    observation_dimension: int
    parameter_count: int
    family: str
    transition_before_first_observation: bool

    def __post_init__(self) -> None:
        if self.family not in {"diagonal_lgssm", "reduced_sir"}:
            raise ValueError("unsupported initialization comparison model")


def diagonal_lgssm_spec() -> InitializationModelSpec:
    return InitializationModelSpec(
        model_id="diagonal_lgssm_d3_initialization_v1",
        state_dimension=3,
        observation_dimension=3,
        parameter_count=5,
        family="diagonal_lgssm",
        transition_before_first_observation=True,
    )


def reduced_sir_spec() -> InitializationModelSpec:
    return InitializationModelSpec(
        model_id="reduced_continuous_preclip_sir_j1_v1",
        state_dimension=2,
        observation_dimension=1,
        parameter_count=3,
        family="reduced_sir",
        transition_before_first_observation=False,
    )


def _normal_log_density_scale(residual: Tensor, scale: Tensor) -> Tensor:
    residual = tf.convert_to_tensor(residual)
    scale = tf.cast(scale, residual.dtype)
    dimension = tf.cast(tf.shape(residual)[-1], residual.dtype)
    return -0.5 * (
        dimension * tf.cast(_LOG_TWO_PI, residual.dtype)
        + 2.0 * dimension * tf.math.log(scale)
        + tf.reduce_sum(tf.square(residual), axis=-1) / tf.square(scale)
    )


def _lgssm_components(theta: Tensor) -> dict[str, Tensor]:
    dtype = theta.dtype
    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    tf.debugging.assert_positive(q_scale)
    tf.debugging.assert_positive(r_scale)
    tf.debugging.assert_less(tf.abs(phi), tf.ones([3], dtype))
    return {
        "transition_matrix": tf.linalg.diag(phi),
        "transition_covariance": tf.square(q_scale) * tf.eye(3, dtype=dtype),
        "observation_covariance": tf.square(r_scale) * tf.eye(3, dtype=dtype),
        "observation_matrix": lgssm_score.lgssm._observation_matrix(dtype),  # noqa: SLF001
        "initial_mean": tf.zeros([3], dtype),
        "initial_cholesky": tf.linalg.diag(
            q_scale / tf.sqrt(1.0 - tf.square(phi))
        ),
    }


def _sir_components(theta: Tensor, static_spec) -> dict[str, Tensor]:
    dtype = theta.dtype
    physical = sir_score._components(  # noqa: SLF001
        tf.cast(theta, sir_score.DTYPE), static_spec
    )
    return {
        "transition_matrix": tf.eye(2, dtype=dtype),
        "transition_covariance": tf.cast(static_spec.process_covariance, dtype),
        "observation_covariance": tf.cast(
            physical["observation_covariance"], dtype
        ),
        "initial_mean": tf.cast(static_spec.initial_mean, dtype),
        "initial_cholesky": tf.linalg.cholesky(
            tf.cast(static_spec.initial_covariance, dtype)
        ),
    }


def _lgssm_callbacks(components: dict[str, Tensor]):
    matrix = components["observation_matrix"]

    def observation_fn(points):
        return tf.einsum("...d,od->...o", points, matrix)

    def jacobian_fn(points):
        return tf.broadcast_to(
            matrix,
            tf.concat([tf.shape(points)[:-1], tf.shape(matrix)], axis=0),
        )

    def residual_fn(predicted, observed):
        return observed - predicted

    return observation_fn, jacobian_fn, residual_fn


def _sir_callbacks(time_index: int):
    def observation_fn(points):
        physical = sir_score._physical_state(  # noqa: SLF001
            points, tf.constant(time_index, tf.int32)
        )
        return physical[..., 1::2]

    def jacobian_fn(points):
        shape = tf.concat([tf.shape(points)[:-1], [1, 2]], axis=0)
        return tf.broadcast_to(tf.constant([[0.0, 1.0]], points.dtype), shape)

    def residual_fn(predicted, observed):
        return observed - predicted

    return observation_fn, jacobian_fn, residual_fn


def _lgssm_observation_log_density(
    theta: Tensor, children: Tensor, observation: Tensor, components: dict[str, Tensor]
) -> Tensor:
    predicted = tf.einsum(
        "nd,od->no", children, components["observation_matrix"]
    )
    return _normal_log_density_scale(observation[None, :] - predicted, theta[4])


def _sir_initial_marks(theta: Tensor, particles: Tensor, static_spec) -> Tensor:
    _, marks = sir_score.initial_log_density_and_score(
        tf.cast(theta, sir_score.DTYPE),
        tf.cast(particles, sir_score.DTYPE),
        spec=static_spec,
    )
    return tf.cast(marks, theta.dtype)


def _sir_local_initial_observation_marks(
    theta: Tensor, particles: Tensor, observation: Tensor, static_spec
) -> Tensor:
    _, initial_marks = sir_score.initial_log_density_and_score(
        tf.cast(theta, sir_score.DTYPE),
        tf.cast(particles, sir_score.DTYPE),
        spec=static_spec,
    )
    _, observation_marks = sir_score.observation_log_density_and_score(
        tf.cast(theta, sir_score.DTYPE),
        tf.cast(particles, sir_score.DTYPE),
        tf.cast(observation, sir_score.DTYPE),
        time_index=0,
        spec=static_spec,
    )
    return tf.cast(initial_marks + observation_marks, theta.dtype)


def _sir_progressive_marks(
    theta: Tensor,
    parents: Tensor,
    parent_log_weights: Tensor,
    parent_marks: Tensor,
    children: Tensor,
    observation: Tensor,
    *,
    time_index: int,
    static_spec,
) -> Tensor:
    count = int(parents.shape[0])
    parent_grid = tf.broadcast_to(parents[None, :, :], [count, count, 2])
    child_grid = tf.broadcast_to(children[:, None, :], [count, count, 2])
    transition_log_density, transition_score = (
        sir_score.transition_log_density_and_score(
            tf.cast(theta, sir_score.DTYPE),
            tf.cast(tf.reshape(parent_grid, [-1, 2]), sir_score.DTYPE),
            tf.cast(tf.reshape(child_grid, [-1, 2]), sir_score.DTYPE),
            time_index=time_index,
            spec=static_spec,
        )
    )
    transition_log_density = tf.cast(
        tf.reshape(transition_log_density, [count, count]), theta.dtype
    )
    transition_score = tf.cast(
        tf.reshape(transition_score, [count, count, 3]), theta.dtype
    )
    backward = tf.nn.softmax(
        transition_log_density + parent_log_weights[None, :], axis=1
    )
    inherited = tf.einsum(
        "ki,kip->kp", backward, parent_marks[None, :, :] + transition_score
    )
    _, observation_score = sir_score.observation_log_density_and_score(
        tf.cast(theta, sir_score.DTYPE),
        tf.cast(children, sir_score.DTYPE),
        tf.cast(observation, sir_score.DTYPE),
        time_index=time_index,
        spec=static_spec,
    )
    return inherited + tf.cast(observation_score, theta.dtype)


def _apply_reset(
    particles: Tensor,
    weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
) -> dict[str, Tensor]:
    return _restore_cloud_primal(
        particles,
        weights,
        design,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
    )


def _transition_records(
    particles: Tensor,
    weights: Tensor,
    score_marks: Tensor,
    innovations: Tensor,
    ancestor_uniforms: Tensor,
    *,
    ancestry_policy: str,
    state_map_location: Tensor,
    state_map_scale: Tensor,
    hilbert_bits: int,
) -> dict[str, Tensor]:
    """Build flow ancestors while retaining the full backward-score cloud."""

    if ancestry_policy not in ANCESTRY_POLICIES:
        raise ValueError(f"unsupported ancestry policy: {ancestry_policy}")
    particle_count = int(particles.shape[0])
    row_identities = tf.range(particle_count, dtype=tf.int32)
    ties = tf.zeros([], tf.int32)
    saturation = tf.zeros([], particles.dtype)

    if ancestry_policy == "existing_one_to_one":
        order = row_identities
    else:
        order, ties, saturation = hilbert_permutation(
            particles,
            state_map_location,
            state_map_scale,
            bits=hilbert_bits,
        )

    backward_parents = tf.gather(particles, order)
    backward_weights = tf.gather(weights, order)
    backward_marks = tf.gather(score_marks, order)
    ordered_row_identities = tf.gather(row_identities, order)

    if ancestry_policy == "hilbert_inverse_cdf":
        ancestor_indices = inverse_cdf_ancestor_indices(
            ancestor_uniforms, backward_weights
        )
    else:
        ancestor_indices = tf.range(particle_count, dtype=tf.int32)

    return {
        "backward_parents": backward_parents,
        "backward_weights": backward_weights,
        "backward_marks": backward_marks,
        "flow_ancestors": tf.gather(backward_parents, ancestor_indices),
        "innovations": innovations,
        "selected_row_identities": tf.gather(
            ordered_row_identities, ancestor_indices
        ),
        "hilbert_ties": ties,
        "state_map_saturation": saturation,
    }


def finite_value_standard_score_ledh_pfpf_genut(
    spec: InitializationModelSpec,
    theta: Tensor,
    observations: Tensor,
    initial_noise: Tensor,
    process_noise: Tensor,
    design: Tensor,
    *,
    sir_static_spec=None,
    ancestry_policy: str = "existing_one_to_one",
    process_ancestor_uniforms: Tensor | None = None,
    state_map_location: Tensor | None = None,
    state_map_scale: Tensor | None = None,
    hilbert_bits: int = 12,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Run the fixed LEDH-PFPF-GenUT value and standard-score recursion."""

    theta = tf.convert_to_tensor(theta, dtype=initial_noise.dtype)
    observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
    initial_noise = tf.convert_to_tensor(initial_noise, dtype=theta.dtype)
    process_noise = tf.convert_to_tensor(process_noise, dtype=theta.dtype)
    design = tf.convert_to_tensor(design, dtype=theta.dtype)
    horizon = observations.shape[0]
    particle_count = initial_noise.shape[0]
    if horizon is None or particle_count is None:
        raise ValueError("XLA core requires static horizon and particle count")
    expected_process_steps = horizon if spec.transition_before_first_observation else horizon - 1
    if theta.shape != (spec.parameter_count,):
        raise ValueError("theta shape mismatch")
    if observations.shape != (horizon, spec.observation_dimension):
        raise ValueError("observation shape mismatch")
    if initial_noise.shape != (particle_count, spec.state_dimension):
        raise ValueError("initial noise shape mismatch")
    if process_noise.shape != (
        expected_process_steps,
        particle_count,
        spec.state_dimension,
    ):
        raise ValueError("process noise shape mismatch")
    if design.shape != (particle_count, spec.state_dimension):
        raise ValueError("GenUT design shape mismatch")
    if spec.family == "reduced_sir" and sir_static_spec is None:
        raise ValueError("reduced SIR requires its repository static score spec")
    if ancestry_policy not in ANCESTRY_POLICIES:
        raise ValueError(f"unsupported ancestry policy: {ancestry_policy}")
    if process_ancestor_uniforms is None:
        process_ancestor_uniforms = tf.zeros(
            [expected_process_steps, particle_count], theta.dtype
        )
    process_ancestor_uniforms = tf.convert_to_tensor(
        process_ancestor_uniforms, dtype=theta.dtype
    )
    if process_ancestor_uniforms.shape != (expected_process_steps, particle_count):
        raise ValueError("process ancestor uniform shape mismatch")
    if state_map_location is None:
        state_map_location = tf.zeros([spec.state_dimension], theta.dtype)
    if state_map_scale is None:
        state_map_scale = tf.ones([spec.state_dimension], theta.dtype)
    state_map_location = tf.convert_to_tensor(state_map_location, dtype=theta.dtype)
    state_map_scale = tf.convert_to_tensor(state_map_scale, dtype=theta.dtype)
    if state_map_location.shape != (spec.state_dimension,):
        raise ValueError("state-map location shape mismatch")
    if state_map_scale.shape != (spec.state_dimension,):
        raise ValueError("state-map scale shape mismatch")

    components = (
        _lgssm_components(theta)
        if spec.family == "diagonal_lgssm"
        else _sir_components(theta, sir_static_spec)
    )
    particles = components["initial_mean"][None, :] + tf.linalg.matmul(
        initial_noise, components["initial_cholesky"], transpose_b=True
    )
    uniform = tf.fill(
        [particle_count], tf.cast(1.0 / float(particle_count), theta.dtype)
    )
    weights = uniform
    total = tf.zeros([], theta.dtype)
    score = tf.zeros([spec.parameter_count], theta.dtype)
    valid = tf.constant(True)
    ess_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    maximum_weight_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    reset_residual_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    ancestry_unique_history = tf.TensorArray(
        tf.int32, size=horizon, element_shape=()
    )
    ancestry_identity_history = tf.TensorArray(
        tf.int32, size=horizon, element_shape=(particle_count,)
    )
    hilbert_tie_history = tf.TensorArray(
        tf.int32, size=horizon, element_shape=()
    )
    state_map_saturation_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )

    if spec.family == "diagonal_lgssm":
        score_marks = lgssm_score._initial_target_model_score_marks(  # noqa: SLF001
            theta, particles
        )
        start_index = 0
    else:
        pre_flow = particles
        prior_mean = tf.broadcast_to(
            components["initial_mean"][None, :], tf.shape(particles)
        )
        observation_fn, jacobian_fn, residual_fn = _sir_callbacks(0)
        flow = batched_ledh_flow_core_tf(
            pre_flow_particles=pre_flow[None, :, :],
            ancestors=particles[None, :, :],
            observation=observations[0],
            transition_matrix=tf.eye(2, batch_shape=[1], dtype=theta.dtype),
            transition_covariance=(
                components["initial_cholesky"]
                @ tf.transpose(components["initial_cholesky"])
            )[None, :, :],
            observation_covariance=components["observation_covariance"][None, :, :],
            observation_fn=observation_fn,
            observation_jacobian_fn=jacobian_fn,
            observation_residual_fn=residual_fn,
            prior_mean_fn=lambda _ancestors: prior_mean[None, :, :],
        )
        children = flow.post_flow_particles[0]
        target_initial, _ = sir_score.initial_log_density_and_score(
            tf.cast(theta, sir_score.DTYPE),
            tf.cast(children, sir_score.DTYPE),
            spec=sir_static_spec,
        )
        target_observation, _ = sir_score.observation_log_density_and_score(
            tf.cast(theta, sir_score.DTYPE),
            tf.cast(children, sir_score.DTYPE),
            tf.cast(observations[0], sir_score.DTYPE),
            time_index=0,
            spec=sir_static_spec,
        )
        logits = (
            tf.cast(target_initial + target_observation, theta.dtype)
            - flow.pre_flow_log_density[0]
            + flow.forward_log_det[0]
            + tf.math.log(uniform)
        )
        increment = tf.reduce_logsumexp(logits)
        step_weights = tf.exp(logits - increment)
        current_marks = _sir_local_initial_observation_marks(
            theta, children, observations[0], sir_static_spec
        )
        score = tf.einsum("n,np->p", step_weights, current_marks)
        restored = _apply_reset(
            children,
            step_weights,
            design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
        )
        restored_marks = _sir_local_initial_observation_marks(
            theta, restored["particles"], observations[0], sir_static_spec
        )
        step_valid = (
            restored["reset_valid"]
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(score))
            & tf.reduce_all(tf.math.is_finite(restored_marks))
        )
        particles = tf.where(step_valid, restored["particles"], particles)
        score_marks = tf.where(
            step_valid,
            restored_marks,
            _sir_initial_marks(theta, particles, sir_static_spec),
        )
        total += tf.where(step_valid, increment, tf.zeros_like(increment))
        valid &= step_valid
        ess_history = ess_history.write(
            0, tf.math.reciprocal(tf.reduce_sum(tf.square(step_weights)))
        )
        maximum_weight_history = maximum_weight_history.write(
            0, tf.reduce_max(step_weights)
        )
        reset_residual_history = reset_residual_history.write(
            0, restored["mean_residual"]
        )
        ancestry_unique_history = ancestry_unique_history.write(
            0, tf.cast(particle_count, tf.int32)
        )
        ancestry_identity_history = ancestry_identity_history.write(
            0, tf.range(particle_count, dtype=tf.int32)
        )
        hilbert_tie_history = hilbert_tie_history.write(0, tf.zeros([], tf.int32))
        state_map_saturation_history = state_map_saturation_history.write(
            0, tf.zeros([], theta.dtype)
        )
        start_index = 1

    for time_index in range(start_index, horizon):
        noise_index = (
            time_index
            if spec.transition_before_first_observation
            else time_index - 1
        )
        records = _transition_records(
            particles,
            weights,
            score_marks,
            process_noise[noise_index],
            process_ancestor_uniforms[noise_index],
            ancestry_policy=ancestry_policy,
            state_map_location=state_map_location,
            state_map_scale=state_map_scale,
            hilbert_bits=hilbert_bits,
        )
        parents = records["backward_parents"]
        parent_log_weights = tf.math.log(records["backward_weights"])
        parent_score_marks = records["backward_marks"]
        flow_ancestors = records["flow_ancestors"]
        if spec.family == "diagonal_lgssm":
            prior_mean = tf.linalg.matmul(
                flow_ancestors,
                components["transition_matrix"],
                transpose_b=True,
            )
            observation_fn, jacobian_fn, residual_fn = _lgssm_callbacks(components)
            prior_mean_fn = None
        else:
            prior_mean, _ = sir_score._transition_mean_and_parameter_tangent(  # noqa: SLF001
                tf.cast(flow_ancestors, sir_score.DTYPE),
                tf.cast(theta, sir_score.DTYPE),
                time_index,
                sir_static_spec,
            )
            prior_mean = tf.cast(prior_mean, theta.dtype)
            observation_fn, jacobian_fn, residual_fn = _sir_callbacks(time_index)
            prior_mean_fn = lambda _ancestors, mean=prior_mean: mean[None, :, :]
        process_cholesky = tf.linalg.cholesky(components["transition_covariance"])
        pre_flow = prior_mean + tf.linalg.matmul(
            records["innovations"], process_cholesky, transpose_b=True
        )
        flow = batched_ledh_flow_core_tf(
            pre_flow_particles=pre_flow[None, :, :],
            ancestors=flow_ancestors[None, :, :],
            observation=observations[time_index],
            transition_matrix=components["transition_matrix"][None, :, :],
            transition_covariance=components["transition_covariance"][None, :, :],
            observation_covariance=components["observation_covariance"][None, :, :],
            observation_fn=observation_fn,
            observation_jacobian_fn=jacobian_fn,
            observation_residual_fn=residual_fn,
            prior_mean_fn=prior_mean_fn,
        )
        children = flow.post_flow_particles[0]
        if spec.family == "diagonal_lgssm":
            target_transition = _normal_log_density_scale(
                children - prior_mean, theta[3]
            )
            target_observation = _lgssm_observation_log_density(
                theta, children, observations[time_index], components
            )
            current_marks = lgssm_score._target_model_progressive_score_marks(  # noqa: SLF001
                theta,
                parents,
                parent_log_weights,
                parent_score_marks,
                children,
                observations[time_index],
            )
        else:
            target_transition, _ = sir_score.transition_log_density_and_score(
                tf.cast(theta, sir_score.DTYPE),
                tf.cast(flow_ancestors, sir_score.DTYPE),
                tf.cast(children, sir_score.DTYPE),
                time_index=time_index,
                spec=sir_static_spec,
            )
            target_observation, _ = sir_score.observation_log_density_and_score(
                tf.cast(theta, sir_score.DTYPE),
                tf.cast(children, sir_score.DTYPE),
                tf.cast(observations[time_index], sir_score.DTYPE),
                time_index=time_index,
                spec=sir_static_spec,
            )
            target_transition = tf.cast(target_transition, theta.dtype)
            target_observation = tf.cast(target_observation, theta.dtype)
            current_marks = _sir_progressive_marks(
                theta,
                parents,
                parent_log_weights,
                parent_score_marks,
                children,
                observations[time_index],
                time_index=time_index,
                static_spec=sir_static_spec,
            )
        logits = (
            tf.math.log(uniform)
            + target_transition
            + target_observation
            - flow.pre_flow_log_density[0]
            + flow.forward_log_det[0]
        )
        increment = tf.reduce_logsumexp(logits)
        step_weights = tf.exp(logits - increment)
        current_score = tf.einsum("n,np->p", step_weights, current_marks)
        restored = _apply_reset(
            children,
            step_weights,
            design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
        )
        if spec.family == "diagonal_lgssm":
            restored_marks = lgssm_score._target_model_progressive_score_marks(  # noqa: SLF001
                theta,
                parents,
                parent_log_weights,
                parent_score_marks,
                restored["particles"],
                observations[time_index],
            )
        else:
            restored_marks = _sir_progressive_marks(
                theta,
                parents,
                parent_log_weights,
                parent_score_marks,
                restored["particles"],
                observations[time_index],
                time_index=time_index,
                static_spec=sir_static_spec,
            )
        step_valid = (
            restored["reset_valid"]
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(current_score))
            & tf.reduce_all(tf.math.is_finite(restored_marks))
        )
        particles = tf.where(step_valid, restored["particles"], particles)
        weights = uniform
        score_marks = tf.where(step_valid, restored_marks, score_marks)
        total += tf.where(step_valid, increment, tf.zeros_like(increment))
        score = tf.where(step_valid, current_score, score)
        valid &= step_valid
        ess_history = ess_history.write(
            time_index, tf.math.reciprocal(tf.reduce_sum(tf.square(step_weights)))
        )
        maximum_weight_history = maximum_weight_history.write(
            time_index, tf.reduce_max(step_weights)
        )
        reset_residual_history = reset_residual_history.write(
            time_index, restored["mean_residual"]
        )
        selected_ids = records["selected_row_identities"]
        sorted_selected_ids = tf.sort(selected_ids)
        unique_count = 1 + tf.reduce_sum(
            tf.cast(
                tf.not_equal(sorted_selected_ids[1:], sorted_selected_ids[:-1]),
                tf.int32,
            )
        )
        ancestry_unique_history = ancestry_unique_history.write(
            time_index, unique_count
        )
        ancestry_identity_history = ancestry_identity_history.write(
            time_index, selected_ids
        )
        hilbert_tie_history = hilbert_tie_history.write(
            time_index, records["hilbert_ties"]
        )
        state_map_saturation_history = state_map_saturation_history.write(
            time_index, records["state_map_saturation"]
        )

    nan = tf.constant(float("nan"), theta.dtype)
    return (
        tf.where(valid, total, nan),
        tf.where(valid, score, tf.fill([spec.parameter_count], nan)),
        {
            "program_valid": valid,
            "ess": ess_history.stack(),
            "maximum_normalized_weight": maximum_weight_history.stack(),
            "reset_mean_residual": reset_residual_history.stack(),
            "ancestry_unique_count": ancestry_unique_history.stack(),
            "selected_ancestor_row_identity": ancestry_identity_history.stack(),
            "hilbert_tie_count": hilbert_tie_history.stack(),
            "state_map_saturation_rate": state_map_saturation_history.stack(),
            "final_particles": particles,
            "final_score_marks": score_marks,
        },
    )


__all__ = [
    "InitializationModelSpec",
    "ANCESTRY_POLICIES",
    "_transition_records",
    "diagonal_lgssm_spec",
    "finite_value_standard_score_ledh_pfpf_genut",
    "reduced_sir_spec",
]

"""Initial-only IID/RQMC LEDH-PFPF-GenUT value and standard-score core."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
    DiagonalLGSSMAnalyticalModel,
    LEDHGenUTModelCallbacks,
)
from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as lgssm_score
from bayesfilter.highdim.sqmc_tf import (
    hilbert_permutation,
    inverse_cdf_ancestor_indices,
)
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    batched_ledh_flow_core_tf,
)


Tensor = tf.Tensor
ANCESTRY_POLICIES = (
    "existing_one_to_one",
    "hilbert_inverse_cdf",
    "hilbert_systematic_equal_weight",
    "hilbert_permutation_one_to_one",
)
RESET_POLICIES = ("contract_e", "ot_only", "none")
STATE_MAP_POLICIES = ("adaptive_empirical", "fixed_supplied")


def _transition_ancestors(
    particles: Tensor,
    weights: Tensor,
    ancestor_uniforms: Tensor,
    *,
    ancestry_policy: str,
    state_map_location: Tensor,
    state_map_scale: Tensor,
    hilbert_bits: int,
    state_map_policy: str = "adaptive_empirical",
) -> dict[str, Tensor]:
    """Select proposal ancestors without changing the all-parent score cloud."""

    if ancestry_policy not in ANCESTRY_POLICIES:
        raise ValueError(f"unsupported ancestry policy: {ancestry_policy}")
    if state_map_policy not in STATE_MAP_POLICIES:
        raise ValueError(f"unsupported state-map policy: {state_map_policy}")
    particle_count = int(particles.shape[0])
    if ancestry_policy == "existing_one_to_one":
        identities = tf.range(particle_count, dtype=tf.int32)
        return {
            "flow_ancestors": particles,
            "selected_row_identities": identities,
            "hilbert_ties": tf.zeros([], tf.int32),
            "state_map_saturation": tf.zeros([], particles.dtype),
            "ancestry_unique_count": tf.cast(particle_count, tf.int32),
            "ancestry_permutation_valid": tf.constant(True),
            "equal_weight_valid": tf.constant(True),
            "equal_weight_error": tf.zeros([], particles.dtype),
        }

    if state_map_policy == "adaptive_empirical":
        empirical_location = tf.reduce_mean(particles, axis=0)
        empirical_scale = tf.sqrt(
            tf.reduce_mean(
                tf.square(particles - empirical_location[None, :]), axis=0
            )
        )
    else:
        empirical_location = tf.cast(state_map_location, particles.dtype)
        empirical_scale = tf.cast(state_map_scale, particles.dtype)
    scale_floor = tf.maximum(
        tf.cast(1.0e-3, particles.dtype) * state_map_scale,
        tf.fill(tf.shape(state_map_scale), tf.cast(1.0e-6, particles.dtype)),
    )
    order, ties, saturation = hilbert_permutation(
        particles,
        empirical_location,
        tf.maximum(empirical_scale, scale_floor),
        bits=hilbert_bits,
    )
    ordered_particles = tf.gather(particles, order)
    ordered_weights = tf.gather(weights, order)
    uniform_weights = tf.fill(
        [particle_count], tf.cast(1.0 / float(particle_count), particles.dtype)
    )
    equal_weight_error = tf.reduce_max(tf.abs(weights - uniform_weights))
    # Contract-E resets are intended to produce equal weights. The explicit
    # one-to-one policies fail closed if a caller uses them with a weighted
    # empirical measure, where a permutation would no longer be a valid
    # inverse-CDF sample.
    equal_weight_valid = equal_weight_error <= tf.cast(2.0e-6, particles.dtype)
    if ancestry_policy == "hilbert_systematic_equal_weight":
        # For exact 1/N weights, every shifted-systematic coordinate falls in
        # its correspondingly ranked CDF bin. Use the integer-equivalent map to
        # avoid float32 cumulative-sum boundary duplication.
        ancestor_indices = tf.range(particle_count, dtype=tf.int32)
    elif ancestry_policy == "hilbert_permutation_one_to_one":
        ancestor_indices = tf.range(particle_count, dtype=tf.int32)
    else:
        ancestor_indices = inverse_cdf_ancestor_indices(
            ancestor_uniforms, ordered_weights
        )
    selected_identities = tf.gather(order, ancestor_indices)
    expected_identities = tf.range(particle_count, dtype=tf.int32)
    sorted_identities = tf.sort(selected_identities)
    ancestry_permutation_valid = tf.reduce_all(
        tf.equal(sorted_identities, expected_identities)
    )
    ancestry_unique_count = tf.reduce_sum(
        tf.cast(
            tf.reduce_any(
                tf.equal(expected_identities[:, None], selected_identities[None, :]),
                axis=1,
            ),
            tf.int32,
        )
    )
    if ancestry_policy in (
        "hilbert_systematic_equal_weight",
        "hilbert_permutation_one_to_one",
    ):
        ancestry_permutation_valid = ancestry_permutation_valid & equal_weight_valid
    return {
        "flow_ancestors": tf.gather(ordered_particles, ancestor_indices),
        "selected_row_identities": selected_identities,
        "hilbert_ties": ties,
        "state_map_saturation": saturation,
        "ancestry_unique_count": ancestry_unique_count,
        "ancestry_permutation_valid": ancestry_permutation_valid,
        "equal_weight_valid": equal_weight_valid,
        "equal_weight_error": equal_weight_error,
    }


def _local_initial_observation_marks(
    callbacks: LEDHGenUTModelCallbacks,
    theta: Tensor,
    particles: Tensor,
    observation: Tensor,
    *,
    observation_index: int,
) -> Tensor:
    target_time = callbacks.target_time(observation_index)
    return tf.cast(
        callbacks.model.initial_log_density_parameter_score(
            tf.cast(theta, tf.float64), tf.cast(particles, tf.float64)
        ),
        theta.dtype,
    ) + tf.cast(
        callbacks.model.observation_log_density_parameter_score(
            tf.cast(theta, tf.float64),
            tf.cast(particles, tf.float64),
            tf.cast(observation, tf.float64),
            target_time,
        ),
        theta.dtype,
    )


def standard_pairwise_backward_marks(
    callbacks: LEDHGenUTModelCallbacks,
    theta: Tensor,
    parents: Tensor,
    parent_log_weights: Tensor,
    parent_marks: Tensor,
    children: Tensor,
    observation: Tensor,
    *,
    observation_index: int,
    child_block_size: int = 0,
) -> Tensor:
    """Apply the repository analytical all-parent backward score recursion."""

    if isinstance(callbacks.model, DiagonalLGSSMAnalyticalModel):
        return lgssm_score._target_model_progressive_score_marks(  # noqa: SLF001
            theta,
            parents,
            parent_log_weights,
            parent_marks,
            children,
            observation,
        )

    particle_count = int(parents.shape[0])
    state_dimension = callbacks.state_dimension
    target_time = callbacks.target_time(observation_index)
    theta64 = tf.cast(theta, tf.float64)
    if child_block_size < 0:
        raise ValueError("child_block_size must be nonnegative")
    block_size = child_block_size or particle_count
    if particle_count % block_size != 0:
        raise ValueError("child_block_size must divide the particle count")

    inherited_blocks = []
    for start in range(0, particle_count, block_size):
        child_block = children[start : start + block_size]
        parent_grid = tf.broadcast_to(
            parents[None, :, :], [block_size, particle_count, state_dimension]
        )
        child_grid = tf.broadcast_to(
            child_block[:, None, :],
            [block_size, particle_count, state_dimension],
        )
        flat_parents = tf.reshape(parent_grid, [-1, state_dimension])
        flat_children = tf.reshape(child_grid, [-1, state_dimension])
        transition_log = callbacks.model.transition_log_density(
            theta64,
            tf.cast(flat_parents, tf.float64),
            tf.cast(flat_children, tf.float64),
            target_time,
        )
        transition_score = callbacks.model.transition_log_density_parameter_score(
            theta64,
            tf.cast(flat_parents, tf.float64),
            tf.cast(flat_children, tf.float64),
            target_time,
        )
        transition_log = tf.cast(
            tf.reshape(transition_log, [block_size, particle_count]), theta.dtype
        )
        transition_score = tf.cast(
            tf.reshape(
                transition_score,
                [block_size, particle_count, callbacks.parameter_count],
            ),
            theta.dtype,
        )
        backward = tf.nn.softmax(
            transition_log + parent_log_weights[None, :], axis=1
        )
        inherited_blocks.append(
            tf.einsum(
                "ki,kip->kp",
                backward,
                parent_marks[None, :, :] + transition_score,
            )
        )
    inherited = tf.concat(inherited_blocks, axis=0)
    observation_score = callbacks.model.observation_log_density_parameter_score(
        theta64,
        tf.cast(children, tf.float64),
        tf.cast(observation, tf.float64),
        target_time,
    )
    return inherited + tf.cast(observation_score, theta.dtype)


def _flow_step(
    callbacks: LEDHGenUTModelCallbacks,
    theta: Tensor,
    flow_ancestors: Tensor,
    pre_flow: Tensor,
    observation: Tensor,
    *,
    observation_index: int,
    prior_mean: Tensor,
    transition_covariance: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    observation_fn, jacobian_fn, residual_fn = callbacks.observation_callbacks(
        theta, observation_index
    )
    flow = batched_ledh_flow_core_tf(
        pre_flow_particles=pre_flow[None, :, :],
        ancestors=flow_ancestors[None, :, :],
        observation=callbacks.proposal_observation(theta, observation),
        transition_matrix=callbacks.transition_matrix(theta)[None, :, :],
        transition_covariance=transition_covariance[None, :, :],
        observation_covariance=callbacks.observation_covariance(theta)[None, :, :],
        observation_fn=observation_fn,
        observation_jacobian_fn=jacobian_fn,
        observation_residual_fn=residual_fn,
        prior_mean_fn=lambda _ancestors: prior_mean[None, :, :],
    )
    return (
        flow.post_flow_particles[0],
        flow.pre_flow_log_density[0],
        flow.forward_log_det[0],
    )


def finite_value_standard_score_initial_rqmc(
    callbacks: LEDHGenUTModelCallbacks,
    theta: Tensor,
    observations: Tensor,
    initial_noise: Tensor,
    process_noise: Tensor,
    design: Tensor,
    *,
    ancestry_policy: str = "existing_one_to_one",
    process_ancestor_uniforms: Tensor | None = None,
    state_map_location: Tensor | None = None,
    state_map_scale: Tensor | None = None,
    hilbert_bits: int = 12,
    state_map_policy: str = "adaptive_empirical",
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
    score_child_block_size: int = 0,
    transport_plan_mode: str = "dense",
    transport_row_chunk_size: int | None = None,
    transport_col_chunk_size: int | None = None,
    ancestor_uniform_policy: str = "supplied",
    functional_time_loop: bool = False,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
    marginal_tolerance: float = 1.0e-4,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Run a fixed LEDH proposal with exact correction and standard score."""

    theta = tf.convert_to_tensor(theta, initial_noise.dtype)
    observations = tf.convert_to_tensor(observations, theta.dtype)
    initial_noise = tf.convert_to_tensor(initial_noise, theta.dtype)
    process_noise = tf.convert_to_tensor(process_noise, theta.dtype)
    design = tf.convert_to_tensor(design, theta.dtype)
    horizon = observations.shape[0]
    particle_count = initial_noise.shape[0]
    if horizon is None or particle_count is None:
        raise ValueError("XLA core requires static horizon and particle count")
    expected_process_steps = (
        horizon if callbacks.transition_before_first_observation else horizon - 1
    )
    if theta.shape != (callbacks.parameter_count,):
        raise ValueError("theta shape mismatch")
    if observations.shape != (horizon, callbacks.observation_dimension):
        raise ValueError("observation shape mismatch")
    if initial_noise.shape != (particle_count, callbacks.state_dimension):
        raise ValueError("initial-noise shape mismatch")
    if process_noise.shape != (
        expected_process_steps,
        particle_count,
        callbacks.state_dimension,
    ):
        raise ValueError("process-noise shape mismatch")
    if design.shape != (particle_count, callbacks.state_dimension):
        raise ValueError("residual-design shape mismatch")
    if ancestry_policy not in ANCESTRY_POLICIES:
        raise ValueError(f"unsupported ancestry policy: {ancestry_policy}")
    if state_map_policy not in STATE_MAP_POLICIES:
        raise ValueError(f"unsupported state-map policy: {state_map_policy}")
    if reset_policy not in RESET_POLICIES:
        raise ValueError(f"unsupported reset policy: {reset_policy}")
    if ancestor_uniform_policy not in ("supplied", "stratified"):
        raise ValueError(f"unsupported ancestor uniform policy: {ancestor_uniform_policy}")
    if process_ancestor_uniforms is None:
        process_ancestor_uniforms = tf.zeros(
            [expected_process_steps, particle_count], theta.dtype
        )
    process_ancestor_uniforms = tf.convert_to_tensor(
        process_ancestor_uniforms, theta.dtype
    )
    if process_ancestor_uniforms.shape != (
        expected_process_steps,
        particle_count,
    ):
        raise ValueError("process-ancestor-uniform shape mismatch")
    if state_map_location is None:
        state_map_location = tf.zeros([callbacks.state_dimension], theta.dtype)
    if state_map_scale is None:
        state_map_scale = tf.ones([callbacks.state_dimension], theta.dtype)
    state_map_location = tf.convert_to_tensor(state_map_location, theta.dtype)
    state_map_scale = tf.convert_to_tensor(state_map_scale, theta.dtype)
    if state_map_location.shape != (callbacks.state_dimension,):
        raise ValueError("state-map-location shape mismatch")
    if state_map_scale.shape != (callbacks.state_dimension,):
        raise ValueError("state-map-scale shape mismatch")

    particles = callbacks.push_adapter.initial_value(theta, initial_noise)
    uniform = tf.fill(
        [particle_count], tf.cast(1.0 / float(particle_count), theta.dtype)
    )
    weights = uniform
    total = tf.zeros([], theta.dtype)
    score = tf.zeros([callbacks.parameter_count], theta.dtype)
    valid = tf.constant(True)
    ess_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    maximum_weight_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    reset_residual_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    ancestry_unique_history = tf.TensorArray(
        tf.int32, size=horizon, element_shape=()
    )
    ancestry_permutation_history = tf.TensorArray(
        tf.bool, size=horizon, element_shape=()
    )
    ancestry_equal_weight_error_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
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
    likelihood_increment_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    particle_mean_history = tf.TensorArray(
        theta.dtype,
        size=horizon,
        element_shape=(callbacks.state_dimension,),
    )
    particle_scale_history = tf.TensorArray(
        theta.dtype,
        size=horizon,
        element_shape=(callbacks.state_dimension,),
    )
    reset_gap_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    reset_gap_condition_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    reset_target_condition_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    reset_injected_condition_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    reset_affine_norm_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    reset_ot_residual_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    log_weight_variance_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    negative_susceptible_history = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    dual_cap_covariance_residual_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    dual_cap_active_fraction_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    dual_cap_displacement_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    dual_cap_pre_absolute_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    dual_cap_post_absolute_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )
    dual_cap_radial_scale_history = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=()
    )

    if callbacks.transition_before_first_observation:
        score_marks = tf.cast(
            callbacks.model.initial_log_density_parameter_score(
                tf.cast(theta, tf.float64), tf.cast(particles, tf.float64)
            ),
            theta.dtype,
        )
        start_index = 0
    else:
        prior_mean = callbacks.push_adapter.initial_value(
            theta, tf.zeros_like(initial_noise)
        )
        covariance = callbacks.initial_covariance(theta)
        children, proposal_log, forward_log_det = _flow_step(
            callbacks,
            theta,
            particles,
            particles,
            observations[0],
            observation_index=0,
            prior_mean=prior_mean,
            transition_covariance=covariance,
        )
        target_time = callbacks.target_time(0)
        theta64 = tf.cast(theta, tf.float64)
        target_initial = callbacks.model.initial_log_density(
            theta64, tf.cast(children, tf.float64)
        )
        target_observation = callbacks.model.observation_log_density(
            theta64,
            tf.cast(children, tf.float64),
            tf.cast(observations[0], tf.float64),
            target_time,
        )
        logits = (
            tf.math.log(uniform)
            + tf.cast(target_initial + target_observation, theta.dtype)
            - proposal_log
            + forward_log_det
        )
        increment = tf.reduce_logsumexp(logits)
        step_weights = tf.exp(logits - increment)
        current_marks = _local_initial_observation_marks(
            callbacks,
            theta,
            children,
            observations[0],
            observation_index=0,
        )
        score = tf.einsum("n,np->p", step_weights, current_marks)
        restored = _restore_cloud_primal(
            children,
            step_weights,
            design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
            reset_policy=reset_policy,
            dual_cap_enabled=dual_cap_enabled,
            dual_cap_diagonal_steps=dual_cap_diagonal_steps,
            dual_cap_diagonal_strength=dual_cap_diagonal_strength,
            dual_cap_pairwise_steps=dual_cap_pairwise_steps,
            dual_cap_pairwise_strength=dual_cap_pairwise_strength,
            dual_cap_pairwise_particle_rms_cap=dual_cap_pairwise_particle_rms_cap,
            dual_cap_coordinate_cap=dual_cap_coordinate_cap,
            dual_cap_coordinate_cap_power=dual_cap_coordinate_cap_power,
            trust_region_enabled=trust_region_enabled,
            trust_region_lm_damping=trust_region_lm_damping,
            trust_region_lm_scale_floor=trust_region_lm_scale_floor,
            trust_region_radius=trust_region_radius,
            transport_plan_mode=transport_plan_mode,
            transport_row_chunk_size=transport_row_chunk_size,
            transport_col_chunk_size=transport_col_chunk_size,
            marginal_tolerance=marginal_tolerance,
        )
        restored_marks = _local_initial_observation_marks(
            callbacks,
            theta,
            restored["particles"],
            observations[0],
            observation_index=0,
        )
        step_valid = (
            restored["reset_valid"]
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(score))
            & tf.reduce_all(tf.math.is_finite(restored_marks))
        )
        particles = tf.where(step_valid, restored["particles"], particles)
        score_marks = tf.where(step_valid, restored_marks, current_marks)
        total += tf.where(step_valid, increment, tf.zeros_like(increment))
        valid &= step_valid
        weights = tf.where(
            step_valid,
            uniform if reset_policy != "none" else step_weights,
            weights,
        )
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
        ancestry_permutation_history = ancestry_permutation_history.write(
            0, tf.constant(True)
        )
        ancestry_equal_weight_error_history = (
            ancestry_equal_weight_error_history.write(0, tf.zeros([], theta.dtype))
        )
        ancestry_identity_history = ancestry_identity_history.write(
            0, tf.range(particle_count, dtype=tf.int32)
        )
        hilbert_tie_history = hilbert_tie_history.write(0, tf.zeros([], tf.int32))
        state_map_saturation_history = state_map_saturation_history.write(
            0, tf.zeros([], theta.dtype)
        )
        likelihood_increment_history = likelihood_increment_history.write(
            0, increment
        )
        particle_mean_history = particle_mean_history.write(
            0, tf.reduce_mean(particles, axis=0)
        )
        particle_scale_history = particle_scale_history.write(
            0, tf.sqrt(tf.reduce_mean(tf.square(particles - tf.reduce_mean(particles, axis=0)[None, :]), axis=0))
        )
        reset_gap_history = reset_gap_history.write(
            0, restored["minimum_gap_eigenvalue"]
        )
        reset_gap_condition_history = reset_gap_condition_history.write(
            0, restored["gap_condition_proxy"]
        )
        reset_target_condition_history = reset_target_condition_history.write(
            0, restored["target_condition_proxy"]
        )
        reset_injected_condition_history = reset_injected_condition_history.write(
            0, restored["injected_condition_proxy"]
        )
        reset_affine_norm_history = reset_affine_norm_history.write(
            0, restored["affine_norm"]
        )
        reset_ot_residual_history = reset_ot_residual_history.write(
            0, restored["post_quotient_column_tv_error"]
        )
        log_weight_variance_history = log_weight_variance_history.write(
            0, tf.reduce_mean(tf.square(logits - tf.reduce_mean(logits)))
        )
        negative_susceptible_history = negative_susceptible_history.write(
            0,
            tf.reduce_mean(tf.cast(particles[:, 0::2] < 0.0, theta.dtype)),
        )
        dual_cap_covariance_residual_history = dual_cap_covariance_residual_history.write(
            0, restored["dual_cap_covariance_residual"]
        )
        dual_cap_active_fraction_history = dual_cap_active_fraction_history.write(
            0, restored["fraction_coordinatewise_cap_active"]
        )
        dual_cap_displacement_history = dual_cap_displacement_history.write(
            0, restored["mean_coordinatewise_cap_displacement"]
        )
        dual_cap_pre_absolute_history = dual_cap_pre_absolute_history.write(
            0, restored["maximum_coordinatewise_pre_cap_absolute"]
        )
        dual_cap_post_absolute_history = dual_cap_post_absolute_history.write(
            0, restored["maximum_coordinatewise_post_cap_absolute"]
        )
        dual_cap_radial_scale_history = dual_cap_radial_scale_history.write(
            0, restored["minimum_pairwise_particle_cap_scale"]
        )
        start_index = 1

    def transition_body(
        observation_index,
        particles_value,
        weights_value,
        score_marks_value,
        total_value,
        score_value,
        valid_value,
        ess_history_value,
        maximum_weight_history_value,
        reset_residual_history_value,
        ancestry_unique_history_value,
        ancestry_permutation_history_value,
        ancestry_equal_weight_error_history_value,
        ancestry_identity_history_value,
        hilbert_tie_history_value,
        state_map_saturation_history_value,
        likelihood_increment_history_value,
        particle_mean_history_value,
        particle_scale_history_value,
        reset_gap_history_value,
        reset_gap_condition_history_value,
        reset_target_condition_history_value,
        reset_injected_condition_history_value,
        reset_affine_norm_history_value,
        reset_ot_residual_history_value,
        log_weight_variance_history_value,
        negative_susceptible_history_value,
        dual_cap_covariance_residual_history_value,
        dual_cap_active_fraction_history_value,
        dual_cap_displacement_history_value,
        dual_cap_pre_absolute_history_value,
        dual_cap_post_absolute_history_value,
        dual_cap_radial_scale_history_value,
    ):
        process_index = (
            observation_index
            if callbacks.transition_before_first_observation
            else observation_index - 1
        )
        parents = particles_value
        parent_log_weights = tf.math.log(weights_value)
        parent_marks = score_marks_value
        ancestor_uniforms = process_ancestor_uniforms[process_index]
        if ancestor_uniform_policy == "stratified":
            ancestor_uniforms = (
                tf.cast(tf.range(particle_count), theta.dtype)
                + ancestor_uniforms[0]
            ) / tf.cast(particle_count, theta.dtype)
        ancestor_records = _transition_ancestors(
            parents,
            weights_value,
            ancestor_uniforms,
            ancestry_policy=ancestry_policy,
            state_map_location=state_map_location,
            state_map_scale=state_map_scale,
            hilbert_bits=hilbert_bits,
            state_map_policy=state_map_policy,
        )
        one_to_one_required = ancestry_policy in (
            "hilbert_systematic_equal_weight",
            "hilbert_permutation_one_to_one",
        )
        ancestry_route_valid = (
            ancestor_records["ancestry_permutation_valid"]
            if one_to_one_required
            else tf.constant(True)
        )
        flow_ancestors = ancestor_records["flow_ancestors"]
        target_time = callbacks.target_time(observation_index)
        prior_mean = callbacks.transition_mean(
            theta, flow_ancestors, target_time
        )
        covariance = callbacks.transition_covariance(theta)
        process_cholesky = tf.linalg.cholesky(covariance)
        pre_flow = prior_mean + tf.linalg.matmul(
            process_noise[process_index], process_cholesky, transpose_b=True
        )
        children, proposal_log, forward_log_det = _flow_step(
            callbacks,
            theta,
            flow_ancestors,
            pre_flow,
            observations[observation_index],
            observation_index=observation_index,
            prior_mean=prior_mean,
            transition_covariance=covariance,
        )
        theta64 = tf.cast(theta, tf.float64)
        target_transition = callbacks.model.transition_log_density(
            theta64,
            tf.cast(flow_ancestors, tf.float64),
            tf.cast(children, tf.float64),
            target_time,
        )
        target_observation = callbacks.model.observation_log_density(
            theta64,
            tf.cast(children, tf.float64),
            tf.cast(observations[observation_index], tf.float64),
            target_time,
        )
        current_marks = standard_pairwise_backward_marks(
            callbacks,
            theta,
            parents,
            parent_log_weights,
            parent_marks,
            children,
            observations[observation_index],
            observation_index=observation_index,
            child_block_size=score_child_block_size,
        )
        logits = (
            parent_log_weights
            + tf.cast(target_transition + target_observation, theta.dtype)
            - proposal_log
            + forward_log_det
        )
        increment = tf.reduce_logsumexp(logits)
        step_weights = tf.exp(logits - increment)
        current_score = tf.einsum("n,np->p", step_weights, current_marks)
        restored = _restore_cloud_primal(
            children,
            step_weights,
            design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
            reset_policy=reset_policy,
            dual_cap_enabled=dual_cap_enabled,
            dual_cap_diagonal_steps=dual_cap_diagonal_steps,
            dual_cap_diagonal_strength=dual_cap_diagonal_strength,
            dual_cap_pairwise_steps=dual_cap_pairwise_steps,
            dual_cap_pairwise_strength=dual_cap_pairwise_strength,
            dual_cap_pairwise_particle_rms_cap=dual_cap_pairwise_particle_rms_cap,
            dual_cap_coordinate_cap=dual_cap_coordinate_cap,
            dual_cap_coordinate_cap_power=dual_cap_coordinate_cap_power,
            trust_region_enabled=trust_region_enabled,
            trust_region_lm_damping=trust_region_lm_damping,
            trust_region_lm_scale_floor=trust_region_lm_scale_floor,
            trust_region_radius=trust_region_radius,
            transport_plan_mode=transport_plan_mode,
            transport_row_chunk_size=transport_row_chunk_size,
            transport_col_chunk_size=transport_col_chunk_size,
            marginal_tolerance=marginal_tolerance,
        )
        restored_marks = standard_pairwise_backward_marks(
            callbacks,
            theta,
            parents,
            parent_log_weights,
            parent_marks,
            restored["particles"],
            observations[observation_index],
            observation_index=observation_index,
            child_block_size=score_child_block_size,
        )
        step_valid = (
            restored["reset_valid"]
            & ancestry_route_valid
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(current_score))
            & tf.reduce_all(tf.math.is_finite(restored_marks))
        )
        particles_next = tf.where(step_valid, restored["particles"], particles_value)
        score_marks_next = tf.where(
            step_valid, restored_marks, score_marks_value
        )
        total_next = total_value + tf.where(
            step_valid, increment, tf.zeros_like(increment)
        )
        score_next = tf.where(step_valid, current_score, score_value)
        valid_next = valid_value & step_valid
        ess_history_value = ess_history_value.write(
            observation_index,
            tf.math.reciprocal(tf.reduce_sum(tf.square(step_weights))),
        )
        maximum_weight_history_value = maximum_weight_history_value.write(
            observation_index, tf.reduce_max(step_weights)
        )
        reset_residual_history_value = reset_residual_history_value.write(
            observation_index, restored["mean_residual"]
        )
        ancestry_unique_history_value = ancestry_unique_history_value.write(
            observation_index,
            ancestor_records["ancestry_unique_count"],
        )
        ancestry_permutation_history_value = ancestry_permutation_history_value.write(
            observation_index,
            ancestor_records["ancestry_permutation_valid"],
        )
        ancestry_equal_weight_error_history_value = (
            ancestry_equal_weight_error_history_value.write(
                observation_index, ancestor_records["equal_weight_error"]
            )
        )
        ancestry_identity_history_value = ancestry_identity_history_value.write(
            observation_index, ancestor_records["selected_row_identities"]
        )
        hilbert_tie_history_value = hilbert_tie_history_value.write(
            observation_index, ancestor_records["hilbert_ties"]
        )
        state_map_saturation_history_value = state_map_saturation_history_value.write(
            observation_index, ancestor_records["state_map_saturation"]
        )
        likelihood_increment_history_value = likelihood_increment_history_value.write(
            observation_index, increment
        )
        particle_mean_history_value = particle_mean_history_value.write(
            observation_index, tf.reduce_mean(particles_next, axis=0)
        )
        particle_scale_history_value = particle_scale_history_value.write(
            observation_index,
            tf.sqrt(
                tf.reduce_mean(
                    tf.square(
                        particles_next
                        - tf.reduce_mean(particles_next, axis=0)[None, :]
                    ),
                    axis=0,
                )
            ),
        )
        reset_gap_history_value = reset_gap_history_value.write(
            observation_index, restored["minimum_gap_eigenvalue"]
        )
        reset_gap_condition_history_value = reset_gap_condition_history_value.write(
            observation_index, restored["gap_condition_proxy"]
        )
        reset_target_condition_history_value = reset_target_condition_history_value.write(
            observation_index, restored["target_condition_proxy"]
        )
        reset_injected_condition_history_value = reset_injected_condition_history_value.write(
            observation_index, restored["injected_condition_proxy"]
        )
        reset_affine_norm_history_value = reset_affine_norm_history_value.write(
            observation_index, restored["affine_norm"]
        )
        reset_ot_residual_history_value = reset_ot_residual_history_value.write(
            observation_index, restored["post_quotient_column_tv_error"]
        )
        log_weight_variance_history_value = log_weight_variance_history_value.write(
            observation_index,
            tf.reduce_mean(tf.square(logits - tf.reduce_mean(logits))),
        )
        negative_susceptible_history_value = negative_susceptible_history_value.write(
            observation_index,
            tf.reduce_mean(tf.cast(particles_next[:, 0::2] < 0.0, theta.dtype)),
        )
        dual_cap_covariance_residual_history_value = dual_cap_covariance_residual_history_value.write(
            observation_index, restored["dual_cap_covariance_residual"]
        )
        dual_cap_active_fraction_history_value = dual_cap_active_fraction_history_value.write(
            observation_index, restored["fraction_coordinatewise_cap_active"]
        )
        dual_cap_displacement_history_value = dual_cap_displacement_history_value.write(
            observation_index, restored["mean_coordinatewise_cap_displacement"]
        )
        dual_cap_pre_absolute_history_value = dual_cap_pre_absolute_history_value.write(
            observation_index, restored["maximum_coordinatewise_pre_cap_absolute"]
        )
        dual_cap_post_absolute_history_value = dual_cap_post_absolute_history_value.write(
            observation_index, restored["maximum_coordinatewise_post_cap_absolute"]
        )
        dual_cap_radial_scale_history_value = dual_cap_radial_scale_history_value.write(
            observation_index, restored["minimum_pairwise_particle_cap_scale"]
        )
        weights_next = tf.where(
            step_valid,
            uniform if reset_policy != "none" else step_weights,
            weights_value,
        )
        return (
            observation_index + 1,
            particles_next,
            weights_next,
            score_marks_next,
            total_next,
            score_next,
            valid_next,
            ess_history_value,
            maximum_weight_history_value,
            reset_residual_history_value,
            ancestry_unique_history_value,
            ancestry_permutation_history_value,
            ancestry_equal_weight_error_history_value,
            ancestry_identity_history_value,
            hilbert_tie_history_value,
            state_map_saturation_history_value,
            likelihood_increment_history_value,
            particle_mean_history_value,
            particle_scale_history_value,
            reset_gap_history_value,
            reset_gap_condition_history_value,
            reset_target_condition_history_value,
            reset_injected_condition_history_value,
            reset_affine_norm_history_value,
            reset_ot_residual_history_value,
            log_weight_variance_history_value,
            negative_susceptible_history_value,
            dual_cap_covariance_residual_history_value,
            dual_cap_active_fraction_history_value,
            dual_cap_displacement_history_value,
            dual_cap_pre_absolute_history_value,
            dual_cap_post_absolute_history_value,
            dual_cap_radial_scale_history_value,
        )

    loop_state = (
        tf.constant(start_index, tf.int32),
        particles,
        weights,
        score_marks,
        total,
        score,
        valid,
        ess_history,
        maximum_weight_history,
        reset_residual_history,
        ancestry_unique_history,
        ancestry_permutation_history,
        ancestry_equal_weight_error_history,
        ancestry_identity_history,
        hilbert_tie_history,
        state_map_saturation_history,
        likelihood_increment_history,
        particle_mean_history,
        particle_scale_history,
        reset_gap_history,
        reset_gap_condition_history,
        reset_target_condition_history,
        reset_injected_condition_history,
        reset_affine_norm_history,
        reset_ot_residual_history,
        log_weight_variance_history,
        negative_susceptible_history,
        dual_cap_covariance_residual_history,
        dual_cap_active_fraction_history,
        dual_cap_displacement_history,
        dual_cap_pre_absolute_history,
        dual_cap_post_absolute_history,
        dual_cap_radial_scale_history,
    )
    if functional_time_loop:
        loop_state = tf.while_loop(
            lambda observation_index, *_: observation_index
            < tf.constant(horizon, tf.int32),
            transition_body,
            loop_state,
            parallel_iterations=1,
        )
    else:
        for _observation_index in range(start_index, horizon):
            loop_state = transition_body(*loop_state)

    (
        _,
        particles,
        weights,
        score_marks,
        total,
        score,
        valid,
        ess_history,
        maximum_weight_history,
        reset_residual_history,
        ancestry_unique_history,
        ancestry_permutation_history,
        ancestry_equal_weight_error_history,
        ancestry_identity_history,
        hilbert_tie_history,
        state_map_saturation_history,
        likelihood_increment_history,
        particle_mean_history,
        particle_scale_history,
        reset_gap_history,
        reset_gap_condition_history,
        reset_target_condition_history,
        reset_injected_condition_history,
        reset_affine_norm_history,
        reset_ot_residual_history,
        log_weight_variance_history,
        negative_susceptible_history,
        dual_cap_covariance_residual_history,
        dual_cap_active_fraction_history,
        dual_cap_displacement_history,
        dual_cap_pre_absolute_history,
        dual_cap_post_absolute_history,
        dual_cap_radial_scale_history,
    ) = loop_state

    nan = tf.constant(float("nan"), theta.dtype)
    return (
        tf.where(valid, total, nan),
        tf.where(valid, score, tf.fill([callbacks.parameter_count], nan)),
        {
            "program_valid": valid,
            "ess": ess_history.stack(),
            "maximum_normalized_weight": maximum_weight_history.stack(),
            "reset_mean_residual": reset_residual_history.stack(),
            "ancestry_unique_count": ancestry_unique_history.stack(),
            "ancestry_permutation_valid": ancestry_permutation_history.stack(),
            "ancestry_equal_weight_error": ancestry_equal_weight_error_history.stack(),
            "ancestry_selected_row_identities": ancestry_identity_history.stack(),
            "hilbert_tie_count": hilbert_tie_history.stack(),
            "state_map_saturation_rate": state_map_saturation_history.stack(),
            "likelihood_increment": likelihood_increment_history.stack(),
            "particle_mean": particle_mean_history.stack(),
            "particle_scale": particle_scale_history.stack(),
            "minimum_gap_eigenvalue": reset_gap_history.stack(),
            "gap_condition_proxy": reset_gap_condition_history.stack(),
            "target_condition_proxy": reset_target_condition_history.stack(),
            "injected_condition_proxy": reset_injected_condition_history.stack(),
            "affine_norm": reset_affine_norm_history.stack(),
            "post_quotient_column_tv_error": reset_ot_residual_history.stack(),
            "log_weight_variance": log_weight_variance_history.stack(),
            "negative_susceptible_fraction": negative_susceptible_history.stack(),
            "dual_cap_covariance_residual": dual_cap_covariance_residual_history.stack(),
            "dual_cap_active_fraction": dual_cap_active_fraction_history.stack(),
            "dual_cap_mean_displacement": dual_cap_displacement_history.stack(),
            "dual_cap_pre_cap_absolute": dual_cap_pre_absolute_history.stack(),
            "dual_cap_post_cap_absolute": dual_cap_post_absolute_history.stack(),
            "dual_cap_minimum_radial_scale": dual_cap_radial_scale_history.stack(),
            "reset_route_id": tf.constant(
                2 if trust_region_enabled else (1 if dual_cap_enabled else 0),
                tf.int32,
            ),
            "trust_region_solver_id": tf.constant(
                1 if trust_region_enabled else 0, tf.int32
            ),
            "score_child_block_size": tf.constant(
                score_child_block_size or particle_count, tf.int32
            ),
            "transport_plan_id": tf.constant(
                1 if transport_plan_mode == "streaming" else 0, tf.int32
            ),
            "transport_row_chunk_size": tf.constant(
                transport_row_chunk_size or particle_count, tf.int32
            ),
            "transport_col_chunk_size": tf.constant(
                transport_col_chunk_size or particle_count, tf.int32
            ),
            "final_particles": particles,
            "final_score_marks": score_marks,
        },
    )


__all__ = [
    "ANCESTRY_POLICIES",
    "_transition_ancestors",
    "finite_value_standard_score_initial_rqmc",
    "standard_pairwise_backward_marks",
]

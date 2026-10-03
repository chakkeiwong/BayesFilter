def tf_batched_svd_sigma_point_value_and_score_with_rule(observations: tf.Tensor, model: TFBatchedStructuralStateSpace, derivatives: TFBatchedStructuralFirstDerivatives, *, sigma_rule: TFSigmaPointRule, backend_name: str='tf_batched_svd_sigma_point', placement_floor: tf.Tensor | float=0.0, innovation_floor: tf.Tensor | float=1e-12, rank_tolerance: tf.Tensor | float=1e-12, spectral_gap_tolerance: tf.Tensor | float=1e-08, fixed_null_tolerance: tf.Tensor | float=1e-10, principal_sqrt_reconstruction_tolerance: tf.Tensor | float=1e-10, principal_sqrt_backend: TFPrincipalSqrtBackend='compiled_custom_op', jitter: tf.Tensor | float=0.0, allow_fixed_null_support: bool=False) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Return batched nonlinear SVD sigma-point likelihood and analytic score."""
    batch_dim, parameter_dim, state_dim, innovation_dim, observation_dim = _check_model_derivative_shapes(model, derivatives)
    y = _as_observation_tensor(observations, batch_dim=batch_dim, observation_dim=observation_dim)
    n_timesteps = _observation_count(y)
    aug_dim = state_dim + innovation_dim
    if sigma_rule.dim != aug_dim:
        raise ValueError('sigma_rule dimension must equal state_dim + innovation_dim')
    lagged_observation_contract = model.has_lagged_observation_contract()
    observation_contract = 'lagged_previous_innovation_predicted' if lagged_observation_contract else 'current_predicted_state'
    if lagged_observation_contract and (derivatives.lagged_observation_previous_jacobian_fn is None or derivatives.lagged_observation_innovation_jacobian_fn is None or derivatives.lagged_observation_next_jacobian_fn is None or (derivatives.d_lagged_observation_fn is None)):
        raise ValueError('lagged observation contract requires previous, innovation, next, and parameter derivative hooks')
    mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    covariance = _symmetrize(model.initial_covariance)
    d_mean = tf.convert_to_tensor(derivatives.d_initial_mean, dtype=tf.float64)
    d_covariance = _symmetrize(derivatives.d_initial_covariance)
    innovation_covariance = _symmetrize(model.innovation_covariance)
    d_innovation_covariance = _symmetrize(derivatives.d_innovation_covariance)
    observation_covariance = _symmetrize(model.observation_covariance)
    d_observation_covariance = _symmetrize(derivatives.d_observation_covariance)
    placement_floor = tf.convert_to_tensor(placement_floor, dtype=tf.float64)
    innovation_floor = tf.convert_to_tensor(innovation_floor, dtype=tf.float64)
    rank_tolerance = tf.convert_to_tensor(rank_tolerance, dtype=tf.float64)
    spectral_gap_tolerance = tf.convert_to_tensor(spectral_gap_tolerance, dtype=tf.float64)
    fixed_null_tolerance = tf.convert_to_tensor(fixed_null_tolerance, dtype=tf.float64)
    principal_sqrt_reconstruction_tolerance = tf.convert_to_tensor(principal_sqrt_reconstruction_tolerance, dtype=tf.float64)
    jitter = tf.convert_to_tensor(jitter, dtype=tf.float64)
    obs_identity = tf.eye(observation_dim, dtype=tf.float64)[tf.newaxis, :, :]
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    log_likelihood = tf.zeros([batch_dim], dtype=tf.float64)
    score = tf.zeros([batch_dim, parameter_dim], dtype=tf.float64)
    max_placement_floor_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_innovation_floor_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_placement_roundoff_repair_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_innovation_roundoff_repair_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_placement_classified_invalid_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_innovation_classified_invalid_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_placement_derivative_rhs_nonfinite_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_innovation_derivative_rhs_nonfinite_count = tf.zeros([batch_dim], dtype=tf.int32)
    max_placement_residual = tf.zeros([batch_dim], dtype=tf.float64)
    max_innovation_residual = tf.zeros([batch_dim], dtype=tf.float64)
    max_support_residual = tf.zeros([batch_dim], dtype=tf.float64)
    max_deterministic_residual = tf.zeros([batch_dim], dtype=tf.float64)
    max_factor_derivative_residual = tf.zeros([batch_dim], dtype=tf.float64)
    max_fixed_null_derivative_residual = tf.zeros([batch_dim], dtype=tf.float64)
    max_structural_null_covariance_residual = tf.zeros([batch_dim], dtype=tf.float64)
    max_integration_rank = tf.zeros([batch_dim], dtype=tf.int32)
    max_structural_null_count = tf.zeros([batch_dim], dtype=tf.int32)
    min_placement_eigen_gap = tf.fill([batch_dim], tf.constant(float('inf'), dtype=tf.float64))
    min_innovation_eigen_gap = tf.fill([batch_dim], tf.constant(float('inf'), dtype=tf.float64))
    min_placement_eigenvalue = tf.fill([batch_dim], tf.constant(float('inf'), dtype=tf.float64))
    min_innovation_eigenvalue = tf.fill([batch_dim], tf.constant(float('inf'), dtype=tf.float64))
    max_placement_covariance_abs_entry = tf.zeros([batch_dim], dtype=tf.float64)
    max_innovation_covariance_abs_entry = tf.zeros([batch_dim], dtype=tf.float64)
    max_placement_derivative_covariance_abs_entry = tf.zeros([batch_dim], dtype=tf.float64)
    max_innovation_derivative_covariance_abs_entry = tf.zeros([batch_dim], dtype=tf.float64)
    last_implemented_innovation_covariance = tf.zeros([batch_dim, observation_dim, observation_dim], dtype=tf.float64)
    n_timesteps_tensor = tf.constant(n_timesteps, dtype=tf.int32)

    def _loop_body(t: tf.Tensor, mean: tf.Tensor, covariance: tf.Tensor, d_mean: tf.Tensor, d_covariance: tf.Tensor, log_likelihood: tf.Tensor, score: tf.Tensor, max_placement_floor_count: tf.Tensor, max_innovation_floor_count: tf.Tensor, max_placement_roundoff_repair_count: tf.Tensor, max_innovation_roundoff_repair_count: tf.Tensor, max_placement_classified_invalid_count: tf.Tensor, max_innovation_classified_invalid_count: tf.Tensor, max_placement_derivative_rhs_nonfinite_count: tf.Tensor, max_innovation_derivative_rhs_nonfinite_count: tf.Tensor, max_placement_residual: tf.Tensor, max_innovation_residual: tf.Tensor, max_support_residual: tf.Tensor, max_deterministic_residual: tf.Tensor, max_factor_derivative_residual: tf.Tensor, max_fixed_null_derivative_residual: tf.Tensor, max_structural_null_covariance_residual: tf.Tensor, max_integration_rank: tf.Tensor, max_structural_null_count: tf.Tensor, min_placement_eigen_gap: tf.Tensor, min_innovation_eigen_gap: tf.Tensor, min_placement_eigenvalue: tf.Tensor, min_innovation_eigenvalue: tf.Tensor, max_placement_covariance_abs_entry: tf.Tensor, max_innovation_covariance_abs_entry: tf.Tensor, max_placement_derivative_covariance_abs_entry: tf.Tensor, max_innovation_derivative_covariance_abs_entry: tf.Tensor, last_implemented_innovation_covariance: tf.Tensor, diagnostic_history) -> tuple[tf.Tensor, ...]:
        aug_mean = tf.concat([mean, tf.zeros([batch_dim, innovation_dim], dtype=tf.float64)], axis=1)
        d_aug_mean = tf.concat([d_mean, tf.zeros([batch_dim, parameter_dim, innovation_dim], dtype=tf.float64)], axis=2)
        upper = tf.concat([covariance, tf.zeros([batch_dim, state_dim, innovation_dim], dtype=tf.float64)], axis=2)
        lower = tf.concat([tf.zeros([batch_dim, innovation_dim, state_dim], dtype=tf.float64), innovation_covariance], axis=2)
        aug_covariance = tf.concat([upper, lower], axis=1)
        d_aug_covariance = tf.concat([tf.concat([d_covariance, tf.zeros([batch_dim, parameter_dim, state_dim, innovation_dim], dtype=tf.float64)], axis=3), tf.concat([tf.zeros([batch_dim, parameter_dim, innovation_dim, state_dim], dtype=tf.float64), d_innovation_covariance], axis=3)], axis=2)
        if backend_name == 'tf_principal_sqrt_ukf':
            if allow_fixed_null_support:
                raise ValueError('tf_principal_sqrt_ukf is strict-SPD-only in Phase 3 and does not support structural null branches')
            placement = _checked_batched_principal_sqrt_factor_first_derivatives(aug_covariance, d_aug_covariance, singular_floor=placement_floor, fixed_null_tolerance=fixed_null_tolerance, lyapunov_tolerance=principal_sqrt_reconstruction_tolerance, factor_backend=principal_sqrt_backend, label='principal-sqrt sigma-point placement')
        else:
            placement = _checked_batched_smooth_eigh_factor_first_derivatives(aug_covariance, d_aug_covariance, singular_floor=placement_floor, rank_tolerance=rank_tolerance, spectral_gap_tolerance=spectral_gap_tolerance, fixed_null_tolerance=fixed_null_tolerance, label='SVD sigma-point placement', allow_fixed_null_support=allow_fixed_null_support)
        point_offsets = tf.einsum('ra,bda->brd', sigma_rule.offsets, placement.factor)
        aug_points = aug_mean[:, tf.newaxis, :] + point_offsets
        d_point_offsets = tf.einsum('rd,bpad->bpra', sigma_rule.offsets, placement.d_factor)
        d_aug_points = d_aug_mean[:, :, tf.newaxis, :] + d_point_offsets
        previous_points = aug_points[:, :, :state_dim]
        innovation_points = aug_points[:, :, state_dim:]
        d_previous_points = d_aug_points[:, :, :, :state_dim]
        d_innovation_points = d_aug_points[:, :, :, state_dim:]
        predicted_points = model.transition(previous_points, innovation_points)
        transition_state_jacobian = derivatives.transition_state_jacobian_fn(previous_points, innovation_points)
        transition_innovation_jacobian = derivatives.transition_innovation_jacobian_fn(previous_points, innovation_points)
        d_transition_param = derivatives.d_transition_fn(previous_points, innovation_points)
        _validate_static_shape(transition_state_jacobian, (batch_dim, sigma_rule.point_count, state_dim, state_dim), 'transition_state_jacobian')
        _validate_static_shape(transition_innovation_jacobian, (batch_dim, sigma_rule.point_count, state_dim, innovation_dim), 'transition_innovation_jacobian')
        _validate_static_shape(d_transition_param, (batch_dim, parameter_dim, sigma_rule.point_count, state_dim), 'd_transition')
        d_predicted_points = _matvec_points(transition_state_jacobian, d_previous_points) + _matvec_points(transition_innovation_jacobian, d_innovation_points) + d_transition_param
        residuals = model.deterministic_residual(previous_points, innovation_points, predicted_points)
        deterministic_residual = tf.zeros([batch_dim], dtype=tf.float64) if residuals.shape[-1] == 0 else tf.reduce_max(tf.abs(residuals), axis=[1, 2])
        predicted_mean = tf.einsum('r,brn->bn', sigma_rule.mean_weights, predicted_points)
        d_predicted_mean = tf.einsum('r,bprn->bpn', sigma_rule.mean_weights, d_predicted_points)
        centered_x = predicted_points - predicted_mean[:, tf.newaxis, :]
        d_centered_x = d_predicted_points - d_predicted_mean[:, :, tf.newaxis, :]
        predicted_covariance = _weighted_covariance(centered_x, sigma_rule.covariance_weights)
        d_predicted_covariance = _weighted_covariance_first_derivatives(centered_x, d_centered_x, sigma_rule.covariance_weights)
        if lagged_observation_contract:
            observation_points = model.observe_structural(previous_points, innovation_points, predicted_points)
            observation_previous_jacobian = derivatives.lagged_observation_previous_jacobian_fn(previous_points, innovation_points, predicted_points)
            observation_innovation_jacobian = derivatives.lagged_observation_innovation_jacobian_fn(previous_points, innovation_points, predicted_points)
            observation_next_jacobian = derivatives.lagged_observation_next_jacobian_fn(previous_points, innovation_points, predicted_points)
            d_observation_param = derivatives.d_lagged_observation_fn(previous_points, innovation_points, predicted_points)
            _validate_static_shape(observation_previous_jacobian, (batch_dim, sigma_rule.point_count, observation_dim, state_dim), 'lagged_observation_previous_jacobian')
            _validate_static_shape(observation_innovation_jacobian, (batch_dim, sigma_rule.point_count, observation_dim, innovation_dim), 'lagged_observation_innovation_jacobian')
            _validate_static_shape(observation_next_jacobian, (batch_dim, sigma_rule.point_count, observation_dim, state_dim), 'lagged_observation_next_jacobian')
            _validate_static_shape(d_observation_param, (batch_dim, parameter_dim, sigma_rule.point_count, observation_dim), 'd_lagged_observation')
            d_observation_points = _matvec_points(observation_previous_jacobian, d_previous_points) + _matvec_points(observation_innovation_jacobian, d_innovation_points) + _matvec_points(observation_next_jacobian, d_predicted_points) + d_observation_param
        else:
            observation_points = model.observe(predicted_points)
            observation_state_jacobian = derivatives.observation_state_jacobian_fn(predicted_points)
            d_observation_param = derivatives.d_observation_fn(predicted_points)
            _validate_static_shape(observation_state_jacobian, (batch_dim, sigma_rule.point_count, observation_dim, state_dim), 'observation_state_jacobian')
            _validate_static_shape(d_observation_param, (batch_dim, parameter_dim, sigma_rule.point_count, observation_dim), 'd_observation')
            d_observation_points = _matvec_points(observation_state_jacobian, d_predicted_points) + d_observation_param
        observation_mean = tf.einsum('r,brm->bm', sigma_rule.mean_weights, observation_points)
        d_observation_mean = tf.einsum('r,bprm->bpm', sigma_rule.mean_weights, d_observation_points)
        centered_y = observation_points - observation_mean[:, tf.newaxis, :]
        d_centered_y = d_observation_points - d_observation_mean[:, :, tf.newaxis, :]
        raw_innovation_covariance = _symmetrize(_weighted_covariance(centered_y, sigma_rule.covariance_weights) + observation_covariance + jitter * obs_identity)
        d_raw_innovation_covariance = _symmetrize(_weighted_covariance_first_derivatives(centered_y, d_centered_y, sigma_rule.covariance_weights) + d_observation_covariance)
        if backend_name == 'tf_principal_sqrt_ukf':
            innovation_factor = _checked_batched_principal_sqrt_factor_first_derivatives(raw_innovation_covariance, d_raw_innovation_covariance, singular_floor=innovation_floor, fixed_null_tolerance=fixed_null_tolerance, lyapunov_tolerance=principal_sqrt_reconstruction_tolerance, factor_backend=principal_sqrt_backend, label='principal-sqrt sigma-point innovation')
        else:
            innovation_factor = _checked_batched_smooth_eigh_factor_first_derivatives(raw_innovation_covariance, d_raw_innovation_covariance, singular_floor=innovation_floor, rank_tolerance=rank_tolerance, spectral_gap_tolerance=spectral_gap_tolerance, fixed_null_tolerance=fixed_null_tolerance, label='SVD sigma-point innovation', allow_fixed_null_support=False)
        cross_covariance = tf.einsum('brn,r,brm->bnm', centered_x, sigma_rule.covariance_weights, centered_y)
        d_cross_covariance = tf.einsum('r,bprn,brm->bpnm', sigma_rule.covariance_weights, d_centered_x, centered_y) + tf.einsum('r,brn,bprm->bpnm', sigma_rule.covariance_weights, centered_x, d_centered_y)
        innovation = _observation_at_time(y, t) - observation_mean
        d_innovation = -d_observation_mean
        if backend_name == 'tf_principal_sqrt_ukf':
            innovation_cholesky = tf.linalg.cholesky(innovation_factor.implemented_covariance)
            solve_innovation = _batched_cholesky_solve(innovation_cholesky, innovation)
            innovation_precision = _batched_cholesky_solve(innovation_cholesky, tf.tile(obs_identity, [batch_dim, 1, 1]))
            log_det = _batched_cholesky_logdet(innovation_cholesky)
        else:
            solve_innovation = _batched_eigh_solve(innovation_factor.eigenvectors, innovation_factor.floored_eigenvalues, innovation)
            innovation_precision = _batched_eigh_solve(innovation_factor.eigenvectors, innovation_factor.floored_eigenvalues, tf.tile(obs_identity, [batch_dim, 1, 1]))
            log_det = _batched_eigh_logdet(innovation_factor.floored_eigenvalues)
        mahalanobis = tf.reduce_sum(innovation * solve_innovation, axis=-1)
        contribution = -0.5 * (tf.cast(observation_dim, tf.float64) * tf.math.log(two_pi) + log_det + mahalanobis)
        trace_terms = tf.einsum('bij,bpji->bp', innovation_precision, d_raw_innovation_covariance)
        innovation_derivative_terms = tf.einsum('bpm,bm->bp', d_innovation, solve_innovation)
        covariance_quadratic_terms = tf.einsum('bm,bpmn,bn->bp', solve_innovation, d_raw_innovation_covariance, solve_innovation)
        score = score - 0.5 * (trace_terms + 2.0 * innovation_derivative_terms - covariance_quadratic_terms)
        kalman_gain = tf.matmul(cross_covariance, innovation_precision)
        rhs = tf.linalg.matrix_transpose(d_cross_covariance) - tf.matmul(d_raw_innovation_covariance, tf.linalg.matrix_transpose(kalman_gain)[:, tf.newaxis, :, :])
        if backend_name == 'tf_principal_sqrt_ukf':
            d_kalman_gain = tf.linalg.matrix_transpose(_batched_cholesky_solve(innovation_cholesky, rhs))
        else:
            d_kalman_gain = tf.linalg.matrix_transpose(_batched_eigh_solve(innovation_factor.eigenvectors, innovation_factor.floored_eigenvalues, rhs))
        mean = predicted_mean + tf.einsum('bnm,bm->bn', kalman_gain, innovation)
        d_mean = d_predicted_mean + tf.einsum('bpnm,bm->bpn', d_kalman_gain, innovation) + tf.einsum('bnm,bpm->bpn', kalman_gain, d_innovation)
        covariance = _symmetrize(predicted_covariance - kalman_gain @ innovation_factor.implemented_covariance @ tf.linalg.matrix_transpose(kalman_gain))
        d_covariance = _symmetrize(d_predicted_covariance - tf.einsum('bpnm,bml,bkl->bpnk', d_kalman_gain, innovation_factor.implemented_covariance, kalman_gain) - tf.einsum('bnm,bpml,bkl->bpnk', kalman_gain, d_raw_innovation_covariance, kalman_gain) - tf.einsum('bnm,bml,bpkl->bpnk', kalman_gain, innovation_factor.implemented_covariance, d_kalman_gain))
        log_likelihood = log_likelihood + contribution
        active = placement.eigenvalues > rank_tolerance
        rank = tf.reduce_sum(tf.cast(active, tf.int32), axis=-1)
        null_weights = tf.cast(tf.logical_not(active), tf.float64)
        null_projector = placement.eigenvectors @ tf.linalg.diag(null_weights) @ tf.linalg.matrix_transpose(placement.eigenvectors)
        support_residual = tf.reduce_max(tf.linalg.norm(point_offsets @ null_projector, axis=-1), axis=-1)
        max_placement_floor_count = tf.maximum(max_placement_floor_count, placement.floor_count)
        max_innovation_floor_count = tf.maximum(max_innovation_floor_count, innovation_factor.floor_count)
        max_placement_roundoff_repair_count = tf.maximum(max_placement_roundoff_repair_count, placement.roundoff_repair_count)
        max_innovation_roundoff_repair_count = tf.maximum(max_innovation_roundoff_repair_count, innovation_factor.roundoff_repair_count)
        max_placement_classified_invalid_count = tf.maximum(max_placement_classified_invalid_count, placement.classified_invalid_count)
        max_innovation_classified_invalid_count = tf.maximum(max_innovation_classified_invalid_count, innovation_factor.classified_invalid_count)
        max_placement_derivative_rhs_nonfinite_count = tf.maximum(max_placement_derivative_rhs_nonfinite_count, placement.derivative_rhs_nonfinite_count)
        max_innovation_derivative_rhs_nonfinite_count = tf.maximum(max_innovation_derivative_rhs_nonfinite_count, innovation_factor.derivative_rhs_nonfinite_count)
        max_placement_residual = tf.maximum(max_placement_residual, placement.psd_projection_residual)
        max_innovation_residual = tf.maximum(max_innovation_residual, innovation_factor.psd_projection_residual)
        max_support_residual = tf.maximum(max_support_residual, support_residual)
        max_deterministic_residual = tf.maximum(max_deterministic_residual, deterministic_residual)
        max_factor_derivative_residual = tf.maximum(max_factor_derivative_residual, tf.maximum(placement.derivative_reconstruction_residual, innovation_factor.derivative_reconstruction_residual))
        max_fixed_null_derivative_residual = tf.maximum(max_fixed_null_derivative_residual, placement.fixed_null_derivative_residual)
        max_structural_null_covariance_residual = tf.maximum(max_structural_null_covariance_residual, placement.structural_null_covariance_residual)
        max_integration_rank = tf.maximum(max_integration_rank, rank)
        max_structural_null_count = tf.maximum(max_structural_null_count, placement.structural_null_count)
        min_placement_eigen_gap = tf.minimum(min_placement_eigen_gap, placement.min_eigen_gap)
        min_innovation_eigen_gap = tf.minimum(min_innovation_eigen_gap, innovation_factor.min_eigen_gap)
        min_placement_eigenvalue = tf.minimum(min_placement_eigenvalue, placement.min_eigenvalue)
        min_innovation_eigenvalue = tf.minimum(min_innovation_eigenvalue, innovation_factor.min_eigenvalue)
        max_placement_covariance_abs_entry = tf.maximum(max_placement_covariance_abs_entry, placement.max_abs_covariance_entry)
        max_innovation_covariance_abs_entry = tf.maximum(max_innovation_covariance_abs_entry, innovation_factor.max_abs_covariance_entry)
        max_placement_derivative_covariance_abs_entry = tf.maximum(max_placement_derivative_covariance_abs_entry, placement.max_abs_derivative_covariance_entry)
        max_innovation_derivative_covariance_abs_entry = tf.maximum(max_innovation_derivative_covariance_abs_entry, innovation_factor.max_abs_derivative_covariance_entry)
        last_implemented_innovation_covariance = innovation_factor.implemented_covariance
        diagnostic_history = (diagnostic_history[0].write(t, tf.cast(aug_covariance, tf.float64)), diagnostic_history[1].write(t, tf.cast(placement.min_eigenvalue, tf.float64)), diagnostic_history[2].write(t, tf.cast(placement.classified_invalid_count, tf.float64)))
        return (t + tf.constant(1, dtype=tf.int32), mean, covariance, d_mean, d_covariance, log_likelihood, score, max_placement_floor_count, max_innovation_floor_count, max_placement_roundoff_repair_count, max_innovation_roundoff_repair_count, max_placement_classified_invalid_count, max_innovation_classified_invalid_count, max_placement_derivative_rhs_nonfinite_count, max_innovation_derivative_rhs_nonfinite_count, max_placement_residual, max_innovation_residual, max_support_residual, max_deterministic_residual, max_factor_derivative_residual, max_fixed_null_derivative_residual, max_structural_null_covariance_residual, max_integration_rank, max_structural_null_count, min_placement_eigen_gap, min_innovation_eigen_gap, min_placement_eigenvalue, min_innovation_eigenvalue, max_placement_covariance_abs_entry, max_innovation_covariance_abs_entry, max_placement_derivative_covariance_abs_entry, max_innovation_derivative_covariance_abs_entry, last_implemented_innovation_covariance, diagnostic_history)
    _t, mean, covariance, d_mean, d_covariance, log_likelihood, score, max_placement_floor_count, max_innovation_floor_count, max_placement_roundoff_repair_count, max_innovation_roundoff_repair_count, max_placement_classified_invalid_count, max_innovation_classified_invalid_count, max_placement_derivative_rhs_nonfinite_count, max_innovation_derivative_rhs_nonfinite_count, max_placement_residual, max_innovation_residual, max_support_residual, max_deterministic_residual, max_factor_derivative_residual, max_fixed_null_derivative_residual, max_structural_null_covariance_residual, max_integration_rank, max_structural_null_count, min_placement_eigen_gap, min_innovation_eigen_gap, min_placement_eigenvalue, min_innovation_eigenvalue, max_placement_covariance_abs_entry, max_innovation_covariance_abs_entry, max_placement_derivative_covariance_abs_entry, max_innovation_derivative_covariance_abs_entry, last_implemented_innovation_covariance, diagnostic_history = tf.while_loop(lambda t, *_unused: t < n_timesteps_tensor, _loop_body, (tf.constant(0, dtype=tf.int32), mean, covariance, d_mean, d_covariance, log_likelihood, score, max_placement_floor_count, max_innovation_floor_count, max_placement_roundoff_repair_count, max_innovation_roundoff_repair_count, max_placement_classified_invalid_count, max_innovation_classified_invalid_count, max_placement_derivative_rhs_nonfinite_count, max_innovation_derivative_rhs_nonfinite_count, max_placement_residual, max_innovation_residual, max_support_residual, max_deterministic_residual, max_factor_derivative_residual, max_fixed_null_derivative_residual, max_structural_null_covariance_residual, max_integration_rank, max_structural_null_count, min_placement_eigen_gap, min_innovation_eigen_gap, min_placement_eigenvalue, min_innovation_eigenvalue, max_placement_covariance_abs_entry, max_innovation_covariance_abs_entry, max_placement_derivative_covariance_abs_entry, max_innovation_derivative_covariance_abs_entry, last_implemented_innovation_covariance, (tf.TensorArray(tf.float64, size=n_timesteps, clear_after_read=False), tf.TensorArray(tf.float64, size=n_timesteps, clear_after_read=False), tf.TensorArray(tf.float64, size=n_timesteps, clear_after_read=False))), parallel_iterations=1)
    total_classified_invalid_count = max_placement_classified_invalid_count + max_innovation_classified_invalid_count
    total_derivative_rhs_nonfinite_count = max_placement_derivative_rhs_nonfinite_count + max_innovation_derivative_rhs_nonfinite_count
    classified_invalid_mask = total_classified_invalid_count > 0
    checked_value = tf.where(classified_invalid_mask, tf.fill(tf.shape(log_likelihood), tf.constant(_PRINCIPAL_SQRT_CLASSIFIED_INVALID_LOG_PROB, tf.float64)), log_likelihood)
    checked_score = tf.where(classified_invalid_mask[:, tf.newaxis], tf.zeros_like(score), score)
    checked_value = tf.where(tf.math.is_finite(checked_value), checked_value, tf.fill(tf.shape(checked_value), tf.constant(_PRINCIPAL_SQRT_CLASSIFIED_INVALID_LOG_PROB, tf.float64)))
    checked_score = tf.where(tf.math.is_finite(checked_score), checked_score, tf.zeros_like(checked_score))
    nonfinite_value_gradient_mask = tf.logical_or(tf.logical_not(tf.math.is_finite(log_likelihood)), tf.reduce_any(tf.logical_not(tf.math.is_finite(score)), axis=-1))
    checked_value = tf.where(nonfinite_value_gradient_mask, tf.fill(tf.shape(checked_value), tf.constant(_PRINCIPAL_SQRT_CLASSIFIED_INVALID_LOG_PROB, tf.float64)), checked_value)
    checked_score = tf.where(nonfinite_value_gradient_mask[:, tf.newaxis], tf.zeros_like(checked_score), checked_score)
    checked_value = tf.debugging.check_numerics(checked_value, 'blocked_nonfinite_value: batched SVD sigma-point value is nonfinite')
    checked_score = tf.debugging.check_numerics(checked_score, 'blocked_nonfinite_score: batched SVD sigma-point score is nonfinite')
    classified_invalid_count = total_classified_invalid_count + tf.cast(tf.logical_and(nonfinite_value_gradient_mask, tf.logical_not(classified_invalid_mask)), tf.int32)
    valid_count = tf.cast(tf.logical_not(tf.logical_or(classified_invalid_mask, nonfinite_value_gradient_mask)), tf.int32)
    roundoff_repair_count = max_placement_roundoff_repair_count + max_innovation_roundoff_repair_count
    target_row_class_code = tf.where(classified_invalid_count > 0, tf.fill(tf.shape(classified_invalid_count), tf.constant(2, dtype=tf.int32)), tf.where(roundoff_repair_count > 0, tf.fill(tf.shape(roundoff_repair_count), tf.constant(1, dtype=tf.int32)), tf.fill(tf.shape(roundoff_repair_count), tf.constant(0, dtype=tf.int32))))
    diagnostics = {'backend': tf.constant(backend_name), 'rule': tf.constant(sigma_rule.name), 'observation_contract': tf.constant(observation_contract), 'observation_contract_runtime_selected': tf.constant(True), 'augmented_dim': tf.constant(aug_dim, dtype=tf.int32), 'point_count': tf.constant(sigma_rule.point_count, dtype=tf.int32), 'polynomial_degree': tf.constant(sigma_rule.polynomial_degree, dtype=tf.int32), 'max_integration_rank': max_integration_rank, 'structural_null_count': max_structural_null_count, 'support_residual': max_support_residual, 'deterministic_residual': max_deterministic_residual, 'min_placement_eigen_gap': min_placement_eigen_gap, 'min_innovation_eigen_gap': min_innovation_eigen_gap, 'min_placement_eigenvalue': min_placement_eigenvalue, 'min_innovation_eigenvalue': min_innovation_eigenvalue, 'max_placement_covariance_abs_entry': max_placement_covariance_abs_entry, 'max_innovation_covariance_abs_entry': max_innovation_covariance_abs_entry, 'max_placement_derivative_covariance_abs_entry': max_placement_derivative_covariance_abs_entry, 'max_innovation_derivative_covariance_abs_entry': max_innovation_derivative_covariance_abs_entry, 'factor_derivative_reconstruction_residual': max_factor_derivative_residual, 'fixed_null_derivative_residual': max_fixed_null_derivative_residual, 'structural_null_covariance_residual': max_structural_null_covariance_residual, 'placement_psd_projection_residual': max_placement_residual, 'innovation_psd_projection_residual': max_innovation_residual, 'placement_floor_count': max_placement_floor_count, 'innovation_floor_count': max_innovation_floor_count, 'placement_roundoff_repair_count': max_placement_roundoff_repair_count, 'innovation_roundoff_repair_count': max_innovation_roundoff_repair_count, 'placement_classified_invalid_count': max_placement_classified_invalid_count, 'innovation_classified_invalid_count': max_innovation_classified_invalid_count, 'placement_derivative_rhs_nonfinite_count': max_placement_derivative_rhs_nonfinite_count, 'innovation_derivative_rhs_nonfinite_count': max_innovation_derivative_rhs_nonfinite_count, 'principal_sqrt_target_valid_count': valid_count, 'principal_sqrt_target_roundoff_repair_count': roundoff_repair_count, 'principal_sqrt_target_classified_invalid_count': classified_invalid_count, 'principal_sqrt_target_derivative_rhs_nonfinite_count': total_derivative_rhs_nonfinite_count, 'principal_sqrt_target_row_class_code': target_row_class_code, 'principal_sqrt_target_row_class_legend': tf.constant('0=valid,1=roundoff_repaired,2=classified_invalid'), 'principal_sqrt_classified_invalid_log_prob': tf.constant(_PRINCIPAL_SQRT_CLASSIFIED_INVALID_LOG_PROB, dtype=tf.float64), 'principal_sqrt_covariance_failure_policy': tf.constant('strict_spd_with_roundoff_repair_and_classified_invalid_finite_reject'), 'principal_sqrt_roundoff_repair_threshold': tf.constant(_PRINCIPAL_SQRT_ROUNDOFF_TOLERANCE, dtype=tf.float64), 'principal_sqrt_roundoff_repair_max_abs_entry': tf.constant(_PRINCIPAL_SQRT_MAX_ABS_ENTRY_FOR_REPAIR, dtype=tf.float64), 'implemented_innovation_covariance': last_implemented_innovation_covariance, 'derivative_branch': tf.constant('strict_spd_principal_sqrt' if backend_name == 'tf_principal_sqrt_ukf' else 'structural_fixed_support_no_active_floor' if allow_fixed_null_support else 'smooth_simple_spectrum_no_active_floor'), 'derivative_method': tf.constant('analytic_first_order_principal_sqrt_sylvester' if backend_name == 'tf_principal_sqrt_ukf' else 'analytic_first_order_structural_fixed_support' if allow_fixed_null_support else 'analytic_first_order_smooth_branch'), 'derivative_provider': tf.constant(derivatives.name), 'principal_sqrt_backend': tf.constant(principal_sqrt_backend)}
    return (checked_value, checked_score, diagnostics, tuple((buffer.stack() for buffer in diagnostic_history)))

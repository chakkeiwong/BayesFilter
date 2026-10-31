"""Li-Coates Algorithm 1 LEDH PF-PF with per-particle UKF covariance state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import tensorflow as tf

from bayesfilter.linear.svd_factor_tf import eigh_solve, floor_count, psd_eigh, symmetrize
from bayesfilter.nonlinear.sigma_points_tf import (
    TFSigmaPointRule,
    tf_svd_sigma_point_placement,
    tf_unit_sigma_point_rule,
)
from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    DTYPE,
    normalize_log_weights_tf,
    weighted_mean_and_variance_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import sinkhorn_resample_tf


METHOD_GENERATION = "li_coates_algorithm1_ukf_covariance_lifecycle"
FLOW_SOURCE_ROUTE = "li_coates_2017_algorithm1_ledh_pfpf"
COVARIANCE_ROUTE = "per_particle_ukf_prediction_update"
FLOW_ANCHOR_ROUTE = "zero_noise_transition"
PREVIOUS_LEDHPFPF_OT_EVIDENCE_STATUS = "quarantined"
DEFAULT_COVARIANCE_FLOOR = 1e-10
DEFAULT_RANK_TOLERANCE = 1e-12
OT_ANNEALED_COVARIANCE_CARRY_ROUTE = "ot_annealed_transport_covariance_carry"
OT_SINKHORN_COVARIANCE_CARRY_ROUTE = "ot_sinkhorn_barycentric_covariance_carry"
OT_COVARIANCE_CARRY_ROUTE = "same_transport_barycentric_covariance_carry"
OT_PFPF_CORRECTION_ROUTE = "algorithm1_pfpf_corrected_log_weight_pre_resampling"
OT_CANONICAL_TRANSPORT_CONVENTION = "target_by_source_row_stochastic"


TransitionMeanFn = Callable[[tf.Tensor, int], tf.Tensor]
TransitionSampleFn = Callable[[tf.Tensor, int, int], tf.Tensor]
TransitionLogDensityFn = Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor]
ObservationMeanFn = Callable[[tf.Tensor, int], tf.Tensor]
ObservationJacobianFn = Callable[[tf.Tensor, int], tf.Tensor]
ObservationLogDensityFn = Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor]
ProcessNoiseCovarianceFn = Callable[[tf.Tensor, int], tf.Tensor]
ObservationCovarianceFn = Callable[[int], tf.Tensor]


@dataclass(frozen=True)
class UKFSingleStepResult:
    mean: tf.Tensor
    covariance: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]


@dataclass(frozen=True)
class LedhAlg1TimeStepResult:
    pre_flow_particles: tf.Tensor
    post_flow_particles: tf.Tensor
    predicted_means: tf.Tensor
    predicted_covariances: tf.Tensor
    updated_means: tf.Tensor
    updated_covariances: tf.Tensor
    auxiliary_anchors: tf.Tensor
    auxiliary_terminal_states: tf.Tensor
    flow_matrices_by_particle_step: tf.Tensor
    flow_offsets_by_particle_step: tf.Tensor
    pseudo_time_steps: tf.Tensor
    forward_log_det: tf.Tensor
    pre_flow_log_density: tf.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class LedhPFPFAlg1UKFTFResult:
    method_id: str
    route_identifiers: Mapping[str, str]
    seed: int
    num_particles: int
    log_likelihood_estimate: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    particle_covariances_by_time: tf.Tensor
    predicted_covariances_by_time: tf.Tensor
    corrected_log_weights_by_time: tf.Tensor
    ess_by_time: tf.Tensor
    resampling_count: int
    resampling_diagnostics: list[dict[str, Any]]
    finite: bool


_TWO_PI = tf.constant(6.2831853071795864769, dtype=DTYPE)


def algorithm1_route_identifiers(
    *,
    resampling_route: str = "none",
) -> dict[str, str]:
    """Return the source-route identifiers required by the reviewed plan."""
    route = {
        "method_generation": METHOD_GENERATION,
        "flow_source_route": FLOW_SOURCE_ROUTE,
        "covariance_route": COVARIANCE_ROUTE,
        "flow_anchor_route": FLOW_ANCHOR_ROUTE,
        "resampling_route": str(resampling_route),
        "previous_ledh_pfpf_ot_evidence_status": PREVIOUS_LEDHPFPF_OT_EVIDENCE_STATUS,
    }
    if resampling_route in {
        OT_ANNEALED_COVARIANCE_CARRY_ROUTE,
        OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
    }:
        route.update(
            {
                "route_variant": "p8h_sv_scalar_graph_ot_resampled_alg1",
                "transport_method": (
                    "annealed_transport"
                    if resampling_route == OT_ANNEALED_COVARIANCE_CARRY_ROUTE
                    else "fixed_target_sinkhorn"
                ),
                "covariance_carry_route": OT_COVARIANCE_CARRY_ROUTE,
                "pfpf_correction_route": OT_PFPF_CORRECTION_ROUTE,
                "canonical_transport_matrix_convention": OT_CANONICAL_TRANSPORT_CONVENTION,
                "relaxed_resampling_not_categorical": "true",
                "p8g_no_resampling_evidence_status": "quarantined_historical_diagnostic_only",
            }
        )
    return route


def validate_algorithm1_route_identifiers(route: Mapping[str, str]) -> None:
    """Fail closed when an artifact claims Algorithm 1 but names an old route."""

    if route.get("method_generation") != METHOD_GENERATION:
        return
    expected = algorithm1_route_identifiers(
        resampling_route=route.get("resampling_route", "none")
    )
    for key in (
        "flow_source_route",
        "covariance_route",
        "flow_anchor_route",
        "previous_ledh_pfpf_ot_evidence_status",
    ):
        if route.get(key) != expected[key]:
            raise ValueError(f"Algorithm 1 route has wrong {key}: {route.get(key)!r}")
    if "ledh_pfpf_ot" in str(route.get("flow_source_route", "")):
        raise ValueError("old LEDH-PFPF-OT route cannot claim Algorithm 1 generation")


def ukf_predict_additive_tf(
    *,
    previous_state: tf.Tensor,
    previous_covariance: tf.Tensor,
    transition_mean_fn: TransitionMeanFn,
    process_noise_covariance: tf.Tensor,
    time_index: int,
    sigma_rule: TFSigmaPointRule | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    rank_tolerance: float | tf.Tensor = DEFAULT_RANK_TOLERANCE,
) -> UKFSingleStepResult:
    """Per-particle additive-noise UKF prediction for Algorithm 1."""

    previous_state = tf.reshape(tf.cast(previous_state, DTYPE), [-1])
    previous_covariance = symmetrize(previous_covariance)
    process_noise_covariance = symmetrize(process_noise_covariance)
    rule = sigma_rule or tf_unit_sigma_point_rule(
        int(previous_state.shape[0]),
        rule="unscented",
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
    sigma_points, placement = tf_svd_sigma_point_placement(
        previous_state,
        previous_covariance,
        rule,
        singular_floor=tf.cast(covariance_floor, DTYPE),
        rank_tolerance=tf.cast(rank_tolerance, DTYPE),
    )
    predicted_points = tf.cast(transition_mean_fn(sigma_points, int(time_index)), DTYPE)
    predicted_mean = tf.linalg.matvec(
        tf.transpose(predicted_points),
        rule.mean_weights,
    )
    centered = predicted_points - predicted_mean[tf.newaxis, :]
    raw_covariance = _weighted_covariance(centered, rule.covariance_weights)
    implemented_covariance, covariance_diag = stabilize_covariance_tf(
        raw_covariance + process_noise_covariance,
        covariance_floor=covariance_floor,
    )
    diagnostics = {
        "ukf_rule": tf.constant(rule.name),
        "ukf_point_count": tf.constant(rule.point_count, dtype=tf.int32),
        "ukf_alpha": tf.constant(alpha, dtype=DTYPE),
        "ukf_beta": tf.constant(beta, dtype=DTYPE),
        "ukf_kappa": tf.constant(kappa, dtype=DTYPE),
        "placement_floor_count": placement.floor_count,
        "placement_psd_projection_residual": placement.psd_projection_residual,
        "prediction_floor_count": covariance_diag["floor_count"],
        "prediction_min_raw_eigenvalue": covariance_diag["min_raw_eigenvalue"],
        "prediction_psd_projection_residual": covariance_diag["psd_projection_residual"],
    }
    return UKFSingleStepResult(
        mean=predicted_mean,
        covariance=implemented_covariance,
        diagnostics=diagnostics,
    )


def ukf_update_additive_tf(
    *,
    predicted_mean: tf.Tensor,
    predicted_covariance: tf.Tensor,
    observation: tf.Tensor,
    observation_mean_fn: ObservationMeanFn,
    observation_covariance: tf.Tensor,
    time_index: int,
    sigma_rule: TFSigmaPointRule | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    rank_tolerance: float | tf.Tensor = DEFAULT_RANK_TOLERANCE,
) -> UKFSingleStepResult:
    """Per-particle additive-noise UKF update for Algorithm 1."""

    predicted_mean = tf.reshape(tf.cast(predicted_mean, DTYPE), [-1])
    predicted_covariance = symmetrize(predicted_covariance)
    observation = tf.reshape(tf.cast(observation, DTYPE), [-1])
    observation_covariance = symmetrize(observation_covariance)
    state_dim = int(predicted_mean.shape[0])
    rule = sigma_rule or tf_unit_sigma_point_rule(
        state_dim,
        rule="unscented",
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
    sigma_points, placement = tf_svd_sigma_point_placement(
        predicted_mean,
        predicted_covariance,
        rule,
        singular_floor=tf.cast(covariance_floor, DTYPE),
        rank_tolerance=tf.cast(rank_tolerance, DTYPE),
    )
    placed_covariance = placement.implemented_covariance
    observation_points = tf.cast(observation_mean_fn(sigma_points, int(time_index)), DTYPE)
    observation_mean = tf.linalg.matvec(
        tf.transpose(observation_points),
        rule.mean_weights,
    )
    centered_x = sigma_points - predicted_mean[tf.newaxis, :]
    centered_y = observation_points - observation_mean[tf.newaxis, :]
    innovation_covariance_raw = (
        _weighted_covariance(centered_y, rule.covariance_weights)
        + observation_covariance
    )
    innovation_covariance, innovation_diag = stabilize_covariance_tf(
        innovation_covariance_raw,
        covariance_floor=covariance_floor,
    )
    cross_covariance = tf.transpose(centered_x) @ (
        centered_y * rule.covariance_weights[:, tf.newaxis]
    )
    eigenvalues, floored, eigenvectors, implemented_innovation, _residual = psd_eigh(
        innovation_covariance,
        tf.cast(covariance_floor, DTYPE),
    )
    innovation_precision = eigh_solve(
        eigenvectors,
        floored,
        tf.eye(int(observation.shape[0]), dtype=DTYPE),
    )
    kalman_gain = cross_covariance @ innovation_precision
    innovation = observation - observation_mean
    updated_mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
    updated_covariance_raw = (
        placed_covariance
        - kalman_gain
        @ implemented_innovation
        @ tf.transpose(kalman_gain)
    )
    updated_covariance, update_diag = stabilize_covariance_tf(
        updated_covariance_raw,
        covariance_floor=covariance_floor,
    )
    diagnostics = {
        "ukf_rule": tf.constant(rule.name),
        "ukf_point_count": tf.constant(rule.point_count, dtype=tf.int32),
        "ukf_alpha": tf.constant(alpha, dtype=DTYPE),
        "ukf_beta": tf.constant(beta, dtype=DTYPE),
        "ukf_kappa": tf.constant(kappa, dtype=DTYPE),
        "placement_floor_count": placement.floor_count,
        "placement_psd_projection_residual": placement.psd_projection_residual,
        "innovation_floor_count": floor_count(eigenvalues, tf.cast(covariance_floor, DTYPE)),
        "innovation_min_raw_eigenvalue": innovation_diag["min_raw_eigenvalue"],
        "innovation_psd_projection_residual": innovation_diag["psd_projection_residual"],
        "update_floor_count": update_diag["floor_count"],
        "update_min_raw_eigenvalue": update_diag["min_raw_eigenvalue"],
        "update_psd_projection_residual": update_diag["psd_projection_residual"],
    }
    return UKFSingleStepResult(
        mean=updated_mean,
        covariance=updated_covariance,
        diagnostics=diagnostics,
    )


def li_coates_ledh_alg1_time_step_tf(
    *,
    ancestors: tf.Tensor,
    previous_covariances: tf.Tensor,
    pre_flow_particles: tf.Tensor,
    observation: tf.Tensor,
    transition_mean_fn: TransitionMeanFn,
    transition_log_density_fn: TransitionLogDensityFn,
    observation_mean_fn: ObservationMeanFn,
    observation_jacobian_fn: ObservationJacobianFn,
    process_noise_covariance_fn: ProcessNoiseCovarianceFn,
    observation_covariance_fn: ObservationCovarianceFn,
    time_index: int,
    pseudo_time_steps: tf.Tensor,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    rank_tolerance: float | tf.Tensor = DEFAULT_RANK_TOLERANCE,
) -> LedhAlg1TimeStepResult:
    """Run one source-form Algorithm 1 LEDH PF-PF time step."""

    ancestors = tf.cast(ancestors, DTYPE)
    previous_covariances = tf.cast(previous_covariances, DTYPE)
    pre_flow_particles = tf.cast(pre_flow_particles, DTYPE)
    observation = tf.reshape(tf.cast(observation, DTYPE), [-1])
    pseudo_time_steps = validate_pseudo_time_steps_tf(pseudo_time_steps)
    num_particles = int(ancestors.shape[0])
    state_dim = int(ancestors.shape[1])
    obs_covariance = symmetrize(observation_covariance_fn(int(time_index)))
    identity = tf.eye(state_dim, dtype=DTYPE)

    predicted_means = []
    predicted_covariances = []
    updated_means = []
    updated_covariances = []
    auxiliary_anchors = []
    auxiliary_terminal_states = []
    flow_matrices_by_particle_step = []
    flow_offsets_by_particle_step = []
    post_flow_particles = []
    forward_log_dets = []
    min_predicted_eigenvalues = []
    max_prediction_floor_counts = []
    determinant_signs = []

    for i in range(num_particles):
        ancestor_i = ancestors[i]
        previous_covariance_i = symmetrize(previous_covariances[i])
        process_covariance_i = symmetrize(
            process_noise_covariance_fn(ancestor_i, int(time_index))
        )
        prediction = ukf_predict_additive_tf(
            previous_state=ancestor_i,
            previous_covariance=previous_covariance_i,
            transition_mean_fn=transition_mean_fn,
            process_noise_covariance=process_covariance_i,
            time_index=int(time_index),
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            covariance_floor=covariance_floor,
            rank_tolerance=rank_tolerance,
        )
        predicted_means.append(prediction.mean)
        predicted_covariances.append(prediction.covariance)
        max_prediction_floor_counts.append(prediction.diagnostics["prediction_floor_count"])
        min_predicted_eigenvalues.append(
            tf.reduce_min(tf.linalg.eigvalsh(symmetrize(prediction.covariance)))
        )

        anchor_i = tf.reshape(
            transition_mean_fn(ancestor_i[tf.newaxis, :], int(time_index))[0],
            [-1],
        )
        auxiliary_state_i = tf.identity(anchor_i)
        eta_i = tf.reshape(pre_flow_particles[i], [-1])
        log_det_i = tf.constant(0.0, dtype=DTYPE)
        lambda_value = tf.constant(0.0, dtype=DTYPE)
        sign_product_i = tf.constant(1.0, dtype=DTYPE)
        a_matrices_i = []
        b_vectors_i = []
        for eps in tf.unstack(pseudo_time_steps, axis=0):
            lambda_value = lambda_value + eps
            coefficients = ledh_alg1_coefficients_tf(
                auxiliary_state=auxiliary_state_i,
                zero_noise_anchor=anchor_i,
                predicted_covariance=prediction.covariance,
                observation=observation,
                observation_mean_fn=observation_mean_fn,
                observation_jacobian_fn=observation_jacobian_fn,
                observation_covariance=obs_covariance,
                lambda_value=lambda_value,
                time_index=int(time_index),
                covariance_floor=covariance_floor,
            )
            a_matrix = coefficients["A"]
            b_vector = coefficients["b"]
            a_matrices_i.append(a_matrix)
            b_vectors_i.append(b_vector)
            auxiliary_state_i = auxiliary_state_i + eps * (
                tf.linalg.matvec(a_matrix, auxiliary_state_i) + b_vector
            )
            eta_i = eta_i + eps * (tf.linalg.matvec(a_matrix, eta_i) + b_vector)
            step_map = identity + eps * a_matrix
            sign, log_abs_det = tf.linalg.slogdet(step_map)
            sign_product_i = sign_product_i * sign
            log_det_i = log_det_i + log_abs_det
        update = ukf_update_additive_tf(
            predicted_mean=prediction.mean,
            predicted_covariance=prediction.covariance,
            observation=observation,
            observation_mean_fn=observation_mean_fn,
            observation_covariance=obs_covariance,
            time_index=int(time_index),
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            covariance_floor=covariance_floor,
            rank_tolerance=rank_tolerance,
        )
        updated_means.append(update.mean)
        updated_covariances.append(update.covariance)
        auxiliary_anchors.append(anchor_i)
        auxiliary_terminal_states.append(auxiliary_state_i)
        flow_matrices_by_particle_step.append(tf.stack(a_matrices_i, axis=0))
        flow_offsets_by_particle_step.append(tf.stack(b_vectors_i, axis=0))
        post_flow_particles.append(eta_i)
        forward_log_dets.append(log_det_i)
        determinant_signs.append(sign_product_i)

    predicted_means_tensor = tf.stack(predicted_means, axis=0)
    predicted_covariances_tensor = tf.stack(predicted_covariances, axis=0)
    updated_means_tensor = tf.stack(updated_means, axis=0)
    updated_covariances_tensor = tf.stack(updated_covariances, axis=0)
    anchors_tensor = tf.stack(auxiliary_anchors, axis=0)
    auxiliary_terminal_tensor = tf.stack(auxiliary_terminal_states, axis=0)
    flow_matrices_tensor = tf.stack(flow_matrices_by_particle_step, axis=0)
    flow_offsets_tensor = tf.stack(flow_offsets_by_particle_step, axis=0)
    post_flow_tensor = tf.stack(post_flow_particles, axis=0)
    forward_log_det_tensor = tf.stack(forward_log_dets, axis=0)
    pre_flow_log_density = tf.cast(
        transition_log_density_fn(pre_flow_particles, ancestors, int(time_index)),
        DTYPE,
    )
    diagnostics = {
        "component_id": "li_coates_algorithm1_ledh_pfpf_ukf_time_step",
        "method_generation": METHOD_GENERATION,
        "flow_source_route": FLOW_SOURCE_ROUTE,
        "covariance_route": COVARIANCE_ROUTE,
        "flow_anchor_route": FLOW_ANCHOR_ROUTE,
        "pseudo_time_step_count": int(pseudo_time_steps.shape[0]),
        "pseudo_time_sum": _float(tf.reduce_sum(pseudo_time_steps)),
        "min_predicted_covariance_eigenvalue": _float(
            tf.reduce_min(tf.stack(min_predicted_eigenvalues, axis=0))
        ),
        "max_prediction_floor_count": int(
            tf.reduce_max(tf.stack(max_prediction_floor_counts, axis=0)).numpy()
        ),
        "min_forward_log_det": _float(tf.reduce_min(forward_log_det_tensor)),
        "max_forward_log_det": _float(tf.reduce_max(forward_log_det_tensor)),
        "finite_forward_log_det": _finite_bool(forward_log_det_tensor),
        "finite_post_flow": _finite_bool(post_flow_tensor),
        "finite_auxiliary_terminal": _finite_bool(auxiliary_terminal_tensor),
        "finite_predicted_covariances": _finite_bool(predicted_covariances_tensor),
        "finite_updated_covariances": _finite_bool(updated_covariances_tensor),
        "min_determinant_sign_product": _float(tf.reduce_min(tf.stack(determinant_signs))),
        "same_affine_trace_available": True,
        "backend": "tensorflow",
    }
    if not diagnostics["finite_forward_log_det"]:
        raise FloatingPointError("Algorithm 1 LEDH determinant product is non-finite")
    if not diagnostics["finite_post_flow"]:
        raise FloatingPointError("Algorithm 1 LEDH flow emitted non-finite particles")
    return LedhAlg1TimeStepResult(
        pre_flow_particles=pre_flow_particles,
        post_flow_particles=post_flow_tensor,
        predicted_means=predicted_means_tensor,
        predicted_covariances=predicted_covariances_tensor,
        updated_means=updated_means_tensor,
        updated_covariances=updated_covariances_tensor,
        auxiliary_anchors=anchors_tensor,
        auxiliary_terminal_states=auxiliary_terminal_tensor,
        flow_matrices_by_particle_step=flow_matrices_tensor,
        flow_offsets_by_particle_step=flow_offsets_tensor,
        pseudo_time_steps=pseudo_time_steps,
        forward_log_det=forward_log_det_tensor,
        pre_flow_log_density=pre_flow_log_density,
        diagnostics=diagnostics,
    )


def li_coates_ledh_alg1_time_step_vectorized_particles_tf(
    *,
    ancestors: tf.Tensor,
    previous_covariances: tf.Tensor,
    pre_flow_particles: tf.Tensor,
    observation: tf.Tensor,
    transition_mean_fn: TransitionMeanFn,
    transition_log_density_fn: TransitionLogDensityFn,
    observation_mean_fn: ObservationMeanFn,
    observation_jacobian_fn: ObservationJacobianFn,
    process_noise_covariance_fn: ProcessNoiseCovarianceFn,
    observation_covariance_fn: ObservationCovarianceFn,
    time_index: int,
    pseudo_time_steps: tf.Tensor,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    rank_tolerance: float | tf.Tensor = DEFAULT_RANK_TOLERANCE,
) -> LedhAlg1TimeStepResult:
    """Run one Algorithm 1 time step with a TensorFlow particle-batched route."""

    ancestors = tf.cast(ancestors, DTYPE)
    previous_covariances = tf.cast(previous_covariances, DTYPE)
    pre_flow_particles = tf.cast(pre_flow_particles, DTYPE)
    observation = tf.reshape(tf.cast(observation, DTYPE), [-1])
    pseudo_time_steps = validate_pseudo_time_steps_tf(pseudo_time_steps)
    state_dim = int(ancestors.shape[1])
    obs_covariance = symmetrize(observation_covariance_fn(int(time_index)))
    identity = tf.eye(state_dim, dtype=DTYPE)

    def one_particle(inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        ancestor_i, previous_covariance_i, pre_flow_i = inputs
        ancestor_i = tf.reshape(tf.cast(ancestor_i, DTYPE), [-1])
        previous_covariance_i = symmetrize(previous_covariance_i)
        eta_i = tf.reshape(tf.cast(pre_flow_i, DTYPE), [-1])
        process_covariance_i = symmetrize(
            process_noise_covariance_fn(ancestor_i, int(time_index))
        )
        prediction = ukf_predict_additive_tf(
            previous_state=ancestor_i,
            previous_covariance=previous_covariance_i,
            transition_mean_fn=transition_mean_fn,
            process_noise_covariance=process_covariance_i,
            time_index=int(time_index),
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            covariance_floor=covariance_floor,
            rank_tolerance=rank_tolerance,
        )
        anchor_i = tf.reshape(
            transition_mean_fn(ancestor_i[tf.newaxis, :], int(time_index))[0],
            [-1],
        )
        auxiliary_state_i = tf.identity(anchor_i)
        log_det_i = tf.constant(0.0, dtype=DTYPE)
        lambda_value = tf.constant(0.0, dtype=DTYPE)
        sign_product_i = tf.constant(1.0, dtype=DTYPE)
        a_matrices_i = []
        b_vectors_i = []
        for eps in tf.unstack(pseudo_time_steps, axis=0):
            lambda_value = lambda_value + eps
            coefficients = ledh_alg1_coefficients_tf(
                auxiliary_state=auxiliary_state_i,
                zero_noise_anchor=anchor_i,
                predicted_covariance=prediction.covariance,
                observation=observation,
                observation_mean_fn=observation_mean_fn,
                observation_jacobian_fn=observation_jacobian_fn,
                observation_covariance=obs_covariance,
                lambda_value=lambda_value,
                time_index=int(time_index),
                covariance_floor=covariance_floor,
            )
            a_matrix = coefficients["A"]
            b_vector = coefficients["b"]
            a_matrices_i.append(a_matrix)
            b_vectors_i.append(b_vector)
            auxiliary_state_i = auxiliary_state_i + eps * (
                tf.linalg.matvec(a_matrix, auxiliary_state_i) + b_vector
            )
            eta_i = eta_i + eps * (tf.linalg.matvec(a_matrix, eta_i) + b_vector)
            step_map = identity + eps * a_matrix
            sign, log_abs_det = tf.linalg.slogdet(step_map)
            sign_product_i = sign_product_i * sign
            log_det_i = log_det_i + log_abs_det
        update = ukf_update_additive_tf(
            predicted_mean=prediction.mean,
            predicted_covariance=prediction.covariance,
            observation=observation,
            observation_mean_fn=observation_mean_fn,
            observation_covariance=obs_covariance,
            time_index=int(time_index),
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            covariance_floor=covariance_floor,
            rank_tolerance=rank_tolerance,
        )
        min_predicted_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(symmetrize(prediction.covariance))
        )
        return (
            prediction.mean,
            prediction.covariance,
            update.mean,
            update.covariance,
            anchor_i,
            auxiliary_state_i,
            tf.stack(a_matrices_i, axis=0),
            tf.stack(b_vectors_i, axis=0),
            eta_i,
            log_det_i,
            prediction.diagnostics["prediction_floor_count"],
            min_predicted_eigenvalue,
            sign_product_i,
        )

    (
        predicted_means_tensor,
        predicted_covariances_tensor,
        updated_means_tensor,
        updated_covariances_tensor,
        anchors_tensor,
        auxiliary_terminal_tensor,
        flow_matrices_tensor,
        flow_offsets_tensor,
        post_flow_tensor,
        forward_log_det_tensor,
        prediction_floor_counts,
        min_predicted_eigenvalues,
        determinant_signs,
    ) = tf.vectorized_map(
        one_particle,
        (ancestors, previous_covariances, pre_flow_particles),
    )
    pre_flow_log_density = tf.cast(
        transition_log_density_fn(pre_flow_particles, ancestors, int(time_index)),
        DTYPE,
    )
    diagnostics = {
        "component_id": "li_coates_algorithm1_ledh_pfpf_ukf_time_step",
        "method_generation": METHOD_GENERATION,
        "flow_source_route": FLOW_SOURCE_ROUTE,
        "covariance_route": COVARIANCE_ROUTE,
        "flow_anchor_route": FLOW_ANCHOR_ROUTE,
        "particle_batch_route": "tf_vectorized_map",
        "pseudo_time_step_count": int(pseudo_time_steps.shape[0]),
        "pseudo_time_sum": _float(tf.reduce_sum(pseudo_time_steps)),
        "min_predicted_covariance_eigenvalue": _float(
            tf.reduce_min(min_predicted_eigenvalues)
        ),
        "max_prediction_floor_count": int(
            tf.reduce_max(prediction_floor_counts).numpy()
        ),
        "min_forward_log_det": _float(tf.reduce_min(forward_log_det_tensor)),
        "max_forward_log_det": _float(tf.reduce_max(forward_log_det_tensor)),
        "finite_forward_log_det": _finite_bool(forward_log_det_tensor),
        "finite_post_flow": _finite_bool(post_flow_tensor),
        "finite_auxiliary_terminal": _finite_bool(auxiliary_terminal_tensor),
        "finite_predicted_covariances": _finite_bool(predicted_covariances_tensor),
        "finite_updated_covariances": _finite_bool(updated_covariances_tensor),
        "min_determinant_sign_product": _float(tf.reduce_min(determinant_signs)),
        "same_affine_trace_available": True,
        "backend": "tensorflow",
    }
    if not diagnostics["finite_forward_log_det"]:
        raise FloatingPointError("Algorithm 1 LEDH determinant product is non-finite")
    if not diagnostics["finite_post_flow"]:
        raise FloatingPointError("Algorithm 1 LEDH flow emitted non-finite particles")
    return LedhAlg1TimeStepResult(
        pre_flow_particles=pre_flow_particles,
        post_flow_particles=post_flow_tensor,
        predicted_means=predicted_means_tensor,
        predicted_covariances=predicted_covariances_tensor,
        updated_means=updated_means_tensor,
        updated_covariances=updated_covariances_tensor,
        auxiliary_anchors=anchors_tensor,
        auxiliary_terminal_states=auxiliary_terminal_tensor,
        flow_matrices_by_particle_step=flow_matrices_tensor,
        flow_offsets_by_particle_step=flow_offsets_tensor,
        pseudo_time_steps=pseudo_time_steps,
        forward_log_det=forward_log_det_tensor,
        pre_flow_log_density=pre_flow_log_density,
        diagnostics=diagnostics,
    )


def ledh_alg1_coefficients_tf(
    *,
    auxiliary_state: tf.Tensor,
    zero_noise_anchor: tf.Tensor,
    predicted_covariance: tf.Tensor,
    observation: tf.Tensor,
    observation_mean_fn: ObservationMeanFn,
    observation_jacobian_fn: ObservationJacobianFn,
    observation_covariance: tf.Tensor,
    lambda_value: tf.Tensor,
    time_index: int,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
) -> Mapping[str, tf.Tensor]:
    """Compute source-form Li-Coates LEDH coefficients for one particle."""

    auxiliary_state = tf.reshape(tf.cast(auxiliary_state, DTYPE), [-1])
    zero_noise_anchor = tf.reshape(tf.cast(zero_noise_anchor, DTYPE), [-1])
    predicted_covariance = symmetrize(predicted_covariance)
    observation = tf.reshape(tf.cast(observation, DTYPE), [-1])
    observation_covariance = symmetrize(observation_covariance)
    h_value = tf.reshape(
        tf.cast(observation_mean_fn(auxiliary_state[tf.newaxis, :], int(time_index))[0], DTYPE),
        [-1],
    )
    h_jacobian = tf.cast(observation_jacobian_fn(auxiliary_state, int(time_index)), DTYPE)
    residual_intercept = h_value - tf.linalg.matvec(h_jacobian, auxiliary_state)
    hpht = h_jacobian @ predicted_covariance @ tf.transpose(h_jacobian)
    lambda_tensor = tf.cast(lambda_value, DTYPE)
    flow_covariance = lambda_tensor * hpht + observation_covariance
    flow_eigenvalues, flow_floored, flow_eigenvectors, _implemented, _residual = psd_eigh(
        flow_covariance,
        tf.cast(covariance_floor, DTYPE),
    )
    solved_h = eigh_solve(flow_eigenvectors, flow_floored, h_jacobian)
    a_matrix = (
        -0.5
        * predicted_covariance
        @ tf.transpose(h_jacobian)
        @ solved_h
    )
    obs_eigenvalues, obs_floored, obs_eigenvectors, _obs_impl, _obs_residual = psd_eigh(
        observation_covariance,
        tf.cast(covariance_floor, DTYPE),
    )
    shifted_observation = observation - residual_intercept
    obs_precision_residual = eigh_solve(obs_eigenvectors, obs_floored, shifted_observation)
    identity = tf.eye(int(auxiliary_state.shape[0]), dtype=DTYPE)
    inner = tf.linalg.matvec(
        (identity + lambda_tensor * a_matrix)
        @ predicted_covariance
        @ tf.transpose(h_jacobian),
        obs_precision_residual,
    ) + tf.linalg.matvec(a_matrix, zero_noise_anchor)
    b_vector = tf.linalg.matvec(identity + 2.0 * lambda_tensor * a_matrix, inner)
    return {
        "A": a_matrix,
        "b": b_vector,
        "H": h_jacobian,
        "e": residual_intercept,
        "flow_floor_count": floor_count(flow_eigenvalues, tf.cast(covariance_floor, DTYPE)),
        "observation_floor_count": floor_count(obs_eigenvalues, tf.cast(covariance_floor, DTYPE)),
    }


def run_ledh_pfpf_alg1_ukf_tf(
    *,
    observations: tf.Tensor,
    initial_sample: Callable[[int, int], tf.Tensor],
    initial_covariance: tf.Tensor,
    transition_sample: TransitionSampleFn,
    transition_mean_fn: TransitionMeanFn,
    transition_log_density_fn: TransitionLogDensityFn,
    observation_mean_fn: ObservationMeanFn,
    observation_jacobian_fn: ObservationJacobianFn,
    observation_log_density_fn: ObservationLogDensityFn,
    process_noise_covariance_fn: ProcessNoiseCovarianceFn,
    observation_covariance_fn: ObservationCovarianceFn,
    seed: int,
    num_particles: int,
    pseudo_time_steps: tf.Tensor | None = None,
    resampling_route: str = "none",
    ess_threshold_ratio: float = 0.5,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    rank_tolerance: float | tf.Tensor = DEFAULT_RANK_TOLERANCE,
    vectorized_particles: bool = False,
    sinkhorn_epsilon: float = 0.5,
    sinkhorn_iterations: int = 80,
    sinkhorn_tolerance: float = 1e-7,
    sinkhorn_epsilon_policy: str = "fixed",
    annealed_scaling: float = 0.9,
    annealed_convergence_threshold: float = 1e-3,
    transport_gradient_mode: str = "filterflow_clipped",
    method_id: str = "ledh_pfpf_alg1_ukf_tf",
) -> LedhPFPFAlg1UKFTFResult:
    """Run Algorithm 1 UKF LEDH with reviewed resampling routes."""

    if resampling_route not in {
        "none",
        "classical_resampling",
        OT_ANNEALED_COVARIANCE_CARRY_ROUTE,
        OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
    }:
        raise ValueError(
            "Algorithm 1 core supports none, classical_resampling, and reviewed P8h OT routes"
        )
    route = algorithm1_route_identifiers(resampling_route=resampling_route)
    validate_algorithm1_route_identifiers(route)
    particles = tf.cast(initial_sample(num_particles, seed), DTYPE)
    initial_covariance = symmetrize(initial_covariance)
    covariances = tf.tile(initial_covariance[tf.newaxis, :, :], [num_particles, 1, 1])
    log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
    pseudo_time = (
        tf.constant([1.0], dtype=DTYPE)
        if pseudo_time_steps is None
        else validate_pseudo_time_steps_tf(pseudo_time_steps)
    )
    filtered_means = []
    filtered_variances = []
    particle_covariances_by_time = []
    predicted_covariances_by_time = []
    corrected_log_weights_by_time = []
    ess_by_time = []
    diagnostics: list[dict[str, Any]] = []
    resampling_count = 0
    log_likelihood = tf.constant(0.0, dtype=DTYPE)

    for t, observation in enumerate(tf.unstack(tf.cast(observations, DTYPE), axis=0)):
        ancestors = tf.identity(particles)
        previous_covariances = tf.identity(covariances)
        pre_flow = tf.cast(transition_sample(ancestors, seed, int(t)), DTYPE)
        step_fn = (
            li_coates_ledh_alg1_time_step_vectorized_particles_tf
            if vectorized_particles
            else li_coates_ledh_alg1_time_step_tf
        )
        step = step_fn(
            ancestors=ancestors,
            previous_covariances=previous_covariances,
            pre_flow_particles=pre_flow,
            observation=observation,
            transition_mean_fn=transition_mean_fn,
            transition_log_density_fn=transition_log_density_fn,
            observation_mean_fn=observation_mean_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            process_noise_covariance_fn=process_noise_covariance_fn,
            observation_covariance_fn=observation_covariance_fn,
            time_index=int(t),
            pseudo_time_steps=pseudo_time,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            covariance_floor=covariance_floor,
            rank_tolerance=rank_tolerance,
        )
        post_flow = step.post_flow_particles
        target_transition = tf.cast(
            transition_log_density_fn(post_flow, ancestors, int(t)),
            DTYPE,
        )
        target_observation = tf.cast(
            observation_log_density_fn(post_flow, observation, int(t)),
            DTYPE,
        )
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - step.pre_flow_log_density
            + step.forward_log_det
        )
        corrected_log_weights_by_time.append(corrected_log_weights)
        weights, incremental = normalize_log_weights_tf(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights)
        mean, variance = weighted_mean_and_variance_tf(post_flow, weights)
        filtered_means.append(mean)
        filtered_variances.append(variance)
        particle_covariances_by_time.append(step.updated_covariances)
        predicted_covariances_by_time.append(step.predicted_covariances)
        ess_by_time.append(ess)
        if not _finite_bool(corrected_log_weights):
            raise FloatingPointError("Algorithm 1 corrected log weights are non-finite")

        normalized_log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE)))
        trigger_resampling = bool((ess < ess_threshold_ratio * num_particles).numpy())
        if resampling_route == "classical_resampling" and bool(
            trigger_resampling
        ):
            indices = tf.random.stateless_categorical(
                tf.reshape(tf.math.log(tf.maximum(weights, 1e-300)), [1, -1]),
                num_particles,
                seed=_seed_pair(seed, 9000 + t),
                dtype=tf.int32,
            )[0]
            particles, covariances = apply_classical_resampling_state_tf(
                particles=post_flow,
                covariances=step.updated_covariances,
                ancestor_indices=indices,
            )
            log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
            resampling_count += 1
            resampling_diag = {
                "resampled": True,
                "resampling_method": "stateless_multinomial_with_covariance_gather",
                "ancestor_indices": [int(v) for v in tf.unstack(indices)],
            }
        elif resampling_route in {
            OT_ANNEALED_COVARIANCE_CARRY_ROUTE,
            OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
        } and trigger_resampling:
            particles, covariances, resampling_diag = apply_ot_resampling_state_tf(
                particles=post_flow,
                covariances=step.updated_covariances,
                weights=weights,
                log_weights=normalized_log_weights,
                resampling_route=resampling_route,
                covariance_floor=covariance_floor,
                sinkhorn_epsilon=sinkhorn_epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_tolerance=sinkhorn_tolerance,
                sinkhorn_epsilon_policy=sinkhorn_epsilon_policy,
                annealed_scaling=annealed_scaling,
                annealed_convergence_threshold=annealed_convergence_threshold,
                transport_gradient_mode=transport_gradient_mode,
            )
            log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
            resampling_count += 1
            resampling_diag.update(
                {
                    "ess_trigger_threshold": float(ess_threshold_ratio),
                    "ess_triggered": True,
                    "resampling_route": resampling_route,
                }
            )
        else:
            particles = post_flow
            covariances = step.updated_covariances
            log_weights = normalized_log_weights
            resampling_diag = {
                "resampled": False,
                "resampling_method": "none",
                "ess_trigger_threshold": float(ess_threshold_ratio),
                "ess_triggered": False,
            }
        diagnostics.append(
            {
                "time_index": int(t),
                "ess": _float(ess),
                "ess_ratio": _float(ess / tf.cast(num_particles, DTYPE)),
                "pfpf_correction": "log_target_transition_plus_observation_minus_q0_plus_forward_logdet",
                "finite_corrected_log_weights": _finite_bool(corrected_log_weights),
                "min_corrected_log_weight": _float(tf.reduce_min(corrected_log_weights)),
                "max_corrected_log_weight": _float(tf.reduce_max(corrected_log_weights)),
                **step.diagnostics,
                **resampling_diag,
            }
        )

    filtered_means_tensor = tf.stack(filtered_means, axis=0)
    filtered_variances_tensor = tf.stack(filtered_variances, axis=0)
    particle_covariances_tensor = tf.stack(particle_covariances_by_time, axis=0)
    predicted_covariances_tensor = tf.stack(predicted_covariances_by_time, axis=0)
    corrected_log_weights_tensor = tf.stack(corrected_log_weights_by_time, axis=0)
    ess_tensor = tf.stack(ess_by_time, axis=0)
    finite = bool(
        tf.math.is_finite(log_likelihood).numpy()
        and _finite_bool(filtered_means_tensor)
        and _finite_bool(filtered_variances_tensor)
        and _finite_bool(particle_covariances_tensor)
        and _finite_bool(predicted_covariances_tensor)
        and _finite_bool(corrected_log_weights_tensor)
        and _finite_bool(ess_tensor)
    )
    return LedhPFPFAlg1UKFTFResult(
        method_id=method_id,
        route_identifiers=route,
        seed=int(seed),
        num_particles=int(num_particles),
        log_likelihood_estimate=log_likelihood,
        filtered_means=filtered_means_tensor,
        filtered_variances=filtered_variances_tensor,
        particle_covariances_by_time=particle_covariances_tensor,
        predicted_covariances_by_time=predicted_covariances_tensor,
        corrected_log_weights_by_time=corrected_log_weights_tensor,
        ess_by_time=ess_tensor,
        resampling_count=int(resampling_count),
        resampling_diagnostics=diagnostics,
        finite=finite,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def _run_ledh_pfpf_alg1_scalar_sv_graph_kernel_tf(
    *,
    flow_observations: tf.Tensor,
    raw_observations: tf.Tensor,
    initial_standard_normal: tf.Tensor,
    transition_standard_normals: tf.Tensor,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    observation_variance: tf.Tensor,
    pseudo_time_steps: tf.Tensor,
    covariance_floor: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """TensorFlow graph kernel for the scalar SV Algorithm 1 no-resampling lane."""

    flow_observations = tf.reshape(tf.cast(flow_observations, DTYPE), [-1])
    raw_observations = tf.reshape(tf.cast(raw_observations, DTYPE), [-1])
    gamma = tf.cast(gamma, DTYPE)
    beta = tf.cast(beta, DTYPE)
    sigma = tf.cast(sigma, DTYPE)
    observation_variance = tf.cast(observation_variance, DTYPE)
    covariance_floor = tf.cast(covariance_floor, DTYPE)
    initial_standard_normal = tf.reshape(tf.cast(initial_standard_normal, DTYPE), [-1])
    transition_standard_normals = tf.cast(transition_standard_normals, DTYPE)
    num_particles = tf.shape(initial_standard_normal)[0]
    pseudo_time_steps = tf.reshape(tf.cast(pseudo_time_steps, DTYPE), [-1])

    initial_covariance = tf.square(sigma) / (1.0 - tf.square(gamma))
    initial_scale = tf.sqrt(initial_covariance)
    particles = initial_scale * initial_standard_normal
    covariances = tf.fill([num_particles], initial_covariance)
    log_weights = tf.fill(
        [num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    horizon = tf.shape(flow_observations)[0]
    filtered_means = tf.TensorArray(DTYPE, size=horizon, clear_after_read=False)
    filtered_variances = tf.TensorArray(DTYPE, size=horizon, clear_after_read=False)
    particle_covariances = tf.TensorArray(DTYPE, size=horizon, clear_after_read=False)
    predicted_covariances = tf.TensorArray(DTYPE, size=horizon, clear_after_read=False)
    corrected_log_weights_by_time = tf.TensorArray(
        DTYPE,
        size=horizon,
        clear_after_read=False,
    )
    ess_by_time = tf.TensorArray(DTYPE, size=horizon, clear_after_read=False)
    log_likelihood = tf.constant(0.0, dtype=DTYPE)

    normal_log_constant = -0.5 * tf.math.log(_TWO_PI)
    sigma_log_constant = normal_log_constant - tf.math.log(sigma)
    transition_variance = tf.square(sigma)
    log_beta = tf.math.log(beta)

    def pseudo_loop(
        observation: tf.Tensor,
        predicted_covariance: tf.Tensor,
        anchor: tf.Tensor,
        eta: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        auxiliary = tf.identity(anchor)
        log_det = tf.zeros_like(eta)
        lambda_value = tf.constant(0.0, dtype=DTYPE)
        j = tf.constant(0, dtype=tf.int32)
        pseudo_count = tf.shape(pseudo_time_steps)[0]

        def pseudo_cond(
            j: tf.Tensor,
            lambda_value: tf.Tensor,
            auxiliary: tf.Tensor,
            eta: tf.Tensor,
            log_det: tf.Tensor,
        ) -> tf.Tensor:
            del lambda_value, auxiliary, eta, log_det
            return j < pseudo_count

        def pseudo_body(
            j: tf.Tensor,
            lambda_value: tf.Tensor,
            auxiliary: tf.Tensor,
            eta: tf.Tensor,
            log_det: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            eps = pseudo_time_steps[j]
            lambda_value = lambda_value + eps
            flow_variance = lambda_value * predicted_covariance + observation_variance
            a_value = -0.5 * predicted_covariance / flow_variance
            inner = (
                (1.0 + lambda_value * a_value)
                * predicted_covariance
                * observation
                / observation_variance
                + a_value * anchor
            )
            b_value = (1.0 + 2.0 * lambda_value * a_value) * inner
            auxiliary = auxiliary + eps * (a_value * auxiliary + b_value)
            eta = eta + eps * (a_value * eta + b_value)
            step_map = 1.0 + eps * a_value
            log_det = log_det + tf.math.log(tf.abs(step_map))
            return j + 1, lambda_value, auxiliary, eta, log_det

        _j, _lambda_value, auxiliary, eta, log_det = tf.while_loop(
            pseudo_cond,
            pseudo_body,
            loop_vars=(j, lambda_value, auxiliary, eta, log_det),
            maximum_iterations=pseudo_count,
        )
        return auxiliary, eta, log_det

    def time_cond(
        t: tf.Tensor,
        particles: tf.Tensor,
        covariances: tf.Tensor,
        log_weights: tf.Tensor,
        log_likelihood: tf.Tensor,
        filtered_means: tf.TensorArray,
        filtered_variances: tf.TensorArray,
        particle_covariances: tf.TensorArray,
        predicted_covariances: tf.TensorArray,
        corrected_log_weights_by_time: tf.TensorArray,
        ess_by_time: tf.TensorArray,
    ) -> tf.Tensor:
        del (
            particles,
            covariances,
            log_weights,
            log_likelihood,
            filtered_means,
            filtered_variances,
            particle_covariances,
            predicted_covariances,
            corrected_log_weights_by_time,
            ess_by_time,
        )
        return t < horizon

    def time_body(
        t: tf.Tensor,
        particles: tf.Tensor,
        covariances: tf.Tensor,
        log_weights: tf.Tensor,
        log_likelihood: tf.Tensor,
        filtered_means: tf.TensorArray,
        filtered_variances: tf.TensorArray,
        particle_covariances: tf.TensorArray,
        predicted_covariances: tf.TensorArray,
        corrected_log_weights_by_time: tf.TensorArray,
        ess_by_time: tf.TensorArray,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
    ]:
        transition_noise = sigma * transition_standard_normals[t]
        ancestors = particles
        pre_flow = gamma * ancestors + transition_noise
        predicted_mean = gamma * ancestors
        predicted_covariance = tf.maximum(
            tf.square(gamma) * covariances + transition_variance,
            covariance_floor,
        )
        flow_observation = flow_observations[t]
        _auxiliary, post_flow, forward_log_det = pseudo_loop(
            flow_observation,
            predicted_covariance,
            predicted_mean,
            pre_flow,
        )

        innovation_variance = tf.maximum(
            predicted_covariance + observation_variance,
            covariance_floor,
        )
        kalman_gain = predicted_covariance / innovation_variance
        updated_covariance = tf.maximum(
            predicted_covariance - tf.square(kalman_gain) * innovation_variance,
            covariance_floor,
        )

        pre_residual = pre_flow - gamma * ancestors
        pre_flow_log_density = (
            sigma_log_constant
            - 0.5 * tf.square(pre_residual) / transition_variance
        )
        target_residual = post_flow - gamma * ancestors
        target_transition = (
            sigma_log_constant
            - 0.5 * tf.square(target_residual) / transition_variance
        )
        log_scale = log_beta + 0.5 * post_flow
        standardized = raw_observations[t] * tf.exp(-log_scale)
        target_observation = (
            normal_log_constant
            - log_scale
            - 0.5 * tf.square(standardized)
        )
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - pre_flow_log_density
            + forward_log_det
        )
        normalizer = tf.reduce_logsumexp(corrected_log_weights)
        weights = tf.exp(corrected_log_weights - normalizer)
        weights = weights / tf.reduce_sum(weights)
        log_likelihood = log_likelihood + normalizer
        ess = 1.0 / tf.reduce_sum(weights * weights)
        mean = tf.reduce_sum(weights * post_flow)
        centered = post_flow - mean
        variance = tf.reduce_sum(weights * centered * centered)
        next_log_weights = tf.math.log(
            tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE))
        )

        filtered_means = filtered_means.write(t, tf.reshape(mean, [1]))
        filtered_variances = filtered_variances.write(t, tf.reshape(variance, [1]))
        particle_covariances = particle_covariances.write(
            t,
            tf.reshape(updated_covariance, [num_particles, 1, 1]),
        )
        predicted_covariances = predicted_covariances.write(
            t,
            tf.reshape(predicted_covariance, [num_particles, 1, 1]),
        )
        corrected_log_weights_by_time = corrected_log_weights_by_time.write(
            t,
            corrected_log_weights,
        )
        ess_by_time = ess_by_time.write(t, ess)
        return (
            t + 1,
            post_flow,
            updated_covariance,
            next_log_weights,
            log_likelihood,
            filtered_means,
            filtered_variances,
            particle_covariances,
            predicted_covariances,
            corrected_log_weights_by_time,
            ess_by_time,
        )

    (
        _t,
        particles,
        covariances,
        log_weights,
        log_likelihood,
        filtered_means,
        filtered_variances,
        particle_covariances,
        predicted_covariances,
        corrected_log_weights_by_time,
        ess_by_time,
    ) = tf.while_loop(
        time_cond,
        time_body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            particles,
            covariances,
            log_weights,
            log_likelihood,
            filtered_means,
            filtered_variances,
            particle_covariances,
            predicted_covariances,
            corrected_log_weights_by_time,
            ess_by_time,
        ),
        maximum_iterations=horizon,
    )
    filtered_means_tensor = filtered_means.stack()
    filtered_variances_tensor = filtered_variances.stack()
    particle_covariances_tensor = particle_covariances.stack()
    predicted_covariances_tensor = predicted_covariances.stack()
    corrected_log_weights_tensor = corrected_log_weights_by_time.stack()
    ess_tensor = ess_by_time.stack()
    finite = tf.reduce_all(
        tf.stack(
            [
                tf.reduce_all(tf.math.is_finite(log_likelihood)),
                tf.reduce_all(tf.math.is_finite(filtered_means_tensor)),
                tf.reduce_all(tf.math.is_finite(filtered_variances_tensor)),
                tf.reduce_all(tf.math.is_finite(particle_covariances_tensor)),
                tf.reduce_all(tf.math.is_finite(predicted_covariances_tensor)),
                tf.reduce_all(tf.math.is_finite(corrected_log_weights_tensor)),
                tf.reduce_all(tf.math.is_finite(ess_tensor)),
            ]
        )
    )
    return {
        "log_likelihood": log_likelihood,
        "filtered_means": filtered_means_tensor,
        "filtered_variances": filtered_variances_tensor,
        "particle_covariances": particle_covariances_tensor,
        "predicted_covariances": predicted_covariances_tensor,
        "corrected_log_weights": corrected_log_weights_tensor,
        "ess": ess_tensor,
        "finite": finite,
        "final_particles": tf.reshape(particles, [num_particles, 1]),
        "final_covariances": tf.reshape(covariances, [num_particles, 1, 1]),
        "final_log_weights": log_weights,
    }


def run_ledh_pfpf_alg1_scalar_sv_graph_tf(
    *,
    flow_observations: tf.Tensor,
    raw_observations: tf.Tensor,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    observation_variance: tf.Tensor,
    seed: int,
    num_particles: int,
    pseudo_time_steps: tf.Tensor | None = None,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    method_id: str = "ledh_pfpf_alg1_scalar_sv_graph_tf",
) -> LedhPFPFAlg1UKFTFResult:
    """Run the P8g scalar-SV graph specialization of Algorithm 1.

    This intentionally covers only the scalar stochastic-volatility row whose
    transition, LEDH flow observation model, and target correction are declared
    in the P8d/P8g benchmark runner.  It is not the generic Algorithm 1 route.
    """

    pseudo_time = (
        tf.constant([1.0], dtype=DTYPE)
        if pseudo_time_steps is None
        else validate_pseudo_time_steps_tf(pseudo_time_steps)
    )
    flow_observations = tf.reshape(tf.cast(flow_observations, DTYPE), [-1])
    horizon = int(flow_observations.shape[0] or tf.shape(flow_observations)[0].numpy())
    initial_standard_normal = tf.random.stateless_normal(
        [int(num_particles)],
        seed=_seed_pair(seed, 110),
        dtype=DTYPE,
    )
    transition_standard_normals = tf.stack(
        [
            tf.random.stateless_normal(
                [int(num_particles)],
                seed=_seed_pair(seed, 1110 + t),
                dtype=DTYPE,
            )
            for t in range(horizon)
        ],
        axis=0,
    )
    kernel_result = _run_ledh_pfpf_alg1_scalar_sv_graph_kernel_tf(
        flow_observations=flow_observations,
        raw_observations=raw_observations,
        initial_standard_normal=initial_standard_normal,
        transition_standard_normals=transition_standard_normals,
        gamma=tf.cast(gamma, DTYPE),
        beta=tf.cast(beta, DTYPE),
        sigma=tf.cast(sigma, DTYPE),
        observation_variance=tf.cast(observation_variance, DTYPE),
        pseudo_time_steps=pseudo_time,
        covariance_floor=tf.cast(covariance_floor, DTYPE),
    )
    route = algorithm1_route_identifiers(resampling_route="none")
    route = {
        **route,
        "graph_specialization_route": "p8g_scalar_sv_graph",
        "time_loop_route": "tf_while_loop",
        "particle_batch_route": "closed_form_scalar_vector_ops",
        "resampling_route": "none",
    }
    return LedhPFPFAlg1UKFTFResult(
        method_id=method_id,
        route_identifiers=route,
        seed=int(seed),
        num_particles=int(num_particles),
        log_likelihood_estimate=kernel_result["log_likelihood"],
        filtered_means=kernel_result["filtered_means"],
        filtered_variances=kernel_result["filtered_variances"],
        particle_covariances_by_time=kernel_result["particle_covariances"],
        predicted_covariances_by_time=kernel_result["predicted_covariances"],
        corrected_log_weights_by_time=kernel_result["corrected_log_weights"],
        ess_by_time=kernel_result["ess"],
        resampling_count=0,
        resampling_diagnostics=[
            {
                "resampled": False,
                "resampling_method": "none",
                "graph_specialization_route": "p8g_scalar_sv_graph",
                "time_loop_route": "tf_while_loop",
                "particle_batch_route": "closed_form_scalar_vector_ops",
                "backend": "tensorflow",
            }
        ],
        finite=bool(kernel_result["finite"].numpy()),
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def _ledh_pfpf_alg1_scalar_sv_graph_loglik_kernel_tf(
    *,
    flow_observations: tf.Tensor,
    raw_observations: tf.Tensor,
    initial_standard_normal: tf.Tensor,
    transition_standard_normals: tf.Tensor,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    observation_variance: tf.Tensor,
    pseudo_time_steps: tf.Tensor,
    covariance_floor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Value-only scalar-SV graph kernel for fixed-randomness gradients."""

    flow_observations = tf.reshape(tf.cast(flow_observations, DTYPE), [-1])
    raw_observations = tf.reshape(tf.cast(raw_observations, DTYPE), [-1])
    initial_standard_normal = tf.reshape(tf.cast(initial_standard_normal, DTYPE), [-1])
    transition_standard_normals = tf.cast(transition_standard_normals, DTYPE)
    gamma = tf.cast(gamma, DTYPE)
    beta = tf.cast(beta, DTYPE)
    sigma = tf.cast(sigma, DTYPE)
    observation_variance = tf.cast(observation_variance, DTYPE)
    pseudo_time_steps = tf.reshape(tf.cast(pseudo_time_steps, DTYPE), [-1])
    covariance_floor = tf.cast(covariance_floor, DTYPE)

    num_particles = tf.shape(initial_standard_normal)[0]
    horizon = tf.shape(flow_observations)[0]
    initial_covariance = tf.square(sigma) / (1.0 - tf.square(gamma))
    particles = tf.sqrt(initial_covariance) * initial_standard_normal
    covariances = tf.fill([num_particles], initial_covariance)
    log_weights = tf.fill(
        [num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    log_likelihood = tf.constant(0.0, dtype=DTYPE)
    ess_min = tf.cast(num_particles, DTYPE)
    ess_sum = tf.constant(0.0, dtype=DTYPE)
    finite = tf.constant(True)

    normal_log_constant = -0.5 * tf.math.log(_TWO_PI)
    sigma_log_constant = normal_log_constant - tf.math.log(sigma)
    transition_variance = tf.square(sigma)
    log_beta = tf.math.log(beta)

    def pseudo_loop(
        observation: tf.Tensor,
        predicted_covariance: tf.Tensor,
        anchor: tf.Tensor,
        eta: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        auxiliary = tf.identity(anchor)
        log_det = tf.zeros_like(eta)
        lambda_value = tf.constant(0.0, dtype=DTYPE)
        j = tf.constant(0, dtype=tf.int32)
        pseudo_count = tf.shape(pseudo_time_steps)[0]

        def pseudo_cond(
            j: tf.Tensor,
            lambda_value: tf.Tensor,
            auxiliary: tf.Tensor,
            eta: tf.Tensor,
            log_det: tf.Tensor,
        ) -> tf.Tensor:
            del lambda_value, auxiliary, eta, log_det
            return j < pseudo_count

        def pseudo_body(
            j: tf.Tensor,
            lambda_value: tf.Tensor,
            auxiliary: tf.Tensor,
            eta: tf.Tensor,
            log_det: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            eps = pseudo_time_steps[j]
            lambda_value = lambda_value + eps
            flow_variance = lambda_value * predicted_covariance + observation_variance
            a_value = -0.5 * predicted_covariance / flow_variance
            inner = (
                (1.0 + lambda_value * a_value)
                * predicted_covariance
                * observation
                / observation_variance
                + a_value * anchor
            )
            b_value = (1.0 + 2.0 * lambda_value * a_value) * inner
            auxiliary = auxiliary + eps * (a_value * auxiliary + b_value)
            eta = eta + eps * (a_value * eta + b_value)
            step_map = 1.0 + eps * a_value
            log_det = log_det + tf.math.log(tf.abs(step_map))
            return j + 1, lambda_value, auxiliary, eta, log_det

        _j, _lambda_value, _auxiliary, eta, log_det = tf.while_loop(
            pseudo_cond,
            pseudo_body,
            loop_vars=(j, lambda_value, auxiliary, eta, log_det),
            maximum_iterations=pseudo_count,
        )
        return eta, log_det

    def time_cond(
        t: tf.Tensor,
        particles: tf.Tensor,
        covariances: tf.Tensor,
        log_weights: tf.Tensor,
        log_likelihood: tf.Tensor,
        ess_min: tf.Tensor,
        ess_sum: tf.Tensor,
        finite: tf.Tensor,
    ) -> tf.Tensor:
        del particles, covariances, log_weights, log_likelihood, ess_min, ess_sum, finite
        return t < horizon

    def time_body(
        t: tf.Tensor,
        particles: tf.Tensor,
        covariances: tf.Tensor,
        log_weights: tf.Tensor,
        log_likelihood: tf.Tensor,
        ess_min: tf.Tensor,
        ess_sum: tf.Tensor,
        finite: tf.Tensor,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        ancestors = particles
        pre_flow = gamma * ancestors + sigma * transition_standard_normals[t]
        predicted_mean = gamma * ancestors
        predicted_covariance = tf.maximum(
            tf.square(gamma) * covariances + transition_variance,
            covariance_floor,
        )
        post_flow, forward_log_det = pseudo_loop(
            flow_observations[t],
            predicted_covariance,
            predicted_mean,
            pre_flow,
        )
        innovation_variance = tf.maximum(
            predicted_covariance + observation_variance,
            covariance_floor,
        )
        kalman_gain = predicted_covariance / innovation_variance
        updated_covariance = tf.maximum(
            predicted_covariance - tf.square(kalman_gain) * innovation_variance,
            covariance_floor,
        )
        pre_residual = pre_flow - gamma * ancestors
        pre_flow_log_density = (
            sigma_log_constant
            - 0.5 * tf.square(pre_residual) / transition_variance
        )
        target_residual = post_flow - gamma * ancestors
        target_transition = (
            sigma_log_constant
            - 0.5 * tf.square(target_residual) / transition_variance
        )
        log_scale = log_beta + 0.5 * post_flow
        standardized = raw_observations[t] * tf.exp(-log_scale)
        target_observation = (
            normal_log_constant
            - log_scale
            - 0.5 * tf.square(standardized)
        )
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - pre_flow_log_density
            + forward_log_det
        )
        normalizer = tf.reduce_logsumexp(corrected_log_weights)
        weights = tf.exp(corrected_log_weights - normalizer)
        weights = weights / tf.reduce_sum(weights)
        ess = 1.0 / tf.reduce_sum(weights * weights)
        next_log_weights = tf.math.log(
            tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE))
        )
        step_finite = tf.reduce_all(
            tf.stack(
                [
                    tf.reduce_all(tf.math.is_finite(post_flow)),
                    tf.reduce_all(tf.math.is_finite(updated_covariance)),
                    tf.reduce_all(tf.math.is_finite(corrected_log_weights)),
                    tf.reduce_all(tf.math.is_finite(ess)),
                    tf.reduce_all(tf.math.is_finite(normalizer)),
                ]
            )
        )
        return (
            t + 1,
            post_flow,
            updated_covariance,
            next_log_weights,
            log_likelihood + normalizer,
            tf.minimum(ess_min, ess),
            ess_sum + ess,
            tf.logical_and(finite, step_finite),
        )

    (
        _t,
        _particles,
        _covariances,
        _log_weights,
        log_likelihood,
        ess_min,
        ess_sum,
        finite,
    ) = tf.while_loop(
        time_cond,
        time_body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            particles,
            covariances,
            log_weights,
            log_likelihood,
            ess_min,
            ess_sum,
            finite,
        ),
        maximum_iterations=horizon,
    )
    ess_mean = ess_sum / tf.cast(horizon, DTYPE)
    finite = tf.logical_and(finite, tf.reduce_all(tf.math.is_finite(log_likelihood)))
    return log_likelihood, ess_min, ess_mean, finite


_ledh_pfpf_alg1_scalar_sv_graph_gradient_loglik_kernel_tf = tf.function(
    _ledh_pfpf_alg1_scalar_sv_graph_loglik_kernel_tf.python_function,
    jit_compile=False,
    reduce_retracing=True,
)


def ledh_pfpf_alg1_scalar_sv_graph_log_likelihood_tf(
    *,
    flow_observations: tf.Tensor,
    raw_observations: tf.Tensor,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    observation_variance: tf.Tensor,
    seed: int,
    num_particles: int,
    pseudo_time_steps: tf.Tensor | None = None,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    jit_compile: bool = True,
) -> Mapping[str, tf.Tensor]:
    """Evaluate the scalar-SV graph objective for fixed-randomness gradients."""

    pseudo_time = (
        tf.constant([1.0], dtype=DTYPE)
        if pseudo_time_steps is None
        else validate_pseudo_time_steps_tf(pseudo_time_steps)
    )
    flow_observations = tf.reshape(tf.cast(flow_observations, DTYPE), [-1])
    horizon = int(flow_observations.shape[0] or tf.shape(flow_observations)[0].numpy())
    initial_standard_normal = tf.random.stateless_normal(
        [int(num_particles)],
        seed=_seed_pair(seed, 110),
        dtype=DTYPE,
    )
    transition_standard_normals = tf.stack(
        [
            tf.random.stateless_normal(
                [int(num_particles)],
                seed=_seed_pair(seed, 1110 + t),
                dtype=DTYPE,
            )
            for t in range(horizon)
        ],
        axis=0,
    )
    kernel = (
        _ledh_pfpf_alg1_scalar_sv_graph_loglik_kernel_tf
        if jit_compile
        else _ledh_pfpf_alg1_scalar_sv_graph_gradient_loglik_kernel_tf
    )
    value, ess_min, ess_mean, finite = kernel(
        flow_observations=flow_observations,
        raw_observations=raw_observations,
        initial_standard_normal=initial_standard_normal,
        transition_standard_normals=transition_standard_normals,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        observation_variance=observation_variance,
        pseudo_time_steps=pseudo_time,
        covariance_floor=tf.cast(covariance_floor, DTYPE),
    )
    return {
        "log_likelihood": value,
        "ess_min": ess_min,
        "ess_mean": ess_mean,
        "finite": finite,
        "route_variant": tf.constant("p8g_sv_scalar_graph"),
        "time_loop_route": tf.constant("tf_while_loop"),
        "particle_batch_route": tf.constant("closed_form_scalar_vector_ops"),
        "randomness_contract": tf.constant("stateless_normals_precomputed_outside_xla"),
    }


def apply_classical_resampling_state_tf(
    *,
    particles: tf.Tensor,
    covariances: tf.Tensor,
    ancestor_indices: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Gather states and Algorithm 1 covariance state with the same ancestry."""

    indices = tf.cast(ancestor_indices, tf.int32)
    return tf.gather(tf.cast(particles, DTYPE), indices), tf.gather(
        tf.cast(covariances, DTYPE),
        indices,
    )


def canonical_transport_from_sinkhorn_coupling_tf(
    coupling: tf.Tensor,
    *,
    floor: float | tf.Tensor = 1e-300,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Convert source-by-target Sinkhorn coupling to A[target, source]."""

    coupling = tf.cast(coupling, DTYPE)
    column_mass = tf.reduce_sum(coupling, axis=0)
    safe_column_mass = tf.maximum(column_mass, tf.cast(floor, DTYPE))
    canonical = tf.transpose(coupling) / safe_column_mass[:, tf.newaxis]
    row_sums = tf.reduce_sum(canonical, axis=1)
    diagnostics = {
        "transport_column_mass_min": tf.reduce_min(column_mass),
        "transport_column_mass_max": tf.reduce_max(column_mass),
        "canonical_transport_row_sum_min": tf.reduce_min(row_sums),
        "canonical_transport_row_sum_max": tf.reduce_max(row_sums),
        "transport_mass_total_residual": tf.abs(tf.reduce_sum(coupling) - 1.0),
        "raw_transport_matrix_convention": tf.constant("source_by_target_coupling"),
        "canonical_transport_matrix_convention": tf.constant(OT_CANONICAL_TRANSPORT_CONVENTION),
    }
    return canonical, diagnostics


def canonical_transport_from_annealed_matrix_tf(
    transport_matrix: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Validate annealed transport as A[target, source]."""

    canonical = tf.cast(transport_matrix, DTYPE)
    row_sums = tf.reduce_sum(canonical, axis=1)
    column_mass = tf.reduce_sum(canonical, axis=0)
    diagnostics = {
        "transport_column_mass_min": tf.reduce_min(column_mass),
        "transport_column_mass_max": tf.reduce_max(column_mass),
        "canonical_transport_row_sum_min": tf.reduce_min(row_sums),
        "canonical_transport_row_sum_max": tf.reduce_max(row_sums),
        "transport_mass_total_residual": tf.abs(
            tf.reduce_sum(canonical) - tf.cast(tf.shape(canonical)[0], DTYPE)
        ),
        "raw_transport_matrix_convention": tf.constant("target_by_source_row_stochastic"),
        "canonical_transport_matrix_convention": tf.constant(OT_CANONICAL_TRANSPORT_CONVENTION),
    }
    return canonical, diagnostics


def carry_covariances_with_canonical_transport_tf(
    *,
    canonical_transport: tf.Tensor,
    covariances: tf.Tensor,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Carry Algorithm 1 covariance state with the same A[target, source] map."""

    canonical_transport = tf.cast(canonical_transport, DTYPE)
    covariances = tf.cast(covariances, DTYPE)
    carried_raw = tf.einsum("ji,iab->jab", canonical_transport, covariances)
    implemented_rows = []
    min_eigenvalues = []
    projection_residuals = []
    for row in tf.unstack(carried_raw, axis=0):
        implemented, diag = stabilize_covariance_tf(
            row,
            covariance_floor=covariance_floor,
        )
        implemented_rows.append(implemented)
        min_eigenvalues.append(diag["min_raw_eigenvalue"])
        projection_residuals.append(diag["psd_projection_residual"])
    carried = tf.stack(implemented_rows, axis=0)
    diagnostics = {
        "covariance_carry_route": tf.constant(OT_COVARIANCE_CARRY_ROUTE),
        "finite_carried_covariances": tf.reduce_all(tf.math.is_finite(carried)),
        "min_carried_covariance_eigenvalue": tf.reduce_min(tf.stack(min_eigenvalues)),
        "covariance_carry_psd_projection_residual": tf.reduce_max(
            tf.stack(projection_residuals)
        ),
        "covariance_transport_shape": tf.shape(canonical_transport),
        "state_transport_shape": tf.shape(canonical_transport),
    }
    return carried, diagnostics


def apply_ot_resampling_state_tf(
    *,
    particles: tf.Tensor,
    covariances: tf.Tensor,
    weights: tf.Tensor,
    log_weights: tf.Tensor,
    resampling_route: str,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
    sinkhorn_epsilon: float = 0.5,
    sinkhorn_iterations: int = 80,
    sinkhorn_tolerance: float = 1e-7,
    annealed_scaling: float = 0.9,
    annealed_convergence_threshold: float = 1e-3,
    transport_gradient_mode: str = "filterflow_clipped",
    canonical_row_sum_tolerance: float = 5e-3,
    sinkhorn_epsilon_policy: str = "fixed",
) -> tuple[tf.Tensor, tf.Tensor, dict[str, Any]]:
    """Apply reviewed P8h relaxed OT resampling and covariance carry."""

    particles = tf.cast(particles, DTYPE)
    covariances = tf.cast(covariances, DTYPE)
    weights = tf.cast(weights, DTYPE)
    log_weights = tf.cast(log_weights, DTYPE)
    if sinkhorn_epsilon_policy not in {"fixed", "cost_mean_max_nominal"}:
        raise ValueError("unknown Sinkhorn epsilon policy")
    nominal_sinkhorn_epsilon = float(sinkhorn_epsilon)
    effective_sinkhorn_epsilon = nominal_sinkhorn_epsilon
    cost_scale_diagnostics: dict[str, Any] = {
        "sinkhorn_epsilon_policy": sinkhorn_epsilon_policy,
        "nominal_epsilon": nominal_sinkhorn_epsilon,
        "effective_epsilon": effective_sinkhorn_epsilon,
    }
    if resampling_route == OT_SINKHORN_COVARIANCE_CARRY_ROUTE:
        cost_matrix = pairwise_cost = None
        if sinkhorn_epsilon_policy == "cost_mean_max_nominal":
            diff = particles[:, None, :] - particles[None, :, :]
            pairwise_cost = tf.reduce_sum(diff * diff, axis=2)
            cost_mean = tf.reduce_mean(pairwise_cost)
            cost_max = tf.reduce_max(pairwise_cost)
            effective_sinkhorn_epsilon = max(
                nominal_sinkhorn_epsilon,
                float(cost_mean.numpy()),
            )
            cost_matrix = pairwise_cost
            cost_scale_diagnostics.update(
                {
                    "effective_epsilon": effective_sinkhorn_epsilon,
                    "sinkhorn_cost_scale_statistic": "pairwise_squared_euclidean_mean",
                    "sinkhorn_cost_min": _float(tf.reduce_min(pairwise_cost)),
                    "sinkhorn_cost_mean": _float(cost_mean),
                    "sinkhorn_cost_max": _float(cost_max),
                    "sinkhorn_cost_std": _float(tf.math.reduce_std(pairwise_cost)),
                    "sinkhorn_cost_over_nominal_epsilon_mean": _float(
                        cost_mean / tf.cast(nominal_sinkhorn_epsilon, DTYPE)
                    ),
                    "sinkhorn_cost_over_nominal_epsilon_max": _float(
                        cost_max / tf.cast(nominal_sinkhorn_epsilon, DTYPE)
                    ),
                }
            )
        result = sinkhorn_resample_tf(
            particles,
            weights,
            epsilon=effective_sinkhorn_epsilon,
            max_iterations=sinkhorn_iterations,
            tolerance=sinkhorn_tolerance,
            cost=cost_matrix,
        )
        canonical_transport, transport_diag = canonical_transport_from_sinkhorn_coupling_tf(
            result.coupling
        )
        transported_particles = tf.linalg.matmul(canonical_transport, particles)
        helper_particles = result.particles
        resampling_method = "fixed_target_sinkhorn"
        helper_diagnostics = result.diagnostics
    elif resampling_route == OT_ANNEALED_COVARIANCE_CARRY_ROUTE:
        result = annealed_transport_resample_tf(
            particles[tf.newaxis, :, :],
            log_weights[tf.newaxis, :],
            epsilon=effective_sinkhorn_epsilon,
            scaling=annealed_scaling,
            convergence_threshold=annealed_convergence_threshold,
            max_iterations=sinkhorn_iterations,
            ess_mask=tf.constant([True]),
            transport_gradient_mode=transport_gradient_mode,
            application_mode="active_rows_only",
        )
        canonical_transport, transport_diag = canonical_transport_from_annealed_matrix_tf(
            result.transport_matrix[0] if len(result.transport_matrix.shape) == 3 else result.transport_matrix
        )
        transported_particles = tf.linalg.matmul(canonical_transport, particles)
        helper_particles = result.particles[0] if len(result.particles.shape) == 3 else result.particles
        resampling_method = "annealed_transport"
        helper_diagnostics = result.diagnostics
    else:
        raise ValueError(f"unknown P8h OT resampling route: {resampling_route}")

    transport_shape = tf.shape(canonical_transport)
    particle_count = tf.shape(particles)[0]
    if len(canonical_transport.shape) != 2:
        raise ValueError("P8h canonical transport must have rank 2")
    if bool(
        tf.logical_or(
            transport_shape[0] != particle_count,
            transport_shape[1] != particle_count,
        ).numpy()
    ):
        raise ValueError("P8h canonical transport must have shape [N, N]")
    row_sums = tf.reduce_sum(canonical_transport, axis=1)
    row_sum_residual = tf.reduce_max(tf.abs(row_sums - 1.0))
    if float(row_sum_residual.numpy()) > float(canonical_row_sum_tolerance):
        raise FloatingPointError(
            "P8h canonical transport row sums exceeded declared tolerance"
        )
    if not _finite_bool(canonical_transport):
        raise FloatingPointError("P8h canonical transport is non-finite")

    carried_covariances, covariance_diag = carry_covariances_with_canonical_transport_tf(
        canonical_transport=canonical_transport,
        covariances=covariances,
        covariance_floor=covariance_floor,
    )
    particle_delta = tf.reduce_max(tf.abs(transported_particles - helper_particles))
    diagnostics: dict[str, Any] = {
        "resampled": True,
        "resampling_method": resampling_method,
        "transport_method": resampling_method,
        "epsilon": float(effective_sinkhorn_epsilon),
        "nominal_epsilon": nominal_sinkhorn_epsilon,
        "effective_epsilon": float(effective_sinkhorn_epsilon),
        "sinkhorn_epsilon_policy": sinkhorn_epsilon_policy,
        "sinkhorn_iterations": int(sinkhorn_iterations),
        "max_iterations": int(sinkhorn_iterations),
        "transport_gradient_mode": transport_gradient_mode,
        "relaxed_resampling_not_categorical": True,
        "pfpf_correction_route": OT_PFPF_CORRECTION_ROUTE,
        "finite_transport": _finite_bool(canonical_transport),
        "finite_particles": _finite_bool(transported_particles),
        "canonical_transport_row_sum_residual": _float(row_sum_residual),
        "canonical_transport_row_sum_tolerance": float(canonical_row_sum_tolerance),
        "canonical_transport_particle_delta_vs_helper": _float(particle_delta),
    }
    diagnostics.update(cost_scale_diagnostics)
    diagnostics.update(_tensor_diag_to_python(transport_diag))
    diagnostics.update(_tensor_diag_to_python(covariance_diag))
    diagnostics.update(
        {
            f"transport_helper_{key}": value
            for key, value in helper_diagnostics.items()
            if key
            in {
                "component_id",
                "mathematical_object",
                "resampling_status",
                "max_row_residual",
                "max_column_residual",
                "total_mass_residual",
                "finite_coupling",
                "finite_transport",
                "finite_particles",
                "iterations_used",
                "max_iterations_used",
                "transport_backward_rule",
                "application_mode",
                "backend",
            }
        }
    )
    if not diagnostics["finite_transport"] or not diagnostics["finite_particles"]:
        raise FloatingPointError("P8h OT resampling emitted non-finite state")
    if not bool(tf.reduce_all(tf.math.is_finite(carried_covariances)).numpy()):
        raise FloatingPointError("P8h OT covariance carry emitted non-finite state")
    diagnostics["canonical_transport_shape"] = [
        int(v) for v in tf.shape(canonical_transport).numpy().tolist()
    ]
    return transported_particles, carried_covariances, diagnostics


def stabilize_covariance_tf(
    covariance: tf.Tensor,
    *,
    covariance_floor: float | tf.Tensor = DEFAULT_COVARIANCE_FLOOR,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Symmetrize and project a covariance through the declared PSD floor."""

    covariance = symmetrize(covariance)
    floor = tf.cast(covariance_floor, DTYPE)
    eigenvalues, floored, eigenvectors, implemented, residual = psd_eigh(covariance, floor)
    diagnostics = {
        "covariance_floor": floor,
        "min_raw_eigenvalue": tf.reduce_min(eigenvalues),
        "floor_count": floor_count(eigenvalues, floor),
        "psd_projection_residual": residual,
        "implemented_covariance": implemented,
        "covariance_floor_policy": tf.constant("declared_diagnostic_psd_floor_not_promotion_threshold"),
    }
    return implemented, diagnostics


def validate_pseudo_time_steps_tf(
    pseudo_time_steps: tf.Tensor,
    *,
    atol: float | tf.Tensor = 1e-12,
) -> tf.Tensor:
    """Validate the Algorithm 1 pseudo-time grid increments."""

    steps = tf.reshape(tf.cast(pseudo_time_steps, DTYPE), [-1])
    if int(tf.size(steps).numpy()) == 0:
        raise ValueError("Algorithm 1 pseudo-time grid must contain at least one step")
    if not _finite_bool(steps):
        raise ValueError("Algorithm 1 pseudo-time grid contains non-finite steps")
    if bool(tf.reduce_any(steps <= tf.constant(0.0, dtype=DTYPE)).numpy()):
        raise ValueError("Algorithm 1 pseudo-time increments must be positive")
    step_sum = tf.reduce_sum(steps)
    if float(tf.abs(step_sum - tf.constant(1.0, dtype=DTYPE)).numpy()) > float(
        tf.cast(atol, DTYPE).numpy()
    ):
        raise ValueError("Algorithm 1 pseudo-time increments must sum to 1")
    return steps


def _weighted_covariance(centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return symmetrize(tf.transpose(centered) @ (centered * weights[:, tf.newaxis]))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _finite_bool(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _tensor_diag_to_python(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, tf.Tensor):
            if value.dtype == tf.bool:
                converted[key] = bool(value.numpy())
            elif value.dtype == tf.string:
                converted[key] = value.numpy().decode("utf-8")
            elif value.shape.rank == 0:
                converted[key] = _float(value)
            else:
                converted[key] = [int(v) for v in value.numpy().tolist()]
        else:
            converted[key] = value
    return converted

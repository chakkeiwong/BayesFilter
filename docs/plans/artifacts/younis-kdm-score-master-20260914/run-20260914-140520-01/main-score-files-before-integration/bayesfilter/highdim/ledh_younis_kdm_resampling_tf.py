"""Phase 4B full-mixture IWSG resampling after the canonical LEDH reset.

This diagnostic route executes Contract-E and the GenUT dual-cap correction,
then applies a Younis-style marginalized Gaussian-mixture resampler. It defines
the separate RESKDM-IWSG-FINITE fixed-anchor replay scalar; it is not another
derivative implementation for the canonical ATOM-FINITE scalar.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    _value_and_analytical_score_impl,
)
from bayesfilter.highdim.ledh_younis_kdm_tf import (
    RESKDM_IWSG_FINITE_TARGET,
    make_full_mixture_iwsg_resampling_kernel,
    make_gaussian_kdm_kernel,
)


Tensor = tf.Tensor

RESAMPLING_ROUTE_ID = "ledh_contract_e_then_younis_iwsg_resampling_reference_v2"
RESAMPLING_ROUTE_CLASSIFICATION = (
    "combined_contract_e_younis_raw_iwsg_with_bayesfilter_extensions_diagnostic_only"
)
RESAMPLING_OPERATION_SEMANTICS = (
    "contract_e_dual_cap_then_fixed_anchor_raw_ratio_marginal_kdm_resampling_v2"
)
RESPONSIBILITY_COVARIANCE_MARK_POLICY = "responsibility_conditional_mean_no_scatter_v1"
SELECTED_LABEL_COVARIANCE_MARK_POLICY = "fixed_selected_component_mark_v1"
COVARIANCE_MARK_POLICY = RESPONSIBILITY_COVARIANCE_MARK_POLICY
COVARIANCE_MARK_POLICIES = (
    RESPONSIBILITY_COVARIANCE_MARK_POLICY,
    SELECTED_LABEL_COVARIANCE_MARK_POLICY,
)
ANCHOR_POLICY = "sequential_stratified_fixed_sample_bank_v1"
REPLAY_POLICY = "fixed_samples_and_fixed_marginal_proposal_density_v1"
BANDWIDTH_POLICY = "common_full_rank_gaussian_covariance_per_time_v1"
INCOMING_WEIGHT_POLICY = "raw_iwsg_ratio_over_particle_count_v1"
DERIVATIVE_SEMANTICS = "fixed_anchor_total_directional_derivative_v1"
SCORE_OUTPUT_SEMANTICS = "one_total_directional_derivative_per_call_v1"
SOURCE_SCOPE = (
    "younis_2023_eq14_15_raw_ratio_and_author_code_b0e2fd5_with_"
    "bayesfilter_contract_e_bandwidth_and_mark_extensions_v1"
)
NORMALIZATION_TOLERANCE_POLICY = "max_configured_or_8n_machine_epsilon_v1"


def _positive_dimension(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _effective_normalization_tolerance(
    dtype: tf.dtypes.DType,
    particle_count: int,
    configured_tolerance: float,
) -> float:
    if float(configured_tolerance) <= 0.0:
        raise ValueError("normalization_tolerance must be positive")
    machine_epsilon = (
        1.1920928955078125e-7 if dtype == tf.float32 else 2.220446049250313e-16
    )
    return max(
        float(configured_tolerance),
        8.0 * float(particle_count) * machine_epsilon,
    )


def _validated_options(options: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(options)
    for forbidden in (
        "with_score",
        "return_trace",
        "observation_factor_override",
        "post_reset_transform",
    ):
        if forbidden in result:
            raise ValueError(f"canonical_options must not contain {forbidden}")
    if result.get("reset_policy") != "contract_e":
        raise ValueError("Phase 4B requires reset_policy='contract_e'")
    if result.get("reset_design") is None:
        raise ValueError("Phase 4B requires an explicit Contract-E reset design")
    if int(result.get("annealed_stages", 1)) != 1:
        raise ValueError("Phase 4B currently requires annealed_stages=1")
    if int(result.get("correction_steps", 0)) < 1:
        raise ValueError("Phase 4B requires the diagonal GenUT correction")
    if int(result.get("pairwise_steps", 0)) < 1:
        raise ValueError("Phase 4B requires the pairwise GenUT correction")
    if float(result.get("coordinate_cap", 0.0)) <= 0.0:
        raise ValueError("Phase 4B requires a positive coordinate cap")
    return result


def _static_problem_dimensions(
    initial_states: Tensor,
    observations: Tensor,
) -> tuple[int, int, int]:
    if initial_states.shape.rank != 2:
        raise ValueError("initial_states must have shape [N,D]")
    if observations.shape.rank != 2:
        raise ValueError("observations must have shape [T,O]")
    if (
        initial_states.shape[0] is None
        or initial_states.shape[1] is None
        or observations.shape[0] is None
    ):
        raise ValueError("Phase 4B requires statically known N, D, and T")
    return (
        int(initial_states.shape[0]),
        int(initial_states.shape[1]),
        int(observations.shape[0]),
    )


def _bandwidth_sample(
    component_means: Tensor,
    bandwidth_covariances: Tensor,
    component_indices: Tensor,
    standard_noises: Tensor,
    *,
    rank_tolerance: float,
    symmetry_tolerance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample safely from selected components and expose validity diagnostics."""

    dtype = component_means.dtype
    dimension = int(component_means.shape[1])
    symmetric = 0.5 * (
        bandwidth_covariances + tf.linalg.matrix_transpose(bandwidth_covariances)
    )
    scale = tf.maximum(
        tf.reduce_max(tf.abs(symmetric), axis=[-2, -1]),
        tf.ones([tf.shape(component_means)[0]], dtype),
    )
    symmetry_error = tf.reduce_max(
        tf.abs(
            bandwidth_covariances - tf.linalg.matrix_transpose(bandwidth_covariances)
        ),
        axis=[-2, -1],
    )
    minimum_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(symmetric), axis=-1)
    valid_components = (
        tf.reduce_all(tf.math.is_finite(bandwidth_covariances), axis=[-2, -1])
        & (symmetry_error <= tf.cast(symmetry_tolerance, dtype) * scale)
        & (minimum_eigenvalue > tf.cast(rank_tolerance, dtype) * scale)
    )
    safe_covariances = tf.where(
        valid_components[:, tf.newaxis, tf.newaxis],
        symmetric,
        tf.eye(dimension, dtype=dtype)[tf.newaxis, :, :],
    )
    selected_factors = tf.gather(
        tf.linalg.cholesky(safe_covariances), component_indices
    )
    samples = tf.gather(component_means, component_indices) + tf.einsum(
        "nij,nj->ni", selected_factors, standard_noises
    )
    return (
        tf.stop_gradient(samples),
        tf.reduce_all(valid_components),
        tf.reduce_min(minimum_eigenvalue),
        tf.reduce_max(symmetry_error),
    )


def _stack_trace(trace: tuple[Mapping[str, Tensor], ...], name: str) -> Tensor:
    return tf.stack([step[name] for step in trace])


def _run_resampling_program(
    model: Any,
    theta: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    bandwidths: Tensor,
    d_bandwidths: Tensor,
    *,
    canonical_options: Mapping[str, Any],
    anchor_mode: bool,
    fixed_samples: Tensor | None,
    fixed_proposal_log_densities: Tensor | None,
    fixed_component_indices: Tensor | None,
    stratified_uniforms: Tensor | None,
    kdm_standard_noises: Tensor | None,
    covariance_mark_policy: str,
    resampling_kernel: Callable[..., Mapping[str, Tensor]],
    density_kernel: Callable[..., Mapping[str, Tensor]],
    rank_tolerance: float,
    normalization_tolerance: float,
    symmetry_tolerance: float,
) -> Mapping[str, Any]:
    options = _validated_options(canonical_options)
    if covariance_mark_policy not in COVARIANCE_MARK_POLICIES:
        raise ValueError(
            "covariance_mark_policy must be one of "
            f"{COVARIANCE_MARK_POLICIES}, got {covariance_mark_policy!r}"
        )
    theta = tf.convert_to_tensor(theta)
    initial_states = tf.convert_to_tensor(initial_states)
    dtype = initial_states.dtype
    initial_covariances = tf.convert_to_tensor(initial_covariances, dtype)
    noises = tf.convert_to_tensor(noises, dtype)
    observations = tf.convert_to_tensor(observations, dtype)
    bandwidths = tf.convert_to_tensor(bandwidths, dtype)
    d_bandwidths = tf.convert_to_tensor(d_bandwidths, dtype)
    particle_count, dimension, horizon = _static_problem_dimensions(
        initial_states, observations
    )
    tf.ensure_shape(
        bandwidths,
        [horizon, dimension, dimension],
    )
    tf.ensure_shape(
        d_bandwidths,
        [horizon, dimension, dimension],
    )

    if anchor_mode:
        if stratified_uniforms is None or kdm_standard_noises is None:
            raise ValueError("anchor mode requires stratified uniforms and KDM noises")
        stratified_uniforms = tf.convert_to_tensor(stratified_uniforms, dtype)
        kdm_standard_noises = tf.convert_to_tensor(kdm_standard_noises, dtype)
        tf.ensure_shape(stratified_uniforms, [horizon, particle_count])
        tf.ensure_shape(kdm_standard_noises, [horizon, particle_count, dimension])
    else:
        if (
            fixed_samples is None
            or fixed_proposal_log_densities is None
            or fixed_component_indices is None
        ):
            raise ValueError(
                "replay mode requires fixed sample, proposal, and component-index banks"
            )
        fixed_samples = tf.convert_to_tensor(fixed_samples, dtype)
        fixed_proposal_log_densities = tf.convert_to_tensor(
            fixed_proposal_log_densities, dtype
        )
        fixed_component_indices = tf.convert_to_tensor(
            fixed_component_indices, tf.int32
        )
        tf.ensure_shape(fixed_samples, [horizon, particle_count, dimension])
        tf.ensure_shape(fixed_proposal_log_densities, [horizon, particle_count])
        tf.ensure_shape(fixed_component_indices, [horizon, particle_count])

    uniform_component_weights = tf.ones([particle_count], dtype) / tf.cast(
        particle_count, dtype
    )
    d_uniform_component_weights = tf.zeros([1, particle_count], dtype)
    uniform_cumulative = tf.cumsum(uniform_component_weights)
    resampling_records: list[Mapping[str, Tensor]] = []

    def post_reset_transform(
        time_index: int,
        reset_states: Tensor,
        d_reset_states: Tensor,
        reset_covariance_marks: Tensor,
        d_reset_covariance_marks: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Mapping[str, Tensor],
    ]:
        common_bandwidth = bandwidths[time_index]
        common_d_bandwidth = d_bandwidths[time_index]
        bandwidth = tf.broadcast_to(
            common_bandwidth[tf.newaxis, :, :],
            [particle_count, dimension, dimension],
        )
        d_bandwidth = tf.broadcast_to(
            common_d_bandwidth[tf.newaxis, :, :],
            [particle_count, dimension, dimension],
        )
        if anchor_mode:
            uniforms = stratified_uniforms[time_index]
            uniform_valid = tf.reduce_all(
                tf.math.is_finite(uniforms)
                & (uniforms >= tf.zeros_like(uniforms))
                & (uniforms < tf.ones_like(uniforms))
            )
            upper = tf.cast(1.0 - (1.0e-7 if dtype == tf.float32 else 1.0e-15), dtype)
            safe_uniforms = tf.clip_by_value(uniforms, tf.zeros([], dtype), upper)
            positions = (
                tf.cast(tf.range(particle_count), dtype) + safe_uniforms
            ) / tf.cast(particle_count, dtype)
            component_indices = tf.minimum(
                tf.searchsorted(uniform_cumulative, positions, side="right"),
                particle_count - 1,
            )
            (
                samples,
                bandwidth_valid,
                minimum_bandwidth_eigenvalue,
                bandwidth_symmetry_error,
            ) = _bandwidth_sample(
                reset_states,
                bandwidth,
                component_indices,
                kdm_standard_noises[time_index],
                rank_tolerance=rank_tolerance,
                symmetry_tolerance=symmetry_tolerance,
            )
            density = density_kernel(
                samples,
                uniform_component_weights,
                reset_states,
                bandwidth,
                tf.zeros([1, particle_count, dimension], dtype),
                d_uniform_component_weights,
                d_reset_states[tf.newaxis, :, :],
                d_bandwidth[tf.newaxis, :, :, :],
            )
            proposal_log_density = tf.stop_gradient(density["log_density"])
            sample_bank_valid = uniform_valid & bandwidth_valid
        else:
            samples = fixed_samples[time_index]
            proposal_log_density = fixed_proposal_log_densities[time_index]
            supplied_component_indices = fixed_component_indices[time_index]
            component_indices_valid = tf.reduce_all(
                (supplied_component_indices >= 0)
                & (supplied_component_indices < particle_count)
            )
            component_indices = tf.clip_by_value(
                supplied_component_indices, 0, particle_count - 1
            )
            sample_bank_valid = (
                tf.reduce_all(tf.math.is_finite(samples))
                & tf.reduce_all(tf.math.is_finite(proposal_log_density))
                & component_indices_valid
            )
            minimum_bandwidth_eigenvalue = tf.reduce_min(
                tf.linalg.eigvalsh(
                    0.5 * (bandwidth + tf.linalg.matrix_transpose(bandwidth))
                )
            )
            bandwidth_symmetry_error = tf.reduce_max(
                tf.abs(bandwidth - tf.linalg.matrix_transpose(bandwidth))
            )

        resampling = resampling_kernel(
            samples,
            uniform_component_weights,
            reset_states,
            bandwidth,
            reset_covariance_marks,
            proposal_log_density,
            d_uniform_component_weights,
            d_reset_states[tf.newaxis, :, :],
            d_bandwidth[tf.newaxis, :, :, :],
            d_reset_covariance_marks[tf.newaxis, :, :, :],
        )
        if covariance_mark_policy == RESPONSIBILITY_COVARIANCE_MARK_POLICY:
            outgoing_covariance_marks = resampling["transported_covariance_marks"]
            d_outgoing_covariance_marks = resampling["d_transported_covariance_marks"][
                0
            ]
        else:
            outgoing_covariance_marks = tf.gather(
                reset_covariance_marks, component_indices
            )
            d_outgoing_covariance_marks = tf.gather(
                d_reset_covariance_marks, component_indices
            )

        anchor_importance_error = tf.abs(
            resampling["importance_weight_sum"] - tf.ones([], dtype)
        )
        anchor_identity_valid = (
            (resampling["anchor_log_ratio_error"] <= normalization_tolerance)
            & (anchor_importance_error <= normalization_tolerance)
            if anchor_mode
            else tf.constant(True)
        )
        step_valid = sample_bank_valid & resampling["valid"] & anchor_identity_valid
        record = {
            "kdm_component_means": reset_states,
            "d_kdm_component_means": d_reset_states,
            "kdm_source_covariance_marks": reset_covariance_marks,
            "d_kdm_source_covariance_marks": d_reset_covariance_marks,
            "kdm_common_bandwidth": common_bandwidth,
            "d_kdm_common_bandwidth": common_d_bandwidth,
            "kdm_samples": samples,
            "kdm_component_indices": component_indices,
            "kdm_proposal_log_density": proposal_log_density,
            "kdm_log_density": resampling["log_density"],
            "d_kdm_log_density": resampling["d_log_density"][0],
            "kdm_log_ratio": resampling["log_ratio"],
            "kdm_importance_log_weights": resampling["importance_log_weights"],
            "d_kdm_importance_log_weights": resampling["d_importance_log_weights"][0],
            "kdm_importance_weights": resampling["importance_weights"],
            "d_kdm_importance_weights": resampling["d_importance_weights"][0],
            "kdm_responsibilities": resampling["responsibilities"],
            "d_kdm_responsibilities": resampling["d_responsibilities"][0],
            "kdm_normalized_weights": resampling["normalized_weights"],
            "d_kdm_normalized_weights": resampling["d_normalized_weights"][0],
            "kdm_responsibility_covariance_marks": resampling[
                "transported_covariance_marks"
            ],
            "d_kdm_responsibility_covariance_marks": resampling[
                "d_transported_covariance_marks"
            ][0],
            "kdm_covariance_marks": outgoing_covariance_marks,
            "d_kdm_covariance_marks": d_outgoing_covariance_marks,
            "kdm_step_valid": step_valid,
            "kdm_pair_count": resampling["complexity_pair_count"],
            "kdm_anchor_log_ratio_error": resampling["anchor_log_ratio_error"],
            "kdm_responsibility_row_sum_error": resampling[
                "responsibility_row_sum_error"
            ],
            "kdm_weight_sum_error": resampling["weight_sum_error"],
            "kdm_weight_tangent_sum_error": resampling["weight_tangent_sum_error"],
            "kdm_importance_weight_sum": resampling["importance_weight_sum"],
            "d_kdm_importance_weight_sum": resampling["importance_weight_tangent_sum"][
                0
            ],
            "kdm_anchor_importance_weight_sum_error": anchor_importance_error,
            "kdm_minimum_bandwidth_eigenvalue": (minimum_bandwidth_eigenvalue),
            "kdm_bandwidth_symmetry_error": bandwidth_symmetry_error,
        }
        resampling_records.append(record)
        return (
            samples,
            tf.zeros_like(samples),
            outgoing_covariance_marks,
            d_outgoing_covariance_marks,
            resampling["importance_log_weights"],
            resampling["d_importance_log_weights"][0],
            record,
        )

    value, score, trace = _value_and_analytical_score_impl(
        model,
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        with_score=True,
        return_trace=True,
        observation_factor_override=None,
        post_reset_transform=post_reset_transform,
        **options,
    )
    step_valid = tf.stack([record["kdm_step_valid"] for record in resampling_records])
    higher_moment_valid = _stack_trace(trace, "higher_moment_valid")
    return {
        "route_id": RESAMPLING_ROUTE_ID,
        "route_classification": RESAMPLING_ROUTE_CLASSIFICATION,
        "operation_semantics": RESAMPLING_OPERATION_SEMANTICS,
        "target_label": RESKDM_IWSG_FINITE_TARGET,
        "anchor_policy": ANCHOR_POLICY,
        "replay_policy": REPLAY_POLICY,
        "covariance_mark_policy": covariance_mark_policy,
        "bandwidth_policy": BANDWIDTH_POLICY,
        "incoming_weight_policy": INCOMING_WEIGHT_POLICY,
        "derivative_semantics": DERIVATIVE_SEMANTICS,
        "score_output_semantics": SCORE_OUTPUT_SEMANTICS,
        "source_scope": SOURCE_SCOPE,
        "effective_normalization_tolerance": tf.constant(
            normalization_tolerance, dtype
        ),
        "mode": "anchor" if anchor_mode else "replay",
        "value": value,
        "score": score,
        "valid": (
            tf.math.is_finite(value)
            & tf.reduce_all(tf.math.is_finite(score))
            & tf.reduce_all(step_valid)
            & tf.reduce_all(higher_moment_valid)
        ),
        "step_valid": step_valid,
        "trace": trace,
        "fixed_samples": _stack_trace(trace, "kdm_samples"),
        "fixed_proposal_log_densities": _stack_trace(trace, "kdm_proposal_log_density"),
        "component_indices": _stack_trace(trace, "kdm_component_indices"),
        "component_means": _stack_trace(trace, "kdm_component_means"),
        "d_component_means": _stack_trace(trace, "d_kdm_component_means"),
        "source_covariance_marks": _stack_trace(trace, "kdm_source_covariance_marks"),
        "d_source_covariance_marks": _stack_trace(
            trace, "d_kdm_source_covariance_marks"
        ),
        "importance_log_weights": _stack_trace(trace, "kdm_importance_log_weights"),
        "d_importance_log_weights": _stack_trace(trace, "d_kdm_importance_log_weights"),
        "importance_weights": _stack_trace(trace, "kdm_importance_weights"),
        "d_importance_weights": _stack_trace(trace, "d_kdm_importance_weights"),
        "responsibilities": _stack_trace(trace, "kdm_responsibilities"),
        "d_responsibilities": _stack_trace(trace, "d_kdm_responsibilities"),
        "normalized_weights": _stack_trace(trace, "kdm_normalized_weights"),
        "d_normalized_weights": _stack_trace(trace, "d_kdm_normalized_weights"),
        "responsibility_covariance_marks": _stack_trace(
            trace, "kdm_responsibility_covariance_marks"
        ),
        "d_responsibility_covariance_marks": _stack_trace(
            trace, "d_kdm_responsibility_covariance_marks"
        ),
        "covariance_marks": _stack_trace(trace, "kdm_covariance_marks"),
        "d_covariance_marks": _stack_trace(trace, "d_kdm_covariance_marks"),
        "incoming_log_weights": _stack_trace(trace, "incoming_log_weights"),
        "d_incoming_log_weights": _stack_trace(trace, "d_incoming_log_weights"),
        "outgoing_log_weights": _stack_trace(trace, "outgoing_log_weights"),
        "d_outgoing_log_weights": _stack_trace(trace, "d_outgoing_log_weights"),
        "prior_observation_logits": _stack_trace(trace, "prior_observation_logits"),
        "d_prior_observation_logits": _stack_trace(trace, "d_prior_observation_logits"),
        "predicted_covariances": _stack_trace(trace, "predicted_covariances"),
        "d_predicted_covariances": _stack_trace(trace, "d_predicted_covariances"),
        "states_after_resampling": _stack_trace(trace, "states_after_reset"),
        "d_states_after_resampling": _stack_trace(trace, "d_states_after_reset"),
        "higher_moment_valid": higher_moment_valid,
        "higher_moment_pairwise_configured": _stack_trace(
            trace, "higher_moment_pairwise_configured"
        ),
        "higher_moment_pairwise_target_mask": _stack_trace(
            trace, "higher_moment_pairwise_target_mask"
        ),
        "higher_moment_maximum_pairwise_pre_cap_particle_rms": _stack_trace(
            trace, "higher_moment_maximum_pairwise_pre_cap_particle_rms"
        ),
        "higher_moment_maximum_pairwise_post_cap_particle_rms": _stack_trace(
            trace, "higher_moment_maximum_pairwise_post_cap_particle_rms"
        ),
        "higher_moment_minimum_pairwise_particle_cap_scale": _stack_trace(
            trace, "higher_moment_minimum_pairwise_particle_cap_scale"
        ),
        "higher_moment_maximum_coordinatewise_pre_cap_absolute": _stack_trace(
            trace, "higher_moment_maximum_coordinatewise_pre_cap_absolute"
        ),
        "higher_moment_maximum_coordinatewise_post_cap_absolute": _stack_trace(
            trace, "higher_moment_maximum_coordinatewise_post_cap_absolute"
        ),
        "higher_moment_mean_coordinatewise_cap_displacement": _stack_trace(
            trace, "higher_moment_mean_coordinatewise_cap_displacement"
        ),
        "higher_moment_fraction_coordinatewise_cap_active": _stack_trace(
            trace, "higher_moment_fraction_coordinatewise_cap_active"
        ),
        "higher_moment_minimum_coordinatewise_cap_derivative": _stack_trace(
            trace, "higher_moment_minimum_coordinatewise_cap_derivative"
        ),
        "pair_count": tf.reduce_sum(_stack_trace(trace, "kdm_pair_count")),
        "maximum_anchor_log_ratio_error": tf.reduce_max(
            _stack_trace(trace, "kdm_anchor_log_ratio_error")
        ),
        "maximum_responsibility_row_sum_error": tf.reduce_max(
            _stack_trace(trace, "kdm_responsibility_row_sum_error")
        ),
        "maximum_weight_sum_error": tf.reduce_max(
            _stack_trace(trace, "kdm_weight_sum_error")
        ),
        "maximum_weight_tangent_sum_error": tf.reduce_max(
            _stack_trace(trace, "kdm_weight_tangent_sum_error")
        ),
        "importance_weight_sums": _stack_trace(trace, "kdm_importance_weight_sum"),
        "d_importance_weight_sums": _stack_trace(trace, "d_kdm_importance_weight_sum"),
        "maximum_anchor_importance_weight_sum_error": tf.reduce_max(
            _stack_trace(trace, "kdm_anchor_importance_weight_sum_error")
        ),
        "minimum_bandwidth_eigenvalue": tf.reduce_min(
            _stack_trace(trace, "kdm_minimum_bandwidth_eigenvalue")
        ),
        "maximum_bandwidth_symmetry_error": tf.reduce_max(
            _stack_trace(trace, "kdm_bandwidth_symmetry_error")
        ),
    }


def _factory_components(
    *,
    particle_count: int,
    state_dimension: int,
    dtype: tf.dtypes.DType,
    rank_tolerance: float,
    normalization_tolerance: float,
    symmetry_tolerance: float,
) -> tuple[Callable[..., Mapping[str, Tensor]], Callable[..., Mapping[str, Tensor]]]:
    resampling_kernel = make_full_mixture_iwsg_resampling_kernel(
        particle_count=particle_count,
        dimension=state_dimension,
        dtype=dtype,
        rank_tolerance=rank_tolerance,
        normalization_tolerance=normalization_tolerance,
        symmetry_tolerance=symmetry_tolerance,
        jit_compile=False,
    )
    density_kernel = make_gaussian_kdm_kernel(
        evaluation_count=particle_count,
        component_count=particle_count,
        dimension=state_dimension,
        dtype=dtype,
        rank_tolerance=rank_tolerance,
        normalization_tolerance=normalization_tolerance,
        jit_compile=False,
    )
    return resampling_kernel, density_kernel


def _tensor_result(result: Mapping[str, Any]) -> Mapping[str, Tensor]:
    names = (
        "value",
        "score",
        "valid",
        "step_valid",
        "fixed_samples",
        "fixed_proposal_log_densities",
        "component_indices",
        "component_means",
        "d_component_means",
        "source_covariance_marks",
        "d_source_covariance_marks",
        "importance_log_weights",
        "d_importance_log_weights",
        "importance_weights",
        "d_importance_weights",
        "responsibilities",
        "d_responsibilities",
        "normalized_weights",
        "d_normalized_weights",
        "responsibility_covariance_marks",
        "d_responsibility_covariance_marks",
        "covariance_marks",
        "d_covariance_marks",
        "incoming_log_weights",
        "d_incoming_log_weights",
        "outgoing_log_weights",
        "d_outgoing_log_weights",
        "prior_observation_logits",
        "d_prior_observation_logits",
        "predicted_covariances",
        "d_predicted_covariances",
        "states_after_resampling",
        "d_states_after_resampling",
        "higher_moment_valid",
        "higher_moment_pairwise_configured",
        "higher_moment_pairwise_target_mask",
        "higher_moment_maximum_pairwise_pre_cap_particle_rms",
        "higher_moment_maximum_pairwise_post_cap_particle_rms",
        "higher_moment_minimum_pairwise_particle_cap_scale",
        "higher_moment_maximum_coordinatewise_pre_cap_absolute",
        "higher_moment_maximum_coordinatewise_post_cap_absolute",
        "higher_moment_mean_coordinatewise_cap_displacement",
        "higher_moment_fraction_coordinatewise_cap_active",
        "higher_moment_minimum_coordinatewise_cap_derivative",
        "pair_count",
        "maximum_anchor_log_ratio_error",
        "maximum_responsibility_row_sum_error",
        "maximum_weight_sum_error",
        "maximum_weight_tangent_sum_error",
        "importance_weight_sums",
        "d_importance_weight_sums",
        "maximum_anchor_importance_weight_sum_error",
        "minimum_bandwidth_eigenvalue",
        "maximum_bandwidth_symmetry_error",
        "effective_normalization_tolerance",
    )
    return {name: result[name] for name in names}


def make_resampling_anchor_kernel(
    model: Any,
    *,
    theta_dimension: int,
    particle_count: int,
    state_dimension: int,
    observation_dimension: int,
    horizon: int,
    canonical_options: Mapping[str, Any],
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-8,
    covariance_mark_policy: str = RESPONSIBILITY_COVARIANCE_MARK_POLICY,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Build the fixed-shape sequential anchor endpoint."""

    theta_dimension = _positive_dimension("theta_dimension", theta_dimension)
    particle_count = _positive_dimension("particle_count", particle_count)
    state_dimension = _positive_dimension("state_dimension", state_dimension)
    observation_dimension = _positive_dimension(
        "observation_dimension", observation_dimension
    )
    horizon = _positive_dimension("horizon", horizon)
    dtype = tf.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError("dtype must be floating")
    if covariance_mark_policy not in COVARIANCE_MARK_POLICIES:
        raise ValueError(
            "covariance_mark_policy must be one of "
            f"{COVARIANCE_MARK_POLICIES}, got {covariance_mark_policy!r}"
        )
    options = _validated_options(canonical_options)
    effective_normalization_tolerance = _effective_normalization_tolerance(
        dtype, particle_count, normalization_tolerance
    )
    resampling_kernel, density_kernel = _factory_components(
        particle_count=particle_count,
        state_dimension=state_dimension,
        dtype=dtype,
        rank_tolerance=rank_tolerance,
        normalization_tolerance=effective_normalization_tolerance,
        symmetry_tolerance=symmetry_tolerance,
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec([theta_dimension], dtype),
            tf.TensorSpec([particle_count, state_dimension], dtype),
            tf.TensorSpec([particle_count, state_dimension, state_dimension], dtype),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, observation_dimension], dtype),
            tf.TensorSpec(
                [horizon, state_dimension, state_dimension],
                dtype,
            ),
            tf.TensorSpec(
                [horizon, state_dimension, state_dimension],
                dtype,
            ),
            tf.TensorSpec([horizon, particle_count], dtype),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        theta: Tensor,
        initial_states: Tensor,
        initial_covariances: Tensor,
        noises: Tensor,
        observations: Tensor,
        bandwidths: Tensor,
        d_bandwidths: Tensor,
        stratified_uniforms: Tensor,
        kdm_standard_noises: Tensor,
    ) -> Mapping[str, Tensor]:
        result = _run_resampling_program(
            model,
            theta,
            initial_states,
            initial_covariances,
            noises,
            observations,
            bandwidths,
            d_bandwidths,
            canonical_options=options,
            anchor_mode=True,
            fixed_samples=None,
            fixed_proposal_log_densities=None,
            fixed_component_indices=None,
            stratified_uniforms=stratified_uniforms,
            kdm_standard_noises=kdm_standard_noises,
            covariance_mark_policy=covariance_mark_policy,
            resampling_kernel=resampling_kernel,
            density_kernel=density_kernel,
            rank_tolerance=rank_tolerance,
            normalization_tolerance=effective_normalization_tolerance,
            symmetry_tolerance=symmetry_tolerance,
        )
        return _tensor_result(result)

    kernel.route_id = RESAMPLING_ROUTE_ID
    kernel.route_classification = RESAMPLING_ROUTE_CLASSIFICATION
    kernel.operation_semantics = RESAMPLING_OPERATION_SEMANTICS
    kernel.target_label = RESKDM_IWSG_FINITE_TARGET
    kernel.bandwidth_policy = BANDWIDTH_POLICY
    kernel.incoming_weight_policy = INCOMING_WEIGHT_POLICY
    kernel.covariance_mark_policy = covariance_mark_policy
    kernel.derivative_semantics = DERIVATIVE_SEMANTICS
    kernel.score_output_semantics = SCORE_OUTPUT_SEMANTICS
    kernel.source_scope = SOURCE_SCOPE
    kernel.normalization_tolerance_policy = NORMALIZATION_TOLERANCE_POLICY
    kernel.effective_normalization_tolerance = effective_normalization_tolerance
    kernel.mode = "anchor"
    kernel.jit_compile = bool(jit_compile)
    return kernel


def make_resampling_replay_kernel(
    model: Any,
    *,
    theta_dimension: int,
    particle_count: int,
    state_dimension: int,
    observation_dimension: int,
    horizon: int,
    canonical_options: Mapping[str, Any],
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-8,
    covariance_mark_policy: str = RESPONSIBILITY_COVARIANCE_MARK_POLICY,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Build the fixed-shape replay endpoint with no anchor regeneration."""

    theta_dimension = _positive_dimension("theta_dimension", theta_dimension)
    particle_count = _positive_dimension("particle_count", particle_count)
    state_dimension = _positive_dimension("state_dimension", state_dimension)
    observation_dimension = _positive_dimension(
        "observation_dimension", observation_dimension
    )
    horizon = _positive_dimension("horizon", horizon)
    dtype = tf.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError("dtype must be floating")
    if covariance_mark_policy not in COVARIANCE_MARK_POLICIES:
        raise ValueError(
            "covariance_mark_policy must be one of "
            f"{COVARIANCE_MARK_POLICIES}, got {covariance_mark_policy!r}"
        )
    options = _validated_options(canonical_options)
    effective_normalization_tolerance = _effective_normalization_tolerance(
        dtype, particle_count, normalization_tolerance
    )
    resampling_kernel, density_kernel = _factory_components(
        particle_count=particle_count,
        state_dimension=state_dimension,
        dtype=dtype,
        rank_tolerance=rank_tolerance,
        normalization_tolerance=effective_normalization_tolerance,
        symmetry_tolerance=symmetry_tolerance,
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec([theta_dimension], dtype),
            tf.TensorSpec([particle_count, state_dimension], dtype),
            tf.TensorSpec([particle_count, state_dimension, state_dimension], dtype),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, observation_dimension], dtype),
            tf.TensorSpec(
                [horizon, state_dimension, state_dimension],
                dtype,
            ),
            tf.TensorSpec(
                [horizon, state_dimension, state_dimension],
                dtype,
            ),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, particle_count], dtype),
            tf.TensorSpec([horizon, particle_count], tf.int32),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        theta: Tensor,
        initial_states: Tensor,
        initial_covariances: Tensor,
        noises: Tensor,
        observations: Tensor,
        bandwidths: Tensor,
        d_bandwidths: Tensor,
        fixed_samples: Tensor,
        fixed_proposal_log_densities: Tensor,
        fixed_component_indices: Tensor,
    ) -> Mapping[str, Tensor]:
        result = _run_resampling_program(
            model,
            theta,
            initial_states,
            initial_covariances,
            noises,
            observations,
            bandwidths,
            d_bandwidths,
            canonical_options=options,
            anchor_mode=False,
            fixed_samples=fixed_samples,
            fixed_proposal_log_densities=fixed_proposal_log_densities,
            fixed_component_indices=fixed_component_indices,
            stratified_uniforms=None,
            kdm_standard_noises=None,
            covariance_mark_policy=covariance_mark_policy,
            resampling_kernel=resampling_kernel,
            density_kernel=density_kernel,
            rank_tolerance=rank_tolerance,
            normalization_tolerance=effective_normalization_tolerance,
            symmetry_tolerance=symmetry_tolerance,
        )
        return _tensor_result(result)

    kernel.route_id = RESAMPLING_ROUTE_ID
    kernel.route_classification = RESAMPLING_ROUTE_CLASSIFICATION
    kernel.operation_semantics = RESAMPLING_OPERATION_SEMANTICS
    kernel.target_label = RESKDM_IWSG_FINITE_TARGET
    kernel.bandwidth_policy = BANDWIDTH_POLICY
    kernel.incoming_weight_policy = INCOMING_WEIGHT_POLICY
    kernel.covariance_mark_policy = covariance_mark_policy
    kernel.derivative_semantics = DERIVATIVE_SEMANTICS
    kernel.score_output_semantics = SCORE_OUTPUT_SEMANTICS
    kernel.source_scope = SOURCE_SCOPE
    kernel.normalization_tolerance_policy = NORMALIZATION_TOLERANCE_POLICY
    kernel.effective_normalization_tolerance = effective_normalization_tolerance
    kernel.mode = "replay"
    kernel.jit_compile = bool(jit_compile)
    return kernel


__all__ = [
    "ANCHOR_POLICY",
    "BANDWIDTH_POLICY",
    "COVARIANCE_MARK_POLICY",
    "COVARIANCE_MARK_POLICIES",
    "DERIVATIVE_SEMANTICS",
    "INCOMING_WEIGHT_POLICY",
    "REPLAY_POLICY",
    "RESAMPLING_OPERATION_SEMANTICS",
    "RESAMPLING_ROUTE_CLASSIFICATION",
    "RESAMPLING_ROUTE_ID",
    "SCORE_OUTPUT_SEMANTICS",
    "RESPONSIBILITY_COVARIANCE_MARK_POLICY",
    "SELECTED_LABEL_COVARIANCE_MARK_POLICY",
    "SOURCE_SCOPE",
    "make_resampling_anchor_kernel",
    "make_resampling_replay_kernel",
]

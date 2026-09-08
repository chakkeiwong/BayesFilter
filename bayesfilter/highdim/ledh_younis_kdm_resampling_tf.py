"""Phase 4B full-mixture IWSG resampling after the canonical LEDH reset.

This diagnostic route executes Contract-E and the GenUT dual-cap correction,
then applies a Younis-style marginalized Gaussian-mixture resampler. It defines
the separate RESKDM-FINITE fixed-anchor replay scalar; it is not another
derivative implementation for the canonical ATOM-FINITE scalar.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    _value_and_analytical_score_impl,
)
from bayesfilter.highdim.ledh_younis_kdm_tf import (
    RESKDM_FINITE_TARGET,
    make_full_mixture_iwsg_resampling_kernel,
    make_gaussian_kdm_kernel,
)


Tensor = tf.Tensor

RESAMPLING_ROUTE_ID = "ledh_contract_e_then_younis_iwsg_resampling_reference_v1"
RESAMPLING_ROUTE_CLASSIFICATION = (
    "combined_contract_e_full_mixture_iwsg_diagnostic_only"
)
RESAMPLING_OPERATION_SEMANTICS = (
    "contract_e_dual_cap_then_fixed_anchor_marginal_kdm_resampling_v1"
)
COVARIANCE_MARK_POLICY = "responsibility_conditional_mean_no_scatter_v1"
ANCHOR_POLICY = "sequential_stratified_fixed_sample_bank_v1"
REPLAY_POLICY = "fixed_samples_and_fixed_marginal_proposal_density_v1"


def _positive_dimension(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


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
        bandwidth_covariances
        + tf.linalg.matrix_transpose(bandwidth_covariances)
    )
    scale = tf.maximum(
        tf.reduce_max(tf.abs(symmetric), axis=[-2, -1]),
        tf.ones([tf.shape(component_means)[0]], dtype),
    )
    symmetry_error = tf.reduce_max(
        tf.abs(
            bandwidth_covariances
            - tf.linalg.matrix_transpose(bandwidth_covariances)
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
    stratified_uniforms: Tensor | None,
    kdm_standard_noises: Tensor | None,
    resampling_kernel: Callable[..., Mapping[str, Tensor]],
    density_kernel: Callable[..., Mapping[str, Tensor]],
    rank_tolerance: float,
    normalization_tolerance: float,
    symmetry_tolerance: float,
) -> Mapping[str, Any]:
    options = _validated_options(canonical_options)
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
        [horizon, particle_count, dimension, dimension],
    )
    tf.ensure_shape(
        d_bandwidths,
        [horizon, particle_count, dimension, dimension],
    )

    if anchor_mode:
        if stratified_uniforms is None or kdm_standard_noises is None:
            raise ValueError("anchor mode requires stratified uniforms and KDM noises")
        stratified_uniforms = tf.convert_to_tensor(stratified_uniforms, dtype)
        kdm_standard_noises = tf.convert_to_tensor(kdm_standard_noises, dtype)
        tf.ensure_shape(stratified_uniforms, [horizon, particle_count])
        tf.ensure_shape(
            kdm_standard_noises, [horizon, particle_count, dimension]
        )
    else:
        if fixed_samples is None or fixed_proposal_log_densities is None:
            raise ValueError("replay mode requires a fixed sample/proposal bank")
        fixed_samples = tf.convert_to_tensor(fixed_samples, dtype)
        fixed_proposal_log_densities = tf.convert_to_tensor(
            fixed_proposal_log_densities, dtype
        )
        tf.ensure_shape(fixed_samples, [horizon, particle_count, dimension])
        tf.ensure_shape(
            fixed_proposal_log_densities, [horizon, particle_count]
        )

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
        bandwidth = bandwidths[time_index]
        d_bandwidth = d_bandwidths[time_index]
        if anchor_mode:
            uniforms = stratified_uniforms[time_index]
            uniform_valid = tf.reduce_all(
                tf.math.is_finite(uniforms)
                & (uniforms >= tf.zeros_like(uniforms))
                & (uniforms < tf.ones_like(uniforms))
            )
            upper = tf.cast(
                1.0 - (1.0e-7 if dtype == tf.float32 else 1.0e-15), dtype
            )
            safe_uniforms = tf.clip_by_value(
                uniforms, tf.zeros([], dtype), upper
            )
            positions = (
                tf.cast(tf.range(particle_count), dtype) + safe_uniforms
            ) / tf.cast(particle_count, dtype)
            component_indices = tf.minimum(
                tf.searchsorted(uniform_cumulative, positions, side="right"),
                particle_count - 1,
            )
            samples, bandwidth_valid, minimum_bandwidth_eigenvalue, bandwidth_symmetry_error = (
                _bandwidth_sample(
                    reset_states,
                    bandwidth,
                    component_indices,
                    kdm_standard_noises[time_index],
                    rank_tolerance=rank_tolerance,
                    symmetry_tolerance=symmetry_tolerance,
                )
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
            component_indices = tf.fill([particle_count], tf.constant(-1, tf.int32))
            sample_bank_valid = (
                tf.reduce_all(tf.math.is_finite(samples))
                & tf.reduce_all(tf.math.is_finite(proposal_log_density))
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
        step_valid = sample_bank_valid & resampling["valid"]
        record = {
            "kdm_component_means": reset_states,
            "d_kdm_component_means": d_reset_states,
            "kdm_source_covariance_marks": reset_covariance_marks,
            "d_kdm_source_covariance_marks": d_reset_covariance_marks,
            "kdm_samples": samples,
            "kdm_component_indices": component_indices,
            "kdm_proposal_log_density": proposal_log_density,
            "kdm_log_density": resampling["log_density"],
            "d_kdm_log_density": resampling["d_log_density"][0],
            "kdm_log_ratio": resampling["log_ratio"],
            "kdm_responsibilities": resampling["responsibilities"],
            "d_kdm_responsibilities": resampling["d_responsibilities"][0],
            "kdm_normalized_weights": resampling["normalized_weights"],
            "d_kdm_normalized_weights": resampling[
                "d_normalized_weights"
            ][0],
            "kdm_covariance_marks": resampling[
                "transported_covariance_marks"
            ],
            "d_kdm_covariance_marks": resampling[
                "d_transported_covariance_marks"
            ][0],
            "kdm_step_valid": step_valid,
            "kdm_pair_count": resampling["complexity_pair_count"],
            "kdm_anchor_log_ratio_error": resampling[
                "anchor_log_ratio_error"
            ],
            "kdm_responsibility_row_sum_error": resampling[
                "responsibility_row_sum_error"
            ],
            "kdm_weight_sum_error": resampling["weight_sum_error"],
            "kdm_weight_tangent_sum_error": resampling[
                "weight_tangent_sum_error"
            ],
            "kdm_minimum_bandwidth_eigenvalue": (
                minimum_bandwidth_eigenvalue
            ),
            "kdm_bandwidth_symmetry_error": bandwidth_symmetry_error,
        }
        resampling_records.append(record)
        return (
            samples,
            tf.zeros_like(samples),
            resampling["transported_covariance_marks"],
            resampling["d_transported_covariance_marks"][0],
            resampling["normalized_log_weights"],
            resampling["d_normalized_log_weights"][0],
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
    return {
        "route_id": RESAMPLING_ROUTE_ID,
        "route_classification": RESAMPLING_ROUTE_CLASSIFICATION,
        "operation_semantics": RESAMPLING_OPERATION_SEMANTICS,
        "target_label": RESKDM_FINITE_TARGET,
        "anchor_policy": ANCHOR_POLICY,
        "replay_policy": REPLAY_POLICY,
        "covariance_mark_policy": COVARIANCE_MARK_POLICY,
        "mode": "anchor" if anchor_mode else "replay",
        "value": value,
        "score": score,
        "valid": (
            tf.math.is_finite(value)
            & tf.reduce_all(tf.math.is_finite(score))
            & tf.reduce_all(step_valid)
        ),
        "step_valid": step_valid,
        "trace": trace,
        "fixed_samples": _stack_trace(trace, "kdm_samples"),
        "fixed_proposal_log_densities": _stack_trace(
            trace, "kdm_proposal_log_density"
        ),
        "component_indices": _stack_trace(trace, "kdm_component_indices"),
        "component_means": _stack_trace(trace, "kdm_component_means"),
        "d_component_means": _stack_trace(trace, "d_kdm_component_means"),
        "responsibilities": _stack_trace(trace, "kdm_responsibilities"),
        "d_responsibilities": _stack_trace(trace, "d_kdm_responsibilities"),
        "normalized_weights": _stack_trace(trace, "kdm_normalized_weights"),
        "d_normalized_weights": _stack_trace(
            trace, "d_kdm_normalized_weights"
        ),
        "covariance_marks": _stack_trace(trace, "kdm_covariance_marks"),
        "d_covariance_marks": _stack_trace(trace, "d_kdm_covariance_marks"),
        "incoming_log_weights": _stack_trace(trace, "incoming_log_weights"),
        "d_incoming_log_weights": _stack_trace(
            trace, "d_incoming_log_weights"
        ),
        "outgoing_log_weights": _stack_trace(trace, "outgoing_log_weights"),
        "d_outgoing_log_weights": _stack_trace(
            trace, "d_outgoing_log_weights"
        ),
        "prior_observation_logits": _stack_trace(
            trace, "prior_observation_logits"
        ),
        "d_prior_observation_logits": _stack_trace(
            trace, "d_prior_observation_logits"
        ),
        "predicted_covariances": _stack_trace(
            trace, "predicted_covariances"
        ),
        "d_predicted_covariances": _stack_trace(
            trace, "d_predicted_covariances"
        ),
        "states_after_resampling": _stack_trace(trace, "states_after_reset"),
        "d_states_after_resampling": _stack_trace(
            trace, "d_states_after_reset"
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
        "responsibilities",
        "d_responsibilities",
        "normalized_weights",
        "d_normalized_weights",
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
        "pair_count",
        "maximum_anchor_log_ratio_error",
        "maximum_responsibility_row_sum_error",
        "maximum_weight_sum_error",
        "maximum_weight_tangent_sum_error",
        "minimum_bandwidth_eigenvalue",
        "maximum_bandwidth_symmetry_error",
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
    options = _validated_options(canonical_options)
    resampling_kernel, density_kernel = _factory_components(
        particle_count=particle_count,
        state_dimension=state_dimension,
        dtype=dtype,
        rank_tolerance=rank_tolerance,
        normalization_tolerance=normalization_tolerance,
        symmetry_tolerance=symmetry_tolerance,
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec([theta_dimension], dtype),
            tf.TensorSpec([particle_count, state_dimension], dtype),
            tf.TensorSpec(
                [particle_count, state_dimension, state_dimension], dtype
            ),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, observation_dimension], dtype),
            tf.TensorSpec(
                [horizon, particle_count, state_dimension, state_dimension],
                dtype,
            ),
            tf.TensorSpec(
                [horizon, particle_count, state_dimension, state_dimension],
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
            stratified_uniforms=stratified_uniforms,
            kdm_standard_noises=kdm_standard_noises,
            resampling_kernel=resampling_kernel,
            density_kernel=density_kernel,
            rank_tolerance=rank_tolerance,
            normalization_tolerance=normalization_tolerance,
            symmetry_tolerance=symmetry_tolerance,
        )
        return _tensor_result(result)

    kernel.route_id = RESAMPLING_ROUTE_ID
    kernel.route_classification = RESAMPLING_ROUTE_CLASSIFICATION
    kernel.operation_semantics = RESAMPLING_OPERATION_SEMANTICS
    kernel.target_label = RESKDM_FINITE_TARGET
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
    options = _validated_options(canonical_options)
    resampling_kernel, density_kernel = _factory_components(
        particle_count=particle_count,
        state_dimension=state_dimension,
        dtype=dtype,
        rank_tolerance=rank_tolerance,
        normalization_tolerance=normalization_tolerance,
        symmetry_tolerance=symmetry_tolerance,
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec([theta_dimension], dtype),
            tf.TensorSpec([particle_count, state_dimension], dtype),
            tf.TensorSpec(
                [particle_count, state_dimension, state_dimension], dtype
            ),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, observation_dimension], dtype),
            tf.TensorSpec(
                [horizon, particle_count, state_dimension, state_dimension],
                dtype,
            ),
            tf.TensorSpec(
                [horizon, particle_count, state_dimension, state_dimension],
                dtype,
            ),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, particle_count], dtype),
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
            stratified_uniforms=None,
            kdm_standard_noises=None,
            resampling_kernel=resampling_kernel,
            density_kernel=density_kernel,
            rank_tolerance=rank_tolerance,
            normalization_tolerance=normalization_tolerance,
            symmetry_tolerance=symmetry_tolerance,
        )
        return _tensor_result(result)

    kernel.route_id = RESAMPLING_ROUTE_ID
    kernel.route_classification = RESAMPLING_ROUTE_CLASSIFICATION
    kernel.operation_semantics = RESAMPLING_OPERATION_SEMANTICS
    kernel.target_label = RESKDM_FINITE_TARGET
    kernel.mode = "replay"
    kernel.jit_compile = bool(jit_compile)
    return kernel


__all__ = [
    "ANCHOR_POLICY",
    "COVARIANCE_MARK_POLICY",
    "REPLAY_POLICY",
    "RESAMPLING_OPERATION_SEMANTICS",
    "RESAMPLING_ROUTE_CLASSIFICATION",
    "RESAMPLING_ROUTE_ID",
    "make_resampling_anchor_kernel",
    "make_resampling_replay_kernel",
]

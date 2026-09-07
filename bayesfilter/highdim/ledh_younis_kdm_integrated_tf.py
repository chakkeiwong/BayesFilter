"""Integrated kernelized-observation score for the full LEDH executor.

This module implements Phase 4A of the Younis-KDM score investigation.  It
replaces the point observation factor by its exact linear-Gaussian integral
under a declared Gaussian state kernel, then sends the resulting weights and
their total tangent through the shared canonical Contract-E/GenUT/dual-cap
executor.  It is a KDM-FINITE diagnostic, not the canonical atom target and
not a complete Younis mixture-density particle filter.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    _value_and_analytical_score_impl,
)
from bayesfilter.highdim.ledh_younis_kdm_tf import (
    ATOM_FINITE_TARGET,
    KDM_FINITE_TARGET,
)


Tensor = tf.Tensor

INTEGRATED_OBSERVATION_ROUTE_ID = (
    "ledh_younis_kdm_integrated_observation_weighting_v1"
)
INTEGRATED_OBSERVATION_CLASSIFICATION = (
    "kdm_finite_full_feedback_diagnostic_only"
)
INTEGRATED_OBSERVATION_SEMANTICS = (
    "post_ledh_pre_observation_gaussian_measure_convolution_v1"
)


def _dtype_machine_epsilon(dtype: tf.dtypes.DType) -> float:
    if dtype == tf.float16:
        return 2.0**-10
    if dtype == tf.bfloat16:
        return 2.0**-7
    if dtype == tf.float32:
        return 2.0**-23
    if dtype == tf.float64:
        return 2.0**-52
    raise ValueError(f"unsupported floating dtype: {dtype.name}")


def _finite_except_leading(value: Tensor) -> Tensor:
    value = tf.convert_to_tensor(value)
    if value.shape.rank is None or value.shape.rank < 1:
        raise ValueError("expected a statically ranked tensor")
    if value.shape.rank == 1:
        return tf.math.is_finite(value)
    return tf.reduce_all(
        tf.math.is_finite(value), axis=list(range(1, value.shape.rank))
    )


def linear_gaussian_kdm_observation_factors(
    states: Tensor,
    d_states: Tensor,
    bandwidths: Tensor,
    d_bandwidths: Tensor,
    observation_matrix: Tensor,
    d_observation_matrix: Tensor,
    observation_covariance: Tensor,
    d_observation_covariance: Tensor,
    observation: Tensor,
    *,
    bandwidth_is_zero: bool,
    covariance_tolerance: float = 1.0e-10,
) -> Mapping[str, Tensor]:
    """Return convolved component log factors and one total tangent.

    Shapes are ``states [N,D]``, ``bandwidths [N,D,D]``, ``C [O,D]``,
    ``R [O,O]``, and matching tangent shapes without a direction axis.  A
    singular positive-semidefinite state bandwidth is permitted: only the
    effective observation covariance is factored.
    """

    if float(covariance_tolerance) <= 0.0:
        raise ValueError("covariance_tolerance must be positive")
    states = tf.convert_to_tensor(states)
    dtype = states.dtype
    if not dtype.is_floating:
        raise ValueError("states must use a floating TensorFlow dtype")
    d_states = tf.convert_to_tensor(d_states, dtype)
    bandwidths = tf.convert_to_tensor(bandwidths, dtype)
    d_bandwidths = tf.convert_to_tensor(d_bandwidths, dtype)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype)
    d_observation_matrix = tf.convert_to_tensor(d_observation_matrix, dtype)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype)
    d_observation_covariance = tf.convert_to_tensor(
        d_observation_covariance, dtype
    )
    observation = tf.convert_to_tensor(observation, dtype)

    particle_count = int(states.shape[0])
    state_dimension = int(states.shape[1])
    observation_dimension = int(observation.shape[0])
    tf.ensure_shape(d_states, [particle_count, state_dimension])
    tf.ensure_shape(
        bandwidths, [particle_count, state_dimension, state_dimension]
    )
    tf.ensure_shape(
        d_bandwidths, [particle_count, state_dimension, state_dimension]
    )
    tf.ensure_shape(
        observation_matrix, [observation_dimension, state_dimension]
    )
    tf.ensure_shape(
        d_observation_matrix, [observation_dimension, state_dimension]
    )
    tf.ensure_shape(
        observation_covariance,
        [observation_dimension, observation_dimension],
    )
    tf.ensure_shape(
        d_observation_covariance,
        [observation_dimension, observation_dimension],
    )

    effective_covariance_tolerance = max(
        float(covariance_tolerance),
        64.0 * _dtype_machine_epsilon(dtype),
    )
    tolerance = tf.constant(effective_covariance_tolerance, dtype)
    bandwidth_transpose = tf.linalg.matrix_transpose(bandwidths)
    d_bandwidth_transpose = tf.linalg.matrix_transpose(d_bandwidths)
    bandwidth_symmetry_error = tf.reduce_max(
        tf.abs(bandwidths - bandwidth_transpose), axis=[-2, -1]
    )
    d_bandwidth_symmetry_error = tf.reduce_max(
        tf.abs(d_bandwidths - d_bandwidth_transpose), axis=[-2, -1]
    )
    bandwidth_scale = tf.maximum(
        tf.reduce_max(tf.abs(bandwidths), axis=[-2, -1]),
        tf.ones([particle_count], dtype),
    )
    d_bandwidth_scale = tf.maximum(
        tf.reduce_max(tf.abs(d_bandwidths), axis=[-2, -1]),
        tf.ones([particle_count], dtype),
    )
    bandwidth_symmetric = bandwidth_symmetry_error <= tolerance * bandwidth_scale
    d_bandwidth_symmetric = (
        d_bandwidth_symmetry_error <= tolerance * d_bandwidth_scale
    )
    symmetric_bandwidths = 0.5 * (bandwidths + bandwidth_transpose)
    symmetric_d_bandwidths = 0.5 * (
        d_bandwidths + d_bandwidth_transpose
    )
    bandwidth_eigenvalues = tf.linalg.eigvalsh(symmetric_bandwidths)
    minimum_bandwidth_eigenvalue = tf.reduce_min(
        bandwidth_eigenvalues, axis=-1
    )
    bandwidth_psd = minimum_bandwidth_eigenvalue >= -tolerance * bandwidth_scale

    if bandwidth_is_zero:
        bandwidth_branch_valid = tf.reduce_all(
            tf.abs(symmetric_bandwidths) <= tolerance, axis=[-2, -1]
        ) & tf.reduce_all(
            tf.abs(symmetric_d_bandwidths) <= tolerance, axis=[-2, -1]
        )
    else:
        # A mixed discrete/continuous component family is valid.  The KDM
        # branch only requires that the collection is not the all-zero atom
        # limit; individual PSD components may have zero bandwidth.
        any_nonzero_bandwidth = tf.reduce_any(
            tf.abs(symmetric_bandwidths) > tolerance
        )
        bandwidth_branch_valid = tf.fill(
            [particle_count], any_nonzero_bandwidth
        )

    covariance_transpose = tf.linalg.matrix_transpose(observation_covariance)
    d_covariance_transpose = tf.linalg.matrix_transpose(
        d_observation_covariance
    )
    covariance_symmetry_error = tf.reduce_max(
        tf.abs(observation_covariance - covariance_transpose)
    )
    d_covariance_symmetry_error = tf.reduce_max(
        tf.abs(d_observation_covariance - d_covariance_transpose)
    )
    covariance_scale = tf.maximum(
        tf.reduce_max(tf.abs(observation_covariance)),
        tf.constant(1.0, dtype),
    )
    d_covariance_scale = tf.maximum(
        tf.reduce_max(tf.abs(d_observation_covariance)),
        tf.constant(1.0, dtype),
    )
    covariance_symmetric = (
        covariance_symmetry_error <= tolerance * covariance_scale
    )
    d_covariance_symmetric = (
        d_covariance_symmetry_error <= tolerance * d_covariance_scale
    )
    symmetric_covariance = 0.5 * (
        observation_covariance + covariance_transpose
    )
    symmetric_d_covariance = 0.5 * (
        d_observation_covariance + d_covariance_transpose
    )
    minimum_observation_eigenvalue = tf.reduce_min(
        tf.linalg.eigvalsh(symmetric_covariance)
    )
    observation_covariance_spd = (
        minimum_observation_eigenvalue > tolerance * covariance_scale
    )

    effective_covariances = (
        symmetric_covariance[tf.newaxis, :, :]
        + tf.einsum(
            "od,ndf,pf->nop",
            observation_matrix,
            symmetric_bandwidths,
            observation_matrix,
        )
    )
    effective_covariances = 0.5 * (
        effective_covariances
        + tf.linalg.matrix_transpose(effective_covariances)
    )
    effective_d_covariances = (
        symmetric_d_covariance[tf.newaxis, :, :]
        + tf.einsum(
            "od,ndf,pf->nop",
            d_observation_matrix,
            symmetric_bandwidths,
            observation_matrix,
        )
        + tf.einsum(
            "od,ndf,pf->nop",
            observation_matrix,
            symmetric_d_bandwidths,
            observation_matrix,
        )
        + tf.einsum(
            "od,ndf,pf->nop",
            observation_matrix,
            symmetric_bandwidths,
            d_observation_matrix,
        )
    )
    effective_d_covariances = 0.5 * (
        effective_d_covariances
        + tf.linalg.matrix_transpose(effective_d_covariances)
    )
    effective_scale = tf.maximum(
        tf.reduce_max(tf.abs(effective_covariances), axis=[-2, -1]),
        tf.ones([particle_count], dtype),
    )
    minimum_effective_eigenvalue = tf.reduce_min(
        tf.linalg.eigvalsh(effective_covariances), axis=-1
    )
    effective_spd = (
        minimum_effective_eigenvalue > tolerance * effective_scale
    )

    finite = (
        _finite_except_leading(states)
        & _finite_except_leading(d_states)
        & _finite_except_leading(bandwidths)
        & _finite_except_leading(d_bandwidths)
        & tf.reduce_all(tf.math.is_finite(observation_matrix))
        & tf.reduce_all(tf.math.is_finite(d_observation_matrix))
        & tf.reduce_all(tf.math.is_finite(observation_covariance))
        & tf.reduce_all(tf.math.is_finite(d_observation_covariance))
        & tf.reduce_all(tf.math.is_finite(observation))
    )
    component_valid = (
        finite
        & bandwidth_symmetric
        & d_bandwidth_symmetric
        & bandwidth_psd
        & bandwidth_branch_valid
        & covariance_symmetric
        & d_covariance_symmetric
        & observation_covariance_spd
        & effective_spd
    )

    identity = tf.eye(observation_dimension, dtype=dtype)
    safe_covariances = tf.where(
        effective_spd[:, tf.newaxis, tf.newaxis],
        effective_covariances,
        identity[tf.newaxis, :, :],
    )
    factors = tf.linalg.cholesky(safe_covariances)
    means = tf.einsum("od,nd->no", observation_matrix, states)
    d_means = tf.einsum(
        "od,nd->no", d_observation_matrix, states
    ) + tf.einsum("od,nd->no", observation_matrix, d_states)
    residual = observation[tf.newaxis, :] - means
    solved = tf.linalg.triangular_solve(
        factors, residual[..., tf.newaxis], lower=True
    )[..., 0]
    precision_residual = tf.linalg.cholesky_solve(
        factors, residual[..., tf.newaxis]
    )[..., 0]
    log_determinant = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(factors)), axis=-1
    )
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype)
    log_factor = -0.5 * (
        tf.cast(observation_dimension, dtype) * log_two_pi
        + log_determinant
        + tf.reduce_sum(tf.square(solved), axis=-1)
    )
    covariance_quadratic = tf.einsum(
        "no,noe,ne->n",
        precision_residual,
        effective_d_covariances,
        precision_residual,
    )
    solved_covariance_tangent = tf.linalg.cholesky_solve(
        factors, effective_d_covariances
    )
    covariance_trace = tf.linalg.trace(solved_covariance_tangent)
    d_log_factor = (
        tf.reduce_sum(precision_residual * d_means, axis=-1)
        + 0.5 * (covariance_quadratic - covariance_trace)
    )

    nan = tf.cast(float("nan"), dtype)
    return {
        "log_factor": tf.where(
            component_valid, log_factor, tf.fill([particle_count], nan)
        ),
        "d_log_factor": tf.where(
            component_valid, d_log_factor, tf.fill([particle_count], nan)
        ),
        "component_valid": component_valid,
        "valid": tf.reduce_all(component_valid),
        "minimum_bandwidth_eigenvalue": tf.reduce_min(
            minimum_bandwidth_eigenvalue
        ),
        "minimum_observation_eigenvalue": minimum_observation_eigenvalue,
        "minimum_effective_eigenvalue": tf.reduce_min(
            minimum_effective_eigenvalue
        ),
        "maximum_bandwidth_symmetry_error": tf.reduce_max(
            bandwidth_symmetry_error
        ),
        "maximum_bandwidth_tangent_symmetry_error": tf.reduce_max(
            d_bandwidth_symmetry_error
        ),
        "effective_covariance_tolerance": tolerance,
    }


def integrated_linear_gaussian_kdm_value_and_analytical_score(
    model: Any,
    theta: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    observation_matrix: Tensor,
    d_observation_matrix: Tensor,
    bandwidths: Tensor,
    d_bandwidths: Tensor,
    *,
    canonical_options: Mapping[str, Any],
    bandwidth_is_zero: bool = False,
    model_tolerance: float = 1.0e-8,
    covariance_tolerance: float = 1.0e-10,
) -> Mapping[str, Any]:
    """Run Phase 4A through the shared full LEDH analytical executor."""

    if float(model_tolerance) <= 0.0:
        raise ValueError("model_tolerance must be positive")
    options = dict(canonical_options)
    for forbidden in (
        "with_score",
        "return_trace",
        "observation_factor_override",
    ):
        if forbidden in options:
            raise ValueError(f"canonical_options must not contain {forbidden}")
    if options.get("reset_policy") != "contract_e":
        raise ValueError("integrated KDM requires reset_policy='contract_e'")
    if options.get("reset_design") is None:
        raise ValueError("integrated KDM requires an explicit Contract-E design")
    if int(options.get("annealed_stages", 1)) != 1:
        raise ValueError("integrated KDM currently requires annealed_stages=1")
    if int(options.get("correction_steps", 0)) < 1:
        raise ValueError("integrated KDM requires the diagonal correction")
    if int(options.get("pairwise_steps", 0)) < 1:
        raise ValueError("integrated KDM requires the pairwise correction")
    if float(options.get("coordinate_cap", 0.0)) <= 0.0:
        raise ValueError("integrated KDM requires the coordinate cap")

    theta = tf.convert_to_tensor(theta)
    initial_states = tf.convert_to_tensor(initial_states)
    dtype = initial_states.dtype
    initial_covariances = tf.convert_to_tensor(initial_covariances, dtype)
    noises = tf.convert_to_tensor(noises, dtype)
    observations = tf.convert_to_tensor(observations, dtype)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype)
    d_observation_matrix = tf.convert_to_tensor(d_observation_matrix, dtype)
    bandwidths = tf.convert_to_tensor(bandwidths, dtype)
    d_bandwidths = tf.convert_to_tensor(d_bandwidths, dtype)

    horizon = int(observations.shape[0])
    particle_count = int(initial_states.shape[0])
    state_dimension = int(initial_states.shape[1])
    observation_dimension = int(observations.shape[1])
    tf.ensure_shape(
        observation_matrix, [observation_dimension, state_dimension]
    )
    tf.ensure_shape(
        d_observation_matrix, [observation_dimension, state_dimension]
    )
    tf.ensure_shape(
        bandwidths,
        [horizon, particle_count, state_dimension, state_dimension],
    )
    tf.ensure_shape(
        d_bandwidths,
        [horizon, particle_count, state_dimension, state_dimension],
    )

    d_observation_covariance = (
        model.observation_covariance_tangent_fn(theta)
        if model.observation_covariance_tangent_fn is not None
        else tf.zeros_like(model.observation_covariance)
    )
    factor_records: list[Mapping[str, Tensor]] = []
    atom_value_errors: list[Tensor] = []
    atom_tangent_errors: list[Tensor] = []
    atom_identity_valid: list[Tensor] = []
    observation_map_value_errors: list[Tensor] = []
    observation_map_tangent_errors: list[Tensor] = []
    observation_map_valid: list[Tensor] = []
    effective_model_tolerance = max(
        float(model_tolerance),
        64.0 * _dtype_machine_epsilon(dtype),
    )

    def observation_factor_override(
        time_index: int,
        points: Tensor,
        d_points: Tensor,
        observation: Tensor,
        base_log_factor: Tensor,
        d_base_log_factor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected_observation = tf.einsum(
            "od,nd->no", observation_matrix, points
        )
        expected_observation_tangent = (
            tf.einsum("od,nd->no", d_observation_matrix, points)
            + tf.einsum("od,nd->no", observation_matrix, d_points)
        )
        actual_observation = model.observation_fn(points)
        actual_observation_tangent = model.observation_tangent_fn(
            points, d_points
        )
        observation_map_value_error = tf.reduce_max(
            tf.abs(actual_observation - expected_observation)
        )
        observation_map_tangent_error = tf.reduce_max(
            tf.abs(
                actual_observation_tangent
                - expected_observation_tangent
            )
        )
        observation_map_value_scale = tf.maximum(
            tf.reduce_max(tf.abs(actual_observation)),
            tf.constant(1.0, dtype),
        )
        observation_map_tangent_scale = tf.maximum(
            tf.reduce_max(tf.abs(actual_observation_tangent)),
            tf.constant(1.0, dtype),
        )
        model_tolerance_tensor = tf.constant(effective_model_tolerance, dtype)
        tf.debugging.assert_less_equal(
            observation_map_value_error,
            model_tolerance_tensor * observation_map_value_scale,
            message=(
                "supplied observation matrix does not match the model's "
                "executed observation map"
            ),
        )
        tf.debugging.assert_less_equal(
            observation_map_tangent_error,
            model_tolerance_tensor * observation_map_tangent_scale,
            message=(
                "supplied observation-matrix tangent does not match the "
                "model's executed observation tangent"
            ),
        )
        zero_bandwidths = tf.zeros_like(bandwidths[time_index])
        atom = linear_gaussian_kdm_observation_factors(
            points,
            d_points,
            zero_bandwidths,
            tf.zeros_like(zero_bandwidths),
            observation_matrix,
            d_observation_matrix,
            model.observation_covariance,
            d_observation_covariance,
            observation,
            bandwidth_is_zero=True,
            covariance_tolerance=covariance_tolerance,
        )
        value_error = tf.reduce_max(
            tf.abs(atom["log_factor"] - base_log_factor)
        )
        tangent_error = tf.reduce_max(
            tf.abs(atom["d_log_factor"] - d_base_log_factor)
        )
        value_scale = tf.maximum(
            tf.reduce_max(tf.abs(base_log_factor)), tf.constant(1.0, dtype)
        )
        tangent_scale = tf.maximum(
            tf.reduce_max(tf.abs(d_base_log_factor)), tf.constant(1.0, dtype)
        )
        tolerance = tf.constant(effective_model_tolerance, dtype)
        tf.debugging.assert_equal(
            atom["valid"],
            True,
            message="explicit Gaussian atom factor is invalid",
        )
        tf.debugging.assert_less_equal(
            value_error,
            tolerance * value_scale,
            message=(
                "supplied Gaussian observation representation does not match "
                "the model's atom log density"
            ),
        )
        tf.debugging.assert_less_equal(
            tangent_error,
            tolerance * tangent_scale,
            message=(
                "supplied Gaussian observation tangent does not match the "
                "model's atom log-density tangent"
            ),
        )
        atom_identity_valid.append(
            (value_error <= tolerance * value_scale)
            & (tangent_error <= tolerance * tangent_scale)
        )
        result = linear_gaussian_kdm_observation_factors(
            points,
            d_points,
            bandwidths[time_index],
            d_bandwidths[time_index],
            observation_matrix,
            d_observation_matrix,
            model.observation_covariance,
            d_observation_covariance,
            observation,
            bandwidth_is_zero=bandwidth_is_zero,
            covariance_tolerance=covariance_tolerance,
        )
        tf.debugging.assert_equal(
            result["valid"],
            True,
            message="invalid KDM observation factor",
        )
        factor_records.append(result)
        atom_value_errors.append(value_error)
        atom_tangent_errors.append(tangent_error)
        observation_map_value_errors.append(observation_map_value_error)
        observation_map_tangent_errors.append(
            observation_map_tangent_error
        )
        observation_map_valid.append(
            (observation_map_value_error
             <= model_tolerance_tensor * observation_map_value_scale)
            & (observation_map_tangent_error
               <= model_tolerance_tensor * observation_map_tangent_scale)
        )
        return result["log_factor"], result["d_log_factor"]

    value, score, trace = _value_and_analytical_score_impl(
        model,
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        with_score=True,
        return_trace=True,
        observation_factor_override=observation_factor_override,
        **options,
    )
    factor_valid = tf.stack([record["valid"] for record in factor_records])
    atom_identity_valid_tensor = tf.stack(atom_identity_valid)
    observation_map_valid_tensor = tf.stack(observation_map_valid)
    minimum_bandwidth_eigenvalue = tf.reduce_min(
        tf.stack(
            [
                record["minimum_bandwidth_eigenvalue"]
                for record in factor_records
            ]
        )
    )
    minimum_effective_eigenvalue = tf.reduce_min(
        tf.stack(
            [
                record["minimum_effective_eigenvalue"]
                for record in factor_records
            ]
        )
    )
    target_label = (
        ATOM_FINITE_TARGET if bandwidth_is_zero else KDM_FINITE_TARGET
    )
    return {
        "route_id": INTEGRATED_OBSERVATION_ROUTE_ID,
        "route_classification": INTEGRATED_OBSERVATION_CLASSIFICATION,
        "operation_semantics": INTEGRATED_OBSERVATION_SEMANTICS,
        "target_label": target_label,
        "value": value,
        "score": score,
        "trace": trace,
        "factor_records": tuple(factor_records),
        "factor_valid": factor_valid,
        "valid": (
            tf.math.is_finite(value)
            & tf.reduce_all(tf.math.is_finite(score))
            & tf.reduce_all(factor_valid)
            & tf.reduce_all(atom_identity_valid_tensor)
            & tf.reduce_all(observation_map_valid_tensor)
        ),
        "atom_identity_valid": atom_identity_valid_tensor,
        "observation_map_valid": observation_map_valid_tensor,
        "kernel_feedback_into_reset": tf.constant(True),
        "complete_mixture_posterior": tf.constant(False),
        "bandwidth_is_zero": tf.constant(bool(bandwidth_is_zero)),
        "atom_factor_value_error_max": tf.reduce_max(
            tf.stack(atom_value_errors)
        ),
        "atom_factor_tangent_error_max": tf.reduce_max(
            tf.stack(atom_tangent_errors)
        ),
        "observation_map_value_error_max": tf.reduce_max(
            tf.stack(observation_map_value_errors)
        ),
        "observation_map_tangent_error_max": tf.reduce_max(
            tf.stack(observation_map_tangent_errors)
        ),
        "minimum_bandwidth_eigenvalue": minimum_bandwidth_eigenvalue,
        "minimum_effective_eigenvalue": minimum_effective_eigenvalue,
        "effective_model_tolerance": tf.constant(
            effective_model_tolerance, dtype
        ),
        "effective_covariance_tolerance": factor_records[0][
            "effective_covariance_tolerance"
        ],
    }


def make_integrated_linear_gaussian_kdm_kernel(
    model: Any,
    *,
    theta_dimension: int,
    particle_count: int,
    state_dimension: int,
    observation_dimension: int,
    horizon: int,
    canonical_options: Mapping[str, Any],
    bandwidth_is_zero: bool = False,
    dtype: tf.dtypes.DType | str = tf.float64,
    jit_compile: bool = True,
    model_tolerance: float = 1.0e-8,
    covariance_tolerance: float = 1.0e-10,
) -> Callable[..., Mapping[str, Tensor]]:
    """Build the fixed-shape complete Phase 4A TensorFlow endpoint.

    The model and numerical controls are captured as scope configuration;
    values, fixed streams, observations, Gaussian representation, and their
    declared tangents are explicit tensor inputs.  The returned trajectory
    stacks are executable evidence that the compiled endpoint reached the
    shared reset and later-time recurrence.
    """

    dimensions = {
        "theta_dimension": theta_dimension,
        "particle_count": particle_count,
        "state_dimension": state_dimension,
        "observation_dimension": observation_dimension,
        "horizon": horizon,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        dimensions[name] = int(value)
    dtype = tf.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError("dtype must be a floating TensorFlow dtype")
    options = dict(canonical_options)

    signature = [
        tf.TensorSpec([dimensions["theta_dimension"]], dtype),
        tf.TensorSpec(
            [dimensions["particle_count"], dimensions["state_dimension"]],
            dtype,
        ),
        tf.TensorSpec(
            [
                dimensions["particle_count"],
                dimensions["state_dimension"],
                dimensions["state_dimension"],
            ],
            dtype,
        ),
        tf.TensorSpec(
            [
                dimensions["horizon"],
                dimensions["particle_count"],
                dimensions["state_dimension"],
            ],
            dtype,
        ),
        tf.TensorSpec(
            [dimensions["horizon"], dimensions["observation_dimension"]],
            dtype,
        ),
        tf.TensorSpec(
            [dimensions["observation_dimension"], dimensions["state_dimension"]],
            dtype,
        ),
        tf.TensorSpec(
            [dimensions["observation_dimension"], dimensions["state_dimension"]],
            dtype,
        ),
        tf.TensorSpec(
            [
                dimensions["horizon"],
                dimensions["particle_count"],
                dimensions["state_dimension"],
                dimensions["state_dimension"],
            ],
            dtype,
        ),
        tf.TensorSpec(
            [
                dimensions["horizon"],
                dimensions["particle_count"],
                dimensions["state_dimension"],
                dimensions["state_dimension"],
            ],
            dtype,
        ),
    ]

    @tf.function(
        input_signature=signature,
        jit_compile=bool(jit_compile),
        autograph=False,
    )
    def kernel(
        theta: Tensor,
        initial_states: Tensor,
        initial_covariances: Tensor,
        noises: Tensor,
        observations: Tensor,
        observation_matrix: Tensor,
        d_observation_matrix: Tensor,
        bandwidths: Tensor,
        d_bandwidths: Tensor,
    ) -> Mapping[str, Tensor]:
        result = integrated_linear_gaussian_kdm_value_and_analytical_score(
            model,
            theta,
            initial_states,
            initial_covariances,
            noises,
            observations,
            observation_matrix,
            d_observation_matrix,
            bandwidths,
            d_bandwidths,
            canonical_options=options,
            bandwidth_is_zero=bandwidth_is_zero,
            model_tolerance=model_tolerance,
            covariance_tolerance=covariance_tolerance,
        )
        trace = result["trace"]
        return {
            "value": result["value"],
            "score": result["score"],
            "valid": result["valid"],
            "factor_valid": result["factor_valid"],
            "atom_identity_valid": result["atom_identity_valid"],
            "observation_map_valid": result["observation_map_valid"],
            "posterior_weights": tf.stack(
                [step["posterior_weights"] for step in trace]
            ),
            "pre_flow": tf.stack([step["pre_flow"] for step in trace]),
            "d_pre_flow": tf.stack(
                [step["d_pre_flow"] for step in trace]
            ),
            "children": tf.stack([step["children"] for step in trace]),
            "d_children": tf.stack(
                [step["d_children"] for step in trace]
            ),
            "prior_observation_logits": tf.stack(
                [step["prior_observation_logits"] for step in trace]
            ),
            "d_prior_observation_logits": tf.stack(
                [step["d_prior_observation_logits"] for step in trace]
            ),
            "observation_log_density": tf.stack(
                [step["observation_log_density"] for step in trace]
            ),
            "d_observation_log_density": tf.stack(
                [step["d_observation_log_density"] for step in trace]
            ),
            "posterior_logits": tf.stack(
                [step["posterior_logits"] for step in trace]
            ),
            "d_posterior_logits": tf.stack(
                [step["d_posterior_logits"] for step in trace]
            ),
            "predicted_covariances": tf.stack(
                [step["predicted_covariances"] for step in trace]
            ),
            "post_covariances": tf.stack(
                [step["post_covariances"] for step in trace]
            ),
            "covariances_after_reset": tf.stack(
                [step["covariances_after_reset"] for step in trace]
            ),
            "d_covariances_after_reset": tf.stack(
                [step["d_covariances_after_reset"] for step in trace]
            ),
            "reset_transport": tf.stack(
                [step["reset_transport"] for step in trace]
            ),
            "d_reset_transport": tf.stack(
                [step["d_reset_transport"] for step in trace]
            ),
            "d_posterior_weights": tf.stack(
                [step["d_posterior_weights"] for step in trace]
            ),
            "states_after_reset": tf.stack(
                [step["states_after_reset"] for step in trace]
            ),
            "d_states_after_reset": tf.stack(
                [step["d_states_after_reset"] for step in trace]
            ),
            "atom_factor_value_error_max": result[
                "atom_factor_value_error_max"
            ],
            "atom_factor_tangent_error_max": result[
                "atom_factor_tangent_error_max"
            ],
            "observation_map_value_error_max": result[
                "observation_map_value_error_max"
            ],
            "observation_map_tangent_error_max": result[
                "observation_map_tangent_error_max"
            ],
            "minimum_bandwidth_eigenvalue": result[
                "minimum_bandwidth_eigenvalue"
            ],
            "minimum_effective_eigenvalue": result[
                "minimum_effective_eigenvalue"
            ],
            "effective_model_tolerance": result[
                "effective_model_tolerance"
            ],
            "effective_covariance_tolerance": result[
                "effective_covariance_tolerance"
            ],
        }

    kernel.route_id = INTEGRATED_OBSERVATION_ROUTE_ID
    kernel.route_classification = INTEGRATED_OBSERVATION_CLASSIFICATION
    kernel.operation_semantics = INTEGRATED_OBSERVATION_SEMANTICS
    kernel.target_label = (
        ATOM_FINITE_TARGET if bandwidth_is_zero else KDM_FINITE_TARGET
    )
    kernel.bandwidth_is_zero = bool(bandwidth_is_zero)
    kernel.jit_compile = bool(jit_compile)
    kernel.complete_mixture_posterior = False
    return kernel


__all__ = [
    "INTEGRATED_OBSERVATION_CLASSIFICATION",
    "INTEGRATED_OBSERVATION_ROUTE_ID",
    "INTEGRATED_OBSERVATION_SEMANTICS",
    "integrated_linear_gaussian_kdm_value_and_analytical_score",
    "linear_gaussian_kdm_observation_factors",
    "make_integrated_linear_gaussian_kdm_kernel",
]

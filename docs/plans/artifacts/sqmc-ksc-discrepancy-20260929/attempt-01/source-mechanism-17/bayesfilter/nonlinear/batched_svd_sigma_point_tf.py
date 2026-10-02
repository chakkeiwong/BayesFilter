"""Production SVD sigma-point value/score custom-gradient bridge."""

from __future__ import annotations

from collections.abc import Mapping

import tensorflow as tf

from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedSVDBackend,
    TFPrincipalSqrtBackend,
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
    tf_batched_svd_sigma_point_value_and_score,
)

_PUBLIC_API_NAME = (
    "bayesfilter.nonlinear."
    "tf_batched_svd_sigma_point_value_and_score_custom_gradient"
)


def tf_batched_svd_sigma_point_value_and_score_custom_gradient(
    theta: tf.Tensor,
    observations: tf.Tensor,
    model: TFBatchedStructuralStateSpace,
    derivatives: TFBatchedStructuralFirstDerivatives,
    *,
    backend: TFBatchedSVDBackend = "tf_principal_sqrt_ukf",
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1.0e-12,
    rank_tolerance: tf.Tensor | float = 1.0e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1.0e-8,
    fixed_null_tolerance: tf.Tensor | float = 1.0e-10,
    principal_sqrt_reconstruction_tolerance: tf.Tensor | float = 1.0e-10,
    principal_sqrt_backend: TFPrincipalSqrtBackend = "compiled_custom_op",
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Return batched value, analytic score, and diagnostics.

    ``theta`` is the differentiable parameter-proposal tensor with shape
    ``[batch, parameter]``.  ``model`` and ``derivatives`` must already be the
    batch-native state-space objects built for the same proposals.  The forward
    value and score come from BayesFilter's analytic SVD sigma-point recursion;
    the custom gradient wires that analytic score to ``theta`` so HMC callers do
    not use autodiff through the filtering recursion.
    """

    if backend != "tf_principal_sqrt_ukf":
        raise ValueError(
            "production SVD sigma-point custom-gradient authority currently "
            "requires backend='tf_principal_sqrt_ukf'"
        )
    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
        observations,
        model,
        derivatives,
        backend=backend,
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        spectral_gap_tolerance=spectral_gap_tolerance,
        fixed_null_tolerance=fixed_null_tolerance,
        principal_sqrt_reconstruction_tolerance=(
            principal_sqrt_reconstruction_tolerance),
        principal_sqrt_backend=principal_sqrt_backend,
        jitter=jitter,
        allow_fixed_null_support=False,
    )
    value_tensor = tf.stop_gradient(tf.convert_to_tensor(value, dtype=tf.float64))
    score_tensor = tf.stop_gradient(tf.convert_to_tensor(score, dtype=tf.float64))
    _validate_static_value_score_shapes(theta_tensor, value_tensor, score_tensor)

    @tf.custom_gradient
    def value_with_analytic_score(
        theta_live: tf.Tensor,
    ) -> tuple[tf.Tensor, object]:
        del theta_live

        def grad(upstream: tf.Tensor) -> tf.Tensor:
            return _broadcast_upstream_to_score(upstream, score_tensor)

        return value_tensor, grad

    custom_value = value_with_analytic_score(theta_tensor)
    production_diagnostics = dict(diagnostics)
    production_diagnostics.update(
        {
            "backend_status": tf.constant("historical_reference_only"),
            "default_backend": tf.constant("direct_factor_srukf"),
            "default_backend_contract": tf.constant("TFFactorSRUKFModel"),
            "score_authority": tf.constant(
                "bayesfilter_batched_svd_sigma_point_custom_gradient_certified"
            ),
            "filter_owner": tf.constant("BayesFilter"),
            "filter_gradient_authority": tf.constant(
                "filter_owned_custom_gradient_certified"
            ),
            "filter_gradient_api": tf.constant(_PUBLIC_API_NAME),
            "filter_gradient_api_export_status": tf.constant(
                "public_bayesfilter_export_checked"
            ),
            "filter_gradient_api_runtime_selected": tf.constant(True),
            "filter_autodiff_allowed_for_hmc": tf.constant(False),
            "full_filter_score_authority": tf.constant(True),
            "full_state_fallback_allowed": tf.constant(False),
            "time_recursion": tf.constant("tf.while_loop"),
            "placement_derivative_policy": tf.constant(
                "strict_spd_principal_sqrt_sylvester"
            ),
            "strict_spd_branch_required": tf.constant(True),
            "rank_deficient_branch_allowed": tf.constant(False),
            "fixed_null_support_allowed": tf.constant(False),
            "batching_contract": tf.constant(
                "batch_over_parameter_proposals_and_row_aligned_observations"
                if tf.convert_to_tensor(observations).shape.rank == 3
                else "batch_over_parameter_proposals"
            ),
            "numpy_runtime_allowed": tf.constant(False),
            "python_loop_over_time_allowed": tf.constant(False),
            "python_loop_over_batch_allowed": tf.constant(False),
            "principal_sqrt_backend": tf.constant(principal_sqrt_backend),
        }
    )
    return custom_value, score_tensor, production_diagnostics


def _broadcast_upstream_to_score(upstream: tf.Tensor, score: tf.Tensor) -> tf.Tensor:
    upstream_tensor = tf.cast(tf.convert_to_tensor(upstream), score.dtype)
    rank = upstream_tensor.shape.rank
    if rank == 0:
        return upstream_tensor * score
    if rank == 1:
        return upstream_tensor[:, tf.newaxis] * score
    if rank == 2:
        return upstream_tensor * score
    rank_delta = tf.rank(score) - tf.rank(upstream_tensor)
    reshaped = tf.reshape(
        upstream_tensor,
        tf.concat(
            [tf.shape(upstream_tensor), tf.ones([rank_delta], dtype=tf.int32)],
            axis=0,
        ),
    )
    return reshaped * score


def _validate_static_value_score_shapes(
    theta: tf.Tensor,
    value: tf.Tensor,
    score: tf.Tensor,
) -> None:
    if theta.shape.rank != 2:
        raise ValueError("theta must have static rank 2 [batch, parameter]")
    if value.shape.rank != 1:
        raise ValueError("SVD sigma-point value must have static rank 1 [batch]")
    if score.shape.rank != 2:
        raise ValueError(
            "SVD sigma-point score must have static rank 2 [batch, parameter]"
        )
    _check_static_dim(theta, value, axis=0, name="value batch")
    _check_static_dim(theta, score, axis=0, name="score batch")
    _check_static_dim(theta, score, axis=1, name="score parameter")


def _check_static_dim(
    expected: tf.Tensor,
    actual: tf.Tensor,
    *,
    axis: int,
    name: str,
) -> None:
    expected_dim = expected.shape[axis]
    actual_dim = actual.shape[axis]
    if expected_dim is None or actual_dim is None:
        return
    if int(expected_dim) != int(actual_dim):
        raise ValueError(
            f"{name} dimension mismatch: expected {int(expected_dim)}, "
            f"got {int(actual_dim)}"
        )

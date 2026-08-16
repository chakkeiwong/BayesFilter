"""Non-admitted legacy covariance-to-factor compatibility boundary."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
)


def covariance_model_to_factor_contract(model: Any, derivatives: Any) -> tuple[TFFactorSRUKFModel, TFFactorSRUKFDerivatives]:
    """Convert legacy covariance tensors once before tracing the factor route."""

    def lower(value: Any, name: str) -> tf.Tensor:
        tensor = tf.convert_to_tensor(value, dtype=tf.float64, name=name)
        if tensor.shape.rank != 3:
            raise ValueError(f"{name} must have shape [B,D,D]")
        tf.debugging.assert_all_finite(tensor, f"{name} contains NaN or Inf")
        return tf.linalg.cholesky(0.5 * (tensor + tf.linalg.matrix_transpose(tensor)))

    def lower_derivative(d_covariance: Any, factor: tf.Tensor, name: str) -> tf.Tensor:
        dc = tf.convert_to_tensor(d_covariance, dtype=tf.float64, name=name)
        if dc.shape.rank != 4:
            raise ValueError(f"{name} must have shape [B,P,D,D]")
        b, p, d, d2 = dc.shape.as_list()
        if None in (b, p, d, d2) or factor.shape.as_list() != [b, d, d] or d != d2:
            raise ValueError(f"{name} shape is incompatible with its factor")
        rows = []
        for i in range(p):
            rhs = 0.5 * (dc[:, i] + tf.linalg.matrix_transpose(dc[:, i]))
            left = tf.linalg.triangular_solve(factor, rhs, lower=True)
            transformed = tf.linalg.triangular_solve(
                factor,
                tf.linalg.matrix_transpose(left),
                lower=True,
            )
            lower_part = tf.linalg.band_part(transformed, -1, 0)
            lower_part -= 0.5 * tf.linalg.diag(tf.linalg.diag_part(lower_part))
            rows.append(tf.einsum("bij,bjk->bik", factor, lower_part))
        return tf.stack(rows, axis=1)

    initial_factor = lower(model.initial_covariance, "initial_covariance")
    process_factor = lower(model.innovation_covariance, "innovation_covariance")
    observation_factor = lower(model.observation_covariance, "observation_covariance")
    factor_model = TFFactorSRUKFModel(
        initial_mean=model.initial_mean,
        initial_factor=initial_factor,
        process_factor=process_factor,
        observation_factor=observation_factor,
        transition_fn=model.transition_fn,
        observation_fn=model.observation_fn,
        name=f"{getattr(model, 'name', 'legacy')}_factor_compatibility",
    )
    factor_derivatives = TFFactorSRUKFDerivatives(
        d_initial_mean=derivatives.d_initial_mean,
        d_initial_factor=lower_derivative(derivatives.d_initial_covariance, initial_factor, "d_initial_covariance"),
        d_process_factor=lower_derivative(derivatives.d_innovation_covariance, process_factor, "d_innovation_covariance"),
        d_observation_factor=lower_derivative(derivatives.d_observation_covariance, observation_factor, "d_observation_covariance"),
        transition_state_jacobian_fn=derivatives.transition_state_jacobian_fn,
        transition_process_jacobian_fn=derivatives.transition_innovation_jacobian_fn,
        d_transition_fn=derivatives.d_transition_fn,
        observation_state_jacobian_fn=derivatives.observation_state_jacobian_fn,
        d_observation_fn=derivatives.d_observation_fn,
        name=f"{getattr(derivatives, 'name', 'legacy')}_factor_compatibility",
    )
    return factor_model, factor_derivatives


__all__ = ["covariance_model_to_factor_contract"]

"""Batch-native SSL-LSTM complexity target for NeuTra training.

The leading tensor axis indexes independent four-coordinate parameter
proposals. Time remains sequential inside the shared batched principal-square-
root UKF recursion. This module intentionally contains no row-mapping fallback.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.batched_svd_sigma_point_tf import (
    tf_batched_svd_sigma_point_value_and_score_custom_gradient,
)
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
    FREE_NAMES,
    PRIOR_CENTER,
    complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    ssl_lstm_parameter_slices,
    unpack_ssl_lstm_parameters,
)


_EVIDENCE_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-direct-batch-native-gpu-xla-training-plan-2026-07-30.md"
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class BatchNativeSSLLSTMComplexityPosteriorTarget:
    """Rank-2 TensorFlow value/score target for one complexity rung."""

    evaluation_policy = "batch_native_tensorflow_status_no_row_mapping_v2"

    def __init__(
        self,
        q: int,
        *,
        jit_compile: bool = True,
        principal_sqrt_backend: str = "compiled_custom_op",
    ) -> None:
        scalar = complexity_posterior_target(int(q), jit_compile=False)
        self.config = scalar.config
        self.q = int(q)
        self._jit_compile = bool(jit_compile)
        self._principal_sqrt_backend = str(principal_sqrt_backend)
        self._target_signature = scalar.target_signature()
        self._adapter_signature = _stable_hash(
            {
                "target_signature": self._target_signature,
                "route": self.evaluation_policy,
                "filter": "tf_principal_sqrt_ukf",
                "score": "analytic_selected_four_coordinate_score",
                "principal_sqrt_backend": self._principal_sqrt_backend,
            }
        )
        self._compiled_batches: dict[int, Any] = {}
        self.supports_retained_flat_batch = True
        self.supports_retained_value_score_status = True
        self._fixed = unpack_ssl_lstm_parameters(
            self.config.fixture,
            self.config.static_config,
            derivative_parameter_indices=self.config.free_indices,
        )

    @property
    def parameter_dim(self) -> int:
        return 4

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return FREE_NAMES

    @property
    def target_scope(self) -> str:
        return f"ssl_lstm_neutra_state_complexity_batch_native:q{self.q}"

    def target_signature(self) -> str:
        return self._target_signature

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=bool(self._jit_compile),
            runtime_backend=(
                "bayesfilter.nonlinear."
                "ssl_lstm_complexity_batched_target_tf"
            ),
            target_scope=self.target_scope,
            evidence_path=_EVIDENCE_PATH,
            nonclaims=(
                "controlled synthetic target only",
                "training target status does not establish HMC readiness",
                "no posterior oracle or model adequacy claim",
            ),
        )

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.ssl_lstm.complexity_batch_native_target.v2",
            "target_signature": self._target_signature,
            "adapter_signature": self._adapter_signature,
            "q": self.q,
            "evaluation_policy": self.evaluation_policy,
            "filter_backend": "tf_principal_sqrt_ukf",
            "principal_sqrt_backend": self._principal_sqrt_backend,
            "batch_native": True,
            "python_loop_over_batch": False,
            "tf_map_fn_over_batch": False,
            "numpy_runtime_allowed": False,
            "evidence_path": _EVIDENCE_PATH,
        }

    def batch_value_and_score(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(free, tf.float64)
        if values.shape.rank != 2 or values.shape[-1] != 4:
            raise ValueError("batch-native target requires shape [batch,4]")
        batch_size = values.shape[0]
        if batch_size is None:
            raise ValueError("batch-native target requires a static batch size")
        size = int(batch_size)
        compiled = self._compiled_batches.get(size)
        if compiled is None:
            compiled = tf.function(
                self._batch_value_score_impl,
                input_signature=(tf.TensorSpec([size, 4], tf.float64),),
                jit_compile=self._jit_compile,
                reduce_retracing=False,
            )
            self._compiled_batches[size] = compiled
        return compiled(values)

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        return _batch_native_neutra_value_score_status(self, theta)

    def _batch_value_score_impl(
        self, free: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, status = _batch_native_neutra_value_score_status(self, free)
        valid = tf.convert_to_tensor(
            status["valid_pre_regularized_score"], tf.bool
        )
        invalid_value = tf.fill(tf.shape(value), tf.constant(float("nan"), tf.float64))
        invalid_score = tf.fill(tf.shape(score), tf.constant(float("nan"), tf.float64))
        return tf.where(valid, value, invalid_value), tf.where(
            valid[:, tf.newaxis], score, invalid_score
        )

    def _batched_components(
        self, free: tf.Tensor
    ) -> tuple[TFBatchedStructuralStateSpace, TFBatchedStructuralFirstDerivatives]:
        batch_size = int(free.shape[0])
        static = self.config.static_config
        k = int(static.latent_dim)
        h = int(static.hidden_dim)
        d = int(static.observation_dim)
        n = int(static.augmented_state_dim)
        parameter_dim = 4
        fixed = self._fixed
        slices = ssl_lstm_parameter_slices(static)

        latent_weight_mask = tf.one_hot(0, k, dtype=tf.float64)[:, tf.newaxis] * tf.one_hot(
            0, h, dtype=tf.float64
        )[tf.newaxis, :]
        latent_bias_mask = tf.one_hot(0, k, dtype=tf.float64)
        observation_weight_mask = tf.one_hot(0, d, dtype=tf.float64)[:, tf.newaxis] * tf.one_hot(
            0, k, dtype=tf.float64
        )[tf.newaxis, :]
        observation_bias_mask = tf.one_hot(0, d, dtype=tf.float64)

        latent_weight = (
            fixed.latent_weight[tf.newaxis, :, :]
            + (free[:, 0] - PRIOR_CENTER[0])[:, tf.newaxis, tf.newaxis]
            * latent_weight_mask[tf.newaxis, :, :]
        )
        latent_bias = (
            fixed.latent_bias[tf.newaxis, :]
            + (free[:, 1] - PRIOR_CENTER[1])[:, tf.newaxis]
            * latent_bias_mask[tf.newaxis, :]
        )
        observation_weight = (
            fixed.observation_weight[tf.newaxis, :, :]
            + (free[:, 2] - PRIOR_CENTER[2])[:, tf.newaxis, tf.newaxis]
            * observation_weight_mask[tf.newaxis, :, :]
        )
        observation_bias = (
            fixed.observation_bias[tf.newaxis, :]
            + (free[:, 3] - PRIOR_CENTER[3])[:, tf.newaxis]
            * observation_bias_mask[tf.newaxis, :]
        )

        def gates(previous: tf.Tensor) -> tuple[tf.Tensor, ...]:
            z_previous = previous[:, :, :k]
            hidden_previous = previous[:, :, k : k + h]
            cell_previous = previous[:, :, k + h :]
            preactivation = (
                tf.einsum("ghk,brk->brgh", fixed.lstm_input, z_previous)
                + tf.einsum("ghl,brl->brgh", fixed.lstm_recurrent, hidden_previous)
                + fixed.lstm_bias[tf.newaxis, tf.newaxis, :, :]
            )
            input_gate = tf.math.sigmoid(preactivation[:, :, 0, :])
            forget_gate = tf.math.sigmoid(preactivation[:, :, 1, :])
            output_gate = tf.math.sigmoid(preactivation[:, :, 2, :])
            candidate = tf.math.tanh(preactivation[:, :, 3, :])
            cell = forget_gate * cell_previous + input_gate * candidate
            hidden = output_gate * tf.math.tanh(cell)
            return (
                z_previous,
                hidden_previous,
                cell_previous,
                input_gate,
                forget_gate,
                output_gate,
                candidate,
                cell,
                hidden,
            )

        def deterministic_transition(previous: tf.Tensor) -> tf.Tensor:
            *_, cell, hidden = gates(previous)
            latent = (
                tf.einsum("bkh,brh->brk", latent_weight, hidden)
                + latent_bias[:, tf.newaxis, :]
            )
            return tf.concat((latent, hidden, cell), axis=2)

        def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
            deterministic = deterministic_transition(previous)
            return tf.concat(
                (deterministic[:, :, :k] + innovation, deterministic[:, :, k:]),
                axis=2,
            )

        def deterministic_residual(
            previous: tf.Tensor,
            innovation: tf.Tensor,
            next_state: tf.Tensor,
        ) -> tf.Tensor:
            del innovation
            expected = deterministic_transition(previous)
            return next_state[:, :, k:] - expected[:, :, k:]

        def transition_state_jacobian(
            previous: tf.Tensor, innovation: tf.Tensor
        ) -> tf.Tensor:
            del innovation
            (
                _z_previous,
                _hidden_previous,
                cell_previous,
                input_gate,
                forget_gate,
                output_gate,
                candidate,
                cell,
                _hidden,
            ) = gates(previous)
            input_derivative = input_gate * (1.0 - input_gate)
            forget_derivative = forget_gate * (1.0 - forget_gate)
            output_derivative = output_gate * (1.0 - output_gate)
            candidate_derivative = 1.0 - tf.square(candidate)
            input_weights = fixed.lstm_input
            recurrent_weights = fixed.lstm_recurrent
            coeff_input = candidate * input_derivative
            coeff_forget = cell_previous * forget_derivative
            coeff_candidate = input_gate * candidate_derivative
            coeff_output = tf.math.tanh(cell) * output_derivative
            coeff_cell_hidden = output_gate * (1.0 - tf.square(tf.math.tanh(cell)))
            dcell_dlatent = (
                coeff_input[:, :, :, tf.newaxis]
                * input_weights[0][tf.newaxis, tf.newaxis, :, :]
                + coeff_forget[:, :, :, tf.newaxis]
                * input_weights[1][tf.newaxis, tf.newaxis, :, :]
                + coeff_candidate[:, :, :, tf.newaxis]
                * input_weights[3][tf.newaxis, tf.newaxis, :, :]
            )
            dcell_dhidden = (
                coeff_input[:, :, :, tf.newaxis]
                * recurrent_weights[0][tf.newaxis, tf.newaxis, :, :]
                + coeff_forget[:, :, :, tf.newaxis]
                * recurrent_weights[1][tf.newaxis, tf.newaxis, :, :]
                + coeff_candidate[:, :, :, tf.newaxis]
                * recurrent_weights[3][tf.newaxis, tf.newaxis, :, :]
            )
            dcell_dcell = tf.linalg.diag(forget_gate)
            dhidden_dlatent = (
                coeff_output[:, :, :, tf.newaxis]
                * input_weights[2][tf.newaxis, tf.newaxis, :, :]
                + coeff_cell_hidden[:, :, :, tf.newaxis] * dcell_dlatent
            )
            dhidden_dhidden = (
                coeff_output[:, :, :, tf.newaxis]
                * recurrent_weights[2][tf.newaxis, tf.newaxis, :, :]
                + coeff_cell_hidden[:, :, :, tf.newaxis] * dcell_dhidden
            )
            dhidden_dcell = coeff_cell_hidden[:, :, :, tf.newaxis] * dcell_dcell
            dlatent_dlatent = tf.einsum(
                "bkh,brhj->brkj", latent_weight, dhidden_dlatent
            )
            dlatent_dhidden = tf.einsum(
                "bkh,brhj->brkj", latent_weight, dhidden_dhidden
            )
            dlatent_dcell = tf.einsum(
                "bkh,brhj->brkj", latent_weight, dhidden_dcell
            )
            latent_rows = tf.concat(
                (dlatent_dlatent, dlatent_dhidden, dlatent_dcell), axis=3
            )
            hidden_rows = tf.concat(
                (dhidden_dlatent, dhidden_dhidden, dhidden_dcell), axis=3
            )
            cell_rows = tf.concat(
                (dcell_dlatent, dcell_dhidden, dcell_dcell), axis=3
            )
            result = tf.concat((latent_rows, hidden_rows, cell_rows), axis=2)
            return tf.ensure_shape(result, [batch_size, None, n, n])

        def transition_innovation_jacobian(
            previous: tf.Tensor, innovation: tf.Tensor
        ) -> tf.Tensor:
            del innovation
            point_count = tf.shape(previous)[1]
            matrix = tf.concat(
                (tf.eye(k, dtype=tf.float64), tf.zeros([2 * h, k], tf.float64)),
                axis=0,
            )
            return tf.broadcast_to(
                matrix[tf.newaxis, tf.newaxis, :, :],
                [batch_size, point_count, n, k],
            )

        def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
            del innovation
            deterministic = deterministic_transition(previous)
            point_count = tf.shape(previous)[1]
            latent_zero = tf.one_hot(0, n, dtype=tf.float64)
            d_weight = deterministic[:, :, k, tf.newaxis] * latent_zero[
                tf.newaxis, tf.newaxis, :
            ]
            d_bias = tf.broadcast_to(
                latent_zero[tf.newaxis, tf.newaxis, :],
                [batch_size, point_count, n],
            )
            zero = tf.zeros([batch_size, point_count, n], tf.float64)
            return tf.stack((d_weight, d_bias, zero, zero), axis=1)

        def observe(state: tf.Tensor) -> tf.Tensor:
            return (
                tf.einsum("bdk,brk->brd", observation_weight, state[:, :, :k])
                + observation_bias[:, tf.newaxis, :]
            )

        def observation_state_jacobian(state: tf.Tensor) -> tf.Tensor:
            point_count = tf.shape(state)[1]
            matrix = tf.concat(
                (observation_weight, tf.zeros([batch_size, d, 2 * h], tf.float64)),
                axis=2,
            )
            return tf.broadcast_to(
                matrix[:, tf.newaxis, :, :],
                [batch_size, point_count, d, n],
            )

        def d_observation(state: tf.Tensor) -> tf.Tensor:
            point_count = tf.shape(state)[1]
            zero = tf.zeros([batch_size, point_count, d], tf.float64)
            d_weight = state[:, :, 0, tf.newaxis] * tf.one_hot(
                0, d, dtype=tf.float64
            )[tf.newaxis, tf.newaxis, :]
            d_bias = tf.broadcast_to(
                tf.one_hot(0, d, dtype=tf.float64)[tf.newaxis, tf.newaxis, :],
                [batch_size, point_count, d],
            )
            return tf.stack((zero, zero, d_weight, d_bias), axis=1)

        initial_mean = tf.broadcast_to(fixed.initial_mean, [batch_size, n])
        initial_covariance = tf.broadcast_to(
            fixed.initial_covariance, [batch_size, n, n]
        )
        innovation_covariance = tf.broadcast_to(
            fixed.ukf_innovation_covariance, [batch_size, k, k]
        )
        observation_covariance = tf.broadcast_to(
            fixed.observation_covariance, [batch_size, d, d]
        )
        model = TFBatchedStructuralStateSpace(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            innovation_covariance=innovation_covariance,
            observation_covariance=observation_covariance,
            transition_fn=transition,
            observation_fn=observe,
            deterministic_residual_fn=deterministic_residual,
            name=f"ssl_lstm_complexity_q{self.q}_batch_native",
        )
        derivatives = TFBatchedStructuralFirstDerivatives(
            d_initial_mean=tf.zeros(
                [batch_size, parameter_dim, n], dtype=tf.float64
            ),
            d_initial_covariance=tf.zeros(
                [batch_size, parameter_dim, n, n], dtype=tf.float64
            ),
            d_innovation_covariance=tf.zeros(
                [batch_size, parameter_dim, k, k], dtype=tf.float64
            ),
            d_observation_covariance=tf.zeros(
                [batch_size, parameter_dim, d, d], dtype=tf.float64
            ),
            transition_state_jacobian_fn=transition_state_jacobian,
            transition_innovation_jacobian_fn=transition_innovation_jacobian,
            d_transition_fn=d_transition,
            observation_state_jacobian_fn=observation_state_jacobian,
            d_observation_fn=d_observation,
            name="ssl_lstm_complexity_batch_native_selected_four_derivatives",
        )
        if slices.latent_weight_start != self.config.free_indices[0]:
            raise ValueError("free-coordinate latent-weight binding changed")
        return model, derivatives


def _batch_native_neutra_value_score_status(
    target: BatchNativeSSLLSTMComplexityPosteriorTarget,
    free: Any,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    values = tf.convert_to_tensor(free, tf.float64)
    if values.shape.rank != 2 or values.shape[-1] != 4:
        raise ValueError("batch-native target requires shape [batch,4]")
    if values.shape[0] is None:
        raise ValueError("batch-native target requires a static batch size")

    model, derivatives = target._batched_components(values)
    likelihood, likelihood_score, diagnostics = (
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            values,
            target.config.observations,
            model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            placement_floor=tf.constant(0.0, tf.float64),
            innovation_floor=tf.constant(1.0e-12, tf.float64),
            principal_sqrt_backend=target._principal_sqrt_backend,
        )
    )
    delta = values - PRIOR_CENTER[tf.newaxis, :]
    variance = tf.constant(16.0, tf.float64)
    value = likelihood - 0.5 * tf.reduce_sum(tf.square(delta) / variance, axis=1)
    score = likelihood_score - delta / variance

    placement_floors = tf.convert_to_tensor(
        diagnostics["placement_floor_count"], tf.int32
    )
    innovation_floors = tf.convert_to_tensor(
        diagnostics["innovation_floor_count"], tf.int32
    )
    floor_count = placement_floors + innovation_floors
    row_class = tf.convert_to_tensor(
        diagnostics["principal_sqrt_target_row_class_code"], tf.int32
    )
    valid_count = tf.convert_to_tensor(
        diagnostics["principal_sqrt_target_valid_count"], tf.int32
    )
    min_eigenvalue = tf.convert_to_tensor(
        diagnostics["min_innovation_eigenvalue"], tf.float64
    )
    finite = tf.logical_and(
        tf.math.is_finite(value),
        tf.reduce_all(tf.math.is_finite(score), axis=1),
    )
    valid = tf.logical_and(
        finite,
        tf.logical_and(
            tf.equal(floor_count, 0),
            tf.logical_and(
                tf.logical_and(tf.equal(row_class, 0), tf.equal(valid_count, 1)),
                tf.logical_and(tf.math.is_finite(min_eigenvalue), min_eigenvalue > 0.0),
            ),
        ),
    )
    status = {
        "status_code": tf.where(valid, 0, 1),
        "valid_pre_regularized_score": valid,
        "floor_count_value": floor_count,
        "placement_floor_count": placement_floors,
        "innovation_floor_count": innovation_floors,
        "principal_sqrt_target_row_class_code": row_class,
        "principal_sqrt_target_valid_count": valid_count,
        "min_innovation_eigenvalue": min_eigenvalue,
        "min_placement_eigenvalue": tf.convert_to_tensor(
            diagnostics["min_placement_eigenvalue"], tf.float64
        ),
        "min_placement_eigen_gap": tf.convert_to_tensor(
            diagnostics["min_placement_eigen_gap"], tf.float64
        ),
        "min_innovation_eigen_gap": tf.convert_to_tensor(
            diagnostics["min_innovation_eigen_gap"], tf.float64
        ),
        "placement_classified_invalid_count": tf.convert_to_tensor(
            diagnostics["placement_classified_invalid_count"], tf.int32
        ),
        "innovation_classified_invalid_count": tf.convert_to_tensor(
            diagnostics["innovation_classified_invalid_count"], tf.int32
        ),
        "placement_derivative_rhs_nonfinite_count": tf.convert_to_tensor(
            diagnostics["placement_derivative_rhs_nonfinite_count"], tf.int32
        ),
        "innovation_derivative_rhs_nonfinite_count": tf.convert_to_tensor(
            diagnostics["innovation_derivative_rhs_nonfinite_count"], tf.int32
        ),
        "principal_sqrt_target_classified_invalid_count": tf.convert_to_tensor(
            diagnostics["principal_sqrt_target_classified_invalid_count"], tf.int32
        ),
        "principal_sqrt_target_derivative_rhs_nonfinite_count": tf.convert_to_tensor(
            diagnostics["principal_sqrt_target_derivative_rhs_nonfinite_count"],
            tf.int32,
        ),
        "placement_roundoff_repair_count": tf.convert_to_tensor(
            diagnostics["placement_roundoff_repair_count"], tf.int32
        ),
        "innovation_roundoff_repair_count": tf.convert_to_tensor(
            diagnostics["innovation_roundoff_repair_count"], tf.int32
        ),
        "value_finite": tf.math.is_finite(value),
        "score_finite": tf.reduce_all(tf.math.is_finite(score), axis=1),
        "input_finite": tf.reduce_all(tf.math.is_finite(values), axis=1),
    }
    return (
        tf.ensure_shape(value, [values.shape[0]]),
        tf.ensure_shape(score, [values.shape[0], 4]),
        status,
    )


def batch_native_complexity_posterior_target(
    q: int,
    *,
    jit_compile: bool = True,
    principal_sqrt_backend: str = "compiled_custom_op",
) -> BatchNativeSSLLSTMComplexityPosteriorTarget:
    return BatchNativeSSLLSTMComplexityPosteriorTarget(
        q,
        jit_compile=jit_compile,
        principal_sqrt_backend=principal_sqrt_backend,
    )


def batch_native_complexity_target_worker_factory(
    config: Mapping[str, Any],
) -> BatchNativeSSLLSTMComplexityPosteriorTarget:
    if "q" not in config:
        raise ValueError("batch-native worker config requires q")
    return batch_native_complexity_posterior_target(
        int(config["q"]),
        jit_compile=bool(config.get("jit_compile", False)),
        principal_sqrt_backend=str(
            config.get("principal_sqrt_backend", "tensorflow_eigh")
        ),
    )


__all__ = [
    "BatchNativeSSLLSTMComplexityPosteriorTarget",
    "batch_native_complexity_posterior_target",
    "batch_native_complexity_target_worker_factory",
]

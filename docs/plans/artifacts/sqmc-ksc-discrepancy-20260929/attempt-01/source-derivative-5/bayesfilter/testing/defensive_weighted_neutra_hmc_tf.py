"""TensorFlow authorities for analytic HMC behind a frozen weighted NeuTra map.

This module is scoped to the reviewed four-dimensional unequal-weight Gaussian
mixture campaign. It verifies and restores one weighted-training checkpoint,
exposes the exact analytic value/score target, and computes retained-draw
diagnostics. It does not train, tune, or select a transport.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.testing.importance_sampling_tf import (
    gaussian_mixture_log_prob_responsibilities_score,
    sample_gaussian_mixture,
)


TARGET_IDENTITY = "separated_two_mode_unequal_weight_d4_v1"
TARGET_SCOPE = "defensive_weighted_neutra_analytic_hmc:unequal_weight_d4_v1"
STATE_SCHEMA = "bayesfilter.neutra.weighted_forward_kl_state.v1"
HASH_SCHEMA = "bayesfilter.defensive_weighted_neutra_analytic_hashes.v1"


class DefensiveWeightedNeuTraHMCError(RuntimeError):
    """Raised when analytic-HMC input evidence fails closed."""


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytic_two_mode_target(dtype: tf.DType = tf.float64) -> Mapping[str, Any]:
    means = tf.constant(
        ((-4.0, -0.5, 0.75, -0.25), (4.0, 0.5, -0.75, 0.25)), dtype
    )
    left_factor = tf.constant(
        (
            (0.8, 0.0, 0.0, 0.0),
            (0.2, 0.6, 0.0, 0.0),
            (0.0, 0.1, 0.5, 0.0),
            (0.1, 0.0, 0.15, 0.4),
        ),
        dtype,
    )
    right_factor = tf.constant(
        (
            (0.5, 0.0, 0.0, 0.0),
            (-0.1, 0.9, 0.0, 0.0),
            (0.05, -0.2, 0.7, 0.0),
            (0.0, 0.1, -0.1, 0.55),
        ),
        dtype,
    )
    covariances = tf.stack(
        (
            tf.matmul(left_factor, left_factor, transpose_b=True),
            tf.matmul(right_factor, right_factor, transpose_b=True),
        )
    )
    probabilities = tf.constant((0.8, 0.2), dtype)
    true_mean = tf.reduce_sum(probabilities[:, tf.newaxis] * means, axis=0)
    centered = means - true_mean
    true_covariance = tf.reduce_sum(
        probabilities[:, tf.newaxis, tf.newaxis]
        * (
            covariances
            + centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :]
        ),
        axis=0,
    )
    signature_payload = {
        "schema": "bayesfilter.analytic_gaussian_mixture_target.v1",
        "identity": TARGET_IDENTITY,
        "probabilities": probabilities,
        "means": means,
        "covariances": covariances,
        "dtype": dtype.name,
    }
    return {
        "identity": TARGET_IDENTITY,
        "probabilities": probabilities,
        "means": means,
        "covariances": covariances,
        "true_mean": true_mean,
        "true_covariance": true_covariance,
        "target_signature": stable_json_hash(signature_payload),
        "signature_payload": signature_payload,
    }


class AnalyticGaussianMixtureValueScoreAdapter:
    """Graph-native normalized Gaussian-mixture value and exact score."""

    parameter_dim = 4
    supports_retained_draw_batch = False
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True
    target_status_invalid_rows_become_nonfinite = False

    def __init__(self, target: Mapping[str, Any] | None = None) -> None:
        self.target = dict(analytic_two_mode_target() if target is None else target)
        if self.target.get("identity") != TARGET_IDENTITY:
            raise DefensiveWeightedNeuTraHMCError("analytic target identity mismatch")
        self.target_scope = TARGET_SCOPE

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        rows = tf.convert_to_tensor(theta, tf.float64)
        if rows.shape.rank != 2 or rows.shape[-1] != self.parameter_dim:
            raise ValueError("analytic mixture target requires [row, 4]")
        value, _responsibilities, score = (
            gaussian_mixture_log_prob_responsibilities_score(
                rows,
                self.target["probabilities"],
                self.target["means"],
                self.target["covariances"],
            )
        )
        return value, score

    def log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        value, score = self.log_prob_and_grad(theta)
        finite = tf.logical_and(
            tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=-1)
        )
        zeros = tf.zeros_like(value, tf.int32)
        ones = tf.ones_like(value, tf.float64)
        return value, score, {
            "status_code": tf.where(finite, zeros, tf.ones_like(zeros)),
            "valid_pre_regularized_score": finite,
            "floor_count_value": zeros,
            "min_innovation_eigenvalue": ones,
            "innovation_condition_estimate": ones,
        }

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        return self.log_prob_and_grad_status(theta)[2]

    def adapter_signature(self) -> str:
        return stable_json_hash(
            {
                "schema": "bayesfilter.analytic_gaussian_mixture_value_score.v1",
                "target_signature": self.target["target_signature"],
                "target_scope": self.target_scope,
                "value_score_authority": "graph_native_exact_mixture_score",
            }
        )

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_exact_full_covariance_gaussian_mixture",
            evidence_path=(
                "docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-plan-2026-08-12.md"
            ),
            target_scope=self.target_scope,
            nonclaims=(
                "analytic four-dimensional target only",
                "no transport or HMC validity claim",
            ),
        )


@dataclass(frozen=True)
class LoadedWeightedNeuTraTransport:
    transport: WeightedDenseIAFTransport
    checkpoint_path: str
    checkpoint_sha256: str
    state_hash: str
    transport_tensor_hash: str
    selected_step: int
    config: WeightedNeuTraConfig

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.loaded_weighted_transport.v1",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_state_hash": self.state_hash,
            "transport_tensor_hash": self.transport_tensor_hash,
            "selected_step": self.selected_step,
            "config": self.config.manifest_payload(),
            "transport_manifest": self.transport.manifest_payload(),
        }


def load_weighted_neutra_transport(
    checkpoint_path: str | Path,
) -> LoadedWeightedNeuTraTransport:
    path = Path(checkpoint_path).resolve()
    if path.name != "trainer_states.json" or not path.is_file():
        raise DefensiveWeightedNeuTraHMCError("weighted checkpoint path is invalid")
    hashes_path = path.with_name("artifact_hashes.json")
    hashes = _read_json(hashes_path)
    if hashes.get("schema") != HASH_SCHEMA:
        raise DefensiveWeightedNeuTraHMCError("checkpoint artifact hash schema mismatch")
    expected_file_hash = hashes.get("artifacts", {}).get(path.name)
    actual_file_hash = sha256_file(path)
    if expected_file_hash != actual_file_hash:
        raise DefensiveWeightedNeuTraHMCError("checkpoint SHA-256 mismatch")
    payload = _read_json(path)
    state = payload.get("weighted")
    if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
        raise DefensiveWeightedNeuTraHMCError("weighted state schema mismatch")
    hash_payload = {key: value for key, value in state.items() if key != "state_hash"}
    state_hash = stable_json_hash(hash_payload)
    if state.get("state_hash") != state_hash:
        raise DefensiveWeightedNeuTraHMCError("weighted state semantic hash mismatch")
    config_payload = state.get("config")
    if not isinstance(config_payload, Mapping):
        raise DefensiveWeightedNeuTraHMCError("weighted state config is missing")
    if config_payload.get("schema") != "bayesfilter.neutra.weighted_forward_kl_config.v1":
        raise DefensiveWeightedNeuTraHMCError("weighted state config schema mismatch")
    config_values = {key: value for key, value in config_payload.items() if key != "schema"}
    config_values["hidden_layers"] = tuple(config_values["hidden_layers"])
    config_values["initialization_seed"] = tuple(config_values["initialization_seed"])
    config = WeightedNeuTraConfig(**config_values)
    if (
        config.dimension != 4
        or config.hidden_layers != (128, 128)
        or config.stages != 6
        or config.activation != "tanh"
        or not config.jit_compile
    ):
        raise DefensiveWeightedNeuTraHMCError("weighted HMC checkpoint protocol mismatch")
    transport = WeightedDenseIAFTransport(config)
    values = state.get("variables")
    if not isinstance(values, Sequence) or len(values) != len(transport.trainable_variables):
        raise DefensiveWeightedNeuTraHMCError("weighted transport variable count mismatch")
    for index, (variable, value) in enumerate(zip(transport.trainable_variables, values)):
        tensor = tf.convert_to_tensor(value, tf.float64)
        if tensor.shape != variable.shape:
            raise DefensiveWeightedNeuTraHMCError(
                f"weighted transport variable {index} shape mismatch"
            )
        tf.debugging.assert_all_finite(tensor, f"weighted transport variable {index}")
        variable.assign(tensor)
    restored = [variable.read_value() for variable in transport.trainable_variables]
    tensor_hash = stable_json_hash(restored)
    identity = {
        "checkpoint_sha256": actual_file_hash,
        "training_state_hash": state_hash,
        "transport_tensor_hash": tensor_hash,
    }
    transport.bind_frozen_identity(identity)
    return LoadedWeightedNeuTraTransport(
        transport=transport,
        checkpoint_path=path.as_posix(),
        checkpoint_sha256=actual_file_hash,
        state_hash=state_hash,
        transport_tensor_hash=tensor_hash,
        selected_step=int(state.get("step", -1)),
        config=config,
    )


def mode_aware_initial_state(
    transport: WeightedDenseIAFTransport,
    target: Mapping[str, Any] | None = None,
) -> tf.Tensor:
    truth = analytic_two_mode_target() if target is None else target
    offsets = tf.constant(
        (
            (-0.08, 0.04, -0.03, 0.02),
            (0.08, -0.04, 0.03, -0.02),
            (-0.05, -0.03, 0.04, 0.02),
            (0.05, 0.03, -0.04, -0.02),
        ),
        tf.float64,
    )
    physical = tf.gather(truth["means"], tf.constant((0, 0, 1, 1))) + offsets
    latent, _forward_logdet = transport.inverse_and_forward_logdet(physical)
    tf.debugging.assert_all_finite(latent, "mode-aware latent initial state")
    return latent


def retained_analytic_diagnostics(
    physical_samples: Any,
    *,
    reference_seed: tuple[int, int] = (20260812, 88001),
    confidence_z: float = 2.5758293035489004,
) -> Mapping[str, Any]:
    samples = tf.convert_to_tensor(physical_samples, tf.float64)
    if samples.shape.rank != 3 or samples.shape[-1] != 4:
        raise ValueError("retained samples must have [draw, chain, 4]")
    draws = int(samples.shape[0])
    chains = int(samples.shape[1])
    if draws < 4 or chains < 2:
        raise ValueError("retained diagnostics require at least four draws and two chains")
    target = analytic_two_mode_target()
    flat = tf.reshape(samples, (-1, 4))
    _value, responsibilities, _score = (
        gaussian_mixture_log_prob_responsibilities_score(
            flat,
            target["probabilities"],
            target["means"],
            target["covariances"],
        )
    )
    minority = tf.reshape(responsibilities[:, 1], (draws, chains))
    minority_mean = tf.reduce_mean(minority)
    minority_mcse = _batch_means_mcse(minority)
    z_value = tf.constant(float(confidence_z), tf.float64)
    minority_interval = tf.stack(
        (minority_mean - z_value * minority_mcse, minority_mean + z_value * minority_mcse)
    )
    sample_mean = tf.reduce_mean(samples, axis=(0, 1))
    mean_mcse = tf.stack(
        [_batch_means_mcse(samples[:, :, index]) for index in range(4)]
    )
    mean_lower = sample_mean - z_value * mean_mcse
    mean_upper = sample_mean + z_value * mean_mcse
    centered = flat - sample_mean
    sample_covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(flat)[0] - 1, tf.float64
    )
    del reference_seed
    centered_truth = samples - target["true_mean"]
    covariance_moment = tf.einsum("dci,dcj->dcij", centered_truth, centered_truth)
    covariance_estimate = tf.reduce_mean(covariance_moment, axis=(0, 1))
    covariance_mcse = tf.stack(
        [
            tf.stack(
                [_batch_means_mcse(covariance_moment[:, :, i, j]) for j in range(4)]
            )
            for i in range(4)
        ]
    )
    covariance_lower = covariance_estimate - z_value * covariance_mcse
    covariance_upper = covariance_estimate + z_value * covariance_mcse
    hard_assignment = tf.argmax(responsibilities, axis=1, output_type=tf.int32)
    hard_assignment = tf.reshape(hard_assignment, (draws, chains))
    per_chain_minority = tf.reduce_mean(minority, axis=0)
    per_chain_hard_minority = tf.reduce_mean(
        tf.cast(hard_assignment == 1, tf.float64), axis=0
    )
    per_chain_hard_both = tf.stack(
        [
            tf.logical_and(
                tf.reduce_any(hard_assignment[:, chain] == 0),
                tf.reduce_any(hard_assignment[:, chain] == 1),
            )
            for chain in range(chains)
        ]
    )
    target_mass = tf.constant(0.2, tf.float64)
    primary_gates = {
        "all_finite": bool(tf.reduce_all(tf.math.is_finite(flat)).numpy()),
        "minority_mass_99pct_interval_contains_truth": bool(
            tf.logical_and(
                minority_interval[0] <= target_mass, target_mass <= minority_interval[1]
            ).numpy()
        ),
        "both_modes_observed_overall": bool(
            tf.reduce_all(tf.math.bincount(tf.reshape(hard_assignment, (-1,)), minlength=2) > 0).numpy()
        ),
        "both_hard_modes_observed_per_chain": bool(tf.reduce_all(per_chain_hard_both).numpy()),
    }
    mean_interval_contains_truth = tf.logical_and(
        mean_lower <= target["true_mean"], target["true_mean"] <= mean_upper
    )
    covariance_interval_contains_truth = tf.logical_and(
        covariance_lower <= target["true_covariance"],
        target["true_covariance"] <= covariance_upper,
    )
    moment_diagnostics = {
        "mean_99pct_interval_contains_truth_by_coordinate": mean_interval_contains_truth,
        "mean_interval_pass_count": int(
            tf.reduce_sum(tf.cast(mean_interval_contains_truth, tf.int32)).numpy()
        ),
        "mean_interval_total_count": 4,
        "covariance_99pct_interval_contains_truth_by_entry": covariance_interval_contains_truth,
        "covariance_interval_pass_count": int(
            tf.reduce_sum(tf.cast(covariance_interval_contains_truth, tf.int32)).numpy()
        ),
        "covariance_interval_total_count": 16,
        "role": "marginal_explanatory_diagnostics_not_joint_vetoes",
    }
    return {
        "schema": "bayesfilter.defensive_weighted_neutra_retained_analytic_diagnostics.v1",
        "sample_shape": tuple(int(value) for value in samples.shape),
        "target_identity": TARGET_IDENTITY,
        "confidence_level": 0.99,
        "minority_mass": float(minority_mean.numpy()),
        "minority_mass_batch_means_mcse": float(minority_mcse.numpy()),
        "minority_mass_interval": minority_interval,
        "analytic_minority_mass": 0.2,
        "sample_mean": sample_mean,
        "analytic_mean": target["true_mean"],
        "mean_batch_means_mcse": mean_mcse,
        "mean_interval_lower": mean_lower,
        "mean_interval_upper": mean_upper,
        "sample_covariance": sample_covariance,
        "analytic_covariance": target["true_covariance"],
        "covariance_absolute_error": tf.abs(sample_covariance - target["true_covariance"]),
        "covariance_moment_estimate": covariance_estimate,
        "covariance_moment_batch_means_mcse": covariance_mcse,
        "covariance_moment_interval_lower": covariance_lower,
        "covariance_moment_interval_upper": covariance_upper,
        "per_chain_soft_minority_mass": per_chain_minority,
        "per_chain_hard_minority_mass": per_chain_hard_minority,
        "per_chain_both_hard_modes_observed": per_chain_hard_both,
        "gates": primary_gates,
        "moment_diagnostics": moment_diagnostics,
        "passed_primary_screens": all(primary_gates.values()),
        "joint_moment_test_performed": False,
        "multiple_testing_note": (
            "Mean/covariance intervals are marginal diagnostics and are not combined "
            "into a joint rejection or p-value."
        ),
    }


def _batch_means_mcse(values: tf.Tensor) -> tf.Tensor:
    rows = tf.convert_to_tensor(values, tf.float64)
    if rows.shape.rank != 2:
        raise ValueError("batch-means input must have [draw, chain]")
    draws = int(rows.shape[0])
    chains = int(rows.shape[1])
    batch_length = max(2, int(math.sqrt(draws)))
    batch_count = draws // batch_length
    if batch_count < 2:
        return tf.math.reduce_std(tf.reshape(rows, (-1,))) / tf.sqrt(
            tf.cast(draws * chains, tf.float64)
        )
    trimmed = rows[: batch_count * batch_length]
    batched = tf.reshape(trimmed, (batch_count, batch_length, chains))
    batch_means = tf.reduce_mean(batched, axis=1)
    variance = tf.math.reduce_variance(tf.reshape(batch_means, (-1,)))
    return tf.sqrt(variance / tf.cast(batch_count * chains, tf.float64))


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DefensiveWeightedNeuTraHMCError(f"unreadable JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise DefensiveWeightedNeuTraHMCError(f"JSON object required: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if tf.is_tensor(value):
        return _json_ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "AnalyticGaussianMixtureValueScoreAdapter",
    "DefensiveWeightedNeuTraHMCError",
    "LoadedWeightedNeuTraTransport",
    "TARGET_IDENTITY",
    "TARGET_SCOPE",
    "analytic_two_mode_target",
    "load_weighted_neutra_transport",
    "mode_aware_initial_state",
    "retained_analytic_diagnostics",
    "sha256_file",
    "stable_json_hash",
]

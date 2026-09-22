"""Exact Gaussian-mixture HMC authorities for frozen weighted NeuTra maps.

This module loads a verified weighted-forward-KL checkpoint, evaluates a
normalized Gaussian-mixture target and its explicit score, and supplies generic
component-aware initial states. It neither trains nor selects a transport.
"""

from __future__ import annotations

import hashlib
import json
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
    validate_gaussian_mixture,
)


STATE_SCHEMA = "bayesfilter.neutra.weighted_forward_kl_state.v1"
HASH_SCHEMA = "bayesfilter.defensive_weighted_neutra_analytic_hashes.v1"


class WeightedNeuTraGaussianMixtureHMCError(RuntimeError):
    """Raised when a frozen transport or exact target fails validation."""


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytic_three_mode_target(dtype: tf.DType = tf.float64) -> Mapping[str, Any]:
    """Return the reviewed separated triangular three-component target."""

    probabilities = tf.constant((0.5, 0.3, 0.2), dtype)
    means = tf.constant(
        (
            (-4.5, -1.0, 0.8, -0.4),
            (4.0, -1.8, -0.7, 0.5),
            (0.5, 4.8, 0.2, -0.6),
        ),
        dtype,
    )
    factors = tf.constant(
        (
            (
                (0.75, 0.0, 0.0, 0.0),
                (0.18, 0.55, 0.0, 0.0),
                (0.05, 0.10, 0.45, 0.0),
                (0.08, -0.03, 0.12, 0.38),
            ),
            (
                (0.48, 0.0, 0.0, 0.0),
                (-0.16, 0.88, 0.0, 0.0),
                (0.08, -0.22, 0.68, 0.0),
                (0.02, 0.14, -0.09, 0.52),
            ),
            (
                (0.62, 0.0, 0.0, 0.0),
                (0.28, 0.58, 0.0, 0.0),
                (-0.12, 0.16, 0.82, 0.0),
                (0.10, 0.04, 0.20, 0.44),
            ),
        ),
        dtype,
    )
    covariances = tf.matmul(factors, factors, transpose_b=True)
    _probabilities, means, covariances, _ = validate_gaussian_mixture(
        probabilities, means, covariances
    )
    signature_payload = {
        "schema": "bayesfilter.analytic_gaussian_mixture_target.v1",
        "identity": "separated_three_mode_unequal_weight_d4_v1",
        "probabilities": probabilities,
        "means": means,
        "covariances": covariances,
        "dtype": dtype.name,
    }
    return {
        "identity": "separated_three_mode_unequal_weight_d4_v1",
        "probabilities": probabilities,
        "means": means,
        "covariances": covariances,
        "target_signature": stable_json_hash(signature_payload),
        "signature_payload": signature_payload,
    }


class AnalyticGaussianMixtureValueScoreAdapter:
    """Graph-native exact Gaussian-mixture value/score adapter."""

    supports_retained_draw_batch = False
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True
    target_status_invalid_rows_become_nonfinite = False

    def __init__(self, target: Mapping[str, Any]) -> None:
        required = {
            "identity",
            "probabilities",
            "means",
            "covariances",
            "target_signature",
        }
        missing = sorted(required - set(target))
        if missing:
            raise ValueError("target missing: " + ", ".join(missing))
        probabilities, means, covariances, _ = validate_gaussian_mixture(
            target["probabilities"], target["means"], target["covariances"]
        )
        self.target = {
            **dict(target),
            "probabilities": probabilities,
            "means": means,
            "covariances": covariances,
        }
        self.parameter_dim = int(means.shape[1])
        self.target_scope = f"weighted_neutra_analytic_mixture:{self.target['identity']}"

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        rows = tf.convert_to_tensor(theta, tf.float64)
        if rows.shape.rank != 2 or rows.shape[-1] != self.parameter_dim:
            raise ValueError(
                f"analytic mixture target requires [row, {self.parameter_dim}]"
            )
        value, _responsibilities, score = gaussian_mixture_log_prob_responsibilities_score(
            rows,
            self.target["probabilities"],
            self.target["means"],
            self.target["covariances"],
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
                "docs/plans/"
                "bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md"
            ),
            target_scope=self.target_scope,
            nonclaims=(
                "analytic target only",
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
    *,
    required_dimension: int,
) -> LoadedWeightedNeuTraTransport:
    """Restore the weighted arm and bind its immutable checkpoint identity."""

    if isinstance(required_dimension, bool) or int(required_dimension) <= 0:
        raise ValueError("required_dimension must be positive")
    path = Path(checkpoint_path).resolve()
    if path.name != "trainer_states.json" or not path.is_file():
        raise WeightedNeuTraGaussianMixtureHMCError("weighted checkpoint path is invalid")
    hashes = _read_json(path.with_name("artifact_hashes.json"))
    if hashes.get("schema") != HASH_SCHEMA:
        raise WeightedNeuTraGaussianMixtureHMCError("checkpoint hash schema mismatch")
    actual_file_hash = sha256_file(path)
    if hashes.get("artifacts", {}).get(path.name) != actual_file_hash:
        raise WeightedNeuTraGaussianMixtureHMCError("checkpoint SHA-256 mismatch")
    payload = _read_json(path)
    state = payload.get("weighted")
    if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
        raise WeightedNeuTraGaussianMixtureHMCError("weighted state schema mismatch")
    hash_payload = {key: value for key, value in state.items() if key != "state_hash"}
    state_hash = stable_json_hash(hash_payload)
    if state.get("state_hash") != state_hash:
        raise WeightedNeuTraGaussianMixtureHMCError("weighted state semantic hash mismatch")
    config_payload = state.get("config")
    if not isinstance(config_payload, Mapping):
        raise WeightedNeuTraGaussianMixtureHMCError("weighted state config is missing")
    values = {key: value for key, value in config_payload.items() if key != "schema"}
    try:
        values["hidden_layers"] = tuple(values["hidden_layers"])
        values["initialization_seed"] = tuple(values["initialization_seed"])
        config = WeightedNeuTraConfig(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise WeightedNeuTraGaussianMixtureHMCError(
            "weighted state config is invalid"
        ) from error
    if config.dimension != int(required_dimension) or not config.jit_compile:
        raise WeightedNeuTraGaussianMixtureHMCError(
            "weighted transport dimension or XLA protocol mismatch"
        )
    transport = WeightedDenseIAFTransport(config)
    saved_variables = state.get("variables")
    if not isinstance(saved_variables, Sequence) or len(saved_variables) != len(
        transport.trainable_variables
    ):
        raise WeightedNeuTraGaussianMixtureHMCError(
            "weighted transport variable count mismatch"
        )
    for index, (variable, value) in enumerate(
        zip(transport.trainable_variables, saved_variables)
    ):
        tensor = tf.convert_to_tensor(value, tf.float64)
        if tensor.shape != variable.shape:
            raise WeightedNeuTraGaussianMixtureHMCError(
                f"weighted transport variable {index} shape mismatch"
            )
        tf.debugging.assert_all_finite(tensor, f"weighted transport variable {index}")
        variable.assign(tensor)
    tensor_hash = stable_json_hash(
        [variable.read_value() for variable in transport.trainable_variables]
    )
    transport.bind_frozen_identity(
        {
            "checkpoint_sha256": actual_file_hash,
            "training_state_hash": state_hash,
            "transport_tensor_hash": tensor_hash,
        }
    )
    return LoadedWeightedNeuTraTransport(
        transport=transport,
        checkpoint_path=path.as_posix(),
        checkpoint_sha256=actual_file_hash,
        state_hash=state_hash,
        transport_tensor_hash=tensor_hash,
        selected_step=int(state.get("step", -1)),
        config=config,
    )


def component_aware_initial_state(
    transport: WeightedDenseIAFTransport,
    target: Mapping[str, Any],
    *,
    chain_count: int = 4,
) -> tf.Tensor:
    """Place chains near every known component before mapping to latent space."""

    if isinstance(chain_count, bool) or int(chain_count) < 2:
        raise ValueError("chain_count must be at least two")
    probabilities, means, _covariances, _ = validate_gaussian_mixture(
        target["probabilities"], target["means"], target["covariances"]
    )
    del probabilities
    component_count = int(means.shape[0])
    dimension = int(means.shape[1])
    if transport.parameter_dim != dimension:
        raise ValueError("transport and target dimensions differ")
    labels = tf.math.floormod(tf.range(int(chain_count), dtype=tf.int32), component_count)
    direction = tf.cast(tf.range(dimension), tf.float64)
    unit = tf.math.divide_no_nan(direction + 1.0, tf.linalg.norm(direction + 1.0))
    signs = tf.where(
        tf.math.floormod(tf.range(int(chain_count)), 2) == 0,
        tf.constant(-1.0, tf.float64),
        tf.constant(1.0, tf.float64),
    )
    physical = tf.gather(means, labels) + 0.05 * signs[:, tf.newaxis] * unit
    latent, _ = transport.inverse_and_forward_logdet(physical)
    tf.debugging.assert_all_finite(latent, "component-aware latent initial state")
    return latent


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WeightedNeuTraGaussianMixtureHMCError(f"unreadable JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise WeightedNeuTraGaussianMixtureHMCError("JSON object required")
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
    "LoadedWeightedNeuTraTransport",
    "WeightedNeuTraGaussianMixtureHMCError",
    "analytic_three_mode_target",
    "component_aware_initial_state",
    "load_weighted_neutra_transport",
    "sha256_file",
    "stable_json_hash",
]

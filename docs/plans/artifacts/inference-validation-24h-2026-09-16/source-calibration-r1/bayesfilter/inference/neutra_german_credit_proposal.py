"""Frozen German NeuTra transport loading and defensive proposal mechanics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)
from bayesfilter.inference.neutra_german_credit_target import GermanCreditTargetSpec


@dataclass(frozen=True)
class FrozenGermanTransport:
    transport: WeightedDenseIAFTransport
    selected_update: int
    state_hash: str
    state_sha256: str
    target_name: str
    target_data_sha256: str
    target_reference_sha256: str


def load_frozen_german_transport(
    state_path: str | Path,
    artifact_hashes_path: str | Path,
    *,
    expected_schema: str,
) -> FrozenGermanTransport:
    """Load and bind a hash-verified frozen German dense-IAF transport."""

    state_file = Path(state_path).resolve()
    hashes_file = Path(artifact_hashes_path).resolve()
    state = _load_json(state_file)
    hashes = _load_json(hashes_file)
    if hashes.get("artifacts", {}).get(state_file.name) != _sha256(state_file):
        raise RuntimeError("German transport state artifact hash mismatch")
    if state.get("schema") != str(expected_schema):
        raise RuntimeError("German transport state schema mismatch")
    expected_state_hash = str(state.get("state_hash", ""))
    semantic_payload = {key: value for key, value in state.items() if key != "state_hash"}
    if len(expected_state_hash) != 64 or _stable_hash(semantic_payload) != expected_state_hash:
        raise RuntimeError("German transport semantic state hash mismatch")
    config_payload = dict(state.get("config", {}))
    config_payload.pop("schema", None)
    config_payload["hidden_layers"] = tuple(config_payload["hidden_layers"])
    config_payload["initialization_seed"] = tuple(config_payload["initialization_seed"])
    config = WeightedNeuTraConfig(**config_payload)
    transport = WeightedDenseIAFTransport(config)
    variables = state.get("variables")
    if not isinstance(variables, list) or len(variables) != len(transport.trainable_variables):
        raise RuntimeError("German transport state variable count mismatch")
    for variable, raw in zip(transport.trainable_variables, variables, strict=True):
        tensor = tf.convert_to_tensor(raw, tf.float64)
        if tensor.shape != variable.shape:
            raise RuntimeError("German transport state variable shape mismatch")
        tf.debugging.assert_all_finite(tensor, "German transport variable")
        variable.assign(tensor)
    tensor_hash = _stable_hash(
        [variable.read_value().numpy().tolist() for variable in transport.trainable_variables]
    )
    transport.bind_frozen_identity(
        {
            "checkpoint_sha256": _sha256(state_file),
            "training_state_hash": expected_state_hash,
            "transport_tensor_hash": tensor_hash,
        }
    )
    return FrozenGermanTransport(
        transport=transport,
        selected_update=int(state["selected_update"]),
        state_hash=expected_state_hash,
        state_sha256=_sha256(state_file),
        target_name=str(state["target_name"]),
        target_data_sha256=str(state["target_data_sha256"]),
        target_reference_sha256=str(state["target_reference_sha256"]),
    )


def validate_defensive_base_mixture(
    scales: Sequence[float],
    probabilities: Sequence[float],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Validate the positive isotropic base-normal mixture parameters."""

    scale_values = tuple(float(value) for value in scales)
    probability_values = tuple(float(value) for value in probabilities)
    if len(scale_values) < 2 or len(scale_values) != len(probability_values):
        raise ValueError("proposal scales/probabilities must share at least two entries")
    if any(not math.isfinite(value) or value <= 0.0 for value in scale_values):
        raise ValueError("proposal scales must be finite and positive")
    if any(not math.isfinite(value) or value <= 0.0 for value in probability_values):
        raise ValueError("proposal probabilities must be finite and positive")
    if not math.isclose(sum(probability_values), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("proposal probabilities must sum to one")
    return (
        tf.constant(scale_values, tf.float64),
        tf.constant(probability_values, tf.float64),
    )


def defensive_base_mixture_log_prob(
    latent: Any,
    scales: Sequence[float],
    probabilities: Sequence[float],
) -> tf.Tensor:
    """Evaluate the normalized isotropic Gaussian scale-mixture density."""

    scale, probability = validate_defensive_base_mixture(scales, probabilities)
    rows = tf.convert_to_tensor(latent, tf.float64)
    if rows.shape.rank != 2 or rows.shape[1] is None:
        raise ValueError("latent must have shape [row, dimension]")
    tf.debugging.assert_all_finite(rows, "proposal latent")
    dimension = tf.cast(tf.shape(rows)[1], tf.float64)
    quadratic = tf.reduce_sum(tf.square(rows), axis=1)[:, tf.newaxis] / tf.square(
        scale[tf.newaxis, :]
    )
    log_normalizer = dimension * (
        tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
        + 2.0 * tf.math.log(scale)
    )
    component = -0.5 * (quadratic + log_normalizer[tf.newaxis, :])
    return tf.reduce_logsumexp(
        tf.math.log(probability)[tf.newaxis, :] + component, axis=1
    )


def sample_defensive_pushed_proposal(
    transport: WeightedDenseIAFTransport,
    sample_count: int,
    scales: Sequence[float],
    probabilities: Sequence[float],
    *,
    seed: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Sample the pushed base mixture and return physical rows and log density."""

    if isinstance(sample_count, bool) or int(sample_count) <= 1:
        raise ValueError("sample_count must exceed one")
    scale, probability = validate_defensive_base_mixture(scales, probabilities)
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 2)
    labels = tf.cast(
        tf.reshape(
            tf.random.stateless_categorical(
                tf.math.log(probability)[tf.newaxis, :], int(sample_count), seed=split[0]
            ),
            (-1,),
        ),
        tf.int32,
    )
    noise = tf.random.stateless_normal(
        (int(sample_count), int(transport.parameter_dim)),
        seed=split[1],
        dtype=tf.float64,
    )
    latent = noise * tf.gather(scale, labels)[:, tf.newaxis]
    physical, forward_logdet = transport.forward_and_logdet(latent)
    log_proposal = defensive_base_mixture_log_prob(
        latent, scales, probabilities
    ) - forward_logdet
    tf.debugging.assert_all_finite(physical, "pushed proposal physical rows")
    tf.debugging.assert_all_finite(log_proposal, "pushed proposal log density")
    return physical, latent, log_proposal, labels


def reference_marginal_unconstrained_parameters(
    spec: GermanCreditTargetSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fit independent unconstrained marginals to committed constrained moments."""

    feature_count = int(spec.feature_count)
    mean = tf.constant(spec.reference_mean, tf.float64)
    square = tf.constant(spec.reference_square, tf.float64)
    z_mean = mean[:feature_count]
    z_variance = square[:feature_count] - tf.square(z_mean)
    scale_mean = mean[feature_count:]
    scale_square = square[feature_count:]
    tf.debugging.assert_positive(z_variance, "German reference z variances")
    tf.debugging.assert_positive(scale_mean, "German reference scale means")
    tf.debugging.assert_greater(
        scale_square, tf.square(scale_mean), "German reference scale variances"
    )
    log_variance = tf.math.log(scale_square / tf.square(scale_mean))
    log_mean = tf.math.log(scale_mean) - 0.5 * log_variance
    location = tf.concat((z_mean, log_mean), axis=0)
    standard_deviation = tf.sqrt(tf.concat((z_variance, log_variance), axis=0))
    tf.debugging.assert_all_finite(location, "German reference proposal location")
    tf.debugging.assert_all_finite(
        standard_deviation, "German reference proposal standard deviation"
    )
    return location, standard_deviation


def reference_diagonal_mixture_log_prob(
    unconstrained: Any,
    spec: GermanCreditTargetSpec,
    scales: Sequence[float] = (1.0, 1.5),
    probabilities: Sequence[float] = (0.85 / 0.95, 0.10 / 0.95),
) -> tf.Tensor:
    """Evaluate the normalized reference-marginal diagonal Gaussian mixture."""

    scale, probability = validate_defensive_base_mixture(scales, probabilities)
    rows = tf.convert_to_tensor(unconstrained, tf.float64)
    if rows.shape.rank != 2 or rows.shape[1] != int(spec.dimension):
        raise ValueError("unconstrained must have shape [row, German dimension]")
    location, standard_deviation = reference_marginal_unconstrained_parameters(spec)
    component_sd = standard_deviation[tf.newaxis, :] * scale[:, tf.newaxis]
    centered = rows[:, tf.newaxis, :] - location[tf.newaxis, tf.newaxis, :]
    quadratic = tf.reduce_sum(
        tf.square(centered / component_sd[tf.newaxis, :, :]), axis=2
    )
    log_normalizer = tf.reduce_sum(
        tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
        + 2.0 * tf.math.log(component_sd),
        axis=1,
    )
    component = -0.5 * (quadratic + log_normalizer[tf.newaxis, :])
    return tf.reduce_logsumexp(
        tf.math.log(probability)[tf.newaxis, :] + component, axis=1
    )


def sample_reference_augmented_proposal(
    transport: WeightedDenseIAFTransport,
    spec: GermanCreditTargetSpec,
    sample_count: int,
    *,
    seed: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Sample 0.85/0.10 reference marginals plus 0.05 reverse pushforward."""

    if isinstance(sample_count, bool) or int(sample_count) <= 1:
        raise ValueError("sample_count must exceed one")
    weights = tf.constant((0.85, 0.10, 0.05), tf.float64)
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 3)
    labels = tf.cast(
        tf.reshape(
            tf.random.stateless_categorical(
                tf.math.log(weights)[tf.newaxis, :], int(sample_count), seed=split[0]
            ),
            (-1,),
        ),
        tf.int32,
    )
    location, standard_deviation = reference_marginal_unconstrained_parameters(spec)
    noise = tf.random.stateless_normal(
        (int(sample_count), int(spec.dimension)), seed=split[1], dtype=tf.float64
    )
    reference_scale = tf.where(
        labels == 1,
        tf.constant(1.5, tf.float64),
        tf.constant(1.0, tf.float64),
    )
    reference_rows = location[tf.newaxis, :] + (
        standard_deviation[tf.newaxis, :] * reference_scale[:, tf.newaxis] * noise
    )
    reverse_latent = tf.random.stateless_normal(
        (int(sample_count), int(spec.dimension)), seed=split[2], dtype=tf.float64
    )
    reverse_rows, reverse_logdet = transport.forward_and_logdet(reverse_latent)
    physical = tf.where((labels == 2)[:, tf.newaxis], reverse_rows, reference_rows)
    reference_log_prob = reference_diagonal_mixture_log_prob(physical, spec)
    reverse_log_prob = transport.log_prob(physical)
    log_proposal = tf.reduce_logsumexp(
        tf.stack(
            (
                tf.math.log(tf.constant(0.95, tf.float64)) + reference_log_prob,
                tf.math.log(tf.constant(0.05, tf.float64)) + reverse_log_prob,
            ),
            axis=1,
        ),
        axis=1,
    )
    del reverse_logdet
    tf.debugging.assert_all_finite(physical, "reference-augmented proposal rows")
    tf.debugging.assert_all_finite(log_proposal, "reference-augmented proposal density")
    return physical, reverse_latent, log_proposal, labels


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()

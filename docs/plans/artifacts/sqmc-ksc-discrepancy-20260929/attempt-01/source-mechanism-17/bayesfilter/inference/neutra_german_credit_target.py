"""TensorFlow target and exact score for German-credit weighted NeuTra tests.

The source target is ``dsge_hmc.benchmarks.neutra_german.GermanGammaTarget``.
This module preserves its numeric-data preprocessing, unconstrained coordinate
order, target value, and score without importing NumPy or the source repository
at candidate runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability


GERMAN_TARGET_NAME = "german_gamma_scales2"


class GermanCreditTargetError(RuntimeError):
    """Raised when source-bound German target inputs are invalid."""


@dataclass(frozen=True)
class GermanCreditTargetSpec:
    """Frozen numeric design, response, and constrained reference moments."""

    name: str
    observation_count: int
    feature_count: int
    dimension: int
    design: tuple[tuple[float, ...], ...]
    response: tuple[float, ...]
    reference_mean: tuple[float, ...]
    reference_square: tuple[float, ...]
    data_path: str
    data_sha256: str
    reference_path: str
    reference_sha256: str

    def __post_init__(self) -> None:
        if self.name != GERMAN_TARGET_NAME:
            raise ValueError("unsupported German-credit target")
        if int(self.observation_count) <= 0 or int(self.feature_count) <= 0:
            raise ValueError("German target sizes must be positive")
        if int(self.dimension) != 2 * int(self.feature_count) + 1:
            raise ValueError("German target dimension must equal 2*d+1")
        if len(self.design) != int(self.observation_count):
            raise ValueError("German design observation count mismatch")
        if any(len(row) != int(self.feature_count) for row in self.design):
            raise ValueError("German design feature count mismatch")
        if len(self.response) != int(self.observation_count):
            raise ValueError("German response observation count mismatch")
        if any(value not in (0.0, 1.0) for value in self.response):
            raise ValueError("German response must contain only zero and one")
        if len(self.reference_mean) != int(self.dimension):
            raise ValueError("German reference mean width mismatch")
        if len(self.reference_square) != int(self.dimension):
            raise ValueError("German reference square width mismatch")
        values = (
            *(item for row in self.design for item in row),
            *self.response,
            *self.reference_mean,
            *self.reference_square,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("German target values must be finite")
        for name in ("data_sha256", "reference_sha256"):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")

    def manifest_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("design")
        payload.pop("response")
        payload.pop("reference_mean")
        payload.pop("reference_square")
        payload.update(
            {
                "schema": "bayesfilter.neutra.german_credit_target.v1",
                "coordinate_contract": (
                    "unconstrained=[z,log_local_scale,log_global_scale]; "
                    "constrained=[z,local_scale,global_scale]"
                ),
                "source_formula": (
                    "dsge_hmc.benchmarks.neutra_german."
                    "GermanGammaTarget.log_prob_batch_tf"
                ),
                "source_preprocessing": (
                    "raw_predictor_divided_by_range_without_min_subtraction; "
                    "mapped_by_2x_minus_1; intercept_appended; labels_1_2_to_0_1"
                ),
                "reference_coordinate": "constrained",
                "reference_mean": list(self.reference_mean),
                "reference_square": list(self.reference_square),
            }
        )
        return payload


def load_german_credit_target_spec(
    data_path: str | Path,
    reference_path: str | Path,
) -> GermanCreditTargetSpec:
    """Load source-bound numeric data and reference moments using stdlib only."""

    data_file = Path(data_path).resolve()
    reference_file = Path(reference_path).resolve()
    data_raw = data_file.read_bytes()
    reference_raw = reference_file.read_bytes()
    rows: list[list[float]] = []
    try:
        for line in data_raw.decode("utf-8").splitlines():
            if line.strip():
                rows.append([float(value) for value in line.split()])
    except (UnicodeDecodeError, ValueError) as error:
        raise GermanCreditTargetError("German numeric data are unreadable") from error
    if not rows or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise GermanCreditTargetError("German numeric data must be a rectangular matrix")
    raw_feature_count = len(rows[0]) - 1
    raw_x = [row[:-1] for row in rows]
    labels = [row[-1] for row in rows]
    minima = [min(row[index] for row in raw_x) for index in range(raw_feature_count)]
    maxima = [max(row[index] for row in raw_x) for index in range(raw_feature_count)]
    # The source computes minima for the range but intentionally does not
    # subtract them when scaling each observation.
    ranges = [maximum - minimum for maximum, minimum in zip(maxima, minima)]
    if any(not math.isfinite(value) or value <= 0.0 for value in ranges):
        raise GermanCreditTargetError("German numeric data contain a constant feature")
    design = tuple(
        tuple(2.0 * row[index] / ranges[index] - 1.0 for index in range(raw_feature_count))
        + (1.0,)
        for row in raw_x
    )
    response = tuple(float(value - 1.0) for value in labels)
    if any(value not in (0.0, 1.0) for value in response):
        raise GermanCreditTargetError("German labels must be encoded as one or two")
    try:
        reference = json.loads(reference_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GermanCreditTargetError("German reference JSON is unreadable") from error
    if not isinstance(reference, Mapping):
        raise GermanCreditTargetError("German reference JSON object is required")
    try:
        mean = tuple(float(value) for value in reference["mean"])
        square = tuple(float(value) for value in reference["square"])
    except (KeyError, TypeError, ValueError) as error:
        raise GermanCreditTargetError("German reference moments are invalid") from error
    feature_count = raw_feature_count + 1
    return GermanCreditTargetSpec(
        name=GERMAN_TARGET_NAME,
        observation_count=len(rows),
        feature_count=feature_count,
        dimension=2 * feature_count + 1,
        design=design,
        response=response,
        reference_mean=mean,
        reference_square=square,
        data_path=data_file.as_posix(),
        data_sha256=hashlib.sha256(data_raw).hexdigest(),
        reference_path=reference_file.as_posix(),
        reference_sha256=hashlib.sha256(reference_raw).hexdigest(),
    )


def german_credit_log_prob_and_score_batch(
    spec: GermanCreditTargetSpec,
    unconstrained: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate the German sparse-logistic target and its exact score."""

    rows = _rank2(unconstrained, spec.dimension, "unconstrained")
    feature_count = int(spec.feature_count)
    design = tf.constant(spec.design, tf.float64)
    response = tf.constant(spec.response, tf.float64)
    value = german_credit_log_prob_batch(spec, rows)
    z = rows[:, :feature_count]
    log_local = rows[:, feature_count : 2 * feature_count]
    log_global = rows[:, 2 * feature_count]
    local = tf.exp(log_local)
    global_scale = tf.exp(log_global)
    beta = z * local * global_scale[:, tf.newaxis]
    logits = tf.matmul(beta, design, transpose_b=True)
    probability = tf.math.sigmoid(logits)
    beta_score = tf.matmul(response[tf.newaxis, :] - probability, design)
    z_score = beta_score * local * global_scale[:, tf.newaxis] - z
    log_local_score = (
        beta_score * beta
        + tf.constant(0.5, tf.float64)
        - tf.constant(0.5, tf.float64) * local
    )
    log_global_score = (
        tf.reduce_sum(beta_score * beta, axis=1)
        + tf.constant(0.5, tf.float64)
        - tf.constant(0.5, tf.float64) * global_scale
    )
    score = tf.concat(
        (z_score, log_local_score, log_global_score[:, tf.newaxis]), axis=1
    )
    tf.debugging.assert_all_finite(value, "German-credit target value")
    tf.debugging.assert_all_finite(score, "German-credit target score")
    return value, score


def german_credit_log_prob_batch(
    spec: GermanCreditTargetSpec,
    unconstrained: Any,
) -> tf.Tensor:
    """Evaluate only the batch-native target value for reverse-KL training."""

    rows = _rank2(unconstrained, spec.dimension, "unconstrained")
    feature_count = int(spec.feature_count)
    design = tf.constant(spec.design, tf.float64)
    response = tf.constant(spec.response, tf.float64)
    z = rows[:, :feature_count]
    log_local = rows[:, feature_count : 2 * feature_count]
    log_global = rows[:, 2 * feature_count]
    local = tf.exp(log_local)
    global_scale = tf.exp(log_global)
    beta = z * local * global_scale[:, tf.newaxis]
    logits = tf.matmul(beta, design, transpose_b=True)
    log_likelihood = tf.reduce_sum(
        response[tf.newaxis, :] * -tf.nn.softplus(-logits)
        + (tf.constant(1.0, tf.float64) - response[tf.newaxis, :])
        * -tf.nn.softplus(logits),
        axis=1,
    )
    return (
        log_likelihood
        - tf.constant(0.5, tf.float64) * tf.reduce_sum(tf.square(z), axis=1)
        + tf.reduce_sum(
            tf.constant(0.5, tf.float64) * log_local
            - tf.constant(0.5, tf.float64) * local,
            axis=1,
        )
        + tf.constant(0.5, tf.float64) * log_global
        - tf.constant(0.5, tf.float64) * global_scale
    )


def constrained_from_unconstrained(
    spec: GermanCreditTargetSpec,
    unconstrained: Any,
) -> tf.Tensor:
    """Map unconstrained rows to the committed Stan reference coordinate."""

    rows = _rank2(unconstrained, spec.dimension, "unconstrained")
    feature_count = int(spec.feature_count)
    return tf.concat(
        (
            rows[:, :feature_count],
            tf.exp(rows[:, feature_count : 2 * feature_count]),
            tf.exp(rows[:, 2 * feature_count :]),
        ),
        axis=1,
    )


class GermanCreditValueScoreAdapter:
    """Graph-native batch value/score adapter for the frozen German target."""

    supports_retained_draw_batch = False
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True
    target_status_invalid_rows_become_nonfinite = False

    def __init__(self, spec: GermanCreditTargetSpec) -> None:
        self.spec = spec
        self.parameter_dim = int(spec.dimension)
        self.target_scope = "weighted_neutra_german_credit:gamma_scales2"

    def log_prob_and_grad(self, unconstrained: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return german_credit_log_prob_and_score_batch(self.spec, unconstrained)

    def log_prob_and_grad_status(
        self, unconstrained: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        value, score = self.log_prob_and_grad(unconstrained)
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

    def target_status_telemetry(self, unconstrained: Any) -> Mapping[str, tf.Tensor]:
        return self.log_prob_and_grad_status(unconstrained)[2]

    def adapter_signature(self) -> str:
        return _stable_hash(
            {
                "schema": "bayesfilter.neutra.german_credit_value_score.v1",
                "target_scope": self.target_scope,
                "target": self.spec.manifest_payload(),
                "value_score_authority": "graph_native_exact_sparse_logistic_score",
            }
        )

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_exact_german_credit_sparse_logistic_score",
            evidence_path=(
                "docs/plans/"
                "bayesfilter-weighted-forward-kl-german-credit-plan-2026-08-13.md"
            ),
            target_scope=self.target_scope,
            nonclaims=(
                "source-bound German-credit gamma-scales target only",
                "committed reference moments have no stored reference MCSE",
                "no HMC validity or objective ranking claim",
            ),
        )


def _rank2(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[-1] != int(dimension):
        raise ValueError(f"{name} must have shape [row, {int(dimension)}]")
    if tensor.shape[0] is None:
        raise ValueError(f"{name} row count must be static")
    tf.debugging.assert_all_finite(tensor, name)
    return tensor


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()

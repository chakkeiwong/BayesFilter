"""Reusable exact 18D deterministic LGSSM target bound to persisted artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.linear.batched_kalman_svd_derivatives_tf import (
    tf_batched_svd_linear_gaussian_score_first_order_graph_status,
)
from bayesfilter.linear.kalman_svd_derivatives_tf import (
    SVD_LINEAR_SCORE_STATUS_INVALID_EIGENSOLVER_INPUT,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.testing import multidim_triangular_lgssm_tf as triangular
from bayesfilter.testing.multidim_triangular_lgssm_batched_tf import (
    gaussian_raw_prior_log_prob_and_score_batch,
    materialize_lower_triangular_lgssm_batch,
)


def stable_config_hash(config: Any) -> str:
    normalized = _json_ready(config)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_full_estimation_rerun_2026_07_13.json"
)
DEFAULT_FIXTURE_PATH = ROOT / (
    "docs/benchmarks/artifacts/"
    "multidim_lgssm_full_estimation_rerun_2026_07_13/"
    "fixture_T120_seed20260709_301.json"
)
TARGET_SCOPE = "bayesfilter_multidim_lower_triangular_lgssm_t120_exact_v1"
CONFIG_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_config.v1"
FIXTURE_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_fixture.v1"
TARGET_SIGNATURE_SCHEMA = "bayesfilter.deterministic_lgssm_exact_target.v1"


class InvalidDeterministicLGSSMTarget(ValueError):
    """Raised when persisted exact-target identity or shape checks fail."""


@dataclass(frozen=True)
class DeterministicLGSSMExactTargetBundle:
    """Fixture-bound adapter and stable target identity."""

    adapter: "DeterministicLGSSMExactPosteriorAdapter"
    config: Mapping[str, Any]
    fixture: Mapping[str, Any]
    contract: Mapping[str, Any]
    target_signature: str
    target_signature_payload: Mapping[str, Any]
    config_path: Path
    fixture_path: Path
    contract_path: Path

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.adapter.parameter_names()

    @property
    def raw_truth(self) -> tf.Tensor:
        return tf.constant(self.fixture["raw_truth"], dtype=tf.float64)


class DeterministicLGSSMExactPosteriorAdapter:
    """Graph-native exact LGSSM value/score adapter with fixture-bound identity."""

    parameter_dim = 18

    def __init__(
        self,
        *,
        observations: Any,
        contract: Mapping[str, Any],
        parameter_names: Sequence[str],
        target_signature: str,
        evidence_path: str,
    ) -> None:
        names = tuple(str(item) for item in parameter_names)
        if len(names) != self.parameter_dim:
            raise InvalidDeterministicLGSSMTarget(
                "parameter_names length must equal 18"
            )
        signature = _bare_sha256(target_signature, "target_signature")
        observations_tensor = tf.convert_to_tensor(observations, dtype=tf.float64)
        if observations_tensor.shape != (120, 4):
            raise InvalidDeterministicLGSSMTarget(
                "observations must have exact shape [120, 4]"
            )
        self._observations = observations_tensor
        self._contract = dict(contract)
        self._parameter_names = names
        self.target_signature = signature
        self._evidence_path = str(evidence_path)

    def adapter_signature(self) -> str:
        return stable_config_hash(
            {
                "schema": "bayesfilter.deterministic_lgssm_exact_adapter.v1",
                "target_signature": self.target_signature,
                "target_scope": TARGET_SCOPE,
                "parameter_names": self._parameter_names,
                "parameter_dim": self.parameter_dim,
                "runtime_backend": (
                    "tensorflow_manual_lgssm_svd_graph_status_score"
                ),
            }
        )

    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_manual_lgssm_svd_graph_status_score",
            evidence_path=self._evidence_path,
            target_scope=TARGET_SCOPE,
            nonclaims=(
                "fixture-bound exact LGSSM target adapter only",
                "no NeuTra training claim",
                "no HMC convergence or recovery claim",
                "no posterior correctness claim",
                "no production or default readiness claim",
            ),
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = self.log_prob_and_grad(theta)
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = self.log_prob_and_grad_status(theta)
        return value, score

    def log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        """Return value, score, and status from one exact-target evaluation."""

        theta_tensor = self._validate_theta_tensor(theta)
        if theta_tensor.shape.rank == 1:
            return self._single_log_prob_grad_status(theta_tensor)
        flat_theta = tf.reshape(theta_tensor, (-1, self.parameter_dim))
        rows = tf.map_fn(
            self._single_log_prob_grad_status_tuple,
            flat_theta,
            fn_output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(self.parameter_dim,), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.bool),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.float64),
            ),
        )
        leading_shape = tf.shape(theta_tensor)[:-1]
        score_shape = tf.concat(
            [leading_shape, tf.constant([self.parameter_dim], dtype=tf.int32)],
            axis=0,
        )
        return (
            tf.reshape(rows[0], leading_shape),
            tf.reshape(rows[1], score_shape),
            {
                "status_code": tf.reshape(rows[2], leading_shape),
                "valid_pre_regularized_score": tf.reshape(rows[3], leading_shape),
                "floor_count_value": tf.reshape(rows[4], leading_shape),
                "min_innovation_eigenvalue": tf.reshape(rows[5], leading_shape),
                "innovation_condition_estimate": tf.reshape(rows[6], leading_shape),
            },
        )

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        """Return the exact posterior batch through the batch-native kernel."""

        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        if values.shape.rank != 2:
            raise ValueError("NeuTra exact target requires rank 2 theta")
        if values.shape[-1] is not None and int(values.shape[-1]) != self.parameter_dim:
            raise ValueError("theta trailing dimension must equal 18")
        finite_input = tf.reduce_all(tf.math.is_finite(values), axis=1)
        safe_values = tf.where(
            finite_input[:, tf.newaxis],
            values,
            tf.zeros_like(values),
        )
        materialized = materialize_lower_triangular_lgssm_batch(
            safe_values,
            self._contract,
        )
        likelihood = tf_batched_svd_linear_gaussian_score_first_order_graph_status(
            self._observations,
            transition_offset=materialized.transition_offset,
            transition_matrix=materialized.transition_matrix,
            transition_covariance=materialized.transition_covariance,
            observation_offset=materialized.observation_offset,
            observation_matrix=materialized.observation_matrix,
            observation_covariance=materialized.observation_covariance,
            initial_state_mean=materialized.initial_mean,
            initial_state_covariance=materialized.initial_covariance,
            d_initial_state_mean=materialized.d_initial_mean,
            d_initial_state_covariance=materialized.d_initial_covariance,
            d_transition_offset=materialized.d_transition_offset,
            d_transition_matrix=materialized.d_transition_matrix,
            d_transition_covariance=materialized.d_transition_covariance,
            d_observation_offset=materialized.d_observation_offset,
            d_observation_matrix=materialized.d_observation_matrix,
            d_observation_covariance=materialized.d_observation_covariance,
            jitter=tf.constant(1.0e-9, tf.float64),
            singular_floor=tf.constant(1.0e-12, tf.float64),
        )
        prior_value, prior_score = gaussian_raw_prior_log_prob_and_score_batch(
            safe_values,
            self._contract,
        )
        finite_prior = tf.logical_and(
            tf.math.is_finite(prior_value),
            tf.reduce_all(tf.math.is_finite(prior_score), axis=1),
        )
        finite_likelihood = tf.logical_and(
            tf.math.is_finite(likelihood.log_likelihood),
            tf.reduce_all(tf.math.is_finite(likelihood.score), axis=1),
        )
        valid = tf.logical_and(
            tf.logical_and(
                finite_input,
                likelihood.valid_pre_regularized_score,
            ),
            tf.logical_and(finite_prior, finite_likelihood),
        )
        nan = tf.constant(float("nan"), tf.float64)
        posterior_value = tf.where(
            valid,
            prior_value + likelihood.log_likelihood,
            nan,
        )
        posterior_score = tf.where(
            valid[:, tf.newaxis],
            prior_score + likelihood.score,
            tf.fill(tf.shape(prior_score), nan),
        )
        status_code = tf.where(
            finite_input,
            likelihood.status_code,
            tf.constant(
                SVD_LINEAR_SCORE_STATUS_INVALID_EIGENSOLVER_INPUT,
                tf.int32,
            ),
        )
        return posterior_value, posterior_score, {
            "status_code": status_code,
            "valid_pre_regularized_score": valid,
            "floor_count_value": likelihood.floor_count_value,
            "min_innovation_eigenvalue": likelihood.min_innovation_eigenvalue,
            "innovation_condition_estimate": likelihood.innovation_condition_estimate,
        }

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        theta_tensor = self._validate_theta_tensor(theta)
        if theta_tensor.shape.rank == 1:
            return self._single_target_status(theta_tensor)
        flat_theta = tf.reshape(theta_tensor, (-1, self.parameter_dim))
        rows = tf.map_fn(
            self._single_target_status_tuple,
            flat_theta,
            fn_output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.bool),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.float64),
            ),
        )
        leading_shape = tf.shape(theta_tensor)[:-1]
        return {
            "status_code": tf.reshape(rows[0], leading_shape),
            "valid_pre_regularized_score": tf.reshape(rows[1], leading_shape),
            "floor_count_value": tf.reshape(rows[2], leading_shape),
            "min_innovation_eigenvalue": tf.reshape(rows[3], leading_shape),
            "innovation_condition_estimate": tf.reshape(rows[4], leading_shape),
        }

    def _single_log_prob_and_grad(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = self._single_log_prob_grad_status(theta)
        return value, score

    def _single_log_prob_grad_status(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        value, score, _likelihood, _likelihood_score, status = (
            triangular.lower_triangular_lgssm_log_prob_score_status(
                theta,
                self._observations,
                self._contract,
            )
        )
        return value, score, status

    def _single_log_prob_grad_status_tuple(
        self, theta: tf.Tensor
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        value, score, status = self._single_log_prob_grad_status(theta)
        return (
            value,
            score,
            status["status_code"],
            status["valid_pre_regularized_score"],
            status["floor_count_value"],
            status["min_innovation_eigenvalue"],
            status["innovation_condition_estimate"],
        )

    def _single_target_status(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        _value, _score, _likelihood, _likelihood_score, status = (
            triangular.lower_triangular_lgssm_log_prob_score_status(
                theta,
                self._observations,
                self._contract,
            )
        )
        return status

    def _single_target_status_tuple(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        status = self._single_target_status(theta)
        return (
            status["status_code"],
            status["valid_pre_regularized_score"],
            status["floor_count_value"],
            status["min_innovation_eigenvalue"],
            status["innovation_condition_estimate"],
        )

    def _validate_theta_tensor(self, theta: Any) -> tf.Tensor:
        tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
        if tensor.shape.rank is None or tensor.shape.rank < 1:
            raise ValueError("theta must have static rank at least 1")
        if tensor.shape[-1] is None or int(tensor.shape[-1]) != self.parameter_dim:
            raise ValueError("theta trailing dimension must equal 18")
        return tensor


def load_deterministic_lgssm_exact_target(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    expected_target_signature: str | None = None,
) -> DeterministicLGSSMExactTargetBundle:
    """Load and validate the persisted 18D target and return a stable adapter."""

    config_file = _absolute(config_path)
    fixture_file = _absolute(fixture_path)
    config = _read_mapping(config_file, "config")
    fixture = _read_mapping(fixture_file, "fixture")
    if config.get("schema") != CONFIG_SCHEMA:
        raise InvalidDeterministicLGSSMTarget("config schema mismatch")
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise InvalidDeterministicLGSSMTarget("fixture schema mismatch")

    tagged_config_hash = f"sha256:{stable_config_hash(config)}"
    if fixture.get("config_hash") != tagged_config_hash:
        raise InvalidDeterministicLGSSMTarget("fixture config hash mismatch")
    _validate_embedded_artifact_hash(fixture, "fixture")

    contract_file = _absolute(config["source_contract"]["path"])
    contract = _read_mapping(contract_file, "source contract")
    if contract.get("schema") != config["source_contract"].get("schema"):
        raise InvalidDeterministicLGSSMTarget("source contract schema mismatch")
    if contract.get("contract_id") != config["source_contract"].get("contract_id"):
        raise InvalidDeterministicLGSSMTarget("source contract id mismatch")
    if contract.get("target_id") != config["source_contract"].get("target_id"):
        raise InvalidDeterministicLGSSMTarget("source contract target id mismatch")

    config_names = tuple(str(item) for item in config["model"]["parameter_names"])
    fixture_names = tuple(str(item) for item in fixture["parameter_names"])
    contract_names = tuple(str(item) for item in contract["parameter_names"])
    if config_names != fixture_names or config_names != contract_names:
        raise InvalidDeterministicLGSSMTarget("parameter order mismatch")
    if len(config_names) != 18:
        raise InvalidDeterministicLGSSMTarget("parameter dimension mismatch")
    if int(fixture.get("horizon", -1)) != 120:
        raise InvalidDeterministicLGSSMTarget("fixture horizon must equal 120")
    observations = fixture.get("observations")
    observations_tensor = tf.convert_to_tensor(observations, dtype=tf.float64)
    if observations_tensor.shape != (120, 4):
        raise InvalidDeterministicLGSSMTarget(
            "fixture observations must have shape [120, 4]"
        )
    if not bool(tf.reduce_all(tf.math.is_finite(observations_tensor)).numpy()):
        raise InvalidDeterministicLGSSMTarget("fixture observations must be finite")

    source_paths = (
        ROOT / "bayesfilter/testing/multidim_triangular_lgssm_tf.py",
        ROOT / "bayesfilter/linear/kalman_svd_derivatives_tf.py",
    )
    target_payload = {
        "schema": TARGET_SIGNATURE_SCHEMA,
        "target_scope": TARGET_SCOPE,
        "coordinate_convention": config["model"]["coordinate_convention"],
        "log_jacobian_convention": config["prior"]["log_jacobian_convention"],
        "parameter_names": config_names,
        "parameter_dim": 18,
        "horizon": 120,
        "config_hash": tagged_config_hash,
        "config_file_sha256": _file_sha256(config_file),
        "fixture_artifact_hash": fixture["artifact_hash"],
        "fixture_file_sha256": _file_sha256(fixture_file),
        "observations_hash": stable_config_hash(observations),
        "source_contract_hash": stable_config_hash(contract),
        "source_contract_file_sha256": _file_sha256(contract_file),
        "target_source_files": {
            str(path.relative_to(ROOT)): _file_sha256(path) for path in source_paths
        },
        "target_math": (
            "manual lower-triangular LGSSM posterior value and SVD graph-status score"
        ),
    }
    target_signature = stable_config_hash(target_payload)
    if expected_target_signature is not None:
        expected = _bare_sha256(expected_target_signature, "expected_target_signature")
        if target_signature != expected:
            raise InvalidDeterministicLGSSMTarget("target signature mismatch")

    evidence_path = str(fixture_file.relative_to(ROOT))
    adapter = DeterministicLGSSMExactPosteriorAdapter(
        observations=observations,
        contract=contract,
        parameter_names=config_names,
        target_signature=target_signature,
        evidence_path=evidence_path,
    )
    return DeterministicLGSSMExactTargetBundle(
        adapter=adapter,
        config=config,
        fixture=fixture,
        contract=contract,
        target_signature=target_signature,
        target_signature_payload=target_payload,
        config_path=config_file,
        fixture_path=fixture_file,
        contract_path=contract_file,
    )


def _validate_embedded_artifact_hash(payload: Mapping[str, Any], label: str) -> None:
    supplied = str(payload.get("artifact_hash", ""))
    if not supplied.startswith("sha256:"):
        raise InvalidDeterministicLGSSMTarget(f"{label} artifact hash is missing")
    unhashed = dict(payload)
    unhashed.pop("artifact_hash", None)
    expected = f"sha256:{stable_config_hash(unhashed)}"
    if supplied != expected:
        raise InvalidDeterministicLGSSMTarget(f"{label} artifact hash mismatch")


def _read_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise InvalidDeterministicLGSSMTarget(f"{label} file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise InvalidDeterministicLGSSMTarget(f"{label} must be a JSON mapping")
    return value


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bare_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise InvalidDeterministicLGSSMTarget(
            f"{label} must be a bare lowercase sha256 digest"
        )
    return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        materialized = value.numpy()
        if hasattr(materialized, "tolist"):
            return _json_ready(materialized.tolist())
        if hasattr(materialized, "item"):
            return _json_ready(materialized.item())
        return _json_ready(materialized)
    return value

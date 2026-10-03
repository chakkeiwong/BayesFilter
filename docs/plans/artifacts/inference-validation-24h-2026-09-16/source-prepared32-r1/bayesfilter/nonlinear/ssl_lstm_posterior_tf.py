"""Masked TensorFlow posterior target for the locked scalar SSL-LSTM fixture."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    tf_ssl_lstm_svd_ukf_score,
)


TARGET_SEMANTIC_SHA256 = (
    "549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"
)
PARAMETER_MASK_SHA256 = (
    "9dc25c878760b2fec5b5ad223662912272c2bda1b0d31590e3f60ec11ef79043"
)
MASKED_POSTERIOR_CONTRACT_SHA256 = (
    "004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556"
)
GOLDEN_SIGNATURES_FILE_SHA256 = (
    "04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34"
)
A0_TARGET_LOCK_FILE_SHA256 = (
    "1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383"
)
A0_DEPENDENCY_MANIFEST_FILE_SHA256 = (
    "2a1e3dcd89c0c5e24f892d14a29ef44329ef3e50c4af40093428082de6806517"
)
A0_IMMUTABLE_AGGREGATE_SHA256 = (
    "6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f"
)
A0_SIGNATURE_AGGREGATE_SHA256 = (
    "af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc"
)
FULL_FIXTURE_RAW_SHA256 = (
    "33b0814b86c5875e6746150762b8ae3b655e5bbcaa0bfd8df51488783bcb601f"
)
OBSERVATION_RAW_SHA256 = (
    "aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98"
)
PRIOR_CENTER_RAW_SHA256 = (
    "e46fb6877d89473071047938f170cef5c3d02b2c87ce7f9834d92c4040e16c2f"
)

TARGET_SCOPE = "ssl_lstm_completion:a1:masked_svd_ukf_four_parameter"
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-"
    "2026-07-11.md"
)

NONCLAIMS = (
    "target extraction and frozen-point engineering canary only",
    "not posterior correctness evidence",
    "not HMC or NeuTra readiness evidence",
    "not predictive equivalence or calibration evidence",
    "not target-wide GPU/XLA or performance evidence",
    "not public API, default, product, or release readiness evidence",
    "not a sampler ranking or scientific claim",
)
TESTING_NONCLAIMS = (
    "testing seam only",
    "production signatures unavailable",
    "artifact generation forbidden",
)

FULL_PARAMETER_NAMES = (
    "lstm_input.input.0.0",
    "lstm_input.forget.0.0",
    "lstm_input.output.0.0",
    "lstm_input.candidate.0.0",
    "lstm_recurrent.input.0.0",
    "lstm_recurrent.forget.0.0",
    "lstm_recurrent.output.0.0",
    "lstm_recurrent.candidate.0.0",
    "lstm_bias.input.0",
    "lstm_bias.forget.0",
    "lstm_bias.output.0",
    "lstm_bias.candidate.0",
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
    "initial_mean.0",
    "initial_mean.1",
    "initial_mean.2",
    "initial_std_unconstrained.0",
    "initial_std_unconstrained.1",
    "initial_std_unconstrained.2",
    "process_std_unconstrained.0",
    "observation_std_unconstrained.0",
)
FREE_PARAMETER_NAMES = (
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
)
FREE_INDICES = (12, 13, 14, 15)
FULL_FIXTURE_VALUES = (
    0.09,
    -0.07,
    0.05,
    0.04,
    0.03,
    -0.02,
    0.06,
    -0.05,
    0.01,
    0.04,
    -0.03,
    0.02,
    0.35,
    -0.08,
    0.65,
    0.05,
    0.15,
    -0.10,
    0.20,
    -0.35,
    0.15,
    0.55,
    0.35,
    -0.15,
)
PRIOR_CENTER_VALUES = (0.35, -0.08, 0.65, 0.05)
OBSERVATION_VALUES = (
    0.348783333509205,
    -0.06319427221788393,
    0.938603323083808,
    1.4622688902045144,
    -0.44815739683239364,
    0.22003438506565143,
    0.12802423285807635,
    0.09088861589914976,
    -0.30892992513107187,
    1.2888202099980806,
    -0.5062346637379318,
    0.23141030375951993,
    -0.7398852277577778,
    -1.637711895122823,
    0.7463924366306034,
    -0.015159995809434501,
    0.821621152232911,
    0.943395801287454,
    0.7983413928708691,
    -0.44994456871443267,
    0.5986856559902419,
    0.9655453011734912,
    0.27912167846629843,
    -1.0212217577883904,
    -0.7212056110030903,
    1.675807439349596,
    -1.0454402378094254,
    -0.5329910449431029,
    -1.6360645459528094,
    -0.6635502479829377,
)

_FALLBACK_LOG_PROB = -1.0e100
_VALID_STATUS = 0
_NONFINITE_STATUS = 1


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_float64_sha256(values: tuple[float, ...]) -> str:
    raw = b"".join(struct.pack("<d", float(value)) for value in values)
    return hashlib.sha256(raw).hexdigest()


def _tensor_values(tensor: tf.Tensor, expected_size: int) -> tuple[float, ...]:
    flat = tf.reshape(tensor, [expected_size])
    return tuple(float(value) for value in tf.unstack(flat, num=expected_size))


def _require_tensor(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype != tf.float64:
        raise TypeError(f"{name} must have dtype float64")
    if tensor.shape.rank != len(shape) or tuple(tensor.shape.as_list()) != shape:
        raise ValueError(f"{name} must have static shape {shape}")
    return tensor


def _default_full_fixture() -> tf.Tensor:
    return tf.constant(FULL_FIXTURE_VALUES, dtype=tf.float64)


def _default_observations() -> tf.Tensor:
    return tf.reshape(
        tf.constant(OBSERVATION_VALUES, dtype=tf.float64),
        [30, 1],
    )


def _default_prior_center() -> tf.Tensor:
    return tf.constant(PRIOR_CENTER_VALUES, dtype=tf.float64)


@dataclass(frozen=True)
class SSLLSTMParameterMask:
    """Locked four-coordinate view of the full scalar SSL-LSTM chart."""

    full_parameter_names: tuple[str, ...] = FULL_PARAMETER_NAMES
    free_parameter_names: tuple[str, ...] = FREE_PARAMETER_NAMES
    free_indices: tuple[int, ...] = FREE_INDICES
    full_values: tf.Tensor = field(default_factory=_default_full_fixture)

    def __post_init__(self) -> None:
        full_names = tuple(str(name) for name in self.full_parameter_names)
        free_names = tuple(str(name) for name in self.free_parameter_names)
        free_indices = tuple(int(index) for index in self.free_indices)
        if len(full_names) != len(set(full_names)):
            raise ValueError("full parameter names must be unique")
        if len(free_names) != len(set(free_names)):
            raise ValueError("free parameter names must be unique")
        if len(free_indices) != len(set(free_indices)):
            raise ValueError("free parameter indices must be unique")
        if full_names != FULL_PARAMETER_NAMES:
            raise ValueError("full parameter chart does not match the A0 lock")
        if free_names != FREE_PARAMETER_NAMES or free_indices != FREE_INDICES:
            raise ValueError("free mask does not match the locked coordinate order")
        if any(index < 0 or index >= len(full_names) for index in free_indices):
            raise ValueError("free parameter index is out of range")
        if tuple(full_names[index] for index in free_indices) != free_names:
            raise ValueError("free parameter names and indices disagree")
        full_values = _require_tensor(
            self.full_values,
            shape=(24,),
            name="full_values",
        )
        values = _tensor_values(full_values, 24)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("full_values must be finite")
        if values != FULL_FIXTURE_VALUES:
            raise ValueError("full_values do not match the A0 fixture")
        if _raw_float64_sha256(values) != FULL_FIXTURE_RAW_SHA256:
            raise ValueError("full_values raw SHA-256 does not match A0")
        object.__setattr__(self, "full_parameter_names", full_names)
        object.__setattr__(self, "free_parameter_names", free_names)
        object.__setattr__(self, "free_indices", free_indices)
        object.__setattr__(self, "full_values", full_values)
        if self.signature() != PARAMETER_MASK_SHA256:
            raise ValueError("parameter mask signature does not match the golden contract")

    @property
    def full_dimension(self) -> int:
        return 24

    @property
    def free_dimension(self) -> int:
        return 4

    def embed(self, free: Any) -> tf.Tensor:
        values = _require_tensor(free, shape=(4,), name="free")
        indices = tf.constant([[index] for index in self.free_indices], dtype=tf.int32)
        return tf.tensor_scatter_nd_update(self.full_values, indices, values)

    def extract(self, full: Any) -> tf.Tensor:
        values = _require_tensor(full, shape=(24,), name="full")
        return tf.gather(values, self.free_indices)

    def signature_payload(self) -> dict[str, Any]:
        free_set = frozenset(self.free_indices)
        values = _tensor_values(self.full_values, 24)
        return {
            "dtype": "float64",
            "fixed_parameters": [
                {
                    "index": index,
                    "name": self.full_parameter_names[index],
                    "value_hex": float(values[index]).hex(),
                }
                for index in range(self.full_dimension)
                if index not in free_set
            ],
            "free_dimension": self.free_dimension,
            "free_indices": list(self.free_indices),
            "free_parameter_names": list(self.free_parameter_names),
            "full_dimension": self.full_dimension,
            "full_fixture_raw_sha256": FULL_FIXTURE_RAW_SHA256,
            "full_parameter_names": list(self.full_parameter_names),
            "schema_version": "bayesfilter.ssl_lstm_completion.parameter_mask.v1",
        }

    def signature(self) -> str:
        return _canonical_sha256(self.signature_payload())


@dataclass(frozen=True)
class SSLLSTMPosteriorConfig:
    """Configuration for the exact A0-locked masked posterior target."""

    static_config: SSLLSTMStaticConfig = field(
        default_factory=lambda: SSLLSTMStaticConfig(
            horizon=30,
            latent_dim=1,
            hidden_dim=1,
            observation_dim=1,
            covariance_mode="diagonal",
        )
    )
    parameter_mask: SSLLSTMParameterMask = field(default_factory=SSLLSTMParameterMask)
    observations: tf.Tensor = field(default_factory=_default_observations)
    prior_center: tf.Tensor = field(default_factory=_default_prior_center)
    prior_standard_deviation: float = 4.0
    prior_normalized: bool = False
    filter_name: str = "svd_ukf"
    std_floor: float = 1.0e-4
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0
    placement_floor: float = 0.0
    innovation_floor: float = 1.0e-12
    rank_tolerance: float = 1.0e-12
    spectral_gap_tolerance: float = 1.0e-10
    fixed_null_tolerance: float = 1.0e-10
    jitter: float = 0.0
    allow_fixed_null_support: bool = False
    return_filtered: bool = False
    jit_compile: bool = True
    execution_role: str = "default_xla"
    backend: str = "tensorflow"
    dtype: str = "float64"
    target_scope: str = TARGET_SCOPE

    def __post_init__(self) -> None:
        config = self.static_config
        if not isinstance(config, SSLLSTMStaticConfig):
            raise TypeError("static_config must be an SSLLSTMStaticConfig")
        expected_static = (30, 1, 1, 1, "diagonal", 24, 3)
        actual_static = (
            config.horizon,
            config.latent_dim,
            config.hidden_dim,
            config.observation_dim,
            config.covariance_mode,
            config.parameter_dim,
            config.augmented_state_dim,
        )
        if actual_static != expected_static or config.parameter_names != FULL_PARAMETER_NAMES:
            raise ValueError("static_config does not match the A0 scalar target")
        if not isinstance(self.parameter_mask, SSLLSTMParameterMask):
            raise TypeError("parameter_mask must be an SSLLSTMParameterMask")
        observations = _require_tensor(
            self.observations,
            shape=(30, 1),
            name="observations",
        )
        observation_values = _tensor_values(observations, 30)
        if any(not math.isfinite(value) for value in observation_values):
            raise ValueError("observations must be finite")
        if observation_values != OBSERVATION_VALUES:
            raise ValueError("observations do not match the A0 lock")
        if _raw_float64_sha256(observation_values) != OBSERVATION_RAW_SHA256:
            raise ValueError("observation raw SHA-256 does not match A0")
        prior_center = _require_tensor(
            self.prior_center,
            shape=(4,),
            name="prior_center",
        )
        prior_values = _tensor_values(prior_center, 4)
        if prior_values != PRIOR_CENTER_VALUES:
            raise ValueError("prior center does not match the A0 lock")
        if _raw_float64_sha256(prior_values) != PRIOR_CENTER_RAW_SHA256:
            raise ValueError("prior center raw SHA-256 does not match A0")
        exact_values = {
            "prior_standard_deviation": (self.prior_standard_deviation, 4.0),
            "std_floor": (self.std_floor, 1.0e-4),
            "alpha": (self.alpha, 1.0),
            "beta": (self.beta, 2.0),
            "kappa": (self.kappa, 0.0),
            "placement_floor": (self.placement_floor, 0.0),
            "innovation_floor": (self.innovation_floor, 1.0e-12),
            "rank_tolerance": (self.rank_tolerance, 1.0e-12),
            "spectral_gap_tolerance": (self.spectral_gap_tolerance, 1.0e-10),
            "fixed_null_tolerance": (self.fixed_null_tolerance, 1.0e-10),
            "jitter": (self.jitter, 0.0),
        }
        for name, (actual, expected) in exact_values.items():
            if not math.isfinite(float(actual)) or float(actual) != expected:
                raise ValueError(f"{name} does not match the A0 lock")
        if self.prior_normalized:
            raise ValueError("the A0 prior is unnormalized")
        if self.filter_name != "svd_ukf":
            raise ValueError("only the historical SVD-UKF route is allowed")
        if self.allow_fixed_null_support or self.return_filtered:
            raise ValueError("filter branch settings do not match A0")
        if self.backend != "tensorflow" or self.dtype != "float64":
            raise ValueError("backend and dtype must be TensorFlow float64")
        if self.target_scope != TARGET_SCOPE:
            raise ValueError("target_scope does not match the A1 contract")
        role = str(self.execution_role)
        if bool(self.jit_compile):
            if role != "default_xla":
                raise ValueError("compiled execution must use the default_xla role")
        elif role != "eager_debug_reference":
            raise ValueError("noncompiled execution is an eager debug reference only")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "prior_center", prior_center)
        object.__setattr__(self, "jit_compile", bool(self.jit_compile))
        object.__setattr__(self, "execution_role", role)
        if self.signature() != MASKED_POSTERIOR_CONTRACT_SHA256:
            raise ValueError("posterior config signature does not match the golden contract")

    def signature_payload(self) -> dict[str, Any]:
        capability = _production_capability_payload()
        return {
            "a0_bindings": {
                "dependency_manifest_file_sha256": A0_DEPENDENCY_MANIFEST_FILE_SHA256,
                "immutable_aggregate_sha256": A0_IMMUTABLE_AGGREGATE_SHA256,
                "signature_aggregate_sha256": A0_SIGNATURE_AGGREGATE_SHA256,
                "target_lock_file_sha256": A0_TARGET_LOCK_FILE_SHA256,
                "target_semantic_sha256": TARGET_SEMANTIC_SHA256,
            },
            "adapter_contract": {
                "adapter_signature_role": "masked_posterior_contract_sha256",
                "manifest_parameter_dim": 4,
                "manifest_parameter_names": list(FREE_PARAMETER_NAMES),
                "target_signature_role": "a0_target_semantic_sha256",
                "value_score_capability": capability,
            },
            "callable_contract": _callable_contract_payload(),
            "classification": "extension_or_invention_preserving_a0_historical_estimand",
            "execution": {
                "backend": "tensorflow",
                "batch_static_sizes_tested": [1, 4, 10],
                "batching_route": "static_tensorflow_unroll_or_reviewed_batch_native_no_python_loop",
                "default_compile_mode": "xla",
                "default_jit_compile": True,
                "dtype": "float64",
                "dynamic_batch_allowed": False,
                "eager_debug_available": True,
                "filter_autodiff_forbidden": True,
                "scalar_input_signature": "float64[4]",
                "seed_policy": "not_used",
            },
            "filter": {
                "allow_fixed_null_support": False,
                "alpha_hex": float(self.alpha).hex(),
                "beta_hex": float(self.beta).hex(),
                "finite_filter_failure": "loud_error",
                "fixed_null_tolerance_hex": float(self.fixed_null_tolerance).hex(),
                "innovation_floor_hex": float(self.innovation_floor).hex(),
                "jitter_hex": float(self.jitter).hex(),
                "kappa_hex": float(self.kappa).hex(),
                "name": "svd_ukf_filtering_log_likelihood",
                "placement_floor_hex": float(self.placement_floor).hex(),
                "rank_tolerance_hex": float(self.rank_tolerance).hex(),
                "return_filtered": False,
                "score_authority": "analytic_eigenderivative",
                "score_helper": "tf_ssl_lstm_svd_ukf_score",
                "spectral_gap_tolerance_hex": float(self.spectral_gap_tolerance).hex(),
                "std_floor_hex": float(self.std_floor).hex(),
            },
            "full_fixture": {
                "dtype": "float64",
                "raw_sha256": FULL_FIXTURE_RAW_SHA256,
                "shape": [24],
            },
            "nonclaims": list(NONCLAIMS),
            "nonfinite_input_reject": {
                "fallback_log_prob_hex": float(_FALLBACK_LOG_PROB).hex(),
                "fallback_score_hex": float(0.0).hex(),
                "finite_filter_failure": "loud_error",
                "schema_version": "bayesfilter.ssl_lstm_completion.nonfinite_input_reject.v1",
                "scope": "nonfinite_input_only",
                "status_codes": {
                    "nonfinite_input_reject": _NONFINITE_STATUS,
                    "valid_finite": _VALID_STATUS,
                },
            },
            "observations": {
                "dtype": "float64",
                "mask_convention": "none",
                "missingness_convention": "none",
                "raw_sha256": OBSERVATION_RAW_SHA256,
                "shape": [30, 1],
            },
            "parameter_mask_sha256": self.parameter_mask.signature(),
            "parameter_transform": {
                "inverse_orientation": "identity",
                "log_det_jacobian_convention": "not_included",
                "orientation": "identity",
                "transform_source": "a0_locked_free_coordinates",
            },
            "prior": {
                "center_hex": [float(value).hex() for value in PRIOR_CENTER_VALUES],
                "center_raw_sha256": PRIOR_CENTER_RAW_SHA256,
                "family": "unnormalized_isotropic_gaussian_log_kernel",
                "log_kernel_formula": "-0.5 * sum((free - truth_free)^2 / 4.0^2)",
                "normalized": False,
                "standard_deviation_hex": float(self.prior_standard_deviation).hex(),
            },
            "schema_version": "bayesfilter.ssl_lstm_completion.masked_posterior_contract.v1",
            "static_config": {
                "augmented_state_dim": 3,
                "covariance_mode": "diagonal",
                "full_parameter_dim": 24,
                "hidden_dim": 1,
                "horizon": 30,
                "latent_dim": 1,
                "observation_dim": 1,
            },
            "target_scope": TARGET_SCOPE,
            "testing_only_injection": {
                "adapter_signature_behavior": "raise_runtime_error",
                "artifact_generation": "forbidden",
                "constructor_flag": "testing_only",
                "finite_branch_argument": "finite_branch_callable",
                "production_default": "real_filter",
                "target_signature_behavior": "raise_runtime_error",
                "value_score_capability": _testing_capability_payload(),
            },
        }

    def signature(self) -> str:
        return _canonical_sha256(self.signature_payload())


class SSLLSTMPosteriorTarget:
    """Graph-native value/score adapter for the locked four-dimensional target."""

    def __init__(
        self,
        config: SSLLSTMPosteriorConfig | None = None,
        *,
        finite_branch_callable: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]] | None = None,
        testing_only: bool = False,
    ) -> None:
        self.config = SSLLSTMPosteriorConfig() if config is None else config
        if not isinstance(self.config, SSLLSTMPosteriorConfig):
            raise TypeError("config must be an SSLLSTMPosteriorConfig")
        self.testing_only = bool(testing_only)
        if finite_branch_callable is not None and not self.testing_only:
            raise ValueError("a nondefault finite branch requires testing_only=True")
        if self.testing_only and finite_branch_callable is None:
            raise ValueError("testing_only targets require a finite_branch_callable")
        self._finite_branch_callable = (
            self._production_finite_value_and_score
            if finite_branch_callable is None
            else finite_branch_callable
        )
        self._compiled_scalar = tf.function(
            self._diagnostic_scalar_impl,
            input_signature=[tf.TensorSpec([4], tf.float64)],
            jit_compile=self.config.jit_compile,
            reduce_retracing=True,
        )
        self._compiled_batches: dict[int, Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]]] = {}
        self._log_prob_target = reviewed_value_score_target_fn(self, dtype=tf.float64)

    @property
    def parameter_dim(self) -> int:
        return 4

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return FREE_PARAMETER_NAMES

    @property
    def target_scope(self) -> str:
        return TARGET_SCOPE

    def full_theta(self, free: Any) -> tf.Tensor:
        return self.config.parameter_mask.embed(free)

    def free_theta(self, full: Any) -> tf.Tensor:
        return self.config.parameter_mask.extract(full)

    def value_and_score(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _require_tensor(free, shape=(4,), name="free")
        value, score, _status = self._compiled_scalar(values)
        return value, score

    def log_prob_and_grad(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.value_and_score(free)

    def value(self, free: Any) -> tf.Tensor:
        value, _score = self.value_and_score(free)
        return value

    def score(self, free: Any) -> tf.Tensor:
        _value, score = self.value_and_score(free)
        return score

    def log_prob(self, free: Any) -> tf.Tensor:
        values = _require_tensor(free, shape=(4,), name="free")
        return self._log_prob_target(values)

    def eager_debug_value_and_score(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _require_tensor(free, shape=(4,), name="free")
        value, score, _status = self._diagnostic_scalar_impl(values)
        return value, score

    def diagnostic_value_and_score(
        self,
        free: Any,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(free)
        if values.dtype != tf.float64:
            raise TypeError("free must have dtype float64")
        if values.shape.rank == 1:
            values = _require_tensor(values, shape=(4,), name="free")
            return self._compiled_scalar(values)
        if values.shape.rank == 2:
            batch_size = values.shape[0]
            if batch_size is None or int(batch_size) <= 0:
                raise ValueError("free batch dimension must be static and positive")
            values = _require_tensor(
                values,
                shape=(int(batch_size), 4),
                name="free",
            )
            return self._compiled_batch(int(batch_size))(values)
        raise ValueError("free must have rank 1 or rank 2")

    def batch_value_and_score(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(free)
        if tensor.dtype != tf.float64:
            raise TypeError("free must have dtype float64")
        if tensor.shape.rank != 2:
            raise ValueError("batch_value_and_score requires rank 2 input")
        batch_size = tensor.shape[0]
        if batch_size is None or int(batch_size) <= 0:
            raise ValueError("free batch dimension must be static and positive")
        tensor = _require_tensor(
            tensor,
            shape=(int(batch_size), 4),
            name="free",
        )
        values, scores, _statuses = self._compiled_batch(int(batch_size))(tensor)
        return values, scores

    def adapter_signature(self) -> str:
        self.assert_production_evidence_target()
        return MASKED_POSTERIOR_CONTRACT_SHA256

    def target_signature(self) -> str:
        self.assert_production_evidence_target()
        return TARGET_SEMANTIC_SHA256

    def adapter_manifest_payload(self) -> dict[str, Any]:
        self.assert_production_evidence_target()
        return {
            "adapter_signature": self.adapter_signature(),
            "parameter_dim": self.parameter_dim,
            "parameter_names": list(self.parameter_names),
            "target_signature": self.target_signature(),
            "target_scope": self.target_scope,
            "value_score_capability": _production_capability_payload(),
        }

    def value_score_capability(self) -> ValueScoreCapability:
        payload = (
            _testing_capability_payload()
            if self.testing_only
            else _production_capability_payload()
        )
        return ValueScoreCapability(
            value_score_authority=payload["value_score_authority"],
            xla_hmc_ready=payload["xla_hmc_ready"],
            full_chain_xla_diagnostic_ready=payload[
                "full_chain_xla_diagnostic_ready"
            ],
            runtime_backend=payload["runtime_backend"],
            evidence_path=payload["evidence_path"],
            target_scope=payload["target_scope"],
            nonclaims=tuple(payload["nonclaims"]),
        )

    def assert_production_evidence_target(self) -> None:
        if self.testing_only:
            raise RuntimeError("testing-only targets cannot publish production evidence")
        if not self.config.jit_compile or self.config.execution_role != "default_xla":
            raise RuntimeError("non-XLA debug targets cannot publish production evidence")

    def compiled_scalar_trace_count(self) -> int:
        return int(self._compiled_scalar.experimental_get_tracing_count())

    def compiled_batch_sizes(self) -> tuple[int, ...]:
        return tuple(sorted(self._compiled_batches))

    def _compiled_batch(
        self,
        batch_size: int,
    ) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        compiled = self._compiled_batches.get(batch_size)
        if compiled is None:
            def batch_program(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                return tf.map_fn(
                    self._diagnostic_scalar_impl,
                    values,
                    fn_output_signature=(
                        tf.TensorSpec([], tf.float64),
                        tf.TensorSpec([4], tf.float64),
                        tf.TensorSpec([], tf.int32),
                    ),
                    parallel_iterations=1,
                )

            compiled = tf.function(
                batch_program,
                input_signature=[tf.TensorSpec([batch_size, 4], tf.float64)],
                jit_compile=self.config.jit_compile,
                reduce_retracing=True,
            )
            self._compiled_batches[batch_size] = compiled
        return compiled

    def _diagnostic_scalar_impl(
        self,
        free: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        finite = tf.reduce_all(tf.math.is_finite(free))

        def finite_branch() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            value, score = self._finite_branch_callable(free)
            value = tf.convert_to_tensor(value)
            score = tf.convert_to_tensor(score)
            if value.dtype != tf.float64 or score.dtype != tf.float64:
                raise TypeError("finite target outputs must have dtype float64")
            value = tf.ensure_shape(value, [])
            score = tf.ensure_shape(score, [4])
            value = tf.debugging.check_numerics(value, "finite target value")
            score = tf.debugging.check_numerics(score, "finite target score")
            return value, score, tf.constant(_VALID_STATUS, dtype=tf.int32)

        def reject_branch() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            return (
                tf.constant(_FALLBACK_LOG_PROB, dtype=tf.float64),
                tf.zeros([4], dtype=tf.float64),
                tf.constant(_NONFINITE_STATUS, dtype=tf.int32),
            )

        return tf.cond(finite, finite_branch, reject_branch)

    def _production_finite_value_and_score(
        self,
        free: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        full = self.config.parameter_mask.embed(free)
        result, _components = tf_ssl_lstm_svd_ukf_score(
            self.config.observations,
            full,
            self.config.static_config,
            evidence_path=RESULT_PATH,
            std_floor=self.config.std_floor,
            alpha=self.config.alpha,
            beta=self.config.beta,
            kappa=self.config.kappa,
            spectral_gap_tolerance=tf.constant(
                self.config.spectral_gap_tolerance,
                dtype=tf.float64,
            ),
        )
        full_score = tf.ensure_shape(
            tf.convert_to_tensor(result.score, dtype=tf.float64),
            [24],
        )
        delta = free - self.config.prior_center
        variance = tf.constant(
            self.config.prior_standard_deviation**2,
            dtype=tf.float64,
        )
        prior_value = -0.5 * tf.reduce_sum(tf.square(delta) / variance)
        prior_score = -delta / variance
        value = tf.convert_to_tensor(result.log_likelihood, dtype=tf.float64) + prior_value
        score = tf.gather(full_score, FREE_INDICES) + prior_score
        return tf.ensure_shape(value, []), tf.ensure_shape(score, [4])


def locked_ssl_lstm_posterior_target() -> SSLLSTMPosteriorTarget:
    """Build the production A1 target with XLA compilation enabled by default."""

    return SSLLSTMPosteriorTarget()


def _production_capability_payload() -> dict[str, Any]:
    return {
        "evidence_path": RESULT_PATH,
        "full_chain_xla_diagnostic_ready": False,
        "nonclaims": list(NONCLAIMS),
        "runtime_backend": "bayesfilter.nonlinear.ssl_lstm_posterior_tf",
        "target_scope": TARGET_SCOPE,
        "value_score_authority": "graph_native",
        "xla_hmc_ready": False,
    }


def _testing_capability_payload() -> dict[str, Any]:
    return {
        "evidence_path": "tests/test_ssl_lstm_posterior_tf.py",
        "full_chain_xla_diagnostic_ready": False,
        "nonclaims": list(TESTING_NONCLAIMS),
        "runtime_backend": "tensorflow_testing_only_injected_finite_branch",
        "target_scope": "ssl_lstm_completion:a1:testing_only_injected_branch",
        "value_score_authority": "debug_only",
        "xla_hmc_ready": False,
    }


def _callable_contract_payload() -> dict[str, Any]:
    return {
        "alias_equalities": [
            "value==log_prob==value_and_score[0]==log_prob_and_grad[0]",
            "score==value_and_score[1]==log_prob_and_grad[1]",
        ],
        "invalid_dtype": "loud_error",
        "invalid_rank_or_shape": "loud_error",
        "parameter_dim": 4,
        "parameter_names": list(FREE_PARAMETER_NAMES),
        "surfaces": [
            {
                "default_execution": "xla",
                "input_shape": [4],
                "name": "value_and_score",
                "score_shape": [4],
                "value_shape": [],
            },
            {
                "default_execution": "xla",
                "input_shape": [4],
                "name": "log_prob_and_grad",
                "score_shape": [4],
                "value_shape": [],
            },
            {
                "default_execution": "xla",
                "input_shape": [4],
                "name": "value",
                "output_shape": [],
            },
            {
                "default_execution": "xla",
                "input_shape": [4],
                "name": "score",
                "output_shape": [4],
            },
            {
                "default_execution": "xla_custom_gradient",
                "input_shape": [4],
                "name": "log_prob",
                "output_shape": [],
            },
            {
                "default_execution": "xla_cached_per_static_B",
                "input_shape": ["B", 4],
                "name": "batch_value_and_score",
                "score_shape": ["B", 4],
                "value_shape": ["B"],
            },
            {
                "default_execution": "eager_debug_only",
                "input_shape": [4],
                "name": "eager_debug_value_and_score",
                "score_shape": [4],
                "value_shape": [],
            },
            {
                "default_execution": "same_target_branch",
                "input_shape": "scalar_or_supported_static_batch",
                "name": "diagnostic_value_and_score",
                "score_shape": "[4]_or_[B,4]",
                "status_shape": "[]_or_[B]",
                "value_shape": "scalar_or_batch",
            },
        ],
    }


__all__ = [
    "MASKED_POSTERIOR_CONTRACT_SHA256",
    "PARAMETER_MASK_SHA256",
    "SSLLSTMParameterMask",
    "SSLLSTMPosteriorConfig",
    "SSLLSTMPosteriorTarget",
    "TARGET_SEMANTIC_SHA256",
    "locked_ssl_lstm_posterior_target",
]

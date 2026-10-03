"""Typed predictive paths for the locked scalar SSL-LSTM target."""

from __future__ import annotations

import hashlib
import json
import operator
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import tensorflow as tf

from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (
    FULL_FIXTURE_RAW_SHA256,
    MASKED_POSTERIOR_CONTRACT_SHA256,
    OBSERVATION_RAW_SHA256,
    PARAMETER_MASK_SHA256,
    PRIOR_CENTER_RAW_SHA256,
    TARGET_SEMANTIC_SHA256,
    SSLLSTMPosteriorConfig,
    SSLLSTMPosteriorTarget,
)
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    make_ssl_lstm_svd_ukf_components,
    ssl_lstm_observation,
    ssl_lstm_transition,
    tf_ssl_lstm_svd_ukf_score,
)


A2_CONTRACT_SIGNATURE = (
    "8719aa65943dcc9e4b0499debfff8ec13a96d4cec12dc48d70a8922920058804"
)
A2_EVIDENCE_FORECAST_CONFIG_SIGNATURE = (
    "ecb5a2cedac5f059da3bd3feee51a1065eb66aeff5aeb8dc0dd3b4e3a6926150"
)
A1_RESULT_FILE_SHA256 = (
    "78f269a53fb0536017d32bd12c2b36967cd013a85dcb1102936ed79ae95e34b5"
)
A2_SUBPLAN_FILE_SHA256 = (
    "6b6b9799782be3304ecbd2dee465c52285688b5e2d1b3087d911ccad1279bbb0"
)
A2_RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-"
    "result-2026-07-11.md"
)

FORECAST_HORIZON = 10
STATE_DIM = 3
LATENT_DIM = 1
OBSERVATION_DIM = 1
FLOAT64_EPSILON = 2.0**-52
COVARIANCE_ROUNDOFF_MULTIPLIER = 64

ROLE_CODES = {
    "paired_diagnostic_shared": 101,
    "independent_arm": 211,
}
FAMILY_CODES = {
    "terminal": 1001,
    "process": 1002,
    "observation": 1003,
}

STATUS_VALID = 0
STATUS_NONFINITE = 1
STATUS_ASYMMETRIC = 2
STATUS_MATERIALLY_INDEFINITE = 4
STATUS_PROJECTION = 8
STATUS_FACTOR_RECONSTRUCTION = 16
STATUS_FILTER_PARITY = 32
STATUS_TOTAL_PARITY = 64

NONCLAIMS = (
    "A2 terminal-state and forecast engineering evidence only",
    "predictive law is conditional on the approximate historical SVD-UKF",
    "not posterior correctness or exact nonlinear filtering evidence",
    "not HMC or NeuTra readiness evidence",
    "not predictive equivalence, calibration, or model adequacy evidence",
    "not performance, product, public API, default, or release evidence",
    "not a sampler ranking or scientific claim",
)

InnovationRole = Literal["paired_diagnostic_shared", "independent_arm"]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_float64_sha256(tensor: tf.Tensor) -> str:
    values = tf.reshape(tf.convert_to_tensor(tensor, tf.float64), [-1])
    raw = b"".join(struct.pack("<d", float(value)) for value in tf.unstack(values))
    return hashlib.sha256(raw).hexdigest()


def _require_float64_tensor(
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


def _require_all_finite(tensor: tf.Tensor, *, name: str) -> None:
    if not bool(tf.reduce_all(tf.math.is_finite(tensor))):
        raise ValueError(f"{name} must contain only finite values")


def _require_seed(value: Any) -> tf.Tensor:
    seed = tf.convert_to_tensor(value)
    if seed.dtype != tf.int32:
        raise TypeError("seed must have dtype int32")
    if seed.shape != (2,):
        raise ValueError("seed must have static shape (2,)")
    return seed


def _scale_tolerance(multiplier: int, left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    scale = tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.maximum(tf.reduce_max(tf.abs(left)), tf.reduce_max(tf.abs(right))),
    )
    return tf.constant(float(multiplier) * FLOAT64_EPSILON, tf.float64) * scale


def _default_posterior_config() -> SSLLSTMPosteriorConfig:
    return SSLLSTMPosteriorConfig()


@dataclass(frozen=True)
class SSLLSTMForecastConfig:
    """Static A2 forecast configuration."""

    posterior_config: SSLLSTMPosteriorConfig = field(default_factory=_default_posterior_config)
    forecast_horizon: int = FORECAST_HORIZON
    replication_count: int = 2
    jit_compile: bool = True
    execution_role: str = "default_xla"
    covariance_roundoff_multiplier: int = COVARIANCE_ROUNDOFF_MULTIPLIER

    def __post_init__(self) -> None:
        if not isinstance(self.posterior_config, SSLLSTMPosteriorConfig):
            raise TypeError("posterior_config must be SSLLSTMPosteriorConfig")
        if self.posterior_config.signature() != MASKED_POSTERIOR_CONTRACT_SHA256:
            raise ValueError("posterior config does not match the accepted A1 contract")
        if isinstance(self.forecast_horizon, bool):
            raise TypeError("forecast_horizon must be an integer")
        try:
            horizon = operator.index(self.forecast_horizon)
        except TypeError as exc:
            raise TypeError("forecast_horizon must be an integer") from exc
        if horizon != FORECAST_HORIZON:
            raise ValueError("A2 forecast_horizon must equal 10")
        replications = _require_static_positive_int(
            self.replication_count,
            name="replication_count",
        )
        if not isinstance(self.jit_compile, bool):
            raise TypeError("jit_compile must be bool")
        role = str(self.execution_role)
        if self.jit_compile:
            if role != "default_xla":
                raise ValueError("compiled forecasts require execution_role=default_xla")
        elif role != "eager_debug_reference":
            raise ValueError("noncompiled forecasts are eager_debug_reference only")
        if isinstance(self.covariance_roundoff_multiplier, bool):
            raise TypeError("covariance_roundoff_multiplier must be an integer")
        try:
            covariance_multiplier = operator.index(
                self.covariance_roundoff_multiplier
            )
        except TypeError as exc:
            raise TypeError(
                "covariance_roundoff_multiplier must be an integer"
            ) from exc
        if covariance_multiplier != COVARIANCE_ROUNDOFF_MULTIPLIER:
            raise ValueError("covariance roundoff multiplier is frozen at 64")
        object.__setattr__(self, "forecast_horizon", FORECAST_HORIZON)
        object.__setattr__(self, "replication_count", replications)
        object.__setattr__(self, "jit_compile", self.jit_compile)
        object.__setattr__(self, "execution_role", role)

    def signature_payload(self) -> dict[str, Any]:
        return {
            "a1_posterior_config_signature": self.posterior_config.signature(),
            "a2_contract_signature": A2_CONTRACT_SIGNATURE,
            "allowed_innovation_roles": [
                "paired_diagnostic_shared",
                "independent_arm",
            ],
            "backend": "tensorflow",
            "covariance_roundoff_multiplier": self.covariance_roundoff_multiplier,
            "dtype": "float64",
            "execution_role": self.execution_role,
            "forecast_horizon": self.forecast_horizon,
            "jit_compile": self.jit_compile,
            "latent_dim": LATENT_DIM,
            "observation_dim": OBSERVATION_DIM,
            "replication_count": self.replication_count,
            "schema_version": (
                "bayesfilter.ssl_lstm_completion.phase_a2_forecast_config.v1"
            ),
            "state_dim": STATE_DIM,
        }

    def signature(self) -> str:
        return _canonical_sha256(self.signature_payload())

    def assert_evidence_config(self) -> None:
        if self.signature() != A2_EVIDENCE_FORECAST_CONFIG_SIGNATURE:
            raise RuntimeError("forecast config is not the reviewed A2 evidence config")


@dataclass(frozen=True)
class SSLLSTMTerminalState:
    """Terminal filtered Gaussian and its fail-closed diagnostics."""

    mean: tf.Tensor
    raw_covariance: tf.Tensor
    symmetrized_covariance: tf.Tensor
    implemented_covariance: tf.Tensor
    factor: tf.Tensor
    raw_eigenvalues: tf.Tensor
    clipped_eigenvalues: tf.Tensor
    minimum_eigenvalue: tf.Tensor
    psd_tolerance: tf.Tensor
    symmetry_residual: tf.Tensor
    projection_residual: tf.Tensor
    factor_reconstruction_residual: tf.Tensor
    filter_log_likelihood: tf.Tensor
    a1_filter_log_likelihood: tf.Tensor
    target_value: tf.Tensor
    total_value: tf.Tensor
    filter_parity_residual: tf.Tensor
    total_parity_residual: tf.Tensor
    filter_parity_tolerance: tf.Tensor
    total_parity_tolerance: tf.Tensor
    full_parameters: tf.Tensor
    status: tf.Tensor


@dataclass(frozen=True)
class SSLLSTMInnovationBank:
    """Materialized standard-normal innovations for replayable forecasts."""

    terminal_standard_normal: tf.Tensor
    process_standard_normal: tf.Tensor
    observation_standard_normal: tf.Tensor
    root_seed: tf.Tensor
    algorithm: Literal["philox"]
    role: InnovationRole
    role_code: int
    arm_id: int
    derived_seeds: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    content_signature: str

    @property
    def draw_count(self) -> int:
        return int(self.terminal_standard_normal.shape[0])

    @property
    def replication_count(self) -> int:
        return int(self.terminal_standard_normal.shape[1])

    def tensor_hashes(self) -> dict[str, str]:
        return {
            "terminal": _raw_float64_sha256(self.terminal_standard_normal),
            "process": _raw_float64_sha256(self.process_standard_normal),
            "observation": _raw_float64_sha256(self.observation_standard_normal),
        }


@dataclass(frozen=True)
class SSLLSTMForecastProvenance:
    """Host-side replay and approximation provenance."""

    schema_version: str
    a2_contract_signature: str
    a1_result_file_sha256: str
    a2_subplan_file_sha256: str
    forecast_config_signature: str
    innovation_bank_signature: str
    free_draw_matrix_raw_sha256: str
    embedded_full_parameter_matrix_raw_sha256: str
    target_semantic_sha256: str
    a1_adapter_signature: str
    parameter_mask_sha256: str
    observation_raw_sha256: str
    full_fixture_raw_sha256: str
    prior_center_raw_sha256: str
    innovation_role: str
    innovation_role_code: int
    innovation_arm_id: int
    innovation_algorithm: str
    innovation_root_seed: tuple[int, int]
    innovation_family_codes: tuple[tuple[str, int], ...]
    innovation_tensor_hashes: tuple[tuple[str, str], ...]
    innovation_replay_authority: str
    innovation_seed_qualification: str
    draw_count: int
    draw_chunk_size: int
    replication_count: int
    forecast_horizon: int
    filter_backend: str
    filter_settings_hex: tuple[tuple[str, str], ...]
    covariance_roundoff_multiplier: int
    terminal_covariance_statuses: tuple[int, ...]
    tensorflow_version: str
    dtype: str
    jit_compile: bool
    execution_role: str
    physical_devices: tuple[tuple[str, str], ...]
    logical_devices: tuple[tuple[str, str], ...]
    output_devices: tuple[str, ...]
    tf32_enabled: bool
    trust_basis: str
    horizon_convention: str
    cluster_unit: str
    approximation_qualification: str
    nonclaims: tuple[str, ...]


@dataclass(frozen=True)
class SSLLSTMForecastPaths:
    """Complete ten-step paths for all draws and replications."""

    terminal_states: tf.Tensor
    states: tf.Tensor
    deterministic_transition_means: tf.Tensor
    process_innovations: tf.Tensor
    observation_means: tf.Tensor
    observation_innovations: tf.Tensor
    observations: tf.Tensor
    terminal: SSLLSTMTerminalState
    provenance: SSLLSTMForecastProvenance


def _require_static_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return int(resolved)


def _seed_values(seed: tf.Tensor) -> tuple[int, int]:
    values = tuple(int(value) for value in tf.unstack(seed, num=2))
    return values[0], values[1]


def _validate_innovation_bank(
    bank: SSLLSTMInnovationBank,
    *,
    draw_count: int,
    config: SSLLSTMForecastConfig,
) -> None:
    if not isinstance(bank, SSLLSTMInnovationBank):
        raise TypeError("innovation_bank must be SSLLSTMInnovationBank")
    expected = (
        (draw_count, config.replication_count, STATE_DIM),
        (draw_count, config.replication_count, FORECAST_HORIZON, LATENT_DIM),
        (draw_count, config.replication_count, FORECAST_HORIZON, OBSERVATION_DIM),
    )
    for name, tensor, shape in zip(
        ("terminal", "process", "observation"),
        (
            bank.terminal_standard_normal,
            bank.process_standard_normal,
            bank.observation_standard_normal,
        ),
        expected,
        strict=True,
    ):
        _require_float64_tensor(tensor, shape=shape, name=f"{name}_standard_normal")
        _require_all_finite(tensor, name=f"{name}_standard_normal")
    if bank.role not in ROLE_CODES or bank.role_code != ROLE_CODES[bank.role]:
        raise ValueError("innovation role metadata is invalid")
    if bank.algorithm != "philox":
        raise ValueError("innovation algorithm must be philox")
    _require_seed(bank.root_seed)
    if bank.role == "paired_diagnostic_shared" and bank.arm_id != 0:
        raise ValueError("paired_diagnostic_shared requires arm_id=0")
    if bank.role == "independent_arm" and bank.arm_id <= 0:
        raise ValueError("independent_arm requires a positive arm_id")
    role_seed = tf.random.experimental.stateless_fold_in(
        bank.root_seed,
        tf.constant(bank.role_code, tf.int32),
        alg="philox",
    )
    arm_seed = tf.random.experimental.stateless_fold_in(
        role_seed,
        tf.constant(bank.arm_id, tf.int32),
        alg="philox",
    )
    expected_derived = (
        role_seed,
        arm_seed,
        *(
            tf.random.experimental.stateless_fold_in(
                arm_seed,
                tf.constant(FAMILY_CODES[name], tf.int32),
                alg="philox",
            )
            for name in ("terminal", "process", "observation")
        ),
    )
    if len(bank.derived_seeds) != len(expected_derived):
        raise ValueError("innovation derived seed metadata is invalid")
    for actual, expected_seed in zip(bank.derived_seeds, expected_derived, strict=True):
        actual_seed = _require_seed(actual)
        if _seed_values(actual_seed) != _seed_values(expected_seed):
            raise ValueError("innovation derived seed metadata is invalid")
    if bank.content_signature != _innovation_bank_signature(bank):
        raise ValueError("innovation bank signature mismatch")


def _innovation_bank_signature(bank: SSLLSTMInnovationBank) -> str:
    return _canonical_sha256(
        {
            "algorithm": "philox",
            "arm_id": bank.arm_id,
            "draw_count": bank.draw_count,
            "family_codes": FAMILY_CODES,
            "horizon": FORECAST_HORIZON,
            "replication_count": bank.replication_count,
            "role": bank.role,
            "role_code": bank.role_code,
            "root_seed": [int(value) for value in tf.unstack(bank.root_seed)],
            "derived_seeds": [
                list(_seed_values(seed)) for seed in bank.derived_seeds
            ],
            "schema_version": (
                "bayesfilter.ssl_lstm_completion.phase_a2_innovation_bank_signature.v1"
            ),
            "tensor_hashes": bank.tensor_hashes(),
        }
    )


def make_ssl_lstm_innovation_bank(
    config: SSLLSTMForecastConfig,
    draw_count: int,
    seed: Any,
    role: InnovationRole,
    arm_id: int,
) -> SSLLSTMInnovationBank:
    """Generate three disjoint stateless-Philox innovation families."""

    if not isinstance(config, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    count = _require_static_positive_int(draw_count, name="draw_count")
    if role not in ROLE_CODES:
        raise ValueError("unknown innovation role")
    if isinstance(arm_id, bool):
        raise TypeError("arm_id must be an integer")
    try:
        arm = int(operator.index(arm_id))
    except TypeError as exc:
        raise TypeError("arm_id must be an integer") from exc
    if arm < -(2**31) or arm > 2**31 - 1:
        raise ValueError("arm_id must fit int32")
    if role == "paired_diagnostic_shared" and arm != 0:
        raise ValueError("paired_diagnostic_shared requires arm_id=0")
    if role == "independent_arm" and arm <= 0:
        raise ValueError("independent_arm requires a positive arm_id")
    root_seed = _require_seed(seed)
    role_seed = tf.random.experimental.stateless_fold_in(
        root_seed,
        tf.constant(ROLE_CODES[role], tf.int32),
        alg="philox",
    )
    arm_seed = tf.random.experimental.stateless_fold_in(
        role_seed,
        tf.constant(arm, tf.int32),
        alg="philox",
    )

    def family_seed(name: str) -> tf.Tensor:
        return tf.random.experimental.stateless_fold_in(
            arm_seed,
            tf.constant(FAMILY_CODES[name], tf.int32),
            alg="philox",
        )

    terminal_seed = family_seed("terminal")
    process_seed = family_seed("process")
    observation_seed = family_seed("observation")
    terminal = tf.random.stateless_normal(
        [count, config.replication_count, STATE_DIM],
        terminal_seed,
        dtype=tf.float64,
        alg="philox",
    )
    process = tf.random.stateless_normal(
        [count, config.replication_count, FORECAST_HORIZON, LATENT_DIM],
        process_seed,
        dtype=tf.float64,
        alg="philox",
    )
    observation = tf.random.stateless_normal(
        [count, config.replication_count, FORECAST_HORIZON, OBSERVATION_DIM],
        observation_seed,
        dtype=tf.float64,
        alg="philox",
    )
    provisional = SSLLSTMInnovationBank(
        terminal_standard_normal=terminal,
        process_standard_normal=process,
        observation_standard_normal=observation,
        root_seed=root_seed,
        algorithm="philox",
        role=role,
        role_code=ROLE_CODES[role],
        arm_id=arm,
        derived_seeds=(
            role_seed,
            arm_seed,
            terminal_seed,
            process_seed,
            observation_seed,
        ),
        content_signature="",
    )
    signature = _innovation_bank_signature(provisional)
    return SSLLSTMInnovationBank(**{**provisional.__dict__, "content_signature": signature})


def _audit_terminal_covariance(
    raw_covariance: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    raw = tf.ensure_shape(
        tf.convert_to_tensor(raw_covariance, tf.float64),
        [STATE_DIM, STATE_DIM],
    )
    finite = tf.reduce_all(tf.math.is_finite(raw))
    safe_raw = tf.where(tf.math.is_finite(raw), raw, tf.zeros_like(raw))
    sym = 0.5 * (safe_raw + tf.transpose(safe_raw))
    symmetry_residual = tf.reduce_max(tf.abs(safe_raw - tf.transpose(safe_raw)))
    frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(sym)))
    tau = tf.constant(
        COVARIANCE_ROUNDOFF_MULTIPLIER * FLOAT64_EPSILON,
        tf.float64,
    ) * tf.maximum(tf.constant(1.0, tf.float64), frobenius_norm)
    eigenvalues, eigenvectors = tf.linalg.eigh(sym)
    clipped = tf.maximum(eigenvalues, tf.zeros_like(eigenvalues))
    implemented = eigenvectors @ tf.linalg.diag(clipped) @ tf.transpose(eigenvectors)
    factor = (
        eigenvectors
        @ tf.linalg.diag(tf.sqrt(clipped))
        @ tf.transpose(eigenvectors)
    )
    projection_residual = tf.sqrt(tf.reduce_sum(tf.square(implemented - sym)))
    reconstruction_residual = tf.sqrt(
        tf.reduce_sum(tf.square(factor @ tf.transpose(factor) - implemented))
    )
    status = tf.constant(STATUS_VALID, tf.int32)
    status = tf.where(finite, status, tf.bitwise.bitwise_or(status, STATUS_NONFINITE))
    status = tf.where(
        symmetry_residual <= tau,
        status,
        tf.bitwise.bitwise_or(status, STATUS_ASYMMETRIC),
    )
    status = tf.where(
        tf.reduce_min(eigenvalues) >= -tau,
        status,
        tf.bitwise.bitwise_or(status, STATUS_MATERIALLY_INDEFINITE),
    )
    status = tf.where(
        tf.logical_and(tf.math.is_finite(projection_residual), projection_residual <= 8.0 * tau),
        status,
        tf.bitwise.bitwise_or(status, STATUS_PROJECTION),
    )
    status = tf.where(
        tf.logical_and(
            tf.math.is_finite(reconstruction_residual),
            reconstruction_residual <= 16.0 * tau,
        ),
        status,
        tf.bitwise.bitwise_or(status, STATUS_FACTOR_RECONSTRUCTION),
    )
    return (
        raw,
        sym,
        implemented,
        factor,
        eigenvalues,
        clipped,
        tf.reduce_min(eigenvalues),
        tau,
        symmetry_residual,
        projection_residual,
        reconstruction_residual,
        status,
    )


def _audit_terminal_covariance_batch_core(
    raw_covariances: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    """Audit terminal covariances with one batched eigensolver invocation."""

    raw = tf.convert_to_tensor(raw_covariances, tf.float64)
    finite = tf.reduce_all(tf.math.is_finite(raw), axis=[1, 2])
    safe_raw = tf.where(tf.math.is_finite(raw), raw, tf.zeros_like(raw))
    transposed = tf.transpose(safe_raw, [0, 2, 1])
    sym = 0.5 * (safe_raw + transposed)
    symmetry_residual = tf.reduce_max(tf.abs(safe_raw - transposed), axis=[1, 2])
    frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(sym), axis=[1, 2]))
    tau = tf.constant(
        COVARIANCE_ROUNDOFF_MULTIPLIER * FLOAT64_EPSILON,
        tf.float64,
    ) * tf.maximum(tf.ones_like(frobenius_norm), frobenius_norm)
    eigenvalues, eigenvectors = tf.linalg.eigh(sym)
    clipped = tf.maximum(eigenvalues, tf.zeros_like(eigenvalues))
    transpose_eigenvectors = tf.transpose(eigenvectors, [0, 2, 1])
    implemented = eigenvectors @ tf.linalg.diag(clipped) @ transpose_eigenvectors
    factor = (
        eigenvectors
        @ tf.linalg.diag(tf.sqrt(clipped))
        @ transpose_eigenvectors
    )
    projection_residual = tf.sqrt(
        tf.reduce_sum(tf.square(implemented - sym), axis=[1, 2])
    )
    reconstruction_residual = tf.sqrt(
        tf.reduce_sum(
            tf.square(factor @ tf.transpose(factor, [0, 2, 1]) - implemented),
            axis=[1, 2],
        )
    )
    status = tf.zeros(tf.shape(tau), tf.int32)
    status = tf.where(finite, status, tf.bitwise.bitwise_or(status, STATUS_NONFINITE))
    status = tf.where(
        symmetry_residual <= tau,
        status,
        tf.bitwise.bitwise_or(status, STATUS_ASYMMETRIC),
    )
    status = tf.where(
        tf.reduce_min(eigenvalues, axis=1) >= -tau,
        status,
        tf.bitwise.bitwise_or(status, STATUS_MATERIALLY_INDEFINITE),
    )
    status = tf.where(
        tf.logical_and(
            tf.math.is_finite(projection_residual),
            projection_residual <= 8.0 * tau,
        ),
        status,
        tf.bitwise.bitwise_or(status, STATUS_PROJECTION),
    )
    status = tf.where(
        tf.logical_and(
            tf.math.is_finite(reconstruction_residual),
            reconstruction_residual <= 16.0 * tau,
        ),
        status,
        tf.bitwise.bitwise_or(status, STATUS_FACTOR_RECONSTRUCTION),
    )
    return (
        raw,
        sym,
        implemented,
        factor,
        eigenvalues,
        clipped,
        tf.reduce_min(eigenvalues, axis=1),
        tau,
        symmetry_residual,
        projection_residual,
        reconstruction_residual,
        status,
    )


def _terminal_single_core(
    free: tf.Tensor,
    config: SSLLSTMForecastConfig,
    target: SSLLSTMPosteriorTarget,
) -> tuple[tf.Tensor, ...]:
    posterior = config.posterior_config
    full = posterior.parameter_mask.embed(free)
    components = make_ssl_lstm_svd_ukf_components(
        full,
        posterior.static_config,
        evidence_path=A2_RESULT_PATH,
        std_floor=posterior.std_floor,
    )
    value_result = tf_svd_sigma_point_filter(
        posterior.observations,
        components.model,
        backend="tf_svd_ukf",
        placement_floor=posterior.placement_floor,
        innovation_floor=posterior.innovation_floor,
        rank_tolerance=posterior.rank_tolerance,
        jitter=posterior.jitter,
        return_filtered=True,
    )
    if value_result.filtered_means is None or value_result.filtered_covariances is None:
        raise RuntimeError("filtered history is required for terminal extraction")
    filtered_means = tf.ensure_shape(value_result.filtered_means, [30, STATE_DIM])
    filtered_covariances = tf.ensure_shape(
        value_result.filtered_covariances,
        [30, STATE_DIM, STATE_DIM],
    )
    analytic_result, _ = tf_ssl_lstm_svd_ukf_score(
        posterior.observations,
        full,
        posterior.static_config,
        evidence_path=A2_RESULT_PATH,
        std_floor=posterior.std_floor,
        alpha=posterior.alpha,
        beta=posterior.beta,
        kappa=posterior.kappa,
        spectral_gap_tolerance=tf.constant(
            posterior.spectral_gap_tolerance,
            tf.float64,
        ),
    )
    target_value = (
        target.value(free)
        if config.jit_compile
        else target.eager_debug_value_and_score(free)[0]
    )
    delta = free - posterior.prior_center
    prior_variance = tf.constant(posterior.prior_standard_deviation**2, tf.float64)
    prior_value = -0.5 * tf.reduce_sum(tf.square(delta) / prior_variance)
    filter_value = tf.ensure_shape(value_result.log_likelihood, [])
    a1_filter_value = tf.ensure_shape(analytic_result.log_likelihood, [])
    total_value = filter_value + prior_value
    finite_values = tf.reduce_all(
        tf.math.is_finite(
            tf.stack([filter_value, a1_filter_value, target_value, total_value])
        )
    )
    safe_filter_value = tf.where(
        tf.math.is_finite(filter_value), filter_value, tf.zeros_like(filter_value)
    )
    safe_a1_filter_value = tf.where(
        tf.math.is_finite(a1_filter_value),
        a1_filter_value,
        tf.zeros_like(a1_filter_value),
    )
    safe_target_value = tf.where(
        tf.math.is_finite(target_value), target_value, tf.zeros_like(target_value)
    )
    safe_total_value = tf.where(
        tf.math.is_finite(total_value), total_value, tf.zeros_like(total_value)
    )
    filter_residual = tf.abs(safe_filter_value - safe_a1_filter_value)
    total_residual = tf.abs(safe_total_value - safe_target_value)
    filter_tolerance = _scale_tolerance(
        64,
        safe_filter_value,
        safe_a1_filter_value,
    )
    total_tolerance = _scale_tolerance(64, safe_total_value, safe_target_value)
    audited = _audit_terminal_covariance(filtered_covariances[-1])
    covariance_status = audited[-1]
    terminal_mean_is_finite = tf.reduce_all(tf.math.is_finite(filtered_means[-1]))
    covariance_status = tf.where(
        terminal_mean_is_finite,
        covariance_status,
        tf.bitwise.bitwise_or(covariance_status, STATUS_NONFINITE),
    )
    status = tf.where(
        filter_residual <= filter_tolerance,
        covariance_status,
        tf.bitwise.bitwise_or(covariance_status, STATUS_FILTER_PARITY),
    )
    status = tf.where(
        total_residual <= total_tolerance,
        status,
        tf.bitwise.bitwise_or(status, STATUS_TOTAL_PARITY),
    )
    status = tf.where(
        finite_values,
        status,
        tf.bitwise.bitwise_or(status, STATUS_NONFINITE),
    )
    return (
        filtered_means[-1],
        *audited[:-1],
        filter_value,
        a1_filter_value,
        target_value,
        total_value,
        filter_residual,
        total_residual,
        filter_tolerance,
        total_tolerance,
        full,
        status,
    )


def _terminal_batch_core(
    free_draws: tf.Tensor,
    config: SSLLSTMForecastConfig,
    target: SSLLSTMPosteriorTarget,
) -> tuple[tf.Tensor, ...]:
    return tf.map_fn(
        lambda free: _terminal_single_core(free, config, target),
        free_draws,
        fn_output_signature=(
            tf.TensorSpec([STATE_DIM], tf.float64),
            tf.TensorSpec([STATE_DIM, STATE_DIM], tf.float64),
            tf.TensorSpec([STATE_DIM, STATE_DIM], tf.float64),
            tf.TensorSpec([STATE_DIM, STATE_DIM], tf.float64),
            tf.TensorSpec([STATE_DIM, STATE_DIM], tf.float64),
            tf.TensorSpec([STATE_DIM], tf.float64),
            tf.TensorSpec([STATE_DIM], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([24], tf.float64),
            tf.TensorSpec([], tf.int32),
        ),
        parallel_iterations=1,
    )


_TERMINAL_PROGRAM_CACHE: dict[
    tuple[str, int],
    Callable[[tf.Tensor], tuple[tf.Tensor, ...]],
] = {}
_TERMINAL_COVARIANCE_AUDIT_PROGRAM_CACHE: dict[
    int,
    Callable[[tf.Tensor], tuple[tf.Tensor, ...]],
] = {}
_FORECAST_PROGRAM_CACHE: dict[
    tuple[str, int],
    Callable[..., tuple[tf.Tensor, ...]],
] = {}


def ssl_lstm_terminal_compiled_program(
    config: SSLLSTMForecastConfig,
    draw_count: int,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, ...]]:
    """Return the reusable XLA terminal program for one static draw shape."""

    if not isinstance(config, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    if not config.jit_compile:
        raise ValueError("compiled terminal programs require jit_compile=True")
    count = _require_static_positive_int(draw_count, name="draw_count")
    key = (config.signature(), count)
    compiled = _TERMINAL_PROGRAM_CACHE.get(key)
    if compiled is None:
        target = SSLLSTMPosteriorTarget(config.posterior_config)

        def terminal_program(values: tf.Tensor) -> tuple[tf.Tensor, ...]:
            return _terminal_batch_core(values, config, target)

        compiled = tf.function(
            terminal_program,
            input_signature=[tf.TensorSpec([count, 4], tf.float64)],
            jit_compile=True,
            reduce_retracing=True,
        )
        _TERMINAL_PROGRAM_CACHE[key] = compiled
    return compiled


def ssl_lstm_terminal_covariance_audit_compiled_program(
    draw_count: int,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, ...]]:
    """Return the staged batched XLA covariance audit for one draw shape."""

    count = _require_static_positive_int(draw_count, name="draw_count")
    compiled = _TERMINAL_COVARIANCE_AUDIT_PROGRAM_CACHE.get(count)
    if compiled is None:
        compiled = tf.function(
            _audit_terminal_covariance_batch_core,
            input_signature=[tf.TensorSpec([count, STATE_DIM, STATE_DIM], tf.float64)],
            autograph=False,
            jit_compile=True,
            reduce_retracing=True,
        )
        _TERMINAL_COVARIANCE_AUDIT_PROGRAM_CACHE[count] = compiled
    return compiled


def _replace_terminal_covariance_audit(
    tensors: tuple[tf.Tensor, ...],
    covariance_audit: tuple[tf.Tensor, ...],
) -> tuple[tf.Tensor, ...]:
    preserved_status = tf.bitwise.bitwise_and(
        tensors[21],
        tf.constant(
            STATUS_NONFINITE | STATUS_FILTER_PARITY | STATUS_TOTAL_PARITY,
            tf.int32,
        ),
    )
    status = tf.bitwise.bitwise_or(preserved_status, covariance_audit[11])
    return (
        tensors[0],
        *covariance_audit[:11],
        *tensors[12:21],
        status,
    )


def _terminal_from_tensors(tensors: tuple[tf.Tensor, ...]) -> SSLLSTMTerminalState:
    return SSLLSTMTerminalState(
        mean=tensors[0],
        raw_covariance=tensors[1],
        symmetrized_covariance=tensors[2],
        implemented_covariance=tensors[3],
        factor=tensors[4],
        raw_eigenvalues=tensors[5],
        clipped_eigenvalues=tensors[6],
        minimum_eigenvalue=tensors[7],
        psd_tolerance=tensors[8],
        symmetry_residual=tensors[9],
        projection_residual=tensors[10],
        factor_reconstruction_residual=tensors[11],
        filter_log_likelihood=tensors[12],
        a1_filter_log_likelihood=tensors[13],
        target_value=tensors[14],
        total_value=tensors[15],
        filter_parity_residual=tensors[16],
        total_parity_residual=tensors[17],
        filter_parity_tolerance=tensors[18],
        total_parity_tolerance=tensors[19],
        full_parameters=tensors[20],
        status=tensors[21],
    )


def _require_valid_terminal(terminal: SSLLSTMTerminalState) -> None:
    statuses = tf.reshape(terminal.status, [-1])
    status_values = [int(value) for value in tf.unstack(statuses)]
    finite_fields = (
        terminal.mean,
        terminal.raw_covariance,
        terminal.symmetrized_covariance,
        terminal.implemented_covariance,
        terminal.factor,
        terminal.raw_eigenvalues,
        terminal.clipped_eigenvalues,
        terminal.minimum_eigenvalue,
        terminal.psd_tolerance,
        terminal.symmetry_residual,
        terminal.projection_residual,
        terminal.factor_reconstruction_residual,
        terminal.filter_log_likelihood,
        terminal.a1_filter_log_likelihood,
        terminal.target_value,
        terminal.total_value,
        terminal.filter_parity_residual,
        terminal.total_parity_residual,
        terminal.filter_parity_tolerance,
        terminal.total_parity_tolerance,
        terminal.full_parameters,
    )
    all_finite = all(
        bool(tf.reduce_all(tf.math.is_finite(tensor))) for tensor in finite_fields
    )
    if any(value != STATUS_VALID for value in status_values) or not all_finite:
        raise ValueError(
            "terminal extraction failed closed with status "
            f"{status_values} and all_finite={all_finite}"
        )


_FORECAST_OUTPUT_NAMES = (
    "terminal_states",
    "states",
    "deterministic_transition_means",
    "process_innovations",
    "observation_means",
    "observation_innovations",
    "observations",
)


def _require_finite_forecast_outputs(tensors: tuple[tf.Tensor, ...]) -> None:
    if len(tensors) != len(_FORECAST_OUTPUT_NAMES):
        raise ValueError("forecast execution returned an invalid output count")
    for name, tensor in zip(_FORECAST_OUTPUT_NAMES, tensors, strict=True):
        _require_all_finite(tensor, name=f"forecast output {name}")


def extract_ssl_lstm_terminal_states(
    free_draws: Any,
    config: SSLLSTMForecastConfig | None = None,
) -> SSLLSTMTerminalState:
    """Extract a valid terminal Gaussian for a static parameter-draw batch."""

    resolved = SSLLSTMForecastConfig() if config is None else config
    if not isinstance(resolved, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    values = tf.convert_to_tensor(free_draws)
    if values.dtype != tf.float64:
        raise TypeError("free_draws must have dtype float64")
    if values.shape.rank != 2 or values.shape[0] is None or values.shape[1] != 4:
        raise ValueError("free_draws must have static shape [draw, 4]")
    draw_count = int(values.shape[0])
    if draw_count <= 0:
        raise ValueError("draw dimension must be positive")
    _require_all_finite(values, name="free_draws")
    if resolved.jit_compile:
        compiled = ssl_lstm_terminal_compiled_program(resolved, draw_count)
        tensors = compiled(values)
        covariance_program = ssl_lstm_terminal_covariance_audit_compiled_program(
            draw_count
        )
        covariance_audit = covariance_program(tensors[1])
    else:
        target = SSLLSTMPosteriorTarget(resolved.posterior_config)
        tensors = _terminal_batch_core(values, resolved, target)
        covariance_audit = _audit_terminal_covariance_batch_core(tensors[1])
    tensors = _replace_terminal_covariance_audit(
        tuple(tensors), tuple(covariance_audit)
    )
    terminal = _terminal_from_tensors(tensors)
    _require_valid_terminal(terminal)
    return terminal


def extract_ssl_lstm_terminal_state(
    free_draw: Any,
    config: SSLLSTMForecastConfig | None = None,
) -> SSLLSTMTerminalState:
    """Scalar convenience form of terminal extraction."""

    values = _require_float64_tensor(free_draw, shape=(4,), name="free_draw")
    batched = extract_ssl_lstm_terminal_states(values[tf.newaxis, :], config)
    return SSLLSTMTerminalState(
        **{
            name: getattr(batched, name)[0]
            for name in SSLLSTMTerminalState.__dataclass_fields__
        }
    )


def _forecast_batch_core(
    free_draws: tf.Tensor,
    terminal_mean: tf.Tensor,
    terminal_factor: tf.Tensor,
    terminal_standard_normal: tf.Tensor,
    process_standard_normal: tf.Tensor,
    observation_standard_normal: tf.Tensor,
    config: SSLLSTMForecastConfig,
) -> tuple[tf.Tensor, ...]:
    draw_count = int(free_draws.shape[0])
    replication_count = config.replication_count
    terminal_states = terminal_mean[:, tf.newaxis, :] + tf.einsum(
        "drn,dkn->drk",
        terminal_standard_normal,
        terminal_factor,
    )
    state_rows = []
    deterministic_rows = []
    process_rows = []
    observation_mean_rows = []
    observation_noise_rows = []
    observation_rows = []
    for draw_index in range(draw_count):
        full = config.posterior_config.parameter_mask.embed(free_draws[draw_index])
        components = make_ssl_lstm_svd_ukf_components(
            full,
            config.posterior_config.static_config,
            evidence_path=A2_RESULT_PATH,
            std_floor=config.posterior_config.std_floor,
        )
        params = components.parameters
        previous = terminal_states[draw_index]
        draw_states = []
        draw_deterministic = []
        draw_process = []
        draw_observation_mean = []
        draw_observation_noise = []
        draw_observations = []
        for horizon_index in range(FORECAST_HORIZON):
            deterministic = ssl_lstm_transition(params, previous)
            process_noise = (
                process_standard_normal[draw_index, :, horizon_index, :]
                * params.process_std[tf.newaxis, :]
            )
            next_state = tf.concat(
                [
                    deterministic[:, :LATENT_DIM] + process_noise,
                    deterministic[:, LATENT_DIM:],
                ],
                axis=1,
            )
            observation_mean = ssl_lstm_observation(params, next_state)
            observation_noise = (
                observation_standard_normal[draw_index, :, horizon_index, :]
                * params.observation_std[tf.newaxis, :]
            )
            observations = observation_mean + observation_noise
            draw_states.append(next_state)
            draw_deterministic.append(deterministic)
            draw_process.append(process_noise)
            draw_observation_mean.append(observation_mean)
            draw_observation_noise.append(observation_noise)
            draw_observations.append(observations)
            previous = next_state
        state_rows.append(tf.stack(draw_states, axis=1))
        deterministic_rows.append(tf.stack(draw_deterministic, axis=1))
        process_rows.append(tf.stack(draw_process, axis=1))
        observation_mean_rows.append(tf.stack(draw_observation_mean, axis=1))
        observation_noise_rows.append(tf.stack(draw_observation_noise, axis=1))
        observation_rows.append(tf.stack(draw_observations, axis=1))
    return (
        tf.ensure_shape(terminal_states, [draw_count, replication_count, STATE_DIM]),
        tf.stack(state_rows, axis=0),
        tf.stack(deterministic_rows, axis=0),
        tf.stack(process_rows, axis=0),
        tf.stack(observation_mean_rows, axis=0),
        tf.stack(observation_noise_rows, axis=0),
        tf.stack(observation_rows, axis=0),
    )


def ssl_lstm_forecast_compiled_program(
    config: SSLLSTMForecastConfig,
    draw_count: int,
) -> Callable[..., tuple[tf.Tensor, ...]]:
    """Return the reusable XLA forecast program for one static draw shape."""

    if not isinstance(config, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    if not config.jit_compile:
        raise ValueError("compiled forecast programs require jit_compile=True")
    count = _require_static_positive_int(draw_count, name="draw_count")
    key = (config.signature(), count)
    compiled = _FORECAST_PROGRAM_CACHE.get(key)
    if compiled is None:

        def forecast_program(
            free_draws: tf.Tensor,
            terminal_mean: tf.Tensor,
            terminal_factor: tf.Tensor,
            terminal_standard_normal: tf.Tensor,
            process_standard_normal: tf.Tensor,
            observation_standard_normal: tf.Tensor,
        ) -> tuple[tf.Tensor, ...]:
            return _forecast_batch_core(
                free_draws,
                terminal_mean,
                terminal_factor,
                terminal_standard_normal,
                process_standard_normal,
                observation_standard_normal,
                config,
            )

        replications = config.replication_count
        compiled = tf.function(
            forecast_program,
            input_signature=[
                tf.TensorSpec([count, 4], tf.float64),
                tf.TensorSpec([count, STATE_DIM], tf.float64),
                tf.TensorSpec([count, STATE_DIM, STATE_DIM], tf.float64),
                tf.TensorSpec([count, replications, STATE_DIM], tf.float64),
                tf.TensorSpec(
                    [count, replications, FORECAST_HORIZON, LATENT_DIM],
                    tf.float64,
                ),
                tf.TensorSpec(
                    [count, replications, FORECAST_HORIZON, OBSERVATION_DIM],
                    tf.float64,
                ),
            ],
            jit_compile=True,
            reduce_retracing=True,
        )
        _FORECAST_PROGRAM_CACHE[key] = compiled
    return compiled


def _device_rows(devices: list[Any]) -> tuple[tuple[str, str], ...]:
    return tuple((str(device.name), str(device.device_type)) for device in devices)


def _resolve_runtime_provenance(
    config: SSLLSTMForecastConfig,
    tensors: tuple[tf.Tensor, ...],
    *,
    runtime_execution_role: str | None,
    trust_basis: str | None,
) -> tuple[
    str,
    str,
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
]:
    physical = _device_rows(tf.config.list_physical_devices())
    logical = _device_rows(tf.config.list_logical_devices())
    output_devices = tuple(str(tensor.device) for tensor in tensors)
    if runtime_execution_role is None:
        role = config.execution_role
        trust = "unclassified_runtime_not_evidence"
    else:
        role = str(runtime_execution_role)
        if trust_basis is None:
            raise ValueError("an evidence runtime role requires an explicit trust_basis")
        trust = str(trust_basis)
    allowed = {
        "default_xla": "unclassified_runtime_not_evidence",
        "eager_debug_reference": "unclassified_runtime_not_evidence",
        "cpu_hidden_xla_reference": "cpu_hidden_reference_exception_not_gpu_evidence",
        "trusted_gpu_xla_canary": (
            "owner_designated_managed_session_visible_gpu_trusted"
        ),
    }
    if role not in allowed or trust != allowed[role]:
        raise ValueError("runtime execution role and trust basis are inconsistent")
    if config.jit_compile and role == "eager_debug_reference":
        raise ValueError("compiled execution cannot claim eager_debug_reference")
    if not config.jit_compile and role != "eager_debug_reference":
        raise ValueError("eager execution cannot claim an XLA runtime role")
    logical_gpu = tuple(row for row in logical if row[1] == "GPU")
    if role == "cpu_hidden_xla_reference" and logical_gpu:
        raise RuntimeError("CPU-hidden reference execution exposed a logical GPU")
    if role == "trusted_gpu_xla_canary":
        if not logical_gpu:
            raise RuntimeError("trusted GPU/XLA execution requires a logical GPU")
        if any("GPU" not in device.upper() for device in output_devices):
            raise RuntimeError("trusted GPU/XLA outputs must be placed on GPU")
    return role, trust, physical, logical, output_devices


def forecast_ssl_lstm_paths(
    free_draws: Any,
    innovation_bank: SSLLSTMInnovationBank,
    config: SSLLSTMForecastConfig | None = None,
    *,
    draw_chunk_size: int | None = None,
    runtime_execution_role: str | None = None,
    trust_basis: str | None = None,
) -> SSLLSTMForecastPaths:
    """Produce replayable ten-step forecast paths for a static draw batch."""

    resolved = SSLLSTMForecastConfig() if config is None else config
    if not isinstance(resolved, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    values = tf.convert_to_tensor(free_draws)
    if values.dtype != tf.float64:
        raise TypeError("free_draws must have dtype float64")
    if values.shape.rank != 2 or values.shape[0] is None or values.shape[1] != 4:
        raise ValueError("free_draws must have static shape [draw, 4]")
    draw_count = int(values.shape[0])
    if draw_count <= 0:
        raise ValueError("draw dimension must be positive")
    chunk_size = (
        draw_count
        if draw_chunk_size is None
        else _require_static_positive_int(draw_chunk_size, name="draw_chunk_size")
    )
    if chunk_size > draw_count:
        raise ValueError("draw_chunk_size cannot exceed the draw dimension")
    _require_all_finite(values, name="free_draws")
    _validate_innovation_bank(innovation_bank, draw_count=draw_count, config=resolved)
    terminal = extract_ssl_lstm_terminal_states(values, resolved)
    inputs = (
        values,
        terminal.mean,
        terminal.factor,
        innovation_bank.terminal_standard_normal,
        innovation_bank.process_standard_normal,
        innovation_bank.observation_standard_normal,
    )
    if resolved.jit_compile:
        chunk_rows: list[list[tf.Tensor]] = [[] for _ in _FORECAST_OUTPUT_NAMES]
        for start in range(0, draw_count, chunk_size):
            stop = min(start + chunk_size, draw_count)
            compiled = ssl_lstm_forecast_compiled_program(resolved, stop - start)
            chunk = compiled(*(tensor[start:stop] for tensor in inputs))
            for rows, tensor in zip(chunk_rows, chunk, strict=True):
                rows.append(tensor)
        tensors = tuple(tf.concat(rows, axis=0) for rows in chunk_rows)
    else:
        if chunk_size != draw_count:
            raise ValueError("draw_chunk_size is supported only for compiled forecasts")
        tensors = _forecast_batch_core(*inputs, resolved)
    tensors = tuple(tensors)
    _require_finite_forecast_outputs(tensors)
    full_parameters = tf.stack(
        [resolved.posterior_config.parameter_mask.embed(values[index]) for index in range(draw_count)],
        axis=0,
    )
    runtime_role, runtime_trust, physical, logical, output_devices = (
        _resolve_runtime_provenance(
            resolved,
            tuple(tensors),
            runtime_execution_role=runtime_execution_role,
            trust_basis=trust_basis,
        )
    )
    tensor_hashes = innovation_bank.tensor_hashes()
    posterior = resolved.posterior_config
    a1_adapter_signature = SSLLSTMPosteriorTarget(posterior).adapter_signature()
    provenance = SSLLSTMForecastProvenance(
        schema_version="bayesfilter.ssl_lstm_completion.phase_a2_provenance.v1",
        a2_contract_signature=A2_CONTRACT_SIGNATURE,
        a1_result_file_sha256=A1_RESULT_FILE_SHA256,
        a2_subplan_file_sha256=A2_SUBPLAN_FILE_SHA256,
        forecast_config_signature=resolved.signature(),
        innovation_bank_signature=innovation_bank.content_signature,
        free_draw_matrix_raw_sha256=_raw_float64_sha256(values),
        embedded_full_parameter_matrix_raw_sha256=_raw_float64_sha256(full_parameters),
        target_semantic_sha256=TARGET_SEMANTIC_SHA256,
        a1_adapter_signature=a1_adapter_signature,
        parameter_mask_sha256=PARAMETER_MASK_SHA256,
        observation_raw_sha256=OBSERVATION_RAW_SHA256,
        full_fixture_raw_sha256=FULL_FIXTURE_RAW_SHA256,
        prior_center_raw_sha256=PRIOR_CENTER_RAW_SHA256,
        innovation_role=innovation_bank.role,
        innovation_role_code=innovation_bank.role_code,
        innovation_arm_id=innovation_bank.arm_id,
        innovation_algorithm=innovation_bank.algorithm,
        innovation_root_seed=_seed_values(innovation_bank.root_seed),
        innovation_family_codes=tuple(sorted(FAMILY_CODES.items())),
        innovation_tensor_hashes=tuple(sorted(tensor_hashes.items())),
        innovation_replay_authority="materialized_tensor_hashes",
        innovation_seed_qualification=(
            "generation_metadata_not_cross_backend_bitwise_regeneration_evidence"
        ),
        draw_count=draw_count,
        draw_chunk_size=chunk_size,
        replication_count=resolved.replication_count,
        forecast_horizon=FORECAST_HORIZON,
        filter_backend="tf_svd_ukf",
        filter_settings_hex=(
            ("std_floor", float(posterior.std_floor).hex()),
            ("alpha", float(posterior.alpha).hex()),
            ("beta", float(posterior.beta).hex()),
            ("kappa", float(posterior.kappa).hex()),
            ("placement_floor", float(posterior.placement_floor).hex()),
            ("innovation_floor", float(posterior.innovation_floor).hex()),
            ("rank_tolerance", float(posterior.rank_tolerance).hex()),
            ("jitter", float(posterior.jitter).hex()),
        ),
        covariance_roundoff_multiplier=COVARIANCE_ROUNDOFF_MULTIPLIER,
        terminal_covariance_statuses=tuple(
            int(value) for value in tf.unstack(tf.reshape(terminal.status, [-1]))
        ),
        tensorflow_version=tf.__version__,
        dtype="float64",
        jit_compile=resolved.jit_compile,
        execution_role=runtime_role,
        physical_devices=physical,
        logical_devices=logical,
        output_devices=output_devices,
        tf32_enabled=bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        trust_basis=runtime_trust,
        horizon_convention="state_and_observation_after_transition_t_plus_1",
        cluster_unit="complete_ten_step_path_per_draw_replication",
        approximation_qualification=(
            "conditional_on_approximate_historical_svd_ukf_not_exact_nonlinear_filter"
        ),
        nonclaims=NONCLAIMS,
    )
    return SSLLSTMForecastPaths(
        terminal_states=tensors[0],
        states=tensors[1],
        deterministic_transition_means=tensors[2],
        process_innovations=tensors[3],
        observation_means=tensors[4],
        observation_innovations=tensors[5],
        observations=tensors[6],
        terminal=terminal,
        provenance=provenance,
    )


def _squeeze_terminal(terminal: SSLLSTMTerminalState) -> SSLLSTMTerminalState:
    return SSLLSTMTerminalState(
        **{
            name: getattr(terminal, name)[0]
            for name in SSLLSTMTerminalState.__dataclass_fields__
        }
    )


def forecast_ssl_lstm_path(
    free_draw: Any,
    innovation_bank: SSLLSTMInnovationBank,
    config: SSLLSTMForecastConfig | None = None,
    *,
    runtime_execution_role: str | None = None,
    trust_basis: str | None = None,
) -> SSLLSTMForecastPaths:
    """Scalar convenience form of the canonical one-row forecast batch."""

    values = _require_float64_tensor(free_draw, shape=(4,), name="free_draw")
    if innovation_bank.draw_count != 1:
        raise ValueError("scalar forecast requires an innovation bank with draw_count=1")
    batched = forecast_ssl_lstm_paths(
        values[tf.newaxis, :],
        innovation_bank,
        config,
        runtime_execution_role=runtime_execution_role,
        trust_basis=trust_basis,
    )
    return SSLLSTMForecastPaths(
        terminal_states=batched.terminal_states[0],
        states=batched.states[0],
        deterministic_transition_means=batched.deterministic_transition_means[0],
        process_innovations=batched.process_innovations[0],
        observation_means=batched.observation_means[0],
        observation_innovations=batched.observation_innovations[0],
        observations=batched.observations[0],
        terminal=_squeeze_terminal(batched.terminal),
        provenance=batched.provenance,
    )


def eager_debug_ssl_lstm_terminal_states(
    free_draws: Any,
    config: SSLLSTMForecastConfig | None = None,
) -> SSLLSTMTerminalState:
    """Run the same terminal extraction as an eager debug reference."""

    base = SSLLSTMForecastConfig() if config is None else config
    if not isinstance(base, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    eager_config = SSLLSTMForecastConfig(
        posterior_config=base.posterior_config,
        forecast_horizon=base.forecast_horizon,
        replication_count=base.replication_count,
        jit_compile=False,
        execution_role="eager_debug_reference",
    )
    return extract_ssl_lstm_terminal_states(free_draws, eager_config)


def eager_debug_ssl_lstm_forecast_paths(
    free_draws: Any,
    innovation_bank: SSLLSTMInnovationBank,
    config: SSLLSTMForecastConfig | None = None,
) -> SSLLSTMForecastPaths:
    """Run the same forecast recursion as an eager debug reference."""

    base = SSLLSTMForecastConfig() if config is None else config
    if not isinstance(base, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    eager_config = SSLLSTMForecastConfig(
        posterior_config=base.posterior_config,
        forecast_horizon=base.forecast_horizon,
        replication_count=base.replication_count,
        jit_compile=False,
        execution_role="eager_debug_reference",
    )
    return forecast_ssl_lstm_paths(free_draws, innovation_bank, eager_config)

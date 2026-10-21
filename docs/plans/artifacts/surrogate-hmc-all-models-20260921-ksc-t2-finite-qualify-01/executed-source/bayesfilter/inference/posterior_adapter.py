"""Model-agnostic posterior adapter and value/score authority contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


ValueScoreAuthority = Literal[
    "graph_native",
    "gradient_tape_fallback",
    "reviewed_gradient_tape_xla_exception",
    "reviewed_tf_py_function_finite_reject_bridge",
    "debug_only",
    "unavailable",
]
TransformOrientation = Literal[
    "identity",
    "constrained_to_unconstrained",
    "unconstrained_to_constrained",
]
MissingnessConvention = Literal[
    "none",
    "boolean_mask_true_observed",
    "nan_missing",
]
RepairRole = Literal["target", "diagnostic", "veto"]
CompileMode = Literal["eager", "tf_function", "xla"]
SeedPolicy = Literal["not_used", "stateless_required", "external"]

_KNOWN_AUTHORITIES = {
    "graph_native",
    "gradient_tape_fallback",
    "reviewed_gradient_tape_xla_exception",
    "reviewed_tf_py_function_finite_reject_bridge",
    "debug_only",
    "unavailable",
}
_ACCEPTED_XLA_AUTHORITIES = {
    "graph_native",
    "reviewed_gradient_tape_xla_exception",
}
_KNOWN_TRANSFORM_ORIENTATIONS = {
    "identity",
    "constrained_to_unconstrained",
    "unconstrained_to_constrained",
}
_KNOWN_MISSINGNESS = {
    "none",
    "boolean_mask_true_observed",
    "nan_missing",
}
_KNOWN_REPAIR_ROLES = {"target", "diagnostic", "veto"}
_KNOWN_COMPILE_MODES = {"eager", "tf_function", "xla"}
_KNOWN_SEED_POLICIES = {"not_used", "stateless_required", "external"}
_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


class UnsupportedValueScoreAuthority(ValueError):
    """Raised when an adapter advertises an unreviewed authority label."""


class InvalidNonlinearSSMContract(ValueError):
    """Raised when nonlinear SSM metadata is incomplete or unsafe to persist."""


@runtime_checkable
class PosteriorAdapter(Protocol):
    """Minimal posterior value adapter protocol."""

    def log_prob(self, theta: Any) -> Any: ...


@runtime_checkable
class ValueScorePosteriorAdapter(PosteriorAdapter, Protocol):
    """Posterior adapter that can return a value and score together."""

    def log_prob_and_grad(self, theta: Any) -> tuple[Any, Any]: ...


@runtime_checkable
class HessianPosteriorAdapter(ValueScorePosteriorAdapter, Protocol):
    """Optional Hessian-capable posterior adapter protocol."""

    def negative_log_prob_hessian(self, theta: Any) -> Any: ...


@dataclass(frozen=True)
class ValueScoreCapability:
    """Operational authority metadata for value/score use in HMC runtimes."""

    value_score_authority: ValueScoreAuthority
    xla_hmc_ready: bool
    runtime_backend: str = "unknown"
    evidence_path: str | None = None
    target_scope: str | None = None
    nonclaims: tuple[str, ...] = ()
    full_chain_xla_diagnostic_ready: bool = False

    def __post_init__(self) -> None:
        authority = str(self.value_score_authority)
        if authority not in _KNOWN_AUTHORITIES:
            raise UnsupportedValueScoreAuthority(
                f"unsupported value/score authority: {authority!r}"
            )
        object.__setattr__(self, "value_score_authority", authority)
        object.__setattr__(self, "xla_hmc_ready", bool(self.xla_hmc_ready))
        object.__setattr__(self, "runtime_backend", str(self.runtime_backend))
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))
        object.__setattr__(
            self,
            "full_chain_xla_diagnostic_ready",
            bool(self.full_chain_xla_diagnostic_ready),
        )
        if self.evidence_path is not None:
            object.__setattr__(self, "evidence_path", str(self.evidence_path))
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))
        if authority == "reviewed_gradient_tape_xla_exception" and self.xla_hmc_ready:
            if not self.evidence_path and not self.nonclaims:
                raise ValueError(
                    "reviewed GradientTape XLA exceptions require scoped evidence_path "
                    "or nonclaims"
                )
            if not self.target_scope:
                raise ValueError(
                    "reviewed GradientTape XLA exceptions require target_scope binding"
                )
        if authority == "reviewed_tf_py_function_finite_reject_bridge":
            if self.xla_hmc_ready:
                raise ValueError("reviewed tf.py_function finite-reject bridge is non-XLA only")
            if not self.evidence_path and not self.nonclaims:
                raise ValueError(
                    "reviewed tf.py_function finite-reject bridges require scoped evidence_path "
                    "or nonclaims"
                )
            if not self.target_scope:
                raise ValueError(
                    "reviewed tf.py_function finite-reject bridges require target_scope binding"
                )
        if self.full_chain_xla_diagnostic_ready:
            if not self.is_accepted_xla_hmc_authority:
                raise ValueError(
                    "full-chain XLA diagnostic authority requires accepted target-XLA authority"
                )
            if not self.target_scope:
                raise ValueError(
                    "full-chain XLA diagnostic authority requires target_scope binding"
                )
            if not self.evidence_path and not self.nonclaims:
                raise ValueError(
                    "full-chain XLA diagnostic authority requires scoped evidence_path "
                    "or nonclaims"
                )

    @property
    def is_accepted_xla_hmc_authority(self) -> bool:
        return bool(
            self.xla_hmc_ready
            and self.value_score_authority in _ACCEPTED_XLA_AUTHORITIES
        )

    @property
    def is_accepted_full_chain_xla_diagnostic_authority(self) -> bool:
        return bool(
            self.full_chain_xla_diagnostic_ready
            and self.is_accepted_xla_hmc_authority
        )


@dataclass(frozen=True)
class NonlinearSSMStaticShape:
    """Fixed-shape nonlinear SSM dimensions used by compiled value/HMC paths."""

    horizon: int
    state_dim: int
    observation_dim: int
    innovation_dim: int
    parameter_dim: int
    constrained_parameter_shape: tuple[int, ...]
    unconstrained_parameter_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "horizon",
            "state_dim",
            "observation_dim",
            "innovation_dim",
            "parameter_dim",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise InvalidNonlinearSSMContract(f"{name} must be positive")
            object.__setattr__(self, name, value)
        constrained = _coerce_shape(self.constrained_parameter_shape)
        unconstrained = _coerce_shape(self.unconstrained_parameter_shape)
        if _shape_size(constrained) != self.parameter_dim:
            raise InvalidNonlinearSSMContract(
                "constrained_parameter_shape size must equal parameter_dim"
            )
        if _shape_size(unconstrained) != self.parameter_dim:
            raise InvalidNonlinearSSMContract(
                "unconstrained_parameter_shape size must equal parameter_dim"
            )
        object.__setattr__(self, "constrained_parameter_shape", constrained)
        object.__setattr__(self, "unconstrained_parameter_shape", unconstrained)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "observation_dim": self.observation_dim,
            "innovation_dim": self.innovation_dim,
            "parameter_dim": self.parameter_dim,
            "constrained_parameter_shape": self.constrained_parameter_shape,
            "unconstrained_parameter_shape": self.unconstrained_parameter_shape,
        }


@dataclass(frozen=True)
class ParameterTransformMetadata:
    """Parameter transform orientation and inverse-transform metadata."""

    orientation: TransformOrientation
    inverse_orientation: TransformOrientation
    log_det_jacobian_convention: str = "not_included"
    transform_source: str = "adapter"

    def __post_init__(self) -> None:
        orientation = str(self.orientation)
        inverse = str(self.inverse_orientation)
        if orientation not in _KNOWN_TRANSFORM_ORIENTATIONS:
            raise InvalidNonlinearSSMContract(f"unknown transform orientation: {orientation}")
        if inverse not in _KNOWN_TRANSFORM_ORIENTATIONS:
            raise InvalidNonlinearSSMContract(f"unknown inverse orientation: {inverse}")
        if orientation == "identity" and inverse != "identity":
            raise InvalidNonlinearSSMContract("identity transform requires identity inverse")
        if orientation == "constrained_to_unconstrained" and inverse != "unconstrained_to_constrained":
            raise InvalidNonlinearSSMContract("inverse transform orientation mismatch")
        if orientation == "unconstrained_to_constrained" and inverse != "constrained_to_unconstrained":
            raise InvalidNonlinearSSMContract("inverse transform orientation mismatch")
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "inverse_orientation", inverse)
        object.__setattr__(
            self,
            "log_det_jacobian_convention",
            _nonempty_text(self.log_det_jacobian_convention, "log_det_jacobian_convention"),
        )
        object.__setattr__(
            self,
            "transform_source",
            _nonempty_text(self.transform_source, "transform_source"),
        )

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "orientation": self.orientation,
            "inverse_orientation": self.inverse_orientation,
            "log_det_jacobian_convention": self.log_det_jacobian_convention,
            "transform_source": self.transform_source,
        }


@dataclass(frozen=True)
class ObservationSemantics:
    """Observation mask and missingness semantics for fixed-shape filtering."""

    mask_convention: str
    missingness_convention: MissingnessConvention
    mask_shape: tuple[int, int] | None = None
    observed_indicator: bool | None = True

    def __post_init__(self) -> None:
        missingness = str(self.missingness_convention)
        if missingness not in _KNOWN_MISSINGNESS:
            raise InvalidNonlinearSSMContract(
                f"unknown missingness convention: {missingness}"
            )
        object.__setattr__(
            self,
            "mask_convention",
            _nonempty_text(self.mask_convention, "mask_convention"),
        )
        object.__setattr__(self, "missingness_convention", missingness)
        if self.mask_shape is not None:
            shape = _coerce_shape(self.mask_shape)
            if len(shape) != 2:
                raise InvalidNonlinearSSMContract("mask_shape must be two-dimensional")
            object.__setattr__(self, "mask_shape", shape)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "mask_convention": self.mask_convention,
            "missingness_convention": self.missingness_convention,
            "mask_shape": self.mask_shape,
            "observed_indicator": self.observed_indicator,
        }


@dataclass(frozen=True)
class RegularizationConvention:
    """Regularization metadata for the covariance law actually implemented."""

    jitter: float
    covariance_floor: float
    psd_repair: str
    symmetrize: bool
    logdet_convention: str
    implemented_covariance: str
    repair_role: RepairRole

    def __post_init__(self) -> None:
        jitter = float(self.jitter)
        floor = float(self.covariance_floor)
        if jitter < 0.0:
            raise InvalidNonlinearSSMContract("jitter must be nonnegative")
        if floor < 0.0:
            raise InvalidNonlinearSSMContract("covariance_floor must be nonnegative")
        role = str(self.repair_role)
        if role not in _KNOWN_REPAIR_ROLES:
            raise InvalidNonlinearSSMContract(f"unknown repair_role: {role}")
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "covariance_floor", floor)
        object.__setattr__(self, "psd_repair", _nonempty_text(self.psd_repair, "psd_repair"))
        object.__setattr__(self, "symmetrize", bool(self.symmetrize))
        object.__setattr__(
            self,
            "logdet_convention",
            _nonempty_text(self.logdet_convention, "logdet_convention"),
        )
        object.__setattr__(
            self,
            "implemented_covariance",
            _nonempty_text(self.implemented_covariance, "implemented_covariance"),
        )
        object.__setattr__(self, "repair_role", role)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "jitter": self.jitter,
            "covariance_floor": self.covariance_floor,
            "psd_repair": self.psd_repair,
            "symmetrize": self.symmetrize,
            "logdet_convention": self.logdet_convention,
            "implemented_covariance": self.implemented_covariance,
            "repair_role": self.repair_role,
        }


@dataclass(frozen=True)
class NonlinearSSMAdapterContract:
    """BayesFilter-owned fail-closed nonlinear SSM posterior boundary."""

    parameter_names: tuple[str, ...]
    static_shape: NonlinearSSMStaticShape
    transform: ParameterTransformMetadata
    observation_semantics: ObservationSemantics
    regularization: RegularizationConvention
    value_score: ValueScoreCapability
    prior_term: str
    likelihood_term: str
    dtype: str
    backend: str
    filter_implementation: str
    compile_mode: CompileMode
    trace_policy: str
    return_filtered: bool
    seed_policy: SeedPolicy
    map_source: str | None = None
    mass_matrix_source: str | None = None
    hessian_source: str | None = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.parameter_names)
        if len(names) != self.static_shape.parameter_dim:
            raise InvalidNonlinearSSMContract(
                "parameter_names length must match parameter_dim"
            )
        if any(not name.strip() for name in names):
            raise InvalidNonlinearSSMContract("parameter_names must be nonempty")
        if len(set(names)) != len(names):
            raise InvalidNonlinearSSMContract("parameter_names must be unique")
        compile_mode = str(self.compile_mode)
        seed_policy = str(self.seed_policy)
        if compile_mode not in _KNOWN_COMPILE_MODES:
            raise InvalidNonlinearSSMContract(f"unknown compile_mode: {compile_mode}")
        if seed_policy not in _KNOWN_SEED_POLICIES:
            raise InvalidNonlinearSSMContract(f"unknown seed_policy: {seed_policy}")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "prior_term", _nonempty_text(self.prior_term, "prior_term"))
        object.__setattr__(
            self,
            "likelihood_term",
            _nonempty_text(self.likelihood_term, "likelihood_term"),
        )
        object.__setattr__(self, "dtype", _nonempty_text(self.dtype, "dtype"))
        object.__setattr__(self, "backend", _nonempty_text(self.backend, "backend"))
        object.__setattr__(
            self,
            "filter_implementation",
            _nonempty_text(self.filter_implementation, "filter_implementation"),
        )
        object.__setattr__(self, "compile_mode", compile_mode)
        object.__setattr__(
            self,
            "trace_policy",
            _nonempty_text(self.trace_policy, "trace_policy"),
        )
        object.__setattr__(self, "return_filtered", bool(self.return_filtered))
        object.__setattr__(self, "seed_policy", seed_policy)
        _reject_process_local_identity(self.signature_payload())

    @property
    def xla_hmc_ready(self) -> bool:
        return bool(
            self.compile_mode == "xla"
            and self.value_score.is_accepted_xla_hmc_authority
        )

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "parameter_names": self.parameter_names,
            "static_shape": self.static_shape.signature_payload(),
            "transform": self.transform.signature_payload(),
            "observation_semantics": self.observation_semantics.signature_payload(),
            "regularization": self.regularization.signature_payload(),
            "value_score": {
                "value_score_authority": self.value_score.value_score_authority,
                "xla_hmc_ready": self.value_score.xla_hmc_ready,
                "full_chain_xla_diagnostic_ready": (
                    self.value_score.full_chain_xla_diagnostic_ready
                ),
                "runtime_backend": self.value_score.runtime_backend,
                "target_scope": self.value_score.target_scope,
            },
            "prior_term": self.prior_term,
            "likelihood_term": self.likelihood_term,
            "dtype": self.dtype,
            "backend": self.backend,
            "filter_implementation": self.filter_implementation,
            "compile_mode": self.compile_mode,
            "trace_policy": self.trace_policy,
            "return_filtered": self.return_filtered,
            "seed_policy": self.seed_policy,
            "map_source": self.map_source,
            "mass_matrix_source": self.mass_matrix_source,
            "hessian_source": self.hessian_source,
        }


def value_score_capability(adapter: Any) -> ValueScoreCapability:
    """Return conservative value/score capability metadata for an adapter.

    Adapters must explicitly advertise XLA-ready authority. A plain
    ``log_prob_and_grad`` method is useful for eager/debug paths but is not by
    itself enough to authorize compiled HMC.
    """

    explicit = getattr(adapter, "value_score_capability", None)
    if explicit is not None:
        capability = explicit() if callable(explicit) else explicit
        return _coerce_capability(capability)
    if hasattr(adapter, "log_prob_and_grad") or hasattr(adapter, "target_log_prob_and_grad"):
        return ValueScoreCapability(
            value_score_authority="gradient_tape_fallback",
            xla_hmc_ready=False,
            runtime_backend="unknown",
            nonclaims=("value+score method without reviewed authority is eager/debug only",),
        )
    if hasattr(adapter, "log_prob"):
        return ValueScoreCapability(
            value_score_authority="unavailable",
            xla_hmc_ready=False,
            runtime_backend="unknown",
            nonclaims=("posterior value is available but score authority is unavailable",),
        )
    return ValueScoreCapability(
        value_score_authority="unavailable",
        xla_hmc_ready=False,
        runtime_backend="unknown",
        nonclaims=("adapter does not expose a posterior value/score contract",),
    )


def _coerce_capability(capability: Any) -> ValueScoreCapability:
    if isinstance(capability, ValueScoreCapability):
        return capability
    if isinstance(capability, Mapping):
        return ValueScoreCapability(
            value_score_authority=capability["value_score_authority"],
            xla_hmc_ready=capability.get("xla_hmc_ready", False),
            runtime_backend=capability.get("runtime_backend", "unknown"),
            evidence_path=capability.get("evidence_path"),
            target_scope=capability.get("target_scope"),
            nonclaims=tuple(capability.get("nonclaims", ())),
            full_chain_xla_diagnostic_ready=capability.get(
                "full_chain_xla_diagnostic_ready",
                False,
            ),
        )
    raise TypeError("value_score_capability must be a ValueScoreCapability or mapping")


def validate_nonlinear_ssm_contract(
    contract: NonlinearSSMAdapterContract,
    *,
    require_xla_hmc_ready: bool = False,
) -> NonlinearSSMAdapterContract:
    """Validate nonlinear SSM metadata and optionally require XLA HMC authority."""

    if not isinstance(contract, NonlinearSSMAdapterContract):
        raise TypeError("contract must be a NonlinearSSMAdapterContract")
    if require_xla_hmc_ready and not contract.xla_hmc_ready:
        raise InvalidNonlinearSSMContract(
            "XLA HMC requires graph-native or reviewed value/score authority"
        )
    return contract


def stable_nonlinear_ssm_program_signature(
    contract: NonlinearSSMAdapterContract,
) -> str:
    """Return a stable SHA-256 signature for nonlinear SSM program metadata."""

    payload = validate_nonlinear_ssm_contract(contract).signature_payload()
    blob = json.dumps(_normalize_for_json(payload), sort_keys=True, separators=(",", ":"))
    _reject_process_local_identity(blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _coerce_shape(shape: Any) -> tuple[int, ...]:
    values = tuple(int(dim) for dim in shape)
    if not values:
        raise InvalidNonlinearSSMContract("shape must be nonempty")
    if any(dim <= 0 for dim in values):
        raise InvalidNonlinearSSMContract("shape dimensions must be positive")
    return values


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size)


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value)
    if not text.strip():
        raise InvalidNonlinearSSMContract(f"{name} must be nonempty")
    return text


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_for_json(item) for item in value]
    return value


def _reject_process_local_identity(value: Any) -> None:
    text = json.dumps(_normalize_for_json(value), sort_keys=True, default=str)
    if any(pattern.search(text) for pattern in _PROCESS_LOCAL_SIGNATURE_PATTERNS):
        raise InvalidNonlinearSSMContract(
            "nonlinear SSM contract signatures must not contain process-local identity"
        )

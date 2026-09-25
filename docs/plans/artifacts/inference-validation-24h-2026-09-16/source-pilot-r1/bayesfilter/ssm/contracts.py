"""Generic Bayesian SSM metadata contracts.

The objects in this module describe target boundaries only.  They do not run a
filter, train a transport, or authorize HMC.  Their job is to give later phases
stable, fail-closed manifests for composing a Bayesian nonlinear SSM target.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


TargetCoordinateConvention = Literal["unconstrained", "constrained"]
LogJacobianConvention = Literal[
    "included_in_prior",
    "included_in_chart",
    "not_included",
]
PriorSupportPolicy = Literal[
    "enforced_by_transform",
    "finite_reject",
    "unbounded",
]
PriorLogDensityAuthority = Literal[
    "graph_native",
    "reviewed_external_adapter",
    "unavailable",
]
DeterministicTargetPolicy = Literal[
    "deterministic",
    "fixed_randomness",
    "stochastic_not_hmc_ready",
]
ApproximationSemantics = Literal[
    "exact",
    "deterministic_approximation",
    "fixed_randomness_approximation",
]

_KNOWN_TARGET_COORDINATES = {"unconstrained", "constrained"}
_KNOWN_LOG_JACOBIAN_CONVENTIONS = {
    "included_in_prior",
    "included_in_chart",
    "not_included",
}
_KNOWN_PRIOR_SUPPORT_POLICIES = {
    "enforced_by_transform",
    "finite_reject",
    "unbounded",
}
_KNOWN_PRIOR_AUTHORITIES = {
    "graph_native",
    "reviewed_external_adapter",
    "unavailable",
}
_KNOWN_DETERMINISTIC_TARGET_POLICIES = {
    "deterministic",
    "fixed_randomness",
    "stochastic_not_hmc_ready",
}
_KNOWN_APPROXIMATION_SEMANTICS = {
    "exact",
    "deterministic_approximation",
    "fixed_randomness_approximation",
}
_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


class InvalidSSMContract(ValueError):
    """Raised when generic SSM metadata is incomplete or unsafe to persist."""


@dataclass(frozen=True)
class SSMStaticShape:
    """Fixed SSM dimensions required by compiled target and HMC paths."""

    horizon: int
    state_dim: int
    observation_dim: int
    innovation_dim: int
    parameter_dim: int

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
                raise InvalidSSMContract(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "observation_dim": self.observation_dim,
            "innovation_dim": self.innovation_dim,
            "parameter_dim": self.parameter_dim,
        }


@dataclass(frozen=True)
class SSMDataSignature:
    """Stable observation/data identity for a Bayesian SSM target."""

    dataset_id: str
    observation_shape: tuple[int, ...]
    mask_shape: tuple[int, ...] | None = None
    data_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _nonempty_text(self.dataset_id, "dataset_id"))
        object.__setattr__(
            self,
            "observation_shape",
            _coerce_shape(self.observation_shape, "observation_shape"),
        )
        if self.mask_shape is not None:
            object.__setattr__(
                self,
                "mask_shape",
                _coerce_shape(self.mask_shape, "mask_shape"),
            )
        if self.data_hash is not None:
            object.__setattr__(self, "data_hash", _nonempty_text(self.data_hash, "data_hash"))
        _reject_process_local_identity(self.manifest_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "observation_shape": self.observation_shape,
            "mask_shape": self.mask_shape,
            "data_hash": self.data_hash,
        }


@dataclass(frozen=True)
class BayesianSSMProblem:
    """Stable identity for one Bayesian nonlinear SSM estimation target."""

    problem_id: str
    static_shape: SSMStaticShape
    data_signature: SSMDataSignature
    target_coordinate_convention: TargetCoordinateConvention
    model_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        coordinate = str(self.target_coordinate_convention)
        if coordinate not in _KNOWN_TARGET_COORDINATES:
            raise InvalidSSMContract(
                f"unknown target_coordinate_convention: {coordinate}"
            )
        manifest = _coerce_manifest(
            self.model_manifest,
            "model_manifest",
            required_keys=("model_id", "model_hash"),
        )
        _require_matching_hash(manifest, "model_hash")
        object.__setattr__(self, "problem_id", _nonempty_text(self.problem_id, "problem_id"))
        object.__setattr__(self, "target_coordinate_convention", coordinate)
        object.__setattr__(self, "model_manifest", manifest)
        if self.data_signature.observation_shape[0] != self.static_shape.horizon:
            raise InvalidSSMContract(
                "data_signature observation horizon must match static_shape horizon"
            )
        _reject_process_local_identity(self.manifest_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "problem_id": self.problem_id,
            "static_shape": self.static_shape.manifest_payload(),
            "data_signature": self.data_signature.manifest_payload(),
            "target_coordinate_convention": self.target_coordinate_convention,
            "model_manifest": self.model_manifest,
        }


@dataclass(frozen=True)
class ParameterChart:
    """Parameter coordinate chart used to form unconstrained HMC targets."""

    parameter_names: tuple[str, ...]
    unconstrained_dim: int
    constrained_shape: tuple[int, ...]
    transform_manifest: Mapping[str, Any]
    log_jacobian_convention: LogJacobianConvention

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.parameter_names)
        if not names:
            raise InvalidSSMContract("parameter_names must be nonempty")
        if any(not name.strip() for name in names):
            raise InvalidSSMContract("parameter_names must be nonempty")
        if len(set(names)) != len(names):
            raise InvalidSSMContract("parameter_names must be unique")
        unconstrained_dim = int(self.unconstrained_dim)
        if unconstrained_dim <= 0:
            raise InvalidSSMContract("unconstrained_dim must be positive")
        if len(names) != unconstrained_dim:
            raise InvalidSSMContract(
                "parameter_names length must match unconstrained_dim"
            )
        constrained_shape = _coerce_shape(self.constrained_shape, "constrained_shape")
        if _shape_size(constrained_shape) != unconstrained_dim:
            raise InvalidSSMContract(
                "constrained_shape size must match unconstrained_dim"
            )
        convention = str(self.log_jacobian_convention)
        if convention not in _KNOWN_LOG_JACOBIAN_CONVENTIONS:
            raise InvalidSSMContract(f"unknown log_jacobian_convention: {convention}")
        manifest = _coerce_manifest(
            self.transform_manifest,
            "transform_manifest",
            required_keys=("transform_id", "transform_hash"),
        )
        _require_matching_hash(manifest, "transform_hash")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "unconstrained_dim", unconstrained_dim)
        object.__setattr__(self, "constrained_shape", constrained_shape)
        object.__setattr__(self, "log_jacobian_convention", convention)
        object.__setattr__(self, "transform_manifest", manifest)
        _reject_process_local_identity(self.manifest_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "parameter_names": self.parameter_names,
            "unconstrained_dim": self.unconstrained_dim,
            "constrained_shape": self.constrained_shape,
            "transform_manifest": self.transform_manifest,
            "log_jacobian_convention": self.log_jacobian_convention,
        }


@dataclass(frozen=True)
class ParameterPrior:
    """Prior metadata and log-density authority for an SSM target."""

    prior_manifest: Mapping[str, Any]
    support_policy: PriorSupportPolicy
    log_density_authority: PriorLogDensityAuthority

    def __post_init__(self) -> None:
        support_policy = str(self.support_policy)
        if support_policy not in _KNOWN_PRIOR_SUPPORT_POLICIES:
            raise InvalidSSMContract(f"unknown prior support_policy: {support_policy}")
        authority = str(self.log_density_authority)
        if authority not in _KNOWN_PRIOR_AUTHORITIES:
            raise InvalidSSMContract(f"unknown prior log_density_authority: {authority}")
        manifest = _coerce_manifest(
            self.prior_manifest,
            "prior_manifest",
            required_keys=("prior_id", "prior_hash"),
        )
        _require_matching_hash(manifest, "prior_hash")
        object.__setattr__(self, "support_policy", support_policy)
        object.__setattr__(self, "log_density_authority", authority)
        object.__setattr__(self, "prior_manifest", manifest)
        _reject_process_local_identity(self.manifest_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "prior_manifest": self.prior_manifest,
            "support_policy": self.support_policy,
            "log_density_authority": self.log_density_authority,
        }


@dataclass(frozen=True)
class FilterProgram:
    """Filter-program identity and target-authority metadata."""

    filter_id: str
    required_model_capabilities: tuple[str, ...]
    deterministic_target_policy: DeterministicTargetPolicy
    approximation_semantics: ApproximationSemantics
    filter_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        capabilities = tuple(
            _nonempty_text(item, "required_model_capability")
            for item in self.required_model_capabilities
        )
        if not capabilities:
            raise InvalidSSMContract("required_model_capabilities must be nonempty")
        if len(set(capabilities)) != len(capabilities):
            raise InvalidSSMContract("required_model_capabilities must be unique")
        target_policy = str(self.deterministic_target_policy)
        if target_policy not in _KNOWN_DETERMINISTIC_TARGET_POLICIES:
            raise InvalidSSMContract(
                f"unknown deterministic_target_policy: {target_policy}"
            )
        semantics = str(self.approximation_semantics)
        if semantics not in _KNOWN_APPROXIMATION_SEMANTICS:
            raise InvalidSSMContract(f"unknown approximation_semantics: {semantics}")
        if target_policy == "stochastic_not_hmc_ready" and semantics == "exact":
            raise InvalidSSMContract(
                "stochastic_not_hmc_ready filter programs cannot claim exact semantics"
            )
        manifest = _coerce_manifest(
            self.filter_manifest,
            "filter_manifest",
            required_keys=("filter_id", "filter_hash"),
        )
        _require_matching_hash(manifest, "filter_hash")
        filter_id = _nonempty_text(self.filter_id, "filter_id")
        if manifest["filter_id"] != filter_id:
            raise InvalidSSMContract("filter_manifest filter_id must match filter_id")
        object.__setattr__(self, "filter_id", filter_id)
        object.__setattr__(self, "required_model_capabilities", capabilities)
        object.__setattr__(self, "deterministic_target_policy", target_policy)
        object.__setattr__(self, "approximation_semantics", semantics)
        object.__setattr__(self, "filter_manifest", manifest)
        _reject_process_local_identity(self.manifest_payload())

    @property
    def hmc_target_ready(self) -> bool:
        return self.deterministic_target_policy in {"deterministic", "fixed_randomness"}

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "filter_id": self.filter_id,
            "required_model_capabilities": self.required_model_capabilities,
            "deterministic_target_policy": self.deterministic_target_policy,
            "approximation_semantics": self.approximation_semantics,
            "filter_manifest": self.filter_manifest,
        }


@dataclass(frozen=True)
class FrozenTransportBinding:
    """Signature binding for a frozen NeuTra transport artifact."""

    transport_id: str
    dimension: int
    target_signature: str
    log_jacobian_available: bool
    transport_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        dimension = int(self.dimension)
        if dimension <= 0:
            raise InvalidSSMContract("transport dimension must be positive")
        manifest = _coerce_manifest(
            self.transport_manifest,
            "transport_manifest",
            required_keys=("transport_id", "transport_hash", "target_signature"),
        )
        _require_matching_hash(manifest, "transport_hash")
        transport_id = _nonempty_text(self.transport_id, "transport_id")
        target_signature = _nonempty_text(self.target_signature, "target_signature")
        if manifest["transport_id"] != transport_id:
            raise InvalidSSMContract(
                "transport_manifest transport_id must match transport_id"
            )
        if manifest["target_signature"] != target_signature:
            raise InvalidSSMContract(
                "transport_manifest target_signature must match target_signature"
            )
        object.__setattr__(self, "transport_id", transport_id)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "target_signature", target_signature)
        object.__setattr__(
            self,
            "log_jacobian_available",
            bool(self.log_jacobian_available),
        )
        object.__setattr__(self, "transport_manifest", manifest)
        _reject_process_local_identity(self.manifest_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "transport_id": self.transport_id,
            "dimension": self.dimension,
            "target_signature": self.target_signature,
            "log_jacobian_available": self.log_jacobian_available,
            "transport_manifest": self.transport_manifest,
        }


@dataclass(frozen=True)
class SSMTargetContract:
    """Complete generic SSM target boundary assembled from Phase 1 surfaces."""

    problem: BayesianSSMProblem
    chart: ParameterChart
    prior: ParameterPrior
    filter_program: FilterProgram
    frozen_transport: FrozenTransportBinding | None = None

    def __post_init__(self) -> None:
        if self.chart.unconstrained_dim != self.problem.static_shape.parameter_dim:
            raise InvalidSSMContract(
                "chart unconstrained_dim must match problem parameter_dim"
            )
        if self.frozen_transport is not None:
            target_signature = stable_ssm_target_signature(
                SSMTargetContract(
                    problem=self.problem,
                    chart=self.chart,
                    prior=self.prior,
                    filter_program=self.filter_program,
                    frozen_transport=None,
                )
            )
            if self.frozen_transport.dimension != self.chart.unconstrained_dim:
                raise InvalidSSMContract(
                    "frozen transport dimension must match chart unconstrained_dim"
                )
            if self.frozen_transport.target_signature != target_signature:
                raise InvalidSSMContract(
                    "frozen transport target_signature must match untransported target"
                )
        _reject_process_local_identity(self.manifest_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "problem": self.problem.manifest_payload(),
            "chart": self.chart.manifest_payload(),
            "prior": self.prior.manifest_payload(),
            "filter_program": self.filter_program.manifest_payload(),
            "frozen_transport": (
                None
                if self.frozen_transport is None
                else self.frozen_transport.manifest_payload()
            ),
        }


def stable_problem_signature(problem: BayesianSSMProblem) -> str:
    """Return a stable SHA-256 signature for one SSM problem identity."""

    if not isinstance(problem, BayesianSSMProblem):
        raise TypeError("problem must be a BayesianSSMProblem")
    return _stable_hash(problem.manifest_payload())


def stable_parameter_chart_signature(chart: ParameterChart) -> str:
    """Return a stable SHA-256 signature for one parameter chart."""

    if not isinstance(chart, ParameterChart):
        raise TypeError("chart must be a ParameterChart")
    return _stable_hash(chart.manifest_payload())


def stable_prior_signature(prior: ParameterPrior) -> str:
    """Return a stable SHA-256 signature for one prior contract."""

    if not isinstance(prior, ParameterPrior):
        raise TypeError("prior must be a ParameterPrior")
    return _stable_hash(prior.manifest_payload())


def stable_filter_program_signature(filter_program: FilterProgram) -> str:
    """Return a stable SHA-256 signature for one filter program."""

    if not isinstance(filter_program, FilterProgram):
        raise TypeError("filter_program must be a FilterProgram")
    return _stable_hash(filter_program.manifest_payload())


def stable_frozen_transport_signature(binding: FrozenTransportBinding) -> str:
    """Return a stable SHA-256 signature for one frozen transport binding."""

    if not isinstance(binding, FrozenTransportBinding):
        raise TypeError("binding must be a FrozenTransportBinding")
    return _stable_hash(binding.manifest_payload())


def stable_ssm_target_signature(contract: SSMTargetContract) -> str:
    """Return a stable SHA-256 signature for a generic SSM target boundary."""

    if not isinstance(contract, SSMTargetContract):
        raise TypeError("contract must be an SSMTargetContract")
    return _stable_hash(contract.manifest_payload())


def validate_ssm_target_contract(
    contract: SSMTargetContract,
    *,
    require_filter_hmc_target_ready: bool = False,
    require_frozen_transport: bool = False,
) -> SSMTargetContract:
    """Validate a generic SSM target boundary and optional phase gates."""

    if not isinstance(contract, SSMTargetContract):
        raise TypeError("contract must be an SSMTargetContract")
    if require_filter_hmc_target_ready and not contract.filter_program.hmc_target_ready:
        raise InvalidSSMContract(
            "filter program must be deterministic or fixed-randomness for HMC target use"
        )
    if require_frozen_transport and contract.frozen_transport is None:
        raise InvalidSSMContract("frozen transport binding is required")
    return contract


def _coerce_shape(shape: Any, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise InvalidSSMContract(f"{name} must be a shape sequence") from exc
    if not values:
        raise InvalidSSMContract(f"{name} must be nonempty")
    if any(dim <= 0 for dim in values):
        raise InvalidSSMContract(f"{name} dimensions must be positive")
    return values


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size)


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value)
    if not text.strip():
        raise InvalidSSMContract(f"{name} must be nonempty")
    return text


def _coerce_manifest(
    manifest: Mapping[str, Any],
    name: str,
    *,
    required_keys: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise InvalidSSMContract(f"{name} must be a mapping")
    normalized = _normalize_for_json(manifest)
    if not isinstance(normalized, Mapping):
        raise InvalidSSMContract(f"{name} must normalize to a mapping")
    for key in required_keys:
        if key not in normalized:
            raise InvalidSSMContract(f"{name} missing required key {key}")
        if not str(normalized[key]).strip():
            raise InvalidSSMContract(f"{name} key {key} must be nonempty")
    _reject_process_local_identity(normalized)
    return normalized


def _require_matching_hash(manifest: Mapping[str, Any], key: str) -> None:
    if key not in manifest:
        raise InvalidSSMContract(f"manifest missing required key {key}")
    _nonempty_text(manifest[key], key)


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_for_json(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_for_json(item) for item in value]
    return value


def _stable_hash(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_for_json(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    _reject_process_local_identity(blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _reject_process_local_identity(value: Any) -> None:
    text = json.dumps(_normalize_for_json(value), sort_keys=True, default=str)
    if any(pattern.search(text) for pattern in _PROCESS_LOCAL_SIGNATURE_PATTERNS):
        raise InvalidSSMContract(
            "SSM contract manifests must not contain process-local identity"
        )

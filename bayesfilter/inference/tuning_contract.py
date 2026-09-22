"""Canonical interface and route-role contracts for HMC tuning.

The registry classifies public orchestration entry points. Internal stage
helpers remain implementation details and do not become additional active
interfaces merely because compatibility modules re-export them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


TUNING_CONTRACT_SCHEMA = "bayesfilter.hmc_tuning_contract.v1"
TUNING_ROUTE_REGISTRY_SCHEMA = "bayesfilter.hmc_tuning_route_registry.v2"
HMC_TUNING_CAPABILITY_SCHEMA = "bayesfilter.hmc_tuning_capability.v2"
HMC_TUNING_CAPABILITY_REGISTRY_SCHEMA = (
    "bayesfilter.hmc_tuning_capability_registry.v2"
)
HMC_TUNING_RUNNER_BINDING_SCHEMA = "bayesfilter.hmc_tuning_runner_binding.v2"
HMC_TUNING_ORDINARY_RHAT_THRESHOLD = 1.01

TuningRouteRole = Literal["active", "historical", "diagnostic"]
HMCInterfaceKind = Literal[
    "public_tuner",
    "diagnostic_helper",
    "historical_helper",
    "stage_helper",
    "chain_runner",
    "runner_binding_factory",
]
HMCCapabilityStatus = Literal[
    "tested_supported",
    "internal_only",
    "diagnostic_only",
    "historical_only",
]
HMCOwnershipCapability = Literal["owned", "fixed", "not_owned", "not_authoritative"]


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    items = tuple(_nonempty_text(item, name) for item in value)
    if not items:
        raise ValueError(f"{name} must be non-empty")
    return items


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported runner-binding payload value: {type(value).__name__}")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HMCTuningScope:
    """Identity fields that prevent cross-target tuning reuse."""

    target_scope: str
    adapter_signature: str
    coordinate_signature: str
    parameter_dimension: int
    backend: str
    dtype: str
    xla_enabled: bool
    chain_execution_mode: str
    transport_signature: str | None = None
    maximum_candidate_step_size: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "target_scope",
            "adapter_signature",
            "coordinate_signature",
            "backend",
            "dtype",
            "chain_execution_mode",
        ):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        dimension = int(self.parameter_dimension)
        if dimension <= 0:
            raise ValueError("parameter_dimension must be positive")
        object.__setattr__(self, "parameter_dimension", dimension)
        object.__setattr__(self, "xla_enabled", bool(self.xla_enabled))
        if self.transport_signature is not None:
            signature = str(self.transport_signature)
            if not signature:
                raise ValueError("transport_signature must be non-empty when provided")
            object.__setattr__(self, "transport_signature", signature)
        if self.maximum_candidate_step_size is not None:
            cap = float(self.maximum_candidate_step_size)
            if not math.isfinite(cap) or cap <= 0.0:
                raise ValueError("maximum_candidate_step_size must be finite and positive")
            object.__setattr__(self, "maximum_candidate_step_size", cap)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": TUNING_CONTRACT_SCHEMA,
            "target_scope": self.target_scope,
            "adapter_signature": self.adapter_signature,
            "coordinate_signature": self.coordinate_signature,
            "transport_signature": self.transport_signature,
            "maximum_candidate_step_size": self.maximum_candidate_step_size,
            "parameter_dimension": self.parameter_dimension,
            "backend": self.backend,
            "dtype": self.dtype,
            "xla_enabled": self.xla_enabled,
            "chain_execution_mode": self.chain_execution_mode,
        }


@dataclass(frozen=True)
class HMCTuningRouteRecord:
    """One repository-owned route classification."""

    interface_name: str
    module: str
    role: TuningRouteRole
    artifact_authority: bool
    replacement: str | None = None
    nonclaims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.interface_name)
        module = str(self.module)
        role = str(self.role)
        if not name or not module:
            raise ValueError("route interface_name and module must be non-empty")
        if role not in {"active", "historical", "diagnostic"}:
            raise ValueError("route role is invalid")
        replacement = None if self.replacement is None else str(self.replacement)
        if role == "active":
            if replacement is not None:
                raise ValueError("active routes cannot name a replacement")
            if not self.artifact_authority:
                raise ValueError("active routes must own canonical artifact authority")
        else:
            if not replacement:
                raise ValueError("non-active routes must name an active replacement")
            if self.artifact_authority:
                raise ValueError("non-active routes cannot own canonical artifact authority")
        object.__setattr__(self, "interface_name", name)
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "artifact_authority", bool(self.artifact_authority))
        object.__setattr__(self, "replacement", replacement)
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.interface_name}"

    def payload(self) -> Mapping[str, Any]:
        return {
            "interface_name": self.interface_name,
            "module": self.module,
            "qualified_name": self.qualified_name,
            "role": self.role,
            "artifact_authority": self.artifact_authority,
            "replacement": self.replacement,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class HMCTuningInterfaceCapability:
    """Executable documentation contract for one tuning-related interface."""

    interface_name: str
    module: str
    interface_kind: HMCInterfaceKind
    capability_status: HMCCapabilityStatus
    artifact_authority: bool
    algorithm_family: str
    target_contract: str
    coordinate_prerequisite: str
    mass_policy: str
    step_size_policy: str
    trajectory_policy: str
    fresh_verification_policy: str
    ess_admission_policy: str
    target_status_telemetry: str
    runner_injection_policy: str
    identity_bindings: tuple[str, ...]
    source_closure_policy: str
    replacement: str | None
    forbidden_uses: tuple[str, ...]
    nonclaims: tuple[str, ...]
    evidence_anchors: tuple[str, ...]
    mass_capability: HMCOwnershipCapability = "not_owned"
    step_size_capability: HMCOwnershipCapability = "not_owned"
    trajectory_capability: HMCOwnershipCapability = "not_owned"
    requires_frozen_transport: bool = False
    fresh_verification_required: bool = False
    acceptance_alone_can_handoff: bool = False
    bare_runner_injection_allowed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "interface_name",
            "module",
            "interface_kind",
            "capability_status",
            "algorithm_family",
            "target_contract",
            "coordinate_prerequisite",
            "mass_policy",
            "step_size_policy",
            "trajectory_policy",
            "fresh_verification_policy",
            "ess_admission_policy",
            "target_status_telemetry",
            "runner_injection_policy",
            "source_closure_policy",
        ):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        if self.interface_kind not in {
            "public_tuner",
            "diagnostic_helper",
            "historical_helper",
            "stage_helper",
            "chain_runner",
            "runner_binding_factory",
        }:
            raise ValueError("interface_kind is invalid")
        if self.capability_status not in {
            "tested_supported",
            "internal_only",
            "diagnostic_only",
            "historical_only",
        }:
            raise ValueError("capability_status is invalid")
        for name in (
            "mass_capability",
            "step_size_capability",
            "trajectory_capability",
        ):
            value = str(getattr(self, name))
            if value not in {"owned", "fixed", "not_owned", "not_authoritative"}:
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, value)
        for name in (
            "requires_frozen_transport",
            "fresh_verification_required",
            "acceptance_alone_can_handoff",
            "bare_runner_injection_allowed",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "artifact_authority", bool(self.artifact_authority))
        replacement = None if self.replacement is None else _nonempty_text(
            self.replacement, "replacement"
        )
        object.__setattr__(self, "replacement", replacement)
        object.__setattr__(
            self,
            "identity_bindings",
            _string_tuple(self.identity_bindings, "identity_bindings"),
        )
        object.__setattr__(
            self,
            "forbidden_uses",
            _string_tuple(self.forbidden_uses, "forbidden_uses"),
        )
        object.__setattr__(
            self,
            "nonclaims",
            _string_tuple(self.nonclaims, "nonclaims"),
        )
        anchors = _string_tuple(self.evidence_anchors, "evidence_anchors")
        object.__setattr__(self, "evidence_anchors", anchors)
        if self.interface_kind == "public_tuner":
            if self.capability_status == "tested_supported":
                if not self.artifact_authority or replacement is not None:
                    raise ValueError(
                        "supported public tuners must own authority and have no replacement"
                    )
                if not self.fresh_verification_required:
                    raise ValueError("supported public tuners require fresh verification")
                if self.acceptance_alone_can_handoff:
                    raise ValueError("acceptance alone cannot authorize a public handoff")
                if self.bare_runner_injection_allowed:
                    raise ValueError("supported public tuners cannot accept bare runners")
            elif self.artifact_authority or replacement is None:
                raise ValueError(
                    "non-active public tuners need a replacement and cannot own authority"
                )
        elif self.artifact_authority:
            raise ValueError("only supported public tuners may own artifact authority")
        if self.interface_kind in {"diagnostic_helper", "historical_helper"}:
            expected_status = (
                "diagnostic_only"
                if self.interface_kind == "diagnostic_helper"
                else "historical_only"
            )
            if self.capability_status != expected_status or replacement is None:
                raise ValueError(
                    "diagnostic and historical helpers require their matching "
                    "status and an active replacement"
                )
        if self.capability_status == "tested_supported" and any(
            value == "unknown"
            for value in (
                self.target_contract,
                self.coordinate_prerequisite,
                self.mass_policy,
                self.step_size_policy,
                self.trajectory_policy,
                self.fresh_verification_policy,
                self.target_status_telemetry,
                self.runner_injection_policy,
                self.source_closure_policy,
            )
        ):
            raise ValueError("tested-supported capabilities cannot contain unknown fields")

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.interface_name}"

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": HMC_TUNING_CAPABILITY_SCHEMA,
            "interface_name": self.interface_name,
            "module": self.module,
            "qualified_name": self.qualified_name,
            "interface_kind": self.interface_kind,
            "capability_status": self.capability_status,
            "artifact_authority": self.artifact_authority,
            "mass_capability": self.mass_capability,
            "step_size_capability": self.step_size_capability,
            "trajectory_capability": self.trajectory_capability,
            "requires_frozen_transport": self.requires_frozen_transport,
            "fresh_verification_required": self.fresh_verification_required,
            "acceptance_alone_can_handoff": self.acceptance_alone_can_handoff,
            "bare_runner_injection_allowed": self.bare_runner_injection_allowed,
            "algorithm_family": self.algorithm_family,
            "target_contract": self.target_contract,
            "coordinate_prerequisite": self.coordinate_prerequisite,
            "mass_policy": self.mass_policy,
            "step_size_policy": self.step_size_policy,
            "trajectory_policy": self.trajectory_policy,
            "fresh_verification_policy": self.fresh_verification_policy,
            "ess_admission_policy": self.ess_admission_policy,
            "target_status_telemetry": self.target_status_telemetry,
            "runner_injection_policy": self.runner_injection_policy,
            "identity_bindings": self.identity_bindings,
            "source_closure_policy": self.source_closure_policy,
            "replacement": self.replacement,
            "forbidden_uses": self.forbidden_uses,
            "nonclaims": self.nonclaims,
            "evidence_anchors": self.evidence_anchors,
        }


_RUNNER_BINDING_ISSUER_TOKEN = object()


@dataclass(frozen=True)
class HMCTuningRunnerBinding:
    """Repository-issued chain runner accepted by the ordinary public tuner.

    The binding is callable so existing internal stages can use their common
    chain-runner protocol. It validates the scope before execution and validates
    identity and required telemetry on the returned result.
    """

    _issuer_token: Any = field(repr=False, compare=False)
    runner: Callable[[Any, Any, Any], Any] = field(repr=False, compare=False)
    tensor_kernel_factory: Callable[..., Any] = field(repr=False, compare=False)
    runner_identity: str
    algorithm_family: str
    target_scope: str
    coordinate_scope: str
    force_identity: str
    force_semantics: str
    endpoint_target_identity: str
    endpoint_target_coordinate_system: str
    endpoint_target_includes_chart_log_jacobian: bool
    affine_log_jacobian_convention: str
    target_status_evidence: str
    supported_target_status_trace_policies: tuple[str, ...]
    supported_chain_execution_modes: tuple[str, ...]
    backend: str
    dtype: str
    xla_capable: bool
    requires_native_affine_mass: bool
    supports_arbitrary_fixed_step_size: bool
    supports_arbitrary_fixed_leapfrog_count: bool
    required_diagnostic_fields: tuple[str, ...]
    required_metadata_fields: tuple[str, ...]
    source_dependency_closure: Mapping[str, Any]
    artifact_authority: bool = False

    def __post_init__(self) -> None:
        if self._issuer_token is not _RUNNER_BINDING_ISSUER_TOKEN:
            raise ValueError("HMC tuning runner bindings must be repository-issued")
        if not callable(self.runner):
            raise TypeError("runner must be callable")
        if not callable(self.tensor_kernel_factory):
            raise TypeError("tensor_kernel_factory must be callable")
        for name in (
            "runner_identity",
            "algorithm_family",
            "target_scope",
            "coordinate_scope",
            "force_identity",
            "force_semantics",
            "endpoint_target_identity",
            "endpoint_target_coordinate_system",
            "affine_log_jacobian_convention",
            "target_status_evidence",
            "backend",
            "dtype",
        ):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        status_policies = _string_tuple(
            self.supported_target_status_trace_policies,
            "supported_target_status_trace_policies",
        )
        if any(item not in {"none", "per_chain_step"} for item in status_policies):
            raise ValueError("unsupported target-status trace policy in runner binding")
        object.__setattr__(
            self,
            "supported_target_status_trace_policies",
            status_policies,
        )
        modes = _string_tuple(
            self.supported_chain_execution_modes,
            "supported_chain_execution_modes",
        )
        if any(item not in {"tf_function", "eager"} for item in modes):
            raise ValueError("unsupported chain execution mode in runner binding")
        object.__setattr__(self, "supported_chain_execution_modes", modes)
        object.__setattr__(
            self,
            "required_diagnostic_fields",
            _string_tuple(self.required_diagnostic_fields, "required_diagnostic_fields"),
        )
        object.__setattr__(
            self,
            "required_metadata_fields",
            _string_tuple(self.required_metadata_fields, "required_metadata_fields"),
        )
        closure = _json_ready(self.source_dependency_closure)
        if not isinstance(closure, Mapping) or not closure:
            raise ValueError("source_dependency_closure must be a non-empty mapping")
        object.__setattr__(self, "source_dependency_closure", closure)
        object.__setattr__(
            self,
            "endpoint_target_includes_chart_log_jacobian",
            bool(self.endpoint_target_includes_chart_log_jacobian),
        )
        if self.endpoint_target_coordinate_system != self.coordinate_scope:
            raise ValueError("binding target and force coordinate scopes must match")
        if self.affine_log_jacobian_convention != "constant_omitted":
            raise ValueError(
                "ordinary affine HMC bindings require constant_omitted convention"
            )
        for name in (
            "xla_capable",
            "requires_native_affine_mass",
            "supports_arbitrary_fixed_step_size",
            "supports_arbitrary_fixed_leapfrog_count",
            "artifact_authority",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.artifact_authority:
            raise ValueError("a chain runner binding cannot issue tuning authority")
        if not self.requires_native_affine_mass:
            raise ValueError("ordinary public runner bindings must require affine mass")
        if not (
            self.supports_arbitrary_fixed_step_size
            and self.supports_arbitrary_fixed_leapfrog_count
        ):
            raise ValueError("bound runners must accept every fixed epsilon and L")

    def _identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema": HMC_TUNING_RUNNER_BINDING_SCHEMA,
            "runner_identity": self.runner_identity,
            "algorithm_family": self.algorithm_family,
            "target_scope": self.target_scope,
            "coordinate_scope": self.coordinate_scope,
            "force_identity": self.force_identity,
            "force_semantics": self.force_semantics,
            "endpoint_target_identity": self.endpoint_target_identity,
            "endpoint_target_coordinate_system": (
                self.endpoint_target_coordinate_system
            ),
            "endpoint_target_includes_chart_log_jacobian": (
                self.endpoint_target_includes_chart_log_jacobian
            ),
            "affine_log_jacobian_convention": self.affine_log_jacobian_convention,
            "tensor_kernel_factory_available": True,
            "target_status_evidence": self.target_status_evidence,
            "supported_target_status_trace_policies": (
                self.supported_target_status_trace_policies
            ),
            "supported_chain_execution_modes": self.supported_chain_execution_modes,
            "backend": self.backend,
            "dtype": self.dtype,
            "xla_capable": self.xla_capable,
            "requires_native_affine_mass": self.requires_native_affine_mass,
            "supports_arbitrary_fixed_step_size": (
                self.supports_arbitrary_fixed_step_size
            ),
            "supports_arbitrary_fixed_leapfrog_count": (
                self.supports_arbitrary_fixed_leapfrog_count
            ),
            "required_diagnostic_fields": self.required_diagnostic_fields,
            "required_metadata_fields": self.required_metadata_fields,
            "source_dependency_closure": self.source_dependency_closure,
            "artifact_authority": self.artifact_authority,
        }

    @property
    def binding_hash(self) -> str:
        return _stable_hash(self._identity_payload())

    def payload(self) -> Mapping[str, Any]:
        return {**self._identity_payload(), "binding_hash": self.binding_hash}

    def validate_public_context(
        self,
        *,
        target_scope: str,
        target_status_trace_policy: str,
        chain_execution_mode: str,
        use_xla: bool,
    ) -> None:
        if str(target_scope) != self.target_scope:
            raise ValueError("runner binding target_scope mismatch")
        if target_status_trace_policy not in self.supported_target_status_trace_policies:
            raise ValueError("runner binding does not support target-status trace policy")
        if chain_execution_mode not in self.supported_chain_execution_modes:
            raise ValueError("runner binding does not support chain execution mode")
        if bool(use_xla) and not self.xla_capable:
            raise ValueError("runner binding is not XLA capable")

    def __call__(self, adapter: Any, initial_state: Any, config: Any) -> Any:
        self.validate_public_context(
            target_scope=_nonempty_text(getattr(config, "target_scope", ""), "target_scope"),
            target_status_trace_policy=str(
                getattr(config, "target_status_trace_policy", "")
            ),
            chain_execution_mode=str(getattr(config, "chain_execution_mode", "")),
            use_xla=bool(getattr(config, "use_xla", False)),
        )
        if self.requires_native_affine_mass:
            if not callable(getattr(adapter, "latent_to_position", None)):
                raise ValueError("runner binding requires native affine latent_to_position")
            transform = getattr(adapter, "transform", None)
            if transform is None or not hasattr(transform, "factor") or not hasattr(
                transform, "center"
            ):
                raise ValueError("runner binding requires native affine mass transform")
        result = self.runner(adapter, initial_state, config)
        diagnostics = getattr(result, "diagnostics", None)
        metadata = getattr(result, "metadata", None)
        if not isinstance(diagnostics, Mapping):
            raise ValueError("bound runner result lacks diagnostics mapping")
        if not isinstance(metadata, Mapping):
            raise ValueError("bound runner result lacks metadata mapping")
        missing_diagnostics = tuple(
            name for name in self.required_diagnostic_fields if name not in diagnostics
        )
        if missing_diagnostics:
            raise ValueError(
                "bound runner result lacks required telemetry: "
                + ", ".join(missing_diagnostics)
            )
        missing_metadata = tuple(
            name for name in self.required_metadata_fields if name not in metadata
        )
        if missing_metadata:
            raise ValueError(
                "bound runner result lacks required identity: "
                + ", ".join(missing_metadata)
            )
        if metadata.get("coordinate_route") != "native_fixed_mass_affine":
            raise ValueError("bound runner used forbidden direct identity-mass fallback")
        if str(metadata.get("force_identity")) != self.force_identity:
            raise ValueError("bound runner force identity mismatch")
        if str(metadata.get("force_semantics")) != self.force_semantics:
            raise ValueError("bound runner force semantics mismatch")
        if str(metadata.get("target_identity")) != self.endpoint_target_identity:
            raise ValueError("bound runner endpoint-target identity mismatch")
        if (
            str(metadata.get("target_coordinate_system"))
            != self.endpoint_target_coordinate_system
        ):
            raise ValueError("bound runner endpoint-target coordinates mismatch")
        if bool(metadata.get("target_includes_chart_log_jacobian")) != (
            self.endpoint_target_includes_chart_log_jacobian
        ):
            raise ValueError("bound runner endpoint-target Jacobian status mismatch")
        if str(metadata.get("affine_log_jacobian_convention")) != (
            self.affine_log_jacobian_convention
        ):
            raise ValueError("bound runner affine log-Jacobian convention mismatch")
        if str(metadata.get("target_scope")) != self.target_scope:
            raise ValueError("bound runner result target_scope mismatch")
        if str(metadata.get("chain_execution_mode")) not in (
            self.supported_chain_execution_modes
        ):
            raise ValueError("bound runner result chain execution mode mismatch")
        if getattr(result, "samples", None) is None:
            raise ValueError("bound runner result lacks samples for movement checks")
        return result

    def build_tensor_kernel(
        self,
        adapter: Any,
        *,
        step_size: Any,
        num_leapfrog_steps: Any,
        target_scope: str,
        target_status_trace_policy: str,
        chain_execution_mode: str,
        use_xla: bool,
    ) -> Any:
        """Build the repository-owned bound kernel without a host chain runner."""

        self.validate_public_context(
            target_scope=target_scope,
            target_status_trace_policy=target_status_trace_policy,
            chain_execution_mode=chain_execution_mode,
            use_xla=use_xla,
        )
        if not callable(getattr(adapter, "latent_to_position", None)):
            raise ValueError("runner binding requires native affine latent_to_position")
        transform = getattr(adapter, "transform", None)
        if transform is None or not hasattr(transform, "factor") or not hasattr(
            transform, "center"
        ):
            raise ValueError("runner binding requires native affine mass transform")
        return self.tensor_kernel_factory(
            adapter=adapter,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )


def _issue_hmc_tuning_runner_binding(
    *,
    runner: Callable[[Any, Any, Any], Any],
    tensor_kernel_factory: Callable[..., Any],
    runner_identity: str,
    algorithm_family: str,
    target_scope: str,
    coordinate_scope: str,
    force_identity: str,
    force_semantics: str,
    endpoint_target_identity: str,
    endpoint_target_coordinate_system: str,
    endpoint_target_includes_chart_log_jacobian: bool,
    affine_log_jacobian_convention: str,
    target_status_evidence: str,
    supported_target_status_trace_policies: tuple[str, ...],
    supported_chain_execution_modes: tuple[str, ...],
    backend: str,
    dtype: str,
    xla_capable: bool,
    required_diagnostic_fields: tuple[str, ...],
    required_metadata_fields: tuple[str, ...],
    source_dependency_closure: Mapping[str, Any],
) -> HMCTuningRunnerBinding:
    """Issue a binding from a repository-owned algorithm module."""

    return HMCTuningRunnerBinding(
        _issuer_token=_RUNNER_BINDING_ISSUER_TOKEN,
        runner=runner,
        tensor_kernel_factory=tensor_kernel_factory,
        runner_identity=runner_identity,
        algorithm_family=algorithm_family,
        target_scope=target_scope,
        coordinate_scope=coordinate_scope,
        force_identity=force_identity,
        force_semantics=force_semantics,
        endpoint_target_identity=endpoint_target_identity,
        endpoint_target_coordinate_system=endpoint_target_coordinate_system,
        endpoint_target_includes_chart_log_jacobian=(
            endpoint_target_includes_chart_log_jacobian
        ),
        affine_log_jacobian_convention=affine_log_jacobian_convention,
        target_status_evidence=target_status_evidence,
        supported_target_status_trace_policies=supported_target_status_trace_policies,
        supported_chain_execution_modes=supported_chain_execution_modes,
        backend=backend,
        dtype=dtype,
        xla_capable=xla_capable,
        requires_native_affine_mass=True,
        supports_arbitrary_fixed_step_size=True,
        supports_arbitrary_fixed_leapfrog_count=True,
        required_diagnostic_fields=required_diagnostic_fields,
        required_metadata_fields=required_metadata_fields,
        source_dependency_closure=source_dependency_closure,
        artifact_authority=False,
    )


_HISTORICAL_NONCLAIMS = (
    "compatibility or diagnostic route only",
    "cannot issue canonical tuning admission artifacts",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
)

HMC_TUNING_ROUTE_REGISTRY: tuple[HMCTuningRouteRecord, ...] = (
    HMCTuningRouteRecord(
        interface_name="tune_hmc_kernel",
        module="bayesfilter.inference.hmc_tuning_dispatch",
        role="active",
        artifact_authority=True,
    ),
    HMCTuningRouteRecord(
        interface_name="tune_fixed_transport_hmc_kernel",
        module="bayesfilter.inference.fixed_transport_hmc_tuning_tf",
        role="active",
        artifact_authority=True,
    ),
    HMCTuningRouteRecord(
        interface_name="tune_hmc_kernel_robust_broad_grid",
        module="bayesfilter.inference.hmc_robust_broad_grid",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_fixed_metric_grid_search",
        module="bayesfilter.inference.hmc_fixed_metric_grid_search",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_operational_broad_grid",
        module="bayesfilter.inference.hmc_operational_broad_grid",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_operational_broad_grid_process_parallel",
        module="bayesfilter.inference.hmc_operational_broad_grid",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_fixed_mass_hmc_tuning_budget_ladder",
        module="bayesfilter.inference.hmc_budget_ladder",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_fixed_mass_step_tuning_diagnostic",
        module="bayesfilter.inference.hmc_tuning",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_windowed_mass_adaptation_diagnostic",
        module="bayesfilter.inference.hmc_tuning",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_fixed_trajectory_tuning_diagnostic",
        module="bayesfilter.inference.hmc_tuning",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_gaussian_dual_averaging_diagnostic",
        module="bayesfilter.inference.hmc_tuning",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_hmc_start_bank_diagnostic",
        module="bayesfilter.inference.hmc_kernel_tuning",
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="discover_fixed_transport_hmc_candidates",
        module=(
            "bayesfilter.inference.fixed_transport_hmc_candidate_discovery_tf"
        ),
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_fixed_transport_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_fixed_transport_hmc_candidate_campaign",
        module=(
            "bayesfilter.inference.fixed_transport_hmc_candidate_discovery_tf"
        ),
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_fixed_transport_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="refine_fixed_transport_hmc_candidates",
        module=(
            "bayesfilter.inference.fixed_transport_hmc_candidate_discovery_tf"
        ),
        role="diagnostic",
        artifact_authority=False,
        replacement="tune_fixed_transport_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_generic_hmc_tuning_orchestration",
        module="bayesfilter.inference.generic_hmc_tuning",
        role="historical",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="orchestrate_generic_hmc_tuning",
        module="bayesfilter.inference.generic_hmc_tuning",
        role="historical",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
    HMCTuningRouteRecord(
        interface_name="run_tiny_gaussian_fixed_trajectory_hmc_tuning_v2",
        module="bayesfilter.inference.fixed_trajectory_hmc_tuning_v2",
        role="historical",
        artifact_authority=False,
        replacement="tune_hmc_kernel",
        nonclaims=_HISTORICAL_NONCLAIMS,
    ),
)


_PUBLIC_TUNER_NONCLAIMS = (
    "tuning admission is not retained posterior convergence",
    "no sampler superiority claim",
    "no target-correctness claim",
    "no GPU or XLA readiness claim",
)


def _nonactive_route_capability(
    record: HMCTuningRouteRecord,
) -> HMCTuningInterfaceCapability:
    return HMCTuningInterfaceCapability(
        interface_name=record.interface_name,
        module=record.module,
        interface_kind=(
            "historical_helper"
            if record.role == "historical"
            else "diagnostic_helper"
        ),
        capability_status=(
            "historical_only" if record.role == "historical" else "diagnostic_only"
        ),
        artifact_authority=False,
        algorithm_family="legacy_or_diagnostic_hmc_tuning",
        target_contract="not authoritative; use the active replacement target contract",
        coordinate_prerequisite="not authoritative; use the active replacement coordinates",
        mass_policy="not authoritative for canonical handoff",
        step_size_policy="not authoritative for canonical handoff",
        trajectory_policy="not authoritative for canonical handoff",
        fresh_verification_policy="cannot issue canonical fresh-verification admission",
        ess_admission_policy="not authoritative for canonical handoff",
        target_status_telemetry="diagnostic or historical only",
        runner_injection_policy="does not confer artifact authority",
        identity_bindings=("route role", "active replacement"),
        source_closure_policy="Git history only; no active source-closure claim",
        replacement=record.replacement,
        forbidden_uses=(
            "canonical tuning artifact",
            "retained-kernel handoff",
            "public tuning recommendation",
        ),
        nonclaims=record.nonclaims or _HISTORICAL_NONCLAIMS,
        evidence_anchors=(
            "tests/test_hmc_tuning_contract.py::test_inventory_has_no_unclassified_or_stale_routes",
            "scripts/inventory_hmc_tuning_routes.py",
        ),
        mass_capability="not_authoritative",
        step_size_capability="not_authoritative",
        trajectory_capability="not_authoritative",
    )


HMC_TUNING_INTERFACE_CAPABILITIES: tuple[HMCTuningInterfaceCapability, ...] = (
    HMCTuningInterfaceCapability(
        interface_name="tune_hmc_kernel",
        module="bayesfilter.inference.hmc_tuning_dispatch",
        interface_kind="public_tuner",
        capability_status="tested_supported",
        artifact_authority=True,
        algorithm_family=(
            "ordinary_exact_value_score_broad_fixed_metric_hmc; typed "
            "position-field config is a conditional mechanics branch"
        ),
        target_contract=(
            "artifact authority requires an ordinary exact value/score target; "
            "the typed position-field branch additionally requires a repository-"
            "issued binding and exact endpoint target but remains mechanics-only"
        ),
        coordinate_prerequisite=(
            "ordinary adapter coordinates with repository-owned affine mass coordinates"
        ),
        mass_policy=(
            "windowed adaptation by default or explicit fixed identity from config"
        ),
        step_size_policy=(
            "bootstrap, then an independent epsilon ladder for every L in the "
            "primary and survivor-midpoint barriers"
        ),
        trajectory_policy=(
            "ordinary primary L grid (3, 5, 9, 13, 18, 25), followed by one "
            "midpoint-refinement barrier adjacent to every primary survivor"
        ),
        fresh_verification_policy=(
            "fresh fixed-kernel verification; default TFP runner requires typed "
            "acceptance, health, minimum draws, and rank-normalized split/folded "
            f"R-hat at or below {HMC_TUNING_ORDINARY_RHAT_THRESHOLD:.2f}"
        ),
        ess_admission_policy=(
            "disabled for ordinary tuning admission; retained posterior ESS is separate"
        ),
        target_status_telemetry=(
            "default runner supports none or per-chain-step; typed bindings must declare equivalent fail-closed endpoint evidence"
        ),
        runner_injection_policy=(
            "default TFP runner for exact-score ordinary HMC; a repository-issued "
            "HMCTuningRunnerBinding is accepted only with "
            "TensorFlowHMCKernelTuningConfig and is conditional mechanics evidence "
            "only; bare callables are forbidden"
        ),
        identity_bindings=(
            "adapter signature",
            "target scope",
            "mass artifact signature",
            "selected epsilon and leapfrog count",
            "runner binding hash and source closure",
        ),
        source_closure_policy="runner source hashes are serialized in public result identity",
        replacement=None,
        forbidden_uses=(
            "arbitrary bare runner callback",
            "engineering_probe_covariance_multiplier as a public ordinary mode",
            "shared epsilon across different ordinary L candidates",
            "fixed nonlinear transport without its transformed target contract",
            "retained posterior convergence claim",
        ),
        nonclaims=_PUBLIC_TUNER_NONCLAIMS,
        evidence_anchors=(
            "tests/test_hmc_tuning_documentation_contract.py::test_ordinary_capability_matches_public_signature",
            "tests/test_hmc_kernel_tuning_public_api.py::test_public_tuner_rejects_failed_sequential_rhat_handoff",
            "tests/test_hmc_kernel_tuning_public_api.py::test_public_ordinary_config_rejects_typed_runner_binding",
            "tests/test_neural_force_hmc.py::test_typed_tuning_binding_rejects_identity_mass_fallback",
        ),
        mass_capability="owned",
        step_size_capability="owned",
        trajectory_capability="owned",
        fresh_verification_required=True,
    ),
    HMCTuningInterfaceCapability(
        interface_name="tune_fixed_transport_hmc_kernel",
        module="bayesfilter.inference.fixed_transport_hmc_tuning_tf",
        interface_kind="public_tuner",
        capability_status="tested_supported",
        artifact_authority=True,
        algorithm_family="frozen_nonlinear_transport_fixed_trajectory_hmc",
        target_contract=(
            "identity-bound frozen transport with exact transformed value including Jacobian and matching score"
        ),
        coordinate_prerequisite="fixed transport latent z coordinates",
        mass_policy="fixed identity mass in z; no ordinary windowed mass adaptation",
        step_size_policy=(
            "measured_joint_grid_v1 explicit step-size candidates; legacy directional "
            "ladder is diagnostic-only"
        ),
        trajectory_policy=(
            "all declared (epsilon, L) pairs measured before replicated selection"
        ),
        fresh_verification_policy=(
            "disjoint replicated fixed-kernel selection and held-out verification"
        ),
        ess_admission_policy=(
            "selection-policy dependent; ordinary-tuner ESS status does not transfer"
        ),
        target_status_telemetry="transformed target health and declared transition telemetry",
        runner_injection_policy=(
            "scoped chain runner only after the frozen transformed adapter is built"
        ),
        identity_bindings=(
            "base and transformed adapter signatures",
            "transport manifest hash",
            "identity-z mass signature",
            "target scope",
            "source dependency closure",
        ),
        source_closure_policy="executable source file hashes are serialized and replay-checked",
        replacement=None,
        forbidden_uses=(
            "arbitrary position-only force without a frozen transport",
            "ordinary mass adaptation claim",
            "reuse after transport identity changes",
            "legacy directional diagnostic policy as an artifact-authority handoff",
        ),
        nonclaims=_PUBLIC_TUNER_NONCLAIMS,
        evidence_anchors=(
            "tests/test_fixed_transport_hmc_tuning.py",
            "tests/test_fixed_transport_hmc_binding.py",
        ),
        mass_capability="fixed",
        step_size_capability="owned",
        trajectory_capability="owned",
        requires_frozen_transport=True,
        fresh_verification_required=True,
    ),
    *tuple(
        _nonactive_route_capability(record)
        for record in HMC_TUNING_ROUTE_REGISTRY
        if record.role != "active"
    ),
    HMCTuningInterfaceCapability(
        interface_name="run_hmc_tune_verify_repair_loop",
        module="bayesfilter.inference.hmc_kernel_tuning",
        interface_kind="stage_helper",
        capability_status="internal_only",
        artifact_authority=False,
        algorithm_family="ordinary_hmc_internal_tune_verify_repair_stage",
        target_contract="prevalidated ordinary adapter, geometry, and bootstrap artifacts",
        coordinate_prerequisite="same adapter and mass-coordinate lineage as public tuner",
        mass_policy="internal windowed adaptation and repair",
        step_size_policy="internal fixed-mass tuning and repair",
        trajectory_policy="internal leapfrog-count selection and repair",
        fresh_verification_policy="internal fresh fixed-kernel verification",
        ess_admission_policy="disabled for ordinary tuning admission",
        target_status_telemetry="inherited from stage config and runner",
        runner_injection_policy="raw internal hook; direct use has no public authority",
        identity_bindings=("geometry hash", "bootstrap hash", "target scope"),
        source_closure_policy="bound only by enclosing public tuner result",
        replacement="tune_hmc_kernel",
        forbidden_uses=("final one-call API", "canonical artifact authority"),
        nonclaims=_HISTORICAL_NONCLAIMS,
        evidence_anchors=(
            "tests/test_hmc_kernel_tuning_outer_loop.py",
            "bayesfilter/inference/hmc_kernel_tuning.py::run_hmc_tune_verify_repair_loop",
        ),
        mass_capability="owned",
        step_size_capability="owned",
        trajectory_capability="owned",
        fresh_verification_required=True,
    ),
    HMCTuningInterfaceCapability(
        interface_name="run_full_chain_tfp_hmc",
        module="bayesfilter.inference.hmc",
        interface_kind="chain_runner",
        capability_status="internal_only",
        artifact_authority=False,
        algorithm_family="tfp_fixed_configuration_hmc_mechanics",
        target_contract="adapter exact value and matching score",
        coordinate_prerequisite="caller-supplied fixed mass coordinates",
        mass_policy="not owned; consumes supplied coordinates",
        step_size_policy="fixed or config-requested dual averaging only",
        trajectory_policy="fixed caller-supplied leapfrog count",
        fresh_verification_policy="none by itself",
        ess_admission_policy="none by itself",
        target_status_telemetry="none or per-chain-step according to config",
        runner_injection_policy="default mechanics runner inside active tuners",
        identity_bindings=("adapter signature", "target scope", "fixed config"),
        source_closure_policy="serialized by the enclosing public tuner",
        replacement="tune_hmc_kernel",
        forbidden_uses=("standalone full-tuning claim", "canonical artifact authority"),
        nonclaims=_HISTORICAL_NONCLAIMS,
        evidence_anchors=(
            "tests/test_hmc_fixed_size_chunk_runner.py",
            "tests/test_hmc_tuning_contract.py",
        ),
    ),
    HMCTuningInterfaceCapability(
        interface_name="run_fixed_transport_full_chain_tfp_hmc",
        module="bayesfilter.inference.fixed_transport_hmc_mechanics_tf",
        interface_kind="chain_runner",
        capability_status="internal_only",
        artifact_authority=False,
        algorithm_family="fixed_transport_fixed_configuration_hmc_mechanics",
        target_contract="preconstructed exact transformed value/score adapter",
        coordinate_prerequisite="frozen-transport latent z coordinates",
        mass_policy="not owned; consumes the supplied fixed coordinate system",
        step_size_policy="fixed or config-requested dual averaging only",
        trajectory_policy="fixed caller-supplied leapfrog count",
        fresh_verification_policy="none by itself",
        ess_admission_policy="none by itself",
        target_status_telemetry="transformed-target finite health from the supplied adapter",
        runner_injection_policy="internal mechanics runner for fixed-transport workflows",
        identity_bindings=("transformed adapter signature", "target scope", "fixed config"),
        source_closure_policy="serialized only by an enclosing fixed-transport workflow",
        replacement="tune_fixed_transport_hmc_kernel",
        forbidden_uses=(
            "standalone full-tuning claim",
            "candidate-selection claim",
            "canonical artifact authority",
        ),
        nonclaims=_HISTORICAL_NONCLAIMS,
        evidence_anchors=(
            "tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py",
            "bayesfilter/inference/fixed_transport_hmc_mechanics_tf.py::run_fixed_transport_full_chain_tfp_hmc",
        ),
    ),
    HMCTuningInterfaceCapability(
        interface_name="run_full_chain_neural_force_hmc",
        module="bayesfilter.inference.neural_force_hmc",
        interface_kind="chain_runner",
        capability_status="diagnostic_only",
        artifact_authority=False,
        algorithm_family="endpoint_corrected_frozen_position_force_hmc_mechanics",
        target_contract="frozen position-only force plus exact deterministic endpoint potential",
        coordinate_prerequisite="native affine mass coordinates when bound; direct identity fallback is diagnostic only",
        mass_policy="not owned; identity in supplied mass coordinates",
        step_size_policy="fixed or config-requested dual averaging only",
        trajectory_policy="fixed caller-supplied leapfrog count",
        fresh_verification_policy="none by itself",
        ess_admission_policy="none by itself",
        target_status_telemetry="endpoint target finite health is fail-closed; per-chain-step status trace unsupported",
        runner_injection_policy=(
            "only through bind_neural_force_hmc_tuning_runner and "
            "TensorFlowHMCKernelTuningConfig at the public facade"
        ),
        identity_bindings=("force identity", "endpoint target identity", "coordinate route"),
        source_closure_policy="serialized by the repository-issued binding",
        replacement="tune_hmc_kernel",
        forbidden_uses=(
            "standalone full-tuning claim",
            "mass-tuning claim",
            "leapfrog-count-tuning claim",
            "canonical artifact authority",
        ),
        nonclaims=_HISTORICAL_NONCLAIMS,
        evidence_anchors=("tests/test_neural_force_hmc.py",),
    ),
    HMCTuningInterfaceCapability(
        interface_name="bind_neural_force_hmc_tuning_runner",
        module="bayesfilter.inference.neural_force_hmc",
        interface_kind="runner_binding_factory",
        capability_status="tested_supported",
        artifact_authority=False,
        algorithm_family="typed_position_field_hmc_mechanics",
        target_contract="coordinate-consistent frozen force and exact raw-coordinate endpoint potential",
        coordinate_prerequisite=(
            "TensorFlow tuning branch native affine mass wrapper; identity fallback "
            "for a direct low-level call is diagnostic only"
        ),
        mass_policy="owned by the typed TensorFlow mechanics branch",
        step_size_policy="owned by the typed TensorFlow mechanics branch",
        trajectory_policy=(
            "powers-of-two candidate screen owned by the typed TensorFlow mechanics "
            "branch; not the ordinary broad-grid policy"
        ),
        fresh_verification_policy="fresh injected-runner fixed-kernel health and acceptance verification",
        ess_admission_policy="disabled for ordinary tuning admission",
        target_status_telemetry="equivalent fail-closed endpoint target, finite, divergence, energy, and movement evidence",
        runner_injection_policy="repository-issued typed binding only",
        identity_bindings=(
            "force identity",
            "endpoint target identity",
            "target and coordinate scope",
            "runner source closure",
        ),
        source_closure_policy="hashed executable source closure serialized in binding and tuning result",
        replacement="tune_hmc_kernel",
        forbidden_uses=(
            "standalone tuning authority",
            "bare callable injection",
            "transformed target masquerading as ordinary raw coordinates",
        ),
        nonclaims=_PUBLIC_TUNER_NONCLAIMS,
        evidence_anchors=(
            "tests/test_neural_force_hmc.py::test_typed_tuning_binding_validates_identity_and_telemetry",
            "tests/test_hmc_tuning_dispatch.py::test_tensorflow_config_limits_authority_to_mechanics_handoff",
        ),
    ),
)


def hmc_tuning_route_registry_payload() -> Mapping[str, Any]:
    """Return the immutable route ledger as JSON-ready metadata."""

    return {
        "schema": TUNING_ROUTE_REGISTRY_SCHEMA,
        "routes": tuple(record.payload() for record in HMC_TUNING_ROUTE_REGISTRY),
    }


def validate_hmc_tuning_interface_capabilities(
    records: tuple[HMCTuningInterfaceCapability, ...] = (
        HMC_TUNING_INTERFACE_CAPABILITIES
    ),
) -> tuple[HMCTuningInterfaceCapability, ...]:
    """Validate capability records and their route-authority relationships."""

    capabilities = tuple(records)
    names = tuple(record.qualified_name for record in capabilities)
    if len(names) != len(set(names)):
        raise ValueError("duplicate HMC tuning interface capability")
    capability_by_name = {
        record.qualified_name: record for record in capabilities
    }
    route_names = {record.qualified_name for record in HMC_TUNING_ROUTE_REGISTRY}
    missing = tuple(sorted(route_names - capability_by_name.keys()))
    if missing:
        raise ValueError("route lacks HMC tuning capability: " + ", ".join(missing))
    stale_public = tuple(
        sorted(
            record.qualified_name
            for record in capabilities
            if record.interface_kind == "public_tuner"
            and record.qualified_name not in route_names
        )
    )
    if stale_public:
        raise ValueError(
            "public tuner capability lacks route classification: "
            + ", ".join(stale_public)
        )
    route_by_name = {record.qualified_name: record for record in HMC_TUNING_ROUTE_REGISTRY}
    for qualified_name in route_names:
        route = route_by_name[qualified_name]
        capability = capability_by_name[qualified_name]
        if route.artifact_authority != capability.artifact_authority:
            raise ValueError(f"route/capability authority mismatch: {qualified_name}")
        if route.replacement != capability.replacement:
            raise ValueError(f"route/capability replacement mismatch: {qualified_name}")
    for capability in capabilities:
        ownership = (
            capability.mass_capability,
            capability.step_size_capability,
            capability.trajectory_capability,
        )
        if capability.interface_kind in {"chain_runner", "runner_binding_factory"}:
            if ownership != ("not_owned", "not_owned", "not_owned"):
                raise ValueError(
                    f"mechanics interface cannot own tuning choices: {capability.qualified_name}"
                )
            if capability.fresh_verification_required:
                raise ValueError(
                    f"mechanics interface cannot own fresh verification: {capability.qualified_name}"
                )
        if (
            capability.interface_kind == "public_tuner"
            and capability.capability_status == "tested_supported"
        ):
            if capability.step_size_capability != "owned":
                raise ValueError("active public tuner must own step-size selection")
            if capability.trajectory_capability != "owned":
                raise ValueError("active public tuner must own trajectory selection")
            expected_mass = "fixed" if capability.requires_frozen_transport else "owned"
            if capability.mass_capability != expected_mass:
                raise ValueError("active public tuner mass capability is inconsistent")
        if capability.interface_name == "tune_fixed_transport_hmc_kernel":
            if not capability.requires_frozen_transport:
                raise ValueError("fixed-transport tuner requires a frozen transport")
        if capability.interface_name == "run_full_chain_neural_force_hmc":
            if capability.interface_kind != "chain_runner":
                raise ValueError("neural-force full-chain interface is mechanics only")
    return capabilities


def hmc_tuning_capability_registry_payload() -> Mapping[str, Any]:
    """Return validated, JSON-ready capability metadata for documentation."""

    capabilities = validate_hmc_tuning_interface_capabilities()
    return {
        "schema": HMC_TUNING_CAPABILITY_REGISTRY_SCHEMA,
        "interfaces": tuple(record.payload() for record in capabilities),
    }


def active_hmc_tuning_routes() -> tuple[HMCTuningRouteRecord, ...]:
    return tuple(record for record in HMC_TUNING_ROUTE_REGISTRY if record.role == "active")


def hmc_tuning_route_record(interface_name: str) -> HMCTuningRouteRecord:
    matches = tuple(
        record for record in HMC_TUNING_ROUTE_REGISTRY if record.interface_name == interface_name
    )
    if len(matches) != 1:
        raise KeyError(f"unclassified or duplicate HMC tuning route: {interface_name}")
    return matches[0]


def hmc_tuning_interface_capability(
    interface_name: str,
) -> HMCTuningInterfaceCapability:
    matches = tuple(
        record
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
        if record.interface_name == interface_name
    )
    if len(matches) != 1:
        raise KeyError(f"unclassified or duplicate HMC tuning interface: {interface_name}")
    return matches[0]


def require_active_hmc_tuning_route(interface_name: str) -> HMCTuningRouteRecord:
    record = hmc_tuning_route_record(interface_name)
    if record.role != "active" or not record.artifact_authority:
        raise ValueError(f"HMC tuning route is not active: {interface_name}")
    return record


__all__ = [
    "HMC_TUNING_CAPABILITY_REGISTRY_SCHEMA",
    "HMC_TUNING_CAPABILITY_SCHEMA",
    "HMC_TUNING_INTERFACE_CAPABILITIES",
    "HMC_TUNING_ORDINARY_RHAT_THRESHOLD",
    "HMC_TUNING_RUNNER_BINDING_SCHEMA",
    "HMC_TUNING_ROUTE_REGISTRY",
    "HMCTuningInterfaceCapability",
    "HMCTuningRunnerBinding",
    "HMCTuningRouteRecord",
    "HMCTuningScope",
    "TUNING_CONTRACT_SCHEMA",
    "TUNING_ROUTE_REGISTRY_SCHEMA",
    "active_hmc_tuning_routes",
    "hmc_tuning_capability_registry_payload",
    "hmc_tuning_interface_capability",
    "hmc_tuning_route_record",
    "hmc_tuning_route_registry_payload",
    "require_active_hmc_tuning_route",
    "validate_hmc_tuning_interface_capabilities",
]

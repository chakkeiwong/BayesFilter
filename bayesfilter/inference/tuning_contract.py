"""Canonical interface and route-role contracts for HMC tuning.

The registry classifies public orchestration entry points. Internal stage
helpers remain implementation details and do not become additional active
interfaces merely because compatibility modules re-export them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


TUNING_CONTRACT_SCHEMA = "bayesfilter.hmc_tuning_contract.v1"
TUNING_ROUTE_REGISTRY_SCHEMA = "bayesfilter.hmc_tuning_route_registry.v1"

TuningRouteRole = Literal["active", "historical", "diagnostic"]


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
        module="bayesfilter.inference.hmc_kernel_tuning",
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


def hmc_tuning_route_registry_payload() -> Mapping[str, Any]:
    """Return the immutable route ledger as JSON-ready metadata."""

    return {
        "schema": TUNING_ROUTE_REGISTRY_SCHEMA,
        "routes": tuple(record.payload() for record in HMC_TUNING_ROUTE_REGISTRY),
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


def require_active_hmc_tuning_route(interface_name: str) -> HMCTuningRouteRecord:
    record = hmc_tuning_route_record(interface_name)
    if record.role != "active" or not record.artifact_authority:
        raise ValueError(f"HMC tuning route is not active: {interface_name}")
    return record


__all__ = [
    "HMC_TUNING_ROUTE_REGISTRY",
    "HMCTuningRouteRecord",
    "HMCTuningScope",
    "TUNING_CONTRACT_SCHEMA",
    "TUNING_ROUTE_REGISTRY_SCHEMA",
    "active_hmc_tuning_routes",
    "hmc_tuning_route_record",
    "hmc_tuning_route_registry_payload",
    "require_active_hmc_tuning_route",
]

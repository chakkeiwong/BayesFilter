"""Filter-program registry and capability gates for generic SSM targets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bayesfilter.ssm.contracts import BayesianSSMProblem, FilterProgram


_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


class InvalidFilterRegistryContract(ValueError):
    """Raised when filter registry metadata is incomplete or unsafe."""


@dataclass(frozen=True)
class FilterProgramDescriptor:
    """Registry descriptor for one admissible filter-program construction."""

    filter_id: str
    required_model_capabilities: tuple[str, ...]
    deterministic_target_policy: str
    approximation_semantics: str
    implementation_backend: str
    filter_hash: str
    manifest_extra: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        filter_id = _nonempty_text(self.filter_id, "filter_id")
        capabilities = tuple(
            _nonempty_text(item, "required_model_capability")
            for item in self.required_model_capabilities
        )
        if not capabilities:
            raise InvalidFilterRegistryContract(
                "required_model_capabilities must be nonempty"
            )
        if len(set(capabilities)) != len(capabilities):
            raise InvalidFilterRegistryContract(
                "required_model_capabilities must be unique"
            )
        backend = _nonempty_text(self.implementation_backend, "implementation_backend")
        filter_hash = _nonempty_text(self.filter_hash, "filter_hash")
        extra = {} if self.manifest_extra is None else _normalize_for_json(self.manifest_extra)
        if not isinstance(extra, Mapping):
            raise InvalidFilterRegistryContract("manifest_extra must be a mapping")
        object.__setattr__(self, "filter_id", filter_id)
        object.__setattr__(self, "required_model_capabilities", capabilities)
        object.__setattr__(
            self,
            "deterministic_target_policy",
            _nonempty_text(
                self.deterministic_target_policy,
                "deterministic_target_policy",
            ),
        )
        object.__setattr__(
            self,
            "approximation_semantics",
            _nonempty_text(self.approximation_semantics, "approximation_semantics"),
        )
        object.__setattr__(self, "implementation_backend", backend)
        object.__setattr__(self, "filter_hash", filter_hash)
        object.__setattr__(self, "manifest_extra", extra)
        _reject_process_local_identity(self.manifest_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "filter_id": self.filter_id,
            "required_model_capabilities": self.required_model_capabilities,
            "deterministic_target_policy": self.deterministic_target_policy,
            "approximation_semantics": self.approximation_semantics,
            "implementation_backend": self.implementation_backend,
            "filter_hash": self.filter_hash,
            "manifest_extra": self.manifest_extra,
        }


@dataclass(frozen=True)
class FilterRegistryDecision:
    """Accepted registry binding plus explanatory metadata."""

    filter_program: FilterProgram
    descriptor_signature: str
    missing_model_capabilities: tuple[str, ...]
    decision: str
    nonclaims: tuple[str, ...] = (
        "filter registry admissibility decision only",
        "no filter correctness claim",
        "no HMC readiness claim",
        "no posterior convergence claim",
    )

    def __post_init__(self) -> None:
        if self.decision != "accepted":
            raise InvalidFilterRegistryContract(
                "FilterRegistryDecision represents accepted decisions only"
            )
        object.__setattr__(
            self,
            "descriptor_signature",
            _nonempty_text(self.descriptor_signature, "descriptor_signature"),
        )
        object.__setattr__(
            self,
            "missing_model_capabilities",
            tuple(str(item) for item in self.missing_model_capabilities),
        )
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "filter_program": self.filter_program.manifest_payload(),
            "descriptor_signature": self.descriptor_signature,
            "missing_model_capabilities": self.missing_model_capabilities,
            "decision": self.decision,
            "nonclaims": self.nonclaims,
        }


class FilterProgramRegistry:
    """Fail-closed registry that converts descriptors into `FilterProgram`s."""

    def __init__(self, descriptors: tuple[FilterProgramDescriptor, ...] | list[FilterProgramDescriptor]) -> None:
        values = tuple(descriptors)
        if not values:
            raise InvalidFilterRegistryContract("registry descriptors must be nonempty")
        seen: set[str] = set()
        for descriptor in values:
            if not isinstance(descriptor, FilterProgramDescriptor):
                raise TypeError("registry descriptors must be FilterProgramDescriptor objects")
            if descriptor.filter_id in seen:
                raise InvalidFilterRegistryContract("filter_id values must be unique")
            seen.add(descriptor.filter_id)
        self._descriptors = values
        self._by_id = {descriptor.filter_id: descriptor for descriptor in values}

    @property
    def descriptors(self) -> tuple[FilterProgramDescriptor, ...]:
        return self._descriptors

    def descriptor_signature(self, filter_id: str) -> str:
        return stable_filter_descriptor_signature(self.require_descriptor(filter_id))

    def require_descriptor(self, filter_id: str) -> FilterProgramDescriptor:
        key = _nonempty_text(filter_id, "filter_id")
        try:
            return self._by_id[key]
        except KeyError as exc:
            raise InvalidFilterRegistryContract(f"unknown filter_id: {key}") from exc

    def bind_filter_program(
        self,
        *,
        filter_id: str,
        problem: BayesianSSMProblem,
        allow_fixed_randomness: bool = True,
    ) -> FilterRegistryDecision:
        if not isinstance(problem, BayesianSSMProblem):
            raise TypeError("problem must be a BayesianSSMProblem")
        descriptor = self.require_descriptor(filter_id)
        model_capabilities = _model_capabilities(problem)
        missing = tuple(
            capability
            for capability in descriptor.required_model_capabilities
            if capability not in model_capabilities
        )
        if missing:
            raise InvalidFilterRegistryContract(
                "model is missing required filter capabilities: " + ", ".join(missing)
            )
        if descriptor.deterministic_target_policy == "stochastic_not_hmc_ready":
            raise InvalidFilterRegistryContract(
                "stochastic filter requires explicit deterministic artifact state"
            )
        if (
            descriptor.deterministic_target_policy == "fixed_randomness"
            and not allow_fixed_randomness
        ):
            raise InvalidFilterRegistryContract(
                "fixed-randomness filter is not allowed by this registry binding"
            )
        program = FilterProgram(
            filter_id=descriptor.filter_id,
            required_model_capabilities=descriptor.required_model_capabilities,
            deterministic_target_policy=descriptor.deterministic_target_policy,
            approximation_semantics=descriptor.approximation_semantics,
            filter_manifest={
                "filter_id": descriptor.filter_id,
                "filter_hash": descriptor.filter_hash,
                "implementation_backend": descriptor.implementation_backend,
                "descriptor_signature": stable_filter_descriptor_signature(descriptor),
                "manifest_extra": descriptor.manifest_extra,
            },
        )
        return FilterRegistryDecision(
            filter_program=program,
            descriptor_signature=stable_filter_descriptor_signature(descriptor),
            missing_model_capabilities=missing,
            decision="accepted",
        )


def stable_filter_descriptor_signature(descriptor: FilterProgramDescriptor) -> str:
    """Return a stable SHA-256 signature for one filter descriptor."""

    if not isinstance(descriptor, FilterProgramDescriptor):
        raise TypeError("descriptor must be a FilterProgramDescriptor")
    return _stable_json_hash(
        {
            "schema": "bayesfilter.ssm.filter_program_descriptor.v1",
            "descriptor": descriptor.manifest_payload(),
        }
    )


def build_filter_program_registry(
    descriptors: tuple[FilterProgramDescriptor, ...] | list[FilterProgramDescriptor],
) -> FilterProgramRegistry:
    """Build a fail-closed filter-program registry from descriptors."""

    return FilterProgramRegistry(descriptors)


def _model_capabilities(problem: BayesianSSMProblem) -> frozenset[str]:
    raw = problem.model_manifest.get("capabilities", ())
    if isinstance(raw, str):
        values = (raw,)
    else:
        try:
            values = tuple(raw)
        except TypeError as exc:
            raise InvalidFilterRegistryContract(
                "model_manifest capabilities must be a string or sequence"
            ) from exc
    capabilities = frozenset(str(item) for item in values if str(item).strip())
    if not capabilities:
        raise InvalidFilterRegistryContract(
            "model_manifest capabilities must be nonempty"
        )
    return capabilities


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value)
    if not text.strip():
        raise InvalidFilterRegistryContract(f"{name} must be nonempty")
    return text


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_for_json(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_for_json(item) for item in value]
    return value


def _stable_json_hash(payload: Mapping[str, Any] | Any) -> str:
    normalized = _normalize_for_json(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    _reject_process_local_identity(blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _reject_process_local_identity(value: Any) -> None:
    text = json.dumps(_normalize_for_json(value), sort_keys=True, default=str)
    if any(pattern.search(text) for pattern in _PROCESS_LOCAL_SIGNATURE_PATTERNS):
        raise InvalidFilterRegistryContract(
            "filter registry manifests must not contain process-local identity"
        )

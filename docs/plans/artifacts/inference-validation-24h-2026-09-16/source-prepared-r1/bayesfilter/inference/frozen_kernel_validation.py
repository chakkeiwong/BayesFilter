"""Generic orchestration for frozen-kernel model validation.

This module validates candidate identity, tuning/data bindings, provenance, and
candidate-local diagnostic outcomes. A model adapter owns target evaluation and
fixed-kernel execution through the runner callback; this module never retunes,
changes controls, or ranks stochastic candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


GENERIC_FROZEN_VALIDATION_NONCLAIMS = (
    "generic frozen-kernel validation orchestration only",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no model-specific scientific validity claim",
    "no stochastic candidate ranking",
)


def _text(value: Any, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _text_tuple(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    result = tuple(_text(value, name=f"{name} item") for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload contains a non-finite float")
        return value
    raise TypeError(f"payload value {type(value).__name__} is not JSON-safe")


def _freeze_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload = _json_ready(value)
    return MappingProxyType(payload)


def _hash_payload(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenValidationCandidate:
    candidate_id: str
    model_id: str
    target_signature: str
    tuning_scope_signature: str
    controls: Mapping[str, Any]
    control_provenance: str
    execution_seed: tuple[int, int]
    parent_candidate_id: str | None = None
    inherited_control_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, name="candidate_id"))
        object.__setattr__(self, "model_id", _text(self.model_id, name="model_id"))
        object.__setattr__(self, "target_signature", _text(self.target_signature, name="target_signature"))
        object.__setattr__(self, "tuning_scope_signature", _text(self.tuning_scope_signature, name="tuning_scope_signature"))
        object.__setattr__(self, "controls", _freeze_mapping(self.controls, name="controls"))
        object.__setattr__(self, "control_provenance", _text(self.control_provenance, name="control_provenance"))
        seed = tuple(self.execution_seed)
        if len(seed) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in seed):
            raise ValueError("execution_seed must contain two nonnegative integers")
        object.__setattr__(self, "execution_seed", seed)
        parent = None if self.parent_candidate_id is None else _text(self.parent_candidate_id, name="parent_candidate_id")
        object.__setattr__(self, "parent_candidate_id", parent)
        object.__setattr__(self, "inherited_control_keys", _text_tuple(self.inherited_control_keys, name="inherited_control_keys"))
        if self.inherited_control_keys and parent is None:
            raise ValueError("inherited_control_keys require parent_candidate_id")
        if any(key not in self.controls for key in self.inherited_control_keys):
            raise ValueError("inherited control key is absent from controls")

    def payload(self) -> Mapping[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_id": self.model_id,
            "target_signature": self.target_signature,
            "tuning_scope_signature": self.tuning_scope_signature,
            "controls": dict(self.controls),
            "control_provenance": self.control_provenance,
            "execution_seed": self.execution_seed,
            "parent_candidate_id": self.parent_candidate_id,
            "inherited_control_keys": self.inherited_control_keys,
        }


@dataclass(frozen=True)
class FrozenTuningArtifactBinding:
    artifact_signature: str
    model_id: str
    target_signature: str
    tuning_scope_signature: str

    def __post_init__(self) -> None:
        for name in ("artifact_signature", "model_id", "target_signature", "tuning_scope_signature"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))

    def payload(self) -> Mapping[str, Any]:
        return {
            "artifact_signature": self.artifact_signature,
            "model_id": self.model_id,
            "target_signature": self.target_signature,
            "tuning_scope_signature": self.tuning_scope_signature,
        }


@dataclass(frozen=True)
class FrozenValidationScope:
    model_id: str
    target_signature: str
    tuning_scope_signature: str
    calibration_partition_signature: str
    validation_partition_signature: str
    validation_data_signature: str
    dtype: str
    backend: str

    def __post_init__(self) -> None:
        for name in (
            "model_id", "target_signature", "tuning_scope_signature",
            "calibration_partition_signature", "validation_partition_signature",
            "validation_data_signature", "dtype", "backend",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.calibration_partition_signature == self.validation_partition_signature:
            raise ValueError("calibration and validation partitions must be disjoint")

    def payload(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "target_signature": self.target_signature,
            "tuning_scope_signature": self.tuning_scope_signature,
            "calibration_partition_signature": self.calibration_partition_signature,
            "validation_partition_signature": self.validation_partition_signature,
            "validation_data_signature": self.validation_data_signature,
            "dtype": self.dtype,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class FrozenValidationPolicy:
    required_diagnostics: tuple[str, ...] = ("finite", "status")
    max_candidates: int = 256
    nonclaims: tuple[str, ...] = GENERIC_FROZEN_VALIDATION_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_diagnostics", _text_tuple(self.required_diagnostics, name="required_diagnostics"))
        limit = int(self.max_candidates)
        if limit <= 0:
            raise ValueError("max_candidates must be positive")
        object.__setattr__(self, "max_candidates", limit)
        object.__setattr__(self, "nonclaims", _text_tuple(self.nonclaims, name="nonclaims"))

    def payload(self) -> Mapping[str, Any]:
        return {
            "required_diagnostics": self.required_diagnostics,
            "max_candidates": self.max_candidates,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class FrozenValidationObservation:
    candidate: FrozenValidationCandidate
    status: str
    diagnostics: Mapping[str, Any]
    hard_vetoes: tuple[str, ...] = ()
    repair_triggers: tuple[str, ...] = ()
    runtime_seconds: float = 0.0
    exception: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, FrozenValidationCandidate):
            raise TypeError("candidate must be FrozenValidationCandidate")
        object.__setattr__(self, "status", _text(self.status, name="status"))
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics, name="diagnostics"))
        object.__setattr__(self, "hard_vetoes", _text_tuple(self.hard_vetoes, name="hard_vetoes") if self.hard_vetoes else ())
        object.__setattr__(self, "repair_triggers", _text_tuple(self.repair_triggers, name="repair_triggers") if self.repair_triggers else ())
        runtime = float(self.runtime_seconds)
        if not math.isfinite(runtime) or runtime < 0.0:
            raise ValueError("runtime_seconds must be finite and nonnegative")
        object.__setattr__(self, "runtime_seconds", runtime)
        object.__setattr__(self, "exception", None if self.exception is None else _text(self.exception, name="exception"))

    @property
    def viable(self) -> bool:
        return not self.hard_vetoes and self.exception is None

    def payload(self) -> Mapping[str, Any]:
        return {
            "candidate": self.candidate.payload(),
            "status": self.status,
            "diagnostics": dict(self.diagnostics),
            "hard_vetoes": self.hard_vetoes,
            "repair_triggers": self.repair_triggers,
            "runtime_seconds": self.runtime_seconds,
            "exception": self.exception,
            "viable": self.viable,
        }


@dataclass(frozen=True)
class FrozenKernelValidationResult:
    scope: FrozenValidationScope
    tuning_artifact: FrozenTuningArtifactBinding
    policy: FrozenValidationPolicy
    observations: tuple[FrozenValidationObservation, ...]
    contract_vetoes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        ids = [item.candidate.candidate_id for item in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("observations must have unique candidate IDs")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "contract_vetoes", _text_tuple(self.contract_vetoes, name="contract_vetoes") if self.contract_vetoes else ())

    @property
    def viable_observations(self) -> tuple[FrozenValidationObservation, ...]:
        return tuple(item for item in self.observations if item.viable)

    @property
    def next_round_candidates(self) -> tuple[FrozenValidationCandidate, ...]:
        return tuple(item.candidate for item in self.viable_observations)

    def payload(self) -> Mapping[str, Any]:
        body = {
            "schema": "bayesfilter.frozen_kernel_validation.result.v1",
            "scope": self.scope.payload(),
            "tuning_artifact": self.tuning_artifact.payload(),
            "policy": self.policy.payload(),
            "observations": tuple(item.payload() for item in self.observations),
            "contract_vetoes": self.contract_vetoes,
            "next_round_candidate_ids": tuple(item.candidate_id for item in self.next_round_candidates),
            "next_round_ranking_performed": False,
            "nonclaims": self.policy.nonclaims,
        }
        return {**body, "artifact_signature": _hash_payload(body)}


Runner = Callable[[FrozenValidationCandidate, FrozenValidationScope, tuple[int, int]], Mapping[str, Any]]


def _contract_vetoes(
    candidates: Sequence[FrozenValidationCandidate],
    artifact: FrozenTuningArtifactBinding,
    scope: FrozenValidationScope,
    policy: FrozenValidationPolicy,
) -> tuple[str, ...]:
    vetoes: list[str] = []
    if not candidates:
        vetoes.append("no_candidates")
    if len(candidates) > policy.max_candidates:
        vetoes.append("candidate_count_exceeds_policy")
    ids = [item.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        vetoes.append("duplicate_candidate_id")
    by_id = {item.candidate_id: item for item in candidates}
    if artifact.model_id != scope.model_id or artifact.target_signature != scope.target_signature or artifact.tuning_scope_signature != scope.tuning_scope_signature:
        vetoes.append("tuning_artifact_scope_mismatch")
    for candidate in candidates:
        if (candidate.model_id, candidate.target_signature, candidate.tuning_scope_signature) != (scope.model_id, scope.target_signature, scope.tuning_scope_signature):
            vetoes.append(f"candidate_scope_mismatch:{candidate.candidate_id}")
        if candidate.parent_candidate_id is not None:
            parent = by_id.get(candidate.parent_candidate_id)
            if parent is None:
                vetoes.append(f"missing_parent_candidate:{candidate.candidate_id}")
            else:
                for key in candidate.inherited_control_keys:
                    if candidate.controls[key] != parent.controls[key]:
                        vetoes.append(f"inherited_control_mismatch:{candidate.candidate_id}:{key}")
    return tuple(dict.fromkeys(vetoes))


def run_frozen_kernel_validation(
    *,
    candidates: Sequence[FrozenValidationCandidate],
    tuning_artifact: FrozenTuningArtifactBinding,
    scope: FrozenValidationScope,
    policy: FrozenValidationPolicy,
    runner: Runner,
) -> FrozenKernelValidationResult:
    """Run model adapters under one immutable, unranked validation contract."""

    if not isinstance(tuning_artifact, FrozenTuningArtifactBinding):
        raise TypeError("tuning_artifact must be FrozenTuningArtifactBinding")
    if not isinstance(scope, FrozenValidationScope):
        raise TypeError("scope must be FrozenValidationScope")
    if not isinstance(policy, FrozenValidationPolicy):
        raise TypeError("policy must be FrozenValidationPolicy")
    if not callable(runner):
        raise TypeError("runner must be callable")
    normalized = tuple(candidates)
    contract_vetoes = _contract_vetoes(normalized, tuning_artifact, scope, policy)
    if contract_vetoes:
        return FrozenKernelValidationResult(
            scope=scope,
            tuning_artifact=tuning_artifact,
            policy=policy,
            observations=(),
            contract_vetoes=contract_vetoes,
        )
    observations: list[FrozenValidationObservation] = []
    for candidate in normalized:
        started = time.perf_counter()
        try:
            raw = runner(candidate, scope, candidate.execution_seed)
            if not isinstance(raw, Mapping):
                raise TypeError("runner must return a mapping")
            echoed_controls = raw.get("controls")
            if echoed_controls is not None and _json_ready(echoed_controls) != _json_ready(candidate.controls):
                raise ValueError("runner changed frozen candidate controls")
            diagnostics = raw.get("diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                raise TypeError("runner diagnostics must be a mapping")
            vetoes = list(str(item) for item in raw.get("hard_vetoes", ()))
            missing = [key for key in policy.required_diagnostics if key not in diagnostics]
            vetoes.extend(f"missing_required_diagnostic:{key}" for key in missing)
            observations.append(
                FrozenValidationObservation(
                    candidate=candidate,
                    status=str(raw.get("status", "completed")),
                    diagnostics=diagnostics,
                    hard_vetoes=tuple(dict.fromkeys(vetoes)),
                    repair_triggers=tuple(str(item) for item in raw.get("repair_triggers", ())),
                    runtime_seconds=float(raw.get("runtime_seconds", time.perf_counter() - started)),
                )
            )
        except Exception as error:  # noqa: BLE001 - candidate-local evidence.
            observations.append(
                FrozenValidationObservation(
                    candidate=candidate,
                    status="candidate_execution_failed",
                    diagnostics={},
                    hard_vetoes=("candidate_execution_exception",),
                    runtime_seconds=time.perf_counter() - started,
                    exception=f"{type(error).__name__}: {error}",
                )
            )
    return FrozenKernelValidationResult(
        scope=scope,
        tuning_artifact=tuning_artifact,
        policy=policy,
        observations=tuple(observations),
    )

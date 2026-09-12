"""Typed numerical adapter boundary for the shared HMC candidate controller.

The controller owns candidate identity, scheduling, repair lineage, and
replay.  This module owns the narrow boundary at which a repository-issued
TensorFlow/TFP adapter evaluates one fixed candidate work item.  It does not
select a nominee and it cannot issue a numerical handoff by itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from bayesfilter.inference.hmc_candidate_set_artifacts import (
    write_candidate_set_result,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig,
    HMCWorkItem,
    HMCCandidateSetScope,
    HMCTuningCandidateRecord,
    HMCTuningCandidateSetResult,
    HMCInfrastructureFailure,
    HMCSharedInvalidity,
    HMCTuningCandidateSetController,
    _sha256,
)


HMC_CANDIDATE_SET_ADAPTER_SCHEMA = "bayesfilter.hmc_candidate_set_adapter.v1"
_ADAPTER_ISSUER_TOKEN = object()


CandidateObservationFn = Callable[
    [HMCWorkItem, HMCTuningCandidateRecord], Mapping[str, Any]
]


def _nonempty(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


@dataclass(frozen=True)
class HMCTypedCandidateSetAdapter:
    """Repository-issued evaluator bound to one immutable tuning scope.

    ``observe`` is intentionally private to the issued object.  The caller
    receives a result from the shared controller, never a callback-owned
    selection or handoff.  The adapter remains non-authoritative until its
    target and compilation qualification gates are completed.
    """

    _issuer_token: Any = field(repr=False, compare=False)
    scope: HMCCandidateSetScope
    adapter_kind: Literal["ordinary", "fixed_transport"]
    observe: CandidateObservationFn = field(repr=False, compare=False)
    source_dependency_closure: Mapping[str, Any]
    target_preparation_identity: str
    transition_identity: str
    qualification_status: Literal["unqualified", "qualified"] = "unqualified"

    def __post_init__(self) -> None:
        if self._issuer_token is not _ADAPTER_ISSUER_TOKEN:
            raise ValueError("candidate-set adapters must be repository-issued")
        if not isinstance(self.scope, HMCCandidateSetScope):
            raise TypeError("scope must be HMCCandidateSetScope")
        if self.adapter_kind not in {"ordinary", "fixed_transport"}:
            raise ValueError("adapter_kind must be ordinary or fixed_transport")
        if not callable(self.observe):
            raise TypeError("observe must be callable")
        closure = dict(self.source_dependency_closure)
        if not closure:
            raise ValueError("source_dependency_closure must be non-empty")
        object.__setattr__(self, "source_dependency_closure", closure)
        _nonempty(self.target_preparation_identity, "target_preparation_identity")
        _nonempty(self.transition_identity, "transition_identity")
        if self.scope.target_preparation_identity != self.target_preparation_identity:
            raise ValueError("scope target preparation identity does not match adapter")
        if self.scope.transition_identity != self.transition_identity:
            raise ValueError("scope transition identity does not match adapter")
        status = str(self.qualification_status)
        if status not in {"unqualified", "qualified"}:
            raise ValueError("qualification_status must be unqualified or qualified")
        object.__setattr__(self, "qualification_status", status)
        closure_hash = _sha256(closure)
        if self.scope.source_dependency_hash not in {
            closure_hash,
            "unbound-source-closure",
        }:
            raise ValueError("scope source dependency hash does not match adapter closure")

    @property
    def source_dependency_hash(self) -> str:
        return _sha256(self.source_dependency_closure)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": HMC_CANDIDATE_SET_ADAPTER_SCHEMA,
            "adapter_kind": self.adapter_kind,
            "scope": self.scope.payload(),
            "adapter_signature": self.scope.adapter_signature,
            "target_preparation_identity": self.target_preparation_identity,
            "transition_identity": self.transition_identity,
            "source_dependency_hash": self.source_dependency_hash,
            "qualification_status": self.qualification_status,
            "artifact_authority": False,
            "numerical_handoff_authority": False,
            "nonclaims": (
                "adapter qualification is not established by controller execution",
                "no posterior convergence claim",
                "no sampler superiority claim",
                "no default-readiness claim",
            ),
        }


@dataclass(frozen=True)
class HMCTypedCandidateSetRun:
    """Controller result plus the optional persisted mechanics artifact."""

    adapter: HMCTypedCandidateSetAdapter = field(repr=False, compare=False)
    result: HMCTuningCandidateSetResult
    artifact_receipt: Mapping[str, Any] | None = None

    @property
    def artifact_authority(self) -> bool:
        return False

    @property
    def numerical_handoff_authority(self) -> bool:
        return False

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_typed_candidate_set_run.v1",
            "adapter": self.adapter.payload(),
            "result": self.result.payload(),
            "artifact_receipt": self.artifact_receipt,
            "artifact_authority": False,
            "numerical_handoff_authority": False,
        }


def issue_hmc_candidate_set_adapter(
    *,
    scope: HMCCandidateSetScope,
    adapter_kind: Literal["ordinary", "fixed_transport"],
    observe: CandidateObservationFn,
    source_dependency_closure: Mapping[str, Any],
    target_preparation_identity: str,
    transition_identity: str,
    qualification_status: Literal["unqualified", "qualified"] = "unqualified",
) -> HMCTypedCandidateSetAdapter:
    """Issue a typed adapter after preparation has frozen the scope.

    Numerical preparation remains route-specific.  Once it has produced this
    object, ordinary and fixed-transport calls use the same controller and
    result semantics.  A qualified adapter is still required before any
    caller may turn a member into a public numerical handoff.
    """

    closure_hash = _sha256(source_dependency_closure)
    if scope.source_dependency_hash == "unbound-source-closure":
        scope = replace(scope, source_dependency_hash=closure_hash)
    elif scope.source_dependency_hash != closure_hash:
        raise ValueError("scope source dependency hash does not match adapter closure")
    return HMCTypedCandidateSetAdapter(
        _issuer_token=_ADAPTER_ISSUER_TOKEN,
        scope=scope,
        adapter_kind=adapter_kind,
        observe=observe,
        source_dependency_closure=source_dependency_closure,
        target_preparation_identity=target_preparation_identity,
        transition_identity=transition_identity,
        qualification_status=qualification_status,
    )


def run_typed_hmc_candidate_set(
    adapter: HMCTypedCandidateSetAdapter,
    config: HMCControllerConfig,
    *,
    output_dir: str | Path | None = None,
    max_work_items: int | None = None,
) -> HMCTypedCandidateSetRun:
    """Evaluate all controller work through one typed numerical adapter."""

    if not isinstance(adapter, HMCTypedCandidateSetAdapter):
        raise TypeError("adapter must be HMCTypedCandidateSetAdapter")
    if not isinstance(config, HMCControllerConfig):
        raise TypeError("config must be HMCControllerConfig")
    if adapter.scope.backend != "tensorflow_probability":
        raise ValueError("candidate-set numerical adapters require TensorFlow Probability")
    if adapter.scope.dtype not in {"float32", "float64"}:
        raise ValueError("candidate-set numerical adapter dtype must be float32 or float64")

    def observe(work: HMCWorkItem, candidate: HMCTuningCandidateRecord) -> Mapping[str, Any]:
        try:
            raw = adapter.observe(work, candidate)
        except (HMCInfrastructureFailure, HMCSharedInvalidity):
            raise
        if not isinstance(raw, Mapping):
            raise TypeError("typed candidate-set adapter must return a mapping")
        decision = str(raw.get("decision", ""))
        if not decision:
            raise ValueError("typed candidate-set observation must declare a decision")
        payload = dict(raw)
        payload.setdefault("adapter_kind", adapter.adapter_kind)
        payload.setdefault("adapter_signature", adapter.scope.adapter_signature)
        payload.setdefault("source_dependency_hash", adapter.source_dependency_hash)
        if payload["adapter_signature"] != adapter.scope.adapter_signature:
            raise HMCSharedInvalidity("candidate observation adapter scope mismatch")
        if payload["source_dependency_hash"] != adapter.source_dependency_hash:
            raise HMCSharedInvalidity("candidate observation source closure mismatch")
        return payload

    controller = HMCTuningCandidateSetController(adapter.scope, config)
    result = controller.run(observe, max_work_items=max_work_items)
    receipt = None
    if output_dir is not None:
        receipt = write_candidate_set_result(
            result, Path(output_dir) / "candidate_set_result.json"
        )
    return HMCTypedCandidateSetRun(
        adapter=adapter,
        result=result,
        artifact_receipt=receipt,
    )


__all__ = [
    "HMC_CANDIDATE_SET_ADAPTER_SCHEMA",
    "HMCTypedCandidateSetAdapter",
    "HMCTypedCandidateSetRun",
    "issue_hmc_candidate_set_adapter",
    "run_typed_hmc_candidate_set",
]

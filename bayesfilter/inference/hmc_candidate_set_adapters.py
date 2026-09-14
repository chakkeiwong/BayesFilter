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
import time
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
    _execution_binding: Any = field(default=None, repr=False, compare=False)
    work_cost: Any = field(default=None, repr=False, compare=False)
    _checkpoint_runtime: Any = field(default=None, repr=False, compare=False)

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
            "execution_binding_hash": getattr(self._execution_binding, "binding_hash", None),
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
    work_cost: Any = None,
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
        work_cost=work_cost,
    )


def run_typed_hmc_candidate_set(
    adapter: HMCTypedCandidateSetAdapter,
    config: HMCControllerConfig,
    *,
    output_dir: str | Path | None = None,
    max_work_items: int | None = None,
    _resume_controller: HMCTuningCandidateSetController | None = None,
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
    if output_dir is not None:
        root = Path(output_dir)
        if (root / "candidate_set_result.json").exists():
            raise FileExistsError(root / "candidate_set_result.json")
        if _resume_controller is None and any((root / name).exists() for name in (
                "tuning_checkpoint.json", "execution_spec.json", "controller_checkpoint.json")):
            raise FileExistsError("existing tuning run; use resume_hmc_candidate_set_tuning")

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

    controller = _resume_controller or HMCTuningCandidateSetController(adapter.scope, config)
    execution = adapter._execution_binding
    runtime = execution if execution is not None else adapter._checkpoint_runtime
    if config.max_gradient_work is not None and execution is None and adapter.work_cost is None:
        raise ValueError("a gradient-work budget requires an adapter work-cost model")
    checkpoint = None
    if output_dir is not None:
        def checkpoint(result):
            if execution is not None:
                from bayesfilter.inference.hmc_candidate_set_checkpoint import write_numerical_tuning_checkpoint
                write_numerical_tuning_checkpoint(execution, result, output_dir)
            elif runtime is not None:
                runtime.write_checkpoint(result, root)
            else:
                write_candidate_set_result(result, root / "controller_checkpoint.json", checkpoint=True)
    if runtime is not None:
        if _resume_controller is None:
            seconds = (execution.config.preparation_elapsed_seconds if execution is not None
                       else runtime.preparation_elapsed_seconds)
            controller._elapsed_seconds += seconds
            controller._accounting.append({"event": "preparation_elapsed", "seconds": seconds})
        runtime._checkpoint_callback = controller._save_checkpoint
        runtime._deadline = (None if config.max_wall_time_seconds is None else
            time.monotonic() + max(0.0, config.max_wall_time_seconds - controller._elapsed_seconds))
    try:
        result = controller.run(observe, max_work_items=max_work_items, checkpoint=checkpoint,
                                work_cost=adapter.work_cost if execution is None else execution.work_cost)
    finally:
        if runtime is not None:
            runtime._checkpoint_callback = None
            runtime._deadline = None
    receipt = None
    if output_dir is not None:
        receipt = write_candidate_set_result(
            result, root / ("candidate_set_result.json" if result.completion_status in
                {"complete", "shared_invalidity"} else (
                    "candidate_set_partial_result.json" if adapter._checkpoint_runtime is not None
                    else "controller_checkpoint.json")),
            checkpoint=result.completion_status not in {"complete", "shared_invalidity"},
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

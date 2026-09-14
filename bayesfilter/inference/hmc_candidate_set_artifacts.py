"""Serialization and replay authority for candidate-set tuning results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMC_CANDIDATE_SET_RESULT_SCHEMA,
    HMCCandidateSetScope,
    HMCTuningCandidateRecord,
    HMCWorkItem,
    HMCTuningCandidateSetController,
    HMCTuningCandidateSetResult,
    _sha256,
)


HMC_CANDIDATE_SET_ARTIFACT_SCHEMA = "bayesfilter.hmc_candidate_set_artifact.v1"


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")


def _validate_result_payload(
    payload: Mapping[str, Any], *, allow_legacy: bool = False
) -> None:
    """Validate authority fields after the outer checksum has been checked."""

    legacy = (
        payload.get("artifact_authority") is True
        and "replay_authority" not in payload
        and "numerical_handoff_authority" not in payload
    )
    if legacy and allow_legacy:
        return
    if payload.get("artifact_authority") is not False:
        raise ValueError("candidate-set controller artifact must be non-authoritative")
    if payload.get("replay_authority") is not True:
        raise ValueError("candidate-set artifact lacks replay authority")
    if payload.get("numerical_handoff_authority") is not False:
        raise ValueError("candidate-set artifact cannot issue a numerical handoff")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping) or not str(scope.get("scope_id", "")):
        raise ValueError("candidate-set artifact has invalid scope")
    checked_scope = HMCCandidateSetScope.from_payload(scope)
    candidates = payload.get("candidates")
    states = payload.get("candidate_states")
    if not isinstance(candidates, (tuple, list)) or not isinstance(states, Mapping):
        raise ValueError("candidate-set artifact has invalid candidate records")
    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate-set artifact contains a non-mapping candidate")
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in by_id:
            raise ValueError("candidate-set artifact has duplicate candidate IDs")
        if candidate.get("scope_id") != scope.get("scope_id"):
            raise ValueError("candidate-set candidate scope mismatch")
        supplied_hash = candidate.get("candidate_record_hash")
        if not supplied_hash:
            raise ValueError("candidate-set candidate is missing its record hash")
        identity = dict(candidate)
        identity.pop("candidate_record_hash", None)
        if _sha256(identity) != supplied_hash:
            raise ValueError("candidate-set candidate record hash mismatch")
        HMCTuningCandidateRecord.from_payload(checked_scope, candidate)
        by_id[candidate_id] = candidate

    state_by_id = {str(key): str(value) for key, value in states.items()}
    valid_states = {
        "proposed",
        "proposal_budget_pending",
        "screened",
        "validating",
        "inconclusive_at_cap",
        "promotion_failed",
        "verified",
    }
    if set(state_by_id) != set(by_id):
        raise ValueError("candidate-set candidate states do not cover candidate records")
    if any(value not in valid_states for value in state_by_id.values()):
        raise ValueError("candidate-set candidate state is invalid")

    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("candidate-set artifact has invalid controller config")
    total_budget = int(config.get("total_budget_units", 0))
    used_budget = int(payload.get("budget_used_units", 0))
    remaining_budget = int(payload.get("remaining_budget_units", -1))
    reserved_budget = int(payload.get("reserved_budget_units", 0))
    if total_budget <= 0 or used_budget < 0 or remaining_budget < 0 or reserved_budget < 0:
        raise ValueError("candidate-set artifact has inconsistent budget accounting")
    if reserved_budget > remaining_budget:
        raise ValueError("candidate-set reserved budget exceeds remaining budget")
    if used_budget + remaining_budget != total_budget:
        raise ValueError("candidate-set artifact has inconsistent budget accounting")

    verified = tuple(str(item) for item in payload.get("verified_candidate_ids", ()))
    if len(verified) != len(set(verified)):
        raise ValueError("candidate-set verified IDs are not unique")
    expected_verified = tuple(
        candidate["candidate_id"]
        for candidate in candidates
        if state_by_id[candidate["candidate_id"]] == "verified"
    )
    if payload.get("completion_status") != "shared_invalidity" and verified != expected_verified:
        raise ValueError("candidate-set verified IDs do not match candidate states")
    for candidate_id in verified:
        if candidate_id not in by_id:
            raise ValueError("verified candidate record is missing")
        if state_by_id.get(candidate_id) != "verified":
            raise ValueError("verified candidate state is not verified")

    viable = tuple(str(item) for item in payload.get("viable_candidate_ids", ()))
    expected_viable = tuple(
        candidate["candidate_id"]
        for candidate in candidates
        if state_by_id[candidate["candidate_id"]]
        in {"screened", "validating", "inconclusive_at_cap", "verified"}
    )
    if payload.get("completion_status") != "shared_invalidity" and viable != expected_viable:
        raise ValueError("candidate-set viable IDs do not match candidate states")

    completion = str(payload.get("completion_status", ""))
    final = str(payload.get("final_status", ""))
    if completion not in {
        "complete",
        "partial_budget",
        "paused_infrastructure",
        "shared_invalidity",
    }:
        raise ValueError("candidate-set completion status is invalid")
    if final not in {
        "complete",
        "partial_budget",
        "repair_budget_exhausted",
        "paused_infrastructure",
        "shared_invalidity",
    }:
        raise ValueError("candidate-set final status is invalid")
    if completion == "complete" and final != "complete":
        raise ValueError("complete candidate-set result has a non-complete final status")
    if completion == "paused_infrastructure" and final != "paused_infrastructure":
        raise ValueError("paused candidate-set result has an inconsistent final status")
    if completion == "shared_invalidity" and final != "shared_invalidity":
        raise ValueError("shared-invalidity result has an inconsistent final status")
    if completion == "shared_invalidity" and (
        payload.get("verified_candidate_ids") or payload.get("viable_candidate_ids")
    ):
        raise ValueError("shared-invalidity result cannot retain replayable members")
    if final == "repair_budget_exhausted" and completion != "partial_budget":
        raise ValueError("repair budget exhaustion requires partial completion")

    work_items = payload.get("work_items", ())
    if not isinstance(work_items, (tuple, list)):
        raise ValueError("candidate-set artifact has invalid work items")
    work_by_id: dict[str, Mapping[str, Any]] = {}
    work_by_attempt: dict[str, Mapping[str, Any]] = {}
    work_ordinals: set[int] = set()
    for index, work in enumerate(work_items, start=1):
        if not isinstance(work, Mapping):
            raise ValueError("candidate-set artifact contains a non-mapping work item")
        try:
            checked_work = HMCWorkItem.from_payload(work)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("candidate-set artifact contains an invalid work item") from exc
        work_id = str(checked_work.work_item_id)
        if work_id in work_by_id:
            raise ValueError("candidate-set artifact has duplicate work-item IDs")
        if checked_work.ordinal != index or checked_work.ordinal in work_ordinals:
            raise ValueError("candidate-set work-item order is not persisted")
        if checked_work.candidate_id not in by_id:
            raise ValueError("candidate-set work item references an unknown candidate")
        candidate = by_id[checked_work.candidate_id]
        if checked_work.candidate_record_hash != candidate["candidate_record_hash"]:
            raise ValueError("candidate-set work item candidate hash mismatch")
        if checked_work.candidate_family_id != candidate["candidate_family_id"]:
            raise ValueError("candidate-set work item family mismatch")
        if checked_work.status == "running":
            raise ValueError("candidate-set artifact contains an active work item")
        if checked_work.cohort_id != f"{scope['scope_id']}:cohort:pending" and not str(
            checked_work.cohort_id
        ).startswith(f"{scope['scope_id']}:cohort:"):
            raise ValueError("candidate-set work item cohort mismatch")
        if checked_work.verification_attempt_id is not None:
            attempt_id = str(checked_work.verification_attempt_id)
            if attempt_id in work_by_attempt:
                raise ValueError("duplicate verification work-item identity")
            work_by_attempt[attempt_id] = work
        else:
            work_by_attempt[work_id + ":attempt"] = work
        work_by_id[work_id] = work
        work_ordinals.add(checked_work.ordinal)

    receipts = payload.get("verification_receipts", ())
    if not isinstance(receipts, (tuple, list)):
        raise ValueError("candidate-set artifact has invalid verification receipts")
    receipt_by_hash: dict[str, Mapping[str, Any]] = {}
    receipt_ids: set[str] = set()
    stream_ranges: dict[str, list[tuple[int, int]]] = {}
    passing_decisions = {"passed", "acceptance_in_band"}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("candidate-set artifact contains a non-mapping receipt")
        candidate_id = str(receipt.get("candidate_id", ""))
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise ValueError("verification receipt references an unknown candidate")
        if receipt.get("candidate_record_hash") != candidate.get("candidate_record_hash"):
            raise ValueError("verification receipt candidate hash mismatch")
        if receipt.get("exact_l") != candidate.get("leapfrog_steps"):
            raise ValueError("verification receipt L mismatch")
        if float(receipt.get("epsilon")) != float(candidate.get("epsilon")):
            raise ValueError("verification receipt epsilon mismatch")
        if receipt.get("mass_signature") != candidate.get("mass_signature"):
            raise ValueError("verification receipt mass mismatch")
        attempt_id = str(receipt.get("verification_attempt_id", ""))
        if not attempt_id or attempt_id in receipt_ids:
            raise ValueError("duplicate verification attempt identity")
        work = work_by_attempt.get(attempt_id)
        if (
            work is None
            or work.get("candidate_id") != candidate_id
            or work.get("stage") != receipt.get("stage", "verification")
            or work.get("status") != "completed"
        ):
            raise ValueError("verification receipt has no matching work item")
        receipt_ids.add(attempt_id)
        draw_range = tuple(int(item) for item in receipt.get("draw_range", ()))
        if len(draw_range) != 2 or draw_range[0] < 0 or draw_range[1] < draw_range[0]:
            raise ValueError("verification receipt draw range is invalid")
        stream_id = str(receipt.get("stream_id", ""))
        if not stream_id:
            raise ValueError("verification receipt stream identity is missing")
        prior_ranges = stream_ranges.setdefault(stream_id, [])
        if any(
            draw_range[0] <= previous_end
            and previous_start <= draw_range[1]
            for previous_start, previous_end in prior_ranges
        ):
            raise ValueError("verification receipt draw ranges overlap within a stream")
        decision = str(receipt.get("decision", ""))
        if not decision:
            raise ValueError("verification receipt decision is missing")
        if (receipt.get("hard_vetoes") or receipt.get("promotion_vetoes")) and decision in passing_decisions:
            raise ValueError("verified verification receipt contains a hard veto")
        prior_ranges.append(draw_range)
        receipt_hash = _sha256(receipt)
        receipt_by_hash[receipt_hash] = receipt
    for candidate_id in verified:
        if not any(
            receipt.get("candidate_id") == candidate_id
            and receipt.get("decision") in passing_decisions
            and receipt.get("stage", "verification") == "verification"
            and not receipt.get("hard_vetoes") and not receipt.get("promotion_vetoes")
            and receipt.get("evidence_validity", "valid") == "valid"
            for receipt in receipts
            if isinstance(receipt, Mapping)
        ):
            raise ValueError("verified candidate lacks a passing verification receipt")

    actions = payload.get("repair_actions", ())
    if not isinstance(actions, (tuple, list)):
        raise ValueError("candidate-set artifact has invalid repair actions")
    action_by_id: dict[str, Mapping[str, Any]] = {}
    action_by_child: dict[str, Mapping[str, Any]] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("candidate-set artifact contains a non-mapping repair")
        action_id = str(action.get("repair_action_id", ""))
        if not action_id or action_id in action_by_id:
            raise ValueError("candidate-set artifact has duplicate repair action IDs")
        parent = by_id.get(str(action.get("parent_candidate_id", "")))
        child_id = str(action.get("child_candidate_id", ""))
        child = by_id.get(child_id)
        if parent is None or child is None:
            raise ValueError("repair references an unknown candidate")
        if child_id in action_by_child:
            raise ValueError("candidate-set child has multiple repair actions")
        action_by_id[action_id] = action
        action_by_child[child_id] = action
        if child.get("parent_candidate_id") != parent.get("candidate_id"):
            raise ValueError("repair child parent identity mismatch")
        if child.get("candidate_family_id") != action.get("candidate_family_id"):
            raise ValueError("repair family identity mismatch")
        for field in (
            "scope_id",
            "search_id",
            "target_signature",
            "coordinate_system",
            "start_bank_signature",
            "warmup_protocol",
            "backend",
            "dtype",
            "execution_mode",
            "adapter_signature",
            "source_dependency_hash",
            "target_preparation_identity",
            "transition_identity",
            "use_xla",
        ):
            if child.get(field) != parent.get(field) or child.get(field) != scope.get(field):
                raise ValueError(f"repair {field} identity mismatch")
        if child.get("leapfrog_steps") != action.get("exact_l"):
            raise ValueError("repair exact L mismatch")
        if child.get("mass_signature") != action.get("mass_signature"):
            raise ValueError("repair mass identity mismatch")
        if action.get("old_epsilon") != parent.get("epsilon"):
            raise ValueError("repair old epsilon mismatch")
        if action.get("new_epsilon") != child.get("epsilon"):
            raise ValueError("repair new epsilon mismatch")
        direction = action.get("direction")
        if direction == "repair_step_higher" and not float(child.get("epsilon")) > float(parent.get("epsilon")):
            raise ValueError("higher repair did not increase epsilon")
        if direction == "repair_step_lower" and not float(child.get("epsilon")) < float(parent.get("epsilon")):
            raise ValueError("lower repair did not decrease epsilon")
        qualified = action.get("qualified_repair_status")
        execution = action.get("execution_status")
        verification = action.get("verification_status")
        reason = action.get("not_executed_reason")
        if qualified == "executed_and_verified":
            if execution != "executed" or verification != "passed" or reason not in (None, ""):
                raise ValueError("qualified repair has inconsistent execution status")
            if child_id not in verified:
                raise ValueError("qualified repair child is not verified")
        elif qualified == "not_executed_with_reason":
            if not reason and execution != "not_started":
                raise ValueError("unqualified repair is missing its reason")
            if str(action.get("child_candidate_id")) in verified:
                raise ValueError("unqualified repair child cannot be verified")
        else:
            raise ValueError("repair qualified status is invalid")
        source_hash = action.get("source_verification_hash")
        if source_hash not in receipt_by_hash:
            raise ValueError("repair source verification receipt is missing")
        source_receipt = receipt_by_hash[source_hash]
        if source_receipt.get("candidate_id") != parent.get("candidate_id"):
            raise ValueError("repair source verification parent mismatch")
        if source_receipt.get("decision") not in {"repair_step_higher", "repair_step_lower"}:
            raise ValueError("repair source verification is not directional")

    for candidate in candidates:
        parent_id = candidate.get("parent_candidate_id")
        if parent_id is not None and candidate.get("candidate_id") not in action_by_child:
            raise ValueError("repair child is missing its repair action")
    for work in work_items:
        action_id = work.get("repair_action_id")
        if action_id is not None:
            action = action_by_id.get(str(action_id))
            if action is None or action.get("child_candidate_id") != work.get("candidate_id"):
                raise ValueError("work item repair identity mismatch")

    accounting = payload.get("accounting_events", ())
    if not isinstance(accounting, (tuple, list)):
        raise ValueError("candidate-set artifact has invalid accounting events")
    charged = 0
    allocated = 0
    released = 0
    for event in accounting:
        if not isinstance(event, Mapping):
            raise ValueError("candidate-set artifact contains an invalid accounting event")
        units = int(event.get("units", 0))
        if units < 0:
            raise ValueError("candidate-set accounting units must be non-negative")
        if event.get("event") == "work_charged":
            charged += units
        elif event.get("event") == "reserve_allocated":
            allocated += units
        elif event.get("event") == "reserve_released":
            released += units
    if charged != used_budget:
        raise ValueError("candidate-set budget usage does not match accounting")
    if allocated - charged - released != reserved_budget:
        raise ValueError("candidate-set reserved budget does not match accounting")


def candidate_set_result_payload(result: HMCTuningCandidateSetResult) -> Mapping[str, Any]:
    """Return the authority payload without adding caller-controlled fields."""
    payload = dict(result.payload())
    payload["artifact_schema"] = HMC_CANDIDATE_SET_ARTIFACT_SCHEMA
    payload["result_hash"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def write_candidate_set_result(
    result: HMCTuningCandidateSetResult,
    path: str | Path,
    *, checkpoint: bool = False,
) -> Mapping[str, Any]:
    """Write one atomic, checksummed result artifact."""
    destination = Path(path)
    if destination.exists() and not checkpoint:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = candidate_set_result_payload(result)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return {
        "path": destination.resolve().as_posix(),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "result_hash": payload["result_hash"],
        "schema": HMC_CANDIDATE_SET_ARTIFACT_SCHEMA,
    }


def load_candidate_set_result_payload(path: str | Path) -> Mapping[str, Any]:
    """Load and verify an authority payload without reconstructing executable state."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != HMC_CANDIDATE_SET_ARTIFACT_SCHEMA:
        raise ValueError("candidate-set artifact schema mismatch")
    supplied = payload.pop("result_hash", None)
    expected = hashlib.sha256(_canonical(payload)).hexdigest()
    if supplied != expected:
        raise ValueError("candidate-set result hash mismatch")
    if payload.get("schema") != HMC_CANDIDATE_SET_RESULT_SCHEMA:
        raise ValueError("candidate-set result schema mismatch")
    _validate_result_payload(payload, allow_legacy=True)
    payload["result_hash"] = supplied
    if (
        payload.get("artifact_authority") is True
        and "replay_authority" not in payload
    ):
        payload["_legacy_non_authoritative"] = True
    return payload


def require_verified_member(
    payload: Mapping[str, Any],
    *,
    scope_id: str,
    candidate_id: str,
    expected_scope: Mapping[str, Any] | None = None,
    expected_candidate_record_hash: str | None = None,
) -> Mapping[str, Any]:
    """Return one verified member only when scope and authority match."""
    if payload.get("artifact_schema") != HMC_CANDIDATE_SET_ARTIFACT_SCHEMA:
        raise ValueError("candidate-set artifact schema mismatch")
    if payload.get("_legacy_non_authoritative"):
        raise ValueError("legacy candidate-set artifact is non-authoritative")
    supplied_hash = payload.get("result_hash")
    if supplied_hash is None:
        raise ValueError("candidate-set replay requires a checksummed artifact")
    body = dict(payload)
    body.pop("result_hash", None)
    if hashlib.sha256(_canonical(body)).hexdigest() != supplied_hash:
        raise ValueError("candidate-set result hash mismatch")
    _validate_result_payload(payload)
    scope = payload.get("scope")
    if not isinstance(scope, Mapping) or scope.get("scope_id") != scope_id:
        raise ValueError("candidate-set scope mismatch")
    if expected_scope is not None:
        for key, expected in expected_scope.items():
            if scope.get(key) != expected:
                raise ValueError(f"candidate-set scope {key} mismatch")
    if payload.get("completion_status") == "shared_invalidity":
        raise ValueError("shared-invalidity result cannot be replayed")
    if candidate_id not in tuple(payload.get("verified_candidate_ids", ())):
        raise ValueError("candidate is not independently verified")
    for candidate in payload.get("candidates", ()):
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id:
            if expected_candidate_record_hash is not None and candidate.get(
                "candidate_record_hash"
            ) != expected_candidate_record_hash:
                raise ValueError("candidate record hash mismatch")
            return candidate
    raise ValueError("verified candidate record is missing")


def resume_hmc_candidate_set(
    artifact: str | Path | Mapping[str, Any],
    outcome_provider: Any,
    *,
    max_work_items: int | None = None,
) -> HMCTuningCandidateSetResult:
    """Resume a checked candidate-set artifact in its persisted work order.

    The artifact is validated before reconstruction. The provider receives only
    the next persisted work item, so completed chunks cannot be silently rerun.
    """

    if isinstance(artifact, (str, Path)):
        payload = load_candidate_set_result_payload(artifact)
    elif isinstance(artifact, Mapping):
        payload = dict(artifact)
        if payload.get("artifact_schema") == HMC_CANDIDATE_SET_ARTIFACT_SCHEMA:
            if payload.get("_legacy_non_authoritative"):
                raise ValueError("legacy candidate-set artifact cannot be resumed")
            supplied = payload.get("result_hash")
            body = dict(payload)
            body.pop("result_hash", None)
            if supplied != hashlib.sha256(_canonical(body)).hexdigest():
                raise ValueError("candidate-set result hash mismatch")
            _validate_result_payload(payload)
        else:
            raise ValueError("candidate-set replay requires a checksummed artifact")
    else:
        raise TypeError("artifact must be a path or mapping")
    if payload.get("_legacy_non_authoritative"):
        raise ValueError("legacy candidate-set artifact cannot be resumed")
    controller = HMCTuningCandidateSetController.from_result_payload(
        payload
    )
    return controller.run(outcome_provider, max_work_items=max_work_items)


__all__ = [
    "HMC_CANDIDATE_SET_ARTIFACT_SCHEMA",
    "candidate_set_result_payload",
    "load_candidate_set_result_payload",
    "resume_hmc_candidate_set",
    "require_verified_member",
    "write_candidate_set_result",
]

from __future__ import annotations

import json
import hashlib

import pytest

from bayesfilter.inference.hmc_candidate_set_artifacts import (
    candidate_set_result_payload,
    load_candidate_set_result_payload,
    require_verified_member,
    resume_hmc_candidate_set,
    write_candidate_set_result,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig,
    HMCTuningCandidateSetController,
    HMCCandidateSetScope,
)


def _result():
    scope = HMCCandidateSetScope(
        "artifact-scope",
        "search",
        "target",
        "mass",
        "ordinary",
        "bank",
        "warmup",
    )
    config = HMCControllerConfig((3,), ((3, (0.3,)),))
    return HMCTuningCandidateSetController(scope, config).run(
        lambda _work, _candidate: {"decision": "passed", "acceptance": 0.7}
    )


def _rehash(payload):
    body = dict(payload)
    body.pop("result_hash", None)
    payload["result_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=list).encode()
    ).hexdigest()
    return payload


def test_artifact_round_trip_preserves_result_hash_and_member(tmp_path):
    result = _result()
    path = tmp_path / "candidate-set.json"
    receipt = write_candidate_set_result(result, path)
    loaded = load_candidate_set_result_payload(path)
    assert receipt["result_hash"] == loaded["result_hash"]
    candidate_id = result.verified_candidate_ids[0]
    member = require_verified_member(
        loaded,
        scope_id=result.scope.scope_id,
        candidate_id=candidate_id,
    )
    assert member["candidate_id"] == candidate_id
    assert member["candidate_record_hash"]


def test_artifact_tampering_is_rejected(tmp_path):
    result = _result()
    path = tmp_path / "candidate-set.json"
    write_candidate_set_result(result, path)
    payload = json.loads(path.read_text())
    payload["verified_candidate_ids"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_candidate_set_result_payload(path)


def test_replay_rejects_wrong_scope_and_unverified_member():
    result = _result()
    payload = candidate_set_result_payload(result)
    candidate_id = result.verified_candidate_ids[0]
    with pytest.raises(ValueError, match="scope mismatch"):
        require_verified_member(payload, scope_id="other", candidate_id=candidate_id)
    with pytest.raises(ValueError, match="independently verified"):
        require_verified_member(payload, scope_id=result.scope.scope_id, candidate_id="missing")


def test_replay_checks_scope_execution_identity_and_candidate_hash():
    result = _result()
    payload = candidate_set_result_payload(result)
    candidate_id = result.verified_candidate_ids[0]
    expected_scope = {
        "target_signature": "target",
        "mass_signature": "mass",
        "backend": "tensorflow_probability",
        "dtype": "float64",
        "execution_mode": "tf_function",
    }
    member = require_verified_member(
        payload,
        scope_id=result.scope.scope_id,
        candidate_id=candidate_id,
        expected_scope=expected_scope,
        expected_candidate_record_hash=result.candidates[0].candidate_record_hash,
    )
    assert member["candidate_id"] == candidate_id
    with pytest.raises(ValueError, match="scope backend mismatch"):
        require_verified_member(
            payload,
            scope_id=result.scope.scope_id,
            candidate_id=candidate_id,
            expected_scope={"backend": "other-backend"},
        )
    with pytest.raises(ValueError, match="candidate record hash mismatch"):
        require_verified_member(
            payload,
            scope_id=result.scope.scope_id,
            candidate_id=candidate_id,
            expected_candidate_record_hash="stale",
        )


def test_shared_invalidity_disables_member_replay_even_after_prior_verification():
    result = _result()
    payload = dict(candidate_set_result_payload(result))
    payload["completion_status"] = "shared_invalidity"
    payload["final_status"] = "shared_invalidity"
    payload["verified_candidate_ids"] = []
    # The caller cannot turn the changed status into an authority artifact by
    # copying the old result hash; the checksum must be recomputed first.
    _rehash(payload)
    with pytest.raises(ValueError, match="shared-invalidity"):
        require_verified_member(
            payload,
            scope_id=result.scope.scope_id,
            candidate_id=result.verified_candidate_ids[0],
        )


def test_replay_rejects_inconclusive_receipt_relabelled_as_verified():
    result = _result()
    payload = dict(candidate_set_result_payload(result))
    candidate_id = result.verified_candidate_ids[0]
    receipts = [dict(receipt) for receipt in payload["verification_receipts"]]
    for receipt in receipts:
        if receipt["candidate_id"] == candidate_id and receipt.get("stage", "verification") == "verification":
            receipt["decision"] = "inconclusive_evidence"
    payload["verification_receipts"] = receipts
    _rehash(payload)

    with pytest.raises(ValueError, match="passing verification receipt"):
        require_verified_member(
            payload,
            scope_id=result.scope.scope_id,
            candidate_id=candidate_id,
        )


def test_replay_rejects_verified_repair_child_without_repair_action():
    scope = _result().scope
    config = HMCControllerConfig((3,), ((3, (0.3,)),))

    def outcome(work, candidate):
        if work.stage == "verification" and candidate.parent_candidate_id is None:
            return {"decision": "repair_step_higher", "acceptance": 0.9}
        return {"decision": "passed", "acceptance": 0.7}

    result = HMCTuningCandidateSetController(scope, config).run(outcome)
    child_id = next(
        candidate.candidate_id
        for candidate in result.candidates
        if candidate.parent_candidate_id is not None
    )
    payload = dict(candidate_set_result_payload(result))
    payload["repair_actions"] = []
    _rehash(payload)

    with pytest.raises(ValueError, match="missing its repair action"):
        require_verified_member(
            payload,
            scope_id=result.scope.scope_id,
            candidate_id=child_id,
        )


def test_replay_rejects_pending_work_receipt_and_overreserved_budget():
    result = _result()
    payload = dict(candidate_set_result_payload(result))
    verification_work = next(
        item
        for item in payload["work_items"]
        if item["stage"] == "verification"
    )
    work_items = [dict(item) for item in payload["work_items"]]
    for item in work_items:
        if item["work_item_id"] == verification_work["work_item_id"]:
            item["status"] = "pending"
    payload["work_items"] = work_items
    payload["reserved_budget_units"] = payload["remaining_budget_units"] + 1
    _rehash(payload)

    with pytest.raises(ValueError, match="reserved budget"):
        require_verified_member(
            payload,
            scope_id=result.scope.scope_id,
            candidate_id=result.verified_candidate_ids[0],
        )
    payload["reserved_budget_units"] = result.reserved_budget_units
    _rehash(payload)
    with pytest.raises(ValueError, match="no matching work item"):
        require_verified_member(
            payload,
            scope_id=result.scope.scope_id,
            candidate_id=result.verified_candidate_ids[0],
        )


def test_resume_rejects_reordered_persisted_work_items():
    scope = HMCCandidateSetScope(
        "resume-order-scope", "search", "target", "mass", "ordinary", "bank", "warmup"
    )
    controller = HMCTuningCandidateSetController(
        scope, HMCControllerConfig((3, 5), ((3, (0.3,)), (5, (0.4,))))
    )
    partial = controller.run(
        lambda _work, _candidate: {"decision": "passed", "acceptance": 0.7},
        max_work_items=1,
    )
    payload = dict(candidate_set_result_payload(partial))
    payload["work_items"] = list(reversed(payload["work_items"]))
    _rehash(payload)

    with pytest.raises(ValueError, match="work-item order"):
        resume_hmc_candidate_set(
            payload,
            lambda _work, _candidate: {"decision": "passed", "acceptance": 0.7},
        )


def test_controller_artifact_is_replayable_mechanics_evidence_not_numerical_authority():
    payload = candidate_set_result_payload(_result())
    assert payload["artifact_authority"] is False
    assert payload["replay_authority"] is True
    assert payload["numerical_handoff_authority"] is False
    member = require_verified_member(
        payload,
        scope_id=payload["scope"]["scope_id"],
        candidate_id=payload["verified_candidate_ids"][0],
    )
    assert member["candidate_record_hash"]


def test_persisted_partial_result_resumes_without_repeating_completed_work(tmp_path):
    # Re-run a deliberately interrupted search so the artifact carries a
    # non-empty active cohort and a partially charged budget.
    from bayesfilter.inference.hmc_candidate_set_tuning import (
        HMCControllerConfig,
        HMCCandidateSetScope,
        HMCTuningCandidateSetController,
    )

    scope = HMCCandidateSetScope(
        "resume-scope", "search", "target", "mass", "ordinary", "bank", "warmup"
    )
    controller = HMCTuningCandidateSetController(
        scope, HMCControllerConfig((3, 5), ((3, (0.3,)), (5, (0.4,))))
    )
    first = controller.run(
        lambda _work, _candidate: {"decision": "passed", "acceptance": 0.7},
        max_work_items=1,
    )
    path = tmp_path / "partial.json"
    write_candidate_set_result(first, path)
    completed_before = tuple(
        item.work_item_id for item in first.work_items if item.status == "completed"
    )
    resumed = resume_hmc_candidate_set(
        path,
        lambda _work, _candidate: {"decision": "passed", "acceptance": 0.7},
    )
    assert resumed.final_status == "complete"
    assert resumed.verified_candidate_ids == tuple(
        candidate.candidate_id for candidate in resumed.candidates
    )
    assert all(
        next(item for item in resumed.work_items if item.work_item_id == work_id).status
        == "completed"
        for work_id in completed_before
    )

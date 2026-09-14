from __future__ import annotations

import pytest
from dataclasses import replace

from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig,
    HMCInfrastructureFailure,
    HMCSharedInvalidity,
    HMCTuningCandidateSetController,
    HMCTuningScopeCollection,
    HMCCandidateSetScope,
)


def _scope(name: str = "scope") -> HMCCandidateSetScope:
    return HMCCandidateSetScope(
        scope_id=name,
        search_id="search-1",
        target_signature="target-v1",
        mass_signature="mass-v1",
        coordinate_system="ordinary",
        start_bank_signature="bank-v1",
        warmup_protocol="warmup-v1",
        epsilon_domain=(0.05, 2.0),
        repair_factor=2.0,
        max_repairs_per_family=2,
    )


def _config(
    grid=(3, 5),
    epsilons=((3, (0.25,)), (5, (0.35,))),
    **kwargs,
) -> HMCControllerConfig:
    return HMCControllerConfig(grid, epsilons, **kwargs)


def _pass(_work, _candidate):
    return {"decision": "passed", "acceptance": 0.70}


def test_broad_cohort_retains_every_verified_candidate():
    controller = HMCTuningCandidateSetController(_scope(), _config())
    result = controller.run(_pass)
    assert result.final_status == "complete"
    assert len(result.candidates) == 2
    assert result.verified_candidate_ids == tuple(item.candidate_id for item in result.candidates)
    assert result.replay_candidate(result.verified_candidate_ids[0]).candidate_record_hash


def test_multiple_epsilons_at_one_l_are_distinct_and_retained():
    config = _config(
        grid=(3,),
        epsilons=((3, (0.20, 0.30)),),
    )
    result = HMCTuningCandidateSetController(_scope(), config).run(_pass)
    records = [item for item in result.candidates if item.leapfrog_steps == 3]
    assert len(records) == 2
    assert records[0].candidate_id != records[1].candidate_id
    assert records[0].candidate_record_hash != records[1].candidate_record_hash
    assert records[0].candidate_family_id != records[1].candidate_family_id


def test_directional_repair_is_same_l_new_record_and_runs_before_later_work():
    controller = HMCTuningCandidateSetController(_scope(), _config())
    seen: list[tuple[str, int, int]] = []
    added = False

    def outcome(work, candidate):
        nonlocal added
        seen.append((work.stage, candidate.leapfrog_steps, work.priority))
        if work.stage == "verification" and candidate.leapfrog_steps == 3 and candidate.parent_candidate_id is None:
            if not added:
                controller.add_exploration_candidate(9, 0.4)
                added = True
            return {"decision": "repair_step_higher", "acceptance": 0.90}
        return {"decision": "passed", "acceptance": 0.70}

    result = controller.run(outcome)
    action = result.repair_actions[0]
    parent = next(item for item in result.candidates if item.candidate_id == action.parent_candidate_id)
    child = next(item for item in result.candidates if item.candidate_id == action.child_candidate_id)
    assert child.leapfrog_steps == parent.leapfrog_steps == 3
    assert child.epsilon > parent.epsilon
    assert child.candidate_family_id == parent.candidate_family_id
    assert child.parent_candidate_id == parent.candidate_id
    assert child.candidate_record_hash != parent.candidate_record_hash
    child_measurement = next(i for i, row in enumerate(seen) if row[1] == 3 and row[2] == 0 and row[0] == "measurement")
    later_measurement = next(i for i, row in enumerate(seen) if row[1] == 9)
    assert child_measurement < later_measurement
    assert action.qualified_repair_status == "executed_and_verified"


def test_inconclusive_evidence_does_not_create_repair_or_switch_l():
    def outcome(work, _candidate):
        if work.stage == "verification":
            return {"decision": "inconclusive_evidence", "acceptance": 0.78}
        return {"decision": "passed"}

    result = HMCTuningCandidateSetController(_scope(), _config()).run(outcome)
    assert result.repair_actions == ()
    assert all(item.leapfrog_steps in {3, 5} for item in result.candidates)
    assert result.verified_candidate_ids == ()


@pytest.mark.parametrize("decision", ["inconclusive_evidence", "inconclusive_conflict", "failed"])
def test_repair_child_requires_a_passing_decision_in_live_and_durable_replay(decision):
    from bayesfilter.inference.hmc_candidate_set_artifacts import (
        candidate_set_result_payload, require_verified_member,
    )

    def outcome(work, candidate):
        if work.stage == "verification":
            return {"decision": "repair_step_higher" if candidate.parent_candidate_id is None else decision}
        return {"decision": "passed"}

    result = HMCTuningCandidateSetController(_scope(), _config(grid=(3,), epsilons=((3, (0.25,)),))).run(outcome)
    child = result.candidates[-1]
    assert result.verified_candidate_ids == ()
    assert result.candidate_states[child.candidate_id] == (
        "promotion_failed" if decision == "failed" else "inconclusive_at_cap"
    )
    for replay in (
        lambda: result.replay_candidate(child.candidate_id),
        lambda: require_verified_member(candidate_set_result_payload(result), scope_id=result.scope.scope_id, candidate_id=child.candidate_id),
    ):
        with pytest.raises(ValueError, match="verified"):
            replay()


def test_verified_grandchild_does_not_require_intermediate_repair_to_pass():
    def outcome(work, candidate):
        if work.stage == "verification" and candidate.epsilon < 1.0:
            return {"decision": "repair_step_higher", "acceptance": 0.90}
        return _pass(work, candidate)

    result = HMCTuningCandidateSetController(_scope(), _config(grid=(3,), epsilons=((3, (0.25,)),))).run(outcome)
    assert [action.qualified_repair_status for action in result.repair_actions] == [
        "not_executed_with_reason", "executed_and_verified",
    ]
    child = result.replay_candidate(result.verified_candidate_ids[0])
    assert child.epsilon == 1.0
    assert child.leapfrog_steps == 3


def test_live_replay_rejects_a_forged_verified_id_without_passing_receipt():
    result = HMCTuningCandidateSetController(_scope(), _config()).run(
        lambda work, candidate: {"decision": "inconclusive_evidence"},
    )
    candidate_id = result.candidates[0].candidate_id
    forged = replace(result, verified_candidate_ids=(candidate_id,))
    with pytest.raises(ValueError):
        forged.replay_candidate(candidate_id)


def test_reserve_release_and_child_allocation_are_separate_events():
    config = _config(repair_reserve_units=3)
    controller = HMCTuningCandidateSetController(_scope(), config)

    def outcome(work, candidate):
        if work.stage == "verification" and candidate.leapfrog_steps == 3 and candidate.parent_candidate_id is None:
            return {"decision": "repair_step_higher"}
        return {"decision": "passed"}

    result = controller.run(outcome)
    events = result.accounting_events
    parent = result.repair_actions[0].parent_candidate_id
    child = result.repair_actions[0].child_candidate_id
    assert any(e.get("event") == "reserve_released" and e.get("candidate_id") == parent for e in events)
    child_allocations = [
        e for e in events
        if e.get("event") == "reserve_allocated" and e.get("candidate_id") == child
    ]
    assert len(child_allocations) == 1
    assert child_allocations[0].get("source") == "repair_reserve"
    assert result.budget_used_units + result.remaining_budget_units == config.total_budget_units
    assert result.repair_actions[0].qualified_repair_status == "executed_and_verified"


def test_unfunded_repair_is_typed_and_not_replayable():
    config = _config(repair_reserve_units=1, allow_repair_from_free_pool=False)
    result = HMCTuningCandidateSetController(_scope(), config).run(
        lambda work, candidate: {"decision": "repair_step_higher"}
        if work.stage == "verification" and candidate.leapfrog_steps == 3
        else {"decision": "passed"}
    )
    action = result.repair_actions[0]
    assert result.final_status == "repair_budget_exhausted"
    assert result.completion_status == "partial_budget"
    assert action.qualified_repair_status == "not_executed_with_reason"
    assert action.not_executed_reason == "repair_budget_exhausted"
    assert action.child_candidate_id not in result.verified_candidate_ids
    with pytest.raises(ValueError, match="verified candidate"):
        result.replay_candidate(action.child_candidate_id)


def test_interrupted_cohort_resumes_without_reordering():
    controller = HMCTuningCandidateSetController(_scope(), _config())
    first = controller.run(_pass, max_work_items=1)
    assert first.completion_status == "partial_budget"
    pending_before = first.resume_pending_work_item_ids
    completed_before = tuple(item.work_item_id for item in first.work_items if item.status == "completed")
    second = controller.run(_pass)
    assert second.final_status == "complete"
    assert completed_before
    assert all(
        next(item for item in second.work_items if item.work_item_id == work_id).status == "completed"
        for work_id in completed_before
    )
    assert pending_before
    assert [item.ordinal for item in second.work_items] == sorted(item.ordinal for item in second.work_items)


def test_budget_accounting_charges_work_once_and_conserves_total():
    config = _config(total_budget_units=20, candidate_reserve_units=3)
    result = HMCTuningCandidateSetController(_scope(), config).run(_pass)
    charged = [
        event
        for event in result.accounting_events
        if event.get("event") == "work_charged"
    ]
    assert len(charged) == 4  # measurement and verification for both candidates
    assert result.budget_used_units == 4
    assert result.reserved_budget_units == 0
    assert result.remaining_budget_units == 16
    assert (
        result.budget_used_units + result.remaining_budget_units
        == config.total_budget_units
    )


def test_unfunded_exploration_is_partial_and_has_no_work_item():
    controller = HMCTuningCandidateSetController(
        _scope(),
        _config(
            total_budget_units=6,
            repair_reserve_units=1,
            candidate_reserve_units=3,
        ),
    )
    candidate_id = controller.add_exploration_candidate(9, 0.4)
    result = controller.run(_pass)
    assert result.final_status == "partial_budget"
    assert result.completion_status == "partial_budget"
    assert result.candidate_states[candidate_id] == "proposal_budget_pending"
    assert all(item.candidate_id != candidate_id for item in result.work_items)


def test_scope_collection_is_read_only_and_keeps_scope_bound_members():
    first = HMCTuningCandidateSetController(_scope("scope-a"), _config()).run(_pass)
    second = HMCTuningCandidateSetController(_scope("scope-b"), _config()).run(_pass)
    collection = HMCTuningScopeCollection(
        (first, second), expected_scope_ids=("scope-a", "scope-b")
    )
    assert collection.complete is True
    first_member = collection.member("scope-a", first.verified_candidate_ids[0])
    second_member = collection.member("scope-b", second.verified_candidate_ids[0])
    assert first_member.scope_id == "scope-a"
    assert second_member.scope_id == "scope-b"
    assert collection.payload()["artifact_authority"] is False
    with pytest.raises(ValueError, match="scope is absent"):
        collection.member("scope-c", "missing")


def test_scope_invalidity_dominates_budget_and_disables_replay():
    controller = HMCTuningCandidateSetController(_scope(), _config())

    def invalid(_work, _candidate):
        raise HMCSharedInvalidity("scope mismatch")

    result = controller.run(invalid)
    assert result.final_status == "shared_invalidity"
    assert result.completion_status == "shared_invalidity"
    assert result.verified_candidate_ids == ()


def test_infrastructure_failure_is_not_scientific_candidate_rejection():
    controller = HMCTuningCandidateSetController(_scope(), _config())

    def broken(_work, _candidate):
        raise HMCInfrastructureFailure("worker stopped")

    result = controller.run(broken)
    assert result.final_status == "paused_infrastructure"
    assert result.completion_status == "paused_infrastructure"
    assert result.resume_pending_work_item_ids


def test_verification_receipt_binds_exact_candidate_identity():
    controller = HMCTuningCandidateSetController(_scope(), _config())
    result = controller.run(_pass)
    receipt = result.verification_receipts[0]
    candidate = next(item for item in result.candidates if item.candidate_id == receipt.candidate_id)
    assert receipt.candidate_record_hash == candidate.candidate_record_hash
    assert receipt.exact_l == candidate.leapfrog_steps
    assert receipt.epsilon == candidate.epsilon
    assert receipt.mass_signature == candidate.mass_signature


def test_lower_repair_stays_in_family_and_decreases_epsilon():
    controller = HMCTuningCandidateSetController(_scope(), _config())

    def outcome(work, candidate):
        if work.stage == "verification" and candidate.parent_candidate_id is None and candidate.leapfrog_steps == 5:
            return {"decision": "repair_step_lower", "acceptance": 0.40}
        return {"decision": "passed", "acceptance": 0.70}

    result = controller.run(outcome)
    action = next(item for item in result.repair_actions if item.exact_l == 5)
    assert action.new_epsilon < action.old_epsilon
    parent = next(item for item in result.candidates if item.candidate_id == action.parent_candidate_id)
    child = next(item for item in result.candidates if item.candidate_id == action.child_candidate_id)
    assert child.leapfrog_steps == parent.leapfrog_steps == 5
    assert child.candidate_family_id == parent.candidate_family_id


def test_family_limit_is_typed_and_does_not_switch_l():
    scope = replace(_scope(), max_repairs_per_family=0)
    controller = HMCTuningCandidateSetController(scope, _config())

    def outcome(work, candidate):
        if work.stage == "verification" and candidate.leapfrog_steps == 3:
            return {"decision": "repair_step_higher", "acceptance": 0.90}
        return {"decision": "passed"}

    result = controller.run(outcome)
    assert result.repair_actions == ()
    assert any(event.get("event") == "repair_limit_exhausted" for event in result.accounting_events)
    assert all(candidate.leapfrog_steps in {3, 5} for candidate in result.candidates)

"""New repair semantics keep the physical request, not the old step count."""
from copy import deepcopy
from dataclasses import replace

import pytest

from bayesfilter.inference.hmc_candidate_set_artifacts import (
    _validate_result_payload, candidate_set_result_payload,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig, HMCTuningCandidateSetController,
)
from tests.test_modern_hmc_pipeline import setup


def repaired_result(*, direction="repair_step_lower", max_steps=30):
    binding, base = setup(family="mams")
    policy = replace(base.replicated_acceptance_policy, max_candidates=3)
    config = replace(base, primary_l_grid=(9,), epsilon_by_l=((9, (.2,)),),
                     max_candidates=3, total_budget_units=12,
                     replicated_acceptance_policy=policy,
                     repair_trajectory_policy="preserve_mean_duration",
                     repair_mean_duration_max_steps=max_steps)
    scope = replace(binding.scope, max_repairs_per_family=2, repair_factor=2.)
    controller = HMCTuningCandidateSetController(scope, config)

    def outcome(work, candidate):
        decision = (direction if work.stage == "verification"
                    and candidate.parent_candidate_id is None else "passed")
        return {"decision": decision, "acceptance": .7,
                "trial_range": work.trial_range,
                "evidence_unit": "independent_fixed_horizon_trial"}

    return controller.run(outcome)


@pytest.mark.parametrize("direction,expected_epsilon,expected_count", [
    ("repair_step_lower", .1, 19),
    ("repair_step_higher", .4, 4),
])
def test_repair_preserves_requested_mean_duration_and_replays(direction, expected_epsilon, expected_count):
    result = repaired_result(direction=direction)
    assert len(result.repair_actions) == 1
    action = result.repair_actions[0]
    parent, child = result.candidates
    assert action.payload()["schema"] == "bayesfilter.hmc_repair_action.v2"
    assert action.parent_exact_l == 9 and action.exact_l == expected_count
    assert action.requested_mean_duration == pytest.approx(1.)
    assert child.leapfrog_steps == expected_count
    assert child.epsilon == pytest.approx(expected_epsilon)
    assert abs(child.epsilon*(child.leapfrog_steps+1)/2 - 1.) <= child.epsilon/4+1e-12
    assert parent.candidate_family_id == child.candidate_family_id
    assert child.candidate_id in result.verified_candidate_ids
    assert result.replay_candidate(child.candidate_id) == child
    payload = candidate_set_result_payload(result)
    _validate_result_payload(payload)
    resumed = HMCTuningCandidateSetController.from_result_payload(result.payload())
    assert resumed.result().repair_actions == result.repair_actions


def test_infeasible_duration_repair_preserves_failure_without_inventing_child():
    result = repaired_result(max_steps=12)
    assert len(result.candidates) == 1
    assert result.repair_actions == ()
    assert any(e.get("event") == "repair_limit_exhausted"
               and e.get("reason") == "trajectory_support"
               for e in result.accounting_events)


def test_legacy_same_count_repair_remains_version_one():
    binding, base = setup(family="mams")
    policy = replace(base.replicated_acceptance_policy, max_candidates=2)
    config = replace(base, primary_l_grid=(9,), epsilon_by_l=((9, (.2,)),),
                     max_candidates=2, total_budget_units=10,
                     replicated_acceptance_policy=policy)
    scope = replace(binding.scope, max_repairs_per_family=2, repair_factor=2.)
    controller = HMCTuningCandidateSetController(scope, config)
    def outcome(work, candidate):
        decision = ("repair_step_lower" if work.stage == "verification"
                    and candidate.parent_candidate_id is None else "passed")
        return {"decision": decision, "acceptance": .7,
                "trial_range": work.trial_range,
                "evidence_unit": "independent_fixed_horizon_trial"}
    result = controller.run(outcome)
    assert result.candidates[1].leapfrog_steps == 9
    assert result.repair_actions[0].payload()["schema"] == "bayesfilter.hmc_repair_action.v1"
    _validate_result_payload(candidate_set_result_payload(result))


def test_duration_repair_policy_rejects_legacy_scope_and_schema_relabeling():
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCCandidateSetScope
    binding, base = setup(family="mams")
    policy = replace(base.replicated_acceptance_policy, max_candidates=3)
    config = replace(base, max_candidates=3, replicated_acceptance_policy=policy,
                     repair_trajectory_policy="preserve_mean_duration",
                     repair_mean_duration_max_steps=30)
    with pytest.raises(ValueError, match="typed uniform-duration scope"):
        HMCTuningCandidateSetController(
            replace(binding.scope, transition_profile=None), config)
    body = dict(config.payload())
    body["schema"] = "bayesfilter.hmc_controller_config.v2"
    with pytest.raises(ValueError, match="legacy controller"):
        HMCControllerConfig.from_payload(body)
    body = dict(config.payload())
    body["repair_trajectory_policy"] = "fixed_count"
    with pytest.raises(ValueError, match="duration-preserving controller"):
        HMCControllerConfig.from_payload(body)


def test_repeated_repair_preserves_first_duration_request_not_rounded_child_mean():
    binding, base = setup(family="mams")
    policy = replace(base.replicated_acceptance_policy, max_candidates=3)
    config = replace(base, primary_l_grid=(3,), epsilon_by_l=((3, (1.,)),),
                     max_candidates=3, total_budget_units=12,
                     replicated_acceptance_policy=policy,
                     repair_trajectory_policy="preserve_mean_duration",
                     repair_mean_duration_max_steps=10)
    scope = replace(binding.scope, max_repairs_per_family=2, repair_factor=1.17)
    controller = HMCTuningCandidateSetController(scope, config)

    def outcome(work, candidate):
        return {"decision": ("repair_step_lower" if work.stage == "verification"
                              and candidate.creation_ordinal < 3 else "passed"),
                "acceptance": .7, "trial_range": work.trial_range,
                "evidence_unit": "independent_fixed_horizon_trial"}

    result = controller.run(outcome)
    assert len(result.repair_actions) == 2
    parent, first_child, second_child = result.candidates
    assert first_child.leapfrog_steps == 4
    assert first_child.epsilon*(first_child.leapfrog_steps+1)/2 > 2.
    assert second_child.leapfrog_steps == 4
    assert all(action.requested_mean_duration == 2. for action in result.repair_actions)
    assert second_child.candidate_id in result.verified_candidate_ids
    payload = candidate_set_result_payload(result)
    _validate_result_payload(payload)
    tampered = deepcopy(payload)
    tampered["repair_actions"][1]["requested_mean_duration"] = (
        first_child.epsilon*(first_child.leapfrog_steps+1)/2)
    with pytest.raises(ValueError, match="requested duration mismatch"):
        _validate_result_payload(tampered)

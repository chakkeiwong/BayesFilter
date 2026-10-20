"""Controller counterexamples for verification starvation; no sampler evidence."""
from dataclasses import replace

import pytest

from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController
from tests.test_hmc_candidate_set_tuning import _scope,_config
from tests.test_hmc_acceptance_protocol import policy


def controller():
    config = replace(_config(grid=(3,5),epsilons=((3,(1.2,)),(5,(1.2,)))),
        evidence_rungs=(1,2,4),max_candidates=8,
        replicated_acceptance_policy=policy(base_repetitions=1,max_repetitions=4,max_candidates=8))
    return HMCTuningCandidateSetController(_scope(),config)


def observer(kind, *, extend_verification=False):
    def observe(work,candidate):
        if candidate.leapfrog_steps == 3:
            decision = ('inconclusive_evidence' if extend_verification
                        and work.stage == 'verification' and work.evidence_rung == 0 else 'passed')
        elif kind == 'repair' and candidate.parent_candidate_id is None:
            decision = 'repair_step_lower'
        else:
            decision = 'inconclusive_evidence'
        return {'decision':decision,'acceptance':.3 if decision == 'repair_step_lower' else .7,
                'draw_range':(0,0),'trial_range':work.trial_range,
                'evidence_unit':'independent_fixed_horizon_trial'}
    return observe


@pytest.mark.parametrize('kind',['repair','extension'])
def test_broad_initial_measurements_then_verification_before_optional_work(kind):
    result = controller().run(observer(kind),max_work_items=3)
    rows = result.verification_receipts
    assert [(r.exact_l,r.stage) for r in rows] == [(3,'measurement'),(5,'measurement'),(3,'verification')]
    assert len(result.verified_candidate_ids) == 1
    assert result.completion_status == 'partial_budget'
    assert result.search_state['controller_policy_version'] == 4


def test_inconclusive_verification_is_extended_without_inventing_membership():
    c = controller()
    partial = c.run(observer('extension',extend_verification=True),max_work_items=3)
    assert not partial.verified_candidate_ids
    resumed = HMCTuningCandidateSetController.from_result_payload(partial.payload())
    result = resumed.run(observer('extension',extend_verification=True),max_work_items=1)
    assert result.verification_receipts[-1].stage == 'verification'
    assert result.verification_receipts[-1].trial_range == (1,2)
    assert len(result.verified_candidate_ids) == 1


@pytest.mark.parametrize('version,expected_stage',[(3,'measurement'),(4,'verification')])
def test_resume_preserves_recorded_controller_order(version,expected_stage):
    c = controller()
    c._controller_policy_version = version  # Reconstruct the historical mechanics baseline.
    partial = c.run(observer('repair'),max_work_items=2)
    resumed = HMCTuningCandidateSetController.from_result_payload(partial.payload())
    result = resumed.run(observer('repair'),max_work_items=1)
    assert result.search_state['controller_policy_version'] == version
    assert result.verification_receipts[-1].stage == expected_stage


def test_sufficient_work_preserves_terminal_pair_states_without_ranking():
    outcomes = []
    for version in (3,4):
        c = controller();c._controller_policy_version = version
        result = c.run(observer('repair'))
        assert result.completion_status == 'complete'
        outcomes.append({(row.leapfrog_steps,row.epsilon):result.candidate_states[row.candidate_id]
                         for row in result.candidates})
    assert outcomes[0] == outcomes[1]


def test_legacy_default_and_unsupported_version_cannot_silently_switch_order():
    legacy = HMCTuningCandidateSetController(_scope(),_config())
    assert legacy.result().search_state['controller_policy_version'] == 3
    payload = legacy.result().payload()
    payload['search_state']['controller_policy_version'] = 4
    with pytest.raises(ValueError,match='requires a replicated'):
        HMCTuningCandidateSetController.from_result_payload(payload)
    payload['search_state']['controller_policy_version'] = 99
    with pytest.raises(ValueError,match='historical controller'):
        HMCTuningCandidateSetController.from_result_payload(payload)

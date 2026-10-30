"""Observable root-cause regressions, including activated diagnostic defects."""
from dataclasses import replace

import numpy as np  # Independent test assertions only.
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_statistics import replicated_acceptance_statistics
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig, HMCCandidateSetScope, HMCTuningCandidateSetController,
)
from tests.test_hmc_acceptance_protocol import policy


def test_adverse_fresh_verification_cannot_inherit_measurement_qualification():
    p = policy(base_repetitions=256,max_repetitions=256,max_candidates=4)
    scope = HMCCandidateSetScope(scope_id="adverse-verification",search_id="fault-fixture",
        target_signature="controller-only",mass_signature="identity",coordinate_system="ordinary",
        start_bank_signature="four",warmup_protocol=p.identity,epsilon_domain=(.05,5.),
        repair_factor=2.,max_repairs_per_family=1)
    search = HMCControllerConfig(primary_l_grid=(1,3),epsilon_by_l=((1,(1.3,)),(3,(1.3,))),
        max_candidates=4,total_budget_units=48,repair_reserve_units=8,evidence_rungs=(1,),
        replicated_acceptance_policy=p)
    calls = []
    def observe(work,candidate):
        # Deliberately opposing stage records test the controller's reaction.
        # This is NOT calibration under a law whose mean changes between stages.
        adverse = candidate.leapfrog_steps == 1 and work.stage == "verification"
        score = .4 if adverse else .7
        report = replicated_acceptance_statistics([[score]*4]*256,policy=p,
            stage=work.stage,evidence_rungs=(1,))
        calls.append((candidate.candidate_id,work.stage,candidate.epsilon))
        return {"decision":report["acceptance_decision"],"acceptance":score,
                "trial_range":work.trial_range,"draw_range":(0,0),
                "evidence_unit":"independent_fixed_horizon_trial"}
    result = HMCTuningCandidateSetController(scope,search).run(observe)
    original = {c.leapfrog_steps:c for c in result.candidates if c.parent_candidate_id is None}
    assert original[1].candidate_id not in result.verified_candidate_ids
    assert original[3].candidate_id in result.verified_candidate_ids
    children = [c for c in result.candidates if c.parent_candidate_id == original[1].candidate_id]
    assert len(children)==1 and children[0].epsilon==.65
    assert [(stage,epsilon) for cid,stage,epsilon in calls if cid==children[0].candidate_id] == [
        ("measurement",.65),("verification",.65)]
    assert children[0].candidate_id not in result.verified_candidate_ids


def test_weighted_start_permutation_preserves_the_estimand_and_full_covariance():
    p = policy(base_repetitions=64,max_repetitions=64,start_weights=(.125,.125,.25,.5))
    rows=np.array([[.61,.68,.73,.80],[.69,.72,.65,.74]]*32)
    permutation=[3,1,0,2]
    args=dict(stage="verification",evidence_rungs=(1,))
    a=replicated_acceptance_statistics(rows,policy=p,**args)
    b=replicated_acceptance_statistics(rows[:,permutation],
        policy=replace(p,start_weights=tuple(p.start_weights[i] for i in permutation)),**args)
    assert a["acceptance_decision"]==b["acceptance_decision"]
    np.testing.assert_allclose(a["pooled_interval"],b["pooled_interval"],atol=1e-12,rtol=0)
    vector=np.column_stack([rows@np.array(p.start_weights),rows])
    np.testing.assert_allclose(a["covariance_of_mean"],np.cov(vector.T,ddof=1)/64,atol=1e-15,rtol=0)


@pytest.mark.parametrize("target,control",[("gaussian","wrong_score"),("gamma","omit_jacobian"),
                                          ("dirichlet","omit_jacobian")])
def test_independent_model_oracle_rejects_activated_score_and_jacobian_mutations(target,control):
    from bayesfilter.testing.acceptance_decision_models import reference_checks
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    correct=ValidationTarget(target)
    positions=tf.constant([[s]*correct.parameter_dim for s in [-.8,-.2,.3,.9]],tf.float64)
    assert reference_checks(correct,positions)["density_max_abs_error"] < 1e-6
    broken=ValidationTarget(target,control=control)
    with pytest.raises(tf.errors.InvalidArgumentError,match="mismatch"):
        reference_checks(broken,positions)


@pytest.mark.parametrize("target",["ssm_lgssm_qr","ssm_nonlinear"])
@pytest.mark.parametrize("missing",["block","all"])
def test_unsupported_missing_data_is_rejected_before_tuning(target,missing):
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.ssm_targets import get_ssm_profile
    data=[0.]*get_ssm_profile(target).horizon
    data[len(data)//2]=float("nan")
    if missing == "all":
        data=[float("nan")]*len(data)
    with pytest.raises(tf.errors.InvalidArgumentError,match="finite"):
        ValidationTarget(target,data=data)


@pytest.mark.parametrize("target",["ssm_lgssm_qr","ssm_nonlinear"])
@pytest.mark.parametrize("parameter,value",[("initial_covariance",0.),("initial_covariance",1e12),
                                           ("initial_mean",100.),("unrecognized_filter_option",True)])
def test_unsupported_initial_laws_cannot_be_silently_ignored(target,parameter,value):
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    with pytest.raises(ValueError,match="unsupported state-space fixture parameters"):
        ValidationTarget(target,parameters={parameter:value})


@pytest.mark.parametrize("target",["ssm_lgssm_qr","ssm_nonlinear"])
@pytest.mark.parametrize("data",[[],[.1]])
def test_fixed_horizon_fixture_rejects_unsupported_short_series(target,data):
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    with pytest.raises(ValueError,match="expects.*observations"):
        ValidationTarget(target,data=data)


def test_fp32_tiny_step_cannot_supply_movement_despite_rounded_unit_acceptance():
    from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_trial_health
    # Finite precision can make q+epsilon*p equal q exactly; favorable alpha
    # cannot promote this state sequence. This is health mechanics, not HMC.
    initial=tf.ones([4,2],tf.float32)
    after=initial+tf.constant(1e-10,tf.float32)
    tf.debugging.assert_equal(after,initial)
    result=evaluate_hmc_trial_health(samples=tf.repeat(after[None,:,:],65,axis=0),
        log_accept_ratio=tf.zeros([65,4],tf.float32),is_accepted=tf.ones([65,4],tf.bool),
        policy=HMCAcceptancePolicy())
    assert "movement_gate_failed" in result.candidate_promotion_vetoes


@pytest.mark.parametrize("report",["high","missing","raises"])
def test_replicated_trial_decisions_do_not_depend_on_rhat(report,monkeypatch):
    import bayesfilter.inference.hmc as hmc
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from tests.test_hmc_acceptance_trials import _replicated_setup
    if report=="raises":
        def replacement(*args,**kwargs):
            raise RuntimeError("injected unavailable R-hat")
    else:
        replacement=lambda *args,**kwargs:{"rhat_max":100. if report=="high" else None,"passed":False}
    monkeypatch.setattr(hmc,"_rhat_summary_from_retained_samples",replacement)
    binding,search=_replicated_setup()
    result=run_typed_hmc_candidate_set(binding.typed_adapter,search).result
    assert set(result.candidate_states.values())=={"inconclusive_at_cap"}
    assert all(r.decision=="inconclusive_evidence" for r in result.verification_receipts)
    for row in binding._evidence.values():
        assert row["rhat_reporting_only"]["role"]=="reporting_only"
        if report=="raises":
            assert "RuntimeError" in row["rhat_reporting_only"]["unavailable_reason"]


def test_statistical_admission_does_not_assume_epsilon_response_is_monotone():
    # Same frozen protocol and actual bounded statistics, but a resonant response
    # table. A direction proposes a child; it cannot qualify an unmeasured pair.
    p=policy(base_repetitions=256,max_repetitions=256,max_candidates=8)
    values=[(.4,.95),(.8,.40),(1.6,.70)]
    reports={eps:replicated_acceptance_statistics([[mean]*4]*256,policy=p,
        stage="measurement",evidence_rungs=(1,))["acceptance_decision"] for eps,mean in values}
    assert reports=={.4:"repair_step_higher",.8:"repair_step_lower",1.6:"passed"}

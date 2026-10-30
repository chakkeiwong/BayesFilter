"""Scheduled positive and negative v7 public-model regressions.

These deliberately small fixed allocations test delivery/export and honest
negative outcomes. Posterior accuracy, scientific superiority and optimal
geometry are not asserted. CPU is an explicit reference execution tier.
"""
import pytest

from bayesfilter.testing.acceptance_decision_models import run_model
from bayesfilter.testing.acceptance_validation_inventory import model_configuration
from tests.test_hmc_acceptance_protocol import policy


@pytest.mark.parametrize("case",["k0","funnel_exact","nonlinear","lgssm_qr","funnel_residual"])
def test_positive_prepared_model_delivers_verifies_exports_and_reloads(tmp_path,case):
    p=policy(base_repetitions=32,max_repetitions=128,max_candidates=2)
    cfg=model_configuration(case,policy_payload=p.payload(),evidence_rungs=(1,2,4),
        seed=(20261002,1401),wall_seconds=270,classification="regression",expected_outcome="positive_delivery")
    if case=="nonlinear":
        cfg["epsilon_by_l"]=[[1,[.15]]]
        cfg["geometry"]={"kind":"explicit_diagonal","center":[0.]*3,"scale":[1.]*3}
        cfg["active_starts"]=[[.6,.4,.7],[.65,.45,.75],[.55,.35,.65],[.7,.5,.8]]
        cfg["provenance"]+='; nonlinear geometry and step from frozen October 2 development pricing; fresh regression streams'
    if case=="lgssm_qr":
        import math
        cfg["epsilon_by_l"]=[[1,[.95]]]
        cfg["geometry"]={"kind":"explicit_diagonal","center":[math.atanh(.6/.999),math.log(.2)],"scale":[1.,1.]}
        cfg["active_starts"]=[[.5,-1.5],[.55,-1.45],[.45,-1.55],[.6,-1.4]]
        cfg["provenance"]+='; full QR geometry, starts and step from frozen October 2 development pricing; fresh regression streams'
    result=run_model(cfg,tmp_path/'model')
    assert result["expectation_met"] is True, result["receipts"]
    assert result["checkpoint_recomputed"]
    assert result["verified_candidate_ids"]
    assert {r["stage"] for r in result["receipts"]}=={"measurement","verification"}
    costs=result["evidence_accounting"]
    assert costs["unique_complete_trials"]>=2*p.base_repetitions
    assert costs["attempted_transitions"]==costs["completed_trial_transitions"]
    if case == "k0":
        _check_location_posterior_reference(cfg, tmp_path/'model', tmp_path)


def _check_location_posterior_reference(cfg, model_root, output):
    """Independent Gaussian posterior moments after checked v7 member reload."""
    import tensorflow as tf
    from bayesfilter.inference import load_hmc_candidate_retained_runner
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy, assess_posterior
    from bayesfilter.inference.hmc_precision import mean_precision
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.references.ssm import location_posterior

    target = ValidationTarget(cfg['target'],cfg['parameters'],cfg['data'])
    member = load_hmc_candidate_retained_runner(next(model_root.glob('*-member.json')),adapter=target)
    discarded = member.run(num_results=256,seed=(20261002,1801),output_dir=output/'discarded')
    retained = member.run(num_results=2048,seed=(20261002,1802),output_dir=output/'retained',
        previous_archive=discarded['archive_path'])
    samples = retained['position_samples']
    reference_mean, reference_variance = location_posterior(cfg['data'])
    # Center the second quantity at the exact mean, not the random sample mean.
    values = tf.concat([samples,tf.square(samples-reference_mean)],axis=-1)
    precision = mean_precision(values)
    errors = tf.abs(precision['estimate']-tf.constant([reference_mean,reference_variance],tf.float64))
    tf.debugging.assert_all_finite(precision['mcse'],'missing posterior MCSE')
    tf.debugging.assert_positive(precision['mcse'])
    # Predeclared diagnostic four-MCSE tolerance under this finite-moment Gaussian
    # reference, not a finite-sample error guarantee or coverage calibration.
    assert bool(tf.reduce_all(errors <= 4.*precision['mcse']))
    assessment = assess_posterior(samples,('location',),stage='retained',
        rhat=rank_normalized_split_rhat_summary(samples,rhat_max=1.01),
        policy=HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=128.,retained_tail_ess_min=128.))
    assert assessment['passed'], assessment
    assert assessment['sample_counts']['draws_per_chain']==2048


@pytest.mark.parametrize("case",["funnel_centered","k2","k3","cauchy","simplex"])
def test_deliberately_underinformed_model_abstains_at_its_frozen_cap(tmp_path,case):
    p=policy(base_repetitions=1,max_repetitions=1,max_candidates=2)
    cfg=model_configuration(case,policy_payload=p.payload(),evidence_rungs=(1,),
        seed=(20261002,1421),wall_seconds=120,classification="regression",expected_outcome="inconclusive_at_cap")
    result=run_model(cfg,tmp_path/'model')
    assert result["expectation_met"] is True, result
    assert not result["positive_delivery"] and not result["verified_candidate_ids"]
    assert result["evidence_accounting"]["unique_complete_trials"]==1


def test_local_mixture_qualification_does_not_certify_posterior_exploration(tmp_path):
    """The independent mode probability must catch a locally healthy kernel."""
    import math
    import tensorflow as tf
    from bayesfilter.inference import load_hmc_candidate_retained_runner
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.inference.hmc_posterior_assessment import (
        HMCPosteriorAssessmentPolicy, assess_posterior,
    )
    from bayesfilter.testing.inference_validation.targets import ValidationTarget

    p=policy(base_repetitions=32,max_repetitions=128,max_candidates=2)
    cfg=model_configuration("mixture",policy_payload=p.payload(),evidence_rungs=(1,2,4),
        seed=(20261002,1511),wall_seconds=180,classification="regression",expected_outcome="positive_delivery")
    root=tmp_path/'model'
    result=run_model(cfg,root)
    assert result['expectation_met'] is True
    original=(root/'result.json').read_bytes()
    target=ValidationTarget(cfg['target'],cfg['parameters'],cfg['data'])
    member=load_hmc_candidate_retained_runner(next(root.glob('*-member.json')),adapter=target)
    # Fixed, diagnostic windows; this is a negative control, not enough-burnin evidence.
    discarded=member.run(num_results=128,seed=(20261002,1601),output_dir=tmp_path/'discarded')
    retained=member.run(num_results=512,seed=(20261002,1602),output_dir=tmp_path/'retained',
        previous_archive=discarded['archive_path'])
    samples=retained['position_samples']
    expected=.3*.5*(1.+math.erf(5./math.sqrt(2.)))+.7*.5*math.erfc(5./math.sqrt(2.))
    observed=float(tf.reduce_mean(tf.cast(samples[...,0]<0.,tf.float64)))
    # A coarse missing-mode screen, not an iid confidence bound on correlated draws.
    assert observed>.9 and abs(observed-expected)>.5
    assessment=assess_posterior(samples,('x0','x1'),stage='retained',
        rhat=rank_normalized_split_rhat_summary(samples,rhat_max=1.01),
        policy=HMCPosteriorAssessmentPolicy(quantities_id='mixture_left_mode_probability_v1',
            binary_quantity_names=('left_mode_probability',)),
        quantities_fn=lambda q:{'left_mode_probability':tf.cast(q[...,0]<0.,tf.float64)})
    assert not assessment['passed']
    assert 'binary_outcomes_observed' in assessment['failed_checks']
    assert (root/'result.json').read_bytes()==original
    assert result['verified_candidate_ids']

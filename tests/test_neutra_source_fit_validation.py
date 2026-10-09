"""CPU-only numerical/reference checks, with no fitted-map promotion claim."""
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
from bayesfilter.inference.hmc_posterior_assessment import assess_posterior
from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_scientific_design import target_catalog, CALIBRATION_TARGETS
from bayesfilter.testing.neutra_source_fit_validation import (
    DevelopmentAdapter, DevelopmentQuantities, GAUSSIAN_CONTROL,
    development_protocol, identity_control_payload,
)


@pytest.mark.parametrize('name',('gaussian',*CALIBRATION_TARGETS))
def test_batch_and_scalar_adapter_scores_match_finite_differences(name):
    spec=GAUSSIAN_CONTROL if name=='gaussian' else target_catalog()[name]
    evaluator=ExactTargetEvaluator(spec)
    adapter=DevelopmentAdapter(evaluator.target,name)
    x=tf.constant([[-1.,.5],[.3,-.2],[2.,1.]],tf.float64)
    compiled=tf.function(adapter.log_prob_and_grad,
        input_signature=[tf.TensorSpec([None,2],tf.float64)],jit_compile=True,autograph=False)
    value,score=compiled(x)
    for j in range(2):
        step=tf.one_hot(j,2,dtype=tf.float64)*tf.constant(1e-5,tf.float64)
        difference=(evaluator.target.log_prob(x+step)-evaluator.target.log_prob(x-step))/tf.constant(2e-5,tf.float64)
        tf.debugging.assert_near(score[:,j],difference,atol=1e-7,rtol=1e-6)
    scalar_value,scalar_score=adapter.log_prob_and_grad(x[1])
    tf.debugging.assert_near(scalar_value,value[1],atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(scalar_score,score[1],atol=1e-12,rtol=1e-12)


def test_identity_control_has_zero_log_jacobian_and_gaussian_score():
    evaluator=ExactTargetEvaluator(GAUSSIAN_CONTROL)
    payload=identity_control_payload(evaluator,11)
    flow=load_frozen_neutra_artifact(payload,expected_target_signature=evaluator.target.signature).transport
    z=tf.constant([[-2.,1.],[.3,-.4],[2.,3.]],tf.float64)
    tf.debugging.assert_near(flow.forward_batch(z),z,atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(flow.inverse_theta_to_z_batch(z),z,atol=1e-12,rtol=1e-12)
    _,score=DevelopmentAdapter(evaluator.target,'control').log_prob_and_grad(flow.forward_batch(z))
    tf.debugging.assert_near(score,-z,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('name',CALIBRATION_TARGETS)
def test_unique_mixture_features_preserve_responsibilities_and_second_moments(name):
    evaluator=ExactTargetEvaluator(target_catalog()[name])
    quantities=DevelopmentQuantities(evaluator)
    x=tf.constant([[-2.,1.],[.3,-.4],[2.,3.]],tf.float64)
    features=quantities.program(x)
    k=len(evaluator.specification['weights'])
    assert len(quantities.indices)==7*k
    assert len(set(quantities.indices))==7*k
    assert len(set(quantities.all_names))==len(quantities.all_names)
    assert not quantities.binary_names
    r=features[:,2:2+k]
    tf.debugging.assert_near(tf.reduce_sum(r,axis=1),tf.ones([3],tf.float64),atol=1e-12,rtol=1e-12)
    z=(evaluator.unwarp(x)[:,None,:]-evaluator.centers)/tf.sqrt(evaluator.variances)[None,:,None]
    for i in range(k):
        for j,m in ((0,0),(0,1),(1,1)):
            index=quantities.all_names.index(f'component_{i}_second_{j}_{m}')
            tf.debugging.assert_near(features[:,index],r[:,i]*z[:,i,j]*z[:,i,m],atol=1e-12,rtol=1e-12)


def test_log_odds_preserve_probabilities_without_saturated_quantile_values():
    spec={'kind':'isotropic_mixture','dimension':2,'centers':[[-10.,0.],[10.,0.]],
          'variances':[1.,1.],'weights':[.5,.5]}
    q=DevelopmentQuantities(ExactTargetEvaluator(spec))
    x=tf.constant([[-11.,0.],[-10.,0.],[-.1,0.],[.1,0.],[10.,0.],[11.,0.]],tf.float64)
    r=q.program(x)[:,2:4]
    odds=q.diagnostic_program(x)[:,2:4]
    tf.debugging.assert_all_finite(odds,'stable component log odds')
    assert float(r[0,0])==float(r[1,0])==1.
    assert float(odds[0,0])!=float(odds[1,0])
    tf.debugging.assert_near(tf.math.sigmoid(odds),r,atol=1e-14,rtol=1e-14)


@pytest.mark.parametrize('name',('gaussian',*CALIBRATION_TARGETS))
def test_iid_reference_passes_and_shifted_draws_fail_with_declared_quantities(name):
    spec=GAUSSIAN_CONTROL if name=='gaussian' else target_catalog()[name]
    index=0 if name=='gaussian' else CALIBRATION_TARGETS.index(name)
    evaluator=ExactTargetEvaluator(spec)
    quantities=DevelopmentQuantities(evaluator)
    iid=tf.reshape(evaluator.sample(16384,tf.constant([31000+index,6202])),[4096,4,2])
    reference=evaluator.sample(32768,tf.constant([31000+index,6203]))
    screen=assess_posterior(iid,quantities.physical_names,policy=quantities.policy(),stage='retained',
        rhat=rank_normalized_split_rhat_summary(iid,rhat_max=1.01),quantities_fn=quantities.additional)
    assert screen['passed'],screen
    assert quantities.original_mean_precision(iid)['passed']
    assert quantities.reference_check(iid,reference)['passed']
    assert not quantities.reference_check(iid+1.,reference)['passed']


def test_public_protocol_has_fixed_latent_mass_and_bounded_budgets():
    from bayesfilter.inference import HMC_TUNING_INTERFACE_CAPABILITIES
    tuning,search,execution=development_protocol(11,'development')
    assert tuning.chain_count==4 and execution.chain_mode=='batched'
    assert execution.use_xla and tuning.use_xla
    assert search.max_wall_time_seconds==900.
    assert development_protocol(11,'control',control=True)[1].max_wall_time_seconds==300.
    assert execution.seed==(11,6301)
    assert HMC_TUNING_INTERFACE_CAPABILITIES

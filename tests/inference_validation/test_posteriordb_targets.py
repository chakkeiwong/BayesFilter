"""Independent density/analytic-score checks for the matched consumer fixtures."""
import numpy as np
import pytest
from scipy import stats
import tensorflow as tf

from bayesfilter.testing.inference_validation.posteriordb_targets import PosteriordbTarget


def case_data(case):
    if case.startswith("eight_schools"):
        return {"J":8,"y":[28,8,-3,7,-1,1,18,12],"sigma":[15,10,16,11,9,11,10,18]}
    rng = np.random.default_rng(34)
    x = rng.normal(size=(100,5))
    return {"N":100,"D":5,"X":x.tolist(),"y":(x@np.arange(5)/3+rng.normal(size=100)).tolist()}


def reference(case,data,q):
    y = np.array(data["y"])
    if case.startswith("eight_schools"):
        mu,ell,z = q[:2].tolist()+[q[2:]]
        tau = np.exp(ell)
        sigma = np.array(data["sigma"])
        residual = y-mu-tau*z
        value = stats.norm.logpdf(mu,scale=5)+stats.halfcauchy.logpdf(tau,scale=5)+ell
        value += stats.norm.logpdf(z).sum()+stats.norm.logpdf(y,loc=mu+tau*z,scale=sigma).sum()
        score = np.r_[-mu/25+np.sum(residual/sigma**2),
            1-2*tau**2/(25+tau**2)+np.sum(residual*tau*z/sigma**2),-z+tau*residual/sigma**2]
    else:
        beta,ell = q[:5],q[5]
        sigma = np.exp(ell)
        x = np.array(data["X"])
        residual = y-x@beta
        value = stats.norm.logpdf(beta,scale=10).sum()+stats.halfnorm.logpdf(sigma,scale=10)+ell
        value += stats.norm.logpdf(y,loc=x@beta,scale=sigma).sum()
        score = np.r_[-beta/100+x.T@residual/sigma**2,1-len(y)-sigma**2/100+sum(residual**2)/sigma**2]
    return value,score


@pytest.mark.parametrize("case",["eight_schools-eight_schools_noncentered","sblrc-blr"])
def test_density_score_and_model_coordinate_order(case):
    data = case_data(case)
    target = PosteriordbTarget(case,data,jit_compile=False)
    positions = np.random.default_rng(520).normal(size=(5,target.parameter_dim))
    actual,score = target.log_prob_and_grad(positions)
    expected = [reference(case,data,q) for q in positions]
    np.testing.assert_allclose(actual.numpy()-actual[0],np.array([r[0] for r in expected])-expected[0][0],rtol=1.e-11,atol=1.e-10)
    np.testing.assert_allclose(score,[r[1] for r in expected],rtol=1.e-11,atol=1.e-10)
    scalar = target.log_prob_and_grad(positions[0])
    np.testing.assert_allclose(scalar[1],score[0])
    model = target.to_model(tf.constant(positions,tf.float64)).numpy()
    if case.startswith("eight_schools"):
        assert target.model_names[:2] == ("mu","tau")
        np.testing.assert_allclose(model[:,2:],positions[:,:1]+np.exp(positions[:,1:2])*positions[:,2:])
    else:
        assert target.model_names[-1] == "sigma"
        np.testing.assert_allclose(model[:,-1],np.exp(positions[:,-1]))
    changed = dict(data,y=[data["y"][0]+1,*data["y"][1:]])
    assert target.adapter_signature() != PosteriordbTarget(case,changed,jit_compile=False).adapter_signature()


def test_real_consumer_adapter_reaches_public_tuner(tmp_path):
    from bayesfilter.inference import (HMCControllerConfig,HMCCandidateExecutionConfig,HMCAcceptancePolicy,
        PrecomputedMassArtifact,bind_hmc_candidate_set_execution,tune_hmc_kernel)
    target = PosteriordbTarget("eight_schools-eight_schools_noncentered",case_data("eight_schools"),jit_compile=False)
    starts = tf.zeros((4,10),tf.float64)
    mass = PrecomputedMassArtifact(position=[0.]*10,covariance=tf.eye(10,dtype=tf.float64),
        factor=tf.eye(10,dtype=tf.float64),adapter_signature=target.adapter_signature(),
        position_role="test_origin",covariance_source="identity smoke; not automatic preparation")
    binding = bind_hmc_candidate_set_execution(adapter=target,initial_position=starts,mass_artifact=mass,
        target_scope="inference_validation",scope_id="posteriordb-test",search_id="public-endpoint",
        epsilon_domain=(.01,2.),repair_factor=2.,max_repairs_per_family=0,
        target_lineage={"model":target.case,"data":target.data},source_paths=[__file__],
        config=HMCCandidateExecutionConfig(measurement_num_results=64,verification_num_results=64,
            num_warmup_steps=8,seed=(813,915),use_xla=False,non_xla_reason="CPU endpoint unit smoke",
            target_status_trace_policy="none",acceptance_policy=HMCAcceptancePolicy()))
    run = tune_hmc_kernel(adapter=target,initial_position=starts,
        config=HMCControllerConfig(primary_l_grid=(3,),initial_epsilon=.3,pilot_enabled=False,
            refinement_rounds=0,max_candidates=1,total_budget_units=8,repair_reserve_units=1),
        candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path/"tuning")
    assert run.result.scope.target_signature == target.adapter_signature()
    assert run.result.observations
    assert (tmp_path/"tuning/execution_spec.json").exists()


def test_invalid_consumer_inputs_are_rejected():
    data = case_data("eight_schools")
    with pytest.raises(ValueError,match="positive"):
        PosteriordbTarget("eight_schools-eight_schools_noncentered",dict(data,sigma=[0.]*8))
    with pytest.raises(ValueError,match="100 by 5"):
        PosteriordbTarget("sblrc-blr",{"N":5,"D":5})

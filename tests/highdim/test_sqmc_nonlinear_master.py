"""CPU reference/mechanics checks, including the nonlinear consumer call chain."""
import importlib.util
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pytest
import tensorflow as tf

from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.highdim import sqmc_campaign_tf as common

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location("nonlinear_master", ROOT / "docs/benchmarks/run_ledh_nonlinear_master.py")
master = importlib.util.module_from_spec(loader)
loader.loader.exec_module(master)


@pytest.mark.parametrize("name,d,o,p", [("predator_prey",2,2,6),("sir_d18",18,9,3)])
def test_adapter_dimensions_and_transition_tangent(name,d,o,p):
    spec = NonlinearSQMCSpec(name)
    theta = spec.default_theta(tf.float64)
    direction = tf.ones([p],tf.float64) * .01
    model,_ = spec.model(theta,direction)
    state = spec.initial_mean(tf.float64)[None,:]
    dstate = tf.ones([1,d],tf.float64)*.003
    analytical = model.transition_mean_tangent_fn(theta,state,dstate)
    eps = 1e-4
    plus,_ = spec.model(theta+eps*direction,direction)
    minus,_ = spec.model(theta-eps*direction,direction)
    finite_difference = (plus.transition_mean_fn(theta+eps*direction,state+eps*dstate)
                        -minus.transition_mean_fn(theta-eps*direction,state-eps*dstate))/(2*eps)
    tf.debugging.assert_near(analytical,finite_difference,atol=2e-6,rtol=2e-5)
    assert model.observation_fn(state).shape == (1,o)
    assert (spec.dimension,spec.observation_dimension,spec.parameter_count)==(d,o,p)
    for dtype in (tf.float32,tf.float64):
        observed = spec.observations(1,261101,dtype=dtype,jit_compile=False)
        assert observed.shape==(1,o)
        assert bool(tf.reduce_all(tf.math.is_finite(observed)))


@pytest.mark.parametrize("name", ["predator_prey","sir_d18"])
def test_campaign_consumes_shared_executor(monkeypatch,name):
    """Prove 9-observation SIR reaches the same canonical executor as PP."""
    spec = NonlinearSQMCSpec(name)
    seen=[]
    def checked(model,theta,states,covs,noise,obs,**kw):
        seen.append((tuple(obs.shape),kw["moment_safety"],kw["pairwise_steps"]))
        assert model.process_covariance.shape==(spec.dimension,spec.dimension)
        tf.debugging.assert_equal(kw["reset_design"], expected_design)
        return tf.reduce_sum(theta),tf.ones([1],theta.dtype)
    monkeypatch.setattr(common,"canonical_value_and_analytical_score",checked)
    common._kernel.cache_clear()
    controls,design=master.arm_settings("guarded_pairwise",master.BASE)
    n=2*spec.dimension
    expected_design=common.reset_design(n,spec.dimension,tf.float64,design)
    theta=spec.default_theta(tf.float64)
    value,score,valid=common.value_and_score(spec,"iid_dual_cap",controls,theta,
        tf.zeros([1,spec.observation_dimension],tf.float64),261102,n,
        jit_compile=False,reset_design_kind=design)
    assert bool(valid) and score.shape==(spec.parameter_count,)
    assert seen==[((1,spec.observation_dimension),True,4)]
    common._kernel.cache_clear()


def test_reference_scope_and_coordinate_validation():
    scope=dict(target_id="test",observation_sha256="abc",horizon=20,theta=[0.],parameter_names=["theta"])
    ref=dict(scope,kind="numerically_converged",verification="two resolutions",source="reference.json",
             log_likelihood=-2.,score=[1.])
    assert master.reference_for([ref],scope)==ref
    assert master.reference_for([dict(ref,theta=[1.])],scope) is None
    with pytest.raises(ValueError,match="duplicate"):
        master.reference_for([ref,ref],scope)
    with pytest.raises(ValueError,match="invalid reference"):
        master.reference_for([dict(ref,score=[float("nan")])],scope)


def test_arms_retain_caps_and_pairwise():
    unguarded,design=master.arm_settings("richer_pairwise",master.BASE)
    guarded,_=master.arm_settings("guarded_pairwise",master.BASE)
    assert design=="normal_quantiles"
    assert guarded==dict(unguarded,moment_safety=True)
    assert guarded["pairwise_steps"]>0 and guarded["pairwise_rms_cap"]>0
    assert guarded["coordinate_cap_identity_radius"]==8


def test_sir_particle_and_parameter_scope_rejected():
    job=dict(model="sir_d18",particles=1000,horizon=20,theta_points=None,data_seed=1,design_seeds=[2])
    with pytest.raises(ValueError,match="2d"):
        master.validate_job(job)
    job.update(particles=1008,theta_points=[[0.,0.]])
    with pytest.raises(ValueError,match="parameter"):
        master.validate_job(job)


def test_empty_failed_campaign_reporting(tmp_path):
    master.summarize(tmp_path,[dict(directory="failed-worker",status="timeout",wall_seconds=1.)])
    assert json.loads((tmp_path/"rows.json").read_text())==[]
    assert "timeout" in (tmp_path/"results.md").read_text()


def test_controller_timeout_preserves_failure_and_refuses_overwrite(tmp_path,monkeypatch):
    output=tmp_path/'campaign'
    args=master.parser().parse_args(['run','--models','predator_prey','--horizons','1',
        '--particles','4','--data-seeds','1','--design-seeds','2','--routes','iid_dual_cap',
        '--arms','original','--budget-seconds','2','--worker-seconds','1','--output',str(output)])
    signals=[]
    class TimedOutWorker:
        pid=12345
        waits=0
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1:
                raise master.subprocess.TimeoutExpired('mock-worker',timeout)
            return -15
    monkeypatch.setattr(master.subprocess,'Popen',lambda *a,**kw:TimedOutWorker())
    monkeypatch.setattr(master.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(master,'git',lambda *a:'test-checkout')
    assert master.controller(args)==2
    assert signals==[(12345,master.signal.SIGTERM)]
    attempts=json.loads((output/'attempts.json').read_text())
    completion=json.loads((output/'completion.json').read_text())
    assert attempts[0]['status']=='timeout'
    assert not completion['complete'] and completion['attempted_workers']==1
    assert (output/attempts[0]['directory']/'worker.log').exists()
    with pytest.raises(FileExistsError):
        master.controller(args)


def test_nonfinite_budget_is_rejected_before_launch(tmp_path):
    args=master.parser().parse_args(['run','--budget-seconds','nan','--output',str(tmp_path/'none')])
    with pytest.raises(ValueError,match='finite'):
        master.controller(args)
    assert not (tmp_path/'none').exists()


def test_heuristic_comparison_is_paired_and_conditional():
    keys=('model','horizon','particles','data_seed','point_index','route','arm')
    base=dict(model='sir_d18',horizon=20,particles=1008,data_seed=1,point_index=0,
              route='iid_dual_cap',arm='original',design_seed=10,valid=True,
              reference={'kind':'numerically_converged'},score_l2_error=1.,log_likelihood_error=.5)
    candidate=dict(base,arm='guarded_pairwise',score_l2_error=2.,log_likelihood_error=.4)
    unmatched=dict(base,data_seed=2,score_l2_error=100.)
    invalid=dict(candidate,valid=False,data_seed=2)
    outcome=master.heuristic_comparisons([base,candidate,unmatched,invalid],keys)
    assert outcome['status']=='observed_underperformance_promotion_veto'
    assert len(outcome['comparisons'])==1
    comparison=outcome['comparisons'][0]
    assert comparison['n']==1 and comparison['score_error_difference']['mean']==1.
    assert comparison['absolute_likelihood_error_difference']['mean']==pytest.approx(-.1)
    assert master.heuristic_comparisons([dict(base,reference=None)],keys)['comparisons']==[]


def test_fixed_dataset_replay_preserves_values_and_rejects_mismatches(tmp_path):
    spec = NonlinearSQMCSpec('sir_d18')
    observations = tf.reshape(tf.range(18, dtype=tf.float32) / 7., [2, 9])
    digest = master.hashlib.sha256(bytes(tf.io.serialize_tensor(observations).numpy())).hexdigest()
    record = dict(observations=observations.numpy().tolist(), observation_sha256=digest,
                  target_id=spec.target_id, data_seed=7, dtype='float32')
    path = tmp_path/'dataset.json'
    master.dump(path,record)
    job = dict(dataset_file=str(path), data_seed=7, horizon=2)
    values, saved = master.prepare_dataset(tf,spec,job,tf.float64)
    tf.debugging.assert_equal(values,tf.cast(observations,tf.float64))
    assert saved['observation_sha256']==digest and saved['dtype']=='float32'
    assert saved['evaluation_dtype']=='float64'
    with pytest.raises(ValueError,match='target or seed'):
        master.prepare_dataset(tf,spec,dict(job,data_seed=8),tf.float64)
    with pytest.raises(ValueError,match='shape'):
        master.prepare_dataset(tf,spec,dict(job,horizon=3),tf.float64)
    master.dump(path,dict(record,observation_sha256='bad'))
    with pytest.raises(ValueError,match='checksum'):
        master.prepare_dataset(tf,spec,job,tf.float64)

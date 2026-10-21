"""CPU-only diagnostic tests of calibration/force numerical call chains."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.testing.surrogate_hmc_scope_tuning import make_context, precision_screen


def fixture():
    dtype = tf.float64
    n,h,d=32,2,1
    spec=LGSSMSpec("diagonal_ar",d)
    controls=ProposalControls(flow_steps=2,reset_steps=4)
    theta=tf.constant([[.9,.6,.8],[.88,.61,.82]],dtype)
    observations=tf.constant([[.5],[.3]],dtype)
    initial=tf.random.stateless_normal([n,d],[401,1],dtype=dtype)
    noise=tf.random.stateless_normal([h,n,d],[402,1],dtype=dtype)
    uniforms=tf.random.stateless_uniform([h,n,2],[403,1],dtype=dtype)
    beta=tf.constant([.3,.3,.4],dtype)
    return spec,controls,theta,observations,initial,noise,uniforms,beta


def test_history_is_exactly_the_pre_observation_cloud_without_changing_score():
    spec,controls,theta,obs,initial,noise,uniforms,beta=fixture()
    plain=make_batched_score_filter(spec,32,2,controls,tf.float64,True,2)
    recorded=make_batched_score_filter(spec,32,2,controls,tf.float64,True,2,record_history=True)
    args=(theta,obs,initial,noise,uniforms,beta)
    a,b=plain(*args),recorded(*args)
    assert "ancestor_history" not in a
    assert b["ancestor_history"].shape==(2,2,32,1)
    for key in a:
        if a[key].dtype.is_floating:
            tf.debugging.assert_near(a[key],b[key],atol=2e-10,rtol=2e-10)
        else:
            tf.debugging.assert_equal(a[key],b[key])
    prefix=make_batched_score_filter(spec,32,1,controls,tf.float64,True,2)(
        theta,obs[:1],initial,noise[:1],uniforms[:1],beta)
    for row in range(2):
        initialized=spec.initial_cloud(theta[row],initial,tf.zeros([3],tf.float64))[0]
        tf.debugging.assert_equal(b["ancestor_history"][row,0],initialized)
    tf.debugging.assert_near(b["ancestor_history"][:,1],prefix["particles"],atol=2e-10,rtol=2e-10)


@pytest.mark.parametrize("contraction", ["direct","matrix"])
def test_context_reaches_shared_maps_and_density_against_dense_reference(monkeypatch, contraction):
    from bayesfilter.highdim import covariance_proposal_tf as old
    from bayesfilter.highdim.analytical_jet_tf import Jet
    from bayesfilter.highdim.covariance_proposal_score_tf import Models,build_maps,draw_from_maps,observation_density
    def forbidden(*args,**kwargs):
        raise AssertionError("calibration reached old directional implementation")
    monkeypatch.setattr(old,"build_maps",forbidden)
    spec,controls,theta,obs,initial,noise,uniforms,beta=fixture()
    from dataclasses import replace
    controls=replace(controls,shared_gaussian_contraction=contraction)
    ancestors=spec.initial_cloud(theta[0],initial,tf.zeros([3],tf.float64))[0]
    actual,target,valid=make_context(spec,theta[0],controls,32)(ancestors,obs[0],noise[0],uniforms[0])
    models=Models(spec,theta[0],tf.zeros([1,3],tf.float64))
    maps=build_maps(models,Jet(ancestors,tf.zeros([1,32,1],tf.float64)),obs[0],controls)
    child=draw_from_maps(maps,tf.constant([1/3]*3,tf.float64),uniforms[0],noise[0])
    parts=[]
    for component in range(3):
        covariance=maps[1].v[component]
        factor=tf.linalg.cholesky(covariance)
        residual=child.v[:,None]-maps[0].v[component][None]
        solved=tf.linalg.cholesky_solve(factor[None],residual[...,None])[...,0]
        logs=-.5*tf.math.log(tf.constant(2.*3.141592653589793,tf.float64))-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)),axis=-1)[None]-.5*tf.reduce_sum(residual*solved,axis=-1)
        parts.append(tf.reduce_logsumexp(logs,axis=-1)-tf.math.log(tf.constant(32.,tf.float64)))
    expected=tf.stack(parts,axis=-1)
    tf.debugging.assert_equal(valid,True)
    tf.debugging.assert_near(actual,expected,atol=1e-10,rtol=1e-10)
    tf.debugging.assert_near(target,observation_density(models,child,obs[0]).v+expected[:,0],atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize("failure",["none","score","invalid","unfinished","nan"])
def test_precision_is_a_veto_even_when_values_agree(failure):
    a=dict(value=tf.constant([-100.],tf.float32),score=tf.constant([[2.,.001]],tf.float32),
           valid=tf.constant([True]),steps_completed=tf.constant([50]))
    b=dict(value=tf.constant([-100.],tf.float64),score=tf.constant([[2.,.001]],tf.float64),
           valid=tf.constant([True]),steps_completed=tf.constant([50]))
    if failure=="score": a["score"]=tf.constant([[2.1,.001]],tf.float32)
    if failure=="invalid": a["valid"]=tf.constant([False])
    if failure=="unfinished": a["steps_completed"]=tf.constant([49])
    if failure=="nan": a["score"]=tf.constant([[float("nan"),.001]],tf.float32)
    assert precision_screen(a,b,50)["passed"]==(failure=="none")

def test_trajectory_health_retains_invalid_counts_and_per_row_extrema():
    from bayesfilter.testing.surrogate_hmc_scope_tuning import trajectory_diagnostics
    from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS
    trace = tf.broadcast_to(tf.constant([[1., 3., float("nan")], [5., 2., 4.]])[:, :, None],
                            [2, 3, len(TRACE_FIELDS)])
    run = dict(trace=trace, valid=tf.constant([False, True]),
               steps_completed=tf.constant([2, 3]))
    report = trajectory_diagnostics(run)
    tf.debugging.assert_equal(report["valid"], run["valid"])
    for name in ("ess", "max_weight", "reset_covariance_relative_error", "reset_displacement"):
        tf.debugging.assert_equal(report[name]["finite_count"], [2, 3])
        tf.debugging.assert_equal(report[name]["minimum"], [1., 2.])
        tf.debugging.assert_equal(report[name]["maximum"], [3., 5.])


def test_independent_reference_uses_same_validation_inputs_and_all_rows():
    from types import SimpleNamespace
    from bayesfilter.testing.surrogate_hmc_scope_tuning import independent_references, VALIDATION_SEEDS
    calls = []
    def reference(theta, observations):
        assert theta.dtype == observations.dtype == tf.float64
        calls.append((theta.numpy().tolist(), observations.numpy().tolist()))
        return tf.reduce_sum(theta)+tf.reduce_sum(observations), tf.ones_like(theta)
    case = SimpleNamespace(name="lgssm", spec=SimpleNamespace(reference_value_and_score=reference))
    positions = tf.constant([[.5, .2], [.7, .3]], tf.float32)
    data = {seed: (tf.constant([[float(i)]], tf.float32), ()) for i, seed in enumerate(VALIDATION_SEEDS)}
    result = independent_references(case, positions, data)
    assert result["status"] == "AVAILABLE"
    assert len(calls) == len(result["rows"]) == 4
    assert [r["seed"] for r in result["rows"]] == [VALIDATION_SEEDS[0]]*2+[VALIDATION_SEEDS[1]]*2
    assert all(row["finite"] for row in result["rows"])

@pytest.mark.parametrize("failure", ["none", "flag", "nan", "fallback", "divergence"])
def test_pilot_health_checks_every_executed_trajectory(failure):
    from types import SimpleNamespace
    from bayesfilter.testing.surrogate_hmc_five_model import pilot_health
    chain = SimpleNamespace(
        finite_status=tf.constant([[failure != "flag"]]),
        positions=tf.zeros([1,1,2], tf.float64),
        potentials=tf.zeros([1,1], tf.float64),
        delta_h=tf.constant([[float("nan") if failure == "nan" else 0.]],tf.float64),
        force_fallback=tf.constant([[failure == "fallback"]]),
        divergence=tf.constant([[failure == "divergence"]]))
    assert pilot_health(chain)["veto"] == (failure != "none")

def test_scalar_density_forwards_frozen_execution_choice(monkeypatch):
    from bayesfilter.highdim import covariance_proposal_score_tf as scalar
    from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch
    from bayesfilter.highdim.analytical_jet_tf import Jet
    calls=[]
    def record(x,means,cov,shared_covariance=False,contraction="direct"):
        calls.append((x.v.shape.as_list(), shared_covariance, contraction))
        return Jet(tf.zeros([1,2],tf.float64),tf.zeros([1,1,2],tf.float64)),tf.constant([True])
    monkeypatch.setattr(batch,"gaussian_mixture",record)
    x=Jet(tf.ones([2,1],tf.float64),tf.zeros([1,2,1],tf.float64))
    cov=Jet(tf.ones([2,1,1],tf.float64),tf.zeros([1,2,1,1],tf.float64))
    result,valid=scalar.gaussian_mixture(x,x,cov,True,contraction="matrix")
    assert calls==[([1,2,1],True,"matrix")]
    assert result.v.shape==(2,)
    tf.debugging.assert_equal(valid,True)

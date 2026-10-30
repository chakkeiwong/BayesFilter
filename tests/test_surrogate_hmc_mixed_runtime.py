"""Deliberate CPU/XLA checks of the actual optional runtime call chain."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
from dataclasses import replace
import importlib.util
from pathlib import Path
import pytest
import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.testing import surrogate_hmc_scope_tuning as tuning
from bayesfilter.testing.surrogate_hmc_precision_islands_diagnostic import make_program
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls

POLICY = "fp64_recursion_fp32_pairs"

@pytest.mark.parametrize("name", ["lgssm", "ksc", "predator_prey", "sir_d18", "range_bearing"])
def test_actual_shared_factory_matches_isolated_diagnostic_and_uniform_baseline(name, monkeypatch):
    # Tiny CPU reference exception, never a campaign tuning scope.
    monkeypatch.setattr(runtime, "N", 64)
    case=runtime.configure_precision(runtime.load_case(name), POLICY)
    h=2
    controls=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=0)
    theta=tf.stack([case.theta0,case.chart(tf.fill([case.p],tf.constant(.01,tf.float64)))[0]])
    inputs=(theta,case.observations[:h],case.fixed[0],case.fixed[1][:h],
            case.fixed[2][:h],case.beta)
    actual=shared.make_batched_score_filter(case.spec,64,h,controls,tf.float64,
                                           precision_policy=POLICY)
    diagnostic,_=make_program(case.spec,64,h,controls,policy="both_fp32")
    a,b=actual(*inputs),diagnostic(*inputs)
    for key in a:
        tf.debugging.assert_equal(a[key],b[key])
    tf.debugging.assert_equal(a["valid"],[True,True])
    tf.debugging.assert_equal(a["steps_completed"],[h,h])
    assert actual.experimental_get_tracing_count()==1
    assert actual.get_concrete_function().function_def.attr["_XlaMustCompile"].b

    # Protected pre-integration source, not an alternative runtime implementation.
    source=Path("docs/plans/artifacts/surrogate-hmc-fresh-scope-advance-20261012-01/runtime-before/bayesfilter/highdim/covariance_proposal_score_batch_tf.py")
    spec=importlib.util.spec_from_file_location("_uniform_baseline_diagnostic",source)
    baseline=importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    ordinary=runtime.configure_precision(case, "uniform")
    before=baseline.make_batched_score_filter(ordinary.spec,64,h,controls,tf.float32)
    after=shared.make_batched_score_filter(ordinary.spec,64,h,controls,tf.float32)
    values=tuple(tf.cast(x,tf.float32) for x in inputs)
    old,new=before(*values),after(*values)
    # This snapshot predates the shared weighted-score and reduction reuse
    # repairs. Reassociation of FP32 sums need not be bit-identical. Keep the
    # actual-vs-isolated mixed-policy check above exact; here use a tighter
    # 1e-4 scaled arithmetic regression screen than the owner's full-T 1%.
    errors = {}
    for key in old:
        if old[key].dtype.is_floating:
            tf.debugging.assert_all_finite(new[key], key)
            errors[key] = float(tf.reduce_max(
                tf.abs(new[key]-old[key])/tf.maximum(1.,tf.abs(old[key]))))
        else:
            tf.debugging.assert_equal(old[key],new[key])
    print(name, "pre-optimization scaled differences", errors)
    assert all(value <= 1e-4 for value in errors.values()), errors

def test_mixed_policy_requires_explicit_fp64_recursion():
    from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
    with pytest.raises(ValueError,match="FP64 recurrence"):
        shared.make_batched_score_filter(LGSSMSpec("diagonal_ar",1),16,2,
            ProposalControls(),tf.float32,precision_policy=POLICY)

def test_scope_binds_precision_and_fresh_partition(monkeypatch):
    monkeypatch.setattr(runtime,"N",64)
    case=runtime.load_case("lgssm")
    mixed=runtime.configure_precision(case,POLICY,20)
    assert case.force_dtype==tf.float32 and case.precision_policy=="uniform"
    assert mixed.force_dtype==tf.float64 and mixed.scope()["pairwise_dtype"]=="float32"
    assert case.scope()!=mixed.scope()
    assert mixed.scope()!=runtime.configure_precision(case,POLICY,40).scope()

def test_actual_hmc_consumer_forwards_mixed_policy(monkeypatch):
    monkeypatch.setattr(runtime,"N",64)
    case=runtime.configure_precision(runtime.load_case("lgssm"),POLICY,20)
    called=[]
    def factory(spec,n,h,controls,dtype,**kwargs):
        called.append((dtype,kwargs["precision_policy"]))
        assert dtype==tf.float64 and kwargs["precision_policy"]==POLICY
        def program(theta,obs,initial,innovations,uniforms,beta):
            assert all(x.dtype==tf.float64 for x in (theta,obs,initial,innovations,uniforms,beta))
            return dict(score=tf.ones_like(theta),valid=tf.ones([2],tf.bool),
                        steps_completed=tf.fill([2],h))
        return program
    monkeypatch.setattr(runtime,"make_batched_score_filter",factory)
    monkeypatch.setattr(runtime,"reference",lambda _: (lambda theta:(-tf.reduce_sum(theta*theta),tf.constant(True)),"test"))
    components=runtime.bindings(case)
    q=tf.zeros([4,case.p],tf.float64)
    result=components["force"].function(q)
    _,jac=case.chart(q)
    tf.debugging.assert_equal(result.force,q-tf.broadcast_to(jac,q.shape))
    tf.debugging.assert_equal(result.force_valid,[True]*4)
    assert called==[(tf.float64,POLICY)]

def test_tuner_uses_candidate_and_separate_uniform_reference(monkeypatch,tmp_path):
    monkeypatch.setattr(runtime,"N",64)
    case=runtime.configure_precision(runtime.load_case("lgssm"),POLICY,20)
    calls=[]
    def factory(spec,n,h,controls,dtype,**kwargs):
        calls.append((kwargs.get("record_history",False),dtype,kwargs.get("precision_policy","uniform")))
        def evaluate(theta,obs,initial,innovations,uniforms,beta):
            b=theta.shape[0]
            return dict(value=tf.zeros([b],dtype),score=tf.zeros([b,case.p],dtype),
                valid=tf.ones([b],tf.bool),steps_completed=tf.fill([b],h),
                ancestor_history=tf.zeros([b,h,n,spec.dimension],dtype),
                trace=tf.ones([b,h,len(tuning.TRACE_FIELDS)],dtype))
        return evaluate
    monkeypatch.setattr(tuning,"make_batched_score_filter",factory)
    context_calls=[]
    def context(spec,theta,controls,n,precision_policy="uniform"):
        context_calls.append(precision_policy)
        return lambda *args:(tf.zeros([n,3],tf.float64),tf.zeros([n],tf.float64),tf.constant(True))
    monkeypatch.setattr(tuning,"make_context",context)
    monkeypatch.setattr(tuning,"independent_references",lambda *args:dict(status="NOT_AVAILABLE"))
    result=tuning.tune(case,tmp_path)
    assert result["selected"] is not None
    assert result["calibration_seeds"]==(26101231,26101232)
    assert result["validation_seeds"]==(26101233,26101234)
    assert calls==[(True,tf.float64,POLICY),(False,tf.float64,POLICY),
                   (False,tf.float64,"uniform")]*3
    assert context_calls==[POLICY]*3
    for row in result["candidates"]:
        for check in row["precision_checks"]:
            assert check["reference_precision_policy"]=="uniform"
            assert check["candidate_precision_policy"]==POLICY
    record=runtime.load_selection(case,tmp_path/"filter-selection.json")
    assert case.beta.dtype==tf.float64
    with pytest.raises(ValueError,match="scope"):
        runtime.load_selection(runtime.configure_precision(case,"uniform",20),
                               tmp_path/"filter-selection.json")


def test_mixed_calibration_context_executes_same_density_boundary(monkeypatch):
    monkeypatch.setattr(runtime,"N",64)
    case=runtime.configure_precision(runtime.load_case("lgssm"),POLICY)
    control=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=0)
    calls=[]
    original=tuning.precision_gaussian_mixture
    def checked(*args,**kwargs):
        calls.append(kwargs["precision_policy"])
        return original(*args,**kwargs)
    monkeypatch.setattr(tuning,"precision_gaussian_mixture",checked)
    mixed=tuning.make_context(case.spec,case.theta0,control,64,POLICY)
    reference=tuning.make_context(case.spec,case.theta0,control,64)
    inputs=(case.fixed[0],case.observations[0],case.fixed[1][0],case.fixed[2][0])
    a,la,valid=mixed(*inputs)
    b,lb,reference_valid=reference(*inputs)
    tf.debugging.assert_equal(valid,True)
    tf.debugging.assert_equal(reference_valid,True)
    tf.debugging.assert_near(a,b,rtol=1e-4,atol=1e-4)
    tf.debugging.assert_near(la,lb,rtol=1e-4,atol=1e-4)
    assert calls==[POLICY]*3
    assert mixed.experimental_get_tracing_count()==1

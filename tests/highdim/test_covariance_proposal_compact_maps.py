"""Independent full-broadcast authority for compact covariance construction."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP","0")
from dataclasses import replace
import pytest
import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_score_tf as score
from bayesfilter.highdim.analytical_jet_tf import Jet
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.testing import surrogate_hmc_covariance_cost_diagnostic as diagnostic


@pytest.mark.parametrize("name",["lgssm","ksc","predator_prey","sir_d18","range_bearing"])
@pytest.mark.parametrize("dtype",[tf.float32,tf.float64])
def test_compact_covariances_equal_full_broadcast_product_rule(name,dtype,monkeypatch):
    monkeypatch.setattr(runtime,"N",32)
    case=runtime.load_case(name)
    if dtype==tf.float64:
        case=runtime.configure_precision(case,"fp64_recursion_fp32_pairs")
    theta=tf.cast(case.theta0,dtype)
    directions=tf.eye(case.p,dtype=dtype)
    initial=[case.spec.initial_cloud(theta,tf.cast(case.fixed[0],dtype),directions[k])
             for k in range(case.p)]
    x=Jet(initial[0][0],tf.stack([row[2] for row in initial]))
    controls=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=0)
    sizes=[]
    checked=score.checked_chol
    def tracked(matrix):
        sizes.append(tuple(matrix.v.shape))
        return checked(matrix)
    monkeypatch.setattr(score,"checked_chol",tracked)
    @tf.function(input_signature=[tf.TensorSpec(theta.shape,dtype)],jit_compile=True,autograph=False)
    def compare(theta):
        models=score.Models(case.spec,theta,directions)
        maps=score.build_maps(models,x,tf.cast(case.observations[0],dtype),controls)
        cov,a=maps[1],maps[2]
        q=models.covariance("process_covariance","process_covariance_tangent_fn")
        av,at,qv,qt=a.v,a.t,q.v,q.t[:,None,None]
        # Independent expansion of every product-rule term, after broadcasting.
        mm=tf.linalg.matmul
        full=mm(mm(av,qv),av,transpose_b=True)
        tangent=(mm(mm(at,qv),av,transpose_b=True)
                 +mm(mm(av,qt),av,transpose_b=True)
                 +mm(mm(av,qv),at,transpose_b=True))
        _,valid,margin=checked(Jet(full,tangent))
        return cov.v,cov.t,full,tangent,maps[-1],margin,valid
    av,at,bv,bt,am,bm,valid=compare(theta)
    tol=3e-6 if dtype==tf.float32 else 5e-12
    tf.debugging.assert_near(av,bv,atol=tol,rtol=tol)
    tf.debugging.assert_near(at,bt,atol=tol,rtol=tol)
    tf.debugging.assert_near(am,bm,atol=tol,rtol=tol)
    tf.debugging.assert_equal(valid,True)
    models=score.Models(case.spec,theta,directions)
    d=case.spec.dimension
    assert sizes[-3:]==[(1,d,d),(1,d,d),(1 if models.constant_observation_jacobian else 32,d,d)]
    assert compare.experimental_get_tracing_count()==1


def test_paired_diagnostic_traces_protected_baseline(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,"N",32)
    case=runtime.load_case("lgssm")
    case.observations=case.observations[:2]
    case.fixed=(case.fixed[0],case.fixed[1][:2],case.fixed[2][:2])
    case.controls=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=0)
    from pathlib import Path
    baseline=tmp_path/"protected.py"
    source=Path(score.__file__).read_text()
    # A deliberate baseline-only shift proves the traced consumer uses that
    # source and not the live module after make_pair restores its global.
    source=source.replace("means=mv(aa,m[None])+cc","means=mv(aa,m[None])+cc+.001")
    baseline.write_text(source)
    before=diagnostic.batch.build_maps
    old,new=diagnostic.make_pair(case,baseline)
    assert diagnostic.batch.build_maps is before
    theta=tf.stack([case.theta0,case.theta0])
    args=(theta,case.observations,*case.fixed,case.beta)
    a,b=old(*args),new(*args)
    assert float(tf.reduce_max(tf.abs(a["value"]-b["value"])))>1e-6
    assert old.experimental_get_tracing_count()==new.experimental_get_tracing_count()==1
    assert diagnostic.compare_outputs(b,b,2)["passed"]
    broken=dict(b,score=b["score"]+1000.)
    assert not diagnostic.compare_outputs(broken,b,2)["passed"]
    broken=dict(b,steps_completed=tf.constant([2,1]))
    assert not diagnostic.compare_outputs(broken,b,2)["passed"]


def test_comparison_artifact_saves_provenance_and_parity(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,"N",32)
    case=runtime.load_case("lgssm")
    case.observations=case.observations[:2]
    case.fixed=(case.fixed[0],case.fixed[1][:2],case.fixed[2][:2])
    case.controls=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=0)
    result=diagnostic.run_diagnostic(case,tmp_path,score.__file__,repetitions=1)
    assert result["status"]=="COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED"
    assert result["comparison"]["passed"]
    assert result["trace_counts"]==[1,1]
    assert result["baseline_source_sha256"]==result["candidate_source_sha256"]
    assert result["cpu_only"]
    assert len(result["input_sha256"])==6
    assert not result["posterior_sampling_complete"]


def test_initial_region_distinguishes_existing_and_new_invalidity():
    baseline = dict(valid=tf.constant([True,False]),steps_completed=tf.constant([2,1]),
                    value=tf.constant([1.,0.]),score=tf.zeros([2,3]),
                    particles=tf.zeros([2,4,1]),particle_tangent=tf.zeros([2,4,1,3]))
    same = diagnostic.compare_initial_region(baseline,baseline,2)
    assert same["passed"] and not same["hmc_health_passed"]
    assert same["newly_invalid"] == [False,False]
    invalid = dict(baseline,valid=tf.constant([False,False]))
    assert not diagnostic.compare_initial_region(invalid,baseline,2)["passed"]
    nonfinite = dict(baseline,score=tf.constant([[float("nan"),0.,0.],[0.,0.,0.]]))
    assert not diagnostic.compare_initial_region(nonfinite,baseline,2)["passed"]

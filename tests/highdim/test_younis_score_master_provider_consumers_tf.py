"""Actual master consumers, independent fit streams and provider scope checks."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
from bayesfilter.score_study.adapters import evaluate_gaussian
from bayesfilter.score_study.registry import default_registry
from bayesfilter.score_study.tuning import candidate_configuration, scope


def fixture(proposal):
    from bayesfilter.score_study.runtime import configure_runtime
    configure_runtime(device="CPU",tf32=False,jit_compile=True)
    from tests.highdim.test_younis_score_master_canonical_tf import CONTROLS
    settings=dict(dimension=1,observation_dimension=1,horizon=2,particles=16,
        dtype="float64",device="CPU",tf32=False,jit_compile=True,
        theta=[.62,-.8,-.6,.9,.25,-.3],data_theta=[.62,-.8,-.6,.9,.25,-.3])
    row=dict(id="provider",model="gaussian_all_parameters",proposal=proposal,
        estimator="analytical_filter",comparison_target="model_score",comparison="approximation_error",
        dataset=100,replicate=0,role="mechanics",controls=CONTROLS.copy(),within_fraction=.5,
        fit_iterations=2,fit_initial_variance=4.,fit_floor_ratio=.01,fit_theta=settings["theta"])
    context=dict(study=dict(seed=841,settings=settings,evidence_class="mechanics"),registry=default_registry())
    return row,context


def _check_fitted_consumer():
    row,context=fixture("fitted_twist")
    result=evaluate_gaussian(row,context)
    other=evaluate_gaussian({**row,"replicate":1},context)
    a,b=result["diagnostics"],other["diagnostics"]
    assert a["fit"]==b["fit"]
    assert a["fit_digest"]==b["fit_digest"]
    fit_seeds={tuple(v) for k,v in a["fit_seed_records"].items() if k.startswith("fit")}
    final_seeds={tuple(v) for k,v in a["fit_seed_records"].items() if k.startswith("final")}
    assert not fit_seeds & final_seeds
    assert a["fit_seed_records"]["final_twist_initial"]!=b["fit_seed_records"]["final_twist_initial"]
    assert result["value"]!=other["value"]
    assert a["candidate_configuration"]==candidate_configuration(row)
    assert result["runtime"]["traces"]==1


def _check_mixture_consumer():
    row,context=fixture("kdm_covariance")
    import tensorflow as tf
    result=evaluate_gaussian(row,context)
    diag=result["diagnostics"]
    assert diag["candidate_configuration"]==candidate_configuration(row)
    weights=tf.exp(tf.constant(diag["final_component_log_weights"],tf.float64))
    tf.debugging.assert_near(tf.reduce_sum(weights),tf.constant(1.,tf.float64))
    assert abs(float(weights[0]-weights[1]))>1e-6
    assert diag["final_provider_covariance"][0][0]>0
    assert len(result["score"])==6
    changed=deepcopy(row);changed["within_fraction"]=.8
    assert candidate_configuration(changed)!=candidate_configuration(row)
    changed_study=deepcopy(context["study"]);changed_study["settings"]["horizon"]=3
    assert scope(changed_study,row)!=scope(context["study"],row)


def _isolated_check(name):
    result=subprocess.run([sys.executable,"-c",
        "import runpy,sys; runpy.run_path(sys.argv[1])[sys.argv[2]]()",str(Path(__file__).resolve()),name],
        cwd=Path(__file__).resolve().parents[2],env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1",
        "BAYESFILTER_PRELOAD_CUSTOM_OP":"0"},capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr


def test_fitted_consumer_freezes_common_offline_fit_before_independent_final_replicates():
    _isolated_check("_check_fitted_consumer")


def test_mixture_master_consumer_emits_conditioned_persistent_covariance_and_full_scope():
    _isolated_check("_check_mixture_consumer")

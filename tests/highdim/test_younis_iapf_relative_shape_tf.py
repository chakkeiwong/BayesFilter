"""CPU/XLA diagnostic references for the explicit Algorithm-3 adaptation."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tensorflow as tf

from bayesfilter.score_study.iapf_fit_tf import (
    _density_profile, bounded_density_fit, make_density_recursive_fit_kernel)


def test_shape_quotient_and_full_analytical_gradient_against_independent_fd():
    z=tf.constant([[-1.2,.2],[.5,-.8],[1.4,.9],[.1,.5]],tf.float64)
    target=tf.constant([.3,1.,.6,.2],tf.float64)
    par=tf.constant([.2,-.1,.1,-.2],tf.float64)
    def reference(p):
        density=tf.exp(-tf.reduce_sum(p[2:])-.5*tf.reduce_sum(((z-p[:2])/tf.exp(p[2:]))**2,1))
        scale=tf.reduce_sum(density*target)/tf.reduce_sum(target**2)
        return tf.reduce_sum((density-scale*target)**2)/tf.reduce_sum(density**2)
    loss,gradient,*_=_density_profile(z,tf.math.log(target),par,"relative_shape")
    tf.debugging.assert_near(loss,reference(par),atol=1e-14)
    for direction in tf.unstack(tf.eye(4,dtype=tf.float64)):
        h=tf.constant(1e-5,tf.float64)
        fd=(reference(par+h*direction)-reference(par-h*direction))/(2*h)
        tf.debugging.assert_near(fd,tf.reduce_sum(gradient*direction),atol=1e-8,rtol=1e-7)
    shifted=_density_profile(z,tf.math.log(target)+80.,par,"relative_shape")
    tf.debugging.assert_near(loss,shifted[0],atol=1e-13)
    tf.debugging.assert_near(gradient,shifted[1],atol=1e-13)


def test_shape_objective_remains_visible_when_density_amplitude_underflows():
    args=(tf.constant([[-1.],[0.],[1.]],tf.float32),tf.zeros([3],tf.float32),
          tf.constant([4.,math.log(.2)],tf.float32))
    assert float(_density_profile(*args)[0])==0.
    value,gradient,*_=_density_profile(*args,"relative_shape")
    tf.debugging.assert_near(value,tf.constant(2/3,tf.float32),atol=2e-7)
    tf.debugging.assert_all_finite(gradient,"relative gradient")


def test_shape_fit_recovers_gaussian_and_preserves_target_amplitude_invariance():
    x=tf.linspace(tf.constant(-3.,tf.float64),tf.constant(3.,tf.float64),51)[:,None]
    target=-.5*(x[:,0]-.4)**2/.7
    controls=dict(mean_bound=4.,sd_lower=.2,sd_upper=4.,max_steps=2000,
        max_backtracks=30,tolerance=1e-8,floor_ratio=.01,objective="relative_shape")
    center,cov,floor,info=bounded_density_fit(x,target,**controls)
    assert bool(info["valid"]) and bool(info["converged"]) and not bool(info["boundary_active"])
    tf.debugging.assert_near(center,tf.constant([.4],tf.float64),atol=2e-5)
    tf.debugging.assert_near(cov,tf.constant([[.7]],tf.float64),atol=2e-5)
    other=bounded_density_fit(x,target+70.,**controls)
    for a,b in zip((center,cov,floor),other[:3]):
        tf.debugging.assert_near(a,b,atol=1e-9)
    tf.debugging.assert_near(info["optimization_loss"],info["normalized_shape_residual"],atol=1e-14)


def test_saved_underflow_cloud_uses_valid_shape_fit_with_single_xla_trace():
    saved=json.loads((Path(__file__).resolve().parents[1]/"fixtures/iapf_density_underflow.json").read_text())
    kernel=make_density_recursive_fit_kernel(1,1,16,2,4.,.2,4.,2000,30,1e-7,.01,
        "float64",objective="relative_shape")
    out=kernel(*(tf.constant(saved[key],tf.float64) for key in ("theta","observations","clouds")))
    assert bool(out[3]) and bool(out[4])
    tf.debugging.assert_all_finite(out[5],"shape diagnostics")
    tf.debugging.assert_equal(out[5][:,9],tf.zeros([2],tf.float64))
    tf.debugging.assert_near(out[5][:,10],out[5][:,1],atol=1e-14)
    assert kernel.experimental_get_tracing_count()==1


def _check_consumer():
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    from bayesfilter.score_study.adapters import evaluate_gaussian
    from bayesfilter.score_study.iapf_scope import validate_result_accounting
    row,context=fixture("iapf")
    settings=context["study"]["settings"]
    row["iapf"]=dict(k=1,tau=100.,max_iterations=4,max_particles=128,mean_bound=4.,
        sd_lower=.2,sd_upper=4.,max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,
        floor_ratio=.01,fit_theta=settings["theta"],fit_objective="relative_shape")
    result=evaluate_gaussian(row,context)
    diag=result["diagnostics"]
    assert diag["fit_objective"]=="relative_shape"
    assert diag["fit_method"].endswith("algorithm3_adaptation")
    validate_result_accounting(result,row,settings)
    for step in [s for rec in diag["fit_iterations"] for s in rec.get("density_fit_diagnostics",[])]:
        assert step[10]==step[1]
    wrong=deepcopy(result)
    wrong["diagnostics"]["fit_objective"]="density_l2"
    with pytest.raises(ValueError,match="objective differs"):
        validate_result_accounting(wrong,row,settings)
    other=evaluate_gaussian({**row,"replicate":1},context)
    assert other["diagnostics"]["fit_digest"]==diag["fit_digest"]
    assert other["diagnostics"]["fit_seed_records"]["iapf_final_process"]!=diag["fit_seed_records"]["iapf_final_process"]


def test_real_consumer_selects_shape_objective_and_scope_rejects_mislabeling():
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1"}
    process=subprocess.run([sys.executable,"-c",
        "from tests.highdim.test_younis_iapf_relative_shape_tf import _check_consumer; _check_consumer()"],
        env=env,capture_output=True,text=True,timeout=180)
    assert process.returncode==0,process.stdout+process.stderr

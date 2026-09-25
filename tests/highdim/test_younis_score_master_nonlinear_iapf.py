"""Independent nonlinear law/derivative checks and actual master call chains.

CPU-only numerical references; these checks do not rank stochastic methods.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tensorflow as tf

from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel, normalizer
from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit, make_density_recursive_fit_kernel


def _density(z, mean, variance):
    return tf.exp(-.5*(z-mean)**2/variance)/tf.sqrt(tf.constant(2*math.pi,z.dtype)*variance)


def test_nonlinear_normalizer_and_mixture_density_against_quadrature():
    dtype=tf.float64
    x=tf.constant([[-1.2],[.3],[1.4]],dtype)
    mean=.62*x+tf.cast(.35,dtype)*tf.sin(x)
    Q=tf.constant([[.45]],dtype);V=tf.constant([[.7]],dtype)
    center=tf.constant([.4],dtype);floor=tf.constant(-3.,dtype)
    logM,_,p=normalizer(mean,tf.zeros([6,3,1],dtype),Q,tf.zeros([6,1,1],dtype),center,V,floor)
    z=tf.linspace(tf.constant(-12.,dtype),tf.constant(12.,dtype),8001)[:,None]
    f=_density(z,tf.transpose(mean),Q[0,0])
    psi=_density(z,center[0],V[0,0])+tf.exp(floor)
    numeric=tf.reduce_sum((f*psi)[:-1]+(f*psi)[1:],axis=0)*(z[1,0]-z[0,0])/2
    tf.debugging.assert_near(tf.exp(logM),numeric,atol=2e-12,rtol=2e-12)
    gain=Q[0,0]/(Q[0,0]+V[0,0])
    adapted=tf.transpose(mean)+gain*(center[0]-tf.transpose(mean))
    q=p[None,:]*_density(z,adapted,Q[0,0]*(1-gain))+(1-p[None,:])*f
    tf.debugging.assert_near(q,f*psi/tf.exp(logM)[None,:],atol=2e-12,rtol=2e-12)
    tf.debugging.assert_near(tf.reduce_sum((q[:-1]+q[1:])/2,axis=0)*(z[1,0]-z[0,0]),tf.ones([3],dtype),atol=2e-12)


@pytest.mark.parametrize("c,b",[(.12,.04),(.35,.12)])
def test_nonlinear_frozen_twist_full_score_matches_fixed_stream_fd(c,b):
    dtype=tf.float64;N=16;T=2
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype)
    args=(tf.constant([[.5],[-.3]],dtype),
          tf.random.stateless_normal([N,1],[901,1],dtype=dtype),
          tf.random.stateless_normal([T,N,1],[901,2],dtype=dtype),
          tf.random.stateless_uniform([T+1,N],[901,3],dtype=dtype),
          tf.random.stateless_uniform([T,N],[901,4],dtype=dtype),
          tf.constant([[.4],[-.1]],dtype),tf.constant([[[.6]],[[.8]]],dtype),
          tf.constant([-3.,-3.5],dtype))
    kernel=make_fitted_twist_kernel(1,1,N,T,transition_curve=c,observation_curve=b)
    value,score,_=kernel(theta,*args)
    tf.debugging.assert_all_finite(value,"value")
    for direction in tf.unstack(tf.eye(6,dtype=dtype)):
        for step in (1e-5,3e-6):
            h=tf.constant(step,dtype)
            fd=(kernel(theta+h*direction,*args)[0]-kernel(theta-h*direction,*args)[0])/(2*h)
            tf.debugging.assert_near(fd,tf.reduce_sum(score*direction),atol=2e-7,rtol=3e-5)
    assert kernel.experimental_get_tracing_count()==1


def test_both_nonlinear_recursive_fit_targets_include_observation_and_future():
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    from bayesfilter.score_study.fitted_twist_tf import make_recursive_fit_kernel,fit_log_quadratic
    dtype=tf.float64;N=31;T=2;c=.12;b=.04
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype);obs=tf.constant([[.5],[-.3]],dtype)
    x=tf.linspace(tf.constant(-1.5,dtype),tf.constant(1.5,dtype),N)[:,None]
    clouds=tf.stack([x,x+.1])
    A,_,H,_,_,_,_,_,Q,_,R,_=parameterized_model(theta,1,1)
    fit_args=dict(mean_bound=4.,sd_lower=.2,sd_upper=4.,max_steps=2000,max_backtracks=30,tolerance=1e-7,floor_ratio=.01)
    density=make_density_recursive_fit_kernel(1,1,N,T,4.,.2,4.,2000,30,1e-7,.01,transition_curve=c,observation_curve=b)
    quadratic=make_recursive_fit_kernel(1,1,N,T,.01,transition_curve=c,observation_curve=b)
    for kind,actual in (("density",density(theta,obs,clouds)),("quadratic",quadratic(theta,obs,clouds))):
        assert bool(actual[3])
        if kind=="density":assert bool(actual[4])
        expected=[]
        for t in (1,0):
            pts=clouds[t][:,0]
            observed=H[0,0]*pts+tf.cast(b,dtype)*pts*pts
            logtarget=tf.math.log(_density(obs[t,0],observed,R[0,0]))
            if t==0:
                mean=A[0,0]*pts+tf.cast(c,dtype)*tf.sin(pts)
                logtarget+=tf.math.log(_density(mean,center[0],Q[0,0]+covariance[0,0])+tf.exp(floor))
            if kind=="density":
                center,covariance,floor,info=bounded_density_fit(clouds[t],logtarget,**fit_args)
                assert bool(info["valid"]) and bool(info["converged"])
            else:
                (center,covariance,floor),(valid,_,_)=fit_log_quadratic(clouds[t],logtarget,.01)
                assert bool(valid)
            expected.insert(0,(center,covariance,floor))
        for i in range(3):tf.debugging.assert_near(actual[i],tf.stack([e[i] for e in expected]),atol=2e-9,rtol=2e-9)


@pytest.mark.parametrize("c,b",[(0.,0.),(.35,.12)])
def test_constant_twist_recovers_independent_physical_bootstrap(c,b):
    from bayesfilter.score_study.nonlinear_tf import make_particle_filter
    dtype=tf.float64;N=16;T=2
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype);obs=tf.constant([[.4],[-.2]],dtype)
    initial=tf.random.stateless_normal([N,1],[927,1],dtype=dtype)
    noise=tf.random.stateless_normal([T,N,1],[927,2],dtype=dtype)
    uniforms=tf.random.stateless_uniform([T,N],[927,3],dtype=dtype)
    initial_uniform=((tf.cast(tf.range(N),dtype)+.5)/N)[None,:]
    kernel=make_fitted_twist_kernel(1,1,N,T,constant_twist=True,transition_curve=c,observation_curve=b)
    result=kernel(theta,obs,initial,noise,tf.concat([initial_uniform,uniforms],0),uniforms,
        tf.zeros([T,1],dtype),tf.ones([T,1,1],dtype),tf.zeros([T],dtype))
    reference=make_particle_filter(N,T,c,b,False,True)(theta,obs,initial,noise,uniforms)
    for a,z in zip(result[:2],reference[:2]):tf.debugging.assert_near(a,z,atol=2e-12,rtol=2e-12)


def _consumer(root):
    from unittest.mock import patch
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    from bayesfilter.score_study import fitted_twist_tf,iapf_fit_tf
    from bayesfilter.score_study.nonlinear_adapter import evaluate_nonlinear
    from bayesfilter.score_study.coordinator import execute,report
    from bayesfilter.score_study.tuning import issue_selection,consume_selection
    from bayesfilter.score_study.iapf_scope import validate_result_accounting
    row,context=fixture("iapf")
    row.update(model="nonlinear_scalar",estimator="nonlinear_analytical")
    settings=context["study"]["settings"]
    settings.update(dtype="float32",transition_curve=.12,observation_curve=.04,
        reference=dict(points=401,radius=9.,relative_tolerance=1e-7,tail_tolerance=1e-9))
    config=dict(k=1,tau=100.,max_iterations=4,max_particles=128,mean_bound=4.,sd_lower=.2,sd_upper=4.,
        max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,floor_ratio=.01,fit_dtype="float64",fit_theta=settings["theta"])
    row["iapf"]=config
    study={**context["study"],"schema":"younis_score_study_v1","phase":"0E","version":1,
        "plan":"docs/plans/younis-score-nonlinear-iapf-2026-09-16.md",
        "budget":{"wall_seconds":400,"max_attempts":2,"max_attempts_per_row":1},
        "partitions":{"calibration":[900],"validation":[910],"claim":[920]},
        "tuning_candidate_family":[{"iapf":config}],
        "rows":[{**row,"id":role,"role":role,"dataset":dataset} for role,dataset in (("calibration",900),("validation",910))]}
    execute(study,context["registry"],root/"tuning")
    summary=report(root/"tuning",context["registry"])
    assert summary["execution_status"]=="complete",summary
    selection_path=root/"selection.json";issue_selection(root/"tuning",selection_path)
    claim={**row,"id":"claim","role":"claim","dataset":920,"tuning_selection":str(selection_path)}
    context["study"]={**study,"rows":[claim]}
    with patch.object(fitted_twist_tf,"make_fitted_twist_kernel",wraps=fitted_twist_tf.make_fitted_twist_kernel) as shared, \
         patch.object(iapf_fit_tf,"make_density_recursive_fit_kernel",wraps=iapf_fit_tf.make_density_recursive_fit_kernel) as fitter:
        result=evaluate_nonlinear(claim,context)
        assert shared.call_count>=4 and fitter.call_count>=2
        for call in shared.call_args_list+fitter.call_args_list:
            assert call.kwargs["transition_curve"]==.12 and call.kwargs["observation_curve"]==.04
    diag=result["diagnostics"]
    assert diag["fit_observation_digest"]==diag["executed_observation_digest"]!=diag["data_version"]
    other=evaluate_nonlinear({**claim,"replicate":1},context)
    assert other["diagnostics"]["fit_digest"]==diag["fit_digest"]
    assert other["diagnostics"]["fit_seed_records"]["iapf_final_process"]!=diag["fit_seed_records"]["iapf_final_process"]
    for key in ("transition_curve","observation_curve"):
        wrong=deepcopy(context);wrong["study"]["settings"][key]+=.01
        with pytest.raises(ValueError,match="scope mismatch"):consume_selection(claim,wrong)
    for field in ("fit_model","fit_observation_digest","executed_observation_digest","physical_observations"):
        wrong=deepcopy(result)
        if field=="fit_model":wrong["diagnostics"][field]["transition_curve"]+=.01
        elif field=="physical_observations":wrong["diagnostics"][field][0][0]+=.01
        else:wrong["diagnostics"][field]="invalid"
        with pytest.raises(ValueError):validate_result_accounting(wrong,claim,settings)
    with pytest.raises(ValueError,match="requires"):
        evaluate_nonlinear({**claim,"tuning_selection":None},context)
    # The nonlinear comparator also consumes the shared sampler and log fitter.
    with patch.object(fitted_twist_tf,"make_fitted_twist_kernel",wraps=fitted_twist_tf.make_fitted_twist_kernel) as shared, \
         patch.object(fitted_twist_tf,"make_recursive_fit_kernel",wraps=fitted_twist_tf.make_recursive_fit_kernel) as fitter:
        fitted=evaluate_nonlinear({**row,"proposal":"fitted_twist","dataset":900,"role":"mechanics"},context)
        assert shared.call_count==fitter.call_count==1
        assert shared.call_args.kwargs==fitter.call_args.kwargs==dict(transition_curve=.12,observation_curve=.04)
        assert fitted["diagnostics"]["fit_model"]==diag["fit_model"]


def test_actual_nonlinear_iapf_selection_data_cast_and_shared_call_chain(tmp_path):
    code="from pathlib import Path; from tests.highdim.test_younis_score_master_nonlinear_iapf import _consumer; _consumer(Path("+repr(str(tmp_path))+"))"
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1","BAYESFILTER_PRELOAD_CUSTOM_OP":"0",
        "TF_NUM_INTRAOP_THREADS":"1","TF_NUM_INTEROP_THREADS":"1","OMP_NUM_THREADS":"1"}
    done=subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).resolve().parents[2],
        env=env,capture_output=True,text=True,timeout=400)
    assert done.returncode==0,done.stdout+done.stderr

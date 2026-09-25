"""Independent objective/derivative fixtures and the real iAPF call chain."""
import math
import json
from pathlib import Path
import os
import subprocess
import sys
import pytest
import tensorflow as tf
from bayesfilter.score_study.iapf_fit_tf import profile_density_loss,bounded_density_fit,make_density_recursive_fit_kernel
from bayesfilter.score_study.iapf_adapter import iteration_decision


FIT=dict(mean_bound=4.,sd_lower=.2,sd_upper=4.,max_steps=2000,
         max_backtracks=30,tolerance=1e-8,floor_ratio=.01)


def test_profile_objective_is_density_loss_and_analytical_gradient_matches_fd():
    dtype=tf.float64
    z=tf.constant([[-1.2,.2],[.5,-.8],[1.4,.9],[.1,.5]],dtype)
    targets=tf.constant([.3,1.,.6,.2],dtype)
    par=tf.constant([.2,-.1,.1,-.2],dtype)
    loss,gradient,scale,_=profile_density_loss(z,targets,par)
    density=tf.exp(-tf.reduce_sum(par[2:])-.5*tf.reduce_sum(((z-par[:2])/tf.exp(par[2:]))**2,1))/(2*math.pi)
    lam=tf.reduce_sum(density*targets)/tf.reduce_sum(targets**2)
    tf.debugging.assert_near(loss/(2*math.pi)**2,tf.reduce_mean((density-lam*targets)**2),atol=1e-14)
    tf.debugging.assert_near(scale/(2*math.pi),lam,atol=1e-14)
    for direction in tf.unstack(tf.eye(4,dtype=dtype)):
        h=tf.constant(1e-5,dtype)
        fd=(profile_density_loss(z,targets,par+h*direction)[0]-profile_density_loss(z,targets,par-h*direction)[0])/(2*h)
        tf.debugging.assert_near(fd,tf.reduce_sum(gradient*direction),atol=1e-8,rtol=1e-7)


def test_density_fit_recovers_interior_gaussian_and_reports_scale_invariance():
    dtype=tf.float64;x=tf.reshape(tf.linspace(tf.constant(-3.,dtype),tf.constant(3.,dtype),51),[-1,1])
    target=-.5*(x[:,0]-.4)**2/.7+2.
    center,V,_,info=bounded_density_fit(x,target,**FIT)
    assert bool(info['valid']) and bool(info['converged']) and not bool(info['boundary_active'])
    tf.debugging.assert_near(center,tf.constant([.4],dtype),atol=2e-5)
    tf.debugging.assert_near(V,tf.constant([[.7]],dtype),atol=2e-5)
    c2,V2,_,info2=bounded_density_fit(x,target+7.,**FIT)
    tf.debugging.assert_near(center,c2,atol=1e-9);tf.debugging.assert_near(V,V2,atol=1e-9)
    tf.debugging.assert_near(info2['log_lambda'],info['log_lambda']-7.,atol=1e-8)


def test_flat_target_reports_bound_and_nonzero_shape_error():
    x=tf.constant([[-1.],[0.],[1.]],tf.float64)
    _,_,_,info=bounded_density_fit(x,tf.zeros([3],tf.float64),**{**FIT,'sd_upper':2.})
    assert bool(info['valid']) and bool(info['converged']) and bool(info['boundary_active'])
    assert float(info['normalized_shape_residual'])>1e-5


def test_invalid_cloud_and_unconverged_fit_cannot_pass():
    x=tf.zeros([5,1],tf.float64)
    *_,info=bounded_density_fit(x,tf.zeros([5],tf.float64),**FIT)
    assert not bool(info['valid'])
    x=tf.constant([[-1.],[0.],[1.],[2.]],tf.float64)
    *_,info=bounded_density_fit(x,-(x[:,0]-.8)**2,**{**FIT,'max_steps':1})
    assert not bool(info['converged'])


def test_density_underflow_preserves_log_scale_and_relative_shape():
    from bayesfilter.score_study.iapf_fit_tf import _density_profile
    outputs=[]
    for dtype in (tf.float64,tf.float32):
        outputs.append(_density_profile(tf.constant([[-1.],[0.],[1.]],dtype),
            tf.zeros([3],dtype),tf.constant([4.,math.log(.2)],dtype)))
    reference,actual=outputs
    assert float(actual[0])==0. and float(reference[0])>0.
    tf.debugging.assert_near(actual[2],tf.cast(reference[2],tf.float32),atol=2e-5)
    tf.debugging.assert_near(actual[3],tf.constant(2/3,tf.float32),atol=2e-7)
    assert bool(tf.reduce_all(tf.math.is_finite(tf.stack([actual[2],actual[3],actual[4]]))))


def test_saved_failed_cloud_is_finite_in_fp32_and_underflow_is_explicit():
    saved=json.loads((Path(__file__).resolve().parents[1]/'fixtures/iapf_density_underflow.json').read_text())
    kernel=make_density_recursive_fit_kernel(1,1,16,2,4.,.2,4.,2000,30,1e-7,.01,'float32')
    out=kernel(*(tf.constant(saved[key],tf.float32) for key in ('theta','observations','clouds')))
    assert not bool(out[3]) and bool(out[4])
    tf.debugging.assert_all_finite(out[5],'underflow diagnostics must remain finite')
    assert float(out[5][0,1])>.3 and float(out[5][0,3])==1.
    assert float(out[5][0,9])==1.


def test_algorithm4_windows_growth_strict_stop_and_capacity_veto():
    kwargs=dict(k=2,tau=.01,max_particles=64)
    assert iteration_decision([1.,0.],[8,8],**kwargs)['next_particles']==8
    d=iteration_decision([1.,0.,.5],[8,8,8],**kwargs)
    assert d['action']=='fit' and d['next_particles']==16
    assert iteration_decision([1.,0.,.5],[8,8,16],**kwargs)['next_particles']==16
    assert iteration_decision([0.,0.,0.],[8,8,8],**kwargs)['action']=='fit'
    assert iteration_decision([0.,0.,0.,0.],[8,8,8,8],**kwargs)['action']=='final'
    assert iteration_decision([1.,0.,.5],[64,64,64],**kwargs)['action']=='capacity_veto'
    shifted=iteration_decision([-1001.,-1000.,-999.,-999.],[8]*4,**kwargs)
    plain=iteration_decision([-1.,0.,1.,1.],[8]*4,**kwargs)
    assert shifted==plain
    with pytest.raises(ValueError):iteration_decision([float('nan')],[8],**kwargs)


def test_recursive_fit_preserves_physical_cloud_spread_in_diagnostics():
    from bayesfilter.score_study.iapf_adapter import FIT_DIAGNOSTIC_COLUMNS
    dtype=tf.float64
    # Both coordinate patterns have population variance 2. Their scales are
    # 3 and 1/2; translating the second cloud must not change its spread.
    first=tf.constant([[-6.,-.5],[-3.,1.],[0.,0.],[3.,-1.],[6.,.5]],dtype)
    clouds=tf.stack([first,2*first+tf.constant([7.,-4.],dtype)])
    kernel=make_density_recursive_fit_kernel(2,1,5,2,4.,.2,4.,2000,30,1e-8,.01)
    *_,diagnostics=kernel(tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype),
                         tf.constant([[.5],[-.3]],dtype),clouds)
    assert diagnostics.shape==(2,len(FIT_DIAGNOSTIC_COLUMNS))
    lo=FIT_DIAGNOSTIC_COLUMNS.index('minimum_cloud_sd')
    hi=FIT_DIAGNOSTIC_COLUMNS.index('maximum_cloud_sd')
    tf.debugging.assert_near(diagnostics[:,lo],
        tf.constant([math.sqrt(.5),math.sqrt(2.)],dtype),atol=1e-12)
    tf.debugging.assert_near(diagnostics[:,hi],
        tf.constant([math.sqrt(18.),math.sqrt(72.)],dtype),atol=1e-12)
    assert kernel.experimental_get_tracing_count()==1


def test_consumer_blocks_unadmitted_claim_before_numerical_execution():
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    with pytest.raises(ValueError,match='verified study context'):
        execute_iapf({'role':'claim'},None,None,None,None)


def test_constant_twist_reduces_to_shared_bootstrap_value_and_score():
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    from bayesfilter.score_study.gaussian_tf import make_particle_kernel
    dtype=tf.float64;N=16;T=2
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype);obs=tf.constant([[.4],[-.2]],dtype)
    initial=tf.random.stateless_normal([N,1],[87,1],dtype=dtype)
    noise=tf.random.stateless_normal([T,N,1],[87,2],dtype=dtype)
    uniforms=tf.random.stateless_uniform([T,N],[87,3],dtype=dtype)
    full_uniforms=tf.concat([((tf.cast(tf.range(N),dtype)+.5)/N)[None,:],uniforms],0)
    kernel=make_fitted_twist_kernel(1,1,N,T,constant_twist=True)
    result=kernel(theta,obs,initial,noise,full_uniforms,tf.zeros([T,N],dtype),
        tf.zeros([T,1],dtype),tf.ones([T,1,1],dtype),tf.zeros([T],dtype))
    baseline=make_particle_kernel(1,1,N,T,resampling=True)(theta,obs,initial,noise,uniforms)
    tf.debugging.assert_near(result[0],baseline[0],atol=1e-12)
    tf.debugging.assert_near(result[1],baseline[1],atol=1e-12)


def test_backward_density_fit_and_frozen_analytical_score():
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    dtype=tf.float64;N=32;T=2
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype);obs=tf.constant([[.5],[-.3]],dtype)
    clouds=tf.random.stateless_normal([T,N,1],[91,1],dtype=dtype)
    fitter=make_density_recursive_fit_kernel(1,1,N,T,4.,.2,4.,2000,30,1e-8,.01)
    centers,V,floors,valid,converged,diagnostics=fitter(theta,obs,clouds)
    assert bool(valid) and bool(converged)
    _,_,H,_,_,_,_,_,_,_,R,_=parameterized_model(theta,1,1)
    tf.debugging.assert_near(centers[-1],obs[-1]/H[0,0],atol=2e-5)
    tf.debugging.assert_near(V[-1],R/H[0,0]**2,atol=2e-5)
    kernel=make_fitted_twist_kernel(1,1,N,T)
    args=(obs,tf.random.stateless_normal([N,1],[91,2],dtype=dtype),
          tf.random.stateless_normal([T,N,1],[91,3],dtype=dtype),
          tf.random.stateless_uniform([T+1,N],[91,4],dtype=dtype),
          tf.random.stateless_uniform([T,N],[91,5],dtype=dtype),centers,V,floors)
    value,score,_=kernel(theta,*args)
    for direction in tf.unstack(tf.eye(6,dtype=dtype)):
        h=tf.constant(1e-6,dtype)
        fd=(kernel(theta+h*direction,*args)[0]-kernel(theta-h*direction,*args)[0])/(2*h)
        tf.debugging.assert_near(fd,tf.reduce_sum(score*direction),atol=2e-7,rtol=3e-5)
    assert fitter.experimental_get_tracing_count()==kernel.experimental_get_tracing_count()==1


def _consumer_check():
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    from bayesfilter.score_study.adapters import evaluate_gaussian
    row,context=fixture('iapf')
    row['iapf']=dict(k=1,tau=100.,max_iterations=4,max_particles=128,mean_bound=4.,sd_lower=.2,sd_upper=4.,
                    max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,floor_ratio=.01,
                    fit_theta=context['study']['settings']['theta'])
    result=evaluate_gaussian(row,context);diag=result['diagnostics']
    assert diag['fit_iterations'][0]['iteration']==0 and diag['fit_iterations'][-1]['action']=='final'
    assert diag['fit_method']=='bounded_diagonal_density_scale_least_squares_profiled_scale'
    assert diag['candidate_configuration']=={'iapf':row['iapf']}
    assert diag['actual_particle_count']==diag['fit']['particles']
    assert all(n==1 for n in diag['fit_trace_counts']+diag['run_trace_counts'])
    row['replicate']=1
    other=evaluate_gaussian(row,context)
    assert other['diagnostics']['fit_digest']==diag['fit_digest']
    assert other['diagnostics']['fit_seed_records']['iapf_final_process']!=diag['fit_seed_records']['iapf_final_process']
    assert len(result['score'])==6 and result['numerical_validity']=='pass'


def test_actual_master_consumer_and_final_stream_independence():
    result=subprocess.run([sys.executable,'-c',
        "import runpy,sys; runpy.run_path(sys.argv[1])['_consumer_check']()",str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[2],env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1',
        'BAYESFILTER_PRELOAD_CUSTOM_OP':'0'},capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+result.stderr


def _mixed_precision_consumer_check():
    from unittest.mock import patch
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    from bayesfilter.score_study.adapters import evaluate_gaussian
    from bayesfilter.score_study import iapf_fit_tf,fitted_twist_tf
    row,context=fixture('iapf')
    context['study']['settings']['dtype']='float32'
    row['iapf']=dict(k=1,tau=100.,max_iterations=4,max_particles=128,mean_bound=4.,sd_lower=.2,sd_upper=4.,
        max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,floor_ratio=.01,fit_dtype='float64',
        fit_theta=context['study']['settings']['theta'])
    with patch.object(iapf_fit_tf,'make_density_recursive_fit_kernel',wraps=iapf_fit_tf.make_density_recursive_fit_kernel) as fit_factory, \
         patch.object(fitted_twist_tf,'make_fitted_twist_kernel',wraps=fitted_twist_tf.make_fitted_twist_kernel) as run_factory:
        result=evaluate_gaussian(row,context)
    assert fit_factory.call_count>0 and run_factory.call_count>0
    assert all(c.args[-2]=='float64' for c in fit_factory.call_args_list)
    assert all(c.args[4]=='float32' for c in run_factory.call_args_list)
    diag=result['diagnostics']
    assert diag['fit_precision']==dict(fit_dtype='float64',filter_dtype='float32',
        cast_policy='explicit_offline_fit_then_cast_coefficients_once',automatic_precision_fallback=False)
    assert result['numerical_validity']=='pass' and len(result['score'])==6
    lo=diag['fit_diagnostic_columns'].index('minimum_cloud_sd')
    hi=diag['fit_diagnostic_columns'].index('maximum_cloud_sd')
    for fit in diag['fit_iterations']:
        if 'coefficient_cast_valid' not in fit:continue
        assert fit['coefficient_cast_valid']
        assert max(fit['coefficient_cast_max_abs_error'].values())<1e-5
        assert all(0<step[lo]<=step[hi] for step in fit['density_fit_diagnostics'])
    row['iapf']['fit_dtype']='float16'
    with pytest.raises(ValueError,match='fit_dtype'):
        evaluate_gaussian(row,context)


def test_explicit_offline_fit_precision_reaches_actual_consumer():
    result=subprocess.run([sys.executable,'-c',
        "import runpy,sys; runpy.run_path(sys.argv[1])['_mixed_precision_consumer_check']()",str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[2],env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1',
        'BAYESFILTER_PRELOAD_CUSTOM_OP':'0'},capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+result.stderr

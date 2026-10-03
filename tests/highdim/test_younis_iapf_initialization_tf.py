"""Known-target, gradient, rejection and real-consumer checks for optional fitting."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tensorflow as tf

from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit, _density_profile

FIT=dict(mean_bound=4.,sd_lower=.2,sd_upper=4.,max_steps=2000,
         max_backtracks=30,tolerance=1e-7,floor_ratio=.01)


def test_fixed_scale_is_constant_in_the_analytical_gradient():
    dtype=tf.float64
    z=tf.constant([[-1.2,.2],[.5,-.8],[1.4,.9],[.1,.5]],dtype)
    target=tf.math.log(tf.constant([.3,1.,.6,.2],dtype))
    par=tf.constant([.2,-.1,.1,-.2],dtype); scale=tf.constant(-2.3,dtype)
    native=_density_profile(z,target,par)
    value,grad,*_=_density_profile(z,target,par,log_density_scale=scale)
    tf.debugging.assert_near(value,native[0]*tf.exp(-2*scale),atol=1e-12)
    tf.debugging.assert_near(grad,native[1]*tf.exp(-2*scale),atol=1e-12)
    h=tf.constant(1e-5,dtype)
    for direction in tf.unstack(tf.eye(4,dtype=dtype)):
        fd=(_density_profile(z,target,par+h*direction,log_density_scale=scale)[0]-
            _density_profile(z,target,par-h*direction,log_density_scale=scale)[0])/(2*h)
        tf.debugging.assert_near(fd,tf.reduce_sum(grad*direction),atol=1e-7,rtol=1e-7)


def test_high_dimension_exact_target_and_amplitude_invariance():
    d=40;dtype=tf.float64
    points=tf.random.stateless_normal([256,d],[92071,d],dtype=dtype)
    center=.4*tf.sin(tf.cast(tf.range(1,d+1),dtype))
    variance=tf.linspace(tf.constant(.6,dtype),tf.constant(.9,dtype),d)
    target=-.5*tf.reduce_sum((points-center)**2/variance,axis=1)
    for scale in ("native","initial_peak"):
        @tf.function(input_signature=[tf.TensorSpec([256,d],dtype),tf.TensorSpec([256],dtype)],jit_compile=True)
        def fit(x,y):
            return bounded_density_fit(x,y,**FIT,initialization="log_quadratic",objective_scale=scale)
        c,v,_,info=fit(points,target)
        assert bool(info['valid']) and bool(info['converged']) and bool(info['initialization_valid'])
        tf.debugging.assert_near(c,center,atol=1e-10,rtol=0.)
        tf.debugging.assert_near(tf.linalg.diag_part(v),variance,atol=1e-10,rtol=0.)
        c2,v2,_,info2=fit(points,target+100.)
        tf.debugging.assert_near(c,c2,atol=1e-10,rtol=0.)
        tf.debugging.assert_near(v,v2,atol=1e-10,rtol=0.)
        tf.debugging.assert_near(info2['log_lambda'],info['log_lambda']-100.,atol=1e-10,rtol=0.)
        assert fit.experimental_get_tracing_count()==1


def test_rank_concavity_and_configuration_fail_closed():
    dtype=tf.float64
    x=tf.random.stateless_normal([32,2],[812,3],dtype=dtype)
    duplicate=tf.stack([x[:,0],x[:,0]],axis=1)
    *_,info=bounded_density_fit(duplicate,-tf.reduce_sum(duplicate**2,1),**FIT,initialization="log_quadratic")
    assert not bool(info['initialization_valid']) and not bool(info['valid'])
    *_,info=bounded_density_fit(x,tf.reduce_sum(x**2,1),**FIT,initialization="log_quadratic")
    assert not bool(info['initialization_valid']) and not bool(info['valid'])
    with pytest.raises(ValueError,match='N >='):
        bounded_density_fit(x[:4],tf.zeros([4],dtype),**FIT,initialization="log_quadratic")
    with pytest.raises(ValueError,match='requires density_l2'):
        bounded_density_fit(x,tf.zeros([32],dtype),**FIT,objective="relative_shape",objective_scale="initial_peak")


def _check_consumer():
    from copy import deepcopy
    from unittest.mock import patch
    from bayesfilter.score_study.adapters import evaluate_gaussian
    from bayesfilter.score_study.iapf_scope import validate_result_accounting
    from bayesfilter.score_study import iapf_fit_tf
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    row,context=fixture("iapf");settings=context['study']['settings']
    settings.update(dimension=2,observation_dimension=2,horizon=4,particles=64)
    row['iapf']=dict(k=2,tau=.5,max_iterations=12,max_particles=512,mean_bound=4.,
        sd_lower=.2,sd_upper=4.,max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,
        floor_ratio=.01,fit_theta=settings['theta'],fit_initialization="log_quadratic",fit_objective_scale="initial_peak")
    with patch.object(iapf_fit_tf,'make_density_recursive_fit_kernel',wraps=iapf_fit_tf.make_density_recursive_fit_kernel) as factory:
        result=evaluate_gaussian(row,context)
    assert result['numerical_validity']=='pass' and factory.call_count>=3
    assert all(c.kwargs['initialization']=='log_quadratic' and c.kwargs['objective_scale']=='initial_peak' for c in factory.call_args_list)
    validate_result_accounting(result,row,settings)
    for key in ['fit_initialization','fit_objective_scale']:
        tampered=deepcopy(result);tampered['diagnostics'][key]='wrong'
        with pytest.raises(ValueError,match='initialization/scale'):
            validate_result_accounting(tampered,row,settings)


def test_optional_controls_reach_real_consumer_and_bind_configuration():
    root=Path(__file__).resolve().parents[2]
    command=[sys.executable,'-c','from tests.highdim.test_younis_iapf_initialization_tf import _check_consumer; _check_consumer()']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
         'TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','OMP_NUM_THREADS':'1'}
    run=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=150)
    assert run.returncode==0,run.stdout+'\n'+run.stderr

"""Independent CPU/FP64 diagnostic checks for the chapter candidate.

NumPy and finite differences below are independent diagnostics, never runtime.
No claim about exact nonlinear filtering follows from finite-program parity.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import dataclasses
import numpy as np
import pytest
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_tf import (ProposalControls,build_maps,component_densities,draw_from_maps,make_filter,moments,reset_cloud,TRACE_FIELDS)
from bayesfilter.highdim.covariance_proposal_beta_tf import make_beta_solver,pilot_objective,project_simplex
from bayesfilter.highdim.covariance_proposal_moments_tf import repair_moments
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
D=tf.float64

def fixed(n,d,h=2):
    return (tf.random.stateless_normal([n,d],[41,1],dtype=D),tf.random.stateless_normal([h,n,d],[41,2],dtype=D),tf.random.stateless_uniform([h,n,2],[41,3],dtype=D))

def test_beta_boundary_optimum_gap_and_simplex():
    ratios=tf.constant([[[1.,2.,.5]]]*16,D)
    lc=tf.math.log(ratios); target=tf.zeros([16,1],D)
    fit=make_beta_solver(16,1,D,jit_compile=False)(lc,target,tf.constant(.1,D),tf.constant(1.e-10,D))
    assert bool(fit['converged'])
    np.testing.assert_allclose(fit['beta'],[.1,.9,0],atol=2e-10)
    assert float(fit['gap'])<1e-10
    for v in ([100.,-2.,4.],[-9.,-7.,-5.],[.2,.3,.4]):
        b=project_simplex(tf.constant(v,D),tf.constant(.2,D)).numpy()
        assert b[0]>=.2-1e-13 and min(b)>=0
        assert abs(sum(b)-1)<1e-13

def test_beta_derivatives_convexity():
    ratios=tf.exp(tf.random.stateless_normal([2,17,3],[2,6],dtype=D))
    lam=tf.ones([2,17],D)/34; b=tf.constant([.2,.3,.5],D)
    f,g,h=pilot_objective(b,ratios,lam)
    for k in range(3):
        e=tf.one_hot(k,3,dtype=D)*1e-5
        fp,gp,_=pilot_objective(b+e,ratios,lam);fm,gm,_=pilot_objective(b-e,ratios,lam)
        np.testing.assert_allclose((fp-fm)/2e-5,g[k],rtol=1e-7)
        np.testing.assert_allclose((gp-gm)/2e-5,h[:,k],rtol=1e-7)
    assert float(tf.reduce_min(tf.linalg.eigvalsh(h)))>=-1e-12

def test_maps_use_conditional_covariance_and_density_matches_sampler():
    spec=LGSSMSpec('diagonal_ar',1);theta=tf.constant([.9,.6,.8],D)
    model,_=spec.model(theta,tf.zeros_like(theta)); initial,noise,u=fixed(32,1,1)
    x=3*initial;ctl=ProposalControls(flow_steps=32)
    maps=build_maps(model,theta,x,tf.zeros_like(x),tf.constant([.5],D),ctl)
    # Scalar quadratic bridge has endpoint covariance Q R/(Q+R); refinement.
    expected=.36*.64/(.36+.64)
    np.testing.assert_allclose(maps[2][2,:,0,0],expected,rtol=.02)
    beta=tf.constant([.2,.3,.5],D);sample,ds=draw_from_maps(maps,beta,u[0],noise[0])
    ld,_,ok=component_densities(sample,ds,*maps[:4]); assert bool(ok)
    mus=maps[0].numpy()[...,0]; variances=maps[2].numpy()[...,0,0]
    xx=sample.numpy()[:,0]
    manual=[]
    for branch in range(3):
        densities=np.exp(-.5*(xx[:,None]-mus[branch][None])**2/variances[branch][None])/np.sqrt(2*np.pi*variances[branch][None])
        manual.append(np.log(np.mean(densities,1)))
    np.testing.assert_allclose(ld,np.stack(manual,1),rtol=1e-12,atol=1e-12)
    q=tf.reduce_logsumexp(ld+tf.math.log(beta)[None],1)
    assert float(tf.reduce_max(tf.exp(ld[:,0]-q)))<=5+1e-12
    one=draw_from_maps(maps,tf.constant([1.,0.,0.],D),u[0],noise[0])[0]
    j=tf.cast(u[0,:,1]*32,tf.int32)
    np.testing.assert_allclose(one,tf.gather(maps[8],j)+.6*noise[0],atol=1e-13)

@pytest.mark.parametrize('correction',[0,2])
def test_reset_moments_caps_and_total_derivative(correction):
    n=32;d=3;x=fixed(n,d)[0]; dx=tf.random.stateless_normal([n,d],[9,8],dtype=D)*.2
    lw=tf.random.stateless_normal([n],[9,9],dtype=D)*.3;dlw=tf.linspace(tf.constant(-.1,D),tf.constant(.1,D),n)
    ctl=ProposalControls(reset_epsilon=2.,correction_steps=correction)
    def calculation(e,derivative):
        xx=x+e*dx;ll=lw+e*dlw;ww=tf.nn.softmax(ll)
        dd=dx if derivative else tf.zeros_like(dx)
        dlog=dlw-tf.reduce_sum(ww*dlw) if derivative else tf.zeros_like(dlw)
        z,dz,mu,dmu,p,dp,valid,diagnostics=reset_cloud(xx,dd,tf.nn.log_softmax(ll),dlog,ctl)
        assert bool(valid)
        before=z
        if correction:
            z,dz,info=repair_moments(xx,dd,ww,ww*dlog,z,dz,mu,dmu,p,dp,ctl)
            assert float(info[0])<=ctl.correction_radius+1e-12
            coordinate_displacement=tf.abs(z-before)/tf.sqrt(tf.linalg.diag_part(p))[None]
            assert float(tf.reduce_max(coordinate_displacement))<=ctl.correction_radius+1e-12
        m=tf.reduce_mean(z,0);zz=z-m;cov=tf.einsum('ni,nj->ij',zz,zz)/n
        np.testing.assert_allclose(m,mu,atol=1e-12)
        np.testing.assert_allclose(cov,p,atol=2e-12)
        return z,dz
    z,dz=calculation(0.,True)
    zp,_=calculation(1e-4,False);zm,_=calculation(-1e-4,False)
    np.testing.assert_allclose((zp-zm)/2e-4,dz,rtol=2e-5,atol=2e-7)

@pytest.mark.parametrize('name',['lgssm','ksc','predator_prey','sir_d18'])
@pytest.mark.parametrize('correction',[0,2])
def test_complete_recursive_score_all_parameters(name,correction):
    spec=LGSSMSpec('frozen_3d',3) if name=='lgssm' else KSCSpec() if name=='ksc' else NonlinearSQMCSpec(name)
    theta=spec.default_theta(D);n=32;h=2;ctl=ProposalControls(reset_epsilon=2.,correction_steps=correction)
    base=fixed(n,spec.dimension,h);model,_=spec.model(theta,tf.zeros_like(theta))
    xx=spec.initial_cloud(theta,base[0])[0]
    obs=[]
    for t in range(h):
        xx=model.transition_mean_fn(theta,xx)
        obs.append(tf.reduce_mean(model.observation_fn(xx),0)+.15)
    obs=tf.stack(obs);beta=tf.constant([.2,.3,.5],D)
    fn=make_filter(spec,n,h,ctl,D,jit_compile=False)
    center=fn(theta,tf.zeros_like(theta),obs,*base,beta)
    assert bool(center['valid']),center['trace'].numpy()
    for k in range(spec.parameter_count):
        direction=tf.one_hot(k,spec.parameter_count,dtype=D)
        analytical=fn(theta,direction,obs,*base,beta)
        assert bool(analytical['valid'])
        errors=[]
        for hstep in (1e-4,5e-5):
            eps=hstep*max(1.,abs(float(theta[k])))
            plus=fn(theta+eps*direction,tf.zeros_like(theta),obs,*base,beta)
            minus=fn(theta-eps*direction,tf.zeros_like(theta),obs,*base,beta)
            assert bool(plus['valid']) and bool(minus['valid'])
            fd=(float(plus['value'])-float(minus['value']))/(2*eps)
            errors.append(abs(fd-float(analytical['score'])))
        assert errors[-1]<2e-4*max(1.,abs(float(analytical['score']))),(name,k,errors,float(analytical['score']))
        np.testing.assert_allclose(center['value'],analytical['value'],atol=1e-12)


def test_invalid_rank_is_flagged_and_no_silent_ridge():
    x=tf.zeros([32,3],D);logw=tf.fill([32],-tf.math.log(tf.constant(32.,D)))
    result=reset_cloud(x,x,logw,tf.zeros([32],D),ProposalControls())
    assert not bool(result[6])


def test_master_calls_single_runtime_authority():
    import ast
    from pathlib import Path
    source=Path('docs/benchmarks/run_ledh_covariance_proposal.py').read_text()
    tree=ast.parse(source)
    imports=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert any(n.module=='bayesfilter.highdim.covariance_proposal_tf' and any(a.name=='make_filter' for a in n.names) for n in imports)
    import bayesfilter.highdim.covariance_proposal_tf as runtime
    assert make_filter is runtime.make_filter


def test_cayley_solve_orthogonality_and_mixed_partial():
    from bayesfilter.highdim.covariance_proposal_moments_tf import skew_basis,cayley_jets
    n=32;r=4; basis=skew_basis(n,r,D);v=tf.constant([.8,-.2,.1,1.2],D);dv=tf.constant([.1,.3,-.1,.05],D)
    kappa=tf.constant(.25/np.sqrt(n),D)
    o,do,go,dgo=cayley_jets(v,dv,basis,kappa)
    g=tf.einsum('r,rij->ij',v,basis);k=kappa*g/tf.sqrt(1+tf.reduce_sum(g*g))
    h=tf.eye(n,dtype=D)-k/2
    np.testing.assert_allclose(tf.matmul(h,o),tf.eye(n,dtype=D)+k/2,atol=1e-14)
    np.testing.assert_allclose(tf.matmul(o,o,transpose_b=True),tf.eye(n,dtype=D),atol=1e-14)
    np.testing.assert_allclose(tf.linalg.matvec(o,tf.ones([n],D)),tf.ones([n],D),atol=1e-14)
    eps=1e-5
    op,_,gp,_=cayley_jets(v+eps*dv,tf.zeros_like(dv),basis,kappa)
    om,_,gm,_=cayley_jets(v-eps*dv,tf.zeros_like(dv),basis,kappa)
    np.testing.assert_allclose((op-om)/(2*eps),do,rtol=2e-6,atol=2e-10)
    np.testing.assert_allclose((gp-gm)/(2*eps),dgo,rtol=2e-6,atol=2e-10)


def test_runtime_call_chain_and_fail_closed(monkeypatch):
    import bayesfilter.highdim.covariance_proposal_tf as core
    import bayesfilter.highdim.covariance_proposal_moments_tf as correction
    calls=[]
    for module,name in [(core,'build_maps'),(core,'component_densities'),(core,'reset_cloud'),(correction,'repair_moments')]:
        original=getattr(module,name)
        def wrapper(*a,_original=original,_name=name,**kw):
            calls.append(_name)
            return _original(*a,**kw)
        monkeypatch.setattr(module,name,wrapper)
    spec=LGSSMSpec('diagonal_ar',1);theta=tf.constant([.8,.6,.8],D)
    fn=core.make_filter(spec,32,2,ProposalControls(correction_steps=2),D,jit_compile=False)
    base=fixed(32,1,2);obs=tf.constant([[.2],[.3]],D)
    result=fn(theta,tf.zeros_like(theta),obs,*base,tf.constant([.2,.3,.5],D))
    assert bool(result['valid']) and int(result['steps_completed'])==2
    assert set(calls)=={'build_maps','component_densities','reset_cloud','repair_moments'}
    invalid=fn(theta,tf.zeros_like(theta),obs,*base,tf.constant([0.,.5,.5],D))
    assert not bool(invalid['valid']) and int(invalid['steps_completed'])==0
    assert bool(tf.reduce_all(tf.math.is_nan(invalid['trace'])))


@pytest.mark.parametrize('field',['reset_epsilon','correction_rate','correction_radius'])
def test_nonfinite_controls_rejected(field):
    with pytest.raises(ValueError): ProposalControls(**{field:float('nan')})

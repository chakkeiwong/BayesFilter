"""CPU/reference mechanics of the optional guarded higher-moment correction."""
import tensorflow as tf
import pytest
from bayesfilter.highdim.higher_moment_contract_e import (
    higher_moment_shape_jvp, _standardize_uniform_jvp)
from bayesfilter.highdim.moment_safety_tf import guarded_moment_step
from bayesfilter.highdim.sqmc_campaign_tf import reset_design
from bayesfilter.highdim.ledh_unified_correction_tf import batched_higher_moment_shape_jvp

D = tf.float64


def fixture(d=2, seed=19):
    n = 48 * d
    q = reset_design(n, d, D, "normal_quantiles")
    x = q + .15 * tf.square(q)
    w = tf.nn.softmax(tf.linspace(tf.constant(-.2, D), tf.constant(.2, D), n))
    dx = tf.random.stateless_normal([n,d,1], [seed,3], dtype=D) * .01
    dq = tf.random.stateless_normal([n,d,1], [seed,5], dtype=D) * .01
    dw = tf.linspace(tf.constant(-.001,D), tf.constant(.001,D), n)[:,None]
    return x,w,dx,dw,q,dq


def controls(**changes):
    config = dict(correction_steps=2,strength=.12,diagonal_lm_damping=.01,
                  diagonal_lm_scale_floor=1e-4,diagonal_trust_radius=.5,
                  pairwise_correction_steps=1,pairwise_strength=.03,
                  pairwise_particle_rms_cap=2.,coordinatewise_standardized_cap=.98,
                  coordinatewise_standardized_identity_radius=8.)
    config.update(changes)
    return config


def test_collapsing_trial_is_reduced_before_cholesky():
    q=fixture(1)[4]; dq=tf.zeros_like(q[:,:,None])
    q,dq=_standardize_uniform_jvp(q,dq)
    out, dout, stats=guarded_moment_step(q,dq,-q,-dq,
        normalize=_standardize_uniform_jvp,loss=lambda u: tf.reduce_mean(u**4))
    assert float(stats[0]) == 2 and float(stats[2]) == .5
    tf.debugging.assert_near(out,q,atol=2e-11)
    tf.debugging.assert_all_finite(dout,"guard tangent")


def test_unacceptable_displacement_preserves_input_and_tangent():
    q=fixture(1)[4]; dq=tf.ones_like(q[:,:,None])*.03
    q,dq=_standardize_uniform_jvp(q,dq)
    out,dout,stats=guarded_moment_step(q,dq,1000*q,1000*dq,
        normalize=_standardize_uniform_jvp,loss=lambda u: tf.reduce_mean(u**4))
    assert float(stats[0]) == 8 and float(stats[1]) == 1
    tf.debugging.assert_equal(out,q);tf.debugging.assert_equal(dout,dq)


def test_healthy_nonzero_step_is_unchanged():
    args=fixture(1)
    cfg=controls(correction_steps=1,strength=.03,pairwise_correction_steps=0)
    old=higher_moment_shape_jvp(*args,**cfg)
    new=higher_moment_shape_jvp(*args,**cfg,moment_safety=True)
    assert float(new["moment_safety_minimum_step"]) == 1.
    assert float(new["moment_safety_final_rejected"]) == 0.
    for key in ("particles","particles_tangent"):
        tf.debugging.assert_equal(new[key],old[key])


@pytest.mark.parametrize("dimension",[1,2,3])
def test_complete_map_and_batch_agree_with_total_finite_difference(dimension):
    args=fixture(dimension)
    cfg=controls(moment_safety=True,return_stages=True)
    result=higher_moment_shape_jvp(*args,**cfg)
    assert bool(result["valid"])
    assert float(result["moment_safety_final_loss"]) <= float(result["moment_safety_baseline_loss"])+1e-12
    batch_args=(args[0][None],args[1][None],tf.transpose(args[2],[2,0,1])[:,None],
                tf.transpose(args[3])[...,None,:],args[4][None],tf.transpose(args[5],[2,0,1])[:,None])
    batched=batched_higher_moment_shape_jvp(*batch_args,**cfg)
    tf.debugging.assert_near(result["particles"],batched["particles"][0],atol=2e-12)
    s,w,ds,dw,p,dp=args;h=tf.constant(1e-5,D)
    def shifted(sign):
        return higher_moment_shape_jvp(s+sign*h*ds[:,:,0],w+sign*h*dw[:,0],ds,dw,
                                      p+sign*h*dp[:,:,0],dp,**cfg)
    plus,minus=shifted(1.),shifted(-1.)
    for key in ("moment_safety_trials","moment_safety_rejected_steps","moment_safety_final_rejected"):
        tf.debugging.assert_equal(plus[key],minus[key])
    fd=(plus["particles"]-minus["particles"])/(2*h)
    tf.debugging.assert_near(result["particles_tangent"][:,:,0],fd,atol=1e-7,rtol=1e-5)


def test_nonfinite_displacement_is_rejected():
    q=fixture(1)[4];dq=tf.zeros_like(q[:,:,None])
    out,dout,stats=guarded_moment_step(q,dq,q*float("nan"),dq,
        normalize=_standardize_uniform_jvp,loss=lambda u: tf.reduce_mean(u**4))
    assert float(stats[1]) == 1.
    tf.debugging.assert_equal(out,q);tf.debugging.assert_equal(dout,dq)


def test_canonical_endpoint_retains_guard_and_pairwise_diagnostics():
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
    spec=LGSSMSpec('diagonal_ar',2)
    theta=spec.default_theta(D)
    direction=tf.one_hot(0,spec.parameter_count,dtype=D)
    design=reset_design(24,2,D,'normal_quantiles')
    noise=tf.random.stateless_normal([2,24,2],[712,3],dtype=D)
    observations=tf.constant([[.2,-.1],[.4,.1]],D)
    def run(parameter):
        model,_=spec.model(parameter,direction)
        states,covariances,ds,dc=spec.initial_cloud(parameter,design,direction)
        return canonical_value_and_analytical_score(model,parameter,states,covariances,noise,observations,
            flow_substeps=2,with_score=True,return_trace=True,initial_state_tangent=ds,
            initial_covariance_tangent=dc,reset_policy='contract_e',reset_design=design,
            reset_sinkhorn_steps=3,reset_balance_steps=3,correction_steps=1,correction_strength=.03,
            pairwise_steps=1,pairwise_strength=.01,moment_safety=True,coordinate_cap=.98,
            coordinate_cap_identity_radius=8.)
    value,score,trace=run(theta)
    tf.debugging.assert_all_finite(value,'canonical value')
    tf.debugging.assert_all_finite(score,'canonical score')
    for step in trace:
        assert float(step['higher_moment_moment_safety_enabled'])==1.
        assert bool(step['higher_moment_pairwise_configured'])
    h=tf.constant(1e-5,D)
    vp,_,tp=run(theta+h*direction);vm,_,tm=run(theta-h*direction)
    for p,m in zip(tp,tm):
        for key in ('higher_moment_moment_safety_trials','higher_moment_moment_safety_final_rejected'):
            tf.debugging.assert_equal(p[key],m[key])
    tf.debugging.assert_near(score[0],(vp-vm)/(2*h),atol=2e-5,rtol=2e-4)


def test_roundoff_loewner_margin_cannot_admit_singular_trial():
    q=fixture(2)[4]
    scale=tf.sqrt(tf.constant(32.*2.**-52,D))
    q=q*tf.stack([scale,tf.constant(1.,D)])[None,:]
    dq=tf.zeros_like(q[:,:,None])
    displacement=tf.stack([-q[:,0],tf.zeros_like(q[:,1])],axis=1)
    def normalize(value,tangent):
        covariance=tf.transpose(value)@value/tf.cast(tf.shape(value)[0],D)
        tf.debugging.assert_positive(tf.linalg.eigvalsh(covariance))
        return _standardize_uniform_jvp(value,tangent,relative_ridge=False)
    out,dout,stats=guarded_moment_step(q,dq,displacement,dq,
        normalize=normalize,loss=lambda u: tf.zeros([],D))
    assert float(stats[0])==3. and float(stats[2])==.25
    tf.debugging.assert_all_finite(out,'strictly positive trial')
    tf.debugging.assert_all_finite(dout,'strictly positive trial tangent')


def test_final_fallback_selects_protected_baseline_and_its_total_tangent(monkeypatch):
    # Isolate final selection: inject an adverse intermediate after the step
    # guard. This is a wiring test, not evidence that real proposals accept it.
    import bayesfilter.highdim.higher_moment_contract_e as implementation
    original=implementation._shape_iteration_jvp
    def adverse(*args,**kwargs):
        values=list(original(*args,**kwargs))
        values[0],values[1]=_standardize_uniform_jvp(args[0]**7,7*args[0][:,:,None]**6*args[1])
        return tuple(values)
    args=fixture(1)
    baseline=higher_moment_shape_jvp(*args,**controls(moment_safety=True,correction_steps=0,pairwise_correction_steps=0))
    monkeypatch.setattr(implementation,'_shape_iteration_jvp',adverse)
    result=higher_moment_shape_jvp(*args,**controls(moment_safety=True,correction_steps=1,pairwise_correction_steps=0))
    assert float(result['moment_safety_final_rejected'])==1.
    for key in ('particles','particles_tangent'):
        tf.debugging.assert_equal(result[key],baseline[key])

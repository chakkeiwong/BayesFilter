"""Independent diagnostic checks of shared analytical transport derivatives."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet
from bayesfilter.highdim.covariance_proposal_score_batch_tf import transport, gaussian_mixture
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, _dense_reset_transport

@pytest.mark.parametrize("tangent_offset", [2048., 262144.])
@pytest.mark.parametrize("reset_steps", [1, 5])
def test_packed_transport_large_common_offsets(tangent_offset, reset_steps):
    # Distance derivatives must be invariant to common position AND tangent
    # offsets. Direct differences provide the independent arithmetic reference.
    p,b,n,d=3,2,32,3
    controls=ProposalControls(reset_steps=reset_steps,reset_epsilon=.5)
    x=tf.random.stateless_normal([b,n,d],[11,7],dtype=tf.float32)+tf.constant(4096.)
    dx=tf.random.stateless_normal([p,b,n,d],[13,7],dtype=tf.float32)+tf.constant(tangent_offset)
    lw=tf.nn.log_softmax(tf.random.stateless_normal([b,n],[17,7]),axis=-1)
    dlw=tf.random.stateless_normal([p,b,n],[19,7])
    dlw-=tf.reduce_sum(tf.exp(lw)[None]*dlw,-1,keepdims=True)
    @tf.function(input_signature=[tf.TensorSpec(x.shape,x.dtype),tf.TensorSpec(dx.shape,dx.dtype),tf.TensorSpec(lw.shape,lw.dtype),tf.TensorSpec(dlw.shape,dlw.dtype)],jit_compile=True,autograph=False)
    def actual(x,dx,lw,dlw):
        y,_,_=transport(Jet(x,dx),Jet(lw,dlw),controls)
        return y.v,y.t
    value,tangent=actual(x,dx,lw,dlw)
    for bi in range(b):
        for pi in range(p):
            expected,derivative,_,_=_dense_reset_transport(tf.cast(x[bi],tf.float64),tf.cast(dx[pi,bi],tf.float64),tf.cast(lw[bi],tf.float64),tf.cast(dlw[pi,bi],tf.float64),controls)
            expected=tf.cast(expected,tf.float32); derivative=tf.cast(derivative,tf.float32)
            tf.debugging.assert_near(value[bi],expected,atol=.02,rtol=1e-6)
            # A relative tolerance on the common shift would hide cancellation.
            tf.debugging.assert_near(tangent[pi,bi]-tangent_offset,
                                     derivative-tangent_offset,atol=.08,rtol=1e-5)


def test_failed_batch_row_freezes_without_corrupting_healthy_row(monkeypatch):
    # A failure occurs after computing a proposal. Preserve the last healthy
    # state, while the other row completes its own observation horizon.
    from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch
    from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
    dtype=tf.float64
    spec=LGSSMSpec("diagonal_ar",1)
    n,h=32,2
    controls=ProposalControls(flow_steps=2,reset_steps=4)
    theta=tf.constant([[.9,.6,.8],[.88,.61,.82]],dtype)
    obs=tf.constant([[.5],[.3]],dtype)
    initial=tf.random.stateless_normal([n,1],[41,1],dtype=dtype)
    noise=tf.random.stateless_normal([h,n,1],[41,2],dtype=dtype)
    uniforms=tf.random.stateless_uniform([h,n,2],[41,3],dtype=dtype)
    beta=tf.constant([.2,.3,.5],dtype)
    original=batch.reset_cloud
    def faulty(x,logw,controls,precision_policy="uniform"):
        z,mu,cov,valid,diagnostics=original(x,logw,controls,precision_policy)
        if x.v.shape[0]==2:
            z=Jet(tf.concat([z.v[:1],z.v[1:]+100.],axis=0),z.t)
        return z,mu,cov,valid,diagnostics
    monkeypatch.setattr(batch,"reset_cloud",faulty)
    parallel=batch.make_batched_score_filter(spec,n,h,controls,dtype,True,batch_size=2)
    scalar=batch.make_batched_score_filter(spec,n,h,controls,dtype,True,batch_size=1)
    result=parallel(theta,obs,initial,noise,uniforms,beta)
    independent=scalar(theta[:1],obs,initial,noise,uniforms,beta)
    tf.debugging.assert_equal(result["valid"],[True,False])
    tf.debugging.assert_equal(result["steps_completed"],[h,1])
    from bayesfilter.highdim.covariance_proposal_tf import VALIDITY_FIELDS
    assert not bool(result["validity_trace"][1,0,VALIDITY_FIELDS.index("mean")])
    tf.debugging.assert_equal(result["validity_trace"][1,1],
                              tf.zeros([len(VALIDITY_FIELDS)],tf.bool))
    tf.debugging.assert_near(result["value"][0],independent["value"][0],atol=1e-8,rtol=1e-8)
    tf.debugging.assert_near(result["score"][0],independent["score"][0],atol=1e-8,rtol=1e-8)
    expected=spec.initial_cloud(theta[1],initial)[0]
    tf.debugging.assert_equal(result["particles"][1],expected)
    tf.debugging.assert_equal(tf.reduce_all(tf.math.is_finite(result["particles"])),True)


@pytest.mark.parametrize("contraction", ["direct","matrix"])
@pytest.mark.parametrize("dimension,shared", [(1,True),(3,False),(18,True),(18,False)])
@pytest.mark.parametrize("dtype", [tf.float64,tf.float32])
def test_mixture_analytical_score_against_independent_finite_values(dimension,shared,dtype,contraction,tangent_offset=0.):
    # Direct Cholesky densities define a separate finite-value authority.
    pcount,batch,n,d=3,2,32,dimension
    x=tf.random.stateless_normal([batch,n,d],[52,d],dtype=dtype)
    mu=tf.random.stateless_normal([batch,n,d],[53,d],dtype=dtype)
    dx=tf.random.stateless_normal([pcount,batch,n,d],[54,d],dtype=dtype)*.1
    dm=tf.random.stateless_normal([pcount,batch,n,d],[55,d],dtype=dtype)*.1
    dx=dx+tf.constant(tangent_offset,dtype)
    dm=dm+tf.constant(tangent_offset,dtype)
    components=1 if shared else n
    root=tf.random.stateless_normal([batch,components,d,d],[56,d],dtype=dtype)
    cov=tf.linalg.matmul(root,root,transpose_b=True)/d+tf.eye(d,batch_shape=[batch,components],dtype=dtype)
    raw=tf.random.stateless_normal([pcount,batch,components,d,d],[57,d],dtype=dtype)
    dc=.03*(raw+tf.linalg.matrix_transpose(raw))
    inputs=(x,dx,mu,dm,cov,dc)
    @tf.function(input_signature=[tf.TensorSpec(v.shape,dtype) for v in inputs],
                 jit_compile=True,autograph=False)
    def actual(xx,ddx,mm,ddm,cc,ddc):
        answer,valid=gaussian_mixture(Jet(xx,ddx),Jet(mm,ddm),Jet(cc,ddc),shared,contraction=contraction)
        return answer.v,answer.t,valid
    value,derivative,valid=actual(*inputs)
    tf.debugging.assert_equal(valid,[True]*batch)
    if tangent_offset:
        unchanged,_,_=actual(x,dx-tangent_offset,mu,dm-tangent_offset,cov,dc)
        tf.debugging.assert_equal(value,unchanged)
    # Compare both modes with FP64 values of the identical quantized inputs.
    x,dx,mu,dm,cov,dc=[tf.cast(v,tf.float64) for v in inputs]
    reference_dtype=tf.float64
    def dense_value(xx,mm,cc):
        factor=tf.linalg.cholesky(cc)
        residual=xx[:,:,None]-mm[:,None]
        solved=tf.linalg.cholesky_solve(factor[:,None],residual[...,None])[...,0]
        logs=-.5*d*tf.math.log(tf.constant(2.*3.141592653589793,reference_dtype))            -tf.math.log(tf.cast(n,reference_dtype))-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)),-1)[:,None]            -.5*tf.reduce_sum(residual*solved,-1)
        return tf.reduce_logsumexp(logs,-1)
    tolerance=2e-7 if dtype==tf.float64 else 2e-5
    tf.debugging.assert_near(tf.cast(value,tf.float64),dense_value(x,mu,cov),atol=tolerance,rtol=tolerance)
    width=tf.constant(1e-5,reference_dtype)
    for axis in range(pcount):
        fd=(dense_value(x+width*dx[axis],mu+width*dm[axis],cov+width*dc[axis])
            -dense_value(x-width*dx[axis],mu-width*dm[axis],cov-width*dc[axis]))/(2*width)
        tf.debugging.assert_near(tf.cast(derivative[axis],tf.float64),fd,atol=tolerance,rtol=tolerance)


def test_mixture_common_tangent_translation():
    # The same large tangent in particle and component means cancels exactly
    # in their differences. FP64 densities of the FP32-quantized inputs are
    # the independent authority, preserving quantization rather than assuming
    # the unrounded random increments are representable at this scale.
    test_mixture_analytical_score_against_independent_finite_values(
        18,False,tf.float32,"direct",tangent_offset=262144.)


@pytest.mark.parametrize("name", ["lgssm","ksc","predator_prey","sir_d18"])
def test_constant_jacobian_maps_match_full_particle_assembly(name):
    from docs.benchmarks.run_ledh_covariance_proposal import spec_for
    from bayesfilter.highdim.covariance_proposal_score_tf import Models, build_maps
    spec=spec_for(name,lgssm_family="p44")
    dtype=tf.float64
    theta=spec.default_theta(dtype)
    p,n,d=spec.parameter_count,32,spec.dimension
    design=tf.random.stateless_normal([n,d],[72,19],dtype=dtype)
    directions=tf.eye(p,dtype=dtype)
    initial=[spec.initial_cloud(theta,design,directions[k]) for k in range(p)]
    xv=initial[0][0]
    xt=tf.stack([row[2] for row in initial])
    obs=spec.model(theta,directions[0])[0].observation_fn(xv[:1])[0]
    controls=ProposalControls(flow_steps=3,reset_steps=4)
    @tf.function(input_signature=[tf.TensorSpec(xv.shape,dtype),tf.TensorSpec(xt.shape,dtype)],jit_compile=True,autograph=False)
    def compare(xv,xt):
        models=Models(spec,theta,directions)
        assert models.constant_observation_jacobian
        shared=build_maps(models,Jet(xv,xt),obs,controls)
        models.constant_observation_jacobian=False
        full=build_maps(models,Jet(xv,xt),obs,controls)
        return tuple((a.v,a.t,b.v,b.t) for a,b in zip(shared[:6],full[:6]))
    for av,at,bv,bt in compare(xv,xt):
        tf.debugging.assert_near(av,bv,atol=3e-8,rtol=3e-8)
        tf.debugging.assert_near(at,bt,atol=3e-8,rtol=3e-8)


@pytest.mark.parametrize("dimension", [3,18])
def test_centered_normalized_moments_large_offsets(dimension):
    from bayesfilter.highdim.analytical_jet_tf import normalized_moments
    p,b,n,d=2,2,32,dimension
    x=tf.random.stateless_normal([b,n,d],[83,d])+.0+1024.
    dx=tf.random.stateless_normal([p,b,n,d],[84,d])*.2+8192.
    w=tf.nn.softmax(tf.random.stateless_normal([b,n],[85,d]),axis=-1)
    dw=tf.random.stateless_normal([p,b,n],[86,d])*.01
    inputs=(x,dx,w,dw)
    @tf.function(input_signature=[tf.TensorSpec(v.shape,v.dtype) for v in inputs],
                 jit_compile=True,autograph=False)
    def actual(x,dx,w,dw):
        mean,cov=normalized_moments(Jet(x,dx),Jet(w,dw))
        return mean.v,mean.t,cov.v,cov.t
    mean,dmean,cov,dcov=actual(*inputs)
    x,dx,w,dw=[tf.cast(v,tf.float64) for v in inputs]
    def direct(xx,ww):
        ww=ww/tf.reduce_sum(ww,-1,keepdims=True)
        mu=tf.reduce_sum(ww[...,None]*xx,axis=-2)
        e=xx-mu[:,None]
        return mu,tf.linalg.matmul(ww[...,None]*e,e,transpose_a=True)
    mu,cc=direct(x,w)
    tf.debugging.assert_near(tf.cast(mean,tf.float64),mu,atol=1e-4,rtol=0.)
    tf.debugging.assert_near(tf.cast(cov,tf.float64),cc,atol=2e-5,rtol=2e-5)
    h=tf.constant(1e-4,tf.float64)
    for direction in range(p):
        plus=direct(x+h*dx[direction],w+h*dw[direction])
        minus=direct(x-h*dx[direction],w-h*dw[direction])
        tf.debugging.assert_near(tf.cast(dmean[direction],tf.float64),
                                (plus[0]-minus[0])/(2*h),atol=.002,rtol=0.)
        tf.debugging.assert_near(tf.cast(dcov[direction],tf.float64),
                                (plus[1]-minus[1])/(2*h),atol=.002,rtol=2e-5)

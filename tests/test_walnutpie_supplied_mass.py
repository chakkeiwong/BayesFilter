"""Engineering-only XLA metric invariants, no posterior or speed promotion."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import pytest


def inputs(tf):
    return (tf.constant([[.2,.4],[-.5,.3]],tf.float64),
        tf.constant([[17,0],[29,0]],tf.int32),
        tf.constant([[2.,3.],[4.,5.]],tf.float64),tf.constant([.1,.2],tf.float64))


@pytest.mark.parametrize('variant',['deterministic','r2p'])
@pytest.mark.parametrize('adapt',[False,True])
def test_supplied_diagonal_lifecycle(variant,adapt):
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import Config,make_supplied_mass_functions
    cfg=Config(2,2,warmup=4,draws=4,max_depth=3,max_halvings=3,variant=variant,
        r2p_max_exponent=3,adapt_mass=adapt)
    f=make_supplied_mass_functions(tf,lambda q:-.5*tf.reduce_sum(q*q,axis=-1),cfg,adapt_step=False)
    q,s,m,e=inputs(tf);out=f.fit(q,s,m,e)
    tf.debugging.assert_near(out.initial.inverse_mass,1/m)
    assert bool(tf.reduce_all(out.valid).numpy())
    if adapt:assert bool(tf.reduce_any(out.warmup.state.inverse_mass!=out.initial.inverse_mass).numpy())
    else:tf.debugging.assert_equal(out.warmup.state.inverse_mass,out.initial.inverse_mass)
    tf.debugging.assert_equal(out.sampling.state.inverse_mass,out.warmup.state.inverse_mass)
    tf.debugging.assert_equal(out.sampling.state.epsilon,out.warmup.state.epsilon)
    step=f.initialize(q,s,m,e);warm=f.warmup(step);sample=f.sample(warm.state)
    tf.debugging.assert_near(sample.draws,out.sampling.draws)
    tf.debugging.assert_equal(sample.state.seed,out.sampling.state.seed)
    f.initialize(q,s,m*2,e)
    assert f.initialize.experimental_get_tracing_count()==1


def test_mass_validation():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import Config,validate_mass,make_supplied_mass_functions
    cfg=Config(2,2);m=validate_mass(tf,[2.,3.],cfg)
    assert m.shape==(2,2)
    for bad in ([0.,1.],[-1.,2.],[float('nan'),2.],[1.,2.,3.]):
        with pytest.raises(ValueError):validate_mass(tf,bad,cfg)
    for bad in ([[1.,2.],[0.,1.]],[[1.,2.],[2.,1.]],[[float('nan'),0.],[0.,1.]]):
        with pytest.raises(ValueError):validate_mass(tf,bad,cfg,dense=True)
    f=make_supplied_mass_functions(tf,lambda q:-.5*tf.reduce_sum(q*q,-1),cfg,adapt_step=False)
    q,s,m,e=inputs(tf)
    out=f.initialize(q,s,tf.constant([[0.,1.],[-1.,2.]],tf.float64),e)
    assert not bool(tf.reduce_any(out.valid).numpy())


@pytest.mark.parametrize('variant',['deterministic','r2p'])
def test_dense_identity_and_coordinate_algebra(variant):
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import Config,make_supplied_mass_functions,make_dense_mass_functions,validate_mass
    cfg=Config(2,2,warmup=3,draws=4,max_depth=3,max_halvings=3,variant=variant,r2p_max_exponent=3,adapt_mass=False)
    model=lambda q:-.5*tf.reduce_sum(q*q,-1)
    d=make_dense_mass_functions(tf,model,cfg,adapt_step=False)
    q,s,m,e=inputs(tf);identity=tf.eye(2,batch_shape=[2],dtype=tf.float64)
    dense=d.fit(q,s,identity,e)
    diagonal=make_supplied_mass_functions(tf,model,cfg,adapt_step=False).fit(q,s,tf.ones_like(m),e)
    tf.debugging.assert_near(dense.sampling.draws,diagonal.sampling.draws,atol=2e-11,rtol=2e-11)
    tf.debugging.assert_equal(dense.sampling.state.sampler.seed,diagonal.sampling.state.seed)
    M=validate_mass(tf,[[4.,1.],[1.,2.]],cfg,dense=True)
    start=d.initialize(q,s,M,e);L=start.mass_cholesky
    tf.debugging.assert_near(d.physical_positions(start),q)
    tf.debugging.assert_near(d.physical_mass(start),M)
    tf.debugging.assert_near(start.sampler.lp+tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)),-1),model(q))
    expected=tf.linalg.triangular_solve(L,(-q)[...,None],lower=True)[...,0]
    tf.debugging.assert_near(start.sampler.g,expected)
    full=d.fit(q,s,M,e);warm=d.warmup(start);sample=d.sample(warm.state)
    tf.debugging.assert_near(full.sampling.draws,sample.draws)
    tf.debugging.assert_near(d.physical_mass(sample.state),M)
    tf.debugging.assert_near(sample.log_density,tf.reshape(model(tf.reshape(sample.draws,[-1,2])),[4,2]))
    assert bool(tf.reduce_all(full.valid).numpy())
    assert d.initialize.experimental_get_tracing_count()==1


def test_dense_adaptation_and_invalid_input():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import Config,make_dense_mass_functions,validate_mass
    cfg=Config(2,2,warmup=4,draws=4,max_depth=3,max_halvings=3,adapt_mass=True)
    f=make_dense_mass_functions(tf,lambda q:-.5*tf.reduce_sum(q*q,-1),cfg,adapt_step=False)
    q,s,m,e=inputs(tf);M=validate_mass(tf,[[4.,1.],[1.,2.]],cfg,dense=True)
    out=f.fit(q,s,M,e)
    assert bool(tf.reduce_all(out.valid).numpy())
    assert bool(tf.reduce_any(tf.abs(f.physical_mass(out.warmup.state)-M)>1e-8).numpy())
    tf.debugging.assert_equal(f.physical_mass(out.sampling.state),f.physical_mass(out.warmup.state))
    bad=f.initialize(q,s,tf.constant([[[1.,0.],[2.,1.]],[[1.,0.],[0.,-1.]]],tf.float64),e)
    assert not bool(tf.reduce_any(bad.valid).numpy())


def test_physical_uturn_negative_control():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.mass_coordinates import physical_uturn
    # With L=diag(2,1), r=(2,-1), delta_y=(1,1), naive dot=1, physical=-.5.
    L=tf.constant([[[2.,0.],[0.,1.]]],tf.float64)
    zero=tf.zeros([1,2],tf.float64);delta=tf.ones([1,2],tf.float64)
    r=tf.constant([[2.,-1.]],tf.float64);inv=tf.ones([1,2],tf.float64)
    assert bool(physical_uturn(tf,zero,r,delta,r,inv,L)[0].numpy())
    assert float(tf.reduce_sum(r*delta).numpy())>0
    M=tf.matmul(L,L,transpose_b=True);p=tf.linalg.matvec(L,r)
    dq=tf.linalg.triangular_solve(L,delta[...,None],lower=True,adjoint=True)[...,0]
    direct=tf.reduce_sum(p*tf.linalg.solve(M,dq[...,None])[...,0],-1)<0
    tf.debugging.assert_equal(physical_uturn(tf,zero,r,delta,r,inv,L),direct)


def test_dense_leapfrog_matches_direct_physical_hamiltonian():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import Config
    from docs.benchmarks.walnutpie_gpu_tf.core import Kernel
    M=tf.constant([[[4.,1.],[1.,2.]]],tf.float64);L=tf.linalg.cholesky(M)
    q=tf.constant([[.2,-.4]],tf.float64);p=tf.constant([[.3,.7]],tf.float64)
    h=tf.constant([.07],tf.float64)
    def model(y):
        x=tf.linalg.triangular_solve(L,y[...,None],lower=True,adjoint=True)[...,0]
        return -.5*tf.reduce_sum(x*x,-1)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)),-1)
    k=Kernel(tf,model,Config(1,2))
    @tf.function(input_signature=[tf.TensorSpec([1,2],tf.float64),tf.TensorSpec([1,2],tf.float64)],jit_compile=True,autograph=False)
    def one(q,p):
        y=tf.linalg.matvec(L,q,transpose_a=True)
        r=tf.linalg.triangular_solve(L,p[...,None],lower=True)[...,0]
        lp,g=k.value_gradient(y);point=k.point(y,r,g,lp,tf.ones_like(q))
        step=k.leapfrogs(point,tf.ones_like(q),h,tf.ones([1],tf.int32),tf.ones([1],tf.bool))
        return (tf.linalg.triangular_solve(L,step.q[...,None],lower=True,adjoint=True)[...,0],
                tf.linalg.matvec(L,step.p))
    got_q,got_p=one(q,p)
    half=p-.5*h[:,None]*q
    expected_q=q+h[:,None]*tf.linalg.solve(M,half[...,None])[...,0]
    expected_p=half-.5*h[:,None]*expected_q
    tf.debugging.assert_near(got_q,expected_q,atol=2e-12,rtol=2e-12)
    tf.debugging.assert_near(got_p,expected_p,atol=2e-12,rtol=2e-12)


def test_supplied_metric_with_runtime_model_data():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import Config,make_supplied_mass_functions,make_dense_mass_functions
    cfg=Config(2,2,warmup=0,draws=2,max_depth=2,max_halvings=2,adapt_mass=False)
    model=lambda q,precision:-.5*tf.reduce_sum(q*q*precision,axis=-1)
    signature=(tf.TensorSpec([2],tf.float64),)
    q,s,m,e=inputs(tf);precision=tf.constant([1.,2.],tf.float64)
    for dense in (False,True):
        f=(make_dense_mass_functions if dense else make_supplied_mass_functions)(tf,model,cfg,signature,adapt_step=False)
        mass=tf.linalg.diag(m) if dense else m
        first=f.fit(q,s,mass,e,precision);second=f.fit(q,s,mass,e,precision*2)
        assert bool(tf.reduce_all(first.valid & second.valid).numpy())
        assert f.fit.experimental_get_tracing_count()==1


@pytest.mark.parametrize('variant',['deterministic','r2p'])
def test_dense_diagonal_matches_direct_diagonal_mass(variant):
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import Config,make_supplied_mass_functions,make_dense_mass_functions
    cfg=Config(2,2,warmup=2,draws=4,max_depth=3,max_halvings=3,variant=variant,r2p_max_exponent=3,adapt_mass=False)
    model=lambda q:-.5*tf.reduce_sum(q*q,axis=-1)
    q,s,m,e=inputs(tf)
    direct=make_supplied_mass_functions(tf,model,cfg,adapt_step=False).fit(q,s,m,e)
    dense=make_dense_mass_functions(tf,model,cfg,adapt_step=False).fit(q,s,tf.linalg.diag(m),e)
    tf.debugging.assert_near(direct.sampling.draws,dense.sampling.draws,atol=2e-11,rtol=2e-11)
    tf.debugging.assert_equal(direct.sampling.depth,dense.sampling.depth)
    tf.debugging.assert_equal(direct.sampling.state.seed,dense.sampling.state.sampler.seed)

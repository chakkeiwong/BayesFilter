"""Diagnostic/reference tests for standalone Walnutpie gap closure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import pytest
from docs.benchmarks.walnutpie_gpu_tf import Config
from docs.benchmarks.walnutpie_gpu_tf.initialization import Initialization,make_initializers
from docs.benchmarks.walnutpie_gpu_tf.control import RunControl,make_controllers,drive_chunks


def gaussian(tf):return lambda q:-.5*tf.reduce_sum(q*q,-1)


def test_supplied_initialization_and_uncapped_search():
    import tensorflow as tf
    q=tf.constant([[.2,.4],[-.5,.3]],tf.float64);s=tf.constant([[17,0],[29,0]],tf.int32)
    mass=tf.constant([[2.,3.],[4.,5.]],tf.float64);e=tf.constant([.1,.2],tf.float64)
    cfg=Config(2,2,init_search_limit=None)
    f=make_initializers(tf,gaussian(tf),cfg,Initialization(mass='supplied',adapt_step=False))
    a=f.initialize(q,s,mass,e)
    tf.debugging.assert_equal(a.q,q);tf.debugging.assert_equal(a.epsilon,e)
    tf.debugging.assert_near(a.inverse_mass,1/mass)
    g=make_initializers(tf,gaussian(tf),cfg,Initialization(mass='supplied'))
    b=g.initialize(q,s,mass,e)
    assert bool(tf.reduce_all(b.valid).numpy())
    tf.debugging.assert_equal(b.seed[:,1],[1,1])


def test_random_starts_and_geometric_mass():
    import tensorflow as tf
    cfg=Config(2,2);model=gaussian(tf)
    s=tf.constant([[17,0],[29,0]],tf.int32);q=tf.zeros([2,2],tf.float64);e=tf.ones([2],tf.float64)*.1
    f=make_initializers(tf,model,cfg,Initialization(positions='normal',scale=2.,average_masses=True,adapt_step=False))
    expected,seeds=f.random_positions(s);r=f.initialize(q,s,q,e)
    tf.debugging.assert_equal(expected,r.q);tf.debugging.assert_equal(seeds,r.seed)
    mass=.99999*tf.abs(expected)+.00001
    avg=tf.exp(tf.reduce_mean(tf.math.log(mass),0))
    tf.debugging.assert_near(r.inverse_mass,tf.broadcast_to(1/avg,[2,2]))
    assert bool(tf.reduce_all(r.valid).numpy())
    bad=f.initialize(q,tf.constant([[17,0],[17,0]],tf.int32),q,e)
    assert not bool(tf.reduce_any(bad.valid).numpy())


def test_control_chunk_continuation_and_freeze():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    cfg=Config(2,2,warmup=0,draws=0)
    p=RunControl(warmup_min=4,warmup_max=4,sample_min=6,sample_max=6,chunk_size=2)
    f=make_controllers(tf,gaussian(tf),cfg,p)
    s=make_functions(tf,gaussian(tf),cfg).initialize(tf.constant([[.2,.3],[-.3,.4]],tf.float64),tf.constant([[17,0],[29,0]],tf.int32))
    start=f.begin(s);full=f.warmup(start);a=f.warmup_chunk(start);b=f.warmup_chunk(a.state)
    tf.debugging.assert_near(full.draws,tf.concat([a.draws,b.draws],0))
    tf.debugging.assert_equal(full.state.sampler.seed,b.state.sampler.seed)
    retained_start=f.begin(full.state.sampler);fullsample=f.sample(retained_start)
    driven=drive_chunks(f.sample_chunk,retained_start)
    assert len(driven['chunks'])==3
    tf.debugging.assert_near(fullsample.draws,tf.concat([x.draws for x in driven['chunks']],0))
    tf.debugging.assert_equal(fullsample.state.sampler.inverse_mass,full.state.sampler.inverse_mass)
    tf.debugging.assert_equal(fullsample.state.sampler.adam.theta,full.state.sampler.adam.theta)
    assert drive_chunks(f.sample_chunk,retained_start,deadline_seconds=0)['reason']=='deadline'
    assert drive_chunks(f.sample_chunk,retained_start,cancelled=lambda:True)['reason']=='cancelled'


def test_legacy_logp_diagnostic_is_not_parameter_convergence():
    import tensorflow as tf
    f=make_controllers(tf,gaussian(tf),Config(2,2))
    # Identical logp summaries can coexist with chains in separated symmetric modes.
    r=f.sampling_statistic(tf.constant([-2.,-2.],tf.float64),tf.constant([4.,4.],tf.float64),tf.constant(10))
    assert float(r.numpy())==1.


@pytest.mark.parametrize('layout',['fieldwise','packed','flat'])
def test_r2p_fit_compiles_and_continues(layout):
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    cfg=Config(2,2,warmup=2,draws=3,max_depth=3,variant='r2p',r2p_max_exponent=3,execution_layout=layout)
    model=gaussian(tf);f=make_functions(tf,model,cfg)
    q=tf.constant([[.2,.3],[-.3,.4]],tf.float64);s=tf.constant([[17,0],[29,0]],tf.int32)
    r=f.fit(q,s);assert bool(tf.reduce_all(r.valid).numpy())
    init=f.initialize(q,s);warm=f.warmup(init);sample=f.sample(warm.state)
    tf.debugging.assert_near(r.sampling.draws,sample.draws)
    lp,g=f.value_gradient(sample.state.q)
    tf.debugging.assert_near(lp,sample.state.lp);tf.debugging.assert_near(g,sample.state.g)
    tf.debugging.assert_equal(sample.state.minimum,[1,1])


def test_r2p_probability_weight_recurrence_negative_control():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.r2p import R2PKernel,MacroResult
    class KnownEdge(R2PKernel):
        def r2p_macro(self,start,inv,h,active,seed,adam,adapt=False):
            p=self.point(start.q+1,start.p,start.g,start.lp,inv)
            return MacroResult(p,self.ints(),self.ints(),self.ints(),self.f(-.7)+self.zeros(),self.ints(),self.ints(),self.active(),self.active(),seed,adam)
    k=KnownEdge(tf,gaussian(tf),Config(2,2,variant='r2p',r2p_jitter=0))
    q=tf.constant([[.2,.3],[-.3,.4]],tf.float64);inv=tf.ones_like(q);lp,g=k.value_gradient(q)
    start=k.span(k.point(q,tf.ones_like(q),g,lp,inv))
    seed=tf.constant([[17,0],[29,0]],tf.int32);adam=k.adam_init(tf.ones([2],tf.float64))
    for direction in (True,False):
        forward=tf.fill([2],direction)
        one,*_=k.grow_leaf(start,forward,inv,tf.ones([2],tf.float64),tf.ones([2],tf.int32),k.active(),seed,adam,False)
        two,*_=k.grow_leaf(one,forward,inv,tf.ones([2],tf.float64),tf.ones([2],tf.int32),k.active(),seed,adam,False)
        tf.debugging.assert_near(two.weight,two.fj-1.4)
        assert float(tf.reduce_max(tf.abs(two.weight-(two.fj-.7))).numpy())>.6


def test_new_runtime_has_no_numpy_pfor_or_callbacks():
    import ast
    from pathlib import Path
    base=Path(__file__).resolve().parents[1]/'docs/benchmarks/walnutpie_gpu_tf'
    for name in ('r2p.py','initialization.py','control.py'):
        for node in ast.walk(ast.parse((base/name).read_text())):
            if isinstance(node,ast.Import):assert all(x.name!='numpy' for x in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module!='numpy'
            if isinstance(node,ast.Attribute):assert node.attr not in ('py_function','numpy_function','vectorized_map','jacobian','batch_jacobian')


def test_initialization_cap_failure_and_supplied_invalid_mass():
    import tensorflow as tf
    q=tf.constant([[.2,.3],[-.3,.4]],tf.float64);seed=tf.constant([[17,0],[29,0]],tf.int32)
    model=gaussian(tf)
    f=make_initializers(tf,model,Config(2,2,init_search_limit=1),Initialization(mass='unit'))
    r=f.initialize(q,seed,tf.ones_like(q),tf.constant([1e-10,1e-10],tf.float64))
    assert not bool(tf.reduce_any(r.valid).numpy())
    g=make_initializers(tf,model,Config(2,2),Initialization(mass='supplied',adapt_step=False))
    bad=g.initialize(q,seed,-tf.ones_like(q),tf.ones([2],tf.float64))
    assert not bool(tf.reduce_any(bad.valid).numpy())


def test_control_invalid_initial_stops_without_draws():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    cfg=Config(2,2,warmup=0,draws=0);model=gaussian(tf)
    init=make_functions(tf,model,cfg).initialize(tf.ones([2,2],tf.float64),tf.constant([[17,0],[29,0]],tf.int32))
    ctrl=make_controllers(tf,model,cfg,RunControl(warmup_max=50,sample_max=50))
    state=ctrl.begin(init._replace(valid=tf.constant([False,True])))
    r=ctrl.sample_chunk(state)
    assert int(r.state.reason.numpy())==3 and int(r.state.count.numpy())==0
    assert not bool(tf.reduce_any(r.mask).numpy())


def test_r2p_nonfinite_proposal_preserves_retained_cache():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.r2p import R2PKernel
    cfg=Config(2,1,variant='r2p',r2p_max_exponent=0,r2p_jitter=0,max_depth=2)
    def model(q):return tf.where(tf.abs(q[:,0])<1.,-.5*q[:,0]**2,tf.constant(float('-inf'),tf.float64))
    k=R2PKernel(tf,model,cfg,tape=True)
    @tf.function(input_signature=[tf.TensorSpec([2,1],tf.float64)],jit_compile=True,autograph=False)
    def step(q):
        lp,g=k.value_gradient(q);eps=tf.fill([2],tf.constant(100.,tf.float64))
        return k.transition(q,lp,g,eps,tf.ones_like(q),tf.ones([2],tf.int32),tf.constant([[17,0],[29,0]],tf.int32),k.adam_init(eps))
    q=tf.constant([[.2],[-.3]],tf.float64);r=step(q)
    tf.debugging.assert_equal(r.q,q);tf.debugging.assert_near(r.lp,model(q));tf.debugging.assert_near(r.g,-q)


def test_r2p_finer_branch_allowed_beyond_search_cap():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.r2p import R2PKernel
    cfg=Config(2,2,variant='r2p',r2p_max_exponent=0,r2p_jitter=0)
    k=R2PKernel(tf,gaussian(tf),cfg,tape=True)
    @tf.function(input_signature=[tf.TensorSpec([2,2],tf.float64)],jit_compile=True,autograph=False)
    def call(q):
        lp,g=k.value_gradient(q);h=tf.fill([2],tf.constant(.1,tf.float64))
        return k.r2p_macro(k.point(q,tf.ones_like(q),g,lp,tf.ones_like(q)),tf.ones_like(q),h,k.active(),tf.constant([[1,0],[48,0]],tf.int32),k.adam_init(h))
    r=call(tf.constant([[.2,.3],[-.3,.4]],tf.float64))
    tf.debugging.assert_equal(r.forward,[0,0]);tf.debugging.assert_equal(r.exponent,[0,1])
    assert bool(tf.reduce_all(r.finite).numpy())


def test_uncapped_search_rejects_invalid_steps_and_metrics():
    import tensorflow as tf
    cfg=Config(4,2,init_search_limit=None)
    f=make_initializers(tf,gaussian(tf),cfg,Initialization(mass='supplied'))
    q=tf.ones([4,2],tf.float64);s=tf.constant([[17,0],[29,0],[43,0],[59,0]],tf.int32)
    steps=tf.constant([0.,-.1,float('inf'),float('nan')],tf.float64)
    r=f.initialize(q,s,tf.ones_like(q),steps)
    tf.debugging.assert_equal(r.valid,[False]*4)
    tf.debugging.assert_equal(r.seed,s)
    mass=tf.constant([[0.,1.],[-1.,1.],[float('inf'),1.],[float('nan'),1.]],tf.float64)
    r=f.initialize(q,s,mass,tf.ones([4],tf.float64))
    tf.debugging.assert_equal(r.valid,[False]*4)
    tf.debugging.assert_equal(r.seed,s)


def test_zero_unit_initialization_and_r2p_minimum():
    import tensorflow as tf
    cfg=Config(2,2,variant='r2p',target_macro_steps=.5)
    f=make_initializers(tf,gaussian(tf),cfg,Initialization(positions='zero',mass='unit',adapt_step=False))
    r=f.initialize(tf.ones([2,2],tf.float64),tf.constant([[17,0],[29,0]],tf.int32),
        tf.zeros([2,2],tf.float64),tf.constant([.1,.2],tf.float64))
    tf.debugging.assert_equal(r.q,tf.zeros_like(r.q))
    tf.debugging.assert_equal(r.inverse_mass,tf.ones_like(r.q))
    tf.debugging.assert_equal(r.minimum,[1,1])
    assert bool(tf.reduce_all(r.valid).numpy())


def test_initializer_rejects_overflow_inverse_mass_and_bad_random_key():
    import tensorflow as tf
    cfg=Config(2,2);q=tf.ones([2,2],tf.float64);s=tf.constant([[17,0],[29,0]],tf.int32)
    f=make_initializers(tf,gaussian(tf),cfg,Initialization(mass='supplied',adapt_step=False))
    r=f.initialize(q,s,tf.fill([2,2],tf.constant(1e-320,tf.float64)),tf.ones([2],tf.float64))
    tf.debugging.assert_equal(r.valid,[False,False])
    g=make_initializers(tf,gaussian(tf),cfg,Initialization(positions='normal',mass='unit',adapt_step=False))
    r=g.initialize(q,tf.constant([[17,-1],[29,0]],tf.int32),q,tf.ones([2],tf.float64))
    tf.debugging.assert_equal(r.valid,[False,True])


def test_control_zero_warmup_has_no_draws():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    cfg=Config(2,2);model=gaussian(tf)
    s=make_functions(tf,model,cfg).initialize(tf.ones([2,2],tf.float64),tf.constant([[17,0],[29,0]],tf.int32))
    ctrl=make_controllers(tf,model,cfg,RunControl(warmup_min=0,warmup_max=0))
    r=ctrl.warmup(ctrl.begin(s))
    assert tuple(r.draws.shape)==(0,2,2) and int(r.state.count.numpy())==0
    assert int(r.state.reason.numpy())==2
    tf.debugging.assert_equal(r.state.sampler.seed,s.seed)
    immediate=make_controllers(tf,model,cfg,RunControl(warmup_min=0,step_tolerance=1e10,mass_tolerance=1e10))
    r=immediate.warmup_chunk(immediate.begin(s))
    assert int(r.state.reason.numpy())==1 and int(r.state.count.numpy())==0
    assert not bool(tf.reduce_any(r.mask).numpy())


def test_controller_terminal_checkpoint_and_cancel_resume():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    cfg=Config(2,2,max_depth=2);model=gaussian(tf)
    init=make_functions(tf,model,cfg).initialize(tf.constant([[.2,.3],[-.3,.4]],tf.float64),tf.constant([[17,0],[29,0]],tf.int32))
    ctrl=make_controllers(tf,model,cfg,RunControl(sample_min=2,sample_max=4,chunk_size=2,logp_rhat_tolerance=0.))
    start=ctrl.begin(init);seen=[]
    partial=drive_chunks(ctrl.sample_chunk,start,cancelled=lambda:bool(seen),on_chunk=lambda r,n:seen.append(n))
    assert partial['reason']=='cancelled' and seen==[2]
    resumed=drive_chunks(ctrl.sample_chunk,partial['state'])
    assert resumed['reason']=='maximum' and int(resumed['state'].count.numpy())==4
    full=ctrl.sample(start)
    tf.debugging.assert_near(tf.concat([partial['chunks'][0].draws,resumed['chunks'][0].draws],0),full.draws)
    exhausted=ctrl.sample_chunk(start._replace(count=tf.constant(4)))
    assert int(exhausted.state.reason.numpy())==2
    tf.debugging.assert_equal(exhausted.mask,[False,False])
    malformed=ctrl.sample_chunk(start._replace(count=tf.constant(-1)))
    assert int(malformed.state.reason.numpy())==3
    with pytest.raises(RuntimeError,match='no progress'):
        drive_chunks(lambda s:full._replace(state=s),start)


@pytest.mark.parametrize('metropolis',[False,True])
def test_r2p_all_zero_weight_subtree_preserves_geometry(metropolis):
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.r2p import R2PKernel
    k=R2PKernel(tf,gaussian(tf),Config(2,2,variant='r2p'),tape=True)
    @tf.function(input_signature=[tf.TensorSpec([2,2],tf.float64)],jit_compile=True,autograph=False)
    def merge(q):
        lp,g=k.value_gradient(q);p=k.point(q,tf.ones_like(q),g,lp,tf.ones_like(q))
        empty=tf.fill([2],tf.constant(float('-inf'),tf.float64))
        old=k.span(p)._replace(weight=empty,bc=empty,fc=empty)
        new=k.span(p._replace(q=q+1))._replace(weight=empty,bc=empty,fc=empty)
        return k.combine(old,new,tf.constant([True,False]),tf.constant([[17,0],[29,0]],tf.int32),k.active(),metropolis)
    q=tf.constant([[.2,.3],[-.3,.4]],tf.float64);r,s=merge(q)
    tf.debugging.assert_equal(r.q,q);tf.debugging.assert_equal(s[:,1],[1,1])
    tf.debugging.assert_equal(r.bq,tf.stack([q[0],q[1]+1]))
    tf.debugging.assert_equal(r.fq,tf.stack([q[0]+1,q[1]]))
    assert bool(tf.reduce_all(tf.math.is_inf(r.weight)&(r.weight<0)).numpy())

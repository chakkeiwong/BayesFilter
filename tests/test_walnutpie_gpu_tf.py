"""Standalone Walnutpie numerical/API checks; no public sampler admission."""
import os
from pathlib import Path
import pytest
from docs.benchmarks.walnutpie_gpu_tf import Config

@pytest.mark.parametrize('kw',[{'chains':0},{'dimension':0},{'max_depth':21},
    {'max_halvings':0},{'max_error':0},{'mass_init_count':1},{'beta1':1},
    {'execution_layout':'bad'},{'draws':-1},{'warmup':1.5},{'initial_step':float('nan')}])
def test_configuration_rejects_invalid(kw):
    with pytest.raises(ValueError):Config(**({'chains':2,'dimension':4}|kw))


def test_pinned_defaults():
    c=Config(4,10)
    assert (c.max_depth,c.max_halvings,c.max_error,c.target_accept)==(5,5,.5,.8)
    assert (c.mass_init_count,c.target_macro_steps,c.beta1,c.beta2)==(4.,15.,.8,.9)


def test_no_host_callback_or_numpy_runtime():
    import ast
    root=Path(__file__).resolve().parents[1]/'docs/benchmarks/walnutpie_gpu_tf'
    for filename in ('core.py','api.py','optimization_worker.py'):
        tree=ast.parse((root/filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(x.name!='numpy' for x in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module!='numpy'
            if isinstance(node,ast.Attribute):assert node.attr not in ('numpy_function','py_function','vectorized_map','jacobian','batch_jacobian')


def test_compiled_zero_warmup_and_immutable_tuning():
    # Deliberate CPU reference; hide GPUs before importing the framework.
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    f=make_functions(tf,lambda q:-.5*tf.reduce_sum(q*q,-1),Config(2,2,warmup=0,draws=2))
    with tf.device('/CPU:0'):
        r=f.fit(tf.constant([[.2,.3],[-.3,.4]],tf.float64),tf.constant([[1,0],[2,0]],tf.int32))
        tf.debugging.assert_equal(r.initial.epsilon,r.sampling.state.epsilon)
        tf.debugging.assert_equal(r.initial.inverse_mass,r.sampling.state.inverse_mass)
        tf.debugging.assert_equal(r.initial.adam,r.sampling.state.adam)
        tf.debugging.assert_equal(r.valid,[True,True])
        assert r.warmup.draws.shape==(0,2,2)
        assert f.fit.experimental_get_tracing_count()==1


def test_diagnostic_axes_use_chain_major_authority():
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.worker import diagnostics
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat
    x=tf.reshape(tf.sin(tf.cast(tf.range(240),tf.float64)),[30,4,2])
    x+=tf.constant([0.,3.,6.,9.],tf.float64)[None,:,None]
    got=diagnostics(tf,x)
    expected=rank_normalized_split_rhat(tf.transpose(x,[1,0,2]))
    assert got['max_rhat']==float(tf.reduce_max(expected['maximum']).numpy())
    assert got['max_rhat']>1.1


def test_initial_step_repairs_negative_infinite_proposal():
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.core import Kernel
    # Outside-domain proposals have -inf log density; finite current state.
    cfg=Config(1,1,initial_step=10.)
    k=Kernel(tf,lambda q:tf.where(tf.abs(q[:,0])<2.,-.5*q[:,0]**2,
        tf.constant(float('-inf'),tf.float64)),cfg,tape=True)
    @tf.function(input_signature=[tf.TensorSpec([1,1],tf.float64)],jit_compile=True,autograph=False)
    def initialize(q):
        lp,g=k.value_gradient(q)
        return k.initial_step(q,lp,g,tf.ones_like(q),tf.constant([[1,0]],tf.int32))
    eps,seed,valid,count=initialize(tf.constant([[.3]],tf.float64))
    assert bool(valid.numpy()[0]);assert float(eps.numpy()[0])<10.
    assert int(count.numpy()[0])>0


@pytest.mark.parametrize('layout',['fieldwise','packed','flat'])
@pytest.mark.parametrize('dimension',[1,4])
def test_tree_stack_roundtrip(layout,dimension):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.core import Kernel,Span
    k=Kernel(tf,lambda q:-.5*tf.reduce_sum(q*q,-1),Config(2,dimension,execution_layout=layout))
    widths=[dimension,dimension,dimension,1,dimension,dimension,dimension,1,dimension,dimension,1,1]
    span=Span(*(tf.fill([2] if i in (3,7,10,11) else [2,w],tf.constant(i+.25,tf.float64)) for i,w in enumerate(widths)))
    @tf.function(input_signature=[],jit_compile=True,autograph=False)
    def roundtrip():
        stack=k.stack_write(k.stack_init(span),tf.constant(2),span)
        return k.stack_read(stack,tf.constant(2)),k.stack_read(stack,tf.constant(1))
    got,untouched=roundtrip()
    for a,b,z in zip(span,got,untouched):
        tf.debugging.assert_equal(a,b);tf.debugging.assert_equal(z,tf.zeros_like(z))


@pytest.mark.parametrize('layout',['packed','flat'])
def test_execution_layout_preserves_full_fit(layout):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    from dataclasses import replace
    cfg=Config(4,4,warmup=8,draws=8,target_macro_steps=1.)
    model=lambda q:-q[:,0]**2/18.-1.5*q[:,0]-.5*tf.exp(-q[:,0])*tf.reduce_sum(q[:,1:]**2,-1)
    q=tf.constant([[-2.,.1,-.2,.3],[.1,.2,-.4,.5],[2.,1.,-1.,.3],[-.4,-.1,.2,.5]],tf.float64)
    seeds=tf.constant([[71,0],[72,0],[73,0],[74,0]],tf.int32)
    base=make_functions(tf,model,cfg).fit(q,seeds)
    candidate=make_functions(tf,model,replace(cfg,execution_layout=layout)).fit(q,seeds)
    for a,b in zip(tf.nest.flatten(base),tf.nest.flatten(candidate)):
        if a.dtype.is_floating:tf.debugging.assert_near(a,b,rtol=2e-11,atol=2e-11)
        else:tf.debugging.assert_equal(a,b)


def test_timing_interval_uses_seed_blocks():
    from docs.benchmarks.walnutpie_gpu_tf.optimization_worker import ratios_summary
    result=ratios_summary([[(2.,1.)]*4 for _ in range(7)])
    assert result['geometric_speed_ratio']==pytest.approx(2.)
    assert result['interval95']==pytest.approx([2.,2.])


def test_fieldwise_remains_default():
    assert Config(4,10).execution_layout=='fieldwise'


def test_sufficient_statistics_preserve_nonzero_response_target():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf import make_functions
    from docs.benchmarks.walnutpie_gpu_tf.worker import model_for
    from docs.benchmarks.walnutpie_gpu_tf.optimization_worker import sufficient_model
    x=tf.constant([[1.,2.],[-1.,.3],[.4,-.7]],tf.float64)
    y=tf.constant([.6,-.8,.2],tf.float64)
    q=tf.constant([[.2,-.1],[-.4,.7]],tf.float64)
    raw=make_functions(tf,model_for(tf,3),Config(2,2),[tf.TensorSpec([3,2],tf.float64),tf.TensorSpec([3],tf.float64)])
    gram=tf.matmul(x,x,transpose_a=True)/3.
    linear=tf.linalg.matvec(x,y,transpose_a=True)/3.
    constant=tf.reduce_sum(y*y)/3.
    data=(gram,linear,constant)
    reduced=make_functions(tf,lambda q,*d:sufficient_model(tf,q,*d),Config(2,2),[tf.TensorSpec(v.shape,v.dtype) for v in data])
    for a,b in zip(raw.value_gradient(q,x,y),reduced.value_gradient(q,*data)):
        tf.debugging.assert_near(a,b,rtol=2e-11,atol=2e-11)


def test_benchmark_honors_requested_execution_layout(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from docs.benchmarks.walnutpie_gpu_tf import worker
    class CapturedConfiguration(Exception):pass
    monkeypatch.setattr(worker,'fixture',lambda *args:(0,2,0,None,None,(),()))
    def capture(tf,model,config,specs):
        assert config.execution_layout=='packed'
        assert config.warmup==3 and config.draws==4
        raise CapturedConfiguration
    monkeypatch.setattr(worker,'make_functions',capture)
    args=SimpleNamespace(targets='gaussian',warmup=3,draws=4,execution_layout='packed')
    with pytest.raises(CapturedConfiguration):worker.benchmark(None,args,{})

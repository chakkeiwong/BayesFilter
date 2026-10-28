"""Standalone Walnutpie numerical/API checks; no public sampler admission."""
import os
from pathlib import Path
import pytest
from docs.benchmarks.walnutpie_gpu_tf import Config

@pytest.mark.parametrize('kw',[{'chains':0},{'dimension':0},{'max_depth':21},
    {'max_halvings':0},{'max_error':0},{'mass_init_count':1},{'beta1':1},
    {'draws':-1},{'warmup':1.5},{'initial_step':float('nan')}])
def test_configuration_rejects_invalid(kw):
    with pytest.raises(ValueError):Config(**({'chains':2,'dimension':4}|kw))


def test_pinned_defaults():
    c=Config(4,10)
    assert (c.max_depth,c.max_halvings,c.max_error,c.target_accept)==(5,5,.5,.8)
    assert (c.mass_init_count,c.target_macro_steps,c.beta1,c.beta2)==(4.,15.,.8,.9)


def test_no_host_callback_or_numpy_runtime():
    import ast
    root=Path(__file__).resolve().parents[1]/'docs/benchmarks/walnutpie_gpu_tf'
    for filename in ('core.py','api.py'):
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

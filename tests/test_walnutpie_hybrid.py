"""CPU-only engineering replay for separately compiled hybrid components."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import pytest


@pytest.mark.parametrize('variant',['deterministic','r2p'])
def test_hybrid_tree_replays_shared_cpu_kernel(variant):
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.core import Config
    from docs.benchmarks.walnutpie_gpu_tf.api import make_kernel,initial_state,run,state_spec
    from docs.benchmarks.walnutpie_gpu_tf.hybrid import make_hybrid
    cfg=Config(4,3,warmup=0,max_depth=4,max_halvings=3,r2p_max_exponent=3,variant=variant,max_error=.05)
    def model(q):
        x=q[:,0];rest=q[:,1:]-.2*x[:,None]**2
        return -.5*x*x-.5*tf.reduce_sum(rest*rest,-1)
    k=make_kernel(tf,model,cfg)
    q=tf.constant([[.1,.2,-.3],[.2,-.5,.1],[1.,-.1,.4],[-.7,.2,.3]],tf.float64)
    h=make_hybrid(tf,model,cfg,leaf_device='/CPU:0')
    @tf.function(input_signature=[state_spec(tf,cfg)],jit_compile=True,autograph=False)
    def full(s):return run(k,s,2,False)
    for key in (29,47,83):
        state=initial_state(k,q,tf.constant([[key+i,0] for i in range(4)],tf.int32),
            mass=tf.constant([[.7,1.2,2.]]*4,tf.float64),epsilon=tf.constant([.1,.3,.5,.7],tf.float64),adapt_step=False)
        expected=full(state);actual=h.sample(state,2)
        for a,b in zip(tf.nest.flatten(expected),tf.nest.flatten(actual)):
            if a.dtype.is_floating:tf.debugging.assert_near(a,b,rtol=2e-11,atol=2e-11)
            else:tf.debugging.assert_equal(a,b)
        assert h.last_round_trips>0
    assert all(f.experimental_get_tracing_count()==1 for f in h.functions.values())

"""Diagnostic-only source and XLA qualification for the author funnel adapter."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import math
import numpy as np


def test_copied_source_hashes_and_unique_weight_patch():
    from docs.benchmarks.walnutpie_gpu_tf.funnel_reference import check_sources,load_author
    assert len(check_sources()['sha256'])==9
    original,*_=load_author(False)
    corrected,ai,td,diff=load_author(True)
    assert diff.count('+                            lwtSumb += intOut.lwt')==1
    assert original.WALNUTS is not corrected.WALNUTS


def test_normalized_target_gradient_and_dimension():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.funnel_reference import load_author,target
    from docs.benchmarks.walnutpie_gpu_tf.worker import model_for
    _,_,td,_=load_author()
    for d in (10,11):
        @tf.function(input_signature=[tf.TensorSpec([4,d],tf.float64)],jit_compile=True,autograph=False)
        def vg(q):
            with tf.GradientTape() as tape:
                tape.watch(q);lp=model_for(tf,2)(q);total=tf.reduce_sum(lp)
            return lp,tape.gradient(total,q)
        q=np.array([[v]+[math.exp(v/2)*(.1*j-.3) for j in range(1,d)] for v in (-30.,-6.,0.,8.)])
        lp,g=vg(tf.constant(q));constant=-.5*d*math.log(2*math.pi)-math.log(3.)
        for i in range(4):
            expected,grad=target(d)(q[i])
            np.testing.assert_allclose(lp.numpy()[i]+constant,expected,atol=2e-11,rtol=2e-11)
            np.testing.assert_allclose(g.numpy()[i],grad,atol=2e-11,rtol=2e-11)
            if d==11:
                al,ag=td.funnel10(q[i]);np.testing.assert_allclose([al],[expected],rtol=2e-12,atol=2e-12)
                np.testing.assert_allclose(ag,grad,rtol=2e-12,atol=2e-12)


def test_literal_author_start_and_fixed_kernel_continuation():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.funnel_reference import start
    from docs.benchmarks.walnutpie_gpu_tf.funnel_xla import functions
    assert abs(start(0,11)[0]-8.292157037902992)<1e-12
    q=tf.constant([start(s,11) for s in range(4)],tf.float64)
    seed=tf.constant([[100+s,0] for s in range(4)],tf.int32)
    init,small,_=functions(tf,11,.3,.3,2)
    _,large,_=functions(tf,11,.3,.3,4)
    state=init(q,seed);a=small(state);b=small(a.state);c=large(state)
    tf.debugging.assert_near(tf.concat([a.draws,b.draws],0),c.draws,atol=2e-11,rtol=2e-11)
    tf.debugging.assert_equal(b.state.seed,c.state.seed)
    tf.debugging.assert_equal(b.state.inverse_mass,tf.ones_like(q))
    tf.debugging.assert_equal(b.state.epsilon,state.epsilon)
    assert bool(tf.reduce_all(b.state.valid).numpy())
    assert small.experimental_get_tracing_count()==1


def test_author_original_and_corrected_smoke():
    from docs.benchmarks.walnutpie_gpu_tf.funnel_reference import load_author,target
    for corrected in (False,True):
        wn,ai,_,_=load_author(corrected);np.random.seed(5)
        q=np.zeros(11)
        samples,d=wn.WALNUTS(target(11),q,integrator=ai.adaptLeapFrogR2P,
            H0=.3,delta0=.3,M=4,numIter=8,warmupIter=0)
        assert samples.shape==(11,9) and d.shape==(8,24)
        assert np.all(np.isfinite(samples))


def test_funnel_dimensions_against_author_macro_paper_tree():
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.core import Config
    from docs.benchmarks.walnutpie_gpu_tf.r2p import R2PKernel
    from docs.benchmarks.walnutpie_gpu_tf.r2p_reference import transition
    from docs.benchmarks.walnutpie_gpu_tf.funnel_reference import load_author
    from docs.benchmarks.walnutpie_gpu_tf.worker import model_for
    _,author,_,_=load_author()
    for d in (10,11):
        points=[];references=[]
        for key in range(1,9):
            v=(-6.,-3.,0.,3.)[(key-1)%4]
            q=np.array([v]+[math.exp(v/2)*(.08*j-.35) for j in range(1,d)])
            points.append(q);references.append(transition(author,2,q,key,.3,
                maxdepth=4,maxc=10,delta=.3))
        cfg=Config(8,d,variant='r2p',max_depth=4,r2p_max_exponent=10,max_error=.3)
        k=R2PKernel(tf,model_for(tf,2),cfg,tape=True)
        @tf.function(input_signature=[tf.TensorSpec([8,d],tf.float64)],jit_compile=True,autograph=False)
        def step(q):
            lp,g=k.value_gradient(q);h=tf.fill([8],tf.constant(.3,tf.float64))
            return k.transition(q,lp,g,h,tf.ones_like(q),tf.ones([8],tf.int32),
                tf.constant([[key,0] for key in range(1,9)],tf.int32),k.adam_init(h))
        out=step(tf.constant(points))
        for field in ('q','lp','g'):
            np.testing.assert_allclose(getattr(out,field).numpy(),[r[field] for r in references],atol=2e-11,rtol=2e-11)
        for field in ('depth','evaluations','macros'):
            np.testing.assert_array_equal(getattr(out,field).numpy(),[r[field] for r in references])
        np.testing.assert_array_equal(out.seed.numpy()[:,1],[r['counter'] for r in references])

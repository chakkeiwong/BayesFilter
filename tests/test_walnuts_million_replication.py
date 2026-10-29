"""Independent-reference mechanics for the million-draw replication harness."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import numpy as np
import pytest


@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('evaluator',['scipy','analytic'])
def test_author_chunking_preserves_rng_and_draws(corrected,evaluator):
    from docs.benchmarks.walnutpie_gpu_tf.funnel_reference import load_author,start
    from docs.benchmarks.walnutpie_gpu_tf.replication import reference_evaluator
    wn,ai,td,_=load_author(corrected)
    model=reference_evaluator(evaluator,td)
    q=start(9,11);rng=np.random.get_state()
    kwargs=dict(integrator=ai.adaptLeapFrogR2P,M=4,H0=.3,delta0=.3,warmupIter=0,recordOrbitStats=True)
    full=wn.WALNUTS(model,q,numIter=12,**kwargs);end=np.random.get_state()
    np.random.set_state(rng)
    a=wn.WALNUTS(model,q,numIter=5,**kwargs)
    b=wn.WALNUTS(model,a[0][:,-1].copy(),numIter=7,**kwargs)
    np.testing.assert_array_equal(full[0],np.concatenate([a[0],b[0][:,1:]],axis=1))
    np.testing.assert_array_equal(full[1],np.concatenate([a[1],b[1]],axis=0))
    for i in (2,3):np.testing.assert_array_equal(full[i],np.concatenate([a[i],b[i]],axis=1))
    observed=np.random.get_state()
    assert end[0]==observed[0] and end[2:]==observed[2:]
    np.testing.assert_array_equal(end[1],observed[1])


def test_tuning_evaluator_identity_rejects_mixing():
    from docs.benchmarks.walnutpie_gpu_tf.replication import (
        EVALUATORS,evaluator_kind,validate_tuning_evaluator)
    for kind,label in EVALUATORS.items():
        record={'evaluator':label}
        assert evaluator_kind(record)==kind
        validate_tuning_evaluator(record,kind)
        other='scipy' if kind=='analytic' else 'analytic'
        with pytest.raises(ValueError,match='does not match'):
            validate_tuning_evaluator(record,other)
        with pytest.raises(ValueError,match='inconsistent'):
            evaluator_kind({**record,'evaluator_kind':other})
    with pytest.raises(ValueError,match='unknown'):
        evaluator_kind({})


def test_analytic_reference_does_not_call_scipy():
    from types import SimpleNamespace
    from docs.benchmarks.walnutpie_gpu_tf.replication import reference_evaluator
    def forbidden(q):
        raise AssertionError('analytic selector called SciPy')
    model=reference_evaluator('analytic',SimpleNamespace(funnel10=forbidden))
    lp,g=model(np.zeros(11))
    assert np.isfinite(lp) and g[0]==-5.
    np.testing.assert_array_equal(g[1:],np.zeros(10))


def test_binary_shape_guard_and_batch_summary(tmp_path):
    from docs.benchmarks.walnutpie_gpu_tf.replication_analysis import read_draws,batch_summary
    x=np.arange(64*4*11,dtype='<f8').reshape(64,4,11)
    (tmp_path/'draws.f64').write_bytes(x.tobytes())
    chains=read_draws(tmp_path,'tf-cpu',64)
    np.testing.assert_array_equal(chains[2],x[:,2,:])
    with pytest.raises(ValueError):read_draws(tmp_path,'tf-cpu',65)
    d=batch_summary(np.ones((4,64)))
    assert d['mean']==1. and d['batch_se']==0.


def test_author_generated_projection_does_not_change_chain():
    from docs.benchmarks.walnutpie_gpu_tf.funnel_reference import load_author,start
    wn,ai,td,_=load_author(False);q=start(4,11);rng=np.random.get_state()
    args=dict(integrator=ai.adaptLeapFrogR2P,M=4,H0=.3,delta0=.3,warmupIter=0,numIter=8,recordOrbitStats=True)
    full=wn.WALNUTS(td.funnel10,q,**args)
    np.random.set_state(rng);projected=wn.WALNUTS(td.funnel10,q,generated=lambda x:x[:2],**args)
    np.testing.assert_array_equal(full[0][:2],projected[0])
    np.testing.assert_array_equal(full[1],projected[1])
    np.testing.assert_array_equal(full[2][:2],projected[2])
    np.testing.assert_array_equal(full[3][:2],projected[3])

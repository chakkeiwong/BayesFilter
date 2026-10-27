"""Independent analytic and finite-difference checks for benchmark targets."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'

import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/benchmarks'))
import benchmark_nuts_xla as base
from nuts_benchmark_targets import make_target, NAMES


@pytest.fixture(scope='module')
def framework(tmp_path_factory):
    args=base.parser().parse_args(['--device','cpu','--output',str(tmp_path_factory.mktemp('geometry'))])
    return base.setup(args)[:2]


@pytest.mark.parametrize('name',(*NAMES,'banana2'))
def test_reference_inverse_density_and_score(framework,name):
    tf,_=framework
    target=make_target(tf,name)
    signature=[tf.TensorSpec([4,target.dimension],tf.float64)]
    @tf.function(input_signature=signature,autograph=False,jit_compile=True)
    def check(z):
        q=target.from_reference(z)
        with tf.GradientTape() as tape:
            tape.watch(q);value=target.log_prob(q)
        score=tape.gradient(value,q)
        errors=[]
        for j in range(target.dimension):
            delta=tf.one_hot(j,target.dimension,dtype=tf.float64)*1e-5
            fd=(target.log_prob(q+delta)-target.log_prob(q-delta))/2e-5
            errors.append(tf.abs(fd-score[:,j])/(1.+tf.abs(score[:,j])))
        expected=-.5*tf.reduce_sum(tf.square(z),axis=-1)
        if name=='funnel10': expected-=9.*q[:,0]/2.
        return target.to_reference(q),value-expected,tf.reduce_max(errors)
    z=tf.random.stateless_normal([4,target.dimension],[411,82],dtype=tf.float64)*.3
    recovered,residual,score_error=check(z)
    assert float(tf.reduce_max(tf.abs(recovered-z)).numpy().item())<1e-10
    assert float(tf.reduce_max(tf.abs(residual)).numpy().item())<1e-9
    assert float(score_error.numpy().item())<1e-7
    assert check.experimental_get_tracing_count()==1


def test_custom_target_compiles_through_shared_sampler(framework,tmp_path):
    tf,tfp=framework
    args=base.parser().parse_args(['--device','cpu','--output',str(tmp_path),
        '--dimension','10','--warmup','8','--draws','8'])
    target=make_target(tf,'funnel10')
    function=base.make_chain(tf,tfp,args,adapting=False,log_prob_fn=target.log_prob)
    q=tf.zeros([4,10],tf.float64);e=tf.constant(.1,tf.float64);seed=tf.constant([7,9],tf.int32)
    result,_=base.timed(function,q,e,seed)
    assert bool(tf.reduce_all(result[3][...,5:]==1).numpy().item())
    assert 'HloModule' in function.experimental_get_compiler_ir(q,e,seed)(stage='optimized_hlo')

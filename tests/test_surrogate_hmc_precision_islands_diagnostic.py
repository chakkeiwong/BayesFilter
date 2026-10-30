"""Deliberate CPU/XLA diagnostic checks; GPU devices are hidden."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pytest
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.testing.surrogate_hmc_precision_islands_diagnostic import private_module, make_program

@pytest.mark.parametrize("policy", ["both_fp32", "gaussian_fp32", "transport_fp32"])
def test_private_module_isolation_and_centered_stage_agreement(policy):
    original = (shared.gaussian_mixture, shared.transport)
    module = private_module(policy)
    assert (shared.gaussian_mixture, shared.transport) == original
    b,n,d,p = 2,16,2,3
    x = tf.random.stateless_normal([b,n,d],[512,1],dtype=tf.float64)
    dx = tf.random.stateless_normal([p,b,n,d],[512,2],dtype=tf.float64)
    # Large common offsets would destroy the local geometry if cast first.
    origin = tf.constant(1.e8,tf.float64)
    tangent_origin = tf.constant(1.e9,tf.float64)
    cloud = Jet(x+origin,dx+tangent_origin)
    means = Jet(.9*x+origin,.8*dx+tangent_origin)
    covariance = Jet(tf.eye(d,batch_shape=[b,n],dtype=tf.float64),
                     tf.zeros([p,b,n,d,d],tf.float64))
    weights = tf.nn.log_softmax(tf.random.stateless_normal([b,n],[512,3],dtype=tf.float64))
    dweights = tf.random.stateless_normal([p,b,n],[512,4],dtype=tf.float64)
    dweights -= tf.reduce_sum(tf.exp(weights)[None]*dweights,-1,keepdims=True)
    logw = Jet(weights,dweights)
    control = ProposalControls(reset_steps=4,reset_epsilon=2.)
    @tf.function(input_signature=[],jit_compile=True,autograph=False)
    def evaluate():
        density,valid = module.gaussian_mixture(cloud,means,covariance,True)
        expected,expected_valid = shared.gaussian_mixture(cloud,means,covariance,True)
        y,_,_ = module.transport(cloud,logw,control)
        reference,_,_ = shared.transport(cloud,logw,control)
        return density.v,density.t,expected.v,expected.t,valid,expected_valid,y.v,y.t,reference.v,reference.t
    a,da,e,de,v,ev,y,dy,ey,edy = evaluate()
    tf.debugging.assert_equal(v,ev)
    tf.debugging.assert_near(a,e,atol=3e-5,rtol=3e-5)
    tf.debugging.assert_near(da,de,atol=3e-5,rtol=3e-5)
    tf.debugging.assert_near(y-origin,ey-origin,atol=3e-5,rtol=3e-5)
    tf.debugging.assert_near(dy-tangent_origin,edy-tangent_origin,atol=3e-5,rtol=3e-5)
    assert (shared.gaussian_mixture, shared.transport) == original


def test_shared_full_filter_mixed_precision_cpu_xla():
    from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
    dtype=tf.float64
    spec=LGSSMSpec("diagonal_ar",1); n,h=32,2
    controls=ProposalControls(flow_steps=2,reset_steps=4)
    inputs=(tf.constant([[.9,.6,.8],[.88,.61,.82]],dtype),
            tf.constant([[.5],[.3]],dtype),
            tf.random.stateless_normal([n,1],[41,1],dtype=dtype),
            tf.random.stateless_normal([h,n,1],[41,2],dtype=dtype),
            tf.random.stateless_uniform([h,n,2],[41,3],dtype=dtype),
            tf.constant([.2,.3,.5],dtype))
    mixed,metadata=make_program(spec,n,h,controls,policy="both_fp32",batch_size=2)
    native=shared.make_batched_score_filter(spec,n,h,controls,dtype,True,batch_size=2)
    a,e=mixed(*inputs),native(*inputs)
    tf.debugging.assert_equal(a["valid"],[True,True])
    tf.debugging.assert_equal(a["steps_completed"],[h,h])
    for key in ("value","score"):
        tf.debugging.assert_near(a[key],e[key],atol=3e-5,rtol=3e-5)
    assert metadata["recurrence_dtype"]=="float64"
    assert metadata["gaussian_pair_dtype"]=="float32"
    assert metadata["runtime_module_patched"] is False
    assert mixed.experimental_get_tracing_count()==1
    assert mixed.get_concrete_function().function_def.attr["_XlaMustCompile"].b

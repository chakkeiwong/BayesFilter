"""Independent diagnostic parity for native B/all-P analytical filtering."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, stack
from bayesfilter.highdim.covariance_proposal_native_models_tf import NativeModels
from bayesfilter.highdim.covariance_proposal_score_tf import Models, observation_density
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter

D=tf.float64


class RangeReferenceBinding:
    """Diagnostic binding of existing M13 model/initial-state authorities."""
    dimension, observation_dimension, parameter_count = 4, 2, 2

    def __init__(self, specification):
        self.spec = specification

    def model(self, theta, direction):
        from bayesfilter.highdim.ledh_canonical_batch_fused_tf import _single_cloud_model
        return _single_cloud_model(self.spec['model'], theta, direction), None

    def initial_cloud(self, theta, initial, direction):
        covariance = tf.broadcast_to(tf.cast(self.spec['fixture'].P0, initial.dtype),
                                     [initial.shape[0], 4, 4])
        return initial, covariance, tf.zeros_like(initial), tf.zeros_like(covariance)

def fixture(name):
    from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
    from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
    from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    if name=='range':
        from bayesfilter.testing.marginal_weight_campaign import specification
        spec = specification('m13-t2', D)
        z = tf.random.stateless_normal([4,32,4], [812,19], dtype=D)
        initial = tf.cast(spec['fixture'].m0,D) + z[0] @ tf.transpose(
            tf.linalg.cholesky(tf.cast(spec['fixture'].P0,D)))
        uniform = tf.random.stateless_uniform([2,32,2],[812,29],dtype=D)
        initial,noise,uniform = (tf.cast(tf.cast(x,tf.float32),D)
                                for x in (initial,z[1:-1],uniform))
        return RangeReferenceBinding(spec),tf.zeros([2],D),initial,noise,uniform
    spec=(LGSSMSpec('frozen_3d',3) if name=='lgssm' else
          LGSSMSpec('p44',3) if name=='p44' else
          FullLGSSMSpec('full_matrix',3) if name=='full' else
          KSCSpec() if name=='ksc' else NonlinearSQMCSpec(name))
    d=spec.dimension
    return (spec,spec.default_theta(D),tf.random.stateless_normal([32,d],[812,1],dtype=D),
            tf.random.stateless_normal([2,32,d],[812,2],dtype=D),
            tf.random.stateless_uniform([2,32,2],[812,3],dtype=D))

@pytest.mark.parametrize('name',['lgssm','p44','full','ksc','predator_prey','sir_d18','range'])
def test_model_values_and_every_partial_match_scalar_callbacks(name):
    spec,q,initial,_,_=fixture(name)
    theta=tf.stack([q,q+.001]); p=spec.parameter_count
    native=NativeModels(spec,theta)
    x=native.initial_cloud(initial)
    reference=[]
    for row in range(2):
        states=[spec.initial_cloud(theta[row],initial,tf.eye(p,dtype=D)[k]) for k in range(p)]
        reference.append(Jet(states[0][0],tf.stack([s[2] for s in states])))
    expected=stack(reference)
    tf.debugging.assert_near(x.v,expected.v,atol=1e-10,rtol=1e-10)
    tf.debugging.assert_near(x.t,expected.t,atol=1e-10,rtol=1e-10)
    x=Jet(x.v,tf.random.stateless_normal(x.t.shape,[817,1],dtype=D))
    scalar=[Models(spec,theta[b],tf.eye(p,dtype=D)) for b in range(2)]
    for value,tangent in [('transition_mean_fn','transition_mean_tangent_fn'),
                          ('observation_fn','observation_tangent_fn'),
                          ('observation_jacobian_fn','observation_jacobian_tangent_fn')]:
        actual=native.apply(value,tangent,x,theta=value.startswith('transition'))
        expected=stack([m.apply(value,tangent,x[b],theta=value.startswith('transition')) for b,m in enumerate(scalar)])
        tf.debugging.assert_near(actual.v,expected.v,atol=1e-9,rtol=1e-9,message=name+value)
        tf.debugging.assert_near(actual.t,expected.t,atol=1e-9,rtol=1e-9,message=name+tangent)
    observed=tf.reduce_mean(native.observation(x).v,[0,1])+.15
    actual=native.observation_density(x,observed)
    expected=stack([observation_density(m,x[b],observed) for b,m in enumerate(scalar)])
    tf.debugging.assert_near(actual.v,expected.v,atol=1e-9,rtol=1e-9)
    tf.debugging.assert_near(actual.t,expected.t,atol=1e-9,rtol=1e-9)
    for value,tangent in [('process_covariance','process_covariance_tangent_fn'),('observation_covariance','observation_covariance_tangent_fn')]:
        actual=native.covariance(value,tangent)
        expected=stack([m.covariance(value,tangent) for m in scalar])
        tf.debugging.assert_near(actual.v,expected.v,atol=1e-10,rtol=1e-10)
        tf.debugging.assert_near(actual.t,expected.t,atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize('name',['lgssm','ksc','predator_prey','sir_d18','range'])
@pytest.mark.parametrize('correction',[0,2])
def test_full_native_xla_recurrence_matches_independent_directional_filter(name,correction):
    spec,q,initial,noise,uniform=fixture(name)
    ctl=ProposalControls(flow_steps=4,reset_steps=5,reset_epsilon=2.,correction_steps=correction)
    theta=tf.stack([q,q+.001]); beta=tf.constant([.2,.3,.5],D)
    model,_=spec.model(q,tf.zeros_like(q)); x=spec.initial_cloud(q,initial,tf.zeros_like(q))[0]
    obs=[]
    for _ in range(2):
        x=model.transition_mean_fn(q,x)
        obs.append(tf.reduce_mean(model.observation_fn(x),0)+.15)
    obs=tf.stack(obs)
    program=make_batched_score_filter(spec,32,2,ctl,D,True,2,
        fixed_draw_design=(beta,uniform))
    actual=program(theta,obs,initial,noise)
    permuted=program(tf.reverse(theta,[0]),obs,initial,noise)
    for key in ('value','score','particles','particle_tangent'):
        tf.debugging.assert_near(actual[key],tf.reverse(permuted[key],[0]),atol=1e-8,rtol=1e-8)
    for key in ('valid','steps_completed','validity_trace'):
        tf.debugging.assert_equal(actual[key],tf.reverse(permuted[key],[0]))
    tf.debugging.assert_equal(actual['valid'],[True,True])
    reference=make_filter(spec,32,2,ctl,D,jit_compile=False)
    for row in range(2):
        for k in range(spec.parameter_count):
            result=reference(theta[row],tf.one_hot(k,spec.parameter_count,dtype=D),obs,initial,noise,uniform,beta)
            tf.debugging.assert_equal(result['valid'],True)
            tf.debugging.assert_near(actual['value'][row],result['value'],atol=1e-8,rtol=1e-8)
            tf.debugging.assert_near(actual['score'][row,k],result['score'],atol=1e-8,rtol=1e-8)
            tf.debugging.assert_near(actual['particles'][row],result['particles'],atol=1e-8,rtol=1e-8)

def test_native_graph_body_does_not_grow_with_batch_size():
    spec,_,_,_,_=fixture('sir_d18')
    counts=[]
    for b in (1,2,4,8,16,32,64,128):
        f=make_batched_score_filter(spec,32,2,ProposalControls(flow_steps=2),tf.float32,True,b)
        graph=f.get_concrete_function().graph.as_graph_def()
        # TF uses Const or Fill for zero tensors at size-dependent thresholds.
        # Inspect the numerical body; those materialization nodes do not replay it.
        nodes=list(graph.node)+[node for fn in graph.library.function for node in fn.node_def]
        counts.append(sum(node.op not in ('Const','Fill') for node in nodes))
    assert len(set(counts))==1,counts


def test_scalar_public_endpoint_preserves_scalar_and_row_outputs():
    from bayesfilter.highdim.covariance_proposal_score_tf import make_score_filter
    spec,q,initial,noise,uniform=fixture('lgssm')
    ctl=ProposalControls(flow_steps=2,reset_steps=2)
    observations=tf.zeros([2,3],D)
    beta=tf.constant([.2,.3,.5],D)
    scalar=make_score_filter(spec,32,2,ctl,D,jit_compile=True)
    native=make_batched_score_filter(spec,32,2,ctl,D,True,1)
    actual=scalar(q,observations,initial,noise,uniform,beta)
    expected=native(q[None],observations,initial,noise,uniform,beta)
    assert actual.keys()==expected.keys()
    for key,value in expected.items():
        value=value[0] if value.shape.rank else value
        if value.dtype.is_floating:
            tf.debugging.assert_equal(tf.math.is_finite(actual[key]),tf.math.is_finite(value))
            tf.debugging.assert_near(tf.where(tf.math.is_finite(actual[key]),actual[key],0),
                                     tf.where(tf.math.is_finite(value),value,0),atol=1e-8,rtol=1e-8)
        else:
            tf.debugging.assert_equal(actual[key],value)

"""Independent Gaussian and consumer wiring checks; deliberate CPU references."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from bayesfilter.highdim import observation_guided_tt_tf as obs
from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
from bayesfilter.highdim import pair_block_tt_tf as pair
from docs.benchmarks import run_observation_aware_tt_complete as runner

D = tf.float64


def fixture():
    model = obs.SVModel(tf.constant([[.6, .1], [-.2, .7]], D), tf.eye(2, dtype=D))
    prior = obs.Chart.from_moments(tf.constant([.1, -.2], D), tf.constant([[1.3, .2], [.2, .8]], D))
    current = obs.Chart.from_moments(tf.constant([-.3, .4], D), tf.constant([[.7, -.1], [-.1, 1.1]], D))
    return model, prior, current


def test_conditional_matches_backward_joint_minus_previous_marginal():
    model, prior, current = fixture()
    step = joint.make_sgqf_joint_step(model, current, prior, 1)
    z = tf.random.stateless_normal([23, 2], [18, 1], dtype=D)
    x = tf.random.stateless_normal([23, 2], [18, 2], dtype=D)
    P = prior.factor @ tf.transpose(prior.factor)
    S = model.transition @ P @ tf.transpose(model.transition)+tf.eye(2, dtype=D)
    K = tf.transpose(tf.linalg.solve(S, model.transition @ P))
    backwards = obs.Chart.from_moments(tf.zeros([2], D), P-K @ S @ tf.transpose(K))
    zm = prior.mean+tf.linalg.matmul(x-tf.linalg.matvec(model.transition, prior.mean), K, transpose_b=True)
    logjoint = current.log_prob(x)+backwards.log_prob(z-zm)
    tf.debugging.assert_near(step.conditional_log_density(x,z), logjoint-step.previous_marginal.log_prob(z), atol=2e-12)
    tf.debugging.assert_equal(step.retained_proposal.physical_log_density(x), current.log_prob(x))
    draws, logq, _ = joint.sample_joint_step(step,z, 31)
    tf.debugging.assert_near(logq,step.conditional_log_density(draws,z),atol=2e-12)


def test_linear_gaussian_posterior_conditional_exact_limit():
    model, prior, _ = fixture()
    A = model.transition
    P = prior.factor @ tf.transpose(prior.factor)
    S = A @ P @ tf.transpose(A)+tf.eye(2,dtype=D)
    H = tf.constant([[1., .3]],D)
    R = tf.constant([[.4]],D)
    y = tf.constant([.6],D)
    predicted = tf.linalg.matvec(A,prior.mean)
    gain = S @ tf.transpose(H) @ tf.linalg.inv(H @ S @ tf.transpose(H)+R)
    current = obs.Chart.from_moments(predicted+tf.linalg.matvec(gain,y-tf.linalg.matvec(H,predicted)),S-gain @ H @ S)
    step = joint.make_sgqf_joint_step(model,current,prior,1)
    z = tf.constant([[.2, -.4], [1., .7]],D)
    # Independently update the transition N(Az,I) using the linear observation.
    V = tf.linalg.inv(tf.eye(2,dtype=D)+tf.transpose(H) @ tf.linalg.solve(R,H))
    means = tf.linalg.matmul(tf.linalg.matmul(z,A,transpose_b=True)+tf.transpose(tf.transpose(H) @ tf.linalg.solve(R,y[:,None])),V,transpose_b=True)
    actual_means = current.mean+tf.linalg.matmul(z-step.previous_marginal.mean,step.gain,transpose_b=True)
    tf.debugging.assert_near(actual_means,means,atol=2e-12)
    tf.debugging.assert_near(step.conditional_factor @ tf.transpose(step.conditional_factor),V,atol=2e-12)


def test_particle_filter_consumes_gaussian_and_tt_steps(monkeypatch):
    # The CLI defers TensorFlow imports until memory policy is configured.
    monkeypatch.setattr(runner,'tf',tf,raising=False)
    monkeypatch.setattr(runner,'D',D,raising=False)
    monkeypatch.setattr(runner,'lib',obs,raising=False)
    model, prior, current = fixture()
    initial = joint.make_sgqf_joint_step(model,current,None,0)
    cores = (tf.ones([1,1,1,1],D), tf.ones([1,1,1,1],D))
    tau = tf.constant(.1,D)
    retained = obs.PairRetainedProposal(cores,current,tf.constant(1.,D),tau,1)
    tt = obs.PairTTStep(cores,current,prior,tau,{},1,retained)
    calls=[]
    original=joint.sample_joint_step
    def tracked(step,previous,seed,jit_compile=True):
        calls.append(type(step).__name__)
        return original(step,previous,seed,jit_compile)
    monkeypatch.setattr(joint,'sample_joint_step',tracked)
    result, history=runner.particle_filter(model,tf.constant([[.1,.2],[.3,.1]],D),
        [(prior,current),(prior,current)],[initial,tt],'tt_sgqf_safeguard',64,421,True,keep_history=True)
    assert calls==['SGQFJointStep','PairTTStep']
    assert len(result['steps'])==2
    # Initial evidence is independently reconstructed from the actual proposal.
    x=history[0]['x']
    expected=tf.reduce_logsumexp(model.prior_log_prob(x)+model.observation_log_prob(x,tf.constant([.1,.2],D))-current.log_prob(x))-tf.math.log(tf.constant(64.,D))
    tf.debugging.assert_near(result['steps'][0]['log_increment'],expected,atol=2e-12)


def test_defensive_diagnostics_expose_conditional_not_joint_fraction():
    core=tf.reshape(tf.constant([0.,1.,0.,0.],D),[1,2,2,1])
    v=tf.constant([[0.],[.5],[1.]],D)
    _,_,info=pair.sample_pair_conditional((core,),v,tf.constant(.1,D),
        tf.constant([.5,.5,.5],D),tf.constant([[.2],[.4],[.6]],D),tf.zeros([3,1],D))
    tf.debugging.assert_near(info['minimum_conditional_gaussian_fraction'],tf.constant(1/11,D))
    tf.debugging.assert_equal(info['maximum_conditional_gaussian_fraction'],tf.constant(1.,D))
    assert int(info['gaussian_component_draws'])==1

"""CPU independent references for optional modern transitions, not promotion."""
import math

import numpy as np  # Independent analytic/finite-difference diagnostic only.
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_modern_kernels import ReusableModernHMCRunner, sphere_force_substep
from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
from tests.test_hmc_candidate_set_execution import GaussianTarget


def runner(*, family="hmc", trajectory="uniform_duration", count=4, chains=4, dim=2, jit=False, target=None, status="none"):
    return ReusableModernHMCRunner(GaussianTarget() if target is None else target,
        tf.zeros([chains, dim], tf.float64), profile=HMCTransitionProfile(family, trajectory),
        num_results=count, use_xla=jit, target_status_trace_policy=status)


def test_one_step_hmc_matches_independent_linear_map_and_energy():
    r = runner(trajectory="fixed")
    q = tf.constant([[.2,-.8],[-1.,1.],[.1,.9],[2.,-.3]], tf.float64)
    key = tf.constant([381, 52], tf.int32)
    _, trace = r.one_step(q, key, tf.constant(.4,tf.float64), tf.constant(3))
    p = tf.random.stateless_normal(q.shape, tf.random.experimental.stateless_split(key,3)[0], dtype=tf.float64, alg="philox").numpy()
    # Exact matrix for one velocity-Verlet step on a unit Gaussian.
    h = .4
    a = np.array([[1-h*h/2, h],[-h+h**3/4,1-h*h/2]])
    qp = np.stack([q.numpy(),p],axis=-1) @ np.linalg.matrix_power(a,3).T
    np.testing.assert_allclose(trace["proposed_state"], qp[...,0],rtol=2e-14,atol=2e-14)
    expected_ratio = -.5 * np.sum(qp[...,0]**2+qp[...,1]**2-q.numpy()**2-p**2,axis=-1)
    np.testing.assert_allclose(trace["log_accept_ratio"],expected_ratio,rtol=2e-13,atol=2e-14)


@pytest.mark.parametrize("family", ["hmc","mams"])
def test_full_refresh_resume_preserves_position_stream_and_work(family):
    q = tf.constant([[-1.,-.4],[-.2,.1],[.4,-.2],[1.,.5]],tf.float64)
    whole = runner(family=family,count=6)
    part = runner(family=family,count=3)
    kw = dict(seed=(730,29),step_size=.2,num_leapfrog_steps=7)
    full = whole.run(current_state=q,**kw)
    first = part.run(current_state=q,**kw)
    second = part.run(current_state=first.samples[-1],transition_offset=3,**kw)
    np.testing.assert_array_equal(full.samples,tf.concat([first.samples,second.samples],0))
    for key in full.trace:
        np.testing.assert_array_equal(full.trace[key],tf.concat([first.trace[key],second.trace[key]],0))
    assert full.metadata["value_score_row_evaluations"] <= full.metadata["maximum_reserved_gradient_work"]
    assert full.metadata["target_status_row_evaluations"] == 0
    assert whole._sample.experimental_get_tracing_count() == 1
    if family == "mams":
        np.testing.assert_allclose(tf.linalg.norm(full.trace["final_momentum"],axis=-1),1.,atol=5e-14)


def test_uniform_duration_induced_count_law_and_stream_separation():
    n = 4096
    r = runner(chains=n,count=1)
    q = tf.zeros([n,2],tf.float64)
    _, trace = r.one_step(q,tf.constant([94,123]),tf.constant(.2,tf.float64),tf.constant(5))
    counts = np.bincount(trace["trajectory_steps"].numpy(),minlength=6)[1:]
    # Explicit conservative Hoeffding union bound at 1e-6 for this mechanics check.
    tolerance=math.sqrt(math.log(2*5/1e-6)/(2*n))
    assert np.max(np.abs(counts/n-.2)) < tolerance
    assert np.unique(trace["initial_momentum"].numpy(),axis=0).shape[0] == n
    assert int(tf.reduce_max(trace["value_score_calls_per_chain"])) == 6


def test_sphere_force_inverse_and_surface_jacobian_against_tangent_difference():
    u=np.array([.3,-.4,math.sqrt(.75)])
    g=np.array([.7,-.2,.4])
    def force(v,h):
        return sphere_force_substep(tf.constant(v[None],tf.float64),tf.constant(g[None],tf.float64),tf.constant(h,tf.float64))
    result,work=force(u,.37); v=result.numpy()[0]
    restored,_=force(v,-.37)
    np.testing.assert_allclose(restored.numpy()[0],u,rtol=0,atol=3e-15)
    # Independent surface differential in orthonormal tangent bases.
    def basis(x):
        a=np.cross(x,[1.,0.,0.]);a/=np.linalg.norm(a)
        return np.stack([a,np.cross(x,a)],axis=1)
    b0,b1=basis(u),basis(v); h=1e-5
    cols=[]
    for b in b0.T:
        plus=np.cos(h)*u+np.sin(h)*b
        minus=np.cos(h)*u-np.sin(h)*b
        cols.append((force(plus,.37)[0].numpy()[0]-force(minus,.37)[0].numpy()[0])/(2*h))
    logj=np.log(abs(np.linalg.det(b1.T@np.stack(cols,axis=1))))
    np.testing.assert_allclose(logj,-float(work[0]),atol=2e-10,rtol=0)
    # Mutation control: dropping/reversing this correction is detected.
    assert abs(logj-float(work[0])) > .1


def test_sphere_zero_force_and_large_exact_poles_have_analytic_limits():
    u=tf.constant([[1.,0.],[-1.,0.],[0.,1.]],tf.float64)
    g=tf.constant([[2.,0.],[2.,0.],[0.,0.]],tf.float64)
    result,work=sphere_force_substep(u,g,tf.constant(400.,tf.float64))
    np.testing.assert_array_equal(result,u)
    np.testing.assert_allclose(work,[800.,-800.,0.],rtol=0,atol=0)
    unchanged, zero=sphere_force_substep(u,g,tf.constant(0.,tf.float64))
    np.testing.assert_array_equal(unchanged,u)
    np.testing.assert_array_equal(zero,[0.,0.,0.])


def test_mams_rejects_one_dimension_and_untyped_profiles():
    with pytest.raises(ValueError,match="dimension"):
        runner(family="mams",dim=1)
    with pytest.raises(ValueError):
        HMCTransitionProfile("nuts","fixed")


@pytest.mark.parametrize("family", ["hmc","mams"])
def test_native_cpu_xla_matches_non_xla_reference(family):
    q=tf.constant([[-1.,-.4],[-.2,.1],[.4,-.2],[1.,.5]],tf.float64)
    ordinary=runner(family=family,count=2,jit=False)
    compiled=runner(family=family,count=2,jit=True)
    # Same supplied auxiliary state is not supported by the public runner;
    # native RNG is allowed backend-specific rounding. Compare invariants and
    # within-backend deterministic replay, not unpromised cross-backend bits.
    kw=dict(current_state=q,seed=(830,21),step_size=.1,num_leapfrog_steps=4)
    a=ordinary.run(**kw);b=compiled.run(**kw);b2=compiled.run(**kw)
    np.testing.assert_array_equal(b.samples,b2.samples)
    for result in (a,b):
        assert np.all(result.trace["target_score_finite"])
        np.testing.assert_allclose(result.trace["target_log_prob"],-.5*np.sum(result.samples.numpy()**2,axis=-1),atol=1e-13)
        ratio=result.trace["proposed_target_log_prob"].numpy()+.5*np.sum(np.concatenate([q.numpy()[None],result.samples.numpy()[:-1]],axis=0)**2,axis=-1)+result.trace["log_acceptance_correction"].numpy()
        np.testing.assert_allclose(ratio,result.trace["log_accept_ratio"],atol=1e-13)


@pytest.mark.parametrize("family", ["hmc","mams"])
def test_exact_stationary_gaussian_anchors_after_one_transition(family):
    # Independent exact target anchors; this checks invariance, not mixing.
    n=4096
    q=tf.random.stateless_normal([n,2],[81,329],dtype=tf.float64,alg="philox")
    r=runner(family=family,count=1,chains=n)
    result=r.run(current_state=q,seed=(31,822),step_size=.7,num_leapfrog_steps=7)
    x=result.samples.numpy()[0]
    assert np.max(np.abs(np.mean(x,axis=0))) < 6/math.sqrt(n)
    assert np.max(np.abs(np.mean(x*x,axis=0)-1.)) < 6*math.sqrt(2/n)
    assert abs(np.mean(x[:,0]*x[:,1])) < 6/math.sqrt(n)
    assert np.all(result.trace["target_score_finite"])


@pytest.mark.parametrize("family", ["hmc","mams"])
def test_existing_validation_engine_executes_same_modern_transition(family):
    from bayesfilter.testing.inference_validation.procedures import FrozenTransition
    target=GaussianTarget()
    q=tf.constant([[-1.,-.4],[-.2,.1],[.4,-.2],[1.,.5]],tf.float64)
    key=tf.constant([61,99])
    profile=HMCTransitionProfile(family,"uniform_duration")
    validation=FrozenTransition(target,chains=4,step_size=.2,leapfrog_steps=4,
        jit_compile=False,transition_profile=profile.payload())
    direct=runner(family=family).one_step(q,key,tf.constant(.2,tf.float64),tf.constant(4))
    checked=validation.audit_step(q,key)
    np.testing.assert_array_equal(checked["state"],direct[0])
    np.testing.assert_array_equal(checked["log_accept_ratio"],direct[1]["log_accept_ratio"])
    _,_,healthy=validation.powered_step(2,with_health=True)(q,key)
    assert bool(healthy)


def test_intermediate_status_failure_cannot_be_hidden_by_healthy_return_endpoint():
    class ForbiddenExcursion(GaussianTarget):
        def target_status_telemetry(self,q):
            # Unit-Gaussian leapfrog at epsilon=1,L=3 returns to q=0 from
            # q=0, with two nonzero intermediate points. Endpoints are healthy.
            bad=tf.reduce_any(tf.abs(q)>1e-8,axis=-1)
            return {"status_code":tf.cast(bad,tf.int32),
                    "valid_pre_regularized_score":~bad,
                    "floor_count_value":tf.zeros_like(tf.cast(bad,tf.int32))}
    r=runner(trajectory="fixed",count=1,target=ForbiddenExcursion(),status="per_chain_step")
    out=r.run(current_state=tf.zeros([4,2],tf.float64),seed=(838,20),step_size=1.,num_leapfrog_steps=3)
    np.testing.assert_allclose(out.trace["proposed_state"],0.,atol=1e-14)
    assert np.all(out.trace["proposed_target_status_telemetry"]["status_code"]==0)
    assert not np.any(out.trace["target_score_finite"])
    assert not np.any(out.trace["is_accepted"])


@pytest.mark.parametrize("family",["hmc","mams"])
def test_bad_proposal_rejection_preserves_state_but_not_healthy_evidence(family):
    class PoisonedScore(GaussianTarget):
        def log_prob_and_grad(self,q):
            value,score=super().log_prob_and_grad(q)
            return value,tf.where(tf.reduce_any(q!=0.,axis=-1)[:,None],
                                  tf.fill(tf.shape(score),tf.constant(float("nan"),tf.float64)),score)
    q=tf.zeros([4,2],tf.float64)
    out=runner(family=family,count=1,target=PoisonedScore()).run(
        current_state=q,seed=(918,122),step_size=.2,num_leapfrog_steps=2)
    np.testing.assert_array_equal(out.samples[0],q)
    assert not np.any(out.trace["target_score_finite"])
    assert not np.any(out.trace["is_accepted"])


@pytest.mark.parametrize('family', ['hmc', 'mams'])
@pytest.mark.parametrize('jit', [False, True])
def test_first_failure_distinguishes_initial_and_proposal_and_keeps_custom_status(family, jit):
    class BadExcursion(GaussianTarget):
        def target_status_telemetry(self, q):
            bad = tf.reduce_any(q != 0., axis=-1)
            return {'status_code': tf.cast(bad, tf.int32),
                    'valid_pre_regularized_score': ~bad,
                    'floor_count_value': tf.zeros([4], tf.int32),
                    'chart_residual': tf.reduce_sum(q*q, axis=-1)}
    r = runner(family=family, trajectory='fixed', count=1, jit=jit,
               target=BadExcursion(), status='per_chain_step')
    q = tf.zeros([4, 2], tf.float64)
    out = r.run(current_state=q, seed=(918,122), step_size=.2, num_leapfrog_steps=3)
    np.testing.assert_array_equal(out.trace['first_failure_substep'], [[1]*4])
    np.testing.assert_array_equal(out.trace['first_failure_stage'], [[4]*4])
    assert np.all(out.trace['first_failure_target_chart_residual'] > 0.)
    np.testing.assert_array_equal(out.samples[0], q)
    initial = r.run(current_state=tf.ones_like(q), seed=(918,122), step_size=.2, num_leapfrog_steps=3)
    np.testing.assert_array_equal(initial.trace['first_failure_substep'], [[0]*4])
    np.testing.assert_array_equal(initial.trace['first_failure_stage'], [[1]*4])


@pytest.mark.parametrize('family', ['hmc', 'mams'])
def test_random_duration_rows_keep_own_terminal_value_score_and_status(family):
    # Deliberately batch-coupled diagnostic callback exposes whether evaluations
    # after a row finishes overwrite its saved terminal target state.
    class BatchDiagnostic(GaussianTarget):
        def log_prob_and_grad(self, q):
            value, score = super().log_prob_and_grad(q)
            return value + tf.reduce_sum(q), score
        def target_status_telemetry(self, q):
            return {'status_code': tf.zeros([4], tf.int32),
                    'valid_pre_regularized_score': tf.ones([4], tf.bool),
                    'floor_count_value': tf.zeros([4], tf.int32),
                    'batch_sum': tf.fill([4], tf.reduce_sum(q))}
    q = tf.zeros([4,2], tf.float64)
    r = runner(family=family, count=1, target=BatchDiagnostic(), status='per_chain_step')
    _, trace = r.one_step(q, tf.constant([83, 122]), tf.constant(.2, tf.float64), tf.constant(9))
    steps = trace['trajectory_steps'].numpy()
    assert len(set(steps)) > 1
    position, momentum = q, trace['initial_momentum']
    _, score = r.adapter.log_prob_and_grad(position)
    expected = np.zeros(4); sums = np.zeros(4)
    for i in range(max(steps)):
        if family == 'hmc':
            middle = momentum + .1*score
        else:
            middle, _ = sphere_force_substep(momentum, score, tf.constant(.1, tf.float64))
        position = tf.where((i < steps)[:,None], position + .2*middle, position)
        value, next_score = r.adapter.log_prob_and_grad(position)
        if family == 'hmc':
            end = middle + .1*next_score
        else:
            end, _ = sphere_force_substep(middle, next_score, tf.constant(.1, tf.float64))
        momentum = tf.where((i < steps)[:,None], end, momentum)
        score = tf.where((i < steps)[:,None], next_score, score)
        for row in range(4):
            if i+1 == steps[row]:
                expected[row] = float(value[row]); sums[row] = float(tf.reduce_sum(position))
    np.testing.assert_allclose(trace['proposed_target_log_prob'], expected, atol=1e-14)
    np.testing.assert_allclose(trace['proposed_target_status_telemetry']['batch_sum'], sums, atol=1e-14)

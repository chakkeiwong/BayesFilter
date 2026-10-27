"""Diagnostic rejection/selection mechanisms, not posterior evidence."""
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_modern_kernels import ReusableModernHMCRunner
from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from tests.test_hmc_candidate_set_execution import GaussianTarget, make_binding, execution_config


class DomainFailure(GaussianTarget):
    """Artificial domain failure: the Gaussian law is not claimed here."""
    def log_prob_and_grad(self, q):
        value, score = super().log_prob_and_grad(q)
        valid = tf.reduce_max(tf.abs(q), -1) < 1.5
        nan = tf.constant(float('nan'), tf.float64)
        return tf.where(valid, value, nan), tf.where(valid[..., None], score, nan)

    def target_status_telemetry(self, q):
        valid = tf.reduce_max(tf.abs(q), -1) < 1.5
        return dict(status_code=tf.where(valid, 0, 1), valid_pre_regularized_score=valid,
                    floor_count_value=tf.zeros_like(valid, tf.int32))


def setup(target=None, family='hmc', trajectory='uniform_duration', jit=False, count=16):
    target = DomainFailure() if target is None else target
    profile = HMCTransitionProfile(family, trajectory)
    policy = HMCReplicatedAcceptancePolicy(trial_num_results=64, discarded_prefix=0,
        base_repetitions=16, max_repetitions=128, max_candidates=24,
        search_family_alpha=.05, verification_family_alpha=.05,
        diagnostic_family_alpha=.05, temporal_tolerance=.1,
        qualification_mode='lower_bound', qualification_region=(.55,1.))
    starts = {'initial_position': tf.zeros([4,2], tf.float64)} if getattr(target, 'start_at_zero', False) else {}
    binding = make_binding(target=target, **starts, config=execution_config(chain_mode='batched',
        transition_profile=profile, num_warmup_steps=0, use_xla=jit, acceptance_policy=policy,
        measurement_num_results=64, verification_num_results=64,
        non_xla_reason='' if jit else 'CPU rejection mechanism diagnostic'))
    runner = ReusableModernHMCRunner(binding._active_adapter, binding.initial_active_state,
        profile=profile, num_results=count, use_xla=jit, target_status_trace_policy='per_chain_step')
    return binding, runner


@pytest.mark.parametrize('family', ['hmc', 'mams'])
@pytest.mark.parametrize('trajectory', ['fixed', 'uniform_duration'])
@pytest.mark.parametrize('jit', [False, True])
def test_numerical_rejections_are_zero_evidence_with_intact_selection(family, trajectory, jit):
    binding, runner = setup(family=family, trajectory=trajectory, jit=jit)
    out = runner.run(current_state=binding.initial_active_state, seed=(20261027, 81),
                     step_size=1.1, num_leapfrog_steps=5)
    rejected = out.trace['numerical_proposal_rejected']
    assert bool(tf.reduce_any(rejected))
    assert bool(tf.reduce_any(~rejected))  # Both kinds in the same batched trace.
    assert not bool(tf.reduce_any(rejected & out.trace['is_accepted']))
    assert bool(tf.reduce_all(tf.boolean_mask(out.trace['log_accept_ratio'], rejected) == -float('inf')))
    assert bool(tf.reduce_all(tf.boolean_mask(tf.exp(tf.minimum(out.trace['log_accept_ratio'], 0.)), rejected) == 0.))
    assert bool(tf.reduce_any(~tf.math.is_finite(out.trace['raw_log_accept_ratio'])))
    assert binding.health_failures(binding.initial_active_state, out.samples, out.trace) == ()
    assessed = binding.analyze_trial(binding.initial_active_state, out.samples, out.trace)
    assert not assessed['hard_vetoes']


class FailureAwayFromZero(GaussianTarget):
    start_at_zero = True
    def log_prob_and_grad(self, q):
        value, score = super().log_prob_and_grad(q)
        return value, tf.where(tf.reduce_any(q != 0., -1)[..., None],
                              tf.constant(float('nan'), tf.float64), score)


@pytest.fixture(scope='module')
def first_rejection():
    binding, runner = setup(target=FailureAwayFromZero(), count=3)
    initial = tf.zeros_like(binding.initial_active_state)
    out = runner.run(current_state=initial, seed=(20261027, 83), step_size=.3, num_leapfrog_steps=3)
    assert bool(tf.reduce_all(out.trace['numerical_proposal_rejected']))
    assert binding.health_failures(initial, out.samples, out.trace) == ()
    return binding, initial, out


@pytest.mark.parametrize('damage', ['selected_value', 'selected_score', 'current_score',
    'current_valid', 'accepted', 'effective_ratio', 'classification', 'stage',
    'substep', 'position', 'missing_score', 'score_dtype'])
def test_first_transition_corruption_cannot_hide_behind_rejection(first_rejection, damage):
    binding, initial, out = first_rejection
    trace = dict(out.trace); samples = out.samples
    fields = {'selected_value': ('target_log_prob', 42.),
        'selected_score': ('accepted_target_score', [42., 42.]),
        'current_score': ('previous_target_score', [float('nan')]*2),
        'current_valid': ('previous_target_valid', False),
        'accepted': ('is_accepted', True), 'effective_ratio': ('log_accept_ratio', 0.),
        'classification': ('numerical_proposal_rejected', False),
        'stage': ('first_failure_stage', 0), 'substep': ('first_failure_substep', 0)}
    if damage in fields:
        name, value = fields[damage]
        trace[name] = tf.tensor_scatter_nd_update(trace[name], [[0, 0]],
                                                tf.constant([value], trace[name].dtype))
    elif damage == 'position':
        samples = tf.tensor_scatter_nd_update(samples, [[0, 0]], tf.constant([[42., 42.]], tf.float64))
    elif damage == 'missing_score':
        del trace['accepted_target_score']
    else:
        trace['accepted_target_score'] = tf.cast(trace['accepted_target_score'], tf.float32)
    if damage in ('missing_score', 'score_dtype'):
        with pytest.raises(ValueError, match='modern selection trace'):
            binding.health_failures(initial, samples, trace)
    else:
        assert binding.health_failures(initial, samples, trace)


@pytest.mark.parametrize('family', ['hmc', 'mams'])
def test_invalid_initial_score_is_execution_failure(family):
    binding, runner = setup(target=FailureAwayFromZero(), family=family, count=2)
    initial = tf.ones_like(binding.initial_active_state)
    out = runner.run(current_state=initial, seed=(20261027, 84),
                     step_size=.3, num_leapfrog_steps=3)
    assert not bool(tf.reduce_any(out.trace['numerical_proposal_rejected']))
    assert 'invalid_current_target' in binding.health_failures(initial, out.samples, out.trace)


def test_finite_status_failure_before_later_nan_remains_veto():
    class StatusBeforeOverflow(GaussianTarget):
        start_at_zero = True
        def log_prob_and_grad(self, q):
            value, score = super().log_prob_and_grad(q)
            # First proposed point is finite with invalid status; the next
            # kick overflows. Later NaN must not erase the earlier veto.
            score = tf.where(tf.reduce_any(q != 0., -1)[..., None],
                             tf.constant(1.7e308, tf.float64), score)
            return value, score
        def target_status_telemetry(self, q):
            bad = tf.reduce_any(q != 0., -1)
            return dict(status_code=tf.cast(bad, tf.int32), valid_pre_regularized_score=~bad,
                        floor_count_value=tf.zeros_like(bad, tf.int32))
    binding, runner = setup(target=StatusBeforeOverflow(), trajectory='fixed', count=1)
    initial = tf.zeros_like(binding.initial_active_state)
    out = runner.run(current_state=initial, seed=(20261027, 85), step_size=1., num_leapfrog_steps=3)
    assert bool(tf.reduce_all(out.trace['first_failure_stage'] == 4))
    assert bool(tf.reduce_all(out.trace['first_failure_score_finite']))
    assert not bool(tf.reduce_any(out.trace['numerical_proposal_rejected']))
    assert bool(tf.reduce_any(~tf.math.is_finite(out.trace['raw_log_accept_ratio'])))
    assert binding.health_failures(initial, out.samples, out.trace)


@pytest.mark.parametrize('family', ['hmc', 'mams'])
def test_rejection_split_replay_preserves_all_trace_fields(family):
    binding, whole = setup(family=family, count=8)
    _, part = setup(family=family, count=4)
    kw = dict(seed=(20261027, 86), step_size=1.1, num_leapfrog_steps=5)
    full = whole.run(current_state=binding.initial_active_state, **kw)
    first = part.run(current_state=binding.initial_active_state, **kw)
    last = part.run(current_state=first.samples[-1], transition_offset=4, **kw)
    joined = tf.concat([first.samples, last.samples], 0)
    assert bool(tf.reduce_all(joined == full.samples))
    combined = tf.nest.map_structure(lambda a,b: tf.concat([a,b],0), first.trace,last.trace)
    for a,b in zip(tf.nest.flatten(full.trace), tf.nest.flatten(combined)):
        equal = a == b
        if a.dtype.is_floating: equal |= tf.math.is_nan(a) & tf.math.is_nan(b)
        assert bool(tf.reduce_all(equal))
    assert binding.health_failures(binding.initial_active_state, full.samples, full.trace) == ()


def test_recomputed_prechoice_value_is_not_treated_as_a_persistent_cache():
    # A rowwise re-evaluation can differ at roundoff from the preceding
    # graph output. The current transition must select its actual input value.
    binding, runner = setup(target=FailureAwayFromZero(), count=2)
    initial = tf.zeros_like(binding.initial_active_state)
    out = runner.run(current_state=initial, seed=(20261027, 87), step_size=.3, num_leapfrog_steps=3)
    trace = dict(out.trace)
    for key in ('previous_target_log_prob', 'target_log_prob'):
        trace[key] = tf.tensor_scatter_nd_update(trace[key], [[1,0]], tf.constant([1e-16],tf.float64))
    assert binding.health_failures(initial, out.samples, trace) == ()

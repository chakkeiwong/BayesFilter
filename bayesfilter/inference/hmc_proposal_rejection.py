"""Opt-in numerical rejection for exact-score, identity-mass batched HMC.

The target remains unchanged. A failed numerical trajectory has probability zero
and retains its current cache through TFP MetropolisHastings. Every visited
position and half/full momentum must be representable; these tests are symmetric
under momentum reversal on successful kick/drift paths. Failed rows use the
current point as a computational placeholder and can never become eligible again.
No exception is caught here: programming, shape and device errors still escape.
"""
from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

POLICY = 'sticky_numerical_rejection_v1'
Results = namedtuple('GuardedExactHMCResults', (
    'log_acceptance_correction', 'target_log_prob', 'grads_target_log_prob',
    'initial_momentum', 'final_momentum', 'step_size', 'num_leapfrog_steps',
    'seed', 'failure_code', 'failure_step', 'status', 'failure_position',
    'failure_value_nonfinite', 'failure_score_nonfinite'))


def _finite_rows(x):
    return tf.reduce_all(tf.math.is_finite(x), axis=-1)


def guarded_leapfrog(evaluate, q, p, value, score, status, step, count):
    """Native batched kick/drift integration with a sticky first failure code.

    evaluate returns unmodified value, score and deterministic per-row status.
    Codes supplied by a target are preserved; 90 means arithmetic in the
    integrator, 91 means an unclassified nonfinite target/score. Starting-cache
    corruption is code 92 and is an execution veto in health assessment.
    """
    original_q, original_value, original_score = q, value, score
    valid = (_finite_rows(q) & _finite_rows(p) & tf.math.is_finite(value)
             & _finite_rows(score) & (status['status_code'] == 0)
             & status['valid_pre_regularized_score'])
    code = tf.where(valid, 0, 92)
    first = tf.where(valid, -1, 0)
    half = tf.cast(.5, q.dtype) * step

    def body(i, q, p, value, score, code, first, status, failure_q, bad_value, bad_score):
        # One complete velocity-Verlet step. The half momentum is the same
        # (up to sign) in the reversed path, so it is part of the guard too.
        ph = p + half * score
        trial = q + step * ph
        arithmetic = _finite_rows(ph) & _finite_rows(trial)
        eligible = (code == 0) & arithmetic
        safe_q = tf.where(eligible[:, None], trial, original_q)
        next_value, next_score, next_status = tf.cond(tf.reduce_any(eligible),
            lambda: evaluate(safe_q), lambda: (original_value, original_score, status))
        target_ok = (tf.math.is_finite(next_value) & _finite_rows(next_score)
                     & (next_status['status_code'] == 0)
                     & next_status['valid_pre_regularized_score'])
        safe_score = tf.where((eligible & target_ok)[:, None], next_score,
                              original_score)
        following_p = ph + half * safe_score
        momentum_ok = _finite_rows(following_p)
        reason = tf.where(~arithmetic | ~momentum_ok, 90,
            tf.where(next_status['status_code'] != 0, next_status['status_code'],
                     tf.where(target_ok, 0, 91)))
        newly_failed = (code == 0) & (reason != 0)
        failure_q = tf.where(newly_failed[:,None], trial, failure_q)
        bad_value = tf.where(newly_failed, ~tf.math.is_finite(next_value), bad_value)
        bad_score = tf.where(newly_failed, ~_finite_rows(next_score), bad_score)
        code = tf.where(newly_failed, reason, code)
        first = tf.where(newly_failed, i + 1, first)
        good = code == 0
        # Placeholders never become a proposal or a target observation.
        return (i + 1, tf.where(good[:, None], trial, original_q),
            tf.where(good[:, None], following_p, tf.zeros_like(p)),
            tf.where(good, next_value, original_value),
            tf.where(good[:, None], next_score, original_score), code, first,
            {k: tf.where(good, next_status[k], status[k]) for k in status},
            failure_q, bad_value, bad_score)

    _, q, p, value, score, code, first, status, failure_q, bad_value, bad_score = tf.while_loop(
        lambda i, *_: i < count, body,
        (tf.constant(0), q, p, value, score, code, first, status, q,
         tf.zeros_like(valid), tf.zeros_like(valid)),
        parallel_iterations=1)
    return q, p, value, score, code, first, status, failure_q, bad_value, bad_score


def guarded_exact_hmc(adapter, step_size, num_leapfrog_steps):
    """TFP MH wrapper; only explicit adapter opt-in selects this kernel."""
    evaluate = adapter.log_prob_and_grad_status

    class Proposal(tfp.mcmc.TransitionKernel):
        @property
        def is_calibrated(self): return False

        @property
        def parameters(self):
            return dict(adapter=adapter, step_size=step_size,
                        num_leapfrog_steps=num_leapfrog_steps)

        def bootstrap_results(self, init_state):
            value, score, status = evaluate(init_state)
            zeros = tf.zeros_like(init_state)
            return Results(tf.zeros_like(value), value, [score], [zeros], [zeros],
                tf.fill(tf.shape(value), step_size), tf.fill(tf.shape(value), tf.cast(num_leapfrog_steps,tf.int32)), tf.zeros([2], tf.int32),
                tf.zeros_like(value, tf.int32), tf.fill(tf.shape(value), -1), status,
                init_state, tf.zeros_like(value,tf.bool), tf.zeros_like(value,tf.bool))

        def one_step(self, current_state, previous_kernel_results, seed=None):
            seed = samplers.sanitize_seed(seed)
            # Match TFP's one-state-part momentum stream.
            part_seed = samplers.split_seed(seed, n=1)[0]
            momentum = samplers.normal(tf.shape(current_state),
                                       dtype=current_state.dtype, seed=part_seed)
            old = previous_kernel_results
            q, p, value, score, code, first, status, failure_q, bad_value, bad_score = guarded_leapfrog(
                evaluate, current_state, momentum, old.target_log_prob,
                old.grads_target_log_prob[0], old.status, step_size,
                num_leapfrog_steps)
            correction = .5 * tf.reduce_sum(momentum**2 - p**2, axis=-1)
            energy_ok = tf.math.is_finite(correction) & tf.math.is_finite(tf.add_n([value, -old.target_log_prob, correction]))
            first = tf.where((code == 0) & ~energy_ok, num_leapfrog_steps, first)
            code = tf.where((code == 0) & ~energy_ok, 90, code)
            correction = tf.where(code == 0, correction,
                                  tf.constant(-float('inf'), value.dtype))
            return q, Results(correction, value, [score], [momentum], [p],
                tf.fill(tf.shape(value), step_size), tf.fill(tf.shape(value), tf.cast(num_leapfrog_steps,tf.int32)), seed, code, first, status,
                failure_q, bad_value, bad_score)

    return tfp.mcmc.MetropolisHastings(Proposal())


def guarded_health_failures(initial, samples, trace):
    """Check rejection/cache invariants; never discard rejected observations."""
    shape = samples.shape[:2]
    if samples.dtype != tf.float64 or samples.shape.rank != 3 or samples.shape[1:] != initial.shape:
        raise ValueError('guarded samples require static float64 [draw,chain,parameter]')
    for name in ('is_accepted', 'log_accept_ratio', 'target_log_prob',
                 'proposed_target_log_prob', 'target_score_finite',
                 'accepted_target_score_finite', 'proposal_failure_code',
                 'proposal_failure_step'):
        if name not in trace or trace[name].shape != shape:
            raise ValueError('missing or misaligned guarded health trace: ' + name)
    previous = tf.concat([initial[None], samples[:-1]], axis=0)
    accepted = trace['is_accepted']
    ratio = trace['log_accept_ratio']
    code = trace['proposal_failure_code']
    failed = code != 0
    failures = []
    def require(ok, reason):
        if not bool(tf.reduce_all(ok)): failures.append(reason)
    require(tf.math.is_finite(samples), 'nonfinite_state')
    require(tf.math.is_finite(initial), 'nonfinite_initial_state')
    require(tf.math.is_finite(trace['target_log_prob']), 'nonfinite_target_log_prob')
    require(trace['accepted_target_score_finite'], 'nonfinite_current_score')
    require(code != 92, 'invalid_current_cache')
    require(~failed | (~accepted & tf.math.is_inf(ratio) & (ratio < 0.)),
            'failed_proposal_not_rejected')
    require(tf.math.is_finite(ratio) | (failed & ~accepted & (ratio == -float('inf'))),
            'unclassified_nonfinite_ratio')
    require(tf.equal(samples, tf.where(accepted[..., None], trace['proposed_state'], previous)),
            'metropolis_state_mismatch')
    current = trace['target_status_telemetry']
    require((current['status_code'] == 0) & current['valid_pre_regularized_score'],
            'target_status_telemetry_failed')
    proposed = trace['proposed_target_status_telemetry']
    require(failed | ((proposed['status_code'] == 0) & proposed['valid_pre_regularized_score']),
            'unclassified_proposed_status')
    require(failed | trace['target_score_finite'], 'unclassified_proposed_score')
    for name in ('proposed_state', 'initial_momentum', 'final_momentum'):
        if trace[name].shape != samples.shape: raise ValueError(name + ' shape mismatch')
        require(failed | tf.reduce_all(tf.math.is_finite(trace[name]), -1),
                'unclassified_nonfinite_' + name)
    require(failed | tf.math.is_finite(trace['proposed_target_log_prob']),
            'unclassified_nonfinite_proposed_target')
    # Cached value must be bitwise unchanged on every recorded rejection;
    # the first cache is validated by the kernel and bootstrap qualification.
    require(accepted[1:] | tf.equal(trace['target_log_prob'][1:], trace['target_log_prob'][:-1]),
            'rejected_value_cache_changed')
    if 'divergence' in trace:
        require(~trace['divergence'], 'native_divergence_positive')
    return tuple(failures)

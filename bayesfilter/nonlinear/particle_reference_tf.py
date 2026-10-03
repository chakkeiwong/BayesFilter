"""Conventional SIR likelihood reference using TensorFlow Probability's SMC.

Value-only reference: no gradient or HMC contract. The model supplies vector
state distributions with batch shape [B] initially and [N,B] conditionally.
Weights implement BayesFilter Chapter19 eq:bf-pf-sis-recursion. In particular,
proposal p/q ratios contribute to the normalizing constant before any weight
normalization. TFP's high-level particle_filter uses a different, self-normalized
ratio; this module uses its lower-level SequentialMonteCarlo kernel instead.

The caller owns device/memory policy and wraps the complete call in tf.function
with stable shapes. There is one TensorFlow date loop and native batched particle
operations. No NumPy, per-particle Python loop, or Python callback is used.
"""
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


def particle_reference(
    observations, initial_prior, transition_fn, observation_fn, *, num_particles,
    seed, initial_proposal=None, proposal_fn=None, resample_ess_fraction=0.5,
):
    """Estimate p(y[0:T]) with explicit first-observation and proposal semantics.

    observations has shape [T,B,...], with fixed positive T and B. The initial
    distribution has batch [B], event [D]; transition_fn(t,x) describes
    x[t+1]|x[t] with batch [N,B], event [D]. observation_fn(t,x) describes
    y[t]|x[t]. proposal_fn has the transition signature and may capture y[t+1].
    Initial proposal, when supplied, has the same shape/support contract as
    the prior. Proposal support must cover the target, including Jacobians.

    The log likelihood is the log of a conventional importance-weighted SIR
    estimate. It is not an unbiased log-likelihood estimate or a score. Normal
    Monte Carlo stability checks are required before treating it as a reference.
    Seed is an explicit stateless int32 pair. Particle draws are batched; B
    rows have independent RNG draws in that stream, not repeated seeds.

    Pre-resampling moments/ESS are recorded at every date. Terminal particles
    and normalized weights are AFTER the last native resampling decision.
    unique_parent_count describes immediate resampling, not full genealogy.
    Invalid rows return valid=False and -inf log likelihood. Placeholder carry
    values only isolate rejected rows; their moments are never eligible output.
    """
    if not 0 <= resample_ess_fraction <= 1:
        raise ValueError("ESS resampling fraction must be in [0,1]")
    if num_particles < 1:
        raise ValueError("num_particles must be positive")
    observations = tf.convert_to_tensor(observations)
    dates, batch = observations.shape[:2]
    if dates is None or batch is None or dates < 1 or batch < 1:
        raise ValueError("fixed positive observation/date batch dimensions required")
    seed = tf.ensure_shape(tf.convert_to_tensor(seed, tf.int32), [2])
    initial_seed, loop_seed = tf.unstack(tf.random.experimental.stateless_split(seed, 2))
    initial = initial_prior if initial_proposal is None else initial_proposal
    particles = initial.sample(num_particles, seed=initial_seed)
    if particles.shape.rank != 3 or particles.shape[:2] != (num_particles, batch):
        raise ValueError("vector-state distributions must produce particles [N,B,D]")
    dtype = particles.dtype
    log_n = tf.math.log(tf.cast(num_particles, dtype))
    uniform = tf.fill([num_particles, batch], -log_n)

    def weight_record(points, log_weights):
        # LSE is defined when at least one weight is finite, even if the first
        # weight is zero (-inf). TFP's first-entry subtraction is not; retain
        # the mathematically equivalent normalizer in extra for our report.
        normalizer = tf.reduce_logsumexp(log_weights, axis=0)
        valid = (tf.math.is_finite(normalizer)
                 & tf.reduce_all(tf.math.is_finite(points), axis=[0, 2])
                 & ~tf.reduce_any(tf.math.is_nan(log_weights), axis=0)
                 & ~tf.reduce_any(log_weights == tf.cast(float("inf"), dtype), axis=0))
        safe = tf.where(valid[None, :], log_weights, uniform)
        normalized = tf.nn.log_softmax(safe, axis=0)
        weights = tf.exp(normalized)
        safe_points = tf.where(valid[None, :, None], points, tf.zeros_like(points))
        mean = tf.einsum("nb,nbd->bd", weights, safe_points)
        centered = safe_points - mean[None]
        covariance = tf.einsum("nb,nbi,nbj->bij", weights, centered, centered)
        record = {
            "increment": tf.where(valid, normalizer, tf.cast(-float("inf"), dtype)),
            "valid": valid,
            "ess": tf.exp(-tf.reduce_logsumexp(2 * normalized, axis=0)),
            "mean": mean, "covariance": covariance,
        }
        return tfp.experimental.mcmc.WeightedParticles(safe_points, safe, record)

    initial_weights = uniform
    if initial_proposal is not None:
        initial_weights += initial_prior.log_prob(particles) - initial_proposal.log_prob(particles)
    initial_weights += observation_fn(tf.constant(0, tf.int32), particles).log_prob(observations[0])
    state = weight_record(particles, initial_weights)

    def propose(step, previous, seed):
        # TFP evaluates a proposal then selects the original state at step0.
        # The one-date case has no transition and must not evaluate one. For
        # T>1 the unused initial proposal is harmless: callbacks are pure and
        # the native kernel discards its state, weights and diagnostics.
        if dates == 1:
            return previous
        prior = transition_fn(step, previous.particles)
        proposal = prior if proposal_fn is None else proposal_fn(step, previous.particles)
        following = proposal.sample(seed=seed)
        weights = previous.log_weights
        if proposal_fn is not None:
            weights += prior.log_prob(following) - proposal.log_prob(following)
        weights += observation_fn(step + 1, following).log_prob(observations[step + 1])
        return weight_record(following, weights)

    def needs_resampling(weighted, particles_dim):
        # TFP's choose expands trailing event axes. Preserve the particle axis
        # as a singleton, as its native ess_below_threshold does; returning
        # only [B] can broadcast over particles instead of independent batches.
        mask = weighted.extra["valid"] & (weighted.extra["ess"] < resample_ess_fraction * num_particles)
        return tf.expand_dims(mask, axis=particles_dim)

    kernel = tfp.experimental.mcmc.SequentialMonteCarlo(
        propose_and_update_log_weights_fn=propose,
        resample_fn=tfp.experimental.mcmc.resample_systematic,
        resample_criterion_fn=needs_resampling, particles_dim=0, unbiased_gradients=False)
    results = kernel.bootstrap_results(state)
    history = {key: tf.zeros([dates, *value.shape], value.dtype) for key, value in state.extra.items()}
    history["resampled"] = tf.zeros([dates, batch], tf.bool)
    history["unique_parent_count"] = tf.zeros([dates, batch], tf.int32)

    def step_body(t, loop_seed, state, results, history):
        step_seed, next_seed = tf.unstack(tf.random.experimental.stateless_split(loop_seed, 2))
        state, results = kernel.one_step(state, results, seed=step_seed)
        ordered = tf.sort(results.parent_indices, axis=0)
        count = 1 + tf.reduce_sum(tf.cast(ordered[1:] != ordered[:-1], tf.int32), axis=0)
        record = dict(state.extra, resampled=tf.squeeze(needs_resampling(state, 0), axis=0),
                      unique_parent_count=count)
        history = {key: tf.tensor_scatter_nd_update(value, tf.reshape(t, [1, 1]), record[key][None])
                   for key, value in history.items()}
        return t + 1, next_seed, state, results, history

    _, _, terminal, _, history = tf.while_loop(
        lambda t, *_: t < dates, step_body,
        (tf.constant(0, tf.int32), loop_seed, state, results, history), parallel_iterations=1)
    valid = tf.reduce_all(history["valid"], axis=0)
    failure_date = tf.reduce_min(tf.where(history["valid"], dates, tf.range(dates)[:, None]), axis=0)
    # Stop-gradient makes the value-only boundary explicit. Native resampling
    # paths do not define the deterministic score required by ordinary HMC.
    return tf.nest.map_structure(tf.stop_gradient, {
        "log_likelihood": tf.where(valid, tf.reduce_sum(history["increment"], axis=0),
                                   tf.cast(-float("inf"), dtype)),
        "valid": valid, "failure_date": tf.where(valid, -1, failure_date),
        "history": history, "particles": terminal.particles,
        "log_weights": terminal.log_weights,
    })

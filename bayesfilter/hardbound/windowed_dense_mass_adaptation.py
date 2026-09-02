"""Windowed full-joint dense mass matrix adaptation for hardbound HMC.

Stan-style windowed warmup for the hardbound G2.3 route: an initial fast
buffer that discards early off-posterior draws, progressively doubling slow
windows that rebuild the metric at each boundary, and a final fast buffer that
tunes the step size against a frozen metric.

Two properties distinguish this route from
`bayesfilter.hardbound.dense_mass_matrix_adaptation`:

*   The metric is **full-joint** over the concatenated state (337 coordinates
    for G2.3), not block-diagonal per state part.  The sampler therefore runs
    on a single flattened state vector and the target log-probability is
    reconstructed by splitting that vector.  Block-dense adaptation cannot
    represent theta/eta cross-correlation at all; whether that coupling is what
    stalls the block route is an open question, not an established result.
*   The empirical covariance is **shrunk toward its own diagonal** before
    factorization, because 337 coordinates carry 56,953 free covariance
    entries and a single slow window supplies far fewer draws than that.

Scope: this module serves the hardbound test harness.  It is a chain runner in
the repository tuning taxonomy -- it executes a supplied window/shrinkage
configuration and holds no canonical artifact authority, so it emits no mass
artifact and no tuning signature.
"""

from __future__ import annotations

import dataclasses
import math

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.distributions import (
    mvn_precision_factor_linop as mvn_pfl,
)

from bayesfilter.inference.hmc_tuning import (
    WindowedMassAdaptationConfig,
    build_windowed_warmup_schedule,
)


def _static_part_sizes(initial_states):
    """Per-part event shapes and flat sizes, as Python ints.

    Raises when an event shape is not statically known: the window loop
    rebuilds the kernel per window and relies on a stable `input_signature`,
    which a dynamic event shape would defeat.
    """
    shapes, sizes = [], []
    for i, part in enumerate(initial_states):
        event_shape = part.shape[1:]
        if not event_shape.is_fully_defined():
            raise ValueError(
                f"state part {i} needs a static event shape, got {part.shape}")
        dims = [int(d) for d in event_shape.as_list()]
        shapes.append(dims)
        sizes.append(math.prod(dims) if dims else 1)
    return shapes, sizes


def _merge_truncated_tail_window(windows):
    """Absorb a short trailing slow window into its predecessor.

    `build_windowed_warmup_schedule` doubles until the slow span runs out and
    emits whatever remains as a final short window: a 4000-step G2.3 warmup
    yields `..., 800, 1600, 700`.  That trailing stub is the window whose metric
    the sampling phase freezes, so the sampler would run under a covariance
    estimated from 700 steps while the 1600-step window immediately before it
    had more than twice the draws.  In the n<<p regime that is a strict loss:
    337 coordinates carry 56,953 covariance entries, and precision in the frozen
    metric is exactly what the route depends on.

    Merging extends the last full window to the end of the slow phase, so the
    G2.3 schedule becomes `..., 800, 2300` and the frozen metric is built from
    2,300 steps instead of 700.  The justification is the argument above --
    never freeze a metric estimated from fewer draws than an earlier window
    already achieved -- not an appeal to Stan: Stan is cited in the plan only
    for the buffer and doubling constants, and its `windowed_adaptation` source
    has not been read here, so no claim of equivalence to Stan's tail handling
    is made.

    The shared builder is left untouched so other callers keep their schedule;
    this is a hardbound-route post-processing step.
    """
    slow_positions = [i for i, w in enumerate(windows) if w.kind == "slow"]
    if len(slow_positions) < 2:
        return tuple(windows)
    last, prev = slow_positions[-1], slow_positions[-2]
    last_w, prev_w = windows[last], windows[prev]
    if (last_w.end - last_w.start) >= (prev_w.end - prev_w.start):
        return tuple(windows)

    merged = dataclasses.replace(prev_w, end=last_w.end)
    kept = list(windows[:prev]) + [merged] + list(windows[last + 1:])
    # Reindex so the recorded window indices stay contiguous.
    return tuple(dataclasses.replace(w, index=i) for i, w in enumerate(kept))


def _sample_covariance(draws):
    """Two-pass sample covariance of `draws` shaped `[n, dim]`.

    Mathematically the estimator `welford_covariance` in
    `bayesfilter.inference.hmc_tuning` returns (`m2 / (n - 1)`, symmetrized),
    computed in TensorFlow instead of a host-side NumPy recursion so the
    inference path stays on the repository backend.  This is not Welford's
    recursion and must not be reported as such; the two-pass form is used
    because the centered product is the numerically better-conditioned route
    when all draws are already resident.
    """
    n = tf.cast(tf.shape(draws)[0], draws.dtype)
    centered = draws - tf.reduce_mean(draws, axis=0, keepdims=True)
    cov = tf.matmul(centered, centered, transpose_a=True) / (n - 1.0)
    return 0.5 * (cov + tf.linalg.adjoint(cov))


def _shrink_toward_diagonal(cov, shrinkage):
    """Shrink `cov` toward its own diagonal: `(1-s) * cov + s * diag(cov)`.

    Only the off-diagonal entries move, so every marginal variance is
    preserved exactly and the result is invariant to per-coordinate rescaling.
    That matters here: the G2.3 raw chart does not have unit-scale marginals,
    so a `(1-s) * cov + s * I` target would swamp the small-variance
    coordinates and destroy the preconditioner rather than regularize it.  The
    diagonal target instead attacks the actual small-sample problem, which is
    the off-diagonal correlations -- each marginal variance is estimated from
    all pooled draws, while the 56,953 covariance entries are not.
    """
    diag = tf.linalg.diag_part(cov)
    return (1.0 - shrinkage) * cov + shrinkage * tf.linalg.diag(diag)


def _metric_diagnostics(cov):
    """Explanatory-only spectrum summary of a proposed metric.

    Emitted rather than discarded so that a convergence failure can be read
    against the conditioning of the metric that produced it.
    """
    eigenvalues = tf.linalg.eigvalsh(cov)
    smallest = eigenvalues[0]
    largest = eigenvalues[-1]
    return {
        "min_eigenvalue": smallest,
        "max_eigenvalue": largest,
        "condition_number": largest / smallest,
        "min_variance": tf.reduce_min(tf.linalg.diag_part(cov)),
        "max_variance": tf.reduce_max(tf.linalg.diag_part(cov)),
    }


def _ridged_cholesky(cov, ridge_rel):
    """Cholesky factor after a scale- and dimension-aware relative ridge.

    The ridge is `ridge_rel * trace(cov) / dim` on the diagonal, following the
    same relative convention as `dense_mass_matrix_adaptation.RIDGE_REL` so the
    guard does not silently expire as the posterior scale or dimension changes.
    An early slow window supplies fewer pooled draws than the 337 coordinates,
    so the estimate is genuinely rank-deficient there and this guard does fire
    rather than being decorative.

    Note this is deliberately *not* `WindowedMassAdaptationConfig.
    covariance_jitter`, which is an absolute `cov + jitter * I` ridge.  An
    absolute ridge cannot serve this target: G2.3 marginal variances span
    roughly 1.9e4 in the raw chart, so one additive constant is simultaneously
    negligible for the large coordinates and dominant for the small ones.
    """
    dim = tf.shape(cov)[-1]
    ridge = ridge_rel * tf.linalg.trace(cov) / tf.cast(dim, cov.dtype)
    ridged = tf.linalg.set_diag(cov, tf.linalg.diag_part(cov) + ridge)
    return tf.linalg.cholesky(ridged)


def _make_flat_target(log_prob_fn, part_shapes, part_sizes):
    """Adapt a per-part target to a single concatenated state vector.

    The full-joint metric requires one flat vector, so the sampler state is the
    concatenation of the parts and the target splits it back before calling the
    original `log_prob_fn`.  No reweighting is involved: concatenation is a
    fixed permutation-free reshape, so the returned function is the same
    density on a relabelled coordinate chart.
    """
    offsets = []
    cursor = 0
    for size in part_sizes:
        offsets.append(cursor)
        cursor += size

    def flat_target(flat_state):
        parts = []
        for shape, size, offset in zip(part_shapes, part_sizes, offsets):
            block = flat_state[..., offset:offset + size]
            leading = tf.shape(flat_state)[:-1]
            parts.append(tf.reshape(block, tf.concat([leading, shape], axis=0)))
        return log_prob_fn(*parts)

    return flat_target


def _momentum_for(chol, num_chains):
    """Full-joint momentum distribution from a state-covariance Cholesky factor.

    Follows the TFP convention used by `PreconditionedHamiltonianMonteCarlo` and
    `bayesfilter.hardbound.dense_mass_matrix_adaptation`: the momentum
    *precision* is the state covariance `Sigma`, so
    `precision_factor = chol(Sigma)`.  The factor is paired with the implied
    precision so the log-density normalization stays consistent.

    The state here is a single flat vector, so the distribution is one
    `BatchBroadcast` block over the chains and needs no `Reshape` bijector.

    `LinearOperatorLowerTriangular` gives a closed-form O(dim) log-abs-det,
    where `LinearOperatorFullMatrix` would take an O(dim^3) LU on every
    leapfrog step.
    """
    mvnpfl = mvn_pfl.MultivariateNormalPrecisionFactorLinearOperator(
        precision_factor=tf.linalg.LinearOperatorLowerTriangular(
            chol, is_non_singular=True),
        precision=tf.linalg.LinearOperatorFullMatrix(
            tf.matmul(chol, chol, transpose_b=True),
            is_non_singular=True, is_self_adjoint=True,
            is_positive_definite=True))
    return tfp.distributions.JointDistributionSequential([
        tfp.distributions.BatchBroadcast(mvnpfl, with_shape=[num_chains])])


def run_windowed_dense_mass_adaptation(
    target_log_prob_fn,
    initial_states,
    num_warmup_steps,
    num_samples,
    initial_step_size,
    target_accept_prob,
    seed,
    num_leapfrog_steps=10,
    jit_compile=True,
    initial_buffer=75,
    final_buffer=50,
    first_window_size=25,
    mass_shrinkage=0.1,
    ridge_rel=1.0e-6,
):
    """Run fixed-trajectory HMC with windowed full-joint dense mass adaptation.

    Returns the same dict shape as `bayesfilter.hardbound.hmc_runner.run_nuts`
    (`samples`, `divergences`, `rhat`, `ess`, `final_step_size`), plus
    `window_diagnostics` and `warmup_divergences` as explanatory-only fields.
    `divergences` counts the sampling phase only, matching the sampling-phase
    denominator the caller compares it against; warmup divergences are reported
    separately rather than folded in, since a windowed schedule deliberately
    starts off-posterior.
    """
    part_shapes, part_sizes = _static_part_sizes(initial_states)
    dtype = initial_states[0].dtype
    num_chains = int(initial_states[0].shape[0])
    dim = sum(part_sizes)

    flat_state = tf.concat(
        [tf.reshape(p, [num_chains, s])
         for p, s in zip(initial_states, part_sizes)], axis=1)
    flat_target = _make_flat_target(target_log_prob_fn, part_shapes, part_sizes)

    schedule = build_windowed_warmup_schedule(
        WindowedMassAdaptationConfig(
            warmup_steps=int(num_warmup_steps),
            initial_buffer=int(initial_buffer),
            final_buffer=int(final_buffer),
            first_window_size=int(first_window_size),
            mass_shrinkage=float(mass_shrinkage),
        ))
    schedule = _merge_truncated_tail_window(schedule)

    # Identity metric for the initial fast buffer, which is exactly what that
    # buffer is for: sample far enough from the initialization that the first
    # covariance estimate is not dominated by transient approach draws.
    identity_chol = tf.eye(dim, dtype=dtype)

    def _hmc_results(pkr):
        while not hasattr(pkr, "log_accept_ratio"):
            pkr = pkr.inner_results
        return pkr

    # One traced graph per (window length, adapting) pair.  Retracing is bounded
    # by the number of distinct window lengths in the doubling schedule (nine
    # for the G2.3 4000-step warmup) rather than by the step count, and each
    # concrete function carries an explicit static `input_signature`.  A single
    # graph with dynamic `num_results` would give the sampled output a dynamic
    # leading dimension, which XLA rejects.
    runners = {}

    def _runner(num_results, adapting):
        key = (int(num_results), bool(adapting))
        if key in runners:
            return runners[key]

        signature = [
            tf.TensorSpec([num_chains, dim], dtype),   # current state
            tf.TensorSpec([], dtype),                  # step size
            tf.TensorSpec([dim, dim], dtype),          # state-covariance factor
            tf.TensorSpec([2], tf.int32),              # stateless seed
        ]

        @tf.function(jit_compile=jit_compile, input_signature=signature)
        def _run(state, step, chol, seed_pair):
            kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=flat_target,
                step_size=step,
                num_leapfrog_steps=int(num_leapfrog_steps),
                momentum_distribution=_momentum_for(chol, num_chains),
            )
            if adapting:
                # Reinstantiating the adapter per window restarts dual averaging
                # at each metric boundary, which is what Stan does: the step
                # size tuned against the previous metric is only a warm start
                # once the metric changes.
                kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                    inner_kernel=kernel,
                    num_adaptation_steps=int(num_results),
                    target_accept_prob=tf.constant(target_accept_prob, dtype),
                )
            def _trace(_state, pkr):
                mh = _hmc_results(pkr)
                traced = {
                    "diverged": tf.math.logical_not(
                        tf.math.is_finite(mh.log_accept_ratio)),
                    "is_accepted": mh.is_accepted,
                    "log_accept_ratio": mh.log_accept_ratio,
                    # During warmup with adapter, `mh.accepted_results.step_size`
                    # is populated by `DualAveragingStepSizeAdaptation` and tracks
                    # the live dual-averaged value.  During sampling without
                    # adapter, `mh.accepted_results.step_size` remains the empty
                    # list `[]` from `bootstrap_results` (TFP hmc.py lines 748-749)
                    # and cannot be traced.  Emit the closure's `step` parameter
                    # instead, which holds the frozen final warmup step size.
                    "step_size": mh.accepted_results.step_size if adapting else step,
                }
                if adapting:
                    # Adapter-level field, one level above the Metropolis-Hastings
                    # results: the dual-averaged step size that the inner kernel
                    # adopts once `step > num_adaptation_steps`.  It exists only
                    # while the DualAveragingStepSizeAdaptation wrapper is present,
                    # which is why the warmup handoff below can read it and the
                    # non-adapting sampling phase cannot.
                    traced["log_averaging_step"] = pkr.log_averaging_step
                return traced

            return tfp.mcmc.sample_chain(
                num_results=int(num_results),
                num_burnin_steps=0,
                current_state=[state],
                kernel=kernel,
                seed=seed_pair,
                trace_fn=_trace,
            )

        runners[key] = _run
        return _run

    state = flat_state
    step = tf.constant(initial_step_size, dtype)
    chol = identity_chol
    seed_counter = int(seed)
    warmup_divergences = 0
    window_diagnostics = []

    for window in schedule:
        length = window.end - window.start
        if length <= 0:
            continue
        seed_pair = tf.constant([seed_counter, seed_counter + 1], tf.int32)
        seed_counter += 2

        draws, trace = _runner(length, adapting=True)(
            state, step, chol, seed_pair)
        window_draws = draws[0]
        state = window_draws[-1]

        # Hand off the final dual-averaged step size, not `trace["step_size"][-1]`.
        # The latter is the step size *used for* the last transition, which is one
        # step stale (`inner[i] == new[i-1]` across the trace).  The dual-averaged
        # value is what the inner kernel itself freezes at once
        # `step > num_adaptation_steps`.
        #
        # `log_averaging_step` follows the step-size structure, so it arrives either
        # as a one-element list of shape (length,) or as a tensor of shape
        # (1, length).  Flatten and take the last element so the handoff does not
        # depend on which, and reduce to a scalar to satisfy the
        # `tf.TensorSpec([], dtype)` slot in `input_signature` and the
        # `float(step.numpy())` cast at record time.
        if "log_averaging_step" not in trace:
            raise RuntimeError(
                "windowed warmup ran without the dual-averaging adapter: "
                "`log_averaging_step` is absent from the trace, so the "
                "inter-window step-size handoff has no tuned value to carry "
                "forward")
        final_log_avg = tf.nest.flatten(trace["log_averaging_step"])[0]
        step = tf.exp(tf.reshape(final_log_avg, [-1])[-1])

        window_divergences = int(
            tf.reduce_sum(tf.cast(trace["diverged"], tf.int32)))
        warmup_divergences += window_divergences

        record = {
            "index": window.index,
            "kind": window.kind,
            "start": window.start,
            "end": window.end,
            "update_mass": window.update_mass,
            "divergences": window_divergences,
            "step_size_after": float(step.numpy()),
            # Realized acceptance inside this window, against the target the
            # adapter was driving toward.  Traced already and previously
            # discarded; kept because it is the one warmup diagnostic that shows
            # whether dual averaging actually reached its target at each metric
            # boundary, rather than leaving that visible only in the final
            # step size.
            "acceptance": float(tf.reduce_mean(
                tf.cast(trace["is_accepted"], dtype)).numpy()),
        }

        if window.update_mass:
            # Per-window draws, not cumulative: this matches Stan's windowed
            # scheme and `run_windowed_mass_adaptation` in
            # `bayesfilter.inference.hmc_tuning`, which slices
            # `draws[window.start:window.end]`.  Each window is sampled under a
            # better metric than the last, so pooling would let the earliest,
            # worst-mixed draws keep contributing indefinitely.
            pooled = tf.reshape(window_draws, [length * num_chains, dim])
            cov = _sample_covariance(pooled)
            cov = _shrink_toward_diagonal(cov, mass_shrinkage)
            diagnostics = _metric_diagnostics(cov)
            chol = _ridged_cholesky(cov, ridge_rel)

            # Fail closed on a metric that is not usable.  A non-finite factor
            # would otherwise propagate silently into the next window and
            # surface only as an unexplained convergence failure.
            if not bool(tf.reduce_all(tf.math.is_finite(chol)).numpy()):
                raise FloatingPointError(
                    f"non-finite Cholesky factor at window {window.index} "
                    f"[{window.start}:{window.end}]; "
                    f"condition number "
                    f"{float(diagnostics['condition_number'].numpy()):.3e}")
            record.update(
                {k: float(v.numpy()) for k, v in diagnostics.items()})
            record["pooled_draws"] = int(length * num_chains)

        window_diagnostics.append(record)

    # Sampling phase: metric and step size both frozen, so the draws come from a
    # single time-homogeneous transition kernel and are valid MCMC output.
    sampling_seed = tf.constant([seed_counter, seed_counter + 1], tf.int32)
    draws, trace = _runner(int(num_samples), adapting=False)(
        state, step, chol, sampling_seed)
    flat_draws = draws[0]  # [num_samples, num_chains, dim]

    if not bool(tf.reduce_all(tf.math.is_finite(flat_draws)).numpy()):
        raise FloatingPointError("non-finite draws in the sampling phase")

    # Split back to the caller's part structure, preserving `sample_chain`'s
    # [num_samples, num_chains, ...] layout so `potential_scale_reduction` and
    # `effective_sample_size` see the axes they expect (sample axis 0, chain
    # axis 1) -- the same convention `run_nuts` passes them.
    samples, cursor = [], 0
    for shape, size in zip(part_shapes, part_sizes):
        block = flat_draws[..., cursor:cursor + size]
        samples.append(
            tf.reshape(block, [int(num_samples), num_chains] + shape))
        cursor += size

    # Rank-normalized split R-hat per VEHTARI-2021, matching the repository
    # standard applied in `bayesfilter.inference.neutra_hmc` and expected by A3.
    # Each `s` is shape [num_samples, num_chains, ...], which
    # `potential_scale_reduction` interprets as sample axis 0, chain axis 1.
    rhat = [tfp.mcmc.potential_scale_reduction(s, split_chains=True) for s in samples]
    if num_chains > 1:
        ess = [tfp.mcmc.effective_sample_size(s, cross_chain_dims=1)
               for s in samples]
    else:
        ess = [tfp.mcmc.effective_sample_size(s) for s in samples]

    return {
        "samples": samples,
        "divergences": int(tf.reduce_sum(
            tf.cast(trace["diverged"], tf.int32))),
        "rhat": rhat,
        "ess": ess,
        "final_step_size": trace["step_size"][-1],
        "warmup_divergences": warmup_divergences,
        "window_diagnostics": window_diagnostics,
        # Amendment A3 requires the trace to capture `is_accepted` and
        # `log_accept_ratio`; both were already traced and then discarded here,
        # which left the caller unable to check the (0.65, 0.75) acceptance band
        # that A3 imposes.  Returned as-is, shape [num_samples, num_chains].
        # Observability only: an in-band acceptance rate is not evidence of good
        # mixing, which is the inference A3 exists to correct.
        "sampling_is_accepted": trace["is_accepted"],
        "sampling_log_accept_ratio": trace["log_accept_ratio"],
    }

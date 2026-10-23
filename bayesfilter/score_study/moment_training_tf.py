"""Simulation-supervised terminal moment training; no admission authority.

Model scores retain the shared analytical recursion. Optimization here uses
only terminal moment loss, with fixed labels and independent particle streams.
"""

import math

import tensorflow as tf

from bayesfilter.score_study.supervised_guidance_tf import moment_features


def calibrate_moments(states):
    center = tf.reduce_mean(states, axis=0)
    scale = tf.maximum(tf.math.reduce_std(states, axis=0), 1e-6)
    features = moment_features((states - center) / scale)
    feature_scale = tf.maximum(tf.math.reduce_std(features, axis=0), 1e-6)
    return center, scale, feature_scale


def make_moment_objective(particles, dimension, center, scale, feature_scale,
                          *, jit_compile=True):
    dtype = center.dtype

    @tf.function(input_signature=[tf.TensorSpec([particles, dimension], dtype),
                                  tf.TensorSpec([dimension], dtype)],
                 jit_compile=jit_compile, autograph=False)
    def objective(states, label):
        estimate = tf.reduce_mean(moment_features((states - center) / scale), axis=0)
        target = moment_features((label - center) / scale)
        squared = tf.square((estimate - target) / feature_scale)
        blocks = tf.stack([tf.reduce_mean(squared[:dimension]),
                           tf.reduce_mean(squared[dimension:])])
        raw = tf.square(tf.reduce_mean(moment_features(states), axis=0)
                        - moment_features(label))
        raw_blocks = tf.stack([tf.reduce_mean(raw[:dimension]),
                               tf.reduce_mean(raw[dimension:])])
        valid = tf.reduce_all(tf.math.is_finite(squared))
        valid &= tf.reduce_all(tf.math.is_finite(raw_blocks))
        return dict(loss=tf.reduce_mean(blocks), blocks=blocks, raw_blocks=raw_blocks,
                    estimate=estimate, valid=valid)

    return objective


def make_guide_adapters(horizon, dimension, components, time_specific, *,
                        dtype=tf.float64, jit_compile=True):
    if horizon < 2 or dimension % 2:
        raise ValueError("Require T>=2 and interleaved two-coordinate regions")
    parameter_rows = horizon - 1 if time_specific else 1
    eta_spec = tf.TensorSpec([parameter_rows, 3], dtype)

    def expand(eta):
        active = eta if time_specific else tf.repeat(eta, horizon - 1, axis=0)
        expanded = tf.concat([active, tf.zeros([1, 3], dtype)], axis=0)
        strength = tf.exp(expanded[:, 0])
        shift = tf.tile(expanded[:, 1:], [1, dimension // 2])
        return strength, shift

    @tf.function(input_signature=[eta_spec,
                                  tf.TensorSpec([horizon, dimension, dimension], dtype),
                                  tf.TensorSpec([horizon, dimension], dtype)],
                 jit_compile=jit_compile, autograph=False)
    def quadratic(eta, precision, linear):
        strength, shift = expand(eta)
        shifted = linear + tf.einsum("tij,tj->ti", precision, shift)
        return strength[:, None, None] * precision, strength[:, None] * shifted

    @tf.function(input_signature=[eta_spec,
                                  tf.TensorSpec([horizon, components], dtype),
                                  tf.TensorSpec([horizon, components, dimension], dtype),
                                  tf.TensorSpec([horizon, components, dimension, dimension], dtype),
                                  tf.TensorSpec([horizon], tf.bool)],
                 jit_compile=jit_compile, autograph=False)
    def mixture(eta, logits, centers, covariances, active):
        strength, shift = expand(eta)
        return (logits, centers + shift[:, None],
                covariances / strength[:, None, None, None], active)

    return quadratic, mixture


def make_moment_endpoint(kernel, objective, dimension, *, dtype=tf.float64,
                         jit_compile=True):
    signature = list(kernel.input_signature) + [tf.TensorSpec([dimension], dtype)]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def endpoint(*arguments):
        value, derivative, trace = kernel.python_function(*arguments[:-1])
        result = objective.python_function(trace[-1]["states_after_reset"], arguments[-1])
        checks = {name: tf.reduce_all(tf.stack([step[name] for step in trace]))
                  for name in trace[-1] if trace[-1][name].dtype == tf.bool}
        diagnostics = {name: tf.stack([step[name] for step in trace])
                       for name in trace[-1] if trace[-1][name].shape.rank == 0
                       and trace[-1][name].dtype != tf.bool}
        if "guide_conditioning_margin_measured" in trace[-1]:
            diagnostics["guide_conditioning_margin_measured"] = tf.stack(
                [step["guide_conditioning_margin_measured"] for step in trace])
        validity = [flag for name, flag in checks.items()
                    if name.endswith("_valid") or name.endswith("_finite")
                    or name == "higher_moment_materialized_shape_available"]
        valid = result["valid"] & tf.reduce_all(tf.stack(validity))
        valid &= tf.reduce_all(tf.math.is_finite(value))
        valid &= tf.reduce_all(tf.math.is_finite(derivative))
        return dict(result, valid=valid, value=value, score=derivative[0],
                    checks=checks, diagnostics=diagnostics)

    return endpoint


def random_direction_descent(objective, initial, *, seed, updates=48,
                             perturbation=.1, step_size=.25, record=None,
                             checkpoint=None, jit_compile=True):
    """Optimize a changing minibatch objective with paired probes and descent.

    objective(eta, update) must use the same labels and particle streams for
    every call within one update. It returns a scalar float; infinity rejects
    invalid trials. Callback checkpoint never feeds test data to the optimizer.
    """
    if updates < 1 or perturbation <= 0 or step_size <= 0:
        raise ValueError("Positive optimization budget and step sizes required")
    eta = tf.identity(initial)
    bounds = tf.constant([math.log(4.), 2., 2.], eta.dtype)

    @tf.function(input_signature=[tf.TensorSpec(eta.shape, eta.dtype),
                                  tf.TensorSpec([], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def probes(current, update):
        random = tf.random.stateless_uniform(current.shape, tf.stack([seed, update]), dtype=current.dtype)
        direction = tf.where(random < .5, -tf.ones_like(current), tf.ones_like(current))
        direction /= tf.linalg.norm(direction)
        return (direction, tf.clip_by_value(current + perturbation * direction, -bounds, bounds),
                tf.clip_by_value(current - perturbation * direction, -bounds, bounds))

    @tf.function(input_signature=[tf.TensorSpec(eta.shape, eta.dtype),
                                  tf.TensorSpec(eta.shape, eta.dtype),
                                  tf.TensorSpec([], eta.dtype)],
                 jit_compile=jit_compile, autograph=False)
    def propose(current, direction, length):
        return tf.clip_by_value(current - length * direction, -bounds, bounds)

    accepted_count = 0
    if checkpoint is not None:
        checkpoint(0, eta)
    for update in range(updates):
        direction, plus, minus = probes(eta, tf.constant(update))
        current = float(objective(eta, update))
        if not math.isfinite(current):
            raise ValueError("Invalid current training candidate")
        plus_loss = float(objective(plus, update))
        minus_loss = float(objective(minus, update))
        accepted, step_length = False, 0.
        trials = []
        if math.isfinite(plus_loss) and math.isfinite(minus_loss) and plus_loss != minus_loss:
            descent = direction if plus_loss > minus_loss else -direction
            for reduction in range(3):
                length = step_size / (2 ** reduction)
                trial = propose(eta, descent, tf.constant(length, eta.dtype))
                trial_loss = float(objective(trial, update))
                trials.append(dict(length=length, loss=trial_loss if math.isfinite(trial_loss) else None))
                if math.isfinite(trial_loss) and trial_loss < current:
                    eta, current, accepted, step_length = trial, trial_loss, True, length
                    accepted_count += 1
                    break
        if record is not None:
            record(dict(update=update + 1, loss=current,
                        plus_loss=plus_loss if math.isfinite(plus_loss) else None,
                        minus_loss=minus_loss if math.isfinite(minus_loss) else None,
                        accepted=accepted, accepted_count=accepted_count,
                        step_length=step_length, trials=trials, eta=eta,
                        bound_hits=tf.reduce_sum(tf.cast(tf.abs(eta) >= bounds - 1e-12, tf.int32))))
        if checkpoint is not None and ((update + 1) % 16 == 0 or update + 1 == updates):
            checkpoint(update + 1, eta)
    return eta, accepted_count

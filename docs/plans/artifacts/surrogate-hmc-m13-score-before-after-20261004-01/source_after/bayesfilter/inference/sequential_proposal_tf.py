"""Native trust-region proposal, exact callback and score acceptance.

The enclosing factor escalation and refinement lifecycle remain separate.
"""

import math
from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference.program_cache_scope import scoped_program_cache
from bayesfilter.inference.sequential_preparation_tf import trust_region_program
from bayesfilter.inference.sequential_structured_preparation_tf import _position_program

D = tf.float64


@lru_cache(maxsize=16)
def acceptance_program(policy, active, *, jit_compile=True):
    """Preserve finite, strict score-resolution and trust-ratio boundaries."""
    if active and policy not in ('fractional', 'resolvable_decrease'):
        raise ValueError(f'unknown proposal score acceptance policy {policy!r}')

    @tf.function(input_signature=[tf.TensorSpec([], D), tf.TensorSpec([], D),
        tf.TensorSpec([], D), tf.TensorSpec([], D), tf.TensorSpec([], D),
        tf.TensorSpec([], D), tf.TensorSpec([], tf.bool)],
        jit_compile=jit_compile, autograph=False)
    def decide(old_norm, new_norm, actual, predicted, fractional_factor, acceptance_ratio, finite):
        rho = tf.where(predicted > 0., actual / predicted, tf.constant(float('-inf'), D))
        finite_norms = tf.math.is_finite(old_norm) & tf.math.is_finite(new_norm)
        legacy_threshold = tf.where(finite_norms, fractional_factor * old_norm, tf.constant(float('nan'), D))
        legacy_passed = finite_norms & (new_norm <= legacy_threshold)
        if not active:
            score_passed = tf.constant(True)
            required = tf.constant(0., D)
            reported_floor = tf.constant(0., D)
        elif policy == 'fractional':
            score_passed = legacy_passed
            required = legacy_threshold
            reported_floor = tf.constant(0., D)
        else:
            # Python max(1., abs(NaN)) returns 1.; preserve that report too.
            floor = tf.constant(math.sqrt(math.ulp(1.)), D) * tf.where(
                tf.abs(old_norm) > 1., tf.abs(old_norm), tf.constant(1., D))
            score_passed = finite_norms & (old_norm - new_norm > floor)
            required = old_norm - floor
            reported_floor = floor
        accepted = finite & (actual > 0.) & (predicted > 0.) & (rho >= acceptance_ratio) & score_passed
        return {'rho': rho, 'score_passed': score_passed, 'legacy_passed': legacy_passed,
            'required_norm_max': required, 'resolution_floor': reported_floor, 'accepted': accepted}

    return decide


@scoped_program_cache(maxsize=64)
def proposal_program(scalar, dimension, policy, active, *, jit_compile=True):
    solve = trust_region_program(dimension, jit_compile=jit_compile).python_function
    position = _position_program(1, dimension, jit_compile)
    decide = acceptance_program(policy, active, jit_compile=jit_compile).python_function

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([], D), tf.TensorSpec([], D)], jit_compile=jit_compile, autograph=False)
    def propose(center, center_value, center_score, scale, precision, radius,
                fractional_factor, acceptance_ratio):
        step, boundary, predicted = solve(precision, scale * center_score, radius)
        proposed = position(step[None, :], center, scale)[0]
        value, score = scalar(proposed)
        value = tf.reshape(tf.convert_to_tensor(value, D), [])
        score = tf.reshape(tf.convert_to_tensor(score, D), [-1])
        if score.shape != (dimension,):
            raise ValueError('target score must have one entry per parameter')
        actual = value - center_value
        old_norm = tf.linalg.norm(scale * center_score)
        new_norm = tf.linalg.norm(scale * score)
        finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
        decision = decide(old_norm, new_norm, actual, predicted, fractional_factor, acceptance_ratio, finite)
        return {'position': proposed, 'value': value, 'score': score, 'step': step,
            'actual': actual, 'predicted': predicted, 'old_norm': old_norm, 'new_norm': new_norm,
            'finite': finite, 'boundary': boundary, **decision}

    return propose

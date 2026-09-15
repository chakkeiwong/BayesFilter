"""Canonical consumer regression, including parameter-dependent initialization.

Tiny CPU/XLA/FP64 mechanics only. These controls are explicit fixtures, not
scope tuning or evidence of scientific/default readiness.
"""
import tensorflow as tf

from bayesfilter.score_study.canonical_adapter_tf import make_canonical_kernel


CONTROLS = dict(flow_substeps=2, reset_policy="contract_e", reset_epsilon=2.0,
    reset_sinkhorn_steps=3, reset_balance_steps=3, reset_ridge=1e-5,
    correction_steps=1, correction_strength=0.2, correction_lm_damping=1e-2,
    correction_lm_scale_floor=1e-4, correction_trust_radius=0.5,
    pairwise_steps=1, pairwise_strength=0.02, pairwise_rms_cap=2.0,
    coordinate_cap=0.4, coordinate_cap_power=8)


def test_actual_canonical_initial_law_and_all_parameter_directions():
    theta = tf.constant([.62, -.8, -.6, .9, .25, -.3], tf.float64)
    observations = tf.constant([[.5], [-.2]], tf.float64)
    initial = tf.random.stateless_normal([8, 2], [3, 7], dtype=tf.float64)
    noise = tf.random.stateless_normal([2, 8, 2], [3, 8], dtype=tf.float64)
    design = tf.random.stateless_normal([8, 2], [3, 9], dtype=tf.float64)
    kernel = make_canonical_kernel(2, 1, 8, 2, tuple(sorted(CONTROLS.items())), jit_compile=True)
    values, scores, differences = [], [], []
    for direction in tf.unstack(tf.eye(6, dtype=tf.float64)):
        value, score = kernel(theta, direction, observations, initial, noise, design)
        assert value.shape == () and score.shape == ()
        values.append(value)
        scores.append(score)
        h = tf.constant(2e-5, tf.float64)
        plus = kernel(theta + h * direction, direction, observations, initial, noise, design)[0]
        minus = kernel(theta - h * direction, direction, observations, initial, noise, design)[0]
        differences.append((plus - minus) / (2 * h))
    tf.debugging.assert_all_finite(tf.stack(scores), "canonical score")
    tf.debugging.assert_near(tf.stack(values), tf.fill([6], values[0]), rtol=1e-10, atol=1e-11)
    tf.debugging.assert_near(tf.stack(scores), tf.stack(differences), rtol=3e-4, atol=3e-5)
    assert kernel.experimental_get_tracing_count() == 1


def test_xla_does_not_discard_final_reset_validity(monkeypatch):
    from bayesfilter.highdim import ledh_unified_correction_tf as correction
    original = correction.batched_higher_moment_shape_jvp
    def flagged(*args, **kwargs):
        result = dict(original(*args, **kwargs))
        result["valid"] = tf.zeros_like(result["valid"])
        return result
    monkeypatch.setattr(correction, "batched_higher_moment_shape_jvp", flagged)
    make_canonical_kernel.cache_clear()
    kernel = make_canonical_kernel(2, 1, 8, 1, tuple(sorted(CONTROLS.items())), jit_compile=True)
    value, score = kernel(tf.constant([.62, -.8, -.6, .9, .25, -.3], tf.float64),
        tf.constant([1., 0., 0., 0., 0., 0.], tf.float64), tf.constant([[.5]], tf.float64),
        tf.random.stateless_normal([8, 2], [3, 7], dtype=tf.float64),
        tf.random.stateless_normal([1, 8, 2], [3, 8], dtype=tf.float64),
        tf.random.stateless_normal([8, 2], [3, 9], dtype=tf.float64))
    assert not bool(tf.math.is_finite(value))
    assert not bool(tf.math.is_finite(score))
    make_canonical_kernel.cache_clear()

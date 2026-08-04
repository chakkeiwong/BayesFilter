from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.models import _zhao_cui_sir_austria_transition_mean_xla
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_sampler_tf import (
    LaneBRetainedGridSampler,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tail_tf import (
    LaneBT2UntouchedTailCloud,
    estimate_tail_log_normalizer,
    evaluate_t2_tail_chunk,
    signed_log_add,
    signed_log_author_transition_mean,
    signed_log_from_real,
    signed_log_to_real,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def test_signed_log_author_transition_matches_ordinary_fp64() -> None:
    state = tf.reshape(
        tf.linspace(tf.constant(2.0, tf.float64), tf.constant(20.0, tf.float64), 36),
        [2, 18],
    )
    direct = _zhao_cui_sir_austria_transition_mean_xla(
        tf.zeros([3], tf.float64), state
    )
    reconstructed = signed_log_to_real(signed_log_author_transition_mean(state))
    tf.debugging.assert_near(reconstructed, direct, rtol=2e-11, atol=2e-11)


def test_signed_log_add_broadcasts_like_tensorflow_addition() -> None:
    left = tf.reshape(
        tf.linspace(tf.constant(-3.0, tf.float64), tf.constant(3.0, tf.float64), 24),
        [4, 6],
    )
    right = tf.reshape(
        tf.linspace(tf.constant(-0.5, tf.float64), tf.constant(0.5, tf.float64), 6),
        [1, 6],
    )
    observed = signed_log_to_real(
        signed_log_add(signed_log_from_real(left), signed_log_from_real(right))
    )
    tf.debugging.assert_near(observed, left + right, rtol=2e-15, atol=2e-15)


def test_known_untouched_tail_column_has_positive_zero_density_certificate() -> None:
    parent = load_lane_b_t1_artifact_v1_compat(PARENT)
    reference = tf.random.stateless_uniform(
        [18, 16384],
        seed=[73804, 1],
        minval=tf.constant(1e-6, tf.float64),
        maxval=tf.constant(1.0 - 1e-6, tf.float64),
        dtype=tf.float64,
    )[:, 12287:12288]
    retained = LaneBRetainedGridSampler(parent).inverse(reference)
    first = 12224
    noise = tf.random.stateless_normal(
        [64, 18], seed=[73814 + first, 2], dtype=tf.float64
    )[63:64]
    _states, observations, _all = generate_sealed_lane_b_dataset()
    result = evaluate_t2_tail_chunk(
        z1=retained.physical_points,
        transition_noise=noise,
        observation=observations[1],
    )
    assert bool(result.nonrepresentable_mask[0].numpy())
    assert bool(result.zero_target_mask[0].numpy())
    assert float(result.overflow_log_margin[0].numpy()) > 100.0


def test_tail_estimate_keeps_zero_row_in_monte_carlo_denominator() -> None:
    cloud = LaneBT2UntouchedTailCloud(
        reference_uniforms=tf.fill([18, 2], tf.constant(0.5, tf.float64)),
        z1=tf.zeros([2, 18], tf.float64),
        transition_noise=tf.zeros([2, 18], tf.float64),
        previous_correction=tf.zeros([2], tf.float64),
        log_likelihood=tf.constant([0.0, float("-inf")], tf.float64),
        transition_log_density=tf.zeros([2], tf.float64),
        nonrepresentable_mask=tf.constant([False, True]),
        overflow_log_margin=tf.constant([0.0, 100.0], tf.float64),
        role="untouched",
        reference_seed=73804,
        transition_seed=73814,
    )
    estimate = estimate_tail_log_normalizer(cloud, tf.constant(0.0, tf.float64))
    tf.debugging.assert_near(
        estimate.log_increment,
        -tf.math.log(tf.constant(2.0, tf.float64)),
        atol=0.0,
    )

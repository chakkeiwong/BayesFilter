from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_sampler_tf import (
    LaneBRetainedGridSampler,
    retained_sampler_workspace_estimate_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _sampler() -> LaneBRetainedGridSampler:
    return LaneBRetainedGridSampler(load_lane_b_t1_artifact_v1_compat(ARTIFACT))


def test_selected_retained_sampler_roundtrip_and_exact_proposal_score() -> None:
    sampler = _sampler()
    reference = tf.random.stateless_uniform(
        [18, 4], seed=[73701, 11], minval=1e-6, maxval=1.0 - 1e-6, dtype=tf.float64
    )
    sample = sampler.inverse(reference)
    replay_reference, replay_proposal, replay_residuals = (
        sampler.forward_and_log_proposal(sample.local_points)
    )
    tf.debugging.assert_near(replay_reference, reference, atol=2e-12)
    tf.debugging.assert_near(
        replay_proposal, sample.proposal_log_density, atol=2e-12
    )
    tf.debugging.assert_near(
        replay_residuals, sample.raw_conditional_mass_residuals, atol=2e-12
    )
    assert float(tf.reduce_max(sample.raw_conditional_mass_residuals).numpy()) <= 5e-10


def test_selected_retained_sampler_replays_exactly_and_carries_tt_correction() -> None:
    sampler = _sampler()
    reference = tf.random.stateless_uniform(
        [18, 3], seed=[73701, 12], minval=1e-6, maxval=1.0 - 1e-6, dtype=tf.float64
    )
    left = sampler.inverse(reference)
    right = sampler.inverse(reference)
    for left_value, right_value in (
        (left.local_points, right.local_points),
        (left.physical_points, right.physical_points),
        (left.proposal_log_density, right.proposal_log_density),
        (left.target_log_density, right.target_log_density),
        (left.correction_log_weights, right.correction_log_weights),
    ):
        tf.debugging.assert_equal(left_value, right_value)
    tf.debugging.assert_near(
        left.correction_log_weights,
        left.target_log_density - left.proposal_log_density,
        atol=2e-12,
    )
    assert sampler.manifest_payload()["production_kr_closure"] is False


def test_b2_sampler_static_workspace_is_below_microbatch_cap() -> None:
    estimate = retained_sampler_workspace_estimate_bytes(
        sample_count=64, grid_size=65, max_rank=4
    )
    assert estimate < 512 * 1024 * 1024

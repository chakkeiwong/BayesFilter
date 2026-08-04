from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_score_tf import (
    physical_z1_to_parent_local_prefix,
    t2_target_log_value_and_manual_score,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_training_jvp_tf import (
    T2ReplayCloudInputs,
    _active_log_weight,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_t1_proposal_cloud,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
)


ROOT = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _child() -> LaneBParameterChild:
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    banks = []
    for axis, core in enumerate(parent.cores):
        banks.append(
            tuple(
                tf.ones_like(core)
                * tf.constant(
                    (parameter + 1) * 2e-7 / (axis + 1), tf.float64
                )
                for parameter in range(3)
            )
        )
    return LaneBParameterChild(parent, tuple(banks))


def _joint_points() -> tf.Tensor:
    cloud = generate_t1_proposal_cloud(
        sample_count=4, seed=74901, role="t2_score_unit"
    )
    z1 = cloud.joint_points[:, :18]
    # Small deterministic displacement is sufficient to exercise T2 terms.
    z2 = z1 + tf.reshape(
        tf.linspace(
            tf.constant(-0.05, tf.float64),
            tf.constant(0.05, tf.float64),
            18,
        ),
        [1, 18],
    )
    return tf.concat([z2, z1], axis=1)


def test_physical_prefix_conversion_roundtrips_parent_frame() -> None:
    child = _child()
    points = _joint_points()
    z1 = points[:, 18:]
    local = physical_z1_to_parent_local_prefix(child, z1)
    matrix = child.parent.frame.matrix[:18, :18]
    reconstructed = tf.transpose(
        tf.linalg.matmul(matrix, tf.transpose(local))
        + child.parent.frame.mu[:18, tf.newaxis]
    )
    tf.debugging.assert_near(reconstructed, z1, atol=2e-12, rtol=2e-12)


def test_t2_manual_score_is_exact_sum_of_carried_and_local_terms() -> None:
    child = _child()
    result = t2_target_log_value_and_manual_score(
        child, tf.zeros([3], tf.float64), _joint_points()
    )
    tf.debugging.assert_equal(
        result["score"],
        result["previous_score"]
        + result["transition_score"]
        + result["observation_score"],
    )
    assert float(tf.reduce_max(tf.abs(result["previous_score"])).numpy()) > 0.0


def test_t2_manual_carried_score_matches_same_row_autodiff() -> None:
    child = _child()
    theta = tf.constant([0.01, -0.015, 0.02], tf.float64)
    points = _joint_points()
    manual = t2_target_log_value_and_manual_score(child, theta, points)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        log_value = t2_target_log_value_and_manual_score(
            child, theta, points
        )["log_value"]
    diagnostic = tape.jacobian(log_value, theta)
    tf.debugging.assert_near(
        manual["score"], diagnostic, atol=3e-10, rtol=3e-10
    )


def test_t2_replay_weight_origin_and_jvp_match_carried_target_score() -> None:
    child = _child()
    points = _joint_points()
    origin = tf.zeros([3], tf.float64)
    target = t2_target_log_value_and_manual_score(child, origin, points)
    origin_weight = tf.constant([-3.0, -2.0, -1.0, -0.5], tf.float64)
    cloud = T2ReplayCloudInputs(
        joint_points=points,
        local_points=tf.zeros([4, 36], tf.float64),
        origin_log_importance_weight=origin_weight,
        origin_target_log_value=target["log_value"],
    )
    tf.debugging.assert_equal(
        _active_log_weight(origin, child, cloud), origin_weight
    )
    rows = []
    for parameter in range(3):
        direction = tf.one_hot(parameter, 3, dtype=tf.float64)
        with tf.autodiff.ForwardAccumulator(origin, direction) as accumulator:
            active = _active_log_weight(origin, child, cloud)
        rows.append(accumulator.jvp(active))
    diagnostic = tf.stack(rows, axis=1)
    tf.debugging.assert_near(
        diagnostic, target["score"], atol=3e-10, rtol=3e-10
    )

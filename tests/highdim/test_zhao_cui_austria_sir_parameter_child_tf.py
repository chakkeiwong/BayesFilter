from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (
    load_lane_b_t2_artifact,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
    load_selected_t2_parameter_parent_compat,
)


ROOT = Path(__file__).resolve().parents[2]
T1_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
T2_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "pilot-final-01/t2_p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _parents():
    t1 = load_lane_b_t1_artifact_v1_compat(T1_DIR)
    t2 = load_selected_t2_parameter_parent_compat(T2_DIR, parent_artifact=t1)
    return t1, t2


def test_canonical_t2_loader_remains_fail_closed_on_cpu_backend_identity_drift() -> None:
    t1 = load_lane_b_t1_artifact_v1_compat(T1_DIR)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_lane_b_t2_artifact(T2_DIR, parent_artifact=t1)


def _tangents(parent, scale: float = 1e-3):
    rows = []
    for axis, core in enumerate(parent.cores):
        values = []
        for parameter in range(3):
            sign = -1.0 if (axis + parameter) % 2 else 1.0
            values.append(
                tf.ones_like(core)
                * tf.constant(sign * scale * (parameter + 1) / (axis + 1), tf.float64)
            )
        rows.append(tuple(values))
    return tuple(rows)


@pytest.mark.parametrize("parent_index", [0, 1])
def test_parameter_child_zero_slice_preserves_parent_cores_and_value(parent_index: int) -> None:
    parents = _parents()
    parent = parents[parent_index]
    before = tuple(tf.identity(core) for core in parent.cores)
    child = LaneBParameterChild(parent, _tangents(parent))
    zero = tf.zeros([3], tf.float64)
    for observed, expected in zip(child.conditioned_cores(zero), before):
        tf.debugging.assert_near(observed, expected, atol=0.0)
    child_increment, _score = child.increment_and_score(zero)
    parent_increment = parent.value() if parent_index == 0 else parent.increment()
    tf.debugging.assert_near(child_increment, parent_increment, atol=2e-13)
    for observed, expected in zip(parent.cores, before):
        tf.debugging.assert_near(observed, expected, atol=0.0)
    assert child.identity.manifest.payload["theta_integration_forbidden"] is True
    assert child.identity.manifest.payload["parent_identity"] == parent.identity.hash.value


def test_parameter_child_origin_prefix_marginal_equals_t1_parent() -> None:
    parent, _t2 = _parents()
    child = LaneBParameterChild(parent, _tangents(parent))
    points = tf.reshape(
        tf.linspace(tf.constant(-0.15, tf.float64), tf.constant(0.15, tf.float64), 54),
        [3, 18],
    )
    parent_log = tf.math.log(
        parent.density().normalized_marginal_density_values(tuple(range(18)), points)
    )
    child_log, _score = child.prefix_log_marginal_and_score(
        tf.zeros([3], tf.float64), points
    )
    tf.debugging.assert_near(child_log, parent_log, atol=2e-12)


@pytest.mark.parametrize("parent_index", [0, 1])
def test_parameter_child_manual_increment_score_matches_diagnostic_tape(parent_index: int) -> None:
    parent = _parents()[parent_index]
    child = LaneBParameterChild(parent, _tangents(parent))
    theta = tf.Variable([0.02, -0.01, 0.015], dtype=tf.float64)
    manual_value, manual_score = child.increment_and_score(theta)
    with tf.GradientTape() as tape:
        tape_value = child.increment_and_score(theta)[0]
    tape_score = tape.gradient(tape_value, theta)
    assert tape_score is not None
    tf.debugging.assert_near(manual_value, tape_value, atol=0.0)
    tf.debugging.assert_near(manual_score, tape_score, atol=2e-11, rtol=2e-11)


def test_parameter_child_manual_point_and_prefix_scores_match_diagnostic_tape() -> None:
    parent, _t2 = _parents()
    child = LaneBParameterChild(parent, _tangents(parent, scale=2e-4))
    theta = tf.Variable([0.01, -0.015, 0.02], dtype=tf.float64)
    full_points = tf.reshape(
        tf.linspace(tf.constant(-0.1, tf.float64), tf.constant(0.1, tf.float64), 72),
        [2, 36],
    )
    manual_log, manual_score = child.point_log_density_and_score(theta, full_points)
    with tf.GradientTape() as tape:
        tape_log = child.point_log_density_and_score(theta, full_points)[0]
    tape_score = tape.jacobian(tape_log, theta)
    tf.debugging.assert_near(manual_log, tape_log, atol=0.0)
    tf.debugging.assert_near(manual_score, tape_score, atol=3e-10, rtol=3e-10)

    prefix = full_points[:, :18]
    manual_prefix, manual_prefix_score = child.prefix_log_marginal_and_score(
        theta, prefix
    )
    with tf.GradientTape() as tape:
        tape_prefix = child.prefix_log_marginal_and_score(theta, prefix)[0]
    tape_prefix_score = tape.jacobian(tape_prefix, theta)
    tf.debugging.assert_near(manual_prefix, tape_prefix, atol=0.0)
    tf.debugging.assert_near(
        manual_prefix_score, tape_prefix_score, atol=3e-10, rtol=3e-10
    )


def test_parameter_child_rejects_tangent_shape_and_identity_tamper() -> None:
    parent, _t2 = _parents()
    tangents = list(_tangents(parent))
    bad_bank = list(tangents[0])
    bad_bank[0] = tf.zeros([1, 1, 1], tf.float64)
    tangents[0] = tuple(bad_bank)
    with pytest.raises(ValueError, match="tangent shape mismatch"):
        LaneBParameterChild(parent, tuple(tangents))

    valid = LaneBParameterChild(parent, _tangents(parent))
    other = LaneBParameterChild(parent, _tangents(parent, scale=2e-3))
    with pytest.raises(ValueError, match="identity mismatch"):
        LaneBParameterChild(parent, valid.tangent_cores, child_identity=other.identity)


def test_parameter_child_storage_is_linear_in_three_tangent_banks() -> None:
    parent, _t2 = _parents()
    child = LaneBParameterChild(parent, _tangents(parent))
    parent_elements = sum(int(tf.size(core).numpy()) for core in parent.cores)
    tangent_elements = sum(
        int(tf.size(core).numpy()) for bank in child.tangent_cores for core in bank
    )
    assert tangent_elements == 3 * parent_elements

from __future__ import annotations

from pathlib import Path
import json

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (
    CenteredThetaFeatures,
    LaneBCenteredResidualChild,
    centered_lane_b_product_basis,
    load_centered_residual_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
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


def _scaled_parent_component(parent, scale: float):
    cores = [tf.identity(core) for core in parent.cores]
    cores[0] = tf.constant(scale, tf.float64) * cores[0]
    return tuple(cores)


def _child(parent):
    return LaneBCenteredResidualChild(
        parent=parent,
        residual_components=tuple(
            _scaled_parent_component(parent, scale)
            for scale in (0.03, -0.02, 0.01)
        ),
    )


def test_centered_feature_map_is_zero_at_origin_and_has_manual_jacobian() -> None:
    features = CenteredThetaFeatures(
        (
            "linear_0",
            "linear_1",
            "linear_2",
            "quadratic_0",
            "interaction_0_2",
        )
    )
    zero_values, _zero_jacobian = features.values_and_jacobian(
        tf.zeros([3], tf.float64)
    )
    tf.debugging.assert_equal(zero_values, tf.zeros_like(zero_values))

    theta = tf.Variable([0.02, -0.03, 0.04], dtype=tf.float64)
    manual_values, manual_jacobian = features.values_and_jacobian(theta)
    with tf.GradientTape() as tape:
        tape_values = features.values_and_jacobian(theta)[0]
    tape_jacobian = tape.jacobian(tape_values, theta)
    tf.debugging.assert_near(manual_values, tape_values, atol=0.0)
    tf.debugging.assert_near(
        manual_jacobian, tape_jacobian, atol=1e-16, rtol=2e-15
    )


def test_centered_basis_freezes_local_nodes_and_matches_parent_cpu_basis() -> None:
    parent, _t2 = _parents()
    frozen = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    points = tf.constant([-0.37331331754608577, 0.0, 0.43336918135755276], tf.float64)
    expected = parent.density().sqrt_tt.product_basis.evaluate_axis(0, points)
    observed = frozen.evaluate_axis(0, points)
    tf.debugging.assert_near(observed, expected, atol=2e-15, rtol=2e-15)
    manifest = frozen.bases[0].manifest_payload()
    assert manifest["centered_evaluation_id"] == "setup_static_cpu_nodes_barycentric_weights_v1"


@pytest.mark.parametrize("parent_index", [0, 1])
def test_centered_child_exactly_preserves_parent_origin_value(parent_index: int) -> None:
    parent = _parents()[parent_index]
    before = tuple(tf.identity(core) for core in parent.cores)
    child = _child(parent)
    child_increment, _score = child.increment_and_score(tf.zeros([3], tf.float64))
    parent_increment = parent.value() if parent_index == 0 else parent.increment()
    tf.debugging.assert_near(child_increment, parent_increment, atol=2e-13)
    for observed, expected in zip(parent.cores, before):
        tf.debugging.assert_equal(observed, expected)
    assert child.identity.manifest.payload["theta_integration_forbidden"] is True
    assert (
        child.identity.manifest.payload["parent_identity"]
        == parent.identity.hash.value
    )


def test_centered_child_origin_point_and_prefix_values_equal_parent() -> None:
    parent, _t2 = _parents()
    child = _child(parent)
    zero = tf.zeros([3], tf.float64)
    full_points = tf.reshape(
        tf.linspace(tf.constant(-0.1, tf.float64), tf.constant(0.1, tf.float64), 72),
        [2, 36],
    )
    child_log, _child_score = child.point_log_density_and_score(zero, full_points)
    parent_log = parent.density().log_density(full_points)
    tf.debugging.assert_near(child_log, parent_log, atol=2e-13)

    prefix_points = full_points[:, :18]
    child_prefix, _prefix_score = child.prefix_log_marginal_and_score(
        zero, prefix_points
    )
    parent_prefix = tf.math.log(
        parent.density().normalized_marginal_density_values(
            tuple(range(18)), prefix_points
        )
    )
    tf.debugging.assert_near(child_prefix, parent_prefix, atol=2e-12)


def test_centered_child_manual_scores_match_diagnostic_autodiff() -> None:
    parent, _t2 = _parents()
    child = _child(parent)
    theta = tf.Variable([0.02, -0.01, 0.015], dtype=tf.float64)
    points = tf.reshape(
        tf.linspace(tf.constant(-0.08, tf.float64), tf.constant(0.08, tf.float64), 72),
        [2, 36],
    )

    manual_increment, manual_increment_score = child.increment_and_score(theta)
    with tf.GradientTape() as tape:
        tape_increment = child.increment_and_score(theta)[0]
    tape_increment_score = tape.gradient(tape_increment, theta)
    tf.debugging.assert_near(manual_increment, tape_increment, atol=0.0)
    tf.debugging.assert_near(
        manual_increment_score, tape_increment_score, atol=2e-12, rtol=2e-12
    )

    manual_log, manual_score = child.point_log_density_and_score(theta, points)
    with tf.GradientTape() as tape:
        tape_log = child.point_log_density_and_score(theta, points)[0]
    tape_score = tape.jacobian(tape_log, theta)
    tf.debugging.assert_near(manual_log, tape_log, atol=0.0)
    tf.debugging.assert_near(manual_score, tape_score, atol=2e-11, rtol=2e-11)

    prefix_points = points[:, :18]
    manual_prefix, manual_prefix_score = child.prefix_log_marginal_and_score(
        theta, prefix_points
    )
    with tf.GradientTape() as tape:
        tape_prefix = child.prefix_log_marginal_and_score(theta, prefix_points)[0]
    tape_prefix_score = tape.jacobian(tape_prefix, theta)
    tf.debugging.assert_near(manual_prefix, tape_prefix, atol=0.0)
    tf.debugging.assert_near(
        manual_prefix_score, tape_prefix_score, atol=3e-11, rtol=3e-11
    )


def test_centered_child_storage_is_linear_and_rank_sum_is_not_materialized() -> None:
    parent, _t2 = _parents()
    child = _child(parent)
    parent_elements = sum(int(tf.size(core)) for core in parent.cores)
    estimate = child.memory_estimate(batch_size=64)
    assert estimate.component_count == 4
    assert estimate.stored_elements == 4 * parent_elements
    assert estimate.stored_bytes == estimate.stored_elements * tf.float64.size
    assert estimate.pair_workspace_elements == 4**4
    assert estimate.prefix_pair_workspace_elements == 64 * 4**4


def test_centered_child_fresh_reload_and_tensor_tamper_rejection(tmp_path: Path) -> None:
    parent, _t2 = _parents()
    child = _child(parent)
    artifact = tmp_path / "child"
    child.save(artifact)
    loaded = load_centered_residual_child(artifact, parent=parent)
    assert loaded.identity == child.identity
    theta = tf.constant([0.01, -0.02, 0.03], tf.float64)
    expected = child.increment_and_score(theta)
    observed = loaded.increment_and_score(theta)
    tf.debugging.assert_near(observed[0], expected[0], atol=0.0)
    tf.debugging.assert_near(observed[1], expected[1], atol=0.0)

    tampered = tf.identity(child.residual_components[0][0])
    tampered = tf.tensor_scatter_nd_add(
        tampered,
        tf.constant([[0, 0, 0]], tf.int32),
        tf.constant([1e-6], tf.float64),
    )
    tf.io.write_file(
        (artifact / "residual_00_core_00.tensor").as_posix(),
        tf.io.serialize_tensor(tampered),
    )
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        load_centered_residual_child(artifact, parent=parent)


def test_centered_child_identity_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    parent, _t2 = _parents()
    artifact = tmp_path / "child"
    _child(parent).save(artifact)
    manifest_path = artifact / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    payload["identity_manifest"]["shift_policy"] = "tampered"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="identity manifest mismatch"):
        load_centered_residual_child(artifact, parent=parent)


def test_centered_child_xla_matches_eager() -> None:
    parent, _t2 = _parents()
    child = _child(parent)
    theta = tf.constant([0.01, -0.02, 0.015], tf.float64)
    points = tf.reshape(
        tf.linspace(tf.constant(-0.05, tf.float64), tf.constant(0.05, tf.float64), 72),
        [2, 36],
    )

    @tf.function(jit_compile=True)
    def compiled(parameter, values):
        increment, increment_score = child.increment_and_score(parameter)
        log_density, point_score = child.point_log_density_and_score(parameter, values)
        prefix, prefix_score = child.prefix_log_marginal_and_score(
            parameter, values[:, :18]
        )
        return (
            increment,
            increment_score,
            log_density,
            point_score,
            prefix,
            prefix_score,
        )

    eager = (
        *child.increment_and_score(theta),
        *child.point_log_density_and_score(theta, points),
        *child.prefix_log_marginal_and_score(theta, points[:, :18]),
    )
    graph = compiled(theta, points)
    for observed, expected in zip(graph, eager):
        tf.debugging.assert_near(observed, expected, atol=3e-12, rtol=3e-12)

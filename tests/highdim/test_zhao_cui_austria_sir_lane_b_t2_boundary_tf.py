from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_boundary_tf import (
    KEEP_AXES,
    LaneBT1RetainedBoundary,
    independent_prefix_marginal_relative_density,
    independent_total_mass_from_cut,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_t1_proposal_cloud,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _artifact():
    return load_lane_b_t1_artifact_v1_compat(ARTIFACT)


def test_selected_t1_prefix_marginal_matches_independent_cut_contraction() -> None:
    artifact = _artifact()
    boundary = LaneBT1RetainedBoundary(artifact)
    cloud = generate_t1_proposal_cloud(sample_count=32, seed=73601, role="b3_probe")
    z1 = cloud.joint_points[:, :18]
    api = boundary.api_log_physical_density(z1)
    independent = boundary.independent_log_physical_density(z1)
    tf.debugging.assert_near(api, independent, atol=2e-12)


def test_independent_cut_total_mass_matches_selected_density_normalizer() -> None:
    artifact = _artifact()
    independent = independent_total_mass_from_cut(artifact)
    expected = artifact.density().normalizer()
    tf.debugging.assert_near(independent, expected, atol=2e-12)


def test_relative_prefix_api_matches_independent_values() -> None:
    artifact = _artifact()
    cloud = generate_t1_proposal_cloud(sample_count=16, seed=73602, role="b3_probe")
    z1 = cloud.joint_points[:, :18]
    local = tf.transpose(
        tf.linalg.triangular_solve(
            artifact.frame.matrix[:18, :18],
            tf.transpose(z1) - artifact.frame.mu[:18, tf.newaxis],
            lower=True,
        )
    )
    api = artifact.density().normalized_marginal_density_values(KEEP_AXES, local)
    independent = independent_prefix_marginal_relative_density(artifact, local)
    tf.debugging.assert_near(api, independent, atol=2e-12)


def test_t2_target_boundary_recomposes_same_scalar_components() -> None:
    artifact = _artifact()
    boundary = LaneBT1RetainedBoundary(artifact)
    cloud = generate_t1_proposal_cloud(sample_count=16, seed=73603, role="b3_probe")
    z1 = cloud.joint_points[:, :18]
    model = __import__(
        "bayesfilter.highdim.sir_latent_preclip_tf",
        fromlist=["latent_preclip_zhao_cui_sir_austria_model"],
    ).latent_preclip_zhao_cui_sir_austria_model()
    z2 = model.transition_push_from_standard_normal(
        tf.zeros([3], tf.float64),
        z1,
        tf.random.stateless_normal([16, 18], seed=[73603, 2], dtype=tf.float64),
        2,
    )
    terms = boundary.t2_log_target(z2, z1)
    tf.debugging.assert_near(terms["previous_api"], terms["previous_independent"], atol=2e-12)
    tf.debugging.assert_near(terms["api_total"], terms["independent_total"], atol=2e-12)
    assert bool(tf.reduce_all(tf.math.is_finite(terms["api_total"])).numpy())
    assert boundary.manifest_payload()["t2_training_status"] == "not_started_boundary_only"

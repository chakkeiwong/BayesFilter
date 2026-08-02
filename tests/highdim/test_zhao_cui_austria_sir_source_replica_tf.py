from __future__ import annotations

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_source_replica_tf import (
    ARTIFACT_SCHEMA,
    AUTHOR_BASIS_DIM,
    AUTHOR_CODE_ANCHORS,
    AUTHOR_PAPER_ANCHORS,
    FRAME_ADAPTER_ID,
    AuthorSIRSourceReplicaSpec,
    fit_author_sir_t1_source_replica,
    make_author_order_block_upper_frame,
)
from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame


def _tiny_spec() -> AuthorSIRSourceReplicaSpec:
    return AuthorSIRSourceReplicaSpec(
        fit_rank=1,
        fit_sample_count=8,
        holdout_sample_count=4,
        train_steps=1,
        optimizer_batch_size=4,
        cdf_grid_size=9,
        cdf_bisection_steps=4,
        kr_max_batch_working_bytes=64 * 1024 * 1024,
    )


def test_author_spec_binds_source_settings_and_memory_without_full_grid() -> None:
    spec = AuthorSIRSourceReplicaSpec()
    payload = spec.manifest_payload()

    assert spec.author_basis_exact is True
    assert payload["source_settings"]["basis"] == "Lagrangep(4,8)"
    assert payload["author_joint_order"] == "x_t_then_theta_then_x_previous"
    assert payload["runtime_forward_compiler_order"] == "x_previous_then_x_t"
    assert payload["joint_order_adapter_classification"] == "extension_or_invention"
    assert payload["frame_adapter_id"] == FRAME_ADAPTER_ID
    assert payload["frame_adapter_classification"] == "fixed_hmc_adaptation"
    assert payload["fit_backend_classification"] == "extension_or_invention"
    assert spec.memory_forecast(particle_count=32)["full_grid_retention_forbidden"] == 1


def test_author_spec_rejects_unbounded_rank_and_memory_controls() -> None:
    with pytest.raises(ValueError, match="max-rank"):
        AuthorSIRSourceReplicaSpec(fit_rank=41)
    with pytest.raises(ValueError, match="kr_max_batch_working_bytes"):
        AuthorSIRSourceReplicaSpec(kr_max_batch_working_bytes=0)


@pytest.fixture(scope="module")
def tiny_artifact():
    return fit_author_sir_t1_source_replica(_tiny_spec(), seed=8615)


def test_t1_candidate_uses_full_frame_and_serialized_immutable_cores(tiny_artifact) -> None:
    assert tiny_artifact.spec.author_basis_exact is True
    assert tiny_artifact.frame.dimension == 36
    assert len(tiny_artifact.cores) == 36
    assert all(int(core.shape[1]) == AUTHOR_BASIS_DIM for core in tiny_artifact.cores)
    assert tiny_artifact.diagnostics["fit_backend"] == (
        "p86_training_base_optimizer_not_author_tt_cross"
    )
    assert tiny_artifact.diagnostics["fit_data_manifest"]["basis_domain"] == (
        "author_Lagrangep_4_8_AlgebraicMapping_1"
    )
    assert tiny_artifact.diagnostics["source_paper_anchors"] == AUTHOR_PAPER_ANCHORS
    assert tiny_artifact.diagnostics["author_code_anchors"] == AUTHOR_CODE_ANCHORS
    tf.debugging.assert_near(
        tiny_artifact.frame.matrix[18:, :18],
        tf.zeros([18, 18], tf.float64),
        atol=1e-12,
    )


def test_block_upper_frame_preserves_full_covariance() -> None:
    source = SourceRouteCoordinateFrame(
        mu=tf.zeros([4], tf.float64),
        matrix=tf.constant(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.4, 1.5, 0.0, 0.0],
                [0.3, -0.2, 1.2, 0.0],
                [-0.1, 0.25, 0.2, 0.9],
            ],
            tf.float64,
        ),
        expansion_factor=4.0,
    )
    adapted = make_author_order_block_upper_frame(source, generated_dimension=2)
    tf.debugging.assert_near(
        adapted.matrix @ tf.transpose(adapted.matrix),
        source.matrix @ tf.transpose(source.matrix),
        atol=1e-12,
    )
    tf.debugging.assert_near(adapted.matrix[2:, :2], tf.zeros([2, 2], tf.float64))


def test_t1_upper_conditional_adapter_roundtrips(tiny_artifact) -> None:
    previous_local = tf.zeros([18, 2], tf.float64)
    uniforms = tf.reshape(
        tf.linspace(tf.constant(0.2, tf.float64), tf.constant(0.8, tf.float64), 36),
        [18, 2],
    )
    current_local = tiny_artifact.upper_conditional_inverse(previous_local, uniforms)
    reconstructed = tiny_artifact.upper_conditional_forward(
        previous_local, current_local
    )
    assert current_local.shape == (18, 2)
    tf.debugging.assert_near(reconstructed, uniforms, atol=0.15)


def test_t1_algorithm3_diagnostic_reports_finite_ess_and_memory(tiny_artifact) -> None:
    diagnostic = tiny_artifact.t1_algorithm3_diagnostic(
        particle_count=2,
        seed=8715,
    )
    assert diagnostic["frame_adapter_id"] == FRAME_ADAPTER_ID
    assert bool(diagnostic["finite"].numpy())
    assert 0.0 < float(diagnostic["ess_fraction"].numpy()) <= 1.0
    assert float(
        diagnostic["numerical_vs_exact_conditional_log_density_max_abs"].numpy()
    ) < 0.1
    assert int(diagnostic["max_kr_working_bytes"]) <= (
        tiny_artifact.spec.kr_max_batch_working_bytes
    )


def test_t1_identity_rejects_caller_stamped_diagnostics(tiny_artifact) -> None:
    diagnostics = dict(tiny_artifact.diagnostics)
    diagnostics["fit_backend"] = "author_tt_cross"
    with pytest.raises(ValueError, match="identity rejected"):
        # Reconstructing through the frozen dataclass is intentionally not a
        # public loader; tampering must still invalidate the artifact id.
        from bayesfilter.highdim.zhao_cui_austria_sir_source_replica_tf import (
            FrozenAuthorSIRT1Artifact,
        )

        FrozenAuthorSIRT1Artifact(
            spec=tiny_artifact.spec,
            frame=tiny_artifact.frame,
            shift_constant=tiny_artifact.shift_constant,
            cores=tiny_artifact.cores,
            target_identity=tiny_artifact.target_identity,
            artifact_id=tiny_artifact.artifact_id,
            diagnostics=diagnostics,
        )


def test_t1_artifact_payload_has_expected_schema(tiny_artifact) -> None:
    payload = tiny_artifact.payload()
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["spec"]["active_target_id"].startswith("zhao_cui_austria_sir")
    assert payload["spec"]["active_observation_sha256"] == (
        "cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07"
    )
    assert payload["diagnostics"]["training_summary"]["normalizer"] is not None

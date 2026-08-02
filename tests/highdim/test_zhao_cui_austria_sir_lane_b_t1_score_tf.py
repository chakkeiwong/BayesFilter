from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t1_score_tf import (
    LaneBT1TangentTrainer,
    LaneBT1TangentTrainingConfig,
    estimate_t1_fisher_score,
    generate_t1_score_batch,
    load_t1_score_artifact,
    make_t1_score_artifact,
    save_t1_score_artifact,
    tangent_validation_metrics,
    tangent_workspace_estimate_bytes,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
T1_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


@pytest.fixture(scope="module")
def parent():
    return load_lane_b_t1_artifact_v1_compat(T1_DIR)


def test_t1_complete_data_score_matches_diagnostic_tape(parent) -> None:
    batch = generate_t1_score_batch(
        parent=parent, sample_count=4, seed=74001, role="local_score_diagnostic"
    )
    model = latent_preclip_zhao_cui_sir_austria_model()
    _states, observations, _all = generate_sealed_lane_b_dataset()
    theta = tf.Variable(tf.zeros([3], tf.float64))
    z1 = batch.physical_points[:, :18]
    z0 = batch.physical_points[:, 18:]
    with tf.GradientTape() as tape:
        values = (
            model.initial_log_density(theta, z0)
            + model.transition_log_density(theta, z0, z1, 1)
            + model.observation_log_density(theta, z1, observations[0], 1)
        )
    diagnostic = tape.jacobian(values, theta)
    tf.debugging.assert_near(batch.target_score, diagnostic, atol=2e-12, rtol=2e-12)


def test_score_roles_are_disjoint_and_validation_reuses_only_training_scale(parent) -> None:
    training = generate_t1_score_batch(
        parent=parent, sample_count=8, seed=74011, role="score_training"
    )
    validation = generate_t1_score_batch(
        parent=parent,
        sample_count=8,
        seed=74012,
        role="score_validation",
        score_scale=training.score_scale,
    )
    assert training.seed != validation.seed
    assert training.role != validation.role
    assert (
        training.manifest_payload()["physical_points_sha256"]
        != validation.manifest_payload()["physical_points_sha256"]
    )
    tf.debugging.assert_near(validation.score_scale, training.score_scale, atol=0.0)


def test_fisher_score_estimate_is_finite_with_positive_mcse(parent) -> None:
    batch = generate_t1_score_batch(
        parent=parent, sample_count=32, seed=74021, role="score_calibration"
    )
    estimate = estimate_t1_fisher_score(batch)
    tf.debugging.assert_all_finite(estimate.score, "score")
    tf.debugging.assert_positive(estimate.standard_error)
    assert 1.0 <= float(estimate.effective_sample_size) <= 32.0
    assert estimate.score_sha256 == batch.manifest_payload()["target_score_sha256"]


def test_zero_tangent_initialization_has_nonzero_finite_data_gradients(parent) -> None:
    batch = generate_t1_score_batch(
        parent=parent, sample_count=8, seed=74031, role="gradient_diagnostic"
    )
    trainer = LaneBT1TangentTrainer(parent)
    with tf.GradientTape() as tape:
        prediction = trainer.unnormalized_log_density_score(batch.local_points)
        residual = (prediction - batch.target_score) / batch.score_scale[tf.newaxis, :]
        loss = tf.reduce_sum(
            batch.target_weights[:, tf.newaxis] * tf.square(residual)
        )
    gradients = tape.gradient(loss, trainer.variables)
    assert all(value is not None for value in gradients)
    tf.debugging.assert_all_finite(tf.linalg.global_norm(gradients), "gradient norm")
    tf.debugging.assert_positive(tf.linalg.global_norm(gradients))


def test_frozen_child_matches_trainer_score_and_preserves_parent(parent) -> None:
    batch = generate_t1_score_batch(
        parent=parent, sample_count=6, seed=74041, role="freeze_diagnostic"
    )
    trainer = LaneBT1TangentTrainer(parent)
    for axis, bank in enumerate(trainer.cores):
        for parameter, value in enumerate(bank):
            value.assign(
                tf.ones_like(value)
                * tf.constant(1e-5 * (parameter + 1) / (axis + 1), tf.float64)
            )
    before = tuple(tf.identity(core) for core in parent.cores)
    trainer_score = trainer.unnormalized_log_density_score(batch.local_points)
    child = trainer.freeze_child()
    _log_density, child_score = child.unnormalized_log_density_and_score(
        tf.zeros([3], tf.float64), batch.local_points
    )
    tf.debugging.assert_near(trainer_score, child_score, atol=3e-12, rtol=3e-12)
    child_value, _child_increment_score = child.increment_and_score(
        tf.zeros([3], tf.float64)
    )
    tf.debugging.assert_near(child_value, parent.value(), atol=2e-13)
    for observed, expected in zip(parent.cores, before):
        tf.debugging.assert_near(observed, expected, atol=0.0)
    assert tangent_workspace_estimate_bytes(parent=parent, batch_size=512) < 6 * 1024**3


def test_training_fisher_gauge_exactly_calibrates_manual_normalizer_score(parent) -> None:
    trainer = LaneBT1TangentTrainer(parent)
    before = tuple(tf.identity(core) for core in parent.cores)
    requested = tf.constant([-5.25, 1.75, -4.5], tf.float64)
    alpha = trainer.calibrate_normalizer_score(requested)
    tf.debugging.assert_all_finite(alpha, "calibration alpha")
    child_value, child_score = trainer.freeze_child().increment_and_score(
        tf.zeros([3], tf.float64)
    )
    tf.debugging.assert_near(child_value, parent.value(), atol=2e-13)
    tf.debugging.assert_near(child_score, requested, atol=3e-12, rtol=3e-12)
    for observed, expected in zip(parent.cores, before):
        tf.debugging.assert_near(observed, expected, atol=0.0)


def test_t1_score_parent_identity_fails_closed_on_other_parent_type() -> None:
    with pytest.raises(TypeError, match="LaneBT1Artifact"):
        LaneBT1TangentTrainer(object())


def test_trained_score_artifact_roundtrips_and_tamper_fails_closed(
    parent, tmp_path: Path
) -> None:
    training = generate_t1_score_batch(
        parent=parent, sample_count=8, seed=74051, role="score_training"
    )
    validation = generate_t1_score_batch(
        parent=parent,
        sample_count=8,
        seed=74052,
        role="score_validation",
        score_scale=training.score_scale,
    )
    trainer = LaneBT1TangentTrainer(parent)
    for axis, bank in enumerate(trainer.cores):
        for parameter, value in enumerate(bank):
            value.assign(
                tf.ones_like(value)
                * tf.constant(1e-6 * (parameter + 1) / (axis + 1), tf.float64)
            )
    config = LaneBT1TangentTrainingConfig(
        arm_id="unit_test_l1_comparator",
        learning_rate=1e-4,
        l1_weight=0.0,
        l2_weight=1e-10,
        gradient_clip_norm=10.0,
        batch_size=4,
        train_steps=1,
        seed=74053,
    )
    metrics = tangent_validation_metrics(trainer, validation)
    artifact = make_t1_score_artifact(
        trainer=trainer,
        config=config,
        training_batch=training,
        validation_batch=validation,
        validation_metrics=metrics,
    )
    output = tmp_path / "score_artifact"
    save_t1_score_artifact(artifact, output)
    reloaded = load_t1_score_artifact(output, parent=parent)
    assert reloaded.identity == artifact.identity
    assert reloaded.child().identity == artifact.child().identity
    original = (output / "tangent_00_0.tensor").read_bytes()
    (output / "tangent_00_0.tensor").write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_t1_score_artifact(output, parent=parent)

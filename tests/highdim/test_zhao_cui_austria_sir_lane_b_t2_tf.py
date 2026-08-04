from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (
    LaneBT2ProposalCloud,
    build_t2_frame,
    build_t2_training_batch,
    estimate_t2_shifted_log_normalizer,
    generate_t2_proposal_cloud,
    load_lane_b_t2_artifact,
    make_lane_b_t2_artifact,
    make_t2_compiled_train_step,
    save_lane_b_t2_artifact,
    select_t2_shift_constant,
    t2_log_weight_cross_entropy,
    verify_b2_admission,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    LaneBT1Settings,
    balanced_initial_cores,
    calibrate_trainer_normalizer,
)
from bayesfilter.highdim.stochastic_density_training import TrainableFunctionalTT
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_prepared_tf import (
    _verify_prepared_source_closure,
    prepared_source_closure,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import t2_trainer_config


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _settings() -> LaneBT1Settings:
    return LaneBT1Settings(
        arm_id="t2-test",
        rank=1,
        basis_order=1,
        basis_num_elems=1,
        learning_rate=1e-4,
        l1_weight=1e-10,
        l2_weight=1e-8,
        batch_size=2,
        train_steps=2,
        expansion_factor=4.0,
        covariance_jitter=1e-5,
        quantile_fraction=0.01,
        use_quantile_scale=False,
        tau=1e-8,
        gradient_clip_norm=10.0,
        cdf_grid_size=65,
        cdf_bisection_steps=24,
        cdf_max_working_bytes=512 * 1024 * 1024,
    )


def _cloud():
    artifact = load_lane_b_t1_artifact_v1_compat(ARTIFACT)
    reference = tf.random.stateless_uniform(
        [18, 4], seed=[73891, 1], minval=1e-6, maxval=1.0 - 1e-6, dtype=tf.float64
    )
    return generate_t2_proposal_cloud(
        t1_artifact=artifact,
        reference_uniforms=reference,
        reference_seed=73891,
        transition_seed=73892,
        role="unit",
    )


def test_b2_admission_hash_is_bound() -> None:
    assert len(verify_b2_admission(ROOT)) == 64


def test_t2_prepared_source_closure_rejects_stale_code() -> None:
    hashes = dict(prepared_source_closure())
    payload = {"run_manifest": {"source_sha256": hashes}}
    _verify_prepared_source_closure(payload)
    payload["run_manifest"]["source_sha256"] = dict(hashes)
    first_path = next(iter(hashes))
    payload["run_manifest"]["source_sha256"][first_path] = "0" * 64
    with pytest.raises(ValueError, match="source closure mismatch"):
        _verify_prepared_source_closure(payload)


def test_t2_cloud_carries_exact_proposal_correction_and_transition_cancellation() -> None:
    cloud = _cloud()
    tf.debugging.assert_near(
        cloud.log_target_physical - cloud.log_proposal_physical,
        cloud.log_importance_weight,
        atol=2e-12,
    )
    tf.debugging.assert_near(
        cloud.log_importance_weight,
        cloud.previous_correction + cloud.log_likelihood,
        atol=0.0,
    )
    assert cloud.manifest_payload()["joint_axis_order"] == ("z2", "z1")


def test_t2_cloud_accepts_negative_infinite_zero_density_and_rejects_invalid_infinity() -> None:
    cloud = _cloud()
    likelihood = tf.tensor_scatter_nd_update(
        cloud.log_likelihood, [[0]], [tf.constant(float("-inf"), tf.float64)]
    )
    zero = LaneBT2ProposalCloud(
        joint_points=cloud.joint_points,
        previous_log_target=cloud.previous_log_target,
        previous_log_proposal=cloud.previous_log_proposal,
        previous_correction=cloud.previous_correction,
        transition_log_density=cloud.transition_log_density,
        log_likelihood=likelihood,
        reference_uniforms=cloud.reference_uniforms,
        reference_seed=cloud.reference_seed,
        transition_seed=cloud.transition_seed,
        role=cloud.role,
    )
    assert int(tf.reduce_sum(tf.cast(zero.zero_target_mask, tf.int32)).numpy()) == 1
    estimate = estimate_t2_shifted_log_normalizer(zero, tf.constant(0.0, tf.float64))
    assert bool(tf.math.is_finite(estimate.log_increment).numpy())

    import pytest

    for invalid in (float("nan"), float("inf")):
        bad = tf.tensor_scatter_nd_update(
            cloud.log_likelihood, [[0]], [tf.constant(invalid, tf.float64)]
        )
        with pytest.raises(ValueError, match="NaN or positive infinity"):
            LaneBT2ProposalCloud(
                joint_points=cloud.joint_points,
                previous_log_target=cloud.previous_log_target,
                previous_log_proposal=cloud.previous_log_proposal,
                previous_correction=cloud.previous_correction,
                transition_log_density=cloud.transition_log_density,
                log_likelihood=bad,
                reference_uniforms=cloud.reference_uniforms,
                reference_seed=cloud.reference_seed,
                transition_seed=cloud.transition_seed,
                role=cloud.role,
            )


def test_t2_shift_and_measure_correct_batch_are_finite() -> None:
    cloud = _cloud()
    settings = _settings()
    frame = build_t2_frame(cloud, settings)
    shift = select_t2_shift_constant(cloud)
    estimate = estimate_t2_shifted_log_normalizer(cloud, shift)
    tf.debugging.assert_near(estimate.log_shifted_normalizer, 0.0, atol=2e-12)
    batch = build_t2_training_batch(cloud, frame, shift)
    assert batch.points.shape == (4, 36)
    assert bool(tf.reduce_all(tf.math.is_finite(batch.log_target_reference)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(batch.log_importance_weight)).numpy())


def test_accumulated_microbatch_gradients_equal_full_cloud_gradients() -> None:
    log_rho = tf.constant([-2.0, -0.5, -4.0, -1.25], tf.float64)
    log_weight = tf.constant([-10.0, -1.0, float("-inf"), -3.0], tf.float64)
    global_lse = tf.reduce_logsumexp(log_weight)
    variable = tf.Variable(log_rho)
    normalizer_parameter = tf.Variable(tf.constant(0.7, tf.float64))

    def loss(first: int, last: int, scale: float) -> tf.Tensor:
        cross_entropy = t2_log_weight_cross_entropy(
            log_rho=variable[first:last],
            log_importance_weight=log_weight[first:last],
            global_log_weight_sum=global_lse,
            scale=scale,
        )
        log_normalizer = tf.math.log(tf.square(normalizer_parameter) + 1.0)
        regularization = tf.constant(0.03, tf.float64) * (
            tf.reduce_sum(tf.abs(variable)) + tf.square(normalizer_parameter)
        )
        return cross_entropy + log_normalizer + regularization

    with tf.GradientTape() as tape:
        full = loss(0, 4, 1.0)
    full_gradients = tape.gradient(full, (variable, normalizer_parameter))
    rows = []
    values = []
    for first in (0, 2):
        with tf.GradientTape() as tape:
            piece = loss(first, first + 2, 2.0)
        rows.append(tape.gradient(piece, (variable, normalizer_parameter)))
        values.append(piece)
    accumulated = tuple(
        tf.add_n([row[index] for row in rows]) / 2.0 for index in range(2)
    )
    tf.debugging.assert_near(tf.reduce_mean(tf.stack(values)), full, atol=2e-15)
    for observed, expected in zip(accumulated, full_gradients):
        tf.debugging.assert_near(observed, expected, atol=2e-15)


def test_t2_train_step_rejects_nondivisible_full_cloud() -> None:
    settings = _settings()
    config = t2_trainer_config(settings)
    trainer = TrainableFunctionalTT(
        config,
        initial_cores=balanced_initial_cores(settings, config.product_basis),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    step = make_t2_compiled_train_step(
        trainer, optimizer, microbatch_size=3
    )
    with pytest.raises(ValueError, match="divide exactly"):
        step(tf.zeros([4, 36], tf.float64), tf.zeros([4], tf.float64))


def test_t2_accumulated_step_matches_direct_full_cloud_update() -> None:
    settings = _settings()
    config = t2_trainer_config(settings)
    initial = balanced_initial_cores(settings, config.product_basis)
    direct = TrainableFunctionalTT(config, initial_cores=initial)
    accumulated = TrainableFunctionalTT(config, initial_cores=initial)
    direct_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    accumulated_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate
    )
    if hasattr(direct_optimizer, "build"):
        direct_optimizer.build(direct.variables)
    points = tf.reshape(
        tf.linspace(tf.constant(-0.2, tf.float64), tf.constant(0.2, tf.float64), 144),
        [4, 36],
    )
    log_weight = tf.constant([-10.0, -1.0, float("-inf"), -3.0], tf.float64)

    with tf.GradientTape() as tape:
        rho = direct.rho_theta(points)
        alpha = tf.nn.softmax(log_weight)
        cross_entropy = -tf.reduce_sum(alpha * tf.math.log(rho))
        log_normalizer = tf.math.log(direct.normalizer())
        l1 = tf.add_n([tf.reduce_sum(tf.abs(core)) for core in direct.variables])
        l2 = tf.add_n([tf.reduce_sum(tf.square(core)) for core in direct.variables])
        regularization = config.l1_weight * l1 + config.l2_weight * l2
        total = cross_entropy + log_normalizer + regularization
    direct_gradients = tape.gradient(total, direct.variables)
    direct_clipped, direct_norm = tf.clip_by_global_norm(
        direct_gradients, tf.constant(config.gradient_clip_norm, tf.float64)
    )
    direct_optimizer.apply_gradients(zip(direct_clipped, direct.variables))

    step = make_t2_compiled_train_step(
        accumulated, accumulated_optimizer, microbatch_size=2
    )
    terms = step(points, log_weight)
    tf.debugging.assert_near(terms[0], total, atol=2e-12)
    tf.debugging.assert_near(terms[4], direct_norm, atol=2e-12)
    for observed, expected in zip(accumulated.variables, direct.variables):
        tf.debugging.assert_near(observed, expected, atol=2e-12)


def test_t2_artifact_roundtrips_identity_and_rejects_tamper(tmp_path: Path) -> None:
    parent = load_lane_b_t1_artifact_v1_compat(ARTIFACT)
    cloud = _cloud()
    settings = _settings()
    frame = build_t2_frame(cloud, settings)
    shift = select_t2_shift_constant(cloud)
    estimate = estimate_t2_shifted_log_normalizer(cloud, shift)
    config = t2_trainer_config(settings)
    trainer = TrainableFunctionalTT(
        config,
        initial_cores=balanced_initial_cores(settings, config.product_basis),
    )
    calibrate_trainer_normalizer(trainer, estimate.log_shifted_normalizer)
    artifact = make_lane_b_t2_artifact(
        parent_artifact=parent,
        settings=settings,
        frame=frame,
        trainer=trainer,
        shift_constant=shift,
        calibration_estimate=estimate,
        validation_estimate=estimate,
        training_cloud_manifest=cloud.manifest_payload(),
        validation_cloud_manifest=cloud.manifest_payload(),
    )
    output = tmp_path / "t2-artifact"
    save_lane_b_t2_artifact(artifact, output)
    reloaded = load_lane_b_t2_artifact(output, parent_artifact=parent)
    assert reloaded.identity == artifact.identity
    tf.debugging.assert_near(reloaded.increment(), artifact.increment(), atol=0.0)
    tf.debugging.assert_near(reloaded.value(), artifact.value(), atol=0.0)

    core_path = output / "core_00.tensor"
    corrupted = bytearray(core_path.read_bytes())
    corrupted[-1] ^= 1
    core_path.write_bytes(bytes(corrupted))
    import pytest

    with pytest.raises(ValueError, match="tensor hash mismatch"):
        load_lane_b_t2_artifact(output, parent_artifact=parent)

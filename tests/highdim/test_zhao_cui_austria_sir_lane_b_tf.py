from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame
from bayesfilter.highdim.stochastic_density_training import (
    TrainableFunctionalTT,
    make_adam_optimizer,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    SIR_OBSERVATION_SHA256,
    SIR_RUNTIME_FP32_OBSERVATION_SHA256,
    generate_sealed_lane_b_dataset,
    generate_t1_proposal_cloud,
    target_manifest,
    tensor_sha256,
    t1_joint_log_density,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    BASELINE_ID,
    LOG_REFERENCE_DENSITY_CONSTANT,
    LaneBLogNormalizerEstimate,
    LaneBT1Settings,
    balanced_initial_cores,
    build_lane_b_frame,
    build_training_batch,
    calibrate_trainer_normalizer,
    estimate_shifted_log_normalizer,
    lane_b_product_basis,
    load_lane_b_t1_artifact,
    make_compiled_train_step,
    make_lane_b_t1_artifact,
    normalizer_estimates_agree,
    save_lane_b_t1_artifact,
    select_shift_constant,
    t1_log_density_relative_to_reference_measure,
    trainer_config,
)


DTYPE = tf.float64


def _settings(*, rank: int = 1) -> LaneBT1Settings:
    return LaneBT1Settings(
        arm_id="unit-rank1-order1",
        rank=rank,
        basis_order=1,
        basis_num_elems=1,
        learning_rate=1e-4,
        l1_weight=1e-10,
        l2_weight=1e-8,
        batch_size=8,
        train_steps=2,
        expansion_factor=4.0,
        covariance_jitter=1e-5,
        quantile_fraction=0.01,
        use_quantile_scale=False,
        tau=1e-8,
        gradient_clip_norm=10.0,
        cdf_grid_size=9,
        cdf_bisection_steps=4,
        cdf_max_working_bytes=4 * 1024 * 1024,
    )


def _equal_estimate(role: str, log_normalizer: float) -> LaneBLogNormalizerEstimate:
    return LaneBLogNormalizerEstimate(
        role=role,
        seed=701 if role == "calibration" else 702,
        sample_count=128,
        shift_constant=tf.constant(0.0, DTYPE),
        log_evidence=tf.constant(log_normalizer, DTYPE),
        log_shifted_normalizer=tf.constant(log_normalizer, DTYPE),
        log_standard_error=tf.constant(1e-3, DTYPE),
        log_likelihood_sha256=("a" if role == "calibration" else "b") * 64,
    )


def test_sealed_dataset_hashes_and_event_order() -> None:
    states, observations, all_observations = generate_sealed_lane_b_dataset()
    assert states.shape == (21, 18)
    assert observations.shape == (20, 9)
    assert tensor_sha256(observations) == SIR_OBSERVATION_SHA256
    assert (
        tensor_sha256(tf.cast(observations, tf.float32))
        == SIR_RUNTIME_FP32_OBSERVATION_SHA256
    )
    assert bool(tf.reduce_all(observations == all_observations[1:]).numpy())
    assert target_manifest()["event_order"] == (
        "z0_then_transition_to_z1_then_observe_sealed_y1"
    )


def test_latent_t1_cloud_has_coherent_joint_density() -> None:
    cloud = generate_t1_proposal_cloud(sample_count=16, seed=7101, role="test")
    log_joint = t1_joint_log_density(cloud.joint_points)
    assert log_joint.shape == (16,)
    assert bool(tf.reduce_all(tf.math.is_finite(log_joint)).numpy())
    # Under the proposal p0 f, the remaining log-density factor is exactly g(y1|z1).
    from bayesfilter.highdim.sir_latent_preclip_tf import (
        latent_preclip_zhao_cui_sir_austria_model,
    )

    model = latent_preclip_zhao_cui_sir_austria_model()
    theta = tf.zeros([3], DTYPE)
    z1, z0 = cloud.joint_points[:, :18], cloud.joint_points[:, 18:]
    log_proposal = model.initial_log_density(theta, z0) + model.transition_log_density(
        theta, z0, z1, 1
    )
    tf.debugging.assert_near(log_joint - log_proposal, cloud.log_likelihood, atol=1e-10)


def test_reference_measure_conversion_contains_exact_2_power_36_factor() -> None:
    cloud = generate_t1_proposal_cloud(sample_count=8, seed=7102, role="test")
    frame = SourceRouteCoordinateFrame(
        mu=tf.zeros([36], DTYPE),
        matrix=tf.eye(36, dtype=DTYPE),
        expansion_factor=1.0,
    )
    terms = t1_log_density_relative_to_reference_measure(cloud.joint_points, frame)
    expected = tf.fill([8], tf.constant(36.0 * math.log(2.0), DTYPE))
    tf.debugging.assert_near(terms["log_inverse_reference_density"], expected, atol=0.0)
    recomposed = (
        terms["log_physical_density"]
        + terms["log_affine_jacobian"]
        + terms["log_algebraic_jacobian"]
        + terms["log_inverse_reference_density"]
    )
    tf.debugging.assert_near(
        recomposed,
        terms["log_density_relative_to_reference_measure"],
        atol=1e-12,
    )
    assert LOG_REFERENCE_DENSITY_CONSTANT == pytest.approx(36.0 * math.log(2.0))


def test_target_and_baseline_source_exclude_drifted_dependencies() -> None:
    import bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf as target_module
    import bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf as baseline_module

    target_source = inspect.getsource(target_module).lower()
    baseline_source = inspect.getsource(baseline_module).lower()
    forbidden_imports = (
        "zhao_cui_austria_sir_fixed_variant_tf",
        "zhao_cui_austria_sir_proposal_tf",
        "zhao_cui_austria_sir_source_replica_tf",
        "zhao_cui_frozen_proposal_apf_tf",
        "zhao_cui_moment_teacher_als",
        "multistate_nonlinear_fixed_design_tt",
    )
    for token in forbidden_imports:
        assert token not in target_source
        assert token not in baseline_source
    assert BASELINE_ID in baseline_source


def test_mc_normalizer_estimates_use_disjoint_clouds_and_uncertainty() -> None:
    calibration_cloud = generate_t1_proposal_cloud(
        sample_count=256, seed=7201, role="calibration"
    )
    validation_cloud = generate_t1_proposal_cloud(
        sample_count=256, seed=7202, role="validation"
    )
    shift = tf.constant(3.0, DTYPE)
    calibration = estimate_shifted_log_normalizer(calibration_cloud, shift)
    validation = estimate_shifted_log_normalizer(validation_cloud, shift)
    assert calibration.log_likelihood_sha256 != validation.log_likelihood_sha256
    assert float(calibration.log_standard_error.numpy()) > 0.0
    assert float(validation.log_standard_error.numpy()) > 0.0
    assert normalizer_estimates_agree(calibration, calibration)


def test_measure_correct_training_batch_is_finite_and_positive() -> None:
    cloud = generate_t1_proposal_cloud(sample_count=32, seed=7301, role="training")
    settings = _settings()
    frame = build_lane_b_frame(cloud, settings)
    calibration = generate_t1_proposal_cloud(
        sample_count=64, seed=7302, role="calibration"
    )
    shift = select_shift_constant(calibration, frame)
    batch = build_training_batch(cloud, frame, shift)
    assert batch.points.shape == (32, 36)
    assert bool(tf.reduce_all(tf.math.is_finite(batch.points)).numpy())
    assert bool(tf.reduce_all(batch.target_sqrt_values > 0.0).numpy())
    assert bool(tf.reduce_all(batch.integration_weights > 0.0).numpy())
    estimate = estimate_shifted_log_normalizer(calibration, shift)
    assert float(estimate.log_shifted_normalizer.numpy()) == pytest.approx(0.0, abs=1e-12)


def test_exact_core_rescale_recovers_requested_normalizer() -> None:
    settings = _settings()
    config = trainer_config(settings)
    trainer = TrainableFunctionalTT(
        config, initial_cores=balanced_initial_cores(settings, config.product_basis)
    )
    target_log_normalizer = tf.constant(0.25, DTYPE)
    scale = calibrate_trainer_normalizer(trainer, target_log_normalizer)
    assert float(scale.numpy()) > 0.0
    tf.debugging.assert_near(
        tf.math.log(trainer.normalizer()), target_log_normalizer, atol=1e-12
    )


def test_compiled_training_kernel_matches_eager_training_base_update() -> None:
    settings = _settings()
    config = trainer_config(settings)
    eager = TrainableFunctionalTT(
        config, initial_cores=balanced_initial_cores(settings, config.product_basis)
    )
    initial = tuple(tf.identity(core) for core in eager.variables)
    compiled = TrainableFunctionalTT(config, initial_cores=initial)
    cloud = generate_t1_proposal_cloud(sample_count=8, seed=7401, role="training")
    frame = build_lane_b_frame(cloud, settings)
    shift = select_shift_constant(cloud, frame)
    training = build_training_batch(cloud, frame, shift)
    batch = training.objective_batch()
    eager_optimizer = make_adam_optimizer(config)
    compiled_optimizer = make_adam_optimizer(config)
    eager_terms = eager.train_step(batch, eager_optimizer)
    compiled_step = make_compiled_train_step(compiled, compiled_optimizer)
    compiled_terms = compiled_step(
        training.points,
        training.target_sqrt_values,
        training.integration_weights,
    )
    tf.debugging.assert_near(compiled_terms[0], eager_terms.total_loss, atol=1e-10)
    tf.debugging.assert_near(
        compiled_terms[1], eager_terms.weighted_empirical_cross_entropy, atol=1e-10
    )
    tf.debugging.assert_near(compiled_terms[2], eager_terms.log_normalizer, atol=1e-10)
    for eager_core, compiled_core in zip(eager.variables, compiled.variables):
        tf.debugging.assert_near(eager_core, compiled_core, atol=1e-10)


def test_artifact_reload_identity_and_tensor_tamper_rejection(tmp_path: Path) -> None:
    settings = _settings()
    config = trainer_config(settings)
    trainer = TrainableFunctionalTT(
        config, initial_cores=balanced_initial_cores(settings, config.product_basis)
    )
    log_normalizer = 0.1
    calibrate_trainer_normalizer(trainer, tf.constant(log_normalizer, DTYPE))
    calibration = _equal_estimate("calibration", log_normalizer)
    validation = _equal_estimate("validation", log_normalizer)
    frame = SourceRouteCoordinateFrame(
        mu=tf.zeros([36], DTYPE),
        matrix=tf.eye(36, dtype=DTYPE),
        expansion_factor=settings.expansion_factor,
    )
    references = tf.fill([36, 2], tf.constant(0.5, DTYPE))
    artifact = make_lane_b_t1_artifact(
        settings=settings,
        frame=frame,
        trainer=trainer,
        shift_constant=tf.constant(0.0, DTYPE),
        calibration_estimate=calibration,
        validation_estimate=validation,
        frozen_reference_points=references,
        training_cloud_manifest={
            "role": "training",
            "seed": 801,
            "joint_axis_order": ("z1", "z0"),
        },
        validation_cloud_manifest={
            "role": "validation",
            "seed": 802,
            "joint_axis_order": ("z1", "z0"),
        },
    )
    output = tmp_path / "lane-b-artifact"
    save_lane_b_t1_artifact(artifact, output)
    reloaded = load_lane_b_t1_artifact_v1_compat(output)
    assert reloaded.identity == artifact.identity
    tf.debugging.assert_near(reloaded.value(), artifact.value(), atol=0.0)

    core_path = output / "core_00.tensor"
    corrupted = bytearray(core_path.read_bytes())
    corrupted[-1] ^= 1
    core_path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        load_lane_b_t1_artifact_v1_compat(output)


def test_balanced_initializer_has_unit_mass_and_nonconstant_seeded_path() -> None:
    settings = _settings(rank=2)
    config = trainer_config(settings)
    trainer = TrainableFunctionalTT(
        config, initial_cores=balanced_initial_cores(settings, config.product_basis)
    )
    mass = trainer.sqrt_square_normalizer()
    assert 0.5 <= float(mass.numpy()) <= 2.0
    points = tf.random.stateless_normal([16, 36], seed=[91, 92], dtype=DTYPE)
    values = trainer.evaluate(points)
    assert float(tf.math.reduce_std(values).numpy()) > 1e-12


@pytest.mark.parametrize(("order", "num_elems"), ((1, 2), (2, 2)))
def test_lane_b_frozen_mass_constants_match_cpu_basis(order: int, num_elems: int) -> None:
    from bayesfilter.highdim.bases import (
        p85_author_sir_lagrangep_algebraic_product_basis_spec,
    )
    from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
        lane_b_measure_convention,
    )

    with tf.device("/CPU:0"):
        raw = p85_author_sir_lagrangep_algebraic_product_basis_spec(
            dimension=1,
            convention=lane_b_measure_convention(),
            order=order,
            num_elems=num_elems,
        ).build_product_basis().bases[0]
        expected_mass = raw.mass_matrix(MassMeasure.REFERENCE_MEASURE)
        expected_integral = raw.integral_vector(MassMeasure.REFERENCE_MEASURE)
    frozen = lane_b_product_basis(order=order, num_elems=num_elems).bases[0]
    tf.debugging.assert_near(
        frozen.mass_matrix(MassMeasure.REFERENCE_MEASURE), expected_mass, atol=1e-15
    )
    tf.debugging.assert_near(
        frozen.integral_vector(MassMeasure.REFERENCE_MEASURE),
        expected_integral,
        atol=1e-15,
    )

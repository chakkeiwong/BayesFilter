from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (
    CenteredThetaFeatures,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_density_training_tf import (
    CenteredResidualTrainer,
    CoreAffineTangentTrainer,
    _additive_rank_two_component,
    _additive_pair_rank_seven_component,
    _within_region_feature_values,
    _within_region_cross_features,
    additive_prefix_score_operator,
    batch_native_t1_from_common_noise,
    build_t1_parameter_density_batch,
    core_affine_released_coordinate_mask,
    core_affine_origin_total_score_loss_arrays,
    core_tangent_banks_from_residual_components,
    core_tangent_to_residual_component,
    estimate_t1_prefix_scores,
    estimate_t1_ratio_score,
    embed_residual_component_at_rank,
    embed_residual_component_with_connected_channels,
    fixed_rank_initial_residual_components,
    make_compiled_absolute_train_step,
    make_compiled_core_affine_gate_minimax_value_and_gradient,
    make_compiled_full_tt_gate_minimax_value_and_gradient,
    make_compiled_core_affine_total_score_value_and_gradient,
    make_compiled_origin_score_prefit_step,
    make_compiled_origin_total_score_train_step,
    rotating_prefix_minibatch_indices,
    residual_components_from_position,
    residual_components_position,
    solve_quadratic_value_gradient_with_conjugate_gradient,
    target_informed_additive_score_initialization,
    within_region_pair_prefix_score_operator,
)


ROOT = Path(__file__).resolve().parents[2]
T1_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _parent():
    return load_lane_b_t1_artifact_v1_compat(T1_DIR)


def _noise(sample_count: int = 16):
    return (
        tf.random.stateless_normal(
            [sample_count, 18], [8101, 1], dtype=tf.float64
        ),
        tf.random.stateless_normal(
            [sample_count, 18], [8101, 2], dtype=tf.float64
        ),
    )


def _theta():
    return tf.constant(
        [
            [0.0, 0.0, 0.0],
            [0.03, -0.02, 0.01],
            [-0.03, 0.02, -0.01],
        ],
        tf.float64,
    )


def test_batch_native_target_and_score_match_scalar_theta_authority() -> None:
    initial_noise, transition_noise = _noise()
    _states, observations, _all = generate_sealed_lane_b_dataset()
    theta = _theta()
    result = batch_native_t1_from_common_noise(
        theta, initial_noise, transition_noise, observations[0]
    )
    model = latent_preclip_zhao_cui_sir_austria_model()
    expected_values = []
    expected_scores = []
    for index in range(int(theta.shape[0])):
        z0 = result["z0"][index]
        z1 = result["z1"][index]
        parameter = theta[index]
        expected_values.append(
            model.initial_log_density(parameter, z0)
            + model.transition_log_density(parameter, z0, z1, 1)
            + model.observation_log_density(parameter, z1, observations[0], 1)
        )
        expected_scores.append(
            model.initial_log_density_parameter_score(parameter, z0)
            + model.transition_log_density_parameter_score(parameter, z0, z1, 1)
            + model.observation_log_density_parameter_score(
                parameter, z1, observations[0], 1
            )
        )
    tf.debugging.assert_near(
        result["complete_log_density"], tf.stack(expected_values), atol=2e-13
    )
    tf.debugging.assert_near(
        result["complete_data_score"], tf.stack(expected_scores), atol=6e-13
    )


def test_parameter_batch_binds_physical_chart_and_absolute_weight_identity() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise()
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=_theta(),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="test",
    )
    assert batch.physical_points.shape == (3, 16, 36)
    assert batch.local_points.shape == (3, 16, 36)
    assert batch.reference_points.shape == (3, 16, 36)
    assert batch.complete_data_score.shape == (3, 16, 3)

    trainer = CenteredResidualTrainer(parent)
    terms = trainer.absolute_density_loss(batch)
    expected_mass = tf.reduce_mean(
        tf.exp(parent.shift_constant + batch.observation_log_density), axis=1
    )
    tf.debugging.assert_near(terms.target_mass_estimate, expected_mass, atol=0.0)
    # Normalizing the weights would force every estimate to one; this batch must
    # retain its absolute scale.
    assert bool(tf.reduce_any(tf.abs(expected_mass - 1.0) > 1e-4).numpy())


def test_derivative_loss_fails_closed_without_complete_origin_inputs() -> None:
    parent = _parent()
    trainer = CenteredResidualTrainer(parent)
    theta = tf.zeros([1, 3], tf.float64)
    points = tf.zeros([1, 2, 36], tf.float64)
    weights = tf.zeros([1, 2], tf.float64)
    with pytest.raises(tf.errors.InvalidArgumentError):
        trainer.absolute_density_loss_arrays(
            theta,
            points,
            weights,
            derivative_weight=0.25,
        )
    with pytest.raises(ValueError, match="all three origin arrays"):
        trainer.absolute_density_loss_arrays(
            theta,
            points,
            weights,
            derivative_points=points[0],
            derivative_weight=0.25,
        )


def test_fixed_rank_residual_initialization_is_deterministic_and_distinct() -> None:
    parent = _parent()
    features = CenteredThetaFeatures()
    first = fixed_rank_initial_residual_components(
        parent=parent, features=features, rank=2, seed=8117
    )
    second = fixed_rank_initial_residual_components(
        parent=parent, features=features, rank=2, seed=8117
    )
    assert len(first) == features.feature_count
    for left_component, right_component in zip(first, second):
        assert len(left_component) == 36
        for axis, (left, right) in enumerate(zip(left_component, right_component)):
            tf.debugging.assert_equal(left, right)
            assert left.shape[0] == (1 if axis == 0 else 2)
            assert left.shape[2] == (1 if axis == 35 else 2)
    assert bool(tf.reduce_any(first[0][0] != first[1][0]).numpy())


def test_target_informed_additive_initialization_reduces_real_origin_score_loss() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=256)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_additive_initialization_test",
    )
    baseline = CenteredResidualTrainer(parent)
    before = baseline.origin_point_score_metrics_arrays(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    fitted = target_informed_additive_score_initialization(
        parent=parent,
        local_points=batch.local_points[0],
        target_complete_data_score=batch.complete_data_score[0],
        importance_log_weight=batch.observation_log_density[0],
        ridge_fraction=1e-4,
        global_score_weight=1.0,
    )
    trainer = CenteredResidualTrainer(
        parent, initial_residual_components=fitted.residual_components
    )
    after = trainer.origin_point_score_metrics_arrays(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    tf.debugging.assert_less(after["loss"], before["loss"])
    global_relative_error = tf.abs(
        after["child_likelihood_score"] - after["target_likelihood_score"]
    ) / tf.maximum(tf.abs(after["target_likelihood_score"]), 1.0)
    tf.debugging.assert_less(
        tf.reduce_max(global_relative_error), tf.constant(0.5, tf.float64)
    )
    for component in fitted.residual_components:
        assert component[0].shape[2] == 2
        assert component[-1].shape[0] == 2
    tf.debugging.assert_near(
        trainer.freeze_child().increment_and_score(tf.zeros([3], tf.float64))[0],
        parent.value(),
        atol=2e-13,
    )


def test_additive_prefix_operator_matches_exact_finite_child_score() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_prefix_operator_test",
    )
    basis_dim = 5
    coefficients = tf.reshape(
        tf.linspace(
            tf.constant(-0.2, tf.float64),
            tf.constant(0.3, tf.float64),
            36 * basis_dim,
        ),
        [36, basis_dim],
    )
    component = _additive_rank_two_component(coefficients, (basis_dim,) * 36)
    zero = list(component)
    zero[0] = tf.zeros_like(zero[0])
    child = CenteredResidualTrainer(
        parent,
        initial_residual_components=(component, tuple(zero), tuple(zero)),
    ).freeze_child()
    local_prefix = batch.local_points[0, :3, :18]
    operator = additive_prefix_score_operator(
        parent=parent, local_prefix_points=local_prefix
    )
    _value, score = child.prefix_log_marginal_and_score(
        tf.zeros([3], tf.float64), local_prefix
    )
    expected = tf.linalg.matvec(operator, tf.reshape(coefficients, [-1]))
    tf.debugging.assert_near(score[:, 0], expected, atol=2e-11)
    tf.debugging.assert_near(score[:, 1:], tf.zeros([3, 2], tf.float64), atol=2e-11)


def test_within_region_pair_rank_seven_tt_matches_explicit_feature_matrix() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_pair_rank_test",
    )
    trainer = CenteredResidualTrainer(parent)
    feature_values = _within_region_feature_values(
        trainer.basis, batch.local_points[0]
    )
    coefficients = tf.linspace(
        tf.constant(-0.07, tf.float64),
        tf.constant(0.11, tf.float64),
        int(feature_values.shape[1]),
    )
    component = _additive_pair_rank_seven_component(coefficients, basis_dim=5)
    actual = __import__(
        "bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf",
        fromlist=["_evaluate_component"],
    )._evaluate_component(component, trainer.basis, batch.local_points[0])
    expected = tf.linalg.matvec(feature_values, coefficients)
    tf.debugging.assert_near(actual, expected, atol=3e-12)
    assert max(int(core.shape[0]) for core in component) == 7


def test_within_region_pair_score_operators_match_exact_finite_child() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_pair_operator_test",
    )
    trainer = CenteredResidualTrainer(parent)
    feature_values = _within_region_feature_values(
        trainer.basis, batch.local_points[0]
    )
    coefficient_count = int(feature_values.shape[1])
    coefficients = tf.linspace(
        tf.constant(-0.03, tf.float64),
        tf.constant(0.05, tf.float64),
        coefficient_count,
    )
    component = _additive_pair_rank_seven_component(coefficients, basis_dim=5)
    zero = list(component)
    zero[0] = tf.zeros_like(zero[0])
    child = CenteredResidualTrainer(
        parent,
        initial_residual_components=(component, tuple(zero), tuple(zero)),
    ).freeze_child()
    parent_amplitude = __import__(
        "bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf",
        fromlist=["_evaluate_component"],
    )._evaluate_component(parent.cores, trainer.basis, batch.local_points[0])
    parent_mass = __import__(
        "bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf",
        fromlist=["_cross_mass"],
    )._cross_mass(parent.cores, parent.cores, trainer.basis)
    cross = _within_region_cross_features(
        parent=parent, basis=trainer.basis, local_prefix_points=None
    )
    global_operator = 2.0 * cross / (parent_mass + parent.settings.tau)
    global_expected = tf.tensordot(global_operator, coefficients, axes=1)
    _value, global_score = child.increment_and_score(tf.zeros([3], tf.float64))
    tf.debugging.assert_near(global_score[0], global_expected, atol=2e-11)
    point_operator = (
        2.0
        * parent_amplitude[:, tf.newaxis]
        * feature_values
        / (tf.square(parent_amplitude) + parent.settings.tau)[:, tf.newaxis]
        - global_operator[tf.newaxis, :]
    )
    _log, point_score = child.point_log_density_and_score(
        tf.zeros([3], tf.float64), batch.local_points[0]
    )
    tf.debugging.assert_near(
        point_score[:, 0], tf.linalg.matvec(point_operator, coefficients), atol=2e-11
    )
    local_prefix = batch.local_points[0, :3, :18]
    prefix_operator = within_region_pair_prefix_score_operator(
        parent=parent, local_prefix_points=local_prefix
    )
    _prefix_value, prefix_score = child.prefix_log_marginal_and_score(
        tf.zeros([3], tf.float64), local_prefix
    )
    tf.debugging.assert_near(
        prefix_score[:, 0], tf.linalg.matvec(prefix_operator, coefficients), atol=2e-11
    )


def test_trainer_global_and_prefix_metrics_match_frozen_child() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=64)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_total_score_metric_test",
    )
    fitted = target_informed_additive_score_initialization(
        parent=parent,
        local_points=batch.local_points[0],
        target_complete_data_score=batch.complete_data_score[0],
        importance_log_weight=batch.observation_log_density[0],
        ridge_fraction=1e-4,
        global_score_weight=10.0,
    )
    trainer = CenteredResidualTrainer(
        parent, initial_residual_components=fitted.residual_components
    )
    child = trainer.freeze_child()
    theta = tf.zeros([3], tf.float64)
    _value, child_global = child.increment_and_score(theta)
    global_metrics = trainer.origin_global_score_metrics_arrays(
        child_global, tf.ones([3], tf.float64)
    )
    tf.debugging.assert_near(global_metrics["child_score"], child_global, atol=2e-12)
    local_prefix = batch.local_points[0, :3, :18]
    _prefix_value, child_prefix = child.prefix_log_marginal_and_score(
        theta, local_prefix
    )
    prefix_metrics = trainer.origin_prefix_score_metrics_arrays(
        local_prefix, child_prefix, tf.ones([3, 3], tf.float64)
    )
    tf.debugging.assert_near(prefix_metrics["child_score"], child_prefix, atol=2e-12)
    tf.debugging.assert_near(global_metrics["loss"], 0.0, atol=1e-20)
    tf.debugging.assert_near(prefix_metrics["loss"], 0.0, atol=1e-20)


def test_compiled_origin_total_score_step_is_finite_and_preserves_origin() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=32)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_compiled_total_score_test",
    )
    trainer = CenteredResidualTrainer(parent)
    child = trainer.freeze_child()
    global_target = tf.constant([-5.0, 2.0, -4.8], tf.float64)
    local_prefix = batch.local_points[0, :2, :18]
    _prefix_value, prefix_target = child.prefix_log_marginal_and_score(
        tf.zeros([3], tf.float64), local_prefix
    )
    prefix_target = prefix_target + tf.constant(
        [[0.1, -0.1, 0.05], [-0.1, 0.1, -0.05]], tf.float64
    )
    before = child.increment_and_score(tf.zeros([3], tf.float64))[0]
    step = make_compiled_origin_total_score_train_step(
        trainer,
        tf.keras.optimizers.Adam(learning_rate=1e-5),
        point_weight=1.0,
        global_weight=1.0,
        prefix_weight=1.0,
        l2_weight=1e-10,
        gradient_clip_norm=100.0,
    )
    terms = step(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
        global_target,
        tf.ones([3], tf.float64),
        local_prefix,
        prefix_target,
        tf.ones([2, 3], tf.float64),
    )
    for value in terms:
        tf.debugging.assert_all_finite(value, "compiled origin total-score term")
    tf.debugging.assert_positive(terms[-2])
    after = trainer.freeze_child().increment_and_score(tf.zeros([3], tf.float64))[0]
    tf.debugging.assert_near(before, parent.value(), atol=2e-13)
    tf.debugging.assert_near(after, parent.value(), atol=2e-13)
    heldout = trainer.heldout_metrics(batch)
    assert set(heldout) == {
        "child_log_mass",
        "target_log_mass",
        "target_log_mass_standard_error",
        "absolute_log_mass_error",
        "normalized_log_density_rms",
        "importance_effective_sample_size",
        "minimum_rho",
    }
    for value in heldout.values():
        tf.debugging.assert_all_finite(value, "core affine heldout metric")


def test_rotating_prefix_schedule_covers_each_epoch_exactly() -> None:
    first_epoch = tf.concat(
        [
            rotating_prefix_minibatch_indices(
                pool_size=32, batch_size=8, update=update, seed=8401
            )
            for update in range(4)
        ],
        axis=0,
    )
    replay = tf.concat(
        [
            rotating_prefix_minibatch_indices(
                pool_size=32, batch_size=8, update=update, seed=8401
            )
            for update in range(4)
        ],
        axis=0,
    )
    second_epoch = tf.concat(
        [
            rotating_prefix_minibatch_indices(
                pool_size=32, batch_size=8, update=update, seed=8401
            )
            for update in range(4, 8)
        ],
        axis=0,
    )
    tf.debugging.assert_equal(tf.sort(first_epoch), tf.range(32, dtype=tf.int32))
    tf.debugging.assert_equal(tf.sort(second_epoch), tf.range(32, dtype=tf.int32))
    tf.debugging.assert_equal(first_epoch, replay)
    assert not bool(tf.reduce_all(first_epoch == second_epoch).numpy())


def test_additive_residual_zero_embedding_preserves_finite_child_score() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=64)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_embedding_test",
    )
    fitted = target_informed_additive_score_initialization(
        parent=parent,
        local_points=batch.local_points[0],
        target_complete_data_score=batch.complete_data_score[0],
        importance_log_weight=batch.observation_log_density[0],
        ridge_fraction=1e-4,
        global_score_weight=10.0,
    )
    rank_two = CenteredResidualTrainer(
        parent, initial_residual_components=fitted.residual_components
    ).freeze_child()
    embedded_components = tuple(
        embed_residual_component_at_rank(component, target_rank=4)
        for component in fitted.residual_components
    )
    rank_four = CenteredResidualTrainer(
        parent, initial_residual_components=embedded_components
    ).freeze_child()
    theta = tf.constant([0.02, -0.01, 0.03], tf.float64)
    rank_four_log, rank_four_score = rank_four.point_log_density_and_score(
        theta, batch.local_points[0]
    )
    rank_two_log, rank_two_score = rank_two.point_log_density_and_score(
        theta, batch.local_points[0]
    )
    tf.debugging.assert_near(rank_four_log, rank_two_log, atol=2e-12)
    tf.debugging.assert_near(rank_four_score, rank_two_score, atol=2e-12)
    rank_four_value, rank_four_global_score = rank_four.increment_and_score(theta)
    rank_two_value, rank_two_global_score = rank_two.increment_and_score(theta)
    tf.debugging.assert_near(rank_four_value, rank_two_value, atol=2e-12)
    tf.debugging.assert_near(
        rank_four_global_score, rank_two_global_score, atol=2e-12
    )


def test_core_tangent_block_tt_matches_product_rule_point_global_and_prefix() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=16)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="core_tangent_block_test",
    )
    tangent_banks = tuple(
        tuple(
            tf.random.stateless_normal(
                core.shape,
                seed=[8423 + 101 * axis + parameter, 1],
                dtype=tf.float64,
            )
            * tf.constant(1e-4, tf.float64)
            for parameter in range(3)
        )
        for axis, core in enumerate(parent.cores)
    )
    components = tuple(
        core_tangent_to_residual_component(
            parent_cores=parent.cores,
            tangent_cores=tuple(bank[parameter] for bank in tangent_banks),
        )
        for parameter in range(3)
    )
    trainer = CenteredResidualTrainer(parent, initial_residual_components=components)
    structured = __import__(
        "bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf",
        fromlist=["LaneBParameterChild"],
    ).LaneBParameterChild(parent, tangent_banks)
    zero = tf.zeros([3], tf.float64)
    points = batch.local_points[0]
    _left_log, left_point = trainer.freeze_child().point_log_density_and_score(
        zero, points
    )
    _right_log, right_point = structured.point_log_density_and_score(zero, points)
    tf.debugging.assert_near(left_point, right_point, atol=3e-11, rtol=3e-11)
    left_value, left_global = trainer.freeze_child().increment_and_score(zero)
    right_value, right_global = structured.increment_and_score(zero)
    tf.debugging.assert_near(left_value, right_value, atol=2e-13)
    tf.debugging.assert_near(left_global, right_global, atol=3e-11, rtol=3e-11)
    prefix = points[:3, :18]
    _left_prefix_value, left_prefix = (
        trainer.freeze_child().prefix_log_marginal_and_score(zero, prefix)
    )
    _right_prefix_value, right_prefix = structured.prefix_log_marginal_and_score(
        zero, prefix
    )
    tf.debugging.assert_near(left_prefix, right_prefix, atol=3e-11, rtol=3e-11)
    assert max(int(core.shape[2]) for core in components[0][:-1]) == 8


def test_current_basis_core_affine_zero_slice_has_finite_score_gradients() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=32)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="core_affine_gradient_test",
    )
    trainer = CoreAffineTangentTrainer(parent)
    before = trainer.freeze_child().increment_and_score(tf.zeros([3], tf.float64))[0]
    prefix = batch.local_points[0, :3, :18]
    with tf.GradientTape() as tape:
        point = trainer.origin_point_score_metrics_arrays(
            batch.local_points[0],
            batch.complete_data_score[0],
            batch.observation_log_density[0],
        )
        global_metrics = trainer.origin_global_score_metrics_arrays(
            tf.constant([-5.0, 2.0, -4.9], tf.float64),
            tf.ones([3], tf.float64),
        )
        prefix_metrics = trainer.origin_prefix_score_metrics_arrays(
            prefix,
            tf.ones([3, 3], tf.float64),
            tf.ones([3, 3], tf.float64),
        )
        loss = point["loss"] + global_metrics["loss"] + prefix_metrics["loss"]
    gradients = tape.gradient(loss, trainer.trainable_variables)
    assert all(gradient is not None for gradient in gradients)
    tf.debugging.assert_all_finite(tf.linalg.global_norm(gradients), "core affine gradient")
    tf.debugging.assert_positive(tf.linalg.global_norm(gradients))
    after = trainer.freeze_child().increment_and_score(tf.zeros([3], tf.float64))[0]
    tf.debugging.assert_near(before, parent.value(), atol=2e-13)
    tf.debugging.assert_near(after, parent.value(), atol=2e-13)


def test_core_affine_functional_value_gradient_and_directional_parity() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="core_affine_functional_test",
    )
    trainer = CoreAffineTangentTrainer(parent)
    position = trainer.position()
    prefix = batch.local_points[0, :2, :18]
    global_target = tf.constant([-5.0, 2.0, -4.9], tf.float64)
    global_se = tf.ones([3], tf.float64)
    prefix_target = tf.ones([2, 3], tf.float64)
    prefix_se = tf.ones([2, 3], tf.float64)
    callback = make_compiled_core_affine_total_score_value_and_gradient(
        parent=parent,
        point_local_points=batch.local_points[0],
        point_target_score=batch.complete_data_score[0],
        point_importance_log_weight=batch.observation_log_density[0],
        global_target_score=global_target,
        global_score_standard_error=global_se,
        prefix_local_points=prefix,
        prefix_target_score=prefix_target,
        prefix_score_standard_error=prefix_se,
        point_weight=100.0,
        global_weight=100.0,
        prefix_weight=1.0,
        l2_weight=1e-10,
    )
    loss, gradient = callback(position)
    eager_loss = core_affine_origin_total_score_loss_arrays(
        parent=parent,
        position=position,
        point_local_points=batch.local_points[0],
        point_target_score=batch.complete_data_score[0],
        point_importance_log_weight=batch.observation_log_density[0],
        global_target_score=global_target,
        global_score_standard_error=global_se,
        prefix_local_points=prefix,
        prefix_target_score=prefix_target,
        prefix_score_standard_error=prefix_se,
        point_weight=100.0,
        global_weight=100.0,
        prefix_weight=1.0,
        l2_weight=1e-10,
    )[0]
    tf.debugging.assert_near(loss, eager_loss, atol=2e-11, rtol=2e-11)
    direction = tf.random.stateless_normal(tf.shape(position), [8441, 1], dtype=tf.float64)
    direction = direction / tf.linalg.norm(direction)
    epsilon = tf.constant(1e-5, tf.float64)
    plus = callback(position + epsilon * direction)[0]
    minus = callback(position - epsilon * direction)[0]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    directional = tf.tensordot(gradient, direction, axes=1)
    tf.debugging.assert_near(finite_difference, directional, atol=3e-6, rtol=3e-6)


def test_core_affine_gate_minimax_callback_is_finite_and_directionally_correct() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="core_affine_minimax_functional_test",
    )
    trainer = CoreAffineTangentTrainer(parent)
    position = trainer.position()
    prefix = batch.local_points[0, :2, :18]
    callback = make_compiled_core_affine_gate_minimax_value_and_gradient(
        parent=parent,
        point_local_points=batch.local_points[0],
        point_target_score=batch.complete_data_score[0],
        point_importance_log_weight=batch.observation_log_density[0],
        global_target_score=tf.constant([-5.0, 2.0, -4.9], tf.float64),
        global_score_standard_error=tf.ones([3], tf.float64),
        prefix_local_points=prefix,
        prefix_target_score=tf.ones([2, 3], tf.float64),
        prefix_score_standard_error=tf.ones([2, 3], tf.float64),
        temperature=64.0,
        l2_weight=1e-10,
    )
    loss, gradient = callback(position)
    tf.debugging.assert_all_finite(loss, "smooth-minimax test loss")
    tf.debugging.assert_all_finite(gradient, "smooth-minimax test gradient")
    direction = tf.random.stateless_normal(tf.shape(position), [8442, 1], dtype=tf.float64)
    direction = direction / tf.linalg.norm(direction)
    epsilon = tf.constant(1e-5, tf.float64)
    finite_difference = (
        callback(position + epsilon * direction)[0]
        - callback(position - epsilon * direction)[0]
    ) / (2.0 * epsilon)
    directional = tf.tensordot(gradient, direction, axes=1)
    tf.debugging.assert_near(finite_difference, directional, atol=3e-6, rtol=3e-6)


def test_full_tt_position_round_trip_and_minimax_directional_parity() -> None:
    parent = _parent()
    core_affine = CoreAffineTangentTrainer(parent)
    affine_position = core_affine.position()
    direction = tf.random.stateless_normal(
        tf.shape(affine_position), [8443, 1], dtype=tf.float64
    )
    core_affine.assign_position(1e-3 * direction)
    child = core_affine.freeze_child()
    position = residual_components_position(child.residual_components)
    restored = residual_components_from_position(
        template_components=child.residual_components, position=position
    )
    for actual_component, expected_component in zip(
        restored, child.residual_components
    ):
        for actual, expected in zip(actual_component, expected_component):
            tf.debugging.assert_equal(actual, expected)
    released_mask = core_affine_released_coordinate_mask(parent.cores)
    assert released_mask.shape == position.shape
    assert int(tf.reduce_sum(tf.cast(released_mask, tf.int32))) == 24600

    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="full_tt_minimax_functional_test",
    )
    prefix = batch.local_points[0, :2, :18]
    callback = make_compiled_full_tt_gate_minimax_value_and_gradient(
        parent=parent,
        template_components=child.residual_components,
        reference_position=position,
        point_local_points=batch.local_points[0],
        point_target_score=batch.complete_data_score[0],
        point_importance_log_weight=batch.observation_log_density[0],
        global_target_score=tf.constant([-5.0, 2.0, -4.9], tf.float64),
        global_score_standard_error=tf.ones([3], tf.float64),
        prefix_local_points=prefix,
        prefix_target_score=tf.ones([2, 3], tf.float64),
        prefix_score_standard_error=tf.ones([2, 3], tf.float64),
        temperature=64.0,
        l2_displacement_weight=1e-10,
    )
    loss, gradient = callback(position)
    tf.debugging.assert_all_finite(loss, "full-TT minimax loss")
    tf.debugging.assert_all_finite(gradient, "full-TT minimax gradient")
    released_gradient = tf.boolean_mask(gradient, released_mask)
    tf.debugging.assert_positive(tf.linalg.norm(released_gradient))
    full_direction = tf.random.stateless_normal(
        tf.shape(position), [8444, 1], dtype=tf.float64
    )
    full_direction = full_direction / tf.linalg.norm(full_direction)
    epsilon = tf.constant(1e-5, tf.float64)
    finite_difference = (
        callback(position + epsilon * full_direction)[0]
        - callback(position - epsilon * full_direction)[0]
    ) / (2.0 * epsilon)
    directional = tf.tensordot(gradient, full_direction, axes=1)
    tf.debugging.assert_near(finite_difference, directional, atol=3e-6, rtol=3e-6)


def test_rank12_connected_expansion_preserves_full_tt_function_at_origin() -> None:
    parent = _parent()
    source = CoreAffineTangentTrainer(parent)
    source.assign_position(
        1e-4
        * tf.random.stateless_normal(
            tf.shape(source.position()), [85701, 1], dtype=tf.float64
        )
    )
    source_child = source.freeze_child()
    expanded = tuple(
        embed_residual_component_at_rank(
            component,
            target_rank=12,
        )
        for component in source_child.residual_components
    )
    rank12 = CenteredResidualTrainer(
        parent, initial_residual_components=expanded
    )
    zero = tf.zeros([3], tf.float64)
    source_value, source_score = source_child.increment_and_score(zero)
    rank12_value, rank12_score = rank12.freeze_child().increment_and_score(zero)
    tf.debugging.assert_near(source_value, rank12_value, atol=2e-12, rtol=2e-12)
    tf.debugging.assert_near(source_score, rank12_score, atol=2e-12, rtol=2e-12)
    assert int(rank12.position().shape[0]) == 3 * (1 * 5 * 12 + 34 * 12 * 5 * 12 + 12 * 5 * 1)


def test_core_affine_product_rule_block_encoding_round_trips_exactly() -> None:
    parent = _parent()
    banks = tuple(
        tuple(
            tf.random.stateless_normal(
                core.shape, [8451 + axis, parameter], dtype=tf.float64
            )
            for parameter in range(3)
        )
        for axis, core in enumerate(parent.cores)
    )
    components = tuple(
        core_tangent_to_residual_component(
            parent_cores=parent.cores,
            tangent_cores=tuple(bank[parameter] for bank in banks),
        )
        for parameter in range(3)
    )
    restored = core_tangent_banks_from_residual_components(
        parent_cores=parent.cores, residual_components=components
    )
    for actual_bank, expected_bank in zip(restored, banks):
        for actual, expected in zip(actual_bank, expected_bank):
            tf.debugging.assert_equal(actual, expected)


def test_matrix_free_conjugate_gradient_solves_quadratic_callback() -> None:
    hessian = tf.constant(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]],
        tf.float64,
    )
    linear = tf.constant([-1.0, 2.0, -3.0], tf.float64)

    def callback(position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        gradient = tf.linalg.matvec(hessian, position) + linear
        value = 0.5 * tf.tensordot(position, tf.linalg.matvec(hessian, position), 1)
        value += tf.tensordot(linear, position, 1)
        return value, gradient

    result = solve_quadratic_value_gradient_with_conjugate_gradient(
        callback,
        initial_position=tf.constant([4.0, -2.0, 1.0], tf.float64),
        tolerance=1e-12,
        max_iterations=8,
        trace_interval=1,
    )
    expected = tf.linalg.solve(hessian, -linear[:, tf.newaxis])[:, 0]
    assert result.converged
    assert not result.failed
    assert result.num_iterations <= 3
    tf.debugging.assert_near(result.position, expected, atol=2e-12, rtol=2e-12)
    assert float(result.relative_residual_norm) <= 1e-12


def test_connected_rank_expansion_has_small_drift_and_added_channel_gradients() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=64)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_connected_embedding_test",
    )
    fitted = target_informed_additive_score_initialization(
        parent=parent,
        local_points=batch.local_points[0],
        target_complete_data_score=batch.complete_data_score[0],
        importance_log_weight=batch.observation_log_density[0],
        ridge_fraction=1e-4,
        global_score_weight=10.0,
    )
    rank_two = CenteredResidualTrainer(
        parent, initial_residual_components=fitted.residual_components
    )
    expanded = tuple(
        embed_residual_component_with_connected_channels(
            component,
            target_rank=4,
            seed=8399 + component_index * 101,
            seeded_channel_epsilon=1e-3,
        )
        for component_index, component in enumerate(fitted.residual_components)
    )
    rank_four = CenteredResidualTrainer(parent, initial_residual_components=expanded)
    two_metrics = rank_two.origin_point_score_metrics_arrays(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    four_metrics = rank_four.origin_point_score_metrics_arrays(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    tf.debugging.assert_less(
        tf.abs(four_metrics["loss"] - two_metrics["loss"]),
        tf.constant(1e-3, tf.float64),
    )
    with tf.GradientTape() as tape:
        loss = rank_four.origin_point_score_loss_arrays(
            batch.local_points[0],
            batch.complete_data_score[0],
            batch.observation_log_density[0],
        )
    gradients = tape.gradient(loss, rank_four.trainable_variables)
    for component_index in range(3):
        offset = component_index * 36
        tf.debugging.assert_positive(
            tf.reduce_sum(tf.abs(gradients[offset][0, :, 2:]))
        )
        tf.debugging.assert_positive(
            tf.reduce_sum(tf.abs(gradients[offset + 1][2:, :, 2:]))
        )
        tf.debugging.assert_positive(
            tf.reduce_sum(tf.abs(gradients[offset + 35][2:, :, 0]))
        )


def test_absolute_density_batch_requires_origin_as_first_theta_row() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=4)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.constant([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="bad_origin_order_test",
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        CenteredResidualTrainer(parent).absolute_density_loss(batch)


def test_absolute_loss_is_minimized_by_its_known_centered_target_family() -> None:
    parent = _parent()
    features = CenteredThetaFeatures()
    # At theta=(1,0,0), H_1=-h_parent cancels the amplitude exactly, leaving
    # rho=tau under the uniform reference probability measure. This supplies a
    # known continuous target and exact constant importance weight.
    cancelling = [tf.identity(core) for core in parent.cores]
    cancelling[0] = -cancelling[0]
    zero_components = []
    for _ in range(2):
        component = [tf.identity(core) for core in parent.cores]
        component[0] = tf.zeros_like(component[0])
        zero_components.append(tuple(component))
    target_components = (tuple(cancelling), *zero_components)
    theta = tf.constant([[1.0, 0.0, 0.0]], tf.float64)
    reference = tf.random.stateless_uniform(
        [1, 4096, 36],
        seed=[8311, 1],
        minval=tf.constant(-1.0, tf.float64),
        maxval=tf.constant(1.0, tf.float64),
        dtype=tf.float64,
    )
    local_points = reference * tf.math.rsqrt(1.0 - tf.square(reference))
    log_absolute_weight = tf.ones([1, 4096], tf.float64) * tf.math.log(
        tf.constant(parent.settings.tau, tf.float64)
    )
    target_trainer = CenteredResidualTrainer(
        parent,
        features=features,
        initial_residual_components=target_components,
    )
    initial_components = tuple(
        tuple(
            (
                tf.constant(1e-4 * (component_index + 1), tf.float64) * core
                if axis == 0
                else tf.identity(core)
            )
            for axis, core in enumerate(parent.cores)
        )
        for component_index in range(3)
    )
    initial_trainer = CenteredResidualTrainer(
        parent,
        features=features,
        initial_residual_components=initial_components,
    )
    target_loss = target_trainer.absolute_density_loss_arrays(
        theta, local_points, log_absolute_weight
    ).total_loss
    initial_loss = initial_trainer.absolute_density_loss_arrays(
        theta, local_points, log_absolute_weight
    ).total_loss
    assert float(target_loss) < float(initial_loss)


def test_optimizer_recovers_toward_known_centered_target_family() -> None:
    parent = _parent()
    features = CenteredThetaFeatures()
    cancelling = [tf.identity(core) for core in parent.cores]
    cancelling[0] = tf.constant(-0.8, tf.float64) * cancelling[0]
    zero_components = []
    for _ in range(2):
        component = [tf.identity(core) for core in parent.cores]
        component[0] = tf.zeros_like(component[0])
        zero_components.append(tuple(component))
    trainer = CenteredResidualTrainer(
        parent,
        features=features,
        initial_residual_components=(tuple(cancelling), *zero_components),
    )
    reference = tf.random.stateless_uniform(
        [1, 512, 36],
        seed=[8323, 1],
        minval=tf.constant(-1.0, tf.float64),
        maxval=tf.constant(1.0, tf.float64),
        dtype=tf.float64,
    )
    local_points = reference * tf.math.rsqrt(1.0 - tf.square(reference))
    theta = tf.constant([[1.0, 0.0, 0.0]], tf.float64)
    log_weight = tf.ones([1, 512], tf.float64) * tf.math.log(
        tf.constant(parent.settings.tau, tf.float64)
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    def loss():
        return trainer.absolute_density_loss_arrays(
            theta, local_points, log_weight
        ).total_loss

    initial_loss = loss()
    for _ in range(20):
        with tf.GradientTape() as tape:
            current_loss = loss()
        gradients = tape.gradient(current_loss, trainer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    final_loss = loss()
    tf.debugging.assert_less(final_loss, tf.constant(0.5, tf.float64) * initial_loss)


def test_real_target_train_step_has_finite_nonzero_gradient_and_preserves_origin() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=_theta(),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_smoke",
    )
    trainer = CenteredResidualTrainer(parent)
    before = trainer.freeze_child().increment_and_score(tf.zeros([3], tf.float64))[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    step = make_compiled_absolute_train_step(
        trainer,
        optimizer,
        l1_weight=0.0,
        l2_weight=0.0,
        gradient_clip_norm=10.0,
    )
    terms = step(
        batch.theta,
        batch.local_points,
        parent.shift_constant + batch.observation_log_density,
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    for value in terms:
        tf.debugging.assert_all_finite(value, "compiled training term")
    tf.debugging.assert_near(terms[0], terms[1], atol=2e-12)
    assert float(terms[-1]) > 0.0
    after = trainer.freeze_child().increment_and_score(tf.zeros([3], tf.float64))[0]
    tf.debugging.assert_near(before, parent.value(), atol=2e-13)
    tf.debugging.assert_near(after, parent.value(), atol=2e-13)


def test_derivative_matching_loss_uses_origin_complete_data_score() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=32)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=_theta(),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="derivative_loss_test",
    )
    trainer = CenteredResidualTrainer(parent)
    base = trainer.absolute_density_loss(batch, derivative_weight=0.0)
    weighted = trainer.absolute_density_loss(batch, derivative_weight=0.25)
    tf.debugging.assert_positive(weighted.derivative_matching_loss)
    tf.debugging.assert_near(
        weighted.total_loss - base.total_loss,
        tf.constant(0.25, tf.float64) * weighted.derivative_matching_loss,
        atol=2e-12,
    )
    metrics = trainer.origin_point_score_metrics_arrays(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    tf.debugging.assert_near(
        metrics["loss"], weighted.derivative_matching_loss, atol=2e-12
    )
    tf.debugging.assert_near(
        metrics["loss"],
        tf.reduce_sum(tf.square(metrics["normalized_score_residual_rms"])),
        atol=2e-12,
    )


def test_origin_score_prefit_reduces_exact_finite_child_score_loss() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=128)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="training_only_prefit_test",
    )
    features = CenteredThetaFeatures()
    trainer = CenteredResidualTrainer(
        parent,
        features=features,
        initial_residual_components=fixed_rank_initial_residual_components(
            parent=parent,
            features=features,
            rank=2,
            seed=8377,
            amplitude_scale=1.0,
            perturbation_scale=0.05,
        ),
    )
    before = trainer.origin_point_score_metrics_arrays(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    step = make_compiled_origin_score_prefit_step(
        trainer,
        tf.keras.optimizers.Adam(learning_rate=3e-4),
        gradient_clip_norm=100.0,
    )
    for _ in range(12):
        terms = step(
            batch.local_points[0],
            batch.complete_data_score[0],
            batch.observation_log_density[0],
        )
    after = trainer.origin_point_score_metrics_arrays(
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    for value in terms:
        tf.debugging.assert_all_finite(value, "origin score prefit term")
    tf.debugging.assert_less(after["loss"], before["loss"])
    tf.debugging.assert_near(
        trainer.freeze_child().increment_and_score(tf.zeros([3], tf.float64))[0],
        parent.value(),
        atol=2e-13,
    )


def test_heldout_metrics_report_absolute_mass_shape_and_ess() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=32)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=_theta(),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="heldout_test",
    )
    metrics = CenteredResidualTrainer(parent).heldout_metrics(batch)
    assert set(metrics) == {
        "child_log_mass",
        "target_log_mass",
        "target_log_mass_standard_error",
        "absolute_log_mass_error",
        "normalized_log_density_rms",
        "importance_effective_sample_size",
        "minimum_rho",
    }
    for value in metrics.values():
        tf.debugging.assert_all_finite(value, "heldout metric")
    tf.debugging.assert_non_negative(metrics["absolute_log_mass_error"])
    tf.debugging.assert_positive(metrics["target_log_mass_standard_error"])
    tf.debugging.assert_non_negative(metrics["normalized_log_density_rms"])
    tf.debugging.assert_greater_equal(
        metrics["importance_effective_sample_size"], tf.ones([3], tf.float64)
    )
    tf.debugging.assert_positive(metrics["minimum_rho"])


def test_compiled_nonzero_derivative_weight_uses_explicit_score_arrays() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=8)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=_theta(),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="compiled_derivative_test",
    )
    trainer = CenteredResidualTrainer(parent)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    step = make_compiled_absolute_train_step(
        trainer,
        optimizer,
        l1_weight=0.0,
        l2_weight=0.0,
        derivative_weight=0.25,
        gradient_clip_norm=10.0,
    )
    terms = step(
        batch.theta,
        batch.local_points,
        parent.shift_constant + batch.observation_log_density,
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    for value in terms:
        tf.debugging.assert_all_finite(value, "compiled derivative training term")
    tf.debugging.assert_positive(terms[2])
    tf.debugging.assert_near(
        terms[0] - terms[1],
        tf.constant(0.25, tf.float64) * terms[2],
        atol=2e-12,
    )


def test_independent_ratio_and_prefix_estimators_report_finite_uncertainty() -> None:
    parent = _parent()
    initial_noise, transition_noise = _noise(sample_count=256)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="authority_smoke",
    )
    global_estimate = estimate_t1_ratio_score(batch, theta_index=0)
    prefix = estimate_t1_prefix_scores(
        prefix_points=batch.physical_points[0, :1, :18],
        global_score=global_estimate,
        sample_count=2048,
        seed=8201,
    )[0]
    tf.debugging.assert_all_finite(global_estimate.score, "global score")
    tf.debugging.assert_all_finite(prefix.score, "prefix score")
    tf.debugging.assert_positive(global_estimate.score_standard_error)
    tf.debugging.assert_positive(prefix.score_standard_error)
    assert float(global_estimate.effective_sample_size) > 1.0
    assert float(prefix.effective_sample_size) > 0.5 * 2048


def test_batch_native_target_compiles_with_xla() -> None:
    initial_noise, transition_noise = _noise(sample_count=8)
    _states, observations, _all = generate_sealed_lane_b_dataset()

    @tf.function(jit_compile=True)
    def compiled(theta, initial, transition):
        result = batch_native_t1_from_common_noise(
            theta, initial, transition, observations[0]
        )
        return result["complete_log_density"], result["complete_data_score"]

    eager = batch_native_t1_from_common_noise(
        _theta(), initial_noise, transition_noise, observations[0]
    )
    graph = compiled(_theta(), initial_noise, transition_noise)
    tf.debugging.assert_near(graph[0], eager["complete_log_density"], atol=2e-12)
    tf.debugging.assert_near(graph[1], eager["complete_data_score"], atol=2e-12)

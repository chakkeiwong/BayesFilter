from __future__ import annotations

import pytest
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import (
    exact_transformed_sv_candidate_adapter,
)
from bayesfilter.highdim.ledh_contract_e_identity import (
    issue_moment_teacher_actual_sv_contract_e_route_identity,
    issue_moment_teacher_predator_prey_contract_e_route_identity,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import (
    MomentTeacherControls,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_nonlinear_tf import (
    ACTUAL_SV_ROUTE_ID,
    EVENT_ORDER,
    freeze_nonlinear_teacher_scale_shift_indices,
    issue_nonlinear_moment_teacher_tuning_artifact,
    make_nonlinear_moment_teacher_value_and_score_tf,
    make_nonlinear_tuning_scope,
    prepare_nonlinear_teacher_inputs,
    route_identity_prepared_inputs,
)
from bayesfilter.testing.zhao_cui_actual_sv_target_tf import (
    actual_sv_unconstrained_theta_tf,
    generate_source_order_actual_sv_dataset_tf,
)


THETA = tf.cast(actual_sv_unconstrained_theta_tf(), tf.float32)


def _controls(correction: bool) -> MomentTeacherControls:
    return MomentTeacherControls(
        sinkhorn_steps=2,
        balance_steps=20,
        correction_steps=int(correction),
        correction_strength=0.01 if correction else 0.0,
        correction_floor=1.0e-6,
        pairwise_correction_steps=0,
        pairwise_strength=0.0,
        pairwise_floor=1.0e-6,
        tt_ridge=1.0e-4,
        column_scale_floor=1.0e-6,
        condition_number_veto=1.0e8,
        fit_residual_veto=2.0,
    )


def _fixture(correction: bool):
    horizon = 2
    particle_count = 8
    adapter = exact_transformed_sv_candidate_adapter(sigma=1.0)
    _, _, observations64 = generate_source_order_actual_sv_dataset_tf(horizon=horizon)
    observations = tf.cast(observations64, tf.float32)
    particle_prepared = {
        "observations": observations,
        "initial_noise": tf.random.stateless_normal([particle_count, 1], [9301, 1]),
        "process_noise": tf.random.stateless_normal(
            [horizon, particle_count, 1], [9301, 2]
        ),
        "residual_design": tf.random.stateless_normal(
            [horizon, particle_count, 1], [9301, 3]
        ),
        "prepared_ridge": tf.fill([horizon], tf.constant(1.0e-5, tf.float32)),
        "epsilon": tf.constant(0.5, tf.float32),
        "scaling": tf.constant(0.9, tf.float32),
    }
    teacher = prepare_nonlinear_teacher_inputs(
        adapter=adapter,
        observations=observations,
        state_offset=tf.constant([0.0], tf.float32),
        state_scale=tf.constant([5.0], tf.float32),
        center_theta=THETA,
        initial_standard_deviation=1.25,
        process_standard_deviation=1.0,
        fit_rows=48,
        basis_size=2,
        rank=1,
        sweeps=1,
        defensive_weight=0.0,
        pair_indices=tf.zeros([0, 2], tf.int32),
        root_seed=9321,
    )
    teacher = freeze_nonlinear_teacher_scale_shift_indices(
        teacher,
        _controls(correction),
        adapter,
        initial_variance=1.0,
        process_variance=1.0,
    )
    scope = make_nonlinear_tuning_scope(
        model_id="exact_transformed_sv_fixture",
        target_id="source_order_exact_sv_t2_fixture",
        route_id=ACTUAL_SV_ROUTE_ID,
        horizon=horizon,
        prepared_data_id="exact_sv_t2_fixture",
        particle_count=particle_count,
        state_dimension=1,
        parameter_count=2,
        dtype=tf.float32,
        tf32_enabled=False,
        jit_compile=False,
    )
    artifact = issue_nonlinear_moment_teacher_tuning_artifact(
        scope=scope,
        controls=_controls(correction),
        calibration_data_id="calibration_seed_1",
        validation_data_id="validation_seed_2",
        selection_record_id="actual_sv_fixture_selection",
        chart_id="offset_0_scale_5",
        pair_set_id="scalar_empty_pairs",
    )
    return adapter, particle_prepared, teacher, scope, artifact


def _compiled(correction: bool, *, teacher: bool = True):
    adapter, particle_prepared, teacher_prepared, scope, artifact = _fixture(correction)
    return make_nonlinear_moment_teacher_value_and_score_tf(
        adapter=adapter,
        particle_prepared=particle_prepared,
        teacher_prepared=teacher_prepared if teacher else None,
        tuning_artifact=artifact,
        expected_scope=scope,
        initial_variance=1.0,
        process_variance=1.0,
        jit_compile=False,
    )


def test_source_order_dataset_observes_first_transition() -> None:
    states, raw, transformed = generate_source_order_actual_sv_dataset_tf(horizon=2)
    assert EVENT_ORDER == "x0_then_transition_then_observe"
    assert states.shape == raw.shape == transformed.shape == (2, 1)
    tf.debugging.assert_near(transformed, tf.math.log(tf.square(raw)), atol=0.0)


def test_stationary_initial_density_tangent_matches_autodiff() -> None:
    adapter = exact_transformed_sv_candidate_adapter(sigma=1.0)
    points = tf.constant([[-1.5], [0.0], [2.0]], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(THETA)
        total = tf.reduce_sum(adapter.initial_log_density(THETA, points))
    autodiff = tape.gradient(total, THETA)
    analytic = tf.reduce_sum(
        adapter.initial_log_density_tangent(THETA, points), axis=0
    )
    tf.debugging.assert_near(analytic, autodiff, atol=1.0e-6, rtol=1.0e-6)


def test_zero_correction_ties_out_to_empirical_contract_e() -> None:
    candidate = _compiled(False, teacher=True)(THETA)
    baseline = _compiled(False, teacher=False)(THETA)
    assert bool(candidate["valid_chart"].numpy())
    assert bool(candidate["teacher_valid"].numpy())
    tf.debugging.assert_near(candidate["objective"], baseline["objective"], atol=0.0)
    tf.debugging.assert_near(candidate["score"], baseline["score"], atol=0.0)


def test_total_score_matches_same_program_centered_difference() -> None:
    evaluate = _compiled(True)
    result = evaluate(THETA)
    assert bool(result["valid_chart"].numpy())
    step = tf.constant(2.0e-3, tf.float32)
    finite_difference = []
    for index in range(2):
        direction = tf.one_hot(index, 2, dtype=tf.float32)
        finite_difference.append(
            (evaluate(THETA + step * direction)["objective"]
             - evaluate(THETA - step * direction)["objective"])
            / (2.0 * step)
        )
    tf.debugging.assert_near(
        result["score"], tf.stack(finite_difference), atol=1.0e-2, rtol=1.0e-2
    )


def test_factory_binds_actual_sv_and_rejects_cross_model_substitution() -> None:
    _, particle_prepared, teacher, _, artifact = _fixture(False)
    prepared = route_identity_prepared_inputs(particle_prepared, teacher, artifact)
    identity = issue_moment_teacher_actual_sv_contract_e_route_identity(
        prepared_inputs=prepared
    ).to_dict()
    assert identity["route_specification_id"].endswith("actual_sv_v1")
    assert identity["parameter_names"] == ["z_gamma", "log_beta"]
    with pytest.raises(ValueError, match="shape"):
        issue_moment_teacher_predator_prey_contract_e_route_identity(
            prepared_inputs=prepared
        )


def test_scalar_trace_does_not_relax_later_multivariate_static_shapes() -> None:
    scalar = _compiled(False)(THETA)
    assert bool(scalar["valid_chart"].numpy())

    from tests.highdim.test_zhao_cui_moment_teacher_nonlinear import (
        THETA as PREDATOR_PREY_THETA,
        _compiled as predator_prey_compiled,
    )

    multivariate = predator_prey_compiled(False)(PREDATOR_PREY_THETA)
    assert bool(multivariate["valid_chart"].numpy())

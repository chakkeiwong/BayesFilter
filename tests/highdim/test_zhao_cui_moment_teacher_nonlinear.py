from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import predator_prey_candidate_adapter
from bayesfilter.highdim.ledh_contract_e_identity import (
    issue_moment_teacher_austria_sir_contract_e_route_identity,
    issue_moment_teacher_predator_prey_contract_e_route_identity,
)
import pytest
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import MomentTeacherControls
from bayesfilter.highdim.zhao_cui_moment_teacher_nonlinear_tf import (
    EVENT_ORDER,
    PREDATOR_PREY_ROUTE_ID,
    freeze_nonlinear_teacher_scale_shift_indices,
    issue_nonlinear_moment_teacher_tuning_artifact,
    latent_preclip_austria_sir_candidate_adapter,
    make_nonlinear_moment_teacher_value_and_score_tf,
    make_nonlinear_tuning_scope,
    prepare_nonlinear_teacher_inputs,
    route_identity_prepared_inputs,
)


THETA = tf.constant([0.6, 114.0, 25.0, 0.3, 0.5, 0.5], tf.float32)


def _controls(correction: bool) -> MomentTeacherControls:
    return MomentTeacherControls(
        sinkhorn_steps=2,
        balance_steps=20,
        correction_steps=1 if correction else 0,
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
    particle_count = 8
    horizon = 2
    adapter = predator_prey_candidate_adapter()
    observations = tf.constant([[52.0, 4.5], [55.0, 5.0]], tf.float32)
    particle_prepared = {
        "observations": observations,
        "initial_noise": tf.random.stateless_normal([particle_count, 2], [9101, 1]),
        "process_noise": tf.random.stateless_normal(
            [horizon, particle_count, 2], [9101, 2]
        ),
        "residual_design": tf.random.stateless_normal(
            [horizon, particle_count, 2], [9101, 3]
        ),
        "prepared_ridge": tf.fill([horizon], tf.constant(1.0e-5, tf.float32)),
        "epsilon": tf.constant(0.5, tf.float32),
        "scaling": tf.constant(0.9, tf.float32),
    }
    teacher = prepare_nonlinear_teacher_inputs(
        adapter=adapter,
        observations=observations,
        state_offset=tf.constant([60.0, 5.0]),
        state_scale=tf.constant([30.0, 12.0]),
        center_theta=THETA,
        initial_standard_deviation=1.0,
        process_standard_deviation=2.0,
        fit_rows=48,
        basis_size=2,
        rank=1,
        sweeps=1,
        defensive_weight=0.0,
        pair_indices=tf.constant([[0, 1], [1, 0]], tf.int32),
        root_seed=9121,
    )
    teacher = freeze_nonlinear_teacher_scale_shift_indices(
        teacher,
        _controls(correction),
        adapter,
        initial_variance=1.0,
        process_variance=4.0,
    )
    scope = make_nonlinear_tuning_scope(
        model_id="predator_prey_fixture",
        target_id="source_order_t2_fixture",
        route_id=PREDATOR_PREY_ROUTE_ID,
        horizon=horizon,
        prepared_data_id="predator_prey_t2_fixture",
        particle_count=particle_count,
        state_dimension=2,
        parameter_count=6,
        dtype=tf.float32,
        tf32_enabled=False,
        jit_compile=False,
    )
    artifact = issue_nonlinear_moment_teacher_tuning_artifact(
        scope=scope,
        controls=_controls(correction),
        calibration_data_id="calibration_seed_1",
        validation_data_id="validation_seed_2",
        selection_record_id="fixture_selection",
        chart_id="offset_60_5_scale_30_12",
        pair_set_id="ordered_01_10",
    )
    return adapter, particle_prepared, teacher, scope, artifact


def _compiled(correction: bool, teacher: bool = True, jit_compile: bool = False):
    adapter, particle_prepared, teacher_prepared, scope, artifact = _fixture(correction)
    return make_nonlinear_moment_teacher_value_and_score_tf(
        adapter=adapter,
        particle_prepared=particle_prepared,
        teacher_prepared=teacher_prepared if teacher else None,
        tuning_artifact=artifact,
        expected_scope=scope,
        initial_variance=1.0,
        process_variance=4.0,
        jit_compile=jit_compile,
    )


def test_source_order_and_zero_correction_tie_out() -> None:
    assert EVENT_ORDER == "x0_then_transition_then_observe"
    candidate = _compiled(False, teacher=True)(THETA)
    baseline = _compiled(False, teacher=False)(THETA)
    assert bool(candidate["valid_chart"].numpy())
    assert bool(candidate["teacher_valid"].numpy())
    tf.debugging.assert_near(candidate["objective"], baseline["objective"], atol=0.0)
    tf.debugging.assert_near(candidate["score"], baseline["score"], atol=0.0)


def test_nonzero_score_matches_same_program_finite_difference() -> None:
    evaluate = _compiled(True)
    result = evaluate(THETA)
    assert bool(result["valid_chart"].numpy())
    # The objective is O(1e2), so smaller FP32 centered steps are subtraction-
    # roundoff dominated.  A step ladder selected 1e-2 for this diagnostic.
    h = tf.constant(1.0e-2, tf.float32)
    finite_difference = []
    for index in range(6):
        direction = tf.one_hot(index, 6, dtype=tf.float32)
        finite_difference.append(
            (evaluate(THETA + h * direction)["objective"]
             - evaluate(THETA - h * direction)["objective"])
            / (2.0 * h)
        )
    tf.debugging.assert_near(
        result["score"], tf.stack(finite_difference), atol=2.0e-2, rtol=2.0e-2
    )


def test_graph_has_control_flow_and_no_host_callbacks() -> None:
    graph = _compiled(True, jit_compile=True).get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    for function in graph.library.function:
        operations.update(node.op for node in function.node_def)
    assert operations.isdisjoint({"PyFunc", "EagerPyFunc"})
    assert operations.intersection({"While", "StatelessWhile"})


def test_scope_uses_exact_divisor_chunks() -> None:
    _, _, _, scope, _ = _fixture(False)
    chunks = select_transport_chunks(8)
    assert scope.row_chunk_size == chunks.row_chunk_size == 8
    assert scope.col_chunk_size == chunks.col_chunk_size == 8


def test_repository_factory_binds_nonlinear_model_and_prepared_program() -> None:
    _, particle_prepared, teacher, _, artifact = _fixture(False)
    identity = issue_moment_teacher_predator_prey_contract_e_route_identity(
        prepared_inputs=route_identity_prepared_inputs(
            particle_prepared, teacher, artifact
        )
    ).to_dict()
    assert identity["route_specification_id"].endswith("predator_prey_v1")
    assert identity["parameter_names"] == ["r", "K", "a", "s", "u", "v"]
    assert identity["prepared_input_sha256"]
    assert identity["source_dependency_closure_sha256"]


def test_repository_factory_rejects_cross_model_prepared_shape_substitution() -> None:
    _, particle_prepared, teacher, _, artifact = _fixture(False)
    with pytest.raises(ValueError, match="shape"):
        issue_moment_teacher_austria_sir_contract_e_route_identity(
            prepared_inputs=route_identity_prepared_inputs(
                particle_prepared, teacher, artifact
            )
        )


def test_austria_adapter_applies_source_order_latent_preclip_tangent() -> None:
    adapter = latent_preclip_austria_sir_candidate_adapter()
    theta = tf.zeros([3], tf.float32)
    particles = tf.zeros([2, 18], tf.float32)
    particles = tf.tensor_scatter_nd_update(
        particles, [[0, 0], [1, 2]], [-3.0, -2.0]
    )
    tangent = tf.ones([2, 18, 3], tf.float32)
    noise = tf.zeros([2, 18], tf.float32)
    first = adapter.transition_tangent(
        theta, particles, noise, tangent, tf.constant(0, tf.int32)
    )
    later = adapter.transition_tangent(
        theta, particles, noise, tangent, tf.constant(1, tf.int32)
    )
    assert bool(tf.reduce_all(tf.math.is_finite(first)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(later)).numpy())
    assert float(tf.reduce_max(tf.abs(first - later)).numpy()) > 0.0

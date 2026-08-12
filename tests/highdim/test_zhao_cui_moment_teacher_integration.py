from __future__ import annotations

import tensorflow as tf
import pytest

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.ledh_contract_e_identity import (
    CONTRACT_E_DERIVATIVE_COMPOSITION_ID,
    CONTRACT_E_RESET_CONTRACT_ID,
    issue_moment_teacher_lgssm_contract_e_route_identity,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import (
    MomentTeacherControls,
    freeze_teacher_scale_shift_indices,
    issue_moment_teacher_tuning_artifact,
    make_lgssm_tuning_scope,
    make_moment_teacher_lgssm_value_and_score_tf,
    prepare_lgssm_teacher_inputs,
    route_identity_prepared_inputs,
)


DTYPE = tf.float64
THETA = tf.constant([0.55, 0.45, 0.35, 0.8, 0.6], DTYPE)


def _controls(*, correction: bool) -> MomentTeacherControls:
    return MomentTeacherControls(
        sinkhorn_steps=2,
        balance_steps=100,
        correction_steps=1 if correction else 0,
        correction_strength=0.025 if correction else 0.0,
        correction_floor=1.0e-6,
        pairwise_correction_steps=0,
        pairwise_strength=0.0,
        pairwise_floor=1.0e-6,
        tt_ridge=1.0e-5,
        column_scale_floor=1.0e-6,
        condition_number_veto=1.0e10,
        fit_residual_veto=2.0,
    )


def _prepared(*, correction: bool):
    time_steps = 2
    particles = 8
    observations = tf.constant(
        [[0.15, -0.1, 0.05], [0.2, 0.0, -0.12]], DTYPE
    )
    chunks = select_transport_chunks(particles)
    particle_result = preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=(91731,),
        num_particles=particles,
        fixed_reset_mask=tf.ones([1, time_steps], tf.bool),
        prepared_ridge=tf.fill([1, time_steps], tf.constant(1.0e-6, DTYPE)),
        epsilon=tf.constant(0.5, DTYPE),
        scaling=tf.constant(0.9, DTYPE),
        sinkhorn_steps=2,
        balance_steps=100,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=DTYPE,
    )
    teacher = prepare_lgssm_teacher_inputs(
        observations=observations,
        time_steps=time_steps,
        fit_rows=48,
        basis_size=2,
        rank=1,
        sweeps=1,
        chart_scale=2.5,
        defensive_weight=0.05,
        root_seed=91751,
        dtype=DTYPE,
        center_theta=THETA,
    )
    teacher = freeze_teacher_scale_shift_indices(
        teacher, _controls(correction=correction)
    )
    scope = make_lgssm_tuning_scope(
        horizon=time_steps,
        prepared_data_id="lgssm_t2_integration_fixture_v1",
        particle_count=particles,
        dtype=DTYPE,
        tf32_enabled=False,
        jit_compile=False,
    )
    artifact = issue_moment_teacher_tuning_artifact(
        scope=scope,
        controls=_controls(correction=correction),
        calibration_data_id="lgssm_t2_calibration_fixture_v1",
        validation_data_id="lgssm_t2_validation_fixture_v1",
        selection_record_id="mechanics_selection_only_v1",
    )
    return particle_result["prepared"], teacher, scope, artifact


def test_tuning_artifact_rejects_scope_mismatch() -> None:
    particles, teacher, scope, artifact = _prepared(correction=False)
    mismatched = make_lgssm_tuning_scope(
        horizon=10,
        prepared_data_id="lgssm_t10_other_scope",
        particle_count=8,
        dtype=DTYPE,
        tf32_enabled=False,
        jit_compile=False,
    )
    with pytest.raises(ValueError, match="does not match"):
        make_moment_teacher_lgssm_value_and_score_tf(
            particles,
            teacher,
            artifact,
            expected_scope=mismatched,
            jit_compile=False,
        )
    assert artifact.scope == scope


def test_factory_identity_binds_teacher_particle_controls_and_source() -> None:
    particles, teacher, scope, artifact = _prepared(correction=True)
    identity = issue_moment_teacher_lgssm_contract_e_route_identity(
        prepared_inputs=route_identity_prepared_inputs(
            particles, teacher, artifact
        )
    )
    payload = identity.to_dict()
    assert payload["reset_contract_id"] == CONTRACT_E_RESET_CONTRACT_ID
    assert payload["derivative_composition_id"] == CONTRACT_E_DERIVATIVE_COMPOSITION_ID
    assert payload["route_specification_id"].endswith("moment_teacher_lgssm_v1")
    names = {row["name"] for row in payload["prepared_input_records"]}
    assert {"initial_noise", "basis_values", "tt_ridge", "row_chunk_size"} <= names
    assert payload["source_dependency_closure_sha256"]
    assert len(payload["dependency_records"]) > 30


def test_zero_correction_ties_out_canonical_value_and_score() -> None:
    particles, teacher, scope, artifact = _prepared(correction=False)
    candidate = make_moment_teacher_lgssm_value_and_score_tf(
        particles,
        teacher,
        artifact,
        expected_scope=scope,
        jit_compile=False,
    )(THETA)
    baseline = canonical.make_canonical_value_and_score_tf(
        particles,
        steps=artifact.controls.sinkhorn_steps,
        balance_steps=artifact.controls.balance_steps,
        row_chunk_size=scope.row_chunk_size,
        col_chunk_size=scope.col_chunk_size,
        jit_compile=False,
        dtype=DTYPE,
    )(THETA)
    assert bool(candidate["teacher_valid"].numpy())
    assert bool(tf.reduce_all(candidate["valid_chart"]).numpy())
    tf.debugging.assert_near(candidate["objective"], baseline["objective"], atol=0.0)
    tf.debugging.assert_near(candidate["score"], baseline["score"], atol=0.0)


def test_nonzero_correction_score_matches_same_program_finite_difference() -> None:
    particles, teacher, scope, artifact = _prepared(correction=True)
    candidate = make_moment_teacher_lgssm_value_and_score_tf(
        particles,
        teacher,
        artifact,
        expected_scope=scope,
        jit_compile=False,
    )
    result = candidate(THETA)
    assert bool(result["teacher_valid"].numpy())
    assert bool(tf.reduce_all(result["valid_chart"]).numpy())
    tf.debugging.assert_less_equal(
        tf.reduce_max(result["mean_residual_history"]), tf.constant(2.0e-12, DTYPE)
    )
    tf.debugging.assert_less_equal(
        tf.reduce_max(result["covariance_residual_history"]),
        tf.constant(2.0e-11, DTYPE),
    )
    h = tf.constant(2.0e-5, DTYPE)
    columns = []
    for index in range(canonical.PARAMETER_COUNT):
        direction = tf.one_hot(index, canonical.PARAMETER_COUNT, dtype=DTYPE)
        plus = candidate(THETA + h * direction)["objective"]
        minus = candidate(THETA - h * direction)["objective"]
        columns.append((plus - minus) / (2.0 * h))
    finite_difference = tf.stack(columns)
    tf.debugging.assert_near(
        result["score"], finite_difference, atol=2.0e-5, rtol=2.0e-4
    )


def test_compiled_graph_uses_tensorflow_control_flow_without_host_callbacks() -> None:
    particles, teacher, scope, artifact = _prepared(correction=True)
    compiled = make_moment_teacher_lgssm_value_and_score_tf(
        particles,
        teacher,
        artifact,
        expected_scope=scope,
        jit_compile=True,
    )
    graph = compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    for function in graph.library.function:
        operations.update(node.op for node in function.node_def)
    assert operations.isdisjoint({"PyFunc", "EagerPyFunc"})
    assert operations.intersection({"While", "StatelessWhile"})

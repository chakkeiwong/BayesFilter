from __future__ import annotations

import json
import inspect
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as model
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
    "phase3_lgssm_order3_center_preparation_20260715/charts.json"
)
LOOKAHEAD_PREPARATIONS = {
    2: ROOT
    / (
        "docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/"
        "t2_attempt1/lgssm_t2_order5_lookahead8_preparation.json"
    ),
    3: ROOT
    / (
        "docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/"
        "t3_attempt1/lgssm_t3_order5_lookahead8_preparation.json"
    ),
    10: ROOT
    / (
        "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
        "phase8b_lgssm_t10_order5_lookahead8_attempt1_20260715/charts.json"
    ),
}
DTYPE = tf.float64
THETA = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)


def _observations(time_steps: int) -> tf.Tensor:
    return tf.convert_to_tensor(_lgssm_dataset(81100)["observations"][:time_steps], DTYPE)


def _rule(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = hermgauss(order)
    return tf.constant(np.sqrt(2.0) * nodes, DTYPE), tf.constant(weights / np.sqrt(np.pi), DTYPE)


def _prepared(time_steps: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    payload = json.loads(PREPARATION.read_text(encoding="utf-8"))
    nodes = tf.constant(payload["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(payload["quadrature"]["weights"], DTYPE)
    indices = tf.constant(payload["active_indices"][: time_steps - 1], tf.int32)
    scales = tf.constant(payload["row_scales"][: time_steps - 1], DTYPE)
    return nodes, weights, indices, scales


def _uncompressed_two_step(theta: tf.Tensor) -> tf.Tensor:
    observations = _observations(2)
    nodes, weights = _rule(3)
    parents, parent_log_weights, innovations, innovation_log_weights = model.initial_parents(
        theta, nodes, weights
    )
    first = model._teacher_step(
        theta,
        parents,
        parent_log_weights,
        innovations,
        innovation_log_weights,
        observations[0],
        observations[1],
    )
    second = model._teacher_step(
        theta,
        first["particles"],
        first["log_unnormalized_weights"] - first["increment"],
        innovations,
        innovation_log_weights,
        observations[1],
        None,
    )
    return first["increment"] + second["increment"]


def test_lgssm_target_and_feature_contract_are_explicit() -> None:
    payload = json.loads(PREPARATION.read_text(encoding="utf-8"))
    assert payload["algorithm_id"] == "contract_e_tp_experimental_v1"
    assert payload["route"] == "corrected_ledh_parent_by_innovation_teacher"
    assert payload["scope"] == "center_only_not_parameter_region_certificate"
    assert payload["feature_names"] == list(model.FEATURE_NAMES)
    assert payload["selection"]["runtime_active_set_selection"] is False
    assert payload["selection"]["clipping"] is False
    assert payload["summary"]["chart_count"] == 49
    assert payload["summary"]["minimum_weight"] > 0.0
    assert payload["target"]["theta"] == THETA.numpy().tolist()


def test_t1_corrected_ledh_quadrature_value_and_score_converge_to_kalman() -> None:
    observations = _observations(1)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        oracle = model.exact_kalman_value(THETA, observations)
        finite_values = []
        for order in (3, 5, 7):
            nodes, weights = _rule(order)
            finite_values.append(
                model.contract_e_tp_lgssm_recursive_core(
                    THETA,
                    observations,
                    nodes,
                    weights,
                    tf.zeros([0, model.FEATURE_COUNT], tf.int32),
                    tf.zeros([0, model.FEATURE_COUNT], DTYPE),
                )["objective"]
            )
    oracle_score = tape.gradient(oracle, THETA)
    value_errors = [float(tf.abs(value - oracle).numpy()) for value in finite_values]
    score_errors = [
        float(tf.linalg.norm(tape.gradient(value, THETA) - oracle_score).numpy())
        for value in finite_values
    ]
    assert value_errors[2] < value_errors[1] < value_errors[0]
    assert score_errors[2] < score_errors[1] < score_errors[0]
    assert value_errors[2] < 1.0e-5
    assert score_errors[2] < 4.0e-4


def test_t2_projection_preserves_same_teacher_value_and_total_score() -> None:
    nodes, weights, indices, scales = _prepared(2)
    observations = _observations(2)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        projected = model.contract_e_tp_lgssm_recursive_core(
            THETA, observations, nodes, weights, indices, scales
        )
        uncompressed = _uncompressed_two_step(THETA)
    projected_score = tape.gradient(projected["objective"], THETA)
    uncompressed_score = tape.gradient(uncompressed, THETA)
    np.testing.assert_allclose(projected["objective"], uncompressed, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(projected_score, uncompressed_score, rtol=2e-13, atol=3e-14)
    np.testing.assert_allclose(
        projected["target_history"], projected["matched_target_history"], rtol=0.0, atol=3e-15
    )
    assert bool(tf.reduce_all(projected["valid_history"]).numpy())
    incoming = projected["incoming_weight_history"]
    np.testing.assert_allclose(tf.reduce_sum(incoming[1]), 1.0, rtol=0.0, atol=3e-15)
    assert not np.allclose(incoming[1].numpy(), np.full(11, 1.0 / 11.0), rtol=1e-12, atol=1e-12)


def test_t5_center_chart_and_same_scalar_fd_pass() -> None:
    nodes, weights, indices, scales = _prepared(5)
    observations = _observations(5)
    with tf.GradientTape() as tape:
        tape.watch(THETA)
        result = model.contract_e_tp_lgssm_recursive_core(
            THETA, observations, nodes, weights, indices, scales
        )
    score = tape.gradient(result["objective"], THETA)
    assert bool(tf.reduce_all(result["valid_history"]).numpy())
    assert float(tf.reduce_min(result["minimum_weight_history"]).numpy()) > 0.0
    assert float(tf.reduce_max(tf.abs(result["feature_residual_history"])).numpy()) < 1.0e-14

    step = 1.0e-5
    finite_difference = []
    for index in range(5):
        direction = np.zeros(5)
        direction[index] = step
        plus = model.contract_e_tp_lgssm_recursive_core(
            tf.constant(THETA.numpy() + direction, DTYPE),
            observations,
            nodes,
            weights,
            indices,
            scales,
        )["objective"]
        minus = model.contract_e_tp_lgssm_recursive_core(
            tf.constant(THETA.numpy() - direction, DTYPE),
            observations,
            nodes,
            weights,
            indices,
            scales,
        )["objective"]
        finite_difference.append(float((plus - minus).numpy() / (2.0 * step)))
    relative = np.abs(score.numpy() - finite_difference) / np.maximum(
        np.maximum(np.abs(score.numpy()), np.abs(finite_difference)), 1.0e-12
    )
    assert float(np.max(relative)) <= 0.05 * np.sqrt(5.0)


def test_score_informed_factory_wraps_finite_lookahead_core() -> None:
    preparation_path = ROOT / (
        "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
        "phase8_continuation_information_v2_lgssm_t2_order5_attempt1_20260715/"
        "charts.json"
    )
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    observations = _observations(2)
    nodes = tf.constant(preparation["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["quadrature"]["weights"], DTYPE)
    active_indices = tf.constant(preparation["active_indices"], tf.int32)
    row_scales = tf.constant(preparation["row_scales"], DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(THETA)
        direct = model.contract_e_tp_lgssm_score_informed_recursive_core(
            THETA,
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            feature_mode="finite_lookahead",
            lookahead_steps=8,
        )
    direct_score = tape.gradient(direct["objective"], THETA)
    evaluate = model.make_contract_e_tp_lgssm_score_informed_recursive_tf(
        observations,
        nodes,
        weights,
        active_indices,
        row_scales,
        feature_mode="finite_lookahead",
        lookahead_steps=8,
        jit_compile=False,
    )
    traced = evaluate(THETA)

    epsilon = np.finfo(np.float64).eps
    objective_scale = max(
        abs(float(traced["objective"].numpy())),
        abs(float(direct["objective"].numpy())),
        1.0,
    )
    score_scale = max(
        float(tf.reduce_max(tf.abs(traced["score"])).numpy()),
        float(tf.reduce_max(tf.abs(direct_score)).numpy()),
        1.0,
    )
    tf.debugging.assert_near(
        traced["objective"], direct["objective"], atol=16.0 * epsilon * objective_scale, rtol=0.0
    )
    tf.debugging.assert_near(
        traced["score"], direct_score, atol=64.0 * epsilon * score_scale, rtol=0.0
    )
    tf.debugging.assert_equal(traced["valid_history"], direct["valid_history"])
    tf.debugging.assert_near(
        traced["feature_residual_history"],
        direct["feature_residual_history"],
        atol=64.0 * epsilon,
        rtol=0.0,
    )


def test_t2_wrong_or_duplicate_chart_fails_closed() -> None:
    nodes, weights, indices, scales = _prepared(2)
    observations = _observations(2)
    broken = tf.tensor_scatter_nd_update(indices, [[0, 1]], [indices[0, 0]])
    with pytest.raises(tf.errors.InvalidArgumentError):
        model.contract_e_tp_lgssm_recursive_core(
            THETA, observations, nodes, weights, broken, scales
        )


@pytest.mark.parametrize("time_steps", [2, 3, 10])
def test_finite_lookahead_loop_matches_unrolled_finite_program(time_steps: int) -> None:
    preparation = json.loads(
        LOOKAHEAD_PREPARATIONS[time_steps].read_text(encoding="utf-8")
    )
    observations = _observations(time_steps)
    nodes = tf.constant(preparation["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["quadrature"]["weights"], DTYPE)
    active_indices = tf.constant(preparation["active_indices"], tf.int32)
    row_scales = tf.constant(preparation["row_scales"], DTYPE)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        old = model.contract_e_tp_lgssm_score_informed_recursive_core(
            THETA,
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            feature_mode="finite_lookahead",
            lookahead_steps=8,
        )
        new = model.contract_e_tp_lgssm_finite_lookahead_loop_core(
            THETA,
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            lookahead_steps=8,
        )
    epsilon = np.finfo(np.float64).eps
    np.testing.assert_allclose(
        new["objective"], old["objective"], rtol=0.0, atol=64.0 * epsilon
    )
    np.testing.assert_allclose(
        tape.gradient(new["objective"], THETA),
        tape.gradient(old["objective"], THETA),
        rtol=3e-13,
        atol=3e-13,
    )
    for name in (
        "increment_history",
        "minimum_weight_history",
        "condition_number_history",
        "feature_residual_history",
        "target_history",
        "matched_target_history",
        "valid_history",
        "final_particles",
        "final_log_unnormalized_weights",
    ):
        np.testing.assert_allclose(new[name], old[name], rtol=3e-13, atol=3e-13)


def test_finite_lookahead_xla_route_uses_functional_loops_not_python_time_unroll() -> None:
    route_source = inspect.getsource(
        model.contract_e_tp_lgssm_finite_lookahead_loop_core
    )
    lookahead_source = inspect.getsource(
        model._finite_lookahead_information_parameters_loop
    )
    assert "tf.while_loop" in route_source
    assert "tf.while_loop" in lookahead_source
    for forbidden in ("for time_index in", "for observation in", "tf.unstack", "reversed("):
        assert forbidden not in route_source
        assert forbidden not in lookahead_source


def test_finite_lookahead_factory_graph_contains_two_functional_loops() -> None:
    preparation = json.loads(
        LOOKAHEAD_PREPARATIONS[10].read_text(encoding="utf-8")
    )
    evaluate = model.make_contract_e_tp_lgssm_score_informed_recursive_tf(
        _observations(10),
        tf.constant(preparation["quadrature"]["nodes"], DTYPE),
        tf.constant(preparation["quadrature"]["weights"], DTYPE),
        tf.constant(preparation["active_indices"], tf.int32),
        tf.constant(preparation["row_scales"], DTYPE),
        feature_mode="finite_lookahead",
        lookahead_steps=8,
        jit_compile=True,
    )
    graph = evaluate.get_concrete_function(THETA).graph.as_graph_def()
    operations = [node.op for node in graph.node]
    function_operations = [
        node.op for function in graph.library.function for node in function.node_def
    ]
    while_count = sum(
        operation in ("While", "StatelessWhile")
        for operation in operations + function_operations
    )
    assert while_count >= 2

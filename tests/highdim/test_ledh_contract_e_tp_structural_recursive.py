from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_structural_tf as structural
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import (
    SourceRouteSpec,
    audit_source_path,
    inventory_graph_def,
)


ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/"
    "phase-02/structural/attempt-01-local-20260715/preparation.json"
)
DTYPE = tf.float64


def _payload() -> dict[str, object]:
    return json.loads(PREPARATION.read_text(encoding="utf-8"))


def _inputs(time_steps: int):
    payload = _payload()
    fixture = payload["fixture"]
    steps = payload["steps"][:time_steps]
    return (
        tf.constant(fixture["theta"], DTYPE),
        tf.constant(fixture["initial_parents"], DTYPE),
        tf.constant(fixture["initial_weights"], DTYPE),
        tf.constant(fixture["innovations"], DTYPE),
        tf.constant(fixture["innovation_weights"], DTYPE),
        tf.constant([step["active_indices"] for step in steps], tf.int32),
        tf.constant([step["row_scales"] for step in steps], DTYPE),
    )


def _unrolled(theta: tf.Tensor, time_steps: int) -> dict[str, tf.Tensor]:
    _, parents, weights, innovations, innovation_weights, indices, scales = _inputs(time_steps)
    total = tf.constant(0.0, DTYPE)
    increments = []
    residuals = []
    for time_index in range(time_steps):
        repeated = tf.repeat(parents, tf.shape(innovations)[0], axis=0)
        tiled = tf.tile(innovations, [tf.shape(parents)[0], 1])
        components = structural.structural_fixture_transition_components_tf(
            repeated, tiled, theta
        )
        candidates = components["candidates"]
        _rho, _sigma, alpha, beta = tf.unstack(theta)
        residuals.append(
            (
                candidates[:, 1]
                - alpha * repeated[:, 1]
                - beta * tf.math.tanh(components["stochastic_preactivation"])
            )[:, None]
        )
        teacher_weights = tf.reshape(
            weights[:, None] * innovation_weights[None, :], [-1]
        )
        features = structural.structural_fixture_features_tf(candidates)
        projection = tp._contract_e_tp_dense_square_forward_core(
            candidates,
            tf.math.log(teacher_weights),
            features,
            indices[time_index],
            scales[time_index],
        )
        increment = tf.math.log1p(
            tf.reduce_sum(tf.square(projection["matched_target"]))
        )
        total += increment
        increments.append(increment)
        parents = projection["student_points"]
        weights = projection["student_weights"]
    return {
        "objective": total,
        "increments": tf.stack(increments),
        "residuals": tf.stack(residuals),
        "final_parents": parents,
        "final_weights": weights,
    }


def test_structural_preparation_is_fixed_positive_and_auditable() -> None:
    payload = _payload()
    assert payload["selection_rule"]["runtime_selection"] is False
    assert [step["active_indices"] for step in payload["steps"]] == [
        [2, 4, 6, 11],
        [3, 4, 7, 10],
        [0, 5, 8, 9],
        [1, 4, 8, 9],
        [2, 5, 6, 9],
    ]
    assert min(step["minimum_weight"] for step in payload["steps"]) > 0.18
    assert min(step["viable_count"] for step in payload["steps"]) >= 79


def test_structural_loop_matches_fixed_unrolled_value_score_and_state() -> None:
    for time_steps in (1, 2, 5):
        theta, parents, weights, innovations, innovation_weights, indices, scales = _inputs(time_steps)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta)
            loop = structural.structural_fixture_recursive_core_tf(
                theta,
                parents,
                weights,
                innovations,
                innovation_weights,
                indices,
                scales,
                tf.constant(0.0, DTYPE),
            )
            reference = _unrolled(theta, time_steps)
        np.testing.assert_allclose(loop["objective"], reference["objective"], rtol=0, atol=2e-15)
        np.testing.assert_allclose(loop["increment_history"], reference["increments"], rtol=0, atol=2e-15)
        np.testing.assert_allclose(loop["final_parents"], reference["final_parents"], rtol=0, atol=2e-15)
        np.testing.assert_allclose(loop["final_weights"], reference["final_weights"], rtol=0, atol=3e-15)
        np.testing.assert_allclose(tape.gradient(loop["objective"], theta), tape.gradient(reference["objective"], theta), rtol=3e-13, atol=3e-14)
        assert bool(loop["valid"])


def test_structural_total_residual_tangent_and_same_scalar_fd() -> None:
    theta, parents, weights, innovations, innovation_weights, indices, scales = _inputs(5)
    evaluate = structural.make_structural_fixture_recursive_tf(
        parents, weights, innovations, innovation_weights, indices, scales, jit_compile=False
    )
    result = evaluate(theta, tf.constant(0.0, DTYPE))
    assert bool(result["valid"])
    assert bool(result["tangent_valid"])
    assert bool(result["expansion_valid"])
    assert bool(tf.reduce_all(result["kernel_match_history"]))
    assert float(tf.reduce_max(tf.abs(result["residual_history"]))) <= float(tf.reduce_max(result["value_bound_history"]))
    assert float(tf.reduce_max(tf.abs(result["residual_jacobian"]))) <= float(tf.reduce_max(result["tangent_bound"]))
    assert int(result["value_operation_count"]) == 16
    assert int(result["tangent_operation_count"]) == 64
    step = 1.0e-5
    fd = []
    for parameter in range(4):
        direction = np.zeros(4)
        direction[parameter] = step
        plus = structural.structural_fixture_recursive_core_tf(
            tf.constant(theta.numpy() + direction, DTYPE), parents, weights,
            innovations, innovation_weights, indices, scales, tf.constant(0.0, DTYPE)
        )["objective"]
        minus = structural.structural_fixture_recursive_core_tf(
            tf.constant(theta.numpy() - direction, DTYPE), parents, weights,
            innovations, innovation_weights, indices, scales, tf.constant(0.0, DTYPE)
        )["objective"]
        fd.append(float((plus - minus) / (2.0 * step)))
    np.testing.assert_allclose(result["score"], fd, rtol=2e-8, atol=2e-10)


def test_same_factory_fails_closed_for_off_support_input() -> None:
    theta, parents, weights, innovations, innovation_weights, indices, scales = _inputs(2)
    evaluate = structural.make_structural_fixture_recursive_tf(
        parents, weights, innovations, innovation_weights, indices, scales, jit_compile=False
    )
    valid = evaluate(theta, tf.constant(0.0, DTYPE))
    invalid = evaluate(theta, tf.constant(1.0e-3, DTYPE))
    assert bool(valid["valid"])
    assert not bool(invalid["valid"])
    assert 1.0e-3 >= 1.0e6 * float(tf.reduce_max(valid["value_bound_history"]))
    for value in (
        invalid["objective"], invalid["score"], invalid["final_parents"], invalid["final_weights"]
    ):
        assert not bool(tf.reduce_all(tf.math.is_finite(value)))


def test_structural_loop_source_and_graph_guard_pass() -> None:
    source = Path(structural.__file__)
    audit = audit_source_path(
        source,
        SourceRouteSpec(
            roots=("structural_fixture_recursive_core_tf",),
            loop_roles={},
            required_reachable=(
                "structural_fixture_recursive_core_tf.cond",
                "structural_fixture_recursive_core_tf.body",
            ),
        ),
    )
    assert audit["approved"], audit
    theta, parents, weights, innovations, innovation_weights, indices, scales = _inputs(2)
    evaluate = structural.make_structural_fixture_recursive_tf(
        parents, weights, innovations, innovation_weights, indices, scales, jit_compile=True
    )
    graph = inventory_graph_def(evaluate.get_concrete_function().graph.as_graph_def())
    assert graph["functional_loop_count"] >= 1


from __future__ import annotations

import os
import json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as sv
from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import (
    SourceRouteSpec,
    audit_source_path,
    inventory_graph_def,
)


DTYPE = tf.float64
ROOT = Path(__file__).resolve().parents[2]


def test_tensorflow_chart_preparation_is_deterministic_and_exact() -> None:
    points = tf.constant([[-2.0], [-1.0], [0.0], [0.5], [1.0], [2.0]], DTYPE)
    weights = tf.constant([0.05, 0.10, 0.30, 0.25, 0.20, 0.10], DTYPE)
    values = points[:, 0]
    features = tf.stack(
        [tf.ones_like(values), values, tf.square(values), tf.exp(-0.2 * tf.square(values))]
    )
    first = sv.prepare_actual_sv_overcomplete_chart_step(
        points, tf.math.log(weights), features, 5
    )
    second = sv.prepare_actual_sv_overcomplete_chart_step(
        points, tf.math.log(weights), features, 5
    )
    assert bool(first["preparation_valid"].numpy())
    np.testing.assert_array_equal(first["active_indices"], second["active_indices"])
    np.testing.assert_allclose(
        first["reference_weights"], second["reference_weights"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        first["matched_target"], first["target"], rtol=0.0, atol=2e-14
    )
    assert float(tf.reduce_min(first["voronoi_weights"]).numpy()) > 0.0
    assert float(tf.reduce_min(first["reference_weights"]).numpy()) > 0.0


def test_overcomplete_loop_t1_requires_no_prepared_projection() -> None:
    spec = sv.make_scalar_sv_spec("zhao_cui_sv_actual_nongaussian_T1000")
    theta = tf.constant([0.2533471031357997, -0.916290731874155], DTYPE)
    nodes = tf.constant([-1.0, 0.0, 1.0], DTYPE)
    weights = tf.constant([0.25, 0.5, 0.25], DTYPE)
    target, flow = sv.target_and_flow_observations(spec, tf.constant([[0.7]], DTYPE))
    result = sv.contract_e_tp_actual_sv_overcomplete_loop_core(
        spec,
        theta,
        target,
        flow,
        nodes,
        weights,
        tf.zeros([0, 5], tf.int32),
        tf.zeros([0, sv.FEATURE_COUNT], DTYPE),
        tf.zeros([0, 5], DTYPE),
        tf.constant([-2.0, 0.0, 2.0], DTYPE),
        tf.constant([1.0, 1.0, 1.0], DTYPE),
        lookahead_steps=1,
    )
    assert bool(tf.reduce_all(result["valid_history"]).numpy())
    assert bool(tf.math.is_finite(result["objective"]).numpy())


def test_actual_sv_initial_teacher_manual_tangent_matches_forward_ad() -> None:
    spec = sv.make_scalar_sv_spec("zhao_cui_sv_actual_nongaussian_T1000")
    theta = tf.constant([0.2533471031357997, -0.916290731874155], DTYPE)
    nodes = tf.constant([-1.2, 0.0, 0.8], DTYPE)
    weights = tf.constant([0.2, 0.5, 0.3], DTYPE)
    target, flow = sv.target_and_flow_observations(spec, tf.constant([[0.7]], DTYPE))
    manual = sv._actual_sv_initial_teacher_multi_jvp(
        spec, theta, nodes, tf.math.log(weights), target[0], flow[0]
    )
    parents, parent_log_weights, standard_nodes, standard_log_weights = sv.initial_rule(
        spec, theta, nodes, weights
    )
    primal = sv._teacher_step(
        spec,
        theta,
        parents,
        parent_log_weights,
        standard_nodes,
        standard_log_weights,
        target[0],
        flow[0],
        0,
    )
    for name in ("particles", "log_unnormalized_weights", "increment"):
        np.testing.assert_allclose(manual[name], primal[name], rtol=2e-13, atol=2e-14)
    for direction in range(2):
        basis = tf.one_hot(direction, 2, dtype=DTYPE)
        with tf.autodiff.ForwardAccumulator(theta, basis) as accumulator:
            ad_parents, ad_parent_logs, ad_nodes, ad_node_logs = sv.initial_rule(
                spec, theta, nodes, weights
            )
            automatic = sv._teacher_step(
                spec,
                theta,
                ad_parents,
                ad_parent_logs,
                ad_nodes,
                ad_node_logs,
                target[0],
                flow[0],
                0,
            )
        for name, tangent_name in (
            ("particles", "particles_tangent"),
            ("log_unnormalized_weights", "log_unnormalized_weights_tangent"),
            ("increment", "increment_tangent"),
        ):
            np.testing.assert_allclose(
                manual[tangent_name][..., direction],
                accumulator.jvp(automatic[name]),
                rtol=4e-12,
                atol=4e-13,
            )


def test_actual_sv_transition_teacher_manual_tangent_matches_forward_ad() -> None:
    spec = sv.make_scalar_sv_spec("zhao_cui_sv_actual_nongaussian_T1000")
    theta = tf.constant([0.2533471031357997, -0.916290731874155], DTYPE)
    parents = tf.constant([-0.8, 0.1, 0.9], DTYPE)
    parent_log_weights = tf.math.log(tf.constant([0.2, 0.5, 0.3], DTYPE))
    nodes = tf.constant([-1.0, 0.0, 1.0], DTYPE)
    node_log_weights = tf.math.log(tf.constant([0.25, 0.5, 0.25], DTYPE))
    target, flow = sv.target_and_flow_observations(
        spec, tf.constant([[0.7], [1.1]], DTYPE)
    )
    parent_tangents = tf.constant(
        [[0.03, -0.04], [0.01, 0.02], [-0.02, 0.05]], DTYPE
    )
    parent_log_weight_tangents = tf.constant(
        [[-0.02, 0.01], [0.04, -0.03], [0.01, 0.02]], DTYPE
    )
    manual = sv._actual_sv_transition_teacher_multi_jvp(
        spec,
        theta,
        parents,
        parent_tangents,
        parent_log_weights,
        parent_log_weight_tangents,
        nodes,
        node_log_weights,
        target[1],
        flow[1],
    )
    for direction in range(2):
        basis = tf.one_hot(direction, 2, dtype=DTYPE)
        with tf.autodiff.ForwardAccumulator(
            (theta, parents, parent_log_weights),
            (
                basis,
                parent_tangents[:, direction],
                parent_log_weight_tangents[:, direction],
            ),
        ) as accumulator:
            automatic = sv._teacher_transition_step_loop(
                spec,
                theta,
                parents,
                parent_log_weights,
                nodes,
                node_log_weights,
                target[1],
                flow[1],
                tf.constant(1, tf.int32),
            )
        for name, tangent_name in (
            ("particles", "particles_tangent"),
            ("log_unnormalized_weights", "log_unnormalized_weights_tangent"),
            ("increment", "increment_tangent"),
        ):
            np.testing.assert_allclose(
                manual[tangent_name][..., direction],
                accumulator.jvp(automatic[name]),
                rtol=5e-12,
                atol=5e-13,
            )


def test_actual_sv_continuation_and_features_manual_tangent_match_forward_ad() -> None:
    spec = sv.make_scalar_sv_spec("zhao_cui_sv_actual_nongaussian_T1000")
    theta = tf.constant([0.2533471031357997, -0.916290731874155], DTYPE)
    points = tf.constant([[-0.7], [0.2], [1.1]], DTYPE)
    point_tangents = tf.constant(
        [[[0.03, -0.02]], [[-0.01, 0.04]], [[0.05, 0.01]]], DTYPE
    )
    target, _ = sv.target_and_flow_observations(
        spec, tf.constant([[0.7], [1.1], [0.9]], DTYPE)
    )
    grid = tf.constant([[-1.5], [-0.3], [0.8], [1.7]], DTYPE)
    grid_weights = tf.constant([0.3, 0.7, 0.8, 0.2], DTYPE)
    manual_value, manual_dot = sv._actual_sv_continuation_multi_jvp(
        spec,
        theta,
        points,
        point_tangents,
        target[1:],
        tf.constant(2, tf.int32),
        grid,
        grid_weights,
        first_future_time_index=tf.constant(1, tf.int32),
    )
    features, feature_dots = sv._actual_sv_features_multi_jvp(
        spec,
        theta,
        points,
        point_tangents,
        target[1:],
        tf.constant(2, tf.int32),
        grid,
        grid_weights,
        first_future_time_index=tf.constant(1, tf.int32),
    )
    for direction in range(2):
        basis = tf.one_hot(direction, 2, dtype=DTYPE)
        with tf.autodiff.ForwardAccumulator(
            (theta, points), (basis, point_tangents[..., direction])
        ) as accumulator:
            automatic = sv.target_continuation_log_likelihood_loop(
                spec,
                theta,
                points,
                target[1:],
                tf.constant(2, tf.int32),
                grid,
                grid_weights,
                first_future_time_index=tf.constant(1, tf.int32),
            )
            automatic_features = sv._features_loop(
                spec,
                theta,
                points,
                target[1:],
                tf.constant(2, tf.int32),
                grid,
                grid_weights,
                first_future_time_index=tf.constant(1, tf.int32),
            )
        np.testing.assert_allclose(manual_value, automatic, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(
            manual_dot[..., direction],
            accumulator.jvp(automatic),
            rtol=8e-12,
            atol=8e-13,
        )
        np.testing.assert_allclose(features, automatic_features, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(
            feature_dots[..., direction],
            accumulator.jvp(automatic_features),
            rtol=1e-11,
            atol=1e-12,
        )


def test_t2_manual_recursive_score_matches_autodiff_oracle() -> None:
    path = ROOT / (
        "docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/"
        "phase-02-smoke/attempt-01-t2-k5-preparation.json"
    )
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = sv.make_scalar_sv_spec(payload["row_id"])
    time_steps = payload["target"]["time_steps"]
    capacity = payload["chart_contract"]["anchor_count"]
    arguments = (
        spec,
        tf.constant(payload["target"]["target_observations"], DTYPE),
        tf.constant(payload["target"]["flow_observations"], DTYPE),
        tf.constant(payload["teacher_quadrature"]["nodes"], DTYPE),
        tf.constant(payload["teacher_quadrature"]["weights"], DTYPE),
        tf.reshape(tf.constant(payload["active_indices"], tf.int32), [time_steps - 1, capacity]),
        tf.reshape(tf.constant(payload["row_scales"], DTYPE), [time_steps - 1, sv.FEATURE_COUNT]),
        tf.reshape(tf.constant(payload["reference_weights"], DTYPE), [time_steps - 1, capacity]),
        tf.constant(payload["continuation_quadrature"]["points"], DTYPE),
        tf.constant(payload["continuation_quadrature"]["weights"], DTYPE),
    )
    manual = sv.make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf(
        *arguments, lookahead_steps=payload["feature_contract"]["lookahead_steps"], jit_compile=False
    )
    automatic = sv.make_contract_e_tp_actual_sv_overcomplete_tf(
        *arguments, lookahead_steps=payload["feature_contract"]["lookahead_steps"], jit_compile=False
    )
    theta = tf.constant(payload["target"]["theta"], DTYPE)
    manual_result = manual(theta)
    automatic_result = automatic(theta)
    assert bool(manual_result["valid"].numpy())
    assert bool(automatic_result["valid"].numpy())
    for manual_name, automatic_name in (
        ("objective", "objective"),
        ("score_manual", "score_autodiff_oracle"),
        ("increment_history", "increment_history"),
        ("final_particles", "final_particles"),
        ("final_log_unnormalized_weights", "final_log_unnormalized_weights"),
    ):
        np.testing.assert_allclose(
            manual_result[manual_name], automatic_result[automatic_name], rtol=2e-11, atol=2e-12
        )


def test_manual_route_source_guard_and_t10_functional_loops() -> None:
    source_audit = audit_source_path(
        Path(sv.__file__),
        SourceRouteSpec(
            roots=("make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf",),
            loop_roles={},
            required_reachable=(
                "make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf.evaluate",
                "contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core",
                "contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core.cond",
                "contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core.body",
                "_actual_sv_continuation_multi_jvp",
                "_actual_sv_continuation_multi_jvp.cond",
                "_actual_sv_continuation_multi_jvp.body",
                "_actual_sv_features_multi_jvp",
                "_actual_sv_initial_teacher_multi_jvp",
                "_actual_sv_transition_teacher_multi_jvp",
            ),
        ),
    )
    assert source_audit["status"] == "PASS_SOURCE_GUARD"

    path = ROOT / (
        "docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/"
        "phase-03-capacity/attempt-01-t10-k7-preparation.json"
    )
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    time_steps = payload["target"]["time_steps"]
    capacity = payload["chart_contract"]["anchor_count"]
    factory = sv.make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf(
        sv.make_scalar_sv_spec(payload["row_id"]),
        tf.constant(payload["target"]["target_observations"], DTYPE),
        tf.constant(payload["target"]["flow_observations"], DTYPE),
        tf.constant(payload["teacher_quadrature"]["nodes"], DTYPE),
        tf.constant(payload["teacher_quadrature"]["weights"], DTYPE),
        tf.reshape(tf.constant(payload["active_indices"], tf.int32), [time_steps - 1, capacity]),
        tf.reshape(tf.constant(payload["row_scales"], DTYPE), [time_steps - 1, sv.FEATURE_COUNT]),
        tf.reshape(tf.constant(payload["reference_weights"], DTYPE), [time_steps - 1, capacity]),
        tf.constant(payload["continuation_quadrature"]["points"], DTYPE),
        tf.constant(payload["continuation_quadrature"]["weights"], DTYPE),
        lookahead_steps=payload["feature_contract"]["lookahead_steps"],
        jit_compile=True,
    )
    graph = inventory_graph_def(factory.get_concrete_function().graph.as_graph_def())
    assert graph["functional_loop_count"] >= 2

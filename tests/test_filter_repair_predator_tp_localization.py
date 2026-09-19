"""First-projection localization of the preserved raw TP residual mismatch.

This is a fresh engineering diagnostic of the existing noncanonical fixture.
"""

import json

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_predator_prey_tf as candidate
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from tests.test_filter_repair_predator_tp import _fixture
from tests.test_filter_repair_remaining_routes import _original

D = tf.float64


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("freeze", ["none", "teacher", "features"])
def test_first_tp_projection_breakdown(jit, freeze, record_property):
    before = _original("ledh_contract_e_tp_predator_prey_tf")
    model = before.p30_predator_prey_fixture_model()
    theta, observations, nodes, weights, indices, scales, grid, grid_weights = _fixture(4, 2)
    parents, log_weights, points, standard_log_weights = before.initial_rule(model, nodes, weights)
    frozen = before._teacher_step(model, theta, parents, log_weights, points,
        standard_log_weights, observations[0], 0)
    frozen_features = before._features(model, theta, frozen["particles"], observations[1:3],
        grid, grid_weights, first_future_time_index=1)
    inputs = (theta, observations, nodes, weights, indices, scales, grid, grid_weights,
              frozen["particles"], frozen["log_unnormalized_weights"], frozen_features)
    specifications = tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)

    def build(module):
        @tf.function(input_signature=specifications, jit_compile=jit, autograph=False)
        def execute(theta, observations, nodes, weights, indices, scales, grid, grid_weights,
                    frozen_points, frozen_log_weights, fixed_features):
            parents, log_weights, points, standard_log_weights = module.initial_rule(model, nodes, weights)
            teacher = module._teacher_step(model, theta, parents, log_weights, points,
                standard_log_weights, observations[0], 0)
            cloud = frozen_points if freeze != "none" else teacher["particles"]
            logs = frozen_log_weights if freeze != "none" else teacher["log_unnormalized_weights"]
            features = module._features(model, theta, cloud, observations[1:3], grid,
                grid_weights, first_future_time_index=1)
            if freeze == "features":
                features = fixed_features
            projected = tp._contract_e_tp_dense_square_forward_core(
                cloud, logs, features, indices[0], scales[0])
            return {"points": cloud, "log_weights": logs, "features": features,
                "scaled_matrix": projected["active_features"] / scales[0, :, None],
                "student_weights": projected["student_weights"],
                "matched_target": projected["matched_target"],
                "raw_residual": projected["feature_residual"]}
        return execute

    expected, actual = build(before)(*inputs), build(candidate)(*inputs)
    errors = {name: float(tf.reduce_max(tf.abs(actual[name] - expected[name]))) for name in actual}
    record_property("maximum_absolute_errors", json.dumps(errors, sort_keys=True))
    record_property("baseline_raw_residual", json.dumps(expected["raw_residual"].numpy().tolist()))
    record_property("candidate_raw_residual", json.dumps(actual["raw_residual"].numpy().tolist()))
    print(json.dumps({"jit": jit, "freeze": freeze, "errors": errors,
        "baseline_raw_residual": expected["raw_residual"].numpy().tolist(),
        "candidate_raw_residual": actual["raw_residual"].numpy().tolist()}, sort_keys=True))
    for name in expected:
        tf.debugging.assert_near(actual[name], expected[name], atol=1e-10, rtol=1e-10, message=name)

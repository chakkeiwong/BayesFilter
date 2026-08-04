from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.fixed_sir_sgqf_tf import (
    FIXED_SIR_SGQF_CLOUD_SHA256,
    FIXED_SIR_SGQF_OBSERVATION_SHA256,
    FIXED_SIR_SGQF_ROUTE_ID,
    FIXED_SIR_SGQF_STATE_SHA256,
    fixed_sir_sgqf_value_only_status,
    generate_fixed_sir_source_dataset_tf,
    make_fixed_sir_sgqf_route,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_level2_axis_cloud
from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
    sir_sgqf_likelihood_value_only_status,
)


ROOT = Path(__file__).resolve().parents[2]


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def test_fixed_sir_dataset_has_twenty_transition_then_observe_steps() -> None:
    states, observations = generate_fixed_sir_source_dataset_tf()

    assert states.shape == (21, 18)
    assert observations.shape == (20, 9)
    assert _tensor_hash(states) == FIXED_SIR_SGQF_STATE_SHA256
    assert _tensor_hash(observations) == FIXED_SIR_SGQF_OBSERVATION_SHA256
    assert float(tf.reduce_min(states).numpy()) > 0.0


def test_fixed_sir_level2_cloud_preserves_known_moments_and_limitation() -> None:
    cloud = tf_fixed_sgqf_level2_axis_cloud(18)
    cloud_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(cloud.points).numpy())
        + bytes(tf.io.serialize_tensor(cloud.weights).numpy())
    ).hexdigest()

    assert cloud.point_count == 37
    assert cloud.negative_weight_count == 1
    assert cloud_hash == FIXED_SIR_SGQF_CLOUD_SHA256
    tf.debugging.assert_near(tf.reduce_sum(cloud.weights), 1.0, atol=2e-15)
    tf.debugging.assert_near(
        tf.einsum("r,ri,rj->ij", cloud.weights, cloud.points, cloud.points),
        tf.eye(18, dtype=tf.float64),
        atol=2e-15,
    )
    # Axis-only points miss mixed fourth moments: this remains a scientific risk.
    mixed_fourth = tf.reduce_sum(
        cloud.weights * tf.square(cloud.points[:, 0]) * tf.square(cloud.points[:, 1])
    )
    tf.debugging.assert_equal(mixed_fourth, tf.constant(0.0, tf.float64))


def test_prefix_mechanics_execute_exactly_one_or_two_filter_steps() -> None:
    _states, observations = generate_fixed_sir_source_dataset_tf()
    value_t1, status_t1 = fixed_sir_sgqf_value_only_status(observations[:1])
    value_t2, status_t2 = fixed_sir_sgqf_value_only_status(observations[:2])

    assert bool(tf.math.is_finite(value_t1).numpy())
    assert bool(tf.math.is_finite(value_t2).numpy())
    assert int(status_t1["status_code"].numpy()) == 0
    assert int(status_t2["status_code"].numpy()) == 0
    assert float(value_t1.numpy()) != float(value_t2.numpy())


def test_fixed_route_is_sealed_value_only_and_replayable() -> None:
    first = make_fixed_sir_sgqf_route()
    second = make_fixed_sir_sgqf_route()
    first_value, first_status = first.value_only_status()
    second_value, second_status = second.value_only_status()

    assert first.manifest["route_id"] == FIXED_SIR_SGQF_ROUTE_ID
    assert first.parameter_dim == 0
    assert first.required_result_kind == "value_only_no_free_theta"
    assert "score" not in type(first).__dict__
    assert first.route_identity == second.route_identity
    tf.debugging.assert_equal(first_value, second_value)
    tf.debugging.assert_equal(first_status["status_code"], 0)
    tf.debugging.assert_equal(second_status["status_code"], 0)


def test_fixed_route_matches_shared_zero_scale_mechanics_only() -> None:
    route = make_fixed_sir_sgqf_route()
    cloud = tf_fixed_sgqf_level2_axis_cloud(18)
    actual, status = route.value_only_status()
    reference, reference_status = sir_sgqf_likelihood_value_only_status(
        tf.zeros([1, 3], dtype=tf.float64),
        observations=route.observations,
        nodes=cloud.points,
        weights=cloud.weights,
    )

    tf.debugging.assert_near(actual, reference[0], atol=1e-10, rtol=1e-12)
    tf.debugging.assert_equal(status["status_code"], reference_status["status_code"][0])


def test_json_safe_serializes_tensor_vectors_without_scalar_coercion() -> None:
    script = ROOT / "docs/benchmarks/run_fixed_sir_sgqf_validation.py"
    spec = importlib.util.spec_from_file_location("fixed_sir_validation_runner", script)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load fixed SIR validation runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._json_safe(tf.constant([1.0, 2.0], tf.float64)) == [1.0, 2.0]


def test_validation_runner_does_not_execute_owner_excluded_sir_ukf() -> None:
    source = (
        ROOT / "docs/benchmarks/run_fixed_sir_sgqf_validation.py"
    ).read_text(encoding="utf-8")

    assert "sir_ukf_likelihood_value_score_status" not in source
    assert '"SIR-UKF": "OWNER_EXCLUDED_METHOD_NOT_APPLICABLE"' in source

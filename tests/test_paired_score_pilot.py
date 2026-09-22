"""Independent NumPy verification oracles for the optional TF local pilot."""

import dataclasses

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.batched_local_center import BatchedLocalCenterConfig
from bayesfilter.inference.batched_quadratic_center import (
    BatchedQuadraticCenterConfig,
    initialize_batched_posterior_local_location_scale,
    refine_batched_quadratic_center,
)
from bayesfilter.inference.paired_score_pilot_tf import (
    covariance_unit_score_error_tf,
    fit_paired_score_precision_tf,
    paired_score_probe_designs_tf,
)


def fit_inputs(precision, center_score=None, steps=(0.001, 0.0001)):
    precision = tf.constant(precision, tf.float64)
    dimension = int(precision.shape[0])
    center = tf.zeros(dimension, tf.float64) if center_score is None else tf.constant(center_score, tf.float64)
    first, second, check = paired_score_probe_designs_tf(dimension, seed=731, round_index=0, steps=steps)
    return (center, center - first @ tf.transpose(precision), center - second @ tf.transpose(precision),
            check, center - check @ tf.transpose(precision))


def gaussian_callback(precision, mean=None, batch=4):
    precision = tf.constant(precision, tf.float64)
    dimension = int(precision.shape[0])
    mean = tf.zeros(dimension, tf.float64) if mean is None else tf.constant(mean, tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([batch, dimension], tf.float64)], autograph=False)
    def callback(points):
        delta = points - mean
        score = -delta @ precision
        return 0.5 * tf.reduce_sum(delta * score, axis=1), score, tf.ones([batch], tf.bool)

    return callback


@pytest.mark.parametrize("linear", [[0., 0.], [12., -31.]])
@pytest.mark.parametrize("steps", [(0.001, 0.0001), (0.01, 0.001), (0.1, 0.01)])
def test_exact_quadratic_matches_unrestricted_lstsq(linear, steps):
    precision = np.array([[7., 2.], [2., 3.]])
    inputs = fit_inputs(precision, linear, steps)
    result = fit_paired_score_precision_tf(*inputs, steps=steps)
    assert bool(result["accepted"])
    first, second, _ = paired_score_probe_designs_tf(2, seed=731, round_index=0, steps=steps)
    for index, offsets in enumerate((first, second)):
        response = inputs[0].numpy() - inputs[index + 1].numpy()
        reference = np.linalg.lstsq(offsets.numpy(), response, rcond=None)[0].T
        np.testing.assert_allclose(result["raw_unrestricted_precisions"][index], reference, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result["raw_precision"], precision, rtol=1e-9, atol=1e-10)


def test_dense_factor_score_orientation():
    factor = np.array([[2., 0.3], [-0.7, 0.5]])
    physical_precision = np.array([[4., 1.2], [1.2, 3.]])
    expected = factor.T @ physical_precision @ factor
    center_score = np.array([0.7, -0.4])
    first, second, check = paired_score_probe_designs_tf(2, seed=901, round_index=0)

    def scores(offsets):
        physical_offsets = offsets.numpy() @ factor.T
        return tf.constant((center_score - physical_offsets @ physical_precision) @ factor, tf.float64)

    result = fit_paired_score_precision_tf(tf.constant(center_score @ factor), scores(first), scores(second), check, scores(check))
    assert bool(result["accepted"])
    np.testing.assert_allclose(result["raw_precision"], expected, rtol=1e-10, atol=1e-10)
    position_covariance = factor @ np.linalg.solve(result["raw_precision"], factor.T)
    np.testing.assert_allclose(position_covariance, np.linalg.inv(physical_precision), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("magnitude", [1e-150, 1., 1e150])
def test_extreme_spd_scaling_stays_finite(magnitude):
    result = fit_paired_score_precision_tf(*fit_inputs(np.diag([1., 1e8]) * magnitude))
    assert bool(result["accepted"])
    assert float(result["precision_condition"]) == pytest.approx(1e8)
    np.testing.assert_allclose(result["raw_precision"] / magnitude, np.diag([1., 1e8]), rtol=1e-12)


@pytest.mark.parametrize("matrix", [np.diag([1., -1.]), np.diag([1., 0.]), np.diag([1., 1e11]), [[1., .1], [0., 1.]]])
def test_bad_curvature_is_rejected_without_projection(matrix):
    result = fit_paired_score_precision_tf(*fit_inputs(matrix))
    assert not bool(result["accepted"])
    expected = np.asarray(matrix, dtype=float)
    np.testing.assert_allclose(result["raw_unrestricted_precisions"][1], expected, atol=1e-12)
    np.testing.assert_allclose(result["raw_precision"], 0.5 * (expected + expected.T), atol=1e-12)


def test_both_steps_must_pass_and_small_step_is_never_replaced():
    inputs = list(fit_inputs(np.eye(2)))
    inputs[1] *= 1.1
    result = fit_paired_score_precision_tf(*inputs)
    assert not bool(result["accepted"])
    np.testing.assert_allclose(result["raw_precision"], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(result["generalized_eigenvalues"], 1.1, atol=1e-12)
    inputs[1] *= -1
    rejected = fit_paired_score_precision_tf(*inputs)
    assert not bool(rejected["accepted"])
    assert not bool(rejected["step_raw_spd"][0])
    assert bool(rejected["step_raw_spd"][1])


def test_holdout_can_only_veto_not_refit():
    inputs = list(fit_inputs([[8., 1.], [1., 2.]]))
    original = fit_paired_score_precision_tf(*inputs)
    inputs[-1] += tf.constant([100., -100.], tf.float64)
    rejected = fit_paired_score_precision_tf(*inputs)
    assert bool(original["accepted"]) and not bool(rejected["accepted"])
    np.testing.assert_array_equal(rejected["raw_precision"], original["raw_precision"])


def test_weak_direction_error_vetoes_good_unweighted_rmse():
    inputs = list(fit_inputs(np.diag([1e-4, 1e4])))
    perturbation = tf.concat((tf.ones([2, 1], tf.float64), -tf.ones([2, 1], tf.float64)), axis=0) * 0.001
    inputs[-1] += tf.concat((perturbation, tf.zeros_like(perturbation)), axis=1)
    result = fit_paired_score_precision_tf(*inputs)
    assert float(result["selection_relative_rmse"]) < 0.20
    assert float(result["covariance_unit_error"]) > 0.05
    assert not bool(result["accepted"])


def test_covariance_error_invariant_at_same_physical_points():
    precision = np.array([[4., 1.], [1., 2.]])
    offsets = np.array([[.01, -.02], [-.02, .03]])
    response = offsets @ precision + [[.002, -.001], [.001, .003]]
    change = np.array([[.7, 1.1], [-.2, 2.]])
    original = covariance_unit_score_error_tf(tf.constant(precision), tf.constant(response), tf.constant(offsets))
    transformed = covariance_unit_score_error_tf(tf.constant(change.T @ precision @ change),
                                                  tf.constant(response @ change),
                                                  tf.constant(np.linalg.solve(change, offsets.T).T))
    assert float(original) == pytest.approx(float(transformed), rel=1e-12)


@pytest.mark.parametrize("width", [0., 1e-150])
def test_covariance_error_zero_and_tiny_denominator(width):
    precision = tf.eye(2, dtype=tf.float64)
    offsets = width * precision
    error = covariance_unit_score_error_tf(precision, offsets, offsets)
    assert np.isinf(float(error)) if width == 0 else float(error) == 0


@pytest.mark.parametrize("index", range(5))
def test_nonfinite_inputs_never_admit(index):
    inputs = list(fit_inputs(np.eye(2)))
    inputs[index] *= tf.constant(np.nan, tf.float64)
    result = fit_paired_score_precision_tf(*inputs)
    assert not bool(result["accepted"])


def test_graph_replay_and_frame_independence():
    inputs = fit_inputs([[3., .7], [.7, 1.]])
    compiled = tf.function(fit_paired_score_precision_tf, input_signature=[tf.TensorSpec([2], tf.float64)] +
                           [tf.TensorSpec([4, 2], tf.float64)] * 4, autograph=False, jit_compile=False)
    original = compiled(*inputs)
    replay = compiled(*inputs)
    assert bool(original["accepted"])
    np.testing.assert_array_equal(original["raw_precision"], replay["raw_precision"])
    assert compiled.experimental_get_tracing_count() == 1
    first = paired_score_probe_designs_tf(2, seed=11, round_index=0)
    other = paired_score_probe_designs_tf(2, seed=11, round_index=1)
    np.testing.assert_array_equal(first[0], other[0])
    assert not np.array_equal(first[2], other[2])
    np.testing.assert_allclose(first[2][:2] @ tf.transpose(first[2][:2]), 1e-6 * np.eye(2), atol=1e-20)


@pytest.mark.parametrize("steps", [(0., .0001), (.0001, .001), (.001, .001), (np.inf, .001)])
def test_invalid_step_contract(steps):
    with pytest.raises(ValueError, match="paired steps"):
        BatchedQuadraticCenterConfig(pilot_method="paired_local", paired_steps=steps)


@pytest.mark.parametrize("units", [[1., 1.], [1e-6, 1e4]])
@pytest.mark.parametrize("steps", [(0.001, 0.0001), (.01, .001)])
def test_initializer_covariance_does_not_shrink(units, steps):
    units = np.array(units)
    covariance = np.array([[.4, .1], [.1, .8]]) * units[:, None] * units[None, :]
    callback = gaussian_callback(np.linalg.inv(covariance))
    config = BatchedQuadraticCenterConfig(pilot_method="paired_local", paired_steps=steps, jit_compile_trust=False)
    result = refine_batched_quadratic_center(callback, [0., 0.], units, config=config)
    assert result.accepted, result.status
    np.testing.assert_allclose((result.pilot_factor @ tf.transpose(result.pilot_factor)) / units[:, None] / units[None, :],
                               covariance / units[:, None] / units[None, :], rtol=1e-10, atol=1e-10)
    assert result.diagnostics["fit_trace_count"] == 1
    assert callback.experimental_get_tracing_count() == 1


def test_reservation_padding_and_failed_rows():
    config = BatchedQuadraticCenterConfig(batch_size=4, pilot_method="paired_local", jit_compile_trust=False)
    assert dataclasses.replace(config, batch_size=48).planned_rows(142) == 7776
    with pytest.raises(ValueError, match="before target calls"):
        refine_batched_quadratic_center(lambda points: pytest.fail("must not evaluate"), [0.], [1.],
                                         config=dataclasses.replace(config, max_physical_rows=1))
    result = refine_batched_quadratic_center(gaussian_callback([[1.]]), [0.], [1.], config=config)
    assert result.accepted
    assert result.diagnostics["physical_rows"] == 24
    assert result.diagnostics["callback_batches"] == 6
    assert result.diagnostics["padded_rows"] == 15
    calls = 0

    def invalid(points):
        nonlocal calls
        calls += 1
        return -tf.reduce_sum(points**2, axis=1), -2 * points, tf.fill([4], calls < 3)

    failed = refine_batched_quadratic_center(invalid, [0.], [1.], config=config)
    assert failed.status == "curvature_target_invalid" and failed.pilot_factor is None
    assert failed.diagnostics["physical_rows"] == 12


def test_improved_probe_never_hands_off_stale_geometry():
    config = BatchedQuadraticCenterConfig(pilot_method="paired_local", max_fit_rounds=1, jit_compile_trust=False)
    result = refine_batched_quadratic_center(gaussian_callback(np.eye(2)), [.1, .1], [1., 1.], config=config)
    assert not result.accepted and result.pilot_factor is None
    assert result.status == "refinement_round_limit"
    assert bool(result.diagnostics["rounds"][0]["cloud_changed_incumbent"])
    assert float(result.diagnostics["rounds"][0]["centeredness"]) < 0.5
    assert float(result.center_value) > -.01


def test_composed_initializer_calls_paired_helper_and_preserves_cap_veto():
    config = BatchedQuadraticCenterConfig(pilot_method="paired_local", jit_compile_trust=False)
    callback = gaussian_callback([[3., 1.], [1., 2.]], mean=[.2, -.3])
    starts = [[1., 1.], [-1., -1.], [2., -2.], [-2., 2.]]
    locator = BatchedLocalCenterConfig(trust_refinement_rounds=1, max_iterations=20, jit_compile=False)
    result = initialize_batched_posterior_local_location_scale(callback, starts, [1., 1.], config=config, locator_config=locator)
    assert result.accepted, result.status
    assert "relative_antisymmetry" in result.diagnostics["rounds"][-1]["model"]
    np.testing.assert_allclose(result.center, [.2, -.3], atol=1e-8)
    capped = initialize_batched_posterior_local_location_scale(callback, starts, [1., 1.], config=config,
              locator_config=dataclasses.replace(locator, max_optimizer_callback_batches_per_round=1))
    assert not capped.accepted and capped.pilot_factor is None
    assert "cap" in capped.status


def test_default_remains_regional_and_explicit_paired_rows_are_rejected():
    assert BatchedQuadraticCenterConfig().pilot_method == "uniform_cloud"
    assert BatchedQuadraticCenterConfig().planned_rows(142) == 9160
    with pytest.raises(ValueError, match="2D"):
        BatchedQuadraticCenterConfig(pilot_method="paired_local", rows_per_cloud=100)

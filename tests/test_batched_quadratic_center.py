"""Independent NumPy/SciPy oracles for the TF initializer; no model runtime."""

import ast
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from scipy.optimize import minimize

from bayesfilter.inference.batched_local_center import BatchedLocalCenterConfig
from bayesfilter.inference.batched_quadratic_center import (
    BatchedQuadraticCenterConfig,
    initialize_batched_posterior_local_location_scale,
    refine_batched_quadratic_center,
    solve_spd_quadratic_trust_region_tf,
)


def gaussian(mean, factor, batch=4, offset=0.0):
    mean = tf.constant(mean, tf.float64)
    factor = tf.constant(factor, tf.float64)
    dimension = int(mean.shape[0])
    precision = tf.linalg.cholesky_solve(factor, tf.eye(dimension, dtype=tf.float64))

    @tf.function(input_signature=[tf.TensorSpec([batch, dimension], tf.float64)], autograph=False)
    def callback(points):
        difference = points - mean
        score = -difference @ precision
        values = offset + 0.5 * tf.reduce_sum(difference * score, axis=1)
        return values, score, tf.ones([batch], tf.bool)

    return callback


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("radius", [0.05, 1.0, 10.0])
@pytest.mark.parametrize("multiplier", [1e-150, 1.0, 1e150])
def test_trust_solution_kkt_and_independent_optimizer(jit, radius, multiplier):
    precision = np.array([[4., 1.], [1., 2.]])
    linear = np.array([2., -3.])
    kernel = tf.function(solve_spd_quadratic_trust_region_tf,
                         input_signature=[tf.TensorSpec([2, 2], tf.float64), tf.TensorSpec([2], tf.float64),
                                          tf.TensorSpec([], tf.float64)], autograph=False, jit_compile=jit)
    result = kernel(precision * multiplier, linear * multiplier, tf.constant(radius, tf.float64))
    assert bool(result["valid"])
    step = result["step"].numpy()
    dual = float(result["multiplier"]) / multiplier
    np.testing.assert_allclose((precision + dual * np.eye(2)) @ step, linear, rtol=1e-9, atol=1e-10)
    assert np.linalg.norm(step) <= radius * (1 + 1e-10)
    assert abs(dual * (np.linalg.norm(step) - radius)) < 1e-9
    reference = minimize(lambda point: 0.5 * point @ precision @ point - linear @ point, np.zeros(2),
                         jac=lambda point: precision @ point - linear,
                         constraints={"type": "ineq", "fun": lambda point: radius**2 - point @ point,
                                      "jac": lambda point: -2*point}, method="SLSQP",
                         options={"ftol": 1e-12, "maxiter": 200})
    np.testing.assert_allclose(step, reference.x, rtol=1e-5, atol=1e-6)
    again = kernel(precision * multiplier, linear * multiplier, tf.constant(radius, tf.float64))
    np.testing.assert_array_equal(step, again["step"])
    assert kernel.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("precision,linear,radius", [
    ([[1., 0.], [0., -1.]], [0., 0.], 1.),
    ([[1., 0.], [0., 0.]], [1., 1.], 1.),
    ([[np.nan, 0.], [0., 1.]], [1., 1.], 1.),
    ([[1., 0.], [0., 1.]], [np.inf, 1.], 1.),
    ([[1., 0.], [0., 1.]], [1., 1.], 0.),
])
def test_invalid_trust_inputs_never_pass(precision, linear, radius):
    assert not bool(solve_spd_quadratic_trust_region_tf(precision, linear, radius)["valid"])


def test_zero_score_trust_solution():
    result = solve_spd_quadratic_trust_region_tf(tf.eye(2, dtype=tf.float64), tf.zeros(2, tf.float64), 1.)
    assert bool(result["valid"])
    assert not bool(result["boundary_active"])
    np.testing.assert_array_equal(result["step"], [0., 0.])


@pytest.mark.parametrize("units", [[1., 1.], [1e-7, 1e3]])
@pytest.mark.parametrize("width", [0.001, 0.1, 1.0])
def test_gaussian_center_and_covariance_do_not_shrink_with_cloud(units, width):
    units = np.asarray(units)
    factor = np.array([[0.4, 0.], [0.2, 0.7]]) * units[:, None]
    mean = np.array([0.4, -0.3]) * units
    callback = gaussian(mean, factor)
    result = refine_batched_quadratic_center(callback, mean + units * [1.2, -1.1], units,
                                             config=BatchedQuadraticCenterConfig(fit_half_width=width, centeredness_cap=1e-9))
    assert result.accepted, result.status
    np.testing.assert_allclose((result.center.numpy() - mean) / units, 0., atol=1e-7)
    covariance = result.pilot_factor.numpy() @ result.pilot_factor.numpy().T
    np.testing.assert_allclose(covariance / units[:, None] / units[None, :],
                               (factor @ factor.T) / units[:, None] / units[None, :], rtol=1e-7, atol=1e-10)
    assert callback.experimental_get_tracing_count() == 1
    assert result.diagnostics["fit_trace_count"] == 1
    json.dumps(result.payload(), allow_nan=False)


def test_half_unit_centeredness_has_gaussian_mahalanobis_meaning():
    factor = np.array([[0.3, 0.], [0.1, 0.4]])
    start = factor @ np.array([0.4, 0.1])
    result = refine_batched_quadratic_center(gaussian([0., 0.], factor), start, [1., 1.],
                                             config=BatchedQuadraticCenterConfig(fit_half_width=1e-12))
    assert result.accepted
    assert float(result.diagnostics["rounds"][-1]["centeredness"]) <= 0.5
    expected = np.linalg.norm(np.linalg.solve(factor, result.center))
    assert float(result.diagnostics["rounds"][-1]["centeredness"]) == pytest.approx(expected, rel=1e-3)


def test_complete_row_reservation_precedes_any_target_call():
    def forbidden(points):
        raise AssertionError("target must not be called")
    cfg = BatchedQuadraticCenterConfig(max_physical_rows=1)
    with pytest.raises(ValueError, match="before target calls"):
        refine_batched_quadratic_center(forbidden, [0., 0.], [1., 1.], config=cfg)
    with pytest.raises(ValueError, match="before target calls"):
        initialize_batched_posterior_local_location_scale(forbidden, np.zeros((4, 2)), [1., 1.], config=cfg)


def test_padding_and_exact_batch_accounting():
    callback = gaussian([0., 0.], np.eye(2))
    cfg = BatchedQuadraticCenterConfig(rows_per_cloud=5)
    result = refine_batched_quadratic_center(callback, [0., 0.], [1., 1.], config=cfg)
    assert result.accepted
    assert result.diagnostics["physical_rows"] == 28
    assert result.diagnostics["callback_batches"] == 7
    assert result.diagnostics["padded_rows"] == 15
    assert result.diagnostics["physical_rows"] <= cfg.planned_rows(2)


@pytest.mark.parametrize("kind", ["value", "score", "eligibility"])
def test_replay_checks_even_padded_duplicate_rows(kind):
    base = gaussian([0., 0.], np.eye(2))
    calls = tf.Variable(0, dtype=tf.int64)

    def callback(points):
        call = calls.assign_add(1)
        values, scores, eligible = base(points)
        def alter():
            if kind == "value":
                return tf.tensor_scatter_nd_add(values, [[3]], tf.constant([1.], tf.float64)), scores, eligible
            if kind == "score":
                return values, tf.tensor_scatter_nd_add(scores, [[3, 0]], tf.constant([1.], tf.float64)), eligible
            return values, scores, tf.tensor_scatter_nd_update(eligible, [[3]], [False])
        return tf.cond(call == 2, alter, lambda: (values, scores, eligible))

    result = refine_batched_quadratic_center(callback, [0., 0.], [1., 1.])
    assert not result.accepted
    assert result.status == "replay_mismatch"
    assert result.pilot_factor is None


@pytest.mark.parametrize("kind", ["finite_sentinel", "nonfinite_score", "saddle", "flat"])
def test_invalid_or_non_spd_cloud_cannot_manufacture_factor(kind):
    def callback(points):
        scores = -points
        values = -0.5 * tf.reduce_sum(points**2, axis=1)
        eligible = tf.ones([4], tf.bool)
        if kind == "finite_sentinel":
            eligible = tf.reduce_all(tf.abs(points) < 0.001, axis=1)
            values = tf.where(eligible, values, -1e200)
            scores = tf.where(eligible[:, None], scores, 0.)
        elif kind == "nonfinite_score":
            scores = tf.where(tf.abs(points) > 0.001, tf.constant(np.nan, tf.float64), scores)
        elif kind == "saddle":
            scores = points * [1., -1.]
            values = 0.5 * tf.reduce_sum(points * scores, axis=1)
        else:
            scores, values = tf.zeros_like(points), tf.zeros([4], tf.float64)
        return values, scores, eligible

    result = refine_batched_quadratic_center(callback, [0., 0.], [1., 1.])
    assert not result.accepted
    assert result.pilot_factor is None and result.precision_z is None


def test_new_incumbent_at_last_round_cannot_reuse_old_factor():
    result = refine_batched_quadratic_center(gaussian([0., 0.], np.eye(2)), [2., 1.], [1., 1.],
                                             config=BatchedQuadraticCenterConfig(max_fit_rounds=1, centeredness_cap=1e-9))
    assert result.status == "refinement_round_limit"
    assert not result.accepted and result.pilot_factor is None
    assert float(result.center_value) > -2.5
    report = result.diagnostics["rounds"][0]
    assert np.any(result.center.numpy() != report["anchor"].numpy())


def test_composed_initializer_and_cap_veto():
    callback = gaussian([0.2, -0.3], np.array([[0.4, 0.], [0.2, 0.8]]))
    initial = np.array([[1., -1.], [-1., 1.], [2., 2.], [-2., -2.]])
    before = initial.copy()
    cfg = BatchedQuadraticCenterConfig(centeredness_cap=1e-8)
    locator = BatchedLocalCenterConfig(trust_refinement_rounds=1, jit_compile=False)
    result = initialize_batched_posterior_local_location_scale(callback, initial, [1., 1.], config=cfg, locator_config=locator)
    assert result.accepted, result.status
    np.testing.assert_allclose(result.center, [0.2, -0.3], atol=1e-7)
    assert result.diagnostics["physical_rows"] == 4 * result.diagnostics["callback_batches"]
    np.testing.assert_array_equal(initial, before)
    capped = initialize_batched_posterior_local_location_scale(
        callback, initial, [1., 1.], config=cfg,
        locator_config=replace(locator, max_optimizer_callback_batches_per_round=1))
    assert not capped.accepted and capped.pilot_factor is None
    assert capped.status == "localizer_callback_cap_exhausted"
    assert "rounds" not in capped.diagnostics


def test_target_programming_exception_is_not_hidden():
    def callback(points):
        raise RuntimeError("fixture programming failure")
    with pytest.raises(RuntimeError, match="fixture programming"):
        refine_batched_quadratic_center(callback, [0., 0.], [1., 1.])


def test_localizer_to_refiner_exact_evidence_mismatch_rejects(monkeypatch):
    import bayesfilter.inference.batched_quadratic_center as module
    located = SimpleNamespace(
        accepted=tf.constant(True), center=tf.zeros(2, tf.float64), center_value=tf.constant(1., tf.float64),
        center_score=tf.zeros(2, tf.float64), physical_target_rows=tf.constant(4),
        target_callback_batches=tf.constant(1), selected_evaluation_index=tf.constant(0), payload=dict,
    )
    monkeypatch.setattr(module, "locate_batched_local_center", lambda *args, **kwargs: located)
    result = initialize_batched_posterior_local_location_scale(gaussian([0., 0.], np.eye(2)), np.zeros((4, 2)), [1., 1.])
    assert result.status == "localizer_refinement_replay_mismatch"
    assert not result.accepted and result.pilot_factor is None


@pytest.mark.parametrize("offset", [-1000., 0., 1000.])
def test_additive_log_density_constant_does_not_change_geometry(offset):
    result = refine_batched_quadratic_center(gaussian([0., 0.], np.eye(2), offset=offset), [1., -1.], [1., 1.],
                                             config=BatchedQuadraticCenterConfig(centeredness_cap=1e-7))
    assert result.accepted
    np.testing.assert_allclose(result.center, [0., 0.], atol=1e-7)
    np.testing.assert_allclose(result.pilot_factor, np.eye(2), atol=1e-8)


@pytest.mark.parametrize("defect", ["dtype", "shape"])
def test_malformed_callback_is_programming_error(defect):
    def callback(points):
        if defect == "dtype":
            return tf.zeros([4]), tf.zeros_like(points), tf.ones([4], tf.bool)
        return tf.zeros([4], tf.float64), tf.zeros_like(points), tf.ones([4, 1], tf.bool)
    with pytest.raises((TypeError, ValueError)):
        refine_batched_quadratic_center(callback, [0., 0.], [1., 1.])


def test_rank_deficient_probe_design_rejects(monkeypatch):
    monkeypatch.setattr(tf.random, "stateless_uniform", lambda shape, seed, **kwargs: tf.zeros(shape, tf.float64))
    result = refine_batched_quadratic_center(gaussian([0., 0.], np.eye(2)), [0., 0.], [1., 1.])
    assert not result.accepted and result.pilot_factor is None


def test_runtime_has_no_numpy_or_historical_initializer_dependency():
    import bayesfilter.inference.batched_quadratic_center as module
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    forbidden = ("numpy", "scipy", "filters", "inference.hmc", "inference.mass_matrix", "inference.posterior_adapter")
    assert not any(name.startswith(forbidden) for name in imports)
    assert not any("quadratic_geometry" in name or "posterior_local_initializer" in name for name in imports)

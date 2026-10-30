"""Pinned preparation decisions, immutable results and actual XLA callbacks.

The pinned wrapper shares the already-qualified geometry dependency in these
checks. They do not certify the remaining host-controlled refinement lifecycle.
"""

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_map_covariance as candidate
from tests.test_quadratic_map_covariance import _geometry_config, _quadratic_target

D = tf.float64


@pytest.fixture(scope="module")
def baseline():
    source = subprocess.check_output(["git", "show",
        "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/inference/quadratic_map_covariance.py"], text=True)
    spec = importlib.util.spec_from_loader("quadratic_initializer_pinned_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - frozen diagnostic source
    return module


def _geometry(*, best=None, value=0., center_value=0., precision=None, scale=None):
    return SimpleNamespace(
        center=tf.constant([.2, -.1], D),
        scale=tf.constant([.5, 2.] if scale is None else scale, D),
        precision=tf.constant([[3., .2], [.2, 2.]] if precision is None else precision, D),
        best_evaluated_position=None if best is None else tf.constant(best, D),
        best_evaluated_value=value, best_evaluated_source="center",
        accepted=True, status="usable", diagnostics={"center_log_prob": center_value,
            "config": {"center_log_prob_tolerance": 1e-8}})


@pytest.mark.parametrize("precision,scale", [([[3., .2], [.2, 2.]], [.5, 2.]),
    ([[1., .001], [.003, 4.]], [1e-3, 7.]), ([[2., 0.], [0., 5.]], [1., 1.])])
def test_coordinate_transform_and_eigen_summary_preserve_pinned_values(baseline, precision, scale):
    geometry = _geometry(precision=precision, scale=scale)
    expected = baseline._precision_from_geometry_to_theta(geometry)
    actual = candidate._precision_from_geometry_to_theta(geometry)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert candidate._coordinate_transform_diagnostics(geometry) == baseline._coordinate_transform_diagnostics(geometry)
    report = candidate._eigen_summary(actual)
    reference = baseline._eigen_summary(expected)
    assert report["finite"] == reference["finite"]
    assert report["positive"] == reference["positive"]
    np.testing.assert_allclose(report["eigenvalues"], reference["eigenvalues"], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("best,value", [(None, 1.), ([.2, -.1], 1.),
    ([.3, -.1], 1e-8), ([.3, -.1], np.nextafter(1e-8, np.inf)),
    ([.3, -.1], np.nan), ([.3, -.1], np.inf), ([.2 + 2e-11, -.1], 1.)])
def test_centeredness_preserves_strict_objective_and_movement_gates(baseline, best, value):
    geometry = _geometry(best=best, value=float(value))
    assert candidate._geometry_incumbent_move_is_material(geometry) == baseline._geometry_incumbent_move_is_material(geometry)


@pytest.mark.parametrize("scale", [1., 1.00001001, np.nextafter(1.00001001, np.inf)])
def test_scale_all_ones_retains_original_reporting_threshold(baseline, scale):
    geometry = _geometry(scale=[scale, 1.])
    assert candidate._coordinate_transform_diagnostics(geometry) == baseline._coordinate_transform_diagnostics(geometry)


def test_complete_target_callback_has_one_trace_and_xla_hlo():
    precision = tf.constant([[2., .3], [.3, 4.]], D)

    def target(point):
        score = -tf.linalg.matvec(precision, point)
        return .5 * tf.reduce_sum(point * score), score

    point = tf.constant([.2, -.4], D)
    expected_score = -np.asarray(precision) @ np.asarray(point)
    for argument in (point, point * 2):
        value, score, status = candidate._evaluate_value_score(target, argument, 2)
        assert status == "finite"
        np.testing.assert_allclose(score, expected_score * (1 if argument is point else 2), atol=1e-10, rtol=1e-10)
        assert value == pytest.approx(.5 * float(tf.reduce_sum(argument * score)), abs=1e-10)
    program = candidate._numerical_program(target, (tf.TensorSpec([2], D),))
    assert program.experimental_get_tracing_count() == 1
    assert "HloModule" in program.experimental_get_compiler_ir(point)(stage="hlo")
    graph = program.get_concrete_function().graph.as_graph_def()
    assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in graph.node)


@pytest.mark.parametrize("kind", ["nonfinite", "shape", "exception"])
def test_callback_rejections_keep_pinned_status(baseline, kind):
    def target(point):
        if kind == "exception":
            raise ValueError("fixture")
        return tf.constant(np.nan if kind == "nonfinite" else 1., D), point[:1] if kind == "shape" else point

    expected = baseline._evaluate_value_score(target, np.ones(2), 2)
    actual = candidate._evaluate_value_score(target, tf.ones([2], D), 2)
    assert actual[2] == expected[2]
    np.testing.assert_allclose(actual[1], expected[1], equal_nan=True)


def test_result_snapshots_buffers_and_variables_and_preserves_payload(baseline):
    vector = np.arange(4, dtype=">f8")[::2]
    matrix = tf.Variable([[3., .2], [.2, 2.]], dtype=D)
    args = {"accepted": True, "status": "fixture", "dimension": 2, "initial_position": vector,
        "locator_position": vector, "map_candidate": vector, "map_candidate_role": "fixture",
        "precision": matrix, "covariance": None, "covariance_source": None, "locator_diagnostics": {},
        "geometry": None, "mass_matrix": None, "diagnostics": {"array": vector}}
    result = candidate.QuadraticMapCovarianceResult(**args)
    expected = baseline.QuadraticMapCovarianceResult(**args)
    vector[:] = 9.
    matrix.assign(tf.zeros([2, 2], D))
    assert tf.is_tensor(result.precision)
    np.testing.assert_array_equal(result.initial_position, expected.initial_position)
    np.testing.assert_array_equal(result.precision, expected.precision)
    actual_payload = result.payload(include_arrays=True)
    expected_payload = expected.payload(include_arrays=True)
    actual_summary = actual_payload.pop("precision_eigen_summary")
    expected_summary = expected_payload.pop("precision_eigen_summary")
    assert actual_payload == expected_payload
    np.testing.assert_allclose(actual_summary["eigenvalues"], expected_summary["eigenvalues"], atol=1e-10, rtol=1e-10)
    json.dumps(actual_payload)


def test_real_locator_retains_selected_quadratic_center_with_xla(baseline):
    def target(point):
        delta = point - tf.constant([.4, -.2], D)
        score = -delta * tf.constant([2., 4.], D)
        return .5 * tf.reduce_sum(delta * score), score

    start = tf.constant([1., -1.], D)
    value, score = target(start)
    options = {"value_and_score_fn": target, "initial_value": float(value),
        "config": candidate.QuadraticMapCovarianceLocatorConfig(max_iterations=20)}
    expected, expected_report = baseline._run_locator(initial_position=start.numpy(), initial_score=score.numpy(), **options)
    actual, report = candidate._run_locator(initial_position=start, initial_score=score, **options)
    assert report["jit_compile"] is True
    assert report["accepted_optimizer_position"] == expected_report["accepted_optimizer_position"] is True
    assert report["joint_center_status"] == expected_report["joint_center_status"]
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual, [.4, -.2], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("iterative", [False, True])
def test_existing_initializer_outcomes_against_pinned_wrapper_on_current_clouds(baseline, iterative, record_property):
    """Localize inherited cloud-dependent outcomes without weakening a gate."""
    precision = np.diag([2., 4.])
    mode = np.array([.32, -.24] if iterative else [.18, -.12])
    target = _quadratic_target(precision, mode=mode)
    settings = {"value_and_score_fn": target, "initial_position": np.zeros(2) if iterative else mode,
        "locator_config": candidate.QuadraticMapCovarianceLocatorConfig(enabled=False),
        "quadratic_config": _geometry_config(rank=1, sample_count=220,
            pilot_direction_count=512 if iterative else 256,
            trust_radius=.1 if iterative else 1., pilot_radius=.05 if iterative else .15,
            holdout_rmse_abs_tolerance=7e-2,
            constrain_center_refinement_to_trust_region=iterative,
            seed=(19, 20) if iterative else (17, 18)),
        "mass_config": candidate.QuadraticMapCovarianceMassConfig(jitter=0.,
            eigenvalue_floor=.1, max_condition_number=100.)}
    function = "estimate_iterative_quadratic_map_covariance" if iterative else "estimate_quadratic_map_covariance"
    if iterative:
        settings["iterative_config"] = candidate.IterativeQuadraticMapCovarianceConfig(
            max_refinement_steps=8, terminal_score_max_abs=1e-8)
    expected = getattr(baseline, function)(**settings)
    actual = getattr(candidate, function)(**settings)
    record_property("pinned_wrapper_status", expected.status)
    record_property("candidate_status", actual.status)
    assert actual.status == expected.status
    assert actual.accepted == expected.accepted
    if iterative:
        assert len(actual.iterations) == len(expected.iterations)
        for left, right in zip(actual.iterations, expected.iterations, strict=True):
            assert left["geometry_status"] == right["geometry_status"]
            assert left["terminal_before_fit"] == right["terminal_before_fit"]
            np.testing.assert_allclose(left["center"], right["center"], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(left["center_score_norm"], right["center_score_norm"], atol=1e-10, rtol=1e-10)

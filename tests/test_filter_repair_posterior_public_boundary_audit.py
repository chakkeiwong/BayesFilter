"""Diagnostic observation of the pinned public initializer's derivative boundary."""

import ast
import dataclasses
import hashlib
import subprocess
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_initializer_controller import original_module
from tests.test_filter_repair_posterior_movement import (
    REVISION,
    verified_reference_tree,
)
from tests.test_posterior_local_initializer import _thresholds

D = tf.float64


@pytest.mark.parametrize("case", ["accepted", "initial_invalid"])
def test_original_public_derivative_boundary(case, request):
    public, _ = original_module()
    sources, hashes = verified_reference_tree()
    path = "bayesfilter/inference/posterior_local_initializer.py"
    oldest = subprocess.check_output(["git", "show", f"3582b4ac:{path}"],
        cwd=Path(__file__).resolve().parents[1], text=True)
    names = {"PosteriorLocalInitializerResult", "_build_result", "_vector", "_positive_vector", "_eigen_summary"}
    excerpts = {"oldest": {}, "immediate": {}}
    for label, source in (("oldest", oldest), ("immediate", sources[path])):
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                excerpts[label][node.name] = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])

    calls = tf.Variable(0, dtype=tf.int64)

    def target(point):
        update = calls.assign_add(1)
        with tf.control_dependencies([update]):
            delta = point - .13
            value, score = -tf.reduce_sum(delta ** 2), -2. * delta
        if case == "initial_invalid":
            value = tf.constant(float("nan"), D)
        return value, score

    config = public.PosteriorLocalInitializerConfig(max_movement_attempts=3, max_curvature_attempts=2,
        factor_max=2, locator_config=JointCenterLocatorConfig(max_iterations=10,
            gradient_tolerance=1e-10, max_objective_evaluations=60), seed=(31, 43))
    move = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=12,
        min_samples_per_parameter=1, fit_max_iterations=8, pilot_direction_count=6,
        trust_radius=.3, holdout_fraction=.25, holdout_rmse_abs_tolerance=.1,
        holdout_rmse_rel_tolerance=.1, constrain_center_refinement_to_trust_region=True, seed=(12, 34))
    initial, scale = tf.constant([.13], D), tf.constant([.8], D)
    field_names = ("center", "center_score", "scale", "precision_z", "covariance_z", "precision_theta",
        "covariance_theta", "marginal_standard_deviations", "initial_output_shift", "initial_output_scale_log")
    recorded_error = None
    try:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch((initial, scale))
            result = public.initialize_posterior_local_location_scale(target, initial,
                scale=scale, config=config, movement_config=move, curvature_thresholds=_thresholds(1))
            totals = {name: None if getattr(result, name) is None else tf.reduce_sum(getattr(result, name))
                      for name in field_names}
    except LookupError as error:
        recorded_error = {"type": type(error).__name__, "message": str(error), "target_calls": int(calls)}
        assert "XlaSelfAdjointEig" in str(error)
        calls.assign(0)
        result = public.initialize_posterior_local_location_scale(target, initial,
            scale=scale, config=config, movement_config=move, curvature_thresholds=_thresholds(1))
        totals = {}
    observations = {}
    for name, total in totals.items():
        gradients = (None, None) if total is None else tape.gradient(total, (initial, scale))
        observations[name] = {"present": total is not None,
            "initial_gradient": clean(gradients[0]), "scale_gradient": clean(gradients[1])}
    del tape
    first = clean(result.payload(include_arrays=True))
    first_calls = int(calls)
    calls.assign(0)
    repeated = public.initialize_posterior_local_location_scale(target, initial,
        scale=scale, config=config, movement_config=move, curvature_thresholds=_thresholds(1))
    second = clean(repeated.payload(include_arrays=True))
    repeat_calls = int(calls)
    calls.assign(0)
    frozen_input_error = None
    try:
        with tf.GradientTape(persistent=True) as frozen_tape:
            frozen_tape.watch((initial, scale))
            frozen_result = public.initialize_posterior_local_location_scale(target, tf.stop_gradient(initial),
                scale=tf.stop_gradient(scale), config=config, movement_config=move, curvature_thresholds=_thresholds(1))
            frozen_totals = {name: None if getattr(frozen_result, name) is None else tf.reduce_sum(getattr(frozen_result, name))
                             for name in field_names}
    except LookupError as error:
        frozen_input_error = {"type": type(error).__name__, "message": str(error), "target_calls": int(calls)}
        assert "XlaSelfAdjointEig" in str(error)
        calls.assign(0)
        with tf.GradientTape(persistent=True) as frozen_tape:
            frozen_tape.watch((initial, scale))
            with tf.init_scope():
                frozen_result = public.initialize_posterior_local_location_scale(target, tf.stop_gradient(initial),
                    scale=tf.stop_gradient(scale), config=config, movement_config=move, curvature_thresholds=_thresholds(1))
            frozen_totals = {name: None if getattr(frozen_result, name) is None else tf.reduce_sum(getattr(frozen_result, name))
                             for name in field_names}
    frozen_gradients = {name: (None, None) if total is None else frozen_tape.gradient(total, (initial, scale))
                        for name, total in frozen_totals.items()}
    del frozen_tape
    frozen_payload = clean(frozen_result.payload(include_arrays=True))
    # For a Gaussian of precision 2, the correct physical covariance is 1/2
    # regardless of coordinate units. This is an explanatory analytic check,
    # not a claim that the observed partial derivatives are total derivatives.
    physical_reference = {"covariance": .5, "marginal": 2. ** -.5, "scale_total_derivative": 0.}
    save(request, f"posterior-public-boundary-{case}.json", {
        "classification": "explanatory_public_reference_audit", "revision": REVISION,
        "source_sha256": hashes, "oldest_source_sha256": hashlib.sha256(oldest.encode()).hexdigest(),
        "excerpts": excerpts, "config": dataclasses.asdict(config), "movement_config": dataclasses.asdict(move),
        "result": first, "repeated_result": second, "target_calls": first_calls,
        "repeated_target_calls": repeat_calls, "field_gradients": observations,
        "tape_execution_error": recorded_error, "frozen_input_result": frozen_payload,
        "frozen_input_target_calls": int(calls), "frozen_input_field_gradients": clean(frozen_gradients),
        "frozen_input_error": frozen_input_error,
        "init_scope_reference_diagnostic_needed": frozen_input_error is not None,
        "physical_gaussian_reference": physical_reference, "candidate_promoted": False})
    assert first == second == frozen_payload and first_calls == repeat_calls == int(calls)
    assert all(pair == (None, None) for pair in frozen_gradients.values())
    assert result.accepted is (case == "accepted")

"""Exported posterior-initializer call-chain, records and error boundaries."""

import hashlib
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter import inference
from bayesfilter.inference import posterior_initializer_controller_tf as native
from bayesfilter.inference import posterior_local_initializer as public
from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_curvature_controller import fixture
from tests.test_filter_repair_posterior_initializer_controller import (
    CASES,
    original_module,
)
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_posterior_local_initializer import _thresholds

D = tf.float64


@pytest.fixture(autouse=True)
def release_owner():
    native.clear_posterior_initializer_cache()
    yield
    native.clear_posterior_initializer_cache()


def configurations(dimension, case):
    config = public.PosteriorLocalInitializerConfig(factor_max=2, seed=(31, 43),
        structured_target_family="factor_2" if case == "twofactor" else None,
        max_exact_evaluations=4 if case == "budget" else 5000,
        locator_config=JointCenterLocatorConfig(max_iterations=10,
            gradient_tolerance=1e-10, max_objective_evaluations=60))
    movement = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=12 * dimension,
        min_samples_per_parameter=1, fit_max_iterations=50 if case == "twofactor" else 8, pilot_direction_count=6,
        trust_radius=.3, holdout_fraction=.25, holdout_rmse_abs_tolerance=.1,
        holdout_rmse_rel_tolerance=.1, constrain_center_refinement_to_trust_region=True, seed=(12, 34))
    return config, movement, _thresholds(dimension)


@pytest.mark.parametrize("dimension,batched,case", (*CASES, (5, True, "twofactor")))
def test_exported_complete_reference(dimension, batched, case, monkeypatch, request, jit_compile=True):
    reference, hashes = original_module()
    config, movement, thresholds = configurations(dimension, case)
    config = replace(config, locator_config=replace(config.locator_config, jit_compile=jit_compile))
    precision = None
    if case == "twofactor":
        loadings = tf.constant([[.7, 0.], [.1, .6], [.5, .35], [-.4, .2], [.2, -.5]], D)
        units = tf.constant([.7, .9, 1.1, .8, 1.2], D)
        covariance = units[:, None] * (tf.linalg.diag(1. - tf.reduce_sum(loadings ** 2, axis=1))
            + tf.linalg.matmul(loadings, loadings, transpose_b=True)) * units[None, :]
        precision = tf.linalg.inv(covariance)
    monitor, reset, record, _ = fixture(dimension, batched, case, config, precision=precision)
    kwargs = {"batched_value_and_score_fn": monitor.batched_fn,
        "eligibility_fn": monitor.eligibility_fn, "batched_eligibility_fn": monitor.batched_eligibility_fn,
        "config": config, "movement_config": movement, "curvature_thresholds": thresholds}
    initial, scale = tf.fill([dimension], tf.constant(.13, D)), tf.ones([dimension], D)
    reset()
    with tf.GradientTape() as construction_tape:
        construction_tape.watch((initial, scale))
        owner = native.posterior_initializer_owner(monitor.scalar_fn, dimension, config, movement, thresholds,
            device=initial.device, batched_callback=monitor.batched_fn,
            eligibility_callback=monitor.eligibility_fn, batched_eligibility_callback=monitor.batched_eligibility_fn)
    assert not record()["extents"]
    calls = []
    original_call = native.PreparedPosteriorInitializer.__call__

    def spy(current, *args):
        calls.append(current)
        return original_call(current, *args)

    def forbid_host_summary(_):
        raise AssertionError("native result must report its completed eigen summaries")

    monkeypatch.setattr(native.PreparedPosteriorInitializer, "__call__", spy)
    monkeypatch.setattr(public, "_eigen_summary", forbid_host_summary)
    endpoint = inference.initialize_posterior_local_location_scale
    assert endpoint is public.initialize_posterior_local_location_scale
    observations, hlos = [], []
    for shift in (0., .01, 0.):
        initial = tf.fill([dimension], tf.constant((-.2 if case == "moving" else .13) + shift, D))
        scale = tf.cast(tf.range(dimension), D) * .1 + .8 + shift
        reset()
        expected = reference.initialize_posterior_local_location_scale(monitor.scalar_fn, initial,
            scale=scale, **kwargs).payload(include_arrays=True)
        expected_calls = {k: v for k, v in record().items() if k != "tracker"}
        reset()
        with tf.GradientTape() as tape:
            tape.watch((initial, scale))
            result = endpoint(monitor.scalar_fn, initial, scale=scale, **kwargs)
            total = tf.reduce_sum(result.center + result.center_score + result.scale)
            if result.marginal_standard_deviations is not None:
                total += tf.reduce_sum(result.marginal_standard_deviations)
        assert tape.gradient(total, (initial, scale)) == (None, None)
        actual = result.payload(include_arrays=True)
        actual_calls = {k: v for k, v in record().items() if k != "tracker"}
        observations.append({"expected": clean(expected), "actual": clean(actual),
            "expected_calls": expected_calls, "actual_calls": actual_calls})
        prepared = owner.prepare_clouds()
        assert "CPU" in prepared["curvature_offsets"].device
        operands = (initial, scale, prepared["directions"], prepared["movement_offsets"],
            prepared["permutation_keys"], prepared["curvature_offsets"])
        hlos.append(stable_hlo(owner.compiled.experimental_get_compiler_ir(*operands)(stage="hlo"))
            if jit_compile else owner.compiled.get_concrete_function().graph.as_graph_def().SerializeToString(
                deterministic=True).hex())
    save(request, f"posterior-exported-{dimension}-{batched}-{case}.json", {
        "reference_sha256": hashes, "reference_entire_module": True,
        "comparisons": observations, "trace_count": owner.compiled.experimental_get_tracing_count(),
        "jit_compile": jit_compile,
        "hlo_sha256" if jit_compile else "graph_sha256": [hashlib.sha256(hlo.encode()).hexdigest() for hlo in hlos],
        "exported_owner_invocations": len(calls), "public_installed": True})
    for comparison in observations:
        _equal_records(comparison["actual"], comparison["expected"])
        _equal_records(comparison["actual_calls"], comparison["expected_calls"])
    assert len(calls) == 3 and all(item is owner for item in calls)
    assert observations[0] == observations[2]
    assert owner.compiled.experimental_get_tracing_count() == 1
    assert hlos[0] == hlos[1] == hlos[2]
    if case == "twofactor":
        for comparison in observations:
            curvature = comparison["actual"]["curvature"]
            assert curvature is not None
            fits = [fit for fit in curvature["fits"] if fit["family"] == "factor_2"]
            assert len(fits) == config.replicate_count
            assert all(fit["raw_precision_z"] is not None for fit in fits)


@pytest.mark.parametrize("dimension,batched,case", [(1, True, "stationary"),
    (3, True, "stationary"), (1, False, "moving"), (1, True, "invalid_partial"),
    (1, True, "mismatch")])
def test_exported_graph_reference(dimension, batched, case, monkeypatch, request):
    """Explicit graph diagnostic, including ordered callbacks and changed inputs."""
    test_exported_complete_reference(dimension, batched, case, monkeypatch, request, jit_compile=False)


@pytest.mark.parametrize("initial_invalid", [True, False])
@pytest.mark.parametrize("configuration_error", ["rows", "factor"])
def test_late_validation_precedence(initial_invalid, configuration_error, request):
    reference, hashes = original_module()
    config, movement, thresholds = configurations(1, "stationary")
    config = (replace(config, training_rows_per_replicate=1, selection_rows_per_replicate=1)
        if configuration_error == "rows" else replace(config, max_condition_number=1.))
    monitor, reset, record, _ = fixture(1, True, "invalid_center" if initial_invalid else "stationary",
        replace(config, training_rows_per_replicate=None, selection_rows_per_replicate=None))
    kwargs = {"scale": [.8], "batched_value_and_score_fn": monitor.batched_fn,
        "config": config, "movement_config": movement, "curvature_thresholds": thresholds}

    def observe(endpoint):
        reset()
        try:
            result = {"payload": endpoint(monitor.scalar_fn, [.13], **kwargs).payload(include_arrays=True)}
        except ValueError as error:
            result = {"error_type": type(error).__name__, "message": str(error)}
        return {"result": clean(result), "calls": {k: v for k, v in record().items() if k != "tracker"}}

    expected = observe(reference.initialize_posterior_local_location_scale)
    actual = observe(inference.initialize_posterior_local_location_scale)
    save(request, f"posterior-public-late-validation-{configuration_error}-{initial_invalid}.json", {
        "reference_sha256": hashes, "expected": expected, "actual": actual})
    _equal_records(actual, expected)
    if initial_invalid:
        assert actual["result"]["payload"]["status"] == "initial_target_invalid"
        assert actual["calls"]["extents"] == [1]
    else:
        assert actual["result"]["message"] == ("training plus selection rows must total at least 4N"
            if configuration_error == "rows" else "max_condition_number must exceed one")
        assert len(actual["calls"]["extents"]) > 1

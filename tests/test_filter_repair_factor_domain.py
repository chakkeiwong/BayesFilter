"""Diagnostic-only observation of graph/XLA factor-domain assertions."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import fixed_center_curvature as curvature
from bayesfilter.inference import fixed_center_fitting_tf as fitting
from tests.test_filter_repair_active_cod import _module, _replace
from tests.test_filter_repair_fixed_fitting_localization import _baseline
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_padded_factor import _data, _public_numerics


def _checkpoint(name):
    source = subprocess.check_output(["git", "show",
        "085baaaa:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
    return _module(source, name)


def _guard_candidate():
    """Trial finite-value guards only; do not alter the runtime source."""
    source = subprocess.check_output(["git", "show", "085baaaa:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
    old = "    return deviations[:, None] * correlation * deviations[None, :]"
    new = """    covariance = deviations[:, None] * correlation * deviations[None, :]
    valid = tf.reduce_all(deviations > 0.) & tf.reduce_all(row_norm_squared < 1. - margin)
    return tf.where(valid, covariance, tf.constant(float('nan'), covariance.dtype))"""
    source = _replace(source, old, new)
    return _module(source, "factor_domain_guard_diagnostic")


def test_factor_domain_public_comparison(request):
    compact, _ = _data(5, 1, 32)
    baseline = _baseline()
    checkpoint = _checkpoint("factor_domain_public_checkpoint")
    reports = []
    for label, module, options in (
        ("original_eager", baseline, {}),
        ("checkpoint_graph", checkpoint, {"jit_compile": False}),
        ("checkpoint_xla", checkpoint, {"jit_compile": True}),
    ):
        result = module.fit_factor_correlation_score_geometry(*compact[:5], training_weights=compact[5],
            config=module.FactorCorrelationGeometryConfig(factor_count=2), **options)
        reports.append({"arm": label, "record": result.payload()})
    directory = Path(request.config.getoption("xmlpath")).parent
    report = {"role": "explanatory_domain_failure_semantics", "observations": reports,
        "runtime_changed": False, "full_graph_warm_timing_available": False}
    with (directory / "factor-domain-public-comparison.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("FACTOR_DOMAIN_PUBLIC " + json.dumps(report, sort_keys=True))


def test_factor_domain_constructor_comparison(request):
    baseline = _baseline()
    checkpoint = _checkpoint("factor_domain_constructor_checkpoint")
    margin = 1.e-6
    deviations = tf.ones([3], tf.float64)
    boundary = tf.sqrt(tf.constant(1. - margin, tf.float64))
    cases = {
        "healthy": (deviations, tf.constant([[.2], [-.1], [.3]], tf.float64)),
        "at_margin": (deviations, tf.reshape(tf.stack((boundary, boundary * 0., boundary * .2)), [3, 1])),
        "outside_margin": (deviations, tf.constant([[1.1], [0.], [.3]], tf.float64)),
        "negative_scale": (-deviations, tf.constant([[.2], [-.1], [.3]], tf.float64)),
    }
    signature = [tf.TensorSpec([3], tf.float64), tf.TensorSpec([3, 1], tf.float64)]
    calls = {"original_eager": baseline.factor_correlation_covariance,
        "checkpoint_graph": tf.function(checkpoint.factor_correlation_covariance,
            input_signature=signature, jit_compile=False, autograph=False),
        "checkpoint_xla": tf.function(checkpoint.factor_correlation_covariance,
            input_signature=signature, jit_compile=True, autograph=False)}
    reports = []
    for label, arguments in cases.items():
        for mode, function in calls.items():
            try:
                result = function(*arguments)
                reports.append({"case": label, "mode": mode, "state": "returned",
                    "finite": bool(tf.reduce_all(tf.math.is_finite(result))),
                    "matrix": result.numpy().tolist()})
            except tf.errors.OpError as error:
                reports.append({"case": label, "mode": mode, "state": "rejected",
                    "exception": type(error).__name__, "message": str(error)})
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "factor-domain-constructor-comparison.json").open("x") as handle:
        json.dump({"role": "explanatory_domain_guards", "observations": reports}, handle, indent=2)
        handle.write("\n")
    print("FACTOR_DOMAIN_CONSTRUCTOR " + json.dumps(reports, sort_keys=True))


def test_factor_domain_guard_trial(request):
    candidate = _guard_candidate()
    compact, _ = _data(5, 1, 32)
    cfg = candidate.FactorCorrelationGeometryConfig(factor_count=2)
    result = candidate.fit_factor_correlation_score_geometry(*compact[:5], training_weights=compact[5], config=cfg)
    reports = {"violating_cloud_result": result.payload(), "constructor_checks": []}
    signature = [tf.TensorSpec([3], tf.float64), tf.TensorSpec([3, 1], tf.float64)]

    def observe(module):
        @tf.function(input_signature=signature, jit_compile=True, autograph=False)
        def compute(deviations, loadings):
            with tf.GradientTape() as tape:
                tape.watch((deviations, loadings))
                covariance = module.factor_correlation_covariance(deviations, loadings)
                objective = tf.reduce_sum(tf.sin(covariance))
            gradients = tape.gradient(objective, (deviations, loadings))
            return covariance, *gradients
        return compute

    original_program, guarded_program = observe(_checkpoint("factor_domain_trial_checkpoint")), observe(candidate)
    healthy = (tf.constant([.8, 1.2, 1.4], tf.float64), tf.constant([[.2], [-.1], [.3]], tf.float64))
    before, after = original_program(*healthy), guarded_program(*healthy)
    errors = [float(tf.reduce_max(tf.abs(x-y))) for x,y in zip(before, after, strict=True)]
    reports["healthy_value_pullback_errors"] = errors
    assert max(errors) <= 1e-12
    for label, inputs in (("negative_scale", (-healthy[0], healthy[1])),
                          ("invalid_loading", (healthy[0], tf.constant([[1.1], [0.], [.2]], tf.float64)))):
        values = guarded_program(*inputs)
        finite = bool(tf.reduce_all(tf.math.is_finite(values[0])))
        reports["constructor_checks"].append({"case": label, "finite": finite})
        assert not finite
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "factor-domain-guard-trial.json").open("x") as handle:
        json.dump({"role": "diagnostic_domain_guard_trial", **reports}, handle, indent=2)
        handle.write("\n")
    print("FACTOR_DOMAIN_GUARD " + json.dumps(reports, sort_keys=True))


def test_factor_domain_all_evaluations_trial(request):
    """Observe every covariance evaluation, retaining the original arithmetic."""
    reports = []
    for dimension, factors in ((3, 1), (5, 2)):
        source = subprocess.check_output(["git", "show", "085baaaa:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
        module = _module(source, f"factor_domain_events_{dimension}_diagnostic")
        compact, _ = _data(dimension, 1, 32)
        with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
            violations = tf.Variable(0, trainable=False, dtype=tf.int64)
            evaluations = tf.Variable(0, trainable=False, dtype=tf.int64)
        original = module.factor_correlation_covariance

        def observed(deviations, loadings, *, loading_margin=1.e-6,
                     violations=violations, evaluations=evaluations, original=original):
            valid = tf.reduce_all(deviations > 0.) & tf.reduce_all(
                tf.reduce_sum(tf.square(loadings), axis=1) < 1. - tf.constant(loading_margin, tf.float64))
            update = violations.assign_add(tf.cast(~valid, tf.int64))
            count = evaluations.assign_add(1)
            with tf.control_dependencies([update, count]):
                return original(deviations, loadings, loading_margin=loading_margin)

        module.factor_correlation_covariance = observed
        cfg = module.FactorCorrelationGeometryConfig(factor_count=factors)
        program = module._make_factor_program(dimension, 2*dimension+1, 2*dimension,
            cfg, True, module._prediction_jacobian_diagnostics)

        @tf.function(input_signature=program.input_signature, jit_compile=True, autograph=False)
        def checked(*arguments, violations=violations, evaluations=evaluations, program=program):
            violations.assign(0)
            evaluations.assign(0)
            result = program(*arguments)
            return result, violations.read_value(), evaluations.read_value()

        checkpoint = _checkpoint(f"factor_domain_events_checkpoint_{dimension}")
        before = checkpoint._make_factor_program(dimension, 2*dimension+1, 2*dimension,
            checkpoint.FactorCorrelationGeometryConfig(factor_count=factors), True,
            checkpoint._prediction_jacobian_diagnostics)(*compact)
        after, invalid_count, count = checked(*compact)
        records = [tf.nest.map_structure(lambda x: x.numpy().tolist(), _public_numerics(row)) for row in (before, after)]
        repeated, repeated_invalid, repeated_count = checked(*compact)
        repeat_record = tf.nest.map_structure(lambda x: x.numpy().tolist(), _public_numerics(repeated))
        assert repeated_invalid == invalid_count and repeated_count == count
        assert repeat_record == records[1]
        reports.append({"dimension": dimension, "factors": factors,
            "invalid_evaluations": int(invalid_count), "total_covariance_evaluations": int(count),
            "record_differences": _record_differences(records[1], records[0]), "records": records,
            "trace_count": checked.experimental_get_tracing_count()})
        if dimension == 5:
            scales = tf.constant([.8, 1.1, .9, 1.2, .7], tf.float64)
            loadings = tf.constant([[.3, 0.], [.12, .25], [-.2, .1], [.15, -.1], [.08, .2]], tf.float64)
            covariance = factor.factor_correlation_covariance(scales, loadings)
            precision = tf.linalg.inv(covariance)
            center, points, _, holdout, _, weights = compact
            healthy = (center, points, center[None, :] - points @ precision,
                holdout, center[None, :] - holdout @ precision, weights)
            _, healthy_invalid, healthy_count = checked(*healthy)
            assert int(healthy_invalid) == 0

            def evaluate(arguments):
                result, invalid, count = checked(*arguments)
                return int(invalid), int(count), result["precision"].numpy().tolist()

            expected = [evaluate(healthy), evaluate(compact)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                concurrent = list(pool.map(evaluate, (healthy, compact, compact, healthy)))
            assert concurrent == [expected[0], expected[1], expected[1], expected[0]]
            reports[-1]["concurrent_same_program_records_match"] = True
            reports[-1]["healthy_same_shape_evaluations"] = int(healthy_count)
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "factor-domain-all-evaluations.json").open("x") as handle:
        json.dump({"role": "diagnostic_all_evaluation_domain_observation", "observations": reports}, handle, indent=2)
        handle.write("\n")
    print("FACTOR_DOMAIN_EVENTS " + json.dumps([{k:v for k,v in row.items() if k != "records"}
        for row in reports], sort_keys=True))


@pytest.mark.parametrize("fault", ["at_margin", "outside_margin", "negative_scale", "nan"])
def test_runtime_covariance_guard_rejects_invalid_xla_inputs(fault):
    baseline = _baseline()
    signature = [tf.TensorSpec([3], tf.float64), tf.TensorSpec([3, 1], tf.float64)]

    def program(module):
        @tf.function(input_signature=signature, jit_compile=True, autograph=False)
        def call(scales, loads):
            with tf.GradientTape() as tape:
                tape.watch((scales, loads))
                result = module.factor_correlation_covariance(scales, loads)
                value = tf.reduce_sum(tf.sin(result))
            return result, *tape.gradient(value, (scales, loads))
        return call

    before, after = program(baseline), program(factor)
    healthy = (tf.constant([.8, 1.2, 1.4], tf.float64), tf.constant([[.2], [-.1], [.3]], tf.float64))
    for expected, actual in zip(before(*healthy), after(*healthy), strict=True):
        np.testing.assert_array_equal(actual, expected)
    boundary = tf.sqrt(tf.constant(1. - 1e-6, tf.float64))
    bad_loading = tf.reshape(tf.stack((boundary if fault == "at_margin" else
        tf.constant(float("nan") if fault == "nan" else 1.1, tf.float64),
        tf.constant(.0, tf.float64), tf.constant(.2, tf.float64))), [3, 1])
    arguments = (-healthy[0], healthy[1]) if fault == "negative_scale" else (healthy[0], bad_loading)
    with pytest.raises(tf.errors.InvalidArgumentError):
        factor.factor_correlation_covariance(*arguments)
    assert bool(tf.reduce_all(tf.math.is_nan(after(*arguments)[0])))


def test_runtime_factor_rejects_invalid_trial_and_resets_checked_state():
    compact, _ = _data(5, 1, 32)
    cfg = factor.FactorCorrelationGeometryConfig(factor_count=2)
    module = _baseline()
    before = module.fit_factor_correlation_score_geometry(*compact[:5], training_weights=compact[5],
        config=module.FactorCorrelationGeometryConfig(factor_count=2))
    after = factor.fit_factor_correlation_score_geometry(*compact[:5], training_weights=compact[5], config=cfg)
    assert after.status == before.status == "factor_optimizer_failed"
    assert not before.accepted and not after.accepted
    assert before.covariance_z is None and after.covariance_z is None
    assert after.diagnostics["invalid_covariance_evaluations"] > 0
    center, points, _, holdout, _, weights = compact
    scales = tf.constant([.8, 1.1, .9, 1.2, .7], tf.float64)
    loads = tf.constant([[.3, 0.], [.12, .25], [-.2, .1], [.15, -.1], [.08, .2]], tf.float64)
    precision = tf.linalg.inv(factor.factor_correlation_covariance(scales, loads))
    healthy = (center, points, center[None, :] - points @ precision,
        holdout, center[None, :] - holdout @ precision, weights)
    program = factor._make_factor_program(5, 11, 10, cfg, True, factor._prediction_jacobian_diagnostics)

    def evaluate(arguments):
        result = program(*arguments)
        return (int(result["invalid_covariance_evaluations"]), result["precision"].numpy().tolist())

    expected = [evaluate(healthy), evaluate(compact)]
    assert expected[0][0] == 0 and expected[1][0] > 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(evaluate, (healthy, compact, compact, healthy)))
    assert actual == [expected[0], expected[1], expected[1], expected[0]]
    assert program.experimental_get_tracing_count() == 1


def test_runtime_fixed_fitter_blocks_any_invalid_covariance_evaluation():
    compact, _ = _data(5, 1, 32)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    signature = [tf.TensorSpec([5], tf.float64), tf.TensorSpec([11, 5], tf.float64),
        tf.TensorSpec([11, 5], tf.float64), tf.TensorSpec([10, 5], tf.float64), tf.TensorSpec([10, 5], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def fit(center, training, scores, holdout, holdout_scores):
        return fitting._structured_fit(center, training, scores, holdout, holdout_scores,
            config=config, jit_compile=True)

    result = fit(*compact[:5])
    assert int(result["status"]) == fitting.FIT_STATUSES.index("factor_optimizer_failed")
    assert not bool(tf.reduce_any(result["flags"]))
    assert int(result["invalid_covariance_evaluations"]) > 0
    row = tf.nest.map_structure(lambda value: value.numpy().tolist(), result)
    native_record = curvature._native_fit_record("factor_2", 0, row, 5, 11, 10,
        config.holdout_score_relative_rmse)
    public_record = curvature._fit_structured_precision(*compact[:5], replicate_index=0,
        factor_count=2, max_condition_number=config.max_condition_number,
        holdout_cap=config.holdout_score_relative_rmse)
    print("FACTOR_DOMAIN_NATIVE_RECORD " + json.dumps({"native": native_record.payload(),
        "public": public_record.payload()}, sort_keys=True))
    native_payload, public_payload = native_record.payload(), public_record.payload()
    assert native_payload["diagnostics"].pop("invalid_covariance_evaluations") == int(
        result["invalid_covariance_evaluations"])
    assert public_payload["diagnostics"].pop("invalid_covariance_evaluations") > 0
    # Once the original assertion would have stopped the fit, standalone and
    # enclosing XLA may explore different invalid trajectories. Their counts
    # are new observability, not accepted geometry or a historical fit field.
    assert native_payload == public_payload
    assert native_record.diagnostics["failure_reason"] == "factor_covariance_domain_violation"
    assert native_record.diagnostics["anchor_indices"]

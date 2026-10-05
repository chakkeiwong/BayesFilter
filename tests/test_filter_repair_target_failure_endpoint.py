"""Pinned diagnostic exception semantics and compiled finite/fallback boundary."""

import dataclasses
import gc
import hashlib
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import target_failure_policy as runtime
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint

D = tf.float64


@pytest.fixture(scope="module")
def original():
    checkpoint = FrozenCheckpoint("3582b4ac", "target_failure_boundary")
    return checkpoint, checkpoint.load("bayesfilter.inference.target_failure_policy")


def save(request, name, report):
    path = Path(request.config.getoption("xmlpath")).parent / (name + ".json")
    with path.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


def record(module, evaluation):
    return {"payload": evaluation.payload(), "score": np.asarray(evaluation.score).tolist(),
        "classification": module.classify_target_failure_mode(evaluation).payload()}


@pytest.mark.parametrize("shape", [(), (3,), (2, 2), (0,)])
@pytest.mark.parametrize("case", ["valid", "value_nan", "score_inf", "both_nonfinite",
    "PriorSupportError", "StationarityError", "CovariancePositiveDefiniteError",
    "FactorizationFailure", "SolveResidualError"])
def test_complete_target_failure_records(original, request, shape, case):
    checkpoint, previous = original
    position = np.full(shape, .25, dtype=np.float64)
    reports = []
    for module in (previous, runtime):
        policy = module.TargetFailurePolicy("frozen_target", fallback_log_prob=-123.5,
            fallback_gradient_value=.125)
        calls = []

        def callback(point, calls=calls, module=module):
            calls.append(point is position)
            if case.endswith("Error") or case == "FactorizationFailure":
                raise getattr(module, case)("declared", details={"coordinate": "rho", "index": 1})
            value, score = -.25, np.full(shape, -.5, dtype=np.float64)
            if case in ("value_nan", "both_nonfinite"):
                value = float("nan")
            if case in ("score_inf", "both_nonfinite"):
                score = np.full(shape, float("inf"), dtype=np.float64)
            return value, score

        evaluation = module.evaluate_target_with_failure_policy(callback, position, policy)
        reports.append(record(module, evaluation))
        assert calls == [True]
        np.testing.assert_array_equal(policy.fallback_score(position), np.full(shape, .125))
    save(request, f"target-failure-{case}-{len(shape)}-{position.size}", {
        "original": reports[0], "actual": reports[1], "original_sources": checkpoint.hashes()})
    assert reports[0] == reports[1]


@pytest.mark.parametrize("case", ["runtime", "value_error", "tf_error", "scalar_shape",
    "score_shape", "disabled_catch", "forbidden_nonfinite", "forbidden_failure",
    "forbidden_branch", "unknown_failure", "malformed_return"])
def test_target_failure_preserves_exact_errors(original, request, case):
    checkpoint, previous = original
    reports = []
    for module in (previous, runtime):
        options = {}
        if case == "disabled_catch":
            options["catch_nonfinite_output"] = False
        if case in ("forbidden_nonfinite", "forbidden_failure"):
            options["allowed_failure_labels"] = ("stationarity",)
        if case == "forbidden_branch":
            options["allowed_branch_labels"] = ("valid",)
        policy = module.TargetFailurePolicy("errors", **options)
        calls = []

        def callback(point, calls=calls, module=module):
            calls.append(1)
            if case == "runtime":
                raise RuntimeError("programmer bug")
            if case == "value_error":
                raise ValueError("programmer value bug")
            if case == "tf_error":
                raise tf.errors.InvalidArgumentError(None, None, "tensor shape bug")
            if case in ("forbidden_failure", "forbidden_branch"):
                raise module.PriorSupportError("declared")
            if case == "unknown_failure":
                error = module.TargetRegionError("custom label")
                error.failure_label = "unregistered"
                raise error
            if case == "malformed_return":
                return (0.,)
            if case == "scalar_shape":
                return [0.], point
            if case == "score_shape":
                return 0., [1.]
            return float("nan"), point

        with pytest.raises(Exception) as caught:
            module.evaluate_target_with_failure_policy(callback, [1., 2.], policy)
        assert calls == [1]
        reports.append({"exception": type(caught.value).__name__, "message": str(caught.value)})
    save(request, f"target-error-{case}", {"original": reports[0], "actual": reports[1],
        "original_sources": checkpoint.hashes()})
    assert reports[0] == reports[1]


def test_runtime_python_exception_and_callback_lifetime(original, request):
    """A callback is not traced or retained by the numerical output cache."""
    checkpoint, previous = original
    reports = []
    for module in (previous, runtime):
        policy = module.TargetFailurePolicy("dynamic", allowed_branch_labels=("fallback_prior_support",))
        calls = []

        def callback(point, calls=calls, module=module):
            calls.append(float(point[0]))
            if point[0] < 0.:
                raise module.PriorSupportError("runtime support")
            return .5, point

        rows = [record(module, module.evaluate_target_with_failure_policy(callback, [x, .5], policy))
            for x in (1., -1., 2.)]
        reference = weakref.ref(callback)
        del callback
        gc.collect()
        assert reference() is None
        assert calls == [1., -1., 2.]
        reports.append(rows)
    assert reports[0] == reports[1]
    save(request, "target-runtime-exception", {"original": reports[0], "actual": reports[1],
        "callback_collected": True, "original_sources": checkpoint.hashes()})


def test_output_operands_and_enclosing_derivatives(request):
    program = runtime._target_output_program((3,))
    reports, hlos = [], []
    for value, declared, fallback, gradient in ((.5, False, -123., .125),
        (float("nan"), False, -77., .25), (.5, True, -7., -.5), (.5, False, -123., .125)):
        args = (tf.constant(value, D), tf.constant([.1, -.3, .5], D),
            tf.constant(declared), tf.constant(fallback, D), tf.constant(gradient, D))
        result = program(*args)
        used = declared or not np.isfinite(value)
        assert bool(result["fallback_used"]) == used
        assert float(result["value"]) == (fallback if used else value)
        np.testing.assert_array_equal(result["score"], np.full(3, gradient) if used else args[1])
        reports.append({key: item.numpy().tolist() for key, item in result.items()})
        hlos.append(program.experimental_get_compiler_ir(*args)(stage="hlo"))
    assert program.experimental_get_tracing_count() == 1
    assert len(set(hlos)) == 1 and program.function_spec.jit_compile
    graph = program.get_concrete_function().graph.as_graph_def()
    operations = {node.op for nodes in (graph.node, *(f.node_def for f in graph.library.function))
        for node in nodes}
    assert not operations & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}

    @tf.function(input_signature=[tf.TensorSpec([3], D)], autograph=False, jit_compile=True)
    def enclosing(point):
        with tf.GradientTape() as tape:
            tape.watch(point)
            value = -.5 * tf.reduce_sum(point ** 2)
            result = program(value, -point, point[0] < 0., tf.constant(-7., D), tf.constant(.125, D))
            scalar = result["value"] + tf.reduce_sum(result["score"] ** 2)
        return result, tape.gradient(scalar, point)

    enclosing_hlos = []
    for point in ([.25, -.5, .75], [-.25, -.5, .75], [.5, -.75, 1.]):
        point = tf.constant(point, D)
        result, derivative = enclosing(point)
        np.testing.assert_array_equal(derivative, tf.zeros_like(point) if point[0] < 0 else point)
        assert bool(result["fallback_used"]) == bool(point[0] < 0)
        enclosing_hlos.append(enclosing.experimental_get_compiler_ir(point)(stage="hlo"))
    assert enclosing.experimental_get_tracing_count() == 1
    assert len(set(enclosing_hlos)) == 1
    assert runtime._target_output_program.cache_info().maxsize == 16
    save(request, "target-output-hlo", {"records": reports, "trace_count": 1,
        "enclosing_trace_count": 1, "hlo_sha256": hashlib.sha256(hlos[0].encode()).hexdigest(),
        "enclosing_hlo_sha256": hashlib.sha256(enclosing_hlos[0].encode()).hexdigest(),
        "cache": runtime._target_output_program.cache_info()._asdict()})


def test_policy_operands_do_not_capture_instance():
    policy = runtime.TargetFailurePolicy("first", fallback_log_prob=-11., fallback_gradient_value=.125)
    other = dataclasses.replace(policy, target_scope="second", fallback_log_prob=-22., fallback_gradient_value=.25)
    position = tf.constant([1., 2., 3.], D)
    np.testing.assert_array_equal(policy.fallback_score(position), [.125] * 3)
    np.testing.assert_array_equal(other.fallback_score(position), [.25] * 3)
    reference = weakref.ref(policy)
    del policy
    gc.collect()
    assert reference() is None

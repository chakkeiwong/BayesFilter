"""Independent diagnostic layout candidate; no runtime policy change."""

import hashlib
import inspect
import os
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import tensorflow as tf

from bayesfilter.highdim import dual_cap_genut_primal_tf as current
from tests.test_filter_repair_genut_transitive import _owner, _write


def _layout_candidate():
    source = inspect.getsource(current)
    replacements = {
        "weights[:, None, None] * centered[:, :, None] * centered[:, None, :], axis=0":
        "weights[None, None, :] * tf.transpose(centered)[:, None, :] * tf.transpose(centered)[None, :, :], axis=2",
        "centered[:, :, None] * centered[:, None, :], axis=0":
        "tf.transpose(centered)[:, None, :] * tf.transpose(centered)[None, :, :], axis=2",
        "weights[:, None, None] * squared[:, :, None] * standardized[:, None, :], axis=0":
        "weights[None, None, :] * tf.transpose(squared)[:, None, :] * tf.transpose(standardized)[None, :, :], axis=2",
        "weights[:, None, None] * squared[:, :, None] * squared[:, None, :], axis=0":
        "weights[None, None, :] * tf.transpose(squared)[:, None, :] * tf.transpose(squared)[None, :, :], axis=2",
        "standardized[:, :, None] * direction[:, None, :], axis=0":
        "tf.transpose(standardized)[:, None, :] * tf.transpose(direction)[None, :, :], axis=2",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    module = ModuleType("_genut_last_particle_axis_diagnostic")
    exec(compile(source, "<genut-last-particle-axis-diagnostic>", "exec"), module.__dict__)  # noqa: S102
    return module, source


def _comparison(actual, expected):
    return {key: {"passed": bool(np.allclose(actual[key], value, atol=2e-5, rtol=2e-5)),
                  "max_abs_error": float(np.max(np.abs(np.asarray(actual[key], np.float64)-np.asarray(value, np.float64))))}
            for key, value in expected.items()}


def _call(owner, inputs):
    return {key: value.numpy() for key, value in owner(*inputs).items()}


def test_layout_attribution(request):
    assert os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "cpu"
    artifact_root = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    output = Path(request.config.getoption("xmlpath")).parent
    candidate, source = _layout_candidate()
    (output / "genut-layout-diagnostic.py").write_text(source)
    report = {"role": "diagnostic_layout_candidate_no_runtime_change", "cases": [],
              "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    try:
        for count, dimension, input_run, reference_run in ((1000, 3, 4114, 4135), (10000, 18, 4122, 4137)):
            saved_path = artifact_root / f"run-{input_run:05d}/genut-capacity-arrays.npz"
            with np.load(saved_path, allow_pickle=False) as saved:
                inputs = tuple(tf.constant(saved[key], tf.float32) for key in ("source", "weights", "reset"))
            with np.load(artifact_root / f"run-{reference_run:05d}/genut-capacity-fp64-arrays.npz", allow_pickle=False) as saved:
                fp64 = dict(saved)
            case = {"count": count, "dimension": dimension, "modes": [], "timings": [],
                    "input_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest()}
            report["cases"].append(case)
            for jit in (False, True):
                owners = {"current": _owner(current.dual_cap_genut_primal, inputs, jit),
                          "layout": _owner(candidate.dual_cap_genut_primal, inputs, jit)}
                results = {name: _call(owner, inputs) for name, owner in owners.items()}
                record = {"jit_compile": jit, "same_mode": _comparison(results["layout"], results["current"]),
                          "layout_vs_fp64": _comparison(results["layout"], fp64), "results": results}
                case["modes"].append(record)
                if jit:
                    for repeat in range(12):
                        order = ("current", "layout") if repeat % 2 == 0 else ("layout", "current")
                        timings = {}
                        for name in order:
                            started = time.perf_counter()
                            replay = _call(owners[name], inputs)
                            timings[name] = time.perf_counter() - started
                            assert all(np.array_equal(value, replay[key]) for key, value in results[name].items())
                        case["timings"].append({"order": order, "seconds": timings})
                    for name, owner in owners.items():
                        assert owner.experimental_get_tracing_count() == 1
                        (output / f"genut-layout-{count}-{dimension}-{name}.hlo").write_text(
                            owner.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo"))
                else:
                    for name, owner in owners.items():
                        replay = _call(owner, inputs)
                        assert all(np.array_equal(value, replay[key]) for key, value in results[name].items())
    finally:
        _write(request, "genut-layout.json", report)
    assert all(field["passed"] for case in report["cases"] for mode in case["modes"]
               for field in mode["same_mode"].values()), "Layout candidate changes complete same-mode record"
    assert all(field["passed"] for case in report["cases"] for mode in case["modes"]
               for key, field in mode["layout_vs_fp64"].items() if key != "fraction_coordinatewise_cap_active")


def _highest_dot_candidate():
    from tensorflow.compiler.tf2xla.ops import gen_xla_ops
    from tensorflow.compiler.xla import xla_data_pb2

    def highest_dot(left, right, left_axis, right_axis):
        dimensions = xla_data_pb2.DotDimensionNumbers(
            lhs_contracting_dimensions=[left_axis], rhs_contracting_dimensions=[right_axis])
        precision = xla_data_pb2.PrecisionConfig(operand_precision=[
            xla_data_pb2.PrecisionConfig.HIGHEST, xla_data_pb2.PrecisionConfig.HIGHEST])
        result = gen_xla_ops.xla_dot_v2(left, right,
            dimension_numbers=dimensions.SerializeToString(), precision_config=precision.SerializeToString(),
            preferred_element_type=left.dtype)
        return tf.ensure_shape(result, [left.shape[1-left_axis], right.shape[1-right_axis]])

    source = inspect.getsource(current)
    replacements = {
        "tf.reduce_sum(\n        weights[:, None, None] * centered[:, :, None] * centered[:, None, :], axis=0\n    )":
        "highest_dot(weights[:, None] * centered, centered, 0, 0)",
        "tf.reduce_sum(\n        centered[:, :, None] * centered[:, None, :], axis=0\n    )":
        "highest_dot(centered, centered, 0, 0)",
        "tf.reduce_sum(\n        weights[:, None, None] * squared[:, :, None] * standardized[:, None, :], axis=0\n    )":
        "highest_dot(weights[:, None] * squared, standardized, 0, 0)",
        "tf.reduce_sum(\n        weights[:, None, None] * squared[:, :, None] * squared[:, None, :], axis=0\n    )":
        "highest_dot(weights[:, None] * squared, squared, 0, 0)",
        "tf.reduce_sum(standardized[:, None, :] * residual3[None, :, :], axis=2)":
        "highest_dot(standardized, residual3, 1, 1)",
        "tf.reduce_sum(squared[:, :, None] * residual3[None, :, :], axis=1)":
        "highest_dot(squared, residual3, 1, 0)",
        "tf.reduce_sum(squared[:, None, :] * residual4[None, :, :], axis=2)":
        "highest_dot(squared, residual4, 1, 1)",
        "tf.reduce_mean(\n        standardized[:, :, None] * direction[:, None, :], axis=0\n    )":
        "highest_dot(standardized, direction, 0, 0) / tf.cast(count, standardized.dtype)",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    module = ModuleType("_genut_highest_dot_diagnostic")
    module.highest_dot = highest_dot
    exec(compile(source, "<genut-highest-dot-diagnostic>", "exec"), module.__dict__)  # noqa: S102
    return module, source, highest_dot


def test_highest_dot_attribution(request):
    artifact_root = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    output = Path(request.config.getoption("xmlpath")).parent
    candidate, source, highest_dot = _highest_dot_candidate()
    (output / "genut-highest-dot-diagnostic.py").write_text(source)
    report = {"role": "diagnostic_highest_dot_candidate_no_runtime_change", "cases": [],
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "tf32": tf.config.experimental.tensor_float_32_execution_enabled()}
    try:
        for count, dimension, input_run, reference_run in ((1000, 3, 4114, 4135), (10000, 18, 4122, 4137)):
            saved_path = artifact_root / f"run-{input_run:05d}/genut-capacity-arrays.npz"
            with np.load(saved_path, allow_pickle=False) as saved, tf.device("/CPU:0"):
                inputs = tuple(tf.constant(saved[key], tf.float32) for key in ("source", "weights", "reset"))
            with np.load(artifact_root / f"run-{reference_run:05d}/genut-capacity-fp64-arrays.npz", allow_pickle=False) as saved:
                fp64 = dict(saved)
            owners = {"current": _owner(current.dual_cap_genut_primal, inputs, True),
                      "highest_dot": _owner(candidate.dual_cap_genut_primal, inputs, True)}
            results = {name: _call(owner, inputs) for name, owner in owners.items()}
            case = {"count": count, "dimension": dimension, "timings": [], "results": results,
                    "input_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                    "same_mode": _comparison(results["highest_dot"], results["current"]),
                    "highest_dot_vs_fp64": _comparison(results["highest_dot"], fp64)}
            report["cases"].append(case)
            for repeat in range(12):
                order = ("current", "highest_dot") if repeat % 2 == 0 else ("highest_dot", "current")
                timings = {}
                for name in order:
                    started = time.perf_counter()
                    replay = _call(owners[name], inputs)
                    timings[name] = time.perf_counter() - started
                    assert all(np.array_equal(value, replay[key]) for key, value in results[name].items())
                case["timings"].append({"order": order, "seconds": timings})
            for name, owner in owners.items():
                assert owner.experimental_get_tracing_count() == 1
                (output / f"genut-highest-dot-{count}-{dimension}-{name}.hlo").write_text(
                    owner.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo"))
        # Raw-op derivative support is an adoption concern even for a primal
        # kernel. Record it explicitly without installing a replacement score.
        x = tf.ones([3, 2], tf.float32)
        try:
            with tf.GradientTape() as tape:
                tape.watch(x)
                value = _owner(lambda a: highest_dot(a, a, 0, 0), (x,), True)(x)
                loss = tf.reduce_sum(value)
            derivative = tape.gradient(loss, x)
            report["raw_dot_derivative"] = {"available": derivative is not None}
        except (LookupError, ValueError, NotImplementedError) as error:
            report["raw_dot_derivative"] = {"available": False, "error": str(error)}
    finally:
        _write(request, "genut-highest-dot.json", report)
    assert all(field["passed"] for case in report["cases"] for field in case["same_mode"].values())
    assert all(field["passed"] for case in report["cases"]
               for key, field in case["highest_dot_vs_fp64"].items() if key != "fraction_coordinatewise_cap_active")

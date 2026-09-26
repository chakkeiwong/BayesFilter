"""Diagnostic-only cap-rounding localization; no numerical policy waiver."""

import ast
import hashlib
import inspect
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.highdim import dual_cap_genut_primal_tf as candidate
from tests.test_filter_repair_genut_transitive import _fixture, _frozen, _owner, _write


def _instrument(module):
    source = inspect.getsource(candidate.dual_cap_genut_primal) if module is candidate else module._diagnostic_source
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "dual_cap_genut_primal")
    returned = function.body[-1].value
    assert isinstance(returned, ast.Dict)
    captures = {
        "pre_cap": "pre_coordinate_cap", "scaled_power": "scaled_power",
        "denominator": "denominator", "capped": "capped", "cap": "cap",
        "power": "power", "displacement": "tf.abs(capped - pre_coordinate_cap)",
        "active": "tf.abs(capped - pre_coordinate_cap) > 1.0e-7",
        "threshold": "tf.cast(1.0e-7, source.dtype)",
    }
    for key, expression in captures.items():
        returned.keys.append(ast.Constant("diagnostic_" + key))
        returned.values.append(ast.parse(expression, mode="eval").body)
    ast.fix_missing_locations(tree)
    emitted = ast.unparse(tree)
    namespace = dict(module.__dict__)
    exec(compile(emitted, "<diagnostic-cap-capture>", "exec"), namespace)  # noqa: S102
    return namespace["dual_cap_genut_primal"], emitted


def _comparison(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return {"exact": bool(np.array_equal(a, b)),
            "max_abs_error": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
            "within_existing_bound": bool(np.allclose(a, b, atol=2e-5, rtol=2e-5))}


def _cap_formula(points, cap, power):
    scaled_power = tf.pow(points / cap, power)
    denominator = tf.pow(1.0 + scaled_power, 1.0 / power)
    capped = points / denominator
    delta = tf.abs(capped - points)
    return {"denominator": denominator, "capped": capped, "displacement": delta,
            "active": delta > 1.0e-7}


def _cap_division(points, denominator):
    capped = points / denominator
    delta = tf.abs(capped - points)
    return {"capped": capped, "displacement": delta, "active": delta > 1.0e-7}


def _decimal_rows(capture, indices):
    result = []
    with localcontext() as context:
        context.prec = 80
        cap = Decimal.from_float(float(capture["diagnostic_cap"]))
        power = Decimal.from_float(float(capture["diagnostic_power"]))
        threshold = Decimal.from_float(float(capture["diagnostic_threshold"]))
        for index in indices:
            value = float(capture["diagnostic_pre_cap"][index])
            x = Decimal.from_float(value)
            rounded_capped = Decimal.from_float(float(capture["diagnostic_capped"][index]))
            denominator = Decimal.from_float(float(capture["diagnostic_denominator"][index]))
            exact_cap = x / ((Decimal(1) + (x / cap) ** power) ** (Decimal(1) / power))
            analytic_delta = abs(exact_cap - x)
            result.append({"index": list(index), "pre_cap": value,
                "fp32_spacing": float(abs(np.spacing(np.float32(value)))),
                "smooth_cap_displacement_decimal": str(analytic_delta),
                "smooth_cap_active": analytic_delta > threshold,
                "rounded_operand_displacement_decimal": str(abs(rounded_capped-x)),
                "division_from_rounded_denominator_displacement_decimal": str(abs(x/denominator-x)),
                "threshold_decimal": str(threshold),
                "captured_displacement": float(capture["diagnostic_displacement"][index]),
                "captured_active": bool(capture["diagnostic_active"][index])})
    return result


def test_cap_operand_localization(request):
    original, reference_sha = _frozen("dual_cap_genut_primal_tf")
    # _frozen has already loaded exact Git bytes. Retrieve them for a diagnostic
    # AST copy; the production module and reference callable remain unchanged.
    import subprocess

    from tests.test_filter_repair_genut_transitive import BASELINE, ROOT
    original._diagnostic_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:bayesfilter/highdim/dual_cap_genut_primal_tf.py"],
        cwd=ROOT, text=True)
    inputs = _fixture(tf.float32)
    output = Path(request.config.getoption("xmlpath")).parent
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    report = {"role": "diagnostic_only_no_equivalence_gate_waiver", "reference_sha256": reference_sha,
              "inputs": inputs, "arms": [], "tf32": tf.config.experimental.tensor_float_32_execution_enabled()}
    try:
        for name, module in (("original", original), ("candidate", candidate)):
            instrumented, emitted = _instrument(module)
            (output / f"genut-cap-instrumented-{name}.py").write_text(emitted + "\n")
            for jit in (False, True):
                if gpu and name == "original" and jit:
                    continue
                plain_owner = _owner(module.dual_cap_genut_primal, inputs, jit)
                capture_owner = _owner(instrumented, inputs, jit)
                plain, capture = plain_owner(*inputs), capture_owner(*inputs)
                comparisons = {key: _comparison(value, capture[key]) for key, value in plain.items()}
                row = {"implementation": name, "jit_compile": jit, "plain": plain,
                       "capture": capture, "instrumentation_comparison": comparisons,
                       "instrumentation_exact": all(item["exact"] for item in comparisons.values()),
                       "instrumented_sha256": hashlib.sha256((emitted + "\n").encode()).hexdigest(),
                       "replay": []}
                report["arms"].append(row)
                for mode in (False, True):
                    formula_inputs = (capture["diagnostic_pre_cap"], capture["diagnostic_cap"],
                                      capture["diagnostic_power"])
                    division_inputs = (capture["diagnostic_pre_cap"], capture["diagnostic_denominator"])
                    formula = _owner(_cap_formula, formula_inputs, mode)(*formula_inputs)
                    division = _owner(_cap_division, division_inputs, mode)(*division_inputs)
                    row["replay"].append({"jit_compile": mode, "formula": formula, "division": division})
                mask = capture["diagnostic_active"].numpy()
                indices = set()
                for replay in row["replay"]:
                    for stage in ("formula", "division"):
                        indices.update(map(tuple, np.argwhere(mask != replay[stage]["active"].numpy())))
                delta = capture["diagnostic_displacement"].numpy()
                indices.update(map(tuple, np.argwhere((delta >= .5e-7) & (delta <= 1.5e-7))))
                row["decimal_near_threshold"] = _decimal_rows(capture, sorted(indices))
                assert all(_comparison(value, plain_owner(*inputs)[key])["exact"]
                           for key, value in plain.items())
                assert plain_owner.experimental_get_tracing_count() == 1
                if jit:
                    for label, owner in (("plain", plain_owner), ("capture", capture_owner)):
                        (output / f"genut-cap-{name}-{label}.hlo").write_text(
                            owner.experimental_get_compiler_ir(*inputs)(stage="hlo"))
    finally:
        _write(request, "genut-cap-localization.json", report)
    assert all(item["instrumentation_exact"] for item in report["arms"]), (
        "Instrumentation changes untouched outputs; captured operands cannot establish the untouched mechanism")


def test_cap_lowering_same_operands(request):
    """An optimization barrier is a mechanism probe, not a runtime repair."""
    import json

    from tensorflow.compiler.tf2xla.ops import gen_xla_ops

    from tests.test_filter_repair_genut_transitive import ROOT

    artifact_root = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    saved_path = artifact_root / "run-04111/genut-cap-localization.json"
    saved = json.loads(saved_path.read_text())
    arm = next(item for item in saved["arms"] if item["implementation"] == "candidate" and not item["jit_compile"])
    assert arm["instrumentation_exact"]
    cap = arm["capture"]
    with tf.device("/CPU:0"):
        inputs = tuple(tf.constant(cap["diagnostic_" + key], tf.float32) for key in ("pre_cap", "cap", "power"))

    def barrier_formula(points, cap, power):
        scaled_power = tf.pow(points / cap, power)
        denominator = tf.pow(1.0 + scaled_power, 1.0 / power)
        denominator, = gen_xla_ops.xla_optimization_barrier([denominator])
        capped = points / denominator
        delta = tf.abs(capped - points)
        return {"denominator": denominator, "capped": capped, "displacement": delta,
                "active": delta > 1.0e-7}

    output = Path(request.config.getoption("xmlpath")).parent
    report = {"role": "diagnostic_lowering_mechanism_no_runtime_change", "saved_capture": str(saved_path),
              "saved_capture_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
              "diagnostic_source_sha256": hashlib.sha256((ROOT / __file__).read_bytes()).hexdigest(),
              "inputs": inputs, "arms": []}
    try:
        for name, function, jit in (("graph", _cap_formula, False), ("xla", _cap_formula, True),
                                     ("xla_barrier", barrier_formula, True)):
            owner = _owner(function, inputs, jit)
            result = owner(*inputs)
            report["arms"].append({"name": name, "result": result})
            if jit:
                (output / f"genut-cap-{name}-optimized.hlo").write_text(
                    owner.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo"))
        graph, xla, barrier = (item["result"] for item in report["arms"])
        report["xla_vs_graph"] = {key: _comparison(value, xla[key]) for key, value in graph.items()}
        report["barrier_vs_graph"] = {key: _comparison(value, barrier[key]) for key, value in graph.items()}
        assert not report["xla_vs_graph"]["active"]["exact"]
        assert report["barrier_vs_graph"]["active"]["exact"]
    finally:
        _write(request, "genut-cap-lowering.json", report)

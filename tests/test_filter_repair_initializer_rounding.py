"""Independent diagnostic of the fixed fitter's weighted initialization.

Decimal arithmetic is a tiny full-rank reference only. Residual correction is
injected only in this test process; no runtime algorithm or tolerance changes.
"""

import importlib.util
import json
import subprocess
import sys
from decimal import Decimal, localcontext

import numpy as np
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference import fixed_center_fitting_tf as native
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq
from tests.test_filter_repair_fixed_fitting import _inputs, _thresholds
from tests.test_filter_repair_fixed_fitting_localization import _baseline

D = tf.float64


def _decimal_lstsq(matrix, rhs, precision):
    # High-precision normal equations are independent of both COD programs.
    # This reference is restricted below to a tiny well-conditioned full-rank
    # fixture; it is not a proposal to form normal equations in runtime.
    with localcontext() as context:
        context.prec = precision
        a = [[Decimal.from_float(float(x)) for x in row] for row in matrix]
        b = [[Decimal.from_float(float(x)) for x in row] for row in rhs]
        n, p = len(a[0]), len(b[0])
        system = [[sum(row[i] * row[j] for row in a) for j in range(n)]
            + [sum(x[i] * y[j] for x, y in zip(a, b, strict=True)) for j in range(p)]
            for i in range(n)]
        for k in range(n):
            pivot = max(range(k, n), key=lambda i: abs(system[i][k]))
            system[k], system[pivot] = system[pivot], system[k]
            divisor = system[k][k]
            assert divisor != 0
            system[k] = [value / divisor for value in system[k]]
            for i in range(n):
                if i != k:
                    multiplier = system[i][k]
                    system[i] = [x - multiplier * y for x, y in zip(system[i], system[k], strict=True)]
        return [row[n:] for row in system]


def _decimal_error(left, right):
    with localcontext() as context:
        context.prec = 90
        return max(abs(x - y) for a, b in zip(left, right, strict=True)
            for x, y in zip(a, b, strict=True))


def _corrected(matrix, rhs):
    initial = complete_orthogonal_lstsq(matrix, rhs)
    return initial + complete_orthogonal_lstsq(matrix, rhs - matrix @ initial)


def _record_differences(left, right, path="result"):
    if isinstance(right, dict):
        return [entry for key in right for entry in _record_differences(left[key], right[key], path + "." + key)]
    if isinstance(right, (tuple, list)):
        return [entry for i, (x, y) in enumerate(zip(left, right, strict=True))
            for entry in _record_differences(x, y, path + f"[{i}]")]
    if isinstance(right, (str, bool, int)) or right is None:
        agrees = left == right
    else:
        agrees = np.isclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True)
    return [] if agrees else [{"path": path, "candidate": left, "baseline": right}]


def test_weighted_initializer_rounding_and_unchanged_fit_records(monkeypatch):
    inputs = _inputs(3)
    center, training, scores = (tf.constant(x, D) for x in inputs[1:4])
    original_factor = _baseline()
    signature = [tf.TensorSpec([9, 3], D), tf.TensorSpec([9, 3], D)]
    solvers = {"eigen_cod": lambda a, b: tf.linalg.lstsq(a, b, fast=False)}
    for label, solver in (("native", complete_orthogonal_lstsq), ("one_correction", _corrected)):
        for jit in (False, True):
            solvers[f"{label}_{'xla' if jit else 'graph'}"] = tf.function(
                solver, input_signature=signature, jit_compile=jit, autograph=False)

    report = {"role": "explanatory_initializer_rounding_diagnostic_only",
        "runtime_modified": False, "record_atol": 1e-10, "record_rtol": 1e-10,
        "initializer": []}
    for index in range(2):
        weights = tf.fill([9], tf.constant(1 / 9, D))
        weights /= tf.reduce_sum(weights)
        matrix = training[index] * tf.sqrt(weights)[:, None]
        rhs = (center[None, :] - scores[index]) * tf.sqrt(weights)[:, None]
        reference = _decimal_lstsq(matrix.numpy(), rhs.numpy(), 90)
        repeat = _decimal_lstsq(matrix.numpy(), rhs.numpy(), 60)
        decimal_agreement = _decimal_error(reference, repeat)
        assert decimal_agreement < Decimal("1e-50")
        singular = tf.linalg.svd(matrix, compute_uv=False)
        assert float(singular[-1] / singular[0]) > .01
        row = {"replicate": index, "condition": float(singular[0] / singular[-1]),
            "decimal_precision_agreement": str(decimal_agreement), "solutions": {}}
        for label, solver in solvers.items():
            value = solver(matrix, rhs)
            decimals = [[Decimal.from_float(float(x)) for x in r] for r in value.numpy()]
            residual = rhs - matrix @ value
            row["solutions"][label] = {"max_decimal_solution_error": str(_decimal_error(decimals, reference)),
                "normal_residual_max": float(tf.reduce_max(tf.abs(tf.transpose(matrix) @ residual))),
                "solution": value.numpy().tolist()}
            assert bool(tf.reduce_all(tf.math.is_finite(value)))
        report["initializer"].append(row)

    source = subprocess.check_output(["git", "show",
        "3582b4ac:bayesfilter/inference/fixed_center_curvature.py"], text=True)
    spec = importlib.util.spec_from_loader("initializer_original_complete_diagnostic", loader=None)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    exec(compile(source, spec.name, "exec"), original.__dict__)  # noqa: S102 - frozen diagnostic
    original.fit_factor_correlation_score_geometry = original_factor.fit_factor_correlation_score_geometry
    original.FactorCorrelationGeometryConfig = original_factor.FactorCorrelationGeometryConfig
    before = original.fit_fixed_center_curvature(*inputs,
        thresholds=_thresholds(original), factor_max=1).payload()
    report["complete_records"] = {"original": before}
    report["complete_record_differences"] = {}
    for label, solver in (("native", complete_orthogonal_lstsq), ("one_correction", _corrected)):
        with monkeypatch.context() as context:
            context.setattr(factor, "complete_orthogonal_lstsq", solver)
            # Each diagnostic must trace its actual solver, not reuse another
            # arm's cached enclosing function. Restore caches at phase close.
            factor._make_factor_program.cache_clear()
            native.fit_program.cache_clear()
            try:
                after = fixed.fit_fixed_center_curvature(*inputs,
                    thresholds=_thresholds(fixed), factor_max=1).payload()
            finally:
                factor._make_factor_program.cache_clear()
                native.fit_program.cache_clear()
        report["complete_records"][label] = after
        report["complete_record_differences"][label] = _record_differences(after, before)
    print("FIXED_FITTING_INITIALIZER_ROUNDING " + json.dumps(report, sort_keys=True))

"""Independent diagnostic decomposition of the residual weight derivative."""

import hashlib
import inspect
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from tests import filter_repair_genut_reverse_candidate as helpers
from tests.filter_repair_highest_dot_candidate import HighestPrecisionDotFamily
from tests.test_filter_repair_genut_dot_pullback import _program_gradient
from tests.test_filter_repair_genut_layout import _highest_dot_candidate
from tests.test_filter_repair_genut_reverse_precision import (
    REFERENCE_COMMIT,
    _coefficients,
    _compare,
    _fields,
    _independent_reference,
    _substitution,
)
from tests.test_filter_repair_genut_transitive import _fixture, _write


def _split_weights():
    original, source = _substitution("solve_gram_primal")
    assert "weight_uses" not in source
    changes = {
        "weights = tf.convert_to_tensor(weights, source.dtype)":
        "weight_uses = tf.convert_to_tensor(weights, source.dtype)\n    weights = weight_uses[0]",
        "weights[:, None] * tf.pow(source_standardized, 3.0)":
        "weight_uses[1, :, None] * tf.pow(source_standardized, 3.0)",
        "weights[:, None] * tf.pow(source_standardized, 4.0)":
        "weight_uses[2, :, None] * tf.pow(source_standardized, 4.0)",
        "source_standardized, weights\n    )":
        "source_standardized, weight_uses[3]\n    )",
    }
    for before, after in changes.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    split = ModuleType("_genut_split_weight_uses_diagnostic")
    split.native_triangular_solve = helpers.native_triangular_solve
    split.precise_gram = helpers.precise_gram
    exec(compile(source, "<genut-split-weight-uses>", "exec"), split.__dict__)  # noqa: S102
    return original, split, source


def test_weight_use_contributions(request):
    original, split, source = _split_weights()
    output = Path(request.config.getoption("xmlpath")).parent
    (output / "genut-split-weight-uses.py").write_text(source)
    (output / "genut-reverse-helper.py").write_text(inspect.getsource(helpers))
    inputs32 = _fixture(tf.float32, 18)
    coefficients = _coefficients(inputs32[0].shape)
    report = {"role": "weight_contribution_localization_no_runtime_or_precision_admission",
              "reference_commit": REFERENCE_COMMIT, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "input_recipe": "CPU stateless seeds 101/102,103/104,105/106; 72x18; identical rounded FP32 inputs",
              "inputs": inputs32, "coefficients": coefficients,
              "branches": ["source_moments", "target_skew", "target_kurtosis", "pair_moments"],
              "records": []}
    try:
        for jit in (False, True):
            for dtype in (tf.float32, tf.float64):
                inputs = tuple(tf.cast(x, dtype) for x in inputs32)
                split_inputs = (inputs[0], tf.stack([inputs[1]] * 4), inputs[2])
                full_owner = _program_gradient(original.dual_cap_genut_primal, inputs,
                    jit=jit, steps=4, coefficients=coefficients)
                split_owner = _program_gradient(split.dual_cap_genut_primal, split_inputs,
                    jit=jit, steps=4, coefficients=coefficients)
                for changed in (False, True):
                    # Form changed inputs in FP32 before casting to either arm.
                    source32 = inputs32[0] + (.1 if changed else 0.)
                    reset32 = inputs32[2] - (.03 if changed else 0.)
                    args = (tf.cast(source32, dtype), inputs[1], tf.cast(reset32, dtype))
                    separate_args = (args[0], split_inputs[1], args[2])
                    full, separate = _fields(full_owner(*args)), _fields(split_owner(*separate_args))
                    parts = separate["weight_gradient"].numpy().astype(np.float64)
                    total = parts.sum(axis=0)
                    magnitude = np.abs(parts).sum(axis=0)
                    # Bound only rounding in an ordinary four-term FP32 sum.
                    # It is NOT a bound on upstream nonlinear rounding.
                    unit = np.finfo(np.float32).eps / 2
                    final_sum_bound = (3 * unit / (1 - 3 * unit)) * magnitude
                    ratio = np.divide(magnitude, np.abs(total), out=np.full_like(total, np.inf), where=total != 0)
                    row = {"jit_compile": jit, "dtype": dtype.name, "changed": changed,
                           "full": full, "separate": separate, "parts_sum_fp64": total,
                           "parts_abs_sum": magnitude, "cancellation_ratio": ratio,
                           "fp32_final_sum_only_bound": final_sum_bound,
                           "forward_bitwise_equal": all(np.array_equal(value.numpy(), separate[key].numpy())
                               for key, value in full.items() if not key.endswith("_gradient"))}
                    report["records"].append(row)
                    assert row["forward_bitwise_equal"], "Instrumented forward computation changed"
                    assert all(np.all(np.isfinite(value.numpy())) for value in separate.values())
                    if dtype == tf.float64:
                        np.testing.assert_allclose(total, full["weight_gradient"].numpy(), rtol=2e-10, atol=2e-10)
                    assert full_owner.experimental_get_tracing_count() == split_owner.experimental_get_tracing_count() == 1
    finally:
        _write(request, "genut-weight-contributions.json", report)


def test_weight_precision_highest_dot(request):
    baseline, source = _substitution("solve_gram_primal")
    candidate, source, _ = _highest_dot_candidate(source)
    family = HighestPrecisionDotFamily()
    candidate.highest_dot = family
    candidate.native_triangular_solve = helpers.native_triangular_solve
    candidate.precise_gram = helpers.precise_gram
    output = Path(request.config.getoption("xmlpath")).parent
    (output / "genut-weight-highest-dot.py").write_text(source)
    inputs = _fixture(tf.float32, 18)
    coefficients = _coefficients(inputs[0].shape)
    cases = []
    for changed in (False, True):
        args = (inputs[0] + (.1 if changed else 0.), inputs[1], inputs[2] - (.03 if changed else 0.))
        reference, fd = _independent_reference(args, coefficients, steps=4)
        cases.append((changed, args, reference, fd))
    report = {"role": "uninstalled_combined_dot_precision_diagnostic",
              "reference_commit": REFERENCE_COMMIT, "inputs": inputs,
              "coefficients": coefficients, "graph_parent_contains_xla_dot": True,
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "records": []}
    try:
        for jit in (False, True):
            owner = _program_gradient(candidate.dual_cap_genut_primal, inputs,
                jit=jit, steps=4, coefficients=coefficients)
            before_owner = _program_gradient(baseline.dual_cap_genut_primal, inputs,
                jit=jit, steps=4, coefficients=coefficients)
            for changed, args, reference, fd in cases:
                actual, before = _fields(owner(*args)), _fields(before_owner(*args))
                report["records"].append({"jit_compile": jit, "changed": changed,
                    "actual": actual, "reference": reference, "reference_fd": fd,
                    "before": before, "comparison": _compare(actual, reference),
                    "before_comparison": _compare(before, reference),
                    "same_mode_comparison": _compare(actual, before)})
                replay = _fields(owner(*args))
                assert all(np.array_equal(value.numpy(), replay[key].numpy()) for key, value in actual.items())
                assert all(np.all(np.isfinite(value.numpy())) for value in actual.values())
            assert owner.experimental_get_tracing_count() == 1
        report["dot_program_count"] = len(family.programs)
        report["dot_traces"] = [program.function.experimental_get_tracing_count() for program in family.programs.values()]
        assert all(count == 1 for count in report["dot_traces"])
    finally:
        _write(request, "genut-weight-highest-dot.json", report)
    assert all(value["passed"] for row in report["records"] for key, value in row["comparison"].items()
        if key != "fraction_coordinatewise_cap_active"), "Combined candidate fails the unchanged FP64 bound"


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=["f32", "f64"])
def test_analytical_moment_pullback(dtype, request):
    # Non-normalized weights exercise the residual term that vanishes only
    # under an exact sum-to-one constraint.
    raw = _fixture(dtype, 18)
    points, weights = raw[0], raw[1] * tf.cast(1.03, dtype)
    mean_cotangent = tf.sin(tf.cast(tf.range(18), dtype) + .2)
    covariance_cotangent = tf.reshape(tf.cos(tf.cast(tf.range(18 * 18), dtype) * .13), [18, 18])

    def calculate(x, w, mean_g, covariance_g):
        with tf.GradientTape() as tape:
            tape.watch((x, w))
            mean, covariance = helpers.analytical_weighted_moments(x, w)
            loss = tf.reduce_sum(mean * mean_g) + tf.reduce_sum(covariance * covariance_g)
        return mean, covariance, tape.gradient(loss, (x, w))

    args = (points, weights, mean_cotangent, covariance_cotangent)
    owner = tf.function(calculate, input_signature=[tf.TensorSpec(x.shape, dtype) for x in args],
        jit_compile=True, autograph=False)
    actual = owner(*args)
    original, _ = _substitution("solve_gram_primal")
    p64, w64, m64, c64 = (tf.cast(x, tf.float64) for x in args)
    with tf.GradientTape() as tape:
        tape.watch((p64, w64))
        mean, covariance = original._weighted_moments(p64, w64)
        loss = tf.reduce_sum(mean * m64) + tf.reduce_sum(covariance * c64)
    reference = (mean, covariance, tape.gradient(loss, (p64, w64)))
    direction_p = tf.sin(p64 + .17) / tf.cast(tf.size(p64), tf.float64)
    direction_w = tf.cos(w64 + .41) / tf.cast(tf.size(w64), tf.float64)
    step = tf.constant(1e-4, tf.float64)
    upper = original._weighted_moments(p64 + step * direction_p, w64 + step * direction_w)
    lower = original._weighted_moments(p64 - step * direction_p, w64 - step * direction_w)
    fd = (tf.reduce_sum((upper[0] - lower[0]) * m64) + tf.reduce_sum((upper[1] - lower[1]) * c64)) / (2 * step)
    projection = tf.reduce_sum(tf.cast(actual[2][0], tf.float64) * direction_p) + tf.reduce_sum(tf.cast(actual[2][1], tf.float64) * direction_w)
    _write(request, f"genut-analytical-moment-pullback-{dtype.name}.json",
        {"role": "independent_moment_derivative_diagnostic", "actual": actual,
         "reference": reference, "fd": fd, "projection": projection})
    tolerance = 2e-5 if dtype == tf.float32 else 2e-10
    for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(reference)):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(projection.numpy(), fd.numpy(), rtol=2e-5 if dtype == tf.float32 else 1e-6,
        atol=2e-5 if dtype == tf.float32 else 1e-6)


def test_weight_precision_analytical_moments(request):
    baseline, source = _substitution("solve_gram_primal")
    candidate, _ = _substitution("solve_gram_primal")
    candidate._weighted_moments = helpers.analytical_weighted_moments
    output = Path(request.config.getoption("xmlpath")).parent
    (output / "genut-analytical-moment-helper.py").write_text(inspect.getsource(helpers))
    inputs = _fixture(tf.float32, 18)
    coefficients = _coefficients(inputs[0].shape)
    report = {"role": "uninstalled_analytical_moment_regrouping", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "inputs": inputs, "coefficients": coefficients, "records": []}
    try:
        for jit in (False, True):
            owner = _program_gradient(candidate.dual_cap_genut_primal, inputs, jit=jit, steps=4, coefficients=coefficients)
            prior = _program_gradient(baseline.dual_cap_genut_primal, inputs, jit=jit, steps=4, coefficients=coefficients)
            for changed in (False, True):
                args = (inputs[0] + (.1 if changed else 0.), inputs[1], inputs[2] - (.03 if changed else 0.))
                reference, fd = _independent_reference(args, coefficients, steps=4)
                actual, before = _fields(owner(*args)), _fields(prior(*args))
                row = {"jit_compile": jit, "changed": changed, "actual": actual, "before": before,
                       "reference": reference, "reference_fd": fd, "comparison": _compare(actual, reference),
                       "forward_bitwise_same": all(np.array_equal(value.numpy(), before[key].numpy())
                           for key, value in actual.items() if not key.endswith("_gradient"))}
                report["records"].append(row)
                assert row["forward_bitwise_same"]
                assert all(np.all(np.isfinite(value.numpy())) for value in actual.values())
            assert owner.experimental_get_tracing_count() == 1
    finally:
        _write(request, "genut-weight-analytical-moments.json", report)
    assert all(value["passed"] for row in report["records"] for key, value in row["comparison"].items()
        if key != "fraction_coordinatewise_cap_active"), "Moment regrouping fails the unchanged FP64 bound"


def _source_feature_cut():
    """Mechanically split the frozen calculation; never imported by runtime."""
    baseline, source = _substitution("solve_gram_primal")
    names = ("target_mean", "target_covariance", "target_cholesky", "target_skew", "target_kurtosis",
             "target_co_skew", "target_co_kurtosis")
    start = source.index("    target_mean, target_covariance = _weighted_moments(source, weights)")
    end = source.index("    standardized = _standardize_uniform(reset_points)", start)
    preparation = ("\n\ndef prepare_source_features(source, weights):\n" + source[start:end]
                   + "    return (" + ", ".join(names) + ")\n")
    continuation = source[:start] + source[end:]
    continuation = continuation.replace("def dual_cap_genut_primal(", "def finish_from_features(")
    arguments = "    reset_points: Tensor,\n"
    assert continuation.count(arguments) == 1
    continuation = continuation.replace(arguments,
        arguments + "".join("    " + name + ": Tensor,\n" for name in names))
    cut_source = continuation + preparation
    module = ModuleType("_genut_source_feature_cut_diagnostic")
    module.native_triangular_solve = helpers.native_triangular_solve
    module.precise_gram = helpers.precise_gram
    exec(compile(cut_source, "<genut-source-feature-cut>", "exec"), module.__dict__)  # noqa: S102
    return baseline, module, cut_source, names


def test_source_feature_error_decomposition(request):
    baseline, split, source, feature_names = _source_feature_cut()
    output = Path(request.config.getoption("xmlpath")).parent
    (output / "genut-source-feature-cut.py").write_text(source)
    (output / "genut-source-feature-helper.py").write_text(inspect.getsource(helpers))
    inputs = _fixture(tf.float32, 18)
    coefficients = _coefficients(inputs[0].shape)

    def wrap(function, values, jit):
        return tf.function(function, input_signature=[tf.TensorSpec(x.shape, x.dtype) for x in values],
                           jit_compile=jit, autograph=False)

    def finish(x, w, reset, *features):
        with tf.GradientTape() as tape:
            tape.watch(features)
            values = split.finish_from_features(x, w, reset, *features,
                                                diagonal_steps=4, pairwise_steps=4)
            loss = tf.reduce_sum(tf.sin(values["particles"]) * tf.cast(coefficients, x.dtype))
        return values, loss, tape.gradient(loss, features, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    def pullback(x, w, *cotangents):
        with tf.GradientTape() as tape:
            tape.watch((x, w))
            features = split.prepare_source_features(x, w)
        return tape.gradient(features, (x, w), output_gradients=cotangents)

    report = {"role": "source_feature_error_localization_only", "reference_commit": REFERENCE_COMMIT,
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "feature_names": feature_names, "inputs": inputs, "coefficients": coefficients, "records": []}
    try:
        for jit in (False, True):
            owners = {}
            for dtype in (tf.float32, tf.float64):
                args = tuple(tf.cast(value, dtype) for value in inputs)
                prepare = wrap(split.prepare_source_features, args[:2], jit)
                features = prepare(*args[:2])
                owners[dtype] = (prepare, wrap(finish, (*args, *features), jit),
                    wrap(pullback, (*args[:2], *features), jit),
                    _program_gradient(baseline.dual_cap_genut_primal, args,
                                      jit=jit, steps=4, coefficients=coefficients))
            for changed in (False, True):
                args32 = (inputs[0] + (.1 if changed else 0.), inputs[1],
                          inputs[2] - (.03 if changed else 0.))
                args64 = tuple(tf.cast(value, tf.float64) for value in args32)
                p32, h32, df32, complete32 = owners[tf.float32]
                p64, h64, df64, complete64 = owners[tf.float64]
                b32, b64 = p32(*args32[:2]), p64(*args64[:2])
                v32, loss32, g32 = h32(*args32, *b32)
                v64, loss64, g64 = h64(*args64, *b64)
                rounded_features = tuple(tf.cast(value, tf.float64) for value in b32)
                _, _, g64_at32 = h64(*args64, *rounded_features)
                derivative32 = df32(*args32[:2], *g32)[1]
                derivative64 = df64(*args64[:2], *g64)[1]
                derivative64_at32 = df64(*args64[:2], *g64_at32)[1]
                derivative64_observed = df64(*args64[:2], *(tf.cast(g, tf.float64) for g in g32))[1]
                full32, full64 = _fields(complete32(*args32)), _fields(complete64(*args64))
                reference, fd = _independent_reference(args32, coefficients, steps=4)
                terms = (tf.cast(derivative32, tf.float64) - derivative64_observed,
                         derivative64_observed - derivative64_at32,
                         derivative64_at32 - derivative64)
                total = tf.cast(derivative32, tf.float64) - derivative64
                row = {"jit_compile": jit, "changed": changed,
                       "features32": b32, "features64": b64, "cotangents32": g32,
                       "cotangents64": g64, "cotangents64_at_features32": g64_at32,
                       "weight_gradient32": derivative32, "weight_gradient64": derivative64,
                       "complete32": full32, "complete64": full64, "reference_fd": fd,
                       "preparation_pullback_error": terms[0], "continuation_cotangent_error": terms[1],
                       "source_feature_rounding_effect": terms[2], "total_error": total,
                       "forward_bitwise_same32": all(np.array_equal(value.numpy(), full32[key].numpy())
                           for key, value in {**v32, "scalar_loss": loss32}.items()),
                       "forward_bitwise_same64": all(np.array_equal(value.numpy(), full64[key].numpy())
                           for key, value in {**v64, "scalar_loss": loss64}.items())}
                bounds = 2e-5 * (1 + np.abs(reference["weight_gradient"].numpy()))
                row["complete_failing_coordinates"] = np.flatnonzero(
                    np.abs(full32["weight_gradient"].numpy() - reference["weight_gradient"].numpy()) > bounds)
                row["cut_failing_coordinates"] = np.flatnonzero(
                    np.abs(derivative32.numpy() - reference["weight_gradient"].numpy()) > bounds)
                report["records"].append(row)
                np.testing.assert_allclose(derivative64.numpy(), reference["weight_gradient"].numpy(),
                                           rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(tf.add_n(terms).numpy(), total.numpy(), rtol=1e-12, atol=1e-12)
            assert all(owner.experimental_get_tracing_count() == 1
                       for group in owners.values() for owner in group)
    finally:
        _write(request, "genut-source-feature-error.json", report)
    assert all(row["forward_bitwise_same32"] and row["forward_bitwise_same64"]
               for row in report["records"]), "Cut changes forward evaluation; cannot attribute the complete program"
    assert all(np.array_equal(row["complete_failing_coordinates"], row["cut_failing_coordinates"])
               for row in report["records"]), "Cut does not reproduce the complete gradient failures"


def test_whole_program_forward_weight_directions(request):
    module, source = _substitution("solve_gram_primal")
    baseline, _ = _substitution("solve_gram_primal")
    inputs = _fixture(tf.float32, 18)
    coefficients = _coefficients(inputs[0].shape)
    # TF ForwardAccumulator cannot capture the custom pullback's loop-local
    # solution from the parent graph. Encapsulate that same callable at its
    # actual operand signature; the XLA parent still compiles its operations.
    # The graph parent is an explicit diagnostic exception.
    dimension = inputs[0].shape[1]
    module.native_triangular_solve = tf.function(helpers.native_triangular_solve,
        input_signature=[tf.TensorSpec([dimension, dimension], tf.float32),
                         tf.TensorSpec([dimension, inputs[0].shape[0]], tf.float32)],
        jit_compile=False, autograph=False)

    def directional(x, w, reset, direction):
        with tf.autodiff.ForwardAccumulator(w, direction) as accumulator:
            values = module.dual_cap_genut_primal(x, w, reset, diagonal_steps=4, pairwise_steps=4)
            loss = tf.reduce_sum(tf.sin(values["particles"]) * coefficients)
        return values, loss, accumulator.jvp(loss)

    output = Path(request.config.getoption("xmlpath")).parent
    (output / "genut-weight-forward-source.py").write_text(source)
    (output / "genut-weight-forward-helper.py").write_text(inspect.getsource(helpers))
    report = {"role": "whole_program_directional_diagnostic_no_score_admission",
              "reference_commit": REFERENCE_COMMIT, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "inputs": inputs, "coefficients": coefficients, "records": []}
    try:
        for jit in (False, True):
            owner = tf.function(directional,
                input_signature=[tf.TensorSpec(x.shape, x.dtype) for x in (*inputs, inputs[1])],
                jit_compile=jit, autograph=False)
            reverse = _program_gradient(baseline.dual_cap_genut_primal, inputs,
                                        jit=jit, steps=4, coefficients=coefficients)
            for changed in (False, True):
                args = (inputs[0] + (.1 if changed else 0.), inputs[1], inputs[2] - (.03 if changed else 0.))
                before = _fields(reverse(*args))
                reference, fd = _independent_reference(args, coefficients, steps=4)
                for coordinate in (18, 21):
                    direction = tf.one_hot(coordinate, inputs[1].shape[0], dtype=tf.float32)
                    values, loss, derivative = owner(*args, direction)
                    forward = {**values, "scalar_loss": loss}
                    observed, expected = float(derivative), float(reference["weight_gradient"][coordinate])
                    row = {"jit_compile": jit, "changed": changed, "coordinate": coordinate,
                           "actual": forward, "before": before, "reference": reference, "reference_fd": fd,
                           "forward_derivative": derivative, "reverse_derivative": before["weight_gradient"][coordinate],
                           "reference_derivative": expected,
                           "gradient_passed": abs(observed - expected) <= 2e-5 * (1 + abs(expected)),
                           "forward_bitwise_same": all(np.array_equal(value.numpy(), before[key].numpy())
                                                       for key, value in forward.items())}
                    report["records"].append(row)
                    assert np.isfinite(observed)
                    replay = owner(*args, direction)
                    assert all(np.array_equal(a.numpy(), b.numpy())
                               for a, b in zip(tf.nest.flatten((values, loss, derivative)), tf.nest.flatten(replay)))
            assert owner.experimental_get_tracing_count() == reverse.experimental_get_tracing_count() == 1
    finally:
        _write(request, "genut-weight-forward-directions.json", report)
    assert all(row["forward_bitwise_same"] for row in report["records"]), "Forward-mode instrumentation changes primal values"
    assert all(row["gradient_passed"] for row in report["records"]), "Forward derivatives also fail the unchanged FP64 bound"

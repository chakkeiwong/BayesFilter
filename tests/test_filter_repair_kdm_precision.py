"""Independent rounded-input reference for the unresolved FP32 reset gate."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_unified_reset_tf import (
    batched_sinkhorn_contract_e_reset_triple_with_tangent,
)
from tests.test_filter_repair_kdm_auxiliary import _json, _original


def _operands():
    path = Path(__file__).parent / "fixtures/filter_repair_kdm_reset_seed131.json"
    data = json.loads(path.read_text())
    record = data["canonical"][2][0]
    dtype = tf.float32
    return (
        tf.constant([record["children"]] * 2, dtype),
        tf.constant([[record["d_children"]] * 2] * 2, dtype),
        tf.constant([record["post_covariances"]] * 2, dtype),
        tf.constant([[record["d_post_covariances"]] * 2] * 2, dtype),
        tf.constant([record["posterior_weights"]] * 2, dtype),
        tf.constant([[record["d_posterior_weights"]] * 2] * 2, dtype),
        tf.constant(data["canonical_options"]["reset_design"], dtype),
    )


def _errors(actual, expected):
    result = []
    for index, (a, b) in enumerate(zip(actual[:-1], expected[:-1], strict=True)):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        delta = np.abs(a - b)
        result.append({"output_index": index, "max_abs": float(np.max(delta)),
                       "max_tolerance_units": float(np.max(delta / (1.e-6 + 1.e-6 * np.abs(b))))})
    return result


def _numpy_reset(states, covariances, weights, design):
    """Independent FP64 primal: dense array contractions and direct solves."""
    def sym(matrix):
        return (matrix + np.swapaxes(matrix, -1, -2)) / 2.

    count, dimension = states.shape[1:]
    deltas = states[:, :, None] - states[:, None, :]
    cost = np.sum(deltas ** 2, axis=-1)
    # Preserve the existing TensorFlow cast-from-Python-float constants. These
    # are part of the finite program, not idealized decimal replacements.
    scale = np.maximum(cost.mean(axis=(1, 2)), float(np.float32(1.e-3)))
    kernel = np.exp(-cost / (2. * scale[:, None, None]))
    tiny, ridge = float(np.float32(1.e-7)), float(np.float32(1.e-5))
    left, right = np.ones_like(weights), np.ones_like(weights)
    for _ in range(16):
        left = (1. / count) / (np.einsum("bij,bj->bi", kernel, right) + tiny)
        right = weights / (np.einsum("bij,bi->bj", kernel, left) + tiny)
    coupling = left[:, :, None] * kernel * right[:, None, :]
    transport = coupling / coupling.sum(axis=-1, keepdims=True)
    barycentric = transport @ states
    mean = np.einsum("bn,bnd->bd", weights, states)
    centered = states - mean[:, None]
    covariance = sym(np.einsum("bn,bni,bnj->bij", weights, centered, centered))
    plus = barycentric - barycentric.mean(axis=1, keepdims=True)
    plus_covariance = sym(np.einsum("bni,bnj->bij", plus, plus) / count)
    regularizer = ridge * np.eye(dimension)
    gap_chol = np.linalg.cholesky(sym(covariance - plus_covariance) + regularizer)
    injected = barycentric + design @ np.swapaxes(gap_chol, -1, -2)
    centered_injected = injected - injected.mean(axis=1, keepdims=True)
    injected_covariance = sym(np.einsum("bni,bnj->bij", centered_injected, centered_injected) / count)
    target_chol = np.linalg.cholesky(covariance + regularizer)
    injected_chol = np.linalg.cholesky(injected_covariance + regularizer)
    affine = np.swapaxes(np.linalg.solve(np.swapaxes(injected_chol, -1, -2),
                                         np.swapaxes(target_chol, -1, -2)), -1, -2)
    particles = centered_injected @ np.swapaxes(affine, -1, -2) + mean[:, None]
    carried = sym(np.einsum("bij,bjkl->bikl", transport, covariances))
    return particles, carried, transport


def test_independent_rounded_precision_authority(request):
    original, _ = _original()
    values = tuple(value.numpy().astype(np.float64) for value in _operands())
    states, d_states, covariances, d_covariances, weights, d_weights, design = values
    options = {"epsilon": 2., "sinkhorn_steps": 8, "balance_steps": 8, "ridge": 1.e-5}
    reference = original.reset_reference(*(tf.constant(v, tf.float64) for v in values), **options)
    independent = _numpy_reset(states, covariances, weights, design)
    derivatives = {}
    for epsilon in (1.e-5, 2.e-5):
        directions = []
        for k in range(d_states.shape[0]):
            plus = _numpy_reset(states + epsilon * d_states[k], covariances + epsilon * d_covariances[k],
                                weights + epsilon * d_weights[k], design)
            minus = _numpy_reset(states - epsilon * d_states[k], covariances - epsilon * d_covariances[k],
                                 weights - epsilon * d_weights[k], design)
            directions.append(tuple((a - b) / (2. * epsilon) for a, b in zip(plus, minus, strict=True)))
        derivatives[str(epsilon)] = tuple(np.stack([row[i] for row in directions]) for i in range(3))
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-independent-precision.json"
    output.write_text(json.dumps(_json({"role": "independent_reference_only", "inputs": values,
        "tf64_reference": reference, "numpy_primal": independent,
        "finite_differences": derivatives}), indent=2, allow_nan=False) + "\n")
    for expected, index in zip(independent, (0, 2, 4), strict=True):
        np.testing.assert_allclose(reference[index].numpy(), expected, atol=2.e-11, rtol=2.e-11)
    for derivative in derivatives.values():
        for expected, index in zip(derivative, (1, 3, 5), strict=True):
            np.testing.assert_allclose(reference[index].numpy(), expected, atol=4.e-8, rtol=4.e-6)


@pytest.mark.parametrize("tf32", [True, False], ids=["tf32", "no_tf32"])
def test_reset_rounded_input_precision_reference(tf32, request):
    # Explicit precision reference arm; never changes a runtime default.
    tf.config.experimental.enable_tensor_float_32_execution(tf32)
    original, _ = _original()
    values = _operands()
    options = {"epsilon": 2., "sinkhorn_steps": 8, "balance_steps": 8, "ridge": 1.e-5}
    matrices = []
    original_cholesky = original.reset_reference.__globals__["safe_cholesky"]

    def capture(matrix, name):
        valid, factor = original_cholesky(matrix, name)
        matrices.append({"name": name, "matrix": matrix, "factor": factor, "valid": valid})
        return valid, factor

    # Capture only eager references; no optimizer witness is inferred from an
    # instrumented graph with changed output visibility.
    original.reset_reference.__globals__["safe_cholesky"] = capture
    try:
        eager32 = original.reset_reference(*values, **options)
        matrices32 = matrices[:]
        matrices.clear()
        rounded64 = tuple(tf.cast(value, tf.float64) for value in values)
        reference64 = original.reset_reference(*rounded64, **options)
        matrices64 = matrices[:]
    finally:
        original.reset_reference.__globals__["safe_cholesky"] = original_cholesky
    modes = {}
    for name, implementation in (("original", original.reset_reference),
                                 ("native", batched_sinkhorn_contract_e_reset_triple_with_tangent)):
        for jit in (False, True):
            def build(fn, compile_mode):
                @tf.function(input_signature=[tf.TensorSpec(v.shape, v.dtype) for v in values],
                             jit_compile=compile_mode, autograph=False)
                def program(*args):
                    return fn(*args, **options)
                return program
            modes[f"{name}_{'xla' if jit else 'graph'}"] = build(implementation, jit)(*values)
    diagnostics = []
    for row32, row64 in zip(matrices32, matrices64, strict=True):
        matrix = row64["matrix"].numpy()
        eigenvalues = np.linalg.eigvalsh(matrix)
        condition = np.linalg.cond(matrix)
        factor32 = row32["factor"].numpy().astype(np.float64)
        matrix32 = row32["matrix"].numpy().astype(np.float64)
        residual = np.linalg.norm(factor32 @ np.swapaxes(factor32, -1, -2) - matrix32,
                                  axis=(-2, -1)) / np.linalg.norm(matrix32, axis=(-2, -1))
        diagnostics.append({"name": row32["name"], "eigenvalues_rounded64": eigenvalues,
            "condition_number_rounded64": condition,
            "condition_times_fp32_epsilon_explanatory_only": condition * np.finfo(np.float32).eps,
            "fp32_factor_relative_residual": residual,
            "fp32_matrix_vs_rounded64_max_abs": float(np.max(np.abs(matrix32 - matrix))),
            "fp32": row32, "rounded64": row64})
    errors = {name: _errors(outputs, reference64) for name, outputs in {"original_eager": eager32, **modes}.items()}
    comparisons = {name: _errors(outputs, eager32) for name, outputs in modes.items()}
    report = {"role": "precision_attribution_only_no_tolerance_change",
        "inputs32": values, "rounded64_reference": reference64, "original_eager32": eager32,
        "modes": modes, "errors_against_rounded64": errors,
        "errors_against_original_eager32": comparisons, "factor_diagnostics": diagnostics,
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "result_device": modes["native_xla"][0].device,
        "limitations": ["Condition proxy is explanatory, not a new acceptance bound",
                        "No intermediate compiled operator has been identified"]}
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-rounded-precision.json"
    output.write_text(json.dumps(_json(report), indent=2, allow_nan=False) + "\n")
    assert len(diagnostics) == 3
    for outputs in (reference64, eager32, *modes.values()):
        assert bool(tf.reduce_all(outputs[-1]).numpy())
        assert all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in outputs[:-1])
    # This gate checks implementation preservation in matching execution mode.
    # Cross-mode errors above remain numerical diagnostics and a master veto.
    for native, expected in zip(modes["native_xla"], modes["original_xla"], strict=True):
        np.testing.assert_array_equal(native.numpy(), expected.numpy())


def test_saved_precision_and_memory_evidence(request):
    """Recompute saved comparison errors and report the construction slope."""
    root = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    summary = {"precision": {}, "construction": {}}
    for number in (3957, 3958, 3959):
        data = json.loads((root / f"run-{number:05d}/kdm-rounded-precision.json").read_text())
        reference = data["rounded64_reference"]
        for name, values in {"original_eager": data["original_eager32"], **data["modes"]}.items():
            assert _errors(values, reference) == data["errors_against_rounded64"][name]
        for name, values in data["modes"].items():
            assert _errors(values, data["original_eager32"]) == data["errors_against_original_eager32"][name]
        maximum = {name: max(row["max_tolerance_units"] for row in rows)
                   for name, rows in data["errors_against_rounded64"].items()}
        assert maximum["native_xla"] < 1.
        assert maximum["original_eager"] > 1.
        assert data["modes"]["original_xla"] == data["modes"]["native_xla"]
        summary["precision"][str(number)] = maximum
    for number in (3965, 3966, 3969, 3970):
        data = json.loads((root / f"run-{number:05d}/kdm-repeated-owner.json").read_text())
        assert data["count"] == len(data["rounds"]) == 6
        rss = [row["released"]["status"]["VmRSS"] for row in data["rounds"]]
        functions = [row["released"]["registered_functions"] for row in data["rounds"]]
        assert len(set(functions)) == 1
        assert all(row["collections"][-1]["released"] == {"owner": True, "graph": True}
                   for row in data["rounds"][1:])
        summary["construction"][str(number)] = {"rss_bytes": rss,
            "tail_mean_growth_bytes_per_construction": (rss[-1] - rss[2]) / 3,
            "function_count": functions[0], "jit_compile": data["jit_compile"]}
    summary["conclusion"] = (
        "Native XLA agrees with original XLA and the independent qualified FP64 reference. "
        "The eager comparator fails the unchanged FP32 tolerance. Repeated XLA construction "
        "retains host memory after later Python graphs are collected; exact native cause remains open.")
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-followup-analysis.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

from __future__ import annotations

import ast
import argparse
import copy
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.linear.kalman_qr_tf as kalman_qr_tf


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
_EXPECTED_BASES: dict[str, dict[str, object]] = {}


def _load_benchmark():
    name = "kalman_qr_batch_native_autodiff_benchmark"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark():
    return _load_benchmark()


def _tolerances(dtype: tf.DType) -> dict[str, tuple[float, float]]:
    if dtype == tf.float32:
        return {"value": (2.0e-4, 2.0e-4), "score": (2.0e-4, 2.0e-4)}
    return {"value": (1.0e-10, 1.0e-10), "score": (1.0e-8, 1.0e-9)}


def _assert_close(
    actual: tf.Tensor,
    expected: tf.Tensor,
    tolerance: tuple[float, float],
) -> None:
    rtol, atol = tolerance
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=rtol, atol=atol)


def _raw_batch_value(benchmark, fixture, params: tf.Tensor) -> tf.Tensor:
    tensors = benchmark._batched_model_tensors(fixture, params)
    return kalman_qr_tf.tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function(
        observations=fixture.observations,
        transition_offset=tensors[0],
        transition_matrix=tensors[1],
        transition_covariance=tensors[2],
        observation_offset=tensors[3],
        observation_matrix=tensors[4],
        observation_covariance=tensors[5],
        initial_state_mean=tensors[6],
        initial_state_covariance=tensors[7],
        jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
        jitter_updates_filtered_covariance=True,
    )


def _synthetic_output(dtype: str, batch_size: int) -> dict[str, object]:
    values = [1.0 + index * 0.1 for index in range(batch_size)]
    scores = [
        [2.0 + row * 0.1 + column * 0.01 for column in range(3)]
        for row in range(batch_size)
    ]
    return {
        "value": values,
        "score": scores,
        "value_dtype": dtype,
        "score_dtype": dtype,
        "value_shape": [batch_size],
        "score_shape": [batch_size, 3],
    }


def _synthetic_expected(benchmark, mode: str, tmp_path: Path) -> dict[str, object]:
    output = str(tmp_path / f"{mode}.json")
    log = str(tmp_path / f"{mode}.log")
    argv = ["synthetic", mode, output, log]
    if mode not in _EXPECTED_BASES:
        _EXPECTED_BASES[mode] = benchmark.phase4_expected_contract(mode)
    expected = copy.deepcopy(_EXPECTED_BASES[mode])
    expected["output_json"] = output
    expected["log_path"] = log
    expected["command_argv"] = argv
    return expected


def _synthetic_common_raw(expected: dict[str, object]) -> dict[str, object]:
    return {
        "schema": expected["schema"],
        "mode": expected["mode"],
        "methods": {
            "schema": expected["method_schema"],
            "contract_version": expected["method_contract_version"],
            "primary_ids": copy.deepcopy(expected["primary_method_ids"]),
            "reference_ids": copy.deepcopy(expected["reference_method_ids"]),
        },
        "versions": copy.deepcopy(expected["versions"]),
        "case_contract": copy.deepcopy(expected["case_contract"]),
        "tolerances": copy.deepcopy(expected["tolerances"]),
        "fixture_identities": copy.deepcopy(expected["fixture_identities"]),
        "declared_path_manifest": copy.deepcopy(expected["declared_path_manifest"]),
        "runtime_manifest": copy.deepcopy(expected["runtime_manifest"]),
        "provenance": {
            "git_commit": expected["git_commit"],
            "command_argv": copy.deepcopy(expected["command_argv"]),
            "cwd": expected["cwd"],
            "python_executable": expected["python_executable"],
            "conda_default_env": expected["conda_default_env"],
            "conda_prefix": expected["conda_prefix"],
            "output_json": expected["output_json"],
            "log_path": expected["log_path"],
            "plan_path": expected["plan_path"],
            "result_path": expected["result_path"],
            "requested_device": expected["requested_device"],
            "cuda_visible_devices": expected["cuda_visible_devices"],
            "gpu_detection_by_harness": expected["gpu_detection_by_harness"],
            "requested_cpu_threads": expected["requested_cpu_threads"],
            "effective_intra_op_threads": 1,
            "effective_inter_op_threads": 1,
            "thread_environment": copy.deepcopy(expected["thread_environment"]),
            "jit_compile": expected["jit_compile"],
            "xla_execution": expected["xla_execution"],
            "tf32_status": expected["tf32_status"],
        },
        "nonclaims": copy.deepcopy(expected["nonclaims"]),
        "collection_error": None,
        "internal_wall_time_seconds": 1.0,
    }


def _synthetic_diagnostic_raw(expected: dict[str, object]) -> dict[str, object]:
    raw = _synthetic_common_raw(expected)
    parity_rows = []
    for dtype in ("float32", "float64"):
        for batch_size in (1, 4):
            output = _synthetic_output(dtype, batch_size)
            parity_rows.append(
                {
                    "dtype": dtype,
                    "batch_size": batch_size,
                    "parameter_count": 3,
                    "batch_native_autodiff": copy.deepcopy(output),
                    "scalar_autodiff": copy.deepcopy(output),
                    "batch_native_analytical": copy.deepcopy(output),
                }
            )
    jacobian_rows = []
    for dtype in ("float32", "float64"):
        output = _synthetic_output(dtype, 4)
        jacobian = [
            [
                copy.deepcopy(output["score"][row]) if row == column else [0.0, 0.0, 0.0]
                for column in range(4)
            ]
            for row in range(4)
        ]
        jacobian_rows.append(
            {
                "dtype": dtype,
                "batch_size": 4,
                "parameter_count": 3,
                "value": copy.deepcopy(output["value"]),
                "score": copy.deepcopy(output["score"]),
                "value_dtype": dtype,
                "score_dtype": dtype,
                "value_shape": [4],
                "score_shape": [4, 3],
                "jacobian": jacobian,
                "jacobian_dtype": dtype,
                "jacobian_shape": [4, 4, 3],
                "perturbation": [0.01, -0.015, 0.02],
                "perturbed_value": copy.deepcopy(output["value"]),
                "perturbed_score": copy.deepcopy(output["score"]),
                "perturbed_value_dtype": dtype,
                "perturbed_score_dtype": dtype,
                "perturbed_value_shape": [4],
                "perturbed_score_shape": [4, 3],
            }
        )
    raw["parity_rows"] = parity_rows
    raw["jacobian_rows"] = jacobian_rows
    return raw


def _synthetic_xla_raw(expected: dict[str, object]) -> dict[str, object]:
    raw = _synthetic_common_raw(expected)
    raw["compiled"] = _synthetic_output("float32", 4)
    raw["non_jit"] = _synthetic_output("float32", 4)
    raw["concrete_function_count"] = 1
    return raw


def test_outside_tape_reduction_is_disconnected_but_vector_vjp_is_finite() -> None:
    params = tf.constant([[0.2, -0.1], [0.3, 0.4]], dtype=tf.float64)
    with tf.GradientTape() as outside_tape:
        outside_tape.watch(params)
        outside_value = tf.square(params)
    outside_reduction = tf.reduce_sum(outside_value)
    assert outside_tape.gradient(outside_reduction, params) is None

    with tf.GradientTape() as vector_tape:
        vector_tape.watch(params)
        vector_value = tf.square(params)
    vector_gradient = vector_tape.gradient(
        vector_value,
        params,
        output_gradients=tf.ones_like(vector_value),
    )
    assert vector_gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(vector_gradient)))

    with tf.GradientTape() as inside_tape:
        inside_tape.watch(params)
        inside_reduction = tf.reduce_sum(tf.square(params))
    inside_gradient = inside_tape.gradient(inside_reduction, params)
    assert inside_gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(inside_gradient)))


def test_batch_native_builder_source_is_one_vjp_without_scalar_fallback(benchmark) -> None:
    source = inspect.getsource(benchmark.build_batch_native_autodiff_fn)
    tree = ast.parse(source)
    called_names = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else None
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "reduce_sum" not in called_names
    assert "fill" not in called_names
    assert "tf_qr_sqrt_kalman_log_likelihood_while_loop" not in source
    assert "tf_qr_sqrt_kalman_score" not in source
    assert source.count("tape.gradient(") == 1
    assert "output_gradients=tf.ones_like(value)" in source
    assert "gradient is disconnected" in source


def test_reduce_sum_candidate_is_one_scalar_vjp_and_keeps_baseline_unchanged(
    benchmark,
) -> None:
    candidate_source = inspect.getsource(
        benchmark.build_batch_native_autodiff_reduce_sum_fn
    )
    baseline_source = inspect.getsource(benchmark.build_batch_native_autodiff_fn)
    candidate_tree = ast.parse(candidate_source)
    called_names = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else None
        for node in ast.walk(candidate_tree)
        if isinstance(node, ast.Call)
    }

    assert "reduce_sum" in called_names
    assert "reduced_value = tf.reduce_sum(value)" in candidate_source
    assert "output_gradients" not in candidate_source
    assert candidate_source.count("tape.gradient(") == 1
    assert "output_gradients=tf.ones_like(value)" in baseline_source
    assert "reduce_sum" not in baseline_source


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_reduce_sum_candidate_matches_explicit_seed_baseline(
    benchmark, dtype: tf.DType, batch_size: int
) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=dtype)
    params = benchmark._make_parameter_batch(fixture, batch_size)
    baseline = benchmark.build_batch_native_autodiff_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    candidate = benchmark.build_batch_native_autodiff_reduce_sum_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )

    baseline_value, baseline_score = baseline(params)
    candidate_value, candidate_score = candidate(params)
    tolerances = _tolerances(dtype)

    assert candidate_value.shape.as_list() == [batch_size]
    assert candidate_score.shape.as_list() == [batch_size, 3]
    assert candidate_value.dtype == candidate_score.dtype == dtype
    assert bool(tf.reduce_all(tf.math.is_finite(candidate_value)))
    assert bool(tf.reduce_all(tf.math.is_finite(candidate_score)))
    _assert_close(candidate_value, baseline_value, tolerances["value"])
    _assert_close(candidate_score, baseline_score, tolerances["score"])


def test_reduce_sum_candidate_preserves_row_independence(benchmark) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float64)
    params = benchmark._make_parameter_batch(fixture, 4)
    candidate = benchmark.build_batch_native_autodiff_reduce_sum_fn(
        fixture,
        batch_size=4,
        jit_compile=False,
    )
    value, score = candidate(params)
    perturbation = tf.constant([0.01, -0.015, 0.02], dtype=tf.float64)
    perturbed = tf.tensor_scatter_nd_add(params, [[2]], [perturbation])
    perturbed_value, perturbed_score = candidate(perturbed)
    unaffected = tf.constant([0, 1, 3])

    _assert_close(
        tf.gather(perturbed_value, unaffected),
        tf.gather(value, unaffected),
        _tolerances(tf.float64)["value"],
    )
    _assert_close(
        tf.gather(perturbed_score, unaffected),
        tf.gather(score, unaffected),
        _tolerances(tf.float64)["score"],
    )


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_value_only_model_constructions_match_full_model_tensors(
    benchmark, dtype: tf.DType, batch_size: int
) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=dtype)
    params = benchmark._make_parameter_batch(fixture, batch_size)
    expected = benchmark._batched_model_tensors(fixture, params)[:8]
    implicit = benchmark._batched_model_value_tensors(fixture, params)
    explicit = benchmark._batched_model_value_tensors_explicit(
        fixture,
        params,
        batch_size=batch_size,
    )
    tolerance = _tolerances(dtype)["value"]

    for candidate in (implicit, explicit):
        assert len(candidate) == len(expected) == 8
        for actual, reference in zip(candidate, expected, strict=True):
            assert actual.shape == reference.shape
            assert actual.dtype == reference.dtype == dtype
            _assert_close(actual, reference, tolerance)


@pytest.mark.parametrize(
    "builder_name",
    [
        "build_batch_native_autodiff_value_only_fn",
        "build_batch_native_autodiff_value_only_explicit_fn",
    ],
)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_value_only_autodiff_candidates_match_full_helper_baseline(
    benchmark, builder_name: str, batch_size: int
) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float32)
    params = benchmark._make_parameter_batch(fixture, batch_size)
    baseline = benchmark.build_batch_native_autodiff_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    candidate = getattr(benchmark, builder_name)(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    baseline_value, baseline_score = baseline(params)
    candidate_value, candidate_score = candidate(params)

    _assert_close(candidate_value, baseline_value, _tolerances(tf.float32)["value"])
    _assert_close(candidate_score, baseline_score, _tolerances(tf.float32)["score"])


def test_dynamic_batch_likelihood_has_xla_gradient_bound_without_static_substitution(
    benchmark,
) -> None:
    source = inspect.getsource(
        kalman_qr_tf.tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function
    )
    builder_source = inspect.getsource(benchmark.build_batch_native_autodiff_fn)
    assert "tf.while_loop" in source
    assert "maximum_iterations=n_timesteps" in source
    assert "parallel_iterations=1" in source
    assert "tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop" in builder_source
    assert "tf_qr_sqrt_kalman_log_likelihood_batched_static.python_function" not in builder_source


def test_batch_native_builder_fails_closed_when_value_is_disconnected(
    benchmark, monkeypatch
) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float64)
    params = benchmark._make_parameter_batch(fixture, 4)

    def disconnected_value(**kwargs):
        batch_size = tf.shape(kwargs["transition_offset"])[0]
        return tf.zeros([batch_size], dtype=kwargs["transition_offset"].dtype)

    monkeypatch.setattr(
        kalman_qr_tf,
        "tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop",
        SimpleNamespace(python_function=disconnected_value),
    )
    with pytest.raises(RuntimeError, match="gradient is disconnected"):
        fn = benchmark.build_batch_native_autodiff_fn(
            fixture,
            batch_size=4,
            jit_compile=False,
        )
        fn.get_concrete_function(params)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_batch_native_autodiff_matches_scalar_autodiff_and_analytical(
    benchmark, dtype: tf.DType, batch_size: int
) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=dtype)
    params = benchmark._make_parameter_batch(fixture, batch_size)
    batch_autodiff = benchmark.build_batch_native_autodiff_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    scalar_autodiff = benchmark.build_autodiff_row_loop_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    analytical = benchmark.build_batch_native_analytic_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )

    value, score = batch_autodiff(params)
    scalar_value, scalar_score = scalar_autodiff(params)
    analytical_value, analytical_score = analytical(params)
    tolerances = _tolerances(dtype)

    assert value.shape.as_list() == [batch_size]
    assert score.shape.as_list() == [batch_size, 3]
    assert value.dtype == score.dtype == dtype
    assert bool(tf.reduce_all(tf.math.is_finite(value)))
    assert bool(tf.reduce_all(tf.math.is_finite(score)))
    _assert_close(value, scalar_value, tolerances["value"])
    _assert_close(score, scalar_score, tolerances["score"])
    _assert_close(value, analytical_value, tolerances["value"])
    _assert_close(score, analytical_score, tolerances["score"])


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_full_jacobian_and_row_perturbation_prove_batch_independence(
    benchmark, dtype: tf.DType
) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=dtype)
    params = benchmark._make_parameter_batch(fixture, 4)
    batch_autodiff = benchmark.build_batch_native_autodiff_fn(
        fixture,
        batch_size=4,
        jit_compile=False,
    )
    value, score = batch_autodiff(params)

    with tf.GradientTape() as tape:
        tape.watch(params)
        raw_value = _raw_batch_value(benchmark, fixture, params)
    jacobian = tape.jacobian(raw_value, params)
    assert jacobian is not None
    assert jacobian.shape.as_list() == [4, 4, 3]
    assert bool(tf.reduce_all(tf.math.is_finite(jacobian)))

    diagonal_indices = tf.stack([tf.range(4), tf.range(4)], axis=1)
    diagonal = tf.gather_nd(jacobian, diagonal_indices)
    off_diagonal = tf.boolean_mask(jacobian, ~tf.eye(4, dtype=tf.bool))
    tolerances = _tolerances(dtype)
    _assert_close(diagonal, score, tolerances["score"])
    off_diagonal_limit = 2.0e-6 if dtype == tf.float32 else 2.0e-12
    assert float(tf.reduce_max(tf.abs(off_diagonal))) <= off_diagonal_limit

    perturbation = tf.constant([0.01, -0.015, 0.02], dtype=dtype)
    perturbed_params = tf.tensor_scatter_nd_add(params, [[2]], [perturbation])
    perturbed_value, perturbed_score = batch_autodiff(perturbed_params)
    unaffected = tf.constant([0, 1, 3])
    _assert_close(
        tf.gather(perturbed_value, unaffected),
        tf.gather(value, unaffected),
        tolerances["value"],
    )
    _assert_close(
        tf.gather(perturbed_score, unaffected),
        tf.gather(score, unaffected),
        tolerances["score"],
    )
    assert bool(tf.reduce_all(tf.math.is_finite(perturbed_value[2])))
    assert bool(tf.reduce_all(tf.math.is_finite(perturbed_score[2])))


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    [
        ("schema", "schema_and_mode"),
        ("mode", "schema_and_mode"),
        ("method_schema", "method_contract_identity"),
        ("method_version", "method_contract_identity"),
        ("primary_ids", "method_contract_identity"),
        ("reference_ids", "method_contract_identity"),
        ("fixture_version", "fixture_version_identity"),
        ("case_contract", "case_contract_identity"),
        ("tolerance", "tolerance_identity"),
        ("fixture_hash", "fixture_hash_identity"),
        ("declared_path", "declared_path_identity"),
        ("declared_path_hash", "declared_path_identity"),
        ("declared_path_status", "declared_path_identity"),
        ("source_fingerprint", "declared_path_identity"),
        ("runtime", "runtime_identity"),
        ("git_commit", "git_commit_identity"),
        ("argv", "argv_identity"),
        ("output_path", "path_identity"),
        ("log_path", "path_identity"),
        ("plan_path", "path_identity"),
        ("result_path", "path_identity"),
        ("python", "path_identity"),
        ("conda", "path_identity"),
        ("device", "cpu_device_identity"),
        ("cuda", "cpu_device_identity"),
        ("enumeration", "cpu_device_identity"),
        ("requested_threads", "thread_identity"),
        ("effective_threads", "thread_identity"),
        ("thread_env", "thread_identity"),
        ("jit", "jit_xla_tf32_identity"),
        ("xla", "jit_xla_tf32_identity"),
        ("tf32", "jit_xla_tf32_identity"),
        ("nonclaim", "nonclaims_identity"),
        ("collection_error", "collection_succeeded"),
        ("wall_time", "positive_internal_wall_time"),
    ],
)
def test_phase4_diagnostic_common_raw_mutations_fail_named_gate(
    benchmark, tmp_path: Path, mutation: str, expected_check: str
) -> None:
    expected = _synthetic_expected(benchmark, "diagnostic", tmp_path)
    raw = _synthetic_diagnostic_raw(expected)
    raw["checks"] = {"forged": True}
    if mutation == "schema":
        raw["schema"] = "stale"
    elif mutation == "mode":
        raw["mode"] = "wrong"
    elif mutation == "method_schema":
        raw["methods"]["schema"] = "v2"
    elif mutation == "method_version":
        raw["methods"]["contract_version"] = "old"
    elif mutation == "primary_ids":
        raw["methods"]["primary_ids"].reverse()
    elif mutation == "reference_ids":
        raw["methods"]["reference_ids"].pop()
    elif mutation == "fixture_version":
        raw["versions"]["fixture_contract_version"] = "old"
    elif mutation == "case_contract":
        raw["case_contract"]["timesteps"] = 5
    elif mutation == "tolerance":
        raw["tolerances"]["float32"]["score"]["atol"] *= 2.0
    elif mutation == "fixture_hash":
        raw["fixture_identities"][0]["observation_hash"] = "changed"
    elif mutation == "declared_path":
        raw["declared_path_manifest"]["files"][0]["path"] = "changed"
    elif mutation == "declared_path_hash":
        raw["declared_path_manifest"]["files"][0]["sha256"] = "changed"
    elif mutation == "declared_path_status":
        raw["declared_path_manifest"]["files"][0]["git_status_short"] = "changed"
    elif mutation == "source_fingerprint":
        raw["declared_path_manifest"]["declared_source_fingerprint"] = "changed"
    elif mutation == "runtime":
        raw["runtime_manifest"]["runtime"]["python_version"] = "changed"
    elif mutation == "git_commit":
        raw["provenance"]["git_commit"] = ""
    elif mutation == "argv":
        raw["provenance"]["command_argv"].append("changed")
    elif mutation == "output_path":
        raw["provenance"]["output_json"] = "changed"
    elif mutation == "log_path":
        raw["provenance"]["log_path"] = "changed"
    elif mutation == "plan_path":
        raw["provenance"]["plan_path"] = "changed"
    elif mutation == "result_path":
        raw["provenance"]["result_path"] = "changed"
    elif mutation == "python":
        raw["provenance"]["python_executable"] = "changed"
    elif mutation == "conda":
        raw["provenance"]["conda_prefix"] = "changed"
    elif mutation == "device":
        raw["provenance"]["requested_device"] = "gpu"
    elif mutation == "cuda":
        raw["provenance"]["cuda_visible_devices"] = "UNSET"
    elif mutation == "enumeration":
        raw["provenance"]["gpu_detection_by_harness"] = "called"
    elif mutation == "requested_threads":
        raw["provenance"]["requested_cpu_threads"] = 2
    elif mutation == "effective_threads":
        raw["provenance"]["effective_intra_op_threads"] = 2
    elif mutation == "thread_env":
        raw["provenance"]["thread_environment"]["omp_num_threads"] = "2"
    elif mutation == "jit":
        raw["provenance"]["jit_compile"] = True
    elif mutation == "xla":
        raw["provenance"]["xla_execution"] = "executed"
    elif mutation == "tf32":
        raw["provenance"]["tf32_status"] = "queried"
    elif mutation == "nonclaim":
        raw["nonclaims"].pop()
    elif mutation == "collection_error":
        raw["collection_error"] = {"type": "Synthetic", "message": "failed"}
    elif mutation == "wall_time":
        raw["internal_wall_time_seconds"] = 0.0
    evaluation = benchmark.evaluate_phase4_diagnostic(raw, expected)
    assert evaluation["checks"][expected_check] is False
    assert evaluation["state"] == "failed"
    assert evaluation["returncode"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    [
        ("missing_row", "parity_rows_complete"),
        ("value", "parity_value_score"),
        ("score", "parity_value_score"),
        ("value_shape", "parity_value_score"),
        ("score_shape", "parity_value_score"),
        ("dtype", "parity_value_score"),
        ("actual_shape", "parity_value_score"),
        ("nonfinite_tag", "parity_value_score"),
        ("nan", "parity_value_score"),
        ("infinity", "parity_value_score"),
        ("jacobian_shape", "jacobian_row_independence"),
        ("jacobian_diagonal", "jacobian_row_independence"),
        ("jacobian_offdiagonal", "jacobian_row_independence"),
        ("perturbed_value", "jacobian_row_independence"),
        ("perturbed_score", "jacobian_row_independence"),
        ("perturbation", "jacobian_row_independence"),
    ],
)
def test_phase4_diagnostic_numeric_raw_mutations_fail_named_gate(
    benchmark, tmp_path: Path, mutation: str, expected_check: str
) -> None:
    expected = _synthetic_expected(benchmark, "diagnostic", tmp_path)
    raw = _synthetic_diagnostic_raw(expected)
    if mutation == "missing_row":
        raw["parity_rows"].pop()
    elif mutation == "value":
        raw["parity_rows"][0]["batch_native_autodiff"]["value"][0] += 1.0
    elif mutation == "score":
        raw["parity_rows"][0]["batch_native_autodiff"]["score"][0][0] += 1.0
    elif mutation == "value_shape":
        raw["parity_rows"][0]["batch_native_autodiff"]["value_shape"] = [2]
    elif mutation == "score_shape":
        raw["parity_rows"][0]["batch_native_autodiff"]["score_shape"] = [1, 2]
    elif mutation == "dtype":
        raw["parity_rows"][0]["batch_native_autodiff"]["score_dtype"] = "float64"
    elif mutation == "actual_shape":
        raw["parity_rows"][0]["batch_native_autodiff"]["score"][0].pop()
    elif mutation == "nonfinite_tag":
        raw["parity_rows"][0]["batch_native_autodiff"]["value"][0] = {"nonfinite": "nan"}
    elif mutation == "nan":
        raw["parity_rows"][0]["batch_native_autodiff"]["value"][0] = float("nan")
    elif mutation == "infinity":
        raw["parity_rows"][0]["batch_native_autodiff"]["score"][0][0] = float("inf")
    elif mutation == "jacobian_shape":
        raw["jacobian_rows"][0]["jacobian_shape"] = [4, 3, 4]
    elif mutation == "jacobian_diagonal":
        raw["jacobian_rows"][0]["jacobian"][0][0][0] += 1.0
    elif mutation == "jacobian_offdiagonal":
        raw["jacobian_rows"][0]["jacobian"][0][1][0] = 1.0e-3
    elif mutation == "perturbed_value":
        raw["jacobian_rows"][0]["perturbed_value"][0] += 1.0
    elif mutation == "perturbed_score":
        raw["jacobian_rows"][0]["perturbed_score"][1][0] += 1.0
    elif mutation == "perturbation":
        raw["jacobian_rows"][0]["perturbation"][0] = 0.02
    evaluation = benchmark.evaluate_phase4_diagnostic(raw, expected)
    assert evaluation["checks"][expected_check] is False
    assert evaluation["state"] == "failed"
    assert evaluation["returncode"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    [
        ("compiled_value", "compiled_non_jit_value_parity"),
        ("compiled_score", "compiled_non_jit_score_parity"),
        ("compiled_shape", "compiled_non_jit_metadata"),
        ("non_jit_shape", "compiled_non_jit_metadata"),
        ("compiled_dtype", "compiled_non_jit_metadata"),
        ("non_jit_dtype", "compiled_non_jit_metadata"),
        ("compiled_nonfinite", "compiled_non_jit_metadata"),
        ("non_jit_nonfinite", "compiled_non_jit_metadata"),
        ("concrete_count", "one_concrete_function"),
        ("wall_time", "positive_internal_wall_time"),
        ("jit", "jit_xla_tf32_identity"),
        ("execution", "jit_xla_tf32_identity"),
        ("tolerance", "tolerance_identity"),
    ],
)
def test_phase4_xla_raw_mutations_fail_named_gate(
    benchmark, tmp_path: Path, mutation: str, expected_check: str
) -> None:
    expected = _synthetic_expected(benchmark, "cpu_xla_smoke", tmp_path)
    raw = _synthetic_xla_raw(expected)
    if mutation == "compiled_value":
        raw["compiled"]["value"][0] += 1.0
    elif mutation == "compiled_score":
        raw["compiled"]["score"][0][0] += 1.0
    elif mutation == "compiled_shape":
        raw["compiled"]["score_shape"] = [4, 2]
    elif mutation == "non_jit_shape":
        raw["non_jit"]["value_shape"] = [3]
    elif mutation == "compiled_dtype":
        raw["compiled"]["score_dtype"] = "float64"
    elif mutation == "non_jit_dtype":
        raw["non_jit"]["value_dtype"] = "float64"
    elif mutation == "compiled_nonfinite":
        raw["compiled"]["value"][0] = {"nonfinite": "nan"}
    elif mutation == "non_jit_nonfinite":
        raw["non_jit"]["score"][0][0] = float("inf")
    elif mutation == "concrete_count":
        raw["concrete_function_count"] = 2
    elif mutation == "wall_time":
        raw["internal_wall_time_seconds"] = None
    elif mutation == "jit":
        raw["provenance"]["jit_compile"] = False
    elif mutation == "execution":
        raw["provenance"]["xla_execution"] = "not_run"
    elif mutation == "tolerance":
        raw["tolerances"]["float32"]["value"]["rtol"] *= 2.0
    evaluation = benchmark.evaluate_phase4_xla_smoke(raw, expected)
    assert evaluation["checks"][expected_check] is False
    assert evaluation["state"] == "failed"
    assert evaluation["returncode"] == 1


def test_phase4_cli_branch_avoids_device_enumeration_and_tf32_queries(
    benchmark, monkeypatch, tmp_path: Path
) -> None:
    expected = _synthetic_expected(benchmark, "diagnostic", tmp_path)
    raw = _synthetic_diagnostic_raw(expected)
    args = argparse.Namespace(
        device="cpu",
        jit_compile=False,
        cpu_threads=1,
        output_json=expected["output_json"],
        phase4_log_path=expected["log_path"],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Phase 4 diagnostic must not enumerate devices or query TF32")

    monkeypatch.setattr(benchmark, "_select_device", forbidden)
    monkeypatch.setattr(tf.config, "list_physical_devices", forbidden)
    monkeypatch.setattr(tf.config, "list_logical_devices", forbidden)
    monkeypatch.setattr(
        tf.config.experimental,
        "tensor_float_32_execution_enabled",
        forbidden,
    )
    monkeypatch.setattr(
        tf.config.experimental,
        "enable_tensor_float_32_execution",
        forbidden,
    )
    monkeypatch.setattr(benchmark, "_configure_cpu_threads", lambda threads: {
        "tf_intra_op_parallelism_threads": 1,
        "tf_inter_op_parallelism_threads": 1,
        "omp_num_threads": "1",
        "tf_num_intraop_threads_env": "1",
        "tf_num_interop_threads_env": "1",
    })
    monkeypatch.setattr(
        benchmark,
        "phase4_expected_contract",
        lambda mode: expected,
    )
    monkeypatch.setattr(
        benchmark,
        "_phase4_raw_shell",
        lambda *args, **kwargs: copy.deepcopy(
            {key: value for key, value in raw.items() if key not in {"parity_rows", "jacobian_rows"}}
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_phase4_collect_parity_row",
        lambda dtype, batch_size: copy.deepcopy(
            next(
                row
                for row in raw["parity_rows"]
                if row["dtype"] == dtype.name and row["batch_size"] == batch_size
            )
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_phase4_collect_jacobian_row",
        lambda dtype: copy.deepcopy(
            next(row for row in raw["jacobian_rows"] if row["dtype"] == dtype.name)
        ),
    )
    assert benchmark.run_phase4_autodiff_diagnostic(args) == 0


def test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value(
    benchmark,
) -> None:
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float32)
    params = benchmark._make_parameter_batch(fixture, 4)
    non_jit = benchmark.build_batch_native_autodiff_fn(
        fixture,
        batch_size=4,
        jit_compile=False,
    )
    compiled = benchmark.build_batch_native_autodiff_fn(
        fixture,
        batch_size=4,
        jit_compile=True,
    )
    with tf.device("/CPU:0"):
        expected_value, expected_score = non_jit(params)
        compiled_value, compiled_score = compiled(params)
        second_value, second_score = compiled(tf.identity(params))

    assert compiled_value.dtype == compiled_score.dtype == tf.float32
    assert compiled_value.shape.as_list() == [4]
    assert compiled_score.shape.as_list() == [4, 3]
    _assert_close(compiled_value, expected_value, (2.0e-4, 2.0e-4))
    _assert_close(compiled_score, expected_score, (2.0e-4, 2.0e-4))
    _assert_close(second_value, expected_value, (2.0e-4, 2.0e-4))
    _assert_close(second_score, expected_score, (2.0e-4, 2.0e-4))
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1

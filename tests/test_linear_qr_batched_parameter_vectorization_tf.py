from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear import kalman_qr_derivatives_tf as derivatives_tf
from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor_first_derivatives,
    factor_covariance_first_derivatives,
    stack_qr_lower_factor_first_derivatives,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"


def _load_benchmark():
    name = "kalman_qr_phase3_parameter_vectorization_benchmark"
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tolerances(dtype: tf.DType) -> tuple[float, float]:
    return (2.0e-5, 2.0e-5) if dtype == tf.float32 else (2.0e-12, 2.0e-12)


def _call_batched_score_python(benchmark, fixture, parameters_batch, **kwargs):
    tensors = benchmark._batched_model_tensors(fixture, parameters_batch)
    return derivatives_tf.tf_qr_sqrt_kalman_score_batched_static.python_function(
        observations=fixture.observations,
        transition_offset=tensors[0],
        transition_matrix=tensors[1],
        transition_covariance=tensors[2],
        observation_offset=tensors[3],
        observation_matrix=tensors[4],
        observation_covariance=tensors[5],
        initial_state_mean=tensors[6],
        initial_state_covariance=tensors[7],
        d_initial_state_mean=tensors[8],
        d_initial_state_covariance=tensors[9],
        d_transition_offset=tensors[10],
        d_transition_matrix=tensors[11],
        d_transition_covariance=tensors[12],
        d_observation_offset=tensors[13],
        d_observation_matrix=tensors[14],
        d_observation_covariance=tensors[15],
        **kwargs,
    )


def _qr_inputs(dtype: tf.DType, batch_size: int, parameter_count: int):
    base = tf.reshape(tf.cast(tf.range(batch_size * 3 * 5), dtype), [batch_size, 3, 5])
    stack = 0.03 * tf.math.sin(base * 0.19 + 0.2)
    stack += tf.eye(3, 5, batch_shape=[batch_size], dtype=dtype)
    directions = tf.reshape(
        tf.cast(tf.range(batch_size * parameter_count * 3 * 5), dtype),
        [batch_size, parameter_count, 3, 5],
    )
    dstack = 0.004 * tf.math.cos(directions * 0.13 + 0.1)
    return stack, dstack


def _cholesky_inputs(dtype: tf.DType, batch_size: int, parameter_count: int):
    raw = tf.reshape(tf.cast(tf.range(batch_size * 3 * 3), dtype), [batch_size, 3, 3])
    lower = tf.linalg.band_part(0.02 * tf.math.sin(raw * 0.17), -1, 0)
    factor = lower + tf.eye(3, batch_shape=[batch_size], dtype=dtype) * 0.8
    covariance = factor @ tf.linalg.matrix_transpose(factor)
    directions = tf.reshape(
        tf.cast(tf.range(batch_size * parameter_count * 3 * 3), dtype),
        [batch_size, parameter_count, 3, 3],
    )
    raw_derivative = 0.003 * tf.math.cos(directions * 0.11 + 0.3)
    dcovariance = 0.5 * (
        raw_derivative + tf.linalg.matrix_transpose(raw_derivative)
    )
    return covariance, dcovariance


def _factor_inputs(dtype: tf.DType, batch_size: int, parameter_count: int):
    raw = tf.reshape(tf.cast(tf.range(batch_size * 3 * 3), dtype), [batch_size, 3, 3])
    factor = tf.linalg.band_part(0.025 * tf.math.cos(raw * 0.23), -1, 0)
    factor += tf.eye(3, batch_shape=[batch_size], dtype=dtype) * 0.7
    directions = tf.reshape(
        tf.cast(tf.range(batch_size * parameter_count * 3 * 3), dtype),
        [batch_size, parameter_count, 3, 3],
    )
    dfactor = 0.005 * tf.math.sin(directions * 0.07 + 0.4)
    return factor, dfactor


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize(("batch_size", "parameter_count"), [(1, 1), (1, 4), (3, 1), (3, 4)])
def test_vectorized_qr_helper_matches_scalar_reference(
    dtype: tf.DType, batch_size: int, parameter_count: int
) -> None:
    stack, dstack = _qr_inputs(dtype, batch_size, parameter_count)
    actual = derivatives_tf._batched_stack_qr_lower_factor_first_derivatives(
        stack, dstack
    )
    expected_rows = [
        stack_qr_lower_factor_first_derivatives(stack[row], dstack[row])
        for row in range(batch_size)
    ]
    expected = tuple(tf.stack(values, axis=0) for values in zip(*expected_rows, strict=True))
    rtol, atol = _tolerances(dtype)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype == dtype
        np.testing.assert_allclose(
            actual_tensor.numpy(), expected_tensor.numpy(), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize(("batch_size", "parameter_count"), [(1, 1), (1, 4), (3, 1), (3, 4)])
def test_vectorized_cholesky_helper_matches_scalar_reference(
    dtype: tf.DType, batch_size: int, parameter_count: int
) -> None:
    covariance, dcovariance = _cholesky_inputs(dtype, batch_size, parameter_count)
    actual = derivatives_tf._batched_cholesky_factor_first_derivatives(
        covariance, dcovariance
    )
    expected_rows = [
        cholesky_factor_first_derivatives(covariance[row], dcovariance[row])
        for row in range(batch_size)
    ]
    expected = tuple(tf.stack(values, axis=0) for values in zip(*expected_rows, strict=True))
    rtol, atol = _tolerances(dtype)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype == dtype
        np.testing.assert_allclose(
            actual_tensor.numpy(), expected_tensor.numpy(), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize(("batch_size", "parameter_count"), [(1, 1), (1, 4), (3, 1), (3, 4)])
def test_vectorized_factor_covariance_helper_matches_scalar_reference(
    dtype: tf.DType, batch_size: int, parameter_count: int
) -> None:
    factor, dfactor = _factor_inputs(dtype, batch_size, parameter_count)
    actual = derivatives_tf._batched_factor_covariance_first_derivatives(
        factor, dfactor
    )
    expected_rows = [
        factor_covariance_first_derivatives(factor[row], dfactor[row])
        for row in range(batch_size)
    ]
    expected = tuple(tf.stack(values, axis=0) for values in zip(*expected_rows, strict=True))
    rtol, atol = _tolerances(dtype)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype == dtype
        np.testing.assert_allclose(
            actual_tensor.numpy(), expected_tensor.numpy(), rtol=rtol, atol=atol
        )


def test_qr_helper_rejects_static_stack_with_k_less_than_n() -> None:
    stack = tf.ones([2, 4, 3], dtype=tf.float32)
    dstack = tf.ones([2, 2, 4, 3], dtype=tf.float32)
    with pytest.raises(ValueError, match="requires K>=N"):
        derivatives_tf._batched_stack_qr_lower_factor_first_derivatives(
            stack, dstack
        )


def test_qr_helper_rejects_dynamic_stack_with_k_less_than_n() -> None:
    @tf.function(
        jit_compile=False,
        input_signature=[
            tf.TensorSpec([2, None, None], tf.float32),
            tf.TensorSpec([2, None, None, None], tf.float32),
        ],
    )
    def wrapped(stack: tf.Tensor, dstack: tf.Tensor):
        return derivatives_tf._batched_stack_qr_lower_factor_first_derivatives(
            stack, dstack
        )

    with pytest.raises(tf.errors.InvalidArgumentError, match="requires K>=N"):
        wrapped(
            tf.ones([2, 4, 3], dtype=tf.float32),
            tf.ones([2, 2, 4, 3], dtype=tf.float32),
        )


@pytest.mark.parametrize("helper", ["qr", "cholesky", "factor_covariance"])
def test_vectorized_helpers_reuse_one_dynamic_parameter_trace(helper: str) -> None:
    if helper == "qr":
        @tf.function(
            jit_compile=False,
            input_signature=[
                tf.TensorSpec([2, 3, 5], tf.float32),
                tf.TensorSpec([2, None, 3, 5], tf.float32),
            ],
        )
        def wrapped(first: tf.Tensor, derivative: tf.Tensor):
            return derivatives_tf._batched_stack_qr_lower_factor_first_derivatives(
                first, derivative
            )

        make_inputs = lambda p: _qr_inputs(tf.float32, 2, p)
        expected_shapes = lambda p: ([2, 3, 3], [2, p, 3, 3], [2])
    elif helper == "cholesky":
        @tf.function(
            jit_compile=False,
            input_signature=[
                tf.TensorSpec([2, 3, 3], tf.float32),
                tf.TensorSpec([2, None, 3, 3], tf.float32),
            ],
        )
        def wrapped(first: tf.Tensor, derivative: tf.Tensor):
            return derivatives_tf._batched_cholesky_factor_first_derivatives(
                first, derivative
            )

        make_inputs = lambda p: _cholesky_inputs(tf.float32, 2, p)
        expected_shapes = lambda p: ([2, 3, 3], [2, p, 3, 3])
    else:
        @tf.function(
            jit_compile=False,
            input_signature=[
                tf.TensorSpec([2, 3, 3], tf.float32),
                tf.TensorSpec([2, None, 3, 3], tf.float32),
            ],
        )
        def wrapped(first: tf.Tensor, derivative: tf.Tensor):
            return derivatives_tf._batched_factor_covariance_first_derivatives(
                first, derivative
            )

        make_inputs = lambda p: _factor_inputs(tf.float32, 2, p)
        expected_shapes = lambda p: ([2, 3, 3], [2, p, 3, 3])

    for parameter_count in (1, 4):
        outputs = wrapped(*make_inputs(parameter_count))
        assert tuple(tensor.shape.as_list() for tensor in outputs) == expected_shapes(
            parameter_count
        )
    assert len(wrapped._list_all_concrete_functions_for_serialization()) == 1


def test_batched_first_derivative_call_graph_has_no_parameter_loop_or_mapping() -> None:
    names = {
        "_batched_stack_qr_lower_factor_first_derivatives",
        "_batched_qr_factor_derivative",
        "_batched_right_solve_upper",
        "_batched_omega_from_a",
        "_batched_cholesky_factor_first_derivatives",
        "_batched_factor_covariance_first_derivatives",
        "_batched_symmetrize",
    }
    module_tree = ast.parse(inspect.getsource(derivatives_tf))
    functions = {
        node.name: node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    assert set(functions) == names
    forbidden_nodes = (
        ast.For,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    for function in functions.values():
        assert not any(isinstance(node, forbidden_nodes) for node in ast.walk(function))
        source = ast.unparse(function)
        assert "_static_dim" not in source
        assert "tf.map_fn" not in source
        assert "tf.vectorized_map" not in source
        assert "tf.numpy_function" not in source
        assert "tf_qr_sqrt_kalman_score" not in source


def test_batched_score_source_contract_has_no_scalar_wrapper_or_mapping() -> None:
    source = inspect.getsource(
        derivatives_tf.tf_qr_sqrt_kalman_score_batched_static.python_function
    )
    assert "tf_qr_sqrt_kalman_score(" not in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.numpy_function" not in source


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_phase2_fixture_batch_native_score_matches_scalar_rows(
    dtype: tf.DType, batch_size: int
) -> None:
    benchmark = _load_benchmark()
    fixture = benchmark.make_fixture(3, 3, 4, dtype=dtype)
    parameters_batch = benchmark._make_parameter_batch(fixture, batch_size)
    batch_fn = benchmark.build_batch_native_analytic_fn(
        fixture, batch_size=batch_size, jit_compile=False
    )
    scalar_fn = benchmark.build_scalar_analytic_row_loop_fn(
        fixture, batch_size=batch_size, jit_compile=False
    )
    batch_value, batch_score = batch_fn(parameters_batch)
    scalar_value, scalar_score = scalar_fn(parameters_batch)

    assert batch_value.shape.as_list() == [batch_size]
    assert batch_score.shape.as_list() == [batch_size, 3]
    assert batch_value.dtype == batch_score.dtype == dtype
    assert bool(tf.reduce_all(tf.math.is_finite(batch_value)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(batch_score)).numpy())
    if dtype == tf.float32:
        rtol = atol = 2.0e-4
    else:
        rtol, atol = 1.0e-8, 1.0e-9
    np.testing.assert_allclose(
        batch_value.numpy(), scalar_value.numpy(), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        batch_score.numpy(), scalar_score.numpy(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_nonjit_analytical_score_matches_scalar_rows_with_distinct_axes(
    batch_size: int,
) -> None:
    benchmark = _load_benchmark()
    fixture = benchmark.make_fixture(2, 4, 4, dtype=tf.float64)
    parameters_batch = benchmark._make_parameter_batch(
        fixture, 4 if batch_size == 3 else batch_size
    )[:batch_size]
    batch_fn = benchmark.build_batch_native_analytic_fn(
        fixture, batch_size=batch_size, jit_compile=False
    )
    scalar_fn = benchmark.build_scalar_analytic_row_loop_fn(
        fixture, batch_size=batch_size, jit_compile=False
    )

    batch_value, batch_score = batch_fn(parameters_batch)
    scalar_value, scalar_score = scalar_fn(parameters_batch)

    assert batch_value.shape.as_list() == [batch_size]
    assert batch_score.shape.as_list() == [batch_size, 4]
    np.testing.assert_allclose(
        batch_value.numpy(), scalar_value.numpy(), rtol=1.0e-8, atol=1.0e-10
    )
    np.testing.assert_allclose(
        batch_score.numpy(), scalar_score.numpy(), rtol=1.0e-8, atol=1.0e-9
    )


def test_nonjit_float32_analytical_score_matches_reverse_mode_distinct_axes() -> None:
    benchmark = _load_benchmark()
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float32)
    parameters_batch = benchmark._make_parameter_batch(fixture, 4)
    analytical_fn = benchmark.build_batch_native_analytic_fn(
        fixture, batch_size=4, jit_compile=False
    )
    reverse_mode_fn = benchmark.build_autodiff_row_loop_fn(
        fixture, batch_size=4, jit_compile=False
    )

    analytical_value, analytical_score = analytical_fn(parameters_batch)
    reverse_value, reverse_score = reverse_mode_fn(parameters_batch)

    assert analytical_value.shape.as_list() == [4]
    assert analytical_score.shape.as_list() == [4, 3]
    assert analytical_value.dtype == analytical_score.dtype == tf.float32
    np.testing.assert_allclose(
        analytical_value.numpy(), reverse_value.numpy(), rtol=2.0e-4, atol=2.0e-4
    )
    np.testing.assert_allclose(
        analytical_score.numpy(), reverse_score.numpy(), rtol=2.0e-4, atol=2.0e-4
    )


def test_nonjit_batched_score_default_jitter_preserves_float32() -> None:
    benchmark = _load_benchmark()
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float32)
    parameters_batch = benchmark._make_parameter_batch(fixture, 4)

    value, score = _call_batched_score_python(
        benchmark, fixture, parameters_batch
    )

    assert value.shape.as_list() == [4]
    assert score.shape.as_list() == [4, 3]
    assert value.dtype == score.dtype == tf.float32


def test_nonjit_batched_score_rejects_missing_parameter_axis() -> None:
    benchmark = _load_benchmark()
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float64)
    parameters_batch = benchmark._make_parameter_batch(fixture, 4)
    tensors = list(benchmark._batched_model_tensors(fixture, parameters_batch))
    tensors[10] = tensors[10][:, 0, :]

    with pytest.raises(ValueError, match="d_transition_offset must have rank 3"):
        derivatives_tf.tf_qr_sqrt_kalman_score_batched_static.python_function(
            observations=fixture.observations,
            transition_offset=tensors[0],
            transition_matrix=tensors[1],
            transition_covariance=tensors[2],
            observation_offset=tensors[3],
            observation_matrix=tensors[4],
            observation_covariance=tensors[5],
            initial_state_mean=tensors[6],
            initial_state_covariance=tensors[7],
            d_initial_state_mean=tensors[8],
            d_initial_state_covariance=tensors[9],
            d_transition_offset=tensors[10],
            d_transition_matrix=tensors[11],
            d_transition_covariance=tensors[12],
            d_observation_offset=tensors[13],
            d_observation_matrix=tensors[14],
            d_observation_covariance=tensors[15],
            jitter=tf.constant(1.0e-9, dtype=tf.float64),
        )


def _passing_graph_rows() -> list[dict[str, object]]:
    source_checks = {
        "all_helpers_found": True,
        "no_python_loop_or_comprehension": True,
        "no_static_parameter_dim": True,
        "no_tensorflow_mapping": True,
        "no_scalar_score_call": True,
    }
    common = {
        "node_count": 42,
        "ordered_op_sequence_digest": "op-digest",
        "op_histogram": {"Placeholder": 1, "Identity": 2},
        "constant_count": 3,
        "source_structure_checks": source_checks,
    }
    return [
        {**common, "parameter_count": 50, "output_shapes": [[4], [4, 50]]},
        {**copy.deepcopy(common), "parameter_count": 150, "output_shapes": [[4], [4, 150]]},
    ]


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("node_count", "node_count_equal"),
        ("ordered_op", "ordered_op_sequence_equal"),
        ("op_histogram", "op_histogram_equal"),
        ("constant_count", "constant_count_equal"),
    ],
)
def test_phase3_graph_gate_fails_closed_for_structural_mutation(
    mutation: str, failed_check: str
) -> None:
    benchmark = _load_benchmark()
    rows = _passing_graph_rows()
    if mutation == "node_count":
        rows[1]["node_count"] = 43
    elif mutation == "ordered_op":
        rows[1]["ordered_op_sequence_digest"] = "changed"
    elif mutation == "op_histogram":
        rows[1]["op_histogram"]["Identity"] = 3
    else:
        rows[1]["constant_count"] = 4

    gate = benchmark._phase3_graph_gate(rows)
    assert gate["checks"][failed_check] is False
    assert gate["state"] == "failed"
    assert gate["returncode"] != 0


@pytest.mark.parametrize("rows", [[_passing_graph_rows()[0], _passing_graph_rows()[0]], list(reversed(_passing_graph_rows()))])
def test_phase3_graph_gate_requires_distinct_ordered_parameter_rows(rows) -> None:
    benchmark = _load_benchmark()
    gate = benchmark._phase3_graph_gate(rows)
    assert gate["checks"]["distinct_ordered_parameter_rows"] is False
    assert gate["state"] == "failed"
    assert gate["returncode"] != 0


def test_phase3_graph_trace_never_executes_score_callable() -> None:
    benchmark = _load_benchmark()
    fixture = benchmark.make_fixture(3, 3, 4, dtype=tf.float32)

    @tf.function(
        jit_compile=False,
        input_signature=[tf.TensorSpec([1, 3], tf.float32)],
    )
    def traced(parameters_batch: tf.Tensor):
        return tf.reduce_sum(parameters_batch, axis=1), parameters_batch

    class TraceOnlyScore:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("score callable must not execute")

        def get_concrete_function(self, *_args, **_kwargs):
            return traced.get_concrete_function()

    row = benchmark._phase3_trace_score_graph(
        fixture,
        batch_size=1,
        build_fn=lambda *_args, **_kwargs: TraceOnlyScore(),
    )
    assert row["parameter_count"] == 3
    assert row["output_shapes"] == [[1], [1, 3]]


def test_phase3_diagnostic_does_not_enumerate_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _load_benchmark()
    rows = _passing_graph_rows()
    monkeypatch.setattr(
        benchmark,
        "_phase3_trace_score_graph",
        lambda fixture, **_kwargs: copy.deepcopy(
            rows[0] if fixture.parameter_count == 50 else rows[1]
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("device enumeration must not run")

    monkeypatch.setattr(benchmark, "_select_device", forbidden)
    monkeypatch.setattr(benchmark.tf.config, "list_physical_devices", forbidden)
    monkeypatch.setattr(benchmark.tf.config, "list_logical_devices", forbidden)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    args = argparse.Namespace(
        device="cpu",
        jit_compile=False,
        cpu_threads=None,
        output_json=str(tmp_path / "phase3.json"),
        phase3_log_path=str(tmp_path / "phase3.log"),
    )
    assert benchmark.run_phase3_parameter_graph_diagnostic(args) == 0
    payload = benchmark.benchmark_contract.read_strict_json(tmp_path / "phase3.json")
    assert payload["state"] == "passed"
    assert all(payload["checks"].values())

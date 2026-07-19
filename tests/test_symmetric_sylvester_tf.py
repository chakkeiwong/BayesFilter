from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.ops.symmetric_sylvester_tf as symmetric_sylvester_tf
from bayesfilter.ops import SymmetricSylvesterStatus, symmetric_sylvester_solve
from bayesfilter.ops.symmetric_sylvester_tf import _validate_shapes


def _reference_solve(
    symmetric_factor: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    values = []
    for b in range(symmetric_factor.shape[0]):
        eigvals, eigvecs = np.linalg.eigh(symmetric_factor[b])
        rows = []
        for p in range(rhs.shape[1]):
            transformed = eigvecs.T @ rhs[b, p] @ eigvecs
            transformed = transformed / (eigvals[:, None] + eigvals[None, :])
            rows.append(eigvecs @ transformed @ eigvecs.T)
        values.append(rows)
    return np.asarray(values, dtype=np.float64)


def _fixture() -> tuple[tf.Tensor, tf.Tensor]:
    symmetric_factor = np.asarray(
        [
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            [
                [3.0, 0.25, 0.0],
                [0.25, 4.0, 0.1],
                [0.0, 0.1, 6.0],
            ],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray(
        [
            [
                [[1.0, 0.2, -0.1], [0.2, 0.5, 0.3], [-0.1, 0.3, 0.7]],
                [[0.4, -0.3, 0.2], [0.1, 0.8, -0.2], [0.0, 0.5, 0.6]],
            ],
            [
                [[0.7, -0.1, 0.4], [-0.1, 1.2, 0.2], [0.4, 0.2, 0.9]],
                [[1.1, 0.0, -0.3], [0.2, 0.6, 0.5], [-0.3, 0.5, 0.4]],
            ],
        ],
        dtype=np.float64,
    )
    return (
        tf.constant(symmetric_factor, dtype=tf.float64),
        tf.constant(rhs, dtype=tf.float64),
    )


def test_symmetric_sylvester_contract_reports_static_shape_metadata() -> None:
    status = _validate_shapes(
        tf.eye(3, batch_shape=[2], dtype=tf.float64),
        tf.zeros([2, 4, 3, 3], dtype=tf.float64),
    )

    assert isinstance(status, SymmetricSylvesterStatus)
    assert status.batch_size == 2
    assert status.parameter_count == 4
    assert status.dimension == 3
    assert status.implemented is True
    assert status.backend == "compiled_symmetric_sylvester"


def test_symmetric_sylvester_solve_matches_reference_and_residual() -> None:
    symmetric_factor, rhs = _fixture()
    actual = symmetric_sylvester_solve(symmetric_factor, rhs)
    expected = _reference_solve(symmetric_factor.numpy(), rhs.numpy())

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-11, atol=1.0e-11)
    residual = (
        symmetric_factor[:, tf.newaxis, :, :] @ actual
        + actual @ symmetric_factor[:, tf.newaxis, :, :]
        - rhs
    )
    assert float(tf.linalg.norm(residual).numpy()) <= 1.0e-10


def test_symmetric_sylvester_repeated_positive_eigenvalue_fixture_passes() -> None:
    symmetric_factor = 2.5 * tf.eye(4, batch_shape=[1], dtype=tf.float64)
    rhs = tf.reshape(
        tf.cast(tf.range(32), tf.float64) / 17.0,
        [1, 2, 4, 4],
    )
    actual = symmetric_sylvester_solve(symmetric_factor, rhs)

    np.testing.assert_allclose(actual.numpy(), (rhs / 5.0).numpy(), atol=1.0e-12)


def test_symmetric_sylvester_graph_and_cpu_xla_match_eager() -> None:
    symmetric_factor, rhs = _fixture()
    eager = symmetric_sylvester_solve(symmetric_factor, rhs)

    @tf.function(reduce_retracing=True)
    def graph_call() -> tf.Tensor:
        return symmetric_sylvester_solve(symmetric_factor, rhs)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_call() -> tf.Tensor:
        return symmetric_sylvester_solve(symmetric_factor, rhs)

    np.testing.assert_allclose(graph_call().numpy(), eager.numpy(), atol=1.0e-11)
    np.testing.assert_allclose(xla_call().numpy(), eager.numpy(), atol=1.0e-11)


def test_symmetric_sylvester_casts_to_float64() -> None:
    symmetric_factor = tf.eye(2, batch_shape=[1], dtype=tf.float32)
    rhs = tf.ones([1, 1, 2, 2], dtype=tf.float32)
    actual = symmetric_sylvester_solve(symmetric_factor, rhs)

    assert actual.dtype == tf.float64


def test_symmetric_sylvester_rejects_non_symmetric_precondition() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError, match="must be symmetric"):
        symmetric_sylvester_solve(
            tf.constant([[[2.0, 1.0], [0.0, 2.0]]], dtype=tf.float64),
            tf.ones([1, 1, 2, 2], dtype=tf.float64),
        )


def test_symmetric_sylvester_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown symmetric Sylvester backend"):
        symmetric_sylvester_solve(
            tf.eye(2, batch_shape=[1], dtype=tf.float64),
            tf.zeros([1, 1, 2, 2], dtype=tf.float64),
            backend="tensorflow_fallback",
        )


def test_symmetric_sylvester_contract_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="batch dimensions must match"):
        _validate_shapes(
            tf.eye(2, batch_shape=[2], dtype=tf.float64),
            tf.zeros([3, 1, 2, 2], dtype=tf.float64),
        )


def test_symmetric_sylvester_source_has_no_tensorflow_solve_fallbacks() -> None:
    source = inspect.getsource(symmetric_sylvester_tf)
    forbidden = (
        "tf.linalg.solve",
        "tf.vectorized_map",
        "pfor",
        "GradientTape",
        "custom_gradient",
    )
    for pattern in forbidden:
        assert pattern not in source


def test_xla_custom_calls_pin_c_row_major_operand_and_result_layouts() -> None:
    source = (
        Path(symmetric_sylvester_tf.__file__)
        .with_name("symmetric_sylvester_op.cc")
        .read_text(encoding="utf-8")
    )

    assert source.count("xla::CustomCallWithLayout(") == 2
    assert "{batch, parameters, n, n}, {3, 2, 1, 0}" in source
    assert source.count("{batch, n, n}, {2, 1, 0}") >= 3
    assert "const std::vector<xla::Shape> operand_layouts" in source


def test_symmetric_sylvester_cpu_only_hides_gpu() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert tf.config.list_physical_devices("GPU") == []

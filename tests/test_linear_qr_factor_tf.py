from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.linear import qr_factor_tf as qr


ROOT = Path(__file__).resolve().parents[1]


def _matrix_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260427)
    matrix = rng.normal(size=(5, 3))
    matrix[:, 0] += np.array([2.0, 0.2, -0.1, 0.3, 0.4])
    dmatrix = rng.normal(scale=0.08, size=(2, 5, 3))
    d2matrix = rng.normal(scale=0.03, size=(2, 2, 5, 3))
    d2matrix = 0.5 * (d2matrix + np.swapaxes(d2matrix, 0, 1))
    return (
        matrix.astype(np.float64),
        dmatrix.astype(np.float64),
        d2matrix.astype(np.float64),
    )


def test_qr_factor_derivative_identities_reconstruct_matrix_derivatives() -> None:
    matrix, dmatrix, d2matrix = _matrix_case()
    q, r, dq, dr, d2q, d2r = qr.qr_factor_full_derivatives(
        tf.convert_to_tensor(matrix, dtype=tf.float64),
        tf.convert_to_tensor(dmatrix, dtype=tf.float64),
        tf.convert_to_tensor(d2matrix, dtype=tf.float64),
    )

    np.testing.assert_allclose((q @ r).numpy(), matrix, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        (tf.transpose(q) @ q).numpy(),
        np.eye(matrix.shape[1]),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.all(np.diag(r.numpy()) > 0.0)
    np.testing.assert_allclose(
        tf.linalg.band_part(r, -1, 0).numpy() - np.diag(np.diag(r.numpy())),
        0.0,
        atol=1e-12,
    )

    for i in range(dmatrix.shape[0]):
        np.testing.assert_allclose(
            (dq[i] @ r + q @ dr[i]).numpy(),
            dmatrix[i],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            (tf.transpose(dq[i]) @ q + tf.transpose(q) @ dq[i]).numpy(),
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            tf.linalg.band_part(dr[i], -1, 0).numpy()
            - np.diag(np.diag(dr[i].numpy())),
            0.0,
            atol=1e-12,
        )
        for j in range(dmatrix.shape[0]):
            reconstructed = d2q[i, j] @ r + dq[i] @ dr[j] + dq[j] @ dr[i] + q @ d2r[i, j]
            np.testing.assert_allclose(
                reconstructed.numpy(),
                d2matrix[i, j],
                rtol=1e-10,
                atol=1e-10,
            )
            second_orthogonality = (
                tf.transpose(d2q[i, j]) @ q
                + tf.transpose(dq[i]) @ dq[j]
                + tf.transpose(dq[j]) @ dq[i]
                + tf.transpose(q) @ d2q[i, j]
            )
            np.testing.assert_allclose(
                second_orthogonality.numpy(),
                0.0,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                tf.linalg.band_part(d2r[i, j], -1, 0).numpy()
                - np.diag(np.diag(d2r[i, j].numpy())),
                0.0,
                atol=1e-12,
            )


def test_stack_lower_factor_derivatives_reconstruct_covariance() -> None:
    rng = np.random.default_rng(6027)
    stack = rng.normal(size=(3, 7)).astype(np.float64)
    dstack = rng.normal(scale=0.04, size=(2, 3, 7)).astype(np.float64)
    d2stack = rng.normal(scale=0.02, size=(2, 2, 3, 7)).astype(np.float64)
    d2stack = 0.5 * (d2stack + np.swapaxes(d2stack, 0, 1))

    factor, dfactor, d2factor, min_pivot = qr.stack_qr_lower_factor_derivatives(
        tf.convert_to_tensor(stack, dtype=tf.float64),
        tf.convert_to_tensor(dstack, dtype=tf.float64),
        tf.convert_to_tensor(d2stack, dtype=tf.float64),
    )
    covariance, dcovariance, d2covariance = qr.factor_covariance_derivatives(
        factor,
        dfactor,
        d2factor,
    )
    stack_covariance, dstack_covariance, d2stack_covariance = (
        qr.stack_covariance_derivatives(
            tf.convert_to_tensor(stack, dtype=tf.float64),
            tf.convert_to_tensor(dstack, dtype=tf.float64),
            tf.convert_to_tensor(d2stack, dtype=tf.float64),
        )
    )

    assert float(min_pivot.numpy()) > 0.0
    np.testing.assert_allclose(covariance.numpy(), stack_covariance.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dcovariance.numpy(), dstack_covariance.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(d2covariance.numpy(), d2stack_covariance.numpy(), rtol=1e-10, atol=1e-10)


def test_cholesky_factor_derivatives_reconstruct_covariance() -> None:
    rng = np.random.default_rng(9027)
    raw = rng.normal(size=(3, 3))
    covariance = raw @ raw.T + 0.2 * np.eye(3)
    dcovariance = rng.normal(scale=0.03, size=(2, 3, 3))
    dcovariance = 0.5 * (dcovariance + np.swapaxes(dcovariance, -1, -2))
    d2covariance = rng.normal(scale=0.02, size=(2, 2, 3, 3))
    d2covariance = 0.5 * (d2covariance + np.swapaxes(d2covariance, 0, 1))
    d2covariance = 0.5 * (d2covariance + np.swapaxes(d2covariance, -1, -2))

    factor, dfactor, d2factor = qr.cholesky_factor_derivatives(
        tf.convert_to_tensor(covariance, dtype=tf.float64),
        tf.convert_to_tensor(dcovariance, dtype=tf.float64),
        tf.convert_to_tensor(d2covariance, dtype=tf.float64),
        jitter=1e-9,
    )
    reconstructed, dreconstructed, d2reconstructed = qr.factor_covariance_derivatives(
        factor,
        dfactor,
        d2factor,
    )

    expected_covariance = covariance + 1e-9 * np.eye(covariance.shape[0])
    np.testing.assert_allclose(reconstructed.numpy(), expected_covariance, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dreconstructed.numpy(), dcovariance, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(d2reconstructed.numpy(), d2covariance, rtol=1e-10, atol=1e-10)


def test_qr_factor_module_does_not_import_numpy_or_call_dot_numpy() -> None:
    text = (ROOT / "bayesfilter" / "linear" / "qr_factor_tf.py").read_text(
        encoding="utf-8"
    )

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert ".numpy(" not in text

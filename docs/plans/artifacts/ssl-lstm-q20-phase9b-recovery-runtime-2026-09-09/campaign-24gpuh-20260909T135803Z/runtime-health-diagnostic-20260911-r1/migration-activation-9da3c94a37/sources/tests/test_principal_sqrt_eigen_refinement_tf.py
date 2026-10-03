"""Independent eigenpair and Phase 9B saved-endpoint repair regressions."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    BatchNativeSSLLSTMComplexityPosteriorTarget,
)


@pytest.mark.parametrize("dimension", (1, 2, 3, 4, 5, 80))
def test_refined_eigh_preserves_eigenpairs_and_indefinite_rows(dimension):
    generator = np.random.default_rng(732)
    basis, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
    spectra = np.stack((
        np.linspace(0.1, 2.0, dimension),
        np.concatenate(([-0.2], np.ones(dimension - 1))),
        np.concatenate(([1e-13], np.ones(dimension - 1))),
        np.zeros(dimension),
    ))
    matrix = (basis[None] * spectra[:, None, :]) @ basis.T

    @tf.function(
        input_signature=(tf.TensorSpec((4, dimension, dimension), tf.float64),),
        jit_compile=True,
        autograph=False,
    )
    def evaluate(batch):
        return core._refined_symmetric_eigh(batch)

    values, vectors = evaluate(tf.constant(matrix, tf.float64))
    np.testing.assert_allclose(values.numpy(), spectra, atol=5e-14, rtol=1e-12)
    np.testing.assert_allclose(
        matrix @ vectors.numpy(),
        vectors.numpy() * values.numpy()[:, None, :],
        atol=5e-14,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.swapaxes(vectors.numpy(), -1, -2) @ vectors.numpy(),
        np.broadcast_to(np.eye(dimension), (4, dimension, dimension)),
        atol=5e-14,
        rtol=1e-12,
    )
    assert values.numpy()[1, 0] < 0.0
    assert values.numpy()[2, 0] > 0.0
    assert evaluate.experimental_get_tracing_count() == 1


def test_refined_eigh_does_not_send_nonfinite_rows_to_backend(monkeypatch):
    original = tf.linalg.eigh

    def finite_only(matrix):
        tf.debugging.assert_all_finite(matrix, "nonfinite eigensolver input")
        return original(matrix)

    monkeypatch.setattr(tf.linalg, "eigh", finite_only)
    matrix = tf.constant([[[1.0, 0.0], [0.0, float("nan")]], [[2.0, 0.0], [0.0, 3.0]]], tf.float64)
    values, vectors = core._refined_symmetric_eigh(matrix)
    assert np.isnan(values.numpy()[0]).all()
    assert np.isnan(vectors.numpy()[0]).all()
    np.testing.assert_array_equal(values.numpy()[1], [2.0, 3.0])


def test_refined_eigh_fails_closed_when_sweeps_do_not_converge(monkeypatch):
    def identity_warm_start(matrix):
        return tf.linalg.diag_part(matrix), tf.eye(6, batch_shape=[1], dtype=tf.float64)

    monkeypatch.setattr(tf.linalg, "eigh", identity_warm_start)
    generator = np.random.default_rng(281)
    raw = generator.normal(size=(6, 6))
    matrix = tf.constant((raw + raw.T)[None], tf.float64)
    values, vectors = core._refined_symmetric_eigh(matrix, sweeps=1)
    assert np.isnan(values.numpy()).all()
    assert np.isnan(vectors.numpy()).all()


@pytest.mark.parametrize("backend", ("tensorflow_eigh_strict", "tensorflow_eigh_strict_factor_cached"))
def test_phase9b_false_indefiniteness_endpoint_matches_cpu_reference(backend):
    """Reference: Sept 11 CPU full-filter diagnostic on the saved L=2 batch."""
    parameters = tf.constant([
        [1.1484269960938254, 1.6302049670650114, 0.2728879033393473, -0.2921336884233377],
        [0.08715895549025965, 1.868557025097924, 0.7010506219942324, -0.7659174446368148],
        [2.9502279704890193, -2.4878301002769323, 0.6704492567780896, 1.7079457777564355],
        [-3.6130528940902975, -0.2409610860997648, 0.5636597915347656, 0.7942885593780721],
    ], tf.float64)
    target = BatchNativeSSLLSTMComplexityPosteriorTarget(
        20, jit_compile=True, principal_sqrt_backend=backend,
    )

    @tf.function(
        input_signature=(tf.TensorSpec((4, 4), tf.float64),),
        jit_compile=True,
        autograph=False,
    )
    def evaluate(batch):
        model, derivatives = target._batched_components(batch)
        value, score, diagnostics = core.tf_batched_svd_sigma_point_value_and_score(
            target.config.observations, model, derivatives,
            backend="tf_principal_sqrt_ukf", principal_sqrt_backend=backend,
        )
        return value, score, diagnostics["principal_sqrt_target_row_class_code"]

    value, score, status = evaluate(parameters)
    np.testing.assert_array_equal(status.numpy(), [0, 0, 0, 0])
    np.testing.assert_allclose(value.numpy(), [
        -40.0635752261776, -39.18775232288809, -38.29379045186498, -40.84682704892044,
    ], atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(score.numpy(), [
        [0.0057993722752186765, 2.100132188497193, 24.928455611530694, 7.732578670504037],
        [0.14607655320541965, -6.890267291908506, -20.26853281002946, -9.844038816673516],
        [0.10090856864541123, 4.29324407749395, -18.046254351048812, 6.511040242538104],
        [-0.06050664372010636, -9.60630665614951, 10.622477313512244, -16.390077744314805],
    ], atol=1e-9, rtol=1e-10)

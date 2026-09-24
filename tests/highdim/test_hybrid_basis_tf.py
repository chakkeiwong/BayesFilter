"""Independent checks for the Gaussian Hermite/RBF hybrid basis.

NumPy is used only for diagnostic Gauss--Hermite quadrature.  TensorFlow is
the implementation and contraction backend.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import DensityMeasure, MassMeasure, MeasureConvention
from bayesfilter.highdim.hybrid_basis_tf import HermiteRBFBasis1D


def _normal_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    return nodes, weights / math.sqrt(2.0 * math.pi)


def test_full_mass_integral_and_cross_block_match_quadrature() -> None:
    basis = HermiteRBFBasis1D(6, centers=(-2.0, -1.0, 0.0, 1.0, 2.0), widths=1.5)
    nodes, weights = _normal_quadrature(100)
    values = basis.evaluate(tf.constant(nodes, tf.float64)).numpy()
    expected_mass = (values * weights[:, None]).T @ values
    expected_integral = weights @ values
    np.testing.assert_allclose(
        basis.mass_matrix(MassMeasure.REFERENCE_MEASURE).numpy(),
        expected_mass,
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        basis.integral_vector(MassMeasure.REFERENCE_MEASURE).numpy(),
        expected_integral,
        rtol=3e-13,
        atol=3e-13,
    )
    cross = basis.cross_mass_matrix().numpy()
    expected_cross = (values[:, :7] * weights[:, None]).T @ values[:, 7:]
    np.testing.assert_allclose(cross, expected_cross, rtol=3e-13, atol=3e-13)


def test_tilted_recurrence_first_cases_and_spd() -> None:
    basis = HermiteRBFBasis1D(4, centers=(-1.0, 0.5), widths=(0.75, 3.0))
    nodes, weights = _normal_quadrature(100)
    values = basis.evaluate(tf.constant(nodes, tf.float64)).numpy()
    expected = (values * weights[:, None]).T @ values
    actual = basis.mass_matrix(MassMeasure.REFERENCE_MEASURE).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    assert np.min(np.linalg.eigvalsh(actual)) > 0.0
    cross = basis.cross_mass_matrix().numpy()
    # The degree-one cross identity is E[U phi] = mu E[phi].
    centers = np.asarray([-1.0, 0.5])
    widths = np.asarray([0.75, 3.0])
    integrals = basis.integral_vector(MassMeasure.REFERENCE_MEASURE).numpy()[5:]
    means = centers / (1.0 + widths**2)
    np.testing.assert_allclose(cross[1], means * integrals, rtol=3e-13, atol=3e-13)


def test_derivative_and_constant_uniqueness() -> None:
    basis = HermiteRBFBasis1D(6, centers=(-1.5, 0.0, 1.5), widths=1.5)
    points = tf.constant([-1.1, -0.2, 0.4, 1.0], tf.float64)
    step = tf.constant(1.0e-6, tf.float64)
    finite_difference = (basis.evaluate(points + step) - basis.evaluate(points - step)) / (2.0 * step)
    tf.debugging.assert_near(basis.derivative(points), finite_difference, atol=3e-8, rtol=3e-8)
    payload = basis.manifest_payload()
    assert payload["hermite_constant"] is True
    assert payload["rbf_constant"] is False
    assert basis.basis_dim == 10


def test_product_basis_and_measure_refusal() -> None:
    basis = HermiteRBFBasis1D(3, centers=(-1.0, 1.0), widths=1.0)
    convention = MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="standard_normal",
        physical_coordinate_name="x",
        reference_coordinate_name="u",
    )
    product = ProductBasis([basis, basis], convention)
    assert product.basis_dim_tuple() == (6, 6)
    with pytest.raises(ValueError):
        basis.mass_matrix(MassMeasure.REFERENCE_LEBESGUE)
    with pytest.raises(ValueError):
        basis.integral_vector(MassMeasure.REFERENCE_LEBESGUE)


def test_invalid_setup_fails_closed() -> None:
    with pytest.raises(ValueError):
        HermiteRBFBasis1D(2, centers=(), widths=1.0)
    with pytest.raises(ValueError):
        HermiteRBFBasis1D(2, centers=(0.0, 1.0), widths=(1.0,))
    with pytest.raises(ValueError):
        HermiteRBFBasis1D(2, centers=(0.0,), widths=0.0)

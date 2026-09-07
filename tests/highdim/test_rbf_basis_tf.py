"""Focused Gaussian-reference RBF basis checks.

NumPy is used only for independent diagnostic quadrature, not by the runtime
basis implementation.  The tests intentionally hide GPUs because they are
small CPU reference checks.
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
from bayesfilter.highdim.rbf_basis_tf import RBFBasis1D


def _normal_quadrature(order: int = 100) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    return nodes, weights / math.sqrt(2.0 * math.pi)


def test_analytic_mass_and_integral_match_independent_quadrature() -> None:
    basis = RBFBasis1D(centers=(-2.0, -1.0, 0.0, 1.0, 2.0), widths=1.5)
    nodes, weights = _normal_quadrature()
    values = basis.evaluate(tf.constant(nodes, tf.float64)).numpy()
    expected_mass = (values * weights[:, None]).T @ values
    expected_integral = weights @ values
    np.testing.assert_allclose(
        basis.mass_matrix(MassMeasure.REFERENCE_MEASURE).numpy(),
        expected_mass,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        basis.integral_vector(MassMeasure.REFERENCE_MEASURE).numpy(),
        expected_integral,
        rtol=2e-13,
        atol=2e-13,
    )


def test_constant_channel_is_exact_and_mass_is_spd() -> None:
    basis = RBFBasis1D(centers=(-2.0, 0.0, 2.0), widths=(0.75, 1.5, 3.0))
    mass = basis.mass_matrix(MassMeasure.REFERENCE_MEASURE)
    integral = basis.integral_vector(MassMeasure.REFERENCE_MEASURE)
    tf.debugging.assert_near(mass[0, 0], tf.constant(1.0, tf.float64), atol=1e-15)
    tf.debugging.assert_near(integral[0], tf.constant(1.0, tf.float64), atol=1e-15)
    tf.debugging.assert_near(mass[0, 1:], integral[1:], atol=1e-15)
    tf.debugging.assert_near(mass[1:, 0], integral[1:], atol=1e-15)
    eigenvalues = tf.linalg.eigvalsh(mass)
    assert float(eigenvalues[0].numpy()) > 0.0


def test_derivative_matches_finite_difference() -> None:
    basis = RBFBasis1D(centers=(-1.5, 0.0, 1.5), widths=(0.75, 1.5, 3.0))
    points = tf.constant([-1.2, -0.3, 0.4, 1.1], tf.float64)
    step = tf.constant(1.0e-6, tf.float64)
    finite_difference = (
        basis.evaluate(points + step) - basis.evaluate(points - step)
    ) / (2.0 * step)
    tf.debugging.assert_near(
        basis.derivative(points), finite_difference, atol=2.0e-10, rtol=2.0e-10
    )


def test_no_constant_channel_and_product_basis_contract() -> None:
    basis = RBFBasis1D(centers=(0.0, 1.0), widths=1.0, include_constant=False)
    assert basis.basis_dim == 2
    assert basis.evaluate(tf.constant([0.0, 1.0], tf.float64)).shape == (2, 2)
    convention = MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="standard_normal",
        physical_coordinate_name="x",
        reference_coordinate_name="u",
    )
    product = ProductBasis([basis, basis], convention)
    assert product.basis_dim_tuple() == (2, 2)
    assert product.manifest_payload()["bases"][0]["family"] == "gaussian_rbf_reference"


def test_invalid_setup_and_measure_fail_closed() -> None:
    with pytest.raises(ValueError):
        RBFBasis1D(centers=(0.0,), widths=(0.0,))
    with pytest.raises(ValueError):
        RBFBasis1D(centers=(0.0, 1.0), widths=(1.0,))
    basis = RBFBasis1D(centers=(0.0,), widths=1.0)
    with pytest.raises(ValueError):
        basis.mass_matrix(MassMeasure.REFERENCE_LEBESGUE)
    with pytest.raises(ValueError):
        basis.integral_vector(MassMeasure.REFERENCE_LEBESGUE)

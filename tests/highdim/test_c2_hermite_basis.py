"""U-HERM-1: normalized Hermite basis unit checks.

C2 Gaussian-reference derivation note 2026-08-24, validation ladder
item 1. NumPy appears as diagnostic reference quadrature only (backend
rule). CPU-only diagnostic: GPU devices are intentionally hidden via
CUDA_VISIBLE_DEVICES=-1 set before the TensorFlow import.
"""

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.bases import (
    GaussianReferenceMeasure,
    HermiteBasis1D,
    ProductBasis,
    RealLine,
)
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)

DEGREE = 12


def _gauss_hermite_probabilists(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    return nodes, weights / math.sqrt(2.0 * math.pi)


def test_mass_identity_under_gaussian_reference() -> None:
    basis = HermiteBasis1D(max_degree=DEGREE)
    nodes, weights = _gauss_hermite_probabilists(60)
    values = basis.evaluate(tf.constant(nodes, tf.float64)).numpy()
    gram = (values * weights[:, None]).T @ values
    assert np.max(np.abs(gram - np.eye(DEGREE + 1))) < 5e-14
    exact = basis.mass_matrix(MassMeasure.REFERENCE_MEASURE).numpy()
    assert np.array_equal(exact, np.eye(DEGREE + 1))


def test_integral_vector_is_first_unit_vector() -> None:
    basis = HermiteBasis1D(max_degree=DEGREE)
    nodes, weights = _gauss_hermite_probabilists(60)
    values = basis.evaluate(tf.constant(nodes, tf.float64)).numpy()
    quad = weights @ values
    expected = np.zeros(DEGREE + 1)
    expected[0] = 1.0
    assert np.max(np.abs(quad - expected)) < 5e-14
    assert np.array_equal(
        basis.integral_vector(MassMeasure.REFERENCE_MEASURE).numpy(), expected
    )


def test_derivative_shift_identity_and_finite_difference() -> None:
    basis = HermiteBasis1D(max_degree=DEGREE)
    rng = np.random.default_rng(20260824)
    points = tf.constant(rng.normal(size=64), tf.float64)
    values = basis.evaluate(points).numpy()
    derivs = basis.derivative(points).numpy()
    for k in range(1, DEGREE + 1):
        exact = math.sqrt(k) * values[:, k - 1]
        assert np.max(np.abs(derivs[:, k] - exact)) < 1e-13
    assert np.max(np.abs(derivs[:, 0])) == 0.0
    step = 1e-6
    fd = (
        basis.evaluate(points + step).numpy() - basis.evaluate(points - step).numpy()
    ) / (2.0 * step)
    assert np.max(np.abs(fd - derivs)) < 1e-5


def test_reference_lebesgue_contraction_refused() -> None:
    basis = HermiteBasis1D(max_degree=DEGREE)
    with pytest.raises(ValueError):
        basis.mass_matrix(MassMeasure.REFERENCE_LEBESGUE)
    with pytest.raises(ValueError):
        basis.integral_vector(MassMeasure.REFERENCE_LEBESGUE)


def test_real_line_domain_exposes_no_box_volume() -> None:
    domain = RealLine()
    assert not hasattr(domain, "length")
    assert isinstance(
        HermiteBasis1D(max_degree=2).reference_measure, GaussianReferenceMeasure
    )


def test_product_basis_accepts_hermite_under_reference_convention() -> None:
    convention = MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="standard_normal",
        physical_coordinate_name="x",
        reference_coordinate_name="u",
    )
    product = ProductBasis(
        [HermiteBasis1D(max_degree=3), HermiteBasis1D(max_degree=3)], convention
    )
    assert product.dimension == 2
    assert product.basis_dim_tuple() == (4, 4)
    payload = product.manifest_payload()
    assert payload["basis_dim_tuple"] == (4, 4)

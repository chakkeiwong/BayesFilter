"""Hermite-plus-Gaussian-RBF basis under the standard-normal measure.

The hybrid is a generic one-dimensional representation component.  It keeps
one normalized Hermite constant channel and appends nonconstant Gaussian RBF
channels.  All evaluations and contractions use TensorFlow float64; the
closed cross-Gram formula follows from a tilted-normal generating function.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import HermiteBasis1D
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.rbf_basis_tf import RBFBasis1D


DTYPE = tf.float64
ROUTE_ID = "gaussian_hermite_rbf_reference_basis_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"


@dataclass(frozen=True)
class HermiteRBFBasis1D:
    """Normalized Hermite channels followed by nonconstant Gaussian RBFs.

    The Hermite block contains degrees ``0`` through ``max_degree``.  The RBF
    block intentionally has no constant channel, so the complete basis has
    exactly one copy of the constant function.
    """

    max_degree: int
    centers: tuple[float, ...]
    widths: tuple[float, ...]

    def __init__(
        self,
        max_degree: int,
        centers: Sequence[float],
        widths: Sequence[float] | float,
    ) -> None:
        degree = int(max_degree)
        if degree < 0:
            raise ValueError("max_degree must be nonnegative")
        center_values = tuple(float(value) for value in centers)
        if not center_values:
            raise ValueError("HermiteRBFBasis1D requires at least one center")
        if isinstance(widths, (int, float)):
            width_values = tuple(float(widths) for _ in center_values)
        else:
            width_values = tuple(float(value) for value in widths)
        if len(width_values) != len(center_values):
            raise ValueError("centers and widths must have equal length")
        if any(not math.isfinite(value) for value in center_values):
            raise ValueError("centers must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in width_values):
            raise ValueError("widths must be finite and positive")
        object.__setattr__(self, "max_degree", degree)
        object.__setattr__(self, "centers", center_values)
        object.__setattr__(self, "widths", width_values)

    @property
    def hermite_dim(self) -> int:
        return self.max_degree + 1

    @property
    def rbf_dim(self) -> int:
        return len(self.centers)

    @property
    def basis_dim(self) -> int:
        return self.hermite_dim + self.rbf_dim

    @property
    def dtype(self) -> tf.DType:
        return DTYPE

    def _hermite(self) -> HermiteBasis1D:
        return HermiteBasis1D(max_degree=self.max_degree)

    def _rbf(self) -> RBFBasis1D:
        return RBFBasis1D(self.centers, self.widths, include_constant=False)

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=DTYPE)
        return tf.concat([self._hermite().evaluate(values), self._rbf().evaluate(values)], axis=-1)

    def derivative(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=DTYPE)
        return tf.concat([self._hermite().derivative(values), self._rbf().derivative(values)], axis=-1)

    def _rbf_integral_vector(self) -> tf.Tensor:
        return self._rbf().integral_vector(MassMeasure.REFERENCE_MEASURE)

    def cross_mass_matrix(self) -> tf.Tensor:
        """Return ``C[k,i] = E_mu[psi_k(U) phi_i(U)]`` analytically."""

        centers = tf.constant(self.centers, dtype=DTYPE)
        widths = tf.constant(self.widths, dtype=DTYPE)
        inverse_width_squared = tf.math.reciprocal(tf.square(widths))
        precision = 1.0 + inverse_width_squared
        means = centers * inverse_width_squared / precision
        variances = tf.math.reciprocal(precision)
        integrals = tf.math.rsqrt(precision) * tf.exp(
            -0.5 * tf.square(centers) * inverse_width_squared / precision
        )

        # The generating function gives q[k+1] = mu*q[k] +
        # k*(variance-1)*q[k-1] for E[He_k(U)] under the tilted normal.
        rows = [tf.ones_like(means)]
        if self.max_degree >= 1:
            rows.append(means)
        for degree in range(1, self.max_degree):
            rows.append(
                means * rows[-1]
                + tf.cast(degree, DTYPE) * (variances - 1.0) * rows[-2]
            )
        raw = tf.stack(rows, axis=0)
        normalizers = tf.sqrt(
            tf.constant([float(math.factorial(k)) for k in range(self.hermite_dim)], DTYPE)
        )[:, tf.newaxis]
        return raw * integrals[tf.newaxis, :] / normalizers

    def mass_matrix(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        if measure is not MassMeasure.REFERENCE_MEASURE:
            raise ValueError(
                "HermiteRBFBasis1D contractions exist only under the "
                "standard-normal reference probability measure"
            )
        hermite_mass = tf.eye(self.hermite_dim, dtype=DTYPE)
        rbf_mass = self._rbf().mass_matrix(MassMeasure.REFERENCE_MEASURE)
        cross = self.cross_mass_matrix()
        top = tf.concat([hermite_mass, cross], axis=1)
        bottom = tf.concat([tf.transpose(cross), rbf_mass], axis=1)
        matrix = tf.concat([top, bottom], axis=0)
        return 0.5 * (matrix + tf.transpose(matrix))

    def integral_vector(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        if measure is not MassMeasure.REFERENCE_MEASURE:
            raise ValueError(
                "HermiteRBFBasis1D contractions exist only under the "
                "standard-normal reference probability measure"
            )
        hermite_integral = tf.concat(
            [tf.ones([1], dtype=DTYPE), tf.zeros([self.max_degree], dtype=DTYPE)], axis=0
        )
        return tf.concat([hermite_integral, self._rbf_integral_vector()], axis=0)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "gaussian_hermite_rbf_reference",
            "basis_dim": self.basis_dim,
            "hermite_degree": self.max_degree,
            "hermite_constant": True,
            "rbf_constant": False,
            "centers": self.centers,
            "widths": self.widths,
            "dtype": self.dtype.name,
            "reference_measure": "standard_normal_probability",
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
        }


HYBRID_BASIS_ROUTE_ID = ROUTE_ID
HYBRID_BASIS_ROUTE_CLASSIFICATION = ROUTE_CLASSIFICATION

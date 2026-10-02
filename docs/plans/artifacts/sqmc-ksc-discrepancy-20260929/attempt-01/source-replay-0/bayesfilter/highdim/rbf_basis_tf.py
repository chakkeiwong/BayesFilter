"""Gaussian RBF bases under the standard-normal reference measure.

This module is a generic one-dimensional basis implementation for the C2
representation ladder.  It contains no model or proposal logic.  Centers and
widths are setup-static, evaluations are TensorFlow float64 operations, and
the Gaussian-reference mass and integral contractions are analytic.  The
constant channel is optional and, when enabled, is included exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.diagnostics import MassMeasure


DTYPE = tf.float64
ROUTE_ID = "gaussian_rbf_reference_basis_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"


@dataclass(frozen=True)
class RBFBasis1D:
    """Fixed Gaussian RBF channels on the real line.

    The nonconstant channels are
    ``exp(-(u - center)**2 / (2 * width**2))``.  With ``include_constant``
    enabled, the first channel is the exact constant function one.  All
    setup values are stored as Python tuples so the basis identity is stable
    and serializable; numerical operations are performed by TensorFlow.
    """

    centers: tuple[float, ...]
    widths: tuple[float, ...]
    include_constant: bool = True

    def __init__(
        self,
        centers: Sequence[float],
        widths: Sequence[float] | float,
        include_constant: bool = True,
    ) -> None:
        center_values = tuple(float(value) for value in centers)
        if not center_values:
            raise ValueError("RBFBasis1D requires at least one center")
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
        object.__setattr__(self, "centers", center_values)
        object.__setattr__(self, "widths", width_values)
        object.__setattr__(self, "include_constant", bool(include_constant))

    @property
    def basis_dim(self) -> int:
        return len(self.centers) + int(self.include_constant)

    @property
    def dtype(self) -> tf.DType:
        return DTYPE

    def _center_tensor(self) -> tf.Tensor:
        return tf.constant(self.centers, dtype=DTYPE)

    def _width_tensor(self) -> tf.Tensor:
        return tf.constant(self.widths, dtype=DTYPE)

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=DTYPE)
        centers = self._center_tensor()
        widths = self._width_tensor()
        normalized = (values[..., tf.newaxis] - centers) / widths
        rbf = tf.exp(-0.5 * tf.square(normalized))
        if not self.include_constant:
            return rbf
        constant = tf.ones(tf.concat([tf.shape(values), [1]], axis=0), DTYPE)
        return tf.concat([constant, rbf], axis=-1)

    def derivative(self, points: tf.Tensor) -> tf.Tensor:
        """Return derivatives with respect to the scalar coordinate."""

        values = tf.convert_to_tensor(points, dtype=DTYPE)
        centers = self._center_tensor()
        widths = self._width_tensor()
        normalized = (values[..., tf.newaxis] - centers) / widths
        rbf = tf.exp(-0.5 * tf.square(normalized))
        derivatives = -normalized * rbf / widths
        if not self.include_constant:
            return derivatives
        constant = tf.zeros(tf.concat([tf.shape(values), [1]], axis=0), DTYPE)
        return tf.concat([constant, derivatives], axis=-1)

    def _rbf_integral_vector(self) -> tf.Tensor:
        centers = self._center_tensor()
        widths = self._width_tensor()
        inverse_width_squared = tf.math.reciprocal(tf.square(widths))
        precision = 1.0 + inverse_width_squared
        exponent = -0.5 * tf.square(centers) * inverse_width_squared / precision
        return tf.math.rsqrt(precision) * tf.exp(exponent)

    def _rbf_mass_matrix(self) -> tf.Tensor:
        centers = self._center_tensor()
        widths = self._width_tensor()
        inverse_width_squared = tf.math.reciprocal(tf.square(widths))
        precision = (
            tf.ones([len(self.centers), len(self.centers)], DTYPE)
            + inverse_width_squared[:, tf.newaxis]
            + inverse_width_squared[tf.newaxis, :]
        )
        linear = (
            centers[:, tf.newaxis] * inverse_width_squared[:, tf.newaxis]
            + centers[tf.newaxis, :] * inverse_width_squared[tf.newaxis, :]
        )
        constant = -0.5 * (
            tf.square(centers)[:, tf.newaxis] * inverse_width_squared[:, tf.newaxis]
            + tf.square(centers)[tf.newaxis, :] * inverse_width_squared[tf.newaxis, :]
        )
        exponent = constant + 0.5 * tf.square(linear) / precision
        matrix = tf.math.rsqrt(precision) * tf.exp(exponent)
        return 0.5 * (matrix + tf.transpose(matrix))

    def mass_matrix(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        if measure is not MassMeasure.REFERENCE_MEASURE:
            raise ValueError(
                "RBFBasis1D currently defines contractions only under the "
                "standard-normal reference probability measure"
            )
        rbf_mass = self._rbf_mass_matrix()
        if not self.include_constant:
            return rbf_mass
        integrals = self._rbf_integral_vector()
        top = tf.concat(
            [tf.ones([1, 1], DTYPE), integrals[tf.newaxis, :]], axis=1
        )
        bottom = tf.concat([integrals[:, tf.newaxis], rbf_mass], axis=1)
        return 0.5 * (tf.concat([top, bottom], axis=0) + tf.transpose(tf.concat([top, bottom], axis=0)))

    def integral_vector(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        if measure is not MassMeasure.REFERENCE_MEASURE:
            raise ValueError(
                "RBFBasis1D currently defines contractions only under the "
                "standard-normal reference probability measure"
            )
        integrals = self._rbf_integral_vector()
        if self.include_constant:
            return tf.concat([tf.ones([1], DTYPE), integrals], axis=0)
        return integrals

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "gaussian_rbf_reference",
            "basis_dim": self.basis_dim,
            "centers": self.centers,
            "widths": self.widths,
            "include_constant": bool(self.include_constant),
            "dtype": self.dtype.name,
            "reference_measure": "standard_normal_probability",
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
        }

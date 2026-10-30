"""Phase-0 diagnostics contracts for internal high-dimensional filters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import tensorflow as tf


class DensityMeasure(str, Enum):
    """Measure with respect to which a represented density is defined."""

    PHYSICAL_LEBESGUE = "PHYSICAL_LEBESGUE"
    REFERENCE_LEBESGUE = "REFERENCE_LEBESGUE"
    REFERENCE_MEASURE = "REFERENCE_MEASURE"


class MassMeasure(str, Enum):
    """Measure used by basis mass matrices and contractions."""

    REFERENCE_LEBESGUE = "REFERENCE_LEBESGUE"
    REFERENCE_MEASURE = "REFERENCE_MEASURE"


class HighDimStatus(str, Enum):
    """Deterministic status namespace for the high-dimensional lane."""

    OK = "OK"
    MEASURE_MISMATCH = "MEASURE_MISMATCH"
    REFERENCE_WEIGHT_MISSING = "REFERENCE_WEIGHT_MISSING"
    NONFINITE_VALUE = "NONFINITE_VALUE"
    INVALID_SHAPE = "INVALID_SHAPE"
    COMPLEXITY_GATE = "COMPLEXITY_GATE"
    BRANCH_HASH_MISSING = "BRANCH_HASH_MISSING"
    INVALID_BRANCH_MISMATCH = "INVALID_BRANCH_MISMATCH"
    SELECTIVE_BRANCH_HASH_REJECTED = "SELECTIVE_BRANCH_HASH_REJECTED"
    CDF_MONOTONICITY_FAILURE = "CDF_MONOTONICITY_FAILURE"
    INVERSE_BRACKET_FAILURE = "INVERSE_BRACKET_FAILURE"
    CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED = (
        "CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED"
    )
    NORMALIZER_FLOOR_EXCEEDED = "NORMALIZER_FLOOR_EXCEEDED"
    RETAINED_STORAGE_BUDGET_EXCEEDED = "RETAINED_STORAGE_BUDGET_EXCEEDED"
    RETAINED_MEASURE_MISMATCH = "RETAINED_MEASURE_MISMATCH"
    RETAINED_AXES_MISMATCH = "RETAINED_AXES_MISMATCH"
    UNSUPPORTED_MOVING_BASIS_DERIVATIVE = "UNSUPPORTED_MOVING_BASIS_DERIVATIVE"
    CONDITION_NUMBER_VETO = "CONDITION_NUMBER_VETO"
    HOLDOUT_RESIDUAL_VETO = "HOLDOUT_RESIDUAL_VETO"
    REPLAY_TAPE_MISMATCH = "REPLAY_TAPE_MISMATCH"
    REPLAY_CORE_HASH_MISMATCH = "REPLAY_CORE_HASH_MISMATCH"
    REPLAY_ENVIRONMENT_STALE = "REPLAY_ENVIRONMENT_STALE"
    NONFINITE_RETAINED_DERIVATIVE = "NONFINITE_RETAINED_DERIVATIVE"
    DERIVATIVE_SOLVE_FAILURE = "DERIVATIVE_SOLVE_FAILURE"
    FINITE_DIFFERENCE_BRANCH_MISMATCH = "FINITE_DIFFERENCE_BRANCH_MISMATCH"
    EXACT_REFERENCE_MISMATCH = "EXACT_REFERENCE_MISMATCH"
    STAGE_SANITY_ONLY = "STAGE_SANITY_ONLY"
    UNSUPPORTED_BACKEND = "UNSUPPORTED_BACKEND"


@dataclass(frozen=True)
class MeasureConvention:
    """Declared density and contraction measure convention."""

    density_measure: DensityMeasure
    mass_measure: MassMeasure
    reference_weight_name: str
    physical_coordinate_name: str = "r"
    reference_coordinate_name: str = "z"
    dtype_name: str = "float64"

    def __post_init__(self) -> None:
        if not isinstance(self.density_measure, DensityMeasure):
            raise TypeError("density_measure must be a DensityMeasure")
        if not isinstance(self.mass_measure, MassMeasure):
            raise TypeError("mass_measure must be a MassMeasure")
        if not str(self.reference_weight_name).strip():
            raise ValueError("reference_weight_name must be nonempty")
        if not str(self.physical_coordinate_name).strip():
            raise ValueError("physical_coordinate_name must be nonempty")
        if not str(self.reference_coordinate_name).strip():
            raise ValueError("reference_coordinate_name must be nonempty")
        if str(self.dtype_name) != "float64":
            raise ValueError("Phase-0 highdim contracts require float64")


@dataclass(frozen=True)
class HighDimValidationResult:
    """Immutable validation result for high-dimensional gate checks."""

    status: HighDimStatus
    message: str
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be a HighDimStatus")
        if not str(self.message).strip():
            raise ValueError("message must be nonempty")
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType({str(k): v for k, v in (self.diagnostics or {}).items()}),
        )


def assert_density_matches_mass(convention: MeasureConvention) -> None:
    """Reject density/mass pairings that would silently change the scalar."""

    if not isinstance(convention, MeasureConvention):
        raise TypeError("convention must be a MeasureConvention")
    density_measure = convention.density_measure
    mass_measure = convention.mass_measure
    if density_measure is DensityMeasure.REFERENCE_MEASURE:
        if mass_measure is not MassMeasure.REFERENCE_MEASURE:
            raise ValueError(HighDimStatus.MEASURE_MISMATCH.value)
        return
    if density_measure is DensityMeasure.REFERENCE_LEBESGUE:
        if mass_measure is not MassMeasure.REFERENCE_LEBESGUE:
            raise ValueError(HighDimStatus.MEASURE_MISMATCH.value)
        return
    raise ValueError(HighDimStatus.MEASURE_MISMATCH.value)


def assert_finite_tensor(name: str, tensor: tf.Tensor) -> None:
    """Require all entries of a TensorFlow tensor to be finite."""

    values = tf.convert_to_tensor(tensor)
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")


def assert_shape(name: str, tensor: tf.Tensor, expected_rank: int | None) -> None:
    """Validate static tensor rank when a rank is declared."""

    values = tf.convert_to_tensor(tensor)
    if expected_rank is not None and values.shape.rank != expected_rank:
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")


def assert_tf_float64(name: str, tensor: tf.Tensor) -> None:
    """Require TensorFlow float64 tensors for production highdim contracts."""

    values = tf.convert_to_tensor(tensor)
    if values.dtype != tf.float64:
        raise TypeError(f"{name}: expected tf.float64")


def freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable shallow string-key mapping."""

    return MappingProxyType({str(k): v for k, v in (values or {}).items()})

"""Tensor-preserving result containers for TensorFlow filter backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics
from bayesfilter.linear.dtypes_tf import as_float_tensor, common_floating_dtype
from bayesfilter.structural import FilterRunMetadata


def _to_tensor_or_none(
    value: Any | None,
    dtype: tf.DType | None = None,
) -> tf.Tensor | None:
    if value is None:
        return None
    if dtype is None:
        dtype = common_floating_dtype(value)
    return as_float_tensor(value, dtype)


def _freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    return MappingProxyType({str(key): value for key, value in values.items()})


@dataclass(frozen=True)
class TFFilterValueResult:
    """Scalar TF likelihood result with optional filtered-state tensors."""

    log_likelihood: tf.Tensor
    filtered_means: tf.Tensor | None
    filtered_covariances: tf.Tensor | None
    metadata: FilterRunMetadata
    diagnostics: TFFilterDiagnostics | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "log_likelihood",
            as_float_tensor(
                self.log_likelihood,
                common_floating_dtype(self.log_likelihood),
            ),
        )
        object.__setattr__(
            self,
            "filtered_means",
            _to_tensor_or_none(self.filtered_means),
        )
        object.__setattr__(
            self,
            "filtered_covariances",
            _to_tensor_or_none(self.filtered_covariances),
        )
        diagnostics = self.diagnostics
        if isinstance(diagnostics, TFFilterDiagnostics):
            frozen_diagnostics: TFFilterDiagnostics | Mapping[str, Any] = diagnostics
        else:
            frozen_diagnostics = _freeze_mapping(diagnostics)
        object.__setattr__(self, "diagnostics", frozen_diagnostics)


@dataclass(frozen=True)
class TFFilterDerivativeResult:
    """TF likelihood, score, and optional Hessian result."""

    log_likelihood: tf.Tensor
    score: tf.Tensor
    hessian: tf.Tensor | None
    metadata: FilterRunMetadata
    diagnostics: TFFilterDiagnostics | Mapping[str, Any] | None = None
    trace: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        dtype = common_floating_dtype(self.log_likelihood, self.score, self.hessian)
        object.__setattr__(
            self,
            "log_likelihood",
            as_float_tensor(self.log_likelihood, dtype),
        )
        object.__setattr__(
            self,
            "score",
            as_float_tensor(self.score, dtype),
        )
        object.__setattr__(self, "hessian", _to_tensor_or_none(self.hessian, dtype))
        diagnostics = self.diagnostics
        if isinstance(diagnostics, TFFilterDiagnostics):
            frozen_diagnostics: TFFilterDiagnostics | Mapping[str, Any] = diagnostics
        else:
            frozen_diagnostics = _freeze_mapping(diagnostics)
        object.__setattr__(self, "diagnostics", frozen_diagnostics)
        if self.trace is None:
            frozen_trace: tuple[Mapping[str, Any], ...] | None = None
        else:
            frozen_trace = tuple(_freeze_mapping(row) for row in self.trace)
        object.__setattr__(self, "trace", frozen_trace)

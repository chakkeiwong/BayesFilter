"""Tensor-preserving result containers for TensorFlow filter backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics
from bayesfilter.structural import FilterRunMetadata


def _to_tensor_or_none(value: Any | None) -> tf.Tensor | None:
    if value is None:
        return None
    return tf.convert_to_tensor(value, dtype=tf.float64)


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
            tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64),
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
        object.__setattr__(
            self,
            "log_likelihood",
            tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64),
        )
        object.__setattr__(
            self,
            "score",
            tf.convert_to_tensor(self.score, dtype=tf.float64),
        )
        object.__setattr__(self, "hessian", _to_tensor_or_none(self.hessian))
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

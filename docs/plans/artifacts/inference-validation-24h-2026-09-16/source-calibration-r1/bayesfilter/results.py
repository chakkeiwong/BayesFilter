"""Shared result and diagnostics containers for filter backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from bayesfilter.structural import FilterRunMetadata


def _freeze_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    array.setflags(write=False)
    return array


def _freeze_optional_array(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    return _freeze_array(value)


def _freeze_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if diagnostics is None:
        return MappingProxyType({})
    frozen: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, np.ndarray):
            frozen[str(key)] = _freeze_array(value)
        elif isinstance(value, list):
            frozen[str(key)] = tuple(value)
        elif isinstance(value, dict):
            frozen[str(key)] = MappingProxyType(dict(value))
        else:
            frozen[str(key)] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class FilterValueResult:
    """Scalar likelihood result with optional filtered-state output."""

    log_likelihood: float
    filtered_means: np.ndarray | None
    filtered_covariances: np.ndarray | None
    metadata: FilterRunMetadata
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_likelihood", float(self.log_likelihood))
        object.__setattr__(
            self,
            "filtered_means",
            _freeze_optional_array(self.filtered_means),
        )
        object.__setattr__(
            self,
            "filtered_covariances",
            _freeze_optional_array(self.filtered_covariances),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_diagnostics(self.diagnostics),
        )


@dataclass(frozen=True)
class FilterDerivativeResult:
    """Likelihood, score, and optional Hessian result."""

    log_likelihood: float
    score: np.ndarray
    hessian: np.ndarray | None
    metadata: FilterRunMetadata
    diagnostics: Mapping[str, Any] | None = None
    trace: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_likelihood", float(self.log_likelihood))
        object.__setattr__(self, "score", _freeze_array(self.score))
        object.__setattr__(self, "hessian", _freeze_optional_array(self.hessian))
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_diagnostics(self.diagnostics),
        )
        if self.trace is None:
            frozen_trace: tuple[Mapping[str, Any], ...] | None = None
        else:
            frozen_trace = tuple(_freeze_diagnostics(row) for row in self.trace)
        object.__setattr__(self, "trace", frozen_trace)

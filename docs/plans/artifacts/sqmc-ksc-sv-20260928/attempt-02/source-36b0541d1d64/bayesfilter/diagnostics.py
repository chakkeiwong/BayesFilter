"""Shared diagnostics contracts for TensorFlow filtering backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

RegularizationTarget = Literal[
    "none",
    "pre_regularized_law",
    "implemented_regularized_law",
    "blocked",
]


def _freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    return MappingProxyType({str(key): value for key, value in values.items()})


@dataclass(frozen=True)
class TFRegularizationDiagnostics:
    """Metadata describing the covariance law actually implemented in TF."""

    jitter: Any = 0.0
    singular_floor: Any = 0.0
    floor_count: Any = 0
    psd_projection_residual: Any = 0.0
    implemented_covariance: Any | None = None
    branch_label: str = "none"
    derivative_target: RegularizationTarget = "none"

    def __post_init__(self) -> None:
        if self.derivative_target not in {
            "none",
            "pre_regularized_law",
            "implemented_regularized_law",
            "blocked",
        }:
            raise ValueError(f"unsupported derivative_target: {self.derivative_target}")
        if not str(self.branch_label).strip():
            raise ValueError("branch_label must be nonempty")

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "jitter": self.jitter,
                "singular_floor": self.singular_floor,
                "floor_count": self.floor_count,
                "psd_projection_residual": self.psd_projection_residual,
                "implemented_covariance": self.implemented_covariance,
                "branch_label": self.branch_label,
                "derivative_target": self.derivative_target,
            }
        )


@dataclass(frozen=True)
class TFFilterDiagnostics:
    """Serializable diagnostic envelope for TF filter results."""

    backend: str
    mask_convention: str = "none"
    regularization: TFRegularizationDiagnostics = TFRegularizationDiagnostics()
    extra: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not str(self.backend).strip():
            raise ValueError("backend must be nonempty")
        if not str(self.mask_convention).strip():
            raise ValueError("mask_convention must be nonempty")
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))

    def as_dict(self) -> Mapping[str, Any]:
        values: dict[str, Any] = {
            "backend": self.backend,
            "mask_convention": self.mask_convention,
            "regularization": self.regularization.as_dict(),
        }
        values.update(dict(self.extra or {}))
        return MappingProxyType(values)

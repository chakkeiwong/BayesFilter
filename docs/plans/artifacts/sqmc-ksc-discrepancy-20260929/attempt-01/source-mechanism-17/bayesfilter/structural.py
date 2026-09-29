"""Structural state-space metadata contracts.

This module is intentionally small and dependency-light.  It records the
metadata that filtering backends need before they decide whether a model can be
handled as an exact linear Gaussian system, a structural nonlinear system, or a
labeled approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


IntegrationSpace = Literal["innovation", "stochastic_state", "full_state"]
CompletionPolicy = Literal["required", "none", "approximate"]


def _as_tuple(values: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class StatePartition:
    """Declared roles for latent state coordinates.

    The partition is source metadata.  It is not inferred solely from a process
    covariance matrix, since small numerical nuggets and zero impact rows do not
    by themselves define structural timing.
    """

    state_names: tuple[str, ...]
    stochastic_indices: tuple[int, ...]
    deterministic_indices: tuple[int, ...]
    auxiliary_indices: tuple[int, ...] = ()
    innovation_dim: int = 0
    external_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "stochastic_indices", _as_tuple(self.stochastic_indices))
        object.__setattr__(
            self, "deterministic_indices", _as_tuple(self.deterministic_indices)
        )
        object.__setattr__(self, "auxiliary_indices", _as_tuple(self.auxiliary_indices))
        object.__setattr__(self, "external_indices", _as_tuple(self.external_indices))
        object.__setattr__(self, "innovation_dim", int(self.innovation_dim))
        self.validate()

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def stochastic_dim(self) -> int:
        return len(self.stochastic_indices)

    @property
    def deterministic_dim(self) -> int:
        return len(self.deterministic_indices)

    @property
    def auxiliary_dim(self) -> int:
        return len(self.auxiliary_indices)

    @property
    def is_mixed(self) -> bool:
        return self.stochastic_dim > 0 and (
            self.deterministic_dim > 0 or self.auxiliary_dim > 0
        )

    def validate(self) -> None:
        if self.state_dim == 0:
            raise ValueError("state_names must not be empty")
        if len(set(self.state_names)) != self.state_dim:
            raise ValueError("state_names must be unique")
        if self.innovation_dim < 0:
            raise ValueError("innovation_dim must be nonnegative")
        if self.stochastic_dim > 0 and self.innovation_dim <= 0:
            raise ValueError("innovation_dim must be positive for stochastic models")

        groups = {
            "stochastic": self.stochastic_indices,
            "deterministic": self.deterministic_indices,
            "auxiliary": self.auxiliary_indices,
            "external": self.external_indices,
        }
        seen: dict[int, str] = {}
        for role, indices in groups.items():
            if len(set(indices)) != len(indices):
                raise ValueError(f"{role}_indices contain duplicates")
            for index in indices:
                if index < 0 or index >= self.state_dim:
                    raise ValueError(
                        f"{role}_indices contain out-of-range index {index}"
                    )
                if index in seen:
                    raise ValueError(
                        f"state index {index} appears in both {seen[index]} and {role}"
                    )
                seen[index] = role

        expected = set(range(self.state_dim))
        covered = set(seen)
        if covered != expected:
            missing = tuple(sorted(expected - covered))
            raise ValueError(f"state partition does not cover indices {missing}")

    def role_of(self, index: int) -> str:
        if index in self.stochastic_indices:
            return "stochastic"
        if index in self.deterministic_indices:
            return "deterministic"
        if index in self.auxiliary_indices:
            return "auxiliary"
        if index in self.external_indices:
            return "external"
        raise ValueError(f"index {index} is not in the partition")


@dataclass(frozen=True)
class StructuralFilterConfig:
    """Configuration that determines the mathematical filtering path."""

    integration_space: IntegrationSpace
    deterministic_completion: CompletionPolicy
    approximation_label: str | None = None
    allow_full_state_for_mixed: bool = False

    def __post_init__(self) -> None:
        if self.integration_space not in {"innovation", "stochastic_state", "full_state"}:
            raise ValueError(f"unsupported integration_space: {self.integration_space}")
        if self.deterministic_completion not in {"required", "none", "approximate"}:
            raise ValueError(
                "deterministic_completion must be required, none, or approximate"
            )
        if self.approximation_label is not None and not self.approximation_label.strip():
            raise ValueError("approximation_label must be nonempty when supplied")
        if self.deterministic_completion == "approximate" and self.approximation_label is None:
            raise ValueError("approximate deterministic completion requires a label")


@dataclass(frozen=True)
class FilterRunMetadata:
    """Provenance returned with a filter likelihood."""

    filter_name: str
    partition: StatePartition | None
    integration_space: IntegrationSpace
    deterministic_completion: CompletionPolicy
    approximation_label: str | None = None
    differentiability_status: str = "not_declared"
    compiled_status: str = "eager"


class StructuralStateSpaceModel(Protocol):
    """Protocol consumed by structural nonlinear reference filters."""

    partition: StatePartition

    def initial_mean(self, theta=None): ...

    def initial_cov(self, theta=None): ...

    def innovation_cov(self, theta=None): ...

    def observation_cov(self, theta=None): ...

    def transition(self, previous_state, innovation, theta=None): ...

    def observe(self, state_points, theta=None): ...


def validate_filter_config(
    partition: StatePartition | None, config: StructuralFilterConfig
) -> None:
    """Validate a backend configuration against declared state structure."""

    if partition is None:
        if config.deterministic_completion == "required":
            raise ValueError(
                "structural deterministic completion requires partition metadata"
            )
        if config.integration_space != "full_state":
            raise ValueError("non-full-state integration requires partition metadata")
        return

    if partition.is_mixed:
        if config.deterministic_completion == "none":
            raise ValueError("mixed structural models require deterministic completion")
        if config.integration_space == "full_state":
            if not config.allow_full_state_for_mixed:
                raise ValueError(
                    "full-state integration for mixed models requires explicit opt-in"
                )
            if config.approximation_label is None:
                raise ValueError(
                    "full-state integration for mixed models requires an approximation label"
                )
    elif partition.deterministic_dim == 0:
        if config.deterministic_completion == "required":
            raise ValueError(
                "deterministic completion cannot be required for an all-stochastic model"
            )

    if config.integration_space in {"innovation", "stochastic_state"}:
        if partition.stochastic_dim == 0:
            raise ValueError("stochastic integration requires stochastic indices")
        if partition.innovation_dim <= 0:
            raise ValueError("stochastic integration requires positive innovation_dim")

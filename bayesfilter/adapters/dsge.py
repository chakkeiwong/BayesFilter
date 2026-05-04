"""Fail-closed structural adapter gates for DSGE-style models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bayesfilter.structural import StatePartition


@dataclass(frozen=True)
class DSGEStructuralAdapterGateResult:
    """Result of checking whether a DSGE model exposes BayesFilter metadata."""

    model_name: str
    partition: StatePartition | None
    adapter_ready: bool
    blockers: tuple[str, ...]
    approximation_label: str | None = None
    source: str = "dsge_structural_adapter_gate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_name", str(self.model_name))
        object.__setattr__(self, "adapter_ready", bool(self.adapter_ready))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))
        if self.approximation_label is not None:
            object.__setattr__(self, "approximation_label", str(self.approximation_label))


def _call_or_value(model: Any, name: str) -> Any:
    value = getattr(model, name)
    return value() if callable(value) else value


def _optional_tuple(model: Any, *names: str) -> tuple[Any, ...] | None:
    for name in names:
        if hasattr(model, name):
            value = _call_or_value(model, name)
            return tuple(value)
    return None


def _optional_int(model: Any, *names: str) -> int | None:
    for name in names:
        if hasattr(model, name):
            return int(_call_or_value(model, name))
    return None


def dsge_structural_adapter_gate(
    model: Any,
    *,
    model_name: str | None = None,
    approximation_label: str | None = None,
) -> DSGEStructuralAdapterGateResult:
    """Normalize DSGE structural metadata or fail closed.

    BayesFilter does not infer DSGE economics from variable names.  Client
    models must expose state names, stochastic/deterministic index sets, and
    innovation dimension through explicit adapter metadata before structural
    nonlinear filtering can be enabled.
    """

    name = model_name or type(model).__name__
    blockers: list[str] = []
    partition: StatePartition | None = None
    try:
        state_names = _optional_tuple(model, "bayesfilter_state_names", "state_names")
        stochastic_indices = _optional_tuple(
            model,
            "bayesfilter_stochastic_indices",
            "stochastic_indices",
        )
        deterministic_indices = _optional_tuple(
            model,
            "bayesfilter_deterministic_indices",
            "deterministic_indices",
        )
        auxiliary_indices = _optional_tuple(
            model,
            "bayesfilter_auxiliary_indices",
            "auxiliary_indices",
        )
        innovation_dim = _optional_int(
            model,
            "bayesfilter_innovation_dim",
            "innovation_dim",
            "n_shocks",
        )
        missing = [
            label
            for label, value in (
                ("state names", state_names),
                ("stochastic indices", stochastic_indices),
                ("deterministic indices", deterministic_indices),
                ("innovation dimension", innovation_dim),
            )
            if value is None
        ]
        if missing:
            blockers.append("missing explicit DSGE structural metadata: " + ", ".join(missing))
        else:
            partition = StatePartition(
                state_names=tuple(str(item) for item in state_names),
                stochastic_indices=tuple(int(item) for item in stochastic_indices),
                deterministic_indices=tuple(int(item) for item in deterministic_indices),
                auxiliary_indices=tuple(int(item) for item in (auxiliary_indices or ())),
                innovation_dim=int(innovation_dim),
            )
            completion = getattr(model, "bayesfilter_deterministic_completion", None)
            if partition.is_mixed and completion is None:
                blockers.append("mixed DSGE model lacks deterministic completion map")
    except Exception as exc:
        blockers.append(f"invalid DSGE structural metadata: {exc}")
        partition = None

    if approximation_label is not None and not approximation_label.strip():
        blockers.append("approximation_label must be nonempty when supplied")

    return DSGEStructuralAdapterGateResult(
        model_name=name,
        partition=partition,
        adapter_ready=not blockers,
        blockers=tuple(blockers),
        approximation_label=approximation_label,
    )

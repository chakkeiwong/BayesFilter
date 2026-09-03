"""Public stage-boundary resume primitives for HMC tuning.

BayesFilter's existing routes continue to own every TensorFlow and TensorFlow
Probability operation. This module only sequences completed stages and records
JSON-safe receipts. Runtime handoffs remain typed in-memory values; a caller
that resumes a process must reconstruct one from the referenced artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any


HMC_STAGE_RESUME_SCHEMA = "bayesfilter.hmc_stage_resume.v1"
HMC_STAGE_SEQUENCE_RESULT_SCHEMA = "bayesfilter.hmc_stage_sequence_result.v1"

ORDINARY_HMC_TUNING_ROUTE = "ordinary_hmc_tuning"
FIXED_TRANSPORT_HMC_TUNING_ROUTE = "fixed_transport_hmc_tuning"
ORDINARY_HMC_STAGE_NAMES = (
    "geometry",
    "bootstrap",
    "windowed_mass",
    "fixed_mass_step",
    "frozen_step_trajectory",
    "verification",
)
FIXED_TRANSPORT_HMC_STAGE_NAME = "candidate_attempt"


class HMCStageResumeError(ValueError):
    """Raised when a stage receipt cannot support an exact boundary resume."""


def _token(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise HMCStageResumeError(f"{name} must be a non-empty token")
    result = value.strip()
    if not result or any(character.isspace() for character in result):
        raise HMCStageResumeError(f"{name} must be a non-empty token")
    return result


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HMCStageResumeError(f"{name} must be a non-empty string")
    return value


def _index(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise HMCStageResumeError(f"{name} must be a non-negative integer")
    return int(value)


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HMCStageResumeError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise HMCStageResumeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _json_safe(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HMCStageResumeError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HMCStageResumeError(f"{path} has a non-string key")
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, path=f"{path}[{position}]")
            for position, item in enumerate(value)
        ]
    raise HMCStageResumeError(
        f"{path} contains unsupported {type(value).__name__}; use JSON metadata"
    )


def _json_object(value: Any, *, name: str, nonempty: bool = False) -> dict[str, Any]:
    result = _json_safe(value, path=name)
    if not isinstance(result, dict):
        raise HMCStageResumeError(f"{name} must be a JSON object")
    if nonempty and not result:
        raise HMCStageResumeError(f"{name} must be a non-empty JSON object")
    return result


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value, path="payload"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HMCStageOutcome:
    """A completed stage's runtime handoff and public receipt metadata."""

    handoff: Any = field(repr=False, compare=False)
    handoff_kind: str
    handoff_ref: str
    handoff_sha256: str
    summary: Mapping[str, Any] = field(default_factory=dict)
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.handoff is None:
            raise HMCStageResumeError("a completed stage must return a runtime handoff")
        object.__setattr__(
            self,
            "handoff_kind",
            _token(self.handoff_kind, name="handoff_kind"),
        )
        object.__setattr__(
            self,
            "handoff_ref",
            _text(self.handoff_ref, name="handoff_ref"),
        )
        object.__setattr__(
            self,
            "handoff_sha256",
            _sha256(self.handoff_sha256, name="handoff_sha256"),
        )
        object.__setattr__(
            self,
            "summary",
            _json_object(self.summary, name="summary"),
        )
        if not isinstance(self.terminal, bool):
            raise HMCStageResumeError("terminal must be a boolean")


@dataclass(frozen=True)
class HMCStageSpec:
    """One stage that consumes the preceding typed runtime handoff."""

    name: str
    index: int
    handoff_kind: str
    run: Callable[[Any | None], HMCStageOutcome]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _token(self.name, name="stage name"))
        object.__setattr__(self, "index", _index(self.index, name="stage index"))
        object.__setattr__(
            self,
            "handoff_kind",
            _token(self.handoff_kind, name="stage handoff_kind"),
        )
        if not callable(self.run):
            raise HMCStageResumeError("stage run must be callable")


@dataclass(frozen=True)
class HMCStageBoundary:
    """JSON-safe metadata emitted after a stage has fully completed."""

    name: str
    index: int
    handoff_kind: str
    handoff_ref: str
    handoff_sha256: str
    summary: Mapping[str, Any]
    terminal: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _token(self.name, name="stage name"))
        object.__setattr__(self, "index", _index(self.index, name="boundary index"))
        object.__setattr__(
            self,
            "handoff_kind",
            _token(self.handoff_kind, name="boundary handoff_kind"),
        )
        object.__setattr__(
            self,
            "handoff_ref",
            _text(self.handoff_ref, name="handoff_ref"),
        )
        object.__setattr__(
            self,
            "handoff_sha256",
            _sha256(self.handoff_sha256, name="handoff_sha256"),
        )
        object.__setattr__(
            self,
            "summary",
            _json_object(self.summary, name="summary"),
        )
        if not isinstance(self.terminal, bool):
            raise HMCStageResumeError("boundary terminal must be a boolean")

    def payload(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "index": self.index,
            "handoff_kind": self.handoff_kind,
            "handoff_ref": self.handoff_ref,
            "handoff_sha256": self.handoff_sha256,
            "summary": self.summary,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class HMCStageResumeCheckpoint:
    """A self-checking receipt for one completed stage boundary."""

    route: str
    run_contract: Mapping[str, Any]
    stage_name: str
    stage_index: int
    handoff_kind: str
    handoff_ref: str
    handoff_sha256: str
    terminal: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", _token(self.route, name="route"))
        object.__setattr__(
            self,
            "run_contract",
            _json_object(self.run_contract, name="run_contract", nonempty=True),
        )
        object.__setattr__(
            self,
            "stage_name",
            _token(self.stage_name, name="stage_name"),
        )
        object.__setattr__(
            self,
            "stage_index",
            _index(self.stage_index, name="stage_index"),
        )
        object.__setattr__(
            self,
            "handoff_kind",
            _token(self.handoff_kind, name="handoff_kind"),
        )
        object.__setattr__(
            self,
            "handoff_ref",
            _text(self.handoff_ref, name="handoff_ref"),
        )
        object.__setattr__(
            self,
            "handoff_sha256",
            _sha256(self.handoff_sha256, name="handoff_sha256"),
        )
        if not isinstance(self.terminal, bool):
            raise HMCStageResumeError("checkpoint terminal must be a boolean")

    @property
    def run_contract_hash(self) -> str:
        return _stable_hash(self.run_contract)

    def _unsigned_payload(self) -> Mapping[str, Any]:
        return {
            "schema": HMC_STAGE_RESUME_SCHEMA,
            "route": self.route,
            "run_contract": self.run_contract,
            "run_contract_hash": self.run_contract_hash,
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "handoff_kind": self.handoff_kind,
            "handoff_ref": self.handoff_ref,
            "handoff_sha256": self.handoff_sha256,
            "terminal": self.terminal,
        }

    @property
    def checkpoint_hash(self) -> str:
        return _stable_hash(self._unsigned_payload())

    def payload(self) -> Mapping[str, Any]:
        return {**self._unsigned_payload(), "checkpoint_hash": self.checkpoint_hash}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_route: str | None = None,
        expected_run_contract: Mapping[str, Any] | None = None,
    ) -> "HMCStageResumeCheckpoint":
        if not isinstance(payload, Mapping):
            raise HMCStageResumeError("checkpoint must be a mapping")
        expected_keys = {
            "schema",
            "route",
            "run_contract",
            "run_contract_hash",
            "stage_name",
            "stage_index",
            "handoff_kind",
            "handoff_ref",
            "handoff_sha256",
            "terminal",
            "checkpoint_hash",
        }
        if set(payload) != expected_keys:
            raise HMCStageResumeError("checkpoint fields do not match the v1 schema")
        if payload.get("schema") != HMC_STAGE_RESUME_SCHEMA:
            raise HMCStageResumeError("checkpoint schema mismatch")
        stored_hash = _sha256(payload.get("checkpoint_hash"), name="checkpoint_hash")
        unsigned = {
            key: value for key, value in payload.items() if key != "checkpoint_hash"
        }
        if _stable_hash(unsigned) != stored_hash:
            raise HMCStageResumeError("checkpoint hash mismatch")
        if payload.get("run_contract_hash") != _stable_hash(
            payload.get("run_contract")
        ):
            raise HMCStageResumeError("run_contract_hash mismatch")
        checkpoint = cls(
            route=payload.get("route"),
            run_contract=payload.get("run_contract"),
            stage_name=payload.get("stage_name"),
            stage_index=payload.get("stage_index"),
            handoff_kind=payload.get("handoff_kind"),
            handoff_ref=payload.get("handoff_ref"),
            handoff_sha256=payload.get("handoff_sha256"),
            terminal=payload.get("terminal"),
        )
        if checkpoint.checkpoint_hash != stored_hash:
            raise HMCStageResumeError("checkpoint canonical hash mismatch")
        if expected_route is not None:
            if checkpoint.route != _token(expected_route, name="expected route"):
                raise HMCStageResumeError(
                    "checkpoint route does not match the request"
                )
        if expected_run_contract is not None:
            if checkpoint.run_contract_hash != _stable_hash(expected_run_contract):
                raise HMCStageResumeError(
                    "checkpoint run contract does not match the request"
                )
        return checkpoint


@dataclass(frozen=True)
class HMCStageSequenceResult:
    """Result of an uninterrupted or resumed stage sequence.

    ``final_handoff`` is runtime state and is intentionally absent from
    :meth:`payload`.
    """

    route: str
    run_contract_hash: str
    completed_stage_names: tuple[str, ...]
    emitted_boundaries: tuple[HMCStageBoundary, ...]
    resumed_from_stage: str | None
    terminal: bool
    final_handoff: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", _token(self.route, name="route"))
        object.__setattr__(
            self,
            "run_contract_hash",
            _sha256(self.run_contract_hash, name="run_contract_hash"),
        )
        names = tuple(
            _token(name, name="completed stage name")
            for name in self.completed_stage_names
        )
        boundaries = tuple(self.emitted_boundaries)
        if any(not isinstance(boundary, HMCStageBoundary) for boundary in boundaries):
            raise HMCStageResumeError(
                "emitted_boundaries must contain HMCStageBoundary values"
            )
        if not isinstance(self.terminal, bool):
            raise HMCStageResumeError("sequence terminal must be a boolean")
        if self.terminal and self.final_handoff is None:
            raise HMCStageResumeError("a terminal sequence requires a final handoff")
        resumed = None
        if self.resumed_from_stage is not None:
            resumed = _token(self.resumed_from_stage, name="resumed_from_stage")
        object.__setattr__(self, "completed_stage_names", names)
        object.__setattr__(self, "emitted_boundaries", boundaries)
        object.__setattr__(self, "resumed_from_stage", resumed)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": HMC_STAGE_SEQUENCE_RESULT_SCHEMA,
            "route": self.route,
            "run_contract_hash": self.run_contract_hash,
            "completed_stage_names": self.completed_stage_names,
            "emitted_boundaries": tuple(
                boundary.payload() for boundary in self.emitted_boundaries
            ),
            "resumed_from_stage": self.resumed_from_stage,
            "terminal": self.terminal,
            "runtime_handoff_serialized": False,
        }


def completed_hmc_stage(
    *,
    handoff: Any,
    handoff_kind: str,
    handoff_ref: str,
    handoff_sha256: str,
    summary: Mapping[str, Any] | None = None,
    terminal: bool = False,
) -> HMCStageOutcome:
    """Construct the outcome accepted from a completed stage callable."""

    return HMCStageOutcome(
        handoff=handoff,
        handoff_kind=handoff_kind,
        handoff_ref=handoff_ref,
        handoff_sha256=handoff_sha256,
        summary={} if summary is None else summary,
        terminal=terminal,
    )


def _validate_stage_specs(stages: Sequence[HMCStageSpec]) -> tuple[HMCStageSpec, ...]:
    normalized = tuple(stages)
    if not normalized:
        raise HMCStageResumeError("at least one stage is required")
    if any(not isinstance(spec, HMCStageSpec) for spec in normalized):
        raise HMCStageResumeError("stages must contain HMCStageSpec values")
    if tuple(spec.index for spec in normalized) != tuple(range(len(normalized))):
        raise HMCStageResumeError("stage indices must be contiguous and start at zero")
    names = tuple(spec.name for spec in normalized)
    if len(set(names)) != len(names):
        raise HMCStageResumeError("stage names must be unique within one sequence")
    return normalized


def _boundary_from_outcome(
    *,
    spec: HMCStageSpec,
    outcome: HMCStageOutcome,
) -> HMCStageBoundary:
    if not isinstance(outcome, HMCStageOutcome):
        raise HMCStageResumeError(
            f"stage {spec.name!r} did not return HMCStageOutcome"
        )
    if outcome.handoff_kind != spec.handoff_kind:
        raise HMCStageResumeError(
            f"stage {spec.name!r} emitted handoff kind {outcome.handoff_kind!r}; "
            f"expected {spec.handoff_kind!r}"
        )
    return HMCStageBoundary(
        name=spec.name,
        index=spec.index,
        handoff_kind=outcome.handoff_kind,
        handoff_ref=outcome.handoff_ref,
        handoff_sha256=outcome.handoff_sha256,
        summary=outcome.summary,
        terminal=outcome.terminal,
    )


def build_hmc_stage_resume_checkpoint(
    *,
    route: str,
    run_contract: Mapping[str, Any],
    boundary: HMCStageBoundary,
) -> HMCStageResumeCheckpoint:
    """Bind a completed public boundary to the exact run contract."""

    if not isinstance(boundary, HMCStageBoundary):
        raise HMCStageResumeError("boundary must be HMCStageBoundary")
    return HMCStageResumeCheckpoint(
        route=route,
        run_contract=run_contract,
        stage_name=boundary.name,
        stage_index=boundary.index,
        handoff_kind=boundary.handoff_kind,
        handoff_ref=boundary.handoff_ref,
        handoff_sha256=boundary.handoff_sha256,
        terminal=boundary.terminal,
    )


def validate_hmc_stage_resume_checkpoint(
    checkpoint: HMCStageResumeCheckpoint | Mapping[str, Any],
    *,
    route: str,
    run_contract: Mapping[str, Any],
    stages: Sequence[HMCStageSpec],
) -> HMCStageResumeCheckpoint:
    """Validate a receipt before loading handoff state or running a stage."""

    normalized_stages = _validate_stage_specs(stages)
    checked = HMCStageResumeCheckpoint.from_payload(
        checkpoint.payload()
        if isinstance(checkpoint, HMCStageResumeCheckpoint)
        else checkpoint,
        expected_route=route,
        expected_run_contract=run_contract,
    )
    if checked.stage_index >= len(normalized_stages):
        raise HMCStageResumeError("checkpoint stage index is absent from this request")
    spec = normalized_stages[checked.stage_index]
    if spec.name != checked.stage_name or spec.handoff_kind != checked.handoff_kind:
        raise HMCStageResumeError(
            "checkpoint stage identity does not match this request"
        )
    is_final_stage = checked.stage_index == normalized_stages[-1].index
    if checked.terminal != is_final_stage:
        raise HMCStageResumeError(
            "checkpoint terminal flag does not match its stage position"
        )
    return checked


def run_hmc_stage_sequence(
    *,
    route: str,
    run_contract: Mapping[str, Any],
    stages: Sequence[HMCStageSpec],
    resume_checkpoint: HMCStageResumeCheckpoint | Mapping[str, Any] | None = None,
    checkpoint_callback: Callable[[HMCStageResumeCheckpoint], None] | None = None,
    handoff_loader: Callable[[HMCStageResumeCheckpoint], Any] | None = None,
) -> HMCStageSequenceResult:
    """Run stages, skipping only a validated and loadable completed prefix.

    The callback runs only after a stage has returned a complete outcome and
    its terminal position is valid. Resuming requires ``handoff_loader`` to
    verify and reconstruct the typed runtime handoff referenced by the receipt.
    A terminal checkpoint is an idempotent completed result.
    """

    route = _token(route, name="route")
    contract = _json_object(run_contract, name="run_contract", nonempty=True)
    normalized_stages = _validate_stage_specs(stages)
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise HMCStageResumeError("checkpoint_callback must be callable")
    if handoff_loader is not None and not callable(handoff_loader):
        raise HMCStageResumeError("handoff_loader must be callable")

    start_index = 0
    resumed_from: str | None = None
    current_handoff: Any | None = None
    if resume_checkpoint is not None:
        if handoff_loader is None:
            raise HMCStageResumeError(
                "resume requires a handoff_loader for the referenced artifact"
            )
        checked = validate_hmc_stage_resume_checkpoint(
            resume_checkpoint,
            route=route,
            run_contract=contract,
            stages=normalized_stages,
        )
        current_handoff = handoff_loader(checked)
        if current_handoff is None:
            raise HMCStageResumeError("handoff_loader returned no runtime handoff")
        start_index = checked.stage_index + 1
        resumed_from = checked.stage_name
        if checked.terminal:
            return HMCStageSequenceResult(
                route=route,
                run_contract_hash=_stable_hash(contract),
                completed_stage_names=tuple(
                    spec.name for spec in normalized_stages[:start_index]
                ),
                emitted_boundaries=(),
                resumed_from_stage=resumed_from,
                terminal=True,
                final_handoff=current_handoff,
            )

    emitted: list[HMCStageBoundary] = []
    completed_names = [spec.name for spec in normalized_stages[:start_index]]
    terminal = False
    for spec in normalized_stages[start_index:]:
        outcome = spec.run(current_handoff)
        boundary = _boundary_from_outcome(spec=spec, outcome=outcome)
        is_final_stage = spec.index == normalized_stages[-1].index
        if boundary.terminal != is_final_stage:
            raise HMCStageResumeError(
                f"stage {spec.name!r} terminal flag does not match its position"
            )
        checkpoint = build_hmc_stage_resume_checkpoint(
            route=route,
            run_contract=contract,
            boundary=boundary,
        )
        if checkpoint_callback is not None:
            if checkpoint_callback(checkpoint) is not None:
                raise HMCStageResumeError("checkpoint_callback must return None")
        emitted.append(boundary)
        completed_names.append(spec.name)
        current_handoff = outcome.handoff
        terminal = boundary.terminal

    if not terminal or current_handoff is None:
        raise HMCStageResumeError("stage sequence did not produce a terminal handoff")
    return HMCStageSequenceResult(
        route=route,
        run_contract_hash=_stable_hash(contract),
        completed_stage_names=tuple(completed_names),
        emitted_boundaries=tuple(emitted),
        resumed_from_stage=resumed_from,
        terminal=True,
        final_handoff=current_handoff,
    )


def write_hmc_stage_resume_checkpoint(
    path: str | os.PathLike[str],
    checkpoint: HMCStageResumeCheckpoint,
) -> None:
    """Atomically persist one receipt for a single-writer run."""

    if not isinstance(checkpoint, HMCStageResumeCheckpoint):
        raise HMCStageResumeError("checkpoint must be HMCStageResumeCheckpoint")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            previous = load_hmc_stage_resume_checkpoint(destination)
        except HMCStageResumeError as exc:
            raise HMCStageResumeError(
                "refusing to overwrite an invalid existing checkpoint"
            ) from exc
        if previous.route != checkpoint.route:
            raise HMCStageResumeError(
                "refusing to overwrite a checkpoint for a different route"
            )
        if previous.run_contract_hash != checkpoint.run_contract_hash:
            raise HMCStageResumeError(
                "refusing to overwrite a checkpoint for a different run contract"
            )
        if previous.payload() == checkpoint.payload():
            return
        if previous.terminal:
            raise HMCStageResumeError("refusing to overwrite a terminal checkpoint")
        if checkpoint.stage_index < previous.stage_index:
            raise HMCStageResumeError(
                "refusing to overwrite a newer checkpoint with a rewound stage"
            )
        if checkpoint.stage_index == previous.stage_index:
            raise HMCStageResumeError(
                "refusing to overwrite a checkpoint with a different same-stage handoff"
            )
        if checkpoint.stage_index != previous.stage_index + 1:
            raise HMCStageResumeError("refusing to write a checkpoint with a stage gap")

    encoded = json.dumps(
        checkpoint.payload(),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_hmc_stage_resume_checkpoint(
    path: str | os.PathLike[str],
    *,
    route: str | None = None,
    run_contract: Mapping[str, Any] | None = None,
) -> HMCStageResumeCheckpoint:
    """Load a persisted receipt and validate its schema and hashes."""

    try:
        with Path(path).open("r", encoding="ascii") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HMCStageResumeError("unable to read checkpoint JSON") from exc
    return HMCStageResumeCheckpoint.from_payload(
        payload,
        expected_route=route,
        expected_run_contract=run_contract,
    )


__all__ = [
    "FIXED_TRANSPORT_HMC_STAGE_NAME",
    "FIXED_TRANSPORT_HMC_TUNING_ROUTE",
    "HMC_STAGE_RESUME_SCHEMA",
    "HMC_STAGE_SEQUENCE_RESULT_SCHEMA",
    "ORDINARY_HMC_STAGE_NAMES",
    "ORDINARY_HMC_TUNING_ROUTE",
    "HMCStageBoundary",
    "HMCStageOutcome",
    "HMCStageResumeCheckpoint",
    "HMCStageResumeError",
    "HMCStageSequenceResult",
    "HMCStageSpec",
    "build_hmc_stage_resume_checkpoint",
    "completed_hmc_stage",
    "load_hmc_stage_resume_checkpoint",
    "run_hmc_stage_sequence",
    "validate_hmc_stage_resume_checkpoint",
    "write_hmc_stage_resume_checkpoint",
]

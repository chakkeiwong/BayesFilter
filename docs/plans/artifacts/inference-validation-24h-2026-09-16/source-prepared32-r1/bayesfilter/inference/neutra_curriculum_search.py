"""Target-agnostic search policy for NeuTra variable-group curricula.

The module deliberately does not train a transport or evaluate a target.  A
model adapter supplies a probe callback that starts every candidate from a
common parent checkpoint and reports held-out loss improvement.  This module
validates the evidence, applies the predeclared uncertainty rule, and performs
bounded prerequisite-constrained beam search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


class NeuTraCurriculumSearchError(RuntimeError):
    """Raised when curriculum-search evidence violates its contract."""


@dataclass(frozen=True)
class NeuTraCurriculumGroup:
    """A model-adapter supplied group and its activation prerequisites."""

    name: str
    prerequisites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("curriculum group name must be nonempty")
        prerequisites = tuple(str(item) for item in self.prerequisites)
        if len(set(prerequisites)) != len(prerequisites):
            raise ValueError("curriculum prerequisites must be unique")
        if self.name in prerequisites:
            raise ValueError("curriculum group cannot depend on itself")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "prerequisites", prerequisites)


@dataclass(frozen=True)
class NeuTraCurriculumSearchConfig:
    """Bounded search and uncertainty policy."""

    probe_updates: int = 100
    probe_replicates: int = 4
    beam_width: int = 2
    maximum_depth: int = 4
    maximum_probe_calls: int = 128
    critical_value: float = 2.0
    minimum_improvement_per_update: float = 0.0

    def __post_init__(self) -> None:
        if int(self.probe_updates) <= 0:
            raise ValueError("probe_updates must be positive")
        if int(self.probe_replicates) < 2:
            raise ValueError("probe_replicates must be at least two")
        if int(self.beam_width) <= 0:
            raise ValueError("beam_width must be positive")
        if int(self.maximum_depth) <= 0:
            raise ValueError("maximum_depth must be positive")
        if int(self.maximum_probe_calls) < int(self.probe_replicates):
            raise ValueError("maximum_probe_calls must fit one replicated probe")
        if not math.isfinite(float(self.critical_value)) or float(self.critical_value) < 0.0:
            raise ValueError("critical_value must be finite and nonnegative")
        if not math.isfinite(float(self.minimum_improvement_per_update)):
            raise ValueError("minimum_improvement_per_update must be finite")


@dataclass(frozen=True)
class NeuTraCurriculumProbe:
    """One immutable adapter-produced probe observation."""

    parent_sequence: tuple[str, ...]
    candidate_group: str
    replicate: int
    incoming_loss: float
    best_loss: float
    executed_updates: int
    probe_updates: int
    finite: bool
    parent_state_hash: str

    def __post_init__(self) -> None:
        sequence = tuple(str(item) for item in self.parent_sequence)
        if len(set(sequence)) != len(sequence):
            raise ValueError("parent_sequence must not contain duplicate groups")
        object.__setattr__(self, "parent_sequence", sequence)
        object.__setattr__(self, "candidate_group", str(self.candidate_group))
        if int(self.replicate) < 0:
            raise ValueError("replicate must be nonnegative")
        for name in ("incoming_loss", "best_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if int(self.executed_updates) <= 0:
            raise ValueError("executed_updates must be positive")
        if int(self.probe_updates) <= 0:
            raise ValueError("probe_updates must be positive")
        digest = str(self.parent_state_hash).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("parent_state_hash must be lowercase SHA-256 hex")
        object.__setattr__(self, "parent_state_hash", digest)


@dataclass(frozen=True)
class NeuTraCurriculumCandidate:
    """Replicated evidence and uncertainty summary for one group addition."""

    parent_sequence: tuple[str, ...]
    candidate_group: str
    observations: tuple[NeuTraCurriculumProbe, ...]
    improvements_per_update: tuple[float, ...]
    mean_improvement_per_update: float
    standard_deviation: float
    lower_confidence_bound: float
    passed: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class NeuTraCurriculumSearchResult:
    """Complete bounded search artifact, including unranked viable paths."""

    representative_sequence: tuple[str, ...] | None
    final_beam_sequences: tuple[tuple[str, ...], ...]
    viable_sequences: tuple[tuple[str, ...], ...]
    terminal_sequences: tuple[tuple[str, ...], ...]
    candidates: tuple[NeuTraCurriculumCandidate, ...]
    probe_calls: int
    stop_reason: str
    nonclaims: tuple[str, ...]


@dataclass(frozen=True)
class NeuTraProtocolSelectionConfig:
    """Paired uncertainty-set rule for full equal-budget protocols."""

    replicates: int = 4
    critical_value: float = 2.0
    practical_loss_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if int(self.replicates) < 2:
            raise ValueError("protocol selection requires at least two replicates")
        if not math.isfinite(float(self.critical_value)) or float(self.critical_value) < 0.0:
            raise ValueError("critical_value must be finite and nonnegative")
        if (
            not math.isfinite(float(self.practical_loss_tolerance))
            or float(self.practical_loss_tolerance) < 0.0
        ):
            raise ValueError("practical_loss_tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class NeuTraProtocolObservation:
    """One full-protocol terminal selection-loss observation."""

    name: str
    sequence: tuple[str, ...]
    replicate: int
    terminal_loss: float
    executed_updates: int
    update_budget: int
    finite: bool
    selection_partition_id: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("protocol name must be nonempty")
        sequence = tuple(str(item) for item in self.sequence)
        if len(set(sequence)) != len(sequence):
            raise ValueError("protocol sequence must not contain duplicates")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "sequence", sequence)
        if int(self.replicate) < 0:
            raise ValueError("replicate must be nonnegative")
        if not math.isfinite(float(self.terminal_loss)):
            raise ValueError("terminal_loss must be finite")
        if int(self.executed_updates) <= 0 or int(self.update_budget) <= 0:
            raise ValueError("protocol update counts must be positive")
        if not self.selection_partition_id:
            raise ValueError("selection_partition_id must be nonempty")


@dataclass(frozen=True)
class NeuTraProtocolComparison:
    name: str
    sequence: tuple[str, ...]
    mean_loss: float
    paired_mean_delta: float
    paired_upper_bound: float
    in_uncertainty_set: bool


@dataclass(frozen=True)
class NeuTraProtocolSelectionResult:
    selected_name: str
    selected_sequence: tuple[str, ...]
    reference_name: str
    uncertainty_set: tuple[str, ...]
    comparisons: tuple[NeuTraProtocolComparison, ...]
    nonclaims: tuple[str, ...]


def select_neutra_protocol(
    *,
    observations: Sequence[NeuTraProtocolObservation | Mapping[str, Any]],
    config: NeuTraProtocolSelectionConfig,
) -> NeuTraProtocolSelectionResult:
    """Select the least complex protocol inside a paired loss uncertainty set."""

    normalized = tuple(
        item
        if isinstance(item, NeuTraProtocolObservation)
        else NeuTraProtocolObservation(**dict(item))
        for item in observations
    )
    by_name: dict[str, list[NeuTraProtocolObservation]] = {}
    for observation in normalized:
        by_name.setdefault(observation.name, []).append(observation)
    if not by_name:
        raise ValueError("at least one full protocol is required")
    expected_replicates = tuple(range(int(config.replicates)))
    partition_by_replicate: dict[int, str] = {}
    update_budget = None
    sequence_by_name: dict[str, tuple[str, ...]] = {}
    losses_by_name: dict[str, tuple[float, ...]] = {}
    for name, rows in by_name.items():
        ordered = tuple(sorted(rows, key=lambda row: row.replicate))
        if tuple(row.replicate for row in ordered) != expected_replicates:
            raise NeuTraCurriculumSearchError("protocol replicate identities are incomplete")
        if not all(row.finite for row in ordered):
            raise NeuTraCurriculumSearchError("protocol selection contains a nonfinite run")
        if len({row.sequence for row in ordered}) != 1:
            raise NeuTraCurriculumSearchError("protocol sequence changed across replicates")
        for row in ordered:
            if row.executed_updates != row.update_budget:
                raise NeuTraCurriculumSearchError(
                    "protocol did not consume the exact declared update budget"
                )
            if update_budget is None:
                update_budget = row.update_budget
            elif update_budget != row.update_budget:
                raise NeuTraCurriculumSearchError("protocol update budgets are not equal")
            expected_partition = partition_by_replicate.get(row.replicate)
            if expected_partition is None:
                partition_by_replicate[row.replicate] = row.selection_partition_id
            elif expected_partition != row.selection_partition_id:
                raise NeuTraCurriculumSearchError(
                    "protocols do not share paired selection partitions"
                )
        sequence_by_name[name] = ordered[0].sequence
        losses_by_name[name] = tuple(float(row.terminal_loss) for row in ordered)
    mean_by_name = {
        name: sum(losses) / len(losses) for name, losses in losses_by_name.items()
    }
    reference_name = min(mean_by_name, key=lambda name: (mean_by_name[name], name))
    reference_losses = losses_by_name[reference_name]
    comparisons = []
    uncertainty_set = []
    for name in sorted(losses_by_name):
        deltas = tuple(
            value - reference
            for value, reference in zip(losses_by_name[name], reference_losses, strict=True)
        )
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((value - mean_delta) ** 2 for value in deltas) / (
            len(deltas) - 1
        )
        upper = mean_delta + float(config.critical_value) * math.sqrt(variance) / math.sqrt(
            len(deltas)
        )
        included = upper <= float(config.practical_loss_tolerance)
        if included:
            uncertainty_set.append(name)
        comparisons.append(
            NeuTraProtocolComparison(
                name=name,
                sequence=sequence_by_name[name],
                mean_loss=mean_by_name[name],
                paired_mean_delta=mean_delta,
                paired_upper_bound=upper,
                in_uncertainty_set=included,
            )
        )
    if not uncertainty_set:
        raise NeuTraCurriculumSearchError("protocol uncertainty set is empty")
    selected_name = min(
        uncertainty_set,
        key=lambda name: (len(sequence_by_name[name]), name),
    )
    return NeuTraProtocolSelectionResult(
        selected_name=selected_name,
        selected_sequence=sequence_by_name[selected_name],
        reference_name=reference_name,
        uncertainty_set=tuple(uncertainty_set),
        comparisons=tuple(comparisons),
        nonclaims=(
            "protocol loss selection nominates a fresh final run only",
            "uncertainty-set membership does not establish predictive equivalence",
            "final promotion requires untouched downstream validation",
        ),
    )


def _validate_groups(groups: Sequence[NeuTraCurriculumGroup]) -> Mapping[str, NeuTraCurriculumGroup]:
    by_name: dict[str, NeuTraCurriculumGroup] = {}
    for group in groups:
        if group.name in by_name:
            raise ValueError(f"duplicate curriculum group: {group.name}")
        by_name[group.name] = group
    for group in groups:
        unknown = tuple(name for name in group.prerequisites if name not in by_name)
        if unknown:
            raise ValueError(
                f"unknown prerequisites for {group.name}: {', '.join(unknown)}"
            )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("curriculum prerequisite graph contains a cycle")
        if name in visited:
            return
        visiting.add(name)
        for prerequisite in by_name[name].prerequisites:
            visit(prerequisite)
        visiting.remove(name)
        visited.add(name)

    for name in by_name:
        visit(name)
    if not by_name:
        raise ValueError("at least one curriculum group is required")
    return by_name


def _summarize_candidate(
    observations: Sequence[NeuTraCurriculumProbe],
    config: NeuTraCurriculumSearchConfig,
) -> NeuTraCurriculumCandidate:
    if len(observations) != int(config.probe_replicates):
        raise NeuTraCurriculumSearchError("probe replicate count mismatch")
    first = observations[0]
    if tuple(observation.replicate for observation in observations) != tuple(
        range(int(config.probe_replicates))
    ):
        raise NeuTraCurriculumSearchError("probe replicate identities are invalid")
    if any(
        observation.probe_updates != int(config.probe_updates)
        or observation.executed_updates != int(config.probe_updates)
        for observation in observations
    ):
        raise NeuTraCurriculumSearchError("probe did not consume the exact declared budget")
    improvements = tuple(
        (observation.incoming_loss - observation.best_loss)
        / float(observation.executed_updates)
        for observation in observations
    )
    mean = sum(improvements) / len(improvements)
    variance = sum((value - mean) ** 2 for value in improvements) / (len(improvements) - 1)
    standard_deviation = math.sqrt(variance)
    lower = mean - float(config.critical_value) * standard_deviation / math.sqrt(
        len(improvements)
    )
    finite = all(observation.finite for observation in observations)
    passed = finite and lower >= float(config.minimum_improvement_per_update)
    reason = None
    if not finite:
        reason = "nonfinite_probe"
    elif lower < float(config.minimum_improvement_per_update):
        reason = "uncertainty_bound_below_threshold"
    return NeuTraCurriculumCandidate(
        parent_sequence=first.parent_sequence,
        candidate_group=first.candidate_group,
        observations=tuple(observations),
        improvements_per_update=improvements,
        mean_improvement_per_update=mean,
        standard_deviation=standard_deviation,
        lower_confidence_bound=lower,
        passed=passed,
        rejection_reason=reason,
    )


def search_neutra_curriculum(
    *,
    groups: Sequence[NeuTraCurriculumGroup],
    probe_fn: Callable[[tuple[str, ...], str, int], NeuTraCurriculumProbe | Mapping[str, Any]],
    config: NeuTraCurriculumSearchConfig,
) -> NeuTraCurriculumSearchResult:
    """Search group activation sequences with replicated equal-budget probes."""

    if not callable(probe_fn):
        raise ValueError("probe_fn must be callable")
    group_by_name = _validate_groups(groups)
    candidates: list[NeuTraCurriculumCandidate] = []
    viable_sequences: list[tuple[str, ...]] = []
    terminal_sequences: list[tuple[str, ...]] = []
    active: list[tuple[str, ...]] = [()]
    final_beam: list[tuple[str, ...]] = []
    probe_calls = 0
    exhausted_by_budget = False
    parent_authority: dict[tuple[str, ...], dict[int, tuple[str, float]]] = {}

    for depth in range(int(config.maximum_depth)):
        next_nodes: list[tuple[tuple[str, ...], NeuTraCurriculumCandidate]] = []
        progressed = False
        for sequence in active:
            eligible = tuple(
                group.name
                for group in groups
                if group.name not in sequence
                and set(group.prerequisites).issubset(sequence)
            )
            if not eligible:
                terminal_sequences.append(sequence)
                continue
            node_progressed = False
            for name in eligible:
                if probe_calls + int(config.probe_replicates) > int(config.maximum_probe_calls):
                    exhausted_by_budget = True
                    terminal_sequences.append(sequence)
                    break
                observations = []
                for replicate in range(int(config.probe_replicates)):
                    raw = probe_fn(sequence, name, replicate)
                    observation = (
                        raw
                        if isinstance(raw, NeuTraCurriculumProbe)
                        else NeuTraCurriculumProbe(**dict(raw))
                    )
                    if observation.parent_sequence != sequence or observation.candidate_group != name:
                        raise NeuTraCurriculumSearchError("probe returned the wrong node identity")
                    if int(observation.replicate) != replicate:
                        raise NeuTraCurriculumSearchError(
                            "probe returned the wrong replicate identity"
                        )
                    signature = (
                        observation.parent_state_hash,
                        float(observation.incoming_loss),
                    )
                    expected = parent_authority.setdefault(sequence, {}).get(replicate)
                    if expected is None:
                        parent_authority[sequence][replicate] = signature
                    elif expected != signature:
                        raise NeuTraCurriculumSearchError(
                            "competing probes do not share the same parent evidence"
                        )
                    observations.append(observation)
                probe_calls += int(config.probe_replicates)
                candidate = _summarize_candidate(observations, config)
                candidates.append(candidate)
                if candidate.passed:
                    node_progressed = True
                    progressed = True
                    next_nodes.append((sequence + (name,), candidate))
                    viable_sequences.append(sequence + (name,))
            if not node_progressed and not exhausted_by_budget:
                terminal_sequences.append(sequence)
        if exhausted_by_budget:
            break
        if not progressed:
            break
        ordered = sorted(
            next_nodes,
            key=lambda item: (
                -float(item[1].lower_confidence_bound),
                item[0],
            ),
        )
        active = [sequence for sequence, _candidate in ordered[: int(config.beam_width)]]
        final_beam = list(active)
        if depth + 1 >= int(config.maximum_depth):
            terminal_sequences.extend(active)
            active = []
            break

    if exhausted_by_budget:
        stop_reason = "probe_budget_exhausted"
    elif not active:
        stop_reason = "depth_or_terminal_node"
    else:
        terminal_sequences.extend(active)
        stop_reason = "no_eligible_group_passed"
    unique_viable = tuple(dict.fromkeys(viable_sequences))
    unique_terminal = tuple(dict.fromkeys(terminal_sequences))
    representative = final_beam[0] if final_beam else None
    return NeuTraCurriculumSearchResult(
        representative_sequence=representative,
        final_beam_sequences=tuple(final_beam),
        viable_sequences=unique_viable,
        terminal_sequences=unique_terminal,
        candidates=tuple(candidates),
        probe_calls=probe_calls,
        stop_reason=stop_reason,
        nonclaims=(
            "probe improvement nominates a curriculum but does not establish target correctness",
            "representative_sequence is execution ordering only, not protocol selection",
            "final promotion requires a fresh frozen protocol and downstream predictive validation",
        ),
    )

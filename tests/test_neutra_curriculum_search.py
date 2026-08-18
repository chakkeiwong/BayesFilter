"""Focused tests for target-agnostic NeuTra curriculum search policy."""

from __future__ import annotations

import pytest

from bayesfilter.inference.neutra_curriculum_search import (
    NeuTraCurriculumGroup,
    NeuTraCurriculumProbe,
    NeuTraCurriculumSearchConfig,
    NeuTraCurriculumSearchError,
    NeuTraProtocolObservation,
    NeuTraProtocolSelectionConfig,
    search_neutra_curriculum,
    select_neutra_protocol,
)


def _probe(sequence, candidate, replicate, *, improvement=0.5, finite=True, updates=4):
    return NeuTraCurriculumProbe(
        parent_sequence=sequence,
        candidate_group=candidate,
        replicate=replicate,
        incoming_loss=10.0,
        best_loss=10.0 - improvement * updates,
        executed_updates=updates,
        probe_updates=updates,
        finite=finite,
        parent_state_hash="a" * 64,
    )


def test_prerequisite_graph_rejects_unknown_groups_and_cycles() -> None:
    with pytest.raises(ValueError, match="unknown prerequisites"):
        search_neutra_curriculum(
            groups=(NeuTraCurriculumGroup("x", ("missing",)),),
            probe_fn=lambda *_args: _probe((), "x", 0),
            config=NeuTraCurriculumSearchConfig(probe_updates=4, maximum_probe_calls=4),
        )
    with pytest.raises(ValueError, match="contains a cycle"):
        search_neutra_curriculum(
            groups=(
                NeuTraCurriculumGroup("x", ("y",)),
                NeuTraCurriculumGroup("y", ("x",)),
            ),
            probe_fn=lambda *_args: _probe((), "x", 0),
            config=NeuTraCurriculumSearchConfig(probe_updates=4, maximum_probe_calls=8),
        )


def test_uncertainty_bound_rejects_noisy_or_weak_group() -> None:
    values = {
        ((), "weak", 0): 0.5,
        ((), "weak", 1): -0.5,
        ((), "weak", 2): 0.5,
        ((), "weak", 3): -0.5,
    }

    def probe(sequence, candidate, replicate):
        return _probe(
            sequence,
            candidate,
            replicate,
            improvement=values[(sequence, candidate, replicate)],
        )

    result = search_neutra_curriculum(
        groups=(NeuTraCurriculumGroup("weak"),),
        probe_fn=probe,
        config=NeuTraCurriculumSearchConfig(
            probe_updates=4,
            probe_replicates=4,
            maximum_probe_calls=4,
            critical_value=2.0,
            minimum_improvement_per_update=0.1,
        ),
    )
    assert result.representative_sequence is None
    assert result.candidates[0].passed is False
    assert result.candidates[0].rejection_reason == "uncertainty_bound_below_threshold"
    assert result.stop_reason == "no_eligible_group_passed"


def test_beam_search_preserves_viable_sequences_and_uses_common_parent() -> None:
    groups = (
        NeuTraCurriculumGroup("a"),
        NeuTraCurriculumGroup("b"),
        NeuTraCurriculumGroup("c", ("a",)),
        NeuTraCurriculumGroup("d", ("b",)),
    )

    def probe(sequence, candidate, replicate):
        improvement = 0.4 if candidate in {"a", "b"} else 0.3
        return _probe(sequence, candidate, replicate, improvement=improvement)

    result = search_neutra_curriculum(
        groups=groups,
        probe_fn=probe,
        config=NeuTraCurriculumSearchConfig(
            probe_updates=4,
            probe_replicates=2,
            beam_width=2,
            maximum_depth=2,
            maximum_probe_calls=16,
            minimum_improvement_per_update=0.1,
        ),
    )
    assert result.probe_calls == 12
    assert result.representative_sequence == ("a", "b")
    assert result.final_beam_sequences == (("a", "b"), ("b", "a"))
    assert ("a", "c") in result.viable_sequences
    assert ("b", "d") in result.viable_sequences


def test_probe_contract_rejects_wrong_identity_and_budget() -> None:
    def wrong_identity(_sequence, _candidate, _replicate):
        return _probe(("wrong",), "wrong", 0)

    with pytest.raises(NeuTraCurriculumSearchError, match="wrong node identity"):
        search_neutra_curriculum(
            groups=(NeuTraCurriculumGroup("a"),),
            probe_fn=wrong_identity,
            config=NeuTraCurriculumSearchConfig(probe_updates=4, maximum_probe_calls=4),
        )

    def wrong_budget(sequence, candidate, replicate):
        return _probe(sequence, candidate, replicate, updates=3)

    with pytest.raises(NeuTraCurriculumSearchError, match="exact declared budget"):
        search_neutra_curriculum(
            groups=(NeuTraCurriculumGroup("a"),),
            probe_fn=wrong_budget,
            config=NeuTraCurriculumSearchConfig(probe_updates=4, maximum_probe_calls=4),
        )


def test_competing_candidates_must_share_parent_evidence_by_replicate() -> None:
    def probe(sequence, candidate, replicate):
        observation = _probe(sequence, candidate, replicate)
        if candidate == "b":
            return NeuTraCurriculumProbe(
                parent_sequence=observation.parent_sequence,
                candidate_group=observation.candidate_group,
                replicate=observation.replicate,
                incoming_loss=observation.incoming_loss,
                best_loss=observation.best_loss,
                executed_updates=observation.executed_updates,
                probe_updates=observation.probe_updates,
                finite=observation.finite,
                parent_state_hash="b" * 64,
            )
        return observation

    with pytest.raises(NeuTraCurriculumSearchError, match="same parent evidence"):
        search_neutra_curriculum(
            groups=(NeuTraCurriculumGroup("a"), NeuTraCurriculumGroup("b")),
            probe_fn=probe,
            config=NeuTraCurriculumSearchConfig(
                probe_updates=4,
                probe_replicates=2,
                maximum_probe_calls=4,
            ),
        )


def test_probe_budget_exhaustion_is_explicit_and_no_partial_candidate_is_recorded() -> None:
    calls = []

    def probe(sequence, candidate, replicate):
        calls.append((sequence, candidate, replicate))
        return _probe(sequence, candidate, replicate)

    result = search_neutra_curriculum(
        groups=(NeuTraCurriculumGroup("a"), NeuTraCurriculumGroup("b")),
        probe_fn=probe,
        config=NeuTraCurriculumSearchConfig(
            probe_updates=4,
            probe_replicates=2,
            maximum_probe_calls=2,
            minimum_improvement_per_update=0.1,
        ),
    )
    assert result.probe_calls == 2
    assert len(result.candidates) == 1
    assert result.stop_reason == "probe_budget_exhausted"
    assert calls == [((), "a", 0), ((), "a", 1)]


def test_nonfinite_probe_is_rejected_without_becoming_a_viable_sequence() -> None:
    def probe(sequence, candidate, replicate):
        return _probe(sequence, candidate, replicate, finite=False)

    result = search_neutra_curriculum(
        groups=(NeuTraCurriculumGroup("a"),),
        probe_fn=probe,
        config=NeuTraCurriculumSearchConfig(probe_updates=4, maximum_probe_calls=4),
    )
    assert result.representative_sequence is None
    assert result.candidates[0].rejection_reason == "nonfinite_probe"


def _protocol(name, sequence, replicate, loss, *, updates=100, partition=None):
    return NeuTraProtocolObservation(
        name=name,
        sequence=sequence,
        replicate=replicate,
        terminal_loss=loss,
        executed_updates=updates,
        update_budget=updates,
        finite=True,
        selection_partition_id=partition or f"partition-{replicate}",
    )


def test_full_protocol_selector_uses_paired_uncertainty_set_then_complexity() -> None:
    observations = []
    for replicate, reference in enumerate((1.00, 1.10, 0.90, 1.05)):
        observations.extend(
            (
                _protocol("cold", (), replicate, reference),
                _protocol("short", ("a",), replicate, reference + 0.01),
                _protocol("long", ("a", "b"), replicate, reference + 0.005),
                _protocol("bad", ("c",), replicate, reference + 0.20),
            )
        )
    result = select_neutra_protocol(
        observations=observations,
        config=NeuTraProtocolSelectionConfig(
            replicates=4,
            critical_value=2.0,
            practical_loss_tolerance=0.02,
        ),
    )
    assert result.reference_name == "cold"
    assert result.uncertainty_set == ("cold", "long", "short")
    assert result.selected_name == "cold"
    assert result.selected_sequence == ()
    assert not next(row for row in result.comparisons if row.name == "bad").in_uncertainty_set


def test_full_protocol_selector_rejects_unequal_budgets_and_unpaired_partitions() -> None:
    unequal = [
        _protocol("a", ("a",), replicate, 1.0, updates=100)
        for replicate in range(2)
    ] + [
        _protocol("b", ("b",), replicate, 1.0, updates=90)
        for replicate in range(2)
    ]
    with pytest.raises(NeuTraCurriculumSearchError, match="update budgets are not equal"):
        select_neutra_protocol(
            observations=unequal,
            config=NeuTraProtocolSelectionConfig(replicates=2),
        )
    unpaired = [
        _protocol("a", ("a",), replicate, 1.0)
        for replicate in range(2)
    ] + [
        _protocol("b", ("b",), replicate, 1.0, partition=f"other-{replicate}")
        for replicate in range(2)
    ]
    with pytest.raises(NeuTraCurriculumSearchError, match="paired selection partitions"):
        select_neutra_protocol(
            observations=unpaired,
            config=NeuTraProtocolSelectionConfig(replicates=2),
        )

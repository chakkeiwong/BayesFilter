from __future__ import annotations

import itertools

import pytest

from bayesfilter.inference.hmc_operational_broad_grid import (
    select_operational_candidate_union,
)


LINEAGE = {
    "metric_signature": "metric",
    "coordinate_signature": "coordinate",
    "lineage_signature": "lineage",
}


def _candidate(leapfrog: int, epsilon: float, *, viable: bool = True):
    return {
        "anchor_l": 4,
        "num_leapfrog_steps": leapfrog,
        "tuned_step_size": epsilon,
        "viable": viable,
        "content_signature": f"candidate-{leapfrog}",
        **LINEAGE,
    }


def test_union_selection_is_permutation_invariant_and_preserves_all_candidates():
    records = (
        _candidate(2, 0.2),
        _candidate(3, 0.3),
        _candidate(4, 0.4),
        _candidate(13, 0.5),
        _candidate(14, 0.6),
    )
    selected = set()
    for permutation in itertools.permutations(records):
        result = select_operational_candidate_union(
            permutation,
            anchor_l=4,
            expected_lineage=LINEAGE,
        )
        chosen = result.candidate_records[result.selected_index]
        selected.add((chosen["num_leapfrog_steps"], chosen["tuned_step_size"]))
        assert len(result.candidate_records) == 5
        assert set(result.selection_order) == set(range(5))
        assert result.stochastic_ranking_performed is False
    assert selected == {(4, 0.4)}


def test_union_selection_uses_policy_order_not_acceptance_or_runtime_fields():
    records = (
        {**_candidate(3, 0.3), "grand_mean": 0.99, "runtime": 1.0},
        {**_candidate(4, 0.4, viable=False), "grand_mean": 0.70, "runtime": 0.1},
        {**_candidate(13, 0.5), "grand_mean": 0.70, "runtime": 1000.0},
    )
    result = select_operational_candidate_union(
        records,
        anchor_l=4,
        expected_lineage=LINEAGE,
    )
    chosen = result.candidate_records[result.selected_index]
    assert chosen["num_leapfrog_steps"] == 3
    assert result.payload()["selection_rule"] == (
        "closest_to_anchor_then_lower_l_then_content_signature"
    )


def test_union_selection_can_close_without_a_viable_candidate():
    result = select_operational_candidate_union(
        (_candidate(3, 0.3, viable=False), _candidate(4, 0.4, viable=False)),
        anchor_l=4,
        expected_lineage=LINEAGE,
    )
    assert result.disposition == "no_viable_candidate"
    assert result.selected_index is None


def test_union_selection_rejects_duplicate_l_and_lineage_drift():
    with pytest.raises(ValueError, match="duplicate L"):
        select_operational_candidate_union(
            (_candidate(4, 0.3), _candidate(4, 0.4)),
            anchor_l=4,
            expected_lineage=LINEAGE,
        )
    drifted = {**_candidate(4, 0.4), "metric_signature": "changed"}
    with pytest.raises(ValueError, match="lineage mismatch"):
        select_operational_candidate_union(
            (drifted,),
            anchor_l=4,
            expected_lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("tuned_step_size", 0.0),
        ("viable", 1),
        ("anchor_l", 9),
    ),
)
def test_union_selection_rejects_malformed_records(field, value):
    record = _candidate(4, 0.4)
    record[field] = value
    with pytest.raises(ValueError):
        select_operational_candidate_union(
            (record,),
            anchor_l=4,
            expected_lineage=LINEAGE,
        )

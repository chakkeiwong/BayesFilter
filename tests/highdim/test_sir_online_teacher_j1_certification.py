from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/run_sir_online_teacher_j1_certification.py"
SPEC = importlib.util.spec_from_file_location("sir_online_teacher_j1_certification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def _summary(contains_zero: bool):
    return {"contains_zero": contains_zero}


def _row(particle_count: int, *, contains_zero: bool = True, finite: bool = True):
    return {
        "particle_count": particle_count,
        "all_finite": finite,
        "maximum_backward_row_sum_error": 0.0,
        "value_summary": _summary(contains_zero),
        "score_summaries": [_summary(contains_zero) for _ in range(3)],
    }


def test_observation_count_maps_to_dense_transition_count() -> None:
    assert campaign._dense_transition_count(1) == 0
    assert campaign._dense_transition_count(2) == 1


def test_simultaneous_summary_is_on_teacher_minus_reference() -> None:
    centered = campaign._summary(
        [0.9, 1.1, 1.0, 1.0],
        reference=1.0,
        reference_diagnostic_uncertainty=0.0,
    )
    shifted = campaign._summary(
        [1.9, 2.1, 2.0, 2.0],
        reference=1.0,
        reference_diagnostic_uncertainty=0.0,
    )
    assert centered["contains_zero"] is True
    assert shifted["contains_zero"] is False
    assert centered["bonferroni_family_size"] == 4


def test_largest_rung_controls_mismatch_classification() -> None:
    rows = [_row(64, contains_zero=False), _row(256, contains_zero=True)]
    assert campaign._classify_largest_rung(rows) == (
        "NO_TEACHER_J1_DISAGREEMENT_DETECTED_AT_CURRENT_PRECISION"
    )
    rows[-1] = _row(256, contains_zero=False)
    assert campaign._classify_largest_rung(rows) == "BLOCK_TEACHER_J1_DISAGREEMENT"


def test_nonfinite_and_backward_normalization_fail_closed() -> None:
    assert campaign._classify_largest_rung([_row(256, finite=False)]) == (
        "BLOCK_TEACHER_NONFINITE"
    )
    row = _row(256)
    row["maximum_backward_row_sum_error"] = 1.0e-6
    assert campaign._classify_largest_rung([row]) == (
        "BLOCK_TEACHER_BACKWARD_NORMALIZATION"
    )


def test_frozen_oracle_uses_observation_count_convention() -> None:
    assert campaign.FROZEN_ORACLE[1]["value"] == -0.37337136725883546
    assert campaign.FROZEN_ORACLE[2]["value"] == -0.8570589548006784
    assert campaign.DEFAULT_PARTICLE_COUNTS == (64, 128, 256)
    assert campaign.DEFAULT_REPLICATES == 16


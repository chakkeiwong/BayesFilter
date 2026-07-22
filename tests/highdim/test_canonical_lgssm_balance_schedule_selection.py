from __future__ import annotations

import inspect

from docs.benchmarks import run_canonical_lgssm_balance_schedule_selection as selection


def test_selection_design_is_frozen_and_disjoint() -> None:
    assert selection.BALANCE_CANDIDATES == (0, 1, 2, 5, 10, 20, 50, 100)
    assert selection.DESIGN_SEEDS == tuple(range(81300, 81308))
    assert selection.AUDIT_SEEDS == tuple(range(81320, 81328))
    assert set(selection.DESIGN_SEEDS).isdisjoint(selection.AUDIT_SEEDS)
    assert selection.TIME_STEPS == 2
    assert selection.NUM_PARTICLES == 128


def test_selection_source_has_no_kalman_or_score_dependency() -> None:
    source = inspect.getsource(selection)
    forbidden = (
        "kalman_tf",
        "tf_kalman",
        "GradientTape",
        "canonical_value_and_score_core",
        "_canonical_manual_jvp_core",
    )
    assert not any(token in source for token in forbidden)
    assert "_canonical_primal_core" in source
    assert '"selection_uses_kalman": False' in source


def test_selection_requires_audit_without_retuning() -> None:
    source = inspect.getsource(selection.main)
    assert "break" in source
    assert "_evaluate(AUDIT_SEEDS, selected)" in source
    assert "selected_schedule_audit_failed_no_retuning" in source

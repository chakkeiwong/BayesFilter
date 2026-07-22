from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/run_contract_e_phase8_reset_bias_n_scaling.py"
SPEC = importlib.util.spec_from_file_location("contract_e_reset_bias_scaling", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def _rows(*columns: list[float]) -> list[list[float]]:
    return [list(row) for row in zip(*columns, strict=True)]


def test_reviewed_roster_and_numerical_tuple_are_frozen() -> None:
    assert diagnostic.PARTICLE_COUNTS == (32, 64, 128)
    assert diagnostic.RESET_POLICIES == (
        "all_active_contract_e",
        "no_reset_weighted",
    )
    assert diagnostic.RIDGE == 0.1225 * 2.0**-24
    assert diagnostic.STEPS == 20
    assert diagnostic.CHUNKS == 16


def test_shared_finite_n_pattern_requires_all_six_quantities() -> None:
    decreasing = [0.3, 0.2, 0.1]
    errors = {
        "all_active_contract_e": _rows(*([decreasing] * 6)),
        "no_reset_weighted": _rows(*([decreasing] * 6)),
    }
    effects = _rows(*([[0.2, 0.1, 0.05]] * 6))
    classification, _ = diagnostic._classify(errors, effects)
    assert classification == "shared_finite_N_pattern"


def test_reset_specific_pattern_requires_all_no_reset_quantities_to_improve() -> None:
    contract = [0.3, 0.35, 0.4]
    no_reset = [0.3, 0.2, 0.1]
    errors = {
        "all_active_contract_e": _rows(*([contract] * 6)),
        "no_reset_weighted": _rows(*([no_reset] * 6)),
    }
    effects = _rows(*([[0.1, 0.15, 0.2]] * 6))
    classification, _ = diagnostic._classify(errors, effects)
    assert classification == "reset_specific_pattern"


def test_any_mixed_component_forces_inconclusive() -> None:
    decreasing = [0.3, 0.2, 0.1]
    mixed = [0.3, 0.31, 0.2]
    errors = {
        "all_active_contract_e": _rows(
            decreasing, decreasing, decreasing, mixed, decreasing, decreasing
        ),
        "no_reset_weighted": _rows(*([decreasing] * 6)),
    }
    effects = _rows(*([[0.2, 0.1, 0.05]] * 6))
    classification, _ = diagnostic._classify(errors, effects)
    assert classification == "mixed_or_nonmonotone_inconclusive"


def test_strict_improvement_rejects_ties() -> None:
    assert diagnostic._strict_improvement([0.3, 0.3, 0.2]) == [False, True]
    assert diagnostic._nonincreasing([0.3, 0.3, 0.2]) == [True, True]

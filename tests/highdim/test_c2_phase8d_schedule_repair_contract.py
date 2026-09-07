"""Contract checks for the bounded Phase 8D schedule-repair diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "docs/benchmarks/run_c2_phase8d_schedule_repair_diagnostic_20260907.py"
SPEC = importlib.util.spec_from_file_location("c2_phase8d_schedule_repair", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repair_variants_are_explicit_fixed_valid_schedules() -> None:
    identifiers = [str(row["config_id"]) for row in MODULE.SCHEDULE_VARIANTS]
    assert len(identifiers) == len(set(identifiers))
    for row in MODULE.SCHEDULE_VARIANTS:
        temperatures = tuple(float(value) for value in row["tempering_schedule"])
        steps = tuple(float(value) for value in row["step_fractions"])
        assert temperatures
        assert len(temperatures) == len(steps)
        assert temperatures[-1] == 1.0
        assert all(left <= right for left, right in zip(temperatures, temperatures[1:]))
        assert all(0.0 < value <= 1.0 for value in temperatures + steps)


def test_repair_markdown_preserves_failed_variant_without_ess() -> None:
    payload = {
        "status": "NO_SCHEDULE_PASSED",
        "fixture_sha256": "fixture",
        "proposal_seed": 106392,
        "particle_count": 8192,
        "variants": [
            {
                "config_id": "quarter_long",
                "status": "CANDIDATE_FAILURE",
                "maximum_relative_stationarity_residual": 6.1e-9,
                "invalid_residual_row_count": 2,
                "invalid_time_indices": [5],
                "minimum_precision_eigenvalue": 1.0,
                "minimum_covariance_eigenvalue": 0.1,
                "minimum_objective_improvement": 0.0,
                "wall_seconds": 1.0,
            }
        ],
    }
    rendered = MODULE._markdown(payload)
    assert "quarter_long" in rendered
    assert "CANDIDATE_FAILURE" in rendered
    assert "No ESS or heuristic value" in rendered

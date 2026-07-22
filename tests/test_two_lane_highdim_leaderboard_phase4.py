from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_two_lane_highdim_leaderboard", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase4_fixed_sgqf_predator_prey_blocks_p47_value_as_t20_source_scope() -> None:
    module = _load_module()
    row = module._apply_score_status(module._cell_for_fixed_sgqf("zhao_cui_predator_prey_T20"))

    assert row["comparison_status"] == "blocked"
    assert row["numeric_execution_status"] == "blocked_predator_prey_sgqf_value"
    assert row["target_contract_status"] == "blocked_missing_t20_fixed_sgqf_evaluator"
    assert row["score_status"] == "blocked_target_alignment"
    assert row["average_log_likelihood"] is None
    assert row["log_likelihood"] is None
    assert row["score"] is None
    assert row["score_coordinate_system"] is None
    assert row["score_derivative_provenance"] is None
    assert "PREDATOR_PREY_T20_FIXED_SGQF_EVALUATOR_REQUIRED" in row["reason_codes"]
    assert "blocked_target_alignment" in row["reason"]
    assert "transitions before y0" in row["reason"]
    assert "assimilates y0 before the first transition" in row["reason"]
    assert any("P47 two-observation lower-rung" in item for item in row["nonclaims"])

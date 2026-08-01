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


def test_phase4_fixed_sgqf_predator_prey_executes_source_order_t20_value_score() -> None:
    module = _load_module()
    row = module._apply_score_status(module._cell_for_fixed_sgqf("zhao_cui_predator_prey_T20"))

    assert row["comparison_status"] == "executed_value_score"
    assert row["numeric_execution_status"] == "executed_predator_prey_sgqf_value_score"
    assert row["target_contract_status"] == "target_compatible_source_order_predator_prey_t20_sgqf"
    assert row["score_status"] == "analytical_score_emitted"
    assert abs(row["log_likelihood"] - (-102.62270352134469)) < 1e-10
    assert len(row["score"]) == 6
    assert row["score_coordinate_system"] == "physical=(r,K,a,s,u,v)"
    assert "manual" in row["score_derivative_provenance"]
    assert row["time_order"] == "x0_then_20_transition_then_observe_steps_y1_y20"
    assert len(row["route_identity"]) == 64
    assert "DIRECT_PREDATOR_PREY_SGQF_VALUE_SCORE_ROUTE" in row["reason_codes"]
    assert any("P47 two-observation lower-rung" in item for item in row["nonclaims"])

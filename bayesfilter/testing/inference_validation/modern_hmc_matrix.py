"""Framework-free required nonlinear comparison inventory, not run evidence.

Historical registry recipes are deliberately not loaded. Each cell needs a
current numerical target, score, geometry and reference check before execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NonlinearComparisonTarget:
    target_id: str
    family: str
    parameter_dim: int | None
    module: str
    factory: str | None
    requirement: str


NONLINEAR_COMPARISON_TARGETS = (
    NonlinearComparisonTarget("NL-ACC", "accumulation", 3,
        "bayesfilter.testing.simple_nonlinear_generic_target_adapter_tf",
        "make_simple_nonlinear_generic_target_fixture", "core"),
    NonlinearComparisonTarget("NL-PP-UKF", "predator_prey", 6,
        "bayesfilter.testing.predator_prey_ukf_neutra_target_tf",
        "make_predator_prey_ukf_neutra_adapter", "core"),
    NonlinearComparisonTarget("NL-PP-SGQF", "predator_prey", 6,
        "bayesfilter.testing.predator_prey_sgqf_neutra_target_tf",
        "make_predator_prey_sgqf_neutra_adapter", "core"),
    NonlinearComparisonTarget("NL-SIR", "sir", 3,
        "bayesfilter.testing.sir_filter_neutra_target_design_tf",
        "make_sir_sgqf_neutra_adapter", "core"),
    NonlinearComparisonTarget("NL-STR", "structural", 5,
        "bayesfilter.testing.structural_ukf_neutra_target_design_tf",
        "make_structural_ukf_neutra_adapter", "core"),
    NonlinearComparisonTarget("NL-GROWTH", "nonlinear_growth", None,
        "bayesfilter.testing.nonlinear_models_tf", None, "posterior_adapter_needed"),
    NonlinearComparisonTarget("NL-SV", "stochastic_volatility", 2,
        "bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf",
        "make_actual_sv_zc_neutra_adapter", "source_route_eligibility_needed"),
)
CORE_METHODS = ("fixed_hmc", "randomized_hmc", "snaper_hmc", "nuts", "mams")
DISPOSITIONS = frozenset({"pending", "unsupported", "repair_needed", "inconclusive",
                          "failed", "completed", "unfunded"})


def initial_comparison_matrix(methods: Sequence[str] = CORE_METHODS) -> list[dict[str, Any]]:
    if not methods or len(set(methods)) != len(methods) or any(m not in CORE_METHODS for m in methods):
        raise ValueError("declare distinct registered methods")
    return [{"target": target.target_id, "method": method, "status": "pending",
             "reason": "current_target_score_geometry_reference_and_method_checks_pending",
             "result_path": None}
            for target in NONLINEAR_COMPARISON_TARGETS if target.requirement == "core"
            for method in methods]


def validate_comparison_matrix(rows: Sequence[Mapping[str, Any]], *,
                               methods: Sequence[str] = CORE_METHODS) -> dict[str, Any]:
    expected = {(r["target"], r["method"]) for r in initial_comparison_matrix(methods)}
    seen = set()
    for row in rows:
        pair = row["target"], row["method"]
        if pair not in expected or pair in seen:
            raise ValueError("duplicate or unsupported comparison cell")
        seen.add(pair)
        if row.get("status") not in DISPOSITIONS or not str(row.get("reason", "")).strip():
            raise ValueError("every comparison needs an explicit disposition and reason")
        if row["status"] == "completed" and not row.get("result_path"):
            raise ValueError("completed comparison requires preserved evidence")
    if seen != expected:
        raise ValueError("required nonlinear comparison cells are missing")
    return {"schema": "bayesfilter.modern_hmc_comparison_inventory.v1",
            "targets": [asdict(t) for t in NONLINEAR_COMPARISON_TARGETS],
            "rows": [dict(r) for r in rows],
            "inventory_complete": True,
            "all_runs_completed": all(r["status"] == "completed" for r in rows),
            "scientific_promotion_authority": False}

#!/usr/bin/env python3
"""Build the Phase 6 row-eligibility ledger from controlling artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
    "phase6_zhao_cui_comparators"
)
OUTPUT = ARTIFACT_ROOT / "phase6_comparator_eligibility_ledger_v2.json"
PLAN = (
    "docs/plans/"
    "bayesfilter-contract-e-tp-phase6-zhao-cui-comparator-certification-plan-2026-07-15.md"
)


CONTROLLING = {
    "actual_sv": {
        horizon: (
            f"actual_sv_t{horizon}_degree8_order17_rank2_reclassified_result.json"
        )
        for horizon in (1, 2, 10)
    },
    "ksc_sv": {
        horizon: f"ksc_sv_t{horizon}_degree8_order17_rank2_reclassified_result.json"
        for horizon in (1, 2, 10)
    },
    "generalized_sv": {
        horizon: (
            f"generalized_sv_t{horizon}_degree8_order17_rank2_reclassified_result.json"
        )
        for horizon in (1, 2, 10)
    },
}


def _load(relative: str) -> dict:
    return json.loads((ARTIFACT_ROOT / relative).read_text(encoding="utf-8"))


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _scalar_row(row: str, files: dict[int, str]) -> dict:
    results = {horizon: _load(path) for horizon, path in files.items()}
    statuses = {result["status"] for result in results.values()}
    if len(statuses) != 1:
        raise ValueError(f"inconsistent controlling statuses for {row}: {statuses}")
    for horizon, result in results.items():
        if result["horizon"] != horizon:
            raise ValueError(f"horizon mismatch for {row} T={horizon}")
        if result["own_scalar_fd"]["status"] != "pass":
            raise ValueError(f"own-scalar FD did not pass for {row} T={horizon}")
        vetoes = result["hard_vetoes"]
        if not (
            vetoes["finite_value_and_score"]
            and vetoes["previous_state_axis_present_from_t1"]
            and vetoes["carried_marginal_mass_valid"]
            and not vetoes["forbidden_retained_grid_route_used"]
            and not vetoes["oracle_alias_used"]
        ):
            raise ValueError(f"hard veto failed for {row} T={horizon}")
    return {
        "row": row,
        "row_id": results[1]["row_id"],
        "status": next(iter(statuses)),
        "route_id": results[1]["route_id"],
        "route_classification": results[1]["route_classification"],
        "route_subtype": results[1]["route_subtype"],
        "horizons": {
            str(horizon): {
                "artifact": str((ARTIFACT_ROOT / files[horizon]).relative_to(ROOT)),
                "value": result["value"],
                "score": result["score"],
                "max_own_scalar_fd_relative_error": result["own_scalar_fd"][
                    "owner_fd_only_policy"
                ]["max_coordinate_relative_error"],
                "fd_threshold": result["own_scalar_fd"][
                    "owner_fd_only_policy"
                ]["max_coordinate_relative_error_threshold"],
                "max_marginal_mass_error": max(
                    result["hard_vetoes"]["carried_marginal_mass_errors"]
                ),
                "max_fit_residual": max(
                    step["fit_residual"] for step in result["finite_program"]["steps"]
                ),
                "max_scaled_condition_number": max(
                    condition
                    for step in result["finite_program"]["steps"]
                    for condition in step["fit_condition_numbers"]
                    if isinstance(condition, (int, float))
                ),
            }
            for horizon, result in results.items()
        },
        "phase7_eligibility": (
            "eligible_as_extension_not_zhaocui_source_comparator"
            if row in ("actual_sv", "ksc_sv")
            else "engineering_certified_but_contract_e_tp_row_negative_result"
        ),
        "what_is_not_concluded": results[10]["what_is_not_concluded"],
    }


def build() -> dict:
    rows = [_scalar_row(row, files) for row, files in CONTROLLING.items()]
    rows.extend(
        [
            {
                "row": "lgssm",
                "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
                "status": "zhaocui_comparator_unavailable",
                "reason": "current leaderboard cell aliases the Kalman oracle; no multidimensional fixed adjacent-state TT route exists",
                "forbidden_substitution": "Kalman oracle adapter",
                "phase7_eligibility": "not_eligible",
            },
            {
                "row": "predator_prey",
                "row_id": "zhao_cui_predator_prey_T20",
                "status": "zhaocui_comparator_unavailable",
                "reason": "current executable helper uses the forbidden generic retained-grid multistate route",
                "forbidden_substitution": "multistate_nonlinear_fixed_design_tt_*",
                "phase7_eligibility": "not_eligible_for_zhaocui_pair",
            },
            {
                "row": "sir_austria",
                "row_id": "zhao_cui_spatial_sir_austria_j9_T20",
                "status": "blocked_target_measure_mismatch",
                "reason": "clipped simulator law and Gaussian transition density are different measures; full observed-data total score remains unavailable",
                "retained_scope": "P90/P91 local complete-data component evidence only",
                "phase7_eligibility": "not_eligible_for_observed_data_pair",
            },
        ]
    )
    return {
        "schema_version": "contract_e_tp.phase6.comparator_eligibility.v2",
        "metadata_date": "2026-07-15",
        "status": "phase6_complete_row_specific_outputs",
        "plan": PLAN,
        "git_commit": _git_commit(),
        "rows": rows,
        "repair_attempts": [
            {
                "artifact": str(
                    (
                        ARTIFACT_ROOT
                        / "actual_sv_t1_degree8_order17_rank2_result.json"
                    ).relative_to(ROOT)
                ),
                "classification": "harness_failure_enum_case_mismatch",
                "repair": "compare FiniteDifferenceRowStatus.value to VALID and preserve artifact",
            },
            {
                "artifact": str(
                    (
                        ARTIFACT_ROOT
                        / "generalized_sv_t2_degree8_order17_rank2_result.json"
                    ).relative_to(ROOT)
                ),
                "classification": "algorithmic_derivative_failure_rank_deficient_initializer",
                "repair": "norm-balanced independent orthonormal polynomial-mode initializer",
            },
            {
                "artifact": str(
                    (
                        ARTIFACT_ROOT
                        / "generalized_sv_t2_degree8_order17_rank2_attempt2_result.json"
                    ).relative_to(ROOT)
                ),
                "classification": "failed_hypothesis_conditioning_only",
                "repair": "QR pullback retained, but root cause required initializer repair",
            },
        ],
        "global_nonclaims": [
            "not adaptive TT-cross or TTSIRT reproduction",
            "not source-faithful",
            "not exact filtering or cross-method equivalence",
            "not statistical ranking",
            "not HMC, default, leaderboard, GPU, or full-horizon readiness",
            "no row currently has a source-route Zhao-Cui parameter-learning comparator",
        ],
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing ledger: {OUTPUT}")
    artifact = build()
    OUTPUT.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": artifact["status"], "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()

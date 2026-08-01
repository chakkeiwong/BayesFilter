#!/usr/bin/env python3
"""Build the Phase 7 same-target comparison from immutable result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15"
)
PLAN = (
    "docs/plans/"
    "bayesfilter-contract-e-tp-phase7-same-target-all-model-comparison-plan-"
    "2026-07-15.md"
)

TP_ALGORITHM_ID = "contract_e_tp_experimental_v1"
EXTENSION_CLASSIFICATION = "extension_or_invention"
EXTENSION_SUBTYPE = "fixed_parameter_adjacent_state_squared_tt_extension"
NO_MARGIN = "descriptive_only_margin_unavailable"

LGSSM_ROW = "benchmark_lgssm_exact_oracle_m3_T50"
ACTUAL_ROW = "zhao_cui_sv_actual_nongaussian_T1000"
KSC_ROW = "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000"
GENERALIZED_ROW = "zhao_cui_generalized_sv_synthetic_from_estimated_values"
PREDATOR_ROW = "zhao_cui_predator_prey_T20"
SIR_ROW = "zhao_cui_spatial_sir_austria_j9_T20"

EXPECTED = {
    LGSSM_ROW: {
        "theta": [0.72, 0.55, 0.35, 0.35, 0.45],
        "parameter_names": ["phi1", "phi2", "phi3", "q_scale", "r_scale"],
    },
    ACTUAL_ROW: {
        "theta": [0.2533471031357997, -0.916290731874155],
        "parameter_names": ["gamma_unconstrained", "log_beta"],
        "target_observation_policy": "exact_log_y_square_log_chi_square",
        "transition_before_first_observation": False,
    },
    KSC_ROW: {
        "theta": [0.2533471031357997, -0.916290731874155],
        "parameter_names": ["gamma_unconstrained", "log_beta"],
        "target_observation_policy": "offset_log_y_square_ksc_mixture",
        "transition_before_first_observation": False,
    },
    GENERALIZED_ROW: {
        "theta": [1.0824113944610982, -2.076793740349318, 0.0],
        "parameter_names": ["gamma_unconstrained", "log_tau", "mu_over_tau"],
        "target_observation_policy": "raw_zero_mean_normal_generalized_sv",
        "transition_before_first_observation": True,
    },
    PREDATOR_ROW: {
        "theta": [0.6, 114.0, 25.0, 0.3, 0.5, 0.5],
        "parameter_names": ["r", "K", "a", "s", "u", "v"],
        "time_order": "initial_law_then_y0; transition_then_yt_for_t_positive",
    },
}

LGSSM_INPUTS = {
    2: (
        "phase8_continuation_information_v2_lgssm_t2_order5_attempt1_20260715/"
        "result.json"
    ),
    10: "phase8b_lgssm_t10_order5_lookahead8_attempt1_20260715/result.json",
    50: (
        "phase8b_lgssm_t50_order5_lookahead8_attempt1_20260715/"
        "result_aggregate.json"
    ),
}

TP_SCALAR_INPUTS = {
    (ACTUAL_ROW, 1): "phase7_actual_sv_t1_bound_target_result_20260715.json",
    (ACTUAL_ROW, 2): "phase7_actual_sv_t2_order41_lookahead1_result_20260715.json",
    (ACTUAL_ROW, 10): "phase5_actual_sv_t10_order41_lookahead8_result_20260715.json",
    (KSC_ROW, 1): "phase7_ksc_sv_t1_order41_current_target_result_20260715.json",
    (KSC_ROW, 2): "phase7_ksc_sv_t2_order41_lookahead1_result_20260715.json",
    (KSC_ROW, 10): "phase5_ksc_sv_t10_order41_lookahead8_result_20260715.json",
    (GENERALIZED_ROW, 1): "phase5_generalized_sv_t1_order25_timeorderfix_result_20260715.json",
    (GENERALIZED_ROW, 2): "phase7_generalized_sv_t2_order41_lookahead1_result_20260715.json",
    (GENERALIZED_ROW, 10): (
        "phase5_generalized_sv_t10_order41_progressive1_4_9_basis_quantile8_"
        "analytic_fill_localized_result_20260715.json"
    ),
    (PREDATOR_ROW, 2): "phase5_predator_prey_t2_order5_analytic_lookahead1_result_20260715.json",
    (PREDATOR_ROW, 5): (
        "phase5_predator_prey_t5_order5_gaussian_closure_lookahead4_"
        "stabilized_result_20260715.json"
    ),
}

EXTENSION_INPUTS = {
    (row_id, horizon): (
        "phase6_zhao_cui_comparators/"
        f"{row_name}_t{horizon}_degree8_order17_rank2_phase7_identity_repaired_result.json"
    )
    for row_id, row_name in (
        (ACTUAL_ROW, "actual_sv"),
        (KSC_ROW, "ksc_sv"),
        (GENERALIZED_ROW, "generalized_sv"),
    )
    for horizon in (1, 2, 10)
}
EXTENSION_INPUTS[(ACTUAL_ROW, 1)] = (
    "phase6_zhao_cui_comparators/"
    "actual_sv_t1_degree8_order17_rank2_phase7_bound_target_result.json"
)
for horizon in (1, 2, 10):
    EXTENSION_INPUTS[(KSC_ROW, horizon)] = (
        "phase6_zhao_cui_comparators/"
        f"ksc_sv_t{horizon}_degree8_order17_rank2_phase7_target_identity_repaired_result.json"
    )
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(relative: str) -> tuple[Path, dict[str, Any]]:
    path = ARTIFACT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing required Phase 7 input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return path, payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_scalar(value: Any, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _finite_vector(value: Any, size: int, label: str) -> list[float]:
    _require(isinstance(value, (tuple, list)), f"{label} is not a vector")
    result = [_finite_scalar(item, f"{label}[{index}]") for index, item in enumerate(value)]
    _require(len(result) == size, f"{label} has size {len(result)}, expected {size}")
    return result


def _same_vector(left: Any, right: Any, label: str) -> None:
    _require(isinstance(left, (tuple, list)), f"{label} is not a vector")
    _require(len(left) == len(right), f"{label} length mismatch")
    for index, (actual, expected) in enumerate(zip(left, right)):
        _require(
            float(actual) == float(expected),
            f"{label}[{index}] mismatch: {actual!r} != {expected!r}",
        )


def _differences(candidate: list[float], reference: list[float]) -> dict[str, Any]:
    difference = [left - right for left, right in zip(candidate, reference)]
    relative = [
        abs(delta) / max(abs(left), abs(right), 1.0e-12)
        for left, right, delta in zip(candidate, reference, difference)
    ]
    return {
        "score_difference": difference,
        "componentwise_relative_error": relative,
        "maximum_componentwise_relative_error": max(relative, default=0.0),
        "sign_reversal": [
            (left < 0.0 < right) or (right < 0.0 < left)
            for left, right in zip(candidate, reference)
        ],
        "equivalence_classification": NO_MARGIN,
    }


def _source(path: Path) -> dict[str, str]:
    return {"path": _relative_path(path), "sha256": _sha256(path)}


def _bound_preparation(
    payload: dict[str, Any], label: str
) -> tuple[dict[str, str], dict[str, Any]]:
    binding = payload.get("preparation")
    _require(isinstance(binding, dict), f"{label} preparation binding missing")
    path = ROOT / binding["path"]
    _require(path.is_file(), f"{label} preparation artifact missing: {path}")
    _require(
        _sha256(path) == binding.get("sha256"),
        f"{label} preparation SHA-256 mismatch",
    )
    preparation = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(preparation, dict), f"{label} preparation is not an object")
    return _source(path), preparation


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _method(
    *,
    name: str,
    value: float,
    score: list[float],
    source: dict[str, str],
    reference_value: float | None = None,
    reference_score: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "available",
        "method": name,
        "value": value,
        "score": score,
        "source": source,
    }
    if reference_value is not None and reference_score is not None:
        result["difference_to_reference"] = {
            "value_difference": value - reference_value,
            **_differences(score, reference_score),
        }
    if extra:
        result.update(extra)
    return result


def _lgssm_row(horizon: int, relative: str) -> dict[str, Any]:
    path, payload = _load(relative)
    _require(payload.get("status") == "PASS_ENGINEERING", f"LGSSM T={horizon} status failed")
    _require(
        int(payload.get("time_steps")) == horizon if "time_steps" in payload else True,
        f"LGSSM T={horizon} horizon mismatch",
    )
    preparation_source, preparation_payload = _bound_preparation(
        payload, f"LGSSM T={horizon}"
    )

    if "target" in payload:
        target = payload["target"]
        _require(target.get("row_id") == LGSSM_ROW, f"LGSSM T={horizon} row mismatch")
        _require(int(target.get("time_steps")) == horizon, f"LGSSM T={horizon} target horizon mismatch")
        _same_vector(target.get("theta"), EXPECTED[LGSSM_ROW]["theta"], "LGSSM theta")
        _require(
            target.get("parameter_names") == EXPECTED[LGSSM_ROW]["parameter_names"],
            "LGSSM parameter order mismatch",
        )
    else:
        preparation_path = ROOT / preparation_source["path"]
        target = preparation_payload["target"]
        _require(target.get("row_id") == LGSSM_ROW, "LGSSM aggregate row mismatch")
        _require(int(target.get("time_steps")) == horizon, "LGSSM aggregate horizon mismatch")
        _same_vector(target.get("theta"), EXPECTED[LGSSM_ROW]["theta"], "LGSSM aggregate theta")

    parameter_names = EXPECTED[LGSSM_ROW]["parameter_names"]
    size = len(parameter_names)
    value = payload["value"]
    score = payload["score"]
    candidate_value = _finite_scalar(value["contract_e_tp"], "LGSSM TP value")
    reference_value = _finite_scalar(value["kalman"], "LGSSM Kalman value")
    candidate_score = _finite_vector(score["contract_e_tp"], size, "LGSSM TP score")
    reference_score = _finite_vector(score["kalman"], size, "LGSSM Kalman score")
    _require(score.get("same_scalar_fd_pass") is True, f"LGSSM T={horizon} own-scalar FD failed")
    _require(payload["chart"].get("chart_pass", payload["chart"].get("valid")) is True,
             f"LGSSM T={horizon} chart failed")
    return {
        "row_id": LGSSM_ROW,
        "horizon": horizon,
        "theta": EXPECTED[LGSSM_ROW]["theta"],
        "parameter_names": parameter_names,
        "comparison_classification": NO_MARGIN,
        "reference": _method(
            name="differentiated_kalman_filter",
            value=reference_value,
            score=reference_score,
            source=_source(path),
            extra={"reference_status": "exact_linear_gaussian_oracle"},
        ),
        "contract_e_tp": _method(
            name=TP_ALGORITHM_ID,
            value=candidate_value,
            score=candidate_score,
            reference_value=reference_value,
            reference_score=reference_score,
            source=_source(path),
            extra={
                "own_scalar_fd_status": "pass",
                "chart_status": "pass",
                "feature_mode": payload.get("feature_mode"),
                "lookahead_steps": payload.get("lookahead_steps"),
                "preparation_source": preparation_source,
            },
        ),
        "contract_e_chol": _unavailable("no_admissible_same_target_campaign_artifact"),
        "fixed_parameter_adjacent_state_extension": _unavailable("not_implemented_for_lgssm"),
        "zhao_cui_source_parameter_learning": _unavailable("zhaocui_comparator_unavailable"),
    }


def _validate_scalar_target(payload: dict[str, Any], row_id: str, horizon: int) -> None:
    expected = EXPECTED[row_id]
    _require(payload.get("row_id") == row_id, f"{row_id} T={horizon} row mismatch")
    target = payload.get("target", {})
    _require(int(target.get("time_steps")) == horizon, f"{row_id} T={horizon} horizon mismatch")
    _same_vector(target.get("theta"), expected["theta"], f"{row_id} theta")
    _require(target.get("parameter_names") == expected["parameter_names"],
             f"{row_id} parameter order mismatch")
    if "target_observation_policy" in expected:
        _require(target.get("target_observation_policy") == expected["target_observation_policy"],
                 f"{row_id} target observation policy mismatch")
    if (
        "transition_before_first_observation" in expected
        and "transition_before_first_observation" in target
    ):
        _require(
            target.get("transition_before_first_observation")
            == expected["transition_before_first_observation"],
            f"{row_id} first-transition convention mismatch",
        )
    if "time_order" in expected:
        _require(target.get("time_order") == expected["time_order"], f"{row_id} time-order mismatch")


def _scalar_row(row_id: str, horizon: int, tp_relative: str) -> dict[str, Any]:
    tp_path, tp = _load(tp_relative)
    _require(tp.get("algorithm_id") == TP_ALGORITHM_ID, f"{row_id} T={horizon} TP identity mismatch")
    _require(tp.get("status") == "PASS_ENGINEERING", f"{row_id} T={horizon} TP status failed")
    _validate_scalar_target(tp, row_id, horizon)
    expected = EXPECTED[row_id]
    preparation_source, preparation_payload = _bound_preparation(
        tp, f"{row_id} T={horizon}"
    )
    preparation_target = preparation_payload["target"]
    if "target_observation_policy" in expected:
        _require(
            preparation_target.get("target_observation_policy")
            == expected["target_observation_policy"],
            f"{row_id} T={horizon} prepared target policy mismatch",
        )
        if "transition_before_first_observation" in preparation_target:
            _require(
                preparation_target["transition_before_first_observation"]
                == expected["transition_before_first_observation"],
                f"{row_id} T={horizon} prepared first-transition convention mismatch",
            )
    size = len(expected["parameter_names"])
    candidate = tp["candidate"]
    candidate_value = _finite_scalar(candidate["value"], f"{row_id} TP value")
    candidate_score = _finite_vector(candidate["score"], size, f"{row_id} TP score")
    _require(candidate["same_scalar_fd_policy"].get("status") == "pass",
             f"{row_id} T={horizon} TP own-scalar FD failed")
    _require(tp["chart"].get("chart_pass") is True, f"{row_id} T={horizon} TP chart failed")
    references = tp.get("dense_references")
    _require(isinstance(references, list) and len(references) >= 2,
             f"{row_id} T={horizon} reference ladder missing")
    finest = references[-1]
    reference_value = _finite_scalar(finest["value"], f"{row_id} reference value")
    reference_score = _finite_vector(finest["score"], size, f"{row_id} reference score")
    _require(
        tp["candidate_vs_finest_reference"].get("equivalence_classification") == NO_MARGIN,
        f"{row_id} T={horizon} has an unsupported equivalence classification",
    )
    row: dict[str, Any] = {
        "row_id": row_id,
        "horizon": horizon,
        "theta": expected["theta"],
        "parameter_names": expected["parameter_names"],
        "target_observation_policy": expected.get("target_observation_policy"),
        "transition_before_first_observation": expected.get("transition_before_first_observation"),
        "time_order": expected.get("time_order"),
        "comparison_classification": NO_MARGIN,
        "reference": _method(
            name=(
                "refined_dense_tensorflow_quadrature"
                if row_id != PREDATOR_ROW
                else (
                    "semianalytic_initial_gauss_hermite"
                    if horizon == 2
                    else "corrected_time_order_fixed_sgqf_approximation"
                )
            ),
            value=reference_value,
            score=reference_score,
            source=_source(tp_path),
            extra={
                "reference_status": tp["reference_refinement"]["classification"],
                "refinement": tp["reference_refinement"],
            },
        ),
        "contract_e_tp": _method(
            name=TP_ALGORITHM_ID,
            value=candidate_value,
            score=candidate_score,
            reference_value=reference_value,
            reference_score=reference_score,
            source=_source(tp_path),
            extra={
                "own_scalar_fd_status": "pass",
                "chart_status": "pass",
                "preparation": tp.get("preparation"),
                "preparation_source": preparation_source,
                "increment_history": candidate.get("increment_history"),
                "increment_score_history": candidate.get("increment_score_history"),
            },
        ),
        "contract_e_chol": _unavailable("no_admissible_same_target_campaign_artifact"),
        "zhao_cui_source_parameter_learning": _unavailable("zhaocui_comparator_unavailable"),
    }
    if row_id in (ACTUAL_ROW, KSC_ROW, GENERALIZED_ROW):
        extension_path, extension = _load(EXTENSION_INPUTS[(row_id, horizon)])
        _require(extension.get("status") == "certified_extension_or_invention",
                 f"{row_id} T={horizon} extension status failed")
        _require(extension.get("row_id") == row_id, f"{row_id} T={horizon} extension row mismatch")
        _require(int(extension.get("horizon")) == horizon, f"{row_id} T={horizon} extension horizon mismatch")
        _require(extension.get("route_classification") == EXTENSION_CLASSIFICATION,
                 f"{row_id} T={horizon} extension classification mismatch")
        _require(extension.get("route_subtype") == EXTENSION_SUBTYPE,
                 f"{row_id} T={horizon} extension subtype mismatch")
        _same_vector(extension.get("theta"), expected["theta"], f"{row_id} extension theta")
        _require(list(extension.get("parameter_names")) == expected["parameter_names"],
                 f"{row_id} extension parameter order mismatch")
        _require(extension["own_scalar_fd"].get("status") == "pass",
                 f"{row_id} T={horizon} extension own-scalar FD failed")
        extension_target = extension.get("target", {})
        _require(int(extension_target.get("time_steps")) == horizon,
                 f"{row_id} T={horizon} extension target horizon mismatch")
        _require(
            extension_target.get("target_observation_policy")
            == expected["target_observation_policy"],
            f"{row_id} T={horizon} extension observation policy mismatch",
        )
        _require(
            extension_target.get("target_observations_sha256")
            == preparation_target.get("target_observations_sha256"),
            f"{row_id} T={horizon} extension observation hash mismatch",
        )
        first_step = extension["finite_program"]["steps"][0]
        if expected["transition_before_first_observation"]:
            first_step_valid = bool(
                int(first_step["fit_dimension"]) == 2
                and tuple(first_step["axis_order"]) == ("x_t", "x_t_minus_1")
                and tuple(first_step["integrated_axes"]) == (1,)
                and first_step["target_kind"]
                == "transitioned_initial_adjacent_state_update"
            )
        else:
            first_step_valid = bool(
                int(first_step["fit_dimension"]) == 1
                and tuple(first_step["axis_order"]) == ("x_0",)
                and tuple(first_step["integrated_axes"]) == ()
                and first_step["target_kind"] == "initial_state_observation"
            )
        _require(first_step_valid, f"{row_id} T={horizon} extension first-step time order failed")
        if "first_step_time_order_valid" in extension["hard_vetoes"]:
            _require(extension["hard_vetoes"]["first_step_time_order_valid"] is True,
                     f"{row_id} T={horizon} explicit time-order veto failed")
            _require(
                extension["hard_vetoes"].get("transition_before_first_observation")
                == expected["transition_before_first_observation"],
                f"{row_id} T={horizon} extension first-transition convention mismatch",
            )
        extension_value = _finite_scalar(extension["value"], f"{row_id} extension value")
        extension_score = _finite_vector(extension["score"], size, f"{row_id} extension score")
        fit_residuals = [
            _finite_scalar(step["fit_residual"], f"{row_id} fit residual")
            for step in extension["finite_program"]["steps"]
        ]
        row["fixed_parameter_adjacent_state_extension"] = _method(
            name=EXTENSION_SUBTYPE,
            value=extension_value,
            score=extension_score,
            reference_value=reference_value,
            reference_score=reference_score,
            source=_source(extension_path),
            extra={
                "route_classification": EXTENSION_CLASSIFICATION,
                "own_scalar_fd_status": "pass",
                "first_step_time_order_status": "pass",
                "maximum_fit_residual": max(fit_residuals),
                "difference_to_contract_e_tp": {
                    "value_difference": extension_value - candidate_value,
                    **_differences(extension_score, candidate_score),
                },
            },
        )
    else:
        row["fixed_parameter_adjacent_state_extension"] = _unavailable(
            "not_implemented_for_predator_prey"
        )
    if row_id == GENERALIZED_ROW and horizon == 10:
        row["contract_e_tp"]["scientific_status"] = (
            "negative_result_tested_progressive_continuation_feature_family_insufficient"
        )
    return row


def build() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rows.extend(_lgssm_row(horizon, relative) for horizon, relative in LGSSM_INPUTS.items())
    rows.extend(
        _scalar_row(row_id, horizon, relative)
        for (row_id, horizon), relative in TP_SCALAR_INPUTS.items()
    )
    rows.append(
        {
            "row_id": SIR_ROW,
            "horizon": 20,
            "status": "blocked_target_measure_mismatch",
            "reason": (
                "author simulator clips susceptible states after Gaussian noise while "
                "the declared transition density is an unclipped Gaussian; no single "
                "observed-data scalar is frozen"
            ),
            "reference": _unavailable("observed_data_target_not_frozen"),
            "contract_e_tp": _unavailable("blocked_target_measure_mismatch"),
            "contract_e_chol": _unavailable("no_admissible_same_target_campaign_artifact"),
            "fixed_parameter_adjacent_state_extension": _unavailable("not_implemented"),
            "zhao_cui_source_parameter_learning": _unavailable(
                "component_evidence_is_not_observed_data_total_score"
            ),
            "comparison_classification": "comparison_invalid_target_not_frozen",
        }
    )
    return {
        "schema": "bayesfilter.contract_e_tp.phase7_same_target_comparison.v1",
        "metadata_date": "2026-07-15",
        "status": "PHASE7_COMPARISON_COMPLETE_WITH_EXPLICIT_GAPS",
        "program_id": "contract-e-tp-all-model-gradient-comparison",
        "plan": PLAN,
        "comparison_scope": "center_only_deterministic_descriptive_same_target",
        "row_count": len(rows),
        "rows": rows,
        "method_eligibility": {
            "contract_e_tp": "experimental_candidate_not_canonical",
            "contract_e_chol": "canonical_policy_route_but_no_admissible_phase7_artifact",
            "fixed_parameter_adjacent_state_extension": EXTENSION_CLASSIFICATION,
            "zhao_cui_source_parameter_learning": "unavailable_all_rows",
        },
        "inference_status": {
            "hard_veto_screen": "passed_for_all_populated_cells",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence": "phase8_one_factor_resolution_feature_and_fit_refinement",
        },
        "nonclaims": [
            "no cross-method equivalence margin",
            "no statistically supported ranking",
            "no Zhao-Cui parameter-learning comparator",
            "no canonical, default, leaderboard, HMC, GPU, or full-horizon readiness",
        ],
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "environment": "artifact_only_standard_library",
            "cpu_gpu_status": "no numerical framework initialized",
            "data_version": "source artifact SHA-256 values embedded per cell",
            "random_seeds": "no new randomness",
        },
    }


def main() -> None:
    args = _parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 7 ledger: {output}")
    payload = build()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload["run_manifest"]["output"] = _relative_path(output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(payload["rows"]), "output": _relative_path(output)}))


if __name__ == "__main__":
    main()

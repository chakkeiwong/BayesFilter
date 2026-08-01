#!/usr/bin/env python3
"""Build the clean-XLA Phase 0 inventory.

This is reporting and source-audit code only.  It deliberately does not import
TensorFlow or participate in a gradient-bearing runtime path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/"
    "phase-00/registry/attempt-01-20260715/phase0_registry.json"
)
DEFAULT_ARTIFACT_ROOT = DEFAULT_OUTPUT.parent
PROGRAM_ID = "contract-e-tp-all-model-clean-xla-validation"
MASTER_PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-contract-e-tp-all-model-clean-xla-validation-master-program-2026-07-15.md"
)
PHASE_PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase0-registry-topology-default-audit-subplan-2026-07-15.md"
)
SOURCE_FILES = (
    "bayesfilter/highdim/ledh_contract_e_tp_lgssm_tf.py",
    "bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py",
    "bayesfilter/highdim/ledh_contract_e_tp_predator_prey_tf.py",
    "bayesfilter/highdim/ledh_contract_e_tp_structural_tf.py",
    "bayesfilter/highdim/ledh_contract_e_tp_models.py",
    "bayesfilter/highdim/models.py",
    "bayesfilter/structural_tf.py",
)

CONTROL_ARTIFACTS = {
    "prior_target_identity_registry": (
        "docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json",
        "43a6511c2523cf5a26bf6effb853068255f19d0d6f1395dbaff8d1f3aa1a1883",
    ),
    "lgssm_graph_audit": (
        "docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/"
        "graph_audit_final.json",
        "c3765e31ce0a1b8ee5cf4523ef141f3c1a86555b87c0669d34fbf3736549b7b3",
    ),
    "lgssm_gpu_t10": (
        "docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/"
        "gpu_t10_attempt2/result.json",
        "f450f5dd214f67253659bac6cf164f03a1352ffe8391fc536bd43bb32880caf0",
    ),
    "lgssm_gpu_t50": (
        "docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/"
        "gpu_t50_attempt1/result.json",
        "5b1810a2505960659bece7072b4aa7e2106dfcb2c8b4f6c3736743e1ce0a618b",
    ),
    "lgssm_invalid_chart": (
        "docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/"
        "invalid_chart_gpu_attempt2/result.json",
        "f745e5747ad6b45905b9a4dc7f089dfa347f26b88115ce828f460fc9901e3759",
    ),
}

ENTRY_EVIDENCE = {
    "actual_sv_prefix": "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_actual_sv_t100_order25_lookahead16_result_20260715.json",
    "ksc_sv_prefix": "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_ksc_sv_t100_order41_lookahead8_result_20260715.json",
    "generalized_sv_negative": "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_generalized_sv_t10_order41_progressive1_4_9_basis_quantile8_analytic_fill_localized_result_20260715.json",
    "predator_prey_prefix": "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_predator_prey_t5_order5_gaussian_closure_lookahead4_stabilized_result_20260715.json",
    "same_target_comparison": "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_same_target_comparison_20260715/comparison_ledger_v2.json",
}

ROW_CONFIG = (
    {
        "item_kind": "model_row",
        "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
        "state_dim": 3,
        "parameter_dim": 5,
        "horizon": 50,
        "classification": "reference_pass",
        "next_phase": "phase1_shared_clean_loop_guardrails",
        "target_scalar": "finite_fixed_program_observed_data_log_likelihood",
        "derivative_owner": "total_autodiff_same_finite_scalar_plus_kalman_reference",
        "factory_paths": [
            "bayesfilter.highdim.ledh_contract_e_tp_lgssm_tf.make_contract_e_tp_lgssm_score_informed_recursive_tf"
        ],
        "reference": "exact_differentiated_kalman_filter",
        "topology": "clean functional fixed-lag and filtering while loops; completed reference",
        "default_audit": {"lookahead": "8 (LGSSM reference only)", "order": "5", "dtype": "float64"},
    },
    {
        "item_kind": "model_row",
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "state_dim": 1,
        "parameter_dim": 2,
        "horizon": 1000,
        "classification": "eligible_inventory_required",
        "next_phase": "phase3_scalar_sv_loop_native_core",
        "target_scalar": "finite_fixed_program_observed_data_log_likelihood",
        "derivative_owner": "total_autodiff_of_scalar_sv_recursive_core",
        "factory_paths": ["bayesfilter.highdim.ledh_contract_e_tp_scalar_sv_tf.contract_e_tp_scalar_sv_recursive_core"],
        "reference": "refined_fixed_sgqf_and_high_accuracy_teacher_not_exact",
        "topology": "Python time and backward-continuation loops reachable from current scalar core",
        "default_audit": {"lookahead": "unset; target-specific phase", "order": "unset; target-specific phase", "dtype": "float64 baseline only"},
    },
    {
        "item_kind": "model_row",
        "row_id": "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "state_dim": 1,
        "parameter_dim": 2,
        "horizon": 1000,
        "classification": "eligible_inventory_required",
        "next_phase": "phase3_scalar_sv_loop_native_core",
        "target_scalar": "finite_fixed_program_observed_data_log_likelihood",
        "derivative_owner": "total_autodiff_of_scalar_sv_recursive_core",
        "factory_paths": ["bayesfilter.highdim.ledh_contract_e_tp_scalar_sv_tf.contract_e_tp_scalar_sv_recursive_core"],
        "reference": "refined_mixture_sgqf_and_high_accuracy_teacher_not_exact",
        "topology": "Python time and backward-continuation loops reachable from current scalar core",
        "default_audit": {"lookahead": "unset; target-specific phase", "order": "unset; target-specific phase", "dtype": "float64 baseline only"},
    },
    {
        "item_kind": "model_row",
        "row_id": "zhao_cui_generalized_sv_synthetic_from_estimated_values",
        "state_dim": 1,
        "parameter_dim": 3,
        "horizon": 1008,
        "classification": "negative_result",
        "next_phase": "phase10_terminal_synthesis",
        "target_scalar": "finite_fixed_program_observed_data_log_likelihood",
        "derivative_owner": "total_autodiff_of_scalar_sv_recursive_core",
        "factory_paths": ["bayesfilter.highdim.ledh_contract_e_tp_scalar_sv_tf.contract_e_tp_scalar_sv_recursive_core"],
        "reference": "refined_dense_target_filter; tested progressive family only",
        "topology": "current route has Python time/backward loops; rejected feature family is not scheduled for GPU",
        "default_audit": {"feature_family": "tested progressive family is negative; new family requires reviewed plan", "dtype": "float64 diagnostic only"},
    },
    {
        "item_kind": "model_row",
        "row_id": "zhao_cui_predator_prey_T20",
        "state_dim": 2,
        "parameter_dim": 6,
        "horizon": 20,
        "classification": "eligible_inventory_required",
        "next_phase": "phase5_predator_prey_loop_native_core",
        "target_scalar": "finite_fixed_program_observed_data_log_likelihood",
        "derivative_owner": "total_autodiff_of_predator_prey_scalar",
        "factory_paths": ["bayesfilter.highdim.ledh_contract_e_tp_predator_prey_tf.contract_e_tp_predator_prey_recursive_core"],
        "reference": "corrected_time_order_short_prefix_SGQF; T2 semi-analytic",
        "topology": "Python time/lookahead loops reachable; RK4 transition also uses a Python fixed-substep loop",
        "default_audit": {"support": "real plane additive Gaussian; no positivity clipping", "lookahead": "unset; target-specific phase", "dtype": "float64 baseline only"},
    },
    {
        "item_kind": "model_row",
        "row_id": "zhao_cui_spatial_sir_austria_j9_T20",
        "state_dim": 18,
        "parameter_dim": 3,
        "horizon": 20,
        "classification": "target_blocked",
        "next_phase": "phase7_sir_dsge_boundary_reaudit",
        "target_scalar": "finite_fixed_program_observed_data_log_likelihood",
        "derivative_owner": "missing_observed_data_total_score_owner",
        "factory_paths": ["bayesfilter.highdim.ledh_contract_e_tp_models.make_sir_contract_e_tp_adapter"],
        "reference": "none; P90/P91 component evidence is not observed-data total score",
        "topology": "adapter boundary only; no legal clean-XLA full filtering factory",
        "default_audit": {"target_law": "blocked clipped-push versus Gaussian-density measure mismatch", "gpu": "forbidden until target contract repaired"},
    },
    {
        "item_kind": "shared_regression_item",
        "row_id": "structural_deterministic_fixture",
        "state_dim": 2,
        "parameter_dim": 4,
        "horizon": None,
        "classification": "required_shared_regression",
        "next_phase": "phase2_structural_support_regression",
        "target_scalar": "structural_teacher_support_and_completion_residual_fixture",
        "derivative_owner": "explicit_structural_tangent_fixture",
        "factory_paths": ["bayesfilter.highdim.ledh_contract_e_tp_structural_tf.structural_parent_innovation_teacher_tf"],
        "reference": "tests/highdim/test_ledh_contract_e_tp_structural.py",
        "topology": "primitive support/tangent helpers; not a complete stochastic model row",
        "default_audit": {"integration_space": "innovation", "deterministic_completion": "required"},
    },
    {
        "item_kind": "model_row",
        "row_id": "dsge_nawm_client",
        "state_dim": None,
        "parameter_dim": None,
        "horizon": None,
        "classification": "target_blocked",
        "next_phase": "phase7_sir_dsge_boundary_reaudit",
        "target_scalar": "missing_executable_client_scalar",
        "derivative_owner": "missing",
        "factory_paths": [],
        "reference": "none",
        "topology": "no executable client metadata, observations, chart, scalar, or comparator registered",
        "default_audit": {"status": "must not infer from structural fixture"},
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_scan(path: Path) -> dict[str, Any]:
    source_text = path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))
    loops: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            loops.append({
                "kind": type(node).__name__,
                "line": int(node.lineno),
                "source": ast.get_source_segment(source_text, node)[:120],
            })
    while_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"while_loop", "scan"}:
                while_calls.append({"name": node.func.attr, "line": int(node.lineno)})
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "python_loop_count": len(loops),
        "python_loops": loops,
        "functional_loop_calls": while_calls,
        "runtime_numpy_or_scipy_import": any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name.split(".")[0] in {"numpy", "scipy"} for alias in node.names)
            for node in ast.walk(tree)
        ),
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _artifact_record(relative: str, expected: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"controlling artifact hash changed: {relative}: {actual}")
    return {"path": relative, "sha256": actual, "hash_status": "verified"}


def _entry_evidence_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {"path": relative, "sha256": _sha256(path), "hash_status": "recorded_at_phase0"}


def build_payload() -> dict[str, Any]:
    if len(ROW_CONFIG) != 8 or len({row["row_id"] for row in ROW_CONFIG}) != 8:
        raise ValueError("Phase 0 must contain exactly eight unique inventory items")
    sources = [_source_scan(ROOT / relative) for relative in SOURCE_FILES]
    rows = []
    for row in ROW_CONFIG:
        item = dict(row)
        item["target_identity_status"] = (
            "bound" if row["classification"] != "target_blocked" else "blocked"
        )
        item["gpu_scheduling"] = (
            "forbidden" if row["classification"] in {"negative_result", "target_blocked"}
            else "phase-specific-gate-required"
        )
        item["target_identity_anchor"] = (
            "docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json#"
            + row["row_id"]
            if row["row_id"] not in {"structural_deterministic_fixture", "dsge_nawm_client"}
            else "repository_owned_fixture" if row["row_id"] == "structural_deterministic_fixture" else "missing"
        )
        rows.append(item)
    source_by_path = {record["path"]: record for record in sources}
    return {
        "schema": "contract_e_tp.clean_xla_phase0_inventory.v1",
        "program_id": PROGRAM_ID,
        "status": "phase0_registry_topology_default_audit_complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_at_git_commit": _git_commit(),
        "device_policy": "cpu_only_registry_build_CUDA_VISIBLE_DEVICES=-1",
        "item_kinds": {"model_row": 7, "shared_regression_item": 1},
        "rows": rows,
        "source_dependencies": sources,
        "controlling_artifacts": {
            name: _artifact_record(relative, expected)
            for name, (relative, expected) in CONTROL_ARTIFACTS.items()
        },
        "entry_evidence": {
            name: _entry_evidence_record(relative)
            for name, relative in ENTRY_EVIDENCE.items()
        },
        "topology_inventory": {
            "compiled_dynamic_python_loops": [
                {
                    "path": "bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py",
                    "symbols": [
                        "target_continuation_log_likelihood",
                        "contract_e_tp_scalar_sv_recursive_kkt_core",
                        "contract_e_tp_scalar_sv_recursive_core",
                    ],
                    "roles": ["backward_continuation", "filter_time"],
                    "status": "phase3_repair_required",
                },
                {
                    "path": "bayesfilter/highdim/ledh_contract_e_tp_predator_prey_tf.py",
                    "symbols": [
                        "target_continuation_log_likelihood",
                        "contract_e_tp_predator_prey_recursive_core",
                    ],
                    "roles": ["backward_continuation", "filter_time"],
                    "status": "phase5_repair_required",
                },
                {
                    "path": "bayesfilter/highdim/models.py",
                    "symbols": [
                        "PredatorPreySSM.transition_mean",
                        "PredatorPreySSM.transition_mean_parameter_jacobian",
                    ],
                    "roles": ["fixed_substep_rk4_solver"],
                    "status": "phase5_repair_required_if_reachable_compiled",
                },
            ],
            "fixed_small_python_loops": [
                {
                    "role": "static parameter/dimension construction",
                    "policy": "permitted only when proved not horizon/window/solver scaling",
                }
            ],
            "functional_tensorflow_loops": [
                {
                    "path": "bayesfilter/highdim/ledh_contract_e_tp_lgssm_tf.py",
                    "count": len(source_by_path["bayesfilter/highdim/ledh_contract_e_tp_lgssm_tf.py"]["functional_loop_calls"]),
                    "status": "completed_reference",
                }
            ],
            "offline_numpy_scipy": [
                {
                    "role": "chart preparation, reference, serialization, and reporting only",
                    "status": "permitted_but_not_runtime_evidence",
                }
            ],
            "gradient_runtime_numpy_scipy_findings": [
                record["path"] for record in sources if record["runtime_numpy_or_scipy_import"]
            ],
        },
        "policies": {
            "contract_e_tp_status": "experimental_only",
            "contract_e_chol_status": "canonical",
            "fd_tolerance_role": "0.05*sqrt(p) individual-direction same-scalar FD only",
            "cross_method_margin": "unavailable; descriptive-only",
            "runtime_numpy_scipy": "forbidden in gradient-bearing paths",
            "dynamic_compiled_loops": "TensorFlow functional control flow required",
        },
        "budget": {
            "authorized_cpu_core_hours": 96,
            "authorized_trusted_gpu_hours": 32,
            "full_horizon_attempts_per_eligible_model": 3,
            "phase0_cpu_cap": 4,
            "phase0_gpu_cap": 0,
            "phase0_full_horizon_attempts": 0,
            "historical_experiments_charged": 0,
        },
        "nonclaims": [
            "not implementation correctness beyond cited controls",
            "not nonlinear clean-XLA readiness",
            "not scientific filtering accuracy or cross-method equivalence",
            "not canonical, default, HMC, or leaderboard readiness",
        ],
        "master_plan": str(MASTER_PLAN.relative_to(ROOT)),
        "phase_plan": str(PHASE_PLAN.relative_to(ROOT)),
        "evidence_roles": {
            "primary": "row identity, topology/default audit, and legal phase classification",
            "explanatory": "Python-loop counts and historical prefix artifacts",
            "veto": "missing source/artifact, silent default transfer, proxy row, or illegal GPU scheduling",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 0 evidence: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(payload["rows"]), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inventory and preserve Phase 6 historical raw LEDH route behavior."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DISCOVERY_PATTERNS = (
    "RAW_COMPACT_ADMITTED_STATUS",
    "LEDH_SCORE_ADMISSION_STATUS_FULL",
    "compact-sensitivity",
    "manual-reverse",
    "COMPACT_SCORE_ROUTE_ID",
    "MANUAL_SCORE_ROUTE_ID",
    "historical_raw_barycentric_diagnostic_only",
    "validate_ledh_score_artifact",
    "build_ledh_score_artifact",
    "score_admission_status",
    "LEDH_FORWARD_ADMISSION_STATUS_ADMITTED",
    "LEDH_FORWARD_ADMISSION_STATUS_TINY",
    "forward_admission_status",
    "admission_status",
)

PATH_CLASSIFICATION = {
    "bayesfilter/highdim/ledh_forward_contract.py": "central_guard",
    "bayesfilter/highdim/ledh_score_artifact.py": "central_guard",
    "bayesfilter/highdim/ledh_score_contract.py": "central_guard",
    "bayesfilter/highdim/ledh_score_contract_v2.py": "central_guard",
    "bayesfilter/highdim/ledh_forward_contract_v2.py": "central_guard",
    "bayesfilter/highdim/ledh_historical_raw_policy.py": "central_guard",
    "bayesfilter/highdim/sv_mixture_cut4.py": "unrelated_symbol_collision",
    "bayesfilter/nonlinear/fixed_sgqf_structural_adapter_tf.py": "unrelated_symbol_collision",
    "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_compact_score_adapter.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py": "edit",
    "docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py": "edit",
    "docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py": "central_guard",
    "docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py": "edit",
    "docs/benchmarks/benchmark_p8p_regression_fd_reparameterization.py": "historical_diagnostic_exclusion",
    "docs/benchmarks/diagnose_ledh_pfpf_ot_contract_e_lgssm_gpu_score.py": "historical_diagnostic_exclusion",
    "docs/benchmarks/diagnose_ledh_pfpf_ot_contract_e_lgssm_gradient.py": "historical_diagnostic_exclusion",
    "docs/benchmarks/diagnose_lgssm_compact_gradient_precision.py": "historical_diagnostic_exclusion",
    "docs/benchmarks/diagnose_lgssm_reset_gradient_variants.py": "historical_diagnostic_exclusion",
    "docs/benchmarks/diagnose_p8p_sir_sinkhorn_budget.py": "historical_diagnostic_exclusion",
    "docs/benchmarks/audit_contract_e_phase6_historical_routes.py": "central_guard",
    "docs/benchmarks/emit_contract_e_phase6_raw_diagnostic_baseline.py": "historical_diagnostic_exclusion",
    "docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py": "central_guard",
    "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py": "unrelated_symbol_collision",
    "docs/benchmarks/benchmark_two_lane_highdim_ledh_leaderboard.py": "central_guard",
    "docs/benchmarks/build_complete_highdim_phase1_canonical_targets.py": "historical_data_consumer",
    "scripts/audit_ledh_no_autodiff.py": "central_guard",
    "scripts/filtering_value_gradient_benchmark_emit_synthetic_truth.py": "unrelated_symbol_collision",
    "tests/test_audit_ledh_no_autodiff.py": "test",
    "tests/test_contract_e_phase3_gradient_route_audit.py": "test",
    "tests/test_contract_e_phase3_r1_design_artifacts.py": "test",
    "tests/test_ledh_fixed_sir_manual_score_phase4.py": "test",
    "tests/test_ledh_lgssm_manual_score_phase4.py": "test",
    "tests/test_ledh_score_memory_n10000.py": "test",
    "tests/highdim/test_ledh_actual_sv_score_phase5_contract.py": "test",
    "tests/highdim/test_ledh_compact_score_gpu_xla_harness.py": "test",
    "tests/highdim/test_ledh_contract_e_phase0_emergency_revocation.py": "test",
    "tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py": "test",
    "tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py": "test",
    "tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py": "test",
    "tests/highdim/test_ledh_lgssm_score_phase2_contract.py": "test",
    "tests/highdim/test_ledh_predator_prey_score_phase4_contract.py": "test",
    "tests/highdim/test_ledh_score_artifact_emitter_phase1.py": "test",
    "tests/highdim/test_ledh_score_contract_phase1.py": "test",
    "tests/highdim/test_ledh_score_wiring_phase8_cross_model.py": "test",
    "tests/highdim/test_ledh_contract_e_schema_v2_factory.py": "test",
    "tests/highdim/test_p43_sv_value_gradient_cut4_zhaocui.py": "historical_diagnostic_exclusion",
    "tests/highdim/test_filtering_value_gradient_benchmark_synthetic_truth_p8.py": "unrelated_symbol_collision",
    "tests/test_nonlinear_benchmark_models_tf.py": "unrelated_symbol_collision",
    "tests/test_two_lane_highdim_leaderboard_phase5.py": "unrelated_symbol_collision",
    "tests/test_two_lane_highdim_leaderboard_phase7.py": "unrelated_symbol_collision",
    "tests/test_two_lane_highdim_ledh_leaderboard.py": "test",
    "tests/highdim/test_ledh_forward_scalar_admission_guard.py": "test",
    "tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py": "test",
    "tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py": "test",
    "tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py": "test",
    "tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py": "test",
    "tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py": "test",
    "tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py": "test",
    "tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py": "test",
    "tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py": "test",
    "tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py": "test",
    "tests/highdim/test_ledh_phase8_value_integration_artifact.py": "test",
}

EXCLUSION_JUSTIFICATION = {
    "unrelated_symbol_collision": "matched a generic status token but does not execute or consume an LEDH raw score route",
    "historical_diagnostic_exclusion": "analysis-only historical diagnostic; it emits no score artifact and reaches raw arithmetic only through an edit-root gate",
    "historical_data_consumer": "reads historical metadata but is not an admission or route dispatcher; central validators remain authoritative",
}

KERNEL_SYMBOLS = {
    "docs.benchmarks.benchmark_ledh_same_target_lgssm_m3_t50_value": (
        "_manual_value_and_score_from_components",
        "_same_target_value_from_components",
        "_compact_value_and_score_from_components",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_fixed_sir_score": (
        "_fixed_sir_manual_score_diagnostic",
        "_compact_value_and_score_from_components",
        "_value_objective_from_components",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_predator_prey_score": (
        "_manual_value_and_score_from_components",
        "_manual_value_only_from_components",
        "_compact_value_and_score_from_components",
        "_value_objective_across_seeds",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_actual_sv_score": (
        "_manual_value_and_score_from_components",
        "_manual_value_only_from_components",
        "_compact_value_and_score_from_components",
        "_value_objective_across_seeds",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_generalized_sv_score": (
        "_manual_value_only_from_components",
        "_compact_value_and_score_from_components",
        "_value_objective_across_seeds",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_ksc_sv_score": (
        "_manual_value_only_from_components",
        "_compact_value_and_score_from_components",
        "_value_objective_across_seeds",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_predator_prey_value": (
        "_build_predator_prey_tensors",
        "_make_predator_prey_callbacks",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_actual_sv_value": (
        "_build_actual_sv_tensors",
        "_actual_sv_value_core",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_generalized_sv_value": (
        "_build_generalized_sv_tensors",
        "_generalized_sv_value_core",
    ),
    "docs.benchmarks.benchmark_ledh_same_target_ksc_sv_value": (
        "_build_ksc_sv_tensors",
        "_ksc_sv_value_core",
    ),
}

EXECUTABLE_ROOTS = (
    {"path": "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py", "symbols": ["_parse_args", "_manual_score_diagnostic", "_score_admission_decision", "_lgssm_score_artifact_from_result", "_aggregate_lgssm_score_shard_payload", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_compact_score_adapter.py", "symbols": ["_require_compact_args", "_compact_value_and_score_from_components"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py", "symbols": ["_require_fixed_sir_score_args", "_fixed_sir_compact_score_artifact_from_diagnostic", "_fixed_sir_score_artifact_from_memory_result"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py", "symbols": ["_require_manual_score_args", "_parse_args", "_score_artifact_from_diagnostic", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py", "symbols": ["_require_manual_score_args", "_parse_args", "_score_artifact_from_diagnostic", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py", "symbols": ["_require_compact_score_args", "_parse_args", "_score_artifact_from_diagnostic", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py", "symbols": ["_require_compact_score_args", "_parse_args", "_score_artifact_from_diagnostic", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py", "symbols": ["_parse_args", "_validate_args", "_aggregate", "main"]},
    {"path": "docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py", "symbols": ["_parse_args", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py", "symbols": ["_parse_args", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py", "symbols": ["_parse_args", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py", "symbols": ["_parse_args", "main"]},
    {"path": "docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py", "symbols": ["_parse_args", "main"]},
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _candidate_paths() -> list[Path]:
    roots = (ROOT / "bayesfilter", ROOT / "docs/benchmarks", ROOT / "scripts", ROOT / "tests")
    return sorted(
        path for root in roots for path in root.rglob("*.py")
        if ".complete-highdim" not in str(path)
    )


def _discovery() -> tuple[list[dict[str, Any]], list[str]]:
    hits = []
    unclassified = []
    for path in _candidate_paths():
        text = path.read_text(encoding="utf-8")
        matched = sorted(pattern for pattern in DISCOVERY_PATTERNS if pattern in text)
        if not matched:
            continue
        relative = str(path.relative_to(ROOT))
        classification = PATH_CLASSIFICATION.get(relative)
        if classification is None:
            unclassified.append(relative)
        hits.append({
            "path": relative,
            "classification": classification,
            "matched_patterns": matched,
            "exclusion_justification": EXCLUSION_JUSTIFICATION.get(classification),
            "sha256": _sha256_bytes(path.read_bytes()),
        })
    return hits, unclassified


def _kernel_hashes() -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for module_name, symbols in KERNEL_SYMBOLS.items():
        module = __import__(module_name, fromlist=["*"])
        module_hashes = {}
        for symbol in symbols:
            source = inspect.getsource(getattr(module, symbol))
            # AST normalization ignores comments and location-only drift while
            # detecting every executable operation change.
            normalized = ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)
            module_hashes[symbol] = _sha256_bytes(normalized.encode())
        output[module_name] = module_hashes
    return output


def _symbol_presence() -> list[dict[str, Any]]:
    output = []
    for root in EXECUTABLE_ROOTS:
        path = ROOT / root["path"]
        names = {node.name for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing = sorted(set(root["symbols"]) - names)
        output.append({**root, "missing_symbols": missing})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-kernel-hashes", type=Path)
    args = parser.parse_args()
    hits, unclassified = _discovery()
    kernel_hashes = _kernel_hashes()
    roots = _symbol_presence()
    expected = None
    kernel_hashes_match = None
    if args.expected_kernel_hashes is not None:
        expected_payload = json.loads(args.expected_kernel_hashes.read_text(encoding="utf-8"))
        expected = expected_payload["numerical_kernel_ast_sha256"]
        kernel_hashes_match = kernel_hashes == expected
    payload = {
        "schema_version": "bayesfilter.contract_e_phase6_historical_route_inventory.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "git_commit": _git_commit(),
        "discovery_queries": list(DISCOVERY_PATTERNS),
        "discovery_hits": hits,
        "unclassified_hits": unclassified,
        "zero_unclassified_hits": not unclassified,
        "executable_roots": roots,
        "all_executable_root_symbols_present": all(not root["missing_symbols"] for root in roots),
        "numerical_kernel_ast_sha256": kernel_hashes,
        "expected_numerical_kernel_ast_sha256": expected,
        "numerical_kernel_hashes_match_baseline": kernel_hashes_match,
        "obligations": [
            "no_opt_in_rejects_before_numerical_work",
            "opt_in_emits_exact_historical_status",
            "inclusive_aggregate_rejects_historical_artifact",
            "contract_e_failure_cannot_invoke_raw_sentinel",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if unclassified or not payload["all_executable_root_symbols_present"] or kernel_hashes_match is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

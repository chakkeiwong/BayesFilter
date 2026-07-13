"""Strict Phase A1 evidence harness for the locked masked SSL-LSTM target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (
    A0_DEPENDENCY_MANIFEST_FILE_SHA256,
    A0_IMMUTABLE_AGGREGATE_SHA256,
    A0_SIGNATURE_AGGREGATE_SHA256,
    A0_TARGET_LOCK_FILE_SHA256,
    FULL_FIXTURE_RAW_SHA256,
    GOLDEN_SIGNATURES_FILE_SHA256,
    MASKED_POSTERIOR_CONTRACT_SHA256,
    NONCLAIMS,
    OBSERVATION_RAW_SHA256,
    PARAMETER_MASK_SHA256,
    SSLLSTMParameterMask,
    SSLLSTMPosteriorConfig,
    SSLLSTMPosteriorTarget,
    TARGET_SEMANTIC_SHA256,
    locked_ssl_lstm_posterior_target,
)


HARNESS_PATH = "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py"
MODULE_PATH = "bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"
EXPORT_PATH = "bayesfilter/nonlinear/__init__.py"
TEST_PATH = "tests/test_ssl_lstm_posterior_tf.py"
GOLDEN_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json"
HISTORICAL_PATH = "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py"
ENTRY_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/a0-entry-verification.json"
BOUNDARY_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-scoped-boundary.json"
CPU_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json"
CPU_LOG_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.log"
GPU_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json"
GPU_LOG_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.log"
PLAN_PATH = "docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md"
RESULT_PATH = "docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md"

HISTORICAL_SHA256 = "fea73716e1d972a5336e3bdedb733dfc31c4a0bb61cf40cdf877d577d68cbe28"
A0_ANCHOR = "a644d29c5c2fd09a0deb3a7b5212799ff1fcb163"
ENTRY_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a1_entry_verification.v2"
BOUNDARY_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a1_scoped_boundary.v2"
CPU_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a1_cpu_reference.v1"
GPU_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a1_gpu_xla_canary.v1"

TRUTH_FREE = np.array([0.35, -0.08, 0.65, 0.05], dtype=np.float64)
PHASE2S_CENTER = np.array(
    [0.5704394246369003, -0.1242247342531544, 0.6609123192759063, 0.1354211218811133],
    dtype=np.float64,
)
SHELL_STEP = 0.25 * 0.35
FD_STEP = 1.0e-5
FD_RTOL = 5.0e-3
FD_ATOL = 8.0e-4
HISTORICAL_SCALE = 8.0 * (2.0**-52)
VALUE_PARITY_SCALE = 1.0e-10
SCORE_PARITY_ATOL = 1.0e-8

TOP_LEVEL_KEYS = {
    "schema_version", "artifact_role", "status", "created_at_utc", "run_manifest",
    "a0_bindings", "a1_signatures", "boundary_bindings", "source_files",
    "test_point_design", "point_results", "reject_results", "contract_checks",
    "evidence_signature", "nonclaims",
}
RUN_MANIFEST_KEYS = {
    "git_commit", "git_dirty", "command", "cwd", "interpreter", "python_version",
    "packages", "conda_env", "environment", "physical_devices", "logical_devices",
    "cpu_gpu_status", "trust_basis", "dtype", "jit_compile", "xla", "tf32_enabled",
    "data_version", "random_seeds", "started_at_utc", "completed_at_utc",
    "wall_time_seconds", "output_path", "log_path", "plan_path", "result_path",
}
A0_KEYS = {
    "target_semantic_sha256", "signature_aggregate_sha256", "immutable_aggregate_sha256",
    "target_lock_file_sha256", "dependency_manifest_file_sha256", "observation_raw_sha256",
    "full_fixture_raw_sha256",
}
A1_KEYS = {
    "target_semantic_sha256", "parameter_mask_sha256",
    "masked_posterior_contract_sha256", "golden_signatures_file_sha256",
}
BOUNDARY_KEYS = {
    "a0_entry_verification_file_sha256", "pre_run_scoped_boundary_file_sha256",
    "protected_dependency_aggregate_sha256", "excluded_dependency_aggregate_sha256",
    "scoped_boundary_aggregate_sha256",
}
CPU_CHECK_KEYS = {
    "a0_entry_verified", "scoped_boundary_verified", "mask_schema_valid",
    "mask_golden_digest_match", "wrapper_schema_valid", "wrapper_golden_digest_match",
    "golden_file_exact", "historical_source_hash_exact",
    "historical_all_ten_points_passed", "target_semantic_digest_match",
    "embed_extract_exact", "prior_convention_exact", "scalar_shapes_exact",
    "batch_shapes_exact", "batch_sizes_1_4_10_exact", "callable_aliases_exact",
    "compiled_default_invoked", "valid_branch_bitwise_equal",
    "nonfinite_input_reject_exact", "reject_gradient_zero", "finite_filter_failure_loud",
    "testing_only_provenance_unavailable", "testing_only_artifact_refused",
    "finite_difference_passed", "eager_cpu_xla_passed", "no_benchmark_import",
    "no_numpy_algorithmic_path", "no_tf_py_function", "historical_filter_route_only",
    "authority_not_self_certified", "point_order_exact", "all_passed",
}
GPU_CHECK_KEYS = {
    "cpu_reference_file_sha256", "cpu_reference_verified", "gpu_visible",
    "trusted_provenance_recorded", "jit_compile_true", "xla_executed",
    "gpu_device_placement_verified", "tf32_recorded", "cpu_gpu_parity_passed",
    "signatures_equal", "point_order_exact", "all_passed",
}
SOURCE_ROW_KEYS = {"path", "sha256", "git_status"}
FINITE_DESIGN_KEYS = {
    "name", "role", "input_hex", "finite_difference_step_hex", "fd_rtol_hex",
    "fd_atol_hex", "historical_rtol_hex", "historical_atol_hex",
    "value_parity_atol_hex", "score_parity_atol_hex",
}
NONFINITE_DESIGN_KEYS = {"name", "role", "input_strings"}
CPU_POINT_KEYS = {
    "name", "input_hex", "historical_value_hex", "historical_score_hex",
    "eager_value_hex", "eager_score_hex", "cpu_xla_value_hex", "cpu_xla_score_hex",
    "status", "finite_difference_score_hex", "historical_value_abs_residual",
    "historical_score_abs_residual_inf", "value_abs_residual",
    "score_abs_residual_inf", "fd_abs_residual_inf", "fd_relative_residual_inf",
    "historical_value_tolerance", "historical_score_tolerance", "value_tolerance",
    "score_tolerance", "fd_atol", "fd_rtol", "passed",
}
GPU_POINT_KEYS = {
    "name", "input_hex", "cpu_xla_value_hex", "cpu_xla_score_hex",
    "gpu_xla_value_hex", "gpu_xla_score_hex", "status", "value_abs_residual",
    "score_abs_residual_inf", "value_tolerance", "score_tolerance", "passed",
}
REJECT_ROW_KEYS = {
    "name", "input_strings", "value_hex", "score_hex", "status", "gradient_hex",
    "finite_branch_runtime_assertion_not_triggered", "passed",
}
ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "TF_DETERMINISTIC_OPS",
    "TF_ENABLE_ONEDNN_OPTS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
    "OMP_NUM_THREADS", "TF_CPP_MIN_LOG_LEVEL",
}
FIXED_ENTRY_PATHS = {
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md",
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-result-final-codex-substitute-review-2026-07-11.md",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json",
    "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py",
    GOLDEN_PATH,
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json",
    HISTORICAL_PATH,
}

CPU_COMMAND = (
    "CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 "
    "TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
    "OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 "
    "PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache "
    "/home/ubuntu/anaconda3/envs/tfgpu/bin/python " + HARNESS_PATH + " --mode cpu-reference "
    "--output " + CPU_PATH + " --log-path " + CPU_LOG_PATH
)
GPU_COMMAND = (
    "env -u CUDA_VISIBLE_DEVICES -u TF_NUM_INTRAOP_THREADS -u TF_NUM_INTEROP_THREADS "
    "-u OMP_NUM_THREADS PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 "
    "TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=1 "
    "PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache "
    "/home/ubuntu/anaconda3/envs/tfgpu/bin/python " + HARNESS_PATH + " --mode gpu-xla-canary "
    "--cpu-reference " + CPU_PATH + " --output " + GPU_PATH + " --log-path " + GPU_LOG_PATH
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def stage(message: str) -> None:
    print(json.dumps({"stage": message}, sort_keys=True), file=sys.stderr, flush=True)


def repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    require(path == ROOT or ROOT in path.parents, f"path escapes repository: {relative}")
    return path


def file_sha256(path: Path | str) -> str:
    item = path if isinstance(path, Path) else repo_path(path)
    return hashlib.sha256(item.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_load(path: Path | str) -> dict[str, Any]:
    item = path if isinstance(path, Path) else repo_path(path)

    def reject(value: str) -> None:
        raise ContractError(f"nonfinite JSON constant in {item}: {value}")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ContractError(f"duplicate JSON key in {item}: {key}")
            result[key] = value
        return result

    value = json.loads(item.read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=pairs)
    require(isinstance(value, dict), f"top-level JSON must be an object: {item}")
    return value


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} keys mismatch: {sorted(set(value) ^ expected)}")


def verify_rfc3339(value: Any, label: str) -> None:
    require(isinstance(value, str) and value.endswith("+00:00"), f"{label} is not UTC RFC3339")
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, f"{label} timezone mismatch")


def require_finite_number(value: Any, label: str) -> None:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    require(math.isfinite(float(value)), f"{label} is nonfinite")


def git_text(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def committed_history_paths(anchor: str, head: str) -> list[str]:
    if anchor == head:
        return []
    commits = git_text(["rev-list", "--reverse", f"{anchor}..{head}"]).splitlines()
    paths: set[str] = set()
    for commit in commits:
        raw = subprocess.check_output(
            ["git", "diff-tree", "--root", "-m", "--no-commit-id", "--name-only", "-r", "-z", commit],
            cwd=ROOT,
        )
        paths.update(os.fsdecode(row) for row in raw.split(b"\0") if row)
    return sorted(paths)


def live_history(entry: Mapping[str, Any], boundary: Mapping[str, Any]) -> tuple[str, list[str]]:
    head = git_text(["rev-parse", "HEAD"])
    require(
        subprocess.run(["git", "merge-base", "--is-ancestor", A0_ANCHOR, head], cwd=ROOT).returncode == 0,
        "A0 anchor is not an ancestor of current HEAD",
    )
    paths = committed_history_paths(A0_ANCHOR, head)
    protected = {row["path"] for row in entry["protected_dependency_rows"]}
    owned = set(boundary["owned_exact"])
    forbidden = sorted(set(paths) & (protected | owned))
    require(not forbidden, f"committed protected/A1-owned path drift: {forbidden}")
    return head, paths


def projection_without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in set(keys)}


def verify_entry_boundary() -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    entry = strict_load(ENTRY_PATH)
    boundary = strict_load(BOUNDARY_PATH)
    exact_keys(entry, {
        "schema_version", "entry_documents", "target_lock_integrity",
        "protected_dependency_rows", "protected_dependency_aggregate_sha256",
        "excluded_dependency_rows", "excluded_dependency_aggregate_sha256",
        "manifest_partition", "commit_policy", "concurrent_lane_policy",
    }, "entry artifact")
    require(entry["schema_version"] == ENTRY_SCHEMA, "entry schema mismatch")
    documents = entry["entry_documents"]
    exact_keys(documents, {
        "schema_version", "a0_anchor_commit", "fixed_sha256", "reviewed_target_sha256",
        "review_sha256", "approval_boundary",
    }, "entry documents")
    require(documents["schema_version"] == "bayesfilter.ssl_lstm_completion.phase_a1_entry_documents.v2", "entry documents schema mismatch")
    require(documents["a0_anchor_commit"] == A0_ANCHOR, "entry documents anchor mismatch")
    require(set(documents["fixed_sha256"]) == FIXED_ENTRY_PATHS, "entry fixed path set mismatch")
    for path, expected in documents["fixed_sha256"].items():
        require(file_sha256(path) == expected, f"entry fixed hash drift: {path}")
    expected_reviewed = {
        "docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md",
        PLAN_PATH,
        GOLDEN_PATH,
    }
    require(set(documents["reviewed_target_sha256"]) == expected_reviewed, "entry reviewed target set mismatch")
    for path, expected in documents["reviewed_target_sha256"].items():
        require(file_sha256(path) == expected, f"entry reviewed target drift: {path}")
    expected_reviews = {
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-result-final-codex-substitute-review-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md",
    }
    require(set(documents["review_sha256"]) == expected_reviews, "entry review set mismatch")
    for path, expected in documents["review_sha256"].items():
        require(file_sha256(path) == expected, f"entry review drift: {path}")
    exact_keys(documents["approval_boundary"], {"path", "section_sha256"}, "approval boundary")
    integrity = entry["target_lock_integrity"]
    exact_keys(integrity, {
        "status", "artifact", "target_lock_file_sha256", "dependency_manifest_file_sha256",
        "immutable_aggregate", "signature_aggregate", "target_semantic_sha256",
    }, "target lock integrity")
    require(integrity["status"] == "target_lock_integrity_verified_under_scoped_concurrency", "target lock integrity status mismatch")
    require(integrity["target_lock_file_sha256"] == A0_TARGET_LOCK_FILE_SHA256, "target lock integrity hash mismatch")
    require(integrity["dependency_manifest_file_sha256"] == A0_DEPENDENCY_MANIFEST_FILE_SHA256, "manifest integrity hash mismatch")
    require(integrity["immutable_aggregate"] == A0_IMMUTABLE_AGGREGATE_SHA256, "immutable aggregate mismatch")
    require(integrity["signature_aggregate"] == A0_SIGNATURE_AGGREGATE_SHA256, "signature aggregate mismatch")
    require(integrity["target_semantic_sha256"] == TARGET_SEMANTIC_SHA256, "target semantic mismatch")
    require(entry["manifest_partition"] == {
        "unique_manifest_path_count": 51, "protected_path_count": 23,
        "excluded_path_count": 28, "exhaustive": True, "disjoint": True,
    }, "entry manifest partition mismatch")
    require(entry["commit_policy"]["a0_anchor_commit"] == A0_ANCHOR, "entry anchor mismatch")
    exact_keys(entry["commit_policy"], {
        "a0_anchor_commit", "entry_checked_commit", "committed_paths_since_anchor",
        "forbidden_committed_paths",
    }, "entry commit policy")
    entry_creation = entry["commit_policy"]["entry_checked_commit"]
    require(
        entry["commit_policy"]["committed_paths_since_anchor"]
        == committed_history_paths(A0_ANCHOR, entry_creation),
        "entry creation history mismatch",
    )
    require(entry["commit_policy"]["forbidden_committed_paths"] == [], "entry forbidden paths nonempty")
    require(
        canonical_sha256(entry["protected_dependency_rows"])
        == entry["protected_dependency_aggregate_sha256"],
        "entry protected aggregate mismatch",
    )
    require(
        canonical_sha256(entry["excluded_dependency_rows"])
        == entry["excluded_dependency_aggregate_sha256"],
        "entry excluded aggregate mismatch",
    )
    for row in entry["protected_dependency_rows"]:
        require(file_sha256(row["path"]) == row["sha256"], f"protected dependency drift: {row['path']}")

    exact_keys(boundary, {
        "schema_version", "a0_anchor_commit", "boundary_creation_commit",
        "committed_paths_through_boundary_creation", "owned_exact", "initial_owned_state",
        "protected_dependency_rows", "protected_dependency_aggregate_sha256",
        "excluded_dependency_rows", "excluded_dependency_aggregate_sha256", "manifest_partition",
        "immutable_inputs", "approval_section_sha256", "concurrent_lane_policy",
        "unrelated_snapshot", "aggregate_sha256",
    }, "scoped boundary")
    require(boundary["schema_version"] == BOUNDARY_SCHEMA, "boundary schema mismatch")
    require(boundary["a0_anchor_commit"] == A0_ANCHOR, "boundary anchor mismatch")
    require(boundary["protected_dependency_rows"] == entry["protected_dependency_rows"], "boundary protected rows mismatch")
    require(boundary["excluded_dependency_rows"] == entry["excluded_dependency_rows"], "boundary excluded rows mismatch")
    require(boundary["manifest_partition"] == entry["manifest_partition"], "boundary partition mismatch")
    require(boundary["protected_dependency_aggregate_sha256"] == entry["protected_dependency_aggregate_sha256"], "boundary protected aggregate mismatch")
    require(boundary["excluded_dependency_aggregate_sha256"] == entry["excluded_dependency_aggregate_sha256"], "boundary excluded aggregate mismatch")
    expected_owned = {
        EXPORT_PATH, MODULE_PATH, HARNESS_PATH, ENTRY_PATH, CPU_PATH, CPU_LOG_PATH, GOLDEN_PATH,
        GPU_PATH, GPU_LOG_PATH,
        "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json",
        BOUNDARY_PATH,
        "docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md",
        RESULT_PATH, PLAN_PATH,
        "docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md",
        "docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md",
        "docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md",
        "docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-codex-substitute-review-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-implementation-codex-substitute-review-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md",
        "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-2026-07-11.md",
        TEST_PATH,
    }
    require(boundary["owned_exact"] == sorted(expected_owned), "boundary owned set mismatch")
    require(
        boundary["committed_paths_through_boundary_creation"]
        == committed_history_paths(A0_ANCHOR, boundary["boundary_creation_commit"]),
        "boundary creation history mismatch",
    )
    for row in boundary["immutable_inputs"]:
        require(file_sha256(row["path"]) == row["sha256"], f"boundary immutable drift: {row['path']}")
    require(boundary["immutable_inputs"] == [
        {"path": ENTRY_PATH, "sha256": file_sha256(ENTRY_PATH)},
        {"path": GOLDEN_PATH, "sha256": GOLDEN_SIGNATURES_FILE_SHA256},
        {"path": "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json", "sha256": "2ec3b605266fe652fc452d58d483914cea71f9ce845345c685d4669cfcf848be"},
    ], "boundary immutable input list mismatch")
    projection = projection_without(boundary, "schema_version", "aggregate_sha256")
    require(canonical_sha256(projection) == boundary["aggregate_sha256"], "boundary aggregate mismatch")
    head, paths = live_history(entry, boundary)
    return entry, boundary, head, paths


def git_status(relative: str) -> str:
    raw = subprocess.check_output(["git", "status", "--porcelain=v1", "--", relative], cwd=ROOT, text=True)
    if raw:
        return raw[:2]
    ignored = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--ignored", "--", relative], cwd=ROOT, text=True
    )
    return "!!" if ignored else "clean"


def source_files() -> list[dict[str, str]]:
    paths = [MODULE_PATH, EXPORT_PATH, TEST_PATH, HARNESS_PATH, GOLDEN_PATH, HISTORICAL_PATH]
    return [
        {"path": path, "sha256": file_sha256(path), "git_status": git_status(path)}
        for path in sorted(paths)
    ]


def finite_points() -> list[tuple[str, str, np.ndarray]]:
    points = [
        ("truth_free", "a0_truth_anchor", TRUTH_FREE.copy()),
        ("phase2s_center", "a0_phase2s_center_anchor", PHASE2S_CENTER.copy()),
    ]
    for index in range(4):
        minus = PHASE2S_CENTER.copy()
        plus = PHASE2S_CENTER.copy()
        minus[index] -= SHELL_STEP
        plus[index] += SHELL_STEP
        points.extend([
            (f"shell_{index}_minus", f"phase2s_shell_coordinate_{index}_minus", minus),
            (f"shell_{index}_plus", f"phase2s_shell_coordinate_{index}_plus", plus),
        ])
    return points


def float_hex(value: float) -> str:
    return float(value).hex()


def hex_array(value: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float_hex(float(array))
    return [hex_array(item) for item in array]


def finite_design() -> dict[str, Any]:
    finite = []
    for name, role, point in finite_points():
        finite.append({
            "name": name, "role": role, "input_hex": hex_array(point),
            "finite_difference_step_hex": float_hex(FD_STEP),
            "fd_rtol_hex": float_hex(FD_RTOL), "fd_atol_hex": float_hex(FD_ATOL),
            "historical_rtol_hex": float_hex(HISTORICAL_SCALE),
            "historical_atol_hex": float_hex(0.0),
            "value_parity_atol_hex": float_hex(VALUE_PARITY_SCALE),
            "score_parity_atol_hex": float_hex(SCORE_PARITY_ATOL),
        })
    nonfinite = [
        {"name": "nan_scalar", "role": "nonfinite_input_reject", "input_strings": ["nan", float_hex(-0.08), float_hex(0.65), float_hex(0.05)]},
        {"name": "inf_scalar", "role": "nonfinite_input_reject", "input_strings": ["inf", float_hex(-0.08), float_hex(0.65), float_hex(0.05)]},
        {"name": "truth_nan_inf_batch", "role": "ordered_mixed_batch_reject", "input_strings": [hex_array(TRUTH_FREE), ["nan", float_hex(-0.08), float_hex(0.65), float_hex(0.05)], ["inf", float_hex(-0.08), float_hex(0.65), float_hex(0.05)]]},
    ]
    return {"finite_points": finite, "nonfinite_cases": nonfinite}


def load_historical_target() -> Any:
    require(file_sha256(HISTORICAL_PATH) == HISTORICAL_SHA256, "historical source hash drift")
    path = repo_path(HISTORICAL_PATH)
    spec = importlib.util.spec_from_file_location("ssl_lstm_a1_historical_harness", path)
    require(spec is not None and spec.loader is not None, "cannot load historical comparator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_filtering_geometry_target()


def guarded_branch(free: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    free = tf.debugging.check_numerics(free, "guarded finite branch received nonfinite input")
    return -0.5 * tf.reduce_sum(tf.square(free)), -free


def cpu_point_results(target: SSLLSTMPosteriorTarget) -> list[dict[str, Any]]:
    historical = load_historical_target()
    lock = strict_load("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json")
    rows = []
    for name, _role, point in finite_points():
        tensor = tf.constant(point, tf.float64)
        eager_value_t, eager_score_t = target.eager_debug_value_and_score(tensor)
        xla_value_t, xla_score_t, status_t = target.diagnostic_value_and_score(tensor)
        historical_value_t, historical_score_t = historical._value_and_score_impl(tensor)
        eager_value = float(eager_value_t.numpy())
        xla_value = float(xla_value_t.numpy())
        historical_value = float(historical_value_t.numpy())
        eager_score = np.asarray(eager_score_t.numpy(), dtype=np.float64)
        xla_score = np.asarray(xla_score_t.numpy(), dtype=np.float64)
        historical_score = np.asarray(historical_score_t.numpy(), dtype=np.float64)
        finite_difference = np.empty(4, dtype=np.float64)
        for index in range(4):
            plus = point.copy(); plus[index] += FD_STEP
            minus = point.copy(); minus[index] -= FD_STEP
            plus_value, _ = target.eager_debug_value_and_score(tf.constant(plus, tf.float64))
            minus_value, _ = target.eager_debug_value_and_score(tf.constant(minus, tf.float64))
            finite_difference[index] = (float(plus_value.numpy()) - float(minus_value.numpy())) / (2.0 * FD_STEP)
        historical_value_residual = abs(eager_value - historical_value)
        historical_score_residual = float(np.max(np.abs(eager_score - historical_score)))
        value_residual = abs(eager_value - xla_value)
        score_residual = float(np.max(np.abs(eager_score - xla_score)))
        fd_abs = float(np.max(np.abs(eager_score - finite_difference)))
        fd_scale = np.maximum(np.abs(eager_score), np.abs(finite_difference))
        fd_relative = float(np.max(np.abs(eager_score - finite_difference) / np.maximum(fd_scale, np.finfo(np.float64).tiny)))
        historical_value_tolerance = HISTORICAL_SCALE * max(1.0, abs(eager_value), abs(historical_value))
        historical_score_tolerance = HISTORICAL_SCALE * max(1.0, float(np.max(np.abs(eager_score))), float(np.max(np.abs(historical_score))))
        value_tolerance = VALUE_PARITY_SCALE * max(1.0, abs(eager_value), abs(xla_value))
        fd_pass = bool(np.all(np.abs(eager_score - finite_difference) <= FD_ATOL + FD_RTOL * np.abs(finite_difference)))
        passed = bool(
            int(status_t.numpy()) == 0
            and all(math.isfinite(item) for item in [eager_value, xla_value, historical_value])
            and np.all(np.isfinite(eager_score)) and np.all(np.isfinite(xla_score))
            and np.all(np.isfinite(historical_score)) and np.all(np.isfinite(finite_difference))
            and historical_value_residual <= historical_value_tolerance
            and historical_score_residual <= historical_score_tolerance
            and value_residual <= value_tolerance and score_residual <= SCORE_PARITY_ATOL
            and fd_pass
        )
        if name in {"truth_free", "phase2s_center"}:
            probe = lock["probe_results"][name]
            probe_value = float(probe["total_value"]["values"])
            probe_score = np.asarray(probe["total_score"]["values"], dtype=np.float64)
            probe_value_tolerance = HISTORICAL_SCALE * max(1.0, abs(eager_value), abs(probe_value))
            probe_score_tolerance = HISTORICAL_SCALE * max(
                1.0, float(np.max(np.abs(eager_score))), float(np.max(np.abs(probe_score)))
            )
            passed = passed and abs(eager_value - probe_value) <= probe_value_tolerance
            passed = passed and float(np.max(np.abs(eager_score - probe_score))) <= probe_score_tolerance
        rows.append({
            "name": name, "input_hex": hex_array(point),
            "historical_value_hex": float_hex(historical_value),
            "historical_score_hex": hex_array(historical_score),
            "eager_value_hex": float_hex(eager_value), "eager_score_hex": hex_array(eager_score),
            "cpu_xla_value_hex": float_hex(xla_value), "cpu_xla_score_hex": hex_array(xla_score),
            "status": int(status_t.numpy()), "finite_difference_score_hex": hex_array(finite_difference),
            "historical_value_abs_residual": historical_value_residual,
            "historical_score_abs_residual_inf": historical_score_residual,
            "value_abs_residual": value_residual, "score_abs_residual_inf": score_residual,
            "fd_abs_residual_inf": fd_abs, "fd_relative_residual_inf": fd_relative,
            "historical_value_tolerance": historical_value_tolerance,
            "historical_score_tolerance": historical_score_tolerance,
            "value_tolerance": value_tolerance, "score_tolerance": SCORE_PARITY_ATOL,
            "fd_atol": FD_ATOL, "fd_rtol": FD_RTOL, "passed": passed,
        })
    return rows


def cpu_reject_results() -> list[dict[str, Any]]:
    target = SSLLSTMPosteriorTarget(finite_branch_callable=guarded_branch, testing_only=True)
    nan_row = tf.constant([np.nan, -0.08, 0.65, 0.05], tf.float64)
    inf_row = tf.constant([np.inf, -0.08, 0.65, 0.05], tf.float64)
    rows = []
    for name, tensor in (("nan_scalar", nan_row), ("inf_scalar", inf_row)):
        value, score, status = target.diagnostic_value_and_score(tensor)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            log_prob = target.log_prob(tensor)
        gradient = tape.gradient(log_prob, tensor)
        value_np = float(value.numpy()); score_np = np.asarray(score.numpy()); gradient_np = np.asarray(gradient.numpy())
        passed = bool(value_np == -1.0e100 and int(status.numpy()) == 1 and np.array_equal(score_np, np.zeros(4)) and np.array_equal(gradient_np, np.zeros(4)))
        rows.append({"name": name, "input_strings": finite_design()["nonfinite_cases"][len(rows)]["input_strings"], "value_hex": float_hex(value_np), "score_hex": hex_array(score_np), "status": int(status.numpy()), "gradient_hex": hex_array(gradient_np), "finite_branch_runtime_assertion_not_triggered": True, "passed": passed})
    batch = tf.stack([tf.constant(TRUTH_FREE, tf.float64), nan_row, inf_row])
    values, scores, statuses = target.diagnostic_value_and_score(batch)
    gradient_rows = []
    for row in tf.unstack(batch, num=3):
        with tf.GradientTape() as tape:
            tape.watch(row)
            row_log_prob = target.log_prob(row)
        gradient_rows.append(tape.gradient(row_log_prob, row))
    gradients = tf.stack(gradient_rows)
    values_np = np.asarray(values.numpy()); scores_np = np.asarray(scores.numpy()); statuses_np = np.asarray(statuses.numpy()); gradients_np = np.asarray(gradients.numpy())
    passed = bool(
        statuses_np.tolist() == [0, 1, 1]
        and values_np[1:].tolist() == [-1.0e100, -1.0e100]
        and np.array_equal(scores_np[1:], np.zeros([2, 4]))
        and np.array_equal(gradients_np[1:], np.zeros([2, 4]))
    )
    rows.append({"name": "truth_nan_inf_batch", "input_strings": finite_design()["nonfinite_cases"][2]["input_strings"], "value_hex": hex_array(values_np), "score_hex": hex_array(scores_np), "status": statuses_np.tolist(), "gradient_hex": hex_array(gradients_np), "finite_branch_runtime_assertion_not_triggered": True, "passed": passed})
    return rows


def a0_bindings() -> dict[str, str]:
    return {
        "target_semantic_sha256": TARGET_SEMANTIC_SHA256,
        "signature_aggregate_sha256": A0_SIGNATURE_AGGREGATE_SHA256,
        "immutable_aggregate_sha256": A0_IMMUTABLE_AGGREGATE_SHA256,
        "target_lock_file_sha256": A0_TARGET_LOCK_FILE_SHA256,
        "dependency_manifest_file_sha256": A0_DEPENDENCY_MANIFEST_FILE_SHA256,
        "observation_raw_sha256": OBSERVATION_RAW_SHA256,
        "full_fixture_raw_sha256": FULL_FIXTURE_RAW_SHA256,
    }


def a1_signatures() -> dict[str, str]:
    return {
        "target_semantic_sha256": TARGET_SEMANTIC_SHA256,
        "parameter_mask_sha256": PARAMETER_MASK_SHA256,
        "masked_posterior_contract_sha256": MASKED_POSTERIOR_CONTRACT_SHA256,
        "golden_signatures_file_sha256": GOLDEN_SIGNATURES_FILE_SHA256,
    }


def boundary_bindings(entry: Mapping[str, Any], boundary: Mapping[str, Any]) -> dict[str, str]:
    return {
        "a0_entry_verification_file_sha256": file_sha256(ENTRY_PATH),
        "pre_run_scoped_boundary_file_sha256": file_sha256(BOUNDARY_PATH),
        "protected_dependency_aggregate_sha256": entry["protected_dependency_aggregate_sha256"],
        "excluded_dependency_aggregate_sha256": entry["excluded_dependency_aggregate_sha256"],
        "scoped_boundary_aggregate_sha256": boundary["aggregate_sha256"],
    }


def devices(kind: str) -> list[dict[str, str]]:
    values = tf.config.list_physical_devices() if kind == "physical" else tf.config.list_logical_devices()
    return sorted(
        ({"device_type": str(item.device_type), "name": str(item.name)} for item in values),
        key=lambda row: (row["device_type"], row["name"]),
    )


def package_versions() -> dict[str, str]:
    return {
        "tensorflow": importlib.metadata.version("tensorflow"),
        "tensorflow_probability_distribution": importlib.metadata.version("tfp-nightly"),
        "numpy": importlib.metadata.version("numpy"),
    }


def environment_values() -> dict[str, str]:
    keys = ["CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "TF_DETERMINISTIC_OPS", "TF_ENABLE_ONEDNN_OPTS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "TF_CPP_MIN_LOG_LEVEL"]
    return {key: os.environ.get(key, "not_set") for key in keys}


def expected_environment(schema: str) -> dict[str, str]:
    common = {
        "PYTHONHASHSEED": "0", "TF_DETERMINISTIC_OPS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0", "TF_CPP_MIN_LOG_LEVEL": "1",
    }
    if schema == CPU_SCHEMA:
        return {
            "CUDA_VISIBLE_DEVICES": "-1", **common, "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1",
        }
    return {
        "CUDA_VISIBLE_DEVICES": "not_set", **common,
        "TF_NUM_INTRAOP_THREADS": "not_set", "TF_NUM_INTEROP_THREADS": "not_set",
        "OMP_NUM_THREADS": "not_set",
    }


def run_manifest(mode: str, command: str, output: str, log_path: str, started: datetime, wall: float, git_commit: str) -> dict[str, Any]:
    physical = devices("physical"); logical = devices("logical")
    cpu = mode == "cpu-reference"
    return {
        "git_commit": git_commit, "git_dirty": bool(git_text(["status", "--porcelain=v1"])),
        "command": command, "cwd": str(ROOT), "interpreter": sys.executable,
        "python_version": sys.version.split()[0], "packages": package_versions(), "conda_env": "tfgpu",
        "environment": environment_values(), "physical_devices": physical, "logical_devices": logical,
        "cpu_gpu_status": "cpu_hidden_no_gpu_visible" if cpu else "trusted_gpu_visible_compiled_output_on_gpu",
        "trust_basis": "cpu_hidden_reference_exception_not_gpu_evidence" if cpu else "owner_designated_managed_session_visible_gpu_trusted",
        "dtype": "float64", "jit_compile": True, "xla": True,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "data_version": OBSERVATION_RAW_SHA256, "random_seeds": "N/A_deterministic_target_no_randomness",
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(UTC).isoformat(),
        "wall_time_seconds": float(wall), "output_path": output, "log_path": log_path,
        "plan_path": PLAN_PATH, "result_path": RESULT_PATH,
    }


def evidence_projection(artifact: Mapping[str, Any]) -> dict[str, Any]:
    keys = ["schema_version", "artifact_role", "a0_bindings", "a1_signatures", "boundary_bindings", "source_files", "test_point_design", "point_results", "reject_results", "contract_checks", "nonclaims"]
    return {key: artifact[key] for key in keys}


def source_contract_checks(target: SSLLSTMPosteriorTarget) -> dict[str, bool]:
    source = repo_path(MODULE_PATH).read_text(encoding="utf-8")
    golden = strict_load(GOLDEN_PATH)
    mask = SSLLSTMParameterMask()
    config = SSLLSTMPosteriorConfig(parameter_mask=mask)
    point = tf.constant(TRUTH_FREE, tf.float64)
    value, score = target.value_and_score(point)
    alias_value, alias_score = target.log_prob_and_grad(point)
    with tf.GradientTape() as tape:
        tape.watch(point); log_prob = target.log_prob(point)
    gradient = tape.gradient(log_prob, point)
    callable_aliases = bool(
        value.numpy().tobytes() == target.value(point).numpy().tobytes()
        and score.numpy().tobytes() == target.score(point).numpy().tobytes()
        and value.numpy().tobytes() == alias_value.numpy().tobytes()
        and score.numpy().tobytes() == alias_score.numpy().tobytes()
        and value.numpy().tobytes() == log_prob.numpy().tobytes()
        and score.numpy().tobytes() == gradient.numpy().tobytes()
    )
    embedded = mask.embed(point); extracted = mask.extract(embedded)
    batch_ok = True
    for size in (1, 4, 10):
        batch = tf.constant(np.stack([item[2] for item in finite_points()[:size]]), tf.float64)
        values, scores = target.batch_value_and_score(batch)
        batch_ok = batch_ok and values.shape == (size,) and scores.shape == (size, 4)
    testing = SSLLSTMPosteriorTarget(finite_branch_callable=guarded_branch, testing_only=True)
    testing_unavailable = True
    for method in (testing.target_signature, testing.adapter_signature, testing.assert_production_evidence_target):
        try:
            method(); testing_unavailable = False
        except RuntimeError:
            pass
    finite_failure_loud = False
    def failing(_free: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.constant(np.nan, tf.float64), tf.zeros([4], tf.float64)
    failure_target = SSLLSTMPosteriorTarget(SSLLSTMPosteriorConfig(jit_compile=False, execution_role="eager_debug_reference"), finite_branch_callable=failing, testing_only=True)
    try:
        failure_target.eager_debug_value_and_score(point)
    except tf.errors.InvalidArgumentError:
        finite_failure_loud = True
    valid_branch_bitwise = True
    for anchor in (TRUTH_FREE, PHASE2S_CENTER):
        anchor_tensor = tf.constant(anchor, tf.float64)
        direct_value, direct_score = guarded_branch(anchor_tensor)
        wrapped_value, wrapped_score = testing.eager_debug_value_and_score(anchor_tensor)
        valid_branch_bitwise = valid_branch_bitwise and direct_value.numpy().tobytes() == wrapped_value.numpy().tobytes()
        valid_branch_bitwise = valid_branch_bitwise and direct_score.numpy().tobytes() == wrapped_score.numpy().tobytes()
    return {
        "mask_schema_valid": mask.signature_payload()["schema_version"] == "bayesfilter.ssl_lstm_completion.parameter_mask.v1",
        "mask_golden_digest_match": mask.signature() == golden["parameter_mask"]["sha256"] == PARAMETER_MASK_SHA256,
        "wrapper_schema_valid": config.signature_payload()["schema_version"] == "bayesfilter.ssl_lstm_completion.masked_posterior_contract.v1",
        "wrapper_golden_digest_match": config.signature() == golden["masked_posterior_contract"]["sha256"] == MASKED_POSTERIOR_CONTRACT_SHA256,
        "golden_file_exact": file_sha256(GOLDEN_PATH) == GOLDEN_SIGNATURES_FILE_SHA256,
        "historical_source_hash_exact": file_sha256(HISTORICAL_PATH) == HISTORICAL_SHA256,
        "target_semantic_digest_match": target.target_signature() == TARGET_SEMANTIC_SHA256,
        "embed_extract_exact": bool(tf.reduce_all(tf.equal(extracted, point)).numpy()),
        "prior_convention_exact": config.prior_standard_deviation == 4.0 and config.prior_normalized is False and tuple(config.prior_center.numpy()) == tuple(TRUTH_FREE),
        "scalar_shapes_exact": value.shape == () and score.shape == (4,),
        "batch_shapes_exact": batch_ok, "batch_sizes_1_4_10_exact": target.compiled_batch_sizes() == (1, 4, 10),
        "callable_aliases_exact": callable_aliases, "compiled_default_invoked": target.compiled_scalar_trace_count() == 1,
        "finite_filter_failure_loud": finite_failure_loud,
        "valid_branch_bitwise_equal": valid_branch_bitwise,
        "testing_only_provenance_unavailable": testing_unavailable,
        "testing_only_artifact_refused": testing_unavailable,
        "no_benchmark_import": "docs.benchmarks" not in source and "benchmark_scalar_ssl_lstm" not in source,
        "no_numpy_algorithmic_path": not re.search(r"(^|\n)\s*(import numpy|from numpy)", source),
        "no_tf_py_function": "tf.py_function" not in source,
        "historical_filter_route_only": "tf_ssl_lstm_svd_ukf_score" in source and "tf_principal_sqrt_ukf_score" not in source,
        "authority_not_self_certified": target.value_score_capability().xla_hmc_ready is False and target.value_score_capability().full_chain_xla_diagnostic_ready is False,
    }


def build_cpu_artifact(output: str, log_path: str) -> dict[str, Any]:
    started = datetime.now(UTC); timer = time.perf_counter()
    stage("cpu_boundary_verification")
    entry, boundary, opening_head, _paths = verify_entry_boundary()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU mode requires CUDA_VISIBLE_DEVICES=-1")
    require(not any(row["device_type"] == "GPU" for row in devices("physical")), "GPU visible in CPU mode")
    target = locked_ssl_lstm_posterior_target(); target.assert_production_evidence_target()
    stage("cpu_ten_point_evidence")
    points = cpu_point_results(target)
    stage("cpu_reject_evidence")
    rejects = cpu_reject_results()
    stage("cpu_source_and_batch_contracts")
    checks = source_contract_checks(target)
    checks.update({
        "a0_entry_verified": True, "scoped_boundary_verified": True,
        "historical_all_ten_points_passed": all(row["historical_value_abs_residual"] <= row["historical_value_tolerance"] and row["historical_score_abs_residual_inf"] <= row["historical_score_tolerance"] for row in points),
        "nonfinite_input_reject_exact": all(row["passed"] for row in rejects),
        "reject_gradient_zero": all(row["passed"] for row in rejects),
        "finite_difference_passed": all(row["fd_abs_residual_inf"] <= row["fd_atol"] + row["fd_rtol"] * max(abs(float.fromhex(item)) for item in row["finite_difference_score_hex"]) for row in points),
        "eager_cpu_xla_passed": all(row["value_abs_residual"] <= row["value_tolerance"] and row["score_abs_residual_inf"] <= row["score_tolerance"] for row in points),
        "point_order_exact": [row["name"] for row in points] == [row["name"] for row in finite_design()["finite_points"]],
    })
    exact_keys(checks, CPU_CHECK_KEYS - {"all_passed"}, "CPU preliminary checks")
    checks["all_passed"] = all(checks.values()) and all(row["passed"] for row in points) and all(row["passed"] for row in rejects)
    require(checks["all_passed"], "CPU contract check failed")
    require(git_text(["rev-parse", "HEAD"]) == opening_head, "HEAD changed during CPU run")
    stage("cpu_artifact_assembly")
    artifact = {
        "schema_version": CPU_SCHEMA, "artifact_role": "phase_a1_cpu_hidden_reference",
        "status": "phase_a1_cpu_reference_passed", "created_at_utc": datetime.now(UTC).isoformat(),
        "run_manifest": run_manifest("cpu-reference", CPU_COMMAND, output, log_path, started, time.perf_counter() - timer, opening_head),
        "a0_bindings": a0_bindings(), "a1_signatures": a1_signatures(),
        "boundary_bindings": boundary_bindings(entry, boundary), "source_files": source_files(),
        "test_point_design": finite_design(), "point_results": points, "reject_results": rejects,
        "contract_checks": checks, "evidence_signature": "", "nonclaims": list(NONCLAIMS),
    }
    artifact["evidence_signature"] = canonical_sha256(evidence_projection(artifact))
    return artifact


def gpu_point_results(target: SSLLSTMPosteriorTarget, cpu: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    device_ok = True
    cpu_rows = {row["name"]: row for row in cpu["point_results"]}
    for name, _role, point in finite_points():
        value_t, score_t, status_t = target.diagnostic_value_and_score(tf.constant(point, tf.float64))
        value = float(value_t.numpy()); score = np.asarray(score_t.numpy(), dtype=np.float64)
        device_ok = device_ok and "GPU" in value_t.device.upper() and "GPU" in score_t.device.upper()
        cpu_row = cpu_rows[name]
        cpu_value = float.fromhex(cpu_row["cpu_xla_value_hex"])
        cpu_score = np.asarray([float.fromhex(item) for item in cpu_row["cpu_xla_score_hex"]], dtype=np.float64)
        value_residual = abs(value - cpu_value); score_residual = float(np.max(np.abs(score - cpu_score)))
        value_tolerance = VALUE_PARITY_SCALE * max(1.0, abs(value), abs(cpu_value))
        passed = bool(int(status_t.numpy()) == 0 and math.isfinite(value) and np.all(np.isfinite(score)) and value_residual <= value_tolerance and score_residual <= SCORE_PARITY_ATOL)
        rows.append({"name": name, "input_hex": hex_array(point), "cpu_xla_value_hex": cpu_row["cpu_xla_value_hex"], "cpu_xla_score_hex": cpu_row["cpu_xla_score_hex"], "gpu_xla_value_hex": float_hex(value), "gpu_xla_score_hex": hex_array(score), "status": int(status_t.numpy()), "value_abs_residual": value_residual, "score_abs_residual_inf": score_residual, "value_tolerance": value_tolerance, "score_tolerance": SCORE_PARITY_ATOL, "passed": passed})
    return rows, device_ok


def build_gpu_artifact(cpu_path: str, output: str, log_path: str) -> dict[str, Any]:
    started = datetime.now(UTC); timer = time.perf_counter()
    stage("gpu_boundary_verification")
    entry, boundary, opening_head, _paths = verify_entry_boundary()
    require("CUDA_VISIBLE_DEVICES" not in os.environ, "GPU mode requires CUDA_VISIBLE_DEVICES unset")
    cpu = strict_load(cpu_path); validate_artifact_structure(cpu, CPU_SCHEMA)
    require(cpu["evidence_signature"] == canonical_sha256(evidence_projection(cpu)), "CPU evidence signature mismatch")
    require(cpu["boundary_bindings"] == boundary_bindings(entry, boundary), "CPU boundary bindings drift")
    physical = devices("physical"); logical = devices("logical")
    gpu_visible = any(row["device_type"] == "GPU" for row in physical) and any(row["device_type"] == "GPU" for row in logical)
    require(gpu_visible, "trusted GPU is not visible")
    target = locked_ssl_lstm_posterior_target(); target.assert_production_evidence_target()
    stage("gpu_ten_point_evidence")
    points, device_ok = gpu_point_results(target, cpu)
    checks: dict[str, Any] = {
        "cpu_reference_file_sha256": file_sha256(cpu_path), "cpu_reference_verified": True,
        "gpu_visible": gpu_visible, "trusted_provenance_recorded": True,
        "jit_compile_true": target.config.jit_compile is True, "xla_executed": target.compiled_scalar_trace_count() == 1,
        "gpu_device_placement_verified": device_ok, "tf32_recorded": isinstance(tf.config.experimental.tensor_float_32_execution_enabled(), bool),
        "cpu_gpu_parity_passed": all(row["passed"] for row in points),
        "signatures_equal": cpu["a1_signatures"] == a1_signatures(),
        "point_order_exact": [row["name"] for row in points] == [row["name"] for row in finite_design()["finite_points"]],
    }
    checks["all_passed"] = all(value for key, value in checks.items() if key != "cpu_reference_file_sha256")
    require(checks["all_passed"], "GPU contract check failed")
    require(git_text(["rev-parse", "HEAD"]) == opening_head, "HEAD changed during GPU run")
    artifact = {
        "schema_version": GPU_SCHEMA, "artifact_role": "phase_a1_trusted_gpu_xla_canary",
        "status": "phase_a1_gpu_xla_canary_passed", "created_at_utc": datetime.now(UTC).isoformat(),
        "run_manifest": run_manifest("gpu-xla-canary", GPU_COMMAND, output, log_path, started, time.perf_counter() - timer, opening_head),
        "a0_bindings": a0_bindings(), "a1_signatures": a1_signatures(),
        "boundary_bindings": boundary_bindings(entry, boundary), "source_files": source_files(),
        "test_point_design": finite_design(), "point_results": points, "reject_results": [],
        "contract_checks": checks, "evidence_signature": "", "nonclaims": list(NONCLAIMS),
    }
    artifact["evidence_signature"] = canonical_sha256(evidence_projection(artifact))
    return artifact


def validate_artifact_structure(artifact: Mapping[str, Any], schema: str) -> None:
    exact_keys(artifact, TOP_LEVEL_KEYS, "artifact")
    require(artifact["schema_version"] == schema, "artifact schema mismatch")
    exact_keys(artifact["run_manifest"], RUN_MANIFEST_KEYS, "run manifest")
    exact_keys(artifact["a0_bindings"], A0_KEYS, "A0 bindings")
    exact_keys(artifact["a1_signatures"], A1_KEYS, "A1 signatures")
    exact_keys(artifact["boundary_bindings"], BOUNDARY_KEYS, "boundary bindings")
    exact_keys(artifact["contract_checks"], CPU_CHECK_KEYS if schema == CPU_SCHEMA else GPU_CHECK_KEYS, "contract checks")
    exact_keys(artifact["run_manifest"]["packages"], {"tensorflow", "tensorflow_probability_distribution", "numpy"}, "run manifest packages")
    exact_keys(artifact["run_manifest"]["environment"], ENVIRONMENT_KEYS, "run manifest environment")
    require(artifact["evidence_signature"] == canonical_sha256(evidence_projection(artifact)), "evidence signature mismatch")
    require(artifact["nonclaims"] == list(NONCLAIMS), "nonclaims mismatch")
    require(artifact["a0_bindings"] == a0_bindings(), "A0 bindings mismatch")
    require(artifact["a1_signatures"] == a1_signatures(), "A1 signatures mismatch")
    require(artifact["test_point_design"] == finite_design(), "test point design mismatch")
    exact_keys(artifact["test_point_design"], {"finite_points", "nonfinite_cases"}, "test point design")
    for row in artifact["test_point_design"]["finite_points"]:
        exact_keys(row, FINITE_DESIGN_KEYS, "finite point design row")
    for row in artifact["test_point_design"]["nonfinite_cases"]:
        exact_keys(row, NONFINITE_DESIGN_KEYS, "nonfinite design row")
    for row in artifact["source_files"]:
        exact_keys(row, SOURCE_ROW_KEYS, "source row")
    require(artifact["source_files"] == source_files(), "source files mismatch")
    expected_point_keys = CPU_POINT_KEYS if schema == CPU_SCHEMA else GPU_POINT_KEYS
    for row in artifact["point_results"]:
        exact_keys(row, expected_point_keys, "point result row")
        require(isinstance(row["status"], int) and not isinstance(row["status"], bool), "point status type mismatch")
        require(isinstance(row["passed"], bool), "point passed type mismatch")
    for row in artifact["reject_results"]:
        exact_keys(row, REJECT_ROW_KEYS, "reject result row")
        require(isinstance(row["passed"], bool), "reject passed type mismatch")
    require(len(artifact["point_results"]) == 10, "point result count mismatch")
    require(len(artifact["reject_results"]) == (3 if schema == CPU_SCHEMA else 0), "reject result count mismatch")
    require(artifact["run_manifest"]["packages"] == {"tensorflow": "2.20.0", "tensorflow_probability_distribution": "0.25.0", "numpy": "2.1.3"}, "package versions mismatch")
    require(artifact["run_manifest"]["python_version"] == "3.13.13", "Python version mismatch")
    manifest = artifact["run_manifest"]
    verify_rfc3339(artifact["created_at_utc"], "created_at_utc")
    verify_rfc3339(manifest["started_at_utc"], "started_at_utc")
    verify_rfc3339(manifest["completed_at_utc"], "completed_at_utc")
    require_finite_number(manifest["wall_time_seconds"], "wall_time_seconds")
    require(float(manifest["wall_time_seconds"]) >= 0.0, "negative wall time")
    require(manifest["git_dirty"] is True, "git_dirty must be true in this worktree")
    require(manifest["cwd"] == str(ROOT), "manifest cwd mismatch")
    require(manifest["interpreter"] == sys.executable, "manifest interpreter mismatch")
    require(manifest["conda_env"] == "tfgpu", "manifest conda env mismatch")
    require(manifest["environment"] == expected_environment(schema), "manifest environment mismatch")
    require(manifest["dtype"] == "float64" and manifest["jit_compile"] is True and manifest["xla"] is True, "manifest execution contract mismatch")
    require(manifest["data_version"] == OBSERVATION_RAW_SHA256, "manifest data version mismatch")
    require(manifest["random_seeds"] == "N/A_deterministic_target_no_randomness", "manifest seed policy mismatch")
    require(manifest["plan_path"] == PLAN_PATH and manifest["result_path"] == RESULT_PATH, "manifest plan/result path mismatch")
    require(isinstance(manifest["tf32_enabled"], bool), "tf32 type mismatch")
    expected_role = "phase_a1_cpu_hidden_reference" if schema == CPU_SCHEMA else "phase_a1_trusted_gpu_xla_canary"
    expected_status = "phase_a1_cpu_reference_passed" if schema == CPU_SCHEMA else "phase_a1_gpu_xla_canary_passed"
    expected_command = CPU_COMMAND if schema == CPU_SCHEMA else GPU_COMMAND
    expected_output = CPU_PATH if schema == CPU_SCHEMA else GPU_PATH
    expected_log = CPU_LOG_PATH if schema == CPU_SCHEMA else GPU_LOG_PATH
    require(artifact["artifact_role"] == expected_role and artifact["status"] == expected_status, "artifact role/status mismatch")
    require(manifest["command"] == expected_command, "manifest command mismatch")
    require(manifest["output_path"] == expected_output and manifest["log_path"] == expected_log, "manifest output/log mismatch")
    physical_gpu = any(
        isinstance(row, dict) and row.get("device_type") == "GPU"
        for row in manifest["physical_devices"]
    )
    logical_gpu = any(
        isinstance(row, dict) and row.get("device_type") == "GPU"
        for row in manifest["logical_devices"]
    )
    for kind in ("physical_devices", "logical_devices"):
        for row in manifest[kind]:
            exact_keys(row, {"device_type", "name"}, f"{kind} row")
        require(manifest[kind] == sorted(manifest[kind], key=lambda row: (row["device_type"], row["name"])), f"{kind} ordering mismatch")
    if schema == CPU_SCHEMA:
        require(not physical_gpu and not logical_gpu, "CPU artifact records a GPU")
        require(manifest["cpu_gpu_status"] == "cpu_hidden_no_gpu_visible", "CPU status mismatch")
        require(manifest["trust_basis"] == "cpu_hidden_reference_exception_not_gpu_evidence", "CPU trust mismatch")
    else:
        require(physical_gpu and logical_gpu, "GPU artifact lacks GPU provenance")
        require(manifest["cpu_gpu_status"] == "trusted_gpu_visible_compiled_output_on_gpu", "GPU status mismatch")
        require(manifest["trust_basis"] == "owner_designated_managed_session_visible_gpu_trusted", "GPU trust mismatch")
    require(all(isinstance(value, bool) and value for key, value in artifact["contract_checks"].items() if key != "cpu_reference_file_sha256"), "contract checks are not all true")


def verify_artifact(path: str) -> None:
    stored = strict_load(path); schema = stored.get("schema_version")
    require(schema in {CPU_SCHEMA, GPU_SCHEMA}, "unsupported artifact schema")
    validate_artifact_structure(stored, schema)
    entry, boundary, current_head, _paths = verify_entry_boundary()
    require(stored["boundary_bindings"] == boundary_bindings(entry, boundary), "stored boundary bindings mismatch")
    recorded_head = stored["run_manifest"]["git_commit"]
    require(
        subprocess.run(["git", "merge-base", "--is-ancestor", A0_ANCHOR, recorded_head], cwd=ROOT).returncode == 0,
        "recorded run commit is outside A0 history",
    )
    recorded_paths = committed_history_paths(A0_ANCHOR, recorded_head)
    protected = {row["path"] for row in entry["protected_dependency_rows"]}
    require(not (set(recorded_paths) & (protected | set(boundary["owned_exact"]))), "recorded run history touched a forbidden path")
    if schema == CPU_SCHEMA:
        recomputed = build_cpu_artifact(stored["run_manifest"]["output_path"], stored["run_manifest"]["log_path"])
    else:
        cpu_path = CPU_PATH
        require(stored["contract_checks"]["cpu_reference_file_sha256"] == file_sha256(cpu_path), "stored CPU file hash mismatch")
        recomputed = build_gpu_artifact(cpu_path, stored["run_manifest"]["output_path"], stored["run_manifest"]["log_path"])
    require(evidence_projection(stored) == evidence_projection(recomputed), "recomputed artifact evidence mismatch")
    require(stored["run_manifest"]["tf32_enabled"] == bool(tf.config.experimental.tensor_float_32_execution_enabled()), "current tf32 setting mismatch")
    require(stored["run_manifest"]["physical_devices"] == devices("physical"), "current physical devices mismatch")
    require(stored["run_manifest"]["logical_devices"] == devices("logical"), "current logical devices mismatch")
    require(git_text(["rev-parse", "HEAD"]) == current_head, "HEAD changed during verification")
    print(json.dumps({"status": "phase_a1_artifact_verified", "artifact": path, "evidence_signature": stored["evidence_signature"]}, sort_keys=True, separators=(",", ":")))


def write_artifact(artifact: Mapping[str, Any], output: str, log_path: str) -> None:
    path = repo_path(output); log = repo_path(log_path)
    expected_head = artifact["run_manifest"]["git_commit"]
    require(git_text(["rev-parse", "HEAD"]) == expected_head, "HEAD changed before artifact publication")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    validate_artifact_structure(strict_load(path), artifact["schema_version"])
    log.write_text(f"status={artifact['status']}\nevidence_signature={artifact['evidence_signature']}\nartifact={output}\n", encoding="utf-8")
    require(git_text(["rev-parse", "HEAD"]) == expected_head, "HEAD changed during artifact publication")
    print(json.dumps({"status": artifact["status"], "artifact": output, "evidence_signature": artifact["evidence_signature"]}, sort_keys=True, separators=(",", ":")))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=("cpu-reference", "gpu-xla-canary"))
    group.add_argument("--verify")
    parser.add_argument("--cpu-reference")
    parser.add_argument("--output")
    parser.add_argument("--log-path")
    args = parser.parse_args(argv)
    if args.mode:
        require(args.output is not None and args.log_path is not None, "generation requires --output and --log-path")
    if args.mode == "gpu-xla-canary":
        require(args.cpu_reference == CPU_PATH, "GPU mode requires the reviewed CPU path")
    if args.mode == "cpu-reference":
        require(args.cpu_reference is None, "CPU mode does not accept --cpu-reference")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.verify:
        verify_artifact(args.verify)
    elif args.mode == "cpu-reference":
        require(args.output == CPU_PATH and args.log_path == CPU_LOG_PATH, "CPU output paths mismatch")
        write_artifact(build_cpu_artifact(args.output, args.log_path), args.output, args.log_path)
    else:
        require(args.output == GPU_PATH and args.log_path == GPU_LOG_PATH, "GPU output paths mismatch")
        write_artifact(build_gpu_artifact(args.cpu_reference, args.output, args.log_path), args.output, args.log_path)


if __name__ == "__main__":
    main()

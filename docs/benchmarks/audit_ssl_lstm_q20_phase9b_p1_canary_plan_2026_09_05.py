#!/usr/bin/env python3
"""Audit the current-dated Phase 9B P1 canary without importing TensorFlow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-plan-2026-09-05.md"
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py"
P0_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight/run_manifest.json"
FACTOR_ROOT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z"
FACTOR_MANIFEST = FACTOR_ROOT / "run_manifest.json"
FACTOR_AUDIT = FACTOR_ROOT / "source-sync-audit.json"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
FACTOR_BACKEND = "tensorflow_eigh_strict_factor_cached"
STRICT_BACKEND = "tensorflow_eigh_strict"
EXPECTED_FACTOR_MANIFEST_SHA256 = "e0225382192ceb9da1e075cb9c7a91ed424e2c5c67adcfec7a0f34c05003a4e1"
EXPECTED_FACTOR_AUDIT_SHA256 = "9848779a1fe1611f3b3acfb666bec8b08bef6a42296fe30e84d3762d892c5933"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def git(command: tuple[str, ...]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    failures: list[str] = []
    if output.exists():
        failures.append("output_dir_exists")
    output.mkdir(parents=True, exist_ok=False)
    required = (PLAN, RUNNER, P0_MANIFEST, FACTOR_MANIFEST, FACTOR_AUDIT)
    for path in required:
        if not path.is_file():
            failures.append(f"missing:{path.relative_to(ROOT)}")
    checks: dict[str, Any] = {
        "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha256(PLAN) if PLAN.is_file() else None},
        "runner": {
            "path": str(RUNNER.relative_to(ROOT)),
            "sha256": sha256(RUNNER) if RUNNER.is_file() else None,
        },
        "p0": {"path": str(P0_MANIFEST.relative_to(ROOT))},
        "factor": {},
    }
    if P0_MANIFEST.is_file():
        p0 = read_json(P0_MANIFEST)
        checks["p0"]["status"] = p0.get("status")
        checks["p0"]["claim_boundary"] = p0.get("checks", {}).get("claim_boundary")
        if p0.get("status") != "PASS_PHASE9B_P0_SOURCE_PREFLIGHT":
            failures.append("p0_status_mismatch")
    if FACTOR_MANIFEST.is_file():
        factor = read_json(FACTOR_MANIFEST)
        observed = sha256(FACTOR_MANIFEST)
        checks["factor"]["manifest_sha256"] = observed
        checks["factor"]["manifest_status"] = factor.get("status")
        checks["factor"]["target_signature"] = factor.get("target_signature")
        checks["factor"]["backend"] = factor.get("profile", {}).get("principal_sqrt_backend")
        if observed != EXPECTED_FACTOR_MANIFEST_SHA256:
            failures.append("factor_manifest_hash_mismatch")
        if factor.get("target_signature") != TARGET_SIGNATURE:
            failures.append("factor_target_signature_mismatch")
        if checks["factor"]["backend"] != FACTOR_BACKEND:
            failures.append("factor_backend_mismatch")
    if FACTOR_AUDIT.is_file():
        audit = read_json(FACTOR_AUDIT)
        observed = sha256(FACTOR_AUDIT)
        checks["factor"]["audit_sha256"] = observed
        checks["factor"]["audit_status"] = audit.get("status")
        if observed != EXPECTED_FACTOR_AUDIT_SHA256:
            failures.append("factor_audit_hash_mismatch")
        if audit.get("status") != "PASS_FACTOR_SOURCE-SYNC_AUDIT":
            failures.append("factor_audit_status_mismatch")
    text = PLAN.read_text(encoding="utf-8") if PLAN.is_file() else ""
    for marker in (
        "PASS_P1_CANARY_PLAN_REVIEW_NO_P2_AUTHORITY",
        FACTOR_BACKEND,
        STRICT_BACKEND,
        "bayesfilter_neutra_sequential_hmc_v1",
        "warmup chunks 500",
        "retained chunks 500",
        "5,200 seconds",
        "P1_BLOCKED_M4_P0_BUDGET_INFEASIBLE_P2_BLOCKED",
        "bayesfilter_non_display_first_load40_headroom5g_v1",
        "TF_FORCE_GPU_ALLOW_GROWTH=true",
        "--phase0-closeout",
        "--sequential-chunk-forecast-seconds",
    ):
        if marker not in text:
            failures.append(f"plan_marker_missing:{marker}")
    runner_text = RUNNER.read_text(encoding="utf-8") if RUNNER.is_file() else ""
    checks["runner"].update(
        {
            "chart_selector": "_select_chart(chart_map, BETA)" in runner_text,
            "diagnostic_transpose": "tf.transpose(tensor, perm=(1, 0, 2))" in runner_text,
            "mcse": "posterior_mean_diagnostics" in runner_text,
            "budget_guard": "budget_check=budget_guard.before_chunk" in runner_text,
            "phase0_closeout": "phase0_closeout = _verify_phase0_closeout(" in runner_text,
            "complete_budget_guard": 'minimum_remaining_seconds=phase0_closeout["complete_schedule_forecast_seconds"]' in runner_text,
            "run_start": '"schema": "bayesfilter.ssl_lstm_q20.phase9b_p1_run_start.v1"' in runner_text,
            "seed_namespaces": "_seed_map" in runner_text,
            "gpu_placement": "gpu_selection = _select_phase9b_gpu()" in runner_text,
            "gpu_identity_binding": '"uuid", "pci_bus_id", "name", "memory_total_mib"' in runner_text,
            "gpu_environment_guard": 'os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != gpu_selection["selected"]["uuid"]' in runner_text,
            "stale_status_callback_absent": "target_status_summary_fn=adapter.target_status_telemetry" not in runner_text,
            "stale_chart_indexing_absent": "chart_map[BETA][CHART_INDEX]" not in runner_text,
        }
    )
    for name, passed in checks["runner"].items():
        if name != "path" and not passed:
            failures.append(f"runner_contract_missing:{name}")
    payload = {
        "schema": "bayesfilter.ssl_lstm_q20.phase9b_p1_plan_audit.v2",
        "status": "PASS_PHASE9B_P1_PLAN_AUDIT" if not failures else "FAIL_PHASE9B_P1_PLAN_AUDIT",
        "failures": failures,
        "claim_boundary": "phase9b_p1_sequential_canary_only",
        "checks": checks,
        "git": {
            "commit": git(("git", "rev-parse", "HEAD")),
            "status_sha256": hashlib.sha256(git(("git", "status", "--porcelain")).encode()).hexdigest(),
            "worktree_dirty": bool(git(("git", "status", "--porcelain"))),
        },
        "nonclaims": [
            "no posterior correctness",
            "no convergence promotion",
            "no ensemble or replica-exchange claim",
            "no sampler superiority",
            "no repository default change",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    (output / "p1-plan-audit.json").write_text(encoded, encoding="utf-8")
    manifest = {**payload, "manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
    (output / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": failures, "output_dir": str(output)}, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

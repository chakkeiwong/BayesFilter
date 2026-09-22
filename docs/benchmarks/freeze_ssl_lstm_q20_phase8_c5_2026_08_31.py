#!/usr/bin/env python3
"""Evaluate the metadata-only q=20 Phase 8 C5 freeze rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-subplan-2026-08-31.md"
CALIBRATION_PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-calibration-subplan-2026-08-29.md"
C2_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/screen/attempt-02-eight-rows/run_manifest.json"
C2_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-result-2026-08-30.md"
C3A_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-result-2026-08-31.md"
C3B_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-result-2026-08-31.md"
C4A_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c4a-joint-feasibility/attempt-01/run_manifest.json"
C4A_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4a-joint-feasibility-result-2026-08-31.md"
C4B_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c4b-joint-replication/attempt-01/run_manifest.json"
C4B_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4b-joint-replication-result-2026-08-31.md"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BACKEND = "tensorflow_eigh_strict"
SCHEMA = "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_c5_freeze.v1"
MATERIAL_CAP_SECONDS = 120.0


class FreezeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(args: tuple[str, ...]) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FreezeError(f"missing prerequisite: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeError(f"expected JSON object: {path}")
    return value


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FreezeError(f"missing prerequisite: {path}")
    return path.read_text(encoding="utf-8")


def _check() -> tuple[dict[str, Any], dict[str, Any]]:
    c2 = _read_json(C2_MANIFEST)
    c4a = _read_json(C4A_MANIFEST)
    c4b = _read_json(C4B_MANIFEST)
    c2_result = _read_text(C2_RESULT)
    c3a_result = _read_text(C3A_RESULT)
    c3b_result = _read_text(C3B_RESULT)
    c4a_result = _read_text(C4A_RESULT)
    c4b_result = _read_text(C4B_RESULT)
    _read_text(PLAN)
    _read_text(CALIBRATION_PLAN)
    checks = (
        c2.get("status") == "PASS_PHASE8_C2_STRICT_CALIBRATION",
        c2.get("target_signature") == TARGET_SIGNATURE,
        c2.get("principal_sqrt_backend") == BACKEND,
        "Status: `PASS_C2_STRICT_CALIBRATION_WITHOUT_WHITENING_PROMOTION`" in c2_result,
        "Status: `PASS_C3_LINEAGE_OVERLAP_WITH_DIVERSITY_REPAIR_NO_ARM_NOMINATION`" in c3a_result,
        "Status: `PASS_C3B_L5_OVERLAP_WITH_PAIRED_DIVERSITY_SIGNAL_NO_PROMOTION`" in c3b_result,
        c4a.get("status") == "PASS_PHASE8_C4A_JOINT_FEASIBILITY",
        c4a.get("target_signature") == TARGET_SIGNATURE,
        "Status: `PASS_C4A_JOINT_FEASIBILITY_NO_PROMOTION`" in c4a_result,
        c4b.get("status") == "PASS_PHASE8_C4B_JOINT_REPLICATION",
        c4b.get("target_signature") == TARGET_SIGNATURE,
        c4b.get("principal_sqrt_backend") == BACKEND,
        not c4b.get("failures"),
        "Status: `PASS_C4B_JOINT_REPLICATION_NO_PROMOTION`" in c4b_result,
    )
    if not all(checks):
        raise FreezeError("calibration prerequisite/status/identity check failed")
    return c2, c4b


def _select(c2: dict[str, Any], c4b: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in c2.get("rows", []):
        architecture = row.get("architecture", {})
        if architecture.get("name") != "compact-high" and architecture.get("name") != "compact-low":
            continue
        improvement = row.get("paired_improvement", {})
        upper = float(improvement.get("two_sided_95_upper", float("nan")))
        delta = float(improvement.get("mean_final_minus_start", float("nan")))
        if row.get("status") != "PASS_C2_ROW" or not math.isfinite(upper) or upper >= 0.0 or not math.isfinite(delta):
            raise FreezeError("an eligible compact C2 row failed the predeclared nomination rule")
        rows.append({"architecture": architecture, "root_index": row.get("root_index"), "delta": delta, "interval_upper": upper, "median_update_seconds": row.get("median_update_seconds")})
    if len(rows) != 4:
        raise FreezeError(f"expected four compact C2 rows, found {len(rows)}")
    aggregate: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["architecture"]["name"])
        item = aggregate.setdefault(name, {"architecture": row["architecture"], "deltas": [], "roots": []})
        item["deltas"].append(row["delta"])
        item["roots"].append(row["root_index"])
    for item in aggregate.values():
        item["mean_delta"] = sum(item["deltas"]) / len(item["deltas"])
    selected_name = min(aggregate, key=lambda name: float(aggregate[name]["mean_delta"]))
    selected = aggregate[selected_name]
    return {
        "k2": {
            "candidate_id": f"phase8-k2-{selected_name}-l3-pure",
            "status": "FROZEN_FOR_PHASE9_TUNING_ONLY",
            "component_count": 2,
            "ladder": [0.0, 0.5, 1.0],
            "branching_policy": "pure_continuation",
            "architecture": selected["architecture"],
            "gamma_policy": {"policy_id": "fixed_state_independent_chart_mixture_v1", "values": [0.5, 0.5], "state_independent": True},
            "selection_basis": "predeclared compact-family parsimony and lower mean paired C2 held-out change; operational nomination only",
            "c2_nomination": {"by_architecture": aggregate, "selected": selected_name},
            "confirmation_policy": "rebuild_or_tune_under_fresh_repository_scope; calibration checkpoints are not claim-bearing",
        },
        "k4": {
            "candidate_id": "phase8-k4-joint",
            "status": "NOT_RETAINED_FOR_PHASE9",
            "component_count": 4,
            "reason": "implementation/resource feasibility replicated, but no paired uncertainty-supported objective benefit; C4B row contrasts have opposite signs and the arm has quadratic cross-density cost",
            "source_manifest_status": c4b["status"],
        },
    }


def _run(output_dir: Path) -> int:
    if output_dir.exists():
        raise FreezeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    c2, c4b = _check()
    candidates = _select(c2, c4b)
    elapsed = time.monotonic() - started
    if elapsed > MATERIAL_CAP_SECONDS:
        raise FreezeError("C5 evaluator exceeded its material wall cap")
    inputs = (PLAN, CALIBRATION_PLAN, C2_MANIFEST, C2_RESULT, C3A_RESULT, C3B_RESULT, C4A_MANIFEST, C4A_RESULT, C4B_MANIFEST, C4B_RESULT, Path(__file__).resolve())
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_PHASE8_C5_FREEZE",
        "role": "metadata_only_confirmation_protocol_freeze",
        "target_signature": TARGET_SIGNATURE,
        "principal_sqrt_backend": BACKEND,
        "candidates": candidates,
        "inputs": {str(path.relative_to(ROOT)): {"sha256": _sha256(path)} for path in inputs},
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _git(("git", "status", "--porcelain")),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "wall_time_seconds": elapsed,
        "budget": {"material_cap_seconds": MATERIAL_CAP_SECONDS, "elapsed_seconds": elapsed},
        "nonclaims": ["no whitening", "no mode discovery", "no posterior correctness", "no convergence", "no HMC readiness", "no statistical or architecture ranking", "no high-dimensional scaling"],
    }
    payload["manifest_hash"] = _stable_hash(payload)
    (output_dir / "freeze_manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output_dir": str(output_dir), "selected_k2": candidates["k2"]["architecture"]["name"], "k4_status": candidates["k4"]["status"], "wall_time_seconds": elapsed}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return _run(args.output_dir.expanduser().resolve())
    except Exception as exc:
        print(json.dumps({"status": "FAIL_PHASE8_C5_FREEZE", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

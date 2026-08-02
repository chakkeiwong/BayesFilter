#!/usr/bin/env python3
"""Issue the deterministic validation-only selection ledger for Lane-B T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "zhao_cui_austria_sir_fixed_variant_training_base_v1"
EXPECTED_ARMS = (
    "p01_r2_b3_lr3e4_l1_0",
    "p02_r2_b3_lr3e4_l1_1e8",
    "p03_r4_b3_lr3e4_l1_1e9",
    "p04_r4_b3_lr3e4_l1_1e8",
    "p05_r4_b5_lr3e4_l1_1e9",
    "p06_r4_b5_lr1e4_l1_1e9",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_selection(pilot_root: Path) -> Mapping[str, Any]:
    root = pilot_root.resolve()
    rows = []
    seen = set()
    common_source_sha256 = None
    for result_path in sorted(root.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        arm = result.get("arm")
        arm_id = arm.get("arm_id") if isinstance(arm, Mapping) else None
        if arm_id not in EXPECTED_ARMS:
            continue
        if arm_id in seen:
            raise ValueError(f"duplicate Lane-B pilot arm: {arm_id}")
        seen.add(arm_id)
        if result.get("baseline_id") != BASELINE_ID:
            raise ValueError(f"Lane-B pilot baseline mismatch: {arm_id}")
        run_manifest = result.get("run_manifest")
        source_sha256 = (
            run_manifest.get("source_sha256")
            if isinstance(run_manifest, Mapping)
            else None
        )
        if not isinstance(source_sha256, Mapping) or not source_sha256:
            raise ValueError(f"Lane-B pilot source closure missing: {arm_id}")
        if common_source_sha256 is None:
            common_source_sha256 = dict(source_sha256)
        elif dict(source_sha256) != common_source_sha256:
            raise ValueError("Lane-B selection requires identical source closure")
        gates = result.get("gates")
        metrics = result.get("validation_metrics")
        if not isinstance(gates, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError(f"Lane-B pilot result is incomplete: {arm_id}")
        viable = result.get("status") == "VIABLE_T1_PILOT_ARM" and gates.get(
            "viable"
        ) is True
        if viable and (
            not result.get("artifact_manifest") or not result.get("artifact_identity")
        ):
            raise ValueError(f"viable Lane-B arm lacks an artifact: {arm_id}")
        rows.append(
            {
                "arm_id": arm_id,
                "result_path": _portable_path(result_path),
                "result_sha256": _sha256(result_path),
                "status": result.get("status"),
                "viable": viable,
                "validation_normalized_log_density_rms": metrics.get(
                    "normalized_log_density_rms"
                ),
                "artifact_manifest": result.get("artifact_manifest"),
                "artifact_identity": result.get("artifact_identity"),
            }
        )
    if seen != set(EXPECTED_ARMS):
        missing = sorted(set(EXPECTED_ARMS) - seen)
        raise ValueError(f"Lane-B selection requires all frozen pilot arms: {missing}")
    viable_rows = [row for row in rows if row["viable"]]
    selected = (
        None
        if not viable_rows
        else min(
            viable_rows,
            key=lambda row: (
                float(row["validation_normalized_log_density_rms"]),
                str(row["arm_id"]),
            ),
        )
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t1_selection.v1",
        "status": (
            "BLOCK_NO_VIABLE_T1_PILOT_ARM"
            if selected is None
            else "SELECTED_VIABLE_T1_PILOT_ARM"
        ),
        "baseline_id": BASELINE_ID,
        "selection_data_role": "validation_only_untouched_claim_not_read",
        "common_source_sha256": common_source_sha256,
        "expected_arm_ids": EXPECTED_ARMS,
        "arms": rows,
        "selected_arm_id": None if selected is None else selected["arm_id"],
        "selected_result_path": None if selected is None else selected["result_path"],
        "selected_result_sha256": None if selected is None else selected["result_sha256"],
        "selected_artifact_manifest": (
            None if selected is None else selected["artifact_manifest"]
        ),
        "selected_artifact_identity": (
            None if selected is None else selected["artifact_identity"]
        ),
        "criterion": (
            "minimum validation normalized-log-density RMS among arms passing "
            "all frozen hard gates; arm id is deterministic tie-breaker"
        ),
        "ranking_status": "descriptive_selection_only_no_statistically_supported_ranking",
        "untouched_seed_used": False,
        "next_action": (
            "same-route tuning repair"
            if selected is None
            else "one untouched T1 claim for the selected artifact"
        ),
        "nonclaims": (
            "no statistically supported superiority",
            "no T1 admission before untouched claim",
            "no score, T2, T20, HMC, or production claim",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"selection output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_selection(args.pilot_root), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

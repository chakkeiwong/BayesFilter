#!/usr/bin/env python3
"""Select one T2 arm using only the frozen validation criterion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "zhao_cui_austria_sir_fixed_variant_training_base_t2_v1"
EXPECTED_ARMS = (
    "t2_p01_r2_b3_lr3e4_l1_0",
    "t2_p02_r2_b3_lr3e4_l1_1e8",
    "t2_p03_r4_b3_lr3e4_l1_1e9",
    "t2_p04_r4_b3_lr3e4_l1_1e8",
    "t2_p05_r4_b5_lr3e4_l1_1e9",
    "t2_p06_r4_b5_lr1e4_l1_1e9",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_selection(pilot_root: Path) -> Mapping[str, Any]:
    rows = []
    seen = set()
    common_source = None
    common_prepared = None
    parent_identity = None
    for result_path in sorted(pilot_root.resolve().glob("*/result.json")):
        result = json.loads(result_path.read_text())
        arm = result.get("arm")
        arm_id = arm.get("arm_id") if isinstance(arm, Mapping) else None
        if arm_id not in EXPECTED_ARMS:
            continue
        if arm_id in seen:
            raise ValueError(f"duplicate T2 pilot arm: {arm_id}")
        seen.add(arm_id)
        if result.get("baseline_id") != BASELINE_ID:
            raise ValueError(f"T2 pilot baseline mismatch: {arm_id}")
        manifest = result.get("run_manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError(f"T2 pilot run manifest missing: {arm_id}")
        source = manifest.get("source_sha256")
        prepared = manifest.get("prepared_result_sha256")
        if not isinstance(source, Mapping) or not isinstance(prepared, Mapping):
            raise ValueError(f"T2 pilot provenance closure missing: {arm_id}")
        common_source = dict(source) if common_source is None else common_source
        common_prepared = dict(prepared) if common_prepared is None else common_prepared
        if dict(source) != common_source or dict(prepared) != common_prepared:
            raise ValueError("T2 selection requires identical source and prepared inputs")
        observed_parent = result.get("parent_t1_identity")
        parent_identity = observed_parent if parent_identity is None else parent_identity
        if observed_parent != parent_identity:
            raise ValueError("T2 selection requires one parent T1 identity")
        gates = result.get("gates")
        metrics = result.get("validation_metrics")
        if not isinstance(gates, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError(f"T2 pilot result incomplete: {arm_id}")
        viable = result.get("status") == "VIABLE_T2_PILOT_ARM" and gates.get("viable") is True
        if viable and (not result.get("artifact_manifest") or not result.get("artifact_identity")):
            raise ValueError(f"viable T2 arm lacks a reloadable artifact: {arm_id}")
        rows.append(
            {
                "arm_id": arm_id,
                "result_path": _portable(result_path),
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
        raise ValueError(
            f"T2 selection requires all frozen pilot arms: {sorted(set(EXPECTED_ARMS) - seen)}"
        )
    viable_rows = [row for row in rows if row["viable"]]
    selected = None if not viable_rows else min(
        viable_rows,
        key=lambda row: (
            float(row["validation_normalized_log_density_rms"]),
            str(row["arm_id"]),
        ),
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_selection.v1",
        "status": (
            "BLOCK_NO_VIABLE_T2_PILOT_ARM"
            if selected is None
            else "SELECTED_VIABLE_T2_PILOT_ARM"
        ),
        "baseline_id": BASELINE_ID,
        "parent_t1_identity": parent_identity,
        "selection_data_role": "validation_only_untouched_not_generated_or_read",
        "common_source_sha256": common_source,
        "common_prepared_result_sha256": common_prepared,
        "expected_arm_ids": EXPECTED_ARMS,
        "arms": rows,
        "selected_arm_id": None if selected is None else selected["arm_id"],
        "selected_result_path": None if selected is None else selected["result_path"],
        "selected_result_sha256": None if selected is None else selected["result_sha256"],
        "selected_artifact_manifest": None if selected is None else selected["artifact_manifest"],
        "selected_artifact_identity": None if selected is None else selected["artifact_identity"],
        "criterion": "minimum validation normalized-log-density RMS among viable frozen arms; arm id tie-breaker",
        "ranking_status": "descriptive_selection_only_no_statistically_supported_ranking",
        "untouched_seed_used": False,
        "next_action": "target_specific_T2_repair" if selected is None else "generate_one_untouched_T2_cloud",
        "nonclaims": (
            "no statistically supported superiority",
            "no T2 admission before untouched claim",
            "no score, T20, HMC, or production claim",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"selection output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_selection(args.pilot_root), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

"""Aggregate complete V6 crossed bundle/path variance campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.independent_score.variance_reduction_tf import (
    ARM_NAMES,
    classify_combined_arm,
    summarize_crossed_outputs,
)


EXPECTED_BUNDLES = 10
AGGREGATOR_PATH = Path(__file__).resolve()
SUMMARY_PATH = ROOT / "bayesfilter/independent_score/variance_reduction_tf.py"


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, tf.Tensor):
        return safe(value.numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_campaign(root: Path, *, expected_bundles: int = EXPECTED_BUNDLES) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    kind: str | None = None
    seen: set[int] = set()
    for bundle in range(int(expected_bundles)):
        path = root / f"bundle_{bundle:02d}" / "result.json"
        if not path.exists():
            raise ValueError(f"missing bundle result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETED":
            raise ValueError(f"bundle {bundle} is not complete")
        if int(payload.get("bundle", -1)) in seen or int(payload.get("bundle", -1)) != bundle:
            raise ValueError(f"duplicate or misnumbered bundle: {bundle}")
        seen.add(bundle)
        if tuple(payload.get("completed_arms", ())) != ARM_NAMES:
            raise ValueError(f"bundle {bundle} has incomplete or reordered arms")
        if kind is None:
            kind = str(payload["kind"])
        elif payload["kind"] != kind:
            raise ValueError("campaign mixes model kinds")
        rows.append(payload)
    return rows, str(kind)


def aggregate(root: Path, *, bootstrap_replicates: int = 5000) -> dict[str, Any]:
    rows, kind = load_campaign(root)
    audit_hashes = {row["audit_path_sha256"] for row in rows}
    fixed_hashes = {row["fixed_path_sha256"] for row in rows}
    if len(audit_hashes) != 1 or len(fixed_hashes) != 1:
        raise ValueError("bundle audit/fixed paths are not shared")
    pairing_audit = []
    for row in rows:
        arm_rows = row["arm_rows"]
        shared = [arm_rows[arm]["shared_split_hashes"] for arm in ARM_NAMES]
        if any(value != shared[0] for value in shared[1:]):
            raise ValueError(f"bundle {row['bundle']} does not share non-training splits")
        for cell_key in arm_rows[ARM_NAMES[0]]["cells"]:
            records = {arm: arm_rows[arm]["cells"][cell_key]["pair_hashes"] for arm in ARM_NAMES}
            for delta in records[ARM_NAMES[0]]:
                independent_small = records["independent_n2048"][delta]
                crn_small = records["crn_n2048"][delta]
                independent_large = records["independent_n8192"][delta]
                crn_large = records["crn_n8192"][delta]
                if independent_small["noise_identical"] or independent_large["noise_identical"]:
                    raise ValueError("independent arm reused plus/minus noise")
                if not crn_small["noise_identical"] or not crn_large["noise_identical"]:
                    raise ValueError("CRN arm did not reuse plus/minus noise")
                checks = (
                    independent_small["minus_noise_sha256"] == independent_large["minus_prefix_sha256"],
                    independent_small["plus_noise_sha256"] == independent_large["plus_prefix_sha256"],
                    crn_small["minus_noise_sha256"] == crn_large["minus_prefix_sha256"],
                    crn_small["plus_noise_sha256"] == crn_large["plus_prefix_sha256"],
                    independent_small["minus_noise_sha256"] == crn_small["minus_noise_sha256"],
                )
                if not all(checks):
                    raise ValueError("training prefix or cross-arm pairing audit failed")
        pairing_audit.append({"bundle": row["bundle"], "shared_splits": True, "nested_training_prefixes": True, "crn_identity": True, "independent_nonidentity": True})
    outputs = tf.stack([tf.cast(row["audit_outputs"], tf.float64) for row in rows], axis=1)
    fixed = tf.stack([tf.cast(row["fixed_outputs"], tf.float64) for row in rows], axis=1)
    if outputs.shape[0] != len(ARM_NAMES) or outputs.shape[1] != EXPECTED_BUNDLES:
        raise ValueError("crossed output shape is incomplete")
    exact = None
    exact_fixed = None
    if kind == "gaussian":
        exact_rows = [tf.cast(row["exact_audit_scores"], tf.float64) for row in rows]
        fixed_rows = [tf.cast(row["exact_fixed_score"], tf.float64) for row in rows]
        for value in exact_rows[1:]:
            tf.debugging.assert_equal(value, exact_rows[0])
        for value in fixed_rows[1:]:
            tf.debugging.assert_equal(value, fixed_rows[0])
        exact = exact_rows[0]
        exact_fixed = fixed_rows[0]
    summary = summarize_crossed_outputs(
        outputs,
        fixed,
        exact_scores=exact,
        exact_fixed_score=exact_fixed,
        bootstrap_replicates=int(bootstrap_replicates),
    )
    decision = classify_combined_arm(summary)
    return {
        "schema": "bayesfilter.classifier_score_variance_campaign.v1",
        "status": "COMPLETED",
        "kind": kind,
        "claim": "training_bundle_variance_reduction_only",
        "summary": summary,
        "combined_arm_decision": decision,
        "pairing_audit": pairing_audit,
        "all_hard_valid": all(
            cell["finite"] and cell["temperature"] > 0.0 and cell["optimizer_complete"]
            for row in rows
            for arm in row["arm_rows"].values()
            for cell in arm["cells"].values()
        ),
        "audit_path_sha256": next(iter(audit_hashes)),
        "fixed_path_sha256": next(iter(fixed_hashes)),
        "bundle_result_sha256": [hashlib.sha256((root / f"bundle_{index:02d}" / "result.json").read_bytes()).hexdigest() for index in range(EXPECTED_BUNDLES)],
        "aggregation_execution": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "device_policy": "cpu_only" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "visible_devices_not_hidden",
            "environment": "tftwogpu",
            "python": sys.executable,
            "git_commit": git_commit(),
            "source_hashes": {
                str(AGGREGATOR_PATH.relative_to(ROOT)): sha(AGGREGATOR_PATH),
                str(SUMMARY_PATH.relative_to(ROOT)): sha(SUMMARY_PATH),
            },
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "nonclaims": ["not natural path variance", "not exact SIR score", "not filter/HMC/default evidence"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    result = aggregate(args.root, bootstrap_replicates=args.bootstrap_replicates)
    result["aggregation_execution"]["command"] = [sys.executable, *sys.argv]
    write(args.output or args.root / "aggregate_result.json", result)


if __name__ == "__main__":
    main()

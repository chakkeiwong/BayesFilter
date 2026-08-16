"""Aggregate the V7 nested classifier-score path-count ladder."""

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

from bayesfilter.independent_score.path_count_scaling_tf import (
    summarize_path_count_scaling,
)


EXPECTED_BUNDLES = 10
BASELINE_ARM_INDEX = 2
BASELINE_ARM_NAME = "independent_n8192"
FROZEN_BASELINE_SOURCE_HASHES = {
    "bayesfilter/independent_score/anchored_orthogonal_ratio_score_tf.py": "ffe48587d582e0dbba11eb57d84a0b5b84933f15b9f0152bfbdc7019ea44f6f3",
    "docs/benchmarks/run_classifier_score_variance_bundle_20260815.py": "90cdf2d89fc9afba964507a541179dfbfc314aa690de4bee8353fb306d2daaa2",
}
AGGREGATOR_PATH = Path(__file__).resolve()
SUMMARY_PATH = ROOT / "bayesfilter/independent_score/path_count_scaling_tf.py"


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
    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline(root: Path) -> list[dict[str, Any]]:
    rows = []
    for bundle in range(EXPECTED_BUNDLES):
        folder = root / f"bundle_{bundle:02d}"
        result = load_json(folder / "result.json")
        manifest = load_json(folder / "manifest.json")
        if result.get("status") != "COMPLETED" or result.get("bundle") != bundle:
            raise ValueError(f"invalid baseline bundle {bundle}")
        if tuple(result.get("completed_arms", ())) != (
            "independent_n2048",
            "crn_n2048",
            "independent_n8192",
            "crn_n8192",
        ):
            raise ValueError("baseline arm order differs from frozen V6")
        for path, digest in FROZEN_BASELINE_SOURCE_HASHES.items():
            if manifest.get("source_hashes", {}).get(path) != digest:
                raise ValueError(f"baseline source hash mismatch: {path}")
            if sha(ROOT / path) != digest:
                raise ValueError(f"current source no longer matches baseline: {path}")
        if manifest.get("result_sha256") != sha(folder / "result.json"):
            raise ValueError("baseline result checksum mismatch")
        rows.append({"result": result, "manifest": manifest})
    return rows


def load_stage(root: Path, count: int) -> list[dict[str, Any]]:
    rows = []
    source_hashes = None
    for bundle in range(EXPECTED_BUNDLES):
        folder = root / f"bundle_{bundle:02d}"
        result = load_json(folder / "result.json")
        manifest = load_json(folder / "manifest.json")
        if result.get("status") != "COMPLETED" or result.get("bundle") != bundle:
            raise ValueError(f"invalid count-{count} bundle {bundle}")
        if result.get("path_count") != count or manifest.get("path_count") != count:
            raise ValueError("stage path count mismatch")
        if manifest.get("result_sha256") != sha(folder / "result.json"):
            raise ValueError("stage result checksum mismatch")
        current_hashes = manifest.get("source_hashes", {})
        if source_hashes is None:
            source_hashes = current_hashes
        elif current_hashes != source_hashes:
            raise ValueError("stage bundles use different source closures")
        for path, digest in current_hashes.items():
            if sha(ROOT / path) != digest:
                raise ValueError(f"stage source hash is stale: {path}")
        rows.append({"result": result, "manifest": manifest})
    return rows


def _pairing_audit(
    baseline: list[dict[str, Any]],
    stages: list[tuple[int, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    audit = []
    for bundle in range(EXPECTED_BUNDLES):
        base = baseline[bundle]["result"]
        base_arm = base["arm_rows"][BASELINE_ARM_NAME]
        base_cells = base_arm["cells"]
        previous_count = 8192
        previous = None
        for count, rows in stages:
            result = rows[bundle]["result"]
            if result["kind"] != base["kind"]:
                raise ValueError("stage kind differs from baseline")
            if result["audit_path_sha256"] != base["audit_path_sha256"]:
                raise ValueError("audit path mismatch")
            if result["fixed_path_sha256"] != base["fixed_path_sha256"]:
                raise ValueError("fixed path mismatch")
            if result["shared_split_hashes"] != base_arm["shared_split_hashes"]:
                raise ValueError("shared held-out split mismatch")
            for cell_key, cell in result["cells"].items():
                coordinate = cell_key.split("_j", 1)[1]
                base_key = next(
                    key for key in base_cells if key.endswith(f"_j{coordinate}")
                )
                base_pairs = base_cells[base_key]["pair_hashes"]
                for delta, hashes in cell["pair_hashes"].items():
                    if hashes["noise_identical"]:
                        raise ValueError("independent ladder reused plus/minus noise")
                    prefix = hashes["prefix_sha256"].get("8192")
                    if prefix is None:
                        raise ValueError("stage is missing the 8192 prefix hash")
                    if prefix["minus"] != base_pairs[delta]["minus_noise_sha256"]:
                        raise ValueError("minus prefix differs from V6 baseline")
                    if prefix["plus"] != base_pairs[delta]["plus_noise_sha256"]:
                        raise ValueError("plus prefix differs from V6 baseline")
                    if previous is not None:
                        previous_prefix = hashes["prefix_sha256"].get(
                            str(previous_count)
                        )
                        previous_hashes = previous["cells"][cell_key]["pair_hashes"][delta]
                        if previous_prefix is None:
                            raise ValueError("stage is missing the previous-count prefix")
                        if previous_prefix["minus"] != previous_hashes["minus_noise_sha256"]:
                            raise ValueError("minus nested prefix mismatch")
                        if previous_prefix["plus"] != previous_hashes["plus_noise_sha256"]:
                            raise ValueError("plus nested prefix mismatch")
            previous_count = count
            previous = result
        audit.append(
            {
                "bundle": bundle,
                "baseline_source_bound": True,
                "shared_evaluation_data": True,
                "independent_plus_minus": True,
                "exact_nested_prefixes": True,
            }
        )
    return audit


def _continuation_decision(
    gaussian: dict[str, Any], sir: dict[str, Any]
) -> dict[str, Any]:
    gaussian_summary = gaussian["summary"]
    sir_summary = sir["summary"]
    sir_audit = sir_summary["audit_adjacent_scaling"][0]
    gaussian_fixed = gaussian_summary["fixed_adjacent_scaling"][0]
    sir_fixed = sir_summary["fixed_adjacent_scaling"][0]
    exact_audit = gaussian_summary["exact_mse_adjacent_scaling"][0]
    exact_fixed = gaussian_summary["exact_fixed_mse_adjacent_scaling"][0]
    checks = {
        "all_hard_valid": gaussian["all_hard_valid"] and sir["all_hard_valid"],
        "sir_audit_reduction": sir_audit["ratio_upper_95"] < 1.0,
        "gaussian_accuracy_not_harmed": exact_audit["ratio_lower_95"] <= 1.0
        and exact_fixed["ratio_lower_95"] <= 1.0,
        "no_fixed_variance_worsening": gaussian_fixed["ratio_lower_95"] <= 1.0
        and sir_fixed["ratio_lower_95"] <= 1.0,
    }
    return {
        "continue_to_32768": all(checks.values()),
        "checks": checks,
        "rule": "frozen_v7_stage1_continuation_rule",
    }


def aggregate_model(
    baseline_root: Path,
    stage_roots: list[tuple[int, Path]],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    baseline = load_baseline(baseline_root)
    stages = [(count, load_stage(root, count)) for count, root in stage_roots]
    counts = (8192, *(count for count, _ in stages))
    pairing = _pairing_audit(baseline, stages)
    kind = baseline[0]["result"]["kind"]
    for row in baseline[1:]:
        if row["result"]["kind"] != kind:
            raise ValueError("baseline mixes model kinds")

    audit_levels = [
        tf.stack(
            [
                tf.cast(row["result"]["audit_outputs"][BASELINE_ARM_INDEX], tf.float64)
                for row in baseline
            ],
            axis=0,
        )
    ]
    fixed_levels = [
        tf.stack(
            [
                tf.cast(row["result"]["fixed_outputs"][BASELINE_ARM_INDEX], tf.float64)
                for row in baseline
            ],
            axis=0,
        )
    ]
    for _, rows in stages:
        audit_levels.append(
            tf.stack(
                [tf.cast(row["result"]["audit_outputs"], tf.float64) for row in rows],
                axis=0,
            )
        )
        fixed_levels.append(
            tf.stack(
                [tf.cast(row["result"]["fixed_outputs"], tf.float64) for row in rows],
                axis=0,
            )
        )
    outputs = tf.stack(audit_levels, axis=0)
    fixed = tf.stack(fixed_levels, axis=0)

    exact = None
    exact_fixed = None
    if kind == "gaussian":
        exact_rows = [
            tf.cast(row["result"]["exact_audit_scores"], tf.float64)
            for row in baseline
        ]
        fixed_rows = [
            tf.cast(row["result"]["exact_fixed_score"], tf.float64)
            for row in baseline
        ]
        for value in exact_rows[1:]:
            tf.debugging.assert_equal(value, exact_rows[0])
        for value in fixed_rows[1:]:
            tf.debugging.assert_equal(value, fixed_rows[0])
        exact = exact_rows[0]
        exact_fixed = fixed_rows[0]

    summary = summarize_path_count_scaling(
        outputs,
        fixed,
        counts=counts,
        exact_scores=exact,
        exact_fixed_score=exact_fixed,
        bootstrap_replicates=int(bootstrap_replicates),
    )
    all_hard_valid = all(
        cell["finite"]
        and cell["temperature"] > 0.0
        and cell["optimizer_complete"]
        for _, rows in stages
        for row in rows
        for cell in row["result"]["cells"].values()
    )
    return {
        "schema": "bayesfilter.classifier_score_path_count_scaling.model.v1",
        "status": "COMPLETED",
        "kind": kind,
        "counts": counts,
        "summary": summary,
        "pairing_audit": pairing,
        "all_hard_valid": all_hard_valid,
        "baseline_root": str(baseline_root),
        "stage_roots": [
            {"count": count, "root": str(root)} for count, root in stage_roots
        ],
        "nonclaims": [
            "not exact SIR score",
            "not fixed-update sample-size scaling",
            "not filter/HMC/default evidence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-baseline", type=Path, required=True)
    parser.add_argument("--sir-baseline", type=Path, required=True)
    parser.add_argument("--gaussian-16384", type=Path, required=True)
    parser.add_argument("--sir-16384", type=Path, required=True)
    parser.add_argument("--gaussian-32768", type=Path)
    parser.add_argument("--sir-32768", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    if (args.gaussian_32768 is None) != (args.sir_32768 is None):
        raise ValueError("both 32768 roots must be provided together")
    gaussian_stages = [(16384, args.gaussian_16384)]
    sir_stages = [(16384, args.sir_16384)]
    if args.gaussian_32768 is not None and args.sir_32768 is not None:
        gaussian_stages.append((32768, args.gaussian_32768))
        sir_stages.append((32768, args.sir_32768))
    gaussian = aggregate_model(
        args.gaussian_baseline,
        gaussian_stages,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    sir = aggregate_model(
        args.sir_baseline,
        sir_stages,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    result = {
        "schema": "bayesfilter.classifier_score_path_count_scaling.campaign.v1",
        "status": "COMPLETED",
        "gaussian": gaussian,
        "sir": sir,
        "stage1_continuation_decision": _continuation_decision(gaussian, sir),
        "aggregation_execution": {
            "command": [sys.executable, *sys.argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "device_policy": "cpu_only"
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            else "visible_devices_not_hidden",
            "environment": "tftwogpu",
            "python": sys.executable,
            "git_commit": git_commit(),
            "source_hashes": {
                str(AGGREGATOR_PATH.relative_to(ROOT)): sha(AGGREGATOR_PATH),
                str(SUMMARY_PATH.relative_to(ROOT)): sha(SUMMARY_PATH),
            },
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    write(args.output, result)


if __name__ == "__main__":
    main()

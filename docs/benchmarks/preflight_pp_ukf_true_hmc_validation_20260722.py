#!/usr/bin/env python3
"""Build a fail-closed preflight manifest for true PP-UKF HMC validation.

This command reconstructs frozen controls and checks the sequential-controller
configuration. It does not launch HMC, consume validation samples, or make a
posterior/convergence claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path("docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md")
SOURCE_PRIVATE = Path(
    "docs/plans/artifacts/bayesfilter-pp-ukf-statistical-compatibility-guard-repair-20260721/"
    "attempt-01/private_result.json"
)
SOURCE_MANIFEST = Path(
    "docs/plans/artifacts/bayesfilter-pp-ukf-statistical-compatibility-guard-repair-20260721/"
    "attempt-01/run_manifest.json"
)
EXPECTED_L = (5, 9, 12, 13, 14, 17, 18, 19, 24, 25)
TARGET_SIGNATURE = "d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5"
TRANSPORT_SHA256 = "b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def build_candidate_manifest() -> Mapping[str, Any]:
    source = json.loads((ROOT / SOURCE_PRIVATE).read_text(encoding="utf-8"))
    grid = source["grid"]
    # The preserved source is an immutable v1 artifact from before the v2
    # next-round field. Reconstruct the union from its primary/coverage rows;
    # do not rewrite or silently upgrade the historical payload.
    primary_by_l = {
        int(item["request"]["num_leapfrog_steps"]): item
        for item in grid["primary_candidates"]
        if item.get("viable") is True
    }
    rows = []
    for item in primary_by_l.values():
        request = item["request"]
        evidence = item["evidence"]
        l_value = int(request["num_leapfrog_steps"])
        if l_value not in EXPECTED_L:
            raise ValueError(f"unexpected next-round candidate L={l_value}")
        epsilon = float(item["tuned_step_size"])
        provenance = "independently_tuned_primary"
        parent = None
        if evidence["disposition"] != "provisional_viable":
            raise ValueError(f"next-round candidate L={l_value} is not compatible")
        candidate_id = f"primary-l{l_value}-source-{_sha256_json(item)[:16]}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "num_leapfrog_steps": l_value,
                "step_size": epsilon,
                "control_provenance": provenance,
                "parent_candidate_id": parent,
                "tuning_evidence_signature": item.get("tune_evidence_signature"),
                "source_evidence_signature": evidence.get("evidence_signature"),
                "target_signature": TARGET_SIGNATURE,
                "transport_sha256": TRANSPORT_SHA256,
            }
        )
    primary_ids = {
        int(row["num_leapfrog_steps"]): row["candidate_id"] for row in rows
    }
    for item in grid["guard_candidates"]:
        if item.get("viable") is not True:
            continue
        request = item["request"]
        evidence = item["evidence"]
        l_value = int(request["num_leapfrog_steps"])
        if evidence["disposition"] != "provisional_viable":
            continue
        parents = tuple(request["parent_l_values"])
        if len(parents) != 1 or int(parents[0]) not in primary_ids:
            raise ValueError(f"coverage L={l_value} has no unique compatible parent")
        epsilon = float(request["inherited_step_size"])
        rows.append(
            {
                "candidate_id": f"coverage-l{l_value}-source-{_sha256_json(item)[:16]}",
                "num_leapfrog_steps": l_value,
                "step_size": epsilon,
                "control_provenance": "inherited_exact_one_hop_coverage",
                "parent_candidate_id": primary_ids[int(parents[0])],
                "parent_l_values": parents,
                "source_parent_candidate_signatures": tuple(
                    request["parent_candidate_signatures"]
                ),
                "tuning_evidence_signature": None,
                "source_evidence_signature": evidence.get("evidence_signature"),
                "target_signature": TARGET_SIGNATURE,
                "transport_sha256": TRANSPORT_SHA256,
            }
        )
    rows.sort(key=lambda item: item["num_leapfrog_steps"])
    if tuple(row["num_leapfrog_steps"] for row in rows) != EXPECTED_L:
        raise ValueError("candidate manifest is incomplete or reordered")
    return {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.candidate_manifest.v1",
        "plan": str(PLAN),
        "source_private_result": str(SOURCE_PRIVATE),
        "source_private_result_sha256": _sha256(ROOT / SOURCE_PRIVATE),
        "source_schema_reconstructed": grid.get("schema"),
        "target_signature": TARGET_SIGNATURE,
        "transport_sha256": TRANSPORT_SHA256,
        "metric_policy": "fixed_identity",
        "candidate_count": len(rows),
        "candidates": rows,
        "selection": "unranked_next_round_union",
        "ranking_performed": False,
        "retained_sampling_launched": False,
        "nonclaims": (
            "preflight only",
            "no HMC execution",
            "no posterior convergence claim",
            "no sampler ranking or scientific claim",
        ),
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_preflight(*, output_root: Path) -> Mapping[str, Any]:
    candidate_manifest = build_candidate_manifest()
    source_manifest = json.loads((ROOT / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig

    controller_policies = []
    for index, candidate in enumerate(candidate_manifest["candidates"]):
        controller = SequentialNeuTraHMCConfig(
            step_size=float(candidate["step_size"]),
            num_leapfrog_steps=int(candidate["num_leapfrog_steps"]),
            warmup_seed=(20260722, 6201 + 1009 * index),
            retained_seed=(20260722, 7201 + 1009 * index),
            warmup_chunk_results=1000,
            warmup_min_results=2000,
            warmup_check_window_results=1000,
            warmup_max_results=10000,
            warmup_rhat_max=1.05,
            retained_chunk_results=1000,
            retained_min_results=1000,
            retained_max_results=10000,
            retained_rhat_max=1.01,
            minimum_chain_count=4,
            jit_compile=True,
        )
        controller_policies.append(
            {
                "candidate_id": candidate["candidate_id"],
                "num_leapfrog_steps": candidate["num_leapfrog_steps"],
                "step_size": candidate["step_size"],
                "policy": controller.payload(chain_count=4),
            }
        )
    result = {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.preflight.v1",
        "status": "blocked_before_sampling_missing_fresh_partition_and_budget",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "plan": str(PLAN),
        "candidate_manifest": candidate_manifest,
        "source_run_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": _sha256(ROOT / SOURCE_MANIFEST),
            "prior_cumulative_charged_seconds": source_manifest.get("cumulative_charged_seconds"),
        },
        "sequential_controller_policies": tuple(controller_policies),
        "retained_ess_thresholds": {
            "status": "not_declared_before_launch",
            "bulk_min": None,
            "tail_min": None,
        },
        "fresh_validation_partition_required": True,
        "new_compute_budget_required": True,
        "gpu_sampling_launched": False,
        "nonclaims": candidate_manifest["nonclaims"],
    }
    _write_new_json(output_root / "candidate_manifest.json", candidate_manifest)
    _write_new_json(output_root / "preflight.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_preflight(output_root=args.output_root)
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

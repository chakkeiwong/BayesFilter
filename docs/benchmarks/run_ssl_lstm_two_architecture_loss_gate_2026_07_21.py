#!/usr/bin/env python3
"""Summarize the q=20 two-architecture loss-only NeuTra gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "bayesfilter.ssl_lstm.two_architecture_loss_gate.v1"
ARCHITECTURES = ((32, 32), (64, 64))
STREAMS = ("seed-a", "seed-b")
EXPECTED_PARAMS = {
    "gradient_clip_norm": 10.0,
    "initialization_scale": 0.01,
    "learning_rate": 4.0e-4,
}
EXPECTED_STREAMS = {
    "seed-a": {
        "initialization_seed": [20260719, 12101],
        "label": "seed-a",
        "training_seed": [20260719, 13101],
        "validation_seed": [20260719, 14101],
    },
    "seed-b": {
        "initialization_seed": [20260719, 12102],
        "label": "seed-b",
        "training_seed": [20260719, 13102],
        "validation_seed": [20260719, 14102],
    },
}
EXPECTED_AUDIT_DEFINITION = "stateless_validation_seed_fold_20260721_final_only"
TWO_SIDED_95_T_CRITICAL_DF_255 = 1.9693105698498752
EXPECTED_PLAN = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-two-architecture-loss-gate-plan-2026-07-21.md"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_interval(left: list[float], right: list[float]) -> dict[str, Any]:
    differences = np.asarray(right, np.float64) - np.asarray(left, np.float64)
    if differences.shape != (256,):
        raise ValueError("audit vectors must contain exactly 256 values")
    mean = float(np.mean(differences))
    se = float(np.std(differences, ddof=1) / math.sqrt(differences.size))
    critical = TWO_SIDED_95_T_CRITICAL_DF_255
    return {
        "n": int(differences.size),
        "mean_difference_right_minus_left": mean,
        "standard_error": se,
        "two_sided_95_lower": mean - critical * se,
        "two_sided_95_upper": mean + critical * se,
    }


def load_arm(root: Path, architecture: tuple[int, int], stream: str) -> dict[str, Any]:
    result_path = root / stream / "result.json"
    summary_path = root / "final-summary.json"
    if not result_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"missing terminal artifacts for {architecture}/{stream}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETED":
        raise RuntimeError(f"arm summary is not completed: {architecture}/{stream}")
    manifest = summary.get("run_manifest", {})
    if (
        summary.get("q") != 20
        or summary.get("params") != EXPECTED_PARAMS
        or summary.get("loss_only_control") is not True
        or manifest.get("batch_size") != 100
        or manifest.get("hidden_layers") != list(architecture)
        or manifest.get("loss_only_control") is not True
        or manifest.get("jit_compile_parent") is not True
        or manifest.get("tf32") is not True
        or manifest.get("plan") != EXPECTED_PLAN
    ):
        raise RuntimeError(f"arm manifest contract mismatch: {architecture}/{stream}")
    if result.get("q") != 20 or result.get("params") != EXPECTED_PARAMS:
        raise RuntimeError(f"arm result contract mismatch: {architecture}/{stream}")
    if result.get("stream") != EXPECTED_STREAMS[stream]:
        raise RuntimeError(f"arm stream mismatch: {architecture}/{stream}")
    if result.get("status") != "ADMITTED" or result.get("vetoes"):
        raise RuntimeError(f"arm failed admission gate: {architecture}/{stream}")
    if result.get("audit", {}).get("batch_size") != 256:
        raise RuntimeError(f"arm audit batch mismatch: {architecture}/{stream}")
    audit = result["audit"]
    if audit.get("audit_definition") != EXPECTED_AUDIT_DEFINITION:
        raise RuntimeError(f"arm audit definition mismatch: {architecture}/{stream}")
    losses = [float(value) for value in audit["per_sample_loss"]]
    if len(losses) != 256 or not np.all(np.isfinite(losses)):
        raise RuntimeError(f"invalid audit losses: {architecture}/{stream}")
    checkpoints = result.get("checkpoints", [])
    if not checkpoints:
        raise RuntimeError(f"arm has no checkpoint receipts: {architecture}/{stream}")
    selected_checkpoint = next(
        (row for row in checkpoints if int(row.get("step", -1)) == int(result["best_step"])),
        None,
    )
    if selected_checkpoint is None:
        raise RuntimeError(f"selected checkpoint is missing: {architecture}/{stream}")
    checkpoint_path = ROOT / selected_checkpoint["path"]
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != selected_checkpoint["sha256"]:
        raise RuntimeError(f"selected checkpoint hash mismatch: {architecture}/{stream}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("checkpoint_hash") != selected_checkpoint.get("checkpoint_hash"):
        raise RuntimeError(f"selected checkpoint payload mismatch: {architecture}/{stream}")
    frozen_path = ROOT / result["best_frozen_payload_path"]
    if not frozen_path.is_file() or sha256(frozen_path) != result["best_frozen_payload_sha256"]:
        raise RuntimeError(f"frozen payload hash mismatch: {architecture}/{stream}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    best_history = next(
        (row for row in result["history"] if int(row["step"]) == int(result["best_step"])),
        None,
    )
    if best_history is None:
        raise RuntimeError(f"selected validation row is missing: {architecture}/{stream}")
    recorded = {
        "architecture": list(architecture),
        "stream": stream,
        "summary_path": summary_path.relative_to(ROOT).as_posix(),
        "summary_sha256": sha256(summary_path),
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": sha256(result_path),
        "status": result["status"],
        "best_step": int(result["best_step"]),
        "terminal_program_step": int(result["terminal_program_step"]),
        "stop_reason": result["stop_reason"],
        "selected_best_validation_mean_loss": float(best_history["mean_loss"]),
        "raw_min_validation_mean_loss": float(
            min(row["mean_loss"] for row in result["history"])
        ),
        "audit_mean_loss": float(audit["mean_loss"]),
        "audit_per_sample_loss": losses,
        "params": result["params"],
        "stream_payload": result["stream"],
        "controller_config": checkpoint["controller_state"]["config"],
        "controller_policy": checkpoint["controller_state"]["repair_trigger_policy"],
        "target_signature": frozen["target_signature"],
        "transport_procedure": frozen["procedure"],
        "source_sha256": summary["source_bindings"]["source_sha256"],
        "support_probe": {
            key: result["support_probe"][key]
            for key in ("all_finite", "roundtrip_max_abs", "moderate_shell_max_inverse_radius")
        },
    }
    if recorded["support_probe"]["all_finite"] is not True:
        raise RuntimeError(f"nonfinite support probe: {architecture}/{stream}")
    if recorded["support_probe"]["roundtrip_max_abs"] > 1.0e-9:
        raise RuntimeError(f"roundtrip hard veto: {architecture}/{stream}")
    if recorded["support_probe"]["moderate_shell_max_inverse_radius"] > 4.3:
        raise RuntimeError(f"support-radius hard veto: {architecture}/{stream}")
    return recorded


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    output = (ROOT / args.output).resolve()
    arms: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for architecture in ARCHITECTURES:
        by_key[architecture] = {}
        for stream in STREAMS:
            arm = load_arm(root / f"arch-{architecture[0]}x{architecture[1]}", architecture, stream)
            arms.append(arm)
            by_key[architecture][stream] = arm
    controller_configs = [arm["controller_config"] for arm in arms]
    controller_policies = [arm["controller_policy"] for arm in arms]
    target_signatures = [arm["target_signature"] for arm in arms]
    protocol_identity_match = bool(
        len({json.dumps(value, sort_keys=True) for value in controller_configs}) == 1
        and len(set(controller_policies)) == 1
        and len(set(target_signatures)) == 1
    )
    if not protocol_identity_match:
        raise RuntimeError("executed controller or target protocol mismatch")
    source_keys = ("runner", "target", "pool", "trainer", "controller")
    terminal_source_snapshot_identity_by_key = {
        key: len({arm["source_sha256"][key] for arm in arms}) == 1
        for key in source_keys
    }
    terminal_source_snapshot_identity_match = all(
        terminal_source_snapshot_identity_by_key.values()
    )
    comparisons = []
    for stream in STREAMS:
        narrow = by_key[(32, 32)][stream]
        wide = by_key[(64, 64)][stream]
        interval = paired_interval(narrow["audit_per_sample_loss"], wide["audit_per_sample_loss"])
        comparisons.append({"stream": stream, **interval})
    directions = [np.sign(row["mean_difference_right_minus_left"]) for row in comparisons]
    interval_passes = [
        row["two_sided_95_lower"] > 0.0 or row["two_sided_95_upper"] < 0.0
        for row in comparisons
    ]
    consistent = bool(
        all(interval_passes)
        and all(direction != 0 for direction in directions)
        and len(set(directions)) == 1
    )
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "primary_metric": "independent_256_draw_audit_mean_loss",
        "architecture_labels": {"left": [32, 32], "right": [64, 64]},
        "arms": arms,
        "paired_audit_comparisons": comparisons,
        "hard_veto_screen": "PASS_ALL_FOUR_ARMS",
        "executed_protocol_identity_match": protocol_identity_match,
        "terminal_source_snapshot_identity_match": terminal_source_snapshot_identity_match,
        "terminal_source_snapshot_identity_by_key": (
            terminal_source_snapshot_identity_by_key
        ),
        "comparison_validity": (
            "VALID"
            if terminal_source_snapshot_identity_match
            else "VALID_WITH_SOURCE_PROVENANCE_LIMITATION"
        ),
        "source_provenance_limitation": (
            None
            if terminal_source_snapshot_identity_match
            else (
                "controller terminal filesystem snapshots differ because the source "
                "changed after the 64x64 process imported it; serialized controller "
                "configs, policies, target signatures, and all exercised hard-veto "
                "decisions match, but launch-time imported hashes were not preserved"
            )
        ),
        "consistent_direction": consistent,
        "fixed_protocol_nomination": (
            "64x64_lower_audit_loss"
            if consistent and directions[0] < 0
            else "32x32_lower_audit_loss"
            if consistent and directions[0] > 0
            else "UNRESOLVED"
        ),
        "nonclaims": [
            "two training seeds do not establish broad statistical superiority",
            "no posterior correctness, convergence proof, HMC readiness, or default promotion",
            "fixed protocol does not establish architecture-specific hyperparameter optimality",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "consistent_direction", "fixed_protocol_nomination")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

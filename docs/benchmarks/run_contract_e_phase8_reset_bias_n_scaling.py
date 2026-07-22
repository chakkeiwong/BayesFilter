#!/usr/bin/env python3
"""Run the reviewed six-node Contract E versus no-reset N diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.transport_chunk_policy import TRANSPORT_CHUNK_POLICY_ID


WORKER = ROOT / "docs/benchmarks/run_contract_e_phase8_lower_rung_node.py"
PLAN = ROOT / "docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-reset-bias-n-scaling-subplan-2026-07-14.md"
PARTICLE_COUNTS = (32, 64, 128)
RESET_POLICIES = ("all_active_contract_e", "no_reset_weighted")
RIDGE = 0.1225 * (2.0**-24)
STEPS = 20
NODE_TIMEOUT_SECONDS = 300
CAMPAIGN_CAP_SECONDS = 7200
PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_errors(node: dict[str, Any]) -> list[float]:
    value = abs(float(node["center"]["objective"]) - float(node["kalman"]["objective"])) / abs(float(node["kalman"]["objective"]))
    gradients = [
        abs(float(candidate) - float(oracle)) / abs(float(oracle))
        for candidate, oracle in zip(
            node["center"]["hmc_score"], node["kalman"]["hmc_score"], strict=True
        )
    ]
    return [value, *gradients]


def _paired_reset_effect(contract_e: dict[str, Any], no_reset: dict[str, Any]) -> list[float]:
    if contract_e["kalman"] != no_reset["kalman"]:
        raise ValueError("Kalman identity differs between paired arms")
    value = abs(float(contract_e["center"]["objective"]) - float(no_reset["center"]["objective"])) / abs(float(contract_e["kalman"]["objective"]))
    gradients = [
        abs(float(left) - float(right)) / abs(float(oracle))
        for left, right, oracle in zip(
            contract_e["center"]["hmc_score"],
            no_reset["center"]["hmc_score"],
            contract_e["kalman"]["hmc_score"],
            strict=True,
        )
    ]
    return [value, *gradients]


def _strict_improvement(values: list[float]) -> list[bool]:
    if len(values) != 3:
        raise ValueError("reviewed diagnostic requires exactly three N values")
    return [values[1] < values[0], values[2] < values[1]]


def _nonincreasing(values: list[float]) -> list[bool]:
    if len(values) != 3:
        raise ValueError("reviewed diagnostic requires exactly three N values")
    return [values[1] <= values[0], values[2] <= values[1]]


def _classify(
    errors: dict[str, list[list[float]]], reset_effects: list[list[float]]
) -> tuple[str, dict[str, Any]]:
    quantities = ("value", *PARAMETER_NAMES)
    details: dict[str, Any] = {}
    shared = True
    reset_specific = True
    for index, quantity in enumerate(quantities):
        contract_values = [row[index] for row in errors["all_active_contract_e"]]
        no_reset_values = [row[index] for row in errors["no_reset_weighted"]]
        reset_values = [row[index] for row in reset_effects]
        contract_improves = _strict_improvement(contract_values)
        no_reset_improves = _strict_improvement(no_reset_values)
        reset_nonincreasing = _nonincreasing(reset_values)
        reset_nondecreasing = [
            reset_values[1] >= reset_values[0],
            reset_values[2] >= reset_values[1],
        ]
        details[quantity] = {
            "contract_e_error": contract_values,
            "no_reset_error": no_reset_values,
            "paired_reset_effect": reset_values,
            "contract_e_improves": contract_improves,
            "no_reset_improves": no_reset_improves,
            "reset_effect_nonincreasing": reset_nonincreasing,
            "reset_effect_nondecreasing": reset_nondecreasing,
        }
        shared = shared and all(contract_improves) and all(no_reset_improves) and all(reset_nonincreasing)
        reset_specific = (
            reset_specific
            and all(no_reset_improves)
            and not all(contract_improves)
            and any(reset_nondecreasing)
        )
    if shared:
        classification = "shared_finite_N_pattern"
    elif reset_specific:
        classification = "reset_specific_pattern"
    else:
        classification = "mixed_or_nonmonotone_inconclusive"
    return classification, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    nodes: dict[tuple[str, int], dict[str, Any]] = {}
    terminal_reason = None
    for policy in RESET_POLICIES:
        for particle_count in PARTICLE_COUNTS:
            remaining = CAMPAIGN_CAP_SECONDS - (time.monotonic() - started)
            if remaining <= 0.0:
                terminal_reason = "aggregate_budget_exhausted"
                break
            node_dir = output_root / f"{policy}-n{particle_count}"
            node_dir.mkdir()
            result_path = node_dir / "result.json"
            log_path = node_dir / "worker.log"
            command = [
                sys.executable,
                str(WORKER.relative_to(ROOT)),
                "--output",
                str(result_path.relative_to(ROOT)),
                "--ridge",
                RIDGE.hex(),
                "--steps",
                str(STEPS),
                "--num-particles",
                str(particle_count),
                "--reset-policy",
                policy,
            ]
            node_started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "-1",
                        "TF_ENABLE_ONEDNN_OPTS": "0",
                        "MPLCONFIGDIR": "/tmp",
                    },
                    capture_output=True,
                    text=True,
                    timeout=min(NODE_TIMEOUT_SECONDS, max(1, int(remaining))),
                    check=False,
                )
                log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
                exit_code = completed.returncode
                timed_out = False
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
                stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
                log_path.write_text(stdout + stderr, encoding="utf-8")
                exit_code = None
                timed_out = True
            attempt = {
                "policy": policy,
                "particle_count": particle_count,
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "wall_time_seconds": time.monotonic() - node_started,
                "result_path": str(result_path.relative_to(ROOT)),
                "result_exists": result_path.is_file(),
            }
            attempts.append(attempt)
            if not result_path.is_file() or exit_code != 0:
                terminal_reason = f"node_failed:{policy}:N={particle_count}"
                break
            node = _load(result_path)
            if not all(bool(value) for value in node["hard_checks"].values()):
                terminal_reason = f"hard_veto:{policy}:N={particle_count}"
                break
            nodes[(policy, particle_count)] = node
        if terminal_reason is not None:
            break

    errors: dict[str, list[list[float]]] = {policy: [] for policy in RESET_POLICIES}
    reset_effects: list[list[float]] = []
    classification = None
    classification_details = None
    if terminal_reason is None:
        for policy in RESET_POLICIES:
            errors[policy] = [
                _relative_errors(nodes[(policy, particle_count)])
                for particle_count in PARTICLE_COUNTS
            ]
        reset_effects = [
            _paired_reset_effect(
                nodes[("all_active_contract_e", particle_count)],
                nodes[("no_reset_weighted", particle_count)],
            )
            for particle_count in PARTICLE_COUNTS
        ]
        classification, classification_details = _classify(errors, reset_effects)

    payload = {
        "schema_version": "bayesfilter.contract_e_phase8.reset_bias_n_scaling.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "continuation_id": "contract-e-canonical-gradient-migration-continuation-20260714-115526",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_COMPLETE" if terminal_reason is None else "DIAGNOSTIC_STOPPED",
        "terminal_reason": terminal_reason,
        "classification": classification,
        "classification_role": "deterministic_one_seed_description_not_statistical_or_causal",
        "particle_counts": list(PARTICLE_COUNTS),
        "reset_policies": list(RESET_POLICIES),
        "quantity_order": ["value", *PARAMETER_NAMES],
        "relative_errors": errors,
        "paired_reset_effects": reset_effects,
        "classification_details": classification_details,
        "attempts": attempts,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(),
            "cpu_only_intentional": True,
            "dtype": "float64",
            "jit_compile": True,
            "time_steps": 2,
            "dataset_seed": 81100,
            "estimator_seed": 80920,
            "ridge": RIDGE,
            "steps": STEPS,
            "transport_chunk_policy": TRANSPORT_CHUNK_POLICY_ID,
            "node_timeout_seconds": NODE_TIMEOUT_SECONDS,
            "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
            "wall_time_seconds": time.monotonic() - started,
            "worker_sha256": _sha256(WORKER),
            "driver_sha256": _sha256(Path(__file__)),
            "plan_sha256": _sha256(PLAN),
        },
        "nonclaims": [
            "one seed cannot support ranking, equivalence, or a causal mechanism claim",
            "N=128 is diagnostic and not primary shape",
            "not canonical admission, full-box HMC readiness, leaderboard release, or program completion",
        ],
    }
    _write_exclusive(output_root / "result.json", payload)
    print(json.dumps({"output": str(output_root / "result.json"), "status": payload["status"], "classification": classification, "terminal_reason": terminal_reason}, sort_keys=True))
    if terminal_reason is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Supervise the repaired Phase 8 T=2, N=32 lower-rung graph."""

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
WORKER = ROOT / "docs/benchmarks/run_contract_e_phase8_lower_rung_node.py"
PLAN = ROOT / "docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-lower-rung-continuation-subplan-2026-07-14.md"
RIDGE_SCALE = 0.35**2
RIDGE_EXPONENTS = (-24, -20, -16, -12, -8, -4, 0, 4, 8)
STEP_COUNTS = (10, 20, 40, 80)
NODE_TIMEOUT_SECONDS = 300
CAMPAIGN_CAP_SECONDS = 7200
MAX_NODE_ATTEMPTS = 14
DELTA_VALUE = 0.001
DELTA_GRAD = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configuration_key(*, ridge: float, steps: int, row_chunk: int, col_chunk: int) -> str:
    payload = json.dumps(
        {
            "ridge_hex": ridge.hex(),
            "steps": steps,
            "row_chunk": row_chunk,
            "col_chunk": col_chunk,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _edge(
    parent: dict[str, Any],
    child: dict[str, Any],
    *,
    require_child_residual_no_larger: bool,
) -> dict[str, Any]:
    kalman_value = float(parent["kalman"]["objective"])
    if child["kalman"] != parent["kalman"]:
        return {"pass": False, "reason": "kalman_identity_mismatch"}
    value_drift = abs(float(child["center"]["objective"]) - float(parent["center"]["objective"])) / abs(kalman_value)
    kalman_gradient = [float(item) for item in parent["kalman"]["hmc_score"]]
    parent_gradient = [float(item) for item in parent["center"]["hmc_score"]]
    child_gradient = [float(item) for item in child["center"]["hmc_score"]]
    gradient_drift = [
        abs(c - p) / abs(g)
        for p, c, g in zip(parent_gradient, child_gradient, kalman_gradient, strict=True)
    ]
    residuals = {}
    residual_direction_pass = True
    for name in (
        "quotient_row_residual_history",
        "quotient_column_residual_history",
    ):
        p = float(parent["telemetry"][name]["max_abs"])
        c = float(child["telemetry"][name]["max_abs"])
        residuals[name] = {
            "parent": p,
            "child": c,
            "child_no_larger": c <= p,
        }
        if require_child_residual_no_larger:
            residual_direction_pass = residual_direction_pass and c <= p
    passed = (
        value_drift <= DELTA_VALUE
        and all(item <= DELTA_GRAD for item in gradient_drift)
        and residual_direction_pass
    )
    return {
        "value_drift_relative_to_abs_kalman": value_drift,
        "value_threshold": DELTA_VALUE,
        "gradient_component_drift_relative_to_abs_kalman_gradient": gradient_drift,
        "gradient_threshold": DELTA_GRAD,
        "residuals": residuals,
        "child_residual_no_larger_required": require_child_residual_no_larger,
        "pass": passed,
    }


def _oracle_gate(node: dict[str, Any]) -> dict[str, Any]:
    kalman_value = float(node["kalman"]["objective"])
    value_relative = abs(float(node["center"]["objective"]) - kalman_value) / abs(kalman_value)
    oracle_gradient = [float(item) for item in node["kalman"]["hmc_score"]]
    candidate_gradient = [float(item) for item in node["center"]["hmc_score"]]
    relative = [
        abs(candidate - oracle) / abs(oracle)
        for candidate, oracle in zip(candidate_gradient, oracle_gradient, strict=True)
    ]
    sign_reversal = [
        candidate * oracle < 0.0
        for candidate, oracle in zip(candidate_gradient, oracle_gradient, strict=True)
    ]
    return {
        "value_relative_error": value_relative,
        "value_threshold": DELTA_VALUE,
        "gradient_component_relative_error": relative,
        "gradient_threshold": DELTA_GRAD,
        "sign_reversal": sign_reversal,
        "pass": value_relative <= DELTA_VALUE and all(item <= DELTA_GRAD for item in relative) and not any(sign_reversal),
    }


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    executed_node_count = 0

    def run_node(*, ridge: float, steps: int, row_chunk: int, col_chunk: int, stage: str, run_fd: bool = False) -> dict[str, Any] | None:
        nonlocal executed_node_count
        elapsed = time.monotonic() - started
        remaining = CAMPAIGN_CAP_SECONDS - elapsed
        if remaining <= 0.0:
            raise TimeoutError("aggregate continuation budget exhausted")
        key = _configuration_key(ridge=ridge, steps=steps, row_chunk=row_chunk, col_chunk=col_chunk)
        if not run_fd and key in cache:
            attempts.append({"stage": stage, "configuration_key": key, "reused": True, "source_attempt": cache[key]["_attempt_index"]})
            return cache[key]
        if executed_node_count >= MAX_NODE_ATTEMPTS:
            raise RuntimeError("maximum executed node attempts exhausted")
        executed_node_count += 1
        attempt_index = executed_node_count
        node_name = f"node-{attempt_index:02d}-{stage}-{key}{'-fd' if run_fd else ''}"
        node_dir = output_root / node_name
        node_dir.mkdir()
        result_path = node_dir / "result.json"
        log_path = node_dir / "worker.log"
        command = [
            sys.executable,
            str(WORKER.relative_to(ROOT)),
            "--output",
            str(result_path.relative_to(ROOT)),
            "--ridge",
            ridge.hex(),
            "--steps",
            str(steps),
        ]
        if run_fd:
            command.append("--run-fd")
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "MPLCONFIGDIR": "/tmp",
        }
        record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "stage": stage,
            "configuration_key": key,
            "configuration": {"ridge": ridge, "ridge_hex": ridge.hex(), "steps": steps, "row_chunk": row_chunk, "col_chunk": col_chunk, "run_fd": run_fd},
            "command": command,
            "timeout_seconds": min(NODE_TIMEOUT_SECONDS, max(1, int(remaining))),
            "reused": False,
            "result_path": str(result_path.relative_to(ROOT)),
            "log_path": str(log_path.relative_to(ROOT)),
        }
        node_started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=record["timeout_seconds"],
                check=False,
            )
            log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            record["exit_code"] = completed.returncode
            record["timed_out"] = False
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            log_path.write_text(stdout + stderr, encoding="utf-8")
            record["exit_code"] = None
            record["timed_out"] = True
        record["wall_time_seconds"] = time.monotonic() - node_started
        record["result_exists"] = result_path.is_file()
        attempts.append(record)
        if not result_path.is_file():
            return None
        node = _load(result_path)
        node["_attempt_index"] = attempt_index
        node["_configuration_key"] = key
        node["_result_path"] = str(result_path.relative_to(ROOT))
        if not run_fd:
            cache[key] = node
        return node

    terminal_reason = None
    selected_ridge: float | None = None
    selected_steps: int | None = None
    selected_chunks: tuple[int, int] | None = None
    ridge_nodes: list[dict[str, Any]] = []
    step_nodes: dict[int, dict[str, Any]] = {}
    step_edges: list[dict[str, Any]] = []
    chunk_edge: dict[str, Any] | None = None
    final_node: dict[str, Any] | None = None
    try:
        for exponent in RIDGE_EXPONENTS:
            ridge = RIDGE_SCALE * (2.0**exponent)
            node = run_node(ridge=ridge, steps=80, row_chunk=8, col_chunk=8, stage=f"ridge-k{exponent:+d}")
            if node is None:
                terminal_reason = "ridge_node_missing_artifact"
                break
            ridge_nodes.append(node)
            if all(bool(value) for value in node["hard_checks"].values()):
                selected_ridge = ridge
                break
        if terminal_reason is None and selected_ridge is None:
            terminal_reason = "no_hard_valid_ridge"
        if terminal_reason is None:
            for steps in STEP_COUNTS:
                node = run_node(ridge=selected_ridge, steps=steps, row_chunk=8, col_chunk=8, stage=f"steps-{steps}")
                if node is None:
                    terminal_reason = "step_node_missing_artifact"
                    break
                step_nodes[steps] = node
        if terminal_reason is None:
            for parent_steps, child_steps in zip(
                STEP_COUNTS[:-1], STEP_COUNTS[1:], strict=True
            ):
                edge = _edge(
                    step_nodes[parent_steps],
                    step_nodes[child_steps],
                    require_child_residual_no_larger=True,
                )
                edge.update({"parent_steps": parent_steps, "child_steps": child_steps})
                step_edges.append(edge)
            for edge in step_edges:
                if edge["pass"]:
                    selected_steps = int(edge["child_steps"])
                    break
            if selected_steps is None:
                terminal_reason = "no_stable_step_edge"
        if terminal_reason is None:
            coarse = run_node(ridge=selected_ridge, steps=selected_steps, row_chunk=16, col_chunk=16, stage="chunks-16")
            fine = run_node(ridge=selected_ridge, steps=selected_steps, row_chunk=8, col_chunk=8, stage="chunks-8")
            if coarse is None or fine is None:
                terminal_reason = "chunk_node_missing_artifact"
            else:
                chunk_edge = _edge(
                    coarse,
                    fine,
                    require_child_residual_no_larger=False,
                )
                chunk_edge.update({"parent_chunks": [16, 16], "child_chunks": [8, 8]})
                if chunk_edge["pass"]:
                    selected_chunks = (16, 16)
                else:
                    terminal_reason = "chunk_edge_failed"
        if terminal_reason is None:
            final_node = run_node(
                ridge=selected_ridge,
                steps=selected_steps,
                row_chunk=selected_chunks[0],
                col_chunk=selected_chunks[1],
                stage="final-verification",
                run_fd=True,
            )
            if final_node is None:
                terminal_reason = "final_node_missing_artifact"
            elif not final_node["fd"]["all_endpoint_checks_valid"] or not final_node["fd"]["all_relative_errors_pass"]:
                terminal_reason = "final_fd_failed_or_inconclusive"
            elif not _oracle_gate(final_node)["pass"]:
                terminal_reason = "lower_rung_kalman_gate_failed"
    except (RuntimeError, TimeoutError) as error:
        terminal_reason = f"{type(error).__name__}:{error}"

    passed = terminal_reason is None and final_node is not None
    payload = {
        "schema_version": "bayesfilter.contract_e_phase8.lower_rung_ladder.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "continuation_id": "contract-e-canonical-gradient-migration-continuation-20260714-115526",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOWER_RUNG_DIAGNOSTIC_PASSED_FACTORY_EMPTY" if passed else "LOWER_RUNG_STOPPED",
        "terminal_reason": terminal_reason,
        "selection": {"ridge": selected_ridge, "steps": selected_steps, "chunks": list(selected_chunks) if selected_chunks else None},
        "attempts": attempts,
        "attempt_count_including_reuse_records": len(attempts),
        "executed_node_count": executed_node_count,
        "maximum_node_attempts": MAX_NODE_ATTEMPTS,
        "ridge_nodes": [node["_result_path"] for node in ridge_nodes],
        "step_edges": step_edges,
        "chunk_edge": chunk_edge,
        "final_result": final_node["_result_path"] if final_node else None,
        "final_oracle_gate": _oracle_gate(final_node) if final_node else None,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(),
            "cpu_only_intentional": True,
            "cuda_visible_devices": "-1",
            "dtype": "float64",
            "jit_compile": True,
            "dataset_seed": 81100,
            "estimator_seed": 80920,
            "time_steps": 2,
            "num_particles": 32,
            "delta_grad": DELTA_GRAD,
            "audit_count_future_only": 16,
            "node_timeout_seconds": NODE_TIMEOUT_SECONDS,
            "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(output_root.relative_to(ROOT)),
            "plan": str(PLAN.relative_to(ROOT)),
            "worker_sha256": _sha256(WORKER),
            "driver_sha256": _sha256(Path(__file__)),
            "plan_sha256": _sha256(PLAN),
        },
        "decision": {
            "engineering": "diagnostic lower-rung graph completed" if passed else "candidate or harness stopped",
            "numerical": "selected tuple passed frozen lower-rung gates" if passed else "no promotable lower-rung tuple",
            "scientific": "not established; factory empty and primary-shape audit not run",
        },
        "nonclaims": [
            "factory is empty; no canonical admission or leaderboard contribution",
            "not full-box HMC readiness or primary-shape T=50,N=10000 evidence",
            "audit count 16 is future exploratory design with no power claim",
            "not nonlinear validity, superiority, release, or program completion",
        ],
    }
    _write_exclusive(output_root / "ladder-result.json", payload)
    print(json.dumps({"output": str(output_root / "ladder-result.json"), "status": payload["status"], "terminal_reason": terminal_reason}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run eight fresh width-128 confirmations, four sequentially on each GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "docs/benchmarks/run_defensive_weighted_neutra_analytic_2026_08_11.py"
PLAN = ROOT / "docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    lock = threading.Lock()
    receipts: list[Mapping[str, Any]] = []

    assignments = {"0": (1, 2, 3, 4), "1": (5, 6, 7, 8)}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_lane,
                gpu,
                replications,
                output_root,
                args.python.resolve(),
                lock,
                receipts,
            )
            for gpu, replications in assignments.items()
        ]
        for future in futures:
            future.result()

    receipts.sort(key=lambda row: int(row["replication"]))
    if [int(row["replication"]) for row in receipts] != list(range(1, 9)):
        raise RuntimeError("confirmation did not produce exactly replications 1 through 8")
    manifest = {
        "schema": "bayesfilter.defensive_weighted_neutra_width128_confirmation.v1",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "command": " ".join(sys.argv),
        "python": str(args.python.resolve()),
        "plan": str(PLAN.relative_to(ROOT)),
        "assignments": {key: list(value) for key, value in assignments.items()},
        "started_at_utc": started_at.isoformat(),
        "wall_time_seconds": time.monotonic() - started,
        "receipts": receipts,
        "stream_policy": "one_child_per_gpu_lane_with_terminal_log_and_hash_validation",
    }
    _write(output_root / "campaign_manifest.json", manifest)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.defensive_weighted_neutra_width128_confirmation_hashes.v1",
            "artifacts": {
                "campaign_manifest.json": _sha256(output_root / "campaign_manifest.json")
            },
        },
    )
    print(
        json.dumps(
            {
                "completed": True,
                "replications": len(receipts),
                "wall_time_seconds": manifest["wall_time_seconds"],
                "output_root": str(output_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run_lane(
    gpu: str,
    replications: tuple[int, ...],
    campaign_root: Path,
    python: Path,
    lock: threading.Lock,
    receipts: list[Mapping[str, Any]],
) -> None:
    for replication in replications:
        run_root = campaign_root.parent / (
            f"capacity-depth6-width128-updates10000-confirmation-{replication}-v1"
        )
        if run_root.exists():
            raise FileExistsError(f"run output root must be fresh: {run_root}")
        log_path = campaign_root / f"replication-{replication}.log"
        command = (
            str(python),
            str(RUNNER),
            "--mode",
            "two-mode-canary",
            "--output-root",
            str(run_root),
            "--updates",
            "10000",
            "--batch-size",
            "4096",
            "--audit-size",
            "65536",
            "--checkpoint-every",
            "250",
            "--replication",
            str(replication),
            "--hidden-width",
            "128",
            "--stages",
            "6",
        )
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            }
        )
        started = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"replication {replication} on GPU {gpu} failed with "
                f"exit code {process.returncode}; inspect {log_path}"
            )
        receipt = _validate_run(run_root, replication, gpu)
        receipt = {
            **receipt,
            "supervisor_wall_time_seconds": time.monotonic() - started,
            "log": str(log_path),
        }
        with lock:
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "gpu": gpu,
                        "replication": replication,
                        "validated": True,
                        "wall_time_seconds": receipt["supervisor_wall_time_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def _validate_run(run_root: Path, replication: int, gpu: str) -> Mapping[str, Any]:
    result_path = run_root / "result.json"
    manifest_path = run_root / "run_manifest.json"
    hashes_path = run_root / "artifact_hashes.json"
    state_path = run_root / "trainer_states.json"
    for path in (result_path, manifest_path, hashes_path, state_path):
        if not path.is_file():
            raise RuntimeError(f"replication {replication} missing artifact: {path}")
    hashes = _read(hashes_path)["artifacts"]
    for path in (result_path, manifest_path, state_path):
        if hashes.get(path.name) != _sha256(path):
            raise RuntimeError(f"replication {replication} hash mismatch: {path}")
    result = _read(result_path)
    manifest = _read(manifest_path)
    if int(result["replication"]) != replication:
        raise RuntimeError(f"replication identity mismatch: {result_path}")
    if result["config"]["hidden_layers"] != [128, 128]:
        raise RuntimeError(f"hidden-layer identity mismatch: {result_path}")
    if int(result["config"]["stages"]) != 6:
        raise RuntimeError(f"stage identity mismatch: {result_path}")
    coverage = result["audit"]["base_component_coverage"]["weighted"]
    if not bool(coverage["all_finite"] and coverage["both_components_observed"]):
        raise RuntimeError(f"weighted coverage veto: {result_path}")
    if manifest["cuda_visible_devices"] != gpu:
        raise RuntimeError(f"GPU identity mismatch: {manifest_path}")
    if not bool(manifest["jit_compile"]):
        raise RuntimeError(f"XLA identity mismatch: {manifest_path}")
    if not bool(
        manifest["gpu_memory_policy"]["all_physical_devices_memory_growth"]
    ):
        raise RuntimeError(f"memory-growth identity mismatch: {manifest_path}")
    if bool(manifest["sample_wise_loop_or_scalar_fallback"]):
        raise RuntimeError(f"scalar fallback veto: {manifest_path}")
    return {
        "replication": replication,
        "gpu": gpu,
        "run_root": str(run_root),
        "result_sha256": _sha256(result_path),
        "minority_probability": coverage[
            "soft_responsibility_component_probabilities"
        ][1],
        "weighted_audit_nll": result["audit"]["weighted"]["weighted_nll"],
        "selected_update": result["checkpoint_selection"]["weighted_update"],
        "run_wall_time_seconds": manifest["wall_time_seconds"],
    }


def _read(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

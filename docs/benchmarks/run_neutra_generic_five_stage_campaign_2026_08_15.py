#!/usr/bin/env python3
"""Run the reviewed generic five-stage known-law model campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-neutra-generic-five-stage-training-plan-2026-08-15.md"
RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-generic-five-stage-training-2026-08-15"

TARGET_BUDGETS = {
    "funnel": {
        "affine": 250,
        "simple": 2000,
        "progressive": 500,
        "joint": 1250,
        "cold": 5000,
    },
    "gaussian": {
        "affine": 100,
        "simple": 300,
        "progressive": 100,
        "joint": 300,
        "cold": 1000,
    },
    "banana": {
        "affine": 100,
        "simple": 300,
        "progressive": 100,
        "joint": 300,
        "cold": 1000,
    },
    "mixture": {
        "affine": 100,
        "simple": 300,
        "progressive": 100,
        "joint": 300,
        "cold": 1000,
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _command(
    args: argparse.Namespace,
    target: str,
    route: str,
    seed_index: int,
    output: Path,
) -> list[str]:
    budget = TARGET_BUDGETS[target]
    return [
        str(args.python),
        str(RUNNER),
        "--output-root",
        str(output),
        "--target",
        target,
        "--route",
        route,
        "--device",
        str(args.device),
        "--seed-index",
        str(seed_index),
        "--batch-size",
        "4096",
        "--learning-rates",
        "2e-4,5e-4,1e-3",
        "--affine-updates",
        str(budget["affine"]),
        "--simple-updates",
        str(budget["simple"]),
        "--progressive-updates",
        str(budget["progressive"]),
        "--joint-updates",
        str(budget["joint"]),
        "--cold-updates",
        str(budget["cold"]),
        "--checkpoint-every",
        "250" if target == "funnel" else "100",
        "--proposal-audit-count",
        "131072",
    ]


def _run(command: list[str], output: Path, resume: bool) -> Mapping[str, Any]:
    result_path = output / "result.json"
    if resume and result_path.is_file():
        return _load(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true"},
        text=True,
        capture_output=True,
        check=False,
    )
    (output.parent / f"{output.name}.process.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"campaign cell failed with exit {completed.returncode}: {' '.join(command)}"
        )
    return _load(result_path)


def _row(target: str, route: str, seed_index: int, output: Path, result: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = result["validation"]
    return {
        "target": target,
        "route": route,
        "seed_index": seed_index,
        "passed": bool(result["decision"]["known_law_gate_passed"]),
        "importance_ess_fraction": float(validation["importance_ess_fraction"]),
        "log_target_to_proposal_ratio_stddev": float(
            validation["log_target_to_proposal_ratio_stddev"]
        ),
        "wall_seconds": float(result["wall_seconds"]),
        "output_root": str(output),
        "result_sha256": _sha256(output / "result.json"),
    }


def main() -> int:
    args = _args()
    if not PLAN.is_file() or not RUNNER.is_file():
        raise FileNotFoundError("reviewed plan or model runner is missing")
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output root must be fresh unless --resume is used: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []

    for target in TARGET_BUDGETS:
        first_seed = []
        for route in ("staged", "cold"):
            output = output_root / target / route / "seed-0"
            command = _command(args, target, route, 0, output)
            result = _run(command, output, bool(args.resume))
            row = _row(target, route, 0, output, result)
            rows.append(row)
            first_seed.append(row)
            _write(
                output_root / "progress.json",
                {
                    "schema": "bayesfilter.neutra.generic_five_stage_progress.v1",
                    "completed_cells": len(rows),
                    "rows": rows,
                },
            )
        if any(row["passed"] for row in first_seed):
            for route in ("staged", "cold"):
                output = output_root / target / route / "seed-1"
                command = _command(args, target, route, 1, output)
                result = _run(command, output, bool(args.resume))
                rows.append(_row(target, route, 1, output, result))
                _write(
                    output_root / "progress.json",
                    {
                        "schema": "bayesfilter.neutra.generic_five_stage_progress.v1",
                        "completed_cells": len(rows),
                        "rows": rows,
                    },
                )

    result = {
        "schema": "bayesfilter.neutra.generic_five_stage_campaign.v1",
        "plan": str(PLAN),
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "target_budgets": TARGET_BUDGETS,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "inference_status": {
            "hard_veto_screen": "see per-cell results",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "model-specific replication and downstream HMC only after known-law passage",
        },
    }
    _write(output_root / "campaign_result.json", result)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.generic_five_stage_campaign_hashes.v1",
            "artifacts": {
                path.relative_to(output_root).as_posix(): _sha256(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"output_root": str(output_root), "wall_seconds": result["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the reviewed Gaussian/banana adaptive five-stage repair campaign."""

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
PLAN = ROOT / "docs/plans/bayesfilter-neutra-generic-adaptive-five-stage-repair-plan-2026-08-15.md"
RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_adaptive_five_stage_model_2026_08_15.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-generic-adaptive-five-stage-repair-2026-08-15"
TARGETS = ("gaussian", "banana")
ROUTES = ("adaptive_reset", "adaptive_carry", "cold")
SEEDS = (0, 1)
CAMPAIGN_TIMEOUT_SECONDS = 2700.0


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


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(
    args: argparse.Namespace,
    target: str,
    route: str,
    seed_index: int,
    output: Path,
) -> list[str]:
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
        "100",
        "--simple-updates",
        "300",
        "--progressive-updates",
        "100",
        "--joint-updates",
        "2300",
        "--cold-updates",
        "3000",
        "--checkpoint-every",
        "100",
        "--adaptive-minimum-updates",
        "400",
        "--adaptive-patience-checkpoints",
        "4",
        "--adaptive-minimum-improvement",
        "1e-5",
        "--adaptive-lr-reduction-factor",
        "0.5",
        "--adaptive-maximum-lr-reductions",
        "3",
        "--proposal-audit-count",
        "131072",
    ]


def _row(
    target: str,
    route: str,
    seed_index: int,
    output: Path,
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    validation = result["validation"]
    training = result["training"]
    row = {
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
    if route.startswith("adaptive_"):
        joint = training["stages"][-1]
        selected_rate = float(joint["selected_learning_rate"])
        selected_candidate = next(
            candidate
            for candidate in joint["candidates"]
            if float(candidate["learning_rate"]) == selected_rate
        )
        row.update(
            {
                "selected_path_updates": int(training["selected_path_updates"]),
                "tuning_optimizer_updates": int(training["tuning_optimizer_updates"]),
                "joint_selected_update": int(joint["selected_update"]),
                "joint_executed_updates": int(selected_candidate["executed_updates"]),
                "joint_lr_reductions": int(selected_candidate["learning_rate_reductions"]),
                "joint_stop_reason": str(selected_candidate["stop_reason"]),
            }
        )
    else:
        row.update(
            {
                "selected_path_updates": int(training["selected"]["selected_update"]),
                "tuning_optimizer_updates": 3 * 3000,
                "joint_selected_update": None,
                "joint_executed_updates": None,
                "joint_lr_reductions": None,
                "joint_stop_reason": None,
            }
        )
    return row


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

    for target in TARGETS:
        for seed_index in SEEDS:
            for route in ROUTES:
                output = output_root / target / route / f"seed-{seed_index}"
                result_path = output / "result.json"
                if args.resume and result_path.is_file():
                    result = _load(result_path)
                else:
                    remaining = CAMPAIGN_TIMEOUT_SECONDS - (time.perf_counter() - started)
                    if remaining <= 0.0:
                        raise TimeoutError("campaign wall-time cap exhausted")
                    command = _command(args, target, route, seed_index, output)
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env={**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true"},
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=remaining,
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    (output.parent / f"{output.name}.process.log").write_text(
                        completed.stdout + completed.stderr, encoding="utf-8"
                    )
                    if completed.returncode != 0 or not result_path.is_file():
                        raise RuntimeError(
                            f"campaign cell failed with exit {completed.returncode}: "
                            + " ".join(command)
                        )
                    result = _load(result_path)
                rows.append(_row(target, route, seed_index, output, result))
                _write(
                    output_root / "progress.json",
                    {
                        "schema": "bayesfilter.neutra.generic_adaptive_five_stage_progress.v1",
                        "completed_cells": len(rows),
                        "maximum_cells": len(TARGETS) * len(ROUTES) * len(SEEDS),
                        "rows": rows,
                    },
                )

    result = {
        "schema": "bayesfilter.neutra.generic_adaptive_five_stage_campaign.v1",
        "plan": str(PLAN),
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "targets": TARGETS,
        "routes": ROUTES,
        "seeds": SEEDS,
        "matched_selected_path_ceiling": 3000,
        "campaign_timeout_seconds": CAMPAIGN_TIMEOUT_SECONDS,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "inference_status": {
            "hard_veto_screen": "see per-cell known-law results",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "target-specific interpretation after all cells complete",
        },
    }
    _write(output_root / "campaign_result.json", result)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.generic_adaptive_five_stage_campaign_hashes.v1",
            "artifacts": {
                path.relative_to(output_root).as_posix(): _sha256(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {"output_root": str(output_root), "wall_seconds": result["wall_seconds"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

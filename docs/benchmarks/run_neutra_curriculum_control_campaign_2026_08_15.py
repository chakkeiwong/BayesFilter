#!/usr/bin/env python3
"""Run the reviewed Gaussian and banana NeuTra curriculum control campaign."""

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
PLAN = ROOT / "docs/plans/bayesfilter-neutra-curriculum-control-campaign-plan-2026-08-15.md"
RUNNER = ROOT / "docs/benchmarks/run_neutra_curriculum_control_target_2026_08_15.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-curriculum-control-campaign-2026-08-15"
TARGETS = ("gaussian", "banana")
TIMEOUT_SECONDS = 3600.0


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = _args()
    if not PLAN.is_file() or not RUNNER.is_file():
        raise FileNotFoundError("reviewed plan or target runner is missing")
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output root must be fresh unless --resume is used: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for target in TARGETS:
        target_root = output_root / target
        result_path = target_root / "result.json"
        if args.resume and result_path.is_file():
            result = _load(result_path)
        else:
            remaining = TIMEOUT_SECONDS - (time.perf_counter() - started)
            if remaining <= 0.0:
                raise TimeoutError("curriculum campaign time cap exhausted")
            command = [
                str(args.python), str(RUNNER), "--output-root", str(target_root),
                "--target", target, "--device", str(args.device),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true"},
                text=True,
                capture_output=True,
                check=False,
                timeout=remaining,
            )
            (output_root / f"{target}.process.log").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"target campaign failed with exit {completed.returncode}: " + " ".join(command)
                )
            result = _load(result_path)
        rows.append(
            {
                "target": target,
                "selected_name": result["tournament"]["selection"]["selected_name"],
                "selected_sequence": result["tournament"]["selection"]["selected_sequence"],
                "uncertainty_set": result["tournament"]["selection"]["uncertainty_set"],
                "selected_passed_both_seeds": result["final"]["selected_protocol_passed_both_seeds"],
                "cold_comparator_name": result["tournament"]["cold_comparator_name"],
                "cold_passed_both_seeds": result["final"]["cold_comparator_passed_both_seeds"],
                "probe_calls": result["search"]["probe_calls"],
                "wall_seconds": result["wall_seconds"],
                "result_sha256": _sha256(result_path),
            }
        )
        _write(
            output_root / "progress.json",
            {
                "schema": "bayesfilter.neutra.curriculum_control_progress.v1",
                "completed_targets": len(rows),
                "maximum_targets": len(TARGETS),
                "rows": rows,
            },
        )
    campaign = {
        "schema": "bayesfilter.neutra.curriculum_control_campaign.v1",
        "plan": str(PLAN),
        "targets": TARGETS,
        "time_cap_seconds": TIMEOUT_SECONDS,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "inference_status": {
            "hard_veto_screen": "see target fresh-final exact-law results",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "interpret control performance before any SSL-LSTM adapter work",
        },
    }
    _write(output_root / "campaign_result.json", campaign)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.curriculum_control_campaign_hashes.v1",
            "artifacts": {
                path.relative_to(output_root).as_posix(): _sha256(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"output_root": str(output_root), "wall_seconds": campaign["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

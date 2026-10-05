"""Diagnostic CLI for the experimental finite-trial acceptance repair.

Preflight uses only the standard library and protocol definitions. It evaluates
no target, initializes no numerical framework, and grants no tuning authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from bayesfilter.inference.hmc_acceptance_protocol import (
    HMCReplicatedAcceptancePolicy, batch_information_preflight,
    replicated_evidence_preflight,
)


CONFIG_SCHEMA = "bayesfilter.acceptance_decision_preflight_config.v1"


def preflight(configuration):
    """Check a frozen design without claiming that the observations will suffice."""
    allowed = {"schema", "policy", "evidence_rungs", "candidate_cap",
               "leapfrog_steps", "max_gradient_work", "pilot_enabled", "legacy_batch"}
    if (not isinstance(configuration, dict) or set(configuration) - allowed
            or configuration.get("schema") != CONFIG_SCHEMA):
        raise ValueError("unsupported preflight configuration")
    policy = HMCReplicatedAcceptancePolicy.from_payload(configuration["policy"])
    result = dict(replicated_evidence_preflight(policy,
        evidence_rungs=tuple(configuration["evidence_rungs"]),
        candidate_cap=configuration["candidate_cap"],
        leapfrog_steps=tuple(configuration["leapfrog_steps"]),
        max_gradient_work=configuration.get("max_gradient_work"),
        pilot_enabled=configuration.get("pilot_enabled", False)))
    if "legacy_batch" in configuration:
        result["legacy_batch"] = batch_information_preflight(**configuration["legacy_batch"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("preflight", help="Check a design without sampling")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    calibration = subparsers.add_parser("calibrate", help="Known-law diagnostic through the shared controller")
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    models = subparsers.add_parser("models", help="One frozen public-tuner numerical model diagnostic")
    models.add_argument("--config", type=Path, required=True)
    models.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    configuration = json.loads(args.config.read_text())
    if args.command in {"calibrate", "models"} and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("controlled calibration is a CPU diagnostic; hide GPU devices before import")
    result = preflight(configuration) if args.command == "preflight" else None
    # Capture only environment settings relevant to the diagnostic; never dump
    # the launch environment into artifacts or exception messages.
    env = {key: os.environ[key] for key in ("PATH",) if key in os.environ}
    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, env=env,
                              cwd=source_root.parent).stdout.strip()
    sources = (Path(__file__), source_root / "inference/hmc_acceptance_protocol.py")
    if args.command in {"calibrate", "models"}:
        sources += (source_root / "testing/acceptance_decision_calibration.py", *(source_root / "inference" / (name+".py")
            for name in ("hmc_acceptance_statistics", "hmc_candidate_set_tuning", "hmc_candidate_decisions", "hmc_candidate_proposals")))
    if args.command == "models":
        sources += (source_root / "testing/acceptance_decision_models.py",)
    manifest = {"schema": "bayesfilter.acceptance_decision_diagnostic_manifest.v1",
        "git_commit": revision,
        "command": [sys.executable, "-m", "bayesfilter.testing.acceptance_decision_validation",
                    *(sys.argv[1:] if argv is None else argv)],
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform()},
        "device": "CPU framework-free structural diagnostic; no GPU or target invoked",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seeds": "N/A: no sampling", "data": "N/A: no target data",
        "configuration": configuration,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "plan": "docs/plans/bayesfilter-hmc-acceptance-decision-repair-plan-2026-10-02.md",
        "started_utc": started_utc,
        "wall_seconds": time.monotonic() - started,
        "result": str(args.output / "preflight.json"), "artifact_authority": False}
    if args.command in {"calibrate", "models"}:
        manifest.update(device="CPU diagnostic; GPU devices intentionally hidden", seeds=configuration.get("seed_namespace",configuration.get("seed")),
            data=configuration.get("data", "known-law bounded trial vectors; exact means in the frozen case definitions"),
            result=str(args.output/"result.json"), jit_compile=True)
        if args.command == "calibrate":
            from .acceptance_decision_calibration import run_calibration
            result = run_calibration(configuration, args.output, manifest=manifest)
        else:
            from .acceptance_decision_models import run_model
            result = run_model(configuration, args.output, manifest=manifest)
        manifest.update(wall_seconds=time.monotonic()-started, budget_stopped=result.get("budget_stopped",False))
        (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"completed_cells": len(result["cells"]) if "cells" in result else 1, "budget_stopped": manifest["budget_stopped"],
                          "result": manifest["result"]}))
        return 2 if manifest["budget_stopped"] or result.get("expectation_met") is False else 0
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("preflight.json", result), ("manifest.json", manifest)):
        (args.output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"first_cohort_fundable": result["first_cohort_fundable"],
                      "result": manifest["result"], "precision_sufficiency": result["precision_sufficiency"]}))
    return 0 if result["first_cohort_fundable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

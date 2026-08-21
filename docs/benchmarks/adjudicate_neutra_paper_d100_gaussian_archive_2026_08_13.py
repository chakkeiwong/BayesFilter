#!/usr/bin/env python3
"""Apply the corrected Gaussian analytic screen to a frozen d100 HMC archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-paper-d100-fresh-baseline-plan-2026-08-13.md"
)
HMC_RUNNER = ROOT / "docs/benchmarks/run_neutra_paper_d100_hmc_2026_08_13.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hmc-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--gaussian-constants", type=Path, required=True)
    parser.add_argument("--interval-level", type=float, default=0.99)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_hmc_runner() -> Any:
    spec = importlib.util.spec_from_file_location("paper_d100_hmc_runner", HMC_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load HMC runner: {HMC_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_hmc_hashes(hmc_root: Path) -> Mapping[str, str]:
    hashes_path = hmc_root / "artifact_hashes.json"
    hashes = _load_json(hashes_path)
    artifacts = hashes.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError("HMC artifact hash ledger is missing or empty")
    verified: dict[str, str] = {}
    for relative, expected in artifacts.items():
        path = hmc_root / str(relative)
        if not path.is_file():
            raise RuntimeError(f"HMC artifact is missing: {relative}")
        actual = _sha256(path)
        if actual != str(expected):
            raise RuntimeError(f"HMC artifact hash mismatch: {relative}")
        verified[str(relative)] = actual
    retained = sorted(key for key in verified if key.endswith("-samples.tftensor") and "/retained/" in f"/{key}")
    if not retained:
        raise RuntimeError("verified HMC ledger contains no retained samples")
    return verified


def main() -> int:
    args = _parse_args()
    output = args.output_root.resolve()
    hmc_root = args.hmc_root.resolve()
    training_root = args.training_root.resolve()
    constants_path = args.gaussian_constants.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if float(args.interval_level) not in (0.99, 0.999):
        raise ValueError("only interval levels 0.99 and 0.999 are reviewed")
    required = (
        PLAN,
        HMC_RUNNER,
        hmc_root / "result.json",
        hmc_root / "run_manifest.json",
        hmc_root / "artifact_hashes.json",
        training_root / "trainer_state.json",
        constants_path,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Gaussian analytic adjudication inputs are missing")

    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.perf_counter()
    verified = _verify_hmc_hashes(hmc_root)
    prior = _load_json(hmc_root / "result.json")
    prior_manifest = _load_json(hmc_root / "run_manifest.json")
    prior_decision = prior.get("decision")
    sequential = prior.get("sequential")
    historical_sampler_statuses = {
        "candidate_sampler_and_analytic_passed",
        "candidate_sampler_passed_analytic_rejected",
    }
    sampler_passed = bool(
        isinstance(prior_decision, Mapping)
        and isinstance(sequential, Mapping)
        and sequential.get("passed") is True
        and prior_decision.get("status") in historical_sampler_statuses
    )
    if not sampler_passed:
        raise RuntimeError("source HMC result did not pass its sampler gates")
    if prior_manifest.get("target", {}).get("name") != "paper_ill_cond_gaussian":
        raise RuntimeError("source HMC target is not the paper Gaussian")

    import tensorflow as tf

    from bayesfilter.inference.neutra_paper_d100_target import load_paper_gaussian_spec

    runner = _load_hmc_runner()
    target = load_paper_gaussian_spec(constants_path)
    objective = str(prior_manifest.get("objective", ""))
    transport, _config, frozen = runner._load_frozen_transport(
        tf, training_root, target.name, objective
    )
    retained = runner._load_retained(tf, hmc_root / "archive", target.dimension)
    flat = tf.reshape(retained, (-1, target.dimension))
    physical = tf.reshape(transport.forward_batch(flat), tf.shape(retained))
    analytic = runner._gaussian_diagnostics(tf, target, physical, args.interval_level)
    screen = analytic.get("structural_screen")
    if not isinstance(screen, Mapping):
        raise RuntimeError("corrected Gaussian structural screen is unavailable")
    analytic_passed = bool(screen.get("all_individual_intervals_contain_exact"))

    manifest = {
        "schema": "bayesfilter.neutra.paper_d100_gaussian_adjudication_manifest.v1",
        "plan": PLAN.as_posix(),
        "source_hmc_root": hmc_root.as_posix(),
        "source_hmc_result_sha256": _sha256(hmc_root / "result.json"),
        "source_hmc_manifest_sha256": _sha256(hmc_root / "run_manifest.json"),
        "source_hmc_hash_ledger_sha256": _sha256(hmc_root / "artifact_hashes.json"),
        "verified_source_artifact_count": len(verified),
        "training_root": training_root.as_posix(),
        "training_state_sha256": frozen["state_sha256"],
        "training_state_hash": frozen["state_hash"],
        "gaussian_constants": constants_path.as_posix(),
        "gaussian_constants_sha256": _sha256(constants_path),
        "hmc_runner": HMC_RUNNER.as_posix(),
        "hmc_runner_sha256": _sha256(HMC_RUNNER),
        "objective": objective,
        "interval_level": float(args.interval_level),
        "execution_target": "cpu_diagnostic_only",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "float64",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    result = {
        "schema": "bayesfilter.neutra.paper_d100_gaussian_adjudication_result.v1",
        "manifest": manifest,
        "source_sampler_decision": prior_decision,
        "analytic_diagnostics": analytic,
        "decision": {
            "status": (
                "source_sampler_and_corrected_analytic_passed"
                if analytic_passed
                else "source_sampler_passed_corrected_analytic_rejected"
            ),
            "candidate_passed": analytic_passed,
            "sampler_passed": True,
            "analytic_individual_intervals_passed": analytic_passed,
            "promotion": False,
            "primary_criterion": (
                "source canonical sequential sampler gates plus corrected "
                "predeclared 11-diagnostic Gaussian screen"
            ),
            "objective_ranking": "not_supported",
            "default_promotion": False,
            "nonclaims": [
                "no rerun or modification of HMC draws",
                "no omnibus p-value",
                "no objective superiority",
                "no default promotion",
            ],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write(
        output / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.paper_d100_gaussian_adjudication_hashes.v1",
            "artifacts": artifacts,
        },
    )
    print(
        json.dumps(
            {
                "analytic_passed": analytic_passed,
                "output_root": output.as_posix(),
                "source_sampler_passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

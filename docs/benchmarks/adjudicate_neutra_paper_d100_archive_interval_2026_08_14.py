#!/usr/bin/env python3
"""Re-adjudicate a frozen d100 HMC archive at a reviewed interval level."""

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
PLAN = ROOT / "docs/plans/bayesfilter-weighted-forward-kl-paper-d100-repair-plan-2026-08-14.md"
HMC_RUNNER = ROOT / "docs/benchmarks/run_neutra_paper_d100_hmc_2026_08_13.py"
CHAIN_COUNT = 4


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hmc-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--gaussian-constants", type=Path, required=True)
    parser.add_argument("--target", choices=("paper_funnel", "paper_ill_cond_gaussian"), required=True)
    parser.add_argument("--interval-level", type=float, choices=(0.99, 0.999), required=True)
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
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(v) for v in value]
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
    path.write_text(json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("paper_d100_hmc_runner_interval", HMC_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import d100 HMC runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    hmc_root = args.hmc_root.resolve()
    training_root = args.training_root.resolve()
    constants = args.gaussian_constants.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    required = (PLAN, HMC_RUNNER, constants, hmc_root / "result.json", hmc_root / "run_manifest.json", hmc_root / "artifact_hashes.json", training_root / "trainer_state.json", training_root / "run_manifest.json", training_root / "artifact_hashes.json")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("interval adjudication input is missing")

    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.perf_counter()
    runner = _load_runner()
    source_manifest = _load_json(hmc_root / "run_manifest.json")
    source_result = _load_json(hmc_root / "result.json")
    source_decision = source_result.get("decision")
    if not isinstance(source_decision, Mapping) or source_decision.get("sampler_passed") is not True:
        raise RuntimeError("source HMC archive lacks a sampler-passed decision")
    if source_manifest.get("target", {}).get("name") != args.target:
        raise RuntimeError("source HMC target mismatch")

    import tensorflow as tf

    from bayesfilter.inference.neutra_paper_d100_target import load_paper_gaussian_spec, make_paper_funnel_spec

    target = make_paper_funnel_spec() if args.target == "paper_funnel" else load_paper_gaussian_spec(constants)
    objective = str(source_manifest.get("objective", ""))
    transport, _config, frozen = runner._load_frozen_transport(tf, training_root, target.name, objective)
    retained = runner._load_retained(tf, hmc_root / "archive", target.dimension)
    physical = tf.reshape(transport.forward_batch(tf.reshape(retained, (-1, target.dimension))), tf.shape(retained))
    diagnostics = (
        runner._funnel_diagnostics(tf, target, physical, args.interval_level)
        if args.target == "paper_funnel"
        else runner._gaussian_diagnostics(tf, target, physical, args.interval_level)
    )
    structural = diagnostics["structural_screen"]
    structural_passed = bool(structural["all_individual_intervals_contain_exact"])
    quantile_passed = bool(diagnostics.get("quantile_screen", {}).get("all_individual_intervals_contain_exact_probability", True))
    manifest = {
        "schema": "bayesfilter.neutra.paper_d100_interval_adjudication_manifest.v1",
        "plan": PLAN.as_posix(),
        "target": args.target,
        "objective": objective,
        "interval_level": float(args.interval_level),
        "source_hmc_root": hmc_root.as_posix(),
        "source_hmc_result_sha256": _sha256(hmc_root / "result.json"),
        "source_hmc_manifest_sha256": _sha256(hmc_root / "run_manifest.json"),
        "source_hmc_hashes_sha256": _sha256(hmc_root / "artifact_hashes.json"),
        "training_state_sha256": frozen["state_sha256"],
        "gaussian_constants_sha256": _sha256(constants),
        "execution_target": "cpu_diagnostic_only",
        "cuda_visible_devices": "-1",
        "dtype": "float64",
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    result = {
        "schema": "bayesfilter.neutra.paper_d100_interval_adjudication_result.v1",
        "manifest": manifest,
        "source_sampler_decision": source_decision,
        "analytic_diagnostics": diagnostics,
        "decision": {
            "status": "source_sampler_and_interval_passed" if structural_passed and quantile_passed else "source_sampler_passed_interval_rejected",
            "candidate_passed": bool(structural_passed and quantile_passed),
            "sampler_passed": True,
            "analytic_structural_intervals_passed": structural_passed,
            "analytic_quantile_intervals_passed": quantile_passed,
            "promotion": False,
            "uniform_interval_policy": True,
            "nonclaims": ["no rerun or modification of HMC draws", "no objective ranking", "no default promotion"],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.paper_d100_interval_adjudication_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"candidate_passed": result["decision"]["candidate_passed"], "output_root": output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

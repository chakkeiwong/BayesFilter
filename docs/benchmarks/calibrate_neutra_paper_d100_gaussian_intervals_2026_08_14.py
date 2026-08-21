#!/usr/bin/env python3
"""Calibrate the Gaussian structural screen on iid exact draws."""

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
CONSTANTS = ROOT / "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/source-r1/paper_ill_cond_gaussian_d100_constants.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=32)
    parser.add_argument("--draws-per-replication", type=int, default=4000)
    return parser.parse_args()


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("paper_d100_hmc_runner_calibration", HMC_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import HMC runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if int(args.replications) < 8 or int(args.draws_per_replication) < 1000:
        raise ValueError("calibration requires at least 8 replications and 1000 draws")
    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.perf_counter()
    import tensorflow as tf
    from bayesfilter.inference.neutra_paper_d100_target import load_paper_gaussian_spec, sample_paper_d100_exact

    runner = _load_runner()
    spec = load_paper_gaussian_spec(CONSTANTS)
    passed_99 = []
    passed_999 = []
    failed_names_99 = []
    failed_names_999 = []
    for index in range(int(args.replications)):
        rows = sample_paper_d100_exact(spec, int(args.draws_per_replication), seed=(20260814, 91000 + index))
        physical = tf.reshape(rows, (int(args.draws_per_replication) // 4, 4, spec.dimension))
        d99 = runner._gaussian_diagnostics(tf, spec, physical, 0.99)["structural_screen"]
        d999 = runner._gaussian_diagnostics(tf, spec, physical, 0.999)["structural_screen"]
        p99 = bool(d99["all_individual_intervals_contain_exact"])
        p999 = bool(d999["all_individual_intervals_contain_exact"])
        passed_99.append(p99)
        passed_999.append(p999)
        failed_names_99.append([name for name, ok in zip(d99["names"], d99["individual_interval_contains_exact"]) if not bool(ok)])
        failed_names_999.append([name for name, ok in zip(d999["names"], d999["individual_interval_contains_exact"]) if not bool(ok)])
    manifest = {
        "schema": "bayesfilter.neutra.paper_d100_gaussian_interval_calibration_manifest.v1",
        "plan": PLAN.as_posix(),
        "gaussian_constants_sha256": _sha256(CONSTANTS),
        "replications": int(args.replications),
        "draws_per_replication": int(args.draws_per_replication),
        "seeds": [[20260814, 91000 + i] for i in range(int(args.replications))],
        "execution_target": "cpu_diagnostic_only",
        "cuda_visible_devices": "-1",
        "tensorflow_version": tf.__version__,
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    result = {
        "schema": "bayesfilter.neutra.paper_d100_gaussian_interval_calibration_result.v1",
        "manifest": manifest,
        "known_correct_iid_target": True,
        "pass_rate_99": sum(passed_99) / len(passed_99),
        "pass_rate_999": sum(passed_999) / len(passed_999),
        "rejection_rate_99": 1.0 - sum(passed_99) / len(passed_99),
        "rejection_rate_999": 1.0 - sum(passed_999) / len(passed_999),
        "failed_names_99": failed_names_99,
        "failed_names_999": failed_names_999,
        "interpretation": "iid calibration only; does not validate HMC archives or transport candidates",
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.paper_d100_gaussian_interval_calibration_hashes.v1", "artifacts": {p.name: _sha256(p) for p in output.iterdir() if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"pass_rate_99": result["pass_rate_99"], "pass_rate_999": result["pass_rate_999"], "output_root": output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

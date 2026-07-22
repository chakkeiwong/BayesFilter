#!/usr/bin/env python3
"""Run one P4 predator-prey corrected neural-force HMC cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = "docs/plans/bayesfilter-hnn-surrogate-hmc-p4-predator-prey-subplan-2026-07-17.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=("PP-UKF", "PP-SGQF"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import predator_prey_neural_force_hmc_tf as p4

    context = p4.load_context(args.cell)
    result = p4.run_smoke(context) if args.smoke else p4.run_cell(context, args.output_root)
    result = {
        **result,
        "run_manifest": {
            "schema": "bayesfilter.predator_prey_neural_force_hmc_p4_manifest.v1",
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "environment": "tf-gpu",
            "tensorflow_version": tf.__version__,
            "tensorflow_probability_version": tfp.__version__,
            "device": "/GPU:0",
            "gpu_memory_policy": memory_policy,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "started_at_utc": started_at.isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "plan_file": PLAN,
            "result_file": str(args.output_root / "result.json"),
            "output_root": str(args.output_root),
            "seed_policy": "disjoint training, tuning, warmup, retained domains",
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
    }
    _write(args.output_root / "result.json", result)
    _write(args.output_root / "run_manifest.json", result["run_manifest"])
    _write(
        args.output_root / "artifact_hashes.json",
        {
            "result_sha256": _sha256(args.output_root / "result.json"),
            "run_manifest_sha256": _sha256(args.output_root / "run_manifest.json"),
        },
    )
    print(json.dumps({"cell": args.cell, "passed": result["passed"], "smoke": args.smoke}))
    return 0 if result["passed"] else 1


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

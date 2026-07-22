#!/usr/bin/env python3
"""Run one repaired same-chart HNN versus exact-gradient NeuTra-HMC cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = "docs/plans/bayesfilter-hnn-neutra-native-tuning-correction-plan-2026-07-18.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=("PP-UKF", "PP-SGQF", "SIR-SGQF", "STR-UKF"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--canary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import hnn_neutra_exact_comparison_tf as comparison

    context = comparison.load_context(args.cell)
    if args.canary:
        result = comparison.run_canary(
            context, comparison.load_historical_force(context), output_root
        )
    else:
        result = comparison.run_full_cell(context, output_root)
    manifest = {
        "schema": "bayesfilter.hnn_neutra_exact_comparison_manifest.v1",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
        ).stdout.strip(),
        "command": " ".join(sys.argv),
        "environment": "tf-gpu",
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "device": "/GPU:0",
        "physical_gpus": tuple(str(value) for value in tf.config.list_physical_devices("GPU")),
        "gpu_memory_policy": memory,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "dtype": "float64",
        "started_at_utc": started_at.isoformat(),
        "wall_time_seconds": time.monotonic() - started,
        "plan_file": PLAN,
        "result_file": str(output_root / "result.json"),
        "output_root": str(output_root),
        "cell": args.cell,
        "canary": bool(args.canary),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }
    payload = {**result, "run_manifest": manifest}
    _write(output_root / "result.json", payload)
    _write(output_root / "run_manifest.json", manifest)
    _write(
        output_root / "artifact_hashes.json",
        {
            "result_sha256": _sha256(output_root / "result.json"),
            "run_manifest_sha256": _sha256(output_root / "run_manifest.json"),
        },
    )
    print(json.dumps({"cell": args.cell, "completed": True, "passed": result["passed"], "canary": args.canary}))
    # A scientifically negative candidate is a completed experiment, not an
    # infrastructure failure. Canary failure remains a hard execution failure.
    return 0 if (not args.canary or result["passed"]) else 1


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _ready(item)
            for key, item in value.items()
            if not str(key).startswith("private_")
        }
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

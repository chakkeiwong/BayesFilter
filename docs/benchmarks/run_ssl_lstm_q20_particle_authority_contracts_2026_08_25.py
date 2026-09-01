"""Run Phase 1 particle-authority contract fixtures.

This is a CPU-hidden, XLA reference lane.  It is intentionally independent of
the q=20 target and cannot emit an authority or posterior claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Phase 1 fixtures require CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 1 fixtures require TF_FORCE_GPU_ALLOW_GROWTH=true")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf

from bayesfilter.testing.particle_authority_contracts_tf import run_all_contracts


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def _markdown(receipt: dict[str, object], elapsed: float) -> str:
    lines = [
        "# Phase 1 Contract Fixture Result",
        "",
        f"Status: `{receipt['status']}`",
        "",
        "This is CPU-hidden XLA fixture evidence only. It does not execute the q=20 target or certify an SMC-U authority.",
        "",
        "| Fixture | Status | Role |",
        "|---|---|---|",
    ]
    for name, result in receipt["results"].items():
        assert isinstance(result, dict)
        lines.append(f"| {name} | `{result['status']}` | {result.get('role', '')} |")
    lines.extend(
        [
            "",
            f"Wall time: `{elapsed:.3f} s`",
            "",
            "## Decision",
            "",
            "A passing fixture establishes only the tested finite identity or diagnostic. It does not establish q=20 mode discovery, posterior correctness, IID samples, or NeuTra/HMC readiness.",
            "",
            "## Nonclaims",
            "",
        ]
    )
    for item in receipt["nonclaims"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    start = time.monotonic()
    receipt = dict(run_all_contracts())
    elapsed = time.monotonic() - start
    receipt["manifest"] = {
        "program": "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md",
        "phase": 1,
        "git_commit": _commit(),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
        "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
        "jit_compile": True,
        "wall_seconds": elapsed,
        "seed_domains": {
            "known_mass": [20260825, 101],
            "mutation": [20260825, 202],
            "metadata": [20260825, 303],
        },
    }
    if receipt["manifest"]["physical_gpus"] or receipt["manifest"]["logical_gpus"]:
        raise RuntimeError("CPU-hidden fixture unexpectedly sees a GPU")
    (args.output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (args.output_root / "result.md").write_text(
        _markdown(receipt, elapsed), encoding="ascii"
    )
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

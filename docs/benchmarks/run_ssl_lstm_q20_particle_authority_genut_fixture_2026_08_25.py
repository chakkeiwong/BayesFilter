"""Run a CPU/XLA source-faithful GenUT sigma-point fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("GenUT fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("GenUT fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("GenUT fixture found a visible GPU")

from bayesfilter.testing.particle_authority_genut_tf import generalized_unscented_transform


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_genut_fixture_2026_08_25.py"
MODULE = ROOT / "bayesfilter/testing/particle_authority_genut_tf.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase20-genut-source-fixture-subplan-2026-08-25.md"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise RuntimeError("output root must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    axis = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], tf.float64)
    axis_weights = tf.constant([0.04, 0.22, 0.40, 0.25, 0.09], tf.float64)
    grid = tf.stack(tf.meshgrid(axis, axis, indexing="ij"), axis=-1)
    points = tf.reshape(grid, [25, 2])
    weights = tf.reshape(tf.tensordot(axis_weights, axis_weights, axes=0), [25])
    sigma_points, sigma_weights, diagnostics = generalized_unscented_transform(
        points, weights
    )
    finite = bool(
        tf.reduce_all(tf.math.is_finite(sigma_points)).numpy()
        and tf.reduce_all(tf.math.is_finite(sigma_weights)).numpy()
    )
    hard = {
        "finite": finite,
        "feasible": bool(diagnostics["feasible"].numpy()),
        "mean_residual": float(diagnostics["mean_residual"].numpy()),
        "covariance_residual": float(diagnostics["covariance_residual"].numpy()),
        "third_moment_residual": float(diagnostics["third_moment_residual"].numpy()),
        "fourth_moment_residual": float(diagnostics["fourth_moment_residual"].numpy()),
        "weight_sum_residual": float(
            tf.abs(tf.reduce_sum(sigma_weights) - 1.0).numpy()
        ),
    }
    status = (
        "PASS_SOURCE_FAITHFUL_GENUT_FIXTURE"
        if hard["finite"]
        and hard["feasible"]
        and hard["mean_residual"] <= 1.0e-8
        and hard["covariance_residual"] <= 1.0e-8
        and hard["third_moment_residual"] <= 1.0e-8
        and hard["fourth_moment_residual"] <= 1.0e-8
        and hard["weight_sum_residual"] <= 1.0e-12
        else "GENUT_FIXTURE_FAIL_REPAIR"
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.genut_fixture.v1",
        "status": status,
        "role": "source_faithful_genut_sigma_point_fixture_candidate",
        "source_anchor": "Ebeigbe et al. equations 13-34 and Algorithm 1",
        "hard_gates": hard,
        "diagnostics": diagnostics,
        "fixture": {
            "particle_count": 25,
            "dimension": 2,
            "points": points,
            "weights": weights,
            "sigma_points": sigma_points,
            "sigma_weights": sigma_weights,
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_intentionally_hidden": True,
            "jit_compile": False,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"runner": _sha(RUNNER), "module": _sha(MODULE), "plan": _sha(PLAN)},
        },
        "nonclaims": [
            "sigma points are quadrature points, not IID posterior samples",
            "selected moments do not identify a density or global modes",
            "no q20 authority, posterior correctness, HMC, or default promotion claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 20 Source-Faithful GenUT Fixture\n\n"
        f"Status: `{status}`\n\n"
        "This is a finite sigma-point moment fixture, not a density or q20 "
        "authority result.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": status, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if status == "PASS_SOURCE_FAITHFUL_GENUT_FIXTURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

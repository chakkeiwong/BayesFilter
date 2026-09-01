"""Run a CPU/XLA source-faithful second-order LETF/ETPF fixture."""

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
    raise RuntimeError("ETPF fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("ETPF fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("ETPF fixture found a visible GPU")

from bayesfilter.testing.particle_authority_etpf_tf import second_order_etpf_transform


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_etpf_fixture_2026_08_25.py"
MODULE = ROOT / "bayesfilter/testing/particle_authority_etpf_tf.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase18-etpf-source-fixture-subplan-2026-08-25.md"


class FixtureError(RuntimeError):
    """Raised when the source-faithful fixture cannot be audited."""


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
        raise FixtureError("output root must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise FixtureError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()

    points = tf.constant(
        [[-2.0, -0.5], [-1.4, 0.2], [-0.7, 1.4], [0.0, -1.1],
         [0.5, 0.6], [1.1, 1.7], [1.8, -0.3], [2.4, 0.9]],
        tf.float64,
    )
    weights = tf.nn.softmax(
        tf.constant([1.1, 0.3, -0.4, 0.7, -0.8, 0.2, -0.2, 0.5], tf.float64)
    )
    analysis, diagnostics = second_order_etpf_transform(
        points,
        weights,
        regularization=10.0,
        sinkhorn_steps=400,
        riccati_step=0.1,
        riccati_max_steps=2000,
        riccati_tolerance=1.0e-3,
    )
    finite = bool(tf.reduce_all(tf.math.is_finite(analysis)).numpy()) and all(
        bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())
        for value in diagnostics.values()
        if tf.is_tensor(value) and value.dtype.is_floating
    )
    hard = {
        "finite": finite,
        "riccati_converged": bool(diagnostics["riccati_converged"].numpy()),
        "corrected_column_residual": float(diagnostics["corrected_column_residual"].numpy()),
        "corrected_row_residual": float(diagnostics["corrected_row_residual"].numpy()),
        "mean_residual": float(diagnostics["mean_residual"].numpy()),
        "covariance_residual": float(diagnostics["covariance_residual"].numpy()),
    }
    status = (
        "PASS_SOURCE_FAITHFUL_ETPF_FIXTURE"
        if hard["finite"]
        and hard["riccati_converged"]
        and hard["corrected_column_residual"] <= 2.0e-6
        and hard["corrected_row_residual"] <= 2.0e-6
        and hard["mean_residual"] <= 1.0e-3
        and hard["covariance_residual"] <= 1.0e-3
        else "ETPF_FIXTURE_FAIL_REPAIR"
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.etpf_fixture.v1",
        "status": status,
        "role": "source_faithful_second_order_letf_etpf_fixture_candidate",
        "source_anchor": "Acevedo et al. equations 16,20,26,42-57",
        "hard_gates": hard,
        "diagnostics": diagnostics,
        "fixture": {
            "particle_count": 8,
            "dimension": 2,
            "points": points,
            "weights": weights,
            "analysis": analysis,
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
            "fixture evidence does not establish q20 target fidelity or posterior correctness",
            "second-order transformed rows are not IID samples and may leave the source convex hull",
            "no HMC, mode-discovery, or default promotion claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 18 Source-Faithful ETPF Fixture\n\n"
        f"Status: `{status}`\n\n"
        "This is a small TensorFlow finite-cloud LETF/ETPF fixture. It is not "
        "q=20 particle-authority or posterior evidence.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": status, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if status == "PASS_SOURCE_FAITHFUL_ETPF_FIXTURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

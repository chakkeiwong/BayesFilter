"""Check a symmetric random-walk Metropolis kernel on a known N(0,1) target.

The fixture is a finite implementation check for the v2.9 mutation kernel. It
is not an invariance proof and does not use the q=20 target.
"""

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
    raise RuntimeError("Phase 47 fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 47 fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("Phase 47 fixture found a visible GPU")

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)


class Phase47FixtureError(RuntimeError):
    """Raised when the finite MH fixture cannot be written."""


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(v) for v in value.as_list()]
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase47FixtureError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


@tf.function(jit_compile=True, reduce_retracing=False)
def _mh_fixture(initial: tf.Tensor, noise: tf.Tensor, uniforms: tf.Tensor, sigma: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    proposal = initial + sigma * noise
    current_log = -0.5 * tf.square(initial) - 0.5 * LOG_TWO_PI
    proposal_log = -0.5 * tf.square(proposal) - 0.5 * LOG_TWO_PI
    log_alpha = tf.minimum(tf.constant(0.0, tf.float64), proposal_log - current_log)
    accepted = tf.math.log(uniforms) < log_alpha
    next_state = tf.where(accepted, proposal, initial)
    return next_state, accepted, log_alpha, proposal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--seed", nargs=2, type=int, default=(20260826, 4701))
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise Phase47FixtureError("output root must be repository-relative")
    if int(args.sample_count) < 256 or not float(args.sigma) > 0.0:
        raise Phase47FixtureError("sample-count or sigma is invalid")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase47FixtureError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    split = tf.random.experimental.stateless_split(tf.constant(args.seed, tf.int32), 3)
    initial = tf.random.stateless_normal((int(args.sample_count),), split[0], dtype=tf.float64)
    noise = tf.random.stateless_normal((int(args.sample_count),), split[1], dtype=tf.float64)
    uniforms = tf.random.stateless_uniform((int(args.sample_count),), split[2], minval=1.0e-12, maxval=1.0, dtype=tf.float64)
    next_state, accepted, log_alpha, proposal = _mh_fixture(initial, noise, uniforms, tf.constant(float(args.sigma), tf.float64))
    movement = tf.abs(next_state - initial)
    mean_error = tf.abs(tf.reduce_mean(next_state))
    second_error = tf.abs(tf.reduce_mean(tf.square(next_state)) - 1.0)
    gates = {
        "finite_state": bool(tf.reduce_all(tf.math.is_finite(next_state)).numpy()),
        "finite_log_alpha": bool(tf.reduce_all(tf.math.is_finite(log_alpha)).numpy()),
        "acceptance_in_range": bool(tf.reduce_all((tf.cast(accepted, tf.int32) >= 0) & (tf.cast(accepted, tf.int32) <= 1)).numpy()),
        "nonzero_movement": bool(tf.reduce_sum(tf.cast(accepted, tf.int32)).numpy() > 0),
        "mean_error_screen": float(mean_error.numpy()) <= 0.06,
        "second_moment_error_screen": float(second_error.numpy()) <= 0.08,
    }
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_mh_fixture.v1",
        "status": "PASS_V2_9_MH_FIXTURE" if all(gates.values()) else "PHASE47_MH_FIXTURE_FAIL",
        "role": "finite_symmetric_mh_invariant_gaussian_fixture",
        "formula": {
            "target": "log pi(x) = -x^2/2 - log(sqrt(2*pi))",
            "proposal": "x_prime = x + sigma * Normal(0,1)",
            "acceptance": "min(1, exp(log pi(x_prime) - log pi(x)))",
            "symmetry": "proposal(x,x_prime) = proposal(x_prime,x)",
        },
        "sample_count": int(args.sample_count),
        "sigma": float(args.sigma),
        "seed": list(args.seed),
        "diagnostics": {
            "mean_abs_error": mean_error,
            "second_moment_abs_error": second_error,
            "acceptance_rate": tf.reduce_mean(tf.cast(accepted, tf.float64)),
            "move_fraction": tf.reduce_mean(tf.cast(movement > 0.0, tf.float64)),
            "proposal_min": tf.reduce_min(proposal),
            "proposal_max": tf.reduce_max(proposal),
        },
        "gates": gates,
        "nonclaims": ["A finite fixture is not an invariance proof for q=20.", "No posterior, whitening, HMC, or LEDH claim."],
        "run_manifest": {
            "program": PLAN.as_posix(), "runner": RUNNER.as_posix(), "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "gpu_hidden_intentionally": True, "jit_compile": True, "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER)},
        },
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text("# v2.9 MH Invariance Fixture\n\nStatus: `" + result["status"] + "`\n\nFinite implementation check only; no q=20 invariance theorem.\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] == "PASS_V2_9_MH_FIXTURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Finite implementation fixture for the v3.1 independent-MH depth lane."""

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
    raise RuntimeError("Phase 49 fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 49 fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)
SAMPLE_COUNT = 8192
DEPTH_STEPS = 8


class Phase49FixtureError(RuntimeError):
    """Raised when the independent-MH fixture cannot be written."""


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(item) for item in value.as_list()]
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
        raise Phase49FixtureError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _normal_log_prob(value: tf.Tensor, mean: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.square(value - mean) - 0.5 * LOG_TWO_PI


@tf.function(
    input_signature=(
        tf.TensorSpec((SAMPLE_COUNT,), tf.float64),
        tf.TensorSpec((SAMPLE_COUNT,), tf.float64),
        tf.TensorSpec((SAMPLE_COUNT,), tf.float64),
        tf.TensorSpec((), tf.float64),
    ),
    jit_compile=True,
    reduce_retracing=False,
)
def _independent_step(
    current: tf.Tensor, candidate: tf.Tensor, uniforms: tf.Tensor, beta: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    log_q_current = _normal_log_prob(current, tf.constant(0.0, tf.float64))
    log_q_candidate = _normal_log_prob(candidate, tf.constant(0.0, tf.float64))
    log_v_current = _normal_log_prob(current, tf.constant(2.0, tf.float64))
    log_v_candidate = _normal_log_prob(candidate, tf.constant(2.0, tf.float64))
    bridge_current = (1.0 - beta) * log_q_current + beta * log_v_current
    bridge_candidate = (1.0 - beta) * log_q_candidate + beta * log_v_candidate
    log_ratio = bridge_candidate - bridge_current + log_q_current - log_q_candidate
    expected = beta * ((log_v_candidate - log_q_candidate) - (log_v_current - log_q_current))
    log_alpha = tf.minimum(tf.constant(0.0, tf.float64), log_ratio)
    accepted = tf.math.log(uniforms) < log_alpha
    next_state = tf.where(accepted, candidate, current)
    return next_state, accepted, log_ratio, log_ratio - expected


def _run_depth(
    current: tf.Tensor, beta: tf.Tensor, seed: tuple[int, int]
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Run the declared repeated-proposal fixture with deterministic seeds."""
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 2 * DEPTH_STEPS)
    state = current
    accepted_total = tf.constant(0, tf.int32)
    max_ratio = tf.constant(0.0, tf.float64)
    max_residual = tf.constant(0.0, tf.float64)
    finite = tf.constant(True)
    for step in range(DEPTH_STEPS):
        candidate = tf.random.stateless_normal((SAMPLE_COUNT,), split[2 * step], dtype=tf.float64)
        uniforms = tf.random.stateless_uniform(
            (SAMPLE_COUNT,), split[2 * step + 1], minval=1.0e-12, maxval=1.0, dtype=tf.float64
        )
        state, accepted, ratio, residual = _independent_step(state, candidate, uniforms, beta)
        accepted_total += tf.reduce_sum(tf.cast(accepted, tf.int32))
        max_ratio = tf.maximum(max_ratio, tf.reduce_max(tf.abs(ratio)))
        max_residual = tf.maximum(max_residual, tf.reduce_max(tf.abs(residual)))
        finite = tf.logical_and(
            finite,
            tf.reduce_all(tf.math.is_finite(tf.concat((state, ratio, residual), axis=0))),
        )
    return state, accepted_total, max_ratio, max_residual, finite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", nargs=2, type=int, default=(20260826, 4801))
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise Phase49FixtureError("output root must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase49FixtureError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    split = tf.random.experimental.stateless_split(tf.constant(args.seed, tf.int32), 2)
    current = tf.random.stateless_normal((SAMPLE_COUNT,), split[0], dtype=tf.float64)
    zero_state, zero_movement, zero_ratio_error, zero_residual_error, zero_finite = _run_depth(
        current, tf.constant(0.0, tf.float64), tuple(int(value) for value in split[1].numpy())
    )
    one_state, one_movement, one_ratio_error, one_residual_error, one_finite = _run_depth(
        current, tf.constant(1.0, tf.float64), (args.seed[0], args.seed[1] + 1)
    )
    one_acceptance = tf.cast(one_movement, tf.float64) / tf.cast(SAMPLE_COUNT * DEPTH_STEPS, tf.float64)
    gates = {
        "finite_states_and_ratios": bool(tf.logical_and(zero_finite, one_finite).numpy()),
        "depth_step_count": DEPTH_STEPS == 8,
        "beta_zero_ratio_identity": float(zero_ratio_error.numpy()) <= 1.0e-12,
        "beta_zero_expected_residual": float(zero_residual_error.numpy()) <= 1.0e-12,
        "beta_zero_all_accepted_over_depth": int(zero_movement.numpy()) == SAMPLE_COUNT * DEPTH_STEPS,
        "beta_one_expected_residual": float(one_residual_error.numpy()) <= 1.0e-12,
        "beta_one_nonzero_movement_over_depth": int(one_movement.numpy()) > 0,
        "beta_one_acceptance_in_unit_interval": 0.0 < float(one_acceptance.numpy()) < 1.0,
    }
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_depth_fixture.v1",
        "status": "PASS_V3_1_INDEPENDENT_MH_DEPTH_FIXTURE" if all(gates.values()) else "PHASE49_INDEPENDENT_MH_DEPTH_FIXTURE_FAIL",
        "role": "finite_repeated_independent_mh_bridge_fixture",
        "formula": {
            "proposal": "q(x)=Normal(0,1)",
            "target": "exp(V(x)) with V(x)=log Normal(x;2,1)",
            "bridge": "(1-beta) log q(x)+beta V(x)",
            "acceptance": "min(1, exp(bridge(x_prime)-bridge(x)+log q(x)-log q(x_prime)))",
            "beta_zero_identity": "log alpha = 0 because pi_0=q",
        },
        "sample_count": SAMPLE_COUNT,
        "depth_steps": DEPTH_STEPS,
        "seed": list(args.seed),
        "diagnostics": {
            "beta_zero_max_abs_log_ratio_over_depth": zero_ratio_error,
            "beta_zero_max_abs_expected_residual": zero_residual_error,
            "beta_zero_acceptance_rate_over_depth": tf.cast(zero_movement, tf.float64) / tf.cast(SAMPLE_COUNT * DEPTH_STEPS, tf.float64),
            "beta_one_max_abs_log_ratio_over_depth": one_ratio_error,
            "beta_one_max_abs_expected_residual_over_depth": one_residual_error,
            "beta_one_acceptance_rate": one_acceptance,
            "beta_one_movement_count": one_movement,
        },
        "gates": gates,
        "nonclaims": ["A finite fixture is not an invariance proof for q=20.", "No posterior, whitening, HMC, or LEDH claim."],
        "run_manifest": {
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "gpu_hidden_intentionally": True,
            "jit_compile": True,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER)},
        },
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(
        "# v3.1 Independent-MH Depth Fixture\n\nStatus: `" + result["status"] + "`\n\nRepeated finite implementation check only; no q=20 invariance theorem.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] == "PASS_V3_1_INDEPENDENT_MH_DEPTH_FIXTURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

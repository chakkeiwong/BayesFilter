"""Finite algebra fixture for the v3.2 q-base/r-proposal MH lane."""

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
    raise RuntimeError("Phase 50 fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 50 fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)
SAMPLE_COUNT = 8192
DIMENSION = 1
DEPTH_STEPS = 8
SUPPORT_RHO = 0.50
SUPPORT_STD = 4.0


class Phase50FixtureError(RuntimeError):
    """Raised when the v3.2 algebra fixture cannot be written."""


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
        raise Phase50FixtureError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _normal_log_prob(x: tf.Tensor, mean: float, scale: float) -> tf.Tensor:
    mean_tensor = tf.constant(mean, tf.float64)
    scale_tensor = tf.constant(scale, tf.float64)
    return -0.5 * tf.square((x - mean_tensor) / scale_tensor) - tf.math.log(scale_tensor) - 0.5 * LOG_TWO_PI


def _log_r(x: tf.Tensor) -> tf.Tensor:
    log_q = _normal_log_prob(x, 0.0, 1.0)
    log_s = _normal_log_prob(x, 0.0, SUPPORT_STD)
    rho = tf.constant(SUPPORT_RHO, tf.float64)
    return tf.reduce_logsumexp(tf.stack((tf.math.log1p(-rho) + log_q, tf.math.log(rho) + log_s), axis=0), axis=0)


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
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    log_q_current = _normal_log_prob(current, 0.0, 1.0)
    log_q_candidate = _normal_log_prob(candidate, 0.0, 1.0)
    log_v_current = _normal_log_prob(current, 2.0, 1.0)
    log_v_candidate = _normal_log_prob(candidate, 2.0, 1.0)
    log_r_current = _log_r(current)
    log_r_candidate = _log_r(candidate)
    bridge_current = (1.0 - beta) * log_q_current + beta * log_v_current
    bridge_candidate = (1.0 - beta) * log_q_candidate + beta * log_v_candidate
    log_ratio = bridge_candidate - bridge_current + log_r_current - log_r_candidate
    direct = (
        (1.0 - beta) * (log_q_candidate - log_q_current)
        + beta * (log_v_candidate - log_v_current)
        + log_r_current
        - log_r_candidate
    )
    log_alpha = tf.minimum(tf.constant(0.0, tf.float64), log_ratio)
    accepted = tf.math.log(uniforms) < log_alpha
    next_state = tf.where(accepted, candidate, current)
    return next_state, accepted, log_ratio, direct, log_r_current, log_r_candidate


def _run_depth(
    current: tf.Tensor, beta: tf.Tensor, seed: tuple[int, int]
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 3 * DEPTH_STEPS)
    state = current
    accepted_total = tf.constant(0, tf.int32)
    broad_total = tf.constant(0, tf.int32)
    max_residual = tf.constant(0.0, tf.float64)
    max_ratio = tf.constant(0.0, tf.float64)
    finite = tf.constant(True)
    for step in range(DEPTH_STEPS):
        choose_broad = tf.random.stateless_uniform((SAMPLE_COUNT,), split[3 * step], dtype=tf.float64) < SUPPORT_RHO
        q_candidate = tf.random.stateless_normal((SAMPLE_COUNT,), split[3 * step + 1], dtype=tf.float64)
        broad_candidate = tf.random.stateless_normal((SAMPLE_COUNT,), split[3 * step + 2], dtype=tf.float64) * SUPPORT_STD
        candidate = tf.where(choose_broad, broad_candidate, q_candidate)
        # A separate deterministic stream keeps uniforms independent of the component draw.
        uniform_seed = tf.random.experimental.stateless_fold_in(tf.constant(seed, tf.int32), step + 1000)
        uniforms = tf.random.stateless_uniform((SAMPLE_COUNT,), uniform_seed, minval=1.0e-12, maxval=1.0, dtype=tf.float64)
        state, accepted, ratio, direct, log_r_current, log_r_candidate = _independent_step(state, candidate, uniforms, beta)
        accepted_total += tf.reduce_sum(tf.cast(accepted, tf.int32))
        broad_total += tf.reduce_sum(tf.cast(choose_broad, tf.int32))
        max_residual = tf.maximum(max_residual, tf.reduce_max(tf.abs(ratio - direct)))
        max_ratio = tf.maximum(max_ratio, tf.reduce_max(tf.abs(ratio)))
        finite = tf.logical_and(
            finite,
            tf.reduce_all(tf.math.is_finite(tf.concat((state, ratio, direct, log_r_current, log_r_candidate), axis=0))),
        )
    return state, accepted_total, broad_total, max_ratio, max_residual, finite, tf.cast(broad_total, tf.float64) / float(SAMPLE_COUNT * DEPTH_STEPS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", nargs=2, type=int, default=(20260826, 5001))
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise Phase50FixtureError("output root must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase50FixtureError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    split = tf.random.experimental.stateless_split(tf.constant(args.seed, tf.int32), 2)
    current = tf.random.stateless_normal((SAMPLE_COUNT,), split[0], dtype=tf.float64)
    zero = _run_depth(current, tf.constant(0.0, tf.float64), tuple(int(value) for value in split[1].numpy()))
    one = _run_depth(current, tf.constant(1.0, tf.float64), (args.seed[0], args.seed[1] + 1))
    zero_acceptance = tf.cast(zero[1], tf.float64) / float(SAMPLE_COUNT * DEPTH_STEPS)
    one_acceptance = tf.cast(one[1], tf.float64) / float(SAMPLE_COUNT * DEPTH_STEPS)
    gates = {
        "finite_states_and_ratios": bool(tf.logical_and(zero[5], one[5]).numpy()),
        "depth_step_count": DEPTH_STEPS == 8,
        "q_base_r_correction_beta_zero": float(zero[4].numpy()) <= 1.0e-12,
        "q_base_r_correction_beta_one": float(one[4].numpy()) <= 1.0e-12,
        "beta_zero_ratio_is_nontrivial": float(zero[3].numpy()) > 1.0e-6,
        "beta_one_ratio_is_nontrivial": float(one[3].numpy()) > 1.0e-6,
        "beta_zero_has_movement": int(zero[1].numpy()) > 0,
        "beta_one_has_movement": int(one[1].numpy()) > 0,
        "acceptance_rates_in_unit_interval": 0.0 < float(zero_acceptance.numpy()) < 1.0 and 0.0 < float(one_acceptance.numpy()) < 1.0,
        "broad_component_fraction_in_expected_band": 0.45 < float(zero[6].numpy()) < 0.55,
    }
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_defensive_support_fixture.v1",
        "status": "PASS_V3_2_DEFENSIVE_SUPPORT_FIXTURE" if all(gates.values()) else "PHASE50_DEFENSIVE_SUPPORT_FIXTURE_FAIL",
        "role": "finite_q_base_r_proposal_independent_mh_algebra_fixture",
        "formula": {
            "base_proposal": "q(x)=Normal(0,1)",
            "support_proposal": "r(x)=(1-rho)q(x)+rho*s(x)",
            "support_component": "s(x)=Normal(0,4^2)",
            "bridge": "(1-beta) log q(x)+beta V(x)",
            "acceptance": "min(1, exp(bridge(x')-bridge(x)+log r(x)-log r(x')))",
            "beta_zero": "log q(x')-log q(x)+log r(x)-log r(x')",
            "beta_one": "V(x')-V(x)+log r(x)-log r(x')",
        },
        "sample_count": SAMPLE_COUNT,
        "dimension": DIMENSION,
        "depth_steps": DEPTH_STEPS,
        "support_rho": SUPPORT_RHO,
        "support_std": SUPPORT_STD,
        "seed": list(args.seed),
        "diagnostics": {
            "beta_zero_max_abs_ratio": zero[3],
            "beta_zero_max_abs_residual": zero[4],
            "beta_zero_acceptance_rate": zero_acceptance,
            "beta_zero_movement_count": zero[1],
            "beta_zero_broad_component_fraction": zero[6],
            "beta_one_max_abs_ratio": one[3],
            "beta_one_max_abs_residual": one[4],
            "beta_one_acceptance_rate": one_acceptance,
            "beta_one_movement_count": one[1],
            "beta_one_broad_component_fraction": one[6],
        },
        "gates": gates,
        "nonclaims": [
            "A finite fixture is not an invariance proof for q=20.",
            "No posterior, whitening, HMC, or LEDH claim.",
        ],
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
        "# v3.2 Defensive Proposal-Support Fixture\n\nStatus: `" + result["status"] + "`\n\nFinite q-base/r-proposal algebra check only; no q=20 invariance theorem.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] == "PASS_V3_2_DEFENSIVE_SUPPORT_FIXTURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

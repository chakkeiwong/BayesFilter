"""Finite full-covariance fixture for the v3.3 mode-aware MH proposal."""

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
    raise RuntimeError("Phase 51 fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 51 fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
PLAN_VERSION = "v3.3-mode-aware-proposal-geometry"
SAMPLE_COUNT = 8192
DIMENSION = 2
DEPTH_STEPS = 8
DEFENSIVE_EPSILON = 0.20
SAFE_STD = 2.0
GEOMETRY_RHO = 0.50
GEOMETRY_SCALE = 2.0
LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)
MEANS = tf.constant(((-2.0, 0.5), (2.0, -0.5)), tf.float64)
BASE_COVARIANCES = tf.constant(
    (
        ((0.50, 0.15), (0.15, 0.80)),
        ((0.70, -0.20), (-0.20, 0.60)),
    ),
    tf.float64,
)
GEOMETRY_COVARIANCES = tf.square(tf.constant(GEOMETRY_SCALE, tf.float64)) * BASE_COVARIANCES
TARGET_COVARIANCES = tf.constant(0.75, tf.float64) * BASE_COVARIANCES
CENTER = tf.reduce_mean(MEANS, axis=0)


class Phase51FixtureError(RuntimeError):
    """Raised when the v3.3 algebra fixture cannot be written."""


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
        raise Phase51FixtureError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _component_log_prob(x: tf.Tensor, means: tf.Tensor, covariances: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariances)
    centered = x[:, tf.newaxis, :] - means[tf.newaxis, :, :]
    solved = tf.linalg.triangular_solve(chol, tf.transpose(centered, (1, 2, 0)))
    quadratic = tf.transpose(tf.reduce_sum(tf.square(solved), axis=1))
    log_determinant = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=1)
    return -0.5 * (quadratic + log_determinant[tf.newaxis, :] + DIMENSION * LOG_TWO_PI)


def _mixture_log_prob(
    x: tf.Tensor,
    means: tf.Tensor,
    covariances: tf.Tensor,
    probabilities: tf.Tensor,
) -> tf.Tensor:
    return tf.reduce_logsumexp(
        _component_log_prob(x, means, covariances) + tf.math.log(probabilities)[tf.newaxis, :],
        axis=1,
    )


def _safe_log_prob(x: tf.Tensor) -> tf.Tensor:
    standardized = (x - CENTER[tf.newaxis, :]) / tf.constant(SAFE_STD, tf.float64)
    return -0.5 * tf.reduce_sum(tf.square(standardized), axis=1) - DIMENSION * (
        tf.math.log(tf.constant(SAFE_STD, tf.float64)) + 0.5 * LOG_TWO_PI
    )


def _log_q(x: tf.Tensor) -> tf.Tensor:
    local = _mixture_log_prob(x, MEANS, BASE_COVARIANCES, tf.constant((0.5, 0.5), tf.float64))
    safe = _safe_log_prob(x)
    epsilon = tf.constant(DEFENSIVE_EPSILON, tf.float64)
    return tf.reduce_logsumexp(
        tf.stack((tf.math.log1p(-epsilon) + local, tf.math.log(epsilon) + safe), axis=1),
        axis=1,
    )


def _log_geometry(x: tf.Tensor) -> tf.Tensor:
    return _mixture_log_prob(x, MEANS, GEOMETRY_COVARIANCES, tf.constant((0.5, 0.5), tf.float64))


def _log_r(x: tf.Tensor) -> tf.Tensor:
    rho = tf.constant(GEOMETRY_RHO, tf.float64)
    return tf.reduce_logsumexp(
        tf.stack((tf.math.log1p(-rho) + _log_q(x), tf.math.log(rho) + _log_geometry(x)), axis=1),
        axis=1,
    )


def _log_target(x: tf.Tensor) -> tf.Tensor:
    return _mixture_log_prob(x, MEANS, TARGET_COVARIANCES, tf.constant((0.60, 0.40), tf.float64))


def _sample_local(seed: tf.Tensor, covariances: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(seed, 2)
    labels = tf.cast(
        tf.random.stateless_uniform((SAMPLE_COUNT,), split[0], dtype=tf.float64) >= 0.5,
        tf.int32,
    )
    noise = tf.random.stateless_normal((SAMPLE_COUNT, DIMENSION), split[1], dtype=tf.float64)
    selected_means = tf.gather(MEANS, labels)
    selected_chol = tf.gather(tf.linalg.cholesky(covariances), labels)
    return selected_means + tf.einsum("nij,nj->ni", selected_chol, noise), labels


def _sample_q(seed: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(seed, 4)
    local, labels = _sample_local(split[0], BASE_COVARIANCES)
    safe = CENTER[tf.newaxis, :] + SAFE_STD * tf.random.stateless_normal(
        (SAMPLE_COUNT, DIMENSION), split[1], dtype=tf.float64
    )
    choose_safe = (
        tf.random.stateless_uniform((SAMPLE_COUNT,), split[2], dtype=tf.float64)
        < DEFENSIVE_EPSILON
    )
    return tf.where(choose_safe[:, None], safe, local), tf.where(choose_safe, 2, labels)


def _sample_r(seed: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(seed, 4)
    q_sample, q_labels = _sample_q(split[0])
    geometry_sample, geometry_labels = _sample_local(split[1], GEOMETRY_COVARIANCES)
    choose_geometry = (
        tf.random.stateless_uniform((SAMPLE_COUNT,), split[2], dtype=tf.float64)
        < GEOMETRY_RHO
    )
    return (
        tf.where(choose_geometry[:, None], geometry_sample, q_sample),
        tf.where(choose_geometry, geometry_labels + 3, q_labels),
    )


@tf.function(
    input_signature=(
        tf.TensorSpec((SAMPLE_COUNT, DIMENSION), tf.float64),
        tf.TensorSpec((SAMPLE_COUNT, DIMENSION), tf.float64),
        tf.TensorSpec((SAMPLE_COUNT,), tf.float64),
        tf.TensorSpec((), tf.float64),
    ),
    jit_compile=True,
    reduce_retracing=False,
)
def _independent_step(
    current: tf.Tensor,
    candidate: tf.Tensor,
    uniforms: tf.Tensor,
    beta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    log_q_current = _log_q(current)
    log_q_candidate = _log_q(candidate)
    log_v_current = _log_target(current)
    log_v_candidate = _log_target(candidate)
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
    accepted = tf.math.log(uniforms) < tf.minimum(tf.constant(0.0, tf.float64), log_ratio)
    return tf.where(accepted[:, None], candidate, current), accepted, log_ratio, direct, log_r_current, log_r_candidate


def _run_depth(
    current: tf.Tensor,
    beta: tf.Tensor,
    seed: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 2 * DEPTH_STEPS)
    state = current
    accepted_total = tf.constant(0, tf.int32)
    geometry_total = tf.constant(0, tf.int32)
    max_residual = tf.constant(0.0, tf.float64)
    max_ratio = tf.constant(0.0, tf.float64)
    finite = tf.constant(True)
    for step in range(DEPTH_STEPS):
        candidate, labels = _sample_r(split[2 * step])
        uniforms = tf.random.stateless_uniform(
            (SAMPLE_COUNT,), split[2 * step + 1], minval=1.0e-12, maxval=1.0, dtype=tf.float64
        )
        state, accepted, ratio, direct, log_r_current, log_r_candidate = _independent_step(
            state, candidate, uniforms, beta
        )
        accepted_total += tf.reduce_sum(tf.cast(accepted, tf.int32))
        geometry_total += tf.reduce_sum(tf.cast(labels >= 3, tf.int32))
        max_residual = tf.maximum(max_residual, tf.reduce_max(tf.abs(ratio - direct)))
        max_ratio = tf.maximum(max_ratio, tf.reduce_max(tf.abs(ratio)))
        finite = tf.logical_and(
            finite,
            tf.reduce_all(
                tf.math.is_finite(
                    tf.concat((tf.reshape(state, (-1,)), ratio, direct, log_r_current, log_r_candidate), axis=0)
                )
            ),
        )
    denominator = tf.cast(SAMPLE_COUNT * DEPTH_STEPS, tf.float64)
    return (
        state,
        accepted_total,
        geometry_total,
        max_ratio,
        max_residual,
        finite,
        tf.cast(geometry_total, tf.float64) / denominator,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", nargs=2, type=int, default=(20260826, 5101))
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise Phase51FixtureError("output root must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase51FixtureError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    split = tf.random.experimental.stateless_split(tf.constant(args.seed, tf.int32), 3)
    current, _ = _sample_q(split[0])
    zero = _run_depth(current, tf.constant(0.0, tf.float64), tuple(int(value) for value in split[1].numpy()))
    one = _run_depth(current, tf.constant(1.0, tf.float64), tuple(int(value) for value in split[2].numpy()))
    zero_acceptance = tf.cast(zero[1], tf.float64) / float(SAMPLE_COUNT * DEPTH_STEPS)
    one_acceptance = tf.cast(one[1], tf.float64) / float(SAMPLE_COUNT * DEPTH_STEPS)
    base_eigenvalues = tf.linalg.eigvalsh(BASE_COVARIANCES)
    geometry_eigenvalues = tf.linalg.eigvalsh(GEOMETRY_COVARIANCES)
    gates = {
        "finite_states_densities_and_ratios": bool(tf.logical_and(zero[5], one[5]).numpy()),
        "full_covariances_are_spd": bool(
            tf.logical_and(tf.reduce_all(base_eigenvalues > 0.0), tf.reduce_all(geometry_eigenvalues > 0.0)).numpy()
        ),
        "normalized_mixture_weights": abs((0.5 + 0.5) - 1.0) <= 1.0e-15,
        "depth_step_count": DEPTH_STEPS == 8,
        "q_base_r_geometry_correction_beta_zero": float(zero[4].numpy()) <= 1.0e-12,
        "q_base_r_geometry_correction_beta_one": float(one[4].numpy()) <= 1.0e-12,
        "beta_zero_ratio_is_nontrivial": float(zero[3].numpy()) > 1.0e-6,
        "beta_one_ratio_is_nontrivial": float(one[3].numpy()) > 1.0e-6,
        "beta_zero_has_movement": int(zero[1].numpy()) > 0,
        "beta_one_has_movement": int(one[1].numpy()) > 0,
        "acceptance_rates_in_unit_interval": 0.0 < float(zero_acceptance.numpy()) < 1.0
        and 0.0 < float(one_acceptance.numpy()) < 1.0,
        "geometry_component_fraction_in_expected_band": 0.45 < float(zero[6].numpy()) < 0.55,
    }
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_mode_aware_geometry_fixture.v1",
        "status": "PASS_V3_3_MODE_AWARE_GEOMETRY_FIXTURE" if all(gates.values()) else "PHASE51_MODE_AWARE_GEOMETRY_FIXTURE_FAIL",
        "plan_version": PLAN_VERSION,
        "role": "finite_full_covariance_q_base_r_geometry_independent_mh_algebra_fixture",
        "formula": {
            "base_proposal": "q=(1-epsilon)*0.5*(N(m_minus,C_minus)+N(m_plus,C_plus))+epsilon*N(center,2^2*I)",
            "geometry_component": "s_geom=0.5*N(m_minus,kappa^2*C_minus)+0.5*N(m_plus,kappa^2*C_plus)",
            "candidate_proposal": "r_geom=(1-rho)*q+rho*s_geom",
            "bridge": "(1-beta)*log q+beta*V",
            "acceptance": "min(1,exp(bridge(x')-bridge(x)+log r_geom(x)-log r_geom(x')))",
        },
        "sample_count": SAMPLE_COUNT,
        "dimension": DIMENSION,
        "depth_steps": DEPTH_STEPS,
        "geometry_rho": GEOMETRY_RHO,
        "geometry_scale": GEOMETRY_SCALE,
        "defensive_epsilon": DEFENSIVE_EPSILON,
        "safe_std": SAFE_STD,
        "seed": list(args.seed),
        "diagnostics": {
            "base_covariance_eigenvalues": base_eigenvalues,
            "geometry_covariance_eigenvalues": geometry_eigenvalues,
            "beta_zero_max_abs_ratio": zero[3],
            "beta_zero_max_abs_residual": zero[4],
            "beta_zero_acceptance_rate": zero_acceptance,
            "beta_zero_movement_count": zero[1],
            "beta_zero_geometry_component_fraction": zero[6],
            "beta_one_max_abs_ratio": one[3],
            "beta_one_max_abs_residual": one[4],
            "beta_one_acceptance_rate": one_acceptance,
            "beta_one_movement_count": one[1],
            "beta_one_geometry_component_fraction": one[6],
        },
        "gates": gates,
        "nonclaims": [
            "A finite two-dimensional fixture is not an invariance proof for q=20.",
            "The fixture covariances are algebra fixtures, not SSL-LSTM posterior covariances.",
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
        "# v3.3 Mode-Aware Proposal-Geometry Fixture\n\nStatus: `"
        + result["status"]
        + "`\n\nFinite full-covariance algebra check only; no q=20 invariance theorem.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] == "PASS_V3_3_MODE_AWARE_GEOMETRY_FIXTURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

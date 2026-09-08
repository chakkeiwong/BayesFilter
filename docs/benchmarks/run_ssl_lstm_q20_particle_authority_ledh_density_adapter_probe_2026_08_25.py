"""Probe q20 LEDH density compatibility without changing the target measure."""

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
    raise RuntimeError("q20 density probe requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("q20 density probe requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("q20 density probe found a visible GPU")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)


TARGET = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
STRUCTURAL = ROOT / "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py"
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-particle-authority-phase25-q20-ledh-density-"
    "adapter-subplan-2026-08-25.md"
)
RUNNER = Path(__file__).resolve()


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


def _finite(value: Any) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def _gaussian_log_density(residual: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    """Density in innovation coordinates only, not in the full state space."""
    residual = tf.convert_to_tensor(residual, tf.float64)
    covariance = tf.convert_to_tensor(covariance, tf.float64)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.triangular_solve(chol, residual[..., tf.newaxis])
    quadratic = tf.reduce_sum(tf.square(solved[..., 0]), axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(tf.shape(residual)[-1], tf.float64)
    return -0.5 * (
        dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))
        + logdet
        + quadratic
    )


def build_probe(rank_tolerance: float = 1.0e-10) -> dict[str, Any]:
    started = time.perf_counter()
    target = batch_native_complexity_posterior_target(
        20, jit_compile=False, principal_sqrt_backend="tensorflow_eigh"
    )
    free = tf.constant(
        [[-0.35, -0.20, 0.15, 0.25], [0.40, 0.10, -0.25, -0.10]],
        dtype=tf.float64,
    )
    model, derivatives = target._batched_components(free)
    batch_size = int(free.shape[0])
    point_count = 3
    state_dim = int(model.state_dim)
    innovation_dim = int(model.innovation_dim)
    parameter_dim = int(target.parameter_dim)
    previous = tf.zeros([batch_size, point_count, state_dim], tf.float64)
    innovation_axis = tf.linspace(
        tf.constant(-0.1, tf.float64), tf.constant(0.1, tf.float64), innovation_dim
    )
    innovation = tf.broadcast_to(
        innovation_axis[tf.newaxis, tf.newaxis, :],
        [batch_size, point_count, innovation_dim],
    )
    next_state = model.transition(previous, innovation)
    zero_innovation = tf.zeros_like(innovation)
    zero_next = model.transition(previous, zero_innovation)
    recovered_innovation = next_state[:, :, :innovation_dim] - zero_next[:, :, :innovation_dim]
    innovation_covariance = model.innovation_covariance
    innovation_log_density = _gaussian_log_density(
        recovered_innovation, innovation_covariance[:, tf.newaxis, :, :]
    )
    innovation_jacobian = derivatives.transition_innovation_jacobian_fn(
        previous, innovation
    )
    induced_covariance = tf.einsum(
        "bpni,bij,bpmj->bpnm",
        innovation_jacobian,
        innovation_covariance,
        innovation_jacobian,
    )
    eigenvalues = tf.linalg.eigvalsh(induced_covariance)
    rank_counts = {
        str(tolerance): tf.reduce_sum(
            tf.cast(eigenvalues > tf.constant(tolerance, tf.float64), tf.int32), axis=-1
        )
        for tolerance in (1.0e-12, rank_tolerance, 1.0e-8)
    }
    perturbed = next_state + tf.one_hot(
        innovation_dim, state_dim, dtype=tf.float64
    )[tf.newaxis, tf.newaxis, :]
    structural_residual = model.deterministic_residual(
        previous, innovation, next_state
    )
    perturbed_residual = model.deterministic_residual(
        previous, innovation, perturbed
    )
    value, score, status = target.neutra_batch_log_prob_and_grad_status(free)
    full_state_rank = rank_counts[str(rank_tolerance)]
    rank_deficient = bool(tf.reduce_any(full_state_rank < state_dim).numpy())
    hard_checks = {
        "target_finite": _finite(value) and _finite(score),
        "covariance_finite": _finite(innovation_covariance),
        "eigenvalues_finite": _finite(eigenvalues),
        "innovation_density_finite": _finite(innovation_log_density),
        "generated_transition_residual_finite": _finite(structural_residual),
        "perturbed_residual_finite": _finite(perturbed_residual),
        "target_status_present": bool(status),
    }
    if not all(hard_checks.values()):
        result_status = "Q20_DENSITY_PROBE_FAIL_REPAIR"
    elif rank_deficient:
        result_status = (
            "DIRECT_FULL_STATE_LEDH_BLOCKED_SINGULAR_MEASURE_REDUCED_REPAIR"
        )
    else:
        result_status = "DIRECT_Q20_LEDH_MEASURE_COMPATIBILITY_UNRESOLVED"
    return {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.ledh_density_probe.v1",
        "status": result_status,
        "role": "q20_measure_compatibility_diagnostic",
        "target": {
            "q": 20,
            "target_scope": target.target_scope,
            "target_signature": target.target_signature(),
            "adapter_signature": target.adapter_signature(),
            "parameter_dim": parameter_dim,
            "aggregate_target": "UKF marginal value and score over four parameters",
        },
        "dimensions": {
            "batch": batch_size,
            "points": point_count,
            "state_dim": state_dim,
            "innovation_dim": innovation_dim,
            "parameter_dim": parameter_dim,
            "deterministic_state_dim": state_dim - innovation_dim,
        },
        "measure": {
            "innovation_density": "Gaussian density in R^20 using model.innovation_covariance",
            "full_state_transition": "pushforward covariance G Q G^T with deterministic residual constraints",
            "parameter_target": "aggregate four-dimensional target.neutra_batch_log_prob_and_grad_status",
            "common_full_state_lebesgue_density": not rank_deficient,
            "direct_ledh_admission": False,
            "rank_tolerance": rank_tolerance,
            "rank_tolerance_provenance": "Phase 25 diagnostic placeholder; not a promotion threshold",
        },
        "rank_receipt": {
            "eigenvalues": eigenvalues,
            "rank_counts": rank_counts,
            "min_eigenvalue": tf.reduce_min(eigenvalues),
            "max_eigenvalue": tf.reduce_max(eigenvalues),
            "generated_structural_residual_max_abs": tf.reduce_max(tf.abs(structural_residual)),
            "perturbed_structural_residual_max_abs": tf.reduce_max(tf.abs(perturbed_residual)),
            "innovation_log_density": innovation_log_density,
        },
        "hard_checks": hard_checks,
        "repair_boundary": {
            "classification": (
                "route_specific_measure_blocker_for_direct_full_state_LEDH; "
                "reduced_innovation_coordinate investigation remains in scope"
            ),
            "smallest_next_step": (
                "construct a reduced-coordinate proposal whose density is explicit "
                "and prove how it targets the four-parameter posterior before any flow"
            ),
            "do_not_do": (
                "do not divide by a zero determinant, assign a Lebesgue density to "
                "the rank-deficient 60-state transition, or relabel the aggregate "
                "parameter target as a state-space transition likelihood"
            ),
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
            "random_seed": "deterministic_fixed_tensor_no_rng",
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "runner": _sha(RUNNER),
                "target": _sha(TARGET),
                "structural": _sha(STRUCTURAL),
                "plan": _sha(PLAN),
            },
        },
        "nonclaims": [
            "a finite innovation-coordinate density is not a full-state transition density",
            "rank deficiency does not prove that a reduced-coordinate flow cannot be designed",
            "no q20 LEDH, posterior, whitening, mode-discovery, or HMC promotion",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rank-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise RuntimeError("output root must be repository-relative")
    if not 0.0 < args.rank_tolerance < 1.0:
        raise ValueError("rank tolerance must be in (0,1)")
    output = ROOT / args.output_root
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    result = build_probe(args.rank_tolerance)
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 25 q20 LEDH Density-Adapter Probe\n\n"
        f"Status: `{result['status']}`\n\n"
        "The induced full-state transition measure and the four-parameter "
        "aggregate target are recorded separately. Direct full-state LEDH is "
        "not admitted by this diagnostic.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] != "Q20_DENSITY_PROBE_FAIL_REPAIR" else 2


if __name__ == "__main__":
    raise SystemExit(main())

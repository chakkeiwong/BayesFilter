"""Bounded timing pilot for Phase 4A calibration row-budget planning.

This runner estimates wall time and peak GPU memory across the intended
horizon/particle-count ladder to set honest row budgets before the serious
Phase 4A.5 campaign. It does NOT tune bandwidth or run untouched validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _lgssm_model(dtype: tf.dtypes.DType, state_dim: int, obs_dim: int):
    """Linear-Gaussian state-space model for timing (not oracle comparison)."""
    from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel

    transition_matrix = 0.95 * tf.eye(state_dim, dtype=dtype)
    process_covariance = 0.1 * tf.eye(state_dim, dtype=dtype)
    observation_matrix = tf.eye(obs_dim, state_dim, dtype=dtype)
    observation_covariance = 0.2 * tf.eye(obs_dim, dtype=dtype)

    def transition_mean_fn(theta, points):
        return tf.linalg.matvec(transition_matrix, points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return tf.linalg.matvec(transition_matrix, d_points)

    def observation_fn(points):
        return tf.linalg.matvec(observation_matrix, points)

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            observation_matrix, [tf.shape(points)[0], obs_dim, state_dim]
        )

    def observation_tangent_fn(points, d_points):
        return tf.linalg.matvec(observation_matrix, d_points)

    def observation_jacobian_tangent_fn(points, d_points):
        return tf.zeros([tf.shape(points)[0], obs_dim, state_dim], dtype=dtype)

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=process_covariance,
        observation_covariance=observation_covariance,
        observation_jacobian_tangent_fn=observation_jacobian_tangent_fn,
    )


def _float(value: tf.Tensor) -> float:
    return float(value.numpy())


def _time_one_configuration(
    particle_count: int,
    horizon: int,
    state_dim: int,
    obs_dim: int,
    bandwidth_rho: float,
    dtype: tf.dtypes.DType,
    jit_compile: bool,
    repeats: int,
) -> dict:
    """Time a single (N, T, rho) configuration."""
    from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
        make_integrated_linear_gaussian_kdm_kernel,
    )

    model = _lgssm_model(dtype, state_dim, obs_dim)

    # Define matrices directly since NonlinearScoreModel doesn't expose them
    observation_matrix = tf.eye(obs_dim, state_dim, dtype=dtype)
    process_covariance = 0.1 * tf.eye(state_dim, dtype=dtype)

    with tf.device("/GPU:0"):
        initial_states = tf.random.stateless_normal(
            [particle_count, state_dim], seed=[2609, 9001], dtype=dtype
        )
        initial_covariances = tf.broadcast_to(
            tf.eye(state_dim, dtype=dtype),
            [particle_count, state_dim, state_dim],
        )
        noises = tf.random.stateless_normal(
            [horizon, particle_count, state_dim],
            seed=[2609, 9002],
            dtype=dtype,
        )
        observations = tf.random.stateless_normal(
            [horizon, obs_dim], seed=[2609, 9003], dtype=dtype
        )
        d_observation_matrix = tf.zeros_like(observation_matrix)
        scale = tf.sqrt(tf.linalg.diag_part(process_covariance))
        # The declared Phase 4A family is B=diag((rho*s_d)^2), where
        # s_d=sqrt(Q_dd).  Keep this pilot on the same covariance scale as
        # the integrated KDM kernel and the research plan.
        bandwidth_diagonal = tf.square(bandwidth_rho * scale)
        bandwidths = tf.broadcast_to(
            tf.linalg.diag(bandwidth_diagonal),
            [horizon, particle_count, state_dim, state_dim],
        )
        d_bandwidths = tf.zeros_like(bandwidths)
        theta = tf.constant([1.0], dtype)
        base = tf.concat(
            [tf.eye(state_dim, dtype=dtype), -tf.eye(state_dim, dtype=dtype)],
            axis=0,
        )
        reset_design = tf.tile(base, [particle_count // (2 * state_dim), 1])

    options = {
        "flow_substeps": 6,
        "reset_policy": "contract_e",
        "reset_design": reset_design,
        "reset_sinkhorn_steps": 4,
        "reset_balance_steps": 2,
        "correction_steps": 1,
        "pairwise_steps": 1,
        "coordinate_cap": 0.95,
    }
    kernel = make_integrated_linear_gaussian_kdm_kernel(
        model=model,
        theta_dimension=1,
        particle_count=particle_count,
        state_dimension=state_dim,
        observation_dimension=obs_dim,
        horizon=horizon,
        canonical_options=options,
        bandwidth_is_zero=(bandwidth_rho == 0.0),
        dtype=dtype,
        jit_compile=jit_compile,
    )
    inputs = (
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        observation_matrix,
        d_observation_matrix,
        bandwidths,
        d_bandwidths,
    )

    # First call includes compile
    compile_start = time.perf_counter()
    result = kernel(*inputs)
    result["value"].numpy()
    compile_elapsed = time.perf_counter() - compile_start

    # Warm repeats
    repeat_times = []
    for _ in range(repeats):
        repeat_start = time.perf_counter()
        repeated = kernel(*inputs)
        repeated["value"].numpy()
        repeat_times.append(time.perf_counter() - repeat_start)

    memory_info = tf.config.experimental.get_memory_info("GPU:0")
    return {
        "particle_count": particle_count,
        "horizon": horizon,
        "state_dimension": state_dim,
        "observation_dimension": obs_dim,
        "bandwidth_rho": bandwidth_rho,
        "bandwidth_is_zero": bandwidth_rho == 0.0,
        "jit_compile": jit_compile,
        "valid": bool(result["valid"].numpy()),
        "compile_and_first_seconds": compile_elapsed,
        "repeat_seconds": repeat_times,
        "mean_repeat_seconds": float(sum(repeat_times) / len(repeat_times)),
        "min_repeat_seconds": float(min(repeat_times)),
        "peak_gpu_bytes": int(memory_info["peak"]),
        "current_gpu_bytes": int(memory_info["current"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dim", type=int, default=2)
    parser.add_argument("--obs-dim", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--tf32-mode", choices=("enabled", "disabled"), default="disabled"
    )
    parser.add_argument(
        "--jit-compile", choices=("true", "false"), default="true"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Run only the first bounded number of ladder rows.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Zero-based ladder row at which to start.",
    )
    arguments = parser.parse_args()

    if arguments.repeats < 2:
        raise ValueError("repeats must be at least two")

    started = time.time()
    wall_start = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(
        arguments.tf32_mode == "enabled"
    )
    logical_devices = tf.config.list_logical_devices("GPU")
    if len(logical_devices) != 1:
        raise RuntimeError("pilot requires exactly one visible logical GPU")

    dtype = tf.float32 if arguments.dtype == "float32" else tf.float64
    jit_compile = arguments.jit_compile == "true"

    # Reduced representative ladder for timing pilot
    # Full ladder: 14 rows from (32,5) to (512,50)
    # This pilot samples 6 representative points for budget estimation
    ladder = [
        # (N, T, rho) — atom and positive bandwidth for each size class
        (32, 5, 0.0),
        (32, 5, 0.20),
        (128, 20, 0.0),
        (128, 20, 0.20),
        (512, 50, 0.0),
        (512, 50, 0.20),
    ]
    if arguments.start_row < 0 or arguments.start_row >= len(ladder):
        raise ValueError("start-row must identify an existing ladder row")
    end_row = len(ladder)
    if arguments.max_rows is not None:
        if arguments.max_rows < 1:
            raise ValueError("max-rows must be positive")
        end_row = min(arguments.start_row + arguments.max_rows, len(ladder))
    ladder = ladder[arguments.start_row:end_row]

    timing_rows = []
    for particle_count, horizon, bandwidth_rho in ladder:
        row = _time_one_configuration(
            particle_count=particle_count,
            horizon=horizon,
            state_dim=arguments.state_dim,
            obs_dim=arguments.obs_dim,
            bandwidth_rho=bandwidth_rho,
            dtype=dtype,
            jit_compile=jit_compile,
            repeats=arguments.repeats,
        )
        timing_rows.append(row)
        print(
            f"N={particle_count:3d} T={horizon:2d} rho={bandwidth_rho:.2f} "
            f"→ {row['mean_repeat_seconds']*1000:.1f} ms "
            f"(peak {row['peak_gpu_bytes']/1e6:.0f} MB)",
            flush=True,
        )

    root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path("bayesfilter/highdim/ledh_canonical_score_tf.py"),
        Path("bayesfilter/highdim/ledh_canonical_score_stages_tf.py"),
        Path("bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py"),
        Path(PLAN_PATH),
    )

    total_ladder_seconds = sum(r["mean_repeat_seconds"] for r in timing_rows)
    max_peak_bytes = max(r["peak_gpu_bytes"] for r in timing_rows)

    result = {
        "schema": "bayesfilter.ledh_younis_kdm_phase4a_timing_pilot.v1",
        "role": "bounded_timing_smoke_for_campaign_budget_planning",
        "question": "What are realistic GPU wall times and peak memory for the Phase 4A.5 intended ladder?",
        "nonclaims": [
            "does not tune bandwidth or LEDH controls",
            "does not compare with Kalman oracle or canonical baseline",
            "does not run untouched validation paths",
            "does not establish production, HMC, or default readiness",
        ],
        "ladder": ladder,
        "max_rows": arguments.max_rows,
        "start_row": arguments.start_row,
        "timing_rows": timing_rows,
        "total_ladder_mean_seconds": total_ladder_seconds,
        "max_peak_gpu_bytes": max_peak_bytes,
        "state_dimension": arguments.state_dim,
        "observation_dimension": arguments.obs_dim,
        "dtype": dtype.name,
        "jit_compile": jit_compile,
        "tf32_mode": arguments.tf32_mode,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "repeats": arguments.repeats,
        "tensorflow_version": tf.__version__,
        "python": sys.executable,
        "environment_prefix": sys.prefix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
        "gpu_memory_policy": memory_policy,
        "logical_devices": [device.name for device in logical_devices],
        "trust_basis": TRUST_BASIS,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "command": shlex.join((sys.executable, *sys.argv)),
        "started_unix": started,
        "wall_time_seconds": time.perf_counter() - wall_start,
        "plan_file": PLAN_PATH,
        "result_file": str(arguments.output),
        "source_sha256": {
            str(path): _sha256(root / path) for path in source_paths
        },
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\nTotal ladder time: {total_ladder_seconds:.1f} s "
        f"({len(timing_rows)} configurations)"
    )
    print(f"Max peak GPU memory: {max_peak_bytes/1e9:.2f} GB")
    print(f"Result: {arguments.output}")


if __name__ == "__main__":
    main()

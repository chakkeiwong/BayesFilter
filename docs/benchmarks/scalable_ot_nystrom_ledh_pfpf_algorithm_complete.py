"""Nystrom LEDH/PFPF-OT algorithm-complete diagnostic harness.

This lane-owned harness exercises the experimental fixed-rank Nystrom kernel
transport as a diagnostic LEDH/PFPF-OT resampling candidate.  It does not change
BayesFilter defaults and does not establish speedup, posterior correctness,
HMC readiness, production readiness, public API readiness, or ranking.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PRE = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE.add_argument(
    "--mode",
    choices=("small-reference", "downstream-smoke", "gpu-scale"),
    default="small-reference",
)
_PRE.add_argument("--device-scope", choices=("cpu", "visible"), default=None)
_PRE.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _ = _PRE.parse_known_args()
if _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
elif _PRE_ARGS.device_scope == "cpu" or (_PRE_ARGS.device_scope is None and _PRE_ARGS.mode != "gpu-scale"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

try:  # noqa: E402
    import tensorflow_probability as tfp  # noqa: E402
except Exception:  # noqa: BLE001
    tfp = None

from experiments.dpf_implementation.tf_tfp.filters import experimental_batched_ledh_pfpf_ot_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (  # noqa: E402
    _log_weight_floor,
    _normalize_log_weights,
    _weighted_mean_and_variance,
    batched_ledh_flow_core_tf,
    uniform_log_weights,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.resampling import nystrom_transport_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (  # noqa: E402
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.nystrom_transport_tf import (  # noqa: E402
    nystrom_transport_resample_tf,
)


PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-nystrom-algorithm-complete-"
    "master-program-2026-06-21.md"
)
P01_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-nystrom-algorithm-complete-"
    "p01-implementation-harness-result-2026-06-21.md"
)
P02_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-nystrom-algorithm-complete-"
    "p02-small-reference-result-2026-06-21.md"
)
P03_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-nystrom-algorithm-complete-"
    "p03-downstream-smoke-result-2026-06-21.md"
)
P04_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-nystrom-algorithm-complete-"
    "p04-gpu-scale-result-2026-06-21.md"
)

ALGORITHM_FAMILY = "fixed_rank_nystrom_kernel_sinkhorn_ledh_pfpf_ot_diagnostic"
SOURCE_ROUTE = "fixed_hmc_adaptation"
SEMANTIC_CLASS = "approximate_kernel"
BASELINE_COMPARATOR = "dense_tensorflow_annealed_transport_small_reference_only"
TRANSPORT_OBJECT_KIND = "kernel_factors"
TRANSPORT_MATRIX_MATERIALIZED = False
FIXTURE_ID = "nystrom_ledh_lgssm_algorithm_complete_v1"
ROW_COLUMN_THRESHOLD = 5.0e-2
DENSE_MAX_ERROR_THRESHOLD = 7.5e-2
DENSE_RMS_ERROR_THRESHOLD = 3.0e-2
LOG_WEIGHT_NORMALIZATION_THRESHOLD = 1.0e-6
ESS_FRACTION_THRESHOLD = 1.0e-2
P04_ROW_TIMEOUT_SECONDS = 1200
P04_PHASE_TIMEOUT_SECONDS = 7200
P04_OPTIONAL_ROW_ELAPSED_LIMIT_SECONDS = 2700
NONCLAIMS = (
    "Nystrom LEDH/PFPF-OT diagnostic only",
    "no speedup claim",
    "no ranking claim",
    "no superiority claim",
    "no production/default readiness claim",
    "no production/default route change",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no dense Sinkhorn equivalence claim beyond checked small fixtures",
    "no broad scalable-OT leaderboard claim",
)
SOURCE_ROUTE_COMPONENTS = {
    "nystrom_factors": "source_faithful",
    "low_rank_scaling": "source_faithful",
    "filterflow_cost_scaling_adapter": "fixed_hmc_adaptation",
    "deterministic_landmarks": "fixed_hmc_adaptation",
    "cholesky_jitter": "fixed_hmc_adaptation",
}

SMALL_REFERENCE_SPECS = {
    "tiny_manual": (4, (2, 3, 4)),
    "small_parity": (8, (2, 4, 8)),
    "high_dim_low_rank": (32, (2, 4, 8, 16, 32)),
    "ledh_specific_smoke": (32, (4, 8, 16, 32)),
}
DOWNSTREAM_ROWS = (
    {"fixture_id": "nystrom_ledh_smoke_n64_rank8", "particle_count": 64, "rank": 8, "time_steps": 2, "state_dim": 6, "obs_dim": 4},
    {"fixture_id": "nystrom_ledh_smoke_n128_rank16", "particle_count": 128, "rank": 16, "time_steps": 2, "state_dim": 6, "obs_dim": 4},
)
GPU_REQUIRED_ROWS = (
    {"fixture_id": "nystrom_gpu_n1024_rank16", "particle_count": 1024, "rank": 16, "time_steps": 2, "state_dim": 8, "obs_dim": 6, "required": True},
    {"fixture_id": "nystrom_gpu_n4096_rank32", "particle_count": 4096, "rank": 32, "time_steps": 2, "state_dim": 8, "obs_dim": 6, "required": True},
    {"fixture_id": "nystrom_gpu_n8192_rank32", "particle_count": 8192, "rank": 32, "time_steps": 2, "state_dim": 8, "obs_dim": 6, "required": True},
)
GPU_OPTIONAL_ROW = {
    "fixture_id": "nystrom_gpu_n16384_rank64",
    "particle_count": 16384,
    "rank": 64,
    "time_steps": 2,
    "state_dim": 8,
    "obs_dim": 6,
    "required": False,
}


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=("small-reference", "downstream-smoke", "gpu-scale"),
        default=_PRE_ARGS.mode,
    )
    parser.add_argument("--fixtures", default="all")
    parser.add_argument("--ranks", default="plan")
    parser.add_argument("--row-spec", default=None)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--baseline-scaling", type=float, default=0.9)
    parser.add_argument("--baseline-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--baseline-max-iterations", type=int, default=12)
    parser.add_argument("--baseline-row-chunk-size", type=int, default=4)
    parser.add_argument("--baseline-col-chunk-size", type=int, default=4)
    parser.add_argument("--nystrom-max-iterations", type=int, default=160)
    parser.add_argument("--nystrom-convergence-threshold", type=float, default=1.0e-4)
    parser.add_argument("--cholesky-jitter", type=float, default=1.0e-8)
    parser.add_argument("--denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default=None)
    parser.add_argument("--trust-context", default=None)
    parser.add_argument("--row-timeout-seconds", type=int, default=None)
    parser.add_argument("--phase-result-path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args(argv)
    args = _apply_mode_defaults(args)
    _validate_args(args)
    return args


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "small-reference":
        args.dtype = args.dtype or "float64"
        args.device_scope = args.device_scope or "cpu"
        args.device = args.device or "/CPU:0"
        args.tf32_mode = args.tf32_mode or "disabled"
        args.trust_context = args.trust_context or "cpu_hidden_local"
        args.row_timeout_seconds = args.row_timeout_seconds or 60
    elif args.mode == "downstream-smoke":
        args.dtype = args.dtype or "float64"
        args.device_scope = args.device_scope or "cpu"
        args.device = args.device or "/CPU:0"
        args.tf32_mode = args.tf32_mode or "disabled"
        args.trust_context = args.trust_context or "cpu_hidden_local"
        args.row_timeout_seconds = args.row_timeout_seconds or 120
    else:
        args.dtype = args.dtype or "float32"
        args.device_scope = args.device_scope or "visible"
        args.device = args.device or "/GPU:0"
        args.tf32_mode = args.tf32_mode or "enabled"
        args.trust_context = args.trust_context or "trusted_gpu_required"
        args.row_timeout_seconds = args.row_timeout_seconds or P04_ROW_TIMEOUT_SECONDS
    return args


def _validate_args(args: argparse.Namespace) -> None:
    positive_float_fields = (
        "epsilon",
        "baseline_scaling",
        "baseline_convergence_threshold",
        "nystrom_convergence_threshold",
        "denominator_floor",
    )
    for field in positive_float_fields:
        if float(getattr(args, field)) <= 0.0:
            raise ValueError(f"{field.replace('_', '-')} must be positive")
    if args.baseline_scaling > 1.0:
        raise ValueError("baseline-scaling must be in (0, 1]")
    if args.baseline_max_iterations <= 0 or args.nystrom_max_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    if args.baseline_row_chunk_size <= 0 or args.baseline_col_chunk_size <= 0:
        raise ValueError("baseline chunk sizes must be positive")
    if args.cholesky_jitter < 0.0:
        raise ValueError("cholesky-jitter must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.mode != "gpu-scale" and args.device_scope != "cpu":
        raise ValueError("small-reference and downstream-smoke modes require --device-scope cpu")
    if args.mode == "gpu-scale" and args.device_scope != "visible":
        raise ValueError("gpu-scale mode requires --device-scope visible")
    if args.row_timeout_seconds <= 0:
        raise ValueError("row-timeout-seconds must be positive")


def _tf_dtype(dtype_name: str) -> tf.DType:
    if dtype_name == "float64":
        return tf.float64
    if dtype_name == "float32":
        return tf.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _configure_tf(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _tf_dtype(args.dtype)
    experimental_batched_ledh_pfpf_ot_tf.DTYPE = dtype
    annealed_transport_tf.DTYPE = dtype
    nystrom_transport_tf.DTYPE = dtype
    nystrom_transport_tf.DEFAULT_DTYPE = dtype
    if args.tf32_mode == "enabled":
        tf.config.experimental.enable_tensor_float_32_execution(True)
    elif args.tf32_mode == "disabled":
        tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        tf32_enabled: bool | str = bool(tf.config.experimental.tensor_float_32_execution_enabled())
    except Exception as exc:  # noqa: BLE001
        tf32_enabled = f"unavailable:{type(exc).__name__}"
    return {
        "dtype": args.dtype,
        "tf_dtype": dtype.name,
        "tf32_requested": args.tf32_mode,
        "tf32_execution_recorded": tf32_enabled,
    }


def _orthonormal_basis(state_dim: int, rank: int) -> np.ndarray:
    grid = np.arange(state_dim, dtype=np.float64)[:, None]
    cols = []
    for k in range(rank):
        cols.append(np.sin((k + 1) * 0.31 * grid[:, 0]) + np.cos((k + 2) * 0.17 * grid[:, 0]))
    q, _ = np.linalg.qr(np.stack(cols, axis=1))
    return q[:, :rank]


def _fixture_tiny_manual() -> tuple[np.ndarray, np.ndarray]:
    particles = np.array(
        [[[-0.45, 0.0, 0.20], [-0.10, 0.15, -0.05], [0.18, -0.18, 0.24], [0.47, 0.05, -0.16]]],
        dtype=np.float64,
    )
    raw = np.array([[0.0, -0.35, -0.75, -1.2]], dtype=np.float64)
    return particles, raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))


def _fixture_small_parity() -> tuple[np.ndarray, np.ndarray]:
    batch_size, num_particles, state_dim = 2, 8, 5
    i = np.arange(num_particles, dtype=np.float64)
    d = np.arange(state_dim, dtype=np.float64)
    b = np.arange(batch_size, dtype=np.float64)
    particles = (
        0.20 * np.sin(0.37 * i[None, :, None] + 0.19 * d[None, None, :])
        + 0.11 * np.cos(0.23 * i[None, :, None] * (d[None, None, :] + 1.0))
        + 0.015 * b[:, None, None]
    )
    raw = -0.06 * i[None, :] + 0.08 * np.sin(0.4 * i[None, :] + 0.3 * b[:, None])
    return particles.astype(np.float64), (raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))).astype(np.float64)


def _fixture_high_dim_low_rank() -> tuple[np.ndarray, np.ndarray]:
    num_particles, state_dim, latent_rank = 32, 32, 4
    i = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    coeffs = np.stack(
        [i, i * i - np.mean(i * i), np.sin(np.pi * i), np.cos(0.5 * np.pi * i) - np.mean(np.cos(0.5 * np.pi * i))],
        axis=1,
    )
    particles = (coeffs @ _orthonormal_basis(state_dim, latent_rank).T)[None, :, :]
    raw = -0.04 * np.arange(num_particles, dtype=np.float64)[None, :]
    return particles.astype(np.float64), (raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))).astype(np.float64)


def _fixture_ledh_specific_smoke() -> tuple[np.ndarray, np.ndarray]:
    num_particles, state_dim, latent_dim = 32, 12, 3
    t = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    latent = np.stack([t, np.sin(1.3 * np.pi * t), np.cos(0.7 * np.pi * t) - np.mean(np.cos(0.7 * np.pi * t))], axis=1)
    grid = np.arange(state_dim, dtype=np.float64)[:, None]
    basis_raw = np.concatenate(
        [
            np.sin(0.23 * (grid + 1.0) * (np.arange(latent_dim, dtype=np.float64)[None, :] + 1.0)),
            np.cos(0.17 * (grid + 1.0) * (np.arange(latent_dim, dtype=np.float64)[None, :] + 2.0)),
        ],
        axis=1,
    )
    basis = np.linalg.qr(basis_raw)[0][:, :latent_dim]
    particles = latent @ basis.T
    shear = np.zeros_like(particles)
    shear[:, 0] = 0.12 * particles[:, 1] * particles[:, 2]
    shear[:, 1] = 0.08 * particles[:, 0] ** 2
    harmonic = 0.025 * np.sin(2.7 * t[:, None] + 0.31 * np.arange(state_dim, dtype=np.float64)[None, :])
    cluster_shift = np.where(np.arange(num_particles)[:, None] < num_particles // 2, -0.08, 0.08)
    particles = particles + shear + harmonic + cluster_shift * basis[:, 0][None, :]
    raw = -0.05 * np.arange(num_particles, dtype=np.float64)[None, :]
    raw += 0.11 * np.sin(1.7 * t[None, :]) - 0.04 * np.cos(2.3 * t[None, :])
    return particles[None, :, :].astype(np.float64), (raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))).astype(np.float64)


def _small_reference_fixtures() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fixtures = {
        "tiny_manual": _fixture_tiny_manual(),
        "small_parity": _fixture_small_parity(),
        "high_dim_low_rank": _fixture_high_dim_low_rank(),
        "ledh_specific_smoke": _fixture_ledh_specific_smoke(),
    }
    for name, (particles, _log_weights) in fixtures.items():
        expected_n, _ranks = SMALL_REFERENCE_SPECS[name]
        if int(particles.shape[1]) != expected_n:
            raise ValueError(f"{name} particle count mismatch: got {particles.shape[1]}, expected {expected_n}")
    return fixtures


def _selected_fixtures(spec: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fixtures = _small_reference_fixtures()
    if spec == "all":
        return fixtures
    names = [item.strip() for item in spec.split(",") if item.strip()]
    missing = sorted(set(names).difference(fixtures))
    if missing:
        raise ValueError(f"unknown fixtures: {missing}")
    return {name: fixtures[name] for name in names}


def _rank_list(spec: str, fixture_name: str) -> list[int]:
    planned = list(SMALL_REFERENCE_SPECS[fixture_name][1])
    if spec == "plan":
        return planned
    ranks = sorted({int(item) for item in spec.split(",") if item.strip()})
    invalid = [rank for rank in ranks if rank not in planned]
    if invalid:
        raise ValueError(f"ranks outside reviewed plan for {fixture_name}: {invalid}")
    return ranks


def _row_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.row_spec:
        rows = []
        for item in args.row_spec.split(","):
            n, rank, time_steps, state_dim, obs_dim = [int(part) for part in item.split(":")]
            rows.append(
                {
                    "fixture_id": f"custom_n{n}_rank{rank}",
                    "particle_count": n,
                    "rank": rank,
                    "time_steps": time_steps,
                    "state_dim": state_dim,
                    "obs_dim": obs_dim,
                    "required": True,
                }
            )
        return rows
    if args.mode == "downstream-smoke":
        return [dict(row, required=True) for row in DOWNSTREAM_ROWS]
    rows = [dict(row) for row in GPU_REQUIRED_ROWS]
    return rows


def _stable_lgssm_fixture(
    *,
    seed: int,
    batch_size: int,
    particle_count: int,
    time_steps: int,
    state_dim: int,
    obs_dim: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + particle_count + 17 * state_dim + 29 * obs_dim)
    batch = np.arange(batch_size, dtype=np.float64)
    particle_grid = np.linspace(-1.0, 1.0, particle_count, dtype=np.float64)
    state_grid = np.linspace(-0.5, 0.5, state_dim, dtype=np.float64)
    initial_particles = (
        0.045 * rng.standard_normal((batch_size, particle_count, state_dim))
        + 0.035 * np.sin(particle_grid[None, :, None] * (np.arange(state_dim)[None, None, :] + 1.0))
        + 0.02 * state_grid[None, None, :]
        + 0.0001 * batch[:, None, None]
    )
    diagonal = 0.72 + 0.04 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    transition_matrix = np.zeros((batch_size, state_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        transition_matrix[row] = np.diag(diagonal + 0.00001 * row)
        transition_matrix[row] += 0.003 * np.eye(state_dim, k=1, dtype=np.float64)
        transition_matrix[row] -= 0.002 * np.eye(state_dim, k=-1, dtype=np.float64)
    q_diag = 0.30 + 0.02 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    r_diag = 0.45 + 0.02 * np.linspace(0.0, 1.0, obs_dim, dtype=np.float64)
    transition_covariance = np.tile(np.diag(q_diag)[None, :, :], (batch_size, 1, 1))
    observation_covariance = np.tile(np.diag(r_diag)[None, :, :], (batch_size, 1, 1))
    observation_matrix = np.zeros((batch_size, obs_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        for obs_index in range(obs_dim):
            state_index = obs_index % state_dim
            observation_matrix[row, obs_index, state_index] = 0.55
            if state_dim > 1:
                observation_matrix[row, obs_index, (state_index + 1) % state_dim] = 0.05
    transitioned_initial = np.einsum("bnj,bdj->bnd", initial_particles, transition_matrix)
    time_grid = np.arange(time_steps, dtype=np.float64)
    time_wave = 0.01 * np.sin(0.31 * (time_grid[:, None] + 1.0) * (np.arange(state_dim) + 1.0))
    particle_wave = 0.004 * np.cos(0.7 * particle_grid[:, None] * (np.arange(state_dim) + 1.0))
    pre_flow_particles = transitioned_initial[:, None, :, :] + time_wave[None, :, None, :] + particle_wave[None, None, :, :]
    observations = 0.03 * np.sin(0.23 * (time_grid[:, None] + 1.0) * (np.arange(obs_dim, dtype=np.float64)[None, :] + 1.0))
    observations += 0.01 * np.cos(0.17 * (time_grid[:, None] + 1.0) * (np.arange(obs_dim, dtype=np.float64)[None, :] + 2.0))
    return {
        "observations": observations,
        "initial_particles": initial_particles,
        "pre_flow_particles": pre_flow_particles,
        "fixed_resampling_mask": np.ones((batch_size, time_steps), dtype=bool),
        "transition_matrix": transition_matrix,
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
        "observation_matrix": observation_matrix,
    }


def _to_tensors(fixture: dict[str, np.ndarray], dtype: tf.DType) -> dict[str, tf.Tensor]:
    return {name: tf.constant(value, dtype=tf.bool if value.dtype == np.bool_ else dtype) for name, value in fixture.items()}


def _make_observation_fn(observation_matrix: tf.Tensor):
    def _observation(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bmd,bnd->bnm", observation_matrix, points)

    return _observation


def _make_observation_jacobian_fn(observation_matrix: tf.Tensor):
    def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
        batch_size = points.shape[0]
        num_particles = points.shape[1]
        if batch_size is None or num_particles is None:
            raise ValueError("Nystrom harness requires static batch and particle dimensions")
        return tf.tile(observation_matrix[:, None, :, :], [1, num_particles, 1, 1])

    return _observation_jacobian


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals)))
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(residuals.shape[-1], dtype)
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * np.pi, dtype)) + logdet[:, None] + quad)


def _make_transition_log_density(transition_matrix: tf.Tensor, transition_covariance: tf.Tensor, dtype: tf.DType):
    def _transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        del _time_index
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        return _batched_gaussian_logpdf(x_next - mean, transition_covariance, dtype)

    return _transition_log_density


def _make_observation_log_density(observation_matrix: tf.Tensor, observation_covariance: tf.Tensor, dtype: tf.DType):
    def _observation_log_density(x: tf.Tensor, observation: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        del _time_index
        predicted = tf.einsum("bmd,bnd->bnm", observation_matrix, x)
        return _batched_gaussian_logpdf(predicted - observation[None, None, :], observation_covariance, dtype)

    return _observation_log_density


def _float(value: Any) -> float:
    return float(tf.reshape(tf.cast(value, tf.float64), [-1])[0].numpy())


def _bool(value: Any) -> bool:
    return bool(tf.reduce_all(tf.cast(value, tf.bool)).numpy())


def _tensor_finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _device_metadata() -> dict[str, Any]:
    return {
        "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
    }


def _phase_result_path(args: argparse.Namespace) -> str:
    if args.phase_result_path:
        return args.phase_result_path
    return {"small-reference": P02_RESULT_PATH, "downstream-smoke": P03_RESULT_PATH, "gpu-scale": P04_RESULT_PATH}[args.mode]


def _common_fields() -> dict[str, Any]:
    return {
        "algorithm_family": ALGORITHM_FAMILY,
        "source_route": SOURCE_ROUTE,
        "source_route_components": dict(SOURCE_ROUTE_COMPONENTS),
        "semantic_class": SEMANTIC_CLASS,
        "baseline_comparator": BASELINE_COMPARATOR,
        "transport_object_kind": TRANSPORT_OBJECT_KIND,
        "transport_matrix_materialized": TRANSPORT_MATRIX_MATERIALIZED,
        "nonclaims": list(NONCLAIMS),
    }


def _run_manifest(
    args: argparse.Namespace,
    *,
    started_at: str,
    ended_at: str,
    wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git_output(["git", "status", "--short"]),
        "command": " ".join(sys.argv),
        "working_directory": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": getattr(tfp, "__version__", "unavailable"),
        "device_scope": args.device_scope,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "trust_context": args.trust_context,
        "seed": args.seed,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_seconds": wall_time_seconds,
        "artifact_paths": {"json": args.output, "markdown": args.markdown_output},
        "plan_path": PLAN_PATH,
        "phase_result_path": _phase_result_path(args),
        **_device_metadata(),
    }


def _run_dense_reference(particles_np: np.ndarray, log_weights_np: np.ndarray, args: argparse.Namespace, dtype: tf.DType) -> tuple[dict[str, Any], tf.Tensor]:
    particles = tf.constant(particles_np, dtype=dtype)
    log_weights = tf.constant(log_weights_np, dtype=dtype)
    start = time.perf_counter()
    with tf.device(args.device):
        result = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=args.epsilon,
            scaling=args.baseline_scaling,
            convergence_threshold=args.baseline_convergence_threshold,
            max_iterations=args.baseline_max_iterations,
            transport_gradient_mode="raw",
            transport_plan_mode="dense",
            row_chunk_size=args.baseline_row_chunk_size,
            col_chunk_size=args.baseline_col_chunk_size,
        )
    wall_time = time.perf_counter() - start
    transport = tf.convert_to_tensor(result.transport_matrix, dtype=dtype)
    source_weights = tf.exp(log_weights)
    n = tf.cast(tf.shape(log_weights)[1], dtype)
    return {
        "transport_object_kind": "dense_matrix",
        "transport_matrix_materialized": True,
        "transport_matrix_shape": transport.shape.as_list(),
        "particles_shape": result.particles.shape.as_list(),
        "finite_particles": _tensor_finite(result.particles),
        "finite_transport_matrix": _tensor_finite(transport),
        "max_row_residual": _float(tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=2) - 1.0))),
        "max_column_residual": _float(tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=1) - source_weights * n))),
        "wall_time_seconds_explanatory": wall_time,
    }, tf.convert_to_tensor(result.particles, dtype=dtype)


def _run_nystrom_reference_row(
    particles_np: np.ndarray,
    log_weights_np: np.ndarray,
    args: argparse.Namespace,
    *,
    fixture_name: str,
    rank: int,
    dense_particles: tf.Tensor,
    dtype: tf.DType,
) -> dict[str, Any]:
    particles = tf.constant(particles_np, dtype=dtype)
    log_weights = tf.constant(log_weights_np, dtype=dtype)
    start = time.perf_counter()
    with tf.device(args.device):
        result = nystrom_transport_resample_tf(
            particles,
            log_weights,
            rank=rank,
            epsilon=args.epsilon,
            max_iterations=args.nystrom_max_iterations,
            convergence_threshold=args.nystrom_convergence_threshold,
            cholesky_jitter=args.cholesky_jitter,
            denominator_floor=args.denominator_floor,
        )
    wall_time = time.perf_counter() - start
    candidate_particles = tf.convert_to_tensor(result.particles, dtype=dtype)
    diff = candidate_particles - dense_particles
    max_error = _float(tf.reduce_max(tf.abs(diff)))
    rms_error = _float(tf.sqrt(tf.reduce_mean(tf.square(diff))))
    diagnostics = dict(result.diagnostics)
    transport_shape = result.transport_matrix.shape.as_list()
    hard_vetoes = []
    if transport_shape[-2:] != [0, 0] or bool(diagnostics.get("transport_matrix_materialized")):
        hard_vetoes.append("candidate_transport_matrix_materialized")
    if not diagnostics.get("finite_particles"):
        hard_vetoes.append("nonfinite_particles")
    if not diagnostics.get("finite_factors"):
        hard_vetoes.append("nonfinite_factors")
    if float(diagnostics["max_row_residual"]) > ROW_COLUMN_THRESHOLD:
        hard_vetoes.append("row_residual_threshold")
    if float(diagnostics["max_column_residual"]) > ROW_COLUMN_THRESHOLD:
        hard_vetoes.append("column_residual_threshold")
    if max_error > DENSE_MAX_ERROR_THRESHOLD:
        hard_vetoes.append("dense_reference_max_error_threshold")
    if rms_error > DENSE_RMS_ERROR_THRESHOLD:
        hard_vetoes.append("dense_reference_rms_error_threshold")
    return {
        "fixture": fixture_name,
        "rank": rank,
        "landmark_indices": diagnostics["landmark_indices"],
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "particles_shape": candidate_particles.shape.as_list(),
        "transport_matrix_shape": transport_shape,
        "transport_object_kind": diagnostics["transport_object_kind"],
        "transport_matrix_materialized": bool(diagnostics["transport_matrix_materialized"]),
        "finite_particles": bool(diagnostics["finite_particles"]),
        "finite_factors": bool(diagnostics["finite_factors"]),
        "max_row_residual": float(diagnostics["max_row_residual"]),
        "max_column_residual": float(diagnostics["max_column_residual"]),
        "dense_reference_max_abs_particle_error": max_error,
        "dense_reference_rms_particle_error": rms_error,
        "factor_shapes": diagnostics["factor_shapes"],
        "wall_time_seconds_explanatory": wall_time,
        "memory_maxrss_kb_explanatory": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "diagnostics": diagnostics,
    }


def _run_small_reference(args: argparse.Namespace, dtype: tf.DType) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    fixtures_out = {}
    for fixture_name, (particles_np, log_weights_np) in _selected_fixtures(args.fixtures).items():
        dense, dense_particles = _run_dense_reference(particles_np, log_weights_np, args, dtype)
        ranks = _rank_list(args.ranks, fixture_name)
        nystrom_rows = [
            _run_nystrom_reference_row(
                particles_np,
                log_weights_np,
                args,
                fixture_name=fixture_name,
                rank=rank,
                dense_particles=dense_particles,
                dtype=dtype,
            )
            for rank in ranks
        ]
        rows.extend(nystrom_rows)
        fixtures_out[fixture_name] = {
            "input_shape": {"particles": list(particles_np.shape), "log_weights": list(log_weights_np.shape)},
            "planned_particle_count": SMALL_REFERENCE_SPECS[fixture_name][0],
            "rank_grid": ranks,
            "dense_reference": dense,
            "nystrom": nystrom_rows,
        }
    return rows, {"fixtures": fixtures_out}


def _run_filter_loop(tensors: dict[str, tf.Tensor], args: argparse.Namespace, *, row_spec: dict[str, Any], dtype: tf.DType) -> dict[str, Any]:
    observations = tensors["observations"]
    particles = tensors["initial_particles"]
    pre_flow_particles = tensors["pre_flow_particles"]
    fixed_mask = tensors["fixed_resampling_mask"]
    transition_matrix = tensors["transition_matrix"]
    transition_covariance = tensors["transition_covariance"]
    observation_covariance = tensors["observation_covariance"]
    observation_matrix = tensors["observation_matrix"]
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    time_steps = int(observations.shape[0])
    log_weights = uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=dtype)
    means = []
    variances = []
    esses = []
    step_diagnostics: list[dict[str, Any]] = []
    transport_shapes: list[list[int]] = []
    transition_log_density = _make_transition_log_density(transition_matrix, transition_covariance, dtype)
    observation_log_density = _make_observation_log_density(observation_matrix, observation_covariance, dtype)
    mask_np = fixed_mask.numpy()
    for t in range(time_steps):
        if not bool(np.all(mask_np[:, t])):
            raise ValueError("Nystrom harness expects active resampling at every planned step")
        observation = observations[t]
        ancestors = particles
        pre_flow = pre_flow_particles[:, t, :, :]
        flow = batched_ledh_flow_core_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=_make_observation_fn(observation_matrix),
            observation_jacobian_fn=_make_observation_jacobian_fn(observation_matrix),
            observation_residual_fn=_observation_residual,
        )
        post_flow = flow.post_flow_particles
        corrected_log_weights = (
            log_weights
            + transition_log_density(post_flow, ancestors, tf.constant(t, dtype=tf.int32))
            + observation_log_density(post_flow, observation, tf.constant(t, dtype=tf.int32))
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        means.append(mean)
        variances.append(variance)
        esses.append(1.0 / tf.reduce_sum(weights * weights, axis=1))
        resample = nystrom_transport_resample_tf(
            post_flow,
            tf.math.log(tf.maximum(weights, _log_weight_floor())),
            rank=int(row_spec["rank"]),
            epsilon=args.epsilon,
            max_iterations=args.nystrom_max_iterations,
            convergence_threshold=args.nystrom_convergence_threshold,
            cholesky_jitter=args.cholesky_jitter,
            denominator_floor=args.denominator_floor,
        )
        step_diagnostics.append(dict(resample.diagnostics))
        transport_shapes.append(resample.transport_matrix.shape.as_list())
        particles = tf.convert_to_tensor(resample.particles, dtype=dtype)
        log_weights = tf.convert_to_tensor(resample.log_weights, dtype=dtype)
    filtered_means = tf.stack(means, axis=0)
    filtered_variances = tf.stack(variances, axis=0)
    ess_by_time = tf.stack(esses, axis=0)
    return {
        "log_likelihood": log_likelihood,
        "filtered_means": filtered_means,
        "filtered_variances": filtered_variances,
        "ess_by_time": ess_by_time,
        "final_particles": particles,
        "final_log_weights": log_weights,
        "step_diagnostics": step_diagnostics,
        "transport_matrix_shapes": transport_shapes,
        "output_log_weight_normalization_residual": _float(tf.reduce_max(tf.abs(tf.reduce_logsumexp(log_weights, axis=1)))),
        "ess_fraction_min": _float(tf.reduce_min(ess_by_time) / tf.cast(num_particles, dtype)),
        "finite_log_likelihood": _bool(tf.math.is_finite(log_likelihood)),
        "finite_filtered_means": _bool(tf.math.is_finite(filtered_means)),
        "finite_filtered_variances": _bool(tf.math.is_finite(filtered_variances)),
        "finite_ess_by_time": _bool(tf.math.is_finite(ess_by_time)),
        "finite_final_particles": _bool(tf.math.is_finite(particles)),
        "finite_final_log_weights": _bool(tf.math.is_finite(log_weights)),
        "filtered_mean_abs_max_explanatory": _float(tf.reduce_max(tf.abs(filtered_means))),
        "filtered_variance_abs_max_explanatory": _float(tf.reduce_max(tf.abs(filtered_variances))),
    }


def _max_diag(step_diagnostics: list[dict[str, Any]], key: str) -> float | None:
    values = [diag.get(key) for diag in step_diagnostics if diag.get(key) is not None]
    return max(float(value) for value in values) if values else None


def _all_diag(step_diagnostics: list[dict[str, Any]], key: str) -> bool | None:
    values = [diag.get(key) for diag in step_diagnostics if diag.get(key) is not None]
    return all(bool(value) for value in values) if values else None


def _run_downstream_row(args: argparse.Namespace, row_spec: dict[str, Any], dtype: tf.DType) -> dict[str, Any]:
    fixture = _stable_lgssm_fixture(
        seed=args.seed,
        batch_size=args.batch_size,
        particle_count=int(row_spec["particle_count"]),
        time_steps=int(row_spec["time_steps"]),
        state_dim=int(row_spec["state_dim"]),
        obs_dim=int(row_spec["obs_dim"]),
    )
    tensors = _to_tensors(fixture, dtype)
    start = time.perf_counter()
    with tf.device(args.device):
        loop = _run_filter_loop(tensors, args, row_spec=row_spec, dtype=dtype)
    wall_time = time.perf_counter() - start
    step_diags = loop["step_diagnostics"]
    max_row = _max_diag(step_diags, "max_row_residual")
    max_col = _max_diag(step_diags, "max_column_residual")
    hard_vetoes: list[str] = []
    for key in (
        "finite_log_likelihood",
        "finite_filtered_means",
        "finite_filtered_variances",
        "finite_ess_by_time",
        "finite_final_particles",
        "finite_final_log_weights",
    ):
        if not loop[key]:
            hard_vetoes.append(key.replace("finite_", "nonfinite_"))
    if _all_diag(step_diags, "finite_factors") is False:
        hard_vetoes.append("nonfinite_factors")
    if _all_diag(step_diags, "finite_particles") is False:
        hard_vetoes.append("nonfinite_resampled_particles")
    if _all_diag(step_diags, "transport_matrix_materialized") is not False:
        hard_vetoes.append("candidate_transport_matrix_materialized")
    if max_row is None or max_row > ROW_COLUMN_THRESHOLD:
        hard_vetoes.append("row_residual_threshold")
    if max_col is None or max_col > ROW_COLUMN_THRESHOLD:
        hard_vetoes.append("column_residual_threshold")
    if loop["output_log_weight_normalization_residual"] > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
        hard_vetoes.append("output_log_weight_normalization_threshold")
    if loop["ess_fraction_min"] < ESS_FRACTION_THRESHOLD:
        hard_vetoes.append("ess_fraction_threshold")
    if any(shape[-2:] != [0, 0] for shape in loop["transport_matrix_shapes"]):
        hard_vetoes.append("transport_matrix_shape_not_sentinel")
    return {
        **row_spec,
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "landmark_indices": [diag.get("landmark_indices") for diag in step_diags],
        "transport_matrix_shapes": loop["transport_matrix_shapes"],
        "transport_object_kind": TRANSPORT_OBJECT_KIND,
        "transport_matrix_materialized": False,
        "max_row_residual": max_row,
        "max_column_residual": max_col,
        "output_log_weight_normalization_residual": loop["output_log_weight_normalization_residual"],
        "ess_fraction_min": loop["ess_fraction_min"],
        "finite_log_likelihood": loop["finite_log_likelihood"],
        "finite_filtered_means": loop["finite_filtered_means"],
        "finite_filtered_variances": loop["finite_filtered_variances"],
        "finite_ess_by_time": loop["finite_ess_by_time"],
        "finite_final_particles": loop["finite_final_particles"],
        "finite_final_log_weights": loop["finite_final_log_weights"],
        "all_finite_factors": _all_diag(step_diags, "finite_factors"),
        "all_finite_resampled_particles": _all_diag(step_diags, "finite_particles"),
        "filtered_mean_abs_max_explanatory": loop["filtered_mean_abs_max_explanatory"],
        "filtered_variance_abs_max_explanatory": loop["filtered_variance_abs_max_explanatory"],
        "wall_time_seconds_explanatory": wall_time,
        "memory_maxrss_kb_explanatory": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "step_diagnostics": step_diags,
    }


def _run_downstream_or_gpu(args: argparse.Namespace, dtype: tf.DType) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    started = time.perf_counter()
    for row_spec in _row_specs(args):
        rows.append(_run_downstream_row(args, row_spec, dtype))
    if args.mode == "gpu-scale" and all(row["status"] == "PASS" for row in rows):
        elapsed = time.perf_counter() - started
        if elapsed <= P04_OPTIONAL_ROW_ELAPSED_LIMIT_SECONDS:
            rows.append(_run_downstream_row(args, dict(GPU_OPTIONAL_ROW), dtype))
    return rows, {"row_specs": rows}


def _summary(rows: list[dict[str, Any]], hard_vetoes: list[str], wall_time: float) -> dict[str, Any]:
    def _max(key: str) -> float | int | None:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        return max(values) if values else None

    return {
        "num_rows": len(rows),
        "num_passed_rows": sum(1 for row in rows if row.get("status") == "PASS"),
        "num_hard_vetoes": len(hard_vetoes),
        "max_row_residual": _max("max_row_residual"),
        "max_column_residual": _max("max_column_residual"),
        "max_output_log_weight_normalization_residual": _max("output_log_weight_normalization_residual"),
        "min_ess_fraction": min((float(row["ess_fraction_min"]) for row in rows if row.get("ess_fraction_min") is not None), default=None),
        "max_dense_reference_particle_error": _max("dense_reference_max_abs_particle_error"),
        "max_dense_reference_rms_error": _max("dense_reference_rms_particle_error"),
        "total_wall_time_seconds_explanatory": wall_time,
        "max_memory_maxrss_kb_explanatory": _max("memory_maxrss_kb_explanatory"),
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    tf.random.set_seed(args.seed)
    tf_metadata = _configure_tf(args)
    dtype = _tf_dtype(args.dtype)
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    start = time.perf_counter()
    if args.mode == "small-reference":
        rows, details = _run_small_reference(args, dtype)
    else:
        rows, details = _run_downstream_or_gpu(args, dtype)
    if args.mode == "small-reference":
        hard_vetoes = []
        by_fixture: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_fixture.setdefault(row["fixture"], []).append(row)
        selected_fixture_names = list(_selected_fixtures(args.fixtures))
        for fixture_name in selected_fixture_names:
            if not any(row["status"] == "PASS" for row in by_fixture.get(fixture_name, [])):
                hard_vetoes.append(f"{fixture_name}:no_viable_rank")
        details_fixtures = details.get("fixtures", {})
        for fixture_name in selected_fixture_names:
            dense = details_fixtures.get(fixture_name, {}).get("dense_reference", {})
            if not dense.get("finite_particles", False):
                hard_vetoes.append(f"{fixture_name}:dense_reference_nonfinite_particles")
            if not dense.get("finite_transport_matrix", False):
                hard_vetoes.append(f"{fixture_name}:dense_reference_nonfinite_transport")
            if dense.get("transport_matrix_materialized") is not True:
                hard_vetoes.append(f"{fixture_name}:dense_reference_missing_materialized_matrix")
    else:
        hard_vetoes = [
            f"{row.get('fixture', row.get('fixture_id'))}:rank_{row.get('rank')}:{veto}"
            for row in rows
            for veto in row.get("hard_vetoes", [])
            if row.get("required", True) or args.mode != "gpu-scale"
        ]
    if args.mode == "gpu-scale":
        if args.dtype == "float32" and tf_metadata["tf32_execution_recorded"] is not True:
            hard_vetoes.append("tf32_not_recorded_enabled_for_float32")
        if not _device_metadata()["logical_gpus"]:
            hard_vetoes.append("gpu_device_evidence_missing")
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    wall_time = time.perf_counter() - start
    manifest = {
        **_run_manifest(args, started_at=started_at, ended_at=ended_at, wall_time_seconds=wall_time),
        **tf_metadata,
    }
    return {
        **_common_fields(),
        "mode": args.mode,
        "status": "PASS" if not hard_vetoes else "FAIL",
        "phase": {"small-reference": "P02", "downstream-smoke": "P03", "gpu-scale": "P04"}[args.mode],
        "hard_vetoes": hard_vetoes,
        "run_manifest": manifest,
        "thresholds": {
            "row_column_residual": ROW_COLUMN_THRESHOLD,
            "dense_reference_max_abs_particle_error": DENSE_MAX_ERROR_THRESHOLD,
            "dense_reference_rms_particle_error": DENSE_RMS_ERROR_THRESHOLD,
            "output_log_weight_normalization": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction_min": ESS_FRACTION_THRESHOLD,
            "p04_row_timeout_seconds": P04_ROW_TIMEOUT_SECONDS,
            "p04_phase_timeout_seconds": P04_PHASE_TIMEOUT_SECONDS,
        },
        "rows": rows,
        "summary": _summary(rows, hard_vetoes, wall_time),
        "details": details,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy().tolist())
    return value


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Nystrom LEDH/PFPF-OT Algorithm-Complete Diagnostics",
        "",
        f"- Status: `{result['status']}`",
        f"- Mode: `{result['mode']}`",
        f"- Algorithm family: `{result['algorithm_family']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in result["summary"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Row | Rank | Status | Hard vetoes | Row residual | Column residual | ESS fraction | Dense max error | Sentinel shapes |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["rows"]:
        row_id = row.get("fixture", row.get("fixture_id"))
        lines.append(
            f"| `{row_id}` | `{row.get('rank')}` | `{row.get('status')}` | `{row.get('hard_vetoes', [])}` | "
            f"`{row.get('max_row_residual')}` | `{row.get('max_column_residual')}` | "
            f"`{row.get('ess_fraction_min')}` | `{row.get('dense_reference_max_abs_particle_error')}` | "
            f"`{row.get('transport_matrix_shapes', row.get('transport_matrix_shape'))}` |"
        )
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = build_result(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.markdown_output))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

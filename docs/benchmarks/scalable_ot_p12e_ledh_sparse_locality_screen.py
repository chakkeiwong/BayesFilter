"""P12E LEDH sparse-locality screen for scalable OT planning.

This diagnostic generates deterministic LEDH-like post-flow particle fixtures,
then measures dense transport locality and 99% row-truncation behavior using
the Phase 8 sparse-locality semantics.

It is not a sparse solver and it does not make speedup, ranking, posterior
correctness, HMC readiness, public API readiness, production/default readiness,
or broad sparse-OT validity claims.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
try:
    import resource
except ImportError:  # pragma: no cover - platform fallback
    resource = None  # type: ignore[assignment]
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
elif _PRE_ARGS.device_scope == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)


DTYPE = tf.float64
annealed_transport_tf.DTYPE = DTYPE

MASS_THRESHOLDS = (0.90, 0.95, 0.99, 0.999)
TRUNCATION_MASS_THRESHOLD = 0.99
ROW_RESIDUAL_THRESHOLD = 5.0e-3
COLUMN_RESIDUAL_THRESHOLD = 5.0e-2
PARTICLE_ERROR_THRESHOLD = 5.0e-2

FINAL_STATUS_REOPENS = (
    "LEDH_SPARSE_LOCALITY_SCREEN_COMPLETED_REOPENS_SPARSE_IMPLEMENTATION_PLAN_ONLY"
)
FINAL_STATUS_BLOCKS = (
    "LEDH_SPARSE_LOCALITY_SCREEN_COMPLETED_DOES_NOT_REOPEN_SPARSE_IMPLEMENTATION"
)
FINAL_STATUS_BLOCKED = "LEDH_SPARSE_LOCALITY_SCREEN_BLOCKED"

NONCLAIMS = (
    "P12E LEDH sparse-locality screen diagnostic only",
    "not a sparse solver implementation",
    "no sparse solver validity claim",
    "no sparse speedup claim",
    "no ranking claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no production default change",
    "no general sparse-OT validation or rejection",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--scaling", type=float, default=0.9)
    parser.add_argument("--convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument("--col-chunk-size", type=int, default=8)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if args.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < args.scaling <= 1.0:
        raise ValueError("scaling must be in (0, 1]")
    if args.convergence_threshold <= 0.0:
        raise ValueError("convergence_threshold must be positive")
    if args.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    return args


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tensor_finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _logsumexp(values: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    summed = np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True)) + max_value
    if keepdims:
        return summed
    return np.squeeze(summed, axis=axis)


def _gaussian_logpdf_np(residuals: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    chol = np.linalg.cholesky(covariance)
    flat = residuals.reshape([-1, covariance.shape[0]])
    solved = np.linalg.solve(chol, flat.T)
    quad = np.sum(solved * solved, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    dim = covariance.shape[0]
    logpdf = -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)
    return logpdf.reshape(residuals.shape[:-1])


def _array_digest(arrays: dict[str, np.ndarray], settings: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(_json_ready(settings), sort_keys=True).encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(json.dumps(list(array.shape)).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _peak_rss_kb() -> int | None:
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _artifact_scope(path: Path) -> str:
    path_text = str(path)
    if "smoke" in path.name or path_text.startswith("/tmp/"):
        return "smoke"
    return "official"


def _next_phase_handoff(scope: str) -> dict[str, str]:
    if scope == "smoke":
        return {
            "next_action": "Run P12E-3 official diagnostic under the reviewed subplan.",
            "next_evidence_needed": "P12E-3 official diagnostic artifact validation.",
            "scope_nonclaim": "Smoke metrics are structural/runtime validation only and are not official P12E evidence.",
        }
    return {
        "next_action": "Use the reviewed P12E-4 closeout subplan to map this artifact to the final lane result.",
        "next_evidence_needed": "P12E-4 closeout under the reviewed evidence contract.",
        "scope_nonclaim": "Official diagnostic metrics require P12E-4 closeout before final lane interpretation.",
    }


def _particle_summary(particles: np.ndarray) -> dict[str, Any]:
    centered = particles - np.mean(particles, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    return {
        "finite": bool(np.all(np.isfinite(particles))),
        "shape": list(particles.shape),
        "max_abs": float(np.max(np.abs(particles))),
        "mean_radius": float(np.mean(norms)),
        "max_radius": float(np.max(norms)),
        "coordinate_mean": np.mean(particles, axis=0).tolist(),
        "coordinate_std": np.std(particles, axis=0).tolist(),
    }


def _stable_prefix_count(row: np.ndarray, threshold: float) -> int:
    row_mass = float(np.sum(row))
    if row_mass <= 0.0:
        return 0
    order = np.argsort(-row, kind="mergesort")
    cumulative = np.cumsum(row[order])
    index = int(np.searchsorted(cumulative, threshold * row_mass, side="left"))
    return min(index + 1, row.shape[0])


def _support_summary(counts: np.ndarray, n_particles: int) -> dict[str, Any]:
    flat = counts.reshape(-1)
    return {
        "median": float(np.median(flat)),
        "p90": float(np.percentile(flat, 90.0)),
        "max": int(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median_fraction_of_n": float(np.median(flat) / n_particles),
        "p90_fraction_of_n": float(np.percentile(flat, 90.0) / n_particles),
        "per_row_counts": flat.astype(int).tolist(),
    }


def _support_count_diagnostics(transport: np.ndarray) -> dict[str, Any]:
    batch_size, num_rows, num_cols = transport.shape
    by_threshold: dict[str, Any] = {}
    counts_by_threshold: dict[float, np.ndarray] = {}
    for threshold in MASS_THRESHOLDS:
        counts = np.zeros([batch_size, num_rows], dtype=np.int64)
        for batch in range(batch_size):
            for row_index in range(num_rows):
                counts[batch, row_index] = _stable_prefix_count(
                    transport[batch, row_index, :],
                    threshold,
                )
        counts_by_threshold[threshold] = counts
        by_threshold[f"{threshold:.3f}"] = _support_summary(counts, num_cols)
    return {
        "definition": (
            "For each row, masses are sorted descending with numpy mergesort; "
            "k_i(t) is the first stable prefix whose cumulative mass reaches "
            "t times that row mass; ties are not expanded beyond that prefix."
        ),
        "by_threshold": by_threshold,
        "counts_99": counts_by_threshold[TRUNCATION_MASS_THRESHOLD],
    }


def _nearest_neighbor_mass_diagnostics(
    particles: np.ndarray,
    transport: np.ndarray,
) -> dict[str, Any]:
    batch_size, num_particles, _state_dim = particles.shape
    k_values = sorted(
        {
            1,
            2,
            4,
            8,
            int(math.ceil(0.25 * num_particles)),
            int(math.ceil(0.50 * num_particles)),
            num_particles,
        }
    )
    k_values = [k for k in k_values if 1 <= k <= num_particles]
    mass_by_k: dict[str, list[float]] = {str(k): [] for k in k_values}
    for batch in range(batch_size):
        diff = particles[batch, :, None, :] - particles[batch, None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        for row_index in range(num_particles):
            row = transport[batch, row_index, :]
            row_mass = float(np.sum(row))
            order = np.argsort(distances[row_index, :], kind="mergesort")
            for k in k_values:
                captured = float(np.sum(row[order[:k]]))
                fraction = captured / row_mass if row_mass > 0.0 else 0.0
                mass_by_k[str(k)].append(fraction)
    return {
        "definition": (
            "For each target row, source columns are sorted by Euclidean "
            "distance in post-flow particle space; reported values are retained "
            "transport-mass fractions."
        ),
        "k_values": k_values,
        "by_k": {
            key: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "p10": float(np.percentile(values, 10.0)),
                "mean": float(np.mean(values)),
            }
            for key, values in mass_by_k.items()
        },
    }


def _truncate_rows_at_mass(
    transport: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, num_rows, _num_cols = transport.shape
    truncated = np.zeros_like(transport)
    counts = np.zeros([batch_size, num_rows], dtype=np.int64)
    for batch in range(batch_size):
        for row_index in range(num_rows):
            row = transport[batch, row_index, :]
            row_mass = float(np.sum(row))
            count = _stable_prefix_count(row, threshold)
            counts[batch, row_index] = count
            if count == 0:
                continue
            order = np.argsort(-row, kind="mergesort")
            keep = order[:count]
            truncated[batch, row_index, keep] = row[keep]
            retained_mass = float(np.sum(truncated[batch, row_index, :]))
            if retained_mass > 0.0:
                truncated[batch, row_index, :] *= row_mass / retained_mass
    return truncated, counts


def _run_dense(
    particles_np: np.ndarray,
    log_weights_np: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    particles_batched = particles_np[None, :, :]
    log_weights_batched = log_weights_np[None, :]
    particles = tf.constant(particles_batched, dtype=DTYPE)
    log_weights = tf.constant(log_weights_batched, dtype=DTYPE)
    start = time.perf_counter()
    with tf.device(args.device):
        result = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=args.epsilon,
            scaling=args.scaling,
            convergence_threshold=args.convergence_threshold,
            max_iterations=args.max_iterations,
            transport_gradient_mode="raw",
            transport_plan_mode="dense",
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
    wall_time = time.perf_counter() - start
    transport = tf.convert_to_tensor(result.transport_matrix, dtype=DTYPE)
    transported = tf.convert_to_tensor(result.particles, dtype=DTYPE)
    source_weights = tf.exp(log_weights)
    num_particles = tf.cast(tf.shape(log_weights)[1], DTYPE)
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=2) - 1.0))
    column_target = source_weights * num_particles
    column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=1) - column_target))
    dense_record = {
        "wall_time_seconds": wall_time,
        "transport_matrix_shape": transport.shape.as_list(),
        "particles_shape": transported.shape.as_list(),
        "finite_transport_matrix": _tensor_finite(transport),
        "finite_particles": _tensor_finite(transported),
        "max_row_residual": _float(row_residual),
        "max_column_residual": _float(column_residual),
        "diagnostics": dict(result.diagnostics),
    }
    return dense_record, transport.numpy(), transported.numpy()


def _truncation_diagnostics(
    particles: np.ndarray,
    log_weights: np.ndarray,
    transport: np.ndarray,
    dense_particles: np.ndarray,
    counts_99: np.ndarray,
) -> dict[str, Any]:
    truncated, truncation_counts = _truncate_rows_at_mass(
        transport,
        TRUNCATION_MASS_THRESHOLD,
    )
    if not np.array_equal(counts_99, truncation_counts):
        raise AssertionError("99% support counts changed between support and truncation diagnostics")
    num_particles = transport.shape[2]
    source_weights = np.exp(log_weights)[None, :]
    column_target = source_weights * float(num_particles)
    row_residual = float(np.max(np.abs(np.sum(truncated, axis=2) - 1.0)))
    column_residual = float(np.max(np.abs(np.sum(truncated, axis=1) - column_target)))
    truncated_particles = np.matmul(truncated, particles[None, :, :])
    diff = truncated_particles - dense_particles
    return {
        "truncation_mass_threshold": TRUNCATION_MASS_THRESHOLD,
        "definition": (
            "Retain the stable minimal per-row 99% mass support, zero all "
            "other entries, then renormalize each retained row to its original "
            "dense row sum before applying the matrix to post-flow particles."
        ),
        "max_row_residual": row_residual,
        "max_column_residual": column_residual,
        "max_transported_particle_error": float(np.max(np.abs(diff))),
        "rms_transported_particle_error": float(np.sqrt(np.mean(np.square(diff)))),
        "finite_truncated_matrix": bool(np.all(np.isfinite(truncated))),
        "finite_truncated_particles": bool(np.all(np.isfinite(truncated_particles))),
        "nonzero_entries": int(np.count_nonzero(truncated)),
        "dense_entries": int(np.prod(transport.shape)),
        "nonzero_fraction_of_dense": float(np.count_nonzero(truncated) / np.prod(transport.shape)),
    }


def _orthonormal_rows(rows: int, cols: int) -> np.ndarray:
    grid = np.arange(cols, dtype=np.float64)[:, None]
    raw_cols = []
    for idx in range(rows):
        raw_cols.append(
            np.sin((idx + 1) * 0.23 * grid[:, 0])
            + np.cos((idx + 2) * 0.19 * grid[:, 0])
        )
    matrix = np.stack(raw_cols, axis=1)
    q, _ = np.linalg.qr(matrix)
    return q[:, :rows].T


def _linear_observation_functions(
    observation_matrix_np: np.ndarray,
) -> tuple[
    Callable[[tf.Tensor], tf.Tensor],
    Callable[[tf.Tensor], tf.Tensor],
    Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
]:
    observation_matrix = tf.constant(observation_matrix_np, dtype=DTYPE)

    def _observation_fn(point: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(observation_matrix, tf.cast(point, DTYPE))

    def _observation_jacobian_fn(_point: tf.Tensor) -> tf.Tensor:
        return observation_matrix

    def _observation_residual_fn(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tf.cast(observation, DTYPE), [-1]) - tf.reshape(tf.cast(h_ref, DTYPE), [-1])

    return _observation_fn, _observation_jacobian_fn, _observation_residual_fn


def _fixture_tiny_manual() -> dict[str, Any]:
    ancestors = np.array(
        [
            [-0.50, -0.20],
            [-0.35, 0.10],
            [-0.18, -0.05],
            [-0.04, 0.20],
            [0.08, -0.18],
            [0.22, 0.04],
            [0.36, 0.18],
            [0.54, -0.08],
        ],
        dtype=np.float64,
    )
    transition_matrix = np.array([[0.88, 0.05], [-0.03, 0.82]], dtype=np.float64)
    offsets = np.array(
        [
            [-0.04, 0.02],
            [0.03, -0.01],
            [-0.02, -0.04],
            [0.01, 0.03],
            [0.02, -0.02],
            [-0.03, 0.01],
            [0.04, 0.03],
            [-0.01, -0.03],
        ],
        dtype=np.float64,
    )
    pre_flow_particles = ancestors @ transition_matrix.T + offsets
    return {
        "fixture_name": "ledh_lgssm_tiny_manual",
        "purpose": "Small deterministic LEDH sanity fixture.",
        "seed": None,
        "ancestors": ancestors,
        "pre_flow_particles": pre_flow_particles,
        "observation": np.array([0.03, -0.06], dtype=np.float64),
        "transition_matrix": transition_matrix,
        "transition_covariance": np.diag([0.11, 0.15]).astype(np.float64),
        "observation_matrix": np.eye(2, dtype=np.float64),
        "observation_covariance": np.diag([0.08, 0.10]).astype(np.float64),
        "construction": {
            "kind": "closed_form_manual_grid",
            "num_particles": 8,
            "state_dim": 2,
            "observation_dim": 2,
        },
    }


def _fixture_moderate_clustered() -> dict[str, Any]:
    seed = 2026061901
    rng = np.random.default_rng(seed)
    num_particles, state_dim, observation_dim, clusters = 64, 6, 4, 4
    observation_matrix = _orthonormal_rows(observation_dim, state_dim)
    transition_matrix = 0.82 * np.eye(state_dim)
    transition_matrix += 0.025 * np.diag(np.ones(state_dim - 1), k=1)
    transition_matrix -= 0.015 * np.diag(np.ones(state_dim - 1), k=-1)
    centers = np.zeros([clusters, state_dim], dtype=np.float64)
    centers[:, :observation_dim] = np.array(
        [
            [-0.55, -0.30, 0.20, 0.10],
            [-0.18, 0.34, -0.12, 0.22],
            [0.20, -0.22, 0.42, -0.18],
            [0.58, 0.24, -0.22, -0.10],
        ],
        dtype=np.float64,
    )
    cluster_ids = np.arange(num_particles) // (num_particles // clusters)
    within = np.arange(num_particles) % (num_particles // clusters)
    deterministic_offsets = np.zeros([num_particles, state_dim], dtype=np.float64)
    deterministic_offsets[:, 0] = 0.010 * (within - np.mean(within))
    deterministic_offsets[:, 1] = 0.012 * np.sin(0.7 * within)
    deterministic_offsets[:, 2] = 0.010 * np.cos(0.5 * within)
    seeded_offsets = 0.018 * rng.standard_normal([num_particles, state_dim])
    ancestors = centers[cluster_ids] + deterministic_offsets + seeded_offsets
    pre_flow_offsets = 0.020 * rng.standard_normal([num_particles, state_dim])
    pre_flow_particles = ancestors @ transition_matrix.T + pre_flow_offsets
    observation = observation_matrix @ np.mean(centers[[1, 2]], axis=0)
    observation += np.array([0.03, -0.02, 0.015, -0.01], dtype=np.float64)
    return {
        "fixture_name": "ledh_lgssm_moderate_clustered",
        "purpose": "Clustered pre-flow proposals with a linear-Gaussian LEDH contraction.",
        "seed": seed,
        "ancestors": ancestors,
        "pre_flow_particles": pre_flow_particles,
        "observation": observation.astype(np.float64),
        "transition_matrix": transition_matrix.astype(np.float64),
        "transition_covariance": np.diag([0.08, 0.09, 0.10, 0.11, 0.12, 0.13]).astype(np.float64),
        "observation_matrix": observation_matrix.astype(np.float64),
        "observation_covariance": np.diag([0.055, 0.060, 0.065, 0.070]).astype(np.float64),
        "construction": {
            "kind": "seeded_clustered_grid",
            "num_particles": num_particles,
            "state_dim": state_dim,
            "observation_dim": observation_dim,
            "clusters": clusters,
        },
    }


def _fixture_moderate_diffuse() -> dict[str, Any]:
    seed = 2026061902
    rng = np.random.default_rng(seed)
    num_particles, state_dim, observation_dim = 64, 6, 4
    observation_matrix = _orthonormal_rows(observation_dim, state_dim)
    transition_matrix = 0.74 * np.eye(state_dim)
    transition_matrix += 0.035 * np.diag(np.ones(state_dim - 1), k=1)
    transition_matrix -= 0.020 * np.diag(np.ones(state_dim - 1), k=-1)
    grid = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    basis = _orthonormal_rows(4, state_dim).T
    coeffs = np.stack(
        [
            0.95 * grid,
            0.75 * (grid * grid - np.mean(grid * grid)),
            0.55 * np.sin(np.pi * grid),
            0.45 * np.cos(0.5 * np.pi * grid),
        ],
        axis=1,
    )
    ancestors = coeffs @ basis.T + 0.070 * rng.standard_normal([num_particles, state_dim])
    pre_flow_particles = (
        ancestors @ transition_matrix.T
        + 0.110 * rng.standard_normal([num_particles, state_dim])
        + 0.020 * np.sin(np.arange(num_particles, dtype=np.float64)[:, None] * 0.31)
    )
    observation = observation_matrix @ np.mean(ancestors[24:40], axis=0)
    observation += np.array([0.08, -0.06, 0.05, -0.04], dtype=np.float64)
    return {
        "fixture_name": "ledh_lgssm_moderate_diffuse",
        "purpose": "Broader deterministic proposals testing the expected diffuse-support failure mode.",
        "seed": seed,
        "ancestors": ancestors.astype(np.float64),
        "pre_flow_particles": pre_flow_particles.astype(np.float64),
        "observation": observation.astype(np.float64),
        "transition_matrix": transition_matrix.astype(np.float64),
        "transition_covariance": np.diag([0.22, 0.24, 0.26, 0.28, 0.30, 0.32]).astype(np.float64),
        "observation_matrix": observation_matrix.astype(np.float64),
        "observation_covariance": np.diag([0.20, 0.22, 0.24, 0.26]).astype(np.float64),
        "construction": {
            "kind": "seeded_diffuse_low_rank_grid",
            "num_particles": num_particles,
            "state_dim": state_dim,
            "observation_dim": observation_dim,
        },
    }


def _fixture_specs() -> list[dict[str, Any]]:
    return [
        _fixture_tiny_manual(),
        _fixture_moderate_clustered(),
        _fixture_moderate_diffuse(),
    ]


def _run_ledh_fixture(spec: dict[str, Any]) -> dict[str, Any]:
    observation_fn, observation_jacobian_fn, observation_residual_fn = _linear_observation_functions(
        spec["observation_matrix"]
    )
    arrays = {
        "ancestors": spec["ancestors"],
        "pre_flow_particles": spec["pre_flow_particles"],
        "observation": spec["observation"],
        "transition_matrix": spec["transition_matrix"],
        "transition_covariance": spec["transition_covariance"],
        "observation_matrix": spec["observation_matrix"],
        "observation_covariance": spec["observation_covariance"],
    }
    digest_settings = {
        "fixture_name": spec["fixture_name"],
        "seed": spec["seed"],
        "construction": spec["construction"],
        "observation_route": "fixed_linear_gaussian_ledh_flow",
        "weight_route": "transition_plus_observation_minus_ledh_proposal_density",
    }
    start = time.perf_counter()
    flow = ledh_flow_batch_tf(
        pre_flow_particles=tf.constant(spec["pre_flow_particles"], dtype=DTYPE),
        ancestors=tf.constant(spec["ancestors"], dtype=DTYPE),
        observation=tf.constant(spec["observation"], dtype=DTYPE),
        transition_matrix=tf.constant(spec["transition_matrix"], dtype=DTYPE),
        transition_covariance=tf.constant(spec["transition_covariance"], dtype=DTYPE),
        observation_covariance=tf.constant(spec["observation_covariance"], dtype=DTYPE),
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_residual_fn=observation_residual_fn,
    )
    wall_time = time.perf_counter() - start
    post_flow_particles = flow.post_flow_particles.numpy()
    prior_means = spec["ancestors"] @ spec["transition_matrix"].T
    transition_log_density = _gaussian_logpdf_np(
        post_flow_particles - prior_means,
        spec["transition_covariance"],
    )
    obs_residuals = post_flow_particles @ spec["observation_matrix"].T - spec["observation"]
    observation_log_density = _gaussian_logpdf_np(obs_residuals, spec["observation_covariance"])
    proposal_log_density = flow.pre_flow_log_density.numpy() - flow.forward_log_det.numpy()
    raw_log_weights = transition_log_density + observation_log_density - proposal_log_density
    log_weights = raw_log_weights - _logsumexp(raw_log_weights, axis=0)
    arrays_with_outputs = dict(arrays)
    arrays_with_outputs.update(
        {
            "post_flow_particles": post_flow_particles,
            "raw_log_weights": raw_log_weights,
            "normalized_log_weights": log_weights,
            "forward_log_det": flow.forward_log_det.numpy(),
            "pre_flow_log_density": flow.pre_flow_log_density.numpy(),
        }
    )
    content_digest = _array_digest(arrays_with_outputs, digest_settings)
    finite_log_weights = bool(np.all(np.isfinite(log_weights)))
    finite_weights_normalized = bool(
        np.allclose(_logsumexp(log_weights, axis=0), 0.0, atol=1.0e-12)
    )
    provenance = {
        "fixture_name": spec["fixture_name"],
        "purpose": spec["purpose"],
        "seed": spec["seed"],
        "construction": spec["construction"],
        "content_digest_sha256": content_digest,
        "observation_route": "fixed_linear_gaussian_ledh_flow",
        "observation_matrix_shape": list(spec["observation_matrix"].shape),
        "transition_matrix_shape": list(spec["transition_matrix"].shape),
        "transition_covariance_diag": np.diag(spec["transition_covariance"]).tolist(),
        "observation_covariance_diag": np.diag(spec["observation_covariance"]).tolist(),
        "weight_route": (
            "normalized diagnostic weights from transition_log_density(post_flow) "
            "+ observation_log_density(post_flow) - "
            "(pre_flow_log_density - forward_log_det)"
        ),
        "weight_route_nonclaim": "diagnostic weighting only; no posterior correctness claim",
    }
    flow_diagnostics = dict(flow.diagnostics)
    flow_diagnostics.update(
        {
            "wall_time_seconds": wall_time,
            "finite_log_weights": finite_log_weights,
            "finite_weight_normalization": finite_weights_normalized,
            "max_log_weight_normalization_residual": float(abs(_logsumexp(log_weights, axis=0))),
            "min_normalized_log_weight": float(np.min(log_weights)),
            "max_normalized_log_weight": float(np.max(log_weights)),
            "raw_log_weight_range": [
                float(np.min(raw_log_weights)),
                float(np.max(raw_log_weights)),
            ],
        }
    )
    return {
        "fixture_name": spec["fixture_name"],
        "provenance": provenance,
        "pre_flow_particles": spec["pre_flow_particles"],
        "post_flow_particles": post_flow_particles,
        "log_weights": log_weights,
        "flow_diagnostics": flow_diagnostics,
        "particle_summaries": {
            "ancestors": _particle_summary(spec["ancestors"]),
            "pre_flow": _particle_summary(spec["pre_flow_particles"]),
            "post_flow": _particle_summary(post_flow_particles),
        },
    }


def _fixture_diagnostics(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ledh_fixture = _run_ledh_fixture(spec)
    dense_record, transport, dense_particles = _run_dense(
        ledh_fixture["post_flow_particles"],
        ledh_fixture["log_weights"],
        args,
    )
    support = _support_count_diagnostics(transport)
    nearest = _nearest_neighbor_mass_diagnostics(ledh_fixture["post_flow_particles"][None, :, :], transport)
    truncation = _truncation_diagnostics(
        ledh_fixture["post_flow_particles"],
        ledh_fixture["log_weights"],
        transport,
        dense_particles,
        support["counts_99"],
    )
    n_particles = int(transport.shape[2])
    median_threshold = max(8, int(math.ceil(0.25 * n_particles)))
    p90_threshold = max(16, int(math.ceil(0.50 * n_particles)))
    support_99 = support["by_threshold"]["0.990"]
    finite_ledh_pass = bool(
        ledh_fixture["flow_diagnostics"]["finite_post_flow"]
        and ledh_fixture["flow_diagnostics"]["finite_forward_log_det"]
        and ledh_fixture["flow_diagnostics"]["finite_pre_flow_log_density"]
        and ledh_fixture["flow_diagnostics"]["finite_log_weights"]
        and ledh_fixture["flow_diagnostics"]["finite_weight_normalization"]
    )
    provenance_pass = bool(ledh_fixture["provenance"]["content_digest_sha256"])
    threshold_checks = {
        "n_particles": n_particles,
        "median_99_support_threshold": median_threshold,
        "p90_99_support_threshold": p90_threshold,
        "median_99_support_pass": bool(support_99["median"] <= median_threshold),
        "p90_99_support_pass": bool(support_99["p90"] <= p90_threshold),
        "row_residual_threshold": ROW_RESIDUAL_THRESHOLD,
        "column_residual_threshold": COLUMN_RESIDUAL_THRESHOLD,
        "particle_error_threshold": PARTICLE_ERROR_THRESHOLD,
        "truncated_row_residual_pass": bool(truncation["max_row_residual"] <= ROW_RESIDUAL_THRESHOLD),
        "truncated_column_residual_pass": bool(
            truncation["max_column_residual"] <= COLUMN_RESIDUAL_THRESHOLD
        ),
        "truncated_particle_error_pass": bool(
            truncation["max_transported_particle_error"] <= PARTICLE_ERROR_THRESHOLD
        ),
        "finite_ledh_flow_pass": finite_ledh_pass,
        "finite_dense_and_truncated_pass": bool(
            dense_record["finite_transport_matrix"]
            and dense_record["finite_particles"]
            and truncation["finite_truncated_matrix"]
            and truncation["finite_truncated_particles"]
        ),
        "fixture_provenance_pass": provenance_pass,
    }
    threshold_checks["advance_thresholds_pass"] = bool(
        threshold_checks["median_99_support_pass"]
        and threshold_checks["p90_99_support_pass"]
        and threshold_checks["truncated_row_residual_pass"]
        and threshold_checks["truncated_column_residual_pass"]
        and threshold_checks["truncated_particle_error_pass"]
        and threshold_checks["finite_ledh_flow_pass"]
        and threshold_checks["finite_dense_and_truncated_pass"]
        and threshold_checks["fixture_provenance_pass"]
    )
    support["counts_99"] = support["counts_99"].astype(int).tolist()
    return {
        "fixture_name": ledh_fixture["fixture_name"],
        "input_shape": {
            "post_flow_particles": list(ledh_fixture["post_flow_particles"].shape),
            "log_weights": list(ledh_fixture["log_weights"].shape),
        },
        "fixture_provenance": ledh_fixture["provenance"],
        "ledh_flow_diagnostics": ledh_fixture["flow_diagnostics"],
        "particle_summaries": ledh_fixture["particle_summaries"],
        "dense": dense_record,
        "support_count_diagnostics": support,
        "nearest_neighbor_mass_diagnostics": nearest,
        "truncation_diagnostics": truncation,
        "threshold_checks": threshold_checks,
        "fixture_decision": (
            "fixture_reopens_sparse_implementation_plan_only"
            if threshold_checks["advance_thresholds_pass"]
            else "fixture_does_not_reopen_sparse_implementation"
        ),
    }


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    fixture_results: dict[str, Any] = {}
    hard_vetoes: list[str] = []
    promotion_vetoes: list[str] = []
    for spec in _fixture_specs():
        fixture = _fixture_diagnostics(spec, args)
        fixture_results[fixture["fixture_name"]] = fixture
        checks = fixture["threshold_checks"]
        if not checks["fixture_provenance_pass"]:
            hard_vetoes.append(f"{fixture['fixture_name']}:missing_fixture_provenance_digest")
        if not checks["finite_ledh_flow_pass"]:
            hard_vetoes.append(f"{fixture['fixture_name']}:nonfinite_ledh_flow_or_weights")
        if not checks["finite_dense_and_truncated_pass"]:
            hard_vetoes.append(f"{fixture['fixture_name']}:nonfinite_dense_or_truncated_artifact")
        if not checks["advance_thresholds_pass"]:
            if not checks["median_99_support_pass"]:
                promotion_vetoes.append(f"{fixture['fixture_name']}:median_99_support_too_large")
            if not checks["p90_99_support_pass"]:
                promotion_vetoes.append(f"{fixture['fixture_name']}:p90_99_support_too_large")
            if not checks["truncated_row_residual_pass"]:
                promotion_vetoes.append(f"{fixture['fixture_name']}:truncated_row_residual_too_large")
            if not checks["truncated_column_residual_pass"]:
                promotion_vetoes.append(f"{fixture['fixture_name']}:truncated_column_residual_too_large")
            if not checks["truncated_particle_error_pass"]:
                promotion_vetoes.append(f"{fixture['fixture_name']}:truncated_particle_error_too_large")

    diagnostic_completed = not hard_vetoes
    advance = diagnostic_completed and not promotion_vetoes
    final_status = (
        FINAL_STATUS_REOPENS
        if advance
        else (FINAL_STATUS_BLOCKS if diagnostic_completed else FINAL_STATUS_BLOCKED)
    )
    status = "PASS" if diagnostic_completed else "FAIL"
    elapsed = time.perf_counter() - start
    summary = {
        "max_dense_row_residual": max(
            fixture["dense"]["max_row_residual"] for fixture in fixture_results.values()
        ),
        "max_dense_column_residual": max(
            fixture["dense"]["max_column_residual"] for fixture in fixture_results.values()
        ),
        "max_truncated_row_residual": max(
            fixture["truncation_diagnostics"]["max_row_residual"]
            for fixture in fixture_results.values()
        ),
        "max_truncated_column_residual": max(
            fixture["truncation_diagnostics"]["max_column_residual"]
            for fixture in fixture_results.values()
        ),
        "max_truncated_particle_error": max(
            fixture["truncation_diagnostics"]["max_transported_particle_error"]
            for fixture in fixture_results.values()
        ),
        "max_99_support_p90_fraction_of_n": max(
            fixture["support_count_diagnostics"]["by_threshold"]["0.990"]["p90_fraction_of_n"]
            for fixture in fixture_results.values()
        ),
        "min_99_support_median_fraction_of_n": min(
            fixture["support_count_diagnostics"]["by_threshold"]["0.990"]["median_fraction_of_n"]
            for fixture in fixture_results.values()
        ),
        "min_nearest_neighbor_mass_k1_median": min(
            fixture["nearest_neighbor_mass_diagnostics"]["by_k"]["1"]["median"]
            for fixture in fixture_results.values()
        ),
    }
    return {
        "p12e_status": final_status,
        "status": status,
        "diagnostic_completed": diagnostic_completed,
        "reopens_sparse_implementation_plan_only": advance,
        "hard_vetoes": hard_vetoes,
        "promotion_vetoes": promotion_vetoes,
        "semantic_class": "reference_only_diagnostic",
        "source_route": "source_reference_only",
        "implementation_scope": "ledh_post_flow_dense_plan_locality_and_truncation_diagnostic",
        "baseline_comparator": "phase1_dense_streaming_conventions_on_same_ledh_like_particles",
        "transport_object": {
            "kind": "dense_matrix",
            "materialized": True,
            "orientation": "target_rows_source_columns_phase1_scaled_dense_transport",
            "semantic_output": "post_flow_particles_after_dense_transport",
            "role": "diagnostic_only_dense_reference_for_locality_screen",
        },
        "thresholds": {
            "median_99_support": "max(8, ceil(0.25 * N))",
            "p90_99_support": "max(16, ceil(0.50 * N))",
            "truncated_max_row_residual": ROW_RESIDUAL_THRESHOLD,
            "truncated_max_column_residual": COLUMN_RESIDUAL_THRESHOLD,
            "truncated_max_transported_particle_error": PARTICLE_ERROR_THRESHOLD,
            "finite_ledh_dense_and_truncated_particles": True,
            "fixture_provenance_digest_required": True,
        },
        "definitions": {
            "N": "transport_matrix.shape[2], the source-particle count",
            "orientation": "target rows, source columns, Phase 1 scaled dense transport",
            "minimal_per_row_99_support": (
                "stable descending row-mass prefix whose cumulative mass first "
                "reaches 99% of the dense row mass; ties are not expanded"
            ),
            "truncated_transport_role": (
                "diagnostic-only materialization used to evaluate locality; "
                "not a sparse solver implementation"
            ),
            "ledh_fixture_route": (
                "deterministic pre-flow particles and ancestors are passed "
                "through ledh_flow_batch_tf with fixed linear-Gaussian "
                "observation functions"
            ),
        },
        "summary": summary,
        "settings": {
            "epsilon": args.epsilon,
            "scaling": args.scaling,
            "convergence_threshold": args.convergence_threshold,
            "max_iterations": args.max_iterations,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "transport_gradient_mode": "raw",
            "transport_plan_mode": "dense",
        },
        "manifest": {
            "git_commit": _git_commit(),
            "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "device_scope": args.device_scope,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
            "dtype": "tf.float64",
            "wall_time_seconds": elapsed,
            "peak_rss_kb": _peak_rss_kb(),
            "command": "scalable_ot_p12e_ledh_sparse_locality_screen.py",
            "plan_path": (
                "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-scalable-ot-"
                "p12e-ledh-sparse-locality-screen-master-program-2026-06-19.md"
            ),
        },
        "fixtures": fixture_results,
        "diagnostic_roles": {
            "fixture_provenance_pass": "continuation_veto",
            "finite_ledh_flow_pass": "continuation_veto",
            "finite_dense_and_truncated_pass": "continuation_veto",
            "median_99_support_pass": "promotion_veto",
            "p90_99_support_pass": "promotion_veto",
            "truncated_row_residual_pass": "promotion_veto",
            "truncated_column_residual_pass": "promotion_veto",
            "truncated_particle_error_pass": "promotion_veto",
            "nearest_neighbor_mass_diagnostics": "explanatory",
            "support_curves_90_95_999": "explanatory",
            "runtime_memory": "explanatory",
            "phase8_context": "explanatory",
        },
        "nonclaims": list(NONCLAIMS),
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    handoff = result["next_phase_handoff"]
    lines = [
        "# P12E LEDH Sparse-Locality Screen",
        "",
        f"- Artifact scope: `{result['artifact_scope']}`",
        f"- Status: `{result['status']}`",
        f"- P12E status: `{result['p12e_status']}`",
        f"- Diagnostic completed: `{result['diagnostic_completed']}`",
        f"- Reopens sparse implementation plan only: `{result['reopens_sparse_implementation_plan_only']}`",
        f"- Semantic class: `{result['semantic_class']}`",
        f"- Implementation scope: `{result['implementation_scope']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        f"- Promotion vetoes: `{result['promotion_vetoes']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| max dense row residual | `{result['summary']['max_dense_row_residual']:.6e}` |",
        f"| max dense column residual | `{result['summary']['max_dense_column_residual']:.6e}` |",
        f"| max truncated row residual | `{result['summary']['max_truncated_row_residual']:.6e}` |",
        f"| max truncated column residual | `{result['summary']['max_truncated_column_residual']:.6e}` |",
        f"| max truncated particle error | `{result['summary']['max_truncated_particle_error']:.6e}` |",
        f"| max 99% support p90 fraction of N | `{result['summary']['max_99_support_p90_fraction_of_n']:.6e}` |",
        f"| min nearest-neighbor k=1 median mass | `{result['summary']['min_nearest_neighbor_mass_k1_median']:.6e}` |",
        "",
        "## Fixture Rows",
        "",
        "| Fixture | Decision | N | Median k99 | P90 k99 | Trunc row residual | Trunc column residual | Trunc max particle error | Nonzero fraction | Digest |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for fixture_name, fixture in result["fixtures"].items():
        support_99 = fixture["support_count_diagnostics"]["by_threshold"]["0.990"]
        truncation = fixture["truncation_diagnostics"]
        checks = fixture["threshold_checks"]
        digest = fixture["fixture_provenance"]["content_digest_sha256"][:12]
        lines.append(
            "| {fixture} | `{decision}` | {n} | `{median:.3f}` | `{p90:.3f}` | `{row:.6e}` | `{col:.6e}` | `{err:.6e}` | `{frac:.6e}` | `{digest}` |".format(
                fixture=fixture_name,
                decision=fixture["fixture_decision"],
                n=checks["n_particles"],
                median=support_99["median"],
                p90=support_99["p90"],
                row=truncation["max_row_residual"],
                col=truncation["max_column_residual"],
                err=truncation["max_transported_particle_error"],
                frac=truncation["nonzero_fraction_of_dense"],
                digest=digest,
            )
        )
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{result['p12e_status']}` | Diagnostic artifact criterion recorded by JSON/Markdown output. | "
                f"Hard vetoes: `{result['hard_vetoes']}`; promotion vetoes: `{result['promotion_vetoes']}`. | "
                "Synthetic LEDH-like fixtures may not represent a future frozen real LEDH run. | "
                f"{handoff['next_action']} | "
                "No sparse solver validity, speedup, ranking, posterior correctness, HMC/API/default/production readiness. |"
            ),
            "",
            "## Inference Status",
            "",
            "| Evidence class | Status |",
            "| --- | --- |",
            f"| Hard veto screen | `{'PASS' if not result['hard_vetoes'] else 'FAIL'}` |",
            "| Statistically supported ranking | `NONE` |",
            "| Descriptive-only differences | Runtime, memory, support curves outside the 99% thresholds, nearest-neighbor mass, and LEDH log-det ranges. |",
            "| Default-readiness | `NOT_ASSESSED_AND_NOT_CLAIMED` |",
            f"| Next evidence needed | {handoff['next_evidence_needed']} |",
            f"| Scope non-claim | {handoff['scope_nonclaim']} |",
            "",
            "## Threshold Definitions",
            "",
            "- `N = transport_matrix.shape[2]`, the source-particle count.",
            "- `k_i(t)` uses a deterministic stable descending sort of row mass.",
            "- Ties are not expanded beyond the first stable prefix reaching the threshold.",
            "- The truncated transport is diagnostic-only and is not a sparse solver.",
            "- Fixture digests hash deterministic inputs, LEDH outputs, and diagnostic weights.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    for nonclaim in result["nonclaims"]:
        lines.append(f"- {nonclaim}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = _build_result(args)
    output = Path(args.output)
    artifact_scope = _artifact_scope(output)
    result["artifact_scope"] = artifact_scope
    result["next_phase_handoff"] = _next_phase_handoff(artifact_scope)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown_output)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(result, markdown)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

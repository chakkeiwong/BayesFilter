"""Minimal SSL-LSTM Zhao-Cui HMC conditional-slice oracle.

This CPU-hidden debug/reference harness compares the internal minimal
``zhaocui_fixed`` HMC target against an independent NumPy replay of the frozen
scalar fixture on selected one-dimensional conditional slices. It does not run
HMC and does not claim full posterior correctness, convergence, ranking,
readiness, source-faithful Zhao-Cui parity, or LEDH evidence.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from bayesfilter.nonlinear.ssl_lstm_zhaocui_hmc_minimal import (  # noqa: E402
    MinimalZhaoCuiHMCTargetAdapter,
    initial_minimal_ssl_lstm_hmc_state,
    minimal_ssl_lstm_config,
    minimal_ssl_lstm_fixture_payload,
    minimal_ssl_lstm_observations,
    minimal_ssl_lstm_theta,
    minimal_ssl_lstm_zhaocui_manifest,
)
from bayesfilter.runtime import atomic_write_json  # noqa: E402


DATE_STAMP = "2026-07-06"
SCRIPT_NAME = "benchmark_minimal_ssl_lstm_zhaocui_hmc_oracle_2026_07_06.py"
DEFAULT_JSON_OUTPUT = (
    ROOT
    / "docs/benchmarks/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_phase2_oracle_cpu_hidden_2026-07-06.json"
)
DEFAULT_MARKDOWN_OUTPUT = (
    ROOT
    / "docs/benchmarks/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_phase2_oracle_cpu_hidden_2026-07-06.md"
)
QUIET_LOG_PATH = (
    "docs/benchmarks/logs/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_gaps_2026-07-06/"
    "phase2_oracle_cpu_hidden_2026-07-06.log"
)
MASTER_PROGRAM_PATH = (
    "docs/plans/"
    "bayesfilter-minimal-ssl-lstm-zhaocui-hmc-validity-gaps-master-program-2026-07-06.md"
)
PHASE2_SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-minimal-ssl-lstm-zhaocui-hmc-validity-gaps-phase2-"
    "oracle-implementation-subplan-2026-07-06.md"
)
PHASE2_RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-minimal-ssl-lstm-zhaocui-hmc-validity-gaps-phase2-"
    "oracle-implementation-result-2026-07-06.md"
)
DEFAULT_PRIOR_SCALE = 5.0
DEFAULT_INITIAL_OFFSET_SCALE = 1.0e-3
DEFAULT_WIDTHS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
DEFAULT_POINTS_PER_WIDTH = 401
EDGE_MASS_THRESHOLD = 1.0e-4
VALUE_ATOL = 1.0e-9
VALUE_RTOL = 1.0e-12
FD_STEP = 1.0e-5
FD_SCORE_ATOL = 1.0e-3
DETERMINISM_ATOL = 1.0e-12
SELECTED_COORDINATES = (
    (0, "lstm_input.input.0.0", "LSTM input gate weight"),
    (4, "lstm_recurrent.input.0.0", "recurrent gate weight"),
    (8, "lstm_bias.input.0", "gate bias"),
    (12, "latent_mean_weight.0.0", "transition mean scale"),
    (13, "latent_mean_bias.0", "transition mean bias"),
    (14, "observation_weight.0.0", "observation loading"),
    (15, "observation_bias.0", "observation intercept"),
    (16, "initial_mean.0", "latent initial mean"),
    (19, "initial_std_unconstrained.0", "latent initial scale transform"),
    (22, "process_std_unconstrained.0", "process scale transform"),
    (23, "observation_std_unconstrained.0", "observation scale transform"),
)
NONCLAIMS = (
    "Phase 2 conditional-slice reference/debug artifact only",
    "CPU-hidden non-JIT reference exception only",
    "not full posterior correctness evidence",
    "not HMC convergence evidence",
    "not R-hat or ESS evidence",
    "not a method ranking or superiority claim",
    "not source-faithful SSL-LSTM Zhao-Cui parity evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
    "not public API or package readiness evidence",
    "not LEDH evidence",
)


def softplus_np(values: np.ndarray) -> np.ndarray:
    """Stable NumPy softplus."""

    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    """Stable NumPy sigmoid."""

    return 1.0 / (1.0 + np.exp(-values))


def logsumexp_np(values: np.ndarray) -> float:
    """Return log(sum(exp(values))) for a one-dimensional array."""

    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def logmeanexp_np(values: np.ndarray) -> float:
    return logsumexp_np(values) - math.log(float(values.shape[0]))


def array_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def parameter_slices() -> dict[str, int]:
    return {
        "lstm_input_start": 0,
        "lstm_recurrent_start": 4,
        "lstm_bias_start": 8,
        "latent_weight_start": 12,
        "latent_bias_start": 13,
        "observation_weight_start": 14,
        "observation_bias_start": 15,
        "initial_mean_start": 16,
        "initial_std_start": 19,
        "process_std_start": 22,
        "observation_std_start": 23,
        "parameter_dim": 24,
    }


def unpack_scalar_parameters(theta: np.ndarray, *, std_floor: float = 1.0e-4) -> dict[str, Any]:
    """Unpack the scalar SSL-LSTM parameter vector using NumPy only."""

    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    slices = parameter_slices()
    if values.shape != (slices["parameter_dim"],):
        raise ValueError("theta must have shape (24,)")
    raw_initial_std = values[slices["initial_std_start"] : slices["initial_std_start"] + 3]
    raw_process_std = values[slices["process_std_start"] : slices["process_std_start"] + 1]
    raw_observation_std = values[
        slices["observation_std_start"] : slices["observation_std_start"] + 1
    ]
    return {
        "lstm_input": values[slices["lstm_input_start"] : slices["lstm_input_start"] + 4],
        "lstm_recurrent": values[
            slices["lstm_recurrent_start"] : slices["lstm_recurrent_start"] + 4
        ],
        "lstm_bias": values[slices["lstm_bias_start"] : slices["lstm_bias_start"] + 4],
        "latent_weight": float(values[slices["latent_weight_start"]]),
        "latent_bias": float(values[slices["latent_bias_start"]]),
        "observation_weight": float(values[slices["observation_weight_start"]]),
        "observation_bias": float(values[slices["observation_bias_start"]]),
        "initial_mean": values[slices["initial_mean_start"] : slices["initial_mean_start"] + 3],
        "initial_std": softplus_np(raw_initial_std) + float(std_floor),
        "process_std": softplus_np(raw_process_std) + float(std_floor),
        "observation_std": softplus_np(raw_observation_std) + float(std_floor),
    }


def transition_np(params: Mapping[str, Any], state: np.ndarray) -> np.ndarray:
    """Evaluate the scalar SSL-LSTM transition mean with NumPy arrays."""

    z_prev = state[:, 0]
    a_prev = state[:, 1]
    c_prev = state[:, 2]
    pre = (
        np.outer(z_prev, np.asarray(params["lstm_input"], dtype=np.float64))
        + np.outer(a_prev, np.asarray(params["lstm_recurrent"], dtype=np.float64))
        + np.asarray(params["lstm_bias"], dtype=np.float64)[None, :]
    )
    input_gate = sigmoid_np(pre[:, 0])
    forget_gate = sigmoid_np(pre[:, 1])
    output_gate = sigmoid_np(pre[:, 2])
    candidate = np.tanh(pre[:, 3])
    c_next = forget_gate * c_prev + input_gate * candidate
    a_next = output_gate * np.tanh(c_next)
    z_mean = a_next * float(params["latent_weight"]) + float(params["latent_bias"])
    return np.stack([z_mean, a_next, c_next], axis=1)


def observation_mean_np(params: Mapping[str, Any], state: np.ndarray) -> np.ndarray:
    z = state[:, 0]
    return z[:, None] * float(params["observation_weight"]) + float(
        params["observation_bias"]
    )


def materialize_replay_noise() -> dict[str, np.ndarray]:
    """Materialize the fixed replay noise arrays from the reviewed fixture seeds."""

    config = minimal_ssl_lstm_config()
    manifest = minimal_ssl_lstm_zhaocui_manifest()
    state_dim = int(config.augmented_state_dim)
    sample_count = int(manifest.reference_sample_count)
    transition_count = max(int(config.horizon) - 1, 0)
    initial_noise = tf.random.stateless_normal(
        [sample_count, state_dim],
        seed=tf.constant(manifest.initial_seed, dtype=tf.int32),
        dtype=tf.float64,
    )
    process_noise = tf.random.stateless_normal(
        [transition_count, sample_count, int(config.latent_dim)],
        seed=tf.constant(manifest.process_seed, dtype=tf.int32),
        dtype=tf.float64,
    )
    return {
        "initial_noise": np.asarray(initial_noise.numpy(), dtype=np.float64),
        "process_noise": np.asarray(process_noise.numpy(), dtype=np.float64),
    }


def reference_log_likelihood_np(
    theta: np.ndarray,
    observations: np.ndarray,
    noise: Mapping[str, np.ndarray],
) -> float:
    """Independent NumPy replay of the fixed scalar ``zhaocui_fixed`` value."""

    params = unpack_scalar_parameters(theta)
    state = (
        np.asarray(params["initial_mean"], dtype=np.float64)[None, :]
        + np.asarray(noise["initial_noise"], dtype=np.float64)
        * np.asarray(params["initial_std"], dtype=np.float64)[None, :]
    )
    particle_log_values = np.zeros(state.shape[0], dtype=np.float64)
    obs_std = float(np.asarray(params["observation_std"], dtype=np.float64)[0])
    variance = obs_std * obs_std
    log_two_pi = math.log(2.0 * math.pi)
    for step in range(observations.shape[0]):
        mean = observation_mean_np(params, state)[:, 0]
        residual = float(observations[step, 0]) - mean
        particle_log_values += -0.5 * (
            log_two_pi + math.log(variance) + np.square(residual) / variance
        )
        if step + 1 < observations.shape[0]:
            deterministic_next = transition_np(params, state)
            process = np.asarray(noise["process_noise"], dtype=np.float64)[step, :, 0]
            deterministic_next[:, 0] += process * float(params["process_std"][0])
            state = deterministic_next
    return logmeanexp_np(particle_log_values)


def reference_log_prob_np(
    theta: np.ndarray,
    observations: np.ndarray,
    noise: Mapping[str, np.ndarray],
    *,
    prior_center: np.ndarray,
    prior_scale: float,
) -> float:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    center = np.asarray(prior_center, dtype=np.float64).reshape(-1)
    delta = values - center
    prior_value = -0.5 * float(np.sum(np.square(delta) / (float(prior_scale) ** 2)))
    return reference_log_likelihood_np(values, observations, noise) + prior_value


def target_log_prob_and_score(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    theta: np.ndarray,
) -> tuple[float, np.ndarray]:
    value, score = adapter.log_prob_and_grad(tf.constant(theta, dtype=tf.float64))
    return float(value.numpy()), np.asarray(score.numpy(), dtype=np.float64).reshape(-1)


def target_log_prob_batch(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    theta_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate target log probabilities for a batch of states."""

    tensor = tf.constant(np.asarray(theta_batch, dtype=np.float64), dtype=tf.float64)
    values = tf.map_fn(
        lambda row: adapter._scalar_log_prob_and_grad(row)[0], tensor,
        fn_output_signature=tf.TensorSpec([], tf.float64), parallel_iterations=1,
    )
    return np.asarray(values.numpy(), dtype=np.float64)


def relative_error(abs_error: float, left: float, right: float) -> float:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return float(abs_error) / scale


def value_check_passed(abs_error: float, rel_error: float) -> bool:
    return bool(abs_error <= VALUE_ATOL or rel_error <= VALUE_RTOL)


def central_difference_subset(
    theta: np.ndarray,
    value_fn: Any,
    indices: Sequence[int],
    *,
    step: float = FD_STEP,
) -> dict[str, Any]:
    values: list[float] = []
    base = np.asarray(theta, dtype=np.float64)
    for index in indices:
        plus = base.copy()
        minus = base.copy()
        plus[int(index)] += float(step)
        minus[int(index)] -= float(step)
        values.append((float(value_fn(plus)) - float(value_fn(minus))) / (2.0 * float(step)))
    return {
        "indices": [int(index) for index in indices],
        "step": float(step),
        "values": values,
    }


def conditional_slice_row(
    *,
    index: int,
    name: str,
    role: str,
    base_theta: np.ndarray,
    widths: Sequence[float],
    points_per_width: int,
    target_values_fn: Any,
    reference_value_fn: Any,
) -> dict[str, Any]:
    width_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    best_value_error = 0.0
    all_finite = True
    for width in widths:
        grid = np.linspace(
            base_theta[index] - float(width),
            base_theta[index] + float(width),
            int(points_per_width),
            dtype=np.float64,
        )
        theta_batch = np.repeat(base_theta[None, :], int(points_per_width), axis=0)
        theta_batch[:, index] = grid
        target_array = np.asarray(target_values_fn(theta_batch), dtype=np.float64)
        reference_array = np.asarray(
            [float(reference_value_fn(theta_row)) for theta_row in theta_batch],
            dtype=np.float64,
        )
        finite = bool(np.all(np.isfinite(target_array)) and np.all(np.isfinite(reference_array)))
        all_finite = bool(all_finite and finite)
        errors = np.abs(target_array - reference_array)
        relative_errors = np.asarray(
            [
                relative_error(float(error), float(target), float(reference))
                for error, target, reference in zip(errors, target_array, reference_array)
            ],
            dtype=np.float64,
        )
        max_value_error = float(np.max(errors))
        max_relative_error = float(np.max(relative_errors))
        best_value_error = max(best_value_error, max_value_error)
        normalized = np.exp(reference_array - logsumexp_np(reference_array))
        edge_mass = float(normalized[0] + normalized[-1])
        mean = float(np.sum(normalized * grid))
        variance = float(np.sum(normalized * np.square(grid - mean)))
        row = {
            "width": float(width),
            "points": int(points_per_width),
            "finite": finite,
            "target_reference_max_abs_error": max_value_error,
            "target_reference_max_rel_error": max_relative_error,
            "target_reference_value_passed": value_check_passed(
                max_value_error,
                max_relative_error,
            ),
            "edge_mass": edge_mass,
            "edge_mass_passed": bool(edge_mass <= EDGE_MASS_THRESHOLD),
            "conditional_mean": mean,
            "conditional_std": math.sqrt(max(variance, 0.0)),
            "map_grid_value": float(grid[int(np.argmax(reference_array))]),
            "log_normalizer": logsumexp_np(reference_array)
            + math.log((float(grid[-1]) - float(grid[0])) / (int(points_per_width) - 1)),
            "grid_min": float(grid[0]),
            "grid_max": float(grid[-1]),
        }
        width_rows.append(row)
        if selected is None and row["edge_mass_passed"]:
            selected = row
    return {
        "index": int(index),
        "name": name,
        "role": role,
        "selected_width": None if selected is None else selected["width"],
        "edge_mass_passed": selected is not None,
        "all_values_finite": all_finite,
        "target_reference_max_abs_error": best_value_error,
        "width_rows": width_rows,
        "selected_summary": selected,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty_summary() -> dict[str, Any]:
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "dirty": bool(lines),
        "line_count": len(lines),
        "preview": lines[:20],
    }


def _tf_device_summary() -> dict[str, Any]:
    physical = tf.config.list_physical_devices()
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "physical_devices": [device.name for device in physical],
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "trust_basis": "cpu_hidden_debug_reference_no_gpu_claim",
    }


def _selected_coordinate_payload() -> list[dict[str, Any]]:
    return [
        {"index": int(index), "name": name, "role": role}
        for index, name, role in SELECTED_COORDINATES
    ]


def build_oracle_artifact(
    *,
    widths: Sequence[float] = DEFAULT_WIDTHS,
    points_per_width: int = DEFAULT_POINTS_PER_WIDTH,
    coordinates: Sequence[tuple[int, str, str]] = SELECTED_COORDINATES,
    prior_scale: float = DEFAULT_PRIOR_SCALE,
    initial_offset_scale: float = DEFAULT_INITIAL_OFFSET_SCALE,
    command: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the Phase 2 target/reference conditional-slice artifact."""

    start = time.perf_counter()
    theta_center = np.asarray(minimal_ssl_lstm_theta().numpy(), dtype=np.float64)
    theta = np.asarray(
        initial_minimal_ssl_lstm_hmc_state(initial_offset_scale).numpy(),
        dtype=np.float64,
    )
    observations = np.asarray(minimal_ssl_lstm_observations().numpy(), dtype=np.float64)
    noise = materialize_replay_noise()
    adapter = MinimalZhaoCuiHMCTargetAdapter(prior_scale=prior_scale)

    def reference_value_fn(local_theta: np.ndarray) -> float:
        return reference_log_prob_np(
            local_theta,
            observations,
            noise,
            prior_center=theta_center,
            prior_scale=prior_scale,
        )

    def target_value_fn(local_theta: np.ndarray) -> float:
        value, _score = target_log_prob_and_score(adapter, local_theta)
        return value

    def target_values_fn(local_theta_batch: np.ndarray) -> np.ndarray:
        return target_log_prob_batch(adapter, local_theta_batch)

    target_value, target_score = target_log_prob_and_score(adapter, theta)
    repeated_value, repeated_score = target_log_prob_and_score(adapter, theta.copy())
    reference_value = reference_value_fn(theta)
    repeat_delta = max(
        abs(target_value - repeated_value),
        float(np.max(np.abs(target_score - repeated_score))),
    )
    selected_indices = [int(index) for index, _name, _role in coordinates]
    fd = central_difference_subset(theta, target_value_fn, selected_indices)
    fd_values = np.asarray(fd["values"], dtype=np.float64)
    analytic_subset = target_score[selected_indices]
    fd_errors = np.abs(analytic_subset - fd_values)
    slice_rows = [
        conditional_slice_row(
            index=int(index),
            name=name,
            role=role,
            base_theta=theta,
            widths=widths,
            points_per_width=points_per_width,
            target_values_fn=target_values_fn,
            reference_value_fn=reference_value_fn,
        )
        for index, name, role in coordinates
    ]
    value_errors = [
        abs(target_value - reference_value),
        *[
            float(row["target_reference_max_abs_error"])
            for row in slice_rows
        ],
    ]
    value_rel_errors = [
        relative_error(abs(target_value - reference_value), target_value, reference_value),
        *[
            max(
                float(width_row["target_reference_max_rel_error"])
                for width_row in row["width_rows"]
            )
            for row in slice_rows
        ],
    ]
    hard_vetoes: list[str] = []
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        hard_vetoes.append("cuda_visible_devices_not_cpu_hidden")
    if not math.isfinite(target_value) or not math.isfinite(reference_value):
        hard_vetoes.append("base_value_nonfinite")
    if not bool(np.all(np.isfinite(target_score))):
        hard_vetoes.append("target_score_nonfinite")
    if repeat_delta > DETERMINISM_ATOL:
        hard_vetoes.append("target_repeat_delta_exceeds_tolerance")
    if not value_check_passed(max(value_errors), max(value_rel_errors)):
        hard_vetoes.append("target_reference_value_mismatch")
    failed_value_rows = [
        row["name"]
        for row in slice_rows
        if any(not width_row["target_reference_value_passed"] for width_row in row["width_rows"])
    ]
    if failed_value_rows and "target_reference_value_mismatch" not in hard_vetoes:
        hard_vetoes.append("target_reference_value_mismatch")
    if float(np.max(fd_errors)) > FD_SCORE_ATOL:
        hard_vetoes.append("finite_difference_score_mismatch")
    failed_edge = [row["name"] for row in slice_rows if not row["edge_mass_passed"]]
    if failed_edge:
        hard_vetoes.append("conditional_slice_edge_mass_failure")
    if any(not row["all_values_finite"] for row in slice_rows):
        hard_vetoes.append("conditional_slice_nonfinite_value")

    runtime_s = time.perf_counter() - start
    artifact = {
        "schema_version": "minimal_ssl_lstm_zhaocui_hmc_validity.phase2_oracle.v1",
        "status": "passed" if not hard_vetoes else "failed",
        "date": DATE_STAMP,
        "phase": "PHASE2",
        "artifact_role": "conditional_slice_reference_debug",
        "filter_name": "zhaocui_fixed",
        "target_quantity": {
            "name": "minimal_zhaocui_fixed_hmc_log_prob",
            "description": (
                "Fixed zhaocui_fixed replay log likelihood plus Gaussian prior "
                "at the minimal scalar-dimension fixture."
            ),
            "prior_scale": float(prior_scale),
            "initial_offset_scale": float(initial_offset_scale),
        },
        "reference_independence_contract": {
            "reference_arithmetic": "plain_numpy_replay_inside_harness",
            "tensorflow_allowed_for": (
                "fixture tensors, stateless noise materialization, and target-under-test evaluation"
            ),
            "forbidden_reference_calls": [
                "tf_ssl_lstm_zhaocui_fixed_score",
                "MinimalZhaoCuiHMCTargetAdapter.log_prob_and_grad",
                "TensorFlow autodiff",
                "TensorFlow transition/observation helpers",
            ],
            "status": "declared_by_harness_structure",
        },
        "fixture": minimal_ssl_lstm_fixture_payload(),
        "selected_coordinates": _selected_coordinate_payload(),
        "grid_settings": {
            "widths": [float(width) for width in widths],
            "points_per_width": int(points_per_width),
            "edge_mass_threshold": EDGE_MASS_THRESHOLD,
            "selected_row_rule": "narrowest_width_with_edge_mass_at_or_below_threshold",
        },
        "tolerances": {
            "target_reference_value_atol": VALUE_ATOL,
            "target_reference_value_rtol": VALUE_RTOL,
            "finite_difference_step": FD_STEP,
            "finite_difference_score_atol": FD_SCORE_ATOL,
            "determinism_atol": DETERMINISM_ATOL,
            "tolerance_role": "reviewed_hypothesis_for_minimal_conditional_slice_screen",
        },
        "target_reference_value_check": {
            "base_target_value": float(target_value),
            "base_reference_value": float(reference_value),
            "base_abs_error": float(abs(target_value - reference_value)),
            "base_rel_error": relative_error(
                abs(target_value - reference_value),
                target_value,
                reference_value,
            ),
            "max_abs_error_over_base_and_slices": float(max(value_errors)),
            "max_rel_error_over_base_and_slices": float(max(value_rel_errors)),
            "passed": value_check_passed(max(value_errors), max(value_rel_errors)),
        },
        "finite_difference_score_check": {
            "indices": selected_indices,
            "step": float(fd["step"]),
            "finite_difference_values": [float(item) for item in fd_values],
            "analytic_subset_values": [float(item) for item in analytic_subset],
            "abs_errors": [float(item) for item in fd_errors],
            "max_abs_error": float(np.max(fd_errors)),
            "passed": bool(float(np.max(fd_errors)) <= FD_SCORE_ATOL),
            "role": "promotion_veto_for_local_score_consistency",
        },
        "determinism_check": {
            "max_abs_delta": float(repeat_delta),
            "passed": bool(repeat_delta <= DETERMINISM_ATOL),
            "role": "promotion_veto",
        },
        "conditional_slice_rows": slice_rows,
        "noise_manifest": {
            "initial_seed": list(minimal_ssl_lstm_zhaocui_manifest().initial_seed),
            "process_seed": list(minimal_ssl_lstm_zhaocui_manifest().process_seed),
            "initial_noise_shape": list(noise["initial_noise"].shape),
            "process_noise_shape": list(noise["process_noise"].shape),
            "initial_noise_sha256": array_hash(noise["initial_noise"]),
            "process_noise_sha256": array_hash(noise["process_noise"]),
        },
        "hard_vetoes": hard_vetoes,
        "diagnostic_roles": {
            "reference_independence_contract": "promotion_veto",
            "target_reference_value_check": "promotion_veto",
            "finite_difference_score_check": "promotion_veto_for_local_score_consistency",
            "determinism_check": "promotion_veto",
            "conditional_slice_edge_mass": "promotion_veto_for_selected_slice_adequacy",
            "conditional_means_std_map": "explanatory",
            "runtime": "explanatory",
        },
        "decision_table": {
            "decision": (
                "conditional-slice reference screen passed"
                if not hard_vetoes
                else "conditional-slice reference screen failed"
            ),
            "primary_criterion_status": "passed" if not hard_vetoes else "failed",
            "veto_diagnostic_status": "no_hard_vetoes" if not hard_vetoes else hard_vetoes,
            "main_uncertainty": (
                "Selected conditional one-dimensional slices do not establish full 24D posterior correctness."
            ),
            "next_justified_action": (
                "review Phase 2 result before longer HMC diagnostics"
                if not hard_vetoes
                else "localize target/reference or score mismatch before longer HMC"
            ),
            "what_is_not_concluded": list(NONCLAIMS),
        },
        "run_manifest": {
            "git_commit": _git_commit(),
            "git_dirty_summary": _git_dirty_summary(),
            "command": list(command or (sys.executable, str(Path(__file__).resolve()))),
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "cpu_gpu_status": _tf_device_summary(),
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "compile_mode": "eager",
            "jit_compile": False,
            "quiet_log_path": QUIET_LOG_PATH,
            "plan_file": MASTER_PROGRAM_PATH,
            "subplan_file": PHASE2_SUBPLAN_PATH,
            "result_file": PHASE2_RESULT_PATH,
            "random_seeds": {
                "zhaocui_initial_seed": list(minimal_ssl_lstm_zhaocui_manifest().initial_seed),
                "zhaocui_process_seed": list(minimal_ssl_lstm_zhaocui_manifest().process_seed),
            },
            "data_version": "frozen_inline_scalar_fixture_2026-07-06",
            "wall_time_s": float(runtime_s),
            "debug_reference_exception": (
                "CPU-hidden non-JIT Phase 2 conditional-slice reference screen"
            ),
        },
        "nonclaims": list(NONCLAIMS),
    }
    return artifact


def render_markdown(artifact: Mapping[str, Any]) -> str:
    value_check = artifact["target_reference_value_check"]
    fd_check = artifact["finite_difference_score_check"]
    lines = [
        "# Minimal SSL-LSTM Zhao-Cui HMC Validity Phase 2 Oracle",
        "",
        f"- Status: `{artifact['status']}`",
        f"- Artifact role: `{artifact['artifact_role']}`",
        f"- Hard vetoes: `{artifact['hard_vetoes']}`",
        f"- Base target/reference abs error: `{value_check['base_abs_error']}`",
        f"- Max target/reference abs error: `{value_check['max_abs_error_over_base_and_slices']}`",
        f"- FD score max abs error: `{fd_check['max_abs_error']}`",
        f"- Slice rows: `{len(artifact['conditional_slice_rows'])}`",
        f"- Runtime seconds: `{artifact['run_manifest']['wall_time_s']}`",
        "",
        "## Decision",
        "",
        f"- Decision: `{artifact['decision_table']['decision']}`",
        f"- Primary criterion: `{artifact['decision_table']['primary_criterion_status']}`",
        f"- Veto diagnostics: `{artifact['decision_table']['veto_diagnostic_status']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {item}" for item in artifact["nonclaims"])
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--points-per-width", type=int, default=DEFAULT_POINTS_PER_WIDTH)
    parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=list(DEFAULT_WIDTHS),
        help="Conditional grid half-widths.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.points_per_width) < 11:
        raise ValueError("points-per-width must be at least 11")
    if any(float(width) <= 0.0 for width in args.widths):
        raise ValueError("all widths must be positive")
    command = (sys.executable, str(Path(__file__).resolve()), *(argv or ()))
    artifact = build_oracle_artifact(
        widths=tuple(float(width) for width in args.widths),
        points_per_width=int(args.points_per_width),
        command=command,
    )
    atomic_write_json(args.output, artifact)
    write_text(args.markdown_output, render_markdown(artifact))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the 24D fully adapted Gaussian mechanics gate for the frozen APF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping


os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf


DTYPE = tf.float32
LOG_TWO_PI = math.log(2.0 * math.pi)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-zhao-cui-highdim-frozen-proposal-apf-review-result-2026-07-22.md"
)
SCHEMA = "bayesfilter.zhao_cui_frozen_proposal_apf.rung0.v1"


def _configure_device(require_gpu: bool) -> Mapping[str, object]:
    physical = tf.config.list_physical_devices("GPU")
    if require_gpu and not physical:
        raise RuntimeError("rung 0 requires a visible trusted GPU")
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
    growth = [tf.config.experimental.get_memory_growth(device) for device in physical]
    if require_gpu and not all(growth):
        raise RuntimeError(f"GPU memory growth verification failed: {growth}")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    return {
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "memory_growth": growth,
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "execution_device": "/GPU:0" if logical else "/CPU:0",
    }


DEVICE = _configure_device("--cpu-reference" not in sys.argv)

from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (  # noqa: E402
    MEASURE_ID,
    ROUTE_CLASSIFICATION,
    SCORE_BACKEND_ID,
    prepare_frozen_proposal_apf_program,
    prepare_frozen_proposal_branch,
)


@dataclass(frozen=True)
class DiagonalGaussianModel:
    """Coupled diagonal LGSSM with two location parameters."""

    dimension: int = 24
    prior_variance: float = 1.25
    transition_variance: float = 0.7
    observation_variance: float = 0.8
    transition_scale: float = 0.65

    def parameter_dim(self) -> int:
        return 2

    def state_dim(self) -> int:
        return self.dimension

    def observation_dim(self) -> int:
        return self.dimension

    def frozen_apf_measure_id(self) -> str:
        return MEASURE_ID

    def frozen_apf_score_backend_id(self) -> str:
        return SCORE_BACKEND_ID

    def initial_log_density(self, theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        return _normal_log_density(state, theta[0], self.prior_variance)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        mean = self.transition_scale * previous + theta[0]
        return _normal_log_density(current, mean, self.transition_variance)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        return _normal_log_density(observation[None, :], state + theta[1], self.observation_variance)

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, state: tf.Tensor
    ) -> tf.Tensor:
        component = tf.reduce_sum(state - theta[0], axis=1) / self.prior_variance
        return tf.stack([component, tf.zeros_like(component)], axis=1)

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        residual = current - (self.transition_scale * previous + theta[0])
        component = tf.reduce_sum(residual, axis=1) / self.transition_variance
        return tf.stack([component, tf.zeros_like(component)], axis=1)

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        residual = observation[None, :] - (state + theta[1])
        component = tf.reduce_sum(residual, axis=1) / self.observation_variance
        return tf.stack([tf.zeros_like(component), component], axis=1)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "diagonal_gaussian_fully_adapted_apf_oracle",
            "dimension": self.dimension,
            "prior_variance": self.prior_variance,
            "transition_variance": self.transition_variance,
            "observation_variance": self.observation_variance,
            "transition_scale": self.transition_scale,
        }


def _normal_log_density(
    value: tf.Tensor, mean: tf.Tensor, variance: float
) -> tf.Tensor:
    residual = value - mean
    dimension = tf.cast(tf.shape(residual)[-1], DTYPE)
    variance_tensor = tf.cast(variance, DTYPE)
    return -0.5 * (
        dimension
        * (tf.cast(LOG_TWO_PI, DTYPE) + tf.math.log(variance_tensor))
        + tf.reduce_sum(tf.square(residual), axis=-1) / variance_tensor
    )


def _observations(time_steps: int, dimension: int) -> tf.Tensor:
    coordinate = tf.linspace(tf.cast(-0.4, DTYPE), tf.cast(0.6, DTYPE), dimension)
    rows = []
    for time_index in range(time_steps):
        time_value = tf.cast(time_index + 1, DTYPE)
        rows.append(
            coordinate
            + 0.04 * tf.math.sin(0.3 * time_value + 2.0 * coordinate)
            + 0.01 * time_value
        )
    return tf.stack(rows)


def _compile_fully_adapted_branch(
    model: DiagonalGaussianModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    particle_count: int,
    seed: int,
):
    dimension = model.dimension
    log_particle_count = tf.math.log(tf.cast(particle_count, DTYPE))
    initial_predictive_variance = model.prior_variance + model.observation_variance
    initial_gain = model.prior_variance / initial_predictive_variance
    initial_posterior_variance = (
        model.prior_variance
        * model.observation_variance
        / initial_predictive_variance
    )
    initial_mean = theta[0] + initial_gain * (
        observations[0] - theta[1] - theta[0]
    )
    initial_noise = tf.random.stateless_normal(
        [particle_count, dimension], [seed, 100], dtype=DTYPE
    )
    current = initial_mean + tf.sqrt(tf.cast(initial_posterior_variance, DTYPE)) * initial_noise
    states = [current]
    initial_log_q = _normal_log_density(
        current, initial_mean, initial_posterior_variance
    )
    ancestors = []
    auxiliary_log_probabilities = []
    transition_log_q = []

    conditional_predictive_variance = (
        model.transition_variance + model.observation_variance
    )
    conditional_gain = (
        model.transition_variance / conditional_predictive_variance
    )
    conditional_posterior_variance = (
        model.transition_variance
        * model.observation_variance
        / conditional_predictive_variance
    )
    uniform_log_weight = -log_particle_count
    for time_index in range(1, int(observations.shape[0])):
        predicted = model.transition_scale * current + theta[0]
        predictive_log_density = _normal_log_density(
            observations[time_index][None, :],
            predicted + theta[1],
            conditional_predictive_variance,
        )
        log_auxiliary = (
            uniform_log_weight
            + predictive_log_density
            - tf.reduce_logsumexp(uniform_log_weight + predictive_log_density)
        )
        ancestor = tf.random.stateless_categorical(
            log_auxiliary[None, :],
            particle_count,
            seed=[seed, 1000 + time_index],
            dtype=tf.int32,
        )[0]
        selected_prediction = tf.gather(predicted, ancestor)
        proposal_mean = selected_prediction + conditional_gain * (
            observations[time_index][None, :]
            - theta[1]
            - selected_prediction
        )
        proposal_noise = tf.random.stateless_normal(
            [particle_count, dimension],
            [seed, 2000 + time_index],
            dtype=DTYPE,
        )
        current = proposal_mean + tf.sqrt(
            tf.cast(conditional_posterior_variance, DTYPE)
        ) * proposal_noise
        states.append(current)
        ancestors.append(ancestor)
        auxiliary_log_probabilities.append(log_auxiliary)
        transition_log_q.append(
            _normal_log_density(
                current, proposal_mean, conditional_posterior_variance
            )
        )

    return prepare_frozen_proposal_branch(
        observations=observations,
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.stack(ancestors),
        auxiliary_log_probabilities=tf.stack(auxiliary_log_probabilities),
        transition_log_proposal_density=tf.stack(transition_log_q),
    )


def _exact_kalman_value_and_score(
    model: DiagonalGaussianModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.fill([model.dimension], theta[0])
    mean_derivative = tf.stack(
        [tf.ones([model.dimension], DTYPE), tf.zeros([model.dimension], DTYPE)],
        axis=1,
    )
    covariance = tf.cast(model.prior_variance, DTYPE)
    value = tf.zeros([], DTYPE)
    score = tf.zeros([2], DTYPE)
    parameter_basis = tf.constant([0.0, 1.0], DTYPE)
    for time_index in range(int(observations.shape[0])):
        if time_index > 0:
            mean = model.transition_scale * mean + theta[0]
            mean_derivative = (
                model.transition_scale * mean_derivative
                + tf.constant([1.0, 0.0], DTYPE)[None, :]
            )
            covariance = (
                model.transition_scale**2 * covariance
                + model.transition_variance
            )
        predictive_mean = mean + theta[1]
        predictive_derivative = mean_derivative + parameter_basis[None, :]
        innovation = observations[time_index] - predictive_mean
        innovation_variance = covariance + model.observation_variance
        value = value + _normal_log_density(
            observations[time_index][None, :],
            predictive_mean,
            float(innovation_variance),
        )[0]
        score = score + tf.reduce_sum(
            innovation[:, None]
            * predictive_derivative
            / innovation_variance,
            axis=0,
        )
        gain = covariance / innovation_variance
        mean = mean + gain * innovation
        mean_derivative = mean_derivative - gain * predictive_derivative
        covariance = (1.0 - gain) * covariance
    return value, score


def _git_payload() -> Mapping[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--short"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_line_count": len(dirty)}


def _write_markdown(path: Path, payload: Mapping[str, object]) -> None:
    gates = payload["gates"]
    diagnostics = payload["diagnostics"]
    lines = [
        "# Zhao-Cui Frozen-Proposal APF Rung-0 Result",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This is a 24D fully adapted diagonal-Gaussian mechanics witness. It is not a TT fit, Austria SIR result, HMC result, or NAWM result.",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | `{value}` |" for name, value in gates.items())
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    lines.extend(
        f"| {name} | `{value}` |" for name, value in diagnostics.items()
    )
    lines.extend(
        [
            "",
            "## Nonclaims",
            "",
            "No source-faithful Zhao-Cui, TT fit quality, posterior correctness, HMC convergence, Austria SIR, NAWM, default-readiness, or superiority claim is made.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--particle-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=220722)
    parser.add_argument("--cpu-reference", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    model = DiagonalGaussianModel(dimension=args.dimension)
    theta = tf.constant([0.18, -0.11], DTYPE)
    observations = _observations(args.time_steps, args.dimension)
    branch = _compile_fully_adapted_branch(
        model,
        theta,
        observations,
        particle_count=args.particle_count,
        seed=args.seed,
    )
    program = prepare_frozen_proposal_apf_program(model, branch)
    compiled = program.compiled()

    compile_started = time.monotonic()
    result = compiled(theta)
    compile_seconds = time.monotonic() - compile_started
    warm_started = time.monotonic()
    warm_result = compiled(theta)
    warmed_seconds = time.monotonic() - warm_started

    fd_step = tf.cast(1e-3, DTYPE)
    fd_entries = []
    for parameter_index in range(model.parameter_dim()):
        direction = tf.one_hot(parameter_index, model.parameter_dim(), dtype=DTYPE)
        plus = compiled(theta + fd_step * direction)["log_likelihood"]
        minus = compiled(theta - fd_step * direction)["log_likelihood"]
        fd_entries.append((plus - minus) / (2.0 * fd_step))
    finite_difference = tf.stack(fd_entries)
    exact_value, exact_score = _exact_kalman_value_and_score(
        model, theta, observations
    )
    score_fd_error = tf.reduce_max(tf.abs(result["score"] - finite_difference))
    final_weight_spread = tf.reduce_max(result["final_log_weights"]) - tf.reduce_min(
        result["final_log_weights"]
    )
    minimum_ess_fraction = result["minimum_ess"] / tf.cast(
        args.particle_count, DTYPE
    )
    output_device = result["log_likelihood"].device
    if DEVICE["logical_gpus"]:
        memory = tf.config.experimental.get_memory_info("GPU:0")
    else:
        memory = {"current": 0, "peak": 0}

    gates = {
        "finite": bool(result["finite"].numpy()),
        "same_scalar_fd_max_abs_error_le_2e-2": float(score_fd_error.numpy()) <= 2e-2,
        "minimum_ess_fraction_ge_0p999": float(minimum_ess_fraction.numpy()) >= 0.999,
        "final_log_weight_spread_le_2e-3": float(final_weight_spread.numpy()) <= 2e-3,
        "xla_enabled": True,
        "expected_device": ("GPU" in output_device) if not args.cpu_reference else ("CPU" in output_device),
        "memory_growth_verified": all(DEVICE["memory_growth"]) if not args.cpu_reference else True,
    }
    passed = all(gates.values())
    diagnostics = {
        "dimension": args.dimension,
        "time_steps": args.time_steps,
        "particle_count": args.particle_count,
        "seed": args.seed,
        "log_likelihood": float(result["log_likelihood"].numpy()),
        "exact_kalman_log_likelihood": float(exact_value.numpy()),
        "descriptive_value_error": float((result["log_likelihood"] - exact_value).numpy()),
        "score": [float(value) for value in result["score"].numpy()],
        "same_scalar_fd_score": [float(value) for value in finite_difference.numpy()],
        "same_scalar_fd_max_abs_error": float(score_fd_error.numpy()),
        "exact_kalman_score": [float(value) for value in exact_score.numpy()],
        "descriptive_score_error": [
            float(value) for value in (result["score"] - exact_score).numpy()
        ],
        "minimum_ess": float(result["minimum_ess"].numpy()),
        "minimum_ess_fraction": float(minimum_ess_fraction.numpy()),
        "maximum_log_weight_spread": float(
            result["maximum_log_weight_spread"].numpy()
        ),
        "final_log_weight_spread": float(final_weight_spread.numpy()),
        "compile_inclusive_seconds": compile_seconds,
        "warmed_seconds": warmed_seconds,
        "output_device": output_device,
        "gpu_allocator_current_bytes": int(memory["current"]),
        "gpu_allocator_peak_bytes": int(memory["peak"]),
    }
    payload = {
        "schema": SCHEMA,
        "status": "PASS_ENGINEERING_RUNG0" if passed else "BLOCK_ENGINEERING_RUNG0",
        "route_id": program.manifest_payload()["route_id"],
        "route_classification": ROUTE_CLASSIFICATION,
        "target_class": program.manifest_payload()["target_class"],
        "proposal_family": "fully_adapted_diagonal_gaussian_oracle_not_tt",
        "program_id": program.program_id,
        "branch_id": branch.branch_id,
        "gates": gates,
        "diagnostics": diagnostics,
        "device": DEVICE,
        "run_manifest": {
            "git": _git_payload(),
            "command": " ".join(sys.argv),
            "environment": sys.executable,
            "tensorflow_version": tf.__version__,
            "dtype": DTYPE.name,
            "tf32_enabled": DEVICE["tf32_enabled"],
            "jit_compile": True,
            "gpu_status": "trusted_visible_gpu" if DEVICE["logical_gpus"] else "cpu_reference",
            "random_seeds": [args.seed],
            "started_at_utc": started_at.isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "output_artifacts": [
                str(args.output_root / "result.json"),
                str(args.output_root / "result.md"),
            ],
            "plan_file": PLAN_PATH,
            "result_file": str(args.output_root / "result.md"),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted" if DEVICE["logical_gpus"] else "explicit_cpu_reference",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "finite-particle differences from exact Kalman value and score",
            "default_readiness": "not assessed",
            "next_evidence_needed": "offline TT proposal fit and fresh nonlinear ESS/rank screen",
        },
        "nonclaims": [
            "no source-faithful Zhao-Cui claim",
            "no TT fit-quality claim",
            "no posterior or HMC claim",
            "no Austria SIR or NAWM claim",
            "no default-readiness or superiority claim",
        ],
    }
    json_path = args.output_root / "result.json"
    markdown_path = args.output_root / "result.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, payload)
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

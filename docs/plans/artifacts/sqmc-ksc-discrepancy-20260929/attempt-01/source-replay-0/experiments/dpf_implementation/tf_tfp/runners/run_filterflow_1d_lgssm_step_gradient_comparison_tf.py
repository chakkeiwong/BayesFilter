"""Controlled 1D LGSSM step-gradient comparison against filterflow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


DTYPE = tf.float64
THETA0 = 0.7
Q = 0.04
R = 0.04
HORIZON = 2
NUM_PARTICLES = 4
INITIAL_PARTICLES = [-1.5, -0.2, 0.4, 1.2]
TRANSITION_NOISES = [[0.0, 0.1, -0.2, 0.3], [0.2, -0.1, 0.0, -0.3]]
OBSERVATIONS = [0.05, -0.1]
ESS_THRESHOLD = 0.9999 * NUM_PARTICLES
EPSILON = 0.25
SCALING = 0.9
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 200
FINITE_DIFF_STEP = 1e-4
PLAN_PATH = "docs/plans/bayesfilter-dpf-1d-lgssm-step-gradient-comparison-plan-2026-06-01.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-1d-lgssm-step-gradient-comparison-result-2026-06-01.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_1d_lgssm_step_gradient_comparison_2026-06-01.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-1d-lgssm-step-gradient-comparison-2026-06-01.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"

TOLERANCES = {
    "predicted_particles": 5e-5,
    "observation_log_likelihoods": 5e-5,
    "normalized_log_weights": 5e-5,
    "transport_cost_matrix": 5e-5,
    "transport_matrix": 5e-5,
    "post_transport_particles": 5e-5,
    "per_step_log_normalizer": 5e-5,
    "total_scalar": 5e-5,
    "row_residual": 1e-4,
    "column_residual": 1e-4,
    "gradient_abs": 1e-3,
    "gradient_rel": 1e-2,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    filterflow = _run_filterflow_subprocess()
    bayesfilter = _bayesfilter_reference()
    comparison = _compare_runs(bayesfilter, filterflow)
    decision = _decision(comparison)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "question": "Controlled 1D LGSSM step-gradient comparison of BayesFilter TF/TFP annealed transport versus executable filterflow.",
        "reference_policy": {
            "canonical_executable_reference": "local patched .localsource/filterflow",
            "fixed_target_sinkhorn": "not used; local comparator only",
            "filterflow_source_mutation": "runner does not mutate filterflow source",
        },
        "filterflow_checkout": _filterflow_checkout_manifest(),
        "model_contract": _model_contract(),
        "tolerances": TOLERANCES,
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "comparison": comparison,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_1d_lgssm_step_gradient_comparison_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _model_contract() -> dict[str, Any]:
    return {
        "model": "controlled_1d_lgssm",
        "state_equation": "x_t = theta x_{t-1} + sqrt(Q) eps_t",
        "observation_equation": "y_t = x_t + sqrt(R) eta_t",
        "theta0": THETA0,
        "Q": Q,
        "R": R,
        "horizon": HORIZON,
        "num_particles": NUM_PARTICLES,
        "initial_particles": INITIAL_PARTICLES,
        "transition_noises": TRANSITION_NOISES,
        "observations": OBSERVATIONS,
        "ess_threshold": ESS_THRESHOLD,
        "expected_resampling_flags": [False, True],
        "epsilon": EPSILON,
        "scaling": SCALING,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_iterations": MAX_ITERATIONS,
        "finite_difference_step": FINITE_DIFF_STEP,
        "scalar": "total log likelihood, sum of per-step predictive log normalizers",
    }


def _bayesfilter_reference() -> dict[str, Any]:
    theta = tf.Variable(THETA0, dtype=DTYPE)
    with tf.GradientTape() as tape:
        scalar_value, ledger = _run_bayesfilter_scalar(theta)
    gradient = tape.gradient(scalar_value, theta)
    plus, _ = _run_bayesfilter_scalar(tf.constant(THETA0 + FINITE_DIFF_STEP, DTYPE))
    minus, _ = _run_bayesfilter_scalar(tf.constant(THETA0 - FINITE_DIFF_STEP, DTYPE))
    finite_difference = (plus - minus) / tf.constant(2.0 * FINITE_DIFF_STEP, DTYPE)
    return {
        "status": "executed",
        "backend": "tensorflow_tensorflow_probability",
        "scalar": _float(scalar_value),
        "gradient_tape": _maybe_float(gradient),
        "finite_difference_gradient": _float(finite_difference),
        "gradient_delta_vs_finite_difference": _maybe_float(gradient - finite_difference if gradient is not None else None),
        "finite_scalar": _finite(scalar_value),
        "finite_gradient": gradient is not None and _finite(gradient),
        "ledger": ledger,
    }


def _run_bayesfilter_scalar(theta: tf.Tensor) -> tuple[tf.Tensor, list[dict[str, Any]]]:
    particles = tf.reshape(tf.constant(INITIAL_PARTICLES, DTYPE), [1, NUM_PARTICLES, 1])
    transition_noises = tf.reshape(tf.constant(TRANSITION_NOISES, DTYPE), [HORIZON, 1, NUM_PARTICLES, 1])
    observations = tf.constant(OBSERVATIONS, DTYPE)
    log_weights = tf.fill([1, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    scalar_value = tf.zeros([], DTYPE)
    ledger: list[dict[str, Any]] = []
    for time_index in range(HORIZON):
        ess = _ess(log_weights)
        flags = ess <= tf.constant(ESS_THRESHOLD, DTYPE)
        pre_particles = particles
        pre_log_weights = log_weights
        cost_matrix = _cost_matrix(pre_particles)
        transport_matrix = tf.eye(NUM_PARTICLES, dtype=DTYPE)[None, :, :]
        row_residual = tf.zeros([], DTYPE)
        column_residual = tf.zeros([], DTYPE)
        iterations_used = 0.0
        if bool(tf.reduce_any(flags).numpy()):
            transported = annealed_transport_resample_tf(
                particles,
                log_weights,
                epsilon=EPSILON,
                scaling=SCALING,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                max_iterations=MAX_ITERATIONS,
                ess_mask=flags,
            )
            particles = transported.particles
            log_weights = transported.log_weights
            transport_matrix = transported.transport_matrix
            row_residual = tf.constant(transported.diagnostics["max_row_residual"], DTYPE)
            column_residual = tf.constant(transported.diagnostics["max_column_residual"], DTYPE)
            iterations_used = float(transported.diagnostics["max_iterations_used"])
        post_transport_particles = particles
        post_transport_log_weights = log_weights
        predicted_particles = theta * particles + tf.sqrt(tf.constant(Q, DTYPE)) * transition_noises[time_index]
        obs_log_prob = _observation_log_prob(predicted_particles, observations[time_index])
        unnormalized = log_weights + obs_log_prob
        normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
        scalar_value = scalar_value + normalizer[0]
        log_weights = unnormalized - normalizer[:, None]
        particles = predicted_particles
        ledger.append(
            _step_ledger(
                time_index=time_index,
                pre_particles=pre_particles,
                pre_log_weights=pre_log_weights,
                ess=ess,
                flags=flags,
                cost_matrix=cost_matrix,
                transport_matrix=transport_matrix,
                post_transport_particles=post_transport_particles,
                post_transport_log_weights=post_transport_log_weights,
                transition_noise=transition_noises[time_index],
                predicted_particles=predicted_particles,
                observation=observations[time_index],
                obs_log_prob=obs_log_prob,
                unnormalized_log_weights=unnormalized,
                normalizer=normalizer,
                post_update_log_weights=log_weights,
                row_residual=row_residual,
                column_residual=column_residual,
                iterations_used=iterations_used,
            )
        )
    return scalar_value, ledger


def _run_filterflow_subprocess() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_1D_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_1D_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow JSON sentinels missing",
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    raw = stdout[start + len("FILTERFLOW_1D_JSON_BEGIN"):end].strip()
    payload = json.loads(raw)
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <controlled 1D LGSSM script>"
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_checkout_manifest() -> dict[str, str]:
    if not FILTERFLOW_PATH.exists():
        return {
            "path": str(FILTERFLOW_PATH),
            "status": "missing",
            "commit": "N/A",
            "branch": "N/A",
            "status_short": "N/A",
        }
    return {
        "path": str(FILTERFLOW_PATH),
        "status": "current_local_patched_checkout",
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "provenance_note": (
            "The comparator is the current local patched filterflow checkout, "
            "not pristine upstream and not asserted clean."
        ),
    }


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _filterflow_script() -> str:
    return textwrap.dedent(
        f"""
        import json
        import math
        import os

        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")

        import tensorflow as tf

        from filterflow.resampling.differentiable.regularized_transport.plan import transport
        from filterflow.resampling.differentiable.regularized_transport.utils import cost, diameter

        THETA0 = {THETA0!r}
        Q = {Q!r}
        R = {R!r}
        HORIZON = {HORIZON}
        NUM_PARTICLES = {NUM_PARTICLES}
        INITIAL_PARTICLES = {INITIAL_PARTICLES!r}
        TRANSITION_NOISES = {TRANSITION_NOISES!r}
        OBSERVATIONS = {OBSERVATIONS!r}
        ESS_THRESHOLD = {ESS_THRESHOLD!r}
        EPSILON = {EPSILON!r}
        SCALING = {SCALING!r}
        CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {MAX_ITERATIONS}
        FINITE_DIFF_STEP = {FINITE_DIFF_STEP!r}
        DTYPE = tf.float32

        def to_json(tensor):
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def maybe_float(tensor):
            if tensor is None:
                return None
            return float(tf.cast(tensor, tf.float64).numpy())

        def finite(tensor):
            if tensor is None:
                return False
            return bool(tf.reduce_all(tf.math.is_finite(tf.cast(tensor, tf.float64))).numpy())

        def ess(logw):
            return tf.exp(-tf.reduce_logsumexp(2.0 * logw, axis=1))

        def observation_log_prob(particles, observation):
            centered = observation - tf.squeeze(particles, axis=2)
            variance = tf.constant(R, DTYPE)
            log_const = tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * variance)
            return -0.5 * (log_const + centered * centered / variance)

        def cost_matrix(particles):
            dimension = tf.cast(particles.shape[-1], DTYPE)
            centered_x = particles - tf.stop_gradient(tf.reduce_mean(particles, axis=1, keepdims=True))
            diameter_value = diameter(particles, particles)
            scale = tf.reshape(diameter_value, [-1, 1, 1]) * tf.sqrt(dimension)
            scaled_x = centered_x / tf.stop_gradient(scale)
            return cost(scaled_x, scaled_x)

        def step_ledger(time_index, pre_particles, pre_log_weights, ess_value, flags,
                        cost_value, transport_matrix, post_transport_particles,
                        post_transport_log_weights, transition_noise, predicted_particles,
                        observation, obs_log_prob, unnormalized, normalizer,
                        post_update_log_weights, row_residual, column_residual):
            return {{
                "time_index": int(time_index),
                "pre_particles": to_json(pre_particles),
                "pre_log_weights": to_json(pre_log_weights),
                "ess": to_json(ess_value),
                "resampling_flags": [bool(v) for v in flags.numpy().tolist()],
                "transport_cost_matrix": to_json(cost_value),
                "transport_matrix": to_json(transport_matrix),
                "post_transport_particles": to_json(post_transport_particles),
                "post_transport_log_weights": to_json(post_transport_log_weights),
                "transition_noise": to_json(transition_noise),
                "predicted_particles": to_json(predicted_particles),
                "observation": float(tf.cast(observation, tf.float64).numpy()),
                "observation_log_likelihoods": to_json(obs_log_prob),
                "unnormalized_log_weights": to_json(unnormalized),
                "per_step_log_normalizer": to_json(normalizer),
                "post_update_log_weights": to_json(post_update_log_weights),
                "row_residual": maybe_float(row_residual),
                "column_residual": maybe_float(column_residual),
                "iterations_used": None,
            }}

        def run_scalar(theta):
            particles = tf.reshape(tf.constant(INITIAL_PARTICLES, DTYPE), [1, NUM_PARTICLES, 1])
            transition_noises = tf.reshape(tf.constant(TRANSITION_NOISES, DTYPE), [HORIZON, 1, NUM_PARTICLES, 1])
            observations = tf.constant(OBSERVATIONS, DTYPE)
            log_weights = tf.fill([1, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
            scalar = tf.zeros([], DTYPE)
            ledger = []
            for time_index in range(HORIZON):
                ess_value = ess(log_weights)
                flags = ess_value <= tf.constant(ESS_THRESHOLD, DTYPE)
                pre_particles = particles
                pre_log_weights = log_weights
                cost_value = cost_matrix(pre_particles)
                transport_matrix = tf.eye(NUM_PARTICLES, dtype=DTYPE)[None, :, :]
                row_residual = tf.zeros([], DTYPE)
                column_residual = tf.zeros([], DTYPE)
                if bool(tf.reduce_any(flags).numpy()):
                    transport_matrix = transport(
                        particles,
                        log_weights,
                        tf.constant(EPSILON, DTYPE),
                        tf.constant(SCALING, DTYPE),
                        tf.constant(CONVERGENCE_THRESHOLD, DTYPE),
                        tf.constant(MAX_ITERATIONS, tf.int32),
                        NUM_PARTICLES,
                    )
                    uniform_log = tf.fill([1, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
                    particles = tf.linalg.matmul(transport_matrix, particles)
                    log_weights = uniform_log
                    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=2) - 1.0))
                    column_target = tf.exp(pre_log_weights) * tf.cast(NUM_PARTICLES, DTYPE)
                    column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=1) - column_target))
                post_transport_particles = particles
                post_transport_log_weights = log_weights
                predicted_particles = theta * particles + tf.sqrt(tf.constant(Q, DTYPE)) * transition_noises[time_index]
                obs_log_prob = observation_log_prob(predicted_particles, observations[time_index])
                unnormalized = log_weights + obs_log_prob
                normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
                scalar = scalar + normalizer[0]
                log_weights = unnormalized - normalizer[:, None]
                particles = predicted_particles
                ledger.append(step_ledger(
                    time_index, pre_particles, pre_log_weights, ess_value, flags,
                    cost_value, transport_matrix, post_transport_particles,
                    post_transport_log_weights, transition_noises[time_index],
                    predicted_particles, observations[time_index], obs_log_prob,
                    unnormalized, normalizer, log_weights, row_residual, column_residual,
                ))
            return scalar, ledger

        theta = tf.Variable(THETA0, dtype=DTYPE)
        with tf.GradientTape() as tape:
            scalar_value, ledger = run_scalar(theta)
        gradient = tape.gradient(scalar_value, theta)
        plus, _ = run_scalar(tf.constant(THETA0 + FINITE_DIFF_STEP, DTYPE))
        minus, _ = run_scalar(tf.constant(THETA0 - FINITE_DIFF_STEP, DTYPE))
        finite_difference = (plus - minus) / tf.constant(2.0 * FINITE_DIFF_STEP, DTYPE)
        payload = {{
            "status": "executed",
            "backend": "executable_filterflow_tensorflow",
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
            "scalar": maybe_float(scalar_value),
            "gradient_tape": maybe_float(gradient),
            "finite_difference_gradient": maybe_float(finite_difference),
            "gradient_delta_vs_finite_difference": maybe_float(gradient - finite_difference if gradient is not None else None),
            "finite_scalar": finite(scalar_value),
            "finite_gradient": gradient is not None and finite(gradient),
            "ledger": ledger,
        }}
        print("FILTERFLOW_1D_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_1D_JSON_END")
        """
    )


def _ess(log_weights: tf.Tensor) -> tf.Tensor:
    return tf.exp(-tf.reduce_logsumexp(2.0 * log_weights, axis=1))


def _observation_log_prob(particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    centered = observation - tf.squeeze(particles, axis=2)
    variance = tf.constant(R, DTYPE)
    log_const = tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * variance)
    return -0.5 * (log_const + centered * centered / variance)


def _cost_matrix(particles: tf.Tensor) -> tf.Tensor:
    centered = particles - tf.stop_gradient(tf.reduce_mean(particles, axis=1, keepdims=True))
    std = tf.math.reduce_std(tf.cast(particles, DTYPE), axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    dimension = tf.cast(tf.shape(particles)[2], DTYPE)
    scale = diameter * tf.sqrt(dimension)
    scaled = centered / tf.stop_gradient(scale[:, None, None])
    xx = tf.reduce_sum(scaled * scaled, axis=2, keepdims=True)
    xy = tf.matmul(scaled, scaled, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(scaled * scaled, axis=-1), axis=1)
    squared = tf.clip_by_value(xx - 2.0 * xy + yy, 0.0, float("inf"))
    return squared / 2.0


def _step_ledger(
    *,
    time_index: int,
    pre_particles: tf.Tensor,
    pre_log_weights: tf.Tensor,
    ess: tf.Tensor,
    flags: tf.Tensor,
    cost_matrix: tf.Tensor,
    transport_matrix: tf.Tensor,
    post_transport_particles: tf.Tensor,
    post_transport_log_weights: tf.Tensor,
    transition_noise: tf.Tensor,
    predicted_particles: tf.Tensor,
    observation: tf.Tensor,
    obs_log_prob: tf.Tensor,
    unnormalized_log_weights: tf.Tensor,
    normalizer: tf.Tensor,
    post_update_log_weights: tf.Tensor,
    row_residual: tf.Tensor,
    column_residual: tf.Tensor,
    iterations_used: float,
) -> dict[str, Any]:
    return {
        "time_index": int(time_index),
        "pre_particles": _json_tensor(pre_particles),
        "pre_log_weights": _json_tensor(pre_log_weights),
        "ess": _json_tensor(ess),
        "resampling_flags": [bool(value) for value in flags.numpy().tolist()],
        "transport_cost_matrix": _json_tensor(cost_matrix),
        "transport_matrix": _json_tensor(transport_matrix),
        "post_transport_particles": _json_tensor(post_transport_particles),
        "post_transport_log_weights": _json_tensor(post_transport_log_weights),
        "transition_noise": _json_tensor(transition_noise),
        "predicted_particles": _json_tensor(predicted_particles),
        "observation": _float(observation),
        "observation_log_likelihoods": _json_tensor(obs_log_prob),
        "unnormalized_log_weights": _json_tensor(unnormalized_log_weights),
        "per_step_log_normalizer": _json_tensor(normalizer),
        "post_update_log_weights": _json_tensor(post_update_log_weights),
        "row_residual": _float(row_residual),
        "column_residual": _float(column_residual),
        "iterations_used": iterations_used,
    }


def _compare_runs(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {
            "status": "blocked_filterflow_reference",
            "blocker": filterflow.get("blocker", "unknown filterflow blocker"),
        }
    rows = []
    max_deltas = {
        "predicted_particles": 0.0,
        "observation_log_likelihoods": 0.0,
        "normalized_log_weights": 0.0,
        "transport_cost_matrix": 0.0,
        "transport_matrix": 0.0,
        "post_transport_particles": 0.0,
        "per_step_log_normalizer": 0.0,
        "row_residual": 0.0,
        "column_residual": 0.0,
    }
    absolute_residuals = {
        "bayesfilter_max_row_residual": 0.0,
        "bayesfilter_max_column_residual": 0.0,
        "filterflow_max_row_residual": 0.0,
        "filterflow_max_column_residual": 0.0,
    }
    trigger_match = True
    for bf_step, ff_step in zip(bayesfilter["ledger"], filterflow["ledger"], strict=True):
        row = {"time_index": bf_step["time_index"], "deltas": {}}
        trigger_match = trigger_match and bf_step["resampling_flags"] == ff_step["resampling_flags"]
        row["resampling_flags_bayesfilter"] = bf_step["resampling_flags"]
        row["resampling_flags_filterflow"] = ff_step["resampling_flags"]
        field_map = {
            "predicted_particles": ("predicted_particles", "predicted_particles"),
            "observation_log_likelihoods": ("observation_log_likelihoods", "observation_log_likelihoods"),
            "normalized_log_weights": ("post_update_log_weights", "post_update_log_weights"),
            "transport_cost_matrix": ("transport_cost_matrix", "transport_cost_matrix"),
            "transport_matrix": ("transport_matrix", "transport_matrix"),
            "post_transport_particles": ("post_transport_particles", "post_transport_particles"),
            "per_step_log_normalizer": ("per_step_log_normalizer", "per_step_log_normalizer"),
        }
        for label, (bf_key, ff_key) in field_map.items():
            delta = _max_abs_nested(bf_step[bf_key], ff_step[ff_key])
            row["deltas"][label] = delta
            max_deltas[label] = max(max_deltas[label], delta)
        for label in ("row_residual", "column_residual"):
            delta = abs(float(bf_step[label]) - float(ff_step[label]))
            row["deltas"][label] = delta
            max_deltas[label] = max(max_deltas[label], delta)
        absolute_residuals["bayesfilter_max_row_residual"] = max(
            absolute_residuals["bayesfilter_max_row_residual"],
            abs(float(bf_step["row_residual"])),
        )
        absolute_residuals["bayesfilter_max_column_residual"] = max(
            absolute_residuals["bayesfilter_max_column_residual"],
            abs(float(bf_step["column_residual"])),
        )
        absolute_residuals["filterflow_max_row_residual"] = max(
            absolute_residuals["filterflow_max_row_residual"],
            abs(float(ff_step["row_residual"])),
        )
        absolute_residuals["filterflow_max_column_residual"] = max(
            absolute_residuals["filterflow_max_column_residual"],
            abs(float(ff_step["column_residual"])),
        )
        rows.append(row)
    scalar_delta = abs(float(bayesfilter["scalar"]) - float(filterflow["scalar"]))
    gradient_delta = _optional_abs_delta(bayesfilter["gradient_tape"], filterflow["gradient_tape"])
    bf_grad_fd_delta = abs(float(bayesfilter["gradient_tape"]) - float(bayesfilter["finite_difference_gradient"]))
    ff_grad_fd_delta = abs(float(filterflow["gradient_tape"]) - float(filterflow["finite_difference_gradient"]))
    gradient_rel_delta = _relative_delta(bayesfilter["gradient_tape"], filterflow["gradient_tape"])
    expected_flags_match = [
        step["resampling_flags"][0] for step in bayesfilter["ledger"]
    ] == [False, True]
    pass_status = {
        "trigger_match": trigger_match,
        "expected_trigger_pattern": expected_flags_match,
        "finite_values": bool(
            bayesfilter["finite_scalar"]
            and bayesfilter["finite_gradient"]
            and filterflow["finite_scalar"]
            and filterflow["finite_gradient"]
        ),
        "scalar_within_tolerance": scalar_delta <= TOLERANCES["total_scalar"],
        "gradient_within_tolerance": (
            gradient_delta <= TOLERANCES["gradient_abs"] or gradient_rel_delta <= TOLERANCES["gradient_rel"]
        ),
        "bayesfilter_gradient_fd_within_tolerance": (
            bf_grad_fd_delta <= TOLERANCES["gradient_abs"]
            or _relative_delta(bayesfilter["gradient_tape"], bayesfilter["finite_difference_gradient"])
            <= TOLERANCES["gradient_rel"]
        ),
        "filterflow_gradient_fd_within_tolerance": (
            ff_grad_fd_delta <= TOLERANCES["gradient_abs"]
            or _relative_delta(filterflow["gradient_tape"], filterflow["finite_difference_gradient"])
            <= TOLERANCES["gradient_rel"]
        ),
    }
    ledger_within_tolerance = True
    for key, value in max_deltas.items():
        ledger_within_tolerance = ledger_within_tolerance and value <= TOLERANCES[key]
    pass_status["ledger_within_tolerance"] = ledger_within_tolerance
    pass_status["absolute_residuals_within_tolerance"] = (
        absolute_residuals["bayesfilter_max_row_residual"] <= TOLERANCES["row_residual"]
        and absolute_residuals["filterflow_max_row_residual"] <= TOLERANCES["row_residual"]
        and absolute_residuals["bayesfilter_max_column_residual"] <= TOLERANCES["column_residual"]
        and absolute_residuals["filterflow_max_column_residual"] <= TOLERANCES["column_residual"]
    )
    return {
        "status": "compared",
        "rows": rows,
        "max_deltas": max_deltas,
        "absolute_residuals": absolute_residuals,
        "scalar_delta": scalar_delta,
        "gradient_delta": gradient_delta,
        "gradient_relative_delta": gradient_rel_delta,
        "bayesfilter_gradient_fd_delta": bf_grad_fd_delta,
        "filterflow_gradient_fd_delta": ff_grad_fd_delta,
        "pass_status": pass_status,
    }


def _decision(comparison: dict[str, Any]) -> str:
    if comparison["status"] != "compared":
        return "one_d_lgssm_step_gradient_filterflow_blocker"
    status = comparison["pass_status"]
    if not status["expected_trigger_pattern"] or not status["trigger_match"]:
        return "one_d_lgssm_step_gradient_trigger_veto"
    if not status["finite_values"]:
        return "one_d_lgssm_step_gradient_nonfinite_veto"
    if all(status.values()):
        return "one_d_lgssm_step_gradient_comparison_passed"
    filterflow_contract_keys = {
        "absolute_residuals_within_tolerance",
        "expected_trigger_pattern",
        "finite_values",
        "gradient_within_tolerance",
        "ledger_within_tolerance",
        "scalar_within_tolerance",
        "trigger_match",
    }
    if all(status[key] for key in filterflow_contract_keys):
        return "one_d_lgssm_step_gradient_filterflow_contract_matches_fd_veto"
    return "one_d_lgssm_step_gradient_mismatch_detected"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "model_contract",
        "bayesfilter",
        "filterflow",
        "comparison",
        "filterflow_checkout",
        "tolerances",
        "run_manifest",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    allowed = {
        "one_d_lgssm_step_gradient_comparison_passed",
        "one_d_lgssm_step_gradient_filterflow_contract_matches_fd_veto",
        "one_d_lgssm_step_gradient_mismatch_detected",
        "one_d_lgssm_step_gradient_trigger_veto",
        "one_d_lgssm_step_gradient_nonfinite_veto",
        "one_d_lgssm_step_gradient_filterflow_blocker",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    manifest = payload["run_manifest"]
    if not manifest.get("cpu_only"):
        raise ValueError("parent run manifest is not marked CPU-only")
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("parent pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError("parent CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"parent GPU devices visible: {manifest.get('gpu_devices_visible')}")
    comparison = payload["comparison"]
    if payload["decision"] == "one_d_lgssm_step_gradient_filterflow_blocker":
        if payload["filterflow"].get("status") == "executed":
            raise ValueError("filterflow blocker decision with executed filterflow payload")
        return
    if payload["filterflow"].get("status") != "executed":
        raise ValueError("filterflow did not execute for non-blocker decision")
    checkout = payload["filterflow_checkout"]
    if checkout.get("status") != "current_local_patched_checkout":
        raise ValueError(f"unexpected filterflow checkout status: {checkout.get('status')}")
    if not checkout.get("commit"):
        raise ValueError("missing filterflow checkout commit")
    if comparison.get("status") != "compared":
        raise ValueError(f"comparison did not run: {comparison.get('status')}")
    filterflow_cpu = payload["filterflow"].get("cpu_only_manifest", {})
    if filterflow_cpu.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("filterflow pre-import CUDA_VISIBLE_DEVICES is not -1")
    if filterflow_cpu.get("cuda_visible_devices") != "-1":
        raise ValueError("filterflow CUDA_VISIBLE_DEVICES is not -1")
    if filterflow_cpu.get("gpu_devices_visible") != []:
        raise ValueError(f"filterflow GPU devices visible: {filterflow_cpu.get('gpu_devices_visible')}")
    pass_status = comparison["pass_status"]
    if not pass_status["trigger_match"]:
        raise ValueError("BayesFilter and filterflow trigger flags do not match")
    if not pass_status["expected_trigger_pattern"]:
        raise ValueError("expected trigger pattern was not observed")
    if not pass_status["absolute_residuals_within_tolerance"]:
        raise ValueError("absolute transport residuals exceeded tolerance")
    if payload["decision"] == "one_d_lgssm_step_gradient_comparison_passed" and not all(pass_status.values()):
        raise ValueError("pass decision inconsistent with failing pass_status")
    if payload["decision"] == "one_d_lgssm_step_gradient_filterflow_contract_matches_fd_veto":
        required_true = {
            "absolute_residuals_within_tolerance",
            "expected_trigger_pattern",
            "finite_values",
            "gradient_within_tolerance",
            "ledger_within_tolerance",
            "scalar_within_tolerance",
            "trigger_match",
        }
        for key in required_true:
            if not pass_status[key]:
                raise ValueError(f"filterflow-contract match decision with failed prerequisite: {key}")
        if (
            pass_status["bayesfilter_gradient_fd_within_tolerance"]
            and pass_status["filterflow_gradient_fd_within_tolerance"]
        ):
            raise ValueError("filterflow-contract match decision without a finite-difference veto")
    if payload["decision"] == "one_d_lgssm_step_gradient_mismatch_detected":
        required_true = {
            "trigger_match",
            "expected_trigger_pattern",
            "finite_values",
            "scalar_within_tolerance",
            "ledger_within_tolerance",
        }
        for key in required_true:
            if not pass_status[key]:
                raise ValueError(f"mismatch decision with failed prerequisite: {key}")
        if pass_status["gradient_within_tolerance"]:
            raise ValueError("mismatch decision but gradient comparison passed")
    if payload["filterflow"].get("status") == "executed":
        if len(payload["bayesfilter"]["ledger"]) != HORIZON or len(payload["filterflow"]["ledger"]) != HORIZON:
            raise ValueError("ledger horizon mismatch")
        bf_flags = [step["resampling_flags"][0] for step in payload["bayesfilter"]["ledger"]]
        if bf_flags != [False, True]:
            raise ValueError(f"BayesFilter trigger pattern mismatch: {bf_flags}")


def _markdown(payload: dict[str, Any]) -> str:
    comparison = payload["comparison"]
    lines = [
        "# 1D LGSSM Step Gradient Comparison",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Setup",
        "",
        "| Key | Value |",
        "| --- | --- |",
        f"| theta0 | `{THETA0}` |",
        f"| horizon | `{HORIZON}` |",
        f"| particles | `{NUM_PARTICLES}` |",
        f"| Q | `{Q}` |",
        f"| R | `{R}` |",
        f"| ESS threshold | `{ESS_THRESHOLD}` |",
        f"| epsilon | `{EPSILON}` |",
        f"| scaling | `{SCALING}` |",
        "",
        "This report evidences matched fixed numeric inputs, same scalar ledger,",
        "matched executable transport output, and AD-vs-finite-difference",
        "mismatch for the same scalar. It does not verify or compare",
        "filterflow's internal annealing iteration count, epsilon schedule,",
        "or convergence trajectory.",
        "",
    ]
    if comparison["status"] != "compared":
        lines.extend(["## Blocker", "", f"`{comparison['blocker']}`", ""])
    else:
        lines.extend(
            [
                "## Scalar And Gradient",
                "",
                "| Metric | BayesFilter | filterflow | delta |",
                "| --- | ---: | ---: | ---: |",
                (
                    f"| total scalar | `{payload['bayesfilter']['scalar']}` | "
                    f"`{payload['filterflow']['scalar']}` | `{comparison['scalar_delta']}` |"
                ),
                (
                    f"| GradientTape gradient | `{payload['bayesfilter']['gradient_tape']}` | "
                    f"`{payload['filterflow']['gradient_tape']}` | `{comparison['gradient_delta']}` |"
                ),
                (
                    f"| finite-difference gradient | `{payload['bayesfilter']['finite_difference_gradient']}` | "
                    f"`{payload['filterflow']['finite_difference_gradient']}` | "
                    f"`{_optional_abs_delta(payload['bayesfilter']['finite_difference_gradient'], payload['filterflow']['finite_difference_gradient'])}` |"
                ),
                "",
                "## Max Step-Ledger Deltas",
                "",
                "| Field | max abs delta | tolerance |",
                "| --- | ---: | ---: |",
            ]
        )
        for key, value in comparison["max_deltas"].items():
            lines.append(f"| `{key}` | `{value}` | `{TOLERANCES[key]}` |")
        lines.extend(
            [
                "",
        "## Pass Status",
                "",
                "| Check | Status |",
                "| --- | --- |",
            ]
        )
        for key, value in comparison["pass_status"].items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.extend(
            [
                "",
                "## Absolute Residuals",
                "",
                "| Field | value | tolerance |",
                "| --- | ---: | ---: |",
            ]
        )
        residual_tolerance = {
            "bayesfilter_max_row_residual": TOLERANCES["row_residual"],
            "bayesfilter_max_column_residual": TOLERANCES["column_residual"],
            "filterflow_max_row_residual": TOLERANCES["row_residual"],
            "filterflow_max_column_residual": TOLERANCES["column_residual"],
        }
        for key, value in comparison["absolute_residuals"].items():
            lines.append(f"| `{key}` | `{value}` | `{residual_tolerance[key]}` |")
        lines.extend(
            [
                "",
                "## Transport Diagnostic Availability",
                "",
                "| Diagnostic | BayesFilter | filterflow | Comparison status |",
                "| --- | --- | --- | --- |",
                "| triggered-step iteration count | `62.0` | not available from this wrapper | explanatory only; not compared |",
                "| epsilon schedule | not serialized | not serialized | not compared |",
                "| convergence trajectory | not serialized | not serialized | not compared |",
            ]
        )
    lines.extend(
        [
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No DSGE/NAWM validation is concluded.",
        "No banking/model-risk claim is concluded.",
        "No monograph claim is concluded.",
        "No gradient correctness beyond this fixed 1D scalar fixture is concluded.",
    ]


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


def _json_tensor(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _maybe_float(value: tf.Tensor | None) -> float | None:
    if value is None:
        return None
    return _float(value)


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _max_abs_nested(left: Any, right: Any) -> float:
    left_tensor = tf.constant(left, DTYPE)
    right_tensor = tf.constant(right, DTYPE)
    return _float(tf.reduce_max(tf.abs(left_tensor - right_tensor)))


def _optional_abs_delta(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return float("inf")
    return abs(float(left) - float(right))


def _relative_delta(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return float("inf")
    denominator = max(abs(float(right)), 1e-12)
    return abs(float(left) - float(right)) / denominator


if __name__ == "__main__":
    raise SystemExit(main())

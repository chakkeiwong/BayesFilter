#!/usr/bin/env python
"""P91 trusted GPU/XLA HMC smoke for local Zhao-Cui SIR d18 target component."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

import tensorflow as tf
import tensorflow_probability as tfp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bayesfilter.highdim as highdim


DTYPE = tf.float64
FINAL_TIME = 4
REFRESH_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase7-hmc-smoke-exact-command-refresh-2026-06-29.md"
)
MANIFEST_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase7-hmc-smoke-2026-06-29.json"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase7-hmc-smoke-result-2026-06-29.md"
)
PHASE8_SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase8-packaging-release-subplan-2026-06-29.md"
)


def _fixture_inputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    theta = tf.constant([0.0, 0.0, 0.0], dtype=DTYPE)
    model = highdim.parameterized_zhao_cui_sir_austria_model()
    states = [model.base_model.initial_mean]
    observations = []
    observation_offsets = tf.linspace(
        tf.constant(-0.2, dtype=DTYPE),
        tf.constant(0.2, dtype=DTYPE),
        model.observation_dim(),
    )
    for time_index in range(FINAL_TIME + 1):
        current = states[-1]
        observations.append(model.infectious_components(current)[0] + observation_offsets)
        if time_index < FINAL_TIME:
            transition_mean = model.transition_mean(theta, current[tf.newaxis, :])[0]
            perturbation = tf.linspace(
                tf.constant(-0.03, dtype=DTYPE),
                tf.constant(0.03, dtype=DTYPE),
                model.state_dim(),
            ) * tf.cast(time_index + 1, DTYPE)
            states.append(transition_mean + perturbation)
    return theta, tf.stack(states), tf.stack(observations)


def _make_target_log_prob(states: tf.Tensor, observations: tf.Tensor):
    state_path = tf.convert_to_tensor(states, dtype=DTYPE)
    observation_path = tf.convert_to_tensor(observations, dtype=DTYPE)

    def target_log_prob(theta: tf.Tensor) -> tf.Tensor:
        theta_tensor = tf.convert_to_tensor(theta, dtype=DTYPE)
        if theta_tensor.shape.rank == 1:
            return highdim.zhao_cui_sir_austria_local_complete_data_log_density_xla(
                theta_tensor,
                state_path,
                observation_path,
            )
        return tf.map_fn(
            lambda current_theta: highdim.zhao_cui_sir_austria_local_complete_data_log_density_xla(
                current_theta,
                state_path,
                observation_path,
            ),
            theta_tensor,
            fn_output_signature=tf.TensorSpec([], DTYPE),
            parallel_iterations=1,
        )

    return target_log_prob


def _initial_state(chains: int) -> tf.Tensor:
    base = tf.zeros([int(chains), 3], dtype=DTYPE)
    offsets = tf.linspace(
        tf.constant(-0.01, dtype=DTYPE),
        tf.constant(0.01, dtype=DTYPE),
        int(chains),
    )
    return base + tf.stack(
        [
            offsets,
            -offsets,
            tf.zeros_like(offsets),
        ],
        axis=1,
    )


def _make_runner(
    *,
    target_log_prob: Any,
    initial_state: tf.Tensor,
    num_results: int,
    num_burnin_steps: int,
    step_size: float,
    num_leapfrog_steps: int,
    xla: bool,
):
    def sample_value_and_gradient(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        current_theta = tf.reshape(tf.convert_to_tensor(theta, dtype=DTYPE), [3])
        with tf.GradientTape() as tape:
            tape.watch(current_theta)
            value = target_log_prob(current_theta)
        gradient = tape.gradient(value, current_theta)
        if gradient is None:
            gradient = tf.fill(
                tf.shape(current_theta),
                tf.constant(float("nan"), dtype=DTYPE),
            )
        return value, gradient

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(float(step_size), dtype=DTYPE),
        num_leapfrog_steps=int(num_leapfrog_steps),
    )

    @tf.function(
        jit_compile=bool(xla),
        input_signature=[tf.TensorSpec([2], tf.int32)],
    )
    def run_chain(seed: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        samples, trace = tfp.mcmc.sample_chain(
            num_results=int(num_results),
            num_burnin_steps=int(num_burnin_steps),
            current_state=initial_state,
            kernel=kernel,
            trace_fn=lambda _state, kernel_results: (
                kernel_results.log_accept_ratio,
                kernel_results.is_accepted,
            ),
            seed=seed,
        )
        log_accept_ratio, is_accepted = trace
        flat_samples = tf.reshape(samples, [-1, 3])
        target_values, sample_gradients = tf.map_fn(
            sample_value_and_gradient,
            flat_samples,
            fn_output_signature=(
                tf.TensorSpec([], DTYPE),
                tf.TensorSpec([3], DTYPE),
            ),
        )
        return samples, log_accept_ratio, is_accepted, target_values, sample_gradients

    return run_chain


def _initial_value_gradient(
    target_log_prob: Any,
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = target_log_prob(theta)
    gradient = tape.gradient(value, theta)
    if gradient is None:
        gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=DTYPE))
    return value, gradient


def _run_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    requested_xla = _str_to_bool(args.xla)
    if not requested_xla:
        raise SystemExit("P91 Phase 7 requires --xla true")
    manifest_path = Path(str(args.manifest))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    device_payload = _device_payload()
    errors: list[str] = []
    if not tf.config.list_physical_devices("GPU"):
        errors.append("TensorFlow reported no physical GPU devices.")

    theta0, states, observations = _fixture_inputs()
    target_log_prob = _make_target_log_prob(states, observations)
    initial_value, initial_gradient = _initial_value_gradient(target_log_prob, theta0)
    initial_state = _initial_state(int(args.chains))

    trace_count_before = None
    trace_count_after_first = None
    trace_count_after_second = None
    first_call_seconds = None
    second_call_seconds = None
    first_outputs = None
    second_outputs = None
    oom_status = False
    retry_count = 0

    if not errors:
        runner = _make_runner(
            target_log_prob=target_log_prob,
            initial_state=initial_state,
            num_results=int(args.num_results),
            num_burnin_steps=int(args.num_burnin_steps),
            step_size=float(args.step_size),
            num_leapfrog_steps=int(args.num_leapfrog_steps),
            xla=requested_xla,
        )
        trace_count_before = int(runner.experimental_get_tracing_count())
        seed = tf.constant([int(args.seed), int(args.seed) + 1], dtype=tf.int32)
        try:
            start = time.perf_counter()
            with tf.device("/GPU:0"):
                first_outputs = runner(seed)
            _materialize(first_outputs)
            first_call_seconds = time.perf_counter() - start
            trace_count_after_first = int(runner.experimental_get_tracing_count())

            start = time.perf_counter()
            with tf.device("/GPU:0"):
                second_outputs = runner(seed)
            _materialize(second_outputs)
            second_call_seconds = time.perf_counter() - start
            trace_count_after_second = int(runner.experimental_get_tracing_count())
        except tf.errors.ResourceExhaustedError as exc:
            oom_status = True
            errors.append(f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - manifest for runtime failures.
            errors.append(f"{type(exc).__name__}: {exc}")

    outputs = second_outputs or first_outputs
    diagnostics = _diagnostics_from_outputs(outputs)
    post_warmup_retrace = (
        trace_count_after_second != trace_count_after_first
        if trace_count_after_second is not None and trace_count_after_first is not None
        else True
    )
    gpu_outputs = bool(
        diagnostics.get("output_devices")
        and all("GPU" in str(device).upper() for device in diagnostics["output_devices"])
    )
    finite_initial = bool(
        tf.math.is_finite(initial_value).numpy()
        and tf.reduce_all(tf.math.is_finite(initial_gradient)).numpy()
    )
    veto_reasons = []
    if errors:
        veto_reasons.extend(errors)
    if not finite_initial:
        veto_reasons.append("initial target value or gradient nonfinite")
    if not diagnostics.get("sample_finite", False):
        veto_reasons.append("samples nonfinite")
    if not diagnostics.get("target_values_finite", False):
        veto_reasons.append("sample target values nonfinite")
    if not diagnostics.get("sample_gradients_finite", False):
        veto_reasons.append("sample gradients nonfinite")
    if not diagnostics.get("log_accept_ratio_finite", False):
        veto_reasons.append("log_accept_ratio nonfinite")
    if not gpu_outputs:
        veto_reasons.append("missing GPU output devices")
    if post_warmup_retrace:
        veto_reasons.append("post-warmup retrace detected")
    if oom_status:
        veto_reasons.append("OOM")
    if retry_count > 0:
        veto_reasons.append("retry_count > 0")

    status = (
        "PASS_P91_PHASE7_HMC_SMOKE"
        if not veto_reasons
        else "BLOCK_P91_PHASE7_HMC_SMOKE"
    )
    payload = {
        "schema_version": "p91.phase7.hmc_smoke.v1",
        "status": status,
        "git_commit": _git_commit(),
        "worktree_status": _git_dirty_note(),
        "command": _command_string(args),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "trusted_escalated_gpu_run": True,
        "requested_xla_status": requested_xla,
        "actual_xla_status": requested_xla,
        "devices": device_payload,
        "target_scope": "local_complete_data_zhao_cui_sir_d18_component",
        "random_seed": int(args.seed),
        "chains": int(args.chains),
        "num_results": int(args.num_results),
        "num_burnin_steps": int(args.num_burnin_steps),
        "step_size": float(args.step_size),
        "num_leapfrog_steps": int(args.num_leapfrog_steps),
        "initial_target_value": float(initial_value.numpy()),
        "initial_gradient": _tensor_to_float_list(initial_gradient),
        "initial_value_gradient_finite": finite_initial,
        "diagnostics": diagnostics,
        "first_call_seconds": first_call_seconds,
        "second_call_seconds": second_call_seconds,
        "trace_count_before": trace_count_before,
        "trace_count_after_first": trace_count_after_first,
        "trace_count_after_second": trace_count_after_second,
        "post_warmup_retrace_detected": bool(post_warmup_retrace),
        "retry_count": retry_count,
        "oom_status": oom_status,
        "divergence_status": "unavailable",
        "divergence_count": None,
        "divergence_source": "native boolean divergence field not exposed in trace_fn",
        "veto_reasons": veto_reasons,
        "decision_veto_status": {
            "hard_veto_failed": bool(veto_reasons),
            "primary_criterion_passed": not bool(veto_reasons),
        },
        "phase3_limited_fd_caveat": (
            "Phase 3 is owner-accepted for continuation with caveats; "
            "not a full FD pass."
        ),
        "blocker_statuses_preserved": {
            "full_observed_data_filtering_score_identity": "NOT_CLAIMED",
            "previous_marginal_derivative": (
                "BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED"
            ),
            "fixed_ttsirt_transport_derivative": (
                "BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED"
            ),
            "full_source_route_fd": "BLOCK_FULL_SOURCE_ROUTE_FD_NOT_CLAIMED",
        },
        "artifact_paths": {
            "exact_command_refresh": REFRESH_PATH,
            "manifest": str(manifest_path),
            "result": RESULT_PATH,
            "phase8_subplan": PHASE8_SUBPLAN_PATH,
        },
        "nonclaims": [
            "no posterior correctness",
            "no convergence claim",
            "no exact likelihood correctness",
            "no full observed-data/filtering HMC target readiness",
            "no package/default readiness",
            "no production readiness",
        ],
    }
    return payload, 0 if status.startswith("PASS_") else 1


def _diagnostics_from_outputs(outputs: tuple[tf.Tensor, ...] | None) -> dict[str, Any]:
    if outputs is None:
        return {
            "sample_finite": False,
            "target_values_finite": False,
            "sample_gradients_finite": False,
            "log_accept_ratio_finite": False,
            "output_devices": [],
            "acceptance_rate": None,
        }
    samples, log_accept_ratio, is_accepted, target_values, sample_gradients = outputs
    return {
        "sample_finite": bool(tf.reduce_all(tf.math.is_finite(samples)).numpy()),
        "target_values_finite": bool(tf.reduce_all(tf.math.is_finite(target_values)).numpy()),
        "sample_gradients_finite": bool(
            tf.reduce_all(tf.math.is_finite(sample_gradients)).numpy()
        ),
        "log_accept_ratio_finite": bool(
            tf.reduce_all(tf.math.is_finite(log_accept_ratio)).numpy()
        ),
        "output_devices": [
            samples.device,
            log_accept_ratio.device,
            is_accepted.device,
            target_values.device,
            sample_gradients.device,
        ],
        "acceptance_rate": float(tf.reduce_mean(tf.cast(is_accepted, tf.float64)).numpy()),
        "sample_mean": _tensor_to_float_list(tf.reduce_mean(samples, axis=[0, 1])),
        "sample_standard_deviation": _tensor_to_float_list(
            tf.math.reduce_std(samples, axis=[0, 1])
        ),
        "log_accept_ratio_min": float(tf.reduce_min(log_accept_ratio).numpy()),
        "log_accept_ratio_max": float(tf.reduce_max(log_accept_ratio).numpy()),
        "target_value_min": float(tf.reduce_min(target_values).numpy()),
        "target_value_max": float(tf.reduce_max(target_values).numpy()),
    }


def _materialize(outputs: tuple[tf.Tensor, ...]) -> None:
    for output in outputs:
        _ = output.numpy()


def _tensor_to_float_list(tensor: tf.Tensor) -> list[float]:
    return [float(value) for value in tf.reshape(tensor, [-1]).numpy()]


def _device_payload() -> dict[str, Any]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    return {
        "physical_gpus": [device.name for device in physical_gpus],
        "logical_gpus": [device.name for device in logical_gpus],
        "gpu_names": [
            tf.config.experimental.get_device_details(device).get("device_name", "N/A")
            for device in physical_gpus
        ],
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _git_dirty_note() -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "UNKNOWN"
    return "dirty research worktree" if completed.stdout.strip() else "clean"


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _command_string(args: argparse.Namespace) -> str:
    return (
        "python scripts/p91_hmc_smoke.py --manifest "
        f"{args.manifest} --chains {args.chains} --num-results {args.num_results} "
        f"--num-burnin-steps {args.num_burnin_steps} --step-size {args.step_size} "
        f"--num-leapfrog-steps {args.num_leapfrog_steps} --seed {args.seed} "
        f"--xla {str(_str_to_bool(args.xla)).lower()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--num-results", type=int, default=8)
    parser.add_argument("--num-burnin-steps", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=1.0e-4)
    parser.add_argument("--num-leapfrog-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument("--xla", default="true")
    args = parser.parse_args()
    payload, exit_code = _run_smoke(args)
    manifest_path = Path(str(args.manifest))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "manifest": str(manifest_path)}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

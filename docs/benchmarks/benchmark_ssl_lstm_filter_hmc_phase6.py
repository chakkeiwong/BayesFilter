"""SSL-LSTM Phase 6 shared benchmark runner and invariant metrics.

This harness evaluates the admitted SSL-LSTM filter-adapter lanes under one
shared deterministic fixture. It is benchmark-only: it does not run HMC, does
not claim filter sufficiency, and does not promote parameter-by-parameter
matching.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.nonlinear.ssl_lstm_protocol import (  # noqa: E402
    SSLLSTMStaticConfig,
    validate_ssl_lstm_value_score_artifact,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_filter  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (  # noqa: E402
    build_ssl_lstm_debug_value_score_artifact,
    make_ssl_lstm_fixed_sgqf_components,
    make_ssl_lstm_svd_ukf_components,
    ssl_lstm_observation,
    ssl_lstm_transition,
    tf_ssl_lstm_fixed_sgqf_score,
    tf_ssl_lstm_svd_ukf_score,
    unpack_ssl_lstm_parameters,
)
from bayesfilter.nonlinear.ssl_lstm_zhaocui_fixed_adapter import (  # noqa: E402
    SSLLSTMZhaoCuiFixedManifest,
    build_ssl_lstm_zhaocui_fixed_value_score_artifact,
    make_ssl_lstm_zhaocui_fixed_components,
    tf_ssl_lstm_zhaocui_fixed_score,
)
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter  # noqa: E402
from bayesfilter.runtime import atomic_write_json  # noqa: E402


PLAN_PATH = "docs/plans/bayesfilter-ssl-lstm-filter-hmc-master-program-2026-07-04.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-filter-hmc-phase6-benchmark-runner-invariant-metrics-"
    "subplan-2026-07-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-filter-hmc-phase6-benchmark-runner-invariant-metrics-"
    "result-2026-07-04.md"
)
DATASET_MANIFEST_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-filter-hmc-phase6-dataset-manifest-2026-07-04.json"
)
GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"

NONCLAIMS = (
    "shared benchmark harness only",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not filter sufficiency evidence",
    "not parameter-recovery evidence",
    "not a ranking claim",
    "parameter-by-parameter matching is not a primary criterion",
    "heldout predictive log score is a filter-likelihood proxy",
    "blocked candidate rows are status-only",
)

ADMITTED_FILTERS = ("fixed_sgqf", "svd_ukf", "zhaocui_fixed")
BLOCKED_FILTERS = ("ledh_streaming_ot",)
PHASE6_BENCHMARK_PHASE = "PHASE6"
PHASE3_PROTOCOL_PHASE = "PHASE3"


def _phase6_target_scope_provenance(filter_name: str, protocol: Any) -> dict[str, Any]:
    protocol_phase = "PHASE2_PLUS_PHASE3_GATE" if filter_name == "zhaocui_fixed" else PHASE3_PROTOCOL_PHASE
    return {
        "benchmark_phase": PHASE6_BENCHMARK_PHASE,
        "protocol_phase": protocol_phase,
        "filter_name": filter_name,
        "source": "protocol.contract.value_score.target_scope",
        "target_scope": protocol.contract.value_score.target_scope,
        "target_scope_relation": "inherited_from_adapter_protocol",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="cpu")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="disabled")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=2)
    parser.add_argument("--observation-dim", type=int, default=1)
    parser.add_argument("--sparse-level", type=int, default=2)
    parser.add_argument("--heldout-start", type=int, default=3)
    parser.add_argument("--fd-step", type=float, default=1.0e-6)
    parser.add_argument("--fd-indices", default="0,4,8,12,13,14,15,16,19,22")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.device_scope == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.horizon <= 1:
        raise ValueError("horizon must be greater than one")
    if args.latent_dim <= 0 or args.hidden_dim <= 0 or args.observation_dim <= 0:
        raise ValueError("dimensions must be positive")
    if args.heldout_start <= 0 or args.heldout_start >= args.horizon:
        raise ValueError("heldout_start must split the horizon into train and heldout")
    if args.sparse_level <= 0:
        raise ValueError("sparse_level must be positive")
    if args.fd_step <= 0.0:
        raise ValueError("fd_step must be positive")


def _apply_tf32_mode(mode: str) -> None:
    if mode == "enabled":
        tf.config.experimental.enable_tensor_float_32_execution(True)
    elif mode == "disabled":
        tf.config.experimental.enable_tensor_float_32_execution(False)


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
    return {"dirty": bool(lines), "line_count": len(lines), "preview": lines[:20]}


def _json_ready(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def _write_markdown(path: Path, artifact: dict[str, Any]) -> None:
    lines = [
        "# SSL-LSTM Phase 6 Shared Benchmark",
        "",
        f"- Schema: `{artifact['schema_version']}`",
        f"- Status: `{artifact['status']}`",
        f"- Git commit: `{artifact['run_manifest']['git_commit']}`",
        f"- Device: `{artifact['run_manifest']['cpu_gpu_status']['device']}`",
        f"- TF32: `{artifact['run_manifest']['tf32_mode']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {item}" for item in artifact["nonclaims"])
    lines.extend(
        [
            "",
            "## Candidate Status",
            "",
            "| filter | status | target scope | gradient path | score finite | heldout predictive log score | decoded latent RMSE |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in artifact["candidate_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["filter_name"]),
                    str(row["status"]),
                    str(row.get("target_scope")),
                    str(row["gradient_path"]),
                    str(row["score_finite"]),
                    str(row["heldout_predictive_log_score"]),
                    str(row["decoded_latent_rmse"]),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _truth_theta(config: SSLLSTMStaticConfig) -> tf.Tensor:
    values = [0.0] * config.parameter_dim
    cursor = 0
    values[cursor : cursor + 4 * config.hidden_dim * config.latent_dim] = [
        0.11,
        -0.07,
        0.05,
        0.02,
        -0.03,
        0.04,
        -0.06,
        0.08,
        0.03,
        -0.02,
        0.04,
        -0.05,
        0.01,
        -0.01,
        0.02,
        -0.03,
    ]
    cursor += 4 * config.hidden_dim * config.latent_dim
    values[cursor : cursor + 4 * config.hidden_dim * config.hidden_dim] = [
        0.02,
        -0.03,
        0.01,
        -0.02,
        0.04,
        -0.01,
        0.03,
        -0.04,
        0.05,
        -0.06,
        0.02,
        -0.01,
        0.03,
        -0.02,
        0.04,
        -0.05,
    ]
    cursor += 4 * config.hidden_dim * config.hidden_dim
    values[cursor : cursor + 4 * config.hidden_dim] = [
        0.01,
        -0.02,
        0.03,
        -0.01,
        0.02,
        -0.03,
        0.04,
        -0.04,
    ]
    cursor += 4 * config.hidden_dim
    values[cursor : cursor + config.latent_dim * config.hidden_dim] = [0.2, -0.15, 0.07, 0.09]
    cursor += config.latent_dim * config.hidden_dim
    values[cursor : cursor + config.latent_dim] = [0.03, -0.02]
    cursor += config.latent_dim
    values[cursor : cursor + config.observation_dim * config.latent_dim] = [0.5, -0.35]
    cursor += config.observation_dim * config.latent_dim
    values[cursor : cursor + config.observation_dim] = [0.05]
    cursor += config.observation_dim
    values[cursor : cursor + config.augmented_state_dim] = [0.0, 0.1, -0.05, 0.02, -0.03, 0.04]
    cursor += config.augmented_state_dim
    values[cursor : cursor + config.augmented_state_dim] = [-2.0, -1.5, -1.2, -1.0, -0.8, -0.6]
    cursor += config.augmented_state_dim
    values[cursor : cursor + config.latent_dim] = [-1.8, -1.4]
    cursor += config.latent_dim
    values[cursor : cursor + config.observation_dim] = [-1.6]
    return tf.constant(values, dtype=tf.float64)


def _data_manifest(args: argparse.Namespace, config: SSLLSTMStaticConfig, theta: tf.Tensor) -> dict[str, Any]:
    return {
        "schema_version": "ssl_lstm.filter_hmc.dataset_manifest.v1",
        "program_signature": "ssl_lstm_phase6_shared_benchmark_v1",
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "observation_dim": int(args.observation_dim),
        "heldout_start": int(args.heldout_start),
        "train_count": int(args.heldout_start),
        "heldout_count": int(args.horizon - args.heldout_start),
        "truth_theta": _json_ready(theta),
        "truth_theta_signature": hashlib.sha256(
            json.dumps(_json_ready(theta), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "noise_scales": {
            "initial_std": [0.2] * config.augmented_state_dim,
            "process_std": [0.1] * config.latent_dim,
            "observation_std": [0.15] * config.observation_dim,
        },
        "split_policy": "prefix_train_suffix_heldout",
        "nonclaims": (
            "synthetic fixture only",
            "not a posterior correctness claim",
            "not a source-faithfulness claim",
        ),
    }


def _simulate_fixture(
    args: argparse.Namespace,
    config: SSLLSTMStaticConfig,
    theta: tf.Tensor,
) -> dict[str, Any]:
    params = unpack_ssl_lstm_parameters(theta, config)
    state = tf.convert_to_tensor(params.initial_mean, dtype=tf.float64)
    states = [state]
    observations = []
    for _ in range(config.horizon):
        state = ssl_lstm_transition(params, state[tf.newaxis, :])[0]
        states.append(state)
        observations.append(ssl_lstm_observation(params, state[tf.newaxis, :])[0])
    states_tensor = tf.stack(states, axis=0)
    observations_tensor = tf.stack(observations, axis=0)
    return {
        "states": states_tensor,
        "observations": observations_tensor,
        "train_observations": observations_tensor[: args.heldout_start],
        "heldout_observations": observations_tensor[args.heldout_start :],
    }


def _alignment_transform(decoded: tf.Tensor, truth: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    decoded = tf.reshape(tf.convert_to_tensor(decoded, dtype=tf.float64), [-1, tf.shape(decoded)[-1]])
    truth = tf.reshape(tf.convert_to_tensor(truth, dtype=tf.float64), [-1, tf.shape(truth)[-1]])
    decoded_center = decoded - tf.reduce_mean(decoded, axis=0, keepdims=True)
    truth_center = truth - tf.reduce_mean(truth, axis=0, keepdims=True)
    cross = tf.transpose(decoded_center) @ truth_center
    _s, u, v = tf.linalg.svd(cross, full_matrices=False)
    rotation = u @ tf.transpose(v)
    return decoded_center @ tf.transpose(rotation), rotation


def _heldout_predictive_log_score(
    train_log_likelihood: tf.Tensor | float,
    full_log_likelihood: tf.Tensor | float,
) -> float:
    train = float(tf.convert_to_tensor(train_log_likelihood, dtype=tf.float64).numpy())
    full = float(tf.convert_to_tensor(full_log_likelihood, dtype=tf.float64).numpy())
    return float(full - train)


def _posterior_predictive_calibration(heldout_obs: tf.Tensor, predicted_obs: tf.Tensor) -> dict[str, Any]:
    residual = tf.convert_to_tensor(heldout_obs, dtype=tf.float64) - tf.convert_to_tensor(predicted_obs, dtype=tf.float64)
    return {
        "mean_abs_residual": float(tf.reduce_mean(tf.abs(residual)).numpy()),
        "max_abs_residual": float(tf.reduce_max(tf.abs(residual)).numpy()),
        "rmse": float(tf.sqrt(tf.reduce_mean(tf.square(residual))).numpy()),
    }


def _fd_subset(
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    observations: tf.Tensor,
    indices: Iterable[int],
    *,
    step: float,
    route: str,
) -> float:
    base = tf.convert_to_tensor(theta, dtype=tf.float64).numpy()
    if route == "fixed_sgqf":
        base_result, _ = tf_ssl_lstm_fixed_sgqf_score(
            observations,
            theta,
            config,
            evidence_path=PLAN_PATH,
            sparse_level=2,
        )
    elif route == "svd_ukf":
        base_result, _ = tf_ssl_lstm_svd_ukf_score(
            observations,
            theta,
            config,
            evidence_path=PLAN_PATH,
        )
    elif route == "zhaocui_fixed":
        base_result, _ = tf_ssl_lstm_zhaocui_fixed_score(
            observations,
            theta,
            config,
            evidence_path=PLAN_PATH,
            manifest=_zhaocui_phase6_manifest(),
        )
    else:
        raise ValueError(f"unknown finite-difference route: {route}")
    analytic_score = tf.convert_to_tensor(base_result.score, dtype=tf.float64).numpy()
    max_error = 0.0
    for index in indices:
        plus = base.copy()
        minus = base.copy()
        plus[index] += step
        minus[index] -= step
        if route == "fixed_sgqf":
            plus_result, _ = tf_ssl_lstm_fixed_sgqf_score(
                observations,
                tf.constant(plus, dtype=tf.float64),
                config,
                evidence_path=PLAN_PATH,
                sparse_level=2,
            )
            minus_result, _ = tf_ssl_lstm_fixed_sgqf_score(
                observations,
                tf.constant(minus, dtype=tf.float64),
                config,
                evidence_path=PLAN_PATH,
                sparse_level=2,
            )
        elif route == "svd_ukf":
            plus_result, _ = tf_ssl_lstm_svd_ukf_score(
                observations,
                tf.constant(plus, dtype=tf.float64),
                config,
                evidence_path=PLAN_PATH,
            )
            minus_result, _ = tf_ssl_lstm_svd_ukf_score(
                observations,
                tf.constant(minus, dtype=tf.float64),
                config,
                evidence_path=PLAN_PATH,
            )
        elif route == "zhaocui_fixed":
            manifest = _zhaocui_phase6_manifest()
            plus_result, _ = tf_ssl_lstm_zhaocui_fixed_score(
                observations,
                tf.constant(plus, dtype=tf.float64),
                config,
                evidence_path=PLAN_PATH,
                manifest=manifest,
            )
            minus_result, _ = tf_ssl_lstm_zhaocui_fixed_score(
                observations,
                tf.constant(minus, dtype=tf.float64),
                config,
                evidence_path=PLAN_PATH,
                manifest=manifest,
            )
        else:
            raise ValueError(f"unknown finite-difference route: {route}")
        fd = float((plus_result.log_likelihood - minus_result.log_likelihood).numpy()) / (2.0 * step)
        max_error = max(max_error, abs(fd - float(analytic_score[index])))
    return float(max_error)


def _candidate_row(
    *,
    filter_name: str,
    status: str,
    protocol: Any,
    score_result: Any,
    train_log_likelihood: tf.Tensor | float,
    full_log_likelihood: tf.Tensor | float,
    decoded_means: tf.Tensor,
    truth_state_path: tf.Tensor,
    heldout_obs: tf.Tensor,
    predicted_obs: tf.Tensor,
    fd_error: float,
    artifact_builder: Any = build_ssl_lstm_debug_value_score_artifact,
    manifest: Any | None = None,
) -> dict[str, Any]:
    aligned, rotation = _alignment_transform(decoded_means, truth_state_path)
    decoded_rmse = float(tf.sqrt(tf.reduce_mean(tf.square(aligned - truth_state_path))).numpy())
    alignment_error = float(tf.reduce_max(tf.abs(aligned - truth_state_path)).numpy())
    calibration = _posterior_predictive_calibration(heldout_obs, predicted_obs)
    target_scope_provenance = _phase6_target_scope_provenance(filter_name, protocol)
    if manifest is None:
        artifact = artifact_builder(
            protocol=protocol,
            log_likelihood=score_result.log_likelihood,
            score=score_result.score,
            finite_difference_max_abs_error=float(fd_error),
            artifact_role="target",
            target_scope=protocol.contract.value_score.target_scope,
            compile_mode="xla",
            jit_compile=True,
            device="/CPU:0",
            tf32_enabled=False,
            nonclaims=(
                "phase6 shared benchmark artifact",
                "not a ranking claim",
                "not a posterior correctness claim",
                "heldout predictive log score is a filter-likelihood proxy",
            ),
        )
    else:
        execution = score_result.diagnostics.get("execution", {})
        actual_jit = execution.get("jit_compile")
        if not isinstance(actual_jit, bool):
            raise ValueError("fixed replay result is missing its execution status")
        if not actual_jit:
            status = "debug_reference"
        artifact = artifact_builder(
            protocol=protocol,
            manifest=manifest,
            log_likelihood=score_result.log_likelihood,
            score=score_result.score,
            finite_difference_max_abs_error=float(fd_error),
            artifact_role="target" if actual_jit else "debug_reference",
            compile_mode="xla" if actual_jit else "graph",
            jit_compile=actual_jit,
            device=score_result.log_likelihood.device,
            tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        )
    artifact["target_scope_provenance"] = target_scope_provenance
    validate_ssl_lstm_value_score_artifact(artifact, protocol=protocol)
    return {
        "filter_name": filter_name,
        "status": status,
        "target_scope": artifact["target_scope"],
        "target_scope_provenance": target_scope_provenance,
        "artifact_role": artifact["artifact_role"],
        "gradient_path": protocol.gradient_path,
        "score_finite": bool(tf.reduce_all(tf.math.is_finite(score_result.score)).numpy()),
        "heldout_predictive_log_score": _heldout_predictive_log_score(
            train_log_likelihood,
            full_log_likelihood,
        ),
        "decoded_latent_rmse": decoded_rmse,
        "trajectory_alignment_error": alignment_error,
        "posterior_predictive_calibration": calibration,
        "score_l2_norm": float(tf.linalg.norm(score_result.score).numpy()),
        "score_norm": float(tf.linalg.norm(score_result.score).numpy()),
        "finite_difference_check": {
            "max_abs_error": float(fd_error),
            "role": "promotion_veto_for_adapter_admission",
        },
        "alignment_rotation_signature": hashlib.sha256(
            json.dumps(_json_ready(rotation), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "artifact": artifact,
        "nonclaims": (
            "status row only",
            "not a success claim",
            "parameter matching is not the primary criterion",
            "heldout predictive log score is a filter-likelihood proxy",
        ),
    }


def _zhaocui_phase6_manifest() -> SSLLSTMZhaoCuiFixedManifest:
    return SSLLSTMZhaoCuiFixedManifest(
        reference_sample_count=9,
        initial_seed=(20260705, 6101),
        process_seed=(20260705, 6102),
    )


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    config = SSLLSTMStaticConfig(
        horizon=args.horizon,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        observation_dim=args.observation_dim,
    )
    theta = _truth_theta(config)
    fixture = _simulate_fixture(args, config, theta)
    observations = fixture["observations"]
    train_obs = fixture["train_observations"]
    heldout_obs = fixture["heldout_observations"]
    truth_states = fixture["states"]
    train_truth_states = truth_states[1 : 1 + int(train_obs.shape[0])]
    heldout_truth_states = truth_states[
        args.heldout_start + 1 : args.heldout_start + 1 + int(heldout_obs.shape[0])
    ]
    indices = tuple(int(value) for value in args.fd_indices.split(",") if value.strip())

    admitted_protocols = {
        "fixed_sgqf": make_ssl_lstm_fixed_sgqf_components(
            theta,
            config,
            evidence_path=PLAN_PATH,
            sparse_level=args.sparse_level,
        ).protocol,
        "svd_ukf": make_ssl_lstm_svd_ukf_components(
            theta,
            config,
            evidence_path=PLAN_PATH,
        ).protocol,
        "zhaocui_fixed": make_ssl_lstm_zhaocui_fixed_components(
            theta,
            config,
            evidence_path=PLAN_PATH,
            manifest=_zhaocui_phase6_manifest(),
        ).protocol,
    }
    fixed_components = make_ssl_lstm_fixed_sgqf_components(
        theta,
        config,
        evidence_path=PLAN_PATH,
        sparse_level=args.sparse_level,
    )
    ukf_components = make_ssl_lstm_svd_ukf_components(
        theta,
        config,
        evidence_path=PLAN_PATH,
    )

    candidate_rows = []
    fixed_result, _fixed_components = tf_ssl_lstm_fixed_sgqf_score(
        train_obs,
        theta,
        config,
        evidence_path=PLAN_PATH,
        sparse_level=args.sparse_level,
    )
    fixed_full_result, _ = tf_ssl_lstm_fixed_sgqf_score(
        observations,
        theta,
        config,
        evidence_path=PLAN_PATH,
        sparse_level=args.sparse_level,
    )
    fixed_value = tf_fixed_sgqf_filter(
        train_obs,
        fixed_components.model,
        cloud=fixed_components.cloud,
        branch_config=fixed_components.branch_config,
        return_filtered=True,
    )
    if fixed_value.failure is not None or fixed_value.filtered_means is None:
        raise RuntimeError("fixed_sgqf value path did not provide filtered means")
    fixed_decoded = tf.convert_to_tensor(fixed_value.filtered_means, dtype=tf.float64)
    fixed_fd_error = _fd_subset(
        theta,
        config,
        train_obs,
        indices,
        step=args.fd_step,
        route="fixed_sgqf",
    )
    fixed_row = _candidate_row(
        filter_name="fixed_sgqf",
        status="admitted",
        protocol=admitted_protocols["fixed_sgqf"],
        score_result=fixed_result,
        train_log_likelihood=fixed_result.log_likelihood,
        full_log_likelihood=fixed_full_result.log_likelihood,
        decoded_means=fixed_decoded,
        truth_state_path=train_truth_states,
        heldout_obs=heldout_obs,
        predicted_obs=ssl_lstm_observation(
            unpack_ssl_lstm_parameters(theta, config),
            heldout_truth_states,
        ),
        fd_error=fixed_fd_error,
    )
    fixed_row["train_log_likelihood"] = float(fixed_result.log_likelihood.numpy())
    fixed_row["full_log_likelihood"] = float(fixed_full_result.log_likelihood.numpy())
    fixed_row["heldout_log_likelihood_proxy"] = fixed_row["heldout_predictive_log_score"]
    candidate_rows.append(fixed_row)

    ukf_result, _ukf_components = tf_ssl_lstm_svd_ukf_score(
        train_obs,
        theta,
        config,
        evidence_path=PLAN_PATH,
    )
    ukf_full_result, _ = tf_ssl_lstm_svd_ukf_score(
        observations,
        theta,
        config,
        evidence_path=PLAN_PATH,
    )
    ukf_value = tf_svd_sigma_point_filter(
        train_obs,
        ukf_components.model,
        backend="tf_svd_ukf",
        innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        return_filtered=True,
    )
    if ukf_value.filtered_means is None:
        raise RuntimeError("svd_ukf value path did not provide filtered means")
    ukf_decoded = tf.convert_to_tensor(ukf_value.filtered_means, dtype=tf.float64)
    ukf_fd_error = _fd_subset(
        theta,
        config,
        train_obs,
        indices,
        step=args.fd_step,
        route="svd_ukf",
    )
    ukf_row = _candidate_row(
        filter_name="svd_ukf",
        status="admitted",
        protocol=admitted_protocols["svd_ukf"],
        score_result=ukf_result,
        train_log_likelihood=ukf_result.log_likelihood,
        full_log_likelihood=ukf_full_result.log_likelihood,
        decoded_means=ukf_decoded,
        truth_state_path=train_truth_states,
        heldout_obs=heldout_obs,
        predicted_obs=ssl_lstm_observation(
            unpack_ssl_lstm_parameters(theta, config),
            heldout_truth_states,
        ),
        fd_error=ukf_fd_error,
    )
    ukf_row["train_log_likelihood"] = float(ukf_result.log_likelihood.numpy())
    ukf_row["full_log_likelihood"] = float(ukf_full_result.log_likelihood.numpy())
    ukf_row["heldout_log_likelihood_proxy"] = ukf_row["heldout_predictive_log_score"]
    candidate_rows.append(ukf_row)

    zhaocui_manifest = _zhaocui_phase6_manifest()
    zhaocui_result, _zhaocui_components = tf_ssl_lstm_zhaocui_fixed_score(
        train_obs,
        theta,
        config,
        evidence_path=PLAN_PATH,
        manifest=zhaocui_manifest,
        jit_compile=bool(args.jit_compile),
    )
    zhaocui_full_result, _ = tf_ssl_lstm_zhaocui_fixed_score(
        observations,
        theta,
        config,
        evidence_path=PLAN_PATH,
        manifest=zhaocui_manifest,
        jit_compile=bool(args.jit_compile),
    )
    zhaocui_decoded = tf.convert_to_tensor(
        zhaocui_result.filtered_means,
        dtype=tf.float64,
    )
    zhaocui_fd_error = _fd_subset(
        theta,
        config,
        train_obs,
        indices,
        step=args.fd_step,
        route="zhaocui_fixed",
    )
    zhaocui_row = _candidate_row(
        filter_name="zhaocui_fixed",
        status="admitted",
        protocol=admitted_protocols["zhaocui_fixed"],
        score_result=zhaocui_result,
        train_log_likelihood=zhaocui_result.log_likelihood,
        full_log_likelihood=zhaocui_full_result.log_likelihood,
        decoded_means=zhaocui_decoded,
        truth_state_path=train_truth_states,
        heldout_obs=heldout_obs,
        predicted_obs=ssl_lstm_observation(
            unpack_ssl_lstm_parameters(theta, config),
            heldout_truth_states,
        ),
        fd_error=zhaocui_fd_error,
        artifact_builder=build_ssl_lstm_zhaocui_fixed_value_score_artifact,
        manifest=zhaocui_manifest,
    )
    zhaocui_row["train_log_likelihood"] = float(zhaocui_result.log_likelihood.numpy())
    zhaocui_row["full_log_likelihood"] = float(zhaocui_full_result.log_likelihood.numpy())
    zhaocui_row["heldout_log_likelihood_proxy"] = zhaocui_row["heldout_predictive_log_score"]
    candidate_rows.append(zhaocui_row)

    candidate_rows.extend(
        [
            {
                "filter_name": "ledh_streaming_ot",
                "status": "blocked",
                "target_scope": None,
                "target_scope_provenance": {
                    "benchmark_phase": PHASE6_BENCHMARK_PHASE,
                    "protocol_phase": PHASE3_PROTOCOL_PHASE,
                    "filter_name": "ledh_streaming_ot",
                    "source": "missing_manual_vjp_streaming_ot_adapter",
                    "target_scope": None,
                    "target_scope_relation": "unavailable",
                },
                "gradient_path": "manual_vjp_streaming_ot",
                "score_finite": None,
                "heldout_predictive_log_score": None,
                "decoded_latent_rmse": None,
                "trajectory_alignment_error": None,
                "posterior_predictive_calibration": None,
                "score_l2_norm": None,
                "score_norm": None,
                "finite_difference_check": {
                    "role": "status_only",
                    "reason": "missing manual VJP streaming-OT score path",
                },
                "nonclaims": ("status-only blocked row", "not a benchmark result"),
            },
        ]
    )

    admitted_rows = [row for row in candidate_rows if row["status"] == "admitted"]
    artifact = {
        "schema_version": "ssl_lstm.filter_hmc.phase6_benchmark.v1",
        "phase": "PHASE6",
        "status": "PHASE6_SHARED_BENCHMARK_READY",
        "plan_file": PLAN_PATH,
        "subplan_file": SUBPLAN_PATH,
        "result_file": RESULT_PATH,
        "dataset_manifest_file": DATASET_MANIFEST_PATH,
        "run_manifest": {
            "git_commit": _git_commit(),
            "dirty_state_summary": _git_dirty_summary(),
            "command": tuple(sys.argv),
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL"),
            },
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cpu_gpu_status": {
                "device_scope": args.device_scope,
                "device": args.device,
                "logical_gpus": [
                    str(device) for device in tf.config.list_logical_devices("GPU")
                ],
            },
            "dtype": "float64",
            "seeds": {"fixture": int(args.seed)},
            "output_json": str(Path(args.output)),
            "markdown_output": str(Path(args.markdown_output)),
            "gpu_trust_basis": (
                GPU_TRUST_BASIS if args.device_scope != "cpu" else "cpu_hidden_debug"
            ),
            "jit_compile": bool(args.jit_compile),
            "tf32_mode": args.tf32_mode,
        },
        "dataset_manifest": _data_manifest(args, config, theta),
        "candidate_rows": candidate_rows,
        "target_scope_provenance": {
            row["filter_name"]: row["target_scope_provenance"] for row in candidate_rows
        },
        "admitted_filters": ADMITTED_FILTERS,
        "blocked_filters": BLOCKED_FILTERS,
        "metric_roles": {
            "heldout_predictive_log_score": "explanatory_proxy",
            "decoded_latent_rmse": "explanatory",
            "trajectory_alignment_error": "explanatory",
            "posterior_predictive_calibration": "explanatory",
            "score_finite_all_admitted": "promotion_veto",
        },
        "invariant_metrics": {
            "heldout_predictive_log_score": {
                row["filter_name"]: row["heldout_predictive_log_score"] for row in admitted_rows
            },
            "decoded_latent_rmse": {
                row["filter_name"]: row["decoded_latent_rmse"] for row in admitted_rows
            },
            "trajectory_alignment_error": {
                row["filter_name"]: row["trajectory_alignment_error"] for row in admitted_rows
            },
            "posterior_predictive_calibration": {
                row["filter_name"]: row["posterior_predictive_calibration"]
                for row in admitted_rows
            },
        },
        "inference_status": {
            "hard_veto_screen": "passed_for_admitted_filters",
            "statistically_supported_ranking": "not_claimed",
            "descriptive_only_differences": "present",
            "default_readiness": "not_checked",
            "next_evidence_needed": "Phase 7 HMC mechanics and evidence ladder",
        },
        "decision_table": {
            "primary_criterion_status": "passed_for_runner_and_schema",
            "veto_diagnostic_status": "no schema veto",
            "main_uncertainty": "HMC mechanics and longer-chain evidence remain future work",
            "next_justified_action": "refresh Phase 7 subplan and execute HMC gates",
            "what_is_not_being_concluded": (
                "No estimation success, no method superiority, no posterior correctness, "
                "no HMC convergence."
            ),
        },
        "nonclaims": NONCLAIMS,
        "parameter_matching_primary_criterion": False,
        "score_finite_all_admitted": all(row["score_finite"] for row in admitted_rows),
    }
    return artifact


def main() -> None:
    args = _parse_args()
    _apply_tf32_mode(args.tf32_mode)
    start = time.perf_counter()
    artifact = build_result(args)
    artifact["runtime_s"] = time.perf_counter() - start
    artifact["timestamp_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    artifact["host"] = platform.node()
    artifact["python_version"] = platform.python_version()
    artifact["tensorflow_version"] = tf.__version__
    artifact["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    artifact["tf32_enabled"] = bool(tf.config.experimental.tensor_float_32_execution_enabled())
    artifact["device_list"] = [
        {"name": device.name, "device_type": device.device_type}
        for device in tf.config.list_physical_devices()
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, artifact)
    markdown = Path(args.markdown_output)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(markdown, artifact)


if __name__ == "__main__":
    main()

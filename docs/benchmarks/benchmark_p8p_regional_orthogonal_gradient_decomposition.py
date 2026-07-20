"""P8p SIR regional kappa/nu orthogonal-coordinate diagnostic.

This Phase 2 diagnostic keeps the scalar SIR forward model on the diagonal but
exposes region-level ``log_kappa_j`` and ``log_nu_j`` score components.  It then
reports the linear coordinates

``rho_j = (log_kappa_j - log_nu_j) / sqrt(2)``
``tau_j = (log_kappa_j + log_nu_j) / sqrt(2)``

as a diagnostic basis only.  This is not a Fisher-orthogonal or production
parameterization claim.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from docs.benchmarks import benchmark_p8p_regional_kappa_gradient_decomposition as regional_kappa
from docs.benchmarks import benchmark_p8p_regression_fd_reparameterization as p8p_reg


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_float_csv(value: str, *, expected: int) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parsed) != expected:
        raise ValueError(f"expected {expected} comma-separated floats")
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError("theta entries must be finite")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-seeds", default="81120,81121,81122,81123,81124")
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--theta", default="0.02,-0.01,0.01")
    parser.add_argument("--phase-label", default="P8p regional orthogonal kappa/nu decomposition")
    parser.add_argument("--fd-step", type=float, default=0.001)
    parser.add_argument("--transport-policy", choices=("active-all", "active-odd", "no-resampling"), default="active-all")
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--transport-plan-mode", choices=("streaming", "dense"), default="streaming")
    parser.add_argument(
        "--transport-gradient-mode",
        choices=(
            "raw",
            p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            p8p.core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        ),
        default=p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    parser.add_argument(
        "--transport-ad-mode",
        choices=("stabilized", "diff-scale", "diff-keys", "diff-potentials", "full"),
        default="stabilized",
    )
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--col-chunk-size", type=int, default=64)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--seed-microbatch-size", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    args.theta_values = _parse_float_csv(args.theta, expected=3)
    if args.time_steps <= 0:
        raise ValueError("time-steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if args.fd_step <= 0.0:
        raise ValueError("fd-step must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn-iterations must be positive")
    for name in ("row_chunk_size", "col_chunk_size", "particle_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.seed_microbatch_size < 0:
        raise ValueError("seed-microbatch-size must be nonnegative")
    return args


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _to_float_matrix(tensor: tf.Tensor) -> list[list[float]]:
    return [[float(value) for value in row] for row in tf.convert_to_tensor(tensor).numpy().tolist()]


def _validate_device(tensors: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in tensors]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU tensors, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU tensors, got {devices}")
    return devices


def _manual_regional_knu_diagnostic_for_contexts(
    contexts: list[dict[str, Any]],
    theta_values: list[float],
) -> dict[str, tf.Tensor]:
    log_likelihoods = []
    per_seed_gradients = []
    kappa_per_seed_values = []
    nu_per_seed_values = []
    kappa_components = []
    nu_components = []
    for context in contexts:
        manual = p8p._manual_value_and_score_from_components(
            context["tensors"],
            context["args"],
            p8p._theta_components(theta_values),
            return_score_decomposition=True,
            return_regional_kappa_decomposition=True,
            return_regional_nu_decomposition=True,
        )
        log_likelihoods.append(tf.convert_to_tensor(manual["log_likelihood"], dtype=p8p.DTYPE))
        per_seed_gradients.append(
            tf.convert_to_tensor(manual["per_seed_gradient"], dtype=p8p.DTYPE)
        )
        kappa_per_seed_values.append(
            tf.convert_to_tensor(manual["regional_kappa_score_per_seed"], dtype=p8p.DTYPE)
        )
        nu_per_seed_values.append(
            tf.convert_to_tensor(manual["regional_nu_score_per_seed"], dtype=p8p.DTYPE)
        )
        kappa_components.append(
            tf.convert_to_tensor(manual["regional_kappa_score_components"], dtype=p8p.DTYPE)
        )
        nu_components.append(
            tf.convert_to_tensor(manual["regional_nu_score_components"], dtype=p8p.DTYPE)
        )
    log_likelihood = tf.concat(log_likelihoods, axis=0)
    per_seed_gradient = tf.concat(per_seed_gradients, axis=0)
    regional_kappa_per_seed = tf.concat(kappa_per_seed_values, axis=0)
    regional_nu_per_seed = tf.concat(nu_per_seed_values, axis=0)
    return {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(per_seed_gradient, axis=0),
        "per_seed_gradient": per_seed_gradient,
        "regional_kappa_score_per_seed": regional_kappa_per_seed,
        "regional_nu_score_per_seed": regional_nu_per_seed,
        "regional_kappa_score_components": tf.concat(kappa_components, axis=1),
        "regional_nu_score_components": tf.concat(nu_components, axis=1),
    }


def _value_for_regional_knu_contexts(
    contexts: list[dict[str, Any]],
    theta_values: list[float],
    regional_log_kappa: tf.Tensor,
    regional_log_nu: tf.Tensor,
) -> tf.Tensor:
    log_obs_noise_scale = p8p._theta_components(theta_values)[2]
    base_kappa = tf.cast(p8p._SIR_BASE_KAPPA, p8p.DTYPE)  # noqa: SLF001
    base_nu = tf.cast(p8p._SIR_BASE_NU, p8p.DTYPE)  # noqa: SLF001
    scaled = p8p._scaled_parameters(
        (
            tf.constant(theta_values[0], dtype=p8p.DTYPE),
            tf.constant(theta_values[1], dtype=p8p.DTYPE),
            log_obs_noise_scale,
        )
    )
    kappa = base_kappa * tf.exp(tf.convert_to_tensor(regional_log_kappa, dtype=p8p.DTYPE))
    nu = base_nu * tf.exp(tf.convert_to_tensor(regional_log_nu, dtype=p8p.DTYPE))
    weighted_objectives = []
    total_seeds = 0
    for context in contexts:
        callbacks = p8p._make_sir_callbacks_from_scaled_parameters(
            tensors=context["tensors"],
            seeds=context["args"].batch_seeds,
            args=context["args"],
            kappa=kappa,
            nu=nu,
            observation_covariance=scaled["observation_covariance"],
        )
        value = p8p.streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
            observations=context["tensors"]["observations"],
            initial_particles=context["tensors"]["initial_particles"],
            fixed_resampling_mask=context["tensors"]["fixed_resampling_mask"],
            transition_matrix=context["tensors"]["transition_matrix"],
            transition_covariance=context["tensors"]["transition_covariance"],
            observation_covariance=callbacks["observation_covariance"],
            observation_fn=callbacks["observation_fn"],
            observation_jacobian_fn=callbacks["observation_jacobian_fn"],
            observation_residual_fn=callbacks["observation_residual_fn"],
            transition_log_density_fn=callbacks["transition_log_density_fn"],
            observation_log_density_fn=callbacks["observation_log_density_fn"],
            prior_mean_fn=callbacks["prior_mean_fn"],
            pre_flow_step_fn=callbacks["pre_flow_step_fn"],
            sinkhorn_epsilon=context["args"].sinkhorn_epsilon,
            annealed_scaling=context["args"].annealed_scaling,
            annealed_convergence_threshold=context["args"].annealed_convergence_threshold,
            sinkhorn_iterations=context["args"].sinkhorn_iterations,
            transport_gradient_mode=context["args"].transport_gradient_mode,
            transport_plan_mode=context["args"].transport_plan_mode,
            transport_ad_mode=context["args"].transport_ad_mode,
            row_chunk_size=context["args"].row_chunk_size,
            col_chunk_size=context["args"].col_chunk_size,
            particle_chunk_size=context["args"].particle_chunk_size,
            return_history=False,
        ).log_likelihood
        seed_count = len(context["seeds"])
        weighted_objectives.append(tf.reduce_mean(value) * tf.cast(seed_count, p8p.DTYPE))
        total_seeds += seed_count
    return tf.add_n(weighted_objectives) / tf.cast(total_seeds, p8p.DTYPE)


def _regional_knu_fd_diagnostics(
    contexts: list[dict[str, Any]],
    theta_values: list[float],
    fd_step: float,
) -> dict[str, list[dict[str, Any]]]:
    region_count = int(tf.shape(tf.cast(p8p._SIR_BASE_KAPPA, p8p.DTYPE))[0])  # noqa: SLF001
    base_kappa = tf.fill([region_count], tf.constant(theta_values[0], dtype=p8p.DTYPE))
    base_nu = tf.fill([region_count], tf.constant(theta_values[1], dtype=p8p.DTYPE))
    step = tf.constant(float(fd_step), dtype=p8p.DTYPE)
    sqrt2 = tf.sqrt(tf.constant(2.0, dtype=p8p.DTYPE))
    diagnostics: dict[str, list[dict[str, Any]]] = {
        "log_kappa_region": [],
        "log_nu_region": [],
        "rho_region": [],
        "tau_region": [],
    }
    for region in range(region_count):
        one = tf.one_hot(region, region_count, dtype=p8p.DTYPE)
        directions = {
            "log_kappa_region": (one, tf.zeros_like(one)),
            "log_nu_region": (tf.zeros_like(one), one),
            "rho_region": (one / sqrt2, -one / sqrt2),
            "tau_region": (one / sqrt2, one / sqrt2),
        }
        for name, (dkappa, dnu) in directions.items():
            plus_objective = _value_for_regional_knu_contexts(
                contexts,
                theta_values,
                base_kappa + step * dkappa,
                base_nu + step * dnu,
            )
            minus_objective = _value_for_regional_knu_contexts(
                contexts,
                theta_values,
                base_kappa - step * dkappa,
                base_nu - step * dnu,
            )
            central = (plus_objective - minus_objective) / (
                tf.constant(2.0, p8p.DTYPE) * step
            )
            diagnostics[name].append(
                {
                    "region": region,
                    "fd_step": float(fd_step),
                    "plus_objective": float(plus_objective.numpy()),
                    "minus_objective": float(minus_objective.numpy()),
                    "central_difference": float(central.numpy()),
                    "finite": bool(tf.reduce_all(tf.math.is_finite(central)).numpy()),
                }
            )
    return diagnostics


def _mcse_by_region(per_seed_regional: tf.Tensor) -> dict[str, Any]:
    tensor = tf.convert_to_tensor(per_seed_regional, dtype=p8p.DTYPE)
    batch_size = int(tensor.shape[0])
    if batch_size <= 1:
        return {"available": False, "reason": "at least two seeds required"}
    mean = tf.reduce_mean(tensor, axis=0)
    centered = tensor - mean[tf.newaxis, :]
    sample_sd = tf.sqrt(
        tf.reduce_sum(tf.square(centered), axis=0)
        / tf.cast(batch_size - 1, p8p.DTYPE)
    )
    mcse = sample_sd / tf.sqrt(tf.cast(batch_size, p8p.DTYPE))
    return {
        "available": True,
        "batch_size": batch_size,
        "mean": [float(item) for item in mean.numpy().tolist()],
        "sample_sd": [float(item) for item in sample_sd.numpy().tolist()],
        "standard_error_of_batch_mean": [float(item) for item in mcse.numpy().tolist()],
    }


def _summary_rows(
    manual: tf.Tensor,
    fd_rows: list[dict[str, Any]],
    mcse: dict[str, Any],
) -> list[dict[str, float | int]]:
    manual_mean = [float(item) for item in tf.reduce_mean(manual, axis=0).numpy().tolist()]
    mcse_values = (
        mcse["standard_error_of_batch_mean"]
        if bool(mcse.get("available", False))
        else [float("nan")] * len(manual_mean)
    )
    return [
        {
            "region": int(fd_row["region"]),
            "manual_mean": float(manual_mean[index]),
            "mcse": float(mcse_values[index]),
            "fd_slope": float(fd_row["central_difference"]),
            "fd_minus_manual": float(fd_row["central_difference"]) - float(manual_mean[index]),
        }
        for index, fd_row in enumerate(fd_rows)
    ]


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    precision = p8p._configure_precision(args)
    physical_gpus, logical_gpus = p8p._configure_gpus()
    contexts, sir_semantics = p8p_reg._build_microbatch_contexts(args)
    start = time.perf_counter()
    memory_before = regional_kappa.p8p._gpu_memory_info()
    with tf.device(args.device):
        manual = _manual_regional_knu_diagnostic_for_contexts(
            contexts,
            args.theta_values,
        )
        objective = tf.convert_to_tensor(manual["objective"], dtype=p8p.DTYPE)
        gradient = tf.convert_to_tensor(manual["gradient_tensor"], dtype=p8p.DTYPE)
        per_seed_gradient = tf.convert_to_tensor(manual["per_seed_gradient"], dtype=p8p.DTYPE)
        regional_kappa_per_seed = tf.convert_to_tensor(
            manual["regional_kappa_score_per_seed"],
            dtype=p8p.DTYPE,
        )
        regional_nu_per_seed = tf.convert_to_tensor(
            manual["regional_nu_score_per_seed"],
            dtype=p8p.DTYPE,
        )
        sqrt2 = tf.sqrt(tf.constant(2.0, dtype=p8p.DTYPE))
        regional_rho_per_seed = (regional_kappa_per_seed - regional_nu_per_seed) / sqrt2
        regional_tau_per_seed = (regional_kappa_per_seed + regional_nu_per_seed) / sqrt2
        regional_fd = _regional_knu_fd_diagnostics(
            contexts,
            args.theta_values,
            args.fd_step,
        )
    elapsed = time.perf_counter() - start
    memory_after = regional_kappa.p8p._gpu_memory_info()
    output_devices = _validate_device((objective, gradient), args.expect_device_kind)

    kappa_mean = tf.reduce_mean(regional_kappa_per_seed, axis=0)
    nu_mean = tf.reduce_mean(regional_nu_per_seed, axis=0)
    rho_mean = tf.reduce_mean(regional_rho_per_seed, axis=0)
    tau_mean = tf.reduce_mean(regional_tau_per_seed, axis=0)
    scalar_kappa = gradient[p8p.PARAMETER_NAMES.index("log_kappa_scale")]
    scalar_nu = gradient[p8p.PARAMETER_NAMES.index("log_nu_scale")]
    rho_scalar = (scalar_kappa - scalar_nu) / sqrt2
    tau_scalar = (scalar_kappa + scalar_nu) / sqrt2

    kappa_mcse = _mcse_by_region(regional_kappa_per_seed)
    nu_mcse = _mcse_by_region(regional_nu_per_seed)
    rho_mcse = _mcse_by_region(regional_rho_per_seed)
    tau_mcse = _mcse_by_region(regional_tau_per_seed)

    chain_rule = {
        "log_kappa": {
            "scalar_manual": float(scalar_kappa.numpy()),
            "sum_regional_manual": float(tf.reduce_sum(kappa_mean).numpy()),
            "abs_sum_minus_scalar": float(tf.abs(tf.reduce_sum(kappa_mean) - scalar_kappa).numpy()),
        },
        "log_nu": {
            "scalar_manual": float(scalar_nu.numpy()),
            "sum_regional_manual": float(tf.reduce_sum(nu_mean).numpy()),
            "abs_sum_minus_scalar": float(tf.abs(tf.reduce_sum(nu_mean) - scalar_nu).numpy()),
        },
        "rho": {
            "scalar_manual": float(rho_scalar.numpy()),
            "sum_regional_manual": float(tf.reduce_sum(rho_mean).numpy()),
            "abs_sum_minus_scalar": float(tf.abs(tf.reduce_sum(rho_mean) - rho_scalar).numpy()),
        },
        "tau": {
            "scalar_manual": float(tau_scalar.numpy()),
            "sum_regional_manual": float(tf.reduce_sum(tau_mean).numpy()),
            "abs_sum_minus_scalar": float(tf.abs(tf.reduce_sum(tau_mean) - tau_scalar).numpy()),
        },
        "tolerance": 1.0e-4,
    }
    chain_rule["pass"] = bool(
        all(
            item["abs_sum_minus_scalar"] <= chain_rule["tolerance"]
            for name, item in chain_rule.items()
            if isinstance(item, dict)
        )
    )

    result: dict[str, Any] = {
        "schema_version": "filter_bench.p8p_regional_orthogonal_gradient_decomposition.v1",
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "git_commit": _git_commit(),
        "phase": args.phase_label,
        "status": "pass" if chain_rule["pass"] else "blocked_or_failed",
        "elapsed_seconds": elapsed,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "output_devices": output_devices,
        "precision": precision,
        "compiler": {
            "explicit_jit_compile": False,
            "xla_status_recorded": False,
            "note": "This diagnostic records GPU/TF32 route only; no explicit XLA compiler status is claimed.",
        },
        "shape": {
            "batch_size": len(args.batch_seeds),
            "seed_microbatch_count": len(contexts),
            "seed_microbatch_size": int(args.seed_microbatch_size),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
            "region_count": int(kappa_mean.shape[0]),
        },
        "sir_semantics": sir_semantics,
        "theta": dict(zip(p8p.PARAMETER_NAMES, [float(x) for x in args.theta_values], strict=True)),
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "objective": float(objective.numpy()),
        "scalar_manual_gradient": {
            name: float(value)
            for name, value in zip(p8p.PARAMETER_NAMES, gradient.numpy().tolist(), strict=True)
        },
        "per_seed_gradient_contributions": _to_float_matrix(per_seed_gradient),
        "regional_log_kappa_manual_mean": [float(item) for item in kappa_mean.numpy().tolist()],
        "regional_log_nu_manual_mean": [float(item) for item in nu_mean.numpy().tolist()],
        "regional_rho_manual_mean": [float(item) for item in rho_mean.numpy().tolist()],
        "regional_tau_manual_mean": [float(item) for item in tau_mean.numpy().tolist()],
        "regional_log_kappa_per_seed": _to_float_matrix(regional_kappa_per_seed),
        "regional_log_nu_per_seed": _to_float_matrix(regional_nu_per_seed),
        "regional_rho_per_seed": _to_float_matrix(regional_rho_per_seed),
        "regional_tau_per_seed": _to_float_matrix(regional_tau_per_seed),
        "regional_log_kappa_mcse": kappa_mcse,
        "regional_log_nu_mcse": nu_mcse,
        "regional_rho_mcse": rho_mcse,
        "regional_tau_mcse": tau_mcse,
        "regional_fd": regional_fd,
        "summary_tables": {
            "log_kappa": _summary_rows(
                regional_kappa_per_seed,
                regional_fd["log_kappa_region"],
                kappa_mcse,
            ),
            "log_nu": _summary_rows(
                regional_nu_per_seed,
                regional_fd["log_nu_region"],
                nu_mcse,
            ),
            "rho": _summary_rows(
                regional_rho_per_seed,
                regional_fd["rho_region"],
                rho_mcse,
            ),
            "tau": _summary_rows(
                regional_tau_per_seed,
                regional_fd["tau_region"],
                tau_mcse,
            ),
        },
        "chain_rule_reconstruction": chain_rule,
        "manual_score_component_names": list(p8p.MANUAL_SCORE_COMPONENT_NAMES),
        "regional_log_kappa_component_sums": _to_float_matrix(
            tf.reduce_sum(manual["regional_kappa_score_components"], axis=1)
        ),
        "regional_log_nu_component_sums": _to_float_matrix(
            tf.reduce_sum(manual["regional_nu_score_components"], axis=1)
        ),
        "transport_policy": args.transport_policy,
        "transport": {
            "value_core_mode": "streaming",
            "transport_plan_mode": args.transport_plan_mode,
            "transport_ad_mode": args.transport_ad_mode,
            "gradient_mode": args.transport_gradient_mode,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
        },
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "nonclaims": list(p8p.NONCLAIMS)
        + [
            "regional kappa/nu diagnostic only",
            "not a Fisher-orthogonal parameterization",
            "not a production regional parameterization",
            "not HMC readiness evidence",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

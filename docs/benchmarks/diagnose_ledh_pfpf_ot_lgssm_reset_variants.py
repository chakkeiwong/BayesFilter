"""Diagnose LGSSM value bias across LEDH-PFPF-OT reset variants.

This diagnostic reuses the small LGSSM harness equations from
``tests/test_ledh_pfpf_ot_lgssm_kalman_statistical.py`` and varies only the
post-correction reset operation:

* no OT: keep the weighted LEDH cloud;
* current OT: current dense finite barycentric reset;
* row-normalized OT: explicitly normalize barycentric rows before applying;
* moment-restored OT: current barycentric reset followed by an affine
  diagnostic correction matching the pre-reset weighted mean/covariance.

It is a debugging diagnostic only.  It does not approve a production reset
policy or certify gradients.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "tests" / "test_ledh_pfpf_ot_lgssm_kalman_statistical.py"


def _parse_setting(value: str) -> dict[str, Any]:
    try:
        epsilon_text, steps_text = value.split(":", maxsplit=1)
        epsilon = float(epsilon_text)
        steps = int(steps_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("settings must be <epsilon>:<steps>") from exc
    if epsilon <= 0.0 or steps <= 0:
        raise argparse.ArgumentTypeError("epsilon and steps must be positive")
    return {"epsilon": epsilon, "steps": steps, "label": f"eps{epsilon:g}_steps{steps}"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--state-dims", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--settings", type=_parse_setting, nargs="+", default=[_parse_setting("0.5:20")])
    parser.add_argument("--jitter", type=float, default=1.0e-5)
    parser.add_argument("--tf32-mode", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()
    if args.num_particles <= 2:
        raise ValueError("--num-particles must exceed 2")
    if args.seed_count != 10:
        raise ValueError("this diagnostic currently reuses harness SEED_COUNT=10")
    if args.time_steps <= 0:
        raise ValueError("--time-steps must be positive")
    if any(dim not in (1, 2) for dim in args.state_dims):
        raise ValueError("--state-dims supports only 1 and 2")
    if args.jitter <= 0.0:
        raise ValueError("--jitter must be positive")
    return args


def _configure_import_environment(args: argparse.Namespace) -> None:
    if args.device_scope == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["BAYESFILTER_RUN_LEDHPFPFOT_LGSSM_N1000"] = "1"
    os.environ["BAYESFILTER_LEDHPFPFOT_LGSSM_NUM_PARTICLES"] = str(args.num_particles)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def _load_harness(args: argparse.Namespace) -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("ledh_lgssm_harness_reset_variants", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TIME_STEPS = int(args.time_steps)
    module.NUM_PARTICLES = int(args.num_particles)
    module.ROW_CHUNK_SIZE = int(args.num_particles)
    module.COL_CHUNK_SIZE = int(args.num_particles)
    module.PARTICLE_CHUNK_SIZE = int(args.num_particles)
    module.core_ledh.DTYPE = module.DTYPE
    module.annealed_transport_tf.DTYPE = module.DTYPE
    module.tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    return module


def _device_manifest(harness: Any, args: argparse.Namespace) -> dict[str, Any]:
    tf = harness.tf
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return {
        "device_scope": args.device_scope,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [str(device) for device in physical_gpus],
        "logical_gpus": [str(device) for device in tf.config.list_logical_devices("GPU")],
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    }


def _kalman_transition_first(harness: Any, state_dim: int, observations: Any) -> dict[str, Any]:
    np = harness.np
    theta = harness.THETA.numpy().astype(np.float64)
    a = float(theta[0])
    q = float(np.exp(theta[1]))
    r = float(np.exp(theta[2]))
    y = observations.astype(np.float64)
    mean = np.zeros(state_dim, dtype=np.float64)
    covariance = 0.7 * np.eye(state_dim, dtype=np.float64)
    transition = a * np.eye(state_dim, dtype=np.float64)
    transition_covariance = q * np.eye(state_dim, dtype=np.float64)
    observation_covariance = r * np.eye(state_dim, dtype=np.float64)
    increments = []
    filtered_means = []
    filtered_covariances = []
    for obs in y:
        predicted_mean = transition @ mean
        predicted_covariance = transition @ covariance @ transition.T + transition_covariance
        innovation = obs - predicted_mean
        innovation_covariance = predicted_covariance + observation_covariance
        sign, logdet = np.linalg.slogdet(innovation_covariance)
        if sign <= 0:
            raise RuntimeError("non-positive innovation covariance")
        increments.append(
            -0.5
            * (
                state_dim * math.log(2.0 * math.pi)
                + logdet
                + innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
        )
        gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
        left = np.eye(state_dim, dtype=np.float64) - gain
        mean = predicted_mean + gain @ innovation
        covariance = left @ predicted_covariance @ left.T + gain @ observation_covariance @ gain.T
        filtered_means.append(mean.copy())
        filtered_covariances.append(covariance.copy())
    increments_np = np.asarray(increments, dtype=np.float64)
    return {
        "increments": increments_np,
        "prefix": np.cumsum(increments_np),
        "total": float(np.sum(increments_np)),
        "filtered_means": np.asarray(filtered_means, dtype=np.float64),
        "filtered_covariances": np.asarray(filtered_covariances, dtype=np.float64),
    }


def _mean_sd_mcse(harness: Any, values: Any) -> dict[str, float]:
    tf = harness.tf
    samples = tf.cast(values, harness.DTYPE)
    mean = tf.reduce_mean(samples)
    centered = samples - mean
    sd = tf.sqrt(tf.reduce_sum(tf.square(centered)) / tf.cast(tf.shape(samples)[0] - 1, harness.DTYPE))
    mcse = sd / tf.sqrt(tf.cast(tf.shape(samples)[0], harness.DTYPE))
    return {"mean": float(mean.numpy()), "sd": float(sd.numpy()), "mcse": float(mcse.numpy())}


def _weighted_moments(harness: Any, points: Any, weights: Any) -> tuple[Any, Any]:
    tf = harness.tf
    mean = tf.reduce_sum(weights[:, :, None] * points, axis=1)
    centered = points - mean[:, None, :]
    covariance = tf.einsum("bn,bni,bnj->bij", weights, centered, centered)
    return mean, covariance


def _uniform_moments(harness: Any, points: Any) -> tuple[Any, Any]:
    tf = harness.tf
    count = tf.cast(tf.shape(points)[1], harness.DTYPE)
    weights = tf.fill([tf.shape(points)[0], tf.shape(points)[1]], 1.0 / count)
    return _weighted_moments(harness, points, weights)


def _affine_restore_moments(
    harness: Any,
    points: Any,
    target_mean: Any,
    target_covariance: Any,
    *,
    jitter: float,
) -> Any:
    tf = harness.tf
    dtype = harness.DTYPE
    source_mean, source_covariance = _uniform_moments(harness, points)
    state_dim = tf.shape(points)[2]
    eye = tf.eye(state_dim, dtype=dtype)[None, :, :]
    source_chol = tf.linalg.cholesky(source_covariance + tf.cast(jitter, dtype) * eye)
    target_chol = tf.linalg.cholesky(target_covariance + tf.cast(jitter, dtype) * eye)
    transform_t = tf.linalg.triangular_solve(
        tf.linalg.matrix_transpose(source_chol),
        tf.linalg.matrix_transpose(target_chol),
        lower=False,
    )
    return tf.matmul(points - source_mean[:, None, :], transform_t) + target_mean[:, None, :]


def _dense_transport_matrix(
    harness: Any,
    post_flow: Any,
    normalized_log_weights: Any,
    setting: dict[str, Any],
) -> Any:
    tf = harness.tf
    annealed = harness.annealed_transport_tf
    center = tf.stop_gradient(tf.reduce_mean(post_flow, axis=1, keepdims=True))
    scale = tf.stop_gradient(annealed._filterflow_scale(post_flow))
    scaled_x = (post_flow - center) / scale[:, None, None]
    epsilon = tf.constant(float(setting["epsilon"]), dtype=harness.DTYPE)
    epsilon0 = tf.stop_gradient(annealed._filterflow_epsilon_start(scaled_x))
    return annealed._filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(
        scaled_x,
        normalized_log_weights,
        epsilon,
        epsilon0,
        tf.constant(harness.ANNEALED_SCALING, dtype=harness.DTYPE),
        steps=int(setting["steps"]),
    )


def _matrix_diagnostics(harness: Any, matrix: Any, normalized_log_weights: Any) -> dict[str, float]:
    tf = harness.tf
    dtype = harness.DTYPE
    num_particles = tf.cast(tf.shape(matrix)[1], dtype)
    source_weights = tf.exp(normalized_log_weights)
    row_mass = tf.reduce_sum(matrix, axis=2)
    column_mass = tf.reduce_sum(matrix, axis=1)
    row_residual = tf.reduce_max(tf.abs(row_mass - 1.0))
    column_residual = tf.reduce_max(tf.abs(column_mass - source_weights * num_particles))
    return {
        "max_row_residual": float(row_residual.numpy()),
        "max_column_residual": float(column_residual.numpy()),
        "min_row_mass": float(tf.reduce_min(row_mass).numpy()),
        "max_row_mass": float(tf.reduce_max(row_mass).numpy()),
    }


def _run_variant(
    harness: Any,
    state_dim: int,
    setting: dict[str, Any],
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tf = harness.tf
    dtype = harness.DTYPE
    batch_size = harness.SEED_COUNT
    num_particles = harness.NUM_PARTICLES
    observations = harness._observations(state_dim)
    initial_noise, transition_noise = harness._stateless_seeded_normals_batch(state_dim)
    theta_batch = tf.tile(harness.THETA[None, :], [batch_size, 1])
    transition_matrix, transition_covariance, observation_covariance = harness._theta_to_lgssm(
        theta_batch,
        state_dim,
    )
    transition_std = tf.sqrt(tf.linalg.diag_part(transition_covariance))
    h_jac = tf.tile(
        tf.eye(state_dim, dtype=dtype)[None, None, :, :],
        [batch_size, num_particles, 1, 1],
    )
    particles = tf.sqrt(tf.constant(0.7, dtype=dtype)) * initial_noise
    log_weights = harness.core_ledh.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=dtype)
    increments = []
    matrix_rows = []
    cov_ratios = []
    mean_shifts = []

    for time_index in range(harness.TIME_STEPS):
        observation = observations[time_index]
        prior_mean = tf.einsum("bnj,bdj->bnd", particles, transition_matrix)
        noise = transition_noise[:, time_index, :, :]
        pre_flow = prior_mean + noise * transition_std[:, None, :]
        residual = observation[None, None, :] - pre_flow
        flow, _flow_aux = harness.core_ledh._batched_ledh_linearized_flow_with_aux_tf(
            pre_flow_particles=pre_flow,
            prior_means=prior_mean,
            observation_jacobian=h_jac,
            observation_residual=residual,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )
        post_flow = flow.post_flow_particles
        transition_log_density = harness._diag_gaussian_logpdf(
            post_flow - prior_mean,
            transition_covariance,
        )
        observation_log_density = harness._diag_gaussian_logpdf(
            post_flow - observation[None, None, :],
            observation_covariance,
        )
        corrected_log_weights = (
            log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = harness.core_ledh._normalize_log_weights(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        increments.append(incremental)
        normalized_log_weights = tf.math.log(
            tf.maximum(weights, harness.core_ledh._log_weight_floor())
        )
        pre_mean, pre_cov = _weighted_moments(harness, post_flow, weights)

        if variant == "ledh_no_ot":
            next_particles = post_flow
            next_log_weights = normalized_log_weights
            post_mean, post_cov = pre_mean, pre_cov
        else:
            matrix = _dense_transport_matrix(harness, post_flow, normalized_log_weights, setting)
            diag = _matrix_diagnostics(harness, matrix, normalized_log_weights)
            matrix_rows.append(diag)
            if variant == "row_normalized_ot":
                row_mass = tf.reduce_sum(matrix, axis=2, keepdims=True)
                matrix = matrix / tf.maximum(row_mass, tf.cast(1.0e-30, dtype))
                diag = _matrix_diagnostics(harness, matrix, normalized_log_weights)
                matrix_rows[-1] = {f"pre_row_norm_{k}": v for k, v in matrix_rows[-1].items()} | {
                    f"post_row_norm_{k}": v for k, v in diag.items()
                }
            next_particles = tf.linalg.matmul(matrix, post_flow)
            if variant == "moment_restored_current_ot":
                next_particles = _affine_restore_moments(
                    harness,
                    next_particles,
                    pre_mean,
                    pre_cov,
                    jitter=args.jitter,
                )
            next_log_weights = harness.core_ledh.uniform_log_weights(batch_size, num_particles)
            post_mean, post_cov = _uniform_moments(harness, next_particles)

        pre_trace = tf.linalg.trace(pre_cov)
        post_trace = tf.linalg.trace(post_cov)
        cov_ratios.append(post_trace / tf.maximum(pre_trace, tf.cast(1.0e-30, dtype)))
        mean_shifts.append(tf.norm(post_mean - pre_mean, axis=1))
        particles, log_weights = next_particles, next_log_weights

    increments_tensor = tf.stack(increments, axis=1)
    kalman = _kalman_transition_first(harness, state_dim, observations.numpy())
    total_stats = _mean_sd_mcse(harness, log_likelihood)
    total_delta = total_stats["mean"] - kalman["total"]
    increment_mean = tf.reduce_mean(increments_tensor, axis=0).numpy()
    increment_delta = increment_mean - kalman["increments"]
    cov_ratio_tensor = tf.stack(cov_ratios, axis=1)
    mean_shift_tensor = tf.stack(mean_shifts, axis=1)
    result = {
        "state_dim": int(state_dim),
        "setting": setting,
        "variant": variant,
        "value": {
            **total_stats,
            "kalman": kalman["total"],
            "delta": float(total_delta),
            "abs_z_mcse": (
                None
                if total_stats["mcse"] <= 0.0
                else float(abs(total_delta) / total_stats["mcse"])
            ),
            "abs_seed_sd_units": (
                None
                if total_stats["sd"] <= 0.0
                else float(abs(total_delta) / total_stats["sd"])
            ),
        },
        "increments": {
            "mean": [float(x) for x in increment_mean],
            "kalman": [float(x) for x in kalman["increments"]],
            "delta": [float(x) for x in increment_delta],
            "max_abs_delta": float(max(abs(float(x)) for x in increment_delta)),
        },
        "moments": {
            "mean_post_pre_cov_trace_ratio_t0": float(tf.reduce_mean(cov_ratio_tensor[:, 0]).numpy()),
            "mean_post_pre_cov_trace_ratio_mean_time": float(tf.reduce_mean(cov_ratio_tensor).numpy()),
            "mean_post_pre_mean_shift_t0": float(tf.reduce_mean(mean_shift_tensor[:, 0]).numpy()),
            "max_post_pre_mean_shift": float(tf.reduce_max(mean_shift_tensor).numpy()),
        },
        "transport": {
            "max_row_residual": None,
            "max_column_residual": None,
            "per_time": matrix_rows,
        },
    }
    if matrix_rows:
        if variant == "row_normalized_ot":
            row_keys = ["post_row_norm_max_row_residual"]
            col_keys = ["post_row_norm_max_column_residual"]
            pre_row_keys = ["pre_row_norm_max_row_residual"]
            pre_col_keys = ["pre_row_norm_max_column_residual"]
        else:
            row_keys = [key for key in matrix_rows[0] if key.endswith("max_row_residual")]
            col_keys = [key for key in matrix_rows[0] if key.endswith("max_column_residual")]
            pre_row_keys = []
            pre_col_keys = []
        result["transport"]["max_row_residual"] = max(
            max(float(row[key]) for key in row_keys) for row in matrix_rows
        )
        result["transport"]["max_column_residual"] = max(
            max(float(row[key]) for key in col_keys) for row in matrix_rows
        )
        if pre_row_keys:
            result["transport"]["pre_row_normalization_max_row_residual"] = max(
                max(float(row[key]) for key in pre_row_keys) for row in matrix_rows
            )
            result["transport"]["pre_row_normalization_max_column_residual"] = max(
                max(float(row[key]) for key in pre_col_keys) for row in matrix_rows
            )
    return result


def _interpret(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["state_dim"], record["setting"]["label"]), {})[
            record["variant"]
        ] = record
    summaries = []
    h3_supported = False
    for (state_dim, label), variants in grouped.items():
        current = variants.get("current_ot")
        moment = variants.get("moment_restored_current_ot")
        no_ot = variants.get("ledh_no_ot")
        row = variants.get("row_normalized_ot")
        if current is None or moment is None or no_ot is None:
            continue
        current_abs = abs(current["value"]["delta"])
        moment_abs = abs(moment["value"]["delta"])
        no_ot_abs = abs(no_ot["value"]["delta"])
        row_abs = abs(row["value"]["delta"]) if row is not None else float("nan")
        improvement = current_abs / moment_abs if moment_abs > 0.0 else float("inf")
        if current_abs > 0.1 and moment_abs < current_abs * 0.25:
            h3_supported = True
        summaries.append(
            {
                "state_dim": state_dim,
                "setting": label,
                "current_abs_delta": current_abs,
                "row_normalized_abs_delta": row_abs,
                "moment_restored_abs_delta": moment_abs,
                "no_ot_abs_delta": no_ot_abs,
                "moment_restoration_improvement_factor": improvement,
                "current_cov_ratio_t0": current["moments"]["mean_post_pre_cov_trace_ratio_t0"],
                "moment_cov_ratio_t0": moment["moments"]["mean_post_pre_cov_trace_ratio_t0"],
            }
        )
    status = (
        "supports_barycentric_covariance_loss_root_cause"
        if h3_supported
        else "inconclusive_or_needs_tighter_fixture"
    )
    return {
        "status": status,
        "summaries": summaries,
        "caution": (
            "Moment restoration is diagnostic only; this does not approve a "
            "production reset route or certify gradients."
        ),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LEDH-PFPF-OT LGSSM Reset-Variant Diagnostic",
        "",
        f"Date: {result['timestamp_utc']}",
        "",
        f"Status: `{result['interpretation']['status']}`",
        "",
        "## Manifest",
        "",
        f"- num_particles: `{result['manifest']['num_particles']}`",
        f"- time_steps: `{result['manifest']['time_steps']}`",
        f"- device_scope: `{result['device']['device_scope']}`",
        f"- cuda_visible_devices: `{result['device']['cuda_visible_devices']}`",
        "",
        "## Variant Table",
        "",
        "| dim | setting | variant | mean | Kalman | delta | abs z MCSE | cov ratio t0 | max row residual |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in result["records"]:
        z = record["value"]["abs_z_mcse"]
        row = record["transport"]["max_row_residual"]
        lines.append(
            "| "
            f"{record['state_dim']} | "
            f"{record['setting']['label']} | "
            f"`{record['variant']}` | "
            f"{record['value']['mean']:.6f} | "
            f"{record['value']['kalman']:.6f} | "
            f"{record['value']['delta']:.6f} | "
            f"{'NA' if z is None else f'{z:.3f}'} | "
            f"{record['moments']['mean_post_pre_cov_trace_ratio_t0']:.6f} | "
            f"{'NA' if row is None else f'{row:.3e}'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "```json",
            json.dumps(result["interpretation"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    _configure_import_environment(args)
    start = time.perf_counter()
    harness = _load_harness(args)
    device = _device_manifest(harness, args)
    records: list[dict[str, Any]] = []
    variants = [
        "ledh_no_ot",
        "current_ot",
        "row_normalized_ot",
        "moment_restored_current_ot",
    ]
    for state_dim in args.state_dims:
        for setting in args.settings:
            for variant in variants:
                records.append(_run_variant(harness, state_dim, setting, variant, args))
    result = {
        "schema_version": "filter_bench.ledh_pfpf_ot_lgssm_reset_variants.v1",
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": harness.tf.__version__,
        "manifest": {
            "num_particles": int(args.num_particles),
            "seed_count": int(args.seed_count),
            "time_steps": int(args.time_steps),
            "state_dims": [int(x) for x in args.state_dims],
            "settings": args.settings,
            "jitter": float(args.jitter),
        },
        "device": device,
        "records": records,
        "interpretation": _interpret(records),
        "elapsed_seconds": time.perf_counter() - start,
        "nonclaims": [
            "not gradient correctness",
            "not SIR correctness",
            "not HMC readiness",
            "not posterior correctness",
            "not production reset approval",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

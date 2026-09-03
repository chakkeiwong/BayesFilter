#!/usr/bin/env python3
"""Bounded performance and transport-score diagnostics for the q=20 route.

This runner is deliberately not a tuning or posterior route.  It compares
batch and serial target evaluation, exercises a diagnostic grouped-HMC
prototype on an analytic target, checks the pullback diagnostic on an exact
affine chart, compares the q=20 analytic score with central differences, and
trains one fresh chart for a few updates.  Outputs are immutable diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BACKEND = "tensorflow_eigh_strict"
DEFAULT_GPU = "0"
SCHEMA = "bayesfilter.ssl_lstm_q20.performance_whitening_diagnostic.v1"
FORBIDDEN_ROUTE_TOKENS = (
    "tf.map_fn",
    "tf.vectorized_map",
    "GradientTape.jacobian",
    "GradientTape.batch_jacobian",
    "pfor",
)
ROUTE_PATHS = (
    ROOT / "bayesfilter/inference/tempered_target_tf.py",
    ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py",
    ROOT / "bayesfilter/inference/tempered_transitions_tf.py",
    ROOT / "bayesfilter/inference/fixed_transport_hmc_mechanics_tf.py",
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DiagnosticError(RuntimeError):
    """Raised when this diagnostic cannot certify its mechanics contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    numpy_like = getattr(value, "numpy", None)
    if callable(numpy_like):
        return _json_ready(numpy_like())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_ready(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_ready(item())
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise DiagnosticError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _nvidia_snapshot() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": type(exc).__name__}


def _static_route_scan() -> Mapping[str, Any]:
    hits: dict[str, list[str]] = {token: [] for token in FORBIDDEN_ROUTE_TOKENS}
    for path in ROUTE_PATHS:
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_ROUTE_TOKENS:
            if token in source:
                hits[token].append(str(path.relative_to(ROOT)))
    return {
        "paths": [str(path.relative_to(ROOT)) for path in ROUTE_PATHS],
        "forbidden_tokens": list(FORBIDDEN_ROUTE_TOKENS),
        "hits": hits,
        "passed": not any(hits.values()),
    }


def _stage_start(output_dir: Path, label: str) -> float:
    started = time.monotonic()
    _write_json(
        output_dir / "stages" / f"{label}-start.json",
        {
            "schema": f"{SCHEMA}.stage.v1",
            "stage": label,
            "status": "STARTED",
            "started_unix": time.time(),
            "started_monotonic": started,
        },
    )
    return started


def _stage_done(
    output_dir: Path, label: str, started: float, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = {
        "schema": f"{SCHEMA}.stage.v1",
        "stage": label,
        "status": "COMPLETED",
        "elapsed_seconds": time.monotonic() - started,
        "completed_unix": time.time(),
        **dict(payload),
    }
    _write_json(output_dir / "stages" / f"{label}-done.json", result)
    return result


def _finite(tf: Any, value: Any) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


class _AnalyticTarget:
    """A four-dimensional Gaussian likelihood fixture for exact mechanics."""

    target_scope = "performance-whitening-analytic-target"

    def target_signature(self) -> str:
        return "performance-whitening-analytic-target-v1"

    def batch_prior_likelihood_value_score_status(self, theta: Any):
        import tensorflow as tf

        values = tf.convert_to_tensor(theta, tf.float64)
        center = tf.constant([1.5, -1.0, 0.75, -0.5], tf.float64)
        delta = values - center
        likelihood = -0.5 * tf.reduce_sum(tf.square(delta), axis=-1)
        likelihood_score = -delta
        prior = -0.5 * tf.reduce_sum(tf.square(values), axis=-1)
        prior_score = -values
        valid = tf.logical_and(
            tf.reduce_all(tf.math.is_finite(values), axis=-1),
            tf.reduce_all(tf.math.is_finite(tf.concat((likelihood_score, prior_score), axis=-1)), axis=-1),
        )
        return likelihood, likelihood_score, prior, prior_score, {
            "status_code": tf.where(valid, 0, 1),
            "valid_pre_regularized_score": valid,
        }


def _analytic_facts() -> Mapping[str, Any]:
    return {
        "horizon": 2,
        "observation_dim": 1,
        "augmented_state_dim": 3,
        "parameter_dim": 4,
        "prior_variance": 1.0,
        "observation_variance": 0.5,
        "sigma_rule": "unscented",
        "sigma_alpha": 1.0,
        "sigma_beta": 2.0,
        "sigma_kappa": 0.0,
        "covariance_weights": [2.0, *(1.0 / 6.0 for _ in range(6))],
        "covariance_weights_nonnegative": True,
        "covariance_weight_sum": 3.0,
        "gaussian_innovation_factorization": True,
        "likelihood_strictly_positive": True,
    }


def _analytic_oracle(tf: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.tempered_target_tf import GaussianLikelihoodBridge
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        AffineDiagonalTransport,
        pullback_gaussianization_diagnostic,
    )

    bridge = GaussianLikelihoodBridge(
        _AnalyticTarget(),
        prior_center=tf.zeros([4], tf.float64),
        prior_variance=1.0,
        source_facts=_analytic_facts(),
        bridge_id="performance_whitening_analytic_bridge_v1",
        jit_compile=True,
    )
    beta = 0.5
    likelihood_center = tf.constant([1.5, -1.0, 0.75, -0.5], tf.float64)
    precision = 1.0 + beta
    exact = AffineDiagonalTransport(
        beta * likelihood_center / precision,
        tf.fill([4], tf.constant(precision ** -0.5, tf.float64)),
        component_id="performance-whitening-exact-affine",
    )
    latent = tf.constant(
        [
            [-1.0, 0.5, 0.25, -0.75],
            [0.0, 0.0, 0.0, 0.0],
            [0.75, -0.25, 1.0, 0.5],
            [1.5, 1.0, -0.5, -1.25],
            [-0.5, 0.25, -1.5, 0.75],
            [0.25, -1.0, 0.5, 1.25],
            [1.0, 0.75, -0.75, -0.25],
            [-1.25, -0.5, 0.25, 1.5],
        ],
        tf.float64,
    )
    diagnostic = pullback_gaussianization_diagnostic(
        exact, bridge, beta=beta, latent=latent
    )
    payload = {
        "finite": bool(diagnostic.finite.numpy()),
        "valid_rows": int(diagnostic.valid_row_count.numpy()),
        "centered_log_density_rms": float(diagnostic.centered_log_density_rms.numpy()),
        "pullback_score_rms_per_coordinate": diagnostic.pullback_score_rms_per_coordinate.numpy().tolist(),
        "pullback_score_maximum_row_norm": float(diagnostic.pullback_score_maximum_row_norm.numpy()),
        "bridge_signature": str(bridge.signature),
        "target_signature": str(bridge.target_signature),
    }
    if not payload["finite"] or payload["valid_rows"] != 8:
        raise DiagnosticError("analytic affine oracle returned invalid rows")
    if payload["centered_log_density_rms"] > 1.0e-11 or payload["pullback_score_maximum_row_norm"] > 1.0e-11:
        raise DiagnosticError("analytic affine Gaussianization oracle failed")
    return payload


def _finite_difference_score(tf: Any, bridge: Any) -> Mapping[str, Any]:
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    dim = int(bridge.parameter_dim)
    offsets = tf.constant(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.25, -0.15, 0.10, -0.20],
            [-0.30, 0.20, -0.05, 0.35],
            [0.45, 0.10, -0.25, -0.15],
        ],
        tf.float64,
    )
    points = center[tf.newaxis, :] + offsets
    steps = (1.0e-3, 1.0e-4, 1.0e-5)
    rows: list[Mapping[str, Any]] = []
    for beta in (0.0, 0.5, 1.0):
        base_value, base_score, base_status = bridge.value_score_status(
            points, tf.constant(beta, tf.float64)
        )
        base_valid = tf.convert_to_tensor(base_status["bridge_valid"], tf.bool)
        if not bool(tf.reduce_all(base_valid).numpy()) or not _finite(tf, base_value) or not _finite(tf, base_score):
            raise DiagnosticError(f"q20 base score probe invalid at beta={beta}")
        for step in steps:
            plus = []
            minus = []
            for index in range(dim):
                direction = tf.one_hot(index, dim, dtype=tf.float64) * tf.constant(step, tf.float64)
                plus.append(points + direction)
                minus.append(points - direction)
            plus_rows = tf.concat(tuple(plus), axis=0)
            minus_rows = tf.concat(tuple(minus), axis=0)
            plus_value, _plus_score, plus_status = bridge.value_score_status(
                plus_rows, tf.constant(beta, tf.float64)
            )
            minus_value, _minus_score, minus_status = bridge.value_score_status(
                minus_rows, tf.constant(beta, tf.float64)
            )
            if not bool(tf.reduce_all(tf.convert_to_tensor(plus_status["bridge_valid"], tf.bool)).numpy()) or not bool(tf.reduce_all(tf.convert_to_tensor(minus_status["bridge_valid"], tf.bool)).numpy()):
                raise DiagnosticError(f"q20 finite-difference status invalid at beta={beta}, h={step}")
            estimate = tf.transpose(
                tf.reshape(plus_value - minus_value, [dim, int(points.shape[0])])
            ) / tf.constant(2.0 * step, tf.float64)
            error = estimate - base_score
            abs_error = tf.abs(error)
            relative = abs_error / tf.maximum(tf.abs(base_score), tf.constant(1.0, tf.float64))
            rows.append(
                {
                    "beta": beta,
                    "step": step,
                    "max_abs_error": float(tf.reduce_max(abs_error).numpy()),
                    "rms_abs_error": float(tf.sqrt(tf.reduce_mean(tf.square(error))).numpy()),
                    "max_relative_error": float(tf.reduce_max(relative).numpy()),
                    "finite": _finite(tf, estimate) and _finite(tf, error),
                }
            )
    return {
        "point_count": int(points.shape[0]),
        "parameter_dim": dim,
        "steps": list(steps),
        "rows": rows,
        "all_finite": all(bool(row["finite"]) for row in rows),
    }


def _target_batch_timing(tf: Any, bridge: Any) -> Mapping[str, Any]:
    rows = tf.random.stateless_normal(
        [32, int(bridge.parameter_dim)], tf.constant([20260902, 91001], tf.int32), dtype=tf.float64
    )
    batches = (8, 16, 32)
    records = []
    for size in batches:
        block = rows[:size]
        bridge.value_score_status(block, tf.constant(0.5, tf.float64))
        started = time.perf_counter()
        repetitions = 2
        for _ in range(repetitions):
            value, score, status = bridge.value_score_status(block, tf.constant(0.5, tf.float64))
            _ = value.numpy(), score.numpy(), status["bridge_valid"].numpy()
        elapsed = time.perf_counter() - started
        trace_count = None
        compiled = getattr(bridge, "_compiled", {}).get(size)
        getter = getattr(compiled, "experimental_get_tracing_count", None)
        if callable(getter):
            trace_count = int(getter())
        records.append(
            {
                "batch_size": size,
                "rows_evaluated": size * repetitions,
                "steady_total_seconds": elapsed,
                "steady_seconds_per_row": elapsed / float(size * repetitions),
                "trace_count": trace_count,
            }
        )
    block = rows[:4]
    bridge.value_score_status(block, tf.constant(0.5, tf.float64))
    started = time.perf_counter()
    serial_repetitions = 8
    for _ in range(serial_repetitions):
        value, score, status = bridge.value_score_status(block, tf.constant(0.5, tf.float64))
        _ = value.numpy(), score.numpy(), status["bridge_valid"].numpy()
    serial_elapsed = time.perf_counter() - started
    records.append(
        {
            "batch_size": 4,
            "rows_evaluated": 4 * serial_repetitions,
            "steady_total_seconds": serial_elapsed,
            "steady_seconds_per_row": serial_elapsed / float(4 * serial_repetitions),
            "mode": "serial_equal_total_rows",
        }
    )
    return {"records": records, "target_signature": str(bridge.target_signature)}


def _analytic_grouped_hmc(tf: Any) -> Mapping[str, Any]:
    import tensorflow_probability as tfp

    def target_log_prob(state: Any) -> Any:
        return -0.5 * tf.reduce_sum(tf.square(state), axis=-1)

    compiled_by_rows: dict[int, Any] = {}

    def run_group(state: Any, step_sizes: Any, seed: tuple[int, int], leapfrog: int) -> Mapping[str, Any]:
        rows = int(state.shape[0])
        compiled = compiled_by_rows.get(rows)
        if compiled is None:
            @tf.function(
                input_signature=(
                    tf.TensorSpec([rows, 4], tf.float64),
                    tf.TensorSpec([rows, 1], tf.float64),
                    tf.TensorSpec([2], tf.int32),
                    tf.TensorSpec([], tf.int32),
                ),
                jit_compile=True,
                reduce_retracing=False,
            )
            def compiled_fn(state_tensor: Any, step_tensor: Any, seed_tensor: Any, leapfrog_tensor: Any):
                kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob,
                    step_size=step_tensor,
                    num_leapfrog_steps=leapfrog_tensor,
                )
                return tfp.mcmc.sample_chain(
                    num_results=4,
                    num_burnin_steps=2,
                    current_state=state_tensor,
                    kernel=kernel,
                    seed=seed_tensor,
                    trace_fn=lambda _state, kernel_results: (
                        kernel_results.is_accepted,
                        kernel_results.log_accept_ratio,
                    ),
                )

            compiled = compiled_fn
            compiled_by_rows[rows] = compiled

        samples, trace = compiled(
            tf.cast(state, tf.float64),
            tf.cast(step_sizes, tf.float64),
            tf.constant(seed, tf.int32),
            tf.constant(leapfrog, tf.int32),
        )
        accepted, log_accept = trace
        getter = getattr(compiled, "experimental_get_tracing_count", None)
        return {
            "rows": int(state.shape[0]),
            "leapfrog": int(leapfrog),
            "sample_shape": tuple(int(value) for value in samples.shape),
            "accepted_shape": tuple(int(value) for value in accepted.shape),
            "acceptance_mean": float(tf.reduce_mean(tf.cast(accepted, tf.float64)).numpy()),
            "all_finite": _finite(tf, samples) and _finite(tf, log_accept),
            "all_rows_moved": bool(tf.reduce_all(tf.reduce_any(tf.not_equal(samples[-1], state), axis=-1)).numpy()),
            "trace_count": None if not callable(getter) else int(getter()),
        }

    # Four candidate step sizes, two independent chains per candidate.
    step_values = (0.10, 0.15, 0.20, 0.25)
    initial = tf.zeros([8, 4], tf.float64)
    grouped = []
    started = time.perf_counter()
    for leapfrog in (3, 8):
        steps = tf.repeat(tf.constant(step_values, tf.float64), repeats=2)
        grouped.append(run_group(initial, steps[:, tf.newaxis], (20260902, 92000 + leapfrog), leapfrog))
    grouped_elapsed = time.perf_counter() - started

    serial = []
    started = time.perf_counter()
    for leapfrog in (3, 8):
        for index, step in enumerate(step_values):
            state = tf.zeros([2, 4], tf.float64)
            result = run_group(
                state,
                tf.fill([2, 1], tf.constant(step, tf.float64)),
                (20260902, 93000 + leapfrog * 10 + index),
                leapfrog,
            )
            serial.append(result)
    serial_elapsed = time.perf_counter() - started
    return {
        "grouped": grouped,
        "serial": serial,
        "grouped_elapsed_seconds": grouped_elapsed,
        "serial_elapsed_seconds": serial_elapsed,
        "equivalence_status": "diagnostic_shape_and_finiteness_only_rng_streams_differ",
        "integration_allowed": False,
    }


def _fresh_q20_transport(tf: Any, bridge: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        IndependentTemperedReverseKLTrainer,
        PreparedTransportInitialization,
        prepare_transport_initialization,
        pullback_gaussianization_diagnostic,
        transport_preflight_state_hash,
    )

    config = WeightedNeuTraConfig(
        dimension=int(bridge.parameter_dim),
        hidden_layers=(16, 16),
        stages=2,
        activation="tanh",
        initialization_scale=0.02,
        initialization_seed=(20260902, 94001),
        learning_rate=1.0e-3,
        jit_compile=True,
    )
    raw = WeightedDenseIAFTransport(config)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    prior_scale = tf.fill(
        [int(bridge.parameter_dim)], tf.sqrt(tf.constant(float(bridge.prior_variance), tf.float64))
    )
    beta0 = prepare_transport_initialization(
        raw,
        bridge,
        component_id="performance-whitening-fresh-chart",
        seed=(20260902, 94002),
        batch_size=32,
        repair_scales=(1.0,),
        beta=0.0,
        reference_center=center,
        reference_scale=prior_scale,
    )
    beta05 = prepare_transport_initialization(
        beta0.transport,
        bridge,
        component_id="performance-whitening-fresh-chart",
        seed=(20260902, 94003),
        batch_size=32,
        repair_scales=(1.0,),
        beta=0.5,
    )
    if not beta0.receipt.valid or not beta05.receipt.valid:
        raise DiagnosticError("fresh q20 chart preflight failed")
    latent = tf.random.stateless_normal(
        [64, int(bridge.parameter_dim)], tf.constant([20260902, 94004], tf.int32), dtype=tf.float64
    )
    initial = pullback_gaussianization_diagnostic(
        beta05.transport, bridge, beta=0.5, latent=latent
    )
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id="performance-whitening-fresh-chart",
        batch_size=32,
        prepared_initialization=PreparedTransportInitialization(beta05.transport, beta05.receipt),
    )
    updates = []
    for index in range(8):
        started = time.perf_counter()
        update = trainer.train_step((20260902, 94100 + index))
        updates.append(
            {
                "update": index + 1,
                "elapsed_seconds": time.perf_counter() - started,
                "loss": float(update.loss.numpy()),
                "gradient_norm": float(update.gradient_norm.numpy()),
                "valid": bool(update.valid.numpy()),
            }
        )
    final = pullback_gaussianization_diagnostic(
        trainer.transport, bridge, beta=0.5, latent=latent
    )
    trace_getter = getattr(trainer._compiled_train_step, "experimental_get_tracing_count", None)
    return {
        "beta": 0.5,
        "batch_size": 32,
        "updates": updates,
        "initial": {
            "centered_log_density_rms": float(initial.centered_log_density_rms.numpy()),
            "pullback_score_rms_per_coordinate": initial.pullback_score_rms_per_coordinate.numpy().tolist(),
            "pullback_score_maximum_row_norm": float(initial.pullback_score_maximum_row_norm.numpy()),
            "finite": bool(initial.finite.numpy()),
        },
        "final": {
            "centered_log_density_rms": float(final.centered_log_density_rms.numpy()),
            "pullback_score_rms_per_coordinate": final.pullback_score_rms_per_coordinate.numpy().tolist(),
            "pullback_score_maximum_row_norm": float(final.pullback_score_maximum_row_norm.numpy()),
            "finite": bool(final.finite.numpy()),
        },
        "initialization_hash": transport_preflight_state_hash(beta05.transport),
        "training_trace_count": None if not callable(trace_getter) else int(trace_getter()),
        "training_updates_all_valid": all(bool(row["valid"]) for row in updates),
        "role": "fresh_target_specific_diagnostic_only",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-analytic-only", action="store_true")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--max-seconds", type=float, default=1100.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_seconds <= 0.0 or not math.isfinite(args.max_seconds):
        raise DiagnosticError("--max-seconds must be finite and positive")
    if args.cpu_analytic_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id or os.environ.get("BAYESFILTER_GPU_ID", DEFAULT_GPU))
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") is None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise DiagnosticError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    manifest_base = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "attempt_id": os.environ.get("BAYESFILTER_PERF_WHITENING_ATTEMPT_ID", "unspecified"),
        "command": list(sys.argv),
        "python": sys.executable,
        "platform": platform.platform(),
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status": _git(("git", "status", "--porcelain")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""),
        "target_signature": TARGET_SIGNATURE,
        "principal_sqrt_backend": BACKEND,
        "cpu_analytic_only": bool(args.cpu_analytic_only),
        "max_seconds": float(args.max_seconds),
        "gpu_snapshot_before": _nvidia_snapshot(),
    }
    _write_json(output_dir / "run_start.json", manifest_base)
    stages: dict[str, Mapping[str, Any]] = {}
    try:
        import tensorflow as tf

        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not args.cpu_analytic_only)
        if args.cpu_analytic_only:
            logical_devices = tuple(tf.config.list_logical_devices("CPU"))
        else:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            logical_devices = tuple(tf.config.list_logical_devices("GPU"))
            if len(logical_devices) != 1:
                raise DiagnosticError("GPU diagnostic requires exactly one visible logical GPU")
        route_scan = _static_route_scan()
        if not route_scan["passed"]:
            raise DiagnosticError(f"forbidden route token found: {route_scan}")

        stage_started = _stage_start(output_dir, "analytic-oracle")
        stages["analytic_oracle"] = _stage_done(output_dir, "analytic-oracle", stage_started, _analytic_oracle(tf))

        stage_started = _stage_start(output_dir, "analytic-grouped-hmc")
        stages["analytic_grouped_hmc"] = _stage_done(output_dir, "analytic-grouped-hmc", stage_started, _analytic_grouped_hmc(tf))

        if args.cpu_analytic_only:
            payload = {
                **manifest_base,
                "status": "PASS_CPU_ANALYTIC_ONLY",
                "tensorflow": str(tf.__version__),
                "memory_policy": memory_policy,
                "logical_devices": [str(device.name) for device in logical_devices],
                "route_scan": route_scan,
                "stages": stages,
                "wall_time_seconds": time.monotonic() - started,
                "gpu_snapshot_after": _nvidia_snapshot(),
                "nonclaims": [
                    "analytic and execution diagnostics only",
                    "no q20 transport, whitening, HMC, posterior, or scaling claim",
                ],
            }
            payload["source_hashes"] = {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (*ROUTE_PATHS, Path(__file__), ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-repair-plan-2026-09-02.md")
                if path.is_file()
            }
            payload["manifest_hash"] = _stable_hash(payload)
            _write_json(output_dir / "run_manifest.json", payload)
            print(json.dumps({"status": payload["status"], "output_dir": str(output_dir)}, sort_keys=True))
            return 0

        if time.monotonic() - started > args.max_seconds:
            raise DiagnosticError("diagnostic budget exhausted before q20 stages")
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

        stage_started = _stage_start(output_dir, "q20-bridge")
        bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend=BACKEND)
        if str(bridge.target_signature) != TARGET_SIGNATURE:
            raise DiagnosticError("q20 target signature mismatch")
        stages["q20_bridge"] = _stage_done(
            output_dir,
            "q20-bridge",
            stage_started,
            {
                "target_signature": str(bridge.target_signature),
                "bridge_signature": str(bridge.signature),
                "properness_receipt": bridge.properness_receipt.payload(),
            },
        )

        stage_started = _stage_start(output_dir, "q20-target-batch-timing")
        stages["q20_target_batch_timing"] = _stage_done(
            output_dir, "q20-target-batch-timing", stage_started, _target_batch_timing(tf, bridge)
        )

        if time.monotonic() - started > args.max_seconds:
            raise DiagnosticError("diagnostic budget exhausted before score stage")
        stage_started = _stage_start(output_dir, "q20-score-finite-difference")
        stages["q20_score_finite_difference"] = _stage_done(
            output_dir, "q20-score-finite-difference", stage_started, _finite_difference_score(tf, bridge)
        )

        if time.monotonic() - started > args.max_seconds:
            raise DiagnosticError("diagnostic budget exhausted before fresh chart stage")
        stage_started = _stage_start(output_dir, "q20-fresh-chart")
        stages["q20_fresh_chart"] = _stage_done(
            output_dir, "q20-fresh-chart", stage_started, _fresh_q20_transport(tf, bridge)
        )
        payload = {
            **manifest_base,
            "status": "PASS_Q20_PERFORMANCE_WHITENING_DIAGNOSTIC",
            "tensorflow": str(tf.__version__),
            "memory_policy": memory_policy,
            "logical_devices": [str(device.name) for device in logical_devices],
            "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "route_scan": route_scan,
            "bridge_signature": str(bridge.signature),
            "properness_receipt": bridge.properness_receipt.payload(),
            "stages": stages,
            "wall_time_seconds": time.monotonic() - started,
            "gpu_snapshot_after": _nvidia_snapshot(),
            "source_hashes": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (*ROUTE_PATHS, Path(__file__), ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-repair-plan-2026-09-02.md")
                if path.is_file()
            },
            "nonclaims": [
                "performance and score diagnostics only",
                "grouped HMC prototype not integrated because RNG streams differ",
                "no IID-Gaussian whitening, mode-discovery, posterior, convergence, or scaling claim",
            ],
        }
        payload["manifest_hash"] = _stable_hash(payload)
        _write_json(output_dir / "run_manifest.json", payload)
        print(json.dumps({"status": payload["status"], "output_dir": str(output_dir), "wall_time_seconds": payload["wall_time_seconds"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve a durable failure receipt.
        failure = {
            **manifest_base,
            "status": "FAIL_PERFORMANCE_WHITENING_DIAGNOSTIC",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stages_completed": stages,
            "wall_time_seconds": time.monotonic() - started,
            "gpu_snapshot_after": _nvidia_snapshot(),
        }
        _write_json(output_dir / "failure.json", failure)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"DIAGNOSTIC_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

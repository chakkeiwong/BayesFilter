#!/usr/bin/env python3
"""Execute the bounded M3-C grouped-transition and training diagnostics.

This is a diagnostic campaign, not a Phase 9A replay or a posterior route.
N1 compares batched TFP HMC with scalar and explicit row-loop semantics. N2
runs a small target-specific reverse-KL ladder on disjoint held-out banks.
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
PLAN_PATH = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md"
MASTER_PATH = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BACKEND = "tensorflow_eigh_strict"
DEFAULT_GPU = "0"
SCHEMA = "bayesfilter.ssl_lstm_q20.performance_whitening_next.v1"
MAX_DIMENSION = 4
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
    """Raised when the campaign contract cannot be evaluated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise DiagnosticError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _gpu_snapshot() -> Mapping[str, Any]:
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


def _route_scan() -> Mapping[str, Any]:
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


def _finite(tf: Any, value: Any) -> bool:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype == tf.bool:
        return True
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _max_abs(tf: Any, left: Any, right: Any) -> float:
    return float(tf.reduce_max(tf.abs(tf.cast(left, tf.float64) - tf.cast(right, tf.float64))).numpy())


def _memory_info(tf: Any, device: str = "GPU:0") -> Mapping[str, Any]:
    try:
        return {str(key): int(value) for key, value in tf.config.experimental.get_memory_info(device).items()}
    except Exception as exc:  # noqa: BLE001 - telemetry must not break a diagnostic.
        return {"unavailable": type(exc).__name__}


def _fold_seed(tf: Any, seed: Any, index: int) -> Any:
    return tf.random.experimental.stateless_fold_in(tf.convert_to_tensor(seed, tf.int32), tf.constant(index, tf.int32))


def _build_hmc_kernel(tf: Any, tfp: Any, *, step: float, leapfrog: int) -> Any:
    def target_log_prob(state: Any) -> Any:
        state_tensor = tf.convert_to_tensor(state, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(state_tensor), axis=-1)

    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(step, tf.float64),
        num_leapfrog_steps=tf.constant(leapfrog, tf.int32),
    )


def _n1_grouped_transition(tf: Any, tfp: Any) -> Mapping[str, Any]:
    rows = 4
    step = 0.20
    leapfrog = 3
    kernel = _build_hmc_kernel(tf, tfp, step=step, leapfrog=leapfrog)

    @tf.function(
        input_signature=(tf.TensorSpec([1, MAX_DIMENSION], tf.float64), tf.TensorSpec([2], tf.int32)),
        jit_compile=True,
        reduce_retracing=False,
    )
    def scalar_transition(state: Any, seed: Any):
        results = kernel.bootstrap_results(state)
        next_state, next_results = kernel.one_step(state, results, seed=seed)
        return (
            tf.ensure_shape(next_state, [1, MAX_DIMENSION]),
            tf.ensure_shape(next_results.is_accepted, [1]),
            tf.ensure_shape(next_results.log_accept_ratio, [1]),
            tf.ensure_shape(next_results.accepted_results.target_log_prob, [1]),
            tf.ensure_shape(next_results.accepted_results.grads_target_log_prob[0], [1, MAX_DIMENSION]),
        )

    @tf.function(
        input_signature=(tf.TensorSpec([rows, MAX_DIMENSION], tf.float64), tf.TensorSpec([2], tf.int32)),
        jit_compile=True,
        reduce_retracing=False,
    )
    def grouped_transition(state: Any, seed: Any):
        results = kernel.bootstrap_results(state)
        next_state, next_results = kernel.one_step(state, results, seed=seed)
        return (
            tf.ensure_shape(next_state, [rows, MAX_DIMENSION]),
            tf.ensure_shape(next_results.is_accepted, [rows]),
            tf.ensure_shape(next_results.log_accept_ratio, [rows]),
            tf.ensure_shape(next_results.accepted_results.target_log_prob, [rows]),
            tf.ensure_shape(next_results.accepted_results.grads_target_log_prob[0], [rows, MAX_DIMENSION]),
        )

    @tf.function(
        input_signature=(tf.TensorSpec([rows, MAX_DIMENSION], tf.float64), tf.TensorSpec([rows, 2], tf.int32)),
        # Nested TFP scalar calls are a diagnostic control.  Keep this
        # wrapper outside XLA while isolating their seed/shape semantics.
        jit_compile=False,
        reduce_retracing=False,
    )
    def row_loop_transition(state: Any, seeds: Any):
        state_ta = tf.TensorArray(tf.float64, size=rows, element_shape=[MAX_DIMENSION])
        accept_ta = tf.TensorArray(tf.bool, size=rows, element_shape=[])
        log_ta = tf.TensorArray(tf.float64, size=rows, element_shape=[])
        target_ta = tf.TensorArray(tf.float64, size=rows, element_shape=[])
        grad_ta = tf.TensorArray(tf.float64, size=rows, element_shape=[MAX_DIMENSION])

        def condition(index: Any, *_unused: Any) -> Any:
            return index < rows

        def body(index: Any, state_acc: Any, accept_acc: Any, log_acc: Any, target_acc: Any, grad_acc: Any):
            one_state = state[index : index + 1]
            # Reuse the scalar graph itself.  Constructing the TFP kernel
            # directly in this loop changes graph-level seed salts and is not
            # a valid semantic control for a per-candidate comparison.
            next_state, accepted, log_accept, target_value, gradient = scalar_transition(
                one_state, seeds[index]
            )
            return (
                index + 1,
                state_acc.write(index, next_state[0]),
                accept_acc.write(index, accepted[0]),
                log_acc.write(index, log_accept[0]),
                target_acc.write(index, target_value[0]),
                grad_acc.write(index, gradient[0]),
            )

        _, state_ta, accept_ta, log_ta, target_ta, grad_ta = tf.while_loop(
            condition,
            body,
            (tf.constant(0, tf.int32), state_ta, accept_ta, log_ta, target_ta, grad_ta),
            parallel_iterations=1,
        )
        return state_ta.stack(), accept_ta.stack(), log_ta.stack(), target_ta.stack(), grad_ta.stack()

    initial = tf.zeros([rows, MAX_DIMENSION], tf.float64)
    base_seed = tf.constant([20260902, 96101], tf.int32)
    row_seeds = tf.stack([_fold_seed(tf, base_seed, index) for index in range(rows)], axis=0)
    scalar = [scalar_transition(initial[index : index + 1], row_seeds[index]) for index in range(rows)]
    scalar_state = tf.concat([item[0] for item in scalar], axis=0)
    scalar_accept = tf.concat([item[1] for item in scalar], axis=0)
    scalar_log = tf.concat([item[2] for item in scalar], axis=0)
    scalar_target = tf.concat([item[3] for item in scalar], axis=0)
    scalar_grad = tf.concat([item[4] for item in scalar], axis=0)
    grouped = grouped_transition(initial, base_seed)
    row_loop = row_loop_transition(initial, row_seeds)

    scalar_getter = getattr(scalar_transition, "experimental_get_tracing_count", None)
    grouped_getter = getattr(grouped_transition, "experimental_get_tracing_count", None)
    row_getter = getattr(row_loop_transition, "experimental_get_tracing_count", None)
    row_errors = {
        "state_max_abs": _max_abs(tf, row_loop[0], scalar_state),
        "log_accept_max_abs": _max_abs(tf, row_loop[2], scalar_log),
        "target_max_abs": _max_abs(tf, row_loop[3], scalar_target),
        "gradient_max_abs": _max_abs(tf, row_loop[4], scalar_grad),
        "accept_equal": bool(tf.reduce_all(tf.equal(row_loop[1], scalar_accept)).numpy()),
    }
    grouped_errors = {
        "state_max_abs": _max_abs(tf, grouped[0], scalar_state),
        "log_accept_max_abs": _max_abs(tf, grouped[2], scalar_log),
        "target_max_abs": _max_abs(tf, grouped[3], scalar_target),
        "gradient_max_abs": _max_abs(tf, grouped[4], scalar_grad),
        "accept_equal": bool(tf.reduce_all(tf.equal(grouped[1], scalar_accept)).numpy()),
    }
    tolerance = 1.0e-12
    row_equivalent = bool(
        row_errors["accept_equal"]
        and all(float(row_errors[key]) <= tolerance for key in ("state_max_abs", "log_accept_max_abs", "target_max_abs", "gradient_max_abs"))
    )
    grouped_equivalent = bool(
        grouped_errors["accept_equal"]
        and all(float(grouped_errors[key]) <= tolerance for key in ("state_max_abs", "log_accept_max_abs", "target_max_abs", "gradient_max_abs"))
    )
    return {
        "rows": rows,
        "step_size": step,
        "leapfrog_steps": leapfrog,
        "seed_policy": "scalar_and_row_loop_use_fold_in(base_seed,candidate); fast_grouped_receives_base_seed",
        "expected_target_call_counts": {"scalar_total": rows, "fast_grouped": 1, "row_loop": rows},
        "scalar": {
            "trace_count": None if not callable(scalar_getter) else int(scalar_getter()),
            "all_finite": all(_finite(tf, value) for value in (scalar_state, scalar_log, scalar_target, scalar_grad)),
        },
        "fast_grouped": {
            "trace_count": None if not callable(grouped_getter) else int(grouped_getter()),
            "all_finite": all(_finite(tf, value) for value in grouped),
            "errors_against_scalar": grouped_errors,
            "equivalent": grouped_equivalent,
        },
        "row_loop_control": {
            "trace_count": None if not callable(row_getter) else int(row_getter()),
            "all_finite": all(_finite(tf, value) for value in row_loop),
            "errors_against_scalar": row_errors,
            "equivalent": row_equivalent,
        },
        "integration_allowed": grouped_equivalent,
        "status": "FAST_GROUPED_EQUIVALENT" if grouped_equivalent else "FAST_GROUPED_REJECTED_ROW_LOOP_CONTROL_CHECKED",
    }


def _diagnostic_payload(tf: Any, diagnostic: Any) -> Mapping[str, Any]:
    return {
        "centered_log_density_rms": float(diagnostic.centered_log_density_rms.numpy()),
        "pullback_score_rms_per_coordinate": diagnostic.pullback_score_rms_per_coordinate.numpy().tolist(),
        "pullback_score_maximum_row_norm": float(diagnostic.pullback_score_maximum_row_norm.numpy()),
        "finite": bool(diagnostic.finite.numpy()),
    }


def _run_n2_candidate(tf: Any, bridge: Any, arm: Mapping[str, Any], seed_index: int, seed_value: int, max_seconds: float, campaign_started: float) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        PreparedTransportInitialization,
        IndependentTemperedReverseKLTrainer,
        prepare_transport_initialization,
        pullback_gaussianization_diagnostic,
        transport_preflight_state_hash,
    )

    if time.monotonic() - campaign_started > max_seconds:
        raise DiagnosticError("N2 budget exhausted before candidate")
    arm_id = str(arm["id"])
    seed = (20260902, int(seed_value))
    config = WeightedNeuTraConfig(
        dimension=int(bridge.parameter_dim),
        hidden_layers=tuple(int(value) for value in arm["hidden_layers"]),
        stages=int(arm["stages"]),
        activation="tanh",
        initialization_scale=0.02,
        initialization_seed=seed,
        learning_rate=float(arm["learning_rate"]),
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    raw = WeightedDenseIAFTransport(config)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    prior_scale = tf.fill([int(bridge.parameter_dim)], tf.sqrt(tf.constant(float(bridge.prior_variance), tf.float64)))
    component_id = f"m3c-{arm_id}-seed-{seed_index}"
    beta0 = prepare_transport_initialization(
        raw,
        bridge,
        component_id=component_id,
        seed=(20260902, 97000 + seed_index * 100 + int(arm["ordinal"])),
        batch_size=32,
        repair_scales=(1.0,),
        beta=0.0,
        reference_center=center,
        reference_scale=prior_scale,
    )
    beta05 = prepare_transport_initialization(
        beta0.transport,
        bridge,
        component_id=component_id,
        seed=(20260902, 97100 + seed_index * 100 + int(arm["ordinal"])),
        batch_size=32,
        repair_scales=(1.0,),
        beta=0.5,
    )
    if not beta0.receipt.valid or not beta05.receipt.valid:
        raise DiagnosticError(f"preflight failed for {component_id}")
    # The held-out bank is shared by all arms for a given seed.  This keeps
    # arm comparisons paired while remaining disjoint from each arm's
    # calibration and training streams.
    validation_seed = (20260902, 98000 + seed_index * 100)
    validation_latent = tf.random.stateless_normal(
        [64, int(bridge.parameter_dim)], tf.constant(validation_seed, tf.int32), dtype=tf.float64
    )
    initial = pullback_gaussianization_diagnostic(beta05.transport, bridge, beta=0.5, latent=validation_latent)
    if not bool(initial.finite.numpy()):
        raise DiagnosticError(f"initial validation diagnostic invalid for {component_id}")
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id=component_id,
        batch_size=32,
        prepared_initialization=PreparedTransportInitialization(beta05.transport, beta05.receipt),
    )
    updates = []
    memory_before = _memory_info(tf)
    for update_index in range(12):
        if time.monotonic() - campaign_started > max_seconds:
            raise DiagnosticError(f"N2 budget exhausted during {component_id}")
        started = time.perf_counter()
        result = trainer.train_step((20260902, 99000 + int(arm["ordinal"]) * 1000 + seed_index * 100 + update_index))
        updates.append(
            {
                "update": update_index + 1,
                "elapsed_seconds": time.perf_counter() - started,
                "loss": float(result.loss.numpy()),
                "gradient_norm": float(result.gradient_norm.numpy()),
                "valid": bool(result.valid.numpy()),
            }
        )
    final = pullback_gaussianization_diagnostic(trainer.transport, bridge, beta=0.5, latent=validation_latent)
    trace_getter = getattr(trainer._compiled_train_step, "experimental_get_tracing_count", None)
    initial_score = tf.linalg.norm(initial.pullback_score_rms_per_coordinate)
    final_score = tf.linalg.norm(final.pullback_score_rms_per_coordinate)
    initial_log = initial.centered_log_density_rms
    final_log = final.centered_log_density_rms
    return {
        "arm": dict(arm),
        "seed_index": seed_index,
        "seed": list(seed),
        "calibration_seeds": [[20260902, 97000 + seed_index * 100 + int(arm["ordinal"])], [20260902, 97100 + seed_index * 100 + int(arm["ordinal"])]],
        "validation_seed": list(validation_seed),
        "batch_size": 32,
        "beta": 0.5,
        "updates_requested": 12,
        "updates": updates,
        "all_updates_valid": all(bool(row["valid"]) for row in updates),
        "initial": _diagnostic_payload(tf, initial),
        "final": _diagnostic_payload(tf, final),
        "aggregate_score_rms_initial": float(initial_score.numpy()),
        "aggregate_score_rms_final": float(final_score.numpy()),
        "aggregate_score_relative_change": float((final_score / tf.maximum(initial_score, tf.constant(1.0e-30, tf.float64))).numpy() - 1.0),
        "centered_log_relative_change": float((final_log / tf.maximum(initial_log, tf.constant(1.0e-30, tf.float64))).numpy() - 1.0),
        "training_trace_count": None if not callable(trace_getter) else int(trace_getter()),
        "initialization_hash": transport_preflight_state_hash(beta05.transport),
        "memory_before": memory_before,
        "memory_after": _memory_info(tf),
        "finite": bool(final.finite.numpy()) and all(bool(row["valid"]) for row in updates),
        "role": "target_specific_diagnostic_only",
    }


def _n2_training_ladder(tf: Any, bridge: Any, *, max_seconds: float, campaign_started: float) -> Mapping[str, Any]:
    arms = (
        {"ordinal": 0, "id": "A_baseline_16x16_s2_lr1e-3", "hidden_layers": (16, 16), "stages": 2, "learning_rate": 1.0e-3},
        {"ordinal": 1, "id": "B_baseline_16x16_s2_lr3e-4", "hidden_layers": (16, 16), "stages": 2, "learning_rate": 3.0e-4},
        {"ordinal": 2, "id": "C_capacity_32x32_s3_lr3e-4", "hidden_layers": (32, 32), "stages": 3, "learning_rate": 3.0e-4},
    )
    seed_values = (97201, 97202, 97203)
    candidates = []
    failures = []
    for arm in arms:
        for seed_index, seed_value in enumerate(seed_values):
            try:
                candidates.append(_run_n2_candidate(tf, bridge, arm, seed_index, seed_value, max_seconds, campaign_started))
            except Exception as exc:  # noqa: BLE001 - preserve candidate-level failure and continue.
                failures.append({"arm": dict(arm), "seed_index": seed_index, "seed": [20260902, seed_value], "error_type": type(exc).__name__, "error": str(exc)})
    nominated = {}
    for arm in arms:
        rows = [row for row in candidates if row["arm"]["id"] == arm["id"]]
        improved = [row for row in rows if row["finite"] and row["aggregate_score_relative_change"] <= -0.10]
        nominated[arm["id"]] = {
            "finite_seed_count": sum(bool(row["finite"]) for row in rows),
            "improved_seed_count": len(improved),
            "nomination_pass": len(rows) == 3 and len(improved) >= 2,
        }
    return {
        "arms": [dict(arm) for arm in arms],
        "seed_values": [list((20260902, value)) for value in seed_values],
        "candidates": candidates,
        "failures": failures,
        "nomination_rule": "all 3 seeds finite and at least 2 of 3 reduce aggregate held-out score RMS by >=10 percent",
        "nominations": nominated,
        "default_change_allowed": False,
        "status": "COMPLETED" if not failures else "COMPLETED_WITH_CANDIDATE_FAILURES",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-n1-only", action="store_true")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_seconds <= 0.0 or not math.isfinite(args.max_seconds):
        raise DiagnosticError("--max-seconds must be finite and positive")
    if args.cpu_n1_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id or os.environ.get("BAYESFILTER_GPU_ID", DEFAULT_GPU))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise DiagnosticError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    campaign_started = time.monotonic()
    manifest_base: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "authority": str(MASTER_PATH.relative_to(ROOT)),
        "plan_file": str(PLAN_PATH.relative_to(ROOT)),
        "attempt_id": os.environ.get("BAYESFILTER_NEXT_ATTEMPT_ID", "unspecified"),
        "command": list(sys.argv),
        "python": sys.executable,
        "platform": platform.platform(),
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status": _git(("git", "status", "--porcelain")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""),
        "target_signature": TARGET_SIGNATURE,
        "principal_sqrt_backend": BACKEND,
        "cpu_n1_only": bool(args.cpu_n1_only),
        "max_seconds": float(args.max_seconds),
        "gpu_snapshot_before": _gpu_snapshot(),
        "budget": {"gpu_seconds": 900, "cpu_seconds": 300},
    }
    _write_json(output_dir / "run_start.json", manifest_base)
    stages: dict[str, Any] = {}
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp

        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not args.cpu_n1_only)
        if args.cpu_n1_only:
            logical_devices = tuple(tf.config.list_logical_devices("CPU"))
        else:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            logical_devices = tuple(tf.config.list_logical_devices("GPU"))
            if len(logical_devices) != 1:
                raise DiagnosticError("N1/N2 GPU run requires exactly one visible logical GPU")
        route_scan = _route_scan()
        if not route_scan["passed"]:
            raise DiagnosticError(f"forbidden route token found: {route_scan}")
        stages["n0_source_audit"] = {
            "status": "COMPLETED",
            "route_scan": route_scan,
            "focused_scope": "M3-C source, seed, target, and memory-policy audit",
        }
        stages["n1_grouped_transition"] = _n1_grouped_transition(tf, tfp)
        if args.cpu_n1_only:
            payload = {
                **manifest_base,
                "status": "PASS_CPU_N1_ANALYTIC_ONLY",
                "tensorflow": str(tf.__version__),
                "memory_policy": memory_policy,
                "logical_devices": [str(device.name) for device in logical_devices],
                "route_scan": route_scan,
                "stages": stages,
                "wall_time_seconds": time.monotonic() - campaign_started,
                "gpu_snapshot_after": _gpu_snapshot(),
                "nonclaims": ["No q=20 training, whitening, posterior, convergence, or scaling claim"],
            }
            payload["source_hashes"] = {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (*ROUTE_PATHS, Path(__file__), PLAN_PATH, MASTER_PATH)
                if path.is_file()
            }
            payload["manifest_hash"] = _stable_hash(payload)
            _write_json(output_dir / "run_manifest.json", payload)
            print(json.dumps({"status": payload["status"], "output_dir": str(output_dir)}, sort_keys=True))
            return 0

        if time.monotonic() - campaign_started > args.max_seconds:
            raise DiagnosticError("budget exhausted after N1")
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

        bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend=BACKEND)
        if str(bridge.target_signature) != TARGET_SIGNATURE:
            raise DiagnosticError("q20 target signature mismatch")
        stages["n2_training_ladder"] = _n2_training_ladder(tf, bridge, max_seconds=args.max_seconds, campaign_started=campaign_started)
        payload = {
            **manifest_base,
            "status": "PASS_M3C_N0_N2_DIAGNOSTIC",
            "tensorflow": str(tf.__version__),
            "memory_policy": memory_policy,
            "logical_devices": [str(device.name) for device in logical_devices],
            "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "route_scan": route_scan,
            "target_signature": str(bridge.target_signature),
            "bridge_signature": str(bridge.signature),
            "properness_receipt": bridge.properness_receipt.payload(),
            "stages": stages,
            "wall_time_seconds": time.monotonic() - campaign_started,
            "gpu_snapshot_after": _gpu_snapshot(),
            "source_hashes": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (*ROUTE_PATHS, Path(__file__), PLAN_PATH, MASTER_PATH)
                if path.is_file()
            },
            "nonclaims": [
                "No IID-Gaussian whitening, mode-discovery, posterior, convergence, superiority, or scaling claim",
                "N1 fast grouped-HMC integration remains conditional on exact equivalence",
                "N2 nominations are diagnostic and cannot change active defaults",
            ],
        }
        payload["manifest_hash"] = _stable_hash(payload)
        _write_json(output_dir / "run_manifest.json", payload)
        print(json.dumps({"status": payload["status"], "output_dir": str(output_dir), "wall_time_seconds": payload["wall_time_seconds"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve durable failure evidence.
        failure = {
            **manifest_base,
            "status": "FAIL_M3C_N0_N2_DIAGNOSTIC",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stages_completed": stages,
            "wall_time_seconds": time.monotonic() - campaign_started,
            "gpu_snapshot_after": _gpu_snapshot(),
        }
        _write_json(output_dir / "failure.json", failure)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"DIAGNOSTIC_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

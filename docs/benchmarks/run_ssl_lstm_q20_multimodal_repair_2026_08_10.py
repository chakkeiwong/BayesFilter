#!/usr/bin/env python3
"""Validate exact replica exchange, then run a bounded SSL-LSTM canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs" / "benchmarks"
for directory in (ROOT, BENCHMARKS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-multimodal-repair-plan-2026-08-10.md"
)
RESULT = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-multimodal-repair-result-2026-08-10.md"
)
RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_multimodal_repair_2026_08_10.py"
)
HELPER = Path("bayesfilter/testing/replica_exchange_tf.py")
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-multimodal-repair-2026-08-10/r1"
)
GEOMETRY = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/"
    "r1/geometry.json"
)

THREADS = 8
SYNTHETIC_BETAS = (1.0, 0.3, 0.09, 0.027)
SYNTHETIC_COLD_STEP = 0.25
SYNTHETIC_STEPS = tuple(
    SYNTHETIC_COLD_STEP / math.sqrt(beta) for beta in SYNTHETIC_BETAS
)
SYNTHETIC_LEAPFROG = 4
SYNTHETIC_TOTAL_STEPS = 1000
SYNTHETIC_WARMUP = 200
SYNTHETIC_CHAINS = 8
SYNTHETIC_CAP_SECONDS = 900.0
SSL_BETAS = (1.0, 0.5, 0.25, 0.125)
SSL_COLD_STEP = 0.1
SSL_STEPS = tuple(SSL_COLD_STEP / math.sqrt(beta) for beta in SSL_BETAS)
SSL_LEAPFROG = 3
SSL_TRANSITIONS = 4
SSL_CHAINS = 2
SSL_CAP_SECONDS = 2400.0
PARAMETER_DIM = 4
OBSERVATION_WEIGHT_INDEX = 2


class RepairRunError(RuntimeError):
    """Raised when a stage-1 evidence-contract invariant fails."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise RepairRunError(f"refusing to overwrite artifact: {path}")
    encoded = (
        json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("ascii")
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)


def _write_tensor(path: Path, tensor: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise RepairRunError(f"refusing to overwrite artifact: {path}")
    value = tf.convert_to_tensor(tensor)
    encoded = bytes(tf.io.serialize_tensor(value).numpy())
    absolute.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": value.dtype.name,
        "shape": list(value.shape),
    }


def _configure_tf() -> tuple[Any, Any]:
    import tensorflow as tf
    import tensorflow_probability as tfp

    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise RepairRunError("CPU diagnostic found a visible TensorFlow GPU")
    return tf, tfp


def _load_replica_exchange_helper() -> Any:
    """Load this lane's helper independently of the historical code root."""

    path = _abs(HELPER)
    spec = importlib.util.spec_from_file_location(
        "bayesfilter_multimodal_repair_replica_exchange_tf", path
    )
    if spec is None or spec.loader is None:
        raise RepairRunError(f"could not load replica-exchange helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(mode: str, started: float, tf: Any, tfp: Any) -> Mapping[str, Any]:
    return {
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"), cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        ),
        "command": " ".join(sys.argv),
        "mode": mode,
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": THREADS,
        "jit_compile": True,
        "wall_time_seconds": time.perf_counter() - started,
        "artifact_root": OUTPUT_ROOT.as_posix(),
        "plan_file": PLAN.as_posix(),
        "result_file": RESULT.as_posix(),
        "source_sha256": {
            "plan": _sha(PLAN),
            "runner": _sha(RUNNER),
            "helper": _sha(HELPER),
        },
    }


def _mixture_target(tf: Any, weights: tuple[float, float]):
    dtype = tf.float64
    means = tf.constant((-4.0, 4.0), dtype)
    scales = tf.constant((0.5, 0.5), dtype)
    log_weights = tf.math.log(tf.constant(weights, dtype))
    log_normalizer = tf.math.log(scales) + 0.5 * tf.math.log(
        tf.constant(2.0 * math.pi, dtype)
    )

    def target(state: Any) -> Any:
        standardized = (state[..., 0, tf.newaxis] - means) / scales
        return tf.reduce_logsumexp(
            log_weights - 0.5 * tf.square(standardized) - log_normalizer,
            axis=-1,
        )

    return target


def _summarize_exchange(
    tf: Any,
    trace: Mapping[str, Any],
    *,
    warmup: int,
    observation_coordinate: int,
) -> Mapping[str, Any]:
    cold = trace["cold_states"][warmup:, ..., observation_coordinate]
    negative = cold < 0.0
    transitions = tf.reduce_sum(
        tf.cast(negative[1:] != negative[:-1], tf.int32)
    )
    proposed = tf.reduce_sum(
        tf.cast(trace["swap_is_proposed_adjacent"][warmup:], tf.int32),
        axis=(0, 2),
    )
    accepted = tf.reduce_sum(
        tf.cast(trace["swap_is_accepted_adjacent"][warmup:], tf.int32),
        axis=(0, 2),
    )
    safe_rates = tf.math.divide_no_nan(
        tf.cast(accepted, tf.float64), tf.cast(proposed, tf.float64)
    )
    return {
        "analyzed_steps": int(trace["cold_states"].shape[0]) - warmup,
        "warmup_steps_excluded": warmup,
        "cold_negative_fraction": tf.reduce_mean(tf.cast(negative, tf.float64)),
        "cold_sign_transitions": transitions,
        "hmc_acceptance_by_temperature": tf.reduce_mean(
            tf.cast(trace["hmc_is_accepted"][warmup:], tf.float64), axis=(0, 2)
        ),
        "adjacent_swap_proposals": proposed,
        "adjacent_swap_acceptances": accepted,
        "adjacent_swap_acceptance_rates": safe_rates,
        "identity_cold_visitors": tf.reduce_sum(
            tf.cast(trace["visited_cold"], tf.int32)
        ),
        "identity_hot_visitors": tf.reduce_sum(
            tf.cast(trace["visited_hot"], tf.int32)
        ),
        "completed_round_trips": tf.reduce_sum(trace["round_trip_returns"]),
    }


def _trace_receipts(prefix: str, trace: Mapping[str, Any], tf: Any) -> Mapping[str, Any]:
    names = (
        "replica_states",
        "hmc_is_accepted",
        "hmc_log_accept_ratio",
        "swap_is_proposed_adjacent",
        "swap_is_accepted_adjacent",
        "swap_is_accepted_matrix",
        "potential_energy",
        "replica_identities_at_temperature",
        "temperature_position_by_chain_identity",
        "round_trip_returns",
    )
    return {
        name: _write_tensor(OUTPUT_ROOT / f"{prefix}-{name}.tftensor", trace[name], tf)
        for name in names
    }


def run_synthetic() -> Mapping[str, Any]:
    started = time.perf_counter()
    tf, tfp = _configure_tf()
    replica_helper = _load_replica_exchange_helper()

    positive = tf.linspace(
        tf.constant(3.5, tf.float64),
        tf.constant(4.5, tf.float64),
        SYNTHETIC_CHAINS,
    )
    exchange_initial = tf.repeat(
        positive[tf.newaxis, :, tf.newaxis], len(SYNTHETIC_BETAS), axis=0
    )
    plain_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=_mixture_target(tf, (0.5, 0.5)),
        step_size=tf.constant(SYNTHETIC_COLD_STEP, tf.float64),
        num_leapfrog_steps=SYNTHETIC_LEAPFROG,
    )

    @tf.function(jit_compile=True, reduce_retracing=False)
    def plain_sample() -> Any:
        return tfp.mcmc.sample_chain(
            num_results=400,
            num_burnin_steps=100,
            current_state=exchange_initial[0],
            kernel=plain_kernel,
            trace_fn=None,
            seed=tf.constant((20260810, 1001), tf.int32),
        )

    plain_states = plain_sample()
    plain_negative = int(tf.reduce_sum(tf.cast(plain_states[..., 0] < 0.0, tf.int32)).numpy())
    fixtures = {}
    fixture_specs = (
        ("equal", (0.5, 0.5), (20260810, 1101), (0.40, 0.60)),
        ("unequal", (0.8, 0.2), (20260810, 1201), (0.68, 0.90)),
    )
    for label, weights, seed, band in fixture_specs:
        trace = replica_helper.run_replica_exchange_fixed_hmc(
            _mixture_target(tf, weights),
            exchange_initial,
            inverse_temperatures=SYNTHETIC_BETAS,
            step_sizes=SYNTHETIC_STEPS,
            num_leapfrog_steps=SYNTHETIC_LEAPFROG,
            num_steps=SYNTHETIC_TOTAL_STEPS,
            seed=seed,
            jit_compile=True,
        )
        summary = _summarize_exchange(
            tf, trace, warmup=SYNTHETIC_WARMUP, observation_coordinate=0
        )
        fraction = float(summary["cold_negative_fraction"].numpy())
        passed = bool(
            replica_helper.replica_exchange_finite(trace).numpy()
            and band[0] <= fraction <= band[1]
            and int(summary["cold_sign_transitions"].numpy()) > 100
            and int(summary["completed_round_trips"].numpy()) > 0
            and bool(tf.reduce_all(summary["adjacent_swap_proposals"] > 0).numpy())
            and bool(tf.reduce_all(summary["adjacent_swap_acceptances"] > 0).numpy())
        )
        fixtures[label] = {
            "weights": weights,
            "expected_negative_fraction": weights[0],
            "reviewed_regression_band": band,
            "seed": seed,
            "finite": bool(replica_helper.replica_exchange_finite(trace).numpy()),
            "summary": summary,
            "passed": passed,
            "trace_receipts": _trace_receipts(f"synthetic-{label}", trace, tf),
        }
    status = (
        "SYNTHETIC_VALIDATION_PASSED"
        if plain_negative == 0 and all(row["passed"] for row in fixtures.values())
        else "SYNTHETIC_VALIDATION_FAILED"
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_multimodal_repair.synthetic.v1",
        "status": status,
        "role": "analytic_multimodal_harness_validation",
        "configuration": {
            "inverse_temperatures": SYNTHETIC_BETAS,
            "step_sizes": SYNTHETIC_STEPS,
            "num_leapfrog_steps": SYNTHETIC_LEAPFROG,
            "total_steps": SYNTHETIC_TOTAL_STEPS,
            "warmup_steps": SYNTHETIC_WARMUP,
            "chain_count": SYNTHETIC_CHAINS,
            "initialization": "all chains and replicas in positive mode",
            "means": (-4.0, 4.0),
            "scales": (0.5, 0.5),
        },
        "plain_hmc": {
            "seed": (20260810, 1001),
            "sample_count": int(tf.size(plain_states[..., 0]).numpy()),
            "negative_count": plain_negative,
            "classification": "known_failure_comparator",
            "samples": _write_tensor(
                OUTPUT_ROOT / "synthetic-plain-hmc-states.tftensor", plain_states, tf
            ),
        },
        "fixtures": fixtures,
        "run_manifest": _manifest("synthetic", started, tf, tfp),
        "nonclaims": (
            "regression fixtures are not SSL posterior evidence",
            "no stochastic method ranking",
            "no repository default promotion",
        ),
    }
    if time.perf_counter() - started > SYNTHETIC_CAP_SECONDS:
        raise RepairRunError("synthetic wall-time cap breached")
    _write_json(OUTPUT_ROOT / "synthetic.json", payload)
    if status != "SYNTHETIC_VALIDATION_PASSED":
        raise RepairRunError(status)
    return payload


def _ssl_target(tf: Any, adapter: Any):
    @tf.custom_gradient
    def value_with_reviewed_score(state: Any) -> tuple[Any, Any]:
        flat = tf.reshape(state, (-1, PARAMETER_DIM))
        value, score, _status = adapter.log_prob_and_grad_status(flat)
        values = tf.reshape(tf.convert_to_tensor(value, tf.float64), tf.shape(state)[:-1])
        scores = tf.reshape(tf.convert_to_tensor(score, tf.float64), tf.shape(state))

        def grad(upstream: Any) -> Any:
            return scores * tf.convert_to_tensor(upstream, tf.float64)[..., tf.newaxis]

        return values, grad

    return value_with_reviewed_score


def run_ssl_canary() -> Mapping[str, Any]:
    synthetic_path = OUTPUT_ROOT / "synthetic.json"
    if not _abs(synthetic_path).exists():
        raise RepairRunError("passed synthetic artifact is required before SSL canary")
    synthetic = json.loads(_abs(synthetic_path).read_text(encoding="utf-8"))
    if synthetic.get("status") != "SYNTHETIC_VALIDATION_PASSED":
        raise RepairRunError("synthetic artifact did not pass")
    started = time.perf_counter()
    # This shared reconstruction checks checkpoint/source bindings and archived
    # target parity before returning the exact transformed target.
    import run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10 as root_cause

    tf, _bridge, transport, adapter, bindings = root_cause._configure_and_build()
    import tensorflow_probability as tfp
    replica_helper = _load_replica_exchange_helper()

    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    endpoints = geometry["transformed_optimization"]["endpoints"]
    plus = tf.constant(endpoints["plus"]["z"], tf.float64)
    minus = tf.constant(endpoints["minus"]["z"], tf.float64)
    pair = tf.stack((plus, minus), axis=0)
    initial = tf.repeat(pair[tf.newaxis, :, :], len(SSL_BETAS), axis=0)
    trace = replica_helper.run_replica_exchange_fixed_hmc(
        _ssl_target(tf, adapter),
        initial,
        inverse_temperatures=SSL_BETAS,
        step_sizes=SSL_STEPS,
        num_leapfrog_steps=SSL_LEAPFROG,
        num_steps=SSL_TRANSITIONS,
        seed=(20260810, 2101),
        jit_compile=True,
    )
    status_invalid = 0
    for step in range(SSL_TRANSITIONS):
        flat = tf.reshape(trace["replica_states"][step], (-1, PARAMETER_DIM))
        _value, _score, target_status = adapter.log_prob_and_grad_status(flat)
        code = tf.convert_to_tensor(target_status["status_code"], tf.int32)
        valid = tf.convert_to_tensor(
            target_status["valid_pre_regularized_score"], tf.bool
        )
        status_invalid += int(
            tf.reduce_sum(
                tf.cast(tf.logical_or(code != 0, tf.logical_not(valid)), tf.int32)
            ).numpy()
        )
    theta = transport.forward_z_to_theta_batch(
        tf.reshape(trace["replica_states"], (-1, PARAMETER_DIM))
    )
    theta = tf.reshape(
        theta, (SSL_TRANSITIONS, len(SSL_BETAS), SSL_CHAINS, PARAMETER_DIM)
    )
    signs = theta[..., OBSERVATION_WEIGHT_INDEX] < 0.0
    cold_signs = signs[:, 0]
    cold_transitions = tf.reduce_sum(
        tf.cast(cold_signs[1:] != cold_signs[:-1], tf.int32)
    )
    proposed = tf.reduce_sum(
        tf.cast(trace["swap_is_proposed_adjacent"], tf.int32), axis=(0, 2)
    )
    accepted = tf.reduce_sum(
        tf.cast(trace["swap_is_accepted_adjacent"], tf.int32), axis=(0, 2)
    )
    finite = bool(replica_helper.replica_exchange_finite(trace).numpy())
    passed = bool(
        finite
        and status_invalid == 0
        and bool(tf.reduce_all(proposed > 0).numpy())
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_multimodal_repair.ssl_canary.v1",
        "status": "SSL_CANARY_PASSED" if passed else "SSL_CANARY_FAILED",
        "role": "exact_target_replica_exchange_mechanics_and_timing_only",
        "configuration": {
            "inverse_temperatures": SSL_BETAS,
            "step_sizes": SSL_STEPS,
            "num_leapfrog_steps": SSL_LEAPFROG,
            "transitions": SSL_TRANSITIONS,
            "chains": SSL_CHAINS,
            "initialization": "both exact transformed stationary sign regions at every temperature",
            "seed": (20260810, 2101),
        },
        "finite": finite,
        "accepted_state_target_status_invalid_count": status_invalid,
        "hmc_acceptance_by_temperature": tf.reduce_mean(
            tf.cast(trace["hmc_is_accepted"], tf.float64), axis=(0, 2)
        ),
        "adjacent_swap_proposals": proposed,
        "adjacent_swap_acceptances": accepted,
        "adjacent_swap_acceptance_rates": tf.math.divide_no_nan(
            tf.cast(accepted, tf.float64), tf.cast(proposed, tf.float64)
        ),
        "cold_sign_transitions": cold_transitions,
        "identity_hot_visitors": tf.reduce_sum(tf.cast(trace["visited_hot"], tf.int32)),
        "completed_round_trips": tf.reduce_sum(trace["round_trip_returns"]),
        "trace_receipts": _trace_receipts("ssl-canary", trace, tf),
        "physical_signs": _write_tensor(
            OUTPUT_ROOT / "ssl-canary-physical-signs.tftensor", signs, tf
        ),
        "geometry": {"path": GEOMETRY.as_posix(), "sha256": _sha(GEOMETRY)},
        "synthetic_gate": {"path": synthetic_path.as_posix(), "sha256": _sha(synthetic_path)},
        "bindings": bindings,
        "run_manifest": _manifest("ssl-canary", started, tf, tfp),
        "nonclaims": (
            "four transitions are not warm-up or posterior sampling",
            "swap acceptance is not global mixing evidence",
            "cold occupancy is not a mode-weight estimate",
            "the two sign regions are not proved exhaustive",
            "no NeuTra or HMC default promotion",
        ),
    }
    if time.perf_counter() - started > SSL_CAP_SECONDS:
        raise RepairRunError("SSL canary wall-time cap breached")
    _write_json(OUTPUT_ROOT / "ssl-canary.json", payload)
    if not passed:
        raise RepairRunError(str(payload["status"]))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("synthetic", "ssl-canary"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_synthetic() if args.mode == "synthetic" else run_ssl_canary()
    print(json.dumps({"status": payload["status"], "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired q=20 strict/factor HMC mechanics promotion diagnostic.

The strict eigensystem route is the numerical authority.  This script binds
the same q=20 bridge, state bank, seeds, step sizes, and leapfrog counts to the
strict and factor-basis routes, then compares one-step and short full-chain
mechanics.  It is a diagnostic gate for implementation equivalence only; it
does not make a convergence, posterior, whitening, or sampler-ranking claim.
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
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STRICT = "tensorflow_eigh_strict"
FACTOR = "tensorflow_eigh_strict_factor_cached"
TRANSITION_RTOL = 1.0e-7
TRANSITION_ATOL = 1.0e-8
ENERGY_ERROR_ABS_LIMIT = 1.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_ready(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ("git", *args), cwd=ROOT, text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable:{type(exc).__name__}"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _source_sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true", help="hide GPUs before TensorFlow import")
    parser.add_argument("--no-jit", action="store_true", help="diagnostic non-XLA exception")
    parser.add_argument("--num-results", type=int, default=1)
    parser.add_argument("--num-burnin-steps", type=int, default=0)
    return parser.parse_args()


def _max_abs_difference(tf: Any, left: Any, right: Any) -> float:
    left_tensor = tf.cast(tf.convert_to_tensor(left), tf.float64)
    right_tensor = tf.cast(tf.convert_to_tensor(right), tf.float64)
    return float(tf.reduce_max(tf.abs(left_tensor - right_tensor)).numpy())


def _all_finite(tf: Any, value: Any) -> bool:
    tensor = tf.convert_to_tensor(value)
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(tensor, tf.float64))).numpy())


def _within_tolerance(tf: Any, left: Any, right: Any) -> bool:
    left_tensor = tf.cast(tf.convert_to_tensor(left), tf.float64)
    right_tensor = tf.cast(tf.convert_to_tensor(right), tf.float64)
    tolerance = tf.constant(TRANSITION_ATOL, tf.float64) + tf.constant(
        TRANSITION_RTOL, tf.float64
    ) * tf.abs(left_tensor)
    return bool(tf.reduce_all(tf.abs(left_tensor - right_tensor) <= tolerance).numpy())


def _tensor_fields(tf: Any, trace: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in (
        "is_accepted",
        "log_accept_ratio",
        "delta_h",
        "target_log_prob",
        "proposed_target_log_prob",
        "target_score",
        "divergence",
    ):
        if name in trace:
            fields[name] = trace[name]
    telemetry = trace.get("target_status_telemetry")
    if isinstance(telemetry, Mapping):
        fields["target_status_telemetry"] = telemetry
    return fields


def _compare_results(tf: Any, strict_result: Any, factor_result: Any) -> dict[str, Any]:
    strict_trace = _tensor_fields(tf, strict_result.trace)
    factor_trace = _tensor_fields(tf, factor_result.trace)
    numeric_fields = (
        "log_accept_ratio",
        "delta_h",
        "target_log_prob",
        "proposed_target_log_prob",
        "target_score",
    )
    numeric: dict[str, Any] = {}
    numeric_pass = True
    finite_pass = _all_finite(tf, strict_result.samples) and _all_finite(
        tf, factor_result.samples
    )
    for field in numeric_fields:
        if field not in strict_trace or field not in factor_trace:
            numeric[field] = {"present": False}
            numeric_pass = False
            finite_pass = False
            continue
        difference = _max_abs_difference(tf, strict_trace[field], factor_trace[field])
        field_finite = _all_finite(tf, strict_trace[field]) and _all_finite(
            tf, factor_trace[field]
        )
        within = _within_tolerance(tf, strict_trace[field], factor_trace[field])
        numeric[field] = {
            "present": True,
            "max_abs_difference": difference,
            "finite": field_finite,
            "within_tolerance": within,
        }
        numeric_pass = numeric_pass and within
        finite_pass = finite_pass and field_finite

    acceptance_equal = False
    if "is_accepted" in strict_trace and "is_accepted" in factor_trace:
        acceptance_equal = bool(
            tf.reduce_all(
                tf.equal(strict_trace["is_accepted"], factor_trace["is_accepted"])
            ).numpy()
        )
    status_equal = True
    if "target_status_telemetry" in strict_trace and "target_status_telemetry" in factor_trace:
        for key in ("status_code", "valid_pre_regularized_score"):
            if key not in strict_trace["target_status_telemetry"] or key not in factor_trace[
                "target_status_telemetry"
            ]:
                status_equal = False
                continue
            status_equal = status_equal and bool(
                tf.reduce_all(
                    tf.equal(
                        strict_trace["target_status_telemetry"][key],
                        factor_trace["target_status_telemetry"][key],
                    )
                ).numpy()
            )
    strict_divergence = strict_trace.get("divergence")
    factor_divergence = factor_trace.get("divergence")
    divergence_equal = True
    divergence_available = strict_divergence is not None and factor_divergence is not None
    if divergence_available:
        divergence_equal = bool(
            tf.reduce_all(tf.equal(strict_divergence, factor_divergence)).numpy()
        )
    energy_error_max = max(
        float(tf.reduce_max(tf.abs(tf.cast(strict_trace["delta_h"], tf.float64))).numpy()),
        float(tf.reduce_max(tf.abs(tf.cast(factor_trace["delta_h"], tf.float64))).numpy()),
    )
    energy_error_within_limit = energy_error_max <= ENERGY_ERROR_ABS_LIMIT
    native_divergence_clear = True
    if divergence_available:
        native_divergence_clear = not bool(
            tf.reduce_any(strict_divergence).numpy()
        ) and not bool(tf.reduce_any(factor_divergence).numpy())
    return {
        "samples_max_abs_difference": _max_abs_difference(
            tf, strict_result.samples, factor_result.samples
        ),
        "samples_within_tolerance": _within_tolerance(
            tf, strict_result.samples, factor_result.samples
        ),
        "samples_finite": finite_pass,
        "numeric": numeric,
        "acceptance_equal": acceptance_equal,
        "status_equal": status_equal,
        "divergence_available": divergence_available,
        "divergence_equal": divergence_equal,
        "native_divergence_clear": native_divergence_clear,
        "native_divergence_status": (
            "clear" if divergence_available else "unavailable"
        ),
        "energy_error_max_abs": energy_error_max,
        "energy_error_abs_limit": ENERGY_ERROR_ABS_LIMIT,
        "energy_error_within_limit": energy_error_within_limit,
        "strict_runner_trace_count": strict_result.metadata.get("runner_trace_count"),
        "factor_runner_trace_count": factor_result.metadata.get("runner_trace_count"),
        "pass": bool(
            finite_pass
            and numeric_pass
            and _within_tolerance(tf, strict_result.samples, factor_result.samples)
            and acceptance_equal
            and status_equal
            and divergence_equal
            and native_divergence_clear
            and energy_error_within_limit
        ),
    }


def main() -> int:
    args = _parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true is required")
    if args.num_results <= 0 or args.num_burnin_steps < 0:
        raise ValueError("result count, burn-in, and repeats are invalid")

    import tensorflow as tf

    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportFullChainConfig,
        FixedTransportHMCPolicy,
        build_fixed_transport_reusable_runner,
    )
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    memory_policy = None
    if args.cpu:
        physical = tuple(tf.config.list_physical_devices("GPU"))
    else:
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        physical = tuple(tf.config.list_physical_devices("GPU"))
        tf.config.experimental.enable_tensor_float_32_execution(True)
    if not physical and not args.cpu:
        raise RuntimeError("no GPU is visible; rerun with --cpu for a diagnostic reference")
    logical = tuple(tf.config.list_logical_devices("GPU"))
    jit_compile = not args.no_jit

    state = tf.constant(
        [
            [0.35, -0.08, 0.65, 0.05],
            [0.25, 0.02, 0.90, 0.12],
            [0.45, -0.20, 0.40, -0.10],
            [0.15, 0.10, 0.70, 0.20],
        ],
        tf.float64,
    )
    # The cold and intermediate levels exercise the likelihood score; beta=0
    # would mostly reduce the transition comparison to the Gaussian prior.
    cases = (
        {"name": "beta_half_eps_1e-3_l2", "beta": 0.5, "step_size": 1.0e-3, "leapfrog": 2},
        {"name": "beta_one_eps_1e-3_l2", "beta": 1.0, "step_size": 1.0e-3, "leapfrog": 2},
    )
    seeds = ((20260904, 601), (20260904, 602))
    records: dict[str, Any] = {}
    started_campaign = time.perf_counter()
    for case in cases:
        backend_results: dict[str, Any] = {}
        runners: dict[str, Any] = {}
        adapters: dict[str, Any] = {}
        for backend in (STRICT, FACTOR):
            bridge = make_q20_tempered_bridge(
                20, jit_compile=jit_compile, principal_sqrt_backend=backend
            )
            adapter = bridge.fixed_beta_adapter(case["beta"])
            config = FixedTransportFullChainConfig(
                num_results=args.num_results,
                num_burnin_steps=args.num_burnin_steps,
                step_size=case["step_size"],
                num_leapfrog_steps=case["leapfrog"],
                seed=seeds[0],
                use_xla=jit_compile,
                trace_policy="standard",
                target_status_trace_policy="per_chain_step",
                tuning_policy=FixedTransportHMCPolicy.fixed(
                    source="q20-factor-route-promotion-test-plan-2026-09-04"
                ),
                target_scope=adapter.target_scope,
                chain_execution_mode="tf_function" if jit_compile else "eager",
            )
            runners[backend] = build_fixed_transport_reusable_runner(adapter, state, config)
            adapters[backend] = adapter
        for seed in seeds:
            seed_name = f"seed_{seed[0]}_{seed[1]}"
            run_results: dict[str, Any] = {}
            for backend in (STRICT, FACTOR):
                run_results[backend] = runners[backend].run(
                    current_state=state,
                    seed=seed,
                    step_size=case["step_size"],
                    num_leapfrog_steps=case["leapfrog"],
                )
            comparison = _compare_results(
                tf, run_results[STRICT], run_results[FACTOR]
            )
            backend_results[seed_name] = {
                "comparison": comparison,
                "strict": {
                    "samples": _json_ready(run_results[STRICT].samples),
                    "trace": _json_ready(run_results[STRICT].trace),
                    "diagnostics": _json_ready(run_results[STRICT].diagnostics),
                },
                "factor": {
                    "samples": _json_ready(run_results[FACTOR].samples),
                    "trace": _json_ready(run_results[FACTOR].trace),
                    "diagnostics": _json_ready(run_results[FACTOR].diagnostics),
                },
            }
        records[case["name"]] = {
            **case,
            "backend_identities": {
                backend: {
                    "target_signature": adapters[backend].bridge.target_signature,
                    "component_target_adapter_signature": adapters[
                        backend
                    ].bridge.component_target_adapter_signature,
                    "bridge_signature": adapters[backend].bridge.signature,
                    "adapter_signature": adapters[backend].adapter_signature(),
                }
                for backend in (STRICT, FACTOR)
            },
            "runner_program_signatures": {
                backend: runners[backend].program_signature
                for backend in (STRICT, FACTOR)
            },
            "runner_trace_counts": {
                backend: runners[backend].tracing_count
                for backend in (STRICT, FACTOR)
            },
            "runs": backend_results,
        }

    all_pass = all(
        run["comparison"]["pass"]
        for case in records.values()
        for run in case["runs"].values()
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm_q20.factor_route_hmc_promotion.v1",
        "status": "PASS_MECHANICS_PARITY" if all_pass else "FAIL_MECHANICS_PARITY",
        "python": sys.executable,
        "platform": platform.platform(),
        "jit_compile": jit_compile,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""),
        "physical_gpus": [str(device) for device in physical],
        "logical_gpus": [str(device) for device in logical],
        "memory_policy": _json_ready(memory_policy),
        "plan_path": "docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-plan-2026-09-04.md",
        "command": list(sys.argv),
        "git": _git_metadata(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "seed_ledger": {
            "seeds": [list(seed) for seed in seeds],
            "all_unique": len(seeds) == len(set(seeds)),
        },
        "source_provenance": {
            relative: {
                "path": relative,
                "sha256": _source_sha256(relative),
            }
            for relative in (
                str(Path(__file__).relative_to(ROOT)),
                "bayesfilter/inference/tempered_target_tf.py",
                "bayesfilter/inference/fixed_transport_hmc_mechanics_tf.py",
                "bayesfilter/inference/batched_value_score.py",
            )
        },
        "state_shape": [int(value) for value in state.shape],
        "transition_rtol": TRANSITION_RTOL,
        "transition_atol": TRANSITION_ATOL,
        "energy_error_abs_limit": ENERGY_ERROR_ABS_LIMIT,
        "cases": records,
        "elapsed_seconds": time.perf_counter() - started_campaign,
        "nonclaims": [
            "no posterior correctness",
            "no convergence or R-hat/ESS claim",
            "no whitening or mode-discovery claim",
            "no sampler superiority or production-readiness claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": payload["status"], "output": str(args.output), "elapsed_seconds": payload["elapsed_seconds"]},
            sort_keys=True,
        )
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

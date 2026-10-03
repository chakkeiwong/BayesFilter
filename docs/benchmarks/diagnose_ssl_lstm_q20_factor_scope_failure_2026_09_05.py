#!/usr/bin/env python3
"""Localize the q=20 factor-route beta=.5 tuning failure.

The diagnostic restores the failed scope's frozen chart and evaluates the same
latent bank through the strict and strict-factor target backends.  It then
runs a short, fixed HMC call at deliberately small step sizes for each
backend.  The output is explanatory evidence only; it cannot promote a
backend, chart, or sampler.
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
CHECKPOINT_PATH = ROOT / (
    "docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/"
    "full-attempt-20260904T185852Z/chart-0/beta-0.5/chart_checkpoint.json"
)
PLAN_PATH = ROOT / (
    "docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md"
)
STRICT = "tensorflow_eigh_strict"
FACTOR = "tensorflow_eigh_strict_factor_cached"
BACKENDS = (STRICT, FACTOR)
DIMENSION = 4
CHAIN_COUNT = 4
# The first receipt used the four very-small steps below.  This extended
# ladder adds the proposed repair band so the next grid is bounded by measured
# finite calls rather than by an untested extrapolation.
DIAGNOSTIC_STEPS = (5.5e-2, 6.0e-2, 6.5e-2, 7.0e-2, 7.5e-2, 8.0e-2, 8.5e-2, 9.0e-2)
SEEDS = ((20260905, 96001), (20260905, 96002))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__nonfinite__": value.__repr__()}
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git() -> Mapping[str, Any]:
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


def _nvidia() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": f"{type(exc).__name__}: {exc}"}


def _finite(tf: Any, value: Any) -> bool:
    tensor = tf.cast(tf.convert_to_tensor(value), tf.float64)
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _range(tf: Any, value: Any) -> Mapping[str, Any]:
    tensor = tf.cast(tf.convert_to_tensor(value), tf.float64)
    finite = tf.math.is_finite(tensor)
    rows = tf.boolean_mask(tf.reshape(tensor, [-1]), tf.reshape(finite, [-1]))
    if int(tf.size(rows).numpy()) == 0:
        return {"finite_count": 0, "min": None, "max": None, "max_abs": None}
    return {
        "finite_count": int(tf.size(rows).numpy()),
        "min": float(tf.reduce_min(rows).numpy()),
        "max": float(tf.reduce_max(rows).numpy()),
        "max_abs": float(tf.reduce_max(tf.abs(rows)).numpy()),
    }


def _load_chart(restore: Any) -> tuple[Any, Mapping[str, Any]]:
    payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("failed chart checkpoint has no checkpoint payload")
    chart = restore(checkpoint)
    binder = getattr(chart, "bind_frozen_identity", None)
    if not callable(binder):
        raise RuntimeError("restored chart has no frozen identity binder")
    binder(
        {
            "checkpoint_sha256": checkpoint["checkpoint_hash"],
            "training_state_hash": checkpoint["transport_state_hash"],
            "transport_tensor_hash": checkpoint["transport_state_hash"],
        }
    )
    return chart, checkpoint


def _main(args: argparse.Namespace) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise RuntimeError("diagnostic requires GPU0")

    import tensorflow as tf

    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportFullChainConfig,
        FixedTransportHMCPolicy,
        FixedTransportReusableRunnerPool,
        build_fixed_transport_value_score_adapter,
    )
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        restore_trainable_transport_checkpoint,
    )
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected exactly one visible logical GPU, got {len(logical)}")

    chart, checkpoint = _load_chart(restore_trainable_transport_checkpoint)
    z_bank = tf.constant(
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.25, 0.0, 0.0, 0.0),
            (-0.25, 0.0, 0.0, 0.0),
            (0.0, 0.25, 0.0, 0.0),
        ),
        tf.float64,
    )
    latent_probe = tf.constant(
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.25, -0.15, 0.10, 0.05),
            (-0.50, 0.30, -0.20, 0.15),
            (1.0, -0.75, 0.5, -0.25),
            (2.0, 1.5, -1.0, 0.75),
            (-2.0, 1.25, 0.5, -1.5),
        ),
        tf.float64,
    )
    physical_probe = chart.forward_batch(latent_probe)
    records: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    for backend in BACKENDS:
        bridge = make_q20_tempered_bridge(
            20, jit_compile=True, principal_sqrt_backend=backend
        )
        base = bridge.fixed_beta_adapter(0.5)
        adapter = build_fixed_transport_value_score_adapter(
            base_adapter=base,
            fixed_transport=chart,
            target_scope=f"{base.target_scope}:chart=phase9a-chart-0:diagnostic",
            evidence_path=str(PLAN_PATH.relative_to(ROOT)),
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
        )
        adapters[backend] = adapter
        value, score, status = adapter.log_prob_and_grad_status(latent_probe)
        base_value, base_score, base_status = base.log_prob_and_grad_status(
            physical_probe
        )
        logdet = chart.log_abs_det_jacobian_batch(latent_probe)
        pullback = chart.pullback_score_batch(latent_probe, base_score)
        logdet_score = chart.log_abs_det_jacobian_score_batch(latent_probe)
        records[backend] = {
            "identity": {
                "bridge_signature": str(bridge.signature),
                "target_signature": str(bridge.target_signature),
                "component_target_adapter_signature": str(
                    bridge.component_target_adapter_signature
                ),
                "adapter_signature": adapter.adapter_signature(),
            },
            "probe": {
                "physical_finite": _finite(tf, physical_probe),
                "value": _range(tf, value),
                "score": _range(tf, score),
                "base_value": _range(tf, base_value),
                "base_score": _range(tf, base_score),
                "logdet": _range(tf, logdet),
                "pullback_score": _range(tf, pullback),
                "logdet_score": _range(tf, logdet_score),
                "value_finite": _finite(tf, value),
                "score_finite": _finite(tf, score),
                "status": _json_ready(status),
                "base_status": _json_ready(base_status),
            },
            "hmc": {},
        }

        pool = FixedTransportReusableRunnerPool()
        for step_index, step in enumerate(DIAGNOSTIC_STEPS):
            config = FixedTransportFullChainConfig(
                num_results=2,
                num_burnin_steps=1,
                step_size=step,
                num_leapfrog_steps=2,
                seed=SEEDS[step_index % len(SEEDS)],
                use_xla=True,
                trace_policy="standard",
                target_status_trace_policy="per_chain_step",
                tuning_policy=FixedTransportHMCPolicy.fixed(
                    source="q20-factor-scope-failure-diagnostic-2026-09-05"
                ),
                target_scope=adapter.target_scope,
                chain_execution_mode="tf_function",
                maximum_candidate_step_size=2.0,
            )
            started = time.perf_counter()
            try:
                result = pool(adapter, z_bank, config)
                diagnostics = dict(result.diagnostics)
                records[backend]["hmc"][str(step)] = {
                    "status": "RETURNED",
                    "elapsed_seconds": time.perf_counter() - started,
                    "diagnostics": _json_ready(diagnostics),
                    "samples_finite": _finite(tf, result.samples),
                    "trace": _json_ready(result.trace),
                }
            except Exception as exc:  # diagnostic must preserve backend-local failure
                records[backend]["hmc"][str(step)] = {
                    "status": "ERROR",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        records[backend]["runner_pool"] = _json_ready(pool.evidence())

    strict_probe = records[STRICT]["probe"]
    factor_probe = records[FACTOR]["probe"]
    strict_bridge = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend=STRICT
    )
    factor_bridge = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend=FACTOR
    )
    strict_value, strict_score, _ = strict_bridge.value_score_status(
        physical_probe, tf.constant(0.5, tf.float64)
    )
    factor_value, factor_score, _ = factor_bridge.value_score_status(
        physical_probe, tf.constant(0.5, tf.float64)
    )
    comparison = {
        "same_physical_probe": True,
        "target_value_max_abs_difference": float(
            tf.reduce_max(tf.abs(strict_value - factor_value)).numpy()
        ),
        "target_score_max_abs_difference": float(
            tf.reduce_max(tf.abs(strict_score - factor_score)).numpy()
        ),
        "strict_target_value_finite": _finite(tf, strict_value),
        "factor_target_value_finite": _finite(tf, factor_value),
        "strict_target_score_finite": _finite(tf, strict_score),
        "factor_target_score_finite": _finite(tf, factor_score),
        "interpretation": (
            "backend-localized"
            if strict_probe["value_finite"] and not factor_probe["value_finite"]
            else (
                "chart_or_proposal_scale_localized"
                if not strict_probe["value_finite"] and not factor_probe["value_finite"]
                else "inconclusive"
            )
        ),
    }
    allocator = tf.config.experimental.get_memory_info(str(logical[0].name))
    payload = {
        "schema": "bayesfilter.ssl_lstm_q20.factor_scope_failure_diagnostic.v1",
        "status": "PASS_DIAGNOSTIC_COMPLETED",
        "role": "explanatory_backend_localization_only",
        "profile_id": "phase9a_factor_tuning_full_v1_failed_scope_1",
        "failed_scope": {"chart_index": 0, "beta": 0.5, "scope_index": 1},
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "backend_records": records,
        "strict_factor_comparison": comparison,
        "device": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpus": [str(item.name) for item in logical],
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "memory_policy": memory_policy,
            "allocator": _json_ready(allocator),
            "nvidia_before": _nvidia(),
        },
        "git": _git(),
        "python": sys.executable,
        "platform": platform.platform(),
        "source_paths": {
            "plan": str(PLAN_PATH.relative_to(ROOT)),
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        },
        "seeds": [list(seed) for seed in SEEDS],
        "nonclaims": [
            "no backend promotion",
            "no posterior correctness",
            "no convergence or R-hat claim",
            "no whitening or mode-discovery claim",
        ],
    }
    _write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())

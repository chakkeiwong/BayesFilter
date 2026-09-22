#!/usr/bin/env python3
"""Bounded q=20 validation-bank chunking diagnostic.

This script tests graph feasibility only.  It never trains a transport and
cannot promote a sampler or a Gaussianization result.  The target is called
on static non-singleton batches throughout; host-side chunking is used solely
to avoid compiling one very large validation graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
TOTAL_ROWS = 256
PARITY_ROWS = 32
DEFAULT_GPU_ID = "0"
SCHEMA = "bayesfilter.ssl_lstm_q20.c1_graph_repair.v1"


def _prepare_gpu_environment() -> Mapping[str, Any]:
    visible_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_before is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "BAYESFILTER_GPU_ID", DEFAULT_GPU_ID
        )
    growth_before = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")
    if growth_before is None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return {
        "cuda_visible_devices_before": visible_before,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tf_force_gpu_allow_growth_before": growth_before,
        "tf_force_gpu_allow_growth": os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH", ""
        ),
        "selection_policy": "repository_default_single_gpu_no_idle_probe",
    }


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _safe(value: Any, tf: Any) -> Any:
    if tf.is_tensor(value):
        return _safe(value.numpy(), tf)
    if isinstance(value, Mapping):
        return {str(key): _safe(item, tf) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item, tf) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _safe(value.tolist(), tf)
    if hasattr(value, "item"):
        return _safe(value.item(), tf)
    return str(value)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _gpu_snapshot() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": type(exc).__name__}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, choices=(8, 32), default=8)
    parser.add_argument(
        "--principal-sqrt-backend",
        choices=("compiled_custom_op", "tensorflow_eigh_strict"),
        default="compiled_custom_op",
    )
    return parser.parse_args()


def _stage(path: Path, name: str, status: str, **extra: Any) -> None:
    _write(
        path / f"stage-{name}.json",
        {
            "schema": "bayesfilter.ssl_lstm_q20.c1_graph_repair_stage.v1",
            "stage": name,
            "status": status,
            "wall_time_unix_seconds": time.time(),
            **extra,
        },
    )


def _diagnostic_payload(result: Any, tf: Any) -> Mapping[str, Any]:
    return {
        "batch_size": int(result.batch_size.numpy()),
        "valid_row_count": int(result.valid_row_count.numpy()),
        "finite": bool(result.finite.numpy()),
        "centered_log_density_rms": float(result.centered_log_density_rms.numpy()),
        "centered_log_density_median_abs": float(
            result.centered_log_density_median_abs.numpy()
        ),
        "centered_log_density_q90_abs": float(
            result.centered_log_density_q90_abs.numpy()
        ),
        "pullback_score_maximum_row_norm": float(
            result.pullback_score_maximum_row_norm.numpy()
        ),
        "reverse_kl_row_count": int(result.reverse_kl_per_sample.shape[0]),
        "score_width": int(result.pullback_score_residual.shape[-1]),
        "_tensor_type": str(type(result.reverse_kl_per_sample).__name__),
    }


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    gpu_environment = _prepare_gpu_environment()
    started = time.monotonic()
    _stage(output_dir, "process", "STARTED", command=sys.argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise RuntimeError("graph repair requires one visible GPU")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true is required")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tensorflow as tf

    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        AffineDiagonalTransport,
        chunked_pullback_gaussianization_diagnostic,
    )
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected exactly one visible logical GPU, got {logical_gpus}")
    device_name = str(logical_gpus[0].name)
    bridge = make_q20_tempered_bridge(
        20,
        jit_compile=True,
        principal_sqrt_backend=args.principal_sqrt_backend,
    )
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise RuntimeError("q20 target signature changed")
    dimension = int(bridge.parameter_dim)
    chart = AffineDiagonalTransport(
        bridge.prior_center,
        tf.fill([dimension], tf.sqrt(tf.constant(bridge.prior_variance, tf.float64))),
        component_id="c1-graph-repair-reference-affine",
    )
    latent = tf.random.stateless_normal(
        [TOTAL_ROWS, dimension], tf.constant([20260830, 81001], tf.int32), dtype=tf.float64
    )
    beta = 0.0
    tolerance = 1.0e-12
    # The q=20 direct B=32 graph is itself the known expensive operation.  Use
    # a four-chunk B=8 prefix as the q=20 parity fixture, then compare its rows
    # with the same prefix in the full chunked bank.  Algebraic direct-vs-
    # chunked parity is covered by the analytic CPU tests, so this diagnostic
    # never needs to compile the large q=20 graph merely to establish parity.
    _stage(
        output_dir,
        "chunked-prefix-start",
        "STARTED",
        rows=PARITY_ROWS,
        static_shape=[args.chunk_size, dimension],
        chunk_size=args.chunk_size,
        beta=beta,
    )
    chunked_parity_started = time.perf_counter()
    chunked_parity = chunked_pullback_gaussianization_diagnostic(
        chart,
        bridge,
        beta=beta,
        latent=latent[:PARITY_ROWS],
        chunk_size=args.chunk_size,
    )
    chunked_parity_seconds = time.perf_counter() - chunked_parity_started
    _stage(
        output_dir,
        "chunked-prefix-done",
        "DONE",
        elapsed_seconds=chunked_parity_seconds,
        chunk_size=args.chunk_size,
        **_diagnostic_payload(chunked_parity, tf),
    )

    _stage(
        output_dir,
        "full-bank-start",
        "STARTED",
        rows=TOTAL_ROWS,
        chunk_size=args.chunk_size,
        chunk_count=TOTAL_ROWS // args.chunk_size,
        static_shape=[args.chunk_size, dimension],
        beta=beta,
    )
    full_started = time.perf_counter()
    full = chunked_pullback_gaussianization_diagnostic(
        chart,
        bridge,
        beta=beta,
        latent=latent,
        chunk_size=args.chunk_size,
    )
    full_seconds = time.perf_counter() - full_started
    full_payload = _diagnostic_payload(full, tf)
    prefix_parity = {
        "reverse_kl_max_abs": float(
            tf.reduce_max(
                tf.abs(
                    full.reverse_kl_per_sample[:PARITY_ROWS]
                    - chunked_parity.reverse_kl_per_sample
                )
            ).numpy()
        ),
        "density_residual_max_abs": float(
            tf.reduce_max(
                tf.abs(
                    full.pullback_log_density_residual[:PARITY_ROWS]
                    - chunked_parity.pullback_log_density_residual
                )
            ).numpy()
        ),
        "score_residual_max_abs": float(
            tf.reduce_max(
                tf.abs(
                    full.pullback_score_residual[:PARITY_ROWS]
                    - chunked_parity.pullback_score_residual
                )
            ).numpy()
        ),
        "tolerance": tolerance,
    }
    prefix_parity["passed"] = bool(
        all(
            float(prefix_parity[key]) <= tolerance
            for key in (
                "reverse_kl_max_abs",
                "density_residual_max_abs",
                "score_residual_max_abs",
            )
        )
    )
    _stage(
        output_dir,
        "full-bank-done",
        "DONE",
        elapsed_seconds=full_seconds,
        **full_payload,
        prefix_parity=prefix_parity,
    )
    if not bool(full.finite.numpy()) or int(full.valid_row_count.numpy()) != TOTAL_ROWS:
        raise RuntimeError("full chunked bank is invalid or incomplete")
    if not prefix_parity["passed"]:
        raise RuntimeError(f"chunked-prefix replay parity failed: {prefix_parity}")

    allocator = {
        key: int(value)
        for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_C1_CHUNKED_DIAGNOSTIC",
        "role": "graph_feasibility_repair_only",
        "command": sys.argv,
        "output_dir": str(output_dir),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "q": 20,
        "parameter_dim": dimension,
        "target_signature": str(bridge.target_signature),
        "bridge_signature": str(bridge.signature),
        "properness_receipt": bridge.properness_receipt.payload(),
        "chart": chart.initialization_payload(),
        "beta": beta,
        "latent_seed": [20260830, 81001],
        "latent_row_count": TOTAL_ROWS,
        "chunk_size": args.chunk_size,
        "chunk_count": TOTAL_ROWS // args.chunk_size,
        "per_chunk_static_shape": [args.chunk_size, dimension],
        "all_rows_accounted": int(full.batch_size.numpy()) == TOTAL_ROWS,
        "per_chunk_target_calls": TOTAL_ROWS // args.chunk_size,
        "direct_parity": {
            "role": "analytic_fixture_only; q20 direct B32 intentionally not compiled",
            "q20_prefix_replay": prefix_parity,
        },
        "full_bank": full_payload,
        "logical_gpus": [str(item.name) for item in logical_gpus],
        "operation_device": device_name,
        "memory_policy": memory_policy,
        "allocator": allocator,
        "gpu_environment": gpu_environment,
        "gpu_launch_mode": os.environ.get(
            "BAYESFILTER_GPU_LAUNCH_MODE", "c1_graph_repair_direct"
        ),
        "gpu_trust_basis": os.environ.get(
            "BAYESFILTER_GPU_TRUST_BASIS",
            "repository_default_gpu_route_external_boundary_unclassified",
        ),
        "external_approval_is_runner_gate": False,
        "gpu_snapshot_before": _gpu_snapshot(),
        "jit_compile": True,
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "wall_time_seconds": time.monotonic() - started,
        "nonclaims": [
            "graph-feasibility diagnostic only",
            "no transport training or candidate selection",
            "no whitening, mode-discovery, posterior, HMC, convergence, or scaling claim",
        ],
    }
    manifest["manifest_hash"] = _hash(manifest)
    _write(output_dir / "run_manifest.json", _safe(manifest, tf))
    _stage(
        output_dir,
        "process-done",
        "DONE",
        wall_time_seconds=manifest["wall_time_seconds"],
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "chunk_size": args.chunk_size,
                "chunk_count": TOTAL_ROWS // args.chunk_size,
                "wall_time_seconds": manifest["wall_time_seconds"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        # The output directory is created before framework import, so preserve
        # a structured failure when an ordinary Python exception occurs.
        try:
            parsed = _parse_args()
            failure_dir = parsed.output_dir.expanduser().resolve()
            if failure_dir.exists():
                _write(
                    failure_dir / "failure.json",
                    {
                        "schema": "bayesfilter.ssl_lstm_q20.c1_graph_repair_failure.v1",
                        "status": "FAIL_C1_CHUNKED_DIAGNOSTIC",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "command": sys.argv,
                        "nonclaims": ["no scientific or sampler conclusion"],
                    },
                )
        except Exception:
            pass
        raise

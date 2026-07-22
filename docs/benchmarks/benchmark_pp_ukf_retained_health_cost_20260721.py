#!/usr/bin/env python3
"""Measure the real PP-UKF retained-health callable cost.

Diagnostic-only Phase 4 artifact for the ten-phase tuning repair plan. The
benchmark never launches HMC tuning or posterior sampling and makes no
convergence or scientific claim.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _finite(value: Any) -> bool:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value)
    if tensor.dtype.is_integer or tensor.dtype.is_bool:
        return True
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _time_call(fn: Any, value: Any, *, warmups: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmups):
        fn(value)
    samples: list[float] = []
    outputs: Any = None
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = fn(value)
        samples.append(time.perf_counter() - started)
    return {
        "warmups": warmups,
        "repeats": repeats,
        "seconds": samples,
        "mean_seconds": sum(samples) / len(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-transport", type=Path, required=True)
    parser.add_argument("--frozen-transport-sha256", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats must be positive")

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == "PP-UKF")
    transport_sha = _sha256(args.frozen_transport)
    if transport_sha != str(args.frozen_transport_sha256).lower():
        raise ValueError(f"frozen transport SHA mismatch: {transport_sha}")
    payload = json.loads(args.frozen_transport.read_text(encoding="utf-8"))
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=spec.target_signature
    )
    raw = spec.adapter_factory()
    bound = BatchNativeBoundAdapter(raw, target_signature=spec.target_signature)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bound,
        transport=loaded.transport,
        target_scope="PP-UKF:ten_phase_retained_health_cost",
        evidence_path=str(Path(__file__).relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    scalar = tf.zeros((spec.parameter_dim,), tf.float64)
    flat = tf.zeros((64, spec.parameter_dim), tf.float64)
    retained = tf.zeros((16, 4, spec.parameter_dim), tf.float64)
    retained_flat = tf.reshape(retained, (-1, spec.parameter_dim))

    scalar_program = tf.function(
        adapter.log_prob_and_grad, jit_compile=True, reduce_retracing=True
    )
    combined_program = tf.function(
        adapter.log_prob_and_grad_status, jit_compile=True, reduce_retracing=True
    )
    scalar_value, scalar_score = scalar_program(scalar)
    flat_value, flat_score, flat_status = combined_program(flat)
    combined = _time_call(combined_program, retained_flat, warmups=args.warmups, repeats=args.repeats)
    scalar_timing = _time_call(scalar_program, scalar, warmups=args.warmups, repeats=args.repeats)
    flat_timing = _time_call(combined_program, flat, warmups=args.warmups, repeats=args.repeats)

    flat_value_expected, flat_score_expected = adapter.log_prob_and_grad(flat)
    parity = {
        "scalar_finite": _finite(scalar_value) and _finite(scalar_score),
        "flat_finite": _finite(flat_value) and _finite(flat_score),
        "flat_value_near": bool(tf.reduce_all(tf.abs(flat_value - flat_value_expected) <= 1.0e-12).numpy()),
        "flat_score_near": bool(tf.reduce_all(tf.abs(flat_score - flat_score_expected) <= 1.0e-12).numpy()),
        "status_fields": sorted(str(key) for key in flat_status),
    }
    if not all(
        [
            parity["scalar_finite"],
            parity["flat_finite"],
            parity["flat_value_near"],
            parity["flat_score_near"],
        ]
    ):
        raise RuntimeError(f"PP-UKF benchmark parity failed: {parity}")

    row = {
        "schema": "bayesfilter.pp_ukf_retained_health_cost.v1",
        "role": "diagnostic_cost_measurement_only",
        "nonclaims": [
            "no tuning result",
            "no posterior sampling",
            "no convergence or posterior correctness claim",
            "no sampler ranking or default-readiness claim",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "target_signature": spec.target_signature,
        "adapter_signature": bound.adapter_signature(),
        "transport_sha256": transport_sha,
        "frozen_transport": str(args.frozen_transport),
        "dtype": "float64",
        "jit_compile": True,
        "device_list": [str(item) for item in tf.config.list_logical_devices()],
        "memory_policy": memory_policy,
        "batch_contract": {
            "supports_retained_flat_batch": bool(adapter.supports_retained_flat_batch),
            "supports_retained_value_score_status": bool(adapter.supports_retained_value_score_status),
            "retained_shape": [16, 4, spec.parameter_dim],
            "retained_flat_shape": [64, spec.parameter_dim],
        },
        "parity": parity,
        "timings": {
            "scalar_value_score": {key: value for key, value in scalar_timing.items() if key != "outputs"},
            "flat_value_score_status": {key: value for key, value in flat_timing.items() if key != "outputs"},
            "retained_flat_value_score_status": {key: value for key, value in combined.items() if key != "outputs"},
        },
        "status_finite": _finite(flat_status["status_code"]),
        "status_fields": sorted(str(key) for key in flat_status),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"benchmark output must be fresh: {args.output}")
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True, default=_json_ready) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output), "retained_mean_seconds": row["timings"]["retained_flat_value_score_status"]["mean_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

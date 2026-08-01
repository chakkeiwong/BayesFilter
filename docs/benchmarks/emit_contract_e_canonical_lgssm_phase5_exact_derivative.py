#!/usr/bin/env python3
"""Emit the CPU-hidden Phase 5 exact same-core derivative certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical


DEFAULT_FIXTURE = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-tiny-fixture-freeze-v2-2026-07-14.json"
)


def _convert(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, str):
        return float(Fraction(value))
    return value


def _ordered_binary64(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits == 1 << 63:
        bits = 0
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def _ulp_table(left: tf.Tensor, right: tf.Tensor) -> list[list[int]]:
    left_values = left.numpy()
    right_values = right.numpy()
    return [
        [
            abs(_ordered_binary64(a) - _ordered_binary64(b))
            for a, b in zip(left_row, right_row, strict=True)
        ]
        for left_row, right_row in zip(left_values, right_values, strict=True)
    ]


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    fixture_path = args.fixture.resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    prepared = canonical._as_prepared_tensors(
        {
            "observations": _convert(fixture["observations"]),
            "initial_noise": _convert(fixture["initial_noise"]),
            "transition_noise": _convert(fixture["transition_noise"]),
            "fixed_reset_mask": fixture["fixed_reset_mask"],
            "residual_design": _convert(fixture["residual_design"]),
            "prepared_ridge": _convert(fixture["prepared_ridge"]),
            "epsilon": _convert(fixture["transport"]["epsilon"]),
            "scaling": _convert(fixture["transport"]["scaling"]),
        }
    )
    theta = tf.constant(_convert(fixture["center_theta"]), tf.float64)
    kwargs = {
        "steps": int(fixture["transport"]["finite_sinkhorn_steps"]),
        "balance_steps": 0,
        "row_chunk_size": int(fixture["transport"]["row_chunk_size"]),
        "col_chunk_size": int(fixture["transport"]["col_chunk_size"]),
    }
    primal = canonical._canonical_primal_core(theta, prepared, **kwargs)
    manual = canonical._canonical_manual_jvp_core(theta, prepared, **kwargs)
    columns = []
    for index in range(canonical.PARAMETER_COUNT):
        with tf.autodiff.ForwardAccumulator(
            theta, tf.one_hot(index, canonical.PARAMETER_COUNT, dtype=tf.float64)
        ) as accumulator:
            per_batch = canonical._canonical_primal_core(
                theta, prepared, **kwargs
            )["per_batch_log_likelihood"]
        columns.append(accumulator.jvp(per_batch))
    automatic = tf.stack(columns, axis=1)
    ulps = _ulp_table(manual["per_batch_score"], automatic)
    aggregate_automatic = tf.reduce_mean(automatic, axis=0)
    aggregate_ulps = _ulp_table(
        manual["score"][None, :], aggregate_automatic[None, :]
    )[0]
    payload = {
        "schema_version": "bayesfilter.contract_e_canonical_lgssm_phase5_exact_derivative.v1",
        "program_id": fixture["program_id"],
        "phase": 5,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ZERO_ULP_SAME_PRIVATE_PRIMAL_CORE_PASSED"
        if max(max(row) for row in ulps) == 0 and max(aggregate_ulps) == 0
        else "ZERO_ULP_SAME_PRIVATE_PRIMAL_CORE_FAILED",
        "fixture": str(fixture_path.relative_to(ROOT)),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            (ROOT / "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py").read_bytes()
        ).hexdigest(),
        "environment": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": [
                device.name for device in tf.config.list_logical_devices()
            ],
            "jit_compile": False,
            "execution_role": "CPU_HIDDEN_REFERENCE_EXACT_DERIVATIVE_CERTIFICATE",
            "wall_time_seconds": time.perf_counter() - started,
        },
        "parameter_names": list(canonical.PARAMETER_NAMES),
        "objective": float(primal["objective"]),
        "per_batch_log_likelihood": primal[
            "per_batch_log_likelihood"
        ].numpy().tolist(),
        "valid_chart": primal["valid_chart"].numpy().tolist(),
        "minimum_mass": primal["minimum_mass"].numpy().tolist(),
        "manual_per_batch_score": manual["per_batch_score"].numpy().tolist(),
        "forward_accumulator_per_batch_score": automatic.numpy().tolist(),
        "per_batch_ulp_distance": ulps,
        "maximum_per_batch_ulp_distance": max(max(row) for row in ulps),
        "manual_score": manual["score"].numpy().tolist(),
        "forward_accumulator_score": aggregate_automatic.numpy().tolist(),
        "aggregate_ulp_distance": aggregate_ulps,
        "maximum_aggregate_ulp_distance": max(aggregate_ulps),
        "hard_checks": {
            "all_charts_valid": bool(tf.reduce_all(primal["valid_chart"]).numpy()),
            "per_batch_zero_ulp": max(max(row) for row in ulps) == 0,
            "aggregate_zero_ulp": max(aggregate_ulps) == 0,
            "parameter_direction_axis_final_and_size_five": manual[
                "final_particles_tangent"
            ].shape.as_list()
            == [2, 4, 3, 5],
        },
        "nonclaims": [
            "this exact tiny-graph certificate is not a general forward-error bound",
            "this does not establish Kalman equivalence or production readiness",
            "FD evidence is separate and explanatory only",
        ],
    }
    _write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "hard_checks": payload["hard_checks"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

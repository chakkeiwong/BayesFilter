#!/usr/bin/env python3
"""Emit the Phase 5 same-callable Contract E LGSSM certificate."""

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
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical


DEFAULT_FIXTURE_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-tiny-fixture-freeze-v2-2026-07-14.json"
)
PLAN_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-canonical-graph-subplan-2026-07-14.md"
)
PARAMETER_NAMES = canonical.PARAMETER_NAMES
BRANCH_FIELDS = (
    "valid_chart",
    "flow_valid_history",
    "geometry_valid_history",
    "quotient_valid_history",
    "reset_valid_history",
    "diameter_max_mask",
    "geometry_max_mask",
    "geometry_min_mask",
    "epsilon0_floor_inactive",
    "sinkhorn_running_branch",
)


def _convert(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, str):
        return float(Fraction(value))
    return value


def _fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepared(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "observations": _convert(fixture["observations"]),
        "initial_noise": _convert(fixture["initial_noise"]),
        "transition_noise": _convert(fixture["transition_noise"]),
        "fixed_reset_mask": fixture["fixed_reset_mask"],
        "residual_design": _convert(fixture["residual_design"]),
        "prepared_ridge": _convert(fixture["prepared_ridge"]),
        "epsilon": _convert(fixture["transport"]["epsilon"]),
        "scaling": _convert(fixture["transport"]["scaling"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_payload(value: tf.Tensor) -> dict[str, Any]:
    tensor = tf.convert_to_tensor(value)
    return {
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
        "values": tensor.numpy().tolist(),
    }


def _tensor_hashes(values: dict[str, tf.Tensor]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, value in sorted(values.items()):
        encoded = json.dumps(
            _tensor_payload(value), sort_keys=True, separators=(",", ":")
        ).encode()
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    return hashes


def _branch_hash(result: dict[str, tf.Tensor]) -> str:
    encoded = json.dumps(
        {name: _tensor_payload(result[name]) for name in BRANCH_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _call_record(result: dict[str, tf.Tensor]) -> dict[str, Any]:
    return {
        "objective": float(result["objective"]),
        "per_batch_log_likelihood": result[
            "per_batch_log_likelihood"
        ].numpy().tolist(),
        "score": result["score"].numpy().tolist(),
        "per_batch_score": result["per_batch_score"].numpy().tolist(),
        "valid_chart": result["valid_chart"].numpy().tolist(),
        "minimum_mass": result["minimum_mass"].numpy().tolist(),
        "branch_hash": _branch_hash(result),
        "objective_hex": float(result["objective"]).hex(),
        "score_hex": [float(value).hex() for value in result["score"].numpy()],
    }


def _same_center(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = (
        "objective_hex",
        "per_batch_log_likelihood",
        "score_hex",
        "per_batch_score",
        "valid_chart",
        "minimum_mass",
        "branch_hash",
    )
    return all(left[name] == right[name] for name in fields)


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--no-jit-compile", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    fixture_path = args.fixture.resolve()
    fixture = _fixture(fixture_path)
    prepared = _prepared(fixture)
    dtype = tf.dtypes.as_dtype(args.dtype)
    prepared_tensors = canonical._as_prepared_tensors(prepared, dtype=dtype)
    theta = tf.constant(_convert(fixture["center_theta"]), dtype)
    jit_compile = not args.no_jit_compile
    callable_ = canonical.make_canonical_value_and_score_tf(
        prepared,
        steps=int(fixture["transport"]["finite_sinkhorn_steps"]),
        balance_steps=1,
        row_chunk_size=int(fixture["transport"]["row_chunk_size"]),
        col_chunk_size=int(fixture["transport"]["col_chunk_size"]),
        jit_compile=jit_compile,
        dtype=dtype,
    )

    center_first = _call_record(callable_(theta))
    center_second = _call_record(callable_(theta))
    center_identity = _same_center(center_first, center_second)
    endpoints: list[dict[str, Any]] = []
    fd_by_step: list[dict[str, Any]] = []
    for step_text in fixture["fd_step_ladder"]:
        step = tf.constant(float(Fraction(step_text)), dtype)
        derivatives = []
        step_endpoints = []
        for index, parameter in enumerate(PARAMETER_NAMES):
            direction = tf.one_hot(index, len(PARAMETER_NAMES), dtype=dtype)
            plus_result = callable_(theta + step * direction)
            minus_result = callable_(theta - step * direction)
            plus = _call_record(plus_result)
            minus = _call_record(minus_result)
            endpoints.extend(
                [
                    {
                        "step": step_text,
                        "parameter": parameter,
                        "sign": "plus",
                        **plus,
                    },
                    {
                        "step": step_text,
                        "parameter": parameter,
                        "sign": "minus",
                        **minus,
                    },
                ]
            )
            step_endpoints.extend([plus, minus])
            derivatives.append(
                (plus["objective"] - minus["objective"]) / (2.0 * float(step))
            )
        fd_by_step.append(
            {
                "step": step_text,
                "derivative": dict(zip(PARAMETER_NAMES, derivatives, strict=True)),
                "all_charts_valid": all(
                    all(item["valid_chart"]) for item in step_endpoints
                ),
                "all_branch_hashes_match_center": all(
                    item["branch_hash"] == center_first["branch_hash"]
                    for item in step_endpoints
                ),
            }
        )

    score = center_first["score"]
    fd_comparison = []
    for index, parameter in enumerate(PARAMETER_NAMES):
        values = [entry["derivative"][parameter] for entry in fd_by_step]
        fd_comparison.append(
            {
                "parameter": parameter,
                "manual_score": score[index],
                "fd_by_step": values,
                "absolute_difference_by_step": [
                    abs(value - score[index]) for value in values
                ],
                "classification": "EXPLANATORY_ONLY",
            }
        )

    source_paths = [
        ROOT / "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        ROOT
        / "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
    ]
    payload = {
        "schema_version": "bayesfilter.contract_e_canonical_lgssm_phase5_certificate.v1",
        "program_id": fixture["program_id"],
        "phase": 5,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTED_ENGINEERING_CERTIFICATE",
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "dtype": dtype.name,
            "git_commit": _git_commit(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": [
                device.name for device in tf.config.list_logical_devices()
            ],
            "jit_compile": jit_compile,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "fixture": str(fixture_path.relative_to(ROOT)),
        "fixture_sha256": _sha256(fixture_path),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "prepared_input_hashes": _tensor_hashes(prepared_tensors),
        "callable_identity": {
            "value_and_score_only": True,
            "concrete_function_count": len(
                callable_._list_all_concrete_functions_for_serialization()
            ),
            "fd_invokes_score_at_every_endpoint": True,
        },
        "center_first": center_first,
        "center_second": center_second,
        "center_bitwise_identity": center_identity,
        "fd_by_step": fd_by_step,
        "fd_comparison": fd_comparison,
        "endpoints": endpoints,
        "hard_checks": {
            "center_bitwise_identity": center_identity,
            "center_chart_valid": all(center_first["valid_chart"]),
            "all_endpoint_charts_valid": all(
                entry["all_charts_valid"] for entry in fd_by_step
            ),
            "all_endpoint_branches_match_center": all(
                entry["all_branch_hashes_match_center"] for entry in fd_by_step
            ),
            "one_concrete_value_and_score_callable": len(
                callable_._list_all_concrete_functions_for_serialization()
            )
            == 1,
        },
        "nonclaims": fixture["nonclaims"]
        + [
            "FD differences are explanatory and cannot certify the manual JVP",
            "this artifact does not establish Kalman equivalence",
        ],
    }
    _write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "hard_checks": payload["hard_checks"],
                "objective": center_first["objective"],
                "score": center_first["score"],
                "wall_time_seconds": payload["environment"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

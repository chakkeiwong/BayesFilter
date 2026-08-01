#!/usr/bin/env python3
"""Emit the seven-step heuristic-only Contract E FD screen."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from docs.benchmarks import run_contract_e_canonical_lgssm_phase5_certificate as phase5


FIXTURE = phase5.DEFAULT_FIXTURE_PATH
BASELINE_CERTIFICATE = ROOT / "docs/plans/logs" / (
    "contract-e-canonical-gradient-migration-2026-07-13/phase5/"
    "cpu-xla-same-callable-certificate-v2.json"
)
EXPECTED_FIXTURE_SHA256 = (
    "f6b6e2895208d7cd5cba0f57b05d4de7fb0de79e50ba62b7e6c70b06879942f4"
)
EXPECTED_BASELINE_SHA256 = (
    "20ec133bc5aee47f5daf3dc54d4c3593189202b1c305640cbd30b5e33b4ca709"
)
EXPECTED_OBJECTIVE_HEX = "-0x1.55564a66d9848p+2"
EXPECTED_SCORE_HEX = (
    "-0x1.c993b9119c770p-2",
    "-0x1.cad12b05cc707p-3",
    "0x1.cc0ca41e05574p-5",
    "-0x1.c6af389364ccfp+1",
    "-0x1.2fce89f3bf0cap+2",
)
EXPECTED_BRANCH_HASH = (
    "bf25ece12ff85525620fdc1284abab76a35a54c28a4f998b89bbabd56aa005d7"
)
EXPECTED_PREPARED_HASHES = {
    "epsilon": "030f81a2c06cf6b8e817ef95cee5e1dbc689d362633663646211eedb010e0425",
    "fixed_reset_mask": "e25f60bb1b783c8a8028955b9c19cd60460c36f90f9d0f653009dcb9ce45cd0a",
    "initial_noise": "1a34bec03362f6045362b47a776c6a5a3c38714ea5a486235f0c05b68dd4b232",
    "observations": "72577efbcb90533c55cbf25bac8c9b7b7b0fd55578c6797917f3a10dabe5862f",
    "prepared_ridge": "00914d116100b0aa34ca487e35812fd87d3094293bcb5a269cb76df4df28aa19",
    "residual_design": "1a892ab5f1164cc235f048883f667e508dcc368b6ad218a369f4a8cfc2af3b8e",
    "scaling": "3b68b91f6bb5eb755e369eff6aa0734ad175f4c846ee035e57ea916581947343",
    "transition_noise": "df590b64250c884879d26163245e0cff807aaf67bd0dda8b8268ce516d31ee8f",
}
LADDER_MULTIPLIERS = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
FD_THRESHOLD = 0.05 * math.sqrt(5.0)
MACHINE_EPSILON = 2.0**-52
NEAREST_DYADIC_EXPONENT = -17
NEAREST_DYADIC_BASE = 2.0**NEAREST_DYADIC_EXPONENT


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


def _validate_environment() -> list[str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 is required before import")
    devices = [device.name for device in tf.config.list_logical_devices()]
    if any("GPU" in device.upper() for device in devices):
        raise RuntimeError(f"CPU-hidden run exposed GPU: {devices}")
    return devices


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _classify_screen(records: list[dict[str, Any]]) -> str:
    if any(not record["endpoint_valid"] for record in records):
        return "FD_HEURISTIC_INCONCLUSIVE_INVALID_ENDPOINT"
    if any(not record["relative_denominator_eligible"] for record in records):
        return "FD_HEURISTIC_INCONCLUSIVE_NEAR_ZERO"
    if all(record["relative_error_pass"] for record in records):
        return "SEVEN_STEP_FD_HEURISTIC_SCREEN_PASSED"
    return "SEVEN_STEP_FD_HEURISTIC_SCREEN_FAILED"


def _nominal_steps(mode: str, center_coordinate: float) -> tuple[float, ...]:
    if mode == "cuberoot":
        base = MACHINE_EPSILON ** (1.0 / 3.0) * max(
            1.0, abs(center_coordinate)
        )
    elif mode == "nearest_dyadic_cuberoot":
        if abs(center_coordinate) >= 1.0:
            raise ValueError("nearest dyadic mode is frozen only for this fixture")
        base = NEAREST_DYADIC_BASE
    else:
        raise ValueError(f"unsupported step mode: {mode}")
    return tuple(multiplier * base for multiplier in LADDER_MULTIPLIERS)


def _representability_preflight(theta: tf.Tensor, mode: str) -> list[dict[str, Any]]:
    checks = []
    for index, parameter in enumerate(canonical.PARAMETER_NAMES):
        center_coordinate = float(theta[index])
        for multiplier, nominal_step in zip(
            LADDER_MULTIPLIERS,
            _nominal_steps(mode, center_coordinate),
            strict=True,
        ):
            plus_coordinate = tf.constant(
                center_coordinate + nominal_step, tf.float64
            )
            minus_coordinate = tf.constant(
                center_coordinate - nominal_step, tf.float64
            )
            actual_plus = float(plus_coordinate - center_coordinate)
            actual_minus = float(center_coordinate - minus_coordinate)
            nominal_hex = float(nominal_step).hex()
            checks.append(
                {
                    "parameter": parameter,
                    "parameter_index": index,
                    "center_hex": center_coordinate.hex(),
                    "multiplier": multiplier,
                    "nominal_step": nominal_step,
                    "nominal_step_hex": nominal_hex,
                    "actual_plus_step_hex": actual_plus.hex(),
                    "actual_minus_step_hex": actual_minus.hex(),
                    "plus_equals_nominal": actual_plus.hex() == nominal_hex,
                    "minus_equals_nominal": actual_minus.hex() == nominal_hex,
                    "actual_steps_equal": actual_plus.hex() == actual_minus.hex(),
                }
            )
    return checks


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--step-mode",
        choices=("cuberoot", "nearest_dyadic_cuberoot"),
        default="cuberoot",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if _sha256(FIXTURE) != EXPECTED_FIXTURE_SHA256:
        raise ValueError("Phase 5 v2 fixture hash drifted")
    if _sha256(BASELINE_CERTIFICATE) != EXPECTED_BASELINE_SHA256:
        raise ValueError("Phase 5 v2 certificate hash drifted")

    started = time.perf_counter()
    logical_devices = _validate_environment()
    fixture = phase5._fixture(FIXTURE)
    prepared = phase5._prepared(fixture)
    prepared_tensors = canonical._as_prepared_tensors(prepared, dtype=tf.float64)
    prepared_hashes = phase5._tensor_hashes(prepared_tensors)
    if prepared_hashes != EXPECTED_PREPARED_HASHES:
        raise ValueError("prepared input hashes drifted")
    theta = tf.constant(phase5._convert(fixture["center_theta"]), tf.float64)
    representability_preflight = _representability_preflight(theta, args.step_mode)
    if not all(
        check["plus_equals_nominal"]
        and check["minus_equals_nominal"]
        and check["actual_steps_equal"]
        for check in representability_preflight
    ):
        if args.step_mode == "nearest_dyadic_cuberoot":
            raise ValueError("representable-step preflight failed before endpoint calls")
    callable_ = canonical.make_canonical_value_and_score_tf(
        prepared,
        steps=int(fixture["transport"]["finite_sinkhorn_steps"]),
        balance_steps=1,
        row_chunk_size=int(fixture["transport"]["row_chunk_size"]),
        col_chunk_size=int(fixture["transport"]["col_chunk_size"]),
        jit_compile=True,
        dtype=tf.float64,
    )
    center_first = phase5._call_record(callable_(theta))
    center_second = phase5._call_record(callable_(theta))
    if (
        center_first["objective_hex"] != EXPECTED_OBJECTIVE_HEX
        or tuple(center_first["score_hex"]) != EXPECTED_SCORE_HEX
        or center_first["branch_hash"] != EXPECTED_BRANCH_HASH
        or not phase5._same_center(center_first, center_second)
    ):
        raise ValueError("center identity drifted before FD execution")

    records: list[dict[str, Any]] = []
    per_parameter: dict[str, list[dict[str, Any]]] = {
        name: [] for name in canonical.PARAMETER_NAMES
    }
    for index, parameter in enumerate(canonical.PARAMETER_NAMES):
        center_coordinate = float(theta[index])
        nominal_steps = _nominal_steps(args.step_mode, center_coordinate)
        base_step = nominal_steps[3]
        direction = tf.one_hot(
            index, len(canonical.PARAMETER_NAMES), dtype=tf.float64
        )
        for multiplier, nominal_step in zip(
            LADDER_MULTIPLIERS, nominal_steps, strict=True
        ):
            plus_theta = theta + tf.constant(nominal_step, tf.float64) * direction
            minus_theta = theta - tf.constant(nominal_step, tf.float64) * direction
            actual_plus = float(plus_theta[index] - theta[index])
            actual_minus = float(theta[index] - minus_theta[index])
            nominal_hex = float(nominal_step).hex()
            symmetric = (
                math.isfinite(actual_plus)
                and math.isfinite(actual_minus)
                and actual_plus > 0.0
                and actual_minus > 0.0
                and float(actual_plus).hex() == nominal_hex
                and float(actual_minus).hex() == nominal_hex
                and float(actual_plus).hex() == float(actual_minus).hex()
            )
            plus = phase5._call_record(callable_(plus_theta))
            minus = phase5._call_record(callable_(minus_theta))
            branches_match = (
                plus["branch_hash"] == center_first["branch_hash"]
                and minus["branch_hash"] == center_first["branch_hash"]
            )
            charts_valid = all(plus["valid_chart"]) and all(minus["valid_chart"])
            finite = all(
                math.isfinite(value)
                for value in (
                    plus["objective"],
                    minus["objective"],
                    *plus["score"],
                    *minus["score"],
                )
            )
            endpoint_valid = symmetric and branches_match and charts_valid and finite
            actual_step = actual_plus if symmetric else float("nan")
            derivative = (
                (plus["objective"] - minus["objective"]) / (2.0 * actual_step)
                if endpoint_valid
                else float("nan")
            )
            cancellation_floor = (
                MACHINE_EPSILON
                * (abs(plus["objective"]) + abs(minus["objective"]))
                / (2.0 * actual_step)
                if endpoint_valid
                else float("nan")
            )
            eligible = endpoint_valid and abs(derivative) > cancellation_floor
            score = center_first["score"][index]
            relative_error = (
                abs(score - derivative) / abs(derivative)
                if eligible
                else None
            )
            record = {
                "parameter": parameter,
                "parameter_index": index,
                "multiplier": multiplier,
                "base_step": base_step,
                "nominal_step": nominal_step,
                "nominal_step_hex": nominal_hex,
                "actual_plus_step": actual_plus,
                "actual_minus_step": actual_minus,
                "actual_plus_step_hex": float(actual_plus).hex(),
                "actual_minus_step_hex": float(actual_minus).hex(),
                "actual_plus_equals_nominal": float(actual_plus).hex() == nominal_hex,
                "actual_minus_equals_nominal": float(actual_minus).hex() == nominal_hex,
                "actual_steps_bitwise_symmetric": symmetric,
                "plus_objective": plus["objective"],
                "minus_objective": minus["objective"],
                "plus_branch_hash": plus["branch_hash"],
                "minus_branch_hash": minus["branch_hash"],
                "branches_match_center": branches_match,
                "charts_valid": charts_valid,
                "finite": finite,
                "endpoint_valid": endpoint_valid,
                "fd_estimate": derivative,
                "manual_score": score,
                "cancellation_floor_diagnostic": cancellation_floor,
                "relative_denominator_eligible": eligible,
                "relative_error": relative_error,
                "relative_error_threshold": FD_THRESHOLD,
                "relative_error_pass": (
                    relative_error <= FD_THRESHOLD if relative_error is not None else False
                ),
            }
            records.append(record)
            per_parameter[parameter].append(record)

    richardson = {}
    for parameter, parameter_records in per_parameter.items():
        richardson[parameter] = [
            {
                "coarse_multiplier": coarse["multiplier"],
                "fine_multiplier": fine["multiplier"],
                "consecutive_difference": (
                    abs(coarse["fd_estimate"] - fine["fd_estimate"])
                    if coarse["endpoint_valid"] and fine["endpoint_valid"]
                    else None
                ),
                "second_order_richardson_estimate": (
                    abs(coarse["fd_estimate"] - fine["fd_estimate"]) / 3.0
                    if coarse["endpoint_valid"] and fine["endpoint_valid"]
                    else None
                ),
                "classification": "EXPLANATORY_ONLY_NOT_AN_ERROR_BOUND",
            }
            for coarse, fine in zip(parameter_records, parameter_records[1:])
        ]

    heuristic_status = _classify_screen(records)
    source_paths = [
        ROOT / "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        ROOT
        / "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
        Path(__file__).resolve(),
    ]
    payload = {
        "schema_version": "bayesfilter.contract_e_phase8_fd_reclassification.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "phase": "8_fd_gate_reclassification",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formal_certificate_status": "UNSUPPORTED_ABSENT_CALLABLE_ERROR_BOUNDS",
        "heuristic_status": heuristic_status,
        "environment": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": logical_devices,
            "dtype": "float64",
            "jit_compile": True,
            "concrete_function_count": len(
                callable_._list_all_concrete_functions_for_serialization()
            ),
            "command": [sys.executable, *sys.argv],
            "output": str(output.relative_to(ROOT)),
            "wall_time_seconds": time.perf_counter() - started,
        },
        "fixture": str(FIXTURE.relative_to(ROOT)),
        "fixture_sha256": _sha256(FIXTURE),
        "baseline_certificate": str(BASELINE_CERTIFICATE.relative_to(ROOT)),
        "baseline_certificate_sha256": _sha256(BASELINE_CERTIFICATE),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "prepared_input_hashes": prepared_hashes,
        "center_first": center_first,
        "center_second": center_second,
        "fd_policy": {
            "scope": "finite_difference_only",
            "role": "heuristic_only_not_confidence_or_rigorous_error_bound",
            "base_step": "cbrt(float64_machine_epsilon)*max(1,abs(theta_j))",
            "multipliers": list(LADDER_MULTIPLIERS),
            "step_mode": args.step_mode,
            "nearest_dyadic_exponent": (
                NEAREST_DYADIC_EXPONENT
                if args.step_mode == "nearest_dyadic_cuberoot"
                else None
            ),
            "nearest_dyadic_base_hex": (
                NEAREST_DYADIC_BASE.hex()
                if args.step_mode == "nearest_dyadic_cuberoot"
                else None
            ),
            "relative_error": "abs(score-fd)/abs(fd)",
            "relative_error_threshold": FD_THRESHOLD,
            "all_35_steps_must_pass": True,
            "cancellation_floor_role": "relative_denominator_eligibility_only",
        },
        "records": records,
        "representability_preflight": representability_preflight,
        "richardson_diagnostics": richardson,
        "hard_checks": {
            "fixture_and_baseline_hashes_match": True,
            "prepared_input_hashes_match": True,
            "center_identity_matches": True,
            "one_concrete_same_value_and_score_callable": len(
                callable_._list_all_concrete_functions_for_serialization()
            )
            == 1,
            "all_35_records_present": len(records) == 35,
            "all_endpoint_pairs_valid": all(
                record["endpoint_valid"] for record in records
            ),
            "all_relative_denominators_eligible": all(
                record["relative_denominator_eligible"] for record in records
            ),
        },
        "nonclaims": [
            "the formal callable-error-bound certificate is unsupported regardless of heuristic outcome",
            "the cancellation floor is not a rigorous error bound or nonzero-derivative proof",
            "the heuristic screen does not prove the derivative",
            "not Kalman equivalence, target-shape FD, HMC, default, leaderboard, release, or integrity readiness",
        ],
    }
    _write_json_exclusive(output, payload)
    print(json.dumps({"output": str(output), "status": heuristic_status}, sort_keys=True))
    if heuristic_status == "SEVEN_STEP_FD_HEURISTIC_SCREEN_FAILED":
        raise SystemExit(3)
    if heuristic_status.startswith("FD_HEURISTIC_INCONCLUSIVE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

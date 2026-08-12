#!/usr/bin/env python3
"""Replay the exact seed-A invalid worker shard with upstream UKF status."""

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
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("exact-shard localization is an explicit CPU-only diagnostic")

import tensorflow as tf  # noqa: E402

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (  # noqa: E402
    batch_native_complexity_posterior_target,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-target-validity-recovery-plan-2026-08-06.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
DEFAULT_INPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/"
    "original-window-replay-r4/target-validity-failure-0930-attempt-0.json"
)
EXPECTED_INPUT_SHA256 = (
    "78e2e5845ea073f1f00528aeef5e37aacf2b17b9afc77ab4a2fa2275f8daefb9"
)
DEFAULT_OUTPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/"
    "exact-shard-localization-r1.json"
)
ROUNDOFF_TOLERANCE = 1.0e-14


class DiagnosticError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> Mapping[str, Any]:
    def reject(value: str) -> None:
        raise DiagnosticError(f"nonfinite JSON constant {value!r}: {path}")

    payload = json.loads(path.read_text(encoding="ascii"), parse_constant=reject)
    if not isinstance(payload, Mapping):
        raise DiagnosticError("input artifact must be a JSON object")
    return payload


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity" if value < 0.0 else "NaN"
    if hasattr(value, "numpy"):
        return json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DiagnosticError(f"refusing to overwrite diagnostic: {path}")
    path.write_text(
        json.dumps(
            json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_program(theta: tf.Tensor) -> Mapping[str, Any]:
    size = int(theta.shape[0])
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    program = tf.function(
        target.neutra_batch_log_prob_and_grad_status,
        input_signature=(tf.TensorSpec([size, 4], tf.float64),),
        jit_compile=True,
        reduce_retracing=False,
    )
    value, score, status = program(theta)
    return {
        "batch_size": size,
        "value": value,
        "score": score,
        "status": status,
        "jit_compile": True,
        "trace_count": int(program.experimental_get_tracing_count()),
    }


def run(input_path: Path, output_path: Path) -> Mapping[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_relative_to(ROOT) or not output_path.is_relative_to(ROOT):
        raise DiagnosticError("input and output must be inside the repository")
    if sha256(input_path) != EXPECTED_INPUT_SHA256:
        raise DiagnosticError("failure artifact SHA-256 mismatch")
    failure = strict_json(input_path)
    if (
        failure.get("continuation_update") != 930
        or failure.get("attempt") != 0
        or failure.get("invalid_row_indices") != [16]
    ):
        raise DiagnosticError("failure artifact identity mismatch")
    rows = failure.get("rows")
    if not isinstance(rows, list) or len(rows) != 100:
        raise DiagnosticError("failure artifact must preserve all 100 rows")
    task = rows[16].get("worker_task")
    if not isinstance(task, Mapping):
        raise DiagnosticError("invalid row has no worker task provenance")
    start = int(task["item_start"])
    stop = int(task["item_stop"])
    if (start, stop) != (16, 20):
        raise DiagnosticError("recorded worker shard is not rows 16:20")

    theta = tf.constant([row["theta"] for row in rows[start:stop]], tf.float64)
    started = time.perf_counter()
    single = run_program(theta[:1])
    shard = run_program(theta)
    wall_seconds = time.perf_counter() - started
    shard_status = shard["status"]
    invalid = {
        key: json_safe(value[0]) for key, value in shard_status.items()
    }
    if not (
        int(invalid["principal_sqrt_target_row_class_code"]) == 2
        and int(invalid["placement_classified_invalid_count"]) == 1
        and int(invalid["innovation_classified_invalid_count"]) == 0
        and int(invalid["placement_derivative_rhs_nonfinite_count"]) == 0
        and float(invalid["min_placement_eigenvalue"]) < -ROUNDOFF_TOLERANCE
    ):
        raise DiagnosticError("exact worker shard did not reproduce placement invalidity")
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_target_validity_shard_diagnostic.v1",
        "status": "EXACT_SHARD_PLACEMENT_COVARIANCE_INVALIDITY_REPRODUCED",
        "question": (
            "Which UKF principal-square-root input caused seed A update 930 row 16 "
            "to be classified invalid?"
        ),
        "input": {
            "path": input_path.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_INPUT_SHA256,
            "continuation_update": 930,
            "attempt": 0,
            "global_invalid_row": 16,
            "worker_task": task,
        },
        "configuration": {
            "q": 20,
            "dtype": "float64",
            "device": "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": True,
            "principal_sqrt_backend": "tensorflow_eigh",
            "roundoff_repair_tolerance": ROUNDOFF_TOLERANCE,
        },
        "single_row_replay": single,
        "exact_four_row_worker_shard_replay": shard,
        "classification": {
            "claimed_target": "finite valid SSL-LSTM q=20 value and analytic score",
            "quantity_computed": (
                "strict principal-square-root UKF status for the exact recorded four-row "
                "CPU/XLA worker shard"
            ),
            "verdict": "wrong relative to valid-target admission for row 16",
            "cause": "placement_covariance_materially_indefinite_under_batch4_xla",
            "batch4_min_placement_eigenvalue": invalid["min_placement_eigenvalue"],
            "roundoff_repair_lower_bound": -ROUNDOFF_TOLERANCE,
            "innovation_min_eigenvalue": invalid["min_innovation_eigenvalue"],
            "derivative_rhs_nonfinite": False,
            "input_value_score_finite": True,
            "single_row_is_valid": bool(
                single["status"]["valid_pre_regularized_score"][0].numpy()
            ),
            "batch_shape_sensitivity_is_explanatory_only": True,
            "threshold_relaxation_supported": False,
        },
        "decision": {
            "primary_criterion_status": "passed_for_localization",
            "veto_diagnostic_status": "placement_covariance_invalidity_confirmed",
            "main_uncertainty": (
                "the earliest UKF time index and raw placement covariance are not preserved"
            ),
            "next_justified_action": (
                "instrument time-of-first-placement-invalidity if a numerical-policy repair "
                "is considered; do not promote the recovered candidate"
            ),
            "not_concluded": [
                "the 1e-14 covariance policy should be relaxed",
                "NeuTra is invalid as a method",
                "the recovered transport is HMC-ready",
            ],
        },
        "run_manifest": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "command": [sys.executable, *sys.argv],
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "cpu_gpu_status": "GPU intentionally hidden with CUDA_VISIBLE_DEVICES=-1",
            "data_version": EXPECTED_INPUT_SHA256,
            "random_seeds": "N/A: deterministic exact proposal replay",
            "wall_seconds": wall_seconds,
            "output_path": output_path.relative_to(ROOT).as_posix(),
            "plan_path": PLAN.as_posix(),
            "result_path": output_path.relative_to(ROOT).as_posix(),
            "source_hashes": {
                SCRIPT.as_posix(): sha256(ROOT / SCRIPT),
                "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py": sha256(
                    ROOT
                    / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
                ),
                "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py": sha256(
                    ROOT
                    / "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py"
                ),
            },
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = run(ROOT / args.input, ROOT / args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "batch4_min_placement_eigenvalue": payload["classification"][
                    "batch4_min_placement_eigenvalue"
                ],
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

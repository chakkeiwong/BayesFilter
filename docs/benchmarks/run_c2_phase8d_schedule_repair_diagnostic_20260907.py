"""Diagnose fixed-schedule convergence for the C2 Phase 8D candidate.

This is a diagnostic reader/runner, not an accepted proposal route.  It
temporarily suppresses the adapter's host-side validity exception so the
generic kernel's row-level validity and residual diagnostics can be recorded.
Invalid rows are never evaluated as a claim-bearing APF branch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


DTYPE = tf.float64
STATE_DIM = 4
HORIZON = 20
PARTICLE_COUNT = 8192
PROPOSAL_SEED = 106392
STATIONARITY_TOLERANCE = 2.0e-10
ASCENT_TOLERANCE = 2.0e-13
PHASE_ID = "c2_exact_likelihood_laplace_phase8d_schedule_repair_v1"
PLAN_PATH = ROOT / "docs/plans/c2-phase8d-candidate-repair-20260907.md"
FIXTURE_PATH = ROOT / "docs/benchmarks/artifacts/c2_phase8_recovery_20260907/phase8d-seed424245/fresh_fixture.json"
DRIVER_PATH = Path(__file__).resolve()
ADAPTER_PATH = ROOT / "bayesfilter/highdim/c2_exact_likelihood_laplace_adapter.py"
GENERIC_PATH = ROOT / "bayesfilter/highdim/exact_likelihood_laplace_apf_tf.py"


SCHEDULE_VARIANTS = (
    {
        "config_id": "quarter_long",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 8,
    },
    {
        "config_id": "quarter_long_12",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 12,
    },
    {
        "config_id": "quarter_long_16",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 16,
    },
    {
        "config_id": "fine_tempering_12",
        "tempering_schedule": (0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
                                0.875, 1.0, 1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 12,
    },
    {
        "config_id": "quarter_long_half_tail",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0, 1.0, 1.0, 1.0) + (0.5,) * 8,
    },
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _fresh_output(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _configure_gpu() -> Mapping[str, object]:
    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    physical = tuple(tf.config.list_physical_devices("GPU"))
    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not logical:
        raise RuntimeError("schedule repair diagnostic requires a logical GPU")
    with tf.device("/GPU:0"):
        probe = tf.reduce_sum(tf.ones([32], DTYPE))
    if "GPU" not in str(probe.device).upper():
        raise RuntimeError(f"GPU placement probe ran on {probe.device}")
    return {
        "physical_devices": [str(item.name) for item in physical],
        "logical_devices": [str(item.name) for item in logical],
        "placement_probe_device": str(probe.device),
        "placement_probe_value": float(probe.numpy()),
        "memory_policy": policy,
    }


def _load_fixture(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_id") != "bayesfilter.c2_sv_frozen_fixture.v1":
        raise ValueError("unexpected C2 fixture schema")
    if int(payload["state_dimension"]) != STATE_DIM or int(payload["horizon"]) != HORIZON:
        raise ValueError("unexpected C2 fixture dimensions")
    return payload


def _finite_scalar(value: object) -> bool:
    return bool(tf.math.is_finite(tf.reshape(tf.convert_to_tensor(value, DTYPE), [])).numpy())


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _residual_summary(value: object) -> Mapping[str, object]:
    residual = tf.reshape(tf.convert_to_tensor(value, DTYPE), [-1])
    finite = tf.math.is_finite(residual)
    absolute = tf.abs(residual)
    safe_absolute = tf.where(finite, absolute, tf.zeros_like(absolute))
    invalid = tf.logical_or(~finite, absolute > tf.constant(STATIONARITY_TOLERANCE, DTYPE))
    invalid_indices = tf.reshape(tf.where(invalid), [-1])
    return {
        "finite": bool(tf.reduce_all(finite).numpy()),
        "max_abs": (float(tf.reduce_max(safe_absolute).numpy()) if int(tf.size(residual).numpy()) else None),
        "invalid_count": int(tf.reduce_sum(tf.cast(invalid, tf.int32)).numpy()),
        "invalid_indices_first": [int(x) for x in invalid_indices[:20].numpy().tolist()],
    }


def _variant_record(
    *,
    adapter: Any,
    runner: Any,
    model: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    spec: Mapping[str, object],
) -> Mapping[str, object]:
    from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import FixedLaplaceConfig

    config = FixedLaplaceConfig(
        tuple(spec["tempering_schedule"]),
        tuple(spec["step_fractions"]),
        0.0,
        STATIONARITY_TOLERANCE,
    )
    started = time.perf_counter()
    record: dict[str, object] = {
        "config_id": str(spec["config_id"]),
        "tempering_schedule": list(config.tempering_schedule),
        "step_fractions": list(config.step_fractions),
        "iteration_count": len(config.tempering_schedule),
        "particle_count": PARTICLE_COUNT,
        "seed": PROPOSAL_SEED,
    }
    try:
        compilation = adapter.compile_c2_exact_likelihood_laplace_apf_k1(
            model=model,
            observations=observations,
            theta_reference=theta,
            particle_count=PARTICLE_COUNT,
            seed=PROPOSAL_SEED,
            laplace_config=config,
            jit_compile=True,
        )
        rows = []
        all_valid = True
        all_finite = True
        maximum_residual = 0.0
        minimum_precision = math.inf
        minimum_covariance = math.inf
        minimum_improvement = math.inf
        invalid_times = []
        invalid_residual_rows = 0
        for row in compilation.proposal_diagnostics:
            residual = _residual_summary(row["relative_stationarity_residual"])
            precision = tf.reshape(tf.convert_to_tensor(row["minimum_precision_eigenvalue"], DTYPE), [-1])
            covariance = tf.reshape(tf.convert_to_tensor(row["minimum_covariance_eigenvalue"], DTYPE), [-1])
            before = tf.convert_to_tensor(row["iteration_objective_before_step"], DTYPE)
            after = tf.convert_to_tensor(row["iteration_objective_after_step"], DTYPE)
            improvement = float(tf.reduce_min(after - before).numpy())
            valid = bool(tf.convert_to_tensor(row["laplace_valid"]).numpy())
            finite = bool(tf.convert_to_tensor(row["proposal_finite"]).numpy())
            all_valid = all_valid and valid and residual["invalid_count"] == 0
            all_finite = all_finite and finite and bool(residual["finite"])
            maximum_residual = max(maximum_residual, float(residual["max_abs"] or 0.0))
            precision_minimum = float(tf.reduce_min(precision).numpy())
            covariance_minimum = float(tf.reduce_min(covariance).numpy())
            minimum_precision = min(minimum_precision, precision_minimum)
            minimum_covariance = min(minimum_covariance, covariance_minimum)
            minimum_improvement = min(minimum_improvement, improvement)
            invalid_residual_rows += int(residual["invalid_count"])
            if not valid or int(residual["invalid_count"]) > 0:
                invalid_times.append(int(row["time_index"]))
            rows.append({
                "time_index": int(row["time_index"]),
                "laplace_valid": valid,
                "proposal_finite": finite,
                "residual": residual,
                "minimum_precision_eigenvalue": _finite_or_none(precision_minimum),
                "minimum_covariance_eigenvalue": _finite_or_none(covariance_minimum),
                "minimum_objective_improvement": _finite_or_none(improvement),
            })
        eligible = bool(
            all_valid
            and all_finite
            and maximum_residual <= STATIONARITY_TOLERANCE
            and minimum_precision > 0.0
            and minimum_covariance > 0.0
            and minimum_improvement >= -ASCENT_TOLERANCE
        )
        record.update({
            "status": "PASS_RESIDUAL_SCREEN" if eligible else "CANDIDATE_FAILURE",
            "eligible_by_validity": eligible,
            "all_laplace_rows_valid": all_valid,
            "all_proposals_finite": all_finite,
            "maximum_relative_stationarity_residual": _finite_or_none(maximum_residual),
            "minimum_precision_eigenvalue": _finite_or_none(minimum_precision),
            "minimum_covariance_eigenvalue": _finite_or_none(minimum_covariance),
            "minimum_objective_improvement": _finite_or_none(minimum_improvement),
            "invalid_residual_row_count": invalid_residual_rows,
            "invalid_time_indices": invalid_times,
            "trace_count": int(compilation.manifest["laplace_kernel_trace_count"]),
            "sampler_trace_count": int(compilation.manifest["sampler_trace_count"]),
            "proposal_diagnostic_row_count": len(rows),
            "rows": rows,
        })
    except Exception as exc:
        record.update({
            "status": "CONTINUATION_VETO",
            "eligible_by_validity": False,
            "failure_class": "continuation_veto",
            "error": f"{type(exc).__name__}: {exc}",
        })
    record["wall_seconds"] = time.perf_counter() - started
    return record


def _markdown(payload: Mapping[str, object]) -> str:
    def number(value: object, digits: int = 6) -> str:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(numeric) else f"{numeric:.{digits}g}"

    lines = [
        "# C2 Phase 8D candidate-schedule repair diagnostic",
        "",
        f"Status: `{payload['status']}`",
        f"Fixture: `{payload['fixture_sha256']}`",
        f"Proposal seed: `{payload['proposal_seed']}`; particle count: `{payload['particle_count']}`",
        "",
        "No ESS or heuristic value is used to select a schedule.  Invalid rows",
        "were recorded only for diagnosis and were not accepted by the APF evaluator.",
        "",
        "| Schedule | screen | max residual | invalid rows | invalid times | min precision | min covariance | min ascent | seconds |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| `{row['config_id']}` | {row['status']} | {number(row.get('maximum_relative_stationarity_residual'))} | "
            f"{row.get('invalid_residual_row_count', 'n/a')} | {row.get('invalid_time_indices', 'n/a')} | "
            f"{number(row.get('minimum_precision_eigenvalue'))} | {number(row.get('minimum_covariance_eigenvalue'))} | "
            f"{number(row.get('minimum_objective_improvement'))} | {number(row.get('wall_seconds'), 3)} |"
        )
        if row.get("error"):
            escaped_error = str(row["error"]).replace("|", "\\|")
            lines.append(f"| failure detail | `{escaped_error}` | | | | | | | |")
    lines += [
        "",
        "A passing residual screen nominates a schedule for independent calibration and fresh-path replay only.",
        "This artifact does not establish proposal efficiency, posterior correctness, a default, or general-model validity.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    output = _fresh_output(args.output_root)
    plan_path = Path(args.plan_path)
    if not plan_path.is_absolute():
        plan_path = (ROOT / plan_path).resolve()
    fixture_path = Path(args.fixture_path)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not plan_path.is_file() or not fixture_path.is_file():
        raise FileNotFoundError("plan and fixture are required")
    (output / "plan-at-launch.md").write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    runtime = _configure_gpu()
    runner = _load_module("c2_phase8d_runner", ROOT / "docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py")
    p2 = runner._load_module("c2_phase8d_p2", ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py")
    p2._load_algorithm_modules()
    fixture = _load_fixture(fixture_path)
    model, theta, observations = runner._model_inputs(p2, fixture)
    import bayesfilter.highdim.c2_exact_likelihood_laplace_adapter as adapter

    original_guard = adapter.require_valid_laplace_result
    adapter.require_valid_laplace_result = lambda result: None
    started = time.perf_counter()
    try:
        variants = [
            _variant_record(
                adapter=adapter,
                runner=runner,
                model=model,
                theta=theta,
                observations=observations,
                spec=spec,
            )
            for spec in SCHEDULE_VARIANTS
        ]
    finally:
        adapter.require_valid_laplace_result = original_guard
    continuation_veto = any(row.get("status") == "CONTINUATION_VETO" for row in variants)
    eligible = [row["config_id"] for row in variants if bool(row.get("eligible_by_validity"))]
    payload: Mapping[str, object] = {
        "schema_version": "c2_phase8d_schedule_repair_diagnostic_v1",
        "phase": PHASE_ID,
        "status": "CONTINUATION_VETO" if continuation_veto else ("NOMINATED_FIXED_SCHEDULE" if eligible else "NO_SCHEDULE_PASSED"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "fixture": str(fixture_path.relative_to(ROOT)),
        "fixture_sha256": _sha256_file(fixture_path),
        "plan": str(plan_path.relative_to(ROOT)),
        "plan_sha256": _sha256_file(plan_path),
        "particle_count": PARTICLE_COUNT,
        "proposal_seed": PROPOSAL_SEED,
        "stationarity_tolerance": STATIONARITY_TOLERANCE,
        "ascent_tolerance": ASCENT_TOLERANCE,
        "variants": variants,
        "eligible_schedule_ids": eligible,
        "selection_rule": "validity/residual/eigenvalue/ascent only; ESS excluded",
        "continuation_veto": continuation_veto,
        "runtime": runtime,
        "sources": {
            "driver": {"path": str(DRIVER_PATH.relative_to(ROOT)), "sha256": _sha256_file(DRIVER_PATH)},
            "adapter": {"path": str(ADAPTER_PATH.relative_to(ROOT)), "sha256": _sha256_file(ADAPTER_PATH)},
            "generic_kernel": {"path": str(GENERIC_PATH.relative_to(ROOT)), "sha256": _sha256_file(GENERIC_PATH)},
        },
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "jit_compile": True,
        },
        "workspace": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_status": _git("status", "--porcelain=v1"),
        },
        "nonclaims": [
            "no accepted APF branch from invalid rows",
            "no ESS or heuristic ranking",
            "no posterior correctness, default, or general-model claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "result.md").write_text(_markdown(payload), encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps({
            "schema_version": "c2_phase8d_schedule_repair_manifest_v1",
            "phase": PHASE_ID,
            "command": " ".join(sys.argv),
            "plan_sha256": payload["plan_sha256"],
            "fixture_sha256": payload["fixture_sha256"],
            "status": payload["status"],
            "continuation_veto": continuation_veto,
            "eligible_schedule_ids": eligible,
            "runtime": runtime,
            "environment": payload["environment"],
            "workspace": payload["workspace"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if continuation_veto:
        raise RuntimeError("CONTINUATION_VETO")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fixture-path", default=str(FIXTURE_PATH))
    parser.add_argument("--plan-path", default=str(PLAN_PATH))
    return parser.parse_args()


def main() -> int:
    output = run(_parse_args())
    print(json.dumps({"output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

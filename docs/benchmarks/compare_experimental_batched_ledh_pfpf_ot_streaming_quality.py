"""Paired-seed downstream quality screen for streaming LEDH-PFPF-OT precision.

This aggregate harness runs the existing streaming precision comparison once
per paired seed.  It preserves compact per-seed/per-output drift records and
default-policy metadata assertions while leaving large output arrays in the
child artifacts.

The primary screen is an engineering sanity gate only: the production-default
FP32+TF32 arm must remain within a predeclared max-relative drift tolerance to
the paired FP64 arm for each downstream filter output.  Timing and memory are
descriptive only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PRECISION_WRAPPER = (
    ROOT / "docs/benchmarks/compare_experimental_batched_ledh_pfpf_ot_streaming_precision.py"
)
DEFAULT_ARM_ID = "fp32_tf32_enabled"
DIAGNOSTIC_ARM_ID = "fp32_tf32_disabled"
REFERENCE_ARM_ID = "fp64_reference"
REQUIRED_OUTPUTS = (
    "log_likelihood",
    "filtered_means",
    "filtered_variances",
    "ess_by_time",
)
DRIFT_FORMULA = (
    "max(abs(candidate - reference) / max(1.0, abs(reference))) per output "
    "array and paired seed"
)
EXPECTED_DEFAULT_METADATA: dict[str, Any] = {
    "precision_default_policy": "production_ledh_pfpf_ot_gpu_tf32",
    "default_execution_target": "gpu",
    "default_algorithm_target": "ledh_pfpf_ot_tf32",
    "default_target_status": "production_default_by_owner_directive",
    "default_dtype": "float32",
    "active_dtype": "float32",
    "default_tf32_mode": "enabled",
    "tf32_mode": "enabled",
    "tf32_execution_enabled": True,
}
NONCLAIMS = (
    "medium synthetic LGSSM-shaped quality screen only",
    "FP64 is a numerical comparator, not an exact posterior oracle",
    "three paired seeds do not support statistical ranking",
    "runtime and memory are descriptive only",
    "no posterior correctness claim",
    "no HMC readiness or sampler-convergence claim",
    "no speedup claim",
    "no dense Sinkhorn equivalence claim",
    "no public API readiness claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--obs-dim", type=int, default=6)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("callback", "tensor"),
        default="callback",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--col-chunk-size", type=int, default=64)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260620)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--max-relative-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.time_steps <= 0:
        raise ValueError("time-steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if args.state_dim <= 0 or args.obs_dim <= 0:
        raise ValueError("state-dim and obs-dim must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn-iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.num_seeds <= 0:
        raise ValueError("num-seeds must be positive")
    if args.seed_stride <= 0:
        raise ValueError("seed-stride must be positive")
    if args.max_relative_tolerance <= 0.0:
        raise ValueError("max-relative-tolerance must be positive")
    if args.child_timeout_seconds <= 0:
        raise ValueError("child-timeout-seconds must be positive")
    return args


def _artifact_dir(args: argparse.Namespace, output: Path) -> Path:
    if args.artifact_dir is not None:
        return Path(args.artifact_dir)
    return output.parent / f"{output.stem}-children"


def _seed_values(args: argparse.Namespace) -> list[int]:
    return [args.base_seed + i * args.seed_stride for i in range(args.num_seeds)]


def _precision_wrapper_timeout_seconds(args: argparse.Namespace) -> int:
    return max(60, args.child_timeout_seconds * 4)


def _child_command(
    args: argparse.Namespace,
    seed: int,
    json_path: Path,
    markdown_path: Path,
    artifact_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PRECISION_WRAPPER),
        "--device-scope",
        args.device_scope,
        "--device",
        args.device,
        "--expect-device-kind",
        args.expect_device_kind,
        "--batch-size",
        str(args.batch_size),
        "--time-steps",
        str(args.time_steps),
        "--num-particles",
        str(args.num_particles),
        "--state-dim",
        str(args.state_dim),
        "--obs-dim",
        str(args.obs_dim),
        "--transport-policy",
        args.transport_policy,
        "--proposal-mode",
        args.proposal_mode,
        "--sinkhorn-iterations",
        str(args.sinkhorn_iterations),
        "--sinkhorn-epsilon",
        str(args.sinkhorn_epsilon),
        "--annealed-scaling",
        str(args.annealed_scaling),
        "--annealed-convergence-threshold",
        str(args.annealed_convergence_threshold),
        "--row-chunk-size",
        str(args.row_chunk_size),
        "--col-chunk-size",
        str(args.col_chunk_size),
        "--particle-chunk-size",
        str(args.particle_chunk_size),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(seed),
        "--child-timeout-seconds",
        str(args.child_timeout_seconds),
        "--artifact-dir",
        str(artifact_dir),
        "--output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ]
    if args.cuda_visible_devices is not None:
        command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    return command


def _tail(text: str | bytes | None, limit: int = 4000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _child_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for child in payload.get("children", []):
        summary: dict[str, Any] = {
            "arm_id": child.get("arm_id"),
            "spec": child.get("spec"),
            "passed": bool(child.get("passed")),
            "returncode": child.get("returncode"),
            "reason": child.get("reason"),
            "json_path": child.get("json_path"),
            "markdown_path": child.get("markdown_path"),
            "stdout_tail": child.get("stdout_tail", ""),
            "stderr_tail": child.get("stderr_tail", ""),
        }
        benchmark = child.get("benchmark")
        if isinstance(benchmark, dict):
            summary.update(
                {
                    "precision": benchmark.get("precision"),
                    "finite_output": bool(benchmark.get("finite_output")),
                    "output_devices": benchmark.get("output_devices"),
                    "physical_gpus": benchmark.get("physical_gpus"),
                    "logical_gpus": benchmark.get("logical_gpus"),
                    "shape": benchmark.get("shape"),
                    "seed": benchmark.get("seed"),
                    "transport_policy": benchmark.get("transport_policy"),
                    "transport": benchmark.get("transport"),
                    "proposal_mode": benchmark.get("proposal_mode"),
                    "return_history": benchmark.get("return_history"),
                    "stores_full_pre_flow_particles": benchmark.get(
                        "stores_full_pre_flow_particles"
                    ),
                    "warm_call_timing_summary_seconds": benchmark.get(
                        "warm_call_timing_summary_seconds"
                    ),
                    "compile_and_first_call_seconds": benchmark.get(
                        "compile_and_first_call_seconds"
                    ),
                }
            )
        summaries.append(summary)
    return summaries


def _comparison(payload: dict[str, Any], arm_id: str) -> dict[str, Any] | None:
    for comparison in payload.get("comparisons", []):
        if comparison.get("arm_id") == arm_id:
            return comparison
    return None


def _metadata_assertions(precision: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(precision, dict):
        return {
            "passed": False,
            "reason": "precision_metadata_missing",
            "expected": EXPECTED_DEFAULT_METADATA,
            "fields": {},
        }
    fields: dict[str, dict[str, Any]] = {}
    for key, expected in EXPECTED_DEFAULT_METADATA.items():
        actual = precision.get(key)
        fields[key] = {
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        }
    return {
        "passed": all(field["passed"] for field in fields.values()),
        "expected": EXPECTED_DEFAULT_METADATA,
        "fields": fields,
    }


def _device_screen(
    child_summaries: list[dict[str, Any]],
    expect_device_kind: str,
) -> dict[str, Any]:
    devices_by_arm = {
        str(child.get("arm_id")): child.get("output_devices")
        for child in child_summaries
        if child.get("output_devices") is not None
    }
    if expect_device_kind == "any":
        return {"passed": True, "devices_by_arm": devices_by_arm}
    token = expect_device_kind.upper()
    passed = True
    for devices in devices_by_arm.values():
        if not isinstance(devices, list) or not devices:
            passed = False
            continue
        if not all(token in str(device).upper() for device in devices):
            passed = False
    return {"passed": passed, "devices_by_arm": devices_by_arm}


def _output_arrays_screen(hard_screen: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(hard_screen, dict):
        return {
            "passed": False,
            "reason": "hard_screen_missing",
            "arrays_present": {},
        }
    arrays_present = hard_screen.get("arrays_present")
    if not isinstance(arrays_present, dict):
        return {
            "passed": False,
            "reason": "output_arrays_presence_missing",
            "arrays_present": arrays_present,
        }
    return {
        "passed": all(bool(value) for value in arrays_present.values()),
        "arrays_present": arrays_present,
    }


def _tolerance_screen(
    comparison: dict[str, Any] | None,
    tolerance: float,
) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        return {
            "passed": False,
            "reason": "default_comparison_missing",
            "required_outputs": list(REQUIRED_OUTPUTS),
            "per_output": {},
            "worst": None,
        }
    drift = comparison.get("drift_vs_fp64")
    if not isinstance(drift, dict):
        return {
            "passed": False,
            "reason": "drift_vs_fp64_missing",
            "required_outputs": list(REQUIRED_OUTPUTS),
            "per_output": {},
            "worst": None,
        }
    per_output: dict[str, dict[str, Any]] = {}
    worst: dict[str, Any] | None = None
    missing: list[str] = []
    for output_name in REQUIRED_OUTPUTS:
        payload = drift.get(output_name)
        if not isinstance(payload, dict):
            missing.append(output_name)
            per_output[output_name] = {"passed": False, "reason": "missing"}
            continue
        value = payload.get("max_relative_to_max1_abs_reference")
        finite = isinstance(value, (int, float)) and math.isfinite(float(value))
        passed = finite and float(value) <= tolerance
        row = {
            "passed": bool(passed),
            "max_relative_to_max1_abs_reference": float(value) if finite else value,
            "tolerance": tolerance,
            "max_abs": payload.get("max_abs"),
            "rms_abs": payload.get("rms_abs"),
            "shape": payload.get("shape"),
        }
        per_output[output_name] = row
        if finite:
            candidate_worst = {
                "output": output_name,
                "max_relative_to_max1_abs_reference": float(value),
                "tolerance": tolerance,
            }
            if worst is None or candidate_worst[
                "max_relative_to_max1_abs_reference"
            ] > worst["max_relative_to_max1_abs_reference"]:
                worst = candidate_worst
    return {
        "passed": not missing and all(row.get("passed") for row in per_output.values()),
        "required_outputs": list(REQUIRED_OUTPUTS),
        "missing_outputs": missing,
        "per_output": per_output,
        "worst": worst,
    }


def _compact_comparisons(payload: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for comparison in payload.get("comparisons", []):
        compact.append(
            {
                "arm_id": comparison.get("arm_id"),
                "precision": comparison.get("precision"),
                "finite_output": comparison.get("finite_output"),
                "drift_vs_fp64": comparison.get("drift_vs_fp64"),
                "compile_and_first_call_seconds": comparison.get(
                    "compile_and_first_call_seconds"
                ),
                "warm_call_timing_summary_seconds": comparison.get(
                    "warm_call_timing_summary_seconds"
                ),
                "warm_median_seconds_ratio_to_fp64": comparison.get(
                    "warm_median_seconds_ratio_to_fp64"
                ),
            }
        )
    return compact


def _run_seed(
    args: argparse.Namespace,
    seed: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    seed_dir = artifact_dir / f"seed-{seed}"
    json_path = seed_dir / "streaming-precision.json"
    markdown_path = seed_dir / "streaming-precision.md"
    command = _child_command(args, seed, json_path, markdown_path, seed_dir / "children")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=_precision_wrapper_timeout_seconds(args),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "seed": seed,
            "passed": False,
            "reason": "precision_wrapper_timeout",
            "timeout_seconds": _precision_wrapper_timeout_seconds(args),
            "stdout_tail": _tail(exc.stdout),
            "stderr_tail": _tail(exc.stderr),
            "command": command,
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        }

    artifact_exists = json_path.exists()
    row: dict[str, Any] = {
        "seed": seed,
        "passed": False,
        "returncode": completed.returncode,
        "returncode_ok": completed.returncode == 0,
        "artifact_exists": artifact_exists,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "command": command,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
    if not artifact_exists:
        row["reason"] = "precision_wrapper_failed_or_missing_artifact"
        return row

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    child_summaries = _child_summaries(payload)
    default_comparison = _comparison(payload, DEFAULT_ARM_ID)
    metadata_assertions = _metadata_assertions(
        default_comparison.get("precision") if isinstance(default_comparison, dict) else None
    )
    tolerance_screen = _tolerance_screen(default_comparison, args.max_relative_tolerance)
    device_screen = _device_screen(child_summaries, args.expect_device_kind)
    output_arrays_screen = _output_arrays_screen(payload.get("hard_screen"))
    seed_matches = payload.get("seed") == seed
    row.update(
        {
            "precision_wrapper_overall_passed": bool(payload.get("overall_passed")),
            "precision_wrapper_seed": payload.get("seed"),
            "seed_matches_request": seed_matches,
            "hard_screen": payload.get("hard_screen"),
            "shape": payload.get("shape"),
            "transport_policy": payload.get("transport_policy"),
            "proposal_mode": payload.get("proposal_mode"),
            "children": child_summaries,
            "comparisons": _compact_comparisons(payload),
            "default_arm_id": DEFAULT_ARM_ID,
            "diagnostic_arm_id": DIAGNOSTIC_ARM_ID,
            "reference_arm_id": REFERENCE_ARM_ID,
            "default_metadata_assertions": metadata_assertions,
            "default_tolerance_screen": tolerance_screen,
            "device_screen": device_screen,
            "output_arrays_screen": output_arrays_screen,
        }
    )
    row["passed"] = bool(
        row["returncode_ok"]
        and row["precision_wrapper_overall_passed"]
        and seed_matches
        and metadata_assertions["passed"]
        and tolerance_screen["passed"]
        and device_screen["passed"]
        and output_arrays_screen["passed"]
    )
    if not row["passed"]:
        row["reason"] = "one_or_more_quality_screens_failed"
    return row


def _summary(rows: list[dict[str, Any]], seeds: list[int], tolerance: float) -> dict[str, Any]:
    worst_by_output: dict[str, dict[str, Any]] = {}
    worst_overall: dict[str, Any] | None = None
    for row in rows:
        per_output = (
            row.get("default_tolerance_screen", {})
            .get("per_output", {})
        )
        if not isinstance(per_output, dict):
            continue
        for output_name, payload in per_output.items():
            if not isinstance(payload, dict):
                continue
            value = payload.get("max_relative_to_max1_abs_reference")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            record = {
                "seed": row.get("seed"),
                "output": output_name,
                "max_relative_to_max1_abs_reference": float(value),
                "tolerance": tolerance,
                "passed": bool(payload.get("passed")),
            }
            if (
                output_name not in worst_by_output
                or record["max_relative_to_max1_abs_reference"]
                > worst_by_output[output_name]["max_relative_to_max1_abs_reference"]
            ):
                worst_by_output[output_name] = record
            if (
                worst_overall is None
                or record["max_relative_to_max1_abs_reference"]
                > worst_overall["max_relative_to_max1_abs_reference"]
            ):
                worst_overall = record
    return {
        "paired_seed_count": len(seeds),
        "paired_seeds": seeds,
        "expected_paired_seed_count": len(seeds),
        "all_rows_passed": all(bool(row.get("passed")) for row in rows),
        "all_precision_wrappers_passed": all(
            bool(row.get("precision_wrapper_overall_passed")) for row in rows
        ),
        "all_seed_matches_passed": all(bool(row.get("seed_matches_request")) for row in rows),
        "all_default_metadata_assertions_passed": all(
            bool(row.get("default_metadata_assertions", {}).get("passed")) for row in rows
        ),
        "all_default_tolerance_screens_passed": all(
            bool(row.get("default_tolerance_screen", {}).get("passed")) for row in rows
        ),
        "all_device_screens_passed": all(
            bool(row.get("device_screen", {}).get("passed")) for row in rows
        ),
        "all_output_arrays_screens_passed": all(
            bool(row.get("output_arrays_screen", {}).get("passed")) for row in rows
        ),
        "required_outputs": list(REQUIRED_OUTPUTS),
        "max_relative_tolerance": tolerance,
        "drift_formula": DRIFT_FORMULA,
        "worst_default_drift_by_output": worst_by_output,
        "worst_default_drift_overall": worst_overall,
    }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Streaming LEDH-PFPF-OT Paired Quality Screen",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Shape: `{result['shape']}`",
        f"- Seeds: `{summary['paired_seeds']}`",
        f"- Drift formula: `{summary['drift_formula']}`",
        f"- Max-relative tolerance: `{summary['max_relative_tolerance']}`",
        "",
        "## Seed Screens",
        "",
        "| seed | wrapper | metadata | tolerance | device | worst default drift |",
        "| ---: | --- | --- | --- | --- | ---: |",
    ]
    for row in result["rows"]:
        worst = row.get("default_tolerance_screen", {}).get("worst")
        worst_value = (
            worst.get("max_relative_to_max1_abs_reference")
            if isinstance(worst, dict)
            else "n/a"
        )
        lines.append(
            f"| {row['seed']} | {row.get('precision_wrapper_overall_passed')} | "
            f"{row.get('default_metadata_assertions', {}).get('passed')} | "
            f"{row.get('default_tolerance_screen', {}).get('passed')} | "
            f"{row.get('device_screen', {}).get('passed')} | {worst_value} |"
        )
    lines.extend(
        [
            "",
            "## Worst Default Drift By Output",
            "",
            "| output | seed | max relative | tolerance | passed |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for output_name in REQUIRED_OUTPUTS:
        record = summary["worst_default_drift_by_output"].get(output_name)
        if not isinstance(record, dict):
            lines.append(f"| {output_name} | n/a | n/a | {summary['max_relative_tolerance']} | false |")
            continue
        lines.append(
            f"| {output_name} | {record['seed']} | "
            f"{record['max_relative_to_max1_abs_reference']:.6g} | "
            f"{record['tolerance']:.6g} | {record['passed']} |"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    output = Path(args.output)
    artifact_dir = _artifact_dir(args, output)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seeds = _seed_values(args)
    rows = [_run_seed(args, seed, artifact_dir) for seed in seeds]
    summary = _summary(rows, seeds, args.max_relative_tolerance)
    overall_passed = bool(
        len(rows) == len(seeds)
        and summary["paired_seed_count"] == args.num_seeds
        and summary["all_rows_passed"]
        and set(summary["worst_default_drift_by_output"]) == set(REQUIRED_OUTPUTS)
    )
    result: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "child_script": str(PRECISION_WRAPPER),
        "artifact_dir": str(artifact_dir),
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "cuda_visible_devices_arg": args.cuda_visible_devices,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
        },
        "transport": {
            "policy": args.transport_policy,
            "proposal_mode": args.proposal_mode,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
        },
        "warmups": args.warmups,
        "repeats": args.repeats,
        "child_timeout_seconds": args.child_timeout_seconds,
        "precision_wrapper_timeout_seconds": _precision_wrapper_timeout_seconds(args),
        "seed_plan": {
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "seed_stride": args.seed_stride,
            "seeds": seeds,
        },
        "tolerance_contract": {
            "max_relative_tolerance": args.max_relative_tolerance,
            "drift_formula": DRIFT_FORMULA,
            "required_outputs": list(REQUIRED_OUTPUTS),
            "threshold_role": "gross_engineering_sanity_screen_only",
        },
        "default_arm_id": DEFAULT_ARM_ID,
        "diagnostic_arm_id": DIAGNOSTIC_ARM_ID,
        "reference_arm_id": REFERENCE_ARM_ID,
        "expected_default_metadata": EXPECTED_DEFAULT_METADATA,
        "summary": summary,
        "rows": rows,
        "overall_passed": overall_passed,
        "nonclaims": list(NONCLAIMS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

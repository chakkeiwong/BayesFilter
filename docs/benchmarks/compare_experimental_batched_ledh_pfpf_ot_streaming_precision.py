"""Compare FP64, FP32, and FP32+TF32 streaming LEDH-PFPF-OT runs.

This aggregate harness launches each precision arm in a fresh Python process so
TensorFlow dtype and TF32 execution settings cannot leak between arms.  The
comparison is a deterministic engineering screen: it checks that all arms ran
the same fixture and reports drift from the FP64 reference.  Single-run timing
and drift metrics are descriptive only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "docs/benchmarks/benchmark_experimental_batched_ledh_pfpf_ot_streaming_lgssm.py"

NONCLAIMS = (
    "single deterministic LGSSM-shaped fixture only",
    "production/default target by owner directive",
    "no posterior validity claim",
    "no HMC readiness or energy-conservation claim",
    "no statistical ranking from single-run timings",
    "TF32 flag is recorded as requested; not every TensorFlow op necessarily uses TF32",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=100)
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--obs-dim", type=int, default=10)
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
    parser.add_argument("--sinkhorn-iterations", type=int, default=4)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=512)
    parser.add_argument("--col-chunk-size", type=int, default=512)
    parser.add_argument("--particle-chunk-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
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
        raise ValueError("batch_size must be positive")
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.state_dim <= 0 or args.obs_dim <= 0:
        raise ValueError("state_dim and obs_dim must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.child_timeout_seconds <= 0:
        raise ValueError("child-timeout-seconds must be positive")
    return args


def _arm_specs() -> list[dict[str, str]]:
    return [
        {"arm_id": "fp64_reference", "dtype": "float64", "tf32_mode": "disabled"},
        {"arm_id": "fp32_tf32_disabled", "dtype": "float32", "tf32_mode": "disabled"},
        {"arm_id": "fp32_tf32_enabled", "dtype": "float32", "tf32_mode": "enabled"},
    ]


def _artifact_dir(args: argparse.Namespace, output: Path) -> Path:
    if args.artifact_dir is not None:
        return Path(args.artifact_dir)
    return output.parent / f"{output.stem}-children"


def _child_command(
    args: argparse.Namespace,
    spec: dict[str, str],
    *,
    json_path: Path,
    markdown_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK),
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
        str(args.seed),
        "--return-history",
        "--include-output-arrays",
        "--dtype",
        spec["dtype"],
        "--tf32-mode",
        spec["tf32_mode"],
        "--output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ]
    if args.cuda_visible_devices is not None:
        command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    return command


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_arm(
    args: argparse.Namespace,
    spec: dict[str, str],
    artifact_dir: Path,
) -> dict[str, Any]:
    json_path = artifact_dir / f"{spec['arm_id']}.json"
    markdown_path = artifact_dir / f"{spec['arm_id']}.md"
    command = _child_command(
        args,
        spec,
        json_path=json_path,
        markdown_path=markdown_path,
    )
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
            timeout=args.child_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "arm_id": spec["arm_id"],
            "spec": spec,
            "passed": False,
            "reason": "timeout",
            "timeout_seconds": args.child_timeout_seconds,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "command": command,
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        }
    passed = completed.returncode == 0 and json_path.exists()
    result: dict[str, Any] = {
        "arm_id": spec["arm_id"],
        "spec": spec,
        "passed": passed,
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "command": command,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
    if passed:
        result["benchmark"] = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        result["reason"] = "child_failed_or_missing_artifact"
    return result


def _config_subset(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": benchmark["shape"],
        "seed": benchmark["seed"],
        "transport_policy": benchmark["transport_policy"],
        "transport": benchmark["transport"],
        "proposal_mode": benchmark["proposal_mode"],
        "stores_full_pre_flow_particles": benchmark["stores_full_pre_flow_particles"],
        "particle_chunk_size": benchmark["particle_chunk_size"],
        "return_history": benchmark["return_history"],
    }


def _arrays(benchmark: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = benchmark.get("output_arrays")
    if not isinstance(arrays, dict):
        raise ValueError("benchmark is missing output_arrays")
    return {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}


def _drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | list[int]]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    if reference.size == 0:
        return {
            "shape": list(reference.shape),
            "max_abs": 0.0,
            "rms_abs": 0.0,
            "max_relative_to_max1_abs_reference": 0.0,
        }
    delta = candidate - reference
    abs_delta = np.abs(delta)
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(abs_delta)),
        "rms_abs": float(np.sqrt(np.mean(delta * delta))),
        "max_relative_to_max1_abs_reference": float(
            np.max(abs_delta / np.maximum(1.0, np.abs(reference)))
        ),
    }


def _warm_median(benchmark: dict[str, Any]) -> float | None:
    summary = benchmark.get("warm_call_timing_summary_seconds")
    if not isinstance(summary, dict):
        return None
    value = summary.get("median")
    if value is None:
        return None
    return float(value)


def _build_comparison(children: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    if not all(child["passed"] for child in children):
        return False, [], {"reason": "one_or_more_children_failed"}
    benchmarks = {child["arm_id"]: child["benchmark"] for child in children}
    reference = benchmarks["fp64_reference"]
    reference_config = _config_subset(reference)
    config_matches = {
        arm_id: _config_subset(benchmark) == reference_config
        for arm_id, benchmark in benchmarks.items()
    }
    arrays_present = {
        arm_id: isinstance(benchmark.get("output_arrays"), dict)
        for arm_id, benchmark in benchmarks.items()
    }
    finite_outputs = {
        arm_id: bool(benchmark.get("finite_output"))
        for arm_id, benchmark in benchmarks.items()
    }
    hard_screen_passed = (
        all(config_matches.values())
        and all(arrays_present.values())
        and all(finite_outputs.values())
    )
    if not hard_screen_passed:
        return False, [], {
            "config_matches": config_matches,
            "arrays_present": arrays_present,
            "finite_outputs": finite_outputs,
        }

    reference_arrays = _arrays(reference)
    reference_warm = _warm_median(reference)
    comparisons: list[dict[str, Any]] = []
    for arm_id, benchmark in benchmarks.items():
        if arm_id == "fp64_reference":
            continue
        candidate_arrays = _arrays(benchmark)
        drift = {
            name: _drift(reference_arrays[name], candidate_arrays[name])
            for name in sorted(reference_arrays)
        }
        candidate_warm = _warm_median(benchmark)
        timing_ratio = None
        if reference_warm and candidate_warm is not None:
            timing_ratio = candidate_warm / reference_warm
        comparisons.append(
            {
                "arm_id": arm_id,
                "precision": benchmark["precision"],
                "finite_output": benchmark["finite_output"],
                "drift_vs_fp64": drift,
                "compile_and_first_call_seconds": benchmark[
                    "compile_and_first_call_seconds"
                ],
                "warm_call_timing_summary_seconds": benchmark[
                    "warm_call_timing_summary_seconds"
                ],
                "warm_median_seconds_ratio_to_fp64": timing_ratio,
            }
        )
    hard_screen = {
        "config_matches": config_matches,
        "arrays_present": arrays_present,
        "finite_outputs": finite_outputs,
        "reference_config": reference_config,
    }
    return True, comparisons, hard_screen


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Streaming LEDH-PFPF-OT Precision Comparison",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Shape: `{result['shape']}`",
        f"- Device request: `{result['device']}`",
        "",
        "## Arms",
        "",
        "| arm | dtype | TF32 mode | TF32 enabled | finite | compile+first s | warm median s |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for child in result["children"]:
        if not child["passed"]:
            lines.append(
                f"| {child['arm_id']} | {child['spec']['dtype']} | "
                f"{child['spec']['tf32_mode']} | n/a | false | n/a | n/a |"
            )
            continue
        benchmark = child["benchmark"]
        precision = benchmark["precision"]
        warm = _warm_median(benchmark)
        lines.append(
            f"| {child['arm_id']} | {precision['dtype']} | "
            f"{precision['tf32_mode']} | {precision['tf32_execution_enabled']} | "
            f"{benchmark['finite_output']} | "
            f"{benchmark['compile_and_first_call_seconds']:.6g} | "
            f"{warm if warm is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Drift Vs FP64",
            "",
            "| arm | output | max abs | rms abs | max relative |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for comparison in result["comparisons"]:
        for name, drift in comparison["drift_vs_fp64"].items():
            lines.append(
                f"| {comparison['arm_id']} | {name} | "
                f"{drift['max_abs']:.6g} | {drift['rms_abs']:.6g} | "
                f"{drift['max_relative_to_max1_abs_reference']:.6g} |"
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

    children = [_run_arm(args, spec, artifact_dir) for spec in _arm_specs()]
    hard_passed, comparisons, hard_screen = _build_comparison(children)
    result: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
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
        "transport_policy": args.transport_policy,
        "proposal_mode": args.proposal_mode,
        "seed": args.seed,
        "artifact_dir": str(artifact_dir),
        "hard_screen": hard_screen,
        "comparisons": comparisons,
        "children": children,
        "overall_passed": hard_passed,
        "nonclaims": list(NONCLAIMS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not hard_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

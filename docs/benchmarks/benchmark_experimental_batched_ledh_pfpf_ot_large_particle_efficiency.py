"""Parent runner for large-particle LEDH-PFPF-OT efficiency diagnostics.

This wrapper orchestrates existing benchmark harnesses in fresh Python
processes. It does not implement filtering math. Its job is to preserve a
compact parent artifact with hard storage/device/precision gates and descriptive
runtime summaries.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STREAMING_HARNESS = (
    ROOT / "docs/benchmarks/benchmark_experimental_batched_ledh_pfpf_ot_streaming_lgssm.py"
)
DENSE_HARNESS = ROOT / "docs/benchmarks/benchmark_experimental_batched_ledh_pfpf_ot_lgssm_scale.py"

NONCLAIMS = (
    "parent orchestration/reporting wrapper only",
    "single synthetic LGSSM-shaped benchmark fixture",
    "runtime and memory are descriptive unless a separate uncertainty plan is used",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no dense Sinkhorn equivalence claim",
    "no public API readiness claim",
)


def _parse_csv_ints(value: str) -> list[int]:
    if not value:
        return []
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if any(item <= 1 for item in parsed):
        raise ValueError("particle counts must be greater than one")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--run-kind",
        choices=("streaming-ladder", "tf32-vs-fp32", "dense-context"),
        default="streaming-ladder",
    )
    parser.add_argument("--particle-counts", default="1000,5000,10000")
    parser.add_argument("--optional-particle-counts", default="")
    parser.add_argument("--num-particles", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=80)
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--obs-dim", type=int, default=20)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--proposal-mode", choices=("callback", "tensor"), default="callback")
    parser.add_argument("--sinkhorn-iterations", type=int, default=4)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--particle-chunk-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--child-timeout-seconds", type=int, default=3600)
    parser.add_argument("--phase-wall-time-budget-seconds", type=int, default=14400)
    parser.add_argument("--optional-max-elapsed-before-seconds", type=int, default=7200)
    parser.add_argument("--optional-max-reference-child-seconds", type=int, default=1200)
    parser.add_argument("--p04-max-reference-child-seconds", type=int, default=1800)
    parser.add_argument("--selected-physical-gpu", default=None)
    parser.add_argument("--gpu-selection-reason", default=None)
    parser.add_argument("--nvidia-smi-summary-json", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    for name in (
        "batch_size",
        "time_steps",
        "state_dim",
        "obs_dim",
        "sinkhorn_iterations",
        "row_chunk_size",
        "col_chunk_size",
        "particle_chunk_size",
        "repeats",
        "child_timeout_seconds",
        "phase_wall_time_budget_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be nonnegative")
    return args


def _artifact_dir(args: argparse.Namespace, output: Path) -> Path:
    if args.artifact_dir is not None:
        return Path(args.artifact_dir)
    return output.parent / f"{output.stem}-children"


def _tail(text: str | bytes | None, limit: int = 4000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _nvidia_smi_summary(args: argparse.Namespace) -> Any:
    if args.nvidia_smi_summary_json is None:
        return None
    try:
        return json.loads(args.nvidia_smi_summary_json)
    except json.JSONDecodeError:
        return {"raw": args.nvidia_smi_summary_json}


def _base_streaming_command(
    args: argparse.Namespace,
    *,
    num_particles: int,
    dtype: str,
    tf32_mode: str,
    json_path: Path,
    markdown_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(STREAMING_HARNESS),
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
        str(num_particles),
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
        "--dtype",
        dtype,
        "--tf32-mode",
        tf32_mode,
        "--output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ]
    if args.cuda_visible_devices is not None:
        command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    return command


def _dense_command(
    args: argparse.Namespace,
    *,
    transport_plan_mode: str,
    num_particles: int,
    json_path: Path,
    markdown_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(DENSE_HARNESS),
        "--mode",
        "compiled-value",
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
        str(num_particles),
        "--state-dim",
        str(args.state_dim),
        "--obs-dim",
        str(args.obs_dim),
        "--transport-policy",
        args.transport_policy,
        "--transport-plan-mode",
        transport_plan_mode,
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
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--dtype",
        "float32",
        "--tf32-mode",
        "enabled",
        "--output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ]
    if args.cuda_visible_devices is not None:
        command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    return command


def _run_child(
    *,
    arm_id: str,
    command: list[str],
    json_path: Path,
    markdown_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - start
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "arm_id": arm_id,
            "passed": False,
            "reason": "timeout",
            "elapsed_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
            "stdout_tail": _tail(exc.stdout),
            "stderr_tail": _tail(exc.stderr),
            "command": command,
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        }
    passed = completed.returncode == 0 and json_path.exists()
    child: dict[str, Any] = {
        "arm_id": arm_id,
        "passed": passed,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "command": command,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
    if json_path.exists():
        child["benchmark"] = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        child["reason"] = "child_failed_or_missing_artifact"
    return child


def _device_kind_ok(benchmark: dict[str, Any], expect_device_kind: str) -> bool:
    if expect_device_kind == "any":
        return True
    devices = benchmark.get("output_devices")
    if not isinstance(devices, list) or not devices:
        return False
    needle = expect_device_kind.upper()
    return all(needle in str(device).upper() for device in devices)


def _precision_ok(benchmark: dict[str, Any], *, dtype: str, tf32_mode: str) -> bool:
    precision = benchmark.get("precision")
    if not isinstance(precision, dict):
        return False
    required = {
        "precision_default_policy": "production_ledh_pfpf_ot_gpu_tf32",
        "default_dtype": "float32",
        "active_dtype": dtype,
        "default_tf32_mode": "enabled",
        "dtype": dtype,
        "tf32_mode": tf32_mode,
    }
    for key, value in required.items():
        if precision.get(key) != value:
            return False
    expected_tf32 = tf32_mode == "enabled"
    return bool(precision.get("tf32_execution_enabled")) == expected_tf32


def _streaming_gate(
    child: dict[str, Any],
    args: argparse.Namespace,
    *,
    dtype: str,
    tf32_mode: str,
) -> dict[str, Any]:
    benchmark = child.get("benchmark")
    checks: dict[str, bool] = {
        "child_passed": bool(child.get("passed")),
        "artifact_present": isinstance(benchmark, dict),
    }
    if isinstance(benchmark, dict):
        transport = benchmark.get("transport")
        checks.update(
            {
                "finite_output": bool(benchmark.get("finite_output")),
                "device_kind": _device_kind_ok(benchmark, args.expect_device_kind),
                "streaming_plan_mode": isinstance(transport, dict)
                and transport.get("plan_mode") == "streaming",
                "no_dense_transport_matrix": isinstance(transport, dict)
                and transport.get("dense_transport_matrix_materialized") is False,
                "no_full_pre_flow_storage": benchmark.get("stores_full_pre_flow_particles")
                is False,
                "no_return_history": benchmark.get("return_history") is False,
                "precision_metadata": _precision_ok(
                    benchmark,
                    dtype=dtype,
                    tf32_mode=tf32_mode,
                ),
            }
        )
    return {"passed": all(checks.values()), "checks": checks}


def _dense_context_gate(child: dict[str, Any], args: argparse.Namespace, mode: str) -> dict[str, Any]:
    benchmark = child.get("benchmark")
    checks = {
        "child_completed_with_artifact": bool(child.get("passed")) and isinstance(benchmark, dict),
    }
    if isinstance(benchmark, dict):
        checks.update(
            {
                "finite_output": bool(benchmark.get("finite_output")),
                "device_kind": _device_kind_ok(benchmark, args.expect_device_kind),
                "transport_plan_mode_recorded": benchmark.get("transport_plan_mode") == mode,
            }
        )
    return {"passed": all(checks.values()), "checks": checks}


def _warm_median(benchmark: dict[str, Any] | None) -> float | None:
    if not isinstance(benchmark, dict):
        return None
    summary = benchmark.get("warm_call_timing_summary_seconds")
    if not isinstance(summary, dict) or summary.get("median") is None:
        return None
    return float(summary["median"])


def _memory_warning_or_oom(children: list[dict[str, Any]]) -> bool:
    needles = ("OOM", "RESOURCE_EXHAUSTED", "OUT OF MEMORY")
    for child in children:
        text = f"{child.get('stderr_tail', '')}\n{child.get('stdout_tail', '')}".upper()
        if any(needle in text for needle in needles):
            return True
    return False


def _run_streaming_ladder(args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    mandatory_counts = _parse_csv_ints(args.particle_counts)
    optional_counts = _parse_csv_ints(args.optional_particle_counts)
    phase_start = time.perf_counter()
    children: list[dict[str, Any]] = []
    optional_decisions: list[dict[str, Any]] = []

    for count in mandatory_counts:
        json_path = artifact_dir / f"streaming_tf32_n{count}.json"
        markdown_path = artifact_dir / f"streaming_tf32_n{count}.md"
        child = _run_child(
            arm_id=f"streaming_tf32_n{count}",
            command=_base_streaming_command(
                args,
                num_particles=count,
                dtype="float32",
                tf32_mode="enabled",
                json_path=json_path,
                markdown_path=markdown_path,
            ),
            json_path=json_path,
            markdown_path=markdown_path,
            timeout_seconds=args.child_timeout_seconds,
        )
        child["particle_count"] = count
        child["required"] = True
        child["hard_gate"] = _streaming_gate(child, args, dtype="float32", tf32_mode="enabled")
        children.append(child)

    mandatory_passed = all(child["hard_gate"]["passed"] for child in children if child["required"])
    for count in optional_counts:
        elapsed = time.perf_counter() - phase_start
        reference_elapsed = children[-1]["elapsed_seconds"] if children else None
        reasons: list[str] = []
        if not mandatory_passed:
            reasons.append("mandatory_rung_failed")
        if elapsed > args.optional_max_elapsed_before_seconds:
            reasons.append("phase_elapsed_budget_exceeded")
        if reference_elapsed is None or reference_elapsed > args.optional_max_reference_child_seconds:
            reasons.append("reference_child_elapsed_budget_exceeded")
        if _memory_warning_or_oom(children):
            reasons.append("memory_warning_or_oom_observed")
        if reasons:
            optional_decisions.append(
                {
                    "particle_count": count,
                    "decision": "skipped",
                    "reasons": reasons,
                }
            )
            continue
        json_path = artifact_dir / f"streaming_tf32_n{count}.json"
        markdown_path = artifact_dir / f"streaming_tf32_n{count}.md"
        child = _run_child(
            arm_id=f"streaming_tf32_n{count}",
            command=_base_streaming_command(
                args,
                num_particles=count,
                dtype="float32",
                tf32_mode="enabled",
                json_path=json_path,
                markdown_path=markdown_path,
            ),
            json_path=json_path,
            markdown_path=markdown_path,
            timeout_seconds=args.child_timeout_seconds,
        )
        child["particle_count"] = count
        child["required"] = False
        child["hard_gate"] = _streaming_gate(child, args, dtype="float32", tf32_mode="enabled")
        children.append(child)
        optional_decisions.append(
            {
                "particle_count": count,
                "decision": "attempted",
                "passed": child["hard_gate"]["passed"],
            }
        )

    return {
        "children": children,
        "optional_decisions": optional_decisions,
        "mandatory_particle_counts": mandatory_counts,
        "optional_particle_counts": optional_counts,
        "mandatory_passed": mandatory_passed,
        "overall_passed": mandatory_passed,
    }


def _run_tf32_vs_fp32(args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    specs = [
        ("fp32_tf32_enabled", "float32", "enabled"),
        ("fp32_tf32_disabled", "float32", "disabled"),
    ]
    children: list[dict[str, Any]] = []
    for arm_id, dtype, tf32_mode in specs:
        json_path = artifact_dir / f"{arm_id}.json"
        markdown_path = artifact_dir / f"{arm_id}.md"
        child = _run_child(
            arm_id=arm_id,
            command=_base_streaming_command(
                args,
                num_particles=args.num_particles,
                dtype=dtype,
                tf32_mode=tf32_mode,
                json_path=json_path,
                markdown_path=markdown_path,
            ),
            json_path=json_path,
            markdown_path=markdown_path,
            timeout_seconds=args.child_timeout_seconds,
        )
        child["hard_gate"] = _streaming_gate(child, args, dtype=dtype, tf32_mode=tf32_mode)
        children.append(child)
    by_id = {child["arm_id"]: child for child in children}
    enabled = by_id["fp32_tf32_enabled"].get("benchmark")
    disabled = by_id["fp32_tf32_disabled"].get("benchmark")
    enabled_median = _warm_median(enabled if isinstance(enabled, dict) else None)
    disabled_median = _warm_median(disabled if isinstance(disabled, dict) else None)
    ratio = None
    if enabled_median is not None and disabled_median not in (None, 0.0):
        ratio = enabled_median / disabled_median
    config_match = False
    if isinstance(enabled, dict) and isinstance(disabled, dict):
        config_match = {
            key: enabled.get(key)
            for key in ("shape", "transport_policy", "proposal_mode", "transport")
        } == {
            key: disabled.get(key)
            for key in ("shape", "transport_policy", "proposal_mode", "transport")
        }
    all_gates_passed = all(child["hard_gate"]["passed"] for child in children)
    return {
        "children": children,
        "comparison": {
            "enabled_warm_median_seconds": enabled_median,
            "disabled_warm_median_seconds": disabled_median,
            "enabled_to_disabled_warm_median_ratio": ratio,
            "timing_interpretation": "descriptive_only",
            "matched_config_except_tf32": config_match,
        },
        "overall_passed": all_gates_passed and config_match,
    }


def _run_dense_context(args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    dense_json = artifact_dir / "dense_context.json"
    dense_md = artifact_dir / "dense_context.md"
    dense_child = _run_child(
        arm_id="dense_context",
        command=_dense_command(
            args,
            transport_plan_mode="dense",
            num_particles=args.num_particles,
            json_path=dense_json,
            markdown_path=dense_md,
        ),
        json_path=dense_json,
        markdown_path=dense_md,
        timeout_seconds=args.child_timeout_seconds,
    )
    dense_child["context_gate"] = _dense_context_gate(dense_child, args, "dense")
    children.append(dense_child)

    streaming_json = artifact_dir / "streaming_context.json"
    streaming_md = artifact_dir / "streaming_context.md"
    streaming_child = _run_child(
        arm_id="streaming_context",
        command=_base_streaming_command(
            args,
            num_particles=args.num_particles,
            dtype="float32",
            tf32_mode="enabled",
            json_path=streaming_json,
            markdown_path=streaming_md,
        ),
        json_path=streaming_json,
        markdown_path=streaming_md,
        timeout_seconds=args.child_timeout_seconds,
    )
    streaming_child["hard_gate"] = _streaming_gate(
        streaming_child,
        args,
        dtype="float32",
        tf32_mode="enabled",
    )
    children.append(streaming_child)
    return {
        "children": children,
        "dense_context_role": "small_n_context_only_non_promotional",
        "dense_artifact_surface": (
            "older dense harness records transport_plan_mode, timing, device, "
            "and finite output; it does not emit streaming-style explicit "
            "dense-matrix or full-storage flags"
        ),
        "overall_passed": True,
    }


def _shape(args: argparse.Namespace, num_particles: int | None = None) -> dict[str, int]:
    return {
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "num_particles": args.num_particles if num_particles is None else num_particles,
        "state_dim": args.state_dim,
        "obs_dim": args.obs_dim,
    }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Large-Particle LEDH-PFPF-OT Efficiency Parent Result",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Run kind: `{result['run_kind']}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Device request: `{result['device']}`",
        f"- CUDA visible devices: `{result['cuda_visible_devices_arg']}`",
        f"- Selected physical GPU: `{result['gpu_selection']['selected_physical_gpu']}`",
        f"- GPU selection reason: `{result['gpu_selection']['reason']}`",
        "",
        "## Children",
        "",
        "| arm | passed | elapsed s | hard/context gate | artifact |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for child in result["children"]:
        gate = child.get("hard_gate", child.get("context_gate", {}))
        lines.append(
            f"| {child['arm_id']} | {child.get('passed')} | "
            f"{child.get('elapsed_seconds', 0.0):.6g} | "
            f"{gate.get('passed', 'n/a')} | `{child.get('json_path')}` |"
        )
    if result.get("comparison") is not None:
        lines.extend(["", "## Comparison", ""])
        for key, value in result["comparison"].items():
            lines.append(f"- {key}: `{value}`")
    if result.get("optional_decisions"):
        lines.extend(["", "## Optional Decisions", ""])
        for decision in result["optional_decisions"]:
            lines.append(f"- `{decision}`")
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    output = Path(args.output)
    artifact_dir = _artifact_dir(args, output)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if args.run_kind == "streaming-ladder":
        run_result = _run_streaming_ladder(args, artifact_dir)
    elif args.run_kind == "tf32-vs-fp32":
        run_result = _run_tf32_vs_fp32(args, artifact_dir)
    else:
        run_result = _run_dense_context(args, artifact_dir)
    elapsed = time.perf_counter() - start

    result: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "run_kind": args.run_kind,
        "shape": _shape(args),
        "transport_policy": args.transport_policy,
        "proposal_mode": args.proposal_mode,
        "seed": args.seed,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "cuda_visible_devices_arg": args.cuda_visible_devices,
        "gpu_selection": {
            "selected_physical_gpu": args.selected_physical_gpu,
            "reason": args.gpu_selection_reason,
            "nvidia_smi_summary": _nvidia_smi_summary(args),
            "child_gpu_mapping_contract": (
                "child CUDA_VISIBLE_DEVICES remaps selected physical GPU to logical /GPU:0"
            ),
        },
        "artifact_dir": str(artifact_dir),
        "phase_elapsed_seconds": elapsed,
        "phase_wall_time_budget_seconds": args.phase_wall_time_budget_seconds,
        "child_timeout_seconds": args.child_timeout_seconds,
        "nonclaims": list(NONCLAIMS),
        **run_result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_path, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

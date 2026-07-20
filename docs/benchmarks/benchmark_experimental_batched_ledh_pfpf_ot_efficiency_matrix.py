"""Orchestrate correctness-first LEDH-PFPF-OT timing diagnostics.

This script runs the standalone correctness gate before launching a matrix of
small benchmark commands.  Timing and memory diagnostics are descriptive only;
the child JSON artifacts are the authoritative records.
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


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = ROOT / "docs/benchmarks/benchmark_experimental_batched_ledh_pfpf_ot_cpu_gpu.py"
CORRECTNESS_SCRIPT = ROOT / "docs/benchmarks/check_experimental_batched_ledh_pfpf_ot_correctness.py"

NONCLAIMS = (
    "timing and memory diagnostics are descriptive only",
    "no GPU superiority claim",
    "production/default target by owner directive",
    "no posterior validity claim",
    "no active-transport score finite-difference claim",
    "no HMC/NeuTra readiness claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="Shape spec like B=20,T=3,N=4. May be repeated.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=(
            "compiled-value",
            "compiled-value-score",
            "scalar-compiled-value-loop",
            "parity",
        ),
        default=None,
    )
    parser.add_argument(
        "--transport-policy",
        action="append",
        choices=("active", "no-resampling"),
        default=None,
    )
    parser.add_argument(
        "--transport-gradient-mode",
        choices=("filterflow_clipped", "filterflow_custom_op", "raw"),
        default="raw",
    )
    parser.add_argument(
        "--transport-plan-mode",
        choices=("dense", "streaming"),
        default="dense",
    )
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--expect-device-kind",
        choices=("any", "cpu", "gpu"),
        default="cpu",
    )
    parser.add_argument("--skip-correctness-preflight", action="store_true")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.transport_plan_mode == "streaming" and args.transport_gradient_mode != "raw":
        raise ValueError("streaming transport requires --transport-gradient-mode raw")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.shape is None:
        args.shape = ["B=2,T=3,N=4", "B=20,T=3,N=4"]
    if args.mode is None:
        args.mode = ["compiled-value"]
    if args.transport_policy is None:
        args.transport_policy = ["no-resampling", "active"]
    return args


def _parse_shape(spec: str) -> dict[str, int]:
    pieces: dict[str, int] = {}
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"shape spec part must be KEY=VALUE, got {part!r}")
        key, value = part.split("=", 1)
        pieces[key.strip().upper()] = int(value)
    required = {"B", "T", "N"}
    missing = required.difference(pieces)
    if missing:
        raise ValueError(f"shape spec {spec!r} missing keys {sorted(missing)}")
    for key in required:
        if pieces[key] <= 0:
            raise ValueError(f"shape key {key} must be positive")
    if pieces["N"] <= 1:
        raise ValueError("N must be greater than one")
    return {
        "batch_size": pieces["B"],
        "time_steps": pieces["T"],
        "num_particles": pieces["N"],
    }


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _run_child(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _base_child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    elif args.device_scope == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def _artifact_stem(
    *,
    mode: str,
    policy: str,
    plan_mode: str,
    shape: dict[str, int],
) -> str:
    return (
        "experimental-ledh-pfpf-ot-efficiency-"
        f"{mode}-{policy}-{plan_mode}-b{shape['batch_size']}-t{shape['time_steps']}"
        f"-n{shape['num_particles']}"
    )


def _summarize_child(path: Path, *, command_result: dict[str, Any]) -> dict[str, Any]:
    if command_result["returncode"] != 0:
        return {
            "artifact": str(path),
            "passed": False,
            "returncode": command_result["returncode"],
            "stderr_tail": command_result["stderr_tail"],
            "stdout_tail": command_result["stdout_tail"],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("warm_call_summary") or data.get("batched_warm_call_summary") or {}
    compiler = data.get("compiler") or {}
    return {
        "artifact": str(path),
        "passed": True,
        "mode": data.get("mode"),
        "shape": data.get("shape"),
        "transport": data.get("transport"),
        "finite_outputs": data.get("finite_outputs"),
        "output_devices": data.get("output_devices"),
        "compile_and_first_call_seconds": compiler.get("compile_and_first_call_seconds"),
        "warm_call_summary": summary,
        "gpu_memory_info_after": data.get("gpu_memory_info_after"),
    }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Experimental Batched LEDH-PFPF-OT Efficiency Matrix",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Correctness preflight: `{result['correctness_preflight'].get('passed')}`",
        "",
        "## Child Runs",
        "",
        "| Mode | Transport | Shape | Passed | Compile+first | Warm summary | Artifact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for child in result["children"]:
        transport = child.get("transport")
        shape = child.get("shape")
        lines.append(
            f"| `{child.get('mode')}` | `{transport}` | `{shape}` | `{child.get('passed')}` | "
            f"`{child.get('compile_and_first_call_seconds')}` | "
            f"`{child.get('warm_call_summary')}` | `{child.get('artifact')}` |"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    output = Path(args.output)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else output.parent / "ledh_pfpf_ot_efficiency_children"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    env = _base_child_env(args)

    correctness_preflight: dict[str, Any] = {"skipped": True, "passed": True}
    if not args.skip_correctness_preflight:
        correctness_output = artifact_dir / "experimental-ledh-pfpf-ot-correctness-preflight.json"
        command = [
            args.python,
            str(CORRECTNESS_SCRIPT),
            "--device-scope",
            args.device_scope,
            "--device",
            args.device,
            "--expect-device-kind",
            args.expect_device_kind,
            "--batch-size",
            "2",
            "--time-steps",
            "3",
            "--num-particles",
            "4",
            "--transport-policy",
            "active",
            "--transport-gradient-mode",
            args.transport_gradient_mode,
            "--transport-plan-mode",
            args.transport_plan_mode,
            "--row-chunk-size",
            str(args.row_chunk_size),
            "--col-chunk-size",
            str(args.col_chunk_size),
            "--sinkhorn-iterations",
            str(args.sinkhorn_iterations),
            "--output",
            str(correctness_output),
        ]
        if args.cuda_visible_devices is not None:
            command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
        command_result = _run_child(command, env=env)
        correctness_preflight = {
            "artifact": str(correctness_output),
            "passed": command_result["returncode"] == 0,
            "returncode": command_result["returncode"],
            "stderr_tail": command_result["stderr_tail"],
            "stdout_tail": command_result["stdout_tail"],
        }
        if correctness_output.exists():
            data = json.loads(correctness_output.read_text(encoding="utf-8"))
            correctness_preflight["overall_passed"] = data.get("overall_passed")
            correctness_preflight["source_diagnostics"] = data.get("source_diagnostics")

    children: list[dict[str, Any]] = []
    if correctness_preflight.get("passed"):
        for shape_spec in args.shape:
            shape = _parse_shape(shape_spec)
            for policy in args.transport_policy:
                for mode in args.mode:
                    stem = _artifact_stem(
                        mode=mode,
                        policy=policy,
                        plan_mode=args.transport_plan_mode,
                        shape=shape,
                    )
                    child_output = artifact_dir / f"{stem}.json"
                    command = [
                        args.python,
                        str(BENCHMARK_SCRIPT),
                        "--mode",
                        mode,
                        "--batch-size",
                        str(shape["batch_size"]),
                        "--time-steps",
                        str(shape["time_steps"]),
                        "--num-particles",
                        str(shape["num_particles"]),
                        "--state-dim",
                        "1",
                        "--obs-dim",
                        "1",
                        "--parameter-dim",
                        "3",
                        "--transport-policy",
                        policy,
                        "--transport-gradient-mode",
                        args.transport_gradient_mode,
                        "--transport-plan-mode",
                        args.transport_plan_mode,
                        "--row-chunk-size",
                        str(args.row_chunk_size),
                        "--col-chunk-size",
                        str(args.col_chunk_size),
                        "--sinkhorn-iterations",
                        str(args.sinkhorn_iterations),
                        "--warmups",
                        str(args.warmups),
                        "--repeats",
                        str(args.repeats),
                        "--device",
                        args.device,
                        "--device-scope",
                        args.device_scope,
                        "--expect-device-kind",
                        args.expect_device_kind,
                        "--output",
                        str(child_output),
                    ]
                    if args.cuda_visible_devices is not None:
                        command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
                    command_result = _run_child(command, env=env)
                    children.append(
                        _summarize_child(child_output, command_result=command_result)
                    )

    overall_passed = bool(
        correctness_preflight.get("passed")
        and children
        and all(child.get("passed") for child in children)
        and all(child.get("finite_outputs", True) for child in children)
    )
    result = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "device": args.device,
        "device_scope": args.device_scope,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "expect_device_kind": args.expect_device_kind,
        "correctness_preflight": correctness_preflight,
        "children": children,
        "overall_passed": overall_passed,
        "evidence_contract": {
            "question": "What fixed-branch shapes remain correct, finite, and descriptively timed?",
            "baseline": "correctness preflight plus child benchmark artifacts",
            "primary_criterion": "correctness preflight passes before timing is interpreted",
            "vetoes": [
                "failed correctness preflight",
                "failed child command",
                "non-finite child output",
                "wrong requested device placement in child artifact",
            ],
            "explanatory_only": [
                "compile-plus-first-call seconds",
                "warm-call timing",
                "GPU memory readings",
            ],
            "must_not_conclude": list(NONCLAIMS),
        },
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

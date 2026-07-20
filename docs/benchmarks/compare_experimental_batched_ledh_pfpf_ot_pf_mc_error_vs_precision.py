"""Compare precision drift to PF seed-to-seed value/score variability."""

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
CHILD = ROOT / "docs/benchmarks/compare_experimental_batched_ledh_pfpf_ot_gradient_structure.py"
TARGET_ARM = "streaming_streaming_tensor"

NONCLAIMS = (
    "focused PF MC noise-floor diagnostic only",
    "small seed count is descriptive, not a precise uncertainty interval",
    "no HMC readiness claim",
    "no production default precision change",
    "seed-to-seed fixture variability is a practical PF MC proxy here",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260615)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=5)
    parser.add_argument("--num-particles", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-odd",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=16)
    parser.add_argument("--col-chunk-size", type=int, default=16)
    parser.add_argument("--particle-chunk-size", type=int, default=16)
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--child-jit-compile", action="store_true")
    parser.add_argument("--child-timeout-seconds", type=int, default=300)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.num_seeds < 2:
        raise ValueError("num-seeds must be at least two")
    if args.seed_stride <= 0:
        raise ValueError("seed-stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.time_steps <= 0:
        raise ValueError("time-steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if args.state_dim <= 0 or args.obs_dim <= 0:
        raise ValueError("state-dim and obs-dim must be positive")
    if args.child_timeout_seconds <= 0:
        raise ValueError("child-timeout-seconds must be positive")
    return args


def _artifact_dir(args: argparse.Namespace, output: Path) -> Path:
    if args.artifact_dir is not None:
        return Path(args.artifact_dir)
    return output.parent / f"{output.stem}-children"


def _precision_specs() -> list[dict[str, str]]:
    return [
        {"precision_id": "fp64", "dtype": "float64", "tf32_mode": "disabled"},
        {"precision_id": "fp32_no_tf32", "dtype": "float32", "tf32_mode": "disabled"},
        {"precision_id": "fp32_tf32", "dtype": "float32", "tf32_mode": "enabled"},
    ]


def _seed_values(args: argparse.Namespace) -> list[int]:
    return [args.base_seed + i * args.seed_stride for i in range(args.num_seeds)]


def _child_command(
    args: argparse.Namespace,
    spec: dict[str, str],
    seed: int,
    json_path: Path,
    markdown_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(CHILD),
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
        "0",
        "--repeats",
        "1",
        "--seed",
        str(seed),
        "--dtype",
        spec["dtype"],
        "--tf32-mode",
        spec["tf32_mode"],
        "--output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ]
    if not args.child_jit_compile:
        command.append("--no-jit-compile")
    if args.cuda_visible_devices is not None:
        command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    return command


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_child(
    args: argparse.Namespace,
    spec: dict[str, str],
    seed: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    stem = f"seed-{seed}-{spec['precision_id']}"
    json_path = artifact_dir / f"{stem}.json"
    markdown_path = artifact_dir / f"{stem}.md"
    command = _child_command(args, spec, seed, json_path, markdown_path)
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
            "seed": seed,
            "precision_id": spec["precision_id"],
            "passed": False,
            "reason": "timeout",
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "command": command,
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        }
    artifact_exists = json_path.exists()
    row: dict[str, Any] = {
        "seed": seed,
        "precision_id": spec["precision_id"],
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
        row["reason"] = "child_failed_or_missing_artifact"
        return row
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    arm = next(arm for arm in payload["arms"] if arm["arm_id"] == TARGET_ARM)
    row["finite"] = bool(arm["finite"])
    row["passed"] = bool(row["finite"])
    row["value"] = arm["value"]
    row["score"] = arm["score"]
    row["child_overall_passed"] = bool(payload["overall_passed"])
    if completed.returncode != 0:
        row["reason"] = "child_returncode_nonzero_but_artifact_parsed"
    row["payload_summary"] = {
        "overall_passed": payload["overall_passed"],
        "precision": payload["precision"],
        "shape": payload["shape"],
        "transport": payload["transport"],
    }
    return row


def _sample_sd(values: np.ndarray) -> np.ndarray:
    return np.std(values, axis=0, ddof=1)


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values * values, axis=0))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, np.finfo(np.float64).tiny)


def _summary(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    if not all(row["passed"] and row.get("finite", False) for row in rows):
        return False, {"reason": "one_or_more_children_failed_or_nonfinite"}
    by_precision: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_precision.setdefault(row["precision_id"], []).append(row)
    required = {"fp64", "fp32_no_tf32", "fp32_tf32"}
    if set(by_precision) != required:
        return False, {"reason": "missing_precision_arm", "present": sorted(by_precision)}
    seeds_by_precision = {
        key: [row["seed"] for row in value]
        for key, value in by_precision.items()
    }
    if len({tuple(v) for v in seeds_by_precision.values()}) != 1:
        return False, {"reason": "seed_sets_do_not_match", "seeds": seeds_by_precision}

    fp64_value = np.asarray([row["value"] for row in by_precision["fp64"]], dtype=np.float64)
    fp64_score = np.asarray([row["score"] for row in by_precision["fp64"]], dtype=np.float64)
    fp64_value_sd = _sample_sd(fp64_value)
    fp64_score_sd = _sample_sd(fp64_score)
    payload: dict[str, Any] = {
        "num_seeds": len(by_precision["fp64"]),
        "seeds": seeds_by_precision["fp64"],
        "fp64_value_mean": np.mean(fp64_value, axis=0).tolist(),
        "fp64_value_sample_sd": fp64_value_sd.tolist(),
        "fp64_score_mean": np.mean(fp64_score, axis=0).tolist(),
        "fp64_score_sample_sd": fp64_score_sd.tolist(),
        "precision_vs_fp64": {},
    }
    for precision_id in ("fp32_no_tf32", "fp32_tf32"):
        value = np.asarray([row["value"] for row in by_precision[precision_id]], dtype=np.float64)
        score = np.asarray([row["score"] for row in by_precision[precision_id]], dtype=np.float64)
        value_delta = value - fp64_value
        score_delta = score - fp64_score
        value_rms = _rms(value_delta)
        score_rms = _rms(score_delta)
        value_ratio = _safe_ratio(value_rms, fp64_value_sd)
        score_ratio = _safe_ratio(score_rms, fp64_score_sd)
        payload["precision_vs_fp64"][precision_id] = {
            "value_rms_error": value_rms.tolist(),
            "value_max_abs_error": np.max(np.abs(value_delta), axis=0).tolist(),
            "value_rms_error_over_fp64_sample_sd": value_ratio.tolist(),
            "score_rms_error": score_rms.tolist(),
            "score_max_abs_error": np.max(np.abs(score_delta), axis=0).tolist(),
            "score_rms_error_over_fp64_sample_sd": score_ratio.tolist(),
            "max_value_ratio": float(np.max(value_ratio)),
            "max_score_ratio": float(np.max(score_ratio)),
        }
    return True, payload


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# LEDH-PFPF-OT PF MC Error vs Precision",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Shape: `{result['shape']}`",
        f"- Seeds: `{result['summary'].get('seeds')}`",
        "",
        "## FP64 MC Noise",
        "",
        f"- Value sample SD: `{result['summary'].get('fp64_value_sample_sd')}`",
        f"- Score sample SD: `{result['summary'].get('fp64_score_sample_sd')}`",
        "",
        "## Precision Ratios",
        "",
        "| precision | value RMS / FP64 SD | max score RMS / FP64 SD |",
        "| --- | ---: | ---: |",
    ]
    precision = result["summary"].get("precision_vs_fp64", {})
    for precision_id, payload in precision.items():
        lines.append(
            f"| {precision_id} | {payload['value_rms_error_over_fp64_sample_sd']} | "
            f"{payload['max_score_ratio']:.6g} |"
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
    rows: list[dict[str, Any]] = []
    for seed in _seed_values(args):
        for spec in _precision_specs():
            rows.append(_run_child(args, spec, seed, artifact_dir))
    passed, summary = _summary(rows)
    result: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
            "parameter_dim": 3,
        },
        "transport": {
            "policy": args.transport_policy,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
        },
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "cuda_visible_devices_arg": args.cuda_visible_devices,
        "child_jit_compile": bool(args.child_jit_compile),
        "target_arm": TARGET_ARM,
        "artifact_dir": str(artifact_dir),
        "rows": rows,
        "summary": summary,
        "overall_passed": passed,
        "nonclaims": list(NONCLAIMS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Grid orchestrator for actual-SIR low-rank tuning rows.

This script is an artifact orchestrator only. It enumerates calls to the owned
actual-SIR low-rank validation harness, optionally runs them as subprocesses,
and aggregates row JSON into candidate labels. It does not implement filtering
or transport logic and does not change BayesFilter defaults.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-low-rank-default-certification-"
    "master-program-2026-06-24.md"
)
DEFAULT_HARNESS_PATH = "docs/benchmarks/benchmark_actual_sir_low_rank_route_validation.py"
SCHEMA_VERSION = "actual_sir_low_rank_tuning_grid.v1"
MAX_ARTIFACT_FILENAME_COMPONENT_LENGTH = 255
ROW_STEM_DIGEST_LENGTH = 16
NONCLAIMS = (
    "actual-SIR low-rank tuning orchestration only",
    "tuning rows nominate candidates only",
    "no held-out support claim",
    "no speedup claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no package-level default readiness claim",
    "no dense Sinkhorn equivalence claim",
    "no broad scalable-OT selection claim",
    "no statistical ranking claim",
)


def _parse_int_csv(value: str) -> list[int]:
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    if not entries:
        raise ValueError("expected at least one integer")
    return [int(item) for item in entries]


def _parse_float_csv(value: str) -> list[float]:
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    if not entries:
        raise ValueError("expected at least one float")
    return [float(item) for item in entries]


def _parse_str_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    if not entries:
        raise ValueError("expected at least one candidate id")
    return entries


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("dry-run", "execute", "aggregate-existing"), default="dry-run")
    parser.add_argument("--harness-path", default=DEFAULT_HARNESS_PATH)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--route", choices=("streaming", "low_rank", "both"), default="both")
    parser.add_argument("--batch-seeds", default="81120")
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--transport-policy", choices=("active-all", "active-odd", "no-resampling"), default="active-all")
    parser.add_argument("--low-rank-ranks", default="16")
    parser.add_argument("--low-rank-assignment-epsilons", default="0.25")
    parser.add_argument("--low-rank-alphas", default="1e-8")
    parser.add_argument("--low-rank-max-projection-iterations-list", default="120")
    parser.add_argument(
        "--candidate-ids",
        default=None,
        help="Optional comma-separated exact candidate ids to execute after grid enumeration.",
    )
    parser.add_argument("--low-rank-convergence-threshold", type=float, default=1.0e-6)
    parser.add_argument("--low-rank-denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--col-chunk-size", type=int, default=128)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--streaming-timing-source", choices=("compiled_core",), default="compiled_core")
    parser.add_argument("--low-rank-timing-source", choices=("compiled_core",), default="compiled_core")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--phase-id-prefix", default="ACTUAL-SIR-LR-TUNING")
    parser.add_argument("--row-timeout-seconds", type=int, default=900)
    parser.add_argument("--log-dir", default="docs/benchmarks/logs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.low_rank_ranks = _parse_int_csv(args.low_rank_ranks)
    args.low_rank_assignment_epsilons = _parse_float_csv(args.low_rank_assignment_epsilons)
    args.low_rank_alphas = _parse_float_csv(args.low_rank_alphas)
    args.low_rank_max_projection_iterations_list = _parse_int_csv(args.low_rank_max_projection_iterations_list)
    args.candidate_ids = _parse_str_csv(args.candidate_ids)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if args.time_steps <= 0:
        raise ValueError("time-steps must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn-iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_timeout_seconds <= 0:
        raise ValueError("row-timeout-seconds must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")
    if args.streaming_timing_source != "compiled_core":
        raise ValueError("streaming-timing-source must be compiled_core")
    if args.low_rank_timing_source != "compiled_core":
        raise ValueError("low-rank-timing-source must be compiled_core")
    if args.jit_compile is not True:
        raise ValueError("jit-compile must be true")
    for rank in args.low_rank_ranks:
        if rank <= 0:
            raise ValueError("low-rank-ranks must be positive")
    for epsilon in args.low_rank_assignment_epsilons:
        if epsilon <= 0.0:
            raise ValueError("low-rank-assignment-epsilons must be positive")
    for alpha in args.low_rank_alphas:
        if alpha <= 0.0:
            raise ValueError("low-rank-alphas must be positive")
    for iterations in args.low_rank_max_projection_iterations_list:
        if iterations <= 0:
            raise ValueError("low-rank max projection iterations must be positive")


def _safe_value(value: Any) -> str:
    text = str(value)
    for source, target in ((".", "p"), ("-", "m"), ("+", "p"), (",", "_")):
        text = text.replace(source, target)
    return text


def _row_stem(args: argparse.Namespace, candidate: dict[str, Any]) -> str:
    seed_token = _safe_value(args.batch_seeds)
    cuda_token = _safe_value(args.cuda_visible_devices if args.cuda_visible_devices is not None else "default")
    return (
        f"{Path(args.output).stem}-"
        f"b{candidate['batch_size']}-t{candidate['time_steps']}-n{candidate['num_particles']}-"
        f"r{candidate['low_rank_rank']}-eps{_safe_value(candidate['low_rank_assignment_epsilon'])}-"
        f"a{_safe_value(candidate['low_rank_alpha'])}-it{candidate['low_rank_max_projection_iterations']}-"
        f"seed{seed_token}-route{args.route}-tp{args.transport_policy}-"
        f"sts{args.streaming_timing_source}-lts{args.low_rank_timing_source}-"
        f"xla{int(bool(args.jit_compile))}-si{args.sinkhorn_iterations}-seps{_safe_value(args.sinkhorn_epsilon)}-"
        f"as{_safe_value(args.annealed_scaling)}-act{_safe_value(args.annealed_convergence_threshold)}-"
        f"rc{args.row_chunk_size}-cc{args.col_chunk_size}-pc{args.particle_chunk_size}-"
        f"{args.dtype}-{args.tf32_mode}-{args.device_scope}-cuda{cuda_token}"
    )


def _bounded_row_stem(args: argparse.Namespace, candidate: dict[str, Any]) -> str:
    verbose_stem = _row_stem(args, candidate)
    longest_suffix = ".json"
    max_stem_length = MAX_ARTIFACT_FILENAME_COMPONENT_LENGTH - len(longest_suffix)
    if len(verbose_stem) <= max_stem_length:
        return verbose_stem
    digest_payload = json.dumps(
        {
            "output_stem": Path(args.output).stem,
            "candidate": candidate,
            "request_signature": _request_signature(args, candidate),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()[:ROW_STEM_DIGEST_LENGTH]
    prefix_budget = max_stem_length - len("-h") - len(digest)
    prefix = verbose_stem[:prefix_budget].rstrip("-_")
    return f"{prefix}-h{digest}"


def _candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    batch_seeds = _parse_int_csv(args.batch_seeds)
    candidates: list[dict[str, Any]] = []
    for rank in args.low_rank_ranks:
        if rank > args.num_particles:
            continue
        for epsilon in args.low_rank_assignment_epsilons:
            for alpha in args.low_rank_alphas:
                if alpha * rank >= 1.0:
                    continue
                for iterations in args.low_rank_max_projection_iterations_list:
                    candidate = {
                        "candidate_id": f"r{rank}_eps{_safe_value(epsilon)}_alpha{_safe_value(alpha)}_it{iterations}",
                        "batch_size": len(batch_seeds),
                        "batch_seeds": batch_seeds,
                        "time_steps": args.time_steps,
                        "num_particles": args.num_particles,
                        "low_rank_rank": rank,
                        "low_rank_assignment_epsilon": epsilon,
                        "low_rank_alpha": alpha,
                        "low_rank_max_projection_iterations": iterations,
                        "low_rank_convergence_threshold": args.low_rank_convergence_threshold,
                        "low_rank_denominator_floor": args.low_rank_denominator_floor,
                    }
                    candidates.append(candidate)
    return candidates


def _select_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = _candidate_grid(args)
    requested_ids = list(getattr(args, "candidate_ids", []))
    if not requested_ids:
        return candidates
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    missing = [candidate_id for candidate_id in requested_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"candidate-ids not present in enumerated grid: {','.join(missing)}")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("candidate-ids must not contain duplicates")
    return [by_id[candidate_id] for candidate_id in requested_ids]


def _command_for_candidate(args: argparse.Namespace, candidate: dict[str, Any], row_json: Path, row_md: Path) -> list[str]:
    cmd = [
        args.python_executable,
        args.harness_path,
        "--route",
        args.route,
        "--streaming-timing-source",
        args.streaming_timing_source,
        "--low-rank-timing-source",
        args.low_rank_timing_source,
        "--batch-seeds",
        args.batch_seeds,
        "--time-steps",
        str(args.time_steps),
        "--num-particles",
        str(args.num_particles),
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
        "--low-rank-rank",
        str(candidate["low_rank_rank"]),
        "--low-rank-assignment-epsilon",
        str(candidate["low_rank_assignment_epsilon"]),
        "--low-rank-alpha",
        str(candidate["low_rank_alpha"]),
        "--low-rank-max-projection-iterations",
        str(candidate["low_rank_max_projection_iterations"]),
        "--low-rank-convergence-threshold",
        str(candidate["low_rank_convergence_threshold"]),
        "--low-rank-denominator-floor",
        str(candidate["low_rank_denominator_floor"]),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--dtype",
        args.dtype,
        "--tf32-mode",
        args.tf32_mode,
        "--device-scope",
        args.device_scope,
        "--device",
        args.device,
        "--expect-device-kind",
        args.expect_device_kind,
        "--phase-id",
        f"{args.phase_id_prefix}-{candidate['candidate_id']}",
        "--output",
        str(row_json),
        "--markdown-output",
        str(row_md),
    ]
    cmd.append("--jit-compile")
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    return cmd


def _request_signature(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": args.route,
        "batch_seeds": _parse_int_csv(args.batch_seeds),
        "time_steps": args.time_steps,
        "num_particles": args.num_particles,
        "transport_policy": args.transport_policy,
        "streaming_timing_source": args.streaming_timing_source,
        "low_rank_timing_source": args.low_rank_timing_source,
        "jit_compile": args.jit_compile,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "annealed_scaling": args.annealed_scaling,
        "annealed_convergence_threshold": args.annealed_convergence_threshold,
        "row_chunk_size": args.row_chunk_size,
        "col_chunk_size": args.col_chunk_size,
        "particle_chunk_size": args.particle_chunk_size,
        "low_rank_rank": candidate["low_rank_rank"],
        "low_rank_assignment_epsilon": candidate["low_rank_assignment_epsilon"],
        "low_rank_alpha": candidate["low_rank_alpha"],
        "low_rank_max_projection_iterations": candidate["low_rank_max_projection_iterations"],
        "low_rank_convergence_threshold": candidate["low_rank_convergence_threshold"],
        "low_rank_denominator_floor": candidate["low_rank_denominator_floor"],
        "dtype": args.dtype,
        "tf32_mode": args.tf32_mode,
        "device_scope": args.device_scope,
        "cuda_visible_devices": args.cuda_visible_devices,
        "device": args.device,
        "expect_device_kind": args.expect_device_kind,
    }


def _float_equal(lhs: Any, rhs: Any, *, rel_tol: float = 1.0e-12, abs_tol: float = 1.0e-12) -> bool:
    try:
        return math.isclose(float(lhs), float(rhs), rel_tol=rel_tol, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return False


def _thresholds(paired: dict[str, Any] | None) -> dict[str, float]:
    defaults = {
        "log_likelihood_max_abs_delta": 10.0,
        "log_likelihood_mean_abs_delta": 5.0,
        "filtered_mean_relative_l2": 0.20,
        "filtered_mean_rms": 2.5,
        "filtered_variance_relative_l2": 0.75,
        "filtered_variance_rms": 25.0,
        "final_particle_mean_relative_l2": 0.20,
        "final_particle_mean_abs_l2": 25.0,
        "warm_median_streaming_over_low_rank": 1.25,
    }
    if paired and isinstance(paired.get("thresholds"), dict):
        defaults.update({key: float(value) for key, value in paired["thresholds"].items()})
    return defaults


def _artifact_error(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    command: list[str],
    row_json: Path,
    row_md: Path,
    row_log: Path,
    status: str,
    veto: str,
    error: str,
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "request_signature": _request_signature(args, candidate),
        "status": status,
        "command": command,
        "row_json": str(row_json),
        "row_markdown": str(row_md),
        "row_log": str(row_log),
        "error": error,
        "classification": {
            "candidate_label": "hard-vetoed",
            "hard_vetoes": [veto],
            "paired_comparability_pass": False,
            "warm_time_screen_pass": False,
            "low_rank_provenance_complete": False,
            "gpu_tf32_provenance_complete": False,
        },
    }


def _read_row_result(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    command: list[str],
    row_json: Path,
    row_md: Path,
    row_log: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not row_json.exists():
        return None, _artifact_error(
            args,
            candidate,
            command,
            row_json,
            row_md,
            row_log,
            "MISSING",
            "row_json_missing",
            "row JSON artifact missing",
        )
    if not row_md.exists():
        return None, _artifact_error(
            args,
            candidate,
            command,
            row_json,
            row_md,
            row_log,
            "MISSING",
            "row_markdown_missing",
            "row Markdown artifact missing",
        )
    try:
        row_result = json.loads(row_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, _artifact_error(
            args,
            candidate,
            command,
            row_json,
            row_md,
            row_log,
            "CORRUPT",
            "row_json_corrupt",
            f"{type(exc).__name__}: {exc}",
        )
    mismatches = _row_request_mismatches(args, candidate, row_result)
    if mismatches:
        return None, _artifact_error(
            args,
            candidate,
            command,
            row_json,
            row_md,
            row_log,
            "MISMATCH",
            "row_request_mismatch",
            ";".join(mismatches),
        )
    return row_result, None


def _value_leq(payload: dict[str, Any], key: str, threshold: float) -> bool:
    value = payload.get(key)
    return value is not None and float(value) <= threshold


def _paired_comparability_pass(paired: dict[str, Any] | None) -> bool:
    if paired is None:
        return False
    thresholds = _thresholds(paired)
    mean_pass = _value_leq(paired, "filtered_mean_relative_l2", thresholds["filtered_mean_relative_l2"]) or _value_leq(
        paired, "filtered_mean_rms", thresholds["filtered_mean_rms"]
    )
    variance_pass = _value_leq(
        paired, "filtered_variance_relative_l2", thresholds["filtered_variance_relative_l2"]
    ) or _value_leq(paired, "filtered_variance_rms", thresholds["filtered_variance_rms"])
    final_mean_pass = _value_leq(
        paired, "final_particle_mean_relative_l2", thresholds["final_particle_mean_relative_l2"]
    ) or _value_leq(paired, "final_particle_mean_abs_l2", thresholds["final_particle_mean_abs_l2"])
    return (
        _value_leq(paired, "log_likelihood_max_abs_delta", thresholds["log_likelihood_max_abs_delta"])
        and _value_leq(paired, "log_likelihood_mean_abs_delta", thresholds["log_likelihood_mean_abs_delta"])
        and mean_pass
        and variance_pass
        and final_mean_pass
    )


def _speed_screen_pass(paired: dict[str, Any] | None) -> bool:
    if paired is None:
        return False
    threshold = _thresholds(paired)["warm_median_streaming_over_low_rank"]
    value = paired.get("warm_median_streaming_over_low_rank")
    return value is not None and float(value) >= threshold


def _low_rank_row(result: dict[str, Any]) -> dict[str, Any] | None:
    for row in result.get("rows", []):
        if row.get("route") == "low_rank":
            return row
    return None


def _row_request_mismatches(args: argparse.Namespace, candidate: dict[str, Any], result: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    shape = result.get("shape") or {}
    if shape.get("batch_size") != candidate["batch_size"]:
        mismatches.append("batch_size")
    if shape.get("time_steps") != candidate["time_steps"]:
        mismatches.append("time_steps")
    if shape.get("num_particles") != candidate["num_particles"]:
        mismatches.append("num_particles")
    if result.get("batch_seeds") != candidate["batch_seeds"]:
        mismatches.append("batch_seeds")
    if result.get("transport_policy") != args.transport_policy:
        mismatches.append("transport_policy")
    if result.get("route_request") != args.route:
        mismatches.append("route_request")
    streaming_rows = [row for row in result.get("rows", []) if row.get("route") == "streaming"]
    if args.route in {"both", "streaming"} and not streaming_rows:
        mismatches.append("missing_streaming_row")
    elif streaming_rows and streaming_rows[0].get("streaming_timing_source") != args.streaming_timing_source:
        mismatches.append("streaming_timing_source")
    for row in result.get("rows", []):
        if row.get("sinkhorn_iterations") is not None and row.get("sinkhorn_iterations") != args.sinkhorn_iterations:
            mismatches.append(f"{row.get('route', 'unknown')}:sinkhorn_iterations")
        if row.get("sinkhorn_epsilon") is not None and not _float_equal(row.get("sinkhorn_epsilon"), args.sinkhorn_epsilon):
            mismatches.append(f"{row.get('route', 'unknown')}:sinkhorn_epsilon")
        if row.get("annealed_scaling") is not None and not _float_equal(row.get("annealed_scaling"), args.annealed_scaling):
            mismatches.append(f"{row.get('route', 'unknown')}:annealed_scaling")
        if row.get("annealed_convergence_threshold") is not None and not _float_equal(
            row.get("annealed_convergence_threshold"), args.annealed_convergence_threshold
        ):
            mismatches.append(f"{row.get('route', 'unknown')}:annealed_convergence_threshold")

    low_rank = _low_rank_row(result)
    if args.route in {"both", "low_rank"} and low_rank is None:
        mismatches.append("missing_low_rank_row")
    elif low_rank is not None:
        expected_float_fields = (
            ("low_rank_assignment_epsilon", candidate["low_rank_assignment_epsilon"]),
            ("low_rank_alpha", candidate["low_rank_alpha"]),
            ("low_rank_convergence_threshold", candidate["low_rank_convergence_threshold"]),
            ("low_rank_denominator_floor", candidate["low_rank_denominator_floor"]),
        )
        if low_rank.get("low_rank_rank") != candidate["low_rank_rank"]:
            mismatches.append("low_rank_rank")
        if low_rank.get("low_rank_max_projection_iterations") != candidate["low_rank_max_projection_iterations"]:
            mismatches.append("low_rank_max_projection_iterations")
        if low_rank.get("low_rank_timing_source") != args.low_rank_timing_source:
            mismatches.append("low_rank_timing_source")
        if low_rank.get("jit_compile") != args.jit_compile:
            mismatches.append("low_rank_jit_compile")
        for key, expected in expected_float_fields:
            if not _float_equal(low_rank.get(key), expected):
                mismatches.append(key)

    manifest = result.get("run_manifest") or {}
    if manifest.get("streaming_timing_source") != args.streaming_timing_source:
        mismatches.append("manifest_streaming_timing_source")
    if manifest.get("low_rank_timing_source") != args.low_rank_timing_source:
        mismatches.append("manifest_low_rank_timing_source")
    if manifest.get("jit_compile") != args.jit_compile:
        mismatches.append("manifest_jit_compile")
    if manifest.get("device_scope") != args.device_scope:
        mismatches.append("device_scope")
    if manifest.get("requested_cuda_visible_devices") != args.cuda_visible_devices:
        mismatches.append("requested_cuda_visible_devices")
    if manifest.get("device") != args.device:
        mismatches.append("device")
    if manifest.get("expect_device_kind") != args.expect_device_kind:
        mismatches.append("expect_device_kind")
    precision = manifest.get("precision") or {}
    if precision.get("dtype") != args.dtype and precision.get("active_dtype") != args.dtype:
        mismatches.append("dtype")
    if precision.get("tf32_mode") != args.tf32_mode and precision.get("tf32_requested") != args.tf32_mode:
        mismatches.append("tf32_mode")
    command = str(manifest.get("command", ""))
    command_checks = {
        "--row-chunk-size": args.row_chunk_size,
        "--col-chunk-size": args.col_chunk_size,
        "--particle-chunk-size": args.particle_chunk_size,
    }
    for flag, expected in command_checks.items():
        needle = f"{flag} {expected}"
        if command and needle not in command:
            mismatches.append(flag.lstrip("-").replace("-", "_"))
    return mismatches


def _manifest_has_basic_provenance(result: dict[str, Any]) -> bool:
    manifest = result.get("run_manifest", {})
    precision = manifest.get("precision", {})
    return (
        "selected_physical_gpu" in manifest
        and "precision" in manifest
        and (
            "tf32_execution_enabled" in precision
            or "tf32_execution_recorded" in precision
            or "tf32_mode" in precision
            or "tf32_requested" in precision
        )
    )


def _manifest_has_gpu_tf32_provenance(result: dict[str, Any]) -> bool:
    manifest = result.get("run_manifest", {})
    precision = manifest.get("precision", {})
    selected = manifest.get("selected_physical_gpu") or {}
    requested_tf32 = precision.get("tf32_mode", precision.get("tf32_requested"))
    recorded_tf32 = precision.get("tf32_execution_enabled", precision.get("tf32_execution_recorded"))
    return (
        manifest.get("device_scope") == "visible"
        and manifest.get("expect_device_kind") == "gpu"
        and manifest.get("streaming_timing_source") == "compiled_core"
        and manifest.get("low_rank_timing_source") == "compiled_core"
        and manifest.get("jit_compile") is True
        and selected.get("uuid") is not None
        and requested_tf32 == "enabled"
        and recorded_tf32 is True
    )


def _low_rank_provenance_complete(result: dict[str, Any]) -> bool:
    row = _low_rank_row(result)
    if row is None:
        return False
    required = (
        "low_rank_rank",
        "low_rank_assignment_epsilon",
        "low_rank_alpha",
        "low_rank_max_projection_iterations",
        "low_rank_convergence_threshold",
        "low_rank_denominator_floor",
        "route_invocations",
        "active_resampling_mask_count",
        "max_factor_marginal_residual",
        "all_finite_factors",
        "all_nonnegative_factors",
        "all_positive_g",
        "low_rank_timing_source",
        "jit_compile",
    )
    return (
        all(key in row for key in required)
        and row.get("low_rank_timing_source") == "compiled_core"
        and row.get("jit_compile") is True
        and _manifest_has_basic_provenance(result)
    )


def _hard_vetoes(result: dict[str, Any]) -> list[str]:
    vetoes = list(result.get("hard_vetoes") or [])
    if result.get("actual_sir_semantics_pass") is False:
        vetoes.append("actual_sir_semantics_missing")
    for row in result.get("rows", []):
        for veto in row.get("hard_vetoes") or []:
            vetoes.append(f"{row.get('route', 'unknown')}:{veto}")
    return vetoes


def _classify_row_result(result: dict[str, Any]) -> dict[str, Any]:
    paired = result.get("paired_comparability")
    hard_vetoes = _hard_vetoes(result)
    comparable = _paired_comparability_pass(paired)
    speed_pass = _speed_screen_pass(paired)
    provenance_complete = _low_rank_provenance_complete(result)
    gpu_tf32_provenance_complete = _manifest_has_gpu_tf32_provenance(result)
    if hard_vetoes:
        label = "hard-vetoed"
    elif comparable and speed_pass and provenance_complete and gpu_tf32_provenance_complete:
        label = "freeze-nominated"
    elif speed_pass and not comparable:
        label = "faster-but-incomparable"
    elif comparable and not speed_pass:
        label = "comparable-but-slow"
    elif comparable and speed_pass and not gpu_tf32_provenance_complete:
        label = "schema-valid-nonpromotional"
    else:
        label = "incomparable"
    low_rank = _low_rank_row(result) or {}
    manifest = result.get("run_manifest") or {}
    return {
        "candidate_label": label,
        "hard_vetoes": hard_vetoes,
        "paired_comparability_pass": comparable,
        "warm_time_screen_pass": speed_pass,
        "low_rank_provenance_complete": provenance_complete,
        "gpu_tf32_provenance_complete": gpu_tf32_provenance_complete,
        "log_likelihood_mean_abs_delta": None if paired is None else paired.get("log_likelihood_mean_abs_delta"),
        "log_likelihood_max_abs_delta": None if paired is None else paired.get("log_likelihood_max_abs_delta"),
        "warm_median_streaming_over_low_rank": None if paired is None else paired.get("warm_median_streaming_over_low_rank"),
        "route_invocations": low_rank.get("route_invocations"),
        "active_resampling_mask_count": low_rank.get("active_resampling_mask_count"),
        "max_factor_marginal_residual": low_rank.get("max_factor_marginal_residual"),
        "selected_physical_gpu": manifest.get("selected_physical_gpu"),
        "precision": manifest.get("precision"),
    }


def _row_paths(args: argparse.Namespace, candidate: dict[str, Any]) -> tuple[Path, Path, Path]:
    stem = _bounded_row_stem(args, candidate)
    output_dir = Path(args.output).parent
    log_dir = Path(args.log_dir)
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md", log_dir / f"{stem}.log"


def _run_candidate(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    row_json, row_md, row_log = _row_paths(args, candidate)
    row_json.parent.mkdir(parents=True, exist_ok=True)
    row_log.parent.mkdir(parents=True, exist_ok=True)
    command = _command_for_candidate(args, candidate, row_json, row_md)
    signature = _request_signature(args, candidate)
    if args.mode == "dry-run":
        return {
            "candidate": candidate,
            "request_signature": signature,
            "status": "DRY_RUN",
            "command": command,
            "row_json": str(row_json),
            "row_markdown": str(row_md),
            "row_log": str(row_log),
        }
    if args.mode == "aggregate-existing":
        row_result, error_row = _read_row_result(args, candidate, command, row_json, row_md, row_log)
        if error_row is not None:
            return error_row
        return {
            "candidate": candidate,
            "request_signature": signature,
            "status": row_result.get("status", "UNKNOWN"),
            "command": command,
            "row_json": str(row_json),
            "row_markdown": str(row_md),
            "row_log": str(row_log),
            "classification": _classify_row_result(row_result),
        }

    started = time.perf_counter()
    try:
        with row_log.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.row_timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired:
        wall_time = time.perf_counter() - started
        timeout_row = _artifact_error(
            args,
            candidate,
            command,
            row_json,
            row_md,
            row_log,
            "TIMEOUT",
            "row_timeout",
            f"row subprocess exceeded {args.row_timeout_seconds}s",
        )
        timeout_row["wall_time_seconds"] = wall_time
        return timeout_row
    wall_time = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "candidate": candidate,
            "request_signature": signature,
            "status": "ERROR",
            "command": command,
            "row_json": str(row_json),
            "row_markdown": str(row_md),
            "row_log": str(row_log),
            "returncode": completed.returncode,
            "wall_time_seconds": wall_time,
            "classification": {
                "candidate_label": "hard-vetoed",
                "hard_vetoes": ["row_subprocess_nonzero_exit"],
                "paired_comparability_pass": False,
                "warm_time_screen_pass": False,
                "low_rank_provenance_complete": False,
                "gpu_tf32_provenance_complete": False,
            },
        }
    row_result, error_row = _read_row_result(args, candidate, command, row_json, row_md, row_log)
    if error_row is not None:
        error_row["returncode"] = completed.returncode
        error_row["wall_time_seconds"] = wall_time
        return error_row
    return {
        "candidate": candidate,
        "request_signature": signature,
        "status": row_result.get("status", "UNKNOWN"),
        "command": command,
        "row_json": str(row_json),
        "row_markdown": str(row_md),
        "row_log": str(row_log),
        "returncode": completed.returncode,
        "wall_time_seconds": wall_time,
        "classification": _classify_row_result(row_result),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    for row in rows:
        label = (row.get("classification") or {}).get("candidate_label", row.get("status", "UNKNOWN"))
        labels[label] = labels.get(label, 0) + 1
    return {
        "num_candidates": len(rows),
        "num_freeze_nominated": labels.get("freeze-nominated", 0),
        "num_comparable_but_slow": labels.get("comparable-but-slow", 0),
        "labels": labels,
    }


def _run_text(args: list[str], *, timeout: float = 10.0) -> str:
    try:
        return subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    start = time.perf_counter()
    candidates = _select_candidates(args)
    rows = [_run_candidate(args, candidate) for candidate in candidates]
    ended_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    status = "DRY_RUN" if args.mode == "dry-run" else "PASS"
    pass_statuses = {"PASS"}
    if args.mode == "dry-run":
        pass_statuses = {"DRY_RUN"}
    if any(row.get("status") not in pass_statuses for row in rows):
        status = "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": args.mode,
        "algorithm_under_test": "actual-SIR d18 low-rank tuning-grid orchestration over the owned validation harness",
        "plan_path": PLAN_PATH,
        "harness_path": args.harness_path,
        "shape": {
            "batch_size": len(_parse_int_csv(args.batch_seeds)),
            "batch_seeds": _parse_int_csv(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
        },
        "grid": {
            "low_rank_ranks": list(args.low_rank_ranks),
            "low_rank_assignment_epsilons": list(args.low_rank_assignment_epsilons),
            "low_rank_alphas": list(args.low_rank_alphas),
            "low_rank_max_projection_iterations": list(args.low_rank_max_projection_iterations_list),
            "low_rank_convergence_threshold": args.low_rank_convergence_threshold,
            "low_rank_denominator_floor": args.low_rank_denominator_floor,
            "candidate_ids": list(args.candidate_ids),
        },
        "execution_policy": {
            "row_timeout_seconds": args.row_timeout_seconds,
            "device_scope": args.device_scope,
            "cuda_visible_devices": args.cuda_visible_devices,
            "device": args.device,
            "expect_device_kind": args.expect_device_kind,
            "dtype": args.dtype,
            "tf32_mode": args.tf32_mode,
            "streaming_timing_source": args.streaming_timing_source,
            "low_rank_timing_source": args.low_rank_timing_source,
            "jit_compile": args.jit_compile,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "summary": _summary(rows),
        "rows": rows,
        "run_manifest": {
            "command": " ".join(sys.argv),
            "working_directory": str(ROOT),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": _run_text(["git", "status", "--short"]),
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_time_seconds": time.perf_counter() - start,
            "output": args.output,
            "markdown_output": args.markdown_output,
        },
        "nonclaims": list(NONCLAIMS),
    }


def write_markdown(result: dict[str, Any], path: Path, json_path: Path) -> None:
    lines = [
        "# Actual-SIR Low-Rank Tuning Grid",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Status: `{result['status']}`",
        f"- Mode: `{result['mode']}`",
        f"- Candidates: `{result['summary']['num_candidates']}`",
        f"- Freeze-nominated: `{result['summary']['num_freeze_nominated']}`",
        "",
        "## Candidate Labels",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(result["summary"]["labels"].items()):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Candidate | Status | Label | JSON | Log |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["rows"]:
        candidate = row["candidate"]
        label = (row.get("classification") or {}).get("candidate_label", row.get("status", "UNKNOWN"))
        lines.append(
            "| `{candidate}` | `{status}` | `{label}` | `{json}` | `{log}` |".format(
                candidate=candidate["candidate_id"],
                status=row.get("status"),
                label=label,
                json=row.get("row_json"),
                log=row.get("row_log"),
            )
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = build_result(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.markdown_output), output)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate one prepared Actual-SV overcomplete chart on a frozen point set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--gpu-memory-limit-mib", type=int, default=None)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument(
        "--point-set", choices=("center", "design", "held-out", "fd"), required=True
    )
    parser.add_argument(
        "--evaluation-mode", choices=("chart", "score"), required=True
    )
    parser.add_argument("--cpu-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-jit-compile", action="store_true")
    return parser.parse_args()


ARGS = _parse()
if ARGS.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU evaluation requires CUDA_VISIBLE_DEVICES=-1 before Python")
if ARGS.device == "gpu" and ARGS.gpu_memory_limit_mib != 8192:
    raise RuntimeError("this campaign requires the frozen 8192 MiB GPU cap")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") in ("true", "1"):
    raise RuntimeError("memory growth is forbidden with a logical-device memory cap")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402


physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
if ARGS.device == "gpu":
    if not physical_gpus:
        raise RuntimeError("trusted GPU mode requires a visible physical GPU")
    tf.config.set_logical_device_configuration(
        physical_gpus[0],
        [
            tf.config.LogicalDeviceConfiguration(
                memory_limit=ARGS.gpu_memory_limit_mib
            )
        ],
    )
logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
if ARGS.device == "gpu" and len(logical_gpus) != 1:
    raise RuntimeError("GPU mode requires exactly one configured logical GPU")
if ARGS.device == "cpu" and logical_gpus:
    raise RuntimeError("CPU mode unexpectedly exposes a logical GPU")

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model  # noqa: E402
from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import (  # noqa: E402
    SourceRouteSpec,
    audit_source_path,
    inventory_graph_def,
)


DTYPE = tf.float64
SCHEMA = "bayesfilter.contract_e_tp.actual_sv_overcomplete_result.v1"
PREPARATION_SCHEMA = "bayesfilter.contract_e_tp.scalar_sv_overcomplete_preparation.v3"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_payload(value: tf.Tensor) -> dict[str, Any]:
    array = value.numpy()
    return {
        "shape": list(array.shape),
        "values": float(array) if array.ndim == 0 else array.tolist(),
        "serialized_tensor_sha256": hashlib.sha256(
            tf.io.serialize_tensor(value).numpy()
        ).hexdigest(),
    }


def _numeric_difference(left: Any, right: Any) -> dict[str, Any]:
    left_tensor = tf.constant(left, DTYPE)
    right_tensor = tf.constant(right, DTYPE)
    if left_tensor.shape != right_tensor.shape:
        return {
            "shape_equal": False,
            "left_shape": left_tensor.shape.as_list(),
            "right_shape": right_tensor.shape.as_list(),
        }
    difference = tf.abs(left_tensor - right_tensor)
    scale = tf.maximum(
        tf.maximum(tf.abs(left_tensor), tf.abs(right_tensor)),
        tf.constant(1.0e-300, DTYPE),
    )
    return {
        "shape_equal": True,
        "max_absolute_difference": float(tf.reduce_max(difference).numpy()),
        "max_symmetric_relative_difference": float(
            tf.reduce_max(difference / scale).numpy()
        ),
    }


def _cpu_comparison(
    cpu_path: Path | None,
    preparation_path: Path,
    preparation: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if cpu_path is None:
        return {"status": "not_applicable_cpu_reference_run"}
    payload = _load(cpu_path)
    if payload.get("schema") != SCHEMA:
        raise ValueError("CPU comparison result schema differs")
    if payload.get("route_id") != preparation["route_id"]:
        raise ValueError("CPU/GPU route identities differ")
    if payload.get("preparation", {}).get("sha256") != _sha256(preparation_path):
        raise ValueError("CPU/GPU preparation identities differ")
    for field, expected in (
        ("time_steps", preparation["target"]["time_steps"]),
        ("capacity", preparation["chart_contract"]["anchor_count"]),
        ("point_set", ARGS.point_set),
        ("evaluation_mode", ARGS.evaluation_mode),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"CPU/GPU {field} differs")
    cpu_rows = {row["name"]: row for row in payload["rows"]}
    gpu_rows = {row["name"]: row for row in rows}
    if cpu_rows.keys() != gpu_rows.keys():
        raise ValueError("CPU/GPU point identities differ")
    comparisons: dict[str, Any] = {}
    for name in cpu_rows:
        cpu_row = cpu_rows[name]
        gpu_row = gpu_rows[name]
        row_comparison = {
            "valid_equal": cpu_row["valid"] == gpu_row["valid"],
            "objective": _numeric_difference(
                cpu_row["objective"]["values"],
                gpu_row["objective"]["values"],
            ),
            "minimum_weight": _numeric_difference(
                cpu_row["minimum_weight"], gpu_row["minimum_weight"]
            ),
            "maximum_gram_condition_number": _numeric_difference(
                cpu_row["maximum_gram_condition_number"],
                gpu_row["maximum_gram_condition_number"],
            ),
            "maximum_scaled_relative_residual": _numeric_difference(
                cpu_row["maximum_scaled_relative_residual"],
                gpu_row["maximum_scaled_relative_residual"],
            ),
            "maximum_feature_residual_abs": _numeric_difference(
                cpu_row["maximum_feature_residual_abs"],
                gpu_row["maximum_feature_residual_abs"],
            ),
        }
        if ARGS.evaluation_mode == "score":
            row_comparison["score_manual"] = _numeric_difference(
                cpu_row["score_manual"]["values"],
                gpu_row["score_manual"]["values"],
            )
        comparisons[name] = row_comparison
    return {
        "status": "descriptive_only_no_cpu_gpu_equivalence_threshold",
        "cpu_result": {
            "path": str(cpu_path.relative_to(ROOT)),
            "sha256": _sha256(cpu_path),
        },
        "points": comparisons,
    }


def _source_audit() -> dict[str, Any]:
    return audit_source_path(
        Path(model.__file__),
        SourceRouteSpec(
            roots=("make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf",),
            loop_roles={},
            required_reachable=(
                "make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf.evaluate",
                "contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core",
                "contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core.cond",
                "contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core.body",
                "_actual_sv_continuation_multi_jvp",
                "_actual_sv_continuation_multi_jvp.cond",
                "_actual_sv_continuation_multi_jvp.body",
            ),
        ),
    )


def _points(preparation: dict[str, Any]) -> list[tuple[str, list[float]]]:
    specification = _load(_path(preparation["specification"]["path"]))
    geometry = specification["parameter_geometry"]
    center = tf.constant(specification["target"]["center_theta"], DTYPE)
    scale = tf.constant(geometry["scale_diagonal"], DTYPE)
    if ARGS.point_set == "center":
        normalized = [("center", [0.0, 0.0])]
    elif ARGS.point_set == "design":
        normalized = [
            (f"design_{index:02d}", item)
            for index, item in enumerate(geometry["design_points_normalized_ordered"])
        ]
    elif ARGS.point_set == "held-out":
        normalized = [
            (f"held_out_{index:02d}", item)
            for index, item in enumerate(geometry["held_out_points_normalized_ordered"])
        ]
    else:
        normalized = [
            ("center", [0.0, 0.0]),
            ("gamma_minus", [-1.0, 0.0]),
            ("gamma_plus", [1.0, 0.0]),
            ("log_beta_minus", [0.0, -1.0]),
            ("log_beta_plus", [0.0, 1.0]),
        ]
    return [
        (name, (center + scale * tf.constant(item, DTYPE)).numpy().tolist())
        for name, item in normalized
    ]


def main() -> int:
    output = _path(ARGS.output)
    preparation_path = _path(ARGS.preparation)
    cpu_path = _path(ARGS.cpu_result) if ARGS.cpu_result else None
    if output.exists():
        raise FileExistsError(output)
    if ARGS.device == "gpu" and cpu_path is None:
        raise ValueError("GPU certification requires --cpu-result")
    if ARGS.device == "cpu" and cpu_path is not None:
        raise ValueError("CPU evaluation must not provide --cpu-result")
    preparation = _load(preparation_path)
    if preparation.get("schema") != PREPARATION_SCHEMA:
        raise ValueError("runner requires a v3 overcomplete preparation")
    if not preparation["summary"]["all_preparation_valid"]:
        payload = {
            "schema": SCHEMA,
            "status": "REJECT_FAILED_PREPARATION",
            "preparation": {
                "path": str(ARGS.preparation),
                "sha256": _sha256(preparation_path),
                "preparation_status": preparation.get("status"),
                "summary": preparation.get("summary"),
            },
            "point_set": ARGS.point_set,
            "evaluation_mode": ARGS.evaluation_mode,
            "execution": {
                "command": " ".join(sys.argv),
                "device": ARGS.device,
                "jit_compile": not ARGS.no_jit_compile,
                "logical_device_memory_limit_mib": ARGS.gpu_memory_limit_mib,
            },
            "failure_classification": (
                "mathematical_preparation_failure_not_runtime_or_timeout"
            ),
            "nonclaims": ["no runtime candidate evaluation occurred"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "preparation": payload["preparation"],
                },
                indent=2,
            )
        )
        return 2
    time_steps = int(preparation["target"]["time_steps"])
    capacity = int(preparation["chart_contract"]["anchor_count"])
    spec = model.make_scalar_sv_spec(preparation["row_id"])
    bound_arguments = (
        spec,
        tf.constant(preparation["target"]["target_observations"], DTYPE),
        tf.constant(preparation["target"]["flow_observations"], DTYPE),
        tf.constant(preparation["teacher_quadrature"]["nodes"], DTYPE),
        tf.constant(preparation["teacher_quadrature"]["weights"], DTYPE),
        tf.reshape(
            tf.constant(preparation["active_indices"], tf.int32),
            [time_steps - 1, capacity],
        ),
        tf.reshape(
            tf.constant(preparation["row_scales"], DTYPE),
            [time_steps - 1, model.FEATURE_COUNT],
        ),
        tf.reshape(
            tf.constant(preparation["reference_weights"], DTYPE),
            [time_steps - 1, capacity],
        ),
        tf.constant(preparation["continuation_quadrature"]["points"], DTYPE),
        tf.constant(preparation["continuation_quadrature"]["weights"], DTYPE),
    )
    factory_builder = (
        model.make_contract_e_tp_actual_sv_overcomplete_forward_tf
        if ARGS.evaluation_mode == "chart"
        else model.make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf
    )
    factory = factory_builder(
        *bound_arguments,
        lookahead_steps=int(preparation["feature_contract"]["lookahead_steps"]),
        jit_compile=not ARGS.no_jit_compile,
    )
    autodiff_factory = (
        model.make_contract_e_tp_actual_sv_overcomplete_tf(
            *bound_arguments,
            lookahead_steps=int(preparation["feature_contract"]["lookahead_steps"]),
            jit_compile=not ARGS.no_jit_compile,
        )
        if ARGS.evaluation_mode == "score" and ARGS.device == "cpu"
        else None
    )
    started = time.perf_counter()
    trace_started = time.perf_counter()
    concrete = factory.get_concrete_function()
    graph = inventory_graph_def(concrete.graph.as_graph_def())
    source_audit = _source_audit() if ARGS.evaluation_mode == "score" else None
    autodiff_concrete = (
        autodiff_factory.get_concrete_function()
        if autodiff_factory is not None
        else None
    )
    trace_seconds = time.perf_counter() - trace_started
    device = "/GPU:0" if ARGS.device == "gpu" else "/CPU:0"
    rows: list[dict[str, Any]] = []
    compile_seconds = None
    warm_seconds = 0.0
    warm_replay_seconds = None
    warm_replay_identity = None
    center_result: dict[str, tf.Tensor] | None = None
    if ARGS.device == "gpu":
        tf.config.experimental.reset_memory_stats("GPU:0")
    with tf.device(device):
        for index, (name, theta_values) in enumerate(_points(preparation)):
            theta = tf.constant(theta_values, DTYPE)
            point_started = time.perf_counter()
            result = concrete(theta)
            autodiff_result = (
                autodiff_concrete(theta) if autodiff_concrete is not None else None
            )
            elapsed = time.perf_counter() - point_started
            if index == 0:
                center_result = result
                compile_seconds = elapsed
                replay_started = time.perf_counter()
                replay = concrete(theta)
                warm_replay_seconds = time.perf_counter() - replay_started
                replay_fields = [
                    "objective",
                    "increment_history",
                    "final_particles",
                    "final_log_unnormalized_weights",
                ]
                if ARGS.evaluation_mode == "score":
                    replay_fields.append("score_manual")
                warm_replay_identity = all(
                    bool(tf.reduce_all(tf.equal(result[field], replay[field])).numpy())
                    for field in replay_fields
                )
            else:
                warm_seconds += elapsed
            valid = bool(result["valid"].numpy())
            row = {
                    "name": name,
                    "theta": theta_values,
                    "valid": valid,
                    "objective": _tensor_payload(result["objective"]),
                    "minimum_weight": float(
                        tf.reduce_min(result["minimum_weight_history"]).numpy()
                    ) if time_steps > 1 else None,
                    "weakest_time_zero_based": int(
                        tf.argmin(result["minimum_weight_history"]).numpy()
                    ) if time_steps > 1 else None,
                    "maximum_gram_condition_number": float(
                        tf.reduce_max(result["gram_condition_number_history"]).numpy()
                    ) if time_steps > 1 else None,
                    "maximum_scaled_relative_residual": float(
                        tf.reduce_max(result["scaled_relative_residual_history"]).numpy()
                    ) if time_steps > 1 else None,
                    "maximum_feature_residual_abs": float(
                        tf.reduce_max(tf.abs(result["feature_residual_history"])).numpy()
                    ) if time_steps > 1 else 0.0,
                    "evaluation_seconds": elapsed,
                }
            if ARGS.evaluation_mode == "score":
                row["score_manual"] = _tensor_payload(result["score_manual"])
                if autodiff_result is not None:
                    row["score_autodiff_oracle"] = _tensor_payload(
                        autodiff_result["score_autodiff_oracle"]
                    )
                    difference = tf.abs(
                        result["score_manual"]
                        - autodiff_result["score_autodiff_oracle"]
                    )
                    row["manual_autodiff_max_absolute_difference"] = float(
                        tf.reduce_max(difference).numpy()
                    )
                    row["manual_autodiff_symmetric_relative_difference"] = float(
                        tf.linalg.norm(difference).numpy()
                        / max(
                            float(tf.linalg.norm(result["score_manual"]).numpy()),
                            float(
                                tf.linalg.norm(
                                    autodiff_result["score_autodiff_oracle"]
                                ).numpy()
                            ),
                            1.0e-12,
                        )
                    )
                else:
                    row["autodiff_oracle_status"] = (
                        "not_run_on_gpu_long_horizon_reverse_mode_known_nonfinite;_"
                        "Phase_5_forward_AD_and_FD_are_the_derivative_references"
                    )
            rows.append(row)
    finite_difference = None
    if ARGS.evaluation_mode == "score" and ARGS.point_set == "fd":
        rows_by_name = {row["name"]: row for row in rows}
        step = 1.0e-5
        fd_score = tf.constant(
            [
                (
                    rows_by_name["gamma_plus"]["objective"]["values"]
                    - rows_by_name["gamma_minus"]["objective"]["values"]
                )
                / (2.0 * step),
                (
                    rows_by_name["log_beta_plus"]["objective"]["values"]
                    - rows_by_name["log_beta_minus"]["objective"]["values"]
                )
                / (2.0 * step),
            ],
            DTYPE,
        )
        manual_score = tf.constant(
            rows_by_name["center"]["score_manual"]["values"], DTYPE
        )
        difference = manual_score - fd_score
        relative = float(
            (
                tf.linalg.norm(difference)
                / tf.maximum(tf.linalg.norm(fd_score), tf.constant(1.0, DTYPE))
            ).numpy()
        )
        tolerance = 0.05 * (2.0 ** 0.5)
        finite_difference = {
            "step": step,
            "score": _tensor_payload(fd_score),
            "manual_score": _tensor_payload(manual_score),
            "difference": _tensor_payload(difference),
            "relative_error": relative,
            "relative_tolerance": tolerance,
            "pass": relative <= tolerance,
            "tolerance_provenance": (
                "owner FD-only policy 5 percent times sqrt(p), p=2"
            ),
        }
    if center_result is None:
        raise RuntimeError("evaluation did not produce a center result")
    output_field_names = [
        "objective",
        "increment_history",
        "final_particles",
        "final_log_unnormalized_weights",
    ]
    if ARGS.evaluation_mode == "score":
        output_field_names.append("score_manual")
    output_devices = {
        field: center_result[field].device
        for field in output_field_names
    }
    requested_device_token = "GPU:0" if ARGS.device == "gpu" else "CPU:0"
    requested_device_placement = all(
        requested_device_token in value.upper() for value in output_devices.values()
    )
    graph_topology_pass = graph["functional_loop_count"] >= 2
    if source_audit is not None:
        graph_topology_pass = graph_topology_pass and source_audit["approved"]
    gpu_memory = (
        {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
        if ARGS.device == "gpu"
        else None
    )
    cpu_comparison = _cpu_comparison(
        cpu_path, preparation_path, preparation, rows
    )
    fd_gate_applicable = (
        ARGS.evaluation_mode == "score" and ARGS.point_set == "fd"
    )
    certification_gates = {
        "all_chart_outputs_valid": all(row["valid"] for row in rows),
        "same_scalar_fd": (
            finite_difference is not None and finite_difference["pass"]
        )
        if fd_gate_applicable
        else True,
        "warm_replay_bitwise_identity": bool(warm_replay_identity),
        "functional_loop_graph_and_source_guard": graph_topology_pass,
        "requested_output_device_placement": requested_device_placement,
        "one_logical_gpu_with_8192_mib_cap": (
            ARGS.device != "gpu"
            or (
                len(logical_gpus) == 1
                and ARGS.gpu_memory_limit_mib == 8192
            )
        ),
        "memory_growth_disabled": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")
        not in ("true", "1"),
    }
    status = (
        "PASS_TRUSTED_GPU_XLA_ENGINEERING_CERTIFICATION"
        if ARGS.device == "gpu" and all(certification_gates.values())
        else (
            "PASS_FINITE_PROGRAM"
            if ARGS.device == "cpu" and all(row["valid"] for row in rows)
            else "FAIL_FINITE_PROGRAM"
        )
    )
    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "route_id": preparation["route_id"],
        "plan": preparation["plan"],
        "preparation": {"path": str(ARGS.preparation), "sha256": _sha256(preparation_path)},
        "point_set": ARGS.point_set,
        "evaluation_mode": ARGS.evaluation_mode,
        "time_steps": time_steps,
        "capacity": capacity,
        "rows": rows,
        "finite_difference": finite_difference,
        "summary": {
            "all_valid": all(row["valid"] for row in rows),
            "minimum_weight": min((row["minimum_weight"] for row in rows if row["minimum_weight"] is not None), default=None),
            "weakest_point": min(
                (row for row in rows if row["minimum_weight"] is not None),
                key=lambda row: (row["minimum_weight"], row["weakest_time_zero_based"]),
                default=None,
            )["name"] if time_steps > 1 else None,
        },
        "timing": {
            "trace_seconds": trace_seconds,
            "first_compile_and_evaluation_seconds": compile_seconds,
            "same_point_warm_replay_seconds": warm_replay_seconds,
            "same_point_warm_replay_bitwise_identity": warm_replay_identity,
            "subsequent_warm_evaluation_seconds_total": warm_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "graph_topology": graph,
        "source_guard": source_audit,
        "output_devices": output_devices,
        "gpu_allocator": gpu_memory,
        "cpu_gpu_comparison": cpu_comparison,
        "certification_gates": certification_gates,
        "execution": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(git_status),
            "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
            "command": " ".join(sys.argv),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "tensorflow_version": tf.__version__,
            "device": ARGS.device,
            "physical_gpus": [item.name for item in physical_gpus],
            "logical_gpus": [item.name for item in logical_gpus],
            "logical_device_memory_limit_mib": ARGS.gpu_memory_limit_mib,
            "memory_growth": False,
            "jit_compile": not ARGS.no_jit_compile,
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "dtype": "float64",
            "trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
                if ARGS.device == "gpu"
                else "deliberate_CPU_hidden_reference"
            ),
            "plan": preparation["plan"],
            "output": str(output.relative_to(ROOT)),
        },
        "derivative_status": (
            "manual_total_jvp_previously_certified_by_Phase_5_forward_AD_and_FD"
            if ARGS.evaluation_mode == "score" and ARGS.device == "gpu"
            else "manual_total_jvp_with_autodiff_oracle"
            if ARGS.evaluation_mode == "score" and ARGS.device == "cpu"
            else "not_evaluated_chart_screen_only"
        ),
        "nonclaims": [
            "not manual score certification",
            "not finite-difference certification",
            "not scientific score equivalence",
            "CPU/GPU differences are descriptive only",
            "not HMC or canonical Contract E readiness",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": payload["summary"], "timing": payload["timing"]}, indent=2))
    if ARGS.device == "gpu":
        return 0 if all(certification_gates.values()) else 2
    return 0 if payload["summary"]["all_valid"] else 2


def _write_failure_artifact(error: BaseException) -> None:
    output = _path(ARGS.output)
    if output.exists():
        return
    payload = {
        "schema": SCHEMA,
        "status": "FAIL_CATCHABLE_RUNTIME_OR_TENSORFLOW_EXCEPTION",
        "phase": 7 if ARGS.device == "gpu" else None,
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "classification": (
                "catchable_infrastructure_or_runtime_failure_not_scientific_evidence"
            ),
        },
        "execution": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "device": ARGS.device,
            "logical_device_memory_limit_mib": ARGS.gpu_memory_limit_mib,
            "memory_growth": False,
            "jit_compile": not ARGS.no_jit_compile,
            "physical_gpus": [item.name for item in physical_gpus],
            "logical_gpus": [item.name for item in logical_gpus],
            "output": str(output.relative_to(ROOT)),
        },
        "nonclaims": [
            "a catchable runtime failure is not evidence against the mathematical candidate",
            "native process aborts cannot be serialized by this handler",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        _write_failure_artifact(error)
        raise

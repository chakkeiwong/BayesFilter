#!/usr/bin/env python3
"""Run one center-scoped scalar-SV full-horizon CPU or trusted-GPU XLA row."""

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


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--phase3-short-result", type=Path, required=True)
    parser.add_argument("--cpu-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse()
if ARGS.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU mode requires CUDA_VISIBLE_DEVICES=-1 before Python")
if ARGS.device == "gpu":
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise RuntimeError("GPU mode requires TF_FORCE_GPU_ALLOW_GROWTH=true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)

import tensorflow as tf  # noqa: E402


GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf, require_gpu=ARGS.device == "gpu"
)

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model  # noqa: E402
from bayesfilter.highdim.ledh_forward_contract import (  # noqa: E402
    ACTUAL_SV_ROW_ID,
    KSC_SV_ROW_ID,
)
from bayesfilter.ledh_fd_policy import evaluate_ledh_fd_policy  # noqa: E402
from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import (  # noqa: E402
    LoopRole,
    SourceRouteSpec,
    audit_source_path,
    inventory_graph_def,
)


DTYPE = tf.float64
TIME_STEPS = 1000
SHORT_TIME_STEPS = 100
FD_STEP = 1.0e-5
INVALID_THETA_GAMMA = 4.0
PARAMETER_NAMES = ("gamma_unconstrained", "log_beta")
PLAN = (
    "docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-"
    "phase4-scalar-sv-gpu-xla-subplan-2026-07-15.md"
)
RESULT = (
    "docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-"
    "phase4-scalar-sv-gpu-xla-result-2026-07-15.md"
)


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _git_status() -> dict[str, Any]:
    text = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    return {
        "dirty": bool(text),
        "entry_count": len(text.splitlines()),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_preparation(
    path: Path, *, expected_time_steps: int = TIME_STEPS
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema") != "bayesfilter.contract_e_tp.scalar_sv_preparation.v1":
        raise ValueError("Phase 4 requires a fixed-square scalar-SV v1 preparation")
    if payload.get("row_id") not in (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID):
        raise ValueError("Phase 4 supports only Actual SV and KSC-SV")
    if payload["target"]["time_steps"] != expected_time_steps:
        raise ValueError(
            f"Phase 4 expected T={expected_time_steps} preparation at {path}"
        )
    if payload["chart_contract"]["mode"] != "fixed_square":
        raise ValueError("Phase 4 requires a fixed-square chart")
    if payload["target"]["transition_before_first_observation"] is not False:
        raise ValueError("wrong scalar-SV initial-observation time order")
    return payload


def _bound_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    time_steps = int(payload["target"]["time_steps"])
    return {
        "theta": tf.constant(payload["target"]["theta"], DTYPE),
        "target": tf.constant(payload["target"]["target_observations"], DTYPE),
        "flow": tf.constant(payload["target"]["flow_observations"], DTYPE),
        "nodes": tf.constant(payload["teacher_quadrature"]["nodes"], DTYPE),
        "weights": tf.constant(payload["teacher_quadrature"]["weights"], DTYPE),
        "active": tf.reshape(
            tf.constant(payload["active_indices"], tf.int32),
            [time_steps - 1, model.FEATURE_COUNT],
        ),
        "scales": tf.reshape(
            tf.constant(payload["row_scales"], DTYPE),
            [time_steps - 1, model.FEATURE_COUNT],
        ),
        "grid": tf.constant(payload["continuation_quadrature"]["points"], DTYPE),
        "grid_weights": tf.constant(
            payload["continuation_quadrature"]["weights"], DTYPE
        ),
        "lookahead": int(payload["feature_contract"]["lookahead_steps"]),
    }


def _factory(spec: model.ScalarSVContractETPSpec, bound: dict[str, Any]):
    return model.make_contract_e_tp_scalar_sv_loop_tf(
        spec,
        bound["target"],
        bound["flow"],
        bound["nodes"],
        bound["weights"],
        bound["active"],
        bound["scales"],
        bound["grid"],
        bound["grid_weights"],
        lookahead_steps=bound["lookahead"],
        jit_compile=True,
    )


def _source_audits() -> dict[str, Any]:
    path = ROOT / "bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py"
    clean = audit_source_path(
        path,
        SourceRouteSpec(
            roots=("make_contract_e_tp_scalar_sv_loop_tf",),
            loop_roles={},
            required_reachable=(
                "contract_e_tp_scalar_sv_loop_core",
                "contract_e_tp_scalar_sv_loop_core.cond",
                "contract_e_tp_scalar_sv_loop_core.body",
                "target_continuation_log_likelihood_loop.cond",
                "target_continuation_log_likelihood_loop.body",
                "make_contract_e_tp_scalar_sv_loop_tf.evaluate.poison",
            ),
        ),
    )
    historical = audit_source_path(
        path,
        SourceRouteSpec(
            roots=("contract_e_tp_scalar_sv_recursive_core",),
            loop_roles={
                "contract_e_tp_scalar_sv_recursive_core": LoopRole(
                    "filter_time_python_unroll", True
                ),
                "target_continuation_log_likelihood": LoopRole(
                    "backward_continuation_python_unroll", True
                ),
            },
            required_reachable=(
                "contract_e_tp_scalar_sv_recursive_core",
                "target_continuation_log_likelihood",
            ),
        ),
    )
    return {"clean_factory": clean, "historical_unrolled": historical}


def _all_finite(result: dict[str, tf.Tensor]) -> bool:
    return all(
        bool(tf.reduce_all(tf.math.is_finite(result[name])).numpy())
        for name in (
            "objective",
            "score",
            "increment_history",
            "final_particles",
            "final_log_unnormalized_weights",
        )
    )


def _array_evidence(value: tf.Tensor) -> dict[str, Any]:
    array = value.numpy()
    return {
        "shape": list(array.shape),
        "values": float(array) if array.ndim == 0 else array.tolist(),
        "serialized_tensor_sha256": _tensor_sha256(value),
    }


def _difference(left: tf.Tensor, right: tf.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": left.shape.as_list(),
            "right_shape": right.shape.as_list(),
        }
    absolute = tf.abs(left - right)
    symmetric = absolute / tf.maximum(
        tf.maximum(tf.abs(left), tf.abs(right)), tf.cast(1.0e-12, left.dtype)
    )
    return {
        "shape_equal": True,
        "max_absolute_difference": float(tf.reduce_max(absolute).numpy()),
        "max_symmetric_relative_difference": float(tf.reduce_max(symmetric).numpy()),
    }


PREPARATION_PATH = _path(ARGS.preparation)
PREPARATION = _load_preparation(PREPARATION_PATH)


def _cpu_comparison(
    path: Path | None, outputs: dict[str, tf.Tensor]
) -> dict[str, Any]:
    if path is None:
        return {"status": "not_applicable_cpu_reference_run"}
    payload = _load_json(path)
    if payload.get("row_id") != PREPARATION["row_id"]:
        raise ValueError("CPU result row identity differs from GPU row")
    if payload["preparation"]["sha256"] != _sha256(PREPARATION_PATH):
        raise ValueError("CPU/GPU preparation identity differs")
    comparison = {}
    for name in (
        "objective",
        "score",
        "increment_history",
        "final_particles",
        "final_log_unnormalized_weights",
    ):
        left = outputs[name]
        right = tf.constant(payload["outputs"][name]["values"], dtype=left.dtype)
        comparison[name] = _difference(left, right)
        comparison[name]["cpu_serialized_tensor_sha256"] = payload["outputs"][name][
            "serialized_tensor_sha256"
        ]
        comparison[name]["current_serialized_tensor_sha256"] = _tensor_sha256(left)
    return {
        "status": "descriptive_only_no_equivalence_or_magnitude_gate",
        "fields": comparison,
    }


def main() -> int:
    output = _path(ARGS.output)
    short_path = _path(ARGS.phase3_short_result)
    cpu_path = _path(ARGS.cpu_result) if ARGS.cpu_result else None
    if output.exists():
        raise FileExistsError(output)
    if ARGS.device == "gpu" and cpu_path is None:
        raise ValueError("GPU mode requires --cpu-result")
    if ARGS.device == "cpu" and cpu_path is not None:
        raise ValueError("CPU mode must not provide --cpu-result")

    started_all = time.perf_counter()
    spec = model.make_scalar_sv_spec(PREPARATION["row_id"])
    bound = _bound_inputs(PREPARATION)
    factory = _factory(spec, bound)
    graph = inventory_graph_def(factory.get_concrete_function().graph.as_graph_def())
    short = _load_json(short_path)
    short_rungs = {item["time_steps"]: item for item in short["rungs"]}
    if short["row_id"] != PREPARATION["row_id"]:
        raise ValueError("Phase 3 short result row identity differs")
    short_preparation_path = _path(
        Path(short_rungs[SHORT_TIME_STEPS]["preparation"]["path"])
    )
    if _sha256(short_preparation_path) != short_rungs[SHORT_TIME_STEPS][
        "preparation"
    ]["sha256"]:
        raise ValueError("Phase 3 short preparation hash differs from its result")
    short_preparation = _load_preparation(
        short_preparation_path, expected_time_steps=SHORT_TIME_STEPS
    )
    if short_preparation["row_id"] != PREPARATION["row_id"]:
        raise ValueError("Phase 3 short preparation row identity differs")
    short_factory = _factory(spec, _bound_inputs(short_preparation))
    short_graph = inventory_graph_def(
        short_factory.get_concrete_function().graph.as_graph_def()
    )
    ratios = {
        "top_level_nodes_t1000_t100": (
            graph["top_level_nodes"] / short_graph["top_level_nodes"]
        ),
        "function_nodes_t1000_t100": (
            graph["function_nodes"] / short_graph["function_nodes"]
        ),
        "graphdef_bytes_t1000_t100": (
            graph["graphdef_bytes"] / short_graph["graphdef_bytes"]
        ),
    }
    graph_gates = {
        "t100_and_t1000_have_functional_loops": (
            short_graph["functional_loop_count"] >= 1
            and graph["functional_loop_count"] >= 1
        ),
        "top_level_ratio_at_most_1_10": (
            ratios["top_level_nodes_t1000_t100"] <= 1.10
        ),
        "function_ratio_at_most_1_10": (
            ratios["function_nodes_t1000_t100"] <= 1.10
        ),
        "graphdef_ratio_at_most_1_25": (
            ratios["graphdef_bytes_t1000_t100"] <= 1.25
        ),
    }

    if ARGS.device == "gpu":
        logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
        if not logical_gpus:
            raise RuntimeError("trusted GPU mode requires a visible logical GPU")
        execution_device = "/GPU:0"
        tf.config.experimental.reset_memory_stats("GPU:0")
    else:
        if tf.config.list_logical_devices("GPU"):
            raise RuntimeError("CPU mode unexpectedly exposes a logical GPU")
        execution_device = "/CPU:0"

    theta = bound["theta"]
    with tf.device(execution_device):
        started = time.perf_counter()
        center = factory(theta)
        compile_first_seconds = time.perf_counter() - started
        started = time.perf_counter()
        warm = factory(theta)
        warm_seconds = time.perf_counter() - started
        endpoints = []
        finite_difference = []
        for index in range(spec.parameter_dimension):
            direction = tf.one_hot(index, spec.parameter_dimension, dtype=DTYPE)
            plus = factory(theta + FD_STEP * direction)
            minus = factory(theta - FD_STEP * direction)
            endpoints.append(
                {
                    "coordinate": index,
                    "plus_valid": bool(plus["valid"].numpy()),
                    "minus_valid": bool(minus["valid"].numpy()),
                    "plus_objective": float(plus["objective"].numpy()),
                    "minus_objective": float(minus["objective"].numpy()),
                }
            )
            finite_difference.append(
                float(
                    (
                        (plus["objective"] - minus["objective"])
                        / (2.0 * FD_STEP)
                    ).numpy()
                )
            )
        invalid_theta = tf.stack(
            [tf.constant(INVALID_THETA_GAMMA, DTYPE), theta[1]]
        )
        invalid = factory(invalid_theta)

    if all(math.isfinite(value) for value in finite_difference):
        fd_policy = evaluate_ledh_fd_policy(
            center["score"].numpy().tolist(), finite_difference, PARAMETER_NAMES
        )
    else:
        fd_policy = {
            "policy_id": "owner_directed_individual_direction_relative_5pct_sqrt_p_v1",
            "diagnostic_scope": "finite_difference_only",
            "status": "fail",
            "reason": "one_or_more_exact_factory_endpoint_objectives_nonfinite",
            "num_parameters": spec.parameter_dimension,
            "max_coordinate_relative_error_threshold": 0.05
            * math.sqrt(spec.parameter_dimension),
            "nonfinite_coordinates": [
                index
                for index, value in enumerate(finite_difference)
                if not math.isfinite(value)
            ],
        }
    replay_equal = all(
        bool(tf.reduce_all(tf.equal(center[name], warm[name])).numpy())
        for name in (
            "objective",
            "score",
            "valid",
            "increment_history",
            "final_particles",
            "final_log_unnormalized_weights",
        )
    )
    invalid_fields = {
        name: bool(tf.reduce_all(tf.math.is_finite(invalid[name])).numpy())
        for name in (
            "objective",
            "score",
            "increment_history",
            "final_particles",
            "final_log_unnormalized_weights",
        )
    }
    invalid_gate = not bool(invalid["valid"].numpy()) and not any(
        invalid_fields.values()
    )
    source_audits = _source_audits()
    output_device = center["objective"].device
    placement_pass = (
        "GPU:0" in output_device.upper()
        if ARGS.device == "gpu"
        else "CPU:0" in output_device.upper()
    )
    gates = {
        "target_and_preparation_identity": True,
        "clean_source_approved": source_audits["clean_factory"]["approved"],
        "historical_unrolled_rejected": not source_audits["historical_unrolled"][
            "approved"
        ],
        "graph_topology": all(graph_gates.values()),
        "center_valid_and_finite": bool(center["valid"].numpy())
        and _all_finite(center),
        "fd_endpoints_valid": all(
            item["plus_valid"] and item["minus_valid"] for item in endpoints
        ),
        "same_scalar_fd": fd_policy["status"] == "pass",
        "warm_replay_equal": replay_equal,
        "same_factory_invalid_fail_closed": invalid_gate,
        "requested_device_placement": placement_pass,
    }
    outputs = {
        name: _array_evidence(center[name])
        for name in (
            "objective",
            "score",
            "increment_history",
            "final_particles",
            "final_log_unnormalized_weights",
        )
    }
    memory = (
        tf.config.experimental.get_memory_info("GPU:0")
        if ARGS.device == "gpu"
        else {"current": 0, "peak": 0}
    )
    comparison = _cpu_comparison(cpu_path, center)
    status = (
        "PASS_SCALAR_SV_FULL_HORIZON_TRUSTED_GPU_XLA"
        if ARGS.device == "gpu" and all(gates.values())
        else (
            "PASS_SCALAR_SV_FULL_HORIZON_CPU_XLA_PREFLIGHT"
            if ARGS.device == "cpu" and all(gates.values())
            else "FAIL_SCALAR_SV_FULL_HORIZON_XLA"
        )
    )
    payload = {
        "schema": "bayesfilter.contract_e_tp.clean_xla_phase4_scalar_sv.v1",
        "status": status,
        "row_id": PREPARATION["row_id"],
        "algorithm_id": model.ALGORITHM_ID,
        "scope": "center_only_full_horizon_xla_engineering",
        "preparation": {
            "path": str(PREPARATION_PATH.relative_to(ROOT)),
            "sha256": _sha256(PREPARATION_PATH),
            "target_observations_sha256": PREPARATION["target"][
                "target_observations_sha256"
            ],
            "flow_observations_sha256": PREPARATION["target"][
                "flow_observations_sha256"
            ],
            "teacher_order": PREPARATION["teacher_quadrature"]["order"],
            "continuation_order": PREPARATION["continuation_quadrature"]["order"],
            "continuation_radius": PREPARATION["continuation_quadrature"]["radius"],
            "lookahead_steps": bound["lookahead"],
            "time_steps": TIME_STEPS,
        },
        "theta": theta.numpy().tolist(),
        "outputs": outputs,
        "finite_difference": finite_difference,
        "finite_difference_step": FD_STEP,
        "finite_difference_endpoints": endpoints,
        "finite_difference_policy": fd_policy,
        "same_factory_invalid_control": {
            "theta": invalid_theta.numpy().tolist(),
            "valid": bool(invalid["valid"].numpy()),
            "finite_fields": invalid_fields,
        },
        "graph": graph,
        "short_graph": short_graph,
        "graph_ratios": ratios,
        "graph_gates": graph_gates,
        "source_audits": source_audits,
        "timing_seconds": {
            "compile_and_first": compile_first_seconds,
            "warm": warm_seconds,
        },
        "cpu_gpu_comparison": comparison,
        "gates": gates,
        "run_manifest": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_status": _git_status(),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "tensorflow_version": tf.__version__,
            "physical_devices": [
                str(item) for item in tf.config.list_physical_devices()
            ],
            "logical_devices": [
                str(item) for item in tf.config.list_logical_devices()
            ],
            "output_device": output_device,
            "device_trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
                if ARGS.device == "gpu"
                else "intentional_cpu_hidden_reference"
            ),
            "gpu_memory_policy": GPU_MEMORY_POLICY,
            "gpu_allocator_current_bytes": int(memory["current"]),
            "gpu_allocator_peak_bytes": int(memory["peak"]),
            "jit_compile": True,
            "tf32": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "dtype": DTYPE.name,
            "data_version": PREPARATION["target"]["raw_observations_sha256"],
            "seed": "81101 deterministic dataset",
            "wall_time_seconds": time.perf_counter() - started_all,
            "attempt_number": output.parent.name,
            "output_path": str(output.relative_to(ROOT)),
            "plan": PLAN,
            "result": RESULT,
        },
        "decision": {
            "hard_veto_screen_pass": all(gates.values()),
            "candidate_remains_viable": all(gates.values()),
            "statistically_supported_ranking": False,
            "default_readiness": False,
            "next_evidence": "Phase 4 close then reviewed predator-prey Phase 5",
        },
        "nonclaims": [
            "not a nonzero-radius parameter-region or HMC result",
            "not nonlinear filtering accuracy or CPU/GPU equivalence",
            "not statistically supported superiority",
            "not canonical, default, leaderboard, or float32/TF32 production readiness",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "row_id": payload["row_id"],
                "gates": gates,
                "timing_seconds": payload["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

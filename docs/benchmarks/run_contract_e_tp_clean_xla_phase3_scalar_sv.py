#!/usr/bin/env python3
"""Certify one scalar-SV row's loop-native CPU/XLA prefix ladder."""

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
    parser.add_argument("--device", choices=("cpu",), required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--preparation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse()
if ARGS.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU mode requires CUDA_VISIBLE_DEVICES=-1 before Python")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model
from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    KSC_SV_ROW_ID,
)
from bayesfilter.ledh_fd_policy import evaluate_ledh_fd_policy
from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import (
    LoopRole,
    SourceRouteSpec,
    audit_source_path,
    inventory_graph_def,
)


DTYPE = tf.float64
EXPECTED_TIMES = (1, 2, 3, 10, 100)
PARAMETER_NAMES = ("gamma_unconstrained", "log_beta")
FD_STEP = 1.0e-5
INVALID_THETA_GAMMA = 4.0
PLAN = (
    "docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-"
    "phase3-scalar-sv-loop-core-subplan-2026-07-15.md"
)
RESULT = (
    "docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-"
    "phase3-scalar-sv-loop-core-result-2026-07-15.md"
)


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _maximum_absolute_difference(left: tf.Tensor, right: tf.Tensor) -> float:
    if tf.size(left) == 0:
        return 0.0
    return float(tf.reduce_max(tf.abs(left - right)).numpy())


def _all_close(left: tf.Tensor, right: tf.Tensor, *, rtol: float, atol: float) -> bool:
    scale = atol + rtol * tf.abs(right)
    return bool(tf.reduce_all(tf.abs(left - right) <= scale).numpy())


def _value_and_score(callable_, theta: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        result = callable_(theta)
    return result, tape.gradient(result["objective"], theta)


def _load_preparation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != "bayesfilter.contract_e_tp.scalar_sv_preparation.v1":
        raise ValueError(f"{path}: Phase 3 requires a fixed-square v1 preparation")
    if payload["row_id"] != ARGS.row_id:
        raise ValueError(f"{path}: wrong row id")
    if payload["chart_contract"]["mode"] != "fixed_square":
        raise ValueError(f"{path}: wrong chart mode")
    if payload["target"]["transition_before_first_observation"] is not False:
        raise ValueError(f"{path}: wrong initial-observation time order")
    return payload


def _bound_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    time_steps = int(payload["target"]["time_steps"])
    return {
        "time_steps": time_steps,
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


def _unrolled_callable(spec, bound: dict[str, Any]):
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        return model.contract_e_tp_scalar_sv_recursive_core(
            spec,
            theta,
            bound["target"],
            bound["flow"],
            bound["nodes"],
            bound["weights"],
            bound["active"],
            bound["scales"],
            bound["grid"],
            bound["grid_weights"],
            lookahead_steps=bound["lookahead"],
        )

    return evaluate


def _loop_callable(spec, bound: dict[str, Any]):
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        return model.contract_e_tp_scalar_sv_loop_core(
            spec,
            theta,
            bound["target"],
            bound["flow"],
            bound["nodes"],
            bound["weights"],
            bound["active"],
            bound["scales"],
            bound["grid"],
            bound["grid_weights"],
            lookahead_steps=bound["lookahead"],
        )

    return evaluate


def _compiled_factory(spec, bound: dict[str, Any]):
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


def _rung(spec, path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    bound = _bound_inputs(payload)
    theta = bound["theta"]
    unrolled, unrolled_score = _value_and_score(_unrolled_callable(spec, bound), theta)
    loop, loop_score = _value_and_score(_loop_callable(spec, bound), theta)
    factory = _compiled_factory(spec, bound)
    graph = inventory_graph_def(factory.get_concrete_function().graph.as_graph_def())

    started = time.perf_counter()
    compiled = factory(theta)
    compile_and_first = time.perf_counter() - started
    started = time.perf_counter()
    compiled_warm = factory(theta)
    warm_seconds = time.perf_counter() - started

    finite_difference = []
    fd_endpoint_valid = []
    for index in range(spec.parameter_dimension):
        direction = tf.one_hot(index, spec.parameter_dimension, dtype=DTYPE)
        plus = factory(theta + FD_STEP * direction)
        minus = factory(theta - FD_STEP * direction)
        fd_endpoint_valid.append(bool(plus["valid"].numpy() and minus["valid"].numpy()))
        finite_difference.append(
            float(((plus["objective"] - minus["objective"]) / (2.0 * FD_STEP)).numpy())
        )
    fd_policy = evaluate_ledh_fd_policy(
        compiled["score"].numpy().tolist(), finite_difference, PARAMETER_NAMES
    )

    parity = {
        "objective_loop_unrolled_abs": _maximum_absolute_difference(
            loop["objective"], unrolled["objective"]
        ),
        "objective_compiled_unrolled_abs": _maximum_absolute_difference(
            compiled["objective"], unrolled["objective"]
        ),
        "score_loop_unrolled_max_abs": _maximum_absolute_difference(
            loop_score, unrolled_score
        ),
        "score_compiled_unrolled_max_abs": _maximum_absolute_difference(
            compiled["score"], unrolled_score
        ),
        "increment_compiled_unrolled_max_abs": _maximum_absolute_difference(
            compiled["increment_history"], unrolled["increment_history"]
        ),
        "final_particles_compiled_unrolled_max_abs": _maximum_absolute_difference(
            compiled["final_particles"], unrolled["final_particles"]
        ),
        "final_log_weights_compiled_unrolled_max_abs": _maximum_absolute_difference(
            compiled["final_log_unnormalized_weights"],
            unrolled["final_log_unnormalized_weights"],
        ),
    }
    parity_pass = all(
        (
            _all_close(compiled[name], unrolled[name], rtol=2.0e-10, atol=2.0e-11)
            and _all_close(loop[name], unrolled[name], rtol=2.0e-12, atol=2.0e-13)
        )
        for name in (
            "objective",
            "increment_history",
            "final_particles",
            "final_log_unnormalized_weights",
        )
    ) and _all_close(compiled["score"], unrolled_score, rtol=5.0e-9, atol=5.0e-10)
    validity_equal = (
        compiled["valid"].numpy().item()
        and bool(tf.reduce_all(tf.equal(loop["valid_history"], unrolled["valid_history"])).numpy())
        and bool(tf.reduce_all(unrolled["valid_history"]).numpy())
    )
    replay_equal = all(
        bool(tf.reduce_all(tf.equal(compiled[name], compiled_warm[name])).numpy())
        for name in ("objective", "score", "valid", "final_particles")
    )
    gates = {
        "target_identity": payload["row_id"] == ARGS.row_id,
        "loop_unrolled_compiled_parity": parity_pass,
        "validity_equal_and_true": validity_equal,
        "same_scalar_fd": fd_policy["status"] == "pass" and all(fd_endpoint_valid),
        "compiled_outputs_finite": all(
            bool(tf.reduce_all(tf.math.is_finite(compiled[name])).numpy())
            for name in (
                "objective",
                "score",
                "final_particles",
                "final_log_unnormalized_weights",
            )
        ),
        "warm_replay_equal": replay_equal,
    }
    result = {
        "time_steps": bound["time_steps"],
        "preparation": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "lookahead_steps": bound["lookahead"],
            "teacher_order": payload["teacher_quadrature"]["order"],
            "continuation_order": payload["continuation_quadrature"]["order"],
            "continuation_radius": payload["continuation_quadrature"]["radius"],
            "target_observations_sha256": payload["target"][
                "target_observations_sha256"
            ],
            "flow_observations_sha256": payload["target"][
                "flow_observations_sha256"
            ],
        },
        "value": float(compiled["objective"].numpy()),
        "score": compiled["score"].numpy().tolist(),
        "finite_difference": finite_difference,
        "finite_difference_step": FD_STEP,
        "finite_difference_policy": fd_policy,
        "parity": parity,
        "graph": graph,
        "timing_seconds": {
            "compile_and_first": compile_and_first,
            "warm": warm_seconds,
        },
        "gates": gates,
    }
    return result, factory


def main() -> int:
    output = _path(ARGS.output)
    if output.exists():
        raise FileExistsError(output)
    if ARGS.row_id not in (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID):
        raise ValueError("Phase 3 supports only actual SV and KSC-SV")
    paths = [_path(path) for path in ARGS.preparation]
    payloads = [_load_preparation(path) for path in paths]
    by_time = {
        int(payload["target"]["time_steps"]): (path, payload)
        for path, payload in zip(paths, payloads, strict=True)
    }
    if tuple(sorted(by_time)) != EXPECTED_TIMES or len(paths) != len(EXPECTED_TIMES):
        raise ValueError(f"preparations must bind exactly T={EXPECTED_TIMES}")
    spec = model.make_scalar_sv_spec(ARGS.row_id)
    source_audits = _source_audits()
    started_all = time.perf_counter()
    rungs = []
    factories = {}
    for time_steps in EXPECTED_TIMES:
        rung, factory = _rung(spec, *by_time[time_steps])
        rungs.append(rung)
        factories[time_steps] = factory

    graph_by_time = {rung["time_steps"]: rung["graph"] for rung in rungs}
    graph_ratios = {
        "top_level_nodes_t100_t3": graph_by_time[100]["top_level_nodes"]
        / graph_by_time[3]["top_level_nodes"],
        "function_nodes_t100_t3": graph_by_time[100]["function_nodes"]
        / graph_by_time[3]["function_nodes"],
        "graphdef_bytes_t100_t3": graph_by_time[100]["graphdef_bytes"]
        / graph_by_time[3]["graphdef_bytes"],
    }
    graph_gates = {
        "t3_and_t100_have_functional_loops": all(
            graph_by_time[value]["functional_loop_count"] >= 1 for value in (3, 100)
        ),
        "top_level_ratio_at_most_1_10": graph_ratios[
            "top_level_nodes_t100_t3"
        ]
        <= 1.10,
        "function_ratio_at_most_1_10": graph_ratios["function_nodes_t100_t3"]
        <= 1.10,
        "graphdef_ratio_at_most_1_25": graph_ratios["graphdef_bytes_t100_t3"]
        <= 1.25,
    }

    center = _bound_inputs(by_time[3][1])["theta"]
    invalid_theta = tf.stack(
        [tf.constant(INVALID_THETA_GAMMA, DTYPE), center[1]]
    )
    invalid = factories[3](invalid_theta)
    invalid_summary = {
        "theta": invalid_theta.numpy().tolist(),
        "valid": bool(invalid["valid"].numpy()),
        "objective_finite": bool(tf.math.is_finite(invalid["objective"]).numpy()),
        "score_finite": bool(tf.reduce_all(tf.math.is_finite(invalid["score"])).numpy()),
        "final_particles_finite": bool(
            tf.reduce_all(tf.math.is_finite(invalid["final_particles"])).numpy()
        ),
        "final_log_weights_finite": bool(
            tf.reduce_all(
                tf.math.is_finite(invalid["final_log_unnormalized_weights"])
            ).numpy()
        ),
    }
    invalid_gate = (
        not invalid_summary["valid"]
        and not invalid_summary["objective_finite"]
        and not invalid_summary["score_finite"]
        and not invalid_summary["final_particles_finite"]
        and not invalid_summary["final_log_weights_finite"]
    )
    all_gates = {
        "clean_source_approved": source_audits["clean_factory"]["approved"],
        "historical_unrolled_rejected": not source_audits["historical_unrolled"][
            "approved"
        ],
        "all_rungs_pass": all(all(rung["gates"].values()) for rung in rungs),
        "graph_topology_pass": all(graph_gates.values()),
        "same_factory_invalid_fail_closed": invalid_gate,
    }
    status = (
        "PASS_SCALAR_SV_LOOP_NATIVE_CPU_XLA_PREFIX"
        if all(all_gates.values())
        else "FAIL_SCALAR_SV_LOOP_NATIVE_CPU_XLA_PREFIX"
    )
    payload = {
        "schema": "bayesfilter.contract_e_tp.clean_xla_phase3_scalar_sv.v1",
        "status": status,
        "row_id": ARGS.row_id,
        "algorithm_id": model.ALGORITHM_ID,
        "scope": "center_only_prefix_loop_native_cpu_xla_engineering",
        "target_identity": {
            "parameter_names": list(PARAMETER_NAMES),
            "target_observation_policy": payloads[0]["target"][
                "target_observation_policy"
            ],
            "flow_observation_policy": payloads[0]["target"][
                "flow_observation_policy"
            ],
            "transition_before_first_observation": False,
        },
        "rungs": rungs,
        "source_audits": source_audits,
        "graph_ratios": graph_ratios,
        "graph_gates": graph_gates,
        "same_factory_invalid_control": invalid_summary,
        "gates": all_gates,
        "decision": {
            "hard_veto_screen_pass": all(all_gates.values()),
            "candidate_remains_viable_for_phase4": all(all_gates.values()),
            "statistically_supported_ranking": False,
            "default_readiness": False,
            "next_evidence": "row-specific trusted full-horizon GPU/XLA Phase 4",
        },
        "parameter_region_audit": {
            "status": "center_scoped_only_no_reviewed_region_certificate",
            "center_preparation": rungs[-1]["preparation"]["path"],
            "same_scalar_fd_endpoints_valid": rungs[-1]["gates"]["same_scalar_fd"],
            "finite_off_center_invalid_control": invalid_summary,
            "full_box_hmc_readiness": "deferred",
            "forbidden_inference": (
                "center and FD-endpoint validity do not imply any nonzero-radius "
                "parameter-region certificate"
            ),
        },
        "run_manifest": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_status_short": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "tensorflow_version": tf.__version__,
            "physical_devices": [str(device) for device in tf.config.list_physical_devices()],
            "visible_gpus": [str(device) for device in tf.config.list_logical_devices("GPU")],
            "device_trust_basis": "cpu_hidden_reference_debug",
            "jit_compile": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "dtype": DTYPE.name,
            "seed": "81101 deterministic dataset",
            "wall_time_seconds": time.perf_counter() - started_all,
            "output_path": str(output.relative_to(ROOT)),
            "plan": PLAN,
            "result": RESULT,
        },
        "nonclaims": [
            "not a T=1000 or trusted GPU result",
            "not nonlinear filtering accuracy or cross-method equivalence",
            "not statistically supported superiority",
            "not canonical, default, HMC, or leaderboard readiness",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "row_id": ARGS.row_id,
                "graph_ratios": graph_ratios,
                "gates": all_gates,
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if all(all_gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

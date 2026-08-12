#!/usr/bin/env python3
"""Tune/freeze the persistent-guide Austria proposal viability scope."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_limit,
)


PLAN = "docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-plan-2026-08-02.md"
SCHEMA = "bayesfilter.zhao_cui_austria_sir_persistent_guide_tuning.v1"
MEMORY_LIMIT_MIB = 6144
ESS_FRACTION_FLOOR = 0.10
MAXIMUM_WEIGHT_CEILING = 0.10
DOMAIN_HALF_WIDTH_LADDER = (0.03, 0.10, 0.25, 0.50)
FORBIDDEN_GRAPH_OPS = frozenset(
    {"PyFunc", "PyFuncStateless", "EagerPyFunc", "MapDefun"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    return hashlib.sha256(bytes(serialized.numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        raw = value.numpy()
        return raw.item() if value.shape.rank == 0 else raw.tolist()
    return value


def _git_payload() -> Mapping[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def _domain_design() -> tf.Tensor:
    """Unit-radius origin, signed axes, corners, and mixed interiors."""

    signs = tf.constant(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        tf.float64,
    )
    axes = tf.concat([tf.eye(3, dtype=tf.float64), -tf.eye(3, dtype=tf.float64)], axis=0)
    mixed = tf.constant(
        [
            [0.5, -0.25, 0.75],
            [-0.5, 0.25, -0.75],
            [0.75, 0.5, -0.25],
            [-0.75, -0.5, 0.25],
            [0.25, 0.75, 0.5],
            [-0.25, -0.75, -0.5],
            [0.6, -0.7, -0.4],
            [-0.6, 0.7, 0.4],
        ],
        tf.float64,
    )
    return tf.concat([tf.zeros([1, 3], tf.float64), axes, signs, mixed], axis=0)


def _graph_audit(
    concrete: tf.types.experimental.ConcreteFunction,
) -> Mapping[str, object]:
    graph_def = concrete.graph.as_graph_def()
    operation_types = {node.op for node in graph_def.node} | {
        node.op
        for function in graph_def.library.function
        for node in function.node_def
    }
    forbidden = tuple(sorted(operation_types.intersection(FORBIDDEN_GRAPH_OPS)))
    while_ops = tuple(
        sorted(operation_types.intersection(("While", "StatelessWhile")))
    )
    return {
        "operation_types": tuple(sorted(operation_types)),
        "while_operations": while_ops,
        "forbidden_operations": forbidden,
        "has_while": bool(while_ops),
        "has_host_callback": bool(forbidden),
    }


def _make_point_sweep(program, point_count: int):
    evaluator = program.compiled()
    guide_count = program.guide_count

    @tf.function(
        input_signature=(tf.TensorSpec([point_count, 3], tf.float64),),
        jit_compile=True,
        autograph=False,
    )
    def evaluate_points(theta_points: tf.Tensor) -> Mapping[str, tf.Tensor]:
        values = tf.TensorArray(
            tf.float64, size=point_count, element_shape=tf.TensorShape([])
        )
        scores = tf.TensorArray(
            tf.float64, size=point_count, element_shape=tf.TensorShape([3])
        )
        viable = tf.TensorArray(
            tf.bool, size=point_count, element_shape=tf.TensorShape([guide_count])
        )
        finite = tf.TensorArray(
            tf.bool, size=point_count, element_shape=tf.TensorShape([])
        )
        all_ess_fraction = tf.TensorArray(
            tf.float64,
            size=point_count,
            element_shape=tf.TensorShape([guide_count]),
        )
        all_maximum_weight = tf.TensorArray(
            tf.float64,
            size=point_count,
            element_shape=tf.TensorShape([guide_count]),
        )
        best_ess_fraction = tf.TensorArray(
            tf.float64, size=point_count, element_shape=tf.TensorShape([])
        )
        best_maximum_weight = tf.TensorArray(
            tf.float64, size=point_count, element_shape=tf.TensorShape([])
        )
        branch_effective_count = tf.TensorArray(
            tf.float64, size=point_count, element_shape=tf.TensorShape([])
        )
        maximum_combination_weight = tf.TensorArray(
            tf.float64, size=point_count, element_shape=tf.TensorShape([])
        )

        def body(
            index: tf.Tensor,
            value_array: tf.TensorArray,
            score_array: tf.TensorArray,
            viable_array: tf.TensorArray,
            finite_array: tf.TensorArray,
            all_ess_array: tf.TensorArray,
            all_maximum_weight_array: tf.TensorArray,
            ess_array: tf.TensorArray,
            maximum_weight_array: tf.TensorArray,
            branch_effective_array: tf.TensorArray,
            combination_array: tf.TensorArray,
        ) -> tuple[object, ...]:
            result = evaluator(tf.gather(theta_points, index))
            minimum_ess_fraction = tf.reduce_min(
                result["ess_by_time_and_guide"], axis=0
            ) / tf.cast(program.particle_count, tf.float64)
            maximum_weight = tf.reduce_max(
                result["maximum_weight_by_time_and_guide"], axis=0
            )
            current_viable = (
                minimum_ess_fraction >= ESS_FRACTION_FLOOR
            ) & (maximum_weight <= MAXIMUM_WEIGHT_CEILING)
            current_best_maximum_weight = tf.reduce_min(
                tf.where(
                    current_viable,
                    maximum_weight,
                    tf.fill(
                        [guide_count], tf.constant(float("inf"), tf.float64)
                    ),
                )
            )
            return (
                index + 1,
                value_array.write(index, result["log_likelihood"]),
                score_array.write(index, result["score"]),
                viable_array.write(index, current_viable),
                finite_array.write(index, result["finite"]),
                all_ess_array.write(index, minimum_ess_fraction),
                all_maximum_weight_array.write(index, maximum_weight),
                ess_array.write(index, tf.reduce_max(minimum_ess_fraction)),
                maximum_weight_array.write(index, current_best_maximum_weight),
                branch_effective_array.write(
                    index, result["branch_effective_count"]
                ),
                combination_array.write(
                    index, tf.reduce_max(result["branch_combination_weights"])
                ),
            )

        (
            _,
            values,
            scores,
            viable,
            finite,
            all_ess_fraction,
            all_maximum_weight,
            best_ess_fraction,
            best_maximum_weight,
            branch_effective_count,
            maximum_combination_weight,
        ) = tf.while_loop(
            lambda index, *_unused: index < point_count,
            body,
            (
                tf.zeros([], tf.int32),
                values,
                scores,
                viable,
                finite,
                all_ess_fraction,
                all_maximum_weight,
                best_ess_fraction,
                best_maximum_weight,
                branch_effective_count,
                maximum_combination_weight,
            ),
            maximum_iterations=point_count,
            parallel_iterations=1,
        )
        viable_values = viable.stack()
        finite_values = finite.stack()
        point_pass = finite_values & tf.reduce_any(viable_values, axis=1)
        return {
            "values": values.stack(),
            "scores": scores.stack(),
            "viable_guides": viable_values,
            "finite": finite_values,
            "minimum_ess_fraction_by_guide": all_ess_fraction.stack(),
            "maximum_weight_by_guide": all_maximum_weight.stack(),
            "point_pass": point_pass,
            "all_points_pass": tf.reduce_all(point_pass),
            "best_ess_fraction": best_ess_fraction.stack(),
            "best_maximum_weight": best_maximum_weight.stack(),
            "branch_effective_count": branch_effective_count.stack(),
            "maximum_combination_weight": maximum_combination_weight.stack(),
        }

    return evaluate_points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--particle-count", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=31201)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.horizon not in (1, 2, 3, 5, 10, 20):
        raise ValueError("unsupported staged horizon")
    if args.output_dir.exists():
        raise ValueError("output-dir must be fresh")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    memory = configure_tensorflow_gpu_memory_limit(
        tf, memory_limit_mib=MEMORY_LIMIT_MIB, require_gpu=True
    )
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError("tuning requires one logical GPU")

    from bayesfilter.highdim.zhao_cui_austria_sir_rank_one_proposal_tf import (
        compile_austria_sir_persistent_guide_program,
    )

    started = time.monotonic()
    ladder = tf.constant(DOMAIN_HALF_WIDTH_LADDER, tf.float64)
    ladder_count = len(DOMAIN_HALF_WIDTH_LADDER)
    full_program = compile_austria_sir_persistent_guide_program(
        particle_count=args.particle_count,
        horizon=20,
        seed=args.seed,
        guide_half_width=DOMAIN_HALF_WIDTH_LADDER[0],
    )
    program = full_program.prefix(args.horizon)
    # The guide family is frozen at 0.03. Wider boxes test extrapolation only.
    design = _domain_design()
    point_count = int(design.shape[0])
    theta_points = tf.reshape(
        ladder[:, tf.newaxis, tf.newaxis] * design[tf.newaxis, :, :],
        [ladder_count * point_count, 3],
    )
    point_sweep = _make_point_sweep(program, ladder_count * point_count)
    concrete = point_sweep.get_concrete_function()
    graph = _graph_audit(concrete)
    if not graph["has_while"] or graph["has_host_callback"]:
        raise RuntimeError("outer tuning sweep failed the XLA graph gate")
    all_results = point_sweep(theta_points)
    point_pass = tf.reshape(
        all_results["point_pass"], [ladder_count, point_count]
    )
    passes = tf.reduce_all(point_pass, axis=1)
    best_ess_fraction = tf.reshape(
        all_results["best_ess_fraction"], [ladder_count, point_count]
    )
    best_maximum_weight = tf.reshape(
        all_results["best_maximum_weight"], [ladder_count, point_count]
    )
    branch_effective_count = tf.reshape(
        all_results["branch_effective_count"], [ladder_count, point_count]
    )
    maximum_combination_weight = tf.reshape(
        all_results["maximum_combination_weight"], [ladder_count, point_count]
    )
    ladder_worst_ess = tf.reduce_min(best_ess_fraction, axis=1)
    ladder_worst_maximum_weight = tf.reduce_max(best_maximum_weight, axis=1)
    ladder_minimum_branch_effective_count = tf.reduce_min(
        branch_effective_count, axis=1
    )
    ladder_maximum_combination_weight = tf.reduce_max(
        maximum_combination_weight, axis=1
    )
    passing_indices = tf.where(passes)[:, 0]
    any_pass = tf.size(passing_indices) > 0
    selected_index = tf.cond(
        any_pass,
        lambda: tf.reduce_max(passing_indices),
        lambda: tf.constant(-1, tf.int64),
    )
    selected_half_width = tf.cond(
        any_pass,
        lambda: tf.gather(ladder, selected_index),
        lambda: tf.constant(float("nan"), tf.float64),
    )
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    identity_payload = {
        "schema": SCHEMA,
        "program_id": program.program_id,
        "parent_t20_program_id": full_program.program_id,
        "horizon": args.horizon,
        "particle_count_per_guide": args.particle_count,
        "guide_half_width": 0.03,
        "guide_count": program.guide_count,
        "domain_half_width_ladder": DOMAIN_HALF_WIDTH_LADDER,
        "domain_design": "origin_signed_axes_all_corners_eight_mixed_interiors",
        "ess_fraction_floor": ESS_FRACTION_FLOOR,
        "maximum_weight_ceiling": MAXIMUM_WEIGHT_CEILING,
        "seed": args.seed,
        "plan_sha256": _sha256(ROOT / PLAN),
    }
    tuning_artifact_id = _semantic_hash(identity_payload)
    primary_pass = bool(any_pass.numpy())
    payload = _json_ready(
        {
            **identity_payload,
            "tuning_artifact_id": tuning_artifact_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": (
                f"PASS_T{args.horizon}_PERSISTENT_GUIDE_TUNING"
                if primary_pass
                else f"BLOCK_T{args.horizon}_PERSISTENT_GUIDE_TUNING"
            ),
            "primary_pass": primary_pass,
            "selected_domain_half_width": selected_half_width,
            "ladder_pass": passes,
            "ladder_worst_best_ess_fraction": ladder_worst_ess,
            "ladder_worst_best_maximum_weight": ladder_worst_maximum_weight,
            "ladder_minimum_branch_effective_count": ladder_minimum_branch_effective_count,
            "ladder_maximum_combination_weight": ladder_maximum_combination_weight,
            "all_theta_points": theta_points,
            "all_point_results": all_results,
            "program_manifest": program.manifest,
            "graph_audit": graph,
            "program_tensor_hashes": {
                "observations": _tensor_hash(program.observations),
                "guide_thetas": _tensor_hash(program.guide_thetas),
                "states": _tensor_hash(program.states),
                "ancestors": _tensor_hash(program.ancestors),
                "auxiliary_log_probabilities": _tensor_hash(
                    program.auxiliary_log_probabilities
                ),
                "transition_log_proposal_density": _tensor_hash(
                    program.transition_log_proposal_density
                ),
            },
            "device": {
                "logical_gpus": [item.name for item in logical],
                "memory_policy": memory,
                "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
                "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            },
            "gpu_allocator": {
                "current_bytes": int(allocator["current"]),
                "peak_bytes": int(allocator["peak"]),
            },
            "git": _git_payload(),
            "wall_time_seconds": time.monotonic() - started,
            "nonclaims": (
                "tuning viability is not a score identity or physical-likelihood claim",
                "guide branches are proposal design, not independent method replications",
                "no HMC, posterior, default, production, or superiority claim",
            ),
        }
    )
    (args.output_dir / "tuning.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "result.md").write_text(
        "# Austria Persistent Guide Tuning\n\n"
        f"Status: `{payload['status']}`\n\n"
        f"Selected domain half-width: `{payload['selected_domain_half_width']}`\n\n"
        f"Artifact ID: `{payload['tuning_artifact_id']}`\n",
        encoding="utf-8",
    )
    return 0 if primary_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

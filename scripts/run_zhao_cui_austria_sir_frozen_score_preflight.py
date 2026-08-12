#!/usr/bin/env python3
"""Run the reviewed FP64 GPU/XLA Austria frozen-score preflight."""

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
REVIEW = "docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-plan-review-result-2026-08-02.md"
SCHEMA = "bayesfilter.zhao_cui_austria_sir_frozen_score_preflight.v1"
MEMORY_LIMIT_MIB = 6144
SCORE_ATOL = 5e-6
SCORE_RTOL = 5e-6
FD_STEPS = (2.5e-4, 1.25e-4, 6.25e-5)
FORBIDDEN_GRAPH_OPS = ("PyFunc", "PyFuncStateless", "EagerPyFunc", "MapDefun")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        raw = value.numpy()
        return raw.item() if value.shape.rank == 0 else raw.tolist()
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if isinstance(value, Path):
        return str(value)
    return value


def _graph_audit(concrete: tf.types.experimental.ConcreteFunction) -> Mapping[str, object]:
    graph_def = concrete.graph.as_graph_def()
    operation_types = {
        node.op for node in graph_def.node
    } | {
        node.op
        for function in graph_def.library.function
        for node in function.node_def
    }
    forbidden = tuple(sorted(operation_types.intersection(FORBIDDEN_GRAPH_OPS)))
    while_ops = tuple(sorted(operation_types.intersection(("While", "StatelessWhile"))))
    return {
        "operation_type_count": len(operation_types),
        "while_operations": while_ops,
        "forbidden_operations": forbidden,
        "has_while": bool(while_ops),
        "has_host_callback": bool(forbidden),
    }


def _mixed_gate(actual: tf.Tensor, expected: tf.Tensor) -> tf.Tensor:
    tolerance = tf.cast(SCORE_ATOL, actual.dtype) + tf.cast(
        SCORE_RTOL, actual.dtype
    ) * tf.maximum(tf.abs(actual), tf.abs(expected))
    return tf.abs(actual - expected) <= tolerance


def _fd_ladder(program, theta: tf.Tensor) -> tf.Tensor:
    """Evaluate all FD coordinates with TensorFlow control flow, not Python loops."""

    steps = tf.constant(FD_STEPS, tf.float64)
    output = tf.TensorArray(
        tf.float64,
        size=tf.size(steps),
        clear_after_read=False,
        element_shape=tf.TensorShape([3]),
    )

    def step_body(
        step_index: tf.Tensor, rows: tf.TensorArray
    ) -> tuple[tf.Tensor, tf.TensorArray]:
        step = tf.gather(steps, step_index)
        coordinates = tf.TensorArray(
            tf.float64,
            size=3,
            clear_after_read=False,
            element_shape=tf.TensorShape([]),
        )

        def coordinate_body(
            parameter_index: tf.Tensor, values: tf.TensorArray
        ) -> tuple[tf.Tensor, tf.TensorArray]:
            direction = tf.one_hot(parameter_index, 3, dtype=tf.float64)
            plus = program.evaluate(theta + step * direction)["log_likelihood"]
            minus = program.evaluate(theta - step * direction)["log_likelihood"]
            return parameter_index + 1, values.write(
                parameter_index, (plus - minus) / (2.0 * step)
            )

        _, coordinates = tf.while_loop(
            lambda parameter_index, *_unused: parameter_index < 3,
            coordinate_body,
            (tf.zeros([], tf.int32), coordinates),
            maximum_iterations=3,
            parallel_iterations=1,
        )
        return step_index + 1, rows.write(step_index, coordinates.stack())

    _, output = tf.while_loop(
        lambda step_index, *_unused: step_index < tf.size(steps),
        step_body,
        (tf.zeros([], tf.int32), output),
        maximum_iterations=len(FD_STEPS),
        parallel_iterations=1,
    )
    return output.stack()


def _run_horizon(*, horizon: int, particles: int, seed: int) -> Mapping[str, object]:
    from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (
        make_bootstrap_mechanics_branch,
        prepare_austria_sir_source_order_program,
    )

    branch = make_bootstrap_mechanics_branch(
        particle_count=particles,
        horizon=horizon,
        proposal_seed=seed,
        dtype=tf.float64,
    )
    program = prepare_austria_sir_source_order_program(branch)
    theta = tf.constant([0.03, -0.02, 0.025], tf.float64)
    evaluator = program.compiled(jit_compile=True)
    concrete = evaluator.get_concrete_function()
    graph = _graph_audit(concrete)

    started = time.monotonic()
    result = evaluator(theta)
    compile_and_first_seconds = time.monotonic() - started
    started = time.monotonic()
    replay = evaluator(theta)
    replay_seconds = time.monotonic() - started
    value_replay_error = tf.abs(
        result["log_likelihood"] - replay["log_likelihood"]
    )
    score_replay_error = tf.reduce_max(tf.abs(result["score"] - replay["score"]))
    value_additivity_error = tf.abs(
        result["log_likelihood"] - tf.reduce_sum(result["log_increments"])
    )
    score_additivity_error = tf.reduce_max(
        tf.abs(result["score"] - tf.reduce_sum(result["increment_scores"], axis=0))
    )
    fd = _fd_ladder(program, theta) if horizon == 2 else None
    fd_gate = (
        tf.reduce_all(_mixed_gate(result["score"][tf.newaxis, :], fd))
        if fd is not None
        else tf.constant(True)
    )
    primary_pass = tf.reduce_all(
        tf.stack(
            (
                result["finite"],
                tf.equal(value_replay_error, 0.0),
                tf.equal(score_replay_error, 0.0),
                value_additivity_error <= tf.constant(5e-12, tf.float64),
                score_additivity_error <= tf.constant(5e-12, tf.float64),
                fd_gate,
                tf.constant(graph["has_while"]),
                tf.constant(not graph["has_host_callback"]),
            )
        )
    )
    return {
        "horizon": horizon,
        "particle_count": particles,
        "seed": seed,
        "dtype": "float64",
        "jit_compile": True,
        "artifact_role": "bootstrap_mechanics_not_proposal_quality_or_claim",
        "primary_pass": primary_pass,
        "value": result["log_likelihood"],
        "score": result["score"],
        "log_increments": result["log_increments"],
        "increment_scores": result["increment_scores"],
        "ess_by_time": result["ess_by_time"],
        "minimum_ess": result["minimum_ess"],
        "maximum_log_weight_spread": result["maximum_log_weight_spread"],
        "finite": result["finite"],
        "value_replay_error": value_replay_error,
        "score_replay_max_abs_error": score_replay_error,
        "value_additivity_error": value_additivity_error,
        "score_additivity_max_abs_error": score_additivity_error,
        "fd_steps": FD_STEPS if fd is not None else (),
        "fd_scores": fd if fd is not None else (),
        "fd_mixed_five_digit_pass": fd_gate,
        "graph_audit": graph,
        "compile_and_first_seconds": compile_and_first_seconds,
        "replay_seconds": replay_seconds,
        "branch": branch.manifest_payload(),
        "program": program.manifest_payload(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t2-particles", type=int, default=24)
    parser.add_argument("--t20-particles", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=30811)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("output-dir must be a fresh versioned directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started = time.monotonic()
    memory = configure_tensorflow_gpu_memory_limit(
        tf, memory_limit_mib=MEMORY_LIMIT_MIB, require_gpu=True
    )
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError("preflight requires exactly one logical GPU")

    with tf.device("/GPU:0"):
        t2 = _run_horizon(horizon=2, particles=args.t2_particles, seed=args.seed)
        t20 = _run_horizon(
            horizon=20,
            particles=args.t20_particles,
            seed=args.seed + 1,
        )
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    primary_pass = bool(t2["primary_pass"].numpy()) and bool(
        t20["primary_pass"].numpy()
    )
    payload = _json_ready(
        {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": (
                "PASS_FP64_GPU_XLA_FROZEN_SCORE_PREFLIGHT"
                if primary_pass
                else "BLOCK_FP64_GPU_XLA_FROZEN_SCORE_PREFLIGHT"
            ),
            "primary_pass": primary_pass,
            "plan": PLAN,
            "plan_sha256": _sha256(ROOT / PLAN),
            "review": REVIEW,
            "review_sha256": _sha256(ROOT / REVIEW),
            "command": " ".join(sys.argv),
            "environment": {
                "python": sys.version,
                "tensorflow": tf.__version__,
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "unset"
                ),
                "tf_force_gpu_allow_growth": os.environ.get(
                    "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
                ),
            },
            "device": {
                "physical_gpus": [
                    item.name for item in tf.config.list_physical_devices("GPU")
                ],
                "logical_gpus": [item.name for item in logical],
                "online_device": "/GPU:0",
                "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
                "memory_policy": memory,
                "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            },
            "gpu_allocator": {
                "current_bytes": int(allocator["current"]),
                "peak_bytes": int(allocator["peak"]),
            },
            "score_gate": {
                "formula": "abs(actual-reference) <= 5e-6 + 5e-6*max(abs(actual),abs(reference))",
                "score_atol": SCORE_ATOL,
                "score_rtol": SCORE_RTOL,
            },
            "t2": t2,
            "t20": t20,
            "git": _git_payload(),
            "wall_time_seconds": time.monotonic() - started,
            "nonclaims": (
                "bootstrap branches are mechanics baselines, not Zhao-Cui proposal claims",
                "no proposal-quality, T20 finite-score admission, physical-likelihood, HMC, posterior, default, or production claim",
            ),
        }
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "result.md").write_text(
        "# Austria Frozen Score FP64 GPU/XLA Preflight\n\n"
        f"Status: `{payload['status']}`\n\n"
        f"Primary pass: `{payload['primary_pass']}`\n\n"
        f"GPU allocator peak bytes: `{payload['gpu_allocator']['peak_bytes']}`\n\n"
        "This is a bootstrap mechanics preflight, not proposal-quality or T20 claim evidence.\n",
        encoding="utf-8",
    )
    return 0 if primary_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

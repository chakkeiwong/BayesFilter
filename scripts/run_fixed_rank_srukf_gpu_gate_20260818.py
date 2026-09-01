#!/usr/bin/env python3
"""Run the bounded fixed-rank SR-UKF GPU/XLA release gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from bayesfilter.nonlinear.rectangular_srukf_tf import (
    TFRectangularSRUKFDerivatives,
    TFRectangularSRUKFFixedBranch,
    TFRectangularSRUKFModel,
    tf_rectangular_srukf_value_and_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/plans/artifacts/direct-factor-srukf-fixed-rank-gpu-gate-20260818"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_tensor(value: tf.Tensor):
    materialized = value.numpy()
    return materialized.tolist() if value.shape.rank else materialized.item()


def _model() -> tuple[TFRectangularSRUKFModel, TFRectangularSRUKFDerivatives]:
    model = TFRectangularSRUKFModel(
        tf.constant([[0.0, 0.0]], tf.float64),
        tf.constant([[[0.0], [0.5]]], tf.float64),
        tf.constant([[[0.2]]], tf.float64),
        tf.constant([[[0.0], [0.0]]], tf.float64),
        lambda state, process: tf.stack(
            [state[..., 0] + process[..., 0], state[..., 1]], axis=-1
        ),
        lambda state: tf.stack([state[..., 0], state[..., 0]], axis=-1),
    )
    derivatives = TFRectangularSRUKFDerivatives(
        tf.zeros([1, 1, 2], tf.float64),
        tf.zeros([1, 1, 2, 1], tf.float64),
        tf.constant([[[[0.1]]]], tf.float64),
        tf.zeros([1, 1, 2, 1], tf.float64),
        lambda state, process: tf.broadcast_to(
            tf.eye(2, dtype=tf.float64), [1, tf.shape(state)[1], 2, 2]
        ),
        lambda state, process: tf.broadcast_to(
            tf.constant([[[1.0], [0.0]]], tf.float64),
            [1, tf.shape(state)[1], 2, 1],
        ),
        lambda state, process: tf.zeros(
            [1, 1, tf.shape(state)[1], 2], tf.float64
        ),
        lambda state: tf.broadcast_to(
            tf.constant([[[1.0, 0.0], [1.0, 0.0]]], tf.float64),
            [1, tf.shape(state)[1], 2, 2],
        ),
        lambda state: tf.zeros([1, 1, tf.shape(state)[1], 2], tf.float64),
    )
    return model, derivatives


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact root: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError("gate requires exactly one visible logical GPU")
    observations = tf.constant([[[0.1, 0.1], [0.2, 0.2]]], tf.float64)
    branch = TFRectangularSRUKFFixedBranch(2, (0, 1), 1, (0, 1), 1, (1, 0))
    with tf.device("/GPU:0"):
        gpu = tf_rectangular_srukf_value_and_score(
            observations, *_model(), branch, jit_compile=True
        )
    with tf.device("/CPU:0"):
        cpu = tf_rectangular_srukf_value_and_score(
            observations, *_model(), branch, jit_compile=False
        )
    value_gap = tf.reduce_max(tf.abs(gpu.log_likelihood - cpu.log_likelihood))
    score_gap = tf.reduce_max(tf.abs(gpu.score - cpu.score))
    passed = bool(
        tf.reduce_all(gpu.diagnostics["score_valid"])
        and value_gap <= tf.constant(1.0e-11, tf.float64)
        and score_gap <= tf.constant(1.0e-11, tf.float64)
    )
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "schema": "bayesfilter.fixed_rank_srukf_gpu_gate.v1",
        "status": "passed" if passed else "failed",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "physical_gpu_selection": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu": logical[0].name,
        "tensorflow": tf.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "memory_policy": memory_policy,
        "allocator_bytes": allocator,
        "git_head_before_commit": head,
        "branch_identity": branch.identity,
        "gpu_value": _json_tensor(gpu.log_likelihood),
        "gpu_score": _json_tensor(gpu.score),
        "cpu_reference_value": _json_tensor(cpu.log_likelihood),
        "cpu_reference_score": _json_tensor(cpu.score),
        "maximum_value_gap": float(value_gap),
        "maximum_score_gap": float(score_gap),
        "score_valid": _json_tensor(gpu.diagnostics["score_valid"]),
        "minimum_chart_pivot": _json_tensor(gpu.diagnostics["minimum_chart_pivot"]),
        "maximum_chart_residual": _json_tensor(gpu.diagnostics["maximum_chart_residual"]),
        "maximum_support_residual": _json_tensor(gpu.diagnostics["maximum_support_residual"]),
        "command": "TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=3 python scripts/run_fixed_rank_srukf_gpu_gate_20260818.py",
        "nonclaims": [
            "bounded structural singular fixture only",
            "no global score across rank, support, or chart changes",
            "no HMC convergence or exact nonlinear Bayesian inference claim",
        ],
    }
    result_path = OUTPUT / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_paths = [
        ROOT / "bayesfilter/linear/rectangular_factor_tf.py",
        ROOT / "bayesfilter/nonlinear/rectangular_srukf_tf.py",
        ROOT / "tests/test_rectangular_factor_tf.py",
        ROOT / "tests/test_rectangular_srukf_tf.py",
        result_path,
    ]
    checksums = {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths}
    (OUTPUT / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

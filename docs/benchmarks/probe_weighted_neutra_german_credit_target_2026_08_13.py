#!/usr/bin/env python3
"""Run a bounded German-credit target value/score and GPU/XLA canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "as_list"):
        return _ready(value.as_list())
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"output must be fresh: {args.output}")
    if int(args.batch_size) <= 1:
        raise ValueError("batch-size must exceed one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import tensorflow as tf

    from bayesfilter.inference.neutra_german_credit_target import (
        GermanCreditValueScoreAdapter,
        constrained_from_unconstrained,
        load_german_credit_target_spec,
    )
    from bayesfilter.inference.batched_value_score import evaluate_batch_native_value_score
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    started = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical_gpus}")
    spec = load_german_credit_target_spec(args.data, args.reference)
    adapter = GermanCreditValueScoreAdapter(spec)
    rows = tf.random.stateless_normal(
        (int(args.batch_size), spec.dimension), seed=(20260813, 42001), dtype=tf.float64
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_value_score(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return adapter.log_prob_and_grad(values)

    compile_started = time.perf_counter()
    value, score = compiled_value_score(rows)
    compile_elapsed = time.perf_counter() - compile_started
    value_score = evaluate_batch_native_value_score(adapter, rows)
    constrained = constrained_from_unconstrained(spec, rows)
    payload = {
        "schema": "bayesfilter.weighted_neutra_german_credit_target_probe.v1",
        "target": spec.manifest_payload(),
        "adapter_signature": adapter.adapter_signature(),
        "adapter_capability": adapter.value_score_capability().__dict__,
        "memory_policy": memory_policy,
        "logical_gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "float64",
        "jit_compile": True,
        "batch_size": int(args.batch_size),
        "value_shape": value.shape,
        "score_shape": score.shape,
        "value_score_metadata": value_score.metadata.__dict__,
        "value_score_diagnostics": value_score.diagnostics,
        "constrained_shape": constrained.shape,
        "all_finite": bool(
            tf.reduce_all(
                tf.math.is_finite(value)
                & tf.reduce_all(tf.math.is_finite(score), axis=1)
                & tf.reduce_all(tf.math.is_finite(constrained), axis=1)
            ).numpy()
        ),
        "compile_plus_run_seconds": compile_elapsed,
        "wall_seconds": time.perf_counter() - started,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "data_sha256": _sha256(args.data),
        "reference_sha256": _sha256(args.reference),
        "nonclaims": (
            "target canary only",
            "no training, HMC, posterior, or objective-ranking claim",
        ),
    }
    if not payload["all_finite"]:
        raise RuntimeError("German target canary produced nonfinite output")
    _write(args.output, payload)
    print(json.dumps({"output": args.output.as_posix(), "all_finite": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

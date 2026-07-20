#!/usr/bin/env python3
"""Bounded SSL-LSTM process-parallel value/score topology benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _select_gpu() -> str:
    if os.environ.get("BAYESFILTER_CPU_VALUE_SCORE_WORKER") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu-worker-hidden"
    probe = subprocess.run(
        ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    available = {
        int(line.strip())
        for line in probe.stdout.splitlines()
        if line.strip().isdigit()
    }
    selected = "1" if 1 in available else ("0" if 0 in available else "")
    if not selected:
        raise RuntimeError("neither physical GPU 1 nor GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_GPU = _select_gpu()
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.cpu_value_score_pool import (  # noqa: E402
    CPUValueScorePool,
    CPUValueScorePoolConfig,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
HOST_RAM_CAP_BYTES = 64 * 1024**3


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _plain(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _source_hashes() -> dict[str, str]:
    return {
        "plan": _sha(PLAN),
        "runner": _sha(SCRIPT),
        "pool": _sha(Path("bayesfilter/inference/cpu_value_score_pool.py")),
        "trainer": _sha(Path("bayesfilter/inference/neutra_training.py")),
        "target": _sha(
            Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py")
        ),
    }


def _trainer(q: int) -> tuple[Any, NeuTraReverseKLTrainer]:
    target = complexity_posterior_target(q, jit_compile=False)
    config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy()),
        target_parameter_names=target.parameter_names,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=4.0e-4,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260719, 12101 + q),
        jit_compile=True,
    )
    return target, NeuTraReverseKLTrainer(target, config)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_hashes = _source_hashes()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("topology benchmark requires the selected GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            if "cannot be modified after being initialized" not in str(exc):
                raise
    tf.config.experimental.enable_tensor_float_32_execution(True)
    target, trainer = _trainer(args.q)
    pool_config = CPUValueScorePoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_target_tf:"
            "complexity_target_worker_factory"
        ),
        worker_config={"q": int(args.q)},
        dimension=4,
        worker_count=int(args.workers),
        cores_per_worker=int(args.cores_per_worker),
        timeout_seconds=float(args.timeout_seconds),
    )
    calls = []
    started = time.perf_counter()
    with CPUValueScorePool(pool_config) as pool:
        for call in range(int(args.calls)):
            seed = tf.random.experimental.stateless_fold_in(
                tf.constant((20260719, 12200 + args.q), tf.int32), call
            )
            z = tf.random.stateless_normal(
                [int(args.batch_size), 4], seed=seed, dtype=tf.float64
            )
            call_started = time.perf_counter()
            theta_started = time.perf_counter()
            theta, _ = trainer.forward_and_logdet(z)
            theta_seconds = time.perf_counter() - theta_started
            worker_started = time.perf_counter()
            values, scores, worker_metadata = pool.evaluate(
                theta.numpy(), request_id=f"q{args.q}-w{args.workers}-call{call}"
            )
            worker_seconds = time.perf_counter() - worker_started
            update_started = time.perf_counter()
            result = trainer.train_step_with_external_value_score(z, values, scores)
            update_seconds = time.perf_counter() - update_started
            calls.append(
                {
                    "call": call,
                    "wall_seconds": time.perf_counter() - call_started,
                    "theta_seconds": theta_seconds,
                    "worker_wall_seconds": worker_seconds,
                    "gpu_update_seconds": update_seconds,
                    "loss": float(result.loss.numpy()),
                    "gradient_norm": float(result.gradient_norm.numpy()),
                    "worker": worker_metadata,
                }
            )
            checkpoint = Path(str(args.output) + ".partial")
            checkpoint_path = ROOT / checkpoint
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(
                _canonical(
                    {
                        "schema": (
                            "bayesfilter.ssl_lstm."
                            "process_parallel_topology.partial.v1"
                        ),
                        "q": int(args.q),
                        "workers": int(args.workers),
                        "completed_calls": calls,
                        "source_hashes": source_hashes,
                    }
                )
            )
    parent_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    worker_rss = max(
        int(row["worker"]["active_worker_ru_maxrss_sum_bytes"]) for row in calls
    )
    combined_rss = parent_rss + worker_rss
    warm = calls[1:]
    return {
        "schema": "bayesfilter.ssl_lstm.process_parallel_topology.v1",
        "q": int(args.q),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "cores_per_worker": int(args.cores_per_worker),
        "calls": calls,
        "warm_summary": {
            "call_count": len(warm),
            "wall_seconds_max": max(float(row["wall_seconds"]) for row in warm),
            "wall_seconds_mean": sum(float(row["wall_seconds"]) for row in warm)
            / len(warm),
            "worker_wall_seconds_max": max(
                float(row["worker_wall_seconds"]) for row in warm
            ),
            "gpu_update_seconds_max": max(
                float(row["gpu_update_seconds"]) for row in warm
            ),
        },
        "memory": {
            "parent_ru_maxrss_bytes": parent_rss,
            "active_worker_ru_maxrss_sum_bytes": worker_rss,
            "combined_conservative_bytes": combined_rss,
            "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        },
        "hard_vetoes": [
            "combined_host_ram_cap_exceeded"
        ]
        if combined_rss > HOST_RAM_CAP_BYTES
        else [],
        "run_manifest": {
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "selected_physical_gpu": SELECTED_GPU,
            "logical_gpus": [gpu.name for gpu in tf.config.list_logical_devices("GPU")],
            "jit_compile_parent_update": True,
            "jit_compile_worker_target": False,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "wall_seconds": time.perf_counter() - started,
            "plan": PLAN.as_posix(),
            "runner": SCRIPT.as_posix(),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "source_hashes": source_hashes,
        },
        "target_signature": target.target_signature(),
        "nonclaims": [
            "execution-topology timing only",
            "no NeuTra quality claim",
            "no HMC convergence or posterior claim",
            "worker-count timing differences are descriptive only",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(1, 2, 5, 10, 20), default=1)
    parser.add_argument("--workers", type=int, choices=(16, 32, 64, 96), required=True)
    parser.add_argument("--cores-per-worker", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calls < 2 or args.batch_size <= 0 or args.cores_per_worker <= 0:
        parser.error("calls must be >=2 and batch/cores must be positive")
    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite topology receipt: {args.output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run(args)
    output.write_bytes(_canonical(_plain(payload)))
    partial = Path(str(output) + ".partial")
    if partial.exists():
        partial.unlink()
    print(
        json.dumps(
            {
                "q": payload["q"],
                "workers": payload["workers"],
                "warm_wall_max": payload["warm_summary"]["wall_seconds_max"],
                "combined_rss": payload["memory"]["combined_conservative_bytes"],
                "hard_vetoes": payload["hard_vetoes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

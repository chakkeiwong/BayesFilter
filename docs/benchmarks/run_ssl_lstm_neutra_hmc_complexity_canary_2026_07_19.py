#!/usr/bin/env python3
"""Single-rung trusted GPU/XLA canary for the SSL-LSTM complexity ladder."""

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


def _select_preferred_gpu() -> str:
    """Select physical GPU 1 when present, otherwise physical GPU 0."""

    if os.environ.get("BAYESFILTER_CPU_VALUE_SCORE_WORKER") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu-worker-hidden"
    try:
        probe = subprocess.run(
            ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        available = {int(line.strip()) for line in probe.stdout.splitlines() if line.strip().isdigit()}
    except (OSError, subprocess.SubprocessError, ValueError):
        available = set()
    selected = "1" if 1 in available else ("0" if 0 in available else "")
    if not selected:
        raise RuntimeError("no physical GPU 1 or GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_GPU = _select_preferred_gpu()
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter  # noqa: E402
from bayesfilter.inference.hmc import (  # noqa: E402
    FullChainHMCConfig,
    build_reusable_full_chain_tfp_hmc_runner,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.inference.cpu_value_score_pool import (  # noqa: E402
    CPUValueScorePool,
    CPUValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


PLAN = Path("docs/plans/bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md")
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
HOST_RAM_CAP_BYTES = 64 * 1024**3


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


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


class HMCReceiptBridge:
    """Grant canary-only full-chain authority to the Phase-1-passed target."""

    def __init__(self, target: Any) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:phase2_canary_only"

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            (self.target.adapter_signature() + ":phase2-canary").encode("ascii")
        ).hexdigest()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_complexity_phase2_canary_bridge",
            evidence_path=PLAN.as_posix(),
            target_scope=self.target_scope,
            nonclaims=(
                "Phase 2 timing/mechanics canary only",
                "no HMC tuning or convergence claim",
                "authority is scoped to the passing Phase 1 target preflight",
            ),
        )

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("canary target requires rank-one or rank-two positions")


def configure_gpu() -> list[Any]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("trusted complexity canary requires a visible GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            # Importing the target may initialize the managed device first. In
            # that case TensorFlow's default allocator is already authoritative;
            # retain the run and record the allocator telemetry instead of
            # misclassifying setup order as a model failure.
            if "cannot be modified after being initialized" not in str(exc):
                raise
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    return gpus


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_hashes = {
        "plan": _sha(PLAN),
        "runner": _sha(SCRIPT),
        "target": _sha(Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py")),
        "pool": _sha(Path("bayesfilter/inference/cpu_value_score_pool.py")),
        "trainer": _sha(Path("bayesfilter/inference/neutra_training.py")),
        "hmc": _sha(Path("bayesfilter/inference/hmc.py")),
    }
    gpus = configure_gpu()
    started = time.perf_counter()
    target = complexity_posterior_target(args.q, jit_compile=True)
    streams = []
    frozen = None
    pool_config = CPUValueScorePoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_target_tf:"
            "complexity_target_worker_factory"
        ),
        worker_config={"q": int(args.q)},
        dimension=4,
        worker_count=int(args.worker_count),
    )
    with CPUValueScorePool(pool_config) as value_score_pool:
        for stream_index, seed_tail in enumerate((3101, 3201)):
            config = ssl_lstm_tuned_capacity_neutra_config(
                dimension=4,
                fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy()),
                target_parameter_names=target.parameter_names,
                target_signature=target.target_signature(),
                target_adapter_signature=target.adapter_signature(),
                learning_rate=4.0e-4,
                initialization_scale=0.01,
                gradient_clip_norm=10.0,
                initialization_seed=(20260719, seed_tail),
                jit_compile=True,
            )
            trainer = NeuTraReverseKLTrainer(target, config)
            step_times = []
            rows = []
            for step in range(1, args.steps + 1):
                seed = tf.random.experimental.stateless_fold_in(
                    tf.constant((20260719, seed_tail + 100), tf.int32), step
                )
                z = tf.random.stateless_normal(
                    (args.batch_size, 4), seed, dtype=tf.float64
                )
                step_started = time.perf_counter()
                theta, _logdet = trainer.forward_and_logdet(z)
                values_np, scores_np, pool_metadata = value_score_pool.evaluate(
                    theta.numpy(),
                    request_id=f"q{args.q}-stream{stream_index}-step{step}",
                )
                result = trainer.train_step_with_external_value_score(
                    z, values_np, scores_np
                )
                step_times.append(time.perf_counter() - step_started)
                rows.append(
                    {
                        "step": step,
                        "loss": float(result.loss.numpy()),
                        "gradient_norm": float(result.gradient_norm.numpy()),
                        "clipped_gradient_norm": float(
                            result.clipped_gradient_norm.numpy()
                        ),
                    }
                )
            validation_z = tf.random.stateless_normal(
                (64, 4),
                tf.constant((20260719, seed_tail + 200), tf.int32),
                dtype=tf.float64,
            )
            validation_theta, _ = trainer.forward_and_logdet(validation_z)
            validation_values, validation_pool_metadata = (
                value_score_pool.evaluate_values(
                    validation_theta.numpy(),
                    request_id=f"q{args.q}-stream{stream_index}-validation",
                )
            )
            validation = trainer.validation_batch_with_external_value(
                validation_z, validation_values
            )
            payload = trainer.frozen_transport_payload(
                transport_id=(
                    f"ssl-lstm-complexity-q{args.q}-canary-seed{stream_index}"
                ),
                target_signature=target.target_signature(),
            )
            loaded = load_frozen_neutra_artifact(
                payload, expected_target_signature=target.target_signature()
            )
            frozen = loaded.transport
            streams.append(
                {
                    "stream": stream_index,
                    "initialization_seed": [20260719, seed_tail],
                    "training_seed_root": [20260719, seed_tail + 100],
                    "step_times_seconds": step_times,
                    "first_step_seconds": step_times[0],
                    "warm_step_max_seconds": (
                        max(step_times[1:]) if len(step_times) > 1 else None
                    ),
                    "warm_step_mean_seconds": (
                        sum(step_times[1:]) / len(step_times[1:])
                        if len(step_times) > 1
                        else None
                    ),
                    "rows": rows,
                    "worker_backend": pool_metadata,
                    "validation_worker_backend": validation_pool_metadata,
                    "validation_loss_mean": float(
                        tf.reduce_mean(validation.per_sample_loss).numpy()
                    ),
                    "validation_finite": bool(
                        tf.reduce_all(
                            tf.math.is_finite(validation.per_sample_loss)
                        ).numpy()
                    ),
                    "transport_hash": loaded.manifest.transport_hash,
                }
            )
        assert frozen is not None
    bridge = HMCReceiptBridge(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=frozen,
        target_scope=f"{bridge.target_scope}:fixed_transport",
        runtime_backend="ssl_lstm_complexity_phase2_fixed_transport_canary",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=("Phase 2 minimal transformed-HMC mechanics only", "no tuning or convergence claim"),
    )
    hmc_config = FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.01,
        num_leapfrog_steps=1,
        seed=(20260719, 4000 + args.q),
        use_xla=True,
        trace_policy="standard",
        target_scope=adapter.target_scope,
    )
    initial = tf.zeros((4, 4), tf.float64)
    runner = build_reusable_full_chain_tfp_hmc_runner(adapter, initial, hmc_config)
    first_hmc = runner.run()
    warm_hmc = runner.run(seed=(20260719, 5000 + args.q))
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    worker_rss = max(
        int(record["worker_backend"]["active_worker_ru_maxrss_sum_bytes"])
        for record in streams
    )
    combined_rss = rss + worker_rss
    gpu_memory = {key: int(value) for key, value in tf.config.experimental.get_memory_info("GPU:0").items()}
    warm_training_max = max(float(row["warm_step_max_seconds"]) for row in streams)
    projected_training = 2.0 * (float(streams[0]["first_step_seconds"]) + 4999.0 * warm_training_max)
    payload = {
        "schema": "bayesfilter.ssl_lstm.neutra_hmc.complexity_canary.v1",
        "q": args.q,
        "state_dim": 3 * args.q,
        "parameter_chart_dim": target.config.static_config.parameter_dim,
        "free_dim": 4,
        "steps_per_stream": args.steps,
        "batch_size": args.batch_size,
        "streams": streams,
        "hmc": {
            "first_call_seconds": first_hmc.metadata["first_call_s"],
            "warm_call_seconds": warm_hmc.metadata["warm_call_s"],
            "samples_shape": list(warm_hmc.samples.shape),
            "finite": bool(tf.reduce_all(tf.math.is_finite(warm_hmc.samples)).numpy()),
            "diagnostics": _plain(warm_hmc.diagnostics),
        },
        "projection": {
            "two_seed_5000_step_training_seconds_before_margin": projected_training,
            "two_seed_5000_step_training_seconds_with_50pct_margin": 1.5 * projected_training,
            "includes_optuna_hmc_retained_predictive": False,
        },
        "run_manifest": {
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "physical_gpus": [gpu.name for gpu in gpus],
            "selected_physical_gpu": SELECTED_GPU,
            "value_score_worker_count": int(args.worker_count),
            "value_score_worker_execution": "persistent_spawn_cpu_eager",
            "logical_gpus": [gpu.name for gpu in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "wall_seconds": time.perf_counter() - started,
            "ru_maxrss_bytes": rss,
            "active_worker_ru_maxrss_sum_bytes": worker_rss,
            "combined_conservative_host_bytes": combined_rss,
            "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
            "gpu_memory_bytes": gpu_memory,
            "plan": PLAN.as_posix(),
            "runner": SCRIPT.as_posix(),
            "source_hashes": source_hashes,
        },
        "hard_vetoes": (
            ["combined_host_ram_cap_exceeded"]
            if combined_rss > HOST_RAM_CAP_BYTES
            else []
        ),
        "nonclaims": [
            "timing and mechanics canary only",
            "no hyperparameter nomination",
            "no NeuTra quality or HMC convergence claim",
            "no posterior correctness or scientific claim",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(1, 2, 5, 10, 20), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--worker-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size <= 0 or args.worker_count <= 0:
        parser.error("--steps must be >=2, --batch-size and --worker-count must be positive")
    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite canary receipt: {args.output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run(args)
    output.write_bytes(_canonical(_plain(payload)))
    print(json.dumps({"q": args.q, "wall_seconds": payload["run_manifest"]["wall_seconds"],
                      "rss": payload["run_manifest"]["ru_maxrss_bytes"],
                      "projected_5000_with_margin": payload["projection"]["two_seed_5000_step_training_seconds_with_50pct_margin"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

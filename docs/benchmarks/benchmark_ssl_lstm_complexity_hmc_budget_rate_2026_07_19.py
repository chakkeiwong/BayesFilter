#!/usr/bin/env python3
"""Refresh batched-chain HMC cost rates under the current execution source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _select_gpu() -> str:
    mode = None
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if index + 1 < len(sys.argv):
            mode = sys.argv[index + 1]
    if mode != "timing-canary":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu-hidden-contract-smoke"
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
        raise RuntimeError("no physical GPU 1 or GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_GPU = _select_gpu()

import numpy as np
import tensorflow as tf


def _enable_memory_growth_before_project_imports() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError(
                "GPU memory growth must be established before project imports"
            ) from exc
        if tf.config.experimental.get_memory_growth(gpu) is not True:
            raise RuntimeError("GPU memory growth verification failed")


_enable_memory_growth_before_project_imports()


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import (  # noqa: E402
    FixedTransportValueScoreAdapter,
)
from bayesfilter.inference.hmc import (  # noqa: E402
    FullChainHMCConfig,
    build_reusable_full_chain_tfp_hmc_runner,
)
from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.complexity_hmc_budget_rate.v1"
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
Q_VALUES = (1, 2, 5, 10, 20)
NUM_RESULTS = 2
NUM_BURNIN = 1
NUM_LEAPFROG = 1
WARM_REPEATS = 2
HMC_TRANSITION_LEAPFROGS_PER_RUNG = 408800
HMC_COLD_RESERVE_SECONDS_PER_RUNG = 9000.0
HMC_MARGIN = 1.50
HOST_RAM_CAP_BYTES = 64 * 1024**3
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)


class HMCBudgetRateError(RuntimeError):
    pass


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):
        return json_safe(value.numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise HMCBudgetRateError(f"{label} must remain inside the repository")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HMCBudgetRateError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def execution_source_signature() -> str:
    paths = (
        SCRIPT,
        Path("bayesfilter/inference/hmc.py"),
        Path("bayesfilter/inference/batched_value_score.py"),
        Path("bayesfilter/inference/neutra_training.py"),
        Path("bayesfilter/inference/neutra_artifacts.py"),
        Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
    )
    return payload_sha256({path.as_posix(): sha256(ROOT / path) for path in paths})


class TargetBridge:
    def __init__(self, target: Any) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:hmc_budget_rate_only"

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            (self.target.adapter_signature() + ":hmc-budget-rate").encode("ascii")
        ).hexdigest()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_complexity_hmc_budget_rate_bridge",
            evidence_path=PLAN.as_posix(),
            target_scope=self.target_scope,
            nonclaims=(
                "batched-chain timing mechanics only",
                "no HMC tuning or convergence claim",
                "no posterior correctness claim",
            ),
        )

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("HMC timing target requires rank-one or rank-two positions")


def hmc_budget_seconds(seconds_per_transition_leapfrog: float) -> float:
    return (
        HMC_MARGIN
        * float(seconds_per_transition_leapfrog)
        * HMC_TRANSITION_LEAPFROGS_PER_RUNG
        + HMC_COLD_RESERVE_SECONDS_PER_RUNG
    )


def contract_payload(q: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": q,
        "num_results": NUM_RESULTS,
        "num_burnin_steps": NUM_BURNIN,
        "num_leapfrog_steps": NUM_LEAPFROG,
        "warm_repeats": WARM_REPEATS,
        "selected_hmc_topology": "single_tfp_sample_chain_batched_four_chain_xla",
        "hmc_transition_leapfrogs_per_rung": HMC_TRANSITION_LEAPFROGS_PER_RUNG,
        "hmc_cold_reserve_seconds_per_rung": HMC_COLD_RESERVE_SECONDS_PER_RUNG,
        "hmc_margin": HMC_MARGIN,
        "material_execution_authorized": False,
        "nonclaims": [
            "contract/import smoke only",
            "no HMC execution, tuning, retention, or convergence claim",
        ],
    }


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise HMCBudgetRateError("HMC budget-rate canary requires a visible GPU")
    for gpu in gpus:
        if tf.config.experimental.get_memory_growth(gpu) is not True:
            raise HMCBudgetRateError("GPU memory growth verification failed")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (ValueError, RuntimeError):
        pass


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    output = repo_path(args.output, label="output")
    configure_gpu()
    source_signature = execution_source_signature()
    target = complexity_posterior_target(args.q, jit_compile=True)
    trainer_config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy()),
        target_parameter_names=target.parameter_names,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=4.0e-4,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260719, 8300 + args.q),
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, trainer_config)
    frozen = trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-q{args.q}-hmc-budget-rate-initialization",
        target_signature=target.target_signature(),
    )
    artifact = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    bridge = TargetBridge(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=artifact.transport,
        target_scope=f"{bridge.target_scope}:fixed_transport",
        runtime_backend="ssl_lstm_complexity_hmc_budget_rate",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "deterministic untrained 32x32 three-stage dense-IAF",
            "batched-chain timing mechanics only",
            "no transport-quality or HMC convergence claim",
        ),
    )
    config = FullChainHMCConfig(
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        step_size=0.01,
        num_leapfrog_steps=NUM_LEAPFROG,
        seed=(20260719, 8400 + args.q),
        use_xla=True,
        trace_policy="standard",
        target_scope=adapter.target_scope,
    )
    initial = tf.constant(INITIAL_Z, tf.float64)
    runner = build_reusable_full_chain_tfp_hmc_runner(adapter, initial, config)
    started = time.perf_counter()
    first = runner.run(seed=(20260719, 8500 + args.q))
    warm_rows = []
    for repeat in range(WARM_REPEATS):
        if execution_source_signature() != source_signature:
            raise HMCBudgetRateError("execution source drift during HMC rate canary")
        result = runner.run(seed=(20260719, 8600 + 100 * args.q + repeat))
        diagnostics = json_safe(result.diagnostics)
        movement = diagnostics.get("tuning_telemetry", {}).get(
            "movement_rate_by_chain", []
        )
        output_devices = sorted({str(result.samples.device)})
        hard_vetoes = []
        if tuple(result.samples.shape) != (NUM_RESULTS, 4, 4):
            hard_vetoes.append("sample_shape_mismatch")
        if not bool(tf.reduce_all(tf.math.is_finite(result.samples)).numpy()):
            hard_vetoes.append("nonfinite_samples")
        if not movement or not all(float(value) > 0.0 for value in movement):
            hard_vetoes.append("unmoved_chain")
        if diagnostics.get("divergence_count") not in (None, 0):
            hard_vetoes.append("positive_native_divergence")
        if not output_devices or not all("GPU:" in value for value in output_devices):
            hard_vetoes.append("gpu_output_placement_missing")
        if hard_vetoes:
            raise HMCBudgetRateError(",".join(hard_vetoes))
        wall = float(result.metadata["sample_chain_call_s"])
        warm_rows.append(
            {
                "repeat": repeat,
                "seed": [20260719, 8600 + 100 * args.q + repeat],
                "sample_chain_seconds": wall,
                "seconds_per_transition_leapfrog": wall
                / ((NUM_RESULTS + NUM_BURNIN) * NUM_LEAPFROG),
                "diagnostics": diagnostics,
                "output_devices": output_devices,
            }
        )
    warm_rate = max(row["seconds_per_transition_leapfrog"] for row in warm_rows)
    budget_seconds = hmc_budget_seconds(warm_rate)
    host_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if host_rss > HOST_RAM_CAP_BYTES:
        raise HMCBudgetRateError("HMC timing parent RSS exceeded 64 GiB")
    try:
        gpu_memory = json_safe(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        gpu_memory = {"status": "unavailable"}
    payload = {
        "schema": SCHEMA,
        "mode": "timing-canary",
        "status": "PASSED",
        "q": args.q,
        "target_signature": target.target_signature(),
        "transport_hash": artifact.manifest.transport_hash,
        "execution_source_signature": source_signature,
        "selected_hmc_topology": "single_tfp_sample_chain_batched_four_chain_xla",
        "config": {
            "num_results": NUM_RESULTS,
            "num_burnin_steps": NUM_BURNIN,
            "num_leapfrog_steps": NUM_LEAPFROG,
            "step_size": 0.01,
            "warm_repeats": WARM_REPEATS,
        },
        "first_call_seconds": float(first.metadata["sample_chain_call_s"]),
        "warm_rows": warm_rows,
        "warm_seconds_per_transition_leapfrog_max": warm_rate,
        "hmc_transition_leapfrogs_per_rung": HMC_TRANSITION_LEAPFROGS_PER_RUNG,
        "hmc_cold_reserve_seconds_per_rung": HMC_COLD_RESERVE_SECONDS_PER_RUNG,
        "hmc_margin": HMC_MARGIN,
        "hmc_budget_seconds": budget_seconds,
        "hmc_budget_hours": budget_seconds / 3600.0,
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "selected_physical_gpu": SELECTED_GPU,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpus": [
                device.name for device in tf.config.list_logical_devices("GPU")
            ],
            "tf_force_gpu_allow_growth": os.environ.get(
                "TF_FORCE_GPU_ALLOW_GROWTH"
            ),
            "gpu_memory_growth_verified": all(
                tf.config.experimental.get_memory_growth(gpu) is True
                for gpu in tf.config.list_physical_devices("GPU")
            ),
            "jit_compile": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "wall_seconds": time.perf_counter() - started,
            "host_ru_maxrss_bytes": host_rss,
            "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
            "gpu_allocator_memory": gpu_memory,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "plan": PLAN.as_posix(),
            "output": args.output.as_posix(),
        },
        "inference_status": {
            "hard_veto_screen": "passed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": ["first/warm HMC wall and acceptance"],
            "default_readiness": "not_assessed",
            "next_evidence_needed": "complete Phase 3--6 numerical budget",
        },
        "nonclaims": [
            "timing and HMC mechanics canary only",
            "no kernel tuning, retention, convergence, or posterior claim",
            "no transport-quality or sampler-superiority claim",
        ],
    }
    write_json(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract-smoke", "timing-canary"), required=True)
    parser.add_argument("--q", type=int, choices=Q_VALUES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorize-timing-canary", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "timing-canary":
        if not args.authorize_timing_canary:
            parser.error("timing-canary requires --authorize-timing-canary")
        if args.output is None:
            parser.error("timing-canary requires --output")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = contract_payload(args.q) if args.mode == "contract-smoke" else run_canary(args)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {"mode": payload["mode"], "status": payload["status"], "q": payload["q"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

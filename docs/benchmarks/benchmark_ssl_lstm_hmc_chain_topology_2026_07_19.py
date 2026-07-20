#!/usr/bin/env python3
"""Compare batched-chain and independent scalar-chain XLA HMC topology."""

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
from typing import Any, Mapping


def _select_preferred_gpu() -> str:
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


SELECTED_GPU = _select_preferred_gpu()
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter  # noqa: E402
from bayesfilter.inference.hmc import (  # noqa: E402
    FullChainHMCConfig,
    build_independent_chain_tfp_hmc_runner,
    build_reusable_full_chain_tfp_hmc_runner,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-neutra-hmc-state-complexity-2026-07-19/hmc-topology"
)
HOST_RAM_CAP_BYTES = 64 * 1024**3


def _sha(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _plain(value.numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _tensor_hash(value: Any) -> str:
    tensor = tf.convert_to_tensor(value)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_tree_equal(left[key], right[key]) for key in left)
    return np.array_equal(
        tf.convert_to_tensor(left).numpy(),
        tf.convert_to_tensor(right).numpy(),
        equal_nan=True,
    )


def _trace_hash(trace: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], path + (str(key),))
            return
        digest.update(".".join(path).encode("utf-8"))
        digest.update(bytes(tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()))

    visit(trace, tuple())
    return digest.hexdigest()


class HMCReceiptBridge:
    """Scope full-chain HMC authority to this topology mechanics preflight."""

    def __init__(self, target: Any) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:phase2c_hmc_topology_only"

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            (self.target.adapter_signature() + ":phase2c-hmc-topology").encode("ascii")
        ).hexdigest()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_complexity_phase2c_hmc_topology_bridge",
            evidence_path=PLAN.as_posix(),
            target_scope=self.target_scope,
            nonclaims=(
                "Phase 2C execution-topology mechanics only",
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
        raise ValueError("topology target requires rank-one or rank-two positions")


def _configure_gpu() -> list[Any]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("trusted HMC topology preflight requires a visible GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            if "cannot be modified after being initialized" not in str(exc):
                raise
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    return gpus


def _summarize_result(result: Any) -> dict[str, Any]:
    diagnostics = _plain(result.diagnostics)
    tuning = diagnostics.get("tuning_telemetry", {})
    movement = tuning.get("movement_rate_by_chain", [])
    return {
        "samples_shape": list(result.samples.shape),
        "samples_hash": _tensor_hash(result.samples),
        "trace_hash": _trace_hash(result.trace),
        "trace_keys": sorted(result.trace),
        "finite": bool(np.all(np.isfinite(result.samples.numpy()))),
        "nonfinite_sample_count": int(diagnostics["nonfinite_sample_count"]),
        "acceptance_rate": diagnostics.get("acceptance_rate"),
        "movement_rate_by_chain": movement,
        "all_chains_moved": bool(movement) and all(float(value) > 0.0 for value in movement),
        "native_divergence_status": diagnostics.get("native_divergence_status"),
        "divergence_count": diagnostics.get("divergence_count"),
        "hmc_health_diagnostics": diagnostics.get("hmc_health_diagnostics"),
    }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_hashes = {
        "plan": _sha(PLAN),
        "runner": _sha(SCRIPT),
        "hmc": _sha(Path("bayesfilter/inference/hmc.py")),
        "target": _sha(Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py")),
        "trainer": _sha(Path("bayesfilter/inference/neutra_training.py")),
        "transport_adapter": _sha(
            Path("bayesfilter/inference/batched_value_score.py")
        ),
    }
    gpus = _configure_gpu()
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
        initialization_seed=(20260719, 8000 + args.q),
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, trainer_config)
    frozen_payload = trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-q{args.q}-hmc-topology-deterministic-initialization",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen_payload,
        expected_target_signature=target.target_signature(),
    )
    bridge = HMCReceiptBridge(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=loaded.transport,
        target_scope=f"{bridge.target_scope}:fixed_transport",
        runtime_backend="ssl_lstm_complexity_phase2c_fixed_transport",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "Phase 2C HMC execution-topology mechanics only",
            "deterministic untrained transport initialization",
            "no transport-quality or HMC convergence claim",
        ),
    )
    config = FullChainHMCConfig(
        num_results=args.num_results,
        num_burnin_steps=args.num_burnin_steps,
        step_size=args.step_size,
        num_leapfrog_steps=args.num_leapfrog_steps,
        seed=(20260719, 9000 + args.q),
        use_xla=True,
        trace_policy="standard",
        target_scope=adapter.target_scope,
    )
    initial = tf.constant(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.10, -0.05, 0.05, 0.02],
            [-0.08, 0.07, -0.03, 0.04],
            [0.04, 0.08, -0.06, -0.05],
        ],
        tf.float64,
    )
    batched = build_reusable_full_chain_tfp_hmc_runner(adapter, initial, config)
    independent = build_independent_chain_tfp_hmc_runner(adapter, initial, config)

    started = time.perf_counter()
    batched_first = batched.run(seed=(20260719, 9100 + args.q))
    independent_first = independent.run(
        root_seed=(20260719, 9200 + args.q), mode="serial"
    )
    compile_and_first_wall_s = time.perf_counter() - started

    warm_rows = []
    parity_rows = []
    batched_walls = []
    serial_walls = []
    threaded_walls = []
    final_results: dict[str, Any] = {
        "batched": batched_first,
        "serial": independent_first,
        "threaded": independent_first,
    }
    for repeat in range(args.repeats):
        root_seed = (20260719, 9300 + 100 * args.q + repeat)
        batched_result = batched.run(seed=root_seed)
        batched_wall = float(batched_result.metadata["sample_chain_call_s"])
        if repeat % 2 == 0:
            serial_result = independent.run(root_seed=root_seed, mode="serial")
            threaded_result = independent.run(root_seed=root_seed, mode="threaded")
        else:
            threaded_result = independent.run(root_seed=root_seed, mode="threaded")
            serial_result = independent.run(root_seed=root_seed, mode="serial")
        serial_wall = float(serial_result.metadata["ensemble_call_s"])
        threaded_wall = float(threaded_result.metadata["ensemble_call_s"])
        samples_equal = np.array_equal(
            serial_result.samples.numpy(), threaded_result.samples.numpy()
        )
        traces_equal = _tree_equal(serial_result.trace, threaded_result.trace)
        parity_rows.append(
            {
                "repeat": repeat,
                "root_seed": list(root_seed),
                "samples_equal": samples_equal,
                "traces_equal": traces_equal,
                "serial_samples_hash": _tensor_hash(serial_result.samples),
                "threaded_samples_hash": _tensor_hash(threaded_result.samples),
                "serial_trace_hash": _trace_hash(serial_result.trace),
                "threaded_trace_hash": _trace_hash(threaded_result.trace),
            }
        )
        warm_rows.append(
            {
                "repeat": repeat,
                "root_seed": list(root_seed),
                "batched_chain_seconds": batched_wall,
                "scalar_chain_serial_seconds": serial_wall,
                "scalar_chain_threaded_seconds": threaded_wall,
                "scalar_chain_serial_per_chain_seconds": list(
                    serial_result.metadata["per_chain_call_s"]
                ),
                "scalar_chain_threaded_per_chain_seconds": list(
                    threaded_result.metadata["per_chain_call_s"]
                ),
            }
        )
        batched_walls.append(batched_wall)
        serial_walls.append(serial_wall)
        threaded_walls.append(threaded_wall)
        final_results = {
            "batched": batched_result,
            "serial": serial_result,
            "threaded": threaded_result,
        }

    summaries = {
        label: _summarize_result(result) for label, result in final_results.items()
    }
    all_mechanics_pass = all(
        summary["finite"]
        and summary["nonfinite_sample_count"] == 0
        and summary["all_chains_moved"]
        and summary["samples_shape"] == [args.num_results, 4, 4]
        for summary in summaries.values()
    )
    parity_pass = all(
        row["samples_equal"] and row["traces_equal"] for row in parity_rows
    )
    timing = {
        "batched_chain_median_seconds": _median(batched_walls),
        "scalar_chain_serial_median_seconds": _median(serial_walls),
        "scalar_chain_threaded_median_seconds": _median(threaded_walls),
    }
    timing["threaded_speedup_over_batched"] = (
        timing["batched_chain_median_seconds"]
        / timing["scalar_chain_threaded_median_seconds"]
    )
    timing["threaded_speedup_over_scalar_serial"] = (
        timing["scalar_chain_serial_median_seconds"]
        / timing["scalar_chain_threaded_median_seconds"]
    )
    threaded_meets_10pct = (
        timing["scalar_chain_threaded_median_seconds"]
        <= 0.9 * timing["batched_chain_median_seconds"]
        and timing["scalar_chain_threaded_median_seconds"]
        <= 0.9 * timing["scalar_chain_serial_median_seconds"]
    )
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    gpu_memory = {
        key: int(value)
        for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
    }
    hard_vetoes = []
    if not all_mechanics_pass:
        hard_vetoes.append("mechanics_or_movement_failure")
    if not parity_pass:
        hard_vetoes.append("scalar_serial_threaded_replay_mismatch")
    if rss > HOST_RAM_CAP_BYTES:
        hard_vetoes.append("host_rss_above_64_gib")
    payload = {
        "schema": "bayesfilter.ssl_lstm.hmc_chain_topology.v1",
        "status": "passed_topology_preflight" if not hard_vetoes else "failed_topology_preflight",
        "q": args.q,
        "state_dim": 3 * args.q,
        "parameter_dim": 4,
        "research_question": (
            "whether independent scalar-chain threaded XLA reduces four-chain HMC "
            "wall time relative to current batched-chain XLA"
        ),
        "baseline": "single_tfp_sample_chain_batched_four_chain_xla",
        "candidate": "four_independent_scalar_chain_xla_thread_pool",
        "source_transfer": {
            "repository": "/home/ubuntu/python/dsge_hmc",
            "implementation": "src/dsge_hmc/_hmc_orchestrator.py",
            "production_lines": "1844-1933",
            "boundary": "per_chain_tf_function_then_ThreadPoolExecutor",
            "not_used": "os_process_pool_for_hmc_target_evaluations",
        },
        "config": {
            "num_results": args.num_results,
            "num_burnin_steps": args.num_burnin_steps,
            "step_size": args.step_size,
            "num_leapfrog_steps": args.num_leapfrog_steps,
            "repeats": args.repeats,
            "initial_state": initial.numpy().tolist(),
            "transport": "deterministic_untrained_32x32_three_stage_dense_iaf",
            "transport_hash": loaded.manifest.transport_hash,
        },
        "compile_and_first_wall_seconds": compile_and_first_wall_s,
        "first_calls": {
            "batched_seconds": float(batched_first.metadata["sample_chain_call_s"]),
            "independent_scalar_serial_seconds": float(
                independent_first.metadata["ensemble_call_s"]
            ),
        },
        "warm_rows": warm_rows,
        "timing": timing,
        "scalar_serial_threaded_parity": parity_rows,
        "mechanics": summaries,
        "decision": {
            "threaded_meets_q20_10pct_topology_criterion": threaded_meets_10pct,
            "criterion_deciding_only_at_q20": True,
            "hard_vetoes": hard_vetoes,
            "selected_if_q20": (
                "independent_scalar_chain_threaded_xla"
                if args.q == 20 and threaded_meets_10pct and not hard_vetoes
                else "retain_batched_chain_xla_pending_or_failed_q20_criterion"
            ),
            "ranking_status": "descriptive_execution_topology_only",
        },
        "run_manifest": {
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "command": " ".join(sys.argv),
            "environment": sys.executable,
            "tensorflow_version": tf.__version__,
            "physical_gpu_selected": SELECTED_GPU,
            "logical_gpu_devices": [device.name for device in gpus],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": True,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "root_seed_family": [20260719, 9300 + 100 * args.q],
            "wall_seconds": compile_and_first_wall_s
            + sum(batched_walls)
            + sum(serial_walls)
            + sum(threaded_walls),
            "host_ru_maxrss_bytes": rss,
            "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
            "gpu_allocator_memory": gpu_memory,
            "source_hashes": source_hashes,
            "plan": PLAN.as_posix(),
        },
        "nonclaims": [
            "no HMC convergence claim",
            "no posterior correctness claim",
            "no HMC tuning claim",
            "no NeuTra transport-quality claim",
            "native divergence unavailability is not zero divergences",
            "timing differences are descriptive execution evidence",
        ],
    }
    output = ROOT / ARTIFACT_ROOT / f"hmc-topology-q{args.q}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(1, 20), required=True)
    parser.add_argument("--num-results", type=int, default=4)
    parser.add_argument("--num-burnin-steps", type=int, default=2)
    parser.add_argument("--num-leapfrog-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

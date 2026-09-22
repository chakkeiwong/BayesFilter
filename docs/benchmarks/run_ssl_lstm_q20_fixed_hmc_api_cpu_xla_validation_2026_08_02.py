#!/usr/bin/env python3
"""q=20 fixed-transport HMC API canary and sequential CPU/XLA campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import select
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-plan-2026-08-02.md"
)
CHECKPOINTS = {
    "chart-a": Path(
        "docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/"
        "r1/seed-a/seed-a/checkpoint-1500.json"
    ),
    "chart-b": Path(
        "docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/"
        "r1/seed-b/seed-b/checkpoint-2500.json"
    ),
}
CHART_CPUS = {"chart-a": tuple(range(16)), "chart-b": tuple(range(16, 32))}
SEQUENTIAL_CHAIN_CPUS = {
    "chart-a": (0, 1, 2, 3),
    "chart-b": (4, 5, 6, 7),
}
SUPERVISOR_CPU = 32
CAMPAIGN_CAP_SECONDS = 20_000.0
PRIOR_CANARY_CHARGE_SECONDS = 1_900.0
SEQUENTIAL_CHUNK_RESULTS = 40
CHAIN_COUNT = 4
WORKER_START_TIMEOUT_SECONDS = 300.0
WORKER_CHUNK_TIMEOUT_SECONDS = 600.0
SCHEMA = "bayesfilter.ssl_lstm.q20_fixed_hmc_api_cpu_xla_validation.v1"


class CampaignError(RuntimeError):
    pass


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CampaignError(f"artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(payload))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _configure_tensorflow(threads: int) -> Any:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise CampaignError("CPU/XLA worker found a visible GPU")
    return tf


def _build_chart(label: str, *, threads: int) -> tuple[Any, Any, Mapping[str, Any]]:
    tf = _configure_tensorflow(threads)
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_training import (
        NeuTraReverseKLTrainer,
        ssl_lstm_tuned_capacity_neutra_config,
    )
    from bayesfilter.inference.neutra_training_control import (
        validate_joint_training_checkpoint,
    )
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
        FREE_NAMES,
        PRIOR_CENTER,
    )

    checkpoint_path = (ROOT / CHECKPOINTS[label]).resolve()
    checkpoint = _read_json(checkpoint_path)
    validate_joint_training_checkpoint(checkpoint)
    best_state = checkpoint["best_trainer_state"]
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    initialization_seed = (20260719, 12101 if label == "chart-a" else 12102)
    trainer_config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy().tolist()),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=0.0004,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=initialization_seed,
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, trainer_config)
    trainer.restore_state(best_state)
    frozen = trainer.frozen_transport_payload(
        transport_id=f"{label}-cpu-xla-best-{best_state['step']}-fixed-hmc-api",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )

    class Bridge:
        parameter_dim = 4
        parameter_names = tuple(FREE_NAMES)
        supports_retained_draw_batch = False
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True
        target_status_invalid_rows_become_nonfinite = True

        def __init__(self) -> None:
            self.target_scope = f"{target.target_scope}:fixed_hmc_api:{label}"

        def adapter_signature(self) -> str:
            return target.adapter_signature()

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="ssl_lstm_q20_fixed_hmc_api_cpu_xla_bridge",
                evidence_path=PLAN.as_posix(),
                target_scope=self.target_scope,
                nonclaims=(
                    "CPU/XLA exception to repository GPU default",
                    "no posterior oracle",
                ),
            )

        def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
            tensor = tf.convert_to_tensor(values, tf.float64)
            if tensor.shape.rank != 2 or tensor.shape[-1] != 4:
                raise ValueError("q=20 HMC target requires static shape [batch,4]")
            return target.batch_value_and_score(tensor)

        def log_prob_and_grad_status(
            self, values: Any
        ) -> tuple[Any, Any, Mapping[str, Any]]:
            return target.neutra_batch_log_prob_and_grad_status(values)

        def target_status_telemetry(self, values: Any) -> Mapping[str, Any]:
            return self.log_prob_and_grad_status(values)[2]

    provenance = {
        "chart": label,
        "checkpoint_path": CHECKPOINTS[label].as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "best_trainer_state_step": int(best_state["step"]),
        "best_trainer_state_hash": best_state["state_hash"],
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "transport_hash": loaded.manifest.transport_hash,
        "transport_artifact_signature": loaded.artifact_signature,
    }
    return Bridge(), loaded.transport, provenance


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "worktree_dirty": bool(status), "status": status.splitlines()}


def _worker_manifest(label: str, *, mode: str, threads: int) -> Mapping[str, Any]:
    import tensorflow as tf
    import tensorflow_probability as tfp

    return {
        "schema": SCHEMA,
        "role": "chart_worker",
        "mode": mode,
        "chart": label,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "configured_intra_op_threads": threads,
        "configured_inter_op_threads": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "dtype": "float64",
        "jit_compile": True,
        "cpu_only_intentional": True,
        "runtime_numerical_backend": "tensorflow_tfp_only",
        "git": _git_manifest(),
        "plan": PLAN.as_posix(),
    }


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _read_worker_response(
    process: subprocess.Popen[str], *, timeout_seconds: float
) -> dict[str, Any]:
    if process.stdout is None:
        raise CampaignError("sequential worker stdout is unavailable")
    ready, _, _ = select.select([process.stdout], [], [], float(timeout_seconds))
    if not ready:
        raise CampaignError(f"sequential worker {process.pid} timed out")
    line = process.stdout.readline()
    if not line:
        raise CampaignError(f"sequential worker {process.pid} closed stdout")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise CampaignError("sequential worker response must be a JSON object")
    return payload


def _send_worker_request(
    process: subprocess.Popen[str], payload: Mapping[str, Any]
) -> None:
    if process.stdin is None:
        raise CampaignError("sequential worker stdin is unavailable")
    process.stdin.write(json.dumps(_json_ready(payload), allow_nan=False) + "\n")
    process.stdin.flush()


def _fold_chain_seed(seed: tuple[int, int], chain_index: int) -> tuple[int, int]:
    if len(seed) != 2:
        raise CampaignError("controller seed must contain exactly two integers")
    index = int(chain_index)
    if not 0 <= index < CHAIN_COUNT:
        raise CampaignError("chain index is outside the four-chain bank")
    # The controller owns the chunk seed; the worker lane only creates a
    # deterministic disjoint substream for each independent frozen chain.
    return int(seed[0]) + index + 1, int(seed[1]) + 100_003 * (index + 1)


def _reassemble_worker_chunk(
    rows: Sequence[Mapping[str, Any]], *, active_results: int, dimension: int
) -> tuple[Any, Mapping[str, Any]]:
    import tensorflow as tf

    ordered = tuple(sorted(rows, key=lambda row: int(row["chain_index"])))
    if tuple(int(row["chain_index"]) for row in ordered) != tuple(range(CHAIN_COUNT)):
        raise CampaignError("sequential worker responses do not cover chains 0..3")
    samples = tf.stack(
        [tf.convert_to_tensor(row["samples"], tf.float64) for row in ordered],
        axis=1,
    )
    expected_shape = (int(active_results), CHAIN_COUNT, int(dimension))
    if tuple(samples.shape) != expected_shape:
        raise CampaignError(
            f"reassembled sample shape {tuple(samples.shape)} != {expected_shape}"
        )

    def stack_trace(name: str, dtype: Any, *, trailing: bool = False) -> Any:
        tensors = [tf.convert_to_tensor(row["trace"][name], dtype) for row in ordered]
        axis = 1
        result = tf.stack(tensors, axis=axis)
        expected = (
            (int(active_results), CHAIN_COUNT, int(dimension))
            if trailing
            else (int(active_results), CHAIN_COUNT)
        )
        if tuple(result.shape) != expected:
            raise CampaignError(
                f"reassembled trace {name} shape {tuple(result.shape)} != {expected}"
            )
        return result

    trace = {
        "is_accepted": stack_trace("is_accepted", tf.bool),
        "log_accept_ratio": stack_trace("log_accept_ratio", tf.float64),
        "target_log_prob": stack_trace("target_log_prob", tf.float64),
        "proposed_target_log_prob": stack_trace(
            "proposed_target_log_prob", tf.float64
        ),
        "target_score": stack_trace("target_score", tf.float64, trailing=True),
        "delta_h": stack_trace("delta_h", tf.float64),
        "target_status_code": stack_trace("target_status_code", tf.int32),
        "target_valid_pre_regularized_score": stack_trace(
            "target_valid_pre_regularized_score", tf.bool
        ),
    }
    return samples, trace


def _load_admitted_kernel(tuning_root: Path, label: str) -> Mapping[str, Any]:
    summary_path = tuning_root / label / "summary.json"
    tuning_path = tuning_root / label / "tuning-result.json"
    summary = _read_json(summary_path)
    tuning = _read_json(tuning_path)
    if summary.get("status") != "KERNEL_ADMITTED" or tuning.get("passed") is not True:
        raise CampaignError(f"{label} has no admitted tuning result")
    kernel = tuning.get("final_kernel_payload")
    if not isinstance(kernel, Mapping):
        raise CampaignError(f"{label} tuning artifact has no final kernel")
    if int(kernel.get("num_leapfrog_steps", 0)) < 2:
        raise CampaignError("sequential kernel must have at least two leapfrog steps")
    if kernel.get("mass_policy") != "fixed_identity_z":
        raise CampaignError("sequential kernel mass policy is not fixed identity z")
    if kernel.get("use_xla") is not True:
        raise CampaignError("sequential kernel was not tuned with XLA")
    if kernel.get("shared_scalar_step_across_chain_bank") is not True:
        raise CampaignError("tuning did not use a shared scalar chain-bank step")
    return kernel


def _archive_json_ready_result(path: Path, payload: Mapping[str, Any]) -> None:
    private_keys = {
        "private_warmup_z",
        "private_warmup_raw",
        "private_retained_z",
        "private_retained_raw",
    }
    public = {key: value for key, value in payload.items() if key not in private_keys}
    _write_json(path, public)


def _run_chart_canary(args: argparse.Namespace) -> int:
    tf = _configure_tensorflow(args.threads)
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
    )
    from bayesfilter.inference.neutra_hmc import (
        SequentialNeuTraHMCConfig,
        _chain_moved,
        _ChunkRunner,
        _target_status,
    )

    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("chart canary output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base, transport, provenance = _build_chart(args.chart, threads=args.threads)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=f"{base.target_scope}:sequential_canary",
        runtime_backend="ssl_lstm_q20_fixed_hmc_api_cpu_xla_sequential_canary",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "representative sequential timing canary only",
            "no convergence or posterior claim",
        ),
    )
    rows = []
    for leapfrog in (2, 4):
        config = SequentialNeuTraHMCConfig(
            step_size=0.01,
            num_leapfrog_steps=leapfrog,
            seed=(20260802, 8200 + leapfrog),
            chain_count=4,
            warmup_chunk_size=args.results,
            warmup_min_results=args.results,
            warmup_window_results=args.results,
            warmup_max_results=args.results,
            retained_chunk_size=args.results,
            retained_min_results=args.results,
            retained_max_results=args.results,
            bulk_ess_min=1.0,
            tail_ess_min=1.0,
            acceptance_min=0.0,
            acceptance_max=1.0,
            use_xla=True,
        )
        initial_state = tf.zeros((4, 4), tf.float64)
        runner = _ChunkRunner(adapter, initial_state, config)
        cold_started = time.perf_counter()
        cold_samples, cold_trace = runner.run(
            initial_state, tf.constant((20260802, 8300 + leapfrog), tf.int32)
        )
        cold_seconds = time.perf_counter() - cold_started
        cold_audit_started = time.perf_counter()
        cold_status = _target_status(adapter, cold_samples)
        cold_audit_seconds = time.perf_counter() - cold_audit_started
        warm_state = tf.convert_to_tensor(cold_samples[-1], tf.float64)
        warm_started = time.perf_counter()
        warm_samples, warm_trace = runner.run(
            warm_state, tf.constant((20260802, 8400 + leapfrog), tf.int32)
        )
        warm_seconds = time.perf_counter() - warm_started
        warm_audit_started = time.perf_counter()
        warm_status = _target_status(adapter, warm_samples)
        warm_audit_seconds = time.perf_counter() - warm_audit_started
        warm_log_accept = tf.convert_to_tensor(
            warm_trace["log_accept_ratio"], tf.float64
        )
        warm_total = warm_seconds + warm_audit_seconds
        row = {
            "leapfrog_steps": leapfrog,
            "sequential_policy_id": config.payload()["policy_id"],
            "sequential_config": config.payload(),
            "representative_results_per_chain": args.results,
            "representative_chain_count": 4,
            "cold_compile_and_chunk_seconds": cold_seconds,
            "cold_status_compile_and_audit_seconds": cold_audit_seconds,
            "warm_chunk_seconds": warm_seconds,
            "warm_status_audit_seconds": warm_audit_seconds,
            "warm_chunk_plus_status_seconds": warm_total,
            "seconds_per_transition_leapfrog_warm": (
                warm_total / (args.results * leapfrog)
            ),
            "minimum_3000_transition_projection_seconds": (
                warm_total * 3000.0 / args.results
            ),
            "cold_status": cold_status,
            "warm_status": warm_status,
            "warm_samples_all_finite": bool(
                tf.reduce_all(tf.math.is_finite(warm_samples)).numpy()
            ),
            "warm_log_accept_all_finite": bool(
                tf.reduce_all(tf.math.is_finite(warm_log_accept)).numpy()
            ),
            "warm_acceptance_probability_by_chain": _json_ready(
                tf.reduce_mean(
                    tf.exp(tf.minimum(warm_log_accept, 0.0)), axis=0
                )
            ),
            "warm_chain_moved": _json_ready(
                _chain_moved(warm_state, warm_samples)
            ),
            "native_divergence_status": "not_exposed_by_kernel",
            "native_divergence_count": None,
            "native_divergence_interpretation": (
                "unavailable is not zero divergences"
            ),
            "finite_delta_h_tail_role": "explanatory_only",
            "movement_role": "explanatory_only",
        }
        rows.append(row)
        _write_json(output / f"l{leapfrog}-timing.json", row)
    result = {
        **_worker_manifest(args.chart, mode="canary", threads=args.threads),
        "status": "CANARY_COMPLETED",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "representative_results_per_chain": args.results,
        "chart_provenance": provenance,
        "rows": rows,
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "nonclaims": [
            "mechanics and timing canary only",
            "step size 0.01 is mechanics-only and ineligible for kernel promotion",
            "no convergence, posterior, ranking, or scientific claim",
        ],
    }
    _write_json(output / "canary-result.json", result)
    print(json.dumps({"status": result["status"], "chart": args.chart, "rows": rows}))
    return 0


def _run_canary_supervisor(args: argparse.Namespace) -> int:
    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("canary supervisor output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    affinity_probes = {}
    for label, cpus in {**CHART_CPUS, "supervisor": (SUPERVISOR_CPU,)}.items():
        specification = (
            str(cpus[0]) if len(cpus) == 1 else f"{cpus[0]}-{cpus[-1]}"
        )
        probe = subprocess.run(
            ["taskset", "-c", specification, "true"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        affinity_probes[label] = {
            "specification": specification,
            "exit_code": probe.returncode,
        }
        if probe.returncode != 0:
            raise CampaignError(f"canary cannot assign {label} CPUs {specification}")
    started = time.perf_counter()
    processes = {}
    logs = {}
    commands = {}
    for label, cpus in CHART_CPUS.items():
        chart_root = output / label
        command = [
            "taskset",
            "-c",
            f"{cpus[0]}-{cpus[-1]}",
            sys.executable,
            str(SCRIPT),
            "--mode",
            "chart-canary",
            "--chart",
            label,
            "--threads",
            str(len(cpus)),
            "--results",
            str(args.results),
            "--output-root",
            str(chart_root.relative_to(ROOT)),
        ]
        log = (output / f"{label}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        commands[label] = command
        processes[label] = process
        logs[label] = log
    terminal = {}
    while processes:
        elapsed = time.perf_counter() - started
        if elapsed >= args.cap_seconds:
            for process in processes.values():
                process.terminate()
            break
        for label, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            logs[label].close()
            terminal[label] = {"pid": process.pid, "exit_code": code}
            del processes[label]
        if processes:
            time.sleep(1.0)
    for label, process in list(processes.items()):
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        logs[label].close()
        terminal[label] = {"pid": process.pid, "exit_code": process.returncode}
    chart_results = {}
    for label in CHART_CPUS:
        path = output / label / "canary-result.json"
        chart_results[label] = _read_json(path) if path.is_file() else None
    projections = [
        float(row["minimum_3000_transition_projection_seconds"])
        for result in chart_results.values()
        if result is not None
        for row in result["rows"]
        if row.get("minimum_3000_transition_projection_seconds") is not None
    ]
    elapsed = time.perf_counter() - started
    remaining = max(0.0, CAMPAIGN_CAP_SECONDS - elapsed)
    minimum_projection = max(projections) if projections else None
    # Reserve 20% for tuning, diagnostics, archives, and supervisor closeout.
    projected_with_reserve = (
        None if minimum_projection is None else minimum_projection * 1.20
    )
    feasible = bool(
        all(row["exit_code"] == 0 for row in terminal.values())
        and len(terminal) == 2
        and projected_with_reserve is not None
        and projected_with_reserve <= remaining
    )
    status = "CANARY_PASS_FULL_CAMPAIGN_FEASIBLE" if feasible else "UNDER_BUDGETED"
    summary = {
        "schema": SCHEMA,
        "role": "canary_supervisor",
        "status": status,
        "plan": PLAN.as_posix(),
        "commands": commands,
        "affinity_probes": affinity_probes,
        "preflight_retry_provenance": {
            "attempt_0_status": "preflight_failed_before_workers_started",
            "attempt_0_failure": (
                "supervisor incorrectly treated its CPU-32 affinity as the machine-wide allowed set"
            ),
            "repair": "probe each child taskset range directly",
            "scientific_contract_changed": False,
            "compute_budget_changed": False,
        },
        "terminal": terminal,
        "wall_seconds": elapsed,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "remaining_campaign_seconds": remaining,
        "maximum_minimum_3000_transition_projection_seconds": minimum_projection,
        "projected_with_20pct_tuning_diagnostic_reserve_seconds": projected_with_reserve,
        "projection_role": "continuation_decision_not_performance_ranking",
        "chart_result_paths": {
            label: str((output / label / "canary-result.json").relative_to(ROOT))
            for label in CHART_CPUS
        },
        "chart_results": chart_results,
        "full_campaign_launched": False,
        "nonclaims": [
            "canary projection only",
            "under-budgeted does not reject the target, chart, kernel family, or NeuTra",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"status": status, "wall_seconds": elapsed, "projection": projected_with_reserve}))
    return 0 if feasible else 3


def _run_one_chain_canary(args: argparse.Namespace) -> int:
    tf = _configure_tensorflow(1)
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
    )
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, _ChunkRunner

    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("one-chain canary output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base, transport, provenance = _build_chart(args.chart, threads=1)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=f"{base.target_scope}:one_chain_sequential_canary",
        runtime_backend="ssl_lstm_q20_fixed_hmc_api_cpu_xla_one_chain_canary",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=("one-chain frozen-kernel timing canary only",),
    )
    adapter.target_status_invalid_rows_become_nonfinite = True
    config = SequentialNeuTraHMCConfig(
        step_size=0.01,
        num_leapfrog_steps=2,
        seed=(20260802, 9100 + args.chain_index),
        chain_count=4,
        warmup_chunk_size=args.results,
        warmup_min_results=args.results,
        warmup_window_results=args.results,
        warmup_max_results=args.results,
        retained_chunk_size=args.results,
        retained_min_results=args.results,
        retained_max_results=args.results,
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
        acceptance_min=0.0,
        acceptance_max=1.0,
        use_xla=True,
    )
    initial = tf.zeros((1, 4), tf.float64)
    runner = _ChunkRunner(adapter, initial, config)
    cold_started = time.perf_counter()
    cold_samples, _cold_trace = runner.run(
        initial,
        tf.constant((20260802, 9200 + args.chain_index), tf.int32),
    )
    cold_seconds = time.perf_counter() - cold_started
    warm_state = tf.convert_to_tensor(cold_samples[-1], tf.float64)
    warm_started = time.perf_counter()
    warm_samples, warm_trace = runner.run(
        warm_state,
        tf.constant((20260802, 9300 + args.chain_index), tf.int32),
    )
    warm_seconds = time.perf_counter() - warm_started
    log_accept = tf.convert_to_tensor(warm_trace["log_accept_ratio"], tf.float64)
    target = tf.convert_to_tensor(warm_trace["target_log_prob"], tf.float64)
    score = tf.convert_to_tensor(warm_trace["target_score"], tf.float64)
    result = {
        **_worker_manifest(args.chart, mode="one-chain-canary", threads=1),
        "status": "ONE_CHAIN_CANARY_COMPLETED",
        "chain_index": args.chain_index,
        "chart_provenance": provenance,
        "cold_compile_and_chunk_seconds": cold_seconds,
        "warm_chunk_seconds": warm_seconds,
        "results_per_chain": args.results,
        "leapfrog_steps": 2,
        "minimum_3000_transition_projection_seconds": (
            warm_seconds * 3000.0 / args.results
        ),
        "warm_samples_all_finite": bool(
            tf.reduce_all(tf.math.is_finite(warm_samples)).numpy()
        ),
        "warm_log_accept_all_finite": bool(
            tf.reduce_all(tf.math.is_finite(log_accept)).numpy()
        ),
        "warm_target_all_finite": bool(
            tf.reduce_all(tf.math.is_finite(target)).numpy()
        ),
        "warm_target_score_all_finite": bool(
            tf.reduce_all(tf.math.is_finite(score)).numpy()
        ),
        "warm_acceptance_probability": float(
            tf.reduce_mean(tf.exp(tf.minimum(log_accept, 0.0))).numpy()
        ),
        "status_semantics": (
            "q20 batch target converts every status-invalid row to nonfinite value and score"
        ),
        "native_divergence_status": "not_exposed_by_kernel",
        "native_divergence_count": None,
        "native_divergence_interpretation": "unavailable is not zero divergences",
        "wall_seconds": time.perf_counter() - started,
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "nonclaims": [
            "step 0.01 timing canary only",
            "no kernel promotion, convergence, posterior, or scientific claim",
        ],
    }
    _write_json(output / "result.json", result)
    print(json.dumps({"status": result["status"], "warm_seconds": warm_seconds}))
    return 0


def _run_distributed_canary_supervisor(args: argparse.Namespace) -> int:
    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("distributed canary output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    assignments = []
    for chart_index, label in enumerate(("chart-a", "chart-b")):
        for chain_index in range(4):
            assignments.append((label, chain_index, chart_index * 4 + chain_index))
    started = time.perf_counter()
    processes = {}
    logs = {}
    commands = {}
    for label, chain_index, cpu in assignments:
        worker_id = f"{label}-chain-{chain_index}"
        worker_root = output / worker_id
        command = [
            "taskset", "-c", str(cpu), sys.executable, str(SCRIPT),
            "--mode", "one-chain-canary",
            "--chart", label,
            "--chain-index", str(chain_index),
            "--results", str(args.results),
            "--output-root", str(worker_root.relative_to(ROOT)),
        ]
        log = (output / f"{worker_id}.log").open("w", encoding="utf-8")
        processes[worker_id] = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        logs[worker_id] = log
        commands[worker_id] = command
    terminal = {}
    while processes:
        if time.perf_counter() - started >= args.cap_seconds:
            for process in processes.values():
                process.terminate()
            break
        for worker_id, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            logs[worker_id].close()
            terminal[worker_id] = {"pid": process.pid, "exit_code": code}
            del processes[worker_id]
        if processes:
            time.sleep(1.0)
    for worker_id, process in list(processes.items()):
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        logs[worker_id].close()
        terminal[worker_id] = {"pid": process.pid, "exit_code": process.returncode}
    results = {}
    for label, chain_index, _cpu in assignments:
        worker_id = f"{label}-chain-{chain_index}"
        path = output / worker_id / "result.json"
        results[worker_id] = _read_json(path) if path.is_file() else None
    valid = [row for row in results.values() if row is not None]
    maximum_projection = (
        None
        if len(valid) != len(assignments)
        else max(float(row["minimum_3000_transition_projection_seconds"]) for row in valid)
    )
    wall = time.perf_counter() - started
    remaining = max(0.0, CAMPAIGN_CAP_SECONDS - wall)
    projected_with_reserve = (
        None if maximum_projection is None else maximum_projection * 1.20
    )
    feasible = bool(
        len(terminal) == len(assignments)
        and all(row["exit_code"] == 0 for row in terminal.values())
        and projected_with_reserve is not None
        and projected_with_reserve <= remaining
    )
    summary = {
        "schema": SCHEMA,
        "role": "distributed_one_chain_canary_supervisor",
        "status": (
            "CANARY_PASS_FULL_CAMPAIGN_FEASIBLE" if feasible else "UNDER_BUDGETED_OR_INCOMPLETE"
        ),
        "commands": commands,
        "terminal": terminal,
        "worker_results": results,
        "wall_seconds": wall,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "remaining_campaign_seconds": remaining,
        "maximum_minimum_3000_transition_projection_seconds": maximum_projection,
        "projected_with_20pct_tuning_diagnostic_reserve_seconds": projected_with_reserve,
        "full_campaign_launched": False,
        "plan": PLAN.as_posix(),
        "nonclaims": [
            "frozen one-chain timing canary only",
            "no kernel promotion or sampler claim",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "projection": projected_with_reserve}))
    return 0 if feasible else 3


def _run_chart_tuning(args: argparse.Namespace) -> int:
    tf = _configure_tensorflow(args.threads)
    from bayesfilter.inference.fixed_transport_hmc_tuning import (
        FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )

    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("chart tuning output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base, transport, provenance = _build_chart(args.chart, threads=args.threads)
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.5,
        leapfrog_grid=(2,),
        chain_count=4,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
        repair_band=(0.55, 0.85),
        selection_policy="acceptance_target_distance",
        selection_replications=1,
        budget_schedule=(8, 16, 32),
        tune_num_results=8,
        screen_num_results=16,
        screen_num_burnin_steps=4,
        verification_num_results=64,
        verification_num_burnin_steps=16,
        target_status_trace_policy="per_chain_step",
        target_scope=f"{base.target_scope}:claim_tuning_l2",
        tuning_policy=FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        use_xla=True,
        output_filename="tuning-result.json",
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=tf.zeros((4,), tf.float64),
        config=config,
        output_dir=output,
    )
    summary = {
        **_worker_manifest(args.chart, mode="chart-tuning", threads=args.threads),
        "status": "KERNEL_ADMITTED" if result.passed else "NO_VIABLE_KERNEL",
        "wall_seconds": time.perf_counter() - started,
        "chart_provenance": provenance,
        "tuning_artifact_path": result.artifact_path,
        "tuning_artifact_sha256": (
            None
            if result.artifact_path is None
            else _sha256(Path(result.artifact_path))
        ),
        "final_kernel": result.final_kernel_payload,
        "hard_vetoes": result.hard_vetoes,
        "repair_triggers": result.repair_triggers,
        "candidate_count": len(result.candidates),
        "nonclaims": [
            "kernel tuning and fresh acceptance verification only",
            "no convergence, posterior, chart ranking, or scientific claim",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"chart": args.chart, "status": summary["status"]}))
    return 0


def _run_tuning_supervisor(args: argparse.Namespace) -> int:
    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("tuning supervisor output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processes = {}
    logs = {}
    commands = {}
    for label, cpus in CHART_CPUS.items():
        command = [
            "taskset", "-c", f"{cpus[0]}-{cpus[-1]}",
            sys.executable, str(SCRIPT),
            "--mode", "chart-tuning",
            "--chart", label,
            "--threads", str(len(cpus)),
            "--output-root", str((output / label).relative_to(ROOT)),
        ]
        log = (output / f"{label}.log").open("w", encoding="utf-8")
        processes[label] = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        logs[label] = log
        commands[label] = command
    terminal = {}
    while processes:
        if time.perf_counter() - started >= args.cap_seconds:
            for process in processes.values():
                process.terminate()
            break
        for label, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            logs[label].close()
            terminal[label] = {"pid": process.pid, "exit_code": code}
            del processes[label]
        if processes:
            time.sleep(1.0)
    for label, process in list(processes.items()):
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        logs[label].close()
        terminal[label] = {"pid": process.pid, "exit_code": process.returncode}
    charts = {}
    for label in CHART_CPUS:
        path = output / label / "summary.json"
        charts[label] = _read_json(path) if path.is_file() else None
    admitted = [
        label for label, row in charts.items()
        if row is not None and row.get("status") == "KERNEL_ADMITTED"
    ]
    all_completed = bool(
        len(terminal) == 2
        and all(row["exit_code"] == 0 for row in terminal.values())
        and all(row is not None for row in charts.values())
    )
    summary = {
        "schema": SCHEMA,
        "role": "tuning_supervisor",
        "status": "TUNING_COMPLETED" if all_completed else "TUNING_INCOMPLETE",
        "commands": commands,
        "terminal": terminal,
        "charts": charts,
        "admitted_charts": admitted,
        "wall_seconds": time.perf_counter() - started,
        "cap_seconds": args.cap_seconds,
        "plan": PLAN.as_posix(),
        "full_sequential_launched": False,
        "nonclaims": [
            "tuning result only",
            "no convergence or posterior claim",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "admitted_charts": admitted}))
    return 0 if all_completed else 2


def _run_sequential_chain_worker(args: argparse.Namespace) -> int:
    tf = _configure_tensorflow(1)
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
    )
    from bayesfilter.inference.neutra_hmc import (
        SequentialNeuTraHMCConfig,
        _ChunkRunner,
    )

    base, transport, provenance = _build_chart(args.chart, threads=1)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=f"{base.target_scope}:claim_tuning_l2",
        runtime_backend="ssl_lstm_q20_fixed_hmc_api_cpu_xla_sequential_worker",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "frozen one-chain execution shard only",
            "no independent adaptation or kernel selection",
        ),
    )
    adapter.target_status_invalid_rows_become_nonfinite = True
    tuning_root = (ROOT / args.tuning_root).resolve()
    kernel = _load_admitted_kernel(tuning_root, args.chart)
    expected = {
        "transformed_adapter_signature": adapter.adapter_signature(),
        "base_adapter_signature": base.adapter_signature(),
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
    }
    for name, value in expected.items():
        if str(kernel.get(name)) != str(value):
            raise CampaignError(f"sequential worker kernel {name} mismatch")
    if float(kernel["step_size"]) != float(args.step_size):
        raise CampaignError("sequential worker step size differs from admitted kernel")
    if int(kernel["num_leapfrog_steps"]) != int(args.leapfrog_steps):
        raise CampaignError("sequential worker trajectory differs from admitted kernel")

    config = SequentialNeuTraHMCConfig(
        step_size=float(args.step_size),
        num_leapfrog_steps=int(args.leapfrog_steps),
        seed=(20260802, 10_000 + int(args.chain_index)),
        chain_count=CHAIN_COUNT,
        warmup_chunk_size=int(args.chunk_results),
        warmup_min_results=int(args.chunk_results),
        warmup_window_results=int(args.chunk_results),
        warmup_max_results=int(args.chunk_results),
        retained_chunk_size=int(args.chunk_results),
        retained_min_results=int(args.chunk_results),
        retained_max_results=int(args.chunk_results),
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
        acceptance_min=0.0,
        acceptance_max=1.0,
        use_xla=True,
    )
    initial = tf.zeros((1, 4), tf.float64)
    runner = _ChunkRunner(adapter, initial, config)
    ready = {
        **_worker_manifest(args.chart, mode="sequential-chain-worker", threads=1),
        "event": "ready",
        "chain_index": int(args.chain_index),
        "chart_provenance": provenance,
        "kernel": _json_ready(kernel),
        "kernel_binding": expected,
        "chunk_results": int(args.chunk_results),
        "execution_topology": "one_frozen_chain_per_persistent_cpu_xla_process",
    }
    print(json.dumps(_json_ready(ready), allow_nan=False), flush=True)

    invocation = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        request = json.loads(line)
        if not isinstance(request, dict):
            raise CampaignError("sequential worker request must be a JSON object")
        if request.get("command") == "stop":
            print(
                json.dumps(
                    {"event": "stopped", "chain_index": int(args.chain_index)}
                ),
                flush=True,
            )
            return 0
        if request.get("command") != "chunk":
            raise CampaignError("unknown sequential worker command")
        active = int(request["active_results"])
        if active != int(args.chunk_results):
            raise CampaignError("real sequential chunks must use the reviewed static size")
        state = tf.convert_to_tensor(request["state"], tf.float64)
        if tuple(state.shape) != (4,):
            raise CampaignError("sequential worker state must have shape [4]")
        seed = tuple(int(value) for value in request["seed"])
        started = time.perf_counter()
        samples, trace = runner.run(state[None, :], tf.constant(seed, tf.int32))
        elapsed = time.perf_counter() - started
        samples = tf.squeeze(tf.convert_to_tensor(samples, tf.float64), axis=1)

        def squeeze_trace(name: str, dtype: Any) -> Any:
            return tf.squeeze(tf.convert_to_tensor(trace[name], dtype), axis=1)

        response_trace = {
            "is_accepted": squeeze_trace("is_accepted", tf.bool),
            "log_accept_ratio": squeeze_trace("log_accept_ratio", tf.float64),
            "target_log_prob": squeeze_trace("target_log_prob", tf.float64),
            "proposed_target_log_prob": squeeze_trace(
                "proposed_target_log_prob", tf.float64
            ),
            "target_score": squeeze_trace("target_score", tf.float64),
            "delta_h": squeeze_trace("delta_h", tf.float64),
            "target_status_code": squeeze_trace("target_status_code", tf.int32),
            "target_valid_pre_regularized_score": squeeze_trace(
                "target_valid_pre_regularized_score", tf.bool
            ),
        }
        invocation += 1
        response = {
            "event": "chunk_done",
            "request_id": request["request_id"],
            "chart": args.chart,
            "chain_index": int(args.chain_index),
            "seed": seed,
            "active_results": active,
            "samples": samples,
            "trace": response_trace,
            "chunk_seconds": elapsed,
            "jit_compile": True,
            "affinity": sorted(os.sched_getaffinity(0)),
            "invocation": invocation,
            "all_required_tensors_finite": bool(
                tf.reduce_all(tf.math.is_finite(samples)).numpy()
                and tf.reduce_all(
                    tf.math.is_finite(response_trace["log_accept_ratio"])
                ).numpy()
                and tf.reduce_all(
                    tf.math.is_finite(response_trace["target_log_prob"])
                ).numpy()
                and tf.reduce_all(
                    tf.math.is_finite(response_trace["target_score"])
                ).numpy()
            ),
            "native_divergence_status": "not_exposed_by_kernel",
            "native_divergence_count": None,
        }
        print(json.dumps(_json_ready(response), allow_nan=False), flush=True)


def _terminate_sequential_workers(
    processes: Sequence[subprocess.Popen[str]], logs: Sequence[Any]
) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                _send_worker_request(process, {"command": "stop"})
            except (CampaignError, OSError):
                process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30.0)
    for log in logs:
        log.close()


def _start_sequential_workers(
    args: argparse.Namespace,
    *,
    output: Path,
    kernel: Mapping[str, Any],
) -> tuple[list[subprocess.Popen[str]], list[Any], tuple[Mapping[str, Any], ...]]:
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    try:
        for chain_index, cpu in enumerate(SEQUENTIAL_CHAIN_CPUS[args.chart]):
            command = [
                "taskset",
                "-c",
                str(cpu),
                sys.executable,
                str(SCRIPT),
                "--mode",
                "sequential-chain-worker",
                "--chart",
                args.chart,
                "--chain-index",
                str(chain_index),
                "--step-size",
                str(kernel["step_size"]),
                "--leapfrog-steps",
                str(kernel["num_leapfrog_steps"]),
                "--chunk-results",
                str(args.chunk_results),
                "--tuning-root",
                str(args.tuning_root),
                "--output-root",
                str(args.output_root),
            ]
            log = (output / f"chain-{chain_index}.stderr.log").open(
                "w", encoding="utf-8"
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_worker_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
            processes.append(process)
            logs.append(log)
        ready = tuple(
            _read_worker_response(
                process, timeout_seconds=WORKER_START_TIMEOUT_SECONDS
            )
            for process in processes
        )
        expected_cpus = SEQUENTIAL_CHAIN_CPUS[args.chart]
        for chain_index, row in enumerate(ready):
            if row.get("event") != "ready":
                raise CampaignError("sequential worker readiness event is invalid")
            if int(row.get("chain_index", -1)) != chain_index:
                raise CampaignError("sequential worker chain identity mismatch")
            if row.get("affinity") != [expected_cpus[chain_index]]:
                raise CampaignError("sequential worker affinity mismatch")
            if row.get("jit_compile") is not True or row.get("physical_gpus") != []:
                raise CampaignError("sequential worker is not CPU/XLA")
        return processes, logs, ready
    except Exception:
        _terminate_sequential_workers(processes, logs)
        raise


def _run_chart_sequential(args: argparse.Namespace) -> int:
    tf = _configure_tensorflow(1)
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
    )
    from bayesfilter.inference.neutra_hmc import (
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )

    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("chart sequential output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    tuning_root = (ROOT / args.tuning_root).resolve()
    kernel = _load_admitted_kernel(tuning_root, args.chart)
    base, transport, provenance = _build_chart(args.chart, threads=1)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=f"{base.target_scope}:claim_tuning_l2",
        runtime_backend="ssl_lstm_q20_fixed_hmc_api_cpu_xla_sequential_supervisor",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "CPU/XLA exception to repository GPU default",
            "finite-sample operational validation only",
        ),
    )
    adapter.target_status_invalid_rows_become_nonfinite = True
    binding = {
        "transformed_adapter_signature": adapter.adapter_signature(),
        "base_adapter_signature": base.adapter_signature(),
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
    }
    for name, value in binding.items():
        if str(kernel.get(name)) != str(value):
            raise CampaignError(f"chart sequential kernel {name} mismatch")

    config = SequentialNeuTraHMCConfig(
        step_size=float(kernel["step_size"]),
        num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
        seed=(20260802, 12_000 if args.chart == "chart-a" else 13_000),
        warmup_chunk_size=int(args.chunk_results),
        warmup_min_results=2000,
        warmup_window_results=1000,
        warmup_max_results=10000,
        retained_chunk_size=int(args.chunk_results),
        retained_min_results=1000,
        retained_max_results=10000,
        warmup_rhat_max=1.05,
        retained_rhat_max=1.01,
        bulk_ess_min=400.0,
        tail_ess_min=400.0,
        acceptance_min=0.35,
        acceptance_max=0.95,
        chain_count=CHAIN_COUNT,
        use_xla=True,
        target_status_required=True,
    )
    started = time.perf_counter()
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    worker_ready: tuple[Mapping[str, Any], ...] = ()
    requests: list[Mapping[str, Any]] = []
    try:
        processes, logs, worker_ready = _start_sequential_workers(
            args, output=output, kernel=kernel
        )
        request_index = 0

        def run_chunk(state: Any, seed: tuple[int, int], _config: Any):
            nonlocal request_index
            state_tensor = tf.convert_to_tensor(state, tf.float64)
            if tuple(state_tensor.shape) != (CHAIN_COUNT, 4):
                raise CampaignError("controller supplied a non-four-chain state bank")
            request_id = f"{args.chart}-chunk-{request_index:04d}"
            folded = tuple(
                _fold_chain_seed(tuple(seed), chain_index)
                for chain_index in range(CHAIN_COUNT)
            )
            for chain_index, process in enumerate(processes):
                _send_worker_request(
                    process,
                    {
                        "command": "chunk",
                        "request_id": request_id,
                        "active_results": int(args.chunk_results),
                        "state": state_tensor[chain_index],
                        "seed": folded[chain_index],
                    },
                )
            rows = tuple(
                _read_worker_response(
                    process, timeout_seconds=WORKER_CHUNK_TIMEOUT_SECONDS
                )
                for process in processes
            )
            for chain_index, row in enumerate(rows):
                if row.get("event") != "chunk_done":
                    raise CampaignError("sequential worker chunk event is invalid")
                if row.get("request_id") != request_id:
                    raise CampaignError("sequential worker request identity mismatch")
                if int(row.get("chain_index", -1)) != chain_index:
                    raise CampaignError("sequential worker chain response mismatch")
                if tuple(row.get("seed", ())) != folded[chain_index]:
                    raise CampaignError("sequential worker seed response mismatch")
                if row.get("all_required_tensors_finite") is not True:
                    raise CampaignError("sequential worker returned nonfinite tensors")
                if row.get("native_divergence_status") != "not_exposed_by_kernel":
                    raise CampaignError("unexpected native divergence status")
            requests.append(
                {
                    "request_id": request_id,
                    "controller_seed": tuple(seed),
                    "folded_chain_seeds": folded,
                    "worker_chunk_seconds": tuple(
                        float(row["chunk_seconds"]) for row in rows
                    ),
                }
            )
            request_index += 1
            return _reassemble_worker_chunk(
                rows, active_results=int(args.chunk_results), dimension=4
            )

        def budget_check(_transition_leapfrog_count: int) -> bool:
            return time.perf_counter() - started + 180.0 <= float(args.cap_seconds)

        result = run_sequential_neutra_hmc(
            adapter,
            tf.zeros((CHAIN_COUNT, 4), tf.float64),
            config,
            archive_root=output / "archive",
            archive_label=args.chart,
            budget_check=budget_check,
            run_chunk=run_chunk,
        )
        result_payload = result.payload()
        _archive_json_ready_result(output / "sequential-result.json", result_payload)
        summary = {
            **_worker_manifest(args.chart, mode="chart-sequential", threads=1),
            "status": (
                "SEQUENTIAL_ADMITTED" if result.passed else "SEQUENTIAL_NOT_ADMITTED"
            ),
            "chart_provenance": provenance,
            "kernel": _json_ready(kernel),
            "kernel_binding": binding,
            "worker_ready": worker_ready,
            "request_count": len(requests),
            "requests": requests,
            "wall_seconds": time.perf_counter() - started,
            "cap_seconds": float(args.cap_seconds),
            "sequential_result_path": str(
                (output / "sequential-result.json").relative_to(ROOT)
            ),
            "sequential_result_sha256": _sha256(
                output / "sequential-result.json"
            ),
            "stop_reason": result.stop_reason,
            "warmup_results_per_chain": result.warmup_results_per_chain,
            "retained_results_per_chain": result.retained_results_per_chain,
            "diagnostics": result.diagnostics,
            "archive": result.archive,
            "native_divergence_status": "not_exposed_by_kernel",
            "native_divergence_count": None,
            "nonclaims": [
                "finite-sample sequential screen only",
                "no posterior truth, chart ranking, or default-readiness claim",
                "native divergence unavailability is not zero divergences",
            ],
        }
        _write_json(output / "summary.json", summary)
        print(
            json.dumps(
                {
                    "chart": args.chart,
                    "status": summary["status"],
                    "stop_reason": result.stop_reason,
                }
            )
        )
        return 0
    finally:
        _terminate_sequential_workers(processes, logs)


def _run_sequential_supervisor(args: argparse.Namespace) -> int:
    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("sequential supervisor output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    tuning_root = (ROOT / args.tuning_root).resolve()
    tuning_summary = _read_json(tuning_root / "summary.json")
    admitted = tuple(str(label) for label in tuning_summary.get("admitted_charts", ()))
    if not admitted:
        summary = {
            "schema": SCHEMA,
            "role": "sequential_supervisor",
            "status": "NOT_LAUNCHED_NO_ADMITTED_KERNEL",
            "tuning_root": str(args.tuning_root),
            "admitted_charts": [],
            "wall_seconds": 0.0,
            "prior_campaign_seconds": float(args.prior_wall_seconds),
            "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
            "nonclaims": ["no sequential sampler evidence"],
        }
        _write_json(output / "summary.json", summary)
        print(json.dumps({"status": summary["status"]}))
        return 0
    remaining = CAMPAIGN_CAP_SECONDS - float(args.prior_wall_seconds)
    if remaining <= 180.0:
        raise CampaignError("insufficient cumulative campaign budget for one chunk")
    cap = min(float(args.cap_seconds), remaining)
    started = time.perf_counter()
    processes: dict[str, subprocess.Popen[str]] = {}
    logs: dict[str, Any] = {}
    commands: dict[str, Sequence[str]] = {}
    for label in admitted:
        chart_root = output / label
        command = [
            "taskset",
            "-c",
            str(SUPERVISOR_CPU),
            sys.executable,
            str(SCRIPT),
            "--mode",
            "chart-sequential",
            "--chart",
            label,
            "--chunk-results",
            str(args.chunk_results),
            "--tuning-root",
            str(args.tuning_root),
            "--cap-seconds",
            str(cap),
            "--output-root",
            str(chart_root.relative_to(ROOT)),
        ]
        log = (output / f"{label}.log").open("w", encoding="utf-8")
        processes[label] = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        logs[label] = log
        commands[label] = command
    terminal: dict[str, Mapping[str, Any]] = {}
    while processes:
        if time.perf_counter() - started >= cap:
            for process in processes.values():
                process.terminate()
            break
        for label, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            logs[label].close()
            terminal[label] = {"pid": process.pid, "exit_code": code}
            del processes[label]
        if processes:
            time.sleep(1.0)
    for label, process in list(processes.items()):
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        logs[label].close()
        terminal[label] = {"pid": process.pid, "exit_code": process.returncode}
    charts = {
        label: (
            _read_json(output / label / "summary.json")
            if (output / label / "summary.json").is_file()
            else None
        )
        for label in admitted
    }
    wall = time.perf_counter() - started
    completed = bool(
        len(terminal) == len(admitted)
        and all(row["exit_code"] == 0 for row in terminal.values())
        and all(row is not None for row in charts.values())
    )
    summary = {
        "schema": SCHEMA,
        "role": "sequential_supervisor",
        "status": "SEQUENTIAL_COMPLETED" if completed else "SEQUENTIAL_INCOMPLETE",
        "commands": commands,
        "terminal": terminal,
        "charts": charts,
        "admitted_charts": list(admitted),
        "sequentially_admitted_charts": [
            label
            for label, row in charts.items()
            if row is not None and row.get("status") == "SEQUENTIAL_ADMITTED"
        ],
        "wall_seconds": wall,
        "prior_campaign_seconds": float(args.prior_wall_seconds),
        "cumulative_campaign_seconds": float(args.prior_wall_seconds) + wall,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "cap_seconds": cap,
        "tuning_root": str(args.tuning_root),
        "plan": PLAN.as_posix(),
        "nonclaims": [
            "finite-sample sequential validation only",
            "no chart ranking or posterior truth claim",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "sequentially_admitted_charts": summary[
                    "sequentially_admitted_charts"
                ],
            }
        )
    )
    return 0 if completed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "canary-supervisor",
            "chart-canary",
            "one-chain-canary",
            "distributed-canary-supervisor",
            "chart-tuning",
            "tuning-supervisor",
            "sequential-chain-worker",
            "chart-sequential",
            "sequential-supervisor",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chart", choices=tuple(CHECKPOINTS))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--chain-index", type=int, default=0)
    parser.add_argument("--results", type=int, default=32)
    parser.add_argument("--cap-seconds", type=float, default=1800.0)
    parser.add_argument("--tuning-root", type=Path)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--leapfrog-steps", type=int)
    parser.add_argument("--chunk-results", type=int, default=SEQUENTIAL_CHUNK_RESULTS)
    parser.add_argument(
        "--prior-wall-seconds", type=float, default=PRIOR_CANARY_CHARGE_SECONDS
    )
    args = parser.parse_args()
    if args.results < 16:
        parser.error("--results must be at least 16 for a representative canary")
    if args.mode == "chart-canary":
        if args.chart is None:
            parser.error("--chart is required for chart-canary")
        return _run_chart_canary(args)
    if args.mode == "one-chain-canary":
        if args.chart is None:
            parser.error("--chart is required for one-chain-canary")
        if not 0 <= args.chain_index < 4:
            parser.error("--chain-index must be in [0, 3]")
        return _run_one_chain_canary(args)
    if args.mode == "distributed-canary-supervisor":
        if args.chart is not None:
            parser.error("--chart is not used by distributed-canary-supervisor")
        return _run_distributed_canary_supervisor(args)
    if args.mode == "chart-tuning":
        if args.chart is None:
            parser.error("--chart is required for chart-tuning")
        return _run_chart_tuning(args)
    if args.mode == "tuning-supervisor":
        if args.chart is not None:
            parser.error("--chart is not used by tuning-supervisor")
        return _run_tuning_supervisor(args)
    if args.mode == "sequential-chain-worker":
        if args.chart is None or args.tuning_root is None:
            parser.error("sequential-chain-worker requires --chart and --tuning-root")
        if args.step_size is None or args.leapfrog_steps is None:
            parser.error("sequential-chain-worker requires the frozen kernel")
        if not 0 <= args.chain_index < CHAIN_COUNT:
            parser.error("--chain-index must be in [0, 3]")
        return _run_sequential_chain_worker(args)
    if args.mode == "chart-sequential":
        if args.chart is None or args.tuning_root is None:
            parser.error("chart-sequential requires --chart and --tuning-root")
        if args.chunk_results <= 0:
            parser.error("--chunk-results must be positive")
        return _run_chart_sequential(args)
    if args.mode == "sequential-supervisor":
        if args.chart is not None or args.tuning_root is None:
            parser.error("sequential-supervisor requires --tuning-root and no --chart")
        if args.prior_wall_seconds < PRIOR_CANARY_CHARGE_SECONDS:
            parser.error("--prior-wall-seconds cannot omit the conservative canary charge")
        return _run_sequential_supervisor(args)
    if args.chart is not None:
        parser.error("--chart is not used by canary-supervisor")
    if not 0.0 < args.cap_seconds <= 3600.0:
        parser.error("--cap-seconds must be in (0, 3600]")
    return _run_canary_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())

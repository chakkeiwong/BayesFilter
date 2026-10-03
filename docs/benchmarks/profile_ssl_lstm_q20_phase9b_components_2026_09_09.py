#!/usr/bin/env python3
"""Bounded component cost localization; no sampler or numerical policy change."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
RECOVERY = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-component-profile-plan-2026-09-09.md"
COMPONENTS = ("transport", "physical_value_score", "transformed_value_score", "target_status")
REPEATS = 4
WORKER_CAP = 600.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.durable_tensor_checkpoint import CheckpointError, DurableTensorCheckpoint, durable_json


def load_recovery():
    spec = importlib.util.spec_from_file_location("phase9b_profile_recovery", RECOVERY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_receipt(result):
    if result.get("status") != "completed" or result.get("banks") != ["initial", "terminal"]:
        raise CheckpointError("component profile is incomplete")
    if result.get("memory_policy", {}).get("all_physical_devices_memory_growth") is not True:
        raise CheckpointError("component profile lacks memory-growth evidence")
    if set(result["components"]) != set(COMPONENTS):
        raise CheckpointError("component profile omits a component")
    for component in result["components"].values():
        if component["tracing_count"] != 1 or not component["hlo_sha256"]:
            raise CheckpointError("component lacks stable XLA evidence")
        if component["repeat_outputs_equal"] is not True or set(component["warmed_seconds"]) != {"initial", "terminal"}:
            raise CheckpointError("component repeatability evidence is incomplete")
        timings = [component["first_seconds"]]
        for times in component["warmed_seconds"].values():
            if len(times) != REPEATS:
                raise CheckpointError("component timing repetitions are incomplete")
            timings.extend(times)
        if any(not math.isfinite(value) or value <= 0 for value in timings):
            raise CheckpointError("component timings must be positive and finite")


def validate_values(tf, name, result, observation_dimension):
    leaves = result.items() if isinstance(result, dict) else enumerate(tf.nest.flatten(result))
    for field, tensor in leaves:
        if not tensor.dtype.is_floating:
            continue
        finite = tf.math.is_finite(tensor)
        if name == "target_status" and field == "min_innovation_eigen_gap" and observation_dimension == 1:
            finite = tf.logical_or(finite, tf.equal(tensor, tf.constant(float("inf"), tensor.dtype)))
        if not bool(tf.reduce_all(finite).numpy()):
            raise CheckpointError(f"nonfinite {name} component field {field}")
    if name == "target_status":
        if not bool(tf.reduce_all(tf.equal(result["status_code"], 0)).numpy()) or not bool(tf.reduce_all(result["valid_pre_regularized_score"]).numpy()):
            raise CheckpointError("component status invalid")


def worker(job_path):
    job = json.loads(Path(job_path).read_bytes())
    recovery = load_recovery()
    if recovery.source_hashes() != job["runtime_job"]["sources"] or file_hash(SCRIPT) != job["script_hash"] or file_hash(PLAN) != job["plan_hash"]:
        raise CheckpointError("component source/plan changed")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true" or os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu_uuid"]:
        raise CheckpointError("component worker memory-growth/device environment mismatch")
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    durable_json(output / "run-start.json", {"job": job, "command": sys.argv, "pid": os.getpid(), "started_unix": time.time()})
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    durable_json(output / "memory-policy.json", memory)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise CheckpointError("component worker requires one GPU")
    parallel = recovery.load_parallel()
    source = parallel._load_source()
    profile = replace(parallel._profile(source, Path(job["campaign_root"]), job["arm"], recovery.ARM_CAP_SECONDS), plan_path=recovery.PLAN)
    runtime = json.loads(Path(job["runtime_result"]).read_bytes())
    if file_hash(job["runtime_result"]) != job["runtime_result_hash"]:
        raise CheckpointError("runtime comparator checksum changed")
    with tf.device(logical[0].name):
        setup_root = Path(job["campaign_root"]) / "setup" / job["arm"] / "committed"
        if not all((setup_root / key / "bundle.json").is_file() for key in ("chart", "tuning-repair-r1")):
            raise CheckpointError("profile requires already committed chart and tuning stages")
        _, handoff, setup = recovery._setup(tf, job["runtime_job"], source, profile)
        if not all(row["replayed"] for row in setup["stages"]):
            raise CheckpointError("component localization may not construct fresh charts or tune")
        adapter = handoff.transformed_adapter
        observation_dimension = int(adapter.base_adapter.bridge.component_target.config.static_config.observation_dim)
        initial = tf.constant(profile.initial_state_bank, tf.float64)
        stream_root = Path(runtime["controller"]["stream_root"]) / "chunks"
        with DurableTensorCheckpoint(stream_root, runtime["controller"]["source_identity"]) as store:
            state = initial
            for index in range(2):
                from bayesfilter.inference.neutra_hmc import _shared_sequential_chunk_seed

                seed = _shared_sequential_chunk_seed(tuple(runtime["controller"]["config"]["warmup_seed"]), index)
                inputs = {"adapter_signature": adapter.adapter_signature(), "config": runtime["controller"]["config"],
                          "active_results": 500, "seed": seed, "initial_state_sha256": store.tensor_hash(state)}
                (samples, _), _metadata = store.load(f"chunk-{index:06d}", inputs)
                state = samples[-1]
            terminal = tf.identity(state)
        banks = {"initial": initial, "terminal": terminal}
        bank_hashes = {name: DurableTensorCheckpoint.tensor_hash(value) for name, value in banks.items()}
        for name, values in banks.items():
            with tf.device("/CPU:0"):
                tf.io.write_file(str(output / f"{name}-bank.tftensor"), tf.io.serialize_tensor(values))

        def transport(values):
            return adapter.latent_to_position(values), adapter.log_abs_det_jacobian(values)

        operations = {"transport": transport, "physical_value_score": adapter.base_adapter.log_prob_and_grad,
                      "transformed_value_score": adapter.log_prob_and_grad, "target_status": adapter.target_status_telemetry}
        physical_banks = {name: adapter.latent_to_position(values) for name, values in banks.items()}
        components = {}
        for name, operation in operations.items():
            program = tf.function(operation, input_signature=[tf.TensorSpec((4, 4), tf.float64)], jit_compile=True, autograph=False)
            active_banks = physical_banks if name == "physical_value_score" else banks

            def measured(values):
                call_started = time.monotonic()
                result = program(values)
                for tensor in tf.nest.flatten(result):
                    tensor.numpy()
                seconds = time.monotonic() - call_started
                validate_values(tf, name, result, observation_dimension)
                return result, seconds

            _, first_seconds = measured(active_banks["initial"])
            warmed = {}
            for bank_name, values in active_banks.items():
                times = []
                reference = None
                for _repeat in range(REPEATS):
                    observed, seconds = measured(values)
                    if reference is not None:
                        for previous, current in zip(tf.nest.flatten(reference), tf.nest.flatten(observed)):
                            if not bool(tf.reduce_all(tf.equal(previous, current)).numpy()):
                                raise CheckpointError("fixed-input component repeatability failed")
                    reference = observed
                    times.append(seconds)
                warmed[bank_name] = times
            hlo = str(program.experimental_get_compiler_ir(active_banks["initial"])(stage="hlo"))
            components[name] = {"first_seconds": first_seconds, "warmed_seconds": warmed,
                                "tracing_count": int(program.experimental_get_tracing_count()),
                                "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(), "repeat_outputs_equal": True}
            durable_json(output / f"{name}-receipt.json", components[name])
        allocator = tf.config.experimental.get_memory_info(logical[0].name)
    result = {"status": "completed", "job": job, "gpu_uuid": job["gpu_uuid"], "banks": list(banks),
              "bank_hashes": bank_hashes, "memory_policy": memory, "components": components,
              "tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__, "allocator": allocator,
              "elapsed_seconds": time.monotonic() - started, "setup_replayed": True,
              "observation_dimension": observation_dimension,
              "telemetry_sentinel": "positive_infinite_min_innovation_eigen_gap_means_no_eigenvalue_pair_when_observation_dimension_is_one",
              "jit_compile": True, "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
              "controller_seconds_per_transition": [row["compute_seconds"] / 500 for row in runtime["controller"]["chunks"]],
              "interpretation": "isolated descriptive localization; no additive cost identity or speedup claim"}
    validate_receipt(result)
    if recovery.source_hashes() != job["runtime_job"]["sources"] or file_hash(SCRIPT) != job["script_hash"]:
        raise CheckpointError("source changed during profiling")
    durable_json(output / "run_manifest.json", result)
    return 0


def coordinator(campaign):
    from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger
    from bayesfilter.runtime.display_gpu_policy import probe_inventory, select_gpus
    from bayesfilter.runtime.parallel_tuning import ParallelTuningTask, run_parallel_tuning_wave

    campaign = Path(campaign).resolve()
    recovery = load_recovery()
    with (campaign / ".coordinator.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = CampaignBudgetLedger(campaign / "campaign_budget_ledger.json")
        if ledger.read()["reserved_seconds"]:
            raise CheckpointError("outstanding campaign reservation must be reconciled")
        runtime_wave = json.loads((campaign / "runtime-complete.json").read_bytes())
        runtime_results = recovery._read_results(runtime_wave)
        sources = recovery.source_hashes()
        if any(sources != row["job"]["sources"] for row in runtime_results.values()):
            raise CheckpointError("profile would change the measured runtime sources")
        output = campaign / "launches" / f"component-profile-{uuid.uuid4().hex[:10]}"
        output.mkdir(parents=True, exist_ok=False)
        binding = {key: ledger.read()[key] for key in ("source_hash", "plan_hash")}
        ledger.start_attempt(attempt_id=output.name, output_root=output, seed_namespace={"no_new_random_draws": True}, **binding)
        reserved, settled, inflight = set(), set(), set()
        rows = {}
        error = None
        durable_json(output / "profile-start.json", {"command": sys.argv, "sources": sources,
                    "plan": str(PLAN), "plan_hash": file_hash(PLAN), "script_hash": file_hash(SCRIPT),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "python": sys.executable, "budget_before": ledger.read()})
        try:
            for arm in recovery.ARMS:
                ledger.reserve_chunk(attempt_id=output.name, arm=arm, chunk_index=0, reserve_seconds=WORKER_CAP,
                    output_root=output, seed_namespace={"no_new_random_draws": True}, **binding)
                reserved.add(arm)
            pending = list(recovery.ARMS)
            while pending:
                selection = select_gpus(probe_inventory(), 2, estimated_peak_mib=4096)
                tasks = []
                for arm in pending:
                    runtime = runtime_results[arm]
                    device = next((item for item in selection["selected"] if item["uuid"] == runtime["gpu_uuid"]), None)
                    if device is None:
                        continue
                    job = {"campaign_root": str(campaign), "arm": arm, "gpu_uuid": device["uuid"],
                           "output_dir": str(output / arm), "runtime_job": runtime["job"],
                           "runtime_result": runtime_wave["results"][arm]["required_artifact"],
                           "runtime_result_hash": runtime_wave["results"][arm]["required_artifact_sha256"],
                           "script_hash": file_hash(SCRIPT), "plan_hash": file_hash(PLAN)}
                    job_path = output / f"{arm}-job.json"
                    durable_json(job_path, job)
                    command = ("timeout", "--signal=TERM", "--kill-after=30s", f"{WORKER_CAP - 30:g}s", sys.executable,
                               str(SCRIPT), "--worker-job", str(job_path))
                    tasks.append(ParallelTuningTask(arm, 2, device["uuid"], output / arm, command))
                if not tasks:
                    raise CheckpointError("original arm GPUs are not eligible; profile deferred")
                durable_json(output / f"placement-{len(rows)}.json", selection)
                inflight = {task.task_id for task in tasks}
                wave = run_parallel_tuning_wave(tasks, timeout_seconds=WORKER_CAP, cwd=ROOT,
                    base_environment={**os.environ, "TF_NUM_INTRAOP_THREADS": str(max(1, len(os.sched_getaffinity(0)) // len(tasks))),
                                      "TF_NUM_INTEROP_THREADS": "1"},
                    artifact_validator=lambda task, result: validate_receipt(result),
                    monitor_callback=recovery.load_parallel()._headroom_monitor(tasks, output, len(rows)))
                durable_json(output / f"wave-{len(rows)}.json", wave)
                for row in wave["results"]:
                    arm = row["task"]["task_id"]
                    ledger.settle_arm(attempt_id=output.name, arm=arm, measured_seconds=row["elapsed_seconds"], status=row["status"])
                    settled.add(arm)
                    rows[arm] = row
                    pending.remove(arm)
                inflight = set()
                if wave["status"] != "completed":
                    raise CheckpointError("component profile worker incomplete")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            for arm in reserved - settled:
                ledger.settle_arm(attempt_id=output.name, arm=arm, measured_seconds=WORKER_CAP if arm in inflight else 0,
                                  status="uncertain_full_reservation" if arm in inflight else "not_launched")
            ledger.finish_attempt(attempt_id=output.name, status="failed" if error else "completed")
        summary = {"status": "failed" if error else "completed", "error": error, "results": rows,
                   "ledger": ledger.read(), "remaining_seconds": ledger.remaining_seconds()}
        durable_json(output / "summary.json", summary)
        print(json.dumps({"status": summary["status"], "output_root": str(output), "remaining_seconds": summary["remaining_seconds"]}))
        return 1 if error else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--worker-job", type=Path)
    args = parser.parse_args()
    if args.worker_job:
        return worker(args.worker_job)
    if args.campaign_root is None:
        parser.error("--campaign-root required")
    return coordinator(args.campaign_root)


if __name__ == "__main__":
    raise SystemExit(main())

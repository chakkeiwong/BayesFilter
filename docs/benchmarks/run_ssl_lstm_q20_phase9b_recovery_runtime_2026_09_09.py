#!/usr/bin/env python3
"""Parallel q=20 recovery, runtime and bounded P1; no posterior admission."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md"
RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md"
PARALLEL_SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py"
ARMS = ("factor", "strict")
ARM_CAP_SECONDS = 28_800.0
TOTAL_GPU_SECONDS = 86_400.0
CLAIM_BOUNDARY = "phase9b_p0_recovery_runtime_and_p1_mechanics_only"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.durable_tensor_checkpoint import (
    CheckpointError, DurableTensorCheckpoint, canonical_bytes, durable_bytes, durable_json, payload_hash,
)


SOURCE_MIGRATION_FILENAME = "source-migration.json"
SOURCE_MIGRATION_SCHEMA = "bayesfilter.phase9b.source_migration.v1"
SOURCE_MIGRATION_ID = "phase9b_recovery_tuning_checkpoint_nonfinite_v1"
NUMERICAL_MIGRATION_ID = "phase9b_principal_root_eigh_refinement_v1"
ACCOUNTING_MIGRATION_ID = "phase9b_campaign_budget_ledger_repair_v1"
NUMERICAL_MIGRATION_IDS = (NUMERICAL_MIGRATION_ID, ACCOUNTING_MIGRATION_ID)
NUMERICAL_CORE_PATH = "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py"
ACCOUNTING_REPAIR_PATH = "bayesfilter/runtime/campaign_budget_ledger.py"


def load_parallel():
    spec = importlib.util.spec_from_file_location("phase9b_recovery_parallel", PARALLEL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_hashes() -> dict[str, str]:
    result = load_parallel()._source_hashes()
    for path in (SCRIPT, ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py"):
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _encode_tuning_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"tuple_items": [_encode_tuning_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_tuning_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_tuning_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite__": value.__repr__()}
    return value


def typed_tuning_payload(result: Any) -> dict[str, Any]:
    return {"schema": "bayesfilter.phase9b.typed_tuning_payload.v2",
            "fields": _encode_tuning_value(asdict(result)),
            "payload_hash": result.artifact_hash}


def restore_tuning(payload: dict[str, Any]) -> Any:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCCandidateResult, FixedTransportHMCKernelTuningConfig,
        FixedTransportHMCKernelTuningResult,
    )

    if payload.get("schema") not in (None, "bayesfilter.phase9b.typed_tuning_payload.v2"):
        raise CheckpointError("unsupported typed tuning checkpoint schema")

    def decode(value):
        if isinstance(value, dict) and set(value) == {"__nonfinite__"}:
            marker = value["__nonfinite__"]
            if marker == "nan":
                return float("nan")
            if marker == "inf":
                return float("inf")
            if marker == "-inf":
                return float("-inf")
            raise CheckpointError(f"unknown nonfinite tuning marker: {marker!r}")
        if isinstance(value, dict) and set(value) == {"tuple_items"}:
            return tuple(decode(item) for item in value["tuple_items"])
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        return value

    values = decode(payload["fields"])
    config_fields = dict(values["config"])
    tuning_policy = config_fields.pop("tuning_policy", "measured_joint_grid_v1")
    values["config"] = FixedTransportHMCKernelTuningConfig(
        tuning_policy=tuning_policy, **config_fields
    )
    values["candidates"] = tuple(FixedTransportHMCCandidateResult(**row) for row in values["candidates"])
    result = FixedTransportHMCKernelTuningResult(**values)
    if result.artifact_hash != payload["payload_hash"]:
        raise CheckpointError("restored typed tuning result differs from the public artifact")
    return result


def _setup(tf: Any, job: dict[str, Any], source: Any, profile: Any):
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        restore_trainable_transport_checkpoint, transport_preflight_state_hash,
    )
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        build_verified_fixed_transport_hmc_handoff_from_tuning_result,
    )
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tuning_contract import hmc_tuning_interface_capability

    setup_root = Path(job["campaign_root"]) / "setup" / job["arm"]
    identity = {"sources": job.get("setup_sources", job["sources"]), "profile": profile.payload(), "target": source.EXPECTED_TARGET_SIGNATURE}
    bridge_started = time.monotonic()
    capability = hmc_tuning_interface_capability("tune_fixed_transport_hmc_kernel")
    if capability.capability_status != "tested_supported" or not capability.artifact_authority:
        raise CheckpointError("public fixed-transport tuner lacks artifact authority")
    bridge = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend=profile.principal_sqrt_backend,
    )
    if str(bridge.target_signature) != source.EXPECTED_TARGET_SIGNATURE:
        raise CheckpointError("target identity changed")
    bridge_seconds = time.monotonic() - bridge_started
    with DurableTensorCheckpoint(setup_root, identity) as store:
        def build_chart():
            attempt = setup_root / "chart-builds" / uuid.uuid4().hex
            _, checkpoints, preflights = source._build_fresh_chart(tf, bridge, 0, source.COMPONENT_IDS[0], profile, attempt)
            return {"checkpoint": checkpoints[-1]["checkpoint"], "preflights": preflights,
                    "artifact_root": str(attempt)}

        chart_data = store.run("chart", {}, build_chart)
        checkpoint = chart_data["checkpoint"]
        chart = restore_trainable_transport_checkpoint(checkpoint, expected_context={
            "component_id": source.COMPONENT_IDS[0], "beta": 1.0,
            "bridge_signature": str(bridge.signature), "target_signature": source.EXPECTED_TARGET_SIGNATURE,
            "checkpoint_scope": source._checkpoint_scope(source.COMPONENT_IDS[0], 1.0, 0, profile),
        })
        state_hash = transport_preflight_state_hash(chart)
        if state_hash != checkpoint["transport_state_hash"]:
            raise CheckpointError("restored chart tensor identity changed")
        chart.bind_frozen_identity({"checkpoint_sha256": checkpoint["checkpoint_hash"],
                                    "training_state_hash": state_hash, "transport_tensor_hash": state_hash})

        def tune():
            attempt = setup_root / "tuning-attempts" / uuid.uuid4().hex
            tuning_profile = profile
            if job.get("tuning_repair") == 1:
                tuning_profile = replace(profile,
                    step_size_candidates=tuple(0.055 / (2**index) for index in range(1, 5)),
                    initial_step_size=0.0275,
                    tuning_roots=tuple((first, second + 10000) for first, second in profile.tuning_roots))
            row = source._tune_scope(tf, bridge, chart, chart_index=0, beta=1.0,
                                     output_dir=attempt, profile=tuning_profile, scope_index=2)
            result = row["_live_tuning_result"]
            return {"typed": typed_tuning_payload(result), "artifact": row["tuning_artifact"],
                    "artifact_hash": hashlib.sha256(Path(row["tuning_artifact"]).read_bytes()).hexdigest(),
                    "handoff_hash": row["handoff_hash"]}

        tuning_key = "tuning-repair-r1" if job.get("tuning_repair") == 1 else "tuning"
        tuned = store.run(tuning_key, {"checkpoint_hash": checkpoint["checkpoint_hash"]}, tune)
        if hashlib.sha256(Path(tuned["artifact"]).read_bytes()).hexdigest() != tuned["artifact_hash"]:
            raise CheckpointError("public tuning artifact checksum changed")
        tuning_result = restore_tuning(tuned["typed"])
        handoff = build_verified_fixed_transport_hmc_handoff_from_tuning_result(
            tuning_result=tuning_result, base_adapter=bridge.fixed_beta_adapter(1.0), fixed_transport=chart,
        )
        if handoff.handoff_hash != tuned["handoff_hash"]:
            raise CheckpointError("verified handoff changed on restore")
        setup = {"stages": list(store.records), "bridge_seconds": bridge_seconds,
                 "handoff": handoff.payload(), "profile": profile.payload(),
                 "tuning_config": tuning_result.config.payload(), "chart_source_identity": identity,
                 "tuning_artifact": tuned["artifact"], "chart_artifact_root": chart_data["artifact_root"]}
    return chart, handoff, setup


def full_health(tf: Any, samples: Any, trace: Any, initial_state: Any, config: Any) -> dict[str, Any]:
    from bayesfilter.inference.neutra_hmc import _summarize_batched_hmc_output, _chunk_policy_vetoes, _chain_moved

    required = {"is_accepted", "target_log_prob", "log_accept_ratio", "target_status"}
    if not required <= trace.keys() or not isinstance(trace["target_status"], dict):
        raise CheckpointError("missing full-controller target status or acceptance trace")
    shape = tuple(samples.shape[:2])
    for name in ("is_accepted", "target_log_prob", "log_accept_ratio"):
        if tuple(trace[name].shape) != shape:
            raise CheckpointError(f"incomplete {name} trace shape")
    status = trace["target_status"]
    for name in ("status_code", "valid_pre_regularized_score"):
        if name not in status or tuple(status[name].shape) != shape:
            raise CheckpointError(f"missing complete {name} telemetry")
    summary = _summarize_batched_hmc_output(initial_state=initial_state, samples=samples, trace=trace,
                                           config=config, chain_count=4, elapsed_seconds=0.0)["diagnostics"]
    delta_h = -trace["log_accept_ratio"]
    finite = lambda tensor: bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())
    status_passed = summary["target_status_telemetry"]["all_status_valid"]
    floors = status.get("floor_count_value")
    eigen = status.get("min_innovation_eigenvalue")
    if floors is not None:
        status_passed = status_passed and bool(tf.reduce_all(tf.equal(floors, 0)).numpy())
    if eigen is not None:
        status_passed = status_passed and finite(eigen) and bool(tf.reduce_all(eigen > 0).numpy())
    vetoes = _chunk_policy_vetoes(
        samples_finite=finite(samples), log_accept_finite=finite(trace["log_accept_ratio"]),
        target_finite=finite(trace["target_log_prob"]), proposed_finite=True,
        target_score_finite=bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()),
        delta_h_finite=finite(delta_h), target_status_passed=status_passed,
        chain_moved=_chain_moved(initial_state, samples),
        native_divergence_status="unavailable", native_divergence_count=None,
    )
    return {"passed": not vetoes, "vetoes": list(vetoes), "shared_health": summary,
            "target_status_passed": status_passed, "all_chain_movement": _chain_moved(initial_state, samples).numpy().tolist(),
            "sampled_state_count": int(tf.size(trace["target_log_prob"]).numpy()),
            "proposed_state_score_check": "not_exposed_by_shared_trace_no_extra_claim",
            "energy": {"identity": "delta_h_equals_negative_log_accept_ratio", "all_finite": finite(delta_h),
                       "maximum_absolute": float(tf.reduce_max(tf.abs(delta_h)).numpy()) if finite(delta_h) else None,
                       "finite_extremes_role": "explanatory_only", "native_divergences": None}}


def diagnostic_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: diagnostic_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_payload(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def stream_seeds(campaign: Path, arm: str, family: str) -> tuple[tuple[int, int], tuple[int, int]]:
    digest = hashlib.sha256(f"{campaign}:{arm}:{family}".encode()).digest()
    first = int.from_bytes(digest[:4], "big") % (2**31 - 1)
    return (first, 20260909), (first, 20261909)


def _run_controller(tf: Any, job: dict[str, Any], chart: Any, handoff: Any, profile: Any) -> dict[str, Any]:
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig, SequentialNeuTraHMCConfig, run_sequential_neutra_hmc,
    )

    campaign = Path(job["campaign_root"])
    mode = job["mode"]
    family = mode if mode in ("runtime", "p1") else "canary"
    branch = job.get("stream_branch", mode)
    chunk_results = 500 if family in ("runtime", "p1") else 8
    root = campaign / "streams" / job["arm"] / branch
    warmup_seed, retained_seed = stream_seeds(campaign, job["arm"], family)
    config = SequentialNeuTraHMCConfig(
        step_size=float(handoff.step_size), num_leapfrog_steps=int(handoff.num_leapfrog_steps),
        warmup_seed=warmup_seed, retained_seed=retained_seed, warmup_chunk_results=chunk_results,
        warmup_min_results=2000, warmup_check_window_results=1000, warmup_max_results=2000,
        warmup_rhat_max=1.05, retained_chunk_results=500, retained_min_results=1000,
        retained_max_results=1000, retained_rhat_max=1.01, minimum_chain_count=4, jit_compile=True,
    )
    initial_state = tf.constant(profile.initial_state_bank, tf.float64)
    identity = {"sources": job.get("stream_sources", job["sources"]), "handoff_hash": handoff.handoff_hash,
                "config": config.payload(chain_count=4), "gpu_uuid": job["gpu_uuid"]}
    calls = 0
    health_records = []
    previous_state = initial_state
    controller_started = time.monotonic()

    def budget_check(_gradients):
        nonlocal calls
        calls += 1
        if mode != "p1":
            return calls <= 2
        key = f"chunk-{calls - 1:06d}"
        if (root / "chunks" / "committed" / key).is_dir():
            return True
        deadline = job.get("worker_deadline_monotonic", controller_started + job["cap_seconds"] - 30.0)
        return time.monotonic() + job["next_chunk_reserve_seconds"] <= deadline

    def event(name, payload):
        if name == "call-start":
            durable_json(Path(job["output_dir"]) / f"{payload['key']}-start.json", payload)
        if name == "call-end":
            durable_json(Path(job["output_dir"]) / f"{payload['key']}-end.json", payload)

    with DurableTensorCheckpoint(root / "chunks", identity, event_callback=event) as store, \
            DurableTensorCheckpoint(root / "archives", identity) as archives:
        def archive(**kwargs):
            nonlocal previous_state
            suffix = f"cumulative-{int(kwargs['latent_samples'].shape[0])}" if kwargs["cumulative"] else kwargs["chunk_index"]
            key = f"{kwargs['stage']}-{suffix}"
            inputs = {"stage": kwargs["stage"], "chunk_index": kwargs["chunk_index"],
                      "seed": kwargs["seed"], "cumulative": kwargs["cumulative"],
                      "latent_hash": store.tensor_hash(kwargs["latent_samples"]),
                      "model_hash": store.tensor_hash(kwargs["model_samples"])}
            archives.run(key, inputs, lambda: {"latent": kwargs["latent_samples"], "model": kwargs["model_samples"]})
            if not kwargs["cumulative"]:
                index = len(health_records)
                seed = kwargs["seed"]
                previous = previous_state
                active_results = int(kwargs["latent_samples"].shape[0])
                chunk_inputs = {"adapter_signature": str(handoff.transformed_adapter.adapter_signature()),
                                "config": config.payload(chain_count=4), "active_results": active_results,
                                "seed": seed, "initial_state_sha256": store.tensor_hash(previous)}
                (samples, trace), _ = store.load(f"chunk-{index:06d}", chunk_inputs)
                started = time.monotonic()
                health = full_health(tf, samples, trace, previous, BatchedHMCConfig(
                    num_results=active_results, num_burnin_steps=0, step_size=config.step_size,
                    num_leapfrog_steps=config.num_leapfrog_steps, seed=seed, jit_compile=True))
                health["health_seconds"] = time.monotonic() - started
                health.update(stage=kwargs["stage"], stage_chunk_index=kwargs["chunk_index"],
                              checkpoint_index=index, seed=seed)
                health_records.append(health)
                previous_state = samples[-1]
                durable_json(Path(job["output_dir"]) / f"chunk-{index:06d}-health.json", health)
                if not health["passed"]:
                    raise CheckpointError(f"controller health veto: {health['vetoes']}")
            return {"checkpoint_root": str(archives.root), "key": key, **inputs}

        started = time.monotonic()
        result = run_sequential_neutra_hmc(
            adapter=handoff.transformed_adapter, initial_state=initial_state,
            model_transform=lambda samples: tf.reshape(chart.forward_batch(tf.reshape(samples, [-1, 4])), tf.shape(samples)),
            parameter_names=tuple(f"theta.{index}" for index in range(4)), config=config,
            archive_callback=archive, budget_check=budget_check, checkpoint_store=store,
        )
        elapsed = time.monotonic() - started
        if len(store.records) != len(health_records) or any(not row["passed"] for row in health_records):
            raise CheckpointError("complete healthy controller calls are required")
        if mode != "p1" and len(store.records) != 2:
            raise CheckpointError("two complete diagnostic calls are required")
        if mode != "p1" and tuple(result["hard_vetoes"]) != ("campaign_resource_cap",):
            raise CheckpointError(f"unexpected controller stop: {result['hard_vetoes']}")
        program_evidence = result["checkpoint_program_evidence"]
        if any(not row["replayed"] for row in store.records):
            if not program_evidence or any(row["tracing_count"] != 1 or not row["hlo_sha256"] for row in program_evidence):
                raise CheckpointError("controller lacks single-trace XLA evidence")
        output = {"chunk_results": chunk_results, "chunks": list(store.records),
                  "archives": list(archives.records), "health": health_records,
                  "controller_elapsed_seconds": elapsed, "config": config.payload(chain_count=4),
                  "stopped_before_third_chunk": mode != "p1", "posterior_admitted": False,
                  "stream_root": str(root), "source_identity": identity,
                  "program_evidence": program_evidence,
                  "sample_sha256": store.tensor_hash(result["private_warmup_z"]),
                  "retained_sample_sha256": store.tensor_hash(result["private_retained_z"]),
                  "sequential_result": diagnostic_payload({key: value for key, value in result.items()
                                                            if not key.startswith("private_")})}
    return output


def worker(job_path: Path) -> int:
    job = json.loads(job_path.read_bytes())
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    current_sources = source_hashes()
    if current_sources != job["sources"]:
        raise CheckpointError("source closure changed before worker launch")
    migration = job.get("source_migration")
    if migration is not None and (migration.get("to_sources") != current_sources
                                  or migration.get("from_sources") != job.get("binding_sources")):
        raise CheckpointError("worker source migration binding changed")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise CheckpointError("memory growth must be enabled before import")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu_uuid"]:
        raise CheckpointError("worker GPU UUID mismatch")
    started = time.monotonic()
    job["worker_deadline_monotonic"] = started + job["cap_seconds"] - 30.0
    durable_json(output / "run_start.json", {"job": job, "pid": os.getpid(), "at_unix": time.time(),
                 "command": sys.argv, "environment": {name: os.environ.get(name) for name in (
                     "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS",
                     "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "XLA_FLAGS", "TF_XLA_FLAGS")}})
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    durable_json(output / "memory_policy.json", memory)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise CheckpointError("one visible GPU required")
    parallel = load_parallel()
    source = parallel._load_source()
    profile = replace(parallel._profile(source, Path(job["campaign_root"]), job["arm"], ARM_CAP_SECONDS),
                      plan_path=PLAN)
    with tf.device(logical[0].name):
        chart, handoff, setup = _setup(tf, job, source, profile)
        durable_json(output / "setup-receipt.json", setup)
        controller = _run_controller(tf, job, chart, handoff, profile)
        allocator = tf.config.experimental.get_memory_info(logical[0].name)
    if source_hashes() != job["sources"]:
        raise CheckpointError("source closure changed during execution")
    result = {"status": "completed", "job": job, "memory_policy": memory, "setup": setup,
              "controller": controller, "allocator": allocator, "gpu_uuid": job["gpu_uuid"],
              "tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__,
              "jit_compile": True, "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
              "elapsed_seconds": time.monotonic() - started, "claim_boundary": CLAIM_BOUNDARY}
    durable_json(output / "run_manifest.json", result)
    return 0


def compare_canary(reference: dict[str, Any], resumed: dict[str, Any]) -> dict[str, Any]:
    original = reference["controller"]
    restored = resumed["controller"]
    if original["source_identity"] != restored["source_identity"]:
        raise CheckpointError("canary comparator changed identity")
    equality = []
    for index in range(2):
        paths = [Path(row["stream_root"]) / "chunks" / "committed" / f"chunk-{index:06d}" / "bundle.json"
                 for row in (original, restored)]
        bundles = []
        for path in paths:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != path.with_name("bundle.sha256").read_text():
                raise CheckpointError("canary bundle checksum changed")
            bundle = json.loads(content)

            def verify_tree(tree):
                if "tensor" in tree:
                    receipt = tree["tensor"]
                    tensor_path = path.parent / receipt["file"]
                    if tensor_path.parent != path.parent or hashlib.sha256(tensor_path.read_bytes()).hexdigest() != receipt["sha256"]:
                        raise CheckpointError("canary tensor checksum changed")
                elif "mapping" in tree:
                    for item in tree["mapping"].values():
                        verify_tree(item)
                elif "sequence" in tree:
                    for item in tree["sequence"]:
                        verify_tree(item)

            verify_tree(bundle["tree"])
            bundles.append(bundle)
        equality.append(bundles[0]["tree"] == bundles[1]["tree"] and bundles[0]["inputs_hash"] == bundles[1]["inputs_hash"])
    if not all(equality) or original["sample_sha256"] != restored["sample_sha256"]:
        raise CheckpointError("resumed samples or full trace differ from uninterrupted reference")
    if restored["chunks"][0]["replayed"] is not True or restored["chunks"][1]["replayed"] is not False:
        raise CheckpointError("canary did not reuse the committed first chunk and recompute the second")
    return {"passed": True, "exact_tensor_and_trace_equality": equality,
            "reused_first_chunk": True, "recomputed_interrupted_second_chunk": True,
            "gpu_uuid": reference["gpu_uuid"], "scope": "eight_transition_q20_recovery_mechanics_only"}


def forecast(result: dict[str, Any]) -> dict[str, Any]:
    controller = result["controller"]
    if controller["chunk_results"] != 500 or len(controller["chunks"]) != 2:
        raise CheckpointError("full 500-transition runtime measurements required")
    first, steady = [float(row["compute_seconds"]) for row in controller["chunks"]]
    setup = sum(float(row["compute_seconds"]) + float(row["serialization_seconds"]) for row in result["setup"]["stages"])
    setup += result["setup"]["bridge_seconds"]
    overhead = max(0.0, float(controller["controller_elapsed_seconds"]) - first - steady)
    per_chunk_overhead = overhead / 2.0
    startup = max(0.0, result["elapsed_seconds"] - controller["controller_elapsed_seconds"])
    estimate = setup + startup + first + 5.0 * steady + 6.0 * per_chunk_overhead
    reserved = estimate + steady + per_chunk_overhead + 30.0
    if not all(math.isfinite(value) and value >= 0 for value in (first, steady, reserved)):
        raise CheckpointError("nonfinite runtime forecast")
    return {"first_chunk_seconds": first, "steady_chunk_seconds": steady,
            "setup_seconds": setup, "restart_startup_seconds": startup,
            "observed_overhead_per_chunk_seconds": per_chunk_overhead,
            "six_chunk_estimate_seconds": estimate, "with_one_steady_chunk_reserve_and_grace_seconds": reserved,
            "arm_cap_seconds": ARM_CAP_SECONDS, "fits_arm_cap": reserved <= ARM_CAP_SECONDS,
            "uncertainty": "descriptive single repeated call, not a statistical bound"}


def _validate_worker(task: Any, result: dict[str, Any]) -> None:
    if result.get("status") != "completed" or result.get("gpu_uuid") != task.gpu_uuid:
        raise CheckpointError("incomplete or mismatched worker result")
    memory = result.get("memory_policy", {})
    if memory.get("all_physical_devices_memory_growth") is not True or memory.get("configured_before_logical_device_initialization") is not True:
        raise CheckpointError("invalid startup memory receipt")
    controller = result["controller"]
    health = controller["health"]
    if len(controller["chunks"]) != len(health) or not all(row["passed"] for row in health):
        raise CheckpointError("incomplete health receipts")
    if result["job"]["mode"] == "p1":
        p1_outcome(controller)
    elif len(health) != 2:
        raise CheckpointError("two healthy diagnostic chunks required")


def p1_outcome(controller: dict[str, Any]) -> str:
    sequential = controller["sequential_result"]
    count = sequential["warmup_results_per_chain"] + sequential["retained_results_per_chain"]
    if count != sum(row["sampled_state_count"] // 4 for row in controller["health"]):
        raise CheckpointError("P1 sample count differs from archived health receipts")
    if sequential["hard_vetoes"]:
        if set(sequential["hard_vetoes"]) == {"campaign_resource_cap"}:
            return "resource_cap_partial_not_completed"
        return "candidate_health_veto"
    if sequential["passed"]:
        return "bounded_sequential_screen_passed_not_posterior_admission"
    if sequential["warmup_cap_hit"] and not sequential["warmup_passed"]:
        if sequential["retained_results_per_chain"] != 0:
            raise CheckpointError("retained sampling after failed warmup")
        return "candidate_warmup_screen_failed"
    if sequential["retained_cap_hit"]:
        return "candidate_retained_screen_failed"
    raise CheckpointError("P1 stopped without a terminal decision or resource receipt")


def p1_allocations(canaries: dict[str, Any], forecasts: dict[str, Any], remaining: float) -> dict[str, Any]:
    if set(canaries) != set(ARMS) or set(forecasts) != set(ARMS):
        raise CheckpointError("both arms are required for P1 readiness")
    allocations = {}
    for arm in ARMS:
        if canaries[arm].get("passed") is not True or canaries[arm].get("exact_tensor_and_trace_equality") != [True, True]:
            raise CheckpointError("P1 requires exact interruption recovery")
        row = forecasts[arm]
        required = float(row["with_one_steady_chunk_reserve_and_grace_seconds"])
        if not math.isfinite(required) or not 0 < required <= ARM_CAP_SECONDS or row["fits_arm_cap"] is not True:
            raise CheckpointError("P1 forecast exceeds per-arm cap or is invalid")
        reserve = 3600.0 * math.ceil(required / 3600.0)
        next_chunk = max(row["first_chunk_seconds"], row["steady_chunk_seconds"]) + row["observed_overhead_per_chunk_seconds"] + 30.0
        if not math.isfinite(next_chunk) or not 0 < next_chunk < reserve:
            raise CheckpointError("invalid next-chunk forecast")
        allocations[arm] = {"cap_seconds": reserve, "next_chunk_reserve_seconds": next_chunk}
    if not math.isfinite(remaining) or sum(row["cap_seconds"] for row in allocations.values()) > remaining:
        raise CheckpointError("remaining campaign budget cannot reserve both P1 arms")
    return allocations


def _source_migration_path(campaign: Path) -> Path:
    return campaign / SOURCE_MIGRATION_FILENAME


def _validate_source_migration(start: dict[str, Any], campaign: Path, current: dict[str, str]) -> dict[str, Any]:
    path = _source_migration_path(campaign)
    if not path.is_file():
        raise CheckpointError("source closure changed after campaign initialization without a recorded migration")
    try:
        migration = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"invalid source migration record: {exc}") from exc
    migration_id = migration.get("migration_id")
    if (migration.get("schema") != SOURCE_MIGRATION_SCHEMA
            or migration_id not in (SOURCE_MIGRATION_ID, *NUMERICAL_MIGRATION_IDS)):
        raise CheckpointError("unsupported Phase 9B source migration")
    if migration.get("from_sources") != start["sources"] or migration.get("to_sources") != current:
        raise CheckpointError("source migration does not bind the current source closure")
    changed = sorted(path for path in set(start["sources"]) | set(current)
                     if start["sources"].get(path) != current.get(path))
    numerical = migration_id in NUMERICAL_MIGRATION_IDS
    if migration_id == ACCOUNTING_MIGRATION_ID:
        allowed = sorted([str(SCRIPT.relative_to(ROOT)), NUMERICAL_CORE_PATH, ACCOUNTING_REPAIR_PATH])
    else:
        allowed = sorted([str(SCRIPT.relative_to(ROOT)), NUMERICAL_CORE_PATH] if numerical
                         else [str(SCRIPT.relative_to(ROOT))])
    if migration.get("changed_paths") != changed or changed != allowed:
        raise CheckpointError("source migration is broader than the declared repair")
    if migration.get("plan_hash") != start["plan_hash"] or migration.get("claim_boundary") != CLAIM_BOUNDARY:
        raise CheckpointError("source migration plan or claim boundary mismatch")
    if (migration.get("scientific_contract_unchanged") is not True
            or migration.get("serializer_only") is not (not numerical)):
        raise CheckpointError("source migration repair kind or scientific contract mismatch")
    if migration_id == ACCOUNTING_MIGRATION_ID:
        reusable_setup_sources = migration.get("reusable_setup_sources")
        if not isinstance(reusable_setup_sources, dict):
            raise CheckpointError("accounting migration lacks reusable setup source identity")
        setup_changed = sorted(path for path in set(reusable_setup_sources) | set(current)
                               if reusable_setup_sources.get(path) != current.get(path))
        expected_setup_changes = sorted([str(SCRIPT.relative_to(ROOT)), ACCOUNTING_REPAIR_PATH])
        if setup_changed != expected_setup_changes:
            raise CheckpointError("accounting migration setup identity changed beyond harness files")
    if numerical:
        namespace = Path(migration.get("execution_namespace", ""))
        if (len(namespace.parts) != 2 or namespace.parts[0] != "numerical-repairs"
                or namespace.parts[1] in (".", "..") or namespace.is_absolute()
                or (campaign / namespace).resolve().parent != (campaign / "numerical-repairs").resolve()):
            raise CheckpointError("numerical migration requires a fresh local execution namespace")
    return migration


def execution_source_binding(start: dict[str, Any], campaign: Path) -> tuple[dict[str, str], dict[str, Any] | None]:
    current = source_hashes()
    if current == start["sources"]:
        return current, None
    migration_root = Path(start.get("source_migration_root", campaign))
    return current, _validate_source_migration(start, migration_root, current)


def verify_binding(start: dict[str, Any], campaign: Path | None = None) -> None:
    current = source_hashes()
    if current != start["sources"]:
        if campaign is None:
            raise CheckpointError("source closure changed after campaign initialization")
        _validate_source_migration(start, Path(start.get("source_migration_root", campaign)), current)
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != start["plan_hash"]:
        raise CheckpointError("source/plan changed after campaign initialization")


def prepare_execution_namespace(campaign: Path, start: dict[str, Any], sources: dict[str, str],
                                migration: dict[str, Any] | None) -> tuple[Path, dict[str, Any]]:
    if migration is None or migration["migration_id"] not in NUMERICAL_MIGRATION_IDS:
        return campaign, start
    execution = campaign / migration["execution_namespace"]
    setup_sources = migration.get("reusable_setup_sources", sources)
    record = {"schema": "bayesfilter.phase9b.numerical_execution.v1", "budget_campaign_root": str(campaign),
              "execution_root": str(execution), "binding_sources": start["sources"],
              "execution_sources": sources, "setup_sources": setup_sources,
              "plan_hash": start["plan_hash"],
              "source_migration": migration, "claim_boundary": CLAIM_BOUNDARY}
    record_path = execution / "execution-start.json"
    if record_path.exists():
        if json.loads(record_path.read_bytes()) != record:
            raise CheckpointError("numerical execution namespace binding changed")
    else:
        if execution.exists() and any(not path.is_file() or not path.name.startswith("execution-start.json.partial-")
                                      for path in execution.iterdir()):
            raise CheckpointError("numerical execution namespace contains unbound prior evidence")
        execution.mkdir(parents=True, exist_ok=True)
        durable_json(record_path, record)
    return execution, {**start, "source_migration_root": str(campaign),
                       "setup_sources": setup_sources, "stream_sources": sources}


def _wave(campaign: Path, mode: str, cap: float | dict[str, float], ledger: Any, start: dict[str, Any],
          *, allocations: dict[str, Any] | None = None) -> dict[str, Any]:
    from bayesfilter.runtime.display_gpu_policy import probe_inventory, select_gpus
    from bayesfilter.runtime.parallel_tuning import ParallelTuningTask, run_parallel_tuning_wave

    verify_binding(start, campaign)
    execution_sources, source_migration = execution_source_binding(start, campaign)
    caps = {arm: float(cap[arm] if isinstance(cap, dict) else cap) for arm in ARMS}
    if any(not math.isfinite(value) or not 30.0 < value <= ARM_CAP_SECONDS for value in caps.values()):
        raise CheckpointError("worker caps must be finite and within the eight-hour ceiling")
    if mode == "p1" and (allocations is None or set(allocations) != set(ARMS)):
        raise CheckpointError("P1 requires measured per-arm allocations")
    parallel = load_parallel()
    output = campaign / "launches" / f"{mode}-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    attempt_id = output.name
    binding = {"source_hash": payload_hash(start["sources"]), "plan_hash": start["plan_hash"]}
    ledger.start_attempt(attempt_id=attempt_id, output_root=output, seed_namespace={"mode": mode}, **binding)
    results = {}
    if mode != "interrupt":
        for prior in sorted((campaign / "launches").glob(f"{mode}-*/summary.json"), key=lambda path: path.stat().st_mtime):
            summary = json.loads(prior.read_bytes())
            for arm, row in summary.get("results", {}).items():
                if row.get("status") != "completed":
                    continue
                path = Path(row["required_artifact"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["required_artifact_sha256"]:
                    raise CheckpointError("previous successful sibling result is corrupt")
                result = json.loads(path.read_bytes())
                task = ParallelTuningTask(arm, 2, row["task"]["gpu_uuid"], Path(row["task"]["output_dir"]), tuple(row["task"]["command"]))
                _validate_worker(task, result)
                if result["job"]["mode"] != mode or Path(result["job"]["campaign_root"]) != campaign:
                    raise CheckpointError("previous sibling belongs to a different wave")
                if (source_migration is not None
                        and source_migration.get("migration_id") in NUMERICAL_MIGRATION_IDS):
                    reusable_setup_sources = source_migration.get("reusable_setup_sources", execution_sources)
                    job_sources = result["job"]
                    if (job_sources.get("setup_sources") != reusable_setup_sources
                            or job_sources.get("sources") not in (execution_sources, reusable_setup_sources)
                            or job_sources.get("stream_sources") not in (execution_sources, reusable_setup_sources)):
                        raise CheckpointError("previous sibling belongs to a different numerical scope")
                if mode == "p1" and p1_outcome(result["controller"]) == "resource_cap_partial_not_completed":
                    continue
                results[arm] = row
    reserved = []
    settled = set()
    inflight = []
    killed = {}
    branches = {}
    error = None
    try:
        for arm in ARMS:
            if arm in results:
                continue
            ledger.reserve_chunk(attempt_id=attempt_id, arm=arm, chunk_index=0, reserve_seconds=caps[arm],
                                 output_root=output, seed_namespace={"mode": mode, "arm": arm}, **binding)
            reserved.append(arm)
        pending = [arm for arm in ARMS if arm not in results]
        while pending:
            selection = select_gpus(probe_inventory(), len(ARMS), estimated_peak_mib=4096)
            if not selection["selected"]:
                raise CheckpointError("no eligible GPU; retry later without spending a worker reservation")
            assigned = []
            unused = list(selection["selected"])
            for arm in pending:
                device_file = campaign / f"{arm}-gpu.json"
                expected = json.loads(device_file.read_bytes())["uuid"] if device_file.exists() else None
                selected = next((row for row in unused if expected is None or row["uuid"] == expected), None)
                if selected is not None:
                    unused.remove(selected)
                    assigned.append((arm, selected))
                    if expected is None:
                        durable_json(device_file, selected)
            if not assigned:
                raise CheckpointError("canary's assigned GPU unavailable; cannot silently change comparator device")
            tasks = []
            for arm, device in assigned:
                arm_cap = caps[arm]
                branch_file = campaign / f"{arm}-interruption-stream.json"
                if mode == "interrupt":
                    branches[arm] = f"interrupted-{attempt_id}"
                    durable_json(branch_file, {"branch": branches[arm]})
                elif mode == "resume":
                    branches[arm] = json.loads(branch_file.read_bytes())["branch"]
                else:
                    branches[arm] = mode
                job = {"arm": arm, "mode": mode, "campaign_root": str(campaign), "output_dir": str(output / arm),
                       "sources": execution_sources, "binding_sources": start["sources"],
                       "gpu_uuid": device["uuid"], "cap_seconds": arm_cap,
                       "setup_sources": start.get("setup_sources", start["sources"]),
                       "stream_sources": start.get("stream_sources", start["sources"]),
                       "tuning_repair": start.get("tuning_repair", 1),
                       "stream_branch": branches[arm]}
                if source_migration is not None:
                    job["source_migration"] = source_migration
                if mode == "p1":
                    job["next_chunk_reserve_seconds"] = allocations[arm]["next_chunk_reserve_seconds"]
                job_path = output / f"{arm}-job.json"
                durable_json(job_path, job)
                command = ("timeout", "--signal=TERM", "--kill-after=30s", f"{arm_cap - 30:g}s",
                           sys.executable, str(SCRIPT), "--worker-job", str(job_path))
                tasks.append(ParallelTuningTask(arm, 2, device["uuid"], Path(job["output_dir"]), command))
            durable_json(output / f"placement-{len(results)}.json", selection)
            headroom = parallel._headroom_monitor(tasks, output, len(results))

            def monitor():
                headroom()
                if mode != "interrupt":
                    return
                for task in tasks:
                    marker = task.output_dir / "chunk-000001-start.json"
                    if task.task_id in killed or not marker.exists():
                        continue
                    record = json.loads(marker.read_bytes())
                    if time.time() - record["at_unix"] < 1.0:
                        continue
                    if (task.output_dir / "chunk-000001-end.json").exists():
                        raise CheckpointError("canary missed mid-call interruption window")
                    supervisor = json.loads((task.supervision_dir / "parallel_worker_start.json").read_bytes())
                    first_commit = campaign / "streams" / task.task_id / branches[task.task_id] / "chunks" / "committed" / "chunk-000000"
                    if not first_commit.is_dir():
                        raise CheckpointError("no durable first chunk before canary interruption")
                    os.killpg(supervisor["pid"], signal.SIGKILL)
                    killed[task.task_id] = {"pid": supervisor["pid"], "worker_pid": record["pid"],
                                             "signal": "SIGKILL", "at_unix": time.time(), "call_start": record,
                                             "first_chunk_committed": True, "second_call_end_absent": True}
                    durable_json(output / f"{task.task_id}-interruption.json", killed[task.task_id])

            inflight = [task.task_id for task in tasks]
            wave = run_parallel_tuning_wave(tasks, timeout_seconds=max(caps[task.task_id] for task in tasks), terminate_grace_seconds=30.0,
                     cwd=ROOT, base_environment={**os.environ, "TF_NUM_INTRAOP_THREADS": str(max(1, len(os.sched_getaffinity(0)) // len(tasks))),
                                                "TF_NUM_INTEROP_THREADS": "1"},
                     artifact_validator=_validate_worker, monitor_callback=monitor)
            durable_json(output / f"wave-{len(results)}.json", wave)
            for row in wave["results"]:
                arm = row["task"]["task_id"]
                ledger.settle_arm(attempt_id=attempt_id, arm=arm, measured_seconds=row["elapsed_seconds"], status=row["status"])
                settled.add(arm)
                results[arm] = row
            inflight = []
            for arm, _ in assigned:
                row = results[arm]
                if mode == "interrupt":
                    if arm not in killed or row["returncode"] != -signal.SIGKILL:
                        raise CheckpointError("expected hard-interruption canary was not observed")
                elif row["status"] != "completed":
                    raise CheckpointError(f"{arm} {mode} worker failed: {row}")
                pending.remove(arm)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for arm in reserved:
            if arm not in settled:
                ledger.settle_arm(attempt_id=attempt_id, arm=arm, measured_seconds=caps[arm] if arm in inflight else 0,
                                  status="conservative_unobserved_worker_bound" if arm in inflight else "not_launched")
        ledger.finish_attempt(attempt_id=attempt_id, status="failed" if error else "completed")
    summary = {"mode": mode, "passed": error is None, "error": error, "results": results,
               "interruptions": killed, "output_root": str(output), "ledger": ledger.read()}
    durable_json(output / "summary.json", summary)
    if error:
        raise CheckpointError(error)
    partial = mode == "p1" and any(p1_outcome(row["controller"]) == "resource_cap_partial_not_completed"
                                    for row in _read_results(summary).values())
    if not partial:
        durable_json(campaign / f"{mode}-complete.json", summary)
    return summary


def _read_results(wave: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for arm, row in wave["results"].items():
        content = Path(row["required_artifact"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["required_artifact_sha256"]:
            raise CheckpointError("completed worker manifest checksum changed")
        results[arm] = json.loads(content)
    return results


def reconcile_interrupted_supervisor(campaign: Path, ledger: Any) -> None:
    payload = ledger.read()
    for attempt in payload["attempts"]:
        if attempt["status"] != "running":
            continue
        output = Path(attempt["output_root"])
        if campaign not in output.parents:
            raise CheckpointError("unrelated reservation cannot be reconciled")
        active = [row for row in payload["reservations"].values()
                  if row["attempt_id"] == attempt["attempt_id"] and row["status"] == "reserved"]
        for arm in sorted({row["arm"] for row in active}):
            reserved = sum(row["reserved_seconds"] for row in active if row["arm"] == arm)
            supervision = output / f"{arm}-supervision"
            receipt_path = supervision / "parallel_worker_result.json"
            start_path = supervision / "parallel_worker_start.json"
            if start_path.exists():
                started = json.loads(start_path.read_bytes())
                process_id = int(started["pid"])
                command_path = Path(f"/proc/{process_id}/cmdline")
                expected_job = str(output / f"{arm}-job.json").encode()
                if command_path.exists() and expected_job in command_path.read_bytes().split(b"\0"):
                    os.killpg(process_id, signal.SIGTERM)
                    deadline = time.monotonic() + 5.0
                    while command_path.exists() and time.monotonic() < deadline:
                        time.sleep(0.1)
                    if command_path.exists():
                        try:
                            os.killpg(process_id, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_bytes())
                measured = float(receipt["elapsed_seconds"])
                if not math.isfinite(measured) or not 0 <= measured <= reserved + 1:
                    raise CheckpointError("invalid prior worker accounting receipt")
                status = "recovered_observed_worker_receipt"
            elif (output / f"{arm}-job.json").exists():
                measured, status = reserved, "parent_lost_conservative_full_reservation"
            else:
                measured, status = 0.0, "parent_lost_before_job_preparation"
            ledger.settle_arm(attempt_id=attempt["attempt_id"], arm=arm, measured_seconds=measured, status=status)
        ledger.finish_attempt(attempt_id=attempt["attempt_id"], status="interrupted_supervisor_reconciled")


def initialize_campaign(campaign: Path, *, resume: bool) -> tuple[dict[str, Any], Any]:
    from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger

    if (campaign / "execution-start.json").exists():
        raise CheckpointError("resume numerical execution through its original budget campaign root")
    ledger = CampaignBudgetLedger(campaign / "campaign_budget_ledger.json")
    start_path = campaign / "campaign-start.json"
    if start_path.exists():
        if not resume:
            raise CheckpointError("initialized campaign requires --resume")
        start = json.loads(start_path.read_bytes())
        verify_binding(start, campaign)
        payload = ledger.read()
        if (payload["campaign_id"] != campaign.name or payload["total_budget_seconds"] != TOTAL_GPU_SECONDS
                or payload["source_hash"] != payload_hash(start["sources"])
                or payload["plan_hash"] != start["plan_hash"] or payload["claim_boundary"] != CLAIM_BOUNDARY):
            raise CheckpointError("campaign ledger binding or allocation changed")
        return start, ledger
    sources = source_hashes()
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    initialization = campaign / "campaign-initialization.json"
    if initialization.exists():
        record = json.loads(initialization.read_bytes())
        start = record["start"]
        verify_binding(start, campaign)
    else:
        start = {"sources": sources, "plan_hash": plan_hash, "plan_file": str(PLAN), "result_file": str(RESULT),
                 "command": sys.argv, "python": sys.executable, "started_at_unix": time.time(),
                 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
                 "git_status": subprocess.check_output(["git", "status", "--short"], text=True, cwd=ROOT),
                 "total_gpu_seconds": TOTAL_GPU_SECONDS, "arm_cap_seconds": ARM_CAP_SECONDS,
                 "claim_boundary": CLAIM_BOUNDARY, "prior_ledgers_not_modified": True, "tuning_repair": 1}
    if not ledger.path.exists():
        ledger = CampaignBudgetLedger.create(ledger.path, campaign_id=campaign.name,
                 total_budget_seconds=TOTAL_GPU_SECONDS, source_hash=payload_hash(sources), plan_hash=plan_hash,
                 claim_boundary=CLAIM_BOUNDARY)
    payload = dict(ledger.read())
    if (payload["campaign_id"] != campaign.name or payload["total_budget_seconds"] != TOTAL_GPU_SECONDS
            or any(payload[name] != 0 for name in ("consumed_seconds", "reserved_seconds", "released_seconds"))
            or payload["attempts"] or payload["reservations"]):
        raise CheckpointError("only the unused prepared allocation may be initialized")
    if any((campaign / name).exists() for name in ("streams", "setup", "launches")):
        raise CheckpointError("worker artifacts exist without a campaign start; cannot adopt ledger")
    archive = campaign / "prepared-budget-ledger.json"
    if not archive.exists():
        durable_bytes(archive, ledger.path.read_bytes())
    original_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if initialization.exists():
        if record["prepared_ledger_sha256"] != original_hash:
            raise CheckpointError("prepared ledger archive changed")
    else:
        if archive.read_bytes() != ledger.path.read_bytes():
            raise CheckpointError("prepared ledger archive differs before initialization")
        durable_json(initialization, {"start": start, "prepared_ledger_sha256": original_hash})
    binding = {"source_hash": payload_hash(sources), "plan_hash": plan_hash, "claim_boundary": CLAIM_BOUNDARY}
    if any(payload[name] != value for name, value in binding.items()):
        if hashlib.sha256(ledger.path.read_bytes()).hexdigest() != original_hash:
            raise CheckpointError("unused ledger changed outside recorded initialization")
        payload.update(binding)
        payload["events"] = [*payload["events"], {"event": "unused_prepared_ledger_bound_for_execution",
                              "at_unix": start["started_at_unix"], "prepared_ledger_sha256": original_hash}]
        durable_json(ledger.path, payload)
    durable_json(start_path, start)
    verify_binding(start, campaign)
    return start, ledger


def verify_campaign_bundles(campaign: Path) -> dict[str, int]:
    bundles = 0
    tensors = 0
    for path in campaign.glob("**/committed/*/bundle.json"):
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != path.with_name("bundle.sha256").read_text().strip():
            raise CheckpointError(f"committed bundle checksum changed: {path}")

        def verify_tree(tree):
            nonlocal tensors
            if "tensor" in tree:
                receipt = tree["tensor"]
                if hashlib.sha256((path.parent / receipt["file"]).read_bytes()).hexdigest() != receipt["sha256"]:
                    raise CheckpointError(f"committed tensor checksum changed: {path}")
                tensors += 1
            for item in tree.get("sequence", []):
                verify_tree(item)
            for item in tree.get("mapping", {}).values():
                verify_tree(item)

        verify_tree(json.loads(content)["tree"])
        bundles += 1
    return {"verified_bundles": bundles, "verified_tensors": tensors}


def coordinator(campaign: Path, *, resume: bool = False, canary_only: bool = False,
                through_p1: bool = False, initialize_only: bool = False) -> int:
    campaign = campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".coordinator.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CheckpointError("campaign already has a live coordinator") from exc
        return _coordinator(campaign, resume=resume, canary_only=canary_only,
                            through_p1=through_p1, initialize_only=initialize_only)


def _coordinator(campaign: Path, *, resume: bool, canary_only: bool,
                 through_p1: bool, initialize_only: bool) -> int:
    campaign = campaign.resolve()
    start, ledger = initialize_campaign(campaign, resume=resume)
    execution_sources, source_migration = execution_source_binding(start, campaign)
    budget_campaign = campaign
    campaign, start = prepare_execution_namespace(campaign, start, execution_sources, source_migration)
    if initialize_only:
        print(json.dumps({"status": "INITIALIZED_NO_GPU_WORK", "campaign": str(budget_campaign),
                          "execution_root": str(campaign), "budget": ledger.read()}))
        return 0
    reconcile_interrupted_supervisor(budget_campaign, ledger)
    for mode, cap in (("reference", 3600.0), ("interrupt", 400.0), ("resume", 400.0)):
        if not (campaign / f"{mode}-complete.json").exists():
            _wave(campaign, mode, cap, ledger, start)
    reference = _read_results(json.loads((campaign / "reference-complete.json").read_bytes()))
    resumed = _read_results(json.loads((campaign / "resume-complete.json").read_bytes()))
    canaries = {arm: compare_canary(reference[arm], resumed[arm]) for arm in ARMS}
    if not (campaign / "canary-result.json").exists():
        durable_json(campaign / "canary-result.json", canaries)
    if canary_only:
        print(json.dumps({"status": "PASS_TWO_ARM_RECOVERY_CANARY", "budget": ledger.read()["consumed_seconds"]}))
        return 0
    if not (campaign / "runtime-complete.json").exists():
        _wave(campaign, "runtime", ARM_CAP_SECONDS, ledger, start)
    runtime = _read_results(json.loads((campaign / "runtime-complete.json").read_bytes()))
    forecasts = {arm: forecast(runtime[arm]) for arm in ARMS}
    result = {"status": "PASS_RECOVERY_HEALTH_RUNTIME_MEASUREMENT", "canaries": canaries,
              "forecasts": forecasts, "both_fit_arm_cap": all(row["fits_arm_cap"] for row in forecasts.values()),
              "ledger": ledger.read(), "remaining_gpu_seconds": ledger.remaining_seconds(),
              "p1_p2_launched": False, "result_file": str(RESULT)}
    if not (campaign / "validated-result.json").exists():
        durable_json(campaign / "validated-result.json", result)
    if through_p1:
        readiness_path = campaign / "phase0-readiness.json"
        if readiness_path.exists():
            readiness = json.loads(readiness_path.read_bytes())
            if (readiness["sources"] != start["sources"] or readiness["plan_hash"] != start["plan_hash"]
                    or readiness.get("execution_sources", readiness["sources"]) != execution_sources):
                raise CheckpointError("stale P1 readiness binding")
            allocations = readiness["allocations"]
        else:
            allocations = p1_allocations(canaries, forecasts, ledger.remaining_seconds())
            for arm in ARMS:
                seeds = [stream_seeds(campaign, arm, family) for family in ("canary", "runtime", "p1")]
                if len(set(seeds)) != 3:
                    raise CheckpointError("diagnostic and P1 seed families collide")
            readiness = {"status": "PASS_CURRENT_SOURCE_PHASE0_FOR_BOUNDED_P1", "sources": start["sources"],
                         "execution_sources": execution_sources, "source_migration": source_migration,
                         "plan_hash": start["plan_hash"], "canaries": canaries, "forecasts": forecasts,
                         "allocations": allocations, "bundle_verification": verify_campaign_bundles(campaign),
                         "p1_entrypoint": str(ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py"),
                         "remaining_gpu_seconds": ledger.remaining_seconds(), "posterior_admitted": False}
            durable_json(readiness_path, readiness)
        if not (campaign / "p1-complete.json").exists():
            p1_wave = _wave(campaign, "p1", {arm: allocations[arm]["cap_seconds"] for arm in ARMS}, ledger, start,
                            allocations=allocations)
        else:
            p1_wave = json.loads((campaign / "p1-complete.json").read_bytes())
        outcomes = _read_results(p1_wave)
        decisions = {arm: p1_outcome(outcomes[arm]["controller"]) for arm in ARMS}
        partial = any(value == "resource_cap_partial_not_completed" for value in decisions.values())
        result = {"status": "P1_RESOURCE_CAP_PARTIAL" if partial else "P1_BOUNDED_EXECUTION_COMPLETED",
                  "decisions": decisions, "p1_completed": not partial, "posterior_admitted": False,
                  "p2_launched": False, "ledger": ledger.read(), "remaining_gpu_seconds": ledger.remaining_seconds(),
                  "bundle_verification": verify_campaign_bundles(campaign), "result_file": str(RESULT)}
        result_path = campaign / (f"p1-partial-{uuid.uuid4().hex[:10]}.json" if partial else "p1-result.json")
        if not result_path.exists():
            durable_json(result_path, result)
    verify_binding(start, campaign)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--through-p1", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    if args.worker_job:
        return worker(args.worker_job)
    if args.campaign_root is None:
        parser.error("--campaign-root is required")
    if args.canary_only and args.through_p1:
        parser.error("--canary-only and --through-p1 are mutually exclusive")
    return coordinator(args.campaign_root, resume=args.resume, canary_only=args.canary_only,
                       through_p1=args.through_p1, initialize_only=args.initialize_only)


if __name__ == "__main__":
    raise SystemExit(main())

"""Tiny proposal-localization diagnostic; never a tuning or inference route."""

from dataclasses import replace
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-uuid")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--sample-chain-results", type=int, default=0)
parser.add_argument("--cached-gradient", action="store_true")
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != (args.gpu_uuid or "-1"):
    raise RuntimeError("Diagnostic device environment mismatch")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Memory growth must be enabled before TensorFlow import")

ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(exist_ok=False)
sys.path.insert(0, str(ROOT))
started = time.monotonic()

import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True) if args.gpu_uuid else None
if args.gpu_uuid:
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    if len(tf.config.list_logical_devices("GPU")) != 1:
        raise RuntimeError("Exactly one GPU is required")

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.inference.neutra_hmc import reviewed_value_score_target_fn

durable_json(OUTPUT / "memory_policy.json", memory)

spec = importlib.util.spec_from_file_location(
    "proposal_recovery_diagnostic",
    ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py",
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
job = json.loads((CAMPAIGN / "launches/runtime-c48f1f0535/strict-job.json").read_text())
assert recovery.source_hashes() == job["sources"]
parallel = recovery.load_parallel()
source = parallel._load_source()
profile = replace(parallel._profile(source, CAMPAIGN, "strict", recovery.ARM_CAP_SECONDS), plan_path=recovery.PLAN)
for key in ("chart", "tuning-repair-r1"):
    assert (CAMPAIGN / "setup/strict/committed" / key / "bundle.json").is_file()
chart, handoff, setup = recovery._setup(tf, job, source, profile)
assert all(row["replayed"] for row in setup["stages"])
durable_json(OUTPUT / "setup.json", setup)
checkpoint = CAMPAIGN / "streams/strict/runtime/chunks"
inputs = json.loads((CAMPAIGN / "launches/runtime-c48f1f0535/strict/chunk-000000-end.json").read_text())["inputs"]
with DurableTensorCheckpoint(checkpoint, json.loads((checkpoint / "identity.json").read_text())) as store:
    (samples, original_trace), metadata = store.load("chunk-000000", inputs)
target = reviewed_value_score_target_fn(handoff.transformed_adapter, dtype=tf.float64, require_batched=True)
carried_trace = None
if args.cached_gradient:
    if args.sample_chain_results:
        raise RuntimeError("Cached-gradient and prefix modes are distinct")
    prefix_root = CAMPAIGN / "launches/prefix-diagnostic-83873ea777/strict/prefix"
    with DurableTensorCheckpoint(prefix_root, json.loads((prefix_root / "identity.json").read_text())) as store:
        (carried_samples, carried_trace), _ = store.load("prefix", {"original_inputs": inputs, "num_results": 159})
    assert bool(tf.reduce_all(carried_samples == samples[:159]).numpy())


@tf.function(input_signature=(tf.TensorSpec((4, 4), tf.float64), tf.TensorSpec((2,), tf.int32), tf.TensorSpec((), tf.int32), tf.TensorSpec((), tf.int32), tf.TensorSpec((4, 4), tf.float64), tf.TensorSpec((4,), tf.float64)), jit_compile=True, autograph=False)
def replay(current_state, seed, transition, leapfrogs, carried_score, carried_target):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target, step_size=tf.constant(handoff.step_size, tf.float64),
        num_leapfrog_steps=leapfrogs, state_gradients_are_stopped=True,
    )
    chain_seed = samplers.sanitize_seed(seed, salt="mcmc.sample_chain")

    def split_step(index, passalong, step_seed):
        next_step, next_passalong = samplers.split_seed(passalong)
        return index + 1, next_passalong, next_step

    _, _, step_seed = tf.while_loop(
        lambda index, passalong, step_seed: index <= transition,
        split_step, (tf.constant(0), chain_seed, chain_seed),
    )
    previous = kernel.bootstrap_results(current_state)
    if args.cached_gradient:
        previous = previous._replace(accepted_results=previous.accepted_results._replace(
            grads_target_log_prob=[carried_score], target_log_prob=carried_target))
    sampled, result = kernel.one_step(current_state, previous, seed=step_seed)
    proposed = result.proposed_results
    momentum_before = proposed.initial_momentum[0]
    momentum_after = proposed.final_momentum[0]
    raw_kinetic = 0.5 * (tf.reduce_sum(momentum_before ** 2, axis=-1) - tf.reduce_sum(momentum_after ** 2, axis=-1))
    return {
        "sampled_state": sampled, "proposed_state": result.proposed_state,
        "log_accept_ratio": result.log_accept_ratio, "is_accepted": result.is_accepted,
        "previous_target": previous.accepted_results.target_log_prob,
        "previous_score": previous.accepted_results.grads_target_log_prob[0],
        "proposed_target": proposed.target_log_prob,
        "proposed_score": proposed.grads_target_log_prob[0],
        "initial_momentum": momentum_before, "final_momentum": momentum_after,
        "kinetic_correction": proposed.log_acceptance_correction,
        "raw_kinetic_correction": raw_kinetic,
        "raw_log_accept_ratio": proposed.target_log_prob - previous.accepted_results.target_log_prob + raw_kinetic,
        "step_seed": step_seed,
        "proposed_model_state": chart.forward_batch(result.proposed_state),
        "proposed_status": handoff.transformed_adapter.target_status_telemetry(result.proposed_state),
    }


rows = []
if args.sample_chain_results:
    if args.sample_chain_results != 159:
        raise RuntimeError("Only the predeclared 159-transition prefix is permitted")
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target, step_size=tf.constant(handoff.step_size, tf.float64),
        num_leapfrog_steps=handoff.num_leapfrog_steps, state_gradients_are_stopped=True,
    )

    def trace_proposal(state, result):
        return {
            "is_accepted": result.is_accepted, "log_accept_ratio": result.log_accept_ratio,
            "target_log_prob": result.accepted_results.target_log_prob,
            "target_status": handoff.transformed_adapter.target_status_telemetry(state),
            "accepted_score": result.accepted_results.grads_target_log_prob[0],
            "proposed_state": result.proposed_state,
            "proposed_target": result.proposed_results.target_log_prob,
            "proposed_score": result.proposed_results.grads_target_log_prob[0],
            "initial_momentum": result.proposed_results.initial_momentum[0],
            "final_momentum": result.proposed_results.final_momentum[0],
            "kinetic_correction": result.proposed_results.log_acceptance_correction,
            "kernel_seed": result.seed,
        }

    @tf.function(input_signature=(tf.TensorSpec((4, 4), tf.float64), tf.TensorSpec((2,), tf.int32)), jit_compile=True, autograph=False)
    def prefix(initial_state, seed):
        return tfp.mcmc.sample_chain(
            num_results=args.sample_chain_results, num_burnin_steps=0,
            current_state=initial_state, kernel=kernel, trace_fn=trace_proposal, seed=seed,
        )

    prefix_inputs = {"original_inputs": inputs, "num_results": args.sample_chain_results}
    with DurableTensorCheckpoint(OUTPUT / "prefix", {"original_sources": job["sources"], "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) as prefix_store:
        prefix_samples, prefix_trace = prefix_store.run(
            "prefix", prefix_inputs,
            lambda: prefix(tf.constant(profile.initial_state_bank, tf.float64), tf.constant(inputs["seed"], tf.int32)),
        )
    equal_rows = tf.reduce_all(prefix_samples == samples[:args.sample_chain_results], axis=-1)
    deviations = tf.where(~equal_rows).numpy().tolist()
    nonfinite = tf.where(~tf.math.is_finite(prefix_trace["log_accept_ratio"])).numpy().tolist()
    bad_proposals = []
    for transition, chain in nonfinite:
        values = {key: value[transition].numpy().tolist() for key, value in prefix_trace.items() if key != "target_status"}
        proposed = prefix_trace["proposed_state"][transition]
        proposed_status = None
        if bool(tf.reduce_all(tf.math.is_finite(proposed)).numpy()):
            proposed_status = tf.nest.map_structure(lambda value: value.numpy().tolist(), handoff.transformed_adapter.target_status_telemetry(proposed))
        bad_proposals.append({"transition_zero_based": transition, "chain_zero_based": chain,
                              "values": values, "finite_proposed_endpoint_status": proposed_status})
    comparison = {
        "all_prefix_samples_exact": bool(tf.reduce_all(equal_rows).numpy()),
        "first_sample_deviation": deviations[0] if deviations else None,
        "maximum_sample_residual": float(tf.reduce_max(tf.abs(prefix_samples - samples[:args.sample_chain_results])).numpy()),
        "all_acceptance_decisions_exact": bool(tf.reduce_all(prefix_trace["is_accepted"] == original_trace["is_accepted"][:args.sample_chain_results]).numpy()),
        "all_log_accept_ratios_exact": bool(tf.reduce_all(prefix_trace["log_accept_ratio"] == original_trace["log_accept_ratio"][:args.sample_chain_results]).numpy()),
        "all_sampled_targets_exact": bool(tf.reduce_all(prefix_trace["target_log_prob"] == original_trace["target_log_prob"][:args.sample_chain_results]).numpy()),
        "nonfinite_events": nonfinite, "bad_proposals": bad_proposals,
    }
    rows.append(comparison)
    durable_json(OUTPUT / "prefix-comparison.json", recovery.diagnostic_payload(comparison))
    print(json.dumps(recovery.diagnostic_payload(comparison)), flush=True)

for transition, leapfrogs in (() if args.sample_chain_results else ((157, 3), (158, 3), (158, 1), (158, 2))):
    current = samples[transition - 1]
    call_started = time.monotonic()
    carried_score = carried_trace["accepted_score"][transition - 1] if carried_trace is not None else tf.zeros((4, 4), tf.float64)
    carried_target = carried_trace["target_log_prob"][transition - 1] if carried_trace is not None else tf.zeros((4,), tf.float64)
    values = replay(current, tf.constant(inputs["seed"], tf.int32), tf.constant(transition, tf.int32), tf.constant(leapfrogs, tf.int32), carried_score, carried_target)
    materialized = tf.nest.map_structure(lambda value: value.numpy().tolist(), values)
    row = {
        "transition_zero_based": transition, "leapfrogs": leapfrogs, "seconds": time.monotonic() - call_started,
        "sampled_states_exact": bool(tf.reduce_all(values["sampled_state"] == samples[transition]).numpy()),
        "log_accept_ratios_exact": bool(tf.reduce_all(values["log_accept_ratio"] == original_trace["log_accept_ratio"][transition]).numpy()),
        "maximum_sample_residual": float(tf.reduce_max(tf.abs(values["sampled_state"] - samples[transition])).numpy()),
        "original_log_accept_ratio": original_trace["log_accept_ratio"][transition].numpy().tolist(),
        "values": materialized,
    }
    rows.append(row)
    durable_json(OUTPUT / f"transition-{transition}-L{leapfrogs}.json", recovery.diagnostic_payload(row))
    print(json.dumps(recovery.diagnostic_payload(row)), flush=True)
assert recovery.source_hashes() == job["sources"]
durable_json(OUTPUT / "run_manifest.json", recovery.diagnostic_payload({
    "status": "completed", "cpu_only_diagnostic": not args.gpu_uuid, "gpu_devices_intentionally_hidden": not args.gpu_uuid,
    "gpu_uuid": args.gpu_uuid, "memory_policy": memory,
    "original_carried_gradient_used": args.cached_gradient,
    "allocator": tf.config.experimental.get_memory_info("GPU:0") if args.gpu_uuid else None,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__, "jit_compile": True,
    "command": [sys.executable, *sys.argv], "sources": job["sources"],
    "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "rows": rows, "wall_seconds": time.monotonic() - started,
    "nonclaims": ["no GPU equivalence without control agreement", "no tuning or inference", "no health-veto removal"],
}))

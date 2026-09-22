"""Tiny proposal-localization diagnostic; never a tuning or inference route."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This diagnostic requires intentionally hidden GPUs")

ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("cpu-replay-r1")
OUTPUT.mkdir(exist_ok=False)
sys.path.insert(0, str(ROOT))
started = time.monotonic()

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.inference.neutra_hmc import reviewed_value_score_target_fn

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
kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target, step_size=tf.constant(handoff.step_size, tf.float64),
    num_leapfrog_steps=handoff.num_leapfrog_steps, state_gradients_are_stopped=True,
)


@tf.function(input_signature=(tf.TensorSpec((4, 4), tf.float64), tf.TensorSpec((2,), tf.int32), tf.TensorSpec((), tf.int32)), jit_compile=True, autograph=False)
def replay(current_state, seed, transition):
    chain_seed = samplers.sanitize_seed(seed, salt="mcmc.sample_chain")

    def split_step(index, passalong, step_seed):
        next_step, next_passalong = samplers.split_seed(passalong)
        return index + 1, next_passalong, next_step

    _, _, step_seed = tf.while_loop(
        lambda index, passalong, step_seed: index <= transition,
        split_step, (tf.constant(0), chain_seed, chain_seed),
    )
    previous = kernel.bootstrap_results(current_state)
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
    }


rows = []
for transition in (157, 158):
    current = samples[transition - 1]
    call_started = time.monotonic()
    values = replay(current, tf.constant(inputs["seed"], tf.int32), tf.constant(transition, tf.int32))
    materialized = {key: value.numpy().tolist() for key, value in values.items()}
    row = {
        "transition_zero_based": transition, "seconds": time.monotonic() - call_started,
        "sampled_states_exact": bool(tf.reduce_all(values["sampled_state"] == samples[transition]).numpy()),
        "maximum_sample_residual": float(tf.reduce_max(tf.abs(values["sampled_state"] - samples[transition])).numpy()),
        "original_log_accept_ratio": original_trace["log_accept_ratio"][transition].numpy().tolist(),
        "values": materialized,
    }
    rows.append(row)
    durable_json(OUTPUT / f"transition-{transition}.json", recovery.diagnostic_payload(row))
    print(json.dumps(recovery.diagnostic_payload(row)), flush=True)
durable_json(OUTPUT / "run_manifest.json", recovery.diagnostic_payload({
    "status": "completed", "cpu_only_diagnostic": True, "gpu_devices_intentionally_hidden": True,
    "tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__, "jit_compile": True,
    "command": [sys.executable, *sys.argv], "sources": job["sources"],
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "rows": rows, "wall_seconds": time.monotonic() - started,
    "nonclaims": ["no GPU equivalence without control agreement", "no tuning or inference", "no health-veto removal"],
}))

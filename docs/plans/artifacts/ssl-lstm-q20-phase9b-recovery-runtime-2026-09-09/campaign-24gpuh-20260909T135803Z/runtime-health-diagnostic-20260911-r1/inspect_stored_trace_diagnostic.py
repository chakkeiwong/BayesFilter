"""CPU-only post-run diagnostic; no sampling, training, or admission authority."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This diagnostic requires intentionally hidden GPUs")

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json

spec = importlib.util.spec_from_file_location(
    "recovery_health_diagnostic",
    ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py",
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)

from bayesfilter.inference.neutra_hmc import BatchedHMCConfig


def finite_summary(tensor):
    values = tf.convert_to_tensor(tensor)
    finite = tf.math.is_finite(values)
    selected = tf.boolean_mask(values, finite)
    return {
        "shape": values.shape.as_list(),
        "nan_count": int(tf.math.count_nonzero(tf.math.is_nan(values)).numpy()),
        "positive_infinity_count": int(tf.math.count_nonzero(values == float("inf")).numpy()),
        "negative_infinity_count": int(tf.math.count_nonzero(values == -float("inf")).numpy()),
        "finite_min": float(tf.reduce_min(selected).numpy()) if int(tf.size(selected)) else None,
        "finite_max": float(tf.reduce_max(selected).numpy()) if int(tf.size(selected)) else None,
    }


started = time.monotonic()
report = {"schema": "bayesfilter.phase9b.stored_runtime_trace_diagnostic.v1", "arms": {}}
for arm in ("strict", "factor"):
    output = CAMPAIGN / "launches/runtime-c48f1f0535" / arm
    checkpoint = CAMPAIGN / "streams" / arm / "runtime/chunks"
    setup = json.loads((output / "setup-receipt.json").read_text())
    previous_state = tf.constant(setup["profile"]["initial_state_bank"], tf.float64)
    rows = []
    with DurableTensorCheckpoint(checkpoint, json.loads((checkpoint / "identity.json").read_text())) as store:
        for bundle in sorted((checkpoint / "committed").glob("chunk-*/bundle.json")):
            key = bundle.parent.name
            inputs = json.loads((output / f"{key}-end.json").read_text())["inputs"]
            (samples, trace), metadata = store.load(key, inputs)
            assert store.tensor_hash(previous_state) == inputs["initial_state_sha256"]
            config = BatchedHMCConfig(
                num_results=inputs["active_results"], num_burnin_steps=0,
                step_size=inputs["config"]["step_size"],
                num_leapfrog_steps=inputs["config"]["num_leapfrog_steps"],
                seed=tuple(inputs["seed"]), jit_compile=True,
            )
            health = recovery.full_health(tf, samples, trace, previous_state, config)
            saved_health = json.loads((output / f"{key}-health.json").read_text())
            assert health["vetoes"] == saved_health["vetoes"]
            positions = tf.where(~tf.math.is_finite(trace["log_accept_ratio"])).numpy().tolist()
            events = []
            for transition, chain in positions:
                before = samples[transition - 1, chain] if transition else previous_state[chain]
                after = samples[transition, chain]
                events.append({
                    "transition_zero_based": transition, "chain_zero_based": chain,
                    "log_accept_ratio": str(float(trace["log_accept_ratio"][transition, chain].numpy())),
                    "is_accepted": bool(trace["is_accepted"][transition, chain].numpy()),
                    "state_unchanged": bool(tf.reduce_all(before == after).numpy()),
                    "previous_latent_state": before.numpy().tolist(),
                    "sampled_target_log_prob": float(trace["target_log_prob"][transition, chain].numpy()),
                })
            rows.append({
                "key": key, "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "all_bundle_and_tensor_checksums_verified": True,
                "recomputed_vetoes": health["vetoes"], "metadata": metadata,
                "log_accept_ratio": finite_summary(trace["log_accept_ratio"]),
                "samples": finite_summary(samples),
                "target_log_prob": finite_summary(trace["target_log_prob"]),
                "nonfinite_events": events, "trace_fields": sorted(trace),
                "status_fields": sorted(trace["target_status"]),
                "sampled_status_passed": health["target_status_passed"],
            })
            previous_state = samples[-1]
    report["arms"][arm] = rows

report.update({
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "command": " ".join([sys.executable, *sys.argv]),
    "environment": {name: os.environ.get(name) for name in (
        "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "tensorflow_version": tf.__version__, "device": "CPU; GPUs intentionally hidden",
    "wall_seconds": time.monotonic() - started,
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "nonclaims": ["no new sampler execution", "no candidate ranking", "no posterior or default promotion"],
})
durable_json(Path(__file__).with_name("stored-trace-report.json"), report)
for arm, rows in report["arms"].items():
    for row in rows:
        print(arm, row["key"], row["log_accept_ratio"], "vetoes", row["recomputed_vetoes"])
        print("nonfinite events", row["nonfinite_events"])

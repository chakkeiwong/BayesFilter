"""CPU-only recovery of completed diagnostic chunks from the bounded cost run."""
from __future__ import annotations

import json
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash
from bayesfilter.inference.hmc_tensorflow_tuning import _TensorChunk
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import STARTS, chunk_health, mechanics

PLAN = "docs/plans/surrogate-hmc-m13-four-chain-summary-2026-09-25.md"
BASE = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-four-chain-cost-02"


def decode_chunk(record):
    bool_fields = {"divergence", "force_fallback", "is_accepted"}
    return _TensorChunk(**{k: tf.constant(v, tf.bool if k in bool_fields else
        tf.int32 if k == "endpoint_status" else tf.float64) for k, v in record.items()})


def assess_chunk(record, start):
    chunk = decode_chunk(record)
    if chunk.states.shape != (8, 4, 2) or chunk.endpoint_status.shape[:2] != (8, 4):
        raise ValueError("unexpected completed chunk shape")
    health = chunk_health(chunk, tf.constant(start, tf.float64))
    no_fallback = not bool(tf.reduce_any(chunk.force_fallback))
    valid = all(bool(value) for value in health.values()) and no_fallback
    return dict(passed=valid, health=mechanics.serial(health), no_force_fallback=no_fallback,
        acceptance_by_chain=mechanics.serial(tf.reduce_mean(tf.cast(chunk.is_accepted, tf.float64), axis=0)),
        mean_acceptance_probability_by_chain=mechanics.serial(tf.reduce_mean(
            tf.exp(tf.minimum(chunk.log_accept_ratio, 0.)), axis=0)),
        maximum_absolute_energy_error=float(tf.reduce_max(tf.abs(chunk.delta_h))),
        final_state=mechanics.serial(chunk.final_state))


def summary_stage(output):
    started = time.monotonic()
    out = BASE / "m13-four-chain-cost"
    parent = json.loads((BASE / "report.json").read_text())
    manifest = json.loads((BASE / "run-manifest.json").read_text())
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text())
    if not (parent.get("source_unchanged") and
            parent["stages"]["m13-four-chain-cost"]["execution"] == "JOB_CPU_BUDGET_LIMIT"):
        raise ValueError("expected unchanged-source resource-limited donor")
    dependencies = set(progress["scope"]["source_hashes"]) | {
        "bayesfilter/testing/surrogate_hmc_m13_four_chain_cost.py",
        "bayesfilter/inference/hmc_tensorflow_tuning.py",
        "bayesfilter/inference/hmc_tensorflow_verified.py",
        "bayesfilter/inference/tuning_contract.py"}
    for path in dependencies:
        expected = progress["scope"]["source_hashes"].get(path, manifest["source_hashes"].get(path))
        if expected != file_hash(ROOT / path):
            raise ValueError("donor numerical dependency changed: "+path)
    chunks, timings = [], {2: [], 4: []}
    for leapfrogs, repeat_count in ((2, 2), (4, 1)):
        state = STARTS
        names = [f"l{leapfrogs}-compile-chunk.json"] + [f"l{leapfrogs}-repeat{i}.json" for i in range(repeat_count)]
        for name in names:
            path = out/name
            record = json.loads(path.read_text())
            is_repeat = "chunk" in record
            data = record["chunk"] if is_repeat else record
            result = assess_chunk(data, state)
            if is_repeat:
                seconds = record["metrics"]["seconds_per_four_chain_transition"]
                if not (0. < seconds < float("inf")):
                    raise ValueError("invalid saved timing")
                timings[leapfrogs].append(seconds)
            chunks.append(dict(path=str(path), sha256=file_hash(path), **result))
            state = result["final_state"]
    t2, t4 = max(timings[2]), max(timings[4])
    healthy = all(c["passed"] for c in chunks)
    prereqs = all(progress["endpoint"][k] for k in ("valid", "replay_exact", "scalar_parity")) and all(
        progress["force"][k] for k in ("valid", "scalar_valid", "replay_exact"))
    return dict(status="M13_FOUR_CHAIN_CONNECTION_PASSED_COST_PARTIAL" if healthy and prereqs
                else "M13_FOUR_CHAIN_RECOVERY_INVALID", qualified=healthy and prereqs,
        plan=PLAN, diagnostic_scope="completed native four-chain mechanics only",
        original_resource_stop_preserved=True, missing_timing_repeats={"L4": 1},
        baseline_report=str(BASE/"report.json"), baseline_sha256=file_hash(BASE/"report.json"),
        progress_path=str(progress_path), progress_sha256=file_hash(progress_path),
        scope=progress["scope"], binding_identity=progress["binding_identity"],
        endpoint=progress["endpoint"], force=progress["force"], chunks=chunks,
        transition_count_per_chain=40, total_proposals=160,
        cost_forecast=dict(seconds_per_four_chain_transition=timings,
            conservative_L4_seconds=1.5*t4,
            extrapolated_L8_seconds=1.5*max(t4, 3*t4-2*t2),
            planning_multiplier=1.5, confidence_interval=False,
            caveat="two L2 repeats and one L4 repeat; long-run cost unverified; compilation excluded"),
        posterior_qualified=False, tuning_admitted=False, ranking_supported=False,
        inference_launched=False, all_models_validated=False,
        next_action="budgeted scoped controls, public tuning, retained comparison under master",
        wall_seconds=time.monotonic()-started)

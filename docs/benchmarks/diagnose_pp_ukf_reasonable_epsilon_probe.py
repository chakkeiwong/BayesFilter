#!/usr/bin/env python3
"""Bounded PP-UKF one-proposal diagnostic for reasonable-epsilon setup.

This is an engineering diagnostic only. It performs one scalar target
evaluation, one HMC bootstrap, and one proposal with the same frozen
transport used by the PP-UKF tuning-only campaign. It does not tune, sample,
or establish numerical or scientific validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-transport", type=Path, required=True)
    parser.add_argument("--frozen-transport-sha256", required=True)
    parser.add_argument("--step-size", type=float, default=0.3194715521231362)
    parser.add_argument("--num-leapfrog-steps", type=int, default=1)
    parser.add_argument("--jit-compile", action="store_true")
    parser.add_argument("--run-epsilon-search", action="store_true")
    parser.add_argument("--epsilon-max-attempts", type=int, default=2)
    parser.add_argument("--epsilon-probe-count", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.hmc_warmup import find_reasonable_epsilon
    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == "PP-UKF")
    raw_base = spec.adapter_factory()
    observed_adapter_signature = raw_base.adapter_signature()
    base = BatchNativeBoundAdapter(raw_base, target_signature=spec.target_signature)
    frozen_sha = _sha256(args.frozen_transport)
    if frozen_sha != str(args.frozen_transport_sha256).lower():
        raise ValueError(f"frozen transport SHA mismatch: {frozen_sha}")
    payload = json.loads(args.frozen_transport.read_text(encoding="utf-8"))
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=spec.target_signature
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope="PP-UKF:fixed_neutra_reasonable_epsilon_probe",
        evidence_path=str(Path(__file__).relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    state = tf.zeros((spec.parameter_dim,), tf.float64)
    row = {
        "schema": "bayesfilter.pp_ukf_reasonable_epsilon_probe.v1",
        "role": "engineering_diagnostic_only",
        "nonclaims": [
            "no tuning result",
            "no HMC sampling result",
            "no posterior convergence claim",
            "no PP-UKF numerical-validity claim",
            "no scientific claim",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "target_signature": spec.target_signature,
        "adapter_signature": observed_adapter_signature,
        "frozen_transport": str(args.frozen_transport),
        "frozen_transport_sha256": frozen_sha,
        "jit_compile": bool(args.jit_compile),
        "step_size": float(args.step_size),
        "num_leapfrog_steps": int(args.num_leapfrog_steps),
        "device_list": [str(item) for item in tf.config.list_logical_devices()],
    }

    started = time.perf_counter()
    value, score = adapter.log_prob_and_grad(state)
    row["target_eval_s"] = time.perf_counter() - started
    row["target_value_finite"] = bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    row["target_score_finite"] = bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    row["target_value"] = _json_ready(value)
    row["target_score"] = _json_ready(score)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda theta: adapter.log_prob_and_grad(theta)[0],
        step_size=tf.constant(float(args.step_size), tf.float64),
        num_leapfrog_steps=int(args.num_leapfrog_steps),
    )
    started = time.perf_counter()
    bootstrap_fn = kernel.bootstrap_results
    if args.jit_compile:
        bootstrap_fn = tf.function(bootstrap_fn, jit_compile=True, reduce_retracing=True)
    results = bootstrap_fn(state)
    row["bootstrap_s"] = time.perf_counter() - started
    row["bootstrap_target_log_prob_finite"] = bool(
        tf.reduce_all(tf.math.is_finite(results.accepted_results.target_log_prob)).numpy()
    )
    started = time.perf_counter()
    one_step = kernel.one_step
    if args.jit_compile:
        one_step = tf.function(one_step, jit_compile=True, reduce_retracing=True)
    next_state, next_results = one_step(
        state, results, seed=tf.constant([20260721, 1], tf.int32)
    )
    row["proposal_s"] = time.perf_counter() - started
    row["proposal_state_finite"] = bool(tf.reduce_all(tf.math.is_finite(next_state)).numpy())
    row["proposal_log_accept_ratio_finite"] = bool(
        tf.reduce_all(tf.math.is_finite(next_results.log_accept_ratio)).numpy()
    )
    row["proposal_log_accept_ratio"] = _json_ready(next_results.log_accept_ratio)
    if args.run_epsilon_search:
        started = time.perf_counter()
        epsilon = find_reasonable_epsilon(
            adapter=adapter,
            current_state=state,
            initial_step_size=float(args.step_size),
            seed=(20260721, 11),
            max_attempts=int(args.epsilon_max_attempts),
            lower_acceptance=0.05,
            upper_acceptance=0.95,
            num_leapfrog_steps=int(args.num_leapfrog_steps),
            momentum_probe_count=int(args.epsilon_probe_count),
            jit_compile=bool(args.jit_compile),
        )
        row["epsilon_search_s"] = time.perf_counter() - started
        row["epsilon_search"] = epsilon.payload()
    row["status"] = "completed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: row[key] for key in ("status", "target_eval_s", "bootstrap_s", "proposal_s")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

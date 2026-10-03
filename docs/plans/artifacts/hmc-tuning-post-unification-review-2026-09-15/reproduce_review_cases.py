"""Bounded CPU engineering diagnostics, not sampler or posterior evidence."""
from __future__ import annotations

import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This review diagnostic requires intentionally hidden GPUs")

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference import (
    HMCAcceptancePolicy, HMCControllerConfig, HMCKernelTuningConfig,
    tune_hmc_kernel, run_typed_hmc_candidate_set,
    resume_hmc_candidate_set_tuning,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCTuningCandidateSetController, HMCTuningScopeCollection,
)
from bayesfilter.inference.hmc_verification import evaluate_hmc_acceptance_evidence
from tests.test_hmc_candidate_set_tuning import _scope, _config, _pass
from tests.test_hmc_candidate_set_execution import (
    GaussianTarget, make_binding, execution_config,
)


def main(destination):
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = {}

    # A valid acceptance sample and native divergence have different roles.
    binding = make_binding(config=execution_config(num_warmup_steps=0,
        measurement_num_results=64, verification_num_results=64,
        acceptance_policy=HMCAcceptancePolicy()))
    samples = tf.random.stateless_normal([64, 4, 2], (20260915, 410), dtype=tf.float64)
    shape = [64, 4]
    log_accept = tf.fill(shape, tf.math.log(tf.constant(.2, tf.float64)))
    accepted = tf.ones(shape, tf.bool)
    divergence = tf.tensor_scatter_nd_update(tf.zeros(shape, tf.bool), [[0, 0]], [True])
    trace = dict(is_accepted=accepted, log_accept_ratio=log_accept,
        target_log_prob=tf.zeros(shape, tf.float64),
        proposed_target_log_prob=tf.zeros(shape, tf.float64),
        target_score_finite=tf.ones(shape, tf.bool), proposed_state=samples,
        initial_momentum=tf.ones_like(samples), final_momentum=tf.ones_like(samples),
        divergence=divergence,
        target_status_telemetry=GaussianTarget().target_status_telemetry(samples),
        proposed_target_status_telemetry=GaussianTarget().target_status_telemetry(samples))
    policy = evaluate_hmc_acceptance_evidence(samples=samples,
        log_accept_ratio=log_accept, is_accepted=accepted,
        policy=HMCAcceptancePolicy(), native_divergence_status="available",
        native_divergence_count=1)
    analysis = binding.analyze(binding.initial_active_state, samples, trace)
    rows["native_divergence_roles"] = {
        "common_policy": {"validity": policy.evidence_validity,
            "decision": policy.acceptance_decision,
            "promotion_vetoes": policy.candidate_promotion_vetoes},
        "numerical_adapter": {key: analysis[key] for key in
            ("decision", "evidence_validity", "hard_vetoes", "repair_eligible")},
        "evidence_kind": "synthetic finite trace; tests classification, not divergence incidence",
    }

    # The next large measurement blocks an already queued affordable check.
    cfg = _config(grid=(3, 25), epsilons=((3, (.25,)), (25, (.25,))),
        max_gradient_work=600)
    result = HMCTuningCandidateSetController(_scope(), cfg).run(_pass,
        work_cost=lambda work, candidate: {"gradient_work": 100 * candidate.leapfrog_steps})
    by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
    rows["affordable_verification_blocked"] = {
        "completion": result.completion_status,
        "used": result.search_state["gradient_work"], "ceiling": cfg.max_gradient_work,
        "verified": result.verified_candidate_ids,
        "work": [{"L": by_id[w.candidate_id].leapfrog_steps, "stage": w.stage,
            "status": w.status, "cost": 100 * by_id[w.candidate_id].leapfrog_steps}
            for w in result.work_items],
    }

    # One scope may legitimately have two searches; collection lookup uses only
    # scope_id and must not reject a verified member in the later search.
    one = HMCTuningCandidateSetController(_scope(), _config()).run(_pass)
    two = HMCTuningCandidateSetController(replace(_scope(), search_id="second-search"),
        _config()).run(_pass)
    collection = HMCTuningScopeCollection((one, two))
    try:
        collection.member(two.scope.scope_id, two.verified_candidate_ids[0])
        lookup_error = None
    except ValueError as exc:
        lookup_error = str(exc)
    rows["collection_search_ambiguity"] = {
        "accepted_search_count": len(collection.results),
        "requested_id": two.verified_candidate_ids[0], "lookup_error": lookup_error,
    }

    # Invalid search config is discovered after preparation starts.
    entered = []
    def preparation_sentinel(**kwargs):
        entered.append(True)
        raise RuntimeError("review sentinel: preparation entered")
    with patch("bayesfilter.inference.hmc_kernel_tuning.prepare_operational_windowed_mass_handoff",
               preparation_sentinel):
        try:
            tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[0., 0.],
                config=HMCKernelTuningConfig.smoke(target_scope="candidate-bridge-test"),
                search_config=object())
        except Exception as exc:
            rows["late_search_preflight"] = {"preparation_entered": bool(entered),
                "exception_type": type(exc).__name__, "message": str(exc)}

    # The typed fixed-transport branch ignores extra controls instead of
    # forwarding them to the ordinary branch's explicit rejection.
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import tune_fixed_transport_hmc_kernel
    captured = {}
    def dispatch_probe(**kwargs):
        captured.update(kwargs)
        return "dispatch inspected"
    with patch("bayesfilter.inference.hmc_tuning_dispatch.tune_hmc_kernel", dispatch_probe):
        tune_fixed_transport_hmc_kernel(base_adapter=object(), fixed_transport=object(),
            initial_position=[0.], config=_config(), candidate_set_adapter=object(),
            search_config=object(), execution_config=object(), target_lineage={"data": "changed"},
            source_paths=["new_dependency.py"])
    rows["fixed_transport_dropped_controls"] = {
        "forwarded_keys": sorted(captured),
        "dropped": [k for k in ("search_config", "execution_config", "target_lineage", "source_paths")
                    if k not in captured],
        "evidence_kind": "dispatch-boundary spy; does not issue numerical evidence",
    }

    # Budgeted numerical restart preserves chunks but charges the entire work
    # item again. Restrict this to one synthetic Gaussian candidate.
    numeric_binding = make_binding(config=execution_config(chunk_max_results=64))
    cfg = _config(grid=(3,), epsilons=((3, (1.3,)),), evidence_rungs=(1,),
        max_gradient_work=4352)
    original_run = numeric_binding._run
    calls = 0
    def fail_second_call(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise tf.errors.ResourceExhaustedError(None, None, "review injected resource interruption")
        return original_run(*args)
    numeric_binding._run = fail_second_call
    first = run_typed_hmc_candidate_set(numeric_binding.typed_adapter, cfg,
        output_dir=destination / "budget-restart")
    first_chunks = [c["count"] for chunks in numeric_binding._partial.values() for c in chunks]
    resumed = resume_hmc_candidate_set_tuning(destination / "budget-restart/tuning_checkpoint.json",
        adapter=GaussianTarget())
    rows["resume_work_charge"] = {
        "first_completion": first.result.completion_status,
        "first_saved_chunk_counts": first_chunks,
        "first_charged_work": first.result.search_state["gradient_work"],
        "resumed_completion": resumed.result.completion_status,
        "resumed_charged_work": resumed.result.search_state["gradient_work"],
        "resumed_work": [{"stage": w.stage, "status": w.status} for w in resumed.result.work_items],
        "candidate_states": resumed.result.candidate_states,
        "verified": resumed.result.verified_candidate_ids,
    }

    # A tiny complete search permits testing retained seed freshness with an
    # actual verified member. Each stage has multiple independently seeded chunks.
    fresh_binding = make_binding(config=execution_config(chunk_max_results=64))
    search = _config(grid=(3,), epsilons=((3, (1.1, 1.3, 1.5)),), evidence_rungs=(1,))
    completed = run_typed_hmc_candidate_set(fresh_binding.typed_adapter, search)
    if completed.result.verified_candidate_ids:
        member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=completed.result,
            candidate_id=completed.result.verified_candidate_ids[0], retained_binding=fresh_binding)
        chunk_seed = tuple(next(e for e in fresh_binding._evidence.values() if len(e["chunks"]) > 1)["chunks"][1]["seed"])
        try:
            retained = member.run(num_results=4, seed=chunk_seed,
                output_dir=destination / "reused-tuning-chunk-seed")
            rows["retained_chunk_seed_reuse"] = {"accepted": True, "seed": chunk_seed,
                "archive": retained["archive_path"]}
        except Exception as exc:
            rows["retained_chunk_seed_reuse"] = {"accepted": False, "seed": chunk_seed,
                "exception_type": type(exc).__name__, "message": str(exc)}
    else:
        rows["retained_chunk_seed_reuse"] = {"not_checked": "no verified fixture member"}

    payload = {"baseline": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip(),
        "command": sys.argv, "environment": {k: os.environ.get(k) for k in
            ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_TEST_DEVICE_SCOPE")},
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "tfp": tfp.__version__},
        "role": "CPU mechanics diagnostic; GPU intentionally hidden; no ranking or posterior evidence",
        "synthetic_seeds": {"role_trace": [20260915, 410], "Gaussian_binding_base": [20260914, 11]},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds_after_imports": time.monotonic() - started, "cases": rows}
    (destination / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""Explicit q20 warm-start repairs; sanity checks never certify training."""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from bayesfilter.inference.q20_production_config import (
    digest, frozen_scope_hash, scoped_seed, validate_protocol, write_json,
)


def clipping_envelope(saved, gradients):
    recent = [r["gradient_norm"] for r in saved["history"] if r["status"] == "accepted"][-128:]
    norms = [math.sqrt(math.fsum(float(x)**2 for x in row)) for row in gradients]
    if not recent or not norms or any(not math.isfinite(x) or x <= 0 for x in recent+norms):
        raise ValueError("clipping envelope requires finite positive observed norms")
    return {"cap": max(recent+norms), "training_batches": len(recent), "diagnostic_batches": len(norms),
        "rule": "maximum_of_declared_saved_norms", "role": "local_envelope_hypothesis_not_tail_bound"}


def repair_session(config, bridge, saved, *, scope, candidate):
    """Preserve the map and original slots; add only paired identity stages.

    A changed recipe is a new scope. New Adam slots are zero at the preserved
    global iteration; this explicit warm start is not an optimizer equivalence.
    """
    import tensorflow as tf
    from bayesfilter.inference.q20_production_training import training_config
    from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport
    from bayesfilter.inference.neutra_training_protocol import TrainingSession, _variables, _checked_assignments
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        restore_trainable_transport_checkpoint, ReferenceAffineTransport,
        IndependentTemperedReverseKLTrainer, prepare_transport_initialization,
    )
    if digest({k: v for k, v in saved.items() if k != "state_hash"}) != saved["state_hash"]:
        raise ValueError("repair input state checksum mismatch")
    old = restore_trainable_transport_checkpoint(saved["map"], expected_context={
        "bridge_signature": bridge.signature, "target_signature": bridge.target_signature})
    if not isinstance(old, ReferenceAffineTransport):
        raise ValueError("repair requires the saved prior-affine IAF")
    cfg = training_config(config, candidate)
    before, after = dict(old.inner.config.manifest_payload()), dict(cfg.manifest_payload())
    old_depth = before["stages"]
    for key in ("stages", "gradient_clip_norm"):
        before.pop(key); after.pop(key)
    if before != after or cfg.stages < old_depth or (cfg.stages-old_depth) % 2:
        raise ValueError("repair changes more than clipping or paired identity depth")
    inner = WeightedDenseIAFTransport(cfg)
    old_variables = old.inner.trainable_variables
    for variable, previous in zip(inner.trainable_variables[:len(old_variables)], old_variables, strict=True):
        if variable.shape != previous.shape:
            raise ValueError("repair parameter prefix differs")
        variable.assign(previous)
    for stage in inner.stages[old_depth:]:
        stage.weights[-1].assign(tf.zeros_like(stage.weights[-1]))
        stage.biases[-1].assign(tf.zeros_like(stage.biases[-1]))
    transport = ReferenceAffineTransport(inner, center=old.center, scale=old.scale, component_id=candidate["id"])
    prepared = prepare_transport_initialization(transport, bridge, component_id=candidate["id"],
        beta=saved["map"]["beta"], seed=scoped_seed(config, "repair-preflight", candidate["id"]),
        batch_size=config["training"]["batch_size"], repair_scales=(1.,))
    trainer = IndependentTemperedReverseKLTrainer(cfg, bridge, beta=saved["map"]["beta"],
        component_id=candidate["id"], batch_size=config["training"]["batch_size"], prepared_initialization=prepared)
    slots = _variables(trainer.optimizer)
    if len(saved["optimizer"]) != 2+2*len(old_variables) or len(slots) != 2+2*len(trainer.variables):
        raise ValueError("unrecognized Adam slot layout")
    for variable, value in _checked_assignments(slots[:len(saved["optimizer"])], saved["optimizer"]):
        variable.assign(value)
    trainer.step.assign(saved["iteration"])
    if int(trainer.optimizer.iterations.numpy()) != saved["iteration"]:
        raise ValueError("repair Adam iteration differs")
    return TrainingSession(trainer=trainer, scope=scope, root_seed=saved["root_seed"],
        rng_index=saved["rng_index"], level_updates=saved["level_updates"], history=saved["history"],
        parent_hash=saved["state_hash"]), old


def run_repair_arm(config, bridge, root, *, request, memory_policy):
    import tensorflow as tf
    from bayesfilter.inference.neutra_training_protocol import (
        export_weighted_transport, transport_parity, assess_training_rung, paired_loss_statistics,
    )
    from bayesfilter.inference.q20_production_training import scope_for, source_snapshot
    from bayesfilter.inference.q20_training_validation import FrozenLossCache, ValidationBudgetExhausted
    from bayesfilter.inference.neutra_post_training import assess_post_training, post_training_settings
    from bayesfilter.inference.neutra_weighted_training import WeightedNeuTraTrainingError
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    deadline = began+request["cooperative_seconds"]
    raw = Path(request["checkpoint"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != request["checkpoint_sha256"]:
        raise ValueError("repair input checkpoint changed")
    old_cohort = json.loads(raw)
    if digest({k: v for k, v in old_cohort.items() if k != "checkpoint_hash"}) != old_cohort["checkpoint_hash"]:
        raise ValueError("repair input cohort checksum differs")
    item = old_cohort["cohort"][request["candidate"]]
    saved = item["session"]
    candidate = saved["scope"]["candidate"]
    sources = source_snapshot()
    scope = scope_for(config, bridge, candidate, sources=sources, memory_policy=memory_policy)
    session, old_transport = repair_session(config, bridge, saved, scope=scope, candidate=candidate)
    initial = session.checkpoint()
    v = config["validation"]
    latent = tf.random.stateless_normal([v["reliability_rows"], bridge.parameter_dim],
        scoped_seed(config, "repair-map-parity"), dtype=tf.float64)
    _, old_frozen = export_weighted_transport(old_transport,
        target_signature=bridge.fixed_beta_adapter(1.).adapter_signature(),
        training_state_hash=saved["state_hash"], transport_id="repair-parent")
    parity = transport_parity(session.trainer.transport, old_frozen, latent,
                              rtol=v["reliability_rtol"], atol=v["reliability_atol"])
    if not parity["passed"]:
        raise ValueError("repair changed the starting map")
    receipt = {"parent_state_hash": saved["state_hash"], "map_equivalence": parity,
        "old_depth": old_transport.inner.config.stages, "new_depth": config["training"]["stages"],
        "old_clip": old_transport.inner.config.gradient_clip_norm,
        "new_clip": config["training"]["gradient_clip_norm"], "batch_size": session.trainer.batch_size,
        "lifetime_updates_before": saved["iteration"], "rng_index_before": saved["rng_index"],
        "optimizer_rule": "preserve_prefix_slots_and_iteration_zero_new_slots",
        "exact_optimizer_continuation": config["training"]["stages"] == old_transport.inner.config.stages
            and config["training"]["gradient_clip_norm"] == old_transport.inner.config.gradient_clip_norm
            and config["training"]["batch_size"] == saved["batch_size"],
        "sources": sources, "parent_sources": saved["scope"]["sources"],
        "map_parity_role": "engineering_only_not_training_or_posterior_qualification"}
    write_json(root/"migration.json", receipt)
    sanity = request.get("sanity_only", False)
    updates = 8 if sanity else request["updates"]
    count = 128 if sanity else v["bank_sizes"][0]
    seed = scoped_seed(config, "repair-common-validation")
    # Fixed evaluation batching preserves identical latent banks across arms.
    cache = FrozenLossCache(root/"validation-cache", bridge=bridge, scope=scope,
                           batch_size=32 if config["role"] != "smoke" else 8, jit_compile=config["jit_compile"])
    before = cache.evaluate(initial["map"], count, seed)
    start_history = len(session.history)
    checkpoint_path = None
    completed = 0
    while completed < updates:
        n = min(config["training"]["checkpoint_every"], updates-completed)
        try:
            status = session.advance(n, log_path=root/"updates.jsonl", budget_check=lambda: time.monotonic() < deadline)
        except (WeightedNeuTraTrainingError, tf.errors.InvalidArgumentError) as error:
            rejected = root/"rejected-state.json"
            write_json(rejected, session.checkpoint())
            return {"status": "candidate_rejected", "arm": request["arm"], "checkpoint": str(rejected),
                    "reason": str(error), "failure_class": "candidate_numerical_veto",
                    "clip_guard_screen": "not_assessed", "production_qualified": False}
        current = session.checkpoint()
        completed = len(session.history)-start_history
        checkpoint_path = root/f"state-{completed:05d}.json"
        write_json(checkpoint_path, current)
        if status == "budget_exhausted":
            return {"status": "budget_paused", "checkpoint": str(checkpoint_path), "updates": completed,
                    "production_qualified": False}
    after = cache.evaluate(current["map"], count, seed)
    increment = paired_loss_statistics(before, after, multiplier=v["normal_interval_multiplier"])
    rows = session.history[start_history:]
    clipped = sum(r["clipping_applied"] for r in rows)/len(rows)
    payload, frozen = export_weighted_transport(session.trainer.transport,
        target_signature=bridge.fixed_beta_adapter(1.).adapter_signature(),
        training_state_hash=current["state_hash"], transport_id=request["arm"])
    export_parity = transport_parity(session.trainer.transport, frozen, latent,
        rtol=v["reliability_rtol"], atol=v["reliability_atol"])
    if not export_parity["passed"]:
        raise ValueError("repaired export failed parity")
    baseline_state = initial
    if not sanity:
        baseline_session, _ = repair_session(config, bridge, item["baseline"], scope=scope, candidate=candidate)
        baseline_state = baseline_session.checkpoint()
    baseline = cache.evaluate(baseline_state["map"], count, seed)
    decision = assess_training_rung(baseline=paired_loss_statistics(baseline, after,
        multiplier=v["normal_interval_multiplier"]), increment=increment, reliability=True,
        prior_plateaus=0, at_cap=False, minimum_improvement=v["minimum_improvement"],
        maximum_half_width=v["maximum_half_width"], plateau_comparisons=v["plateau_comparisons"],
        minimum_updates_met=not sanity and session.level_updates >= config["training"]["cohort_min_updates"])
    fresh = None
    if not sanity:
        fresh_seed = scoped_seed(config, "repair-untouched-final-validation")
        a = cache.evaluate(initial["map"], count, fresh_seed)
        b = cache.evaluate(current["map"], count, fresh_seed)
        fresh = paired_loss_statistics(a, b, multiplier=v["normal_interval_multiplier"])
    assessment = {"decision": decision, "map_reliability": export_parity, "increment": increment,
        "untouched_final_increment": fresh, "calibration_only": sanity, "updates": session.level_updates,
        "lifetime_updates": current["iteration"], "beta": 1., "seed": list(seed),
        "previous_map_hash": initial["map"]["transport_state_hash"],
        "current_map_hash": current["map"]["transport_state_hash"]}
    try:
        probe = cache.post_training_probe(current["map"], **post_training_settings(config, sanity_only=sanity),
            seed=scoped_seed(config, "post-training-geometry", candidate["id"], 1.),
            budget_check=lambda: time.monotonic() < deadline)
    except ValidationBudgetExhausted:
        return {"status": "budget_paused", "checkpoint": str(checkpoint_path),
            "updates": completed, "next_action": "complete_post_training_assessment",
            "production_qualified": False}
    report = assess_post_training(state=current, decision=decision, parity=export_parity,
                                  probe=probe, history=rows, sanity_only=sanity)
    assessment["post_training"] = report
    if not report["numerical_check_passed"]:
        decision.update(status="numerically_invalid", plateaus=0,
                        hmc_trial_eligible=False, development_eligible=False)
    export_path = root/"transport.json"
    write_json(export_path, {"frozen_transport": payload, "assessment": assessment,
        "candidate": candidate, "config_hash": digest(config), "role": config["role"],
        "frozen_scope_hash": frozen_scope_hash(config), "sources": sources,
        "training_checkpoint_hash": current["state_hash"], "sanity_only": sanity})
    cohort = {"schema": "bayesfilter.q20.training_cohort.v1", "config_hash": digest(config),
        "sources": sources, "status": "repair_tranche_complete", "calibration_only": sanity,
        "elapsed_seconds": time.monotonic()-began, "total_elapsed_seconds": time.monotonic()-began,
        "cohort": {candidate["id"]: {"session": current, "baseline": baseline_state, "previous": current,
            "last_distinct_previous": initial, "last_distinct_assessment": assessment,
            "assessments": [assessment], "historical_assessments": item.get("assessments", []),
            "plateaus": decision["plateaus"], "status": decision["status"],
            "exports": {} if sanity else {"1.0": str(export_path)}}}}
    cohort_path = root/"cohort-00000.json"
    write_json(cohort_path, {**cohort, "checkpoint_hash": digest(cohort)})
    return {"status": ("candidate_rejected" if not report["numerical_check_passed"] else
                       "sanity_pilot_complete" if sanity else "repair_tranche_complete"),
        "arm": request["arm"], "checkpoint": str(cohort_path), "export": str(export_path),
        "updates": completed, "target_training_rows": completed*session.trainer.batch_size,
        "lifetime_updates": current["iteration"], "assessment": assessment,
        "scale_diagnostics": probe["scale_diagnostics"], "next_action": report["next_action"],
        "clipped_fraction": clipped, "clip_guard_screen": ("not_assessed" if not report["numerical_check_passed"]
            else "failed_check" if clipped > .5 else "no_issue_detected"),
        "training_seconds": sum(r["wall_seconds"] for r in rows), "training_traces": session.trainer._compiled_train_step.experimental_get_tracing_count(),
        "sample_wise_target_fallback": False, "jit_compile": config["jit_compile"],
        "training_device": session.trainer.variables[0].device,
        "calibration_complete": False, "statistically_supported_ranking": False,
        "production_qualified": False}


def execute_repair_master(request, *, repo, root):
    """Run sanity first, then a bounded four-arm queue within inherited funds."""
    from bayesfilter.inference.q20_campaign_runtime import Campaign, atomic_json
    from bayesfilter.inference.q20_parallel_training import execute_queue
    from bayesfilter.inference.q20_gpu_runtime import probe_gpu_inventory
    parent_path = Path(request["previous_campaign"])/"campaign.json"
    parent = json.loads(parent_path.read_text())
    if any(a["status"] == "running" for a in parent["attempts"]):
        raise ValueError("predecessor has unsettled work")
    config = parent["config"]
    allowance = {"campaign_remaining_seconds": parent["campaign_limit"]-parent["spent_seconds"],
        "diagnostic_remaining_seconds": parent["diagnostic_limit"]-parent["diagnostic_spent_seconds"],
        "source_ledger": str(parent_path), "source_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "scope": "aggregate_worker_seconds_including_parallel_workers; no_allocation_increase"}
    campaign = Campaign(root, repo=repo, config=config, allowance=allowance)
    predecessor = Campaign(Path(request["previous_campaign"]), repo=request["previous_source_root"], config=config)
    with predecessor.locked():
        successor = predecessor.state.get("successor_campaign")
        if successor is not None and successor != str(campaign.root):
            raise ValueError("predecessor allowance already transferred to another campaign")
        if not campaign.path.exists() and (
                predecessor.remaining() != allowance["campaign_remaining_seconds"] or
                predecessor.remaining(True) != allowance["diagnostic_remaining_seconds"]):
            raise ValueError("predecessor budget changed during recovery")
        predecessor.state.update(status="SUPERSEDED_BY_TRAINING_REPAIR", successor_campaign=str(campaign.root))
        predecessor.save()
    checkpoint = Path(request["checkpoint"])
    saved = json.loads(checkpoint.read_text())["cohort"][request["candidate"]]["session"]
    gradient_path = Path(request["gradients"])
    envelope = clipping_envelope(saved, json.loads(gradient_path.read_text())["terms"]["total"])
    arms = [("control", 2, 32, 10., 1024), ("clip", 2, 32, envelope["cap"], 1024),
            ("depth", 4, 32, envelope["cap"], 1024), ("batch128", 2, 128, envelope["cap"], 256)]
    with campaign.locked():
        if (campaign.root/"result.json").exists():
            return json.loads((campaign.root/"result.json").read_text())
        inventory = probe_gpu_inventory()
        devices = [g["index"] for g in inventory["gpu_inventory"] if not g["compute_busy"]]
        atomic_json(campaign.root/"gpu-inventory.json", inventory)
        atomic_json(campaign.root/"clipping-envelope.json", {**envelope,
            "gradient_source": str(gradient_path), "gradient_sha256": hashlib.sha256(gradient_path.read_bytes()).hexdigest()})
        command = [sys.executable, "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py", "worker",
                   "--request", "{attempt}/request.json", "--output-dir", "{attempt}/worker"]
        all_results = {}
        for sanity in (True, False):
            jobs = []
            for arm, depth, batch, clip, updates in arms:
                if sanity and arm == "control":
                    continue  # Existing unchanged-map canary already passed.
                if not sanity and arm != "control" and all_results.get("sanity-"+arm, {}).get("clip_guard_screen") != "no_issue_detected":
                    continue
                name = ("sanity-" if sanity else "continue-")+arm
                arm_config = copy.deepcopy(config)
                arm_config["training"].update(stages=depth, batch_size=batch, gradient_clip_norm=clip)
                validate_protocol(arm_config)
                cap = 200. if sanity else 5400.
                if datetime.fromisoformat(request["deadline"]).timestamp()-time.time() < cap:
                    raise ValueError("remaining wall deadline cannot cover another repair worker")
                campaign.state.setdefault("stage_limits", {})[name] = cap
                remaining = campaign.stage_remaining(name)
                prior = [a for a in campaign.state["attempts"] if a["stage"] == name]
                for attempt in prior:
                    path = Path(attempt["directory"])/"worker/worker-result.json"
                    if attempt["status"] == "completed" and path.exists():
                        all_results[name] = json.loads(path.read_text())["result"]
                if name in all_results:
                    continue
                if len(prior) >= 2 or remaining <= 2*config["execution"]["termination_grace_seconds"]:
                    raise ValueError("repair attempt allowance exhausted: "+name)
                jobs.append({"stage": name, "command": command, "cap_seconds": remaining, "diagnostic": sanity,
                    "request": {"stage": "repair-training", "arm": arm, "config": arm_config,
                        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "candidate": request["candidate"], "updates": updates, "sanity_only": sanity,
                        "plan_file": request["plan_file"]}})
            campaign.save()
            attempts = execute_queue(campaign, jobs, devices)
            for attempt in attempts:
                path = Path(attempt["directory"])/"worker/worker-result.json"
                if attempt["status"] != "completed" or not path.exists():
                    raise RuntimeError("repair worker incomplete: "+str(attempt["directory"]))
                worker = json.loads(path.read_text())
                if not worker["completed"] or worker.get("status") == "budget_paused":
                    raise RuntimeError("repair worker requires continuation: "+str(path))
                all_results[attempt["stage"]] = worker["result"]
            atomic_json(campaign.root/("sanity-results.json" if sanity else "training-results.json"), all_results)
        result = {"status": "TRAINING_REPAIR_TRANCHE_COMPLETE", "arms": all_results,
            "remaining_campaign_seconds": campaign.remaining(), "remaining_diagnostic_seconds": campaign.remaining(True),
            "aggregate_worker_seconds": campaign.state["spent_seconds"], "production_qualified": False,
            "next_actions": {name: item.get("next_action", "repair_incomplete_training")
                             for name, item in all_results.items() if name.startswith("continue-")},
            "next_action": "follow_per_map_post_training_assessments"}
        campaign.state["status"] = result["status"]
        campaign.save()
        atomic_json(campaign.root/"result.json", result)
        return result

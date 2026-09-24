"""q20 cohort scheduler and measured training protocol; no six-update default."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import tensorflow as tf

from bayesfilter.inference.q20_production_config import digest, frozen_scope_hash, scoped_seed, training_cohort, method_betas, write_json
from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig
from bayesfilter.inference.neutra_training_protocol import (
    TrainingSession, HeldoutLoss, paired_loss_statistics, assess_training_rung,
    export_weighted_transport, transport_parity,
)
from bayesfilter.inference.q20_training_validation import FrozenLossCache, ValidationBudgetExhausted
from bayesfilter.inference.tempered_transport_ensemble_tf import (
    IndependentTemperedReverseKLTrainer, prepare_transport_initialization,
)


def source_snapshot():
    """Ordinary checksums in addition to Git; includes uncommitted source."""
    from bayesfilter.inference.q20_campaign_runtime import source_snapshot as snapshot
    return snapshot(Path(__file__).resolve().parents[2])


def training_config(config, candidate, *, batch_size=None):
    t = config["training"]
    return WeightedNeuTraConfig(dimension=config["target"]["dimension"],
        hidden_layers=(candidate["width"], candidate["width"]), stages=t["stages"],
        activation=t["activation"], s_max=t["s_max"], permutation_policy="full_reverse",
        initialization_scale=t["initialization_scale"],
        initialization_seed=scoped_seed(config, "initialization", candidate["id"]),
        learning_rate=candidate["learning_rate"], beta1=t["beta1"], beta2=t["beta2"],
        epsilon=t["epsilon"], gradient_clip_norm=t["gradient_clip_norm"],
        jit_compile=config["jit_compile"])


def scope_for(config, bridge, candidate, *, sources, memory_policy):
    return json.loads(json.dumps({"config_hash": digest(config), "candidate": candidate,
        "data_identity": bridge.target_signature, "bridge_signature": bridge.signature,
        "dtype": "float64", "backend": "tensorflow", "jit_compile": config["jit_compile"],
        "training_seed_derivation": {"root": config["seed"], "method": "sha256_labels_then_stateless_fold_in"},
        "validation_bank_ids": [digest([config["seed"], "validation", candidate["id"], beta])
                                for beta in config["training"]["betas"]],
        "sources": sources, "memory_policy": memory_policy, "tensorflow": tf.__version__,
        "role": config["role"], "sample_wise_target_fallback": False,
        "target_backend": "batch_native_tensorflow_value_score"}))


def new_session(config, bridge, candidate, *, scope, batch_size=None):
    cfg = training_config(config, candidate)
    batch = batch_size or config["training"]["batch_size"]
    inner = WeightedDenseIAFTransport(cfg)
    # Exact prior-affine beta-zero initialization. Hidden weights remain random
    # to break symmetry; zero output layers make every stage exactly identity.
    for stage in inner.stages:
        stage.weights[-1].assign(tf.zeros_like(stage.weights[-1]))
        stage.biases[-1].assign(tf.zeros_like(stage.biases[-1]))
    prepared = prepare_transport_initialization(inner, bridge,
        component_id=candidate["id"], seed=scoped_seed(config, "preflight", candidate["id"], 0.),
        batch_size=batch, beta=0., repair_scales=(1.,),
        reference_center=bridge.prior_center,
        reference_scale=tf.sqrt(tf.convert_to_tensor(bridge.prior_variance, tf.float64)))
    trainer = IndependentTemperedReverseKLTrainer(cfg, bridge, beta=0.,
        component_id=candidate["id"], batch_size=batch, prepared_initialization=prepared)
    return TrainingSession(trainer=trainer, scope=scope,
        root_seed=scoped_seed(config, "train", candidate["id"], 0.))


def _evaluate_rung(config, bridge, candidate, session, baseline_state, previous_state, previous_plateaus,
                   *, cache, budget_check=None, calibration_only=False):
    from bayesfilter.inference.neutra_post_training import assess_post_training, post_training_settings
    v, t = config["validation"], config["training"]
    beta = session.trainer.beta
    state = session.checkpoint()
    maps = (baseline_state["map"], previous_state["map"], state["map"])
    evaluated_before, reused_before = cache.evaluated_rows, cache.reused_rows
    seed = scoped_seed(config, "validation", candidate["id"], beta)
    looks = []
    minimum_updates_met = not calibration_only and session.level_updates >= t["cohort_min_updates"]
    # Resume may retain the historical expansion ladder. Only its initial bank
    # is funded: unresolved learning calls for training, not plateau precision.
    for count in v["bank_sizes"][:1]:
        start, before, after = [cache.evaluate(map_state, count, seed, budget_check=budget_check)
                                for map_state in maps]
        first = paired_loss_statistics(start, after, multiplier=v["normal_interval_multiplier"])
        distinct = previous_state["map"]["transport_state_hash"] != state["map"]["transport_state_hash"]
        increment = (paired_loss_statistics(before, after, multiplier=v["normal_interval_multiplier"])
                     if distinct else None)
        decision = assess_training_rung(baseline=first, increment=increment, reliability=True,
            prior_plateaus=previous_plateaus, at_cap=session.level_updates >= t["rungs"][-1],
            minimum_improvement=v["minimum_improvement"], maximum_half_width=v["maximum_half_width"],
            plateau_comparisons=v["plateau_comparisons"], minimum_updates_met=minimum_updates_met)
        looks.append({"baseline": first, "increment": increment,
            "decision": decision, "estimated_rows_for_half_width": {
                label: math.ceil(count * (stats["half_width"] / v["maximum_half_width"])**2)
                for label, stats in (("baseline", first), ("increment", increment)) if stats is not None}})
        if decision["validation_resolved"]:
            break
    target = bridge.fixed_beta_adapter(beta)
    payload, frozen = export_weighted_transport(session.trainer.transport,
        target_signature=target.adapter_signature(), training_state_hash=state["state_hash"],
        transport_id=f"{candidate['id']}-beta{beta:g}-u{session.level_updates}")
    with tf.device("/CPU:0"):
        latent = tf.random.stateless_normal([v["reliability_rows"], bridge.parameter_dim],
            scoped_seed(config, "map-reliability", candidate["id"], beta), dtype=tf.float64)
    parity = transport_parity(session.trainer.transport, frozen, latent,
        rtol=v["reliability_rtol"], atol=v["reliability_atol"])
    if not parity["passed"]:
        decision = {**decision, "status": "numerically_invalid", "plateaus": 0,
                    "hmc_trial_eligible": False, "development_eligible": False}
    probe = cache.post_training_probe(state["map"], **post_training_settings(config, sanity_only=calibration_only),
        seed=scoped_seed(config, "post-training-geometry", candidate["id"], beta),
        budget_check=budget_check)
    report = assess_post_training(state=state, decision=decision, parity=parity, probe=probe,
        history=session.history[len(previous_state["history"]):], sanity_only=calibration_only)
    if not report["numerical_check_passed"]:
        decision = {**decision, "status": "numerically_invalid", "plateaus": 0,
                    "hmc_trial_eligible": False, "development_eligible": False}
    return {"decision": decision, "looks": looks, "map_reliability": parity,
            "post_training": report,
            "calibration_only": calibration_only,
            "assessment_role": "sanity_pilot" if calibration_only else "training_development_screen",
            "previous_map_hash": previous_state["map"]["transport_state_hash"],
            "current_map_hash": state["map"]["transport_state_hash"],
            "validation_evaluated_rows": cache.evaluated_rows - evaluated_before,
            "validation_reused_rows": cache.reused_rows - reused_before,
            "validation_stop": "bounded_bank_completed",
            "validation_bank_cap": v["bank_sizes"][0],
            "seed": list(seed), "updates": session.level_updates, "beta": beta}, payload


def run_training_cohort(config, bridge, root, *, memory_policy, max_seconds, resume=None,
                        stop_after_rung=None, calibration_only=False, method=None,
                        stop_when_trial_ready=False):
    """Round-robin complete cohorts; every seed reaches the floor before pruning.

    Resume consumes the previous cohort checkpoint in a fresh output directory.
    The checkpoint owns all baseline/previous maps and optimizer state, so no
    future validation bank or warm-up is substituted for missing history.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started, sources = time.monotonic(), source_snapshot()
    candidates = training_cohort(config, method=method)
    if calibration_only:
        candidates = [candidate for candidate in candidates if candidate["root"] == config["training"]["roots"][0]]
    cohort, live_sessions = {}, {}
    caches = {}
    cache_root = root / "validation-cache"
    old = None
    if resume is not None:
        from bayesfilter.inference.q20_training_resume import read_training_checkpoint
        old = read_training_checkpoint(resume, config, sources=sources)
        cohort = old["cohort"]
        from bayesfilter.inference.neutra_post_training import POST_TRAINING_SCHEMA
        for item in cohort.values():
            beta = item["session"]["map"]["beta"]
            missing = [a for a in item["assessments"] if a["beta"] == beta
                       and a.get("post_training", {}).get("schema") != POST_TRAINING_SCHEMA]
            if missing:
                # Recheck the current immutable endpoint, including at its cap.
                # Preserve old evidence and any pre-existing training veto.
                item.setdefault("historical_assessments", []).extend(missing)
                item["reassessment_requires_update"] = item.get("reassessment_requires_update", False) or any(
                    a["updates"] == item["session"]["level_updates"] and
                    a["decision"]["status"] in {"deterioration_repair_trigger", "numerically_invalid"}
                    for a in missing)
                item["assessments"] = [a for a in item["assessments"] if a not in missing]
                item["exports"].pop(str(beta), None)
        if old.get("calibration_only") and not calibration_only:
            for item in cohort.values():
                beta = item["session"]["map"]["beta"]
                updates = item["session"]["level_updates"]
                latest = [a for a in item["assessments"] if a["beta"] == beta and a["updates"] == updates]
                item["reassessment_requires_update"] = item.get("reassessment_requires_update", False) or any(
                    a["decision"]["status"] in {"deterioration_repair_trigger", "numerically_invalid"}
                    for a in latest)
                # Calibration is never admission, even when its endpoint meets
                # the full cohort floor. Reassess cached losses before advancing.
                item["assessments"] = [a for a in item["assessments"] if a not in latest]
        previous_cache = Path(resume).parent / "validation-cache"
        if previous_cache.exists():
            shutil.copytree(previous_cache, cache_root)
    for candidate in candidates:
        if candidate["id"] in cohort:
            continue
        scope = scope_for(config, bridge, candidate, sources=sources, memory_policy=memory_policy)
        if time.monotonic()-started >= max_seconds:
            break
        session = new_session(config, bridge, candidate, scope=scope)
        live_sessions[candidate["id"]] = session
        cohort[candidate["id"]] = {"session": session.checkpoint(), "assessments": [],
            "baseline": None, "previous": None, "plateaus": 0,
            "status": "pending", "exports": {}}

    def save(status):
        state = {"schema": "bayesfilter.q20.training_cohort.v1", "config_hash": digest(config),
                 "sources": sources, "cohort": cohort, "status": status,
                 "calibration_only": calibration_only, "method": method,
                 "elapsed_seconds": time.monotonic()-started,
                 "previous_elapsed_seconds": (old or {}).get("total_elapsed_seconds", 0.)}
        state["total_elapsed_seconds"] = state["elapsed_seconds"] + state["previous_elapsed_seconds"]
        if (old or {}).get("source_import"):
            state["source_import"] = old["source_import"]
        version = len(list(root.glob("cohort-*.json")))
        path = root / f"cohort-{version:05d}.json"
        write_json(path, {**state, "checkpoint_hash": digest(state)})
        return {"status": status, "checkpoint": str(path), "elapsed_seconds": state["elapsed_seconds"],
                "cohort_complete": status == "complete", "calibration_complete": False,
                "sanity_pilot_complete": status == "sanity_pilot_complete",
                "method": method, "method_complete": status in {"method_complete", "paused_at_requested_rung"},
                "production_qualified": False}

    if any(candidate["id"] not in cohort for candidate in candidates):
        return save("partial_initialization_budget")
    save("initialized")
    for beta in config["training"]["betas"][1:]:
        for rung in config["training"]["rungs"]:
            if stop_after_rung is not None and rung > stop_after_rung:
                continue
            if calibration_only and rung != config["training"]["rungs"][0]:
                continue
            for candidate in candidates:
                if candidate["schedule"] == "direct" and beta != 1.:
                    continue
                if calibration_only and candidate["schedule"] == "continuation" and beta != config["training"]["betas"][1]:
                    continue
                item = cohort[candidate["id"]]
                stored_beta = item["session"]["map"]["beta"]
                if item["status"] == "failed" or stored_beta > beta:
                    continue
                if stored_beta == beta and (item["session"]["level_updates"] > rung or
                        (item.get("reassessment_requires_update", False) and item["session"]["level_updates"] == rung)):
                    continue
                if stored_beta == beta and any(a["beta"] == beta and a["updates"] >= rung
                                               for a in item["assessments"]):
                    continue
                if time.monotonic()-started >= max_seconds:
                    return save("partial_budget")
                # A reviewed coordinator import preserves original numerical
                # state/cache identities, with current sources recorded above.
                origin = item["session"]["scope"]["sources"]
                scope = scope_for(config, bridge, candidate, sources=origin, memory_policy=memory_policy)
                session = None
                try:
                    session = live_sessions.get(candidate["id"])
                    if session is None:
                        session = TrainingSession.restore(item["session"], bridge=bridge,
                            config=training_config(config, candidate), expected_scope=scope,
                            preflight_seed=scoped_seed(config, "preflight", candidate["id"], stored_beta))
                    if stored_beta < beta:
                        session = session.next_beta(beta,
                            root_seed=scoped_seed(config, "train", candidate["id"], beta),
                            preflight_seed=scoped_seed(config, "preflight", candidate["id"], beta))
                        item.update(baseline=session.checkpoint(), previous=session.checkpoint(),
                                    plateaus=0, status="training")
                    live_sessions[candidate["id"]] = session
                    while session.level_updates < rung:
                        count = min(config["training"]["checkpoint_every"], rung-session.level_updates)
                        status = session.advance(count, log_path=root / f"{candidate['id']}.jsonl",
                            budget_check=lambda: time.monotonic()-started < max_seconds)
                        item["session"] = session.checkpoint()
                        save("training")
                        if status == "budget_exhausted":
                            return save("partial_budget")
                    if candidate["id"] not in caches:
                        caches[candidate["id"]] = FrozenLossCache(cache_root, bridge=bridge, scope=scope,
                            batch_size=config["training"]["batch_size"], jit_compile=config["jit_compile"])
                    previous = item["previous"]
                    if previous["map"]["transport_state_hash"] == session.checkpoint()["map"]["transport_state_hash"]:
                        previous = item.get("last_distinct_previous") or previous
                    assessment, payload = _evaluate_rung(config, bridge, candidate, session,
                        item["baseline"], previous, item["plateaus"], cache=caches[candidate["id"]],
                        budget_check=lambda: time.monotonic()-started < max_seconds,
                        calibration_only=calibration_only)
                    last = item.get("last_distinct_assessment")
                    reused_pair = last is not None and all(last.get(key) == assessment[key]
                        for key in ("previous_map_hash", "current_map_hash"))
                    if reused_pair and assessment["map_reliability"]["passed"]:
                        # Reissuing an export after a coordinator import cannot
                        # count the same distinct pair as a second plateau.
                        for key in ("plateaus", "plateau_observed"):
                            if key in last["decision"]:
                                assessment["decision"][key] = last["decision"][key]
                                assessment["post_training"]["learning"][key] = last["decision"][key]
                        assessment["reused_distinct_comparison"] = True
                    item["assessments"].append(assessment)
                    if assessment["decision"].get("distinct_increment_observed"):
                        item["last_distinct_previous"] = previous
                        item["last_distinct_assessment"] = assessment
                    item.update(previous=session.checkpoint(), session=session.checkpoint(),
                        plateaus=assessment["decision"]["plateaus"], status=assessment["decision"]["status"],
                        reassessment_requires_update=False)
                    export_path = root / f"{candidate['id']}-beta{beta:g}-u{rung}.json"
                    write_json(export_path, {"frozen_transport": payload, "assessment": assessment,
                        "candidate": candidate, "config_hash": digest(config), "role": config["role"],
                        "frozen_scope_hash": frozen_scope_hash(config),
                        "sources": sources, "training_checkpoint_hash": session.checkpoint()["state_hash"]})
                    item["exports"][str(beta)] = str(export_path)
                except ValidationBudgetExhausted:
                    item["session"] = session.checkpoint()
                    return save("partial_validation_budget")
                except Exception as error:
                    if session is not None:
                        item["session"] = session.checkpoint()
                    item.update(failure={"type": type(error).__name__, "message": str(error)})
                    save("interrupted_candidate")
                    # Infrastructure and programming errors invalidate the attempt;
                    # they are not evidence that a numerical candidate failed.
                    raise
                save("rung_member_completed")
            if stop_when_trial_ready and not calibration_only and rung >= config["training"]["cohort_min_updates"]:
                # The whole declared cohort has completed this rung. For an
                # ensemble, keep only roots admitted at every temperature so far.
                required = [b for b in method_betas(config, method) if b <= beta]
                roots = set()
                for candidate in candidates:
                    exports = cohort[candidate["id"]]["exports"]
                    if not required or any(str(b) not in exports for b in required):
                        continue
                    assessments = [json.loads(Path(exports[str(b)]).read_text())["assessment"] for b in required]
                    if all(a["map_reliability"]["passed"] and (config["role"] == "smoke" or
                            a["decision"].get("hmc_trial_eligible", False)) for a in assessments):
                        roots.add(candidate["root"])
                count = 1 if method == "neutra" else config["ensemble"]["charts"]
                if len(roots) >= count:
                    break
    return save("sanity_pilot_complete" if calibration_only else
                "paused_at_requested_rung" if stop_after_rung is not None else
                "complete" if method is None else "method_complete")


def price_training(config, bridge, root, *, memory_policy, batches=None,
                   reservation_limit_seconds=None, calibration_only=False, method=None, checkpoint=None):
    from bayesfilter.inference.q20_campaign_costs import training_reservation
    from bayesfilter.inference.neutra_post_training import PostTrainingProbe, post_training_settings, PROBE_SCHEMA
    if reservation_limit_seconds is not None and (
            not math.isfinite(reservation_limit_seconds) or reservation_limit_seconds <= 0):
        raise ValueError("pricing reservation limit must be finite and positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    sources = source_snapshot()
    # LR changes do not change numerical work. Price every declared width/batch,
    # both positive beta graphs, and preserve compile versus steady work.
    for width in config["training"]["widths"]:
        candidate = {"id": f"price-w{width}", "width": width,
                     "learning_rate": config["training"]["learning_rates"][0], "root": 0}
        for batch in batches or [config["training"]["batch_size"]]:
            scope = scope_for(config, bridge, candidate, sources=sources, memory_policy=memory_policy)
            began_setup = time.monotonic()
            session = new_session(config, bridge, candidate, scope=scope, batch_size=batch)
            initialization_seconds = time.monotonic() - began_setup
            for beta in (config["training"]["betas"][1:] if method is None else method_betas(config, method)):
                began_setup = time.monotonic()
                session = session.next_beta(beta,
                    root_seed=scoped_seed(config, "pricing-train", width, batch, beta),
                    preflight_seed=scoped_seed(config, "pricing-preflight", width, batch, beta))
                setup_seconds = initialization_seconds + time.monotonic() - began_setup
                session.advance(config["budget"]["pricing_updates"], log_path=root / "updates.jsonl")
                timings = [row["wall_seconds"] for row in session.history if row["beta"] == beta]
                evaluation = HeldoutLoss(session.trainer.transport, bridge, beta,
                                         batch_size=batch, jit_compile=config["jit_compile"])
                validation_times = []
                for repeat in range(2):
                    began = time.monotonic()
                    evaluation(2*batch, scoped_seed(config, "pricing-heldout", width, batch, beta, repeat)).numpy()
                    validation_times.append(time.monotonic()-began)
                probe_settings = post_training_settings(config)
                probe = PostTrainingProbe(session.trainer.transport, bridge, beta,
                    **probe_settings, jit_compile=config["jit_compile"])
                z = probe.latent_bank(scoped_seed(config, "pricing-geometry", width, batch, beta))
                geometry_times = []
                for repeat in range(2):
                    began = time.monotonic()
                    # Timing only; repeated first block cannot certify a full bank.
                    probe.batch(z[:probe.batch_size])
                    geometry_times.append(time.monotonic()-began)
                began = time.monotonic()
                synthetic_scale_shape = [probe.rows//probe.batch_size, config["training"]["stages"], bridge.parameter_dim]
                summary = probe.summary_graph(tf.zeros([probe.rows, bridge.parameter_dim], tf.float64),
                    tf.zeros([probe.rows, bridge.parameter_dim], tf.float64), tf.zeros([probe.rows], tf.float64),
                    tf.zeros(synthetic_scale_shape, tf.float64), tf.zeros(synthetic_scale_shape, tf.float64))
                summary["score_residual_rms"].numpy()
                summary_seconds = time.monotonic()-began
                rows.append({"width": width, "batch_size": batch, "beta": beta,
                             "first_update_seconds": timings[0], "steady_update_seconds": max(timings[1:] or timings),
                             "heldout_first_seconds": validation_times[0],
                             "heldout_seconds_per_batch": validation_times[1]/2,
                             "heldout_first_batches": 2,
                             "setup_seconds": setup_seconds,
                             "post_training_probe_schema": PROBE_SCHEMA,
                             "post_training_points": probe.rows, "post_training_batch_size": probe.batch_size,
                             "post_training_first_batch_seconds": geometry_times[0],
                             "post_training_steady_batch_seconds": geometry_times[1],
                             "post_training_summary_seconds": summary_seconds,
                             "post_training_pricing_role": "two_timed_blocks_and_synthetic_summary_not_verification",
                             "training_device": session.trainer.variables[0].device,
                             "target_backend": "batch_native_tensorflow_value_score",
                             "jit_compile": config["jit_compile"], "sample_wise_target_fallback": False})
                write_json(root / f"price-w{width}-b{batch}-beta{beta:g}.json", rows[-1])
                reservation = training_reservation(config, rows, method=method, checkpoint=checkpoint)
                reservation_key = "calibration_seconds" if calibration_only else "minimum_cohort_seconds"
                if (reservation_limit_seconds is not None
                        and reservation[reservation_key] > reservation_limit_seconds):
                    result = {"config_hash": digest(config), "target_signature": bridge.target_signature,
                              "sources": sources, "memory_policy": memory_policy, "rows": rows, "method": method,
                              "status": "training_reservation_exceeds_allowance",
                              "reservation": reservation, "reservation_limit_seconds": reservation_limit_seconds,
                              "scalar_hmc_reference_pricing": "not_measured", "production_qualified": False}
                    write_json(root / "pricing.json", result)
                    return result
    result = {"config_hash": digest(config), "target_signature": bridge.target_signature,
              "sources": sources, "memory_policy": memory_policy, "rows": rows, "method": method,
              "status": "training_priced", "scalar_hmc_reference_pricing": "not_measured",
              "production_qualified": False}
    write_json(root / "pricing.json", result)
    return result


def training_quote(config, pricing):
    if pricing["config_hash"] != digest(config) or pricing["sources"] != source_snapshot():
        raise ValueError("pricing scope changed")
    from bayesfilter.inference.q20_campaign_costs import training_reservation
    result = training_reservation(config, pricing["rows"], method=pricing.get("method"))
    if result["missing_training_scopes"]:
        raise ValueError("pricing lacks a requested width/batch/beta scope")
    return result

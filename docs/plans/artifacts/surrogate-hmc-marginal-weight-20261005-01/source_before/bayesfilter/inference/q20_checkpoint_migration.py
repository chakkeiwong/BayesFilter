"""Checked v2-to-v3 warm starts; historical evidence never gains new admission."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from bayesfilter.inference.q20_production_config import digest, scoped_seed, validate_protocol, write_json
from bayesfilter.inference.q20_training_resume import checksum, read_training_checkpoint


def audit_migration(config, previous_config, checkpoint, previous_root, current_root):
    """Inspect compatible scientific choices and preserved sources without TF."""
    from bayesfilter.inference.q20_campaign_runtime import source_snapshot
    validate_protocol(config)
    if previous_config.get("schema") != "bayesfilter.q20.production_protocol.v2":
        raise ValueError("migration requires the historical v2 protocol")
    before, after = copy.deepcopy(previous_config), copy.deepcopy(config)
    comparison = before.pop("comparison")
    assessment = after.pop("assessment")
    if {k: v for k, v in comparison.items() if k not in {"methods", "confirmation_replicates"}} != assessment:
        raise ValueError("migration cannot change posterior assessment tolerances")
    after.pop("estimation")
    for row in (before, after):
        for key in ("schema", "provenance", "budget", "execution"):
            row.pop(key)
    backends = (before["target"].pop("principal_sqrt_backend"), after["target"].pop("principal_sqrt_backend"))
    if backends not in {("tensorflow_eigh_strict", "tensorflow_eigh_strict"),
                        ("tensorflow_eigh_strict", "tensorflow_eigh_strict_factor_cached")}:
        raise ValueError("unsupported migration backend transition")
    if before != after:
        raise ValueError("migration changes target/data, training, seeds or scientific rules")
    state = read_training_checkpoint(checkpoint, previous_config)
    if not state["cohort"]:
        raise ValueError("empty historical training cohort")
    if source_snapshot(previous_root) != state["sources"]:
        raise ValueError("preserved source directory does not match migration checkpoint")
    current = source_snapshot(current_root)
    return {"previous_checkpoint": str(Path(checkpoint).resolve()), "previous_sha256": checksum(checkpoint),
        "previous_config_hash": digest(previous_config), "config_hash": digest(config),
        "previous_source_root": str(Path(previous_root).resolve()),
        "changed_sources": sorted(p for p in state["sources"].keys() | current.keys()
                                  if state["sources"].get(p) != current.get(p)),
        "backend_transition": list(backends), "candidate_count": len(state["cohort"]),
        "state_policy": "warm_start_preserves_map_adam_rng_history; fresh_scope_training_and_assessment",
        "historical_updates": sum(i["session"]["iteration"] for i in state["cohort"].values()),
        "current_scope_update_credit": 0, "validation_cache_reused": False,
        "exact_numerical_continuation_claimed": False, "promotion_eligible": False}


def migrate_training_checkpoint(config, bridge, root, *, previous_config, checkpoint,
                                previous_root, memory_policy, reference_bridge=None):
    """Restore and validate every map before publishing a new cohort checkpoint."""
    import tensorflow as tf
    from bayesfilter.inference.q20_production_training import scope_for, training_config, source_snapshot
    from bayesfilter.inference.neutra_training_protocol import TrainingSession, export_weighted_transport, transport_parity
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        capture_trainable_transport_checkpoint, restore_trainable_transport_checkpoint,
    )
    root = Path(root)
    current_root = Path(__file__).resolve().parents[2]
    receipt = audit_migration(config, previous_config, checkpoint, previous_root, current_root)
    state = read_training_checkpoint(checkpoint, previous_config)
    if reference_bridge is None:
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        reference_bridge = make_q20_tempered_bridge(previous_config["target"]["q"],
            jit_compile=config["jit_compile"],
            principal_sqrt_backend=previous_config["target"]["principal_sqrt_backend"])
    root.mkdir(parents=True, exist_ok=False)
    sources, cohort, checks = source_snapshot(), {}, []
    v = config["validation"]

    def rebind(saved, scope):
        old_scope = saved["scope"]
        if (old_scope["bridge_signature"] != reference_bridge.signature or
                old_scope["data_identity"] != reference_bridge.target_signature):
            raise ValueError("historical checkpoint target differs from reconstructed strict target")
        transport = restore_trainable_transport_checkpoint(saved["map"], expected_context={
            "checkpoint_scope": old_scope, "bridge_signature": old_scope["bridge_signature"],
            "target_signature": old_scope["data_identity"],
            "component_id": scope["candidate"]["id"], "update_count": saved["iteration"]})
        migrated = copy.deepcopy(saved)
        migrated.update(scope=scope, parent_hash=saved["state_hash"], level_updates=0)
        migrated["map"] = capture_trainable_transport_checkpoint(transport,
            component_id=scope["candidate"]["id"], beta=saved["map"]["beta"],
            bridge_signature=bridge.signature, target_signature=bridge.target_signature,
            parent_checkpoint_hash=saved["state_hash"], update_count=saved["iteration"], checkpoint_scope=scope)
        migrated.pop("state_hash")
        migrated["state_hash"] = digest(migrated)
        return migrated

    for name, old in state["cohort"].items():
        saved = old["session"]
        candidate, beta = saved["scope"]["candidate"], saved["map"]["beta"]
        if beta not in config["training"]["betas"][1:] or old["baseline"] is None:
            raise ValueError("migration requires trained positive-beta state and initial baseline")
        if saved["root_seed"] != list(scoped_seed(previous_config, "train", name, beta)):
            raise ValueError("historical training random stream differs")
        scope = scope_for(config, bridge, candidate, sources=sources, memory_policy=memory_policy)
        rebound = rebind(saved, scope)
        session = TrainingSession.restore(rebound, bridge=bridge, config=training_config(config, candidate),
            expected_scope=scope, preflight_seed=scoped_seed(config, "migration-preflight", name, beta))
        current = session.checkpoint()
        for key in ("optimizer", "rng_index", "root_seed", "iteration", "history", "batch_size"):
            if current[key] != saved[key]:
                raise ValueError("migration changed numerical state: " + key)
        if any(current["map"][key] != saved["map"][key]
               for key in ("variables", "structure", "transport_state_hash")):
            raise ValueError("migration changed learned transport")
        baseline = rebind(old["baseline"], scope)
        with tf.device("/CPU:0"):
            latent = tf.random.stateless_normal([v["reliability_rows"], bridge.parameter_dim],
                scoped_seed(config, "migration-parity", name, beta), dtype=tf.float64)
        _, frozen = export_weighted_transport(session.trainer.transport,
            target_signature=bridge.fixed_beta_adapter(beta).adapter_signature(),
            training_state_hash=current["state_hash"], transport_id="migration-" + name)
        parity = transport_parity(session.trainer.transport, frozen, latent,
            rtol=v["reliability_rtol"], atol=v["reliability_atol"])
        if not parity["passed"]:
            raise ValueError("migrated map failed frozen codec parity: " + name)
        physical, _ = session.trainer.transport.forward_and_logdet(latent)
        left = reference_bridge.value_score_status(physical, tf.constant(beta, tf.float64))
        right = bridge.value_score_status(physical, tf.constant(beta, tf.float64))
        residuals = {}
        for label, a, b in zip(("value", "score"), left[:2], right[:2], strict=True):
            tf.debugging.assert_all_finite(a, "strict migration " + label)
            tf.debugging.assert_all_finite(b, "current migration " + label)
            tf.debugging.assert_near(a, b, rtol=v["reliability_rtol"], atol=v["reliability_atol"])
            residuals[label] = float(tf.reduce_max(tf.abs(a-b)).numpy())
        if left[2].keys() != right[2].keys():
            raise ValueError("strict/current target status fields differ")
        for key in left[2]:
            tf.debugging.assert_equal(left[2][key], right[2][key], message=key)
        tf.debugging.assert_equal(tf.reduce_all(right[2]["bridge_valid"]), True)
        cohort[name] = {"session": current, "baseline": baseline, "previous": current,
            "assessments": [], "plateaus": 0, "status": "warm_start_requires_assessment", "exports": {}}
        checks.append({"candidate": name, "beta": beta, "historical_updates": saved["iteration"],
            "map_adam_rng_preserved": True, "map_parity": parity, "backend_max_absolute_errors": residuals,
            "backend_status_equal": True, "current_scope_updates": current["level_updates"],
            "old_state_hash": saved["state_hash"], "new_state_hash": current["state_hash"]})
        write_json(root / (name + "-migration.json"), checks[-1])
    receipt.update(checks=checks, migration_passed=True)
    body = {"schema": "bayesfilter.q20.training_cohort.v1", "config_hash": digest(config),
        "sources": sources, "source_import": receipt, "cohort": cohort,
        "status": "migrated_warm_start_requires_training_assessment", "calibration_only": False,
        "elapsed_seconds": 0., "previous_elapsed_seconds": state["total_elapsed_seconds"],
        "total_elapsed_seconds": state["total_elapsed_seconds"]}
    path = root / "cohort-00000.json"
    write_json(path, {**body, "checkpoint_hash": digest(body)})
    read_training_checkpoint(path, config, sources=sources)
    return {"status": "training_migration_complete", "checkpoint": str(path), "sha256": checksum(path),
        "source_import": receipt, "production_qualified": False}

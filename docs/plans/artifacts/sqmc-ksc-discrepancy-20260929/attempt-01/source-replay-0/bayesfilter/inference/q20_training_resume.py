"""Checked q20 checkpoint import across the September 18 coordinator repair.

Original map/Adam/RNG scopes and validation identities remain unchanged. The
import records current execution sources separately from historical training.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil

from bayesfilter.inference.q20_production_config import digest, scoped_seed, training_cohort, write_json


COORDINATOR_PATHS = frozenset({
    "bayesfilter/inference/q20_campaign_runtime.py",
    "bayesfilter/inference/q20_campaign_costs.py",
    "bayesfilter/inference/q20_master_program.py",
    "bayesfilter/inference/q20_master_stages.py",
    "bayesfilter/inference/q20_master_refresh.py",
    "bayesfilter/inference/q20_training_resume.py",
    "bayesfilter/inference/q20_stage_budget.py",
    "bayesfilter/inference/q20_pricing.py",
    "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py",
})
FUNCTION_CHANGES = {
    "bayesfilter/inference/q20_production_training.py": {
        "run_training_cohort", "price_training", "training_quote", "_evaluate_rung"},
    "bayesfilter/inference/q20_production_config.py": {"protocol_template"},
    "bayesfilter/inference/q20_production_hmc.py": {
        "validate_training_export", "tune_scope", "_export_tuning_result", "sample_member", "sample_ensemble"},
    "bayesfilter/inference/q20_production_reference.py": {"run_reference"},
    "bayesfilter/inference/neutra_training_protocol.py": {"assess_training_rung"},
}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unchanged_program(path, allowed_functions):
    tree = ast.parse(Path(path).read_text())
    tree.body = [node for node in tree.body if not
                 (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed_functions)]
    return ast.dump(tree, include_attributes=False)


def check_refresh_sources(previous, previous_root, current_root):
    from bayesfilter.inference.q20_campaign_runtime import source_snapshot
    previous_root, current_root = Path(previous_root), Path(current_root)
    if source_snapshot(previous_root) != previous:
        raise ValueError("preserved source directory does not match checkpoint")
    current = source_snapshot(current_root)
    changed = sorted(p for p in previous.keys() | current.keys() if previous.get(p) != current.get(p))
    for path in changed:
        if path in COORDINATOR_PATHS:
            continue
        functions = FUNCTION_CHANGES.get(path)
        if (functions is None or path not in previous or path not in current or
                _unchanged_program(previous_root / path, functions) !=
                _unchanged_program(current_root / path, functions)):
            raise ValueError("numerical source change prevents checkpoint import: " + path)
    return current, changed


def read_training_checkpoint(path, config, *, sources=None):
    state = json.loads(Path(path).read_text())
    expected = state.pop("checkpoint_hash")
    if (digest(state) != expected or state.get("schema") != "bayesfilter.q20.training_cohort.v1"
            or state["config_hash"] != digest(config)):
        raise ValueError("cohort checkpoint configuration/checksum mismatch")
    if sources is not None and state["sources"] != sources:
        raise ValueError("cohort checkpoint source mismatch")
    candidates = {c["id"]: c for c in training_cohort(config)}
    accepted = [state["sources"], *state.get("source_import", {}).get("accepted_training_sources", [])]
    for name, item in state["cohort"].items():
        if name not in candidates:
            raise ValueError("checkpoint candidate outside protocol")
        for key in ("session", "baseline", "previous", "last_distinct_previous"):
            session = item.get(key)
            if session is None:
                continue
            if digest({k: v for k, v in session.items() if k != "state_hash"}) != session["state_hash"]:
                raise ValueError("training state checksum mismatch")
            scope = session["scope"]
            if (scope["config_hash"] != digest(config) or scope["candidate"] != candidates[name]
                    or scope["sources"] not in accepted):
                raise ValueError("training state source/configuration lineage mismatch")
            if session["level_updates"] < 0 or session["rng_index"] < session["level_updates"]:
                raise ValueError("invalid training progress")
    return state


def cached_loss_rows(checkpoint_path, config=None):
    """Only checksummed complete prefixes earn cost credit."""
    rows = {}
    root = Path(checkpoint_path).parent / "validation-cache"
    for path in root.glob("*/prefix.json"):
        record = json.loads(path.read_text())
        identity, count = record["identity"], record["rows"]
        if (digest(identity) != path.parent.name or type(count) is not int or count <= 0
                or count % identity["batch_size"]):
            raise ValueError("invalid cached loss prefix identity")
        if checksum(path.parent / f"loss-{count:08d}.tensor") != record["sha256"]:
            raise ValueError("cached loss prefix checksum mismatch")
        if config is not None:
            candidate = identity["scope"]["candidate"]["id"]
            if (identity["scope"]["config_hash"] != digest(config) or
                    identity["seed"] != list(scoped_seed(config, "validation", candidate, identity["beta"]))):
                continue
        rows[identity["map"]["checkpoint_hash"]] = max(count, rows.get(identity["map"]["checkpoint_hash"], 0))
    return rows


def import_training_checkpoint(path, config, root, *, previous_root, current_root):
    path, root = Path(path).resolve(), Path(root).resolve()
    state = read_training_checkpoint(path, config)
    current, changed = check_refresh_sources(state["sources"], previous_root, current_root)
    cached = cached_loss_rows(path)
    accepted = [state["sources"], *state.get("source_import", {}).get("accepted_training_sources", [])]
    root.mkdir(parents=True, exist_ok=False)
    old_cache = path.parent / "validation-cache"
    if old_cache.exists():
        shutil.copytree(old_cache, root / "validation-cache")
    receipt = {"previous_checkpoint": str(path), "previous_sha256": checksum(path),
        "previous_source_root": str(Path(previous_root).resolve()), "changed_sources": changed,
        "accepted_training_sources": accepted,
        "state_policy": "original_map_optimizer_rng_history_and_cache_identity_preserved",
        "cached_maps": len(cached), "cached_rows": sum(cached.values()), "promotion_eligible": False}
    for item in state["cohort"].values():
        # Old assessments/exports remain in the parent checkpoint. Source import
        # never turns a calibration export into a currently admitted map.
        item["exports"] = {}
        beta, updates = item["session"]["map"]["beta"], item["session"]["level_updates"]
        current_assessments = [a for a in item["assessments"] if a["beta"] == beta]
        item.setdefault("historical_assessments", []).extend(current_assessments)
        # A source import neither performs an optimizer update nor observes a
        # plateau. Preserve the last genuine comparison separately from exports.
        distinct = [a for a in current_assessments if a.get("previous_map_hash")
                    and a["previous_map_hash"] != a.get("current_map_hash")]
        if distinct:
            item["last_distinct_assessment"] = distinct[-1]
        # Legacy identical-map reports have no comparison identity and cannot
        # supply a plateau count in the repaired assessment protocol.
        genuine = item.get("last_distinct_assessment")
        item["plateaus"] = genuine["decision"]["plateaus"] if genuine else 0
        item["reassessment_requires_update"] = item.get("reassessment_requires_update", False) or any(
            a["updates"] == updates and a["decision"]["status"] in
            {"deterioration_repair_trigger", "numerically_invalid"} for a in current_assessments)
        # The parent retains old decisions. Cached numerical losses remain valid,
        # but the saved current map must receive the current admission decision.
        item["assessments"] = [a for a in item["assessments"] if a["beta"] != beta]
        item["status"] = "continue_training"
    state.update(sources=current, source_import=receipt, status="imported_for_continuation")
    imported = root / "cohort-00000.json"
    write_json(imported, {**state, "checkpoint_hash": digest(state)})
    return {"checkpoint": str(imported), "sha256": checksum(imported),
            "source_import": receipt, "completed_updates": sum(i["session"]["iteration"] for i in state["cohort"].values())}

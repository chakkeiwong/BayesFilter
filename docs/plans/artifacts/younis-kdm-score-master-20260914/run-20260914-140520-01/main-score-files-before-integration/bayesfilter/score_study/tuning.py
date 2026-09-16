"""Offline selection issued from completed, scoped numerical study evidence.

The selected controls are recomputed from validated calibration records. A
caller-provided scope/selected-control label is insufficient for consumption.
Tiny mechanics studies exercise plumbing only and cannot authorize a quality
claim. The same settings and data regime must hold on the claim consumer.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import digest, validate_result, validate_study
from .coordinator import fingerprint, write_json
from .registry import default_registry


def scope(study, row):
    s = study["settings"]
    # Dataset IDs are intentionally omitted: partitions differ, their generating
    # regime does not. Every execution/numerical setting is included.
    return {"model": row["model"], "proposal": row["proposal"], "estimator": row["estimator"],
            "comparison_target": row["comparison_target"], "settings": s,
            "reset_contract_id": "contract_e_chol_v1",
            "chunk_policy_id": "dpf_transport_exact_divisor_cap3000_v1",
            "parameter_dimension": 6,
            "candidate_family": study.get("tuning_candidate_family", [])}


def _derive_selection(source_run):
    root = Path(source_run).resolve()
    state = json.loads((root / "state.json").read_text())
    study = state["study"]
    registry = default_registry()
    validate_study(study, registry)
    if state["fingerprint"] != fingerprint(study, registry):
        raise ValueError("tuning evidence source/registry is stale")
    rows = [r for r in study["rows"] if r["proposal"] == "ledh" and r.get("role") in ("calibration", "validation")]
    if not rows:
        raise ValueError("no completed calibration/validation candidates")
    from .runtime import configure_runtime
    configure_runtime(device=study["settings"]["device"], tf32=study["settings"]["tf32"], jit_compile=study["settings"]["jit_compile"])
    import tensorflow as tf
    records, evidence, reference_scope = {}, [], scope(study, rows[0])
    for row in rows:
        if scope(study, row) != reference_scope:
            raise ValueError("mixed tuning scopes")
        saved = state["rows"][row["id"]]
        if saved["execution_status"] != "complete":
            raise ValueError("failed or missing tuning candidate cannot disappear from selection")
        result = json.loads((root / saved["result_path"]).read_text())
        validate_result(result, row, registry)
        if digest(result) != saved["result_digest"]:
            raise ValueError("tuning numerical evidence digest mismatch")
        if result["diagnostics"].get("controls") != row["controls"]:
            raise ValueError("candidate controls differ from executed controls")
        candidate = digest(row["controls"])
        record = records.setdefault(candidate, {"controls": row["controls"], "calibration": {}, "validation": {}})
        key = (row["dataset"], row["replicate"])
        if key in record[row["role"]]:
            raise ValueError("duplicate tuning replicate")
        error = tf.constant(result["score"], tf.float64) - tf.constant(result["oracle_score"], tf.float64)
        record[row["role"]][key] = tf.reduce_sum(error**2)
        evidence.append({"row": row["id"], "digest": saved["result_digest"]})
    calibration_keys = None
    rows_out = []
    def dataset_mean(records):
        means = [tf.reduce_mean(tf.stack([value for (ds, _), value in records.items() if ds == dataset]))
                 for dataset in sorted({ds for ds, _ in records})]
        return float(tf.reduce_mean(tf.stack(means)).numpy())
    for candidate, record in sorted(records.items()):
        if not record["calibration"] or not record["validation"]:
            raise ValueError("every tuning candidate needs disjoint calibration and validation")
        keys = (set(record["calibration"]), set(record["validation"]))
        if calibration_keys is not None and keys != calibration_keys:
            raise ValueError("unbalanced candidate calibration/validation coverage")
        calibration_keys = keys
        rows_out.append({"candidate": candidate, "controls": record["controls"],
                         "calibration_score_mse": dataset_mean(record["calibration"]),
                         "validation_score_mse": dataset_mean(record["validation"])})
    family = study.get("tuning_candidate_family")
    if not family or {digest(c) for c in family} != set(records):
        raise ValueError("executed candidates differ from declared tuning family")
    # Validation is reserved for checks/reporting; calibration nominates. With
    # no powered validation criterion this issues only a frozen research arm.
    losses = tf.constant([r["calibration_score_mse"] for r in rows_out], tf.float64)
    selected = rows_out[int(tf.argmin(losses).numpy())]
    return {"schema": "younis_score_offline_selection_v1", "issuer": "bayesfilter.score_study.tuning.issue_selection",
            "source_run": str(root), "source_fingerprint": digest(state["fingerprint"]),
            "scope": reference_scope, "evidence": evidence, "candidates": rows_out,
            "selected_controls": selected["controls"], "criterion": "calibration_mean_squared_error_to_exact_score",
            "validation_status": "complete_finite_descriptive_only", "evidence_class": study["evidence_class"],
            "partitions": study["partitions"], "statistically_supported_ranking": False,
            "default_ready": False}


def issue_selection(source_run, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("preserve existing tuning selections; choose a new destination")
    result = _derive_selection(source_run)
    write_json(destination, result)
    return result


def consume_selection(row, context):
    selection = json.loads(Path(row["tuning_selection"]).read_text())
    derived = _derive_selection(selection["source_run"])
    if selection != derived:
        raise ValueError("selection is stale, modified, or caller-stamped")
    if scope(context["study"], row) != selection["scope"]:
        raise ValueError("tuning scope mismatch")
    if selection["evidence_class"] == "mechanics" and context["study"].get("evidence_class") != "mechanics":
        raise ValueError("mechanics tuning cannot support a scientific claim")
    if row["dataset"] in selection["partitions"]["calibration"] + selection["partitions"]["validation"]:
        raise ValueError("claim data leaked into tuning")
    return derived["selected_controls"]

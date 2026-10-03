"""Untouched multi-scope comparisons; CPU diagnostic reporting, no admission.

Bootstrap dataset pairs, never individual particle replicates. Intervals are
descriptive pilot intervals and do not issue a method ranking or default.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import digest
from .coordinator import write_json
from .normalization_reporting import _validated_rows
from .reporting import paired_dataset_summary


def paired_bootstrap_summary(left, right, *, resamples=20000, seed=95231):
    from .runtime import configure_runtime
    configure_runtime(device="CPU", tf32=False, jit_compile=True)
    result = paired_dataset_summary(left, right)
    if result["status"] == "coverage_veto":
        return result
    if type(resamples) is not int or resamples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    import tensorflow as tf
    differences = tf.constant([r["mean_squared_error_difference"]
                               for r in result["per_dataset"]], tf.float64)
    if not bool(tf.reduce_all(tf.math.is_finite(differences))):
        raise ValueError("nonfinite paired errors")
    count = result["dataset_count"]
    if count < 2:
        return {**result, "bootstrap_interval": None,
                "interval_status": "insufficient_independent_datasets"}
    indices = tf.random.stateless_uniform([resamples, count], [seed, 17],
                                          maxval=count, dtype=tf.int32)
    means = tf.sort(tf.reduce_mean(tf.gather(differences, indices), axis=1))
    def quantile(probability):
        position = probability * (resamples - 1)
        lower = int(position)
        return float((means[lower] + (position-lower) *
                      (means[min(lower+1, resamples-1)]-means[lower])).numpy())
    return {**result, "bootstrap_interval": [quantile(.025), quantile(.975)],
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "interval_status": "descriptive_dataset_percentile_95pct_unadjusted",
            "uncertainty_limit": "Independent datasets assumed; pilot bootstrap tail coverage and multiplicity are not certified. No ranking issued."}


def assemble_heldout(run_roots, destination, contract):
    """Join complete held-out scopes with exact model/data/replicate agreement."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("preserve existing held-out reports")
    from .runtime import configure_runtime
    configure_runtime(device="CPU", tf32=False, jit_compile=True)
    import tensorflow as tf
    expected = {(d, r) for d in contract["datasets"] for r in contract["replicates"]}
    expected_groups = {(c, m) for c in contract["conditions"] for m in contract["methods"]}
    if not expected or not expected_groups:
        raise ValueError("empty held-out evidence contract")
    groups, settings_by_condition, data, configurations, raw = {}, {}, {}, {}, []
    for root in run_roots:
        state, records = _validated_rows(root)
        study = state["study"]
        if study.get("evidence_class") == "mechanics":
            raise ValueError("mechanics source cannot enter untouched research comparison")
        for _, row, result in records:
            if row["role"] != "claim":
                raise ValueError("calibration or validation leaked into held-out report")
            condition, method = row["condition"], row["proposal"]
            group = (condition, method)
            key = (row["dataset"], row["replicate"])
            if group not in expected_groups or key not in expected:
                raise ValueError("unexpected condition, method or replicate")
            scope = (row["model"], row["comparison_target"], digest(study["settings"]), study["seed"])
            if settings_by_condition.setdefault(condition, scope) != scope:
                raise ValueError("different model or execution scopes within a condition")
            identity = (result["diagnostics"]["data_version"], result["oracle_value"], result["oracle_score"])
            if data.setdefault((condition, row["dataset"]), identity) != identity:
                raise ValueError("different data or reference scores within a condition")
            config = result["diagnostics"].get("candidate_configuration")
            if configurations.setdefault(group, config) != config:
                raise ValueError("multiple tuned configurations in one held-out arm")
            values = groups.setdefault(group, {})
            if key in values:
                raise ValueError("duplicate held-out replicate")
            error = tf.constant(result["score"], tf.float64) - tf.constant(result["oracle_score"], tf.float64)
            component_error = error**2
            if not bool(tf.reduce_all(tf.math.is_finite(component_error))):
                raise ValueError("nonfinite held-out squared error")
            values[key] = float(tf.reduce_sum(component_error).numpy())
            raw.append({"condition": condition, "method": method, "dataset": key[0],
                        "replicate": key[1], "score_squared_error": values[key],
                        "coordinate_squared_error": component_error.numpy().tolist(),
                        "kernel_wall_seconds": result["runtime"]["kernel_wall_seconds"],
                        "result_root": str(root), "row": row["id"]})
    if set(groups) != expected_groups or any(set(v) != expected for v in groups.values()):
        raise ValueError("missing held-out condition/method/dataset/replicate; no complete-case deletion")
    summaries = []
    for (condition, method), values in sorted(groups.items()):
        rows = [r for r in raw if (r["condition"], r["method"]) == (condition, method)]
        summaries.append({"condition": condition, "method": method,
                          "score_mse": float(tf.reduce_mean(tf.constant(list(values.values()), tf.float64)).numpy()),
                          "coordinate_mse": tf.reduce_mean(tf.constant([r["coordinate_squared_error"] for r in rows], tf.float64), 0).numpy().tolist(),
                          "configuration": configurations[(condition, method)]})
    comparisons = []
    for condition in contract["conditions"]:
        for candidate in contract["candidates"]:
            for comparator in contract["heuristics"] + ["ledh"]:
                if comparator == candidate:
                    continue
                comparison = paired_bootstrap_summary(groups[(condition, candidate)], groups[(condition, comparator)],
                    resamples=contract["bootstrap_resamples"], seed=contract["bootstrap_seed"])
                comparisons.append({"condition": condition, "candidate": candidate,
                                    "comparator": comparator, **comparison})
    losses = [r for r in comparisons if r["comparator"] in contract["heuristics"]
              and r["mean_squared_error_difference"] > 0]
    result = {"schema": "younis_score_heldout_comparison_v1", "contract": contract,
              "rows": raw, "summaries": summaries, "paired_comparisons": comparisons,
              "hard_veto_screen": "complete_finite_reference_and_scope_checks_passed",
              "heuristic_dominance_verdict": "promotion_veto_observed_losses" if losses else "no_observed_loss_pilot_only",
              "observed_heuristic_losses": losses, "statistically_supported_ranking": False,
              "default_ready": False, "aggregation_backend": "TensorFlow CPU diagnostic; GPU intentionally hidden",
              "next_evidence": "Broader control calibration, more independent datasets and streams, matched-cost replication and a predeclared ranking protocol."}
    write_json(destination, result)
    return result

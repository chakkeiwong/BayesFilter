"""Post-run diagnostic aggregation; no selection or default-admission authority.

Datasets, not particle replicates, are the independent sampling units. Raw
failures and unequal coverage remain visible. Confidence intervals here describe
paired dataset errors; mechanics runs never issue a statistical ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import digest, validate_result
from .coordinator import fingerprint, write_json
from .registry import default_registry


def paired_dataset_summary(left, right):
    """Compare {(dataset, replicate): squared_error} with exact paired coverage."""
    import tensorflow as tf
    if not left or set(left) != set(right):
        return {"status": "coverage_veto", "left_count": len(left), "right_count": len(right)}
    dataset_ids = sorted({key[0] for key in left})
    per_dataset = []
    for dataset in dataset_ids:
        differences = [left[k] - right[k] for k in sorted(left) if k[0] == dataset]
        values = tf.constant(differences, tf.float64)
        per_dataset.append({"dataset": dataset, "replicates": len(differences),
                            "mean_squared_error_difference": float(tf.reduce_mean(values).numpy())})
    values = tf.constant([x["mean_squared_error_difference"] for x in per_dataset], tf.float64)
    mean = tf.reduce_mean(values)
    n = len(per_dataset)
    se = tf.sqrt(tf.reduce_sum((values - mean)**2) / (n * (n - 1))) if n > 1 else None
    return {"status": "descriptive_paired", "dataset_count": n, "replicate_count": len(left),
            "mean_squared_error_difference": float(mean.numpy()),
            "dataset_mcse": float(se.numpy()) if se is not None else None,
            "per_dataset": per_dataset, "statistically_supported_ranking": False,
            "uncertainty_limit": "MCSE uses independent dataset means; no powered or multiplicity-adjusted ranking test"}


def assemble_diagnostics(run_root):
    root = Path(run_root)
    state = json.loads((root / "state.json").read_text())
    study, registry = state["study"], default_registry()
    if state["fingerprint"] != fingerprint(study, registry):
        raise ValueError("cannot assemble current-source evidence from a stale run")
    from .runtime import configure_runtime
    configure_runtime(device="CPU", tf32=False, jit_compile=True)
    import tensorflow as tf
    groups, failures, raw, data_versions = {}, [], [], {}
    for row in study["rows"]:
        saved = state["rows"].get(row["id"], {})
        if saved.get("execution_status") != "complete":
            failures.append({"row": row["id"], "proposal": row["proposal"], **saved})
            continue
        result = json.loads((root / saved["result_path"]).read_text())
        validate_result(result, row, registry)
        if digest(result) != saved["result_digest"]:
            raise ValueError("corrupt numerical result")
        condition = row.get("condition", "affine_gaussian_correctness_fixture")
        # Different tuning candidates stay separate instead of being averaged.
        candidate = row["proposal"] + (":" + digest(row["controls"])[:12] if "controls" in row else "")
        dataset_key = (condition, row["dataset"])
        version = result["diagnostics"]["data_version"]
        if dataset_key in data_versions and data_versions[dataset_key] != version:
            raise ValueError("paired methods evaluated different observation data")
        data_versions[dataset_key] = version
        error = tf.constant(result["score"], tf.float64) - tf.constant(result["oracle_score"], tf.float64)
        squared_error = float(tf.reduce_sum(error**2).numpy())
        group = groups.setdefault((condition, candidate), {})
        key = (row["dataset"], row["replicate"])
        if key in group:
            raise ValueError("duplicate scientific replicate")
        group[key] = squared_error
        raw.append({"row": row["id"], "condition": condition, "proposal": row["proposal"],
                    "candidate": candidate, "dataset": row["dataset"], "replicate": row["replicate"],
                    "score_squared_error": squared_error,
                    "kernel_wall_seconds": result["runtime"]["kernel_wall_seconds"],
                    "kernel_calls": result["runtime"]["kernel_calls"]})
    comparisons = []
    for (condition, candidate), values in sorted(groups.items()):
        if candidate == "kalman":
            continue
        for adversary in ("bootstrap", "ukf", "adapted", "prior_sis", "adapted_sis"):
            if candidate == adversary:
                continue
            comparison = paired_dataset_summary(values, groups.get((condition, adversary), {}))
            comparisons.append({"condition": condition, "candidate": candidate, "heuristic": adversary, **comparison})
    result = {"schema": "younis_score_diagnostic_comparison_v1", "raw_rows": raw,
              "aggregation_backend": "TensorFlow CPU diagnostic, GPU intentionally hidden",
              "failed_or_missing_rows": failures, "conditional_heuristic_table": comparisons,
              "hard_veto_screen": "failed_rows_preserved" if failures else "passed_numerical_smoke",
              "heuristic_dominance_verdict": "correctness_fixture_only_no_promotion",
              "statistically_supported_ranking": False, "default_ready": False,
              "interpretation": "Affine Gaussian UKF equals the exact Kalman score. This tests implementation and error reporting; particle error does not reject the research direction.",
              "next_evidence": "scope tuning, nonlinear oracle models, independent datasets and predeclared paired uncertainty criteria"}
    write_json(root / "comparison.json", result)
    return result

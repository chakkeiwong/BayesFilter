"""Diagnostic normalization and particle-count comparisons; no selection authority.

V = exp(log Zhat - log Zref) * score is defined from the same finite program.
Neither this construction nor its empirical covariance identity establishes
unbiasedness of Zhat, its derivative, their ratio, or a consistency surrogate.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import digest, validate_result
from .coordinator import fingerprint, write_json
from .registry import default_registry


def normalization_summary(log_values, scores, reference_value, reference_score):
    """Summarize independent particle replicates conditional on one dataset."""
    import tensorflow as tf
    logs = tf.constant(log_values, tf.float64)
    s = tf.constant(scores, tf.float64)
    target = tf.constant(reference_score, tf.float64)
    if logs.shape.rank != 1 or s.shape.rank != 2 or s.shape[0] != logs.shape[0] or s.shape[1:] != target.shape:
        raise ValueError("normalization replicates have incompatible shapes")
    count = int(logs.shape[0])
    if count < 2:
        raise ValueError("at least two particle replicates required")
    u = tf.exp(logs - tf.constant(reference_value, tf.float64))
    reciprocal = tf.math.reciprocal(u)
    v = u[:, None] * s
    for value in (logs, s, target, u, reciprocal, v):
        if not bool(tf.reduce_all(tf.math.is_finite(value))):
            raise ValueError("normalization diagnostic overflow/nonfinite veto")
    mean_u, mean_s, mean_v = tf.reduce_mean(u), tf.reduce_mean(s, 0), tf.reduce_mean(v, 0)
    var_u = tf.reduce_sum((u-mean_u)**2) / (count-1)
    cov_vu = tf.reduce_sum((v-mean_v)*(u-mean_u)[:, None], 0)/(count-1)
    cov_us = tf.reduce_mean((u-mean_u)[:, None]*(s-mean_s), 0)
    ratio = mean_v / mean_u
    expansion = (mean_v*var_u/mean_u**3-cov_vu/mean_u**2)/count
    error = s-target
    def scalar(x):
        return float(x.numpy())
    def vector(x):
        return x.numpy().tolist()
    quantiles = {}
    ordered = tf.sort(reciprocal)
    for name, q in (("q50", .5), ("q95", .95), ("q99", .99)):
        position = q*(count-1)
        lower = int(position)
        upper = min(lower+1, count-1)
        quantiles[name] = scalar(ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower]))
    answer = {
        "replicates": count, "mean_scaled_normalizer": scalar(mean_u),
        "scaled_normalizer_mcse": scalar(tf.sqrt(var_u/count)),
        "normalizer_cv": scalar(tf.sqrt(var_u)/mean_u),
        "mean_scaled_derivative": vector(mean_v), "mean_individual_log_score": vector(mean_s),
        "ratio_of_means": vector(ratio), "ratio_minus_mean_score": vector(ratio-mean_s),
        "normalizer_score_covariance_divisor_R": vector(cov_us),
        "covariance_identity_residual": vector(mean_v-mean_u*mean_s-cov_us),
        "derivative_normalizer_sample_covariance": vector(cov_vu),
        "ratio_of_means_bias_expansion_plugin": vector(expansion),
        "empirical_score_bias": vector(mean_s-target),
        "empirical_score_mse": scalar(tf.reduce_mean(tf.reduce_sum(error**2, -1))),
        "score_sample_variance_trace": scalar(tf.reduce_sum((s-mean_s)**2)/(count-1)),
        "reciprocal_normalizer": {**quantiles, "maximum": scalar(tf.reduce_max(reciprocal))},
        "unbiasedness_established": False, "statistically_supported_ranking": False,
        "interpretation": "Conditional replicate diagnostics. The plug-in Taylor term is not an estimated exact score bias; reciprocal tails and MCSE are descriptive with few replicates.",
    }
    # Products/ratios can overflow even when the input reciprocals are finite.
    json.dumps(answer, allow_nan=False)
    return answer


def descriptive_correlation(x, y):
    import tensorflow as tf
    a, b = tf.constant(x, tf.float64), tf.constant(y, tf.float64)
    if a.shape.rank != 1 or a.shape != b.shape or int(a.shape[0]) < 2:
        raise ValueError("correlation requires matching nonempty replicate vectors")
    if not bool(tf.reduce_all(tf.math.is_finite(a))) or not bool(tf.reduce_all(tf.math.is_finite(b))):
        raise ValueError("nonfinite correlation input")
    da, db = a-tf.reduce_mean(a), b-tf.reduce_mean(b)
    # Roundoff-scale variation does not identify a correlation. No ridge alters it.
    eps = tf.constant(2.220446049250313e-16, tf.float64)
    for values, centered in ((a, da), (b, db)):
        if bool(tf.reduce_max(tf.abs(centered)) <= 32*eps*tf.reduce_max(tf.abs(values))):
            return {"status": "undefined_negligible_variance", "correlation": None}
    value = tf.reduce_sum(da*db)/tf.sqrt(tf.reduce_sum(da**2)*tf.reduce_sum(db**2))
    return {"status": "descriptive_only", "correlation": float(value.numpy())}


def _validated_rows(run_root):
    root = Path(run_root)
    state = json.loads((root/"state.json").read_text())
    registry = default_registry()
    if state["fingerprint"] != fingerprint(state["study"], registry):
        raise ValueError("stale source in normalization evidence")
    records = []
    for row in state["study"]["rows"]:
        saved = state["rows"].get(row["id"], {})
        if saved.get("execution_status") != "complete":
            raise ValueError("failed or missing normalization row; no complete-case deletion")
        result = json.loads((root/saved["result_path"]).read_text())
        validate_result(result, row, registry)
        if digest(result) != saved["result_digest"]:
            raise ValueError("corrupt normalization evidence")
        config = result["diagnostics"].get("candidate_configuration", row.get("controls"))
        key = (row["model"], row.get("condition", row["model"]), row["proposal"],
               row["estimator"], row["comparison_target"], row["role"], digest(config))
        records.append((key, row, result))
    return state, records


def assemble_normalization(run_root):
    state, records = _validated_rows(run_root)
    from .runtime import configure_runtime
    configure_runtime(device="CPU", tf32=False, jit_compile=True)
    groups = {}
    for key, row, result in records:
        groups.setdefault((key, row["dataset"]), []).append((row, result))
    summaries = []
    for (key, dataset), group in sorted(groups.items()):
        rows, values = zip(*group)
        if len({r["replicate"] for r in rows}) != len(rows):
            raise ValueError("duplicate particle replicate")
        reference = {(v["diagnostics"]["data_version"], v["oracle_value"], tuple(v["oracle_score"])) for v in values}
        if len(reference) != 1:
            raise ValueError("different dataset/reference among conditional replicates")
        summary = normalization_summary([v["value"] for v in values], [v["score"] for v in values],
                                        values[0]["oracle_value"], values[0]["oracle_score"])
        summaries.append({"model": key[0], "condition": key[1], "proposal": key[2],
                          "configuration_digest": key[-1], "dataset": dataset,
                          "particle_count": state["study"]["settings"]["particles"], **summary})
    result = {"schema": "younis_normalization_diagnostics_v1", "groups": summaries,
              "source_fingerprint": state["fingerprint"], "source_run": str(Path(run_root).resolve()),
              "backend": "TensorFlow CPU diagnostic; GPU intentionally hidden", "default_ready": False,
              "derivative_construction": "V = exp(log Zhat - log Zref) * finite-program log score; unbiasedness not assumed"}
    write_json(Path(run_root)/"normalization.json", result)
    return result


def assemble_consistency(run_roots, output_path):
    """Join N, 2N, 4N at the same data and numerical scope, retaining all rows."""
    if len(run_roots) != 3:
        raise ValueError("exactly N, 2N, 4N runs required")
    runs = [_validated_rows(root) for root in run_roots]
    runs.sort(key=lambda item: item[0]["study"]["settings"]["particles"])
    counts = [s["study"]["settings"]["particles"] for s, _ in runs]
    if counts != [counts[0], 2*counts[0], 4*counts[0]]:
        raise ValueError("particle counts must be N, 2N, 4N")
    from .runtime import configure_runtime
    configure_runtime(device="CPU", tf32=False, jit_compile=True)
    import tensorflow as tf
    def common_scope(state):
        study = state["study"]
        return {"settings": {k: v for k, v in study["settings"].items() if k != "particles"},
                "seed": study["seed"], "partitions": study["partitions"],
                "evidence_class": study["evidence_class"]}
    if any(common_scope(s) != common_scope(runs[0][0]) for s, _ in runs[1:]):
        raise ValueError("incompatible scope in particle-count comparison")
    maps = []
    for _, records in runs:
        mapping = {}
        for key, row, result in records:
            identity = (key, row["dataset"], row["replicate"], row.get("coupling_group", "baseline"))
            if identity in mapping:
                raise ValueError("duplicate consistency replicate")
            mapping[identity] = result
        maps.append(mapping)
    if any(set(m) != set(maps[0]) for m in maps[1:]):
        raise ValueError("incomplete paired particle-count coverage")
    groups = {}
    for identity in sorted(maps[0]):
        values = [m[identity] for m in maps]
        data = {(v["diagnostics"]["data_version"], v["oracle_value"], tuple(v["oracle_score"])) for v in values}
        if len(data) != 1:
            raise ValueError("different observation data/reference across particle counts")
        scores = [tf.constant(v["score"], tf.float64) for v in values]
        oracle = tf.constant(values[0]["oracle_score"], tf.float64)
        norms = [float(tf.linalg.norm(s-oracle).numpy()) for s in scores]
        gaps = [float(tf.linalg.norm(scores[i]-scores[i+1]).numpy()) for i in (0, 1)]
        groups.setdefault(identity[0], []).append({"dataset": identity[1], "replicate": identity[2],
             "error_norms": norms, "successive_difference_norms": gaps})
    result_groups = []
    for key, records in sorted(groups.items()):
        per_dataset = []
        for dataset in sorted({r["dataset"] for r in records}):
            subset = [r for r in records if r["dataset"] == dataset]
            per_dataset.append({"dataset": dataset, **descriptive_correlation(
                [r["successive_difference_norms"][0] for r in subset], [r["error_norms"][0] for r in subset])})
        result_groups.append({"model": key[0], "condition": key[1], "proposal": key[2],
             "configuration_digest": key[-1], "rows": records, "per_dataset": per_dataset,
             "pooled_descriptive_association": descriptive_correlation(
                 [r["successive_difference_norms"][0] for r in records], [r["error_norms"][0] for r in records])})
    result = {"schema": "younis_particle_consistency_diagnostics_v1", "particle_counts": counts,
              "source_runs": [str(Path(p).resolve()) for p in run_roots], "groups": result_groups,
              "default_ready": False, "statistically_supported_ranking": False,
              "coupling": "common seed labels; shape changes do not establish nested draws",
              "interpretation": "Descriptive association with oracle error, not calibrated bias prediction or a consistency proof. Dataset clustering precludes treating pooled rows as independent inference units."}
    write_json(Path(output_path), result)
    return result

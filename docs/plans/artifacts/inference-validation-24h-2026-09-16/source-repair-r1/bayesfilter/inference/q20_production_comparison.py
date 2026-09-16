"""Named posterior quantities and uncertainty-aware q20 comparisons.

All intervals are marginal asymptotic intervals. Passing every screen is not
proof of simultaneous coverage or global mode discovery.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.q20_production_config import PARAMETERS, frozen_scope_hash, write_json


def quantity_key(name, kind="mean", probability=None):
    return name if kind == "mean" else f"{name}:quantile:{probability:g}"


def quantity_layout(config):
    rows = [{"name": name, "kind": "mean", "coordinate": i, "probability": None}
            for i, name in enumerate(PARAMETERS)]
    rows += [{"name": name, "kind": "quantile", "coordinate": i, "probability": p}
             for i, name in enumerate(PARAMETERS) for p in config["posterior"]["quantiles"]]
    rows.append({"name": "positive_theta_2", "kind": "mean", "coordinate": None, "probability": None})
    return rows


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def posterior_quantities_summary(config, retained):
    from bayesfilter.inference.hmc_precision import precision_report
    from bayesfilter.inference.q20_production_hmc import posterior_policy
    retained = tf.convert_to_tensor(retained, tf.float64)
    if retained.shape.rank != 3 or retained.shape[-1] != len(PARAMETERS):
        raise ValueError("posterior archive must have [draw,chain,physical_parameter] shape")
    if retained.shape[0] < 4 or retained.shape[1] < 2:
        return {}
    event = tf.cast(retained[:, :, config["posterior"]["event_coordinate"]] > 0, tf.float64)
    values = tf.concat([retained, event[..., None]], axis=-1)
    report = precision_report(values, (*PARAMETERS, "positive_theta_2"), posterior_policy(config).precision)
    sd = tf.math.reduce_std(retained, axis=(0, 1))
    result = {}
    for row in report["targets"]:
        index = PARAMETERS.index(row["name"]) if row["name"] in PARAMETERS else None
        key = quantity_key(row["name"], row["kind"], row["probability"])
        result[key] = {"estimate": _finite(row["estimate"]), "mcse": row["mcse"],
                       "posterior_sd": None if index is None else _finite(sd[index]),
                       "kind": row["kind"], "name": row["name"], "probability": row["probability"],
                       "valid": row["valid"], "precision_passed": row["passed"]}
    return result


def posterior_summary(config, retained, *, target_signature, label, sequential_passed):
    count = None
    shape = None
    if retained is not None:
        shape = [int(v) for v in retained.shape]
        count = int(retained.shape[0])
    result = {
        "schema": "bayesfilter.q20.posterior_summary.v1",
        "label": str(label),
        "target_signature": str(target_signature),
        "frozen_scope_hash": frozen_scope_hash(config),
        "retained_shape": shape,
        "retained_draws_per_chain": count,
        "sequential_declared_checks_passed": bool(sequential_passed),
        "reference_status": "unavailable",
        "reference_agreement": "incomplete",
        "method_ranking": "not_estimated",
        "production_qualified": False,
        "nonclaims": [
            "R-hat, ESS or MCSE does not establish target correctness or mode coverage",
            "a missing independent reference cannot be replaced by a training score",
            "one chart or one seed cannot establish ensemble or method superiority",
        ],
    }
    result["quantities"] = {} if retained is None else posterior_quantities_summary(config, retained)
    # Two start strata contain two independent chains each. A missing/constant
    # group MCSE cannot be converted to zero in the equivalence screen.
    result["start_groups"] = {} if retained is None else {
        "negative_start": posterior_quantities_summary(config, retained[:, :2]),
        "positive_start": posterior_quantities_summary(config, retained[:, 2:]),
    }
    return result


def compare_quantities(config, left, right):
    c = config["comparison"]
    critical = float(tfp.distributions.Normal(tf.constant(0., tf.float64), tf.constant(1., tf.float64)).quantile(
        (1+c["interval_probability"])/2))
    expected = {quantity_key(r["name"], r["kind"], r["probability"]) for r in quantity_layout(config)}
    if set(left) != expected or set(right) != expected:
        return {"passed": False, "status": "missing_named_quantities", "quantities": {}}
    rows = {}
    for key in sorted(expected):
        a, b = left[key], right[key]
        numbers = [a.get("estimate"), b.get("estimate"), a.get("mcse"), b.get("mcse")]
        valid = (a.get("valid") is True and b.get("valid") is True
                 and all(v is not None and math.isfinite(v) for v in numbers)
                 and all(v > 0 for v in numbers[2:])
                 and all(a.get(field) == b.get(field) for field in ("name", "kind", "probability")))
        event = a["name"] == "positive_theta_2"
        scale = None if event else b.get("posterior_sd")
        valid = valid and (event or (scale is not None and math.isfinite(scale) and scale > 0))
        margin = (c["event_margin"] if event else c["quantile_margin_sd"]*scale if valid and a["kind"] == "quantile"
                  else c["mean_margin_sd"]*scale if valid else None)
        bound = abs(numbers[0]-numbers[1]) + critical*math.hypot(numbers[2],numbers[3]) if valid else None
        rows[key] = {"valid": valid, "absolute_difference_plus_uncertainty": bound,
                     "margin": margin, "passed": bool(valid and bound <= margin)}
    return {"passed": all(r["passed"] for r in rows.values()), "status": "assessed",
            "interval_probability": c["interval_probability"], "interval_role": "marginal_asymptotic_not_simultaneous",
            "quantities": rows}


def compare_campaign(config, root, *, reference_path, posterior_paths):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    reference = json.loads(Path(reference_path).read_text())
    from bayesfilter.inference.q20_production_training import source_snapshot
    if reference.get("sources") != source_snapshot():
        raise ValueError("comparison reference source mismatch")
    if config["role"] != "smoke" and reference.get("role") == "smoke":
        raise ValueError("smoke reference cannot qualify serious comparison")
    rows = {}
    for method, path in posterior_paths.items():
        record = json.loads(Path(path).read_text())
        if record.get("role") != config["role"]:
            raise ValueError("posterior role differs from comparison")
        summary = record["summary"]
        if summary["frozen_scope_hash"] != frozen_scope_hash(config) or reference["frozen_scope_hash"] != frozen_scope_hash(config):
            raise ValueError("comparison protocol scope mismatch")
        if summary["target_signature"] != reference["target_signature"]:
            raise ValueError("comparison posterior/reference target mismatch")
        agreement = compare_quantities(config, summary["quantities"], reference["quantities"])
        groups = summary["start_groups"]
        starts = compare_quantities(config, groups.get("negative_start", {}), groups.get("positive_start", {}))
        rows[method] = {"posterior_checks_passed": summary["sequential_declared_checks_passed"],
                        "reference_agreement": agreement, "start_equivalence": starts,
                        "passed": bool(summary["sequential_declared_checks_passed"] and
                            reference["qualified"] and agreement["passed"] and starts["passed"])}
    complete = set(rows) == set(config["comparison"]["methods"])
    passed = complete and all(r["passed"] for r in rows.values())
    result = {"status": "comparison_passed" if passed else "comparison_incomplete_or_failed",
              "passed": passed, "complete_method_inventory": complete,
              "reference_qualified": reference["qualified"], "methods": rows,
              "method_ranking": "not_estimated", "production_qualified": False,
              "role": config["role"], "frozen_scope_hash": frozen_scope_hash(config)}
    write_json(root / "result.json", result)
    return result

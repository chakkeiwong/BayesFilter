"""Diagnostic comparison of MCSE intervals with empirical grid sensitivity.

Numerical integration sensitivity is not a certified error bound. These
comparisons must never be pooled into exact-reference coverage counts.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist


def numerical_interval_agreement(estimate, names, metadata):
    """Report containment/intersection without upgrading numerical truth."""
    unavailable = {"available": False, "finding": "reference_unavailable",
                   "exact_coverage_eligible": False, "integration_error_bound": False}
    if not isinstance(metadata, dict) or metadata.get("checked") is not True:
        return unavailable
    if metadata.get("method") != "independent_same_target_trapezoidal_grid":
        return unavailable
    name, kind = estimate.get("name"), estimate.get("kind")
    if list(names).count(name) != 1 or kind not in {"mean", "quantile"}:
        return {**unavailable, "finding": "unsupported_quantity"}
    # The reference computes medians only. Explicitly reject other quantiles.
    if kind == "quantile" and estimate.get("probability") != .5:
        return {**unavailable, "finding": "unsupported_quantile"}
    key = ("mean" if kind == "mean" else "median") + str(list(names).index(name))
    order = metadata.get("summary_order", [])
    if not isinstance(order, list) or order.count(key) != 1:
        return {**unavailable, "finding": "missing_or_duplicate_reference_quantity"}
    index = order.index(key)
    try:
        truth = metadata["summary"][index]
        radius = metadata["sensitivity"][index]
        limit = metadata["sensitivity_limits"][index]
        value, se = estimate["estimate"], estimate["mcse"]
        edge_limit = metadata["edge_limit"]
        finite = all(type(v) in (int, float) and math.isfinite(v)
                     for v in (truth, radius, limit, value, se, edge_limit))
        valid_edges = all(type(metadata[k]) in (int, float) and math.isfinite(metadata[k])
                         and 0 <= metadata[k] <= edge_limit
                         for k in ("edge_mass", "expanded_edge_mass"))
    except (KeyError, IndexError, TypeError):
        return {**unavailable, "finding": "invalid_reference_or_precision"}
    if (not finite or estimate.get("valid") is not True or se < 0 or radius < 0
            or limit < 0 or radius > limit or edge_limit < 0 or not valid_edges):
        return {**unavailable, "finding": "invalid_reference_or_precision"}
    half_width = NormalDist().inv_cdf(.975) * se
    interval = [value - half_width, value + half_width]
    envelope = [truth - radius, truth + radius]
    if not all(math.isfinite(v) for v in interval + envelope):
        return {**unavailable, "finding": "invalid_reference_or_precision"}
    contains = interval[0] <= envelope[0] and envelope[1] <= interval[1]
    disjoint = interval[1] < envelope[0] or envelope[1] < interval[0]
    return {"available": True, "finding": "contains_sensitivity_interval" if contains else
            "disjoint_from_sensitivity_interval" if disjoint else "overlaps_sensitivity_interval",
            "reference": truth, "empirical_sensitivity": radius,
            "reference_sensitivity_interval": envelope, "nominal_mcse_interval": interval,
            "point_reference_inside": interval[0] <= truth <= interval[1],
            "exact_coverage_eligible": False, "integration_error_bound": False,
            "interpretation": "descriptive sensitivity comparison; no integration or sequential coverage guarantee"}


def report_saved_intervals(cell, output):
    """Reassess recorded final checks without changing or rerunning any fit."""
    from ..catalog import get_target
    from ..designs import ValidationDesign
    from ..engines.pipeline import stopped_intervals
    from ..storage import read_json, write_json, file_hash

    cell, output = Path(cell), Path(output)
    if output.exists():
        raise ValueError("interval report output already exists")
    design_file = cell / "isolated_design.json"
    assessment_file = cell / "replication-0000/independent_assessment.json"
    hashes = {str(p): file_hash(p) for p in (design_file, assessment_file)}
    design = ValidationDesign.from_payload(read_json(design_file))
    assessment = read_json(assessment_file)
    spec = get_target(design.scenario.target)
    rows = []
    for row in assessment["members"]:
        if row.get("status") != "assessed":
            rows.append({"candidate_id": row["candidate_id"], "status": row["status"]})
            continue
        rows.append({"candidate_id": row["candidate_id"], "status": "assessed",
            "stopped_intervals": stopped_intervals(row["member_record"], spec,
                design.scenario.parameters, design.options.get("data"),
                reference_metadata=row["assessment"].get("reference_metadata"))})
    if any(file_hash(p) != sha for p, sha in hashes.items()):
        raise ValueError("saved evidence changed during reporting")
    result = {"design_identity": design.identity, "target": spec.target_id,
        "input_hashes": hashes, "members": rows, "new_sampling": False,
        "sequential_coverage_established": False, "reporter_sha256": file_hash(__file__)}
    write_json(output, result)
    return result

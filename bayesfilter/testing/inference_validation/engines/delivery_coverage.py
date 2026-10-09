"""Diagnostic complete-fit delivery and interval coverage; no sampler decisions."""
from __future__ import annotations

import math

import numpy as np  # Diagnostic scalar types only; never a runtime decision path.

from .statistics import binomial_interval


def summarize_delivery_coverage(records, planned, *, declared_names, member_rule,
                                member_l, coverage_floor=None, alpha=.05):
    """Count the predeclared member once; missing slots are unsuccessful events.

    Readiness alone is insufficient: corrupted inventory, duplicated chains or
    leaked warmup veto delivery. Intervals at a precision cap remain available
    for the historical coverage event, but cannot count as qualified delivery.
    """
    if type(planned) is not int or planned < 1:
        raise ValueError("planned fits must be a positive integer")
    if (not math.isfinite(alpha) or not 0 < alpha < 1
            or coverage_floor is not None and (
                not math.isfinite(coverage_floor) or not 0 < coverage_floor < 1)):
        raise ValueError("invalid confidence or coverage floor")
    if member_rule not in {"first_verified", "declared_l_first"}:
        raise ValueError("delivery report requires a predeclared single-member rule")
    # In-memory unit fixtures may omit the persisted index; their list order is
    # the declared order. Persisted workers always write the explicit index.
    implicit_indices = any("replication" not in r for r in records)
    if coverage_floor is not None and implicit_indices:
        raise ValueError("coverage screens require explicit replication indices")
    indices = [r.get("replication", index) for index, r in enumerate(records)]
    if (any(type(i) is not int or not 0 <= i < planned for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("replications must have distinct in-range planned indices")
    names = tuple(declared_names)
    if len(set(names)) != len(names):
        raise ValueError("duplicate declared quantities")
    counts = {name: {"available": 0, "covered": 0, "delivered_and_covered": 0}
              for name in names}
    delivered = 0
    for record in records:
        if "execution_failure" in record:
            continue
        group = sorted((m for m in record.get("members", ())
                        if m.get("status") != "unassessed_by_design"
                        and (member_rule == "first_verified" or m.get("L") == member_l)),
                       key=lambda m: m["candidate_id"])
        if not group:
            continue
        member = group[0]
        valid = (record.get("inventory", {}).get("failures") == []
                 and member.get("runtime_checks_passed") is True
                 and member.get("warmup_exclusion_matches") is True
                 and member.get("duplicate_chains") is False)
        delivered += int(valid)
        seen = set()
        for row in member.get("stopped_intervals", {}).get("quantities", ()):
            name = row["name"] + ":" + row["kind"]
            if name not in counts or name in seen:
                raise ValueError("unknown or duplicate interval quantity")
            seen.add(name)
            available, covered = row["available"], row["covered"]
            if (not isinstance(available, (bool, np.bool_))
                    or not isinstance(covered, (bool, np.bool_))
                    or covered and not available):
                raise ValueError("invalid interval availability/coverage")
            counts[name]["available"] += int(available)
            counts[name]["covered"] += int(covered)
            counts[name]["delivered_and_covered"] += int(valid and covered)

    def frequency(count, total=planned):
        interval = binomial_interval(count, total, alpha) if total else None
        return {"count": count, "total": total, "interval": interval,
                "screen_passed": (None if coverage_floor is None or interval is None
                                  else interval[0] >= coverage_floor)}

    quantities = {}
    for name, row in counts.items():
        quantities[name] = {
            "available": row["available"], "unavailable": planned - row["available"],
            "coverage": frequency(row["covered"]),
            "delivery_and_coverage": frequency(row["delivered_and_covered"]),
            "conditional_coverage": {k: v for k, v in
                frequency(row["covered"], row["available"]).items() if k != "screen_passed"},
        }
    return {"planned": planned, "recorded": len(records), "unstarted": planned-len(records),
            "replication_identity": "implicit_diagnostic_order" if implicit_indices else "explicit_indices",
            "delivery": frequency(delivered), "quantities": quantities,
            "coverage_floor": coverage_floor, "interval_alpha": alpha,
            "interval_scope": "pointwise two-sided exact binomial; not simultaneous coverage",
            "independent_unit": "complete fit, one predeclared member; siblings never pooled",
            "missing_policy": "every unavailable or unstarted slot is unsuccessful",
            "ranking_supported": False, "sequential_coverage_established": False}

"""Post-run diagnostic accounting for independent controller replications."""
from __future__ import annotations

from collections import Counter

from .statistics import binomial_interval


def summarize_controller_records(records, planned, *, oracle_alpha, coverage_floor):
    """Keep caps, unavailable intervals and absent replications in denominators."""
    if (type(planned) is not int or planned < 1 or not 0 < oracle_alpha < 1
            or not 0 < coverage_floor < 1):
        raise ValueError("positive planned count and valid oracle alpha required")
    indices = [row["rep"] for row in records]
    if (any(type(i) is not int or not 0 <= i < planned for i in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("replications must have distinct predeclared indices")
    complete_rows = [row for row in records if row.get("status") == "complete"]
    def frequency(count, total=planned, *, alpha=.05):
        return {"count": count, "total": total,
                "interval": binomial_interval(count, total, alpha) if total else None}
    intervals = {}
    for arm in ("stopped", "fixed"):
        rows = [r.get(arm, {}) for r in records]
        available = sum(bool(row.get("available")) for row in rows)
        covered = sum(bool(row.get("available") and row.get("covered")) for row in rows)
        intervals[arm] = {"available": available, "unavailable": planned - available,
                          "unconditional": frequency(covered),
                          "conditional": frequency(covered, available),
                          "reported_interval_screen_passed": frequency(covered)["interval"][0] >= coverage_floor}
    oracle = [row.get("fixed_oracle", {}) for row in records]
    oracle_available = sum(bool(row.get("available")) for row in oracle)
    oracle_covered = sum(bool(row.get("available") and row.get("covered_transient_expectation")) for row in oracle)
    oracle_interval = binomial_interval(oracle_covered, planned, oracle_alpha)
    passed = sum(bool(row.get("passed")) for row in complete_rows)
    readiness = frequency(passed)
    return {
        "planned": planned, "recorded": len(records), "complete": len(complete_rows) == planned,
        "unstarted": planned - len(records),
        "implementation_failures": sum(row.get("status") == "implementation_failure" for row in records),
        "hard_veto_replications": sum(bool(row.get("hard_vetoes")) for row in records),
        "posterior_checks_passed": readiness,
        "posterior_delivery_screen_passed": readiness["interval"][0] >= coverage_floor,
        "declared_coverage_floor": coverage_floor,
        "warmup_cap_count": sum(bool(row.get("warmup_cap")) for row in records),
        "retained_cap_count": sum(bool(row.get("retained_cap")) for row in records),
        "warmup_counts": dict(sorted(Counter(row.get("warmup_count") for row in records
                                             if row.get("warmup_count") is not None).items())),
        "retained_counts": dict(sorted(Counter(row.get("retained_count") for row in records
                                               if row.get("retained_count") is not None).items())),
        "intervals": intervals,
        "oracle": {"available": oracle_available, "covered_transient_expectation": oracle_covered,
                   "interval": oracle_interval, "alpha": oracle_alpha,
                   "screen_passed": oracle_available == planned and oracle_interval[0] <= .95 <= oracle_interval[1],
                   "target": "exact Gaussian distribution about its fixed-count transient expectation"},
        "interval_interpretation": "reported intervals include precision-cap outcomes; posterior delivery is assessed separately",
        "coverage_denominator": "all predeclared replications; unavailable intervals are not covered",
        "ranking_supported": False, "anytime_coverage_established": False,
    }

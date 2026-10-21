"""Current method scope over the immutable pre-correction target inventory.

The old inventory is hashed by ongoing numerical runs and saved references.
Use this module for current comparison requirements. Do not rewrite old hashes
or count the excluded NUTS rows as outstanding scientific work.
"""
from __future__ import annotations

from copy import deepcopy

from . import modern_hmc_matrix as historical
from .modern_hmc_matrix import NONLINEAR_COMPARISON_TARGETS

SCOPE_ID = "bayesfilter.modern_hmc_owner_scope.2026_10_10"
CORE_METHODS = ("fixed_hmc", "randomized_hmc", "snaper_hmc", "mams")
CLI_METHODS = ("fixed", "randomized", "snaper", "mams")


def initial_comparison_matrix(methods=CORE_METHODS):
    if any(method not in CORE_METHODS for method in methods):
        raise ValueError("method excluded from the active HMC comparison scope")
    return historical.initial_comparison_matrix(methods=methods)


def validate_comparison_matrix(rows, *, methods=CORE_METHODS):
    initial_comparison_matrix(methods)
    result = historical.validate_comparison_matrix(rows, methods=methods)
    return {**result, "scope_id": SCOPE_ID, "excluded_methods": ["nuts"]}


def apply_owner_scope(snapshot):
    """Preserve all original outcomes while separating excluded historical rows."""
    result = deepcopy(snapshot)
    active, excluded, seen = [], [], set()
    expected = {(row["target"], row["method"]) for row in initial_comparison_matrix()}
    targets = {target for target, _ in expected}
    for row in result["rows"]:
        pair = row["target"], row["method"]
        if pair in seen:
            raise ValueError("duplicate comparison cell")
        seen.add(pair)
        if pair in expected:
            active.append(row)
        elif row["method"] == "nuts" and row["target"] in targets:
            excluded.append(row)
        else:
            raise ValueError("unknown comparison cell")
    if {(row["target"], row["method"]) for row in active} != expected:
        raise ValueError("required active comparison cells are missing")
    result.update(
        scope_id=SCOPE_ID,
        rows=active,
        planned_core_cells=len(expected),
        historical_excluded_rows=[*result.get("historical_excluded_rows", []), *excluded],
        exclusion_reason="owner_excludes_nuts_including_diagnostic_comparators",
        historical_costs_preserved=True,
        scientific_promotion_authority=False,
    )
    return result

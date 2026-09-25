"""Measured contributions to the declared q20 training reservation.

Partial sums can reject affordability under this rule. They cannot admit a
campaign or establish a lower statistical bound on adaptive runtime.
"""
from __future__ import annotations

import math

from bayesfilter.inference.q20_production_config import training_cohort


def training_reservation(config, rows):
    training = config["training"]
    batch = training["batch_size"]
    indexed = {}
    for row in rows:
        key = (row["width"], row["batch_size"], row["beta"])
        if (key in indexed or row["width"] not in training["widths"]
                or row["batch_size"] not in training["pricing_batches"]
                or row["beta"] not in training["betas"][1:]):
            raise ValueError("duplicate or unsupported training pricing scope")
        for field in ("first_update_seconds", "steady_update_seconds",
                      "heldout_first_seconds", "heldout_seconds_per_batch"):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError("training prices must be finite and positive")
        indexed[key] = row
    validation_batches = sum((n + batch - 1) // batch
                             for n in config["validation"]["bank_sizes"])
    lower = upper = validation = 0.
    missing, measured = set(), set()
    for candidate in training_cohort(config):
        for beta in training["betas"][1:]:
            if candidate["schedule"] == "direct" and beta != 1.:
                continue
            key = (candidate["width"], batch, beta)
            row = indexed.get(key)
            if row is None:
                missing.add(key)
                continue
            measured.add(key)
            lower += row["first_update_seconds"] + row["steady_update_seconds"] * training["cohort_min_updates"]
            upper += row["first_update_seconds"] + row["steady_update_seconds"] * training["rungs"][-1]
            validation += row["heldout_first_seconds"] + validation_batches * row["heldout_seconds_per_batch"]
    validation *= 3 * len(training["rungs"])
    factor = config["budget"]["forecast_safety_factor"]
    return {"minimum_cohort_seconds": factor * (lower + validation),
            "full_training_cap_seconds": factor * (upper + validation),
            "raw_optimizer_floor_seconds": lower,
            "raw_all_rungs_validation_reserve_seconds": validation,
            "measured_training_scopes": [list(key) for key in sorted(measured)],
            "missing_training_scopes": [list(key) for key in sorted(missing)],
            "forecast_safety_factor": factor, "factor_status": "engineering_hypothesis",
            "minimum_cohort_seconds_role": "optimizer_floor_plus_all_planned_validation_reserves",
            "runtime_bound_claim": "not_a_statistical_lower_bound_on_adaptive_runtime",
            "full_campaign_priced": False,
            "remaining_to_price": ["scalar_HMC", "classical_preparation", "ensemble", "reference", "repair"],
            "status": "partial_training_reservation" if missing else "training_only_forecast"}

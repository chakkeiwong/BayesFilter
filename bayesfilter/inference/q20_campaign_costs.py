"""Distinct calibration, floor and cap reservations from actual training work."""
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
        if "setup_seconds" in row and (not math.isfinite(row["setup_seconds"]) or row["setup_seconds"] < 0):
            raise ValueError("training setup price must be finite and nonnegative")
        if type(row.get("heldout_first_batches", 2)) is not int or row.get("heldout_first_batches", 2) < 1:
            raise ValueError("heldout first call must declare a positive batch count")
        indexed[key] = row
    first_bank, largest_bank = config["validation"]["bank_sizes"][0], config["validation"]["bank_sizes"][-1]
    if first_bank % batch or largest_bank % batch:
        raise ValueError("validation prices require whole batches")
    floor_rungs = sum(n <= training["cohort_min_updates"] for n in training["rungs"])
    all_rungs = len(training["rungs"])
    scenarios = {name: {"optimizer_seconds": 0., "validation_seconds": 0., "setup_seconds": 0.,
                        "optimizer_updates": 0, "target_validation_rows": 0, "map_graphs": 0,
                        "training_scopes": 0}
                 for name in ("calibration", "floor_first_bank", "floor_validation_cap", "full_cap")}
    missing, measured = set(), set()
    calibration_missing = set()
    def add(name, row, updates, maps, bank):
        record = scenarios[name]
        # The first timed call already executes one update (or two loss blocks).
        compile_loss = max(0., row["heldout_first_seconds"]
                           - row.get("heldout_first_batches", 2) * row["heldout_seconds_per_batch"])
        record["optimizer_seconds"] += row["first_update_seconds"] + (updates-1) * row["steady_update_seconds"]
        record["validation_seconds"] += maps * (compile_loss + (bank // batch) * row["heldout_seconds_per_batch"])
        record["setup_seconds"] += row.get("setup_seconds", 0.)
        record["optimizer_updates"] += updates
        record["target_validation_rows"] += maps * bank
        record["map_graphs"] += maps
        record["training_scopes"] += 1
    for candidate in training_cohort(config):
        for beta in training["betas"][1:]:
            if candidate["schedule"] == "direct" and beta != 1.:
                continue
            key = (candidate["width"], batch, beta)
            calibration = (candidate["root"] == training["roots"][0]
                and (candidate["schedule"] == "direct" or beta == training["betas"][1]))
            row = indexed.get(key)
            if row is None:
                missing.add(key)
                if calibration:
                    calibration_missing.add(key)
                continue
            measured.add(key)
            add("floor_first_bank", row, training["cohort_min_updates"], floor_rungs+1, first_bank)
            add("floor_validation_cap", row, training["cohort_min_updates"], floor_rungs+1, largest_bank)
            add("full_cap", row, training["rungs"][-1], all_rungs+1, largest_bank)
            if calibration:
                add("calibration", row, training["rungs"][0], 2, first_bank)
    factor = config["budget"]["forecast_safety_factor"]
    for record in scenarios.values():
        record["raw_seconds"] = sum(record[key] for key in ("optimizer_seconds", "validation_seconds", "setup_seconds"))
        record["reserved_seconds"] = factor * record["raw_seconds"]
    floor, cap = scenarios["floor_validation_cap"], scenarios["full_cap"]
    return {"schema": "bayesfilter.q20.training_cost_scenarios.v2", "scenarios": scenarios,
            "calibration_seconds": scenarios["calibration"]["reserved_seconds"],
            "missing_calibration_scopes": [list(key) for key in sorted(calibration_missing)],
            "minimum_cohort_seconds": floor["reserved_seconds"],
            "full_training_cap_seconds": cap["reserved_seconds"],
            "raw_optimizer_floor_seconds": floor["optimizer_seconds"],
            "raw_all_rungs_validation_reserve_seconds": cap["validation_seconds"],
            "measured_training_scopes": [list(key) for key in sorted(measured)],
            "missing_training_scopes": [list(key) for key in sorted(missing)],
            "forecast_safety_factor": factor, "factor_status": "engineering_hypothesis",
            "minimum_cohort_seconds_role": "compatibility_name_for_floor_with_validation_cap_reservation_not_minimum",
            "runtime_bound_claim": "not_a_statistical_lower_bound_on_adaptive_runtime",
            "full_campaign_priced": False,
            "remaining_to_price": ["batched_HMC_across_L_and_maps", "classical_preparation", "ensemble", "reference", "repair"],
            "status": "partial_training_reservation" if missing else "training_only_forecast"}

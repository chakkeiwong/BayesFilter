"""Distinct calibration, floor and cap reservations from actual training work."""
from __future__ import annotations

import math

from bayesfilter.inference.q20_production_config import training_cohort


def training_reservation(config, rows, *, checkpoint=None, method=None):
    from bayesfilter.inference.q20_training_resume import cached_loss_rows, read_training_checkpoint
    from bayesfilter.inference.neutra_post_training import post_training_settings, PROBE_SCHEMA
    probe_settings = post_training_settings(config)
    training = config["training"]
    saved = {} if checkpoint is None else read_training_checkpoint(checkpoint, config)["cohort"]
    cached = {} if checkpoint is None else cached_loss_rows(checkpoint, config)
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
        for field in ("post_training_first_batch_seconds", "post_training_steady_batch_seconds",
                      "post_training_summary_seconds"):
            if field in row and (not math.isfinite(row[field]) or row[field] <= 0):
                raise ValueError("post-training diagnostic prices must be finite and positive")
        indexed[key] = row
    first_bank, largest_bank = config["validation"]["bank_sizes"][0], config["validation"]["bank_sizes"][-1]
    if first_bank % batch or largest_bank % batch:
        raise ValueError("validation prices require whole batches")
    floor_rungs = sum(n <= training["cohort_min_updates"] for n in training["rungs"])
    all_rungs = len(training["rungs"])
    scenarios = {name: {"optimizer_seconds": 0., "validation_seconds": 0., "setup_seconds": 0.,
                        "post_training_seconds": 0., "post_training_points": 0,
                        "optimizer_updates": 0, "target_validation_rows": 0, "map_graphs": 0,
                        "training_scopes": 0, "credited_updates": 0, "credited_validation_rows": 0}
                 for name in ("calibration", "floor_first_bank", "floor_validation_cap", "full_cap")}
    missing, measured = set(), set()
    calibration_missing = set()
    def add(name, row, updates, maps, bank, candidate, beta):
        record = scenarios[name]
        item = saved.get(candidate["id"])
        done, assessed = 0, []
        if item:
            assessed = [a["updates"] for a in item["assessments"] if a["beta"] == beta]
            done = max(assessed, default=0)
            if item["session"]["map"]["beta"] == beta:
                done = max(done, item["session"]["level_updates"])
        credited = min(updates, done)
        record["credited_updates"] += credited
        pending_rungs = [n for n in training["rungs"] if n <= updates and not any(a >= n for a in assessed)]
        # A migrated warm start has two distinct initial validation maps:
        # original initialization and imported learned state. Neither is cached.
        if (item and not done and item.get("baseline") and item.get("previous")
                and item["session"]["map"]["beta"] == beta
                and item["baseline"]["map"]["transport_state_hash"] != item["previous"]["map"]["transport_state_hash"]):
            maps += 1
        missing_rows = [bank] * maps
        if item and done:
            pending_rungs = [n for n in pending_rungs if n >= done and not
                             (n == done and item.get("reassessment_requires_update", False))]
            missing_rows = [bank] * sum(n > done for n in pending_rungs)
            if pending_rungs:
                known = {}
                if item["session"]["map"]["beta"] == beta:
                    keys = ["baseline", "previous"] + (["session"] if done in pending_rungs else [])
                    for key in keys:
                        if item.get(key) is not None:
                            m = item[key]["map"]
                            known[m["checkpoint_hash"]] = min(bank, cached.get(m["checkpoint_hash"], 0))
                missing_rows += [bank-n for n in known.values()] if known else [bank]
            record["credited_validation_rows"] += maps * bank - sum(missing_rows)
        updates -= credited
        geometry_rungs = set(pending_rungs)
        if item and done and item["session"]["map"]["beta"] == beta:
            if not any(a["updates"] == done and a["beta"] == beta and
                    a.get("post_training", {}).get("geometry", {}).get("schema") == PROBE_SCHEMA and
                    a["post_training"]["geometry"].get("rows") == probe_settings["rows"] and
                    a["post_training"]["geometry"].get("complete") is True for a in item["assessments"]):
                geometry_rungs.add(done)
        geometry_count = len(geometry_rungs)
        geometry_priced = (row.get("post_training_probe_schema") == PROBE_SCHEMA and
            row.get("post_training_points") == probe_settings["rows"] and
            row.get("post_training_batch_size") == probe_settings["batch_size"] and
            all(key in row for key in ("post_training_first_batch_seconds",
                "post_training_steady_batch_seconds", "post_training_summary_seconds")))
        if not geometry_priced:
            key = (row["width"], row["batch_size"], beta)
            missing.add(key)
            if name == "calibration":
                calibration_missing.add(key)
        if not updates and not any(missing_rows) and not geometry_count:
            return
        if geometry_priced:
            # Sanity pilots may use fewer points. The serious batch measurement
            # remains a conservative setup/throughput estimate for that exception.
            settings = post_training_settings(config, sanity_only=name == "calibration")
            blocks = math.ceil(settings["rows"]/row["post_training_batch_size"])
            record["post_training_seconds"] += geometry_count*(row["post_training_first_batch_seconds"]
                + (blocks-1)*row["post_training_steady_batch_seconds"] + row["post_training_summary_seconds"])
            record["post_training_points"] += geometry_count*settings["rows"]
        # The first timed call already executes one update (or two loss blocks).
        compile_loss = max(0., row["heldout_first_seconds"]
                           - row.get("heldout_first_batches", 2) * row["heldout_seconds_per_batch"])
        if updates:
            record["optimizer_seconds"] += row["first_update_seconds"] + (updates-1) * row["steady_update_seconds"]
        record["validation_seconds"] += sum(compile_loss + (n // batch) * row["heldout_seconds_per_batch"]
                                            for n in missing_rows if n)
        record["setup_seconds"] += row.get("setup_seconds", 0.)
        record["optimizer_updates"] += updates
        record["target_validation_rows"] += sum(missing_rows)
        record["map_graphs"] += sum(bool(n) for n in missing_rows)
        record["training_scopes"] += 1
    for candidate in training_cohort(config, method=method):
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
            add("floor_first_bank", row, training["cohort_min_updates"], floor_rungs+1, first_bank, candidate, beta)
            add("floor_validation_cap", row, training["cohort_min_updates"], floor_rungs+1, largest_bank, candidate, beta)
            add("full_cap", row, training["rungs"][-1], all_rungs+1, first_bank, candidate, beta)
            if calibration:
                add("calibration", row, training["rungs"][0], 2, first_bank, candidate, beta)
    factor = config["budget"]["forecast_safety_factor"]
    for record in scenarios.values():
        record["raw_seconds"] = sum(record[key] for key in (
            "optimizer_seconds", "validation_seconds", "setup_seconds", "post_training_seconds"))
        record["reserved_seconds"] = factor * record["raw_seconds"]
    scenarios["floor_validation_cap"]["role"] = "inactive_large_bank_counterfactual_not_admission_reservation"
    floor, cap = scenarios["floor_first_bank"], scenarios["full_cap"]
    return {"schema": "bayesfilter.q20.training_cost_scenarios.v4", "scenarios": scenarios, "method": method,
            "post_training_points_per_map": probe_settings["rows"],
            "validation_policy": "bounded_learning_screen_for_hmc_trial_v1",
            "validation_rows_per_map": first_bank,
            "checkpoint_credit_source": None if checkpoint is None else str(checkpoint),
            "calibration_seconds": scenarios["calibration"]["reserved_seconds"],
            "missing_calibration_scopes": [list(key) for key in sorted(calibration_missing)],
            "minimum_cohort_seconds": floor["reserved_seconds"],
            "full_training_cap_seconds": cap["reserved_seconds"],
            "raw_optimizer_floor_seconds": floor["optimizer_seconds"],
            "raw_all_rungs_validation_reserve_seconds": cap["validation_seconds"],
            "measured_training_scopes": [list(key) for key in sorted(measured)],
            "missing_training_scopes": [list(key) for key in sorted(missing)],
            "forecast_safety_factor": factor, "factor_status": "engineering_hypothesis",
            "minimum_cohort_seconds_role": "training_floor_with_bounded_validation_not_time_to_posterior",
            "runtime_bound_claim": "not_a_statistical_lower_bound_on_adaptive_runtime",
            "full_campaign_priced": False,
            "remaining_to_price": ["batched_HMC_across_L_and_maps", "reference", "repair"] +
                (["ensemble"] if method != "neutra" else []),
            "status": "partial_training_reservation" if missing else "training_only_forecast"}

"""Reproducible work-count scenarios using explicitly identified GPU prices.

Historical timing reuse is an estimate, not current-source price admission.
This standard-library report does not execute TensorFlow or any GPU workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bayesfilter.inference.q20_campaign_costs import training_reservation
from bayesfilter.inference.q20_production_config import protocol_template, training_cohort


def legacy_quote(config, rows):
    indexed = {(row["width"], row["batch_size"], row["beta"]): row for row in rows}
    training = config["training"]
    batches = sum(n // training["batch_size"] for n in config["validation"]["bank_sizes"])
    optimizer = validation = 0.
    for candidate in training_cohort(config):
        for beta in training["betas"][1:]:
            if candidate["schedule"] == "direct" and beta != 1.:
                continue
            row = indexed[(candidate["width"], training["batch_size"], beta)]
            optimizer += row["first_update_seconds"] + training["cohort_min_updates"] * row["steady_update_seconds"]
            validation += row["heldout_first_seconds"] + batches * row["heldout_seconds_per_batch"]
    validation *= 3 * len(training["rungs"])
    return {"optimizer_seconds": optimizer, "validation_seconds": validation,
            "reserved_seconds": config["budget"]["forecast_safety_factor"] * (optimizer+validation),
            "role": "historical_mixed_floor_and_all_rungs_rule"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowance", type=Path, required=True)
    parser.add_argument("--historical-timings", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = protocol_template()
    receipts = sorted(args.pricing_root.glob("price-w*-b32-beta*.json"))
    rows = [json.loads(path.read_text()) for path in receipts]
    quote = training_reservation(config, rows)
    if quote["missing_training_scopes"]:
        raise ValueError("budget report requires every declared training price scope")
    allowance = json.loads(args.allowance.read_text())
    result = {"schema": "bayesfilter.q20.validation_repair_budget_report.v1",
        "timing_basis": "historical_GPU_measurements_new_work_counts" if args.historical_timings else "current_GPU_measurements",
        "plan_file": "docs/plans/bayesfilter-ssl-lstm-q20-validation-budget-repair-plan-2026-09-16.md",
        "price_receipts": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in receipts],
        "allowance_path": str(args.allowance.resolve()),
        "available_campaign_hours": allowance["campaign_remaining_seconds"]/3600,
        "available_diagnostic_hours_included": allowance["diagnostic_remaining_seconds"]/3600,
        "legacy": legacy_quote(config, rows), "repaired": quote,
        "full_campaign_priced": False, "affordability_admission": False,
        "unpriced_additions": ["HMC tuning across actual L and learned maps", "classical preparation",
            "posterior and confirmation chains", "replica exchange and chart mixtures", "reference",
            "cache storage/checkpoint and process overhead"],
        "limitations": ["update and validation counts are hypotheses, not demonstrated required training",
            "first-bank floor scenario assumes every assessment resolves without expanding",
            "cap scenario assumes every history reaches its maximum and every needed map expands fully",
            "factor two is an engineering reserve, not a statistical runtime bound",
            "no guarantee of posterior qualification at any priced cap"]}
    if any("setup_seconds" not in row for row in rows):
        result["unpriced_additions"].append("training initialization and beta-change preflight")
    if args.historical_timings:
        result["limitations"].append("fresh source timing was unavailable; changed cache and decision logic were tested separately")
    for scenario in quote["scenarios"].values():
        scenario["raw_hours"] = scenario["raw_seconds"]/3600
        scenario["reserved_hours"] = scenario["reserved_seconds"]/3600
    (args.output_dir / "budget.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# q20 budget after validation repair", "", "Timing basis: `" + result["timing_basis"] + "`.", "",
        "| Scope | Updates | Validation target rows | Raw hours | Hours with factor 2 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in quote["scenarios"].items():
        lines.append(f"| {name} | {row['optimizer_updates']:,} | {row['target_validation_rows']:,} | {row['raw_hours']:.3f} | {row['reserved_hours']:.3f} |")
    lines += ["", f"Old mixed floor/all-rungs reservation: {result['legacy']['reserved_seconds']/3600:.3f} hours.", "",
        f"Available campaign allowance: {result['available_campaign_hours']:.3f} hours, including "
        f"{result['available_diagnostic_hours_included']:.3f} diagnostic hours.", "",
        "These are training scenarios, not a complete campaign quote. Unpriced work:", ""]
    lines += ["- " + item for item in result["unpriced_additions"]]
    lines += ["", "Interpretation:", ""] + ["- " + item for item in result["limitations"]]
    (args.output_dir / "budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

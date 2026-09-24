"""Diagnostic planning arithmetic only; no inference or runtime decisions.

Reproduce exact binomial-screen operating characteristics and descriptive cost
scenarios. This program simulates no chains, imports no array framework, and
does not infer power or coverage from oracle assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys
import time


def binomial_pmf(n, p):
    return [math.comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n + 1)]


def lower_screen(n, floor, tail_alpha=.025):
    probabilities = binomial_pmf(n, floor)
    return next(k for k in range(n + 1)
                if math.fsum(probabilities[k:]) <= tail_alpha)


def upper_screen(n, ceiling, tail_alpha=.025):
    probabilities = binomial_pmf(n, ceiling)
    return max(k for k in range(n + 1)
               if math.fsum(probabilities[:k + 1]) <= tail_alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    sources = {}

    def read(relative):
        path = args.repo / relative
        payload = path.read_bytes()
        sources[relative] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    old_root = "docs/plans/artifacts/hmc-repair-master-2026-09-16"
    old = read(old_root + "/m31-r1/design-arithmetic.json")
    for n, k in ((64, 63), (128, 122), (256, 240), (384, 358)):
        assert lower_screen(n, .90) == k
        assert upper_screen(n, .10) == n - k
        assert abs(math.fsum(binomial_pmf(n, .95)) - 1) < 1e-12
    for row in old["coverage"]:
        n, k = row["fits"], row["minimum_successes"]
        assert abs(math.fsum(binomial_pmf(n, .95)[k:])
                   - row["assurance_at_095"]) < 1e-12

    prices = {}
    for target in ("gaussian", "beta-binomial"):
        root = old_root + f"/m30-r1/{target}-full-dynamic-gpu-r1"
        run = read(root + "/execution.json")
        fit = read(root + "/fits/replication-0000/pipeline.json")
        members = [m for m in fit["members"] if m["status"] == "assessed"]
        assert run["exit_code"] == 0 and len(members) == 1
        comparator = members[0]["fixed_comparator"]["elapsed_seconds"]
        prices[target] = {
            "observed_full_fit_seconds": run["elapsed_seconds"],
            "observed_fixed_comparator_seconds": comparator,
            "subtraction_only_no_fixed_scenario_seconds": run["elapsed_seconds"] - comparator,
            "scope": "one historical fit; subtraction is not a new measured execution price",
        }

    coverage = []
    for n in (64, 128, 256, 384):
        k = lower_screen(n, .90)
        coverage.append({
            "fits_per_target": n, "minimum_successes": k,
            "assurance_at_actual_joint_probabilities": {
                str(p): math.fsum(binomial_pmf(n, p)[k:])
                for p in (.90, .93, .94, .95, .97)
            },
            "two_target_full_fit_gpu_hours": sum(
                n * p["observed_full_fit_seconds"] for p in prices.values()) / 3600,
            "two_target_no_fixed_subtraction_scenario_gpu_hours": sum(
                n * p["subtraction_only_no_fixed_scenario_seconds"]
                for p in prices.values()) / 3600,
        })

    detector = []
    for n in (64, 128):
        lo, hi = lower_screen(n, .80), upper_screen(n, .10)
        detector.append({
            "independent_fits": n, "minimum_detected": lo,
            "maximum_null_alarms_including_unavailable": hi,
            "null_screen_assurance": {
                str(p): math.fsum(binomial_pmf(n, p)[:hi + 1])
                for p in (.01, .025, .05)
            },
            "power_screen_assurance": {
                str(p): math.fsum(binomial_pmf(n, p)[lo:])
                for p in (.90, .95, .99)
            },
        })

    normal = NormalDist()
    critical = normal.inv_cdf(1 - .01 / 2)
    ideal_powers = {
        str(delta): normal.cdf(-critical - delta/.05)
        + 1 - normal.cdf(critical - delta/.05)
        for delta in (.25, .5)
    }
    activation = {}
    for name in ("baseline", "quarter", "half"):
        run = read(old_root + f"/m34-r1/{name}-dynamic-gpu-r1/execution.json")
        assert run["exit_code"] == 0
        activation[name] = run["elapsed_seconds"]
    detector_hours = (128 * activation["baseline"]
                      + 64 * (activation["quarter"] + activation["half"])) / 3600
    opening = read(old_root + "/m38-r1/program-reconciliation.json")["remaining"]
    coverage256 = next(row for row in coverage if row["fits_per_target"] == 256)
    core_hours = coverage256["two_target_no_fixed_subtraction_scenario_gpu_hours"] + detector_hours
    available_gpu = opening["gpu"] / 3600
    payload = {
        "schema": "bayesfilter.hmc_remaining_gap_design.v1",
        "scope": "deterministic design arithmetic; no new HMC or achieved-power evidence",
        "planning_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "python": sys.version.split()[0],
        "gpu_status": "not queried or used; no framework imports",
        "plan": "docs/plans/bayesfilter-hmc-remaining-gap-program-2026-09-24.md",
        "confidence": {"interval": "two-sided exact binomial", "level": .95,
                       "tail_probability": .025},
        "observed_prices_and_subtraction_scenarios": prices,
        "coverage_designs": coverage,
        "per_fit_detector_designs": detector,
        "per_fit_detector_hypothesis": {
            "nominal_alarm_alpha": .01, "critical_standardized_error": critical,
            "assumed_mcse_over_posterior_sd": .05,
            "ideal_gaussian_error_power": ideal_powers,
            "provenance": "new development hypotheses, not established stopped-HMC laws",
        },
        "historical_normal_activation_fit_seconds": activation,
        "r3_128_null_64_each_shift_transferred_cost_gpu_hours": detector_hours,
        "r2_256_each_plus_r3_subtraction_and_transfer_scenario_gpu_hours": core_hours,
        "available_allowance_seconds": opening,
        "scenario_shortfall_gpu_hours_before_repairs_and_training": max(0, core_hours - available_gpu),
        "design_checks": "M31 boundaries, tail probabilities and upper/lower symmetry reproduced",
        "source_sha256": sources,
        "limitations": [
            "Actual coverage events include missing outputs; .95 is a hypothesis, not a measured rate.",
            "Per-fit mean alarm answers a different question from the earlier endpoint/SBC power study.",
            "Nominal alarm alpha and ideal Gaussian power do not establish empirical false-positive rate or power.",
            "Historical runtimes have one fit per cell and different source/policies; no runtime bound or speedup claim.",
            "Removing a comparator changes the validation inventory, not the HMC method; new prices are still required.",
            "No new default, statistical ranking, general convergence or training-quality conclusion.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "coverage256": coverage256,
                      "detector_designs": detector, "core_gpu_hours_scenario": core_hours}))


if __name__ == "__main__":
    main()

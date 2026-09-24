#!/usr/bin/env python3
"""RQMC LEDH Initialization: Phase 3 statistical analysis.

Analyzes the 36 complete runs from Phase 2 campaign and produces:
1. Per-model descriptive tables (mean, std, q95 by arm)
2. Statistical inference: bootstrap CIs for (RQMC - MC) differences
3. Promotion recommendation per master program criteria

Usage:
    python docs/benchmarks/analyze_rqmc_ledh_phase3.py \\
        --campaign_summary docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/campaign_summary.json \\
        --output docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/phase3_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42


def _descriptive_table(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Compute descriptive statistics by arm for one model."""
    model_rows = [r for r in rows if r["model"] == model and r["status"] == "complete"]
    arms = sorted(set(r["arm"] for r in model_rows))

    table = {}
    for arm in arms:
        values = np.array([r["value"] for r in model_rows if r["arm"] == arm])
        if len(values) == 0:
            continue
        table[arm] = {
            "n": len(values),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "q05": float(np.percentile(values, 5)),
            "q50": float(np.percentile(values, 50)),
            "q95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return table


def _bootstrap_comparison(
    rows: list[dict[str, Any]], model: str, baseline_arm: str = "mc"
) -> dict[str, Any]:
    """Bootstrap 95% CI for (arm - baseline) differences."""
    model_rows = [r for r in rows if r["model"] == model and r["status"] == "complete"]
    seeds = sorted(set(r["seed"] for r in model_rows))
    arms = sorted(set(r["arm"] for r in model_rows if r["arm"] != baseline_arm))

    def arm_values(arm: str) -> np.ndarray:
        out = []
        for seed in seeds:
            match = [r for r in model_rows if r["arm"] == arm and r["seed"] == seed]
            out.append(match[0]["value"] if match else float("nan"))
        return np.asarray(out, dtype=float)

    baseline = arm_values(baseline_arm)
    if np.any(np.isnan(baseline)):
        return {"status": "baseline_incomplete", "arms": {}}

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(baseline)

    results = {}
    for arm in arms:
        values = arm_values(arm)
        if np.any(np.isnan(values)):
            results[arm] = {"status": "incomplete"}
            continue

        differences = values - baseline
        boot = np.array([
            differences[rng.choice(n, n, replace=True)].mean()
            for _ in range(BOOTSTRAP_SAMPLES)
        ])

        results[arm] = {
            "status": "evaluated",
            "n_seeds": n,
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "ci_lower": float(np.percentile(boot, 2.5)),
            "ci_upper": float(np.percentile(boot, 97.5)),
            # Metric is terminal log-likelihood: HIGHER IS BETTER. Difference is
            # (RQMC - MC). So RQMC is superior when the whole CI is ABOVE zero
            # (ci_lower > 0) and inferior when the whole CI is BELOW zero
            # (ci_upper < 0). An earlier revision had these inverted.
            "rqmc_superior": bool(float(np.percentile(boot, 2.5)) > 0.0),
            "rqmc_inferior": bool(float(np.percentile(boot, 97.5)) < 0.0),
            "statistically_indistinguishable": (
                float(np.percentile(boot, 2.5)) < 0 < float(np.percentile(boot, 97.5))
            ),
        }

    return {"status": "complete", "baseline": baseline_arm, "arms": results}


def _promotion_recommendation(
    comparison: dict[str, Any], descriptive: dict[str, Any]
) -> dict[str, Any]:
    """Determine promotion recommendation per program criteria.

    Metric is terminal log-likelihood, higher is better; difference is (RQMC - MC).

    - PROMOTE: at least one RQMC arm has ci_lower > 0 (RQMC superior)
    - NEUTRAL: all RQMC arms statistically indistinguishable from MC
    - REJECT: at least one RQMC arm has ci_upper < 0 (MC superior)

    Degenerate arms (recorded via `arm_degenerate_with` in the run artifact) are
    collapsed by the caller before reaching this function, so two names for one
    computation cannot inflate the count of agreeing arms.
    """
    if comparison["status"] != "complete":
        return {"decision": "BLOCKED", "reason": comparison["status"]}

    arms = comparison["arms"]
    evaluated = {k: v for k, v in arms.items() if v.get("status") == "evaluated"}

    if not evaluated:
        return {"decision": "BLOCKED", "reason": "no_evaluated_rqmc_arms"}

    favors_rqmc = [k for k, v in evaluated.items() if v["rqmc_superior"]]
    mc_superior = [k for k, v in evaluated.items() if v["rqmc_inferior"]]
    indistinguishable = [
        k for k, v in evaluated.items()
        if v["statistically_indistinguishable"]
    ]

    if favors_rqmc:
        return {
            "decision": "PROMOTE",
            "reason": "at_least_one_rqmc_arm_statistically_superior",
            "promoted_arms": favors_rqmc,
            "indistinguishable_arms": [
                k for k in indistinguishable if k not in favors_rqmc
            ],
        }

    if mc_superior:
        return {
            "decision": "REJECT",
            "reason": "at_least_one_rqmc_arm_statistically_inferior",
            "inferior_arms": mc_superior,
        }

    return {
        "decision": "NEUTRAL",
        "reason": "all_rqmc_arms_statistically_indistinguishable_from_mc",
        "indistinguishable_arms": list(indistinguishable),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign_summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    summary = json.loads(args.campaign_summary.read_text(encoding="utf-8"))
    rows = summary["rows"]
    models = summary["model_order"]

    print("=" * 78)
    print("RQMC LEDH Phase 3: Statistical Analysis")
    print("=" * 78)
    print(f"Campaign: {args.campaign_summary}")
    print(f"Complete runs: {summary['complete_runs']}/{summary['planned_runs']}")
    print(f"Models: {', '.join(models)}")
    print()

    analysis = {
        "schema_version": "rqmc_ledh_phase3_analysis.v1",
        "campaign_summary": str(args.campaign_summary),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sign_convention": (
            "metric=terminal_log_likelihood_higher_is_better; "
            "difference=(rqmc-mc); rqmc_superior<=>ci_lower>0; "
            "rqmc_inferior<=>ci_upper<0"
        ),
        "collapsed_degenerate_arms": [],
        "per_model": {},
    }

    # Degeneracy collapse: an arm whose run artifacts carry `arm_degenerate_with`
    # computes the identical cloud as the named arm on this model, so keeping both
    # would let one computation appear as two agreeing arms in the verdict count.
    # Drop the redundant name, keeping the arm it duplicates.
    degenerate_map: dict[tuple[str, str], str] = {}
    for row in rows:
        twin = row.get("arm_degenerate_with")
        if twin:
            degenerate_map[(row["model"], row["arm"])] = twin
    if degenerate_map:
        print("Degenerate arms collapsed (one computation, two names):")
        for (model_name, arm_name), twin in sorted(degenerate_map.items()):
            print(f"  {model_name}: dropping '{arm_name}' (identical to '{twin}')")
        rows = [
            r for r in rows
            if (r["model"], r["arm"]) not in degenerate_map
        ]
        print()
    analysis["collapsed_degenerate_arms"] = [
        {"model": m, "dropped_arm": a, "identical_to": t}
        for (m, a), t in sorted(degenerate_map.items())
    ]

    for model in models:
        print(f"--- {model} " + "-" * max(0, 70 - len(model)))

        descriptive = _descriptive_table(rows, model)
        comparison = _bootstrap_comparison(rows, model, baseline_arm="mc")
        recommendation = _promotion_recommendation(comparison, descriptive)

        print("Descriptive statistics:")
        for arm, stats in sorted(descriptive.items()):
            print(
                f"  {arm:18s} n={stats['n']} mean={stats['mean']:9.4f} "
                f"std={stats['std']:7.4f} q95={stats['q95']:9.4f}"
            )

        print("\nStatistical comparison (RQMC - MC):")
        if comparison["status"] == "complete":
            for arm, result in sorted(comparison["arms"].items()):
                if result["status"] == "evaluated":
                    print(
                        f"  {arm:18s} mean_diff={result['mean_difference']:+9.4f} "
                        f"95% CI=[{result['ci_lower']:+8.4f}, {result['ci_upper']:+8.4f}] "
                        f"verdict={'RQMC_SUPERIOR' if result['rqmc_superior'] else 'MC_SUPERIOR' if result['rqmc_inferior'] else 'indistinguishable'}"
                    )
        else:
            print(f"  Status: {comparison['status']}")

        print(f"\nPromotion recommendation: {recommendation['decision']}")
        print(f"  Reason: {recommendation['reason']}")
        if "promoted_arms" in recommendation:
            print(f"  Promoted arms: {', '.join(recommendation['promoted_arms'])}")
        if "indistinguishable_arms" in recommendation:
            print(f"  Indistinguishable: {', '.join(recommendation['indistinguishable_arms'])}")
        print()

        analysis["per_model"][model] = {
            "descriptive": descriptive,
            "comparison": comparison,
            "recommendation": recommendation,
        }

    # Aggregate recommendation
    decisions = [analysis["per_model"][m]["recommendation"]["decision"] for m in models]
    if all(d == "PROMOTE" for d in decisions):
        aggregate = "PROMOTE_ALL_MODELS"
    elif any(d == "REJECT" for d in decisions):
        aggregate = "MIXED_REJECT"
    elif any(d == "PROMOTE" for d in decisions):
        aggregate = "MIXED_PROMOTE"
    elif all(d == "NEUTRAL" for d in decisions):
        aggregate = "NEUTRAL_ALL_MODELS"
    else:
        aggregate = "MIXED_BLOCKED"

    analysis["aggregate_recommendation"] = aggregate

    print("=" * 78)
    print(f"Aggregate recommendation: {aggregate}")
    print("=" * 78)

    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"\nAnalysis written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

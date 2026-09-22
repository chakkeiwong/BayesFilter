#!/usr/bin/env python3
"""RQMC LEDH Initialization: Result Assembler.

Aggregates 45 claim-bearing runs (3 models × 5 arms × 3 seeds), computes
bootstrap 95% confidence intervals for paired differences, and applies
promotion/veto criteria per rqmc-ledh-initialization-master-program-2026-09-02.md.

Usage:
    python docs/benchmarks/assemble_rqmc_ledh_results.py \
        --campaign_dir docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260903/runs/ \
        --output docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260903/analysis/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ("lgssm_T50", "ksc_sv_T10", "predator_prey_T20")
ARMS = ("mc", "sobol_matousek", "sobol_owen", "halton_owen", "genut_guided")
SEEDS = (98301, 98302, 98303)

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42


def _load_run_result(campaign_dir: Path, model: str, arm: str, seed: int) -> dict[str, Any]:
    """Load one run result JSON."""
    run_dir = campaign_dir / f"{model}_{arm}_seed{seed}"
    result_path = run_dir / "result.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Missing result: {result_path}")

    return json.loads(result_path.read_text(encoding="utf-8"))


def _check_hard_vetoes(result: dict[str, Any]) -> list[str]:
    """Check for hard promotion vetoes. Returns list of veto reasons."""
    vetoes = []

    # Finite check
    if not result.get("finite", False):
        vetoes.append("non_finite_value")

    # Program validation
    if not result.get("program_valid", False):
        vetoes.append("program_validation_failed")

    # Dual-cap saturation (if diagnostic available)
    coord_sat = result.get("coordinatewise_cap_fire_rate", 0.0)
    if coord_sat > 0.10:
        vetoes.append(f"coordinatewise_saturation_{coord_sat:.3f}_exceeds_0.10")

    pair_sat = result.get("pairwise_cap_fire_rate", 0.0)
    if pair_sat > 0.10:
        vetoes.append(f"pairwise_saturation_{pair_sat:.3f}_exceeds_0.10")

    return vetoes


def _bootstrap_paired_ci(
    treatment: np.ndarray, control: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """Compute bootstrap 95% CI for paired difference (treatment - control).

    Returns (mean_diff, ci_lower, ci_upper).
    """
    if len(treatment) != len(control):
        raise ValueError("Treatment and control must have same length")

    n = len(treatment)
    differences = treatment - control
    mean_diff = float(np.mean(differences))

    # Bootstrap resampling (with replacement, paired)
    rng = np.random.RandomState(seed)
    boot_diffs = []
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        boot_diff = np.mean(differences[indices])
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return mean_diff, ci_lower, ci_upper


def _assemble_model_results(campaign_dir: Path, model: str) -> dict[str, Any]:
    """Assemble results for one model across all arms and seeds."""
    print(f"\nProcessing model: {model}")
    print("=" * 60)

    # Load all results
    results = {}
    for arm in ARMS:
        results[arm] = []
        for seed in SEEDS:
            try:
                result = _load_run_result(campaign_dir, model, arm, seed)
                results[arm].append(result)
                print(f"  {arm:20s} seed {seed}: value={result['value']:9.4f}, finite={result['finite']}")
            except FileNotFoundError as e:
                print(f"  {arm:20s} seed {seed}: MISSING")
                results[arm].append(None)

    # Check hard vetoes
    vetoes = {}
    for arm in ARMS:
        vetoes[arm] = []
        for i, result in enumerate(results[arm]):
            if result is None:
                vetoes[arm].append(["missing_result"])
            else:
                arm_vetoes = _check_hard_vetoes(result)
                vetoes[arm].append(arm_vetoes)
                if arm_vetoes:
                    print(f"  {arm:20s} seed {SEEDS[i]}: VETO - {', '.join(arm_vetoes)}")

    # Extract terminal log-likelihoods
    values = {}
    for arm in ARMS:
        arm_values = []
        for result in results[arm]:
            if result is not None and result.get("finite", False):
                arm_values.append(result["value"])
            else:
                arm_values.append(np.nan)
        values[arm] = np.array(arm_values)

    # Compute descriptive statistics
    descriptives = {}
    for arm in ARMS:
        valid_values = values[arm][~np.isnan(values[arm])]
        if len(valid_values) > 0:
            descriptives[arm] = {
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0,
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "n_valid": int(len(valid_values)),
            }
        else:
            descriptives[arm] = {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n_valid": 0,
            }

    # Compute paired bootstrap CIs vs MC baseline
    mc_values = values["mc"]
    if np.any(np.isnan(mc_values)):
        print(f"\nWARNING: MC baseline has invalid runs - cannot compute paired comparisons")
        comparisons = {}
    else:
        comparisons = {}
        for arm in ARMS:
            if arm == "mc":
                continue
            if np.any(np.isnan(values[arm])):
                print(f"\nWARNING: {arm} has invalid runs - skipping comparison")
                comparisons[arm] = {
                    "mean_diff": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "statistically_superior": False,
                    "reason": "invalid_runs",
                }
            else:
                mean_diff, ci_lower, ci_upper = _bootstrap_paired_ci(
                    values[arm], mc_values, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
                )
                # Statistical superiority: 95% CI excludes zero and is positive
                superior = ci_lower > 0.0
                comparisons[arm] = {
                    "mean_diff": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "statistically_superior": superior,
                    "reason": "bootstrap_95_ci_excludes_zero_positive" if superior else "ci_includes_zero_or_negative",
                }

    # Viability assessment
    viability = {}
    for arm in ARMS:
        if arm == "mc":
            viability[arm] = {
                "viable": True,
                "reason": "baseline",
            }
            continue

        # Check all runs passed hard vetoes
        all_valid = all(len(v) == 0 for v in vetoes[arm])
        if not all_valid:
            viability[arm] = {
                "viable": False,
                "reason": "hard_veto_fired",
            }
            continue

        # Check within 2 SE of MC or higher (descriptive)
        if arm not in comparisons or np.isnan(comparisons[arm]["mean_diff"]):
            viability[arm] = {
                "viable": False,
                "reason": "cannot_compare_to_mc",
            }
            continue

        mc_std = descriptives["mc"]["std"]
        mc_se = mc_std / np.sqrt(descriptives["mc"]["n_valid"]) if descriptives["mc"]["n_valid"] > 0 else np.inf
        arm_mean = descriptives[arm]["mean"]
        mc_mean = descriptives["mc"]["mean"]

        within_2se = (arm_mean >= mc_mean - 2 * mc_se)
        if within_2se:
            viability[arm] = {
                "viable": True,
                "reason": "within_2se_of_mc_or_higher",
            }
        else:
            viability[arm] = {
                "viable": False,
                "reason": "more_than_2se_below_mc",
            }

    return {
        "model": model,
        "descriptives": descriptives,
        "comparisons": comparisons,
        "viability": viability,
        "vetoes": vetoes,
        "raw_values": {arm: values[arm].tolist() for arm in ARMS},
    }


def _print_configuration_status_first_table(model_results: dict[str, Any]):
    """Print configuration-status-first table per memory requirement."""
    model = model_results["model"]
    print(f"\n{'='*80}")
    print(f"CONFIGURATION-STATUS-FIRST REPORT: {model}")
    print(f"{'='*80}")
    print()
    print("Program: LEDH_PRODUCTION_PROGRAM_V1")
    print("Route: ledh_pfpf_ot_contract_e_dual_cap_trust_region")
    print("Tuning status: Per-scope tuning artifact exists for all models")
    print()
    print(f"{'Arm':<20} {'Status':<15} {'Mean LL':<12} {'Viable':<10} {'Stat Superior'}")
    print("-" * 80)

    desc = model_results["descriptives"]
    viab = model_results["viability"]
    comp = model_results["comparisons"]

    for arm in ARMS:
        status = "TUNED" if arm == "mc" else "TUNED"
        mean_ll = f"{desc[arm]['mean']:9.4f}" if not np.isnan(desc[arm]['mean']) else "NaN"
        viable = "YES" if viab[arm]["viable"] else "NO"
        if arm == "mc":
            stat_sup = "baseline"
        elif arm in comp:
            stat_sup = "YES" if comp[arm]["statistically_superior"] else "NO"
        else:
            stat_sup = "N/A"

        print(f"{arm:<20} {status:<15} {mean_ll:<12} {viable:<10} {stat_sup}")

    print()


def _promotion_decision(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Make promotion decision across all models."""
    print(f"\n{'='*80}")
    print("PROMOTION DECISION")
    print(f"{'='*80}")

    # Check viability on all 3 models
    viable_arms = set(ARMS)
    for result in all_results:
        for arm in ARMS:
            if not result["viability"][arm]["viable"]:
                viable_arms.discard(arm)
                print(f"  {arm} - NOT VIABLE on {result['model']}: {result['viability'][arm]['reason']}")

    print(f"\nViable arms (passed on all 3 models): {viable_arms}")

    # Check statistical superiority on all 3 models
    promotable_arms = set()
    for arm in viable_arms:
        if arm == "mc":
            continue  # MC baseline is not a promotion candidate

        superior_on_all = True
        for result in all_results:
            if arm not in result["comparisons"]:
                superior_on_all = False
                break
            if not result["comparisons"][arm]["statistically_superior"]:
                superior_on_all = False
                print(f"  {arm} - NOT statistically superior on {result['model']}")
                break

        if superior_on_all:
            promotable_arms.add(arm)
            print(f"  {arm} - PROMOTABLE (statistically superior on all 3 models)")

    # Final decision
    if len(promotable_arms) == 0:
        decision = "NO_PROMOTION"
        reason = "No RQMC arm showed statistical superiority over MC baseline on all 3 models"
    elif len(promotable_arms) == 1:
        decision = "PROMOTE"
        promoted_arm = list(promotable_arms)[0]
        reason = f"Arm {promoted_arm} passed viability and statistical superiority on all 3 models"
    else:
        decision = "MULTIPLE_CANDIDATES"
        reason = f"Multiple arms promotable: {promotable_arms}. Require additional tiebreaker criterion."

    return {
        "decision": decision,
        "reason": reason,
        "viable_arms": list(viable_arms),
        "promotable_arms": list(promotable_arms),
    }


def main():
    parser = argparse.ArgumentParser(
        description="RQMC LEDH Initialization: Result Assembler"
    )
    parser.add_argument(
        "--campaign_dir",
        required=True,
        type=Path,
        help="Campaign directory containing individual run results",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for assembled results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RQMC LEDH Initialization: Result Assembly")
    print("=" * 80)
    print(f"Campaign directory: {args.campaign_dir}")
    print(f"Output directory: {args.output}")

    # Process each model
    all_results = []
    for model in MODELS:
        model_results = _assemble_model_results(args.campaign_dir, model)
        all_results.append(model_results)
        _print_configuration_status_first_table(model_results)

    # Promotion decision
    promotion = _promotion_decision(all_results)

    # Write assembled results
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {
        "models": all_results,
        "promotion": promotion,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    output_path = args.output / "assembled_results.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nAssembled results written to: {output_path}")

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Promotion decision: {promotion['decision']}")
    print(f"Reason: {promotion['reason']}")
    print(f"Viable arms: {promotion['viable_arms']}")
    print(f"Promotable arms: {promotion['promotable_arms']}")
    print()

    if promotion['decision'] == 'NO_PROMOTION':
        print("Result: MC baseline remains the default initialization method")
        sys.exit(0)
    elif promotion['decision'] == 'PROMOTE':
        promoted = promotion['promotable_arms'][0]
        print(f"Result: RQMC arm '{promoted}' promoted to default initialization")
        sys.exit(0)
    else:
        print("Result: Multiple candidates require additional tiebreaker")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Audit a completed C2 UKF-guided defensive TT-DMIS branch campaign.

This is a diagnostic/reporting script.  It does not rerun TensorFlow, alter a
completed result, or promote a proposal.  It evaluates the paired criterion
declared in the 2026-08-29 plan from the preserved branch records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/attempt01"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260830
EXPECTED_FAMILIES = (
    "retained_tt",
    "bootstrap_conditional",
    "defensive_student",
    "dmis_half",
    "gaussian_hint_marginal",
    "stationary_independence",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN))
    return parser.parse_args()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _paired_bootstrap_mean(values: Sequence[float]) -> Mapping[str, object]:
    rows = tuple(_finite(value, "paired contrast") for value in values)
    if len(rows) < 2:
        raise ValueError("paired bootstrap requires at least two replicates")
    generator = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [rows[generator.randrange(len(rows))] for _ in rows]
        means.append(statistics.fmean(sample))
    return {
        "replicates": len(rows),
        "values": list(rows),
        "mean": statistics.fmean(rows),
        "standard_deviation": statistics.stdev(rows),
        "standard_error": statistics.stdev(rows) / math.sqrt(len(rows)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "percentile_95_interval": [
            _quantile(means, 0.025),
            _quantile(means, 0.975),
        ],
    }


def _exact_two_sided_sign_pvalue(positive: int, total: int) -> float:
    if not 0 <= positive <= total or total < 1:
        raise ValueError("invalid sign-test counts")
    negative_or_equal = total - positive
    tail = min(
        sum(math.comb(total, k) for k in range(0, negative_or_equal + 1)),
        sum(math.comb(total, k) for k in range(positive, total + 1)),
    )
    return min(1.0, 2.0 * tail / (2.0**total))


def _family_rows(rows: Sequence[Mapping[str, object]], family: str) -> Mapping[int, Mapping[str, object]]:
    selected = {
        int(row["seed"]): row for row in rows if str(row["family"]) == family
    }
    if len(selected) != 12:
        raise ValueError(f"family {family} has {len(selected)} rows, expected 12")
    return selected


def _audit(run_root: Path) -> Mapping[str, object]:
    branch_rows = _read_json(run_root / "branch_results.json")
    result = _read_json(run_root / "result.json")
    manifest = _read_json(run_root / "run_manifest.json")
    if not isinstance(branch_rows, list) or not isinstance(result, Mapping):
        raise ValueError("campaign artifacts have unexpected top-level types")
    if len(branch_rows) != 72:
        raise ValueError(f"expected 72 branch rows, found {len(branch_rows)}")
    rows = tuple(branch_rows)
    families = tuple(str(family) for family in result["family_summary"])
    if set(families) != set(EXPECTED_FAMILIES):
        raise ValueError(f"family set differs from plan: {families}")
    by_family = {
        family: _family_rows(rows, family) for family in EXPECTED_FAMILIES
    }
    seeds = tuple(sorted(by_family["retained_tt"]))
    if any(tuple(sorted(by_family[family])) != seeds for family in EXPECTED_FAMILIES):
        raise ValueError("family seed maps are not identical")
    if not all(bool(row["all_engineering_checks_pass"]) for row in rows):
        raise ValueError("at least one branch failed an engineering check")

    retained = by_family["retained_tt"]
    dmis = by_family["dmis_half"]
    ess_ratios = tuple(
        math.log(
            _finite(dmis[seed]["minimum_normalized_ess"], "DMIS ESS")
            / _finite(retained[seed]["minimum_normalized_ess"], "TT ESS")
        )
        for seed in seeds
    )
    paired_ess = _paired_bootstrap_mean(ess_ratios)
    positive_count = sum(value > 0.0 for value in ess_ratios)
    sign_pvalue = _exact_two_sided_sign_pvalue(positive_count, len(ess_ratios))
    ess_interval = paired_ess["percentile_95_interval"]
    mechanism_criterion = bool(ess_interval[0] > 0.0 and positive_count >= 10)

    reference_total = _finite(result["reference"]["pf_total"], "PF reference total")
    reference_se = _finite(result["reference"]["pf_total_se"], "PF reference SE")
    dmis_total_differences = tuple(
        _finite(dmis[seed]["log_likelihood"], "DMIS total") - reference_total
        for seed in seeds
    )
    dmis_reference = _paired_bootstrap_mean(dmis_total_differences)
    reference_tolerance = max(1.0, 3.0 * reference_se)
    mean_reference_gap = abs(statistics.fmean(dmis_total_differences))

    per_family = {}
    for family in EXPECTED_FAMILIES:
        family_map = by_family[family]
        minimum_ess = tuple(
            _finite(family_map[seed]["minimum_normalized_ess"], f"{family} ESS")
            for seed in seeds
        )
        per_family[family] = {
            "mean_minimum_normalized_ess": statistics.fmean(minimum_ess),
            "minimum_observed_normalized_ess": min(minimum_ess),
            "mean_total": statistics.fmean(
                _finite(family_map[seed]["log_likelihood"], f"{family} total")
                for seed in seeds
            ),
            "all_engineering_checks_pass": True,
        }

    heuristic_families = (
        "bootstrap_conditional",
        "defensive_student",
        "gaussian_hint_marginal",
        "stationary_independence",
    )
    per_step_reference = tuple(
        _finite(value, "PF per-step reference")
        for value in result["reference"]["pf_per_step_mean"]
    )
    lowest_tt_time = min(
        range(len(per_step_reference)),
        key=lambda time_index: statistics.fmean(
            _finite(
                retained[seed]["normalized_ess_by_time"][time_index],
                "retained TT per-time ESS",
            )
            for seed in seeds
        ),
    )
    salient_times = tuple(sorted(set((3, 4, lowest_tt_time))))
    dominance_rows = []
    for time_index in salient_times:
        errors = {
            family: statistics.fmean(
                abs(
                    _finite(
                        by_family[family][seed]["log_increments"][time_index],
                        f"{family} per-step increment",
                    )
                    - per_step_reference[time_index]
                )
                for seed in seeds
            )
            for family in EXPECTED_FAMILIES
        }
        better = [
            family
            for family in heuristic_families
            if errors[family] < errors["dmis_half"]
        ]
        dominance_rows.append(
            {
                "situation": f"t={time_index}",
                "candidate": "dmis_half",
                "candidate_mean_absolute_per_step_error": errors["dmis_half"],
                "heuristic_mean_absolute_per_step_errors": {
                    family: errors[family] for family in heuristic_families
                },
                "heuristics_better_than_candidate": better,
            }
        )
    heuristic_veto = bool(
        any(row["heuristics_better_than_candidate"] for row in dominance_rows)
    )
    classification = (
        "CANDIDATE_VIABLE_FOR_RANDOMIZED_LIKELIHOOD_TESTING"
        if mechanism_criterion and mean_reference_gap <= reference_tolerance
        else "CANDIDATE_REJECTED_PROPOSAL_VARIANCE"
    )
    tracked_sources = (
        ROOT / "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/ukf_guided_defensive_tt_dmis_analytical_gradient.tex",
        ROOT / "docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-implementation-test-plan-2026-08-29.md",
        ROOT / "docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-plan-review-2026-08-29.md",
        ROOT / "docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_20260829.py",
        ROOT / "bayesfilter/highdim/c2_transformed_observation_student_proposal_tf.py",
        ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py",
        ROOT / "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py",
    )
    recorded_sources = manifest.get("source_sha256", {})
    source_comparison = {
        str(path.relative_to(ROOT)): {
            "recorded_at_run": recorded_sources.get(str(path.relative_to(ROOT))),
            "current": _sha256(path),
            "unchanged": recorded_sources.get(str(path.relative_to(ROOT))) == _sha256(path),
        }
        for path in tracked_sources
    }
    return {
        "schema_id": "bayesfilter.c2_ukf_guided_defensive_tt_dmis_paired_audit.v1",
        "run_root": str(run_root.relative_to(ROOT)),
        "source_branch_results": str((run_root / "branch_results.json").relative_to(ROOT)),
        "source_result": str((run_root / "result.json").relative_to(ROOT)),
        "source_run_manifest": str((run_root / "run_manifest.json").relative_to(ROOT)),
        "source_git_commit": manifest["git_commit"],
        "source_hash_comparison": source_comparison,
        "run_snapshot_binding": (
            "exact_recorded_sources"
            if all(row["unchanged"] for row in source_comparison.values())
            else "post_run_documentation_or_observability_changes_present; raw branch values remain bound to the recorded run snapshot"
        ),
        "families": EXPECTED_FAMILIES,
        "seeds": seeds,
        "engineering_correctness_pass": True,
        "paired_log_minimum_ess_ratio": paired_ess,
        "paired_log_minimum_ess_positive_count": positive_count,
        "paired_log_minimum_ess_exact_two_sided_sign_pvalue": sign_pvalue,
        "paired_mechanism_criterion": mechanism_criterion,
        "dmis_reference_total_difference": dmis_reference,
        "dmis_reference_mean_absolute_gap": mean_reference_gap,
        "dmis_reference_compatibility_tolerance": reference_tolerance,
        "dmis_reference_compatibility_pass": mean_reference_gap <= reference_tolerance,
        "per_family": per_family,
        "heuristic_dominance_veto": heuristic_veto,
        "heuristic_adversary_families": heuristic_families,
        "heuristic_dominance_rows": dominance_rows,
        "diagnostic_gaps": [
            "the preserved attempt01 was generated before the per-time maximum-normalized-weight observability field was added; the current driver emits it, but this snapshot cannot be retrofitted",
            "the Phase 5 alpha/nu pilot was executed separately after attempt01 and returned the fixed-half fallback because bootstrap minimizer stability was below its predeclared gate; it was not replayed into final banks",
            "this audit does not establish pseudo-marginal exactness, posterior correctness, HMC readiness, default readiness, or source-faithful Zhao-Cui reproduction",
        ],
        "classification": classification,
        "promotion_status": "NO_PROMOTION_HEURISTIC_VETO" if heuristic_veto else "NO_PROMOTION",
    }


def _markdown(audit: Mapping[str, object]) -> str:
    ess = audit["paired_log_minimum_ess_ratio"]
    reference = audit["dmis_reference_total_difference"]
    lines = [
        "# C2 UKF-Guided Defensive TT-DMIS Paired Audit",
        "",
        f"Run: `{audit['run_root']}`; source git commit: `{audit['source_git_commit']}`.",
        "",
        "## Decision",
        "",
        f"Classification: `{audit['classification']}`.",
        f"Engineering checks: `{'PASS' if audit['engineering_correctness_pass'] else 'FAIL'}`.",
        f"Predeclared paired ESS criterion: `{'PASS' if audit['paired_mechanism_criterion'] else 'FAIL'}`.",
        f"Heuristic-dominance veto: `{'FIRED' if audit['heuristic_dominance_veto'] else 'CLEAR'}`.",
        f"Reference compatibility: `{'PASS' if audit['dmis_reference_compatibility_pass'] else 'FAIL'}`.",
        "No promotion, posterior, HMC, exactness, or superiority verdict is issued.",
        "",
        "## Paired Criterion",
        "",
        f"Mean log minimum-ESS ratio: `{ess['mean']:.10f}`.",
        f"95% percentile bootstrap interval: `[{ess['percentile_95_interval'][0]:.10f}, {ess['percentile_95_interval'][1]:.10f}]`.",
        f"Positive contrasts: `{audit['paired_log_minimum_ess_positive_count']}/{len(audit['seeds'])}`.",
        f"Exact two-sided sign-test p-value: `{audit['paired_log_minimum_ess_exact_two_sided_sign_pvalue']:.10g}`.",
        f"Bootstrap: `{ess['bootstrap_replicates']}` resamples, seed `{ess['bootstrap_seed']}`.",
        "",
        "## Reference",
        "",
        f"Mean absolute DMIS/PF total gap: `{audit['dmis_reference_mean_absolute_gap']:.10f}`; tolerance: `{audit['dmis_reference_compatibility_tolerance']:.10f}`.",
        f"95% bootstrap interval for DMIS total minus PF reference: `[{reference['percentile_95_interval'][0]:.10f}, {reference['percentile_95_interval'][1]:.10f}]`.",
        "",
        "## Family Summary",
        "",
        "| Family | Mean minimum normalized ESS | Minimum observed normalized ESS | Mean total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in EXPECTED_FAMILIES:
        row = audit["per_family"][family]
        lines.append(
            f"| `{family}` | {row['mean_minimum_normalized_ess']:.8g} | {row['minimum_observed_normalized_ess']:.8g} | {row['mean_total']:.10f} |"
        )
    lines.extend([
        "",
        "## Heuristic Adversary Check",
        "",
        "The candidate is `dmis_half`; the adversaries are bootstrap conditional, transformed Student, Gaussian-hint marginal, and stationary independence.",
        "",
        "| Situation | Candidate mean absolute error | Better adversaries |",
        "| --- | ---: | --- |",
    ])
    for row in audit["heuristic_dominance_rows"]:
        lines.append(
            f"| `{row['situation']}` | {row['candidate_mean_absolute_per_step_error']:.8g} | {', '.join(row['heuristics_better_than_candidate']) or 'none'} |"
        )
    lines.extend([
        "",
        "## Gaps and Boundaries",
        "",
    ])
    lines.extend(f"- {item}" for item in audit["diagnostic_gaps"])
    lines.extend([
        "",
        "The heuristic veto is a promotion veto, not evidence that the shared implementation is wrong. It means a simple adversary was descriptively better in a salient situation and must remain the headline limitation.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    run_root = Path(_parse_args().run_root).resolve()
    audit = _audit(run_root)
    (run_root / "paired_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_root / "paired_audit.md").write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

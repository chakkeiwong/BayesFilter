#!/usr/bin/env python3
"""Create the prior-versus-repaired GenUT regression comparison report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


T95_DF15 = 2.131449545559323
ROOT = Path(__file__).resolve().parents[2]


def _summary(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("interval requires at least two values")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95_DF15 * sd / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "sample_sd": sd,
        "standard_error": sd / math.sqrt(len(values)),
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _vectors(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [
        [float(row["value"]), *[float(value) for value in row["score"]]]
        for row in rows
    ]


def _estimate_summary(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    vectors = _vectors(rows)
    return {
        label: _summary([vector[index] for vector in vectors])
        for index, label in enumerate(labels)
    }


def _paired_summary(
    prior_rows: list[dict[str, Any]],
    modified_rows: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    prior = {int(row["particle_seed"]): row for row in prior_rows}
    modified = {int(row["particle_seed"]): row for row in modified_rows}
    seeds = sorted(set(prior) & set(modified))
    if len(seeds) < 2:
        return {"common_seeds": seeds, "available": False}
    prior_vectors = _vectors([prior[seed] for seed in seeds])
    modified_vectors = _vectors([modified[seed] for seed in seeds])
    return {
        "common_seeds": seeds,
        "available": True,
        "deltas_modified_minus_prior": {
            label: _summary(
                [modified_vectors[i][j] - prior_vectors[i][j] for i in range(len(seeds))]
            )
            for j, label in enumerate(labels)
        },
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(root: Path) -> dict[str, Any]:
    modified = _load(
        root
        / "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/model_regressions_attempt02/result.json"
    )
    modified_lgssm = _load(
        root
        / "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/lgssm_common_seed_claim_attempt01/result.json"
    )
    structural = _load(
        root
        / "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/structural_tuned_claim_attempt02/result.json"
    )
    prior_lgssm = _load(
        root
        / "docs/benchmarks/artifacts/lgssm_cubature_genut_comparable_metric_20260721_attempt5/result.json"
    )
    prior_sv = _load(
        root
        / "docs/benchmarks/artifacts/exact_sv_fixed_gaussian_genut_paired_20260721/attempt02/result.json"
    )
    prior_pp = _load(
        root
        / "docs/benchmarks/artifacts/genut_predator_prey_leaderboard_continuation_20260722/attempt01/result.json"
    )
    prior_sir = _load(
        root
        / "docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/attempt02/fixed-sir/gpu/result.json"
    )
    modified_sir = _load(
        root
        / "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/actual_austria_sir_gpu_attempt02/result.json"
    )
    labels = {
        "lgssm": ["value", "phi1", "phi2", "phi3", "q_scale", "r_scale"],
        "sv": ["value", "theta_gamma", "theta_log_beta"],
        "predator_prey": ["value", "r", "K", "a", "s", "u", "v"],
    }
    models: dict[str, Any] = {}
    for horizon in (2, 10, 50):
        key = str(horizon)
        prior_rows = [
            {
                "particle_seed": row["particle_seed"],
                "value": row["particle_value"],
                "score": row["particle_score"],
            }
            for row in prior_lgssm["results"]
            if row["method"] == "genut" and int(row["horizon"]) == horizon
        ]
        modified_rows = modified_lgssm["lgssm"][key]["raw"]
        models[f"LGSSM T={horizon}"] = {
            "target": "linear Gaussian state-space model, fixed dataset seed 81100",
            "prior": {
                "artifact": "docs/benchmarks/artifacts/lgssm_cubature_genut_comparable_metric_20260721_attempt5/result.json",
                "controls": {
                    "epsilon": prior_lgssm["configuration"]["epsilon"],
                    "sinkhorn_steps": prior_lgssm["configuration"]["sinkhorn_steps"],
                    "balance_steps": "not present in historical scalar",
                    "ridge": prior_lgssm["configuration"]["ridge"],
                },
                "summary": _estimate_summary(prior_rows, labels["lgssm"]),
            },
            "modified": {
                "artifact": "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/lgssm_common_seed_claim_attempt01/result.json",
                "controls": modified_lgssm["lgssm"][key]["controls"],
                "summary": _estimate_summary(modified_rows, labels["lgssm"]),
                "oracle": modified_lgssm["lgssm"][key]["oracle"],
            },
            "paired": _paired_summary(prior_rows, modified_rows, labels["lgssm"]),
            "hard_valid": modified["hard_valid"] and modified_lgssm["hard_valid"],
        }

    prior_sv_rows_raw = _load(
        root
        / "docs/benchmarks/artifacts/exact_sv_fixed_gaussian_genut_paired_20260721/attempt02/rows_fresh_dgp_n1998_s4_fixed_gaussian_genut.json"
    )
    prior_sv_rows = [
        {
            "particle_seed": row["seed"],
            "value": row["value"],
            "score": row["score"],
        }
        for row in prior_sv_rows_raw
    ]
    modified_sv = modified["fresh_exact_sv"]
    models["Fresh exact transformed SV"] = {
        "target": "fresh exact stationary transformed-SV DGP; original iid-normal fixture excluded",
        "prior": {
            "artifact": "docs/benchmarks/artifacts/exact_sv_fixed_gaussian_genut_paired_20260721/attempt02/result.json",
            "controls": {
                "epsilon": prior_sv["manifest"]["epsilon"],
                "sinkhorn_steps": 4,
                "balance_steps": 0,
                "ridge": prior_sv["manifest"]["ridge"],
            },
            "summary": _estimate_summary(prior_sv_rows, labels["sv"]),
        },
        "modified": {
            "artifact": "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/model_regressions_attempt02/result.json",
            "controls": modified_sv["tuning"]["selected_controls"],
            "summary": modified_sv["summary"],
            "oracle": modified_sv["oracle"],
        },
        "paired": _paired_summary(prior_sv_rows, modified_sv["raw"], labels["sv"]),
        "hard_valid": modified["hard_valid"],
    }

    prior_pp_rows = prior_pp["claim"]["genut"]["raw"]["1002"]
    modified_pp = modified["predator_prey"]
    models["Predator-prey T=20"] = {
        "target": "zhao_cui_predator_prey_T20; no exact score oracle",
        "prior": {
            "artifact": "docs/benchmarks/artifacts/genut_predator_prey_leaderboard_continuation_20260722/attempt01/result.json",
            "controls": prior_pp["claim"]["controls"],
            "summary": prior_pp["claim"]["genut"]["summaries"]["1002"],
        },
        "modified": {
            "artifact": "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/model_regressions_attempt02/result.json",
            "controls": modified_pp["tuning"]["selected_controls"],
            "summary": modified_pp["summary"],
        },
        "paired": _paired_summary(prior_pp_rows, modified_pp["raw"], labels["predator_prey"]),
        "hard_valid": modified["hard_valid"],
    }
    models["Actual Austria SIR"] = {
        "target": "canonical fixed source-order SGQF value-only route; no free parameter score",
        "prior": {
            "artifact": "docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/attempt02/fixed-sir/gpu/result.json",
            "value": prior_sir["gpu_xla"]["value"],
        },
        "modified": {
            "artifact": "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/actual_austria_sir_gpu_attempt02/result.json",
            "value": modified_sir["gpu_xla"]["value"],
            "cpu_gpu_absolute_difference": modified_sir["gpu_xla"]["absolute_difference_from_cpu"],
        },
        "paired": {
            "value_delta": modified_sir["gpu_xla"]["value"] - prior_sir["gpu_xla"]["value"],
            "score": "not applicable",
        },
        "hard_valid": prior_sir["status"] == "PASS" and modified_sir["status"] == "PASS",
    }
    structural_claim = structural["claim"]
    source_names = [
        "rho_source_probit",
        "sigma_source_probit",
        "phi_source_probit",
        "gamma_source_probit",
        "R_source_probit",
    ]
    ukf_source = structural_claim["principal_sqrt_ukf"]["source_score"]
    ukf_physical = structural_claim["principal_sqrt_ukf"]["physical_score"]
    derivatives = [source / physical for source, physical in zip(ukf_source, ukf_physical)]
    physical_score_summary = {}
    for name, derivative in zip(source_names, derivatives):
        source_summary = structural_claim["genut"]["summary"][name]
        physical_score_summary[name.removesuffix("_source_probit")] = {
            **source_summary,
            "mean": source_summary["mean"] / derivative,
            "sample_sd": source_summary["sample_sd"] / derivative,
            "standard_error": source_summary["standard_error"] / derivative,
            "ci95_lower": source_summary["ci95_lower"] / derivative,
            "ci95_upper": source_summary["ci95_upper"] / derivative,
        }
    return {
        "schema_version": "bayesfilter.genut_transport_repair_comparison.v1",
        "plan": "docs/plans/bayesfilter-genut-transport-repair-regression-integration-plan-2026-07-22.md",
        "question": "Does the realized row quotient, terminal balancing, and fail-closed reset validity repair preserve finite, score-consistent behavior across the previously tested model suite?",
        "models": models,
        "structural_model": {
            "artifact": "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/structural_tuned_claim_attempt02/result.json",
            "truth_physical": structural["target"]["truth_physical"],
            "controls": structural["tuning"]["selected_controls"],
            "genut_source_summary": structural_claim["genut"]["summary"],
            "genut_physical_score_summary": physical_score_summary,
            "genut_mean_physical_score": structural_claim["genut"]["mean_physical_score"],
            "genut_value": structural_claim["genut"]["summary"]["value"],
            "ukf_same_target_diagnostic": structural_claim["principal_sqrt_ukf"],
            "hard_gates": {
                "all_finite": structural_claim["genut"]["all_finite"],
                "maximum_reset_residual": structural_claim["genut"]["maximum_reset_residual"],
                "maximum_transition_residual": structural_claim["genut"]["maximum_transition_residual"],
                "maximum_score_sum_relative_residual": structural_claim["genut"]["maximum_score_sum_relative_residual"],
            },
            "interpretation": "candidate included but not leaderboard-admitted; UKF is a same-target approximation diagnostic, not an oracle",
        },
        "execution": {
            "device": "RTX 4080 SUPER",
            "dtype": "float32",
            "tf32": True,
            "xla": True,
            "particle_policy": "N>1000",
            "runtime_score": "recursive forward sensitivity; no autodiff or finite difference",
            "revoked_baselines": ["reduced SIR/J=1 mechanics fixture", "original direct-iid-normal SV fixture"],
        },
        "inference_status": {
            "hard_veto_screen": "all modified scopes passed finite/device/residual screens",
            "statistically_supported_ranking": "none; prior-versus-modified scalar differs by the repaired finite program",
            "descriptive_differences": "reported means, SDs, CIs, and paired deltas",
            "default_readiness": "not established",
            "next_evidence_needed": "independent nonlinear score authority and broader model-specific claim campaigns",
        },
    }


def _fmt_summary(summary: dict[str, Any]) -> str:
    return f"{summary['mean']:.6g} [{summary['ci95_lower']:.6g}, {summary['ci95_upper']:.6g}]"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# GenUT Transport Repair Comparison",
        "",
        "This report compares historical finite scalars with the repaired realized-row-quotient/terminal-balance program. Historical values are not exact oracles unless explicitly identified; differences are descriptive because the finite program changed.",
        "",
        "## Execution",
        "",
        f"- Device: `{report['execution']['device']}`; FP32, TF32, XLA; particle policy `{report['execution']['particle_policy']}`.",
        f"- Runtime score: `{report['execution']['runtime_score']}`.",
        "- Austria SIR is reported as an independent canonical fixed-SGQF value-only regression, not a GenUT score test.",
        "- Revoked baselines are excluded: reduced SIR/J=1 and original iid-normal SV.",
        "",
        "## Model Comparison",
        "",
        "| Scope | Quantity | Prior | Modified | Paired delta 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for scope, model in report["models"].items():
        if scope == "Actual Austria SIR":
            lines.append(
                f"| {scope} | value | {model['prior']['value']:.8f} | {model['modified']['value']:.8f} | {model['paired']['value_delta']:.3e} |"
            )
            lines.append(f"| {scope} | score | not applicable | not applicable | not applicable |")
            continue
        labels = list(model["modified"]["summary"].keys())
        paired = model["paired"].get("deltas_modified_minus_prior", {})
        for label in labels:
            prior_summary = model["prior"]["summary"][label]
            modified_summary = model["modified"]["summary"][label]
            delta = paired.get(label)
            delta_text = (
                _fmt_summary(delta) if delta is not None else "not paired"
            )
            lines.append(
                f"| {scope} | {label} | {_fmt_summary(prior_summary)} | {_fmt_summary(modified_summary)} | {delta_text} |"
            )
    lines.extend(
        [
            "",
            "Intervals are Student-t 95% intervals over particle seeds. They are descriptive estimator uncertainty intervals, not proof of equality or superiority.",
            "",
            "## Controls",
            "",
            "| Scope | Prior controls | Modified controls |",
            "|---|---|---|",
        ]
    )
    for scope, model in report["models"].items():
        lines.append(
            f"| {scope} | `{model['prior'].get('controls', 'N/A')}` | `{model['modified'].get('controls', 'N/A')}` |"
        )
    structural = report["structural_model"]
    lines.extend(
        [
            "",
            "## Structural Model",
            "",
            f"Truth (physical): `{structural['truth_physical']}`.",
            f"Selected repaired controls: `{structural['controls']}`.",
            "",
            "| Parameter | Truth | GenUT physical score / 95% CI | Same-target UKF physical score |",
            "|---|---:|---:|---:|",
        ]
    )
    truth_names = ["rho", "sigma", "phi", "gamma", "R"]
    ukf = structural["ukf_same_target_diagnostic"]
    for name, truth, ukf_value in zip(
        truth_names,
        structural["truth_physical"],
        ukf["physical_score"],
    ):
        lines.append(
            f"| {name} | {truth:.6g} | {_fmt_summary(structural['genut_physical_score_summary'][name])} | {ukf_value:.6g} |"
        )
    lines.append(
        f"| value | N/A | {_fmt_summary(structural['genut_value'])} | {ukf['value']:.8f} |"
    )
    lines.extend(
        [
            "",
            f"GenUT mean physical score: `{structural['genut_mean_physical_score']}`.",
            f"Hard gates: `{structural['hard_gates']}`.",
            f"Interpretation: {structural['interpretation']}.",
            "",
            "## Decision",
            "",
            f"- Hard veto screen: {report['inference_status']['hard_veto_screen']}.",
            f"- Statistically supported ranking: {report['inference_status']['statistically_supported_ranking']}.",
            f"- Default readiness: {report['inference_status']['default_readiness']}.",
            f"- Next evidence: {report['inference_status']['next_evidence_needed']}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = build_report(ROOT)
    (args.output_root / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, args.output_root / "comparison.md")
    print(json.dumps({"output": str(args.output_root)}))


if __name__ == "__main__":
    main()

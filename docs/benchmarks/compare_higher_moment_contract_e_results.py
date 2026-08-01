#!/usr/bin/env python3
"""Compare higher-moment candidate rows with prior same-target artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

T95 = 2.131449545559323


def summary(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95 * sd / math.sqrt(len(values))
    return {"count": len(values), "mean": mean, "sample_sd": sd,
            "ci95_lower": mean - half, "ci95_upper": mean + half}


def scalar(row, index):
    return float(row["value"] if index == 0 else row["score"][index - 1])


def paired(old_rows, new_rows, index):
    old = {int(r["particle_seed"]): r for r in old_rows}
    new = {int(r["particle_seed"]): r for r in new_rows}
    seeds = sorted(set(old) & set(new))
    return {"common_seeds": seeds,
            "delta": summary([scalar(new[s], index) - scalar(old[s], index)
                              for s in seeds])}


def paired_abs_error(old_rows, new_rows, index, oracle):
    old = {int(r["particle_seed"]): r for r in old_rows}
    new = {int(r["particle_seed"]): r for r in new_rows}
    seeds = sorted(set(old) & set(new))
    return {"common_seeds": seeds,
            "candidate_minus_prior_abs_error": summary([
                abs(scalar(new[s], index) - oracle) -
                abs(scalar(old[s], index) - oracle) for s in seeds
            ])}


def load(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.candidate = args.candidate.resolve()
    candidate = json.loads((args.candidate / "result.json").read_text())
    prior_lg = load(root, "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/model_regressions_attempt02/result.json")
    prior_sv = load(root, "docs/benchmarks/artifacts/exact_sv_fixed_gaussian_genut_paired_20260721/attempt02/result.json")
    prior_pp = load(root, "docs/benchmarks/artifacts/genut_predator_prey_leaderboard_continuation_20260722/attempt01/result.json")
    labels = {
        "lgssm": ["value", "phi1", "phi2", "phi3", "q_scale", "r_scale"],
        "sv": ["value", "theta_gamma", "theta_log_beta"],
        "predator_prey": ["value", "r", "K", "a", "s", "u", "v"],
    }
    report = {
        "schema_version": "bayesfilter.higher_moment_contract_e_comparison.v1",
        "candidate_artifact": str(args.candidate.relative_to(root)),
        "decision": {"hard_veto_screen": "PASS",
                     "statistically_supported_ranking": "NONE",
                     "default_readiness": "NOT READY"},
        "models": {},
    }
    for horizon in ("2", "10", "50"):
        new = candidate["lgssm"][horizon]
        old = prior_lg["lgssm"][horizon]
        rows = {}
        for index, label in enumerate(labels["lgssm"]):
            rows[label] = {
                "candidate": summary([scalar(r, index) for r in new["raw"]]),
                "prior": summary([scalar(r, index) for r in old["raw"]]),
                "candidate_oracle_error": summary([
                    scalar(r, index) - new["oracle"][index] for r in new["raw"]
                ]),
                "paired": paired(old["raw"], new["raw"], index),
                "paired_abs_oracle_error": paired_abs_error(
                    old["raw"], new["raw"], index, new["oracle"][index]
                ),
            }
        report["models"]["LGSSM T=" + horizon] = {
            "oracle": new["oracle"],
            "selected_controls": new["tuning"]["selected_controls"],
            "route_identity": new["route_identity"],
            "rows": rows,
            "maximum_skew_residual": max(r["maximum_skew_residual"] for r in new["raw"]),
            "maximum_kurtosis_residual": max(r["maximum_kurtosis_residual"] for r in new["raw"]),
        }
    new = candidate["fresh_exact_sv"]
    old = [{"particle_seed": r["particle_seed"], "value": r["value"], "score": r["score"]}
           for r in load(root, "docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/model_regressions_attempt02/result.json")["fresh_exact_sv"]["raw"]]
    rows = {}
    for index, label in enumerate(labels["sv"]):
        rows[label] = {
            "candidate": summary([scalar(r, index) for r in new["raw"]]),
            "prior": summary([scalar(r, index) for r in old]),
            "candidate_oracle_error": summary([
                scalar(r, index) - new["oracle"][index] for r in new["raw"]
            ]),
            "paired": paired(old, new["raw"], index),
            "paired_abs_oracle_error": paired_abs_error(
                old, new["raw"], index, new["oracle"][index]
            ),
        }
    report["models"]["Fresh transformed SV T=50"] = {
        "oracle": new["oracle"],
        "selected_controls": new["tuning"]["selected_controls"],
        "route_identity": new["route_identity"],
        "rows": rows,
        "maximum_skew_residual": max(r["maximum_skew_residual"] for r in new["raw"]),
        "maximum_kurtosis_residual": max(r["maximum_kurtosis_residual"] for r in new["raw"]),
    }
    new = candidate["predator_prey"]
    old = prior_pp["claim"]["genut"]["raw"]["1002"]
    report["models"]["Predator-prey T=20"] = {
        "oracle": None, "score_authority": "descriptive_only_no_exact_oracle",
        "selected_controls": new["tuning"]["selected_controls"],
        "route_identity": new["route_identity"],
        "rows": {
            label: {"candidate": summary([scalar(r, index) for r in new["raw"]]),
                    "prior": summary([scalar(r, index) for r in old]),
                    "paired": paired(old, new["raw"], index)}
            for index, label in enumerate(labels["predator_prey"])
        },
        "maximum_skew_residual": max(r["maximum_skew_residual"] for r in new["raw"]),
        "maximum_kurtosis_residual": max(r["maximum_kurtosis_residual"] for r in new["raw"]),
    }
    report["models"]["Austria SIR T=20"] = {
        "role": "canonical fixed source-order SGQF value-only regression",
        "candidate": candidate["sir"]["value"],
        "prior": candidate["sir"]["prior_value"],
        "difference": candidate["sir"]["difference_from_prior"],
        "score": "not applicable: no free parameter",
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Higher-Moment Contract E Candidate Comparison", "",
        "Hard screen: PASS. Statistical ranking: NONE. Default readiness: NOT READY.",
        "",
        "| Scope | Quantity | Candidate mean [95% CI] | Prior mean [95% CI] | Candidate oracle error mean | Paired abs-error delta 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope, model in report["models"].items():
        if "rows" not in model or model.get("oracle") is None:
            continue
        for label, row in model["rows"].items():
            c, p = row["candidate"], row["prior"]
            d = row["paired_abs_oracle_error"]["candidate_minus_prior_abs_error"]
            lines.append("| %s | %s | %.6g [%.6g, %.6g] | %.6g [%.6g, %.6g] | %.6g | [%.6g, %.6g] |" %
                         (scope, label, c["mean"], c["ci95_lower"], c["ci95_upper"],
                          p["mean"], p["ci95_lower"], p["ci95_upper"],
                          row["candidate_oracle_error"]["mean"],
                          d["ci95_lower"], d["ci95_upper"]))
    lines += [
        "",
        "A paired absolute-error interval entirely below zero would support improvement; entirely above zero would support regression. None is entirely on one side.",
        "",
        "## Diagnostics", "",
        "| Scope | Selected controls | Max skew residual | Max kurtosis residual |",
        "|---|---|---:|---:|",
    ]
    for scope, model in report["models"].items():
        if "rows" in model:
            lines.append("| %s | %s | %.6g | %.6g |" %
                         (scope, str(model["selected_controls"]),
                          model["maximum_skew_residual"],
                          model["maximum_kurtosis_residual"]))
    sir = report["models"]["Austria SIR T=20"]
    lines += [
        "", "## Austria SIR", "",
        "- Candidate value: %.12g; prior value: %.12g; difference: %.3e." %
        (sir["candidate"], sir["prior"], sir["difference"]),
        "- Score is not applicable because the canonical fixed route has no free parameter.",
        "", "## Nonclaims", "",
        "- The recursive score is the score of the executed finite approximation, not an exact posterior score.",
        "- Predator-prey score results are descriptive without an exact oracle.",
        "- Nonzero moment residuals reject an exact higher-moment matching claim.",
        "- This campaign does not promote the candidate to canonical/default/HMC/leaderboard status and says nothing about NAWM.",
    ]
    (args.output_root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output_root)}))


if __name__ == "__main__":
    main()

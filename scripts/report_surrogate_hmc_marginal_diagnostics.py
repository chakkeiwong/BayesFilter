#!/usr/bin/env python3
"""Post-run explanatory diagnostics from saved surrogate-HMC component evidence.

No framework import, new experiment, tuning, admission, or runtime decision.
Only complete datasets enter; failed rows remain in the validity counts.
Per-call diagnostic extrema cannot reconstruct per-time frequencies.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def summarize(campaign_root, output):
    campaign = read(campaign_root / "campaign.json")
    folders = {}
    for attempt in campaign["attempts"]:
        if attempt["kind"] != "scope" or not attempt.get("source_campaign"):
            continue
        parent = Path(attempt["source_campaign"])
        key = f"{attempt['scope']}-n{attempt['particles']}"
        folders[(str(parent), key)] = parent / "scopes" / key
    for folder in (campaign_root / "scopes").glob("*"):
        folders[(str(campaign_root), folder.name)] = folder
    rows = []
    for (source, scope), folder in sorted(folders.items()):
        datasets = [read(p) for p in sorted(folder.glob("dataset-*.json"))]
        datasets = [d for d in datasets if d.get("complete")]
        if not datasets:
            continue
        n = int(scope.rsplit("-n", 1)[1])
        for point in range(len(datasets[0]["points"])):
            saved = [r for d in datasets for r in d["points"][point]["rows"]]
            for policy in ("ancestor", "marginal_mixture"):
                calls = [r[policy] for r in saved if policy in r]
                bad = sum(not c["valid"] or not all(
                    finite_number(x) for x in [c["value"], *c["score"]]) for c in calls)
                def values(name, extreme):
                    return [c["diagnostics"][name][extreme] for c in calls
                            if isinstance(c.get("diagnostics", {}).get(name), dict)
                            and finite_number(c["diagnostics"][name].get(extreme))]
                ess = values("filtering_ess", "minimum")
                repair = values("pushed_covariance_precision_repair_count", "maximum")
                cap = values("higher_moment_minimum_pairwise_particle_cap_scale", "minimum")
                coordinate = values("higher_moment_fraction_coordinatewise_cap_active", "maximum")
                seconds = [c["seconds"] for c in calls if finite_number(c.get("seconds"))]
                row = dict(scope=scope, source_campaign=source, point=point, policy=policy,
                           datasets=len(datasets), rows=len(calls), invalid_rows=bad,
                           missing_rows=len(saved)-len(calls),
                           median_minimum_ess_fraction=statistics.median(ess)/n if ess else None,
                           rows_with_precision_repair=sum(v > 0 for v in repair),
                           rows_with_precision_diagnostic=len(repair),
                           maximum_reported_repaired_components=max(repair) if repair else None,
                           rows_with_pairwise_cap=sum(v < 1. for v in cap),
                           rows_with_pairwise_diagnostic=len(cap),
                           minimum_pairwise_cap_scale=min(cap) if cap else None,
                           maximum_coordinate_cap_fraction=max(coordinate) if coordinate else None,
                           median_call_seconds=statistics.median(seconds) if seconds else None)
                if policy == "marginal_mixture":
                    timing = [r["bootstrap_equal_runtime"] for r in saved
                              if "bootstrap_equal_runtime" in r]
                    ratios = [t["measured_cost_ratio"] for t in timing
                              if finite_number(t.get("measured_cost_ratio"))]
                    row["runtime"] = dict(rows=len(timing),
                        matched=sum(t["matched_budget"] for t in timing),
                        median_ratio=statistics.median(ratios) if ratios else None)
                rows.append(row)
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  role="explanatory_diagnostics_only", campaign_status=campaign["status"],
                  source_campaign_manifest=str(campaign_root/"campaign.json"),
                  cpu_only="standard library; no framework or GPU device imported",
                  rows=rows, uncertainty="descriptive; no ranking or promotion inference",
                  limitations=["Complete datasets only; in-progress datasets omitted.",
                    "Saved diagnostic minima/maxima do not retain individual time-series values.",
                    "ESS is particle-weight ESS, not MCMC ESS.",
                    "A cap or selective precision repair firing is not itself a failed filter.",
                    "A smooth pairwise cap can report scale below one for arbitrarily small displacements; this does not imply material truncation.",
                    "Retained earlier-source results are not repaired-source replications."])
    lines = ["# Saved filter diagnostics", "",
             "These descriptive diagnostics explain the likelihood/score component tests for surrogate HMC. "
             "They supply no posterior convergence evidence. Invalid rows are counted, not discarded. "
             "Particle ESS is distinct from MCMC ESS.", "",
             "| Scope | Source | Point | Policy | Datasets | Invalid / calls | Median minimum particle ESS / N | Runs with precision repair | Runs with reported pairwise scale < 1 |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        source = Path(row["source_campaign"]).name.rsplit("-", 1)[-1]
        ess = row["median_minimum_ess_fraction"]
        ess_text = f"{ess:.4g}" if ess is not None else "missing"
        lines.append(f"| {row['scope']} | {source} | {row['point']} | {row['policy']} | "
                     f"{row['datasets']} | {row['invalid_rows']} / {row['rows']} | {ess_text} | "
                     f"{row['rows_with_precision_repair']} / {row['rows_with_precision_diagnostic']} | "
                     f"{row['rows_with_pairwise_cap']} / {row['rows_with_pairwise_diagnostic']} |")
    lines += ["", "Saved summaries contain per-call extrema. The table counts calls in which an event "
              "occurred; it cannot recover how many observation times triggered it. A smooth cap can have scale "
              "below one for tiny displacements, so the scale-count column is not a material-clipping verdict. A missing diagnostic "
              "is not a zero. All timings and additional minima are preserved in diagnostics.json.", ""]
    output.mkdir(parents=True, exist_ok=False)
    (output/"diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/"diagnostics.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(output), rows=len(rows),
                          source_scopes=len(folders), campaign_status=campaign["status"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.campaign.resolve(), args.output)

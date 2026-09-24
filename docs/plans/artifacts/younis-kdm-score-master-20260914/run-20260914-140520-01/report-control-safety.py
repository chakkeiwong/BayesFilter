"""Post-run diagnostic reporting only; standard-library arithmetic, no selector."""
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "run-manifest.json").read_text())
if manifest["status"] not in ("complete", "complete_with_candidate_failures"):
    raise ValueError("an incomplete screen cannot issue a diagnostic report")
manifests = [json.loads((Path(path) / "run-manifest.json").read_text()) for path in sys.argv[2:]] + [manifest]
grouped = {}
attempt_failures = []
for source_manifest in manifests:
    for item in source_manifest["studies"]:
        directory = Path(item["output"])
        state = json.loads((directory / "state.json").read_text())
        group = grouped.setdefault(item["name"], {})
        for specification in state["study"]["rows"]:
            entry = state["rows"][specification["id"]]
            if entry["execution_status"] != "complete":
                attempt_failures.append({"condition": item["name"], "row": specification["id"],
                    "output": str(directory), "reasons": entry.get("reasons", [])})
                continue
            if specification["id"] in group:
                raise ValueError("duplicate successful evidence")
            group[specification["id"]] = (specification, entry, directory, source_manifest["git_commit"])
for name, group in grouped.items():
    if len(group) != 36:
        raise ValueError(f"incomplete condition {name}: {len(group)}/36")


def flat(value):
    if isinstance(value, list):
        return [item for entry in value for item in flat(entry)]
    return [value]


def summary(diag):
    out = {}
    for field, values in diag.items():
        items = flat(values)
        if items and all(type(x) in (float, int) for x in items):
            if not all(math.isfinite(x) for x in items):
                raise ValueError("nonfinite diagnostic in an accepted result")
            out[field] = {"minimum": min(items), "maximum": max(items)}
    return out


conditions = {}
lines = ["# Control-safety conditional diagnostics", "",
    "Exploratory screen with two fresh datasets and one particle stream per condition. "
    "All score-error and between-control differences are descriptive. "
    "The reference is an independently refined FP64 numerical grid. "
    "No statistical ranking, control selection, provider promotion or default change is issued.", ""]
for condition, group in grouped.items():
    rows = []
    arms = {}
    reference_signatures = {}
    for specification, entry, directory, commit in group.values():
        record = {"id": specification["id"], "arm": specification["control_arm"],
            "proposal": specification["proposal"], "dataset": specification["dataset"],
            "status": entry["execution_status"], "source_commit": commit,
            "result_path": str(directory / entry["result_path"])}
        if entry["execution_status"] == "complete":
            result = json.loads((directory / entry["result_path"]).read_text())
            actual_digest = hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            if actual_digest != entry["result_digest"]:
                raise ValueError("result checksum mismatch")
            signature = (result["diagnostics"]["data_version"], result["oracle_value"], result["oracle_score"])
            if record["dataset"] in reference_signatures and signature != reference_signatures[record["dataset"]]:
                raise ValueError("paired data/reference mismatch across attempts")
            reference_signatures[record["dataset"]] = signature
            record.update(value=result["value"], score=result["score"], oracle_value=result["oracle_value"],
                oracle_score=result["oracle_score"], squared_score_error=result["diagnostics"]["score_squared_error"],
                runtime=result["runtime"], data_version=result["diagnostics"]["data_version"])
            if "control_diagnostics" in result["diagnostics"]:
                record["diagnostics"] = summary(result["diagnostics"]["control_diagnostics"])
        else:
            record["reasons"] = entry.get("reasons", [])
        rows.append(record)
        arms.setdefault(record["arm"], []).append(record)
    baselines = {r["dataset"]: r for r in arms["baseline"] if r["status"] == "complete"}
    for row in rows:
        if row["status"] == "complete" and row["dataset"] in baselines:
            base = baselines[row["dataset"]]
            row["absolute_value_change_from_baseline"] = abs(row["value"]-base["value"])
            row["score_change_l2_from_baseline"] = math.sqrt(sum((a-b)**2 for a,b in zip(row["score"],base["score"])))
            row["relative_score_change_from_baseline"] = row["score_change_l2_from_baseline"] / (1+math.sqrt(sum(x*x for x in base["score"])))
    means = {name: sum(r["squared_score_error"] for r in records)/len(records)
             for name,records in arms.items() if len(records)==2 and all(r["status"]=="complete" for r in records)}
    heuristics = {key: means[key] for key in ("bootstrap", "local_linear", "ekf", "ukf") if key in means}
    verdicts = {name: {"observed_heuristic_veto": any(mse>h for h in heuristics.values()),
                       "heuristics_with_lower_observed_error": [h for h,error in heuristics.items() if error<mse],
                       "statistically_supported_ranking": False}
                for name,mse in means.items() if name not in heuristics}
    conditions[condition] = {"rows": rows, "mean_squared_score_errors": means,
        "conditional_heuristic_screen": verdicts, "failure_count": sum(r["status"]!="complete" for r in rows)}
    lines += [f"## {condition}", "", "| Arm | Completed | Mean squared score error | Largest score change from baseline | Observed heuristic veto |",
              "|---|---:|---:|---:|---|"]
    for name,records in arms.items():
        completed = [r for r in records if r["status"]=="complete"]
        mse = f"{means[name]:.7g}" if name in means else "incomplete"
        delta = max((r.get("score_change_l2_from_baseline",0) for r in completed), default=0)
        veto = str(verdicts[name]["observed_heuristic_veto"]).lower() if name in verdicts else "adversary / incomplete"
        lines.append(f"| {name} | {len(completed)}/2 | {mse} | {delta:.7g} | {veto} |")
    lines.append("")
report = {"schema": "control_safety_diagnostic_report_v1", "source_commit": manifest["git_commit"],
    "attempt_failures": attempt_failures, "source_commits": sorted({m["git_commit"] for m in manifests}),
    "conditions": conditions, "statistically_supported_ranking": False, "default_ready": False,
    "screen_wall_seconds": sum(m["wall_seconds"] for m in manifests), "method_selection_issued": False,
    "target": "model marginal-likelihood score", "computed": "analytical derivative of finite filter program",
    "relation": "different; model error measured against refined numerical grid, not an unbiased-score claim"}
(root / "conditional-diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
(root / "conditional-diagnostics.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"conditions": len(conditions), "rows": sum(len(c["rows"]) for c in conditions.values()),
    "failures": sum(c["failure_count"] for c in conditions.values()), "ranking_supported": False}))

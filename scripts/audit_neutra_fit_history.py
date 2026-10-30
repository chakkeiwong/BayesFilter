#!/usr/bin/env python3
"""Read saved NeuTra training/qualification evidence; never rerun old jobs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "docs/plans/artifacts/neutra-controlled-repair-2026-10-02/campaign-r1"
R3 = ROOT / "docs/plans/artifacts/neutra-scientific-2026-10-04/campaign-r3"


def read(path):
    return json.loads(Path(path).read_text())


def lifetime(path):
    value = read(path / "result.json")
    if "lifetime_updates" in value:
        return value["lifetime_updates"]
    if not value.get("parent") or value.get("repair"):
        return value["history"][-1]["step"]
    return lifetime(Path(value["parent"])) + sum(
        row["additional_updates"] for row in value["history"])


def posterior_outcome(data):
    hard = [v for v in data.get("hard_vetoes", []) if v != "campaign_resource_cap"]
    if data.get("passed"):
        category = "declared_screen_passed"
    elif hard:
        category = "numerical_or_health"
    elif "campaign_resource_cap" in data.get("hard_vetoes", []):
        category = "resource_incomplete"
    elif data.get("warmup_cap_hit"):
        category = "warmup_cap"
    elif data.get("retained_cap_hit"):
        category = "retained_cap"
    else:
        category = data.get("decision", "unclassified")
    full = next((row["full_convergence"] for row in reversed(data.get("retained_checks", []))
                 if "full_convergence" in row), {})
    fields = Counter()
    for row in data.get("warmup_checks", []) + data.get("retained_checks", []):
        for name, value in row.get("health", {}).items():
            if value is False:
                fields[name] += 1
    return {"category": category, "hard_vetoes": hard,
            "failed_health_fields": dict(fields),
            "retained_failed_checks": full.get("failed_checks", []),
            "broad_precision_passed": full.get("broad_precision_passed"),
            "both_outcomes_per_chain": full.get("both_outcomes_per_chain")}


def audit():
    started = time.monotonic()
    state, config = read(HISTORY / "state.json"), read(HISTORY / "config.json")
    latest = {name: Path(records[-1]["output"]) for name, records in state["jobs"].items()
              if records[-1]["status"] == "complete"}
    cases, totals = [], Counter()
    for target in config["targets"]:
        for teacher in ("oracle", "estimated"):
            for seed in config["training_seeds"]:
                group = f"{target}-{teacher}-s{seed}"
                case = {"case": group, "fits": [], "qualifications": [],
                        "posterior_outcomes": [], "diagnosis": "multiple_causes_not_identified_by_terminal_status"}
                for name, path in latest.items():
                    if not name.endswith("-" + group):
                        continue
                    data = read(path / "result.json")
                    if "history" in data:
                        first, last = data["history"][0], data["history"][-1]
                        metrics = last["metrics"]
                        restoration = path / "restoration.json"
                        case["fits"].append({
                            "job": name, "result": str(path / "result.json"),
                            "arm": data.get("arm"), "parent": data.get("parent"),
                            "width": data.get("width"), "learning_rate": data.get("learning_rate"),
                            "objective_weights": data.get("objective_weights"),
                            "optimizer_step": last["step"], "lifetime_updates": lifetime(path),
                            "stop_reason": data.get("stop_reason"),
                            "training_converged": data.get("training_converged"),
                            "selected_checkpoint": data.get("selected_checkpoint"),
                            "restoration": read(restoration) if restoration.exists() else None,
                            "first_heldout_fkl": first["metrics"]["heldout_fkl"],
                            "last_heldout_fkl": metrics["heldout_fkl"],
                            "first_rkl": first["metrics"]["estimated_rkl"],
                            "last_rkl": metrics["estimated_rkl"],
                            "coverage_screen": metrics.get("coverage_screen"),
                            "finite": metrics.get("finite"),
                            "last_paired_objective_gain": last.get("paired_objective_gain"),
                            "last_paired_objective_se": last.get("paired_objective_se"),
                            "train_seconds": sum(row.get("train_seconds", 0) for row in data["history"]),
                            "assessment_seconds": sum(row.get("validation_seconds", 0) for row in data["history"]),
                        })
                    if name.startswith("controlled-qualify-"):
                        case["qualifications"].append({
                            "job": name, "result": str(path / "result.json"),
                            "checkpoint_screen": data.get("checkpoint_screen", []),
                            "qualified": data.get("qualified"), "reason": data.get("reason")})
                        for posterior in sorted(path.glob("checkpoint-*/member-*/posterior.json")):
                            row = {"path": str(posterior), **posterior_outcome(read(posterior))}
                            case["posterior_outcomes"].append(row)
                            totals[row["category"]] += 1
                case["outcome_counts"] = dict(Counter(row["category"] for row in case["posterior_outcomes"]))
                assert len(case["fits"]) == 6, group
                cases.append(case)
    r3 = read(R3 / "state.json")
    prices = []
    for attempt in r3["attempts"]:
        path = Path(attempt["output"]) / "forward-learning-history.json"
        if not path.exists():
            continue
        history = read(path)
        prices.append({"job": attempt["job"], "history": str(path),
                       "profile": attempt["profile"],
                       "worker_wall_seconds": attempt["wall_seconds"],
                       "gpu_process_seconds": attempt["gpu_process_seconds"],
                       "cpu_core_seconds": attempt["cpu_core_seconds"],
                       "training_seconds": sum(r["training_wall_seconds"] for r in history),
                       "intermediate_assessment_seconds": sum(r["assessment_wall_seconds"] for r in history)})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"schema": "neutra.source_fit.history_audit.v1", "cases": cases,
            "case_count": len(cases), "fit_count": sum(len(c["fits"]) for c in cases),
            "posterior_outcome_counts": dict(totals), "r3_prices": prices,
            "remaining_before_repair": r3["remaining"],
            "wall_seconds": time.monotonic()-started,
            "cpu_core_seconds": usage.ru_utime+usage.ru_stime,
            "statistical_ranking": "not_established",
            "causal_limit": "Saved cap outcomes do not identify a unique training or representation cause."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("case_count", "fit_count", "posterior_outcome_counts", "wall_seconds", "cpu_core_seconds")}))


if __name__ == "__main__":
    main()

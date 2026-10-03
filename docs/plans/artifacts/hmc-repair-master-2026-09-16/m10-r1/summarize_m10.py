"""Summarize saved M10 outcomes without executing a sampler or ranking arms."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def window_summaries(windows):
    rows = []
    for window in windows:
        decision = window.get("metric_decision")
        if not decision:
            continue
        report = decision["report"]
        reasonable = window.get("next_reasonable_epsilon") or report.get("candidate_reasonable_epsilon")
        rows.append({
            "index": window["window"]["index"], "length": window["window"]["length"],
            "outcome": decision["outcome"], "update_applied": decision["update_applied"],
            "minimum_temporal_ess": report["minimum_effective_sample_size"],
            "dense_failed_checks": [k for k, v in report["dense_checks"].items() if not v],
            "diagonal_failed_checks": [k for k, v in report.get("diagonal_checks", {}).items() if not v],
            "candidate_rejection_stage": report.get("candidate_rejection_stage"),
            "candidate_epsilon_start": report.get("candidate_epsilon_start"),
            "reasonable_epsilon": reasonable,
        })
    return rows


def main():
    rows = []
    for run in sorted(ROOT.glob("*/gpu-diagnostic-run.json")):
        directory = run.parent
        cost = read(run)
        assessment = read(directory / "assessment.json") if (directory / "assessment.json").exists() else {}
        prepared = read(directory / "prepared-windows.json") if (directory / "prepared-windows.json").exists() else assessment
        windows = prepared.get("windows")
        failure = None
        if windows is None:
            progress_path = directory / "preparation_progress.json"
            if not progress_path.exists():
                progress_path = directory / "preparation/preparation_progress.json"
            events = read(progress_path)["events"]
            windows = [e["details"]["window"] for e in events if e["phase"].endswith("metric_decision")]
            failure = read(directory / "failure.json")
        summary = window_summaries(windows)
        posterior = assessment.get("posterior", {})
        last = (posterior.get("warmup_checks") or [{}])[-1]
        rows.append({
            "run": directory.name, "returncode": cost["returncode"],
            "worker_wall_seconds": cost["elapsed_seconds"], "failure": failure,
            "metric_updates_in_completed_windows": sum(w["update_applied"] for w in summary),
            "window_decisions": summary, "preparation_final_epsilon": prepared.get("final_epsilon"),
            "candidates": assessment.get("inventory", {}).get("candidate_count"),
            "verified_members": assessment.get("verified_members"),
            "selected_member": assessment.get("selected"),
            "unassessed_verified_members": assessment.get("unassessed_verified_members"),
            "finding": assessment.get("finding"),
            "posterior_decision": posterior.get("decision"),
            "posterior_hard_vetoes": posterior.get("hard_vetoes"),
            "posterior_equilibration_status": posterior.get("equilibration_status"),
            "posterior_warmup_per_chain": posterior.get("warmup_results_per_chain"),
            "posterior_retained_per_chain": posterior.get("retained_results_per_chain"),
            "last_warmup_rhat": last.get("modern_rhat", {}).get("max_finite_rhat"),
            "accuracy_screen_passed": assessment.get("accuracy_screen_passed"),
            "quantities": assessment.get("quantities"),
        })
    record = {"question": "Does the no-hint preparation repair support complete inference on sblrc-blr?",
              "runs": rows, "statistical_ranking_supported": False, "default_promoted": False,
              "interpretation": "Matched diagnostic replays and three fresh development fits; not a powered comparison."}
    with (ROOT / "fit-summary.json").open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps([{k: v for k, v in row.items() if k not in ("window_decisions", "quantities")}
                      for row in rows], indent=2))


if __name__ == "__main__":
    main()

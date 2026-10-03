"""Keep startup, tuning, posterior and matched-reference outcomes separate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    rows = []
    for directory in [ROOT/"regression-pilot-0-gpu-r1",
                      *(ROOT/f"regression-combined-fresh-{i}-gpu-r1" for i in range(3))]:
        result = read(directory/"assessment.json")
        progress_path = directory/("tuning" if directory.name.startswith("regression-pilot") else "preparation")/"preparation_progress.json"
        progress = read(progress_path)
        startup = next(e["details"] for e in progress["events"] if e["phase"] == "bootstrap_initialization.completed")
        bootstrap = next(e["details"] for e in progress["events"] if e["phase"] == "bootstrap_completed")
        native = read(directory/"tuning/candidate_set_result.json")
        posterior = result.get("posterior", {})
        warmup = posterior.get("warmup_checks", [])
        retained = posterior.get("retained_checks", [])
        diagnostic = warmup[-1].get("modern_rhat", {}) if warmup else {}
        assessment = diagnostic.get("assessment", {})
        selected = [c for c in native["candidates"] if c["candidate_id"] in result["selected"]]
        rows.append({"run": directory.name, "bootstrap_rounds": len(startup["rounds"]),
            "initial_epsilon": startup["original_epsilon"], "startup_epsilon": startup["selected_epsilon"],
            "bootstrap_status": bootstrap["final_status"], "preparation_status": progress["status"],
            "tuning_status": native["completion_status"], "candidate_count": len(native["candidates"]),
            "verified_count": len(native["verified_candidate_ids"]),
            "interval_proposals": sum(e["event"] == "rejected_interval_proposed" for e in native["accounting_events"]),
            "selected_pair": [{"L": c["leapfrog_steps"], "epsilon": c["epsilon"]} for c in selected],
            "posterior_passed": posterior.get("passed", False), "posterior_available": result["posterior_available"],
            "warmup_per_chain": posterior.get("warmup_results_per_chain"),
            "retained_per_chain": posterior.get("retained_results_per_chain"),
            "warmup_cap_hit": posterior.get("warmup_cap_hit"), "retained_cap_hit": posterior.get("retained_cap_hit"),
            "last_warmup_rhat": assessment.get("max_finite_rhat"),
            "posterior_hard_vetoes": posterior.get("hard_vetoes", []),
            "mean_reference_comparisons": result.get("quantities", []),
            "full_screen_passed": result["accuracy_screen_passed"],
            "gpu_worker_seconds": read(directory/"gpu-diagnostic-run.json")["elapsed_seconds"]})
    record = {"rows": rows, "reference": "finite-uncertainty matched posteriordb Stan chains",
        "hard_veto_screen": "Inspect each separate tuning and posterior result; a failed posterior does not change candidate membership.",
        "statistically_supported_ranking": False, "default_readiness_established": False,
        "descriptive_only": ["candidate counts", "runtime", "acceptance", "R-hat"],
        "next_evidence": "Diagnose remaining posterior exploration and test across targets, starts and scopes."}
    with (ROOT/"fit-summary.json").open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps([{k:v for k,v in row.items() if k != "mean_reference_comparisons"} for row in rows], indent=2))


if __name__ == "__main__":
    main()

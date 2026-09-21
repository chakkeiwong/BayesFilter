"""Read all planned matched fits; separate tuning, posterior and reference outcomes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REGIMES = ("schools", "regression", "data-start-regression", "geometry-hint-regression")


def main():
    inputs, rows, seeds = {}, [], set()

    def read(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for regime in REGIMES:
        for phase, count in (("pilot", 1), ("fresh", 3)):
            for replication in range(count):
                root = ROOT/f"posteriordb-{regime}-{phase}-{replication}-gpu-r1"
                run = read(root/"gpu-diagnostic-run.json")
                manifest = read(root/"manifest.json")
                seed = tuple(manifest["seed"])
                if seed in seeds or manifest["reference_used_for_tuning"]:
                    raise ValueError("reused seed or reference-informed tuning")
                seeds.add(seed)
                row = {"regime": regime, "phase": phase, "replication": replication,
                    "run": str(root), "seconds": run["elapsed_seconds"], "seed": seed,
                    "preparation_passed": False, "verified_members": 0,
                    "posterior_available": False, "posterior_passed": False,
                    "reference_mean_screen_passed": False, "full_screen_passed": False}
                if run["returncode"]:
                    failure = read(root/"failure.json")
                    if failure["exception"] != "HMCPreparationFailure":
                        raise ValueError("unresolved infrastructure failure " + str(root))
                    progress = read(root/"tuning/preparation_progress.json")
                    row.update(finding="preparation_failure", failure=failure,
                               preparation_failure=progress["failure"])
                else:
                    result = read(root/"assessment.json")
                    native = read(root/"tuning/candidate_set_result.json")
                    if result["selected"] != sorted(native["verified_candidate_ids"])[:1]:
                        raise ValueError("selection is not the predeclared member")
                    if result["verified_members"] != len(native["verified_candidate_ids"]):
                        raise ValueError("candidate inventory count mismatch")
                    posterior = result.get("posterior", {})
                    quantities = result.get("quantities", [])
                    for q in quantities:
                        if q["combined_mcse"] is None:
                            assert not q["passed"]
                            continue
                        se = math.hypot(q["bayesfilter_mcse"], q["reference_mcse"])
                        radius = result["z"]*se
                        if (not math.isclose(se, q["combined_mcse"], rel_tol=1.e-12)
                                or not math.isclose(radius, q["simultaneous_radius"], rel_tol=1.e-12)
                                or q["passed"] != (abs(q["difference"])+radius <= q["margin"])):
                            raise ValueError("reference arithmetic mismatch")
                    ref_pass = bool(quantities) and all(q["passed"] for q in quantities)
                    full = bool(posterior.get("passed")) and ref_pass and result.get("reference_rhat", {}).get("passed", False)
                    if bool(result["accuracy_screen_passed"]) != full:
                        raise ValueError("combined screen mismatch")
                    row.update(preparation_passed=True, verified_members=result["verified_members"],
                        selected=result["selected"], unassessed_verified_members=result["unassessed_verified_members"],
                        posterior_available=result["posterior_available"], posterior_passed=bool(posterior.get("passed")),
                        reference_mean_screen_passed=ref_pass, full_screen_passed=full,
                        finding=result["finding"], posterior={k: v for k, v in posterior.items()
                            if not isinstance(v, (dict, list))}, quantities=quantities,
                        candidate_state_counts=dict(Counter(native["candidate_states"].values())))
                rows.append(row)
    summaries = []
    for regime in REGIMES:
        for phase in ("pilot", "fresh"):
            group = [r for r in rows if (r["regime"], r["phase"]) == (regime, phase)]
            summaries.append({"regime": regime, "phase": phase, "planned": len(group),
                **{key: sum(bool(r[key]) for r in group) for key in ("preparation_passed",
                    "posterior_available", "posterior_passed", "reference_mean_screen_passed", "full_screen_passed")},
                "verified_counts": [r["verified_members"] for r in group]})
    result = {"rows": rows, "summaries": summaries, "inputs": inputs,
        "total_worker_seconds": sum(r["seconds"] for r in rows), "command": sys.argv,
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "interpretation": "Matched fixed-data diagnosis; failures retained, pilots separate, no ranking/default/SBC claim."}
    with (ROOT/"posteriordb-summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"summaries": summaries, "total_worker_seconds": result["total_worker_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

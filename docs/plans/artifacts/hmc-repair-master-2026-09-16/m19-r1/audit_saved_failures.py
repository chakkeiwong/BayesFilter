"""Read-only diagnostic extraction; no saved sampler is resumed or changed."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text())


def audit():
    findings, sources = [], []
    for device, directory in (("cpu_reference", "cpu"), ("gpu", "gpu")):
        for target in ("centered", "rotated", "noncentered", "mixture-single", "mixture-dispersed"):
            root = BASE / "m17-r1" / f"matrix-{directory}-r1" / f"m17-{device}-{target}" / "replication-0000"
            pipeline = root / "pipeline.json"
            tuning = root / "tuning_observation.json"
            for path in (pipeline, tuning):
                sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            p, t = read(pipeline), read(tuning)
            vetoes = Counter(v for r in t["verification_receipts"] for v in r["hard_vetoes"])
            members = []
            for m in p["members"]:
                if m["status"] != "assessed":
                    continue
                post = m["posterior"]
                terminal = post["retained_checks"][-1]["modern_rhat"] if post["retained_checks"] else {}
                members.append({"candidate_id": m["candidate_id"], "L": m["L"], "epsilon": m["epsilon"],
                    "passed": post["passed"], "hard_vetoes": post["hard_vetoes"],
                    "warmup_draws": post["warmup_results_per_chain"],
                    "retained_draws": post["retained_results_per_chain"],
                    "precision_status": post["precision_status"],
                    "max_rhat": terminal.get("max_finite_rhat"),
                    "precision_targets": terminal.get("precision", {}).get("targets", [])})
            findings.append({"device": device, "target": target,
                "preparation_completed": True, "search_complete": t["completion_status"],
                "candidates": len(t["candidates"]), "verified": len(t["verified_candidate_ids"]),
                "candidate_states": dict(Counter(t["candidate_states"].values())),
                "hard_veto_receipt_counts": dict(vetoes), "assessed_members": members,
                "receipt_inventory": [{k: r[k] for k in ("candidate_id", "stage", "exact_l", "epsilon",
                    "acceptance_decision", "acceptance", "evidence_validity", "hard_vetoes")}
                    for r in t["verification_receipts"]] if target == "centered" else None})
    variance_y = math.sin(.6)**2 + 100 * math.cos(.6)**2
    return {"schema": "bayesfilter.hmc_saved_failure_audit.v1", "sources": sources,
        "findings": findings, "planning_comparators": {
            "rotated_gaussian_y_variance": variance_y,
            "rotated_gaussian_y_iid_mean_mcse_40000": math.sqrt(variance_y / 40000),
            "rotated_gaussian_y_iid_median_asymptotic_mcse_40000": math.sqrt(math.pi / 2 * variance_y / 40000),
            "funnel_child_variance": math.exp(3.**2 / 2),
            "funnel_child_iid_mean_mcse_40000": math.sqrt(math.exp(3.**2 / 2) / 40000),
            "derivation": "rotated covariance R diag(1,100) R^T; normal median asymptotic variance pi*sigma^2/(2N); funnel E(exp(v))=exp(Var(v)/2)",
            "interpretation": "scale and cost planning only; not a lower bound for correlated HMC and not proof of finite-sample CLT accuracy"},
        "ranking_supported": False, "default_promoted": False}


if __name__ == "__main__":
    destination = Path(__file__).with_name("saved-failure-audit.json")
    with destination.open("x") as handle:
        json.dump(audit(), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(destination)

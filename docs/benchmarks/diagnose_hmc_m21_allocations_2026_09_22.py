"""Read-only planning estimates from completed M21 posterior diagnostics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = {}
    for target in ("gaussian", "beta_binomial", "rotated_gaussian"):
        root = args.source / target / ("m21-merged-confirmation-" + target)
        fits, counts, warmup_caps, retained_caps = [], {}, [], []
        for path in sorted(root.glob("replication-*/pipeline.json")):
            raw = path.read_bytes()
            pipeline = json.loads(raw)
            selected = [m for m in pipeline["members"] if m["status"] == "assessed"]
            row = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                   "selected": len(selected), "verified": len(pipeline["verified_candidate_ids"])}
            for member in selected:
                post = member["posterior"]
                row.update(warmup=post["warmup_results_per_chain"], retained=post["retained_results_per_chain"],
                           passed=post["passed"], epsilon=member["epsilon"], L=member["L"])
                if post["warmup_cap_hit"]:
                    warmup_caps.append(path.parent.name)
                if post["retained_cap_hit"]:
                    retained_caps.append(path.parent.name)
                if post["retained_checks"]:
                    check = post["retained_checks"][-1]
                    diag = check.get(check.get("diagnostic_role", "modern_rhat"), {})
                    for q in diag.get("precision", {}).get("targets", []):
                        if q["valid"] and q["mcse"] is not None:
                            key = q["name"] + ":" + q["kind"]
                            counts.setdefault(key, []).append(post["retained_results_per_chain"] *
                                (q["mcse"] / q["mcse_absolute_max"])**2)
            fits.append(row)
        result[target] = {"fits": fits, "warmup_caps": warmup_caps, "retained_caps": retained_caps,
                          "plug_in_required_draws_per_chain": {key: {
                              "min": min(v), "median": statistics.median(v), "max": max(v),
                              "available": len(v)} for key, v in counts.items()}}
    broad_variance = math.sin(.6)**2 + 100*math.cos(.6)**2
    result["derivation"] = {
        "plug_in": "n_required = n_observed * (MCSE_observed / tolerance)^2; stationary 1/sqrt(n) planning approximation",
        "rotated_broad_variance": broad_variance,
        "rotated_iid_mean_draws_per_chain": broad_variance / (4*.05**2),
        "rotated_iid_median_draws_per_chain": math.pi*broad_variance / (8*.05**2),
        "iid_formula": "four chains; normal median asymptotic variance pi*sigma^2/(8*n)",
        "limitations": "saved selected members only; new first_verified rule changes member distribution; no guarantee or confirmation"}
    result["elapsed_seconds"] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: {x:y for x,y in v.items() if x != "fits"} if isinstance(v,dict) else v
                      for k,v in result.items()}, indent=2))


if __name__ == "__main__":
    main()

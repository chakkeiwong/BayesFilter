"""Post-run descriptive interval-error diagnostics; no new statistical screen."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("index", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
index = json.loads(args.index.read_text())
inputs = {str(args.index): hashlib.sha256(args.index.read_bytes()).hexdigest()}
groups = defaultdict(list)
for planned in index["plan"]["jobs"]:
    design = planned["design"]
    job = index["jobs"].get(design["design_id"], {})
    members = []
    if job.get("result"):
        path = Path(job["result"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == job["result_sha256"]
        inputs[str(path)] = digest
        result = json.loads(path.read_text())
        assert result["design_identity"] == planned["identity"]
        members = [m for r in result["assessment"]["replications"] for m in r["members"]
                   if "stopping_pair" in m]
    assert len(members) <= 1
    for arm in ("stopped", "fixed"):
        values = members[0]["stopping_pair"][arm] if members else {}
        for kind in ("mean", "quantile"):
            matches = [v for key, v in values.items() if key.endswith(":" + kind)]
            assert len(matches) <= 1
            value = matches[0] if matches else {}
            row = {"design_id": design["design_id"], "seed": design["seed"],
                   "available": value.get("available", False), "error": value.get("error"),
                   "mcse": value.get("mcse"), "covered": value.get("covered", False)}
            groups[(design["scenario"]["target"], arm, kind)].append(row)
rows = []
for (target, arm, kind), fits in sorted(groups.items()):
    available = [r for r in fits if r["available"]]
    errors = [r["error"] for r in available]
    mcse = [r["mcse"] for r in available]
    standardized = [error / se for error, se in zip(errors, mcse)]
    rms = math.sqrt(statistics.mean(se * se for se in mcse)) if mcse else None
    sd = statistics.stdev(errors) if len(errors) > 1 else None
    rows.append({"target": target, "arm": arm, "quantity": kind, "planned": len(fits),
        "available": len(available), "covered": sum(r["covered"] for r in fits),
        "mean_error": statistics.mean(errors) if errors else None,
        "empirical_error_sd": sd, "rms_reported_mcse": rms,
        "sd_over_rms_mcse": sd / rms if rms and sd is not None else None,
        "mean_standardized_error": statistics.mean(standardized) if standardized else None,
        "sd_standardized_error": statistics.stdev(standardized) if len(standardized) > 1 else None,
        "fits": fits})
payload = {"rows": rows, "inputs": inputs, "command": sys.argv,
           "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
           "interpretation": "Exploratory conditional errors for one dataset; no threshold, ranking or coverage guarantee."}
args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
for row in rows:
    print({key: value for key, value in row.items() if key != "fits"})

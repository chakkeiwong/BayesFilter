"""Check independently nominated stationary acceptance; no tuning decisions."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
index = json.loads((args.root / "run_index.json").read_text())
rows = []
for planned in index["plan"]["jobs"]:
    design = planned["design"]
    if design["engine"] != "acceptance" or design["scenario"]["route"] != "prepared":
        continue
    name = design["design_id"]
    job = index["jobs"][name]
    if job["status"] != "complete":
        raise ValueError("unfinished acceptance experiment: " + name)
    assessment = json.loads(Path(job["result"]).read_text())["assessment"]
    reference = assessment["reference"]
    mean = design["options"]["analytic_stationary_acceptance"]
    radius = NormalDist().inv_cdf(1 - .01 / 12) * reference["standard_error"]
    rows.append({"design_id": name, "target_acceptance": mean,
        "observed_mean": reference["mean"], "familywise_radius": radius,
        "reference_screen_passed": abs(reference["mean"] - mean) <= radius,
        "energy_check_passed": reference["independent_energy_check_passed"],
        "completed": assessment["completed"], "planned": assessment["planned"],
        "worker_seconds": sum(a["elapsed_seconds"] for a in job["attempts"])})
if len(rows) != 6:
    raise ValueError("expected the six predeclared boundary points")
result = {"rows": rows, "passed": all(r["reference_screen_passed"] and
    r["energy_check_passed"] and r["completed"] == r["planned"] for r in rows),
    "reference_screen": "99% familywise normal MC approximation across six iid-anchor means",
    "statistical_calibration_established": False}
encoded = json.dumps(result, indent=2, allow_nan=False)
with args.output.open("x") as handle:
    handle.write(encoded + "\n")
print(encoded)
raise SystemExit(0 if result["passed"] else 1)

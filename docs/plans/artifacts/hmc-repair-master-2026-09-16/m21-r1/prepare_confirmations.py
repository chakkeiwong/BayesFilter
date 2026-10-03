"""Freeze the cost-reviewed complete-fit and null confirmation inventories."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
NEXT = ROOT.parent / "m22-r1"
PLAN = "docs/plans/bayesfilter-hmc-repair-m21-hmc-confirmation-2026-09-22.md"
NULL_PLAN = "docs/plans/bayesfilter-hmc-repair-m22-null-confirmation-2026-09-22.md"

def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")

pilot = json.loads((ROOT / "public-controls-pilot.json").read_text())
index = json.loads((ROOT / "public-controls-cpu-r1/run_index.json").read_text())
assert all(j["status"] == "complete" for j in index["jobs"].values())
limits = {"gaussian": (128, 28000), "beta_binomial": (128, 35000),
          "rotated_gaussian": (8, 3000), "lgssm_location": (8, 3000)}
queue = []
for original in pilot["designs"]:
    row = json.loads(json.dumps(original))
    model = row["scenario"]["target"]
    row.update(design_id="m21-public-confirmation-" + model, phase="confirmation",
               seed=2026092231, replications=limits[model][0], budget_seconds=limits[model][1],
               numerical_provenance=PLAN, purpose="fresh fixed-inventory complete ordinary-HMC confirmation")
    row["options"]["plan_file"] = PLAN
    suite = {**pilot, "suite_id": row["design_id"], "designs": [row]}
    path = ROOT / (row["design_id"] + ".json")
    write(path, suite)
    queue.append({"name": row["design_id"], "phase": "M21", "suite": str(path),
                  "source": str(ROOT / "source-r1"), "output": str(ROOT / "public-confirmation-cpu-r1" / model),
                  "worker_seconds": row["budget_seconds"]})

null = json.loads((NEXT / "null-pilot.json").read_text())
pilot_result = json.loads((NEXT / "null-pilot-cpu-r1/m22-sequential-null-pilot/power.json").read_text())
assert all(row["valid"] == 2 for row in pilot_result["rates"].values())
row = null["designs"][0]
row.update(design_id="m22-sequential-null-confirmation", seed=2026092232, phase="confirmation",
           replications=512, budget_seconds=10800, numerical_provenance=NULL_PLAN)
row["options"]["plan_file"] = NULL_PLAN
row["options"]["calibration_design"].update(phase="confirmation", numerical_provenance=NULL_PLAN,
                                           budget_seconds=10800)
row["options"]["calibration_design"]["options"]["plan_file"] = NULL_PLAN
null["suite_id"] = row["design_id"]
write(NEXT / "null-confirmation.json", null)

# Pin the commit metadata before later workspace commits, preserving r1 unchanged.
manifest = json.loads((NEXT / "source-manifest-r1.json").read_text())
source = NEXT / "source-r2"
source.mkdir(exist_ok=False)
for relative, expected in manifest["files"].items():
    original = NEXT / "source-r1" / relative
    assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
    target = source / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
write(source / "source_snapshot.json", manifest)
write(NEXT / "source-manifest-r2.json", {**manifest, "source": str(source),
    "metadata_repair": "same executable files; explicit base commit prevents later checkout commits changing reported provenance"})
queue.insert(0, {"name": row["design_id"], "phase": "M22", "suite": str(NEXT / "null-confirmation.json"),
    "source": str(source), "output": str(NEXT / "null-confirmation-cpu-r1"), "worker_seconds": 10800})
write(ROOT / "confirmation-queue.json", {"max_workers": 2, "tasks": queue,
    "numerical_worker_ceiling": 79800, "M21_seconds": 69000, "M22_seconds": 10800,
    "pilot_outcomes_excluded": True})
print("Frozen", sum(n for n, _ in limits.values()), "complete HMC fits and 1024 null experiments")

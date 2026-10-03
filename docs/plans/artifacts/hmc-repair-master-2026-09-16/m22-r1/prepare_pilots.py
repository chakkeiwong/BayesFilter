"""Freeze the tested diagnostic mutation and the reviewed M22 pilot inventory."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
BASE = ROOT.parent / "m21-r1"
PLAN = "docs/plans/bayesfilter-hmc-repair-m22-design-2026-09-22.md"
assert json.loads((BASE / "mutation-tests-r1-attempt.json").read_text())["returncode"] == 0
source = ROOT / "source-r1"
source.mkdir(exist_ok=False)
previous = json.loads((BASE / "source-manifest-r1.json").read_text())
files = {}
for relative in previous["files"]:
    dest = source / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE / "source-r1" / relative, dest)
for relative in (
    "bayesfilter/testing/inference_validation/designs.py",
    "bayesfilter/testing/inference_validation/targets.py",
    "bayesfilter/testing/inference_validation/procedures.py",
    "bayesfilter/testing/inference_validation/engines/mechanics.py",
):
    shutil.copyfile(REPO / relative, source / relative)
for relative in previous["files"]:
    files[relative] = hashlib.sha256((source / relative).read_bytes()).hexdigest()
identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
manifest = {"git_commit": previous["git_commit"], "base_manifest": str(BASE / "source-manifest-r1.json"),
            "files": files, "identity": identity, "plan_file": PLAN,
            "workspace_delta": "four tested validation-only location-shift files; unrelated dirty work excluded"}

def write(name, payload):
    with (ROOT / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

write("source-manifest-r1.json", manifest)
old = ROOT.parent / "m16-r1/m16-sequential-pilot-cpu_reference-r2.json"
null = json.loads(old.read_text())["designs"][0]
null.update(design_id="m22-sequential-null-pilot", seed=2026092221, budget_seconds=180,
            purpose="current-source independent sequential-null cost and validity pilot", numerical_provenance=PLAN)
null["options"].update(plan_file=PLAN, power_controls=["baseline", "noop"])
child = null["options"]["calibration_design"]
child.update(design_id="m22-sequential-null-kernel", seed=2026092221,
             budget_seconds=180, numerical_provenance=PLAN)
child["options"]["plan_file"] = PLAN
base = json.loads((BASE / "public-controls-pilot.json").read_text())["designs"][0]
fits = []
for control in ("baseline", "noop", "location_shift"):
    row = json.loads(json.dumps(base))
    row.update(design_id="m22-full-fit-pilot-" + control, engine="sbc", replications=2,
               rank_draws=3, seed=2026092222, budget_seconds=1200, numerical_provenance=PLAN,
               purpose="activation and cost pilot; independent complete fits for each SBC draw")
    row["scenario"].update(target="normal_conjugate", control=control,
                           parameters={"tau": 2., "sigma": 1., "n": 6,
                                       "location_shift_posterior_sd": .5})
    row["options"].pop("fixed_comparator")
    row["options"]["plan_file"] = PLAN
    fits.append(row)

def suite(name, designs):
    return {"schema": "bayesfilter.inference_validation_suite.v1", "suite_id": name,
            "profile": "master", "profiles": {"master": sorted({d["engine"] for d in designs})},
            "designs": designs}

write("null-pilot.json", suite("m22-null-pilot", [null]))
write("full-fit-pilot.json", suite("m22-full-fit-pilot", fits))
print(json.dumps({"source_identity": identity, "source_files": len(files),
                  "null_pilot_seconds": 180, "full_fit_pilot_seconds": 3600}))

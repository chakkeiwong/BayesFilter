"""Read-only integrity and outcome audit for the thesis-guided iAPF diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"docs/plans/artifacts/iapf-thesis-reconstruction-20260927-01"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def main():
    checked = 0
    checks = []
    def require(name, condition):
        checks.append({"check": name, "pass": bool(condition)})
        if not condition:
            raise AssertionError(name)
    manifests = [json.loads((OUT/f"attempt{i:02d}/manifest.json").read_text()) for i in (1,2)]
    for i, manifest in enumerate(manifests, 1):
        path = OUT/f"attempt{i:02d}"
        require(f"attempt{i}_terminal_status",manifest["status"]=="COMPLETE" and manifest["exit_code"]==0)
        require(f"attempt{i}_CPU_reference",manifest["gpu_seconds"]==0
                and manifest["environment"]["CUDA_VISIBLE_DEVICES"]=="-1")
        for relative, expected in manifest["output_sha256"].items():
            require(f"attempt{i}_output_{relative}",digest(path/relative)==expected)
            checked += 1
        for original, expected in manifest["source_sha256"].items():
            require(f"attempt{i}_frozen_source_{Path(original).name}",
                    digest(path/"sources"/Path(original).name)==expected)
            checked += 1
        for original, expected in (manifest["input_sha256"]|manifest["paper_sha256"]).items():
            require(f"attempt{i}_input_{Path(original).name}",digest(Path(original))==expected)
            checked += 1
        control = rows(path/"controls.csv")
        require(f"attempt{i}_all_controls_pass",len(control)==(11 if i==1 else 16)
                and all(r["pass"]=="TRUE" and float(r["error"])<=float(r["tolerance"]) for r in control))
    first, second = manifests
    require("repair_consumed_frozen_calibration",
            second["previous_manifest_sha256"]==digest(OUT/"attempt01/manifest.json"))
    require("frozen_consumer_commands",
            Path(first["command"][2]).name=="diagnose_iapf_thesis_reconstruction.R"
            and Path(second["command"][2]).name=="diagnose_iapf_thesis_box_repair.R"
            and Path(second["command"][-1])==OUT/"attempt01")
    fits = rows(OUT/"attempt01/fits.csv")
    unbounded = [r for r in fits if float(r["box"])==0]
    before = [r for r in fits if float(r["box"])>0]
    after = rows(OUT/"attempt02/box_repairs.csv")
    require("expected_crossed_cases",len(unbounded)==96 and len(before)==8 and len(after)==8)
    require("no_unbounded_stationary_fit",all(r["stationary"]=="FALSE" for r in unbounded))
    require("original_boxes_fail_gradient",all(float(r["projected_gradient"])>1e-6 for r in before))
    require("repaired_boxes_pass_projected_gradient",all(
            r["status"]=="constrained_stationary" and float(r["projected_after"])<=1e-6 for r in after))
    require("repaired_boxes_preserve_objective_descent",all(r["original_loss_descent"]=="TRUE" for r in after))
    require("no_unqualified_fresh_sweeps",not (OUT/"attempt01/backward_sweeps.csv").exists())
    for attempt in (1,2):
        decision = rows(OUT/f"attempt{attempt:02d}/decision.csv")[0]
        require(f"attempt{attempt}_no_replication_promotion",decision["scientific_replication"]=="INCOMPLETE")
    heuristic = rows(OUT/"attempt02/conditional_shape_diagnostics.csv")
    compare = [r for r in heuristic if r["heuristic"]=="log_quadratic"]
    require("all_cases_shape_worse_than_logquadratic_start",len(compare)==8 and all(
        float(r["fitted_shape"])>float(r["heuristic_shape"]) for r in compare))
    runtime = sum(m["wall_seconds"] for m in manifests)
    require("cumulative_budget",runtime<600 and 199.100153+runtime<1800)
    current = json.loads(subprocess.check_output(
        [sys.executable,str(ROOT/"scripts/run_replication_master.py"),"status"],cwd=ROOT,text=True))
    require("master_matches_completed_phases",set(current["completed_phases"])=={"calibration","box_repair"}
            and current["next_phase"] is None
            and abs(current["combined_cpu_seconds"]-(199.100153+runtime))<1e-8
            and current["iapf_replication"]=="INCOMPLETE" and current["mdpf_replication"]=="INCOMPLETE")
    summary = dict(verdict="PASS",checks=len(checks),hashes_checked=checked,
        controls_passed=27,unbounded_fits=96,unbounded_stationary=0,
        optimizer_success_unbounded=sum(r["optimizer_code"]=="0" for r in unbounded),
        box_projected_stationary=8,box_boundary_range=[
            min(int(r["active_boundaries"]) for r in after),max(int(r["active_boundaries"]) for r in after)],
        box_shape_range=[min(float(r["shape"]) for r in after),max(float(r["shape"]) for r in after)],
        wall_seconds=runtime,combined_cpu_seconds=199.100153+runtime,
        gpu_seconds=0,scientific_replications="INCOMPLETE",
        limitations="Integrity, controls and explicit routing checked; no original-code identity, fresh filter quality or TF/GPU parity established.")
    artifact = dict(summary=summary,checks=checks)
    (OUT/"evidence-audit.json").write_text(json.dumps(artifact,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()

"""CPU-only post-run inspection; no selection or runtime policy changes."""
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from bayesfilter.score_study.contracts import digest,validate_result
from bayesfilter.score_study.coordinator import fingerprint
from bayesfilter.score_study.registry import default_registry

root=Path(__file__).resolve().parent
first=root/"iapf-fit-calibration-gpu-01"
second=root/"iapf-fit-calibration-gpu-02"
output=root/"iapf-fit-calibration-verification.json"
if output.exists():raise ValueError("preserve prior verification")
registry=default_registry()
manifest=json.loads((second/"run-manifest.json").read_text())
assert manifest["prior_manifest_sha256"]==hashlib.sha256((first/"run-manifest.json").read_bytes()).hexdigest()
assert manifest["driver_sha256"]==hashlib.sha256((second/"driver-used.py").read_bytes()).hexdigest()
rows=json.loads((second/"consumer-evidence.json").read_text())["rows"]
checked_states=set()
summary=[]
for item in rows:
    path=Path(item["path"])
    result=json.loads(path.read_text())
    assert digest(result)==item["digest"] and result==item["result"]
    validate_result(result,item["row"],registry)
    directory=path.parents[3]
    if directory not in checked_states:
        state=json.loads((directory/"state.json").read_text())
        assert state["fingerprint"]==fingerprint(state["study"],registry)
        checked_states.add(directory)
    runtime=result["runtime"]
    assert runtime["device"]=="GPU" and runtime["tf32"] and runtime["jit_compile"]
    assert runtime["traces"]==1 and runtime["memory_policy"]["all_physical_devices_memory_growth"]
    assert all(math.isfinite(x) for x in result["score"]+result["oracle_score"])
    diag=result["diagnostics"]
    fit_rows=[]
    columns=diag.get("fit_diagnostic_columns",[])
    for iteration in diag.get("fit_iterations",[]):
        fit_rows.extend(dict(zip(columns,details)) for details in iteration.get("density_fit_diagnostics",[]))
    summary.append({"stage":item["stage"],"row":item["row"]["id"],
        "score_squared_error":diag["score_squared_error"],"actual_particles":diag.get("actual_particle_count"),
        "fit_calls":diag.get("work_accounting",{}).get("offline_fit_calls",0),
        "density_time_fits":len(fit_rows),
        "boundary_fits":sum(bool(x["boundary_active"]) for x in fit_rows),
        "underflow_fits":sum(bool(x["objective_underflow"]) for x in fit_rows),
        "max_normalized_shape_residual":max((x["normalized_shape_residual"] for x in fit_rows),default=None),
        "stored_numerical_validity":result["numerical_validity"],
        "fit_validity_audit":"objective_underflow_veto" if any(x["objective_underflow"] for x in fit_rows) else "no_underflow_veto",
        "reused":item["reused_completed_evidence"]})
for regime in ("weak","curved"):
    claims=[x for x in rows if x["stage"]==regime+"-evaluation"]
    a,b=(x["result"]["diagnostics"] for x in claims)
    assert a["fit_digest"]==b["fit_digest"]
    assert a["fit_seed_records"]["iapf_final_process"]!=b["fit_seed_records"]["iapf_final_process"]
charges=[x for directory in (first,second) for x in json.loads((directory/"attempt-accounting.json").read_text())]
total=sum(x["row_attempts"]+x["recursive_fits"] for x in charges)
assert total==64==manifest["charged_row_or_recursive_fit_attempts"]
assert manifest["status"]=="partial_budget_exhausted"
assert all(not x["numerical_work_started"] for x in manifest["budget_deferred_rows"])
verification={"schema":"iapf_fit_calibration_verification_v1","engineering_verdict":"pass",
    "complete_unique_rows":len(rows),"source_checked_studies":len(checked_states),"charged_attempts":total,
    "numerical_row_attempts":len(charges),"recursive_fit_calls":sum(x["recursive_fits"] for x in charges),
    "conditional_heuristics":json.loads((second/"conditional-heuristics.json").read_text()),
    "rows_with_objective_underflow":sum(x["underflow_fits"]>0 for x in summary),
    "selected_claim_rows_with_objective_underflow":sum(x["underflow_fits"]>0 and "evaluation" in x["stage"] for x in summary),
    "statistically_supported_ranking":False,"default_ready":False,
    "missing_evidence":["curved heuristic comparators","baseline iAPF on the claim data","powered independent replications"],
    "runtime":"CPU-only post-run inspection; CUDA_VISIBLE_DEVICES=-1; no numerical kernels",
    "rows":summary}
output.write_text(json.dumps(verification,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in verification.items() if k not in ("rows","conditional_heuristics")},indent=2))

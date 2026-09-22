"""CPU-only post-run provenance/reference/selection audit; no new experiment."""
import json
from pathlib import Path
from bayesfilter.score_study.contracts import digest, validate_result
from bayesfilter.score_study.coordinator import fingerprint
from bayesfilter.score_study.registry import default_registry

root=Path(__file__).resolve().parent/"launch02"
read=lambda p:json.loads(p.read_text())
manifest=read(root/"run-manifest.json")
rows=read(root/"consumer-evidence.json")
failures=read(root/"invalid-fits.json")
charges=read(root/"attempt-accounting.json")
assert manifest["status"] in ("completed","completed_with_numerical_vetoes")
assert len(rows)==manifest["numerical_rows"] and len(failures)==manifest["invalid_rows"]
assert len(rows)+len(failures)==len(charges)
assert len({(c["stage"],c["row"]) for c in charges})==len(charges)
assert sum(c["charges"] for c in charges)+manifest["prior_charges"]==manifest["charged_attempts"]<=280
registry=default_registry()
specs={}
for stage in manifest["studies"]:
    spec=read(root/(stage["name"]+"-study.json"))
    state=read(Path(stage["path"])/"state.json")
    assert fingerprint(spec,registry)==state["fingerprint"]==stage["fingerprint"]
    specs[stage["name"]]={r["id"]:r for r in spec["rows"]}

references={}
refinements=[]
objective_rows={"density_l2":0,"relative_shape":0}
for row in rows:
    result=read(Path(row["path"]))
    spec=specs[row["stage"]][row["row"]]
    assert digest(result)==row["digest"]
    validate_result(result,spec,registry)
    diag=result["diagnostics"]
    key=row["regime"],row["dataset"]
    reference=result["oracle_value"],result["oracle_score"],diag["executed_observation_digest"]
    if key in references:
        assert references[key]==reference
    references[key]=reference
    refinement=[diag[k] for k in ("reference_mesh_relative_error","reference_domain_relative_error","reference_max_tail_mass")]
    assert refinement[0]<=1e-7 and refinement[1]<=1e-7 and refinement[2]<=1e-9
    refinements.append(refinement)
    if row["proposal"]=="iapf":
        objective=spec["iapf"].get("fit_objective","density_l2")
        assert diag["fit_objective"]==objective
        objective_rows[objective]+=1
        assert row["fit_summary"]["underflow_steps"]==0
        for step in [s for rec in diag["fit_iterations"] for s in rec.get("density_fit_diagnostics",[])]:
            assert len(step)==11 and step[10]==step[1 if objective=="relative_shape" else 0]
        assert diag["fit_observation_digest"]==diag["executed_observation_digest"]

for table in read(root/"conditional-heuristics.json"):
    for entry in table["datasets"]:
        assert entry["heuristic_table_complete"]
        for arm in ("selected","baseline"):
            pair=[r for r in rows if r["regime"]==table["regime"] and r["dataset"]==entry["dataset"] and r["arm"]==arm]
            assert entry[arm]["complete"]==(len(pair)==2)
            if len(pair)==2:
                assert pair[0]["fit_digest"]==pair[1]["fit_digest"]
                assert pair[0]["final_seeds"]!=pair[1]["final_seeds"]
                assert entry[arm]["mse"]==sum(r["score_squared_error"] for r in pair)/2
                assert set(entry[arm]["observed_losses_to"])=={k for k,v in entry["heuristics"].items() if v<entry[arm]["mse"]}
        for rep in range(2):
            pair=[r for r in rows if r["regime"]==table["regime"] and r["dataset"]==entry["dataset"] and r["replicate"]==rep and r["arm"] in ("selected","baseline")]
            if len(pair)==2:
                assert pair[0]["final_seeds"]==pair[1]["final_seeds"]

summary=dict(status="pass",gpu_intentionally_hidden=True,numerical_experiment=False,
    checked_studies=len(specs),checked_results=len(rows),checked_attempts=len(charges),
    invalid_rows=len(failures),objectives=objective_rows,
    reference_maxima={key:max(r[i] for r in refinements) for i,key in enumerate(("mesh","domain","tail"))})
(root.parent/"saved-run-audit.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))

"""Recompute rank assessments from saved observations without sampler execution.

Changing source/analysis after inspection is exploratory. Other engines retain
their original assessments and explicitly report unsupported reassessment.
"""
from pathlib import Path

from .designs import ValidationDesign, seed_for
from .storage import read_json, write_json, file_hash


def assess(root, output):
    from .engines.statistics import rank_uniform_test, two_sample_test
    root,output=Path(root),Path(output)
    if output.exists():
        raise ValueError("fresh assessment output required")
    index=read_json(root/"run_index.json")
    rows=[]
    for job in index["plan"]["jobs"]:
        d=ValidationDesign.from_payload(job["design"])
        execution=index["jobs"].get(d.design_id,{})
        if execution.get("status")!="complete":
            rows.append({"design_id":d.design_id,"status":"unavailable"}); continue
        if file_hash(execution["result"])!=execution["result_sha256"]:
            raise ValueError("corrupt source result")
        original=read_json(execution["result"])["assessment"]
        if d.engine=="invariance":
            ranks=original["ranks"]
            family=original["multiplicity"]
        elif d.engine=="sbc":
            complete=[r for r in original["datasets"] if r["status"]=="complete"]
            ranks={k:[r["ranks"][k] for r in complete] for k in complete[0]["ranks"]} if complete else {}
            family=max(d.multiplicity,len(ranks))
        else:
            rows.append({"design_id":d.design_id,"status":"reassessment_not_supported",
                         "original_result":execution["result"]}); continue
        tests={key:rank_uniform_test(values,d.rank_draws,null_draws=d.null_draws,
            seed=seed_for(d.seed,d.design_id,"null",key) if d.engine=="invariance" else seed_for(d.seed,d.design_id,key,"null"),
            alpha=d.alpha,multiplicity=family) for key,values in ranks.items()}
        two={}
        if d.engine=="invariance":
            obs=original["two_sample_observations"]
            two={key:two_sample_test(obs["left"][key],obs["right"][key],
                 seed=seed_for(d.seed,d.design_id,"two",key),permutations=d.null_draws,
                 alpha=d.alpha,multiplicity=family) for key in obs["left"]}
        rows.append({"design_id":d.design_id,"status":"reassessed","rank_tests":tests,
            "two_sample_tests":two,"conditional_tests_only":original.get("conditional_tests_only",False),
            "observations_result":execution["result"],"observations_sha256":execution["result_sha256"]})
    from .execution import source_state
    result={"schema":"bayesfilter.inference_validation_reassessment.v1","rows":rows,
        "source":source_state(),"source_run":str(root),"analysis_role":"exploratory_reassessment",
        "sampler_executed":False,"accuracy_established":False}
    write_json(output,result)
    return result

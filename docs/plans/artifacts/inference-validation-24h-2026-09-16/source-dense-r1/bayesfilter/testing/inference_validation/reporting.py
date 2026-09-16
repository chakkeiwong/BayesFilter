"""Evidence coverage and findings without a universal 'validated' flag."""
from __future__ import annotations
from pathlib import Path
from .storage import read_json,write_json,file_hash


def assessment_complete(assessment):
    """Recognize unavailable outputs in older saved assessments as well."""
    if "assessment_complete" in assessment:
        return assessment["assessment_complete"]
    if assessment.get("finding") in {None,"incomplete","calibration_incomplete",
            "no_verified_members","posterior_incomplete","unavailable","invalid"}:
        return False
    records=assessment.get("replications")
    if isinstance(records,list):
        return (len(records)==assessment["planned"] and bool(records)
                and all(r["members"] and all(m.get("assessment",{}).get("finding") in
                    {"within_descriptive_tolerance","reference_discrepancy"} for m in r["members"])
                    for r in records))
    return True


def report(root):
    root=Path(root); index=read_json(root/"run_index.json"); rows=[]
    from .execution import source_state
    stale=source_state()["identity"]!=index["source"]["identity"]
    for job in index["plan"]["jobs"]:
        identity=job["design"]["design_id"]; execution=index["jobs"].get(identity,{"status":"not_run"})
        finding=None; status=execution["status"]; response=None; assessed=False
        if status=="complete":
            if not Path(execution["result"]).is_file() or file_hash(execution["result"])!=execution["result_sha256"]:
                status="invalid_artifact"
            else:
                result=read_json(execution["result"])
                if result["design_identity"]!=job["identity"]:
                    status="invalid_artifact"
                else:
                    finding=result["assessment"].get("finding")
                    response=result.get("test_response")
                    assessed=assessment_complete(result["assessment"])
        from .designs import ValidationDesign
        coverage_key=ValidationDesign.from_payload(job["design"]).coverage_key()
        rows.append({"design_id":identity,**coverage_key,"execution_status":status,"finding":finding,
            "assessment_complete":assessed,
            "source_status":"stale" if stale else "matches_current_source",
            "test_response":response,
            "seconds":sum(a["elapsed_seconds"] for a in execution.get("attempts",[])),
            "reason":execution.get("reason",job.get("reason")),"result":execution.get("result")})
    coverage=[]
    for required in index["plan"]["required_coverage"]:
        matching=[r for r in rows if all(r.get(k)==v for k,v in required.items())]
        coverage.append({"required":required,"designs":[r["design_id"] for r in matching],
            "status":"stale" if stale and matching else "executed" if any(r["execution_status"]=="complete" for r in matching) else "uncovered",
            "assessment_complete":not stale and any(r["execution_status"]=="complete" and r["assessment_complete"] for r in matching)})
    result={"schema":"bayesfilter.inference_validation_report.v1","rows":rows,"coverage":coverage,
        "all_required_executed":bool(coverage) and all(c["status"]=="executed" for c in coverage),
        "all_required_assessed":bool(coverage) and all(c["assessment_complete"] for c in coverage),
        "interpretation":"Execution, statistical finding and expected defect detection are separate; no universal validation claim."}
    write_json(root/"report.json",result)
    text=["# Inference validation results","",result["interpretation"],"",
        "| Design | Target / route | Engine | Execution / source | Complete assessment | Finding | Test response | Seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |"]
    for row in rows:
        link=f"[{row['design_id']}]({row['result']})" if row["result"] else row["design_id"]
        text.append(f"| {link} | {row['target']} / {row['route']} | {row['engine']} | {row['execution_status']} / {row['source_status']} | {row['assessment_complete']} | {row['finding'] or row['reason'] or 'not assessed'} | {(row['test_response'] or {}).get('status','unassessed')} | {row['seconds']:.2f} |")
    text.extend(["","## Required coverage","","| Requirement | Status | Evidence |","| --- | --- | --- |"])
    for c in coverage:
        text.append(f"| {', '.join(f'{k}={v}' for k,v in c['required'].items())} | {c['status']} | {', '.join(c['designs']) or 'none'} |")
    text.extend(["","Unassessed or failed fits remain in their original denominator. Short-run differences are descriptive. Statistical non-rejection does not establish accuracy or mixing.",""])
    (root/"report.md").write_text("\n".join(text))
    return result

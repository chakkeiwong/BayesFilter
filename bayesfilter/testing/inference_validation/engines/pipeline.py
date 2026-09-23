"""Independent assessment of native candidate records and posterior outputs."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import time
import copy
import numpy as np

from ..catalog import get_target
from ..designs import seed_for
from ..references import analytic
from ..storage import read_json,read_tensor,write_json
from .statistics import accuracy_assessment,binomial_interval


def stopped_intervals(member, spec, params, data):
    """Evaluate the actual final controller check, including unfavorable stops."""
    from scipy import stats
    references={(spec.parameters[index], kind): truth for (kind, index), truth in
                analytic.exact_functionals(spec.target_id,params,data).items()}
    if spec.target_id == "mixture":
        # P(X < 0) for w N(-a,1) + (1-w) N(a,1), in model coordinates.
        a, w = params.get("separation", 5.), params.get("weight", .3)
        references[("left_mode_probability", "mean")] = float(
            w * stats.norm.cdf(a) + (1-w) * stats.norm.cdf(-a))
    checks=member["posterior"]["retained_checks"]
    final=checks[-1] if checks else {}
    diagnostic=final.get(final.get("diagnostic_role","modern_rhat")) or {}
    estimates=diagnostic.get("precision",{}).get("targets",())
    rows=[]
    for estimate in estimates:
        truth=references.get((estimate["name"], estimate["kind"]))
        se=estimate["mcse"]
        available=truth is not None and estimate["valid"] and se is not None
        error=estimate["estimate"]-truth if truth is not None and estimate["estimate"] is not None else None
        available=available and error is not None
        rows.append({"name":estimate["name"],"kind":estimate["kind"],"reference":truth,
            "estimate":estimate["estimate"],"error_at_stop":error,"reported_mcse":se,
            "available":available,"covered":available and abs(error)<=stats.norm.ppf(.975)*se})
    return {"quantities":rows,"runtime_passed":member["posterior"]["passed"],
        "warmup_cap_hit":member["posterior"]["warmup_cap_hit"],
        "retained_cap_hit":member["posterior"]["retained_cap_hit"],
        "nominal_interval":"estimate +/- normal 97.5% quantile * reported MCSE",
        "sequential_coverage_established":False,
        "interpretation":"coverage at actual stopping is measured, never inferred from fixed-size MCSE calibration"}


def check_inventory(payload):
    """Oracle uses recorded stages, exact settings and receipts, not the scheduler."""
    candidates={c["candidate_id"]:c for c in payload["candidates"]}
    verified=set(payload["verified_candidate_ids"])
    receipts=payload.get("verification_receipts",[])
    failures=[]
    if payload.get("nominee_id") is not None: failures.append("unexpected_nominee")
    observations=payload.get("observations",[])
    if len(candidates) != len(payload["candidates"]): failures.append("duplicate_candidate")
    successful_receipts=set()
    streams = set()
    seeds = set()
    for rec in receipts:
        if rec.get("decision") in {"passed","acceptance_in_band"} and rec.get("evidence_validity", "valid") == "valid" and not rec.get("hard_vetoes") and not rec.get("promotion_vetoes") and rec.get("stage", "verification") == "verification":
            successful_receipts.add(rec["candidate_id"])
        stream = rec.get("stream_id")
        seed = tuple(rec.get("seed_lineage", ()))
        if not stream or stream in streams or not seed or seed in seeds:
            failures.append("missing_or_reused_stream:" + rec["candidate_id"])
        streams.add(stream)
        seeds.add(seed)
        c = candidates.get(rec["candidate_id"])
        if c is None or any(rec.get(key) != c.get(expected) for key, expected in
            (("epsilon", "epsilon"), ("exact_l", "leapfrog_steps"),
             ("mass_signature", "mass_signature"), ("candidate_record_hash", "candidate_record_hash"))):
            failures.append("cross_pair_verification:" + rec["candidate_id"])
    for cid in verified:
        if cid not in candidates:
            failures.append("missing_candidate:" + cid)
            continue
        c=candidates[cid]
        own=[o for o in observations if o["candidate_id"]==cid]
        stages={o["stage"] for o in own}
        if not {"measurement","verification"}<=stages: failures.append("missing_own_fresh_stages:"+cid)
        if cid not in successful_receipts: failures.append("missing_passing_receipt:"+cid)
        parent=c.get("parent_candidate_id")
        if parent and (parent not in candidates or candidates[parent]["leapfrog_steps"]!=c["leapfrog_steps"]):
            failures.append("repair_changed_L:"+cid)
    # A shared-invalidity event legitimately retires historical passing receipts.
    if payload.get("completion_status")!="shared_invalidity":
        states=payload.get("candidate_states",{})
        expected={cid for cid,state in states.items() if state=="verified"}
        if verified!=expected: failures.append("verified_set_incomplete")
        if verified != successful_receipts: failures.append("passing_receipts_not_retained")
    works = {w["work_item_id"]: w for w in payload.get("work_items", ())}
    completed = {key for key, work in works.items() if work["status"] == "completed"}
    terminal_shared = set()
    if payload.get("completion_status") == "shared_invalidity":
        for observation in observations:
            work = works.get(observation["work_item_id"], {})
            if (work.get("status") == "interrupted"
                    and observation["candidate_id"] == work.get("candidate_id")
                    and observation["stage"] == work.get("stage")
                    and observation["observation"].get("evidence_validity") == "shared_execution_invalid"):
                terminal_shared.add(observation["work_item_id"])
    observed = [o["work_item_id"] for o in observations]
    if completed | terminal_shared != set(observed) or len(observed) != len(set(observed)):
        failures.append("missing_or_duplicated_observation")
    return {"finding":"inventory_passed" if not failures else "inventory_discrepancy",
            "failures":failures,"candidate_count":len(candidates),"verified_count":len(verified),
            "numerical_correctness_established":False}


def controller_experiment(design,root):
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCCandidateSetScope,HMCControllerConfig,HMCTuningCandidateSetController
    from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload
    scope=HMCCandidateSetScope(scope_id="validation",search_id="controller",target_signature="scripted",
        mass_signature="identity",coordinate_system="ordinary",start_bank_signature="scripted_starts",
        warmup_protocol="none",epsilon_domain=(.05,2.),repair_factor=2.,max_repairs_per_family=2)
    cfg=HMCControllerConfig(primary_l_grid=(3,5),epsilon_by_l=((3,(.5,)),(5,(.5,))),
        total_budget_units=7,repair_reserve_units=3,evidence_rungs=(1,2,4))
    def make_provider(rhat):
        def observe(work,candidate):
            if candidate.leapfrog_steps==3:
                if candidate.parent_candidate_id is None: return {"decision":"repair_step_lower","acceptance":.2,"rhat":rhat}
                if work.stage=="measurement" and work.evidence_rung<2: return {"decision":"inconclusive_evidence","acceptance":.7,"rhat":rhat}
            return {"decision":"passed","acceptance":.7,"rhat":rhat}
        return observe
    base=HMCTuningCandidateSetController(scope,cfg).run(make_provider(1.))
    paused=HMCTuningCandidateSetController(scope,cfg).run(make_provider(None),max_work_items=2)
    restored=HMCTuningCandidateSetController.from_result_payload(candidate_set_result_payload(paused)).run(make_provider(None))
    high=HMCTuningCandidateSetController(scope,cfg).run(make_provider(8.))
    payload=candidate_set_result_payload(base)
    if design.scenario.control=="drop_candidate": payload["verified_candidate_ids"]=list(payload["verified_candidate_ids"])[1:]
    if design.scenario.control=="cross_l_epsilon":
        payload["verification_receipts"]=[dict(r,exact_l=99) for r in payload["verification_receipts"]]
    if design.scenario.control=="lost_chunk": payload["observations"]=payload["observations"][1:]
    result=check_inventory(payload)
    invariants={"all_members_retained":len(payload["verified_candidate_ids"])==2,
                "released_budget_completes":base.completion_status=="complete" and base.budget_used_units==7,
                "resume_matches":restored.verified_candidate_ids==base.verified_candidate_ids,
                "rhat_no_effect":high.verified_candidate_ids==base.verified_candidate_ids}
    result.update(invariants=invariants,evidence_class="controller_double",native_payload=payload)
    result["finding"]="inventory_passed" if all(invariants.values()) and not result["failures"] else "inventory_discrepancy"
    write_json(root/"controller.json",result)
    return result


def run_replication(design, root, replication, deadline=None):
    """One complete fit and independent assessment, reusable across processes."""
    from ..procedures import execute_pipeline

    path = Path(root) / f"replication-{replication:04d}"
    if (path / "independent_assessment.json").exists():
        return read_json(path / "independent_assessment.json")
    data=design.options.get("data")
    output=execute_pipeline(design,path,data=data,fit_id=replication,deadline=deadline)
    payload=read_json(output["tuning_path"])
    # Mutations act on a copy of observations. Native tuning authority remains intact.
    payload=copy.deepcopy(payload)
    if design.scenario.control=="drop_candidate": payload["verified_candidate_ids"]=payload["verified_candidate_ids"][1:]
    if design.scenario.control=="cross_l_epsilon":
        payload["verification_receipts"]=[dict(r,exact_l=99) for r in payload["verification_receipts"]]
    if design.scenario.control=="lost_chunk" and design.engine=="search": payload["observations"]=payload["observations"][1:]
    inventory=check_inventory(payload)
    members=[]
    spec=get_target(design.scenario.target)
    for member in output["members"]:
        if member["status"]!="assessed":
            members.append(member); continue
        draws=read_tensor(member["draws_path"]).numpy()
        reference=analytic.model_coordinates(spec.target_id,analytic.draw(spec.target_id,
            max(4096,design.draws),seed_for(design.seed,design.design_id,replication,member["candidate_id"],"reference"),
            design.scenario.parameters,data))
        assessment=accuracy_assessment(draws,reference,tolerance=design.accuracy_tolerance,
            finite_variance=spec.finite_variance)
        reported=member["posterior"]["passed"]
        row={"candidate_id":member["candidate_id"],"L":member["L"],"epsilon":member["epsilon"],
            "assessment":assessment,"runtime_checks_passed":reported,
            "false_favorable_screen_observed":reported and assessment["finding"]=="reference_discrepancy",
            "warmup_exclusion_matches":member["warmup_exclusion_matches"],
            "duplicate_chains":member["duplicate_chains"],
            "stopped_intervals":stopped_intervals(member,spec,design.scenario.parameters,data),
            "warmup_count":member["posterior"]["warmup_results_per_chain"],
            "retained_count":member["recorded_retained_count"],"member_record":member}
        if design.options.get("fixed_comparator") is not None:
            from .stopping import arm_quantities
            comparator=member.get("fixed_comparator",{})
            fixed=(read_tensor(comparator["draws_path"]).numpy() if comparator.get("status")=="assessed"
                   else np.empty((0,draws.shape[1],draws.shape[2])))
            interval_options = {"jit_compile": design.device == "gpu",
                "method": design.options.get("posterior_precision_method", "lugsail")}
            row["stopping_pair"]={"stopped":arm_quantities(draws,spec,design.scenario.parameters,data,**interval_options),
                "fixed":arm_quantities(fixed,spec,design.scenario.parameters,data,**interval_options),
                "fixed_status":comparator.get("status","unavailable")}
        members.append(row)
    record={"replication":replication,"inventory":inventory,"members":members,
                    "tuning_completion":output["completion"],
                    "pipeline":str(path/"pipeline.json")}
    if design.options.get("member_rule") == "shortest_verified_l":
        record["selection"] = output["selection"]
    write_json(path/"independent_assessment.json",record)
    return record


def run(design,root,deadline=None):
    if design.scenario.route=="controller": return controller_experiment(design,root)
    records=[]
    for replication in range(design.replications):
        if deadline and time.monotonic()>=deadline: break
        records.append(run_replication(design, root, replication, deadline))
    result=summarize_replications(design,records)
    write_json(root/"assessment.json",result)
    return result


def summarize_replications(design, records):
    """Keep calls, usable posterior outputs and full assessment distinct."""
    if design.options.get("member_rule") == "shortest_verified_l":
        return _summarize_siblings(design, records)
    # Independent replication-level summaries; never count siblings as iid trials.
    completed=sum("execution_failure" not in row for row in records)
    favorable=sum(any(m.get("false_favorable_screen_observed") for m in r["members"])
                  for r in records if "execution_failure" not in r)
    members=[m for r in records for m in r["members"]]
    requested=[m for m in members if m.get("status")!="unassessed_by_design"]
    assessed=[m for m in members if "assessment" in m]
    available=[m for m in assessed if m["assessment"]["finding"] in
               {"within_descriptive_tolerance","reference_discrepancy"}]
    complete=(completed==design.replications and bool(members)
              and all(any(m.get("status")!="unassessed_by_design" for m in r["members"]) for r in records)
              and len(available)==len(requested))
    discrepancies = any(r["inventory"]["failures"] for r in records) or any(
        not m["warmup_exclusion_matches"] or m["duplicate_chains"] or
        m["assessment"]["finding"] in {"reference_discrepancy", "invalid"} for m in assessed)
    finding = "pipeline_discrepancy" if discrepancies else (
        "no_verified_members" if not members else "pipeline_assessed")
    if not discrepancies:
        if completed != design.replications or len(assessed) != len(requested): finding="incomplete"
        elif members and not complete: finding="posterior_incomplete"
    # One predeclared candidate group per replication, with missing members
    # counted as unavailable. Siblings never inflate binomial denominators.
    spec=get_target(design.scenario.target)
    interval_groups={name+":"+kind:{"covered":0,"available":0}
                     for name in spec.parameters
                     for kind in (("quantile","mean") if spec.finite_variance else ("quantile",))}
    interval_groups.update({name+":mean": {"covered":0,"available":0}
                            for name in design.options.get("global_quantities", ())})
    for rep in records:
        if "execution_failure" in rep:
            continue
        group=sorted((m for m in rep["members"]
                      if m.get("status") != "unassessed_by_design"
                      and (design.options.get("member_rule","declared_l_first")=="first_verified"
                           or m.get("L")==design.member_l)),
                     key=lambda m:m["candidate_id"])
        if group and "stopped_intervals" in group[0]:
            for row in group[0]["stopped_intervals"]["quantities"]:
                key=row["name"]+":"+row["kind"]
                counts=interval_groups[key]
                counts["covered"]+=int(row["covered"])
                counts["available"]+=int(row["available"])
    for counts in interval_groups.values():
        counts["planned"]=design.replications
        counts["unavailable"]=design.replications-counts["available"]
        counts["coverage_interval"]=binomial_interval(counts["covered"],design.replications)
    result={"replications":records,"completed":completed,"planned":design.replications,
            "attempted_replications":len(records),
            "execution_failures":sum("execution_failure" in row for row in records),
            "interval_coverage_at_stop":interval_groups,"coverage_member_L":design.member_l,
            "verified_members":len(members),"assessed_members":len(assessed),
            "requested_members":len(requested),
            "posterior_output_members":len(available),
            "posterior_unavailable_members":len(requested)-len(available),
            "unassessed_by_design_members":len(members)-len(requested),
            "all_members_without_posterior_output":len(members)-len(available),
            "member_accounting":"requested unavailable excludes siblings unassessed by design",
            "all_verified_members_assessed":len(available)==len(members) and bool(members),
            "assessment_complete":complete,
            "search_complete":completed==design.replications and all(
                r["tuning_completion"]=="complete" for r in records),
            "posterior_available_replications":sum(any(m in available for m in r["members"]) for r in records),
            "actual_controller_stopping_test":design.engine=="stopping",
            "independent_unit":"complete pipeline replication; candidate siblings clustered",
            "false_favorable_replications":favorable,
            "false_favorable_interval":binomial_interval(favorable,completed) if completed else None,
            "false_favorable_interpretation":"observed false-favorable outcomes per executed replication; missing outputs remain unavailable, never successful",
            "finding":finding,
            "accuracy_established":False}
    if design.options.get("fixed_comparator") is not None:
        from .stopping import summarize_pairs
        pairs=[]
        for record in records:
            if "execution_failure" in record:
                pairs.append({})
                continue
            group=sorted((m for m in record["members"]
                          if m.get("status") != "unassessed_by_design"
                          and (design.options.get("member_rule","declared_l_first")=="first_verified"
                               or m.get("L")==design.member_l)),
                         key=lambda m:m["candidate_id"])
            pairs.append(group[0].get("stopping_pair",{}) if group else {})
        references=analytic.exact_functionals(spec.target_id,design.scenario.parameters,design.options.get("data"))
        names=[spec.parameters[index]+":"+kind for kind,index in references]
        if spec.target_id=="mixture": names.append("left_mode_probability")
        result["stopped_versus_fixed"]=summarize_pairs(pairs,design.replications,declared_names=names)
        result["comparison_complete"]=(completed==design.replications and all(p.get("fixed_status")=="assessed" for p in pairs))
        result["assessment_complete"]=complete and result["comparison_complete"]
        if not result["assessment_complete"] and not discrepancies: result["finding"]="incomplete"
    return result


def _summarize_siblings(design, records):
    """Apply the established full-fit denominator separately to each ordinal slot."""
    count = design.options["posterior_member_count"]
    options = {key: value for key, value in design.options.items() if key != "posterior_member_count"}
    options["member_rule"] = "first_verified"
    single = replace(design, options=options)
    # Reuse total inventory/accounting, then replace single-member summaries.
    result = summarize_replications(single, records)
    for key in ("interval_coverage_at_stop", "coverage_member_L", "stopped_versus_fixed"):
        result.pop(key, None)
    selections = []
    for row in records:
        if "selection" not in row and "execution_failure" not in row:
            raise ValueError("missing predeclared sibling selection in completed fit")
        ids = row.get("selection", {}).get("candidate_ids", [])
        if len(ids) > count or len(set(ids)) != len(ids):
            raise ValueError("invalid predeclared sibling inventory")
        by_id = {m["candidate_id"]: m for m in row["members"]}
        if any(cid not in by_id for cid in ids):
            raise ValueError("selected sibling missing from candidate inventory")
        selections.append([by_id[cid] for cid in ids])
    slots = {}
    for index in range(count):
        projected = [dict(row, members=group[index:index+1]) for row, group in zip(records, selections)]
        summary = summarize_replications(single, projected)
        selected = [group[index] for group in selections if len(group) > index]
        slots[str(index+1)] = {key: summary[key] for key in (
            "interval_coverage_at_stop", "stopped_versus_fixed", "assessment_complete", "comparison_complete",
            "posterior_available_replications", "posterior_unavailable_members") if key in summary}
        slots[str(index+1)].update(planned=design.replications,
            selected_members=[{"replication": row["replication"], "candidate_id": group[index]["candidate_id"],
                               "L": group[index]["L"]} for row, group in zip(records, selections) if len(group) > index],
            selection_shortfall=design.replications-len(selected),
            posterior_checks_passed=sum(bool(group[index].get("runtime_checks_passed"))
                for row, group in zip(records, selections) if len(group)>index and "execution_failure" not in row))
        available = sum(group[index].get("assessment", {}).get("finding") in
            {"within_descriptive_tolerance", "reference_discrepancy"}
            for row, group in zip(records, selections) if len(group)>index and "execution_failure" not in row)
        slots[str(index+1)].update(posterior_available_replications=available,
                                  posterior_unavailable_slots=design.replications-available)
    result.update(member_slot_assessments=slots,
        declared_member_slots=design.replications*count,
        selected_member_slots=sum(len(group) for group in selections),
        selection_shortfall=design.replications*count-sum(len(group) for group in selections),
        coverage_scope="separate ordinal member slots; complete-fit denominators; no pooled sibling estimate",
        assessment_complete=all(slot["assessment_complete"] for slot in slots.values()))
    if design.options.get("fixed_comparator") is not None:
        result["comparison_complete"] = all(slot["comparison_complete"] for slot in slots.values())
    if not result["assessment_complete"] and result["finding"] != "pipeline_discrepancy":
        result["finding"] = "incomplete"
    return result

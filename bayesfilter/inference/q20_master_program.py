"""Executable q20 campaign coordinator; no accelerator imports in this process."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from bayesfilter.inference.q20_campaign_runtime import Campaign, CampaignBudgetError, atomic_json
from bayesfilter.inference.q20_production_config import digest, training_cohort


def forecast_campaign(config, pricing):
    """Declared maximum work counts times measured costs; never price gaps at zero."""
    if pricing["config_hash"] != digest(config):
        raise ValueError("pricing protocol differs")
    p,h,t,c=config["posterior"],config["tuning"],config["training"],config["comparison"]
    rows=[r for r in pricing["hmc"] if r["kind"]!="classical_preparation"]
    if not rows:
        raise ValueError("HMC costs were not measured")
    # State-dependent runtime and future learned weights remain forecast risks.
    step=max(r["steady_seconds"] for r in rows)
    compile_cost=max(r["first_seconds"] for r in rows)
    preparation=max(r["wall_seconds"] for r in pricing["hmc"] if r["kind"]=="classical_preparation")
    levels=len(t["betas"])-1
    charts=config["ensemble"]["charts"]
    scopes=2+levels+charts*levels
    repetitions=1+c["confirmation_replicates"]
    # A work item is one pilot/measurement/verification. Max rung covers
    # uncertainty extensions; total_budget_units bounds every such work item.
    work_results=h["startup"]+max(h["pilot"],h["measurement"],h["verification"])*max(h["evidence_rungs"])
    tuning=scopes*(h["total_budget_units"]*work_results*step+len(h["l_grid"])*(1+len(h["evidence_rungs"]))*compile_cost)+levels*preparation
    reverify=c["confirmation_replicates"]*scopes*(2*sum(h["evidence_rungs"])*work_results*step+compile_cost)
    transitions=p["warmup_max"]+p["retained_max"]
    # Replica exchange is measured separately below; it includes all levels,
    # chart transforms and swaps. Single-member runs use measured four-chain work.
    exchange=pricing["exchange_steady_seconds"]
    posterior=repetitions*(transitions*(3*step+2*exchange)+3*compile_cost+2*pricing["exchange_first_seconds"])
    reference_batches=config["reference"]["banks"]*max(config["reference"]["rungs"])/config["reference"]["batch_size"]
    reference=pricing["reference_batch_first_seconds"]+reference_batches*pricing["reference_batch_steady_seconds"]
    reference+=pricing["reference_analysis_seconds_per_row"]*reference_batches*config["reference"]["batch_size"]
    checks=math.ceil(p["warmup_max"]/p["warmup_chunk"])+math.ceil(p["retained_max"]/p["retained_chunk"])
    analysis=repetitions*len(c["methods"])*(checks+1)*pricing["posterior_analysis_seconds"]
    startup=pricing["worker_initialization_seconds"]+pricing.get("process_overhead_seconds",0.)+2*config["execution"]["termination_grace_seconds"]
    reference+=startup
    tuning+=scopes*startup
    reverify+=c["confirmation_replicates"]*scopes*startup
    posterior+=repetitions*len(c["methods"])*startup
    analysis+=repetitions*startup
    factor=config["budget"]["forecast_safety_factor"]
    raw={"reference":reference,"tuning":tuning,"confirmation_reverification":reverify,
         "posterior":posterior,"comparison_analysis":analysis}
    # One complete maximum-cost scope is the declared localized repair reserve.
    repair=max(tuning/scopes, transitions*max(step,exchange), reference)
    reserves={key:factor*value for key,value in raw.items()}
    reserves["localized_repair"]=factor*repair
    minimum=pricing["training_quote"]["minimum_cohort_seconds"]+factor*startup
    return {"schema":"bayesfilter.q20.complete_cost_forecast.v1","config_hash":digest(config),
        "reserves":reserves,"training_floor_seconds":minimum,
        "training_cap_seconds":pricing["training_quote"]["full_training_cap_seconds"]+factor*startup,
        "minimum_complete_seconds":minimum+sum(reserves.values()),
        "forecast_safety_factor":factor,"factor_status":"engineering_hypothesis",
        "scope_count":scopes,"status":"all_required_stages_forecast",
        "limitations":["few timings do not bound compile or runtime tails",
                       "new learned maps can change numerical behavior",
                       "precision and reference qualification may still fail at declared caps"]}


def choose_maps(config, checkpoint, *, schedule="continuation", count=None):
    cohort=json.loads(Path(checkpoint).read_text())["cohort"]
    chosen=[]
    used_roots=set()
    # Operational ordering established before comparisons. No loss/ESS ranking.
    for candidate in training_cohort(config):
        if candidate["schedule"] != schedule:
            continue
        item=cohort.get(candidate["id"])
        if item is None or candidate["root"] in used_roots:
            continue
        exports=item["exports"]
        valid=True
        for beta in config["training"]["betas"][1:]:
            if schedule == "direct" and beta != 1.:
                continue
            if str(beta) not in exports:
                valid=False
                break
            row=json.loads(Path(exports[str(beta)]).read_text())
            if not row["assessment"]["map_reliability"]["passed"]:
                valid=False
            if config["role"]!="smoke" and not row["assessment"]["decision"]["development_eligible"]:
                valid=False
        if valid:
            chosen.append({"candidate":candidate,"exports":exports})
            used_roots.add(candidate["root"])
        if len(chosen)==(config["ensemble"]["charts"] if count is None else count):
            break
    return chosen


def selected_member(tuned):
    if tuned["status"]!="complete" or not tuned["verified_members"]:
        return None
    key=sorted(tuned["verified_members"])[0]
    return {"candidate_id":key,"path":tuned["verified_members"][key],
            "selection_policy":"lexicographic_verified_id_operational_not_efficiency_ranking"}


class StageIncomplete(RuntimeError):
    pass


def execute_master(config, root, *, repo, allowance=None, fixture=False, stop_after=None):
    campaign=Campaign(root,repo=repo,config=config,allowance=allowance)
    with campaign.locked():
        return _execute(campaign,fixture=fixture,stop_after=stop_after)


def _execute(campaign, *, fixture=False,stop_after=None):
    config=campaign.config
    diagnostic_cap=config["execution"]["diagnostic_attempt_seconds"]
    gpu=None
    def finish(status, **details):
        campaign.state.update(status=status,details=details,production_qualified=False)
        campaign.save()
        result={"status":status,"details":details,"campaign_root":str(campaign.root),
            "remaining_campaign_seconds":campaign.remaining(),"remaining_diagnostic_seconds":campaign.remaining(True),
            "declared_confirmation_checks_passed":bool(details.get("confirmation_passed")),
            "production_qualified":False,
            "method_ranking":"not_estimated"}
        atomic_json(campaign.root/"result.json",result)
        return result
    if config["role"]!="smoke" or not config["cpu_reference"]:
        if min(campaign.remaining(True),campaign.stage_remaining("gpu-readiness"))<=config["execution"]["termination_grace_seconds"]:
            return finish("BUDGET_EXHAUSTED_BEFORE_GPU_PROBE")
        probe=campaign.execute("gpu-readiness",[str(Path.home()/".codex/bin/codex-gpu-probe"),
            "--framework","tensorflow","--gpu","auto","--python",sys.executable],
            cap_seconds=min(diagnostic_cap,campaign.remaining(True)),diagnostic=True)
        log=Path(probe["directory"])/"console.log"
        try:
            receipt=json.loads(log.read_text())
        except (ValueError,OSError):
            return finish("GPU_PROBE_FAILED",attempt=probe)
        if probe["status"]!="completed" or receipt.get("status")!="PASSED":
            waiting = "no_idle_policy_permitted_gpu" in json.dumps(receipt)
            return finish("WAITING_FOR_GPU" if waiting else "GPU_PROBE_FAILED",probe=receipt,attempt=probe)
        gpu=receipt["selected_host_gpu"]
    qualification=None
    def stage(name, request, *, diagnostic=False, reserve=None):
        base={**request,"fixture":fixture}
        if qualification is not None:
            base["qualification_path"]=qualification
        if request["stage"] == "train":
            base.pop("cooperative_seconds",None)
        if name in campaign.state["stages"]:
            return campaign.numerical_stage(name,base,cap_seconds=0.,diagnostic=diagnostic,gpu=gpu)
        spent=sum(a.get("elapsed_seconds",0.) for a in campaign.state["attempts"] if a["stage"]==name)
        cap=min(campaign.stage_remaining(name),campaign.remaining(diagnostic),
                diagnostic_cap if diagnostic else max(0.,reserve-spent) if reserve is not None else config["budget"]["arm_cap_seconds"])
        if cap<=config["execution"]["termination_grace_seconds"]:
            raise StageIncomplete("budget_exhausted:"+name)
        attempts=[a for a in campaign.state["attempts"] if a["stage"]==name]
        if name not in campaign.state["stages"] and len(attempts)>=config["execution"]["stage_attempts"]:
            raise StageIncomplete("attempt_limit:"+name)
        diagnostic_attempts=[a for a in campaign.state["attempts"] if a["diagnostic"] and a["stage"]!="gpu-readiness"]
        if diagnostic and name not in campaign.state["stages"] and len(diagnostic_attempts)>=config["execution"]["diagnostic_attempts"]:
            raise StageIncomplete("diagnostic_attempt_limit:"+name)
        # Training leaves time for its final receipt; the external supervisor
        # still owns the deadline, including worker initialization.
        if request["stage"] == "train":
            base.pop("cooperative_seconds",None)
        result=campaign.numerical_stage(name,base,cap_seconds=cap,diagnostic=diagnostic,gpu=gpu)
        if not result.get("completed"):
            # Retry only interrupted/checkpointed computation with unchanged
            # inputs. Programming and candidate failures need a recorded repair.
            retryable=(result["status"] in {"partial_training_checkpointed","partial_tuning_checkpointed"} or
                       result.get("attempt",{}).get("status") in {"timed_out","interrupted"})
            if retryable and len(attempts)+1<config["execution"]["stage_attempts"] and campaign.stage_remaining(name)>config["execution"]["termination_grace_seconds"]:
                return stage(name,request,diagnostic=diagnostic,reserve=reserve)
            raise StageIncomplete(result["status"])
        return result
    try:
        if config["jit_compile"]:
            receipts=[]
            for beta in config["training"]["betas"][1:]:
                qualified=stage(f"qualify-beta{beta:g}",{"stage":"qualify","betas":[beta]},diagnostic=True)
                receipt=json.loads(Path(qualified["result_path"]).read_text())
                checksum=receipt.pop("checksum")
                if checksum != digest(receipt):
                    raise ValueError("qualification receipt checksum mismatch")
                receipts.append(receipt)
            combined={**receipts[0],"betas":{}}
            for receipt in receipts:
                if any(receipt[k]!=combined[k] for k in receipt if k!="betas"):
                    raise ValueError("qualification receipt scopes differ")
                combined["betas"].update(receipt["betas"])
            qualification=str(campaign.root/"qualification.json")
            atomic_json(qualification,{**combined,"checksum":digest(combined)})
        pricing=stage("price",{"stage":"price"},diagnostic=True)
        pricing["result"]["process_overhead_seconds"]=max(0.,pricing["supervisor_seconds"]-pricing["wall_seconds"])
        quote=forecast_campaign(config,pricing["result"])
        atomic_json(campaign.root/"forecast.json",quote)
        if stop_after=="price":
            return finish("PRICING_COMPLETE",forecast=quote)
        if quote["minimum_complete_seconds"]>campaign.remaining():
            return finish("UNDER_BUDGETED",forecast=quote,unmet="complete training floor and downstream reserve")
        reserve=quote["reserves"]
        reference=stage("reference",{"stage":"reference"},reserve=reserve["reference"])
        # Reference failure vetoes promotion, not the planned training repair.
        # Its complete reference result is preserved while development proceeds.
        training_budget=min(quote["training_cap_seconds"],campaign.remaining()-sum(
            value for key,value in reserve.items() if key!="reference"))
        trained=stage("train",{"stage":"train","cooperative_seconds":training_budget-config["execution"]["termination_grace_seconds"]},reserve=training_budget)
        if stop_after=="train":
            return finish("TRAINING_STAGE_COMPLETE",training=trained["result"])
        maps=choose_maps(config,trained["result"]["checkpoint"])
        direct=choose_maps(config,trained["result"]["checkpoint"],schedule="direct",count=1)
        if len(maps)<config["ensemble"]["charts"] or not direct:
            return finish("TRAINING_REPAIR_REQUIRED",training=trained["result"],eligible_charts=len(maps),
                          next_action="diagnose recorded learning dispositions; no unqualified map may be tuned")
        positive=config["training"]["betas"][1:]
        scopes=[("identity",1.,None,"identity-beta1")]
        scopes += [("neutra",1.,direct[0]["exports"]["1.0"],"plain-neutra-beta1")]
        scopes += [("classical",b,None,f"classical-beta{b:g}") for b in positive]
        scopes += [("neutra",b,item["exports"][str(b)],f"neutra-{i}-beta{b:g}") for i,item in enumerate(maps) for b in positive]
        members={}
        for method,beta,export,label in scopes:
            tuned=stage("tune-"+label,{"stage":"tune","method":method,"beta":beta,"training_export":export,
                "start_label":"development-matched-starts"},reserve=reserve["tuning"]/len(scopes))
            member=selected_member(tuned["result"])
            if member is None:
                return finish("TUNING_REPAIR_REQUIRED",scope=label,tuning=tuned["result"])
            members[label]=member
        frozen={"schema":"bayesfilter.q20.frozen_procedure.v1","config_hash":digest(config),
            "maps":maps,"direct_baseline":direct,"members":members,"method_inventory":config["comparison"]["methods"],
            "selection_policy":"eligible_declared_cohort_order_then_verified_id; no stochastic_ranking"}
        frozen_path=campaign.root/"frozen-procedure.json"
        if frozen_path.exists() and json.loads(frozen_path.read_text())!=frozen:
            raise ValueError("frozen procedure changed on resume")
        atomic_json(frozen_path,frozen)

        def posterior_system(label, current, role):
            posterior_paths={}
            for method in config["comparison"]["methods"]:
                request={"label":label+"-"+method,"role":role,"start_label":label+"-matched-starts"}
                if method in {"identity","classical","neutra"}:
                    key="plain-neutra-beta1" if method=="neutra" else method+"-beta1"
                    request.update(stage="sample",member_path=current[key]["path"])
                else:
                    request.update(stage=method,members_by_beta={str(b):
                        ([current[f"classical-beta{b:g}"]["path"]] if method=="replica_exchange" else
                         [current[f"neutra-{i}-beta{b:g}"]["path"] for i in range(len(maps))]) for b in positive})
                sampled=stage(label+"-"+method,request,reserve=reserve["posterior"]/(1+config["comparison"]["confirmation_replicates"])/len(config["comparison"]["methods"]))
                posterior_paths[method]=sampled["result_path"]
            return stage(label+"-compare",{"stage":"compare","role":role,"reference_path":reference["result_path"],
                "posterior_paths":posterior_paths},reserve=reserve["comparison_analysis"]/(1+config["comparison"]["confirmation_replicates"]))

        development=posterior_system("development",members,"development")
        if not development["result"]["passed"]:
            return finish("DEVELOPMENT_REPAIR_REQUIRED",comparison=development["result"],
                candidate_rejection_only=True,next_action="inspect reference, start-equivalence and posterior failures before a bounded repair")
        confirmations=[]
        for replicate in range(config["comparison"]["confirmation_replicates"]):
            label=f"confirmation-{replicate}"
            fresh={}
            for _,beta,_,scope_label in scopes:
                result=stage(label+"-reverify-"+scope_label,{"stage":"reverify","role":"confirmation",
                    "beta":beta,"member_path":members[scope_label]["path"],"label":label+"-"+scope_label,
                    "start_label":label+"-matched-starts"},reserve=reserve["confirmation_reverification"]/config["comparison"]["confirmation_replicates"]/len(scopes))
                member=selected_member(result["result"])
                if member is None:
                    return finish("CONFIRMATION_REVERIFICATION_FAILED",scope=scope_label,replicate=replicate,result=result["result"])
                fresh[scope_label]=member
            confirmation=posterior_system(label,fresh,"confirmation")
            confirmations.append(confirmation["result"])
            if not confirmation["result"]["passed"]:
                return finish("CONFIRMATION_FAILED",replicate=replicate,comparison=confirmation["result"],
                              next_action="preserve failed confirmation; return to development with new streams")
        return finish("DECLARED_CONFIRMATION_CHECKS_PASSED",confirmation_passed=True,confirmations=confirmations)
    except (StageIncomplete,CampaignBudgetError) as error:
        return finish("MASTER_INCOMPLETE",reason=str(error))
    except Exception as error:
        finish("MASTER_INFRASTRUCTURE_FAILURE",type=type(error).__name__,reason=str(error))
        raise

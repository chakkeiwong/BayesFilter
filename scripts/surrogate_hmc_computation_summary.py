"""Host-only result assembly. No posterior or default admission."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
MODELS = ("lgssm", "ksc", "predator_prey", "sir_d18", "range_bearing")
COMPLETE = "COMPUTATION_EVIDENCE_COMPLETE_POSTERIOR_NOT_RUN"
REPAIR = "COMPUTATION_EVIDENCE_REPAIR_REQUIRED"
STATUS = dict(selection="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC", mechanics="PASS_HMC_MECHANICS_ONLY",
              cost="CONSUMER_PARITY_PASS_COST_MEASURED", paired="CONSUMER_PAIRED_PARITY_PASS_COST_MEASURED",
              precision="SELECTED_CONSUMER_PRECISION_PASS")
def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def validate_model(name, records, root):
    issues=[]
    def check(ok,msg):
        if not ok: issues.append(msg)
    for key,status in STATUS.items():
        record=records.get(key)
        check(isinstance(record,dict) and record.get("status")==status,key+": missing or unsuccessful")
    if issues: return issues
    selection=records["selection"]; signature=selection.get("selection_hash"); chosen=selection.get("selected")
    if not isinstance(chosen,dict): return ["missing selected candidate"]
    unsigned={k:v for k,v in selection.items() if k!="selection_hash"}
    check(signature==hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest(),"selection hash")
    check(chosen.get("valid") is True,"invalid selected candidate")
    scope=selection.get("scope",{})
    expected=dict(model=name,particles=4032,chunk_size=2016,tf32=False,jit_compile=True,
                  dtype="float64" if name=="sir_d18" else "float32",pairwise_dtype="float32",force_batch_size=2,
                  horizon=20 if name=="range_bearing" else 50,
                  precision_policy="fp64_recursion_fp32_pairs" if name=="sir_d18" else "uniform")
    check(all(scope.get(k)==v for k,v in expected.items()),"campaign scope mismatch")
    for key in ("mechanics","cost","paired","precision"): check(records[key].get("scope")==scope,key+": scope mismatch")
    for key in ("cost","paired","precision"): check(records[key].get("selection_hash")==signature,key+": selection mismatch")
    for key in ("paired","precision"):
        for field in ("controls","beta"): check(records[key].get(field)==chosen.get(field),key+": "+field+" mismatch")
    p=records["precision"]
    for key,maximum in (("likelihood",.01),("score",.01),("force",.01),("binding_wiring",1e-6)):
        c=(p.get("comparisons") or {}).get(key,{}); err,tol=c.get("maximum_scaled_error"),c.get("tolerance")
        check(c.get("finite") is True and c.get("passed") is True and number(err) and number(tol)
              and 0<=err<=tol<=maximum,"precision: "+key)
    for key in ("candidate","reference","actual_binding"):
        h=p.get("health",{}).get(key)
        check(isinstance(h,list) and len(h)==4 and all(x is True for x in h),"health: "+key)
    snap=p.get("source_snapshot_checks",{})
    check(snap.get("all_match") is True and snap.get("mismatches")==[],"snapshot check")
    sources=list(scope.get("source_sha256",{}).items())+list(p.get("source_sha256",{}).items())
    check(bool(sources),"missing source identity")
    for source,expected_hash in sources:
        source=Path(source); actual=source if source.is_absolute() else root/source
        try:
            relative=actual.relative_to(root); archived=Path(p["source_snapshot"])/relative
            check(digest(actual)==expected_hash and digest(archived)==expected_hash,"source: "+str(relative))
        except (OSError,ValueError,KeyError): issues.append("source unavailable: "+str(source))
    pair=records["paired"]; err,tol=pair.get("maximum_scaled_force_gap"),pair.get("tolerance")
    check(pair.get("healthy") is True and number(err) and number(tol) and 0<=err<=tol<=.01,"paired force")
    ratio=pair.get("candidate_baseline_warm_ratio"); check(number(ratio) and ratio>0,"paired timing")
    cost=records["cost"]; check(cost.get("healthy") is True,"cost health")
    for key in ("actual_force_B4_in_B2_groups","independent_endpoint_B4"):
        rows=[x for x in cost.get("measurements",[]) if x.get("name")==key]
        times=rows[0].get("warmed_seconds",[]) if len(rows)==1 else []
        check(bool(times) and all(number(x) and x>0 for x in times),"timing: "+key)
    pilot=records["mechanics"].get("pilot",{})
    check(pilot.get("finite") is True and pilot.get("divergences")==0 and pilot.get("force_fallback_count")==0,"mechanics health")
    return sorted(set(issues))

def ledger_issues(ledger):
    if any(a.get("status")=="RUNNING" for a in ledger.get("attempts",[])): return ["GPU worker remains active"]
    charges=[a.get("wall_seconds",a.get("reserved_seconds",0)) for a in ledger.get("attempts",[])]
    if not all(number(x) and x>=0 for x in charges): return ["budget ledger inconsistent"]
    charged=sum(charges)
    used,authorized=ledger.get("used_seconds"),ledger.get("authorized_gpu_wall_seconds")
    if not (number(used) and number(authorized) and 0<=used<=authorized and abs(charged-used)<1e-6
            and number(ledger.get("remaining_seconds"))
            and abs(ledger["remaining_seconds"]-(authorized-used))<1e-6): return ["budget ledger inconsistent"]
    return []

def assemble(campaign, advance, paired, precision, root):
    ledger=read(campaign/"budget.json"); issues=ledger_issues(ledger)
    for path,status in ((advance/"advance.json","COMPUTATION_CHECKS_COMPLETE_POSTERIOR_NOT_RUN"),
                        (paired/"next-phase-plan.json","PAIRED_CONSUMER_CHECKS_COMPLETE"),
                        (precision/"next-phase-plan.json","SELECTED_CONSUMER_PRECISION_COMPLETE")):
        if not path.is_file() or read(path).get("status")!=status: issues.append("nonterminal parent: "+str(path))
    models=[]
    for name in MODELS:
        base=advance/name
        paths=dict(selection=base/"filter-tuning"/name/"filter-selection.json", mechanics=base/"mechanics"/name/"result.json",
                   cost=base/"consumer-cost"/name/"consumer-cost.json",paired=paired/name/name/"consumer-paired.json",
                   precision=precision/name/"selected-consumer-precision.json")
        records={k:read(p) for k,p in paths.items() if p.is_file()}; failures=validate_model(name,records,root)
        row=dict(model=name,issues=failures,status=REPAIR if failures else "CHECKED",
                 evidence={k:dict(path=str(p),sha256=digest(p)) for k,p in paths.items() if p.is_file()})
        if not failures:
            s,p,c,m=(records[k] for k in ("selection","precision","cost","mechanics"))
            row.update(scope=s["scope"],selection_hash=s["selection_hash"],controls=s["selected"]["controls"],beta=s["selected"]["beta"],
                precision_errors={k:v["maximum_scaled_error"] for k,v in p["comparisons"].items()},
                paired_force_gap=records["paired"]["maximum_scaled_force_gap"],
                candidate_baseline_warm_ratio=records["paired"]["candidate_baseline_warm_ratio"],
                measurements=[{k:x[k] for k in ("name","cold_compile_execute_seconds","warmed_seconds","allocator")} for x in c["measurements"]],
                long_chain_cost_model=c["long_chain_cost_model"],
                mechanics={k:m["pilot"].get(k) for k in ("status","steps_per_chain","acceptance_per_chain","max_abs_delta_h","finite","divergences","force_fallback_count")})
            row["four_chain_force_seconds"]=statistics.mean(next(x["warmed_seconds"] for x in c["measurements"] if x["name"]=="actual_force_B4_in_B2_groups"))
        models.append(row)
    passed=not issues and all(not row["issues"] for row in models)
    return dict(schema="surrogate_hmc_computation_summary_v1",status=COMPLETE if passed else REPAIR,
        created_utc=datetime.now(timezone.utc).isoformat(),issues=issues,models=models,command=sys.argv,python=sys.executable,
        git_commit=ledger.get("git_commit"),budget={k:ledger[k] for k in ("authorized_gpu_wall_seconds","used_seconds","remaining_seconds","max_attempts")},
        attempts=len(ledger["attempts"]),campaign=str(campaign),ledger=dict(path=str(campaign/"budget.json"),sha256=digest(campaign/"budget.json")),
        posterior_sampling_complete=False,canonical_admission=False,statistically_supported_runtime_ranking=False,practical_cost_resolved=False,
        remaining_master_work=["Establish practical HMC cost at tuned trajectory lengths","Resolve scientific target/prior and independent posterior references",
                               "Public-authority HMC tuning, adaptive warmup and cumulative retained sampling"],
        next="Computation checks complete; practical cost and full posterior phases remain unfinished" if passed else "Repair listed issues within the existing campaign budget")

def render(result):
    lines=["# Corrected-source surrogate-HMC computation results","","Status: **"+result["status"]+"**.",
           "The goal remains testing surrogate HMC. Computation checks do not complete the posterior experiment.",
           "FP64 is the same finite program in higher precision, not a model oracle. Errors below are",
           "100 times abs(candidate-reference)/max(1,abs(reference)); the threshold is 1%.","",
           "| Model | T / state / parameter dimension | Precision | Value error % | Score error % | Force error % | Four-chain force s | Candidate/baseline time |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in result["models"]:
        if row["issues"]:
            lines.append("| "+row["model"]+" | REPAIR REQUIRED | "+"; ".join(row["issues"])+" | | | | | |")
        else:
            s,e=row["scope"],row["precision_errors"]
            lines.append(f"| {row['model']} | {s['horizon']} / {s['state_dimension']} / {s['parameter_dimension']} | {s['precision_policy']} | {100*e['likelihood']:.6g} | {100*e['score']:.6g} | {100*e['force']:.6g} | {row['four_chain_force_seconds']:.3f} | {row['candidate_baseline_warm_ratio']:.4f} |")
    lines += ["","N4032/K2016, TF32 off, GPU/XLA and analytical all-parameter recursion apply throughout.",
        "Uniform precision means FP32; SIR uses FP64 recursion with FP32 pair kernels.",
        "Runtime ratios are descriptive paired timings with few repeats. No statistically supported ranking is established.",
        "Mechanics runs contain two transitions per chain; acceptance does not establish convergence.","",
        "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
        "|---|---|---|---|---|---|",
        f"| {result['status']} | Saved all-model checks | See per-model issues in result.json | Four local positions | Resolve practical posterior execution | Convergence or canonical LEDH admission |","",
        "| Inference status | Finding |","|---|---|",
        "| Hard veto screen | Source, scope, finite outputs, precision, binding and mechanics checked separately. |",
        "| Statistically supported ranking | None. |",
        "| Descriptive-only differences | Runtime ratios, fixed-position costs, tiny-pilot acceptance and energy error. |",
        "| Default-readiness | FP32 remains default; mixed SIR is an optional scope. No posterior or canonical LEDH promotion. |",
        "| Next evidence needed | Target-specific posterior tuning, cumulative convergence and adequate references. |","",
        "The extension used %.4f of 24 authorized worker-wall hours; %.4f hours remain." % (result["budget"]["used_seconds"]/3600,result["budget"]["remaining_seconds"]/3600),
        "Unused time is retained. The next posterior phase belongs to the existing master, which remains unfinished.",
        "It has not started: the benchmark prior and tiny mechanics do not resolve the scientific target/reference",
        "question, and practical long-chain cost is unestablished. L1 timing scenarios are not warmup bounds.","",
        "Post-run skeptical review: arithmetic agreement can coexist with model approximation error; local",
        "force agreement can coexist with poor mixing. The tiny local test is the strongest alternative",
        "explanation for apparently good mechanics. Wider posterior checks could overturn that impression.",
        "Evidence paths, exact scopes, source hashes, commands, timings and memory measurements are preserved",
        "in result.json and its linked worker manifests."]
    if result["issues"]: lines += ["","Campaign issues: "+"; ".join(result["issues"])]
    return chr(10).join(lines)+chr(10)

def finalize(args,root,save):
    if args.campaign_ledger.name!="budget.json": raise ValueError("campaign ledger must be the explicit budget.json")
    result=assemble(args.campaign_ledger.parent.resolve(),args.advance_dir.resolve(),args.paired_dir.resolve(),args.selected_precision_dir.resolve(),root)
    args.output.mkdir(parents=True,exist_ok=False); save(args.output/"result.json",result)
    (args.output/"result.md").write_text(render(result))
    state=dict(status=result["status"],result=str((args.output/"result.json").resolve()),part_of_existing_master=True,
        posterior_sampling_complete=False,remaining_seconds=result["budget"]["remaining_seconds"],unfinished=result["remaining_master_work"],next=result["next"])
    save(args.output/"next-phase-plan.json",state)
    if "GPU worker remains active" not in result["issues"]: save(args.campaign_ledger.parent/"next-phase-plan.json",state)
    print(json.dumps({k:result[k] for k in ("status","issues","budget","next")}))
    return 0 if result["status"]==COMPLETE else 1

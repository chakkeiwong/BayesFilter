"""Read-only numerical/artifact review of the bounded coupled-trajectory diagnostic.

Computes descriptive summaries from preserved results; no experimental kernels,
parameter selection or replication promotion. Standard library only.
"""
import ast, hashlib, importlib.util, json, math, subprocess, sys
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
RUN=BASE/"coupled-trajectory-20261012/attempt03"
OUT=RUN/"terminal-review"
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,float):return math.isfinite(x)
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return True
def main():
    assert not sys.argv[1:]
    OUT.mkdir(exist_ok=False)
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    m=read(RUN/"manifest.json");r=read(RUN/"decision.json");env=read(RUN/"diagnostic/environment.json")
    check("completed",m["status"]==r["status"]=="COMPLETED")
    check("diagnostic_boundary",r["scope"]=="DIAGNOSTIC_ONLY" and r["replication"]=="INCOMPLETE")
    for rel,digest in m["source_hashes"].items():
        check("snapshot/"+rel,sha(RUN/"sources"/rel)==digest)
        if not rel.endswith(".md"):check("current_code/"+rel,sha(ROOT/rel)==digest)
    a=read(RUN/"diagnostic/author-provenance.json")
    author=ROOT/".localresources/code/younis-mdpf-neurips-2023"
    check("author_commit",a["author_commit"]==env["author"]["source_commit"]=="b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18")
    for rel,digest in a["author_hashes"].items():
        saved=subprocess.check_output(["git","show",a["author_commit"]+":"+rel],cwd=author)
        check("author_pin/"+rel,hashlib.sha256(saved).hexdigest()==digest and sha(author/rel)==digest)
    check("data_hash",sha(Path(env["author"]["data_path"]))==env["author"]["data_sha256"])
    for x in r["provenance"]:check("starting_state/"+x["path"],sha(Path(x["path"]))==x["sha256"])
    check("cpu_reference",env["author"]["cpu_only"] and env["author"]["gpu_intentionally_hidden"])
    check("gpu_xla",len(env["gpu_devices"])>0 and env["jit_compile"] and env["tf32"] is False)
    check("memory_growth",env["memory_growth"]["all_physical_devices_memory_growth"] and env["memory_growth"]["configured_before_logical_device_initialization"])
    check("training_single_trace",r["training_trace_count_per_signature"]==1)
    controls=read(RUN/"diagnostic/adapter-controls.json")
    check("adapter_count",len(controls)==r["adapter_checks"]==70)
    for c in controls:check("control/"+c["name"],c["pass_check"])
    records=[read(p) for p in sorted((RUN/"diagnostic").glob("*-update*.json"))]
    check("32_updates",len(records)==r["updates"]==32)
    summary={}
    for role in ("starting","best"):
        rows=sorted([x for x in records if x["role"]==role],key=lambda x:x["update"])
        check("ordered_updates/"+role,[x["update"] for x in rows]==list(range(16)))
        summaries={}
        for label in ("tf32_source32","tf64_source64","perturb32_source32"):
            cs=[x["comparisons"][label] for x in rows]
            check("finite/"+role+"/"+label,all(finite(c) for c in cs))
            check("complete_steps/"+role+"/"+label,all([x["time"] for x in c["times"]]==list(range(1,17)) and [x["time"] for x in c["ancestors"]]==list(range(2,17)) for c in cs))
            check("sampler_cap/"+role+"/"+label,max(c["max_sampling_iterations"] for c in cs)<128)
            summaries[label]=dict(
                max_abs_loss=max(abs(c["loss_difference"]) for c in cs),
                ancestor_disagreements=sum(x["disagreements"] for c in cs for x in c["ancestors"]),
                first_ancestor_disagreement=next((dict(update_zero_based=x["update"],**t) for x,c in zip(rows,cs) for t in c["ancestors"] if t["disagreements"]),None),
                maximum_angle_difference=max(t["angle"]["max_abs"] for c in cs for t in c["times"]),
                minimum_angle_radius=min(t["minimum_angle_radius"] for c in cs for t in c["times"]),
                max_position_difference=max(t["position"]["max_abs"] for c in cs for t in c["times"]),
                max_weight_difference=max(t["weights"]["max_abs"] for c in cs for t in c["times"]),
                max_sampling_iterations=max(c["max_sampling_iterations"] for c in cs),
                groups={k:dict(max_abs=max(c[k]["max_abs"] for c in cs),max_scaled=max(c[k]["max_scaled"] for c in cs),
                    flagged_updates=[x["update"] for x,c in zip(rows,cs) if c[k]["max_scaled"]>1]) for k in ("parameters","gradients","first_moments","second_moments")})
        check("realized_perturbation/"+role,rows[0]["perturbation"]["realized"]!=0)
        summary[role]=dict(arms=summaries,perturbation=rows[0]["perturbation"])
    points=[]
    perturbation_effect=[]
    for role in ("starting","best"):
        for index in (0,15):
            d=read(RUN/f"diagnostic/{role}-tensors{index:02d}.json")
            check("tensor_finiteness/"+role+"/"+str(index),finite(d))
            # Raw proposal identity checks whether the injected noise reached a measured quantity.
            perturbation_effect.append(dict(role=role,update=index,
                same_raw_proposals=d["source32"]["captured"]["raw"]==d["perturb32"]["captured"]["raw"],
                same_parameters=d["source32"]["parameters"]==d["perturb32"]["parameters"],
                same_moments=d["source32"]["moments"]==d["perturb32"]["moments"]))
            candidates=[]
            for t,(at,bt) in enumerate(zip(d["tf32"]["captured"]["outputs"],d["source32"]["captured"]["outputs"]),1):
                for i,(ar,br) in enumerate(zip(at["particles"],bt["particles"])):
                    for j,(av,bv) in enumerate(zip(ar,br)):
                        delta=math.atan2(math.sin(av[2]-bv[2]),math.cos(av[2]-bv[2]))
                        candidates.append((abs(delta),t,i,j,delta))
            for _,t,i,j,obs in sorted(candidates,reverse=True)[:3]:
                ar=d["tf32"]["captured"]["raw"][t-1][i][j];br=d["source32"]["captured"]["raw"][t-1][i][j]
                u,v=br[-2:];du=ar[-2]-u;dv=ar[-1]-v;rad=math.hypot(u,v)
                points.append(dict(role=role,update=index,time=t,row=i,particle=j,
                    source_radius=rad,raw_delta=[du,dv],predicted_angle=(v*du-u*dv)/(rad*rad),
                    observed_angle=obs))
    ledger=read(BASE/"reference-diagnostics.json")
    manifests=sorted((BASE/"coupled-trajectory-20261012").glob("attempt*/manifest.json"))
    cost={k:sum(read(p)["resource_wall_seconds"][k] for p in manifests) for k in ("cpu","gpu")}
    check("phase_attempt_cap",len(manifests)==3)
    check("phase_compute_cap",max(cost.values())<=3600)
    for p in manifests:
        for resource in ("cpu","gpu"):
            matches=[x for x in ledger if x["id"]=="kdm-coupled-trajectory-"+p.parent.name+"_"+resource]
            check("charge/"+p.parent.name+"/"+resource,len(matches)==1 and matches[0]["charged_seconds"]==read(p)["resource_wall_seconds"][resource])
    sys.path.insert(0,str(ROOT/"docs/benchmarks"))
    from run_replication_gap_localization import campaign
    state=campaign()
    remaining={k:min(state["remaining_seconds"][k],state["tranche_remaining_seconds"][k]) for k in ("cpu","gpu")}
    check("no_campaign_run",not state["running"])
    for name in ("run_kdm_coupled_trajectory.py","reference_kdm_coupled_trajectory.py","diagnose_kdm_coupled_trajectory_tf.py"):
        ast.parse((ROOT/"docs/benchmarks"/name).read_text());check("syntax/"+name,True)
    result=dict(status="PASS" if all(c["passed"] for c in checks) else "FAIL",checks=checks,
        summaries=summary,angle_conditioning_examples=points,perturbation_effect=perturbation_effect,
        campaign_remaining_seconds=remaining,phase_cost_seconds=cost,
        source_reference_equations=True,shared_completed_particles=False,
        inference="Descriptive diagnostic only; no method ranking or quality qualification.",
        source_self_perturbation="Input changed, but no measured proposal/update effect. No useful numerical tolerance follows.",
        next_action="Separate effective numerical perturbations and different pretrained starts before any long learned-quality comparison.")
    (OUT/"review.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"],checks=len(checks),passed=sum(c["passed"] for c in checks),
        failed=[c["name"] for c in checks if not c["passed"]],phase_cost_seconds=cost,remaining_seconds=remaining)))
    return 0 if result["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())

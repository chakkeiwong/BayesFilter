"""CPU-only post-run artifact audit; not a numerical runtime or qualification."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from pathlib import Path
import json, gzip, hashlib, math, subprocess, sys
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE=BASE/"causal-localization-20261012"
def main():
    checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    manifests=sorted(PHASE.glob("attempt*/manifest.json"))+sorted((PHASE/"saved-case").glob("attempt*/manifest.json"))
    charges=dict(cpu=0.,gpu=0.);statuses=[]
    for p in manifests:
        d=json.loads(p.read_text());statuses.append([str(p.relative_to(PHASE)),d["status"]])
        check(str(p.relative_to(PHASE))+"/terminal",d["status"] in ["COMPLETED","FAILED","FAILED_PREFLIGHT"])
        for r in charges:charges[r]+=d["resource_wall_seconds"][r]
        for rel,expected in d.get("source_hashes",{}).items():
            saved=p.parent/"sources"/rel
            check(str(p.relative_to(PHASE))+"/"+rel,saved.is_file() and hashlib.sha256(saved.read_bytes()).hexdigest()==expected)
    p=PHASE/"saved-case/attempt02/diagnostic"
    r=json.loads((p/"result.json").read_text());e=json.loads((p/"environment.json").read_text())
    check("second_checks",r["checks"]==r["passed"]==2358)
    check("capture_exact",r["capture_change"]["exact_numerical_outputs"])
    check("helper_traces",all(x<=1 for x in e["helper_traces"].values()))
    check("growth",e["memory_growth"]["all_physical_devices_memory_growth"])
    check("xla",e["jit_compile"] and e["tf32"] is False)
    check("event",r["event"]==[4,52,1])
    for arm in ["first","all"]:
        c=r["comparisons"][arm]
        check(arm+"/loss",c["loss_difference"]==0)
        check(arm+"/gradients",c["gradients"]["max_scaled"]<=1)
        check(arm+"/parameters",c["parameters"]["max_scaled"]<=1)
        check(arm+"/ancestors",sum(x["disagreements"] for x in c["ancestors"])==0)
    for perturb in r["source_perturbations"]:
        a=perturb["amplitude"];sign=perturb["sign"];v=perturb["perturbation"];c=perturb["comparison"]
        if a==0 or a==1e-8:
            check(str((a,sign))+"/zero_effect",v["realized"]==0 and c["loss_difference"]==0 and c["gradients"]["max_abs"]==0)
        elif sign>0:check(str((a,sign))+"/no_branch_change",c["loss_difference"]==0 and c["gradients"]["max_abs"]==0)
        else:check(str((a,sign))+"/matches_loss_jump",c["loss_difference"]==r["comparisons"]["natural"]["loss_difference"])
    with gzip.open(p/"capture-on.json.gz","rt") as f:tf=json.load(f)["record"]
    with gzip.open(p/"source-perturb-04-tensors.json.gz","rt") as f:source=json.load(f)
    def maxdiff(a,b):
        if isinstance(a,dict):
            check("parameter_names",set(a)==set(b))
            return max(maxdiff(a[n],b[n]) for n in a)
        if isinstance(a,list):
            assert len(a)==len(b)
            return max((maxdiff(x,y) for x,y in zip(a,b)),default=0.)
        assert math.isfinite(a) and math.isfinite(b)
        return abs(a-b)
    same_branch={name:maxdiff(tf[name],source[name]) for name in ["gradients","parameters"]}
    check("moment_parameter_names",set(tf["moments"])==set(source["moments"]))
    source_moments={n:{k:m[k] for k in ("exp_avg","exp_avg_sq")} for n,m in source["moments"].items()}
    same_branch["moments"]=maxdiff(tf["moments"],source_moments)
    same_branch["loss"]=abs(tf["loss"]-source["loss"])
    check("perturbed_source_loss",same_branch["loss"]==0)
    check("perturbed_source_gradient",same_branch["gradients"]<1e-6)
    pin=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT/".localresources/code/younis-mdpf-neurips-2023",text=True).strip()
    check("author_pin",pin=="b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18")
    sys.path.insert(0,str(ROOT/"docs/benchmarks"))
    from run_replication_gap_localization import campaign
    balance=campaign()
    check("no_running_campaign",not balance["running"])
    audit=dict(status="PASS",scope="terminal_local_diagnostic_audit_not_qualification",checks=len(checks),
        passed=sum(x["passed"] for x in checks),manifest_statuses=statuses,resource_seconds_charged=charges,
        remaining_seconds=balance["remaining_seconds"],tranche_remaining_seconds=balance["tranche_remaining_seconds"],
        natural_tf_vs_negative_source_perturbation=same_branch,author_pin=pin,replication="INCOMPLETE",checks_detail=checks,
        cpu_only=True,cuda_visible_devices="-1",audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=PHASE/"terminal-audit.json"
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in audit.items() if k not in ["checks_detail","manifest_statuses"]},indent=2))
if __name__=="__main__":main()

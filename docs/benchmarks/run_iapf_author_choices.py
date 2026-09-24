"""Bounded supervisor for independent CPU R author-choice hypotheses."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/plans/artifacts/iapf-r-author-choice-hypotheses-20260922-01"
PLAN = ROOT / "docs/plans/iapf-r-author-choice-hypotheses-2026-09-22.md"
DEADLINE = datetime.fromisoformat("2026-09-21T20:04:26+00:00").timestamp()
BUDGET = 7200
PREVIOUS = 107389.79640308
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES":"-1", "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"}
LOCK = threading.Lock()
RESERVED = 0
VERSION = "v1"
FIT_ORDER = ["f1_loose", "f2_loose", "f2_strict", "wlog1", "wlog2"]
SD_PAPER = {5:.09,20:.19,80:.35}
N_PAPER = {5:1000,20:1000,80:1142}
METHODS = ("qr", "bpf", "fully_adapted", "sis")

def read_csv(p):
    if not p.exists(): return []
    with p.open() as f: return list(csv.DictReader(f))

def write_json(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)

def attempts():
    p = OUT / "attempts.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def manifest(status):
    records=attempts(); used=sum(x["wall_seconds"] for x in records)
    reports=[Path(x["output"])/"result.md" for x in records
      if x["mode"]=="report" and x["status"]=="finished" and (Path(x["output"])/"result.md").exists()]
    doc=dict(status=status,plan=str(PLAN),result=str(reports[-1] if reports else OUT/"result.md"),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        cpu_only=True,GPU_intentionally_hidden=True,environment={k:ENV[k] for k in
          ("CUDA_VISIBLE_DEVICES","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},
        numerical_backend="independent_reference_base_R",data_version="paper_linear_Gaussian_T100_alpha0.42_generated_fixed_datasets",
        local_budget_seconds=BUDGET,worker_seconds=used,remaining_seconds=BUDGET-used,
        previous_campaign_seconds=PREVIOUS,aggregate_campaign_seconds=PREVIOUS+used,
        original_deadline_utc="2026-09-21T20:04:26Z",max_workers=2,max_launches=100,
        attempts=len(records),reserved_seconds=RESERVED,active_workers=int(RESERVED>0),
        source_version=VERSION,updated_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT/"manifest.json",doc)
    return doc

def snapshot():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f"source-{VERSION}"
    if target.exists():
        hashes=json.loads((target/"sha256.json").read_text())
        assert all(digest(target/p)==h for p,h in hashes.items()),"snapshot changed"
        return
    names=["docs/benchmarks/"+f for f in (
      "reference_iapf_paper.R","reference_iapf_author_choices.R",
      "diagnose_iapf_r_validation_tails.R","diagnose_iapf_author_choice_failures.R","run_iapf_author_choices.R",
      "run_iapf_author_choices.py","summarize_iapf_author_choices.py")]
    names += ["tests/reference_iapf_author_choices.R",str(PLAN.relative_to(ROOT))]
    diagnostic_plan=OUT/"post-run-diagnostic-plan.md"
    if diagnostic_plan.exists():names.append(str(diagnostic_plan.relative_to(ROOT)))
    for name in names:
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write_json(target/"sha256.json",{name:digest(target/name) for name in names})
    (target/"working-tree.diff").write_text(subprocess.check_output(
      ["git","diff","--",*names],cwd=ROOT,text=True))
    legacy=ROOT/"docs/plans/artifacts/iapf-r-equation15-resolution-20260921-01/source-v1/docs/benchmarks/reference_iapf_paper.R"
    fixture=OUT/"runs/floor_probe-d20-j2-wlog1-peak4-sample/results/wlog1-3-failure.rds"
    external={str(legacy):digest(legacy)}
    if fixture.exists():external[str(fixture)]=digest(fixture)
    write_json(target/"external-test-input.json",external)
    (target/"R-version.txt").write_text(subprocess.check_output(["Rscript","--version"],text=True,stderr=subprocess.STDOUT))

def run_job(spec, ceiling=6600):
    global RESERVED
    spec=dict(spec); job_id=spec["job_id"]
    with LOCK:
        old=[x for x in attempts() if x["job_id"]==job_id]
        if old:return old[0]
        current=manifest("executing")
        cap=min(spec.get("cap",180),ceiling-current["worker_seconds"]-RESERVED,
          BUDGET-current["worker_seconds"]-RESERVED,DEADLINE-time.time()-3)
        if cap<5 or current["attempts"]>=100:
            return dict(job_id=job_id,status="budget_stop",wall_seconds=0)
        RESERVED+=cap
        parent=OUT/"runs"/job_id;parent.mkdir(parents=True,exist_ok=False)
        source=OUT/f"source-{VERSION}"
        if spec["mode"]=="tests":
            cmd=["Rscript","--vanilla",str(source/"tests/reference_iapf_author_choices.R"),str(source),str(ROOT)]
        elif spec["mode"]=="report":
            cmd=["python3",str(source/"docs/benchmarks/summarize_iapf_author_choices.py"),str(OUT),str(parent/"results")]
        elif spec["mode"]=="failure_diagnostics":
            cmd=["Rscript","--vanilla",str(source/"docs/benchmarks/diagnose_iapf_author_choice_failures.R"),
              str(OUT),str(parent/"results")]
        else:
            cmd=["Rscript","--vanilla",str(source/"docs/benchmarks/run_iapf_author_choices.R"),
              str(source),str(parent/"results"),spec["mode"],str(spec["dimension"]),
              str(spec["data_seed"]),spec["arm"],spec.get("floor","tail8"),spec.get("sd","sample"),
              str(spec.get("repeats",4)),str(spec.get("first",1))]
        entry={**spec,"command":cmd,"timeout_seconds":cap,"source_version":VERSION,
          "output":str(parent/"results"),"started_utc":datetime.now(timezone.utc).isoformat()}
        write_json(parent/"launch.json",entry)
    started=time.monotonic()
    try:
        with (parent/"worker.log").open("w") as log:
            proc=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=cap)
        entry.update(exit_code=proc.returncode,status="finished" if proc.returncode==0 else "worker_error")
    except subprocess.TimeoutExpired:
        entry.update(exit_code=None,status="timeout")
    finally:
        entry["wall_seconds"]=time.monotonic()-started
        entry["ended_utc"]=datetime.now(timezone.utc).isoformat()
        with LOCK:
            RESERVED-=cap
            with (OUT/"attempts.jsonl").open("a") as f:f.write(json.dumps(entry)+"\n")
            manifest("workers executing" if RESERVED else "stage finished")
    print(json.dumps({k:entry[k] for k in ("job_id","status","wall_seconds")}),flush=True)
    return entry

def batch(jobs,ceiling=4800):
    with ThreadPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(lambda j:run_job(j,ceiling),jobs))
    infrastructure=[e for e in result if e["status"]=="worker_error"]
    if infrastructure:raise RuntimeError("worker error; inspect "+infrastructure[0]["job_id"])
    return result

def jobs(stage,arm,floor="tail8",sd="sample",dims=(5,20,80),family=92100000,repeats=4):
    return [dict(job_id=f"{stage}-d{d}-j{j}-{arm}-{floor}-{sd}",stage=stage,mode="run",
      dimension=d,data_seed=family+100*j+d,arm=arm,floor=floor,sd=sd,
      repeats=repeats,first=1,cap=300 if stage=="validation" else 180)
      for d in dims for j in (1,2)]

def records():
    result=[]
    for e in attempts():
        if e["mode"]!="run":continue
        for row in read_csv(Path(e["output"])/"replicates.csv"):
            result.append({**row,"stage":e["stage"],"job_id":e["job_id"],"output":e["output"]})
    superseded=set()
    for entry in attempts():
        if "supersedes" not in entry or entry["status"]!="finished":continue
        old=entry["supersedes"]
        key=(old["job_id"],old["method"],str(old["replication"]))
        assert sum((r["job_id"],r["method"],r["replication"])==key for r in result)==1
        assert sum(r["job_id"]==entry["job_id"] and r["method"]==old["method"] and
          r["replication"]==str(old["replication"]) for r in result)==1
        superseded.add(key)
    return [r for r in result if (r["job_id"],r["method"],r["replication"]) not in superseded]

def quantile(xs,p):
    ys=sorted(xs); a=(len(ys)-1)*p;i=int(a);fraction=a-i
    return ys[i]*(1-fraction)+ys[min(i+1,len(ys)-1)]*fraction

def screen(rows,d,expected=4):
    good=[r for r in rows if r["status"]=="complete"]
    if len(good)!=expected or len(rows)!=expected:
        return dict(passed=False,reason="incomplete_or_failed",complete=len(good),expected=expected)
    x=[float(r["ratio"]) for r in good]
    if not all(math.isfinite(y) for y in x):return dict(passed=False,reason="nonfinite")
    rng=random.Random(923501+d)
    samples=[[rng.choice(x) for _ in x] for _ in range(2000)]
    means=[statistics.mean(s) for s in samples];sds=[statistics.stdev(s) for s in samples]
    ci=[quantile(means,.025),quantile(means,.975)];upper=quantile(sds,.975)
    n=statistics.mean(float(r["particles"]) for r in good)
    tail=all(r["tail_pass"]=="TRUE" for r in good)
    practical=ci[0]>=.8 and ci[1]<=1.2 and upper<=2*SD_PAPER[d] and n<=1.5*N_PAPER[d]
    return dict(passed=practical and tail,practical_pass=practical,gaussian_limit_tail_pass=tail,
      mean=statistics.mean(x),mean_ci95=ci,sd=statistics.stdev(x),sd_upper95=upper,
      mean_particles=n,complete=len(good),expected=expected)

def combination_screen(arm,floor,sd):
    selected=[r for r in records() if r["stage"]!="validation" and r["method"]==arm
      and r["floor_rule"]==floor and r["sd_mode"]==sd]
    cells=[]
    for d in (5,20,80):
        for j in (1,2):
            seed=92100000+100*j+d
            rs=[r for r in selected if int(r["dimension"])==d and int(r["data_seed"])==seed]
            # Identical controls are reused, not rerun in later calibration stages.
            keys=[r["replication"] for r in rs]
            assert len(keys)==len(set(keys)),"duplicate calibration control"
            cells.append(dict(dimension=d,dataset=j,**screen(rs,d)))
    return dict(arm=arm,floor=floor,sd=sd,passed=all(c["passed"] for c in cells),cells=cells)

def main():
    global VERSION
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("tests","fixtures","fits","floors","controller","validation","report","diagnostics","repair","all"),default="all")
    ap.add_argument("--version",default="v1");args=ap.parse_args();VERSION=args.version
    snapshot()
    if args.stage in ("tests","diagnostics","repair","all"):
        e=run_job(dict(job_id=f"tests-{VERSION}",stage="tests",mode="tests",cap=180),
          7200 if args.stage in ("diagnostics","repair") else 1200)
        if e["status"]!="finished":raise RuntimeError("focused tests failed")
    tests=[e for e in attempts() if e["mode"]=="tests" and e["status"]=="finished" and e["source_version"]==VERSION]
    if not tests:raise RuntimeError("test current source snapshot before numerical work")
    if args.stage=="repair":
        e=run_job(dict(job_id=f"floor-repair-d20-j2-wlog1-peak4-sample-{VERSION}",
          stage="floor_probe",mode="run",dimension=20,data_seed=92100220,
          arm="wlog1",floor="peak4",sd="sample",repeats=1,first=3,cap=180,
          supersedes=dict(job_id="floor_probe-d20-j2-wlog1-peak4-sample",method="wlog1",replication=3)),7200)
        if e["status"]!="finished":raise RuntimeError("localized repair did not complete")
        fit=json.loads((OUT/"fit-selection.json").read_text())["selected"]
        options=[combination_screen(fit,floor,sd) for floor in ("tail8","peak8","peak4","peak2")
          for sd in ("sample","population")]
        selected=[x for x in options if x["passed"]][:2]
        previous=json.loads((OUT/"validation-selection.json").read_text())
        write_json(OUT/f"validation-selection-{VERSION}.json",dict(candidates=selected,
          calibration_screens=options,previous_candidates=previous["candidates"],
          no_validation_reason=None if selected else "no combination passed every calibration dataset"))
        if selected!=previous["candidates"]:raise RuntimeError("selection changed: continue authorized conditional phases")
        e=run_job(dict(job_id=f"report-{VERSION}",stage="report",mode="report",cap=300),7200)
        if e["status"]!="finished":raise RuntimeError("revised report did not complete")
        manifest("campaign complete after localized diagnostic-guard repair")
        return
    if args.stage=="diagnostics":
        if not (OUT/"result.md").exists():raise RuntimeError("finish frozen campaign before diagnosis")
        e=run_job(dict(job_id=f"diagnostic-failures-{VERSION}",stage="diagnostics",
          mode="failure_diagnostics",cap=300),7200)
        if e["status"]!="finished":raise RuntimeError("saved-failure diagnostic did not complete")
        manifest("campaign complete with diagnostic repair")
        return
    if args.stage in ("fixtures","all"):
        batch([dict(job_id=f"fixtures-d{d}-j{j}",stage="fixtures",mode="fixtures",
          dimension=d,data_seed=91100000+100*j+d,arm="all",cap=300)
          for d in (5,20,80) for j in (1,2)],1200)
    if args.stage in ("fits","all"):
        batch([j for arm in ["f1_strict",*FIT_ORDER] for j in jobs("fit_probe",arm,dims=(5,20))])
        batch(jobs("calibration_baselines","baselines"))
        rs=records();eligible=[]
        for arm in FIT_ORDER:
            group=[r for r in rs if r["stage"]=="fit_probe" and r["method"]==arm]
            if len(group)==16 and all(r["status"]=="complete" for r in group):eligible.append(arm)
        chosen=eligible[0] if eligible else "qr"
        write_json(OUT/"fit-selection.json",dict(selected=chosen,eligible=eligible,
          reason="first fully complete prespecified fit; QR fallback if none",ranking_claim=False))
    if args.stage in ("floors","all"):
        fit=json.loads((OUT/"fit-selection.json").read_text())["selected"]
        todo=[j for floor in ("peak8","peak4","peak2") for j in jobs("floor_probe",fit,floor)]
        if fit!="qr":todo+=jobs("floor_control",fit,dims=(80,))
        batch(todo)
    if args.stage in ("controller","all"):
        fit=json.loads((OUT/"fit-selection.json").read_text())["selected"]
        options=[combination_screen(fit,floor,"sample") for floor in ("tail8","peak8","peak4","peak2")]
        eligible=[x for x in options if x["passed"]][:2]
        nominated=eligible or [options[0]]
        write_json(OUT/"floor-selection.json",dict(options=options,controller_candidates=nominated,
          fallback_diagnostic_only=not bool(eligible)))
        batch([j for x in nominated for j in jobs("controller_probe",fit,x["floor"],"population")])
    if args.stage in ("validation","all"):
        fit=json.loads((OUT/"fit-selection.json").read_text())["selected"]
        options=[]
        for floor in ("tail8","peak8","peak4","peak2"):
            for sd in ("sample","population"):
                options.append(combination_screen(fit,floor,sd))
        selected=[x for x in options if x["passed"]][:2]
        write_json(OUT/"validation-selection.json",dict(candidates=selected,calibration_screens=options,
          no_validation_reason=None if selected else "no combination passed every calibration dataset"))
        if selected:
            batch([j for x in selected for j in jobs("validation",fit,x["floor"],x["sd"],
              family=93100000,repeats=16)]+jobs("validation","baselines",family=93100000,repeats=16),6600)
    if args.stage in ("report","all"):
        e=run_job(dict(job_id=f"report-{VERSION}",stage="report",mode="report",cap=300),7200)
        if e["status"]!="finished":raise RuntimeError("report failed")
    manifest("campaign complete" if args.stage in ("report","all") else f"{args.stage} complete")

if __name__=="__main__":main()

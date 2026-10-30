"""Bounded independent CPU reference study; standard-library supervision."""
import argparse, csv, datetime, hashlib, json, os, pathlib, shutil, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
REF=ROOT/".localresources/worktrees/ledh-iapf-execution-20260922"
CAMPAIGN=ROOT/"docs/plans/artifacts/iapf-eq15-choice-calibration-20260925-01"
PLAN=ROOT/"docs/plans/iapf-eq15-choice-calibration-20260925.md"
RSCRIPT=ROOT/"docs/benchmarks/diagnose_iapf_choice_calibration_reference.R"
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,data): p.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--timeout",type=int,default=900)
    parser.add_argument("--audit",action="store_true")
    args=parser.parse_args()
    script=ROOT/"docs/benchmarks/diagnose_iapf_choice_terminal_audit_reference.R" if args.audit else RSCRIPT
    CAMPAIGN.mkdir(parents=True,exist_ok=True)
    previous=[json.loads(p.read_text()) for p in CAMPAIGN.glob("attempt*/manifest.json")]
    if any(p.get("status")=="RUNNING" for p in previous):
        raise SystemExit("Existing unfinished attempt: investigate before rerun")
    spent=sum(p.get("wall_seconds",0) for p in previous)
    timeout=min(args.timeout,1200-spent)
    if timeout<1: raise SystemExit("CPU calibration budget exhausted")
    attempt=CAMPAIGN/f"attempt{len(previous)+1:02d}"
    attempt.mkdir(exist_ok=False); sources=attempt/"sources";sources.mkdir()
    deps=[PLAN,script,RSCRIPT,pathlib.Path(__file__),
          REF/"docs/benchmarks/reference_iapf_paper.R",
          REF/"docs/benchmarks/reference_iapf_eq15_solvers.R"]
    if args.audit:
        deps += [CAMPAIGN/"attempt01/fixtures.rds",CAMPAIGN/"attempt01/sources/diagnose_iapf_choice_calibration_reference.R"]
    hashes={}
    for p in deps:
        shutil.copy2(p,sources/("input-"+p.name if p.name=="fixtures.rds" else p.name));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    def git(path,arg):return subprocess.check_output(["git",*arg],cwd=path,text=True).strip()
    command=["/usr/bin/Rscript","--vanilla",str(script),str(ROOT),str(REF),str(attempt)]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    manifest=dict(status="RUNNING",start_utc=stamp(),command=command,
        git_commit=git(ROOT,["rev-parse","HEAD"]),branch=git(ROOT,["branch","--show-current"]),
        reference_git_commit=git(REF,["rev-parse","HEAD"]),
        reference_branch=git(REF,["branch","--show-current"]),source_sha256=hashes,
        interpreter=subprocess.check_output(["/usr/bin/Rscript","--version"],stderr=subprocess.STDOUT,text=True).strip(),
        environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
        hardware="independent FP64 CPU reference; GPU deliberately hidden",
        GPU_seconds=0,CPU_wall_limit=timeout,previous_CPU_wall_seconds=spent,
        total_CPU_wall_limit=1200,plan=str(PLAN),result=str(attempt/"result.md"),
        data="frozen generated fixtures.rds and ten deduplicated historical failure inputs",
        seed_families=dict(calibration=9251000,heldout=9252000,confirmation=9253000,pilot=9254001),
        output_root=str(attempt),controller_pid=os.getpid())
    write(attempt/"manifest.json",manifest)
    t=time.monotonic();code=None
    with (attempt/"worker.log").open("w") as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        manifest["worker_pid"]=proc.pid;write(attempt/"manifest.json",manifest)
        try:code=proc.wait(timeout=timeout);status="COMPLETED" if code==0 else "FAILED"
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            status="TIMEOUT"
    manifest.update(status=status,exit_code=code,end_utc=stamp(),wall_seconds=time.monotonic()-t)
    write(attempt/"manifest.json",manifest)
    with (CAMPAIGN/"attempts.jsonl").open("a") as stream:stream.write(json.dumps(manifest)+"\n")
    print(json.dumps({k:manifest[k] for k in ("status","exit_code","wall_seconds","output_root")}),flush=True)
    if code:raise SystemExit(code)
    if status=="TIMEOUT":raise SystemExit(124)
if __name__=="__main__":main()

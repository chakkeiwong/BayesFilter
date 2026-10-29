"""Bounded retry after copying the already-built native library into the snapshot."""
import datetime,json,pathlib,subprocess,time,os
ROOT=pathlib.Path(__file__).resolve().parents[4]
OUT=pathlib.Path(__file__).resolve().parent
PYTHON="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
old=json.loads((OUT/"run-manifest-attempt1.json").read_text())
used=sum(r["wall_seconds"] for r in old["workers"])
folder=OUT/"attempt-02";folder.mkdir(exist_ok=False)
manifest=old.copy();manifest["workers"]=list(old["workers"]);manifest["status"]="RUNNING"
manifest["repair"]="Copied unchanged existing native shared library into post-sync source snapshot."
env=dict(os.environ,M13_SCORE_OUTPUT_ROOT=str(folder))
start=time.monotonic()
for arm,dtype in (("after","float32"),("before","float64"),("after","float64")):
    remaining=3600-used-(time.monotonic()-start)
    if remaining<=0:
        manifest["status"]="BUDGET_EXHAUSTED";break
    cmd=[PYTHON,str(OUT/"measure.py"),"--worker",arm,"--dtype",dtype]
    logfile=folder/(arm+"-"+dtype+".log")
    row=dict(command=cmd,log=str(logfile),output_root=str(folder),timeout_seconds=min(1200,remaining))
    manifest["workers"].append(row)
    (OUT/"run-manifest.json").write_text(json.dumps(manifest,indent=2))
    t=time.monotonic()
    with logfile.open("w") as stream:
        try:
            p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=row["timeout_seconds"])
            row["exit_code"]=p.returncode
        except subprocess.TimeoutExpired:
            row["exit_code"]=124
    row["wall_seconds"]=time.monotonic()-t
    print(arm,dtype,"exit",row["exit_code"],"seconds",row["wall_seconds"],flush=True)
    if row["exit_code"]:
        manifest["status"]="REPAIR_NEEDED";break
else:
    manifest["status"]="COMPLETED"
manifest["wall_seconds"]=used+time.monotonic()-start
manifest["remaining_worker_budget_seconds"]=max(0,3600-manifest["wall_seconds"])
(OUT/"run-manifest.json").write_text(json.dumps(manifest,indent=2))
raise SystemExit(0 if manifest["status"]=="COMPLETED" else 1)

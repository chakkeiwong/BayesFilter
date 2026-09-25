from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
source=ROOT.parent/"m13-r1/source-gpu-r1"
records=[]
for name in ("gaussian","beta_binomial","lgssm_location","funnel_noncentered"):
    cmd=[sys.executable,str(REPO/"scripts/resume_inference_validation_job.py"),str(ROOT/"public-model-matrix-continuation-r1"),"m13-public-"+name,"--source",str(source),"--seconds","600","--reason","M14 same-source continuation after M13 worker cap; prepared checkpoint preserved"]
    code=subprocess.run(cmd,cwd=source).returncode
    records.append({"model":name,"returncode":code,"command":cmd})
    (ROOT/"matrix-queue-status.json").write_text(json.dumps(records,indent=2)+"\n")
    if code not in (0,1): raise SystemExit(code)

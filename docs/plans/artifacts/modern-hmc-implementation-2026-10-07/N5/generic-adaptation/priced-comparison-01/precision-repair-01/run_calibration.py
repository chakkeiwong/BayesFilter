from pathlib import Path
import hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
root=Path(__file__).resolve().parent
out=root/"calibration-supervision-01"
out.mkdir(exist_ok=False)
def write(value):
    tmp=out/"result.json.tmp"
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(out/"result.json")
started=time.monotonic()
report=dict(status="running",allocation_seconds=3000.,attempts=[],
    started_utc=datetime.now(timezone.utc).isoformat(),gpu_intentionally_hidden=True,
    driver_sha256=hashlib.sha256((root/"calibration-driver.py").read_bytes()).hexdigest(),
    plan="docs/plans/bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md")
write(report)
parent=None
try:
    for index in range(1,3):
        remaining=3000.-(time.monotonic()-started)
        allowance=min(1800.,remaining-15.)
        if allowance<=0: break
        output=root/f"calibration-{index:02d}"
        command=[sys.executable,str(root/"calibration-driver.py"),"--source-root",str(root/"source"),
                 "--output",str(output),"--replications","128","--seed","73113",
                 "--max-wall-seconds",str(allowance),"--profile","batch_maximum_repair"]
        if parent is not None: command += ["--resume-from",str(parent)]
        attempt=dict(command=command,status="running",output=str(output),allowance_seconds=allowance)
        report["attempts"].append(attempt);write(report)
        before=time.monotonic()
        with (out/f"attempt-{index:02d}.log").open("w") as log:
            try:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=allowance+10.)
                attempt["returncode"]=proc.returncode
            except subprocess.TimeoutExpired:
                attempt.update(status="supervisor_timeout",elapsed_seconds=time.monotonic()-before)
                report["status"]="repair_required";break
        attempt["elapsed_seconds"]=time.monotonic()-before
        if proc.returncode!=0:
            attempt["status"]="failed";report["status"]="repair_required";break
        result=json.loads((output/"result.json").read_text())
        attempt["status"]=result["status"]
        attempt["completed"]=[len(c["records"]) for c in result["cells"]]
        write(report)
        if result["status"]=="completed":
            report.update(status="completed",terminal_result=str(output/"result.json"));break
        if result["status"]!="budget_exhausted":
            report["status"]="repair_required";break
        parent=output
    if report["status"]=="running":report["status"]="budget_exhausted"
except BaseException as exc:
    report.update(status="repair_required",error=str(exc));raise
finally:
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(report)

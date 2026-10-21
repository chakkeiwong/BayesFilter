"""Bounded supervisor for the diagnostic FP64 covariance-product candidate."""
from pathlib import Path
import datetime,json,os,signal,subprocess,sys,time
ROOT=Path("/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917")
CAMPAIGN=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03"
STATE=CAMPAIGN/"covariance-repair-supervisor.json"
PROBE=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-covariance-repair-20261006-01"
MASTER=3631371
TARGET_PREFIX='scope-ksc-t50-n1024'
state=dict(status="WAITING_FOR_WORKER",master_pid=MASTER,target=TARGET_PREFIX,
 plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
 output=str(PROBE),budget_seconds=900,
 accounting="Probe occupies the selected worker slot inside the original campaign budget.")
def save():
 state["updated_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
 t=STATE.with_suffix(".tmp");t.write_text(json.dumps(state,indent=2)+"\n");t.replace(STATE)
def expected():
 try:
  c=Path(f"/proc/{MASTER}/cmdline").read_bytes().replace(b"\0",b" ").decode()
  return "run_surrogate_hmc_marginal_expanded.py" in c and str(CAMPAIGN) in c
 except FileNotFoundError:return False
save()
paused=False
try:
 deadline=time.monotonic()+12000
 while time.monotonic()<deadline:
  if not expected(): raise RuntimeError("master exited before repair slot")
  c=json.loads((CAMPAIGN/"campaign.json").read_text())
  active=[a for a in c["attempts"] if a["status"]=="RUNNING"]
  if active and active[-1]["key"].startswith(TARGET_PREFIX):
   os.kill(MASTER,signal.SIGSTOP);paused=True
   state.update(status="DRAINING_WORKER",pause_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                attempt=active[-1]["attempt"]);save()
   command=active[-1]["command"]; worker_deadline=float(command[command.index("--deadline")+1])
   children=Path(f"/proc/{MASTER}/task/{MASTER}/children").read_text().split()
   if len(children)!=1: raise RuntimeError(f"expected one worker, found {children}")
   child=int(children[0]);state["worker_pid"]=child;save()
   while True:
    try: st=Path(f"/proc/{child}/stat").read_text().split(") ",1)[1].split()[0]
    except FileNotFoundError: break
    if st=="Z":break
    if time.time()>worker_deadline+15:
     os.killpg(child,signal.SIGTERM);raise RuntimeError("worker deadline exceeded")
    time.sleep(2)
   state["status"]="RUNNING_DIAGNOSTIC";save()
   cmd=[sys.executable,str(ROOT/"scripts/probe_marginal_covariance_repair.py"),
        "--output",str(PROBE)]
   st=time.monotonic()
   with (CAMPAIGN/"covariance-repair-probe.log").open("w") as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try: code=proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     code=124
   state.update(status="DIAGNOSTIC_FINISHED",exit_code=code,wall_seconds=time.monotonic()-st,
                command=cmd)
   save();break
  time.sleep(2)
 else: raise RuntimeError("repair slot not reached within bounded wait")
except Exception as e:
 state.update(status="DIAGNOSTIC_UNFINISHED",error=str(e));save()
finally:
 if paused and expected():
  os.kill(MASTER,signal.SIGCONT);state["master_resumed_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat();save()

"""Local bounded supervisor: pause dispatch, drain worker, probe, resume."""
from pathlib import Path
import datetime
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path("/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917")
CAMPAIGN = ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03"
PROBE = ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-covariance-diagnostic-20261006-01"
STATE = CAMPAIGN/"covariance-probe-supervisor.json"
MASTER = 3631371
paused = False
state = dict(status="WAITING_FOR_SHORT_WORKER", master_pid=MASTER,
             next_key="scope-lg10-t20-n1024",
             plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
             accounting="Master wall time during the selected worker includes the exclusive probe; no extra uncharged budget.")


def save():
    state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2)+"\n")
    temp.replace(STATE)


def expected_master():
    try:
        command = Path(f"/proc/{MASTER}/cmdline").read_bytes().replace(b"\0",b" ").decode()
        return "run_surrogate_hmc_marginal_expanded.py" in command and str(CAMPAIGN) in command
    except FileNotFoundError:
        return False


save()
try:
    deadline = time.monotonic()+8000
    while time.monotonic()<deadline:
        if not expected_master():
            raise RuntimeError("Expected master exited before the diagnostic slot")
        c=json.loads((CAMPAIGN/"campaign.json").read_text())
        active=[a for a in c["attempts"] if a["status"]=="RUNNING"]
        if active and active[-1]["key"] == state["next_key"]:
            os.kill(MASTER,signal.SIGSTOP)
            paused=True
            state.update(status="DRAINING_WORKER", pause_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         attempt=active[-1]["attempt"])
            save()
            command=active[-1]["command"]
            worker_deadline=float(command[command.index("--deadline")+1])
            children=Path(f"/proc/{MASTER}/task/{MASTER}/children").read_text().split()
            if len(children)!=1:
                raise RuntimeError(f"Expected one worker, found {children}")
            child=int(children[0])
            state["worker_pid"]=child
            save()
            while True:
                try:
                    stat=Path(f"/proc/{child}/stat").read_text().split(") ",1)[1].split()[0]
                except FileNotFoundError:
                    break
                if stat=="Z":
                    break
                if time.time()>worker_deadline+15:
                    os.killpg(child,signal.SIGTERM)
                    time.sleep(2)
                    raise RuntimeError("Existing worker exceeded its own deadline")
                time.sleep(2)
            state.update(status="RUNNING_DIAGNOSTIC")
            save()
            command=[sys.executable,str(ROOT/"scripts/probe_marginal_covariance_failure.py"),
                     "--output",str(PROBE)]
            start=time.monotonic()
            with (CAMPAIGN/"covariance-probe.log").open("w") as log:
                p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=p.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid,signal.SIGKILL)
                        p.wait()
                    code=124
            state.update(status="DIAGNOSTIC_FINISHED", exit_code=code,
                         diagnostic_wall_seconds=time.monotonic()-start, output=str(PROBE),
                         actual_command=command)
            save()
            timing_output=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-runtime-diagnostic-20261006-01"
            timing_command=[sys.executable,str(ROOT/"scripts/probe_marginal_runtime_measurement.py"),
                            "--output",str(timing_output)]
            state.update(status="RUNNING_TIMING_DIAGNOSTIC")
            save()
            timing_start=time.monotonic()
            with (CAMPAIGN/"runtime-probe.log").open("w") as log:
                timing_process=subprocess.Popen(timing_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    timing_code=timing_process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(timing_process.pid,signal.SIGTERM)
                    try: timing_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(timing_process.pid,signal.SIGKILL)
                        timing_process.wait()
                    timing_code=124
            state.update(status="DIAGNOSTICS_FINISHED",timing_exit_code=timing_code,
                         timing_wall_seconds=time.monotonic()-timing_start,
                         timing_output=str(timing_output),timing_command=timing_command)
            save()
            break
        time.sleep(2)
    else:
        raise RuntimeError("Diagnostic slot was not reached within the bounded wait")
except Exception as error:
    state.update(status="DIAGNOSTIC_UNFINISHED", error=str(error))
finally:
    if paused and expected_master():
        os.kill(MASTER,signal.SIGCONT)
        state["master_resumed_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()

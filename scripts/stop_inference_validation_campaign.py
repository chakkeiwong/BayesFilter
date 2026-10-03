"""Stop one locally owned validation suite and preserve cancellation accounting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--reason",required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    rows=subprocess.check_output(["ps","-eo","pid,ppid,pgid,args"],text=True).splitlines()[1:]
    coordinators=[]
    workers=[]
    for row in rows:
        pid,ppid,pgid,command=row.strip().split(None,3)
        if str(root) not in command:
            continue
        if "-m bayesfilter.testing.inference_validation run " in command:
            coordinators.append(int(pid))
        elif "-m bayesfilter.testing.inference_validation _worker " in command:
            workers.append((int(pid),int(ppid),int(pgid)))
    if len(coordinators)!=1:
        raise RuntimeError("expected exactly one matching live suite coordinator")
    coordinator=coordinators[0]
    os.kill(coordinator,signal.SIGSTOP)
    try:
        index=json.loads((root/"run_index.json").read_text())
        with (root/"cancellation-before-index.json").open("x") as handle:
            json.dump(index,handle,indent=2)
        for pid,parent,group in workers:
            if parent==coordinator:
                try: os.killpg(group,signal.SIGTERM)
                except ProcessLookupError: pass
        os.kill(coordinator,signal.SIGTERM)
    finally:
        try: os.kill(coordinator,signal.SIGCONT)
        except ProcessLookupError: pass
    stopped=time.time()
    for key,job in index["jobs"].items():
        if job["status"]!="running":
            continue
        elapsed=max(0.,stopped-job["started_at"])
        job["attempts"].append({"status":"cancelled", "elapsed_seconds":elapsed,
            "attempt":len(job["attempts"])+1,"reason":args.reason,
            "accounting":"elapsed from recorded coordinator launch to cancellation",
            "log":job["log"],"exit_code":None})
        job.update(status="cancelled",reason=args.reason,result=None)
    index["cancellation"]={"reason":args.reason,"stopped_at":stopped,
                            "coordinator_pid":coordinator,"worker_pids":[r[0] for r in workers]}
    temporary=root/"cancelled-index.tmp"
    temporary.write_text(json.dumps(index,indent=2))
    temporary.replace(root/"run_index.json")
    print(json.dumps(index["cancellation"],indent=2))


if __name__=="__main__": main()

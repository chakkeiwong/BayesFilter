"""Independent CPU reference check of source case process isolation; no port claim."""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source-lock",type=Path,required=True)
    p.add_argument("--source-runner",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    started=time.monotonic()
    spec=importlib.util.spec_from_file_location("author_runner",args.source_runner)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from scipy.io import loadmat
    import numpy as np  # Explicit independent reference equality checks only.
    lock=json.loads((args.source_lock/"source-lock.json").read_text())
    for entry in lock["verified_snapshot_files"]:
        relative=Path(entry["path"]).relative_to(".localresources/code/heng-controlled-smc")
        assert module.sha(args.source_lock/"author-source"/relative)==entry["sha256"]
    assert lock["status"]=="PASS"
    for entry in lock["runtime_dependencies"]:
        assert module.sha(args.root/entry["path"])==entry["sha256"],"reference dependency changed"
    stats=args.root/".localresources/code/octave-statistics-1.5.0/inst"
    results=[]
    commands=[]
    cases=[(1,1),(1,2),(2,1)]
    for key,lane,driver,variable,indices,full_cases,seed in module.DRIVERS:
        sources=args.source_lock/"author-source/StateSpaceModels"/lane
        for arm in ("uninterrupted","isolated"):
            dest=args.output/arm/lane
            dest.mkdir(parents=True,exist_ok=True)
            for f in sources.iterdir():
                if f.is_file() and not (dest/f.name).exists():
                    shutil.copyfile(f,dest/f.name)
            text=module.instrument((sources/driver).read_text(),key,lane,variable,indices,"full")
            text=module.add_case_checkpoint(text,key,variable,isolated=arm=="isolated")
            if arm=="uninterrupted":
                text,count=re.subn(r"(?m)^for igrid = 1:ngrid","for igrid = [1 2]",text)
                assert count==1
                text,count=re.subn(r"(?m)^(\s*)for irep = 1:nrep",r"\1for irep = 1:(2-(igrid==2))",text)
                assert count==1
            script=dest/(Path(driver).stem+"_isolation_check.m")
            script.write_text(text)
            command=module.octave_command(stats,dest,script,seed+100)
            commands.append(dict(driver=key,arm=arm,command=command,cwd=str(dest),instrumented_sha256=module.sha(script)))
            (args.output/"commands.json").write_text(json.dumps(commands,indent=2)+"\n")
            # One uninterrupted process executes the three cases; the isolated arm uses three.
            limit=580-(time.monotonic()-started)
            outcome=module.run_isolated_cases(command,dest,key,cases if arm=="isolated" else [(2,1)],
                limit,args.output/(key+"-"+arm+".log"),args.output/(key+"-"+arm+"-processes.jsonl"))
            assert outcome["returncode"]==0,(key,arm,outcome)
        base=args.output/"uninterrupted"/lane
        isolated=args.output/"isolated"/lane
        with (base/(key+"-cases.csv")).open() as s: a=list(csv.reader(s))
        with (isolated/(key+"-cases.csv")).open() as s: b=list(csv.reader(s))
        assert len(a)==len(b)==3
        for left,right in zip(a,b):
            for index in (0,1,3,4,5,6,7,8,9):
                x,y=float(left[index]),float(right[index])
                assert x==y or (math.isnan(x) and math.isnan(y)),(key,index,x,y)
        a=loadmat(base/(key+"-checkpoint.mat"))
        b=loadmat(isolated/(key+"-checkpoint.mat"))
        keys=[k for k in a if not k.startswith("__")]
        assert set(keys)=={k for k in b if not k.startswith("__")}
        for name in keys:
            np.testing.assert_array_equal(a[name],b[name],err_msg=key+":"+name)
        results.append(dict(driver=key,cases=3,exact_case_outputs=True,exact_rng_states=True,
                            checkpoint_arrays=keys,cumulative_peak_child_rss_kib=outcome.get("cumulative_peak_child_rss_kib")))
        (args.output/"progress.json").write_text(json.dumps(results,indent=2)+"\n")
    summary=dict(status="PASS",scope="SOURCE_PROCESS_ISOLATION_PARITY",replication="INCOMPLETE",
        scientific_verdict="Exact source consumer values and random states preserved in this bounded reference check; full source run and port remain incomplete",
        cpu_only=True,gpu_intentionally_hidden=True,author_algorithm_modified=False,
        numerical_cases=24,drivers=results,wall_seconds=time.monotonic()-started,
        memory_limit_bytes=8*1024**3,source_lock_sha256=module.sha(args.source_lock/"source-lock.json"),
        checks=[dict(name="all_four_source_consumers_and_RNG_states_exact",passed=True)])
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"PASS","drivers":len(results),"numerical_cases":24}))


if __name__=="__main__": main()

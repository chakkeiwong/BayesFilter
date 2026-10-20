"""Independent CPU-only diagnostic of saved rejected Lorenz trajectories."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import tensorflow as tf

def rk4(x):
    def f(z):
        return (np.roll(z,-1,axis=-1)-np.roll(z,2,axis=-1))*np.roll(z,1,axis=-1)-z+2.5
    z=x.copy()
    for _ in range(10):
        a=f(z); b=f(z+.005*a); c=f(z+.005*b); d=f(z+.01*c)
        z=z+(.01/6)*(a+2*b+2*c+d)
    return z

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    args=p.parse_args(); start=time.monotonic()
    summary=dict(status="FAIL",scope="LORENZ_REJECTED_TRAJECTORY_DIAGNOSIS",
        replication="INCOMPLETE",device="CPU",gpu_intentionally_hidden=True,
        diagnostic_numpy_reference=True,policies=[],checks=[])
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    try:
        candidates=json.loads((args.candidate/"summary.json").read_text())["candidates"]
        for candidate in candidates:
            policy=candidate["policy"]; folder=args.candidate/policy
            history={}
            for key in ("A","b","c","x","log_incremental_weight","ess","logZ","ancestors"):
                dtype=tf.int32 if key=="ancestors" else tf.float64
                history[key]=tf.io.parse_tensor(tf.io.read_file(str(folder/(key+".tensor"))),dtype).numpy()
            report=dict(policy=policy,nonfinite=[],affected=[])
            for key,value in history.items():
                indices=np.argwhere(~np.isfinite(value))
                if len(indices):
                    report["nonfinite"].append(dict(field=key,count=len(indices),first=indices[0].tolist()))
            for stage in range(1,5):
                for row in range(4):
                    if stage>candidate["iterations"][row]:continue
                    x=history["x"][stage,:,row]
                    lw=history["log_incremental_weight"][stage,:,row]
                    bad=np.flatnonzero(~np.all(np.isfinite(x),axis=(1,2)) |
                        ~np.all(np.isfinite(lw),axis=1))
                    if not len(bad):continue
                    t=int(bad[0])
                    data=dict(stage=stage,row=row,first_bad_weight_or_cloud_time=t,
                        probes=[])
                    for u in range(max(0,t-1),min(101,t+2)):
                        a,b=history["A"][stage,u,row],history["b"][stage,u,row]
                        precision=np.eye(8)*(100 if u==0 else 1000)+2*a
                        eigen=np.linalg.eigvalsh(precision)
                        if u==0:previous=np.zeros((512,8)); mean=previous
                        else:
                            previous=history["x"][stage,u-1,row][history["ancestors"][stage,u-1,row]]
                            mean=rk4(previous)
                        cov=np.linalg.inv(precision)
                        cm=(mean*(100 if u==0 else 1000)-b)@cov
                        data["probes"].append(dict(time=u,min_precision_eigenvalue=float(eigen[0]),
                            max_precision_eigenvalue=float(eigen[-1]),
                            previous_cloud_maxabs=float(np.max(np.abs(previous))),
                            deterministic_mean_maxabs=float(np.max(np.abs(mean))),
                            twisted_mean_maxabs=float(np.max(np.abs(cm))),
                            largest_covariance_eigenvalue=float(1/eigen[0]),
                            cloud_maxabs=float(np.max(np.abs(x[u]))),
                            lookahead_mean_maxabs=float(np.max(np.abs(rk4(x[u])))),
                            logweight_maxabs=float(np.max(np.abs(lw[u]))),
                            linear_coefficient_maxabs=float(np.max(np.abs(b)))))
                    report["affected"].append(data)
            summary["policies"].append(report)
        summary["checks"]=[dict(name="both_rejected_policies_located",
            passed=len(summary["policies"])==2 and all(row["affected"] for row in summary["policies"]))]
        summary["status"]="PASS" if all(c["passed"] for c in summary["checks"]) else "FAIL"
        summary["interpretation"]="Saved-trajectory diagnosis only; no candidate promotion or solver ranking."
    except Exception as error:summary["error"]=repr(error)
    summary["wall_seconds"]=time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=summary["status"],policies=[
        dict(policy=p["policy"],nonfinite=p["nonfinite"],affected=[
            {k:v for k,v in a.items() if k!="probes"} for a in p["affected"]])
        for p in summary["policies"]],error=summary.get("error"))))
    if summary["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":
    with np.errstate(over="ignore",invalid="ignore"):main()

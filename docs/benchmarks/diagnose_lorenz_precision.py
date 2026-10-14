"""CPU-only independent SVD diagnosis of a rejected Lorenz fit.

NumPy/SciPy are independent reference tools only. GPU devices are deliberately
hidden; this module is never imported by algorithmic or selection code.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import reference_lorenz_stable_fit as ref

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    checks,rows=[],[]
    result=dict(status="FAIL",scope="LORENZ_REJECTED_PRECISION_INDEPENDENT_SVD",
        replication="INCOMPLETE",cpu_only=True,gpu_intentionally_hidden=True,
        numpy_role="INDEPENDENT_REFERENCE_ONLY",checks=checks,rows=rows,
        heuristic_dominance_verdict="NOT_ASSESSED_ROOT_CAUSE_ONLY")
    def check(name,good,**detail):
        checks.append(dict(name=name,passed=bool(good),**detail))
        if not good:raise ValueError(name)
    try:
        fixture=json.loads((args.reference/"fixture.json").read_text())["metadata"]
        fixture["forcing"]=2.5
        tensors={}
        result["input_sha256"]={}
        for key in ("A","b","c","x","log_incremental_weight"):
            path=args.candidate/"rejected_batch"/(key+".tensor");payload=path.read_bytes()
            result["input_sha256"][str(path)]=hashlib.sha256(payload).hexdigest()
            tensors[key]=tf.io.parse_tensor(payload,tf.float64).numpy()
        prior=np.asarray(fixture["initial_covariance"]);transition=np.asarray(fixture["transition_covariance"])
        diag=json.loads((args.candidate/"failure-diagnostics.json").read_text())
        for row,count in enumerate(diag["counts"]):
            for stage in range(1,count+1):
                old=[tensors[k][stage-1,:,row] for k in ("A","b","c")]
                x=tensors["x"][stage-1,:,row]
                weights=tensors["log_incremental_weight"][stage-1,:,row]
                increment=None;failure=None;largest=0.;minimum=1.;svd_min=1.
                for t in reversed(range(len(x))):
                    correction=0.
                    if t<len(x)-1:
                        mean=ref.rk4(x[t],fixture["forcing"])
                        old_mean,old_cov,_,_=ref.twist(mean,transition,*(v[t+1] for v in old))
                        zmean=(old_mean-increment["center"])/increment["scale"]
                        zcov=old_cov/(increment["scale"][:,None]*increment["scale"][None,:])
                        correction=ref.twist(zmean,zcov,increment["A"],increment["b"],increment["c"])[2]
                    fit=ref.regress(x[t],-weights[t]-correction)
                    new=[old[k][t]+fit[k] for k in range(3)]
                    increment=fit[5];svd_min=min(svd_min,fit[4])
                    for key,value in zip(("A","b","c"),new):
                        actual=tensors[key][stage,t,row]
                        largest=max(largest,float(np.max(np.abs(actual-value)/(1e-7+1e-9*np.abs(value)))))
                    base=np.linalg.inv(prior if t==0 else transition)
                    eigen=np.linalg.eigvalsh(base+2*new[0])
                    margin=float(eigen[0]/np.max(np.abs(eigen)));minimum=min(minimum,margin)
                    if margin<=64*np.finfo(float).eps:
                        actual_eigen=np.linalg.eigvalsh(base+2*tensors["A"][stage,t,row])
                        failure=dict(time_index=t,min_eigenvalue=float(eigen[0]),
                            max_abs_eigenvalue=float(np.max(np.abs(eigen))),
                            margin=margin,gpu_min_eigenvalue=float(actual_eigen[0]),
                            original_precision_min_eigenvalue=float(np.linalg.eigvalsh(base+2*old[0][t])[0]))
                        break
                expected_bad=(row==0 and stage==3)
                check("validity_row%d_stage%d"%(row,stage),(failure is not None)==expected_bad)
                check("coefficients_row%d_stage%d"%(row,stage),largest<=1,max_scaled_error=largest)
                check("full_rank_row%d_stage%d"%(row,stage),svd_min>np.sqrt(64*np.finfo(float).eps),minimum_singular_ratio=svd_min)
                rows.append(dict(row=row,stage=stage,failure=failure,minimum_precision_margin=minimum,
                    max_scaled_coefficient_error=largest,minimum_singular_ratio=svd_min))
        result.update(status="PASS",verdict="INDEFINITE_UNCONSTRAINED_REFIT_CONFIRMED",
            interpretation="A full-rank independent least-squares solver produces the same invalid precision. The unprotected candidate is rejected; this is not a batching, retracing, or QR-only failure.")
    except Exception as error:result["error"]=repr(error)
    result["wall_seconds"]=time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"],rows=len(rows),checks=len(checks),
        failure=[r for r in rows if r["failure"]],error=result.get("error"),wall_seconds=result["wall_seconds"])))
    if result["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()

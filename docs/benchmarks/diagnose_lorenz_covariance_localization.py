"""CPU-only localization of already rejected covariance-repair discrepancies."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
from pathlib import Path
import time
import tensorflow as tf
from validate_lorenz_covariance_reference import fitted_policy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","reference","candidate"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument('--stable-integral',action='store_true')
    p.add_argument('--relative-floor',type=float,default=2.220446049250313e-16**.5)
    p.add_argument('--candidate-folder',default='nearest_covariance')
    args=p.parse_args();start=time.monotonic();rows=[];checks=[]
    summary=dict(status="FAIL",scope="REJECTED_COVARIANCE_RECURSION_LOCALIZATION",
        replication="INCOMPLETE",device="CPU",gpu_intentionally_hidden=True,
        diagnostic_numpy_reference=True,rows=rows,checks=checks)
    try:
        fixture=json.loads((args.reference/"fixture.json").read_text())["metadata"]
        folder=args.candidate/args.candidate_folder
        history={k:tf.io.parse_tensor(tf.io.read_file(str(folder/(k+".tensor"))),tf.float64).numpy()
            for k in ("A","b","c","x","log_incremental_weight")}
        for stage in (3,4):
            details=[]
            fitted_policy(fixture,[history[k][stage-1,:,0] for k in ("A","b","c")],
                history["x"][stage-1,:,0],history["log_incremental_weight"][stage-1,:,0],
                details,[history[k][stage,:,0] for k in ("A","b","c")],relative_floor=args.relative_floor,stable_integral=args.stable_integral)
            mismatches=[d for d in details if d["max_scaled_error"]>1]
            projections=[d for d in details if d["projected"]]
            checks.append(dict(name="localized_stage%d"%stage,passed=True,has_mismatches=bool(mismatches)))
            path=args.output/("stage%d.json"%stage)
            path.write_text(json.dumps(details,indent=2,allow_nan=False)+"\n")
            rows.append(dict(stage=stage,first_backward_mismatch=mismatches[0] if mismatches else None,
                first_projection=projections[0] if projections else None,
                mismatched_times=len(mismatches),projection_count=len(projections),
                details=path.name))
        summary["status"]="PASS" if all(c["passed"] for c in checks) else "FAIL"
        summary["interpretation"]="Discrepancy localized; optional covariance repair remains rejected. No variance or efficacy inference."
    except Exception as error:summary["error"]=repr(error)
    summary["wall_seconds"]=time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary))
    if summary["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()

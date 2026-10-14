"""CPU-only independent-reference localization of Lorenz coefficient drift."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import reference_lorenz_stable_fit as reference

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    checks,cases=[],[]
    summary=dict(status="FAIL",scope="LORENZ_SAME_INPUT_COEFFICIENT_DIAGNOSIS",
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role="INDEPENDENT_REFERENCE_ONLY",
        replication="INCOMPLETE",checks=checks,cases=cases)
    def flag(name,good,**details):
        checks.append(dict(name=name,passed=bool(good),**details))
        if not good:raise ValueError(name)
    def load(item):
        path=args.reference/item["file"];payload=path.read_bytes()
        flag("hash_"+path.name,hashlib.sha256(payload).hexdigest()==item["sha256"])
        return np.frombuffer(payload,dtype="<i4" if item["dtype"]=="int32" else "<f8").reshape(item["shape"])
    def tensor(name):
        return tf.io.parse_tensor(tf.io.read_file(str(args.candidate/(name+".tensor"))),tf.float64).numpy()
    def compare(policy,expected,cloud):
        errors={k:policy[k]-expected[k] for k in ("A","b","c")}
        a,b,c=(errors[k] for k in ("A","b","c"))
        values=np.einsum("hni,hij,hnj->hn",cloud,a,cloud)+np.einsum("hni,hi->hn",cloud,b)+c[:,None]
        gradients=2*np.einsum("hni,hij->hnj",cloud,a)+b[:,None,:]
        scale=cloud.std(1)
        result=dict(coefficients={k:dict(max_error=float(np.max(np.abs(v))),
            max_scaled_error=float(np.max(np.abs(v)/(1e-7+1e-9*np.abs(expected[k])))),
            worst_index=list(np.unravel_index(np.argmax(np.abs(v)),v.shape)))
            for k,v in errors.items()},
            max_policy_error=float(np.max(np.abs(values))),
            max_gradient_error=float(np.max(np.abs(gradients))),
            max_scaled_gradient_error=float(np.max(np.abs(gradients*scale[:,None,:]))),
            max_scaled_curvature_error=float(np.max(np.abs(2*a*scale[:,:,None]*scale[:,None,:]))))
        # Convert NumPy index scalars only at this diagnostic reporting boundary.
        for v in result["coefficients"].values():v["worst_index"]=[int(x) for x in v["worst_index"]]
        return result
    try:
        index=json.loads((args.reference/"fixture.json").read_text())
        for i,case in enumerate(index["cases"]):
            local=dict(index["metadata"],forcing=case["forcing"])
            gpu={k:tensor("case%d_%s"%(i,k))[:, :,0] for k in ("A","b","c","x","log_incremental_weight")}
            source={k:gpu[k][0] for k in ("A","b","c","x","log_incremental_weight")}
            for k,v in source.items():flag("finite_case%d_%s"%(i,k),np.all(np.isfinite(v)))
            fitted,rss,ratios,margins=reference.fitted_policy(local,source)
            flag("same_input_valid_%d"%i,np.all(np.isfinite(rss)) and np.min(ratios)>0
                 and np.min(margins)>64*np.finfo(float).eps)
            observed={k:gpu[k][1] for k in ("A","b","c")}
            original={k:load(case["stages"][1]["policy"][k]) for k in ("A","b","c")}
            old_inputs={k:load(case["stages"][0]["output"][k]) for k in ("x","log_incremental_weight")}
            item=dict(forcing=case["forcing"],
                input_max_errors={k:float(np.max(np.abs(source[k]-old_inputs[k]))) for k in old_inputs},
                original_reference=compare(observed,original,source["x"]),
                same_input_reference=compare(observed,fitted,source["x"]),
                minimum_cloud_sd=float(source["x"].std(1).min()),
                minimum_svd_ratio=float(ratios.min()),minimum_precision_margin=float(margins.min()))
            # Keep the refrozen independent fit for a later exact-input parity check.
            for k,v in fitted.items():
                (args.output/("case%d_%s.bin"%(i,k))).write_bytes(np.asarray(v,dtype="<f8").tobytes())
            cases.append(item)
        summary.update(status="PASS",interpretation="Localization only; original coefficient-parity failure remains open.")
    except Exception as exc:
        summary.update(failure=type(exc).__name__+": "+str(exc))
    finally:
        summary["wall_seconds"]=time.monotonic()-started
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:summary.get(k) for k in ("status","failure","wall_seconds")}),flush=True)
    if summary["status"]!="PASS":raise SystemExit(1)

if __name__=="__main__":
    main()

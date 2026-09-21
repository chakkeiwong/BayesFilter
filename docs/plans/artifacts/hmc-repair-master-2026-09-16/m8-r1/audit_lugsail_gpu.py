"""Independent post-run NumPy arithmetic check; no inference decisions."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[4]))
from bayesfilter.testing.inference_validation.storage import read_json,read_tensor,write_json


def independent_se(values):
    n,m,_ = values.shape
    b = math.isqrt(n)
    def batch_lrv(size):
        batches = values[:n//size*size].reshape(n//size,size,m,-1).mean(axis=1)
        return size*np.var(batches,axis=0,ddof=1)
    lrv = 2*batch_lrv(b)-batch_lrv(b//3)
    se = np.sqrt(lrv.sum(axis=0)/(m*m*n))
    return se,np.all(lrv > 0,axis=0)


def main():
    index = read_json(ROOT/"stopping-fresh-gpu-r1/run_index.json")
    rows = []
    for name,job in sorted(index["jobs"].items()):
        result = read_json(job["result"])["assessment"]
        member = next(m for m in result["replications"][0]["members"] if "assessment" in m)
        for arm in ("stopped","fixed"):
            record = member["member_record"]
            path = Path(record["draws_path"] if arm == "stopped" else record["fixed_comparator"]["draws_path"])
            values = read_tensor(path).numpy()
            expected,valid = independent_se(values)
            reported = [v["mcse"] for k,v in member["stopping_pair"][arm].items() if k.endswith(":mean")]
            if not np.all(valid) or not np.allclose(reported,expected,rtol=1.e-10,atol=1.e-14):
                raise AssertionError((name,arm,reported,expected))
            rows.append({"fit":name,"arm":arm,"shape":list(values.shape),
                "draws":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "reported_mcse":reported,"independent_mcse":expected.tolist(),
                "maximum_absolute_difference":float(np.max(np.abs(expected-reported)))})
    output = ROOT/"lugsail-arithmetic-gpu.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output,{"rows":rows,"all_agree":True,"arms":len(rows),
        "relative_tolerance":1.e-10,"absolute_tolerance":1.e-14,
        "tolerance_provenance":"FP64 arithmetic tie-out; much tighter than Monte Carlo uncertainty",
        "formula":"LRV=2*b*sample_variance(batch means)-small_b*sample_variance(small batch means); variance=sum(LRV)/(m*m*n)",
        "coverage_established":False,"command":sys.argv})
    print(json.dumps({"arms":len(rows),"all_agree":True,"maximum_absolute_difference":max(r["maximum_absolute_difference"] for r in rows)}))


if __name__ == "__main__":
    main()

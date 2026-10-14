"""Independent CPU/SciPy reference for Lorenz adaptive refinement.

NumPy is confined to this independent-reference harness and its reference helper.
The TensorFlow consumer reads fixed binary fixtures without importing this module.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import reference_lorenz_stable_fit as reference

SEED = 202609360
COUNTS = {1: 1382, 2: 2867, 3: 4505, 4: 6144}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    checks, completed, arrays = [], [], []
    def flag(name, value, **details):
        checks.append(dict(name=name, passed=bool(value), **details))
        if not value:
            raise ValueError(name)
    def save(name, value, integer=False):
        a = np.asarray(value, dtype="<i4" if integer else "<f8")
        path = args.output/(name+".bin")
        payload = a.tobytes(order="C")
        path.write_bytes(payload)
        item = dict(file=path.name, shape=list(a.shape),
                    dtype="int32" if integer else "float64",
                    sha256=hashlib.sha256(payload).hexdigest())
        arrays.append(item)
        return item
    def save_stage(prefix, policy, output, fit=None):
        record = dict(policy={k: save(prefix+"_"+k, v) for k, v in policy.items()},
                      output={k: save(prefix+"_"+k, v, k=="ancestors") for k, v in output.items()})
        if fit is not None:
            rss, ratio, margin = fit
            record["diagnostics"] = dict(minimum_svd_ratio=float(np.min(ratio)),
                minimum_precision_margin=float(np.min(margin)),
                maximum_rss=float(np.max(rss)))
        return record
    summary = dict(status="FAIL", scope="LORENZ_REFINEMENT_INDEPENDENT_REFERENCE",
        replication="INCOMPLETE", cpu_only=True, gpu_intentionally_hidden=True,
        numpy_role="INDEPENDENT_REFERENCE_ONLY", seed=SEED, checks=checks,
        result_file="summary.json")
    try:
        fixture = json.loads(args.fixture.read_text())
        h, n, d = [int(fixture[k]) for k in ("horizon", "particles", "dimension")]
        y = np.asarray(fixture["y"])
        flag("frozen_source_scope", (h,n,d,y.shape)==(101,512,8,(101,6))
             and fixture["observation_variance"]==1e-4)
        q = np.asarray(fixture["transition_covariance"])
        c0 = np.asarray(fixture["initial_covariance"])
        flag("source_covariances", np.array_equal(q, .001*np.eye(d))
             and np.array_equal(c0, .01*np.eye(d)))
        initial = {k:np.asarray(fixture["replicates"][0]["rounds"][0][k])
                   for k in ("A","b","c")}
        obs_precision = 1/fixture["observation_variance"]
        closed_a = np.tile(np.diag([.5*obs_precision]*6+[0.,0.])[None],(h,1,1))
        closed_b = -obs_precision*np.pad(y, ((0,0),(0,2)))
        closed_c = .5*obs_precision*np.sum(y*y,axis=1)
        flag("source_apf_policy_formula", all(np.allclose(initial[k],v,atol=1e-10,rtol=1e-14)
             for k,v in zip(("A","b","c"),(closed_a,closed_b,closed_c))))
        rng = np.random.default_rng(SEED)
        normal = rng.standard_normal((5,h,n,d))
        uniform = rng.random((5,h))
        index = dict(schema="lorenz_refinement_fixture_v1", seed=SEED,
            metadata={k:fixture[k] for k in ("horizon","particles","dimension","initial_covariance",
                      "transition_covariance","observation_variance","y")},
            normal=save("normal",normal), uniform=save("uniform",uniform),
            cases=[], apf=[], arrays=arrays)
        for case_index, forcing in enumerate((2.5,4.8801,8.5)):
            local = dict(fixture, forcing=forcing)
            policy = {k:v.copy() for k,v in initial.items()}
            stages = []
            for iteration in range(5):
                random = dict(normal=normal[iteration],uniform=uniform[iteration])
                output = reference.forward(local,policy,random)
                flag("finite_case%d_stage%d"%(case_index,iteration),
                     all(np.all(np.isfinite(v)) for v in output.values()))
                stages.append(save_stage("case%d_stage%d"%(case_index,iteration),policy,output,
                                         fit if iteration else None))
                if iteration and (np.min(output["ess"])*100/n>90 or iteration==4):
                    break
                policy, rss, ratio, margin = reference.fitted_policy(local,dict(output,**policy))
                flag("fit_case%d_stage%d"%(case_index,iteration+1),
                     np.min(ratio)>0 and np.min(margin)>64*np.finfo(float).eps
                     and all(np.all(np.isfinite(v)) for v in policy.values()))
                fit = rss,ratio,margin
            item=dict(forcing=forcing,iterations=iteration,stages=stages,
                      minimum_final_ess_fraction=float(np.min(output["ess"])/n),
                      apf_particles=COUNTS[iteration])
            index["cases"].append(item)
            completed.append({k:item[k] for k in ("forcing","iterations","minimum_final_ess_fraction","apf_particles")})
            (args.output/"progress.json").write_text(json.dumps(completed,indent=2)+"\n")
            print(json.dumps(completed[-1]),flush=True)
        apf_normal = rng.standard_normal((h,6144,d))
        apf_uniform = rng.random(h)
        index.update(apf_normal=save("apf_normal",apf_normal),
                     apf_uniform=save("apf_uniform",apf_uniform))
        for count in COUNTS.values():
            local=dict(fixture,forcing=4.8801,particles=count)
            output=reference.forward(local,initial,dict(normal=apf_normal[:,:count],uniform=apf_uniform))
            flag("apf_finite_%d"%count,all(np.all(np.isfinite(v)) for v in output.values()))
            index["apf"].append(dict(particles=count,stage=save_stage("apf%d"%count,initial,output)))
        (args.output/"fixture.json").write_text(json.dumps(index,indent=2,allow_nan=False)+"\n")
        summary.update(status="PASS", cases=completed, apf_counts=list(COUNTS.values()),
            binary_array_count=len(arrays), binary_bytes=sum((args.output/a["file"]).stat().st_size for a in arrays),
            interpretation="Fixed-input independent reference only; no stochastic ranking.")
    except Exception as exc:
        summary.update(failure=type(exc).__name__+": "+str(exc),cases=completed)
    finally:
        summary["wall_seconds"]=time.monotonic()-started
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:summary.get(k) for k in ("status","failure","wall_seconds")}),flush=True)
    if summary["status"]!="PASS":
        raise SystemExit(1)

if __name__=="__main__":
    main()

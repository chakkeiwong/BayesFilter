"""Bounded causal interventions for the SQMC 93001 score diagnostic; UNTUNED."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose_sqmc_score_93001 as diag
tf = diag.tf
campaign, SPEC, DTYPE, ROOT = diag.campaign, diag.SPEC, diag.DTYPE, diag.ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    theta=SPEC.default_theta()
    obs=SPEC.simulate(theta,2,93001,jit_compile=False)
    physical=SPEC.parts(theta,tf.constant([1.,0.,0.,0.],DTYPE))
    marginal_mean=physical["phi"]*physical["m"]
    marginal_variance=tf.square(physical["phi"])*physical["p"]+physical["q"]+physical["r"]
    dm=physical["dphi"]*physical["m"]
    dv=2.*physical["phi"]*physical["dphi"]*physical["p"]
    residual=obs[0]-marginal_mean
    exact_terms=residual*dm/marginal_variance+.5*dv*(tf.square(residual)/tf.square(marginal_variance)-1./marginal_variance)
    report={"program":"UNTUNED causal interventions on saved repair diagnostic; no production claim",
            "cpu_gpu_status":"CPU float64 non-XLA; CUDA_VISIBLE_DEVICES=-1 intentionally hides GPUs",
            "theta":theta,"observations":obs,"exact_first_observation_score_by_state":exact_terms,
            "exact_full_score":SPEC.reference_value_and_score(theta,obs)[1],"cases":[]}
    for route in ("iid_dual_cap","previous_inverse_cdf","repaired_permutation"):
        saved=json.loads((diag.SAVED/route/"result.json").read_text())
        controls=saved["selected_controls"]
        configurations=[]
        for steps in (8,24):
            configurations.append((f"flow_steps_{steps}",route,dict(controls,flow_substeps=steps),12,93001))
        for n in (96,384):
            configurations.append((f"particles_{n}",route,controls,n,93001))
        if route != "iid_dual_cap":
            configurations.append(("fixed_data_scramble_93002",route,controls,12,93002))
        for name,active_route,active_controls,n,input_seed in configurations:
            inputs=campaign.random_inputs(active_route,input_seed,n,3,2,DTYPE)
            compute=diag.trace_kernel(active_route,active_controls,n)
            result=compute(theta,*inputs,obs)
            row=dict(intervention=name,route=route,n=n,input_seed=input_seed,controls=active_controls,
                     **diag.summarize(theta,inputs,obs,result))
            report["cases"].append(row)
            diag.dump(out/"results.json",report)
            diag.dump(out/f"{route}-{name}-inputs.json",{"initial_normals":inputs[0],"process_normals":inputs[1],"uniforms":inputs[2]})
            print(json.dumps({k:row[k] for k in ("intervention","route","n","score1")} |
                             {"increments":[r["score_increment"] for r in row["steps"]],"integrated_t1":row["steps"][0]["process_observation_integrated_score"]}),flush=True)
    saved=json.loads((diag.SAVED/"repaired_permutation"/"result.json").read_text())
    controls=saved["selected_controls"]
    inputs=campaign.random_inputs("repaired_permutation",93001,12,3,2,DTYPE)
    result=diag.trace_kernel("iid_dual_cap",controls,12)(theta,*inputs,obs)
    row=dict(intervention="Halton_inputs_identity_pairing",route="diagnostic_identity_pairing",n=12,input_seed=93001,
             controls=controls,**diag.summarize(theta,inputs,obs,result))
    report["cases"].append(row)
    print(json.dumps({"intervention":row["intervention"],"score1":row["score1"],
                      "increments":[r["score_increment"] for r in row["steps"]]}),flush=True)
    report["wall_seconds"]=time.perf_counter()-start
    diag.dump(out/"results.json",report)
    sources=["docs/benchmarks/diagnose_sqmc_score_93001.py","docs/benchmarks/diagnose_sqmc_score_93001_interventions.py"]
    diag.dump(out/"manifest.json",{"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "command":[sys.executable,*sys.argv],"environment":sys.executable,"tensorflow":tf.__version__,
        "cpu_gpu_status":report["cpu_gpu_status"],"jit_compile":False,"seeds":[93001,93002],
        "data_version":"same saved P44 93001 observations at every intervention; preserved in results",
        "wall_seconds":report["wall_seconds"],"output":str(out),
        "plan":"docs/plans/sqmc-score-93001-diagnostic-20260926.md","result":str(out/"results.json"),
        "baseline_manifest":"../attempt-01/manifest.json",
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}})
    print(json.dumps({"complete":True,"wall_seconds":report["wall_seconds"]}),flush=True)

if __name__=="__main__":
    main()

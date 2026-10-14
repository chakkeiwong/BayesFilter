"""Independent CPU-only verification of the optional Lorenz covariance repair."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf
import reference_lorenz_stable_fit as ref

EPS=np.finfo(float).eps

def projected_integral(old_mean,old_cov,increment):
    """Independent evaluation at the combined quadratic minimizer."""
    mu,scale=increment["center"],increment["scale"]
    a,b,c=(increment[k] for k in ("unprotected_A","unprotected_b","unprotected_c"))
    added=increment["projection_added_A"]
    precision=np.linalg.inv(old_cov)
    raw_a=a/(scale[:,None]*scale[None,:])
    raw_b=b/scale-2*mu@raw_a
    covariance=np.linalg.inv(precision+2*(raw_a+added))
    new_mean=(old_mean@precision-raw_b)@covariance
    z=(new_mean-mu)/scale
    polynomial=np.sum((z@a)*z,-1)+z@b+c
    diff=new_mean-old_mean
    return (.5*(np.linalg.slogdet(covariance)[1]-np.linalg.slogdet(old_cov)[1])
            -polynomial-np.sum((new_mean@added)*new_mean,-1)
            -.5*np.sum((diff@precision)*diff,-1))

def fitted_policy(fixture,old,x,weights,diagnostics=None,actual=None,relative_floor=np.sqrt(EPS),stable_integral=False,activation_margin=None):
    new=[np.zeros_like(v) for v in old];increment=None;ratios=[];protections=0
    q=np.asarray(fixture["transition_covariance"]);p0=np.asarray(fixture["initial_covariance"])
    for t in reversed(range(len(x))):
        correction=0.
        if t<len(x)-1:
            mean=ref.rk4(x[t],2.5)
            old_mean,old_cov,_,_=ref.twist(mean,q,*(v[t+1] for v in old))
            zmean=(old_mean-increment["center"])/increment["scale"]
            zcov=old_cov/(increment["scale"][:,None]*increment["scale"][None,:])
            correction=ref.twist(zmean,zcov,increment["A"],increment["b"],increment["c"])[2]
            if stable_integral:
                correction=projected_integral(old_mean,old_cov,increment)
        a,b,c,rss,ratio,increment=ref.regress(x[t],-weights[t]-correction)
        increment.update(unprotected_A=increment["A"].copy(),unprotected_b=increment["b"].copy(),
                         unprotected_c=increment["c"],projection_added_A=np.zeros_like(a))
        ratios.append(ratio)
        raw=np.linalg.inv(p0 if t==0 else q)+2*(old[0][t]+a)
        values,vectors=np.linalg.eigh(raw)
        protected=False
        margin=values[0]/max(abs(values))
        if (margin<relative_floor if activation_margin is None else margin<=activation_margin):
            if np.any(values==0):raise ValueError("Singular inverse")
            inverse=1/values;floor=relative_floor*max(abs(inverse))
            projected=(vectors*(1/np.maximum(inverse,floor)))@vectors.T
            projected=.5*(projected+projected.T)
            added=.5*(projected-raw)
            increment["projection_added_A"]=added
            a=a+added;mu=increment["center"];s=increment["scale"]
            increment["A"]=increment["A"]+added*s[:,None]*s[None,:]
            increment["b"]=increment["b"]+2*s*(mu@added)
            increment["c"]=increment["c"]+mu@added@mu
            protections+=1;protected=True
        for j,value in enumerate((a,b,c)):new[j][t]=old[j][t]+value
        if diagnostics is not None:
            errors=[float(np.max(np.abs(actual[j][t]-new[j][t])/
                (1e-7+1e-9*np.abs(new[j][t])))) for j in range(3)]
            diagnostics.append(dict(time=t,projected=protected,
                raw_precision_margin=float(values[0]/max(abs(values))),
                minimum_singular_ratio=ratio,max_scaled_error=max(errors),
                coefficient_scaled_errors=errors,
                correction_maxabs=float(np.max(np.abs(correction))),
                target_maxabs=float(np.max(np.abs(-weights[t]-correction))),
                reference_A_maxabs=float(np.max(np.abs(new[0][t]))),
                gpu_A_maxabs=float(np.max(np.abs(actual[0][t])))))
    return new,min(ratios),protections

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","reference","candidate","design"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument('--stable-integral',action='store_true')
    p.add_argument('--relative-floor',type=float,default=float(np.sqrt(EPS)))
    p.add_argument('--activation-margin',type=float)
    p.add_argument('--candidate-index',type=int,default=0)
    p.add_argument('--candidate-folder',default='nearest_covariance')
    args=p.parse_args(argv);start=time.monotonic()
    checks=[];rows=[]
    summary=dict(status="FAIL",scope="LORENZ_COVARIANCE_INDEPENDENT_CORRECTNESS",
        replication="INCOMPLETE",device="CPU",gpu_intentionally_hidden=True,
        diagnostic_numpy_reference=True,relative_floor=args.relative_floor,activation_margin=args.activation_margin,checks=checks,rows=rows)
    def compare(name,actual,expected):
        actual,expected=np.asarray(actual),np.asarray(expected)
        scaled=np.abs(actual-expected)/(1e-7+1e-9*np.abs(expected))
        good=bool(np.isfinite(scaled).all() and np.max(scaled)<=1)
        checks.append(dict(name=name,passed=good,max_scaled_error=float(np.max(scaled)),
            max_absolute_error=float(np.max(np.abs(actual-expected)))))
    try:
        fixture=json.loads((args.reference/"fixture.json").read_text())["metadata"]
        fixture["forcing"]=2.5
        source=json.loads((args.candidate/"summary.json").read_text())["candidates"][args.candidate_index]
        if source["policy"]!="nearest_covariance" or not source["eligible"]:raise ValueError("Ineligible repair input")
        def load(folder,key,dtype=tf.float64):
            return tf.io.parse_tensor(tf.io.read_file(str(folder/(key+".tensor"))),dtype).numpy()
        folder=args.candidate/args.candidate_folder
        history={k:load(folder,k) for k in ("A","b","c","x","log_incremental_weight","logZ","ess")}
        ancestors=load(folder,"ancestors",tf.int32)
        normal=load(args.design/"rejected_batch","normal")
        prior=np.asarray(fixture["initial_covariance"]);transition=np.asarray(fixture["transition_covariance"])
        y=np.asarray(fixture["y"]);n=512
        for row,count in enumerate(source["iterations"]):
            for stage in range(1,count+1):
                name="row%d_stage%d"%(row,stage)
                old=[history[k][stage-1,:,row] for k in ("A","b","c")]
                x=history["x"][stage-1,:,row]
                policy,ratio,protections=fitted_policy(fixture,old,x,history["log_incremental_weight"][stage-1,:,row],relative_floor=args.relative_floor,stable_integral=args.stable_integral,activation_margin=args.activation_margin)
                for k,value in zip(("A","b","c"),policy):compare(name+"_fit_"+k,history[k][stage,:,row],value)
                checks.append(dict(name=name+"_full_rank",passed=bool(ratio>np.sqrt(64*EPS)),minimum_singular_ratio=ratio))
                a,b,c=(history[k][stage,:,row] for k in ("A","b","c"))
                current=history["x"][stage,:,row];cloud=[];weights=[]
                _,_,initial,_=ref.twist(np.zeros((1,8)),prior,a[0],b[0],c[0])
                for t in range(len(x)):
                    mean=np.zeros((n,8)) if t==0 else ref.rk4(current[t-1][ancestors[stage,t-1,row]],2.5)
                    cm,cov,_,_=ref.twist(mean,prior if t==0 else transition,a[t],b[t],c[t])
                    cloud.append(cm+normal[stage,t,row]@np.linalg.cholesky(cov).T)
                    lookahead=0. if t==len(x)-1 else ref.twist(ref.rk4(current[t],2.5),
                        transition,a[t+1],b[t+1],c[t+1])[2]
                    potential=-.5/fixture["observation_variance"]*np.sum((y[t]-current[t,:,:6])**2,-1)
                    weights.append(potential+lookahead+np.sum((current[t]@a[t])*current[t],-1)+current[t]@b[t]+c[t])
                weights=np.asarray(weights)
                logz=initial[0]+np.cumsum(logsumexp(weights,axis=-1)-np.log(n))
                compare(name+"_fixed_input_cloud",current,np.asarray(cloud))
                compare(name+"_fixed_input_weights",history["log_incremental_weight"][stage,:,row],weights)
                compare(name+"_fixed_input_logZ",history["logZ"][stage,:,row],logz)
                rows.append(dict(row=row,stage=stage,reference_protections=protections,minimum_singular_ratio=ratio))
        summary["status"]="PASS" if all(c["passed"] for c in checks) else "FAIL"
        summary["interpretation"]="Independent coefficient and fixed-input forward checks only. No variance ranking or full replication."
    except Exception as error:summary["error"]=repr(error)
    summary["wall_seconds"]=time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=summary["status"],rows=len(rows),checks=len(checks),
        failures=[c for c in checks if not c["passed"]],error=summary.get("error"),
        wall_seconds=summary["wall_seconds"])))
    if summary["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()

"""Diagnostic positive-weight box extension for the same-data M13 reference."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
import tensorflow as tf
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load, OLD_SEED, FRESH_SEEDS
from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import serial
from bayesfilter.testing.surrogate_hmc_m13_same_data_posterior import F64, PRIOR_SD
from bayesfilter.testing.surrogate_hmc_m13_same_data_refine import checked_grid, DEPENDENCIES

PLAN = "docs/plans/surrogate-hmc-m13-domain-sensitivity-2026-09-25.md"
BASE = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-same-data-refine-01"


def quadrature_design():
    unit = PRIOR_SD/16
    total, shell = {}, {}
    def weights(indices):
        step = (indices[1]-indices[0])*unit
        return [step/3*(1 if i in (0,len(indices)-1) else 4 if i%2 else 2)
                for i in range(len(indices))]
    for indices, sign, is_shell in ((list(range(-64,65)),1,False),
        (list(range(-80,81,2)),1,True),(list(range(-64,65,2)),-1,True)):
        w = weights(indices)
        for i,x in enumerate(indices):
            for j,y in enumerate(indices):
                key = (x,y)
                total[key] = total.get(key,0.)+sign*w[i]*w[j]
                if is_shell: shell[key] = shell.get(key,0.)+sign*w[i]*w[j]
    keys = sorted(total)
    # The shell construction has no subtraction between unequal interior weights.
    if any(total[k]<=0 or shell.get(k,0.) < -1e-18 for k in keys):
        raise ValueError("invalid positive-weight quadrature")
    return keys, [[x*unit,y*unit] for x,y in keys], [total[k] for k in keys], [shell.get(k,0.) for k in keys]


def integrator(points, weights, shell_weights, replicates):
    points, weights, shell_weights = [tf.constant(v,F64) for v in (points,weights,shell_weights)]
    q,r = points[:,0],points[:,1]
    logw = tf.math.log(weights)-(q*q+r*r)/(2*PRIOR_SD**2)-math.log(2*math.pi*PRIOR_SD**2)
    features = tf.stack([q,r,q*q,r*r,shell_weights/weights],-1)
    count = int(points.shape[0])
    @tf.function(input_signature=[tf.TensorSpec([count,replicates],F64)],jit_compile=True,autograph=False)
    def integrate(logs):
        weighted = logs+logw[:,None]
        zr = tf.reduce_logsumexp(weighted,axis=0)
        z = tf.reduce_logsumexp(zr)-tf.math.log(tf.cast(replicates,F64))
        ratio = tf.exp(zr-z)
        moments = tf.einsum("nr,nf->rf",tf.exp(weighted-zr[None,:]),features)
        pooled = tf.reduce_mean(ratio[:,None]*moments,0)
        influence = ratio[:,None]*(moments-pooled)
        variance = pooled[2:4]-pooled[:2]**2
        sd = tf.sqrt(variance)
        metric = tf.concat([pooled[:2],sd,pooled[4:]],0)
        influence = tf.concat([influence[:,:2],
            (influence[:,2:4]-2*pooled[None,:2]*influence[:,:2])/(2*sd),influence[:,4:]],1)
        mcse = tf.sqrt(tf.reduce_sum((influence-tf.reduce_mean(influence,0))**2,0)/(replicates*(replicates-1)))
        return dict(metrics=metric,mcse=mcse,influence=influence,log_evidence=z,
            finite=tf.reduce_all(tf.math.is_finite(logs)) & tf.reduce_all(tf.math.is_finite(metric)) &
                   tf.reduce_all(tf.math.is_finite(mcse)) & tf.reduce_all(variance>0.))
    return integrate


def compare_domain(current, previous):
    difference = current["metrics"][:4]-tf.constant(previous["metrics"][:4],F64)
    influence = current["influence"][:,:4]-tf.constant(previous["influence"],F64)[:,:4]
    n = int(influence.shape[0])
    mcse = tf.sqrt(tf.reduce_sum((influence-tf.reduce_mean(influence,0))**2,0)/(n*(n-1)))
    scale = tf.concat([current["metrics"][2:4],current["metrics"][2:4]],0)
    envelope = (tf.abs(difference)+3*mcse)/scale
    mass_envelope = current["metrics"][4]+3*current["mcse"][4]
    passed = bool(current["finite"]) and bool(tf.reduce_all(envelope<=.01)) and float(mass_envelope)<=1e-4
    return serial(dict(passed=passed,mean_sd_difference=difference,paired_mcse=mcse,
        envelope_in_posterior_sd=envelope,new_shell_probability=current["metrics"][4],
        new_shell_probability_mcse=current["mcse"][4],shell_probability_envelope=mass_envelope))


def domain_stage(output):
    from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference
    started = time.monotonic()
    out = Path(output).parent
    out.mkdir(parents=True,exist_ok=True)
    _,fixture,_ = _load()
    snapshot = json.loads((BASE/"run-manifest.json").read_text())
    if not json.loads((BASE/"report.json").read_text())["source_unchanged"]:
        raise ValueError("refinement sources changed")
    for name in (*DEPENDENCIES,"bayesfilter/testing/surrogate_hmc_m13_same_data_refine.py"):
        if snapshot["source_hashes"].get(name)!=file_hash(ROOT/name):
            raise ValueError("refinement numerical dependency changed: "+name)
    old_path = BASE/"m13-same-data-refine/report.json"
    old = json.loads(old_path.read_text())
    if not old["qualified"] or old["invalid_nodes"] or old["observation_checksum"]!=fixture.observation_checksum:
        raise ValueError("refinement scope or validity mismatch")
    write_json(out/"data.json",dict(data_seed=1704,observation_checksum=fixture.observation_checksum,
        observations=fixture.observations.numpy().tolist(),model=fixture.model_definition()))
    keys,points,weights,shell = quadrature_design()
    write_json(out/"quadrature.json",dict(keys=keys,points=points,weights=weights,shell_weights=shell))
    specifications = [(4096,OLD_SEED)]+[(16384,s) for s in FRESH_SEEDS]
    saved = {}
    for n,seed in specifications:
        key = f"n{n}-s{seed[0]}"
        saved[key] = checked_grid(old_path.parent/key/"grid.json",
            expected_hash=old["diagnostics"][key]["grid_sha256"],checksum=fixture.observation_checksum,
            particles=n,seed=seed,order=129)
    evaluators = {n:trajectory_importance_reference(fixture,n,8,full_support=True) for n in (4096,16384)}
    all_logs,diagnostics,invalid = {},{},[]
    for n,seed in specifications:
        key=f"n{n}-s{seed[0]}"
        directory=out/key;directory.mkdir(exist_ok=False)
        logs=[]; records=[]; last_i=None; evaluated=0; min_ess=float("inf"); max_weight=0.
        for index,((i,j),theta) in enumerate(zip(keys,points)):
            if last_i is not None and i!=last_i and records:
                write_json(directory/f"row-{last_i+80:03}.json",records)
                print(key,"new",evaluated,"invalid",len(invalid),"wall",round(time.monotonic()-started,2),flush=True)
                records=[]
            last_i=i
            if abs(i)<=64 and abs(j)<=64:
                logs.append(saved[key]["log_likelihood"][i+64][j+64]);continue
            result=evaluators[n](tf.constant(theta,F64),tf.constant(seed,tf.int32))
            values=result["log_likelihood"]; valid=result["valid"] & tf.math.is_finite(values)
            logs.append(serial(values));evaluated+=1
            if not bool(tf.reduce_all(valid)):invalid.append(dict(group=key,index=index,key=[i,j]))
            ess=float(tf.reduce_min(result["minimum_ess"]))
            weight=float(tf.reduce_max(result["proposal_maximum_normalized_weight"]))
            if math.isfinite(ess):min_ess=min(min_ess,ess)
            if math.isfinite(weight):max_weight=max(max_weight,weight)
            records.append(dict(index=index,key=[i,j],theta=theta,log_likelihood=serial(values),
                valid=serial(valid),minimum_ess=serial(result["minimum_ess"]),
                maximum_weight=serial(result["proposal_maximum_normalized_weight"])))
        if records:write_json(directory/f"row-{last_i+80:03}.json",records)
        path=directory/"grid.json"
        write_json(path,dict(log_likelihood=logs,seed=list(seed),particles=n,replicates=8,
            observation_checksum=fixture.observation_checksum,reused_grid_sha256=old["diagnostics"][key]["grid_sha256"]))
        diagnostics[key]=dict(grid_sha256=file_hash(path),new_evaluations=evaluated,
            reused_evaluations=129**2,new_shell_minimum_ess=min_ess,new_shell_maximum_weight=max_weight)
        all_logs[key]=tf.constant(logs,F64)
    report=dict(plan=PLAN,scope="finite_box_domain_sensitivity_only",data_seed=1704,
        observation_checksum=fixture.observation_checksum,inner_radius_prior_sd=4,outer_radius_prior_sd=5,
        baseline_report=str(old_path),baseline_sha256=file_hash(old_path),diagnostics=diagnostics,
        invalid_nodes=invalid,posterior_qualified=False,endpoint_qualified=False,inference_launched=False,
        prior_support="R^2",tail_bound="NOT_ESTABLISHED",dtype="float64",jit_compile=True,
        ranking_supported=False,heuristic_dominance=dict(verdict="SAMPLER_COMPARISON_NOT_RUN",
            comparators=["fixed endpoint","independent importance seeds","expanded integration domain"],
            situations=["checked fine core","new outer shell"]),wall_seconds=time.monotonic()-started)
    if invalid:return report|dict(status="M13_DOMAIN_REFERENCE_INVALID",qualified=False,
        next_action="Existing master: localize invalid new-shell likelihoods within remaining budget.")
    fixed=integrator(points,weights,shell,8)(all_logs[f"n4096-s{OLD_SEED[0]}"])
    reference=integrator(points,weights,shell,24)(tf.concat([all_logs[f"n16384-s{s[0]}"] for s in FRESH_SEEDS],1))
    comparisons=[compare_domain(fixed,old["baseline"]),compare_domain(reference,old["reference"])]
    scale=tf.concat([reference["metrics"][2:4],reference["metrics"][2:4]],0)
    endpoint=(tf.abs(reference["metrics"][:4]-fixed["metrics"][:4])+3*reference["mcse"][:4])/scale
    passed=all(x["passed"] for x in comparisons) and bool(tf.reduce_all(endpoint<=.05))
    report.update(status="M13_DOMAIN_SENSITIVITY_PASSED" if passed else "M13_DOMAIN_SENSITIVITY_REPAIR_REQUIRED",
        qualified=passed,baseline=serial(fixed),reference=serial(reference),domain_comparisons=comparisons,
        endpoint_envelope_in_posterior_sd=serial(endpoint),
        next_action="Existing master: scoped controls and public-tuner four-chain connection/cost; finite-domain sensitivity is not an infinite-tail proof.")
    write_json(out/"summary.json",report)
    return report

"""Same-data finite-box endpoint posterior precision diagnostic; no admission."""
from __future__ import annotations
import math
from pathlib import Path
import time
import tensorflow as tf
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load, FRESH_SEEDS, OLD_SEED
from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import serial
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference

PLAN = "docs/plans/surrogate-hmc-m13-same-data-posterior-2026-09-25.md"
ORDER = 65
PRIOR_SD = math.log(2.)
RADIUS = 4*PRIOR_SD
F64 = tf.float64
NAMES = ("lambda_Q_mean", "lambda_R_mean", "lambda_Q_sd", "lambda_R_sd",
         "Q_sd_multiplier_mean", "R_sd_multiplier_mean", "outer_region_probability")


def posterior_integrator(order, radius, replicates):
    """Ratio-estimator delta-method influences include replicate evidence weights."""
    if order < 5 or order % 2 != 1 or radius <= 0 or replicates < 2:
        raise ValueError("odd order>=5, positive radius and >=2 replicates required")
    axis = tf.linspace(tf.constant(-radius, F64), tf.constant(radius, F64), order)
    weights = tf.constant([1 if i in (0,order-1) else 4 if i%2 else 2
                           for i in range(order)],F64)*(2*radius/(order-1)/3)
    q,r = tf.meshgrid(axis,axis,indexing="ij")
    log_weights = (tf.math.log(weights[:,None]*weights[None,:])
                   -(q*q+r*r)/(2*PRIOR_SD**2)-math.log(2*math.pi*PRIOR_SD**2))
    boundary = tf.cast((tf.abs(q)>=3*PRIOR_SD)|(tf.abs(r)>=3*PRIOR_SD),F64)
    features = tf.stack([q,r,q*q,r*r,tf.exp(q),tf.exp(r),boundary],-1)

    @tf.function(input_signature=[tf.TensorSpec([order,order,replicates],F64)],
                 jit_compile=True,autograph=False)
    def integrate(log_values):
        weighted = log_values+log_weights[...,None]
        log_z = tf.reduce_logsumexp(weighted,axis=[0,1])
        log_evidence = tf.reduce_logsumexp(log_z)-tf.math.log(tf.cast(replicates,F64))
        ratio = tf.exp(log_z-log_evidence)
        replicate_mass = tf.exp(weighted-log_z[None,None,:])
        moments = tf.einsum("ijr,ijf->rf",replicate_mass,features)
        pooled = tf.reduce_mean(ratio[:,None]*moments,axis=0)
        influence = ratio[:,None]*(moments-pooled[None,:])
        variance = pooled[2:4]-pooled[:2]**2
        sd = tf.sqrt(variance)
        sd_influence = (influence[:,2:4]-2*pooled[None,:2]*influence[:,:2])/(2*sd[None,:])
        metric_influence = tf.concat([influence[:,:2],sd_influence,influence[:,4:]],axis=1)
        metrics = tf.concat([pooled[:2],sd,pooled[4:]],axis=0)
        centered = metric_influence-tf.reduce_mean(metric_influence,axis=0)
        denominator = tf.cast(replicates*(replicates-1),F64)
        mcse = tf.sqrt(tf.reduce_sum(centered**2,axis=0)/denominator)
        evidence_mcse = tf.sqrt(tf.reduce_sum((ratio-1.)**2)/denominator)
        return dict(metrics=metrics,mcse=mcse,influence=metric_influence,
                    log_evidence=log_evidence,log_evidence_mcse=evidence_mcse,
                    replicate_log_evidence=log_z,
                    finite=tf.reduce_all(tf.math.is_finite(log_values)) &
                           tf.reduce_all(tf.math.is_finite(metrics)) &
                           tf.reduce_all(tf.math.is_finite(mcse)) & tf.reduce_all(variance>0.))
    return axis,integrate


def summarize(logs):
    logs=tf.convert_to_tensor(logs,F64)
    order,replicates=int(logs.shape[0]),int(logs.shape[2])
    if order<9 or (order-1)%4 or logs.shape[1]!=order:
        raise ValueError("complete nested Simpson grid required")
    _,fine=posterior_integrator(order,RADIUS,replicates)
    _,coarse=posterior_integrator((order+1)//2,RADIUS,replicates)
    f,c=fine(logs),coarse(logs[::2,::2])
    influence=f["influence"]-c["influence"]
    centered=influence-tf.reduce_mean(influence,axis=0)
    error=tf.sqrt(tf.reduce_sum(centered**2,axis=0)/(replicates*(replicates-1)))
    return f,c,error


def precision_screen(baseline,reference,grid_comparisons):
    scale=tf.concat([reference["metrics"][2:4],reference["metrics"][2:4]],0)
    difference=reference["metrics"][:4]-baseline["metrics"][:4]
    envelope=(tf.abs(difference)+3*reference["mcse"][:4])/scale
    grid=[]
    for fine,coarse,error in grid_comparisons:
        local_scale=tf.concat([fine["metrics"][2:4],fine["metrics"][2:4]],0)
        grid.append((tf.abs(fine["metrics"][:4]-coarse["metrics"][:4])+3*error[:4])/local_scale)
    valid=bool(baseline["finite"]) and bool(reference["finite"]) and all(
        bool(f["finite"]) and bool(c["finite"]) for f,c,_ in grid_comparisons)
    passed=valid and bool(tf.reduce_all(envelope<=.05)) and all(
        bool(tf.reduce_all(x<=.01)) for x in grid)
    return serial(dict(passed=passed,finite=valid,reference_minus_fixed=difference,
        discrepancy_envelope_in_posterior_sd=envelope,grid_envelopes_in_posterior_sd=grid,
        endpoint_margin_sd=.05,grid_margin_sd=.01))


def posterior_stage(output):
    start=time.monotonic()
    out=Path(output).parent
    out.mkdir(parents=True,exist_ok=True)
    _,fixture,baseline_path=_load()
    axis,_=posterior_integrator(ORDER,RADIUS,8)
    nodes=axis.numpy().tolist()
    sequence=sorted(range(ORDER),key=lambda i:abs(nodes[i]))
    write_json(out/"data.json",dict(data_seed=1704,observation_checksum=fixture.observation_checksum,
        observations=fixture.observations.numpy().tolist(),model=fixture.model_definition()))
    specifications=[(4096,OLD_SEED)]+[(16384,seed) for seed in FRESH_SEEDS]
    evaluators={n:trajectory_importance_reference(fixture,n,8,full_support=True) for n in (4096,16384)}
    grids={}; all_invalid=[]; diagnostics={}
    for particles,seed in specifications:
        key=f"n{particles}-s{seed[0]}"
        directory=out/key;directory.mkdir(exist_ok=False)
        logs=[[[None]*8 for _ in nodes] for _ in nodes]
        invalid=[]
        regions={name:dict(count=0,minimum_ess=float("inf"),maximum_weight=0.)
                 for name in ("nominal","intermediate","outer")}
        for i in sequence:
            records=[]
            for j in sequence:
                result=evaluators[particles](tf.constant([nodes[i],nodes[j]],F64),tf.constant(seed,tf.int32))
                values=result["log_likelihood"]
                valid=result["valid"] & tf.math.is_finite(values)
                logs[i][j]=serial(values)
                if not bool(tf.reduce_all(valid)):
                    invalid.append([i,j]); all_invalid.append(dict(group=key,index=[i,j]))
                extent=max(abs(nodes[i]),abs(nodes[j]))/PRIOR_SD
                region="nominal" if extent<=1. else "intermediate" if extent<3. else "outer"
                ess=float(tf.reduce_min(result["minimum_ess"]))
                weight=float(tf.reduce_max(result["proposal_maximum_normalized_weight"]))
                regions[region]["count"]+=1
                if math.isfinite(ess):regions[region]["minimum_ess"]=min(regions[region]["minimum_ess"],ess)
                if math.isfinite(weight):regions[region]["maximum_weight"]=max(regions[region]["maximum_weight"],weight)
                records.append(dict(index=[i,j],theta=[nodes[i],nodes[j]],
                    log_likelihood=logs[i][j],valid=serial(valid),region=region,
                    minimum_ess=serial(result["minimum_ess"]),
                    maximum_weight=serial(result["proposal_maximum_normalized_weight"])))
            write_json(directory/f"row-{i:02}.json",records)
            print(key,"row",i,"invalid",len(invalid),"wall",round(time.monotonic()-start,2),flush=True)
        path=directory/"grid.json"
        write_json(path,dict(axis=nodes,seed=list(seed),particles=particles,replicates=8,
            data_seed=1704,observation_checksum=fixture.observation_checksum,
            log_likelihood=logs,invalid_nodes=invalid,complete=True))
        diagnostics[key]=serial(dict(regions=regions,invalid_nodes=invalid,grid_sha256=file_hash(path)))
        if not invalid:grids[key]=tf.constant(logs,F64)
    report=dict(plan=PLAN,qualification_scope="finite_box_endpoint_precision_only",
        endpoint_qualified=False,posterior_reference_qualified=False,inference_launched=False,
        data_seed=1704,observation_checksum=fixture.observation_checksum,order=ORDER,radius=RADIUS,
        prior_support="R^2",tail_bound="NOT_ESTABLISHED",dtype="float64",jit_compile=True,
        independent_reference_exception=True,baseline_report=str(baseline_path),
        baseline_sha256=file_hash(baseline_path),metric_names=list(NAMES),
        diagnostics=diagnostics,invalid_nodes=all_invalid,wall_seconds=time.monotonic()-start,
        heuristic_dominance=dict(verdict="SAMPLER_COMPARISON_NOT_RUN",comparators=[
            "fixed endpoint","independent importance seeds","nested quadrature"],
            situations=["nominal","intermediate","outer"]),ranking_supported=False)
    if all_invalid:
        return report|dict(status="M13_SAME_DATA_POSTERIOR_REFERENCE_INVALID",qualified=False,
            next_action="Existing master: localize invalid reference nodes; do not run posterior admission.")
    baseline=grids[f"n4096-s{OLD_SEED[0]}"]
    fresh=tf.concat([grids[f"n16384-s{s[0]}"] for s in FRESH_SEEDS],axis=2)
    b=summarize(baseline);r=summarize(fresh)
    screen=precision_screen(b[0],r[0],[b,r])
    report.update(status="M13_SAME_DATA_POSTERIOR_PRECISION_PASSED" if screen["passed"] else
        "M13_SAME_DATA_POSTERIOR_PRECISION_REPAIR_REQUIRED",qualified=screen["passed"],screen=screen,
        baseline=serial(b[0]),reference=serial(r[0]),
        baseline_grid_difference=serial(b[0]["metrics"]-b[1]["metrics"]),
        reference_grid_difference=serial(r[0]["metrics"]-r[1]["metrics"]),
        baseline_grid_difference_mcse=serial(b[2]),reference_grid_difference_mcse=serial(r[2]),
        next_action="Existing master: domain sensitivity, scoped controls and public HMC tuning; no admission from this finite-box screen.")
    write_json(out/"summary.json",report)
    return report

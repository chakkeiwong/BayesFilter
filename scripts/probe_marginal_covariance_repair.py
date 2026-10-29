#!/usr/bin/env python3
"""Bounded diagnostic evaluation of FP64 pushed-covariance products.

This process monkey-patches only the imported consumer; it never edits or
changes the claim-bearing runtime.  Results are diagnostic and require fresh
calibration before any adoption.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=1024)
    args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
                      TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2",
                      TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="2")
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("trusted GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import marginal_weight_campaign as base
    from bayesfilter.testing import marginal_expanded as ex
    from bayesfilter.highdim import ledh_marginal_weights_tf as mixture

    original=mixture.marginal_prior_ratio_tangent
    def repaired(points,d_points,anchors,d_anchors,process_covariance,
                 d_process_covariance,proposal_means,d_proposal_means,
                 affine_matrix,d_affine_matrix,log_weights,d_log_weights):
        dtype=points.dtype
        q64=tf.broadcast_to(tf.cast(process_covariance,tf.float64),
                            [tf.shape(affine_matrix)[0],tf.shape(affine_matrix)[1],
                             tf.shape(affine_matrix)[2]])
        dq64=tf.broadcast_to(tf.cast(d_process_covariance,tf.float64),
                             tf.shape(affine_matrix)[0:3])
        b64=tf.cast(affine_matrix,tf.float64)
        db64=tf.cast(d_affine_matrix,tf.float64)
        bq64=tf.linalg.matmul(b64,q64)
        s64=tf.linalg.matmul(bq64,b64,transpose_b=True)
        ds64=(tf.linalg.matmul(tf.linalg.matmul(db64,q64)+tf.linalg.matmul(b64,dq64),
                               b64,transpose_b=True)
              +tf.linalg.matmul(bq64,db64,transpose_b=True))
        s=tf.cast(s64,dtype); ds=tf.cast(ds64,dtype)
        p,dp,pv,pf=mixture.gaussian_mixture_log_density_tangent(
            points,d_points,anchors,d_anchors,
            tf.cast(q64,dtype),tf.cast(dq64,dtype),log_weights,d_log_weights)
        pp,dpp,qv,qf=mixture.gaussian_mixture_log_density_tangent(
            points,d_points,proposal_means,d_proposal_means,s,ds,
            log_weights,d_log_weights)
        return p-pp,dp-dpp,pv&qv,pf&qf

    def scalar(x):
        value=base.serial(x)
        return dict(value=value["value"][0],score=value["score"][0],
                    valid=value["valid"][0],diagnostics=value["diagnostics"])

    def compiled(spec,n,policy,candidate=False,dtype=tf.float32):
        mixture.marginal_prior_ratio_tangent=repaired if candidate else original
        try:
            fn=ex.compile_batch(spec,n,base.controls(0),policy,dtype=dtype)
            fn.get_concrete_function()
            return fn
        finally:
            mixture.marginal_prior_ratio_tangent=original

    rows=[]; start=time.monotonic()
    model_names=("lg1-t10","lg3-t10","ksc-t10","m13-t10","lg10-t10","lg20-t10")
    for name in model_names:
        spec=base.specification(name)
        fns={policy:compiled(spec,args.n,policy)
             for policy in ("ancestor","marginal_mixture")}
        candidate=compiled(spec,args.n,"marginal_mixture",candidate=True)
        for partition in ("calibration","validation"):
            observed=tf.cast(base.dataset(spec,partition,dataset_id=1),tf.float32)
            for design_id in (0,1):
                design=base.designs(spec,args.n,partition,design_id,dataset_id=1)
                for point,theta in enumerate(spec["points"]):
                    inputs=(tf.constant([theta],tf.float32),*design,observed)
                    rec=dict(scope=name,model=spec["kind"],d=spec["d"],point=point,
                             design=design_id,partition=partition,n=args.n,controls=0)
                    for policy,fn in fns.items():
                        rec[policy]=scalar(fn(*inputs))
                    cand=scalar(candidate(*inputs))
                    rec["marginal_mixture_fp64_product"]=cand
                    old=rec["marginal_mixture"]
                    rec["validity_equal"]=(old["valid"]==cand["valid"])
                    rec["absolute_value_delta"]=None
                    rec["scaled_score_delta"]=None
                    if old["valid"] and cand["valid"]:
                        rec["absolute_value_delta"]=abs(old["value"]-cand["value"])
                        rec["scaled_score_delta"]=0.1*(sum((a-b)**2 for a,b in
                            zip(old["score"],cand["score"]))/len(old["score"]))**0.5
                    rows.append(rec)
                    base.save(args.output/"progress.json",dict(rows=rows))
                print(name,partition,design_id,"complete",flush=True)

    fd=[]; equality=[]
    for name in ("lg10-t10",):
        spec=base.specification(name,tf.float64); short=dict(spec,t=2)
        observed=tf.cast(base.dataset(spec,"calibration",dataset_id=1),tf.float64)[:2]
        whole=base.designs(spec,32,"calibration",0,dtype=tf.float64,dataset_id=1)
        design=(whole[0],whole[1][:2],whole[2])
        old_fn=compiled(short,32,"marginal_mixture",dtype=tf.float64)
        new_fn=compiled(short,32,"marginal_mixture",candidate=True,dtype=tf.float64)
        theta=spec["points"][0]
        inputs=(tf.constant([theta],tf.float64),*design,observed)
        old=scalar(old_fn(*inputs)); new=scalar(new_fn(*inputs))
        equality.append(dict(scope=name,t=2,n=32,old=old,candidate=new,
            values_equal=old["value"]==new["value"],
            scores_equal=old["score"]==new["score"],
            validity_equal=old["valid"]==new["valid"]))
        for point,theta in enumerate(spec["points"]):
            out=scalar(new_fn(tf.constant([theta],tf.float64),*design,observed))
            estimates=[]; perturbed_valid=[]
            for j in range(spec["p"]):
                eps=1e-5
                plus=list(theta); minus=list(theta)
                plus[j]+=eps; minus[j]-=eps
                up=scalar(new_fn(tf.constant([plus],tf.float64),*design,observed))
                down=scalar(new_fn(tf.constant([minus],tf.float64),*design,observed))
                perturbed_valid.extend([up["valid"],down["valid"]])
                estimates.append((up["value"]-down["value"])/(2*eps)
                                 if up["valid"] and down["valid"] else None)
            rel=(max(abs(a-b)/(1+abs(b)) for a,b in zip(out["score"],estimates))
                 if out["valid"] and all(perturbed_valid) else None)
            fd.append(dict(theta=theta,analytic_score=out["score"],
                finite_difference=estimates,relative_error=rel,
                valid=out["valid"] and all(perturbed_valid)))
    mixture.marginal_prior_ratio_tangent=original
    no_harm=all(r["validity_equal"] and (r["absolute_value_delta"] is None or
                 (r["absolute_value_delta"]<.005 and r["scaled_score_delta"]<.005))
                for r in rows if r["d"]<10)
    high_valid=all(r["marginal_mixture_fp64_product"]["valid"] for r in rows if r["d"]>=10)
    fd_ok=all(r["valid"] and r["relative_error"]<1e-4 for r in fd) and len(fd)==3
    fp64_equal=all(r["values_equal"] and r["scores_equal"] and r["validity_equal"]
                   for r in equality)
    mixture.marginal_prior_ratio_tangent=original
    result=dict(status="COMPLETE",rows=rows,finite_difference=fd,
        fp64_equality=equality, passed=no_harm and high_valid and fd_ok and fp64_equal,
        checks=dict(low_dim_no_harm=no_harm,high_dim_valid=high_valid,
                    finite_difference=fd_ok,fp64_equal=fp64_equal),
        wall_seconds=time.monotonic()-start, command=sys.argv,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        environment=dict(python=sys.executable,tensorflow=tf.__version__,
            gpu=os.environ["CUDA_VISIBLE_DEVICES"],memory_policy=memory,
            jit_compile=True,tf32=True,cpu_only=False),
        data_version="calibration/validation dataset 1; designs 0 and 1; controls 0",
        seeds="repository calibration seeds; no statistical selection",
        plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        nonclaim="Diagnostic Class C candidate only; no source adoption, accuracy promotion, or HMC claim",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [pathlib.Path(__file__),ROOT/"bayesfilter/highdim/ledh_marginal_weights_tf.py",
                    ROOT/"bayesfilter/testing/marginal_expanded.py"]})
    base.save(args.output/"result.json",result)
    print("COMPLETE",round(result["wall_seconds"],2),flush=True)

if __name__=="__main__": main()

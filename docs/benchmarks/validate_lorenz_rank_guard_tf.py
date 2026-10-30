"""GPU/XLA no-fire and rejection checks for the QR conditioning guard."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,"/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
import lorenz_refinement_tf as controller
D=tf.float64

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","healthy","candidate","design"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();start=time.monotonic();checks=[];rows=[]
    summary=dict(status="FAIL",scope="LORENZ_QR_RANK_GUARD_CORRECTNESS",replication="INCOMPLETE",
        gpu_memory_policy=memory,device="GPU",dtype="float64",tf32=False,jit_compile=True,
        checks=checks,rows=rows,heuristic_dominance_verdict="NOT_ASSESSED_VALIDITY_GUARD_ONLY")
    def flag(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
        if not value:raise ValueError(name)
    def load(folder,name):return tf.io.parse_tensor(tf.io.read_file(str(folder/(name+".tensor"))),D)
    def engine(cls):
        return cls(observations=101,particles=512,replicas=101,dimension=8,observation_dimension=6,
            initial_mean=[0.]*8,initial_covariance=tf.eye(8,dtype=D)*.01,
            transition_covariance=tf.eye(8,dtype=D)*.001,transition_mean=lambda x:x,
            observation_log_density=lambda y,x:tf.reduce_sum(x*0,axis=-1),
            solver="centered_qr",jit_compile=True)
    try:
        live=Path(inspect.getfile(controller.MultivariateGaussianControlledSMC))
        flag("kernel_snapshot",live.read_bytes()==(Path(__file__).parent/live.name).read_bytes())
        frozen=args.candidate/"sources/multivariate_controlled_smc_tf.py"
        spec=importlib.util.spec_from_file_location("frozen_lorenz_core",frozen)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        before=engine(old.MultivariateGaussianControlledSMC)
        after=engine(controller.MultivariateGaussianControlledSMC)
        for case,count in enumerate((4,1,3)):
            x=load(args.healthy,"case%d_x"%case)
            w=load(args.healthy,"case%d_log_incremental_weight"%case)
            for stage in range(count):
                cloud,target=x[stage,:,0],-w[stage,:,0]
                a=before.regress(cloud,target);b=after.regress(cloud,target)
                flag("healthy_valid_%d_%d"%(case,stage),bool(a["valid"].numpy()) and bool(b["valid"].numpy()))
                equal=True
                for key in ("A","b","c"):
                    delta=tf.abs(a[key]-b[key])
                    flag("unchanged_%d_%d_%s"%(case,stage,key),
                        tf.reduce_all(delta<=1e-7+1e-9*tf.abs(a[key])).numpy(),
                        max_absolute_error=float(tf.reduce_max(delta).numpy()))
                    equal &= bool(tf.reduce_all(delta==0).numpy())
                rows.append(dict(case=case,stage=stage,bitwise_equal=equal,
                    minimum_qr_rcond=float(tf.reduce_min(b["qr_reciprocal_condition_lower_bound"]).numpy())))
        bad=args.candidate/"nearest_covariance"
        x=load(bad,"x")[3,:,0];w=load(bad,"log_incremental_weight")[3,:,0]
        rejected=after.regress(x,-w)
        rcond=rejected["qr_reciprocal_condition_lower_bound"]
        flag("finite_rank_diagnostics",tf.reduce_all(tf.math.is_finite(rcond)).numpy())
        flag("saved_ill_conditioned_design_rejected",not bool(rejected["valid"].numpy()),
            minimum_qr_rcond=float(tf.reduce_min(rcond).numpy()),
            first_rejected_time=int(tf.where(rcond<=(64*2.220446049250313e-16)**.5)[0,0].numpy()))
        # The actual refinement endpoint must propagate the shared guard.
        consumer=controller.LorenzRefinement(forcing=2.5,replicas=4,precision_policy="nearest_covariance")
        folder=args.design/"rejected_batch"
        outcome=consumer.run(load(folder,"y"),load(folder,"normal"),load(folder,"uniform"),tf.constant(4))
        flag("consumer_rejects_unresolved_fit",not bool(outcome["valid"].numpy()))
        summary["consumer"]=dict(valid=bool(outcome["valid"].numpy()),
            iterations=outcome["iterations"].numpy().tolist(),
            minimum_qr_rcond=outcome["safety"]["minimum_qr_rcond"].numpy().tolist())
        summary["traces"]={name:f.experimental_get_tracing_count() for name,f in
            (("old_regress",before.regress),("new_regress",after.regress),
             ("consumer",consumer.run),("fit",consumer.engine.fit),("forward",consumer.engine.forward))}
        flag("fixed_single_trace",all(n==1 for n in summary["traces"].values()))
        summary["status"]="PASS"
        summary["interpretation"]="Rank guard validated; covariance repair remains rejected, whole replication incomplete."
    except Exception as error:summary["error"]=repr(error)
    summary["wall_seconds"]=time.monotonic()-start
    summary["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=summary["status"],checks=len(checks),healthy_fits=len(rows),
        consumer=summary.get("consumer"),error=summary.get("error"),wall_seconds=summary["wall_seconds"])))
    if summary["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()

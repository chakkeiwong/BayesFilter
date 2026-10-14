"""GPU/XLA non-harm evaluation of explicit Lorenz precision safeguards."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse
import gc
import hashlib
import inspect
import json
import math
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
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--policies",nargs="+",choices=("nearest_precision","nearest_covariance","backtrack_precision"),
        default=["nearest_precision","backtrack_precision"])
    p.add_argument("--calibration",type=Path)
    args=p.parse_args();start=time.monotonic()
    summary=dict(status="FAIL",scope="LORENZ_PRECISION_NONHARM_DIAGNOSTIC",
        replication="INCOMPLETE",gpu_memory_policy=memory,device="GPU",
        dtype="float64",tf32=False,jit_compile=True,tensorflow=tf.__version__,
        candidates=[],checks=[],heuristic_dominance_verdict="NOT_ASSESSED_SAFETY_ONLY")
    def check(name,good,**details):
        summary["checks"].append(dict(name=name,passed=bool(good),**details))
        if not good:raise ValueError(name)
    def load(entry):
        payload=(args.reference/entry["file"]).read_bytes()
        check("hash_"+entry["file"],hashlib.sha256(payload).hexdigest()==entry["sha256"])
        return tf.reshape(tf.io.decode_raw(payload,D),entry["shape"])
    def tensor(key):
        return tf.io.parse_tensor(tf.io.read_file(str(args.candidate/"rejected_batch"/(key+".tensor"))),D)
    def clean(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    def persist():
        summary["wall_seconds"]=time.monotonic()-start
        (args.output/"summary.json").write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+"\n")
    try:
        path=Path(inspect.getfile(controller.MultivariateGaussianControlledSMC))
        check("shared_kernel_snapshot",path.read_bytes()==(Path(__file__).parent/path.name).read_bytes())
        fixture=json.loads((args.reference/"fixture.json").read_text())
        y=tf.constant(fixture["metadata"]["y"],D)[:,None,:]
        normal=load(fixture["normal"])[:,:,None,:,:]
        uniform=load(fixture["uniform"])[:,:,None]
        baseline=controller.LorenzRefinement(forcing=2.5)
        expected=baseline.run(y,normal,uniform,tf.constant(4))
        check("unprotected_healthy_valid",expected["valid"].numpy())
        bad_y,bad_normal,bad_uniform=(tensor(key) for key in ("y","normal","uniform"))
        eigen_tolerance=(json.loads((args.calibration/"summary.json").read_text()).get("eigen_tolerance") if args.calibration else None)
        eigen_refinement=(json.loads((args.calibration/"summary.json").read_text()).get("eigen_refinement",False) if args.calibration else False)
        stable_integral=(json.loads((args.calibration/"summary.json").read_text()).get("stable_integral",False) if args.calibration else False)
        activation_margin=(json.loads((args.calibration/"summary.json").read_text()).get("activation_margin") if args.calibration else None)
        summary["activation_margin"]=activation_margin
        summary["stable_integral"]=stable_integral
        summary["eigen_tolerance"]=eigen_tolerance
        summary["eigen_refinement"]=eigen_refinement
        floors=(json.loads((args.calibration/"summary.json").read_text())["eligible_floors"] if args.calibration else None)
        planned=([("nearest_covariance",floor) for floor in floors] if floors is not None
                 else [(policy,None) for policy in args.policies])
        check("nonempty_calibration_survivors",bool(planned))
        for policy,floor in planned:
            folder_name=policy if floor is None else "covariance_floor_"+format(floor,".8g")
            item=dict(policy=policy,floor=floor,activation_margin=activation_margin,folder=folder_name,eligible=False,healthy_checks=[])
            summary["candidates"].append(item)
            before=time.monotonic()
            try:
                healthy=controller.LorenzRefinement(forcing=2.5,precision_policy=policy,precision_floor=floor,precision_eigen_tolerance=eigen_tolerance,precision_eigen_refinement=eigen_refinement,precision_stable_integral=stable_integral,precision_activation_margin=activation_margin)
                result=healthy.run(y,normal,uniform,tf.constant(4))
                valid=bool(result["valid"].numpy())
                fires=float(tf.reduce_sum(result["safety"]["protection_count"]).numpy())
                item.update(healthy_valid=valid,healthy_protections=fires)
                item["healthy_checks"].append(dict(name="valid_without_activation",passed=valid and fires==0))
                for key in ("A","b","c","x","ancestors","log_incremental_weight","logZ","ess"):
                    actual,reference=tf.cast(result["history"][key],D),tf.cast(expected["history"][key],D)
                    difference=tf.abs(actual-reference)
                    limit=tf.zeros_like(reference) if key=="ancestors" else 1e-7+1e-9*tf.abs(reference)
                    good=bool(tf.reduce_all(tf.math.is_finite(actual)&(difference<=limit)).numpy())
                    item["healthy_checks"].append(dict(name=key,passed=good and (floor is None or bool(tf.reduce_all(actual==reference).numpy())),
                        bitwise_equal=bool(tf.reduce_all(actual==reference).numpy()),
                        max_error=float(tf.reduce_max(difference).numpy())))
                item["healthy_checks"].append(dict(name="stopping",passed=bool(
                    (result["iterations"]==expected["iterations"]).numpy())))
                bad=controller.LorenzRefinement(forcing=2.5,replicas=4,precision_policy=policy,precision_floor=floor,precision_eigen_tolerance=eigen_tolerance,precision_eigen_refinement=eigen_refinement,precision_stable_integral=stable_integral,precision_activation_margin=activation_margin)
                recovered=bad.run(bad_y,bad_normal,bad_uniform,tf.constant(4))
                valid=bool(recovered["valid"].numpy())
                safety={k:v.numpy().tolist() for k,v in recovered["safety"].items()}
                item.update(rejected_design_valid=valid,iterations=recovered["iterations"].numpy().tolist(),
                    safety=safety,minimum_forward_precision_margin=float(
                        tf.reduce_min(recovered["history"]["precision_margin"]).numpy()))
                finite=all(bool(tf.reduce_all(tf.math.is_finite(recovered["history"][k])).numpy())
                    for k in ("A","b","c","x","log_incremental_weight","logZ","ess"))
                item["rejected_design_finite"]=finite
                item["trace_counts"]={}
                for lane,engine in (("healthy",healthy),("rejected",bad)):
                    for name,func in (("controller",engine.run),("fit",engine.engine.fit),("forward",engine.engine.forward)):
                        item["trace_counts"][lane+"_"+name]=func.experimental_get_tracing_count()
                item["eligible"]=(all(c["passed"] for c in item["healthy_checks"])
                    and valid and finite and all(v==1 for v in item["trace_counts"].values())
                    and float(tf.reduce_sum(recovered["safety"]["protection_count"]).numpy())>0)
                item["verdict"]="VIABLE_FOR_CORRECTNESS_VALIDATION" if item["eligible"] else "REJECTED_NONHARM_SCREEN"
                folder=args.output/folder_name;folder.mkdir()
                for name,value in dict(y=bad_y,**recovered["history"]).items():
                    tf.io.write_file(str(folder/(name+".tensor")),tf.io.serialize_tensor(value))
                item["saved_history"]=str(folder)
                del healthy,bad,result,recovered
            except Exception as error:
                item.update(error=repr(error),verdict="REJECTED_EXECUTION_FAILURE")
            item["seconds"]=time.monotonic()-before
            persist();gc.collect()
        check("all_predeclared_candidates_evaluated",len(summary["candidates"])==len(planned))
        eligible=[row["policy"] for row in summary["candidates"] if row["eligible"]]
        summary.update(status="PASS",eligible_policies=eligible,
            next_validation_policy=eligible[0] if eligible else None,
            interpretation="Completed safety evaluation only. Candidate eligibility does not certify correctness, variance reduction, source-code replication, or default readiness.")
    except Exception as error:summary["error"]=repr(error)
    summary["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    persist()
    print(json.dumps(dict(status=summary["status"],candidates=[
        {k:row.get(k) for k in ("policy","floor","verdict","healthy_valid","healthy_protections","rejected_design_valid","iterations","error")}
        for row in summary["candidates"]],wall_seconds=summary["wall_seconds"],error=summary.get("error"))))
    if summary["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()

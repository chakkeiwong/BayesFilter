"""GPU/XLA diagnostic capture of the actual shared Lorenz fit call chain."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,"/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
import lorenz_refinement_tf as controller
def main():
    p=argparse.ArgumentParser()
    for key in ("output","reference","candidate"):p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    summary=dict(status="FAIL",device="GPU",gpu_memory_policy=memory,jit_compile=True,
                 dtype="float64",tf32=False,checks=[],replication="INCOMPLETE")
    try:
        folder=args.candidate/"covariance_floor_0.001"
        history={k:tf.io.parse_tensor(tf.io.read_file(str(folder/(k+".tensor"))),tf.float64)
                 for k in ("A","b","c","x","log_incremental_weight")}
        consumer=controller.LorenzRefinement(forcing=2.5,replicas=4,
            precision_policy="nearest_covariance",precision_floor=.001,
            precision_eigen_tolerance=8*2.220446049250313e-16,
            precision_eigen_refinement=True,precision_stable_integral=True)
        result=consumer.engine.fit(*(history[k][3] for k in ("A","b","c","x","log_incremental_weight")))
        for key in ("A","b","c"):
            error=tf.abs(result[key]-history[key][4])/(1e-7+1e-9*tf.abs(history[key][4]))
            summary["checks"].append(dict(name="shared_consumer_"+key,
                passed=bool(tf.reduce_all(error<=1).numpy()),max_scaled=float(tf.reduce_max(error).numpy())))
        summary["checks"].append(dict(name="valid_and_one_trace",
            passed=bool(result["valid"].numpy()) and consumer.engine.fit.experimental_get_tracing_count()==1))
        for key,value in result.items():
            tf.io.write_file(str(args.output/(key+".tensor")),tf.io.serialize_tensor(value))
        summary["status"]="PASS" if all(c["passed"] for c in summary["checks"]) else "FAIL"
    except Exception as e:summary["error"]=repr(e)
    summary["wall_seconds"]=time.monotonic()-start
    summary["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))
    if summary["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()

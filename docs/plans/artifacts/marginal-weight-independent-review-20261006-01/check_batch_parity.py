"""CPU FP64 XLA mechanics check; no scientific accuracy or GPU claim."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys, json, time, hashlib, subprocess
from pathlib import Path
source = Path("/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917")
sys.path.insert(0, str(source))
import tensorflow as tf
from bayesfilter.testing.marginal_weight_campaign import specification, controls, designs, dataset, compile_filter
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score
start = time.monotonic()
dtype, n, batch = tf.float64, 16, 2
spec = specification("m13-t2", dtype)
base, noise, reset = designs(spec, n, "calibration", 0, dtype)
observed = tf.cast(dataset(spec, "calibration"), dtype)
settings = controls(0)
theta = tf.constant(spec["points"], dtype)
covariance = tf.broadcast_to(tf.cast(spec["fixture"].P0, dtype), [n, spec["d"], spec["d"]])
@tf.function(input_signature=[tf.TensorSpec([batch,spec["p"]],dtype)], jit_compile=True, autograph=False)
def multi(theta_rows):
    return canonical_batch_fused_value_score(spec["model"], theta_rows,
        tf.broadcast_to(tf.eye(spec["p"],dtype=dtype),[batch,spec["p"],spec["p"]]),
        base,covariance,noise,observed,reset_design=reset,return_filter_moments=True,
        importance_weight_policy="marginal_mixture",**settings)
value, score, status = multi(theta)
single = compile_filter(spec,n,settings,"marginal_mixture",dtype)
rows = [single(theta[i],base,noise,reset,observed) for i in range(batch)]
ref_value = tf.stack([r["value"] for r in rows]);ref_score=tf.stack([r["score"] for r in rows])
revalue,rescore,restatus=multi(theta)
valid=bool(tf.reduce_all(status["program_valid"])) and all(bool(r["valid"]) for r in rows) and bool(tf.reduce_all(restatus["program_valid"]))
err_v=float(tf.reduce_max(tf.abs(value-ref_value)));err_s=float(tf.reduce_max(tf.abs(score-ref_score)))
repeat_v=float(tf.reduce_max(tf.abs(value-revalue)));repeat_s=float(tf.reduce_max(tf.abs(score-rescore)))
passed=valid and err_v<1e-9 and err_s<1e-8 and repeat_v==0 and repeat_s==0
tracked=["bayesfilter/highdim/ledh_marginal_weights_tf.py","bayesfilter/highdim/ledh_canonical_score_tf.py","bayesfilter/highdim/ledh_canonical_batch_fused_tf.py"]
result=dict(passed=passed,valid=valid,cpu_only=True,gpus_intentionally_hidden=True,dtype="float64",jit_compile=True,
    model="m13-t2",particles=n,batch_rows=batch,value=value.numpy().tolist(),score=score.numpy().tolist(),
    max_value_difference=err_v,max_score_difference=err_s,repeat_value_difference=repeat_v,repeat_score_difference=repeat_s,
    wall_seconds=time.monotonic()-start,tensorflow=tf.__version__,settings=settings,
    source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip(),
    hashes={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in tracked},
    limitation="Small CPU FP64 mechanics fixture; no GPU/FP32 repeatability or oracle accuracy claim")
Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ["passed","valid","max_value_difference","max_score_difference","repeat_value_difference","repeat_score_difference","wall_seconds"]}))
assert passed, result

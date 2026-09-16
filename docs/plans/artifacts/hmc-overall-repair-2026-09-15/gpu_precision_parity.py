"""Bounded GPU/XLA estimator parity, with an independent CPU formula."""
from pathlib import Path
import json, os, sys, time
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
started = time.monotonic()
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
from bayesfilter.inference.hmc_precision import mean_precision, _batch_program
with tf.device("/CPU:0"):
    x = tf.random.stateless_normal((512,4,2), seed=(915,44), dtype=tf.float64)
    def bm(batch):
        a=512//batch
        means=tf.reduce_mean(tf.reshape(x[:a*batch], (a,batch,4,2)),axis=1)
        return batch*tf.math.reduce_variance(means,axis=0)*a/(a-1.)
    expected_lrv=2*bm(22)-bm(7)
    expected=tf.sqrt(tf.reduce_sum(expected_lrv,axis=0)/(16*512))
with tf.device("/GPU:0"):
    device_values=tf.identity(x)
    result=mean_precision(device_values, method="lugsail", jit_compile=True)
    program=_batch_program((512,4,2),22,7,.5,True)
    hlo=str(program.experimental_get_compiler_ir(device_values)(stage="hlo"))
    import hashlib
    error=float(tf.reduce_max(tf.abs(expected-result["mcse"])))
    if error > 1e-12:
        raise AssertionError("GPU/XLA MCSE disagrees with independent formula")
    report={"passed":True,"max_abs_error":error,"tolerance":1e-12,
        "tolerance_provenance":"float64 formula comparison, same as unit formula tests",
        "device":result["mcse"].device,"hlo_sha256":hashlib.sha256(hlo.encode()).hexdigest(),
        "trace_count":program.experimental_get_tracing_count(),
        "signature":str(program.input_signature),"seed":[915,44],"jit_compile":True,
        "memory_policy":memory,"allocator":tf.config.experimental.get_memory_info("GPU:0"),
        "wall_seconds":time.monotonic()-started,"interpretation":"arithmetic parity only"}
Path(__file__).with_name("gpu-precision-parity.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))

"""Trusted GPU/XLA availability check; no scientific comparison."""
import json
import os
from pathlib import Path
import time
from bayesfilter.score_study.runtime import configure_runtime, memory_usage

started=time.monotonic()
cpu=time.process_time()
runtime=configure_runtime(device="GPU",tf32=True,jit_compile=True)
import tensorflow as tf
devices=[dict(name=d.name,**tf.config.experimental.get_device_details(d))
         for d in tf.config.list_physical_devices("GPU")]
assert len(devices)==1 and "5080" in devices[0]["device_name"]
@tf.function(input_signature=[tf.TensorSpec([2,2],tf.float32)],jit_compile=True)
def kernel(x):
    return x@x
with tf.device("/GPU:0"):
    output=kernel(tf.eye(2,dtype=tf.float32))
tf.debugging.assert_equal(output,tf.eye(2,dtype=tf.float32))
assert "GPU" in output.device and kernel.experimental_get_tracing_count()==1
result=dict(status="pass",runtime=runtime,physical_devices=devices,
    CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
    output_device=output.device,traces=kernel.experimental_get_tracing_count(),
    allocator=memory_usage("GPU"),wall_seconds=time.monotonic()-started,
    process_cpu_seconds=time.process_time()-cpu,scientific_evidence=False)
Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("status","physical_devices","traces","wall_seconds")}))

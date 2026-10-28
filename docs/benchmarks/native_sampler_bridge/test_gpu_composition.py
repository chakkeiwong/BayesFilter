"""Tiny trusted GPU/XLA composition diagnostic; not posterior evidence."""
import os
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("memory growth must be set before framework import")
import argparse
import json
from pathlib import Path
import sys
import time

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    import tensorflow as tf
    devices=tf.config.list_physical_devices("GPU")
    if not devices:raise RuntimeError("GPU required")
    memory=[]
    for device in devices:
        tf.config.experimental.set_memory_growth(device,True)
        growth=tf.config.experimental.get_memory_growth(device)
        if not growth:raise RuntimeError("memory growth was not established")
        memory.append({"name":device.name,"growth":growth})
    tf.config.set_soft_device_placement(False)
    from validate import Native
    from wrapper import FitConfig,compiled_fit
    native=Native();checks=[]
    for engine in ("stan","walnutpie"):
        config=FitConfig(engine,1,4,2,10,10,5)
        fit=compiled_fit(config,"/GPU:0")
        controls=[.3,.1,.8,.5]
        inverse=[1.,.5,0.,0.,0.,2.,.25,0.,0.,0.,.75,.5,0.,0.,0.,1.]
        starts=[.25,-.5,.75,-1.,-.25,.5,-.75,1.]
        @tf.function(input_signature=[tf.TensorSpec([2,4],tf.float64),tf.TensorSpec([2],tf.int64)],
                     jit_compile=True,autograph=False)
        def composed(raw,seeds):
            with tf.device("/GPU:0"):
                # A dynamic producer and a transposing/scaling consumer ensure
                # the callback obeys the supplied stream and explicit layouts.
                rows,meta=fit(tf.constant(controls,tf.float64),raw+.125,
                             tf.reshape(tf.constant(inverse,tf.float64),[4,4]),seeds)
                return tf.transpose(rows[:,10:,7:],[1,0,2])*2.,meta[:,:5]
        for seeds in ([101,202],[303,404]):
            with tf.device("/GPU:0"):
                inputs=[tf.reshape(tf.constant(starts,tf.float64),[2,4]),tf.constant(seeds,tf.int64)]
                actual,meta=composed(*inputs)
            if "GPU:0" not in actual.device:raise AssertionError(actual.device)
            rows,expected_meta=native.fit(config,controls,[x+.125 for x in starts],inverse,seeds)
            expected=[rows[(chain*20+iteration)*11+7+j]*2.
                      for iteration in range(10,20) for chain in range(2) for j in range(4)]
            if actual.numpy().reshape(-1).tolist()!=expected:raise AssertionError("GPU producer/consumer mismatch")
            expected_m=[expected_meta[chain*8+j] for chain in range(2) for j in range(5)]
            if meta.numpy().reshape(-1).tolist()!=expected_m:raise AssertionError("GPU metric mismatch")
        hlo=composed.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
        if "BfNativeSamplerV1" not in hlo:raise AssertionError("custom call absent")
        (args.output/(engine+".hlo")).write_text(hlo)
        if composed.experimental_get_tracing_count()!=1:raise AssertionError("unexpected retracing")
        checks.append({"engine":engine,"exact_composed_result":True,"seed_vectors":[[101,202],[303,404]],
                       "trace_count":1,"device":actual.device})
    result={"command":sys.argv,"checks":checks,"memory_policy":memory,
            "allocator":tf.config.experimental.get_memory_info("GPU:0"),
            "wall_seconds":time.monotonic()-start,"trust_basis":"trusted_escalated_tool_execution"}
    (args.output/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()

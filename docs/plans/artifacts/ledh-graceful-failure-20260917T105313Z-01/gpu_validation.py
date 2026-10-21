"""Escalated GPU/XLA mechanics smoke; no tuning or scientific admission."""
import json,os,runpy,time
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory_policy=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    canonical_batch_fused_value_score,canonical_batch_fused_value_score_whileloop)
from bayesfilter.inference.ledh_dual_parameter_target import DualParameterLEDHTarget
fixture=runpy.run_path("tests/integration/test_ledh_failure_contract.py")["fixture"]
report={"classification":"mechanics_smoke_only","memory_policy":memory_policy,
        "tf32_enabled":True,"jit_compile":True,"gpu_visible":os.environ["CUDA_VISIBLE_DEVICES"],
        "checks":[]}
for dtype in (tf.float32,tf.float64):
    model,inputs,controls=fixture(dtype=dtype,horizon=3)
    theta=tf.constant([[.6],[-.2],[.6]],dtype)
    directions=tf.constant([[[1.],[.5]],[[1.],[.5]],[[1.],[-1.]]],dtype)
    for entry in (canonical_batch_fused_value_score,canonical_batch_fused_value_score_whileloop):
        fn=tf.function(lambda q,v:entry(model,q,v,*inputs,**controls),
            input_signature=[tf.TensorSpec([3,1],dtype),tf.TensorSpec([3,2,1],dtype)],jit_compile=True)
        started=time.monotonic()
        value,force,status=fn(theta,directions)
        values=value.numpy().tolist()
        elapsed=time.monotonic()-started
        assert status["value_valid"].numpy().tolist()==[True,False,True]
        assert status["force_valid"].numpy().tolist()==[True,False,False]
        tf.debugging.assert_equal(force[1:],tf.zeros_like(force[1:]))
        row={"dtype":dtype.name,"endpoint":entry.__name__,"compile_first_seconds":elapsed,
             "value_valid":status["value_valid"].numpy().tolist(),
             "force_valid":status["force_valid"].numpy().tolist(),
             "first_failure_stage":status["first_failure_stage"].numpy().tolist(),
             "first_failure_time":status["first_failure_time"].numpy().tolist()}
        assert "GPU" in value.device
        started=time.monotonic()
        for _ in range(3):
            fn(theta,directions)[0].numpy()
        row["mean_warm_seconds"]=(time.monotonic()-started)/3
        report["checks"].append(row)
    target=DualParameterLEDHTarget(model,*inputs,exact_params=controls,biased_params=controls)
    fn=target.as_graph_value_score_callable(1)
    value,force,status=fn(tf.constant([-.2],dtype))
    assert float(value)==float("-inf") and not bool(status["value_valid"])
    tf.debugging.assert_equal(force,tf.zeros_like(force))
report["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
report["status"]="passed"
out=Path(os.environ["LEDH_PROGRAM_RESULT_DIR"])/"gpu-validation.json" if "LEDH_PROGRAM_RESULT_DIR" in os.environ else Path("docs/plans/artifacts/ledh-graceful-failure-20260917T105313Z-01/gpu-validation.json")
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"status":"passed","result":str(out),"checks":len(report["checks"])}))

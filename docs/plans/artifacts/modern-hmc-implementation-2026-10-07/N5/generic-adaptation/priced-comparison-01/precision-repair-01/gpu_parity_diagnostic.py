"""Diagnostic/reference arithmetic only; no HMC or performance comparison."""
from pathlib import Path
import hashlib, json, math, os, sys, time
from datetime import datetime, timezone
started=time.monotonic()
root=Path(__file__).resolve().parent
out=root/"gpu-check-01";out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/"source"))
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")!="true":raise ValueError("growth required before import")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
import numpy as np  # independent diagnostic input generation and comparison
from bayesfilter.inference.hmc_precision import mean_precision, _batch_maximum_program
from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
paths=source_dependencies([root/"source/bayesfilter/inference/hmc_precision.py"],repository_root=root/"source")
hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
report=dict(status="running",command=[sys.executable,*sys.argv],environment=sys.executable,
    started_utc=datetime.now(timezone.utc).isoformat(),memory_policy=memory,source_sha256=hashes,
    source=json.loads((root/"source/.validation-source.json").read_text()),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),jit_compile=True,
    tensorflow_version=tf.__version__,tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
    trust_basis="trusted_escalated_execution",role="GPU CPU arithmetic parity diagnostic",
    prior_inventory={"gpu":1,"used_mib":7003,"total_mib":32760,"utilization_percent":100},
    shared_device=True,relative_tolerance=1e-10,absolute_tolerance=1e-12,
    performance_comparison=False,default_promotion=False,cases=[])
def save():(out/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
save()
try:
    for n,batch in [(240,20),(247,21),(10000,100)]:
        for rho in (0.,.9,-.9):
            values=np.random.default_rng(73115).normal(size=(n,4,3))
            for i in range(1,n):values[i]=rho*values[i-1]+math.sqrt(1-rho*rho)*values[i]
            outputs=[]
            for device in ("/CPU:0","/GPU:0"):
                with tf.device(device):
                    result=mean_precision(tf.constant(values),method="lugsail_max_batch_means",
                        batch_size=batch,min_batches=2,jit_compile=True)
                    output={k:result[k].numpy() for k in ("per_chain_long_run_variance",
                        "per_chain_raw_lugsail_long_run_variance","per_chain_plain_long_run_variance",
                        "mcse","per_chain_batch_maximum_activated","valid")}
                    if device=="/GPU:0":
                        placement=result["per_chain_long_run_variance"].device
                        if "GPU:0" not in placement:raise ValueError("GPU output silently placed elsewhere")
                outputs.append(output)
            errors={}
            for key,a in outputs[0].items():
                b=outputs[1][key]
                if a.dtype.kind=="b":np.testing.assert_array_equal(a,b)
                else:
                    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12,equal_nan=True)
                    errors[key]=float(np.max(np.abs(a-b)))
            report["cases"].append(dict(n=n,batch_size=batch,rho=rho,placement=placement,max_absolute_errors=errors))
            save()
    plain=[[1.,1.,float("nan"),float("inf"),0.,1.,2.]]
    raw=[[float("nan"),float("inf"),1.,1.,2.,-1.,2.]]
    with tf.device("/GPU:0"):
        maximum,mask=_batch_maximum_program((1,7),True)(tf.constant(plain,tf.float64),tf.constant(raw,tf.float64))
    np.testing.assert_array_equal(np.isfinite(maximum.numpy()),[[False,False,False,False,False,True,True]])
    np.testing.assert_array_equal(mask.numpy(),[[False,False,False,False,False,True,False]])
    report["nonfinite_and_activation_parity"]="passed"
    report["allocator_bytes"]=tf.config.experimental.get_memory_info("GPU:0")
    for name,digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError("changed source")
    report["status"]="passed"
except BaseException as exc:
    report.update(status="failed",error=str(exc));raise
finally:
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat());save()

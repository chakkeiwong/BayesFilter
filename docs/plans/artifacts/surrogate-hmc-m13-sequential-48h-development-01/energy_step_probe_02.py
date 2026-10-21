"""Discarded GPU/XLA diagnostic of the bounded M13 energy-step repair."""
import json, sys, time
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.inference.surrogate_hmc_program import ROOT,write_json,file_hash
from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.inference import hmc_tensorflow_tuning as native
from bayesfilter.inference.hmc_tensorflow_verified import _health,_save_chunk
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import selected_components,read_tensor
out=Path(sys.argv[1]); root=ROOT/"docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-02/m13-public-tuning/tuning"
p=json.loads((root/"fresh-progress.json").read_text())
saved=json.loads((root/"nomination/manifest.json").read_text())
center=read_tensor(root/"nomination",saved["tensors"]["center"])
factor=read_tensor(root/"nomination",saved["tensors"]["factor"])
epsilon=float(read_tensor(root/"nomination",saved["tensors"]["step_size"]))
archive=p["chunks"][0]["archive"]
initial=read_tensor(root,archive["initial"])
states=read_tensor(root,archive["tensors"]["states"])
starts=[initial,states[54]]
adapter,binding,target,fixture,selection,path=selected_components(step_multipliers=(.125,1.))
old=json.loads((root.parent/"report.json").read_text())
if json.loads(json.dumps(binding.payload()))!=old["runner_binding"]: raise ValueError("proposal binding drift")
affine=native._TensorFlowAffineAdapter(native._TensorFlowAffineTransform(center,factor),adapter.adapter_signature(),adapter.target_scope)
kernel=binding.build_tensor_kernel(affine,step_size=tf.constant(epsilon/2,tf.float64),num_leapfrog_steps=1,
    target_scope=adapter.target_scope,target_status_trace_policy="none",chain_execution_mode="tf_function",use_xla=True)
@tf.function(input_signature=[tf.TensorSpec([4,2],tf.float64),tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
def run(state,seed):
    return native._run_tensor_steps(kernel=kernel,initial_state=state,num_results=64,seed=seed,adaptive=False)
source_paths=[Path(__file__),ROOT/"bayesfilter/inference/hmc_tensorflow_tuning.py",ROOT/"bayesfilter/inference/hmc_tensorflow_verified.py",ROOT/"bayesfilter/testing/surrogate_hmc_m13_step_mixture.py",ROOT/"bayesfilter/testing/surrogate_hmc_m13_public_tuning.py"]
source_hashes={str(x):file_hash(x) for x in source_paths}
report=dict(status="RUNNING",qualified=False,posterior_qualified=False,ranking_supported=False,
    role="discarded energy-repair screen; public tuning still required",memory_policy=memory,
    device="gpu",jit_compile=True,tf32=True,tensorflow=tf.__version__,source_root=str(root),
    binding=binding.payload(),original_step=epsilon,step_cap=epsilon/2,multipliers=[.125,1.],
    data_seed=1704,observation_checksum=fixture.observation_checksum,counts_per_chain=64,
    source_hashes=source_hashes,plan="docs/plans/surrogate-hmc-m13-48h-continuation-2026-09-29.md",
    reference="saved failed first250 warmup transitions; same initial seed for first diagnostic prefix",
    replications=[])
write_json(out/"report.json",finite_report(report))
for i,start in enumerate(starts):
    seed=p["chunks"][0]["seed"] if i==0 else [20260929,4101]
    begin=time.monotonic(); chunk=run(start,tf.constant(seed,tf.int32)); chunk.states.numpy()
    checks={k:bool(v) for k,v in _health(chunk,start,1000.).items()}
    report["replications"].append(dict(seed=seed,wall_seconds=time.monotonic()-begin,
        checks=checks,health_passed=all(checks.values()),archive=_save_chunk(out,f"rep-{i}",chunk),
        initial=native._write_tensor(out/f"rep-{i}-initial.tfbin",start),
        acceptance_by_chain=tf.reduce_mean(tf.cast(chunk.is_accepted,tf.float64),axis=0).numpy().tolist(),
        max_abs_energy_by_chain=tf.reduce_max(tf.abs(chunk.delta_h),axis=0).numpy().tolist()))
    write_json(out/"report.json",finite_report(report))
    print("replication",i,"healthy",all(checks.values()),flush=True)
report["source_unchanged"]=all(file_hash(Path(x))==h for x,h in source_hashes.items())
report["status"]="DIAGNOSTIC_COMPLETE"
report["repair_screen_passed"]=report["source_unchanged"] and all(x["health_passed"] for x in report["replications"])
write_json(out/"report.json",finite_report(report))
print("repair_screen_passed",report["repair_screen_passed"],flush=True)

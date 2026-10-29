"""Bounded M13 mixing diagnostic; all short chains are discarded and non-promoting."""
import argparse
import json
from pathlib import Path
import time
import tensorflow as tf

F64=tf.float64
ARMS=((1,1),(1,2),(1,4),(1,8),(2,2),(2,4),(4,2))
COUNT=32

def run(output,memory):
    from bayesfilter.inference.surrogate_hmc_program import ROOT,write_json,file_hash
    from bayesfilter.inference import hmc_tensorflow_tuning as native
    from bayesfilter.inference.hmc_tensorflow_verified import _save_chunk,_health
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import selected_components
    from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import native_force,batched_endpoint
    from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import endpoint_components
    from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    root=ROOT/"docs/plans/artifacts/surrogate-hmc-m13-sequential-recovery-03/m13-public-tuning/tuning"
    progress=json.loads((root/"fresh-progress.json").read_text())
    saved=json.loads((root/"nomination/manifest.json").read_text())
    def read(where,record):
        p=where/record["path"]
        if file_hash(p)!=record["sha256"]:raise ValueError("archive checksum mismatch")
        return tf.io.parse_tensor(tf.io.read_file(str(p)),tf.as_dtype(record.get("dtype","float64")))
    center=read(root/"nomination",saved["tensors"]["center"])
    factor=read(root/"nomination",saved["tensors"]["factor"])
    epsilon=float(read(root/"nomination",saved["tensors"]["step_size"]))
    starts=[read(root,progress["chunks"][i]["archive"]["tensors"]["final_state"]) for i in (18,24)]
    adapter,binding,target,fixture,selection,path=selected_components()
    affine=native._TensorFlowAffineAdapter(native._TensorFlowAffineTransform(center,factor),adapter.adapter_signature(),adapter.target_scope)
    result=dict(memory_policy=memory,device="gpu",jit_compile=True,tf32=True,tensorflow=tf.__version__,status="RUNNING",qualified=False,posterior_qualified=False,ranking_supported=False,
        role="discarded exploration diagnostic; public tuner still required",
        source_root=str(root),selection_sha256=file_hash(path),binding=binding.payload(),
        epsilon=epsilon,center=center.numpy().tolist(),factor=factor.numpy().tolist(),
        phase_plan="docs/plans/surrogate-hmc-m13-48h-continuation-2026-09-29.md",
        common_random_numbers_across_arms=True,count_per_replication=COUNT,arms=[])
    raw=affine.latent_to_position(starts[0])
    force=native_force(target)(raw)
    _, evaluator=endpoint_components(fixture)
    endpoint=batched_endpoint(evaluator)
    gradients=[]
    for h in (.001,.0005):
        components=[]
        for d in range(2):
            delta=tf.one_hot(d,2,dtype=F64)*h
            plus,minus=endpoint(raw+delta),endpoint(raw-delta)
            if not bool(tf.reduce_all(plus.status["value_valid"] & minus.status["value_valid"])):
                raise ValueError("finite-difference endpoint invalid")
            components.append((plus.potential-minus.potential)/(2*h))
        gradients.append(tf.stack(components,axis=-1))
    result["force_diagnostic"]=dict(raw_positions=raw.numpy().tolist(),
        force=force.force.numpy().tolist(),force_valid=force.force_valid.numpy().tolist(),
        endpoint_central_difference=[g.numpy().tolist() for g in gradients],
        adapted_force=tf.matmul(force.force,factor).numpy().tolist(),
        adapted_endpoint_gradient=tf.matmul(gradients[-1],factor).numpy().tolist())
    write_json(output/"report.json",finite_report(result))
    print("Force diagnostic saved",flush=True)
    for arm_index,(leapfrogs,multiplier) in enumerate(ARMS):
        step=epsilon*multiplier
        kernel=binding.build_tensor_kernel(affine,step_size=tf.constant(step,F64),
            num_leapfrog_steps=leapfrogs,target_scope=adapter.target_scope,
            target_status_trace_policy="none",chain_execution_mode="tf_function",use_xla=True)
        @tf.function(input_signature=[tf.TensorSpec([4,2],F64),tf.TensorSpec([2],tf.int32)],
            jit_compile=True,autograph=False)
        def chunk(state,seed):
            return native._run_tensor_steps(kernel=kernel,initial_state=state,
                num_results=COUNT,seed=seed,adaptive=False)
        arm=dict(leapfrog_steps=leapfrogs,step_size=step,multiplier=multiplier,replications=[])
        result["arms"].append(arm)
        for replication,start in enumerate(starts):
            seed=[20260929,2100+replication]
            begin=time.monotonic()
            sample=chunk(start,tf.constant(seed,tf.int32))
            sample.states.numpy()
            wall=time.monotonic()-begin
            previous=tf.concat([start[None,...],sample.states[:-1]],axis=0)
            delta=sample.states-previous
            health={k:bool(v) for k,v in _health(sample,start,1000.).items()}
            rep=dict(seed=seed,wall_seconds=wall,checks=health,health_passed=all(health.values()),
                archive=_save_chunk(output,f"arm-{arm_index}-rep-{replication}",sample),
                acceptance_by_chain=tf.reduce_mean(tf.cast(sample.is_accepted,F64),axis=0).numpy().tolist(),
                expected_acceptance_by_chain=tf.reduce_mean(tf.exp(tf.minimum(sample.log_accept_ratio,0.)),axis=0).numpy().tolist(),
                squared_jump_by_chain_coordinate=tf.reduce_mean(delta*delta,axis=0).numpy().tolist(),
                squared_total_displacement_by_chain_coordinate=tf.square(sample.final_state-start).numpy().tolist())
            arm["replications"].append(rep)
            write_json(output/"report.json",finite_report(result))
            print("Mixing probe",arm_index,leapfrogs,multiplier,replication,"health",rep["health_passed"],"wall",wall,flush=True)
    result["status"]="DIAGNOSTIC_COMPLETE"
    write_json(output/"report.json",finite_report(result))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    run(args.output,memory)

if __name__=="__main__":main()

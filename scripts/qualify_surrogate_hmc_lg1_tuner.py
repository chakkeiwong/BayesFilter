#!/usr/bin/env python3
"""Full LG1 DCTR public-tuner GPU/XLA wiring and cost diagnostic only."""
import argparse
import dataclasses
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP","0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_lg1_tf import make_diagnostic_dataset,make_lg1_consumer
    from bayesfilter.inference import (tune_hmc_kernel,TensorFlowHMCKernelTuningConfig,
        FourChainMeanBandAcceptancePolicy,bind_neural_force_hmc_tuning_runner,
        build_retained_bound_hmc_archive_runner_from_tuning_result)
    from bayesfilter.inference.surrogate_hmc_program import write_json
    from bayesfilter.inference.hmc_tensorflow_verified import _source_hashes
    dataset=make_diagnostic_dataset()
    consumer=make_lg1_consumer(dataset,batch_size=4,particle_count=1024)
    binding=bind_neural_force_hmc_tuning_runner(force=consumer.force,target=consumer.endpoint,
                                             target_scope="M01-LG1-T10-N1024-tuner-qualification")
    class Adapter:
        target_scope=binding.target_scope
        def adapter_signature(self):
            return consumer.endpoint.identity+":"+consumer.force.identity
    cfg=TensorFlowHMCKernelTuningConfig(parameter_dimension=1,evidence_role="diagnostic_only",
        mass_window_results=(1,),step_adaptation_results=1,verification_results=1,
        max_leapfrog_steps=1,initial_step_size=.01,
        budget_provenance="one-step XLA interface and timing diagnostic",
        initial_step_size_provenance="small convenience mechanics hypothesis",
        geometry_provenance="unit diagnostic scales",target_scope=Adapter.target_scope,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
        target_accept_prob=.7,verification_repair_rounds=0,step_repair_factor=2.,
        mass_shrinkage=.1,covariance_jitter=1e-9,eigenvalue_floor=1e-9,max_condition_number=1e8,
        seed=(20260919,600),use_xla=True)
    report=dict(memory_policy=memory,tensorflow=tf.__version__,tf32=True,source_hashes=_source_hashes(),
                consumer=consumer.manifest,data=dataset,classification="diagnostic_only",jit_compile=True)
    start=time.monotonic()
    result=tune_hmc_kernel(adapter=Adapter(),initial_position=tf.zeros([1],tf.float64),
        parameter_scales=tf.ones([1],tf.float64),config=cfg,runner_binding=binding,
        output_dir=args.output/"tuning")
    report["compile_and_first_seconds"]=time.monotonic()-start
    start=time.monotonic()
    second=result.graph_function(tf.zeros([1],tf.float64),tf.ones([1],tf.float64))
    # Synchronize materialization before recording timing.
    report["finite_second_state"]=bool(tf.reduce_all(tf.math.is_finite(second["final_raw_state"])))
    report["three_transition_steady_seconds"]=time.monotonic()-start
    report["health_passed"]=bool(result.health_passed)
    report["admission_supported"]=result.admission_supported
    try:
        build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=result,runner_binding=binding)
    except ValueError as exc:
        report["diagnostic_rejected"]= "not admission-capable" in str(exc)
    else:
        report["diagnostic_rejected"]=False
    report["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    write_json(args.output/"report.json",report)
    assert report["finite_second_state"] and report["diagnostic_rejected"] and report["health_passed"]
    print("PASS: full LG1 consumer public-tuner XLA wiring; no inference admission")


if __name__=="__main__":
    main()

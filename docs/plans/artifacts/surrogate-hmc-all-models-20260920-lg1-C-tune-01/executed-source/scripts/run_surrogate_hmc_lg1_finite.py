#!/usr/bin/env python3
"""Separate public C tuning and two-look C/D,C/R retained comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP","0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("tune","sample"),required=True)
    for name in ("output","reference-root","selection","endpoint-root","finite-reference-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--tuning-root",type=Path)
    args=parser.parse_args()
    if args.stage=="sample" and args.tuning_root is None:
        parser.error("sampling requires C --tuning-root")
    args.output.mkdir(parents=True,exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_lg1_finite_consumer import prepare_finite_consumer
    from bayesfilter.inference import (TensorFlowHMCKernelTuningConfig,VerifiedTensorFlowHMCConfig,
        FourChainMeanBandAcceptancePolicy,tune_hmc_kernel,load_verified_tensorflow_hmc_result,
        build_retained_bound_hmc_archive_runner_from_tuning_result,BoundRetainedHMCArchiveConfig)
    from bayesfilter.inference.surrogate_hmc_program import write_json
    from bayesfilter.inference.surrogate_hmc_equivalence import assess_equivalence,finite_report
    consumer,adapter,binding,finite_reference,provenance=prepare_finite_consumer(
        args.reference_root,args.selection,args.endpoint_root,args.finite_reference_root)
    report=dict(memory_policy=memory,tensorflow=tf.__version__,tf32=True,jit_compile=True,
        provenance=provenance,arms={},inference_attempt=4,heuristic_dominance="NOT_RUN",default_readiness=False,
        comparisons={"B/R":"NOT_RUN","C/D":"NOT_RUN","D/R":finite_reference["D_vs_R"]["status"],"C/R":"NOT_RUN"},
        plan="docs/plans/surrogate-hmc-lg1-finite-comparisons-2026-09-20.md")
    if args.stage=="tune":
        nomination=TensorFlowHMCKernelTuningConfig(parameter_dimension=1,evidence_role="diagnostic_candidate_screen",
            mass_window_results=(32,64),step_adaptation_results=128,verification_results=64,max_leapfrog_steps=2,
            initial_step_size=.2,budget_provenance="LG1 attempt4 separately scoped C tuning",
            initial_step_size_provenance=".2 chart-scale warm start independently adapted on C",
            geometry_provenance="unit chart warm start, windowed covariance adapted on C",target_scope=adapter.target_scope,
            acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
            target_accept_prob=.7,verification_repair_rounds=0,step_repair_factor=2.,mass_shrinkage=.1,
            covariance_jitter=1e-9,eigenvalue_floor=1e-9,max_condition_number=1e8,seed=(20260920,5000),use_xla=True)
        config=VerifiedTensorFlowHMCConfig(nomination,((-2.5,),(-.8,),(.8,),(2.5,)),
            "four prior-region starts frozen before C claim chains",250,1000,(20260920,5001),1000.)
        start=time.monotonic()
        result=tune_hmc_kernel(adapter=adapter,initial_position=(0.,),config=config,parameter_scales=(1.,),
                               runner_binding=binding,output_dir=args.output/"C")
        report["arms"]["C"]=dict(passed=result.passed,checks=result.checks,manifest=result.artifact_manifest_path,
            selected_step_size=float(result.nomination.step_size),selected_leapfrog_steps=int(result.nomination.num_leapfrog_steps),
            metric_covariance=result.nomination.covariance.numpy().tolist(),wall_seconds=time.monotonic()-start)
    else:
        from bayesfilter.testing.surrogate_hmc_lg1_tf import kalman_physical
        reference_path=args.reference_root/"claim/reference.json"
        reference=json.loads(reference_path.read_text())
        rec=reference["tensors"]["summary"];path=reference_path.parent/rec["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec["sha256"]:
            raise ValueError("reference summary checksum mismatch")
        summary=tf.io.parse_tensor(tf.io.read_file(str(path)),tf.float64)
        margins=tf.constant(reference["margins"],tf.float64)
        if reference["margins"]!=finite_reference["margins"] or reference["function_names"]!=finite_reference["function_names"]:
            raise ValueError("comparison functions or margins differ")
        observations=tf.cast(consumer.fixture.observations,tf.float64)
        @tf.function(input_signature=[tf.TensorSpec([250,4,1],tf.float64)],jit_compile=True,autograph=False)
        def functions_for_chunk(samples):
            phi=-.95+1.9*tf.sigmoid(samples[:,:,0])
            standardized=(phi-summary[0])/summary[1]
            predictive=kalman_physical(tf.reshape(phi,[-1]),observations)[2]
            functions=tf.stack([standardized,tf.square(standardized),tf.cast(phi<=summary[2],tf.float64),
                tf.cast(phi<=summary[3],tf.float64),tf.cast(phi<=summary[4],tf.float64),tf.reshape(predictive,[250,4])],axis=-1)
            return phi[:,:,None],functions
        result=load_verified_tensorflow_hmc_result(args.tuning_root/"C/manifest.json",adapter=adapter,runner_binding=binding)
        runner=build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=result,runner_binding=binding)
        previous,parameters,functions,chunks,looks=None,[],[],[],[]
        start=time.monotonic()
        for index in range(8):
            chunk_root=args.output/"C"/f"chunk-{index+1:02d}"
            begun=time.monotonic()
            archive=runner.run(BoundRetainedHMCArchiveConfig(250,(20260920,6000+index),chunk_root,
                "LG1 C fixed-kernel retained; warmup excluded",previous))
            previous=archive["archive_manifest_path"]
            chunks.append(dict(manifest=previous,health_passed=archive["health_passed"],wall_seconds=time.monotonic()-begun))
            if not archive["health_passed"]:
                report["arms"]["C"]=dict(status="HEALTH_VETO",chunks=chunks,looks=looks)
                report["comparisons"].update({"C/D":"HEALTH_VETO","C/R":"HEALTH_VETO"})
                break
            manifest=json.loads(Path(previous).read_text())
            samples=tf.io.parse_tensor(tf.io.read_file(str(chunk_root/manifest["raw_samples"]["path"])),tf.float64)
            phi,values=functions_for_chunk(samples);parameters.append(phi);functions.append(values)
            if index+1 in (4,8):
                look={}
                for name,ref in (("C/D",finite_reference),("C/R",reference)):
                    look[name]=assess_equivalence(tf.transpose(tf.concat(parameters,0),[1,0,2]),
                        tf.transpose(tf.concat(functions,0),[1,0,2]),tf.constant(ref["reference_expectations"],tf.float64),
                        tf.constant(ref["empirical_refinement_allowance"],tf.float64),margins,comparison_families=2)
                    report["comparisons"][name]=look[name]["status"]
                looks.append(look)
                report["arms"]["C"]=dict(status=look["C/D"]["status"],chunks=chunks,looks=looks,
                    wall_seconds=time.monotonic()-start,retained_draws=(index+1)*250)
                write_json(args.output/"report.json",finite_report(report))
                print("C LOOK",index+1,{k:v["status"] for k,v in look.items()},flush=True)
                if all(row["status"] in ("EQUIVALENT","DISCREPANCY") for row in look.values()):
                    break
            print("C CHUNK",index+1,chunks[-1]["wall_seconds"],flush=True)
    report["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    report["source_hashes"]={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__),Path("bayesfilter/testing/surrogate_hmc_lg1_finite_consumer.py"),
        Path("bayesfilter/inference/surrogate_hmc_equivalence.py"))}
    write_json(args.output/"report.json",finite_report(report))
    print(json.dumps({"comparisons":report["comparisons"],"C":report["arms"].get("C",{}).get("passed")}),flush=True)


if __name__=="__main__":
    main()

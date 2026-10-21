#!/usr/bin/env python3
"""Retained LG1 A/B comparisons using separately verified fixed HMC kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tuning-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.inference import (load_verified_tensorflow_hmc_result,
        build_retained_bound_hmc_archive_runner_from_tuning_result, BoundRetainedHMCArchiveConfig)
    from bayesfilter.inference.surrogate_hmc_equivalence import assess_equivalence, finite_report
    from bayesfilter.testing.surrogate_hmc_lg1_scoped import prepare_consumer, bind_arm
    from bayesfilter.testing.surrogate_hmc_lg1_tf import kalman_physical
    from bayesfilter.inference.surrogate_hmc_program import write_json

    consumer, provenance = prepare_consumer(args.reference_root, args.selection)
    ref_root = args.reference_root / "claim"
    reference = json.loads((ref_root / "reference.json").read_text())
    tensors = {}
    for key, record in reference["tensors"].items():
        path = ref_root / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("reference tensor checksum mismatch")
        tensors[key] = tf.io.parse_tensor(tf.io.read_file(str(path)), tf.float64)
    summary = tensors["summary"]
    expectation = tf.constant(reference["reference_expectations"], tf.float64)
    allowance = tf.constant(reference["empirical_refinement_allowance"], tf.float64)
    margins = tf.constant(reference["margins"], tf.float64)
    observations = tf.cast(consumer.fixture.observations, tf.float64)
    @tf.function(input_signature=[tf.TensorSpec([250,4,1],tf.float64)],jit_compile=True,autograph=False)
    def functions_for_chunk(samples):
        phi = -.95 + 1.9*tf.sigmoid(samples[:,:,0])
        standardized = (phi-summary[0])/summary[1]
        _, _, predictive = kalman_physical(tf.reshape(phi,[-1]), observations)
        functions = tf.stack([standardized,tf.square(standardized),
            tf.cast(phi <= summary[2],tf.float64),tf.cast(phi <= summary[3],tf.float64),
            tf.cast(phi <= summary[4],tf.float64),tf.reshape(predictive,[250,4])],axis=-1)
        return phi[:,:,None], functions

    report = dict(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        provenance=provenance, arms={}, comparisons={"B/R":"NOT_RUN","C/D":"NOT_RUN","D/R":"NOT_RUN","C/R":"NOT_RUN"},
        heuristic_dominance="NOT_RUN", default_readiness=False,
        plan="docs/plans/surrogate-hmc-lg1-inference-2026-09-19.md", inference_attempt=4)
    write_json(args.output / "report.json", report)
    tuning_report = json.loads((args.tuning_root / "report.json").read_text())
    for arm, seed in (("A",3000),("B",4000)):
        if not tuning_report["arms"][arm]["passed"]:
            report["arms"][arm] = dict(status="TUNING_REJECTED", retained_draws=0)
            continue
        adapter, binding = bind_arm(consumer,arm)
        result = load_verified_tensorflow_hmc_result(args.tuning_root/arm/"manifest.json",
                                                    adapter=adapter,runner_binding=binding)
        runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=result,runner_binding=binding)
        previous, parameters, functions, chunks, looks = None, [], [], [], []
        start = time.monotonic()
        for index in range(8):
            chunk_root = args.output / arm / f"chunk-{index+1:02d}"
            chunk_start = time.monotonic()
            archive = runner.run(BoundRetainedHMCArchiveConfig(250,(20260919,seed+index),chunk_root,
                "LG1 fixed-kernel retained inference; warmup excluded",previous))
            previous = archive["archive_manifest_path"]
            chunks.append(dict(manifest=previous, health_passed=archive["health_passed"],
                               wall_seconds=time.monotonic()-chunk_start))
            if not archive["health_passed"]:
                report["arms"][arm] = dict(status="HEALTH_VETO", chunks=chunks, looks=looks)
                if arm == "B":
                    report["comparisons"]["B/R"] = "HEALTH_VETO"
                break
            manifest = json.loads(Path(previous).read_text())
            samples = tf.io.parse_tensor(tf.io.read_file(str(chunk_root/manifest["raw_samples"]["path"])),tf.float64)
            phi, values = functions_for_chunk(samples)
            parameters.append(phi)
            functions.append(values)
            if index+1 in (4,8):
                look = assess_equivalence(tf.transpose(tf.concat(parameters,axis=0),[1,0,2]),
                    tf.transpose(tf.concat(functions,axis=0),[1,0,2]),expectation,allowance,margins)
                looks.append(look)
                report["arms"][arm] = dict(status=look["status"],chunks=chunks,looks=looks,
                    wall_seconds=time.monotonic()-start,retained_draws=(index+1)*250)
                if arm == "B":
                    report["comparisons"]["B/R"] = look["status"]
                write_json(args.output / "report.json", finite_report(report))
                print(arm,"LOOK",index+1,look["status"],"diagnostics",look["diagnostic_valid"],flush=True)
                if look["status"] in ("EQUIVALENT","DISCREPANCY"):
                    break
            print(arm,"CHUNK",index+1,chunks[-1]["wall_seconds"],flush=True)
        write_json(args.output / "report.json", finite_report(report))
    report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    write_json(args.output / "report.json", finite_report(report))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Separate public verified HMC tuning for LG1 exact and imperfect forces."""
import argparse
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.inference import (TensorFlowHMCKernelTuningConfig, VerifiedTensorFlowHMCConfig,
        FourChainMeanBandAcceptancePolicy, tune_hmc_kernel)
    from bayesfilter.testing.surrogate_hmc_lg1_scoped import prepare_consumer, bind_arm
    from bayesfilter.inference.surrogate_hmc_program import write_json

    consumer, provenance = prepare_consumer(args.reference_root, args.selection)
    report = dict(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
        provenance=provenance, arms={}, posterior_comparisons="NOT_RUN",
        plan="docs/plans/surrogate-hmc-lg1-inference-2026-09-19.md")
    for arm, seed in (("A", 1000), ("B", 2000)):
        adapter, binding = bind_arm(consumer, arm)
        nomination = TensorFlowHMCKernelTuningConfig(
            parameter_dimension=1, evidence_role="diagnostic_candidate_screen",
            mass_window_results=(32,64), step_adaptation_results=128, verification_results=64,
            max_leapfrog_steps=2, initial_step_size=.2,
            budget_provenance="LG1 attempt4 bounded tuning and two-look inference plan",
            initial_step_size_provenance=".2 latent-scale warm start, adapted independently per arm",
            geometry_provenance="unit chart warm start, separate windowed covariance adaptation per arm",
            target_scope=adapter.target_scope,
            acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
            target_accept_prob=.7, verification_repair_rounds=0, step_repair_factor=2.,
            mass_shrinkage=.1, covariance_jitter=1e-9, eigenvalue_floor=1e-9,
            max_condition_number=1e8, seed=(20260919,seed), use_xla=True)
        config = VerifiedTensorFlowHMCConfig(nomination, ((-2.5,),(-.8,),(.8,),(2.5,)),
            "four prior-region chart starts frozen before claim chains", 2000, 1000,(20260919,seed+1),1000.)
        start = time.monotonic()
        print("TUNING", arm, flush=True)
        try:
            result = tune_hmc_kernel(adapter=adapter, initial_position=(0.,), config=config, parameter_scales=(1.,),
                                    runner_binding=binding, output_dir=args.output / arm)
            row = dict(passed=result.passed, checks=result.checks, manifest=result.artifact_manifest_path,
                       selected_step_size=float(result.nomination.step_size),
                       selected_leapfrog_steps=int(result.nomination.num_leapfrog_steps),
                       metric_covariance=result.nomination.covariance.numpy().tolist())
        except ValueError as exc:
            # Preserve an explicitly rejected nominee without treating it as a
            # broken reference or preventing the independent other-arm check.
            if "nomination failed" not in str(exc):
                raise
            row = dict(passed=False, failure="nomination_failed", detail=str(exc))
        row["wall_seconds"] = time.monotonic() - start
        report["arms"][arm] = row
        write_json(args.output / "report.json", report)
        print(arm, row, flush=True)
    report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    report["both_handoffs_eligible"] = all(r["passed"] for r in report["arms"].values())
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
